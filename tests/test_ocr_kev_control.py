import io
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image

from src.tools.computer_control import ComputerController
from src.tools.kev_decision import KevActionDecision, KevDecisionClient
from src.tools.kev_computer_agent import KevComputerAgent
from src.tools.ocr import OCRRegion, ScreenOCR


def png_bytes(width=320, height=200):
    image = Image.new("RGB", (width, height), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_screen_ocr_converts_results_to_bounded_regions():
    class Engine:
        def __call__(self, image):
            return ([
                ([[10, 20], [100, 20], [100, 50], [10, 50]], "  Start   Recording ", 0.91),
                ([[2, 2], [4, 2], [4, 4], [2, 4]], "noise", 0.01),
            ], 0.01)

    regions = ScreenOCR(engine=Engine()).read(png_bytes())
    assert len(regions) == 1
    assert regions[0].ref == "O1"
    assert regions[0].text == "Start Recording"
    assert regions[0].center == (55, 35)
    assert "O1" in ScreenOCR.format(regions)


def test_screen_ocr_reads_modern_rapidocr_result():
    class EngineResult:
        boxes = np.asarray([[[10, 20], [100, 20], [100, 50], [10, 50]]], dtype=np.float32)
        txts = ("Settings",)
        scores = (0.95,)

    class Engine:
        def __call__(self, image):
            return EngineResult()

    regions = ScreenOCR(engine=Engine()).read(png_bytes())
    assert len(regions) == 1
    assert regions[0].text == "Settings"
    assert regions[0].center == (55, 35)


def test_screen_ocr_treats_modern_empty_result_as_no_text():
    class EmptyResult:
        boxes = None
        txts = None
        scores = None

    class Engine:
        def __call__(self, image):
            return EmptyResult()

    assert ScreenOCR(engine=Engine()).read(png_bytes()) == []


def test_screen_ocr_caps_candidates_before_recognition_and_reports_partial_read():
    class Engine:
        def __init__(self):
            self.text_rec = SimpleNamespace(rec_batch_num=6)
            self.recognized_count = 0

        def detect_and_crop(self, image, op_record):
            boxes = np.asarray([
                [[0, 0], [10, 0], [10, 10], [0, 10]],
                [[20, 0], [30, 0], [30, 10], [20, 10]],
                [[40, 0], [50, 0], [50, 10], [40, 10]],
            ], dtype=np.float32)
            return [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(3)], SimpleNamespace(
                boxes=boxes, scores=[0.1, 0.9, 0.8]
            )

        def recognize_txt(self, crops):
            self.recognized_count = len(crops)
            return SimpleNamespace(txts=[f"region-{i}" for i in range(len(crops))])

        def __call__(self, image):
            crops, detected = self.detect_and_crop(image, {})
            recognized = self.recognize_txt(crops)
            return SimpleNamespace(
                boxes=detected.boxes,
                txts=recognized.txts,
                scores=[0.9] * len(recognized.txts),
            )

    engine = Engine()
    ocr = ScreenOCR(max_regions=1, max_candidates=2, engine=engine)
    regions = ocr.read(png_bytes())

    assert engine.recognized_count == 2
    assert len(regions) == 1
    assert ocr.last_read_diagnostics["detected_candidates"] == 3
    assert ocr.last_read_diagnostics["recognized_candidates"] == 2
    assert ocr.last_read_diagnostics["candidate_limit_hit"] is True


def test_kev_client_selects_only_a_listed_ocr_region():
    seen = {}

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    def opener(request, timeout):
        seen["url"] = request.full_url
        seen["payload"] = json.loads(request.data)
        seen["timeout"] = timeout
        response = {"answers": {"target": {"choice": "O2", "confidence": 0.84}}}
        return Response(json.dumps(response).encode())

    regions = [
        OCRRegion("O1", "Cancel", 0.96, 0, 0, 80, 30),
        OCRRegion("O2", "Start Recording", 0.90, 100, 0, 240, 30),
    ]
    selected, message = KevDecisionClient(opener=opener).choose_ocr_target(
        "Start a recording", "Start Recording", "OCR screen state", regions
    )
    assert selected == regions[1]
    assert "Jev selected OCR target O2" in message
    assert seen["url"] == "https://openrouter.ai/api/alpha/decisions"
    assert seen["payload"]["model"] == "typesafe/jev-1.13"
    assert seen["payload"]["questions"]["target"]["criteria"]["O2"].find("Start Recording") >= 0


def test_kev_abstains_for_none_unknown_and_low_confidence():
    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    regions = [OCRRegion("O1", "Delete", 0.9, 0, 0, 50, 30)]

    def client_for(answer):
        return KevDecisionClient(opener=lambda *args, **kwargs: Response(
            json.dumps({"answers": {"target": answer}}).encode()
        ))

    assert client_for({"choice": "NONE", "confidence": 1}).choose_ocr_target("", "", "", regions)[0] is None
    assert client_for({"choice": "O99", "confidence": 1}).choose_ocr_target("", "", "", regions)[0] is None
    assert client_for({"choice": "O1", "confidence": 0.01}).choose_ocr_target("", "", "", regions)[0] is None


def test_kev_decides_operation_then_ocr_target():
    seen = []

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    def opener(request, timeout):
        payload = json.loads(request.data)
        seen.append(payload)
        answer = {
            "operation": {"choice": "CLICK", "confidence": 0.72},
            "target": {"choice": "O2", "confidence": 0.72},
        }
        return Response(json.dumps({"answers": answer}).encode())

    regions = [
        OCRRegion("O1", "Cancel", 0.96, 0, 0, 80, 30),
        OCRRegion("O2", "Save changes", 0.90, 100, 0, 240, 30),
    ]
    decision = KevDecisionClient(opener=opener).decide_next(
        "Save browser settings", "Browser settings dialog", regions
    )
    assert decision.operation == "CLICK"
    assert decision.region == regions[1]
    assert len(seen) == 1
    assert "operation" in seen[0]["questions"]
    assert "target" in seen[0]["questions"]
    assert "coordinates" not in seen[0]["state"]


def test_kev_computer_agent_uses_model_decisions_and_returns_fresh_ocr():
    region = OCRRegion("O1", "Open settings", 0.9, 10, 20, 120, 50)

    class Controller:
        max_text_length = 1000
        snapshot_id = "s1"
        _ocr_state = "Open settings"
        _ocr_regions = [region]

        def __init__(self):
            self.calls = []

        def run(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs["action"] == "inspect" and len(self.calls) == 1:
                return type("Result", (), {"message": "Fresh OCR"})()
            if kwargs["action"] == "click":
                self.snapshot_id = "s2"
                self._ocr_state = "Settings panel"
            return type("Result", (), {"message": f"{kwargs['action']} returned fresh OCR"})()

    class Decisions:
        def __init__(self):
            self.values = [
                KevActionDecision("CLICK", region, 0.7, "Jev selected CLICK."),
            ]

        def decide_next(self, *args):
            return self.values.pop(0)

        def check_goal_done(self, *args):
            return True, "Jev completion check: YES."

    controller = Controller()
    response = KevComputerAgent(controller, Decisions()).run("Open settings", max_steps=4)
    assert "Jev selected CLICK" in response
    assert "COMPLETION_CANDIDATE" in response
    assert controller.calls[1]["ocr_region_ref"] == "O1"


def test_kev_computer_agent_rejects_unmatched_quoted_target_and_marks_incomplete():
    region = OCRRegion("O1", "VIDEOS", 0.99, 10, 20, 120, 50)

    class Controller:
        max_text_length = 1000
        snapshot_id = "s1"
        _ocr_state = "VIDEOS"
        _ocr_regions = [region]

        def __init__(self):
            self.calls = []

        def run(self, **kwargs):
            self.calls.append(kwargs)
            return type("Result", (), {"message": "Fresh OCR"})()

    class Decisions:
        def decide_next(self, *args):
            return KevActionDecision("CLICK", region, 0.7, "Kev selected CLICK.")

    controller = Controller()
    result = KevComputerAgent(controller, Decisions()).run(
        'Open the video titled "Why Are Gas Station Roofs So HUGE?"'
    )
    assert len(controller.calls) == 1  # initial inspect only; no click dispatched
    assert "does not match the explicitly named target" in result
    assert "DESKTOP_TASK_STATUS: INCOMPLETE" in result


def test_ocr_only_computer_control_withholds_image_and_requires_kev_text_target(monkeypatch):
    region = OCRRegion("O1", "Open", 0.9, 20, 30, 80, 60)

    class Reader:
        def read(self, _image):
            return [region]

    controller = ComputerController(
        enabled=True,
        screenshot_fn=png_bytes,
        environ={"XDG_SESSION_TYPE": "x11", "DISPLAY": ":0"},
        ocr_only=True,
        ocr_reader=Reader(),
        target_selector=lambda goal, target, state, regions: (regions[0], "Kev selected O1."),
    )
    controller._read_active_window_state = lambda: ("test-window", (0, 0, 320, 200))
    inspected = controller._capture("Inspect")
    assert inspected.screenshot is None
    assert "text='Open'" in inspected.message
    assert "screenshot pixels" not in inspected.message
    assert controller._ocr_regions == [region]

    no_text = controller.run("click", snapshot_id=controller.snapshot_id, x=50, y=45)
    assert "target_text" in no_text.message
    assert controller._snapshot_id == ""
    fresh = controller.run("inspect", screenshot_delay_seconds=0)
    assert controller._snapshot_id

    monkeypatch.setattr(ComputerController, "available", property(lambda _self: True))
    clicks = []
    monkeypatch.setattr(controller, "_click", lambda x, y, button: clicks.append((x, y, button)))
    clicked = controller.run(
        "click", snapshot_id=controller.snapshot_id,
        target_text="Open this item", screenshot_delay_seconds=0
    )
    assert clicks == [(50, 45, "left")]
    assert "Kev selected O1" in clicked.message


def test_visual_inspection_can_skip_ocr_and_region_grounding():
    calls = {"ocr": 0, "grounder": 0}

    class Reader:
        def read_zoomed_band(self, *_args, **_kwargs):
            calls["ocr"] += 1
            return [OCRRegion("P1", "Ready", 0.99, 5, 5, 50, 25)]

    def grounder(image):
        calls["grounder"] += 1
        return image, "Visual regions"

    screenshot = png_bytes()
    controller = ComputerController(
        screenshot_fn=lambda: screenshot,
        environ={"XDG_SESSION_TYPE": "x11", "DISPLAY": ":0"},
        ocr_reader=Reader(),
        visual_grounder=grounder,
    )
    controller._read_active_window_state = lambda: ("test-window", (0, 0, 320, 200))

    visual_only = controller.run("inspect", include_ocr=False, screenshot_delay_seconds=0)
    assert visual_only.screenshot == screenshot
    assert "Ready" not in visual_only.message
    assert "Visual regions" not in visual_only.message
    assert calls == {"ocr": 0, "grounder": 0}

    with_details = controller.run("inspect", include_ocr=True, screenshot_delay_seconds=0)
    assert with_details.screenshot is not None
    assert "Ready" in with_details.message
    assert "Visual regions" in with_details.message
    assert calls == {"ocr": 2, "grounder": 1}


def test_screen_ocr_load_preloads_engine():
    loaded = []

    class MockEngine:
        pass

    class LazyOCR(ScreenOCR):
        def _get_engine(self):
            if self._engine is None:
                engine = MockEngine()
                loaded.append(engine)
                self._engine = engine
            return self._engine

    ocr = LazyOCR(engine=None)
    assert ocr._engine is None
    engine = ocr.load()
    assert ocr._engine is engine
    assert len(loaded) == 1
    # Calling load again returns cached engine
    assert ocr.load() is engine
    assert len(loaded) == 1


def test_screen_ocr_preload_flag():
    loaded = []

    class MockEngine:
        pass

    class PreloadingOCR(ScreenOCR):
        def _get_engine(self):
            if self._engine is None:
                engine = MockEngine()
                loaded.append(engine)
                self._engine = engine
            return self._engine

    ocr = PreloadingOCR(engine=None, preload=True)
    assert ocr._engine is not None
    assert len(loaded) == 1


def test_cuda_ocr_uses_bounded_arena_and_lightweight_cudnn_search(monkeypatch):
    import rapidocr

    captured = {}

    class Session:
        def get_providers(self):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    component = SimpleNamespace(session=SimpleNamespace(session=Session()))
    engine = SimpleNamespace(text_det=component, text_cls=component, text_rec=component)

    def build_engine(params):
        captured.update(params)
        return engine

    monkeypatch.setattr(rapidocr, "RapidOCR", build_engine)
    monkeypatch.setattr(ScreenOCR, "_resolve_gpu_index", staticmethod(lambda _uuid: 0))
    ocr = ScreenOCR(device="cuda", gpu_uuid="GPU-test")

    assert ocr.load() is engine
    assert captured["EngineConfig.onnxruntime.cuda_ep_cfg.arena_extend_strategy"] == "kSameAsRequested"
    assert captured["EngineConfig.onnxruntime.cuda_ep_cfg.cudnn_conv_algo_search"] == "HEURISTIC"
    assert captured["EngineConfig.onnxruntime.cuda_ep_cfg.cudnn_conv_use_max_workspace"] is False


def test_brain_preloads_ocr_at_startup():
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch
    from src.llm.brain import AdamBrain

    cfg = SimpleNamespace(
        computer_control=SimpleNamespace(enabled=True, ocr_only=True, ocr_device="cpu"),
        computer_vision=None,
        llm=SimpleNamespace(
            provider="local",
            local_model="test",
            cloud_model="",
            ollama_host="http://localhost:11434",
            temperature=0.3,
            num_ctx=16384,
            think="low",
            api_key="",
        ),
        desktop=SimpleNamespace(default_browser=""),
    )

    with patch.object(ScreenOCR, "load", autospec=True) as mock_load:
        brain = AdamBrain(cfg, None, None, None, None, preload_ocr=True)
        assert brain.screen_ocr is not None
        mock_load.assert_called_once()


def test_visual_inspection_uses_full_screen_ocr_reader_when_available():
    calls = {"read": 0, "read_zoomed_band": 0}

    class FullReader:
        def read(self, _image):
            calls["read"] += 1
            return [OCRRegion("O1", "Latest Deposit: $1,250", 0.99, 100, 200, 400, 240)]

        def read_zoomed_band(self, *_args, **_kwargs):
            calls["read_zoomed_band"] += 1
            return []

    screenshot = png_bytes()
    controller = ComputerController(
        screenshot_fn=lambda: screenshot,
        environ={"XDG_SESSION_TYPE": "x11", "DISPLAY": ":0"},
        ocr_reader=FullReader(),
    )
    res = controller.run("inspect", include_ocr=True, screenshot_delay_seconds=0)
    assert calls["read"] == 1
    assert calls["read_zoomed_band"] == 0
    assert "Latest Deposit: $1,250" in res.message
