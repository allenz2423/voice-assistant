from tools.setup_wizard import NEMOTRON_DIARIZATION_MODEL, choose_nvidia_device
from src.config import SpeakerDiarizationConfig, STTConfig


def test_setup_uses_official_nemotron_transformers_model():
    assert NEMOTRON_DIARIZATION_MODEL == "nvidia/Nemotron-3-Diarization"


def test_legacy_gguf_config_migrates_to_official_model():
    config = SpeakerDiarizationConfig(model="assets/models/Nemotron-3-Diarization.q8_0.gguf", device="vulkan:0")

    assert config.model == NEMOTRON_DIARIZATION_MODEL
    assert config.device == "auto"


def test_setup_can_select_exact_cuda_device(monkeypatch):
    import tools.setup_wizard as wizard

    monkeypatch.setattr(wizard, "prompt_choice", lambda *_args: 1)
    assert choose_nvidia_device(
        "Select GPU", [("0", "GTX 1080 Ti"), ("1", "RTX 3070")], "0"
    ) == "1"


def test_qwen_defaults_to_english_language_hint():
    assert STTConfig().language == "English"
