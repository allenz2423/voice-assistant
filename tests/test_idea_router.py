import json
from collections import Counter
from pathlib import Path

import numpy as np

from src.intent.idea_router import IdeaRouter, selected_onnx_file


def test_compact_model_selection_uses_host_architecture():
    assert selected_onnx_file("x86_64", {"avx2"}) == "onnx/model_quint8_avx2.onnx"
    assert selected_onnx_file("aarch64", set()) == "onnx/model_qint8_arm64.onnx"
    assert selected_onnx_file("x86_64", {"sse4_2"}) == "onnx/model.onnx"


def test_background_candidate_threshold_is_stricter_than_command_threshold():
    router = IdeaRouter()
    assert router.thresholds["command"] == 0.30
    assert router.thresholds["background_calendar"] == 0.49


def test_idea_router_requires_threshold_and_runner_up_margin(tmp_path):
    idea_path = tmp_path / "ideas.json"
    idea_path.write_text(json.dumps({"ideas": [
        {"id": "launch", "title": "Launch app", "description": "Start an app", "route": "command"},
        {"id": "event", "title": "Future event", "description": "Personal event later", "route": "background_calendar"},
        {"id": "ignore", "title": "Conversation", "description": "Unrelated chat", "route": "ignore"},
    ]}), encoding="utf-8")
    router = IdeaRouter(
        ideas_path=idea_path,
        command_threshold=0.30,
        background_threshold=0.46,
        minimum_margin=0.025,
    )
    router._idea_vectors = np.eye(3, dtype=np.float32)
    router.encode = lambda texts, batch_size=32: np.asarray([[0.8, 0.3, 0.1]], dtype=np.float32)
    match = router.match("Open Spotify")
    assert match is not None
    assert match.idea_id == "launch"
    assert match.accepted

    router.encode = lambda texts, batch_size=32: np.asarray([[0.8, 0.79, 0.1]], dtype=np.float32)
    uncertain = router.match("Something ambiguous")
    assert uncertain is not None
    assert not uncertain.accepted


def test_idea_router_ignores_empty_input(tmp_path):
    router = IdeaRouter(ideas_path=tmp_path / "not-loaded.json")
    assert router.match(" \n ") is None


def test_phrase_corpus_has_two_thousand_labeled_phrases():
    path = Path(__file__).parent / "data" / "idea_router_phrases_2000.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2000
    assert len({row["text"] for row in rows}) == 2000
    assert Counter(row["expected"] for row in rows) == {
        "direct_action": 500,
        "personal_calendar_commitment": 500,
        "none": 1000,
    }
