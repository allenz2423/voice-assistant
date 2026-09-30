import json

from src.llm.brain import _completion_verdict


def test_completion_requires_fresh_observation_id():
    verdict = {
        "status": "complete",
        "reason": "Requested state is visible.",
        "evidence": ["The current app shows the requested result."],
        "outcomes": [{
            "outcome": "Requested state is visible",
            "status": "complete",
            "evidence": ["The current app shows the requested result."],
        }],
        "observation_id": "obs_123",
    }

    assert _completion_verdict(json.dumps(verdict), "obs_123") == verdict
    assert _completion_verdict(json.dumps(verdict), "obs_456") is None
    assert _completion_verdict(json.dumps({**verdict, "observation_id": ""}), "") is None


def test_completion_verdict_rejects_unstructured_or_unknown_status():
    assert _completion_verdict("Looks done.", "obs_123") is None
    assert _completion_verdict(
        '{"status":"probably","observation_id":"obs_123"}', "obs_123"
    ) is None
    assert _completion_verdict(json.dumps({
        "status": "complete", "observation_id": "obs_123",
        "evidence": ["claim"], "outcomes": [],
    }), "obs_123") is None


def test_incomplete_verdict_does_not_require_observation_id():
    verdict = {"status": "incomplete", "reason": "The file is not saved yet."}
    assert _completion_verdict(json.dumps(verdict), "obs_123") == verdict


def test_authoritative_readback_can_verify_outcome_without_window_identity():
    verdict = {
        "status": "complete",
        "reason": "Readback confirms the saved file.",
        "evidence": ["The exact file contents were read back."],
        "outcomes": [{
            "outcome": "File exists with requested content",
            "status": "complete",
            "evidence": "Readback confirms the exact contents.",
        }],
    }
    assert _completion_verdict(json.dumps(verdict), "", authoritative_readback=True)
