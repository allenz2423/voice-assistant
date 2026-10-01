import json

from src.llm.brain import _tool_result_message


def test_tool_result_separates_execution_from_goal_completion():
    result = json.loads(_tool_result_message(
        call_id="call_1",
        origin="native",
        status="ok",
        result="Clicked left at (100, 200). Fresh screenshot attached.",
        duration_ms=25,
        dispatched=True,
    ))
    assert result["call_id"] == "call_1"
    assert result["status"] == "ok"
    assert result["effect_status"] == "dispatched"
    assert result["goal_status"] == "not_assessed"
    assert result["result_completeness"] == "complete"
    assert "does not" in result["message"] or "assess" in result["message"]


def test_tool_result_marks_unstructured_effect_as_unknown():
    result = json.loads(_tool_result_message(
        call_id="call_2",
        origin="native",
        status="returned",
        result="Application reported that the requested item was not found.",
        duration_ms=12,
    ))
    assert result["status"] == "returned"
    assert result["effect_status"] == "unknown"
    assert result["goal_status"] == "not_assessed"
    assert result["result_completeness"] == "unknown"


def test_tool_result_timeout_is_partial_with_recovery_guidance():
    result = json.loads(_tool_result_message(
        call_id="call_3",
        origin="native",
        status="timed_out",
        result="The request timed out.",
        duration_ms=90000,
    ))
    assert result["result_completeness"] == "partial"
    assert "inspect state" in result["message"]


def test_invalid_arguments_are_not_labeled_as_unavailable_capability():
    result = json.loads(_tool_result_message(
        call_id="call_4",
        origin="native",
        status="invalid_input",
        result="computer_control.click is missing y.",
        duration_ms=0,
        dispatched=False,
    ))
    assert result["status"] == "invalid_input"
    assert result["result_completeness"] == "unknown"
    assert "Correct the arguments" in result["message"]
