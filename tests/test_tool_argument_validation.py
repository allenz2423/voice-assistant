from src.llm.tools import ADAM_TOOLS, CanonicalTool, validate_tool_arguments


def _tool(name: str) -> CanonicalTool:
    return next(tool for tool in ADAM_TOOLS if tool.name == name)


def test_schema_validation_checks_required_fields_and_nested_types():
    tool = CanonicalTool(
        name="sample",
        description="",
        parameters={
            "type": "object",
            "properties": {
                "label": {"type": "string", "minLength": 1},
                "count": {"type": "integer", "minimum": 1},
            },
            "required": ["label", "count"],
            "additionalProperties": False,
        },
    )

    assert validate_tool_arguments(tool, {"label": "ok", "count": 2}) is None
    assert "sample.count" in validate_tool_arguments(tool, {"label": "ok", "count": True})
    assert "Missing required" in validate_tool_arguments(tool, {"label": "ok"})
    assert "not an accepted argument" in validate_tool_arguments(tool, {"label": "ok", "count": 2, "extra": 1})


def test_computer_action_validation_rejects_incomplete_coordinates():
    error = validate_tool_arguments(
        _tool("computer_control"),
        {"action": "click", "snapshot_id": "s1", "x": 12},
    )
    assert "missing: y" in error


def test_sequence_schema_validates_each_action():
    error = validate_tool_arguments(
        _tool("computer_control"),
        {
            "action": "sequence",
            "snapshot_id": "s1",
            "actions": [{"action": "type"}],
        },
    )
    assert "missing: text" in error
