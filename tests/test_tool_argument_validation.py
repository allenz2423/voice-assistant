from src.llm.tools import ADAM_TOOLS, CanonicalTool, normalize_tool_arguments, validate_tool_arguments


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


def test_schema_normalization_coerces_only_exact_numeric_strings_at_numeric_fields():
    tool = _tool("computer_control")
    arguments = {
        "action": "sequence",
        "snapshot_id": "s1",
        "actions": [
            {"action": "click", "x": "520", "y": " 350 "},
            {"action": "type", "text": "hello"},
            {"action": "click", "target_text": "Save note"},
        ],
    }

    normalized = normalize_tool_arguments(tool, arguments)

    assert normalized == {
        "action": "sequence",
        "snapshot_id": "s1",
        "actions": [
            {"action": "click", "x": 520, "y": 350},
            {"action": "type", "text": "hello"},
            {"action": "click", "target_text": "Save note"},
        ],
    }
    assert arguments["actions"][0]["x"] == "520"
    assert validate_tool_arguments(tool, normalized) is None

    fractional = normalize_tool_arguments(
        tool,
        {"action": "click", "snapshot_id": "s1", "x": "520.5", "y": "350"},
    )
    assert fractional["x"] == "520.5"
    assert "must be integer" in validate_tool_arguments(tool, fractional)


def test_standalone_wait_is_a_documented_delayed_inspection_alias():
    tool = _tool("computer_control")

    assert "standalone action='wait' is accepted as a delayed inspection" in tool.description
    assert "wait" in tool.parameters["properties"]["action"]["enum"]
    assert validate_tool_arguments(
        tool,
        {"action": "wait", "snapshot_id": "s1", "seconds": 1},
    ) is None
    assert validate_tool_arguments(tool, {"action": "wait", "seconds": 1}) is None
    assert validate_tool_arguments(
        tool,
        {
            "action": "sequence",
            "snapshot_id": "s1",
            "actions": [{"action": "wait", "seconds": 1}],
        },
    ) is None
