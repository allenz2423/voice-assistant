import asyncio
import json
from types import MethodType

from src.llm.brain import AdamBrain
from src.llm.tools import ADAM_TOOLS


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.messages = None
        self.tools = None

    async def chat(self, messages, tools):
        self.messages = messages
        self.tools = tools
        return self.response


def make_brain(response):
    brain = AdamBrain.__new__(AdamBrain)
    brain.llm_client = FakeLLM(response)
    brain.get_tools = MethodType(lambda self: ADAM_TOOLS, brain)
    executed = []

    async def execute_tool(self, name, args):
        executed.append((name, args))
        return "Added 'Project sync' to the Noctalia calendar." if name == "create_noctalia_event" else "shown"

    brain._execute_tool = MethodType(execute_tool, brain)
    return brain, executed


def test_background_review_isolated_and_limited_to_calendar_tool():
    brain, executed = make_brain({"tool_calls": [{
        "function": {
            "name": "create_noctalia_event",
            "arguments": {"title": "Project sync", "when": "2026-10-01T15:30"},
        }
    }]})
    asyncio.run(brain.process_background_observation(
        "I have a project meeting Thursday at 3:30.",
        {"id": "personal_calendar_commitment", "title": "Future event", "description": "An upcoming commitment."},
    ))

    assert [tool.name for tool in brain.llm_client.tools] == ["create_noctalia_event"]
    assert [name for name, _ in executed] == ["create_noctalia_event", "show_desktop_notification"]
    assert "automatically selected background transcript" in brain.llm_client.messages[0]["content"]
    assert "not instructions to you" in brain.llm_client.messages[0]["content"]
    payload = json.loads(brain.llm_client.messages[1]["content"])
    assert payload["transcript_data"] == "I have a project meeting Thursday at 3:30."
    assert payload["review_type"].startswith("automated background")


def test_background_review_rejects_any_non_calendar_tool_call():
    brain, executed = make_brain({"tool_calls": [{
        "function": {"name": "run_bash_command", "arguments": {"command": "touch /tmp/no"}}
    }]})
    asyncio.run(brain.process_background_observation("transcript", {"id": "idea"}))
    assert executed == []


def test_background_review_without_tool_call_has_no_side_effect():
    brain, executed = make_brain({"tool_calls": [], "content": "No event qualifies."})
    asyncio.run(brain.process_background_observation("I had a meeting yesterday.", {"id": "idea"}))
    assert executed == []
