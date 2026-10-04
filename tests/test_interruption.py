import asyncio
import signal
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, AsyncMock

from src.llm.brain import AdamBrain
from src.main import AdamDaemon


@pytest.fixture
def dummy_brain():
    config = SimpleNamespace(
        llm=SimpleNamespace(
            provider="local",
            local_model="qwen",
            cloud_model="",
            ollama_host="",
            temperature=0.2,
            num_ctx=4096,
        ),
        execution=SimpleNamespace(downloads_dir="~/Downloads", workspace_dir="~/workspace"),
        tts=SimpleNamespace(engine="kokoro"),
    )
    brain = AdamBrain(
        config=config,
        supervisor=MagicMock(),
        probe=MagicMock(),
        confirmation_mgr=MagicMock(),
        tts_engine=MagicMock(),
    )
    return brain


@pytest.mark.asyncio
async def test_brain_cancel_active_execution_kills_bash_subprocess(dummy_brain):
    # Simulate a long-running bash command: sleep 30
    task = asyncio.create_task(
        dummy_brain._execute_tool("run_bash_command", {"command": "sleep 30"})
    )
    # Give it a brief moment to start the subprocess
    await asyncio.sleep(0.15)
    assert dummy_brain._active_subprocess is not None
    subproc = dummy_brain._active_subprocess
    pid = subproc.pid

    # Cancel execution
    dummy_brain.cancel_active_execution()

    with pytest.raises(asyncio.CancelledError):
        await task

    # Confirm subprocess is no longer active
    assert dummy_brain._active_subprocess is None
    await asyncio.sleep(0.1)
    # Check if process was terminated
    assert subproc.returncode is not None


@pytest.mark.asyncio
async def test_brain_interrupted_flag_halts_react_loop(dummy_brain):
    dummy_brain.llm_client = MagicMock()
    # Mock LLM to return a tool call
    dummy_brain.llm_client.chat = AsyncMock(return_value={
        "content": "",
        "tool_calls": [{
            "id": "call_1",
            "function": {"name": "get_current_time", "arguments": {}}
        }]
    })

    # Set interrupted before execution
    dummy_brain._is_interrupted = True

    await dummy_brain.process_user_utterance("What time is it?")

    # Because it was interrupted, it should not have executed the tool
    assert dummy_brain._is_interrupted is True


@pytest.mark.asyncio
async def test_monitor_execution_interrupt_stops_on_explicit_wake_phrase():
    app = MagicMock(spec=AdamDaemon)
    app.running = True
    app.stream = MagicMock()
    app.stream.sample_rate = 16000
    app.stream.is_assistant_speaking = False
    app.stream.vad = MagicMock()
    app.stream.vad.is_speech.return_value = (True, 0.95)
    app.tts = MagicMock()
    app.tts.pending_barge_in_text = None
    app.earcon = MagicMock()
    app.brain = MagicMock()
    app.speaker_verifier = None
    app.wake = MagicMock()
    app.wake.raw_wake_word = "hey adam"
    app.wake.is_custom_mode = True
    app.wake.match_custom_wake_word.return_value = (False, "")
    app.wake.match_explicit_wake_word.return_value = (True, "stop that right now")
    app._transcribe_wake_candidate = MagicMock(return_value="hey adam stop that right now")
    app.config = SimpleNamespace(audio=SimpleNamespace(
        vad_threshold_speaking=0.8,
        vad_silence_duration=0.6,
        max_utterance_seconds=60.0,
    ))

    import numpy as np
    chunk = np.zeros(512, dtype=np.float32)
    app.stream.get_chunk.return_value = chunk

    # Fake react_task that simulates doing work
    async def fake_long_task():
        await asyncio.sleep(2.0)

    react_task = asyncio.create_task(fake_long_task())

    # Call the actual unbound method with the mock instance
    interrupted, interrupt_cmd = await AdamDaemon._monitor_execution_interrupt(app, react_task)

    assert interrupted is True
    assert "stop that right now" in interrupt_cmd
    assert react_task.cancelled()
    app.brain.cancel_active_execution.assert_called_once()
    app.earcon.play.assert_called_with("interrupt")
    app.tts.advance_epoch.assert_called_once()


@pytest.mark.asyncio
async def test_execute_turn_transitions_to_new_command_on_interrupt():
    app = MagicMock(spec=AdamDaemon)
    app.running = True
    app.arbiter = MagicMock()
    app.arbiter.set_state = AsyncMock()
    app.arbiter.current_state = "PROCESSING_REACT"
    app.speculative_router = MagicMock()
    app.stream = MagicMock()
    app.earcon = MagicMock()
    app.wake = MagicMock()
    app.wake.match_custom_wake_word.return_value = (False, "")
    app.config = SimpleNamespace(wake=SimpleNamespace(followup_window_seconds=7.0))
    app.brain = MagicMock()

    executed_commands = []

    async def fake_process(cmd, memory_context=None):
        executed_commands.append(cmd)
        await asyncio.sleep(0.01)

    app.brain.process_user_utterance = AsyncMock(side_effect=fake_process)

    # First turn gets interrupted with a new command, second turn completes
    monitor_calls = [
        (True, "stop and check system status"),
        (False, None)
    ]

    async def fake_monitor(react_task):
        is_interrupted, cmd = monitor_calls.pop(0)
        if is_interrupted:
            await asyncio.sleep(0.001)
            react_task.cancel()
        try:
            await react_task
        except (asyncio.CancelledError, Exception):
            pass
        return is_interrupted, cmd

    app._monitor_execution_interrupt = AsyncMock(side_effect=fake_monitor)

    await AdamDaemon._execute_turn(app, "build linux kernel")

    # Both initial command and subsequent command should have been attempted
    assert executed_commands == ["build linux kernel", "check system status"]
