"""Focused tests for Adam's optional WebUI sidecar."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock
import pytest
from aiohttp import web
from aiohttp.client_exceptions import WSServerHandshakeError
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop, TestClient, TestServer

from src.config import WebUIConfig, AppConfig
from src.arbiter.arbiter import SystemState
from sidecar.bridge import (
    RuntimeBridge,
    DaemonBridge,
    DisconnectedBridge,
    StandaloneBridge,
)
from sidecar.server import (
    _is_loopback,
    create_app,
    SidecarServer,
    security_headers_middleware,
    create_auth_middleware,
    create_loopback_middleware,
)


# ---------------------------------------------------------------------------
# Configuration & Host Binding Tests
# ---------------------------------------------------------------------------


def test_webui_config_defaults():
    config = WebUIConfig()
    assert config.enabled is False
    assert config.host == "127.0.0.1"
    assert config.port == 8765
    assert config.auth_token == ""


def test_is_loopback_classification():
    assert _is_loopback("127.0.0.1") is True
    assert _is_loopback("127.0.1.1") is True
    assert _is_loopback("127.255.255.255") is True
    assert _is_loopback("::1") is True
    assert _is_loopback("localhost") is True
    assert _is_loopback("testclient") is True
    assert _is_loopback("::ffff:127.0.0.1") is True

    # Remote / non-loopback
    assert _is_loopback("0.0.0.0") is False
    assert _is_loopback("192.168.1.50") is False
    assert _is_loopback("10.0.0.1") is False
    assert _is_loopback("adam.example.com") is False
    assert _is_loopback("") is False
    assert _is_loopback(None) is False


def test_create_app_enforces_loopback_only():
    bridge = DisconnectedBridge()
    remote_config = WebUIConfig(host="192.168.1.100")
    with pytest.raises(ValueError) as excinfo:
        create_app(remote_config, bridge)
    assert "Insecure host" in str(excinfo.value)
    assert "loopback only" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Disconnected & Standalone Bridge Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disconnected_bridge_truthful_reports():
    bridge = DisconnectedBridge()
    status = await bridge.get_status()

    assert status["connected"] is False
    assert status["mode"] == "disconnected"
    assert status["system_state"] == "DISCONNECTED"
    assert status["voice_default"] is True
    assert status["tools_count"] == 0
    assert status["features"]["chat"]["available"] is False
    assert "daemon is not connected" in status["features"]["chat"]["reason"]
    assert status["features"]["long_running_tasks"]["available"] is False
    assert status["features"]["remote_approvals"]["available"] is False

    history = await bridge.get_history()
    assert history == []

    res = await bridge.handle_user_message("Hello Adam")
    assert res["status"] == "error"
    assert "disconnected" in res["error"].lower()


def test_standalone_bridge_does_not_create_second_brain():
    bridge = StandaloneBridge()
    assert isinstance(bridge, DisconnectedBridge)
    # Does not have brain or arbiter instances claiming live integration
    assert not hasattr(bridge, "brain") or bridge.brain is None


# ---------------------------------------------------------------------------
# Daemon Bridge Tests
# ---------------------------------------------------------------------------


class MockArbiter:
    def __init__(self, initial_state=SystemState.IDLE_LISTENING):
        self.current_state = initial_state
        self.state_history = [initial_state]

    async def set_state(self, state):
        if isinstance(state, str):
            state = getattr(SystemState, state, state)
        self.current_state = state
        self.state_history.append(state)


class MockBrain:
    def __init__(self):
        self.messages = [
            {"role": "system", "content": "You are Adam"},
            {
                "role": "user",
                "content": "[Current Desktop State]\nWindow: Terminal\n[Local Time: 12:00:00]\nOpen firefox",
            },
            {
                "role": "assistant",
                "content": "Opening Firefox now.",
                "tool_calls": [{"function": {"name": "launch_application"}}],
            },
        ]
        self.llm_client = MagicMock(model="qwen3.8-27b")

    def get_tools(self):
        return ["launch_application", "get_weather", "web_search"]

    async def process_user_utterance(self, text, memory_context=None):
        self.messages.append({"role": "user", "content": text})
        self.messages.append({
            "role": "assistant",
            "content": f"Processed: {text}",
            "tool_calls": [{"function": {"name": "get_weather"}}],
        })


def attach_mock_turn_executor(daemon, arbiter, brain):
    async def execute_turn(text, memory_context=None):
        await arbiter.set_state("PROCESSING_REACT")
        try:
            await brain.process_user_utterance(text, memory_context=memory_context)
        finally:
            if arbiter.current_state != SystemState.AWAITING_CONFIRMATION:
                await arbiter.set_state("IDLE_LISTENING")

    daemon._execute_turn = AsyncMock(side_effect=execute_turn)


@pytest.mark.asyncio
async def test_daemon_bridge_status_and_features():
    arbiter = MockArbiter(SystemState.IDLE_LISTENING)
    brain = MockBrain()
    daemon = MagicMock(arbiter=arbiter, brain=brain, memory_manager=None)

    bridge = DaemonBridge(daemon)
    status = await bridge.get_status()

    assert status["connected"] is True
    assert status["mode"] == "daemon"
    assert status["system_state"] == "IDLE_LISTENING"
    assert status["model"] == "qwen3.8-27b"
    assert status["tools_count"] == 3
    assert status["features"]["chat"]["available"] is True
    # Long running tasks and remote approvals are truthfully marked unavailable
    assert status["features"]["long_running_tasks"]["available"] is False
    assert status["features"]["remote_approvals"]["available"] is False


@pytest.mark.asyncio
async def test_daemon_bridge_history_strips_desktop_headers():
    arbiter = MockArbiter()
    brain = MockBrain()
    daemon = MagicMock(arbiter=arbiter, brain=brain)

    bridge = DaemonBridge(daemon)
    history = await bridge.get_history()

    assert len(history) == 2
    # Verify desktop header was stripped from user message
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Open firefox"
    assert "[Current Desktop State]" not in history[0]["content"]

    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "Opening Firefox now."
    assert history[1]["tool_calls"] == ["launch_application"]


@pytest.mark.asyncio
async def test_daemon_bridge_refuses_when_awaiting_verbal_confirmation():
    # If Adam is awaiting verbal confirmation via mic, WebUI MUST NOT bypass confirmation
    arbiter = MockArbiter(SystemState.AWAITING_CONFIRMATION)
    brain = MockBrain()
    daemon = MagicMock(arbiter=arbiter, brain=brain)

    bridge = DaemonBridge(daemon)
    res = await bridge.handle_user_message("delete files")

    assert res["status"] == "busy"
    assert "awaiting verbal confirmation" in res["error"].lower()
    assert "microphone" in res["error"].lower()
    # Arbiter state must not be modified
    assert arbiter.current_state == SystemState.AWAITING_CONFIRMATION


@pytest.mark.asyncio
async def test_daemon_bridge_refuses_when_busy_with_voice_task():
    # If Adam is speaking or processing a voice task, WebUI waits safely
    arbiter = MockArbiter(SystemState.ASSISTANT_SPEAKING)
    brain = MockBrain()
    daemon = MagicMock(arbiter=arbiter, brain=brain)

    bridge = DaemonBridge(daemon)
    res = await bridge.handle_user_message("what is the weather")

    assert res["status"] == "busy"
    assert "busy" in res["error"].lower()
    assert arbiter.current_state == SystemState.ASSISTANT_SPEAKING


@pytest.mark.asyncio
async def test_daemon_bridge_processes_turn_safely():
    arbiter = MockArbiter(SystemState.IDLE_LISTENING)
    brain = MockBrain()
    daemon = MagicMock(arbiter=arbiter, brain=brain, memory_manager=None)
    attach_mock_turn_executor(daemon, arbiter, brain)

    broadcast_events = []
    bridge = DaemonBridge(daemon)

    async def on_broadcast(payload):
        broadcast_events.append(payload)

    await bridge.register_listener(on_broadcast)

    res = await bridge.handle_user_message("Check battery level")

    assert res["status"] == "completed"
    assert res["response"] == "Processed: Check battery level"
    assert res["tool_calls"] == ["get_weather"]

    # Verify arbiter transitioned to PROCESSING_REACT and restored IDLE_LISTENING
    assert SystemState.PROCESSING_REACT in arbiter.state_history
    assert arbiter.current_state == SystemState.IDLE_LISTENING
    daemon._execute_turn.assert_awaited_once_with("Check battery level", memory_context=None)
    assert any(e.get("system_state") == "PROCESSING_REACT" for e in broadcast_events)


@pytest.mark.asyncio
async def test_bridge_captures_reply_after_real_brain_history_compaction():
    from src.llm.brain import AdamBrain

    class CompactingBrain(MockBrain):
        async def process_user_utterance(self, text, memory_context=None):
            self.system_prompt = "You are Adam"
            self._turn_message_start = len(self.messages)
            AdamBrain._compact_history_for_new_turn(self)
            await super().process_user_utterance(text, memory_context)

    arbiter = MockArbiter()
    brain = CompactingBrain()
    daemon = MagicMock(arbiter=arbiter, brain=brain, memory_manager=None)
    attach_mock_turn_executor(daemon, arbiter, brain)
    bridge = DaemonBridge(daemon)
    for index in range(10):
        text = f"Request {index}"
        result = await bridge.handle_user_message(text)
        assert result["status"] == "completed"
        assert result["response"] == f"Processed: {text}"
        assert result["tool_calls"] == ["get_weather"]


@pytest.mark.asyncio
async def test_bridge_does_not_claim_empty_turn_completed():
    arbiter = MockArbiter()
    brain = MockBrain()
    # A previous turn's boundary must not make a no-op return an old reply.
    brain._turn_message_start = 1
    daemon = MagicMock(arbiter=arbiter, brain=brain, memory_manager=None)
    daemon._execute_turn = AsyncMock()
    result = await DaemonBridge(daemon).handle_user_message("Read a folder")
    assert result["status"] == "error"
    assert "without a text reply" in result["error"]


# ---------------------------------------------------------------------------
# HTTP & WebSocket Server Tests (AioHTTP)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_server_status_and_security_headers():
    arbiter = MockArbiter(SystemState.IDLE_LISTENING)
    brain = MockBrain()
    daemon = MagicMock(arbiter=arbiter, brain=brain, memory_manager=None)
    bridge = DaemonBridge(daemon)

    config = WebUIConfig(enabled=True, host="127.0.0.1", port=8765, auth_token="")
    app = create_app(config, bridge)
    async with TestClient(TestServer(app)) as client:
        # Status endpoint
        resp = await client.get("/api/status")
        assert resp.status == 200
        data = await resp.json()
        assert data["connected"] is True
        assert data["mode"] == "daemon"
        assert data["auth_required"] is False
        assert data["is_loopback"] is True

        # Security headers
        assert resp.headers["X-Content-Type-Options"] == "nosniff"
        assert resp.headers["X-Frame-Options"] == "DENY"
        assert "default-src 'self'" in resp.headers["Content-Security-Policy"]


@pytest.mark.asyncio
async def test_server_authentication_flow():
    arbiter = MockArbiter()
    brain = MockBrain()
    daemon = MagicMock(arbiter=arbiter, brain=brain)
    bridge = DaemonBridge(daemon)

    token = "secret-token-123"
    config = WebUIConfig(enabled=True, host="127.0.0.1", port=8765, auth_token=token)
    app = create_app(config, bridge)
    async with TestClient(TestServer(app)) as client:
        # 1. Unauthenticated auth/status succeeds and reports auth is required
        resp = await client.get("/api/auth/status")
        assert resp.status == 200
        auth_data = await resp.json()
        assert auth_data["auth_required"] is True
        assert auth_data["authenticated"] is False

        # 2. Protected endpoints reject missing token
        resp = await client.get("/api/status")
        assert resp.status == 401
        err = await resp.json()
        assert "Unauthorized" in err["error"]
        assert err["auth_required"] is True

        resp = await client.get("/api/history")
        assert resp.status == 401

        resp = await client.post("/api/chat", json={"message": "hello"})
        assert resp.status == 401

        # 3. Protected endpoints reject incorrect token
        resp = await client.get("/api/status", headers={"Authorization": "Bearer wrong-token"})
        assert resp.status == 401

        # 4. Bearer header authentication succeeds
        resp = await client.get("/api/status", headers={"Authorization": f"Bearer {token}"})
        assert resp.status == 200
        data = await resp.json()
        assert data["connected"] is True

        # Tokens in URLs are rejected so access logs cannot capture them.
        resp = await client.get(f"/api/status?token={token}")
        assert resp.status == 401

        resp = await client.get(f"/api/auth/status?token={token}")
        assert resp.status == 200
        assert (await resp.json())["authenticated"] is False

        with pytest.raises(WSServerHandshakeError):
            await client.ws_connect(f"/api/ws?token={token}")

        # The UI exchanges a bearer header for a scoped HttpOnly WebSocket cookie.
        resp = await client.post(
            "/api/auth/session", headers={"Authorization": f"Bearer {token}"}
        )
        assert resp.status == 200
        cookie = resp.cookies["adam_webui_ws_auth"]
        assert cookie["httponly"] is True
        assert cookie["path"] == "/api/ws"
        assert cookie["samesite"] == "Strict"

        ws = await client.ws_connect("/api/ws")
        assert (await ws.receive_json())["type"] == "status"
        await ws.close()

        resp = await client.delete(
            "/api/auth/session", headers={"Authorization": f"Bearer {token}"}
        )
        assert resp.status == 200


@pytest.mark.asyncio
async def test_server_loopback_middleware_rejection():
    bridge = DisconnectedBridge()
    config = WebUIConfig(enabled=True, host="127.0.0.1", port=8765)
    app = create_app(config, bridge)

    # Simulate remote request through custom request modification
    @web.middleware
    async def spoof_remote_middleware(request, handler):
        # Mutate remote address to simulate external request
        request._remote_override = "203.0.113.195"
        return await handler(request)

    app.middlewares.insert(0, spoof_remote_middleware)
    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/api/status")
        assert resp.status == 403
        err = await resp.json()
        assert "loopback by default" in err["error"]


@pytest.mark.asyncio
async def test_server_rejects_dns_rebinding_and_cross_origin_requests():
    bridge = DisconnectedBridge()
    app = create_app(WebUIConfig(enabled=True, host="127.0.0.1", port=8765), bridge)
    async with TestClient(TestServer(app)) as client:
        valid_origin = str(client.make_url("/")).rstrip("/")
        response = await client.get("/api/status", headers={"Origin": valid_origin})
        assert response.status == 200

        response = await client.get("/api/status", headers={"Origin": "http://evil.example"})
        assert response.status == 403
        assert "same-origin" in (await response.json())["error"]

        response = await client.get("/api/status", headers={"Host": "evil.example"})
        assert response.status == 403
        assert "Host" in (await response.json())["error"]

        with pytest.raises(WSServerHandshakeError):
            await client.ws_connect(
                "/api/ws", headers={"Origin": "http://evil.example"}
            )


@pytest.mark.asyncio
async def test_websocket_chat_and_status():
    arbiter = MockArbiter(SystemState.IDLE_LISTENING)
    brain = MockBrain()
    daemon = MagicMock(arbiter=arbiter, brain=brain, memory_manager=None)
    attach_mock_turn_executor(daemon, arbiter, brain)
    bridge = DaemonBridge(daemon)

    config = WebUIConfig(enabled=True, host="127.0.0.1", port=8765, auth_token="")
    app = create_app(config, bridge)
    async with TestClient(TestServer(app)) as client:
        # Connect WebSocket
        ws = await client.ws_connect("/api/ws")

        # Initial status packet on connection
        init_msg = await ws.receive_json()
        assert init_msg["type"] == "status"
        assert init_msg["connected"] is True

        # Ping / Pong
        await ws.send_json({"type": "ping"})
        pong = await ws.receive_json()
        assert pong["type"] == "pong"

        # Status request
        await ws.send_json({"type": "status"})
        status_msg = await ws.receive_json()
        assert status_msg["type"] == "status"
        assert status_msg["model"] == "qwen3.8-27b"

        # Chat request
        await ws.send_json({"type": "chat", "message": "List active windows"})

        # Collect the responses produced by the chat turn
        # DaemonBridge broadcasts state transitions (PROCESSING_REACT -> IDLE_LISTENING)
        # and ws sends chat_response
        received = []
        for _ in range(3):
            msg = await ws.receive_json()
            received.append(msg)

        msg_types = [m["type"] for m in received]
        assert "state" in msg_types
        assert "chat_response" in msg_types

        # Verify thinking state was broadcast
        state_msgs = [m for m in received if m["type"] == "state"]
        states = [m["system_state"] for m in state_msgs]
        assert "PROCESSING_REACT" in states
        assert "IDLE_LISTENING" in states

        # Verify chat response result
        chat_msg = next(m for m in received if m["type"] == "chat_response")
        assert chat_msg["result"]["status"] == "completed"
        assert chat_msg["result"]["response"] == "Processed: List active windows"

        await ws.close()


# ---------------------------------------------------------------------------
# Daemon Startup Wiring Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_daemon_webui_default_disabled():
    from src.main import AdamDaemon

    # Mock all dependencies to avoid loading models or hardware
    with (
        pytest.MonkeyPatch.context() as mp,
    ):
        mp.setattr("src.main.load_config", lambda *args, **kwargs: AppConfig())
        mp.setattr("src.main.WhisperTranscriber", MagicMock())
        mp.setattr("src.main.RobustEarconEngine", MagicMock())
        mp.setattr("src.main.StreamingVoiceSynthesizer", MagicMock())
        mp.setattr("src.main.PriorityAudioArbiter", MagicMock())
        mp.setattr("src.main.TriStateConfirmationManager", MagicMock())
        mp.setattr("src.main.HardwareEncoderProbe", MagicMock())
        mp.setattr("src.main.HardenedJobSupervisor", MagicMock())
        mp.setattr("src.main.create_transcriber", MagicMock())
        mp.setattr("src.main.WakeWordDetector", MagicMock())
        mp.setattr("src.main.AudioStreamManager", MagicMock())
        mp.setattr("src.main.MeetingSession", MagicMock())
        mp.setattr("src.main.SemanticEndpointer", MagicMock())
        mp.setattr("src.main.SpeculativeRouter", MagicMock())
        mp.setattr("src.main.MemoryManager", MagicMock())
        mp.setattr("src.main.AdamBrain", MagicMock())

        daemon = AdamDaemon()
        assert daemon.config.webui.enabled is False
        assert daemon.sidecar_runner is None

        # Explicit override
        daemon_optin = AdamDaemon(enable_webui=True)
        assert daemon_optin.config.webui.enabled is True

@pytest.mark.asyncio
async def test_tool_events_stream_before_turn_finishes_with_logging_disabled(tmp_path):
    from types import SimpleNamespace
    from src.telemetry.events import configure_telemetry, emit_event
    configure_telemetry(SimpleNamespace(enabled=False, path=str(tmp_path / 'events.jsonl')))
    gate = asyncio.Event()
    arbiter = MockArbiter()
    brain = MockBrain()
    daemon = MagicMock(arbiter=arbiter, brain=brain, memory_manager=None)

    async def execute(text, memory_context=None):
        emit_event('tool.started', span_id='live-tool', status='started',
                   attributes={'tool_name': 'find_files', 'arguments': 'private'})
        await gate.wait()
        emit_event('tool.completed', span_id='live-tool', status='ok',
                   attributes={'tool_name': 'find_files', 'outcome': 'returned', 'result': 'private'})
        brain.messages.extend([{'role': 'user', 'content': text},
                               {'role': 'assistant', 'content': 'Folder inspected.'}])

    daemon._execute_turn = AsyncMock(side_effect=execute)
    bridge = DaemonBridge(daemon)
    async with TestClient(TestServer(create_app(WebUIConfig(), bridge))) as client:
        async with client.ws_connect('/api/ws') as ws:
            await ws.receive_json(timeout=2)  # initial status
            await ws.send_json({'type': 'chat', 'message': 'Check the folder'})
            while True:
                event = await ws.receive_json(timeout=2)
                if event.get('type') == 'tool_activity':
                    break
            assert event['tool'] == 'find_files'
            assert event['phase'] == 'started'
            assert not gate.is_set()
            assert 'private' not in str(event)
            gate.set()
            finished = response = None
            while finished is None or response is None:
                event = await ws.receive_json(timeout=2)
                if event.get('type') == 'tool_activity':
                    finished = event
                elif event.get('type') == 'chat_response':
                    response = event
            assert finished['phase'] == 'finished'
            assert finished['outcome'] == 'returned'
            assert finished['duration_ms'] >= 0
            assert 'private' not in str(finished)
            assert response['result']['response'] == 'Folder inspected.'
    assert bridge._event_task is None
    assert bridge._unsubscribe_events is None
    assert not (tmp_path / 'events.jsonl').exists()
