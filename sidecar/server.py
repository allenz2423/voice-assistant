"""Aiohttp-based HTTP and WebSocket server for Adam's WebUI sidecar.

Binds to loopback by default and enforces optional token authentication.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import hashlib
import inspect
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from aiohttp import web, WSMsgType

from src.config import WebUIConfig
from sidecar.bridge import RuntimeBridge

STATIC_DIR = Path(__file__).parent / "static"
logger = logging.getLogger("adam.sidecar")
WS_AUTH_COOKIE = "adam_webui_ws_auth"


def ui_revision() -> str:
    return hashlib.sha256(b"".join((STATIC_DIR / name).read_bytes()
                                  for name in ("app.js", "style.css", "index.html"))).hexdigest()[:12]


def _is_loopback(host: str | None) -> bool:
    if not host:
        return False
    clean = host.strip().lower()
    if clean.startswith("::ffff:"):
        clean = clean[7:]
    if clean in ("localhost", "testclient"):
        return True
    try:
        return ipaddress.ip_address(clean).is_loopback
    except ValueError:
        return False


def _normalized_host(host: str | None) -> str | None:
    if not host:
        return None
    clean = host.strip().lower().rstrip(".")
    if clean == "localhost":
        return clean
    try:
        return ipaddress.ip_address(clean).compressed
    except ValueError:
        return None


@web.middleware
async def security_headers_middleware(request: web.Request, handler: Any) -> web.StreamResponse:
    response = await handler(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Cache-Control"] = "no-store"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "script-src 'self'; "
        "img-src 'self' data:; "
        "connect-src 'self' ws: wss:;"
    )
    return response


def create_auth_middleware(auth_token: str):
    @web.middleware
    async def auth_middleware(request: web.Request, handler: Any) -> web.StreamResponse:
        # If no auth token is configured, loopback access is allowed without auth
        if not auth_token:
            return await handler(request)

        # Allow static assets and root page to load login prompt/assets if needed
        # But protect API and WebSocket routes
        if not request.path.startswith("/api"):
            return await handler(request)

        # Allow unauthenticated discovery of auth requirement
        if request.path == "/api/auth/status":
            return await handler(request)

        # The WebSocket uses a scoped HttpOnly cookie; HTTP APIs use a bearer header.
        auth_header = request.headers.get("Authorization", "")
        token = ""
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
        else:
            token = request.cookies.get(WS_AUTH_COOKIE, "")

        if token != auth_token:
            return web.json_response(
                {
                    "error": "Unauthorized: valid authentication token required",
                    "auth_required": True,
                },
                status=401,
            )
        return await handler(request)

    return auth_middleware


def create_loopback_middleware(bound_host: str):
    is_bound_loopback = _is_loopback(bound_host)

    @web.middleware
    async def loopback_middleware(request: web.Request, handler: Any) -> web.StreamResponse:
        # If server is intended for loopback, reject remote origins
        if is_bound_loopback:
            remote = getattr(request, "_remote_override", None) or request.remote or ""
            if not _is_loopback(remote):
                return web.json_response(
                    {"error": "Forbidden: Adam WebUI binds strictly to loopback by default"},
                    status=403,
                )
        return await handler(request)

    return loopback_middleware


def create_same_origin_middleware():
    """Reject DNS-rebinding Host headers and cross-origin browser requests."""
    @web.middleware
    async def same_origin_middleware(request: web.Request, handler: Any) -> web.StreamResponse:
        try:
            target = urlsplit(f"//{request.host}")
            target_host = _normalized_host(target.hostname)
            target_port = target.port or (443 if request.scheme == "https" else 80)
            valid_target = (
                target.username is None
                and target.password is None
                and target_host is not None
                and (target_host == "localhost" or _is_loopback(target_host))
            )
        except ValueError:
            target_host = None
            target_port = None
            valid_target = False

        if not valid_target:
            return web.json_response({"error": "Forbidden: invalid local Host header"}, status=403)

        origin = request.headers.get("Origin")
        if origin:
            try:
                parsed = urlsplit(origin)
                origin_host = _normalized_host(parsed.hostname)
                origin_port = parsed.port or (443 if parsed.scheme == "https" else 80)
                valid_origin = (
                    parsed.scheme == request.scheme
                    and parsed.username is None
                    and parsed.password is None
                    and parsed.path in ("", "/")
                    and not parsed.query
                    and not parsed.fragment
                    and origin_host == target_host
                    and origin_port == target_port
                )
            except ValueError:
                valid_origin = False

            if not valid_origin:
                return web.json_response(
                    {"error": "Forbidden: WebUI accepts same-origin browser requests only"},
                    status=403,
                )
        return await handler(request)

    return same_origin_middleware


class SidecarServer:
    """Manages endpoints, websocket connections, and runtime bridge."""

    def __init__(self, config: WebUIConfig, bridge: RuntimeBridge) -> None:
        self.config = config
        self.bridge = bridge
        self.active_websockets: set[web.WebSocketResponse] = set()
        self.chat_tasks: set[asyncio.Task] = set()
        self.active_chat_request_id: str | None = None
        self.active_chat_ws: web.WebSocketResponse | None = None

    async def _deliver_chat(self, ws, user_text, request_id, allowed_tools=None):
        # Keep receiving heartbeats while the daemon owns the running turn.
        # A browser disconnect must not replay or cancel desktop actions.
        try:
            try:
                handle_message = self.bridge.handle_user_message
                try:
                    supports_request_id = "request_id" in inspect.signature(handle_message).parameters
                except (TypeError, ValueError):
                    supports_request_id = False
                kwargs = {}
                if allowed_tools is not None:
                    kwargs["allowed_tools"] = allowed_tools
                if supports_request_id:
                    kwargs["request_id"] = request_id
                result = await handle_message(user_text, **kwargs)
            except Exception:
                logger.exception("WebUI chat failed")
                result = {"status": "error", "error": "The assistant turn failed."}
            if not ws.closed:
                try:
                    await ws.send_json({
                        "type": "chat_response", "request_id": request_id,
                        "user_message": user_text, "result": result,
                    })
                except (ConnectionError, RuntimeError):
                    logger.debug("Chat finished after browser disconnected")
            # Conversation history is shared by the daemon across WebSocket
            # clients. Tell every connected UI to fetch the canonical snapshot
            # after this request has been committed, including clients that did
            # not originate the chat.
            await self.broadcast({"type": "history_changed"})
            try:
                await self.bridge.broadcast_state({"type": "status", **await self.bridge.get_status()})
            except Exception:
                logger.debug("Could not broadcast status after chat completion", exc_info=True)
        finally:
            if self.active_chat_request_id == request_id and self.active_chat_ws is ws:
                self.active_chat_request_id = None
                self.active_chat_ws = None

    async def shutdown(self, app):
        for ws in list(self.active_websockets):
            await ws.close()
        tasks = list(self.chat_tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def handle_index(self, request: web.Request) -> web.Response:
        index_path = STATIC_DIR / "index.html"
        revision, html = await asyncio.gather(
            asyncio.to_thread(ui_revision),
            asyncio.to_thread(index_path.read_text),
        )
        html = html.replace('/static/app.js', f'/static/app.js?v={revision}')
        html = html.replace('/static/style.css', f'/static/style.css?v={revision}')
        return web.Response(text=html, content_type="text/html")

    async def handle_status(self, request: web.Request) -> web.Response:
        status_data = await self.bridge.get_status()
        status_data["host"] = self.config.host
        status_data["port"] = self.config.port
        status_data["is_loopback"] = _is_loopback(self.config.host)
        status_data["auth_required"] = bool(self.config.auth_token)
        status_data["ui_revision"] = await asyncio.to_thread(ui_revision)
        return web.json_response(status_data)

    async def handle_auth_status(self, request: web.Request) -> web.Response:
        auth_header = request.headers.get("Authorization", "")
        token = ""
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
        else:
            token = request.cookies.get(WS_AUTH_COOKIE, "")

        required = bool(self.config.auth_token)
        valid = (not required) or (token == self.config.auth_token)
        return web.json_response({
            "auth_required": required,
            "authenticated": valid,
        })

    async def handle_auth_session(self, request: web.Request) -> web.Response:
        """Exchange a bearer header for a short-lived WebSocket-only cookie."""
        response = web.json_response({"ok": True, "auth_required": bool(self.config.auth_token)})
        if request.method == "POST" and self.config.auth_token:
            response.set_cookie(
                WS_AUTH_COOKIE,
                self.config.auth_token,
                max_age=3600,
                httponly=True,
                secure=request.secure,
                samesite="Strict",
                path="/api/ws",
            )
        elif request.method == "DELETE":
            response.del_cookie(WS_AUTH_COOKIE, path="/api/ws")
        return response

    async def handle_history(self, request: web.Request) -> web.Response:
        history = await self.bridge.get_history()
        return web.json_response({"messages": history})

    async def handle_chat(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body"}, status=400)

        message = str(body.get("message", "")).strip()
        if not message:
            return web.json_response({"error": "Message text is required"}, status=400)

        allowed_tools = body.get("allowed_tools")
        if "allowed_tools" in body and (
            not isinstance(allowed_tools, list)
            or any(not isinstance(name, str) or not name.strip() for name in allowed_tools)
            or len(set(allowed_tools)) != len(allowed_tools)
        ):
            return web.json_response({
                "error": "allowed_tools must be a list of unique non-empty tool names"
            }, status=400)

        if allowed_tools is None:
            result = await self.bridge.handle_user_message(message)
        else:
            result = await self.bridge.handle_user_message(message, allowed_tools=allowed_tools)
        status_code = 200 if result.get("status") in ("completed", "busy") else 500
        return web.json_response(result, status=status_code)

    async def handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=30.0)
        await ws.prepare(request)
        self.active_websockets.add(ws)
        chat_task = None

        # Send initial status snapshot upon connection
        try:
            status_data = await self.bridge.get_status()
            status_data["type"] = "status"
            status_data["host"] = self.config.host
            status_data["port"] = self.config.port
            status_data["is_loopback"] = _is_loopback(self.config.host)
            status_data["auth_required"] = bool(self.config.auth_token)
            status_data["ui_revision"] = await asyncio.to_thread(ui_revision)
            await ws.send_json(status_data)
        except Exception as exc:
            logger.warning(f"Error sending initial WS status: {exc}")

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except Exception:
                        await ws.send_json({"type": "error", "error": "Invalid JSON"})
                        continue

                    msg_type = data.get("type")
                    if msg_type == "ping":
                        await ws.send_json({"type": "pong"})
                    elif msg_type == "status":
                        status_data = await self.bridge.get_status()
                        status_data["type"] = "status"
                        await ws.send_json(status_data)
                    elif msg_type == "cancel_chat":
                        request_id = data.get("request_id")
                        if (
                            not isinstance(request_id, str)
                            or not request_id.strip()
                            or len(request_id) > 128
                        ):
                            await ws.send_json({
                                "type": "cancel_ack", "request_id": request_id,
                                "cancelled": False, "error": "A valid request_id is required.",
                            })
                            continue
                        accepted = False
                        if request_id == self.active_chat_request_id and ws is self.active_chat_ws:
                            cancel_message = getattr(self.bridge, "cancel_user_message", None)
                            if callable(cancel_message):
                                try:
                                    accepted = bool(await cancel_message(request_id))
                                except Exception:
                                    logger.exception("WebUI chat cancellation failed")
                        await ws.send_json({
                            "type": "cancel_ack", "request_id": request_id,
                            "cancelled": accepted,
                        })
                    elif msg_type == "chat":
                        user_text = str(data.get("message", "")).strip()
                        if not user_text:
                            await ws.send_json({"type": "error", "error": "Empty message"})
                            continue
                        allowed_tools = data.get("allowed_tools")
                        if "allowed_tools" in data and (
                            not isinstance(allowed_tools, list)
                            or any(not isinstance(name, str) or not name.strip() for name in allowed_tools)
                            or len(set(allowed_tools)) != len(allowed_tools)
                        ):
                            await ws.send_json({
                                "type": "error",
                                "error": "allowed_tools must be a list of unique non-empty tool names",
                            })
                            continue
                        request_id = data.get("request_id")
                        if (
                            not isinstance(request_id, str)
                            or not request_id.strip()
                            or len(request_id) > 128
                        ):
                            await ws.send_json({"type": "error", "error": "A valid request_id is required for chat."})
                            continue
                        if self.chat_tasks or self.active_chat_request_id is not None:
                            await ws.send_json({"type": "chat_response", "request_id": request_id,
                                                "user_message": user_text,
                                                "result": {"status": "busy", "error": "Adam is already working on a request."}})
                            continue
                        register_message = getattr(self.bridge, "register_user_message", None)
                        if callable(register_message) and not register_message(request_id):
                            await ws.send_json({"type": "chat_response", "request_id": request_id,
                                                "user_message": user_text,
                                                "result": {"status": "busy", "error": "Adam is already working on a request."}})
                            continue
                        self.active_chat_request_id = request_id
                        self.active_chat_ws = ws
                        chat_task = asyncio.create_task(
                            self._deliver_chat(ws, user_text, request_id, allowed_tools=allowed_tools)
                        )
                        self.chat_tasks.add(chat_task)
                        chat_task.add_done_callback(self.chat_tasks.discard)
                    else:
                        await ws.send_json({"type": "error", "error": f"Unknown message type '{msg_type}'"})
                elif msg.type == WSMsgType.ERROR:
                    logger.warning(f"WebSocket closed with exception {ws.exception()}")
        finally:
            self.active_websockets.discard(ws)

        return ws

    async def broadcast(self, payload: dict[str, Any]) -> None:
        if not self.active_websockets:
            return
        coros = []
        for ws in list(self.active_websockets):
            if not ws.closed:
                coros.append(ws.send_json(payload))
        if coros:
            await asyncio.gather(*coros, return_exceptions=True)


def create_app(config: WebUIConfig, bridge: RuntimeBridge) -> web.Application:
    """Create and configure the aiohttp WebUI application."""
    if not _is_loopback(config.host):
        raise ValueError(
            f"Insecure host '{config.host}': WebUI server must bind loopback only (127.0.0.1 or ::1) "
            "unless a future explicit secure remote design exists."
        )

    sidecar = SidecarServer(config, bridge)

    # Register sidecar broadcast as listener on bridge
    bridge._listeners.add(sidecar.broadcast)

    middlewares = [
        security_headers_middleware,
        create_loopback_middleware(config.host),
        create_same_origin_middleware(),
        create_auth_middleware(config.auth_token),
    ]

    app = web.Application(middlewares=middlewares)
    sidecar_server_key = web.AppKey("sidecar_server", SidecarServer)
    bridge_key = web.AppKey("bridge", RuntimeBridge)
    app[sidecar_server_key] = sidecar
    app[bridge_key] = bridge

    async def live_events(app):
        await bridge.start_live_events()
        try:
            yield
        finally:
            await bridge.stop_live_events()
            await bridge.unregister_listener(sidecar.broadcast)

    app.cleanup_ctx.append(live_events)
    app.on_shutdown.append(sidecar.shutdown)

    # Static assets
    app.router.add_static("/static", path=str(STATIC_DIR), name="static")

    # Routes
    app.router.add_get("/", sidecar.handle_index)
    app.router.add_get("/api/auth/status", sidecar.handle_auth_status)
    app.router.add_post("/api/auth/session", sidecar.handle_auth_session)
    app.router.add_delete("/api/auth/session", sidecar.handle_auth_session)
    app.router.add_get("/api/status", sidecar.handle_status)
    app.router.add_get("/api/history", sidecar.handle_history)
    app.router.add_post("/api/chat", sidecar.handle_chat)
    app.router.add_get("/api/ws", sidecar.handle_websocket)

    return app


async def start_sidecar(config: WebUIConfig, bridge: RuntimeBridge) -> web.AppRunner:
    """Start the sidecar web server."""
    app = create_app(config, bridge)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host=config.host, port=config.port)
    await site.start()
    return runner
