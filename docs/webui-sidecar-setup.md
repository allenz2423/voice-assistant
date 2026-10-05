# Adam WebUI Sidecar Setup & Architecture Guide

## Overview

The Adam WebUI Sidecar is an **opt-in**, lightweight browser interface for Adam. It provides real-time conversation history, two-way text interaction wired to Adam's ReAct execution engine, and honest runtime telemetry.

Per the Interaction UX principles defined in `docs/proposed-codebase-fixes.md`:
1. **Voice is the primary default:** Adam remains voice-activated and voice-first. The WebUI listener is **disabled by default**.
2. **Strict loopback binding:** The sidecar binds to a loopback address only. It rejects non-loopback Host headers and cross-origin browser requests to reduce unintended exposure, including DNS-rebinding attempts.
3. **No fabricated live support:** A standalone sidecar process does not spin up a duplicate secondary `AdamBrain` pretending to be connected to the live daemon. It runs in a truthful disconnected mode unless wired into the active daemon instance.
4. **Verbal confirmation preserved:** The WebUI never fabricates approvals or bypasses voice confirmation. If an action requires confirmation, Adam awaits spoken confirmation via the microphone.
5. **Truthful runtime capabilities:** Durable run checkpoints (P3 gate) and remote multi-channel approval APIs (P4 gate) are marked unavailable in the UI until safe backend runtime support is completed.

---

## Configuration

In `config.yaml` (or `config.yaml.example`), configure the `webui` section:

```yaml
webui:
  enabled: false       # Keep false by default; set to true to enable
  host: "127.0.0.1"    # Loopback address (strictly loopback only)
  port: 8765           # Default sidecar port
  auth_token: ""       # Optional Bearer token for authentication
```

### Options

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enabled` | boolean | `false` | Enables WebUI sidecar on daemon startup. |
| `host` | string | `"127.0.0.1"` | Host interface. Must be a loopback address (`127.0.0.1`, `::1`, or `localhost`). |
| `port` | integer | `8765` | Port number to listen on. |
| `auth_token` | string | `""` | Optional authentication token. When set, requests must supply a valid Bearer token. |

---

## Starting the WebUI

### 1. In-Daemon Live Mode (Recommended)

When WebUI is enabled, the sidecar is hosted in-process alongside the Adam daemon, directly wired into the active `AdamDaemon` instance via `DaemonBridge`.

**Option A: Command-line flag:**
```bash
python -m src.main --webui
```

**Option B: Via config file:**
Set `webui.enabled: true` in `config.yaml`, then run:
```bash
python -m src.main
```

When started, the daemon logs:
```
[WebUI] Sidecar listening on http://127.0.0.1:8765 (loopback only, opt-in).
```

Open `http://127.0.0.1:8765` in your desktop browser.

### 2. Standalone Process (Disconnected Inspection Mode)

If launched as a standalone process:
```bash
python -m sidecar
# or
python sidecar/main.py
```

The standalone sidecar serves the UI in disconnected mode (`DisconnectedBridge`). It truthfully displays `Disconnected (No Live Daemon)` and prevents sending messages, rather than creating a disconnected secondary brain with duplicate models.

---

## Security & Authentication

### Loopback-Only Policy
The sidecar server checks incoming requests using loopback middleware. Remote connections are rejected with `403 Forbidden`. If an insecure non-loopback host is configured without an explicit protected transport design, server startup raises an error.

### Token Authentication
If `webui.auth_token` is configured:
- Unauthenticated requests to `/api/status`, `/api/history`, and `/api/chat` return `401 Unauthorized`.
- The browser frontend prompts for the token via the **Auth Token Modal**.
- Tokens are stored locally in the browser's `localStorage` and sent with:
  - HTTP header: `Authorization: Bearer <token>`
  - An authenticated session exchange that sets a one-hour, HttpOnly, Strict-SameSite cookie scoped to `/api/ws`.
- The Auth badge in the UI displays `Auth: Active` when authenticated or `Auth: Required` when authentication is needed.

---

## Architecture & Request Serialization

```
┌────────────────────────────────────────────────────────┐
│                      Browser UI                        │
│            (Chat, Telemetry, Status Badges)            │
└──────────────────────────┬─────────────────────────────┘
                           │ HTTP / WebSocket (127.0.0.1)
                           ▼
┌────────────────────────────────────────────────────────┐
│                   SidecarServer                        │
│         (Security Headers, Loopback, Auth)             │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│                    DaemonBridge                        │
│    (Serializes WebUI turns; checks voice state)         │
└────────────┬─────────────────────────────┬─────────────┘
             │                             │
             ▼                             ▼
┌─────────────────────────┐   ┌──────────────────────────┐
│   PriorityAudioArbiter  │   │        AdamBrain         │
│ (State synchronization) │   │ (ReAct execution engine) │
└─────────────────────────┘   └──────────────────────────┘
```

1. **Serialized Requests:** WebUI messages acquire `self._lock` and verify that the arbiter is `IDLE_LISTENING`. If Adam is actively speaking or handling a voice turn, the WebUI returns a `busy` status. Accepted messages use the daemon's normal turn executor, including microphone interruption monitoring and turn cleanup.
2. **Verbal Confirmation Priority:** If the arbiter is `AWAITING_CONFIRMATION`, the WebUI refuses commands and instructs the user to answer via the microphone.
3. **State Broadcasts:** As Adam transitions between reasoning, tool execution, and standby, state changes are broadcast over WebSocket to update UI spinner and badges in real time.

---

## Current Runtime Limitations

Per `docs/proposed-codebase-fixes.md`:
- **Long-Running Tasks (P3 Gate):** Durable task checkpoints, checkpoint persistence, and remote pause/resume controls are not yet implemented in the runtime. The UI truthfully displays this feature as unavailable.
- **Remote Approvals (P4 Gate):** Multi-channel approval APIs (e.g. approving sensitive commands from browser or phone) are not yet implemented. Voice confirmation via microphone remains Adam's sole confirmation path.
