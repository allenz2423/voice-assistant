# Adam WebUI Sidecar Setup & Architecture Guide

## Overview

The Adam WebUI Sidecar is an **opt-in**, lightweight browser dashboard for Adam. It provides real-time conversation history, two-way text interaction wired to Adam's ReAct execution engine, live tool execution telemetry, and runtime status monitoring.

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

## Interface Layout & User Experience

The dashboard interface uses a desktop-first, KDE Plasma Breeze Dark-inspired aesthetic with full responsive adaptability, accessible keyboard workflows, and real-time execution feedback.

### 1. Desktop Two-Panel Layout
On viewports wider than 860px, the dashboard displays a unified side-by-side workstation layout:
- **Primary Panel (Left):** Conversation history log, active turn status indicator, prompt suggestion chips, and the expandable message composer.
- **System Sidebar (Right):** Live tool execution telemetry, runtime configuration snapshot (mode, model routing, available tool count, loopback binding), and explicit callouts for currently unavailable runtime capabilities (long-running task checkpoints and remote approvals).

### 2. Mobile Responsive Layout & Tabs
On small viewports (≤ 860px), the dashboard automatically switches to a single-column layout with an accessible top tab bar:
- **Conversation Tab:** Full-width view of conversation history, quick prompts, turn status, and pinned message composer.
- **System & Tools Tab:** Full-width view of live tool activity, runtime telemetry, and runtime limitation cards.
- **Accessible Tab Semantics:** Built with WAI-ARIA `role="tablist"`, `role="tab"`, and `role="tabpanel"` semantics. Supports keyboard navigation with `ArrowLeft`, `ArrowRight`, `Home`, and `End` keys to switch tabs with automatic focus management. Resizing between mobile and desktop automatically synchronizes panel visibility.

### 3. Message Composer & Keyboard Controls
- **Send Message:** Press **`Enter`** (without Shift) or activate the **Send** button to submit the message to Adam's daemon turn executor.
- **Newline Entry:** Press **`Shift + Enter`** to insert a line break into the message textarea without submitting.
- **Auto-Expanding Textarea:** The composer automatically grows to fit multi-line inputs up to 140px high before scrolling.
- **Quick Prompts:** Interactive prompt chips allow one-click submission of common queries (e.g., system status, time, active desktop windows).
- **In-Flight Disconnect Handling:** If disconnected when submitting a turn, the message is safely queued in the client while attempting automatic reconnection before reporting failure.

### 4. Authentication Dialog
- **Modal Trigger:** The header `Auth` badge serves as an accessible button indicating the current state (`Auth: Active`, `Auth: Required`, or `Auth: Off (Loopback)`). Activating it via click or keyboard opens the configuration dialog.
- **Keyboard Focus Trap:** While the modal is open, keyboard focus is trapped within the dialog's interactive elements (`Tab` and `Shift + Tab` cycle focus without leaking to background elements).
- **Escape Dismissal:** Pressing **`Escape`** closes the dialog and restores focus to the trigger button.
- **Token Storage:** Tokens are stored locally in the browser's `localStorage`. The frontend exchanges the Bearer token via loopback HTTP `POST /api/auth/session` to set a one-hour, HttpOnly, Strict-SameSite cookie scoped to `/api/ws`.

### 5. Live Status Badges & Tool Feed
- **Header Status Cluster:** Live indicator badges show connection health (`Connected (Daemon)`, `Connected (Loopback)`, or `Disconnected`), runtime state (`IDLE`, `PROCESSING_REACT`, `AWAITING_CONFIRMATION`, `ASSISTANT_SPEAKING`), authentication state, voice priority, and loopback binding.
- **Activity Status Bar:** Displays animated spinner and contextual descriptions during turns, including model rate-limit backoff retries (`429`), tool recovery attempts, and model completion recoveries.
- **Live Tool Stream:** Displays tool calls in real time as they start and complete. Each entry reports tool name, active execution status, outcome (`Returned`, `Failed`, `Rejected`, `Timed out`, `Cancelled`, `Partial`, or `Unconfirmed`), and execution duration in milliseconds.

### 6. Voice Confirmation Limit
- **Microphone Exclusivity:** The WebUI cannot approve actions or bypass confirmation requirements. When Adam plans a sensitive tool execution requiring user confirmation, the system transitions to `AWAITING_CONFIRMATION`.
- **Turn Refusal:** While awaiting confirmation, the WebUI displays an urgent status warning and rejects incoming text turns with a `busy` status, directing the user to speak their confirmation through the microphone.

---

## Security Policies & Route Contract

### Loopback-Only Policy
The sidecar server checks incoming requests using loopback middleware. Remote connections are rejected with `403 Forbidden`. If an insecure non-loopback host is configured without an explicit protected transport design, server startup raises an error. It strictly rejects DNS-rebinding Host headers and cross-origin requests.

### Token Authentication
If `webui.auth_token` is configured:
- Unauthenticated requests to `/api/status`, `/api/history`, and `/api/chat` return `401 Unauthorized`.
- The discovery route `/api/auth/status` remains accessible unauthenticated to report whether token authentication is required.
- The browser frontend prompts for the token via the **Auth Token Modal**.
- Tokens are stored locally in the browser's `localStorage` and sent with:
  - HTTP request header: `Authorization: Bearer <token>`
  - A loopback HTTP session exchange (`POST /api/auth/session`) that sets a one-hour, HttpOnly, Strict-SameSite cookie scoped to `/api/ws`.
- The Auth badge in the UI displays `Auth: Active` when authenticated or `Auth: Required` when authentication is needed.

### Existing HTTP & WebSocket Routes
The WebUI operates entirely within existing endpoints; **no new routes, external network assets, remote fonts, or deployment requirements are introduced**:

| Route | Method | Description |
|---|---|---|
| `/` | `GET` | Serves dashboard HTML with cache-busted, SHA-256 versioned static assets. |
| `/api/status` | `GET` | Snapshot of runtime status, mode, tool count, host binding, and feature availability. |
| `/api/auth/status` | `GET` | Checks if token authentication is required and whether the current session is authenticated. |
| `/api/auth/session` | `POST` / `DELETE` | Exchanges a Bearer token for a one-hour HttpOnly, Strict-SameSite cookie scoped to `/api/ws`, or clears it. |
| `/api/history` | `GET` | Active session conversation history (system desktop headers stripped from display). |
| `/api/chat` | `POST` | Direct HTTP endpoint for conversational turn execution (requires Bearer token when auth is configured). |
| `/api/ws` | `GET` (WebSocket) | Bidirectional WebSocket stream for turns, heartbeats, runtime state, and live tool telemetry. |

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
4. **Live Tool Activity:** Both voice and browser turns stream tool names and execution status over WebSocket as calls start and finish. The UI shows Running, Returned, Failed, Rejected, Timed out, or Cancelled with elapsed time. Returned describes tool execution, not verification of the user's overall goal. Disconnecting marks unfinished calls as outcome unknown. This in-memory feed works with disk telemetry disabled, omits arguments and raw results, and unregisters its bounded event queue when the server shuts down. Live events are not replayed after reconnecting.

---

## Current Runtime Limitations

Per `docs/proposed-codebase-fixes.md`:
- **Long-Running Tasks (P3 Gate):** Durable task checkpoints, checkpoint persistence, and remote pause/resume controls are not yet implemented in the runtime. The UI truthfully displays this feature as unavailable.
- **Remote Approvals (P4 Gate):** Multi-channel approval APIs (e.g. approving sensitive commands from browser or phone) are not yet implemented. Voice confirmation via microphone remains Adam's sole confirmation path.
