// Adam WebUI Sidecar Client Script

(function () {
  let ws = null;
  let reconnectTimer = null;
  let isSending = false;
  let historyLoaded = false;
  const liveTools = new Map();

  // Token management
  const STORAGE_KEY = "adam_webui_token";

  function getStoredToken() {
    return (localStorage.getItem(STORAGE_KEY) || "").trim();
  }

  function setStoredToken(token) {
    if (token) {
      localStorage.setItem(STORAGE_KEY, token.trim());
    } else {
      localStorage.removeItem(STORAGE_KEY);
    }
  }

  function getAuthHeaders() {
    const token = getStoredToken();
    const headers = { "Content-Type": "application/json" };
    if (token) {
      headers["Authorization"] = `Bearer ${token}`;
    }
    return headers;
  }

  // DOM Elements
  const connBadge = document.getElementById("connBadge");
  const connText = document.getElementById("connText");
  const stateBadge = document.getElementById("stateBadge");
  const stateText = document.getElementById("stateText");
  const hostText = document.getElementById("hostText");
  const authBadge = document.getElementById("authBadge");
  const authText = document.getElementById("authText");

  const chatMessages = document.getElementById("chatMessages");
  const chatForm = document.getElementById("chatForm");
  const chatInput = document.getElementById("chatInput");
  const sendBtn = document.getElementById("sendBtn");
  const statusBar = document.getElementById("statusBar");
  const statusMessage = document.getElementById("statusMessage");

  // Telemetry Elements
  const telemetryMode = document.getElementById("telemetryMode");
  const telemetryModel = document.getElementById("telemetryModel");
  const telemetryTools = document.getElementById("telemetryTools");

  // Auth Modal Elements
  const authModal = document.getElementById("authModal");
  const tokenInput = document.getElementById("tokenInput");
  const saveTokenBtn = document.getElementById("saveTokenBtn");
  const clearTokenBtn = document.getElementById("clearTokenBtn");
  const closeAuthModalBtn = document.getElementById("closeAuthModalBtn");
  const authModalStatus = document.getElementById("authModalStatus");

  // Format timestamp
  function formatTime(isoStr) {
    if (!isoStr) return new Date().toLocaleTimeString();
    try {
      const d = new Date(isoStr);
      return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    } catch {
      return new Date().toLocaleTimeString();
    }
  }

  // Append a message to the chat container
  function appendMessage(role, content, toolCalls, timestamp) {
    const row = document.createElement("div");
    row.className = `message-row message-${role}`;

    const header = document.createElement("div");
    header.className = "message-header";

    const author = document.createElement("span");
    author.className = "message-author";
    author.textContent = role === "user" ? "You" : "Adam";

    const time = document.createElement("span");
    time.className = "message-time";
    time.textContent = formatTime(timestamp);

    header.appendChild(author);
    header.appendChild(time);
    row.appendChild(header);

    if (content) {
      const body = document.createElement("div");
      body.className = "message-body";
      body.textContent = content;
      row.appendChild(body);
    }

    if (toolCalls && toolCalls.length > 0) {
      const toolContainer = document.createElement("div");
      toolContainer.className = "tool-calls-container";
      toolCalls.forEach((tc) => {
        const badge = document.createElement("span");
        badge.className = "tool-badge";
        badge.textContent = `⚡ ${tc}`;
        toolContainer.appendChild(badge);
      });
      row.appendChild(toolContainer);
    }

    chatMessages.appendChild(row);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return row;
  }

  function showToolActivity(data) {
    let row = liveTools.get(data.span_id);
    if (!row) {
      row = appendMessage("assistant", "", []);
      const badge = document.createElement("span");
      badge.className = "tool-badge";
      row.appendChild(badge);
      liveTools.set(data.span_id, row);
      if (liveTools.size > 256) liveTools.delete(liveTools.keys().next().value);
    }
    const labels = { returned: "Returned", ok: "Returned", failed: "Failed",
      error: "Failed", invalid_input: "Rejected", timed_out: "Timed out",
      cancelled: "Cancelled", partial: "Partial", uncertain: "Unconfirmed" };
    const running = data.phase === "started";
    row.dataset.toolRunning = running ? "true" : "false";
    row.dataset.toolName = data.tool;
    const outcome = running ? "Running" : (labels[data.outcome] || data.outcome || "Finished");
    const duration = !running && data.duration_ms != null ? ` (${data.duration_ms} ms)` : "";
    row.querySelector(".tool-badge").textContent = `⚡ ${data.tool} · ${outcome}${duration}`;
    if (isSending) statusMessage.textContent = running
      ? `Adam is using ${data.tool}…` : "Adam is processing the result…";
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  // Set connection UI state
  function setConnectionStatus(connected, mode, extra) {
    if (connected && mode === "daemon") {
      connBadge.className = "badge badge-online";
      connText.textContent = "Connected (Daemon)";
    } else if (connected) {
      connBadge.className = "badge badge-online";
      connText.textContent = "Connected (Loopback)";
    } else {
      connBadge.className = "badge badge-offline";
      connText.textContent = extra || "Disconnected";
    }
  }

  // Set runtime state UI
  function setRuntimeState(state) {
    const s = state || "IDLE";
    stateText.textContent = s;
    if (s === "PROCESSING_REACT") {
      stateBadge.style.color = "var(--accent-amber)";
      stateBadge.style.borderColor = "rgba(245, 158, 11, 0.4)";
      statusBar.classList.remove("hidden");
      statusMessage.textContent = "Adam is reasoning and executing tools...";
    } else if (s === "AWAITING_CONFIRMATION") {
      stateBadge.style.color = "var(--accent-red)";
      stateBadge.style.borderColor = "rgba(239, 68, 68, 0.4)";
      statusBar.classList.remove("hidden");
      statusMessage.textContent = "Adam is awaiting verbal confirmation (via mic)...";
    } else if (s === "ASSISTANT_SPEAKING") {
      stateBadge.style.color = "var(--accent-blue)";
      stateBadge.style.borderColor = "rgba(59, 130, 246, 0.4)";
      statusBar.classList.remove("hidden");
      statusMessage.textContent = "Adam is speaking...";
    } else if (s === "DISCONNECTED" || s === "OFFLINE") {
      stateBadge.style.color = "var(--text-muted)";
      stateBadge.style.borderColor = "var(--border-color)";
      if (!isSending) {
        statusBar.classList.add("hidden");
      }
    } else {
      stateBadge.style.color = "var(--accent-cyan)";
      stateBadge.style.borderColor = "rgba(6, 182, 212, 0.4)";
      if (!isSending) {
        statusBar.classList.add("hidden");
      }
    }
  }

  // Update telemetry display
  function updateTelemetry(data) {
    if (data.mode) telemetryMode.textContent = data.mode;
    if (data.model) telemetryModel.textContent = data.model;
    if (data.tools_count !== undefined) telemetryTools.textContent = `${data.tools_count} tools`;
    if (data.host) hostText.textContent = `${data.host}:${data.port || 8765}`;
    if (data.system_state) setRuntimeState(data.system_state);
  }

  // Update auth status display
  async function checkAuthStatus() {
    try {
      const res = await fetch("/api/auth/status", { headers: getAuthHeaders() });
      if (res.ok) {
        const data = await res.json();
        if (!data.auth_required) {
          authBadge.className = "badge badge-auth badge-auth-off";
          authText.textContent = "Auth: Off (Loopback)";
        } else if (data.authenticated) {
          authBadge.className = "badge badge-auth badge-auth-valid";
          authText.textContent = "Auth: Active";
        } else {
          authBadge.className = "badge badge-auth badge-auth-required";
          authText.textContent = "Auth: Required";
        }
        return data;
      }
    } catch (err) {
      console.warn("Auth status check failed:", err);
    }
    return null;
  }

  // Load conversation history from API
  async function loadHistory() {
    if (historyLoaded) return;
    try {
      const res = await fetch("/api/history", { headers: getAuthHeaders() });
      if (res.status === 401) {
        handleAuthRequired();
        return;
      }
      if (!res.ok) return;
      const data = await res.json();
      if (Array.isArray(data.messages) && data.messages.length > 0) {
        data.messages.forEach((m) => {
          appendMessage(m.role, m.content, m.tool_calls, m.timestamp);
        });
      }
      historyLoaded = true;
    } catch (err) {
      console.warn("Could not load history:", err);
    }
  }

  // Fetch status snapshot via HTTP
  async function fetchStatus() {
    try {
      const res = await fetch("/api/status", { headers: getAuthHeaders() });
      if (res.status === 401) {
        handleAuthRequired();
        setConnectionStatus(false, "offline", "Auth Required");
        return;
      }
      if (res.ok) {
        const data = await res.json();
        setConnectionStatus(data.connected, data.mode, data.connected ? null : "Disconnected (No Live Daemon)");
        updateTelemetry(data);
      } else {
        setConnectionStatus(false, "offline", "Status Unavailable");
      }
    } catch {
      setConnectionStatus(false, "offline", "Offline");
    }
  }

  function handleAuthRequired() {
    authBadge.className = "badge badge-auth badge-auth-required";
    authText.textContent = "Auth: Required";
    if (!getStoredToken()) {
      openAuthModal("Authentication token required to connect to Adam WebUI.");
    }
  }

  // Modal helpers
  function openAuthModal(msg) {
    tokenInput.value = getStoredToken();
    if (msg) {
      authModalStatus.textContent = msg;
      authModalStatus.className = "auth-modal-status status-error";
      authModalStatus.classList.remove("hidden");
    } else {
      authModalStatus.classList.add("hidden");
    }
    authModal.classList.remove("hidden");
    tokenInput.focus();
  }

  function closeAuthModal() {
    authModal.classList.add("hidden");
  }

  // Initialize WebSocket connection
  async function connectWebSocket() {
    if (ws) {
      try { ws.close(); } catch {}
      ws = null;
    }

    const token = getStoredToken();
    if (token) {
      try {
        const response = await fetch("/api/auth/session", {
          method: "POST",
          headers: getAuthHeaders(),
        });
        if (!response.ok) {
          handleAuthRequired();
          setConnectionStatus(false, "offline", "Auth Required");
          return;
        }
      } catch {
        setConnectionStatus(false, "offline", "Connection Failed");
        return;
      }
    }
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/api/ws`;

    try {
      ws = new WebSocket(wsUrl);
    } catch (err) {
      setConnectionStatus(false, "offline", "Connection Failed");
      return;
    }

    ws.onopen = () => {
      if (reconnectTimer) {
        clearInterval(reconnectTimer);
        reconnectTimer = null;
      }
      fetchStatus();
      loadHistory();
      checkAuthStatus();
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === "status") {
          updateTelemetry(data);
          setConnectionStatus(data.connected, data.mode, data.connected ? null : "Disconnected (No Live Daemon)");
        } else if (data.type === "state") {
          setRuntimeState(data.system_state);
        } else if (data.type === "tool_activity") {
          showToolActivity(data);
        } else if (data.type === "chat_response") {
          const res = data.result || {};
          if (res.status === "completed") {
            appendMessage("assistant", res.response, res.tool_calls);
          } else if (res.status === "busy") {
            appendMessage("assistant", `⚠️ ${res.error || "Adam is currently busy."}`);
          } else {
            appendMessage("assistant", `❌ Error: ${res.error || "Unknown execution error"}`);
          }
          isSending = false;
          sendBtn.disabled = false;
          statusBar.classList.add("hidden");
        } else if (data.type === "error") {
          appendMessage("assistant", `❌ Error: ${data.error || "WebSocket error"}`);
          isSending = false;
          sendBtn.disabled = false;
          statusBar.classList.add("hidden");
        }
      } catch (err) {
        console.error("WS message parse error:", err);
      }
    };

    ws.onclose = (event) => {
      for (const row of liveTools.values()) {
        if (row.dataset.toolRunning === "true") {
          row.querySelector(".tool-badge").textContent =
            `⚡ ${row.dataset.toolName} · Connection lost (outcome unknown)`;
          row.dataset.toolRunning = "false";
        }
      }
      if (event.code === 4001 || event.code === 4401) {
        handleAuthRequired();
        setConnectionStatus(false, "offline", "Auth Required");
      } else {
        setConnectionStatus(false, "offline", "Disconnected (Reconnecting...)");
      }

      if (isSending) {
        isSending = false;
        sendBtn.disabled = false;
        statusBar.classList.add("hidden");
        appendMessage("assistant", "❌ Disconnected while awaiting response.");
      }

      if (!reconnectTimer) {
        reconnectTimer = setInterval(connectWebSocket, 3000);
      }
    };

    ws.onerror = () => {
      setConnectionStatus(false, "offline", "Connection Error");
      if (isSending) {
        isSending = false;
        sendBtn.disabled = false;
        statusBar.classList.add("hidden");
      }
    };
  }

  // Send a user chat message
  async function sendMessage(text) {
    if (!text || isSending) return;

    isSending = true;
    sendBtn.disabled = true;
    chatInput.value = "";
    chatInput.style.height = "auto";

    appendMessage("user", text);
    statusBar.classList.remove("hidden");
    statusMessage.textContent = "Adam is processing request...";

    // If WebSocket is open, send via WS; otherwise HTTP fallback
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "chat", message: text }));
    } else {
      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: getAuthHeaders(),
          body: JSON.stringify({ message: text }),
        });
        if (res.status === 401) {
          handleAuthRequired();
          appendMessage("assistant", "❌ Authentication required: please configure your authentication token.");
          return;
        }
        const data = await res.json();
        if (data.status === "completed") {
          appendMessage("assistant", data.response, data.tool_calls);
        } else if (data.status === "busy") {
          appendMessage("assistant", `⚠️ ${data.error || "Adam is currently busy."}`);
        } else {
          appendMessage("assistant", `❌ Error: ${data.error || "Failed to execute"}`);
        }
      } catch (err) {
        appendMessage("assistant", `❌ Network error: ${err.message}`);
      } finally {
        isSending = false;
        sendBtn.disabled = false;
        statusBar.classList.add("hidden");
      }
    }
  }

  // Event Listeners
  chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = chatInput.value.trim();
    if (text) sendMessage(text);
  });

  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      const text = chatInput.value.trim();
      if (text) sendMessage(text);
    }
  });

  // Prompt chips
  document.querySelectorAll(".prompt-chip").forEach((btn) => {
    btn.addEventListener("click", () => {
      const prompt = btn.getAttribute("data-prompt");
      if (prompt) sendMessage(prompt);
    });
  });

  // Auto-resize textarea
  chatInput.addEventListener("input", () => {
    chatInput.style.height = "auto";
    chatInput.style.height = `${Math.min(chatInput.scrollHeight, 120)}px`;
  });

  // Auth modal handlers
  authBadge.addEventListener("click", () => {
    openAuthModal();
  });

  closeAuthModalBtn.addEventListener("click", () => {
    closeAuthModal();
  });

  authModal.addEventListener("click", (e) => {
    if (e.target === authModal) closeAuthModal();
  });

  saveTokenBtn.addEventListener("click", async () => {
    const token = tokenInput.value.trim();
    setStoredToken(token);
    authModalStatus.textContent = "Connecting with token...";
    authModalStatus.className = "auth-modal-status status-success";
    authModalStatus.classList.remove("hidden");

    const authResult = await checkAuthStatus();
    if (authResult && authResult.auth_required && !authResult.authenticated) {
      authModalStatus.textContent = "Invalid token. Please check and try again.";
      authModalStatus.className = "auth-modal-status status-error";
      return;
    }

    closeAuthModal();
    historyLoaded = false;
    connectWebSocket();
  });

  clearTokenBtn.addEventListener("click", async () => {
    const token = getStoredToken();
    if (token) {
      try {
        await fetch("/api/auth/session", {
          method: "DELETE",
          headers: getAuthHeaders(),
        });
      } catch {}
    }
    setStoredToken("");
    tokenInput.value = "";
    authModalStatus.textContent = "Token cleared.";
    authModalStatus.className = "auth-modal-status status-success";
    authModalStatus.classList.remove("hidden");
    checkAuthStatus();
    historyLoaded = false;
    connectWebSocket();
  });

  // Initial startup
  checkAuthStatus();
  fetchStatus();
  loadHistory();
  connectWebSocket();
})();
