// Adam WebUI Sidecar Client Script

(function () {
  let ws = null;
  let reconnectTimer = null;
  let wsConnecting = false;
  let wsGeneration = 0;
  let pendingChatMessage = null;
  let pendingChatTimeout = null;
  let isSending = false;
  let historyLoaded = false;
  let lastFocusedElement = null;
  const liveTools = new Map();
  const loadedRevision = new URL(document.currentScript.src).searchParams.get("v");

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

  // DOM Elements - Badges & Navigation
  const connBadge = document.getElementById("connBadge");
  const connText = document.getElementById("connText");
  const stateBadge = document.getElementById("stateBadge");
  const stateText = document.getElementById("stateText");
  const hostText = document.getElementById("hostText");
  const authBadge = document.getElementById("authBadge");
  const authText = document.getElementById("authText");

  // Layout & Sections
  const mainLayout = document.getElementById("mainLayout");
  const chatSection = document.getElementById("chatSection");
  const sidebarSection = document.getElementById("sidebarSection");
  const mobileTabChat = document.getElementById("mobileTabChat");
  const mobileTabSidebar = document.getElementById("mobileTabSidebar");

  // Chat Elements
  const chatMessages = document.getElementById("chatMessages");
  const chatForm = document.getElementById("chatForm");
  const chatInput = document.getElementById("chatInput");
  const sendBtn = document.getElementById("sendBtn");
  const statusBar = document.getElementById("statusBar");
  const statusMessage = document.getElementById("statusMessage");

  // Telemetry & Tool Activity Elements
  const telemetryMode = document.getElementById("telemetryMode");
  const telemetryModel = document.getElementById("telemetryModel");
  const telemetryTools = document.getElementById("telemetryTools");
  const toolActivity = document.getElementById("toolActivity");
  const toolActivityEmpty = document.getElementById("toolActivityEmpty");

  // Auth Modal Elements
  const authModal = document.getElementById("authModal");
  const tokenInput = document.getElementById("tokenInput");
  const saveTokenBtn = document.getElementById("saveTokenBtn");
  const clearTokenBtn = document.getElementById("clearTokenBtn");
  const closeAuthModalBtn = document.getElementById("closeAuthModalBtn");
  const authModalStatus = document.getElementById("authModalStatus");

  // Format timestamp helper
  function formatTime(isoStr) {
    if (!isoStr) return new Date().toLocaleTimeString();
    try {
      const d = new Date(isoStr);
      return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
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

  // Show live tool execution events
  function showToolActivity(data) {
    let row = liveTools.get(data.span_id);
    if (!row) {
      if (toolActivityEmpty) toolActivityEmpty.hidden = true;
      row = document.createElement("div");
      row.className = "tool-activity-item";
      const badge = document.createElement("span");
      badge.className = "tool-badge";
      row.appendChild(badge);
      toolActivity.appendChild(row);
      liveTools.set(data.span_id, row);
      if (liveTools.size > 40) {
        const oldest = liveTools.keys().next().value;
        const oldRow = liveTools.get(oldest);
        if (oldRow) oldRow.remove();
        liveTools.delete(oldest);
      }
    }
    const labels = {
      returned: "Returned",
      ok: "Returned",
      failed: "Failed",
      error: "Failed",
      invalid_input: "Rejected",
      timed_out: "Timed out",
      cancelled: "Cancelled",
      partial: "Partial",
      uncertain: "Unconfirmed",
    };
    const running = data.phase === "started";
    row.dataset.toolRunning = running ? "true" : "false";
    row.dataset.toolName = data.tool;
    const outcome = running ? "Running" : (labels[data.outcome] || data.outcome || "Finished");
    const duration = !running && data.duration_ms != null ? ` (${data.duration_ms} ms)` : "";
    const badgeEl = row.querySelector(".tool-badge");
    if (badgeEl) {
      badgeEl.textContent = `⚡ ${data.tool} · ${outcome}${duration}`;
    }
    if (isSending) {
      statusMessage.textContent = running
        ? `Adam is using ${data.tool}…`
        : "Adam is processing the result…";
    }
    toolActivity.scrollTop = toolActivity.scrollHeight;
  }

  // Show live task retry / recovery progress
  function showTaskProgress(data) {
    if (!isSending) return;
    const count = Number.isInteger(data.attempt) && Number.isInteger(data.max_attempts)
      ? ` (${data.attempt}/${data.max_attempts})`
      : "";
    if (data.event === "llm.retrying") {
      const reason = data.reason === "429" ? "rate limited the request" : "request failed";
      statusMessage.textContent = `Model ${reason}; retrying${count}…`;
    } else if (data.event === "brain.tool_recovery") {
      statusMessage.textContent = `A tool failed; asking Adam to recover safely${count}…`;
    } else if (data.event === "brain.empty_completion_recovery") {
      statusMessage.textContent = `Adam received no usable model response; retrying${count}…`;
    } else if (data.event === "brain.capability_recovery") {
      statusMessage.textContent = `Adam is re-checking available tools${count}…`;
    }
  }

  // Set connection UI badge state
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

  // Set runtime state UI badge & status bar
  function setRuntimeState(state) {
    const s = state || "IDLE";
    stateText.textContent = s;
    if (s === "PROCESSING_REACT") {
      stateBadge.className = "badge badge-state state-processing";
      statusBar.classList.remove("hidden");
      statusMessage.textContent = "Adam is reasoning and executing tools...";
    } else if (s === "AWAITING_CONFIRMATION") {
      stateBadge.className = "badge badge-state state-confirm";
      statusBar.classList.remove("hidden");
      statusMessage.textContent = "Adam is awaiting verbal confirmation (via mic)...";
    } else if (s === "ASSISTANT_SPEAKING") {
      stateBadge.className = "badge badge-state state-speaking";
      statusBar.classList.remove("hidden");
      statusMessage.textContent = "Adam is speaking...";
    } else if (s === "DISCONNECTED" || s === "OFFLINE") {
      stateBadge.className = "badge badge-state state-offline";
      if (!isSending) {
        statusBar.classList.add("hidden");
      }
    } else {
      stateBadge.className = "badge badge-state state-idle";
      if (!isSending) {
        statusBar.classList.add("hidden");
      }
    }
  }

  // Update telemetry display
  function updateTelemetry(data) {
    if (loadedRevision && data.ui_revision && loadedRevision !== data.ui_revision) {
      // Preserve a submitted turn; reload once it finishes to load the new UI.
      if (!isSending && !chatInput.value.trim()) {
        window.location.reload();
        return;
      }
    }
    if (data.mode && telemetryMode) telemetryMode.textContent = data.mode;
    if (data.model && telemetryModel) {
      telemetryModel.textContent = data.model;
      telemetryModel.title = data.tool_free_model
        ? `Short generic chat route: ${data.tool_free_model}`
        : "";
    }
    if (data.tools_count !== undefined && telemetryTools) {
      telemetryTools.textContent = `${data.tools_count} tools`;
    }
    if (data.host && hostText) {
      hostText.textContent = `${data.host}:${data.port || 8765}`;
    }
    if (data.system_state) {
      setRuntimeState(data.system_state);
    }
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

  // Focus trap & keyboard management for modal dialog
  function getModalFocusableElements() {
    return Array.from(
      authModal.querySelectorAll(
        'button:not([disabled]), input:not([disabled]), [tabindex]:not([tabindex="-1"])'
      )
    );
  }

  function handleModalKeyDown(e) {
    if (authModal.classList.contains("hidden")) return;

    if (e.key === "Escape") {
      e.preventDefault();
      closeAuthModal();
      return;
    }

    if (e.key === "Tab") {
      const focusables = getModalFocusableElements();
      if (focusables.length === 0) {
        e.preventDefault();
        return;
      }
      const first = focusables[0];
      const last = focusables[focusables.length - 1];

      if (e.shiftKey) {
        if (document.activeElement === first || !authModal.contains(document.activeElement)) {
          e.preventDefault();
          last.focus();
        }
      } else {
        if (document.activeElement === last || !authModal.contains(document.activeElement)) {
          e.preventDefault();
          first.focus();
        }
      }
    }
  }

  function openAuthModal(msg) {
    lastFocusedElement = document.activeElement;
    tokenInput.value = getStoredToken();
    if (msg) {
      authModalStatus.textContent = msg;
      authModalStatus.className = "auth-modal-status status-error";
      authModalStatus.classList.remove("hidden");
    } else {
      authModalStatus.classList.add("hidden");
    }
    authBadge.setAttribute("aria-expanded", "true");
    authModal.classList.remove("hidden");
    document.addEventListener("keydown", handleModalKeyDown);
    tokenInput.focus();
  }

  function closeAuthModal() {
    authModal.classList.add("hidden");
    authBadge.setAttribute("aria-expanded", "false");
    document.removeEventListener("keydown", handleModalKeyDown);
    if (lastFocusedElement && typeof lastFocusedElement.focus === "function") {
      lastFocusedElement.focus();
    }
  }

  // Initialize WebSocket connection
  async function connectWebSocket(force = false) {
    if (!force && ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
    if (!force && wsConnecting) return;
    const generation = ++wsGeneration;
    wsConnecting = true;

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
        if (generation !== wsGeneration) return;
        if (!response.ok) {
          wsConnecting = false;
          handleAuthRequired();
          setConnectionStatus(false, "offline", "Auth Required");
          return;
        }
      } catch {
        if (generation !== wsGeneration) return;
        wsConnecting = false;
        setConnectionStatus(false, "offline", "Connection Failed");
        return;
      }
    }
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/api/ws`;

    try {
      ws = new WebSocket(wsUrl);
    } catch (err) {
      if (generation !== wsGeneration) return;
      wsConnecting = false;
      setConnectionStatus(false, "offline", "Connection Failed");
      return;
    }
    const socket = ws;

    socket.onopen = () => {
      if (ws !== socket || generation !== wsGeneration) return;
      wsConnecting = false;
      if (reconnectTimer) {
        clearInterval(reconnectTimer);
        reconnectTimer = null;
      }
      fetchStatus();
      loadHistory();
      checkAuthStatus();
      if (pendingChatMessage !== null) {
        const message = pendingChatMessage;
        pendingChatMessage = null;
        if (pendingChatTimeout) {
          clearTimeout(pendingChatTimeout);
          pendingChatTimeout = null;
        }
        socket.send(JSON.stringify({ type: "chat", message }));
        statusMessage.textContent = "Adam is processing request...";
      }
    };

    socket.onmessage = (event) => {
      if (ws !== socket || generation !== wsGeneration) return;
      try {
        const data = JSON.parse(event.data);
        if (data.type === "status") {
          updateTelemetry(data);
          setConnectionStatus(data.connected, data.mode, data.connected ? null : "Disconnected (No Live Daemon)");
        } else if (data.type === "state") {
          setRuntimeState(data.system_state);
        } else if (data.type === "tool_activity") {
          showToolActivity(data);
        } else if (data.type === "task_progress") {
          showTaskProgress(data);
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

    socket.onclose = (event) => {
      if (ws !== socket || generation !== wsGeneration) return;
      wsConnecting = false;
      for (const row of liveTools.values()) {
        if (row.dataset.toolRunning === "true") {
          const badgeEl = row.querySelector(".tool-badge");
          if (badgeEl) {
            badgeEl.textContent = `⚡ ${row.dataset.toolName} · Connection lost (outcome unknown)`;
          }
          row.dataset.toolRunning = "false";
        }
      }
      if (event.code === 4001 || event.code === 4401) {
        handleAuthRequired();
        setConnectionStatus(false, "offline", "Auth Required");
      } else {
        setConnectionStatus(false, "offline", "Disconnected (Reconnecting...)");
      }

      if (isSending && pendingChatMessage === null) {
        isSending = false;
        sendBtn.disabled = false;
        statusBar.classList.add("hidden");
        appendMessage("assistant", "❌ Disconnected while awaiting response.");
      }

      if (!reconnectTimer) {
        reconnectTimer = setInterval(() => connectWebSocket(), 3000);
      }
    };

    socket.onerror = () => {
      if (ws !== socket || generation !== wsGeneration) return;
      wsConnecting = false;
      setConnectionStatus(false, "offline", "Connection Error");
      if (isSending && pendingChatMessage === null) {
        isSending = false;
        sendBtn.disabled = false;
        statusBar.classList.add("hidden");
      }
      try { socket.close(); } catch {}
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

    // Chat must use this socket so tool starts, finishes, and recovery attempts
    // can reach the page while Adam is working. Queue until reconnect completes.
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "chat", message: text }));
    } else {
      pendingChatMessage = text;
      statusMessage.textContent = "Connecting to Adam's live activity stream…";
      if (pendingChatTimeout) clearTimeout(pendingChatTimeout);
      pendingChatTimeout = setTimeout(() => {
        if (pendingChatMessage !== text) return;
        pendingChatMessage = null;
        pendingChatTimeout = null;
        isSending = false;
        sendBtn.disabled = false;
        statusBar.classList.add("hidden");
        appendMessage("assistant", "❌ Could not connect to Adam's live activity stream. Your message was not sent; please try again when the connection is restored.");
      }, 15000);
      if (!wsConnecting && (!ws || ws.readyState === WebSocket.CLOSED || ws.readyState === WebSocket.CLOSING)) {
        connectWebSocket();
      }
    }
  }

  // Mobile View Switcher - Accessible Tablist Implementation
  function switchMobileTab(targetTab) {
    const isChat = targetTab === "chat";

    if (mobileTabChat) {
      mobileTabChat.classList.toggle("active", isChat);
      mobileTabChat.setAttribute("aria-selected", isChat ? "true" : "false");
      mobileTabChat.tabIndex = isChat ? 0 : -1;
    }

    if (mobileTabSidebar) {
      mobileTabSidebar.classList.toggle("active", !isChat);
      mobileTabSidebar.setAttribute("aria-selected", !isChat ? "true" : "false");
      mobileTabSidebar.tabIndex = !isChat ? 0 : -1;
    }

    if (mainLayout) {
      mainLayout.dataset.activeTab = isChat ? "chat" : "sidebar";
    }
  }

  if (mobileTabChat && mobileTabSidebar) {
    const tabs = [mobileTabChat, mobileTabSidebar];

    mobileTabChat.addEventListener("click", () => switchMobileTab("chat"));
    mobileTabSidebar.addEventListener("click", () => switchMobileTab("sidebar"));

    tabs.forEach((tab, index) => {
      tab.addEventListener("keydown", (e) => {
        let newIndex = null;
        if (e.key === "ArrowRight" || e.key === "ArrowDown") {
          newIndex = (index + 1) % tabs.length;
        } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
          newIndex = (index - 1 + tabs.length) % tabs.length;
        } else if (e.key === "Home") {
          newIndex = 0;
        } else if (e.key === "End") {
          newIndex = tabs.length - 1;
        }
        if (newIndex !== null) {
          e.preventDefault();
          const target = tabs[newIndex];
          target.focus();
          switchMobileTab(target.dataset.tab);
        }
      });
    });
  }

  // Event Listeners - Chat Form
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
    chatInput.style.height = `${Math.min(chatInput.scrollHeight, 140)}px`;
  });

  // Auth modal handlers - Rely on native click for button triggers
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
    connectWebSocket(true);
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
    connectWebSocket(true);
  });

  // Initial startup sequence
  switchMobileTab("chat");
  checkAuthStatus();
  fetchStatus();
  loadHistory();
  connectWebSocket();
})();
