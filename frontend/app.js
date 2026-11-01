"use strict";

function alertTitle(event) {
  if (event.weapon >= 0.5 && event.violence >= 0.5) return "Armed altercation detected";
  if (event.violence >= 0.5) return "Violent activity detected";
  if (event.weapon >= 0.5) return "Weapon detected";
  if (event.loitering >= 0.5) return "Suspicious loitering detected";
  return event.level === "LOW" ? "Unusual activity detected" : "Suspicious activity detected";
}

function filterEvents(events, { tab, severity, query, acknowledged }) {
  const search = query.trim().toLowerCase();
  return events.filter(event =>
    (tab !== "unread" || !acknowledged.has(event.id)) &&
    (severity === "all" || event.level === severity) &&
    (!search || [alertTitle(event), event.track_id, event.identity, event.level,
      ...event.messages, ...event.factors].join(" ").toLowerCase().includes(search))
  );
}

function connectionView(state, failed, now = Date.now() / 1000) {
  const phase = state?.connection?.status || "external";
  const inferenceLive = !failed && Boolean(state?.updated_at) && now - state.updated_at < 10;
  const cameraLive = !failed && state?.has_frame && state.preview_mode === "camera" &&
    ["starting", "camera_only"].includes(phase) && now - state.connection.camera_updated_at < 10;
  const labels = { disconnected: "No Pi connected", checking: "Checking Pi gateway", starting: "Starting detection",
    camera_only: "Camera only · Detection offline", stopping: "Disconnecting Pi", error: "Gateway unavailable" };
  const status = failed ? "Host disconnected" : labels[phase] || (inferenceLive ? "System connected" : state?.updated_at ? "Inference stale" : "Waiting for inference");
  return { phase, inferenceLive, cameraLive: Boolean(cameraLive), status,
    frameAt: cameraLive ? state.connection.camera_updated_at : state?.updated_at };
}

function monitoringOrigin(value, pageOrigin, pageProtocol) {
  if (!value.trim()) {
    if (!["localhost", "127.0.0.1", "[::1]"].includes(new URL(pageOrigin).hostname)) {
      throw new Error("Enter the HTTPS URL of the host running dashboard.py.");
    }
    return pageOrigin;
  }
  let url;
  try { url = new URL(value.trim()); } catch { throw new Error("Enter a complete monitoring host URL, including https://."); }
  if (!["http:", "https:"].includes(url.protocol) || url.username || url.password || url.search || url.hash || url.pathname !== "/") {
    throw new Error("Use the monitoring host origin, without a path, credentials, or query parameters.");
  }
  if (pageProtocol === "https:" && url.protocol !== "https:") {
    throw new Error("A Vercel HTTPS page needs an HTTPS monitoring host. Expose the local host through an HTTPS tunnel or reverse proxy.");
  }
  return url.origin;
}

function startDashboard() {
  const $ = id => document.getElementById(id);
  const node = (tag, className, text) => {
    const item = document.createElement(tag);
    if (className) item.className = className;
    if (text !== undefined) item.textContent = text;
    return item;
  };
  const badge = level => node("span", `severity-badge severity-${level.toLowerCase()}`, level);
  const formatTime = timestamp => new Date(timestamp * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  const percent = value => `${Math.round(value * 100)}%`;
  const storageKey = "sentinel-acknowledged";
  let acknowledged = new Set();
  try {
    const saved = JSON.parse(localStorage.getItem(storageKey) || "[]");
    if (Array.isArray(saved)) acknowledged = new Set(saved.filter(id => typeof id === "string").slice(-200));
  } catch { /* Browser storage can be unavailable; reviews still work for this page. */ }
  let state = null;
  let tab = "all";
  let selectedId = null;
  let seenIds = new Set();
  let previousSession = null;
  let framePending = false;
  let frameTimestamp = null;
  let notificationsEnabled = false;
  let requestFailed = false;
  let lastRenderKey = null;
  let commandPending = false;
  let gatewayInputLoaded = false;
  let apiOrigin = ["localhost", "127.0.0.1", "[::1]"].includes(location.hostname) ? location.origin : null;
  let accessKey = "";
  let hostError = "";
  let hostGeneration = 0;
  let frameObjectUrl = null;

  try { $("monitoring-url").value = localStorage.getItem("sentinel-monitoring-host") || ""; } catch { /* Host selection remains available. */ }

  async function apiFetch(path, options = {}) {
    if (!apiOrigin) throw new Error("Select a monitoring host first.");
    const headers = new Headers(options.headers);
    if (accessKey) headers.set("Authorization", `Bearer ${accessKey}`);
    return fetch(apiOrigin + path, { ...options, headers, cache: "no-store", credentials: "omit", redirect: "error", referrerPolicy: "no-referrer" });
  }

  function announce(message) {
    $("notice").textContent = message;
    $("notice").hidden = false;
  }

  function saveAcknowledgements() {
    try { localStorage.setItem(storageKey, JSON.stringify([...acknowledged].slice(-200))); }
    catch { announce("Browser storage is unavailable. Acknowledgements last until this page closes."); }
  }

  function acknowledge(ids) {
    ids.forEach(id => acknowledged.add(id));
    saveAcknowledgements();
    render();
    if (selectedId) $("detail-ack").disabled = acknowledged.has(selectedId);
  }

  function showDetails(event) {
    selectedId = event.id;
    const body = $("detail-body");
    body.replaceChildren(badge(event.level), node("h2", "", alertTitle(event)));
    event.messages.forEach(message => body.append(node("p", "", message)));
    const details = node("dl");
    [["Detected", new Date(event.timestamp * 1000).toLocaleString()], ["Track", `#${event.track_id}`],
      ["Identity estimate", event.identity], ["Fused risk score", percent(event.score)],
      ["Violence / weapon / loitering", `${percent(event.violence)} / ${percent(event.weapon)} / ${percent(event.loitering)}`]
    ].forEach(([label, value]) => {
      const row = node("div");
      row.append(node("dt", "", label), node("dd", "", value));
      details.append(row);
    });
    body.append(details, node("p", "", "Scores are model indicators, not verified incidents. Review the camera context before responding."));
    if (event.factors.length) {
      const list = node("ul");
      event.factors.forEach(factor => list.append(node("li", "", factor)));
      body.append(list);
    }
    $("detail-ack").disabled = acknowledged.has(event.id);
    $("detail-ack").textContent = acknowledged.has(event.id) ? "Acknowledged" : "Acknowledge alert";
    if (!$("detail-dialog").open) $("detail-dialog").showModal();
  }

  function renderConnection() {
    const timestamp = state?.updated_at;
    const { inferenceLive: connected, cameraLive, phase, status: connectionStatus, frameAt } = connectionView(state, requestFailed);
    const status = !apiOrigin ? "Choose monitoring host" : connectionStatus;
    $("connection-badge").textContent = state?.demo && connected ? "Demo connected" : status;
    $("connection-badge").className = `connection-badge ${connected ? "connected" : "stale"}`;
    $("side-status").textContent = status;
    $("host-status").textContent = status;
    $("last-update").textContent = timestamp ? `${Math.max(0, Math.round(Date.now() / 1000 - timestamp))}s ago` : "—";
    $("side-dot").classList.toggle("amber-dot", !connected);
    $("device-dot").classList.toggle("amber-dot", !(connected || cameraLive));
    $("live-label").className = `live-label ${connected || cameraLive ? "online" : ""}`;
    $("live-label").replaceChildren(node("span", "tiny-dot"), document.createTextNode(state?.demo ? " SIMULATED" : cameraLive ? " CAMERA ONLY" : connected ? " LIVE" : " NO SIGNAL"));
    $("connection-note").textContent = requestFailed ? "Retrying the detection host automatically…" : !connected && timestamp ? "No new inference in 10 seconds. Check the camera and pipeline." : cameraLive ? "Camera footage is live. Threat detection is not active." : "Monitoring continues while this page is open.";
    $("threat-level").textContent = connected ? state.level : "—";
    $("threat-level").className = `metric-value level-value ${connected ? `severity-${state.level.toLowerCase()}` : ""}`;
    $("threat-note").textContent = connected ? state.level === "NORMAL" ? "No unusual activity in the latest frame" : "From stabilized threat fusion" : "Current threat is unavailable";
    $("people-count").textContent = connected ? state.persons.length : "—";
    $("fps").textContent = connected ? state.fps.toFixed(1) : "—";
    $("camera-time").textContent = connected || cameraLive ? formatTime(frameAt) : "NO SIGNAL";
    $("preview-kind").textContent = cameraLive ? "RAW CAMERA · DETECTION OFFLINE" : "ANNOTATED PREVIEW · 1 FPS";
    $("gateway-form").hidden = Boolean(state && !state.can_connect);
    $("monitoring-status").textContent = hostError || (!apiOrigin ? "Choose the HTTPS monitoring host to begin." : apiOrigin === location.origin ? "Using this local monitoring host." : `Using ${apiOrigin}`);
    const busy = ["checking", "starting", "running", "camera_only", "stopping"].includes(phase);
    $("gateway-address").disabled = busy || commandPending;
    $("gateway-connect").disabled = !state?.can_connect || busy || commandPending || requestFailed;
    $("gateway-disconnect").disabled = !state?.can_connect || !busy || phase === "stopping" || commandPending || requestFailed;
    $("gateway-connect").textContent = phase === "checking" ? "Checking gateway…" : phase === "starting" ? "Starting detection…" : ["running", "camera_only"].includes(phase) ? "Camera connected" : "Connect & monitor";
    $("gateway-status").textContent = state?.connection?.error || (!state ? "Select a reachable monitoring host above, then connect the Pi." : phase === "demo" ? "Connect a Pi to leave demo mode and monitor real footage." : phase === "external" ? "Camera source is managed by main.py. Run dashboard.py to connect from this page." : phase === "starting" ? "Gateway verified. Loading detection models on this host…" : phase === "checking" ? "Verifying /health and reading a camera frame…" : phase === "disconnected" ? "Enter the Pi address to connect." : status);
    if (!gatewayInputLoaded && state) {
      try { $("gateway-address").value = state.connection?.gateway || localStorage.getItem("sentinel-gateway") || ""; } catch { /* Input still works without storage. */ }
      gatewayInputLoaded = true;
    }
    if (!(connected || cameraLive)) {
      $("camera-image").hidden = true;
      $("camera-placeholder").hidden = false;
      $("camera-empty-title").textContent = requestFailed ? "Detection host disconnected" : phase === "checking" ? "Connecting to Raspberry Pi" : phase === "starting" ? "Starting detection" : timestamp ? "Camera feed is stale" : "Waiting for the camera";
      $("camera-empty-note").textContent = state?.can_connect ? "Use Connect Raspberry Pi to choose the camera gateway." : "Check that the camera and detection pipeline are running.";
    }
    return { connected, cameraLive, frameAt };
  }

  function render() {
    if (!state) {
      renderConnection();
      ["review-count", "nav-count", "tab-count"].forEach(id => { $(id).textContent = "0"; });
      $("event-total").textContent = "0 EVENTS";
      $("ack-all").disabled = true;
      $("event-rows").replaceChildren();
      $("empty-state").hidden = false;
      $("latest-alert").replaceChildren(node("h3", "", "Waiting for monitoring host"), node("p", "", "Alerts will appear after the selected host connects."));
      $("source-label").textContent = "No camera selected";
      $("frame-count").textContent = "Awaiting processed frames";
      $("demo-banner").hidden = true;
      return;
    }
    const events = state.events;
    const unread = events.filter(event => !acknowledged.has(event.id));
    $("demo-banner").hidden = !state.demo;
    $("review-count").textContent = unread.length;
    $("nav-count").textContent = unread.length;
    $("tab-count").textContent = unread.length;
    $("event-total").textContent = `${events.length} EVENT${events.length === 1 ? "" : "S"}`;
    $("ack-all").disabled = unread.length === 0;
    $("source-label").textContent = state.source;
    $("frame-count").textContent = `${state.frames.toLocaleString()} frames processed this session`;
    const { connected, cameraLive, frameAt } = renderConnection();
    if (state.demo) {
      $("camera-placeholder").hidden = false;
      $("camera-image").hidden = true;
      $("camera-empty-title").textContent = "Demo camera · simulated events";
      $("camera-empty-note").textContent = "Run the detection pipeline to see real annotated footage.";
    } else if ((connected || cameraLive) && state.has_frame && !framePending && frameTimestamp !== frameAt) {
      framePending = true;
      frameTimestamp = frameAt;
      loadFrame(frameAt);
    }

    // Keep review controls focused while camera and telemetry updates arrive.
    const renderKey = JSON.stringify([state.session_id, events.map(event => event.id), [...acknowledged], tab, $("severity").value, $("search").value]);
    if (renderKey === lastRenderKey) return;
    lastRenderKey = renderKey;

    const latest = events[0];
    if (latest) {
      const content = $("latest-alert");
      const details = node("div", "latest-meta");
      details.append(node("span", "", `TRACK #${latest.track_id}`), node("span", "", formatTime(latest.timestamp)));
      const button = node("button", "button", "Review detection ↗");
      button.addEventListener("click", () => showDetails(latest));
      content.replaceChildren(badge(latest.level), node("h3", "", alertTitle(latest)),
        node("p", "", latest.messages[0] || "Unusual activity flagged by the detection pipeline."), details, button);
    } else {
      $("latest-alert").replaceChildren(node("span", "clear-icon", "✓"), node("h3", "", "No alerts yet"), node("p", "", "Suspicious activity will appear here as it is detected."));
    }

    const visible = filterEvents(events, { tab, severity: $("severity").value, query: $("search").value, acknowledged });
    const rows = document.createDocumentFragment();
    visible.forEach(event => {
      const row = node("tr", "event-row");
      const description = node("td");
      description.append(node("div", "event-title", alertTitle(event)), node("div", "event-subtitle", event.factors.join(" · ") || "Temporal fusion · Edge camera 01"));
      const severity = node("td");
      severity.append(badge(event.level));
      const detected = node("td");
      detected.append(node("div", "", formatTime(event.timestamp)), node("div", "time-secondary", new Date(event.timestamp * 1000).toLocaleDateString([], { month: "short", day: "numeric" })));
      const review = node("td");
      if (acknowledged.has(event.id)) review.append(node("span", "acknowledged", "✓ Reviewed"));
      else {
        const button = node("button", "review-button", "Acknowledge");
        button.setAttribute("aria-label", `Acknowledge ${event.level.toLowerCase()} alert for track ${event.track_id}`);
        button.addEventListener("click", () => acknowledge([event.id]));
        review.append(button);
      }
      const detail = node("td");
      const open = node("button", "icon-button", "↗");
      open.setAttribute("aria-label", `View detection for track ${event.track_id}`);
      open.addEventListener("click", () => showDetails(event));
      detail.append(open);
      row.append(description, severity, node("td", "", `#${event.track_id}`), detected, review, detail);
      rows.append(row);
    });
    $("event-rows").replaceChildren(rows);
    $("empty-state").hidden = visible.length > 0;
    $("empty-state").querySelector("h3").textContent = events.length ? "No alerts match this view" : "Ready when something needs your attention";
    $("empty-state").querySelector("p").textContent = events.length ? "Try another filter or switch to all alerts." : "New detections will appear here automatically.";
  }

  async function poll() {
    const generation = hostGeneration;
    try {
      if (!apiOrigin) { renderConnection(); return; }
      const response = await apiFetch("/api/state", { signal: AbortSignal.timeout(5000) });
      if (response.status === 401) throw new Error("Access key rejected. Enter the monitoring host's DASHBOARD_ACCESS_TOKEN above.");
      if (response.status === 403) throw new Error("This frontend origin is not allowed. Add this page's origin with --allow-origin on the host.");
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const next = await response.json();
      if (generation !== hostGeneration) return;
      if (!Array.isArray(next.events) || !Array.isArray(next.persons)) throw new Error("Invalid dashboard response");
      if (next.session_id === previousSession && notificationsEnabled && Notification.permission === "granted") {
        next.events.filter(event => !seenIds.has(event.id)).slice(0, 3).forEach(event => {
          try {
            new Notification(`${next.demo ? "[DEMO] " : ""}${event.level}: ${alertTitle(event)}`, { body: `Track #${event.track_id} · ${event.messages[0] || "Review the live monitor."}`, tag: event.id });
          } catch { announce("Desktop notifications are unavailable. In-page alerts remain active."); }
        });
      }
      previousSession = next.session_id;
      seenIds = new Set(next.events.map(event => event.id));
      acknowledged = new Set([...acknowledged].filter(id => seenIds.has(id)));
      state = next;
      requestFailed = false;
      hostError = "";
      render();
    } catch (error) {
      if (generation !== hostGeneration) return;
      requestFailed = true;
      hostError = error instanceof TypeError ? "Could not reach the monitoring host. Check its HTTPS URL, network, and --allow-origin configuration." : error.message;
      renderConnection();
    } finally {
      setTimeout(poll, 1000);
    }
  }

  async function loadFrame(frameAt) {
    const generation = hostGeneration;
    try {
      const response = await apiFetch(`/api/frame.jpg?t=${frameAt}`, { signal: AbortSignal.timeout(5000) });
      if (!response.ok) throw new Error("Frame unavailable");
      const blob = await response.blob();
      if (generation !== hostGeneration) return;
      const previous = frameObjectUrl;
      frameObjectUrl = URL.createObjectURL(blob);
      $("camera-image").src = frameObjectUrl;
      if (previous) URL.revokeObjectURL(previous);
    } catch {
      if (generation !== hostGeneration) return;
      framePending = false;
      frameTimestamp = null;
      $("camera-image").hidden = true;
      $("camera-placeholder").hidden = false;
    }
  }

  $("camera-image").addEventListener("load", () => {
    framePending = false;
    const view = connectionView(state, requestFailed);
    if (state && !state.demo && (view.inferenceLive || view.cameraLive)) {
      $("camera-image").hidden = false;
      $("camera-placeholder").hidden = true;
    }
  });
  $("camera-image").addEventListener("error", () => {
    framePending = false;
    frameTimestamp = null;
    $("camera-image").hidden = true;
    $("camera-placeholder").hidden = false;
  });
  document.querySelectorAll("[data-tab]").forEach(button => button.addEventListener("click", () => {
    tab = button.dataset.tab;
    document.querySelectorAll("[data-tab]").forEach(item => {
      item.classList.toggle("active", item === button);
      item.setAttribute("aria-pressed", String(item === button));
    });
    render();
  }));
  $("search").addEventListener("input", render);
  $("severity").addEventListener("change", render);
  $("ack-all").addEventListener("click", () => acknowledge(state.events.map(event => event.id)));
  $("close-dialog").addEventListener("click", () => $("detail-dialog").close());
  $("detail-dialog").addEventListener("close", () => { selectedId = null; });
  $("detail-ack").addEventListener("click", () => {
    if (selectedId) acknowledge([selectedId]);
    $("detail-ack").textContent = "Acknowledged";
  });
  async function gatewayCommand(action) {
    if (!state || commandPending) return;
    const address = $("gateway-address").value.trim();
    commandPending = true;
    renderConnection();
    try {
      const response = await apiFetch(`/api/gateway/${action}`, {
        method: "POST", headers: { "Content-Type": "application/json", "X-Sentinel-Token": state.control_token },
        body: JSON.stringify(action === "connect" ? { address } : {}), signal: AbortSignal.timeout(8000),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "Gateway request was rejected.");
      if (action === "connect") {
        try { localStorage.setItem("sentinel-gateway", address); } catch { /* Saving the address is optional. */ }
      }
      $("notice").hidden = true;
    } catch (error) { announce(error.message || "Could not reach the dashboard. Please retry."); }
    finally { commandPending = false; renderConnection(); }
  }
  $("gateway-form").addEventListener("submit", event => { event.preventDefault(); gatewayCommand("connect"); });
  $("gateway-disconnect").addEventListener("click", () => gatewayCommand("disconnect"));
  $("monitoring-form").addEventListener("submit", event => {
    event.preventDefault();
    try {
      apiOrigin = monitoringOrigin($("monitoring-url").value, location.origin, location.protocol);
      accessKey = $("monitoring-key").value.trim();
      hostGeneration += 1;
      state = null;
      previousSession = null;
      requestFailed = false;
      hostError = "Connecting to monitoring host…";
      lastRenderKey = null;
      framePending = false;
      frameTimestamp = null;
      $("camera-image").removeAttribute("src");
      if (frameObjectUrl) URL.revokeObjectURL(frameObjectUrl);
      frameObjectUrl = null;
      try { localStorage.setItem("sentinel-monitoring-host", $("monitoring-url").value.trim()); } catch { /* Saving the host URL is optional. */ }
      render();
    } catch (error) { hostError = error.message; renderConnection(); }
  });
  $("notifications").addEventListener("click", async () => {
    if (!("Notification" in window) || !window.isSecureContext) {
      announce("Desktop notifications need HTTPS or localhost and browser support. In-page alerts remain active.");
      return;
    }
    try {
      const permission = await Notification.requestPermission();
      notificationsEnabled = permission === "granted";
      $("notifications").textContent = notificationsEnabled ? "✓ Notifications enabled" : "Notifications blocked";
      $("notifications").disabled = notificationsEnabled;
      announce(notificationsEnabled ? "New alerts will notify you while this page is open." : "Allow notifications in browser settings to receive desktop alerts. In-page alerts remain active.");
    } catch { announce("The browser could not enable notifications. In-page alerts remain active."); }
  });
  function updateClock() {
    $("clock").textContent = new Date().toLocaleString([], { day: "2-digit", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" });
  }
  updateClock();
  setInterval(() => { updateClock(); renderConnection(); }, 1000);
  poll();
}

if (typeof document !== "undefined") startDashboard();
if (typeof module !== "undefined") module.exports = { alertTitle, filterEvents, connectionView, monitoringOrigin };
