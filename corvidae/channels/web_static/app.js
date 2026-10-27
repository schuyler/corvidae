"use strict";
(function () {
  // ---- Element handles and state ----
  const $ = (id) => document.getElementById(id);
  const itemsEl = $("items"), workingEl = $("working"), bannerEl = $("banner");
  const inputEl = $("input"), listEl = $("conversations"), sidebarEl = $("sidebar");
  const SLUG = /^[a-z0-9][a-z0-9_-]{0,63}$/;
  let ws = null, backoff = 1000, failures = 0, opened = false, stopped = false;
  let current = "main", knownNames = new Set(), lastSent = "", convos = [];
  // Sends on this socket not yet answered, oldest first, as
  // {conversation, text}. The server answers each send in order with its
  // `user` echo or an `error`; whatever is unanswered when the socket dies
  // may never have reached it.
  let pending = [];
  const sendBtn = $("send");

  // ---- Markdown sanitising: links open in a new tab safely ----
  DOMPurify.addHook("afterSanitizeAttributes", (node) => {
    if (node.tagName === "A" && node.hasAttribute("href")) {
      node.setAttribute("target", "_blank");
      node.setAttribute("rel", "noopener noreferrer");
    }
  });

  // ---- Small DOM helper: element with class and textContent ----
  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined && text !== null) e.textContent = text;
    return e;
  }

  // ---- Scrolling: stick to the bottom only if the user was near it ----
  function nearBottom() {
    return itemsEl.scrollHeight - itemsEl.scrollTop - itemsEl.clientHeight < 80;
  }
  function append(node) {
    const stick = nearBottom();
    itemsEl.appendChild(node);
    if (stick) itemsEl.scrollTop = itemsEl.scrollHeight;
  }

  // ---- Per-type renderers: each returns a node (or null) ----
  function renderMarkdown(cls, text) {
    const box = el("div", "item " + cls);
    const body = el("div", "md");
    // The single HTML sink in this file: sanitised Markdown output.
    body.innerHTML = DOMPurify.sanitize(marked.parse(String(text || "")), { FORBID_ATTR: ["style"], FORBID_TAGS: ["style"] });
    box.appendChild(body);
    return box;
  }
  function renderCollapsed(cls, label, text) {
    const d = el("details", "item " + cls);
    d.appendChild(el("summary", null, label));
    d.appendChild(el("div", "body", text || ""));
    return d;
  }
  const renderers = {
    // User text, tagged with the server id so rejected frames can find it.
    user(f) {
      const d = el("div", "item user", f.text || "");
      if (f.id !== null && f.id !== undefined) d.dataset.id = f.id;
      return d;
    },
    message: (f) => renderMarkdown("message", f.text),
    progress: (f) => renderMarkdown("progress", f.text),
    thinking: (f) => renderCollapsed("thinking", "thinking", f.text),
    divider: (f) => renderCollapsed("divider", "earlier conversation summarised", f.text),
    // Tool chip plus a muted, CSS-truncated summary line.
    tool_status(f) {
      const d = el("div", "item tool");
      d.appendChild(el("span", "chip", "⚙ " + f.tool + " — " + f.status));
      if (f.summary) d.appendChild(el("div", "summary", f.summary));
      return d;
    },
    // Mark the matching user item as refused, or show a standalone notice.
    rejected(f) {
      if (f.id !== null && f.id !== undefined) {
        for (const u of itemsEl.querySelectorAll(".item.user")) {
          if (u.dataset.id === String(f.id)) {
            if (!u.classList.contains("refused")) {
              u.classList.add("refused");
              u.appendChild(el("span", "refused-label", "refused"));
            }
            return null;
          }
        }
        return null;
      }
      return el("div", "item notice", "a message was refused");
    },
  };
  // Render one frame into the pane if its type has a renderer.
  function renderFrame(f) {
    const fn = renderers[f && f.type];
    const node = fn ? fn(f) : null;
    if (node) append(node);
  }
  function notice(text) { append(el("div", "item notice", text)); }

  // ---- Working indicator ----
  function setWorking(v) { workingEl.classList.toggle("hidden", !v); }

  // ---- Conversation list in the sidebar ----
  function renderList(items) {
    // The server sends epoch seconds, newest first; keep that order.
    convos = items.slice().sort((a, b) => (b.last_active || 0) - (a.last_active || 0));
    knownNames = new Set(convos.map((c) => c.name));
    listEl.textContent = "";
    for (const c of convos) {
      const li = el("li", c.name === current ? "current" : "", c.name);
      if (c.last_active) li.appendChild(el("span", "when", new Date(c.last_active * 1000).toLocaleString()));
      li.addEventListener("click", () => switchTo(c.name));
      listEl.appendChild(li);
    }
  }
  function markCurrent() {
    for (const li of listEl.children) {
      li.classList.toggle("current", li.firstChild.textContent === current);
    }
  }

  // ---- Sending frames (only when the socket is open) ----
  function send(obj) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return false;
    ws.send(JSON.stringify(obj));
    return true;
  }

  // ---- Switching conversation: clear pane, resubscribe ----
  function subscribe() {
    itemsEl.textContent = "";
    setWorking(false);
    send({ type: "subscribe", conversation: current });
  }
  function switchTo(name) {
    sidebarEl.classList.remove("open");
    $("sidebar-toggle").setAttribute("aria-expanded", "false");
    if (location.hash.slice(1) !== name) location.hash = name; // triggers hashchange
  }
  function hashName() {
    // A malformed escape (e.g. "#%") must not kill the page.
    let h = "";
    try { h = decodeURIComponent(location.hash.slice(1)); } catch (e) { h = ""; }
    return SLUG.test(h) ? h : "main";
  }
  window.addEventListener("hashchange", () => {
    const n = hashName();
    if (n === current) return;
    current = n;
    markCurrent();
    subscribe();
  });

  // ---- Terminal "session expired" state ----
  function expired() {
    stopped = true;
    bannerEl.textContent = "session expired — log in again with the URL from the daemon log";
    bannerEl.classList.remove("hidden");
    sendBtn.disabled = true;
  }

  // ---- Incoming frame dispatch ----
  function onFrame(f) {
    if (!f || typeof f !== "object") return;
    if (f.type === "conversations") { renderList(f.items || []); return; }
    if (f.type === "error") { onError(f); return; }
    // Match our own send's echo before the filter below: the user may have
    // switched conversation since sending.
    if (f.type === "user" && pending.length && f.conversation === pending[0].conversation
        && f.text === pending[0].text) pending.shift();
    // Drop anything for a conversation we are no longer viewing.
    if (f.conversation !== undefined && f.conversation !== current) return;
    if (f.type === "history") {
      itemsEl.textContent = "";
      for (const it of f.items || []) renderFrame(it);
      itemsEl.scrollTop = itemsEl.scrollHeight;
      setWorking(!!f.working);
    } else if (f.type === "working") {
      setWorking(!!f.value);
    } else {
      renderFrame(f);
    }
  }
  function onError(f) {
    // Any error for a send answers it: no echo will follow.
    if (f.request === "send" && pending.length) pending.shift();
    if (f.code === "starting") {
      // Server not ready: retry list/subscribe after 2s; restore refused send text.
      if (f.request === "list") setTimeout(() => send({ type: "list" }), 2000);
      else if (f.request === "subscribe") setTimeout(() => send({ type: "subscribe", conversation: current }), 2000);
      else if (f.request === "send") {
        notice("server is starting — message not sent");
        if (!inputEl.value) inputEl.value = lastSent;
      }
      return;
    }
    notice("error" + (f.request ? " (" + f.request + ")" : "") + ": " + (f.message || f.code || "unknown"));
  }

  // ---- Recovering unanswered sends when the socket dies ----
  // This conversation's texts go back into the composer, ahead of anything
  // already typed; the banner (which a `history` reload doesn't clear) says
  // so, and names other conversations that had unanswered sends.
  function recoverPending() {
    const here = pending.filter((p) => p.conversation === current).map((p) => p.text);
    const elsewhere = [...new Set(pending.filter((p) => p.conversation !== current)
      .map((p) => p.conversation))];
    pending = [];
    if (here.length) inputEl.value = [...here, inputEl.value].filter((t) => t).join("\n\n");
    const parts = [];
    if (here.length) parts.push((here.length === 1 ? "1 message" : here.length + " messages") +
      " put back in the composer; check and resend");
    if (elsewhere.length) parts.push("also check " + elsewhere.join(", "));
    showBanner("connection lost before the server confirmed your send: " + parts.join("; "));
  }
  function showBanner(text) {
    bannerEl.textContent = text;
    bannerEl.classList.remove("hidden");
  }

  // ---- Connection with exponential backoff ----
  function connect() {
    if (stopped) return;
    opened = false;
    ws = new WebSocket((location.protocol === "https:" ? "wss" : "ws") + "://" + location.host + "/ws");
    ws.addEventListener("open", () => {
      opened = true; backoff = 1000; failures = 0;
      send({ type: "list" });
      subscribe();
    });
    ws.addEventListener("message", (ev) => {
      let f;
      try { f = JSON.parse(ev.data); } catch (e) { console.error("bad frame", e); return; }
      onFrame(f);
    });
    ws.addEventListener("close", (ev) => onClose(ev));
  }
  async function onClose(ev) {
    // Unanswered sends may not have reached the server (a half-open socket
    // after sleep): hand them all back rather than lose them.
    if (pending.length) recoverPending();
    if (ev.code === 4001) { expired(); return; }
    if (!opened && ++failures >= 3) {
      // Repeated connect failures: probe the page to detect an expired session.
      try {
        const r = await fetch("/", { cache: "no-store" });
        if (r.status === 401) { expired(); return; }
      } catch (e) { console.warn("probe failed", e); }
    }
    if (stopped) return;
    setTimeout(connect, backoff);
    backoff = Math.min(backoff * 2, 30000);
  }

  // ---- Composer: send text, Enter handling ----
  function submitMessage() {
    const text = inputEl.value;
    if (!text.trim()) return;
    // Keep the text in the composer if the socket is down.
    if (!send({ type: "send", conversation: current, text })) {
      notice("not connected — message not sent");
      return;
    }
    lastSent = text;
    pending.push({ conversation: current, text });
    inputEl.value = "";
    if (!stopped) bannerEl.classList.add("hidden");  // a fresh send clears the recovery note
    // A send to an unlisted conversation should show up in the list.
    if (!knownNames.has(current)) send({ type: "list" });
  }
  $("composer").addEventListener("submit", (e) => { e.preventDefault(); submitMessage(); });
  inputEl.addEventListener("keydown", (e) => {
    if (e.key !== "Enter" || e.shiftKey || e.isComposing) return;
    if (window.matchMedia("(pointer: coarse)").matches) return; // touch: newline
    e.preventDefault();
    submitMessage();
  });

  // ---- New-conversation form and mobile sidebar toggle ----
  $("new-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const input = $("new-name"), name = input.value.trim();
    if (!SLUG.test(name)) { input.reportValidity(); return; }
    input.value = "";
    switchTo(name);
  });
  $("sidebar-toggle").addEventListener("click", () => {
    const open = sidebarEl.classList.toggle("open");
    $("sidebar-toggle").setAttribute("aria-expanded", String(open));
  });

  // ---- Start ----
  current = hashName();
  connect();
})();
