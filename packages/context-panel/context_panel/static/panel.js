// Context panel client: lists loaded channels, opens one channel's window and
// follows it by polling. Every string reaches the DOM through textContent, so
// untrusted window content can never become markup.
"use strict";
(function () {
  const NAME = document.currentScript.dataset.panel;
  const POLL_MS = 1000;
  const NON_BODY_KEYS = ["role", "content", "reasoning_content", "tool_calls", "tool_call_id"];

  const section = document.querySelector(`.panel-section[data-panel="${NAME}"]`);
  const state = {
    channel: "",   // channel id of the selected tab, or "" when none is selected
    gen: 0,        // bumped on every selection or open, so late responses are dropped
    cursor: null,  // {count, digest} of the rows the view holds
    opening: false,
    scrollPending: false,  // set on tab selection; the next completed open scrolls to the end
    stopped: false,
    liveFirst: null,  // seq of the first live event shown
    liveCount: 0,
    toolNames: new Map(),  // tool_call_id -> tool name, filled as assistant rows render
    rowsText: "",  // the status line for the rows shown, restored after a failed poll
  };

  // Create an element with properties and children.
  function el(tag, props = {}, ...kids) {
    const node = document.createElement(tag);
    Object.assign(node, props);
    node.append(...kids);
    return node;
  }

  // Build the sidebar section's tab list and the main-pane view: status line, entries, live area.
  const refresh = el("button", { type: "button", textContent: "refresh" });
  const tabs = el("ul", { id: "context-tabs" });
  const view = el("div", { id: "context-view", className: "panel-view hidden" });
  const status = el("span", { id: "status" });
  const showAll = el("button", { type: "button", textContent: "show all", hidden: true });
  const entries = el("div", { id: "entries" });
  const liveEntries = el("div", { id: "live-entries" });
  const liveArea = el("details", { id: "live", open: true, hidden: true },
    el("summary", { textContent: "live (latest turn)" }), liveEntries);
  section.append(refresh, tabs);
  view.append(el("div", { id: "statusline" }, status, showAll), entries, liveArea);
  document.getElementById("main").append(view);

  // One collapsible entry: summary label plus a text-only body.
  function entry(cat, label, text, extraClasses = []) {
    const d = el("details", { className: ["entry", cat, ...extraClasses].join(" ") },
      el("summary", { textContent: label }), el("pre", { className: "body", textContent: text }));
    d.open = cat === "cat-user" || cat === "cat-assistant";
    return d;
  }

  // Strings show as they are; anything else as indented JSON.
  function show(value) {
    return typeof value === "string" ? value : JSON.stringify(value, null, 2);
  }

  // A tool call shows as name(arguments); one without that shape as its JSON.
  function callText(call) {
    const fn = call && call.function;
    if (fn && typeof fn.name === "string" && "arguments" in fn) {
      return `${fn.name}(${show(fn.arguments)})`;
    }
    return JSON.stringify(call, null, 2);
  }

  // The tool name of a call, or "" when it has none.
  function callName(call) {
    const name = call && call.function && call.function.name;
    return typeof name === "string" ? name : "";
  }

  // Local-time YYYY-MM-DD HH:MM for epoch seconds.
  function stamp(ts) {
    const d = new Date(ts * 1000);
    const p = (n) => String(n).padStart(2, "0");
    return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
  }

  // Turn one wire row into its entries, first matching rule wins.
  function rowEntries(index, row) {
    const msg = row.msg;
    const tag = `#${index} · ${msg.role} · ${row.type}` + (row.ts === undefined ? "" : ` · ${stamp(row.ts)}`);
    const extras = Object.keys(msg).filter((k) => !NON_BODY_KEYS.includes(k))
      .map((k) => `${k}: ${show(msg[k])}\n`).join("");
    const out = [];
    const content = msg.content === undefined || msg.content === null ? "" : show(msg.content);
    if (row.type === "summary" || row.type === "context" || msg.role === "system") {
      out.push(entry("cat-inserted", tag, extras + content));
    } else if (msg.role === "tool") {
      const name = state.toolNames.get(msg.tool_call_id);
      out.push(entry("cat-tool-output", `${tag} · ${msg.tool_call_id}${name ? ` · ${name}` : ""}`,
        extras + content));
    } else if (msg.role === "user") {
      out.push(entry("cat-user", tag, extras + content));
    } else if (msg.role === "assistant") {
      if (msg.reasoning_content) {
        out.push(entry("cat-thinking", `${tag} · thinking`, show(msg.reasoning_content)));
      }
      if (content !== "") {
        const withheld = row.type === "unsent";
        out.push(entry("cat-assistant", withheld ? `${tag} · withheld (not sent)` : tag, content,
          withheld ? ["withheld"] : []));
      }
      for (const call of msg.tool_calls || []) {
        const name = callName(call);
        if (call && call.id !== undefined) state.toolNames.set(call.id, name);
        out.push(entry("cat-tool-call", `${tag} · tool call${name ? ` · ${name}` : ""}`, callText(call)));
      }
      if (out.length === 0) {
        out.push(entry("cat-assistant", tag, ""));
      }
    } else {
      out.push(entry("cat-user", tag, extras + content));
    }
    // Row-level extras ride on the row's first entry only.
    if (out.length && msg.role === "assistant") {
      out[0].querySelector(".body").textContent = extras + out[0].querySelector(".body").textContent;
    }
    return out;
  }

  // Append rows starting at the window index `start`.
  function appendRows(rows, start) {
    rows.forEach((row, i) => entries.append(...rowEntries(start + i, row)));
  }

  // Live event -> entry: only progress (assistant text) starts open, like every other entry.
  function liveEntry(ev) {
    if (ev.kind === "thinking") return entry("cat-thinking", "thinking", ev.text);
    if (ev.kind === "progress") return entry("cat-assistant", "progress", ev.text);
    const cat = ev.status === "completed" ? "cat-tool-output" : "cat-tool-call";
    return entry(cat, `tool ${ev.tool} · ${ev.status}`, ev.summary);
  }

  // Show the live events, appending when the list only grew, replacing when it cleared and refilled.
  function renderLive(events) {
    if (events.length === 0) {
      liveEntries.replaceChildren();
      liveArea.hidden = true;
      state.liveFirst = null;
      state.liveCount = 0;
      return;
    }
    const grew = events[0].seq === state.liveFirst && events.length >= state.liveCount;
    if (!grew) {
      liveEntries.replaceChildren();
      state.liveCount = 0;
    }
    for (const ev of events.slice(state.liveCount)) liveEntries.append(liveEntry(ev));
    state.liveFirst = events[0].seq;
    state.liveCount = events.length;
    liveArea.hidden = false;
  }

  function setStatus(text) {
    status.textContent = text;
  }

  // GET a panel route; resolves to {status, body} (body is null unless 200).
  async function get(path, params) {
    const resp = await fetch(`/panel/${NAME}/${path}?${new URLSearchParams(params)}`);
    return { status: resp.status, body: resp.status === 200 ? await resp.json() : null };
  }

  // Handle a non-200 or failed fetch; returns true when the response is unusable.
  function failed(res) {
    if (res.status === 401) {
      state.stopped = true;
      setStatus("session expired; reload to log in");
      return true;
    }
    if (res.status !== 200) {
      setStatus(`connection lost; retrying (HTTP ${res.status})`);
      return true;
    }
    return false;
  }

  // Clear the view for a channel that has no loaded window.
  function showUnloaded() {
    state.cursor = null;
    state.toolNames.clear();
    entries.replaceChildren();
    showAll.hidden = true;
    renderLive([]);
    setStatus("not loaded yet: it loads on the channel's next turn");
  }

  // Fetch the window and render it from scratch. A pending scroll (set on tab selection) runs once the newest open has rendered its first poll.
  async function openWindow(all) {
    const gen = ++state.gen;
    state.opening = true;
    try {
      const params = { channel: state.channel };
      if (all) params.all = "1";
      const res = await get("window", params);
      if (gen !== state.gen || failed(res)) return;
      const body = res.body;
      if (!body.loaded) {
        showUnloaded();
        return;
      }
      state.toolNames.clear();
      entries.replaceChildren(
        entry("cat-system-prompt", "system prompt", body.system_prompt),
        entry("cat-system-prompt", "tools",
          body.tools === null ? "not resolved yet" : JSON.stringify(body.tools, null, 2)));
      appendRows(body.rows, body.first);
      renderLive([]);
      state.cursor = body.cursor;
      showAll.hidden = body.first === 0;
      state.rowsText = body.first === 0 ? `rows 1–${body.total}`
        : `rows ${body.first + 1}–${body.total} of ${body.total}`;
      setStatus(state.rowsText);
      // The window carries no live events; fetch them now rather than a second later.
      state.opening = false;
      await pollOnce();
      if (state.scrollPending && gen === state.gen) {
        state.scrollPending = false;
        view.scrollTop = view.scrollHeight;
      }
    } catch (err) {
      console.warn("context panel: window fetch failed", err);
      setStatus("connection lost; retrying");
    } finally {
      // A stale open must not clear the flag a newer open set.
      if (gen === state.gen) state.opening = false;
    }
  }

  // One poll: tail the selected channel, then show its live events.
  async function pollOnce() {
    const gen = state.gen;
    const sent = state.cursor;
    const params = { channel: state.channel };
    if (sent) Object.assign(params, sent);
    const res = await get("poll", params);
    // Drop the response if the selection or the page's rows changed meanwhile.
    if (gen !== state.gen || sent !== state.cursor || failed(res)) return;
    const body = res.body;
    if (!body.loaded) {
      showUnloaded();
      return;
    }
    if (body.resync) {
      await openWindow(false);
      return;
    }
    if (body.rows.length) appendRows(body.rows, sent.count);
    state.cursor = body.cursor;
    renderLive(body.live);
    setStatus(state.rowsText);
  }

  // The poll chain: the next tick is scheduled only after the last one finishes.
  async function tick() {
    if (state.channel && !state.opening && !state.stopped) {
      try {
        await pollOnce();
      } catch (err) {
        console.warn("context panel: poll failed", err);
        setStatus("connection lost; retrying");
      }
    }
    if (!state.stopped) setTimeout(tick, POLL_MS);
  }

  // Mark the tab of the selected channel.
  function markTabs() {
    for (const a of tabs.querySelectorAll("a")) a.classList.toggle("current", a.dataset.id === state.channel);
  }

  // One tab per channel: its name or id, then its size.
  function renderTabs(channels) {
    tabs.replaceChildren(...channels.map((c) => {
      const a = el("a", { href: `#${NAME}/${encodeURIComponent(c.id)}` },
        el("span", { textContent: c.name ? `${c.name} (${c.id})` : c.id }),
        el("span", { className: "size", textContent: `${c.messages} msgs, ${c.tokens} / ${c.budget} tokens` }));
      a.dataset.id = c.id;
      return el("li", {}, a);
    }));
    markTabs();
  }

  // Fill the tab list, keeping the selected tab.
  async function loadChannels() {
    try {
      const res = await get("channels", {});
      if (failed(res)) return;
      renderTabs(res.body.channels);
    } catch (err) {
      console.warn("context panel: channel list failed", err);
    }
  }

  // Show the view while the hash is this panel's route, hide and idle it otherwise.
  function route() {
    const prefix = `#${NAME}/`;
    let id = "";
    if (location.hash.startsWith(prefix)) {
      // A malformed escape means no channel.
      try { id = decodeURIComponent(location.hash.slice(prefix.length)); } catch (err) { id = ""; }
    }
    state.channel = id;
    state.cursor = null;
    markTabs();
    if (id) {
      view.classList.remove("hidden");
      state.scrollPending = true;
      openWindow(false);
    } else {
      state.gen++;
      state.scrollPending = false;
      entries.replaceChildren();
      renderLive([]);
      setStatus("");
      view.classList.add("hidden");
    }
  }

  refresh.addEventListener("click", loadChannels);
  showAll.addEventListener("click", () => openWindow(true));
  window.addEventListener("hashchange", route);

  loadChannels();
  route();
  setTimeout(tick, POLL_MS);
})();
