# ACP transport for Corvidae

**Status:** decisions settled — ready for work-package planning; do not implement until WPs exist and a session is kicked off to execute them.  
**Goal:** full **ACP v1** conformance as the destination; **bb** (and other clients) as interoperability milestones on the way.  
**Companion research:** [acp-conformance-research.md](./acp-conformance-research.md).

## North star

```
full ACP v1 conformance
        ↑
   richer capabilities (load/resume/close, permissions,
   client fs/terminal, slash commands, config options, …)
        ↑
   bb customAcpAgents milestone (usable coding threads)
        ↑
   MVP: initialize + session/new + prompt/update/cancel
        ↑
   ACP transport plugin (`AcpPlugin`) + `corvidae acp` command
```

bb is a **client** of ACP. We do not implement bb’s Provider Bridge Protocol. We implement ACP; bb’s `provider-acp` already translates.

**Executable work packages:** [implementation/acp-transport.md](./implementation/acp-transport.md) (off-sequence, signal-transport conventions). This file is the design-of-record (decisions + architecture); that file is what an implementer executes.

---

## Plugin shape

ACP is a **first-party Corvidae plugin** (plus a CLI command), not a fork of the runtime. Same pattern as CLI/IRC/Signal.

| Piece | Registration | Responsibility |
|-------|--------------|----------------|
| **`acp` command** | `[project.entry-points."corvidae.commands"]` → click command | Builds `Runtime`, sets ACP-mode overrides (logging to stderr/file, no human CLI stdin), `asyncio.run(runtime.run())` until the ACP connection ends |
| **`AcpPlugin`** | `[project.entry-points.corvidae] acp = "corvidae.channels.acp:AcpPlugin"` | Transport: owns stdio ACP via `agent-client-protocol`, maps sessions → `Channel(transport="acp", scope=sessionId)`, inbound → `on_message`, `send_*` → `session/update`, cancel → interrupt in-flight work |

**Plugin rules (match existing transports):**

- Subclass `CorvidaePlugin`; `depends_on = frozenset({"registry"})`; resolve with `get_dependency`.
- Every `send_message` / `send_thinking` / `send_tool_status` / `send_progress` opens with `if not channel.matches_transport("acp"): return`.
- Absent ACP mode (e.g. under `corvidae serve` / `cli`): plugin stays **inert** — does not touch stdin/stdout. Prefer an explicit Runtime override from the `acp` command (e.g. `config["_acp_mode"]=True`) over guessing.
- Optional: list `acp` in `plugins.disabled` to force off; omitting the `acp:` config block must not break non-ACP commands.
- Module path: `corvidae/channels/acp.py` (alongside `cli.py` / `irc.py`).

**Optional later plugins** (not required for MVP; keep seams so they can land without rewriting `AcpPlugin`):

| Plugin (future) | Milestone | Role |
|-----------------|-----------|------|
| Client tool backend / router | **M-CLIENT-TOOLS** | Prefer ACP `fs`/`terminal` + permissions for `acp:*` channels |
| Registry auth helper | **M-REGISTRY** | Terminal Auth `--setup` flow |
| Remote ACP listener | **M-REMOTE** | HTTP/WS on long-lived `serve` |

Turn-completion and active-prompt bookkeeping live **inside `AcpPlugin`** (or a private helper module it owns). Do not add new hookspecs until a second plugin needs to observe ACP-specific events.

---

## Decision log

Each decision records what we chose now, why, and the **deferred long-term option** as an explicit future milestone so it is not forgotten.

### D1 — Protocol SDK

| | |
|--|--|
| **Chose** | Official Python [`agent-client-protocol`](https://github.com/agentclientprotocol/python-sdk) |
| **Rejected** | Hand-rolled JSON-RPC / homemade schema types |
| **Why** | Schema models, `Agent` base, stdio framing, and golden fixtures track upstream; avoids silent drift from ACP |
| **Long-term** | Stay on the official SDK. If the SDK lags a needed v1 feature, contribute upstream or temporarily vendor a pin — do not fork a parallel protocol stack |

### D2 — Process model

| | |
|--|--|
| **Chose** | Dedicated `corvidae acp` subprocess (client-spawned; owns stdio) |
| **Rejected for now** | ACP attached to long-lived `corvidae serve` |
| **Why** | Matches ACP’s stdio spawn model and every real client (bb, Zed, Hermes). Keeps JSON-RPC stdout clean; isolates IDE sessions from the IRC daemon |
| **Future milestone — M-REMOTE** | Optional **non-stdio ACP transport** on the daemon (Streamable HTTP / WebSocket once ACP stabilizes them — see [transports RFD](https://agentclientprotocol.com/rfds/streamable-http-websocket-transport.md) and python-sdk `http_*` / `ws_*` examples). Goal: one long-lived Corvidae process exposing ACP to remote or always-on clients **without** replacing stdio `corvidae acp` as the primary local path. Prerequisites: primary stdio path green; ACP remote transport no longer draft-only for our target clients |

### D3 — Phase-1 tool execution

| | |
|--|--|
| **Chose** | Keep **in-process** shell / file tools for ACP MVP (effectively full-trust) |
| **Rejected for now** | Client-mediated `fs/*` + `terminal/*` as the only Phase-1 path |
| **Why** | Fastest path to a working bb thread; reuses existing tools and tests; works when the client advertises no fs/terminal |
| **Design constraint (non-negotiable)** | Tool backends for ACP channels must be **swappable**. Do not hard-wire “always local” into session code — introduce an interface (or channel-scoped tool routing) so Phase 3 can prefer client-mediated backends without a rewrite |
| **Future milestone — M-CLIENT-TOOLS** | When the client advertises `fs` / `terminal`, prefer client-mediated read/write/shell for `acp:*` channels; use `session/request_permission` for gated actions; keep in-process tools as fallback when capabilities are absent. Unlocks bb permission modes beyond `full` and editor-native diffs |

### D4 — Authentication / ACP Registry

| | |
|--|--|
| **Chose** | No auth for now — advertise `authMethods: []` |
| **Rejected for now** | Agent Auth or Terminal Auth as a Phase-0/1 requirement |
| **Why** | Local llama-server setups need no OAuth; bb/`customAcpAgents` do not require Registry listing; auth is orthogonal to protocol correctness |
| **Future milestone — M-REGISTRY** | If we want listing on the [ACP Registry](https://agentclientprotocol.com/get-started/registry.md), implement **Terminal Auth** (interactive `--setup` / login TUI — see [registry AUTHENTICATION.md](https://github.com/agentclientprotocol/registry/blob/main/AUTHENTICATION.md)) and pass registry CI’s `authMethods` check. Agent Auth (browser OAuth) only if we add cloud-provider login that needs it |

### D5 — Protocol version track

| | |
|--|--|
| **Chose** | **ACP v1 stable** only until Gate B (homegrown conformance) is green |
| **Rejected for now** | Dual-track v1+v2 implementation |
| **Why** | v2 is draft; bb and current clients speak v1; one conformance matrix |
| **Future milestone — M-V2** | After v1 conformance is green, add draft/stable v2 support behind explicit capability negotiation (python-sdk / rust-sdk v2 APIs as they mature). Track upstream via rust-sdk `testy` v2 scenarios. Do not block bb or v1 milestones on v2 |

---

## Process model (chosen: D2)

| Mode | Command | Lifetime |
|------|---------|----------|
| Daemon (today) | `corvidae serve` / `cli` | Long-lived; IRC/CLI transports |
| ACP (new) | `corvidae acp` | Client-spawned stdio subprocess; one ACP connection per process |

Inside the ACP process we still boot `Runtime` + the usual plugins so the agent loop, persistence, tools, compaction, memory, etc. remain one codebase. The new piece is an **ACP transport plugin** that owns the connection.

**Multi-session:** one ACP connection may open many `sessionId`s. Map each to `Channel(transport="acp", scope=<sessionId>)`. Persistence keys stay `acp:<sessionId>` (stable across resume if we reuse the same id).

## Layering

```
┌─────────────────────────────────────────────┐
│  ACP client (bb provider-acp, Zed, …)       │
│  JSON-RPC NDJSON over stdio                 │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│  corvidae acp  (click command)              │
│  boots Runtime; ACP owns stdin/stdout       │
│  logging → stderr / file only               │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│  AcpPlugin (transport)                      │
│  - speaks ACP via agent-client-protocol SDK │
│  - session table → Channel                  │
│  - inbound prompt → on_message              │
│  - send_* → session/update                  │
│  - cancel → interrupt in-flight turn work   │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│  Agent loop + plugins (unchanged contract)  │
│  Tool backends swappable per D3             │
└─────────────────────────────────────────────┘
```

Reuse the transport pattern from CLI/IRC ([signal-transport.md](./implementation/signal-transport.md) / `channels/cli.py`):

- `on_start` / `on_stop` lifecycle  
- every `send_*` opens with `if not channel.matches_transport("acp"): return`  
- inbound → `pm.ahook.on_message(channel=..., sender=..., text=...)`

## Hook → ACP mapping

| Corvidae hook / event | ACP |
|-----------------------|-----|
| `on_message` (user text) | driven by `session/prompt` (flatten text blocks; phase-1 ignore image/audio unless advertised) |
| `send_message` | `session/update` `agent_message_chunk` (then end turn) |
| `send_thinking` | `session/update` `agent_thought_chunk` |
| `send_progress` | `agent_message_chunk` (or thought — pick one in WPs and document) |
| `send_tool_status` dispatched | `tool_call` (pending/in_progress) |
| `send_tool_status` completed | `tool_call_update` (completed + content) |
| turn finished (agent produced final text / no tools) | respond to `session/prompt` with `stopReason: end_turn` |
| cancel | `session/cancel` → cancel queue item / tasks; `stopReason: cancelled` |
| compaction (optional later) | slash `/compact` or advertise command; trigger existing `compact_conversation` |

**Turn completion semantics:** ACP’s `session/prompt` is request/response: the response is sent when the turn ends. Corvidae’s loop is queue-driven and may re-enter via tool notifications. The ACP plugin must track “active prompt” per session and only resolve the JSON-RPC response when the agent has finished the turn (no pending tools for that correlation, final message sent) — mirroring how other adapters wait for idle.

## Capability roadmap

Near-term phases implement the chosen decisions. Deferred milestones (D2–D5) are listed after Phase 4 so they stay on the project map.

### Phase 0 — Skeleton

- `corvidae acp [--config PATH]`
- `initialize` → protocolVersion 1, minimal `agentCapabilities`, `agentInfo`, `authMethods: []`
- Reject sessions until ready; log to stderr only
- Homegrown pytest: spawn process, initialize handshake
- Depend on `agent-client-protocol` (optional extra `acp` until first-class)

### Phase 1 — MVP (bb milestone target)

- `session/new` (honor `cwd`; store on channel; chdir or tool path root)
- `session/prompt` (text) → `on_message`
- Stream message / thought / tool updates
- `session/cancel`
- In-process tools (D3); introduce swappable backend seam even if only local is wired
- Optional: connect MCP servers from `session/new` params into existing McpClientPlugin (or defer)
- **Gate:** bb `customAcpAgents` can run a coding thread

### Phase 2 — Session durability

- `loadSession: true` + `session/load` (replay from `sessions.db` via `session/update`, then ready)
- `session/resume` / `session/close` / `session/list` / `session/delete` as advertised
- Map ACP session ids carefully so persistence survives process restarts

### Phase 3 — Editor-native tools (= **M-CLIENT-TOOLS**)

- When client advertises `fs` / `terminal`, prefer client-mediated read/write/shell for ACP channels
- `session/request_permission` for gated tools
- Keep in-process tools as fallback when client capabilities are absent

### Phase 4 — Product polish toward “full v1”

- Slash commands (`available_commands_update`), including `/compact`
- Session config options (model from `llm.*` profiles)
- Prompt capabilities (image, etc.) as needed
- Homegrown conformance Gate B green for every advertised capability

### Deferred milestones (chosen “later” options)

| ID | Decision | Milestone | Trigger to start |
|----|----------|-----------|------------------|
| **M-CLIENT-TOOLS** | D3 | Phase 3 above — client fs/terminal + permissions | Phase 1–2 green; need bb modes beyond `full` or Zed-class diffs |
| **M-REMOTE** | D2 | ACP on long-lived daemon via stabilized HTTP/WS (stdio remains primary) | Stdio path + Gate B green; target client supports remote ACP |
| **M-REGISTRY** | D4 | Terminal Auth + registry `agent.json` + CI | Desire public Registry listing or cloud login UX |
| **M-V2** | D5 | ACP v2 support alongside v1 | v1 Gate B green; v2 no longer blocking-draft for our clients |

### Explicitly out of scope (not planned milestones)

- bb Provider Bridge Protocol (bb already adapts ACP)
- Replacing IRC/CLI transports with ACP
- Claiming Registry listing without **M-REGISTRY**

## Config

Proposed `agent.yaml` section (names TBD in WPs):

```yaml
acp:
  # enabled implicitly by `corvidae acp`; section for knobs only
  agent_info:
    name: corvidae
    title: Corvidae
  # cwd handling, tool backend preference, …
```

Channels may appear as `acp:<sessionId>` under `channels:` for overrides (system prompt, token budget).

## Testing strategy

See [acp-conformance-research.md](./acp-conformance-research.md).

1. **Unit / protocol:** fake stdio pair; assert JSON-RPC shapes (SDK models + golden fixtures).  
2. **Conformance harness (Gate B):** client spawns `corvidae acp`, runs scenario matrix inspired by rust-sdk `testy`.  
3. **bb milestone:** manual + scripted `bb thread spawn --provider acp-corvidae`.  
4. **Optional:** Contenox `acp-validator` opt-in CI.  
5. **Harness:** consider an ACP-driven cousin of `harness/` later for restart/compaction proofs.

## Dependencies

- Add `agent-client-protocol` via optional extra `acp = ["agent-client-protocol"]`; the `acp` command fails with a clear install hint if missing.
- Entry points as in [Plugin shape](#plugin-shape).

## Planning conventions

This design follows the same split as the rest of the repo:

| Artifact | Role | Analog |
|----------|------|--------|
| `plans/acp-transport-design.md` (this file) | Design-of-record, decisions, deferred milestones | Feature-level cousin of `bootstrap-mapping.md` |
| `plans/acp-conformance-research.md` | Primary-source research | Supporting note, not executable |
| `plans/bb-provider-integration-research.md` | How bb consumes ACP/providers | Supporting note for the bb milestone |
| `plans/implementation/acp-transport.md` | Ordered work packages, red tests, traps | `implementation/signal-transport.md` |

Implementers execute **only** the implementation plan. If design and WPs disagree, update the design first, then the WPs.

## Next step

Execute [implementation/acp-transport.md](./implementation/acp-transport.md) starting at **WP-A0.1** (red tests in `tests/test_acp_command.py`, then packaging + `acp_command`).
