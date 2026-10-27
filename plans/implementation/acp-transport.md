# ACP transport — stdio agent for editors

**Effort:** L. **Dependencies:** none in the phase sequence — orthogonal to
the Phase 2 appraisal/critique arc and can land beside it (same posture as
Signal).
**Normative references:** `plans/acp-transport-design.md` (decisions D1–D5,
plugin shape, capability roadmap); `plans/acp-conformance-research.md`
(conformance gates); `docs/design.md` "Channel System" and "Transports";
`docs/plugin-guide.md` "Channels" / "Registering CLI subcommands"; AGENTS.md
for TDD and exception discipline.

**Goal:** Corvidae speaks **ACP v1** over stdio as a first-party transport
plugin so editors (bb via `customAcpAgents`, Zed, …) can drive it; full v1
conformance is the destination, bb usability is the first product milestone.

**Status:** PHASE 2 DONE — WP-A2.1–A2.4 green (`loadSession` +
`session/load` / `resume` / `close` / `list`; `session/delete` deferred).
Deferred milestones M-CLIENT-TOOLS / M-REMOTE / M-REGISTRY / M-V2 stay in
the design until their trigger fires.

If this document and `acp-transport-design.md` disagree, **the design wins** —
update this file after.

## Read first

- `plans/acp-transport-design.md` — settled decisions, plugin shape, hook→ACP
  map, deferred milestones.
- `corvidae/channels/cli.py` — transport pattern: `on_start` / `send_*` with
  `matches_transport`, click command that builds `Runtime` with overrides.
- `corvidae/channels/irc.py` — inert-when-unconfigured pattern; backoff is
  irrelevant for stdio ACP but lifecycle discipline is the same.
- `plans/implementation/signal-transport.md` — off-sequence WP conventions this
  document follows (requirements, traps, ordered WPs, docs checklist).
- `corvidae/runtime.py` — how subcommands boot the plugin manager.
- `corvidae/main.py` + `pyproject.toml` — `corvidae` / `corvidae.commands`
  entry points.
- Official Python SDK: `agent-client-protocol` 0.12.x (`Agent`, `run_agent`,
  helpers); examples `echo_agent.py` / `agent.py` in the SDK repo.
- `tests/test_cli_plugin.py` / `tests/test_irc_plugin.py` — transport test
  patterns to mirror (in-process hooks; ACP adds optional subprocess smoke).

## Requirements

Tests in work packages cite these. Numbered for the ACP surface (not Signal’s
R-set).

### Process and packaging

- **A1** `corvidae acp [--config PATH]` starts an ACP agent on stdio and exits
  cleanly when the client closes the connection (or on fatal protocol error).
- **A2** While ACP mode runs, stdout carries **only** ACP JSON-RPC; logs go to
  stderr and/or a log file.
- **A3** `agent-client-protocol` is an optional extra (`acp`); missing install
  yields a clear error from the `acp` command, not an import crash of `serve` /
  `cli`.
- **A4** `AcpPlugin` is registered via the `corvidae` entry point group and is
  **inert** when not in ACP mode (no stdin/stdout ownership under `serve`/`cli`).
- **A5** Omitting `acp:` config or listing `acp` in `plugins.disabled` does not
  break other transports (plugin optionalism).

### Baseline ACP v1 (MVP)

- **A6** `initialize` negotiates protocol version 1, returns `agentInfo`,
  `agentCapabilities`, and `authMethods: []` (D4).
- **A7** `session/new` creates `Channel(transport="acp", scope=<sessionId>)`,
  honors `cwd` for the session.
- **A8** `session/prompt` (text content) enqueues via `on_message`; agent
  output streams as `session/update` (`agent_message_chunk`,
  `agent_thought_chunk`, tool call / update).
- **A9** The `session/prompt` JSON-RPC response is sent only when the turn is
  complete (`end_turn`), with cancel → `cancelled`.
- **A10** `session/cancel` interrupts in-flight turn / tool work for that
  session; no hang forever.
- **A11** Every `send_*` implementation broadcast-filters on
  `matches_transport("acp")`.

### Tools (Phase 1 / D3)

- **A12** Phase-1 ACP sessions use existing in-process tools (shell/files).
- **A13** Tool execution for ACP channels goes through a **swappable backend
  seam** so M-CLIENT-TOOLS can prefer client `fs`/`terminal` without rewriting
  session code.

### Conformance and interop

- **A14** Homegrown Gate B tests cover initialize, session/new,
  prompt→updates→end_turn, and cancel (extend as capabilities are advertised).
- **A15** bb milestone: document `customAcpAgents` wiring; a thread can
  complete a simple coding turn (manual or scripted smoke).

### Session durability (Phase 2 — after MVP)

- **A16** `session/load` (and other advertised session methods) restore from
  `sessions.db` keyed by `acp:<sessionId>` without forking history.

### Docs

- **A17** `docs/design.md`, `docs/plugin-guide.md`, `docs/configuration.md`,
  and `agent.yaml.example` describe ACP before the feature is declared done.

## Design constraints and traps

1. **Do not speak bb Provider Bridge Protocol** — only ACP; bb’s `provider-acp`
   adapts.
2. **Do not attach ACP stdio to `corvidae serve`** — D2; remote ACP is
   **M-REMOTE**, later.
3. **Broadcast-filter every `send_*`** — omitting it leaks CLI/IRC traffic
   into ACP sessions and vice versa.
4. **Turn completion ≠ `send_message` alone** — tools re-enter via `on_notify`;
   resolving `session/prompt` too early breaks clients. Track active prompt +
   pending tool ids per session.
5. **SDK only** — D1; no parallel JSON-RPC stack.
6. **Swappable tools from day one** — A13 / D3; a “temporary” hard-wire to
   local tools that cannot be redirected later is a bug.
7. **Entry-point load order is non-deterministic** — use `depends_on` /
   `get_dependency` for registry (and any later tool-router plugin).
8. **Hook params with defaults** — follow `tests/test_hook_arg_binding.py`
   if new hooks are ever added; prefer no new hookspecs for MVP.
9. **pytest timeouts** — all async tests inherit the global timeout; mark
   longer ACP subprocess tests explicitly if needed, do not raise the global.
10. **KV-cache / append-only invariants** — ACP must not invent mid-window
    mutation or `message_log` deletes.
11. **SDK Python ceiling** — `agent-client-protocol` 0.12.x declares
    `requires-python = ">=3.10,<3.15"`. Corvidae’s floor is `>=3.13`; stay on
    3.13/3.14 for the `acp` extra until the SDK raises its ceiling.
12. **Do not infer ACP mode from an `acp:` YAML block alone** — that would
    steal stdio under `serve`. Only `config["_acp_mode"]` from `acp_command`.

## Design (executor reference)

### Files

| Path | Role |
|------|------|
| `corvidae/channels/acp.py` | `AcpPlugin` + click `acp_command` + thin SDK `Agent` adapter (split only if file grows past ~400 lines) |
| `corvidae/tools/backends.py` | WP-A1.3 — `ToolBackend` protocol + `LocalToolBackend` |
| `tests/test_acp_command.py` | WP-A0.1 |
| `tests/test_acp_plugin.py` | WP-A0.2–A1.2 |
| `tests/test_acp_tool_backend.py` | WP-A1.3 |
| `tests/test_acp_conformance.py` | WP-A1.4 Gate B |
| `pyproject.toml` | optional-extra `acp`, entry points for plugin + command |

### Config / mode flag

```yaml
acp:
  agent_info:
    name: corvidae
    title: Corvidae
```

`acp_command` only:

```python
overrides={
    "_acp_mode": True,
    "logging": {"file": "corvidae-acp.log"},
}
```

### SDK pin and adapter surface

- Extra: `acp = ["agent-client-protocol>=0.12.1,<0.13"]` (import package `acp`).
- Implement on `acp.Agent` for Phase 0–1:

```python
async def initialize(...) -> InitializeResponse
async def new_session(cwd: str, ...) -> NewSessionResponse
async def prompt(session_id: str, prompt: list[...], ...) -> PromptResponse
async def cancel(session_id: str, ...) -> None
```

- Helpers: `run_agent`, `PROTOCOL_VERSION`, `text_block`, `update_agent_message`,
  plus thought/tool helpers from the SDK as needed.
- Channels: `registry.get_or_create("acp", session_id)` → `acp:<session_id>`.
  Store `cwd` in `channel.runtime_overrides["cwd"]` (lock this in WP-A1.1).

### Turn completion (A9)

Per session: `active_prompt` future + awareness of
`channel.pending_tool_call_ids`. Resolve `PromptResponse(stop_reason="end_turn")`
only when the turn is idle; cancel → `stop_reason="cancelled"`.

## Work packages (in order)

Red tests first, per AGENTS.md. Each package names the requirements its tests
cover.

### WP-A0.1 — Packaging and `acp` command stub (A1, A2, A3)

**Files:** `pyproject.toml`; create `corvidae/channels/acp.py` (command +
minimal stubs); `tests/test_acp_command.py`.

**Ship:**

- `[project.optional-dependencies] acp = ["agent-client-protocol>=0.12.1,<0.13"]`
- `[project.entry-points."corvidae.commands"] acp = "corvidae.channels.acp:acp_command"`
- `acp_command`: `--config` default `agent.yaml`; try `import acp` before
  `Runtime`; on `ImportError`, print install hint (`uv sync --extra acp`) and
  `sys.exit(1)`; else `Runtime(..., overrides={["_acp_mode"]: True, "logging": {"file": "corvidae-acp.log"}})` and `asyncio.run(runtime.run())`.

**Red tests (`tests/test_acp_command.py`):**

- `test_acp_help_lists_subcommand`
- `test_acp_missing_sdk_exits_with_hint` — patch import of `acp` to
  `ImportError`; non-zero exit; hint mentions `acp` extra
- `test_serve_and_cli_modules_do_not_import_acp` — loading serve/cli must not
  require the `acp` package

### WP-A0.2 — `AcpPlugin` inert + initialize (A4, A5, A6, A11)

**Files:** `AcpPlugin` + `CorvidaeAcpAgent` in `acp.py`;
`[project.entry-points.corvidae] acp = "corvidae.channels.acp:AcpPlugin"`;
`tests/test_acp_plugin.py`.

```python
class AcpPlugin(CorvidaePlugin):
    depends_on = frozenset({"registry"})
```

- `on_start`: return immediately unless `config.get("_acp_mode")`; else run
  ACP on stdio (injectable streams for tests).
- `initialize` → `InitializeResponse` with negotiated/`PROTOCOL_VERSION`,
  minimal `AgentCapabilities`, `Implementation` from config/`corvidae`
  version, empty auth methods.

**Red tests (`tests/test_acp_plugin.py`):**

- `test_on_start_without_acp_mode_starts_no_stdio_task`
- `test_initialize_handshake_in_process` — protocol 1, agent_info, empty auth
- `test_send_message_ignores_non_acp_channel`

### WP-A1.1 — Sessions and prompt text path (A7, A8, A9, A11)

**Files:** extend agent adapter; `tests/test_acp_plugin.py`.

- `new_session` → uuid/hex `session_id`; `get_or_create("acp", session_id)`;
  `runtime_overrides["cwd"] = cwd`.
- `prompt` → flatten text → `on_message`; `send_*` → `session_update`;
  `PromptResponse(stop_reason="end_turn")` after tool drain.
- Mock LLM via existing fixtures (`tests/llm_response_fixtures.py`); no network.

**Red tests:**

- `test_new_session_creates_acp_channel_with_cwd`
- `test_prompt_text_streams_agent_message_and_end_turn`
- `test_send_thinking_and_tool_status_map_to_session_updates`
- `test_prompt_waits_for_tool_drain_before_end_turn`

### WP-A1.2 — Cancel (A10)

**Red tests:**

- `test_cancel_during_prompt_yields_cancelled_stop_reason` — must finish under
  the test timeout
- `test_cancel_unknown_session_is_safe`

Implement the minimal channel-scoped interrupt the tests require; document any
new agent-loop hookup in the WP commit message / design if it escapes
`AcpPlugin`.

### WP-A1.3 — Swappable tool backend seam (A12, A13)

**Files:** `corvidae/tools/backends.py`; `tests/test_acp_tool_backend.py`.

```python
class ToolBackend(Protocol):
    async def run(self, name: str, args: dict, channel: Channel) -> str: ...

class LocalToolBackend:
    async def run(self, name: str, args: dict, channel: Channel) -> str: ...
```

Default for `acp:*` = `LocalToolBackend`. Tests inject a fake backend.

**Red tests:**

- `test_acp_channel_uses_tool_backend_seam`
- `test_non_acp_channel_unaffected`

### WP-A1.4 — Gate B harness + bb smoke docs (A14, A15, A17)

**Files:** `tests/test_acp_conformance.py`; docs checklist below.

Prefer in-process client↔agent (stay under 15s). Optional subprocess smoke with
explicit `@pytest.mark.timeout(30)` only if needed.

**Tests:**

- `test_gate_b_initialize_session_prompt_cancel` — initialize → new_session →
  prompt → cancel path (or sibling test for cancel)

**Manual A15 checklist** (record in docs or WP notes when done):

1. `uv sync --extra acp --extra dev`
2. `customAcpAgents` → `uv run corvidae acp`
3. `bb thread spawn --provider acp-corvidae --prompt "…"`
4. Confirm reply + tool rows in timeline

### Phase 2 — Session durability (A16)

Advertise **only** what these WPs implement. Keep D1–D5, `ToolBackend`,
`_acp_mode`, and broadcast-filters untouched.

**Ruling — defer `session/delete`:** python-sdk 0.12 `Agent` Protocol has no
`delete_session` (router does not dispatch `session/delete`). Deleting
`message_log` rows would also fight the append-only / KV-cache invariant
(trap 10). Do **not** set `sessionCapabilities.delete`. Revisit under a
later WP with an explicit tombstone / list-filter design.

**Session identity (non-negotiable):** `session/new` mints `sessionId`;
channel key is always `acp:<sessionId>`. `session/load` and `session/resume`
**reuse** that id via `registry.get_or_create("acp", session_id)` — never mint
a second id for the same conversation (A16: no forked history).

**cwd:** Protocol requires clients to send `cwd` on load/resume. Store it on
`channel.runtime_overrides["cwd"]` (already RUNTIME_META). For `session/list`
after process restart, WP-A2.3 persists an `acp_session_index` sidecar table
in the same `sessions.db` (cwd + updated_at). Do not invent mid-window
`message_log` mutation.

**History source:** `pm.ahook.load_conversation(channel=…)`
(`PersistencePlugin`). Leave `channel.conversation is None` after load/resume
so the agent loop’s lazy init reloads from DB on the next prompt (same rows,
no fork).

**Replay mapping (session/load only):**

| Stored message | ACP `session/update` |
|----------------|----------------------|
| `role=user` | `user_message_chunk` (`update_user_message_text`) |
| `role=assistant` | `agent_message_chunk` (`update_agent_message_text`) |
| `message_type=summary` | `agent_message_chunk` (summary `content` text) |
| `role=tool` / `role=system` | skip (client sees text turns; tools re-run live) |

Extract text from `content` when it is a string; if it is a list of parts,
concatenate `.get("text")` / `.text` fragments. Skip empty text.

**Unknown session:** no in-registry channel **and** `load_conversation`
returns `None` → raise `acp.RequestError.resource_not_found(uri=f"acp:{session_id}")`.

#### WP-A2.1 — `loadSession` + `session/load` replay (A16 core)

**Files:** `corvidae/channels/acp.py`; `tests/test_acp_session_durability.py`
(new).

**Ship:**

- `initialize` → `AgentCapabilities(load_session=True)` (keep `auth_methods=[]`).
  Do **not** yet set `session_capabilities` (resume/close/list land in
  WP-A2.2 / A2.3).
- `CorvidaeAcpAgent.load_session(cwd, session_id, …) -> LoadSessionResponse`:
  1. `get_or_create("acp", session_id)`; set `runtime_overrides["cwd"] = cwd`.
  2. `history = await plugin.pm.ahook.load_conversation(channel=channel)`.
  3. If `history is None` and this channel was not already in the registry
     before `get_or_create` (detect via `registry.get` **before** create, or
     equivalently: history None and no prior `message_log` / not a
     same-process live channel) → `RequestError.resource_not_found`.
     Practical rule for tests: call `registry.get(f"acp:{session_id}")`
     first; if missing, `get_or_create`, load history, and if history is
     still `None` → not found (empty brand-new id is not loadable). If the
     channel already existed in-registry (e.g. just created via
     `session/new`), empty history is OK (replay zero updates).
  4. For each replayable message, `await conn.session_update(session_id=…,
     update=…)`.
  5. Return `LoadSessionResponse()` (modes/config_options stay `None`).
- Do not resolve or create an active prompt; do not call `on_message`.

**Red tests (`tests/test_acp_session_durability.py`):**

- `test_initialize_advertises_load_session`
- `test_load_session_replays_user_and_assistant_from_db` — seed
  `message_log` for `acp:<id>` via PersistencePlugin (in-memory DB); call
  `load_session`; assert `session_update` order is user then assistant;
  assert channel id is exactly `acp:<id>` (same scope, no new uuid).
- `test_load_session_unknown_id_raises_resource_not_found`
- `test_load_session_same_process_empty_new_session_ok` — `new_session` then
  `load_session` with no rows → success, zero replay updates.
- `test_load_session_does_not_fork_channel_id` — after load, a follow-up
  `prompt` / registry lookup still uses the loaded id (not a freshly minted
  one).

#### WP-A2.2 — `session/resume` + `session/close`

**Files:** `corvidae/channels/acp.py`; extend
`tests/test_acp_session_durability.py`.

**Ship:**

- `initialize` also advertises
  `session_capabilities=SessionCapabilities(resume=SessionResumeCapabilities(),
  close=SessionCloseCapabilities())` (keep `load_session=True`).
- `resume_session(session_id, cwd, …) -> ResumeSessionResponse`: same
  reattach / cwd / unknown-session rules as load; **must not** call
  `session_update` for history.
- `close_session(session_id) -> CloseSessionResponse`:
  - `cancel_prompt(session_id)` (same as cancel).
  - Drop in-memory channel from `registry.channels` if present (free
    `conversation` / pending tools). **Do not** delete `message_log` rows.
  - Unknown id → `RequestError.resource_not_found` (not currently active).
- After close, `load_session` / `resume_session` of the same id must still
  restore from DB when rows exist.

**Red tests:**

- `test_initialize_advertises_resume_and_close`
- `test_resume_session_reattaches_without_replay`
- `test_resume_unknown_raises_resource_not_found`
- `test_close_session_cancels_prompt_and_drops_registry_entry`
- `test_load_after_close_restores_from_db`

#### WP-A2.3 — `session/list` + durable session index

**Files:** `corvidae/channels/acp.py` (or small helper module beside it if
`acp.py` exceeds ~400 lines); extend durability tests.

**Ship:**

- Advertise `session_capabilities.list=SessionListCapabilities()` (still no
  `delete` / `fork` / `additional_directories`).
- On `new_session`, `load_session`, and `resume_session`, upsert into table
  `acp_session_index` on PersistencePlugin’s DB (create-if-missing):

```sql
CREATE TABLE IF NOT EXISTS acp_session_index (
    session_id TEXT PRIMARY KEY,
    cwd TEXT NOT NULL,
    updated_at REAL NOT NULL
)
```

  Resolve DB via `get_dependency(self.pm, "persistence", PersistencePlugin)`
  when available; if persistence is absent, list returns only in-registry
  ACP channels (tests without PersistencePlugin stay meaningful).
- `list_sessions(cwd=None, cursor=None) -> ListSessionsResponse`:
  - Ignore/reject unknown cursors simply: Phase 2 returns the full list and
    `next_cursor=None` (no pagination required until volume demands it).
  - Optional `cwd` filter: exact string match on stored cwd.
  - Each `SessionInfo(session_id=…, cwd=…)`; `updated_at` as ISO-8601 UTC
    when known; `title` may be omitted.
- Index upsert failures: log + continue (fail-soft); do not break
  new/load/resume.

**Red tests:**

- `test_initialize_advertises_list`
- `test_list_sessions_includes_indexed_cwd`
- `test_list_sessions_filters_by_cwd`
- `test_list_sessions_survives_registry_clear` — upsert via new_session,
  drop registry entry, list still returns the session from the index.

#### WP-A2.4 — Gate B durability + docs (A14 extend, A16, A17)

**Files:** `tests/test_acp_conformance.py`; `docs/design.md`,
`docs/plugin-guide.md`, `docs/configuration.md` as needed;
`plans/implementation/README.md` status.

**Ship:**

- Extend Gate B (or sibling test) :
  `initialize` (assert `load_session`) → `session/new` → prompt/end_turn →
  simulate persistence → `session/load` → assert replay updates →
  `session/resume` / `session/close` smoke as implemented.
- Docs: document advertised session capabilities, `acp:<sessionId>` stability
  across load/resume, and that delete is not advertised.

**Red / update tests:**

- `test_gate_b_session_load_resume_close` (or extend existing Gate B)

### Phase 3+ / deferred milestones

| Milestone | Becomes |
|-----------|---------|
| M-CLIENT-TOOLS | WP-A3.x — client fs/terminal + permissions |
| M-REMOTE | Separate plan / WP-A9.x — HTTP/WS on `serve` |
| M-REGISTRY | WP-A8.x — Terminal Auth + registry entry |
| M-V2 | Separate plan after v1 Gate B green |
| (Phase 2 follow-up) | `session/delete` + tombstone — only when SDK routes it and append-only policy is settled |

## Docs checklist

### Phase 1 (done)

- [x] `docs/design.md` — ACP transport, channel id form, process model
- [x] `docs/plugin-guide.md` — `AcpPlugin`, command registration, inert rules
- [x] `docs/configuration.md` + `agent.yaml.example` — `acp:` keys + bb
      `customAcpAgents` pointer
- [x] `plans/implementation/README.md` — status → in progress / done

### Phase 2 (WP-A2.4)

- [x] `docs/design.md` — load/resume/close/list; stable `acp:<sessionId>`; no delete
- [x] `docs/plugin-guide.md` / `docs/configuration.md` — capability notes
- [x] `plans/implementation/README.md` — Phase 2 status

## Execution order

1. **WP-A0.1 → WP-A0.2** (Phase 0 skeleton) — done
2. **WP-A1.1 → WP-A1.2 → WP-A1.3 → WP-A1.4** (MVP + Gate B + docs) — done
3. Review / bb smoke (A15) — done
4. **WP-A2.1 → WP-A2.2 → WP-A2.3 → WP-A2.4** (session durability) — done

**Next action for an executor:** Phase 3 / M-CLIENT-TOOLS when the design
trigger fires (client `fs`/`terminal` + permissions). Until then, extend
Gate B only for newly advertised capabilities.

## Conventions note

Executable plans live under `plans/implementation/` in signal-transport style.
The writing-plans skill’s `docs/superpowers/plans/` path and checkbox
micro-steps are superseded by this repo; TDD and named red tests in each WP
remain mandatory.
