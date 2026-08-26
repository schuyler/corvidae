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

**Status:** PLANNED — Phases 0–1 work packages below are executable
(signal-transport bar: files, signatures, named red tests). Fill Phase 2+
only after Phase 1 green. Deferred milestones M-CLIENT-TOOLS / M-REMOTE /
M-REGISTRY / M-V2 stay in the design until their trigger fires.

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

### Phase 2 — Session durability (after MVP green)

#### WP-A2.x — load / resume / close / list / delete as advertised (A16)

Split into concrete WPs when Phase 1 is green; advertise only what we
implement.

### Phase 3+ / deferred milestones

| Milestone | Becomes |
|-----------|---------|
| M-CLIENT-TOOLS | WP-A3.x — client fs/terminal + permissions |
| M-REMOTE | Separate plan / WP-A9.x — HTTP/WS on `serve` |
| M-REGISTRY | WP-A8.x — Terminal Auth + registry entry |
| M-V2 | Separate plan after v1 Gate B green |

## Docs checklist (before declaring Phase 1 done)

- [ ] `docs/design.md` — ACP transport, channel id form, process model
- [ ] `docs/plugin-guide.md` — `AcpPlugin`, command registration, inert rules
- [ ] `docs/configuration.md` + `agent.yaml.example` — `acp:` keys + bb
      `customAcpAgents` pointer
- [ ] `plans/implementation/README.md` — status → in progress / done

## Execution order

1. **WP-A0.1 → WP-A0.2** (Phase 0 skeleton)
2. **WP-A1.1 → WP-A1.2 → WP-A1.3 → WP-A1.4** (MVP + Gate B + docs)
3. Stop for review / bb smoke before Phase 2

**Next action for an executor:** start **WP-A0.1** — write the three red tests
in `tests/test_acp_command.py`, run them (expect fail), then implement the
extra + `acp_command` until green.

## Conventions note

Executable plans live under `plans/implementation/` in signal-transport style.
The writing-plans skill’s `docs/superpowers/plans/` path and checkbox
micro-steps are superseded by this repo; TDD and named red tests in each WP
remain mandatory.
