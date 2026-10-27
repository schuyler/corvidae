# Kernel, cognition, and the hook grammar

This document specifies two changes: moving the cognition plugins into a
second distribution in the same repository, and giving the hook set a grammar
so that a hook's name tells you its resolution semantics. It is written for a
Rule of Two implementer: tagged requirements first, then the design, then
ordered work packages with red tests named. Status: implemented
2026-10-27. Proposed 2026-10-26, re-measured the next day (#117), then
landed in the order WP1, WP3, WP2:

- WP1, hook grammar test and five renames: `4f4220c` (#123, closes #97).
- WP3, `send` hook and `TransportPlugin`: `a40a4f5` (#124, closes #98).
- WP2, package split and the `discord` extra: `86c34fd` (#127, closes #38).
- WP4: the README tables, the R7 paragraph and the design.md architecture
  sentence landed with WP2; this status block is the rest.

Found and fixed on the way, as separate PRs: compaction and `on_idle` were
skipped for the first 30 s of machine uptime (`9aaf3ec`, #126), and the redact
CLI's `range` form crashed on every call and its `memory` form (without `--dry-run`)
crashed on a database without the memory schema (`e5f785f`, #128).

Where the result differs from the text below:

- The CLAUDE.md broadcast-filter bullet was rewritten, not deleted: inbound
  hooks still reach every transport and still need an ownership check.
- `resolve_single_result` logs and uses the first value; it does not raise.
- `redact` prints no skip notices on a database without the memory schema
  (Part 2 §2 assumed it did); design.md describes what it does.
- `harness/consolidate_replay.py` also imported a cognition module.
- The boundary allowlist includes `corvidae.outbound`, which WP3 added.

Not verified: R4's harness criterion (`harness/run.sh` needs a
llama-server) and WP2's `sessions.db` schema diff against a pre-split
database (no schema-creating code changed, but the diff was not run). The
other acceptance items are covered by tests in CI, or were checked by grep
and by the reviews on `86c34fd` (the retired-name and `matches_transport`
greps, the plugin-guide grammar section, the README tables). Still open:
switching `AcpPlugin` to `OutboundToolStatus.tool_call_id` (Open
questions), R8.

The rest of this document is the plan as approved, kept for its reasoning.

Every number in this document was measured against `618c831` (main,
2026-10-27), after the Discord, web and ACP transports and after #116 and
#100 landed. The first draft was measured at `3109d84`; re-running its
commands there reproduces its numbers exactly, so where this document
gives both, the difference is the tree moving, not the method. File counts
use `grep -rl --include='*.py'`; without the include, `__pycache__` adds one
or two. Re-measure before trusting them if the tree has moved.

## Why

The README calls corvidae "a minimalist, extensible architecture". That is
true of the architecture and false of the distribution, and the two have
drifted far enough apart that the sentence now misleads.

The kernel really is small. `Agent.depends_on` names four plugins: registry,
task, llm, tools. The agent loop, turn, queue, context window, channel,
hooks, tool, LLM client and task plugin come to roughly 4,100 of 19,000
source lines. A test (`tests/test_trace_core.py`, rule R1) greps every
non-cognition module for the words appraisal, critique, funnel, salience,
withheld and engagement and fails if any appear. The core is forbidden from
naming the cognition plugins even in a comment.

The distribution is not small. Memory, retention, appraisal, critique,
funnel, outcome_log and memory_tools are about 4,900 lines, own six of the
ten tables in `sessions.db`, and pull in sqlite-vec. They are on by default;
`agent.minimal.yaml.example` is a list of six things to switch off. The
README's claim that "without PersistencePlugin, messages drop but the daemon
doesn't crash" is no longer true: appraisal, scheduler, outcome_log,
usage_log and memory all declare `depends_on = {"persistence"}`, and
`validate_dependencies` aborts startup if a dependent is loaded without it.
The transports drifted the same way: ACP's SDK is an optional extra
(`--extra acp`), but `discord-py` is a hard dependency of every install,
used by one of six transports.

The hook set grew the same way. There are 31 hookspecs. Each is a cell in a
grid, a point in the pipeline crossed with one of a handful of resolution
semantics, but the names were chosen one at a time as a plugin needed each
cell, so the grid is invisible. Three verbs (`load_`, `resolve_`, `on_`) name
the same first-result semantics. `on_llm_error` sounds like an observation
and is actually a provider whose return value the core uses.
`on_conversation_event` sounds like an observation and is actually how the
core learns the rowid persistence assigned. `compact_conversation` is
declared `firstresult=True` in `hooks.py`, documented as the chain-stopping
replacement mechanism in `plugin-guide.md`, and described as broadcast in
three places in `design.md` and as `firstresult` in a fourth. Four `send_*` hooks have one shape, every transport has to
repeat the same broadcast filter in each, and `CLAUDE.md` carries a warning
about forgetting it.

Measured implementer counts, in-tree, excluding tests and the
`corvidae scaffold` template string:

| Hook | Implementers |
|---|---|
| `process_tool_result`, `on_llm_request` | 0 |
| `after_persist_assistant`, `transform_display_text`, `before_compaction`, `on_metrics`, `on_llm_error`, `load_conversation`, `compact_conversation` | 1 |
| `on_llm_response` | 2 (both in `metrics.py`) |
| `resolve_channel` | 2 (Discord, Signal) |
| `send_thinking`, `send_tool_status` | 3 (ACP, CLI, web) |
| `send_message`, `send_progress` | 6 (one per transport) |
| `on_init`, `on_start`, `on_stop` | 18 to 23 |

Counts here and in Part 1 §2 are `@hookimpl` methods under `corvidae/`,
found by walking the AST, excluding `scaffold.py` and the
`_SeedHooksPlugin` seeds.

None of this is a bug. It is what a working project looks like when it is
driven by one consumer. The fix is to make the structure explicit, and to put
the consumer in its own package so the kernel's boundary is a package
boundary rather than a grep.

## Requirements

Provenance tags: *stated* traces to Schuyler's words in the discussion that
produced this document; *inferred* is an assumption, individually vetoable;
*open* is a question nobody has answered yet. The user gates this list.

**R1 (stated).** A reader of `pyproject.toml` and the README can tell what the
kernel is and what is optional cognition, and installing corvidae without the
cognition extra yields a working daemon.
*So that* "minimalist" describes what you get, not only how it is built.
*Acceptance:* a CI job syncs without the cognition extra and runs every test
outside `tests/cognition/` green; `uv run corvidae` with
`agent.minimal.yaml.example` boots and answers a CLI message (the existing
`tests/test_minimal_profile.py` boot test, run in that job); `sqlite-vec` is
absent from the kernel's dependency list. (`discord-py` leaves too; that is
R10.)

**R2 (stated).** A hook's name tells you its resolution semantics without
reading `hooks.py`, and there is a written rule for when a new hook may be
added.
*So that* the hook set stops reading as arbitrary.
*Acceptance:* `tests/test_hook_grammar.py` passes: every hookspec name has a
grammar prefix or is on the imperative allowlist, and each prefix's pluggy
options match the grammar table. The test checks names against pluggy
options; whether the core reads a hook's return value is not something
pluggy exposes, so that half of the grammar is enforced by the admission
criterion at review time, not by the test. `docs/plugin-guide.md` states
the grammar and the criterion.

**R3 (stated).** The outbound transport surface has one broadcast filter, in
one place, that a transport author cannot forget.
*So that* the `CLAUDE.md` warning can be deleted rather than maintained.
*Acceptance:* none of the six transports (`AcpPlugin`, `CLIPlugin`,
`DiscordPlugin`, `IRCPlugin`, `SignalPlugin`, `WebPlugin`) contains a call
to `matches_transport` inside a send path; a test registers a transport that
implements only `send_message` on the base class and shows that thinking,
tool-status and progress events for another transport's channel never reach
it.

**R4 (inferred).** No behaviour change for an operator running the full
configuration: same `sessions.db` schema, same channel ids, same jsonl log
format, same replies.
*So that* existing deployments (the harness, Schuyler's daemon) keep working
with a `uv sync --extra cognition` and nothing else, plus `--extra discord`
for a deployment that uses Discord (R10).
*Acceptance:* the full test suite (2,278 collected at `618c831`; 1,925 at
`3109d84`) is green with every extra installed; `harness/run.sh` passes
`--extra cognition` and the harness pass criteria are unchanged.

**R5 (inferred).** Renamed hooks get no compatibility aliases.
*So that* `hooks.py` does not carry two names for one thing. The README
already warns that interfaces change without warning, and the only external
plugin template, `corvidae scaffold`, uses only the three lifecycle hooks
and needs no change.
*Veto this if* a plugin exists outside this repository that you want to keep
working across the rename.

**R6 (inferred).** One repository, one `uv.lock`, one CI workflow.
*So that* a hook change and its first consumer still land in one pull
request. A second repository was considered and rejected: the cognition
plugins are the reason most recent hooks exist, and an unversioned hook API
across two repositories turns every seam into a two-PR dance.

**R7 (inferred).** The README's optionality claims are corrected to match the
dependency graph.
*So that* the document does not promise a persistence-free daemon that
`validate_dependencies` refuses to start.

**R8 (open).** Whether to replace the observe-only hooks with a typed event
bus. Deferred; see Part 1 §5 for the go/no-go criterion. Nothing in this
document depends on the answer.

**R9 (open).** Whether `thinking.py` is kernel. This document says yes
(`SubagentPlugin` imports `strip_thinking`, `tests/helpers.py` registers it,
and it contains none of the R1 vocabulary), and removes it from the test's
`COGNITION_MODULES` set. Veto if you think of it as cognition.

**R10 (stated, 2026-10-27, #117).** `discord-py` leaves the kernel's
dependency list. The Discord transport stays in the `corvidae` package and
keeps its entry point, but its SDK installs with `--extra discord`, the way
ACP's installs with `--extra acp`.
*So that* the kernel install does not pull a chat client for one of six
transports, and R1's "minimalist" covers transports too.
*Acceptance:* after `uv sync --extra dev --extra acp`,
`import discord` fails and `import corvidae.channels.discord` succeeds;
the daemon boots with that install (the R1 boot test); a test shows that
with a `discord:` config block and no SDK, startup fails with a message
naming `uv sync --extra discord`. Part 2 §2a has the design.

## Decisions taken in discussion

- **Same repository, second distribution.** See R6.
- **Grammar by renaming, not by a new mechanism.** pluggy stays. The grammar
  is a naming convention enforced by a test, plus one consolidation (`send`).
- **The event bus is deferred, not rejected.** The first pass at this design
  proposed turning the conversation-log family into typed events. That was
  wrong on a fact: `on_conversation_event` returns the rowid persistence
  assigned, which the core writes into `_db_id`, so it is not an observation.
  The remaining candidates are small enough that the bus would be a second
  mechanism bought for three hooks. Revisit when the criterion in §5 trips.
- **`compact_conversation` keeps its chain-stop.** It is `firstresult=True`
  in `hooks.py`, `plugin-guide.md` documents a default-priority implementation that
  returns non-`None` as the way to replace the built-in compactor (and a
  `tryfirst` one returning `None` as the way to observe without replacing),
  and `tests/test_hook_dispatch.py` holds three tests for exactly that. The
  first draft of this document proposed dropping `firstresult` to match
  `design.md`; that would have been a plugin-API change dressed as a
  rename. The hook is renamed into the `resolve_` class instead, and the
  three `design.md` lines that say "broadcast" are corrected.
- **Logger names stay in the `corvidae.` namespace.** `corvidae/logging.py`
  configures the `"corvidae"` logger at the operator's level and the root at
  WARNING. Four cognition modules already name their logger explicitly
  (`corvidae.memory`, `corvidae.retention`, `corvidae.funnel`,
  `corvidae.tools.memory_tools`); three use `__name__`
  (`appraisal`, `critique`, `outcome_log`) and would silently drop to the
  root logger's WARNING after the move. WP2 gives those three explicit
  `corvidae.<name>` loggers. Nothing in a log line changes for an operator.
- **`corvidae redact` stays in the kernel.** It tombstones `message_log`
  rows, a kernel table, and cascades to `memory`, `memory_vec` and
  `message_fts` only after probing for them, printing a SKIP notice when
  they are absent. The first draft moved it on the false premise that it
  only touched the memory table. It imports no corvidae module, so it needs
  no boundary allowlist entry either.
- **The prompt copies do not move.** `prompts/*.md` are documentation
  copies; no code loads them, but `docs/configuration.md` and
  `agent.yaml.example` cite several of them by path. Leaving the directory
  where it is costs nothing and avoids a path churn.
- **Minimum commitment on the message-admission trio.**
  `on_message_admitted`, `on_message_rejected` and `on_message_persisted`
  have four, three and five implementers. `outcome_log` uses all three;
  `WebPlugin` uses all three to track each browser send through its
  lifecycle by trace id; `SignalPlugin` uses admitted and persisted (the
  typing indicator and read receipts at processing start) and since #116
  releases veto or never-admitted state in a `finally` around `on_message`
  instead of implementing rejected; `DiscordPlugin` uses rejected only;
  `critique` uses admitted and persisted, `appraisal` persisted. They are
  correctly named under the grammar and stay as they are.
- **Order WP1, WP3, WP2, WP4** (decided 2026-10-27, #117). The first draft
  put the send consolidation last. Its cost grows with every transport (it
  doubled from three to six between drafts) while the package split's cost
  has not moved (the import surface between kernel and cognition is
  byte-identical at both measurements). See Work packages.
- **Discord's SDK becomes an extra** (R10), folded into WP2.
- **`OutboundToolStatus` carries `tool_call_id`** (decided 2026-10-27).
  WP3 adds the field and passes the id through; it changes no behaviour.
  Switching `AcpPlugin` to use it is a separate pull request (Open
  questions).

# Part 1 — The hook grammar

## 1. The grammar

Every hookspec name starts with a prefix that fixes its pluggy options and
its resolution semantics, or is a bare imperative verb on a short allowlist.

| Prefix | pluggy spec | Resolution | Core uses result | Implementations |
|---|---|---|---|---|
| `on_`, `before_`, `after_` | broadcast | none | no | any number; side effects allowed, including on objects passed in |
| `should_` | broadcast | `resolve_reject_wins` | yes, `bool` | any number; `False` from any one rejects; `None` abstains |
| `resolve_` | `firstresult=True` | first non-`None` | yes | any number; first to answer wins, so order matters and the docstring must say who answers |
| `transform_` | `firstresult=True` wrapper chain with a seed in `_SeedHooksPlugin` | chain | yes | any number, LIFO |
| bare imperative | broadcast | caller-defined over the result list | sometimes | any number; the docstring states what the caller does with results |

`resolve_channel` shows the other safe shape for `resolve_`: its two
implementers (Discord and Signal) each answer only for ids under their own
`transport:` prefix and return `None` for the rest, so the order pluggy
calls them in cannot change the answer. A keyed `resolve_` hook's docstring
says what the key is.

The observe class gets three prefixes because `before_agent_turn`,
`before_compaction` and `after_persist_assistant` are observations placed
relative to a stage, and `on_before_compaction` reads worse than it is worth.
All three prefixes mean exactly the same thing to pluggy.

The rule that separates observe from transform: an observe hook may mutate an
object it was handed (`after_persist_assistant` strips `reasoning_content`
in place; `before_agent_turn` appends to `channel.conversation`), but its
return value is never read. A transform returns the replacement and the core
uses it. This matches current practice; it is written down so that nobody
adds a fourth way.

The bare imperative allowlist after this work is `send`, `register_tools`
and `persist_message`. `on_notify` is not on it: it
is an observe hook, producers announce and `Agent` reacts, nothing is
returned. `on_message` is the same shape.

## 2. Disposition of every hook

Counts are in-tree implementers at `618c831` (method in Why). "Keep"
means no rename.

| Current name | Class | Impls | Disposition |
|---|---|---|---|
| `on_init` | observe | 23 | keep |
| `on_start` | observe | 22 | keep |
| `on_stop` | observe | 18 | keep |
| `on_config_reload` | observe | 9 | keep |
| `on_idle` | observe | 2 | keep |
| `on_message` | observe | 1 (agent) | keep |
| `on_notify` | observe | 1 (agent) | keep |
| `should_process_message` | veto | 3 | keep; `should_` is the veto prefix, and Phase 2 already plans `should_send_response` |
| `on_message_admitted` | observe | 4 | keep |
| `on_message_rejected` | observe | 3 | keep |
| `on_message_persisted` | observe | 5 | keep |
| `before_agent_turn` | observe | 4 | keep |
| `on_agent_response` | observe | 2 | keep |
| `after_persist_assistant` | observe | 1 | keep; docstring already says in-place mutation is allowed |
| `transform_display_text` | transform | 1 | keep |
| `process_tool_result` | transform | 0 | **rename** `transform_tool_result`; it is a wrapper chain with a seed and the name hides that |
| `on_llm_error` | resolve | 1 | **rename** `resolve_llm_error`; `firstresult=True`, returns the error text the core sends |
| `load_conversation` | resolve | 1 | **rename** `resolve_conversation`; `firstresult=True`. Called by `agent.py` and, since ACP, by `channels/acp.py` for `session/load` |
| `resolve_channel` | resolve | 2 | keep; keyed by `transport:` prefix (§1) |
| `on_conversation_event` | imperative | 2 | **rename** `persist_message`; all implementations run (jsonl_log logs and returns `None`), the caller applies `resolve_single_result` and expects at most one rowid. The docstring must say that. |
| `compact_conversation` | resolve | 1 | **rename** `resolve_compaction`; keep `firstresult=True`. The chain stops at the first non-`None` return, which is how `plugin-guide.md` says to replace the built-in compactor (`FancyCompactionPlugin`). The core ignores the value (`agent.py` step 5), so "resolve" here means "resolve who compacts". Fix `design.md` lines 134, 614 and 840, which say broadcast. |
| `before_compaction` | observe | 1 | keep |
| `on_compaction` | observe | 3 | keep |
| `register_tools` | imperative | 7 | keep; implementers append to the list passed as `tool_registry` |
| `send_message` | imperative | 6 | **fold into `send`** (§3) |
| `send_thinking` | imperative | 3 | fold into `send` |
| `send_tool_status` | imperative | 3 | fold into `send` |
| `send_progress` | imperative | 6 | fold into `send` |
| `on_llm_request` | observe | 0 | keep; observability hooks come in request/response pairs and are for consumers outside the tree. This is the one zero-consumer hook retained deliberately; say so in its docstring |
| `on_llm_response` | observe | 2 | keep |
| `on_metrics` | observe | 1 | keep |

Net: 31 hooks become 28. Five renames, no spec-option changes, one
consolidation. Parameter lists and pluggy options do not change for any
renamed hook, so `TestHookspecSignatures` in `tests/test_trace_core.py`
and the three chain-stop tests in `tests/test_hook_dispatch.py` keep passing
once the names are substituted.

Rename blast radius, measured by `.py` files (docs: any file) that mention
the name, at `618c831`:

| Rename | `corvidae/` | `tests/` | `docs/` + README |
|---|---|---|---|
| `compact_conversation` → `resolve_compaction` | 3 | 13 | 3 |
| `on_conversation_event` → `persist_message` | 6 | 12 | 3 |
| `process_tool_result` → `transform_tool_result` | 3 | 3 | 3 |
| `load_conversation` → `resolve_conversation` | 4 | 5 | 3 |
| `on_llm_error` → `resolve_llm_error` | 4 | 6 | 2 |

These are mechanical: the hook name appears as a method name, a
`pm.ahook.<name>` call, an `AsyncMock` assignment or a docs table row. A
single `grep -rl` and `sed` per rename, then run the suite.

## 3. The outbound `send` consolidation

### What exists

Four hookspecs with one shape: broadcast to every transport, each transport
checks `channel.matches_transport("<name>")` and returns if the channel is
not its own. `AcpPlugin`, `CLIPlugin` and `WebPlugin` implement all four;
`DiscordPlugin`, `IRCPlugin` and `SignalPlugin` implement `send_message`
and `send_progress` only, so reasoning and tool status are silently not
shown there (which is the intended behaviour on a chat channel, and the
kind of fact that should be a no-op default rather than an absence).

Two transports deviate from the plain filter, and the base class has to
absorb both:

- `WebPlugin` filters through `_mine(channel)`, which is
  `self._enabled and channel.matches_transport("web")`. A base-class check
  on the transport name alone would drop the `_enabled` half. The base
  class therefore filters through an overridable `owns(channel)`, and
  `WebPlugin` overrides it with what `_mine` does today.
- `AcpPlugin` maps tool status onto ACP `tool_call` updates and needs a
  call id. `send_tool_status` does not carry one, so it mints a uuid on
  `dispatched` and keys it on `(channel.id, tool_name)`. Both call sites
  have the real id in scope (`call["id"]` in `agent.py`, `task.tool_call_id`
  in `task.py`), so the event carries it.

### What replaces it

One module, `corvidae/outbound.py`, with the event types:

```python
@dataclass(frozen=True, slots=True)
class OutboundMessage:
    text: str
    latency_ms: float | None = None

@dataclass(frozen=True, slots=True)
class OutboundThinking:
    text: str

@dataclass(frozen=True, slots=True)
class OutboundToolStatus:
    tool_name: str
    status: str                      # "dispatched" | "completed", as today
    args_summary: str | None = None
    result_summary: str | None = None
    tool_call_id: str | None = None  # the LLM's call id; None if unknown

@dataclass(frozen=True, slots=True)
class OutboundProgress:
    text: str

Outbound = OutboundMessage | OutboundThinking | OutboundToolStatus | OutboundProgress
```

One hookspec replacing the four:

```python
@hookspec
async def send(self, channel: "Channel", event: "Outbound") -> None:
    """Deliver an outbound event to the transport that owns ``channel``.

    Broadcast. Every transport receives every event; ``TransportPlugin.send``
    drops events for channels it does not own, so transports subclass it
    and never implement this hook directly.
    """
```

One base class in `corvidae/hooks.py`, next to `CorvidaePlugin`.
`hooks.py` has no module-level logger today (only `_pm_logger` and
`_resolve_logger`), so the snippet adds one:

```python
logger = logging.getLogger("corvidae.hooks")

class TransportPlugin(CorvidaePlugin):
    """Base for transports. Subclasses set ``transport`` and override the
    ``send_*`` methods they can display; the rest default to no-ops."""

    transport: ClassVar[str]
    depends_on = frozenset({"registry"})

    def owns(self, channel: "Channel") -> bool:
        """The broadcast filter. Override to narrow it (WebPlugin adds
        its enabled flag); never to widen it past ``transport``."""
        return channel.matches_transport(self.transport)

    @hookimpl
    async def send(self, channel: "Channel", event: "Outbound") -> None:
        # The one broadcast filter. Nothing below this line runs for a
        # channel another transport owns.
        if not self.owns(channel):
            return
        match event:
            case OutboundMessage():    await self.send_message(channel, event)
            case OutboundThinking():   await self.send_thinking(channel, event)
            case OutboundToolStatus(): await self.send_tool_status(channel, event)
            case OutboundProgress():   await self.send_progress(channel, event)
            case _:
                logger.warning("unknown outbound event %r", type(event).__name__)

    async def send_message(self, channel, event: OutboundMessage) -> None: ...
    async def send_thinking(self, channel, event: OutboundThinking) -> None: ...
    async def send_tool_status(self, channel, event: OutboundToolStatus) -> None: ...
    async def send_progress(self, channel, event: OutboundProgress) -> None: ...
```

The `send_*` methods keep their names but become plain methods, not
hookimpls, so a transport's code reads the same as before minus the filter.
`depends_on = frozenset({"registry"})` moves onto the base class, since all
six transports declare it. Inbound hookimpls that filter by transport
(`should_process_message` and the `on_message_*` trio in Discord, Signal
and web; eight methods) may call `self.owns(channel)` so that a transport
has one predicate, but WP3 does not fold them: they are observe and veto
hooks with distinct signatures, not one shape.

Call sites change from `pm.ahook.send_message(channel=c, text=t,
latency_ms=ms)` to `pm.ahook.send(channel=c, event=OutboundMessage(t, ms))`.
There are two in `corvidae/agent.py` for messages, one for thinking, one for
tool status, one for progress, and one in `corvidae/task.py` for tool
status; at `618c831` `grep -rn 'ahook.send_' corvidae` finds exactly
those six, and nothing in the subagent tool or the scheduler.

### Why this and not a smaller change

The filter moves from 24 places (four hooks across six transports; 18
written today and six missing) to one. A new outbound kind, say a
withheld-response marker from Phase 2's output gate, becomes a dataclass and
a no-op method rather than a hookspec, a seed and three transport edits. The
single `event` parameter also cannot be silently dropped by pluggy's
arg-binding bug, which the four separate signatures with their defaulted
`latency_ms` and `args_summary` can be (`_check_hook_arg_binding` exists
because of this).

### Blast radius

`send_message` is named in 23 test files (18 at `3109d84`), mostly as
`pm.ahook.send_message = AsyncMock()` in `tests/helpers.py` and then
`send_message.assert_called...` in the tests that use it. The helper changes
once; the assertions change to unpack the event. The larger half is now the
transport tests: 67 direct calls to a transport's `send_*` method across
nine files (`test_acp_plugin.py`, `test_acp_conformance.py`,
`test_acp_session_durability.py`, `test_cli_plugin.py`,
`test_cli_thinking_tools.py`, `test_discord_plugin.py`,
`test_irc_plugin.py`, `test_signal_plugin.py`, `test_web_plugin.py`; 35 at
`3109d84`). Each changes to pass an event. Two files,
`tests/test_integration.py` and `tests/test_hook_safety.py`, register fake
transports with `send_*` hookimpls of their own (four defs); once the specs
are gone those become spec-less hookimpls that never fire, so they must be
rewritten as `TransportPlugin` subclasses. This is the largest mechanical
edit in the document and the reason the send consolidation is its own work
package, droppable without affecting the others.

## 4. Admission criterion for new hooks

Written into `docs/plugin-guide.md` under the grammar, and enforced by
review rather than by test:

A hookspec earns a name when it is a decision point (veto, resolve,
transform), or when it is an observation with at least two plausible
implementers, or when it is an observability hook that pairs with an
existing one (`on_llm_request` with `on_llm_response`). A single-consumer
observation does not get a hookspec; it gets an entry in `meta`, a return
value on an existing hook, or waits for its second consumer.

Applied retroactively, the criterion is what §2 did: it kept everything with
a decision in it, folded the four sends, and left `on_llm_request` with an
explicit justification.

## 5. The event bus, deferred

The alternative to §1 is a single `on_event(event)` hookspec with typed
dataclass payloads, replacing every observe hook. It was considered and
deferred, not rejected.

What it would buy: adding an observation becomes adding a dataclass; the
arg-binding bug class disappears for observations; the spec file shrinks.

What it costs: two mechanisms (pluggy decisions, bus observations); plugins
dispatch on `isinstance`; pluggy stops validating observation argument names
against the spec; `TestHookspecSignatures` has to be rewritten against
dataclass fields; and the observe hooks most worth converting
(`on_message_admitted`/`rejected`/`persisted`) have four, three and five
implementers and carefully specified signatures from Phase 2 and #116.

Go criterion: build the bus when a third `send`-shaped consolidation turns
up, or when the observe class passes 20 hookspecs (it is 18 after this
work, and the hookspec set did not change between `3109d84` and
`618c831`). Until then the grammar does the job the bus would do, at the
cost of one prefix per hook.

The nearest candidate at `618c831` is not `send`-shaped: eight inbound
hookimpls in three transports open with the same transport filter
(`TransportPlugin.owns` gives them one predicate). If a fourth transport
adds inbound bookkeeping, that is the place to look again.

# Part 2 — The package split

## 1. Layout

A uv workspace. The root project stays `corvidae` and becomes the kernel.
The cognition plugins move to a workspace member the root depends on through
an optional extra.

```
corvidae/                         # kernel package, unchanged name
packages/
  corvidae-cognition/
    pyproject.toml                # name = "corvidae-cognition"
    corvidae_cognition/
      __init__.py
      memory.py
      retention.py
      appraisal.py
      critique.py
      funnel.py
      outcome_log.py
      memory_tools.py             # was corvidae/tools/memory_tools.py
prompts/                          # stays; documentation copies, cited by path from docs
scripts/eval_memory.py            # stays; it reads tests/evals and tests/fixtures by path
tests/
  ...                             # kernel tests, unchanged location
  evals/                          # metrics.py and test_metrics.py stay (shared with scripts/)
  cognition/
    __init__.py                   # tests/ is a package; keep it one
    ...                           # the 22 test files that import a cognition module,
    test_retrieval_benchmark.py   # including the benchmark from tests/evals/, placed flat
docs/
  design.md                       # kernel only after this work
  cognition.md                    # MemoryPlugin, FunnelPlugin, appraisal, critique, outcome_log
```

Root `pyproject.toml` changes:

```toml
[project]
dependencies = [ ... ]            # sqlite-vec and discord-py removed

[project.optional-dependencies]
dev = [ ... ]
acp = [ ... ]                     # unchanged
discord = ["discord-py>=2.4"]     # R10, §2a
cognition = ["corvidae-cognition"]

[tool.uv.workspace]
members = ["packages/*"]

[tool.uv.sources]
corvidae-cognition = { workspace = true }
```

`packages/corvidae-cognition/pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=75"]
build-backend = "setuptools.build_meta"

[project]
name = "corvidae-cognition"
version = "0.0.1"
requires-python = ">=3.13"
dependencies = ["corvidae", "sqlite-vec>=0.1.9"]

[project.entry-points.corvidae]
memory = "corvidae_cognition.memory:MemoryPlugin"
memory_tools = "corvidae_cognition.memory_tools:MemoryToolsPlugin"
funnel = "corvidae_cognition.funnel:FunnelPlugin"
outcome_log = "corvidae_cognition.outcome_log:OutcomeLogPlugin"
appraisal = "corvidae_cognition.appraisal:AppraisalPlugin"
critique = "corvidae_cognition.critique:CritiquePlugin"

[tool.uv.sources]
corvidae = { workspace = true }

[tool.setuptools.packages.find]
include = ["corvidae_cognition*"]
```

The `packages.find` block is not optional: setuptools' flat-layout
discovery refuses to build a project whose top level holds more than one
package-shaped directory, and the root `pyproject.toml` already carries the
same guard (`include = ["corvidae*"]`). The member's top level holds only
`corvidae_cognition/`, but the explicit include keeps a later `tests/` or
`docs/` directory from breaking the build.

The dependency graph is a cycle: `corvidae[cognition]` depends on
`corvidae-cognition`, which depends on `corvidae`. uv resolves cycles
between workspace members, but WP2 step 2 proves it with `uv lock` before
anything else moves; if it does not resolve, the fallback is to drop
`corvidae` from the member's dependency list (the extra always co-installs
it) and note that the member is not installable on its own.

`Runtime.start()` calls `pm.load_setuptools_entrypoints("corvidae")`, which
scans every installed distribution, so the moved entry points load exactly
as before. Entry-point names do not change, so `plugins.disabled` lists in
existing `agent.yaml` files keep working. With the extra not installed, the
six names in `agent.minimal.yaml.example` block nothing, which is already
documented as harmless.

## 2. What moves, and why it can

Everything the cognition modules import from the kernel, measured:

| Kernel module | Imported names | Importers |
|---|---|---|
| `hooks` | `CorvidaePlugin`, `hookimpl`, `get_dependency`, `resolve_single_result` | all but retention |
| `fork` | `fork_agent`, `quote`, `window_db_ids` | memory, appraisal, critique |
| `attribution` | `set_attribution`, `reset_attribution` | memory, appraisal, critique, retention |
| `tuning` | `config_value`, `resolve_tunable` | appraisal, critique |
| `context` | `MessageType`, `ContextWindow`, `count_tokens` | memory, funnel, critique |
| `task` | `Task` | appraisal, critique |
| `tool` | `Tool`, `tool_to_schema` | memory_tools |
| `persistence` | `PersistencePlugin`, `add_column_if_missing` | memory, outcome_log |
| `llm_plugin` | `LLMPlugin` | memory |

The table was built from every `from corvidae.` line in the moved modules,
including function-local imports; re-run
`grep -rhoE 'from corvidae\.[a-z_.]+ import [A-Za-z_, ]+' <moved files> | sort -u`
before writing the allowlist, since a local import is easy to miss.

No cognition module touches a private attribute of `Agent`, `Channel` or the
registry. Each owns its tables (`memory`, `memory_fts`, `memory_vec`,
`embedding_meta`, `consolidation_watermark`, `retention_meta`,
`retrieval_log`, `exchange_log`). One thing crosses the line: `memory.py`
creates `message_fts` and two triggers, `message_log_ai` and
`message_log_au`, on the kernel's `message_log` table. The kernel never reads
them, so this is not a dependency, but it is a cognition-owned object living
on a kernel table; see Traps. `memory_tools` imports `ChannelScope`,
`MemoryPlugin`, `_format_age` and `_fts_match_query` from `memory`, which is
fine once both are in the same package. `memory` lazily imports
`retention`; both move.

That table is the kernel's plugin API. `docs/plugin-guide.md` already
documents attribution, `fork_agent`, `window_db_ids` and `config_value`;
after this work it also lists the table above under a heading "What the
kernel exports to plugins", and `tests/test_kernel_boundary.py` (§4)
enforces that `corvidae_cognition` imports nothing else.

Dependencies that leave the kernel: `sqlite-vec` (memory and retention
only), and `discord-py` to its own extra (§2a). `tiktoken` stays
(`context.count_tokens`), `ddgs` stays (`tools/web.py`).

`corvidae/commands/redact.py` does not move (Decisions). Its memory cascade
is already guarded by a schema probe, so on a kernel-only database it
tombstones `message_log` rows and reports that the cascade was skipped.
The `redact` section of `design.md` is today nested under MemoryPlugin;
when that section moves to `cognition.md`, re-parent the `redact` section
to the kernel side and give its cascade paragraph a line saying the memory
surfaces come with the extra.

`tests/test_redact.py` is one of the 23 grep matches because it imports
`MemoryPlugin` at module level for the cascade tests, yet it is the only
test file for a kernel command. Split it: the tests that need no memory
plugin (the pre-1b database case, `_extract_sample_token`,
`_verify_fts_clean` on absent tables) stay in `tests/test_redact.py` with
the `corvidae.memory` import removed, and the cascade tests move to
`tests/cognition/test_redact_cascade.py`. So the 22 "move" files are 21
moves and one split.

## 2a. Discord to an extra (R10)

The Discord transport stays where it is (`corvidae/channels/discord.py`,
entry point `discord`); only its SDK becomes optional. That rules out a
module-level `import discord`: `Runtime.start()` loads every entry point
with `ep.load()`, which in pluggy 1.7.0 has no exception handling, so an
`ImportError` in that module would abort startup for every install
without the extra, Discord configured or not.

What uses the SDK today is small: `import discord` at line 20, the
module-level `_DiscordClient(discord.Client)` subclass, and ten `discord.`
references, all reached through `_make_client()`. The change follows
`AcpPlugin`, which imports `acp` inside the functions that need it:

- Move `import discord` and the `_DiscordClient` class body into a
  function that `_make_client()` calls, so the module imports without the
  SDK. Type annotations that name `discord.Message` become strings or
  `Any`.
- When a `discord:` block is configured and the import fails, fail
  startup with the error style #115 set: what failed, which plugin, and
  the next step (`uv sync --extra discord`). `acp.py`'s `except
  ImportError` branch (the `acp_command` path) is the existing example.
  With no `discord:` block the plugin stays inert and never imports the
  SDK.
- `tests/test_discord_plugin.py` imports `discord` at module level; it
  gains `pytest.importorskip("discord")` so the kernel job skips it, and
  the full job's install check (§5) makes sure that skip never fires
  there.

`docs/configuration.md`'s Discord section and `docs/discord-ops.md` get the
install line.

## 3. The vocabulary test becomes a boundary test

`tests/test_trace_core.py` R1 currently exempts a hand-maintained
`COGNITION_MODULES` set. After the split the set is empty: every file under
`corvidae/` is kernel and must pass the vocabulary grep. The set is deleted
and `_core_python_files()` yields everything. This also resolves R9 for
`thinking.py`, which contains none of the words.

A new `tests/test_kernel_boundary.py` adds two static checks that do not
need the cognition extra installed:

1. No file under `corvidae/` imports `corvidae_cognition`.
2. If `packages/corvidae-cognition/` exists on disk, every `from corvidae.X
   import` in it has `X` on the allowlist from §2. The test reads source
   text, so it runs in the kernel-only job too.

## 4. Tests

Pytest's rootdir stays the repository root; `tests/conftest.py` and
`tests/aiosqlite_guard.py` apply to `tests/cognition/` by inheritance, so
nothing about fixtures changes. `pythonpath = ["tests", "."]` keeps
`from helpers import ...` and `from llm_response_fixtures import ...`
working from the subdirectory.

The files to move are the ones matching
`grep -rlE 'corvidae\.(memory|appraisal|critique|funnel|outcome_log|retention)|corvidae\.tools\.memory_tools' tests --include='*.py'`,
which is 23 files at `618c831` (the same list as at `3109d84`): 22 move
and one
(`test_depends_on_declarations.py`, below) is adjusted in place. The 22
include `tests/test_phase2a_wp21.py`, which matches only through a
function-local import and moves anyway, and
`tests/evals/test_retrieval_benchmark.py`, which goes flat into
`tests/cognition/` rather than into a `tests/cognition/evals/` subpackage,
so that there is one package named `evals` under `tests/`, not two.
`tests/evals/metrics.py` and `test_metrics.py` stay: `scripts/eval_memory.py`
reads them by path, and the benchmark's `from evals.metrics import` keeps
resolving through `pythonpath = ["tests"]`.

The module rename is import-only. Tests patch kernel modules by dotted
string in about a hundred places (`corvidae.runtime.validate_dependencies`,
`corvidae.agent.run_agent_turn`, and so on), but none patches a cognition
module: `grep -rnE 'patch\("corvidae\.(memory|appraisal|critique|funnel|outcome_log|retention|tools\.memory_tools)' tests`
is empty at `618c831`. The cognition dotted strings that do appear are nine `caplog.at_level(..., logger="corvidae.memory")`
and `logger="corvidae.funnel"` lines in `test_memory_schema.py`,
`test_memory_consolidation.py` and `test_funnel.py`; those name loggers, not
modules, and stay exactly as they are because logger names do not change
(Decisions). Do not run a blanket `sed` for `corvidae.memory`; change the
`from corvidae.memory import` lines and leave the logger strings alone.

Three moved files compute paths from `__file__`; two break one level
deeper and one does not (the fixtures stay in `tests/fixtures/`, the
harness files in `harness/`):

| File | Line | Fix |
|---|---|---|
| `tests/test_memory_retrieval.py` | `Path(__file__).parent / "fixtures" / "memory_retrieval_basic.json"` | one more `.parent` |
| `tests/test_driver_memory_role.py` | `REPO = Path(__file__).resolve().parent.parent` | one more `.parent` |
| `tests/evals/test_retrieval_benchmark.py` | `FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"` | none: it moves from `tests/evals/` to `tests/cognition/`, the same depth |

The benchmark reads its fixture without an existence skip, so if it is
placed anywhere but flat in `tests/cognition/`, every benchmark test errors
in the full job rather than skipping.

Two kernel tests reference cognition modules by name and must be adjusted
rather than moved:

- `tests/test_depends_on_declarations.py` parametrizes over
  `_REGISTERED_PLUGIN_CLASSES`, which names `corvidae.memory.MemoryPlugin`
  and `corvidae.funnel.FunnelPlugin`. Those two rows move to a sibling test
  in `tests/cognition/`.
- `tests/test_minimal_profile.py` disables the six cognition plugins by name
  and asserts the daemon boots. It stays in the kernel suite as the R1 boot
  test; with the extra absent the disable list is a no-op and the test still
  proves the boot.

`tests/test_driver_scheduler.py` mentions `plugins.disabled` but imports no
cognition module; it stays. `tests/test_acp_stdio_smoke.py` disables the
six cognition plugins by name in its generated config, like the minimal
profile; with the extra absent that list is a no-op, and it stays.

## 5. CI

`.github/workflows/tests.yml` gets two jobs:

- **kernel**: `uv sync --locked --extra dev --extra acp`, then
  `uv run pytest -q --ignore=tests/cognition`. No sqlite-vec check. The
  `acp` extra is in because the ACP tests import the SDK and CI already
  installs it; `discord` is out, so this job also proves R10. This job is
  R1's acceptance test.
- **full**: `uv sync --locked --extra dev --extra acp --extra discord
  --extra cognition`, the existing sqlite-vec load check, a one-line
  `import discord` check beside it (so the Discord tests' importorskip
  cannot pass silently), then `uv run pytest -q`.

`uv.lock` stays single; a workspace has one lock. `harness/run.sh` line 113
becomes `uv sync --project "$REPO_DIR" --extra cognition` (the harness
drives IRC, so it needs no `discord`). The CLAUDE.md sync line (`uv sync
--extra dev --extra acp` once #120 lands) becomes `uv sync --extra dev
--extra acp --extra discord --extra cognition`, with a sentence saying
the kernel alone is `--extra dev --extra acp`.

## 6. Config, README, docs

- `README.md`: the one-liner stays. The "Included plugins" table splits into
  "Kernel plugins" and "Cognition plugins (`--extra cognition`)". The
  paragraph claiming the daemon runs without PersistencePlugin is replaced
  by one sentence pointing at the dependency graph in `design.md` (R7).
- `agent.minimal.yaml.example`: the header comment says the kernel without
  the extra needs no disable list, and that the list is kept for installs
  that have the extra. `tests/test_minimal_profile.py` asserts the file's
  text contains "memory" and either "appraisal" or "critique"; keep those
  words in the rewritten header.
- `docs/design.md`: the sections MemoryPlugin, FunnelPlugin, and the
  cognition-specific parts of the dependency graph move to
  `docs/cognition.md`. The "Current dependency graph" block stays in
  `design.md` with a line for each cognition plugin marked "(cognition
  extra)". The Hook reference table is regenerated from the disposition in
  Part 1 §2.
- `docs/plugin-guide.md`: a new "Hook grammar" section with the table from
  Part 1 §1 and the admission criterion from §4; the "Available hooks"
  tables renamed; a "What the kernel exports to plugins" list from Part 2
  §2; the `TransportPlugin` base documented under Channels; the per-plugin
  sections for the cognition plugins move to `docs/cognition.md`.
- `docs/configuration.md`: `memory`, `funnel` and the cognition keys under
  `agent.forks.stages` get a "requires the cognition extra" note. No key
  changes; the `prompts/*.md` paths it cites stay valid because the prompts
  do not move.
- `docs/threat-model.md`: no change in what untrusted text can reach; add a
  line that the memory tables only exist with the extra and that `redact`
  skips its memory cascade without them.
- `README.md` Installation: `uv sync --extra cognition` for the full
  daemon (plus `--extra discord` to use Discord), `uv sync` for the
  kernel; the development line becomes
  `uv sync --extra dev --extra acp --extra discord --extra cognition`.
- `.claude/hooks/session-start.sh`: its sync line (`--extra dev --extra
  acp` once #120 lands) gains `--extra discord --extra cognition`, or
  every web session's `uv run pytest` errors on the `tests/cognition/`
  imports and skips the Discord tests.
- `CLAUDE.md`: update the sync command and delete the "every `send_*` hook
  must broadcast-filter" bullet once WP3 lands.

## 7. Traps

- **Entry-point duplication is silent.** pluggy's
  `load_setuptools_entrypoints` skips an entry point whose name is already
  registered (the `has_plugin(ep.name)` branch of its skip condition), so a stale
  `memory = "corvidae.memory:MemoryPlugin"` left in the root
  `pyproject.toml` either loads nothing or, if iterated first, raises
  `ImportError` on the moved module, depending on distribution order. Do not
  rely on pluggy. The WP2 runtime test asserts each cognition entry point's
  `value` starts with `corvidae_cognition.`.
- **`message_fts` and its triggers outlive the extra.** `memory.py` installs
  `message_fts` and the `message_log_ai`/`message_log_au` triggers on the
  kernel's `message_log`. They live in `sessions.db`, not in code, so an
  operator who removes the extra keeps them and the triggers keep firing
  into a table nothing reads. Harmless, and the kernel-only job starts from
  an empty database so it never sees them, but say so in `docs/cognition.md`
  and in the `redact` docs.
- **Local syncs can hide a kernel-only break, and a plain sync hides the
  extra.** `uv sync` is exact by default and removes `corvidae-cognition`
  when the extra is not requested, but `--inexact` keeps it, and a stale
  `.venv` is easy to carry around. The reverse also bites: every `uv sync`
  line in the tree (`README.md`, `CLAUDE.md`, `harness/run.sh`,
  `.claude/hooks/session-start.sh`) must grow `--extra cognition` or the
  daemon it produces is the kernel. The
  kernel CI job is the check of record for R1, not a local run.
- **The vocabulary grep now covers every kernel file.** `retention.py`
  moves out; `thinking.py` stays and passes today. If a future kernel change
  trips the grep, rename the thing; do not re-add an exemption set.
- **A module-level SDK import in a transport aborts every install
  without the extra.** pluggy loads every entry point eagerly and does not
  catch `ImportError` (§2a). `AcpPlugin` and, after WP2, `DiscordPlugin`
  import their SDKs lazily; a new transport with an optional SDK must too.
  WP2's test imports `corvidae.channels.discord` with `discord` blocked in
  `sys.modules` and asserts it succeeds.
- **`add_column_if_missing` is now public.** It is imported by two cognition
  plugins; a signature change is an API change and needs a docs line.
- **`__name__` loggers change meaning when a module moves.**
  `appraisal.py`, `critique.py` and `outcome_log.py` call
  `logging.getLogger(__name__)`; after the move that is
  `corvidae_cognition.appraisal`, which `corvidae/logging.py` does not
  configure, so their INFO and DEBUG lines vanish from the operator's log and
  `%(name)s` changes. WP2 replaces the three with explicit
  `corvidae.appraisal`, `corvidae.critique`, `corvidae.outcome_log` names
  and adds a test that every module under `corvidae_cognition` except
  `__init__` has a module-level `logger` whose name starts with `corvidae.`
  and not with `corvidae_cognition`. The comments in those modules that point at
  `prompts/*.md` as "the documented copy" stay correct because the prompts
  stay put.
- **Pluggy order with `firstresult`.** `resolve_conversation` and
  `resolve_llm_error` keep `firstresult=True`; if a cognition plugin ever
  implements one, registration order decides who answers. The grammar table
  says this; the docstrings must say who is expected to answer.

# Work packages

Each work package is a pull request. Red tests first, then green, then docs,
then a cold review, per `plans/implementation/README.md` conventions and
AGENTS.md. Effort letters are relative: S is an afternoon, M is a day or two,
L is more.

Order: WP1, WP3, WP2, WP4 (decided 2026-10-27, #117). The packages keep
their first-draft numbers so that `PENDING_WP3` and the cross-references
stay valid; WP3 simply runs second.

WP1 and WP2 both touch the cognition modules (WP1 renames hook methods in
them, WP2 moves them), so WP1 goes first. WP3 touches transports,
`agent.py`, `task.py` and `hooks.py`, none of which WP2 moves; on the test
side the two overlap in one file (`tests/test_phase2a_wp21.py` is in both
WP2's move list and WP3's `send_message` list), which is a conflict to
take on rebase, not a reason to order them. WP3 goes before WP2 because
its cost grows with each transport (three at `3109d84`, six at `618c831`)
while WP2's has not moved, and because it shortens the time the WP1
strict xfails sit in the tree. Whichever runs later re-greps on its own
base rather than trusting the counts here. WP4 is docs that each earlier
package already updated, plus the parts that only make sense once all
three are in.

## WP1 — Hook grammar: test and renames (M)

**Red.** `tests/test_hook_grammar.py`:

```python
GRAMMAR = {
    "observe":    (("on_", "before_", "after_"), {"firstresult": False}),
    "veto":       (("should_",),                 {"firstresult": False}),
    "resolve":    (("resolve_",),                {"firstresult": True}),
    "transform":  (("transform_",),              {"firstresult": True}),
}
IMPERATIVE = {"send", "register_tools", "persist_message"}
PENDING_WP3 = {"send_message", "send_thinking", "send_tool_status", "send_progress"}
```

Tests, using `create_plugin_manager()` and walking
`pm.hook.__dict__.values()` the way `_check_hook_arg_binding` does (pluggy
1.7.0: `caller.spec.opts["firstresult"]` and
`caller.get_hookimpls()[i].trylast` are both available). Every test is
parametrized per hook name so that a single row can be marked
`xfail(strict=True)` for the gap between WP1 and WP3:

1. Each hookcaller name matches a grammar prefix or is in `IMPERATIVE`. The
   four `PENDING_WP3` rows are `xfail(strict=True)`.
2. For each matched prefix, `caller.spec.opts["firstresult"]` equals the
   table.
3. Each `transform_` hook has a `trylast=True` seed implementation on
   `_SeedHooksPlugin`.
4. Each name in `IMPERATIVE` has a hookcaller with `firstresult=False`. The
   `send` row is `xfail(strict=True)` until WP3 adds the spec.
5. Each old name (`on_conversation_event`, `process_tool_result`,
   `load_conversation`, `on_llm_error`, `compact_conversation`, and the four
   `PENDING_WP3` names) is absent from `AgentSpec`. The four `PENDING_WP3`
   rows are `xfail(strict=True)`.

WP3 deletes `PENDING_WP3` and every xfail marker; from then on the file has
no expected failures.

Test 2 only checks names that matched a prefix in test 1; a name with no
prefix is test 1's failure, not test 2's, so test 2 skips it rather than
failing twice for one cause.

At `618c831` (as at `3109d84`) the non-xfail rows fail as follows: test 1 on
`process_tool_result`, `load_conversation` and `compact_conversation`; test
2 on `on_llm_error` (it matches `on_` and is `firstresult=True`); test 4 on
`persist_message` (no hookcaller yet); test 5 on all five old names. Test 3
passes today for `transform_display_text` and is a guard for
`transform_tool_result`.

`on_conversation_event` is caught by test 5 alone. It starts with `on_` and
is a plain broadcast spec, so tests 1 and 2 pass for it today and would pass
for any `on_`-prefixed hook whose result the core happens to read. That is
the one half of the grammar a test cannot see (R2 acceptance); the rename
to `persist_message` is a review decision recorded in §2, and test 5 pins
it.

**Green.**

- `hooks.py`: rename the five specs, changing no pluggy options; update
  `_SeedHooksPlugin` for `transform_tool_result`; update every docstring to
  name its class, and for `resolve_` hooks, say who is expected to answer.
- `grep -rl` and `sed` for each rename across `corvidae/`, `tests/`,
  `docs/`, `README.md`, `CLAUDE.md`, `corvidae/scaffold.py` (template text),
  and `harness/` (check; probably none). `resolve_compaction` is the widest
  (13 test files) and all seven tests in `tests/test_hook_dispatch.py`
  (three on compaction, two on `load_conversation`, two on `on_llm_error`)
  must still pass after the substitution, unchanged in what they assert.
- `docs/design.md` lines 134, 614 and 840: `resolve_compaction` is
  `firstresult=True`; the first non-`None` return stops the chain. The
  worked `FancyCompactionPlugin` example in `plugin-guide.md` is renamed,
  not rewritten.
- `persist_message` docstring: "Broadcast. Every implementation runs.
  `PersistencePlugin` returns the `message_log` rowid; every other
  implementation returns `None`. The caller resolves with
  `resolve_single_result`, which logs an error and uses the first value
  if two implementations answer." (The first draft said "raises"; the
  code logs, and WP1 documented the code.)

**Docs.** `plugin-guide.md`: new "Hook grammar" section (Part 1 §1 table and
§4 criterion); rename rows in "Available hooks". `design.md`: Hook reference
table rows. CLAUDE.md: nothing yet.

**Acceptance.** Full suite green; `test_hook_grammar.py` green except the
strict xfail rows listed above;
`grep -rn --exclude-dir=fixtures 'on_conversation_event\|process_tool_result\|load_conversation\|on_llm_error\|compact_conversation' corvidae tests docs README.md CLAUDE.md`
returns only `tests/test_hook_grammar.py`, which names the retired hooks
to assert they are gone (historical mentions in `plans/` are fine, and
`tests/fixtures/` holds recorded transcripts that keep the old names).

## WP2 — Package split and Discord extra (M–L; runs third)

**Red.**

- `tests/test_kernel_boundary.py` as in Part 2 §3. Check 1 passes
  trivially today and is a guard, not a red test. Check 2 skips when
  `packages/corvidae-cognition/` is absent and runs against its source once
  WP2 creates it; write it before the move so it goes red the moment a
  moved module imports something off the allowlist.
- Emptying `COGNITION_MODULES` in `test_trace_core.py` is the red
  state for the vocabulary rule: prove the test can fail by temporarily
  adding the word "appraisal" to a kernel comment, watch it fail, revert.
- A test in `tests/test_runtime.py` that reads
  `importlib.metadata.entry_points(group="corvidae")` and asserts: when
  `corvidae_cognition` is importable, the six cognition names are present
  and each one's `value` starts with `corvidae_cognition.`; when it is not
  importable, none of the six names is present. Red today on the `value`
  prefix (every value starts with `corvidae.`). The prefix assertion is
  what catches a stale root entry point, since pluggy will not (Traps).
- A test in `tests/cognition/test_logger_names.py`: import every module
  under `corvidae_cognition` except `__init__`, assert each has a
  module-level `logger` whose `name` starts with `corvidae.` and not
  `corvidae_cognition`. This cannot
  be red before the move (the modules do not exist under that name yet);
  it goes red the moment step 1 lands without step 3b, which is the point.
- CI: the kernel job itself is the red test for R1; it cannot pass until
  `sqlite-vec` is out of the root dependencies and the moved tests are in
  `tests/cognition/`.
- `tests/test_discord_plugin.py` (or a sibling): import
  `corvidae.channels.discord` with `sys.modules["discord"] = None` and
  assert it succeeds; build a runtime with a `discord:` block and no SDK
  and assert startup fails with a message naming `uv sync --extra
  discord`. Red today on the first: the module imports `discord` at line
  20.

**Green.**

1. `git mv` the seven modules to `packages/corvidae-cognition/corvidae_cognition/`.
   `prompts/` and `scripts/eval_memory.py` stay where they are; the script
   resolves `tests/evals` and `tests/fixtures` relative to the repository
   root and only its imports change (`corvidae.memory`, `corvidae.funnel`,
   `corvidae.retention`).
2. Write the member `pyproject.toml` (Part 2 §1, including
   `[build-system]` and `[tool.setuptools.packages.find]`). Edit the root
   `pyproject.toml`: remove the six cognition entry points (the `redact`
   command entry point stays); remove `sqlite-vec`; add the extra, the
   workspace and the source. Run `uv lock` and confirm the cycle resolves (Part 2 §1);
   commit `uv.lock`. Then `uv sync --extra dev --extra cognition` and
   `uv run python -c "import corvidae_cognition.memory"` before going on.
3. Fix imports inside the moved modules (`from corvidae.memory` →
   `from corvidae_cognition.memory`, etc.).
3b. Replace `logging.getLogger(__name__)` in `appraisal.py`, `critique.py`
   and `outcome_log.py` with `logging.getLogger("corvidae.appraisal")`,
   `"corvidae.critique"`, `"corvidae.outcome_log"`. Leave the four explicit
   names as they are.
4. Create `tests/cognition/__init__.py`. `git mv` the 22 test files
   (Part 2 §4) flat into `tests/cognition/`, the retrieval benchmark
   included, with `tests/test_redact.py` split as §2 says rather than
   moved. Change their `from corvidae.<x> import` lines only; fix the
   two `__file__`-relative paths the §4 table marks. Split the two
   `_REGISTERED_PLUGIN_CLASSES` rows out of `test_depends_on_declarations.py`
   into `tests/cognition/test_depends_on_declarations.py`.
5. Empty `COGNITION_MODULES` in `test_trace_core.py`; make
   `_core_python_files()` yield everything under `corvidae/`.
6. Discord (Part 2 §2a): move `discord-py` from the root dependencies to
   the `discord` extra, make the SDK import lazy, add the startup error
   and the importorskip. `uv lock` again.
7. Two CI jobs (Part 2 §5). Sync lines in `harness/run.sh`, `CLAUDE.md`,
   `README.md` and `.claude/hooks/session-start.sh`.

**Docs.** Part 2 §6, all bullets except the `send_*` CLAUDE.md deletion.

**Acceptance.** Both CI jobs green. `uv sync --extra dev --extra acp`
followed by `uv run python -c "import sqlite_vec"` fails, and so does
`import discord` (both really are gone from the kernel), while
`uv run python -c "import corvidae.channels.discord"` succeeds. `uv sync --extra dev --extra cognition` followed by a `harness/run.sh`
smoke run (it generates its own `agent.yaml` under `$STATE_DIR`) behaves as
before. `sessions.db` schema diff against a pre-split database is
empty.

## WP3 — Outbound `send` consolidation (M–L; runs second)

**Red.**

- `tests/test_outbound.py`: a `TransportPlugin` subclass with
  `transport = "fake"` that overrides only `send_message` and records calls;
  register it with a second subclass `transport = "other"`; call
  `pm.ahook.send` with each of the four event types on a channel owned by
  `"fake"` and on one owned by `"other"`; assert `fake` saw exactly one
  message and nothing else, `other` saw nothing, and no exception was
  raised for the no-op defaults. A third subclass overrides `owns` to
  return `False` and sees nothing even for its own transport's channel
  (the `WebPlugin` disabled case). A tool-status event built with a
  `tool_call_id` reaches `send_tool_status` with the id intact. Fails
  today: no `TransportPlugin`, no `send`.
- The four strict xfails in `test_hook_grammar.py` flip to passing (the
  implementer removes the `xfail` markers; the test then asserts absence).
- Existing transport tests under `tests/` that assert on
  `send_message`/`send_progress` behaviour are the regression suite; they
  go red at the rename and green when the helper and assertions are
  updated.

**Green.**

1. `corvidae/outbound.py` with the four dataclasses (including
   `OutboundToolStatus.tool_call_id`) and the `Outbound` union.
2. `hooks.py`: delete the four `send_*` specs; add `send`; add
   `TransportPlugin` after `CorvidaePlugin` (Part 1 §3).
3. All six transports (`AcpPlugin`, `CLIPlugin`, `DiscordPlugin`,
   `IRCPlugin`, `SignalPlugin`, `WebPlugin`): subclass `TransportPlugin`,
   set `transport`, delete `depends_on` and every `matches_transport` call
   in a send path, turn the `send_*` hookimpls into plain overrides taking
   an event. `WebPlugin` overrides `owns` with today's `_mine` body and
   keeps using it for its inbound hooks. Keep the `should_process_message`
   and `on_message_*` filters in Discord, Signal and web; they are inbound
   and not this package's business. `IRCPlugin.send_progress` currently
   delegates to `self.send_message`; after the change it must build an
   `OutboundMessage` from the progress event's text to do so, or format
   the line itself. `AcpPlugin.send_message` completes the pending ACP
   prompt in a `finally`; keep that. `AcpPlugin.send_tool_status` keeps
   its `(channel.id, tool_name)` keying; switching to
   `event.tool_call_id` is a follow-up (Open questions).
4. Call sites: `agent.py` (message ×2, thinking, tool status, progress),
   `task.py` (tool status), and whatever `grep -rn 'ahook.send_' corvidae`
   finds beyond those. The two tool-status sites pass `tool_call_id`
   (`call["id"]` in `agent.py`, `task.tool_call_id` in `task.py`).
5. `tests/helpers.py`: `pm.ahook.send = AsyncMock()`; re-grep
   `send_message` under `tests/` (23 files at `618c831`; WP2 has not run,
   so nothing has moved) and update the assertions, and the 67 direct
   transport-method calls (Part 1 §3). Rewrite the fake transports in
   `tests/test_integration.py` and `tests/test_hook_safety.py` as
   `TransportPlugin` subclasses; a bare `send_message` hookimpl with no spec
   is silently never called, which is exactly the failure this package
   exists to remove.
6. `test_hook_grammar.py`: delete `PENDING_WP3` and every xfail marker.
7. `corvidae/scaffold.py`: if the template mentions `send_message`, update
   it.

**Docs.** `plugin-guide.md` Messaging table: one `send` row and a
`TransportPlugin` subsection under Channels with the dispatch table.
`design.md` Transports section. CLAUDE.md: delete the `send_*`
broadcast-filter bullet. `threat-model.md`: no change.

**Acceptance.** CI green (WP2's two jobs do not exist yet); `grep -rn
matches_transport corvidae/channels` shows only inbound uses and
`WebPlugin.owns`; `test_outbound.py` green;
`test_hook_grammar.py` fully green with no xfails.

## WP4 — README and the closing docs pass (S)

- README "Included plugins" split; the optionality paragraph rewritten (R7);
  the one-liner unchanged.
- `docs/design.md` Architecture paragraph: one sentence that the cognition
  plugins live in `corvidae-cognition` and are documented in
  `docs/cognition.md`.
- `plans/kernel-split.md` (this file): add a Status line with the commit
  hashes of WP1–WP3, the way `plans/forks.md` does.

**Acceptance.** A reader of the README can answer "what do I get without
the extra" from the plugin tables alone. Cold review of the docs diff.

# Open questions

- **R8**, the event bus. Deferred with a criterion.
- **R9**, `thinking.py` as kernel. Decided yes; veto in the requirements
  gate.
- **Out-of-tree compaction replacements.** Keeping `firstresult=True` on
  `resolve_compaction` means the rename is the only change for them. If you
  would rather the hook were broadcast, that is a plugin-API change and
  needs its own requirement; this document does not make it.
- **ACP tool-call ids.** After WP3, `AcpPlugin` can key ACP `tool_call`
  updates on `event.tool_call_id` instead of minting a uuid per
  `(channel.id, tool_name)`. Not yet verified: whether two concurrent
  calls to the same tool in one turn collide today. If they do, the
  switch is a bug fix with its own test; it is a separate pull request
  either way, so WP3 stays behaviour-neutral (R4).
- **Version numbers.** Both distributions are `0.0.1`. Nothing here needs
  them to differ. If the kernel is ever published, the cognition member pins
  `corvidae` to a compatible range at that point, not now.
- **`docs/cognition.md` vs a `docs/` directory in the member.** This
  document picks a single docs tree so that existing links keep working.
  The implementer may move it later; it is a file move.

# Definition of done

- R1 through R7 and R10 acceptance criteria met, in CI, on main.
- `tests/test_hook_grammar.py`, `tests/test_kernel_boundary.py` and
  `tests/test_outbound.py` exist and are green with no xfails.
- The CLAUDE.md `send_*` bullet is gone.
- The README's first sentence is true of both the architecture and the
  default install.
