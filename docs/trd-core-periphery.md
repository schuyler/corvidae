# Corvidae: technical reference — core/periphery architecture

**Status: accepted (owner, 2026-11-17).**
`docs/prd-core-periphery-supplement.md` ("the supplement") says what
must be true under the core/periphery strategy. This document says how
each of those outcomes is met and in what order it is built.
Conformance to the supplement is checked from this document's
acceptance. That acceptance does not settle what TRD §15.1 leaves open
for the owner, nor design what TRD §15.2 leaves out. `docs/prd.md` is
"base".

Section numbers written "§3.5" are the supplement's; "base §3.5" is the
base PRD's; "TRD §5" is this document's.

## 1. Inputs and conventions

### 1.1 Inputs

- The supplement and `docs/core-periphery-open-questions.md`, on `main`
  at `23490ae`.
- The owner's decisions of 2026-11-16 on #979, #980, #981 and #982.
- The owner's decisions in the TRD session of 2026-11-16, recorded in
  `docs/core-periphery-open-questions.md` §5. This document cites them as
  "TRD decision A1" to "A11" and "C1" to "C6", and those taken after
  the draft merged as "D1" to "D4".
- The tree. Every statement about current code names a file, and was
  read at `23490ae`. Nothing here was run: no statement about
  llama-server is verified on the pinned build (#1009).

### 1.2 What this document fixes and what it leaves

It fixes contracts: which component meets a requirement, hook
signatures, tool names and arguments, table columns, config keys, state
transitions and the build order. It leaves prompt wording to the prompt
files (open question 3) and function bodies to the implementation.

Numbers given as defaults are commented best guesses (base principle 7).
Each is an operator config key, and none has been measured.

### 1.3 Identifiers

Identifiers a plugin owns follow `<plugin>:<component>:<uid>` (owner,
2026-11-16). For a channel that is the base `transport:scope` id: the
strategy's transport plugin is named `core`, and `<component>:<uid>` is
its scope. The kernel splits a channel id at the first colon only
(`corvidae/channel.py`, `resolve_persisted_channel`).

| Identifier | Names |
|---|---|
| `core:main:<uid>` | the core's channel |
| `core:fork:<uid>` | a long-lived fork's channel |
| `core:task:<uid>` | the log rows of a task-bound peripheral |

A uid is 12 lowercase hex characters from `secrets.token_hex(6)`. The
core's uid is minted once at first start and stored (TRD §11); a fork's
or task's is minted when it starts.

### 1.4 Terms

- **Crossing**: an item one context causes on another context's queue. An
  escalation, an instruction, a join report and a note are crossings. A
  tool result returning to the turn that called the tool is not.
- **Chain**: the crossings that follow from one starting item, sharing a
  `chain_id`.
- **Resident**: a context whose KV state is in a server slot.
- **Paged out**: a context whose KV state is in a saved slot file.

## 2. What has to exist first

The supplement names its prerequisites as #16, #362, #363, #364 and
#367, all closed. The decisions of 2026-11-16 added #814 and #138.
Reading the tree for this document found five more that no list names.
The decision of 2026-11-17 on #1031 adds one more.

| Needed | Issue | State | What stands on it |
|---|---|---|---|
| Write-ahead log of accepted work (base §3.4.8) | #318 | open | the stored hop count, the ceiling on waiting work, forks replaying after a restart (TRD §9, §11) |
| Speaking on any channel (base §3.5.9) | #234 | open | exact wording, core-initiated contact, a fork working a channel (TRD §5.4, §7.6) |
| Reading another channel (base §3.5.11) | #235 | open | tail import (TRD §5.5) |
| Latency bounds on leases (base §3.13.5) | #365 | open, on hold | the core queue's bound and every "overdue" report (TRD §5.5) |
| Pluggable lease order (base §3.13.7) | #366 | open | the slot manager's ordering (TRD §8) |
| Origin never taken from a tool argument | #814 | closed (2026-11-17) | every origin in TRD §5.2 |
| Trust plugin | #138 | open | trust stamping (TRD §9.4) |
| Critique judges every main-loop turn (base §3.7 Critique 3–4) | #1031 | decided, not built | every "Critiqued: yes" in TRD §5.2 |

Evidence, each read at `23490ae`:

- **No write-ahead log.** `corvidae/agent.py` and
  `corvidae/persistence.py` have no replay path for queued items.
- **No send or read across channels.** The cross-channel-send package
  registers `notify_other_channel` and `list_channels` only.
- **No latency bound.** `corvidae/lease.py` has no bound and no overdue
  state.
- **No order hook.** The grant order is fixed in
  `ProviderLeases._grant` (`corvidae/lease.py`).
- **Origin is a tool argument today.** #981 records that `add_job` and
  `update_job` pass a model-chosen `origin` into the notification.

Critique judges every main-loop turn whatever its origin (#1031), so
the origins of TRD §5.2 need no registration.

## 3. Components

### 3.1 What a strategy is (TRD decision C1)

The kernel has no concept of a strategy. The core/periphery strategy is
a distribution, `packages/corvidae-core-periphery`, behind a
`core-periphery` extra. Its plugins, `limits` aside (TRD §3.2), do nothing without a
`core_periphery:` block in `agent.yaml`, as base principle 5 says of any
plugin with no config block. The base topology is what runs when the
block is absent or the plugins are in `plugins.disabled`. Changing
strategy is a restart (base principle 1) and migrates no data, because
nothing is keyed to a peripheral (open question 17).

A peripheral is the kernel's existing per-channel agent loop on a
transport's channel, with settings the strategy supplies. No second loop
is written.

### 3.2 Plugins

| Entry-point name | Role | Depends on |
|---|---|---|
| `core` | A `TransportPlugin` that owns `core:*` channels. Registers the core's channel at startup, receives idle events, serves the observables, mints uids, and registers `hold_queue` and `resume_queue` (TRD §8.3). | `registry`, `idle`, `llm` |
| `periphery` | Supplies peripheral settings and prompt sections, the `escalate` tool, the pre-model checks, prompt revisions. | `core`, `registry` |
| `forks` | Long-lived forks: fork channels, the fork record, joins, blocking, routing, pass-through. | `core`, `registry` |
| `slots` | The slot-management plugin: lease order, paging, saved slot files. | `core`, `llm`; #366 |
| `limits` | Verdicts at the inbound gate and the admission seam (TRD §9). Has its own `limits:` config block, so it runs under either topology. | `agent`, `clock` |

Each is registered in `[project.entry-points.corvidae]` of the
package's `pyproject.toml`, and CI gains a job as
`docs/plugin-checklist.md` describes.

### 3.3 Existing plugins under the strategy

| Plugin | Under the strategy |
|---|---|
| `heartbeat` (cognition) | Must not be loaded. If it is, `core.on_init` aborts startup with `Config error: core_periphery is configured and the heartbeat plugin is loaded — add "heartbeat" to plugins.disabled in agent.yaml`. Its trigger and diff logic move to a shared module that both import (TRD §5.3). |
| `subagent` (kernel tool) | Becomes the task-bound peripheral, extended (TRD §6.6). |
| `fork` tool (kernel) | Unchanged but for its call to the admission seam (K1). It starts from a fresh window (`corvidae/tools/fork_tool.py`), so it is neither of §3.5's two forks. |
| cross-channel-send | Optional. `notify_other_channel` calls the admission seam and carries the chain (TRD §9.3). |
| `goals`, `memory`, `critique`, `appraisal`, `outcome_log` | Changed: TRD §6.4 (`outcome_log`), §10.1 (`goals`), §10.3 (the rest). |

## 4. Kernel changes

The kernel changes below are contracts other plugins rely on, so each
needs the owner's approval before it is built (AGENTS.md, "Process
instructions"). K1 was agreed in session on 2026-11-16, with a refusal
that carries its reason (TRD decision A7). K7 follows TRD decision
C3. K2 to K6, K8 and K9 are new in this document, and the owner
approved them on 2026-11-16 ("I approve K2–K6, K8 and K9"), and K2's
rule for forks later the same day (TRD decision D4). K10 was added
while splitting phase 2, and the owner approved it on 2026-11-18
(#1058, ruling 4).

### K1. Admission seam for internal work (§3.12 item 2)

```python
@dataclass(frozen=True)
class WorkRefusal:
    limit: str    # the config key that was reached, e.g. "limits.max_waiting"
    remedy: str   # what the caller can do, in plain language

@hookspec
async def should_start_work(
    self,
    channel: Channel,          # the queue the work would land on
    kind: str,                 # "task" | "fork" | "instruction" | "escalation" | "note"
    source_channel: str | None,
    trace_id: str | None,
    annotations: dict,
) -> WorkRefusal | None:
    """Veto. None is no opinion; the first WorkRefusal wins."""
```

- **Who calls it.** The tool that starts the work, before it enqueues
  anything: `subagent`, `fork`, the strategy's `escalate`, `instruct`
  and `fork_context`, and `notify_other_channel`. The Task that carries
  an ordinary tool call (base §3.1.2) is not offered.
- **What is never offered.** The result of work already started: a
  notification carrying a `tool_call_id`, a fork's join or abandonment
  report, and an item moved to a parent as late. §3.12 item 2 forbids
  refusing those.
- **A refusal** becomes the starting tool's result,
  `Not started: <limit> reached — <remedy>`, and is logged at warning
  with the kind, both channels and the limit.
- **An exception** in an implementation is logged and treated as no
  opinion, as `should_send_response` fails open
  (`corvidae/hooks.py`).
- **Why not `bool`.** The existing veto hooks return `bool | None`,
  which cannot carry the limit and remedy §3.12 item 2 requires. This
  adds one result type to the veto grammar.
- **Depth.** `Agent.waiting_work() -> dict[str, int]` returns the count
  of accepted items not yet finished, per channel id: items queued on
  the channel's queue, the item in flight, and tasks queued or running
  on the TaskQueue that the channel started (owner, 2026-11-17: a
  subagent or fork sits on the TaskQueue until it finishes, and its
  result can never be refused, so a count of channel queues alone never
  sees fan-out from one turn). It reads the queues until #318 lands and
  the write-ahead log after.

Base PRD text, added as §3.1 item 20 when this was built (#1050):

> Internal work that is about to start (a task, a fork, a note or other
> item one context puts on another's queue) is offered to plugins for a
> verdict before it is accepted, as inbound messages and outbound
> emissions are. A refusal returns to whatever tried to start the work
> as a result naming the limit and the remedy, and is logged at
> warning. The result of work already started is never offered. The
> kernel gives no verdict of its own, and lets plugins read how much
> accepted work is waiting.

### K2. A channel's main turn can name its profile (§3.2.7, §3.8.1)

`Agent` takes one client for every channel's main turn
(`llm.get_client()` in `corvidae/agent.py`), and `ChannelConfig`
(`corvidae/channel.py`) has no profile field.

`ChannelConfig` gains `profile: str | None`, added to
`CHANNEL_ENTRY_KEYS`. When it is set, the main turn uses that profile's
client; `get_client` takes a stage today (`corvidae/llm_plugin.py`), so
it gains a lookup by profile.

A fork of a channel's window takes its client from
`llm.get_client(stage)` today (`corvidae/agent.py`), which resolves a
stage through `llm.stages` or its kind's default profile and never
through the channel. For a channel that names a profile that changes
(TRD decision D4):

- a stage that `llm.stages` maps runs on the mapped profile, as now;
- a chat stage it does not map forks on the forked channel's profile,
  in place of the kind's default.

The same profile serves the stage's other work for that channel, such
as a consolidation written from the rendered view (TRD §10.5), so the
stage still has one profile on every path it takes for a channel. The
owner's approval named forks; the other paths follow from base §3.8
item 4 and are listed in TRD §15.1. Work with no channel is unchanged.

That amends base §3.6 item 5 and §3.8 item 4, and §4 of the supplement
lists it. Proposed base text for §3.6 item 5, in place of "A stage the
operator hasn't mapped uses the deployment's default profile for its
kind": "A stage the operator hasn't mapped uses the deployment's
default profile for its kind, or, for a chat stage's work on a channel
that names a profile, that channel's." And for §3.8 item 4, in place of "Each
background stage has one profile": "Each background stage has one
profile for a given channel".

A channel with no profile is unchanged, and so is the base topology.
Without the rule, on a deployment with two models every fork of a
peripheral's window (critique, consolidation, compaction) would run on
the default profile and prefill that window from cold.

### K3. Plugin defaults for channel settings (§3.2.8)

A plugin can give its own channels a config through
`ChannelRegistry.get_or_create(transport, scope, config)`, but cannot
set defaults for channels another transport owns:
`set_wide_entries` replaces the operator's entries whole.

`ChannelRegistry.set_plugin_defaults(source: str, config: ChannelConfig,
tunables: dict[str, object], applies: Callable[[str], bool])` adds a
layer beneath every operator entry.

- **Settings** in `config` apply where no operator entry sets the field.
- **Tools.** `Channel.disabled_tools` is a union of every layer
  (`corvidae/channel.py`), so a default that joined the union could
  never be undone. The plugin layer's `disabled_tools` applies only to a
  channel for which no operator entry, its own or a matching wide one,
  has a `tools` key. An operator who sets `tools.disabled` for a
  peripheral replaces the default list (§3.2.8).
- **Tunables.** `resolve_tunable` (`corvidae/tuning.py`) reads
  `channel.runtime_overrides`, then global config. It gains the plugin
  layer between the two, so a plugin can default a dotted tunable for a class of channels. A
  plugin default therefore outranks the global value in `agent.yaml`.
  For the defaults set this way (TRD §10.3, §10.5), the global key does
  not reach those channels; a channel's `runtime_overrides` still does.

### K4. A queue can be held (§3.5.7)

`SerialQueue` (`corvidae/queue.py`) has no hold. A lease stop cannot
serve: a blocking fork shares its parent's lease, and §3.5.7 says
routing does not stop that lease.

`SerialQueue.hold(reason: str)` and `release()` are added. A held queue
accepts items and starts none. A hold is in memory; the plugin that set
it re-applies it after a restart. `release()` needs no turn, so a hold
cannot leave a state no turn can leave (base §3.13.7).

### K5. A channel can record without responding (§3.5.6)

`Channel.record_only: bool`. While it is set, the Agent persists an
admitted inbound message at the tail of the channel's window, fires
`on_message_persisted`, and makes no model call. A turn already in
flight is not interrupted.

### K6. A channel can take its window's head from another (§3.5)

At conversation init the Agent resolves a channel's system prompt and
tool view for that channel's own id (`corvidae/agent.py`, where
`channel.conversation is None`), outside `resolve_conversation`. A fork
channel would get a head built for itself, with no goal block, and
share no prefix with its parent.

`Channel.head_from: str | None`. When it names a channel with a live
conversation, the Agent copies that conversation's `system_prompt` and
`tool_view` in place of resolving its own. If the named channel has no live
conversation, as after a restart, the Agent initialises it first. At
the channel's first compaction the head is resolved for the channel
itself, as for any channel.

### K7. `subagent` arguments (§3.2.8, §3.2.9; TRD decision C3)

`subagent(instructions, description, tools=None, append_prompt=None)`.

- `tools` is a list of tool names. The subagent gets the intersection
  with the set it would have had, so the argument only narrows. An
  unknown name is refused, naming it and the accepted names.
- `append_prompt` is appended after the persona-neutral prompt.
- The loop's messages are persisted to `message_log` under
  `core:task:<uid>`, and the uid is in the tool's result. The id is a log key only: it is
  written under every topology and needs no `core` transport.

The arguments exist under every topology. That widens §4's "fewer
tools" amendment from this strategy to the base; the owner accepted it
(TRD decision C3). Proposed base text for §3.5.7: "A caller may give a
subagent fewer tools than its own set, never more."

### K8. A channel can run under another's lease key (§3.5)

The Agent wraps each turn of a channel, and each one-shot fork of its
window, in `lease_scope(channel.id)` (`corvidae/agent.py`), and no hook
lets a plugin name another key. A fork channel would always lease as
itself, never under its parent's key.

`Channel.lease_key: str | None`. When it is set, the Agent uses it in
place of the channel's id at both sites, as does the kernel `fork` tool
at its own (`corvidae/tools/fork_tool.py`), and reads it at each model
call, so a change takes effect at the channel's next call. `forks` sets
it when a fork starts and clears it at the switch of TRD §7.2.

### K9. The CLI can be attached to another channel (§3.1.1)

`CLIPlugin` routes stdin to `cli:local` always
(`corvidae/channels/cli.py`, `_read_loop`), and `TransportPlugin.owns`
matches a transport's own channels only (`corvidae/hooks.py`), so the
CLI neither reads for a `core:*` channel nor writes to one.

`CLIPlugin.attach(channel: Channel)`: stdin goes to that channel in
place of `cli:local`, and the CLI also delivers that channel's outbound
events. The channel keeps its id, so its history stays the core's.

### K10. A failed turn is a metric (§3.1.11)

When `run_agent_turn` raises, `Agent` (`corvidae/agent.py`) logs and
calls `resolve_llm_error`, a firstresult hook that picks the error text
and stops at the first answer, so no plugin can count failed turns.
Observables' errors in the last hour (TRD §5.5) needs that count.

`Agent` emits one `on_metrics` event, `agent.turn.failed`, tagged with
the channel id and the exception type, on every failed turn, a failed
retry included. It emits before `resolve_llm_error` is called, so a
failure in the error path cannot suppress it, and a consumer that
raises is logged at warning without stopping the user's error message.
The exact tag keys are fixed by #1060 and documented in
`docs/design.md`.

### Changes other issues already carry

#318, #234, #235, #365, #366 and #814 have their own issues (TRD §2);
#234 and #235 are plugin work (base §3.5.11). This document adds nothing to their scope except one
ruling on #366: a stop never lapses, and a deferral is the pause that
does (owner, 2026-11-16; TRD decision C6). Proposed base text for
§3.13.7, replacing "whether it lapses by itself is not decided": "A stop
does not lapse; it ends when the plugin or the operator resumes it."

## 5. The core

### 5.1 The channel (§3.1.1, principle 17, open question 8)

`core.on_start` reads the uid (TRD §11) or mints one, then calls
`registry.get_or_create("core", f"main:{uid}", config)`. The config
comes from `core_periphery.core`. Nothing hard-codes the id:
`core.core_channel() -> Channel` is the lookup, and while notifications
are routed to a fork (TRD §7.3) it returns the fork's channel.

`core_periphery.core.attach_cli: false` is the default. Set true,
`core.on_start` attaches the CLI transport to the core's channel (K9),
and startup fails naming the key if that transport is not loaded. No
other transport can be attached.

### 5.2 Origins and what the core receives

Every item the strategy enqueues goes through `on_notify`, with
`source` and `annotations["origin"]` set by plugin code (open question
12). No strategy tool takes either as an argument.

| Origin | Enqueued by | On | Critiqued |
|---|---|---|---|
| `escalation` | `periphery` | the core | yes |
| `instruction` | `core` | a peripheral | yes |
| `idle` | `core` | the core | yes (§3.1.2) |
| `fork_join` | `forks` | the parent: the notification's text is the report of TRD §7.4, and it wakes a turn | yes |
| `anomaly` | `core` | the core | yes |
| `fork` | `forks` | a new fork (its task) | yes |
| `late` | `forks` | the parent | yes |

Task completions and scheduler firings keep the origins they have.

### 5.3 Idle (§3.1.2, principle 20, open question 20)

`core` registers `idle.watch_system(after, ...)` as the heartbeat does
(`heartbeat.py`, `on_start`). On firing it enqueues one `idle`
notification on the core's channel, made of:

1. new memories since the last beat from `core:` channels; the
   peripherals' come by the digest (TRD §10.5);
2. goals retired since the last beat;
3. the idle prompt, from the key `core_periphery.core.idle_prompt`,
   resolved by `resolve_prompt` (`corvidae/prompts.py`).

Open goals are not repeated; they are in the goal block (TRD §10.1). The
diff state is the heartbeat's, moved to
`corvidae_cognition/idle_beat.py` and imported by both plugins. The
repeat while the system stays idle is the idle tracker's
(`corvidae/idle.py`).

Base §3.7 Commitments routes dormant commitments and long-quiet
relationships to the heartbeat as goal candidates. Nothing implements
that feed today (#982), and this document does not add it.

### 5.4 Reaching a user (§3.1.6, §3.1.8, §3.1.9, §3.9, principle 21)

Two tools, on the core and on forks:

- `instruct(channel_id, text)` enqueues an `instruction` notification on
  the peripheral's queue. The periphery's turn composes and sends. If
  the channel is stopped, `instruct` resumes it first and removes its
  `cp_stops` row (§3.9.1).
- **Exact wording** is base §3.5.9's send tool (#234), unchanged. It
  records the message in the target channel's history through that
  channel's queue, so the peripheral's context holds it (principle 21),
  and it passes `should_send_response`, so §3.9.4's limits apply.

A reply a periphery composes from an instruction is that channel's
ordinary final emission, so it passes the same gate.

The strategy builds no send path of its own (TRD decision A9).

### 5.5 Observables, anomalies and tail import (§3.1.11, §3.1.12)

`observables()` is a tool that returns, per channel: queue depth,
messages in the last hour, errors in the last hour, residency (resident,
paged out, or not managed), whether it is stopped or pass-through, and
for a fork its parent, age and blocking state. Reading it starts
nothing.

`core` samples the same values every `observables_period` (60 s) and
logs them at info (§3.10.5). It logs a warning and enqueues an
`anomaly` notification for:

- a queue whose depth has grown over three samples with nothing served;
- a queue past its latency bound (needs #365);
- a peripheral with `error_threshold` (3) failed turns in a row;
- a fork with an empty queue and nothing in flight for `fork.idle_after`
  (600 s) (gap 3).

One notice per kind and channel per `anomaly_interval` (900 s).

Phase 2 builds the tool, the sampler, the pacing, and the fields and
anomaly it can compute: queue depth, messages and errors in the last
hour (errors counted from K10), residency reported as not managed, and
the queue-growth anomaly. The rest arrives with the feature it watches
(owner, 2026-11-18; #1058, ruling 3): the `error_threshold` anomaly
with peripherals in phase 3; the stopped and pass-through fields, the
fork fields and the fork-idle anomaly with forks in phase 5; residency
other than not managed with slots in phase 6; the latency-bound anomaly
with #365.

`import_tail(channel_id, n)` is base §3.5.11's read (#235) aimed at a
peripheral: the last `n` rows of its window, appended at the core's tail
framed as data from that channel. It writes nothing to the peripheral.

### 5.6 Work that ends (§3.10.1)

The task queue keeps the last `daemon.completed_task_buffer` results
(100, `corvidae/task.py`). `core` records every escalation,
instruction, fork and task-bound peripheral in `cp_work` (TRD §11) when
it starts and when it ends, with how it ended. `work_status(id=None)`
reads it.

The core declines an escalation with `decline_escalation(row_id,
reason)`, which sets the row's outcome to `declined` and logs one info
line naming the channel and the reason (§3.5.5).

`cp_work`, `work_status` and `decline_escalation` are built in phase 3,
with escalation, the first kind they record (TRD §13).

## 6. Peripherals

### 6.1 Settings (§3.2.1, §3.2.7, §3.2.8, open questions 5, 7, 9)

One peripheral per channel. `periphery.on_init` calls
`set_plugin_defaults` (K3) for every channel no `core` transport owns,
from `core_periphery.periphery`:

| Key | Default | Meaning |
|---|---|---|
| `profile` | unset | the peripheral's main-turn profile (K2); unset means the core's |
| `tools.disabled` | `shell`, `write_file` | §3.2.8; the operator extends or replaces the list (K3) |
| `max_context_tokens` | unset | the existing per-channel key (open question 5) |

A channel-bound peripheral gets the identity in full (open question 9)
and the agent's whole memory (§3.2.5). Both are what any channel gets
today, so nothing is built for them.

### 6.2 Prompt sections (§3.2.4, §3.2.6, open question 3)

`periphery` implements `transform_system_prompt` for peripheral
channels and appends, after the register:

1. `core_periphery.periphery.prompt`, an ordinary prompt key;
2. the escalation criteria: the channel's file under
   `state/prompts/criteria/` if one exists, else the key
   `core_periphery.periphery.criteria`.

Base §3.3.8 re-resolves plugin sections at a compaction boundary or
conversation start, so a change to either takes effect there and not
mid-window.

### 6.3 Escalation (§3.4, principle 19, open questions 2, 19)

`escalate(reason, summary="")` is a tool on peripheral channels.

- `reason` is one of `needs_reasoning`, `multi_step`, `goal`,
  `decision`, `unfamiliar`, `sensitive`, `needs_tool`. An unknown value
  is refused, naming the accepted ones. `structural` is set only by the
  plugin (below) and is not accepted from the model.
- `summary` is cut to `summary_max_chars` (600).

The notification the core receives is built by `periphery`:

| Field | Set by | Content |
|---|---|---|
| channel, sender | plugin, from the triggering item's `metadata` | where and who |
| row id | plugin | the triggering message's `message_log` id |
| chain id, hops left | plugin | TRD §9.3 |
| trust level | plugin, when #138 exists | TRD §9.4 |
| reason | the tool argument, validated | the fixed list above |
| message | plugin | the triggering message verbatim, when it is at most `verbatim_max_chars` (500) |
| summary | the model | optional when the message is included |

It is framed as data from the source channel by the kernel's prompt
build, like any notification from another channel (§3.4.6).

**Structural rules.** These act on a person's message only.
`on_message_persisted` also fires for a notification that starts its
own trace (`corvidae/agent.py`), an instruction among them, so
`periphery` records the trace id and sender of each message in
`on_message_admitted`, which fires for admitted inbound messages, and
acts in `on_message_persisted` only for a trace it recorded there. It
does nothing on a channel that is pass-through (K5). For such a message
it escalates with reason `structural`, with no model deciding it, when
the message has an attachment, the sender has no earlier row on the
channel, or the message is longer than `structural.max_chars` (1500;
`null` turns the rule off). The other two rules have an on/off key. It
appends one CONTEXT row at the peripheral's tail saying the message was
escalated, and the periphery's turn runs as usual.

**The eval switch (§3.11.1).** `core_periphery.periphery.force` is
`always`, `never`, or unset. `always` escalates every person's message
by the same path; `never` removes the tool and skips the structural
rules.

**Not delivered.** There are two cases. In both the plugin sends the
status text itself, as an ordinary emission on the user's channel, so
telling the user does not depend on what a model writes (§3.12 item 2).

- *Refused.* The admission seam refused the escalation, at a limit or
  at zero hops (TRD §9.3). Nothing is queued. The user is told the
  request was not passed on and why. Where a tool made the call, its
  result says the same and adds that the user has been told.
- *Core unreachable.* The escalation was accepted, and the core's last
  `error_threshold` turns failed for an unreachable model. It stays
  queued, and the user is told once that it is waiting (open question
  11).

### 6.4 Escalation quality (§3.4.5, §3.10.2, gap 8)

`outcome_log` gains a table and one method:

```python
async def record_label(
    self, trace_id: str, channel_id: str, kind: str, source: str, reason: str | None
) -> None
```

`kind` is `false_alarm` or `miss`. `source` is one of `core`,
`user_correction`, `critique`, `tail_review`. Each source records when
it fires, and the list can grow.

- The core labels a false alarm with `label_escalation(row_id,
  "false_alarm")`.
- A miss is recorded by the corrections path on a user correction, by
  critique on a bad verdict for a periphery reply, and by the core with
  `label_escalation(row_id, "miss")` after `import_tail`.

Rates per peripheral and by `reason` are queries over the table.

### 6.5 Criteria rewrites (§3.1.5, §3.2.4, §3.4.4, gap 7; TRD decisions C4, C5)

`rewrite_criteria(channel_id, text)` is a tool on the core.

1. It is refused if `rewrites.frozen` is true, if the channel is in
   `rewrites.frozen_channels`, or if `text` is longer than
   `rewrites.max_chars` (4000). The refusal names the key.
2. Otherwise a row is added to `cp_prompt_revisions` with state
   `pending`, and the tool returns the revision id.
3. With the critique plugin loaded, `periphery` asks it for a verdict:
   `critique.review_text(channel, text, instruction_key) -> Verdict`, a
   new method that forks the core's window with the rewrite. `Verdict`
   has `ok: bool` and `reason: str`. A verdict that is not ok sets the
   revision `rejected`.
4. At the peripheral's next compaction boundary (`on_compaction`) or
   conversation start, the
   newest pending revision is applied if critique is not loaded or its
   verdict is ok: the text is written to the channel's criteria file,
   the row becomes `applied`, older pending rows become `superseded`,
   and one info line names the revision id and the size change. A
   revision still waiting for a verdict stays pending for a later
   boundary.

The core writes only a per-channel criteria file. The shared default,
the identity, the register and `periphery.prompt` are the operator's.
Critique forks the window that may hold an injection, so it is a
quality check and the lock is the boundary; `docs/threat-model.md`
records default-on as an accepted risk when this is built.

### 6.6 Task-bound peripherals (§3.2.9, principle 17, open question 17)

A task-bound peripheral is the `subagent` tool with K7's arguments. It
has no persona, writes no memory and no goals, and consolidation does
not run for `core:task:*` (TRD §10.3). When its task ends its result
returns to the launcher as a task notification, its saved slot files
are deleted, and its rows stay in the log.

### 6.7 Instruction fidelity (open question 18)

Critique reviews a periphery reply afterwards. When the reply answered
an `instruction` and the verdict is bad, `periphery` enqueues an
`anomaly` notification on the core, rate-limited as in TRD §5.5.

A critique fork runs on the critique stage's profile, which is one
profile for every channel today. Under K2 an unmapped critique stage
forks on the peripheral's own profile, so a small periphery model
reviews its own expansion. Mapping the stage to the core's profile
gets a better judge at the cost of a cold prefill on every review.
That is the operator's choice, and the default is the weaker check.

## 7. Long-lived forks

### 7.1 Taking one (§3.5, §3.1.7, open question 10)

`fork_context(task, blocking=False, route=False)` is a tool on the core
and on forks. `route=True` with `blocking=False` is refused.

1. `forks` mints a uid and writes a `cp_forks` row: parent channel, the
   parent's last `message_log` id (the fork point), `blocking`, `route`,
   the time, state `running`.
2. It registers `core:fork:<uid>` with the parent's config and
   `head_from` set to the parent (K6), and implements
   `resolve_conversation` for it: the parent's window rows up to the
   fork point, then the fork's own rows. The head is the parent's, and
   prompt build is a pure function of the rows after it (base principle
   2), so the fork's prompt begins with what the parent's slot holds.
3. It enqueues `task` on the fork's queue with origin `fork`.

A fork has the core's identity, tools and memory access (§3.5.2).

### 7.2 Lease key (§3.5; TRD decision A2)

A fork's calls run under `lease_scope(key)`, where `key` is set through
K8 and is:

- the key its parent is using, until the fork's first compaction or
  until a `lease.yield` or stop removes the shared lease while the fork
  has a call to send, or until the fork itself is held (TRD §8.3),
  which is the scheduler taking the shared lease back (§3.5);
- its own channel id from then on.

`cp_forks.own_lease` records the switch, so a restart does not undo it.

When a fork and its parent share a lease they share a slot, and the
slot holds the last caller's tokens. The other context reuses the
common prefix and prefills its own suffix (TRD decision A1).

### 7.3 Blocking, routing and the bound (§3.5.7, gap 3)

| | Parent's queue | Core notifications go to |
|---|---|---|
| non-blocking | runs | the parent |
| blocking | held (K4) | the parent's queue, where they wait |
| blocking and routed | held (K4) | the fork (`core_channel()`, TRD §5.1) |

`forks` sets a timer for `fork.blocking_wait` (900 s) from
`cp_forks.wait_started`, which is stored, so the bound counts across
restarts. When it fires and the fork has not joined:

- `blocking` and `route` are set false in the record;
- the parent's queue is released;
- one warning names the fork, its parent and the bound;
- the fork keeps running.

That is what keeps the core able to receive work through a fork that
never joins (§3.3.7).

A plugin's own forks (compaction, critique, appraisal, consolidation)
are one-shot forks under base §3.8. Each keeps its present behaviour as
its default, and `<stage>.fork_blocking` overrides it (§3.5.7).
Compaction's key is read in the kernel (`corvidae/compaction.py`); no
other plugin relies on it.

### 7.4 Join and abandonment (§3.5.3, §3.5.4, gap 11)

A fork ends when it calls `join(summary)`, or when the core calls
`abandon_fork(fork_id)`, or the operator runs `corvidae fork end <id>`.
An idle fork is reported (TRD §5.5) and never joined silently.

On either ending `forks`:

1. builds the **effects list** from the fork's log rows: every message
   sent (channel, row id), every tool call (name, row id), and every
   notification it handled while routed;
2. enqueues one `fork_join` notification on the parent (TRD §5.2)
   holding the effects list and, for a join, the model's summary. For an abandonment the
   summary is absent and the row says so;
3. moves items still on the fork's queue to the parent, each with
   origin `late`;
4. releases the parent's queue, deletes the fork's saved slot files,
   and sets the record `joined` or `abandoned`.

Each change of a fork's state (start, own lease, bound reached, join,
abandonment) is logged at info with the fork's id and parent; an
abandonment, and any fork ended without a join, is logged at warning
with the reason (§3.10.4).

The fork's rows stay in `message_log` and never enter the parent's
window. An item addressed to an ended fork goes to its parent as
`late`.

Summary completeness is measured, as the share of effects the summary
mentions, in the outcome log. It is not required (gap 11).

### 7.5 Checkpoints (§3.5.4; TRD decision A4)

A checkpoint is the pair (parent channel, fork point). `refork(fork_id,
task)` abandons the fork and takes a new one from the same pair, whose
window is the parent's rows up to that id. This holds after the wait
bound has let the parent move on, because the parent's rows before the
fork point do not change (base principles 2 and 4).

Where the slot manager runs, taking a fork also saves the parent's slot
(a save that keeps its slot), and a refork restores from that file when
its stamp matches. Without the file the new fork prefills from the log.

### 7.6 Working a channel (§3.5.6, gap 5)

`work_channel(channel_id, task)` takes a fork and sets the channel
pass-through: `Channel.record_only` (K5) is set, the flag is stored in
`cp_stops`, and `forks` copies each admitted inbound message the channel persists
(found as the structural rules find one, TRD §6.3) to the fork's queue,
framed as data from that channel. The fork replies with
the send tool (TRD §5.4). When the fork ends, the flag is cleared.

### 7.7 What runs on a fork channel (TRD decision A3)

| Machinery | Runs | Reason |
|---|---|---|
| log rows, JSONL, metering | yes | base principles 4, 11 |
| compaction | yes | also the fork's lease boundary |
| critique | yes | §3.7.1 |
| memory retrieval | yes | §3.5.2 |
| consolidation | no | the join report is consolidated in the parent's window |
| idle-started turns | no | an idle fork is an anomaly |
| `list_channels`, inbound gate | no | no user reaches it |

Detail of a long fork beyond its report is in the raw log only.

## 8. The slot manager

**Deferred (TRD decision D1).** The owner judges llama-server's RAM
prompt cache adequate for the workload for now, so this section and
phase 6 are not on the path. Phase 7 needs phase 6 and waits with it;
the owner named only phase 6. `hold_queue` and `resume_queue`
are not deferred.

Everything in this section waits for #366, except `hold_queue` and
`resume_queue` (TRD §8.3), which `core` builds in phase 5 on the
kernel's stop and resume (TRD decision C6). No lever in it has been run
on the pinned build (#1009).

### 8.1 Scope of the first build (TRD decision A8)

`slots` pages only on a provider declared with `heads: 1`. With one
slot there is none to choose, so the design does not depend on whether
`/v1/chat/completions` honours `id_slot` (#1009 check 4). On any other
provider `slots` gives no opinion on order and its tools report no
effect. More heads per provider, and router mode, are TRD §13 phase 7.

A provider is managed when `core_periphery.slots.providers.<name>` has
a `slot_save_path`. Any other provider is the degenerate case of
§3.3.4: only base §3.13 applies.

### 8.2 Order and paging (§3.3.6, §3.3.7, open questions 1, 6)

`slots` implements #366's order hook. Its rule, cheap enough for every
call boundary because it reads only the lease view and its own table:

1. a waiter past its latency bound goes first (base §3.13.7);
2. then a key the core brought forward or pinned;
3. then FIFO.

The holder that gives way is the least recently directed, and never the
core's own key unless the core asked (`page_out` on its own channel) or
a waiter is past its latency bound (base §3.13.5).

- **Page-out**: when a holder's lease is removed at a call boundary,
  `POST /slots/0?action=save` to a file named for the lease key.
- **Page-in**: when that key is next granted the head,
  `?action=restore` from its file if the stamp matches (TRD §8.4),
  before its call is sent.
- **Checkpoint**: a save that leaves the slot as it is.

The server's RAM prompt cache is off (`--cache-ram 0`), so every
context is in the slot or in a file `slots` wrote.

Work arriving on the core's own queue while the core is paged out or
stopped by its own direction resumes it: `slots` watches `on_notify`
and `on_message_admitted` for the core's channel, the second because an
attached CLI's input arrives as a message (TRD §5.1), calls
`LeaseManager.resume` (`corvidae/lease.py`) and removes the `cp_stops`
row, with no core turn deciding it (§3.3.6). While
`slots` is not loaded nothing would bring the core back, so `hold_queue`
refuses the core's own channel (TRD §8.3).

An upkeep stream yields at each chunk boundary with
`LeaseManager.yield_key`. Nothing calls it today; memory consolidation
gains the call (TRD §10.3).

### 8.3 Tools (§3.1.3, §3.3.6, §3.3.8)

On the core and on forks:

| Tool | Effect |
|---|---|
| `hold_queue(channel_id)` | `LeaseManager.stop` for the channel's key on every provider that declares `heads`; stored in `cp_stops` |
| `resume_queue(channel_id)` | `LeaseManager.resume`; removes the `cp_stops` row |
| `bring_forward(channel_id)` | marks the key directed now, and next in order |
| `pin_context(channel_id, on)` | the key is not chosen to give way, except at a latency bound (base §3.13.5) |
| `checkpoint(channel_id)` | a save that keeps the slot |
| `page_out(channel_id)` | save and release at the next call boundary |

`hold_queue` and `resume_queue` are `core`'s and need no `slots`. They
act on every provider that declares `heads`, because stop and resume
are the kernel's. A provider with no heads has no leases and so no stop
(`corvidae/lease.py`, `_bounded`): the result of either tool names each
such provider with `No effect on provider "<name>": it has no heads, so
no queue can be held there`, and the turn continues (§3.3.8). When no
provider declares heads, no `cp_stops` row is stored. The other four,
on a provider `slots` does not manage, return `No effect: provider
"<name>" has no slot control` in the same way.

The key a hold stops is the channel's own id. A fork still on its
parent's lease is moved to its own first (TRD §7.2), so holding a fork
never stops its parent. The core's own channel, here and in TRD §8.2,
is `core:main:<uid>`, whichever channel `core_channel()` returns.

While `slots` is not loaded, `hold_queue` on the core's own channel is
refused with `Not held: nothing would resume the core's own queue —
load the slots plugin, or hold another channel`, because the guard of
TRD §8.2 is the slot manager's (§3.3.6, §3.3.7).

### 8.4 Saved files (open question 15, gap 4)

Each save writes a row to `slot_saves`: lease key, provider, file,
token count and bytes from the save response, and a stamp of model id,
chat-template hash, server build and the channel's compaction-boundary
row id.

- **Restore** only when all four stamp fields match the running values.
  Otherwise the file is deleted, the context prefills from the log, and
  one warning names the context, the file and the field that differed.
- **`core_periphery.slots.skip_restore: true`** skips every restore for
  that start. A boot the supervisor marks safe
  (`packages/corvidae-restart`) skips them too.
- **Shutdown.** In `on_stop`, `slots` saves each resident context.
  `daemon.shutdown_timeout` defaults to 3 s
  (`docs/configuration.md`); a save that would not finish in the time
  left is skipped with a warning naming the context. Restore is lazy,
  at the key's next grant.
- **Space.** Files are deleted when their context ends, when a restore
  rejects them, and oldest first when the directory passes
  `slots.max_bytes` (8 GiB).

### 8.5 Logging (§3.10.3)

Each page-in, page-out and checkpoint logs the context id, the
transition, the trigger, and the core direction it followed if any. On
a provider `slots` does not manage, the lease manager's existing
`lease.grant` and yield metrics are the context-switch record; a
provider with no heads emits none and has no switch to record.

## 9. Limits and trust

### 9.1 Inbound (§3.12 item 1, gap 9)

`limits` implements `should_process_message`: a message is rejected
when its channel already has `limits.inbound_per_channel` (50) accepted
and unfinished messages. Caps are per channel only. There is no bypass
by sender. A rejected sender on a one-to-one channel is told once per
`limits.notice_interval` that the channel is full.

Phase 1 defers that notice (owner, 2026-11-17): the kernel has no
notion of a one-to-one channel, and none of the transports declares one. Phase 1
rejects at the cap and logs the rejection at warning at most once per
channel per `limits.notice_interval`. The notice waits on a
transport-declared one-to-one flag (#1051).

No per-response tool-call cap is built; `max_turns` and
`tools.max_agent_depth` bound a response today.

### 9.2 Waiting work (§3.12 item 2, open question 21)

`limits` implements `should_start_work` (K1) and refuses new work when
the sum of `Agent.waiting_work()`, less the messages from users in it
(`Agent.waiting_messages()`), is at `limits.max_waiting` (200).

Messages from users are left out (owner, 2026-11-17, #1052): §9.1
bounds them per channel, and counted here a flood on a few channels
would fill the ceiling and refuse every escalation, which is the
lockout gap 9 removed from the inbound caps. The ceiling bounds what
the contexts produce for each other; work derived from a message (an
escalation, a task, a result) still counts.

The sum also leaves out channels that are stopped. With messages from
users already out, what a stopped channel holds is internal items
(notifications, tool results, notes and escalations waiting for it),
and they wait for a resume, not for the system to catch up, so counting
them would refuse other contexts' work for as long as the channel stays
stopped. No decision states this rule; it is the agent's judgment. It
is built in phase 5, with `hold_queue`: until then nothing stops a
channel (`LeaseManager.stop` has no caller), so phase 1 sums every
channel.

### 9.3 Chains and hops (gap 1; TRD decision C2)

Three crossings start work on another context: an escalation, an
instruction and a fork's start. Each carries `annotations["chain_id"]`
and `annotations["hops_left"]`, set by the plugin that enqueues it,
never by a tool argument.

- Every item a turn causes carries the turn's `chain_id`,
  `hops_left` and `chain_channel`: the Tasks it starts and their
  completions (base §3.1.4 carries annotations through both), and
  every item a strategy plugin enqueues because of it, join and
  abandonment reports, late items and anomaly notices included. The
  plugin copies those three keys and nothing else, so the new item
  mints its own trace.
- An item that arrives with no chain (a user message, an idle event, a
  scheduler firing) starts one at its first crossing: the plugin mints a
  `chain_id`, sets `hops_left` to `limits.max_hops` (4), and records the
  user's channel, if there is one, as `chain_channel`.
- Only those three, and a note sent with `notify_other_channel`
  (below), subtract one from `hops_left`. A chain's first crossing
  subtracts too, so it leaves `max_hops` less one. A report, a late item
  or a notice carries the count unchanged, so the next crossing its
  turn makes subtracts from it.
- At zero, `limits` refuses the crossing at the admission seam. The
  plugin whose crossing was refused sends the status text on
  `chain_channel` (TRD §6.3, "Not delivered").

Each of the three starts its own exchange: a fresh trace, a fresh
`max_turns` budget and its own origin (TRD §5.2). That differs from
base §3.1.3 and §3.5.10, where a note-woken turn is charged to the
chain's budget and carries the chain's origin; §4 of the supplement
lists the amendment (owner, 2026-11-16). A note keeps the base rule.

Join and abandonment reports, late items and anomaly notices are never
offered to the seam and never refused (K1).

The count survives a restart once annotations are stored with accepted
items (#318). Before that it holds within one run.

`notify_other_channel` keeps its own `hop_count` and
`cross_channel_send.max_hops`. It gains two things: a call to the
admission seam with kind `note`, and, when the sending turn carries a
chain, the chain's annotations on the note with `hops_left` reduced. At zero
the note is refused like the other three. A loop cannot then alternate
between notes and instructions to stay under both counts (TRD decision
C2).

### 9.4 Trust (§3.4.6, §3.4.7, gap 2)

Nothing here is built until the trust plugin exists (#138): rows carry
no trust level today.

When it does, an escalation is stamped with the trust level of the
message that triggered it (§3.4.6), and the plugin that enqueues an
instruction, a join report, a task result or a core memory stamps it with the lowest trust level of
any row in the source window since that window's last compaction. No
tool takes a citation. Most instructions will be low-trust; that is the
cost of §3.4.7.

## 10. Cognition by context

### 10.1 Goal block (§3.6, principle 20, gap 6)

`goals` implements `transform_system_prompt` for the core's channel
and renders the block there. Base §3.3.8 re-resolves plugin sections
only at a compaction boundary or conversation start, which gives the
pinning and the byte stability of §3.6.2–3.

- One line per open goal, in `position` order: the text collapsed to
  one line, `status`, and the age of that status.
- `Goal` (`goals.py`) has `status` and `updated_at`. A
  `status_changed_at` column is added, so editing a goal's text does
  not reset the age.
- When the block would pass `goals.block_share` (5%) of the core's
  `max_context_tokens`, it lists the lines that fit and ends
  `and K more`. One warning names K and the key.
- A goal created, changed or retired between boundaries is appended as
  a CONTEXT row at the core's tail, as `goals` does through
  `before_agent_turn` today.

### 10.2 What runs where (§3.7, §3.1.10)

| | core | peripheral | fork | task-bound |
|---|---|---|---|---|
| compaction | yes | yes | yes | no; open, TRD §15.1 |
| critique | yes | yes | yes | no; its result is critiqued in the launcher's window (#1031) |
| consolidation | yes | yes | no | no |
| consolidation at an age (TRD §10.5) | no | yes | no | no |
| digest of peripheral memories (TRD §10.5) | yes | no | no | no |
| appraisal | yes | no | no | no |
| goal block | yes | no | the parent's, through K6, until the fork's first compaction | no |

### 10.3 Switches the cognition plugins need

- **memory**: a per-channel tunable `memory.consolidate` (default
  true), read with `resolve_tunable` (`corvidae/tuning.py`). `forks`
  defaults it false for `core:fork:*` through K3's tunables layer, and
  memory skips channel ids under `core:task:`. The three triggers are
  unchanged (open question 16) and a fourth is added (TRD §10.5), and
  idle consolidation stays upkeep
  under its own lease key. Consolidation gains a
  `LeaseManager.yield_key` call at each chunk boundary (§3.3.7), in
  phase 6.
- **appraisal**: a per-channel tunable `appraisal.enabled` (default
  true). `periphery` defaults it false for peripheral channels and
  `forks` for `core:fork:*`, both through K3.
- **critique**: `review_text` (TRD §6.5). Eligibility needs no switch
  (#1031).

### 10.4 Two memories of one exchange (open question 14)

A peripheral exchange and the core's handling of it are consolidated
separately, and both records carry the chain id. The measure is
co-retrieval: how often both come back in one retrieval. It is logged
by the funnel and decides nothing yet.

### 10.5 Bounded awareness (§3.1.13, §3.7.4; TRD decisions D2, D3)

The core learns what the peripherals handled themselves through their
memories, admitted at its tail. Two delays lie between a peripheral
exchange and the core holding it. This section says what each is made
of; it claims no single number.

**From exchange to memory.** `MemoryPlugin` gains a fourth trigger and
a per-channel tunable for it, `memory.consolidate_max_age` (seconds;
unset, the default, turns the trigger off), read with `resolve_tunable`.

- **The rule.** Whenever a channel with the tunable set has
  un-consolidated messages and no consolidation running, a timer stands
  for the time its oldest un-consolidated message was persisted plus
  the age (`Clock.call_later`, `corvidae/clock.py`). A time already
  past fires at once.
- **When it is re-evaluated.** After every message persisted on the
  channel, after every consolidation of the channel ends, whether it
  committed, was skipped or failed, and at startup for every channel
  with the tunable set. A timer is in memory only, so startup is what
  restores it.
- **When it fires,** memory runs the consolidation the idle trigger
  runs (`docs/cognition.md`, "Consolidation (write path)"), over the
  channel's un-consolidated range up to the newest row, with the age as
  its condition in place of quiet. So, as for idle, a range wholly in
  the live window is forked, and a range that is not, as after a
  restart or behind a lagging watermark, is written from its rendered
  view (base §3.7 Memory item 1). The turn trigger's token threshold
  does not apply.
- **A turn in flight** on the channel when it fires is not interrupted:
  the consolidation runs when that turn ends.
- **After an attempt that leaves the watermark where it was,** a
  failure or a skip, the timer stands for one age after that attempt
  ended, and a new message does not bring it forward. Memory's existing
  warning names the channel and the range.
- **Forks.** A live-window range needs forks on for the
  `consolidation` stage; with them off memory skips it every time
  (`memory.py`, `_live_record`). §3.7.4 is conditional on them. With
  them off, `periphery` logs one warning at startup naming both keys.

`periphery` defaults the tunable for peripheral channels through K3,
from `core_periphery.periphery.consolidate_max_age` (300). It is unset
for the core, forks and task-bound peripherals. The three existing
triggers are unchanged.

Proposed base text for §3.7 Memory item 1, to be added when this is
built: "A channel may also set an age, `memory.consolidate_max_age`:
once its oldest un-consolidated message reaches that age, its
un-consolidated messages are consolidated as on idle."

**From memory to the core.** `core` implements `before_agent_turn`
(`corvidae/hooks.py`) for `core:main:<uid>` and no other channel, and
acts only with the memory and funnel plugins loaded. That hook fires
before each model call, tool-result turns included, except the re-run
of a turn after an unreachable backend (`corvidae/agent.py`,
`if not item.retry`).

- **Reading.** `MemoryPlugin.since(after_id: int, limit: int,
  exclude_prefix: str) -> list[dict]` is new: the indexed, unredacted
  memories with an id above `after_id`, oldest first, leaving out those
  whose channel id starts with `exclude_prefix`. `core` passes `core:`,
  so the core's own memories and its forks' are not echoed back, and a
  `limit` of 20, a constant of the plugin; more than that wait for the
  next call. Each dict has what `MemoryPlugin.recent` returns today
  (`id`, `summary`, `channel_id`, `age_seconds`) and one more,
  `oldest_age_seconds`: the age of the first message the record's own
  consolidation covered. Memory stores that time with the record when it
  is written, in a new `memory.range_started_at` column; a record no
  consolidation wrote (`write_record`) carries its `created_at` there.
  The stored range will not do, because a near-duplicate merge widens
  the stored range back to the older record's start (`memory.py`,
  `_insert_record`) and would make a new exchange look as old as the one
  it resembles. `age_seconds` is counted from the last (`memory.py`,
  `source_timestamp`), and §3.7.4's lag is counted from the oldest.
- **Admitting.** The memories go in through the funnel, as base §3.7
  requires of every subsystem's output: `core` calls
  `funnel.admit_detailed` (`funnel.py`) with source `digest` and one
  line per memory, oldest first: a marker naming the memory's id, its
  channel id, `oldest_age_seconds` and its summary. The funnel frames
  them as data, appends and persists the CONTEXT row, and applies the
  source's budget (`funnel.budgets.digest`, else
  `funnel.default_budget`). That budget is §3.1.13's bound on what one
  call is given.
- **The key.** The funnel finds a duplicate by looking for the key's
  text in the source's rows, so the key has to be in the line. Each
  line's key is its marker, in a form no other id's marker contains, as
  memory's own `memory <id>)` is (`memory.py`).
- **The watermark.** `digest_memory_id` in `cp_state` (TRD §11) is the
  highest id up to which every memory was admitted or dropped as a
  duplicate,
  both of which `admit_detailed` reports. `core` stores it after the
  call returns. An entry the budget left out stops it, so that entry
  and the ones after it are offered again on the next call; those
  already in the window are dropped by their key. A line that alone is
  over the budget is cut to fit, so no memory can hold the watermark
  for good, short of a window cap the operator sets
  (`funnel.window_budgets.digest`), which holds entries until a
  compaction makes room. The funnel reports an entry admitted even when persisting
  its row failed (`funnel.py`), so such a row is in the live window and
  lost at a restart. At the strategy's first start the
  watermark is the highest memory id there is, so history is not
  replayed into the core.
- **Duplicates.** The funnel matches a key against the same source's
  rows only. A memory the core's own retrieval admitted under source
  `memory` can therefore also arrive once under `digest`.
- **Logging.** Each admission is logged at info with its count and the
  largest `oldest_age_seconds` in it: the measured lag of §3.7.4. When
  any are left out, or `since` returned a full page, one warning per
  `anomaly_interval` names the count
  and the funnel keys that limit the source (`funnel.budgets.digest`,
  `funnel.window_budgets.digest`).
- **No turn.** Nothing is enqueued. A core with no call to make is told
  at its next idle beat (TRD §5.3), whose notification starts a turn
  and so a call.

**What the lag is made of.** At each model call on `core:main:<uid>`
that the hook precedes, the core has been offered every peripheral
memory indexed before that call, less what the budget or the page of
20 left out. A
peripheral message's consolidation is due `consolidate_max_age` after
the message was persisted. These add to that:

- a consolidation of the channel already running, by its duration: the
  rule is re-evaluated when it ends, and fires at once;
- a peripheral turn in flight, by that turn;
- the consolidation's own wait for a head and its run;
- each attempt that fails or is skipped, by one more age. Their number
  has no limit; each logs a warning;
- time the daemon is down, in full;
- a blocking fork, during which `core:main:<uid>` makes no call, by
  `fork.blocking_wait` (TRD §7.3). The fork itself reaches the memories
  by retrieval (TRD §7.7);
- a backlog over the budget or the page of 20, by the calls it takes
  to drain, or over a window cap, until a compaction makes room.

**What the rows cost.** A CONTEXT row is never summarised, and it
leaves the window when a compaction's cut passes it (`docs/design.md`).
After that the memory is reached by retrieval, like any other. Base
§3.7 Memory item 1 leaves CONTEXT rows out of the count of
un-consolidated messages, so a digest row does not by itself start a
consolidation in the core.

**Cost on the periphery.** An active channel costs one consolidation
call per `consolidate_max_age`. A fork of the channel's window prefills
only the new tail when it runs on the model that holds that window.
With one model, as in phases 1 to 5, it does. With two, K2 has an
unmapped `consolidation` stage fork on the peripheral's profile, so it
does there too. An operator who maps the stage in `llm.stages` puts
every channel's consolidation on that one profile and loses this for
the channels on the other.

## 11. Storage and restarts

All tables are in `sessions.db` (base principle 8), created by the
plugin that owns them.

| Table | Owner | Columns |
|---|---|---|
| `cp_state` | `core` | `key`, `value` (holds `core_uid` and `digest_memory_id`) |
| `cp_forks` | `forks` | `uid`, `parent`, `fork_point`, `blocking`, `route`, `wait_started`, `own_lease`, `state`, `ended_at` |
| `cp_stops` | `core`, `forks` | `channel_id`, `kind` (`stop` or `pass_through`), `set_at`, `set_by` |
| `cp_prompt_revisions` | `periphery` | `id`, `channel_id`, `text`, `trace_id`, `chain_id`, `state`, `verdict_ok`, `verdict_reason`, `created_at`, `applied_at` |
| `cp_work` | `core` | `id`, `kind`, `channel_id`, `chain_id`, `started_at`, `ended_at`, `outcome` |
| `cp_requests` | `core` | `id`, `command`, `argument`, `created_at`, `applied_at` |
| `slot_saves` | `slots` | TRD §8.4 |
| `outcome_labels` | `outcome_log` | `trace_id`, `channel_id`, `kind`, `source`, `reason`, `created_at` |

**At startup:**

- `core` registers the core's channel, and `forks` every fork in state
  `running`, with its `head_from` and `lease_key` as recorded; each
  replays from the log like any channel (gap 4), and
  queued items return once #318 lands.
- `forks` re-holds the queue of each parent still inside its wait
  bound, and applies the bound at once to any past it.
- Each `cp_stops` row is re-applied, and one warning per row names the
  channel and how long it has been stopped.

**The operator's path (§3.3.7, gap 9; TRD decision A5).** The strategy
registers two subcommands in the `corvidae.commands` entry-point group:

- `corvidae resume <channel-id|--all>`
- `corvidae fork end <fork-uid>`

Each writes a `cp_requests` row from a second process, as `corvidae
redact` writes to the same database. `core` reads the table every
`request_poll` (5 s) with `Clock.call_later` (`corvidae/clock.py`) and
applies each row; a resume removes the `cp_stops` row. No channel queue
is involved.

**Editor sessions (open question 13; TRD decision A6).** Every plugin in the
strategy's distribution except `limits` returns from `on_init` without acting when `config["_acp_mode"]`
is set, which `corvidae acp` does (`corvidae/channels/acp.py`), and
logs one info line saying so. An editor session runs the base topology,
so there is one core and one idle actor per goal store. `corvidae
serve` and `corvidae cli` both run the strategy.

## 12. Configuration

```yaml
core_periphery:
  core:
    profile: main              # K2
    idle_after: 3600           # seconds of system idle before an idle beat
    idle_prompt: {file: prompts/core-idle.md}
    attach_cli: false
    latency_bound: 120         # seconds; needs #365
    observables_period: 60
    anomaly_interval: 900
    error_threshold: 3
    request_poll: 5
  periphery:
    profile: small             # K2; unset = the core's
    tools:
      disabled: [shell, write_file]
    max_context_tokens: null   # unset = the channel's own setting
    prompt: {file: prompts/periphery.md}
    criteria: {file: prompts/periphery-criteria.md}
    force: null                # always | never
    summary_max_chars: 600
    verbatim_max_chars: 500
    consolidate_max_age: 300   # seconds; the age of TRD §10.5
    structural:
      attachment: true
      first_sender: true
      max_chars: 1500          # null turns the rule off
    rewrites:
      frozen: false
      frozen_channels: []
      max_chars: 4000
  fork:
    blocking_wait: 900
    idle_after: 600
  slots:
    skip_restore: false
    max_bytes: 8589934592
    providers:
      local:
        slot_save_path: /var/lib/corvidae/slots
limits:
  inbound_per_channel: 50
  notice_interval: 300
  max_waiting: 200
  max_hops: 4
# funnel:
#   budgets:
#     digest: 512              # tokens of peripheral memories per core call; absent = funnel.default_budget
goals:
  block_share: 0.05            # share of the core's max_context_tokens
memory:
  consolidate: true            # per-channel tunable; false on forks
  consolidate_max_age: null    # per-channel tunable; set on peripherals from the key above
appraisal:
  enabled: true                # per-channel tunable; false on peripherals and forks
# <stage>.fork_blocking: true | false   # per forking stage; unset keeps today's behaviour
```

Every key is validated at startup; a bad value aborts with the key, the
value and the accepted values (base principle 5). Each is documented in
`docs/configuration.md` and `agent.yaml.example` in the phase that adds
it.

## 13. Build order

Each phase leaves a system that runs, and ends with harness probes
(`harness/README.md`) whose pass criteria are read from `sessions.db`
or the log. Phases 1 to 5 run with the core and peripherals on one
model and one head (§3.8.4, §3.8.6), on the kernel's default lease
order.

| Phase | Builds | Needs | Probe |
|---|---|---|---|
| 1 | K1; `limits` (TRD §9.1, §9.2) | #814 | a task started at `max_waiting` returns the refusal text and a warning is logged; a tool result at the same depth is accepted |
| 2 | K9, K10; `core`: channel, uid, `attach_cli`, idle beat, observables (TRD §5.5, the phase-2 part); goal block | #1031, phase 1 | the core's channel id is the same across a restart; an idle period produces one `idle` row on it; the system prompt's hash is equal on every call between two compactions while a goal changes |
| 3 | K2, K3; the `appraisal.enabled` switch; `instruct` on the core; `cp_work`, `work_status` and `decline_escalation` (TRD §5.6); the `error_threshold` anomaly; `periphery`: settings, sections, `escalate`, structural rules, `force`, exact wording, labels, chains, the `notify_other_channel` change of TRD §9.3; the age trigger and the digest (TRD §10.5) | #234, phase 2 | with `force: always`, a message produces an `escalation` row on the core and a reply on the channel; with `never`, none; a forced loop stops at `max_hops` with the user told; the core's exact text is byte-equal in the peripheral's window; a peripheral message that is not escalated is in a CONTEXT row at the core's tail by the core's first call after `consolidate_max_age` and the consolidation's duration, and no core turn was started for it; with the peripheral on a second profile and the `consolidation` stage unmapped, that consolidation's request goes to the peripheral's profile |
| 4 | K7; tail import; criteria rewrites | #235, phase 3 | a subagent given `tools=["read_file"]` has no other tool in its request; a rewrite is absent from the peripheral's prompt before its next compaction and present after; a frozen channel refuses |
| 5 | K4, K5, K6, K8; `forks`: long-lived forks, blocking, routing, bound, join, abandon, refork, pass-through, restart, `hold_queue` and `resume_queue` in `core` (TRD §8.3, on the kernel's stop and resume), the `memory.consolidate` switch, `cp_requests` with `corvidae resume` and `corvidae fork end`, `<stage>.fork_blocking` | #318, phase 3 | a blocking routed fork that never joins: after `blocking_wait` an escalation is handled by the main context; a join appends one row to the parent whose effects list names the message the fork sent; a restart mid-fork leaves the fork running and the wait still bounded; under pass-through an inbound message is in the peripheral's window with no model call on that channel; `hold_queue` on the core's own channel returns the refusal text and stores no `cp_stops` row; after `hold_queue` on a fork that shared its parent's lease, the parent's next call is granted |
| 6 | `slots` on one head: order, paging, stamped saves, shutdown save, the other four tools of TRD §8.3; consolidation's yield | #366, #365, phase 5; deferred (D1) | after a page-out and page-in the next call's `cache_n` is within 5% of the prompt length; a file with a changed stamp is deleted with a warning and the call prefills; `pin_context` on a hosted provider returns the no-effect text |
| 7 | Two models on separate providers, more heads, router mode: not designed | #1009's report, phase 6; deferred with it | alternating a core and a periphery call loads neither model twice (§3.8.5) |
| 8 | Trust stamping | #138 | an instruction written after an untrusted escalation carries the untrusted level |

K3, K4, K5, K6 and K8 were in phase 1 until the owner moved each to
the phase that first uses it (2026-11-17): built earlier, each would
have shipped with no caller. `cp_work`, `work_status` and
`decline_escalation` moved from phase 2 to phase 3 for the same reason
(owner, 2026-11-18; #1058, ruling 2): every kind `cp_work` records is
first built in phase 3 or later. Phases 6 and 7 are deferred (TRD decision D1). Phase 7 is not designed here. Until #1009 reports, all that exists is
the README's account of router mode.

Each phase updates `docs/design.md`, `docs/configuration.md`,
`docs/plugin-guide.md` and `docs/threat-model.md` for what it adds. The
threat-model entries known now: `escalate` and `instruct` (untrusted
text reaches the core and comes back out), criteria rewrites on by
default (TRD §6.5), `work_channel`, the two subcommands, and the digest
(TRD §10.5): memories made from users' messages reach the core's window
without the core asking for them, and a summary that contains another
memory's marker makes the funnel drop that memory as a duplicate, as
source `memory` already allows.

## 14. Requirements to sections

Explanatory and merged items of the supplement are left out. "Base"
means an existing base mechanism meets it and nothing is built.

| Supplement | Met by | Phase |
|---|---|---|
| §1 (the strategy is plugins; the base topology stays) | TRD §3.1 | 2 |
| §4 (a peripheral per transport channel; the core a channel) | TRD §5.1, §6.1 | 2, 3 |
| Principle 17 | TRD §5.1, §6.1, §6.6 | 2, 3, 4 |
| Principle 18 | TRD §8 | 6 |
| Principle 19 | TRD §6.3 | 3 |
| Principle 20 | TRD §5.3, §10.1 | 2 |
| Principle 21 | TRD §5.4 | 3 |
| §3.1.1 | K9, TRD §5.1, §5.2 | 2 |
| §3.1.2 | TRD §3.3, §5.3; §2 (critique, #1031) | 2 |
| §3.1.3 | TRD §8.3 | 5, 6 |
| §3.1.4 | TRD §10.1 | 2 |
| §3.1.5 | TRD §6.5 | 4 |
| §3.1.6 | TRD §5.4 | 3 |
| §3.1.7 | TRD §7.1, §7.3 | 5 |
| §3.1.8, §3.1.9 | TRD §5.4 | 3 |
| §3.1.10 | TRD §10.2, §10.3 | 3 |
| §3.1.11 | K10, TRD §5.5 | 2, 3, 5, 6 |
| §3.1.12 | TRD §5.5 | 4 |
| §3.1.13 | TRD §10.5 | 3 |
| §3.2.1, §3.2.7 | K2, TRD §6.1 | 3, 7 |
| §3.2.4 | TRD §6.2, §6.5 | 3, 4 |
| §3.2.5, §3.2.6 | Base; TRD §6.1 | 3 |
| §3.2.8 | K3, K7, TRD §6.1 | 3, 4 |
| §3.2.9 | K7, TRD §6.6 | 4 |
| §3.3.1–3 | TRD §8.2, §8.3 | 6 |
| §3.3.4, §3.3.5, §3.3.8 | TRD §8.1, §8.3 | 5, 6 |
| §3.3.6 | TRD §8.2, §8.3 | 5, 6 |
| §3.3.7 | TRD §7.3, §8.2, §11; #365, #366 | 5, 6 |
| §3.4.2, §3.4.3 | TRD §6.3 | 3 |
| §3.4.4 | TRD §6.5 | 4 |
| §3.4.5 | TRD §6.4 | 3 |
| §3.4.6 | TRD §6.3; the trust label, TRD §9.4 | 3, 8 |
| §3.4.7 | TRD §9.4 | 8 |
| §3.5 preamble | K6, K8, TRD §7.1, §7.2, §7.4 | 5 |
| §3.5.1–3 | TRD §7.1, §7.4 | 5 |
| §3.5.4 | TRD §7.4, §7.5 | 5 |
| §3.5.5 | TRD §5.4, §5.6, §7.1 | 3, 5 |
| §3.5.6 | K5, TRD §7.6 | 5 |
| §3.5.7 | K4, TRD §7.3 | 5 |
| §3.6.1–3, §3.6.5 | TRD §10.1 | 2 |
| §3.7.1–3 | TRD §10.2, §10.3; compaction of a task-bound peripheral is open (TRD §15.1); its critique is settled (#1031) | 3, 5 |
| §3.7.4 | TRD §10.5 | 3 |
| §3.8.1 | K2 | 3, 7 |
| §3.8.4, §3.8.6 | TRD §13 (phases 1–5 run this way) | 2–5 |
| §3.8.5 | not designed; waits for #1009 (TRD §13) | 7 |
| §3.9.1 | TRD §5.4 | 3 |
| §3.9.3 | Base (`on_notify`) | — |
| §3.9.4 | Base §3.5.9 through #234 | 3 |
| §3.10.1 | TRD §5.6 | 3 |
| §3.10.2 | TRD §6.4 | 3 |
| §3.10.3 | TRD §8.5 | 6 |
| §3.10.4 | TRD §7.4 (the `cp_forks` state changes are logged; an abandonment at warning with its reason) | 5 |
| §3.10.5 | TRD §5.5 | 2 |
| §3.11.1 | TRD §6.3 | 3 |
| §3.12.1 | TRD §9.1, §9.2 | 1 |
| §3.12.2 | K1, TRD §9.2, §9.3 | 1, 3 |

## 15. Open, and not in this document

### 15.1 Open for the owner

- **Compaction of a task-bound peripheral.** §3.7.1 says compaction
  runs for every context and excepts only consolidation. The
  `subagent` loop has no compaction and is bounded by
  `subagent.max_turns`. Either §3.7.1 gains the exception or K7 gains
  the mechanism. This document builds neither and recommends the
  exception: a task that outgrows its window is better split by its
  launcher. Critique is settled (#1031): a task's result is critiqued
  where it lands, and the loop itself is not.
- **K8 leaves one edge undesigned.** The owner approved K2 to K6, K8
  and K9 as kernel contracts on 2026-11-16 (TRD §4). The edge: a fork's call already waiting under
  the shared key when that key is stopped waits with it, and the switch
  of TRD §7.2 applies from the fork's next call.
- **#365** was not among the prerequisites accepted in session
  (TRD §2).
- **A consolidation of very little.** The age trigger consolidates a
  range however small (§3.7.4), so a channel with one short message
  per period writes one memory per message. Whether such a range
  should be held for the next is not decided; the measured lag and
  the memories written are the evidence.
- **Phase 7 is deferred with phase 6** (TRD §8). The owner named
  phase 6; phase 7 needs it.
- **`consolidate_max_age` defaults to 300 s**, unmeasured.
- **The digest's default budget is small.** At `funnel.default_budget`
  (512 tokens) and memory's entry target (256), one or two memories go
  in per core call. `funnel.budgets.digest` is unmeasured like the
  other defaults.
- **D4 beyond forks.** The owner approved the rule for forks; K2 also
  applies it to a stage's other work for the same channel, so that
  base §3.8 item 4's one profile per stage holds per channel.
- **Trust on a digest row** (TRD §10.5) waits for #138 like the rest
  of TRD §9.4: the row would carry the lowest trust level of its
  memories.
- **The ceiling's treatment of stopped channels** (TRD §9.2) is the
  agent's judgment.
- **A hold on a provider with no heads has no effect** (TRD §8.3). The
  kernel has no stop there, and §3.3.7's "keep a paused channel
  stopped" is then met only as §3.3.8's no-effect report. K4's queue
  hold could serve instead, if the owner wants pausing to hold on
  hosted providers. Where only some providers declare heads, a channel
  served by one that does not is recorded as stopped while nothing
  holds it.
- **TRD decision C4 calls the revision table append-only**, and TRD
  §6.5 updates a row's `state` in place. Either the wording or the
  design gives way.

### 15.2 Not designed here

- **The editor as a pass-through peripheral.** It needs `corvidae acp`
  to relay to the daemon, a base PRD change (open question 13).
- **Cryptographic provenance** (#507).
- **Logprob confidence** for escalation, which waits for labels (open
  question 2).
- **A peripheral serving several channels** (open question 7).
- **Consolidating before a page-out**, which waits for measurement
  (open question 16).
- **The commitments feed** to idle turns (TRD §5.3).
- **Tone continuity**, which the operator judges (open question 4).
