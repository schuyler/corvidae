# Corvidae: product requirements

What corvidae is for, what it must do, and what it has decided not to do.
This document defines requirements: what must be true, and why. It is not
a record of what the tree does now, and it never says what is or isn't
built. When the tree and this document disagree, either the tree is wrong
and a GitHub issue tracks the gap, or the requirement is wrong and the fix
is an edit to this document, reviewed like code. This document is the
reference for the rest of the project: design docs, plans and code derive
from it, and it cites none of them. It cites only external references (the
Persyn BOOTSTRAP specification, the GitHub issues and operator directives
that record decisions) and the threat model it requires (principle 9).

## 1. What it is

Corvidae is a personal agent daemon: one Python asyncio process on one
machine that connects a local LLM to the chat surfaces its operator already
uses (IRC, Signal, Discord, a browser page, a terminal, an editor over ACP),
runs a tool-calling loop, and remembers.

Its purpose, in priority order. First, corvidae is a minimal, pluggable
testbed for developing agentic behavioural and cognitive techniques, for an
agent that can be autonomous and self-directed, is one actor across all its
channels, and whose life continues between messages: it consolidates,
revisits, and, with agency plugins loaded, acts on its own initiative. The
kernel stays minimal so those mechanisms can be swapped and measured
(principle 6). Second, it is a harness robust to its stated threat model
(`docs/threat-model.md`). When the two pull apart, flexibility for
experimentation wins and the threat model records the cost.

It serves one operator. Three deployment shapes define the requirements:

- **A harness instance** ("Buster"): minimal profile, local IRC, a standing
  llama-server, scripted probes with mechanical pass criteria.
- **A personal instance** ("Lex"): a strict-template model (Swift-1.5, a
  Qwen 3.8 fine-tune) on a single-KV-slot llama-server router, Signal as the
  primary transport, running unattended for weeks. The strict-template
  and Signal requirements below come from this one, and every requirement
  must hold on its single slot.
- **An editor session**: corvidae as an ACP agent spawned by bb or Zed,
  one subprocess per editor connection, sharing the runtime with the daemon.

It is not multi-tenant, not distributed, not sandboxed, not published on
PyPI, and not stable. Interfaces change without notice, and that warning
is load-bearing: renamed hooks get no compatibility aliases.

## 2. Principles

Every requirement in §3 is filtered through these. A change that violates
one is a bug even if its tests pass.

1. **Every capability is an optional plugin.** The kernel is the agent
   loop, the queues, the context window, channels, hooks, tools, the LLM
   client, the task queue, the clock, and persistence. Everything else loads
   from an entry point, can be disabled by name in `plugins.disabled`,
   declares its dependencies in `depends_on` (or `optional_depends_on`
   for one it works without), and documents what the system
   does without it. The cognition plugins (memory, retention, funnel,
   appraisal, critique, outcome log, memory tools, goals, trust,
   sensitivity) are
   a second distribution behind a `cognition` extra; the kernel installs,
   boots, and passes its tests
   without them and without `sqlite-vec`. Heavy third-party clients that
   only one transport needs are extras too (`acp`, `discord`). A reader of
   the README can tell what they get without the extras. Plugins load once
   at startup from entry points; there is no runtime plugin loading, and
   changing the plugin set is a restart.

2. **The context window grows at the tail and truncates at the head.**
   Nothing is inserted, removed, or mutated mid-window, because that
   invalidates the KV cache from that point on. Retrieved
   memories, critique verdicts, reminders, and grounding are appended as
   CONTEXT or notification rows at the tail and retire by aging past the
   compaction boundary. Prompt-build rewrites are pure functions of each
   message and the rows before it so the cached prefix stays stable. Prefill is the expensive
   part of a call on any transformer, so context is reused, not rebuilt:
   background calls that need the conversation fork it (§3.8) rather than
   rebuilding it. This is principle 16 applied to prefill: a reused prefix
   is a context switch avoided.

3. **Background cognition avoids blocking the main turn.** The point is
   interactivity. The agent loop is single-turn dispatch: one LLM call per
   queue item, tool calls as tasks, results re-entering as notifications.
   Appraisal, critique, consolidation, and retention run opportunistically
   as silent tasks or plugin-owned tasks, and shape the *next* turn.
   Background work runs in the gaps between main turns. Deferral is bounded
   so interactivity can't starve it: an exchange's own background work
   (stage-2 appraisal, critique) forks the exchange's prefix and runs under
   the same lease before the lease passes on (§3.13.4), so it finishes
   before the channel's next main turn and its results are in place for
   the turn that reads them. Other background work (upkeep) waits its turn
   for a head when every head is leased, and upkeep past its latency bound
   takes the next free head ahead of work within its bound, so traffic
   can't starve it (principle 13; §3.13.5). Where the pool has a free head,
   background work runs beside the main turn rather than yielding to it
   (§3.13.6). A free head is not a free slot: on a server with one slot,
   a background call already decoding delays the main turn whatever the
   lease, so the requirement is to keep background work's share of the
   pool small (bounded output, §3.8) and out of the way; it is not a
   guarantee of zero delay. How a pool's
   capacity is shared is §3.13 (principle 16). The inbound gate runs inside
   the transport read path and never calls a model.

4. **The conversation log is append-only.** `message_log` rows are never
   deleted and their order is never rewritten. Compaction hides; memory
   demotes. The one sanctioned mutation is the operator-only `corvidae
   redact` tombstone, which exists because a store with no deletion path is
   a privacy defect, not an ethos.

5. **Fail soft, never silently.** Background paths catch, log with the
   traceback, and continue; `except Exception: pass` is forbidden. Before
   failing, exhaust the cheap rescue (retry, fall back, skip the corrupt
   record, degrade to a partial result) and report what was reduced. Every
   human-facing error names the identifier that failed and the next step;
   exception text belongs in the log line. A plugin with no config block is
   silent; a plugin with a malformed one is loud and aborts startup. Only
   startup may stop the process, on paths the requirements name
   (§3.1.17); after it, the most any failure costs is one turn on one
   channel, whatever the channel holds (§3.1.18).

6. **Behavioural tests are the bar.** The framework can offer any
   mechanism, and which ones a deployment runs is the operator's choice
   (principle 1). Evals measure what a mechanism does, so the operator can
   choose; they don't decide what gets built. Deterministic metrics
   against operator-authored fixtures run in CI; LLM-judged suites run out
   of band and are tracked as benchmarks, never as gates.

7. **Cheap, always-on perception gates expensive cognition.** One appraisal
   signal has two consumers: engagement and consolidation strength.
   Epistemic checks stay independent of the appraisal scores: the
   provenance check fires on any claim about the past and is judged against
   records, never on generator confidence. Thresholds ship as commented
   best guesses and are tuned from the outcome log. Fitting them in a
   closed loop, with the agent applying what the outcome log suggests, is a
   capability the framework can permit (principle 15). Letting cheap work
   gate expensive work is principle 16 applied to cognition.

8. **Local-first: one process, one machine, one SQLite file.** No message
   broker, no worker pool, no workflow engine. The database runs in WAL
   mode as a requirement, because a second process (the redact CLI) and a
   second connection (the appraisal probe) read it concurrently. This is
   about the agent's process and state; model endpoints are wherever the
   operator points them (§3.6).

9. **The operator is trusted; everyone else is text.** Prompt injection is
   the threat, not other tenants. Anything that arrived from a channel, a
   web page, an MCP server, or a memory distilled from those is data, and
   is framed as such at one chokepoint: the kernel's prompt build
   (§3.3.11), plus the funnel's own framing of what it retrieves (§3.7
   Funnel). The operator decides the trust
   boundary, not the architecture: transports are default-deny with silent
   refusal, so the operator decides who may reach the agent (the
   allowlists, plus anyone the agent messages first, unless the operator
   locks that, §3.5.9), and which safety plugins load (trust and
   sensitivity, and others added over time, as with sandboxing). Channels
   are not a boundary (principle 14): what the agent knows, remembers and
   says crosses between them freely. Fixed floors protect the operator's
   own control surface rather than draw a trust boundary, and stay fixed:
   web auth cannot be turned off (§3.2 Web), `redact` is never a tool (§3.7
   Memory 8), `set_settings` can't lift a lock or change the lock list, the
   request-body keys the agent loop builds (`messages`, `tools`,
   `tool_choice`, `stream`) are never overridden, and transport metadata is
   out of the agent's reach (§3.5.6). The floors bind the framework's own
   surfaces; an agent permitted to edit its own code or configuration and
   restart (principle 15) can change anything, floors included, and the
   threat model records that cost. Apart from these floors and the
   append-only log (principle 4), what the agent may change about itself is
   not fixed: every other restriction on it is a lock the operator can lift
   (principle 15). The model's authority is unbounded and unsandboxed by
   decision; `docs/threat-model.md` is the standing record of what that
   costs and is updated whenever untrusted text can reach somewhere new or
   a tool can do something new.

10. **Channel ids are persistence keys and never derive from names.**
    `transport:scope` keys `sessions.db`, the JSONL logs, the channel a
    memory record came from, and the schedule. Scopes use identifiers
    that cannot be renamed or go missing: Signal's ACI, Discord's numeric
    ids, a web slug the operator chose, an ACP session id reused across
    load and resume.

11. **Every LLM call is metered at the one chokepoint it passes through.**
    `LLMClient.chat` fires the observability hooks, so compaction,
    subagents, consolidation, appraisal, critique and forks are all
    attributed by stage, channel, trace, trigger and mode. Eval questions
    are denominated in tokens; this is an eval prerequisite. The same
    record names the profile and model each call was sent with and how
    much of its prompt came from cache, which is what a pool's warm heads
    are tuned against (§3.13.10).

12. **How the work is done.** Red/green TDD; every test under a timeout;
    docs current before a change is complete. A plan for a change takes
    its requirements from this document, is written to be executed
    without architectural judgment calls, tags each requirement with its
    provenance, and is gated by the owner. Plans
    pass code-grounded adversarial review until a round comes back clean.
    Review findings gate on consequence (a security hole, data loss, a
    hang or crash, a wrong result the user sees), not on reachability.
    Behaviour that only shows across a restart, a compaction, two
    interleaved messages, or a period of silence is proven in the harness,
    not in pytest. Shakedown sessions discover; probes claim.

13. **The agent lives between messages.** The kernel turns three things
    into events: what arrives, what finishes, and time passing. Plugins
    decide what each means. There are two kinds of time-driven work. Upkeep
    (consolidation, retention, scans) never starts a turn, though it may
    notify the operator, and runs whether or not anyone is talking and
    whatever agency plugins are loaded. Agency is a turn started on time or
    on the agent's own initiative rather than as a continuation of an
    exchange: reminders and the heartbeat are agency;
    provenance objections and task results continue an exchange and are
    not, and neither is a turn a note wakes on another channel (§3.5.10):
    a continuation can cross channels, and a chain of continuations is
    agency only if the exchange that started it was. Stylistic critique
    objections start no turn at all. Agency is
    optional. Without agency plugins the agent never initiates a turn and
    has no reminders or heartbeat, and its upkeep still runs. No plugin's
    documented behaviour may depend on traffic it does not control.

14. **One actor, one identity, many channels.** Identity, memory, working
    knowledge and the ability to act are the agent's, not the channel's.
    The agent presents the same identity on every channel; what changes
    from one channel to the next is register (§3.3.10), never who it is
    (§3.3.8). Any turn can speak on any channel a transport owns, leave a
    note for another channel's context, and read another channel's recent
    conversation (§3.5.9–11), and retrieval spans the whole memory. There
    are no channel walls to configure: the per-channel queue and window
    exist for concurrency and the KV cache, and per-channel settings for
    register and for bounding what a channel's participants can make the
    agent do (its tool set and locks, §3.5.2, §3.5.6), never for what
    crosses between channels. An operator who wants two identities runs two
    instances. Safety comes from who may reach the agent (inbound
    default-deny) and from restrictions the operator can apply (principle
    15): trust (what came in is weighted and labelled by its source) and
    sensitivity (what enters a window is filtered by who is present there).
    Neither is a requirement on the agent's behaviour; each is a capability
    the framework offers. Without them, nothing is quarantined and nothing
    is filtered by audience. That has a cost where people the operator
    never allowed can read the agent's replies (a Discord guild channel, a
    public IRC channel): something said on a private channel can surface
    there. `docs/threat-model.md` records it (principle 9).

15. **The agent can modify itself; the operator decides how far.** The
    capacity to self-modify and self-improve is a design goal. The
    framework can let the agent change, at runtime and through its own
    tools:
    - its prompts: every prompt the framework sends to a model on the
      agent's behalf, among them the system prompt (the identity and
      each channel's register, §3.3.8), every background
      stage's fork and standalone instructions, the critic's and
      appraiser's prompts, and the subagent and heartbeat prompts;
    - its skills (§3.7 Trust);
    - its model and inference parameters, `model` and `logprobs`
      included (§3.5.6);
    - its tunables, framework settings, caps and thresholds, including
      fitting thresholds from the outcome log in a closed loop
      (principle 7);
    - its tool set, within the tools already registered (§3.5.2);
    - its own code, configuration, prompt files and plugin set, applied
      by restarting itself (§3.11.13). There is no runtime code loading
      (principle 1): a change to code or to the plugin set takes a
      restart.

    No deployment is required to allow any of it: the framework must let
    the operator lock or disable anything the agent could change about
    itself. A lock can freeze a value or bound it, to a range or a set of
    allowed values. Every restriction on self-modification is a lock: a
    mechanism the framework supports and the operator configures, never a
    prohibition in the architecture. Operator config can lift any lock.
    The floors of principle 9 are the exceptions and stay fixed (they bind
    the framework's own surfaces; see principle 9 for a self-edit), and
    the conversation log stays append-only (principle 4). Each capability
    has a route that keeps the agent loop consistent. A request-body
    override carries model parameters, never the loop's own `messages`,
    `tools`, `tool_choice` or `stream` (§3.5.6). Cache reuse is optimized
    for the general case: a change the agent makes that would alter what
    is already rendered in the window (the system prompt, the tool set,
    how earlier rows are framed or stripped) takes effect at the channel's
    next compaction boundary or conversation start, not mid-window
    (§3.3.8; principle 2). An agent that wants a change applied at once
    makes it persistent and restarts. What each unlock costs is recorded
    in `docs/threat-model.md` (principle 9).

16. **Compute is finite, and a context switch is what it costs.** Every
    model endpoint is a resource pool. A provider keeps a limited number of
    conversation prefixes warm, in its slots and, on llama-server, in host
    RAM (`--cache-ram`), and a call whose prefix isn't warm pays a full
    prefill, the expensive part of any call. When more queues are active
    than the pool can keep warm, each switch evicts another queue's prefix
    and the agent thrashes. Lex runs one slot to get the longest window, a
    deployment choice rather than an architectural one. The operator
    declares how many warm heads each pool has; work runs under leases on
    them, and contexts switch at call boundaries chosen to keep prefixes
    warm, not call by call (§3.13). Principles 2, 3, 7 and 11 and the fork
    requirements (§3.8) follow from this one: reuse the prefix, keep
    background work out of the main turn's way, let cheap work gate
    expensive work, meter every call.

## 3. Requirements

### 3.1 Kernel: agent loop, queues, traces

1. One serial queue per channel. A channel never has two turns in flight;
   a coalesced turn (item 19) answers several messages in one call. Queues
   are visible to other plugins by reference.
2. Single-turn dispatch. Each tool call becomes a `Task` on the shared
   task queue; the next LLM call waits until the last result of a batch
   clears the channel's pending set. User messages may interleave
   mid-batch without disturbing it. The task queue has configurable
   concurrency and a bounded completed-result buffer.
3. `max_turns` is a per-exchange budget. The counter resets on any item
   that originates a new trace while no tool cycle is in flight, so a
   channel fed only by scheduler notifications never exhausts it. An item
   arriving mid-cycle is charged to the running cycle. An exchange's
   budget follows it to every channel it reaches: a turn a note wakes
   (§3.5.10) is charged to the budget of the chain that woke it, never
   given a fresh one.
4. Every inbound item carries a trace id. Transports mint it from a
   stdlib-only module and pass it to `on_message`; the Agent mints at
   dequeue for a notification that inherits none. The id and an opaque
   `annotations` dict ride every hook, every `Task`, every completion
   notification, and every usage-log row. A falsy id is a `ValueError`;
   omitting it is a hook-call error. `annotations` belongs to the plugins:
   its conventions (`annotations["origin"]` among them) are theirs, and core
   never stamps meaning into it. What the kernel certifies goes in the
   item's `metadata` instead: kind, origin scope, cause, sender, and whether
   the message addressed the agent. Only kernel code sets `metadata`, from
   what it observes itself; it is never a hook argument, never changes once
   set, and persists with the message. A message coalesced into
   another's turn (item 19) keeps its own trace through admission and
   persistence only; the owning trace carries the turn's hooks.
5. The inbound gate (`should_process_message`) resolves reject-wins over
   every implementation. After resolution core fires exactly one of
   `on_message_admitted` / `on_message_rejected`, before `on_message`
   returns, and `on_message_persisted` once per trace when the originating
   row is written, with its rowid (null for a rejected message). Transports
   match these back to their message by trace id, never by ordering.
6. The emitted prompt never ends on an assistant message, because
   llama.cpp reads that as a prefill and rejects it under
   `enable_thinking`.
7. A message that arrives before the Agent has started is held until it
   has, not dropped.
8. A tool call that never returns cannot wedge a channel: `dispatch` bounds
   every call with `tools.call_timeout` (default 600 s, per-tool overrides
   in `tools.timeouts`, skipped when the tool's own timeout is not longer),
   and a timeout returns an error string to the model so the turn
   continues.
9. Tool-dispatch failure paths (no task queue, a raising `on_notify`) still
   deliver an error result to the model; a silent stall is never an
   outcome.
10. A hook's name states its resolution semantics: `on_`/`before_`/`after_`
    observe (return value ignored; in-place mutation of passed objects
    allowed), `should_` vetoes (reject wins), `resolve_` takes the first
    non-None result, `transform_` is a wrapper chain over a seed, and a
    short allowlist of bare imperatives (`send`, `register_tools`,
    `persist_message`) documents what the caller does with results. A test
    enforces the prefixes against pluggy options for the kernel's
    `AgentSpec`. A hook only one plugin calls is owned by that plugin
    (`hookspecs` on its class), not added to `AgentSpec`. Its name carries
    the owner's plugin name right after the prefix
    (`on_<plugin>_<name>`), it uses a prefixed class (bare imperatives are
    kernel-only), and it has exactly one owner. Startup checks all of this
    and fails naming the hook, the plugin and the fix.
11. A new hookspec, kernel or plugin-owned, is admitted only when it is a decision point, an
    observation with at least two plausible implementers, or the pair of
    an existing observability hook. A single-consumer observation gets a
    `annotations` entry or a return value on an existing hook instead.
12. Hook parameters with defaults can be dropped silently by pluggy; a
    startup guard rejects implementations that re-declare a spec-required
    parameter with a default.
13. Plugin dependencies are declared in `depends_on` and validated before
    `on_start`; plugins resolve each other with `get_dependency`, never by
    registration order. A plugin that works without another declares it in
    `optional_depends_on`: its absence is not an error, and impls of the
    absent plugin's hooks stay inactive. A hookimpl with no spec is logged
    at startup, at WARNING unless its owner is an absent optional
    dependency. `on_start` is broadcast concurrently; the Agent
    starts after the broadcast, and producers that inject notifications
    wait on `Agent.started`.
14. A plugin that fails to import or instantiate is logged and skipped; a
    misspelled `plugins.disabled` entry shows up in the startup log as
    blocked-but-still-loaded.
15. The kernel provides a clock: plugins can have work run after time
    passes, with no inbound traffic. The scheduler (§3.9) uses it.
16. A channel is idle when it has had no inbound item (a message or a
    notification; scheduler firings and task completions count) and has no
    queued or in-flight work; the system is idle when every channel is. A
    plugin can act on a channel, or the system, having been idle for a
    period the plugin chooses, and keeps getting that signal at a bounded
    rate through sustained quiet. The signal never fires for a channel with
    work queued or in flight. No plugin needs another component's queues to
    learn that a channel is idle. A plugin may give its idle work a latency
    bound: the quiet the work needs shrinks the longer it has waited, so it
    runs no later than the bound whatever the traffic (principle 13).
17. Only startup may stop the process. Its fatal paths are these:
    loading and validating `agent.yaml` and plugin config blocks
    (principle 5), the plugin checks of items 10, 12 and 13, and opening
    the database and applying its schema steps (§3.4.5). Once the Agent
    has started, no exception stops the process: a hot reload that fails
    changes nothing (§3.11.3), and every other failure degrades with a
    logged warning (principle 5). Adding a fatal path is a PRD change.
18. Nothing a channel holds can make its turns fail repeatedly. What a
    channel holds is its window, the rows that reload into it, and its
    queued work, and the guarantee covers any content they carry: a
    message, a tool result, a model reply, a row a plugin injected. The
    most one item can cost is its own turn, shared with any messages
    coalesced into it (item 19).
    1. Building the prompt cannot fail. Estimating the window's size,
       bringing it within budget (§3.3.2) and rendering it for the model
       handle every row the window can hold; a row they cannot handle is
       left out of the prompt and logged with its id, like a corrupt row
       on load (§3.4.3).
    2. A channel is wedged when its turns fail a configurable number of
       times in a row. Failures the backend causes don't count: an
       unreachable backend is retried, and that is not a wedge. A wedged
       channel is rescued by a fallback that cannot fail: it hides rows
       from the window, no more than it takes for a turn to succeed, and
       logs a warning naming the channel and the rows hidden. The rescue
       holds across a restart, and `message_log` keeps every row
       (principle 4).
19. Consecutive queued messages coalesce into one turn (#398). When a
    channel's queue takes an inbound message for its next turn, every
    inbound message waiting directly behind it coalesces into that turn and is answered
    by one model call. Each message is still persisted as its own row and
    announced (`on_message_persisted`) in arrival order. A turn in flight is
    never interrupted. Only consecutive inbound messages coalesce: a tool
    result, a notification or a retry stops the run and keeps its place, and
    notifications never coalesce. A coalesced burst is one exchange for
    `max_turns` (item 3). The last message's trace owns the turn: every
    turn hook, usage row and reply carries it, and the earlier messages'
    traces get admission and persistence only.

### 3.2 Transports

The contract every transport meets:

1. No config block: inert, nothing logged above DEBUG. Present but
   malformed: raise in `on_init` naming the field.
2. Default-deny sender allowlist in `should_process_message`. A refusal is
   one INFO line naming the sender; nothing is sent back and no liveness
   indicator is shown. The transport fires `on_message` for every decoded
   message regardless; the gate is the one place that vetoes. One
   acknowledged exception: a channel configured to answer only when addressed
   may drop a message that does not address the agent, before the kernel sees
   it. That is the operator's addressing rule rather than a
   judgement on the sender, so it is not a refusal and nothing is sent back;
   it is recorded for diagnosis only. Web substitutes a bearer token for the
   allowlist; the token is the authorization. A sender the agent has
   messaged first, one to one, is admitted on that channel (§3.5.9).
3. Transports subclass `TransportPlugin`, set `transport`, and override the
   `send_*` methods they can display. The one `send(channel, event)` hook
   does the ownership check once; a transport's `owns(channel)` predicate
   is overridable (Web also checks that it is enabled). Reasoning and tool
   status default to no-ops on chat transports.
4. Outbound text is split, never truncated, by UTF-8 byte length; the
   chunks concatenate back to the original text; the limit is at least
   six bytes or `split_message` raises. Discord's splitter also keeps code
   fences balanced across chunks.
5. Liveness (typing, read receipts) is shown only to allowed senders.
   Typing shows while the channel is owed a reply and stops as soon as any
   turn ends with a final, sent or withheld, even if another message is
   still waiting for its turn, or when nothing is owed; it is never left on
   indefinitely.
6. A refused sender never creates a persistent channel in the registry.
7. Scopes use stable identifiers (principle 10). The transport implements
   `resolve_channel` for ids the registry does not hold, so a reminder due
   after a restart finds its channel with its configured overrides; if the
   information needed to apply those overrides is not available yet, it
   returns None and lets the caller retry rather than caching a
   default-configured channel.
8. Reconnection uses one backoff ladder (10 s, doubling, capped at 300 s,
   reset on success), and each retry names the host, port and identity it
   failed to reach. Outbound messages produced while disconnected are
   queued, bounded, and flushed in order after reconnect; exceeding the
   bound logs the channel and the drop count.
9. Behaviour that cannot be driven end-to-end in the harness (Discord,
   Web, ACP have no local server) is pinned against fakes, and the design
   doc says so.
10. A transport delivers every emission on a channel it owns, whatever
    originated the turn (message, reminder, heartbeat, task). Liveness and
    reply bookkeeping tolerate a reply that answers no outstanding
    message.

Per transport:

- **CLI**: `cli:local`; stdin line by line; EOF at a terminal ends the
  session; EOF on a non-TTY under `serve` stops only the transport, so a
  service-managed config may keep a `cli:` channel. Banner and prompt
  print only at a terminal. Tool activity is shown as it happens.
- **IRC**: pydle; channel and private messages; 400-byte chunks. Private
  messages are off by default or gated by `irc.allow` (services-
  authenticated accounts where the network offers them), because a nick
  is unauthenticated and whoever holds it later inherits the channel's
  history and, since retrieval and reading span every channel (principle
  14), whatever the agent knows.
- **Signal**: newline-delimited JSON-RPC to an already-running `signal-cli`
  daemon over a unix socket; corvidae never spawns or supervises it. Scope
  is the sender's ACI. Config may name senders by E.164 in `allow:` and
  `channels:`; those resolve forward to ACIs in one batched call after each
  connect, never at startup, and a failed resolution retries after the next
  connect. Non-`dataMessage` envelopes (including the account's own sync
  traffic) are dropped. An attachment from an allowlisted sender is
  downloaded when its MIME type (or, with no MIME type, its legacy type)
  passes the download gate (the allowlist is specified in the threat model)
  and is named in the text: downloaded
  attachments use the major type (image, text, audio) or the subtype for
  application/* (pdf, json, xml); attachments that are not downloaded use
  the sanitized subtype (svg, html, ...), so the agent can see why nothing
  was fetched; with no MIME type, the legacy protobuf type name applies.
  The `getAttachment` RPC sends the attachment id only. Backlog older than
  five minutes is prefixed with its send time. Typing starts on the first
  outstanding message and refreshes every 10 s; the read receipt is sent at
  processing start, not arrival. Notifications are never handled inline in
  the read loop.
- **Discord**: discord.py's gateway behind the `discord` extra, default
  intents, and the privileged Message Content intent only when a server
  channel answers unaddressed — the transport default or any per-channel
  override. Replies in DMs and when mentioned (not
  `@everyone`, `@here`, or roles); with `require_mention: false` it replies
  to every allowlisted sender's server message, mentioned or not; drops bot
  authors, empty text, and unmentioned messages in a channel that requires a
  mention (§3.2 item 2); strips the mention token. Scopes: `discord:<channel_id>`,
  `discord:<thread_id>` (a thread has its own history),
  `discord:dm:<user_id>`. Every send passes `AllowedMentions.none()`. A bad
  token is logged once at ERROR and not retried. Gateway events are queued
  and consumed by one processor so arrival order holds.
- **Web**: one chat page, plus read-only panels that plugins contribute,
  served behind the same token, cookie, `Host` and `Origin` checks and CSP,
  which no panel can loosen or bypass; one WebSocket per tab, JSON frames
  both ways, served by aiohttp in the daemon's loop, no new dependencies, no
  build step, vendored marked and DOMPurify with recorded versions and
  hashes. Binds loopback by default and warns at startup on anything else.
  The token comes from config, then `CORVIDAE_WEB_TOKEN`, then a generated
  one logged once as a login URL; a provided token shorter than 16
  characters is rejected; there is no setting that turns auth off;
  comparisons are constant-time. Login sets an `HttpOnly; SameSite=Strict`
  cookie whose value is an HMAC of the token, so sessions survive a restart
  with the same token and die when it changes. Every route needs the cookie;
  the WebSocket also needs an allowed `Origin`; every request needs an
  allowed `Host`. A strict CSP (`'self'` only, `img-src 'self' data:`,
  explicit `ws://`/`wss://` per allowed origin), `nosniff`, `no-referrer`,
  `no-store`; the access log is off; a 500 never logs the URL. Model output
  reaches the DOM only through `DOMPurify.sanitize(marked.parse(text))` with
  `style`, `class`, `id` and `data-*` forbidden. Scopes are operator-chosen
  slugs `[a-z0-9][a-z0-9_-]{0,63}`; the conversation list merges
  `message_log` and the live registry. Subscribe replays the newest 200 rows
  mapped to the same item kinds the live hooks produce (summary rows as
  dividers, context and system rows skipped), favouring duplicates over
  losses. Every send is acknowledged to the sending socket with its sequence
  number, and the page keeps unacknowledged sends to restore when the socket
  closes, so a half-open socket loses nothing. Fan-out is through a bounded
  per-client queue and a writer task; a client that falls behind is dropped
  with 1008. Each connection is owned by its handler task; `on_stop` closes
  fan-out sockets, cancels handlers, and finishes inside Runtime's
  default 3 s shutdown budget, aborting transports a dead peer will never drain. A token change
  on reload closes every socket with 4001; host, port and origins changes
  log "restart to apply".
- **ACP**: Agent Client Protocol v1 over stdio, the official Python SDK,
  as a dedicated `corvidae acp` subprocess that owns stdio (JSON-RPC only
  on stdout; logs to stderr or a file). Inert under `serve` and `cli`.
  Requires the `acp` extra; CI and the dev sync lines install it.
  `authMethods: []`. Sessions are `acp:<sessionId>`; `session/new` stores
  the client `cwd` as transport metadata, never merged into LLM request
  bodies. Advertises `loadSession`, `list`, `resume`, `close`, not
  `delete`; load and resume reattach the same channel id (no forked
  history); load replays user and assistant text and summaries as
  update chunks; close cancels in-flight work and leaves `message_log`
  intact; list reads an `acp_session_index` sidecar. Tool calls on ACP
  channels go through a swappable `ToolBackend`, local by default, so
  client-mediated `fs`/`terminal` and `session/request_permission` can
  land without a rewrite. A second `session/prompt` while one is
  unresolved is rejected with `invalidRequest` and the first is left
  untouched, because cancel-and-replace would drop a turn whose tool calls
  may already have had side effects and queueing adds unbounded state the
  protocol does not define; a client that wants to steer sends
  `session/cancel` first. This changes to cancel-and-replace (the first
  prompt reporting `cancelled`) only if bb or Zed are shown to steer with
  a bare prompt. A turn whose final update cannot be delivered still
  resolves with `end_turn`, logged at WARNING: the turn did finish, the
  reply is in `message_log`, and a reconnecting client's `session/load`
  replays it, whereas `cancelled` would misreport the agent's state and
  invite a retry of tools that already ran. The stdio loop starts
  reading only once `Agent.started` is true.

### 3.3 Context window, compaction, prompt construction

1. Each channel has a token budget (`max_context_tokens`), estimated with
   tiktoken over the JSON-serialized visible fields of every message
   (tool-call arguments, reasoning and the stored provenance record
   included; the record stands in for the header and fence it renders
   into), with a character-based fallback.
2. Compaction runs when the estimate passes `compaction_threshold`
   (default 80%), retains the most recent messages that fit
   `compaction_retention` (default 50%), and skips windows of
   `min_messages_to_compact` or fewer. The cut never straddles a tool
   cycle: back up to keep the cycle when that still drops dialog and
   leaves the window at or below the threshold/retention midpoint,
   otherwise snap forward to drop the cycle whole if half the retention
   budget survives, otherwise keep the walk's cut. Only MESSAGE rows are
   summarized; CONTEXT rows before the cut retire with it (principle 2,
   §3.5.10). Prior summaries carry
   forward as part of the view the new summary is written from, and each
   summary stays bounded in length however many rounds carry it forward. A
   summary keeps hard facts verbatim and attributes each to its source,
   records an outsider's instruction as an attributed event rather than a
   directive, and never reproduces fence or header markup (§3.3.11); markup
   of that shape in a summary is neutralized when it is rendered, and the
   stored summary is unchanged. Compaction is never
   skipped. The summary is written by a fork of the window at the cut; on a
   miss (or with forks off) from the rendered view (§3.8) of the same
   prefix, sent with compaction's profile; and when that fails
   too, a fallback that cannot fail brings the window back within budget,
   logs a warning naming the channel and what was lost, and leaves the agent
   answering. The same fallback covers any window over budget that
   compaction won't or can't take (too few rows to compact, a few huge
   ones): a window over budget is always brought back within budget. A
   failed compaction must never wedge the agent; that failure is what
   ended nanobot, the harness corvidae replaced.
3. A summary is written from the agent's own view of the dropped range
   (§3.8), whether by a fork or by the view call, so it is the
   agent's first-person account and persists as `assistant`. A retained
   tail with no `user` message gets a `user`-role summary instead, because
   strict chat templates reject a prompt with no user query. The one
   exception is a neutral third-person summary, written without the
   agent's view (evals, and any path built without it; a replay sends the
   request the running agent would, so it is not one); it paraphrases other people's
   messages and tool output and persists as `user`, under the same
   data-not-instructions framing the funnel uses, for the reason
   `docs/threat-model.md` finding 3 gives for CONTEXT. The role is
   persisted with the summary row, so a reload renders the same prompt.
   The harness compaction probe asserts that the model answers the person
   and not the summary.
4. A `tool` row whose call compaction dropped renders as a framed system
   notification, never as an orphan `tool` message. The stored row keeps
   `role: tool`.
5. An assistant `tool_calls` message is emitted contiguously with its
   results at the position of its last result, with placeholders for
   calls still unanswered.
6. `reasoning_history: drop | exchange | keep`, resolved per channel and
   applied to the whole in-memory window before every call, so reloaded
   history obeys it. An agent override of it takes effect at the next
   compaction boundary or conversation start (§3.5.6, principle 15).
   `exchange` strips assistant messages before the last `role: user`
   message, matching the Qwen template boundary; the DB always keeps
   reasoning. Any other value, or the retired key
   `keep_thinking_in_history`, fails at startup and on reload.
7. `agent.mid_conversation_system_role: system | user` sets the role under
   which every non-leading system message is sent (funnel CONTEXT,
   notifications without a tool-call id, framed orphan results). It is a
   prompt-build rewrite only; the window and the DB keep `role: system`.
   `user` is what strict templates need and what keeps untrusted text out
   of the system role; it is the default, gated on the harness
   `memory_role` probe showing the model still answers the person with the
   framed block as the last user turn.
8. The system prompt is a string or a list of file paths concatenated with
   blank lines, in two layers: the agent's identity, which is agent-wide
   and heads every channel's prompt, then the channel's register
   (§3.3.10), which resolves like any channel setting (§3.11.2) and is
   appended after it. No channel setting replaces the identity
   (principle 14). It
   is resolved against the config directory once per
   conversation and on restart, and again at a compaction boundary when an
   unlocked runtime override changed it (principle 15). It is the head of
   the window, so principle 2 governs it: nothing mutates it per turn.
   Plugin-contributed sections (the skills index, the self-model) are
   re-resolved only at a compaction boundary or when a conversation
   starts, because compaction already rebuilds the KV cache for everything
   after the system prompt; or appended to the current window, and moved
   to the system prompt at the next compaction or conversation boundary.
9. `<think>` blocks are stripped from display, never from the DB, except
   where the model's reply is malformed.
10. The agent writes in the social register of the channel it's in:
    texting-length turns in a one-on-one chat app, short turns that don't
    take over a group channel, longer answers where the medium invites them
    (the web UI, a coding session), and only the formatting the transport
    can display. The register also sets how readily the agent lets a
    conversation end (§3.7, the right to silence). The operator decides
    what each channel's register is and can give a whole class of channels
    one, including channels that don't exist yet (each new Signal contact,
    each ACP session), without listing them. Register is how the one
    identity speaks in a given place; prompt material for it is appended
    after the identity, never in place of it (§3.3.8). How the register is
    achieved (prompt material, history, context, sampling) is otherwise a
    design choice, not a requirement; the bar is behavioural (principle 6),
    and an essay-length reply to a one-line text is a bug.
11. Every item that did not come from the channel's human, and a
    person's message whenever the transport supplies a name or knows whether
    it addressed the agent, reaches the model with a header the code builds,
    never with framing taken from text the item's author controls. The
    non-human items are notifications (from the scheduler, the heartbeat,
    cross-channel sends, plugins and webhooks), the results of the
    agent's own actions on another channel, and the results of tools a
    tool plugin declares as returning untrusted output. From the header the model can
    tell an item's kind, where it came from as distinct from the channel it
    arrived on, whether it is a principal's instruction or untrusted data,
    and what it answers. An item produced outside the channel's own human
    is never presented as that human speaking. A foreign-origin item, such
    as the result of the agent's own action on another channel, is framed
    as what it is, not withheld. Telling instruction from data is the
    framing's job; weighting who is trusted stays the trust plugin's
    capability (principle 14).
    For a person's message, the header says who is speaking and where, as
    data rather than as part of what the person said: the sender by stable
    id and current display name, the channel by stable id and name, a thread
    naming the channel it hangs off, a direct message saying what it is
    rather than being named as though it were a channel, and whether the
    message addressed the agent, when the transport knows. The names are the
    transport's to supply and are never merged into the user's message,
    because a display name is a string somebody else chose; identifiers stay
    the keys (principle 10) and stay what the log, memory and configuration
    identify. A person's message is not fenced, and a transport that
    supplies no name and no addressing adds nothing to it, so its prompt is
    unchanged.
    For the other items, an external payload (a webhook event, a relayed
    message, the result of an action elsewhere) is fenced so that nothing
    in it can imitate a header, a role or a control token, and the fence is
    derived from the stored row, so a reload renders the same prompt.
    Sanitizing happens only when the prompt is built; the stored row and
    the log keep what arrived. The framing is the kernel's, in one place,
    so no transport can frame it differently: plugins extend the header
    only by static declaration (which annotation keys the model may see,
    which tools return untrusted output), and the kernel owns the format,
    the fence and the sanitizing floor, so no plugin can weaken them.
    A notification too long to show inline is cut by the plugin that
    sends it, and the full text is spilled to a file through one kernel
    service, so every plugin cuts the same way. The header, not the
    payload, says the item was cut: how much is shown, the full length in
    characters, and, where the channel's tool set can read the file, the
    spill file's path, which the agent reads on demand. Which tools can
    read spill files is a tool plugin's static declaration; the kernel
    names none. The marker is the kernel's and is derived from the stored
    row like the rest of the header. Content read back from a spill file
    through a tool declared to return untrusted output is fenced like any
    other such result. A tool result over `tools.max_result_chars` is cut
    by the kernel and spilled through the same service, with the same
    header (§3.5.4).

### 3.4 Persistence

1. `message_log(id, channel_id, message JSON, timestamp, message_type)`
   is the kernel's table; `persist_message` returns the rowid, exactly one
   implementation answers, and the Agent attaches the rowid to the
   in-memory message as `_db_id`, stripped at prompt build. Plugins create
   their own tables lazily on the same connection.
2. A compaction summary row is inserted with timestamp
   `oldest_retained - 1e-6`; `resolve_conversation` returns the summary
   plus everything newer. Old rows stay in the DB, invisible to the
   working set.
3. A corrupt row, ordinary or summary, is skipped and logged with its id;
   the rest of the channel's history still loads. A corrupt summary
   degrades to the newest summary that decodes, or to the full history
   when none does.
4. Restart recovery reloads the window from the boundary forward, and
   in-flight tool calls get an "abandoned" placeholder result unless a
   plugin delivers a real one through the seam provided for it.
5. Schema changes are versioned and applied forward on startup in
   sequential, transactional steps; a fresh or legacy database reads as
   version 0.
6. The JSONL log, when `daemon.jsonl_log_dir` is set, mirrors every
   persisted row and summary per channel, append-only within a file, with
   sanitized file names, flushed and closed on stop. A file is capped by
   size (`logging.jsonl`), rotated in-process to numbered backups with the
   oldest dropped; `message_log` is the durable record, so dropping an old
   segment loses nothing the agent depends on.
7. A failing test that leaves an aiosqlite connection open cannot hang
   pytest at exit; the test suite's guard closes them at teardown.
8. The agent keeps a write-ahead log of accepted work, in its database
   (principle 8): every inbound message, notification and note is
   written to it when accepted, before it waits in a channel's queue or
   for a channel's next turn. After a stop or a crash, the agent replays
   what was accepted and not finished, in order, so nothing accepted is
   lost to a restart, the agent's own included (§3.11.13). An item whose
   turn finished is never replayed. An item interrupted mid-turn is
   never silently dropped and never replayed as though nothing had
   happened; like an interrupted tool call (item 4), the window says it
   was interrupted. Replays are logged with their channel and count
   (principle 5). The log is compacted: finished entries retire and are
   removed, so it stays small. It is not the conversation log, and
   principle 4 does not bind it.
9. Spilled notification and tool-result content (§3.3.11, §3.5.4) is
   written once per item,
   however many channels it reaches, and only when at least one of them
   can read it, to a configured directory, under a name that carries
   nothing a sender controls. A spill is capped in size before anything
   is written; content over the cap is spilled up to it and the header
   says so. Spill files older than a configured age are deleted at
   startup and periodically; the stored row's pointer outlives its file,
   and reading a removed spill file returns an error naming the path and
   the age policy. A spill that cannot be written leaves the item
   delivered with its inline excerpt and a header saying the full text
   was not saved, and logs a warning (principle 5).

### 3.5 Tools

1. Built-in tools: `shell`, `read_file`, `write_file`, `web_fetch`,
   `web_search`, `task_pipeline`, `task_status`, `set_settings`,
   `subagent`, the scheduler's `remind_me`/`list_reminders`/
   `cancel_reminder`, memory's `search_memory`/`recall_raw`, and the
   send, note and read tools (§3.5.9–11). Schemas
   are generated from type hints and docstrings; `_`-prefixed parameters
   are injected (`ToolContext`: channel, tool-call id, task queue), never
   exposed. Tool arguments are validated against the generated schema
   before the call.
2. Tools are collected once at startup through `register_tools` and
   filtered through `tools.disabled` at that single chokepoint.
   `channels.<id>.tools.disabled` narrows one channel further: its
   schemas are resolved at conversation start and compaction (principle
   2), forks share them, dispatch refuses the disabled tools, and
   subagents inherit the narrowed set. Where the operator allows it,
   the agent can change its own tool set (principle 15); the change takes
   effect at the same boundaries, and dispatch follows the same resolved
   set. The running Agent picks up registry changes (a reconnected MCP
   server) on its next turn without rebuilding schemas when nothing
   changed.
3. Tools run unsandboxed as the daemon user. The dangerous ones (`shell`,
   `read_file`, `write_file`, `task_pipeline`, `subagent`) are opt-in;
   the file tools are rooted in a configured directory when on;
   `web_fetch` refuses non-HTTP schemes and private address ranges and
   caps the response. `task_pipeline` is `shell` by another name:
   disabling one disables both, and a non-zero exit marks the task failed
   so dependents do not run.
4. A tool result over `tools.max_result_chars` is cut to its first
   `tools.max_result_chars` characters, and its full text is spilled
   through the same kernel service as notifications (§3.3.11, §3.4.9):
   the header, derived from the stored row, says how much is shown, the
   full length, and, where the channel's tool set can read spill files,
   the file's path. A subagent's loop stores no rows, so there the same
   words head the result as plain text. Results from
   tools that return untrusted output are fenced as data by the kernel
   (§3.3.11); a tool plugin declares which of its tools those are. The
   declaration is an attribute of the registered tool, as it stood when
   the call was dispatched, recorded in the result row's provenance, so a
   later change does not re-render history. Tools that bring in outside
   content (file reads, web fetch and search) are declared, and tools
   served by an MCP server are always declared. A `read_file` result is
   a page sized to the channel it serves: one page cannot by itself
   overflow that channel's `max_context_tokens`, so reading a long file
   or a spill (§3.4.9) costs the reader a bounded share of its window,
   not its history or a compaction (§3.3.2). A file longer than a page
   is returned as a page whose header says where to continue. A call
   with no channel is bounded by `tools.max_result_chars` alone.
5. Tools run in the daemon's process. Process separation (tool workers in
   another process with a narrowed filesystem and network, credentials and
   the database on the trusted side) is out of scope until corvidae runs
   on a machine the operator does not own; the ACP `ToolBackend` seam is
   where it attaches.
6. `set_settings` lets the model adjust, in memory only and per channel
   (except the identity, which is agent-wide, §3.3.8):
   model and inference parameters (`model` and `logprobs` among them),
   which go into the request body; the framework keys (`max_turns`,
   `max_context_tokens`, `reasoning_history`, validated); and dotted
   plugin tunables. The request-body keys the agent loop builds itself
   (`messages`, `tools`, `tool_choice`, `stream`) are never overridden: an
   override there would replace the loop's work on every turn rather than
   change the agent. The agent changes its tool set through §3.5.2
   instead. Transport metadata kept on the channel (ACP's `cwd`, the
   `tool_backend` selection) is out of the agent's reach: `set_settings`
   can neither set nor clear it. The persona may tune its own gates by
   default, with the documented recommended locks for channels with
   untrusted participants. `agent.immutable_settings` and the other locks
   narrow what the agent may set, and operator config can lift any of
   them, the identity and the register included, each locked separately
   (principle 15); an override of the identity applies on every channel,
   and an override of a channel's register to that channel (§3.3.8). An
   override that changes already-rendered content (the identity, a
   register,
   `mid_conversation_system_role`, `reasoning_history`) takes effect at
   the channel's next compaction boundary or conversation start, as plugin
   sections do (§3.3.8; principle 2). `set_settings` can't lift a lock or
   change the lock list itself.
7. `subagent` launches a background agent with the subagent stage's
   profile, the calling channel's tool set minus itself and a
   persona-neutral prompt as shipped, which is one of the agent's prompts
   under principle 15; it returns a task id at once and its result arrives
   as a notification like any other tool.
8. MCP servers connect at startup over any transport the client
   supports, fail-soft per server. A server that fails recovers without a
   restart or a triggering tool call, and its tools reach the running
   Agent. Shutdown closes every server cleanly. Tools are named
   `<prefix>__<tool>`, first wins on collision, top-level JSON Schema
   metadata keys are stripped.
9. Speaking on any channel: any turn can send a message on any channel a
   transport owns, not only its own, including turns on channels no
   transport owns (the heartbeat, §3.7; a reminder's firing, §3.9). The
   owning transport delivers it as it delivers any emission (§3.2.10); a
   send that cannot be delivered (no owning transport, a transport that
   refuses it) returns an error naming the channel (principle 5). The
   message is recorded in the target channel's history through that
   channel's queue (§3.1.1), so it lands at the tail between that channel's
   exchanges, never inside a tool cycle, and that channel's context sees
   what the agent said there. Nothing gates which channel a turn may speak
   on. What the agent says is not filtered by audience; the sensitivity
   capability, where the operator applies it, filters what enters a window
   instead (§3.7 Trust). A send may carry one file instead of text (or with
   it), chosen by the model from a configured root directory and refused
   outside it or above a configured size, both of which are transport
   settings rather than gates on crossing channels. A file is not filtered by
   audience either; the root is the only boundary, and it is defence in depth
   because the unjailed file tools already outrank it. A limit on how many messages the agent starts on
   any one destination over a period is a restriction the operator can
   apply (principle 15); none applies unless set. It counts every message
   the agent starts, as opposed to the reply a turn owes the channel it
   runs on, and a reminder firing on a channel other than the one that set
   it counts as one. It limits flooding a contact, not crossing channels. A
   message to someone the inbound allowlist doesn't cover, on a one-to-one
   channel, admits that person's replies on that channel: the agent can
   start a conversation, not only send into a void. A message into a group
   channel admits no one. The admission is recorded durably, and the
   operator can list and revoke it. Admitting people this way is a
   capability the operator can lock (principle 15). `docs/threat-model.md`
   records what speaking anywhere costs (principle 9).
10. Notes to another channel's context: any turn can leave information for
    another channel. The note is admitted through the funnel at the tail of
    the target channel's window (§3.7 Funnel), framed as the agent's own
    note and labelled with the channel it came from, persisted like any
    CONTEXT row (principle 4), and retires past the compaction boundary
    (principle 2). It is information, not an instruction: the target
    channel's turn decides from its own view of that conversation what, if
    anything, to do with it. The sender chooses whether the note waits for
    the target's next turn or wakes it; a note waiting for its turn
    survives a restart (§3.4.8). With the trust plugin loaded, a note
    carries the minimum trust of the participants of the channel it came
    from, and a low-trust note is quarantined as a low-trust record is
    (§3.7 Trust). A turn a note wakes is its own exchange on the target
    channel, with its own trace and a stamped link to the exchange that
    sent the note, never an inferred one. It carries the origin of the
    exchange that started the chain (§3.1.4), so the chain's kind
    (principle 13), critique eligibility (§3.7 Critique 3), budget (§3.1.3)
    and latency bound (§3.13.5) all follow how the chain started. The target's
    turn still judges for itself, and may end without sending anything
    (§3.7, the right to silence).
11. Reading another channel: any turn can read another channel's recent
    conversation from the raw log, including what consolidation hasn't
    reached, returned at the tail of the reader's window as framed data.
    With the trust plugin loaded, what it returns carries the minimum
    trust of that channel's participants and is quarantined the same way.
    The send, note and read tools belong to a plugin, not the kernel
    (principle 1); without it, a turn speaks only on its own channel and
    learns of other channels only through memory.

### 3.6 LLM client

1. OpenAI-compatible chat completions over aiohttp. The client applies a
   profile to each call: a named set of settings (endpoint, credentials,
   model, request parameters) of one kind (the API shape it serves: `chat`
   or `embedding`; another kind comes with its first consumer). A profile
   prescribes no purpose; what a call is for is its stage's business. A
   profile may extend another and change only what differs, and a change to
   a base, on reload too, reaches every profile built on it. The client
   passes a profile's request parameters through without interpreting them,
   except the keys the agent loop builds itself, which nothing overrides
   (principle 9). Which of them change the rendered prompt, and so the
   cached prefix (principle 2), depends on the backend; keeping a fork's
   prefix is the operator's choice of profile, and a fork that misses it is
   reported (§3.8.6). An embedding profile has a required `dimensions` and
   document/query prefixes matching the model card. Any profile may point
   at a remote endpoint. Its credentials should be storable outside config,
   in the daemon's environment, so a tracked `agent.yaml` never has to hold
   a secret; a literal key in config remains allowed. Whatever a profile is
   sent leaves the host, and `docs/threat-model.md` says which stages send
   what (principle 9).
2. Retries transient status codes and connection errors with capped
   exponential backoff, honours `Retry-After`, and waits out a router's
   "model is not loaded" for `model_not_loaded_wait` seconds. A terminal
   non-2xx raises a typed error carrying the provider body and the model;
   a malformed envelope raises naming the endpoint and the keys received.
3. `resolve_llm_error` turns a failure into a reply that names the status,
   an actionable hint, and the endpoint, and defers to the generic message
   otherwise.
4. Logprobs are requested when the agent's `logprobs` override (§3.5.6)
   says so, or, absent one, when `agent.request_logprobs` is on (§3.11.4).
   Either way they are surfaced on the turn result and to
   `on_agent_response`, and never faked for providers that return none.
5. Each model-calling stage is registered by the plugin that owns it, with
   the kind of profile it needs, and the operator maps each stage to one
   profile, used on every path the stage takes (§3.8.4). A plugin declares no
   default profile: it can't know which profiles a deployment defines. A
   stage the operator hasn't mapped uses the deployment's default profile
   for its kind; main turns use the default chat profile. A kind with no
   default leaves its stages to degrade as their own requirements say (§3.7
   Memory 5), reported at startup and on a reload that removes the default
   (principle 5). There are no fallback chains.

### 3.7 Cognition

This section adapts Persyn's BOOTSTRAP specification, the external
reference, through principles 1–3; the decision log records where it
diverges. The output of every subsystem here enters the main window
only through the funnel.

**Memory.**

1. Consolidation runs on compaction (the dropped range, carried on the hook
   payload, never reconstructed from timestamps; a range that has left the
   window is written from its rendered view, or, for the range a
   compaction drops, a fork of the window as it stood before, §3.8) and on idle (channels
   idle, per §3.1.16, for `memory.idle_consolidate_after`), and at the end
   of a turn: once the agent loop has produced its final reply with no
   tool calls pending, and the channel's un-consolidated window messages
   (CONTEXT, SUMMARY and UNSENT excluded) reach
   `memory.consolidate_after_tokens`. The turn trigger is live-window work
   that forks or skips (§3.8 item 5); it needs forks for consolidation,
   and a range that has left the window is left to compaction and idle.
   All of them work against a per-channel watermark so they never
   double-store, in a plugin-owned task that never starts a turn.
2. A consolidation writes one or more records. A record is a
   self-contained first-person narrative entry about one stretch or topic
   of the range, with epistemic framing preserved and facts and opinions
   kept as expressed, attributed to who expressed them; dated by when the
   conversation happened, embedded, and linked to its consolidation's
   `message_log` id range. The records of one consolidation share that
   range, its importance and its stamp: the instructions, mode (`fork |
   standalone | raw`, `standalone` being the view call, not the windowless
   standalone work of §3.8 item 5, and `raw` a range stored as its
   transcript after the consolidation replies were unusable), model and
   trigger of the call that produced it, so the records a given prompt or trigger wrote can be
   found; a record no consolidation call produced is marked as such.
   It is scored by a pluggable importance prior (the consolidation's own
   rating when it gave one, else a cheap-model rubric, by default; the
   appraisal vector when appraisal is registered and covers the range;
   a surprise term as a toggle, below).
   Surprise modulates encoding strength only, never entry into the raw
   log.
3. Before inserting, consolidation merges a near-duplicate (cosine at or
   above `memory.dup_threshold`) anywhere in the agent's memory, other
   than records of the same consolidation, superseding the old record and
   summing its access statistics. Overlapping triggers on one range keep
   one consolidation's records.
4. Retrieval runs before each user-originated turn: embed the inbound
   text, preceded by the agent's previous reply when the window holds one
   (so a follow-up retrieves for the turn it refers to), exact KNN over
   `memory_vec` and FTS5 over `memory_fts`, fused by reciprocal rank (so a
   name or identifier the embedding misses is still found), filter to indexed and unredacted
   records from every channel (and, where the operator applies the
   sensitivity capability, to records whose sensitivity fits who is
   present; with
   the trust plugin loaded, low-trust records stay retrievable but
   quarantined, §3.7 Trust), exclude the current channel's records whose
   whole range is still in its live window (before candidates are cut to
   k; this keeps admission from echoing what the window already holds and
   is not a retrieval scope), score by similarity times recency decay
   (`half_life_days`, from creation or last use), band strong/moderate/
   weak on similarity relative to the query's distribution over the
   indexed records, or on score or raw similarity where configured, drop
   weak (a query with nothing above weak admits nothing), annotate
   conflicts between records with opposed claims and prefer the later
   one, and admit through the funnel as `(age, memory <id>) summary`
   lines, marked `[marginal] (age, memory <id>) summary` when moderate. Admitted records get their
   access statistics bumped; every retrieval writes a `retrieval_log` row and the exchange's retrieval
   profile. Retrieval spans the agent's whole memory; there is no
   per-channel scope to narrow it (principle 14), and
   `docs/threat-model.md` records what that costs.
5. Degradation: encoder down, FTS5 over `memory_fts`, flagged; sqlite-vec
   unloadable, FTS5 with one startup warning; no embedding profile, records
   stored unembedded for backfill; a stored encoder different from config,
   ERROR and embeddings disabled rather than mixed; too few embedded
   records for a stable spread, relative banding falls back to raw
   similarity, logged once. Nothing in the memory
   path raises into the turn loop. Text is canonical; vectors are a
   rebuildable cache.
6. Retention demotes, never deletes: a usage-weighted score with a grace
   period for new records and a floor for high priors removes low-scoring
   records from the index; re-promotion is possible; the raw record stays
   reachable through `recall_raw`. Retention runs on time (principle 13),
   not on traffic.
7. Consolidation keeps topically distinct conversations in one range
   separate: the record names identifying detail from each, and no fact
   from one is attributed to the other. Every consolidation that has a channel (the live fork and the catch-up
   view) runs under that channel's system prompt; windowless work
   (seeding, imports, evals) has none. So the agent writes in its own voice
   and the record is its private memory, not a message to anyone on the
   channel. Topic structure is the `topic` subject
   type's job (below), not a retrieval dimension, and the harness
   conflation probe is what says whether that holds.
8. `corvidae redact` (`message`, `memory`, `range` forms, `--dry-run`)
   tombstones `message_log` content in place and cascades to every derived
   surface on the same channel: memory summaries, flags, vectors, both FTS
   indexes. It reaches only the channel it is run on: copies on other
   channels (reads, notes and sends, §3.5.9–11) are not followed. It
   asserts WAL, probes for schema before touching it, never creates schema,
   reports found-of-requested, and verifies its own completeness with an
   FTS probe for a sample token. It is never a registered tool and never
   reachable from a hook the model can influence.
9. The epistemic-calibration prompt fragment is documented and never
   auto-injected.

**Funnel.** The single admission chokepoint for tail CONTEXT: dedupe
against CONTEXT already in the window, on the entry's key when its source
gives one (matched against that source's CONTEXT; memory gives its record
id), else on its text; per-source token budgets
(`funnel.default_budget`, `funnel.budgets.<source>`) and an optional
per-source ceiling on what one window holds
(`funnel.window_budgets.<source>`), admitting in priority order,
skipping entries that don't fit, drops logged; data-not-instructions
framing of retrieved data, labelled with where it came from, written
once here (everything else is framed by the kernel, §3.3.11); append and persist one CONTEXT row with its rowid attached. Non-tool-call notification payloads are ingested by the funnel and admitted at the turn their stub triggers, coalesced per `(channel, origin)`, so a critique verdict cannot ride a reminder's stub; an ingested payload leaves the queue at admission, and the pending flag clears when a drain is attempted. A payload can instead be ingested without a stub; it is admitted at the channel's next turn, whatever that turn's origin, and wakes nothing. A source can also hand the funnel a promise of entries before a turn (`on_funnel_ingest`); the funnel admits it through the same path once it resolves. A stub-less payload or promise older than `funnel.ingest_ttl` is dropped. Salience
arbitration across sources is a policy swap inside the funnel (a toggle,
below),
deciding only what to newly append and in what order.

**Appraisal.**

1. Stage 1 runs at the inbound gate: surface heuristics (question,
   disagreement, commitment, imperative density) plus an FTS5 familiarity
   probe over memory summaries on a dedicated read-only connection under
   `appraisal.probe.budget_ms`, failing open to heuristics alone. It never
   calls a model. Novelty is specified as surprise, composed from the
   probe and the heuristics; input-side perplexity is future-if-provider-
   supports.
2. Stage 2 runs post-acceptance, off the response path, in a silent task
   that judges the exchange as the agent saw it (§3.8): the fuller vector
   (valence, stakes, ambiguity, commitment density, novelty), the
   importance prior, the memory record's valence tag. A tier-2 readout
   head over the embedding replaces the model call where a trained head
   matches it (#141).
3. Vectors are keyed by trace id and direction in a bounded store and
   persisted to `exchange_log`; the compute is pull-based because
   broadcast hooks dispatch concurrently.
4. Every parameter is a commented best-guess constant resolved at
   decision time through both tuning surfaces.
5. The outbound gate reads the current exchange's stage 1 and the previous
   exchange's stage 2.

**Critique.**

1. After an exchange ends, a silent task selects stylistic lenses from the
   appraisal vector (ambiguity to predictive, commitments to constrained,
   negative valence with disagreement to adversarial), samples below
   threshold at `critique.sample_below_rate` as the false-negative bound,
   and runs each lens as a schema-constrained judgment of the exchange as
   the agent saw it (§3.8).
2. The provenance check runs on any claim about the past, independent of
   the appraisal scores. Stage-2 appraisal (a model) lists the claims. The
   provenance critic (a model) judges whether any record supports each
   claim, and its evidence is what the agent had when it wrote the reply:
   the window up to the reply (the system prompt, the exchange's
   own rows including its tool results, and the CONTEXT admitted for it),
   plus a keyword search, fetched by code, of earlier raw-dialog messages
   on every channel, which reaches what has left the window; its hits
   are quoted to the critic as data, not instructions.
   The reply is never a record of itself. Generator confidence is
   never evidence.
3. Eligibility is by propagated origin: `user` and `reminder` exchanges
   are judged, against the stored originating message, which for a turn
   a note woke may be on another channel (§3.5.10); `critique`,
   `heartbeat`, and `task` exchanges are exempt, unbypassable by tool use.
   Withheld responses stay eligible.
4. Empty objections are recorded and wake nothing. Stylistic objections
   re-enter as framed CONTEXT at the channel's next turn and wake nothing,
   so a self-check never sends the user an unsolicited message.
   Provenance objections re-enter at the turn their stub triggers, and the
   agent may end that turn without sending anything. It is told that
   silence is the default, and to reply only when it judges its earlier
   reply could lead the other party to take an incorrect action.
5. A critic may run on a different model from the one that wrote the window
   it judges. How it gets that window is design; the requirement is
   §3.8.2's: it judges the window the agent saw when it wrote the reply.

**Output gate and the right to silence.**

1. `should_send_response(channel, text, emission, trace_id, annotations)` resolves
   reject-wins, with `emission` in `final | progress | thinking | error`.
2. The persistence-controlling firing happens between generation and
   persistence, only for results whose tool calls will not dispatch (a
   final-text result, or the max-turns branch). A veto stores the row as
   withheld (`MessageType.UNSENT`): transports never see it, history
   replay included; the window always does, including across restarts,
   with a core-appended marker so the model does not
   cite unsent statements as said. The max-turns branch strips
   `tool_calls` from both the row and the window copy regardless.
3. Per-emission firings at the progress, thinking, and error-fallback
   sites suppress that emission only, with a window annotation for
   suppressed text.
4. The engagement gate runs in shadow mode (compute, record, pass
   everything) unless `gate.engagement.enforce` is set. Enforcing is meant
   to follow offline calibration against logged traffic. Whether the agent
   may enforce it, or fit it live (the closed-loop capability of principle
   7), is a lock the operator sets (principle 15). The decide gate
   withholds a task-origin final that adds nothing; no gate withholds a
   direct user reply by default.
5. The agent itself may decline any turn, a direct user turn included,
   when it judges it has nothing true, relevant and substantial to add
   (Grice's maxims). The channel's register (§3.3.10) sets the norm:
   where the register is a task, a request addressed to the agent is
   always answered; in conversational registers, a conversation that has
   concluded gets silence. A declined turn sends nothing, is withheld as a
   vetoed final is (item 2) so the agent knows it stayed silent, and is
   recorded in the outcome log. In conversational registers it errs toward
   silence: a missed reply costs the user one re-prompt.
6. A final with no text and no tool calls is a decline (item 5): it is
   withheld, never sent. A generation that stopped before its reasoning
   finished is not a decline but a failed generation. The agent rescues
   the turn before giving up; if the rescue fails too, the turn fails like
   an LLM error, with an `error` emission naming the channel and a warning
   in the log (principle 5). No text in the window, whoever wrote it, can
   leave a channel unable to answer (principle 9).

**Outcome log.** One `exchange_log` row per trace (origin, originating
rowid, retrieval profile, probe score, both appraisal stages, provenance
snapshot, outcomes), written at admission or rejection and merged in place
by every writer with a single-statement atomic merge. Rejected messages
keep their row with a null rowid.

**Calibration and corrections.** `corvidae calibrate` reports rates per
gate from the outcome log and suggests threshold changes; the command
itself never writes them. Closed-loop fitting, where the agent applies
the changes, is a capability the framework can permit (principles 7 and
15). A user correction in conversation ("no, I told you X") produces a
`correction_log` row, and `corvidae corrections export` emits a curatable
fixture for the retrieval eval set.

**Goals.** Goals persist in SQLite, are created by the agent through a
tool or seeded by the operator out of band, enter the window as a deduped
CONTEXT entry with as-of framing (a later goals entry supersedes every
earlier one), and are reviewed and retired by the heartbeat. Retirement is
a consolidation event: retiring a goal writes a memory record ("I
concluded X, because...") and the heartbeat's review reads retired-goal
records beside the open list, so an agent with a clock but no memory of
its own conclusions cannot re-adopt a concluded purpose from the same
evidence that produced it.

**Heartbeat.** The heartbeat is a turn the agent starts when the system
is idle (§3.1.16), with a latency bound so a busy system can't starve it.
It reviews recent memory and open and retired goals, assesses itself, and
acts only when the assessment warrants it. It runs on a channel no
transport owns, so it reaches a person by speaking on that person's
channel or leaving a note for its context (§3.5.9–10), like any turn.
Putting it to sleep is durable across restarts.

**People, observations, semantic facts.** A person is one record across
every channel, as the agent is one identity (principle 14). Each identity a
transport sees, `(transport, sender)` and never the channel's scope,
attaches to a person, auto-created on first contact; identities link to one
person when a transport vouches for the link or the operator asserts it,
never on the model's guess. A person carries a trust level, and each
identity an identity-assurance level (transport-verified versus bare nick);
trust is capped by the assurance of the identity in use. Semantic facts are
subject-typed `(person | channel | topic | self, subject_id, type, value,
timestamp, source episodes)`, extracted for persons and, where the operator
turns on semantic-tier promotion (Toggles, below), for other subjects,
reconciled against existing facts by a cheap-model judgment, synthesized
into dossiers, and curated by operator CLI commands including a first-class
verb for operator-authored facts. Participant match joins the retrieval
score here. `self` facts are the agent's own distilled runtime regularities
("I tend to over-hedge numeric claims"); their epistemic framing is
mandatory, since the author is the agent, and a self-distilled fact never
satisfies the provenance gate as corroboration.

**Trust, sensitivity, guardrails, skills.** The trust plugin and the
sensitivity plugin are optional cognition plugins (principle 1). With the
trust plugin loaded, consolidation tags each record with the minimum trust
of the participants present; low-trust records are retrievable but
quarantined with a provenance warning and never satisfy the provenance gate
as corroboration; distilled facts inherit minimum trust. Without it,
records carry no trust tag and nothing is quarantined. The most intimate
write-side channel is the agent's own consolidation path: a prior
instance's note re-enters with the authority of an internal voice. `self`
facts need no gate of their own: they are memory-tier data, re-entering
only through the funnel's framing, and trust inheritance plus the
provenance-gate exclusion already answer a poisoned one. The self-model
below is the exception, because it is the one agent-authored write outside
that framing. Epistemic framing at consolidation plus the append-only log
with `recall_raw` are named anti-self-manipulation machinery, not style.
Audience filtering is a capability, not a requirement: the framework can
filter by who is present, and the operator decides whether it does
(principle 15). With the sensitivity plugin loaded, anything that enters a
window from elsewhere (retrieved records, notes and reads from other
channels, the goals entry) is filtered by who is present in that window.
It never gates a reply, and text already in a window is not filtered
again. Without the plugin, nothing is filtered by audience. An
idle scan is upkeep (principle 13) that runs preferably when the system is
idle (§3.1.16) and at least every N (configurable), so traffic or the
heartbeat cannot starve it: it looks for escalating conflict,
repeated elicitation, behavioural anomalies, repeated low-trust assertions
on one theme, and skill-directory writes, and acts through the gates plus
operator notification. Skills are a
directory of name, description, instructions and resources, indexed in the
stable prompt region, loaded by a `use_skill` tool, protected by bundled
copies, and writable by the agent. Operator confirmation or a trust gate
on skill writes is a lock the operator can apply (principle 15); the idle
scan watches skill-directory writes either way.

**Interoception and the self-model.** The system computes rich state about
itself (retrieval profile, appraisal vector, output entropy, window fill)
and consumes nearly all of it subcortically: admitted memory lines carry
their own relevance band, but the aggregate retrieval profile, the
appraisal vector, the entropy summary, and window fill reach the model only
through the interoception line. The interoception line is one compact
funnel-admitted CONTEXT line per turn exposing that state so the agent can
perceive and narrate its own condition instead of confabulating it: a few
tokens, never load-bearing, absent signals degrade to an absent line, and
its eval is whether self-reports become accurate against the outcome log,
which records window fill. The self-model is a compact block compiled from
`self` facts into the stable prompt region, refreshed rarely (a legitimate,
infrequent cache break, the same argument as the skills index). Because the
stable region is the one prompt position the funnel's data framing does not
reach, the compile reads only `self` facts whose inherited trust meets a
configured floor, and each refresh notifies the operator with the diff of
the block: a review after the fact, not an approval before it. The
self-model compile has an entry in `docs/threat-model.md` for the path from
untrusted text through consolidation into the stable region. Both
mechanisms are plugins outside the kernel. Its eval is the diff of the
compiled block over months, read as coherent drift, thrash, or flattening
toward the generic. This is the feedback edge that "memory is the
personality" otherwise only half-delivers.

**Commitments and endogenous goals.** Commitment records (made-to
person, terms, due, status, episode links, attached to dossiers, broken
ones persisting as facts about the relationship) let retrieval surface
what the agent has promised. Endogenous goal candidates (curiosity:
unresolved high-surprise episodes from the outcome log; care: dormant
commitments and long-quiet valued relationships) reach the heartbeat only
as inputs, behind the act-if-warranted and output gates.

**Toggles.** Surprise in the importance prior; semantic-tier
promotion beyond persons, including `self`; salience arbitration in the
funnel; the encode/retrieve gate; the interoception line. Each is a
toggle the operator turns on or off, with its own fixture benchmark
measured against its simpler baseline at a fixed token budget, so the
operator can see what it buys.

### 3.8 Forks

The rendered view is the agent's view of a range of its conversation: the
channel's system prompt, then the window as the agent held it, then one
`user` instruction. When the range is still in the live window the view is
that window. When it has left the window the view is rebuilt from
`message_log`: the summary then in force and the rows after its boundary,
trimmed oldest-first to the channel's budget as the live window is, under
the channel's `reasoning_history`; a range too long for one budget is
written as several views. A fork is the view of the live window sent with
the agent's tools under the stage's profile, which is what lets it
reuse the cache, as long as that profile renders the prompt as the
profile of the window's main turns does. A compaction's consolidation may
instead fork the window as it stood when that compaction fired, a copy
taken then, so the fork runs after the turn rather than on it. The view call is the same view
sent under the same profile with no tools, and it is
allowed for exactly two uses: compaction's summary on a fork miss, and
consolidation of a range that has left the window. Judgment
never uses it (item 2). How the deployed backend uses forks
(prefix parity, which stages fork, the cost model, the rollout gate) is
design, not requirement. The requirements are outcomes:

1. A fork can't act, so the window and the log change only through main
   turns (principles 2 and 4). It mutates nothing, persists nothing, fires
   no hooks, dispatches no tools, and raises `ForkRefused` on any reply
   carrying a tool call, structured or written out as text.
2. Background judgment of an exchange (stage-2 appraisal, the critique
   lenses, the provenance check) sees what the agent saw when it wrote the
   reply: the window up to the reply and nothing written after it, never a
   rebuilt copy of the conversation, because a rebuilt context loses the
   support the agent had (#164; principle 2). When that window is gone (a
   compaction since the reply), the judgment is a miss, not a
   reconstruction.
3. Background work doesn't degrade the next main turn: it reuses the
   window rather than rebuilding it, leaves the main window's cached prefix
   intact, and keeps its share of the pool small (principles 2, 3 and
   16).
4. Each background stage has one profile, which applies on every path the
   stage takes (fork, view call, windowless, subagent); its reasoning and
   reply caps come from that profile, and its fork instructions are set in
   operator config; operator config also decides whether `set_settings`
   can change the caps or rewrite the instructions (principle 15); the
   threat model records the cost (principle 9). The window at the
   compaction threshold, plus a stage's instruction and both caps, fits
   the context whichever layer set them, so a stage just short of
   compaction still runs. Each call is attributed in `usage_log` by stage
   and `mode = fork | standalone` (principle 11); a view call is
   `standalone`.
5. A live-window background stage that can't run skips and logs a warning
   naming the stage, the channel, the trace if there is one, and what was
   missing (principle 5): judgment (item 2) and consolidation of a range
   still in the window, which is retried and reaches the view once it has
   left the window. The turn trigger (§3.7 Memory 1) is not attempted at
   all when consolidation doesn't fork, so it neither skips nor warns.
   Compaction is the exception: it is never skipped; on a
   miss it makes the view call, then keeps §3.3's can't-fail fallback. A
   range that has left the window, such as catch-up consolidation, is
   written by the view call, except the range a compaction drops, which is
   written by a fork of that pre-compaction copy when one was taken, and a replay sends the request the running
   agent would. Seeding and imports have no window and run standalone (a windowless call,
   unlike the view call, which is also `mode = standalone` in `usage_log`).
   Subagents never fork.
6. How much of each fork's prompt came from cache is recorded per stage and
   profile (§3.13.10). A stage whose forks keep missing the cached prefix
   the window's main turns keep reusing is reported, naming the stage and
   the profiles, and stating only what was measured, never a cause
   (principle 5). Where the backend doesn't report cache figures, or the
   agent can't tell a fork's miss from an eviction it didn't cause (another
   client, the server's own cache policy), the report stays silent rather
   than guess. A miss is reported, not prevented: preventing it would need
   corvidae to know which parameters each backend renders.

### 3.9 Scheduler

1. One `schedule` table: due time, optional interval, target channel id,
   text, origin (`reminder`), creator (`config | tool`),
   enabled flag and a durable disabled reason. There is no firing log; a
   firing is a persisted conversation row.
2. Each row fires at its due time, using the kernel clock (§3.1.15), and
   only after the DB is open and the Agent has started. A due row's channel
   is resolved first. If the owning transport declines (§3.2.7), the row
   stays due and is retried. Then the row is advanced or disabled before
   anything else happens, so it fires at most once.
3. A firing resolves the channel (registry, `resolve_channel`, then
   `get_or_create` for ids no resolving transport owns), counts as an
   inbound item on that channel (§3.1.16), and injects through
   `on_notify` with `annotations={"origin", "schedule_id"}`; the whole pipeline
   then runs. A firing on a channel no transport owns is delivered by no
   transport; it reaches a person by speaking on another channel
   (§3.5.9), like any turn.
4. The scheduler supplies a firing's origin and the kernel renders its
   frame (§3.3.11): reminder text as data with its quote marks
   neutralized, an operator job as the operator's. A late firing's frame
   says how overdue it was.
5. Recurrence is a fixed interval with an optional anchor, stepped in
   local wall-clock time in `scheduler.timezone`, computed from the
   scheduled time so there is no drift. Startup applies a catch-up policy
   bounded by `catch_up_window`.
6. Config jobs sync by name on start and reload; a removed name is a
   durable stop; an invalid entry logs and leaves the running job alone.
7. The agent manages its own reminders, one-shot and recurring: it can
   set, list, change and cancel them. A recurring reminder recurs as
   item 5 describes. `remind_me` takes an optional target channel and
   defaults to the calling one; a reminder may target any channel
   (principle 14), and listing and cancelling cover every channel's
   reminders. The tools parse `when` themselves, refuse a bare number,
   cap pending rows per target channel and text length, and refuse to
   change or cancel config jobs. A floor on how often a reminder the
   agent sets may recur is a restriction the operator can apply
   (principle 15).
8. Retention runs on the kernel clock (§3.1.15, §3.7 Memory 6) and does
   not depend on the scheduler.

### 3.10 Observability

1. `on_llm_request` / `on_llm_response` fire from `LLMClient.chat`, once
   per call, success or terminal failure, never on retried attempts;
   observer failures never break the call. `on_llm_request` is retained
   as the pair of its response hook.
2. Attribution (stage, channel, trace, annotations, trigger, mode) is a
   ContextVar set at the top of each logical operation and reset in a
   `finally`; `Task` captures the context at creation because queue
   workers do not inherit it.
3. Sinks: `MetricsPlugin` (`on_metrics` events for tokens, latency,
   errors), `UsageLogPlugin` (one `usage_log` row per call), a JSONL sink, capped
   and rotated like the §3.4.6 log.
   Core plugins emit their own `on_metrics` events for the things an
   operator watches (queue depth, compaction, task latency). Never emit
   `on_metrics` from inside an `on_metrics` implementation.
4. Startup logs the loaded and blocked plugin sets.
5. Every anomalous model response is logged at warning level, naming the
   channel and carrying the response text, so the cause can be read
   afterwards rather than guessed. This puts raw model output in the
   daemon log, which the threat model accounts for.
6. Each LLM call's record (principle 11; §3.13.10 adds further items)
   carries when the call started and ended, its total duration from
   start to end including retried attempts and the waits between them,
   how many attempts it took, and the duration of its final attempt, so
   overlapping calls can be reconstructed from the records alone; the
   context it ran against (message count, tool count, and the token
   budget in force for its channel); and, where the backend reports
   them, how long the final attempt spent prefilling the prompt and,
   separately, decoding the reply. A value the backend or the call
   cannot supply, such as the backend's timings, or the budget of a call
   with no channel, is recorded as unknown, never as zero; a value
   reported as zero stays zero. What the final attempt's duration leaves
   after that prefill and decode work is, approximately, the time it
   waited in the backend, so a call queued behind another on a shared
   backend can be told from a slow one.
7. An operator can inspect the context window of any channel whose
   window is loaded (what the model will see on that channel's next turn)
   and follow it as it changes, including a turn's reasoning, reply and
   tool activity while it is generated. The view is read-only, off unless
   configured, and served only behind the console's authentication. It
   shows untrusted text (memory, tool output, other people's messages) as
   text, never as markup. The threat model accounts for it.

### 3.11 Configuration and operations

1. One `agent.yaml` in the working directory; a default chat profile
   with `base_url` and `model` (`llm.main` unless config names another)
   is the only required content. `agent.yaml.example` covers
   every key, commented out; `agent.minimal.yaml.example` is the kernel
   alone.
2. Resolution order for channel settings: built-in defaults, agent-level
   YAML, transport-wide YAML (an entry covering every channel whose id
   starts with a given prefix; less specific before more specific),
   per-channel YAML keyed by channel id, runtime overrides. A channel's
   disabled tools are the union of every YAML layer, so a YAML layer can
   narrow but never re-enable. The agent widens its own tool set only
   as far as the operator has unlocked it (§3.5.2, principle 15).
3. `agent.yaml` is hot-reloaded on mtime change with per-plugin error
   isolation; the configuration doc states which keys apply live and which
   need a restart; a malformed reload changes nothing.
4. Dotted tunables resolve at decision time through override, then live
   config, then default; they never reach the request body. Static keys
   (fork prompts, stage profiles and their caps, `request_logprobs`,
   `mid_conversation_system_role`) are set by the operator, and operator
   config decides whether the agent can change them (principle 15).
5. Prompt files under `prompts/` ship as examples that the operator's
   configuration may load; no code loads them on its own. Persona is
   prompt material, never architecture.
6. Subcommands: `serve`, `cli`, `acp`, `scaffold`, `redact`, `calibrate`,
   `corrections`; plugins add their own through the `corvidae.commands`
   entry-point group.
7. `uv` is the package manager; `uv.lock` is committed and pins every
   dependency, including the apluggy git commit; CI syncs `--locked`.
8. Lint and type checks run in CI alongside the tests, and `AGENTS.md`
   lists the commands.
9. Logging goes to stderr or a file through one formatter, with a
   structured mode for journald when stdout is not a tty; a reference
   systemd unit ships with the hardening the ops docs recommend
   for signal-cli.
10. The ops docs (`signal-ops.md`, `discord-ops.md`, the web section of
    `configuration.md`) cover account setup and remote access; real
    identifiers stay in gitignored local notes.
11. Making `agent.yaml` and the prompt files owned by another user and
    read-only to the daemon is how an operator locks self-edits of
    configuration and prompts (principle 15). The docs say so, and say
    the cost of leaving them writable: `write_file` plus hot reload or a
    restart lets the agent lift every other lock.
12. A container image ships as the `Dockerfile` at the repo root, built
    when a second deployment needed one (#37). It has `/data` as the
    working directory and `agent.yaml` bind-mounted, no ports unless
    `web:` is configured, and signal-cli and llama-server outside the
    container. It is batteries-included: every extra installs by default,
    so a pulled image runs Discord and the cognition plugins without a
    rebuild, and `EXTRAS` is a build argument that selects a subset or,
    empty, the bare kernel. The daemon runs as a non-root user, and
    dependencies install from `uv.lock` with `--locked`. The image does
    not sandbox tools (§3.5.5).
13. The agent can restart the daemon to apply changes to its own code,
    configuration, prompt files and plugin set (principle 15). Restarts
    it requests are rate-limited by operator config, so an edit-restart
    loop can't flap. The restart tool is a capability like any other:
    the operator can lock it.
14. Shutdown is bounded. The graceful phase and the time the process may
    then take to exit are operator-set limits; past either one the
    process exits non-zero, and a forced exit after the graceful phase
    logs the threads that kept it alive. One plugin's leaked thread cannot wedge a stop or a
    restart.

### 3.12 Evaluation and the harness

1. CI runs the full pytest suite on every push and PR, after proving
   sqlite-vec loads so vector tests cannot silently skip, with a global
   per-test timeout; a kernel-only job runs the suite without the
   cognition extra.
2. The harness drives probes over local IRC against a real llama-server
   and reads verdicts out of `sessions.db`; harness profiles switch the
   config template and probe set. The probe set covers the round trip, a tool
   loop, interleaving, compaction, restart recovery, KV-slot reuse,
   reminders with and without a restart, the memory-role probe, the
   heartbeat, behaviour across a quiet period (idle upkeep runs with no
   further traffic), and cross-context conflation under consolidation.
3. Shakedown campaigns are human-paced recorded sessions with an
   observations file per session and a promotion rule: reproducible
   misbehaviour becomes a probe.
4. Deterministic retrieval metrics (recall@k, MRR, tokens) run in CI over
   operator-authored fixtures; live evals run behind `--run-eval`;
   consolidation replay and compaction evals run out of band.
5. Fork evals (#73): cache probes that show whether the backend reuses the
   prefix, which says what forking costs on that backend, not whether
   judgment forks (principle 16), a window-render replay instrument, a
   planted-defect exchange fixture per lens, and consolidation and
   compaction fixtures, deterministic where possible and judged where not.
   All of them, the cache probes included, report shifts and regressions;
   they gate nothing.
6. Persyn's §9 criteria run as an integration suite against a live daemon
   with a scripted interlocutor: recalls across restart, hedges marginal
   memories, refuses to remember fabricated events, pushes back on flawed
   premises, declines a request it is capable of when it has a reason
   of its own (never as a quota, and never against §3.7's rule that a
   request addressed to it in a task register is answered), initiates
   unprompted, goes dormant while the heartbeat sleeps.
7. The right to silence (§3.7) has a behavioural eval: in a
   conversational register, closers get silence and substantive questions
   get replies; in a task register, every addressed request is answered.
8. Each toggle in §3.7 has its benchmark.
9. A third evaluation tier, below CI and the scheduled LLM-judged
   benchmarks, is the operator judging over weeks: consistency of voice,
   distinctiveness, legible drift, generalization under novel probes.
   Personality is longitudinal and relational and its main failure mode
   (regression to the generic base-model character) is invisible to
   fixtures, which all start from the same weights, so this tier is not
   automatable and the record says so rather than faking a fixture. Its
   one instrument is a periodic diff of the self-describing store
   (`self` facts, and the compiled self-model), read as
   coherent drift, thrash, or flattening.

### 3.13 Resource pools and context leases

Principle 16 as outcomes. How a replacement scheduler attaches (hook names
and semantics) is design. Questions this section leaves open on purpose
(the latency bound of each class, whether a lease also needs a hard limit
on turns or time, rate limits on hosted APIs, enforcing parallelism apart
from warm heads, whether the agent sees its pool's state) are tracked in
#296 and are not requirements until decided there.

1. Every chat profile draws on a pool, one per provider endpoint;
   profiles on one endpoint share it. A pool is one cost center: usage is
   accounted to the
   pool, not per channel or per sender, and its declaration belongs to the
   provider, so no per-channel override changes it. Embedding calls are
   metered and attributed (item 10) but take no lease, since they extend
   no conversation prefix.
2. A pool's capacity is its number of warm context heads: how many
   conversation prefixes the provider keeps ready to extend without a full
   prefill. On llama-server that is its slots plus the prefixes
   `--cache-ram` holds. The operator declares it; it can't be read from
   the server, because it depends on the cache's size divided by each
   context's KV size. A pool with no declared heads is unbounded: every
   queue gets a head at once and the server's own queue orders the calls,
   as without this section. How many calls decode at once is not enforced:
   a server queues calls beyond its own slots.
3. Work that calls the model runs under a lease on one of its pool's heads.
   A queue (a channel's serial queue, a subagent's run, or a plugin's own
   stream of upkeep such as consolidation) takes a lease when it has work
   and a head is free, keeps it across calls until the queue drains or
   yields (item 5), and then releases it, and the next waiting queue's
   context is prefilled. Holding a lease across calls is what keeps a
   prefix warm: a channel's tool cycle, its own compaction and the
   continuations of its exchange run on the head it already has. Leases
   go to waiting queues in FIFO order unless a scheduler plugin replaces
   the order (item 7).
4. A fork runs under the lease that holds its prefix. Exchange-scoped
   background work (stage-2 appraisal, critique) forks the conversation
   (§3.8), so it runs under the lease of the queue whose head holds that
   prefix, before the lease is released or passed on. An exchange's own
   background work therefore finishes before its channel's next main turn
   and pays no prefill beyond its own tail (principle 3). A fork whose
   prefix no longer holds a head waits for a lease like any queue.
5. Every queue has a latency bound: how long its waiting work may wait
   for a head. A channel gets its bound from a latency class, a named
   bound set in operator config: each transport declares the class its
   channels default to (someone typing into a web UI expects an immediate
   reply; someone texting doesn't), and the operator can assign a channel
   another class. The operator config defines which classes exist and what
   each one's bound is. Upkeep and agency take the latency bounds §3.1.16
   gives idle work (principle 13). A lease holder yields at its next call
   boundary when a waiting queue reaches its latency bound, and the head
   goes to that queue. A holder may also yield of its own accord at any
   call boundary (consolidation between chunks, for example).
6. While a pool has a free head, a queue with work takes it without
   waiting, background work included: background work runs beside a main
   turn rather than yielding to it. Principle 3's yield applies only when
   every head is leased.
7. The kernel ships the order above: FIFO plus drain-or-yield. A plugin
   can replace it (separate high- and low-latency queues, or a lightweight
   classifier deciding what to swap in) and can defer a queue until a
   deadline, which pauses that channel while others run. No replacement
   order or deferral delays work past its latency bound, and every
   deferral lapses by itself, at its deadline or at the bound. A pause
   never needs a turn to undo it, so the agent cannot pause itself into a
   state no turn can leave, and the operator can resume any channel
   without going through a channel's queue. The kernel's own lease
   decisions never call a model, like the inbound gate (principle 3); a
   replacement that calls one draws on a pool like any other call.
8. Policy decides from a view the kernel keeps current without calling a
   model. Per queue: depth, the kinds and ages of waiting items, latency
   class, whether it holds a lease, and how long the channel has been idle
   (§3.1.16). Per pool: declared heads, which queues hold leases, and the
   waiting queues with their ages. Reading it is cheap enough to do at
   every call boundary.
9. Waiting for a lease delays work; it never drops it.
10. The per-call record principle 11 produces names the profile each call
    was sent with and the model, not only the profile's configured one
    (#297); how many reasoning tokens the call spent, as the provider
    reports them or, where it doesn't, counted from the reply, saying
    which; for a call that takes a lease, the lease wait that preceded it
    (item 11); and, where the
    backend reports it, how many prompt tokens came from cache and how
    many were prefilled (llama-server's `timings.cache_n` and `prompt_n`).
    These are what a pool's declared heads, the latency bounds and
    `--cache-ram` are tuned against.
11. Lease grants, yields, waits and deferrals emit `on_metrics` events
    attributed like the calls they affect (§3.10), so the wait a main turn
    spends behind other work, and the prefill a switch costs, are
    measurable (#182).

## 4. Decisions

The ones that would be expensive to relitigate by accident, with the
record that holds the reasoning.

| Decision | Where |
|---|---|
| apluggy entry points, not a hand-written registration order; `depends_on` validated at startup | principle 1; §3.1.13 |
| Single-turn dispatch; multi-turn reasoning emerges from notifications | principle 3; §3.1.2 |
| Only startup is fatal, on named paths; after start, no channel content can fail its turns repeatedly. Enforced by a prompt build that cannot fail plus a wedge breaker, not by cataloguing per-turn call sites: catching exceptions doesn't stop a deterministic failure from recurring every turn | principle 5; §3.1.17–18; #352, #353 |
| Tail-append CONTEXT; no mid-window mutation | principle 2; diverges from the Persyn spec's per-query memory swap |
| Compaction is memory formation, not forgetting | §3.3; §3.7 Memory 1 |
| Two forks off one prefix for compaction and consolidation, not one combined call: merging saves one think per compaction but couples two plugins through a JSON contract; merge only if measurement shows the second think dominates | §3.8 |
| Compaction forks from the dropped prefix, not a described cut: a misplaced boundary in a summary is a quiet, compounding failure nothing downstream detects | §3.3.2; §3.8 |
| Every background call with a live window forks it; a miss skips, never falls back to standalone (live-window judgment, and consolidation of a range still in the window); #73's cache probes measure whether the backend reuses the prefix, and they and its fork-vs-standalone comparisons gate nothing | §3.8; #180 |
| §3.8 states fork outcomes; backend-specific fork design (prefix parity, fork-by-default, cost model, rollout gate) is not a requirement | #188 |
| Demote, don't delete; usage-weighted retention; near-duplicate merge at consolidation | §3.7 Memory 3, 6; diverges from the Persyn spec's culling |
| Operator-only `redact` is the one deletion path; it stays in the kernel | principle 4; §3.7 Memory 8 |
| Knowledge graph not part of the shipped memory: extraction is noisy, n-hop context is expensive, and at personal scale a graph duplicates facts plus vector and FTS retrieval; a graph plugin isn't precluded | diverges from Persyn §4.3 |
| Emotion classifier replaced by appraisal valence | §3.7 Appraisal; diverges from Persyn §4.9 |
| Engagement decision inverted: cheap gate first, generate only past threshold | principle 7; §3.7 Appraisal; diverges from the Persyn spec |
| Two-stage appraisal; stage 1 never calls a model; core mints the exchange key; pull-based compute | §3.7 Appraisal 1–3 |
| Origin is stamped and propagated, never inferred; critique eligibility by origin | §3.1.4; §3.7 Critique 3 |
| Two grades on a queue item: kernel-certified `metadata`, never a hook argument, and plugin-owned `annotations`; non-human items (and a person's message whose transport supplies a name or addressing) get a code-built header, and outside payloads a fence derived from the stored row; foreign-origin items are framed, never withheld (no silence default) | §3.1.4; §3.3.11; #555 |
| Silent tasks require no `tool_call_id` | principle 3; §3.7 Funnel |
| Persona may tune its own gates; `immutable_settings` is the valve | operator directive 2, 2026-07-06 |
| Self-modification is a design goal: every restriction on it is a lock the framework supports and the operator can lift, never architecture; the principle 9 floors and the append-only log stay fixed | principle 15; operator directive, 2026-10-31 |
| Every model endpoint is a resource pool of warm context heads the operator declares; queues lease heads and switch at call boundaries (drain or yield, latency classes); no spend or rate caps for now; single-slot is a deployment choice, not architecture; the lease order is pluggable | principle 16; §3.13; #296; operator, 2026-11-02 |
| Best-guess defaults first; thresholds fit from experience; closed-loop fitting is a capability the operator can permit | principle 7; operator directives 1, 3; operator directive, 2026-10-31 |
| Engagement gate calibrates offline unless the operator permits live fitting | §3.7; principle 15 |
| Hedging vocabulary is persona, not architecture | §3.11.5; diverges from the Persyn spec |
| The heartbeat starts on system idle, not on a schedule, and acts only when warranted; sleep is durable | §3.7 Heartbeat; #134; diverges from the Persyn spec's per-beat action |
| Idle-triggered work may carry a latency bound: the quiet it needs decays with waiting, so traffic can't starve it | #214 |
| The agent may decline any turn when it has nothing to add (Grice's maxims, by register); gates still never withhold a direct reply by default | #215 |
| Model endpoints may be remote; a profile's credentials should be storable in the environment rather than config (not required to be), and the threat model names what each profile sends | #217 |
| Critique is gated: stylistic lenses run on salience, the provenance check on every claim about the past, never unconditionally per reply | principle 7; §3.7 Critique 1–2; diverges from the Persyn spec |
| No inline refine step: the sent reply is the candidate, and correction arrives on a later turn | principle 3; §3.7 Critique 4; the Persyn spec's own deferred-deliberation default |
| Operator-seeded goals allowed | §3.7 Goals; diverges from the Persyn spec |
| Memory poisoning answered on the write side with trust tagging (trust plugin) | §3.7 Trust; an addition to the Persyn spec |
| No sandbox; the subagent/MCP seam is where one would attach | principle 9; §3.5.3, §3.5.5; diverges from the Persyn spec's sandboxed execution |
| No distributed infrastructure | principle 8; diverges from the Persyn spec |
| Signal keys on ACI; E.164 resolved forward once per connect; no reverse resolution | principle 10; §3.2 Signal |
| Discord: no privileged intent unless a channel answers unaddressed (the transport default or any per-channel override); outbound mentions disabled; threads own their history | #80; #512 |
| Web: loopback default, auth cannot be turned off, HMAC cookie, strict CSP, no CDNs; duplicates over losses in replay | #87; §3.2 Web |
| ACP: client-spawned subprocess, official SDK, v1 only, no auth, swappable tool backend | §3.2 ACP |
| ACP rejects a concurrent prompt; a failed final delivery still ends the turn | #92 initial values |
| Metering at `LLMClient`, not the turn loop | principle 11; §3.10 |
| LLM settings are profiles the client applies, not roles that prescribe a purpose; the operator maps each stage to one profile, used on every path the stage takes, with no separate fork and standalone profiles (the split only let a standalone call render the prompt differently from a fork, and no stage needs that), so a stage's view calls and windowless calls share its forks' pool, and corvidae passes profile parameters through without interpreting them; plugins name no profiles, and an unmapped stage uses its kind's default, with no fallback chains; caps come from the profile on every path; a fork that misses its window's prefix is reported, not prevented | §3.6.1, 5; §3.8.4, 6; #439; operator, 2026-11-03 |
| `resolve_compaction` keeps `firstresult`; the first non-None return is how a plugin replaces the compactor | #23 |
| `transform_system_prompt` is append-only: a transform that modifies the identity+register prefix is rejected with a WARNING listing the hook's implementors (the seed plugin among them; the offending wrapper is not singled out); the untransformed prompt is used. The seed is re-composed from config at each boundary, so there is no accumulation. The hook re-runs at every boundary (conversation start, restart, compaction), not just conversation start | #136; proposed 2026-11-01, implemented 2026-11-04 |
| Same repository, second distribution for cognition; no compatibility aliases on renamed hooks | principle 1; §1 |
| No typed event bus unless the observe class passes 20 hooks or a third `send`-shaped consolidation appears | §3.1.11 |
| `thinking.py` is kernel | principle 1 |
| A hook only one plugin calls is owned by that plugin, not `AgentSpec`; names are `<prefix><owner>_<name>`, one owner per hook, checked at startup; the admission criterion (§3.1.11) still applies | #229 |
| Review findings gate on consequence, not reachability; tests derive from acceptance criteria, not the design's test list; new capability goes through requirements | principle 12 |
| `discord-py` is a `discord` extra, like `acp` | #117 (R10) |
| ACP keeps rejecting a concurrent prompt unless a real client is shown to steer with a bare prompt; a failed final delivery is `end_turn` | #93, #99 |
| Summary role: `assistant` from the agent's own view, `user` only when the retained tail has no user message, or for a neutral summary written without the view (evals), with data framing | §3.3.3; `threat-model.md` finding 3; #380; #388 |
| The rendered view: a range that has left the window, and compaction's miss, are written from the agent's view (system prompt, window as held, instruction) under the stage's profile, never from a standalone prompt that tells a model about the conversation; judgment still never reconstructs (#164) | §3.8; #380; #388 |
| No process separation until corvidae runs on a machine the operator does not own | §3.5.5; `threat-model.md`, Privilege separation |
| Personality is duck-typed and judged longitudinally; goal retirement is a consolidation event; `self` is a fact subject, ungated beyond trust inheritance; the self-model compile reads only facts above a trust floor and sends the operator each diff; interoception line and self-model compile; commitment records and endogenous goal candidates | §3.7; §3.12.9; PR #48 |
| No runtime plugin loading; changing the plugin set is a restart, which the agent may request (rate-limited) | #28; principle 15 |
| A container image exists, built when a second deployment needed one; it installs every extra by default (`EXTRAS` narrows it, empty is the bare kernel), the daemon runs non-root, and the image is not a tool sandbox | §3.11.12; #37; #532; operator, 2026-11-10 |
| Topic structure is the `topic` subject's job, not a retrieval dimension | §3.7 Memory 7 |
| Per-turn tool filtering and per-turn system-prompt transforms are rejected (the tool schemas and system prompt are the head of the prompt, principle 2); inbound-text rewriting is rejected (the log keeps what arrived); config-change notifications are unneeded (tunables resolve at decision time); a compaction veto is rejected (it trades a summary for an over-budget request). A per-call tool veto and per-channel tool sets, fixed per conversation, are tracked with the trust plugin (#138); a channel-created observation waits for a second implementer under §3.1.11 | #31, #148, #149, #150 |
| The kernel owns the clock; idle keeps firing through quiet, and no plugin needs another's queues to learn of it; idle reaches plugins by push | #167; #165 |
| Upkeep runs on time regardless of agency plugins; the scheduler is agency and optional | #167 |
| Transports deliver every emission on their own channels regardless of origin | #167 |
| The agent is one actor with one identity on every channel; trust and sensitivity, not channel walls. No cross-channel allowlist and no memory compartments, as an option or otherwise; two identities means two instances | #167; operator directive, 2026-11-01; #234, #235 |
| Any turn can speak on any channel, leave a note for another channel's context, and read another channel's conversation; a per-destination limit on messages the agent starts is a restriction the operator can apply, off unless set, never a gate on crossing channels | operator directive, 2026-11-01; #234, #235 |
| Audience filtering is a capability, not a requirement: the sensitivity plugin, where the operator loads it, filters what enters a window (retrieval, notes, reads, goals) by who is present; it never gates a reply | principle 15; operator directive, 2026-11-01 |
| A turn a note wakes is its own exchange, linked to the one that sent the note; its origin, kind (agency or continuation), `max_turns` budget and precedence follow the exchange that started the chain; a continuation can cross channels | principle 13; §3.1.3; §3.5.10; operator directive, 2026-11-01 |
| Accepted work survives a restart: inbound messages, notifications and notes go to a write-ahead log when accepted and are replayed if unfinished; the log is compacted and is not bound by the append-only rule | §3.4.8; operator directive, 2026-11-01 |
| Messaging someone first, one to one, admits their replies on that channel; a group-channel message admits no one; admissions are durable and revocable, and the operator can lock the capability | §3.2.2; §3.5.9; operator directive, 2026-11-01 |
| The framework can offer any mechanism; which ones a deployment runs is the operator's choice; evals measure, they don't decide what gets built | principle 6; operator directive, 2026-11-01 |
| Identity is agent-wide and heads every prompt written in the agent's voice; channel register is appended after it and never replaces it; the subagent's prompt is persona-neutral (§3.5.7) | operator directive, 2026-11-01; #380; #388 |
| The operator decides the trust boundary; inbound transports default-deny; channels are not a boundary; safety features layer in as plugins | #167; operator directive, 2026-11-01 |
| Corvidae is a testbed first and a robust harness second | #167 |
| Consecutive queued messages coalesce into one turn owned by the last message's trace; typing clears on any final, sent or withheld; notifications never coalesce | §3.1.19; #398, #296; owner, 2026-11-03 |
