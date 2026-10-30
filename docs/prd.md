# Corvidae: product requirements

What corvidae is for, what it must do, and what it has decided not to do.
This document defines requirements: what must be true, and why. It is not
a record of what the tree does now, and it never says what is or isn't
built. When the tree and this document disagree, either the tree is wrong
and a GitHub issue tracks the gap, or the requirement is wrong and the fix
is an edit to this document, reviewed like code. `docs/design.md` says how
the tree does these things; `plans/` says how a change gets built; this
says what has to be true.

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
  primary transport, running unattended for weeks. The strict-template,
  single-slot, and Signal requirements below come from this one.
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
   declares its dependencies in `depends_on`, and documents what the system
   does without it. The cognition plugins (memory, retention, funnel,
   appraisal, critique, outcome log, memory tools, trust, sensitivity) are
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
   message so the cached prefix stays stable. Prefill is the expensive
   part of a call on any transformer, so context is reused, not rebuilt:
   background calls that need the conversation fork it (§3.8) rather than
   rebuilding it.

3. **Background cognition avoids blocking the main turn.** The point is
   interactivity. The agent loop is single-turn dispatch: one LLM call per
   queue item, tool calls as tasks, results re-entering as notifications.
   Appraisal, critique, consolidation, and retention run opportunistically
   as silent tasks or plugin-owned tasks, and shape the *next* turn.
   Background work runs in the gaps between main turns. Deferral is bounded
   so interactivity can't starve it: an exchange's own background work
   (stage-2 appraisal, critique) finishes before the next main turn on the
   slot begins, so its results are in place for the turn that reads them.
   Other queued background work (upkeep) yields to a waiting main turn:
   beyond any exchange-scoped work still owed and past-due upkeep, the main
   turn waits at most for the call already running. Upkeep that is past due
   runs next, ahead of a waiting main turn, so traffic can't starve it
   (principle 13). On one KV slot a background call can't run beside the
   main turn, so the requirement is to keep its slot time short (bounded
   output, §3.8) and out of the way; it is not a guarantee of
   zero delay. The inbound gate runs inside the transport read path and
   never calls a model.

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
   silent; a plugin with a malformed one is loud and aborts startup.

6. **Behavioural tests are the bar; mechanisms are hypotheses.** A
   cognitive mechanism is built where an eval demands it, not because a
   doctrine describes it. Deterministic metrics against operator-authored
   fixtures run in CI; LLM-judged suites run out of band and are tracked as
   benchmarks, never as gates. The pretrained weights are frozen and vast,
   so mechanisms about the agent's own runtime stream (this user, these
   channels, this deployment) are preferred over mechanisms that
   reconstruct general knowledge the weights already hold. Expect the
   honest eval to retire scaffolding.

7. **Cheap, always-on perception gates expensive cognition.** One appraisal
   signal has three consumers: engagement, deliberation depth,
   consolidation strength. Epistemic checks stay independent of the
   appraisal scores: the provenance check fires on any claim about the past
   and is judged against records, never on generator confidence. Thresholds
   ship as commented best guesses, are tuned from the outcome log, and are
   never auto-fitted in a closed loop.

8. **Local-first: one process, one machine, one SQLite file.** No message
   broker, no worker pool, no workflow engine. The database runs in WAL
   mode as a requirement, because a second process (the redact CLI) and a
   second connection (the appraisal probe) read it concurrently. This is
   about the agent's process and state; model endpoints are wherever the
   operator points them (§3.6).

9. **The operator is trusted; everyone else is text.** Prompt injection is
   the threat, not other tenants. Anything that arrived from a channel, a
   web page, an MCP server, or a memory distilled from those is data, and
   is framed as such at one chokepoint. Transports are default-deny with
   silent refusal. The operator decides the trust boundary, not the
   architecture: defaults are open on what the agent's memory reaches and
   what may cross between channels through memory, inbound transports stay
   default-deny, and the cross-channel send allowlist starts empty. The
   architecture supplies the mechanisms (transport allowlists, the
   cross-channel send allowlist and budget, memory compartments, the trust
   and sensitivity plugins); the operator configures where the boundary
   falls, and safety features are added as plugins over time, as with
   sandboxing. Fixed floors protect the operator's own control surface
   rather than draw a trust boundary, and stay fixed, among them: web auth
   cannot be turned off (§3.2 Web), `system_prompt` is never model-settable
   (§3.5.6), `redact` is never a tool (§3.7 Memory 8), and static keys are
   operator-only (§3.11.4). The model's authority is unbounded and
   unsandboxed by decision; `docs/threat-model.md` is the standing record
   of what that costs and is updated whenever untrusted text can reach
   somewhere new or a tool can do something new.

10. **Channel ids are persistence keys and never derive from names.**
    `transport:scope` keys `sessions.db`, the JSONL logs, memory
    compartments, and the schedule. Scopes use identifiers that cannot be
    renamed or go missing: Signal's ACI, Discord's numeric ids, a web slug
    the operator chose, an ACP session id reused across load and resume.

11. **Every LLM call is metered at the one chokepoint it passes through.**
    `LLMClient.chat` fires the observability hooks, so compaction,
    subagents, consolidation, appraisal, critique and forks are all
    attributed by stage, channel, trace, trigger and mode. Eval questions
    are denominated in tokens; this is an eval prerequisite.

12. **How the work is done.** Red/green TDD; every test under a timeout;
    docs current before a change is complete. Plans are the design of
    record, written to be executed without architectural judgment calls,
    with requirements tagged by provenance and gated by the owner. Plans
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
    not; stylistic critique objections start no turn at all. Agency is
    optional. Without agency plugins the agent never initiates a turn and
    has no reminders or heartbeat, and its upkeep still runs. No plugin's
    documented behaviour may depend on traffic it does not control.

14. **One actor, many channels.** Memory and identity are the agent's, not
    the channel's. Safety comes from trust (what came in is weighted and
    labelled by its source) and sensitivity (what goes out is filtered by
    who is present), not from walling channels off. Compartments
    (`memory.channel_groups`) are an operator option. Principle 1 applies:
    without the sensitivity plugin, retrieval is not filtered by audience;
    without the trust plugin, nothing is quarantined; retrieval spans the
    whole memory unless the operator draws a boundary with compartments
    (principle 9). The whole-memory default has an entry in
    `docs/threat-model.md` (principle 9), since it lets text distilled on
    one channel reach prompts on every other.

## 3. Requirements

### 3.1 Kernel: agent loop, queues, traces

1. One serial queue per channel. The LLM is never called with two items
   from one channel in flight. Queues are visible to other plugins by
   reference.
2. Single-turn dispatch. Each tool call becomes a `Task` on the shared
   task queue; the next LLM call waits until the last result of a batch
   clears the channel's pending set. User messages may interleave
   mid-batch without disturbing it. The task queue has configurable
   concurrency and a bounded completed-result buffer.
3. `max_turns` is a per-exchange budget. The counter resets on any item
   that originates a new trace while no tool cycle is in flight, so a
   channel fed only by scheduler notifications never exhausts it. An item
   arriving mid-cycle is charged to the running cycle.
4. Every inbound item carries a trace id. Transports mint it from a
   stdlib-only module and pass it to `on_message`; the Agent mints at
   dequeue for a notification that inherits none. The id and an opaque
   `meta` dict ride every hook, every `Task`, every completion
   notification, and every usage-log row. A falsy id is a `ValueError`;
   omitting it is a hook-call error. Core never stamps meaning into `meta`;
   `meta["origin"]` is a plugin-side convention.
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
   in `tools.timeouts`, skipped when the tool's own timeout is shorter),
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
    enforces the prefixes against pluggy options.
11. A new hookspec is admitted only when it is a decision point, an
    observation with at least two plausible implementers, or the pair of
    an existing observability hook. A single-consumer observation gets a
    `meta` entry or a return value on an existing hook instead.
12. Hook parameters with defaults can be dropped silently by pluggy; a
    startup guard rejects implementations that re-declare a spec-required
    parameter with a default.
13. Plugin dependencies are declared in `depends_on` and validated before
    `on_start`; plugins resolve each other with `get_dependency`, never by
    registration order. `on_start` is broadcast concurrently; the Agent
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

### 3.2 Transports

The contract every transport meets:

1. No config block: inert, nothing logged above DEBUG. Present but
   malformed: raise in `on_init` naming the field.
2. Default-deny sender allowlist in `should_process_message`. A refusal is
   one INFO line naming the sender; nothing is sent back and no liveness
   indicator is shown. The transport fires `on_message` for every decoded
   message regardless; the gate is the one place that vetoes. Web
   substitutes a bearer token for the allowlist; the token is the
   authorization.
3. Transports subclass `TransportPlugin`, set `transport`, and override the
   `send_*` methods they can display. The one `send(channel, event)` hook
   does the ownership check once; a transport's `owns(channel)` predicate
   is overridable (Web also checks that it is enabled). Reasoning and tool
   status default to no-ops on chat transports.
4. Outbound text is split, never truncated, by UTF-8 byte length; the
   chunks concatenate back to the original text; the limit is at least
   six bytes or `split_message` raises. Discord's splitter also keeps code
   fences balanced across chunks.
5. Liveness (typing, read receipts) is shown only to allowed senders. A
   slot taken for a message is released when that message's reply goes
   out, keyed by trace id so a notification-born reply cannot release a
   slot another message is owed, and in a `finally` if the message was
   never admitted. One refresh loop per channel; `on_stop` cancels it.
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
  history and whatever the agent's memory admits to that channel, which by
  default is all of it.
- **Signal**: newline-delimited JSON-RPC to an already-running `signal-cli`
  daemon over a unix socket; corvidae never spawns or supervises it.
  Direct messages only. Scope is the sender's ACI. Config may name
  senders by E.164 in `allow:` and `channels:`; those resolve forward to
  ACIs in one batched call after each connect, never at startup, and a
  failed resolution retries after the next connect. Non-`dataMessage`
  envelopes (including the account's own sync traffic) are dropped;
  attachments are logged and discarded; text accompanying them is
  processed; backlog older than five minutes is prefixed with its send
  time. Typing starts on the first outstanding message and refreshes
  every 10 s; the read receipt is sent at processing start, not arrival.
  Notifications are never handled inline in the read loop. Groups
  and disappearing-message handling are out of scope.
- **Discord**: discord.py's gateway behind the `discord` extra, default
  intents, no privileged Message Content intent. Replies in DMs and when mentioned (not
  `@everyone`, `@here`, or roles); drops bot authors and empty text;
  strips the mention token. Scopes: `discord:<channel_id>`,
  `discord:<thread_id>` (a thread has its own history), `discord:dm:<user_id>`.
  Every send passes `AllowedMentions.none()`. A bad token is logged once
  at ERROR and not retried. Gateway events are queued and consumed by one
  processor so arrival order holds. Voice, slash commands, embeds,
  attachments, reactions: out of scope.
- **Web**: one page, one WebSocket per tab, JSON frames both ways, served
  by aiohttp in the daemon's loop, no new dependencies, no build step,
  vendored marked and DOMPurify with recorded versions and hashes. Binds
  loopback by default and warns at startup on anything else. The token
  comes from config, then `CORVIDAE_WEB_TOKEN`, then a generated one
  logged once as a login URL; a provided token shorter than 16 characters
  is rejected; there is no setting that turns auth off; comparisons are
  constant-time. Login sets an `HttpOnly; SameSite=Strict` cookie whose
  value is an HMAC of the token, so sessions survive a restart with the
  same token and die when it changes. Every route needs the cookie; the
  WebSocket also needs an allowed `Origin`; every request needs an
  allowed `Host`. A strict CSP (`'self'` only, `img-src 'self' data:`,
  explicit `ws://`/`wss://` per allowed origin), `nosniff`,
  `no-referrer`, `no-store`; the access log is off; a 500 never logs the
  URL. Model output reaches the DOM only through
  `DOMPurify.sanitize(marked.parse(text))` with `style`, `class`, `id`
  and `data-*` forbidden. Scopes are operator-chosen slugs
  `[a-z0-9][a-z0-9_-]{0,63}`; the conversation list merges `message_log`
  and the live registry. Subscribe replays the newest 200 rows mapped to
  the same item kinds the live hooks produce (summary rows as dividers,
  context and system rows skipped), favouring duplicates over losses.
  Every send is acknowledged to the sending socket with its sequence
  number, and the page keeps unacknowledged sends to restore when the
  socket closes, so a half-open socket loses nothing. Fan-out is through
  a bounded per-client queue and a writer task; a client that falls
  behind is dropped with 1008. Each connection is owned by its handler
  task; `on_stop` closes fan-out sockets, cancels handlers, and finishes
  inside Runtime's 3 s budget, aborting transports a dead peer will never
  drain. A token change on reload closes every socket with 4001; host,
  port and origins changes log "restart to apply". Token streaming,
  paging past 200 rows, an inspector pane, and views of other transports'
  channels are out of scope for v1 and each gets its own decision.
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
  reading only once `Agent.started` is true. Deferred, each with a named
  trigger: client-mediated tools (M-CLIENT-TOOLS), a remote listener on
  the daemon (M-REMOTE), registry auth (M-REGISTRY), v2 (M-V2).

### 3.3 Context window, compaction, prompt construction

1. Each channel has a token budget (`max_context_tokens`), estimated with
   tiktoken over the JSON-serialized visible fields of every message
   (tool-call arguments and reasoning included), with a character-based
   fallback.
2. Compaction runs when the estimate passes `compaction_threshold`
   (default 80%), retains the most recent messages that fit
   `compaction_retention` (default 50%), and skips windows of
   `min_messages_to_compact` or fewer. The cut never straddles a tool
   cycle: back up to keep the cycle when that still drops dialog and
   leaves the window at or below the threshold/retention midpoint,
   otherwise snap forward to drop the cycle whole if half the retention
   budget survives, otherwise keep the walk's cut. Only MESSAGE rows are
   summarized; SUMMARY and CONTEXT rows survive. Prior summaries carry
   forward, bounded to a fraction of the budget. Compaction is never
   skipped: when its summary can't be written (a missed or failed fork), a
   fallback that cannot fail brings the window back within budget, logs a
   warning naming the channel and what was lost, and leaves the agent
   answering. The same fallback covers any window over budget that
   compaction won't or can't take (too few rows to compact, a few huge
   ones): a window over budget is always brought back within budget. A
   failed compaction must never wedge the agent; that failure is what
   ended nanobot, the harness corvidae replaced.
3. The summary's role follows how it was written. A forked summary is
   the agent's own first-person account and persists as `assistant`. A
   standalone summary (one written without the window, as in replays and
   evals) is a neutral paraphrase of other people's messages
   and tool output and persists as `user`, under the same
   data-not-instructions framing the funnel uses, for the reason
   `docs/threat-model.md` finding 3 gives for CONTEXT. Either way a
   retained tail with no `user` message gets a `user`-role summary,
   because strict chat templates reject a prompt with no user query. The
   role is persisted with the summary row, so a reload renders the same
   prompt. The harness compaction probe asserts that the model answers the
   person and not the summary.
4. A `tool` row whose call compaction dropped renders as a framed system
   notification, never as an orphan `tool` message. The stored row keeps
   `role: tool`.
5. An assistant `tool_calls` message is emitted contiguously with its
   results at the position of its last result, with placeholders for
   calls still unanswered.
6. `reasoning_history: drop | exchange | keep`, resolved per channel and
   applied to the whole in-memory window before every call, so reloaded
   history obeys it. `exchange` strips assistant messages before the last
   `role: user` message, matching the Qwen template boundary; the DB
   always keeps reasoning. Any other value, or the retired key
   `keep_thinking_in_history`, fails at startup and on reload.
7. `agent.mid_conversation_system_role: system | user` sets the role under
   which every non-leading system message is sent (funnel CONTEXT,
   notifications without a tool-call id, framed orphan results). It is a
   prompt-build rewrite only; the window and the DB keep `role: system`.
   `user` is what strict templates need and what keeps untrusted text out
   of the system role; it is the default, gated on the harness
   `memory_role` probe showing the model still answers the person with the
   framed block as the last user turn.
8. The system prompt is a string or a list of file paths concatenated
   with blank lines, resolved against the config directory once per
   conversation. It is the head of the window, so principle 2 governs it:
   nothing mutates it per turn. Plugin-contributed sections (the skills
   index, the self-model) are re-resolved only at a compaction boundary or
   when a conversation starts, because compaction already rebuilds the KV
   cache for everything after the system prompt; or appended to the
   current window, and moved to the system prompt at the next compaction
   or conversation boundary.
9. `<think>` blocks are stripped from display, never from the DB.
10. Double-buffer compaction (build the compacted prompt in a second KV
    slot and swap) is out of scope until it has its own decision.
11. The agent writes in the social register of the channel it's in:
    texting-length turns in a one-on-one chat app, short turns that don't
    take over a group channel, longer answers where the medium invites them
    (the web UI, a coding session), and only the formatting the transport
    can display. The register also sets how readily the agent lets a
    conversation end (§3.7, the right to silence). The operator decides
    what each channel's register is and can give a whole class of channels
    one, including channels that don't exist yet (each new Signal contact,
    each ACP session), without listing them. How the register is achieved
    (prompt material, history, context, sampling) is a design choice, not a
    requirement; the bar is behavioural (principle 6), and an essay-length
    reply to a one-line text is a bug.

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
   degrades to the full history.
4. Restart recovery reloads the window from the boundary forward, and
   in-flight tool calls get an "abandoned" placeholder result unless a
   plugin delivers a real one through the seam provided for it.
5. Schema changes are versioned and applied forward on startup in
   sequential, transactional steps; a fresh or legacy database reads as
   version 0.
6. The JSONL log, when `daemon.jsonl_log_dir` is set, mirrors every
   persisted row and summary per channel, append-only, with sanitized
   file names, flushed and closed on stop.
7. A failing test that leaves an aiosqlite connection open cannot hang
   pytest at exit; the test suite's guard closes them at teardown.

### 3.5 Tools

1. Built-in tools: `shell`, `read_file`, `write_file`, `web_fetch`,
   `web_search`, `task_pipeline`, `task_status`, `set_settings`,
   `subagent`, the scheduler's `remind_me`/`list_reminders`/
   `cancel_reminder`, and memory's `search_memory`/`recall_raw`. Schemas
   are generated from type hints and docstrings; `_`-prefixed parameters
   are injected (`ToolContext`: channel, tool-call id, task queue), never
   exposed. Tool arguments are validated against the generated schema
   before the call.
2. Tools are collected once at startup through `register_tools` and
   filtered through `tools.disabled` at that single chokepoint.
   `channels.<id>.tools.disabled` narrows one channel further: its
   schemas are resolved at conversation start and compaction (principle
   2), forks share them, dispatch refuses the disabled tools, and
   subagents inherit the narrowed set. The
   running Agent picks up registry changes (a reconnected MCP server) on
   its next turn without rebuilding schemas when nothing changed.
3. Tools run unsandboxed as the daemon user. The dangerous ones (`shell`,
   `read_file`, `write_file`, `task_pipeline`, `subagent`) are opt-in;
   the file tools are rooted in a configured directory when on;
   `web_fetch` refuses non-HTTP schemes and private address ranges and
   caps the response. `task_pipeline` is `shell` by another name:
   disabling one disables both, and a non-zero exit marks the task failed
   so dependents do not run.
4. Tool results are truncated to `tools.max_result_chars`. Results from
   MCP servers are framed as data by the default `transform_tool_result`,
   the way the funnel frames CONTEXT.
5. Tools run in the daemon's process. Process separation (tool workers in
   another process with a narrowed filesystem and network, credentials and
   the database on the trusted side) is out of scope until corvidae runs
   on a machine the operator does not own; the ACP `ToolBackend` seam is
   where it attaches.
6. `set_settings` lets the model adjust, per channel and in memory only:
   inference keys from an allowlist (`temperature`, `top_p`, `top_k`, the
   penalties, `max_tokens`, `seed`), the framework keys (`max_turns`,
   `max_context_tokens`, `reasoning_history`, validated), and dotted plugin
   tunables. Nothing else reaches the request body. `system_prompt` is
   always blocked; `agent.immutable_settings` blocks more. The persona may
   tune its own gates by default, with the documented recommended
   blocklist for channels with untrusted participants.
7. `subagent` launches a background agent on the shared `background`
   client with the calling channel's tool set minus itself and a persona-neutral
   prompt; it returns a task id at once and its result arrives as a
   notification like any other tool.
8. MCP servers connect at startup over stdio or SSE, each inside its own
   exit stack with a bounded handshake, fail-soft per server; servers that
   failed at startup are retried on a bounded backoff loop without
   needing a tool call to trigger it; a reconnected server's tools reach
   the running Agent; sessions are owned by long-lived tasks so entering
   and exiting happen in the same task and shutdown terminates stdio
   subprocesses cleanly. Tools are named `<prefix>__<tool>`, first wins
   on collision, top-level JSON Schema metadata keys are stripped.
9. Cross-channel sends: the agent can send to a channel other than the
   current turn's own, only to destinations on an operator allowlist,
   which starts empty, and within a send budget. When the trust and
   sensitivity plugins are loaded, it sends only content whose trust and
   sensitivity fit the destination's audience (§3.7 Trust); without them,
   the allowlist and the budget are the whole gate. It has an entry in
   `docs/threat-model.md` before it ships (principle 9).

### 3.6 LLM client

1. OpenAI-compatible chat completions over aiohttp. Roles `main`,
   `background`, `embedding`; any unconfigured role falls back to `main`;
   consuming plugins implement their own two-step fallbacks (appraisal,
   critic) on their windowless paths. The embedding role has its own
   endpoint, a required `dimensions`, and document/query prefixes
   matching the model card. Any role may point at a remote endpoint. Its
   credentials come from the environment, never from config. Whatever a
   role is sent leaves the host, and `docs/threat-model.md` says which
   stages send what (principle 9).
2. Retries transient status codes and connection errors with capped
   exponential backoff, honours `Retry-After`, and waits out a router's
   "model is not loaded" for `model_not_loaded_wait` seconds. A terminal
   non-2xx raises a typed error carrying the provider body and the model;
   a malformed envelope raises naming the endpoint and the keys received.
3. `resolve_llm_error` turns a failure into a reply that names the status,
   an actionable hint, and the endpoint, and defers to the generic message
   otherwise.
4. Logprobs are requested when `agent.request_logprobs` is on (operator
   only), surfaced on the turn result and to `on_agent_response`, and
   never faked for providers that return none.
5. No streaming until a core change is planned for it; the web transport
   states the reason.

### 3.7 Cognition

The design of record is `plans/bootstrap-mapping.md`, which translates
Persyn's BOOTSTRAP architecture through principles 1–3 and records its
divergences. The output of every subsystem here enters the main window
only through the funnel.

**Memory.**

1. Consolidation runs on compaction (the dropped range, carried on the hook
   payload, never reconstructed from timestamps) and on idle (channels
   idle, per §3.1.16, for `memory.idle_consolidate_after`), against a
   per-channel watermark so the two never double-store, in a plugin-owned
   task that never starts a turn.
2. A record is a first-person summary with epistemic framing preserved,
   dated by when the conversation happened, embedded, linked to its
   `message_log` id range, and scored by a pluggable importance prior
   (cheap-model rubric by default; the appraisal vector when appraisal is
   registered; a surprise term as a Phase 6 toggle). Surprise modulates
   encoding strength only, never entry into the raw log.
3. Before inserting, consolidation merges a near-duplicate (cosine at or
   above `memory.dup_threshold`) within the channel's retrieval scope,
   only with a record of equal reach, superseding the old record and
   summing its access statistics. Overlapping triggers on one range keep
   one record.
4. Retrieval runs before each user-originated turn: embed the inbound
   text, exact KNN over `memory_vec`, filter to indexed and unredacted
   records in the channel's retrieval scope (and, with the sensitivity
   plugin loaded, to records whose sensitivity fits who is present; with
   the trust plugin loaded, low-trust records stay retrievable but
   quarantined, §3.7 Trust), score by similarity times recency decay
   (`half_life_days`, from creation or last use), band strong/moderate/
   weak on score or raw similarity, drop weak unless nothing else matched,
   annotate conflicts between records with opposed claims and prefer the
   later one, and admit through the funnel as `[band] (age) summary`
   lines. Admitted records get their access statistics bumped; every
   retrieval writes a `retrieval_log` row and the exchange's retrieval
   profile. A channel's *retrieval scope* is the agent's whole memory
   unless the operator narrows it with compartments
   (`memory.channel_groups`, whose members may be globs); the whole-memory
   default has an entry in `docs/threat-model.md` (principle 14).
5. Degradation: encoder down, FTS5 over `memory_fts`, flagged; sqlite-vec
   unloadable, FTS5 with one startup warning; no embedding role, records
   stored unembedded for backfill; a stored encoder different from config,
   ERROR and embeddings disabled rather than mixed. Nothing in the memory
   path raises into the turn loop. Text is canonical; vectors are a
   rebuildable cache.
6. Retention demotes, never deletes: a usage-weighted score with a grace
   period for new records and a floor for high priors removes low-scoring
   records from the index; re-promotion is possible; the raw record stays
   reachable through `recall_raw`. Retention runs on time (principle 13),
   not on traffic.
7. Consolidation keeps topically distinct conversations in one range
   separate: the record names identifying detail from each, and no fact
   from one is attributed to the other. The standalone consolidation
   prompt (seeding, imports, replays and evals, which have no window to
   fork) carries its own calibration language, since the channel's system
   prompt does not reach it. Topic structure is the `topic` subject
   type's job (below), not a retrieval dimension, and the harness
   conflation probe is what says whether that holds.
8. `corvidae redact` (`message`, `memory`, `range` forms, `--dry-run`)
   tombstones `message_log` content in place and cascades to every
   derived surface on the same channel: memory summaries, flags, vectors,
   both FTS indexes. It asserts WAL, probes for schema before touching it,
   never creates schema, reports found-of-requested, and verifies its own
   completeness with an FTS probe for a sample token. It is never a
   registered tool and never reachable from a hook the model can influence.
9. The epistemic-calibration prompt fragment is documented and never
   auto-injected.

**Funnel.** The single admission chokepoint for tail CONTEXT: dedupe
against CONTEXT already in the window; per-source token budgets
(`funnel.default_budget`, `funnel.budgets.<source>`), greedy, drops
logged; data-not-instructions framing with a source label written once
here; append and persist one CONTEXT row with its rowid attached. Non-
tool-call notification payloads register with the funnel and are admitted
at the turn their stub triggers, coalesced per `(channel, origin)`, so a
critique verdict cannot ride a reminder's stub; payloads unregister at
admission, and the pending flag clears when a drain is attempted. A payload
can instead register without a stub; it is admitted at the channel's next
turn, whatever that turn's origin, and wakes nothing. Salience
arbitration across sources is a policy swap inside the funnel (Phase 6),
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
   head over the embedding replaces the model call only once trained
   (Phase 5; #141).
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
   within the channel's retrieval scope, which reaches what has left the
   window; its hits are quoted to the critic as data, not instructions.
   The reply is never a record of itself. Generator confidence is
   never evidence.
3. Eligibility is by propagated origin: `user` and `reminder` exchanges
   are judged, against the stored originating message; `critique`,
   `heartbeat`, and `task` exchanges are exempt, unbypassable by tool use.
   Withheld responses stay eligible.
4. Empty objections are recorded and wake nothing. Stylistic objections
   re-enter as framed CONTEXT at the channel's next turn and wake nothing,
   so a self-check never sends the user an unsolicited message.
   Provenance objections re-enter at the turn their stub triggers, and the
   agent may end that turn without sending anything. It is told that
   silence is the default, and to reply only when it judges its earlier
   reply could lead the other party to take an incorrect action. Where two models are deployed (a critic on a
   different model from `llm.main`), each keeps its own append-only
   window, and the critic forks the window of the one that did not
   generate.

**Output gate and the right to silence.**

1. `should_send_response(channel, text, emission, trace_id, meta)` resolves
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
   everything) unless `gate.engagement.enforce` is set; enforcing is the
   operator's choice, made after calibrating offline against logged
   traffic, never by live exploration. The decide gate withholds a
   task-origin final that adds nothing; no gate withholds a direct user
   reply by default.
5. The agent itself may decline any turn, a direct user turn included,
   when it judges it has nothing true, relevant and substantial to add
   (Grice's maxims). The channel's register (§3.3.11) sets the norm:
   where the register is a task, a request addressed to the agent is
   always answered; in conversational registers, a conversation that has
   concluded gets silence. A declined turn sends nothing, is withheld as a
   vetoed final is (item 2) so the agent knows it stayed silent, and is
   recorded in the outcome log. In conversational registers it errs toward
   silence: a missed reply costs the user one re-prompt.

**Outcome log.** One `exchange_log` row per trace (origin, originating
rowid, retrieval profile, probe score, both appraisal stages, provenance
snapshot, outcomes), written at admission or rejection and merged in place
by every writer with a single-statement atomic merge. Rejected messages
keep their row with a null rowid.

**Calibration and corrections.** `corvidae calibrate` reports rates per
gate from the outcome log and suggests threshold changes; it never writes
them. A user correction in conversation ("no, I told you X") produces a
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
transport owns, so it reaches a person only through a gated cross-channel
send (§3.5.9). Putting it to sleep is durable across restarts.

**People, observations, semantic facts.** `person` records keyed by
`(transport, scope, sender)`, auto-created on first contact, carrying a
trust level and an identity-assurance level (transport-verified versus
bare nick), with trust capped by assurance. Semantic facts are
subject-typed `(person | channel | topic | self, subject_id, type, value,
timestamp, source episodes)`, extracted for persons first and for other
subjects only when the eval demands it, reconciled against existing facts
by a cheap-model judgment, synthesized into dossiers, curated by operator
CLI commands including a first-class verb for operator-authored facts, and
restricted to agent-specific, post-training regularities. Participant
match joins the retrieval score here. `self` facts are the agent's own
distilled runtime regularities ("I tend to over-hedge numeric claims"),
the purest case of the frozen-weights criterion; their epistemic framing
is mandatory, since the author is the agent, and a self-distilled fact
never satisfies the provenance gate as corroboration.

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
With the sensitivity plugin loaded, a sensitivity column filters retrieval
by who is present; without it, retrieval is not filtered by audience. An
idle scan is upkeep (principle 13) that runs preferably when the system is
idle (§3.1.16) and at least every N (configurable), so traffic or the
heartbeat cannot starve it: it looks for escalating conflict,
repeated elicitation, behavioural anomalies, repeated low-trust assertions
on one theme, and skill-directory writes, and acts through the gates plus
operator notification. Skills are a
directory of name, description, instructions and resources, indexed in the
stable prompt region, loaded by a `use_skill` tool, protected by bundled
copies, and writable by the agent only behind operator confirmation or a
trust gate.

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
self-model compile has an entry in `docs/threat-model.md`, before it ships,
for the path from untrusted text through consolidation into the stable
region. Both mechanisms are plugins outside the kernel. Its eval is the
diff of the compiled block over months, read as coherent drift, thrash, or
flattening toward the generic. This is the feedback edge that "memory is
the personality" otherwise only half-delivers.

**Hypothesis tier.** Two mechanisms are built only on a named failure of
the longitudinal test (§3.12): commitment records (made-to person, terms,
due, status, episode links, attached to dossiers, broken ones persisting
as facts about the relationship), triggered by dropped commitments that
retrieval should have surfaced; and endogenous goal candidates (curiosity:
unresolved high-surprise episodes from the outcome log; care: dormant
commitments and long-quiet valued relationships), delivered only as inputs
to the heartbeat behind the act-if-warranted and
output gates, triggered by a month of heartbeats in which the agent never
initiates anything the operator judges worth initiating.

**Phase 6 toggles.** Surprise in the importance prior; semantic-tier
promotion beyond persons, including `self`; salience arbitration in the
funnel; the encode/retrieve gate; the interoception line. Each ships as a
toggle with its fixture benchmark authored first, is measured against its
simpler baseline at a fixed token budget, and graduates to a default only
if it wins. The encode/retrieve gate is expected to lose.

### 3.8 Forks

A fork is one LLM call whose prompt is a prefix of the channel's rendered
window plus one `user` instruction. How the deployed backend uses forks
(prefix parity, which stages fork, the cost model, the rollout gate) is
design: `docs/design.md`, Forks. The requirements are outcomes:

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
   intact, and keeps its slot time short (principles 2 and 3).
4. Each background stage's reasoning and reply are capped, per stage, by
   operator config applied after runtime overrides, and its instructions
   are static config, so `set_settings` can neither lift the caps nor
   rewrite the instructions (principle 9). The window at the compaction
   threshold, plus a stage's instruction and both caps, fits the context,
   so a stage just short of compaction still runs. Each call is
   attributed in `usage_log` by stage and `mode = fork | standalone`
   (principle 11).
5. A background stage that can't run skips and logs a warning naming the
   stage, the channel, the trace if there is one, and what was missing
   (principle 5). Compaction is the exception: it is never skipped and
   keeps §3.3's can't-fail fallback. Seeding, imports, replays and evals
   have no window and run standalone. Subagents never fork.

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
   `on_notify` with `meta={"origin", "schedule_id"}`; the whole pipeline
   then runs. A firing on a channel no transport owns is delivered by no
   transport; it reaches a person only through a cross-channel send
   (§3.5.9).
4. Reminder text is framed as data with its quote marks neutralized;
   operator jobs are framed as the operator's; a late firing is prefixed
   with how overdue it was.
5. Recurrence is a fixed interval with an optional anchor, stepped in
   local wall-clock time in `scheduler.timezone`, computed from the
   scheduled time so there is no drift. Startup applies a catch-up policy
   bounded by `catch_up_window`.
6. Config jobs sync by name on start and reload; a removed name is a
   durable stop; an invalid entry logs and leaves the running job alone.
7. Tools act on the calling channel; a reminder for any other channel is a
   cross-channel send and passes its gate (§3.5.9). They parse `when`
   themselves, refuse a bare number, cap pending rows per target channel
   and text length, and refuse to cancel config jobs.
8. Cross-channel sends are specified in §3.5.9. Recurring reminders
   by tool need a per-channel budget first. Cron syntax waits for an
   operator to need it. Retention runs on the kernel clock (§3.1.15,
   §3.7 Memory 6) and does not depend on the scheduler.

### 3.10 Observability

1. `on_llm_request` / `on_llm_response` fire from `LLMClient.chat`, once
   per call, success or terminal failure, never on retried attempts;
   observer failures never break the call. `on_llm_request` is retained
   as the pair of its response hook.
2. Attribution (stage, channel, trace, meta, trigger, mode) is a
   ContextVar set at the top of each logical operation and reset in a
   `finally`; `Task` captures the context at creation because queue
   workers do not inherit it.
3. Sinks: `MetricsPlugin` (`on_metrics` events for tokens, latency,
   errors), `UsageLogPlugin` (one `usage_log` row per call), a JSONL sink.
   Core plugins emit their own `on_metrics` events for the things an
   operator watches (queue depth, compaction, task latency). Never emit
   `on_metrics` from inside an `on_metrics` implementation.
4. Startup logs the loaded and blocked plugin sets.

### 3.11 Configuration and operations

1. One `agent.yaml` in the working directory; `llm.main` with `base_url`
   and `model` is the only required content. `agent.yaml.example` covers
   every key, commented out; `agent.minimal.yaml.example` is the kernel
   alone.
2. Resolution order for channel settings: built-in defaults, agent-level
   YAML, transport-wide YAML (an entry covering every channel whose id
   starts with a given prefix; less specific before more specific),
   per-channel YAML keyed by channel id, runtime overrides. A channel's disabled tools are the
   union of every layer, so a layer can narrow but never re-enable.
3. `agent.yaml` is hot-reloaded on mtime change with per-plugin error
   isolation; the configuration doc states which keys apply live and which
   need a restart; a malformed reload changes nothing.
4. Dotted tunables resolve at decision time through override, then live
   config, then default; they never reach the request body. Static keys
   (fork prompts, fork output caps, `request_logprobs`,
   `mid_conversation_system_role`) are operator-only.
5. Prompts under `prompts/` are documentation copies cited by path; no
   code loads them. Persona is prompt material, never architecture.
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
11. `agent.yaml` and the prompt files should be owned by another user and
    read-only to the daemon, and the docs say so, because `write_file`
    plus hot reload otherwise undoes every other control.
12. No container image until a second deployment needs one. When
    one is written it is a kernel image with `/data` as the working
    directory and `agent.yaml` bind-mounted, no ports unless `web:` is
    configured, and signal-cli and llama-server outside the container.

### 3.12 Evaluation and the harness

1. CI runs the full pytest suite on every push and PR, after proving
   sqlite-vec loads so vector tests cannot silently skip, with a global
   per-test timeout; a kernel-only job runs the suite without the
   cognition extra.
2. The harness drives probes over local IRC against a real llama-server
   and reads verdicts out of `sessions.db`; profiles switch the config
   template and probe set. The probe set covers the round trip, a tool
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
5. Fork evals (#73): the cache probes that gate enabling forks
   (`docs/design.md`, Forks),
   then a window-render replay instrument, a planted-defect exchange
   fixture per lens, and consolidation and compaction fixtures,
   deterministic where possible and judged where not. Past the cache
   probes they are benchmarks that report shifts and regressions; they
   gate nothing.
6. Persyn's §9 criteria run as an integration suite against a live daemon
   with a scripted interlocutor: recalls across restart, hedges weak
   memories, refuses to remember fabricated events, pushes back on flawed
   premises, declines a capable request, initiates unprompted, goes
   dormant while the heartbeat sleeps.
7. The right to silence (§3.7) has a behavioural eval: in a
   conversational register, closers get silence and substantive questions
   get replies; in a task register, every addressed request is answered.
8. Phase 6 benchmarks are authored before their toggles are built.
9. A third evaluation tier, below CI and the scheduled LLM-judged
   benchmarks, is the operator judging over weeks: consistency of voice,
   distinctiveness, legible drift, generalization under novel probes.
   Personality is longitudinal and relational and its main failure mode
   (regression to the generic base-model character) is invisible to
   fixtures, which all start from the same weights, so this tier is not
   automatable and the record says so rather than faking a fixture. Its
   one instrument is a periodic diff of the self-describing store
   (`self` facts, and the compiled self-model), read as
   coherent drift, thrash, or flattening. The hypothesis-tier triggers in
   §3.7 are defined against this tier.

## 4. Decisions

The ones that would be expensive to relitigate by accident, with the
record that holds the reasoning.

| Decision | Where |
|---|---|
| apluggy entry points, not a hand-written registration order; `depends_on` validated at startup | `docs/design.md`, Plugin Registration Order |
| Single-turn dispatch; multi-turn reasoning emerges from notifications | README; `design.md`, Agent Loop |
| Tail-append CONTEXT; no mid-window mutation | `bootstrap-mapping.md` §2.2 |
| Compaction is memory formation, not forgetting | §3.1 |
| Two forks off one prefix for compaction and consolidation, not one combined call | `plans/forks.md` |
| Compaction forks from the dropped prefix, not a described cut | `plans/forks.md` |
| Every background call with a live window forks it; a miss skips, never falls back to standalone; #73's cache probes gate forking, its fork-vs-standalone comparisons don't | #180; `design.md`, Forks |
| §3.8 states fork outcomes; backend-specific fork design (prefix parity, fork-by-default, cost model, rollout gate) lives in `design.md` | #188 |
| Demote, don't delete; usage-weighted retention; near-duplicate merge at consolidation | §3.1 |
| Operator-only `redact` is the one deletion path; it stays in the kernel | §3.1; `kernel-split.md` Decisions |
| Knowledge graph cut, not optional | §5.7 |
| Emotion classifier replaced by appraisal valence | §5.2 |
| Engagement decision inverted: cheap gate first, generate only past threshold | §5.3 |
| Two-stage appraisal; stage 1 never calls a model; core mints the exchange key; pull-based compute | §3.2; `phase-2.md` traps 1–2 |
| Origin is stamped and propagated, never inferred; critique eligibility by origin | §3.3 |
| Silent tasks require no `tool_call_id` | §2.3 |
| Persona may tune its own gates; `immutable_settings` is the valve | operator directive 2, 2026-07-06 |
| Best-guess defaults first; thresholds fit from experience; no closed-loop auto-fit | operator directives 1, 3 |
| Engagement gate calibrates offline only | §3.2 |
| Hedging vocabulary is persona, not architecture | §5.9 |
| The heartbeat starts on system idle, not on a schedule, and acts only when warranted; sleep is durable | §5.8; #134 |
| Idle-triggered work may carry a latency bound: the quiet it needs decays with waiting, so traffic can't starve it | #214 |
| The agent may decline any turn when it has nothing to add (Grice's maxims, by register); gates still never withhold a direct reply by default | #215 |
| Model endpoints may be remote; a role's credentials come from the environment, and the threat model names what each role sends | #217 |
| Cross-channel sends are a requirement, gated by an operator allowlist that starts empty, a budget, a threat-model entry, and trust and sensitivity when those plugins are loaded | #167 |
| Operator-seeded goals allowed | §5.14 |
| Memory poisoning answered on the write side with trust tagging (trust plugin) | §3.9, §5.10 |
| No sandbox; the subagent/MCP seam is where one would attach | §5.11; `design.md`, Known Risks |
| No distributed infrastructure | §5.12 |
| Signal keys on ACI; E.164 resolved forward once per connect; no reverse resolution | `design.md`, Signal |
| Discord: no privileged intent; outbound mentions disabled; threads own their history | #80 |
| Web: loopback default, auth cannot be turned off, HMAC cookie, strict CSP, no CDNs; duplicates over losses in replay | #87; `retro-web-transport-session.md` |
| ACP: client-spawned subprocess, official SDK, v1 only, no auth, swappable tool backend | `acp-transport-design.md` D1–D5 |
| ACP rejects a concurrent prompt; a failed final delivery still ends the turn | #92 initial values |
| Metering at `LLMClient`, not the turn loop | §3.7 |
| `resolve_compaction` keeps `firstresult`; the first non-None return is how a plugin replaces the compactor | `kernel-split.md` Decisions; #23 |
| Same repository, second distribution for cognition; no compatibility aliases on renamed hooks | `kernel-split.md` R5, R6 |
| Typed event bus deferred with a go criterion (observe class past 20 hooks, or a third `send`-shaped consolidation) | `kernel-split.md` Part 1 §5 |
| `thinking.py` is kernel | `kernel-split.md` R9 |
| Review findings gate on consequence, not reachability; tests derive from acceptance criteria, not the design's test list; new capability goes through requirements | `retro-web-transport-session.md` |
| `discord-py` is a `discord` extra, like `acp` | #117 (R10) |
| ACP keeps rejecting a concurrent prompt unless a real client is shown to steer with a bare prompt; a failed final delivery is `end_turn` | #93, #99 |
| Summary role follows the mode: fork `assistant`, standalone `user` with data framing | `threat-model.md` finding 3 |
| No process separation until corvidae runs on a machine the operator does not own | `threat-model.md`, Privilege separation |
| Personality is duck-typed and judged longitudinally; goal retirement is a consolidation event; `self` is a fact subject, ungated beyond trust inheritance; the self-model compile reads only facts above a trust floor and sends the operator each diff; interoception line and self-model compile; commitment records and endogenous goals only on a named failure | `plans/personality-and-autonomy.md` (PR #48) |
| No runtime plugin loading; changing the plugin set is a restart | #28 |
| No container image until a second deployment needs one | #37 |
| Topic structure is the `topic` subject's job, not a retrieval dimension | `shakedown-findings.md` |
| `plans/new-hooks.md` retired. Per-turn tool filtering and system-prompt transforms are rejected (the tool schemas and system prompt are the head of the prompt, principle 2); inbound-text rewriting is rejected (the log keeps what arrived); config-change notifications are unneeded (tunables resolve at decision time); a compaction veto is rejected (it trades a summary for an over-budget request). A per-call tool veto and per-channel tool sets, fixed per conversation, are tracked for Phase 5's trust work; a channel-created observation waits for a second implementer under §3.1.11 | #31, #148, #149, #150 |
| The kernel owns the clock; idle keeps firing through quiet, and no plugin needs another's queues to learn of it; the April push-based decoupling stands | #167; #165 |
| Upkeep runs on time regardless of agency plugins; the scheduler is agency and optional | #167 |
| Transports deliver every emission on their own channels regardless of origin | #167 |
| The agent is one actor; trust and sensitivity, not channel walls; channel compartments are an operator option; the whole-memory default has a threat-model entry | #167 |
| The operator decides the trust boundary; memory reach and cross-channel flow default open, inbound transports default-deny; safety features layer in as plugins | #167 |
| Corvidae is a testbed first and a robust harness second | #167 |
