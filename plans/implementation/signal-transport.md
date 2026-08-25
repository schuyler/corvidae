# Signal transport — a DM channel for the daemon

**Effort:** M. **Dependencies:** none in the phase sequence — this is
orthogonal to the Phase 2 appraisal/critique arc and can land beside it.
**Normative references:** `docs/design.md` "Channel System" and "Transports";
`docs/plugin-guide.md` "Channels"; AGENTS.md for TDD and exception discipline.

**Goal:** the operator can hold a private, continuous conversation with the
daemon over Signal from an ordinary phone, and the daemon can start one.

The work packages are tiered. **v1 — the pre-departure critical path — is
WP-S1 through WP-S7 plus WP-S11**: the transport itself, registration, config,
and docs. **WP-S8 (unprompted sends) and WP-S9/WP-S10 (the private flag) are
follow-on**, post-departure. WP-S9/S10 are not transport-local — they add a
privacy flag to `message_log` and propagate it through compaction, memory
consolidation, and redaction, and that schema change is the part that can
break other things; it lands only after the transport is confirmed working.
In v1, content arriving under a disappearing-message timer is retained as
ordinary content — no private flag — accepted by the operator, who does not
send disappearing messages.

## Read first

**Pipeline status.** Requirements are gated by the operator as of
2026-08-25, including the two-tier scope: v1 is WP-S1–S7 plus WP-S11;
WP-S8, WP-S9, and WP-S10 are follow-on. The design passed a cold Rule of
Two review on 2026-08-25. The next phase is Red (failing tests) for the v1
tier, under the standard-task pipeline: red → review → gate → green →
review → gate → docs → review → gate → full suite → operator acceptance.
Reviewers treat the requirements and this design as settled premises.
Implementation artifacts must not reference R-numbers or WP-numbers. The
pre-departure operational checklist is operator-owned and partially
parallelizable with implementation — its registration steps need no code.

- `corvidae/channels/irc.py` — the transport pattern this follows: `on_init`
  reads config, `on_start` opens a connection under a retry loop, inbound
  calls `pm.ahook.on_message`, `send_message` broadcast-filters. Also
  `split_message`, which WP-S5 moves.
- `corvidae/channels/cli.py` — the four `send_*` hooks a transport may
  implement and how each opens with `matches_transport`.
- `corvidae/hooks.py:489` — `should_process_message`, the gate WP-S4 uses.
  No built-in plugin implements it today; this is the first.
- `corvidae/persistence.py:36` — the `message_log` DDL that WP-S9 alters.
- `corvidae/memory.py:622` — the consolidation range query WP-S10 must
  exclude private rows from without stalling the watermark.
- `corvidae/compaction.py:126` — `compact_conversation`, where WP-S10 makes
  summaries inherit the flag.
- `tests/test_irc_plugin.py` — the transport test pattern. Signal's is easier:
  a fake JSON-RPC server on a unix socket in `tmp_path`, no library mocking.
- `harness/README.md` and `harness/session_driver.py` — the Buster shakedown
  harness, IRC-driven today. Its existing probes already cover several
  requirements end-to-end that unit tests can only approximate: `interleave`
  (R8), `restart_recovery` (R4), `compaction` (R27c's blast radius). A Signal
  equivalent is out of scope here, but R4, R8, R15, and R34 are only really
  proven at that level — note the gap rather than claiming unit coverage.

## Requirements

Numbered as agreed. Tests in the work packages cite these. R37 is
intentionally absent — a numbering gap, and nothing renumbers. R27b is also
absent; see the note under R27a.

### Conversation

- **R1** Receives text sent to the bot account in a 1:1 conversation, replies in
  the same conversation.
- **R2** Replies arrive as ordinary Signal messages from the bot account.
- **R3** A reply over the per-message limit is delivered in full, split at
  paragraph, then sentence, then word boundaries. Never truncated.
- **R4** History persists across restarts, keyed to one stable identity. A
  restart loses no context; an identity change does not fork the history.
- **R5** Messages sent while the daemon is down are processed on return, or
  their loss is logged. Silent loss is not acceptable.
- **R6** The agent never responds to its own output.
- **R7** Messages originating from the bot's own account, from any device
  signed into it, are not user input.
- **R8** Messages are processed in the order received; rapid succession does
  not interleave.

### Identity and authorization

- **R9** Only an explicitly authorized sender can drive the agent. Others are
  refused and logged — an attempt is visible.
- **R10** The authorized sender maps to the same conversation regardless of how
  Signal identifies them on a given message, and continues to if that changes.
- **R11** The bot's identity is distinct from the operator's personal account.

### Liveness and feedback

- **R12** During slow work the sender gets an indication the agent is working.
  In a DM, a minute of silence is indistinguishable from a crash.
- **R13** A delivery failure is logged and visible. The agent must not appear
  to have answered when it did not.
- **R14** The conversation is observable from the host without a Signal client.
- **R38** No spend cap is required — inference is local. But local inference is
  serial and contended with compaction, appraisal, critique, and subagents, so
  a message can wait before generation starts. R12's indicator must therefore
  cover queueing delay, not only generation.

### Operational

- **R15** Reconnects on its own after connectivity loss, without a daemon
  restart or human intervention.
- **R16** Optional in the sense every corvidae plugin is: omitting its config
  disables it — as does listing it in `plugins.disabled` — and its failure or
  absence neither crashes the daemon nor affects other transports.
- **R17** Bringing the daemon up requires no interactive step. Interactive
  setup is one-time, out of band, and documented.
- **R18** Ongoing operation requires no recurring human maintenance task.
  Operational, not a work package: primary-device registration removes the
  30-day re-link ritual, and supervision is operator-managed — discharged by
  the pre-departure checklist, not by a test.
- **R43** Signal is strictly optional at deploy time. The daemon starts and
  runs normally with no `signal:` block in agent.yaml and no signal-cli
  installed; the transport is inert unless configured; no packaging-level
  hard dependency is added for non-Signal deployments — JSON-RPC over a unix
  socket is stdlib.

### Data handling

- **R19** It is documented explicitly that end-to-end-encrypted content is
  persisted in plaintext on the host — session database, logs, memory — and
  that the conversation key contains a personal identifier.
- **R20** There is a supported way to tombstone a conversation's stored
  content — content wiped in place, rows retained — matching
  `corvidae/commands/redact.py`'s existing semantic.

### Scope boundaries for v1

Each must degrade gracefully, not merely be absent.

- **R21** Group conversations are out of scope; a group message is ignored
  without error.
- **R22** Inbound attachments do not crash anything. Accompanying text is still
  processed; the attachment is logged and dropped.
- **R23** Reactions, quotes, edits, and delete-for-everyone are ignored without
  error.
- **R24** Outbound messages are text only.

### Disappearing messages — retain and flag (follow-on)

R25–R29 belong to the follow-on tier (WP-S9, WP-S10). In v1, content arriving
under a disappearing-message timer is retained as ordinary content — no
private flag — accepted by the operator, who does not send disappearing
messages.

- **R25** Content in a conversation with a disappearing-message timer is
  retained. Expiry does not delete it from the store.
- **R26** Such content is flagged private at rest. The flag is durable, is
  attached to the stored message, and is not re-derived later from the timer's
  current value.
- **R26a** Private content is retained in the conversation and in the jsonl log,
  unchanged. Private limits how far content travels *inside the agent*, not
  whether the operator can see it.
- **R27a** Private content is excluded from long-term memory consolidation.
  This is also what keeps private content away from subagents: their context
  is built from `SUBAGENT_SYSTEM_PROMPT` plus model-authored instructions
  alone (`corvidae/tools/subagent.py`) — conversation history never reaches
  them, and memory recall (memory tools are in the subagent tool set) is the
  only route by which stored conversation content could. R27b is
  intentionally absent; the standing constraint lives in trap 9.
- **R27c** A compaction summary covering any private message inherits the flag.
  Without this, summarization launders private content into unflagged storage
  that outlives the window.
- **R28a** Private content is bulk-selectable for redaction, without hunting
  individual message IDs.
- **R29** A change to the conversation's timer is recorded, so it is later
  possible to tell which messages fell under which policy.

### Unprompted messages (follow-on)

R30–R33 belong to the follow-on tier (WP-S8).

- **R30** The agent can send with no inbound message triggering it.
- **R31** Unprompted sends are rate-bounded. A bug or a chatty idle loop cannot
  produce a flood. This is a correctness guard, not a courtesy feature — there
  are deliberately no quiet hours.
- **R32** Unprompted messages are distinguishable from replies in the logs.
- **R33** The self-echo rule (R6, R7) holds for unprompted sends.

### Backlog

- **R34** Every message queued during downtime is processed, in order sent.
- **R35** Each carries its original send time into the agent's context. A
  three-day-old message must be legible to the agent as three days old.
- **R36** Backlog processing does not bypass R9 or R8.
- **R42** Backlog is one message per turn, one reply per message. Batching is
  deferred to the engagement gate, not solved here — see Deferred.

### Read receipts

- **R39** Inbound messages from the authorized sender are marked read.
- **R40** Read receipts are not sent to unauthorized senders. A refused message
  leaves no signal that the account is live and monitored.
- **R41** The receipt is emitted when the agent begins processing, paired with
  the working indicator, so "read" is never followed by unexplained silence.

## Design

### Backend and account topology

signal-cli in daemon mode, JSON-RPC 2.0 over a unix socket, newline-delimited.
Inbound arrives as notifications with method `receive`; outbound is a `send`
request. This adds no Python dependency — an asyncio unix connection and
`json.loads` per line.

corvidae **connects to** an already-running daemon; it does not spawn or
supervise one. Registration is interactive and stateful and belongs to the
operator; supervision is operator-managed (screen, per standing practice)
and must survive an unattended reboot of the host — corvidae likewise.

The bot account is a separate Signal account on its own phone number,
currently registered on a spare phone. The operator's personal account and
its primary device travel with him and are untouched by this design.

**signal-cli is registered as the primary device on the dedicated bot number,
not linked as a secondary device.** This is an operational requirement, not a
preference. Signal unlinks secondary devices if the primary phone does not come
online at least once every 30 days, and only the primary can re-link. With the
spare phone kept powered off, a linked deployment would need a recurring
monthly ritual and would fail silently when missed — violating R18 and, on
failure, R5. As primary, signal-cli can also add devices, so Signal Desktop
can be linked to the bot account for a live mirror of the conversation.

Registering signal-cli as primary deregisters the spare phone — expected.
Afterward Signal must be logged out of the spare phone and the phone kept
off: a re-registration from it steals the number back mid-trip,
unrecoverably (recovery needs an SMS to the bot SIM). Registration must
happen pre-departure, while the spare phone's SIM is reachable — both the
captcha'd `register` call and the SMS/voice verification code need it.

One-time setup, documented for the operator, not automated:

```sh
# captcha token from https://signalcaptchas.org/registration/generate.html
signal-cli -a +BOTNUM register --captcha 'signalcaptcha://…'
signal-cli -a +BOTNUM verify CODE     # code arrives by SMS at the bot number
signal-cli -a +BOTNUM daemon \
    --socket /run/corvidae/signal.sock \
    --receive-mode=on-connection
```

Verify the exact flag spellings against the installed signal-cli
(`signal-cli register --help`, `signal-cli daemon --help`) before running.

**`--receive-mode=on-connection` is load-bearing.** Under other receive
modes a running signal-cli daemon retrieves and acks inbound messages even
with no JSON-RPC client attached, so anything arriving while corvidae is
down but signal-cli is up is consumed without a trace — violating R5 and
R34. `on-connection` defers retrieval until a client is attached, leaving
the corvidae-down window queued server-side. If the installed version
spells the mode differently, the property to verify is exactly that:
signal-cli neither receives nor acks while no JSON-RPC client is connected.

signal-cli's data directory is the sole credential for the bot account;
it is backed up off-host (see the pre-departure checklist). If it is lost
mid-trip, re-registration needs SMS at the bot SIM.

If registration lock is set on the account, the PIN is required to
re-register. Record it before starting; without it, re-registration waits
seven days.

### Plugin shape

`corvidae/channels/signal.py`, `SignalPlugin(CorvidaePlugin)`, following
`IRCPlugin`:

- `depends_on = frozenset({"registry"})`, resolved with `get_dependency`.
- `on_init` reads the `signal:` config block. An absent block means the
  plugin loads and stays inert, silently — no connection, no channels,
  nothing above DEBUG (R16, R43). A present but malformed block raises at
  startup with a clear message: silence is for absence, not for errors.
- `on_start` opens the socket under a retry loop with the same backoff ladder
  IRC uses — 10s initial, x2, 300s cap (R15).
- `on_stop` cancels the read task and closes the socket, logging rather than
  swallowing on the way out.
- `send_message` and `send_progress` are implemented; `send_thinking` and
  `send_tool_status` are deliberately not. Nobody wants raw tool traces in a
  DM, but intermediate assistant text before a tool dispatch reads as an
  ordinary message ("let me check that") and is real liveness, so
  `send_progress` delivers through the same path as `send_message` — the
  same choice `IRCPlugin` made in 20a3b79. It bypasses the reply/unprompted
  classification, so the R31 rate bound structurally cannot apply to it —
  see "Unprompted sends and reply classification".

The broadcast-filter rule, stated precisely: every implemented `send_*`
hookimpl must be side-effect-free for channels that are not this
transport's. A hookimpl satisfies that either by opening with
`if not channel.matches_transport("signal"): return`, or by delegating
unconditionally to one of the transport's own methods that does — IRC's
`send_progress` forwards to its `send_message`, which filters. Signal's
`send_message` filters directly; its `send_progress` filters itself, then
hands text to the internal delivery path.

### Channel scope and identity

Scope is the sender's **ACI UUID**, so channel IDs look like
`signal:8f2c…`. Phone numbers are not used as the stored scope: `sourceNumber`
can be absent under phone-number privacy, and `transport:scope` is the
persistence key for `sessions.db`, the jsonl logs, and memory. Keying on a
field that can vanish forks the history (R4, R10).

Config accepts `signal:+15551234567` as a friendly alias — in the `allow`
list and as a `channels:` key — but the **recommended form for both is the
ACI itself**: an ACI-keyed `channels:` entry pre-registers the durable
channel directly at startup, and an ACI allowlist entry needs no resolution
at all.

E.164 aliases resolve in the transport's inbound path, **before the channel
exists**. `ChannelRegistry.get_or_create` applies `config` only when it
creates the channel (`corvidae/channel.py:135–155`), so the `ChannelConfig`
must be in hand on the first inbound message from that sender — applying it
any later is a no-op. On each inbound, after filtering, the transport takes
the sender's ACI from the envelope and checks it against the configured
E.164 aliases: an alias matches when it equals the envelope's
`sourceNumber` (when present), or when signal-cli's recipient store maps it
to that ACI (a JSON-RPC contact lookup — `listContacts` or the installed
version's equivalent; verify the method name). The matching alias channel's
`ChannelConfig` — read from the channel `load_channel_config` pre-registered
under the literal alias id — is passed into that same first
`registry.get_or_create("signal", aci, config=...)` call, so the
`system_prompt` override lands at creation on the channel messages actually
use. The lookup result is cached per ACI for the life of the process;
signal-cli's own data directory is the durable number↔ACI record, so
corvidae persists no mapping of its own.

Allowlist matching for an E.164 entry uses the same inbound-path step:
match `sourceNumber` when present, else the recipient store. An alias
signal-cli cannot resolve, from a sender whose number is hidden by
phone-number privacy, simply never matches — putting the ACI in the config
sidesteps all of this.

`load_channel_config` also leaves the literal `signal:+15551234567` alias
channel sitting in the registry. That phantom is tolerated rather than
avoided: the registry is in-memory, no inbound traffic ever lands on the
alias id, and `sessions.db` rows key only on channels that carry messages —
so it writes nothing and dies with the process, and removing it would need
a registry deletion API that nothing else wants. ACI-keyed `channels:`
entries avoid it entirely.

### Inbound filtering

Process an envelope only when all hold:

1. It carries a `dataMessage`. A `syncMessage` is never input (R7) — that is
   how the account's own traffic from other devices arrives, and treating it
   as input is the infinite-loop failure mode.
2. Its source is not the bot's own account (R6).
3. It carries no `groupInfo` (R21).
4. Its text is non-empty after attachments are dropped (R22).

Everything else is logged at DEBUG and discarded. Reactions, quotes, edits,
and remote deletes fall out of rule 4 without special cases (R23).

### Authorization

`should_process_message` returns `False` for any sender whose ACI is not in
the configured allowlist, default-deny. The gate is `REJECT_WINS`, and
`on_message_rejected` fires so the refusal reaches the outcome log (R9).
The hookspec also passes `correlation_id` (`corvidae/hooks.py:489`), which
is spec-required — the hookimpl must bind it without a default, or the
registration-time arg-binding guard rejects the plugin.

Default-deny is right for this transport specifically, even though IRC is
open: a Signal DM is a direct line from anyone who knows the number to an
agent holding `shell`, `read_file`, and `write_file`.

### Liveness

Typing and the read receipt are decoupled: typing must cover queueing delay
(R38), the receipt marks processing start (R41).

**Typing indicator (R12, R38).** Typing starts in the transport's own
inbound handler, the moment an envelope passes the inbound filter and the
transport's allowlist check — the same predicate its
`should_process_message` hookimpl applies — before the message is handed to
`on_message`. Starting transport-side is what covers queueing delay: a
message that arrives while a long turn holds the per-channel `SerialQueue`
(`corvidae/agent.py:275–282`) would otherwise sit queued with no indicator
until dequeue. Signal clients expire a typing indicator after roughly 15
seconds without refresh, so the transport runs a per-channel refresh task
that re-sends it every 10 seconds. The task is driven by a per-channel
count of outstanding admitted messages: incremented at inbound accept,
decremented on each final reply send, each error-path send, and each
`on_message_rejected` on the channel — another gate plugin can veto after
the transport's own check passed. Typing runs while the count is positive
and stops when it reaches zero or a send fails. The design assumes signal's
own `should_process_message` implementation is the only gate plugin on
signal channels; if a second gate plugin ever vetoes a message after
signal's check has pushed an envelope timestamp, the receipt FIFO below
gains no pop for that veto and later read receipts attach to a stale
timestamp — any future gate plugin on signal channels must add a pop on
`on_message_rejected`. WP-S8 must also revisit this decrement rule: under
WP-S8's classification, a notification-originated turn's final send would
decrement the counter while a user message is still owed a reply, so the
rule becomes decrement-only-on-`reply`.

**Read receipt (R39, R41).** The receipt fires when the agent begins
processing the message, not at arrival: with contended local inference a
message can sit queued, and a "read" marker followed by two minutes of
silence is worse than no marker. Processing start is observed as
`on_message_persisted` firing for a correlation id the transport recorded
from `on_message_admitted` on one of its channels. The receipt needs the
sender's envelope timestamp, which the correlation id does not carry, so
the transport keeps a per-channel FIFO of pending envelope timestamps —
pushed at inbound accept, popped at each such processing start — which
stays aligned because the serial queue preserves order.

A failure to send the receipt or an indicator is logged and never blocks
the reply (R12). Unauthorized senders get neither (R40).

### Unprompted sends and reply classification (follow-on — WP-S8)

**Trigger.** Core already carries the mechanism for R30: any plugin can
enqueue a notification turn via `on_notify` on a signal channel, and the
agent's response leaves through `send_message` like any other. What
*decides* to speak — an initiative policy over `on_idle`, appraisal, or
anything else — is not designed here; see Deferred. WP-S8 ships the
transport side in full: delivery, classification, the rate bound, and
logging, all testable by driving `on_notify` directly.

**Classification.** `send_message(channel, text, latency_ms)` carries no
origin, so the transport classifies from per-channel turn state. That state
has three values: **idle** (nothing owed), **queued** (one or more admitted
messages awaiting dequeue), and **in flight** (a user turn being processed):

- `on_message_admitted` (filtered to signal channels) records the
  correlation id as owed a reply — the queued state. This is the same
  id-recording the read receipt uses.
- `on_message_persisted` for a recorded id marks that turn in flight —
  the same processing-start observation the read receipt fires on.
- The next final `send_message` on that channel while a turn is in flight
  is a **reply**; it clears the in-flight state (back to queued or idle).
  This pairing is sound because the per-channel serial queue runs one turn
  at a time, and a turn produces exactly one final send (tool-cycle
  re-entries included; the error path's `send_message` also closes the
  turn).
- A final `send_message` with no turn in flight is **unprompted** — in the
  idle state and equally in the queued window between admission and
  dequeue: the only turn that can complete on the channel while user
  messages sit queued is a notification turn ahead of them in the serial
  queue, and notification turns are unprompted by construction —
  `on_message_admitted` fires only for USER messages, so their correlation
  ids are never recorded.

**Rate bound (R31).** An unprompted send within `unprompted_min_interval`
seconds of the previous one on the same channel is suppressed and logged at
WARNING. The assistant turn still exists in `message_log` — suppression is
delivery-side — so the log makes the drop visible. `send_progress` is
exempt structurally: it bypasses classification by calling the internal
delivery path, and it only ever fires mid-turn, where it would classify as
part of a reply anyway.

**Log distinction (R32).** Every outbound is logged with its
classification: `reply`, `unprompted`, or `progress`.

**Self-echo (R33).** signal-cli does not receive sync copies of its own
sends on the sending device, and any copy surfacing via a linked device
arrives as a `syncMessage`, which inbound rule 1 discards.

### Backlog timestamps

With `--receive-mode=on-connection`, messages queued during downtime are
delivered in order the moment corvidae attaches; each envelope carries the
sender's send `timestamp` (epoch milliseconds). The original send time
reaches the agent's context transport-locally: when an envelope's timestamp
is more than five minutes older than local time at decode (a module
constant, not config), the transport prepends a line of the form
`[sent 2026-08-22 09:14 UTC]` to the text before calling `on_message`
(R35). The prefix is part of the message content, so the agent's context,
`message_log`, and the jsonl log agree (R14), and per-message identity is
untouched (R26, R28a).

No hook signature changes, so the pluggy default-argument-dropping trap
(`tests/test_hook_arg_binding.py`) is not engaged and IRC and CLI — which
call the same `on_message` hook — are untouched. The alternative, a
spec-optional `sent_at` parameter on `on_message` following `latency_ms`'s
precedent, was declined: it would touch the hookspec, `Agent.on_message`,
`QueueItem`, and context rendering for one transport's need.

Backlog messages are ordinary messages otherwise: each passes the
authorization gate (R36), and the serial queue preserves arrival order and
yields one turn and one reply per message (R34, R42, trap 8).

### Private content (follow-on — WP-S9, WP-S10)

Until this tier lands, disappearing-message content is retained as ordinary
content with no flag — the accepted v1 posture.

A `private INTEGER NOT NULL DEFAULT 0` column on `message_log`, set when the
inbound envelope's `expiresInSeconds` is non-zero. Written at insert; never
recomputed from the timer's later value (R26).

The flag propagates along every path by which content leaves the conversation:

| Path | Behavior |
|---|---|
| jsonl log | unchanged — the operator's window stays whole (R26a, R14) |
| memory consolidation | private rows excluded from the range query (R27a) |
| compaction summary | summary inherits `private=1` if its range contains any (R27c) |
| `corvidae redact` | new `private` form tombstones every flagged row — content wiped, rows retained (R20, R28a) |

Consolidation excludes private rows but the watermark must still advance past
them. Skipping rows without advancing stalls consolidation permanently.

Two design questions are open; both are settled when this tier is picked up,
before its red tests are written:

- **How the flag travels from the transport's inbound handler to the
  persistence write.** `on_message(channel, sender, text)` has no room for
  it; the agent builds the user dict at `agent.py:293` without a flag; and
  no existing hookspec carries it — `on_conversation_event` passes
  `message_type` as a hookspec parameter (`hooks.py:735`), not as a
  message-dict key, so there is no dict-key path to piggyback on, and
  `_strip_internal_keys` drops any `_`-prefixed in-window tag before
  serialization anyway. Resolving this likely means a hookspec change.
- **The storage form of R29's timer-change record** — a `message_type`, a
  separate table, or jsonl-only. The red test cannot be written until this
  is chosen.

### Message splitting

`split_message` moves from `channels/irc.py` to `corvidae/channels/split.py`,
with an import shim left in `irc.py` so existing call sites and tests keep
working. Signal calls it with a larger limit than IRC's 400 bytes; the default
is configurable and starts at 2000 characters.

### Configuration

```yaml
signal:
  socket: /run/corvidae/signal.sock   # signal-cli daemon socket
  account: "+15550001111"             # the bot's own number
  allow:                              # ACIs or E.164; default-deny
    - "+15551234567"
  message_chunk_size: 2000            # max chars per outbound message
  unprompted_min_interval: 300        # seconds between agent-initiated sends

channels:
  signal:+15551234567:      # ACI-keyed form (signal:<aci>) is recommended
    system_prompt: "..."
```

Omitting the `signal:` block disables the transport entirely — the plugin
loads and stays inert, silently (R16, R43). A present but malformed block is
a loud startup error.

## Design constraints and traps

1. **`syncMessage` is not input.** The single highest-consequence rule. It is
   less visible day-to-day when signal-cli is the only active device — a
   device does not receive sync copies of its own sends — which is exactly why
   it must be a test and not a comment. It starts firing the moment Signal
   Desktop is linked or the phone is booted.
2. **Register the plugin** in `[project.entry-points.corvidae]` in
   `pyproject.toml`. Without it the plugin is never loaded and nothing errors.
3. **Broadcast-filter every `send_*` hook.** pluggy calls all transports for
   every send.
4. **Do not key channels on phone numbers.** See "Channel scope and identity".
   Getting this wrong is only discoverable after history has accumulated
   against the wrong key.
5. **Never swallow.** Socket errors, malformed JSON, and send failures are
   logged with `exc_info=True` and the read loop continues. A single bad frame
   must not kill the transport (R13, R15).
6. **The consolidation watermark must advance past skipped private rows**
   (WP-S10). Otherwise memory stops consolidating anything, quietly.
7. **There is no dict-key path into persistence for a per-message flag.**
   `_strip_internal_keys` drops `_`-prefixed keys before serialization, and
   `message_type` reaches persistence as a hookspec parameter on
   `on_conversation_event`, not as a message-dict key. WP-S9's flag needs
   its own route — see the open questions under "Private content".
8. **The unbounded queue.** `SerialQueue` uses `put_nowait` on an unbounded
   `asyncio.Queue`. A large backlog is admitted all at once; R42 accepts the
   resulting burst of turns rather than dropping messages.
9. **Subagents and the private flag — standing constraint.** Subagent
   context today is `SUBAGENT_SYSTEM_PROMPT` plus model-authored
   instructions (`corvidae/tools/subagent.py`); conversation history never
   reaches it, which is why no exclusion path exists to test. Any future
   code path that feeds conversation content to subagents must respect the
   private flag. Model-authored subagent instructions remain an
   unfilterable laundering path — the model can restate private content in
   its own words — and that is accepted.

## Work packages

Red tests first, per AGENTS.md. Each package names the requirements its tests
cover. **v1 — the pre-departure critical path — is WP-S1 through WP-S7 plus
WP-S11. WP-S8, WP-S9, and WP-S10 are follow-on**, picked up post-departure.

**Implementation note.** Implementation artifacts — code, comments,
docstrings, test names — must not reference R-numbers or WP-numbers. This
plan document is deleted once implemented, and such references dangle.
Tests are named for the behavior they assert.

**Testing posture.** Unit tests fake the narrowest surface needed: a local
socket speaking canned JSON-RPC frames, nothing more — no elaborate mock of
signal-cli. Behavior that only manifests against the real daemon —
receive-mode semantics, typing-indicator expiry timing, restart windows —
belongs to the live shakedown and the harness, not pytest; thinner unit
coverage there is the accepted trade.

### WP-S1 — Transport skeleton and connection lifecycle (R15, R16, R17)

`SignalPlugin` with `on_init`/`on_start`/`on_stop`, connecting to the socket
under backoff retry.

Red tests (`tests/test_signal_plugin.py`): a fake JSON-RPC server on a unix
socket in `tmp_path`; plugin connects and reads frames; a dropped connection
retries with backoff; absent `signal:` config means no connection and no
error; `on_stop` cancels cleanly with no pending-task warnings.

### WP-S2 — Inbound decoding and filtering (R1, R6, R7, R21, R22, R23)

Envelope decoding and the four filter rules.

Red tests: a `dataMessage` fires `on_message` with the decoded text; a
`syncMessage` does not; a `groupInfo` envelope does not; an envelope from the
bot's own account does not; an attachment-only message is dropped without
error; attachment-plus-text processes the text; a malformed frame is logged
and the loop survives.

### WP-S3 — Channel scope and identity (R4, R10)

ACI-keyed scope with E.164 alias resolution.

Red tests: scope is the ACI, not the number; a config alias resolves to the
same channel; an envelope with `sourceNumber` absent still resolves; the
channel ID is byte-identical across both forms; a `channels:` alias's
`system_prompt` override is present on the ACI channel the first inbound
message creates.

### WP-S4 — Authorization gate (R9, R11, R40)

`should_process_message` with a default-deny allowlist.

Red tests: an allowlisted ACI is admitted; a non-allowlisted one is rejected
and `on_message_rejected` fires; empty allowlist rejects everything; a
rejected sender receives no read receipt. R11 is carried by the account
topology and the default-deny allowlist, not by a dedicated test.

### WP-S5 — Outbound send and splitting (R2, R3, R13, R24)

`send_message`, plus moving `split_message` to `corvidae/channels/split.py`.

Red tests: existing IRC splitting tests pass against the new module location;
a 5000-character reply splits at paragraph boundaries under a 2000-char limit;
`send_message` on a non-signal channel is a no-op; a send failure is logged
and does not raise into the agent loop. R24 is carried by `send_message`'s
text-only signature, not by a dedicated test.

### WP-S6 — Liveness (R12, R38, R39, R41)

Typing from the transport's inbound handler, with the per-channel refresh
task; read receipt at processing start.

Red tests: typing starts at inbound accept, before the turn is dequeued —
with a prior turn holding the queue, a newly arrived message produces typing
requests at the fake server while it waits; with the refresh interval
shortened for the test, a turn that outlasts that interval — including
across a tool loop — produces **at least two** typing requests, so the test
fails without the refresh loop; typing stops when the last owed reply is
sent and does not resume after; a rejection on the channel decrements the
outstanding count; the receipt is emitted at processing start, not at
arrival — for a queued message, only when its own turn begins; a failure to
send either is logged and does not block the reply. The real ~15-second
client-side expiry is shakedown territory.

### WP-S7 — Backlog timestamps (R5, R34, R35, R36, R42)

Original send time carried into context via the stale-message prefix.

Red tests: three envelopes with old, ordered timestamps produce three turns
in order, one reply per message (R42); each turn's context carries its
original send time as the `[sent …]` prefix, and a fresh message carries no
prefix; backlog still goes through the authorization gate; a fake server
that delivers queued envelopes immediately on connection — the
`on-connection` contract for the corvidae-down/signal-cli-up window — has
all of them processed in order (R5, R34). signal-cli's own side of that
window (no receive, no ack while disconnected) is pinned by the daemon flag
and proven in the shakedown, not in pytest.

### WP-S8 — Unprompted sends (R30, R31, R32, R33) — follow-on

Transport-side classification, rate bound, and logging, driven by
notification turns.

Red tests: a notification-originated turn's response is delivered with no
inbound message and logged as unprompted; a second unprompted send inside
the minimum interval is suppressed and logged; a reply to an inbound message
is delivered and logged as a reply even when it falls between unprompted
sends; an unprompted send does not re-enter as input.

### WP-S9 — Private flag: schema and capture (R25, R26, R29) — follow-on

`ALTER TABLE message_log ADD COLUMN private`, idempotent for existing
databases, plus capture from `expiresInSeconds`. The open questions under
"Private content" — the flag's route into the persistence write, and R29's
storage form — are settled before this package's red tests are written.

Red tests: migration is idempotent and preserves existing rows; a non-zero
timer sets `private=1`; a zero timer sets `0`; the flag survives a reload; a
later timer change does not rewrite earlier rows; a timer change is recorded
(unwritable until R29's storage form is chosen).

### WP-S10 — Private flag: propagation and redaction (R14, R20, R26a, R27a, R27c, R28a) — follow-on

The four paths in the table above.

Red tests: a summary over a range containing a private message is itself
private; consolidation skips private rows **and the watermark still advances**;
the jsonl log still contains private messages (R26a, R14); `corvidae redact
private` tombstones every flagged row — content wiped, rows retained — and
the FTS cascade completes.

Docs: `docs/design.md` gains the privacy-flag schema when this lands.

### WP-S11 — Registration, config, docs (R19, R20, R43)

Entry point, config parsing and validation, documentation.

Red tests: the entry point loads the plugin; daemon startup with no
`signal:` block succeeds and the plugin registers nothing; a malformed
`signal:` block raises with a clear message; `agent.yaml.example` parses.

Docs: `docs/design.md` gains a Signal subsection under Transports
(`design.md:1223`, beside CLI and IRC); `docs/configuration.md` gains the
`signal:` block and a plain statement of R19 — that E2EE content lands in
plaintext on this host, and the channel key contains a phone-derived
identifier; `docs/plugin-guide.md` gains the transport's config block beside
IRC's; `agent.yaml.example` gains the commented `signal:` section and
replaces the existing `# signal:+15551234567:` channel stub with a real one.
A new `docs/signal-ops.md` is the durable provisioning and operations
document: install (Java, signal-cli), registration (captcha token, SMS
verification, registration lock), the receive-mode requirement and the
property it guarantees, credential-directory backup and restore, and
re-registration. The procedure must survive this plan's deletion; the
pre-departure checklist below is the trip-specific instance of it.

## Deferred

- **Groups.** Out of scope. Signal's structured `mentions` (ACI plus offset
  and length) is a better trigger than IRC's nick-prefix heuristic when this
  is picked up.
- **Backlog batching.** "Take these N in, respond once" is a decision about
  whether to respond, which is what the engagement gate
  (`gate.engagement.enforce`) is for — planned for WP2.5+ in
  `plans/implementation/phase-2.md`; `docs/configuration.md` lists the key
  only as a runtime-tunables blocklist entry, with no plugin resolving it
  yet. Doing it in the transport would put message-merging policy in one
  transport and would destroy the per-message identity R35, R28a, and R26
  depend on. When the engagement gate lands, batching becomes a policy
  above it, for every transport at once.
- **Initiative policy.** What decides to send unprompted — an idle-driven
  or appraisal-driven policy enqueueing notification turns via `on_notify`
  — is not designed here. WP-S8 ships the transport side of R30–R33
  (delivery, classification, rate bound, logging); the trigger is whatever
  plugin chooses to speak, and none ships yet.
- **Attachments, in and out.** R22 and R24 bound v1 to text.
- **Reactions as acknowledgment.** A cheap "received" signal that costs no
  message. Attractive once the basics work.

## Pre-departure operational checklist

Ordered; everything here is operator-managed (screen-based supervision, per
standing practice), and all of it must be done while the spare phone's SIM
is still reachable — nothing on this list is recoverable from the road.

1. Confirm the bot number can receive an SMS (or voice call) today — Signal
   rejects many VoIP numbers.
2. On sagan: install Java 21+ and signal-cli; choose the socket path (a
   user-writable directory, or `RuntimeDirectory=` if keeping
   `/run/corvidae`).
3. Register the bot account as primary: captcha token from
   signalcaptchas.org, `register --captcha …`, receive the SMS code on the
   spare phone's SIM, `verify` — flags checked against the installed
   version first. If the number has registration lock, have the PIN or
   budget the 7-day wait; decide whether to set a PIN on the new
   registration. This deregisters Signal on the spare phone — expected.
4. Log Signal out of the spare phone and keep the phone powered off for the
   duration: a re-registration from it steals the number back mid-trip,
   unrecoverably.
5. Start `signal-cli daemon` with `--receive-mode=on-connection` (or the
   installed version's equivalent — the property is no receive and no ack
   while no JSON-RPC client is attached), under supervision that survives
   an unattended reboot of sagan; corvidae likewise.
6. Back up signal-cli's data directory — the sole credential for the bot
   account — off-host, after registration completes.
7. Land the v1 tier: WP-S1 through WP-S7 plus WP-S11. WP-S8 through WP-S10
   are follow-on, post-departure.
8. Add the `signal:` block and allowlist (your ACI) to agent.yaml on sagan;
   restart the daemon.
9. Shake down end-to-end from the traveling phone: send/receive; a
   slow-tool turn (read receipt, typing indicator refreshing past 15 s); a
   corvidae restart with messages sent while it was down and signal-cli up
   (verifies the receive mode); a signal-cli restart; confirm `sessions.db`
   rows key on `signal:<aci>` and the `channels:` override applies.
10. Optionally link Signal Desktop to the bot account for the mirror; it
    unlinks after 30 days offline but is re-linkable from sagan.
11. Confirm remote SSH access to sagan from the road (R14 observability and
    any surgery).
