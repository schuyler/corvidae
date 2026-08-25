# Signal transport — a DM channel for the daemon

**Effort:** M. **Dependencies:** none in the phase sequence — this is
orthogonal to the Phase 2 appraisal/critique arc and can land beside it.
**Normative references:** `docs/design.md` "Channel System" and "Transports";
`docs/plugin-guide.md` "Channels"; AGENTS.md for TDD and exception discipline.

**Goal:** the operator can hold a private, continuous conversation with the
daemon over Signal from an ordinary phone, and the daemon can start one.

Two work packages at the end (WP-S9, WP-S10) are not transport-local: they add
a privacy flag to `message_log` and propagate it through compaction, memory
consolidation, and redaction. They are listed last deliberately. Land the
transport first and confirm it works; the schema change is the part that can
break other things.

## Read first

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

Numbered as agreed. Tests in the work packages cite these.

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

### Data handling

- **R19** It is documented explicitly that end-to-end-encrypted content is
  persisted in plaintext on the host — session database, logs, memory — and
  that the conversation key contains a personal identifier.
- **R20** There is a supported way to delete a conversation's stored content.

### Scope boundaries for v1

Each must degrade gracefully, not merely be absent.

- **R21** Group conversations are out of scope; a group message is ignored
  without error.
- **R22** Inbound attachments do not crash anything. Accompanying text is still
  processed; the attachment is logged and dropped.
- **R23** Reactions, quotes, edits, and delete-for-everyone are ignored without
  error.
- **R24** Outbound messages are text only.

### Disappearing messages — retain and flag

- **R25** Content in a conversation with a disappearing-message timer is
  retained. Expiry does not delete it from the store.
- **R26** Such content is flagged private at rest. The flag is durable, is
  attached to the stored message, and is not re-derived later from the timer's
  current value.
- **R26a** Private content is retained in the conversation and in the jsonl log,
  unchanged. Private limits how far content travels *inside the agent*, not
  whether the operator can see it.
- **R27a** Private content is excluded from long-term memory consolidation.
- **R27b** Private content is excluded from subagent context.
- **R27c** A compaction summary covering any private message inherits the flag.
  Without this, summarization launders private content into unflagged storage
  that outlives the window.
- **R28a** Private content is bulk-selectable for redaction, without hunting
  individual message IDs.
- **R29** A change to the conversation's timer is recorded, so it is later
  possible to tell which messages fell under which policy.

### Unprompted messages

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
operator and systemd.

**signal-cli is registered as the primary device on the dedicated bot number,
not linked as a secondary device.** This is an operational requirement, not a
preference. Signal unlinks secondary devices if the primary phone does not come
online at least once every 30 days, and only the primary can re-link. With the
bot phone kept powered off, a linked deployment would need a recurring monthly
ritual and would fail silently when missed — violating R18 and, on failure,
R5. As primary, signal-cli can also add devices, so Signal Desktop can be
linked to the bot account for a live mirror of the conversation.

One-time setup, documented for the operator, not automated:

```sh
signal-cli -a +BOTNUM register            # SMS code to the bot number
signal-cli -a +BOTNUM verify CODE
signal-cli -a +BOTNUM daemon --socket /run/corvidae/signal.sock
```

If registration lock is set on the account, the PIN is required to
re-register. Record it before starting; without it, re-registration waits
seven days.

### Plugin shape

`corvidae/channels/signal.py`, `SignalPlugin(CorvidaePlugin)`, following
`IRCPlugin`:

- `depends_on = frozenset({"registry"})`, resolved with `get_dependency`.
- `on_init` reads the `signal:` config block; absent block means the plugin
  stays inert (R16).
- `on_start` opens the socket under a retry loop with the same backoff ladder
  IRC uses — 10s initial, x2, 300s cap (R15).
- `on_stop` cancels the read task and closes the socket, logging rather than
  swallowing on the way out.
- `send_message` and `send_progress` are implemented; `send_thinking` and
  `send_tool_status` are deliberately not. Nobody wants raw tool traces in a
  DM, but intermediate assistant text before a tool dispatch reads as an
  ordinary message ("let me check that") and is real liveness, so
  `send_progress` forwards to `send_message` — the same choice `IRCPlugin`
  made in 20a3b79. It is part of a reply, so the R31 unprompted rate bound
  does not apply to it.

Every implemented `send_*` hook opens with
`if not channel.matches_transport("signal"): return`.

### Channel scope and identity

Scope is the sender's **ACI UUID**, so channel IDs look like
`signal:8f2c…`. Phone numbers are not used as the stored scope: `sourceNumber`
can be absent under phone-number privacy, and `transport:scope` is the
persistence key for `sessions.db`, the jsonl logs, and memory. Keying on a
field that can vanish forks the history (R4, R10).

Config accepts `signal:+15551234567` as a friendly alias. On first contact the
alias resolves to the ACI and the resolution is recorded, so subsequent
messages land on the same channel whichever identifier Signal supplies.

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

Default-deny is right for this transport specifically, even though IRC is
open: a Signal DM is a direct line from anyone who knows the number to an
agent holding `shell`, `read_file`, and `write_file`.

### Liveness

On beginning to process an admitted message: send a read receipt and start a
typing indicator. Stop typing when the reply is sent (R39, R41). The receipt
fires at processing start rather than at arrival, because with contended local
inference a message can sit queued — a "read" marker followed by two minutes
of silence is worse than no marker (R38).

Unauthorized senders get neither (R40).

### Private content

A `private INTEGER NOT NULL DEFAULT 0` column on `message_log`, set when the
inbound envelope's `expiresInSeconds` is non-zero. Written at insert; never
recomputed from the timer's later value (R26).

The flag propagates along every path by which content leaves the conversation:

| Path | Behavior |
|---|---|
| jsonl log | unchanged — the operator's window stays whole (R26a, R14) |
| memory consolidation | private rows excluded from the range query (R27a) |
| subagent context | private messages excluded from the prompt (R27b) |
| compaction summary | summary inherits `private=1` if its range contains any (R27c) |
| `corvidae redact` | new `private` form deletes all flagged rows (R28a) |

Consolidation excludes private rows but the watermark must still advance past
them. Skipping rows without advancing stalls consolidation permanently.

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
  signal:+15551234567:
    system_prompt: "..."
```

Omitting the `signal:` block disables the transport entirely (R16).

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
7. **`_strip_internal_keys` drops `_`-prefixed keys before serialization.** An
   in-window `_private` tag needs explicit handling in the persistence write
   path, the way `_message_type` does.
8. **The unbounded queue.** `SerialQueue` uses `put_nowait` on an unbounded
   `asyncio.Queue`. A large backlog is admitted all at once; R42 accepts the
   resulting burst of turns rather than dropping messages.

## Work packages

Red tests first, per AGENTS.md. Each package names the requirements its tests
cover.

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
channel ID is byte-identical across both forms.

### WP-S4 — Authorization gate (R9, R11, R40)

`should_process_message` with a default-deny allowlist.

Red tests: an allowlisted ACI is admitted; a non-allowlisted one is rejected
and `on_message_rejected` fires; empty allowlist rejects everything; a
rejected sender receives no read receipt.

### WP-S5 — Outbound send and splitting (R2, R3, R13, R24)

`send_message`, plus moving `split_message` to `corvidae/channels/split.py`.

Red tests: existing IRC splitting tests pass against the new module location;
a 5000-character reply splits at paragraph boundaries under a 2000-char limit;
`send_message` on a non-signal channel is a no-op; a send failure is logged
and does not raise into the agent loop.

### WP-S6 — Liveness (R12, R38, R39, R41)

Read receipt and typing indicator at processing start; typing stops on send.

Red tests: the receipt is emitted at processing start, not at arrival; typing
starts before the LLM call and persists across a tool loop; typing stops when
the reply is sent; a failure to send either is logged and does not block the
reply.

### WP-S7 — Backlog timestamps (R5, R34, R35, R36)

Original send time carried into context.

Red tests: three envelopes with old, ordered timestamps produce three turns in
order; each turn's context carries its original send time; backlog still goes
through the authorization gate.

### WP-S8 — Unprompted sends (R30, R31, R32, R33)

Agent-initiated send with a minimum interval.

Red tests: an initiated send is delivered with no inbound message; a second
send inside the minimum interval is suppressed and logged; an initiated send
does not re-enter as input; the log distinguishes initiated from reply.

### WP-S9 — Private flag: schema and capture (R25, R26, R29)

`ALTER TABLE message_log ADD COLUMN private`, idempotent for existing
databases, plus capture from `expiresInSeconds`.

Red tests: migration is idempotent and preserves existing rows; a non-zero
timer sets `private=1`; a zero timer sets `0`; the flag survives a reload; a
later timer change does not rewrite earlier rows; a timer change is recorded.

### WP-S10 — Private flag: propagation and redaction (R26a, R27a, R27b, R27c, R28a)

The five paths in the table above.

Red tests: a summary over a range containing a private message is itself
private; consolidation skips private rows **and the watermark still advances**;
subagent context excludes private messages; the jsonl log still contains them;
`corvidae redact private` tombstones every flagged row and the FTS cascade
completes.

### WP-S11 — Registration, config, docs (R19, R20)

Entry point, config parsing and validation, documentation.

Red tests: the entry point loads the plugin; a malformed `signal:` block
raises with a clear message; `agent.yaml.example` parses.

Docs: `docs/design.md` gains a Signal subsection under Transports
(`design.md:1223`, beside CLI and IRC) and the privacy-flag schema; `docs/configuration.md` gains the `signal:` block and a plain
statement of R19 — that E2EE content lands in plaintext on this host, and the
channel key contains a phone-derived identifier; `docs/plugin-guide.md` gains
the transport's config block beside IRC's; `agent.yaml.example` gains the
commented `signal:` section and replaces the existing
`# signal:+15551234567:` channel stub with a real one.

## Deferred

- **Groups.** Out of scope. Signal's structured `mentions` (ACI plus offset
  and length) is a better trigger than IRC's nick-prefix heuristic when this
  is picked up.
- **Backlog batching.** "Take these N in, respond once" is a decision about
  whether to respond, which is what `gate.engagement.enforce` is for — already
  documented in `docs/configuration.md` as landing in WP2.5+. Doing it in the
  transport would put message-merging policy in one transport and would
  destroy the per-message identity R35, R28a, and R26 depend on. When the
  engagement gate lands, batching becomes a policy above it, for every
  transport at once.
- **Attachments, in and out.** R22 and R24 bound v1 to text.
- **Reactions as acknowledgment.** A cheap "received" signal that costs no
  message. Attractive once the basics work.
