# Signal `send_signal_attachment` (#544)

## Requirements (settled; autonomous run, minimum-scope reading)

Provenance: stated = user/issue/PRD words; inferred = my reading, vetoable.

- **R1 (stated: issue #544; PRD §3.5.9)** The agent can send one file to a
  Signal contact, with an optional caption, so that it can reply with images
  and documents.
  *Accept:* `send_signal_attachment(channel_id, path, caption="")` on a
  `signal:<aci>` channel, with a caption within `message_chunk_size`, issues
  exactly one `send` RPC to that contact carrying the file and the caption as
  its message (no `message` key when the caption is blank); returns a
  confirmation naming file and channel. With the Discord and
  Signal plugins both registered, their tools have distinct names, so a call
  for a Signal channel can only reach Signal's (the registry keeps the last
  tool of a given name; issue #544's `send_attachment` name would collide with
  Discord's, so the Signal tool is `send_signal_attachment`).
- **R2 (stated: PRD §3.5.9, issue)** A file outside the configured root is
  refused. *Accept:* the error names the path, the root, and the setting that
  moves it; no RPC is sent. A relative path is read relative to the root.
- **R3 (stated: PRD §3.5.9, issue)** A file over the size cap is refused.
  *Accept:* error names the file, its size, the cap, and `signal.max_attachment_size`;
  no RPC sent. The cap is the existing inbound one (issue: "same cap").
- **R4 (inferred, from issue "same send_attachment_root jail")** The root is
  `signal.send_attachment_root`, default `state`, mirroring Discord.
  *Accept:* default `state` when the key is absent; a configured value is
  honoured; any other value that is not a non-empty string (an empty string,
  a number, or an explicit `null`) fails `on_init` with `ValueError("signal:
  send_attachment_root must be a non-empty string, got <value!r>")`, naming
  the key, the rejected value and the accepted shape. (Discord reads a `null`
  as absent; here `null` is rejected, which is the stricter reading of the
  stated shape.)
- **R5 (inferred)** A target that is not a Signal channel this transport owns
  is refused with an error naming the channel and saying which case it is.
  *Accept:* an id whose transport prefix is not `signal` (e.g. `discord:x`,
  whether or not it is in the registry) returns an error naming the id and
  saying the target must be a Signal channel (`signal:<id>`); a `signal:<id>`
  not in the registry returns an error naming the id and saying it is not a
  known Signal channel (an allowlisted contact who has messaged since this
  start is known);
  no RPC either way. With no registry set (defensive: the tool is collected
  after `on_start`, but a guard is cheap), it returns an error, not a raise.
- **R6 (stated: AGENTS.md error rules)** A failure to deliver (not connected,
  signal-cli error reply, unreadable file) returns an error string naming the
  file/channel and the remedy, logs with traceback, and never raises into the
  turn. *Accept:* with signal-cli answering the RPC with an error, the tool
  returns a string that starts with "Could not send:" and names the file and
  the channel, whichever frame fails (the file frame, or a later caption
  chunk); it never returns the success form "Sent <file> to <channel>." for a
  failed send. The message says how far it got, so the remedy is right:
  - the file frame failed: nothing was delivered, so the whole send may be
    retried;
  - a later caption chunk failed: the file and the earlier chunks were
    delivered, so the message says "caption part i of n failed" and to send
    only the rest of the caption as text, not the file again.
  A file that cannot be stat'ed or a path that cannot be resolved (an
  `OSError` or `ValueError`) is the same kind of failure: "Could not send:"
  naming the path, no RPC. A file-frame failure that is not a lost connection
  also says the signal-cli daemon must be able to read the file (the sandbox
  note in the design); a lost connection says signal-cli is not connected
  and to retry once it is.
- **R7 (inferred)** A caption longer than `signal.message_chunk_size` is not
  truncated: the first chunk rides with the file and the rest follow as text.
  *Accept:* a 2.5x-chunk caption yields the file+first chunk then the rest,
  in order, all text present. Blank chunks are not sent: a blank caption sends
  the file with no message and no follow-up.
- **R8 (inferred)** The tool exists only when the Signal transport is
  configured (a `signal:` block is present). *Accept:* `register_tools`
  appends it then, and not for an inert plugin.
- **R9 (stated: AGENTS.md)** Docs are current: `signal-ops.md` (including the
  stale "Attachments are dropped" line, which has been wrong since #520),
  `configuration.md`, `design.md`, `threat-model.md`, `agent.yaml.example`.

## PRD comparison

- PRD §3.5.9 already specifies exactly this: one file per send, from a
  configured root, refused outside it or above a configured size, both being
  transport settings, root as defence in depth only. **No PRD edit is needed.**
- Differences, none resolved by changing the PRD:
  - PRD §3.5.9 says an agent-started send is recorded in the target channel's
    history. Issue #544 and the Discord precedent (`send_attachment`) do not
    record the file send. Following the issue and Discord: not recorded here.
    Proposed requirement (deferred): file sends join the target's history.
  - PRD §3.5.9 admits replies from a non-allowlisted person messaged first,
    and a per-destination rate limit. Both belong to the cross-channel send
    plugin, not a transport tool; this tool does neither (same as Discord).
  - PRD's Signal paragraph (§3.2) is silent on outbound attachments; PRD
    §3.5.9 covers it, so no addition there.

## Design

Mirror `DiscordPlugin.send_attachment` (`corvidae/channels/discord.py`) in
`SignalPlugin`; no new module, no kernel change.

- Config (`on_init`): `signal.send_attachment_root` (str, default `"state"`),
  stored as `Path`. Same validation shape as `max_attachment_size`.
- `def register_tools(self, tool_registry: list) -> None` (a sync hookimpl,
  as the hookspec is): append `self.send_signal_attachment` only if
  `self._socket_path is not None`.
- `async send_signal_attachment(channel_id: str, path: str, caption: str = "")
  -> str`; the method name is the tool name (`Tool.from_function` uses
  `__name__`), hence not `send_attachment`, which Discord's tool already
  holds. Only the docstring's first line reaches the model as the tool
  description (`tool_to_schema`), so that line names the Signal contact and
  the root-relative path; Discord's docstring says "relative to the working
  directory", which its code contradicts, so do not copy it. `tool_to_schema`
  marks every parameter required, so the model must pass `caption=""`;
  "optional" holds for Python callers (same as Discord).
  Flow: validate args as Discord does (`channel_id` and `path` non-empty
  strings; a `None` caption is `""`; any other non-string caption is an
  error; each failure returns an error naming the argument) -> registry guard
  (None -> error) -> transport check:
  `channel_id.partition(":")[0] != "signal"` -> not-Signal error (as
  `resolve_channel` does, so a foreign id absent from the registry still gets
  it) -> `registry.get(channel_id)` is None -> not-known error (a `signal:`
  id that is in the registry is a Signal channel by construction, so there is
  no separate `owns` check) -> inside `try/except (OSError, ValueError)`
  (logged at warning with traceback, returning "Could not send: cannot read
  <path> (<exc type>); check the path and its permissions"): resolve, then
  the root jail (`is_relative_to(root)` -> R2 error)
  before `is_file()` and `stat()`, so a missing path outside the root gets
  the R2 message; a path in the root that is not a file returns "Could not
  send: <path> is not a file; check it exists under <root>" -> size vs `_max_attachment_size` (R3) -> chunks, computed
  outside that `try`: the non-blank (`c.strip()`) pieces of
  `split_message(caption, max_len=_message_chunk_size)` -> inside
  `try/except Exception` (so `CancelledError` still propagates): the file
  frame `_send_rpc("send", {"recipient": channel.scope, "attachments":
  [str(resolved)]})` plus `"message": chunks[0]` when there is one, then each
  remaining chunk by `_send_rpc("send", {"recipient": channel.scope,
  "message": chunk})` (not `_deliver`, which swallows failures) -> the
  success confirmation. The `except Exception` handler knows whether the file
  frame has returned and which chunk index failed (a loop index; "part i of
  n" counts the chunk that rode with the file as part 1 and n is all
  non-blank chunks), logs a warning with traceback naming file and channel,
  and returns one of three strings, each starting "Could not send:":
  - file frame failed, not a lost connection: "Could not send: <file> to
    <channel> failed (<exc type>); nothing was delivered, retry, and check the
    signal-cli daemon can read <path>";
  - file frame failed on `ConnectionError`: "Could not send: signal-cli is not
    connected (<file> to <channel>); retry once it is";
  - a later chunk failed: "Could not send: <file> reached <channel> but
    caption part i of n failed (<exc type>); do not resend the file, send the
    rest of the caption as a message".
- Wire format: signal-cli's JSON-RPC man page (v0.14.7, lines 91-95) says
  `--attachment A B` is `"attachments": ["A","B"]` (camelCase, multi-value
  as array) and `send` params are `recipient`, `message`. Verified from the
  man page and `SendCommand.java`; not exercised against a live daemon.
- The path sent is the absolute resolved path: the daemon's cwd is not ours.
  The daemon must run on the same host and be able to read the file
  (`harness/install-signal-cli.sh` and `docs/signal-ops.md` require it to run
  as the corvidae user, since the unix socket is owner-writable). **Sandbox:**
  the unit `docs/signal-ops.md` tells operators to install (lines ~168-172)
  sets `PrivateTmp=yes`, `ProtectHome=yes`, `ProtectSystem=strict`, and the
  harness unit (`harness/install-signal-cli.sh` ~123-126) sets
  `PrivateTmp`/`ProtectSystem=strict` only; under the ops unit the daemon
  cannot see a root under `/home`, and under both it cannot see `/tmp`, so a
  send of such a file would fail inside signal-cli.
  So the docs phase must tell operators to give the daemon read access to
  `signal.send_attachment_root` (what the unit change is, is checked against
  the systemd documentation then, not assumed), and the error text above
  points at it. The path premise stands; the data URI is the escalation. *shortcut: path
  not data URI; switch to a `data:` URI if the daemon ever runs as another
  user or in a container.*
- MIME type: left to signal-cli.
- Test notes for Red (fake daemon, `tests/signal_fake.py` unchanged):
  `requests_matching("send")` and `fail_next_matching("send")` also match
  `sendTyping`/`sendReceipt`, so assert on `frame["method"] == "send"` and
  register the contact with `h.registry.get_or_create("signal", ACI)`, not by
  pushing an inbound envelope (that starts typing frames). To fail only the
  second `send` frame, count `send` frames in `server.on_request` and arm
  `fail_next_matching("send")` on the second (`on_request` runs before the
  auto-respond for the same frame). A lost connection: `drop_connection()`,
  wait until `h.plugin._writer is None`, then call. Pass
  `signal_overrides={"send_attachment_root": str(tmp_path)}` so the root stays
  out of the checkout. R4 rejections follow the existing `_start_plugin` +
  `pytest.raises` pattern.
- Shared helper: the root-jail/size block is duplicated between Discord and
  Signal. Not extracted now (would touch Discord; no requirement asks).

## Rejected alternatives

- Shell out to `signal-cli send --file`: the socket RPC supports it, and the
  daemon holds the data-directory lock (signal-ops.md warns against a second
  process).
- `data:` URI payload: needs read + base64 of up to 10 MB on the loop and
  bigger frames; path is enough while the daemon is local. (shortcut above.)

## Deferred (proposed requirements, tagged *proposed*)

- File sends recorded in the target channel's history (PRD §3.5.9 sentence).
  So that the channel's context shows what the agent sent there.
- Extract the root-jail/size check shared by Discord and Signal into
  `corvidae/attachments.py`. So that the two jails cannot drift.
- Data-URI send for a daemon that cannot read our filesystem.
- A bound on the `send` RPC so a wedged signal-cli degrades to an error
  sooner (E1). Today `tools.call_timeout` (600 s, `corvidae/tool.py`) bounds
  the tool call, so a wedged daemon holds a turn ten minutes, not forever.
  `send` is deliberately unbounded for text (#413: a timed-out send may still
  have been delivered).
- Inspect the `send` result for per-recipient failures (signal-cli can reply
  success with a failed recipient inside the result), for text and files
  alike. So that "Sent" is never said of a message that did not go. The
  result's JSON shape is not verified here; `_deliver` has the same blind
  spot today, so this is pre-existing and transport-wide.
- Validate `signal.message_chunk_size` at `on_init` (a value under 6, or a
  non-integer, makes `split_message` raise).
- Sends to a `signal:+E164` alias placeholder channel, or to a contact known
  to persistence but not yet in the registry, resolve to the real channel
  (`resolve_channel`) instead of being refused or keyed on the wrong scope
  (E2, E3, U1).

## Docs to touch (R9)

`docs/signal-ops.md`: lines 282-283 (attachments are surfaced, not dropped),
286 (outbound now sends one file via `send_signal_attachment`), 287-289 (the
"can't message a different channel" claim, now half-true: the cross-channel
send package queues a turn on any registered channel, and this tool sends a
file to any registered Signal channel), and a short section saying the
daemon must be able to read `signal.send_attachment_root` and what to change
in the unit (the sandbox note above); `docs/configuration.md`
(`signal.send_attachment_root`, and `max_attachment_size` now covers both
directions; note that the same value sizes the inbound read buffer, so
raising it for larger files also raises that limit); `docs/design.md` (Signal
outbound files); `docs/threat-model.md`
(generalise the `discord.send_attachment` paragraph to cover both tools, do not
add a parallel one, and fix its stale "defaults to the working directory":
the default is `state`); `agent.yaml.example`. No PRD change.

## Design review: tranche 2 triage and meta-review

The loop failed twice, so this is the meta-review. Diagnosis: converging, not
a flawed design. Round 1 found one critical (tool-name collision); round 2
found none and three important, each a gap in what the design text says for an
error path or a deployment fact, none reopening a premise. The remedy is to
state those paths and re-review, not to redesign.

I-1 (R5 unreachable for a foreign id): fixed, prefix check before lookup. I-2
(a trailing chunk failure reported as nothing sent): fixed, one flag and two
messages (R6). I-3 (documented unit sandboxes the daemon off the file):
verified in `docs/signal-ops.md`; accepted as a docs requirement and an error
remedy (R6, docs list). Cosmetics accepted: docstring line 1 and the
all-required schema, R5 parenthetical, remedy wording without `list_channels`
(an optional package), `stat`/NUL handler, E1 wording, threat-model default.
Fake-daemon test notes go to the Red phase. Edge cases: per-recipient send
results (pre-existing, transport-wide, shape unverified) and `message_chunk_size`
validation deferred; symlink wording dropped. Out of scope, filed at wrap-up:
`docs/signal-ops.md` lines ~165 and ~183-184 call `--ignore-attachments` safe
because "corvidae is going to discard" attachments, which #520 contradicts
(whether the flag breaks `getAttachment` is unverified), and `docs/design.md`
~3492-3495 is stale in the Discord section.

## Design review: tranche 3 triage

Converged: no critical, one important, the rest wording. I-1 (R6's acceptance
quoted "Could not send:" while the design's later-chunk message began "Sent"):
fixed; all three failure strings now begin "Could not send:" and R6 says so.
Cosmetics accepted and applied: R1 caption scope (C-1), jail before
`is_file()` (C-2), "part i of n" index (C-3), exception classes and logging
per branch (C-4), test notes (C-5, a Red-phase section above), the `owns` half
of the lookup dropped as unreachable (C-6), `null` root rejected, stated
(C-7), the lost-connection remedy (C-8). E-a: chunks computed outside the
filesystem `try`. E-c: the `c.strip()` filter is stated. U-1: one sentence in
the configuration docs.

## Design review: tranche 1 triage

C1 tool-name collision: fixed (rename, R1). I1 trailing chunks via `_deliver`:
fixed (R6, design flow). I2 R4 message shape: fixed (R4). K1-K8: accepted
(R5, R7, design, docs list). E1 wedged daemon: deferred (not normal
operation; #413 keeps `send` unbounded). E2, E3, U1: deferred. E4 (sync
filesystem calls on the loop; Discord does the same), E5 (TOCTOU): dropped.
