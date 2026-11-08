# Signal `send_attachment` (#544)

## Requirements (settled; autonomous run, minimum-scope reading)

Provenance: stated = user/issue/PRD words; inferred = my reading, vetoable.

- **R1 (stated: issue #544; PRD §3.5.9)** The agent can send one file to a
  Signal contact, with an optional caption, so that it can reply with images
  and documents.
  *Accept:* `send_signal_attachment(channel_id, path, caption="")` on a
  `signal:<aci>` channel issues exactly one `send` RPC to that contact carrying
  the file and the caption as its message (no `message` key when the caption is
  blank); returns a confirmation naming file and channel. With the Discord and
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
  *Accept:* default `state`; a configured value is honoured; a non-string or
  empty value fails `on_init` with `ValueError("signal: send_attachment_root
  must be a non-empty string, got <value!r>")`, naming the key, the rejected
  value and the accepted shape.
- **R5 (inferred)** A target that is not a Signal channel this transport owns
  is refused with an error naming the channel and saying which case it is.
  *Accept:* a `discord:x` id returns an error naming the id and saying the
  target must be a Signal channel (`signal:<id>`); a `signal:<id>` not in the
  registry returns an error naming the id and saying it is not a known Signal
  channel (list_channels shows the known ids); no RPC either way. Called
  before the registry is set (tool registered ahead of `on_start`), it returns
  an error rather than raising.
- **R6 (stated: AGENTS.md error rules)** A failure to deliver (not connected,
  signal-cli error reply, unreadable file) returns an error string naming the
  file/channel and the remedy, logs with traceback, and never raises into the
  turn. *Accept:* with signal-cli answering the RPC with an error, the tool
  returns "Could not send: …" naming the file and channel, and does so
  whichever frame fails: the file frame or a later caption chunk (a failed
  chunk must not be reported as "Sent").
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
  holds. Its docstring is the LLM-facing description: say the path is read
  relative to the root, as the code does (Discord's docstring says otherwise;
  do not copy it).
  Flow: validate args -> registry guard (None -> error) -> registry lookup +
  `self.owns(channel)` (two distinct errors, R5) -> resolve under root ->
  `is_file()` and `stat()` size vs `_max_attachment_size` -> chunks =
  non-blank pieces of `split_message(caption, max_len=_message_chunk_size)`
  -> inside one `try`: `_send_rpc("send", {"recipient": channel.scope,
  "attachments": [str(resolved)]})` plus `"message": chunks[0]` when there is
  one, then each remaining chunk by `_send_rpc("send", {"recipient":
  channel.scope, "message": chunk})` (not `_deliver`, which swallows
  failures) -> confirmation. Any exception in the `try`: log warning with
  traceback naming file and channel, return "Could not send: …".
- Wire format: signal-cli's JSON-RPC man page (v0.14.7, lines 91-95) says
  `--attachment A B` is `"attachments": ["A","B"]` (camelCase, multi-value
  as array) and `send` params are `recipient`, `message`. Verified from the
  man page and `SendCommand.java`; not exercised against a live daemon.
- The path sent is the absolute resolved path: the daemon's cwd is not ours.
  The daemon must run on the same host and be able to read the file
  (`harness/install-signal-cli.sh` and `docs/signal-ops.md` require it to run
  as the corvidae user, since the unix socket is owner-writable). *shortcut: path
  not data URI; switch to a `data:` URI if the daemon ever runs as another
  user or in a container.*
- MIME type: left to signal-cli.
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
- A bound on the `send` RPC so a wedged signal-cli cannot hold a turn (E1).
  So that a hung daemon degrades to an error. Deliberately unbounded today
  for text (#413: a timed-out send may still have been delivered).
- Sends to a `signal:+E164` alias placeholder channel, or to a contact known
  to persistence but not yet in the registry, resolve to the real channel
  (`resolve_channel`) instead of being refused or keyed on the wrong scope
  (E2, E3, U1).

## Docs to touch (R9)

`docs/signal-ops.md`: lines 282-283 (attachments are surfaced, not dropped),
286 (outbound now sends one file via `send_signal_attachment`), 287-289 (the
"can't message a different channel" claim, check against the cross-channel
send package); `docs/configuration.md` (`signal.send_attachment_root`, and
`max_attachment_size` now covers both directions); `docs/design.md` (Signal
outbound files); `docs/threat-model.md` (generalise the `discord.send_attachment`
paragraph to cover both tools, do not add a parallel one); `agent.yaml.example`.
No PRD change.

## Design review: tranche 1 triage

C1 tool-name collision: fixed (rename, R1). I1 trailing chunks via `_deliver`:
fixed (R6, design flow). I2 R4 message shape: fixed (R4). K1-K8: accepted
(R5, R7, design, docs list). E1 wedged daemon: deferred (not normal
operation; #413 keeps `send` unbounded). E2, E3, U1: deferred. E4 (sync
filesystem calls on the loop; Discord does the same), E5 (TOCTOU): dropped.
