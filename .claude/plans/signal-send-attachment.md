# Signal `send_attachment` (#544)

## Requirements (settled; autonomous run, minimum-scope reading)

Provenance: stated = user/issue/PRD words; inferred = my reading, vetoable.

- **R1 (stated: issue #544; PRD §3.5.9)** The agent can send one file to a
  Signal contact, with an optional caption, so that it can reply with images
  and documents.
  *Accept:* `send_attachment(channel_id, path, caption="")` on a `signal:<aci>`
  channel issues exactly one `send` RPC to that contact carrying the file and
  the caption as its message; returns a confirmation naming file and channel.
- **R2 (stated: PRD §3.5.9, issue)** A file outside the configured root is
  refused. *Accept:* the error names the path, the root, and the setting that
  moves it; no RPC is sent. A relative path is read relative to the root.
- **R3 (stated: PRD §3.5.9, issue)** A file over the size cap is refused.
  *Accept:* error names the file, its size, the cap, and `signal.max_attachment_size`;
  no RPC sent. The cap is the existing inbound one (issue: "same cap").
- **R4 (inferred, from issue "same send_attachment_root jail")** The root is
  `signal.send_attachment_root`, default `state`, mirroring Discord.
  *Accept:* default `state`; a configured value is honoured; a non-string or
  empty value fails `on_init` naming the key and the value.
- **R5 (inferred)** A target that is not a Signal channel this transport owns
  is refused with an error naming the channel. *Accept:* `discord:x` and an
  unknown id both return an error naming the id; no RPC.
- **R6 (stated: AGENTS.md error rules)** A failure to deliver (not connected,
  signal-cli error reply, unreadable file) returns an error string naming the
  file/channel and the remedy, logs with traceback, and never raises into the
  turn. *Accept:* with signal-cli answering the RPC with an error, the tool
  returns "Could not send: …" naming the file and channel.
- **R7 (inferred)** A caption longer than `signal.message_chunk_size` is not
  truncated: the first chunk rides with the file and the rest follow as text.
  *Accept:* a 2.5x-chunk caption yields the file+first chunk then the rest,
  in order, all text present.
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
- `register_tools(tool_registry)`: append `self.send_attachment` only if
  `self._socket_path is not None`.
- `async send_attachment(channel_id: str, path: str, caption: str = "") -> str`:
  validate args -> registry lookup + `self.owns(channel)` -> resolve under
  root -> `is_file()` and `stat()` size vs `_max_attachment_size` -> split
  caption with `split_message` -> `_send_rpc("send", {"recipient":
  channel.scope, "attachments": [str(resolved)], "message": first_chunk})`
  -> remaining chunks via `_deliver` -> confirmation. RPC failure: log
  warning with traceback, return "Could not send: …".
- Wire format: signal-cli's JSON-RPC man page (v0.14.7, lines 91-95) says
  `--attachment A B` is `"attachments": ["A","B"]` (camelCase, multi-value
  as array) and `send` params are `recipient`, `message`. Verified from the
  man page and `SendCommand.java`; not exercised against a live daemon.
- The path sent is the absolute resolved path: the daemon's cwd is not ours.
  The daemon must run on the same host and be able to read the file (the
  harness runs it as the corvidae user over a unix socket). *shortcut: path
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
