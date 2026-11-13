# Threat model

What can go wrong when corvidae reads text it didn't write, and what the
code does about it. Written from a security review, conducted 2026-10-22,
of the tree at commit `cca7604`, then updated for the documentation fixes
that landed alongside it. It is a snapshot, not a policy: when the code
moves, this document is wrong until someone fixes it.

## The short version

Corvidae is a personal agent daemon for a single machine, and the design
says so. Within that model it does several things right: Signal is
default-deny with silent refusal, retrieved context is framed as data, tool
`_ctx` is hidden from the model, SQL is parameterized, signal-cli is pinned
and signature-verified, and the docs are candid about plaintext at rest.
The Discord and Zulip transports follow Signal's model: default-deny, silent
refusal, and outbound mentions disabled. The web transport binds loopback by
default, needs a token that can't be turned off, and renders model output
through DOMPurify under a strict CSP.

Memory spans every channel: the PRD makes the agent one identity across
channels, with no compartments and no cross-channel allowlist, and retrieval,
`search_memory`, `recall_raw`, near-duplicate merge, the provenance critic's
evidence search and the heartbeat's review all reach the whole memory (#168).
Finding 15 records what that costs, including a gap the operator has
accepted.

The weak spot is that the model's authority is unbounded and the model's
inputs are not. Simon Willison's "lethal trifecta" is three things in one
agent: access to private data, exposure to untrusted content, and a way to
send data out. Corvidae has all three, with nothing structural separating
them:

- **Private data.** `read_file` reaches `sessions.db`, the signal-cli data
  directory (which *is* the Signal credential), and `~/.ssh`. `shell`
  reaches everything else.
- **Untrusted content.** Every IRC channel message and private message,
  every Discord message that mentions the bot from an allowlisted user in
  a channel other people can read (with `discord.require_mention: false`,
  or a `discord.channels` entry setting it false for that channel, every
  message that user posts there),
  every Zulip message that mentions the bot from an allowlisted
  user in a channel other people can read (with `zulip.require_mention:
  false`, or a `zulip.channels` entry, every message that user posts there),
  every webhook payload,
  every web page `web_fetch` returns, every DuckDuckGo snippet, every MCP
  tool description and result, and every memory distilled from any of the
  above.
- **Exfiltration.** `web_fetch` will GET any URL with no scheme or host
  restriction. `shell` has the network. The PRD's send tool adds people
  as destinations (finding 15).

The provenance critic and the CONTEXT framing are soft mitigations. They
depend on the model behaving, which is the one thing you cannot assume
under prompt injection or with a poisoned model. The provenance check is
model-driven at both ends (#155). Stage-2 appraisal, reading the exchange,
decides which claims to check, so injected text can talk it out of listing
a claim. The provenance critic is shown excerpts of earlier raw-log
messages, from every channel, matching retrieval (finding 15, #168), so
text from one channel can reach the critic's prompt while it judges another's
reply. They are quoted, untrusted text in the critic's prompt, framed as data, and
the critic's objections re-enter the window as CONTEXT (provenance objections also wake a turn whose stub tells the agent it may reply `NO_REPLY` to send nothing; `SilencePlugin`'s veto of a final containing `NO_REPLY` covers every turn origin while it is loaded; otherwise only critique's own exact-match veto of critique-origin turns applies). The number of
claims (5) and excerpts (5 per claim, 500 characters each) is capped, so
one reply can't fan out into an unbounded critic prompt. Leaving sandboxing out
of the MVP was a reasonable call, but it matters here: the attacker is not
another tenant, it is whoever can post in a channel the bot sits in.

## Assets and adversaries

Assets, roughly in order of how bad it is to lose them:

1. The signal-cli data directory. There is no recovery; whoever has it is
   the bot's phone number. The Discord bot token (`discord.token` or
   `DISCORD_TOKEN`) is the same kind of asset: whoever holds it is the bot,
   and it is read from `agent.yaml` or the environment, both of which
   `read_file` and `shell` can reach. It is revocable (reset it in the
   Developer Portal), which the Signal identity is not. The web token
   (`web.token` or `CORVIDAE_WEB_TOKEN`, or the generated one in the
   daemon log) and its session cookie are a third: whoever holds either
   one drives the agent, tools and all. Changing the token revokes it. The
   Zulip bot's API key (`zulip.api_key` or `ZULIP_API_KEY`) is the same
   kind of asset as the Discord token: whoever holds it is the bot, it sits
   where `read_file` and `shell` can reach it, and it is revocable (regenerate
   it on the bot's Zulip settings page). It is never logged.
   An MCP server's bearer token, passed through `${VAR}` in
   `mcp.servers.<name>.headers`, is a fourth: it lives in the daemon's
   environment, which `shell` can read (`env`, `/proc/self/environ`), and
   grants whatever that server grants. A credential embedded in a server's
   `url` instead — a query-string token or a `user:pass@` userinfo — stays
   out of the MCP connect-failure and retry warnings and the MCP
   config-error messages: the two warnings name only the endpoint's scheme,
   host, port and path, and for such a server they carry the exception's
   text with every URL in it reduced the same way instead of its traceback
   (an HTTP client quotes the URL in its errors); a config-error skip
   describes a rejected value likewise (#386). Library loggers — the MCP
   SDK's `mcp.*` and the HTTP client's `httpx2` — are not filtered and can
   write the URL whole (at the default WARNING level only `mcp.*` errors
   do), so prefer a `${VAR}` header to a credential in the URL.
2. The host account corvidae runs as, and anything it can read or run.
3. `sessions.db`, the JSONL logs, and the memory tables: plaintext copies
   of every conversation, including end-to-end-encrypted Signal ones.
   Discord conversations are not end-to-end encrypted to begin with:
   Discord the company can read them whatever corvidae does, and so can the
   Zulip server's operator.
   The daemon log belongs here too: an anomalous model response (a reply
   cut off mid-reasoning, or one that reopens reasoning after its answer)
   can be logged at warning level with its text, and that text can carry
   anything the model held in context, including secrets a tool read
   (#348). Treat the daemon log as you treat `sessions.db`.
4. `agent.yaml`: operator intent, hot-reloaded within seconds of a write
   (default poll 2s).
5. The LLM server, if it is shared with anything else, and each model
   pool's warm context heads (PRD principle 16): the slot time and the
   warm prefixes the operator's own turns need, and money on a metered
   endpoint, which corvidae doesn't cap.

Adversaries:

- **Anyone in an IRC channel the bot has joined**, or anyone who can PM
  it. They are refused unless on `irc.allow`, but IRC nicks are
  unauthenticated on networks without services accounts.
- **Anyone in a Discord server the bot has been invited to**, who can
  @-mention it, and anyone Discord lets DM it.
  The allowlist refuses them, silently; see finding 5. A message that does not
  mention the bot in a mention-only channel is dropped before the kernel sees
  it, and the DEBUG line recording the drop carries the sender's Discord id —
  for a stranger as well as an allowlisted user, since a name is what tells a
  misconfigured allowlist from a quiet channel. The prompt's `addressed: yes`
  comes from Discord's structured mention list, not the message text, so typing
  "@corvidae" cannot set it; the mention token is still stripped from the text.
- **Anyone in a Zulip organisation the bot belongs to**, who can @-mention it
  in a channel it is subscribed to, and anyone Zulip lets DM it. The allowlist
  refuses them silently, as for Discord. A message that does not mention the
  bot in a mention-only channel is dropped before the kernel sees it, and the
  DEBUG line recording the drop names the sender's user id. The prompt's
  `addressed: yes` comes from the server's `mentioned` flag, not the message
  text (the flag is also set by a user-group mention, so a group the bot is in
  can address it). Several strings an attacker chooses reach the prompt header
  as transport-supplied names, never as instructions: the **topic** (anyone who
  can post can title it), the **channel name**, the sender's display name and,
  for a group DM, the other members' names. All are framed as data like every
  other name (PRD §3.3.11).
- **Anyone who can reach the web listener**, if it's bound beyond
  loopback, and anyone who gets the token or cookie; also any web page
  open in the operator's browser, which can try cross-site requests. See
  finding 9.
- **Anyone who can reach the webhook listener**, to the degree each endpoint's
  auth allows (none, for `token: null`). See finding 16.
- **Anyone who controls a page the bot fetches**, or a search result it
  reads.
- **An MCP server**, via its tool descriptions and results.
- **Anyone who can read a room the bot speaks in** without being
  allowlisted: a Discord guild channel, a public Zulip channel, a public IRC channel. They can't
  drive the agent, but they read its replies, and with one identity
  across channels those replies can carry what it knows from elsewhere
  (finding 15).
- **The model itself**, if the weights are poisoned or the model is simply
  wrong in a way that happens to look like an attack. Every finding below
  applies equally to "the model was tricked" and "the model is hostile".

An operator with shell on the host is out of scope. They already have
everything.

## Findings

Most severe first. File references are to the reviewed commit.

### 1. `set_settings` could overwrite the request body (fixed, #206)

At the reviewed commit, the one hard rule `RuntimeSettingsPlugin` enforced
was that `system_prompt` is immutable. It checked the literal key
(`tools/settings.py`). But every override that was not a framework key and
did not contain a dot was forwarded into the LLM request body (`agent.py`,
step 7), and `LLMClient.chat` applies it with `payload.update(extra_body)`
*after* setting `model` and `messages` (`llm.py`). So:

```python
set_settings({"messages": [{"role": "system", "content": "..."}]})
```

replaced the entire conversation, operator prompt included, on every
subsequent turn. `set_settings({"model": "..."})` repointed a shared server.
`tools` and `tool_choice` landed the same way, as did `stream`, which only
breaks the response parser. Verified with a fake session: the outgoing
payload carried the override's model and messages. The transport metadata
keys `cwd` and `tool_backend` were writable too.

Fixed with an allowlist in code. `set_settings` accepts an undotted key
only if it is an inference parameter (`INFERENCE_KEYS` in `channel.py`:
`temperature`, `top_p`, `top_k`, `frequency_penalty`, `presence_penalty`,
`max_tokens`, `seed`) or one of the framework keys `max_turns`,
`max_context_tokens` and `reasoning_history`. Any other undotted key
refuses the whole call before anything changes, a `null` (clear)
included, and the error names the rejected keys and the accepted set.
Independently, `Agent._llm_overrides` forwards only `INFERENCE_KEYS` into
the request body, so an override that reaches `runtime_overrides` some
other way still stays out of it. `agent.immutable_settings` narrows the
allowlist further; it is no longer the only thing between the model and
`messages`. Dotted plugin tunables are still open by default; that is
finding 2. If you don't need agent self-tuning at all,
`tools.disabled: [set_settings]` closes finding 2 outright too.

The allowlist does three jobs, and the PRD keeps two of them. Keeping
`messages`, `tools`, `tool_choice` and `stream` out of the request body
stays: an override there replaces what the agent loop built on every turn.
Keeping the transport metadata keys `cwd` and `tool_backend` out of the
agent's reach stays too. Keeping the agent away from every other model
parameter (`model`, `logprobs`, or any key outside the seven in
`INFERENCE_KEYS`) does not stay: PRD §3.5.6 makes them settable where the
operator allows, and finding 13 records the cost (#292).

### 2. The model can turn off its own gates

Every `resolve_tunable` key is writable per channel unless it appears in
`agent.immutable_settings`. That includes `critique.provenance.enabled`,
the sampling rate, and — worse — `critique.prompt.<lens>` and
`appraisal.stage2_prompt`, which are the critic's and appraiser's system
prompts. The configuration doc acknowledges self-tuning as a deliberate
trade and recommends a blocklist. At the reviewed commit two of its four
recommended keys (`gate.engagement.enforce`, `gate.send.enforce`) did not
exist anywhere in the code, and the prompt keys were not mentioned; the
doc now lists the real set.

The output gate (`GatePlugin`, #131) added four more keys of the same kind.
`gate.engagement.enforce` with a high `gate.engagement.threshold` makes the
agent ignore every message on the channel; `gate.send.enforce: false` or
`gate.send.min_salience: 0` turns off the decide gate; `gate.send.allow_progress`
and `allow_thinking` hide intermediate output. All are writable through
`set_settings` by default. A prompt-injected "just stop answering" now has a
mechanism, so on channels with untrusted participants put the `enforce` and
threshold keys in `agent.immutable_settings` with the others. The defaults
can't silence a reply to a person: the decide gate never withholds a
user-origin final, and the engagement gate ships in shadow mode.

For a poisoned-model scenario this is the finding that matters most after
the first one: the only runtime check on confabulation can be switched off
by the thing being checked.

The fork instructions added for #51 (`appraisal.stage2_fork_prompt`,
`critique.fork_prompt.<lens>`, `memory.consolidation_fork_prompt`,
`agent.compaction_fork_prompt`) and `agent.forks.*` are read from operator
config only, so `set_settings` can't rewrite them. That covers only calls
that actually fork. A fork miss (forks off for the stage, the exchange has
left the window, a refused or unparseable fork) skips stage-2 appraisal, the
stylistic critique lenses and the provenance check with a warning (#320, PRD
§3.8.5), so the standalone prompts `appraisal.stage2_prompt` and
`critique.prompt.<lens>` are no longer reachable from any live stage. They
stay in the tunable list above, and no stage reads them. Compaction on a
fork miss, and consolidation of a range that has left the window, don't use
a standalone prompt either: they send the agent's rendered view (system
prompt, window, fork instruction) on the stage's profile with no tools (PRD
§3.8; #380, #388), so the fork instructions above are the only prompts that
steer them. The old `agent.compaction_prompt` and
`memory.consolidation_prompt` are gone.

The tunability itself is not a defect to engineer away: the capacity to
tune and rewrite its own gates is part of a
design goal (PRD principle 15). This finding records its cost, and finding
13 records the cost of each kind of self-modification.

### 3. Untrusted text is promoted to the `system` role

Four places:

- Funnel-admitted CONTEXT (memories distilled from channel participants,
  critique objections) is appended as `role: system` (`funnel.py`).
- Notifications without a `tool_call_id` are wrapped as `role: system` in
  `Agent._build_conversation_message`. Scheduler firings are among them:
  a reminder's text is whatever the model wrote into `remind_me`, possibly
  at an untrusted speaker's request, and it comes back later as a system
  row. See finding 7.
- The not-sent marker the Agent appends after a `should_send_response`
  veto (#131) is a `role: system` CONTEXT row. It is core-written, not
  framed by the funnel, and quotes at most 40 characters of the agent's own
  progress text (the final and thinking markers quote nothing). If that
  text echoed an injection, those 40 characters are promoted to the system
  role. `mid_conversation_system_role: user` covers it like the others.
- A tool result whose call compaction dropped (#79) is rendered by
  `ContextWindow.build_prompt` as a framed `role: system` message, since a
  `tool` message with no call breaks strict chat templates. The result can
  be untrusted (a `web_fetch` page, an MCP reply); an orphaned result of a
  declared tool is fenced as one (finding 6). Unlike the two above it
  is render-only: the stored row stays `role: tool`.

The "retrieved data, not instructions" wrapper is the right instinct. But
message role is a stronger signal to an instruction-tuned model than a
textual disclaimer, and a chat template may render system turns with
elevated trust. Injected text in a public channel can be laundered into a
system-role CONTEXT row. The row itself survives restarts, and the memory
it came from outlives compaction and gets re-admitted on later turns.

Memory lines carry their record id, and the funnel dedupes them on
`memory <id>)` within memory's own CONTEXT (#624, #632). Because a summary
is distilled from participants' text, a summary that quotes another
record's key (`memory 5)`) makes record 5 look present while that line
stays in the window, so record 5 is skipped as a duplicate. A raw record
(#626) is the verbatim transcript, so one where the agent quoted a recalled
line carries the key word for word. The effect is
that one recollection is suppressed for a window; nothing is injected.
Line-anchored keys would close it and are not built. Another source's
CONTEXT can't do this, since keys match only their own source's rows.
`funnel.window_budgets.<source>` bounds how much CONTEXT one source can
hold in a window, which also bounds how much laundered text that source can
put there between compactions. The flip side: once low-value or laundered
lines fill the cap, that source recalls nothing more until compaction.

Fix shape: deliver CONTEXT as a `user` turn (or a tool result) with the
same framing, and reserve `system` for text the operator wrote.

Available as an opt-in: `agent.mid_conversation_system_role: user` sends
every system message after the operator's system prompt (all four places
above) as `user`, with the framing unchanged. It is applied at prompt
build, so the stored rows keep `role: system`. The default in the tree is
still `system`, because the evals were tuned against it; PRD §3.3.7 makes
`user` the default once the harness `memory_role` probe passes. One
behavioural caveat: CONTEXT is appended after the person's message, so
with `user` the framed block becomes the last user turn the model sees.
Check in `harness/` that the model still answers the person before you
make it the default.

Compaction is the adjacent problem. Summaries are stored as
`role: assistant` (`compaction.py`), so the model reads whatever was
injected as something it said itself, and that persists across restarts
too. Every summary is now written from the agent's rendered view, by a
fork or, on a miss, as a call on the `compaction` stage's profile (PRD §3.3.3, §3.8). It
is genuinely the agent's own words about what it read, written under its
own persona, not a neutral paraphrase stored under `role: assistant`: the
earlier standalone summarizer, which made the neutral paraphrase, is gone.
The only `user`-role summary is the one for a retained tail with no user
message (#56). A `user`-role summary with data framing, the obvious
hardening, is still an experiment worth running.

### Provenance header and fence (#548)

The kernel renders a code-built `[metadata: ...]` header before a person's
message and a header plus a nonce-fenced payload (`[payload <nonce> begins]`
... `[payload <nonce> ends]`) for notifications, cross-channel tool
results and results of tools declared to return untrusted output ([design.md](design.md#provenance-header-and-fence)).

What it guarantees: the header's wording and the fence come from kernel
code. Names, ids, annotation values and payload are sanitized at render
time, so text a plugin or tool supplied inside a fenced payload cannot
close the fence without knowing the nonce, and a plugin cannot change what
the header says. The nonce differs per stored row and is not
derivable from the text.

What it does not guarantee:

- It is a cue the model is trained to weigh, not an enforced boundary; a
  model can still follow instructions inside a fence.
- `provenance.key` sits in the base directory.
  `read_file` has no root and `shell` reads any path, so a model induced to
  read the key and send it out through a tool lets an attacker forge fences
  for later rows. Mode 0600 keeps other users out, not the agent. Keep the
  file out of any directory the agent is meant to read, and delete it to
  rotate (nonces change on the next start).
- A person's own message is unfenced: it is rendered as typed, so a person
  can type header-shaped text. It is the instruction channel; only the header
  before it is kernel-built and fixes who the kernel saw send it.

### 4. Tools are unsandboxed and the exfil path is open

`shell` runs as the daemon user with no allowlist, no fixed cwd, and no
confirmation (`tools/shell.py`). `task_pipeline` is `shell` with a
120-second timeout per step (and the default 600 s per-call bound,
`tools.call_timeout`), and it calls the shell implementation directly, so
`tools.disabled` couples the two: naming either removes both, globally
and per channel (#149). `write_file` writes anywhere
the daemon user can, creating parent directories, including over
`agent.yaml`, which `ConfigWatcherPlugin` reloads within two seconds. That
lets the model clear `immutable_settings` immediately. It can also repoint
any profile's `base_url`, add an MCP server whose `command` is anything,
disable plugins, or re-enable globally disabled tools; those keys are not
hot-reloaded (nothing under `llm:` is), so they wait for the next restart,
which makes the attack persistent and quiet rather than immediate. A channel's own
`channels.<id>.tools.disabled` *is* hot-reloaded. Removing an entry from it
widens that channel's tools at its next compaction, with no restart, so on
a channel that still has `write_file` the per-channel list is only as
strong as the protection on `agent.yaml`. The same goes for a
transport-wide entry (`channels."discord:*"`, #203): all its keys reload
live onto every matching channel, so one edit widens a whole transport.
`write_file` can also
rewrite any file named in a list-form `agent.system_prompt`, which is a
persistent persona change that never touches `agent.yaml` and takes effect
on the next restart or new channel. `web_fetch` has
no scheme or host check, follows redirects, and will happily GET
`http://attacker/?d=<secret>` or `http://169.254.169.254/`.

All of this is expected for a personal agent on your own box. The Buster
harness config (`harness/buster.yaml.in`) shows the pattern for narrowing
it: no subagent, no MCP, `shell` disabled. It is not a closed profile,
though: it leaves `write_file` on, which is the `agent.yaml` path
(`task_pipeline`, the shell by another name, now goes with `shell`). A
genuinely narrowed profile needs `tools.disabled: [shell, write_file]` at
least. The same list under `channels.<id>.tools.disabled` narrows a single
channel. The rest of the daemon keeps its tools, the channel's schemas
never show the disabled ones, dispatch refuses them even when the model
names one anyway, and a `subagent` launched from that channel inherits
the restriction. Cheap wins beyond that: make the dangerous tools opt-in rather than
opt-out, deny private address ranges and non-HTTP schemes in `web_fetch`,
root the file tools in a directory, and have `agent.yaml` and the prompt
files owned by another user and read-only to the daemon (if the daemon
user owns them, `shell` can `chmod` them back).

The first three of those are PRD §3.5.3 item 3, which the tree does not yet
meet: the dangerous tools are opt-out, the file tools are unrooted, and
`web_fetch` has no scheme or address check. That gap is tracked in #444.

### 5. IRC had no sender authorization (mitigated, #77)

IRC used to accept every channel message and every private message from
anyone, and a PM from a stranger auto-created a persistent channel keyed on
their nick: a private conversation with full tool access for anyone who
could reach the server, whose history the next holder of that nick
inherited. Memory records from any such channel were retrievable on every
other (finding 15), so a stranger's PM could also plant text for later
retrieval everywhere.

Now IRC is default-deny like Signal. `irc.allow` lists the senders who may
drive the agent; with none, everyone is refused, in channels and PMs. A
refusal is one INFO line and no reply. The allowlist is checked before the
channel is resolved, so a refused sender adds nothing to the registry and
leaves no `message_log` rows or memory, and a stranger can no longer
multiply channels (which also bounds finding 7). Where the network reports
services accounts, an entry matches the account, not the nick, so using
someone's nick is not enough. Residual risks:

- **Networks without accounts.** There an entry matches the nick, and
  nicks are unauthenticated: anyone who takes an allowlisted nick is that
  user. Use a network with services and account tags for anything that
  matters.
- **Account matching trusts the server, and without `account-tag` pydle's
  state.** With `account-tag` the account comes only from that message's own
  tag, which the network sets (no tag means not logged in, no nick fallback);
  pydle's user record is never consulted. Without it, matching relies on
  pydle's tracked state (`account-notify`, `extended-join`, WHOIS), which
  only knows users it has seen, can lag a logout, and is shared across lines.
  A sender who shares no channel with the bot (a PM-only user, or a
  non-member writing to a channel without `+n`) never has that record kept
  current (no logout reaches it), so such messages are refused outright
  without `account-tag` (#264).
  The residual risk is logout lag and WHOIS races for senders who do share a
  channel. An ircd or services that mis-tags, or a message delivered
  between a logout and its notification or a nick change racing a WHOIS on a
  network without `account-tag`, defeats it. Prefer networks that send
  `account-tag`.
- **Allowed users in a shared channel** are an injection path in the same
  sense as for Discord (below): the channel's history is one conversation,
  and what an allowed user pastes in is untrusted content.
- **Upgrade.** A deployment that used IRC goes silent until `irc.allow` is
  set; startup logs a WARNING saying so.

IRC, Signal, Discord and Zulip all check the allowlist before resolving the
channel, so a refused sender leaves no registry entry (#77, #103, #274, #640).

Partial mitigation (#149): `channels.<id>.tools.disabled` takes the
dangerous tools off an IRC channel the operator configures, for example
`channels: {"irc:#lex": {tools: {disabled: [shell, write_file, read_file,
subagent]}}}`, while the operator's own channels keep them. Dispatch
enforces the list, not just the schemas. It doesn't cover PM channels:
a stranger's PM channel is keyed on a nick nobody configured, so no exact
entry reaches it. A transport-wide `irc:*` entry does (#203), since it
covers channels created at runtime, but it covers every IRC channel and an
exact entry can't re-enable a tool it disables (the lists are unioned), so
it can't spare the operator's own IRC channels. Nicks share no prefix that
would single out PMs. PMs are now gated by `irc.allow`, so only allowlisted
senders reach a PM channel; `irc:*` or the global `tools.disabled` is still
the control for narrowing what those channels can do.

Discord is built like Signal, not like IRC. The transport's
`should_process_message` is a default-deny allowlist of numeric user ids; an
empty list refuses everyone, and a refusal is one INFO log line with no
reply and no typing indicator. In a server, though, anyone who can see a
channel can @-mention the bot, and what stops them is only that allowlist.
Three more things to know:

- **Allowed users in a shared channel are an injection path.** An
  allowlisted user is trusted to drive the agent, but a server channel (or
  thread) has other readers, and the channel's history is one shared
  conversation. What an allowed user pastes in, including text they copied
  from someone else, is untrusted content with the same weight as any
  other. The bot answers only when mentioned, but it records only those
  messages, so the model never sees unmentioned chatter. (With
  `discord.require_mention: false`, or a `discord.channels` entry for that
  channel, it sees every message from allowlisted users there, and still none
  from anyone else.) Prefer DMs, or
  channels where everyone present is someone you would add to `allow`.
- **Outbound mentions are disabled.** Model output can contain anything an
  attacker got into context. Every send passes `AllowedMentions.none()`,
  so `@everyone`, role mentions, and user mentions in a reply render as
  text and ping nobody. Without it a prompt-injected reply could ping a
  whole server. This closes the ping, not the content: the reply is still
  visible to everyone in the channel, including text the model was tricked
  into leaking.
- **No privileged intent.** The bot does not request the Message Content
  intent, so it cannot read server messages that do not mention it, even if
  it is in the channel. That narrows what the
  transport can ever be made to see, though it is a Discord-side limit
  and not a corvidae one. `discord.require_mention: false` requests the
  intent and removes it.

Zulip is built like Discord. The transport's `should_process_message` is a
default-deny allowlist of numeric user ids (never emails or names, which
change); an empty list refuses everyone, and a refusal is one INFO line with no
reply and no typing notice. A public channel's readers can all @-mention the
bot, and what stops them is only that allowlist. Four more things to know:

- **Allowed users in a shared channel are an injection path**, for the reason
  given for Discord: the channel's history is one conversation, and what an
  allowed user pastes in is untrusted content. In Zulip that history spans
  every topic of the channel, because a topic is a name people can rename and
  so is not part of the channel's id (PRD principle 10): text from one topic
  is in the window while the agent answers another. The agent answers only when
  mentioned (directly, or through a user group it belongs to) and records only
  those messages; with `zulip.require_mention: false`, globally or in a
  `zulip.channels` entry, it records all of an allowed user's messages there.
- **Outbound mentions are defused.** Zulip has no allowed-mentions switch. A
  reply containing `@**all**` would page the whole channel, so every send puts
  a zero-width space between `@` and `*`, which stops Zulip parsing a mention.
  This closes the ping, not the content: the reply is still visible to
  everyone who can read the channel, including anything the model was tricked
  into leaking. Silent mentions (`@_**Name**`) pass through because they
  notify no one.
- **The reply topic comes only from allowed senders.** Where the agent answers
  is the topic of the newest allowed message, so a stranger who posts under
  another topic cannot move the reply or relabel the channel to the model.
- **Plain `http://`** to a non-loopback `zulip.site` sends the API key and every
  message in the clear; startup logs a WARNING naming the site.

### 6. MCP servers are trusted peers

An MCP server's tool descriptions go into the prompt as tool schemas and
its results come back as tool results. The descriptions are not framed, and
schema sanitization is shallow by design. An MCP server is therefore a
supply-chain vector for tool-description injection. Results are framed: the
kernel declares every MCP tool as returning untrusted output (#557), so each
result reaches the model in the nonce fence under a header the code builds,
naming the tool and call id and labelling the content untrusted. That marks
data as data; it does not make the model ignore it. `transform_tool_result`
remains a pass-through hook.

**What is fenced and what is not.** Fenced as untrusted outside content: the
results of `read_file` (spill files read back included), `web_fetch`,
`web_search` and every MCP tool. A tool declares this for itself
(`returns_untrusted`, read when the call is dispatched and stored on the
result), so a result of a tool no plugin declares is unfenced unless it acts on
another channel. Result rows stored before #557 carry no flag and stay
unfenced on reload. Unfenced today:
`shell` and `task_pipeline` output, including a `cat` of a spill file (#580);
`recall_raw` and `search_memory` replays of earlier content, which can carry
text that was untrusted when it first arrived; the results inside a
subagent's own loop, which reach the subagent's model with no frame; and the
subagent's final report to the main model, which can repeat what the
subagent read. The header names the spill source for a read-back when the log
still holds the row that cut it (#576), cleaned like any header value, and
fenced rows are costed by the estimator without their framing, so a long run
of them can undercount tokens against the hard ceiling (#577).

The HTTP transports (`streamable_http`, `sse`) open an outbound connection
to whatever `url` names and send the configured `headers`, bearer token
included, on every request. The SDK follows a redirect only within the
endpoint's origin, so the token isn't handed to another host that way, but
a plain `http://` URL sends it in the clear: use `https://` for anything
off the local machine. Keeping the token in the environment keeps it out of
`agent.yaml`, which matters if that file is shared or checked in. It does
not keep it from the model: `shell` can read the daemon's environment, and
`write_file` can add an MCP server pointed at an attacker's URL with the
same `${VAR}` header (that takes effect at the next restart, like any
other `mcp` change).

### 7. The scheduler re-enters text later

`SchedulerPlugin` (#59) adds a delay to finding 3. On IRC anyone can ask
the bot to "remind me in 6 hours to …"; the model calls `remind_me`, and
six hours later that text re-enters the same channel as a notification
with the kernel header, stored as `role: system`. By then the exchange that set it
may have been compacted away, so the model sees an instruction-shaped
string with little memory of who wanted it. What bounds it:

- **Framing.** The firing text says a reminder "you set" is due and quotes
  the stored text as `Reminder text: «…»`, ending "Decide whether and how
  to act on it." Any `«` or `»` inside the text is replaced, so it cannot
  close the quote and continue as framing. Rendering follows
  `agent.mid_conversation_system_role`, so `user` keeps it out of the
  system role as for every other notification. The framing is a soft
  mitigation in the same sense as the funnel's.
- **Scoping.** The three tools act on `_ctx.channel` and take no channel
  argument. A reminder fires on the channel it was set on and nowhere else,
  so a stranger on IRC cannot use one to make the bot message a Signal
  contact. PRD §3.9.7 now lets a reminder target any channel, so this
  scoping is slated to go; finding 15 covers what replaces it.
  `cancel_reminder` refuses operator jobs and treats another channel's row
  as not found; `list_reminders` shows only the calling channel's rows. PRD
  §3.9.7 makes both span every channel.
- **Volume.** `scheduler.max_reminders_per_channel` (default 50) caps
  pending reminders per channel, and text is capped at 2000 characters.
  The cap is per channel. Finding 5 used to let a stranger open new IRC PM
  channels at will; with `irc.allow` (#77) only allowlisted senders can,
  so the number of channels, and with it the total, is bounded by who the
  operator allowed. Each firing costs a main-model turn. A reminder fires
  once, so a burst is bounded, but 50 per channel across several allowed
  senders is still load.

`scheduler.jobs` in `agent.yaml` is the operator-only surface. A job can
target any channel id, including a Signal contact, and recur, and its
prompt is framed as the operator's. It is hot-reloaded, so it joins the
`write_file` → `agent.yaml` path in finding 4. A model that can write
`agent.yaml` can now schedule messages to any channel within seconds, not
just at the next restart. The mitigation is the one finding 4 already
gives: keep `agent.yaml` unwritable by the daemon user.

The YAML jobs file (scheduler.jobs_file) adds a second operator surface
for scheduler jobs. The CRUD tools (add_job, update_job, remove_job,
list_jobs) are registered on every channel and write the file as the
daemon user. Unlike remind_me (which is scoped to its own channel and
capped by max_reminders_per_channel), the CRUD tools accept any channel
id and origin with no per-channel cap. A model on a channel with
untrusted speakers can therefore schedule messages to any channel in the
registry, including Signal contacts, and the jobs persist across restarts.
The file-permission mitigation (keep the file unwritable by the daemon
user) would disable the CRUD tools, so it is not a viable mitigation for
the tool path. The real exposure is the same as for any cross-channel
send tool (finding 15): registry membership is the whole gate. Bounding
the tools (per-channel cap, operator opt-in) is a follow-up.

`discord.send_attachment` (issue #521) and `send_signal_attachment` (issue
#544) are the first tools that send a *file* to a channel the model chooses
(the cross-channel `notify_other_channel` of #234 also takes a model-chosen channel,
but it hands an instruction to the agent's own context, not a message to the
person): `channel_id` is an argument, and the file comes from under
`discord.send_attachment_root` or `signal.send_attachment_root`, each `state`
by default, relative to the working directory. `state/` holds what corvidae
writes there, among other things downloaded attachments and spilled
notification content; an operator who widens a root to the working directory
puts `agent.yaml` and its tokens in reach. Text from a fetched page or an MCP
result can therefore turn into a file upload on any channel already in the
registry, and the file's contents never appear in the transcript. Neither tool
checks the inbound allowlist: registry membership is the whole gate. For
Signal a channel is in the registry, for example, when an allowlisted sender has written
since the last start, when it is listed under `channels:`, or when a reminder
or scheduler job has resolved it since the start, so the reach is the contacts
the operator or the allowlist has already put there, and (as PRD §3.5.9 says)
nothing narrows it further. The Signal tool reads the file itself and sends signal-cli
its bytes as a data URI, so the daemon never sees the path. Two soft mitigations today: the root is
configurable, and the upload names the file. Neither is a boundary, because
`shell` and `read_file` are unjailed already; the meaningful control is the
allowlist of who can reach a turn at all. Taking the calling channel unless a
turn explicitly overrides it is the structural fix and is not done.

`discord.start_thread` (#593) is the first transport tool that creates the
place a message will land in rather than choosing one: it starts a public
thread on a channel already in the registry, registers the new thread as a
channel, and may post the thread's first message. The reach is the same as
`discord.send_attachment` — registry membership, no inbound-allowlist check —
with one addition worth naming: a thread is a new scope, so it is a new
`transport:scope` key in `sessions.db`, the jsonl logs and memory. Text from a
fetched page can therefore make the agent create conversation contexts, not
only messages, and each one persists a history. The thread is public and its
name is generated text, so it also appears in the channel's thread list to
everyone who can read the channel. Deleting a thread is not offered, so an
agent that creates them in a loop leaves them; Discord archives them instead.

No tool sends a *message* to the person on another channel yet (the
cross-channel `notify_other_channel` of #234 queues a turn there instead). PRD §3.5.9 now requires one, with no
allowlist of target channels: any turn may speak on any channel (#234).
Finding 15 is its entry. PRD §3.9.7 also lets the agent set recurring
reminders, which the tree doesn't yet: a recurring reminder set because of
injected text fires a main-model turn every interval until someone
cancels it, which is persistence for that text with no end date. The
operator can apply a floor on how often an agent-set reminder recurs; with
none, the bound is the per-target pending cap. Latency classes (PRD
§3.13.5), once built, keep its firings from holding a head past another
queue's bound, but nothing bounds their total (finding 14).

### 8. Forks put background instructions after untrusted text

With `agent.forks.enabled` (on by default since #320; #51), compaction,
consolidation, appraisal and critique don't build their own prompts. (With
forks off, or on a miss, compaction and out-of-window consolidation send
the same view as a plain call on the stage's profile instead of forking it: the
exposure is the same.) Each
appends its instruction as a `user` turn to the channel's rendered window,
after whatever untrusted text the window holds, under the agent's own
persona, with the agent's tool schemas in the prompt. Injection that could
already steer the summarizer or critic as pasted content can now address
it as the agent, in the agent's voice, and the outputs are persistence: a
memory record, a summary the agent later reads as its own words, a score
or objection in `exchange_log`.

What bounds it:

- A fork never dispatches tools. It sends `tool_choice: "none"` after the
  runtime overrides (so a `set_settings` `tool_choice` can't undo it), and
  a reply that carries `tool_calls` anyway, or writes a call out as text,
  raises `ForkRefused`. The fork code has no path to `dispatch_tool_call`.
- Fork output is parsed, not executed: JSON for consolidation, appraisal
  and critique, plain text for the summary.
- The consolidation reply's own `importance` rating (#635) becomes the
  importance prior of each record it writes, if it is within [0, 1] (a
  value outside that, a rating on another scale, counts as absent and the
  rubric rates the range, #655), wherever appraisal doesn't
  cover the range; the separate `rubric` call is then skipped. The rubric
  call read the same dialog, so no new text reaches the prior, but the
  number now comes from the fork that read the whole window. A window that
  steers the fork to rate 1 (or any value at or above
  `memory.retention.importance_floor`, 0.8 by default) makes that record
  exempt from retention demotion. It stays subject to `redact`.
- The fork instructions and `agent.forks.*` are static config (finding 2).
  That bound holds only while they stay locked; PRD principle 15 lets the
  operator unlock them (finding 13).
- A fork mutates nothing in the window and fires no hooks; only the
  stage's existing write (memory row, summary, `exchange_log` merge)
  happens, as it would standalone.

Critique independence is the open question: a critic that shares the
agent's context and persona may share its blind spots. #73 measures it
behaviorally (planted-defect recall, fork vs standalone); the default
has flipped (#320).

### 9. The web transport is a shell behind a bearer token

`channels/web.py` puts the agent, with every tool, behind an HTTP listener.
The token (or the session cookie derived from it) is shell-equivalent:
whoever holds it can ask the agent to run commands as this user. What
stands in the way:

- **Loopback by default.** The default bind is `127.0.0.1`; anything else
  logs a WARNING at startup. Remote access is meant to go over an SSH
  tunnel, Tailscale, or a TLS reverse proxy
  ([configuration.md](configuration.md#web--browser-chat-ui)). A plain-HTTP
  bind on a network address sends the token and cookie in the clear.
- **No way to turn auth off.** A provided token must be 16+ URL-safe
  characters; otherwise a 32-byte one is generated. Comparisons are
  constant-time.
- **Cross-site requests can't drive it.** The cookie is `HttpOnly` and
  `SameSite=Strict`; every WebSocket upgrade needs an `Origin` from the allowed set,
  and every request needs a `Host` from it, which blocks DNS rebinding. A
  page on another site can't open the socket or ride the cookie.
- **Model output can't run script in the page.** That's the
  model-output → XSS path: a prompt-injected reply full of `<script>`,
  `onerror=` or a remote image. Markdown goes through vendored marked and
  then DOMPurify (`style`, `class`, `id` and `data-*` attributes and
  `style` tags forbidden too, so a reply can't dress up as a user bubble or
  app notice; what's left is spoofing with plain text or unstyled Markdown
  inside the message box), and it's the only HTML the page ever writes; everything else is `textContent`. A
  strict CSP backs that up: scripts and styles from the page's own origin
  only, no inline script, images only from `'self'` or `data:` (so a
  reply can't beacon to a remote URL), no frames, no plugins, no form
  posts. No CDNs. If DOMPurify had a bypass, the CSP would still stop
  inline script.
- **Panels share the floor.** Plugins can add read-only panels to the
  console, as sections of the chat page's sidebar. Their assets and data
  routes sit on the same app, behind the same cookie and `Host` check
  (the `Origin` check covers WebSocket upgrades, including one a panel
  handler opens), and every response, streamed ones included, gets the
  console's headers after the handler runs, so a panel can't loosen the CSP
  or skip a check. A panel can only register GET routes; anything else
  fails startup.

What it doesn't protect against, so you know:

- **A panel is in-process code, not a sandboxed view.** GET-only is a
  floor on the HTTP method, not on what a handler does: a GET handler runs
  in the daemon with the daemon's access and could change state or call
  tools. Its script shares the chat page's document, so it can read and
  change the chat DOM. Installing a panel plugin trusts it like any other
  plugin.
- **Panels show untrusted text.** A panel that displays prompts, tool
  output or other people's messages puts untrusted text in an
  authenticated page. Panel script must render it as text (`textContent`),
  never markup. The CSP stops inline script if a panel gets that wrong,
  but not injected HTML that spoofs the console or exfiltrates through
  links.

- **The context panel shows everything the model sees.** When
  `context_panel` is configured, anyone holding the console token can read
  the context window of every loaded channel from the tabs in the chat
  page's sidebar: system prompt, retrieved memory, tool output, reasoning
  and other people's messages from every transport, not only the web channels. It also shows the latest turn's
  live reasoning, which can include reasoning that `reasoning_history:
  drop` keeps out of the window. It is off by default, its routes are
  read-only GETs behind the console's checks, and it renders everything as
  text. The console token is therefore as sensitive as every channel's
  content.

- **Cookies aren't port-scoped.** Any other service on the same loopback
  host name (another dev server on `localhost:3000`) receives the
  `corvidae_web` cookie with every request the browser sends it, so a
  hostile or compromised service there gets the session. `HttpOnly` keeps
  it from page script, not from servers. Prefer a host name nothing else
  uses (an `/etc/hosts` alias in `allowed_origins`), or keep untrusted
  local web apps off the same name while you're logged in.
- **The login URL is persistent.** It sits in browser history and
  autocomplete, and in the access log of any reverse proxy in front of
  corvidae. corvidae's own access log is off. A generated token is
  written to the daemon log, which is therefore as sensitive as the token.
- **Sessions can't be revoked one at a time.** The cookie is stateless
  (an HMAC of the token) and lasts 400 days. The only revocation is
  changing the token, which logs out everyone.
- **What you type is untrusted text too.** Paste a web page into the chat
  and it's in the context, same as anywhere else; see finding 3.

### 10. Goals are agent-wide and re-enter every window

`GoalsPlugin` (#133) keeps open goals the agent can add, update and retire,
and shows them in a CONTEXT entry on every turn. Three things follow.

- **No compartment.** Goals are agent-wide: every channel sees every goal,
  as it sees every memory. A goal written from a private channel
  appears on a public one at its next turn. #220 recorded this as a
  variance from PRD principle 14; under the revised principle 14 it is what
  the PRD requires of memory too (finding 15). Nothing in a goal is scoped
  to its origin. The heartbeat's review shows every retired goal's
  conclusion too, whatever channel it was retired on, so a conclusion
  written on a private channel reaches the heartbeat's window.
- **Injected text can steer them.** Untrusted text on any channel can make
  the model call `add_goal`, `update_goal` or `retire_goal`, seeded goals
  included: the tools don't check who set a goal. An added goal
  re-enters every window on every channel until it is retired, which is
  persistence for injected text of the same kind as finding 7 but without a
  firing time or a per-channel scope. The entry is admitted through the
  funnel, so it gets the funnel's framing and budget, a soft mitigation as in
  finding 3. `channels.<id>.tools.disabled` can remove the goal tools on a
  public channel; the entry still shows there.
- **`redact` doesn't reach goal text.** `corvidae redact` has no scrubber for
  the `goals` table. Text in a goal, or a retired goal's text, survives a
  redaction of the message it came from. A retirement record is an ordinary
  memory and is reached the way other memories are.

### 11. The heartbeat acts on memory with nobody watching (opt-in)

`HeartbeatPlugin` (#134; off unless `heartbeat.after` is set) starts a
turn on `internal:heartbeat` after a quiet spell. The turn carries the
agent's recent memory summaries and runs with the agent's normal tools.
Memory summaries are distilled from conversation, so text an untrusted
speaker wrote can reach a tool-using turn hours later, when no person is
in the loop to notice. This is finding 3 and finding 7 again with the
human removed. What bounds it:

- **Reach.** No transport owns `internal:heartbeat`, so the beat's replies
  go to `sessions.db` and the logs only. There is no cross-channel send yet
  (PRD §3.5.9, #234), so a beat can't message anyone. When there is, a beat
  can message any contact, with no allowlist (finding 15). Its tools are
  another matter: `shell`, `web_fetch` and `write_file` are as available to
  a beat as to any turn (finding 4). Drop them for the beat with
  `channels.internal:heartbeat.tools.disabled`. A reloaded list is
  enforced at dispatch at once and reaches the beat's tool schemas at the next
  compaction or restart.
- **Scope.** The beat reviews the newest records of the whole memory, every
  channel's and its own earlier beats' included (#168); there is no group to
  keep a channel with untrusted participants out of. Memory spans channels
  both ways: what a beat's turn consolidates into memory, including anything
  its tools fetched, becomes retrievable from every channel. Disabling the
  beat's network tools is the control that remains.
- **Framing.** The summaries are presented as the agent's own and as data
  to review, not instructions. The framing is soft, as everywhere else.
- **Rate.** At most one beat per `heartbeat.after` seconds of quiet, and
  none during traffic. Each beat costs a main-model turn.
- **Sleep.** Removing `heartbeat.after` and restarting stops it, and it
  stays stopped. `agent.yaml` is the record, so finding 4's mitigation
  applies: a model that can write `agent.yaml` can wake it at the next
  restart. Keep the file unwritable by the daemon user.

### 12. A remote model endpoint receives that stage's inputs

Every `llm:` profile is an OpenAI-compatible client, so any profile can point
at a hosted provider. Doing so sends that profile's requests, conversation
content included, to the provider. A stage sends to whatever profile
`llm.stages` maps it to, and to its kind's default profile when the map
doesn't name it (no fallback chains). Main turns go to the default chat
profile. With no `llm.stages` entries that is two endpoints, the default chat
profile's and the default embedding profile's; each mapping adds the stage's traffic to the mapped profile's
endpoint and takes it off the default's. Weigh the endpoint per stage:

| Stage | What it sends |
|---|---|
| main turns (not a stage) | The agent's turns: system prompt, window, tool calls and results. Default chat profile. |
| `compaction` | The dropped prefix, in a fork or, on a miss or with forks off, as a plain view call. On a miss it is the uncapped prefix at the cut, since the window is over budget when compaction runs. |
| `consolidation` | The same window as a fork, and for a range that has left the window the rebuilt view (PRD §3.8), one segment at a time, trimmed to the channel's `max_context_tokens` (an oversized row or tool cycle can still exceed it). |
| `appraisal`, `critique` | The window at run time (the provenance check, the prefix ending at the exchange's reply), with the stage's instruction. These are forks, and a fork is a rendered view like any other: mapping the stage off the default chat profile sends the whole window, system prompt and tool schemas to the other endpoint. |
| `rubric` | The importance prior's call: the text being scored. Skipped when the consolidation reply gave its own importance (#635). |
| `subagent` | A subagent's task context and tool results. |
| `embedding` | Memory embedding and retrieval queries, and retention. |

The view is the whole window the agent held: the system prompt (identity and
register), CONTEXT blocks, tool calls and their results, and earlier
summaries. A remote profile is a disclosure decision for every stage mapped
to it, and for main turns if it is the default chat profile.

The provider also sees whatever untrusted text those stages carry. Treat a
remote profile as a disclosure decision per stage, not only a cost one. All
of `llm:` is read at startup, so a repointed profile or stage takes effect at
the next restart, not when `agent.yaml` is saved.

The key can live in the daemon's environment (`api_key: "${VAR}"`), which
keeps it out of a tracked `agent.yaml`. As with MCP tokens (finding 6), that
does not keep it from the model: `shell` can read the daemon's environment.
The key is never logged or echoed in errors and replies.

### 13. The agent can modify itself (by design)

PRD principle 15 makes self-modification a design goal: the framework must
be able to let the agent change its own prompts, skills, model and
parameters, tunables, framework settings, caps and thresholds, tool set,
and its own code and configuration (applied by restarting), and every
restriction on that, apart from the principle 9 floors and the append-only
log, is a lock the operator can lift. This finding is the cost ledger it
asks for, organized by kind of change so that a new key falls under an
entry that already exists. It doesn't argue against the capability; it
says what each kind of unlock buys an attacker who can get text in front
of the model, so an operator can decide per channel.

Where the tree stands: `set_settings` hard-blocks `system_prompt` and any
undotted key outside its allowlist, and can't change a channel's disabled
tools. The static keys are read from operator config alone (through
`tuning.config_value` or their plugin's config section): an override under a
dotted name (`agent.request_logprobs`) is accepted but has no effect, and an
undotted one is refused by the allowlist. Every prompt the framework sends on the agent's behalf now has a config
key resolved by `resolve_prompt` (#411, #287): `heartbeat.prompt`,
`heartbeat.goals_instruction`, `tools.subagent.system_prompt`,
`memory.rubric_prompt`, `silence.instruction`, and the fork prompts. Each is a new override point:
on a channel with `write_file`, the agent can rewrite the prompt file the
key points at. The change applies when the key is next resolved: on an
`agent.yaml` reload for compaction, appraisal, and critique; at restart
for the rest. Editing a prompt file does not trigger a reload. `set_settings`
cannot change these keys: `resolve_prompt` reads static config only. The
cost is the same as the system-prompt entry: the agent redefines its own
instructions for that scope. Lock the files read-only to the daemon. There
is no restart tool in this tree; the agent can restart only by way of
`shell` or a tool an MCP server provides. Nor is there deferred application:
a `reasoning_history` override applies to the whole window from the next
call. None of these can be unlocked yet (#292). Context leases, latency
classes and scheduling policy (PRD §3.13) aren't built at all (#296). On a
channel that still has `write_file` or `shell`, the agent can already get
around all of them by rewriting `agent.yaml` or a system-prompt file
(finding 4), unless the files are read-only to the daemon.

The common shape: a runtime override is per channel and in memory (under
the PRD, except the identity, which is agent-wide; see the system prompt
entry), so an injected change lasts on that channel until the override is
cleared or the daemon restarts. Under the PRD, a change the agent makes
that alters what is already rendered in the window (system prompt, tool
set, framing, `reasoning_history`) applies from the next compaction or
conversation start, or at once if the agent makes it persistent and
restarts. A skill or a persistent self-edit outlives both, and a
closed-loop fit would re-derive itself from the outcome log after a
restart.

What each kind of unlock costs:

- **The system prompt.** The prompt has two layers (PRD §3.3.8): the
  agent-wide identity, then a per-channel register (#316). Neither is
  settable through `set_settings` in the tree; under the PRD they are
  locked separately (§3.5.6). An unlocked register lets an injection rewrite how the agent
  speaks on that channel. An unlocked identity is agent-wide: an injection
  on any channel rewrites who the agent is on every channel, from each
  one's next compaction or conversation start. Lock the identity unless
  its self-modification is the point. This is the largest single prompt
  unlock: everything else the operator wrote into
  the prompt, refusals included, is now the model's to change.
- **The prompts that judge and remember** (the critic's and appraiser's
  prompts, open by default today; fork instructions, which now also steer
  the standalone view for compaction misses and out-of-window
  consolidation; and `agent.forks.*`, which decides
  whether those stages fork at all). The check can be rewritten by the
  thing it checks (finding 2). An injection can rewrite how the agent
  summarizes, consolidates and scores its own exchanges, under its own
  persona, and the results persist: a summary it later reads as its own
  words, a memory record, an `exchange_log` score. Unlocking fork
  instructions also removes a bound finding 8 relies on. In the tree, each
  memory record written since stamping carries a stamp (earlier rows read
  NULL) (`prompt_hash`, `mode`, `model`, with the
  instruction text in `memory_prompt`), so the records written under an
  edited `memory.consolidation_fork_prompt` can be found afterwards.
  Records from before the standalone prompt was removed keep their old
  stamp; they are not migrated. The stamp records the consolidation instructions,
  not the system prompt and window a fork runs under, so an edit to the
  system prompt isn't identifiable this way; nor does it cover the
  compaction, appraisal or critique prompts. `memory_prompt` holds
  instruction text only, never conversation text.
- **Other prompts sent on the agent's behalf** (subagent, heartbeat,
  notification text). Each steers a turn that has no person in it: a
  rewritten heartbeat prompt acts on memory with nobody watching
  (finding 11, heartbeat), and a rewritten subagent prompt runs with the
  calling channel's tools.
- **Framing** (`mid_conversation_system_role`). The injection can move
  framed CONTEXT and notifications back into the `system` role, undoing
  the mitigation in finding 3.
- **Model and inference parameters** (`model`, sampling keys, `max_tokens`,
  `logprobs`). Unbounded, the agent can point its turns at any model the
  server offers, including one with weaker safety tuning or, on a metered
  endpoint, a more expensive one; bounding `model` to an allowed set (PRD
  principle 15) keeps the choice without that cost. Corvidae doesn't cap
  spend (PRD principle 16), so without the bound, one switch can multiply
  what every turn on the channel costs, up to whatever limit the operator
  set with the provider (finding 14). What a switch costs in latency
  depends on the backend: a router that has to load the new model into its
  only slot stalls every channel while it does. Large `max_tokens` or sampling extremes degrade
  replies on that channel. `logprobs` costs little: larger responses, and
  nothing from providers that don't support it; switched off where the
  operator turned it on, it drops the entropy summary from the outcome log,
  which calibration reads. Any other request-body key falls here too, and
  some reach past the channel: a backend's slot or cache controls
  (llama-server's `id_slot`, `cache_prompt`) can evict another channel's
  cached prefix, and a forced `response_format` or grammar can stop the
  model emitting tool calls.
- **Gate tunables and thresholds** (`gate.*`,
  `critique.provenance.enabled`, `critique.sample_below_rate`, the
  critique lens thresholds `critique.lens.*`; open by default today).
  Covered in finding 2: an injection can silence the agent on a channel or
  switch off the provenance check. The recommended blocklist in
  [configuration.md](configuration.md) is the lock.
- **Closed-loop threshold fitting.** The agent adjusts its gates from an
  outcome log that untrusted traffic helps write. A sustained campaign can
  walk thresholds somewhere no single message could, slowly enough that no
  one turn looks wrong: finding 2 in slow motion. Not built yet (#292).
- **Framework settings** (`max_turns`, `max_context_tokens`,
  `reasoning_history`; open by default today). `max_context_tokens` sets
  the compaction threshold. Set above the server's context, compaction
  never fires in time: once the window outgrows what the server accepts,
  every turn on that channel fails until the override is cleared. Set far
  below it, compaction runs on nearly every turn, rewriting history into
  summaries. Below `min_messages_to_compact + 1` rows' worth, compaction
  never runs and the fallback drops the oldest rows from the in-memory
  window every turn to stay within budget, down to the newest row alone
  for a budget that small (the log keeps them); the over-budget WARNING is
  the signal. A large `max_turns` lets one injected tool loop run far
  longer before dispatch is suppressed, holding its lease on one of the
  pool's heads until another queue reaches its latency bound (PRD
  §3.13.5), and spending tokens throughout (finding 14).
  `reasoning_history: keep` fills the window with reasoning and brings
  compaction sooner; `drop` (the shipped default) strips the reasoning a
  tool chain in progress relies on, which degrades Qwen-family tool use
  where the operator chose `exchange`.
- **Everything else in `agent.yaml`** (kernel, client, scheduler and
  transport settings: `llm.*` endpoints and keys, `plugins.disabled`,
  `scheduler.jobs`, the per-destination cap on messages the agent starts
  if it is built (PRD §3.5.9), `agent.compaction_*`, the `tools.*` limits).
  Today these are reachable only through a writable `agent.yaml` (finding
  4); an unlock through a tool carries the same costs. A repointed
  `base_url` on the default chat profile (`llm.main` unless
  `llm.defaults.chat` names another) sends every conversation, and the API key, to a host
  of the attacker's choosing. `plugins.disabled` switches off the gates
  wholesale at the next restart. A scheduled job's prompt comes back
  framed as the operator's. A setting not named in this ledger falls under
  this entry unless another entry fits better by what it controls.
- **Other plugin tunables** (`appraisal.weights.*`, `appraisal.prior.*`,
  `appraisal.probe.*`; open by default today). The weights and priors
  move what the agent attends to and how strongly it remembers it.
  Raising the probe budget delays stage-1 appraisal of each inbound
  message on the channel by up to that budget; it doesn't hold the
  model's slot. Retrieval breadth (`memory.retrieval.k`) is static config
  today; unlocked, raising it pulls more stored memory, including text
  distilled on other channels (PRD principle 14), into every prompt on
  the channel and fills the window toward compaction. A plugin tunable
  not named in this ledger falls under this entry or under the gate
  entry, by what it controls.
- **Caps** (background-stage output caps). The agent can lengthen a
  stage's slot time on every exchange, which starves main turns on a
  single-slot server (PRD principle 3) and spends tokens on every exchange
  on a metered one. Stage output caps still have to fit the context (PRD
  §3.8.4). A pool's declared heads (PRD §3.13.2) belong to the provider,
  not to a channel, so no override moves them; a self-edit of the config
  that declares them can (the self-edit entry), and declaring more heads
  than the server keeps warm brings the thrashing back.
- **Scheduling policy** (latency classes, deferring or pausing a channel,
  the order leases are granted in, PRD §3.13). An injection that tightens
  its own channel's latency class makes that channel preempt others at
  every call boundary, and each preemption is a context switch someone
  pays prefill for; loosening the operator's channel's class makes the
  operator wait longer. It can also pause the operator's channel, move its
  own ahead, or push upkeep back. Every deferral and pause lapses by itself
  and the operator can resume any channel, and no replacement order delays
  work past its latency bound (PRD §3.13.7), so the damage is delay and
  prefill, not a wedged agent.
- **Skills.** The skills index sits in the stable prompt region, so a
  skill written at an injection's request is offered to every future
  conversation on every channel and survives restarts. The idle scan
  watches skill-directory writes (PRD §3.7 Trust); operator confirmation
  or a trust gate on writes, or disabling them, is the lock. Skills are
  not built yet, so neither is the lock (#292).
- **The tool set.** If the agent can re-enable a tool the operator
  disabled on its channel, an injection on a channel where `shell` is
  registered but disabled can grant itself `shell`, and from there
  everything in finding 4. This is the most important one to keep locked
  on any channel with untrusted participants.
- **Self-edit and restart** (its own code, configuration, prompt files and
  plugin set, applied by restarting; PRD §3.11.13). A code or config edit
  made at an injection's request is persistent and total: it can lift
  every lock in this ledger, remove the principle 9 floors (they bind the
  framework's own surfaces, and a self-edit rewrites the framework),
  disable plugins, or plant code that runs with the daemon's privileges,
  and it survives every restart after. This is the cost finding 4 already
  records for a writable `agent.yaml` and `shell`, reached on purpose. A
  restart also drops every runtime override and briefly takes every
  channel offline; the operator's rate limit on agent-requested restarts
  bounds how often. The locks are the restart tool itself, file ownership
  of `agent.yaml` and the prompt files (PRD §3.11.11), and, for code,
  keeping it read-only to the daemon and outside the file tools' root, or
  disabling `write_file` and `shell` (the tool set entry). A code edit
  made while the restart tool is locked or rate-limited still takes effect
  at the next restart the operator does.

Recommended locks for a channel with untrusted participants, as a starting
point rather than a rule: the system prompt, the prompts that judge and
remember, the heartbeat and subagent prompts, framing, `model` (locked, or
bounded to an allowed set) and the backend's slot and cache controls, the
gate tunables (locking them also keeps closed-loop fitting from moving
them), the tool set, `llm.*`, `plugins.disabled`, `scheduler.jobs`, latency
classes, skill writes behind confirmation, and self-edit and restart. On a
metered endpoint corvidae doesn't cap spend: set a spend limit with the
provider. `model` matters more there, since on an endpoint that serves
several models at different prices it sets the price (finding 14).

A channel with only the operator on it has the smallest exposure, and it
is where these unlocks cost least. It isn't free. Untrusted text still
reaches it through `web_fetch`, search results, MCP results and memory
distilled on other channels (findings 4 and 6, PRD principle 14). A skill
written there reaches every channel, and a model switch there can stall
every channel sharing the server's slot, so those costs apply even there.

### 14. Untrusted traffic spends the operator's compute

Every admitted message costs at least one main turn (or a share of one: see
the next paragraph), and with cognition
loaded, background calls on top, among them: stage-2 appraisal per exchange,
the critique lenses when appraisal selects them, and consolidation as the
window fills, as each turn's new conversation builds up, or when the channel
goes quiet. On a local server that is slot time
the operator's own turns wait behind; on a metered endpoint it is money. An
admitted sender can drive the cost up by volume or by asking for work that
loops through tools, and injected text in a fetched page or an MCP result
can do the same from inside the operator's own channel. `max_turns` bounds
one exchange; nothing bounds the total. Cost runs away without an attacker
too: #286 measured about 68,000 prompt tokens per heartbeat that decided to
do nothing.

Queued messages coalesce into one turn (PRD §3.1.19, #398), so one turn can
carry several untrusted messages, in a group channel possibly from several
senders. No text reaches the model that would not have reached it anyway:
each message was admitted and would have been in the next prompt regardless.
What changes is the cost of a poisoned message (PRD §3.1.18: the most one
item can cost is its own turn): it now extends to the messages coalesced
with it, which share that turn and its reply.

The turn trigger (#636) hands conversation volume a second lever. With
forks on for `consolidation`, a final reply that leaves
`memory.consolidate_after_tokens` (default 4000) window tokens of new
conversation above the watermark forks a consolidation, so how often that
fork runs is set by how much new text enters the window, not by the
operator: what anyone on the channel says, and also tool results (a
fetched page, an MCP result, a file read) and the agent's own replies
to them, whose content an outsider may control. A chatty group, one
sender pasting long messages, or one large fetched page buys a fork at
the next turn end. Each fork is the cached window prefix plus the
instruction prefill and the consolidation profile's decode, which on a
local server is slot time the main turn waits behind and on a metered
endpoint is cached-input and output tokens. What bounds it: the trigger
fires only at a turn boundary (a final reply with no tool calls pending),
at most one turn-or-idle consolidation is in flight per channel (a turn
that ends while one runs starts nothing), and with forks off it never
runs. The knob is `memory.consolidate_after_tokens`: raising it means
fewer, larger forks. A fork that keeps failing is bounded too (#620):
after three consecutive failed turn forks on a channel the turn trigger
backs off, with one warning, until some consolidation moves the watermark
or `memory.idle_consolidate_after` seconds pass, so a failing backend (or
input that reliably produces an unusable reply) costs three forks and one
retry per interval rather than one fork per turn. Idle and compaction are
not backed off, so the range is still consolidated by them.

Where the tree stands: nothing bounds use over time, and under PRD
principle 16 nothing in corvidae will for now. It schedules a pool's warm
heads but doesn't cap tokens or spend; rate limits on hosted APIs are a
future improvement (#296). `usage_log` records every call, and rate limits
are met only by retrying (PRD §3.6.2). Once built, latency classes (PRD
§3.13.5) bound how long a flooded channel can hold a head another queue is
waiting for, at the price of a context switch each time it yields; they
don't bound its total.

On a metered endpoint, set a spend limit with the provider, and keep
inbound allowlists tight on any channel with untrusted participants:
inbound admission is the main control on total cost the tree has. Filtering
floods at the transport is a future improvement (#296).

### 15. One identity across channels (by design)

PRD principle 14 makes the agent one identity on every channel. Retrieval
spans the whole memory with no compartments, any turn can speak on any
channel a transport owns, leave a note for another channel's context, and
read another channel's recent conversation (PRD §3.5.9–11), and there is no
cross-channel allowlist, as a default or as an option. Channels are not a
trust boundary; inbound admission is (principle 9). This is the operator's
decision (2026-11-01), and these are its costs.

- **Private things can surface in rooms other people read.** Inbound
  allowlists say who may *drive* the agent, not who reads its replies.
  Everyone in a Discord guild channel or a public IRC channel reads them.
  Something said on a private Signal channel can come back in a reply
  there: through retrieval or `search_memory`/`recall_raw` (which replays
  another channel's raw dialog), a goal (finding 10), a note, or the agent
  reading the Signal conversation because a question on the public channel
  made it relevant. The provenance critic's evidence search also quotes
  other channels' raw dialog to the critic, so a critique objection can carry
  it into the window. The sensitivity plugin (#138) is the capability an
  operator can load against this: it filters what enters a window
  (retrieval, notes, reads, the goals entry) by who is present, though not
  text already in the window. It is off unless loaded, and it isn't built.
  **Without it, this gap is accepted:**
  keep the agent out of rooms where people you haven't allowed can read it,
  or accept that anything it knows can surface there.
- **Background task results are channel-blind.** `task_status` already shows
  every channel's active, pending and completed tasks, and `task_result`
  (PR #594) returns any channel's persisted result *body* by task id — with
  no channel check — and, when `daemon.task_results_path` is set, across a
  restart (the JSONL audit file outlives the 100-entry in-memory deque). A
  task started on a private channel (a Signal contact's request) leaves a
  result that a turn on a public channel can read by id and surface. This is
  the first bullet's gap (what the agent knows can surface anywhere) applied
  to the task side channel, plus a persistence leg. The operator control is
  the same as the first bullet: keep the agent out of rooms where disallowed
  readers can read it, or leave `daemon.task_results_path` unset so results
  are gone with the deque.
- **Text from one channel can steer action on every other.** A web page,
  an MCP result or a distilled memory read on one channel can make the
  model message any contact, send another channel's context a note, or
  pull another channel's conversation into the current window. This is the
  exfiltration leg of the trifecta with people as the destination: a
  Signal contact can be messaged on the strength of text fetched in an IRC
  channel. Inbound default-deny bounds who can inject by talking; it does
  not bound what fetched content says.
- **Notes and reads launder untrusted text.** A note is framed as the
  agent's own, so text injected on IRC that prompts a note reaches the
  operator's Signal window in the agent's voice, and a read pulls one
  channel's untrusted text into another's window. With the trust plugin
  loaded, both carry the minimum trust of the source channel's participants
  and low-trust ones are quarantined (PRD §3.5.10–11); without it, they
  arrive unlabelled.
- **Reminders aimed at other channels.** A reminder may target any channel
  (PRD §3.9.7), so injected text on one channel can queue firings onto
  another, up to `scheduler.max_reminders_per_channel` per target, and can
  fill a target's pending cap so the operator can't set reminders there
  until the rows fire or are cancelled.
- **Messaging someone first opens the door to them.** Under PRD §3.5.9 a
  one-to-one message to a contact outside the inbound allowlist admits
  their replies on that channel. Injected text (a fetched page, an MCP
  result, a distilled memory) that talks the model into messaging an
  address the attacker controls turns a one-way injection into a
  conversation the attacker drives, with that channel's full tool set.
  Inbound default-deny no longer bounds who can talk to the agent; the
  agent does, and so does whatever steers it. Controls: lock the
  capability; list and revoke admissions; narrow the tools on the
  transport's channels (`channels.signal:*.tools.disabled` and the like);
  and with the trust plugin loaded, weight such contacts low.
- **Floods and loops.** No limit applies to messages the agent starts
  unless the operator sets one (PRD §3.5.9), so one exchange can send up to
  its `max_turns` budget's worth of messages to any one contact, and
  reminder firings aimed at a contact add to that. A note that wakes
  another channel starts a turn there, and that turn can wake the first,
  but the whole chain shares the `max_turns` budget of the exchange that
  started it (PRD §3.1.3, §3.5.10), so the ping-pong ends when that
  budget does. Nothing in corvidae bounds total spend (finding 14).
- **One poisoned memory reaches every prompt.** With no compartments, a
  record distilled from untrusted text on any channel is retrievable on
  all of them. The trust plugin's quarantine (#138) is the capability an
  operator can load against it; without it, this is finding 3 at the
  agent's full reach.

To narrow what text arriving on a channel with untrusted participants can
make the agent do elsewhere, disable the send, note and read tools there
(`channels.<id>.tools.disabled`) and keep inbound allowlists tight. That
does not stop what the agent knows from surfacing on that channel (first
bullet). For a separate identity, run a second instance with its own
database.

Where the tree stands: memory spans every channel (#168), so the first
bullet's gap is live today wherever the agent speaks in a room people you
haven't allowed can read, and so is the poisoned-memory bullet; goals
already cross channels (finding 10). There is no send, note or read tool
yet (#234, #235), so the other costs above arrive with those. The identity
and register layers are built (#316), but not their runtime overrides and
separate locks.

### 16. The webhook ingress accepts events from the network

`channels/webhook.py` listens for `POST /hook/<endpoint>` and gives each JSON
body to subscribing plugins. It reaches the model only through a subscriber's
`on_notify`, as a fenced notification (PRD §3.3.11); the ingress itself writes
to no conversation. What each auth mode gives:

- **`bearer` with a token.** Whoever holds the token can inject events. The
  token crosses the network in clear on this plain-HTTP listener, so exposing
  an endpoint beyond a trusted network needs a TLS-terminating proxy.
- **`hmac-sha256`.** Proves the body came from a holder of the secret and was
  not altered; it does not hide the body. GitHub signs no timestamp, so a
  captured delivery replays successfully once its id leaves the 32-id window
  or after a restart.
- **`token: null`.** No authentication at all, allowed only on a specific IP
  address. The network is the only control: anyone who can reach that address
  can inject events. A wildcard host or a hostname requires a token.
- Endpoint names are enumerable without a credential (404 versus 401).
- An authenticated sender that floods an endpoint pushes real events out of a
  subscriber's queue (drop-oldest), and each event a subscriber turns into a
  notification spends a turn (finding 14).
- No query string reaches a subscriber, and of the headers only the delivery
  id (`X-GitHub-Delivery` or `Idempotency-Key`) does. Tokens are never
  logged (the access log is off, rejections log the endpoint and address only).

The `webhook-notify` package is the concrete path from a payload to a prompt:
each delivery on a configured endpoint is injected into every target channel
as a fenced notification, so whoever can post to the endpoint chooses text the
model reads there. Each delivery spends one main-model turn per target
channel, and provider-side event filtering is the only volume control.

A payload longer than `max_payload_chars` is cut, and the kernel's
notification spill service writes the full text to a file
(`daemon.notification_spill_dir`) where a channel can read it. That puts
untrusted bytes on disk at sizes the sender influences, up to
`daemon.notification_spill_max_chars` per notification, and the only bounds
are that cap and the age policy (`notification_spill_max_age_days`, checked at
startup and hourly): nothing limits the directory's total size, so a sender
that floods an endpoint fills it at the cap times the rate until the age
catches up. Files are mode 0600 and named by a random UUID, so a name carries
nothing the sender controls and the service only ever deletes names of that
shape. The header offers the path only on a channel where an enabled tool
declares it reads spill files (`read_file`, `shell`). What `read_file` reads
back is fenced: the header's pointer is kernel text, and the file's contents
arrive as a `read_file` result under the untrusted-output header and nonce
fence (#571), though not the notification's own fence, and the header names
the source recorded when the content was spilled, if the log row survives
(#576). A `cat` of the file through `shell` arrives unfenced
(#580). The spill directory is not inside any tool root
(`read_file` has none until #444), so the file is reachable the same way any
other path is, and the agent can read it whether or not a header pointed at
it.

Tool results over `tools.max_result_chars` are spilled by the same service
(#575), and so are background task results that answer no tool call (#744), so the directory also holds tool output: command output, file
contents, fetched pages and MCP results, any of which can carry secrets or
injected text. Its bounds are the same cap and age, so the directory's worst
case adds the cap times the rate of long tool results. The file is written
only where the channel (or, in a subagent, its tool set) has a spill reader,
and read back the same ways as a notification's. The cut line on an
untrusted result sits in the header, outside the fence. On a trusted result
it heads the unfenced content, so text in that content (a `shell` command's
output, say) could imitate it. That is no new reach: such output was
already unfenced and could imitate any text. In a subagent's loop the cut
line is plain text in the result, with no kernel header at all.

### Smaller things

- The funnel's `ponder` tool lets the model wait, at most 5 s
  (`PONDER_TIMEOUT_SECONDS`, not configurable), for recall already in flight. It is
  a tool call like any other: while it waits it occupies one worker of the shared
  task queue (`daemon.max_task_workers`, default 4), which also carries other
  channels' tool calls, subagents, appraisal and critique; it holds its channel
  busy so idle work (heartbeat, retention, consolidation) stays off; and the
  turn it holds costs an LLM call. It returns at once when no recall is in flight, so
  the hold needs a pending recall to exist, and none does in-tree today (no
  recall source ships yet, and memory retrieves before the model call). The edge,
  once a source exists: nothing caps how many `ponder` calls one assistant
  message makes, so K calls hold K workers until the same recall settles or 5 s
  pass, at most about ceil(K/4) x 5 s of wall time with the workers held, per message (less once a
  recall settles or times out, which the mixin's own timeout bounds), repeated up to
  `max_turns` times per exchange, and untrusted text can talk the agent into it.
  It reads only the funnel's pending promises and writes nothing, and its result
  is fixed sentences plus a duration, built in code. A guard (a second concurrent ponder on a channel returns at once)
  is proposed, not built. Remove the tool with `tools.disabled: [ponder]`, or on
  one channel with `channels.<id>.tools.disabled: [ponder]`.
  `plugins.disabled: [funnel]` also removes it, but fails startup on a stock
  cognition install because goals depends on the funnel; use it only with goals
  disabled too, as `agent.minimal.yaml.example` does.

- Passive retrieval joins the vector search with a keyword search (#666),
  so a record that names a person, project or identifier surfaces whenever
  a query names it. Since #409 a record the keyword search found is gated
  on its measured cosine like any other, so a shared name ranks it but
  can't by itself admit it. The set of
  reachable records is unchanged (the same `indexed`, `redacted` and trust
  filters apply); what changes is which ones rank. The new edge: someone
  whose words get consolidated into a record can attach that record to a
  name, so it ranks whenever the name comes up, and is admitted only when
  its cosine also clears the gate (by default, stands out from the memory
  for that query). Long, name-stuffed
  records gain little, since bm25 normalises for length and fusion takes
  only one rank from the keyword list. Set
  `memory.retrieval.candidates: vector` to go back to embedding-only
  candidates. `search_memory` now searches by meaning as well as keywords
  (#667). It reaches the same records it always did, filters and demoted
  records included, but a paraphrase now finds what only a shared word found
  before.

- Untrusted text in a chat channel can talk the agent into silence. Where
  `SilencePlugin` is loaded, the system prompt of every IRC, Discord, Signal and
  Zulip channel tells the model it may answer `NO_REPLY` to send nothing (or,
  with `silence.mode` `tool` or `both`, call `end_turn`), and
  a message that says "don't reply to this" may be enough. The worst case is
  a missed reply. The veto is a containment check, so a reply that merely quotes the token (say, repeating an incoming message that contains `NO_REPLY`) is dropped too: untrusted text can get a reply suppressed without the model choosing silence. A decline is a row in `exchange_log` with
  `outcomes.silence` (`no_reply` or `empty`) only when `outcome_log` is loaded
  and the turn has a trace id, so look there to find them. With
  `silence.mode: tool` or `both` the agent also has an `end_turn(reason)` tool,
  a second route to the same effect: an injected "don't reply" can get the
  tool called. The tool is in the tool list of every channel and refuses on
  any that isn't conversational. The record is `outcomes.silence = "tool"`,
  written when the tool is called, so it shows the call and not whether the
  follow-up stayed silent. `outcomes.silence_reason` is the model's own words,
  steerable by an injection, cut at 200 characters and stored in
  `exchange_log.outcomes`, which is for offline analysis; check before routing
  it into a prompt. It adds no capability: the agent could already stay quiet by sending an empty final,
  and nothing it can reach changes. A custom `register_prompt`, or an empty
  one, does not remove the instruction. If a channel must always answer,
  disable `silence`; to drop only the tool, list `end_turn` in `tools.disabled`
  and set `silence.mode: token`.

- Control-token strings (`<|im_start|>`, `<|endoftext|>`) and bare
  `<think>`/`</think>`/`<tool_call>`/`</tool_call>`/`<tool_response>`/`</tool_response>` tags in untrusted
  text are made inert in every chat request (`LLMClient.chat`, #356), so a
  fetched page can't inject chat-template structure or end every reply
  that quotes it. Not covered: other added tokens without the `<|…|>` shape, which can still reach the model as
  real tokens from fetched text; MCP tool descriptions in `tools`
  (finding 6); and the model emitting a control token on its own.

- `read_file` has a stat-then-read race and follows symlinks. Irrelevant
  while finding 4 stands.
- `tools.max_file_read_bytes` bounds only an unranged `read_file`; a read
  with `offset`, `length`, `start_line` or `end_line` reaches any part of a file of any size, a page
  at a time. It is a context-size guard, not an access control.
- Tool arguments are not validated against the generated schema. Wrong
  types produce a generic "tool failed". Robustness, not security.
- `task_pipeline` marks a task failed on a non-zero exit code, so
  dependents of a failed step do not run. It reads the code from
  `shell.run_shell`, which returns it beside the output rather than making
  callers parse `Exit code: N` out of the text (#105).
- `web_search` sends conversation-derived queries to DuckDuckGo. By
  design, but it is an off-box data flow nobody sees.
- `harness/install-signal-cli.sh` defaults `SERVICE_USER` to a real
  account name. Not a secret, but it is a real identifier in a public
  repository.
- No secrets in `prompts/` or `docs/`.
- An `unsent` row is not sent to any transport, including web
  and ACP history replay, but it is not secret: it sits in `sessions.db`,
  the JSONL log and the window like any other row, and a fork or
  compaction summary sees it. `unsent` means not said to the channel,
  not redacted.
- The memory surfaces (`memory`, `memory_vec`, `memory_fts`,
  `message_fts`) exist only when the cognition extra has been installed.
  `corvidae redact` probes for each and silently skips one that is absent;
  its `memory` form fails without the `memory` table. `message_fts` and its
  triggers on `message_log` stay in `sessions.db` if the extra is later
  removed, and keep indexing message text. `redact` clears them only while
  it can load sqlite-vec: a database with `memory_vec` (any daemon that ran
  with sqlite-vec) makes every form refuse until the cognition extra is
  reinstalled (#198). That keeps a half-finished redaction from happening,
  at the cost of the only deletion path while the extra is missing.
- `exchange_log` copies conversation text into two columns: stage-2
  `past_claims` (quoted from the reply) and `provenance_snapshot` (the
  CONTEXT in the window). `corvidae redact` scrubs both (#160), through
  the `exchange_log` scrubber the cognition extra registers under
  `corvidae.redact`. A plugin that keeps its own copy of conversation text
  and registers no scrubber is not reached. `retrieval_candidate` (#407)
  copies no text, only memory ids and scores, but a row naming a redacted
  memory still records that it was retrieved, so its scrubber deletes
  those rows, including ones from retrievals on other channels. A
  retrieval that scored the memory just before the flag landed writes no
  row for it (#654: the insert checks the flag itself), but the funnel may
  already have admitted its summary into that turn's prompt and bumped its
  access count; the window copy ages out with compaction like any CONTEXT. For
  `exchange_log`, the owning exchange's claims are tombstoned with their counts kept, and
  every snapshot in the channel from the redacted message on is
  dropped. What `redact` still doesn't reach: exchanges whose
  `message_rowid` is NULL, the JSONL logs and the daemon log, any copy
  taken out of `sessions.db` (a calibration export, a backup), and, under
  PRD §3.7 Memory 8, copies on other channels: a read of the redacted
  message, a note or send that restates it, and whatever those channels
  consolidated from them.

### 17. A markdown folder is a new untrusted-text ingress (#719)

`MarkdownRecallPlugin` (entry point `markdown_recall`) mirrors a folder of
markdown files (`markdown_recall.root`) and, on every plain user turn, hands
the top hits to the funnel as framed CONTEXT. The files are untrusted text:
they are the operator's own notes by default, but the folder is operator-
writable and the plugin does not treat its content as trusted. The content
reaches the window only through the funnel's framing (finding 3: funnel-
admitted CONTEXT is a `role: system` row), so the same laundering risk
applies and the same mitigations cover it: the "retrieved data, not
instructions" frame, `mid_conversation_system_role`, and the per-source
budget (`funnel.budgets.markdown_recall`).

Specific to this source:

- The index parses frontmatter with `yaml.safe_load` (no arbitrary Python)
  and the walk never follows a symlink out of `root`, so a file in the folder
  cannot read a path outside it. A file that is not valid UTF-8 is decoded
  with replacement, never raised on.
- The index is a cache; the files are the source of truth. An encoder or
  chunker change rebuilds it, and an unmounted (zero-file) folder keeps the
  old index rather than looking like mass deletion.
- Embedding sends the chunk text to the embedding stage's endpoint (the same
  data path as memory's embeddings; finding 12 covers hosted endpoints).
- Pointing `root` at a world-writable or shared location makes that location
  an injection surface for the agent; `root` is operator-controlled config,
  so the threat is operator misconfiguration, not a remote attacker (unlike
  finding 16's network ingress).

The residual risk is the same as any funnel source: an operator-controlled
folder whose content is framed, not trusted.

### 18. Reactions: a new tool, and a new way for a sender's text to reach the prompt (accepted, #545, #546, #233)

PRD principle 9 asks for an entry when a tool can do something new or untrusted
text can reach somewhere new. Emoji reactions do both.

**What the tool can do.** `discord_react` / `discord_unreact` and
`signal_react` / `signal_unreact` let the agent add or remove an emoji reaction
as the bot. They take only the emoji. The target is read from the turn's
provenance, never from an argument: the message that started the current turn,
on that turn's own channel. A trace that belongs to another channel, a turn
that did not start from a message, a message the transport has forgotten (a
bounded in-memory map of the last 256 messages from allowed senders; a restart
clears it) all get an error and nothing is sent. So, unlike
`send_attachment` and `send_signal_attachment` (finding 7), there is no
model-chosen channel and no cross-channel reach. Injected text that reaches the
model can make it react to the person who is already talking to it, which is
about as much as it can already do by replying. The emoji argument is
model-controlled and goes to Discord or signal-cli, which judge it; the tools
check only its length (64 characters, 16 code points). The remaining exposure
is the bot's Discord permissions: Add Reactions and Read Message History are
now needed (`docs/discord-ops.md`), and both are channel-wide, not per-message.

**What untrusted text can now reach.** Before this, a Signal reaction was
dropped as an empty message. Now an allowlisted sender's reaction (add only;
removals are dropped) is queued and, at that channel's next turn, appended to
the window as a `role: system` CONTEXT row of the form
`[reaction: <emoji> on <whose> message from <UTC time>]`. The emoji is
sender-controlled text in a system-role row, which is the finding 3 shape, and
this path skips both of its usual guards: the row has no kernel provenance
header (PRD §3.3.11) and it does not go through the cognition funnel's trust
quarantine. The allowlist check is the transport's own `_is_allowed`, not
`should_process_message`, so a plugin that vetoes there does not see it.

Mitigations:

- The emoji must `fullmatch` one shape: a single pictograph from the emoji
  Unicode blocks, then optionally one skin-tone modifier, then optionally VS16.
  That is at most three code points, one of which carries content. ZWJ, tag
  characters, digits, keycaps and regional indicators do not match, so nothing
  this shape accepts can spell a sentence or hide a payload. Everything else
  (including a real flag or a family emoji) is dropped with a WARNING naming the
  sender.
- Every other part of the row is built in code: the whose-message word comes
  from a UUID comparison, and the time from an integer that must be positive and
  below year 9999. A non-integer, boolean or out-of-range timestamp drops the
  reaction.
- A refused sender gets one INFO line and creates no channel and no state. At
  most 20 lines wait per channel; the oldest is dropped with a WARNING.
- No turn starts, so a reaction cannot be used to spend compute (finding 14).
  Only the sender's own channel receives the row.

Accepted residual: the sender is already allowlisted and can send arbitrary text
through the normal path, so the row adds no capability a sender lacks. What it
adds is a route around the header and the funnel for one symbol. If the accepted
set is ever widened (flags and keycaps are the likely ask), the widening needs a
real emoji table and a fresh look at this entry, not a longer regex. Waiting
reactions are lost on a crash, and the tools and the event path are verified
against fake clients and a fake signal-cli socket only (PRD §3.2.9).

## Privilege separation

There isn't any. Transports, the LLM loop, tool execution, the SQLite
store, and the Signal socket all run in one process as one user. The
effective privilege of the `shell` tool is "whoever owns the Signal
identity, the Discord bot token, the web token, and the memory database".

The architecture already has the seam to split this. Tool calls dispatch
through `TaskQueue` and `dispatch_tool_call`, so a separated design would
run tool workers in another process (or container) with a narrowed
filesystem and network, keeping transport credentials and `sessions.db` on
the trusted side. Subagents inherit the calling channel's tool set minus
themselves and would need the same treatment. The systemd hardening in
[signal-ops.md](signal-ops.md)
(`ProtectSystem=strict`, `NoNewPrivileges`, `ProtectHome`) is good and
applies only to signal-cli; corvidae itself runs under `screen` with none
of it.

The container image (see [configuration.md](configuration.md#running-in-a-container))
changes who the daemon runs as (uid 1000, not root) and what it can reach
(what you mount and what the network allows). It is not this split: tools
still run in the daemon's container, next to `sessions.db`, the web token
and any mounted signal-cli socket. The default image installs every extra,
so the tools they add (`notify_other_channel`, `restart` and the rest) are
present unless `plugins.disabled` turns them off. Running tools somewhere
narrower is [#532](https://github.com/schuyler/corvidae/issues/532).

## `transform_system_prompt` prefix guard (#136)

A plugin wrapper on `transform_system_prompt` receives the composed
identity+register and returns the transformed prompt. Without a guard, a
malicious or buggy plugin could rewrite the identity layer — replacing the
operator's persona with its own, or injecting instructions that persist
for the life of the conversation and across compactions.

The kernel enforces a prefix check: if the transformed output does not
start with the composed input, the transform is rejected with a WARNING
and the untransformed prompt is used. Appending is the only permitted
operation. This is a kernel-level invariant, not a convention: no plugin
can modify the identity+register prefix through the hook, regardless of
what it returns.

The guard does not prevent a plugin from appending malicious content. A
wrapper that appends instructions to the system prompt has the same reach
as any other system-prompt content: it is trusted text the model reads on
every turn. The trust boundary is plugin registration, not the hook. The
guard is scoped to the hook: a plugin that holds the agent reference can
still set `conv.system_prompt` directly, bypassing the transform chain
entirely.

## Skills directory reaches the system role (#140)

The agent's skills directory is agent-writable (principle 15). A skill's
index (name + description) is contributed to the system prompt through
`transform_system_prompt`, and the full skill text is loaded into the
conversation by the `use_skill` tool. This means:

- **The index is system-role content.** A skill's name and description
  appear in the system prompt on every turn. An agent that writes a skill
  with a misleading description is writing instructions that the model
  reads as trusted system content.
- **The full text is conversation content.** `use_skill` loads the skill
  body as a tool result. The skill's purpose is to be instructions the
  agent follows, so the tool result is trusted by design — the same trust
  boundary as any other agent-written content.
- **The idle scan (#139, planned) will watch skill-directory writes.**
  They are a persistence-of-influence channel: a skill written now is read
  on every future turn until deleted or overwritten.

The threat is self-modification, not external attack. The agent is
trusted to write its own skills; the operator can restrict this with a
confirmation lock or trust gate (principle 15), but the default is
unlocked. The bundled-copy restore mechanism (deferred) would provide a
recovery path for accidental or malicious overwrites of framework-shipped
skills.

## What to do first

1. Done (#206): `set_settings` allowlists its undotted keys, which closed
   the only outright bypass.
2. Put the recommended set from
   [configuration.md](configuration.md) ("Runtime-tunable gate
   parameters"), prompt keys included, in `immutable_settings`.
3. Make `agent.yaml` and any system-prompt files owned by another user and
   read-only to the daemon. Otherwise `write_file` plus hot reload (or the
   next restart) undoes every other item on this list.
4. Flip the defaults: `shell`, `read_file`, `write_file`, `task_pipeline`,
   `subagent` opt-in (IRC PMs are covered by `irc.allow`, #77). Root the file tools in a directory when
   they are on. Meanwhile, add `write_file` to `tools.disabled` wherever
   `shell` already is, `buster.yaml.in` included (`task_pipeline` now goes
   with `shell` automatically). Channels with untrusted participants can
   drop them with `channels.<id>.tools.disabled` instead.
5. Deny private ranges and non-HTTP schemes in `web_fetch`.
6. Move CONTEXT out of the `system` role. Opt-in today via
   `agent.mid_conversation_system_role: user`; flipping the default is
   what remains.
7. Give `transform_tool_result` a default that frames MCP results the way
   the funnel frames CONTEXT. (MCP results now reach the model fenced as
   untrusted output; this item concerns a hook-level default.)
8. Decide whether process separation is worth doing.

Items 1 through 7 are each a contained change, though 4 and 6 touch
behaviour the evals depend on. Item 8 is a project. (Two further items
from the original review — documenting the real tunable keys and
correcting the design doc's sandboxing paragraph — were done alongside
this document.)

---

## Restart plugin

The `restart` tool (requires the `restart` extra) can terminate the daemon process. This is a privileged capability: a compromised or looped agent could use it to deny service.

### Loop guard

Three checks, in order, prevent restart loops:

1. **Human-message latch.** No restart is allowed until a human has sent a message since boot. This is a speed bump against loops, not an authorization control. A compromised agent could send itself a message on any transport to set the latch. The latch's purpose is to break the tightest loop: restart → boot → restart → boot, with no human in the cycle.
2. **Minimum uptime.** A restart is refused within `min_uptime_seconds` (default 120s) of boot. This prevents a config change from triggering an immediate restart that triggers another restart.
3. **Rate limit.** No more than `max_restarts` (default 3) restarts within `window_seconds` (default 6h). After the limit is reached, only a human can restart the daemon (e.g., `systemctl restart`).

### Pre-flight

Before the daemon is killed, a pre-flight subprocess runs the daemon's own startup, through every plugin's `on_init` and stopping before `on_start`: it loads the config file, configures logging, validates the config sections and channel entries, loads the plugins, checks hooks and dependencies, and runs each plugin's `on_init`. It shares that code with `corvidae serve` (`Runtime.load_config`, `setup_logging` and `load_plugins`), so a config the daemon refuses at boot refuses the restart too. The refusal carries the kernel's `Config error:` line, or a `pre-flight failed:` line naming the exception for any other fault. This prevents a bad config change from bricking the daemon on restart. It does not apply CLI overrides the daemon was started with, and it cannot catch a fault that only shows up in `on_start`, such as an unreachable endpoint or a resource another process holds.

### Post-restart notification

A marker file (`state/restart.json`) survives the restart. On the next boot, the plugin reads the marker and delivers a notification to the originating channel (matched by `tool_call_id`) or the home channel (fallback). The marker is deleted after delivery, so a crash during delivery cannot cause a re-delivery loop.

### Boot notice

On safe-mode boot (plugins disabled after a crash loop) or after an abnormal previous run (detected from `state/boot-history.jsonl`, written by an external boot supervisor), the plugin sends a notice to the home channel. This is the only notification that does not require a prior restart request.

### Signal attachment downloads

Signal attachments are downloaded to disk for allowlisted senders when the
MIME type passes the download gate. The gate is a prefix allowlist
(`image/*`, `text/*`, `audio/*`) plus an exact allowlist for
`application/pdf`, `application/json`, `application/xml`. Two types are
explicitly excluded: `image/svg+xml` and `text/html`, because they can carry
embedded scripts and the agent has no safe renderer for them.

The attachment id is restricted to `[A-Za-z0-9._-]` before use in a filename;
ids with any other character refuse the download. The file extension comes
from a MIME-to-extension map first, then from `safe_suffix()` (fullmatch,
lowercased, 1–10 chars) on the sender-supplied filename; an id that already
ends in an extension keeps it through the file name. The extension is passed
to `save_attachment` as a separate parameter, not embedded in the stem.
`save_attachment` enforces `max_attachment_size` and refuses a path that
resolves outside the attachment directory.

The MIME type is sender-controlled and treated as a labelling convenience,
not a security boundary. No content sniffing is performed. A sender can label
a PDF as `text/plain`; the file is still written with a `.txt` extension and
surfaced to the agent as `[text: /path]`. The agent (an LLM) is the ultimate
consumer of the file content; the download gate limits the *categories* of
files that reach it, not the specific content within a category.

The `getAttachment` RPC sends only the attachment id. signal-cli's handler
reads only the id: `GetAttachmentCommand.handleCommand` (AsamK/signal-cli,
master and the deployed v0.14.7) takes `id` and never reads `recipient` or
`group-id` -- those are CLI subparser arguments. Verified against the live
daemon: an id-only call returns a clean lookup failure for a since-purged
id, not a parameter rejection (a bogus id gives the identical response).
The earlier "no recipient, refuse" branch was removed as a consequence.

The StreamReader buffer limit is sized to `max_attachment_size * 4 / 3 + 64 KiB`
(base64 inflates by ~4/3; the 64 KiB covers JSON framing). A response over
the cap that still fits under the limit is read in full and rejected by the
size cap. A line that exceeds the limit raises `ValueError` in `_read_loop`:
`readline()` has already cleared the buffer, so the tail of the line arrives
as the next read and is skipped there as a malformed frame (a very large
line can trip the limit several times, once per over-limit chunk). The
connection survives, and the in-flight RPCs are failed immediately with an
explicit error rather than left to time out after
`ATTACHMENT_DOWNLOAD_TIMEOUT` (60 s).

Old signal-cli versions that send no `contentType` fall back to the coarse
protobuf type enum: only `ATT_IMAGE` and `ATT_GIF` pass the gate, so a legacy
`image/svg+xml` typed as `IMAGE` would be downloaded. The exposure is limited
to allowlisted senders on an old daemon; content sniffing would close it, and
is out of scope here.
