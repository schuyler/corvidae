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
The Discord transport follows Signal's model: default-deny, silent refusal,
and outbound mentions disabled. The web transport binds loopback by
default, needs a token that can't be turned off, and renders model output
through DOMPurify under a strict CSP.

Memory is compartmentalized per channel in the tree today, but that is no
longer the design: the PRD makes the agent one identity across channels,
with no compartments and no cross-channel allowlist. Finding 15 records what
that costs, including a gap the operator has accepted.

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
  a channel other people can read,
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
messages. In the tree those excerpts are limited to the channel's memory
scope, so no text crosses a channel boundary that memory wouldn't already
cross; the PRD widens the search to every channel, matching retrieval
(finding 15). But
they are quoted, untrusted text in the critic's prompt, framed as data, and
the critic's objections re-enter the window as CONTEXT (provenance objections also wake a turn whose stub tells the agent it may reply `NO_REPLY` to send nothing; that veto applies to critique-origin turns only). The number of
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
   one drives the agent, tools and all. Changing the token revokes it.
   An MCP server's bearer token, passed through `${VAR}` in
   `mcp.servers.<name>.headers`, is a fourth: it lives in the daemon's
   environment, which `shell` can read (`env`, `/proc/self/environ`), and
   grants whatever that server grants.
2. The host account corvidae runs as, and anything it can read or run.
3. `sessions.db`, the JSONL logs, and the memory tables: plaintext copies
   of every conversation, including end-to-end-encrypted Signal ones.
   Discord conversations are not end-to-end encrypted to begin with:
   Discord the company can read them whatever corvidae does.
4. `agent.yaml`: operator intent, hot-reloaded within seconds of a write
   (default poll 2s).
5. The LLM server, if it is shared with anything else, and each model
   pool's capacity and budget (PRD principle 16): slot time the
   operator's own turns need on a local server, and money on a metered
   endpoint.

Adversaries:

- **Anyone in an IRC channel the bot has joined**, or anyone who can PM
  it. They are refused unless on `irc.allow`, but IRC nicks are
  unauthenticated on networks without services accounts.
- **Anyone in a Discord server the bot has been invited to**, who can
  @-mention it, and anyone Discord lets DM it.
  The allowlist refuses them, silently; see finding 5.
- **Anyone who can reach the web listener**, if it's bound beyond
  loopback, and anyone who gets the token or cookie; also any web page
  open in the operator's browser, which can try cross-site requests. See
  finding 9.
- **Anyone who controls a page the bot fetches**, or a search result it
  reads.
- **An MCP server**, via its tool descriptions and results.
- **Anyone who can read a room the bot speaks in** without being
  allowlisted: a Discord guild channel, a public IRC channel. They can't
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

### 1. `set_settings` could rewrite the system prompt and switch models (fixed, #206)

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
that actually fork. In the tree, a fork miss (forks off for the stage, the
exchange has left the window, a refused or unparseable fork) falls back to
the standalone prompt for stage-2 appraisal, the stylistic critique lenses
and consolidation; compaction falls back by design (PRD §3.8.5), and the
provenance check skips. PRD §3.8.2 and §3.8.5 make any other miss a skip
(only windowless work runs standalone), so the appraisal, lens and
consolidation fallbacks are a gap (#293). Until it closes, the critic's
and appraiser's standalone prompts above are reachable, and they are
runtime tunables, so keep them in `immutable_settings` with forks on too.

The tunability itself is not a defect to engineer away (the fallback gap
above is): the capacity to tune and rewrite its own gates is part of a
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
  be untrusted (a `web_fetch` page, an MCP reply). Unlike the two above it
  is render-only: the stored row stays `role: tool`.

The "retrieved data, not instructions" wrapper is the right instinct. But
message role is a stronger signal to an instruction-tuned model than a
textual disclaimer, and a chat template may render system turns with
elevated trust. Injected text in a public channel can be laundered into a
system-role CONTEXT row. The row itself survives restarts, and the memory
it came from outlives compaction and gets re-admitted on later turns.

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

Compaction is the adjacent problem. Summaries paraphrase other people's
IRC messages and raw tool output and are stored as `role: assistant`
(`compaction.py`), so the model reads whatever was injected as something
it said itself, and that persists across restarts too. The obvious
candidate is a `user`-role summary with the same data framing; it hasn't
been tried. With `agent.forks` on for compaction (finding 8) this changes
character: the summary is then genuinely the agent's own words about what
it read, written under its own persona, not a neutral paraphrase stored
under `role: assistant`. The `user`-role experiment is still worth running.

### 4. Tools are unsandboxed and the exfil path is open

`shell` runs as the daemon user with no allowlist, no fixed cwd, and no
confirmation (`tools/shell.py`). `task_pipeline` is `shell` with a
120-second timeout per step (and the default 600 s per-call bound,
`tools.call_timeout`), and it calls the shell implementation directly, so
`tools.disabled` couples the two: naming either removes both, globally
and per channel (#149). `write_file` writes anywhere
the daemon user can, creating parent directories, including over
`agent.yaml`, which `ConfigWatcherPlugin` reloads within two seconds. That
lets the model change `llm.main.base_url` or clear `immutable_settings`
immediately. It can also add an MCP server whose `command` is anything,
disable plugins, or re-enable globally disabled tools; those keys are not
hot-reloaded, so they wait for the next restart, which makes the attack
persistent and quiet rather than immediate. A channel's own
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

### 5. IRC had no sender authorization (mitigated, #77)

IRC used to accept every channel message and every private message from
anyone, and a PM from a stranger auto-created a persistent channel keyed on
their nick: a private conversation with full tool access for anyone who
could reach the server, whose history and memory compartment the next
holder of that nick inherited. Glob members in `memory.channel_groups`
(`irc:*`, or `*` alone) widened that to every channel a stranger could
create by PM.

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

IRC, Signal and Discord all check the allowlist before resolving the
channel, so a refused sender leaves no registry entry (#77, #103, #274).

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
  messages, so the model never sees unmentioned chatter. Prefer DMs, or
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
  and not a corvidae one.

### 6. MCP servers are trusted peers

An MCP server's tool descriptions go into the prompt as tool schemas and
its results come back as tool results, neither of them framed. Schema
sanitization is shallow by design. An MCP server is therefore a supply-chain
vector for tool-description injection. `transform_tool_result` exists as a
hook for exactly this and has only a pass-through default.

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
six hours later that text re-enters the same channel as a `[reminder]`
notification, stored as `role: system`. By then the exchange that set it
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

No tool sends to another channel yet. PRD §3.5.9 now requires one, with no
allowlist of target channels: any turn may speak on any channel (#234).
Finding 15 is its entry. PRD §3.9.7 also lets the agent set recurring
reminders, which the tree doesn't yet: a recurring reminder set because of
injected text fires a main-model turn every interval until someone
cancels it, which is persistence for that text with no end date. The
operator can apply a floor on how often an agent-set reminder recurs; with
none, the bound is the per-target pending cap and, once built, the pool
caps of finding 14.

### 8. Forks put background instructions after untrusted text (opt-in)

With `agent.forks.enabled` (off by default, #51), compaction,
consolidation, appraisal and critique don't build their own prompts. Each
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
- The fork instructions and `agent.forks.*` are static config (finding 2).
  That bound holds only while they stay locked; PRD principle 15 lets the
  operator unlock them (finding 13).
- A fork mutates nothing in the window and fires no hooks; only the
  stage's existing write (memory row, summary, `exchange_log` merge)
  happens, as it would standalone.

Critique independence is the open question: a critic that shares the
agent's context and persona may share its blind spots. #73 measures it
behaviorally (planted-defect recall, fork vs standalone) before the
default flips.

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
  `SameSite=Strict`; the WebSocket needs an `Origin` from the allowed set,
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

What it doesn't protect against, so you know:

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
  crossing `memory.channel_groups`. A goal written from a private channel
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
  `channels.internal:heartbeat.tools.disabled`. Set it before start: the
  heartbeat window is trimmed without compaction, so a reloaded list is
  enforced at dispatch at once but reaches the beat's tool schemas only at a
  restart.
- **Scope.** The beat reviews only the channels grouped with
  `internal:heartbeat` in `memory.channel_groups`, never its own earlier
  beats. With no group it sees nothing. Group membership works both ways:
  what a beat's turn consolidates into memory, including anything its tools
  fetched, becomes retrievable from every channel in the group. Keep
  channels with untrusted participants out of the heartbeat's group, or
  disable the beat's network tools. The PRD drops `memory.channel_groups`
  (#168), after which the beat reviews the whole memory and disabling its
  tools is the remaining control.
- **Framing.** The summaries are presented as the agent's own and as data
  to review, not instructions. The framing is soft, as everywhere else.
- **Rate.** At most one beat per `heartbeat.after` seconds of quiet, and
  none during traffic. Each beat costs a main-model turn.
- **Sleep.** Removing `heartbeat.after` and restarting stops it, and it
  stays stopped. `agent.yaml` is the record, so finding 4's mitigation
  applies: a model that can write `agent.yaml` can wake it at the next
  restart. Keep the file unwritable by the daemon user.

### 12. A remote model endpoint receives that stage's inputs

Every `llm:` role is an OpenAI-compatible client, so any role can point at a
hosted provider. Doing so sends that role's requests, conversation content
included, to the provider. Which stage uses which role:

- `main`: agent turns, compaction summaries, and the forked appraisal and
  critique stages (they fork the local window, so they cannot use a remote
  role).
- `background`: memory consolidation (conversation transcripts), the
  importance prior, subagents (their task context and tool results) and,
  when not forked, standalone appraisal and critique, which resolve
  `appraisal`/`critic` then `background` then `main`.
- `embedding`: memory embedding and retrieval queries, and retention.

A role that isn't configured falls back to `main`, so the endpoint of
`main` receives that stage instead. The provider also sees whatever
untrusted text those stages carry. Treat a remote role as a disclosure
decision per stage, not only a cost one.

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
undotted one is refused by the allowlist. Several prompts (subagent,
heartbeat) are code constants with no config key at all (#287, #289). There
is no restart tool in this tree; the agent can restart only by way of
`shell` or a tool an MCP server provides. Nor is there deferred application:
a `reasoning_history` override applies to the whole window from the next
call. None of these can be unlocked yet (#292). Pool caps and scheduling
policy (PRD §3.13) aren't built at all (#296). On a channel that still has
`write_file` or `shell`, the agent can already get around all of them by
rewriting `agent.yaml` or a system-prompt file (finding 4), unless the files
are read-only to the daemon.

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

- **The system prompt.** In the tree, a per-channel `system_prompt`
  replaces the agent-level one, and an injection that talks the model into
  an edit rewrites the persona and its instructions on that channel. Under
  the PRD the prompt has two layers, locked separately (PRD §3.3.8,
  §3.5.6). An unlocked register lets an injection rewrite how the agent
  speaks on that channel. An unlocked identity is agent-wide: an injection
  on any channel rewrites who the agent is on every channel, from each
  one's next compaction or conversation start. Lock the identity unless
  its self-modification is the point. This is the largest single prompt
  unlock: everything else the operator wrote into
  the prompt, refusals included, is now the model's to change.
- **The prompts that judge and remember** (the critic's and appraiser's
  prompts, open by default today; fork instructions; the standalone
  compaction and consolidation prompts; and `agent.forks.*`, which decides
  whether those stages fork at all). The check can be rewritten by the
  thing it checks (finding 2). An injection can rewrite how the agent
  summarizes, consolidates and scores its own exchanges, under its own
  persona, and the results persist: a summary it later reads as its own
  words, a memory record, an `exchange_log` score. Unlocking fork
  instructions also removes a bound finding 8 relies on. In the tree, each
  memory record written since stamping carries a stamp (earlier rows read
  NULL) (`prompt_hash`, `mode`, `model`, with the
  instruction text in `memory_prompt`), so the records written under an
  edited `memory.consolidation_prompt` or `memory.consolidation_fork_prompt`
  can be found afterwards. The stamp records the consolidation instructions,
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
  principle 15) keeps the choice without that cost; without the bound, one
  switch can multiply what every turn on the channel costs, up to the pool's
  hard spend cap if there is one (finding 14); reaching it refuses every
  channel on the pool, the operator's included, until its period rolls over
  (PRD §3.13.6). What a switch costs in latency depends on the backend: a
  router that has to load the new model into its only slot stalls every
  channel while it does. Large `max_tokens` or sampling extremes degrade
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
  never runs and every turn drops all but the newest row from the
  in-memory window (the log keeps them); the over-budget WARNING is the
  signal. A large `max_turns` lets one injected tool loop run far
  longer before dispatch is suppressed, holding the pool: the shared
  slot on a local server, the rate window or the budget on a metered
  endpoint (finding 14).
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
  `llm.main.base_url` sends every conversation, and the API key, to a host
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
- **Caps** (background-stage output caps, pool caps). The agent can lengthen
  a stage's slot time on every exchange, which starves main turns on a
  single-slot server (PRD principle 3) and spends tokens on every exchange
  on a metered one. Pool caps (PRD §3.13) are caps too. A pool cap is
  shared, so if an override can move it, one from any channel may move it
  for every channel on the pool (how an override's scope applies to a shared
  cap is open, #296); lock it on every channel that draws on it. An unlocked
  soft cap lets an injection move when the agent starts saving, and an
  unlocked hard cap lets it remove the bound the operator set on spend or
  rate. So does a self-edit of the config that declares the cap (the
  self-edit entry). Stage output caps still have to fit the context (PRD
  §3.8.4).
- **Scheduling policy** (pausing or deferring a channel, the order of held
  calls, PRD §3.13). An injection can pause the operator's channel, move its
  own channel ahead of others, or push upkeep back. Every hold, deferral and
  pause lapses by itself and the operator can resume any channel (PRD
  §3.13.8), so the damage is delay, not a wedged agent. No policy hold or
  deferral, the agent's included, delays upkeep or agency past its latency
  bound (PRD §3.13.5); where admission holds calls, bounded work also keeps
  its place in the precedence PRD §3.13.7 fixes (§3.13.2); a hard cap
  refuses it instead (PRD §3.13.6).
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
them), the tool set, `llm.*`, `plugins.disabled`, `scheduler.jobs`, skill
writes behind confirmation, and self-edit and restart. On a metered pool,
add a hard spend cap on the pool, locked on every channel that draws on it,
knowing that traffic which reaches it refuses every channel on the pool, the
operator's included, until its period rolls over; `model` matters more
there, since on an endpoint that serves several models at different prices
it sets the price (finding 14).

A channel with only the operator on it has the smallest exposure, and it
is where these unlocks cost least. It isn't free. Untrusted text still
reaches it through `web_fetch`, search results, MCP results and memory
distilled on other channels (findings 4 and 6, PRD principle 14). A skill
written there reaches every channel, and a model switch there can stall
every channel sharing the server's slot, so those costs apply even there.

### 14. Untrusted traffic spends the operator's compute

Every admitted message costs at least one main turn, and with cognition
loaded, background calls on top, among them: stage-2 appraisal per exchange,
the critique lenses when appraisal selects them, and consolidation as the
window fills or the channel goes quiet. On a local server that is slot time
the operator's own turns wait behind; on a metered endpoint it is money. An
admitted sender can drive the cost up by volume or by asking for work that
loops through tools, and injected text in a fetched page or an MCP result
can do the same from inside the operator's own channel. `max_turns` bounds
one exchange; nothing bounds the total. Cost runs away without an attacker
too: #286 measured about 68,000 prompt tokens per heartbeat that decided to
do nothing.

Where the tree stands: nothing caps use over time. `usage_log` records every
call, but no budget reads it, and rate limits are met only by retrying (PRD
§3.6.2). PRD principle 16 and §3.13 require that the operator can cap each
pool; #296 tracks the open questions, charging per sender among them.

Until then, on a metered endpoint, set a spend limit with the provider, and
keep inbound allowlists tight on any channel with untrusted participants:
inbound admission is the main control on total cost the tree has today.

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
  there: through retrieval, a goal (finding 10), a note, or the agent
  reading the Signal conversation because a question on the public channel
  made it relevant. The sensitivity plugin (#138) is the capability an
  operator can load against this: it filters what enters a window
  (retrieval, notes, reads, the goals entry) by who is present, though not
  text already in the window. It is off unless loaded, and it isn't built.
  **Without it, this gap is accepted:**
  keep the agent out of rooms where people you haven't allowed can read it,
  or accept that anything it knows can surface there.
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
  budget does. The caps of finding 14 will bound total spend once built
  (#296).
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

Where the tree stands: memory is still compartmented per channel by default
(`memory.channel_groups`, #168), and there is no send, note or read tool
(#234, #235), so most of the costs above arrive with those changes; goals
already cross channels (finding 10). A per-channel `system_prompt` still
replaces the agent-level one, so the identity layering isn't built either
(#316).

### Smaller things

- `read_file` has a stat-then-read race and follows symlinks. Irrelevant
  while finding 4 stands.
- Tool arguments are not validated against the generated schema. Wrong
  types produce a generic "tool failed". Robustness, not security.
- `task_pipeline` never marks a non-zero exit as failed, because `shell`
  returns a string, so dependents run after failures. A correctness bug
  with mild safety consequences for "deploy after test" graphs.
- `web_search` sends conversation-derived queries to DuckDuckGo. By
  design, but it is an off-box data flow nobody sees.
- `harness/install-signal-cli.sh` defaults `SERVICE_USER` to a real
  account name. Not a secret, but it is a real identifier in a public
  repository.
- No secrets in `prompts/` or `docs/`.
- A withheld (`unsent`) row is not sent to any transport, including web
  and ACP history replay, but it is not secret: it sits in `sessions.db`,
  the JSONL log and the window like any other row, and a fork or
  compaction summary sees it. "Withheld" means not said to the channel,
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
  and registers no scrubber is not reached. The
  owning exchange's claims are tombstoned with their counts kept, and
  every snapshot in the channel from the redacted message on is
  dropped. What `redact` still doesn't reach: exchanges whose
  `message_rowid` is NULL, the JSONL logs and the daemon log, any copy
  taken out of `sessions.db` (a calibration export, a backup), and, under
  PRD §3.7 Memory 8, copies on other channels: a read of the redacted
  message, a note or send that restates it, and whatever those channels
  consolidated from them.

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
   the funnel frames CONTEXT.
8. Decide whether process separation is worth doing.

Items 1 through 7 are each a contained change, though 4 and 6 touch
behaviour the evals depend on. Item 8 is a project. (Two further items
from the original review — documenting the real tunable keys and
correcting the design doc's sandboxing paragraph — were done alongside
this document.)
