# corvidae

Corvidae is a personal agent daemon. It connects an LLM to the chat apps you already use, runs a tool-calling loop on your behalf, and remembers what you talked about.

It talks to any OpenAI-compatible Chat Completions endpoint, hosted or local. It runs as a single Python process, and is meant to run in a container or a VM you control.

**WARNING: This is a personal project, not a formal release. Interfaces and config keys will change without warning.**

## What it does

- **Lives where you chat.** It has transports for a browser chat page, Discord, Signal, Zulip, IRC, a terminal, and editors over the Agent Client Protocol. It's one agent with one identity and one memory across all of them. Each channel can have its own prompt, token budget and tool set.
- **Uses tools.** It has `shell`, `read_file`/`write_file`, `web_fetch`/`web_search`, a DAG runner for shell pipelines, and background subagents with their own LLM sessions. MCP servers (stdio or HTTP) add their tools at startup. Tool calls run as async tasks, so a long tool chain never blocks new messages.
- **Remembers.** Conversations persist in SQLite and survive restarts. When a conversation nears its token budget, older turns get summarized. Long-term memory consolidates conversations into records it retrieves later, using vector search if you give it an embedding model and keyword search if you don't.
- **Thinks about how it's doing.** Appraisal scores each exchange, critique reviews replies after the fact, and an outcome log records what happened.
- **Acts on its own, if you let it.** It can set reminders for itself, and you can schedule jobs for it. It can keep agent-wide goals. An optional heartbeat starts a turn of its own after a quiet spell. It can also decide a message needs no reply and stay silent.
- **Bends to fit.** Everything is a plugin, and any plugin can be switched off in config. Most config changes reload live without a restart. `corvidae scaffold` starts a new plugin for you.

## Quick start

You need an OpenAI-compatible endpoint URL, a model name and an API key from your provider. To use a model on your own hardware instead, read this section and then [Local models](#local-models).

### Try it

**This is a demo and development setup, not a deployment.** It runs in your terminal, as your own user, against your own files, and it stops when you close the terminal. It's the quickest way to see corvidae working. To keep it running, use [Docker](#run-it-in-docker).

You need git and [uv](https://docs.astral.sh/uv/). uv fetches the Python 3.13 that corvidae needs if you don't have it. Corvidae isn't on PyPI, and the `corvidae` package you'll find there is an unrelated project, so clone it:

```sh
git clone https://github.com/schuyler/corvidae.git
cd corvidae
uv sync --extra cognition
```

Put this in `agent.yaml` in the checkout:

```yaml
llm:
  main:
    base_url: https://api.example.com/v1   # your provider's OpenAI-compatible endpoint
    model: your-chat-model
    api_key: "${LLM_API_KEY}"              # read from the environment, quotes and all

agent:
  system_prompt: "You are a helpful assistant."

tools:
  disabled: [shell, write_file]            # see "Giving it tools" below

web: {}                                    # the browser chat page, on 127.0.0.1:8765
```

Start it:

```sh
LLM_API_KEY='sk-...' uv run corvidae serve
```

Among the startup lines is a warning that says `web: log in at http://127.0.0.1:8765/?token=...`. Open that URL and say hello. Ctrl-C stops it. The conversation is saved in `sessions.db`, and scratch files go in `state/`, both in the current directory. Start it again and the conversation picks up where you left off.

`--extra cognition` adds long-term memory, appraisal, critique, goals and the rest of the [cognition plugins](docs/cognition.md) to the kernel (the agent loop, persistence, tools and transports). Two things to know about them:

- **They cost money.** Memory consolidation, appraisal and critique each make model calls of their own, in the background. On a metered API that adds up. To run just the core loop, copy the `plugins.disabled` list from [`agent.minimal.yaml.example`](agent.minimal.yaml.example).
- **Without an embedding model,** memory falls back to keyword search. A startup warning about the missing embedding profile is telling you that. [Local models](#local-models) shows how to add one.

### Run it in Docker

This is the recommended way to run corvidae for real. The container limits what the agent's tools can reach to the one directory you give it, and Docker restarts the daemon if it stops.

There's no published image yet ([#731](https://github.com/schuyler/corvidae/issues/731)), so build one from the checkout:

```sh
docker build -t corvidae .
```

The image includes every optional plugin. It runs `corvidae serve` as uid 1000 in `/data`, which is where you mount a directory that holds `agent.yaml`. The database, memory and state files get written beside it:

```sh
mkdir -p ~/corvidae-data
```

Copy your demo config there, and change the `web:` block:

```sh
cp agent.yaml ~/corvidae-data/agent.yaml
```

```yaml
web:
  host: 0.0.0.0                            # inside the container; -p below keeps it on loopback
  allowed_origins:
    - http://127.0.0.1:8765                # the address you'll browse to
```

Inside a container, the page has to listen on every interface for Docker's port mapping to reach it. A wildcard bind can't work out which address you'll browse to, so you name it in `allowed_origins`.

Then hand the directory to the container's user and start it:

```sh
sudo chown -R 1000:1000 ~/corvidae-data
docker run -d --name corvidae --restart unless-stopped \
  -p 127.0.0.1:8765:8765 \
  -e LLM_API_KEY='sk-...' \
  -v ~/corvidae-data:/data \
  corvidae
docker logs corvidae 2>&1 | grep 'log in at'
```

That prints a login URL with a token in it. Open it and say hello. The token sets a cookie in your browser, so you don't paste it again. Corvidae generates a new token every time it starts, though, so a restart logs you out and you fetch the new URL the same way. [Keep your login](#keep-your-login) fixes that once you're past the first run.

**The token is the agent.** Whoever has it can tell the agent to do anything its tools allow. That's why the port is published on loopback only, and why the config starts with the two most dangerous tools turned off. To reach the page from another machine, use an SSH tunnel (`ssh -L 8765:127.0.0.1:8765 yourhost`), Tailscale, or a TLS reverse proxy, and add that origin to `allowed_origins` ([details](docs/configuration.md#web--browser-chat-ui)).

If it doesn't work, `docker logs corvidae` is the first place to look:

| Symptom | Cause |
|---|---|
| The page says `403 bad host` | The address in your browser isn't in `allowed_origins`. Add it (e.g. `http://localhost:8765`) and `docker restart corvidae`. |
| `unable to open database file` at startup | uid 1000 can't write `~/corvidae-data`. Redo the `chown`. |
| `llm.main.api_key` error at startup | `LLM_API_KEY` is unset or empty in the container. |
| Every reply is an error about the model | `base_url` or `model` is wrong, or the key is. `base_url` usually ends in `/v1`. |

### Add Discord

Corvidae logs in as a Discord bot. Follow [docs/discord-ops.md](docs/discord-ops.md) through "Find ids" to create the application, get its bot token, invite it to a server you're in, and find your own numeric user id.

Add a `discord:` block to `agent.yaml`:

```yaml
discord:
  allow:
    - 123456789012345678    # your numeric user id, not your username
```

`allow` is default-deny, and it refuses quietly: if your id isn't on the list, the bot simply ignores you.

The bot token goes in the environment, not the file. `docker restart` keeps a container's original environment, so recreate the container instead, with the same command as before plus one line:

```sh
docker rm -f corvidae
docker run -d --name corvidae --restart unless-stopped \
  -p 127.0.0.1:8765:8765 \
  -e LLM_API_KEY='sk-...' \
  -e DISCORD_TOKEN='your-bot-token' \
  -v ~/corvidae-data:/data \
  corvidae
```

That's a fresh start, so the web page has a new login URL in `docker logs` too. The log should also say `discord: connected as user id <id>`. Two warnings that `voice will NOT be supported` are expected. DM the bot and it answers everything you send. In a server channel it answers only messages that @-mention it, unless you change `require_mention` ([docs/discord-ops.md](docs/discord-ops.md#answering-without-a-mention-experimental)).

Your conversation history moves with you: the web page and Discord are separate channels, but the agent's memory covers both.

### Keep your login

A generated web token lasts until the next restart. To keep your browser logged in across restarts, choose a token yourself and pass it in the environment. Corvidae then uses it instead of generating one:

```sh
TOKEN=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
echo "$TOKEN"     # save it somewhere safe, like your password manager
docker rm -f corvidae
docker run -d --name corvidae --restart unless-stopped \
  -p 127.0.0.1:8765:8765 \
  -e LLM_API_KEY='sk-...' \
  -e DISCORD_TOKEN='your-bot-token' \
  -e CORVIDAE_WEB_TOKEN="$TOKEN" \
  -v ~/corvidae-data:/data \
  corvidae
```

Log in once at `http://127.0.0.1:8765/?token=<your token>`. The cookie lasts 400 days, as long as the token stays the same. Changing the token is also the only way to revoke a session, and it logs out every browser at once. Once you set a token, the daemon stops logging the login URL.

### Giving it tools

The quick-start config disables `shell` (which also removes `task_pipeline`) and `write_file`. Here's why both matter:

- `shell` runs any command as the daemon's user, with no allowlist and no confirmation.
- `write_file` can rewrite `agent.yaml`, which reloads within two seconds. From there the agent can turn its own tools back on, or add an MCP server that runs any command it likes.

Without both, the agent can't run commands, but it can still read files and fetch URLs. So a prompt-injected agent can still read its own environment variables and send them somewhere. The [threat model](docs/threat-model.md) covers this in full; read findings 4 and 9 at least.

When you're ready, remove entries from `tools.disabled`, or narrow them per channel with `channels.<id>.tools.disabled`. If you turn `write_file` back on, consider mounting the config read-only so the agent can't edit it: add `-v ~/corvidae-data/agent.yaml:/data/agent.yaml:ro` to `docker run`. A container limits what the tools can reach. It does not sandbox them from the daemon.

## Local models

Corvidae doesn't care what serves the model, as long as it speaks OpenAI's Chat Completions API. The common local options:

| Server | `base_url` from the host |
|---|---|
| llama.cpp's `llama-server` | `http://127.0.0.1:8080/v1` |
| Ollama | `http://127.0.0.1:11434/v1` |
| vLLM | `http://127.0.0.1:8000/v1` |

Local servers usually need no `api_key`, so leave it out.

**Reaching the host from Docker.** Inside a container, `127.0.0.1` is the container itself. You have two options:

- **Share the host's network.** Run with `--network host` in place of the `-p` line. Use the `127.0.0.1` URLs above, and write the web block as plain `web: {}`. The defaults bind loopback, and the wildcard plus `allowed_origins` would expose the page on every host address.
- **Keep the bridge network.** Add `--add-host host.docker.internal:host-gateway` and use `http://host.docker.internal:8080/v1`. On Linux, that address is the Docker bridge, so a model server listening only on `127.0.0.1` can't be reached through it. Make it listen on an address the container can reach, and mind your firewall.

**Single-slot servers.** Memory, appraisal and critique make model calls in the background. On a server that handles one request at a time, those calls queue behind your chat turns. Either give the server more slots, or point those stages at a second, smaller model with `llm.stages` ([docs/configuration.md](docs/configuration.md#stages)).

**An embedding model** turns on vector search for memory. Add a second profile:

```yaml
llm:
  main:
    base_url: http://127.0.0.1:8080/v1
    model: your-chat-model
  embedding:
    kind: embedding
    base_url: http://127.0.0.1:8081/v1   # llama-server with --embeddings, or the same Ollama
    model: your-embedding-model
    dimensions: 1024                     # must match the model exactly
```

Get `dimensions` right before the first start. Corvidae records the model and dimensions in its database. If you change either one later, vector memory switches off until you put the old values back. [docs/discord-remote-api.md](docs/discord-remote-api.md#measure-the-embedding-dimensions) shows how to ask the endpoint for the number.

## Installing without Docker

This is [Try it](#try-it) made permanent: a dedicated user, every plugin, and a systemd unit. It also works as a recipe for a fresh Debian or Ubuntu LXC container or VM, which gives you most of Docker's containment without Docker. Run this part as root:

```sh
apt-get update && apt-get install -y git curl ca-certificates
useradd --create-home --shell /bin/bash corvidae
su - corvidae
```

Then, as `corvidae`:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
git clone https://github.com/schuyler/corvidae.git
cd corvidae
uv sync --extra discord --extra cognition --extra restart --extra timepiece
cp agent.yaml.example agent.yaml       # or start from the Try it config
$EDITOR agent.yaml
LLM_API_KEY='sk-...' uv run corvidae serve
```

`uv sync` installs exactly the extras you name and removes any you don't. Running `uv sync --extra discord` after the line above would uninstall the cognition plugins. Always list every extra you want on one line. A bare `uv sync` is the kernel: the agent loop, persistence, tools and the transports, minus the Discord and ACP SDKs. The full list of extras is in `[project.optional-dependencies]` in [`pyproject.toml`](pyproject.toml).

To run it as a service, put the secrets in a mode-0600 env file, one `NAME=value` per line, with no `export` and no quotes. Then add a unit:

```ini
# /etc/systemd/system/corvidae.service
[Unit]
Description=corvidae
After=network-online.target
Wants=network-online.target

[Service]
User=corvidae
WorkingDirectory=/home/corvidae/corvidae
EnvironmentFile=/etc/corvidae/corvidae.env
ExecStart=/home/corvidae/corvidae/.venv/bin/corvidae serve
Restart=always
RestartSec=5
TimeoutStopSec=60

[Install]
WantedBy=multi-user.target
```

`systemctl enable --now corvidae`, then `journalctl -u corvidae -f` to watch it come up. `WorkingDirectory` is where `agent.yaml`, `sessions.db` and `state/` live.

The LXC container, like the Docker one, limits what the agent's tools can reach. That's the main reason to run corvidae in one. Everything in [Giving it tools](#giving-it-tools) applies here too.

## Configuration

Corvidae reads `agent.yaml` from its working directory. Two annotated examples ship in the repo:

- [`agent.yaml.example`](agent.yaml.example) covers every built-in plugin, with most keys commented out at their defaults.
- [`agent.minimal.yaml.example`](agent.minimal.yaml.example) is just the core loop, with the cognition plugins disabled.

The shape of a typical config:

```yaml
llm:                                  # model profiles; main is the default chat model
  main:
    base_url: https://api.example.com/v1
    model: your-chat-model
    api_key: "${LLM_API_KEY}"

agent:
  system_prompt: "You are Corvus. You speak only in riddles."
  max_context_tokens: 64000           # compaction kicks in as history nears this

tools:
  disabled: [shell, write_file]       # tools no channel gets

web: {}                               # transports: present means on
discord:
  allow: [123456789012345678]

channels:                             # per-channel overrides, keyed transport:scope
  discord:*:                          # every Discord channel
    max_context_tokens: 32000
  web:main:
    register_prompt: "This is a browser chat; longer, structured answers are fine."

plugins:
  disabled: [critique, appraisal]     # entry-point names of plugins to skip
```

A few things that aren't obvious from the example:

- **Channel IDs are `transport:scope`:** `web:main`, `discord:<channel id>`, `discord:dm:<user id>`, `irc:#general`, `cli:local`. They're the keys for history and memory, and they show up in the logs.
- **Every transport that takes messages from other people is default-deny.** Discord, Signal, Zulip and IRC all need an `allow` list.
- **Most changes apply live.** The daemon polls `agent.yaml` and reloads it. Changes to anything under `llm:` and to transport connection settings need a restart, and the log names each key that's waiting for one.
- **A misspelled plugin name in `plugins.disabled` disables nothing.** It logs a warning. The `plugins loaded` line at startup shows what actually loaded.

## Reference documentation

For running it:

- [docs/configuration.md](docs/configuration.md) documents every `agent.yaml` key, plus hot reload and containers.
- [docs/threat-model.md](docs/threat-model.md) covers what untrusted text can reach and what the tools can do. Read it before you add anyone else to an `allow` list.
- [docs/discord-ops.md](docs/discord-ops.md), [docs/signal-ops.md](docs/signal-ops.md) and [docs/zulip-ops.md](docs/zulip-ops.md) are setup guides for each transport.
- [docs/discord-remote-api.md](docs/discord-remote-api.md) walks through a Discord bot on a hosted API, including the embedding step.
- [docs/prompt-guide.md](docs/prompt-guide.md) explains how the prompt gets assembled and how to write a system prompt for it.
- [docs/cognition.md](docs/cognition.md) covers memory, appraisal, critique, goals, the heartbeat and the rest of the cognition set.

For hacking on it:

- [docs/prd.md](docs/prd.md) holds the requirements: what has to be true, and why.
- [docs/design.md](docs/design.md) describes how the current tree does it: the architecture, hook specs and schemas.
- [docs/plugin-guide.md](docs/plugin-guide.md) and [docs/plugin-checklist.md](docs/plugin-checklist.md) cover writing a plugin. `corvidae scaffold <name>` generates the skeleton.

To set up a development checkout:

```sh
uv sync --extra dev --extra acp --extra discord --extra cognition --extra cross-channel-send \
  --extra webhook-notify --extra context-panel --extra restart --extra timepiece --extra inotify
uv run pytest
```

## Inspiration

Corvidae was inspired by [pi.dev](https://pi.dev/) and [mira](https://github.com/taylorsatula/mira-OSS).

## License

It's MIT licensed, yo. See LICENSE for details.
