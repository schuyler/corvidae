# corvidae

Corvidae is a minimalist, extensible architecture for building LLM agent harnesses. It connects a local LLM to chat transports — CLI, IRC, Signal, Discord, and a browser UI are built in — routes messages through any OpenAI-compatible Chat Completions API, runs a multi-turn tool-calling loop, and sends replies. It's designed as a personal agent daemon for a single machine.

**WARNING: This is a personal project, not a formal release. Interfaces will change without warning.**

## Design

[docs/prd.md](docs/prd.md) is the requirements of record: what has to be true and why. [docs/design.md](docs/design.md) is how the current tree does it.

Three layers: a plugin system ([apluggy](https://github.com/nextline-dev/apluggy)), an agent loop, and transport plugins. Each transport converts platform-specific messages to a common Channel abstraction. The agent loop handles prompt construction and LLM interaction. The plugin system wires everything together via lifecycle hooks.

The distribution is split the same way. `uv sync` installs the kernel: the agent loop, persistence, compaction, tools and transports. The cognition plugins — long-term memory, appraisal, critique, the context funnel and the outcome log — are a separate package in this repository, installed with `--extra cognition` ([docs/cognition.md](docs/cognition.md)).

Any plugin can be switched off with `plugins.disabled` in `agent.yaml`, but plugins declare what they depend on and startup stops with an error if a dependency is missing: disabling PersistencePlugin, for instance, also means disabling SchedulerPlugin, UsageLogPlugin and the cognition plugins that need the database. The graph is in [docs/design.md](docs/design.md#current-dependency-graph). Without CompactionPlugin, history grows unbounded; without ThinkingPlugin, `<thinking>` blocks pass through to the channel.

The agent loop uses single-turn dispatch: one LLM call per queue item. Tool calls become independent async tasks. When a task completes, the result arrives as a notification that feeds back into the queue, triggering the next LLM call. Multi-turn reasoning emerges from this cycle rather than from a blocking loop. This keeps the channel queue responsive — user messages are never stuck behind a long tool chain.

## Synopsis

```yaml
# agent.yaml — minimal working config
llm:
  main:
    base_url: http://localhost:8080
    model: llama3

daemon:
  session_db: sessions.db

agent:
  system_prompt: "You are Corvus. You speak only in riddles."
  max_context_tokens: 24000
```

```sh
uv run corvidae          # reads agent.yaml from cwd
```

The daemon connects to configured transports, listens for messages, and runs until SIGINT or SIGTERM.

For a chat page in your browser, add a `web:` block (an empty one will do) and open the login URL the daemon logs. It listens on `127.0.0.1:8765`, and its token is shell access to the machine, so read the [`web` section](docs/configuration.md#web--browser-chat-ui) before you point it anywhere else.

## Installation

Clone the repo — corvidae isn't published on PyPI.

```sh
uv sync --extra cognition    # the daemon with memory, appraisal and critique
uv run corvidae
```

`uv sync` alone installs the kernel: the agent loop, persistence, tools and
the transports, without the cognition plugins or sqlite-vec. Add
`--extra discord` for the Discord transport and `--extra acp` for
`corvidae acp`, on the same line: `uv sync` removes any extra you don't
name.

For development (pytest, every extra its tests import):

```sh
uv sync --extra dev --extra acp --extra discord --extra cognition
uv run pytest
```

## Configuration

Corvidae reads `agent.yaml` from the current working directory. The file configures the LLM backend, daemon settings, agent defaults, transports, and per-channel overrides. `agent.yaml.example` covers every built-in plugin; `agent.minimal.yaml.example` disables the memory/appraisal/critique plugins and runs just the core agent loop.

See [docs/configuration.md](docs/configuration.md) for the full reference.

Key sections:

- `llm.main` — required; `base_url` and `model` are the only mandatory fields
- `daemon` — SQLite path, task worker count
- `agent` — system prompt, token budget, compaction, tool result limits
- `irc` — omit entirely to disable IRC
- `channels` — per-channel overrides keyed by `transport:scope` (e.g., `irc:#general`, `cli:local`)
- `mcp` — MCP server connections (optional)
- `scheduler` — reminder timezone, per-channel reminder cap, and operator jobs such as a heartbeat (optional)
- `logging` — simplified options (`level`, `file`); see docs/configuration.md

Before pointing the bot at a channel with people you don't trust, read
[docs/threat-model.md](docs/threat-model.md). The built-in tools run
unsandboxed as the daemon user, and prompt injection is the threat, not
other tenants.

## Features

- IRC transport via pydle; CLI transport (stdin/stdout) built in
- Multi-turn tool-calling loop: tool calls run as async tasks, results trigger the next turn
- Context window management: estimates token count, compacts (summarizes) when nearing the budget
- SQLite conversation persistence with an append-only message log
- `<thinking>` block handling: stripped from display, optionally retained in DB
- MCP (Model Context Protocol) client: connects to external tool servers over stdio or SSE
- Per-channel config overrides for system prompt, token budget, turn limits, and more
- Runtime settings tool: the LLM can adjust its own parameters mid-session (with optional immutable keys)
- Plugin system via apluggy (async pluggy) with setuptools entry points

## Included plugins

### Kernel plugins

| Plugin | Purpose |
|--------|---------|
| PersistencePlugin | SQLite database lifecycle and per-channel conversation initialization |
| CoreToolsPlugin | Registers the built-in tools (shell, read_file, write_file, web_fetch) |
| CLIPlugin | stdin/stdout transport (routes to `cli:local`) |
| IRCPlugin | IRC transport via pydle with reconnect backoff |
| TaskPlugin | Async task queue for tool call dispatch, configurable concurrency |
| SubagentPlugin | Launches background agents with their own LLM sessions |
| McpClientPlugin | Connects to external MCP servers and exposes their tools |
| CompactionPlugin | Summarizes older messages when nearing the token budget |
| ThinkingPlugin | Strips `<thinking>` blocks from display, optionally from prompt history |
| RuntimeSettingsPlugin | Lets the LLM adjust its own per-channel parameters mid-session |
| SchedulerPlugin | Reminders the agent sets with `remind_me`, operator-scheduled jobs, and a heartbeat, fired into channels on the clock |
| SignalPlugin | Signal transport via a running `signal-cli` daemon |
| DiscordPlugin | Discord transport (needs `--extra discord`) |
| WebPlugin | Browser chat UI on localhost |
| AcpPlugin | Agent Client Protocol over stdio for editors (`corvidae acp`, needs `--extra acp`) |

### Cognition plugins (`--extra cognition`)

| Plugin | Purpose |
|--------|---------|
| MemoryPlugin | Consolidates compacted conversation into long-term memories and retrieves them into context |
| MemoryToolsPlugin | `search_memory` and `recall_raw` tools |
| FunnelPlugin | Admits retrieved context into the prompt under a budget |
| AppraisalPlugin | Scores each exchange as it happens |
| CritiquePlugin | Reviews the agent's replies after the fact |
| OutcomeLogPlugin | Records each exchange and its outcome |

## CLI subcommands

`corvidae scaffold <plugin-name> [-o/--output-dir <dir>]` generates a new plugin project directory containing `pyproject.toml`, a plugin module, and a test file. The plugin name is lowercased and non-alphanumeric runs are replaced with underscores to form the package name.

Additional subcommands can be added by plugins via the `corvidae.commands` entry point group. See [docs/plugin-guide.md](docs/plugin-guide.md#registering-cli-subcommands) for details.

## Plugin system

External plugins register via setuptools entry points under the `corvidae` group. See [docs/plugin-guide.md](docs/plugin-guide.md) for the hook API and a worked example.

## Built-in tools

The LLM has access to these tools out of the box:

- `shell` — run a shell command, capture stdout/stderr
- `read_file` — read a file from the local filesystem
- `write_file` — write or overwrite a file
- `web_fetch` — fetch a URL and return the response body
- `web_search` — DuckDuckGo search, results as title/URL/snippet
- `task_pipeline` — run a YAML/JSON DAG of shell commands in dependency order
- `task_status` — check status of background tasks
- `set_settings` — adjust agent parameters (the `system_prompt` key is blocked, but read docs/threat-model.md finding 1 before trusting that; other keys configurable as immutable)
- `subagent` — launch a background agent with its own LLM session

MCP servers add their tools to this list at startup.

## Inspiration

Corvidae was inspired by [pi.dev](https://pi.dev/) and [mira](https://github.com/taylorsatula/mira-OSS).

## License

It's MIT licensed, yo. See LICENSE for details.
