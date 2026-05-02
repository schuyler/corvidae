# corvidae

Corvidae is a minimalist, extensible architecture for building LLM agent harnesses. It connects a local LLM to chat transports — IRC and CLI are built in — routes messages through any OpenAI-compatible Chat Completions API, runs a multi-turn tool-calling loop, and sends replies. It's designed as a personal agent daemon for a single machine.

**WARNING: This is a personal project, not a formal release. Interfaces will change without warning.**

## Design

Three layers: a plugin system ([apluggy](https://github.com/nextline-dev/apluggy)), an agent loop, and transport plugins. Each transport converts platform-specific messages to a common Channel abstraction. The agent loop handles prompt construction and LLM interaction. The plugin system wires everything together via lifecycle hooks.

Every built-in plugin is optional. Without PersistencePlugin, messages drop but the daemon doesn't crash. Without CompactionPlugin, history grows unbounded. Without ThinkingPlugin, `<thinking>` blocks pass through to the channel. The system is designed to work with any subset of its parts — add what you need, leave out what you don't.

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

## Installation

Clone the repo — corvidae isn't published on PyPI.

```sh
uv sync
uv run corvidae
```

For development (includes pytest):

```sh
uv sync --extra dev
uv run pytest
```

## Configuration

Corvidae reads `agent.yaml` from the current working directory. The file configures the LLM backend, daemon settings, agent defaults, transports, and per-channel overrides.

See [docs/configuration.md](docs/configuration.md) for the full reference.

Key sections:

- `llm.main` — required; `base_url` and `model` are the only mandatory fields
- `daemon` — SQLite path, task worker count, idle behavior
- `agent` — system prompt, token budget, compaction, tool result limits
- `irc` — omit entirely to disable IRC
- `channels` — per-channel overrides keyed by `transport:scope` (e.g., `irc:#general`, `cli:local`)
- `mcp` — MCP server connections (optional)
- `logging` — simplified options (`level`, `file`); see docs/configuration.md

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
| IdleMonitorPlugin | Fires `on_idle` when all queues are quiescent |

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
- `task_status` — check status of background tasks
- `set_settings` — adjust agent parameters (system prompt excluded; other keys configurable as immutable)
- `subagent` — launch a background agent with its own LLM session

MCP servers add their tools to this list at startup.

## Inspiration

Corvidae was inspired by [pi.dev](https://pi.dev/) and [mira](https://github.com/taylorsatula/mira-OSS).

## License

It's MIT licensed, yo. See LICENSE for details.
