# Corvidae Configuration Reference

Corvidae is configured via a single YAML file, `agent.yaml`, passed to the daemon at startup. The full parsed dict is available to every plugin via the `on_start(config: dict)` hook.

The key `_base_dir` (a `pathlib.Path`) is injected by `main.py` alongside the parsed config. It points to the directory containing `agent.yaml` and is used to resolve relative file paths (e.g., `system_prompt` file lists).

## Minimal example

```yaml
llm:
  main:
    base_url: http://localhost:8080
    model: my-model

daemon:
  session_db: sessions.db
  max_task_workers: 4
  completed_task_buffer: 100
  idle_cooldown_seconds: 30
  idle_poll_interval: 2

agent:
  system_prompt: "You are a helpful assistant."
  max_context_tokens: 24000
  max_turns: 10
  keep_thinking_in_history: false
  max_tool_result_chars: 100000
  compaction_threshold: 0.8
  compaction_retention: 0.5
  min_messages_to_compact: 5
  chars_per_token: 3.5

tools:
  shell_timeout: 30
  web_fetch_timeout: 15
  web_max_response_bytes: 50000
  max_file_read_bytes: 1048576

irc:
  host: irc.libera.chat
  port: 6667
  nick: corvidae
  tls: false
  channels: []
  message_chunk_size: 400

channels:
  irc:#general:
    system_prompt: "You are the channel bot."
    max_context_tokens: 16000
    max_turns: 10
    keep_thinking_in_history: false
```

---

## `llm` — LLM backend

The `llm.main` block is required. `llm.background` is optional; when absent, subagents use `llm.main`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `llm.main.base_url` | string | — | Base URL for an OpenAI-compatible Chat Completions endpoint. **Required.** |
| `llm.main.model` | string | — | Model identifier sent in each request. **Required.** |
| `llm.main.api_key` | string | `null` | API key for authentication. Omit if the backend does not require one. |
| `llm.main.extra_body` | mapping | `null` | Provider-specific fields merged into the request body (e.g., `id_slot` for llama.cpp). |
| `llm.main.max_retries` | integer | `3` | Maximum retry attempts on transient HTTP errors (429, 500, 502, 503, 504) and connection failures. Set to 0 to disable retries. |
| `llm.main.retry_base_delay` | float | `2.0` | Base delay in seconds for exponential backoff between retry attempts. |
| `llm.main.retry_max_delay` | float | `60.0` | Maximum delay cap in seconds for exponential backoff. |
| `llm.main.timeout` | float | `null` | Total HTTP timeout in seconds per LLM request. `null` uses aiohttp's 300-second default. |
| `llm.background.base_url` | string | — | Base URL for the background (subagent) LLM endpoint. Optional. |
| `llm.background.model` | string | — | Model identifier for subagent calls. Optional. |
| `llm.background.api_key` | string | `null` | API key for the background endpoint. |
| `llm.background.extra_body` | mapping | `null` | Provider-specific fields for background LLM requests. |
| `llm.background.max_retries` | integer | `3` | Maximum retry attempts for background LLM calls. |
| `llm.background.retry_base_delay` | float | `2.0` | Base delay in seconds for background LLM retry backoff. |
| `llm.background.retry_max_delay` | float | `60.0` | Maximum delay cap in seconds for background LLM retries. |
| `llm.background.timeout` | float | `null` | HTTP timeout in seconds for background LLM requests. |

`llm.background` is read by `SubagentPlugin`. If the key is absent, `SubagentPlugin` falls back to `llm.main` for all subagent calls.

Retry behavior applies to transient status codes (429, 500, 502, 503, 504) and connection errors. The `Retry-After` header (429 responses) is honored when present; otherwise exponential backoff is used. Retries do not apply to client errors (4xx) other than 429.

---

## `daemon` — daemon-level settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `daemon.session_db` | string | `sessions.db` | Path to the SQLite database file for conversation history. Relative paths are resolved from the working directory. |
| `daemon.max_task_workers` | integer | `4` | Number of concurrent background task workers in `TaskQueue`. |
| `daemon.completed_task_buffer` | integer | `100` | Number of completed task records retained in memory for `task_status` output. |
| `daemon.idle_cooldown_seconds` | number | `30` | Minimum seconds between consecutive `on_idle` hook firings. |
| `daemon.idle_poll_interval` | number | `2` | Seconds between idle-state checks by `IdleMonitorPlugin`. |

---

## `agent` — agent-level defaults

These values apply to all channels. Per-channel overrides in the `channels` section take precedence where set.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `agent.system_prompt` | string or list of strings | `"You are a helpful assistant."` | System prompt prepended to each LLM call. A list of strings is treated as file paths; each file is read and concatenated with double newlines. Relative paths are resolved against `_base_dir`. |
| `agent.max_context_tokens` | integer | `24000` | Token budget for conversation history. `CompactionPlugin` compacts when the estimated token count exceeds `compaction_threshold` × this value. |
| `agent.max_turns` | integer | `10` | Maximum consecutive tool-calling turns per user message before tool dispatch is suppressed. |
| `agent.keep_thinking_in_history` | boolean | `false` | When `false`, `ThinkingPlugin` strips `reasoning_content` from in-memory assistant messages after they are persisted. Does not affect the DB record. |
| `agent.max_tool_result_chars` | integer | `100000` | Maximum characters in a tool result string before truncation. Applies to both `AgentPlugin` and `SubagentPlugin`. Truncated results have `[truncated — N chars total]` appended. |
| `agent.compaction_threshold` | float | `0.8` | Compaction triggers when `token_estimate / max_context_tokens` exceeds this value. |
| `agent.compaction_retention` | float | `0.5` | After compaction, retain messages that fit within `compaction_retention × max_context_tokens` tokens, counting from the most recent. |
| `agent.min_messages_to_compact` | integer | `5` | Skip compaction if the conversation has this many messages or fewer. |
| `agent.chars_per_token` | float | `3.5` | Character-to-token ratio used by `token_estimate()`. Must match across `CompactionPlugin` and `PersistencePlugin`; configure once here. |
| `agent.immutable_settings` | list of strings | `[]` | Keys the agent is blocked from changing via the `set_settings` tool. `system_prompt` is always blocked regardless of this list. |

---

## `tools` — built-in tool settings

Configured by `CoreToolsPlugin`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `tools.shell_timeout` | integer | `30` | Seconds before a `shell` command is killed. |
| `tools.web_fetch_timeout` | integer | `15` | Seconds before a `web_fetch` HTTP request is aborted. |
| `tools.web_max_response_bytes` | integer | `50000` | Maximum bytes of HTTP response body returned by `web_fetch`. Content beyond this limit is truncated with `[truncated]`. This limit is independent of `agent.max_tool_result_chars`. |
| `tools.max_file_read_bytes` | integer | `1048576` | Maximum file size in bytes that `read_file` will return. Files larger than this limit return an error string. Default is 1 MB (1,048,576 bytes). |

---

## `irc` — IRC transport

The `irc` section is optional. If absent, `IRCPlugin` does not connect.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `irc.host` | string | `irc.libera.chat` | IRC server hostname. |
| `irc.port` | integer | `6667` | IRC server port. |
| `irc.nick` | string | `corvidae` | Bot nickname. |
| `irc.tls` | boolean | `false` | Use TLS for the IRC connection. |
| `irc.channels` | list of strings | `[]` | IRC channels to join on connect (e.g., `["#general", "#dev"]`). |
| `irc.message_chunk_size` | integer | `400` | Maximum UTF-8 bytes per outbound IRC message. Messages longer than this are split using three-tier splitting: paragraphs → sentences → words. |

---

## `channels` — per-channel overrides

Keys are `transport:scope` identifiers (e.g., `irc:#general`, `cli:local`). Each value is a mapping of overrides applied on top of `agent` defaults. Any key absent from the channel block uses the `agent`-level default.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `system_prompt` | string or list of strings | `agent.system_prompt` | System prompt for this channel. Same format as `agent.system_prompt`. |
| `max_context_tokens` | integer | `agent.max_context_tokens` | Token budget for this channel's conversation history. |
| `max_turns` | integer | `agent.max_turns` | Maximum consecutive tool-calling turns for this channel. |
| `keep_thinking_in_history` | boolean | `agent.keep_thinking_in_history` | Controls `reasoning_content` retention for this channel. |

Channels not listed here are created on-demand when a message arrives and use `agent` defaults for all fields.

```yaml
channels:
  irc:#general:
    system_prompt: "You are the channel bot."
    max_context_tokens: 8000
  cli:local:
    system_prompt:
      - prompts/base.txt
      - prompts/cli-addendum.txt
    keep_thinking_in_history: true
```

---

## `mcp` — MCP server connections

Configured by `McpClientPlugin`. The `mcp` section is optional; if absent, no MCP servers are connected. See the [Plugin Guide](plugin-guide.md#mcpclientplugin-corvidae-mcp_clientpy) for full details.

```yaml
mcp:
  servers:
    my_server:
      transport: stdio         # "stdio" or "sse"
      command: npx             # command to launch (stdio only)
      args: ["-y", "@modelcontextprotocol/server-filesystem", "/path"]
      env:                     # optional environment variables (stdio only)
        NODE_ENV: production
      tool_prefix: "fs"        # optional; defaults to server name
      timeout_seconds: 30      # optional; default 30
```

---

## `logging` — Python logging configuration

The `logging` section is passed directly to `logging.config.dictConfig`. Any valid Python logging dict-config schema is accepted. Corvidae uses structured logging with extra fields; the `StructuredFormatter` in `corvidae.main` appends them to the log line.

```yaml
logging:
  version: 1
  disable_existing_loggers: false
  formatters:
    standard:
      (): corvidae.main.StructuredFormatter
      format: "%(asctime)s %(levelname)s %(name)s %(message)s"
  handlers:
    console:
      class: logging.StreamHandler
      formatter: standard
      stream: ext://sys.stderr
  loggers:
    corvidae:
      level: INFO
      handlers: [console]
      propagate: false
```
