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

agent:
  system_prompt: "You are a helpful assistant."
  max_context_tokens: 24000
  max_turns: 10
  keep_thinking_in_history: false
  compaction_threshold: 0.8
  compaction_retention: 0.5
  min_messages_to_compact: 5

tools:
  shell_timeout: 30
  web_fetch_timeout: 15
  web_max_response_bytes: 50000
  max_file_read_bytes: 1048576
  max_result_chars: 100000

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

The `llm.main` block is required. Every other key under `llm:` is an optional **role** (`background`, `embedding`, and future roles) built into its own client; `LLMPlugin.get_client(role)` falls back to `llm.main` for unconfigured or unknown roles. Only `llm.main` is hot-swapped on config reload — other roles are restart-only.

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
| `llm.embedding.base_url` | string | — | Base URL for an OpenAI-compatible `/embeddings` endpoint (embedding servers are usually separate from generation servers). Optional. |
| `llm.embedding.model` | string | — | Embedding model identifier. Recorded in `embedding_meta`; changing it against an existing store logs an ERROR and disables embedding. Config revert is the only remediation. |
| `llm.embedding.dimensions` | integer | — | **Required when `llm.embedding` is present.** Fixed vector dimension of the embedding model; the `memory_vec` table schema depends on it. Validated at startup. |
| `llm.embedding.document_prefix` | string | `""` | String prepended to every text when embedding stored content (`kind="document"`). Recorded in `embedding_meta`; changing it against an existing store logs an ERROR and disables embedding. Config revert is the only remediation. Trailing space is the operator's responsibility (match the model card). |
| `llm.embedding.query_prefix` | string | `""` | String prepended to every text when embedding a retrieval query (`kind="query"`). See `document_prefix` for mismatch semantics. |
| `llm.embedding.*` | — | — | The remaining keys (`api_key`, `max_retries`, ...) match `llm.main`. `document_prefix` and `query_prefix` are embedding-only — they do not exist on `llm.main` or `llm.background`. |

`llm.background` is parsed by `LLMPlugin`. If the key is absent, `LLMPlugin.get_client("background")` returns the main client, so all callers (including `SubagentPlugin`) fall back to `llm.main`. When `llm.embedding` is absent, `MemoryPlugin` stores records with `embedded=0` and retrieval degrades to FTS5 keyword search.

Retry behavior applies to transient status codes (429, 500, 502, 503, 504) and connection errors. The `Retry-After` header (429 responses) is honored when present; otherwise exponential backoff is used. Retries do not apply to client errors (4xx) other than 429.

---

## `daemon` — daemon-level settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `daemon.session_db` | string | `sessions.db` | Path to the SQLite database file for conversation history. Relative paths are resolved from the working directory. |
| `daemon.max_task_workers` | integer | `4` | Number of concurrent background task workers in `TaskQueue`. |
| `daemon.completed_task_buffer` | integer | `100` | Number of completed task records retained in memory for `task_status` output. |
| `daemon.idle_cooldown_seconds` | number | `30` | Minimum seconds between consecutive `on_idle` hook firings. Idle detection is push-based (`Agent._maybe_fire_idle`); this value governs the cooldown between successive firings. |
| `daemon.sqlite_journal_mode` | string | `wal` | SQLite journal mode for the session database. Allowed values: `delete`, `truncate`, `persist`, `memory`, `wal`, `off`. Read by `PersistencePlugin`. |
| `daemon.jsonl_log_dir` | string | — | Directory for per-channel JSONL conversation logs. When set, `JsonlLogPlugin` appends one JSON line per conversation event to `<dir>/<channel_id>.jsonl`. Relative paths are resolved against `_base_dir`. Omit to disable JSONL logging. |
| `daemon.metrics_jsonl` | string | — | Path of the metrics event log. When set, `MetricsJsonlPlugin` appends one JSON line (`{"ts", "name", "value", "tags"}`) per `on_metrics` event. Relative paths are resolved against `_base_dir`. Omit to disable the metrics JSONL sink. |
| `daemon.config_poll_interval` | float | `2.0` | Seconds between `agent.yaml` mtime checks by `ConfigWatcherPlugin`. |

---

## `agent` — agent-level defaults

These values apply to all channels. Per-channel overrides in the `channels` section take precedence where set.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `agent.system_prompt` | string or list of strings | `"You are a helpful assistant."` | System prompt prepended to each LLM call. A list of strings is treated as file paths; each file is read and concatenated with double newlines. Relative paths are resolved against `_base_dir`. |
| `agent.max_context_tokens` | integer | `24000` | Token budget for conversation history. `CompactionPlugin` compacts when the estimated token count exceeds `compaction_threshold` × this value. |
| `agent.max_turns` | integer | `10` | Maximum consecutive tool-calling turns per user message before tool dispatch is suppressed. |
| `agent.keep_thinking_in_history` | boolean | `false` | When `false`, `ThinkingPlugin` strips `reasoning_content` from in-memory assistant messages after they are persisted. Does not affect the DB record. |
| `agent.compaction_threshold` | float | `0.8` | Compaction triggers when `token_estimate / max_context_tokens` exceeds this value. |
| `agent.compaction_retention` | float | `0.5` | After compaction, retain messages that fit within `compaction_retention × max_context_tokens` tokens, counting from the most recent. |
| `agent.min_messages_to_compact` | integer | `5` | Skip compaction if the conversation has this many messages or fewer. |
| `agent.chars_per_token` | float | `3.5` | **Deprecated.** Accepted for interface and config compatibility. Has no effect on token counting — `count_tokens()` uses the module-level `_FALLBACK_CHARS_PER_TOKEN` constant (3.5) regardless of this value. |
| `agent.immutable_settings` | list of strings | `[]` | Keys the agent is blocked from changing via the `set_settings` tool. `system_prompt` is always blocked regardless of this list. |

### `agent.context_compact` — removed

`ContextCompactPlugin` has been removed: superseded by `MemoryPlugin` (memory consolidation and retrieval), with per-turn token stats now in the Phase 0 `usage_log` table. Any `agent.context_compact.*` keys in existing configs are ignored.

---

## `memory` — autobiographical memory

`MemoryPlugin` consolidates compacted dialog into first-person memory records and retrieves them into the prompt (see [design.md](design.md)). The former `dream.*` keys are gone with `DreamPlugin` (absorbed by this plugin).

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `memory.idle_consolidate_after` | float | `1800` | Seconds of channel inactivity before the `on_idle` trigger consolidates that channel's un-consolidated tail. |
| `memory.half_life_days` | float | `30` | Recency half-life of the retrieval score (`similarity × exp(-age_days / half_life)`). |
| `memory.retrieval.k` | integer | `8` | Vector-KNN candidate count per retrieval. |
| `memory.retrieval.bands.strong` | float | `0.75` | Score at or above which a retrieved memory is banded `[strong]`. |
| `memory.retrieval.bands.moderate` | float | `0.60` | Score at or above which a retrieved memory is banded `[moderate]`; below is `[weak]` (dropped unless nothing else matched). |
| `memory.channel_groups` | mapping | `{}` | `{group_name: [channel ids]}` — channels in the same group share retrieval scope. Memory is otherwise compartmentalized per channel. |
| `memory.consolidation_prompt` | string | built-in | Overrides the consolidation system prompt (documented copy: `prompts/memory_consolidation.md`). |

---

## `funnel` — context-admission funnel

`FunnelPlugin` is the single chokepoint for tail CONTEXT admission (dedupe, budgets, injection framing).

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `funnel.default_budget` | integer | `512` | Token budget per `admit()` call when the source has no override. |
| `funnel.budgets.<source>` | integer | — | Per-source budget override (e.g. `funnel.budgets.memory`). |

---

## `tools` — built-in tool settings

Configured by `CoreToolsPlugin`. `tools.max_result_chars` is read by `ToolCollectionPlugin`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `tools.max_result_chars` | integer | `100000` | Maximum characters in a tool result string before truncation. Read by `ToolCollectionPlugin`. Truncated results have `[truncated — N chars total]` appended. The legacy key `agent.max_tool_result_chars` is accepted but deprecated; use `tools.max_result_chars` instead. |
| `tools.shell_timeout` | integer | `30` | Seconds before a `shell` command is killed. |
| `tools.web_fetch_timeout` | integer | `15` | Seconds before a `web_fetch` HTTP request is aborted. |
| `tools.web_max_response_bytes` | integer | `50000` | Maximum bytes of HTTP response body returned by `web_fetch`. Content beyond this limit is truncated with `[truncated]`. This limit is independent of `tools.max_result_chars`. |
| `tools.max_file_read_bytes` | integer | `1048576` | Maximum file size in bytes that `read_file` will return. Files larger than this limit return an error string. Default is 1 MB (1,048,576 bytes). |
| `tools.web_search_max_results` | integer | `8` | Maximum number of DuckDuckGo results returned by the `web_search` tool. |

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

For `transport: stdio`, the required keys are `command` and optionally `args` and `env`. For `transport: sse`, the required key is `url`; `command`, `args`, and `env` are not used.

```yaml
mcp:
  servers:
    # stdio transport example
    my_fs_server:
      transport: stdio
      command: npx
      args: ["-y", "@modelcontextprotocol/server-filesystem", "/path"]
      env:                     # optional environment variables (stdio only)
        NODE_ENV: production
      tool_prefix: "fs"        # optional; defaults to server name
      timeout_seconds: 30      # optional; default 30

    # sse transport example
    my_sse_server:
      transport: sse
      url: http://localhost:9000/sse   # required for sse transport
      tool_prefix: "remote"            # optional; defaults to server name
      timeout_seconds: 30              # optional; default 30
```

---

## Hot-reload

`ConfigWatcherPlugin` polls `agent.yaml` for mtime changes every `daemon.config_poll_interval` seconds (default 2.0). When a change is detected, the file is re-parsed, CLI overrides are re-applied, and `on_config_reload` is dispatched to all registered plugins.

No restart is required for the changes listed below. Changes that require restart are listed separately.

### What updates on reload

| Setting | Effect |
|---------|--------|
| `llm.main` | New `LLMClient` created and started; old client closed asynchronously after in-flight requests finish. |
| `agent.compaction_threshold` | `CompactionPlugin` reads the new threshold on its next compaction check. |
| `agent.chars_per_token` | Deprecated. The value is reloaded into `Agent` and `CompactionPlugin` for config compatibility, but has no effect on token counting — `count_tokens()` uses the module-level constant. |
| `daemon.idle_cooldown_seconds` | `Agent` uses the new cooldown on the next idle check. |
| `agent.immutable_settings` | `RuntimeSettingsPlugin` resets its blocklist and re-applies the new list. Constructor-supplied immutable keys are always retained. |
| New channel entries in `channels:` | `load_channel_config` registers newly added channels. |
| `agent` defaults | `ChannelRegistry.agent_defaults` is updated; new channels created after the reload use the new defaults. |

### What requires restart

| Setting | Reason |
|---------|--------|
| `irc.host`, `irc.port`, `irc.nick` | IRC transport is connected at startup; changing connection parameters requires reconnection. |
| Existing channel config in `channels:` | Channels already registered retain their original `ChannelConfig` until restart. |
| `llm.background` | Not updated on reload. |
| `tools.max_result_chars`, `tools.shell_timeout`, `tools.web_fetch_timeout` | Deferred; not currently implemented in hot-reload path. |

### Error handling

- **Invalid YAML**: logged at `ERROR`; reload is skipped and the running config is unchanged.
- **Missing `llm.main` after merge**: logged at `ERROR`; reload is skipped.
- **Config file deleted**: logged at `WARNING`; polling continues. The file may reappear.
- **Per-plugin failure in `on_config_reload`**: logged at `ERROR`; other plugins still receive the reload.

---

## `logging` — Logging configuration

The `logging` section is optional. Omitting it applies defaults (INFO level; file output in CLI mode, stderr in programmatic use).

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `logging.level` | string | `INFO` | Minimum log level. One of `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. Case-insensitive. |
| `logging.file` | string | none (CLI: `corvidae.log`) | Path to a rotating log file (10 MB, 5 backups). When set, logs go to this file. When absent, logs go to stderr. In CLI mode, defaults to `corvidae.log`. |

When `logging.file` is set in YAML, logs go to that rotating file regardless of mode. When `logging.file` is absent, logs go to stderr — except in CLI mode, where the default is `corvidae.log` to avoid polluting the chat interface.
