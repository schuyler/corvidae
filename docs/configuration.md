# Corvidae Configuration Reference

Corvidae is configured via a single YAML file, `agent.yaml`, passed to the daemon at startup. The full parsed dict is available to every plugin via the `on_start(config: dict)` hook.

The key `_base_dir` (a `pathlib.Path`) is set by `Runtime.start()` (`runtime.py`) alongside the parsed config. It points to the directory containing `agent.yaml` and is used to resolve relative file paths (e.g., `system_prompt` file lists). The daemon also keeps its fence key there, in `provenance.key` (32 random bytes, created mode 0600 on first start; see [threat-model.md](threat-model.md)); keep the file across restarts so fences stay stable.

Two example configs ship at the repo root: `agent.yaml.example` (every built-in plugin, most keys commented with their default) and `agent.minimal.yaml.example` (the core agent loop only — long-term memory, appraisal, and critique disabled via `plugins.disabled`; see the `plugins` section below).

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

agent:
  system_prompt: "You are a helpful assistant."
  max_context_tokens: 24000
  max_turns: 10
  reasoning_history: drop
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
  allow: []               # default-deny: nobody may drive the agent until set
  channels: []
  message_chunk_size: 400

channels:
  irc:#general:
    register_prompt: "This is #general; keep it brief."
    max_context_tokens: 16000
    max_turns: 10
    reasoning_history: drop
```

---

## `plugins` — enabling and disabling plugins

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `plugins.disabled` | list of strings | `[]` | Entry-point plugin names to block from loading. `Runtime.start()` calls `pm.set_blocked(name)` for each entry before entry points are loaded; blocked plugins are never instantiated. Names must match the entry-point name from `[project.entry-points.corvidae]` in `pyproject.toml` (e.g. `memory`, `funnel`, `appraisal`, `critique`, `outcome_log`, `memory_tools`, `goals`, `silence`, `skills`, `subagent`, `mcp`, `cli`). Only entry-point plugins can be disabled this way — manually registered plugins cannot be blocked via config. A name that matches no installed plugin (a typo, the retired `core_tools`, or a cognition plugin on a kernel-only install) blocks nothing and logs one `WARNING` at startup naming the value and the installed plugin names. A null `plugins:` or `disabled:` means nothing is disabled; any other non-list value (a bare string, say) or a non-string entry fails startup with a `Config error` naming the key and value. |

`agent.minimal.yaml.example` disables `memory`, `memory_tools`, `funnel`, `appraisal`, `critique`, `outcome_log`, `gates`, `silence`, `heartbeat` and `goals` — the cognition plugin set — leaving the core agent loop, multi-turn tool calling, compaction, and persistence/restart recovery intact. See the [Plugin Guide](plugin-guide.md#plugin-disable) for the hook-level mechanics.

A misspelled entry is not an error: it blocks nothing, the daemon starts normally with the plugin still loaded, and a `WARNING` names the entry and the installed plugin names (see the table above). To confirm a disable took effect, check the `"plugins loaded"` line the daemon logs at startup: it carries `loaded` (the plugins that actually registered) and `blocked` (this list, verbatim). A plugin you meant to disable that still shows up under `loaded` was not blocked.

---

## `llm` — LLM backend

Every key under `llm:` except `defaults` and `stages` is a **profile**: an endpoint, a model and request parameters, built into its own client. Each model-calling stage is registered by the plugin that owns it, with a kind (`chat` or `embedding`). A stage sends to the profile `llm.stages` maps it to; a stage the map doesn't name uses its kind's default profile, `llm.main` for chat and `llm.embedding` for embedding unless `llm.defaults` says otherwise. Main turns use the default chat profile. There are no fallback chains. Only a default chat profile is required.

```yaml
llm:
  main:                       # a profile; kind omitted means chat
    base_url: http://host:8080/v1
    model: swift-1.5
    extra_body: {reasoning_effort: xhigh}
  forks:                      # a profile derived from main
    extends: main
    extra_body:
      reasoning_budget_tokens: 8192
  embedding:
    kind: embedding
    base_url: http://host:8081/v1
    model: bge-m3
    dimensions: 1024
  defaults: {chat: main, embedding: embedding}   # optional; these are the fallbacks
  stages:                     # optional: stage -> profile
    compaction: forks
    consolidation: forks
```

**All of `llm:` is read at startup. A change to any of it, `llm.main` included, applies on restart.** A reload that changes it logs the usual per-key restart-only warning for each changed key, plus one summary, `llm config changed; LLM profiles, llm.stages and llm.defaults apply on restart`, and keeps the running clients. A reload whose `llm:` section is malformed is refused whole, like any other bad section.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `llm.main.base_url` | string | — | Base URL for an OpenAI-compatible Chat Completions endpoint. Required in every profile; the `llm.main` block itself is required only when `llm.defaults.chat` is unset. |
| `llm.main.model` | string | — | Model identifier sent in each request. Required in every profile. |
| `llm.<profile>.kind` | `chat` \| `embedding` | `chat` | What the profile serves. A chat profile takes chat stages and main turns; an embedding profile takes the `embedding` stage and needs `dimensions`. An omitted `kind` means `chat`, applied after `extends` is merged, so a child of an embedding profile stays one. Any other value fails startup. |
| `llm.<profile>.extends` | string | — | Name of another profile to build on. See [Derived profiles](#derived-profiles-extends). Consumed by corvidae; never sent to the endpoint. |
| `llm.main.api_key` | string | `null` | API key for authentication, sent as `Authorization: Bearer <key>`. Omit if the backend does not require one. Either a literal key or a quoted `"${VAR}"` reference to an environment variable (see below). Give the bare key: a value such as `Bearer ${X}` would send "Bearer Bearer ...". |
| `llm.main.extra_body` | mapping | `null` | Provider-specific fields merged into the request body (e.g., `id_slot` for llama.cpp). On llama.cpp, `reasoning_format: none` returns the reply's reasoning inline, for `raw_reasoning`. This is where a stage's reasoning budget (`reasoning_budget_tokens`) or token cap (`max_tokens`) goes. Corvidae passes the keys through without interpreting them. Where a channel's `set_settings` override and a fork's profile set the same key, the profile wins on a fork (see [`agent.forks`](#agentforks--background-processes-as-forks)); on a main turn the override wins. |
| `llm.main.raw_reasoning` | boolean | `false` | The endpoint returns reasoning inline in `content`, closed by `</think>` (llama.cpp with `extra_body.reasoning_format: none`). Corvidae splits it into `reasoning_content` itself and treats a reply with no `</think>` as cut off mid-reasoning: the turn is re-run once, then reported as failed (#355). A call that sends `response_format` (a fork or background stage's JSON schema) runs in the server's parsed mode instead: `reasoning_format` is dropped from that request and its reply is not split, so it gets no cut-off detection (#360). shortcut: this relies on the server's own default being a parsed mode; a llama-server started with `--reasoning-format none` still rejects schema calls; upgrade to an explicit per-call override if such a deployment appears. shortcut: set it only where the chat template forces thinking open; otherwise every non-reasoning reply reads as cut off. Upgrade when a per-reply signal replaces the template assumption. |
| `llm.main.reasoning_reopened` | string | `truncate` | What happens to a final reply that reopens reasoning after its answer: a second `<think>`, or a stray `</think>`, after the first `</think>` (#348). Needs `raw_reasoning`: in parsed mode the client sees no tags to check. Either way the client cuts the reply at that tag, so the user never sees the text after it, and logs a warning with the channel and the full raw completion. `truncate` sends and stores the text before the tag. `rerun` throws the cut reply away and regenerates once with the same messages; if that reply reopens too, its cut text is used. A stray `</think>` has no start marker, so the second block's text before the tag survives the cut. Only the main turn loop reruns; forks, subagents and background stages always truncate. Schema calls (`response_format`) aren't split, so they aren't checked. An invalid value is logged as a warning and read as `truncate`; it never stops startup. Applies on restart. |
| `llm.main.max_retries` | integer | `3` | Maximum retry attempts on transient HTTP errors (429, 500, 502, 503, 504) and connection failures. Set to 0 to disable retries. After they run out, the turn sends the error reply and then retries itself (5 s doubling to 60 s) until the backend answers. |
| `llm.main.retry_base_delay` | float | `2.0` | Base delay in seconds for exponential backoff between retry attempts. |
| `llm.main.retry_max_delay` | float | `60.0` | Maximum delay cap in seconds for exponential backoff. |
| `llm.main.model_not_loaded_wait` | float | `60.0` | Total seconds to keep retrying when the server answers HTTP 400 "model is not loaded" (a llama-server router running `--no-models-autoload` with another model swapped in). Separate from `max_retries`; uses the same backoff, with at least 1 s between attempts (the last sleep is cut short to fit the remaining wait). Must be a finite, non-negative number, or startup fails. The turn waits for this long before replying, so set it a little longer than a model swap takes. `0` disables. When it runs out, the reply names the unloaded model instead of the generic error, and the turn then retries itself (5 s doubling to 60 s) until the model is loaded. |
| `llm.main.timeout` | float | `null` | Total HTTP timeout in seconds per LLM request. `null` means no timeout: the request waits as long as the server takes, and a connect to a host that drops packets waits until the OS gives up. |
| `llm.embedding.base_url` | string | — | Base URL for an OpenAI-compatible `/embeddings` endpoint (embedding servers are usually separate from generation servers). Optional. |
| `llm.embedding.model` | string | — | Embedding model identifier. Recorded in `embedding_meta`; changing it against an existing store logs an ERROR and disables embedding. Config revert is the only remediation. |
| `llm.embedding.kind` | string | — | **Must be `embedding`** on the profile the embedding stage uses. Without it the block is a chat profile and startup fails (see [Migration](#migration-from-the-single-main-block)). |
| `llm.embedding.dimensions` | integer | — | **Required for an embedding profile** (any profile with `kind: embedding`, not only one named `embedding`). Fixed vector dimension of the embedding model; the `memory_vec` table schema depends on it. Validated at startup. |
| `llm.embedding.document_prefix` | string | `""` | String prepended to every text when embedding stored content (`kind="document"`). Recorded in `embedding_meta`; changing it against an existing store logs an ERROR and disables embedding. Config revert is the only remediation. Trailing space is the operator's responsibility (match the model card). |
| `llm.embedding.query_prefix` | string | `""` | String prepended to every text when embedding a retrieval query (`kind="query"`). See `document_prefix` for mismatch semantics. |
| `llm.defaults.chat`, `llm.defaults.embedding` | string | `main`, `embedding` | The default profile for each kind: what main turns and any unmapped stage of that kind use. Each must name a defined profile of that kind. A kind may have no default (no `llm.embedding` block and no `llm.defaults.embedding`: memory degrades to keyword search unless `llm.stages.embedding` maps the stage); a chat default is required, so `defaults.chat: foo` with no `llm.main` boots and sends main turns to `foo`. Any other key under `defaults` fails startup. |
| `llm.stages.<stage>` | string | — | The profile `<stage>` sends to. See [Stages](#stages). |
| `llm.embedding.*` | — | — | The remaining keys (`api_key`, which also takes `"${VAR}"`, `max_retries`, ...) match `llm.main`. `document_prefix` and `query_prefix` are embedding-only — they do not exist on `llm.main`. |

**API key from the environment.** Quote the reference so YAML keeps it as text: `api_key: "${OPENAI_API_KEY}"`. The variable is read from the environment the daemon runs in, so a tracked `agent.yaml` holds no secret. This works for every profile, `main` included, and uses the same `${VAR}` expansion as MCP `headers`. A literal key still works; a value with no `${NAME}` (including one containing a bare `$`) is used as is. If the variable is unset, empty or contains a newline or NUL (often a trailing newline), startup fails with an error naming `llm.<profile>.api_key` and the variable; set it in the environment corvidae runs in and restart. The key's value is never logged or put in an error or a reply. To pick up a changed environment, restart. A reload doesn't rebuild clients, so a changed variable or a newly referenced one is picked up only on restart.

**Remote profiles.** Every profile is an OpenAI-compatible client, so any of them can point at a hosted endpoint. Whatever that profile is sent then leaves the host. A profile receives the main turns if it is the default chat profile, and everything the stages mapped to it send (see [Stages](#stages)): the agent's rendered view (the system prompt, CONTEXT blocks, tool calls and results, up to the channel's budget; compaction's fallback sends the uncapped prefix), the importance prior, subagent turns, memory embedding, retrieval and retention. Conversation content is in all of these inputs. Every stage `llm.stages` doesn't map goes to its kind's default profile, so with no `llm.stages` entries the default chat profile gets all of the chat stages and the default embedding profile gets the embedding stage. See `threat-model.md` finding 12. Hosted endpoints give no latency guarantee, so set `llm.<profile>.timeout` generously.

At startup `LLMPlugin` logs where every registered stage resolves, for example `llm stages resolved: appraisal→main, compaction→main, consolidation→main, critique→main, embedding→embedding, rubric→main, subagent→main`. A stage with no profile (its kind has no default and the map doesn't name it) shows `→(none)`.

Retry behavior applies to transient status codes (429, 500, 502, 503, 504) and connection errors. The `Retry-After` header (429 responses) is honored when present; otherwise exponential backoff is used. Retries do not apply to client errors (4xx) other than 429.

### Derived profiles (`extends`)

`extends: <profile>` lays the block over the named profile, so a derived profile states only what differs. The merge:

- Mappings merge recursively: a child's `extra_body.reasoning_budget_tokens` adds to the base's `extra_body`, keeping its other keys.
- Lists and scalars replace.
- A child's `null` deletes the inherited key. With `extends: main` and `extra_body: {reasoning_effort: null}`, the derived profile has no `reasoning_effort`.
- Chains resolve root first. `kind` is an ordinary key, so a child inherits its base's.

Every resolved profile needs `base_url` and `model` after the merge (a base block that exists only to be extended is not supported), and an embedding profile needs `dimensions`. Startup fails, and a reload is refused, on an unknown base, an `extends` cycle, an unknown `kind`, a missing `base_url`, `model` or `dimensions`, a non-mapping profile, an unknown or wrong-kind `defaults` entry, no default chat profile, a `defaults` or `stages` value that isn't a profile name (a list or mapping), or a `stages` entry naming a missing profile. Each message names the key (`llm.forks.extends names unknown profile 'mian'; defined profiles: embedding, forks, main`).

A change to a base is a change to every profile that extends it, but only at the next restart.

### Stages

| Stage | Kind | What it sends |
|-------|------|---------------|
| `compaction` | chat | The summary fork, and the view call when the fork misses. |
| `subagent` | chat | A subagent's turns. |
| `consolidation` | chat | The consolidation fork, and the catch-up view call for a range that left the window. |
| `rubric` | chat | The importance prior's call. |
| `appraisal` | chat | The stage-2 appraisal fork. |
| `critique` | chat | Each stylistic lens and the provenance check. |
| `embedding` | embedding | Memory embedding, retrieval and retention. |

Every call a stage makes goes to its profile, on every path: fork, view call, windowless call and subagent. A fork renders what it renders without the map; only its endpoint, model and `extra_body` come from the stage's profile. `usage_log` and the metrics name the stage's profile on every call, forks included.

The `consolidation`, `rubric`, `appraisal`, `critique` and `embedding` stages belong to the cognition extra; without it only `compaction` and `subagent` exist.

An `llm.stages` entry naming a profile that doesn't exist, or one of the wrong kind (a chat stage mapped to an embedding profile), fails startup. An entry naming no registered stage logs a warning and is ignored: a disabled plugin's stage is expected there, and so is a typo, so check the warning. If a kind has no default profile and some stage of that kind is unmapped, startup logs a warning naming those stages, which then have no client and degrade.

Keeping a fork's cached prefix is the operator's job. A stage mapped to another endpoint or model sends a cold prefill, and a profile that sets a template-visible key (such as `reasoning_effort`) differently from main's also misses. Derive the stage's profile from `main` with `extends` and set only sampler-side keys to keep it. #440 reports misses.

### Migration from the single `main` block

- An `llm.embedding` block needs `kind: embedding`. Without it startup fails with `llm.embedding is a chat profile but is the default embedding profile by name; add kind: embedding to it, or set llm.defaults.embedding`.
- `agent.forks.reasoning_budget_tokens` is removed, and leaving it in config is a startup error that names the key. To keep a budget, put `reasoning_budget_tokens` in the `extra_body` of a profile, and map the fork stages to it under `llm.stages`. An unmapped fork carries a budget only if the default chat profile's `extra_body` sets one; the old default of 2048 is gone.
- `llm.background` has no special meaning. It was accepted but nothing used it; map a stage to it under `llm.stages` if you want it used.
- `llm.main` is no longer hot-swapped on reload; restart to apply a change.
- The `main_client` and `background_client` properties and `LLMPlugin.embedding_dimensions` are removed; use `get_client(stage)` and `profile_config(stage)`.

---

## `daemon` — daemon-level settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `daemon.session_db` | string | `sessions.db` | Path to the SQLite database file for conversation history. Relative paths are resolved from the working directory. |
| `daemon.max_task_workers` | integer | `4` | Number of concurrent background task workers in `TaskQueue`. |
| `daemon.completed_task_buffer` | integer | `100` | Number of completed task records retained in memory for `task_status` output. |
| `daemon.task_results_path` | string | — | Path of the task-results JSONL file. When set, `TaskQueue` appends one JSON line (`{"ts", "task_id", "channel", "description", "result"}`) per completion (success or failure), and the `task_result` tool retrieves a stored result by task id — results survive deque eviction and daemon restarts. Relative paths are resolved against `_base_dir`. Omit to disable persistence (deque-only, the historical behavior). |
| `daemon.sqlite_journal_mode` | string | `wal` | SQLite journal mode for the session database. Allowed values: `delete`, `truncate`, `persist`, `memory`, `wal`, `off`. Read by `PersistencePlugin`. |
| `daemon.jsonl_log_dir` | string | — | Directory for per-channel JSONL conversation logs. When set, `JsonlLogPlugin` appends one JSON line per conversation event to `<dir>/<channel_id>.jsonl`. Relative paths are resolved against `_base_dir`. Omit to disable JSONL logging. Each file is capped by `logging.jsonl.*`. |
| `daemon.metrics_jsonl` | string | — | Path of the metrics event log. When set, `MetricsJsonlPlugin` appends one JSON line (`{"ts", "name", "value", "tags"}`) per `on_metrics` event. Relative paths are resolved against `_base_dir`. Omit to disable the metrics JSONL sink. |
| `daemon.shutdown_timeout` | number | `3` | Seconds the graceful shutdown phase (every plugin's `on_stop`, then the idle tracker and clock) may run. Past it the daemon logs a WARNING naming this key and exits 1. Also bounds the cleanup after a failed startup. Must be a number > 0 (booleans, `0`, `null` and non-finite values are rejected; a bad value fails startup naming the key and the value). Values under 3 can cut a plugin's own shutdown short: the MCP and web plugins size theirs at about 2.5 s. Restart to apply. |
| `daemon.exit_timeout` | number or null | `5` | Seconds the process may take to exit after the graceful phase ends (or raises). A thread a plugin left running, such as an unclosed database connection, otherwise keeps the process alive forever. Past the bound the daemon logs a WARNING naming the threads still running and this key, then exits 1, so a supervisor sees an abnormal run. `0` or `null` disables the watchdog, for instance to attach a profiler to a hung process. Must be a number >= 0 or null; a bad value fails startup naming the key and the value. The worst case after SIGTERM is `shutdown_timeout + exit_timeout`; very small values are for tests. Applies to `corvidae serve`, `cli` and `acp`. Restart to apply. |
| `daemon.config_poll_interval` | float | `2.0` | Seconds between `agent.yaml` mtime checks by `ConfigWatcherPlugin`. |
| `daemon.notification_spill_dir` | string | `state/notification-spill` | Directory where the full text of a cut notification or tool result is written, one file per item, mode 0600. Despite the `notification_` prefix, all three `notification_spill_*` keys cover tool results too (#575). Relative paths are resolved against `_base_dir`. Startup fails, naming the key, if the resolved path contains characters a notification header would not render verbatim (brackets, `<`, `>`, `;`, control characters, doubled or edge whitespace, or anything NFKC normalisation changes). Restart to apply. |
| `daemon.notification_spill_max_chars` | integer | `1000000` | Most characters written to one spill file; a longer text is cut to this and the header says how many characters the file holds. Must be an integer ≥ 1 (booleans and floats are rejected; a bad value fails startup naming the key and the value). Restart to apply. |
| `daemon.notification_spill_max_age_days` | number | `7` | Spill files older than this many days (by modification time) are deleted: once at startup, then hourly. Only the age is checked; nothing limits the directory's total size. Only files the service named (`<32 hex digits>.txt`) are deleted. Must be a number > 0 (booleans and NaN are rejected; a bad value fails startup naming the key and the value). Restart to apply. |

A notification cut by the plugin that sent it (see `webhook_notify.max_payload_chars`) shows the model its first characters inline. When the channel has a tool that reads spill files (`read_file` or `shell`, not disabled there), the header also names the file holding the rest, and the agent pages through it with `read_file`'s `offset` and `length`; a page is sized to the channel's budget (`tools.read_file_window_fraction`). A file deleted by age gives `read_file` an error that says so and names `notification_spill_max_age_days`. If the write fails, the notification is delivered cut and the header says the full text was not saved; the daemon logs a warning naming the directory. A tool result over `tools.max_result_chars` is cut and spilled the same way.

`daemon.idle_cooldown_seconds` and `daemon.idle_poll_interval` are removed with the `on_idle` hook. Either one in the config is an error at startup, and a reload carrying one is rejected. Idle work now runs on the kernel clock and each plugin sets its own period (for example `memory.idle_consolidate_after`); delete the key.

---

## `agent` — agent-level defaults

These values apply to all channels. Per-channel overrides in the `channels` section take precedence where set.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `agent.system_prompt` | string or list of strings | `"You are a helpful assistant."` | The agent's identity: the head of every channel's system prompt. No channel setting replaces it. A list of strings is treated as file paths; each file is read and concatenated with double newlines. Relative paths are resolved against `_base_dir`. |
| `agent.register_prompt` | string or list of strings | unset | A register for every channel that doesn't set its own, appended after the identity. Replaces the transports' shipped defaults; see [Registers](#registers). On a conversational transport the silence plugin still appends its instruction after it. Same format as `agent.system_prompt`. |
| `agent.max_context_tokens` | integer | `24000` | Token budget for conversation history; also bounds each catch-up consolidation view. `CompactionPlugin` compacts when the estimated token count exceeds `compaction_threshold` × this value. A hard ceiling: a window still over it after compaction loses its oldest rows from memory (the log keeps them). |
| `agent.max_turns` | integer | `10` | Maximum consecutive tool-calling turns per exchange before tool dispatch is suppressed. The counter resets when an item starts a new exchange with no tool cycle in flight: a user message, or a notification that originates its own trace (a scheduler reminder or heartbeat). |
| `agent.reasoning_history` | `drop` \| `exchange` \| `keep` | `drop` | `drop` strips `reasoning_content` from every in-memory assistant message. `exchange` strips it only from completed exchanges (assistant messages before the last `role: user` message), so a tool chain in progress keeps its reasoning — the trained distribution for Qwen-family models. `keep` never strips it. Applied to the whole window before each LLM call, so history reloaded after a restart follows it too. The DB record always keeps reasoning. Any other value, or the removed `keep_thinking_in_history` key, fails at startup. Use `exchange` with the server's `preserve_thinking` off (see below). |
| `agent.compaction_threshold` | float | `0.8` | Compaction triggers when `token_estimate / max_context_tokens` exceeds this value. |
| `agent.compaction_retention` | float | `0.5` | After compaction, retain messages that fit within `compaction_retention × max_context_tokens` tokens, counting from the most recent. |
| `agent.min_messages_to_compact` | integer | `5` | Skip compaction if the conversation has this many messages or fewer. |
| `agent.compaction_fork_prompt` | string | built-in | Instruction appended to the dropped prefix as a `user` turn when compaction writes its summary: to a fork when forks are on for `compaction` (see [`agent.forks`](#agentforks--background-processes-as-forks)), and to the same render sent on the `compaction` stage's profile on a fork miss or with forks off. Documented copy: `prompts/compaction_fork.md`. Replaces the removed `agent.compaction_prompt`: a config that still sets that key logs a startup warning naming this one and the key is ignored. Move any custom text here, rewritten as an instruction to the agent about its own conversation, not as a summarizer's system prompt. |
| `agent.forks` | mapping | on | Run compaction, consolidation, appraisal and critique as forks of the main window. See [`agent.forks`](#agentforks--background-processes-as-forks). |
| `agent.chars_per_token` | float | `3.5` | **Deprecated.** Accepted for interface and config compatibility. Has no effect on token counting — `count_tokens()` uses the module-level `_FALLBACK_CHARS_PER_TOKEN` constant (3.5) regardless of this value. |
| `agent.immutable_settings` | list of strings | `[]` | Keys the agent is blocked from changing via the `set_settings` tool, on top of its built-in allowlist. `set_settings` only accepts the inference parameters `temperature`, `top_p`, `top_k`, `frequency_penalty`, `presence_penalty`, `max_tokens` and `seed`, the framework keys `max_turns`, `max_context_tokens` and `reasoning_history`, and dotted plugin tunables, so this list is for narrowing that set (e.g. the gate keys under "Runtime-tunable gate parameters"). `system_prompt` is always blocked. |
| `agent.request_logprobs` | boolean | `false` | When `true`, `Agent` merges `"logprobs": true` into every LLM request body. llama-server returns a logprobs envelope (surfaced to plugins via `on_agent_response`); Anthropic-style providers return nothing and the field surfaces as `None`. Static operator-only flag — captured at init and on config reload, not per-channel tunable. |
| `agent.mid_conversation_system_role` | `system` \| `user` | `system` | Role under which every `system` message *after the leading system prompt* is sent to the LLM: funnel CONTEXT, notifications without a `tool_call_id`, and tool results whose call compaction dropped (#79; stored as `role: tool`, framed as a system notification at render). Strict chat templates (Qwen 3.8, Swift-1.5) raise "System message must be at the beginning" on those; `user` satisfies them and also keeps untrusted text out of the `system` role ([threat-model.md](threat-model.md), finding 3). Applied when the prompt is built: the in-memory window and `sessions.db` keep `role: system`, and the rewrite is deterministic, so the cached prompt prefix stays stable. Any other value logs a warning and uses `system`. Static operator-only flag, re-read on config reload. Scheduled firings (`scheduler` reminders and jobs) are notifications without a `tool_call_id`, so they follow this key too, as do framed orphaned tool results; a strict template needs `user` for them to render at all. |

### `agent.forks` — background processes as forks

On by default (GitHub issue #51; the default flipped in #320). A background
stage doesn't build its own prompt. It renders the channel's window exactly
as the main turn does, appends one `user` instruction, and makes one call on
the stage's profile (`llm.stages`, else the default chat profile). When that
profile is the main turn's, or differs only in sampler-side keys, a single-slot
llama-server reuses the cached prefix instead of evicting it. The agent's own system prompt is in view either way, so
records and summaries are written in its voice. A fork miss (forks off for
the stage, the exchange left the window, the fork failed or its reply was
unusable) skips the stage with a warning naming the stage, channel, trace
and what was missing; it makes no standalone call. That covers stage-2
appraisal, every critique lens, the provenance check and idle
consolidation of a range still in the window. Two exceptions send the
agent's rendered view on the stage's profile as a plain call instead, with no tools: compaction
on a miss (then the fallback described in design.md applies), and
consolidation of a range that has left the window, including the compaction
trigger's task, rebuilt from `message_log`. Both use the same fork-instruction
keys. See [design.md](design.md#forks). The view call costs a cold prefill, on
the slot of the stage's profile (the default chat profile's when the stage is unmapped): for compaction's fallback it sends the whole uncapped
prefix at the cut (the window is over budget when compaction runs). Catch-up consolidation trims each segment's view to
`max_context_tokens`; a single oversized row or tool cycle can still exceed
it, and a warning names the segment.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `agent.forks.enabled` | boolean | `true` | Master switch. `false` makes every live-window stage except compaction skip on every run (appraisal stage 2, all lenses, provenance, idle consolidation of in-window ranges; the turn consolidation trigger doesn't run at all); compaction and out-of-window consolidation send the view as a plain call on the stage's profile. With forks off, a channel that never compacts never consolidates its in-window rows. |
| `agent.forks.stages` | list of strings or null | `null` | Which stages fork: `compaction`, `consolidation`, `appraisal`, `critique`. `null` means all of them. A name no plugin uses matches nothing. `consolidation`, `appraisal` and `critique` belong to the cognition extra; without it only `compaction` exists. |

`agent.forks.reasoning_budget_tokens` was removed, and a config that still sets it fails startup with an error naming the key. A fork's reasoning budget, token cap and any other request parameter belong to the profile its stage maps to: put them in that profile's `extra_body` (see [`llm`](#llm--llm-backend)). A fork keeps the main reasoning effort only if its profile does: the chat template renders the effort into the system block, so a profile that changes `reasoning_effort` misses the cached prefix. An unmapped fork resolves to the default chat profile and sends no budget unless that profile has one.

**Body precedence on a fork.** Later wins: `model`, `messages` and `tools`; the stage profile's `extra_body`; the channel's runtime overrides (`set_settings`), minus any key the profile's `extra_body` also sets; the stage's own `extra_body` (its `response_format`); `tool_choice: "none"`. So the profile beats a runtime override: a channel's `max_tokens: 200` against a profile's `extra_body.max_tokens: 4096` sends 4096, and the fork's cached prefix is unaffected because every runtime override is a sampler-side key. A main turn is unchanged, and there the override beats the profile. The same applies to an unmapped fork: an override of a key in the default chat profile's `extra_body` is dropped for forks and still applies to the channel's main turns.

Any other key under `agent.forks`, a non-boolean `enabled`, `reasoning_budget_tokens` (removed; see
[`llm.stages`](#stages)), or `stages` that isn't a list of strings fails at
startup, and a reload carrying one is rejected.

Each stage has a fork instruction:

| Key | Default copy | Notes |
|-----|--------------|-------|
| `agent.compaction_fork_prompt` | `prompts/compaction_fork.md` | Appended to the dropped prefix. |
| `memory.consolidation_fork_prompt` | `prompts/memory_consolidation_fork.md` | Followed by a sentence naming where the range starts. Restart-only, like the rest of `memory.*`. The output schema is fixed in code: an override must still ask for the reply shape: `entries` (each with a `narrative` and `topic_tags`), `participants` and `importance`. A legacy `summary` reply is still read as one entry. |
| `appraisal.stage2_fork_prompt` | `prompts/appraisal_fork.md` | Followed by quotes of the exchange's opening message and of the response sent, and the retrieval profile. |
| `critique.fork_prompt.<lens>` | `prompts/critique_<lens>_fork.md` | Followed by quotes of the exchange's opening message and of the response sent; provenance also gets each claim's search hits and the CONTEXT snapshot. An override must ask for its per-claim verdicts format or every run skips. |

These are resolved by `resolve_prompt` (#411): a string is inline text,
`{"file": path}` or a list of paths reads files relative to the config
directory, absent uses the built-in default. They aren't runtime tunables:
`set_settings` can write the key, but nothing reads the override. The
standalone prompts `appraisal.stage2_prompt` and `critique.prompt.<lens>`
are no longer read by any live stage (stage 2 and the lenses run only as
forks); the keys remain accepted.

### Prompt key format

Every prompt the framework sends on the agent's behalf goes through
`resolve_prompt` (PRD principle 15, #411). The value can be:

- **String**: inline prompt text, used verbatim. A bare string that
  happens to name an existing file is *not* treated as a path.
- **Object with `file` key**: `{"file": "path/to/prompt.md"}` — reads the
  file relative to the config directory.
- **List of strings**: each entry is a file path, read and concatenated
  with double newlines.
- **Absent or `null`**: the built-in default for that key.

A missing or unreadable file fails at startup with the key and path named.
On a config reload, a failed resolution keeps the previous value and logs
the key.

| Key | Default copy | Applies |
|-----|--------------|---------|
| `agent.compaction_fork_prompt` | `prompts/compaction_fork.md` | `agent.yaml` reload |
| `memory.consolidation_fork_prompt` | `prompts/memory_consolidation_fork.md` | restart |
| `appraisal.stage2_fork_prompt` | `prompts/appraisal_fork.md` | `agent.yaml` reload |
| `critique.fork_prompt.<lens>` | `prompts/critique_<lens>_fork.md` | `agent.yaml` reload |
| `heartbeat.prompt` | `prompts/heartbeat.md` | restart |
| `heartbeat.goals_instruction` | `prompts/heartbeat-goals.md` | restart |
| `tools.subagent.system_prompt` | built-in | restart |
| `memory.rubric_prompt` | built-in | restart |
| `silence.instruction` | built-in (`REGISTER_FRAGMENT`; `TOOL_FRAGMENT` or `BOTH_FRAGMENT` when `silence.mode` is `tool` or `both`) | restart |

Editing a prompt file does not trigger a reload; the change applies when
the key is next resolved (on reload or at restart, per the table). Whether
file edits trigger a reload is left to #408.

A fork goes to its stage's profile, else to the default chat profile. Only the model whose window is forked shares its cache, so a stage mapped to another model forks cold.

Each forkable stage tags its own LLM call (the summary, the record, the
stage-2 appraisal, each critique lens) `mode = 'fork'` or `'standalone'`
in `usage_log`, so `cached_tokens` per fork row shows whether the prefix
was reused (on llama-server, `cache_n` and `prompt_n` also record the
prompt tokens reused from cache and the ones prefilled). Other calls the
stage makes, such as consolidation's importance prior, leave `mode` NULL.

### `reasoning_history: exchange` and `preserve_thinking`

Qwen-family chat templates (Swift 1.5 included) render `<think>` on every
assistant message after the last user query, and on earlier ones only when
the server's `preserve_thinking` template kwarg is unset or true. `exchange`
mirrors that: corvidae keeps reasoning for the current exchange and drops it
for completed ones, so the old turns stop costing context budget.

Turn `preserve_thinking` off on the server when you use it. With it on, the
template renders every stripped older turn as an empty `<think></think>`
block, which is off-distribution. With it off, the template drops `<think>`
before its own last-query boundary, and corvidae's boundary is never later
than the template's, so nothing corvidae stripped is rendered.

Two known gaps, both narrow:

- `build_prompt` moves a tool-call message to where its last result
  arrived. If a user message interrupts a tool chain, a call made before
  it can render after it in the prompt. Corvidae has already stripped that
  call, and the template renders it with an empty `<think>` block.
- Compaction runs before the stripping pass, so its first token estimate
  after a new user message still counts the previous exchange's reasoning.

### Runtime-tunable gate parameters

The appraisal/critique parameters use dotted keys (e.g. `critique.sample_below_rate`, `appraisal.probe.budget_ms`) and are adjustable at runtime **without a daemon restart**, through two surfaces resolved per-decision by `corvidae.tuning.resolve_tunable` (operator directive 2, 2026-07-06):

1. **`set_settings` tool** (agent-facing, per-channel) — writes `channel.runtime_overrides`; highest precedence.
2. **Hot config reload** (operator, global) — edit `agent.yaml`; `ConfigWatcherPlugin` dispatches `on_config_reload` and plugins re-read the dotted path on the next decision.

When neither surface sets a key, the plugin's commented best-guess default applies. Resolution is first-hit-wins: per-channel override → config dotted path → default. Dotted keys never reach the LLM request body — the `extra_body` filter in `agent.py` forwards only the allowlisted inference parameters.

The persona is allowed to tune its own gates by default — a deliberate operator decision favoring emergence from experience. The per-key safety valve is `agent.immutable_settings`.

`messages`, `model`, `tools` and `tool_choice` need no entry: `set_settings` refuses every undotted key outside its allowlist (see the note below the key list).

**Recommended blocklist for multi-speaker channels with untrusted participants:**

| Key | Why block it |
|-----|--------------|
| `critique.provenance.enabled` | The correctness gate — a prompt-injected "stop second-guessing yourself" must not be able to disable the confabulation check. |
| `critique.sample_below_rate` | The standing experiment's integrity — an agent tuning its own sampling to zero blinds the critique false-negative bound. |
| `critique.prompt.adversarial`, `critique.prompt.constrained`, `critique.prompt.predictive` | These are the critic's *system prompts*. An agent that can rewrite its own critic's instructions has no critic. |
| `appraisal.stage2_prompt` | Same shape: the stage-2 appraisal system prompt. It is also where past claims are detected for the provenance check, so a prompt that never lists any switches the confabulation check off without touching `critique.provenance.enabled`. |
| `gate.engagement.enforce`, `gate.send.enforce` | An injected message should not be able to silence the agent wholesale (enforce engagement with a threshold of 1.0) or un-silence it (turn the decide gate off). |
| `gate.engagement.threshold`, `gate.send.min_salience` | The same lever by other means: a threshold above 1.0 rejects or withholds everything the gate covers once enforcement is on; 0 turns the gate off. |

These are recommendations, not hardcoded blocks: `set_settings` writes are channel-influenceable, which is exactly the persistence-of-influence shape the design polices elsewhere — but two-process discipline (agent tunes, operator audits via the outcome log) is the accepted trade-off, with the blocklist as the per-key valve.

The complete set of runtime-tunable keys (anything passed to `resolve_tunable`) as of this writing:

- `appraisal.novelty.no_probe_default`, `appraisal.prior.w_novelty`, `appraisal.prior.w_stakes`, `appraisal.prior.w_valence`, `appraisal.probe.budget_ms`, `appraisal.probe.max_tokens`, `appraisal.probe.rank_scale`, `appraisal.stage2_prompt`, `appraisal.weights.commitment`, `appraisal.weights.disagreement`, `appraisal.weights.imperative`, `appraisal.weights.novelty`, `appraisal.weights.question`
- `critique.lens.ambiguity`, `critique.lens.commitment`, `critique.lens.disagreement`, `critique.lens.neg_valence`, `critique.prompt.<lens>`, `critique.provenance.enabled`, `critique.sample_below_rate`
- `gate.engagement.enforce`, `gate.engagement.threshold`, `gate.send.allow_progress`, `gate.send.allow_thinking`, `gate.send.enforce`, `gate.send.min_salience`

The fork instruction keys (`appraisal.stage2_fork_prompt`, `critique.fork_prompt.<lens>`) are deliberately not on that list: they are read from config only, so they need no `immutable_settings` entry.

Any *dotted* key not on that list is operator-only: it is read once at init or reload, and while `set_settings` can write it into `runtime_overrides`, nothing reads the override. Undotted keys are allowlisted in code: `set_settings` accepts only the inference parameters `temperature`, `top_p`, `top_k`, `frequency_penalty`, `presence_penalty`, `max_tokens` and `seed`, which go into the LLM request body, and the framework keys `max_turns`, `max_context_tokens` and `reasoning_history`. Any other undotted key (`messages`, `model`, `tools`, `tool_choice`, …) rejects the whole call; see [threat-model.md](threat-model.md), finding 1. If you don't need agent self-tuning, `tools.disabled: [set_settings]` removes the surface altogether.

#### `appraisal.*` — stage-1 appraisal

Requires the cognition extra (`--extra cognition` on your `uv sync` line; see [cognition.md](cognition.md)). Without it these keys are ignored.

`AppraisalPlugin` scores every inbound message at the gate: surface heuristics plus an FTS5 familiarity probe on a dedicated read-only connection, under a hard latency budget, failing open. The probe connection opens lazily on first use; if `memory_fts` doesn't exist yet, novelty uses `appraisal.novelty.no_probe_default` and the open is retried at most every 30 s (a warning per failed attempt naming the path and reason, one INFO on recovery), so no restart is needed. All keys resolve per-decision.

| Key | Default | Description |
|-----|---------|-------------|
| `appraisal.probe.budget_ms` | `50` | Hard latency budget for the FTS5 probe query (not the once-per-interval connection open). Timeout ⇒ no probe result ⇒ the vector is built from surface heuristics alone (never blocks the gate). |
| `appraisal.probe.max_tokens` | `12` | Cap on sanitized tokens in the FTS5 MATCH query. |
| `appraisal.probe.rank_scale` | `10.0` | bm25 rank normalization: familiarity scales with `min(1, −rank/rank_scale)` (bm25 ranks are negative; more-negative = stronger match). |
| `appraisal.novelty.no_probe_default` | `0.5` | Novelty when no probe result arrived in budget — a value, never a null. |
| `appraisal.weights.novelty` | `0.35` | Salience weight for novelty (1 − familiarity). |
| `appraisal.weights.question` | `0.15` | Salience weight for question density. |
| `appraisal.weights.disagreement` | `0.20` | Salience weight for max(disagreement markers, negation density). |
| `appraisal.weights.commitment` | `0.20` | Salience weight for numbers/commitment density. |
| `appraisal.weights.imperative` | `0.10` | Salience weight for imperative markers (feeds salience only, not the vector). |

`critique.provenance.enabled` (default `true`) switches the provenance check. It has no other tunables; its instruction is the static `critique.fork_prompt.provenance`. Stage-2 appraisal detects the past claims and a fork of the window judges them, so the check needs the `appraisal` plugin and `agent.forks` covering `critique`. See [cognition.md](cognition.md#critiqueplugin-the-provenance-check-packagescorvidae-cognitioncorvidae_cognitioncritiquepy).

*(The `critique.*` tunable keys are not individually tabulated here; the complete list is under "Runtime-tunable gate parameters" above.)*

#### `gate.*` — engagement and decide gates

Requires the cognition extra. `GatePlugin` (entry point `gates`) reads these per decision; see [cognition.md](cognition.md#gateplugin) for what each gate does and what it records in `exchange_log.outcomes`. Both gates fail open: without `AppraisalPlugin`, or when the appraisal compute fails, nothing is rejected or withheld.

| Key | Default | Description |
|-----|---------|-------------|
| `gate.engagement.enforce` | `false` | Shadow mode while false: the engagement gate computes and records `would_reject` but admits every message. Set it only after calibrating the threshold offline against the recorded verdicts. |
| `gate.engagement.threshold` | `0.2` | Inbound salience below this would reject (and does once `enforce` is true). |
| `gate.send.enforce` | `true` | When false, the decide gate records `would_veto` but withholds nothing. |
| `gate.send.min_salience` | `0.15` | A final from a non-user origin (reminder, heartbeat, task, critique) whose outbound salience is below this is withheld: stored as `unsent`, never sent. A reply to a user is never withheld by this key. |
| `gate.send.allow_progress` | `true` | False suppresses the intermediate text sent alongside tool calls. |
| `gate.send.allow_thinking` | `true` | False suppresses reasoning (`OutboundThinking`). |

```yaml
gate:
  engagement:
    enforce: false     # shadow mode
    threshold: 0.2
  send:
    enforce: true
    min_salience: 0.15
```

### `agent.context_compact` — removed

`ContextCompactPlugin` has been removed: superseded by `MemoryPlugin` (memory consolidation and retrieval), with per-turn token stats now in the `usage_log` table. Any `agent.context_compact.*` keys in existing configs are ignored.

---

## `memory` — autobiographical memory

Requires the cognition extra (`--extra cognition` on your `uv sync` line; see [cognition.md](cognition.md)). Without it these keys are ignored.

`MemoryPlugin` consolidates dialog (at turn ends, on idle, and at compaction) into first-person memory records and retrieves them into the prompt (see [cognition.md](cognition.md#memoryplugin)). The former `dream.*` keys are gone with `DreamPlugin` (absorbed by this plugin).

A record's `created_at` is the `sessions.db` timestamp of the last message in its source range, i.e. when the conversation happened, not when consolidation ran. When a near-duplicate merge folds an older record into a new one, the survivor keeps the later of the two dates. Everything that ages a record (the retrieval score, the displayed `(age)`, retention decay and `memory.retention.grace_days`, and the `search_memory` date filters) measures from that date, so history consolidated long after the fact is aged correctly.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `memory.idle_consolidate_after` | float | `1800` | Seconds a channel must be quiet before idle consolidation runs on its un-consolidated tail, and the period between repeat runs while it stays quiet. A channel that never goes quiet is consolidated at turn ends (`memory.consolidate_after_tokens`, forks on) and through compaction instead. Also how long the turn trigger backs off after three consecutive failed turn forks on a channel (#620), unless a consolidation moves the watermark sooner, so raising it makes both idle runs and turn-fork retries rarer. Must be a positive finite number; zero, negative or non-finite values fail at startup naming this key. Takes effect at restart. |
| `memory.consolidate_after_tokens` | integer | `4000` | Window tokens of new conversation that start a turn consolidation. At the end of each turn (a final reply with no tool calls pending), consolidation forks the live window once the messages above the watermark, minus admitted memories, summaries and unsent (gate-vetoed) messages, add up to this many tokens by the window's own estimate. Only matters with `agent.forks` on for `consolidation`; with forks off the turn trigger never runs and compaction and idle do the work. Raise it for fewer, larger consolidations (each one is a fork: the cached window prefix plus the consolidation decode). Three consecutive failed turn forks on a channel back the turn trigger off for `memory.idle_consolidate_after` seconds or until a consolidation moves the watermark (#620). Must be a positive integer; `0`, negatives, floats, strings, `true`/`false` and `null` fail at startup with `memory.consolidate_after_tokens: got <value>; expected a positive integer number of window tokens (default 4000) — check agent.yaml`. Takes effect at restart. |
| `memory.half_life_days` | float | `30` | Recency decay constant of the retrieval score (`similarity × exp(-age_days / half_life)`). It is an e-folding time: the factor is `1/e`, not `1/2`, at `half_life_days`. |
| `memory.retrieval.k` | integer | `8` | Memories returned per retrieval. The KNN, and the FTS side under `candidates: hybrid`, each fetch `k × 4` candidates to rank. |
| `memory.retrieval.candidates` | `hybrid` \| `vector` | `hybrid` | Where vector-path candidates come from (#666). `hybrid` joins the KNN with an FTS5 search over `memory_fts` (common English words left out) and ranks them together by reciprocal rank fusion, so a record that names a person, project or identifier is found even when the embedding doesn't place it near the query. A record only FTS found bands on its measured cosine like any other, and one with no stored vector isn't admitted passively (#409). `vector` is the KNN alone, as before #666. FTS-degraded retrieval is unaffected. `scripts/eval_memory.py --snapshot --candidates` scores either against the same snapshot. Any other value logs a warning and uses `hybrid`. |
| `memory.retrieval.bands.strong` | float | `4.5` under `relative`, `0.75` otherwise | Value at or above which a retrieved memory bands strong and is shown with no label. Under `band_on: relative` it is a z-score; under `score` or `similarity` it is compared against the decayed score or raw similarity. Setting `bands` without a valid `band_on` (unset or rejected) logs a startup warning, because thresholds tuned as cosine values before #409 would now be read as z-scores; set `band_on: score` to keep their meaning. |
| `memory.retrieval.bands.moderate` | float | `3.0` under `relative`, `0.60` otherwise | Value at or above which a retrieved memory bands moderate and is shown `[marginal]`; below is weak, and a weak memory is never admitted (#409). Compared against the same quantity as `bands.strong`. |
| `memory.retrieval.band_on` | `relative` \| `score` \| `similarity` | `relative` | What the bands are applied to. `relative` (#409) bands the z-score of a record's raw cosine against the query's cosine distribution over every indexed, unredacted, embedded record, so the gate holds across queries of different register and an off-topic message admits nothing; with fewer than 50 embedded records (a z over N records can't exceed √(N − 1)) or no spread it bands on raw similarity at 0.75 / 0.60 instead, logged once (a failed spread scan does the same, with a warning per query), and each `retrieval_candidate` row records the `z` (NULL when none was computed). `score` bands the recency-decayed score, so with the default half-life nothing older than about 9 days can be strong. `similarity` bands raw relevance, so an old, apt memory is shown bare rather than as a guess. Candidates are ranked by the decayed score under every mode (by fused rank under `candidates: hybrid`). A record only the hybrid FTS search found bands on its measured cosine, and its `retrieval_candidate.source` reads `fts`. In FTS5-degraded retrieval there is no similarity to band on: the top `k` keyword hits are admitted, each marked `[marginal]`. `retrieval_log.top_score` (and `exchange_log.retrieval_top_score`) record the top-ranked candidate's decayed score; `retrieval_log.top_similarity` (and `exchange_log.retrieval_top_similarity`) record the same candidate's raw similarity. Under `relative` these two columns are not what the bands compare: refit the z thresholds from `retrieval_candidate.z`. Under `score` or `similarity`, calibrate from whichever of the two `band_on` names. Either way, exclude `degraded_to_fts = 1` rows, whose similarity is the 0.6 proxy rather than a measurement (`exchange_log` has no such flag; join `retrieval_log` on `trace_id` to filter it). Rows written before these columns existed read NULL. Any other value logs a warning and uses `relative`. |
| `memory.retrieval.query_context` | `previous_reply` \| `none` | `previous_reply` | What passive retrieval's query holds besides the unanswered user message(s). `previous_reply` puts the agent's last sent reply (the last assistant message with text; tool-call-only and unsent rows are skipped) before them, as its own paragraph, so a follow-up such as "thanks, that worked" retrieves records about the turn it answers (#665). `none` is the user-message-only query used before #665. `scripts/eval_memory.py --snapshot --query-context` scores either against the same snapshot. Any other value logs a warning and uses `previous_reply`. |
| `memory.retrieval.recency_from` | `created` \| `last_used` | `created` | Where the ranking age is measured from. `last_used` uses `max(created_at, last_retrieved_at)`, the same rule retention uses, so a memory stays fresh while it keeps being recalled. The `(age)` shown to the model is always measured from `created_at`. Any other value logs a warning and uses `created`. |
| `memory.consolidation_fork_prompt` | string | built-in | Overrides the instruction appended to the window for a consolidation record: to a fork when `agent.forks` is on for `consolidation`, and to the agent's rebuilt view sent on the `consolidation` stage's profile for a range that has left the window (documented copy: `prompts/memory_consolidation_fork.md`). Replaces the removed `memory.consolidation_prompt`: a config that still sets that key logs a startup warning naming this one and the key is ignored. Must be a string; a non-string value fails startup naming the key and the value, because records are stamped with the text the call was sent with ([cognition.md](cognition.md#record-stamps)). Static config, not a runtime tunable. The output schema is fixed in code: an override must still ask for the reply shape: `entries` (each with a `narrative` and `topic_tags`), `participants` and `importance`. A legacy `summary` reply is still read as one entry. |
| `memory.dup_threshold` | float | `0.95` | Near-duplicate merge threshold. When a new consolidation embedding has cosine similarity ≥ this value against an existing record anywhere in the agent's memory, on any channel, the older record is superseded: its statistics are folded into the new record and its vec row is deleted. Retrieval spans the whole memory, so every channel sees the survivor. (There is no `memory.channel_groups` any more: it was removed in #168 and a config that still sets it is not read, with no warning. Memory has no per-channel scope; for a second identity, run a second instance.) Applied only when sqlite-vec is available and the new record's embedding succeeded; silently skipped otherwise. |

### `memory.retention` — Retention job

The retention job runs as a silent background task on daemon startup if it is due, and then on the kernel clock every `memory.retention.interval`. When a run comes due it waits for 60 seconds of system-wide quiet, but at most `memory.retention.max_delay`, so steady traffic cannot starve it. It runs three passes: demotion (remove under-used records from the vector index), re-promotion (restore records whose access stats have risen since demotion), and backfill (embed re-promoted records that are missing vectors). With `memory.retention.retrieval_log_max_age_days` set, a pass between re-promotion and backfill also prunes old retrieval logs (#654). The backfill pass is skipped when embedding is disabled by the `embedding_meta` mismatch guard.

The retention score formula is `importance × (1 + 0.5 × log1p(retrieval_count)) × exp(-age_days / half_life_days)`. All threshold and shape constants are the config keys documented below.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `memory.retention.grace_days` | float | `14` | Records younger than this many days are exempt from demotion regardless of score. Ensures newly consolidated records are not immediately demoted because they have not yet been retrieved. |
| `memory.retention.importance_floor` | float | `0.8` | Records with `importance ≥ importance_floor` are never demoted for lack of retrieval traffic. The importance prior ranges 0.0–1.0 (`AppraisalPlugin` wraps the default `RubricPrior` in `AppraisalPrior` when appraisal is enabled). |
| `memory.retention.demote_below` | float | `0.15` | Retention score threshold. An indexed record whose score falls below this value (and which passes the grace-period and importance-floor checks) is demoted: `indexed=0`, `embedded=0`, vec row deleted. The `memory` row and its FTS entry are preserved. Re-promotion requires the score to rise back to or above this threshold, which happens when the record is accessed via `recall_raw`. |
| `memory.retention.half_life_days` | float | `90` | Recency half-life for the retention score. Distinct from `memory.half_life_days` (retrieval scoring, default 30). The longer retention half-life gives passive retrieval time to re-warm a record before it demotes. |
| `memory.retention.interval` | float | `21600` | Seconds between retention job runs (default 6 hours); the period of the clock timer. The last-run timestamp is persisted in the `retention_meta` table so daemon restarts do not reset the interval. A daemon idle longer than `interval` since its last run will still run the job on next start (zero-traffic guarantee). Takes effect at restart. |
| `memory.retention.max_delay` | float | `3600` | Longest a due run waits for system quiet, in seconds. A run starts when the system has been quiet 60 seconds or when this much time has passed since it came due, whichever is first. Takes effect at restart. |
| `memory.retention.retrieval_log_max_age_days` | float | unset (off) | Age in days after which the retention job deletes `retrieval_log` rows and their `retrieval_candidate` rows (#654). Unset or `null` keeps every row; the log then grows by about 35 candidate rows (~1.5 KB) per user turn. These tables are the calibration stream behind `scripts/eval_memory.py` and threshold tuning, so pick an age that covers the period you want to measure. Must be a positive finite number; `0`, negatives, `true`/`false`, strings and non-finite values fail at startup with `memory.retention.retrieval_log_max_age_days: got <value>; expected a positive number of days, or leave it unset to keep every retrieval_log row — check agent.yaml`. Takes effect at restart. |
| `memory.retention.backfill_batch` | integer | `32` | Maximum number of unembedded indexed records the retention job embeds per run (backfill pass). Bounds embedding API usage when many records need vectors, such as after a re-promotion wave or a period during which embedding was disabled. |

---

## `scheduler` — reminders and periodic jobs

`SchedulerPlugin` fires due rows of its `schedule` table into their channels as notifications (see [design.md](design.md#schedulerplugin)). The agent creates one-shot reminders with the `remind_me` tool, always on the channel it is talking on; the operator declares recurring jobs here. The heartbeat is no longer a scheduler job: it is its own plugin, started on system idle (see [`heartbeat`](#heartbeat--a-turn-on-system-idle)). The block is optional: without it the tools still work, in UTC, with the defaults below.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `scheduler.timezone` | IANA name | `UTC` | Timezone for `when` values with no offset (ISO without offset, bare `HH:MM`), for `at` anchors, for day-step recurrence (a daily job keeps its wall-clock time across DST), and for times shown to the model. An unknown name logs a warning and uses UTC. |
| `scheduler.catch_up_window` | float (seconds) | `3600` | On startup, a one-shot reminder that came due while the daemon was down fires once (prefixed `[overdue by …, daemon was down]`) if it is late by less than this; older ones are disabled as `missed`. |
| `scheduler.max_reminders_per_channel` | integer | `50` | Pending tool-created reminders allowed per channel. `remind_me` refuses beyond it. Bounds what a hostile speaker can park in the table ([threat-model.md](threat-model.md)). |
| `scheduler.jobs` | list | `[]` | Operator jobs, synced by `name` on startup and on every reload. See below. |
| `scheduler.jobs_file` | path | — | Path to a YAML file with job definitions (relative to `agent.yaml`). When set and the file exists, jobs are read from the file instead of `scheduler.jobs`. The file is watched for changes (mtime polling, 5 s); edits trigger a re-sync without a restart. Format (YAML):
```
jobs:
  - name: morning-checkin
    channel: signal:4d8d...
    cron: "0 8 * * *"
    origin: reminder
    prompt: "Good morning."
```
Cron supports `*`, `N`, `N-M`, `*/N`, lists, and range-with-step. The file is watched for changes (mtime polling, 5 s); edits trigger a re-sync without a restart. A malformed file leaves existing rows alone (logged, not fatal). |

Each job:

| Key | Required | Description |
|-----|----------|-------------|
| `name` | yes | Unique; the sync key. Renaming a job disables the old row and creates a new one. |
| `channel` | yes | `transport:scope` to fire into. Any channel, including one no transport owns, such as `internal:notes`, whose replies go to the log and `sessions.db` only. |
| `origin` | no | `reminder` (default); becomes `annotations["origin"]`, which the model sees as `[annotations: origin=reminder]` under the notification header. `heartbeat` is still accepted so old configs load, but a scheduler heartbeat is superseded by the [`heartbeat`](#heartbeat--a-turn-on-system-idle) plugin; delete the job when you enable the plugin, or both will run turns on `internal:heartbeat`. |
| `every` | one of `every`/`cron` | Interval: `30s`, `20m`, `6h`, `1d`, `7d`, `1h30m`, or a number of seconds. Mutually exclusive with `cron`. |
| `cron` | one of `every`/`cron` | Cron expression: `M H DOM MON DOW`. Fire times are the next matching slots. Mutually exclusive with `every`. |
| `at` | no | Anchor. `HH:MM`: the first fire is the next such time, then every `every` (or next cron slot). An ISO datetime: slots are that instant plus multiples of `every`. Without it, the first fire is one interval after the job is first synced (or the next cron slot). |
| `prompt` / `prompt_file` | one of them | The job's text. `prompt_file` is relative to the directory holding `agent.yaml`. |

The next fire time is computed from the scheduled time, not the fire time, so jobs do not drift; a job more than one interval behind (after downtime) skips to the next slot rather than replaying. On reload, a changed `every`, `at`, or `timezone` reschedules a job from now; a changed `prompt`, `channel`, or `origin` keeps its schedule. A job removed from the list is disabled (`removed_from_config`), which survives restart. An invalid entry logs an ERROR and leaves its existing row as it was.

```yaml
scheduler:
  timezone: America/Los_Angeles
  jobs:
    - name: weekly-review
      channel: irc:#lex
      every: 7d
      at: "2026-10-16T09:00"
      prompt: "Post a short summary of what we worked on this week."
    - name: morning-brief
      channel: signal:4d8d...
      cron: "0 8 * * *"
      prompt: "Good morning. Here's what's on today."
```

The four CRUD tools (`add_job`, `update_job`, `remove_job`, `list_jobs`) write to the `jobs_file` and trigger a hot reload. They are registered on every channel; see [threat-model.md](threat-model.md) for the exposure.

A job costs a main-model turn per firing on the shared backend; keep intervals in hours. The firing text is framed as data for the model to judge ("Decide whether and how to act on it"), not as an order.

---

## `heartbeat` — a turn on system idle

`HeartbeatPlugin` (cognition) starts a turn when the whole system has been quiet for a while: no channel has had inbound work for `heartbeat.after` seconds. The turn runs on `internal:heartbeat`, a channel no transport owns, so nothing it writes there reaches anyone. Its prompt asks the agent to review its recent memories and act only if something warrants it (see [cognition.md](cognition.md#heartbeatplugin)). The block is optional, and without it the heartbeat is asleep.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `heartbeat.after` | number (seconds) | none | Seconds of system-wide quiet before a beat, and the minimum gap between beats: each beat makes the system busy, so the next one comes `after` seconds after it settles. Traffic on any channel cancels the pending beat and starts the count again. Absent, `null`, or an empty `heartbeat:` block: the heartbeat is asleep. Zero, negative, non-finite, boolean or non-numeric values fail startup with a `ValueError` naming the key and the value. Read at startup only. |
| `heartbeat.memory_limit` | integer | `20` | How many of the newest memories a beat reviews, across every channel; the limit bounds each beat's notification; compaction bounds the window. `0` reviews none. Negative, fractional, boolean or string values fail startup with a `ValueError` naming the key, the value and what is accepted, even while the heartbeat sleeps. Read at startup only. |
| `heartbeat.retired_goal_limit` | integer | `20` | How many of the newest retired goals a beat reviews, when the goals plugin is loaded; `0` reviews none. Same validation and read timing as `heartbeat.memory_limit`. |
| `heartbeat.prompt` | string / file / list | built-in | The beat's instruction. Resolved by `resolve_prompt` (#411): a string is inline text, `{"file": path}` or a list of paths reads files relative to the config directory, absent uses the built-in default. `prompts/heartbeat.md` is the documented copy. Read at startup only. |
| `heartbeat.goals_instruction` | string / file / list | built-in | Appended when the goals plugin is loaded: the instruction to review open and retired goals. Same resolution as `heartbeat.prompt`. `prompts/heartbeat-goals.md` is the documented copy. Read at startup only. |

**Sleep.** The heartbeat sleeps whenever `heartbeat.after` is absent. To put it to sleep, delete or comment out the block and restart; it stays asleep across every restart after that, because `agent.yaml` is the record. `plugins.disabled: [heartbeat]` does the same. Reloading the config doesn't wake it or put it to sleep; restart.

**Memory.** Setting `heartbeat.after` requires the `memory` plugin; with memory disabled, startup fails with a message saying so. The beat reviews the most recent memories (up to `heartbeat.memory_limit`, default 20) of the whole memory, every channel's, newest first. Memory has no per-channel scope, so the beat sees every channel's records, its own earlier ones included. By the same token, what a beat's turn consolidates into memory, anything its tools fetched included, is retrievable from every channel (see [threat-model.md](threat-model.md)).

```yaml
heartbeat:
  after: 3600           # an hour of quiet
```

**Cost.** Each beat is a main-model turn on the shared backend. A beat waits for quiet, so it never competes with a conversation, but a short `after` keeps the system from staying quiet long enough for memory retention's 60-second quiet window; retention then runs on its `max_delay` instead.

Compaction bounds a beat's window (`agent.compaction_threshold` and `agent.compaction_retention`, applied to the heartbeat channel's `max_context_tokens`). Each beat appends to a stable prefix, so a backend that caches prompt prefixes reuses it. `channels.internal:heartbeat.max_context_tokens` (see [`channels`](#channels--per-channel-overrides)) sets how often it compacts. The log keeps every row, and memory consolidation keeps the durable record.

---

## `skills` — skills directory and index (#140)

| Key | Type | Default | Notes |
|-----|------|---------|-------|
| `skills.directory` | string | `skills` | Path to the skills directory (relative to the workspace or absolute). Each subdirectory with a `SKILL.md` file is a skill. The index (names + descriptions) is contributed to the system prompt via `transform_system_prompt` at conversation start and compaction boundaries. The `use_skill` tool loads a skill's full text on demand. |
Skills are agent-writable (principle 15). The idle scan (#139, planned)
will watch skill-directory writes as a persistence-of-influence channel.
See [threat-model.md](threat-model.md#skills-directory-reaches-the-system-role-140).

## `silence` — the right to silence

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `silence.instruction` | string / file / list | built-in | The paragraph `SilencePlugin` appends to a conversational channel's system prompt, telling the model it may stay silent, by replying `NO_REPLY` or, with `silence.mode` `tool` or `both`, by calling `end_turn` (see [cognition.md](cognition.md#silenceplugin-packagescorvidae-cognitioncorvidae_cognitionsilencepy)). Same resolution as `heartbeat.prompt`. Keep the literal `NO_REPLY` in any replacement in `token` or `both` mode: the veto drops any message containing that token (a containment check, not an exact match), not your wording; in `tool` or `both` mode the replacement must also mention `end_turn`. Read at startup only. |
| `silence.mode` | string | `token` | How the agent is asked to decline a turn (#797). `token`: the instruction tells it to reply `NO_REPLY`. `tool`: `SilencePlugin` registers an `end_turn(reason)` tool and the instruction tells the model to call it. `both`: the tool first, `NO_REPLY` as a fallback. The tool works on conversational transports only and refuses on any other channel, web, CLI and ACP included. The veto of a `NO_REPLY` final is on in every mode. Any other value, or a `silence:` block that isn't a mapping, fails startup with a `Config error` naming the key, the value and what is accepted. A custom `silence.instruction` replaces the built-in text in every mode, so in `tool` or `both` mode it must mention `end_turn`. Read at startup only. |

---

## `funnel` — context-admission funnel

Requires the cognition extra (`--extra cognition` on your `uv sync` line; see [cognition.md](cognition.md)). Without it these keys are ignored.

`FunnelPlugin` is the single chokepoint for tail CONTEXT admission (dedupe, budgets, injection framing). It also owns deferred registration: non-tool-call notification payloads queue per `(channel, origin)` via `ingest_and_wake()`, one stub notification wakes the channel per pending pair, and the drain in `before_agent_turn` admits everything queued for the triggering exchange's origin. `ingest()` queues a payload with no stub and no wake: it is admitted at the channel's next turn, whatever that turn's origin. Providers can also submit promises of entries through the `on_funnel_ingest` hook (see [plugin-guide.md](plugin-guide.md#implementing-on_funnel_ingest)). The deferred registry is in-memory by design (payloads pending at shutdown are dropped). This change adds one key, `funnel.ingest_ttl`, below; drained payloads spend the same per-source budgets.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `funnel.default_budget` | integer | `512` | Token budget per `admit()` call when the source has no override. |
| `funnel.budgets.<source>` | integer | — | Per-source budget override (e.g. `funnel.budgets.memory`). The goals plugin admits under `funnel.budgets.goals`, which falls back to `funnel.default_budget` (`512`). |
| `funnel.window_budgets.<source>` | non-negative integer | unset (no cap) | Cap on the tokens of one source's CONTEXT a single window holds (e.g. `funnel.window_budgets.memory: 4096`), so admission can't crowd the window between compactions (#632). Each admission fits only what is left under the cap after that source's rows already in the window, frames included, and never more than the per-call budget; the new row's own frame (~40 tokens) is not charged, so the window can pass the cap by that much. Compaction frees the rows it drops; rows in the retained tail still count. A drop the cap caused is logged at info naming the channel, the tokens held, the cap and this key; a deferred entry larger than the cap is dropped with a warning. `0` disables the source. A non-integer, bool or negative value, or a `window_budgets` that is not a mapping, logs a warning naming the key, the value and what is accepted, and is ignored (that source stays uncapped). Re-read on config reload. |
| `funnel.ingest_ttl` | number of seconds, greater than `0` | unset (no expiry) | How long a stub-less queued payload (`ingest`) or a pending `on_funnel_ingest` promise may wait before the funnel drops it; an unresolved promise is cancelled. Age counts from creation, so a promise's age runs from its submit and a result that finished while the channel was idle expires too. Payloads queued with `ingest_and_wake` never expire. Each drop is logged at info naming the source, the channel, the age and this key. A value that is not a number greater than 0 (a bool, a string, zero or a negative number, say) logs a warning naming the key, the value and what is accepted, and expiry stays off. A reload to an invalid value also turns expiry off rather than keeping the previous value. Re-read on config reload. |

---

## `markdown_recall` — markdown-folder recall source

Requires the cognition extra (`--extra cognition` on your `uv sync` line; see [cognition.md](cognition.md)). Without it these keys are ignored.

`MarkdownRecallPlugin` (entry point `markdown_recall`) is a recall source: on every plain user turn it searches a folder of markdown files for what the turn is about and hands the top hits to the funnel as framed CONTEXT. The folder is the source of truth; the SQLite index under `db_path` (a vec0 vector surface + FTS5, kept current by a startup and an idle reconcile) is a cheap-to-rebuild cache. A search before the index is built resolves to nothing, never a block; an embedding failure degrades the affected chunks to FTS5-only. There is no `on_funnel_ingest` of its own — the [recall mixin](plugin-guide.md#implementing-on_funnel_ingest) owns the gate, cooldown and timeout, and this plugin supplies `search` and `render`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `markdown_recall.root` | path | `state/markdown_recall` | The folder of markdown files to mirror, relative to the workspace. Symlinks are not followed out of the folder; frontmatter is parsed with `yaml.safe_load`. A missing or unreadable folder is fail-soft: no index, no CONTEXT, no crash. |
| `markdown_recall.db_path` | path | `state/markdown_recall/index.db` | The SQLite index file. The files under `root` are the source of truth; this is a cache, safe to delete (the next reconcile rebuilds it). |
| `markdown_recall.max_hits` | positive integer | `5` | The top-k hits returned per search, in fused rank order (vec KNN + FTS5 by reciprocal rank, as memory does). A non-integer, bool or non-positive value is a config error at startup, naming the key and the value. |
| `markdown_recall.cooldown` | seconds, `>= 0` | mixin default | Seconds between searches per channel (the recall mixin's `SearchGate`). A search while the cooldown is active is skipped; the next turn asks again. |
| `markdown_recall.timeout` | seconds, `> 0` | mixin default | Seconds a search may run before it is cancelled and the turn proceeds without it (fail-soft). |

The source admits under `funnel.budgets.markdown_recall` (falling back to `funnel.default_budget`), so a per-source token cap is available like any other source's.

---

## `tools` — built-in tool settings

The built-in tool keys are read by the plugin that owns the tool: `shell_timeout` by `ShellToolPlugin` (entry point `shell_tools`), `max_file_read_bytes` by `FileToolsPlugin` (`file_tools`), and the `web_*` keys by `WebToolsPlugin` (`web_tools`). These three replaced `CoreToolsPlugin` (`core_tools`); a config that still lists `core_tools` under `plugins.disabled` disables nothing and logs a WARNING, so list the plugins you meant instead (`shell_tools` also removes `task_pipeline`). `tools.max_result_chars` and the other dispatch keys are read by `ToolCollectionPlugin`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `tools.max_result_chars` | integer | `100000` | Longest tool result kept whole. A longer one reaches the model as its first `max_result_chars` characters behind the kernel's cut line (`[metadata: cut to the first N of M characters; the full text is in <path>]`), and its full text is written to a spill file under the `daemon.notification_spill_*` settings, where the channel has a tool that reads spill files. A subagent's results get the same words as a plain-text line. A background task's result that answers no tool call is cut and spilled the same way, by `TaskPlugin`. Read by `ToolCollectionPlugin`. The legacy key `agent.max_tool_result_chars` is accepted but deprecated; use `tools.max_result_chars` instead. |
| `tools.read_file_window_fraction` | number | `0.08` | The fraction of the channel's `max_context_tokens` one `read_file` page may take. A page is `min(tools.max_result_chars, max(2000, fraction x max_context_tokens x 3.0))` characters, so it never exceeds `tools.max_result_chars`; a call with no channel, or with no channel registry loaded, uses `tools.max_result_chars`. Must be a number with `0 < value <= 1`; any other value is a config error at startup and refuses a reload. Read by `ToolCollectionPlugin`. |
| `tools.shell_timeout` | integer | `30` | Seconds before a `shell` command is killed. |
| `tools.web_fetch_timeout` | integer | `15` | Seconds before a `web_fetch` HTTP request is aborted. |
| `tools.web_max_response_bytes` | integer | `50000` | Maximum bytes of HTTP response body returned by `web_fetch`. Content beyond this limit is truncated with `[truncated]`. This limit is independent of `tools.max_result_chars`. |
| `tools.max_file_read_bytes` | integer | `1048576` | Maximum file size in bytes that an unranged `read_file` will return. Files larger than this limit return an error string telling the agent to pass `offset` and `length`; a ranged read (character or line range) is not bounded by it. Default is 1 MB (1,048,576 bytes). |
| `tools.web_search_max_results` | integer | `8` | Maximum number of DuckDuckGo results returned by the `web_search` tool. |
| `tools.call_timeout` | number or null | `600` | Seconds a single tool call may run before dispatch cancels it and returns `Error: tool '<name>' timed out after <n>s` to the model, so the turn continues and the task worker is freed. `0` or `null` disables the bound. Read by `ToolCollectionPlugin`; covers every dispatch path (Agent, subagent loop, ACP backend). Skipped for a tool whose own timeout is not longer: `shell` (`tools.shell_timeout`), `web_fetch` (`tools.web_fetch_timeout`) and MCP tools (`timeout_seconds`) report their own timeout error instead. A negative, non-numeric or non-finite value is rejected at startup. |
| `tools.timeouts` | mapping of tool name to number or null | `{}` | Per-tool override of `tools.call_timeout`; `0` or `null` unbounds that tool only. Example: `timeouts: {subagent: 3600, task_pipeline: 3600}`. `subagent` returns a task id at once, so in practice only `task_pipeline` (120 s per step) needs a longer bound; neither is a default. An entry naming no registered tool logs one `WARNING` and is kept (an MCP tool registers when its server connects). A bad entry is rejected at startup. |
| `tools.disabled` | list of strings | `[]` | Tool names dropped after collection, from any plugin (built-in or MCP), in `ToolCollectionPlugin.rebuild_registry`. This is the single chokepoint every registered tool passes through, so a disabled tool is absent from every channel's schemas and from subagents. `shell` and `task_pipeline` disable together: `task_pipeline` shells out on its own, so naming either removes both (PRD §3.5.3). A value that is not a list of names (a bare string, say) is rejected at startup. Not hot-reloaded. To narrow one channel instead, use `channels.<id>.tools.disabled`. See [threat-model.md](threat-model.md). |
| `subagent.max_turns` | integer | `10` | Tool-round budget per subagent, passed by `SubagentPlugin._launch` to `run_agent_loop` (#494). The default of 10 is too small for fetch-heavy research tasks — on 2026-11-04 three parallel subagents each exhausted 10 rounds mid-fetch and delivered no usable result. When a subagent hits the budget, the task result is the marker `(max tool-calling rounds reached)` plus, when the subagent's last assistant message carried text, that text as `Last progress: …` — a partial result, not a bare four-word notification. A non-positive-integer value (a bool, a float, a numeric string, zero, a negative) is rejected at startup with a one-line config error. Applies on restart. |

---

## `irc` — IRC transport

The `irc` section is optional. If absent, `IRCPlugin` does not connect.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `irc.host` | string | `irc.libera.chat` | IRC server hostname. |
| `irc.port` | integer | `6667` | IRC server port. |
| `irc.nick` | string | `corvidae` | Bot nickname. |
| `irc.tls` | boolean | `false` | Use TLS for the IRC connection. |
| `irc.allow` | list of strings | `[]` (nobody) | Senders who may drive the agent, in channels and in private messages. **IRC is default-deny:** with no `allow`, or an empty one, every sender is refused, so a config that used IRC before this key existed goes silent until it is set. Startup logs one WARNING naming `irc.allow` when the key is absent. Where the network reports services accounts (IRCv3 `account-notify`/`account-tag`, or account data pydle learned from WHOIS), an entry matches the sender's **account**, not the nick: a user logged in as `alice` is admitted under any nick, and someone using the nick `alice` who is logged in as another account, or not identified (or whose account the bot doesn't know), is refused. With `account-tag` negotiated the account comes only from that message's own tag (no tag means not logged in, and the nick is never used as a fallback); pydle's tracked user record is not consulted, so a stale record, a WHOIS reply or another line can't admit a later user of the same nick. Each message is judged once, when it arrives, and that verdict is the one used for it: a later line in the same socket read can't change it. Without `account-tag`, the account is whatever pydle's tracked user record holds at that moment, which can lag a logout for a sender who shares a channel with the bot (a residual risk, see `docs/threat-model.md`). A message from a sender who shares no channel with the bot, private or to a channel they haven't joined, is refused even if their tracked account is listed, because that record is never kept current (no logout reaches it); the INFO line says so and names the remedy (join a channel the bot is in, or use a network with `account-tag`). When the network reports accounts (`account-notify` or `extended-join`) but not `account-tag`, and `allow` is set, startup logs one WARNING per connection saying this. On a network with no account support, an entry matches the **nick**, which is unauthenticated: anyone can take the nick, so treat the list as a convenience there, not a credential. Matching uses the server's CASEMAPPING (ascii, rfc1459), so `alice[` and `alice{` are one nick on an rfc1459 network. A refused sender gets no reply, no channel and no history, and one INFO line names the nick (and account when known). Anything but a list of strings raises at startup: `irc: config field 'allow' must be a list of nicks or account names`. Read at startup; changing it needs a restart. |
| `irc.channels` | list of strings | `[]` | IRC channels to join on connect (e.g., `["#general", "#dev"]`). |
| `irc.message_chunk_size` | integer | `400` | Maximum UTF-8 bytes per outbound IRC message. Messages longer than this are split using three-tier splitting: paragraphs → sentences → words. Must be at least 6; a smaller value raises `ValueError` the first time a reply is split. |

---

## `signal` — Signal transport

The `signal` section is optional. If absent, `SignalPlugin` loads but stays inert: no socket, no channels, nothing logged above `DEBUG`. If present, it must be well-formed — a missing `socket` or `account`, or an `allow` that is not a list, raises at startup naming the offending field.

corvidae talks JSON-RPC to a `signal-cli daemon` that is already running; it does not start or supervise one. Registering the bot's number and running the daemon are covered in [signal-ops.md](signal-ops.md).

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `signal.socket` | string | *required* | Path to the `signal-cli daemon --socket` unix socket. |
| `signal.account` | string | *required* | The bot's own number, E.164, e.g. `"+15550001111"`. Quote it — bare `+…` is not valid YAML. Used to recognize and ignore the account's own traffic. |
| `signal.allow` | list of strings | `[]` | Senders permitted to drive the agent. Each entry is either an E.164 number or an ACI (the account UUID). Default-deny: an empty or absent list refuses everyone, silently — no reply, no typing indicator, no read receipt. |
| `signal.message_chunk_size` | integer | `2000` | Maximum UTF-8 bytes per outbound Signal message. Longer replies are split paragraphs → sentences → words and sent as several messages; nothing is truncated. Must be at least 6; a smaller value raises `ValueError` the first time a reply is split. |
| `signal.attachment_dir` | string | `state/signal-attachments` | Directory where downloaded attachments are written. Created on first download. Files are not pruned automatically — retention is the operator's responsibility. |
| `signal.max_attachment_size` | integer | `10485760` (10 MB) | Largest single file, in bytes, in either direction. A larger inbound attachment is rejected with `[<type>: download failed]`; a larger file passed to `send_signal_attachment` is refused with an error naming this setting. Must be a positive integer. The value also sizes the StreamReader buffer limit for the signal-cli connection (the cap in base64 terms, 4/3, plus 64 KiB for JSON framing; a response that trips the limit fails its in-flight RPC immediately), so raising it for larger outbound files raises that inbound limit too. |
| `signal.send_attachment_root` | string | `state` | The only directory tree the `send_signal_attachment` tool may read from. A relative path is read under this root, and anything resolving outside it is refused with an error naming the root. Must be a non-empty string; an empty value, a number, or an explicit `null` raises `ValueError` at startup. It is defence in depth, not a security boundary: `shell` and `read_file` are unjailed already, so widen it only as far as the agent needs. |

```yaml
signal:
  socket: /run/corvidae/signal.sock
  account: "+15550001111"
  allow:
    - "+15551234567"
  message_chunk_size: 2000
```

E.164 entries in `allow` are resolved to ACIs by a batched `getUserStatus` call after the transport connects. A number that does not resolve — not registered on Signal, or a rate-limited lookup — is logged at `ERROR` and that entry simply never matches. Channels, however, are always keyed on the ACI: `sessions.db` rows and log lines show `signal:<aci>` whatever form the config used.

**Signal content is stored in plaintext on this host.** Messages are end-to-end encrypted in transit and then written to `sessions.db`, to the JSONL conversation log (`<daemon.jsonl_log_dir>/signal_<aci>.jsonl`, when that is enabled), and into consolidated memory records, all as ordinary text — and the channel id itself contains a personal identifier. File permissions and whatever disk encryption the host has are all that protect them. Disappearing messages are stored the same way as any other message; the transport does not treat them differently.

To remove a conversation's stored content, find its message ids and tombstone them with `corvidae redact` (content is overwritten in place; rows and ids remain):

```sh
sqlite3 sessions.db \
  "SELECT id FROM message_log WHERE channel_id = 'signal:<aci>' ORDER BY id"
uv run corvidae redact --db sessions.db message <id> [<id> ...]
```

Pass the ids explicitly rather than using the `range` form: `message_log` ids are global across all channels, so an id range spans whatever other channels were active in the same window. The redaction cascades to intersecting memory records, to both FTS surfaces, and to the text `exchange_log` copies from the conversation (`past_claims`, `provenance_snapshot`); see "Operator redaction" in [design.md](design.md). The jsonl logs are separate files and are not touched — delete them yourself.

---

## `discord` — Discord transport

The transport needs the Discord SDK: add `--extra discord` to your `uv sync` line. A `discord:` block without it fails at startup with a message saying so; with no block, the SDK is never loaded.

The `discord` section is optional. If absent, `DiscordPlugin` loads but stays inert: no client, no tasks, nothing logged above `DEBUG`. If present, it must be well-formed — no token (in the block or the environment), an `allow` that is not a list, a `message_chunk_size` that is not an integer from 100 to 2000, a `require_mention` or `allow_bots` that is not a boolean, a `channels` entry keyed on anything but a numeric channel id, or an unknown key inside one, raises at startup naming the offending field and the remedy.

corvidae logs in as a Discord bot account. Creating the application, getting the token, and inviting the bot are covered in [discord-ops.md](discord-ops.md). In DMs the bot answers every message from an allowed user; in servers and threads it answers only messages that @-mention it, unless `discord.require_mention` is `false`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `discord.token` | string | `$DISCORD_TOKEN` | The bot token. If the key is absent, null, or empty, the `DISCORD_TOKEN` environment variable is used; if neither is set, startup fails. |
| `discord.allow` | list of user ids | `[]` | Discord user ids permitted to drive the agent. Numeric ids, written as YAML ints or quoted strings; they are compared as strings. Default-deny: an empty or absent list refuses everyone, silently — no reply and no typing indicator. An explicit `allow: null` is an error, not an empty list. |
| `discord.message_chunk_size` | integer | `2000` | Maximum UTF-8 bytes per outbound Discord message, 100 to 2000 inclusive. Longer replies are split, keeping code fences intact, and sent as several messages; nothing is truncated. An explicit `null` is rejected (unlike a null `token`, which falls back to the environment). The limit is bytes, while Discord's 2000 is characters, so text with multi-byte characters splits earlier than strictly needed. |
| `discord.allow_bots` | boolean | `false` | Whether messages from other bot accounts reach the agent. With `false` (the default) every message authored by a bot is dropped before the allowlist is consulted, so two agents cannot hear each other. With `true` a bot-authored message is treated like any other: it must still come from an id in `discord.allow`, and the bot still never answers itself, so opening this admits one named bot at a time, never every bot in the server. Keep the allowlist short: two bots that both answer every message will answer each other until something stops them, and `require_mention` is what makes stopping possible. Read at startup only. |
| `discord.channels` | mapping | `{}` | Per-channel overrides, keyed by the Discord channel id — the number alone, quoted or not, as it appears after `discord:` in the agent's logs — digits, no prefix, no leading zeros. Each entry is a mapping whose only key is `require_mention`; listing one id twice (quoted and as a number) fails startup. A thread uses its parent channel's entry; a thread whose parent has no entry stays on `discord.require_mention`. A key that is not a bare number (`discord:123…`, `#general`) fails startup rather than matching nothing, because a silently ignored override is a channel left conversational. An empty section means no overrides. Read at startup only. |
| `discord.require_mention` | boolean | `true` | The default for every channel. With `true`, servers and threads get an answer only when a message @-mentions the bot. With `false`, every message from an allowlisted user in any channel the bot can see reaches the agent, which may still decline to reply; a mention token, if present, is still stripped. Messages from anyone else are refused by the allowlist before they are stored, as always. Answering unaddressed anywhere makes the client request the privileged Message Content intent, which must also be switched on for the application in the Developer Portal — the intent is per client, so one conversational channel is enough to need it. Per-channel overrides live in `discord.channels` below. Read at startup only. |
| `discord.attachment_dir` | string | `state/discord-attachments` | Directory where images attached to inbound messages are written. Created on first download; files are not pruned automatically, so retention is the operator's responsibility. |
| `discord.max_attachment_size` | integer | `10485760` (10 MB) | Largest single file, in bytes, in either direction. A larger inbound attachment is noted as `[image: download failed]`; a larger file passed to `send_attachment` is refused with an error. |
| `discord.send_attachment_root` | string | `state` | The only directory tree the `send_attachment` tool may read from. A relative path is taken relative to this root, and anything resolving outside it is refused with an error naming the root. It is defence in depth, not a security boundary: `shell` and `read_file` are unjailed already, so widen it only as far as the agent needs. |

```yaml
discord:
  token: "MTIz...your.bot.token"   # or leave out and set DISCORD_TOKEN
  allow:
    - 123456789012345678
  message_chunk_size: 2000
  # require_mention: false         # answer allowed users without an @-mention
  # allow_bots: true               # let an allowlisted bot's messages through
  channels:                       # per-channel overrides (#512)
    "1568093636044787772":        # the channel id the agent sees
      require_mention: false      # this one channel may answer unaddressed
```

User ids are the long numeric snowflakes you get from Discord's Developer Mode, not usernames; see [discord-ops.md](discord-ops.md#find-ids). Prefer the environment variable if `agent.yaml` is world-readable or checked in anywhere: the token is the bot, and anyone holding it can act as it.

**Discord content is stored in plaintext on this host, and Discord has it too.** Discord messages are not end-to-end encrypted; Discord (the company) can read them in transit and at rest. On this side they are written to `sessions.db`, to the JSONL conversation log (`<daemon.jsonl_log_dir>/discord_<channel_id>.jsonl`, or `discord_dm_<user_id>.jsonl` for a DM, when that is enabled), and into consolidated memory records, all as ordinary text. Messages from senders who are not allowlisted are refused before they are persisted. In a server channel, though, the stored history is whatever allowlisted users said there, and the model's replies to it. File permissions and whatever disk encryption the host has are all that protect the local copies.

To remove a conversation's stored content, use `corvidae redact` as described for Signal above, with `channel_id = 'discord:<channel_id>'` or `'discord:dm:<user_id>'`.

---

## `zulip` — Zulip transport

The transport needs no SDK and no extra: it talks to Zulip's REST API over aiohttp, which corvidae already depends on. The server must be Zulip 9.0 (feature level 248) or newer; an older one is refused at startup with an ERROR naming the server and its level.

The `zulip` section is optional. If absent, `ZulipPlugin` loads but stays inert: no session, no tasks, nothing logged above `DEBUG`. If present, it must be well-formed — a missing `site`, `email` or `api_key` (the last two may come from the environment), a `site` that is not a bare `http(s)://host[:port]` URL, an `allow` that is not a list of user ids, a `message_chunk_size` outside 100 to 10000, a non-boolean `require_mention`, an empty or over-long `default_topic`, or a `channels` entry keyed on anything but a numeric stream id, or with an unknown key, raises at startup naming the field and the remedy.

corvidae logs in as a Zulip bot user (a **Generic bot**); creating it, subscribing it and finding ids are in [zulip-ops.md](zulip-ops.md). In 1:1 DMs every message from an allowed user reaches the agent. In channels and group DMs only a message that mentions the bot does, directly or through a user group it belongs to (Zulip's `mentioned` flag; a wildcard such as `@**all**` doesn't count), unless `zulip.require_mention` is `false`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `zulip.site` | string | required | The organisation's URL, `scheme://host[:port]` with no path. Plain `http://` to anything but loopback logs a WARNING, because the API key and every message cross the network unencrypted. |
| `zulip.email` | string | `$ZULIP_EMAIL` | The bot's email. If the key is absent, null or empty the environment variable is used; if neither is set, startup fails. |
| `zulip.api_key` | string | `$ZULIP_API_KEY` | The bot's API key, resolved the same way. Prefer the environment variable if `agent.yaml` is readable by anyone else: the key is the bot. It is never logged. |
| `zulip.allow` | list of user ids | `[]` | Zulip user ids permitted to drive the agent, as YAML ints or quoted digit strings (compared as strings). Default-deny: an empty or absent list refuses everyone, silently. An email or a name is rejected at startup, naming the entry, because neither is stable. |
| `zulip.require_mention` | boolean | `true` | The default for every channel and group DM. With `true` only a mention of the bot (directly, or through a user group it belongs to) reaches the agent. With `false` every message from an allowlisted user the bot can see reaches the agent, which may still decline to reply; the bot's own mention token, if present, is still stripped. 1:1 DMs never need a mention. Read at startup only. |
| `zulip.channels` | mapping | `{}` | Per-channel overrides, keyed by Zulip **stream id** (digits, no prefix, no leading zeros). Each entry's only key is `require_mention`. A name, a `zulip:` prefix or an unknown key fails startup rather than matching nothing. Read at startup only. |
| `zulip.message_chunk_size` | integer | `10000` | Maximum UTF-8 bytes per outbound message, 100 to 10000. The server's own `max_message_length` is used instead when it is smaller. Longer replies are split and sent as several messages; nothing is truncated. The limit is bytes while Zulip's is code points, so multi-byte text splits earlier than strictly needed. |
| `zulip.default_topic` | string | `corvidae` | The topic a reply goes to when nothing is remembered for the channel: the first turn after a restart, or a channel no allowed message has reached yet. Otherwise a reply, a reminder or a cross-channel send goes to the topic of the latest allowed message that reached the agent there. 1 to 60 characters, so it fits every Zulip server. |

```yaml
zulip:
  site: https://chat.example.org
  email: corvidae-bot@chat.example.org   # or leave out and set ZULIP_EMAIL
  api_key: "..."                          # or leave out and set ZULIP_API_KEY
  allow:
    - 123
  # require_mention: false                # answer allowed users without an @-mention
  channels:
    "7":                                  # the stream id, as in `zulip:7` in the logs
      require_mention: false
  default_topic: corvidae
```

**Scopes and topics.** A channel is `zulip:<stream_id>`; a topic is a name anyone can rename, so it is never part of an id (PRD principle 10). Every topic in a channel shares one history. A reply goes to the topic of the newest allowed message, which is held in memory only. The model sees the channel as `#general > plans`.

**Replies can't ping.** Zulip has no allowed-mentions switch, so a notifying mention in a reply (`@**all**`, `@**Name**`, `@*group*`) is broken with a zero-width space after the `@`. Silent mentions (`@_**Name**`) pass through.

**Zulip content is stored in plaintext on this host, and your Zulip server has it too.** Messages land in `sessions.db`, the JSONL conversation log and consolidated memory as ordinary text; messages from senders who are not allowlisted are refused before they are persisted. To remove a conversation, use `corvidae redact` as described for Signal above, with `channel_id = 'zulip:<stream_id>'` or `'zulip:dm:<user ids>'`.

---

## `web` — browser chat UI

The `web` section is optional. If absent, `WebPlugin` loads but stays inert: no listener, nothing logged above `DEBUG`. A bare `web:` with no value turns it on with every default. If present, it must be well-formed: a bad field raises at startup naming it, and a port that's already taken fails startup too.

It serves a chat page on `http://127.0.0.1:8765/`. Each named conversation is its own channel, `web:<name>`, kept in `sessions.db` like any other and listed in the page's sidebar. Replies, thinking (collapsed), tool calls and progress show up as they happen; reloading the page, or opening it later, replays the last 200 rows of the conversation, including replies that arrived while no tab was open.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `web.host` | string | `127.0.0.1` | Address to bind. A wildcard (`0.0.0.0`, `::`) needs `allowed_origins`, since no origin a browser could use can be derived from it. Anything other than loopback logs a WARNING at startup. Bracketed IPv6 (`"[::1]"`) is fine. |
| `web.port` | integer | `8765` | Port to bind, 0 to 65535. `0` (an ephemeral port, for tests) needs `host` to be an IP literal. |
| `web.token` | string | `$CORVIDAE_WEB_TOKEN`, else generated | The login token. At least 16 characters from `A-Z a-z 0-9 . _ ~ -`, because you paste it into a URL by hand. Null or empty falls through to the environment variable, then to a generated one. There's no way to turn auth off. |
| `web.allowed_origins` | list of strings | `null` | Extra origins the page may be served from, such as `https://box.tailnet.ts.net` or `http://mybox.local:8765`. Each is `scheme://host[:port]`, nothing more. The `Host` header must match one of the allowed origins too. |

```yaml
web:
  host: 127.0.0.1
  port: 8765
  token: "paste-a-long-random-token-here"   # or set CORVIDAE_WEB_TOKEN
  allowed_origins:
    - https://box.tailnet.ts.net
```

**Logging in.** Open `http://127.0.0.1:8765/?token=<token>`. That sets a session cookie and drops the token from the address bar. The cookie lasts 400 days and survives restarts as long as the token doesn't change. Generate a token with:

```sh
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

If you don't set one, corvidae generates a token at every start and logs the whole login URL once at `WARNING`. That means every restart logs you out, and `logging.level: ERROR` hides the URL entirely: set a token if you run at that level. Changing `web.token` in `agent.yaml` logs out every browser on the next reload and closes their sockets. That's also the only way to revoke a session. `CORVIDAE_WEB_TOKEN` is read from the daemon's own environment, which can't change under it, so rotating that one takes a restart (which revokes just the same).

**The token is the keys to the machine.** Whoever has it, or the cookie, can make the agent run shell commands as this user. Two places it ends up that you might not think about:

- *Your log file*, when it's generated. The log is as sensitive as the token, so keep `logging.file` readable only by you.
- *Your browser history*, and the access log of any reverse proxy in front of corvidae, because the login URL carries it. corvidae's own access log is off for this reason.

There's no second channel for the token (no QR code, no file); the log line is it.

**Remote access.** The default bind is loopback only, and that's the recommended way to keep it. To reach it from elsewhere, pick one:

- *An SSH tunnel.* `ssh -L 8765:127.0.0.1:8765 host`, then open `http://127.0.0.1:8765/` locally. Nothing else to configure.
- *Tailscale.* Either `tailscale serve --bg 8765` (HTTPS on your tailnet name, with corvidae still on loopback) and `allowed_origins: [https://<machine>.<tailnet>.ts.net]`; or bind `host` to the machine's tailnet address and list `http://<that address>:8765`.
- *A TLS reverse proxy.* List its public origin in `allowed_origins`. The proxy must pass the original `Host` header through: nginx's default `proxy_set_header Host $proxy_host` breaks both the Host check and the cookie's `Secure` flag, so set `proxy_set_header Host $host`. It must also forward WebSocket upgrades on `/ws`.

Whichever you pick, the origin you browse to has to be in the allowed set. Don't bind a public address without TLS in front: the token and cookie would cross the network in the clear.

**Web content is stored in plaintext on this host**, like every other transport's: `sessions.db`, the JSONL log (`web_<name>.jsonl`, when enabled) and memory. To remove a conversation, use `corvidae redact` as described for Signal above, with `channel_id = 'web:<name>'`.

## `webhook` — webhook ingress

The `webhook` section is optional. If it is absent, null, empty or has no `endpoints`, `WebhookPlugin` loads but stays inert: no listener, nothing logged above `DEBUG`. If present, it must be well-formed: a bad field raises at startup naming it. A port that is already taken does not: the plugin logs an `ERROR` naming `host:port` and stays inert.

It listens for `POST /hook/<endpoint>` and hands each accepted JSON object to the plugins that subscribed to that endpoint (see [plugin-guide.md](plugin-guide.md#subscribing-to-webhooks)). The ingress writes nothing to a conversation itself; with no subscribing plugin loaded, every endpoint answers 404.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `webhook.host` | string | `127.0.0.1` | Address to bind. A null token is allowed only when this is a specific IP address; a wildcard (`0.0.0.0`, `::`) or a hostname needs a token on every endpoint. |
| `webhook.port` | integer | `8400` | Port to bind, 0 to 65535. `0` needs `host` to be an IP literal. |
| `webhook.max_body_bytes` | integer | `1048576` | Largest accepted body; a larger one gets 413 without being read in full. |
| `webhook.endpoints.<name>` | mapping or null | none | One entry per endpoint. The name matches `[a-z0-9][a-z0-9_-]{0,63}`. Null means every default. |
| `webhook.endpoints.<name>.auth` | string | `bearer` | `bearer` (`Authorization: Bearer <token>`) or `hmac-sha256` (`X-Hub-Signature-256: sha256=<hex>`, the HMAC-SHA256 of the raw body keyed by the token). |
| `webhook.endpoints.<name>.token` | string | `null` | The bearer token or HMAC secret; `${VAR}` expands from the environment. Non-empty ASCII. Required for `hmac-sha256` and on a non-specific host. `null` on a bearer endpoint means no auth. |

```yaml
webhook:
  host: 127.0.0.1
  port: 8400
  endpoints:
    github:
      auth: hmac-sha256
      token: ${GITHUB_WEBHOOK_SECRET}
    siri:
      token: ${SIRI_WEBHOOK_TOKEN}
```

**Responses.** `202 {"status": "accepted", "deliveries": n}` once the delivery is queued for every subscriber; it never waits for one. `200 {"status": "duplicate"}` for a repeated delivery id. `400` for a body that is not a JSON object, `401` for failed auth, `404` for an endpoint that is unconfigured or has no subscriber, `405` for any method but POST, `413` for a body over `max_body_bytes`. The 400, 401 and 404 bodies are JSON naming the endpoint and the fix.

**Duplicates.** The delivery id is the `X-GitHub-Delivery` header, else `Idempotency-Key`. The last 32 ids per endpoint are kept in memory; a request carrying neither header is never a duplicate. A restart forgets them.

**Slow subscribers.** Each subscriber has a queue of 100 deliveries. When it is full the oldest delivery is dropped and a `WARNING` names the endpoint, the subscriber and the running drop count. Delivery order per subscriber is preserved. Queued deliveries are lost on shutdown.

**GitHub setup.** Add a webhook with Payload URL `http://<host>:8400/hook/github`, Content type `application/json`, and a secret equal to `webhook.endpoints.github.token`, and use `auth: hmac-sha256`. GitHub signs the body but sends no timestamp. The listener is plain HTTP; see [threat-model.md](threat-model.md) finding 16 before exposing it.

---

## `acp` — ACP stdio transport

The `acp` section is optional and does not enable ACP by itself. ACP mode
starts only via `corvidae acp`, which sets an internal `_acp_mode` flag.
Install the SDK by adding `--extra acp` to your `uv sync` line.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| *(reserved)* | — | — | Future agent capability / logging hints. Omit the block until needed. |

Channel ids use `acp:<sessionId>` and are persistence keys: load/resume
reuse the same id so history is not forked. Per-session `cwd` from
`session/new` (and load/resume) is stored on the channel as transport
metadata (not merged into LLM request bodies) and indexed for
`session/list` in `acp_session_index` inside `daemon.session_db`.

Advertised ACP session capabilities: `loadSession`, `list`, `resume`,
`close`. `session/delete` is not advertised.

**bb:** register Corvidae under `customAcpAgents` with command
`uv run corvidae acp` (from a checkout with the `acp` extra). See
[plugin-guide.md](plugin-guide.md#acp-transport) for the smoke checklist.

---

## `channels` — per-channel overrides

Keys are `transport:scope` identifiers (e.g., `irc:#general`, `cli:local`), or transport-wide keys ending in `*` (e.g., `signal:*`) that cover every channel of a transport; see [Transport-wide entries](#transport-wide-entries). Each value is a mapping of overrides applied on top of `agent` defaults. Any key absent from the channel block uses the transport-wide entry's value if one matches, else the `agent`-level default.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `register_prompt` | string or list of strings | `agent.register_prompt`, else the transport's shipped default | This channel's register, appended after `agent.system_prompt`. Replaces the shipped default; `""` or `[]` means no register. On a conversational transport (IRC, Discord, Signal, Zulip) the `silence` plugin still appends its instruction to a custom register, or to `""`; see [Registers](#registers). Same format as `agent.system_prompt`. |
| `max_context_tokens` | integer | `agent.max_context_tokens` | Token budget for this channel's conversation history. A hard ceiling: a window still over it after compaction loses its oldest rows from memory (the log keeps them). Also the budget for this channel's catch-up consolidation views. |
| `max_turns` | integer | `agent.max_turns` | Maximum consecutive tool-calling turns for this channel. Must be an integer; a wrong type is rejected when the file is parsed (a `Config error` at startup, a logged reload error on reload), not at turn time. |
| `reasoning_history` | `drop` \| `exchange` \| `keep` | `agent.reasoning_history` | Controls `reasoning_content` retention for this channel. |
| `tools.disabled` | list of strings | `[]` | Tools this channel cannot use, on top of the global `tools.disabled`. The channel can only narrow; it can't re-enable a globally disabled tool, and `set_settings` can't change the list. `shell` and `task_pipeline` disable together. See [Per-channel tool sets](#per-channel-tool-sets). |

Any other key in an entry, exact or transport-wide, is rejected: at startup as `Config error: ... — check <path>`, on reload by refusing the whole reload (see [Error handling](#error-handling)). The message names `channels.<id>.<key>`, the rejected value and the accepted keys above. In particular `max_tokens` is not a channel key; it exists only as a `set_settings` runtime override. `system_prompt` is the one exception to this rule: it is no longer a channel key, because the identity is agent-wide, but a stale one is skipped with a WARNING pointing at `agent.system_prompt` and `channels.<id>.register_prompt`, and the rest of the entry still loads. An operator `agent.yaml` that carries a stray key will fail to start after upgrading until the key is removed.

Channels not listed here are created on-demand when a message arrives and use matching transport-wide entries, then `agent` defaults.

```yaml
channels:
  irc:#general:
    register_prompt: "This is #general; keep it brief."
    max_context_tokens: 8000
  cli:local:
    register_prompt:
      - prompts/cli-addendum.txt
    reasoning_history: keep
```

### Registers

Every channel's system prompt is the agent's identity (`agent.system_prompt`) followed by a blank line and the channel's register, if it has one. The register resolves like any channel setting: the exact entry, then transport-wide entries from most to least specific, then `agent.register_prompt`, then the default the channel's transport ships, then none. Each layer replaces the ones below it; nothing is merged. An explicit `""` or `[]` at any layer means no register, which is how to switch a shipped default off.

Signal, Discord, Zulip and IRC ship defaults (texting-length turns and only the formatting the transport shows; see [prompt-guide.md](prompt-guide.md#chat-register-fragments)). Web, CLI and ACP ship none. A `register_prompt` replaces the whole default. While the `silence` plugin is loaded, a conversational transport's system prompt then gets one more paragraph after the register, which lets the agent stay silent, by replying `NO_REPLY` or, with `silence.mode` `tool` or `both`, by calling `end_turn` ([cognition.md](cognition.md#silenceplugin-packagescorvidae-cognitioncorvidae_cognitionsilencepy) has the text). It is added whatever the register's source, so a custom register and `register_prompt: ""` still get it; to opt a channel out, disable the plugin (`plugins.disabled: [silence]`). A register file that can't be read logs a WARNING naming the channel and the path, and the conversation starts on the identity alone. Like the identity, the register is fixed when a conversation starts.

### Transport-wide entries

Some channels can't be listed ahead of time. Signal DMs are keyed on each contact's ACI, ACP sessions get a fresh id per session, and the bot answers in any Discord channel it's mentioned in. A key ending in `*` covers all of them: it matches every channel whose id starts with the text before the `*`, including channels that first appear after the config was loaded.

```yaml
channels:
  signal:*:                        # every Signal DM, current and future
    max_turns: 6
  discord:*:                       # Discord server channels and threads (and DMs, below)
    register_prompt: [prompts/my_server.md]   # replaces the shipped Discord register
    tools:
      disabled: [shell, web_fetch]
  discord:dm:*:                    # Discord DMs: more specific than discord:*
    register_prompt: []            # no register in DMs
  signal:8f2c1c9a-3e21-4b77-9c3f-1a2b3c4d5e6f:
    max_turns: 20                  # one contact keeps the Signal settings, with more turns
```

A transport-wide entry accepts the same keys as an exact one. Each setting resolves in this order, last wins: built-in default, `agent:`, matching transport-wide entries from least to most specific (so `discord:dm:*` beats `discord:*` whatever order they're written in), the channel's exact entry, then `set_settings` runtime overrides. A Signal phone-number alias counts as an exact entry. `tools.disabled` is the exception: it's the union of the global list and every matching entry, so a more specific entry can narrow further but can't re-enable a tool.

The accepted key forms are `transport:*` and `transport:<scope-prefix>*`. A `*` anywhere else in a key ending in `*`, or a key with no `transport:` before the `*`, is rejected at startup, and on reload, naming the key. Only a trailing `*` is special, so an exact key like `irc:#c*t` still works; the one casualty is an IRC channel whose name itself ends in `*`, which now reads as a transport-wide key. Matching is on channel ids, so `signal:+1*` would match only phone-number alias entries, never a conversation (Signal conversations are keyed on ACIs). A transport-wide key is never a channel itself. This is a prefix match, not a glob: `*` means "the rest of the id" and only at the end, and `?` has no meaning.

Transport-wide entries reload like `agent:` does: a running channel sees the new values at its next use. `max_turns`, `max_context_tokens` and `reasoning_history` apply from the next turn, and `tools.disabled` follows the rules under [Per-channel tool sets](#per-channel-tool-sets). A conversation's system prompt (identity and register) is fixed when it starts and survives compaction, so a register edit reaches only channels created after the reload, or everything after a restart. Removing a transport-wide entry also takes effect live. Each running channel whose matching entries change is logged at INFO.

### Per-channel tool sets

`channels.<id>.tools.disabled` gives one channel a narrower tool set than the rest, for example an IRC channel with strangers in it that runs without `shell` while your own channels keep it:

```yaml
tools:
  disabled: [write_file]          # everywhere
channels:
  irc:#lex:
    tools:
      disabled: [shell, web_fetch] # here too; task_pipeline goes with shell
```

The channel's set is the global list plus its own and every matching transport-wide entry's, resolved when its conversation starts, the same way its system prompt is. The tool schemas sit at the head of the prompt, so they stay fixed for the conversation; rewriting them every turn would rebuild the llama-server KV cache every turn. Forks of the channel send the same schemas.

Hiding a tool isn't access control. A model can still call a tool it saw earlier in the conversation, or one named in text someone pasted in. So dispatch enforces the list too: a call to a disabled tool doesn't run. The model gets back `Error: tool 'shell' is disabled on channel irc:#lex (channels.irc:#lex.tools.disabled); it was not run. ...`, and the refusal is logged at WARNING with the tool and channel. Both name every key that disables the tool, so a tool disabled by `discord:*` is reported as `channels.discord:*.tools.disabled`. A `subagent` launched from the channel gets the same narrowed set, so the tool isn't one hop away.

`tools` accepts only `disabled`. Anything else is rejected at startup, and on reload, naming the key, the rejected value and the accepted keys. A name that matches no loaded tool is logged at WARNING, naming the key it's under, when the conversation starts. That catches typos, though an MCP tool whose server hasn't connected yet reads the same way.

**On hot reload** the list is live for channels that are already running. In an exact entry it's the only live key; the others need a restart. A transport-wide entry's keys are all live (see [Transport-wide entries](#transport-wide-entries)).

- **Narrowing** (adding a tool to the list) is enforced at dispatch on the next call. The schemas keep listing the tool until the next compaction or conversation start, so the model may try it and be refused.
- **Widening** (removing a tool from the list) takes effect at the next compaction or conversation start. Until then the tool stays hidden and refused, so the head of the prompt never changes mid-conversation.
- A registry change is the exception to "schemas change only at compaction or conversation start": when a plugin rebuilds the registry (an MCP server reconnecting) the new tool set applies at the next turn, narrowed by the conversation's existing disabled set.
- A call that was already dispatched runs to completion.
- Removing a channel's whole exact entry from `channels:` keeps its last list until restart. To re-enable everything on a running channel, set `disabled: []`. Removing a transport-wide entry takes effect live.

Discord channels are keyed on numeric ids: `discord:<channel_id>` for a server channel, `discord:<thread_id>` for a thread (a thread has its own id and its own history, so it does not inherit its parent channel's override), and `discord:dm:<user_id>` for a DM. Names are not accepted, because they change and the id is the persistence key.

Zulip channels are keyed the same way: `zulip:<stream_id>` for a channel (every topic in it shares the entry, because a topic is a name people can rename), `zulip:dm:<user_id>` for a 1:1 DM, and `zulip:dm:<id>,<id>` for a group DM, with the ids ascending and the bot's own left out.

Signal channels accept either form of key. `signal:<aci>` is the real channel id and matches directly. `signal:+15551234567` is an alias: its overrides are copied onto the ACI-keyed channel the first time that person writes, once the number has resolved. The alias entry itself never becomes a live channel, and an alias naming someone absent from `signal.allow` is simply inert — an unauthorized sender never gets a turn for the overrides to apply to.

---

## `mcp` — MCP server connections

Configured by `McpClientPlugin`. The `mcp` section is optional; if absent, no MCP servers are connected. See the [Plugin Guide](plugin-guide.md#mcpclientplugin-corvidae-mcp_clientpy) for full details.

For `transport: stdio`, the required keys are `command` and optionally `args` and `env`. For `transport: streamable_http` (MCP's current HTTP transport: one endpoint, usually `/mcp`) and the older `transport: sse`, the required key is `url`, and `headers` is optional; `command`, `args`, and `env` are not used.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `mcp.servers.<name>.headers` | map of string to string | none | HTTP headers sent on every request to the server (`streamable_http` and `sse` only). A value may contain `${VAR}`, replaced at connect time from the daemon's environment; put secrets there, not in `agent.yaml`. If a referenced variable is unset, empty, or contains a newline or NUL (a trailing newline from a secrets file, say), that server is skipped with one WARNING naming the server, the header and the variable, and it is not retried, since the environment can't change without a restart. Only the braced `${VAR}` form is substituted. Header values are never logged. |

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
      timeout_seconds: 30      # optional; per tool call; default 30 (above tools.call_timeout, the call is cut at call_timeout)
      connect_timeout_seconds: 30  # optional; connect + initialize + list_tools; default 30

    # streamable HTTP transport example, with a bearer token from the environment
    wiki:
      transport: streamable_http
      url: http://localhost:8090/mcp   # required for streamable_http
      headers:                         # optional; sse takes it too
        Authorization: "Bearer ${OTTERWIKI_MCP_TOKEN}"   # set in the daemon's environment

    # sse transport example
    my_sse_server:
      transport: sse
      url: http://localhost:9000/sse   # required for sse transport
      tool_prefix: "remote"            # optional; defaults to server name
      timeout_seconds: 30              # optional; per tool call; default 30 (above tools.call_timeout, the call is cut at call_timeout)
```

`connect_timeout_seconds` bounds each server's startup handshake (transport
open, `initialize`, `list_tools`); it must be positive. A server that does not
finish in time is logged as unavailable (`MCP server connect timed out`), its
subprocess is terminated at once, and the daemon starts with the remaining
servers. MCP servers connect one after another during startup, so the worst
case adds the sum of the configured bounds to startup time. A server that
failed to connect at startup is retried in the background on the transports'
backoff ladder (10 s, doubling to 300 s, reset when a retry succeeds), with no
tool call needed, so a daemon whose servers were all down recovers without a
restart; each failed retry logs a WARNING naming the server, its endpoint, the error and the
next delay. An MCP tool call also retries pending servers as a fast path,
unless a retry is already running. A server skipped for a config error is logged
once, naming `mcp.servers.<name>.<key>` and the rejected value, and never
retried: fix the config and restart. The errors are: a server entry that is not
a mapping, an unknown `transport`, a missing or null (or non-string) `command`/`url`,
`args` that is not a list of strings, `env` that is not a mapping, `headers`
that is not a mapping, a `timeout_seconds` or `connect_timeout_seconds` that is
not a finite number (`.nan`, `.inf`, a string or a list), a non-positive
`connect_timeout_seconds`, and an unset `${VAR}` in `headers`. A malformed
block as a whole fails startup with a `Config error` naming the key and value:
`mcp` that is not a mapping, or `mcp.servers` that is neither null nor a mapping
(`mcp: null` and `servers: null` just mean no servers). The `mcp` section itself
is not hot-reloaded.

---

## Hot-reload

`ConfigWatcherPlugin` polls `agent.yaml` for mtime changes every `daemon.config_poll_interval` seconds (default 2.0). When a change is detected, the file is re-parsed, CLI overrides are re-applied, and `on_config_reload` is dispatched to all registered plugins.

No restart is required for the changes listed below. Changes that require restart are listed separately.

### What updates on reload

| Setting | Effect |
|---------|--------|
| `agent.compaction_threshold` | `CompactionPlugin` reads the new threshold on its next compaction check. |
| `agent.chars_per_token` | Deprecated. The value is reloaded into `Agent` and `CompactionPlugin` for config compatibility, but has no effect on token counting — `count_tokens()` uses the module-level constant. |
| `agent.request_logprobs` | `Agent` re-reads the flag; takes effect on the next LLM call. |
| `agent.mid_conversation_system_role` | `Agent` re-reads the role; takes effect on the next prompt build. The changed rendering of earlier turns costs one full prompt-cache miss. |
| `agent.compaction_fork_prompt` | `CompactionPlugin` uses the new prompt on its next compaction. |
| `agent.forks.*` | `Agent` re-reads the section; takes effect on the next fork. A reload with an invalid section is rejected whole. |
| `appraisal.stage2_fork_prompt`, `critique.fork_prompt.<lens>` | Read from the reloaded config on the next appraisal/critique fork. |
| `scheduler.*` | `SchedulerPlugin` re-reads its settings and re-syncs `jobs` by name (insert, update, disable removed), then wakes its timer so the change takes effect immediately. |
| Dotted plugin tunables (`appraisal.*`, `critique.*`) | Resolved per-decision via `corvidae.tuning.resolve_tunable`; a reloaded value applies on the next appraisal/critique decision. Per-channel `set_settings` overrides still take precedence. |
| `agent.immutable_settings` | `RuntimeSettingsPlugin` resets its blocklist and re-applies the new list. Constructor-supplied immutable keys are always retained. |
| New channel entries in `channels:` | `load_channel_config` registers newly added channels. |
| `channels.<id>.tools.disabled` on a registered channel | Updated in place. Dispatch refuses a newly disabled tool at once; the schemas follow at the next compaction or conversation start, never mid-conversation. See [Per-channel tool sets](#per-channel-tool-sets). A reload with a malformed entry changes no channel. |
| Transport-wide entries (`channels.<transport>:*`) | Replaced and re-matched against every running channel. Scalar keys apply from the next turn, `tools.disabled` as above, the register only to conversations that start after the reload. See [Transport-wide entries](#transport-wide-entries). |
| `web.token` | A changed token logs out every browser (their sockets close and the page says the session expired). A token removed from config falls back to `CORVIDAE_WEB_TOKEN` as the daemon started with it, else a fresh generated one, which is logged. Changing `CORVIDAE_WEB_TOKEN` itself needs a restart. |
| `agent` defaults | `ChannelRegistry.agent_defaults` is updated. `max_turns`, `max_context_tokens` and `reasoning_history` are resolved every turn, so running channels pick them up at their next turn; `system_prompt` and `register_prompt` reach only conversations that start after the reload. |
| `inotify_ping.*` | `InotifyPingPlugin` re-validates the section and re-arms its watches; `enabled: false` stops them. |

### What requires restart

A restart-only key that a reload changes is named in a warning, one line per
key: `config reload: <key> changed; restart-only, takes effect on restart`.
The comparison is against the config the running daemon is using, so editing a
key back to its running value draws no warning — the file and the process agree
and a restart would change nothing. A key that stays pending is reported once,
when its pending value first appears or changes, and not again on later reloads
that leave it alone; to see a still-pending key, read the reload that
introduced it. Reverting the key clears that record, so a later edit is reported
again.

| Setting | Reason |
|---------|--------|
| `irc.host`, `irc.port`, `irc.nick` | IRC transport is connected at startup; changing connection parameters requires reconnection. |
| `irc.allow` | The allowlist is parsed once in `on_init`; a reload keeps the old one. |
| `webhook.*` | `WebhookPlugin` reads its block in `on_init` and binds in `on_start`; it has no reload hook, so a change takes a restart. |
| `web.host`, `web.port`, `web.allowed_origins`, adding or removing the `web` block | The listener is bound at startup. A reload logs `restart to apply` and keeps serving with the old values. |
| Existing exact entries in `channels:` | Channels already registered retain their original `ChannelConfig` until restart, except `tools.disabled` (above). Transport-wide entries are live (above). |
| `llm.*` (every profile, including `llm.main`, plus `llm.defaults` and `llm.stages`) | Read at startup. A reload that changes any of it logs a restart-only warning per changed key and `llm config changed; LLM profiles, llm.stages and llm.defaults apply on restart` and keeps the running clients. A malformed `llm:` section still refuses the whole reload. A reload is not checked against the registered stages, so a `llm.stages` entry of the wrong kind passes the reload and fails the next startup. |
| `memory.*` (including `memory.consolidation_fork_prompt`) | `MemoryPlugin` reads its section at init only. That includes `memory.idle_consolidate_after`, `memory.consolidate_after_tokens` and `memory.retention.*`: the idle watch and retention timer are set when the plugin starts, and the token threshold is read once then. |
| `heartbeat.after` | `HeartbeatPlugin` reads it at init and registers its idle watch at start. Putting the heartbeat to sleep or waking it takes a restart. |
| `timepiece.*` | `TimepiecePlugin` reads its section in `on_init` and has no reload hook. |
| `daemon.shutdown_timeout`, `daemon.exit_timeout` | Read once at startup by the runtime. A reload that changes either logs the restart-only warning and does not validate the value; a bad value fails the next startup (and the restart tool's pre-flight). |
| `tools.max_result_chars`, `tools.read_file_window_fraction`, `tools.shell_timeout`, `tools.web_fetch_timeout`, `tools.call_timeout`, `tools.timeouts` | Deferred; not currently implemented in hot-reload path. A bad `tools.call_timeout`, `tools.timeouts` or `tools.read_file_window_fraction` value still refuses the whole reload (see Error handling). |

### Error handling

At startup (not reload), a missing or malformed config file, or a rejected
value in the `agent` or `channels` sections (e.g. a `channels.<id>.tools`
block that is not `{disabled: [names]}`, or an unknown key such as
`channels.<id>.max_turn`), in the `llm` section (profiles, `extends`, `llm.defaults`, `llm.stages`), or in `tools.call_timeout` / `tools.timeouts` / `tools.read_file_window_fraction`, prints
`Config error: <detail> — check <path>` to stderr and exits non-zero. The
detail names the key, the rejected value and what is accepted. The file is
read as UTF-8 whatever the locale: one that isn't valid UTF-8 says so, with
the byte offset of the first bad byte and "re-save it as UTF-8", and YAML
nested too deeply to load says to reduce the nesting.

- **Invalid YAML**: logged at `ERROR`; reload is skipped and the running config is unchanged.
- **Not valid UTF-8, or YAML nested too deeply**: logged at `ERROR` naming the file, the cause (with the byte offset for UTF-8) and the remedy; reload is skipped and the running config stays in force.
- **Rejected value in `agent:`, `daemon:` or `channels:`** (e.g. a malformed `channels.<id>.tools`): logged at `ERROR` naming the file. For `channels:` the message also names the key and the value and says "reload skipped"; for `agent:` and `daemon:` the key is in the logged traceback. The reload is skipped whole: no section is applied and no plugin receives `on_config_reload`, so a typo in one `channels:` entry cannot change `agent:` values.
- **Rejected `tools.call_timeout`, `tools.timeouts` or `tools.read_file_window_fraction`**: logged at `ERROR` naming the key and the value, with "reload skipped"; the running config is unchanged.
- **Malformed `llm:` section** (an unknown `extends` base, an `extends` cycle, an unknown `kind`, an embedding profile with no `dimensions`, no default chat profile, an `llm.stages` entry naming a missing profile, and so on): logged at `ERROR` naming the key, with "reload skipped"; the running config is unchanged.
- **Config file deleted**: logged at `WARNING`; polling continues. The file may reappear.
- **Per-plugin failure in `on_config_reload`**: logged at `ERROR`; other plugins still receive the reload.

---

## `logging` — Logging configuration

The `logging` section is optional. Omitting it applies defaults (INFO level; file output in CLI mode, stderr in programmatic use).

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `logging.level` | string | `INFO` | Minimum log level. One of `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. Case-insensitive. |
| `logging.file` | string | none (CLI: `corvidae.log`) | Path to a rotating log file (10 MB, 5 backups). When set, logs go to this file. When absent, logs go to stderr. In CLI mode, defaults to `corvidae.log`. |
| `logging.jsonl.max_bytes` | integer | `10485760` | Size cap, in bytes, for each JSONL file: every per-channel transcript under `daemon.jsonl_log_dir` and the metrics log `daemon.metrics_jsonl`. A write that would pass it first rotates the file to `<name>.1` (the previous `.1` becomes `.2`, and so on). `0` never rotates. Restart-only. |
| `logging.jsonl.backup_count` | integer | `5` | How many rotated files each JSONL sink keeps; the oldest is deleted. `0` keeps none, so the live file simply starts over. Restart-only. |

The JSONL files are a debugging aid; `message_log` in `sessions.db` is the durable record. Rotation is in-process, so no external `logrotate` is needed; a file already over the cap when the daemon starts rotates on its first write. Each file is capped separately, so the worst case on disk is the number of channels times `(1 + backup_count) * max_bytes`, plus the metrics log. If a rotation fails (a read-only directory, a full disk), the daemon logs one warning naming the file and keeps appending past the cap until rotation works again, rather than dropping records. A value that is not an integer of 0 or greater is a startup config error naming the key.

When `logging.file` is set in YAML, logs go to that rotating file regardless of mode. When `logging.file` is absent, logs go to stderr — except in CLI mode, where the default is `corvidae.log` to avoid polluting the chat interface.

---

## `cross_channel_send` — cross-channel send tools

Requires the cross-channel-send extra (`--extra cross-channel-send` on your `uv sync` line). Without it the `notify_other_channel` and `list_channels` tools do not exist.

`CrossChannelSendPlugin` lets any turn send an instruction to another channel context. The message is delivered as a system-role notification via `on_notify` (the same mechanism the scheduler uses for reminders); the target mints its own trace and gets a fresh turn budget. Loop prevention is a `hop_count` in the notification annotations, refused when it exceeds `max_hops`. A call whose target is the current channel is refused, with a warning in the log (a self-send would queue a duplicate turn).

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `cross_channel_send.max_hops` | integer | `3` | Maximum send hops before a `notify_other_channel` call is refused. A send from a context whose attribution carries `hop_count` n is refused when n + 1 exceeds this limit. Must be an integer ≥ 1 (bools and floats are rejected); a non-integer value fails startup naming the key and the value. A missing block keeps the default silently. |

---

## `webhook_notify` — webhook deliveries as notifications

Requires the webhook-notify extra (`--extra webhook-notify` on your `uv sync` line) and the `webhook` ingress. Deliveries on a listed endpoint are sent to every channel listed for it as notifications: the payload as compact JSON, fenced, with `origin=webhook:<endpoint>` in the header; an over-long payload is cut and spilled to a file (see `daemon.notification_spill_*`). Restart to apply a change.

**Throttling is on by default (#793).** A delivery on an endpoint with no open window opens one for `batch_seconds`, and everything that arrives in the window is sent as **one** notification: a single delivery as before, more than one as a compact JSON array in arrival order with a `deliveries=<n>` annotation. A burst of forty CI events therefore costs one turn per target channel, not forty. On top of that, at most `max_notifications` flushes go out per endpoint per `per_seconds`; a flush over the cap is held and its deliveries stay buffered up to `max_buffered`, beyond which the oldest are dropped. The next flush that gets through carries `dropped=<n>`, so a drop reaches the model and not just the log (a drop count always rides on a flush that carries deliveries; a drop-only notification exists only as a defensive path); every drop also logs a `WARNING` naming the endpoint and the running count. To restore the pre-#793 behaviour, set `batch_seconds: 0` and a high `max_notifications`.

```yaml
webhook_notify:
  max_payload_chars: 8000
  defaults:                 # every endpoint inherits these; these are the built-in values
    batch_seconds: 60
    max_notifications: 10
    per_seconds: 3600
    max_buffered: 500
  endpoints:
    github:                 # mapping form: channels plus overrides
      channels: [irc:#corvidae, discord:123456789]
      max_notifications: 20
    siri: [signal:+15551234567]   # bare-list form: these channels, the defaults
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `webhook_notify.max_payload_chars` | integer | `8000` | Longest compact-JSON payload passed inline, in characters. A longer one is cut, and the header says it was cut to the first N of M characters. Where a channel's tools can read spill files, the full text goes to a file the header names (see `daemon.notification_spill_*`). Must be an integer ≥ 1. |
| `webhook_notify.defaults.<key>` | see the four keys below | the values above | The throttle every endpoint inherits unless it overrides it. |
| `webhook_notify.defaults.batch_seconds` | number | `60` | Length of an endpoint's digest window, in seconds. `0` turns batching off and delivers on arrival. Must be ≥ 0. |
| `webhook_notify.defaults.max_notifications` | integer | `10` | Flushes allowed per endpoint per `per_seconds`. A flush over the cap is held, not dropped. Must be an integer ≥ 1. |
| `webhook_notify.defaults.per_seconds` | number | `3600` | The window the cap counts over, in seconds. Must be > 0. |
| `webhook_notify.defaults.max_buffered` | integer | `500` | Deliveries an endpoint may buffer, in an ordinary window as well as while its flush is held. Whenever the buffer exceeds it the oldest is dropped, counted and reported. Must be an integer ≥ 1. |
| `webhook_notify.endpoints.<endpoint>` | list of strings, or mapping | none | Either a non-empty list of channel ids (`transport:scope`) notified of the endpoint's digests, or a mapping with `channels:` plus any of the four throttle keys to override `defaults` for this endpoint. Repeated ids are notified once. All of an endpoint's channels receive the same digest, so the window and the cap are per endpoint; give a channel its own limits by routing it through its own endpoint. |

A malformed block fails startup naming the key, the value and the accepted range, including an unknown key under `defaults` or under an endpoint; an absent block leaves the plugin inert. A target on a transport that cannot build its channels yet (Discord or Signal before they start) is skipped with a `WARNING` naming the endpoint and how many deliveries the digest carried. Buffered deliveries are lost at shutdown, with a `WARNING` naming the endpoint and the count. For GitHub, filter events in the webhook's own settings: payloads such as `pull_request` and `check_run` are large and are cut at the cap, with the rest in a spill file.

---

## `context_panel` — live context window view

Requires the context-panel extra (`--extra context-panel` on your `uv sync` line) and the `web` block. Adds a "Context" section at the bottom of the chat page's sidebar, below the conversation list, with one tab per channel whose context window is loaded, on any transport. A tab shows the channel's readable name when the transport has one (the id otherwise), its message count and its token estimate against the budget. Selecting a tab shows that channel's window in the main pane in place of the chat, read-only, and scrolls to the end; selecting a conversation returns to the chat. The window appears as the model sees it: system prompt, tool schemas and stored rows with their kinds, tool names and local-time stamps, followed live for the latest turn. Only user and assistant messages (and the live progress text) start expanded, in different colors; everything else starts collapsed. Stamps come from the session database, so a row not yet persisted has none. The tab list loads with the page and again when you press the section's `refresh` button; a channel whose window loads later has no tab until then. The panel shows the whole window to anyone holding the console token; see the threat model's finding 9.

```yaml
context_panel: {}
```

The block has no keys. `context_panel: {}` or an empty `context_panel:` enables the panel; without the block it is off. Restart to apply a change. A block that is not a mapping fails startup naming `context_panel`.

---

## `restart` — Daemon restart plugin

Requires the restart extra (`--extra restart` on your `uv sync` line). Without it the `restart` tool does not exist.

`RestartPlugin` provides a `restart` tool that the agent can call to restart the daemon after plugin, prompt, or config changes. A loop guard prevents restart loops: no restart until a human has messaged since boot, a minimum uptime, and a maximum number of restarts per time window. A pre-flight check boots the daemon's startup as far as plugin `on_init` (logging, config sections, channels, plugin loading, hook and dependency checks) before the daemon is killed, and refuses the restart if that fails, with the kernel's `Config error:` line or a `pre-flight failed:` line naming the exception. It skips `on_start` and CLI overrides; see the threat model's "Pre-flight" section. After the restart, a notification is delivered to the originating channel (or the home channel as fallback) with the reason, downtime, and plugin list.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `restart.home_channel` | string | none | Channel id for boot notices and fallback notifications. When absent, notices are logged but not sent. |
| `restart.config_path` | string | `agent.yaml` | Path to the config file the pre-flight validates, relative to the config directory. |
| `restart.min_uptime_seconds` | integer | `120` | Minimum seconds since boot before a restart is allowed. Must be an integer ≥ 0; a non-integer value fails startup naming the key. |
| `restart.max_restarts` | integer | `3` | Maximum number of restarts allowed within `window_seconds`. Must be an integer ≥ 0; a non-integer value fails startup naming the key. |
| `restart.window_seconds` | integer | `21600` | Time window (seconds) for the `max_restarts` count. Must be an integer ≥ 0; a non-integer value fails startup naming the key. |
| `restart.preflight_timeout_seconds` | number | `120` | Timeout for the pre-flight subprocess. Must be a number > 0; a non-numeric value fails startup naming the key. |
| `restart.notify_delay_seconds` | number | `5` | Delay after the agent is ready before sending the post-restart notification. Must be a number ≥ 0; a non-numeric value fails startup naming the key. |

### State files

The plugin reads and writes files under `<config_dir>/state/`:

- `restart.json` — pending restart marker (written before SIGTERM, read and deleted on next boot).
- `restart-history.jsonl` — append-only log of restart requests and deliveries.
- `boot-history.jsonl` — written by an **external boot supervisor** (not by this plugin). The plugin reads it to determine the boot mode (normal vs. safe) and whether the previous run ended abnormally. Record format: one JSON object per line; start records have a `mode` key, exit records have a `service_result` key, reset records have a `reset` key. Nothing in this repository writes this file; the boot supervisor is deployment-specific.

---

## `timepiece` — wall-clock stamps in the window

Requires the timepiece extra (`--extra timepiece` on your `uv sync` line). Without the block the plugin runs with its defaults; to turn it off, list `timepiece` in `plugins.disabled`.

Before a turn, if the channel has had no stamp for `interval_minutes`, the plugin appends one CONTEXT line to the conversation, such as `The current date and time are now Monday 2026-11-09 at 17:12 PST`. Every channel that takes a turn is stamped, internal ones included. The line is persisted like any CONTEXT row. The last-stamp times are kept in memory, so the first turn on each channel after a restart is stamped. It adds no tool and starts no turn.

```yaml
timepiece:
  interval_minutes: 10
  timezone: America/Los_Angeles
  format: "%A %Y-%m-%d at %H:%M %Z"
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `timepiece.interval_minutes` | number | `10` | Minimum minutes between stamps on one channel. Must be a number > 0. |
| `timepiece.timezone` | string | daemon's local time | IANA timezone name for the stamp. Must name a known zone. |
| `timepiece.format` | string | `"%A %Y-%m-%d at %H:%M %Z"` | `strftime` format for the time in the stamp. Must be a non-empty string. |

A malformed key fails startup naming the key and the value. Restart to apply a change.

---

## `inotify_ping` — completion pings for background jobs

Requires the inotify extra (`--extra inotify` on your `uv sync` line). Linux only.

Watches files, typically the logs of jobs the agent started in the background, and sends a notification to one channel when a watched file has been written and then stays quiet for `debounce_seconds`. A burst of writes yields one ping, and the watch re-arms after each, so a long-lived log pings again after its next burst. The notification text is `[inotify] <path> settled — background work likely complete`, with `origin=inotify`. It does not include the file's contents. Off unless `enabled: true`.

```yaml
inotify_ping:
  enabled: true
  channel: internal:heartbeat
  debounce_seconds: 5
  paths:
    - /var/log/jobs/train.log
  terminal_marker: done
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `inotify_ping.enabled` | bool | `false` | Turns the watcher on. When false, only this key is validated, so the others can stay in place. |
| `inotify_ping.channel` | string | none | Channel id (`transport:scope`) that receives the pings. Required when enabled. Resolved when a ping fires, so a channel not yet loaded still works; a ping to a channel that can't be resolved is dropped with a `WARNING`. |
| `inotify_ping.debounce_seconds` | number | `5` | Quiet period, in seconds, after the last write before a file counts as settled. Must be a number ≥ 0. |
| `inotify_ping.paths` | list of strings | none | Files to watch. Required when enabled; must be a non-empty list of non-empty strings. A file that doesn't exist yet is watched through its parent directory, so it can appear later. A path whose parent doesn't exist either is skipped with a `WARNING`. |
| `inotify_ping.terminal_marker` | string or null | `null` | When set, a settled file pings only if its last non-empty line contains this text, so a pause in the middle of a job is not reported as completion. Must be a non-empty string or null. |

A malformed block fails startup naming the key and the value. The whole section reloads live: a reload re-validates it and re-arms the watches, and setting `enabled: false` stops them.

---

## `rss` — feed polling and notification delivery

`RSSPlugin` polls the configured feeds on a timer and delivers new items as notifications on the configured targets. Feed content is untrusted external data (see [threat-model.md](threat-model.md)).

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `rss.feeds` | list | — | Non-empty list of feed URLs (http/https). Required. |
| `rss.targets` | list | — | Non-empty list of `transport:scope` channels to deliver to. Required. |
| `rss.poll_interval` | string | `"30m"` | Duration string: `30s`, `20m`, `6h`, `1d`, `7d`, `1h30m`. |
| `rss.max_items_per_poll` | integer | `10` | Cap on items delivered per cycle. Items beyond the cap are retried on the next cycle. |
| `rss.max_payload_chars` | integer | `2000` | Max characters per item (title + link + summary). |
| `rss.state_file` | path | `state/rss/seen.json` | Where seen guids are tracked. |

```yaml
rss:
  feeds:
    - https://example.com/feed.xml
    - https://blog.example.com/atom.xml
  targets:
    - internal:heartbeat
  poll_interval: 30m
  max_items_per_poll: 10
  state_file: state/rss/seen.json
```

The plugin is optional; it is not installed by default. Add `rss-reader` to the `EXTRAS` build arg or install it from the worktree. Config changes require a restart.

**What lands in `/data`:** `sessions.db`, `provenance.key` (keep it with the directory), `state/`, and a log file if you set `logging.file`; otherwise logs go to `docker logs`. Back that directory up and you've backed up the bot. The daemon runs as uid 1000, not root, because the shell tool runs as the daemon's user ([threat-model.md](threat-model.md), finding 4), so the directory has to be writable by that uid. That's what the `chown` is for, and after it the files belong to uid 1000, so editing `agent.yaml` on the host needs `sudo`. If you'd rather not, skip the `chown` and run with `--user $(id -u):$(id -g)` on a directory you own. If the daemon can't write the directory, startup dies with `sqlite3.OperationalError: unable to open database file`, which doesn't name the directory.

**Locking the config (optional).** Add `-v ~/corvidae-data/agent.yaml:/data/agent.yaml:ro` to `docker run` and the agent can't edit its own configuration, which is the config half of [prd.md](prd.md) §3.11.11 (prompt files need the same treatment). A host edit that replaces the file (`sed -i`, most editors) detaches the read-only mount until the container restarts, so run `docker restart corvidae` after each edit.

**Extras are chosen at build time** with `EXTRAS`, space separated, using the names in `[project.optional-dependencies]` in `pyproject.toml` (all but `dev` are installed by default): `--build-arg EXTRAS="cognition discord"` for a subset, or `--build-arg EXTRAS=""` for the bare kernel. `corvidae acp` isn't a container use: your editor starts it against a local checkout. Signal needs signal-cli running outside the container with its socket mounted in; see [signal-ops.md](signal-ops.md).

**A container is not a sandbox for the tools.** It limits the daemon and its tools to what you mounted and what the network lets them reach. It does not separate the tools from the daemon: `shell` still runs unsandboxed, as the same user in the same container, next to `sessions.db` and the web token. Running tools somewhere narrower is a different feature, tracked in [#532](https://github.com/schuyler/corvidae/issues/532).

## Running in a container

The `Dockerfile` at the repo root builds corvidae with every extra (Discord, the cognition plugins and the rest) into an image that runs `corvidae serve` against an `agent.yaml` mounted at `/data`. The quickest useful setup is the browser chat page:

```sh
docker build -t corvidae .
mkdir -p ~/corvidae-data
```

Put this in `~/corvidae-data/agent.yaml`:

```yaml
llm:
  main:
    base_url: http://host.docker.internal:8080/v1   # your model server, on the Docker host
    model: llama3
agent:
  system_prompt: "You are a helpful assistant."
web:
  host: 0.0.0.0                  # inside the container; the -p below keeps it on loopback
  allowed_origins:
    - http://127.0.0.1:8765      # the address you browse to
```

```sh
sudo chown -R 1000:1000 ~/corvidae-data     # the daemon's uid; -R matters if the directory already holds data
TOKEN=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
docker run -d --name corvidae --restart unless-stopped \
  --add-host host.docker.internal:host-gateway \
  -p 127.0.0.1:8765:8765 \
  -e CORVIDAE_WEB_TOKEN=$TOKEN \
  -v ~/corvidae-data:/data \
  corvidae
echo "http://127.0.0.1:8765/?token=$TOKEN"
```

Open that URL. Why the config looks like this: inside a container the page has to listen on a wildcard address, and a wildcard bind derives no allowed origins of its own, so the address you browse to must be in `allowed_origins` or the page answers `403 bad host` (add `http://localhost:8765` if you use that name). The wildcard bind logs a `WARNING` at startup, as described in the [web section](#web--browser-chat-ui). The token comes from the environment so a restart doesn't log you out. It is shell access to the container and everything mounted into it, so keep the port on loopback and put TLS or a tunnel in front of it for anything remote.

**The model server.** On a Linux host, `host.docker.internal` is the Docker host's bridge address, so a model server listening only on `127.0.0.1` can't be reached through it, and every turn fails to connect. Either make the server listen on an address the container can reach (mind your firewall), or run the container with `--network host` in place of the `-p` and `--add-host` lines, so it shares the host's loopback. In that case use `http://127.0.0.1:8080/v1` as the `base_url` and write the `web:` block as just `web: {}`: the default bind is loopback, and the wildcard and `allowed_origins` above would expose the page on every host address.

**The batteries are on.** Because the image carries every extra, memory, appraisal, critique and goals run by default. Memory consolidation, appraisal and the critique forks make model calls of their own, which matters on a single-slot local server. Without an `llm.embedding` profile memory still works but stores records without vectors, which is what the startup warning about a missing embedding profile means ([cognition.md](cognition.md)). List plugins under `plugins.disabled`, as `agent.minimal.yaml.example` does, to turn any of them off. The `restart` tool ends the daemon with SIGTERM, which is why the run command has `--restart unless-stopped`. Two other defaults are worth knowing: `timepiece` stamps each turn in the container's clock, which is UTC unless you add `-e TZ=…` or set `timepiece.timezone`, and `silence` lets the agent answer `NO_REPLY` and send nothing.

**Discord.** The bot only connects out, so there is nothing to publish. Add a `discord:` block with your user id under `allow` (the default refuses everyone), and pass the bot token as `-e DISCORD_TOKEN=…` rather than writing it into `agent.yaml`. A new `discord:` block needs only `docker restart corvidae`, but `docker restart` keeps the old environment, so to add or change `DISCORD_TOKEN`, `docker rm -f corvidae` and run it again. Creating the application and inviting the bot are in [discord-ops.md](discord-ops.md). Two `voice will NOT be supported` warnings at startup are expected.

