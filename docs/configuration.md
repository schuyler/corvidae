# Corvidae Configuration Reference

Corvidae is configured via a single YAML file, `agent.yaml`, passed to the daemon at startup. The full parsed dict is available to every plugin via the `on_start(config: dict)` hook.

The key `_base_dir` (a `pathlib.Path`) is set by `Runtime.start()` (`runtime.py`) alongside the parsed config. It points to the directory containing `agent.yaml` and is used to resolve relative file paths (e.g., `system_prompt` file lists).

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

## `plugins` — enabling and disabling plugins

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `plugins.disabled` | list of strings | `[]` | Entry-point plugin names to block from loading. `Runtime.start()` calls `pm.set_blocked(name)` for each entry before entry points are loaded; blocked plugins are never instantiated. Names must match the entry-point name from `[project.entry-points.corvidae]` in `pyproject.toml` (e.g. `memory`, `funnel`, `appraisal`, `critique`, `outcome_log`, `memory_tools`, `subagent`, `mcp`, `cli`). Only entry-point plugins can be disabled this way — manually registered plugins cannot be blocked via config. |

`agent.minimal.yaml.example` disables `memory`, `memory_tools`, `funnel`, `appraisal`, `critique`, and `outcome_log` — the cognition plugin set — leaving the core agent loop, multi-turn tool calling, compaction, and persistence/restart recovery intact. See the [Plugin Guide](plugin-guide.md#plugin-disable) for the hook-level mechanics.

A misspelled entry is not an error — `pm.set_blocked` on a name that matches nothing simply blocks nothing, and the daemon starts normally with the plugin still loaded. To confirm a disable took effect, check the `"plugins loaded"` line the daemon logs at startup: it carries `loaded` (the plugins that actually registered) and `blocked` (this list, verbatim). A name present in `blocked` but still showing up in `loaded` is a typo.

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
| `llm.main.model_not_loaded_wait` | float | `60.0` | Total seconds to keep retrying when the server answers HTTP 400 "model is not loaded" (a llama-server router running `--no-models-autoload` with another model swapped in). Separate from `max_retries`; uses the same backoff, with at least 1 s between attempts. Must be a non-negative number, or startup fails. The turn waits for this long before replying, so set it a little longer than a model swap takes. `0` disables. When it runs out, the reply names the unloaded model instead of the generic error. |
| `llm.main.timeout` | float | `null` | Total HTTP timeout in seconds per LLM request. `null` uses aiohttp's 300-second default. |
| `llm.background.base_url` | string | — | Base URL for the background (subagent) LLM endpoint. Optional. |
| `llm.background.model` | string | — | Model identifier for subagent calls. Optional. |
| `llm.background.api_key` | string | `null` | API key for the background endpoint. |
| `llm.background.extra_body` | mapping | `null` | Provider-specific fields for background LLM requests. |
| `llm.background.max_retries` | integer | `3` | Maximum retry attempts for background LLM calls. |
| `llm.background.retry_base_delay` | float | `2.0` | Base delay in seconds for background LLM retry backoff. |
| `llm.background.retry_max_delay` | float | `60.0` | Maximum delay cap in seconds for background LLM retries. |
| `llm.background.model_not_loaded_wait` | float | `60.0` | As `llm.main.model_not_loaded_wait`, for background LLM calls. |
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
| `agent.compaction_prompt` | string | built-in | System prompt for the compaction summarizer. Used in every compaction round; when earlier summaries exist, an instruction to carry them forward and the (bounded) prior-summary text are appended after it. |
| `agent.chars_per_token` | float | `3.5` | **Deprecated.** Accepted for interface and config compatibility. Has no effect on token counting — `count_tokens()` uses the module-level `_FALLBACK_CHARS_PER_TOKEN` constant (3.5) regardless of this value. |
| `agent.immutable_settings` | list of strings | `[]` | Keys the agent is blocked from changing via the `set_settings` tool. `system_prompt` is always blocked regardless of this list, but see the note under "Runtime-tunable gate parameters" for what that does and does not protect. |
| `agent.request_logprobs` | boolean | `false` | When `true`, `Agent` merges `"logprobs": true` into every LLM request body. llama-server returns a logprobs envelope (surfaced to plugins via `on_agent_response`); Anthropic-style providers return nothing and the field surfaces as `None`. Static operator-only flag — captured at init and on config reload, not per-channel tunable. |
| `agent.mid_conversation_system_role` | `system` \| `user` | `system` | Role under which every `system` message *after the leading system prompt* is sent to the LLM: funnel CONTEXT and notifications without a `tool_call_id`. Strict chat templates (Qwen 3.8, Swift-1.5) raise "System message must be at the beginning" on those; `user` satisfies them and also keeps untrusted text out of the `system` role ([threat-model.md](threat-model.md), finding 3). Applied when the prompt is built: the in-memory window and `sessions.db` keep `role: system`, and the rewrite is deterministic, so the cached prompt prefix stays stable. Any other value logs a warning and uses `system`. Static operator-only flag, re-read on config reload. |

### Runtime-tunable gate parameters

The appraisal/critique parameters use dotted keys (e.g. `critique.sample_below_rate`, `appraisal.probe.budget_ms`) and are adjustable at runtime **without a daemon restart**, through two surfaces resolved per-decision by `corvidae.tuning.resolve_tunable` (operator directive 2, 2026-07-06):

1. **`set_settings` tool** (agent-facing, per-channel) — writes `channel.runtime_overrides`; highest precedence.
2. **Hot config reload** (operator, global) — edit `agent.yaml`; `ConfigWatcherPlugin` dispatches `on_config_reload` and plugins re-read the dotted path on the next decision.

When neither surface sets a key, the plugin's commented best-guess default applies. Resolution is first-hit-wins: per-channel override → config dotted path → default. Dotted keys never reach the LLM request body — the `extra_body` filter in `agent.py` excludes any key containing a dot.

The persona is allowed to tune its own gates by default — a deliberate operator decision favoring emergence from experience. The per-key safety valve is `agent.immutable_settings`.

**Always block**, on every deployment: `messages`, `model`, `tools`, `tool_choice`. These are not tunables; they are undotted keys the LLM request body honors, and they override the conversation itself (see the note below the key list).

**Recommended blocklist for multi-speaker channels with untrusted participants:**

| Key | Why block it |
|-----|--------------|
| `critique.provenance.enabled` | The correctness gate — a prompt-injected "stop second-guessing yourself" must not be able to disable the confabulation check. |
| `critique.sample_below_rate` | The standing experiment's integrity — an agent tuning its own sampling to zero blinds the critique false-negative bound. |
| `critique.prompt.provenance`, `critique.prompt.adversarial`, `critique.prompt.constrained`, `critique.prompt.predictive` | These are the critic's *system prompts*. An agent that can rewrite its own critic's instructions has no critic. |
| `appraisal.stage2_prompt` | Same shape: the stage-2 appraisal system prompt. |
| `critique.provenance.weak_score`, `critique.provenance.max_terms` | Setting `max_terms` to 0 disables the provenance gate's second tier outright; setting `weak_score` to 0 stops it firing whenever memory retrieved anything. Either guts the gate without touching `enabled`. |

These are recommendations, not hardcoded blocks: `set_settings` writes are channel-influenceable, which is exactly the persistence-of-influence shape the design polices elsewhere — but two-process discipline (agent tunes, operator audits via the outcome log) is the accepted trade-off, with the blocklist as the per-key valve.

The complete set of runtime-tunable keys (anything passed to `resolve_tunable`) as of this writing:

- `appraisal.novelty.no_probe_default`, `appraisal.prior.w_novelty`, `appraisal.prior.w_stakes`, `appraisal.prior.w_valence`, `appraisal.probe.budget_ms`, `appraisal.probe.max_tokens`, `appraisal.probe.rank_scale`, `appraisal.stage2_prompt`, `appraisal.weights.commitment`, `appraisal.weights.disagreement`, `appraisal.weights.imperative`, `appraisal.weights.novelty`, `appraisal.weights.question`
- `critique.lens.ambiguity`, `critique.lens.commitment`, `critique.lens.disagreement`, `critique.lens.neg_valence`, `critique.prompt.<lens>`, `critique.provenance.enabled`, `critique.provenance.max_terms`, `critique.provenance.weak_score`, `critique.sample_below_rate`

Any *dotted* key not on that list is operator-only: it is read once at init or reload, and while `set_settings` can write it into `runtime_overrides`, nothing reads the override. Undotted keys are a different matter: `set_settings` keys *without* a dot are forwarded verbatim into the LLM request body, where they override `model` and `messages` — see [threat-model.md](threat-model.md), finding 1. `immutable_settings` can block the known-dangerous keys (`messages`, `model`, `tools`, `tool_choice`) as a stopgap, but a blocklist cannot be complete; the real fix is an allowlist in code. If you don't need agent self-tuning, `tools.disabled: [set_settings]` removes the surface altogether.

#### `appraisal.*` — stage-1 appraisal

`AppraisalPlugin` scores every inbound message at the gate: surface heuristics plus an FTS5 familiarity probe on a dedicated read-only connection, under a hard latency budget, failing open. All keys resolve per-decision.

| Key | Default | Description |
|-----|---------|-------------|
| `appraisal.probe.budget_ms` | `50` | Hard latency budget for the FTS5 probe. Timeout ⇒ no probe result ⇒ the vector is built from surface heuristics alone (never blocks the gate). |
| `appraisal.probe.max_tokens` | `12` | Cap on sanitized tokens in the FTS5 MATCH query. |
| `appraisal.probe.rank_scale` | `10.0` | bm25 rank normalization: familiarity scales with `min(1, −rank/rank_scale)` (bm25 ranks are negative; more-negative = stronger match). |
| `appraisal.novelty.no_probe_default` | `0.5` | Novelty when no probe result arrived in budget — a value, never a null. |
| `appraisal.weights.novelty` | `0.35` | Salience weight for novelty (1 − familiarity). |
| `appraisal.weights.question` | `0.15` | Salience weight for question density. |
| `appraisal.weights.disagreement` | `0.20` | Salience weight for max(disagreement markers, negation density). |
| `appraisal.weights.commitment` | `0.20` | Salience weight for numbers/commitment density. |
| `appraisal.weights.imperative` | `0.10` | Salience weight for imperative markers (feeds salience only, not the vector). |

*(The `critique.*` tunable keys are not individually tabulated here; the complete list is under "Runtime-tunable gate parameters" above. No plugin resolves a `gate.*` key.)*

### `agent.context_compact` — removed

`ContextCompactPlugin` has been removed: superseded by `MemoryPlugin` (memory consolidation and retrieval), with per-turn token stats now in the `usage_log` table. Any `agent.context_compact.*` keys in existing configs are ignored.

---

## `memory` — autobiographical memory

`MemoryPlugin` consolidates compacted dialog into first-person memory records and retrieves them into the prompt (see [design.md](design.md)). The former `dream.*` keys are gone with `DreamPlugin` (absorbed by this plugin).

A record's `created_at` is the `sessions.db` timestamp of the last message in its source range, i.e. when the conversation happened, not when consolidation ran. When a near-duplicate merge folds an older record into a new one, the survivor keeps the later of the two dates. Everything that ages a record (the retrieval score, the displayed `(age)`, retention decay and `memory.retention.grace_days`, and the `search_memory` date filters) measures from that date, so history consolidated long after the fact is aged correctly.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `memory.idle_consolidate_after` | float | `1800` | Seconds of channel inactivity before the `on_idle` trigger consolidates that channel's un-consolidated tail. |
| `memory.half_life_days` | float | `30` | Recency decay constant of the retrieval score (`similarity × exp(-age_days / half_life)`). It is an e-folding time: the factor is `1/e`, not `1/2`, at `half_life_days`. |
| `memory.retrieval.k` | integer | `8` | Vector-KNN candidate count per retrieval. |
| `memory.retrieval.bands.strong` | float | `0.75` | Value at or above which a retrieved memory is banded `[strong]`. The value is the decayed score, or raw similarity under `band_on: similarity`. |
| `memory.retrieval.bands.moderate` | float | `0.60` | Value at or above which a retrieved memory is banded `[moderate]`; below is `[weak]` (dropped unless nothing else matched). Compared against the same quantity as `bands.strong`. |
| `memory.retrieval.band_on` | `score` \| `similarity` | `score` | What the bands are applied to. `score` bands the recency-decayed score, so with the default half-life nothing older than about 9 days can be `[strong]`. `similarity` bands raw relevance, so an old, apt memory reads `[strong] (180d)` rather than as a guess; candidates are still ranked by the decayed score. In FTS5-degraded retrieval every keyword hit carries the similarity proxy (0.6), so under `similarity` all keyword hits get the same band: whichever band 0.6 falls in under the configured thresholds (`[moderate]` with the defaults). `retrieval_log.top_score` (and `exchange_log.retrieval_top_score`) always record the decayed score, so under `similarity` they are not the quantity the bands compare against; calibrate the thresholds from similarity instead. Any other value logs a warning and uses `score`. |
| `memory.retrieval.recency_from` | `created` \| `last_used` | `created` | Where the ranking age is measured from. `last_used` uses `max(created_at, last_retrieved_at)`, the same rule retention uses, so a memory stays fresh while it keeps being recalled. The `(age)` shown to the model is always measured from `created_at`. Any other value logs a warning and uses `created`. |
| `memory.channel_groups` | mapping | `{}` | `{group_name: [channel ids]}` — channels in the same group share retrieval scope (passive retrieval, `search_memory`, and `recall_raw`) and near-duplicate merging (see `dup_threshold`). Memory is otherwise compartmentalized per channel. A member may be a glob using `*` (any run of characters, including none) and `?` (exactly one character), matched case-sensitively against the full `transport:scope` id, so `signal:*` covers Signal channels created at runtime. A group applies to a calling channel that equals one of its literal members or matches one of its patterns; the caller's own id is always in scope by equality. A member containing `*` or `?` is a pattern, even when you meant it literally (`irc:#what?` also matches `irc:#whats`), and there is no escape syntax. A pattern containing `[` or `]` fails startup with a `ValueError`; a member with no wildcard is matched exactly, so a bracketed IRC nick such as `irc:[bob]` is fine. A group containing `*` alone makes memory global across every channel. |
| `memory.consolidation_prompt` | string | built-in | Overrides the consolidation system prompt (documented copy: `prompts/memory_consolidation.md`). |
| `memory.dup_threshold` | float | `0.95` | Near-duplicate merge threshold. When a new consolidation embedding has cosine similarity ≥ this value against an existing record in the channel's memory scope, the older record is superseded: its statistics are folded into the new record and its vec row is deleted. A record on another channel qualifies only when both channels' scopes reach the same channels (always true within a single `*` group; see `channel_groups`), so no channel loses sight of a merged fact. Applied only when sqlite-vec is available and the new record's embedding succeeded; silently skipped otherwise. |

### `memory.retention` — Retention job

The retention job runs as a silent background task on daemon startup and after each `on_idle` firing (rate-limited to `memory.retention.interval`). It runs three passes: demotion (remove under-used records from the vector index), re-promotion (restore records whose access stats have risen since demotion), and backfill (embed re-promoted records that are missing vectors). The backfill pass is skipped when embedding is disabled by the `embedding_meta` mismatch guard.

The retention score formula is `importance × (1 + 0.5 × log1p(retrieval_count)) × exp(-age_days / half_life_days)`. All threshold and shape constants are the config keys documented below.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `memory.retention.grace_days` | float | `14` | Records younger than this many days are exempt from demotion regardless of score. Ensures newly consolidated records are not immediately demoted because they have not yet been retrieved. |
| `memory.retention.importance_floor` | float | `0.8` | Records with `importance ≥ importance_floor` are never demoted for lack of retrieval traffic. The importance prior ranges 0.0–1.0 (`AppraisalPlugin` wraps the default `RubricPrior` in `AppraisalPrior` when appraisal is enabled). |
| `memory.retention.demote_below` | float | `0.15` | Retention score threshold. An indexed record whose score falls below this value (and which passes the grace-period and importance-floor checks) is demoted: `indexed=0`, `embedded=0`, vec row deleted. The `memory` row and its FTS entry are preserved. Re-promotion requires the score to rise back to or above this threshold, which happens when the record is accessed via `recall_raw`. |
| `memory.retention.half_life_days` | float | `90` | Recency half-life for the retention score. Distinct from `memory.half_life_days` (retrieval scoring, default 30). The longer retention half-life gives passive retrieval time to re-warm a record before it demotes. |
| `memory.retention.interval` | float | `21600` | Minimum seconds between retention job runs (default 6 hours). The last-run timestamp is persisted in the `retention_meta` table so daemon restarts do not reset the interval. A daemon idle longer than `interval` since its last run will still run the job on next start (zero-traffic guarantee). |
| `memory.retention.backfill_batch` | integer | `32` | Maximum number of unembedded indexed records the retention job embeds per run (backfill pass). Bounds embedding API usage when many records need vectors, such as after a re-promotion wave or a period during which embedding was disabled. |

---

## `funnel` — context-admission funnel

`FunnelPlugin` is the single chokepoint for tail CONTEXT admission (dedupe, budgets, injection framing). It also owns deferred registration: non-tool-call notification payloads queue per `(channel, origin)` via `register_and_wake()`, one stub notification wakes the channel per pending pair, and the drain in `before_agent_turn` admits everything queued for the triggering exchange's origin. The deferred registry is in-memory by design (payloads pending at shutdown are dropped) and adds no config keys — drained payloads spend the same per-source budgets below.

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
| `tools.disabled` | list of strings | `[]` | Tool names dropped after collection, from any plugin (built-in or MCP). Read by `ToolCollectionPlugin`; this is the single chokepoint, so a disabled tool is absent from the schema the LLM sees and from subagents. Note `task_pipeline` is a separate tool that shells out on its own; disabling `shell` does not disable it. Not hot-reloaded. See [threat-model.md](threat-model.md). |
| `tools.disabled` | list of strings | `[]` | Tool names dropped after collection, in `ToolCollectionPlugin.rebuild_registry`. The single chokepoint every registered tool passes through — no per-plugin opt-outs. |

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

## `signal` — Signal transport

The `signal` section is optional. If absent, `SignalPlugin` loads but stays inert: no socket, no channels, nothing logged above `DEBUG`. If present, it must be well-formed — a missing `socket` or `account`, or an `allow` that is not a list, raises at startup naming the offending field.

corvidae talks JSON-RPC to a `signal-cli daemon` that is already running; it does not start or supervise one. Registering the bot's number and running the daemon are covered in [signal-ops.md](signal-ops.md).

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `signal.socket` | string | *required* | Path to the `signal-cli daemon --socket` unix socket. |
| `signal.account` | string | *required* | The bot's own number, E.164, e.g. `"+15550001111"`. Quote it — bare `+…` is not valid YAML. Used to recognize and ignore the account's own traffic. |
| `signal.allow` | list of strings | `[]` | Senders permitted to drive the agent. Each entry is either an E.164 number or an ACI (the account UUID). Default-deny: an empty or absent list refuses everyone, silently — no reply, no typing indicator, no read receipt. |
| `signal.message_chunk_size` | integer | `2000` | Maximum UTF-8 bytes per outbound Signal message. Longer replies are split paragraphs → sentences → words and sent as several messages; nothing is truncated. |

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

Pass the ids explicitly rather than using the `range` form: `message_log` ids are global across all channels, so an id range spans whatever other channels were active in the same window. The redaction cascades to intersecting memory records and to both FTS surfaces; see "Operator redaction" in [design.md](design.md). The jsonl logs are separate files and are not touched — delete them yourself.

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

Signal channels accept either form of key. `signal:<aci>` is the real channel id and matches directly. `signal:+15551234567` is an alias: its overrides are copied onto the ACI-keyed channel the first time that person writes, once the number has resolved. The alias entry itself never becomes a live channel, and an alias naming someone absent from `signal.allow` is simply inert — an unauthorized sender never gets a turn for the overrides to apply to.

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
| `agent.request_logprobs` | `Agent` re-reads the flag; takes effect on the next LLM call. |
| `agent.mid_conversation_system_role` | `Agent` re-reads the role; takes effect on the next prompt build. The changed rendering of earlier turns costs one full prompt-cache miss. |
| `agent.compaction_prompt` | `CompactionPlugin` uses the new prompt on its next compaction. |
| Dotted plugin tunables (`appraisal.*`, `critique.*`) | Resolved per-decision via `corvidae.tuning.resolve_tunable`; a reloaded value applies on the next appraisal/critique decision. Per-channel `set_settings` overrides still take precedence. |
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
