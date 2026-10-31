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
    system_prompt: "You are the channel bot."
    max_context_tokens: 16000
    max_turns: 10
    reasoning_history: drop
```

---

## `plugins` — enabling and disabling plugins

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `plugins.disabled` | list of strings | `[]` | Entry-point plugin names to block from loading. `Runtime.start()` calls `pm.set_blocked(name)` for each entry before entry points are loaded; blocked plugins are never instantiated. Names must match the entry-point name from `[project.entry-points.corvidae]` in `pyproject.toml` (e.g. `memory`, `funnel`, `appraisal`, `critique`, `outcome_log`, `memory_tools`, `goals`, `subagent`, `mcp`, `cli`). Only entry-point plugins can be disabled this way — manually registered plugins cannot be blocked via config. A null `plugins:` or `disabled:` means nothing is disabled; any other non-list value (a bare string, say) or a non-string entry fails startup with a `Config error` naming the key and value. |

`agent.minimal.yaml.example` disables `memory`, `memory_tools`, `funnel`, `appraisal`, `critique`, `outcome_log`, `gates`, `heartbeat` and `goals` — the cognition plugin set — leaving the core agent loop, multi-turn tool calling, compaction, and persistence/restart recovery intact. See the [Plugin Guide](plugin-guide.md#plugin-disable) for the hook-level mechanics.

A misspelled entry is not an error — `pm.set_blocked` on a name that matches nothing simply blocks nothing, and the daemon starts normally with the plugin still loaded. To confirm a disable took effect, check the `"plugins loaded"` line the daemon logs at startup: it carries `loaded` (the plugins that actually registered) and `blocked` (this list, verbatim). A name present in `blocked` but still showing up in `loaded` is a typo.

---

## `llm` — LLM backend

The `llm.main` block is required. Every other key under `llm:` is an optional **role** (`background`, `embedding`, and future roles) built into its own client; `LLMPlugin.get_client(role)` falls back to `llm.main` for unconfigured or unknown roles. Only `llm.main` is hot-swapped on config reload — other roles are restart-only.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `llm.main.base_url` | string | — | Base URL for an OpenAI-compatible Chat Completions endpoint. **Required.** |
| `llm.main.model` | string | — | Model identifier sent in each request. **Required.** |
| `llm.main.api_key` | string | `null` | API key for authentication, sent as `Authorization: Bearer <key>`. Omit if the backend does not require one. Either a literal key or a quoted `"${VAR}"` reference to an environment variable (see below). Give the bare key: a value such as `Bearer ${X}` would send "Bearer Bearer ...". |
| `llm.main.extra_body` | mapping | `null` | Provider-specific fields merged into the request body (e.g., `id_slot` for llama.cpp). |
| `llm.main.max_retries` | integer | `3` | Maximum retry attempts on transient HTTP errors (429, 500, 502, 503, 504) and connection failures. Set to 0 to disable retries. |
| `llm.main.retry_base_delay` | float | `2.0` | Base delay in seconds for exponential backoff between retry attempts. |
| `llm.main.retry_max_delay` | float | `60.0` | Maximum delay cap in seconds for exponential backoff. |
| `llm.main.model_not_loaded_wait` | float | `60.0` | Total seconds to keep retrying when the server answers HTTP 400 "model is not loaded" (a llama-server router running `--no-models-autoload` with another model swapped in). Separate from `max_retries`; uses the same backoff, with at least 1 s between attempts (the last sleep is cut short to fit the remaining wait). Must be a finite, non-negative number, or startup fails. The turn waits for this long before replying, so set it a little longer than a model swap takes. `0` disables. When it runs out, the reply names the unloaded model instead of the generic error. |
| `llm.main.timeout` | float | `null` | Total HTTP timeout in seconds per LLM request. `null` uses aiohttp's 300-second default. |
| `llm.background.base_url` | string | — | Base URL for the background (subagent) LLM endpoint. Optional. |
| `llm.background.model` | string | — | Model identifier for subagent calls. Optional. |
| `llm.background.api_key` | string | `null` | API key for the background endpoint. Same forms as `llm.main.api_key`, including `"${VAR}"`. |
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
| `llm.embedding.*` | — | — | The remaining keys (`api_key`, which also takes `"${VAR}"`, `max_retries`, ...) match `llm.main`. `document_prefix` and `query_prefix` are embedding-only — they do not exist on `llm.main` or `llm.background`. |

**API key from the environment.** Quote the reference so YAML keeps it as text: `api_key: "${OPENAI_API_KEY}"`. The variable is read from the environment the daemon runs in, so a tracked `agent.yaml` holds no secret. This works for every role, `main` included, and uses the same `${VAR}` expansion as MCP `headers`. A literal key still works; a value with no `${NAME}` (including one containing a bare `$`) is used as is. If the variable is unset, empty or contains a newline or NUL (often a trailing newline), startup fails with an error naming `llm.<role>.api_key` and the variable; set it in the environment corvidae runs in and restart. The key's value is never logged or put in an error or a reply. A config reload re-reads the same variable; to pick up a changed environment, restart. A reload that newly references an unset variable fails, is logged, and keeps the old `llm.main` client.

**Remote roles.** Every role is an OpenAI-compatible client, so any of them can point at a hosted endpoint. Whatever that role is sent then leaves the host: `main` serves turns and compaction summaries (and the forked appraisal and critique stages, which need the local window and so cannot move); `background` serves memory consolidation, the importance prior, subagents and, when not forked, standalone appraisal and critique; `embedding` serves memory embedding, retrieval and retention. An unconfigured role falls back to `main`. Conversation content is in all of these inputs. See `threat-model.md` finding 12. Hosted endpoints give no latency guarantee, so set `llm.<role>.timeout` generously.

`llm.background` is parsed by `LLMPlugin`. If the key is absent, `LLMPlugin.get_client("background")` returns the main client, so all callers (including `SubagentPlugin`) fall back to `llm.main`. When `llm.embedding` is absent, `MemoryPlugin` stores records with `embedded=0` and retrieval degrades to FTS5 keyword search.

Retry behavior applies to transient status codes (429, 500, 502, 503, 504) and connection errors. The `Retry-After` header (429 responses) is honored when present; otherwise exponential backoff is used. Retries do not apply to client errors (4xx) other than 429.

---

## `daemon` — daemon-level settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `daemon.session_db` | string | `sessions.db` | Path to the SQLite database file for conversation history. Relative paths are resolved from the working directory. |
| `daemon.max_task_workers` | integer | `4` | Number of concurrent background task workers in `TaskQueue`. |
| `daemon.completed_task_buffer` | integer | `100` | Number of completed task records retained in memory for `task_status` output. |
| `daemon.sqlite_journal_mode` | string | `wal` | SQLite journal mode for the session database. Allowed values: `delete`, `truncate`, `persist`, `memory`, `wal`, `off`. Read by `PersistencePlugin`. |
| `daemon.jsonl_log_dir` | string | — | Directory for per-channel JSONL conversation logs. When set, `JsonlLogPlugin` appends one JSON line per conversation event to `<dir>/<channel_id>.jsonl`. Relative paths are resolved against `_base_dir`. Omit to disable JSONL logging. |
| `daemon.metrics_jsonl` | string | — | Path of the metrics event log. When set, `MetricsJsonlPlugin` appends one JSON line (`{"ts", "name", "value", "tags"}`) per `on_metrics` event. Relative paths are resolved against `_base_dir`. Omit to disable the metrics JSONL sink. |
| `daemon.config_poll_interval` | float | `2.0` | Seconds between `agent.yaml` mtime checks by `ConfigWatcherPlugin`. |

`daemon.idle_cooldown_seconds` and `daemon.idle_poll_interval` are removed with the `on_idle` hook. Either one in the config is an error at startup, and a reload carrying one is rejected. Idle work now runs on the kernel clock and each plugin sets its own period (for example `memory.idle_consolidate_after`); delete the key.

---

## `agent` — agent-level defaults

These values apply to all channels. Per-channel overrides in the `channels` section take precedence where set.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `agent.system_prompt` | string or list of strings | `"You are a helpful assistant."` | System prompt prepended to each LLM call. A list of strings is treated as file paths; each file is read and concatenated with double newlines. Relative paths are resolved against `_base_dir`. |
| `agent.max_context_tokens` | integer | `24000` | Token budget for conversation history. `CompactionPlugin` compacts when the estimated token count exceeds `compaction_threshold` × this value. |
| `agent.max_turns` | integer | `10` | Maximum consecutive tool-calling turns per exchange before tool dispatch is suppressed. The counter resets when an item starts a new exchange with no tool cycle in flight: a user message, or a notification that originates its own trace (a scheduler reminder or heartbeat). |
| `agent.reasoning_history` | `drop` \| `exchange` \| `keep` | `drop` | `drop` strips `reasoning_content` from every in-memory assistant message. `exchange` strips it only from completed exchanges (assistant messages before the last `role: user` message), so a tool chain in progress keeps its reasoning — the trained distribution for Qwen-family models. `keep` never strips it. Applied to the whole window before each LLM call, so history reloaded after a restart follows it too. The DB record always keeps reasoning. Any other value, or the removed `keep_thinking_in_history` key, fails at startup. Use `exchange` with the server's `preserve_thinking` off (see below). |
| `agent.compaction_threshold` | float | `0.8` | Compaction triggers when `token_estimate / max_context_tokens` exceeds this value. |
| `agent.compaction_retention` | float | `0.5` | After compaction, retain messages that fit within `compaction_retention × max_context_tokens` tokens, counting from the most recent. |
| `agent.min_messages_to_compact` | integer | `5` | Skip compaction if the conversation has this many messages or fewer. |
| `agent.compaction_prompt` | string | built-in | System prompt for the compaction summarizer. Used in every compaction round; when earlier summaries exist, an instruction to carry them forward and the (bounded) prior-summary text are appended after it. |
| `agent.compaction_fork_prompt` | string | built-in | Instruction a compaction *fork* appends to the dropped prefix, used instead of `compaction_prompt` when forks are on for `compaction` (see [`agent.forks`](#agentforks--background-processes-as-forks)). Documented copy: `prompts/compaction_fork.md`. |
| `agent.forks` | mapping | off | Run compaction, consolidation, appraisal and critique as forks of the main window. See [`agent.forks`](#agentforks--background-processes-as-forks). |
| `agent.chars_per_token` | float | `3.5` | **Deprecated.** Accepted for interface and config compatibility. Has no effect on token counting — `count_tokens()` uses the module-level `_FALLBACK_CHARS_PER_TOKEN` constant (3.5) regardless of this value. |
| `agent.immutable_settings` | list of strings | `[]` | Keys the agent is blocked from changing via the `set_settings` tool, on top of its built-in allowlist. `set_settings` only accepts the inference parameters `temperature`, `top_p`, `top_k`, `frequency_penalty`, `presence_penalty`, `max_tokens` and `seed`, the framework keys `max_turns`, `max_context_tokens` and `reasoning_history`, and dotted plugin tunables, so this list is for narrowing that set (e.g. the gate keys under "Runtime-tunable gate parameters"). `system_prompt` is always blocked. |
| `agent.request_logprobs` | boolean | `false` | When `true`, `Agent` merges `"logprobs": true` into every LLM request body. llama-server returns a logprobs envelope (surfaced to plugins via `on_agent_response`); Anthropic-style providers return nothing and the field surfaces as `None`. Static operator-only flag — captured at init and on config reload, not per-channel tunable. |
| `agent.mid_conversation_system_role` | `system` \| `user` | `system` | Role under which every `system` message *after the leading system prompt* is sent to the LLM: funnel CONTEXT, notifications without a `tool_call_id`, and tool results whose call compaction dropped (#79; stored as `role: tool`, framed as a system notification at render). Strict chat templates (Qwen 3.8, Swift-1.5) raise "System message must be at the beginning" on those; `user` satisfies them and also keeps untrusted text out of the `system` role ([threat-model.md](threat-model.md), finding 3). Applied when the prompt is built: the in-memory window and `sessions.db` keep `role: system`, and the rewrite is deterministic, so the cached prompt prefix stays stable. Any other value logs a warning and uses `system`. Static operator-only flag, re-read on config reload. Scheduled firings (`scheduler` reminders and jobs) are notifications without a `tool_call_id`, so they follow this key too, as do framed orphaned tool results; a strict template needs `user` for them to render at all. |

### `agent.forks` — background processes as forks

Off by default (GitHub issue #51; the evals that gate turning it on are
#73). With forks on, a background stage doesn't build its own prompt. It
renders the channel's window exactly as the main turn does, appends one
`user` instruction, and makes one call on `llm.main`. On a single-slot
llama-server that reuses the cached prefix instead of evicting it, and the
agent's own system prompt is in view, so records and summaries are written
in its voice. Any fork failure falls back to the stage's standalone call,
as does a stage whose rows have left the window. The provenance check is
the exception: it runs only as a fork, and without one it skips with a
warning ([cognition.md](cognition.md#critiqueplugin-the-provenance-check-packagescorvidae-cognitioncorvidae_cognitioncritiquepy)). See [design.md](design.md#forks).

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `agent.forks.enabled` | boolean | `false` | Master switch. `false` leaves every stage on its standalone call; the provenance check then skips, with a warning, on every exchange that has past claims. |
| `agent.forks.reasoning_budget_tokens` | integer or null | `2048` | Sent as `reasoning_budget_tokens` on every fork. A fork keeps the main reasoning effort, because the chat template renders it into the system block and changing it would break the cached prefix; this per-request budget bounds the fork's think instead. `null` sends nothing (server default). llama-server applies it only when the template declares thinking tags. |
| `agent.forks.stages` | list of strings or null | `null` | Which stages fork: `compaction`, `consolidation`, `appraisal`, `critique`. `null` means all of them. A name no plugin uses matches nothing. `consolidation`, `appraisal` and `critique` belong to the cognition extra; without it only `compaction` exists. |

Any other key under `agent.forks`, a non-boolean `enabled`, a negative or
non-integer budget, or `stages` that isn't a list of strings fails at
startup, and a reload carrying one is rejected.

Each stage has a fork instruction, next to its standalone prompt where it has one:

| Key | Default copy | Notes |
|-----|--------------|-------|
| `agent.compaction_fork_prompt` | `prompts/compaction_fork.md` | Appended to the dropped prefix. |
| `memory.consolidation_fork_prompt` | `prompts/memory_consolidation_fork.md` | Followed by a sentence naming where the range starts. Restart-only, like the rest of `memory.*`. The output schema is fixed in code: an override must still ask for the record shape (`summary`, `topic_tags`, `participants`). |
| `appraisal.stage2_fork_prompt` | `prompts/appraisal_fork.md` | Followed by quotes of the exchange's opening message and of the response sent, and the retrieval profile. |
| `critique.fork_prompt.<lens>` | `prompts/critique_<lens>_fork.md` | Followed by quotes of the exchange's opening message and of the response sent; provenance also gets each claim's search hits and the CONTEXT snapshot. Provenance has no standalone prompt, and an override must ask for its per-claim verdicts format or every run skips. |

These are static config, read with `corvidae.tuning.config_value`. Unlike
`appraisal.stage2_prompt` and `critique.prompt.<lens>`, they aren't runtime
tunables: `set_settings` can write the key, but nothing reads the override.

Forks always run on `llm.main`, because only the main model shares the
window's cache. If `llm.background` names a different model, enabling
forks moves those stages' cost onto the main model, and the daemon logs a
warning at startup saying so.

Each forkable stage tags its own LLM call (the summary, the record, the
stage-2 appraisal, each critique lens) `mode = 'fork'` or `'standalone'`
in `usage_log`, so `cached_tokens` per fork row shows whether the prefix
was reused. Other calls the stage makes, such as consolidation's
importance prior, leave `mode` NULL.

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

`MemoryPlugin` consolidates compacted dialog into first-person memory records and retrieves them into the prompt (see [cognition.md](cognition.md#memoryplugin)). The former `dream.*` keys are gone with `DreamPlugin` (absorbed by this plugin).

A record's `created_at` is the `sessions.db` timestamp of the last message in its source range, i.e. when the conversation happened, not when consolidation ran. When a near-duplicate merge folds an older record into a new one, the survivor keeps the later of the two dates. Everything that ages a record (the retrieval score, the displayed `(age)`, retention decay and `memory.retention.grace_days`, and the `search_memory` date filters) measures from that date, so history consolidated long after the fact is aged correctly.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `memory.idle_consolidate_after` | float | `1800` | Seconds a channel must be quiet before idle consolidation runs on its un-consolidated tail, and the period between repeat runs while it stays quiet. A channel that never goes quiet is consolidated through compaction instead. Takes effect at restart. |
| `memory.half_life_days` | float | `30` | Recency decay constant of the retrieval score (`similarity × exp(-age_days / half_life)`). It is an e-folding time: the factor is `1/e`, not `1/2`, at `half_life_days`. |
| `memory.retrieval.k` | integer | `8` | Vector-KNN candidate count per retrieval. |
| `memory.retrieval.bands.strong` | float | `0.75` | Value at or above which a retrieved memory is banded `[strong]`. The value is the decayed score, or raw similarity under `band_on: similarity`. |
| `memory.retrieval.bands.moderate` | float | `0.60` | Value at or above which a retrieved memory is banded `[moderate]`; below is `[weak]` (dropped unless nothing else matched). Compared against the same quantity as `bands.strong`. |
| `memory.retrieval.band_on` | `score` \| `similarity` | `score` | What the bands are applied to. `score` bands the recency-decayed score, so with the default half-life nothing older than about 9 days can be `[strong]`. `similarity` bands raw relevance, so an old, apt memory reads `[strong] (180d)` rather than as a guess; candidates are still ranked by the decayed score. In FTS5-degraded retrieval every keyword hit carries the similarity proxy (0.6), so under `similarity` all keyword hits get the same band: whichever band 0.6 falls in under the configured thresholds (`[moderate]` with the defaults). `retrieval_log.top_score` (and `exchange_log.retrieval_top_score`) record the top-ranked candidate's decayed score; `retrieval_log.top_similarity` (and `exchange_log.retrieval_top_similarity`) record the same candidate's raw similarity. Calibrate the thresholds from whichever one `band_on` names, excluding `degraded_to_fts = 1` rows, whose similarity is the 0.6 proxy rather than a measurement (`exchange_log` has no such flag; join `retrieval_log` on `trace_id` to filter it). Rows written before these columns existed read NULL. Any other value logs a warning and uses `score`. |
| `memory.retrieval.recency_from` | `created` \| `last_used` | `created` | Where the ranking age is measured from. `last_used` uses `max(created_at, last_retrieved_at)`, the same rule retention uses, so a memory stays fresh while it keeps being recalled. The `(age)` shown to the model is always measured from `created_at`. Any other value logs a warning and uses `created`. |
| `memory.channel_groups` | mapping | `{}` | `{group_name: [channel ids]}` — channels in the same group share retrieval scope (passive retrieval, `search_memory`, and `recall_raw`) and near-duplicate merging (see `dup_threshold`). Memory is otherwise compartmentalized per channel. A member may be a glob using `*` (any run of characters, including none) and `?` (exactly one character), matched case-sensitively against the full `transport:scope` id, so `signal:*` covers Signal channels created at runtime. A group applies to a calling channel that equals one of its literal members or matches one of its patterns; the caller's own id is always in scope by equality. A member containing `*` or `?` is a pattern, even when you meant it literally (`irc:#what?` also matches `irc:#whats`), and there is no escape syntax. A pattern containing `[` or `]` fails startup with a `ValueError`; a member with no wildcard is matched exactly, so a bracketed IRC nick such as `irc:[bob]` is fine. A group containing `*` alone makes memory global across every channel. |
| `memory.consolidation_prompt` | string | built-in | Overrides the consolidation system prompt (documented copy: `prompts/memory_consolidation.md`). The output schema is fixed in code: an override must still ask for the record shape (`summary`, `topic_tags`, `participants`). |
| `memory.consolidation_fork_prompt` | string | built-in | Overrides the instruction a consolidation *fork* appends to the window when `agent.forks` is on for `consolidation` (documented copy: `prompts/memory_consolidation_fork.md`). Static config, not a runtime tunable. The output schema is fixed in code: an override must still ask for the record shape (`summary`, `topic_tags`, `participants`). |
| `memory.dup_threshold` | float | `0.95` | Near-duplicate merge threshold. When a new consolidation embedding has cosine similarity ≥ this value against an existing record in the channel's memory scope, the older record is superseded: its statistics are folded into the new record and its vec row is deleted. A record on another channel qualifies only when both channels' scopes reach the same channels (always true within a single `*` group; see `channel_groups`), so no channel loses sight of a merged fact. Applied only when sqlite-vec is available and the new record's embedding succeeded; silently skipped otherwise. |

### `memory.retention` — Retention job

The retention job runs as a silent background task on daemon startup if it is due, and then on the kernel clock every `memory.retention.interval`. When a run comes due it waits for 60 seconds of system-wide quiet, but at most `memory.retention.max_delay`, so steady traffic cannot starve it. It runs three passes: demotion (remove under-used records from the vector index), re-promotion (restore records whose access stats have risen since demotion), and backfill (embed re-promoted records that are missing vectors). The backfill pass is skipped when embedding is disabled by the `embedding_meta` mismatch guard.

The retention score formula is `importance × (1 + 0.5 × log1p(retrieval_count)) × exp(-age_days / half_life_days)`. All threshold and shape constants are the config keys documented below.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `memory.retention.grace_days` | float | `14` | Records younger than this many days are exempt from demotion regardless of score. Ensures newly consolidated records are not immediately demoted because they have not yet been retrieved. |
| `memory.retention.importance_floor` | float | `0.8` | Records with `importance ≥ importance_floor` are never demoted for lack of retrieval traffic. The importance prior ranges 0.0–1.0 (`AppraisalPlugin` wraps the default `RubricPrior` in `AppraisalPrior` when appraisal is enabled). |
| `memory.retention.demote_below` | float | `0.15` | Retention score threshold. An indexed record whose score falls below this value (and which passes the grace-period and importance-floor checks) is demoted: `indexed=0`, `embedded=0`, vec row deleted. The `memory` row and its FTS entry are preserved. Re-promotion requires the score to rise back to or above this threshold, which happens when the record is accessed via `recall_raw`. |
| `memory.retention.half_life_days` | float | `90` | Recency half-life for the retention score. Distinct from `memory.half_life_days` (retrieval scoring, default 30). The longer retention half-life gives passive retrieval time to re-warm a record before it demotes. |
| `memory.retention.interval` | float | `21600` | Seconds between retention job runs (default 6 hours); the period of the clock timer. The last-run timestamp is persisted in the `retention_meta` table so daemon restarts do not reset the interval. A daemon idle longer than `interval` since its last run will still run the job on next start (zero-traffic guarantee). Takes effect at restart. |
| `memory.retention.max_delay` | float | `3600` | Longest a due run waits for system quiet, in seconds. A run starts when the system has been quiet 60 seconds or when this much time has passed since it came due, whichever is first. Takes effect at restart. |
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

Each job:

| Key | Required | Description |
|-----|----------|-------------|
| `name` | yes | Unique; the sync key. Renaming a job disables the old row and creates a new one. |
| `channel` | yes | `transport:scope` to fire into. Any channel, including one no transport owns, such as `internal:notes`, whose replies go to the log and `sessions.db` only. |
| `origin` | no | `reminder` (default); becomes the `[origin]` notification prefix and `meta["origin"]`. `heartbeat` is still accepted so old configs load, but a scheduler heartbeat is superseded by the [`heartbeat`](#heartbeat--a-turn-on-system-idle) plugin; delete the job when you enable the plugin, or both will run turns on `internal:heartbeat`. |
| `every` | yes | Interval: `30s`, `20m`, `6h`, `1d`, `7d`, `1h30m`, or a number of seconds. |
| `at` | no | Anchor. `HH:MM`: the first fire is the next such time, then every `every`. An ISO datetime: slots are that instant plus multiples of `every` (`every: 7d` with `at: "2026-10-16T09:00"`, a Friday, is every Friday at 09:00). Without it, the first fire is one interval after the job is first synced. |
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
```

A job costs a main-model turn per firing on the shared backend; keep intervals in hours. The firing text is framed as data for the model to judge ("Decide whether and how to act on it"), not as an order.

---

## `heartbeat` — a turn on system idle

`HeartbeatPlugin` (cognition) starts a turn when the whole system has been quiet for a while: no channel has had inbound work for `heartbeat.after` seconds. The turn runs on `internal:heartbeat`, a channel no transport owns, so nothing it writes there reaches anyone. Its prompt asks the agent to review its recent memories and act only if something warrants it (see [cognition.md](cognition.md#heartbeatplugin)). The block is optional, and without it the heartbeat is asleep.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `heartbeat.after` | number (seconds) | none | Seconds of system-wide quiet before a beat, and the minimum gap between beats: each beat makes the system busy, so the next one comes `after` seconds after it settles. Traffic on any channel cancels the pending beat and starts the count again. Absent, `null`, or an empty `heartbeat:` block: the heartbeat is asleep. Zero, negative, non-finite, boolean or non-numeric values fail startup with a `ValueError` naming the key and the value. Read at startup only. |

**Sleep.** The heartbeat sleeps whenever `heartbeat.after` is absent. To put it to sleep, delete or comment out the block and restart; it stays asleep across every restart after that, because `agent.yaml` is the record. `plugins.disabled: [heartbeat]` does the same. Reloading the config doesn't wake it or put it to sleep; restart.

**Memory.** Setting `heartbeat.after` requires the `memory` plugin; with memory disabled, startup fails with a message saying so. The beat reviews the most recent memories (up to 20) of the *other* channels in its memory scope, newest first, never its own. Memory is compartmented per channel, so with no `memory.channel_groups` entry for `internal:heartbeat` it has nothing to review. Put it in a group with the channels it should look after:

```yaml
heartbeat:
  after: 3600           # an hour of quiet
memory:
  channel_groups:
    lex: [internal:heartbeat, "irc:#lex", "signal:*"]
```

Group membership works both ways, so the other channels in the group can retrieve the memories consolidated from beats too (see [threat-model.md](threat-model.md)).

**Cost.** Each beat is a main-model turn on the shared backend. A beat waits for quiet, so it never competes with a conversation, but a short `after` keeps the system from staying quiet long enough for memory retention's 60-second quiet window; retention then runs on its `max_delay` instead.

---

## `funnel` — context-admission funnel

Requires the cognition extra (`--extra cognition` on your `uv sync` line; see [cognition.md](cognition.md)). Without it these keys are ignored.

`FunnelPlugin` is the single chokepoint for tail CONTEXT admission (dedupe, budgets, injection framing). It also owns deferred registration: non-tool-call notification payloads queue per `(channel, origin)` via `register_and_wake()`, one stub notification wakes the channel per pending pair, and the drain in `before_agent_turn` admits everything queued for the triggering exchange's origin. `register()` queues a payload with no stub and no wake: it is admitted at the channel's next turn, whatever that turn's origin. The deferred registry is in-memory by design (payloads pending at shutdown are dropped) and adds no config keys — drained payloads spend the same per-source budgets below.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `funnel.default_budget` | integer | `512` | Token budget per `admit()` call when the source has no override. |
| `funnel.budgets.<source>` | integer | — | Per-source budget override (e.g. `funnel.budgets.memory`). The goals plugin admits under `funnel.budgets.goals`, which falls back to `funnel.default_budget` (`512`). |

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
| `tools.disabled` | list of strings | `[]` | Tool names dropped after collection, from any plugin (built-in or MCP), in `ToolCollectionPlugin.rebuild_registry`. This is the single chokepoint every registered tool passes through, so a disabled tool is absent from every channel's schemas and from subagents. `shell` and `task_pipeline` disable together: `task_pipeline` shells out on its own, so naming either removes both (PRD §3.5.3). A value that is not a list of names (a bare string, say) is rejected at startup. Not hot-reloaded. To narrow one channel instead, use `channels.<id>.tools.disabled`. See [threat-model.md](threat-model.md). |

---

## `irc` — IRC transport

The `irc` section is optional. If absent, `IRCPlugin` does not connect.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `irc.host` | string | `irc.libera.chat` | IRC server hostname. |
| `irc.port` | integer | `6667` | IRC server port. |
| `irc.nick` | string | `corvidae` | Bot nickname. |
| `irc.tls` | boolean | `false` | Use TLS for the IRC connection. |
| `irc.allow` | list of strings | `[]` (nobody) | Senders who may drive the agent, in channels and in private messages. **IRC is default-deny:** with no `allow`, or an empty one, every sender is refused, so a config that used IRC before this key existed goes silent until it is set. Startup logs one WARNING naming `irc.allow` when the key is absent. Where the network reports services accounts (IRCv3 `account-notify`/`account-tag`, or account data pydle learned from WHOIS), an entry matches the sender's **account**, not the nick: a user logged in as `alice` is admitted under any nick, and someone using the nick `alice` who is logged in as another account, or not identified (or whose account the bot doesn't know, as with a PM from someone who shares no channel with it on a network without `account-tag`), is refused. With `account-tag` negotiated the account comes only from that message's own tag (no tag means not logged in, and the nick is never used as a fallback); pydle's tracked user record is not consulted, so a stale record, a WHOIS reply or another line can't admit a later user of the same nick. Each message is judged once, when it arrives, and that verdict is the one used for it: a later line in the same socket read can't change it. Without `account-tag`, the account is whatever pydle's tracked user record holds at that moment, which can lag a logout (a residual risk, see `docs/threat-model.md`). On a network with no account support, an entry matches the **nick**, which is unauthenticated: anyone can take the nick, so treat the list as a convenience there, not a credential. Matching uses the server's CASEMAPPING (ascii, rfc1459), so `alice[` and `alice{` are one nick on an rfc1459 network. A refused sender gets no reply, no channel and no history, and one INFO line names the nick (and account when known). Anything but a list of strings raises at startup: `irc: config field 'allow' must be a list of nicks or account names`. Read at startup; changing it needs a restart. |
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

The `discord` section is optional. If absent, `DiscordPlugin` loads but stays inert: no client, no tasks, nothing logged above `DEBUG`. If present, it must be well-formed — no token (in the block or the environment), an `allow` that is not a list, or a `message_chunk_size` that is not an integer from 100 to 2000 raises at startup naming the offending field.

corvidae logs in as a Discord bot account. Creating the application, getting the token, and inviting the bot are covered in [discord-ops.md](discord-ops.md). In DMs the bot answers every message from an allowed user; in servers and threads it answers only messages that @-mention it.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `discord.token` | string | `$DISCORD_TOKEN` | The bot token. If the key is absent, null, or empty, the `DISCORD_TOKEN` environment variable is used; if neither is set, startup fails. |
| `discord.allow` | list of user ids | `[]` | Discord user ids permitted to drive the agent. Numeric ids, written as YAML ints or quoted strings; they are compared as strings. Default-deny: an empty or absent list refuses everyone, silently — no reply and no typing indicator. An explicit `allow: null` is an error, not an empty list. |
| `discord.message_chunk_size` | integer | `2000` | Maximum UTF-8 bytes per outbound Discord message, 100 to 2000 inclusive. Longer replies are split, keeping code fences intact, and sent as several messages; nothing is truncated. An explicit `null` is rejected (unlike a null `token`, which falls back to the environment). The limit is bytes, while Discord's 2000 is characters, so text with multi-byte characters splits earlier than strictly needed. |

```yaml
discord:
  token: "MTIz...your.bot.token"   # or leave out and set DISCORD_TOKEN
  allow:
    - 123456789012345678
  message_chunk_size: 2000
```

User ids are the long numeric snowflakes you get from Discord's Developer Mode, not usernames; see [discord-ops.md](discord-ops.md#find-ids). Prefer the environment variable if `agent.yaml` is world-readable or checked in anywhere: the token is the bot, and anyone holding it can act as it.

**Discord content is stored in plaintext on this host, and Discord has it too.** Discord messages are not end-to-end encrypted; Discord (the company) can read them in transit and at rest. On this side they are written to `sessions.db`, to the JSONL conversation log (`<daemon.jsonl_log_dir>/discord_<channel_id>.jsonl`, or `discord_dm_<user_id>.jsonl` for a DM, when that is enabled), and into consolidated memory records, all as ordinary text. Messages from senders who are not allowlisted are refused before they are persisted. In a server channel, though, the stored history is whatever allowlisted users said there, and the model's replies to it. File permissions and whatever disk encryption the host has are all that protect the local copies.

To remove a conversation's stored content, use `corvidae redact` as described for Signal above, with `channel_id = 'discord:<channel_id>'` or `'discord:dm:<user_id>'`.

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
| `system_prompt` | string or list of strings | `agent.system_prompt` | System prompt for this channel. Same format as `agent.system_prompt`. |
| `max_context_tokens` | integer | `agent.max_context_tokens` | Token budget for this channel's conversation history. |
| `max_turns` | integer | `agent.max_turns` | Maximum consecutive tool-calling turns for this channel. Must be an integer; a wrong type is rejected when the file is parsed (a `Config error` at startup, a logged reload error on reload), not at turn time. |
| `reasoning_history` | `drop` \| `exchange` \| `keep` | `agent.reasoning_history` | Controls `reasoning_content` retention for this channel. |
| `tools.disabled` | list of strings | `[]` | Tools this channel cannot use, on top of the global `tools.disabled`. The channel can only narrow; it can't re-enable a globally disabled tool, and `set_settings` can't change the list. `shell` and `task_pipeline` disable together. See [Per-channel tool sets](#per-channel-tool-sets). |

Any other key in an entry, exact or transport-wide, is rejected: at startup as `Config error: ... — check <path>`, on reload by refusing the whole reload (see [Error handling](#error-handling)). The message names `channels.<id>.<key>`, the rejected value and the accepted keys above. In particular `max_tokens` is not a channel key; it exists only as a `set_settings` runtime override. An operator `agent.yaml` that carries a stray key will fail to start after upgrading until the key is removed.

Channels not listed here are created on-demand when a message arrives and use matching transport-wide entries, then `agent` defaults.

```yaml
channels:
  irc:#general:
    system_prompt: "You are the channel bot."
    max_context_tokens: 8000
  cli:local:
    system_prompt:
      - prompts/base.txt
      - prompts/cli-addendum.txt
    reasoning_history: keep
```

### Transport-wide entries

Some channels can't be listed ahead of time. Signal DMs are keyed on each contact's ACI, ACP sessions get a fresh id per session, and the bot answers in any Discord channel it's mentioned in. A key ending in `*` covers all of them: it matches every channel whose id starts with the text before the `*`, including channels that first appear after the config was loaded.

```yaml
channels:
  signal:*:                        # every Signal DM, current and future
    system_prompt: [prompts/SOUL.md, prompts/chat.md, prompts/signal_dm.md]
  discord:*:                       # Discord server channels and threads (and DMs, below)
    system_prompt: [prompts/SOUL.md, prompts/chat.md, prompts/discord_group.md]
    tools:
      disabled: [shell, web_fetch]
  discord:dm:*:                    # Discord DMs: more specific than discord:*
    system_prompt: [prompts/SOUL.md, prompts/chat.md]
  signal:8f2c1c9a-3e21-4b77-9c3f-1a2b3c4d5e6f:
    max_turns: 20                  # one contact keeps the Signal prompt, with more turns
```

A transport-wide entry accepts the same keys as an exact one. Each setting resolves in this order, last wins: built-in default, `agent:`, matching transport-wide entries from least to most specific (so `discord:dm:*` beats `discord:*` whatever order they're written in), the channel's exact entry, then `set_settings` runtime overrides. A Signal phone-number alias counts as an exact entry. `tools.disabled` is the exception: it's the union of the global list and every matching entry, so a more specific entry can narrow further but can't re-enable a tool.

The accepted key forms are `transport:*` and `transport:<scope-prefix>*`. A `*` anywhere else in a key ending in `*`, or a key with no `transport:` before the `*`, is rejected at startup, and on reload, naming the key. Only a trailing `*` is special, so an exact key like `irc:#c*t` still works; the one casualty is an IRC channel whose name itself ends in `*`, which now reads as a transport-wide key. Matching is on channel ids, so `signal:+1*` would match only phone-number alias entries, never a conversation (Signal conversations are keyed on ACIs). A transport-wide key is never a channel itself. This is a prefix match, not the glob syntax `memory.channel_groups` uses: `*` means "the rest of the id" and only at the end, and `?` has no meaning.

Transport-wide entries reload like `agent:` does: a running channel sees the new values at its next use. `max_turns`, `max_context_tokens` and `reasoning_history` apply from the next turn, and `tools.disabled` follows the rules under [Per-channel tool sets](#per-channel-tool-sets). A conversation's system prompt is fixed when it starts and survives compaction, so a prompt edit reaches only channels created after the reload, or everything after a restart. Removing a transport-wide entry also takes effect live. Each running channel whose matching entries change is logged at INFO.

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
      timeout_seconds: 30      # optional; per tool call; default 30
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
      timeout_seconds: 30              # optional; per tool call; default 30
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
restart; each failed retry logs a WARNING naming the server, the error and the
next delay. An MCP tool call also retries pending servers as a fast path,
unless a retry is already running. A server skipped for a config error (unknown
`transport`, missing `command`/`url`, non-positive `connect_timeout_seconds`,
unset `${VAR}` in `headers`) is never retried: fix the config and restart. The
`mcp` section itself is not hot-reloaded.

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
| `agent.request_logprobs` | `Agent` re-reads the flag; takes effect on the next LLM call. |
| `agent.mid_conversation_system_role` | `Agent` re-reads the role; takes effect on the next prompt build. The changed rendering of earlier turns costs one full prompt-cache miss. |
| `agent.compaction_prompt`, `agent.compaction_fork_prompt` | `CompactionPlugin` uses the new prompt on its next compaction. |
| `agent.forks.*` | `Agent` re-reads the section; takes effect on the next fork. A reload with an invalid section is rejected whole. |
| `appraisal.stage2_fork_prompt`, `critique.fork_prompt.<lens>` | Read from the reloaded config on the next appraisal/critique fork. |
| `scheduler.*` | `SchedulerPlugin` re-reads its settings and re-syncs `jobs` by name (insert, update, disable removed), then wakes its timer so the change takes effect immediately. |
| Dotted plugin tunables (`appraisal.*`, `critique.*`) | Resolved per-decision via `corvidae.tuning.resolve_tunable`; a reloaded value applies on the next appraisal/critique decision. Per-channel `set_settings` overrides still take precedence. |
| `agent.immutable_settings` | `RuntimeSettingsPlugin` resets its blocklist and re-applies the new list. Constructor-supplied immutable keys are always retained. |
| New channel entries in `channels:` | `load_channel_config` registers newly added channels. |
| `channels.<id>.tools.disabled` on a registered channel | Updated in place. Dispatch refuses a newly disabled tool at once; the schemas follow at the next compaction or conversation start, never mid-conversation. See [Per-channel tool sets](#per-channel-tool-sets). A reload with a malformed entry changes no channel. |
| Transport-wide entries (`channels.<transport>:*`) | Replaced and re-matched against every running channel. Scalar keys apply from the next turn, `tools.disabled` as above, the system prompt only to conversations that start after the reload. See [Transport-wide entries](#transport-wide-entries). |
| `web.token` | A changed token logs out every browser (their sockets close and the page says the session expired). A token removed from config falls back to `CORVIDAE_WEB_TOKEN` as the daemon started with it, else a fresh generated one, which is logged. Changing `CORVIDAE_WEB_TOKEN` itself needs a restart. |
| `agent` defaults | `ChannelRegistry.agent_defaults` is updated. `max_turns`, `max_context_tokens` and `reasoning_history` are resolved every turn, so running channels pick them up at their next turn; `system_prompt` reaches only conversations that start after the reload. |

### What requires restart

| Setting | Reason |
|---------|--------|
| `irc.host`, `irc.port`, `irc.nick` | IRC transport is connected at startup; changing connection parameters requires reconnection. |
| `irc.allow` | The allowlist is parsed once in `on_init`; a reload keeps the old one. |
| `web.host`, `web.port`, `web.allowed_origins`, adding or removing the `web` block | The listener is bound at startup. A reload logs `restart to apply` and keeps serving with the old values. |
| Existing exact entries in `channels:` | Channels already registered retain their original `ChannelConfig` until restart, except `tools.disabled` (above). Transport-wide entries are live (above). |
| `llm.background` | Not updated on reload. |
| `memory.*` (including `memory.consolidation_prompt` and `memory.consolidation_fork_prompt`) | `MemoryPlugin` reads its section at init only. That includes `memory.idle_consolidate_after` and `memory.retention.*`: the idle watch and retention timer are set when the plugin starts. |
| `heartbeat.after` | `HeartbeatPlugin` reads it at init and registers its idle watch at start. Putting the heartbeat to sleep or waking it takes a restart. |
| `tools.max_result_chars`, `tools.shell_timeout`, `tools.web_fetch_timeout` | Deferred; not currently implemented in hot-reload path. |

### Error handling

At startup (not reload), a missing or malformed config file, or a rejected
value in the `agent` or `channels` sections (e.g. a `channels.<id>.tools`
block that is not `{disabled: [names]}`, or an unknown key such as
`channels.<id>.max_turn`), prints
`Config error: <detail> — check <path>` to stderr and exits non-zero. The
detail names the key, the rejected value and what is accepted.

- **Invalid YAML**: logged at `ERROR`; reload is skipped and the running config is unchanged.
- **Rejected value in `agent:`, `daemon:` or `channels:`** (e.g. a malformed `channels.<id>.tools`): logged at `ERROR` naming the file. For `channels:` the message also names the key and the value and says "reload skipped"; for `agent:` and `daemon:` the key is in the logged traceback. The reload is skipped whole: no section is applied and no plugin receives `on_config_reload`, so a typo in one `channels:` entry cannot change `agent:` values.
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
