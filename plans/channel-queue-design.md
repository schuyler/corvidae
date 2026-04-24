# Per-Channel Serial Queue, on_notify Hook, and Multi-LLM Config

## Context

Sherman's agent loop has no concurrency protection per channel. Two messages arriving on the same channel race through conversation init, message appending, agent loop execution, and persistence. Background task completions call `send_message` directly — the LLM never sees task results in its conversation history and can't react to them.

This design introduces:
1. Per-channel serial queue for all agent loop invocations
2. `on_notify` hookspec for plugins to inject messages into a channel
3. Separate LLM provider config for background tasks (breaking config change)

## Config Change: `llm.main` / `llm.background`

**Breaking change.** The flat `llm` block becomes structured:

```yaml
llm:
  main:
    base_url: http://sagan.lan:8080
    model: qwen3.6-35b
    extra_body:
      id_slot: 0
  background:                    # optional — absent means use llm.main
    base_url: http://sagan.lan:8080
    model: qwen3.6-35b
    extra_body:
      id_slot: 1
```

- `llm.main` is required (replaces current flat `llm` fields).
- `llm.background` is optional. If absent, background tasks use `llm.main`.
- If `llm.background` is present, ALL fields must be specified (no per-field inheritance from main). Keeps config semantics simple and explicit.
- `extra_body` does not deep-merge — each block is self-contained.
- Extensible: future blocks like `llm.summarizer` follow the same pattern.

**Files:** `sherman/agent_loop_plugin.py` (on_start config parsing), `agent.yaml.example`, `sherman/main.py` (if config validation exists there)

## New Module: `sherman/channel_queue.py`

```python
@dataclass
class QueueItem:
    role: str            # "user" or "notification"
    content: str
    sender: str | None   # For user messages; None for notifications
    source: str | None   # For notifications: "background_task", etc.
    tool_call_id: str | None  # For deferred tool results (background_task completion)
    meta: dict           # Extensible (task_id, etc.)

class ChannelQueue:
    # asyncio.Queue[QueueItem] + consumer task
    # enqueue(item) — add to queue
    # start(process_fn) — launch consumer task
    # stop() — cancel consumer
    # drain() — await queue.join() (for tests)
```

- One `ChannelQueue` per channel, created lazily on first enqueue.
- Owned by `AgentLoopPlugin` as `_queues: dict[str, ChannelQueue]`.
- Consumer calls `_process_queue_item` which contains the current `on_message` body.
- Errors logged and swallowed — consumer continues to next item.

**New file:** `sherman/channel_queue.py`
**New test file:** `tests/test_channel_queue.py`

## Hook Changes

### New hookspec: `on_notify`

```python
@hookspec
async def on_notify(
    self, channel: Channel, source: str, text: str,
    tool_call_id: str | None = None, meta: dict | None = None
) -> None:
```

- Enqueues a `QueueItem(role="notification", ...)` on the channel's queue.
- `tool_call_id` is set when the notification is a deferred result of a prior tool call (e.g., background task completion).

**File:** `sherman/hooks.py`

### Notification message role strategy

The queue consumer formats notification messages based on whether a `tool_call_id` is present:

- **With `tool_call_id`** (background task completion): `{"role": "tool", "tool_call_id": "...", "content": "..."}`. Semantically correct — the LLM initiated an async operation and this is its deferred result.
- **Without `tool_call_id`** (other notification sources): `{"role": "system", "content": "[{source}]\n\n{text}"}`. Injected system context, not a user message.

The `tool` role approach for background tasks requires `BackgroundTask` to capture the `tool_call_id` from the original `background_task` tool call at enqueue time.

**If llama-server rejects deferred `tool` messages** (tool response far from its originating assistant message), fall back to `system` for everything. This is a testable hypothesis — implement `tool` first, verify against llama-server.

## AgentLoopPlugin Changes

### `on_start`
- Parse `llm.main` (required) instead of flat `llm`.
- Create `self.client` from `llm.main`.
- Parse `llm.background` (optional). If present, create `self.bg_client` and `self.bg_extra_body`. Call `bg_client.start()`.
- Initialize `self._queues: dict[str, ChannelQueue] = {}`.

### `on_message` (refactored)
- Becomes fire-and-enqueue: validates client, logs, enqueues `QueueItem(role="user")`, returns.
- Current body moves to `_process_queue_item(channel_id, item)`.

### `on_notify` (new hookimpl)
- Enqueues `QueueItem(role="notification")` on the channel's queue.

### `_process_queue_item` (new, extracted from on_message)
- Dispatches on `item.role` to format the conversation message.
- `"user"` → `{"role": "user", "content": item.content}`
- `"notification"` with `tool_call_id` → `{"role": "tool", "tool_call_id": "...", "content": item.content}`
- `"notification"` without `tool_call_id` → `{"role": "system", "content": f"[{item.source}]\n\n{item.content}"}`
- Then: ensure_conversation, append, compact, build prompt, run agent loop, persist, strip thinking, send_message.

### `_on_task_complete` (changed)
- Keeps `pm.ahook.on_task_complete()` call for observability.
- Replaces direct `send_message` with `pm.ahook.on_notify(channel, source="background_task", text=..., tool_call_id=task.tool_call_id, meta={"task_id": ...})`.
- The agent loop now processes the notification and generates a response — the LLM sees the result and can react.

### `BackgroundTask` change (`sherman/background.py`)
- Add `tool_call_id: str | None = None` field to `BackgroundTask` dataclass.

### `run_agent_loop` change (`sherman/agent_loop.py`)
- When calling a tool function, inspect its signature. If it declares a `_tool_call_id` parameter, pass `_tool_call_id=call_id`. Otherwise, call with just `**args` as before. Backward compatible — existing tools are unaffected.
- The `background_task` closure in `on_message` adds `_tool_call_id: str` to its signature. It stores the value on the `BackgroundTask` at enqueue time.
- This flows: LLM tool_call → agent_loop passes call_id → background_task closure → BackgroundTask.tool_call_id → on_notify(tool_call_id=...) → QueueItem → conversation message `{"role": "tool", "tool_call_id": "..."}`.

### `_execute_background_task` (changed)
- Uses `self.bg_client or self.client` and corresponding `extra_body`.

### `on_stop` (changed)
- Stops all channel queue consumers.
- Stops `bg_client` if present.

**File:** `sherman/agent_loop_plugin.py`

## CLI Plugin Changes

`on_message` now returns before the agent loop runs. The `>` prompt in `_read_loop` currently appears after `on_message` completes (i.e., after the response). With fire-and-enqueue, it appears immediately.

Options:
- Move prompt display to `send_message` (print `>` after each response).
- Or: accept that CLI prompt timing changes slightly — the user experience is effectively the same for a serial single-channel transport.

Recommendation: move prompt to `send_message`. Minor change.

**File:** `sherman/cli_plugin.py`

## Test Migration

Existing tests await `on_message` then assert on `send_message` calls. With the queue, `on_message` returns before processing. Tests need to call `await plugin._queues[channel.id].drain()` after `on_message` to wait for processing to complete.

The `_build_plugin_and_channel` helper in tests should be updated to expose a drain helper.

**File:** `tests/test_agent_loop_plugin.py`

## Verification

1. All existing tests pass (after migration to drain pattern).
2. New tests for:
   - `ChannelQueue` serialization (two items enqueued, processed in order)
   - `on_notify` enqueues and triggers agent loop
   - `_on_task_complete` uses `on_notify` path
   - Background tasks use `bg_client` when configured
   - Background tasks fall back to `self.client` when `llm.background` absent
   - `on_stop` cleans up all queues and both clients
   - Config parsing for `llm.main` / `llm.background`
3. Manual test: `sherman --cli` with a background task — verify the LLM acknowledges task completion.

## Implementation Order

1. Design review gate
2. Red: failing tests for channel_queue, on_notify, config change, bg_client
3. Red review gate
4. Green: implement all changes
5. Green review gate
6. Documentation (MEMORY.md, agent.yaml.example)
7. Documentation review gate
8. Final test verification
9. Acceptance check

---

## Design Review

**Reviewer:** Groucho (Project Architect)
**Files reviewed:** `plans/channel-queue-design.md`, `sherman/agent_loop_plugin.py`, `sherman/hooks.py`, `sherman/agent_loop.py`, `sherman/background.py`, `sherman/cli_plugin.py`, `tests/test_agent_loop_plugin.py`, `tests/test_background.py`, `sherman/conversation.py`, `sherman/channel.py`, `sherman/llm.py`

### CRITICAL

**C1. drain() deadlocks if the consumer task has crashed.**

`drain()` as `await queue.join()` blocks until every `task_done()` is called. If the consumer crashes mid-item without calling `task_done()`, drain blocks forever. The design says "errors logged and swallowed" but doesn't specify that `task_done()` must be in a `finally` block. The existing `TaskQueue.run_worker` in `background.py` already uses `finally: self.queue.task_done()` — the `ChannelQueue` consumer must replicate this pattern. Additionally, if the outer consumer loop itself crashes, the consumer stops entirely and drain deadlocks on all future items. The design must require `task_done()` in a `finally` block and specify that the outer loop catches all exceptions.

**C2. `_process_queue_item` needs a full `Channel` object, not just `channel_id`.**

The existing `on_message` body uses `channel` everywhere (`ensure_conversation(channel)`, `send_message(channel=channel, ...)`, etc.). `QueueItem` as designed carries only `sender`, `source`, `tool_call_id`, `content`, `meta` — no `Channel`. The design must resolve this: either (a) add `channel: Channel` to `QueueItem` (analogous to `BackgroundTask.channel`), or (b) make `_process_queue_item(channel: Channel, item: QueueItem)` with the Channel looked up before enqueue. Option (a) is cleaner. Must be resolved before Red TDD.

**C3. `_tool_call_id` parameter in `background_task` signature will appear in LLM tool schema.**

`tool_to_schema` iterates `sig.parameters` and emits JSON schema for all of them. Adding `_tool_call_id: str` to the `background_task` closure means the LLM will be told to supply it — which is wrong. The design must specify that `tool_to_schema` skips `_`-prefixed parameters, and that `run_agent_loop` injects them separately (not from the LLM-supplied `args` dict). Red TDD tests must cover both the schema-exclusion case and the call-injection case.

### IMPORTANT

**I1. on_stop drops in-flight items silently.** Cancelling the consumer task abandons any queued or mid-processing items. This is consistent with existing behavior for background tasks, but now affects user messages too. Should be documented explicitly.

**I2. Notification-triggered `on_agent_response` will have unexpected `request_text`.** Notifications trigger the agent loop, which calls `on_agent_response(request_text=...)`. The design doesn't specify what `request_text` is set to for notifications. Logging/observability plugins will see notification content as if it were a user message. Needs a decision.

**I3. `background_task` closure is built in `_process_queue_item`, not `on_message`.** The design's language says "the closure in `on_message`" — this should say `_process_queue_item`. Clarify in the design to avoid implementer confusion.

**I4. `on_notify` hookspec + hookimpl on the same class.** This is valid in apluggy, but the design should clarify whether external plugins implementing `on_notify` is intentional (observer pattern) or not.

**I5. `tool` role hypothesis: high structural-validity risk.** A `tool` message many turns after its originating `assistant/tool_calls` message violates the chat completions protocol as most llama-server builds implement it. Test against llama-server early in Green before investing in the full `tool` path. Have the `system` fallback ready.

**I6. `bg_client` / `bg_extra_body` not initialized in `__init__`.** Must add `self.bg_client: LLMClient | None = None` and `self.bg_extra_body: dict | None = None` to `AgentLoopPlugin.__init__` to avoid `AttributeError` when `llm.background` is absent.

### COSMETIC

**Co1.** `meta: dict` in `QueueItem` needs `field(default_factory=dict)` — bare mutable default fails at class definition time.

**Co2.** `ChannelQueue.start(process_fn)` — document `process_fn` signature explicitly.

**Co3.** Existing `BASE_CONFIG` in tests uses flat `llm` block; migration to `llm.main` must be an explicit Red TDD checklist item.

**Co4.** `sherman/main.py` not audited — check whether it preprocesses the `llm` config block before Red TDD.

### Recommendation

**Proceed to Red TDD: YES**, with these preconditions resolved in test specification (not requiring redesign):
1. (C1) Consumer uses `task_done()` in `finally`; drain test covers consumer crash case.
2. (C2) `QueueItem` carries `channel: Channel` (or signature is `(channel, item)`).
3. (C3) `tool_to_schema` filters `_`-prefixed params; tests cover schema-exclusion and call-injection.

---

## Baseline Test Gate

- Date: 2026-04-23
- Command: uv run pytest --tb=no -q
- Total tests: 233
- Result: PASS
- Notes: 1 warning (coroutine never awaited in test_tools.py::TestShell::test_shell_no_output) — pre-existing, not a failure

## Red TDD Report

- Git HEAD: 73213f622c322267242698394c28389dad8b4679
- New test file: tests/test_channel_queue.py (14 tests — collection error: ModuleNotFoundError for sherman.channel_queue, as expected)
- Modified: tests/test_agent_loop_plugin.py (28 tests migrated with drain() calls + BASE_CONFIG to llm.main, 12 new tests added; 40 total)
- Test run result: 36 failed, 4 passed, 1 collection error (expected: all new/migrated fail)
- Notes:
  - The 4 passing tests are: test_on_stop_cleans_up (pre-existing behavior unchanged), test_tool_call_id_not_in_background_task_schema, test_underscore_params_excluded_from_schema (schema tests pass because pydantic excludes params with default=None from the required set — these serve as regression guards), test_main_client_used_when_llm_background_absent (current code already uses self.client when no bg_client is configured).
  - test_on_start_missing_llm_main_raises: the old flat-llm config raises KeyError at 'base_url', but the test expects KeyError or ValueError — it DID NOT RAISE because the old config structure caused the plugin to look up llm_config["base_url"] on the dict {"base_url": ..., "model": ...} and succeed. This correctly fails, confirming the test will gate on the new config parsing.
  - test_channel_queue.py uses --continue-on-collection-errors to get it counted separately; in the joint run it blocks all collection. All 14 tests in that file will fail with ImportError until the module is created.

---

## Red TDD Review

**Reviewer:** Chico (Code Reviewer)
**Files reviewed:** design doc, tests/test_channel_queue.py, tests/test_agent_loop_plugin.py, sherman/agent_loop.py, sherman/agent_loop_plugin.py, sherman/hooks.py, sherman/background.py

### CRITICAL

**CR1. C3 schema-exclusion tests pass vacuously.**
`test_tool_call_id_not_in_background_task_schema` and `test_underscore_params_excluded_from_schema` pass in pre-implementation code because pydantic silently drops `_`-prefixed field names from `create_model` as private attributes — not because any explicit filter exists. The Green implementer must add an explicit `_`-prefix filter in `tool_to_schema` (`sherman/agent_loop.py` lines 147-153) regardless of whether pydantic would handle it incidentally. Do not rely on pydantic's private-attribute convention as the implementation strategy.

**CR2. `test_on_task_complete_does_not_call_send_message_directly` produces misleading failure.**
After `send_message.assert_not_awaited()`, the test calls `await _drain(plugin, channel)`. With the current code, `plugin._queues[channel.id]` raises `KeyError` — not an assertion failure. The `_drain` helper should be guarded: `if channel.id in plugin._queues: await plugin._queues[channel.id].drain()`.

### IMPORTANT

**I1. `on_notify` hookspec missing from `hooks.py`.**
`test_on_notify_*` tests call `pm.ahook.on_notify(...)` which will raise `AttributeError` in apluggy when the hookspec is not registered. Adding `on_notify` to `hooks.py` is a Green TDD prerequisite (acceptable that it fails with AttributeError rather than NotImplemented in Red).

**I2. Outer consumer loop crash not tested (partial C1 gap).**
`test_drain_returns_even_when_process_fn_raises` covers `task_done()` in `finally`. Neither test covers an exception escaping the inner handler entirely. Low risk but noted.

### COSMETIC

- Co1: `test_new_enqueues_not_processed_after_stop` uses `asyncio.sleep(0.05)` — timing-dependent, non-deterministic under load.
- Co2: `test_multiple_items_all_processed` (5 items) is redundant with `test_items_processed_in_order` (2 items).
- Co3: `test_on_start_collects_tools` uses index `[0]` — should use `any(...)` in case built-in tools are inserted first.

### Positive

Drain migration is consistent across all 28 existing tests. `test_drain_returns_even_when_process_fn_raises` uses `asyncio.wait_for` with `pytest.fail` — correct deadlock test pattern. `BASE_CONFIG` migration complete.

### Proceed to Green TDD: YES

Both critical findings are addressable as explicit implementer instructions — no new Red loop required. All design Verification items are covered with the right intent.

---

## Green TDD Report

- Git HEAD: 79af48fc72e1a7ed3f1ef96dd38cf6d72aea9cf1
- Files changed: sherman/channel_queue.py (new), tests/test_channel_queue.py (new), sherman/hooks.py, sherman/background.py, sherman/agent_loop.py, sherman/agent_loop_plugin.py, sherman/cli_plugin.py, agent.yaml.example, tests/test_agent_loop_plugin.py, tests/test_background.py, tests/test_logging.py, tests/test_main.py
- Test result: 255 passed, 0 failed (baseline was 233; +22 new tests)
- Notes:
  - CR1 (C3 schema exclusion): Explicit `_`-prefix filter added in `tool_to_schema` as required. Pydantic's private-attribute behavior was NOT relied upon.
  - CR2 (_drain guard): `_drain` helper updated to guard against missing queue with `if channel.id in plugin._queues`.
  - I1 (on_notify hookspec): Added to `hooks.py` as required.
  - Tool role hypothesis (I5): Implemented `tool` role path for notifications with `tool_call_id`. Test `test_on_notify_with_tool_call_id` passes — the `tool` message is stored in conversation history correctly. Runtime behavior against llama-server remains to be tested manually.
  - Critical pluggy/apluggy finding: Optional parameters with defaults in hookspecs are NOT forwarded to hookimpls by pluggy. This affected `on_notify(tool_call_id=..., meta=...)`. Fix: removed defaults from the `on_notify` hookspec and hookimpl signatures, making all params required. All callers updated to pass `tool_call_id=None, meta=None` explicitly when not applicable.
  - Background integration tests needed drain() calls added since `_on_task_complete` now routes through `on_notify` (async queue) instead of calling `send_message` directly.
  - A commit `4636477` (IRC constants fix) was made to main between Red TDD and Green TDD, so the starting HEAD for Green TDD was `4636477` not `73213f6`. No conflicts with channel-queue changes.
