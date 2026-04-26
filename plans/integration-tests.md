# Integration Test Suite Design

## Current State

Sherman has ~20 unit test files covering individual components in isolation. The test suite includes near-integration tests in `tests/test_agent_single_turn.py` that wire AgentPlugin + TaskPlugin + real SerialQueues together. However:

- No test assembles the full plugin graph as `main.py` does
- `send_message` and `on_agent_response` hooks are mocked in every test that uses them
- Conversation persistence is never tested across a stop/restart cycle
- Multi-channel concurrent behavior is untested
- Tool collection across all plugins (CoreToolsPlugin + SubagentPlugin + TaskPlugin) is never verified as an integrated whole

## Test Infrastructure

### Location

`tests/test_integration.py` (single file; split if it exceeds ~500 lines)

### IntegrationHarness

A pytest fixture that assembles the real plugin graph, substituting only the LLM client and transport plugins.

```python
@dataclass
class IntegrationHarness:
    pm: PluginManager
    agent: AgentPlugin
    task_plugin: TaskPlugin
    registry: ChannelRegistry
    transport: FakeTransportPlugin
    mock_client: MagicMock

    def set_llm_responses(self, responses: list[dict]) -> None:
        self.mock_client.chat = AsyncMock(side_effect=responses)

    async def inject_message(self, channel_key: str, sender: str, text: str) -> None:
        transport, scope = channel_key.split(":", 1)
        channel = self.registry.get_or_create(transport, scope)
        await self.pm.ahook.on_message(channel=channel, sender=sender, text=text)

    async def drain_all(self) -> None:
        for queue in self.agent.queues.values():
            await queue.drain()
        if self.task_plugin.task_queue:
            await self.task_plugin.task_queue.queue.join()
            # Worker needs 3 cycles: resume → await on_complete → on_notify → enqueue
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            await asyncio.sleep(0)
        # Second drain for notification re-entry
        for queue in self.agent.queues.values():
            await queue.drain()

    async def drain_until_stable(self, max_iterations: int = 20) -> None:
        """Drain repeatedly until no new messages are sent.

        For multi-tool-call tests where re-entry generates further
        notifications. Requires 2 consecutive stable iterations (no new
        sent messages and task_queue.is_idle) before returning.
        Raises AssertionError if the limit is hit — likely indicates
        infinite re-enqueueing.
        """
        stable_count = 0
        for i in range(max_iterations):
            prev_count = len(self.sent_messages)
            await self.drain_all()
            queue_idle = (
                self.task_plugin.task_queue is None
                or self.task_plugin.task_queue.is_idle()
            )
            if len(self.sent_messages) == prev_count and queue_idle:
                stable_count += 1
                if stable_count >= 2:
                    return
            else:
                stable_count = 0
        raise AssertionError(
            f"drain did not stabilize after {max_iterations} iterations"
        )

    @property
    def sent_messages(self) -> list[tuple[Channel, str, float | None]]:
        return self.transport.sent
```

### FakeTransportPlugin

Replaces CLIPlugin and IRCPlugin. Captures `send_message` calls.

```python
class FakeTransportPlugin:
    depends_on = {"registry"}

    def __init__(self, pm):
        self.pm = pm
        self.sent: list[tuple[Channel, str, float | None]] = []
        self._registry = None

    @hookimpl
    async def on_start(self, config):
        self._registry = get_dependency(self.pm, "registry", ChannelRegistry)

    @hookimpl
    async def send_message(self, channel, text, latency_ms=None):
        self.sent.append((channel, text, latency_ms))

    @hookimpl
    async def on_stop(self):
        pass
```

### Fixture assembly

Mirrors `main.py` plugin registration order:

1. ChannelRegistry (named "registry")
2. PersistencePlugin (named "persistence")
3. CoreToolsPlugin (named "core_tools")
4. FakeTransportPlugin (named "fake_transport") — replaces CLI + IRC
5. TaskPlugin (named "task")
6. SubagentPlugin (named "subagent")
7. McpClientPlugin (named "mcp")
8. CompactionPlugin (named "compaction")
9. ThinkingPlugin (named "thinking")
10. AgentPlugin (named "agent_loop")
11. IdleMonitorPlugin (named "idle_monitor")

### LLM client injection

Patch `sherman.agent.LLMClient` before calling `on_start`, matching the
approach used in `test_main.py`. The patched class returns the harness's
`mock_client` from its constructor. This allows `AgentPlugin._start_plugin`
to run its normal path (including `await self.client.start()`) without
hitting a real API.

```python
mock_client = AsyncMock(spec=LLMClient)
mock_client.start = AsyncMock()
mock_client.stop = AsyncMock()

with patch("sherman.agent.LLMClient", return_value=mock_client):
    await pm.ahook.on_start(config=config)
```

**SubagentPlugin LLM patch**: `SubagentPlugin._launch` imports `LLMClient`
from `sherman.llm` (not `sherman.agent`) and creates its own client instance.
The patch above does not cover subagent LLM calls. Tests that exercise the
`subagent` tool must additionally patch `sherman.tools.subagent.LLMClient`.
For the standard integration harness this is not needed — subagent tool calls
are not exercised in the core test groups.

### Config fixture

The harness needs a config dict that matches what `on_start` expects. Minimal
shape:

```python
def make_config(*, db_path: str = ":memory:", extra: dict | None = None) -> dict:
    config = {
        "agent": {
            "system_prompt": "You are a test agent.",
            "max_context_tokens": 4096,
        },
        "llm": {
            "main": {
                "base_url": "http://fake",
                "model": "test-model",
            },
        },
        "daemon": {
            "session_db": db_path,
        },
        "_base_dir": Path("."),
    }
    if extra:
        config.update(extra)
    return config
```

For Group C persistence tests, pass `db_path=str(tmp_path / "test.db")`.

Call `validate_dependencies(pm)` and `pm.ahook.on_start(config=config)`.

### Boundary Decisions

| Component | Real or Mock | Rationale |
|---|---|---|
| Plugin manager (apluggy) | Real | Core integration surface; mocking defeats the purpose |
| ChannelRegistry | Real | Stateful, key to multi-channel tests |
| SerialQueue | Real | Async ordering correctness is what we're testing |
| TaskQueue + TaskPlugin | Real | The task → notify → re-entry cycle is the primary integration target |
| PersistencePlugin + SQLite | Real | Persistence correctness is a key integration concern |
| CompactionPlugin | Real | Compaction → summary cycle is an integration target |
| ThinkingPlugin | Real | `<think>` stripping and display hooks must fire in the integrated path |
| IdleMonitorPlugin | Real | Background poll task affects shutdown sequence |
| LLMClient | Mock (AsyncMock) | No real LLM server; scripted responses via side_effect |
| CoreToolsPlugin | Real registration | Tool schema collection matters |
| CLIPlugin / IRCPlugin | Replaced by FakeTransportPlugin | Avoid stdin/stdout/network dependencies |
| SubagentPlugin | Real registration | Verify tool collection participation |
| Actual tool functions | Mock or real depending on test | Shell/web_fetch mocked; simple test tools created per-test |

### Drain pattern

The existing `_drain` / `_drain_task_queue` pattern from `test_agent_single_turn.py` works well. The harness `drain_all()` method chains: drain all channel queues, then join the task queue + yield (3 times — see design review finding below), then drain channel queues again (to process notifications that arrived from completed tasks). For multi-tool-call tests, use `drain_until_stable()` which loops `drain_all()` until `sent_messages` count stabilizes (2 consecutive stable iterations with `task_queue.is_idle` checks), with a hard limit of 20 iterations that raises `AssertionError` on breach.

## Test Cases

### Group A: Full Lifecycle

**A1. Plugin graph assembly and dependency validation**

Assemble all 11 plugins as main.py does (with FakeTransportPlugin instead of CLI/IRC). Call `validate_dependencies(pm)`. Verify no RuntimeError.

**A2. on_start initializes all components**

After on_start, verify:
- `AgentPlugin.tools` contains expected names: shell, read_file, write_file, web_fetch, subagent, task_status
- `AgentPlugin.tool_schemas` has 6 entries, each with `type: "function"` *(count is sensitive to plugin graph — update if new tool-providing plugins are added)*
- `AgentPlugin.client` is the mock LLM client
- `TaskPlugin.task_queue` is not None
- `TaskPlugin._worker_task` is running

**A3. on_stop tears down cleanly**

After on_start, process one message, then call on_stop. Verify:
- All SerialQueue consumer tasks are done
- TaskPlugin worker task is done
- IdleMonitorPlugin monitor task is cancelled
- LLM client `stop()` was called
- No unfinished asyncio tasks remain (check via `asyncio.all_tasks()` delta). Note: IdleMonitorPlugin creates a background polling task (`trylast=True` on `on_start`); its `on_stop` cancels it. The delta check must account for this.

### Group B: Message-Response Round-Trip

**B1. Simple message → text response**

Send user message. LLM returns `{"choices": [{"message": {"role": "assistant", "content": "Hello!"}}]}`. Verify:
- FakeTransportPlugin.sent has exactly one entry
- Sent text is "Hello!"
- Conversation log has 2 messages: user + assistant

**B2. Message → tool call → result → response (full cycle)**

Send user message. LLM response sequence (2 calls):
1. Tool call for `test_echo` with args `{"text": "hi"}`
2. Text response "The output was: hi"

After `on_start`, override `agent.tools["test_echo"]` with a simple async
function that returns `"hi\n"` and add its schema to `agent.tool_schemas`.
The override-after approach avoids colliding with the real `shell` tool
registered by CoreToolsPlugin. Use `drain_until_stable()` (not `drain_all()`)
for this 2-turn cycle. Verify:
- Tool function was called with correct args
- Conversation log contains: user, assistant (with tool_calls), tool (result), assistant (text)
- FakeTransportPlugin.sent has exactly one entry with final text
- latency_ms is present and >= 0

**B3. Multiple tool calls dispatched as separate tasks**

LLM returns 3 tool calls in one response. Mock `side_effect` list has 4 entries:
1. Response with 3 tool calls (initial call)
2. Text response after first tool result (continuation 1)
3. Text response after second tool result (continuation 2)
4. Text response after third tool result (continuation 3)

Verify:
- 3 tasks enqueued on TaskQueue
- All 3 tool results delivered via on_notify
- LLM called 4 times total (1 initial + 3 continuations)
- `drain_until_stable()` used to handle non-deterministic notification ordering

**B4. max_turns enforcement across the re-entrant cycle**

Channel config: `max_turns=2` (set via `ChannelConfig(max_turns=2)` on the
channel object). Mock `side_effect` list has 3 entries — each returning a
tool call response. The third LLM call still happens (the agent calls the
LLM on the notification), but its tool calls are suppressed because
`turn_counter` (2) is no longer less than `max_turns` (2). Sequence:
1. User message → LLM returns tool call → `turn_counter` 0 < 2: dispatch (counter → 1)
2. Tool result notification → LLM returns tool call → `turn_counter` 1 < 2: dispatch (counter → 2)
3. Tool result notification → LLM returns tool call → `turn_counter` 2 < 2 is false: suppressed

Verify: fallback text "(max tool-calling rounds reached)" sent. Only 2 tasks were dispatched. LLM was called 3 times total.

**B5. User message interleaving during tool execution**

Use a gated tool (asyncio.Event blocks execution). Sequence:
1. User sends "msg1", LLM returns tool call, tool blocks on gate
2. User sends "msg2" while tool is blocked
3. Release gate

Expected behavior (because SerialQueue processes one item at a time, and tool dispatch returns the queue slot immediately):

- msg1 enters queue → LLM called (call 1) → returns tool call → task dispatched → queue slot freed
- msg2 enters queue → LLM called (call 2) → returns text response → sent to channel
- Gate releases → task completes → on_notify fires → notification enters queue → LLM called (call 3) → returns text response → sent to channel

Expected conversation log state:
```
user("msg1"), assistant(tool_call), user("msg2"), assistant("response to msg2"),
tool(result), assistant("response to tool result")
```

Verify:
- 3 total LLM calls
- 2 entries in FakeTransportPlugin.sent (response to msg2, response to tool result)
- Conversation log matches expected ordering above

### Group C: Conversation Persistence

*Depends on PersistencePlugin. These tests use `tmp_path / "test.db"` as DB path via the config fixture's `db_path` parameter.*

**C1. Messages survive stop/restart (file-based DB)**

First harness: `make_config(db_path=str(tmp_path / "test.db"))`. Send 3 messages (3 user + 3 assistant responses), drain, call `on_stop`. Create a second harness with a fresh plugin graph against the same DB file. Call `on_start`. Send a new message to the same channel. Verify: the LLM's `chat()` call includes the 6 prior messages in the conversation history (loaded by PersistencePlugin). Note: `build_prompt()` prepends the system message, so the full messages list passed to `chat()` is `[system] + [6 history] + [new user message]` = 8 entries.

**Important**: when using a `capturing_chat` function as `side_effect`, its
signature must match `client.chat()` call signature:
`async def capturing_chat(messages, tools=None, extra_body=None)`.

**C2. Compaction survives restart**

First harness: config with small `max_context_tokens` (e.g., 256) and file-based DB. Send enough messages to trigger compaction (CompactionPlugin handles this via the `compact_conversation` hook). **Important**: The test patches `CompactionPlugin._summarize` directly (rather than threading summarization through mock_client `side_effect`) to return a fixed summary string. This avoids the ordering complexity of injecting an extra LLM response into the side_effect list. Verify compaction occurred (summary row in DB). Call `on_stop`. Second harness against same DB. Verify: loaded conversation contains summary + retained messages, not the full pre-compaction history.

**C3. Tool call messages round-trip through DB**

First harness with file-based DB. Run a tool-call cycle (same as B2). Call `on_stop`. Second harness against same DB. Verify: the assistant message with `tool_calls` array and the tool result message are both present and structurally intact in the loaded conversation.

### Group D: Multi-Channel Isolation

**D1. Two channels, independent conversations**

Send messages to "test:chan_a" and "test:chan_b". Verify:
- Each has its own ConversationLog
- Each has its own SerialQueue in `agent.queues`
- Responses routed to correct channel (check FakeTransportPlugin.sent channel field)

**Important**: do not use `side_effect` with ordered responses for two
channels — which channel's LLM call fires first is non-deterministic. Use
`return_value` with a single response, or use a function `side_effect` that
inspects the messages argument to determine which channel is calling.

**D2. Per-channel config resolution**

Pre-register channels: "test:limited" with `max_turns=1`, "test:default" with no override. Send tool-call-triggering messages to both. Verify: limited channel hits max_turns after 1 round, default channel continues.

**D3. Concurrent compaction across channels**

Channel A has enough messages to trigger compaction. Channel B is mid-tool-call (tool dispatched, not yet returned). Trigger compaction on channel A. The test uses calibrated token thresholds: `max_context_tokens=140` with 56-character messages to reliably cross the compaction threshold. Verify:
- Channel A's compaction completes and summary is written
- Channel B's in-flight tool call completes and conversation is not corrupted
- Neither channel's ConversationLog references the other's data

### Group E: Error Handling

**E1. LLM error mid-conversation**

LLM raises `aiohttp.ClientResponseError` on chat(). Verify:
- Error message sent to channel
- Subsequent messages to same channel still work (conversation not corrupted)
- Conversation log does not contain a corrupt assistant message

**E2. Tool exception in integrated context**

Register a tool that raises RuntimeError. LLM calls it. Verify:
- Task completes with error string. Note: `make_work` in `AgentPlugin._dispatch_tool_calls` catches tool exceptions and returns `f"Error: tool '{fn_name}' failed"`. This is the string that reaches `_on_task_complete`, which wraps it as `f"[Task {task_id}] Error: tool '{fn_name}' failed"`. The `_run_one_worker` except clause does NOT fire because `make_work` already caught the exception.
- on_notify delivers error to agent
- LLM called again with error in tool result
- Conversation continues to final text response

**E3. Missing TaskPlugin degradation**

Assemble plugin graph without TaskPlugin. Send message triggering tool call. Verify:
- Error logged ("no TaskQueue available")
- No crash
- Channel processes subsequent text-only messages normally

**E4. MCP server disconnect mid-tool-call** *(placeholder — gated on McpClientPlugin landing)*

MCP server becomes unreachable while a tool call is in flight. Verify graceful error propagation back through the tool result path.

### Group F: Plugin Tool Collection

**F1. All expected tools registered**

After on_start with full plugin graph, verify `AgentPlugin.tools.keys()` == `{"shell", "read_file", "write_file", "web_fetch", "subagent", "task_status"}`. *(Update expected set when McpClientPlugin or other tool-providing plugins land.)*

**F2. Tool schemas structurally valid**

Each schema in `AgentPlugin.tool_schemas`:
- Has `type == "function"`
- Has `function.name` matching a key in `AgentPlugin.tools`
- Has `function.parameters` with `type == "object"`
- Does not contain `_ctx` in parameters.properties

## Priority Order

1. **IntegrationHarness fixture + FakeTransportPlugin** (infrastructure)
2. **Group A** (lifecycle) — validates the fixture itself works
3. **Group B** (round-trip) — highest-value gap coverage
4. **Group F** (tool collection) — low-effort, high-confidence
5. **Group E** (error handling) — important for reliability
6. **Group C** (persistence) — depends on PersistencePlugin
7. **Group D** (multi-channel) — completeness

## Design Validation Report

Validated 2026-04-25 against codebase at commit 9a5999c.

### Verified Correct

- All plugin class names and import paths
- Plugin registration order matches `main.py`
- All 14 hookspec signatures match AgentSpec
- `depends_on` declarations for all plugins
- Constructor signatures for all plugins
- Tool set: {shell, read_file, write_file, web_fetch, subagent, task_status} = 6
- Config shape: agent, llm.main, daemon.session_db, _base_dir
- LLM client patch target `sherman.agent.LLMClient`
- ConversationLog, SerialQueue, TaskQueue, ChannelRegistry APIs
- FakeTransportPlugin hookimpl compatibility with hookspecs
- drain_all pattern matches validated approach from test_agent_single_turn.py

### Discrepancies Amended

1. **B2 tool name collision**: `shell` collides with CoreToolsPlugin. Changed to `test_echo`.
2. **SubagentPlugin LLM patch gap**: Added note about separate `sherman.tools.subagent.LLMClient` import.
3. **C2 compaction LLM call**: Added note about mock_client needing summarization response.
4. **B4 mock side_effect**: Added explicit 3-entry specification with turn_counter trace.
5. **E2 task result wrapping**: Corrected — `make_work` catches tool exceptions, not `_run_one_worker`.
6. **C1 message count**: Clarified build_prompt() system message prepending.

### Recommendation

**YES** — design is ready to proceed with amendments applied.

## Design Review

Reviewed 2026-04-25 by independent reviewer (25 tool uses across codebase).

### Critical

1. **E2 tool exception path incorrect** (fixed): The original design claimed `TaskQueue._run_one_worker` catches tool exceptions. In fact, `make_work` in `AgentPlugin._dispatch_tool_calls` catches them first. **Amended in-place above.**

### Important

None.

### Cosmetic

1. **B2 latency_ms assertion**: should be `>= 0` rather than `> 0` to avoid flakiness with AsyncMock.
2. **B3 sent_messages count**: Design doesn't assert count (should be 3, one per continuation).
3. **B4 channel config mechanism**: Should use `ChannelConfig(max_turns=2)` on the channel object.

### Recommendation

**YES** — design is ready to proceed.

## Red TDD Review

Reviewed 2026-04-26. First implementation produced 20 tests (16 passed, 3 failed, 1 skipped). All issues were resolved in the final implementation.

### Confirmed Failure Root Causes

**drain_all insufficient yields** (critical): Single `asyncio.sleep(0)` after
`task_queue.queue.join()` is not enough. The worker calls `task_done()` before
`await on_complete(task, result)`. After `join()` unblocks, the worker needs 3
event loop cycles to complete: resume → `await on_complete(...)` → `on_notify`
→ enqueue notification. Fix: 3 `asyncio.sleep(0)` calls. This was the root
cause of B2 and B4 failures. **Fixed in drain_all code above.**

**B2 used drain_all instead of drain_until_stable** (important): A 2-turn
tool-call cycle needs the stability loop. **Fixed in B2 description above.**

**C1 capturing_chat missing tools kwarg** (critical): `agent_loop.py` calls
`client.chat(messages, tools=...)`. The `capturing_chat` function must accept
`tools=None, extra_body=None`. **Fixed in C1 description above.**

**D1 non-deterministic LLM call ordering** (important): Using `side_effect`
with ordered responses for two channels is flaky — which channel drains first
is non-deterministic. **Fixed in D1 description above.**

### Recommendation

**YES** — all issues resolved. Implementation is complete.

## Implementation Notes

- Baseline test count: 525 (all passing before integration tests added)
- Final test count: 544 (baseline 525 + 19 new integration tests)
- 19 integration tests implemented across Groups A–F; E4 skipped as placeholder (gated on McpClientPlugin MCP server disconnect support)
- pytest-timeout is installed; use `--timeout=30` on all test runs
- Add `pytestmark = pytest.mark.timeout(30)` at module level
- Do NOT use git stash or any destructive git commands in subagents
- When adding tools in B3/B4 tests, also add schemas to `agent.tool_schemas` for consistency
- `drain_until_stable` requires 2 consecutive stable iterations; stability defined as no new sent_messages and `task_queue.is_idle` (property, not method); includes 5 extra `asyncio.sleep(0)` yields per iteration for event loop settling; `max_iterations=20`
- D3 uses `max_context_tokens=140` with 56-character messages to calibrate compaction trigger reliably
- C2 patches `CompactionPlugin._summarize` directly rather than threading summarization through `mock_client.side_effect`
- `AgentPlugin.on_start` has no `@hookimpl` decorator — the harness calls it via `pm.ahook.on_start`, but the explicit call in `_build_harness` is the only invocation, matching the `main.py` pattern
