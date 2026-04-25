# Plan: Unify Tool Dispatch + Eliminate run_agent_loop Duplication

## Context

Tool dispatch logic (_ctx injection, signature inspection, invocation) is duplicated in two places:
- `agent.py:AgentPlugin._dispatch_tool_calls()` (lines 282-307) — main agent path
- `agent_loop.py:run_agent_loop()` (lines 196-229) — subagent path

Additionally, `run_agent_loop()` duplicates the LLM-call → parse → log → append logic (lines 146-170) that `run_agent_turn()` already provides (lines 76-100).

## Changes

### 1. New function: `execute_tool_call()` in `tool.py`

```python
async def execute_tool_call(
    tool_fn: Callable,
    args: dict,
    *,
    channel: Channel | None = None,
    tool_call_id: str,
    task_queue: TaskQueue | None = None,
) -> str:
```

Handles _ctx injection and invocation only. Does **not** catch exceptions — the two callers have different error messages and logging, so each wraps the call in its own try/except.

Implementation:
1. Inspect signature of tool_fn for `_ctx` parameter
2. Build call_kwargs from args
3. If `_ctx` in signature, inject `ToolContext(channel=channel, tool_call_id=tool_call_id, task_queue=task_queue)`
4. Await tool_fn(**call_kwargs)
5. Return str(result)

### 2. Update `_dispatch_tool_calls()` in `agent.py`

Replace inline signature inspection + _ctx injection (lines 288-302) with a call to `execute_tool_call()`. Remove `import inspect`.

The try/except stays in agent.py — it has agent-specific error messages and logging.

### 3. Refactor `run_agent_loop()` in `agent_loop.py`

Two changes:
- Replace the inline LLM call block (lines 146-170) with `result = await run_agent_turn(client, messages, tool_schemas)`. Check `result.tool_calls` instead of `msg.get("tool_calls")`.
- Replace inline _ctx injection (lines 196-212) with `execute_tool_call()`. Remove `import inspect`.

The try/except stays in agent_loop.py — it has loop-specific error messages and logging.

Minor behavioral change: `run_agent_turn` shallow-copies messages before passing to `client.chat()`. Current `run_agent_loop` does not. This is strictly safer.

## Files modified

| File | Change |
|------|--------|
| `sherman/tool.py` | Add `execute_tool_call()` |
| `sherman/agent_loop.py` | Use `run_agent_turn` + `execute_tool_call`, remove `inspect` import |
| `sherman/agent.py` | Use `execute_tool_call` in `_dispatch_tool_calls`, remove `inspect` import |

## Verification

1. `uv run pytest tests/ -v` — all tests must pass
2. Key test files to watch:
   - `tests/test_agent_loop.py` — run_agent_loop behavior
   - `tests/test_tool_context.py` — _ctx injection
   - `tests/test_agent_single_turn.py` — AgentPlugin round-trip
   - `tests/test_subagent.py` — SubagentPlugin (calls run_agent_loop)
3. Verify `inspect` is no longer imported in `agent.py` or `agent_loop.py`
4. Verify `execute_tool_call` is importable from `sherman.tool`

---

## Implementation report — agent.py changes

Done. Three changes applied to `sherman/agent.py`:

1. Removed `import inspect` (line 30).
2. Added `execute_tool_call` to the `from sherman.tool import ...` line.
3. Replaced the inline signature-inspection + _ctx injection block in `make_work` with a single `await execute_tool_call(tool_fn, args, channel=channel, tool_call_id=call_id, task_queue=task_queue)` call. The surrounding try/except is unchanged.

**Proceed: yes**
