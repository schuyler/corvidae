# Phase 1 Implementation Plan — Design Review

**Reviewer:** Chico
**Date:** 2026-04-22

---

## Summary

The plan covers all six Phase 1 modules (hooks, plugin_manager, llm, agent_loop, conversation, main) plus tests. The documented deviations from design.md are appropriate and well-reasoned. Three issues would cause tests to fail immediately on first run; several others would cause confusion or wasted debugging time during implementation.

---

## Requirements Verification

| Phase 1 requirement (design.md lines 1192–1199) | Status |
|---|---|
| Hook definitions (`hooks.py`) | Covered — Module 2 |
| Plugin manager setup (`plugin_manager.py`) | Covered — Module 3 |
| LLM client (`llm.py`) | Covered — Module 4 |
| Agent loop (`agent_loop.py`) — tool dispatch, schema generation | Covered — Module 5 |
| Conversation log (`conversation.py`) — append-only log, compaction | Covered — Module 6 |
| Daemon entry point (`main.py`) — starts, waits for signal, stops | Covered — Module 7 |
| Tests for hooks, agent loop (mocked HTTP), conversation log | Covered — Tests section |

---

## Critical Issues

### C1: pytest-asyncio strict mode — all async tests will error without explicit configuration

No `asyncio_mode` configuration is specified. The installed pytest-asyncio 1.3.0 defaults to `"strict"` mode, which requires explicit `@pytest.mark.asyncio` on every async test and `@pytest_asyncio.fixture` on every async fixture. Without configuration, the entire test suite fails on first run.

**Fix:** Add to `pyproject.toml`:
```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```
Note in the plan that async fixtures must use `@pytest_asyncio.fixture`.

### C2: `test_main_calls_on_start_and_on_stop` is untestable as described

`main()` creates its own PM instance internally. The test cannot register a mock plugin before `on_start` fires. Requires patching `sherman.main.create_plugin_manager` to return a pre-populated PM.

**Fix:** Specify the patching strategy explicitly.

### C3: `conftest.py` db fixture is underspecified

`init_db(db)` takes a connection argument. The one-line description omits this. An implementer would write `await init_db()` and get a TypeError.

**Fix:** Show the fixture body:
```python
@pytest_asyncio.fixture
async def db():
    async with aiosqlite.connect(":memory:") as conn:
        await init_db(conn)
        yield conn
```

---

## Important Concerns

### I1: Pydantic v2 `model_json_schema()` injects `title` fields

`tool_to_schema` should strip `title` fields from the generated schema, or the plan should note that llama-server accepts them.

### I2: Incorrect dependency claim — `agent_loop.py` does not depend on `hooks.py`

Phase 1 `agent_loop.py` has no `hookimpl` usage. The dependency graph should say `agent_loop.py -> llm.py, pydantic`.

### I3: Private helpers `_persist` and `_summarize` are only implied

Add a "Private methods" block to the `ConversationLog` spec listing these explicitly.

### I4: `test_compact_if_needed_triggers` doesn't specify the LLMClient mock shape

The mock must return `{"choices": [{"message": {"content": "mock summary"}}]}`. Specify this in the test description.

### I5: `test_main_missing_config_raises` implicitly requires async handling

Note that the test body must be `async def` with `await main(...)` inside `pytest.raises`.

---

## Minor Suggestions

- M1: Task numbers in parallelization section reference a non-existent task list. Use module names only.
- M2: `[project.scripts]` entry point is buried in `main.py` section, not in the dependencies section.
- M3: `"sherman"` marker name change from `"agent"` deserves a comment in `hooks.py`.
- M4: `strip_thinking` import move from function-level to module-level is not listed as a deviation.

---

## Overall Assessment

**Needs minor revision.** Fix C1, C2, C3 before handing to an implementer. Address I1–I5 to prevent debugging time. Minor suggestions are optional.
