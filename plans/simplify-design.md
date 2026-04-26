# Simplification Design — Items 1-6

## Item 1: Remove `_complete_wrapper` in `corvidae/task.py`

### File
`corvidae/task.py`

### Current code (lines 216-221)
```python
async def _complete_wrapper(task: Task, result: str) -> None:
    return await self._on_task_complete(task, result)

self._worker_task = asyncio.create_task(
    self.task_queue.run_worker(_complete_wrapper)
)
```

### Change
Replace the 4 lines above with:
```python
self._worker_task = asyncio.create_task(
    self.task_queue.run_worker(self._on_task_complete)
)
```

### Rationale
`_complete_wrapper` is a closure that delegates to `self._on_task_complete` with identical arguments. `self._on_task_complete` is a bound method with signature `(task: Task, result: str) -> None`, which matches `Callable[[Task, str], Awaitable[None]]` required by `run_worker`.

### Risks
None. The bound method reference captures `self` at binding time, same as the closure did.

### Test impact
- No new tests needed.
- Existing tests in `test_task.py` (`TestTaskPlugin`) already exercise `on_start` and `_on_task_complete`. They should pass unchanged.

---

## Item 2: Collapse `web_fetch` into `web_fetch_with_session`

### File
`corvidae/tools/web.py`

### Current code
Two functions with identical response-handling logic (lines 14-33 and 36-54). `web_fetch` creates an ephemeral session; `web_fetch_with_session` uses a provided session.

### Change
Replace the body of `web_fetch` (lines 14-33) with delegation:
```python
async def web_fetch(
    url: str,
    max_response_bytes: int = 50_000,
    timeout: int = 15,
) -> str:
    """Fetch a URL and return its text content."""
    async with aiohttp.ClientSession() as session:
        return await web_fetch_with_session(
            session, url,
            max_response_bytes=max_response_bytes,
            timeout=timeout,
        )
```

Keep `web_fetch_with_session` unchanged -- it is the canonical implementation.

### Rationale
Removes ~12 lines of duplicated logic. The timeout behavior difference (session-level vs per-request) is not observable for a single-request ephemeral session.

### Risks
- The `aiohttp.ClientSession()` is now created without a session-level timeout, unlike before. For the single-request case this is harmless since `web_fetch_with_session` applies per-request timeout via `session.get(url, timeout=...)`.
- If `web_fetch_with_session` raises an exception not caught by its own try/except (e.g. `BaseException`), the `async with` context manager for the session still cleans up. Same as before.

### Test impact
- **Existing tests that need updating**: `tests/test_tools.py` `TestWebFetch` tests mock `aiohttp.ClientSession` and expect specific interaction patterns (session-level timeout in constructor). After this change, `web_fetch` creates a plain `ClientSession()` then calls `web_fetch_with_session` which applies per-request timeout via `session.get()`. The mocks need adjustment.
- **`tests/test_parameterized_config.py`**: Tests inspect `web_fetch` signature and call it with `max_response_bytes` override. The signature is unchanged. `test_web_fetch_max_response_override` patches `aiohttp.ClientSession` and will need the same mock adjustment as above.
- **No new tests needed.** Behavioral equivalence is verified by existing tests once mocks are adjusted.

---

## Item 3: Extract row-mapping helper in `ConversationLog.load`

### File
`corvidae/conversation.py`

### Current code (lines 92-96 and 106-110)
Both branches contain:
```python
loaded = []
for row in rows:
    msg = json.loads(row[0])
    msg["_message_type"] = MessageType(row[1])
    loaded.append(msg)
```

### Change
Add a module-level helper:
```python
def _parse_message_rows(rows: list[tuple]) -> list[dict]:
    """Parse (message_json, message_type) rows into tagged message dicts."""
    result = []
    for row in rows:
        msg = json.loads(row[0])
        msg["_message_type"] = MessageType(row[1])
        result.append(msg)
    return result
```

Place it as a module-level function before the `ConversationLog` class (after `MessageType`, around line 40). Then replace both loop blocks in `load()`:

**Summary branch:**
```python
loaded = _parse_message_rows(rows)
self.messages = [summary_msg] + loaded
```

**Non-summary branch:**
```python
self.messages = _parse_message_rows(rows)
```

### Risks
None. Pure extraction of identical logic.

### Test impact
- Existing `tests/test_conversation.py` tests cover `load()` behavior. They should pass unchanged.
- **New test**: Add a unit test for `_parse_message_rows` directly to confirm it parses rows correctly and tags `_message_type`. Place in `tests/test_conversation.py`.

---

## Item 4: Move `_reset_corvidae_logger` to `conftest.py`

### Files affected
- `tests/conftest.py` (add fixture)
- `tests/test_task.py` (remove local copy)
- `tests/test_persistence.py` (remove local copy)
- `tests/test_mcp_client.py` (remove local copy)

### Change
Add to `conftest.py`:
```python
@pytest.fixture(autouse=True)
def _reset_corvidae_logger():
    """Ensure the corvidae logger propagates to root so caplog captures records.

    Other test modules (test_logging.py) may apply dictConfig with
    propagate=False on the corvidae logger. This fixture resets it.
    """
    corvidae_logger = logging.getLogger("corvidae")
    original_propagate = corvidae_logger.propagate
    original_handlers = corvidae_logger.handlers[:]
    corvidae_logger.propagate = True
    yield
    corvidae_logger.propagate = original_propagate
    corvidae_logger.handlers = original_handlers
```

Add `import logging` to conftest.py imports.

Remove the identical fixture definition from all three test files listed above.

### Risks
- `autouse=True` in conftest.py applies to ALL tests. Safe because the fixture restores original state on teardown.

### Test impact
- No new tests needed.
- All existing tests should pass unchanged.

---

## Item 5: Consolidate `_build_plugin_and_channel` into `conftest.py`

### Files affected
- `tests/conftest.py` (add helper + drain helpers)
- `tests/test_hook_safety.py` (remove local copy, use conftest)
- `tests/test_agent_single_turn.py` (remove local copy, use conftest)
- `tests/test_agent_loop_plugin.py` (remove local copy, use conftest)

### Variant analysis

| File | Mocks `send_message`? | Mocks `on_agent_response`? |
|---|---|---|
| `test_hook_safety.py` | NO | NO |
| `test_agent_single_turn.py` | YES (`AsyncMock()`) | YES (`AsyncMock()`) |
| `test_agent_loop_plugin.py` | YES (`AsyncMock()`) | YES (`AsyncMock()`) |

### Change
Add to `conftest.py`:

```python
async def build_plugin_and_channel(
    agent_defaults=None,
    channel_config=None,
    *,
    mock_send_message: bool = True,
    mock_on_agent_response: bool = True,
):
    """Assemble plugin graph with in-memory DB.

    Returns (plugin, channel, db). Callers are responsible for teardown:
    task_plugin.on_stop() and db.close().

    Args:
        agent_defaults: Override AGENT_DEFAULTS for channel registry.
        channel_config: Override default ChannelConfig.
        mock_send_message: If True, replace send_message hook with AsyncMock.
        mock_on_agent_response: If True, replace on_agent_response hook with AsyncMock.
    """
    from unittest.mock import AsyncMock

    import aiosqlite

    from corvidae.agent import AgentPlugin
    from corvidae.channel import ChannelConfig, ChannelRegistry
    from corvidae.conversation import init_db
    from corvidae.hooks import create_plugin_manager
    from corvidae.persistence import PersistencePlugin
    from corvidae.task import TaskPlugin
    from corvidae.thinking import ThinkingPlugin

    _AGENT_DEFAULTS = {
        "system_prompt": "You are a test assistant.",
        "max_context_tokens": 8000,
        "keep_thinking_in_history": False,
    }

    if agent_defaults is None:
        agent_defaults = _AGENT_DEFAULTS

    db = await aiosqlite.connect(":memory:")
    await init_db(db)

    pm = create_plugin_manager()
    registry = ChannelRegistry(agent_defaults)
    pm.register(registry, name="registry")

    if mock_send_message:
        pm.ahook.send_message = AsyncMock()
    if mock_on_agent_response:
        pm.ahook.on_agent_response = AsyncMock()

    task_plugin = TaskPlugin(pm)
    pm.register(task_plugin, name="task")
    await task_plugin.on_start(config={})

    persistence = PersistencePlugin(pm)
    persistence.db = db
    persistence._registry = registry
    pm.register(persistence, name="persistence")

    thinking_plugin = ThinkingPlugin(pm)
    pm.register(thinking_plugin, name="thinking")

    plugin = AgentPlugin(pm)
    pm.register(plugin, name="agent_loop")
    plugin._registry = registry

    channel = registry.get_or_create(
        "test",
        "scope1",
        config=channel_config or ChannelConfig(),
    )

    return plugin, channel, db


async def drain(plugin, channel):
    """Drain the channel's SerialQueue."""
    if channel.id in plugin.queues:
        await plugin.queues[channel.id].drain()


async def drain_task_queue(plugin):
    """Wait for all TaskQueue tasks to complete including on_complete callbacks."""
    import asyncio
    task_plugin = plugin.pm.get_plugin("task")
    if task_plugin and task_plugin.task_queue:
        await task_plugin.task_queue.queue.join()
        await asyncio.sleep(0)
```

Update each test file to import and use these helpers with appropriate kwargs.

### Risks
- Plain functions (not fixtures) require explicit import. Intentional -- they take arguments.
- `AGENT_DEFAULTS` is embedded in the helper. Test files that define it for other purposes keep their own copy.

### Test impact
- No new tests needed. All existing tests should pass after import/call-site changes.

---

## Item 6: Consolidate `_chars_per_token` constant

### Files affected
- `corvidae/conversation.py` (add constant)
- `corvidae/compaction.py` (import constant)
- `corvidae/persistence.py` (import constant)

### Change

In `conversation.py`, add a module-level constant after the `MessageType` class:
```python
#: Default characters-per-token estimate for rough token counting.
#: Used by ConversationLog, CompactionPlugin, and PersistencePlugin.
DEFAULT_CHARS_PER_TOKEN: float = 3.5
```

Update `ConversationLog.__init__` default to use the constant.

In `compaction.py` and `persistence.py`:
- Import `DEFAULT_CHARS_PER_TOKEN` from `corvidae.conversation`
- Replace hardcoded `3.5` with the constant
- Remove "keep in sync" comments

### Risks
- No circular import: `compaction.py` already imports from `conversation.py` lazily; adding top-level import is safe since `conversation.py` does not import from `compaction.py`.
- `persistence.py` already imports from `conversation.py` at module level.

### Test impact
- No new tests strictly needed. Optional: add a test verifying `DEFAULT_CHARS_PER_TOKEN == 3.5`.
- Existing `tests/test_parameterized_config.py` tests pass unchanged.

---

## Implementation order

1. Item 6 first (constant extraction) -- no behavioral change, establishes the shared constant.
2. Item 3 (row-mapping helper) -- no behavioral change, local to conversation.py.
3. Item 1 (remove wrapper) -- 3-line change in task.py.
4. Item 2 (collapse web_fetch) -- requires test mock adjustments.
5. Item 4 (logger fixture to conftest) -- test infrastructure change, no source changes.
6. Item 5 (build_plugin_and_channel to conftest) -- largest test refactor, depends on item 4 being done first.

## Design Review Report

**Reviewer**: Chico
**Date**: 2026-04-26
**Scope**: Design verification for Items 1-6

### Exclusion Verification

`call_firstresult_hook` does not appear anywhere in this design document. Confirmed excluded from scope.

### Item 1: Remove `_complete_wrapper` — **Accurate.** No issues.

### Item 2: Collapse `web_fetch` — **Accurate.** Cosmetic: "~12 lines" should be "~15 lines."

### Item 3: Extract row-mapping helper — **Accurate.** No issues.

### Item 4: Move `_reset_corvidae_logger` — **Accurate. Important divergence from requirements doc.** The requirements doc names `test_notify_depth.py`; the design correctly identifies `test_mcp_client.py` instead. No such file as `test_notify_depth.py` exists. Implementation should follow the design.

### Item 5: Consolidate `_build_plugin_and_channel` — **Accurate.** Cosmetic: `drain_task_queue` uses local `import asyncio`; move to module-level.

### Item 6: Consolidate `_chars_per_token` — **Accurate.** Cosmetic: each of `compaction.py` and `persistence.py` has two occurrences of `3.5` (`__init__` and `on_start`). Implementer must update both.

### Overall Assessment

**No critical issues.** One important finding (Item 4/5 file list diverges from stale requirements doc — design is correct). Three cosmetic findings.

**Recommendation: PROCEED (YES)**

## Red TDD Review Report

**Reviewer**: Chico
**Date**: 2026-04-26

### Summary
6 failing tests added to `tests/test_conversation.py`: 4 for `_parse_message_rows` (Item 3), 2 for `DEFAULT_CHARS_PER_TOKEN` (Item 6). Items 1, 2, 4, 5 correctly skipped.

### Findings
- **No critical issues.**
- **No important issues.**
- **Cosmetic**: No test verifies `DEFAULT_CHARS_PER_TOKEN` is wired into `ConversationLog.__init__` as the default parameter. Existing behavioral tests cover this indirectly.
- **Cosmetic**: `test_parse_message_rows_basic` re-imports `json` locally (already at module level). Harmless.

### Verdict
All 6 tests fail with `ImportError` (correct red state). Skip decisions for Items 1, 2, 4, 5 are justified.

**Recommendation: PROCEED (YES)**

## Green TDD Review Report

**Reviewer**: Chico
**Date**: 2026-04-26

### Requirements Verification
- **Item 1**: Fully implemented. `_complete_wrapper` removed, bound method passed directly.
- **Item 2**: Fully implemented. `web_fetch` delegates to `web_fetch_with_session`.
- **Item 3**: Fully implemented. `_parse_message_rows` extracted, both branches use it.
- **Item 4**: Fully implemented. Fixture moved to conftest.py, removed from 3 files.
- **Item 5**: Implemented with deviation — helpers in `tests/helpers.py` instead of conftest.py. `pythonpath = ["tests"]` added to pyproject.toml.
- **Item 6**: Fully implemented. Constant defined, both `__init__` and `on_start` updated in both files. "Keep in sync" comments removed.

### Critical Issues: None.

### Important Concerns
**Item 5 — `tests/__init__.py` + `pythonpath = ["tests"]`**: `tests/` has `__init__.py` (package) while `pythonpath` adds it to `sys.path`. Module could load twice under different `sys.modules` keys. Pre-existing structural issue, not a regression. 592 tests pass. Two clean resolutions: remove `tests/__init__.py` or use `from tests.helpers import ...` and drop `pythonpath`.

### Cosmetic: `drain_task_queue` has redundant local `import asyncio`.

### Confirmed: `call_firstresult_hook` not modified. No bare `except Exception: pass`.

**Recommendation: PROCEED (YES)**

## Documentation Review Report

**Reviewer**: Harpo
**Date**: 2026-04-26

### Files reviewed

- `corvidae/task.py` — Item 1
- `corvidae/tools/web.py` — Item 2
- `corvidae/conversation.py` — Items 3, 6
- `tests/conftest.py` — Item 4
- `tests/helpers.py` — Item 5
- `corvidae/compaction.py`, `corvidae/persistence.py` — Item 6 consumers

### Findings

**Item 1 — task.py**: No documentation changes needed. `_on_task_complete` docstring is accurate. `run_worker` docstring covers the callback signature.

**Item 2 — web.py**: Both `web_fetch` and `web_fetch_with_session` had minimal one-line docstrings that did not describe parameters, return values, or error behavior. Updated both with full Args/Returns docstrings. `web_fetch` now notes it creates an ephemeral session and delegates to `web_fetch_with_session`. `web_fetch_with_session` now documents all return cases (200, non-200, timeout, ClientError) and the TRUNCATION_INDICATOR behavior.

**Item 3 — conversation.py**: `_parse_message_rows` has the docstring from the design. Adequate.

**Item 4 — conftest.py**: `_reset_corvidae_logger` docstring explains why it resets propagation and handlers. Adequate.

**Item 5 — tests/helpers.py**: Module docstring and all three function docstrings (`build_plugin_and_channel`, `drain`, `drain_task_queue`) are present and accurate. Adequate.

**Item 6 — conversation.py**: `DEFAULT_CHARS_PER_TOKEN` has the `#:` Sphinx-style comment naming its consumers. `compaction.py` and `persistence.py` use the constant correctly with no stale "keep in sync" comments. Adequate.

**CHANGELOG**: No CHANGELOG file exists in this project. No entry needed.

### Changes made

- `corvidae/tools/web.py`: Expanded docstrings for `web_fetch` and `web_fetch_with_session`.

**Recommendation: YES**
