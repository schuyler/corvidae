"""Tests for Phase 2 hook dispatch migration.

These tests verify firstresult=True semantics on compact_conversation,
load_conversation, and on_llm_error hookspecs, including trylast/tryfirst
ordering and chain-stop behavior.
"""

from __future__ import annotations

import pytest
import apluggy as pluggy

from corvidae.hooks import create_plugin_manager, hookimpl


# ---------------------------------------------------------------------------
# Test 1: compact_conversation — trylast handler is fallback
# ---------------------------------------------------------------------------


async def test_compact_conversation_firstresult_trylast_fallback():
    """A trylast handler runs and its return value is returned directly (not a list).

    With firstresult=False (current), pm.ahook.compact_conversation returns a list.
    With firstresult=True (target), it returns the single non-None value.

    This test fails now because compact_conversation lacks firstresult=True,
    so pm.ahook.compact_conversation returns a list instead of a scalar True.
    """
    call_log: list[str] = []

    class FallbackPlugin:
        # trylast=True marks this as a fallback handler
        @hookimpl(trylast=True)
        async def compact_conversation(self, channel, conversation, max_tokens):
            call_log.append("fallback")
            return True

    pm = create_plugin_manager()
    pm.register(FallbackPlugin(), name="fallback")

    # Dummy channel and conversation objects — the hookimpl does not use them
    result = await pm.ahook.compact_conversation(
        channel=object(), conversation=object(), max_tokens=8000
    )

    # With firstresult=True: result is True (scalar), not [True] (list)
    assert result is True, (
        f"expected True (scalar firstresult), got {result!r} — "
        "compact_conversation hookspec needs firstresult=True"
    )
    assert call_log == ["fallback"]


# ---------------------------------------------------------------------------
# Test 2: compact_conversation — chain stops when default-priority handler wins
# ---------------------------------------------------------------------------


async def test_compact_conversation_firstresult_chain_stops_on_non_none():
    """A default-priority handler returning True stops the chain before trylast.

    With firstresult=True: the chain stops after the default-priority handler
    returns True. The trylast handler is never called.

    This test fails now because compact_conversation is a broadcast hook —
    all handlers run regardless of return value.
    """
    call_log: list[str] = []

    class ReplacementPlugin:
        # Default priority — runs before trylast
        @hookimpl
        async def compact_conversation(self, channel, conversation, max_tokens):
            call_log.append("replacement")
            return True

    class FallbackPlugin:
        @hookimpl(trylast=True)
        async def compact_conversation(self, channel, conversation, max_tokens):
            call_log.append("fallback")
            return True

    pm = create_plugin_manager()
    # Register fallback first; pluggy reverses registration order for execution
    pm.register(FallbackPlugin(), name="fallback")
    pm.register(ReplacementPlugin(), name="replacement")

    result = await pm.ahook.compact_conversation(
        channel=object(), conversation=object(), max_tokens=8000
    )

    # With firstresult=True: replacement returns True, chain stops, fallback never called
    assert result is True
    assert call_log == ["replacement"], (
        f"expected only ['replacement'], got {call_log!r} — "
        "compact_conversation needs firstresult=True to stop the chain"
    )


# ---------------------------------------------------------------------------
# Test 3: compact_conversation — tryfirst returning None does not stop chain
# ---------------------------------------------------------------------------


async def test_compact_conversation_tryfirst_returns_none_chain_continues():
    """A tryfirst handler returning None does not stop the chain.

    With firstresult=True: tryfirst runs first. If it returns None, the chain
    continues to the trylast handler.

    This test fails now because compact_conversation is a broadcast hook and
    the result is a list. Checking result is True will fail.
    """
    call_log: list[str] = []

    class ObserverPlugin:
        # tryfirst=True — runs before default-priority and trylast handlers
        @hookimpl(tryfirst=True)
        async def compact_conversation(self, channel, conversation, max_tokens):
            call_log.append("observer")
            return None  # Does not stop the chain

    class FallbackPlugin:
        @hookimpl(trylast=True)
        async def compact_conversation(self, channel, conversation, max_tokens):
            call_log.append("fallback")
            return True

    pm = create_plugin_manager()
    pm.register(FallbackPlugin(), name="fallback")
    pm.register(ObserverPlugin(), name="observer")

    result = await pm.ahook.compact_conversation(
        channel=object(), conversation=object(), max_tokens=8000
    )

    # Both handlers must have run; result is True from the fallback
    assert call_log == ["observer", "fallback"], (
        f"expected ['observer', 'fallback'], got {call_log!r} — "
        "tryfirst returning None should not stop the chain"
    )
    assert result is True, (
        f"expected True (from fallback), got {result!r} — "
        "compact_conversation needs firstresult=True"
    )


# ---------------------------------------------------------------------------
# Test 4: load_conversation — single handler returns the list directly
# ---------------------------------------------------------------------------


async def test_load_conversation_firstresult_returns_single_value():
    """A handler returning a list is returned directly by pm.ahook.load_conversation.

    With firstresult=False (current): pm.ahook.load_conversation returns
    [[{"role": "user", "content": "hi"}]] — a list wrapping the returned list.
    With firstresult=True (target): it returns [{"role": "user", "content": "hi"}] directly.

    This test fails now because load_conversation lacks firstresult=True.
    """
    history = [{"role": "user", "content": "hi"}]

    class StoragePlugin:
        @hookimpl
        async def load_conversation(self, channel):
            return history

    pm = create_plugin_manager()
    pm.register(StoragePlugin(), name="storage")

    result = await pm.ahook.load_conversation(channel=object())

    # With firstresult=True: result is the list itself, not wrapped in another list
    assert result == history, (
        f"expected {history!r} directly, got {result!r} — "
        "load_conversation hookspec needs firstresult=True"
    )


# ---------------------------------------------------------------------------
# Test 5: load_conversation — None fallthrough when no handler returns
# ---------------------------------------------------------------------------


async def test_load_conversation_firstresult_none_when_no_handlers_return():
    """When the only handler returns None, pm.ahook.load_conversation returns None.

    With firstresult=False (current): returns [] (empty list after filtering Nones).
    With firstresult=True (target): returns None directly.

    This test fails now because load_conversation lacks firstresult=True.
    """

    class NoDataPlugin:
        @hookimpl
        async def load_conversation(self, channel):
            return None

    pm = create_plugin_manager()
    pm.register(NoDataPlugin(), name="no_data")

    result = await pm.ahook.load_conversation(channel=object())

    # With firstresult=True and all-None returns: result is None, not []
    assert result is None, (
        f"expected None, got {result!r} — "
        "load_conversation with firstresult=True and all-None returns should be None"
    )


# ---------------------------------------------------------------------------
# Test 6: on_llm_error — first non-None return value wins
# ---------------------------------------------------------------------------


async def test_on_llm_error_firstresult_first_non_none_wins():
    """First handler returning a non-None string wins; second handler is never called.

    With firstresult=False (current): both handlers run and the call site uses
    resolve_hook_results(VALUE_FIRST) to pick a winner.
    With firstresult=True (target): the chain stops after the first non-None result.

    This test fails now because on_llm_error lacks firstresult=True.
    """
    call_log: list[str] = []

    class FirstPlugin:
        @hookimpl
        async def on_llm_error(self, channel, error):
            call_log.append("first")
            return "custom error"

    class SecondPlugin:
        @hookimpl
        async def on_llm_error(self, channel, error):
            call_log.append("second")
            return "other error"

    pm = create_plugin_manager()
    # Register second first; pluggy reverses registration order
    pm.register(SecondPlugin(), name="second")
    pm.register(FirstPlugin(), name="first")

    exc = RuntimeError("boom")
    result = await pm.ahook.on_llm_error(channel=object(), error=exc)

    # With firstresult=True: "first" runs first (registered last), returns "custom error",
    # chain stops, "second" is never called.
    assert result == "custom error", (
        f"expected 'custom error', got {result!r} — "
        "on_llm_error needs firstresult=True"
    )
    assert call_log == ["first"], (
        f"expected only ['first'], got {call_log!r} — "
        "second handler should not be called after first returns non-None"
    )


# ---------------------------------------------------------------------------
# Test 7: on_llm_error — None fallthrough when no handler returns
# ---------------------------------------------------------------------------


async def test_on_llm_error_firstresult_none_fallthrough():
    """When the only handler returns None, pm.ahook.on_llm_error returns None.

    With firstresult=False (current): returns [] (empty list).
    With firstresult=True (target): returns None directly.

    This test fails now because on_llm_error lacks firstresult=True.
    """

    class PassthroughPlugin:
        @hookimpl
        async def on_llm_error(self, channel, error):
            return None

    pm = create_plugin_manager()
    pm.register(PassthroughPlugin(), name="passthrough")

    exc = RuntimeError("boom")
    result = await pm.ahook.on_llm_error(channel=object(), error=exc)

    # With firstresult=True and all-None returns: result is None, not []
    assert result is None, (
        f"expected None, got {result!r} — "
        "on_llm_error with firstresult=True and all-None returns should be None"
    )
