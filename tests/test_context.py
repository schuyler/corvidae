"""Tests for corvidae.context.ContextWindow, MessageType, and DEFAULT_CHARS_PER_TOKEN.

RED phase: all tests fail with ImportError until corvidae/context.py is created.
"""

import pytest

# This import fails until corvidae/context.py exists — that is the intended red state.
from corvidae.context import ContextWindow, MessageType, DEFAULT_CHARS_PER_TOKEN


class TestMessageType:
    def test_message_type_message_value(self):
        """MessageType.MESSAGE must equal the string 'message'."""
        assert MessageType.MESSAGE == "message"

    def test_message_type_summary_value(self):
        """MessageType.SUMMARY must equal the string 'summary'."""
        assert MessageType.SUMMARY == "summary"

    def test_message_type_context_value(self):
        """MessageType.CONTEXT must equal the string 'context'."""
        assert MessageType.CONTEXT == "context"

    def test_message_type_roundtrip_from_string(self):
        """MessageType must construct from its string value."""
        assert MessageType("message") is MessageType.MESSAGE
        assert MessageType("summary") is MessageType.SUMMARY
        assert MessageType("context") is MessageType.CONTEXT

    def test_message_type_is_str_subclass(self):
        """MessageType members must compare equal to plain strings."""
        assert MessageType.MESSAGE == "message"
        assert MessageType.SUMMARY == "summary"
        assert MessageType.CONTEXT == "context"


class TestDefaultCharsPerToken:
    def test_default_chars_per_token_value(self):
        """DEFAULT_CHARS_PER_TOKEN must equal 3.5."""
        assert DEFAULT_CHARS_PER_TOKEN == 3.5

    def test_default_chars_per_token_type(self):
        """DEFAULT_CHARS_PER_TOKEN must be a float."""
        assert isinstance(DEFAULT_CHARS_PER_TOKEN, float)


class TestContextWindowInit:
    def test_init_sets_channel_id(self):
        """ContextWindow.__init__ must store channel_id."""
        cw = ContextWindow("test:scope")
        assert cw.channel_id == "test:scope"

    def test_init_messages_empty_list(self):
        """ContextWindow.__init__ must set messages to an empty list."""
        cw = ContextWindow("test:scope")
        assert cw.messages == []
        assert isinstance(cw.messages, list)

    def test_init_system_prompt_empty_string(self):
        """ContextWindow.__init__ must set system_prompt to an empty string."""
        cw = ContextWindow("test:scope")
        assert cw.system_prompt == ""

    def test_init_default_chars_per_token(self):
        """ContextWindow.__init__ must use DEFAULT_CHARS_PER_TOKEN by default."""
        cw = ContextWindow("test:scope")
        assert cw.chars_per_token == DEFAULT_CHARS_PER_TOKEN

    def test_init_custom_chars_per_token(self):
        """ContextWindow.__init__ must accept a custom chars_per_token."""
        cw = ContextWindow("test:scope", chars_per_token=4.0)
        assert cw.chars_per_token == 4.0


class TestContextWindowAppend:
    def test_append_is_synchronous(self):
        """append() must be a regular method, not a coroutine."""
        import inspect
        cw = ContextWindow("test:scope")
        result = cw.append({"role": "user", "content": "hello"})
        # If it were async, calling it without await would return a coroutine.
        # Verify it is not a coroutine object.
        assert not inspect.iscoroutine(result), (
            "append() must be synchronous — it returned a coroutine"
        )

    def test_append_adds_message_to_messages(self):
        """append() must add the message to self.messages."""
        cw = ContextWindow("test:scope")
        msg = {"role": "user", "content": "hello"}
        cw.append(msg)
        assert len(cw.messages) == 1

    def test_append_tags_message_type_default(self):
        """append() with no explicit message_type must tag _message_type=MESSAGE."""
        cw = ContextWindow("test:scope")
        cw.append({"role": "user", "content": "hello"})
        assert cw.messages[0]["_message_type"] == MessageType.MESSAGE

    def test_append_tags_explicit_message_type(self):
        """append() with explicit message_type must use that type."""
        cw = ContextWindow("test:scope")
        cw.append({"role": "system", "content": "ctx"}, message_type=MessageType.CONTEXT)
        assert cw.messages[0]["_message_type"] == MessageType.CONTEXT

    def test_append_does_not_mutate_caller_dict(self):
        """append() must not add _message_type to the caller's original dict."""
        cw = ContextWindow("test:scope")
        original = {"role": "user", "content": "hello"}
        cw.append(original)
        assert "_message_type" not in original, (
            "append() must not mutate the caller's dict"
        )

    def test_append_multiple_messages(self):
        """append() called multiple times accumulates messages in order."""
        cw = ContextWindow("test:scope")
        cw.append({"role": "user", "content": "first"})
        cw.append({"role": "assistant", "content": "second"})
        assert len(cw.messages) == 2
        assert cw.messages[0]["content"] == "first"
        assert cw.messages[1]["content"] == "second"

    def test_append_summary_type(self):
        """append() with SUMMARY type tags correctly."""
        cw = ContextWindow("test:scope")
        cw.append({"role": "assistant", "content": "[Summary]"}, message_type=MessageType.SUMMARY)
        assert cw.messages[0]["_message_type"] == MessageType.SUMMARY


class TestContextWindowReplaceWithSummary:
    def test_replace_with_summary_is_synchronous(self):
        """replace_with_summary() must be a regular method, not a coroutine."""
        import inspect
        cw = ContextWindow("test:scope")
        cw.append({"role": "user", "content": "msg"})
        summary = {"role": "assistant", "content": "[Summary]"}
        result = cw.replace_with_summary(summary, retain_count=0)
        assert not inspect.iscoroutine(result), (
            "replace_with_summary() must be synchronous — it returned a coroutine"
        )

    def test_replace_with_summary_basic(self):
        """replace_with_summary(summary, 2) on 5 messages gives [summary_tagged, last2]."""
        cw = ContextWindow("test:scope")
        for i in range(5):
            cw.append({"role": "user", "content": f"msg {i}"})

        expected_retained = cw.messages[-2:]
        summary_msg = {"role": "assistant", "content": "[Summary]"}
        cw.replace_with_summary(summary_msg, retain_count=2)

        assert len(cw.messages) == 3
        assert cw.messages[0]["_message_type"] == MessageType.SUMMARY
        assert cw.messages[1:] == expected_retained

    def test_replace_with_summary_zero_retain(self):
        """replace_with_summary(summary, 0) leaves only the summary."""
        cw = ContextWindow("test:scope")
        for i in range(5):
            cw.append({"role": "user", "content": f"msg {i}"})

        summary_msg = {"role": "assistant", "content": "[Summary]"}
        cw.replace_with_summary(summary_msg, retain_count=0)

        assert len(cw.messages) == 1
        assert cw.messages[0]["_message_type"] == MessageType.SUMMARY

    def test_replace_with_summary_tags_summary(self):
        """replace_with_summary() must tag the summary with _message_type=SUMMARY."""
        cw = ContextWindow("test:scope")
        cw.append({"role": "user", "content": "msg"})
        summary_msg = {"role": "assistant", "content": "[Summary]"}
        cw.replace_with_summary(summary_msg, retain_count=0)

        assert cw.messages[0] == {**summary_msg, "_message_type": MessageType.SUMMARY}

    def test_replace_with_summary_does_not_mutate_summary_dict(self):
        """replace_with_summary() must not mutate the caller's summary_msg dict."""
        cw = ContextWindow("test:scope")
        cw.append({"role": "user", "content": "msg"})
        summary_msg = {"role": "assistant", "content": "[Summary]"}
        cw.replace_with_summary(summary_msg, retain_count=0)

        assert "_message_type" not in summary_msg, (
            "replace_with_summary() must not mutate the caller's summary_msg"
        )

    def test_replace_with_summary_raises_on_too_many_retained(self):
        """replace_with_summary(retain_count > len(messages)) must raise ValueError."""
        cw = ContextWindow("test:scope")
        for i in range(3):
            cw.append({"role": "user", "content": f"msg {i}"})

        summary_msg = {"role": "assistant", "content": "[Summary]"}
        with pytest.raises(ValueError):
            cw.replace_with_summary(summary_msg, retain_count=10)

    def test_replace_with_summary_in_memory_only(self):
        """replace_with_summary() must be purely in-memory — no DB or async side effects."""
        # This test verifies there is no DB argument on ContextWindow and that
        # calling replace_with_summary doesn't raise due to missing DB.
        cw = ContextWindow("test:scope")
        cw.append({"role": "user", "content": "msg"})
        # If this raises for any reason other than ValueError, the design contract is broken.
        cw.replace_with_summary({"role": "assistant", "content": "[Summary]"}, retain_count=0)
        assert len(cw.messages) == 1


class TestContextWindowBuildPrompt:
    def test_build_prompt_prepends_system_message(self):
        """build_prompt() must prepend a system message from system_prompt."""
        cw = ContextWindow("test:scope")
        cw.system_prompt = "You are a helpful assistant."
        prompt = cw.build_prompt()
        assert prompt[0] == {"role": "system", "content": "You are a helpful assistant."}

    def test_build_prompt_includes_messages(self):
        """build_prompt() must include all messages after the system message."""
        cw = ContextWindow("test:scope")
        cw.system_prompt = "sys"
        cw.append({"role": "user", "content": "hello"})
        cw.append({"role": "assistant", "content": "hi"})

        prompt = cw.build_prompt()

        assert len(prompt) == 3
        assert prompt[1]["content"] == "hello"
        assert prompt[2]["content"] == "hi"

    def test_build_prompt_strips_message_type(self):
        """build_prompt() must strip _message_type from every message dict."""
        cw = ContextWindow("test:scope")
        cw.system_prompt = "sys"
        cw.append({"role": "user", "content": "hello"})
        cw.append({"role": "system", "content": "ctx"}, message_type=MessageType.CONTEXT)

        prompt = cw.build_prompt()

        for i, msg in enumerate(prompt):
            assert "_message_type" not in msg, (
                f"Prompt message at index {i} still contains '_message_type': {msg!r}"
            )

    def test_build_prompt_does_not_modify_messages_in_place(self):
        """build_prompt() must not mutate self.messages."""
        cw = ContextWindow("test:scope")
        cw.system_prompt = "sys"
        cw.append({"role": "user", "content": "hello"})

        # Confirm _message_type present in self.messages before build_prompt
        assert "_message_type" in cw.messages[0]

        cw.build_prompt()

        # Still present after
        assert "_message_type" in cw.messages[0], (
            "build_prompt() must not mutate self.messages"
        )

    def test_build_prompt_empty_messages(self):
        """build_prompt() with no messages returns only the system message."""
        cw = ContextWindow("test:scope")
        cw.system_prompt = "sys"
        prompt = cw.build_prompt()
        assert prompt == [{"role": "system", "content": "sys"}]


class TestContextWindowTokenEstimate:
    def test_token_estimate_basic(self):
        """token_estimate() returns int(total_chars / chars_per_token)."""
        cw = ContextWindow("test:scope")
        cw.system_prompt = "You are helpful."  # 16 chars
        cw.messages = [
            {"role": "user", "content": "Hello there!", "_message_type": MessageType.MESSAGE},  # 12 chars
            {"role": "assistant", "content": "Hi!", "_message_type": MessageType.MESSAGE},      # 3 chars
        ]
        # total = 16 + 12 + 3 = 31 chars; int(31 / 3.5) = 8
        assert cw.token_estimate() == int(31 / 3.5)

    def test_token_estimate_empty(self):
        """token_estimate() on empty ContextWindow returns 0."""
        cw = ContextWindow("test:scope")
        assert cw.token_estimate() == 0

    def test_token_estimate_none_content(self):
        """token_estimate() treats None content as 0 chars."""
        cw = ContextWindow("test:scope")
        cw.messages = [
            {"role": "assistant", "content": None, "_message_type": MessageType.MESSAGE},
        ]
        assert cw.token_estimate() == 0

    def test_token_estimate_list_content(self):
        """token_estimate() treats non-string content as 0 chars."""
        cw = ContextWindow("test:scope")
        cw.messages = [
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}], "_message_type": MessageType.MESSAGE},
        ]
        assert cw.token_estimate() == 0

    def test_token_estimate_custom_chars_per_token(self):
        """token_estimate() uses self.chars_per_token."""
        cw = ContextWindow("test:scope", chars_per_token=4.0)
        cw.messages = [
            {"role": "user", "content": "aaaa", "_message_type": MessageType.MESSAGE},  # 4 chars
        ]
        assert cw.token_estimate() == 1  # int(4 / 4.0)


class TestContextWindowRemoveByType:
    def test_remove_by_type_removes_context(self):
        """remove_by_type(CONTEXT) removes CONTEXT entries from self.messages."""
        cw = ContextWindow("test:scope")
        cw.append({"role": "user", "content": "msg"})
        cw.append({"role": "system", "content": "ctx1"}, message_type=MessageType.CONTEXT)
        cw.append({"role": "system", "content": "ctx2"}, message_type=MessageType.CONTEXT)

        removed = cw.remove_by_type(MessageType.CONTEXT)

        assert removed == 2
        assert len(cw.messages) == 1
        assert cw.messages[0]["_message_type"] == MessageType.MESSAGE

    def test_remove_by_type_returns_count(self):
        """remove_by_type() returns the number of entries removed."""
        cw = ContextWindow("test:scope")
        cw.append({"role": "system", "content": "ctx"}, message_type=MessageType.CONTEXT)
        removed = cw.remove_by_type(MessageType.CONTEXT)
        assert removed == 1

    def test_remove_by_type_returns_zero_when_none_match(self):
        """remove_by_type() returns 0 when no entries of that type exist."""
        cw = ContextWindow("test:scope")
        cw.append({"role": "user", "content": "msg"})
        removed = cw.remove_by_type(MessageType.CONTEXT)
        assert removed == 0

    def test_remove_by_type_is_synchronous(self):
        """remove_by_type() must be a regular method, not a coroutine."""
        import inspect
        cw = ContextWindow("test:scope")
        result = cw.remove_by_type(MessageType.CONTEXT)
        assert not inspect.iscoroutine(result), (
            "remove_by_type() must be synchronous — it returned a coroutine"
        )

    def test_remove_by_type_rejects_message(self):
        """remove_by_type(MESSAGE) must raise ValueError."""
        cw = ContextWindow("test:scope")
        with pytest.raises(ValueError):
            cw.remove_by_type(MessageType.MESSAGE)

    def test_remove_by_type_rejects_summary(self):
        """remove_by_type(SUMMARY) must raise ValueError."""
        cw = ContextWindow("test:scope")
        with pytest.raises(ValueError):
            cw.remove_by_type(MessageType.SUMMARY)

    def test_remove_by_type_in_memory_only(self):
        """remove_by_type() only modifies self.messages, no DB required."""
        # ContextWindow has no DB — this verifies it doesn't fail on missing DB.
        cw = ContextWindow("test:scope")
        cw.append({"role": "system", "content": "ctx"}, message_type=MessageType.CONTEXT)
        cw.remove_by_type(MessageType.CONTEXT)
        assert cw.messages == []
