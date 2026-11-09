"""Tests for the retrieval query's conversational context (#665, variant A).

Passive retrieval's query is the agent's previous reply followed by the
unanswered user message(s), so a deictic follow-up ("what about the other
one?") retrieves records about the turn it refers to.
``memory.retrieval.query_context: none`` restores the user-message-only query.
"""

import logging

from corvidae.context import MessageType
from corvidae_cognition.memory import (
    QUERY_CONTEXT_CHOICES,
    retrieval_query,
)

from .test_memory_retrieval import (
    EMBED_CONFIG,
    build_retrieval_env,
    context_messages,
    seed_memory,
)


def user(text: str) -> dict:
    """A plain user MESSAGE row."""
    return {"role": "user", "content": text}


def assistant(text, **extra) -> dict:
    """An assistant row; extra keys (tool_calls, _message_type) pass through."""
    return {"role": "assistant", "content": text, **extra}


class TestRetrievalQuery:
    """The pure query builder shared by the daemon and the eval."""

    def test_choices_default_to_previous_reply(self):
        assert QUERY_CONTEXT_CHOICES[0] == "previous_reply"
        assert "none" in QUERY_CONTEXT_CHOICES

    def test_previous_reply_precedes_the_user_message(self):
        messages = [user("which board?"), assistant("The ESP32 on the porch."),
                    user("thanks, that worked")]
        assert retrieval_query(messages, "previous_reply") == (
            "The ESP32 on the porch.\n\nthanks, that worked"
        )

    def test_mode_none_is_the_user_messages_only(self):
        messages = [user("which board?"), assistant("The ESP32 on the porch."),
                    user("thanks, that worked")]
        assert retrieval_query(messages, "none") == "thanks, that worked"

    def test_no_assistant_row_gives_the_user_messages_alone(self):
        assert retrieval_query([user("hello there")], "previous_reply") == "hello there"

    def test_burst_keeps_every_unanswered_message_after_the_reply(self):
        messages = [assistant("Done."), user("one"), user("two")]
        assert retrieval_query(messages, "previous_reply") == "Done.\n\none\n\ntwo"

    def test_tool_call_rows_without_text_are_skipped(self):
        """A tool-call-only assistant row has no text to refer back to; the
        last assistant row with text is the previous reply."""
        messages = [
            user("check the sensor"),
            assistant("The greenhouse sensor reads high."),
            user("and now?"),
            assistant(None, tool_calls=[{"id": "c1", "type": "function"}]),
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
            assistant(""),
            user("thanks"),
        ]
        # unanswered_user_messages counts from the last assistant row of any
        # kind, so the query's user part is "thanks" alone.
        assert retrieval_query(messages, "previous_reply") == (
            "The greenhouse sensor reads high.\n\nthanks"
        )

    def test_unsent_reply_is_not_context(self):
        """An UNSENT reply never reached the person, so it can't be what a
        follow-up refers to."""
        messages = [
            assistant("The porch ESP32."),
            user("ok"),
            assistant("vetoed text", _message_type=MessageType.UNSENT),
            user("and the other one?"),
        ]
        assert retrieval_query(messages, "previous_reply") == (
            "The porch ESP32.\n\nand the other one?"
        )

    def test_context_rows_are_not_the_reply(self):
        messages = [
            assistant("The porch ESP32."),
            {"role": "user", "content": "[CONTEXT from memory]",
             "_message_type": MessageType.CONTEXT},
            user("and the other one?"),
        ]
        assert retrieval_query(messages, "previous_reply") == (
            "The porch ESP32.\n\nand the other one?"
        )


class TestBeforeAgentTurnQuery:
    """before_agent_turn embeds the combined query and retrieves by it."""

    async def test_follow_up_retrieves_the_previous_turns_topic(self):
        config = {**EMBED_CONFIG, "memory": {"retrieval": {"k": 1}}}
        memory, embedder, channel, conv, db = await build_retrieval_env(config)
        await seed_memory(db, channel.id, "zephyr quasar nebula pulsar telescope")
        await seed_memory(db, channel.id, "thanks worked")

        conv.append(user("what scope did I buy?"))
        conv.append(assistant("The zephyr quasar nebula pulsar telescope."))
        conv.append(user("thanks, that worked"))
        await memory.before_agent_turn(channel=channel, trace_id=None, annotations={})

        assert embedder.calls[-1][0].startswith("The zephyr quasar")
        recalled = " ".join(m["content"] for m in context_messages(conv))
        assert "zephyr quasar nebula pulsar telescope" in recalled
        await db.close()

    async def test_query_context_none_embeds_the_user_message_only(self):
        config = {**EMBED_CONFIG,
                  "memory": {"retrieval": {"query_context": "none"}}}
        memory, embedder, channel, conv, db = await build_retrieval_env(config)
        await seed_memory(db, channel.id, "anything")

        conv.append(assistant("The zephyr telescope."))
        conv.append(user("thanks, that worked"))
        await memory.before_agent_turn(channel=channel, trace_id=None, annotations={})

        assert embedder.calls[-1] == ["thanks, that worked"]
        await db.close()

    async def test_unknown_query_context_warns_and_uses_the_default(self, caplog):
        config = {**EMBED_CONFIG,
                  "memory": {"retrieval": {"query_context": "everything"}}}
        with caplog.at_level(logging.WARNING):
            memory, embedder, channel, conv, db = await build_retrieval_env(config)
        message = " ".join(r.getMessage() for r in caplog.records)
        # Names the key, the rejected value and the accepted ones.
        assert "memory.retrieval.query_context" in message
        assert "'everything'" in message
        assert "previous_reply" in message and "none" in message
        assert memory._query_context == "previous_reply"
        await db.close()
