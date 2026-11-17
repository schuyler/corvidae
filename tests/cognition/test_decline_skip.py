"""A decline gets no critique and no stage-2 appraisal (#420).

PRD §3.7 Critique 3: "A response a gate withheld is still judged; the
agent's choice to stay silent is not a response, and is neither judged nor
appraised." A decline is an unsent final whose text carries ``NO_REPLY`` or
is empty (Output gate items 5-6). A final another gate withheld, which has
content, stays eligible.

The unit matrix fires each plugin's ``on_agent_response`` with a window
whose newest assistant row is the final, tagged as the kernel tags it. The
agent-level tests run a real turn through the real Agent with SilencePlugin,
AppraisalPlugin and CritiquePlugin loaded and watch what reaches the task
queue.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from helpers import build_plugin_and_channel, drain
from llm_response_fixtures import _make_text_response

from corvidae.channel import Channel
from corvidae.context import ContextWindow, MessageType
from corvidae.hooks import CorvidaePlugin, create_plugin_manager, hookimpl
from corvidae.trace import mint_trace_id

# (response_text, the final row's type, whether the turn is judged/appraised)
DECLINE_MATRIX = [
    pytest.param("NO_REPLY", MessageType.UNSENT, False, id="unsent-no-reply"),
    pytest.param("  NO_REPLY\n", MessageType.UNSENT, False, id="unsent-padded-no-reply"),
    pytest.param("quiet: NO_REPLY", MessageType.UNSENT, False, id="unsent-embedded-no-reply"),
    # A decline whose text asks permission: the proactivity check would fire
    # on it if the skip came after that check.
    pytest.param(
        "Should I restart it? NO_REPLY", MessageType.UNSENT, False,
        id="unsent-permission-question-no-reply",
    ),
    pytest.param("", MessageType.UNSENT, False, id="unsent-empty"),
    pytest.param("  \n", MessageType.UNSENT, False, id="unsent-whitespace"),
    # Another gate withheld a response with content: still judged.
    pytest.param("Should I restart it?", MessageType.UNSENT, True, id="withheld-content"),
    # A sent final is a response, whatever it says.
    pytest.param("Should I restart it?", MessageType.MESSAGE, True, id="sent"),
    # A token that was sent anyway (silence disabled, or its veto failed
    # open) reached the person: a response, not a decline.
    pytest.param("NO_REPLY", MessageType.MESSAGE, True, id="sent-no-reply"),
]


def _window_ending_in(channel: Channel, text: str, final_type: MessageType) -> ContextWindow:
    """A window shaped as the kernel leaves it when on_agent_response fires:
    the originating user row, then the final's assistant row tagged with its
    type (UNSENT when it was withheld)."""
    conv = ContextWindow(channel.id)
    conv.system_prompt = "You are Lex."
    conv.append({"role": "user", "content": "q", "_db_id": 1})
    conv.append({"role": "assistant", "content": text, "_db_id": 2}, final_type)
    return conv


class _QueueSpy:
    """Stands in for TaskPlugin: records each enqueued task's description."""

    def __init__(self):
        self.descriptions: list[str] = []
        self.task_queue = SimpleNamespace(enqueue=self._enqueue)

    async def _enqueue(self, task):
        self.descriptions.append(task.description)


# ---------------------------------------------------------------------------
# Appraisal: no stage-2 task for a decline
# ---------------------------------------------------------------------------


class TestAppraisalSkipsDeclines:
    @pytest.mark.parametrize("text, final_type, judged", DECLINE_MATRIX)
    async def test_stage2_enqueued_only_for_a_response(self, text, final_type, judged):
        from corvidae_cognition.appraisal import AppraisalPlugin

        pm = create_plugin_manager()
        spy = _QueueSpy()
        pm.register(spy, name="task")
        appraisal = AppraisalPlugin()
        pm.register(appraisal, name="appraisal")
        await appraisal.on_init(pm=pm, config={})
        channel = Channel(transport="test", scope="decline")
        channel.conversation = _window_ending_in(channel, text, final_type)

        await appraisal.on_agent_response(
            channel=channel, request_text="q", response_text=text,
            trace_id="t-decline", annotations={}, logprobs=None,
        )

        assert spy.descriptions == (["appraisal:stage2"] if judged else [])

    async def test_check_failure_logs_and_appraises(self, monkeypatch, caplog):
        """A decline check that raises fails toward appraising, with a
        warning naming the channel and trace."""
        from corvidae_cognition import appraisal as appraisal_mod

        def _boom(channel, text):
            raise RuntimeError("window unreadable")

        monkeypatch.setattr(appraisal_mod, "is_decline", _boom)
        pm = create_plugin_manager()
        spy = _QueueSpy()
        pm.register(spy, name="task")
        appraisal = appraisal_mod.AppraisalPlugin()
        pm.register(appraisal, name="appraisal")
        await appraisal.on_init(pm=pm, config={})
        channel = Channel(transport="test", scope="decline")

        with caplog.at_level("WARNING", logger="corvidae.appraisal"):
            await appraisal.on_agent_response(
                channel=channel, request_text="q", response_text="NO_REPLY",
                trace_id="t-boom", annotations={}, logprobs=None,
            )

        assert spy.descriptions == ["appraisal:stage2"]
        assert "test:decline" in caplog.text
        assert "t-boom" in caplog.text


# ---------------------------------------------------------------------------
# Critique: no stylistic lens and no proactivity check for a decline
# ---------------------------------------------------------------------------


class _LensAppraisal(CorvidaePlugin):
    """An appraisal store whose stage 2 always selects the predictive lens."""

    depends_on = frozenset()

    async def get_appraisal(self, trace_id):
        return None

    def get_last_stage2(self, channel_id):
        return {"ambiguity": 0.9}

    def get_originating_text(self, trace_id):
        return "q"

    def add_stage2_listener(self, callback):
        pass


async def _critique_env(tmp_path):
    """CritiquePlugin with a lens-selecting appraisal and a queue spy."""
    from corvidae_cognition.critique import CritiquePlugin

    pm = create_plugin_manager()
    spy = _QueueSpy()
    pm.register(spy, name="task")
    pm.register(_LensAppraisal(), name="appraisal")
    critique = CritiquePlugin()
    pm.register(critique, name="critique")
    cfg = {"daemon": {"session_db": str(tmp_path / "sessions.db")}}
    await critique.on_init(pm=pm, config=cfg)
    return SimpleNamespace(pm=pm, spy=spy, critique=critique)


class TestCritiqueSkipsDeclines:
    @pytest.mark.parametrize("text, final_type, judged", DECLINE_MATRIX)
    async def test_lens_and_proactivity_only_for_a_response(
        self, tmp_path, text, final_type, judged
    ):
        env = await _critique_env(tmp_path)
        channel = Channel(transport="test", scope="decline")
        channel.conversation = _window_ending_in(channel, text, final_type)
        # A user-origin turn: admitted, no origin annotation.
        await env.critique.on_message_admitted(
            channel=channel, trace_id="t-decline", sender="user", text="q",
        )

        await env.critique.on_agent_response(
            channel=channel, request_text="q", response_text=text,
            trace_id="t-decline", annotations={}, logprobs=None,
        )

        # A judged text that asks permission also gets the proactivity
        # check; the stylistic task (the predictive lens) comes from the
        # stage-2 ambiguity.
        expected = []
        if judged:
            expected = (["critique:proactivity"] if "Should I" in text else []) + ["critique"]
        assert env.spy.descriptions == expected

    async def test_check_failure_logs_and_judges(self, tmp_path, monkeypatch, caplog):
        """A decline check that raises fails toward judging, with a warning
        naming the channel and trace."""
        from corvidae_cognition import critique as critique_mod

        def _boom(channel, text):
            raise RuntimeError("window unreadable")

        monkeypatch.setattr(critique_mod, "is_decline", _boom)
        env = await _critique_env(tmp_path)
        channel = Channel(transport="test", scope="decline")
        await env.critique.on_message_admitted(
            channel=channel, trace_id="t-boom", sender="user", text="q",
        )

        with caplog.at_level("WARNING", logger="corvidae.critique"):
            await env.critique.on_agent_response(
                channel=channel, request_text="q", response_text="NO_REPLY",
                trace_id="t-boom", annotations={}, logprobs=None,
            )

        assert env.spy.descriptions == ["critique"]
        assert "test:decline" in caplog.text
        assert "t-boom" in caplog.text


# ---------------------------------------------------------------------------
# Agent level: a real turn, the real hook order
# ---------------------------------------------------------------------------


class _WithholdAll(CorvidaePlugin):
    """Another gate: vetoes every final, as an enforcing engagement gate would."""

    depends_on = frozenset()

    @hookimpl
    async def should_send_response(self, channel, text, emission, trace_id, annotations):
        return False if emission == "final" else None


@pytest.fixture
async def turn_env(tmp_path):
    """A real Agent with SilencePlugin, AppraisalPlugin and CritiquePlugin, and
    the task queue's enqueue recorded instead of run."""
    from corvidae_cognition.appraisal import AppraisalPlugin
    from corvidae_cognition.critique import CritiquePlugin
    from corvidae_cognition.silence import SilencePlugin

    db_path = str(tmp_path / "sessions.db")
    plugin, channel, db = await build_plugin_and_channel(
        mock_on_agent_response=False, db_path=db_path,
    )
    cfg = {"daemon": {"session_db": db_path}}
    silence, appraisal, critique = SilencePlugin(), AppraisalPlugin(), CritiquePlugin()
    for name, p in (("silence", silence), ("appraisal", appraisal), ("critique", critique)):
        plugin.pm.register(p, name=name)
        await p.on_init(pm=plugin.pm, config=cfg)
    for p in (appraisal, critique):
        await p.on_start(config=cfg)
    # Record what the turn hands the task queue; running it would need models.
    task_plugin = plugin.pm.get_plugin("task")
    enqueued: list[str] = []

    async def _record(task):
        enqueued.append(task.description)

    task_plugin.task_queue.enqueue = _record
    yield SimpleNamespace(plugin=plugin, channel=channel, enqueued=enqueued)
    for p in (critique, appraisal, silence):
        await p.on_stop()
    await task_plugin.on_stop()
    await db.close()


async def _user_turn(env, reply: str) -> None:
    """Run one admitted user turn whose model answers ``reply``."""
    client = MagicMock()
    client.chat = AsyncMock(side_effect=[_make_text_response(reply)])
    env.plugin._client = client
    env.plugin._mark_started()
    trace_id = mint_trace_id()
    # The kernel admits a user message before queuing it; critique keys
    # user origin on that hook.
    await env.plugin.pm.ahook.on_message_admitted(
        channel=env.channel, trace_id=trace_id, sender="user", text="Should I restart it?",
    )
    await env.plugin.on_message(
        channel=env.channel, sender="user", text="Should I restart it?", trace_id=trace_id,
    )
    await drain(env.plugin, env.channel)


class TestDeclineAtAgentLevel:
    # The permission-question decline makes the critique half deterministic:
    # without the skip it would queue the proactivity check, where the plain
    # token could only be caught by the critique.sample_below_rate draw.
    @pytest.mark.parametrize("reply", ["NO_REPLY", "", "Should I restart it? NO_REPLY"])
    async def test_declined_turn_is_neither_appraised_nor_judged(self, turn_env, reply):
        await _user_turn(turn_env, reply)
        # The final went unsent, and nothing was queued to appraise or judge it.
        finals = [m for m in turn_env.channel.conversation.messages if m.get("role") == "assistant"]
        assert finals[-1]["_message_type"] == MessageType.UNSENT
        assert turn_env.enqueued == []

    async def test_withheld_response_is_still_appraised_and_judged(self, turn_env):
        turn_env.plugin.pm.register(_WithholdAll(), name="withhold_all")
        await _user_turn(turn_env, "Should I restart it?")
        finals = [m for m in turn_env.channel.conversation.messages if m.get("role") == "assistant"]
        assert finals[-1]["_message_type"] == MessageType.UNSENT
        # Stage 2 runs, and the permission question draws the proactivity check.
        assert "appraisal:stage2" in turn_env.enqueued
        assert "critique:proactivity" in turn_env.enqueued
