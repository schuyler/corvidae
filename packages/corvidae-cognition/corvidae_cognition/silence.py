"""SilencePlugin — the agent's right to silence (#215; PRD §3.7 items 5-6).

The agent may end any turn without sending anything. The plugin does three
things:

  * It appends an instruction telling the model how (reply with
    ``NO_REPLY``) to the system prompt of every channel whose transport is
    conversational, through the kernel's ``transform_system_prompt`` hook.
  * It vetoes a final emission that carries the ``NO_REPLY`` token — an
    exact reply, or a reply that embeds it — so nothing reaches the
    transport and the window records it as unsent.
  * It records every decline (a vetoed ``NO_REPLY`` or an empty final) in
    ``exchange_log.outcomes`` as ``{"silence": "no_reply" | "empty"}``.

``silence.mode`` (#797) adds a second way to ask for silence. In ``tool`` and
``both`` modes the plugin also registers an ``end_turn(reason)`` tool: a
tool-trained model takes a tool call more readily than a magic string. The
tool's result tells the model to echo ``NO_REPLY``, which the veto then
withholds, so the follow-up model call the loop makes after any tool result
cannot leak text when the model complies. The call is recorded as
``{"silence": "tool", "silence_reason": ...}``; the row means the tool was
called, not that nothing was sent.

With the plugin disabled there is no instruction and no veto.
"""

import asyncio
import logging

from corvidae.attribution import get_attribution
from corvidae.hooks import CorvidaePlugin, TransportPlugin, final_was_unsent, hookimpl
from corvidae.prompts import resolve_prompt
from corvidae.tool import Tool, ToolContext, tool_to_schema

logger = logging.getLogger("corvidae.silence")

# Same literal as critique.py's NO_REPLY_MARKER; critique is unchanged.
NO_REPLY = "NO_REPLY"


def _carries_no_reply(text: str) -> bool:
    """True if the final text carries the NO_REPLY token anywhere in it.

    Containment, not equality: the model is told to reply with exactly the
    token, but a reply that embeds it ("quiet: NO_REPLY") is a silence
    signal too and must not leak to the transport. The token is a
    distinctive control string; ordinary prose does not contain it.
    """
    return NO_REPLY in text


# The built-in instruction appended to a conversational channel's system
# prompt (S7); it reaches custom and empty registers too (I1). Operators
# override it with the silence.instruction key (PRD principle 15).
REGISTER_FRAGMENT = (
    "You don't have to answer every message. Before replying, ask whether you "
    "have anything true, relevant and worth saying to add. When a conversation "
    "has wound down (thanks, goodnight, \"lol ok\", an acknowledgement that "
    "needs nothing back), stay silent: reply with NO_REPLY and nothing else, "
    "and nothing is sent. If they wanted more, they'll say so. A question "
    "or request addressed to you still gets a real answer. "
    "Never write NO_REPLY anywhere else in a message; any reply containing "
    "the token is dropped entirely."
)

# silence.mode values: how the agent is asked to decline a turn (#797).
SILENCE_MODES = ("token", "tool", "both")

# The instruction in ``tool`` mode: the same norm as REGISTER_FRAGMENT, with
# the end_turn tool as the way to act on it.
TOOL_FRAGMENT = (
    "You don't have to answer every message. Before replying, ask whether you "
    "have anything true, relevant and worth saying to add. When a conversation "
    "has wound down (thanks, goodnight, \"lol ok\", an acknowledgement that "
    "needs nothing back), call the end_turn tool, on its own and with no other "
    "text, and nothing is sent. If they wanted more, they'll say so. A question "
    "or request addressed to you still gets a real answer."
)

# The instruction in ``both`` mode: the tool first, the token as a fallback.
BOTH_FRAGMENT = (
    TOOL_FRAGMENT
    + " If you can't call the tool, reply with exactly NO_REPLY and nothing "
    "else; that sends nothing too."
)

# What end_turn returns. The loop calls the model again after any tool
# result; the exact-token echo it asks for is what the veto withholds.
END_TURN_RESULT = (
    "Nothing was sent. Your next message must be exactly NO_REPLY and nothing else."
)

_MODE_FRAGMENTS = {
    "token": REGISTER_FRAGMENT,
    "tool": TOOL_FRAGMENT,
    "both": BOTH_FRAGMENT,
}


class SilencePlugin(CorvidaePlugin):
    """The right to silence: the instruction on conversational transports
    (via transform_system_prompt), the NO_REPLY veto, and the decline record."""

    # Records through outcome_log when present; absent, nothing is recorded.
    optional_depends_on = frozenset({"outcome_log"})

    def __init__(self, pm=None) -> None:
        if pm is not None:
            self.pm = pm
        # Strong refs to fire-and-forget outcome writes, so a write is never
        # GC-dropped mid-flight and on_stop can drain them.
        self._record_tasks: set[asyncio.Task] = set()
        # The instruction in effect; on_init resolves silence.instruction.
        self._instruction: str = REGISTER_FRAGMENT
        # How silence is asked for: token (default), tool or both.
        self._mode: str = "token"
        # Traces that called end_turn, so the echoed NO_REPLY final does not
        # overwrite the tool's outcome under the same key. A trace leaves the
        # set when its final reaches on_agent_response.
        self._ended: set[str] = set()

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        """Store pm, read silence.mode, and resolve the instruction (file,
        inline, or the built-in for the mode).

        Read at startup only: the instruction lands in a conversation's
        system prompt when it starts, and a restart rebuilds every one.
        """
        await super().on_init(pm, config)
        block = config.get("silence")
        # The runtime reports a ValueError from on_init as "Config error:
        # <detail> — check <path>", so the message carries the detail only.
        # A scalar block (``silence: tool``) is a typo, not "no block".
        if block is not None and not isinstance(block, dict):
            raise ValueError(
                f"silence is {block!r}; use a mapping such as "
                f"'silence: {{mode: tool}}', or 'plugins.disabled: [silence]' "
                "to switch the plugin off"
            )
        mode = block.get("mode", "token") if block else "token"
        # Reject an unknown mode loudly: a typo must not quietly fall back.
        if mode not in SILENCE_MODES:
            raise ValueError(
                f"silence.mode is {mode!r}; use one of {', '.join(SILENCE_MODES)}"
            )
        self._mode = mode
        self._instruction = resolve_prompt(
            config, "silence.instruction", _MODE_FRAGMENTS[mode]
        )

    @hookimpl
    def register_tools(self, tool_registry: list) -> None:
        """Offer end_turn in ``tool`` and ``both`` modes; token mode has no tool."""
        if self._mode == "token":
            return
        plugin = self

        async def end_turn(reason: str = "", _ctx: ToolContext | None = None) -> str:
            """End your turn without sending anything. Call it on its own, with no other text, when nothing needs saying. reason: one short phrase on why."""
            # A call with no channel cannot be checked or recorded; refuse it.
            channel = _ctx.channel if _ctx is not None else None
            if channel is None:
                return (
                    "Error: no channel context available; end_turn did nothing, "
                    "so answer normally"
                )
            # Only a conversational transport may be left unanswered; a task
            # session's request addressed to the agent is always answered.
            if not plugin._is_conversational(channel):
                return (
                    f"Error: end_turn is not available on {channel.id}, which is not "
                    "a conversational channel; answer normally"
                )
            # Record the call against the turn's trace, if it has one.
            trace_id = get_attribution().get("trace_id")
            if trace_id is None:
                logger.warning(
                    "silence: end_turn on %s has no trace id; no exchange_log row "
                    "recorded", channel.id, extra={"channel": channel.id},
                )
            else:
                plugin._ended.add(trace_id)
                plugin._record(
                    trace_id, channel.id,
                    {"silence": "tool", "silence_reason": str(reason)[:200]},
                )
            return END_TURN_RESULT

        tool_registry.append(Tool(name="end_turn", fn=end_turn, schema=tool_to_schema(end_turn)))

    def _is_conversational(self, channel) -> bool:
        """True when the transport that owns ``channel`` is conversational."""
        for plugin in self.pm.get_plugins():
            if isinstance(plugin, TransportPlugin) and plugin.owns(channel):
                return bool(plugin.conversational)
        return False

    def _record(self, trace_id: str, channel_id: str, outcomes: dict) -> None:
        """Write ``outcomes`` to exchange_log from a tracked fire-and-forget
        task, so the reply path never waits on the write connection."""
        outcome_log = self.pm.get_plugin("outcome_log")
        if outcome_log is None:
            return

        async def _write() -> None:
            try:
                await outcome_log.upsert_exchange(trace_id, channel_id, outcomes=outcomes)
            except Exception:
                logger.warning(
                    "silence outcome write failed for %s on %s; the decline is "
                    "not in exchange_log", trace_id, channel_id, exc_info=True,
                    extra={"trace_id": trace_id, "channel": channel_id},
                )

        task = asyncio.create_task(_write())
        self._record_tasks.add(task)
        task.add_done_callback(self._record_tasks.discard)

    @hookimpl(wrapper=True)
    def transform_system_prompt(self, channel, text):
        """Append the instruction after the composed prompt, on chat channels only."""
        result = yield
        # Find the transport that delivers to this channel; only a
        # conversational one gets the instruction (web, CLI and ACP always answer).
        for plugin in self.pm.get_plugins():
            if isinstance(plugin, TransportPlugin) and plugin.owns(channel):
                if isinstance(result, str) and plugin.conversational:
                    return f"{result}\n\n{self._instruction}"
                break
        return result

    @hookimpl
    async def on_stop(self) -> None:
        """Let in-flight outcome writes land rather than abandoning them."""
        while self._record_tasks:
            await asyncio.gather(*set(self._record_tasks), return_exceptions=True)

    @hookimpl
    async def should_send_response(self, channel, text, emission, trace_id, annotations) -> bool | None:
        """Veto a final that carries the NO_REPLY token, for every origin.

        A reply that is exactly the token, or that embeds it, is a silence
        signal: nothing reaches the transport. Anything else is no opinion.
        An error fails open: a broken plugin must never silence the agent.
        """
        try:
            if emission == "final" and _carries_no_reply(text):
                return False
            return None
        except Exception:
            logger.warning(
                "silence veto failed for the %s emission of %s on %s; sending it",
                emission, trace_id, channel.id, exc_info=True,
                extra={"channel": channel.id, "trace_id": trace_id},
            )
            return None

    @hookimpl
    async def on_agent_response(
        self,
        channel,
        request_text: str,
        response_text: str,
        trace_id: str | None,
        annotations: dict,
        logprobs: dict | None,
    ) -> None:
        """Record a decline: an unsent NO_REPLY or an unsent empty final."""
        # Read before the first await: the kernel may move on once a sibling
        # impl raises, and the window must still show this final's row.
        try:
            unsent = final_was_unsent(channel)
        except Exception:
            logger.warning(
                "silence: could not tell whether the final on %s was unsent; "
                "recording nothing", channel.id, exc_info=True,
                extra={"channel": channel.id, "trace_id": trace_id},
            )
            return
        # A turn that called end_turn already has its outcome; its final,
        # echoed or not, must not replace it under the same key. Leave the
        # set whether or not the final was withheld.
        ended = trace_id is not None and trace_id in self._ended
        if ended:
            self._ended.discard(trace_id)
        if not unsent or ended:
            return
        # Only the two decline shapes count; another gate's veto is not ours.
        text = response_text.strip()
        if _carries_no_reply(text):
            kind = "no_reply"
        elif text == "":
            kind = "empty"
        else:
            return
        if trace_id is None:
            logger.warning(
                "silence: a %s decline on %s has no trace id; no exchange_log row "
                "recorded", kind, channel.id, extra={"channel": channel.id},
            )
            return
        self._record(trace_id, channel.id, {"silence": kind})
