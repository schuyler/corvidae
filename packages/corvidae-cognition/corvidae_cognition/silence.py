"""SilencePlugin — the agent's right to silence (#215; PRD §3.7 items 5-6).

The agent may end any turn without sending anything. The plugin does three
things:

  * It appends an instruction telling the model how (reply with exactly
    ``NO_REPLY``) to the system prompt of every channel whose transport is
    conversational, through the kernel's ``transform_system_prompt`` hook.
  * It vetoes a final emission that is exactly ``NO_REPLY``, so nothing
    reaches the transport and the window records it as unsent.
  * It records every decline (a vetoed ``NO_REPLY`` or an empty final) in
    ``exchange_log.outcomes`` as ``{"silence": "no_reply" | "empty"}``.

With the plugin disabled there is no instruction and no veto.
"""

import asyncio
import logging

from corvidae.hooks import CorvidaePlugin, TransportPlugin, final_was_unsent, hookimpl
from corvidae.prompts import resolve_prompt

logger = logging.getLogger("corvidae.silence")

# Same literal as critique.py's NO_REPLY_MARKER; critique is unchanged.
NO_REPLY = "NO_REPLY"

# The built-in instruction appended to a conversational channel's system
# prompt (S7); it reaches custom and empty registers too (I1). Operators
# override it with the silence.instruction key (PRD principle 15).
REGISTER_FRAGMENT = (
    "You don't have to answer every message. Before replying, ask whether you "
    "have anything true, relevant and worth saying to add. When a conversation "
    "has wound down (thanks, goodnight, \"lol ok\", an acknowledgement that "
    "needs nothing back), stay silent: reply with exactly NO_REPLY and nothing "
    "else, and nothing is sent. If they wanted more, they'll say so. A question "
    "or request addressed to you still gets a real answer."
)


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

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        """Store pm and resolve the instruction (file, inline, or built-in).

        Read at startup only: the instruction lands in a conversation's
        system prompt when it starts, and a restart rebuilds every one.
        """
        await super().on_init(pm, config)
        self._instruction = resolve_prompt(config, "silence.instruction", REGISTER_FRAGMENT)

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
        """Veto a final that is exactly NO_REPLY, for every origin.

        Anything else is no opinion. An error fails open: a broken plugin
        must never silence the agent.
        """
        try:
            if emission == "final" and text.strip() == NO_REPLY:
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
        if not unsent:
            return
        # Only the two decline shapes count; another gate's veto is not ours.
        text = response_text.strip()
        if text == NO_REPLY:
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
        outcome_log = self.pm.get_plugin("outcome_log")
        if outcome_log is None:
            return

        async def _write() -> None:
            # Fire-and-forget so the reply path never waits on the write connection.
            try:
                await outcome_log.upsert_exchange(
                    trace_id, channel.id, outcomes={"silence": kind},
                )
            except Exception:
                logger.warning(
                    "silence outcome write failed for %s on %s; the decline is "
                    "not in exchange_log", trace_id, channel.id, exc_info=True,
                    extra={"trace_id": trace_id, "channel": channel.id},
                )

        task = asyncio.create_task(_write())
        self._record_tasks.add(task)
        task.add_done_callback(self._record_tasks.discard)
