"""SilencePlugin — the agent's right to silence (#215; PRD §3.7 items 5-6).

The agent may end any turn without sending anything. The plugin does three
things:

  * It owns the register fragment that tells the model how (reply with
    exactly ``NO_REPLY``); chat transports append it to their default
    register when this plugin is loaded.
  * It vetoes a final emission that is exactly ``NO_REPLY``, so nothing
    reaches the transport and the window records it as unsent.
  * It records every decline (a vetoed ``NO_REPLY`` or an empty final) in
    ``exchange_log.outcomes`` as ``{"silence": "no_reply" | "empty"}``.

With the plugin disabled there is no instruction and no veto.
"""

import asyncio
import logging

from corvidae.hooks import CorvidaePlugin, final_was_withheld, hookimpl

logger = logging.getLogger("corvidae.silence")

# Same literal as critique.py's NO_REPLY_MARKER; critique is unchanged.
NO_REPLY = "NO_REPLY"

# The instruction chat transports append to their default register (S7).
# Operators with a custom register_prompt copy this text into it (I1).
REGISTER_FRAGMENT = (
    "You don't have to answer every message. Before replying, ask whether you "
    "have anything true, relevant and worth saying to add. When a conversation "
    "has wound down (thanks, goodnight, \"lol ok\", an acknowledgement that "
    "needs nothing back), stay silent: reply with exactly NO_REPLY and nothing "
    "else, and nothing is sent. If they wanted more, they'll say so. A question "
    "or request addressed to you still gets a real answer."
)


class SilencePlugin(CorvidaePlugin):
    """The NO_REPLY veto and the decline record."""

    # Records through outcome_log when present; absent, nothing is recorded.
    optional_depends_on = frozenset({"outcome_log"})

    def __init__(self, pm=None) -> None:
        if pm is not None:
            self.pm = pm
        # Strong refs to fire-and-forget outcome writes, so a write is never
        # GC-dropped mid-flight and on_stop can drain them.
        self._record_tasks: set[asyncio.Task] = set()

    def register_fragment(self) -> str:
        """The silence instruction a transport appends to its register."""
        return REGISTER_FRAGMENT

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        """Store pm (via the base class) so the plugin can find outcome_log."""
        await super().on_init(pm, config)

    @hookimpl
    async def on_stop(self) -> None:
        """Let in-flight outcome writes land rather than abandoning them."""
        while self._record_tasks:
            await asyncio.gather(*set(self._record_tasks), return_exceptions=True)

    @hookimpl
    async def should_send_response(self, channel, text, emission, trace_id, meta) -> bool | None:
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
        meta: dict,
        logprobs: dict | None,
    ) -> None:
        """Record a decline: a withheld NO_REPLY or a withheld empty final."""
        # Read before the first await: the kernel may move on once a sibling
        # impl raises, and the window must still show this final's row.
        try:
            withheld = final_was_withheld(channel)
        except Exception:
            logger.warning(
                "silence: could not tell whether the final on %s was withheld; "
                "recording nothing", channel.id, exc_info=True,
                extra={"channel": channel.id, "trace_id": trace_id},
            )
            return
        if not withheld:
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
