"""AppraisalPlugin — the stage-2 appraisal of a finished exchange.

After the agent answers, a silent task forks the live window and asks the
agent to list the claims about the past its reply made and to score the
critique lenses on offer. The claims feed critique's provenance check and
the lens scores its lens selection; both are persisted to the outcome log,
where the lens scores are the labels a trained readout head learns from
(#141). Stage-1 appraisal, which scored text at the gates from English word
lists and a probe of memory's full-text index, was removed (#1185); the
five-dimension appraisal vector, the correction flag, the importance prior
built from them and memory's valence tag were removed as unused (#1189).
"""

from __future__ import annotations

import collections
import json
import logging
import math
from collections.abc import Awaitable, Callable

from corvidae.attribution import reset_attribution, set_attribution
from corvidae.fork import fork_agent, quote, window_db_ids
from corvidae.hooks import CorvidaePlugin, hookimpl
from corvidae.task import Task
from corvidae.prompts import resolve_prompt
from corvidae_cognition.config_keys import warn_removed_keys
from corvidae_cognition.silence import is_decline

# Named explicitly: corvidae/logging.py configures the "corvidae" logger,
# and __name__ would be corvidae_cognition.appraisal.
logger = logging.getLogger("corvidae.appraisal")

# Keys that configure nothing any more: each is warned by name at startup
# and reload, and otherwise ignored. The stage-1 keys went with stage 1
# (#1185); the importance-prior weights went with the appraisal vector (#1189).
_PRIOR_REMOVED = "the appraisal vector and the importance prior built from it were removed (#1189)"
_REMOVED_KEYS = {
    "appraisal.probe": "stage-1 appraisal and its memory probe were removed (#1185)",
    "appraisal.weights": "stage-1 appraisal and its salience score were removed (#1185)",
    "appraisal.novelty": "stage-1 appraisal and its memory probe were removed (#1185)",
    "appraisal.prior.w_stakes": _PRIOR_REMOVED,
    "appraisal.prior.w_valence": _PRIOR_REMOVED,
    "appraisal.prior.w_novelty": _PRIOR_REMOVED,
}

# Bound on the originating-text LRU (AppraisalPlugin._originating_text).
# Keyed by trace_id, not per-channel — user messages interleave
# mid-cycle by design, so a single per-channel slot would be clobbered.
ORIGINATING_TEXT_LRU_MAXSIZE = 512

# Documented copy of this text lives in prompts/appraisal.md. No live stage
# reads it: stage 2 runs only as a fork and skips when there is none (#320), so
# the appraisal.stage2_prompt key is accepted and never consulted (#1129).
DEFAULT_STAGE2_PROMPT = (
    "You are appraising a completed exchange for the agent's own perception "
    "system. You are given the originating message and the agent's final "
    "response. List the claims about the past the response makes.\n\n"
    "- past_claims: every statement in the agent's final response that "
    "presents something as established BEFORE this exchange — something "
    "said, agreed, promised, decided or done earlier — in any language or "
    "phrasing. Leave out what the originating message itself says, general "
    "knowledge, and claims the response frames as uncertain. For each, give "
    "the claim in a few words and up to 3 short search queries: the "
    "distinctive words (names, numbers, dates, topics) the original "
    "statement would have used. An empty list when there are none.\n\n"
    "Respond with a single JSON object, nothing else:\n"
    "{\n"
    '  "past_claims": [{"claim": "...", "queries": ["..."]}]\n'
    "}"
)

# Fork instruction for stage 2 (GitHub issue #51). With agent.forks on, the
# appraisal is a fork of the live window: this text is appended as a user
# turn, followed by a quote of the exchange's opening message and of the
# response. Second person: the fork is the agent appraising its own
# exchange. Documented copy in prompts/appraisal_fork.md; the static config
# key appraisal.stage2_fork_prompt (not a runtime tunable) overrides it.
DEFAULT_STAGE2_FORK_PROMPT = (
    "[Appraisal] Pause the conversation here and appraise, for your own "
    "perception system, the exchange you just completed: from the message "
    "quoted below through your final response. List the claims about the "
    "past your final response makes.\n\n"
    "- past_claims: every statement in your final response that presents "
    "something as established BEFORE this exchange — something said, "
    "agreed, promised, decided or done earlier — in any language or "
    "phrasing. Leave out what the opening message itself says, general "
    "knowledge, and claims you framed as uncertain. For each, give the claim "
    "in a few words and up to 3 short search queries: the distinctive words "
    "(names, numbers, dates, topics) the original statement would have "
    "used. An empty list when there are none.\n\n"
    "Do not call tools. Respond with a single JSON object, nothing else:\n"
    "{\n"
    '  "past_claims": [{"claim": "...", "queries": ["..."]}]\n'
    "}"
)

# JSON schema constraining the stage-2 call on servers that honour it
# (llama-server grammar / json_schema via extra_body). Providers that ignore
# it still return the JSON the prompt asks for; _parse_json_block reads it.
STAGE2_SCHEMA = {
    "type": "object",
    "properties": {
        "past_claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "queries": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["claim", "queries"],
            },
        },
    },
    "required": ["past_claims"],
}

# Bounds on the stage-2 past_claims list (#155), so one verbose reply
# can't fan out into an unbounded evidence search and critic prompt.
MAX_PAST_CLAIMS = 5
MAX_QUERIES_PER_CLAIM = 3
MAX_QUERY_CHARS = 200
STAGE2_EXTRA_BODY = {
    "response_format": {
        "type": "json_schema",
        "json_schema": {"name": "stage2_appraisal", "schema": STAGE2_SCHEMA},
    }
}


def _extra_body_with_lenses(names: list[str]) -> dict:
    """STAGE2_EXTRA_BODY with a required ``lenses`` object scoring ``names``
    and "none". Built per call from a copy; the shared schema is untouched."""
    keys = [*names, "none"]
    schema = {
        **STAGE2_SCHEMA,
        "properties": {
            **STAGE2_SCHEMA["properties"],
            "lenses": {
                "type": "object",
                "properties": {key: {"type": "number"} for key in keys},
                "required": keys,
            },
        },
        "required": [*STAGE2_SCHEMA["required"], "lenses"],
    }
    return {
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "stage2_appraisal", "schema": schema},
        }
    }


def _parse_json_block(text: str) -> dict:
    """Extract the first JSON object from LLM output.

    Tolerates surrounding prose and markdown code fences. Raises ValueError
    when no parseable object is present — callers own their degradation.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"appraisal stage 2: no JSON object in LLM output: {text[:500]!r}")
    return json.loads(text[start:end + 1])


def _lens_vector(raw: object, offered: list[str], trace_id: str) -> dict[str, float] | None:
    """Normalize stage 2's critique vector: shares of 1 over ``offered`` + "none".

    Drops (each with a warning naming the entry and the trace) names that
    were not offered, and values that are not finite, non-negative real
    numbers (a bool is not one). Renormalizes the rest to sum to 1; an
    offered name the reply left out counts as 0. Returns None, with a
    warning naming the trace and the reason, when the field is missing, is
    not an object, or has nothing usable left (sums to 0).
    """
    # The field must be an object of name -> percentage.
    if raw is None:
        logger.warning(
            "stage-2 lenses for trace %s: the reply has no \"lenses\" field; "
            "no critique vector for this exchange", trace_id)
        return None
    if not isinstance(raw, dict):
        logger.warning(
            "stage-2 lenses for trace %s: \"lenses\" is a %s, not an object; "
            "no critique vector for this exchange", trace_id, type(raw).__name__)
        return None
    allowed = {*offered, "none"}
    usable: dict[str, float] = {}
    for name, value in raw.items():
        # An unknown name cannot be run, so it takes no share.
        if name not in allowed:
            logger.warning(
                "stage-2 lenses for trace %s: dropped %r, which was not "
                "offered (offered: %s)", trace_id, name, ", ".join(sorted(allowed)))
            continue
        # A share must be a finite, non-negative real number.
        if (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0
        ):
            logger.warning(
                "stage-2 lenses for trace %s: dropped %r, whose value %r is not "
                "a finite non-negative number", trace_id, name, value)
            continue
        usable[name] = float(value)
    total = sum(usable.values())
    if total <= 0:
        logger.warning(
            "stage-2 lenses for trace %s: no usable shares left (sum is 0); "
            "no critique vector for this exchange", trace_id)
        return None
    return {name: value / total for name, value in usable.items()}


def _stage2_from_text(
    text: str, offered: list[str] | None = None, trace_id: str | None = None,
) -> dict:
    """Parse a stage-2 reply into its past claims; raises when the reply
    holds no JSON object.

    With ``offered`` (the critique lenses the fork was asked to score) the
    result also carries ``lenses``, the normalized critique vector, unless
    the reply has no usable one: a bad ``lenses`` field never costs the
    claims. Any other field the reply carries is ignored.
    """
    parsed = _parse_json_block(text)
    stage2: dict = {"past_claims": _past_claims(parsed.get("past_claims"))}
    # The critique vector, when lenses were offered; absent when unusable.
    if offered:
        vector = _lens_vector(parsed.get("lenses"), offered, str(trace_id))
        if vector is not None:
            stage2["lenses"] = vector
    return stage2


def _past_claims(raw) -> list[dict]:
    """Normalize stage 2's past_claims: a list of {claim, queries}.

    Tolerant by design: a missing field is no claims, and a malformed entry
    is dropped with a warning rather than failing the whole stage 2.
    Each claim keeps its non-empty string queries; everything is capped.
    """
    if raw is None:
        return []
    if not isinstance(raw, list):
        logger.warning(
            "stage-2 past_claims is a %s, not a list; treating it as no claims",
            type(raw).__name__,
        )
        return []
    claims: list[dict] = []
    dropped = 0
    for entry in raw:
        # An entry needs a non-blank claim; queries are optional.
        claim = entry.get("claim") if isinstance(entry, dict) else None
        if not isinstance(claim, str) or not claim.strip():
            dropped += 1
            continue
        queries = entry.get("queries")
        queries = [
            q.strip()[:MAX_QUERY_CHARS]
            for q in (queries if isinstance(queries, list) else [])
            if isinstance(q, str) and q.strip()
        ][:MAX_QUERIES_PER_CLAIM]
        claims.append({"claim": claim.strip(), "queries": queries})
    if dropped:
        logger.warning(
            "stage-2 past_claims: dropped %d malformed entr%s (need a "
            "non-empty \"claim\" string)", dropped, "y" if dropped == 1 else "ies",
        )
    return claims[:MAX_PAST_CLAIMS]


def entropy_summary(logprobs: dict | None) -> dict | None:
    """Summarize per-token entropy from an OpenAI-style logprobs envelope.

    Interoceptive, optional, never load-bearing (trap #4). Returns the
    ``entropy`` envelope value ``{"kind", "mean", "max", "n_tokens"}`` or
    None when no usable logprobs are present (absent ⇒ omit the key, never
    null it — RFC 7386).

    Per-token entropy is computed over the returned top-N logprobs plus a
    residual bucket ``p_resid = max(0, 1 - sum p_i)`` contributing
    ``-p_resid*log(p_resid)`` (N as provided, no re-request). If the payload
    lacks per-token alternatives (chosen-token-only), it falls back to
    ``-chosen_token_logprob`` (NLL) with ``kind == "nll"``.
    """
    if not logprobs:
        return None
    content = logprobs.get("content") if isinstance(logprobs, dict) else logprobs
    if not content:
        return None
    per_token: list[float] = []
    any_topn = False
    for tok in content:
        if not isinstance(tok, dict):
            continue
        tops = tok.get("top_logprobs")
        if tops:
            probs = [
                math.exp(alt["logprob"])
                for alt in tops
                if isinstance(alt, dict) and alt.get("logprob") is not None
            ]
            if not probs:
                continue
            any_topn = True
            ent = -sum(p * math.log(p) for p in probs if p > 0.0)
            resid = max(0.0, 1.0 - sum(probs))
            if resid > 0.0:
                ent += -resid * math.log(resid)
            per_token.append(ent)
        else:
            lp = tok.get("logprob")
            if lp is None:
                continue
            per_token.append(-lp)  # chosen-token NLL fallback
    if not per_token:
        return None
    return {
        "kind": "topn" if any_topn else "nll",
        "mean": sum(per_token) / len(per_token),
        "max": max(per_token),
        "n_tokens": len(per_token),
    }


def _declined(channel, response_text: str, trace_id: str | None) -> bool:
    """Whether this final is a decline (#420), failing toward appraising.

    A check that raises costs one stage-2 call rather than dropping one, and
    logs a warning so the break is visible.
    """
    try:
        return is_decline(channel, response_text)
    except Exception:
        logger.warning(
            "could not tell whether the final of %s on %s was declined; "
            "appraising it", trace_id, channel.id, exc_info=True,
            extra={"channel": channel.id, "trace_id": trace_id},
        )
        return False


class AppraisalPlugin(CorvidaePlugin):
    """Stage-2 appraisal: a fork of the window lists each finished
    exchange's past claims and scores the critique lenses on offer.

    Soft-uses the outcome-log writer; fail-soft when it is absent.
    """

    depends_on = frozenset({"persistence"})

    def __init__(self, pm=None) -> None:
        if pm is not None:
            self.pm = pm
        self.config: dict = {}
        # Callbacks awaited after each stage 2 lands (add_stage2_listener):
        # how CritiquePlugin hears past_claims without waiting on a task.
        self._stage2_listeners: list[Callable[..., Awaitable[None]]] = []
        # The critique lenses stage 2 scores (set_lens_offer): a provider of
        # (name, question) pairs, or None when no plugin offers any.
        self._lens_offer: Callable[[], list[tuple[str, str]]] | None = None
        # Trace-keyed LRU of originating text (the exchange's true
        # originating USER/notification text), bounded so long-running
        # exchanges don't leak memory. Fed by on_message_persisted.
        self._originating_text: collections.OrderedDict[str, str] = collections.OrderedDict()
        # Same discipline for the originating row's message_log rowid: a
        # stage-2 fork runs only while that row is still in the window.
        self._originating_rowid: collections.OrderedDict[str, int] = collections.OrderedDict()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @hookimpl
    def register_llm_stages(self, stages: dict[str, str]) -> None:
        stages["appraisal"] = "chat"

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)
        self.config = config
        # A removed key (#1185, #1189) configures nothing: say so by name.
        warn_removed_keys(config, _REMOVED_KEYS)
        self._fork_prompt = resolve_prompt(
            config, "appraisal.stage2_fork_prompt", DEFAULT_STAGE2_FORK_PROMPT
        )

    @hookimpl
    async def on_config_reload(self, config: dict) -> None:
        # Swap the config reference; it, the removed-key report and the
        # fork prompt are all reload needs.
        self.config = config
        warn_removed_keys(config, _REMOVED_KEYS)
        try:
            self._fork_prompt = resolve_prompt(
                config, "appraisal.stage2_fork_prompt", DEFAULT_STAGE2_FORK_PROMPT
            )
        except (OSError, ValueError) as exc:
            logger.warning(
                "appraisal: fork prompt reload failed (%s); keeping previous",
                exc,
            )

    @hookimpl
    async def on_message_persisted(self, channel, trace_id, rowid, text, annotations) -> None:
        """Record the exchange's true originating text in the bounded LRU.

        Fires once per originating item (user messages and
        notification-born traces alike), regardless of persistence
        outcome (rowid may be None).
        """
        self._originating_text[trace_id] = text
        self._originating_text.move_to_end(trace_id)
        while len(self._originating_text) > ORIGINATING_TEXT_LRU_MAXSIZE:
            self._originating_text.popitem(last=False)
        # The rowid, when persistence produced one (the fork containment check).
        if rowid is not None:
            self._originating_rowid[trace_id] = rowid
            self._originating_rowid.move_to_end(trace_id)
            while len(self._originating_rowid) > ORIGINATING_TEXT_LRU_MAXSIZE:
                self._originating_rowid.popitem(last=False)

    def get_originating_text(self, trace_id: str) -> str | None:
        """Pure reader for the exchange's true originating text, or None."""
        return self._originating_text.get(trace_id)

    def get_originating_rowid(self, trace_id: str) -> int | None:
        """Pure reader for the originating row's message_log rowid, or None."""
        return self._originating_rowid.get(trace_id)

    # ------------------------------------------------------------------
    # Readers
    # ------------------------------------------------------------------

    async def get_stage2(self, trace_id: str) -> dict | None:
        """Pure reader for the exchange's stored stage 2 (past claims and
        lens scores), or None."""
        return await self._read_vector(trace_id, "stage2")

    def add_stage2_listener(self, callback: Callable[..., Awaitable[None]]) -> None:
        """Register a coroutine function to hear each completed stage 2.

        Every enqueued stage 2 notifies each listener exactly once, awaited
        with keyword arguments ``channel, trace_id, response_text, annotations,
        stage2``, outside the appraisal attribution: ``stage2`` is the
        persisted stage 2, or ``None`` when none was produced (the fork
        missed, the reply was unusable, or the task raised). A listener must
        accept ``None``. Cancellation (shutdown) notifies no one.
        A listener's exception is logged and never reaches the others. This
        is a push, not a wait: a consumer that awaited stage 2 from a queued
        task would hold a TaskQueue worker the stage-2 task may need.
        """
        self._stage2_listeners.append(callback)

    def set_lens_offer(
        self, provider: Callable[[], list[tuple[str, str]]] | None,
    ) -> None:
        """Register the provider of the critique lenses stage 2 scores.

        Called by CritiquePlugin at start-up. The provider returns the
        ``(name, question)`` pairs on offer at the moment of each stage-2
        call; one provider, since critique is the only consumer.
        """
        self._lens_offer = provider

    async def _notify_stage2(self, channel, trace_id, response_text, annotations, stage2) -> None:
        """Await every stage-2 listener; one failure doesn't stop the rest."""
        for listener in list(self._stage2_listeners):
            try:
                await listener(
                    channel=channel, trace_id=trace_id, response_text=response_text,
                    annotations=annotations, stage2=stage2,
                )
            except Exception:
                logger.warning(
                    "stage-2 listener %r failed for trace %s", listener, trace_id,
                    exc_info=True, extra={"channel": channel.id, "trace_id": trace_id},
                )

    # ------------------------------------------------------------------
    # Stage 2 — LLM appraisal off the response path (WP2.5)
    # ------------------------------------------------------------------

    @hookimpl
    async def on_agent_response(
        self, channel, request_text, response_text, trace_id, annotations, logprobs,
    ) -> None:
        """Enqueue the silent stage-2 appraisal task (never on the response path).

        Fires once per exchange-ending turn, whatever its origin: a correction
        (``critique`` origin) is judged too, and its provenance check needs
        the claims stage 2 lists (#1031). Never for a decline (#420): the
        agent's choice to stay silent is not a response. The task is silent (``deliver=False``): its empty result triggers no
        on_notify and therefore no main-model turn (trap #10).
        """
        if trace_id is None:
            return
        if _declined(channel, response_text, trace_id):
            logger.debug(
                "stage-2 appraisal skipped for %s on %s: the turn was declined",
                trace_id, channel.id,
                extra={"channel": channel.id, "trace_id": trace_id},
            )
            return
        task_plugin = self.pm.get_plugin("task") if hasattr(self, "pm") else None
        queue = getattr(task_plugin, "task_queue", None)
        if queue is None:
            logger.debug("stage-2 appraisal skipped: task queue unavailable")
            return
        text_in = self.get_originating_text(trace_id) or request_text

        async def _work() -> str:
            # An unexpected Exception is a miss like any other: log it and
            # notify with None. Cancellation (shutdown) is a BaseException,
            # passes through, and notifies no one.
            try:
                stage2 = await self._run_stage2(
                    channel, trace_id, text_in, response_text, logprobs)
            except Exception:
                logger.warning(
                    "stage-2 appraisal failed unexpectedly for trace %s on %s; "
                    "listeners are told stage 2 is missing", trace_id, channel.id,
                    exc_info=True, extra={"channel": channel.id, "trace_id": trace_id},
                )
                stage2 = None
            # Every enqueued stage 2 notifies exactly once, with the result or
            # None on a miss. Listeners run after _run_stage2 has reset its
            # attribution, so their own work isn't billed to appraisal.
            await self._notify_stage2(channel, trace_id, response_text, annotations, stage2)
            return ""

        try:
            await queue.enqueue(Task(
                work=_work,
                channel=channel,
                description="appraisal:stage2",
                trace_id=trace_id,
                annotations=annotations,
                deliver=False,
                tool_call_id=None,
            ))
        except Exception:
            logger.warning(
                "failed to enqueue stage-2 appraisal task", exc_info=True,
                extra={"channel": channel.id, "trace_id": trace_id},
            )

    async def _run_stage2(
        self, channel, trace_id: str, originating_text: str | None,
        response_text: str | None, logprobs: dict | None,
    ) -> dict | None:
        """The silent task body: one fork of the window, merge-persist stage 2.

        Returns the stage 2, or None when none was produced: a fork miss
        skips the stage with a warning and makes no standalone call
        (PRD §3.8.5, #320), and the exchange gets no stage 2.

        Sets attribution inside the body (usage rows join to the exchange for
        WP2.10's per-band cost report). Malformed model output is logged and
        the exchange gets no stage 2 — no exception escapes (trap #10).
        """
        token = set_attribution(
            stage="appraisal", channel_id=channel.id, trace_id=trace_id
        )
        try:
            # A fork of the live window; a miss skips (logged inside).
            stage2 = await self._fork_stage2(
                channel, trace_id, originating_text, response_text
            )
            # A miss was already logged by _fork_stage2: the exchange gets
            # no stage 2 (no persist, no listeners, no standalone call).
            if stage2 is None:
                return None

            # Merge, never overwrite: a full-column write would erase the
            # envelope's other keys (a row written before #1185 also holds
            # stage1/stage1_out). Absent entropy is omitted, never nulled
            # (RFC 7386).
            merge: dict = {"stage2": stage2}
            entropy = entropy_summary(logprobs)
            if entropy is not None:
                merge["entropy"] = entropy
            outcome_log = self.pm.get_plugin("outcome_log")
            if outcome_log is not None:
                try:
                    await outcome_log.update_exchange(trace_id, appraisal=merge)
                except Exception:
                    logger.warning(
                        "stage-2 appraisal persist failed", exc_info=True,
                        extra={"trace_id": trace_id},
                    )
            return stage2
        finally:
            reset_attribution(token)

    def _skip_stage2(
        self, channel, trace_id: str, reason: str, **kwargs
    ) -> None:
        """Log that stage 2 was skipped, naming the stage, trace, channel and
        what was missing (PRD §3.8.5); the exchange gets no stage 2."""
        logger.warning(
            "stage-2 appraisal skipped for trace %s on channel %s: %s; "
            "the exchange gets no stage 2",
            trace_id, channel.id, reason,
            extra={"channel": channel.id, "trace_id": trace_id}, **kwargs,
        )

    async def _fork_stage2(
        self, channel, trace_id: str, originating_text: str | None,
        response_text: str | None,
    ) -> dict | None:
        """Stage 2 from a fork of the channel's live window, or None (a skip).

        Forks only when forks are on for "appraisal" and the exchange's
        originating row is still in the window (compaction hasn't dropped
        it). Reads the window at run time: if the user has said more since,
        the fork still sees the exchange, named by its opening message and
        the response the user was sent (which can differ from the window's
        last assistant message, e.g. the max-turns fallback). The check and
        Agent.fork's prompt render run with no await between them.
        Every miss (forks off, row unknown or out of the window, fork
        refused, failed, empty or unparseable) logs a skip warning and
        returns None; there is no standalone call.
        """
        # Forks off for this stage (or no agent plugin) is a miss.
        agent = fork_agent(self.pm if hasattr(self, "pm") else None, "appraisal")
        if agent is None:
            self._skip_stage2(
                channel, trace_id,
                "forks are not enabled for appraisal (agent.forks.enabled / agent.forks.stages)",
            )
            return None
        # The originating row must be known and still in the window.
        rowid = self.get_originating_rowid(trace_id)
        conv = getattr(channel, "conversation", None)
        if rowid is None:
            self._skip_stage2(
                channel, trace_id, "the originating message was not recorded or not persisted")
            return None
        if conv is None:
            self._skip_stage2(channel, trace_id, "no live window")
            return None
        if rowid not in window_db_ids(conv.messages):
            self._skip_stage2(channel, trace_id, "the exchange is not in the window")
            return None
        # Build the instruction from the static fork prompt and the exchange.
        prompt = self._fork_prompt
        instruction = (
            f"{prompt}\n\n"
            f"The exchange began with the message «{quote(originating_text)}», "
            f"and the response you sent began «{quote(response_text)}»."
        )
        # The critique lenses on offer, each with its question (none when no
        # provider is set or it raised: stage 2 then asks for the claims only).
        offer = self._current_offer(trace_id)
        extra_body = STAGE2_EXTRA_BODY
        if offer:
            instruction += (
                "\n\n[Critique selection] Which ONE of these self-checks is "
                "most worth running on this exchange, if any? Give each a "
                'percentage, plus "none", summing to 100.\n'
                + "\n".join(f"- {name}: {question}" for name, question in offer)
                + '\nInclude them in your JSON object as "lenses": {'
                + ", ".join(f'"{name}": 0' for name, _ in offer)
                + ', "none": 0}.'
            )
            extra_body = _extra_body_with_lenses([name for name, _ in offer])
        # Fork and parse; any failure or unusable reply is a miss.
        try:
            result = await agent.fork(
                channel, instruction, stage="appraisal", extra_body=extra_body
            )
            return _stage2_from_text(
                result.text, [name for name, _ in offer] or None, trace_id)
        except Exception as exc:
            self._skip_stage2(
                channel, trace_id,
                f"the fork failed or its reply was unusable ({type(exc).__name__})",
                exc_info=True,
            )
            return None

    def _current_offer(self, trace_id: str) -> list[tuple[str, str]]:
        """The lenses on offer for this stage-2 call; empty when none.

        A provider that raises costs the exchange its critique vector, not
        its claims: warn naming the trace and offer nothing.
        """
        if self._lens_offer is None:
            return []
        try:
            return list(self._lens_offer())
        except Exception:
            logger.warning(
                "stage-2 lens offer failed for trace %s; stage 2 asks for "
                "the claims only and the exchange gets no critique vector",
                trace_id, exc_info=True, extra={"trace_id": trace_id},
            )
            return []

    def _exchange_db(self):
        """Return the persistence connection, which can read exchange_log,
        or None when there is none."""
        persistence = self.pm.get_plugin("persistence") if hasattr(self, "pm") else None
        return getattr(persistence, "db", None)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _read_vector(self, trace_id: str, envelope_key: str) -> dict | None:
        """Read one key of the exchange's appraisal envelope in exchange_log.
        Returns None when there is none."""
        db = self._exchange_db()
        if db is None:
            return None
        try:
            async with db.execute(
                "SELECT appraisal FROM exchange_log WHERE trace_id = ?",
                (trace_id,),
            ) as cursor:
                row = await cursor.fetchone()
        except Exception:
            # Table may not exist yet (no exchange recorded) — not an error.
            return None
        if row is None or row[0] is None:
            return None
        try:
            return json.loads(row[0]).get(envelope_key)
        except (json.JSONDecodeError, AttributeError):
            logger.warning(
                "malformed appraisal envelope for exchange %s", trace_id
            )
            return None
