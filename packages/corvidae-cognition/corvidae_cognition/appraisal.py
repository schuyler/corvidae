"""AppraisalPlugin — the stage-2 appraisal of a finished exchange, and the
lens scores before a reply is sent.

Appraisal owns the lens contract (#1169): ``AppraisalSpec`` declares three
hooks, and peers such as critique implement them. ``on_appraisal_prompts``
collects the ``(lens, question)`` offers for a trigger ("idle" or
"before_send"); ``on_appraisal_before_send`` tells peers the scores of a
draft before it is sent; ``on_appraisal_idle`` tells them the scores and
past claims after the exchange.

After the agent answers, a silent task forks the live window and asks the
agent to list the claims about the past its reply made and to score the
lenses on offer; the claims and scores reach peers through
``on_appraisal_idle`` and are persisted to the outcome log, where the lens
scores are the labels a trained readout head learns from (#141). Before a
final reply is sent, ``should_send_response`` forks once on the draft to
score the before-send offers and fires ``on_appraisal_before_send``; the
objections peers return refuse the draft, which is held for one revising
turn (``corvidae_cognition.presend``, #1149). Stage-1 appraisal, which scored text at the gates from English word
lists and a probe of memory's full-text index, was removed (#1185); the
five-dimension appraisal vector, the correction flag, the importance prior
built from them and memory's valence tag were removed as unused (#1189).
"""

from __future__ import annotations

import collections
import json
import logging
import math

from corvidae.attribution import reset_attribution, set_attribution
from corvidae.fork import fork_agent, quote, window_db_ids
from corvidae_cognition.fork_log import ForkLog, digest_of
from corvidae.hooks import CorvidaePlugin, get_dependency, hookimpl, hookspec
from corvidae.task import Task, TaskPlugin
from corvidae.prompts import resolve_prompt
from corvidae_cognition.config_keys import warn_removed_keys
from corvidae_cognition.presend import PreSendHold
from corvidae_cognition.silence import decline_kind, is_decline

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

# Fork instruction for the before-send scoring fork. Appended as a user turn
# before the quoted draft and the offer block; asks only for the lens scores.
# Documented copy in prompts/appraisal_before_send_fork.md; the static config
# key appraisal.before_send_fork_prompt (not a runtime tunable) overrides it.
DEFAULT_BEFORE_SEND_FORK_PROMPT = (
    "[Appraisal] Pause the conversation here. You have drafted the reply "
    "quoted below but not sent it yet. For your own perception system, score "
    "the self-checks listed below against that draft: which one, if any, is "
    "most worth running on the reply before it goes out.\n\n"
    "Do not call tools. Respond with a single JSON object, nothing else, "
    'holding only the "lenses" object described below.'
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


def _extra_body_with_lenses(names: list[str], *, claims: bool = True) -> dict:
    """STAGE2_EXTRA_BODY with a required ``lenses`` object scoring ``names``
    and "none"; with ``claims`` False the reply holds the ``lenses`` object
    only (the before-send fork asks for no past claims). Built per call from
    a copy; the shared schema is untouched."""
    keys = [*names, "none"]
    # The claims part of the schema, or an empty object schema without it.
    base = STAGE2_SCHEMA if claims else {"type": "object", "properties": {}, "required": []}
    schema = {
        **base,
        "properties": {
            **base["properties"],
            "lenses": {
                "type": "object",
                "properties": {key: {"type": "number"} for key in keys},
                "required": keys,
            },
        },
        "required": [*base["required"], "lenses"],
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


def _lens_vector(
    raw: object, offered: list[str], trace_id: str, point: str = "stage-2",
) -> dict[str, float] | None:
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
            "%s lenses for trace %s: the reply has no \"lenses\" field; "
            "no critique vector for this exchange", point, trace_id)
        return None
    if not isinstance(raw, dict):
        logger.warning(
            "%s lenses for trace %s: \"lenses\" is a %s, not an object; "
            "no critique vector for this exchange", point, trace_id, type(raw).__name__)
        return None
    allowed = {*offered, "none"}
    usable: dict[str, float] = {}
    for name, value in raw.items():
        # An unknown name cannot be run, so it takes no share.
        if name not in allowed:
            logger.warning(
                "%s lenses for trace %s: dropped %r, which was not "
                "offered (offered: %s)", point, trace_id, name, ", ".join(sorted(allowed)))
            continue
        # A share must be a finite, non-negative real number.
        if (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0
        ):
            logger.warning(
                "%s lenses for trace %s: dropped %r, whose value %r is not "
                "a finite non-negative number", point, trace_id, name, value)
            continue
        usable[name] = float(value)
    total = sum(usable.values())
    if total <= 0:
        logger.warning(
            "%s lenses for trace %s: no usable shares left (sum is 0); "
            "no critique vector for this exchange", point, trace_id)
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


# The miss reason when forks are off for appraisal's stage.
FORKS_OFF_REASON = "forks are not enabled for appraisal (agent.forks.enabled / agent.forks.stages)"

# One lens on offer: (lens name, one-line question).
LensOffer = tuple[str, str]
# Shares over the offered lens names plus "none", summing to 1.
LensScores = dict[str, float]


def _offer_block(offer: list[LensOffer]) -> str:
    """The "[Critique selection]" block of a fork instruction: each offered
    lens as a ``- name: question`` line and the shape of the ``lenses`` reply."""
    return (
        "[Critique selection] Which ONE of these self-checks is "
        "most worth running on this exchange, if any? Give each a "
        'percentage, plus "none", summing to 100.\n'
        + "\n".join(f"- {name}: {question}" for name, question in offer)
        + '\nInclude them in your JSON object as "lenses": {'
        + ", ".join(f'"{name}": 0' for name, _ in offer)
        + ', "none": 0}.'
    )


class AppraisalSpec:
    """Hooks the appraisal plugin owns (#1169): the lens contract peers
    such as critique implement."""

    @hookspec
    async def on_appraisal_prompts(self, channel, trigger: str) -> list[LensOffer] | None:
        """Offer lenses for appraisal to score at ``trigger``.

        ``trigger`` is "idle" (after the exchange) or "before_send" (on a
        draft, before it is sent). Return ``(lens name, one-line question)``
        pairs, or None to offer nothing. Offers merge in hook-result order; a
        later duplicate name, a non-list result and a malformed pair are
        dropped with a warning.
        """

    @hookspec
    async def on_appraisal_before_send(
        self, channel, trace_id: str, text: str, annotations: dict,
        lenses: LensScores | None,
    ) -> list[str] | None:
        """Hear the scores of a final draft before it is sent.

        ``lenses`` is the normalized share per offered lens and "none", or
        None when the scoring fork missed. Return objections, a list of
        strings, or None. Appraisal concatenates them in hook-result order
        and refuses the draft with them; a draft held in time is revised
        once (#1149). A non-list result and a non-string item are dropped
        with a warning.
        """

    @hookspec
    async def on_appraisal_idle(
        self, channel, trace_id: str, response_text: str, annotations: dict,
        lenses: LensScores | None, past_claims: list[dict] | None,
    ) -> None:
        """Hear the scores and past claims of a finished exchange.

        Fires once per exchange appraisal ran on, after its fork lands and
        outside the appraisal attribution. ``lenses`` is the normalized
        critique vector (None when the fork gave none) and ``past_claims``
        the claims the reply made about the past (None when the fork missed).
        """


class AppraisalPlugin(CorvidaePlugin):
    """Stage-2 appraisal: a fork of the window lists each finished
    exchange's past claims and scores the lenses peers offer; the same
    scoring runs on a final draft before it is sent. Peers hear both
    through ``AppraisalSpec``'s hooks.

    Reports each stage 2 on ``on_trace_record`` (source ``appraisal``).
    """

    # trace owns the hook stage 2 is reported on (#1179).
    depends_on = frozenset({"persistence", "trace", "task", "held"})
    # Owns the on_appraisal_* hooks peers implement.
    hookspecs = AppraisalSpec

    def __init__(self, pm=None) -> None:
        if pm is not None:
            self.pm = pm
        self.config: dict = {}
        # Trace-keyed LRU of originating text (the exchange's true
        # originating USER/notification text), bounded so long-running
        # exchanges don't leak memory. Fed by on_message_persisted.
        self._originating_text: collections.OrderedDict[str, str] = collections.OrderedDict()
        # Same discipline for the originating row's message_log rowid: a
        # stage-2 fork runs only while that row is still in the window.
        self._originating_rowid: collections.OrderedDict[str, int] = collections.OrderedDict()
        # The pre-send hold-and-revise state (#1149); built on first use
        # because pm may arrive at on_init.
        self._presend_hold: PreSendHold | None = None

    @property
    def _hold(self) -> PreSendHold:
        """This plugin's ``PreSendHold``, created on first use."""
        if self._presend_hold is None:
            self._presend_hold = PreSendHold(self.pm, "appraisal")
        return self._presend_hold

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
        self._before_send_prompt = resolve_prompt(
            config, "appraisal.before_send_fork_prompt", DEFAULT_BEFORE_SEND_FORK_PROMPT
        )
        self._last_fork: dict[str, tuple[int, list[bytes]]] = {}
        self._fork_log = ForkLog(
            config.get("daemon", {}).get("session_db", "sessions.db"),
            sample_every=int(config.get("fork_log", {}).get("sample_every", 50)),
        )

    @hookimpl
    async def on_stop(self) -> None:
        log = getattr(self, "_fork_log", None)
        if log is not None:
            await log.close()

    async def _record_fork(
        self,
        *,
        channel,
        stage: str,
        point: str | None = None,
        blocks: list[dict],
        instruction: str,
        trace_id: str | None,
        result,
        question: str | None = None,
        findings: list[dict] | None = None,
    ) -> None:
        """Record one fork for #1204; never let a logging failure break it.

        ``blocks`` is the window the fork rendered. v1 hashes the window, not
        the fully rendered request: the tool schemas and the mid-conversation
        role are applied inside ``agent.fork`` and are not part of this hash.
        """
        log = getattr(self, "_fork_log", None)
        if log is None or not blocks:
            return
        try:
            blocks_in = [digest_of(b) for b in blocks]
            # Chain off the previous fork in this channel when the window only
            # grew: store the appended tail, not the whole array again.
            parent_id, parent_blocks = self._last_fork.get(channel.id, (0, []))
            shared = 0
            if parent_blocks:
                while (
                    shared < len(parent_blocks)
                    and shared < len(blocks_in)
                    and parent_blocks[shared] == blocks_in[shared]
                ):
                    shared += 1
            chained = parent_blocks and shared == len(parent_blocks)
            fork_id = await log.record(
                stage=stage,
                point=point,
                channel_id=channel.id,
                messages=blocks,
                digests=blocks_in,
                chain_parent=parent_id if chained else None,
                chain_from=shared if chained else 0,
                system_prompt=getattr(
                    getattr(channel, "conversation", None), "system_prompt", None
                ),
                instruction=instruction,
                question=question,
                findings=findings,
                verdict=getattr(result, "text", None),
                trace_id=trace_id,
                prompt_tokens=getattr(result, "prompt_tokens", None),
                completion_tokens=getattr(result, "completion_tokens", None),
            )
            self._last_fork[channel.id] = (fork_id, blocks_in)
        except Exception:
            logger.warning(
                "appraisal: could not record a %s fork for %s (#1204)",
                stage, channel.id, exc_info=True,
            )

    @hookimpl
    async def on_start(self, config: dict) -> None:
        """Warn once when forks are off for appraisal: with no fork it scores
        nothing, so peers hear no vector and no claims (#1169 R5.5)."""
        if fork_agent(self.pm if hasattr(self, "pm") else None, "appraisal") is None:
            logger.warning(
                "appraisal: forks are not enabled for the appraisal stage, so "
                "it scores no lenses and lists no claims, before or after the "
                "send; set agent.forks.enabled and add appraisal to "
                "agent.forks.stages"
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
            self._before_send_prompt = resolve_prompt(
                config, "appraisal.before_send_fork_prompt", DEFAULT_BEFORE_SEND_FORK_PROMPT
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

    def get_originating_rowid(self, trace_id: str) -> int | None:
        """Pure reader for the originating row's message_log rowid, or None."""
        return self._originating_rowid.get(trace_id)

    # ------------------------------------------------------------------
    # Readers
    # ------------------------------------------------------------------

    async def _collect_offers(self, channel, trigger: str) -> list[LensOffer]:
        """The lens offers peers make for ``trigger``, merged in hook-result order.

        Any plugin may implement the hook and its output is not trusted: a
        non-list result and a malformed pair are dropped with a warning, and
        a later duplicate name is dropped with a warning naming it.
        """
        offers: list[LensOffer] = []
        seen: set[str] = set()
        for result in await self.pm.ahook.on_appraisal_prompts(channel=channel, trigger=trigger):
            if result is None:
                continue
            if not isinstance(result, list):
                logger.warning(
                    "appraisal: on_appraisal_prompts(%s) result %r is a %s, not a "
                    "list of (name, question) pairs; dropped", trigger, result,
                    type(result).__name__)
                continue
            for pair in result:
                # A pair is two non-blank strings: the lens name, its question.
                if (
                    not isinstance(pair, (tuple, list)) or len(pair) != 2
                    or not all(isinstance(part, str) and part.strip() for part in pair)
                ):
                    logger.warning(
                        "appraisal: on_appraisal_prompts(%s) offered %r, which is not "
                        "a (name, question) pair of non-blank strings; dropped",
                        trigger, pair)
                    continue
                name, question = pair
                if name in seen:
                    logger.warning(
                        "appraisal: lens %r was offered twice for %s; the later "
                        "offer is dropped", name, trigger)
                    continue
                seen.add(name)
                offers.append((name, question))
        return offers

    # ------------------------------------------------------------------
    # Stage 2 — LLM appraisal off the response path (WP2.5)
    # ------------------------------------------------------------------

    @hookimpl
    async def before_agent_turn(self, channel, trace_id, annotations) -> None:
        """Tell the pre-send hold which turn is starting (#1149 R6)."""
        self._hold.on_turn(channel, trace_id, annotations)

    @hookimpl
    async def on_held_admitted(
        self, channel, source, trace_id, admitted_keys, duplicate_keys,
    ) -> None:
        """Tell the pre-send hold a batch was offered (#1149 R6)."""
        await self._hold.on_admitted(channel, source, admitted_keys, duplicate_keys)

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

        The pre-send hold learns the final's outcome first, before any early
        return, so a declined revision still clears; a held trace gets no
        stage 2 (#1149 R8).
        """
        # Hold outcome first: True means this trace was held, so skip stage 2.
        if await self._hold.on_final(channel, trace_id, response_text, annotations):
            return
        if trace_id is None:
            return
        if _declined(channel, response_text, trace_id):
            logger.debug(
                "stage-2 appraisal skipped for %s on %s: the turn was declined",
                trace_id, channel.id,
                extra={"channel": channel.id, "trace_id": trace_id},
            )
            return
        queue = get_dependency(self.pm, "task", TaskPlugin).task_queue
        if queue is None:
            logger.debug("stage-2 appraisal skipped: task queue unavailable")
            return
        text_in = self._originating_text.get(trace_id) or request_text

        async def _work() -> str:
            # An unexpected Exception is a miss like any other: log it and
            # fire the hook with None. Cancellation (shutdown) is a
            # BaseException, passes through, and fires nothing.
            try:
                stage2 = await self._run_stage2(
                    channel, trace_id, text_in, response_text, logprobs)
            except Exception:
                logger.warning(
                    "stage-2 appraisal failed unexpectedly for trace %s on %s; "
                    "peers are told stage 2 is missing", trace_id, channel.id,
                    exc_info=True, extra={"channel": channel.id, "trace_id": trace_id},
                )
                stage2 = None
            # Every enqueued stage 2 fires on_appraisal_idle exactly once, with
            # the result or None on a miss. It fires after _run_stage2 has
            # reset its attribution, so the peers' own work isn't billed to
            # appraisal.
            await self.pm.ahook.on_appraisal_idle(
                channel=channel, trace_id=trace_id,
                response_text=response_text, annotations=annotations,
                lenses=(stage2 or {}).get("lenses"),
                past_claims=None if stage2 is None else stage2["past_claims"],
            )
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
            # no stage 2 (no persist, no standalone call).
            if stage2 is None:
                return None

            # Report stage 2 on the trace (lens scores are #141's labels).
            # Absent entropy is omitted, never nulled. Observe dispatch logs
            # a listener's failure, naming the hook and the plugin.
            record: dict = {"phase": "idle", "stage2": stage2}
            entropy = entropy_summary(logprobs)
            if entropy is not None:
                record["entropy"] = entropy
            await self.pm.ahook.on_trace_record(
                trace_id=trace_id, channel_id=channel.id, source="appraisal",
                origin=None, record=record,
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

    def _fork_target(self, channel, trace_id: str):
        """The Agent to fork through, or the reason a fork cannot run.

        Forks only when forks are on for "appraisal" and the exchange's
        originating row is still in the window (compaction hasn't dropped
        it). Synchronous on purpose: callers run it with no await between it
        and ``agent.fork``, whose prompt render then sees the same window.
        A string result is the miss reason.
        """
        # Forks off for this stage (or no agent plugin) is a miss.
        agent = fork_agent(self.pm if hasattr(self, "pm") else None, "appraisal")
        if agent is None:
            return FORKS_OFF_REASON
        # The originating row must be known and still in the window.
        rowid = self.get_originating_rowid(trace_id)
        conv = getattr(channel, "conversation", None)
        if rowid is None:
            return "the originating message was not recorded or not persisted"
        if conv is None:
            return "no live window"
        if rowid not in window_db_ids(conv.messages):
            return "the exchange is not in the window"
        return agent

    async def _fork_stage2(
        self, channel, trace_id: str, originating_text: str | None,
        response_text: str | None,
    ) -> dict | None:
        """Stage 2 from a fork of the channel's live window, or None (a skip).

        Reads the window at run time: if the user has said more since,
        the fork still sees the exchange, named by its opening message and
        the response the user was sent (which can differ from the window's
        last assistant message, e.g. the max-turns fallback). The offers are
        collected first; the containment check (``_fork_target``) and
        Agent.fork's prompt render then run with no await between them.
        Every miss (forks off, row unknown or out of the window, fork
        refused, failed, empty or unparseable) logs a skip warning and
        returns None; there is no standalone call.
        """
        # The lenses on offer at the idle trigger, each with its question
        # (none when no peer offers: stage 2 then asks for the claims only).
        # Collected before the containment check: the dispatch awaits.
        offer = await self._collect_offers(channel, "idle")
        agent = self._fork_target(channel, trace_id)
        if isinstance(agent, str):
            self._skip_stage2(channel, trace_id, agent)
            return None
        # Build the instruction from the static fork prompt and the exchange.
        prompt = self._fork_prompt
        instruction = (
            f"{prompt}\n\n"
            f"The exchange began with the message «{quote(originating_text)}», "
            f"and the response you sent began «{quote(response_text)}»."
        )
        extra_body = STAGE2_EXTRA_BODY
        if offer:
            instruction += "\n\n" + _offer_block(offer)
            extra_body = _extra_body_with_lenses([name for name, _ in offer])
        # Fork and parse; any failure or unusable reply is a miss.
        try:
            result = await agent.fork(
                channel, instruction, stage="appraisal", extra_body=extra_body
            )
            conv = channel.conversation
            await self._record_fork(
                channel=channel,
                stage="appraisal",
                point="after-send",
                blocks=list(conv.messages) if conv is not None else [],
                instruction=instruction,
                trace_id=trace_id,
                result=result,
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

    # ------------------------------------------------------------------
    # Before send — the lens scores of a draft (#1169)
    # ------------------------------------------------------------------

    @hookimpl
    async def should_send_response(self, channel, text, emission, trace_id, annotations):
        """Score a final draft's lenses before it is sent; refuse on objections.

        A non-final, a decline or a trace-less draft is not checked, and
        neither is any final while a hold is outstanding on the channel (the
        revision goes out unchecked, #1149 R6). Otherwise peers hear the
        scores and the objections they return refuse the draft. Fails open:
        every Exception is caught and logged, and the answer is None.
        """
        try:
            # Not a checkable final, or a hold is outstanding: no opinion.
            if (
                emission != "final" or trace_id is None or decline_kind(text) is not None
                or self._hold.suppressed(channel.id)
            ):
                return None
            objections = await self._score_before_send(
                channel, text, trace_id, annotations)
            if objections:
                return await self._hold.refuse(
                    channel, trace_id, text, annotations, objections)
        except Exception:
            logger.warning(
                "before-send appraisal failed unexpectedly for trace %s on %s; "
                "allowing send", trace_id, channel.id, exc_info=True,
                extra={"channel": channel.id, "trace_id": trace_id},
            )
        return None

    async def _score_before_send(
        self, channel, text, trace_id, annotations,
    ) -> list[str]:
        """Fork once on the draft, fire ``on_appraisal_before_send`` and return
        the objection texts peers answered.

        No offers fires nothing. A miss after that (forks off, the exchange
        out of the window, the fork failing or unusable) logs a warning and
        fires the hook with ``lenses=None``. Reports the scores on
        ``on_trace_record`` with ``"phase": "before_send"``. Runs under
        attribution ``appraisal``.
        """
        token = set_attribution(
            stage="appraisal", channel_id=channel.id, trace_id=trace_id)
        try:
            # An empty offer is a configuration, not a miss.
            offer = await self._collect_offers(channel, "before_send")
            if not offer:
                logger.debug(
                    "before-send appraisal skipped for trace %s on %s: no lenses "
                    "are offered", trace_id, channel.id,
                    extra={"channel": channel.id, "trace_id": trace_id},
                )
                return []
            lenses = await self._fork_before_send(channel, trace_id, text, offer)
            # Report the scores (None on a miss, so a miss is countable).
            await self.pm.ahook.on_trace_record(
                trace_id=trace_id, channel_id=channel.id, source="appraisal",
                origin=None, record={"phase": "before_send", "lenses": lenses},
            )
            results = await self.pm.ahook.on_appraisal_before_send(
                channel=channel, trace_id=trace_id,
                text=text, annotations=annotations, lenses=lenses,
            )
            return self._collect_objections(results, trace_id, channel)
        finally:
            reset_attribution(token)

    def _collect_objections(self, results: list, trace_id: str, channel) -> list[str]:
        """The objection strings of the hook results, in hook-result order.

        Any plugin may implement the hook and its output is not trusted: a
        non-list result and a non-string item are dropped with a warning
        naming the trace, the channel and the result type.
        """
        objections: list[str] = []
        for result in results:
            if result is None:
                continue
            if not isinstance(result, list):
                logger.warning(
                    "appraisal: on_appraisal_before_send result for trace %s on "
                    "channel %s is a %s, not a list of objection strings; dropped. "
                    "Return a list of str (or None) from on_appraisal_before_send",
                    trace_id, channel.id, type(result).__name__)
                continue
            for item in result:
                if not isinstance(item, str):
                    logger.warning(
                        "appraisal: on_appraisal_before_send item for trace %s on "
                        "channel %s is a %s, not a string; dropped. "
                        "Return a list of str from on_appraisal_before_send",
                        trace_id, channel.id, type(item).__name__)
                    continue
                # A blank objection says nothing and would refuse with no reason.
                if item.strip():
                    objections.append(item)
        return objections

    async def _fork_before_send(
        self, channel, trace_id: str, text: str, offer: list[LensOffer],
    ) -> LensScores | None:
        """The lens scores of the draft from one fork of the window, or None
        (a logged miss).

        The instruction quotes the whole draft on one line, untruncated: it is
        the thing being scored and the window doesn't hold it yet. No claims
        are listed (provenance runs only after the exchange). The containment
        check and the fork have no await between them.
        """
        # Without its opening text the fork could not name the exchange.
        originating_text = self._originating_text.get(trace_id)
        if originating_text is None:
            self._skip_before_send(channel, trace_id, "the originating message was not recorded")
            return None
        agent = self._fork_target(channel, trace_id)
        if isinstance(agent, str):
            self._skip_before_send(channel, trace_id, agent)
            return None
        instruction = (
            f"{self._before_send_prompt}\n\n"
            f"The exchange began with the message "
            f"«{quote(originating_text)}». The reply you are "
            f"about to send, in full: «{quote(text, len(text))}».\n\n"
            + _offer_block(offer)
        )
        names = [name for name, _ in offer]
        try:
            result = await agent.fork(
                channel, instruction, stage="appraisal",
                extra_body=_extra_body_with_lenses(names, claims=False),
            )
            before_conv = channel.conversation
            await self._record_fork(
                channel=channel,
                stage="appraisal",
                point="before-send",
                blocks=list(before_conv.messages) if before_conv is not None else [],
                instruction=instruction,
                trace_id=trace_id,
                result=result,
            )
            return _lens_vector(
                _parse_json_block(result.text).get("lenses"), names, trace_id,
                point="before-send")
        except Exception as exc:
            self._skip_before_send(
                channel, trace_id,
                f"the fork failed or its reply was unusable ({type(exc).__name__})",
                exc_info=True,
            )
            return None

    def _skip_before_send(self, channel, trace_id: str, reason: str, **kwargs) -> None:
        """Log that the before-send scoring missed, naming the point, trace,
        channel and what was missing; peers are told there is no vector.

        Forks being off is stated once at ``on_start``, so per final it logs
        at DEBUG only (#1169 R5.5).
        """
        level = logging.DEBUG if reason == FORKS_OFF_REASON else logging.WARNING
        logger.log(level,
            "before-send appraisal skipped for trace %s on channel %s: %s; "
            "peers are told there is no lens vector",
            trace_id, channel.id, reason,
            extra={"channel": channel.id, "trace_id": trace_id}, **kwargs,
        )

