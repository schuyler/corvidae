"""AppraisalPlugin stage 1 — gate appraisal, FTS5 probe, direction-keyed store.

Cheap, always-on perception that gates expensive cognition (bootstrap-mapping
§3.2; plans/implementation/phase-2.md WP2.4). Stage 1 is surface heuristics
plus an FTS5 familiarity probe on a **dedicated read-only** SQLite connection
(WAL permits concurrent readers; borrowing the persistence connection would
queue the gate behind consolidation writes), under a hard latency budget,
failing open — no probe result within budget means the vector is built from
surface heuristics alone (trap #1). Stage 1 never blocks and never calls a
model.

The stage-1 vector is produced by a pull-based compute
(:meth:`AppraisalPlugin.get_or_compute`), NOT by cross-hook ordering: apluggy
dispatches broadcast async hooks concurrently via ``asyncio.gather``, so a
push model would race the gate consumer against this plugin's write. Callers
pull; the compute runs once per ``(trace_id, direction)`` and caches.

Every parameter here is a commented best-guess constant, runtime-tunable
through both surfaces via ``corvidae.tuning.resolve_tunable`` (operator
directive 2, 2026-07-06).
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import math
import re
from collections.abc import Awaitable, Callable
from pathlib import Path

from corvidae.attribution import reset_attribution, set_attribution
from corvidae.fork import fork_agent, quote, window_db_ids
from corvidae.hooks import CorvidaePlugin, hookimpl
from corvidae.task import Task
from corvidae.tuning import config_value, resolve_tunable
from corvidae_cognition.probe import ReadOnlyProbe

# Named explicitly: corvidae/logging.py configures the "corvidae" logger,
# and __name__ would be corvidae_cognition.appraisal.
logger = logging.getLogger("corvidae.appraisal")

# --------------------------------------------------------------------------
# Best-guess defaults (§6-tunable; every one resolves through resolve_tunable
# at decision time — never cached at init).
# --------------------------------------------------------------------------
PROBE_BUDGET_MS_DEFAULT = 50          # appraisal.probe.budget_ms
PROBE_MAX_TOKENS_DEFAULT = 12         # appraisal.probe.max_tokens
PROBE_RANK_SCALE_DEFAULT = 10.0       # appraisal.probe.rank_scale (bm25 ranks are negative)
NOVELTY_NO_PROBE_DEFAULT = 0.5        # appraisal.novelty.no_probe_default — a value, never a null
WEIGHT_NOVELTY_DEFAULT = 0.35         # appraisal.weights.novelty
WEIGHT_QUESTION_DEFAULT = 0.15        # appraisal.weights.question
WEIGHT_DISAGREEMENT_DEFAULT = 0.20    # appraisal.weights.disagreement
WEIGHT_COMMITMENT_DEFAULT = 0.20      # appraisal.weights.commitment
WEIGHT_IMPERATIVE_DEFAULT = 0.10      # appraisal.weights.imperative

# In-memory store bound — constant, not tunable (stage 1 runs before enqueue,
# outside SerialQueue serialization, so per-channel slots would race; §3.2).
CACHE_MAXSIZE = 512

# Bound on the originating-text LRU (AppraisalPlugin._originating_text).
# Keyed by trace_id, not per-channel — user messages interleave
# mid-cycle by design, so a single per-channel slot would be clobbered.
ORIGINATING_TEXT_LRU_MAXSIZE = 512

# Stage-2 importance-prior weights (WP2.5 point 3). Best-guess commented
# defaults AND runtime tunables (directive 2), resolved at consolidation time.
PRIOR_W_STAKES_DEFAULT = 0.4      # appraisal.prior.w_stakes
PRIOR_W_VALENCE_DEFAULT = 0.3     # appraisal.prior.w_valence
PRIOR_W_NOVELTY_DEFAULT = 0.3     # appraisal.prior.w_novelty

# Documented copy of this text lives in prompts/appraisal.md; the
# appraisal.stage2_prompt config key overrides it with a literal string.
DEFAULT_STAGE2_PROMPT = (
    "You are appraising a completed exchange for the agent's own perception "
    "system. You are given the originating message, the agent's final "
    "response, and a summary of what the agent's memory retrieval found for "
    "this exchange. Score the EXCHANGE (not the agent's performance) on each "
    "dimension, 0.0-1.0.\n\n"
    "- valence: the emotional tone of the exchange for the agent. 0.0 = "
    "strongly negative (conflict, failure, distress), 0.5 = neutral, 1.0 = "
    "strongly positive (success, warmth, praise).\n"
    "- stakes: how much depends on this exchange being handled well. "
    "Commitments, deadlines, personal disclosures, decisions = high; idle "
    "chat = low.\n"
    "- ambiguity: how much of the message's intent remained open to "
    "interpretation. 0.0 = fully explicit; 1.0 = the agent had to guess.\n"
    "- commitment_density: how many concrete commitments, facts, numbers, or "
    "promises the exchange contains, relative to its length.\n"
    "- novelty: how new this exchange's content is relative to what the "
    "retrieval summary shows the agent already knows. Familiar ground = low.\n"
    "- correction: true if the user was correcting the agent — telling it "
    "that something it said, remembered, or did was wrong.\n"
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
    '  "valence": 0.5,\n'
    '  "stakes": 0.0,\n'
    '  "ambiguity": 0.0,\n'
    '  "commitment_density": 0.0,\n'
    '  "novelty": 0.5,\n'
    '  "correction": false,\n'
    '  "past_claims": [{"claim": "...", "queries": ["..."]}]\n'
    "}"
)

# Fork instruction for stage 2 (GitHub issue #51). With agent.forks on, the
# appraisal is a fork of the live window: this text is appended as a user
# turn, followed by a quote of the exchange's opening message and the
# retrieval profile. Second person: the fork is the agent appraising its own
# exchange. Documented copy in prompts/appraisal_fork.md; the static config
# key appraisal.stage2_fork_prompt (not a runtime tunable) overrides it.
DEFAULT_STAGE2_FORK_PROMPT = (
    "[Appraisal] Pause the conversation here and appraise, for your own "
    "perception system, the exchange you just completed: from the message "
    "quoted below through your final response. Score the EXCHANGE (not your "
    "performance) on each dimension, 0.0-1.0.\n\n"
    "- valence: the emotional tone of the exchange for you. 0.0 = strongly "
    "negative (conflict, failure, distress), 0.5 = neutral, 1.0 = strongly "
    "positive (success, warmth, praise).\n"
    "- stakes: how much depends on this exchange being handled well. "
    "Commitments, deadlines, personal disclosures, decisions = high; idle "
    "chat = low.\n"
    "- ambiguity: how much of the message's intent remained open to "
    "interpretation. 0.0 = fully explicit; 1.0 = you had to guess.\n"
    "- commitment_density: how many concrete commitments, facts, numbers, or "
    "promises the exchange contains, relative to its length.\n"
    "- novelty: how new this exchange's content is relative to what you "
    "already knew, going by the memory retrieval note below. Familiar "
    "ground = low.\n"
    "- correction: true if the user was correcting you, telling you that "
    "something you said, remembered, or did was wrong.\n"
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
    '  "valence": 0.5,\n'
    '  "stakes": 0.0,\n'
    '  "ambiguity": 0.0,\n'
    '  "commitment_density": 0.0,\n'
    '  "novelty": 0.5,\n'
    '  "correction": false,\n'
    '  "past_claims": [{"claim": "...", "queries": ["..."]}]\n'
    "}"
)

# JSON schema constraining the stage-2 call on servers that honour it
# (llama-server grammar / json_schema via extra_body). Providers that ignore
# it still return the JSON the prompt asks for; _parse_json_block reads it.
STAGE2_SCHEMA = {
    "type": "object",
    "properties": {
        "valence": {"type": "number"},
        "stakes": {"type": "number"},
        "ambiguity": {"type": "number"},
        "commitment_density": {"type": "number"},
        "novelty": {"type": "number"},
        "correction": {"type": "boolean"},
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
    "required": [
        "valence", "stakes", "ambiguity", "commitment_density",
        "novelty", "correction", "past_claims",
    ],
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

# Surface-heuristic marker sets (tier 1c). Deliberately crude: these are
# density signals, not NLP — the §6 standing experiment is what tunes the
# blend, not the marker lists.
_NEGATION_WORDS = frozenset({
    "no", "not", "never", "none", "nothing", "neither", "nor", "cannot",
    "without", "nowhere", "nobody",
})
_DISAGREEMENT_WORDS = frozenset({
    "wrong", "incorrect", "disagree", "actually", "however", "but", "nope",
    "false", "mistaken", "untrue", "error", "misremember", "misremembered",
})
_IMPERATIVE_LEADS = frozenset({
    "please", "do", "don't", "stop", "go", "make", "remember", "write",
    "check", "run", "tell", "give", "let", "take", "try", "use", "call",
    "send", "get", "put", "keep", "find", "look", "wait", "help", "add",
    "remove", "fix", "update", "list", "show", "explain",
})
_COMMITMENT_WORDS = frozenset({
    "will", "must", "promise", "deadline", "commit", "shall", "guarantee",
    "tomorrow", "tonight", "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday", "always", "definitely",
})

# Density → score scaling factors: how many marker hits per unit of text
# saturate the signal at 1.0. Best-guess constants (unit-tested at the
# boundaries, tuned by the §6 experiment if ever).
_NEGATION_SCALE = 4.0      # 25% negation tokens → 1.0
_DISAGREEMENT_SCALE = 5.0  # 20% disagreement tokens → 1.0
_COMMITMENT_SCALE = 5.0    # 20% commitment/number tokens → 1.0


def clamp01(x: float) -> float:
    """Clamp a float into [0.0, 1.0]."""
    return max(0.0, min(1.0, x))


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


def _stage2_from_text(text: str) -> dict:
    """Parse a stage-2 reply into the clamped vector; raises on bad output."""
    parsed = _parse_json_block(text)
    return {
        "valence": clamp01(float(parsed["valence"])),
        "stakes": clamp01(float(parsed["stakes"])),
        "ambiguity": clamp01(float(parsed["ambiguity"])),
        "commitment_density": clamp01(float(parsed["commitment_density"])),
        "novelty": clamp01(float(parsed["novelty"])),
        "correction": bool(parsed.get("correction", False)),
        "past_claims": _past_claims(parsed.get("past_claims")),
    }


def _past_claims(raw) -> list[dict]:
    """Normalize stage 2's past_claims: a list of {claim, queries}.

    Tolerant by design: a missing field is no claims, and a malformed entry
    is dropped with a warning rather than failing the whole stage-2 vector.
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


def _sentences(text: str) -> list[str]:
    """Split text into sentence-ish segments, keeping their terminators."""
    return [s.strip() for s in re.findall(r"[^.?!\n]+[.?!]?", text) if s.strip()]


def surface_signals(text: str) -> dict:
    """Score the tier-1c surface heuristics for a message, 0–1 each.

    Pure function: no model, no I/O. Returns
    ``{negation, question, imperative, disagreement, commitment}``.
    Empty/whitespace text scores all zeros (no division by zero).
    """
    words = re.findall(r"[a-z0-9']+", text.lower())
    sentences = _sentences(text)
    if not words or not sentences:
        return {
            "negation": 0.0, "question": 0.0, "imperative": 0.0,
            "disagreement": 0.0, "commitment": 0.0,
        }

    n_words = len(words)
    n_sentences = len(sentences)

    # Negation density: negation words plus n't contractions, per token.
    negation_hits = sum(
        1 for w in words if w in _NEGATION_WORDS or w.endswith("n't")
    )
    negation = clamp01(_NEGATION_SCALE * negation_hits / n_words)

    # Question density: fraction of sentences that end with a question mark.
    question_sentences = sum(1 for s in sentences if s.endswith("?"))
    question = clamp01(question_sentences / n_sentences)

    # Imperative markers: fraction of sentences opening with an
    # imperative-ish lead word.
    def _lead(s: str) -> str:
        m = re.match(r"[a-z']+", s.lower())
        return m.group(0) if m else ""

    imperative_sentences = sum(1 for s in sentences if _lead(s) in _IMPERATIVE_LEADS)
    imperative = clamp01(imperative_sentences / n_sentences)

    # Disagreement markers per token.
    disagreement_hits = sum(1 for w in words if w in _DISAGREEMENT_WORDS)
    disagreement = clamp01(_DISAGREEMENT_SCALE * disagreement_hits / n_words)

    # Numbers/commitment density: numeric tokens plus commitment words.
    commitment_hits = sum(
        1 for w in words if w in _COMMITMENT_WORDS or any(c.isdigit() for c in w)
    )
    commitment = clamp01(_COMMITMENT_SCALE * commitment_hits / n_words)

    return {
        "negation": negation,
        "question": question,
        "imperative": imperative,
        "disagreement": disagreement,
        "commitment": commitment,
    }


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


class _LRUDict(collections.OrderedDict):
    """A bounded dict that evicts its least-recently-used entry."""

    def __init__(self, maxsize: int) -> None:
        super().__init__()
        self._maxsize = maxsize

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value) -> None:
        super().__setitem__(key, value)
        self.move_to_end(key)
        while len(self) > self._maxsize:
            self.popitem(last=False)

    def get(self, key, default=None):
        # Route through __getitem__ so reads refresh recency — C-level
        # dict.get would bypass the override and silently degrade the LRU
        # to insertion order for pure readers.
        try:
            return self[key]
        except KeyError:
            return default


class AppraisalPrior:
    """Consolidation-time importance prior driven by stored appraisals (WP2.5).

    Wraps a fallback prior (the existing RubricPrior at install time). For a
    consolidation range it scores each covered exchange from its appraisal
    envelope and takes the max; when no appraisal covers the range — or the
    lookup fails — it delegates to the wrapped prior. Satisfies the
    ``ImportancePrior`` protocol with the two additive optional parameters.
    """

    def __init__(self, appraisal: "AppraisalPlugin", fallback) -> None:
        self._appraisal = appraisal
        self._fallback = fallback

    async def score(self, messages, msg_id_range=None, channel=None) -> float:
        if msg_id_range is not None:
            try:
                covered = await self._appraisal.importance_over_range(
                    channel, msg_id_range
                )
            except Exception:
                logger.warning(
                    "appraisal importance prior failed; using fallback",
                    exc_info=True,
                )
                covered = None
            if covered is not None:
                return covered
        return await self._fallback.score(
            messages, msg_id_range=msg_id_range, channel=channel
        )


class AppraisalPlugin(CorvidaePlugin):
    """Stage-1 appraisal: heuristics + FTS5 probe behind a pull API.

    Soft-uses the memory FTS surface (via its own read-only connection) and
    the outcome-log writer; fail-soft when either is absent.
    """

    depends_on = frozenset({"persistence"})

    def __init__(self, pm=None) -> None:
        if pm is not None:
            self.pm = pm
        self.config: dict = {}
        # Both maps keyed (trace_id, direction); direction ∈ {"in","out"}.
        # get_or_compute (inbound) and get_or_compute_out (outbound) share
        # this discipline — the two directions compute over different text
        # and must never alias.
        self._cache: _LRUDict = _LRUDict(maxsize=CACHE_MAXSIZE)
        self._inflight: dict[tuple[str, str], asyncio.Future] = {}
        # Strong refs to fire-and-forget persist tasks — prevents a
        # GC-dropped write mid-flight (trap #10).
        self._persist_tasks: set[asyncio.Task] = set()
        # Lazy memory_fts probe; built in on_start (None until then).
        self._probe: ReadOnlyProbe | None = None
        # Last-completed stage-2 vector per channel — the synchronous
        # advisory reader (get_last_stage2). Consumers never wait for the
        # CURRENT exchange's stage-2 (WP2.5 point 2).
        self._last_stage2: dict[str, dict] = {}
        # Callbacks awaited after each stage 2 lands (add_stage2_listener):
        # how CritiquePlugin hears past_claims without waiting on a task.
        self._stage2_listeners: list[Callable[..., Awaitable[None]]] = []
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
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)
        self.config = config

    @hookimpl
    async def on_config_reload(self, config: dict) -> None:
        # Swap the config reference; every tunable resolves through
        # resolve_tunable at decision time, so this is all reload needs.
        self.config = config

    @hookimpl
    async def on_start(self, config: dict) -> None:
        """Build the lazy read-only memory_fts probe (it opens nothing here).

        The DB path comes from the same config key PersistencePlugin reads.
        The connection opens on first use and retries at a bounded rate;
        while the DB or memory_fts is missing each attempt logs a WARNING
        and the vector is built from surface heuristics alone.
        """
        db_path = config.get("daemon", {}).get("session_db", "sessions.db")
        base_dir = config.get("_base_dir", Path("."))
        db_path = str(Path(base_dir) / db_path)
        # The connection opens lazily on first use and retries at a bounded
        # rate, so a memory_fts created after startup is picked up (#237).
        self._probe = ReadOnlyProbe(
            db_path, "memory_fts", "appraisal",
            "stage-1 novelty uses no_probe_default until memory_fts exists",
        )

        # Install the appraisal-driven importance prior over MemoryPlugin's
        # default (§3.2 consumer 3). Fail-soft when memory is absent — the
        # degradation contract keeps the system on the spec rubric.
        memory = self.pm.get_plugin("memory") if hasattr(self, "pm") else None
        if memory is not None:
            memory.importance_prior = AppraisalPrior(
                appraisal=self, fallback=memory.importance_prior
            )

    @hookimpl
    async def on_stop(self) -> None:
        # Close the probe connection; later get() calls open nothing.
        if self._probe is not None:
            await self._probe.close()
        # Let in-flight persists finish rather than abandoning writes. Loop
        # until the set drains: a compute finishing between snapshots can
        # spawn a persist task a single gather would miss (dropped write +
        # "task destroyed" warning at loop teardown).
        while self._persist_tasks:
            await asyncio.gather(*set(self._persist_tasks), return_exceptions=True)

    # ------------------------------------------------------------------
    # The thin gate trigger (computes; never decides)
    # ------------------------------------------------------------------

    @hookimpl
    async def should_process_message(self, channel, sender, text, trace_id):
        """Thin trigger: compute the inbound stage-1 vector, return None.

        The try/except is load-bearing, not defensive dressing: this is a
        plain broadcast hook, dispatched concurrently via asyncio.gather
        WITHOUT return_exceptions — a raw exception here would propagate to
        the transport read path, discard sibling impls' results, and
        prevent reject-wins from resolving. An appraisal failure never
        rejects or crashes the inbound path (traps #1/#10).
        """
        if trace_id is None:
            return None
        try:
            await self.get_or_compute(channel, trace_id, text)
        except Exception:
            logger.warning(
                "stage-1 appraisal failed at the gate (failing open)",
                exc_info=True,
                extra={"channel": channel.id, "trace_id": trace_id},
            )
        return None

    @hookimpl
    async def should_send_response(self, channel, text, emission, trace_id, meta):
        """Thin trigger: compute the outbound vector for a final, return None.

        So the vector exists even when no gate plugin is registered, or a
        gate's coroutine runs first (the pull API dedups the compute). The
        try/except is load-bearing for the same reason as the inbound
        trigger: one raising impl in this broadcast would discard every
        sibling's verdict. An appraisal failure never withholds.
        """
        if emission != "final" or trace_id is None:
            return None
        try:
            await self.get_or_compute_out(channel, trace_id, text)
        except Exception:
            logger.warning(
                "outbound stage-1 appraisal failed for %s (failing open)",
                trace_id, exc_info=True,
                extra={"channel": channel.id, "trace_id": trace_id},
            )
        return None

    @hookimpl
    async def on_message_persisted(self, channel, trace_id, rowid, text, meta) -> None:
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
    # Pull API
    # ------------------------------------------------------------------

    async def get_or_compute(self, channel, trace_id: str, text: str) -> dict:
        """Return the INBOUND stage-1 vector for this exchange, computing once.

        Idempotent and concurrency-safe: concurrent callers for the same key
        await a single shared in-flight future, so the probe runs exactly
        once per (trace_id, direction) regardless of how many hookimpls
        request it or what order they fire in. On first compute,
        fire-and-forget persists probe_score + the vector under the
        appraisal envelope's "stage1" key.

        Returned vectors are the SAME dict object that lives in the cache —
        treat them as immutable; copy on write if mutation is ever needed.

        Cross-task caller warning (2B review A-1): when the FIRST caller
        (the owner) is cancelled mid-compute, waiters sharing the in-flight
        future are woken with CancelledError — a BaseException that will
        sail through an ``except Exception`` fail-open handler. A caller
        awaiting from a DIFFERENT task than the owner (e.g. WP2.9's gates)
        must treat that CancelledError as compute failure, not as its own
        cancellation. Today owner and waiters only coexist inside one
        broadcast gather whose children cancel together, so this is
        unreachable; it stops being so the moment a cross-task caller
        appears.
        """
        return await self._get_or_compute(channel, trace_id, text, "in")

    async def get_or_compute_out(self, channel, trace_id: str, text: str) -> dict:
        """Return the OUTBOUND stage-1 vector for this exchange, computing once.

        The twin of get_or_compute over the final response text: the same
        heuristics + probe blend and the same fail-open probe budget (no
        model call), but its own (trace_id, "out") cache slot and in-flight
        future. A user exchange computes both directions under one trace
        id; sharing a slot would hand this caller the inbound vector and
        never compute or persist ``stage1_out``. First compute
        fire-and-forget persists it under the envelope's "stage1_out" key.
        The same cross-task CancelledError warning as get_or_compute applies.
        """
        return await self._get_or_compute(channel, trace_id, text, "out")

    async def get_appraisal(self, trace_id: str) -> dict | None:
        """Pure reader for the inbound stage-1 vector (no compute)."""
        return await self._read_vector(trace_id, "in", "stage1")

    async def get_appraisal_out(self, trace_id: str) -> dict | None:
        """Pure reader for the outbound stage-1 vector (no compute)."""
        return await self._read_vector(trace_id, "out", "stage1_out")

    async def get_stage2(self, trace_id: str) -> dict | None:
        """Pure reader for the stage-2 vector (WP2.5 writes it)."""
        return await self._read_vector(trace_id, None, "stage2")

    def get_last_stage2(self, channel_id: str) -> dict | None:
        """Synchronous advisory reader for the LAST completed stage-2 vector.

        Advisory context only (WP2.5 point 2): consumers (WP2.7 lens
        selection, WP2.9 outbound gate) must never wait for the CURRENT
        exchange's stage-2 to complete — it runs as a silent task that
        finishes after the turn. Returns None until one stage-2 lands.
        """
        return self._last_stage2.get(channel_id)

    def add_stage2_listener(self, callback: Callable[..., Awaitable[None]]) -> None:
        """Register a coroutine function to hear each completed stage 2.

        After a stage-2 vector is persisted, every listener is awaited with
        keyword arguments ``channel, trace_id, response_text, meta, stage2``,
        outside the appraisal attribution. A failed stage 2 notifies no one.
        A listener's exception is logged and never reaches the others. This
        is a push, not a wait: a consumer that awaited stage 2 from a queued
        task would hold a TaskQueue worker the stage-2 task may need.
        """
        self._stage2_listeners.append(callback)

    async def _notify_stage2(self, channel, trace_id, response_text, meta, stage2) -> None:
        """Await every stage-2 listener; one failure doesn't stop the rest."""
        for listener in list(self._stage2_listeners):
            try:
                await listener(
                    channel=channel, trace_id=trace_id, response_text=response_text,
                    meta=meta, stage2=stage2,
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
        self, channel, request_text, response_text, trace_id, meta, logprobs,
    ) -> None:
        """Enqueue the silent stage-2 appraisal task (never on the response path).

        Fires once per exchange-ending turn. Never for ``meta["origin"] ==
        "critique"`` exchanges — nothing downstream consumes them and it
        doubles cost. The task is silent (``deliver=False``): its empty
        result triggers no on_notify and therefore no main-model turn
        (trap #10).
        """
        origin = (meta or {}).get("origin")
        if trace_id is None or origin == "critique":
            return
        task_plugin = self.pm.get_plugin("task") if hasattr(self, "pm") else None
        queue = getattr(task_plugin, "task_queue", None)
        if queue is None:
            logger.debug("stage-2 appraisal skipped: task queue unavailable")
            return
        text_in = self.get_originating_text(trace_id) or request_text

        async def _work() -> str:
            stage2 = await self._run_stage2(channel, trace_id, text_in, response_text, logprobs)
            # Listeners run after _run_stage2 has reset its attribution, so
            # their own work isn't billed to appraisal.
            if stage2 is not None:
                await self._notify_stage2(channel, trace_id, response_text, meta, stage2)
            return ""

        try:
            await queue.enqueue(Task(
                work=_work,
                channel=channel,
                description="appraisal:stage2",
                trace_id=trace_id,
                meta=meta,
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
        """The silent task body: one tier-3 call, merge-persist stage 2.

        Returns the stage-2 vector, or None when no vector was produced.

        Sets attribution inside the body (usage rows join to the exchange for
        WP2.10's per-band cost report). Malformed model output is logged and
        the row keeps stage 1 only — no exception escapes (trap #10).
        """
        token = set_attribution(
            stage="appraisal", channel_id=channel.id, trace_id=trace_id
        )
        try:
            profile = await self._retrieval_profile_summary(trace_id)
            # A fork of the live window when one is available; else (and
            # on any fork miss) the standalone tier-3 call.
            stage2 = await self._fork_stage2(
                channel, trace_id, originating_text, response_text, profile
            )
            if stage2 is None:
                client = self._appraisal_client()
                if client is None:
                    logger.debug("stage-2 appraisal skipped: no LLM client available")
                    return None
                prompt = resolve_tunable(
                    channel, self.config, "appraisal.stage2_prompt", DEFAULT_STAGE2_PROMPT
                )
                user_content = (
                    f"Originating message:\n{originating_text or '(unknown)'}\n\n"
                    f"Agent's final response:\n{response_text or '(none)'}\n\n"
                    f"Memory retrieval for this exchange:\n{profile}"
                )
                standalone_token = set_attribution(mode="standalone")
                try:
                    response = await client.chat(
                        [
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": user_content},
                        ],
                        extra_body=STAGE2_EXTRA_BODY,
                    )
                    stage2 = _stage2_from_text(response["choices"][0]["message"]["content"])
                except Exception:
                    logger.warning(
                        "stage-2 appraisal call/parse failed; row keeps stage 1",
                        exc_info=True, extra={"trace_id": trace_id},
                    )
                    return None
                finally:
                    reset_attribution(standalone_token)

            # Merge, never overwrite: a full-column write would erase
            # WP2.9's stage1_out on every exchange that gets stage 2. Do NOT
            # re-pass stage1 (already in the envelope from WP2.4). Absent
            # entropy is omitted, never nulled (RFC 7386).
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
            # Advisory synchronous reader for the last completed stage-2.
            self._last_stage2[channel.id] = stage2
            return stage2
        finally:
            reset_attribution(token)

    async def _fork_stage2(
        self, channel, trace_id: str, originating_text: str | None,
        response_text: str | None, profile: str,
    ) -> dict | None:
        """Stage 2 from a fork of the channel's live window, or None.

        Forks only when forks are on for "appraisal" and the exchange's
        originating row is still in the window (compaction hasn't dropped
        it). Reads the window at run time: if the user has said more since,
        the fork still sees the exchange, named by its opening message and
        the response the user was sent (which can differ from the window's
        last assistant message, e.g. the max-turns fallback). The check and
        Agent.fork's prompt render run with no await between them.
        Any fork failure or unusable reply is logged and returns None.
        """
        agent = fork_agent(self.pm if hasattr(self, "pm") else None, "appraisal")
        if agent is None:
            return None
        rowid = self.get_originating_rowid(trace_id)
        conv = getattr(channel, "conversation", None)
        if rowid is None or conv is None or rowid not in window_db_ids(conv.messages):
            logger.debug(
                "stage-2 exchange not in the window; using the standalone call",
                extra={"trace_id": trace_id},
            )
            return None
        prompt = config_value(
            self.config, "appraisal.stage2_fork_prompt", DEFAULT_STAGE2_FORK_PROMPT
        )
        instruction = (
            f"{prompt}\n\n"
            f"The exchange began with the message «{quote(originating_text)}», "
            f"and the response you sent began «{quote(response_text)}».\n"
            f"Memory retrieval for this exchange: {profile}"
        )
        try:
            result = await agent.fork(channel, instruction, extra_body=STAGE2_EXTRA_BODY)
            return _stage2_from_text(result.text)
        except Exception:
            logger.warning(
                "stage-2 appraisal fork failed; using the standalone call",
                exc_info=True, extra={"trace_id": trace_id},
            )
            return None

    def _appraisal_client(self):
        """Resolve the tier-3 client: appraisal → background → main.

        LLMPlugin.get_client only falls back to main, so the two-step
        appraisal→background fallback is implemented here (WP2.5 point 2).
        """
        llm = self.pm.get_plugin("llm") if hasattr(self, "pm") else None
        if llm is None:
            return None
        clients = getattr(llm, "_clients", {})
        for role in ("appraisal", "background"):
            client = clients.get(role)
            if client is not None:
                return client
        try:
            return llm.get_client("main")
        except Exception:
            return None

    async def _retrieval_profile_summary(self, trace_id: str) -> str:
        """Render the exchange's retrieval profile as a short prompt line."""
        db = self._exchange_db()
        if db is None:
            return "no retrieval recorded"
        try:
            async with db.execute(
                "SELECT retrieval_top_score, retrieval_hit_count "
                "FROM exchange_log WHERE trace_id = ?",
                (trace_id,),
            ) as cursor:
                row = await cursor.fetchone()
        except Exception:
            return "no retrieval recorded"
        if row is None or (row[0] is None and row[1] is None):
            return "no retrieval recorded"
        top, hits = row
        if top is not None:
            return f"{hits or 0} memory hit(s); top relevance score {top:.2f}"
        return f"{hits or 0} memory hit(s)"

    # ------------------------------------------------------------------
    # Range readers (consolidation-time importance prior + valence, WP2.5)
    # ------------------------------------------------------------------

    async def importance_over_range(self, channel, msg_id_range) -> float | None:
        """Max importance over this channel's exchanges whose message_rowid
        is in the range (other channels' interleaved exchanges are ignored).

        Per-exchange score = max(stage1.salience, stage2_composite) when
        stage 2 is present, else stage-1 salience alone, else the exchange is
        skipped. Returns None when no appraisal covers the range (the caller
        falls back to the wrapped prior).
        """
        # Only this channel's exchanges: message_log ids are global.
        envelopes = await self._appraisals_in_range(channel.id, msg_id_range)
        if not envelopes:
            return None
        cfg = self.config
        w_stakes = resolve_tunable(channel, cfg, "appraisal.prior.w_stakes", PRIOR_W_STAKES_DEFAULT)
        w_valence = resolve_tunable(channel, cfg, "appraisal.prior.w_valence", PRIOR_W_VALENCE_DEFAULT)
        w_novelty = resolve_tunable(channel, cfg, "appraisal.prior.w_novelty", PRIOR_W_NOVELTY_DEFAULT)
        scores: list[float] = []
        for env in envelopes:
            stage1 = env.get("stage1") or {}
            stage2 = env.get("stage2")
            s1 = stage1.get("salience")
            s1 = float(s1) if s1 is not None else None
            if isinstance(stage2, dict):
                composite = clamp01(
                    w_stakes * float(stage2.get("stakes", 0.0))
                    + w_valence * abs(float(stage2.get("valence", 0.5)) - 0.5) * 2
                    + w_novelty * float(stage2.get("novelty", 0.0))
                )
                scores.append(max(s1, composite) if s1 is not None else composite)
            elif s1 is not None:
                scores.append(s1)
        if not scores:
            return None
        return max(scores)

    async def mean_valence(self, channel_id: str, msg_id_range) -> float | None:
        """Mean stage-2 valence over the range on this channel, or None when
        none present."""
        # Only this channel's exchanges: message_log ids are global.
        envelopes = await self._appraisals_in_range(channel_id, msg_id_range)
        valences: list[float] = []
        for env in envelopes:
            stage2 = env.get("stage2")
            if isinstance(stage2, dict) and stage2.get("valence") is not None:
                valences.append(float(stage2["valence"]))
        if not valences:
            return None
        return sum(valences) / len(valences)

    async def _appraisals_in_range(
        self, channel_id: str, msg_id_range
    ) -> list[dict]:
        """Return appraisal envelopes for this channel's exchanges whose
        message rowid is in range."""
        if not msg_id_range:
            return []
        lo, hi = msg_id_range
        db = self._exchange_db()
        if db is None:
            return []
        try:
            async with db.execute(
                "SELECT appraisal FROM exchange_log "
                "WHERE message_rowid IS NOT NULL AND message_rowid >= ? "
                "AND message_rowid <= ? AND appraisal IS NOT NULL "
                "AND channel_id = ?",
                (lo, hi, channel_id),
            ) as cursor:
                rows = await cursor.fetchall()
        except Exception as exc:
            # Rescue: treat as "no appraisals" but log it so the fallback
            # to the rubric prior / NULL valence is visible.
            logger.warning(
                "appraisal: exchange_log lookup failed for channel %s, "
                "messages %d–%d (%s); the record falls back to the rubric "
                "prior and NULL valence",
                channel_id, lo, hi, type(exc).__name__,
                exc_info=True, extra={"channel": channel_id},
            )
            return []
        envelopes: list[dict] = []
        for (appraisal_json,) in rows:
            try:
                env = json.loads(appraisal_json)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(env, dict):
                envelopes.append(env)
        return envelopes

    def _exchange_db(self):
        """Return a connection that can read exchange_log.

        Prefer the persistence write-connection (authoritative for
        exchange_log, and correct even when it is an in-memory DB the
        read-only probe file cannot see); fall back to the probe connection.
        """
        persistence = self.pm.get_plugin("persistence") if hasattr(self, "pm") else None
        db = getattr(persistence, "db", None)
        if db is not None:
            return db
        return self._probe.connection if self._probe is not None else None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _get_or_compute(self, channel, trace_id: str, text: str, direction: str) -> dict:
        k = (trace_id, direction)
        if k in self._cache:                    # 1. hit — return before any await
            return self._cache[k]
        if (fut := self._inflight.get(k)):      # 2. compute in progress: share it.
            # SHIELD is load-bearing: a cancelled waiter cancels only its
            # own wrapper, never the shared future; siblings are unharmed.
            return await asyncio.shield(fut)
        fut = asyncio.get_running_loop().create_future()
        # 3. Registered BEFORE the first await — THIS is the dedup:
        # concurrent callers for k now take branch 2; the probe runs once
        # per (key, direction).
        self._inflight[k] = fut
        try:
            vector, probe_score = await self._compute(channel, text, direction)
        except BaseException as exc:
            # 4. EVICT, never cache a failure — the next request retries.
            # BaseException, not Exception: OWNER cancellation
            # (CancelledError) must also evict and wake waiters, or they
            # hang forever on an abandoned future. The fut.exception() call
            # marks it retrieved (no GC-time warning). Fail-open is the
            # CALLERS' job (thin hookimpls, gates — traps #1/#10), never
            # this method's.
            del self._inflight[k]
            if not fut.done():
                fut.set_exception(exc)
                fut.exception()
            raise
        # 5. Cache, clear in-flight, THEN wake waiters — a waking waiter
        # observes a populated cache. done() guard: a cancelled future must
        # never raise InvalidStateError into the owner's return path.
        self._cache[k] = vector
        del self._inflight[k]
        if not fut.done():
            fut.set_result(vector)
        # 6. Fire-and-forget persist — the gate path never awaits it; the
        # strong ref prevents a GC-dropped write.
        t = asyncio.create_task(
            self._persist_stage1(channel, trace_id, direction, vector, probe_score)
        )
        self._persist_tasks.add(t)
        t.add_done_callback(self._persist_tasks.discard)
        return vector

    async def _compute(self, channel, text: str, direction: str) -> tuple[dict, float | None]:
        """Blend surface heuristics with the probe into the stage-1 vector.

        Applies the probe timeout internally and degrades to probe-less
        blending — a probe timeout is NOT a failure and never evicts the
        in-flight future (trap #1); only unexpected exceptions propagate.

        Returns (vector, probe_score) where probe_score is the familiarity
        the probe produced, or None when no probe result arrived in budget.
        """
        signals = surface_signals(text)
        cfg = self.config

        familiarity: float | None = None
        # The open (and its retry throttle) happens here, outside the
        # probe.budget_ms timeout, which covers only the query.
        probe_db = await self._probe.get() if self._probe is not None else None
        if probe_db is not None:
            budget_ms = resolve_tunable(
                channel, cfg, "appraisal.probe.budget_ms", PROBE_BUDGET_MS_DEFAULT
            )
            try:
                familiarity = await asyncio.wait_for(
                    self._probe_query(channel, text), timeout=budget_ms / 1000
                )
            except (TimeoutError, asyncio.TimeoutError):
                logger.debug("appraisal probe timed out; failing open")
            except asyncio.CancelledError:
                raise  # owner cancellation is not a probe degradation
            except Exception:
                logger.warning("appraisal probe failed; failing open", exc_info=True)

        # Signal→vector mapping (pinned in the plan's Stage-1 constants):
        # probe absent → no_probe_default, a value, never a null.
        if familiarity is not None:
            novelty = clamp01(1.0 - familiarity)
        else:
            novelty = resolve_tunable(
                channel, cfg, "appraisal.novelty.no_probe_default", NOVELTY_NO_PROBE_DEFAULT
            )
        question = signals["question"]
        disagreement = max(signals["disagreement"], signals["negation"])
        commitment = signals["commitment"]
        imperative = signals["imperative"]  # feeds salience only

        w_nov = resolve_tunable(channel, cfg, "appraisal.weights.novelty", WEIGHT_NOVELTY_DEFAULT)
        w_q = resolve_tunable(channel, cfg, "appraisal.weights.question", WEIGHT_QUESTION_DEFAULT)
        w_dis = resolve_tunable(channel, cfg, "appraisal.weights.disagreement", WEIGHT_DISAGREEMENT_DEFAULT)
        w_com = resolve_tunable(channel, cfg, "appraisal.weights.commitment", WEIGHT_COMMITMENT_DEFAULT)
        w_imp = resolve_tunable(channel, cfg, "appraisal.weights.imperative", WEIGHT_IMPERATIVE_DEFAULT)

        salience = clamp01(
            w_nov * novelty
            + w_q * question
            + w_dis * disagreement
            + w_com * commitment
            + w_imp * imperative
        )
        vector = {
            "novelty": novelty,
            "commitment_density": commitment,
            "disagreement": disagreement,
            "question": question,
            "salience": salience,
        }
        return vector, familiarity

    async def _probe_query(self, channel, text: str) -> float | None:
        """Run the FTS5 familiarity probe. Returns familiarity or None.

        Raw user text in MATCH is a syntax-error generator — each token is
        quoted and OR-joined, capped at appraisal.probe.max_tokens. The
        probe is FTS5-only by design: sqlite-vec is brute-force exact KNN,
        so a "coarse vector probe" would cost the same as full retrieval.
        """
        db = self._probe.connection if self._probe is not None else None
        if db is None:
            return None
        max_tokens = resolve_tunable(
            channel, self.config, "appraisal.probe.max_tokens", PROBE_MAX_TOKENS_DEFAULT
        )
        tokens = re.findall(r"\w+", text)[:max_tokens]
        if not tokens:
            return None
        match_expr = " OR ".join(f'"{t}"' for t in tokens)
        async with db.execute(
            "SELECT rank FROM memory_fts WHERE memory_fts MATCH ? ORDER BY rank LIMIT 3",
            (match_expr,),
        ) as cursor:
            rows = await cursor.fetchall()
        # Familiarity = bounded transform of top bm25 rank and hit count.
        # bm25 ranks are negative; more-negative = stronger match.
        if not rows:
            return 0.0
        hits = len(rows)
        top_rank = rows[0][0]
        rank_scale = resolve_tunable(
            channel, self.config, "appraisal.probe.rank_scale", PROBE_RANK_SCALE_DEFAULT
        )
        return clamp01((min(hits, 3) / 3) * min(1.0, -top_rank / rank_scale))

    async def _persist_stage1(
        self, channel, trace_id: str, direction: str, vector: dict,
        probe_score: float | None,
    ) -> None:
        """Fire-and-forget stage-1 persist via the atomic-merge upsert.

        At gate time the on_message_admitted/rejected insert has not fired
        yet, and a plain update_exchange would silently no-op on the
        missing row — hence the upsert. Catches and LOGS its own
        exceptions: a persist failure must never propagate anywhere near
        the gate path, but it must be visible in the log (trap #10).
        """
        try:
            outcome_log = self.pm.get_plugin("outcome_log")
            if outcome_log is None:
                logger.debug("stage-1 persist skipped: outcome_log not registered")
                return
            envelope_key = "stage1" if direction == "in" else "stage1_out"
            columns: dict = {"appraisal": {envelope_key: vector}}
            if probe_score is not None:
                columns["probe_score"] = probe_score
            # Inbound: origin "user" — the inbound gate only fires for user
            # messages. Outbound: None, because upsert_exchange patches
            # origin on EXISTING rows and the outbound compute runs for
            # every origin; the row already exists by the time a final is
            # gated (2B review A-4), so None leaves its origin alone.
            origin = "user" if direction == "in" else None
            await outcome_log.upsert_exchange(
                trace_id, channel.id, origin, **columns
            )
        except Exception:
            logger.warning(
                "stage-1 appraisal persist failed",
                exc_info=True,
                extra={"trace_id": trace_id, "direction": direction},
            )

    async def _read_vector(
        self, trace_id: str, direction: str | None, envelope_key: str
    ) -> dict | None:
        """Read a vector: in-memory store first, then the appraisal envelope
        in exchange_log. Returns None when neither has it."""
        if direction is not None:
            cached = self._cache.get((trace_id, direction))
            if cached is not None:
                return cached
        db = self._probe.connection if self._probe is not None else None
        if db is None:
            # Fall back to the persistence connection for the read; the
            # pure readers are not latency-critical.
            persistence = self.pm.get_plugin("persistence") if hasattr(self, "pm") else None
            db = getattr(persistence, "db", None)
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
