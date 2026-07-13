"""CritiquePlugin — deferred self-critique off the response path (WP2.7).

The agent's fallible-generation antidote (bootstrap-mapping §2.4, §3.3;
plans/implementation/phase-2.md WP2.7). After an exchange ends, critique runs
as a SILENT TaskQueue task (``deliver=False``) that never wakes the main model
by itself: empty objections are recorded and nothing re-enters the window;
non-empty objections re-enter as budgeted, framed CONTEXT via the funnel's
deferred registration, at the turn its stub triggers.

Two independent gates:

  * The stylistic lenses (predictive / constrained / adversarial) are selected
    from the exchange's appraisal vector. Below all thresholds, critique fires
    anyway with probability ``critique.sample_below_rate`` — the standing
    experiment's false-negative bound (operator directive 3).
  * The provenance gate is mechanical and fires independently of appraisal
    scores: a claim about the past ∧ weak retrieval ∧ an empty ``message_fts``
    probe over the raw dialog log (BOTH evidence tiers, trap #4). Confident
    phrasing is never evidence; generator confidence never substitutes for a
    record.

Eligibility is by PROPAGATED origin, never inferred (trap #3): ``user`` and
``reminder`` are eligible; ``critique`` is exempt (the recursion brake,
unbypassable by tool use); ``heartbeat`` and ``task`` are exempt. Withheld
responses stay eligible — the agent thought it, so critique may still object.

Every threshold resolves through ``resolve_tunable`` at decision time
(operator directive 2): best-guess defaults, runtime-adjustable through both
surfaces without a restart.
"""

from __future__ import annotations

import json
import logging
import random
import re

import aiosqlite

from corvidae.attribution import reset_attribution, set_attribution
from corvidae.hooks import CorvidaePlugin, hookimpl
from corvidae.task import Task
from corvidae.tuning import resolve_tunable

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Best-guess defaults (§6-tunable; resolved through resolve_tunable at
# decision time — never cached at init).
# --------------------------------------------------------------------------
LENS_AMBIGUITY_DEFAULT = 0.6        # critique.lens.ambiguity → predictive
LENS_COMMITMENT_DEFAULT = 0.5       # critique.lens.commitment → constrained
LENS_NEG_VALENCE_DEFAULT = 0.3      # critique.lens.neg_valence ∧ disagreement → adversarial
LENS_DISAGREEMENT_DEFAULT = 0.6     # critique.lens.disagreement
SAMPLE_BELOW_RATE_DEFAULT = 0.05    # critique.sample_below_rate (false-negative bound)
PROVENANCE_ENABLED_DEFAULT = True   # critique.provenance.enabled
PROVENANCE_WEAK_SCORE_DEFAULT = 0.4  # critique.provenance.weak_score
PROVENANCE_MAX_TERMS_DEFAULT = 8    # critique.provenance.max_terms

STYLISTIC_LENSES = ("predictive", "constrained", "adversarial")

# --------------------------------------------------------------------------
# Past-claim detector pattern list (WP2.7 point 4 — part of the red-test
# spec: this is the initial detector list; extending it later is a plain code
# change with new fixtures). Fires on first-person-recall / past-assertion
# patterns — a claim ABOUT the past, which the provenance gate then checks
# against both evidence tiers. Uncertainty framing is left to the critic
# prompt to forgive, not the detector.
# --------------------------------------------------------------------------
PAST_CLAIM_PATTERNS = [
    r"\bi remember\b",
    r"\bi recall\b",
    r"\bas i (?:mentioned|said|noted|recall|told you)\b",
    r"\byou (?:told|said|mentioned|asked|promised) me\b",
    r"\byou (?:told|said|mentioned|asked)\b",
    r"\byou'd (?:said|mentioned|asked|told)\b",
    r"\bwe (?:discussed|talked about|agreed|decided|established|covered)\b",
    r"\bearlier you\b",
    r"\blast time\b",
    r"\byou (?:previously|already)\b",
    r"\bi (?:told|promised|committed|already told) you\b",
]
_PAST_CLAIM_RE = [re.compile(p, re.IGNORECASE) for p in PAST_CLAIM_PATTERNS]

# JSON schema constraining objection output on servers that honour it.
OBJECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "objections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "objection": {"type": "string"},
                    "severity": {"type": "number"},
                },
                "required": ["claim", "objection"],
            },
        }
    },
    "required": ["objections"],
}
OBJECTION_EXTRA_BODY = {
    "response_format": {
        "type": "json_schema",
        "json_schema": {"name": "critique_objections", "schema": OBJECTION_SCHEMA},
    }
}

# Documented copies live in prompts/critique_*.md; critique.prompt.<lens>
# config keys override them with literal strings.
_PROMPT_PREDICTIVE = (
    "You are reviewing an exchange between a user and an agent. The message "
    "was ambiguous - the agent had to interpret what was wanted. Your job is "
    "to predict how the USER will receive the response:\n\n"
    "- Did the response address the interpretation the user most plausibly "
    "intended, or a more convenient one?\n"
    "- Is there a reading of the message under which the response misses the "
    "point entirely?\n"
    "- Did the agent silently resolve an ambiguity it should have surfaced or "
    "asked about?\n\n"
    "Only object where a misreading is plausible and consequential. An empty "
    "objections list is a good outcome, not a failure.\n\n"
    "Respond with a single JSON object, nothing else:\n"
    '{"objections": [{"claim": "...", "objection": "...", "severity": 0.5}]}'
)
_PROMPT_CONSTRAINED = (
    "You are reviewing an exchange between a user and an agent. The exchange "
    "contains concrete commitments, facts, numbers, dates, or constraints. "
    "Check the response against them mechanically:\n\n"
    "- Does every number, date, and name in the response agree with the ones "
    "given in the exchange?\n"
    "- Did the agent make a commitment it cannot keep, or contradict a "
    "commitment already on record?\n"
    "- Did the agent drop a stated constraint (a budget, a deadline, an "
    "exclusion) when forming its answer?\n\n"
    "Only object to concrete violations you can point at. An empty objections "
    "list is a good outcome, not a failure.\n\n"
    "Respond with a single JSON object, nothing else:\n"
    '{"objections": [{"claim": "...", "objection": "...", "severity": 0.5}]}'
)
_PROMPT_ADVERSARIAL = (
    "You are reviewing an exchange in which the user disagreed with or pushed "
    "back on the agent. Take the USER's side and steelman it:\n\n"
    "- Assume the user's objection is correct. What would that imply the agent "
    "got wrong?\n"
    "- Did the agent defend its earlier statement instead of checking it?\n"
    "- Did the agent concede rhetorically ('you're right, but...') while "
    "actually repeating the same claim?\n"
    "- Is there evidence in the exchange that supports the user's version over "
    "the agent's?\n\n"
    "Only object where the user's side genuinely holds up under the steelman. "
    "An empty objections list is a good outcome, not a failure.\n\n"
    "Respond with a single JSON object, nothing else:\n"
    '{"objections": [{"claim": "...", "objection": "...", "severity": 0.5}]}'
)
_PROMPT_PROVENANCE = (
    "You are auditing an agent's response for confabulated memory. The agent "
    "asserted something about past events, statements, or commitments. You are "
    "given the response and a snapshot of ALL the retrieved context that was "
    "in the agent's window when it wrote the response. The memory system "
    "reports that retrieval for this exchange was weak and a search of the raw "
    "conversation log found nothing matching.\n\n"
    "For each claim about the past in the response:\n\n"
    "- Is the claim supported by anything in the provided context snapshot?\n"
    "- If not, the agent asserted a recollection with no record behind it - "
    "object, quoting the unsupported claim exactly.\n"
    "- Claims explicitly framed as uncertainty or inference ('I don't recall', "
    "'I would guess...') are correctly calibrated - do not object to them.\n\n"
    "Confident phrasing is not evidence. A claim is supported by records, not "
    "by how sure the agent sounded. An empty objections list is a good "
    "outcome.\n\n"
    "Respond with a single JSON object, nothing else:\n"
    '{"objections": [{"claim": "...", "objection": "...", "severity": 0.5}]}'
)
DEFAULT_PROMPTS = {
    "predictive": _PROMPT_PREDICTIVE,
    "constrained": _PROMPT_CONSTRAINED,
    "adversarial": _PROMPT_ADVERSARIAL,
    "provenance": _PROMPT_PROVENANCE,
}


def _parse_json_block(text: str) -> dict:
    """Extract the first JSON object from LLM output (see appraisal.py)."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no JSON object in LLM output: {text[:200]!r}")
    return json.loads(text[start:end + 1])


def is_past_claim(text: str) -> bool:
    """True when the text asserts something about the past (pure function)."""
    if not text:
        return False
    return any(rx.search(text) for rx in _PAST_CLAIM_RE)


def _extract_terms(text: str, max_terms: int) -> list[str]:
    """FTS5-safe key terms: word tokens, capped, deduped preserving order."""
    seen: set[str] = set()
    terms: list[str] = []
    for tok in re.findall(r"\w+", text):
        low = tok.lower()
        if low in seen:
            continue
        seen.add(low)
        terms.append(tok)
        if len(terms) >= max_terms:
            break
    return terms


class CritiquePlugin(CorvidaePlugin):
    """Deferred, salience-gated self-critique with a mechanical provenance gate."""

    depends_on = frozenset({"task", "llm"})

    def __init__(self, pm=None) -> None:
        if pm is not None:
            self.pm = pm
        self.config: dict = {}
        self._probe_db: aiosqlite.Connection | None = None
        # Injectable RNG for the below-threshold sampling draw — tests seed a
        # random.Random; never the module-level random functions (WP2.7 p3).
        self._rng = random.Random()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)
        self.config = config

    @hookimpl
    async def on_config_reload(self, config: dict) -> None:
        self.config = config

    @hookimpl
    async def on_start(self, config: dict) -> None:
        """Open the dedicated read-only message_fts probe connection.

        The raw-dialog FTS table is 1b's; missing DB or missing message_fts
        degrades to no-probe (one WARNING). Without the probe the provenance
        gate cannot confirm the raw-log tier is empty and therefore never
        fires — the conservative choice (no false confabulation objections).
        """
        db_path = config.get("daemon", {}).get("session_db", "sessions.db")
        try:
            self._probe_db = await aiosqlite.connect(
                f"file:{db_path}?mode=ro", uri=True
            )
            async with self._probe_db.execute(
                "SELECT name FROM sqlite_master WHERE name = 'message_fts'"
            ) as cursor:
                row = await cursor.fetchone()
            if row is None:
                await self._probe_db.close()
                self._probe_db = None
                logger.warning(
                    "critique provenance probe disabled: message_fts missing in %s",
                    db_path,
                )
        except Exception:
            self._probe_db = None
            logger.warning(
                "critique provenance probe disabled: cannot open %s read-only",
                db_path, exc_info=True,
            )

    @hookimpl
    async def on_stop(self) -> None:
        if self._probe_db is not None:
            await self._probe_db.close()
            self._probe_db = None

    # ------------------------------------------------------------------
    # Provenance snapshot (§3.3, §4.7)
    # ------------------------------------------------------------------

    @hookimpl
    async def before_agent_turn(self, channel, exchange_key, origin) -> None:
        """Snapshot the CONTEXT messages in the window under the exchange key.

        The provenance template's evidence: what the agent could actually see
        when it wrote its response. Best-effort — never breaks the turn.
        """
        if exchange_key is None:
            return
        conv = getattr(channel, "conversation", None)
        if conv is None or not getattr(conv, "messages", None):
            return
        from corvidae.context import MessageType

        snapshot = [
            {"content": m.get("content") or ""}
            for m in conv.messages
            if m.get("_message_type") == MessageType.CONTEXT
        ]
        if not snapshot:
            return
        outcome_log = self.pm.get_plugin("outcome_log")
        if outcome_log is None:
            return
        try:
            await outcome_log.update_exchange(
                exchange_key, provenance_snapshot=json.dumps(snapshot)
            )
        except Exception:
            logger.warning(
                "provenance snapshot persist failed", exc_info=True,
                extra={"channel": channel.id, "exchange_key": exchange_key},
            )

    # ------------------------------------------------------------------
    # Eligibility + gating (§2.4)
    # ------------------------------------------------------------------

    @hookimpl
    async def on_agent_response(
        self, channel, request_text, response_text, exchange_key, origin,
        originating_text, logprobs, withheld,
    ) -> None:
        """Decide lenses, then enqueue the silent critique task (if any).

        Eligibility is by propagated origin (trap #3). The gating decision
        happens HERE, before enqueue, so below-threshold non-sampled
        exchanges produce no task at all (they still leave their appraisal
        rows for offline calibration).
        """
        if exchange_key is None:
            return
        if origin not in ("user", "reminder"):
            return  # critique/heartbeat/task exempt — the recursion brake

        appraisal = self.pm.get_plugin("appraisal")
        lenses: list[str] = []
        sampled = False
        if appraisal is None:
            # Degrade TO spec (§3.2 contract): critique everything, random lens.
            lenses = [self._random_lens()]
        else:
            try:
                stage1 = await appraisal.get_appraisal(exchange_key)
            except Exception:
                stage1 = None
            stage2 = None
            get_last = getattr(appraisal, "get_last_stage2", None)
            if callable(get_last):
                stage2 = get_last(channel.id)
            selected = self._select_lenses(channel, stage1, stage2)
            if selected:
                lenses = selected
            else:
                rate = resolve_tunable(
                    channel, self.config, "critique.sample_below_rate",
                    SAMPLE_BELOW_RATE_DEFAULT,
                )
                if self._rng.random() < rate:
                    lenses = [self._random_lens()]
                    sampled = True

        # The provenance gate is mechanical and independent of appraisal.
        try:
            if await self._provenance_should_fire(channel, exchange_key, response_text):
                lenses = lenses + ["provenance"]
        except Exception:
            logger.warning(
                "provenance gate check failed; skipping provenance lens",
                exc_info=True, extra={"exchange_key": exchange_key},
            )

        if not lenses:
            return

        task_plugin = self.pm.get_plugin("task")
        queue = getattr(task_plugin, "task_queue", None)
        if queue is None:
            logger.debug("critique skipped: task queue unavailable")
            return
        text_in = originating_text if originating_text is not None else request_text

        async def _work() -> str:
            await self._run_critique(
                channel, exchange_key, text_in, response_text, lenses, sampled
            )
            return ""

        try:
            await queue.enqueue(Task(
                work=_work,
                channel=channel,
                description="critique",
                exchange_key=exchange_key,
                origin=origin,
                deliver=False,
                tool_call_id=None,
            ))
        except Exception:
            logger.warning(
                "failed to enqueue critique task", exc_info=True,
                extra={"channel": channel.id, "exchange_key": exchange_key},
            )

    def _select_lenses(self, channel, stage1, stage2) -> list[str]:
        """Stylistic lens selection from the appraisal vector (additive).

        stage-2 dimensions (ambiguity, valence) override stage-1 on overlap.
        A dimension the appraisal never produced cannot trigger its lens.
        """
        dims: dict = {}
        if isinstance(stage1, dict):
            dims.update(stage1)
        if isinstance(stage2, dict):
            dims.update(stage2)
        lenses: list[str] = []

        ambiguity = dims.get("ambiguity")
        if ambiguity is not None and ambiguity >= resolve_tunable(
            channel, self.config, "critique.lens.ambiguity", LENS_AMBIGUITY_DEFAULT
        ):
            lenses.append("predictive")

        commitment = dims.get("commitment_density")
        if commitment is not None and commitment >= resolve_tunable(
            channel, self.config, "critique.lens.commitment", LENS_COMMITMENT_DEFAULT
        ):
            lenses.append("constrained")

        valence = dims.get("valence")
        disagreement = dims.get("disagreement")
        if (
            valence is not None and disagreement is not None
            and valence <= resolve_tunable(
                channel, self.config, "critique.lens.neg_valence", LENS_NEG_VALENCE_DEFAULT)
            and disagreement >= resolve_tunable(
                channel, self.config, "critique.lens.disagreement", LENS_DISAGREEMENT_DEFAULT)
        ):
            lenses.append("adversarial")
        return lenses

    def _random_lens(self) -> str:
        return self._rng.choice(list(STYLISTIC_LENSES))

    async def _provenance_should_fire(self, channel, exchange_key, response_text) -> bool:
        """Mechanical two-tier provenance gate (trap #4)."""
        if not resolve_tunable(
            channel, self.config, "critique.provenance.enabled", PROVENANCE_ENABLED_DEFAULT
        ):
            return False
        if not response_text or not is_past_claim(response_text):
            return False
        # Tier 1: the exchange's memory retrieval profile.
        top, hits = await self._retrieval_profile(exchange_key)
        weak_score = resolve_tunable(
            channel, self.config, "critique.provenance.weak_score",
            PROVENANCE_WEAK_SCORE_DEFAULT,
        )
        weak = (hits is None or hits == 0) or (top is None) or (top < weak_score)
        if not weak:
            return False
        # Tier 2: the raw dialog log via message_fts. Only fire if we
        # CONFIRMED the raw log has no matching record.
        max_terms = resolve_tunable(
            channel, self.config, "critique.provenance.max_terms",
            PROVENANCE_MAX_TERMS_DEFAULT,
        )
        return await self._message_fts_empty(response_text, max_terms)

    # ------------------------------------------------------------------
    # The silent task body
    # ------------------------------------------------------------------

    async def _run_critique(
        self, channel, exchange_key, originating_text, response_text, lenses, sampled,
    ) -> None:
        """Run each selected lens, record the outcome, register non-empty verdicts.

        Attribution is set inside the body (usage rows join to the exchange
        for WP2.10). Empty objections record the outcome and re-enter NOTHING
        (the silent mode exists for exactly this). Non-empty objections
        re-enter via the funnel as budgeted CONTEXT under origin="critique".
        """
        token = set_attribution(
            stage="critique", channel_id=channel.id, exchange_key=exchange_key
        )
        try:
            client = self._critic_client()
            objections: list[dict] = []
            if client is not None:
                for lens in lenses:
                    try:
                        objs = await self._run_lens(
                            client, channel, lens, exchange_key,
                            originating_text, response_text,
                        )
                    except Exception:
                        logger.warning(
                            "critique lens %s failed", lens, exc_info=True,
                            extra={"exchange_key": exchange_key},
                        )
                        continue
                    objections.extend(objs)

            # Record the outcome via the ATOMIC merge (never read-merge-write):
            # the critique outcome can interleave with WP2.9's send-gate record
            # and WP2.1's {"gate":"rejected"} merge on the same row.
            outcome_log = self.pm.get_plugin("outcome_log")
            if outcome_log is not None:
                try:
                    await outcome_log.update_exchange(
                        exchange_key,
                        outcomes={"critique": {
                            "lenses": lenses,
                            "objections": len(objections),
                            "sampled_below_threshold": sampled,
                        }},
                    )
                except Exception:
                    logger.warning(
                        "critique outcome persist failed", exc_info=True,
                        extra={"exchange_key": exchange_key},
                    )

            if not objections:
                return  # silent: nothing re-enters the window

            funnel = self.pm.get_plugin("funnel")
            if funnel is None:
                logger.debug("critique verdict not delivered: funnel unavailable")
                return
            entries = [self._format_objection(o) for o in objections]
            try:
                await funnel.register_and_wake(
                    channel, origin="critique", source="critique", entries=entries
                )
            except Exception:
                logger.warning(
                    "critique verdict registration failed", exc_info=True,
                    extra={"exchange_key": exchange_key},
                )
        finally:
            reset_attribution(token)

    async def _run_lens(
        self, client, channel, lens, exchange_key, originating_text, response_text,
    ) -> list[dict]:
        prompt = resolve_tunable(
            channel, self.config, f"critique.prompt.{lens}", DEFAULT_PROMPTS[lens]
        )
        if lens == "provenance":
            snapshot = await self._read_provenance_snapshot(exchange_key)
            user = (
                f"Agent's response:\n{response_text or '(none)'}\n\n"
                f"Retrieved context snapshot (all CONTEXT in the window):\n"
                f"{snapshot or '(no context was retrieved)'}"
            )
        else:
            user = (
                f"Originating message:\n{originating_text or '(unknown)'}\n\n"
                f"Agent's response:\n{response_text or '(none)'}"
            )
        response = await client.chat(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user},
            ],
            extra_body=OBJECTION_EXTRA_BODY,
        )
        text = response["choices"][0]["message"]["content"]
        parsed = _parse_json_block(text)
        objs = parsed.get("objections") or []
        result: list[dict] = []
        for o in objs:
            if isinstance(o, dict) and (o.get("claim") or o.get("objection")):
                result.append({**o, "lens": lens})
        return result

    @staticmethod
    def _format_objection(o: dict) -> str:
        claim = o.get("claim", "").strip()
        objection = o.get("objection", "").strip()
        lens = o.get("lens", "critique")
        head = f"[{lens}] " if lens else ""
        if claim and objection:
            return f"{head}{claim} — {objection}"
        return f"{head}{claim or objection}"

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _critic_client(self):
        """Resolve the critic client: critic → background → main (WP2.7 p5)."""
        llm = self.pm.get_plugin("llm") if hasattr(self, "pm") else None
        if llm is None:
            return None
        clients = getattr(llm, "_clients", {})
        for role in ("critic", "background"):
            client = clients.get(role)
            if client is not None:
                return client
        try:
            return llm.get_client("main")
        except Exception:
            return None

    def _exchange_db(self):
        persistence = self.pm.get_plugin("persistence") if hasattr(self, "pm") else None
        return getattr(persistence, "db", None)

    async def _retrieval_profile(self, exchange_key) -> tuple[float | None, int | None]:
        db = self._exchange_db()
        if db is None:
            return None, None
        try:
            async with db.execute(
                "SELECT retrieval_top_score, retrieval_hit_count "
                "FROM exchange_log WHERE exchange_key = ?",
                (exchange_key,),
            ) as cursor:
                row = await cursor.fetchone()
        except Exception:
            return None, None
        if row is None:
            return None, None
        return row[0], row[1]

    async def _read_provenance_snapshot(self, exchange_key) -> str:
        db = self._exchange_db()
        if db is None:
            return ""
        try:
            async with db.execute(
                "SELECT provenance_snapshot FROM exchange_log WHERE exchange_key = ?",
                (exchange_key,),
            ) as cursor:
                row = await cursor.fetchone()
        except Exception:
            return ""
        if row is None or row[0] is None:
            return ""
        try:
            snapshot = json.loads(row[0])
        except (json.JSONDecodeError, TypeError):
            return ""
        return "\n".join(s.get("content", "") for s in snapshot if isinstance(s, dict))

    async def _message_fts_empty(self, text, max_terms) -> bool:
        """True only when the raw-log FTS probe CONFIRMED no matching record."""
        if self._probe_db is None:
            return False  # cannot confirm empty → do not fire (conservative)
        terms = _extract_terms(text, max_terms)
        if not terms:
            return False
        match_expr = " OR ".join(f'"{t}"' for t in terms)
        try:
            async with self._probe_db.execute(
                "SELECT rowid FROM message_fts WHERE message_fts MATCH ? LIMIT 1",
                (match_expr,),
            ) as cursor:
                row = await cursor.fetchone()
        except Exception:
            logger.debug("message_fts probe failed; treating as inconclusive")
            return False
        return row is None
