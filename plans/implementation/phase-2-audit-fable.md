# Phase 2 plan audit — ambiguities and staleness (Fable, 2026-07-11)

Design-tier audit of `phase-2.md` (full read, lines 1–2436) against source at
HEAD `1a15700` (WP2.1 landed; WP2.2/2.3 red tests written, green not started).
Scope: what an implementing agent would have to *guess* — underdetermined
decisions, cross-section contradictions, stale code references. NOT a design
review; the design is PASSED and its merits are not judged here.

Intermediate file — do not commit.

## Summary

- Critical: 0
- Important: 7 (I1–I7)
- Cosmetic: 7 (C1–C7)

The plan's prior consistency review + fix + re-review cycle (lines 1470–1600)
already closed the contradictions it found; this audit found no new
cross-section contradictions. The remaining findings are underdetermined
decisions (constants, formulas, one cross-plugin mechanism) and post-WP2.1
staleness. Verdict at bottom.

Code claims spot-checked and CONFIRMED (no action needed):
`get_client` falls back only to main (`llm_plugin.py:217` — the two-step
appraisal→background→main fallback is correctly assigned to the plugin);
`memory_fts` / `message_fts` both exist and are distinct (`memory.py:230,280`);
`funnel.admit` exists (`funnel.py:81`); `message_type` round-trips through
`load_conversation` and re-attaches `_message_type` (`persistence.py:67` —
WP2.8's "verify, don't assume" resolves to "no loader work needed beyond the
enum member"); non-summary rows are loaded regardless of type
(`persistence.py:142–153`), so WITHHELD rows will reload into the window;
`_run_turn` error fallback (`on_llm_error` → `send_message`) exists;
`send_progress` (`agent.py:412`), `send_thinking` step 8b (`agent.py:702–708`),
`MAX_TURNS_FALLBACK_MESSAGE` (`agent.py:58,425`) all match the WP2.8 firing-site
descriptions; `immutable_settings` blocklist mechanism exists
(`tools/settings.py:45–58`); `[project.entry-points."corvidae.commands"]`
exists in `pyproject.toml` (WP2.10's CLI registration pattern is real);
`retrieval_log.exchange_key` and all `exchange_log` Phase-2 columns present.

---

## Important

### I1 — WP2.4: stage-1 vector composition is underdetermined (signals → vector mapping, weights, familiarity transform)

**Plan refs:** WP2.4 points 1–4 (lines ~761–843).

**Problem.** Three distinct gaps compound into one: two implementers would
produce incompatible stage-1 vectors, and every downstream threshold
(`gate.engagement.threshold` 0.2, `gate.send.min_salience` 0.15, critique lens
gates) is calibrated against whichever composite the implementer invents.

1. `surface_signals(text)` scores five signals: *negation density, question
   density, imperative markers, disagreement markers, numbers/commitment
   density*. The stage-1 vector is `{novelty, commitment_density, disagreement,
   question, salience}`. The mapping is not stated: where do *negation* and
   *imperative* go? (Presumably folded into salience and/or disagreement — but
   that is a guess.)
2. "blended with the surface signals (weights are commented best-guess
   constants, `appraisal.weights.*` tunable)" — the key names under
   `appraisal.weights.*` and their default values are never enumerated.
   WP2.11 point 2 requires documenting "every `appraisal.*` key with its
   best-guess default," which is impossible to check against the plan.
3. "Familiarity = bounded transform of top bm25 rank and hit count (best-guess
   mapping, commented §6-tunable)" — no formula. bm25 rank in SQLite FTS5 is
   negative-better and unbounded; the bounding transform is load-bearing for
   `novelty = 1 - familiarity`.

**Proposed resolution (plan amendment, WP2.4).** Add a "Stage-1 constants"
block to WP2.4:

- Signal→vector mapping: `question` = question density; `disagreement` =
  max(disagreement markers, negation density); `commitment_density` =
  numbers/commitment density; `novelty` = `1 - familiarity` (probe absent →
  novelty falls back to `appraisal.novelty.no_probe_default` = 0.5);
  imperative markers feed salience only.
- `salience = clamp01(w_nov*novelty + w_q*question + w_dis*disagreement +
  w_com*commitment_density + w_imp*imperative)` with defaults
  `appraisal.weights.novelty` 0.35, `.question` 0.15, `.disagreement` 0.20,
  `.commitment` 0.20, `.imperative` 0.10 (best-guess, all runtime-tunable).
- Familiarity: `familiarity = clamp01((min(hits,3)/3) * norm(top_rank))` where
  `norm(r) = min(1.0, -r / appraisal.probe.rank_scale)` with
  `appraisal.probe.rank_scale` default 10.0 (bm25 ranks are negative;
  more-negative = stronger match). Zero hits → familiarity 0.0. All commented
  as §6-tunable best guesses.

Any concrete choice is fine — the amendment's job is that ONE choice is
written down before two agents make two.

### I2 — WP2.7: the adversarial lens's "disagreement high" has no named tunable or default

**Plan ref:** WP2.7 point 3 (line ~1053): "valence ≤ `critique.lens.neg_valence`
(0.3) ∧ disagreement high → adversarial."

**Problem.** The sibling conditions all have named keys and defaults
(`critique.lens.ambiguity` 0.6, `critique.lens.commitment` 0.5,
`critique.lens.neg_valence` 0.3). "disagreement high" has neither, and
directive 2 requires every gate parameter to be runtime-tunable — an unnamed
threshold can't be tuned. An implementer must invent both the key and value,
and WP2.11's config docs can't enumerate it.

**Proposed resolution.** Amend to: "valence ≤ `critique.lens.neg_valence` (0.3)
∧ disagreement ≥ `critique.lens.disagreement` (0.6) → adversarial", and add
`critique.lens.disagreement` to the WP2.11 key list.

### I3 — WP2.7/WP2.9: "previous stage 2" — which exchange, and how the reader learns it completed — is unspecified

**Plan refs:** WP2.7 point 3 (line ~1050 "stage 1 + previous stage 2 if
present"); WP2.9 point 3 (lines ~1281–1284 "the PREVIOUS exchange's stage 2
(… the plugin keeps a per-channel last-completed-stage-2 key …)").

**Problem.** Two gaps:

1. WP2.7's "previous stage 2" is ambiguous between (a) the current exchange's
   stage-2 (which by construction is racing the critique task — both are
   `on_agent_response`-triggered silent tasks) and (b) the previous exchange's
   completed stage-2, which is what WP2.9 explicitly means. An implementer
   reading WP2.7 alone can pick (a) and read a row the stage-2 writer hasn't
   populated yet — or worse, `await`-poll for it.
2. WP2.9 says "the plugin keeps a per-channel last-completed-stage-2 key" but
   stage-2 completion happens inside AppraisalPlugin's silent task body —
   GatePlugin has no way to observe it. No hook fires on stage-2 completion,
   and polling `exchange_log` per gate decision contradicts the
   cheap-perception framing. The keeper must be AppraisalPlugin, with a reader
   the consumers call — but that is inference, not specification.

**Proposed resolution.** Amend WP2.5 (the stage-2 writer) to add: "On stage-2
persist success, AppraisalPlugin records `(channel.id → exchange_key)` in a
per-channel `_last_stage2` dict and exposes
`get_last_stage2(channel_id) -> dict | None` (returns the stage-2 vector via
`get_stage2` for that key, else None). This is advisory context only — never
awaited, never computed on demand." Amend WP2.7 point 3 to read "…stage 1 +
the previous completed stage-2 via `appraisal.get_last_stage2(channel.id)`
(the CURRENT exchange's stage-2 is by construction unfinished — never wait for
it)" and WP2.9 point 3 to name the same reader.

### I4 — WP2.10: "appraisal band" is undefined

**Plan ref:** WP2.10 point 2 (line ~1368): "Reports, per channel and appraisal
band: …" (also WP2.1 point 4's "WP2.10's per-band cost join").

**Problem.** Retrieval bands exist in code (`memory.py:63–64`, strong 0.75 /
moderate 0.60) but those band memories by *similarity score*, not appraisal.
No definition of appraisal bands (over which scalar? what boundaries?) exists
anywhere in the plan or code. The calibrate report's primary grouping dimension
is uninventable without it.

**Proposed resolution.** Amend WP2.10 point 2: "Appraisal band = the exchange's
stage-1 `salience` bucketed at `[0, 0.2), [0.2, 0.5), [0.5, 1.0]` → low/medium/
high (best-guess boundaries, `calibrate` flags `--band-edges 0.2,0.5` to
override at report time; exchanges with no stage-1 vector report as band
`none`). Bands are a reporting construct only — nothing at runtime reads them."

### I5 — WP2.5: AppraisalPrior's "salience composite" inputs are ambiguous

**Plan ref:** WP2.5 point 3 (line ~958): "score = max per-exchange salience
composite (best-guess aggregation, commented)."

**Problem.** The stage-2 vector `{valence, stakes, ambiguity,
commitment_density, novelty}` has no `salience` key; `salience` is a stage-1
key. Whether the composite is (a) stage-1 salience alone, (b) a new composite
over stage-2 keys, or (c) a blend, is unspecified — and the choice materially
changes what consolidation considers important (stage-1 salience is
surface-heuristic only; stakes/valence exist only in stage-2).

**Proposed resolution.** Amend WP2.5 point 3: "Per-exchange score =
`max(stage1.salience, stage2_composite)` where `stage2_composite =
clamp01(w_stakes*stakes + w_valence*|valence - 0.5|*2 + w_novelty*novelty)`
when stage-2 is present, else stage-1 salience alone, else skip the exchange
(fall through to the wrapped RubricPrior when NO exchange in the range has
any appraisal). Weights are best-guess commented defaults AND runtime
tunables per directive 2: `appraisal.prior.w_stakes` 0.4,
`appraisal.prior.w_valence` 0.3, `appraisal.prior.w_novelty` 0.3, resolved
via `resolve_tunable` at consolidation time (pass the consolidating channel;
where no channel is in scope the config/default surfaces still apply —
`resolve_tunable` tolerates an object with empty `runtime_overrides`). Add
the three keys to WP2.11's list."

*(Revised per cold review — the original clause "not runtime-tunable in this
phase (consolidation is not a gate)" contradicted operator directive 2 and
is withdrawn.)*

### I6 — WP2.5: the logprob `entropy` summary has no formula and no envelope schema

**Plan ref:** WP2.5 point 2 (lines ~940–949): "a logprob entropy summary when
`AgentTurnResult.logprobs` arrived (mean/max token entropy over the response)"
persisted as `appraisal={"stage2": …, "entropy": entropy}`.

**Problem.** True token entropy needs the full distribution; providers return
top-N logprobs per token (llama-server: `logprobs` with `top_logprobs`
entries). The estimator (entropy over the truncated top-N? negative mean
chosen-token logprob?) and the `entropy` envelope's shape (scalar? dict?) are
unspecified. `entropy` is one of the four canonical envelope keys named in the
cross-sub-phase invariant (line ~504), so its shape is cross-WP surface, not a
private detail.

**Proposed resolution.** Amend WP2.5 point 2: "`entropy` envelope value is
`{"mean": float, "max": float, "n_tokens": int}` where per-token entropy is
computed over the returned top-N logprobs plus a residual bucket
(`p_resid = max(0, 1 - Σ p_i)` contributing `-p_resid*log(p_resid)`; N as
provided, no re-request). If the logprobs payload lacks per-token
alternatives (chosen-token-only), fall back to mean/max of
`-chosen_token_logprob` and note the estimator in a `"kind"` field
(`"topn" | "nll"`). Absent logprobs → omit the `entropy` key entirely (RFC 7386:
never null it)."

### I7 — WP2.8: the progress/thinking-veto "funnel marker … row annotation" is self-contradictory

**Plan ref:** WP2.8 point 4 (lines ~1168–1171): "A progress/thinking veto
suppresses that emission only; the message persists as ordinary MESSAGE, tool
dispatch proceeds, and a funnel marker notes the suppression (a row
annotation, never a message-type change — preserving the tool-pairing shape)."

**Problem.** The noun and the parenthetical point at two different mechanisms.
"A funnel marker" reads as the window-injection mechanism the final-veto path
uses (`admit()` of a one-line `source="gate"` marker); "a row annotation" reads
as a DB-side annotation on the persisted message row — but `message_log` has
no annotation column, and adding one is a schema change no WP declares. An
implementer must guess between (a) admit a window marker, (b) add a column,
(c) record it only in `exchange_log.outcomes`. The red-test bullet ("Progress
veto: … marker admitted") leans (a), but "row annotation" then has no referent.

**Proposed resolution.** Amend WP2.8 point 4: "…and the suppression is
recorded two ways: (1) the VETOING PLUGIN records it in its WP2.9 outcomes
write (fragment `outcomes={"suppressed": {"progress": true}}` /
`{"thinking": true}`, merged via the atomic upsert) — this is already covered
by WP2.9 point 3's 'every enforced veto and every pass is recorded' mandate;
core never writes the outcome log; (2) core `admit()`s a one-line
`source="gate"` marker into the window, same mechanism as the final-veto
marker — this is WP2.8's only recording responsibility. The persisted message
row itself is NEVER modified — no message-type change, no new column
(preserving the tool-pairing shape)." Delete the phrase "a row annotation."

*(Revised per cold review — the original part (1) assigned an `exchange_log`
write to WP2.8, a core-only WP; outcome-log writes are plugin territory and
WP2.9 already owns veto recording.)*

---

## Cosmetic

### C1 — Stale line numbers throughout 2B+ sections (pre-WP2.1 HEAD)

The plan cites `agent.py:442` (set_attribution — now ~`agent.py:532`),
`agent.py:565` (step-7 filter — now `agent.py:681`), `memory.py:819/:887`
(shifted), all verified against pre-WP2.1 HEAD `bb03fa5`. WP2.1's landing
moved them. **Resolution:** add one blanket note after the "Read first"
section: "Line numbers in this document were verified at HEAD `bb03fa5`
(pre-WP2.1). WP2.1 has landed; treat line refs as approximate and locate by
the named structure (step numbers, function names), not the line."

### C2 — `on_message_persisted` landed with 4 params; two later sections still show 3

The landed hookspec is `(channel, exchange_key, rowid, origin)` (amendment at
plan line 667; confirmed `hooks.py:558–560`). The 2A design-section hookspec
block (line ~1742) and WP2.11 point 1's hookspec list predate it.
**Resolution:** WP2.11 must document the landed 4-param signature; add a
parenthetical to WP2.11 point 1: "(as landed — `on_message_persisted` carries
`origin`; see WP2.1 implementation amendment)".

### C3 — WP2.5's `valence` column already exists; no migration

`memory.py:195` already ships `valence REAL` on the memory table ("NULL until
Phase 2 appraisal"). WP2.5 point 3 says consolidation "sets the record's
`valence` column" without noting the column pre-exists. **Resolution:** add the
same style of discrepancy note 2A used for `exchange_log`: "No schema migration
— the `valence` column exists at HEAD (`memory.py:195`); WP2.5 only writes it."
(Same pattern: stored embeddings for WP2.10's pairwise similarity also already
exist — `memory.py:205` `embedded` flag + vector store.)

### C4 — WP2.6: restart semantics of the pending-stub registry are unstated

`register_and_wake`'s payload queue and pending flags are (implicitly)
in-memory; a daemon restart between registration and drain silently drops
queued critique verdicts. Probably acceptable (verdicts are advisory,
fail-soft), but an implementer could reasonably decide to persist them.
**Resolution:** state it: "The registry is in-memory by design; payloads
pending at shutdown are dropped (critique verdicts are advisory — losing one
across a restart is acceptable). Do not persist."

### C5 — Hedged LRU sizes ("e.g. 512")

WP2.1's `originating_text` LRU and WP2.4's stage-1 store both say "e.g. 512".
WP2.1 landed with whatever was chosen; WP2.4's implementer needs a number, not
an example. **Resolution:** strike "e.g." — "bounded LRU, 512 entries
(constant, not tunable)".

### C6 — WP2.7 heuristic pattern lists are implementer-chosen; red tests must be authored against the implementer's own list

The provenance detector ("past-tense assertion + first-person-recall
patterns") and the key-term extraction for the `message_fts` probe are
deliberately heuristic, but the red-test author and green implementer may be
different agents — the red tests' fixture sentences must trip whatever
patterns the green agent ships. **Resolution:** add to WP2.7: "The pattern
list is part of the red-test spec: the red author defines the initial pattern
list (as a module-level constant spec in the test file's docstring) and writes
fixtures against it; green implements exactly that list. Extending patterns
later is a plain code change with new tests." (Same note applies to WP2.10's
correction-heuristic phrase list, which the plan already seeds with three
examples.)

### C7 — WP2.7: the below-threshold sampling RNG is not injectable by spec

**Plan ref:** WP2.7 point 3 and its red-test bullet ("except when the sampling
RNG (seeded) fires").

**Problem.** The red test requires deterministic control of the
`critique.sample_below_rate` draw, but the plan never says how the RNG is
exposed — module-level `random`, an instance attribute, or a parameter. Two
implementers will pick different seams and the red author can't write the test
first without choosing one.

**Resolution:** amend WP2.7: "CritiquePlugin holds `self._rng =
random.Random()`; tests inject a seeded `random.Random` (or monkeypatch the
attribute). The sampling draw is `self._rng.random() < rate` — never the
module-level `random` functions."

---

## Verdict

**Not safe to hand off as-is for 2B+ — amend first (half a day of plan
edits, no design changes).** Zero critical findings: nothing in the plan would
produce a *silently wrong* implementation. But I1–I7 are exactly the guess
points where parallel implementers diverge: I1 and I3 sit on the phase's two
highest-risk WPs (2.4, 2.9/2.7) and on cross-WP surfaces (the salience
composite every threshold reads; the stage-2 reader two plugins share). 2A
(WP2.2/2.3 green) is unaffected by every finding and can proceed immediately.

## Cold review (reviewer 1)

Reviewed 2026-07-11, no prior context. Every finding's plan citation re-read
against `phase-2.md` at its current text; every code claim re-verified against
the working tree (HEAD `1a15700`). Outcomes: 12 ACCEPT, 2 REVISE, 0 REJECT.
Two important problems in the audit's amendments (I5, I7); three cosmetic
inaccuracies in the audit's own claims.

### Verification of code claims (audit spot-check list)

All confirmed: `on_message_persisted(channel, exchange_key, rowid, origin)`
(`hooks.py:558–560`); `memory_fts`/`message_fts` distinct; `funnel.admit`
present; `_parse_message_rows` re-attaches `_message_type` and non-summary
rows reload (`persistence.py:60–70,142–155`); `get_client` falls back only to
main (`llm_plugin.py:217`); `MAX_TURNS_FALLBACK_MESSAGE` (`agent.py:58,425`);
`send_thinking` step 8b (`agent.py:702–708`); `valence REAL` and `embedded`
pre-exist in the memory DDL; retrieval bands 0.75/0.60 in `memory.py`;
`message_log` DDL has NO annotation column (I7's premise);
`ORIGINATING_TEXT_LRU_MAXSIZE = 512` landed (`agent.py:63` — C5's proposed
"512" matches the landed value); step-7 filter now `agent.py:681`, turn
`set_attribution` now `agent.py:532` (C1's claims). No fabricated or stale
code claim found in the audit.

### Per-finding outcomes

**I1 — ACCEPT.** All three gaps verified at WP2.4 points 1–4 (lines 760–846):
the five surface signals vs. the five-key stage-1 vector have no stated
mapping; `appraisal.weights.*` keys/defaults are never enumerated (and WP2.11
point 2 does require documenting them); the familiarity transform has no
formula while bm25's negative-unbounded rank makes the bound load-bearing.
Amendment checks out: weights sum to 1.0; `no_probe_default` 0.5 is a value,
not a null (consistent with the line-504 omit-not-null invariant and the
WP2.4 "probe-less novelty" fail-open red test); rank polarity stated
correctly; all keys fall under WP2.11's blanket "every `appraisal.*` key."
Ready to apply verbatim.

**I2 — ACCEPT.** Verified at line 1053: "disagreement high" is the only lens
condition with no key and no default; directive 2's letter covers critique
parameters. Amendment is minimal, matches the siblings' form, and names the
WP2.11 addition. Ready verbatim.

**I3 — ACCEPT** (one cosmetic wording nit). Both gaps verified: WP2.7 line
1050 says "previous stage 2" without WP2.9's "PREVIOUS exchange's"
qualifier, and WP2.9 line ~1282's "the plugin keeps a per-channel
last-completed-stage-2 key" has an ambiguous referent in a GatePlugin
paragraph while only AppraisalPlugin's silent-task body can observe stage-2
completion (no hook fires on it; `deliver=False` tasks end silently).
Assigning the keeper to AppraisalPlugin matches the plan's own "WP2.9
decides, it does not appraise" division. Nit (cosmetic): "never awaited" sits
oddly next to a reader that goes through `get_stage2` (async, DB fallback) —
consumers do `await` the *call*; the intent is "never wait for the current
exchange's stage-2 to complete." Either rephrase, or have `_last_stage2`
store the vector itself (making the reader sync). Not blocking.

**I4 — ACCEPT.** Verified: `grep -n band phase-2.md` — the only band
occurrences are the WP2.10/WP2.1 per-band mentions and the retrieval-band
"Read first" item; nothing defines an appraisal band, and the retrieval
bands (similarity 0.75/0.60) are a different scalar. Amendment is concrete,
report-only (consistent with calibrate's nothing-writes rule), handles the
no-vector case, and its 0.2 low edge aligns with `gate.engagement.threshold`.
Ready verbatim.

**I5 — REVISE (important).** The finding is valid: verified at line 958 —
"salience composite" over a stage-2 vector that has no salience key is
underdetermined, and the (a)/(b)/(c) choice materially changes consolidation.
The formula and fallback chain are fine and match the plan's existing
RubricPrior fallback wording. But the closing clause — "not runtime-tunable
in this phase (consolidation is not a gate)" — contradicts operator
directive 2's letter: "**Every gate/appraisal/critique parameter** is
runtime-adjustable without a daemon restart" (plan lines 65–68). The
AppraisalPrior composite weights are appraisal parameters, and the plan's own
precedent (WP2.4's blend weights: "commented best-guess constants,
`appraisal.weights.*` tunable") treats exactly this kind of weight as
tunable. Downstream agents are explicitly told not to "correct" the plan
against the directives; an amendment must not hand them a sentence that does.
Fix: either name the weights (e.g. `appraisal.prior.w_stakes` 0.4,
`.w_valence` 0.3, `.w_novelty` 0.3, resolved via `resolve_tunable` at
consolidation time, added to WP2.11's key list) or, minimally, delete the
"not runtime-tunable…" clause and leave them commented constants without an
explicit anti-directive ruling. Rest of the amendment stands.

**I6 — ACCEPT** (one cosmetic nit). Verified at lines 939–949: "mean/max
token entropy" with no estimator, and `entropy` is indeed one of the four
canonical envelope keys at line 504, so its shape is cross-WP surface. The
residual-bucket estimator, the chosen-token-NLL fallback, and omit-don't-null
(matching the RFC 7386 invariant) are all sound and concrete. Nit: say
explicitly that `"kind"` is always present (`"topn"` in the normal case),
not only on fallback — the current sentence attaches it to the fallback
branch. One-word fix at apply time.

**I7 — REVISE (important).** The finding is fully valid: verified at lines
1168–1171 — "a funnel marker … (a row annotation, …)" names two mechanisms;
`message_log` has no annotation column (verified DDL: channel_id, message,
timestamp, message_type); the red test's "marker admitted" leans window-
marker. The amendment's two-way recording is the right resolution and its
part (2) matches the final-veto path exactly. But part (1) as drafted amends
WP2.8 — a core-only WP (hooks/context/persistence/agent) — to perform an
`exchange_log` outcomes upsert. Everywhere else in the plan, core never
writes the outcome log: core fires hooks and OutcomeLogPlugin/GatePlugin
write (WP2.1 point 7, WP2.9 point 3). WP2.9 point 3 already mandates "every
enforced veto and every pass is recorded into `outcomes` … via the upsert
path" — the decision record for a progress/thinking veto is the vetoing
gate plugin's job, not the firing site's. As written, an implementer of
WP2.8 would have to import an outcome-log handle into `agent.py`, a layering
change no WP declares — recreating the same class of ambiguity the finding
fixes. Fix: reword part (1) to "the vetoing plugin records the suppression
in its WP2.9 outcomes write (fragment
`outcomes={"suppressed": {"progress": true}}` / `{"thinking": true}`,
merged via the atomic upsert)" and keep WP2.8's core responsibility as
part (2)'s marker admit only. Part (2) and the "row never modified" sentence
stand.

**C1 — ACCEPT** (with one inaccuracy in the audit, cosmetic). `agent.py:442`
→ 532 and `:565` → 681 confirmed. But `memory.py:819`
(`before_agent_turn`) is still at 819 in the working tree — only the
retrieval INSERT moved (887 → 895). The blanket resolution note is still the
right fix and is unaffected; the audit's "all … moved" claim is merely
overstated.

**C2 — ACCEPT** (with one inaccuracy in the audit, cosmetic). The stale
3-param block at plan line ~1742 is confirmed. However, WP2.11 point 1 lists
hookspec *names* without signatures — it does not "show 3"; the audit
overstates. The proposed WP2.11 parenthetical is still worthwhile (it is
where doc authors will look), but the resolution leaves the line-1742 block
— the actual stale signature it cites — untouched. Suggest the fix also
append "(landed with a 4th param, `origin` — see line-667 amendment)" at the
1742 block. Trivially fixable at apply time.

**C3 — ACCEPT.** `valence REAL — "NULL until Phase 2 appraisal"` and the
`embedded` flag verified in the memory DDL. Resolution matches the plan's
existing discrepancy-note style.

**C4 — ACCEPT.** WP2.6 text verified silent on restart; the stated
in-memory/fail-soft ruling matches the advisory-verdict design and C4's
"do not persist" prevents scope creep.

**C5 — ACCEPT.** Both "e.g. 512" occurrences verified (lines 636, 836), and
the landed WP2.1 constant is exactly 512 (`agent.py:63`), so striking "e.g."
is correct against reality, not just tidier.

**C6 — ACCEPT.** Real red-author/green-implementer seam; the plan's WP2.10
list is "tunable" but the resolution correctly fixes only the *default*
list for test purposes. Consistent with the plan's TDD structure.

**C7 — ACCEPT.** Verified at the WP2.7 red-test bullet; the `self._rng`
seam is the standard resolution and contradicts nothing.

### Severity check (task item 4)

No mis-ratings found. I3 cannot escalate to critical: WP2.9's own text
("the current stage 2 is by construction unfinished") stops the worst
misreading at the gate site, and a critique task that awaited the current
stage-2 would add background latency, not broken behavior. I1's divergence
risk is calibration skew, not silent wrongness. Zero-critical is right.

### Ambiguities tripped over (not a full audit)

None beyond the findings. The one candidate — who writes the outcomes record
for a WP2.8 progress/thinking veto — is exactly the seam I7's revision must
settle, so it is covered above rather than filed as a new finding.

### Problems with the audit itself

- **Important:** I5's amendment adds an explicit "not runtime-tunable"
  ruling that contradicts operator directive 2 (see I5 REVISE above).
- **Important:** I7's amendment part (1) assigns an outcome-log write to a
  core-only WP, a layering change no WP declares (see I7 REVISE above).
- **Cosmetic:** C1 claims `memory.py:819` moved (it did not); C2 claims
  WP2.11 "shows 3" params (it shows no signature) and its resolution skips
  the line-1742 block it cites; I3's "never awaited" phrasing is confusable
  with the reader's own async call.

### Gate recommendation

FAIL as-is — two amendments (I5, I7) need the precise rewrites above before
the audit's amendments are applied to the plan. All fourteen findings are
otherwise valid and verified; no REJECT, no missed criticals. One focused
fix pass plus re-review should clear it.

## Cold re-review (reviewer 2)

Reviewed 2026-07-11, no prior context beyond reviewer 1's stated objections.
All citations independently re-verified against `phase-2.md` (2435 lines) and
the working tree at HEAD `1a15700`. Outcome: **PASS** — both revisions
resolve reviewer 1's objections, and a fresh pass over all fourteen findings
surfaced no critical/important problem in the revised document. Five
apply-time cosmetic notes (reviewer 1's four, plus one new on I5).

### The two revisions

**I5 — RESOLVED.** The "not runtime-tunable in this phase" clause is gone;
the revision names `appraisal.prior.w_stakes` 0.4 / `.w_valence` 0.3 /
`.w_novelty` 0.3, resolves them via `resolve_tunable` at consolidation time,
and adds them to WP2.11's key list — exactly reviewer 1's first suggested
fix. Verified against operator directive 2 (plan lines 65–73): both surfaces
are honored on the primary path. The channel-argument note is consistent
with `resolve_tunable`'s spec — plan lines 1929–1931 explicitly allow
"a duck-typed object exposing `runtime_overrides`", so the
empty-overrides-stub escape hatch invents nothing.

Sanity check of the channel note against the code (new, cosmetic):
`ImportancePrior.score` is `score(self, messages: list[dict]) -> float`
(`memory.py:153`) — no channel parameter — and the consolidation call site
(`memory.py:624`, inside `_consolidate_range(channel_id: str, …)`,
`memory.py:575`) holds only a `channel_id` string, not a channel object. So
"pass the consolidating channel" requires plumbing the implementer must
choose: either a second additive optional parameter on `score`
(`channel=None`, mirroring the `msg_id_range` pattern WP2.5 already adds;
`RubricPrior` ignores it) or a registry lookup (the idle trigger already
does one at `memory.py:458`), with the sanctioned stub only on a miss. Both
readings are directive-2-compliant and confined to one intra-WP call site,
so this is an apply-time clarification, not a gate failure — but the apply
pass should spell out the plumbing so the `set_settings` surface actually
reaches these weights whenever a real channel exists.

**I7 — RESOLVED.** Part (1) now assigns the `outcomes` suppression fragment
to the VETOING PLUGIN under WP2.9 point 3's existing mandate — verified at
plan lines 1298–1300 ("Every enforced veto and every pass is recorded into
`outcomes` … via the upsert path") — and states "core never writes the
outcome log". WP2.8's file list (`hooks.py`, `context.py`, `persistence.py`,
`agent.py` — lines 1112–1113) stays core-only; its sole recording
responsibility is the part-(2) `admit()` marker, which matches the
final-veto path (lines 1149–1153) and the WP2.8 red-test bullet ("marker
admitted", line 1189–1190). The "row never modified" sentence is consistent
with the `message_log` DDL (verified: id, channel_id, message, timestamp,
message_type — no annotation column, `persistence.py:34–40`). The fragment
shape `{"suppressed": {…}}` rides the WP2.1 point 7 atomic `json_patch`
merge and collides with no other outcome key (`engagement`, `critique`,
`gate`). Layering objection fully addressed.

### Fresh spot-check of the other twelve

- **I1 — confirmed.** Premises re-verified at lines 760–763 (five signals),
  771–772 (no familiarity formula), 779–780 (unenumerated weights), 834–835
  (five-key vector). Amendment internally consistent: weights sum 1.0;
  `no_probe_default` 0.5 is a value, not a null (envelope invariant, lines
  503–507); negation and imperative both land somewhere (disagreement-max
  and salience-only respectively); bm25 negative-rank polarity correct. No
  namespace collision with revised I5 (`appraisal.weights.*` vs
  `appraisal.prior.*`).
- **I2 — confirmed.** Line 1053–1054: "disagreement high" is the only lens
  condition without key/default. Amendment matches sibling form; coheres
  with I1's disagreement definition.
- **I3 — confirmed.** Line 1050 ("previous stage 2") vs lines 1281–1284
  ("the plugin keeps a per-channel last-completed-stage-2 key" — in a
  GatePlugin paragraph, but only AppraisalPlugin's silent-task body observes
  stage-2 completion). Keeper assignment matches "WP2.9 decides, it does not
  appraise" (lines 1235–1236). Reviewer 1's nit stands: `get_stage2` has a
  DB fallback (lines 877–881), so "never awaited" should read "never wait
  for the CURRENT exchange's stage-2 to complete."
- **I4 — confirmed.** Only bands in code are similarity bands
  (`memory.py:62–63`, 0.75/0.60) — a different scalar. The `--band-edges`
  report-time flag is fine under directive 2 (a reporting construct, not a
  gate/appraisal/critique runtime parameter) and consistent with calibrate's
  nothing-writes rule (line 1374).
- **I6 — confirmed.** Lines 940–942 give no estimator; `entropy` is a
  canonical envelope key (line 504); omit-don't-null matches the RFC 7386
  delete semantics (lines 505–507). Reviewer 1's `"kind"` nit stands.
- **C1 — confirmed cosmetic.** `memory.py:819` (`before_agent_turn`) is
  indeed still at 819 — the audit's "all moved" is overstated, but the
  blanket approximate-lines note remains the right fix.
- **C2 — confirmed cosmetic.** Stale 3-param block verified at the
  line-1742 region; landed 4-param hookspec verified at `hooks.py:557–559`.
  Reviewer 1's addition (also patch the 1742 block itself) is correct.
- **C3 — confirmed.** `valence REAL` (`memory.py:195`) and `embedded`
  (`memory.py:205`) pre-exist.
- **C4 — confirmed.** WP2.6 (lines 981–1019) is silent on restart; the
  in-memory/do-not-persist ruling matches the advisory fail-soft design.
- **C5 — confirmed.** Exactly two "e.g. 512" occurrences;
  `ORIGINATING_TEXT_LRU_MAXSIZE = 512` landed (`agent.py:63`).
- **C6 — confirmed.** Heuristic lists at lines 1060–1064 and 1376–1377;
  fixing only the initial red-spec list preserves the "tunable list" intent.
- **C7 — confirmed.** Line 1097's seeded-RNG red test has no specified seam;
  `self._rng` is the standard resolution, contradicts nothing.

No amendment contradicts the operator directives, the design constraints
and traps (checked in particular: trap #1 fail-open — I1's
`no_probe_default` and I5's fallback chain; trap #3 no-inference — I7's
origin-free fragment; trap #8 decision-time reads — I5's `resolve_tunable`
at consolidation time), or another amendment.

### Apply-time notes (all cosmetic; the apply pass must pick these up)

1. **I3 phrasing:** replace "never awaited, never computed on demand" with
   "never wait for the CURRENT exchange's stage-2 to complete; the reader
   is advisory" (or make `_last_stage2` store the vector so the reader is
   sync).
2. **I6 `"kind"`:** state that `"kind"` is always present (`"topn"` in the
   normal case), not only on the NLL fallback.
3. **C1 accuracy:** drop the implication that `memory.py:819` moved (only
   the retrieval INSERT did); the blanket note is unaffected.
4. **C2 completeness:** also append the 4th-param note at the line-1742
   hookspec block, not just WP2.11.
5. **I5 channel plumbing (new):** when applying I5 to WP2.5 point 3, state
   how the channel reaches the prior — `score` gains a second additive
   optional parameter `channel=None` alongside `msg_id_range` (RubricPrior
   ignores both), MemoryPlugin passes the real channel when it can obtain
   one (registry lookup, as the idle trigger at `memory.py:458` already
   does), and the empty-`runtime_overrides` stub applies only when the
   lookup misses.

### Gate recommendation

**PASS.** Both REVISE items are resolved as specified; no new
critical/important finding. Apply the fourteen amendments with the five
cosmetic notes above.

## Apply review (reviewer 3)

Reviewed 2026-07-11, no prior context. `plans/implementation/phase-2.md` read
in full (2540 lines, untracked, verified by direct read since no git diff
exists) against every gated amendment text (I1–I7, C1–C7) and the five
apply-time notes (a–e) in reviewer 2's report above. Outcome: **PASS** — all
fourteen amendments and five notes are applied with semantic fidelity; no
stale contradicting occurrences found; both judgment calls are correct; the
flagged non-application is acceptable.

### Fidelity check, all 14 findings

- **I1** — WP2.4 point 3 "Stage-1 constants" block (lines 791–807) matches
  the gate exactly: signal→vector mapping, `salience` weighted-composite
  formula with the five named `appraisal.weights.*` keys and 0.35/0.15/0.20/
  0.20/0.10 defaults, familiarity formula with `appraisal.probe.rank_scale`
  10.0. Verbatim in substance.
- **I2** — WP2.7 point 3 (line 1121–1122): "disagreement ≥
  `critique.lens.disagreement` (0.6) → adversarial" present and matches the
  sibling-condition form.
- **I3** — WP2.5 point 2 "Last-completed stage-2 reader" block (lines
  990–995) implements chosen option (b): `_last_stage2` stores the vector
  itself, `get_last_stage2` is synchronous. Consumer rewrites present at
  WP2.7 point 3 (line 1116, `appraisal.get_last_stage2(channel.id)`, sync,
  called with `await` removed) and WP2.9 point 3 (lines 1369–1373, same
  reader, explicitly marked "safe because it is advisory context, not the
  keyed record"). Apply-note (a)'s rephrase is honored: "never wait for the
  CURRENT exchange's stage-2 to complete" appears at line 995 and 1370,
  replacing the confusable "never awaited" phrasing — note (a)'s alternative
  fix (sync store) was chosen over the rewording, which is consistent with
  the note's own "(or make `_last_stage2` store the vector so the reader is
  sync)" branch.
- **I4** — WP2.10 point 2 (lines 1460–1464): band definition present
  verbatim — `[0, 0.2), [0.2, 0.5), [0.5, 1.0]`, `--band-edges` override,
  `none` for no-vector, "reporting construct only."
- **I5** — WP2.5 point 3 full rewrite (lines 996–1021) present: `max(stage1.
  salience, stage2_composite)`, named `appraisal.prior.w_*` keys (0.4/0.3/
  0.3) resolved via `resolve_tunable` at consolidation time. The withdrawn
  "not runtime-tunable" clause is absent (confirmed by grep — no match
  anywhere in the file). Channel plumbing per note (e) is present at lines
  1010–1014: MemoryPlugin passes the real channel via registry lookup when
  obtainable, else a stub with empty `runtime_overrides` on a miss — matches
  note (e) precisely, including the registry-lookup precedent citation.
- **I6** — WP2.5 point 2 entropy schema (lines 970–978): `{"kind":
  "topn"|"nll", "mean", "max", "n_tokens"}`, and apply-note (b) is honored —
  "`"kind"` is ALWAYS present (`"topn"` in the normal case)" appears
  explicitly at line 972, not only attached to the fallback branch.
- **I7** — WP2.8 point 4 (lines 1250–1258): two-way recording present —
  part (1) the vetoing plugin's WP2.9 outcomes fragment
  `outcomes={"suppressed": {...}}`, "core never writes the outcome log";
  part (2) core's `admit()` marker, "WP2.8's only recording responsibility."
  Matches WP2.9 point 3's mirror at lines 1389–1394
  (`outcomes={"suppressed": {...}}`, "the vetoing plugin, never core, writes
  this record").
- **C1** — Staleness note present after "Read first" (lines 56–61); the
  `memory.py:819` claim is not repeated as "moved" anywhere the amendment
  touches (apply-note (c) honored) — the design section below correctly
  still cites `memory.py:819` as the live, unmoved line for
  `before_agent_turn` (lines 1745, 2115, 2218), which is consistent, not
  stale, since that section is HEAD-bb03fa5-scoped and unaffected by C1's
  WP2.1-era note.
- **C2** — Both locations amended: WP2.11 point 1 (lines 1506–1508,
  "as landed — `on_message_persisted` carries a 4th parameter, `origin`")
  AND the ~line-1846 hookspec block itself now carries "*(Landed with a 4th
  parameter, `origin: str | None` — see the WP2.1 implementation
  amendment.)*" — apply-note (d) explicitly honored.
- **C3** — WP2.5 point 3 "No schema migration" note present (lines
  1023–1026), citing `memory.py:195` and the embedded-vector precedent.
- **C4** — WP2.6 "Restart semantics" paragraph present verbatim (lines
  1081–1084).
- **C5** — Both LRU mentions ("512 entries — constant, not tunable") found
  at lines 643 and 862; no "e.g." qualifier remains on either (grep for
  "e.g. 512" returns zero hits).
- **C6** — WP2.7 "Pattern lists are part of the red-test spec" block present
  (lines 1141–1147) and WP2.10 cross-references it ("the WP2.7 pattern-list
  red-spec rule applies," line 1477).
- **C7** — WP2.7 point 3 injectable `self._rng = random.Random()` present
  (line 1126–1128), matching the gate text.

### Consistency sweep

Grepped the whole plan for each amendment-introduced name:
`get_last_stage2`/`_last_stage2` (4 occurrences, all consistent — the WP2.5
definition plus the two WP2.7/WP2.9 consumer sites, no stale 3-line
description survives); `appraisal.weights.*` (only in WP2.4, no collision
with `appraisal.prior.*`); `appraisal.prior.w_*` (only in WP2.5, as
expected); `appraisal.probe.*` (WP2.4 only, both occurrences — budget_ms and
rank_scale, consistent); `critique.lens.disagreement` (single occurrence,
WP2.7 point 3, no orphaned old "disagreement high" wording remains anywhere
in the file — confirmed no stale hits); the `{"suppressed": ...}` fragment
(three occurrences — WP2.8 point 4 twice-worded consistently, WP2.9 point 3
— all agree on shape and on "vetoing plugin, never core" ownership);
`ImportancePrior.score` channel param (WP2.5 point 3's `channel=None`
addition is the only definition; the 2A design-section discrepancy notes
predate Phase 2B+ and do not reference `score`'s signature, so no
contradiction). No stale contradicting occurrence found for any of the
seven searched surfaces.

### Judgment call (i) — skipping WP2.11 key-list edits

**Confirmed correct.** WP2.11 point 2 (lines 1512–1515) reads "every
`appraisal.*`, `critique.*`, `gate.*`, `memory.contradiction.*` key with its
best-guess default" — a blanket namespace enumeration, not a hardcoded list
of individual key names. None of I1/I2/I5's new keys
(`appraisal.weights.*`, `critique.lens.disagreement`,
`appraisal.prior.w_*`) needed a textual add to WP2.11 point 2 because they
already fall under the stated namespaces; the "add to WP2.11's key list"
language in the gated I2/I5 amendments is satisfied by the namespace clause
already covering them, not by literal enumeration. Rating: correct, no
issue.

### Judgment call (ii) — "overall score = max over the covered exchanges" addition to I5

**Confirmed correct, semantics preserved.** The original WP2.5 point 3
sentence (pre-audit) was "score = max per-exchange salience composite
(best-guess aggregation, commented)" — i.e., the *aggregation across
exchanges in a consolidation range* was already specified as max, with
"per-exchange salience composite" naming what's being maxed. I5's gated
amendment redefines *only* the per-exchange composite formula (the
`max(stage1.salience, stage2_composite)` piece) and does not touch the
cross-exchange aggregation. The applied text's added clause "overall score =
max over the covered exchanges" (line 1005) restates the original
aggregation rule using the new per-exchange terminology, rather than
changing it — necessary because the gated I5 amendment text used
"per-exchange score" language that, read alone, could be mistaken for
replacing the original max-aggregation with a single value. The added
clause disambiguates without altering meaning. Rating: correct, no issue.

### Non-application — WP2.4 point 5 reader-list cross-reference

**Acceptable, not a correctness problem.** WP2.4 point 5 (lines 894–909)
lists the pure readers `get_appraisal`, `get_appraisal_out`, and
`get_stage2` — all WP2.4/WP2.5-era readers — but does not list
`get_last_stage2` (the new I3 synchronous advisory reader, defined in WP2.5
point 2, consumed in WP2.7/WP2.9). This is a genuine incompleteness: an
implementer reading WP2.4 point 5 alone would not discover
`get_last_stage2` exists. However: (1) `get_last_stage2` is fully specified
at its owning site (WP2.5 point 2) and both its consumer sites (WP2.7 point
3, WP2.9 point 3) with consistent signature and semantics; (2) WP2.4 point 5
is temporally WP2.4-scoped documentation — `get_last_stage2` is a WP2.5
deliverable, arriving after WP2.4, so its absence from a WP2.4-era API
inventory is defensible on the same "additive, documented at point of
introduction" precedent the plan already uses for `get_or_compute_out`
(WP2.9-era, also absent from being a WP2.4 *deliverable* though it is
listed in that same section for architectural context); (3) no consumer or
test depends on discovering `get_last_stage2` via WP2.4 point 5 — WP2.7 and
WP2.9 each cite it directly at their own gating sites. Rating: cosmetic, not
important/critical. Correctly left out of the gated scope.

### Collateral spot-check

Sections untouched by any finding verified unaltered from the pre-existing
2026-07-07/07-09 material: operator directives (lines 63–92, all 5 items
intact, unchanged text), design constraints/traps 1–10 (lines 94–166,
verified against I1/I5/I7's own citations of them — e.g. trap #1 fail-open,
trap #8 decision-time reads — content matches, no rewrite), dependency graph
and edges (lines 186–236, WP numbering and depends_on unchanged), sub-phase
2A/2B/2C/2D/2E structure and risk register (lines 240–514, no amendment
touches session/parallelism boundaries), WP2.1/2.2/2.3 bodies (lines
518–759) unchanged except the one sanctioned C1/C2 edits (staleness note
after "Read first," hookspec 4th-param parenthetical at line ~1846 inside
the WP2.1-era design section, which is itself pre-existing 2A design
material, not new WP2.1-body text — confirmed the WP2.1 *work-package*
section proper, lines 518–675, carries no unexpected edits beyond its
pre-existing "Implementation amendment (WP2.1 complete)" paragraph already
in place before this audit). The Consistency review / fix report / re-review
(lines 1573–1685) and the entire Phase 2A design/review/red-TDD history
(lines 1697–2540) are pre-existing and untouched by any of the 14 findings —
spot-checked their line ranges for accidental edits and found none.

### Gate recommendation

**PASS.** No critical or important findings. All fidelity, consistency,
judgment-call, and collateral checks clear.

## Second-pass review (reviewer 4)

Reviewed 2026-07-11, no prior context. Scope: the two second-pass additions
only — (A) the WP2.4 point 4 reference sketch (plan lines 848–891) and
(B) the newly pinned constants (`appraisal.probe.max_tokens` line 777,
`critique.provenance.max_terms` lines 1184–1185, calibrate `--since-days`
line 1504 and the suggestion trigger lines 1516–1519). Everything cited
below re-verified by direct read. Outcome: **FAIL** — one critical and one
important finding, both in the sketch.

### Sketch — checks that pass

- **(a) Synchronous in-flight registration.** Cache check, `_inflight.get`,
  `create_future()`, and `self._inflight[k] = fut` all execute with no
  intervening `await`; the first suspension point is `await self._compute(...)`.
  A second concurrent caller for the same `(exchange_key, direction)`
  deterministically takes branch 2 and shares the future. Dedup holds.
- **(b) Failure path, nominal case.** `del self._inflight[k]` before
  `set_exception` evicts the key so the next request retries;
  `fut.set_exception(exc)` reaches all branch-2 waiters; `raise` reaches the
  direct caller. Nothing is cached, so `get_appraisal`/`get_appraisal_out`
  keep returning None until a compute succeeds — consistent with the
  compute-failure paragraph above the sketch.
- **(c) No dangling in-flight entry.** Both exits (success line pair,
  except block) delete `self._inflight[k]`; ownership of `k` is unique
  because registration is synchronous, so the `del` cannot KeyError. The
  success ordering (cache set → in-flight del → `set_result`) means a waking
  waiter observes a populated cache. No early return exists between
  registration and the try block.
- **(e) LRU bound.** `LRUDict(maxsize=512)` is constructed in the sketch and
  matches the surrounding text's "bounded LRU dict, 512 entries — constant,
  not tunable" (line 908) and the C5-pinned wording. Enforced, not deferred.
- **(f) Re-entrancy.** `_persist_stage1` writes via `upsert_exchange` and
  never calls back into `_get_or_compute`; a re-entrant get for the same key
  after completion is a cache hit; during compute it becomes a branch-2
  waiter, not a second compute. No deadlock/livelock path found.
- **Consistency with WP2.4/WP2.9 text.** Direction-keyed `(exchange_key,
  direction)` maps, "probe runs exactly once per (key, direction)",
  fail-open assigned to CALLERS (traps #1/#10), probe-timeout-is-not-a-
  failure (never takes branch 4), and readers-return-None-until-success all
  match the amended prose (lines 825–846, 886–891, 921–939, 1370–1409).

### Critical — R4-C1: a cancelled waiter corrupts the shared future for everyone

**Location:** sketch branch 2 (`return await fut`, line 861) and the
completion sites (`fut.set_exception(exc)` line 872, `fut.set_result(vector)`
line 880).

**Problem:** when an asyncio Task is cancelled while `await`ing a plain
Future, `Task.cancel()` cancels that Future (`self._fut_waiter.cancel()` in
CPython's `Task.cancel`). The sketch shares ONE future among all branch-2
waiters, so a single cancelled waiter (shutdown teardown of the hook-dispatch
gather; any future consumer wrapping the pull in `asyncio.wait_for` — a
plausible move in a codebase whose gate path is built around hard latency
budgets) puts `fut` into CANCELLED state while the compute is still running.
Consequences: (1) every OTHER waiter sharing `fut` gets a spurious
`CancelledError` even though the compute goes on to succeed; (2) on the
success path, `fut.set_result(vector)` raises `InvalidStateError` — after
the cache set and in-flight del, so the cache is consistent, but the DIRECT
caller receives `InvalidStateError` instead of the vector and line 881's
persist task is never spawned (the stage-1 row — the offline calibration
corpus — is silently lost for that exchange); (3) on the failure path,
`fut.set_exception(exc)` raises `InvalidStateError` from inside the except
block, masking the original compute exception. This violates check-(b)'s
"exception reaches all waiters" contract under a real concurrency scenario,
and the sketch is marked NORMATIVE FOR STRUCTURE — implementers will copy
exactly this shape.

**Fix (structure-level, either suffices; both is better):** (i) insulate the
shared future from waiter cancellation — branch 2 becomes
`return await asyncio.shield(fut)` (cancelling the waiter then cancels only
its shield wrapper, never the shared future); and/or (ii) guard both
completion sites — `if not fut.cancelled(): fut.set_result(vector)` (and the
same for `set_exception`) so the owner's return value and the persist spawn
survive regardless. Add the corresponding regression to the WP2.4 red tests
(cancel one of N concurrent waiters mid-compute; assert the remaining
waiters and the direct caller still receive the vector and the persist
fires).

### Important — R4-I1: the fire-and-forget persist task is unreferenced — GC-drop and unretrieved-exception hole

**Location:** sketch lines 881–882 (`asyncio.create_task(self._persist_stage1(...))`,
return value discarded) and the Notes paragraph (lines 886–891).

**Problem:** two related holes, both structural. (1) The event loop keeps
only a weak reference to tasks; a `create_task` result that nobody stores can
be garbage-collected before the coroutine completes (the documented CPython
pitfall — "save a reference to the result of this function"). The dropped
write is exactly the stage-1 persist the plan elsewhere calls the offline
calibration corpus (lines 834–835, 938–939), and the loss is silent. (2) If
`_persist_stage1` raises, no one ever awaits or inspects the task, so the
exception surfaces — if at all — only as a GC-time "Task exception was never
retrieved" log message. The sketch's Notes explicitly plug the analogous
hole for the failed *future* (`fut.exception()` / no-op done-callback,
lines 888–891) but are silent on the persist *task*, which is the hole the
same trap-#10 reasoning ("never swallow exceptions") applies to. WP2.9
point 1 repeats the bare-`create_task` idiom for the shadow-record write,
so whatever discipline the sketch pins will be copied there too.

**Fix:** pin the standard discipline in the sketch or its Notes: hold the
task in an instance-level set with a `discard` done-callback
(`self._persist_tasks.add(t); t.add_done_callback(self._persist_tasks.discard)`),
and require `_persist_stage1` to catch and LOG its own exceptions (a persist
failure must never propagate anywhere near the gate path, but it must be
visible in the log — trap #10). One sentence in the Notes naming both the
strong-reference requirement and the log-own-exceptions requirement is
enough; the current text covers neither.

### Constants (B) — all pass

- **`appraisal.probe.max_tokens` = 12 (WP2.4 point 2, line 777).** Right
  section (the FTS5 sanitizer that caps quoted OR-joined tokens); 12 is a
  sensible cap for a bounded MATCH under a 50 ms budget. Marked
  runtime-tunable; `appraisal.*` keys resolve through the WP2.3 helper per
  trap #8/directive 2, and `channel` is in scope at the probe site
  (`_compute(channel, text, direction)`), so decision-time resolution is
  implementable as written. Falls under WP2.11 point 2's blanket
  "every `appraisal.*` key" namespace clause (reviewer-3 judgment call (i)),
  so no doc-list edit was needed. No collision: `appraisal.probe.*` now
  holds `budget_ms`, `rank_scale`, `max_tokens` — three distinct keys,
  single definitions each (grepped).
- **`critique.provenance.max_terms` = 8 (WP2.7 point 4, lines 1184–1185).**
  Right section (the `message_fts` key-term probe, tier 2 of the provenance
  check); 8 terms is proportionate to the 12-token appraisal-probe cap.
  Runtime-tunable under the `critique.*` namespace; WP2.7 point 3's "all
  thresholds via `resolve_tunable`" discipline extends naturally. Interacts
  correctly with the C6 pattern-list red-spec rule (the extraction list is
  red-defined; the cap is the tunable). No conflicts.
- **Calibrate `--since-days` = 7 and the suggestion trigger (WP2.10 point 2,
  lines 1504, 1516–1519).** Values sensible: 7-day default window matches
  the "day of traffic" live check's spirit; empty-critique rate ≥ 0.8 over
  ≥ 20 exchanges per (channel, lens) cell is a reasonable minimum-sample
  guard against noisy cells; +0.05 proposed delta is small relative to the
  lens defaults (0.6/0.5/0.3/0.6) and consistent with "suggests, never
  writes" (directive 3). The NOT-runtime-keys ruling is correctly reasoned:
  the plan cites the appraisal-band reporting-only precedent, and reviewer
  2's I4 verification already rated `--band-edges` "fine under directive 2
  (a reporting construct, not a gate/appraisal/critique runtime parameter)"
  — the identical logic covers these report-time CLI constants, which no
  daemon decision path reads. Flag-overridability satisfies operator
  adjustability without a restart trivially (it's a second process). No
  contradiction with any pinned value elsewhere (grepped 0.8, +0.05,
  max_tokens, max_terms, since-days — single definition sites only).

### Scope check — clean

Reviewer 3 read the file at 2540 lines; it is now 2593 (+53). The shift
accounting attributes every added line to the two deliverables: reviewer-3
citations before WP2.4 are unmoved (staleness note 56–61, directives 63–92,
traps 94–166, dependency graph 186–236, WP2.1 body 518–675 — spot-read,
unchanged in substance); WP2.4 point 5, cited at 894–909, now sits at
940–955 (+46, the sketch block); WP2.10's band text, cited at 1460–1464, now
1509–1513 (+49, sketch + max_terms lines); WP2.11 point 2, cited at
1512–1515, now 1565–1568 (+53, all insertions upstream). Grep for
"amendment 2026-07-11" surfaces no marker outside the previously gated
fourteen plus the two deliverables under review. WP2.11's body text is
unchanged. Nothing outside scope was altered.

### Cosmetic

- **R4-cos1:** WP2.4 point 2 says "runtime-tunable" for `max_tokens` where
  the adjacent `budget_ms` shows the explicit
  `resolve_tunable(channel, cfg, …)` form; mirroring it would remove any
  ambiguity about decision-time resolution. No semantic gap.
- **R4-cos2:** all three sketch return paths hand callers the SAME dict
  object that lives in the cache; a caller mutating its vector corrupts the
  cache for every later reader. A one-line note ("treat returned vectors as
  immutable" or return a copy) would close it.

### Gate recommendation

**FAIL.** R4-C1 (critical): waiter cancellation cancels the shared in-flight
future — spurious CancelledError for sibling waiters, InvalidStateError at
`set_result`/`set_exception` (masking the original exception on the failure
path), and a silently skipped persist; fix with `asyncio.shield` on branch 2
and/or `fut.cancelled()` guards at both completion sites. R4-I1 (important):
the fire-and-forget persist task is unreferenced (GC-drop risk) and its
exceptions are never retrieved — trap #10; pin the strong-reference +
log-own-exceptions discipline. Constants (B) all pass; scope is clean.
