# Memory retrieval fidelity

A record of what we found when we looked at why retrieved memories come
back vague, and what we propose to do about it. Read-only analysis as of
2026-11-08; nothing here is built yet. The code under discussion is
`packages/corvidae-cognition/corvidae_cognition/memory.py`, the PRD
section is §3.7 Memory, and the running description is in
[cognition.md](cognition.md#memoryplugin).

The short version: the relevance bands are hard to tune because they
threshold a number that doesn't mean anything stable, but even perfectly
tuned bands would hand the model a few sentences summarising ten
thousand tokens of conversation. The fidelity ceiling is set when the
record is written, not when it is retrieved. Fix the write side first,
make the read side honest second, and measure both against a labelled
query set because right now nobody can see what the right record scored
when it lost.

The design that follows from this, with corrections to some findings
below, is [memory-consolidation-incremental.md](memory-consolidation-incremental.md).

## How retrieval works today

For orientation; [cognition.md](cognition.md#retrieval-read-path) has the
detail.

**Write side.** Compaction drops a range of the window (with the default
`max_context_tokens: 24000`, `compaction_threshold: 0.8` and
`compaction_retention: 0.5`, roughly ten thousand tokens), and
consolidation turns that range into one record. The idle trigger does the
same for a channel's quiet tail. A range that has left the window is
rebuilt as the agent's view and cut into segments that each fit
`max_context_tokens`, one record per segment. The consolidation
instruction (`prompts/memory_consolidation_fork.md`) asks for "a
first-person summary, a few sentences", plus `topic_tags` and
`participants`. Only the summary is embedded. An importance prior is
scored separately. A new record whose embedding is within
`dup_threshold` (0.95 cosine) of an existing one supersedes it.

**Read side.** Before each user-originated turn, the unanswered user
text is embedded, an exact KNN over `memory_vec` returns `k × 4`
candidates filtered to indexed, unredacted records, and each is scored
`similarity × exp(-age_days / half_life_days)`. The band
(strong ≥ 0.75, moderate ≥ 0.60, else weak) is applied to that score by
default, or to raw similarity under `band_on: similarity`. Weak drops
unless nothing else matched. Survivors enter the window through the
funnel as `[band] (age) summary` lines under a 512-token budget. The
active tools are `search_memory` (FTS5 keyword only) and `recall_raw`
(replays a record's source range by memory id).

## Findings

Each finding says whether it is measured, read from the code, or
inferred. The only real-encoder measurements we have are the six probes
in [#409](https://github.com/schuyler/corvidae/issues/409), taken on Lex
with bge-m3 over 146 records.

### 1. The retrieval unit is a compaction range, and its embedding is a centroid

*Read from the code; the consequence is inferred and consistent with
#409's numbers.*

One record covers one consolidation range, which is as many topics as
happened to be in the window when compaction fired. Its summary is a few
sentences about all of them. A dense embedding of a multi-topic paragraph
lands near the mean of its topics, a middling distance from every
specific query. #409 shows the shape: the correct record for the Yoz
Grahame question had cosine 0.43 and ranked below a tangential hit for an
unrelated sourdough question at 0.46. No threshold separates those two
numbers. The geometry of embedding a blend does not allow it.

The near-duplicate merge inherits the problem. Two multi-topic summaries
rarely reach 0.95 cosine even when they restate the same fact, so the
merge almost never fires, and the facts that *should* collide never do.

### 2. The detail is gone before ranking starts

*Read from the code.*

The compression ratio from source range to record is in the hundreds.
Whatever the ranker does, the best it can admit is the gist. The raw
dialog is still in `message_log` and `recall_raw` can replay it, but the
admitted line carries no memory id, so the model cannot drill into a
memory it was just shown. It would have to call `search_memory` first,
and that tool is FTS5-only, a weaker retriever than the passive path it
is supposed to supplement.

### 3. The band thresholds a quantity with no dynamic range

*Measured in #409; the recency point is read from the code.*

The band is presented to the model as a trust label (the calibration
prompt reads `[weak]` as "at most a guess"), but it is computed from
absolute cosine, and absolute cosine from a dense encoder mostly reflects
the query's register. #409 measured per-query median similarity between
0.28 and 0.45 across six probes. When the baseline moves more than the
signal, a fixed threshold pair cannot work: the same day's calibration
banded the exactly right Yoz record `[weak]` and pulled `[strong]`
check-in records for "Thanks, this has been helpful :)".

Multiplying recency in before banding (`band_on: score`, the default)
adds a second error. A perfect match from ninety days ago scores about
0.05 under a thirty-day e-folding time and can only be `[weak]`. The band
then tells the model to distrust a memory because it is old, which is a
different fact from being uncertain about it. `band_on: similarity`
removes that but keeps the baseline problem.

The query-relative position does separate the cases. In #409, queries
with a relevant record had a top z-score of 3.5 to 5.7 against the
query's own distribution over all records; queries with none had about
2.4.

### 4. Nothing in CI or production can see any of this

*Read from the code.*

`tests/cognition/test_retrieval_benchmark.py` runs recall@k and MRR, but
over a hashed bag-of-words stub embedder. It proves the plumbing moves
records from one end to the other; it says nothing about how bge-m3 or
nomic behave. `retrieval_log` records only the top candidate's score and
similarity ([#407](https://github.com/schuyler/corvidae/issues/407) asks
for every candidate), so when the right record loses in production there
is no row saying what it scored. (Since #407, `retrieval_candidate` logs
every candidate; see `docs/cognition.md`.) The band thresholds have been tuned
blind against a non-calibrated quantity, and that is why tuning them has
not converged.

### 5. The query is the inbound message alone

*Read from the code; the effect is visible in #409's thank-you probe.*

"Thanks, that worked" and "what about the other one" embed to nothing
useful, and the thank-you probe pulled `[strong]` generic check-in
records. Anaphora and short acknowledgements are a large share of chat
turns. The previous assistant turn is not part of the query, and the
model has no cheap way to issue a better one.

### 6. Smaller things noticed, not measured

- Only `summary` is embedded. `topic_tags` and `participants` never reach
  the vector. Appending them might sharpen the embedding; it might also
  add noise. Untested.
- `llm.embedding.query_prefix` and `document_prefix` default to empty.
  bge-m3 needs neither; nomic-embed-text does. Worth confirming against
  the production encoder's model card.
- FTS5 is wired in but used only as the degraded fallback. Proper nouns
  ("Yoz", "Tacoma", a hostname) are where dense similarity is weakest and
  keyword match is exact.

## Recommendations

In the order we would do them. Each depends on the one before it to be
measurable.

### 1. Instrument, and build a labelled query set

Land #407 so every candidate is logged with similarity, score, band and
whether it was admitted. Then label twenty to forty real queries from
Lex's `message_log` against its memory records (you know what was said,
so you know which record should have come back) and score them with the
real encoder through `scripts/eval_memory.py`. The fixture format is
documented in `tests/evals/metrics.py`. This is a day's work and it is
the precondition for every other step being a result rather than an
opinion.

### 2. Re-consolidate the whole history with richer records

The structural fix for findings 1 and 2, and the user's proposed first
move. See [the plan below](#re-consolidation-plan). The point is a bigger,
more specific baseline to run the other changes against, and a way to
A/B consolidation prompts and segment sizes on identical input.

### 3. Decouple ranking from banding; stop banding on absolute cosine

Rank by whatever works (score with recency is fine for ranking), but
derive the band from a quantity with per-query dynamic range:

- **Relative band** (#409's proposal): z-score within the query's own
  similarity distribution. The KNN is already exact over every record,
  so the distribution is free. The proposed 4.5 / 3.0 thresholds rest on
  six probes; treat them as a starting point.
- **Reranker**: a cross-encoder over the top N (llama-server serves
  bge-reranker-v2-m3 on `/v1/rerank`). A reranker score is trained to
  mean "does this passage answer this query", which is what the band
  pretends to measure. Shape it like `ImportancePrior`, a swappable
  scorer inside the memory plugin, falling back to the relative band
  when the model is down.

### 4. Hybrid candidates

Union the FTS5 and KNN candidate lists with reciprocal rank fusion before
scoring. Both candidate functions already exist (`_vec_candidates`,
`_fts_candidates`); this is a small change that fixes proper nouns.

### 5. Two cheap fidelity wins on the read side

- Put the memory id on the admitted line (`[strong] (3d) #142 ...`) so
  the model can `recall_raw` a hit directly.
- Make `search_memory` use the vector path plus FTS, not FTS alone. The
  active tool should not be weaker than the passive one.

### 6. Change the retrieval unit to atomic claims

Once the eval set exists and re-consolidation has stored the model's full
reply (see below), index each atomic first-person claim ("Schuyler told
me he sold the Tacoma in October") as its own embedded row pointing at
the parent record and message range. Retrieve over claims, admit the
claim with its parent id and age. A single-topic claim embeds to a point,
not a centroid; near-dup merge becomes meaningful because restated facts
actually collide; and what lands in the window is the fact, not the
gist.

This is not the knowledge graph the PRD decision log rejected: no
relations, no n-hop traversal, a finer-grained index over the same
episodic records. It is a PRD change all the same (§3.7 Memory 2 defines
a record as "a first-person summary"), so it needs the owner's decision
before it is built.

### 7. Better queries

Cheapest: include the previous assistant turn in the query text.
Stronger: let the model issue its own query, which is what
[#406](https://github.com/schuyler/corvidae/issues/406) proposes from a
different trigger (first use of a tool in a turn).

## Re-consolidation plan

A one-time operator-run pass that rewrites every consolidation record
from the full `message_log`, with a better instruction and smaller
segments, superseding the old records.

### What already exists

- `build_catchup_segments` rebuilds the agent's view of any range that
  has left the window and cuts it into budget-sized segments. The daemon
  and the replay harness both call it, so a result from one is a result
  about the other.
- `harness/consolidate_replay.py` drives it against a frozen snapshot
  with any prompt file and N trials, for one range at a time. That is
  the A/B half, already built. It writes nothing.
- Records carry `prompt_hash`, `mode` and `model`, and `memory_prompt`
  stores each instruction text once, so records from a rerun are
  distinguishable from the originals and from each other.
- The near-duplicate merge path already does "supersede an older record,
  fold its statistics in": `superseded_by`, `indexed = 0`, vec row
  deleted, summed `retrieval_count`, max `importance`.

What is missing is a batch driver that walks the whole history and
writes.

### The command

An operator-only `corvidae reconsolidate`, a sibling of `redact`: never a
registered tool, never reachable from a hook the model can influence.

1. **Snapshot first.** Everything runs against a copy of `sessions.db`
   until a variant wins. Variants go into separate copies; two variants
   in one live store would both be retrieved.
2. **Segment and write.** Per channel, segment the whole `message_log`
   from id 0 with `build_catchup_segments`, call the consolidation stage
   with the chosen instruction, and insert one record per segment,
   stamped `mode = reconsolidate` with the prompt hash and model.
3. **Supersede, don't delete.** Every existing consolidation record whose
   range intersects the new record's gets `superseded_by`, `indexed = 0`
   and its vec row dropped. The new record takes the summed
   `retrieval_count` and max `importance` of what it replaces. "Demote,
   never delete" holds and `recall_raw` still reaches the old rows. Only
   records consolidation produced are touched: goal retirement writes
   records with `DIRECT_STAMP` (`mode = direct`, null prompt hash), and
   those stay as they are.
4. **Resumable, with a dry run.** Lex has about 150 records, so a full
   pass is a few hundred LLM calls at up to twenty thousand tokens each,
   hours on a local llama-server. Progress is per channel and survives an
   interruption; `--dry-run` reports segment count and estimated tokens
   without calling anything.

### Three changes to the rerun itself, not just the prompt

- **Segment budget as a flag**, independent of `max_context_tokens`.
  Rerunning with today's segmentation and a better prompt still folds a
  24k-token view into one record. Our guess, and it is a guess, is that
  four to six thousand token segments will beat prompt changes alone.
  Stamp the budget into the record next to `prompt_hash`, or the A/B
  cannot be read back from the table.
- **Store the model's full JSON reply**, not just `summary`. If a variant
  asks for a summary plus a list of atomic claims, keeping the array
  means recommendation 6 can be built from stored data without a second
  pass over the history.
- **Record the system prompt used.** PRD §3.7 Memory 7 requires every
  consolidation with a channel to run under that channel's system
  prompt, so the agent writes in its own voice. Offline that has to come
  from somewhere (below).

### Prompt variants to test

Against identical segments, scored on the labelled query set:

- **A, current**: `prompts/memory_consolidation_fork.md` as is. The
  control.
- **B, detailed**: the same framing, but asking for every concrete
  particular (names, identifiers, numbers, dates, decisions, who said
  what, what was resolved and what was left open), with no length
  ceiling beyond "omit filler".
- **C, summary plus claims**: B's summary plus a `claims` array of
  atomic first-person statements, each one fact, each with the epistemic
  framing kept ("X told me", "I inferred"). The summary is what gets
  embedded and admitted today; the claims are stored for recommendation
  6.

Crossed with two segment budgets (the window budget, and about 5k
tokens), that is six runs. Recall@k and MRR on the query set pick the
winner; tokens admitted per retrieval is the cost column.

### Open decisions

- **Where the command runs.** Inside the daemon process it gets
  `view_inputs` for free (the channel's resolved system prompt and
  budget), but it competes with live traffic for the backend. Offline it
  needs a system-prompt file the way the replay harness does, and the
  harness's `view_inputs_from_config` skips per-channel overrides.
- **Access statistics on a many-to-one or one-to-many supersede.** Old
  and new segments will not align. Summing `retrieval_count` across
  everything a new record overlaps inflates it; taking the max
  under-counts. Summing matches what near-dup merge does and the
  retention job tolerates inflation better than starvation, so sum,
  but it is a choice.
- **What happens to `exchange_log` rows** that reference superseded
  record ids, if anything does. To check before writing the supersede
  step.

### PRD comparison

Differences to resolve with the owner before implementing, per
`AGENTS.md`:

- **A new consolidation trigger.** §3.7 Memory 1 names compaction and
  idle. Item 7 already allows "windowless work (seeding, imports,
  evals)", so an operator rerun fits there, but it should be written in,
  most naturally beside `redact` in item 8 as the second operator-only
  memory command.
- **Segment budget decoupled from `max_context_tokens`.** Item 1 and
  §3.8 tie a catch-up view to the channel's budget. A smaller segment is
  still a legitimate rebuilt view, but the PRD should say the budget is
  a consolidation parameter.
- **Storing the full reply and later indexing claims** (recommendation
  6) changes what a record *is* under item 2. Not needed for the
  re-consolidation itself, which can store the JSON opaquely, but the
  reason for storing it should be on the record.
- **Relative banding and a reranker** (recommendation 3) change item 4's
  "band strong/moderate/weak on score or raw similarity". #409 already
  proposes the wording.

Nothing here contradicts principles 2 (tail-append, no mid-window
mutation), 4 (`redact` is the one deletion path) or 14 (retrieval spans
the whole memory). The supersede step demotes; it does not delete.

## What we would not do

- **Keep tuning the band thresholds** before step 1. The quantity is not
  calibrated and the measurement is not there.
- **Build the knowledge graph.** The decision log's reasons still hold.
  Atomic claims get most of the benefit at a fraction of the noise.
- **Narrow retrieval by channel.** Principle 14 is settled and the
  fidelity problem is not about scope.
