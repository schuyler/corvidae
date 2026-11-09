# Incremental narrative consolidation

Consolidation forks the live window at the end of a turn whenever a few
thousand tokens of new conversation have built up, and the fork writes a
handful of short first-person narrative entries instead of one summary of
a few sentences. Each entry is its own memory record. Retrieval stops
admitting records of conversation the agent can still see, and the funnel
stops letting one oversized entry block the rest.

This is a design. Status: §1, §2 and §4 (#635), §3, the turn trigger
(#636), §5 (#634), §6's funnel change (#631) and §7 (#633) are
implemented; [cognition.md](cognition.md#consolidation-write-path)
describes the code as built. It answers the analysis in
[memory-retrieval-fidelity.md](memory-retrieval-fidelity.md), with the
corrections listed at the end of this doc.

## Why

On Lex (`max_context_tokens: 120000`, `compaction_threshold: 0.85`,
`compaction_retention: 0.3`) each compaction drops about 66k tokens, and
consolidation turns that range into one record of "a few sentences". The
model reads the raw rows (the compaction-time fork runs in
`before_compaction`, before the prefix is replaced), so nothing is lost
on the way in. It is lost on the way out: roughly a thousand to one, mostly
over tool-heavy work. The embedding of a record like that is the average
of every topic in it, which fits what band calibration found (Lex
DECISIONS round 8: top similarity p50 0.53, p90 0.61 over 399 retrievals,
right and wrong records under 0.1 apart, and a right record can score
below a wrong one).

Writing longer records alone doesn't fix it. A long narrative embedded as
one vector is still an average, and the funnel admits at most 512 tokens
of memory per turn and stops at the first entry that doesn't fit. Records
have to be both more detailed and smaller.

## Requirements

Approved 2026-11-08; requirement 3 amended the same day after design
review (see Rejected alternatives, "cut ranges into token spans").
Tags: *stated* (the owner said so), *memory* (established project
context), *inferred* (approved assumption).

1. **Records keep concrete detail**, so the agent can recall accurately.
   The consolidation instruction asks for step-by-step first-person
   narrative that keeps exact facts and opinions as expressed and the
   actions taken, without verbatim quotes or tool output, and has no
   "few sentences" ceiling. *(stated)*
2. **Told stays apart from inferred**, so recalled guesses don't come back
   as facts. The instruction keeps that distinction. *(inferred)*
3. **Each record is a short, self-contained entry about one stretch or
   topic**, so its embedding isn't a centroid and it fits through
   admission. The turn trigger keeps the backlog each consolidation covers
   near N tokens, N configurable. *(stated, amended)*
4. **Small records don't cost extra prefill or slot switches.** The record
   is written by a fork of the live window; the turn trigger fires at a
   turn boundary, never inside a tool loop. Compaction's existing inline
   fork can still run mid-loop. *(stated; amended 2026-11-08 to scope
   "never inside a tool loop" to the turn trigger)*
5. **No recall gap at compaction**: compaction consolidates only the rows
   past the watermark, normally few. *(memory)*
6. **Retrieval doesn't echo the window**: a record whose message range is
   still in the live window is not admitted. *(stated)*
7. **One large entry can't block admission**: an entry that doesn't fit
   the remaining budget is skipped and smaller ones after it are still
   admitted. *(inferred)*
8. **The result is measurable**: this doc names the instrumentation and
   query set to evaluate against, without redesigning #407. *(memory)*
9. **Later reconsolidation stays possible**: each record carries its
   message range, prompt hash and how it was made. *(stated; "the budget
   that bounded it" became "how it was made" with requirement 3's
   amendment, since a backlog threshold doesn't bound a range)*
10. **The PRD changes are listed** (below). *(stated)*

Settled with the owner: the new behaviour is the default for every
deployment (there are two, on one machine), and the turn trigger needs
forks, doing nothing when consolidation doesn't fork. A larger memory
admission budget is wanted, but only alongside the rest of this design.

## Design

### 1. One consolidation, several entries

Every consolidation (from the compaction, idle and eval triggers, the new
turn trigger, and the catch-up path any of them can fall through to)
makes the same single call it makes today, a fork or a catch-up view, and
gets back a list of entries instead of one summary. Each entry
becomes one row in `memory`.

The reply shape (`CONSOLIDATION_SCHEMA`, sent as `response_format`):

```json
{
  "entries": [
    {"narrative": "<first-person narrative of one stretch or topic>",
     "topic_tags": ["<short topic tag>", "..."]}
  ],
  "participants": ["<sender name>", "..."],
  "importance": "<number from 0 to 1>"
}
```

- Entries with an empty `narrative` are dropped. If none remain, the reply
  is unusable, handled as a parse failure is today on each path (a skip
  for idle and turn, the catch-up view for a compaction fork, the raw
  transcript after two unusable catch-up replies).
- `participants` and `importance` describe the whole consolidation and are
  shared by its entries. An `importance` that doesn't parse as a number
  (a provider that ignored `response_format` and echoed the placeholder)
  is treated as absent; a numeric string ("0.7") is converted.
- A reply in today's shape (`{"summary": ...}`) is read as one entry. It
  can only come from an override prompt that still carries the old
  template, sent to a provider that ignores `response_format`.

The entries of one consolidation are stored in one `_commit`: one
watermark compare-and-set, then one `_insert_record` per entry, in the
same transaction. They share `msg_id_start`/`msg_id_end`, `created_at`,
`importance`, `valence`, `participants` and the stamp; each has its own
`summary` (the narrative), `topic_tags` and embedding. Each entry is
embedded separately; one embedding call with a list of texts does it.

Two existing behaviours need a line each:

- **Near-duplicate merge skips siblings.** The dup check excludes the ids
  already inserted by this `_commit`, so two entries of one consolidation
  never merge into each other. They are adjacent steps of one story and
  can be close in wording; merging would supersede one and lose its
  detail. Excluding by id, not by shared range, still holds when an
  earlier sibling has merged with an older record and widened its range.
  Merging with older records works as today, but will fire more often
  with short entries; §8 measures it.
- **The importance prior is scored once per consolidation**, as today
  (it is already scored over the range, not the summary). See §4.

Unchanged, and fine with several records per range: `recall_raw` (replays
the shared range), `redact` (the `memory` form already cascades to every
same-channel record overlapping the range, so siblings go together),
retention (per record), the fork stash (holds one reply), and the
watermark (one step per consolidation).

### 2. The instruction

`DEFAULT_CONSOLIDATION_FORK_PROMPT` and its documented copy,
`prompts/memory_consolidation_fork.md`, change. A deployment that sets
`memory.consolidation_fork_prompt` keeps its own text (Lex doesn't), but
the schema change applies to it too. The opening paragraph ("Pause the
conversation here and write a record of part of it... This record is your
private memory...") stays as it is. The new body, replacing the paragraph
after it and the JSON template:

> Narrate this stretch of the conversation from your own point of view,
> step by step, in your own words: what was said, what you did, what you
> learned, and what was decided or left open. Don't quote anyone verbatim,
> but keep facts and opinions exactly as they were expressed, and who
> expressed each (names, numbers, dates, identifiers, preferences,
> commitments, corrections), so you can recall them accurately later.
> Keep what you were told apart from what you inferred or speculated
> ("so-and-so told me...", "I guessed that..."). Say what you did with
> tools and what it told you, not their raw output. Bracketed CONTEXT
> blocks are things you recalled or were shown, not new events; don't
> record them as if they happened now. Skip pleasantries and chatter.
>
> Write it as separate entries, one for each distinct stretch or topic,
> in order. Each entry is a paragraph at most and must make sense read on
> its own, months from now, with nothing else: name the people, projects
> and things involved rather than "he" or "it". Don't attribute a fact
> from one entry's topic to another's.
>
> Then rate how important this stretch is to remember long-term, from 0
> (routine) to 1 (essential). Do not call tools.

followed by the JSON template above.

The range sentence `MemoryPlugin` appends ("Record the conversation above
from the message that begins «…» onward...") is unchanged. It quotes the
first user or assistant text in the range, falling back to the first row
only when there is none.

The instruction text is stamped as today, so records written under it are
findable by `prompt_hash`.

### 3. The turn trigger

A new trigger, `turn`, beside compaction, idle and eval. `MemoryPlugin` implements `on_agent_response`,
which fires once per agent loop after the final reply text is produced,
never on the tool-call branch. It is awaited before the reply is sent, so
the hook only makes in-memory checks and spawns:

1. If `fork_agent(self.pm, "consolidation")` is `None`, return, silently;
   with forks off this would otherwise warn on every turn.
2. If `channel.pending_tool_call_ids` is non-empty, return. A reply can be
   produced while tools are still outstanding (a user message mid-loop);
   that isn't a turn boundary.
3. If the channel is in the in-flight set, return. Otherwise add it,
   before any await, and spawn the task, which removes it when it ends.
   The set idle uses (`_idle_inflight`) becomes the set for both
   triggers, and idle's own check gets the same add-before-await order.
4. Before spawning, record `range_end`: the highest `_db_id` in the window
   at this moment. Read in the hook, synchronously, it is the turn's last
   row; read later in the task, it could include a message queued during
   the turn and processed after the send.

The task:

1. Reads the channel's watermark.
2. Sums `message_tokens` (`corvidae/context.py`, the window's estimator)
   over the window's messages with a `_db_id` above the watermark and at
   or below `range_end`, skipping CONTEXT, SUMMARY and UNSENT messages. Window copies, not
   `message_log` rows: they are what the fork will read, and they don't
   carry stripped reasoning. CONTEXT is skipped because admitted memories
   are not new conversation.
3. Below `memory.consolidate_after_tokens` (N), stops.
4. Otherwise calls `_consolidate_range(channel_id, range_end, "turn")`.
   Using the window's highest id from the hook, not `MAX(id)` from
   `message_log`, keeps a row persisted after the hook out of the range.

`_live_record` treats `turn` like `idle`: a range wholly in the live
window is forked with `_fork_record`, and a fork failure or unusable reply
is a skip that leaves the watermark, so the next turn retries (PRD
§3.8.5). A `turn` range not wholly in the window is also a skip, never the
rebuilt view; that only happens when the watermark lags behind the window
(a compaction consolidation failed), and compaction or idle owns that
case. That skip logs at debug, not warning, since it would otherwise
repeat every turn until idle or compaction catches up.
`_skip_consolidation`'s log text, which says "the next idle trigger
retries", names the trigger instead.

N defaults to **4000**, is read at startup like `idle_consolidate_after`,
must be a positive integer, and is rejected naming the key otherwise.

The fork renders the whole live window with the instruction appended, so
its prefix is the main window's: prefill is about the instruction's
length, and the main turn's next request rolls back to a checkpoint near
the window's end. The cost that grows is decode.

On a remote API the fork re-sends the whole window, but that window is
the prefix of the main request the provider just served, so where the
provider caches prefixes (automatically, or when the request marks them)
nearly all of it is billed at the cached-input rate, a fraction of the
normal input price. What a fork costs there is the cached prefix, the instruction and
the decode. A deployment that wants fewer forks raises
`consolidate_after_tokens`.

What happens to the other triggers falls out of the watermark:

- **Compaction** still forks once in `before_compaction`, inline, over
  `(watermark, range_end]`. With the turn trigger running, the watermark
  is usually within N of the window's end and compaction keeps far more
  than N, so the range is usually empty and `before_compaction` returns
  early. It is large when compaction fires in
  the middle of a long tool loop, since the turn trigger doesn't fire
  inside the loop;
  then one fork writes several entries over it. The entries are small
  whatever the range's size, which is what requirement 3 asks; the decode
  is on the response path, as the compaction fork's always has been.
- **Idle** is unchanged except for the in-flight ordering. It picks up
  whatever backlog is left when a channel goes quiet, one fork, several
  entries.
- **Catch-up** is unchanged, including segmenting at `max_context_tokens`;
  each segment's view returns entries.

Overlap between triggers is the existing watermark compare-and-set in
`_commit`: whichever commits first wins and the other's reply is
discarded (PRD §3.7 Memory 3). The stash holds one reply per
`(channel, range_end)`, as today; if a turn task moves the watermark
first, the stashed reply is discarded and the remaining rows go to
catch-up, as today.

### 4. Importance without an extra call

`_commit` scores one prior per consolidation. With `AppraisalPrior`
registered, a range that contains an appraised exchange is scored from it
with no model call; turn ranges from user-originated turns contain the
turn's own user message, so they are normally covered. Notification
turns (heartbeats) aren't, and fall back as below. Otherwise `RubricPrior` makes a
separate model call on the `rubric` stage. On a single-slot server that
is a slot switch, and the turn trigger makes consolidations many times
more frequent.

So the fork rates importance itself (the `importance` field), and
`_commit` passes that value to `importance_prior.score` as an optional
`hint`. `RubricPrior` returns the hint, clamped to [0, 1], when one is
given, and calls the model only when it isn't (catch-up replies from a
provider that dropped the field, older override prompts). `AppraisalPrior`
ignores the hint where appraisal covers the range and passes it to its
`RubricPrior` fallback where it doesn't.

### 5. Retrieval skips what the window still holds

Passive retrieval excludes records that belong to the current channel and
whose `msg_id_start` is at or past the oldest `message_log` id still in
its window (`min(window_db_ids(conv.messages))`), that is, records whose
whole range is still in view. A record that straddles the compaction cut
is admitted: part of it echoes the window, but keying on `msg_id_end`
would hide its dropped part until the next compaction, about 66k tokens
later on Lex. A window with no ids applies no exclusion.

Both conditions are needed: `message_log` ids are global across channels,
so an id comparison alone would also drop other channels' recent records.

The exclusion applies before candidates are cut to k, or a long
conversation's own in-window records fill all k = 8 slots:

- `retrieve()` takes an optional exclusion `(channel_id, min_msg_id)`.
  `before_agent_turn` passes it. The other caller, `scripts/eval_memory.py`,
  doesn't; `search_memory` doesn't call `retrieve()`.
- `_vec_candidates` and `_fts_candidates` apply it in their `WHERE`
  clauses, and the KNN asks for `k × 4` plus the count of excluded records
  (one `COUNT(*)` on `memory`). The KNN is exact and brute force, so a
  larger k costs the same scan. (#407 later moved the vec query's exclusion
  from `WHERE` to a selected flag; the rows returned are the same.)

This narrows what is admitted to *this* window, not what can be retrieved.
Other channels' records are untouched, and a record is retrievable
normally once its rows leave the window.

### 6. Admission: the funnel skips, and Lex's budget grows

`FunnelPlugin.admit`'s budget loop skips an entry that would overrun the
budget and keeps going, instead of stopping. Entries are still admitted in
the given order, so priority holds among what fits, and the drop count is
still logged. This applies to every source (memory, goals, critique): none
wants one large entry to block the smaller ones after it, and the
deferred-drain leftover logic already treats any unadmitted entry the same
way.

The memory budget stays 512 by default and goes to **2048 on Lex**
(`funnel.budgets.memory`, matching its `goals` budget). An entry is a
paragraph, roughly 100 to 250 tokens, so 512 holds two to four and 2048
eight or more, which is k. The default is sized for the default 24k window;
Lex's window is five times that.

Admitted memory costs window space until compaction, not just for the
turn. Each admission appends one persisted CONTEXT message at the window's
tail; the window is append-only, so it stays until compaction drops it.
Dedupe stops an identical line being admitted twice, but each line
carries its age ("12m", "5h"), so a record retrieved again after its age
string changes is admitted again. At 2048, ten user turns that each admit
the full budget add about 20k tokens against the roughly 66k Lex grows
between compactions. The budget is a cap, not a spend, but weak matches
don't keep it unspent: when every candidate is weak, all of them are
admitted, and on Lex's bands that is common. §8 measures what is
actually spent.

### 7. Records say how they were made

A nullable `trigger TEXT` column on `memory`, holding the trigger that
started the consolidation (`turn | compaction | idle | eval`, the values
`_consolidate_range` is called with; NULL for direct records and records
written before it existed), added with `add_column_if_missing` like
`prompt_hash` and listed in `MEMORY_DDL`, and carried on `RecordStamp`.
Catch-up is a path, not a trigger: a catch-up record carries the trigger
that fell through to it and `mode = standalone`, which already marks the
view call. With the message range, `prompt_hash`, `mode` and `model`, a
later pass can select records by how they were made; catch-up's segment
size is the channel's `max_context_tokens`, readable from config.

### 8. Measurement

Not designed here; listed so the change can be judged:

- **#407**: one `retrieval_log` candidate row per candidate with
  similarity, score, band, whether it was excluded as in-window, and
  whether it was admitted. Implemented as the `retrieval_candidate` table
  (see [cognition.md](cognition.md), "Retrieval (read path)").
- **A labelled query set**: twenty to forty real queries from Lex's
  `message_log`, each with the record(s) that should come back.
  `scripts/eval_memory.py --snapshot` (#658) scores such a set: it runs
  the real retrieval path over a copy, taken through a read-only connection, of an existing
  `sessions.db`, with labels keyed to its memory ids, and writes each
  query's scored pool (see [cognition.md](cognition.md), "Measuring
  retrieval on a real database").
- **Before and after**: the same queries against a snapshot consolidated
  the old way and one consolidated the new way. Recall@k and MRR, entry
  lengths, tokens admitted per retrieval, how often an entry is skipped
  for not fitting (§6), and how often near-duplicate merge fires, since
  short routine entries ("the station read 12°C" against "14°C") may
  reach `dup_threshold` (0.95) and need it raised.

## Decisions

- **The model chunks, not the code.** One call per consolidation, many
  entries. The model knows where one topic ends and another starts; a
  token count doesn't, and cutting by tokens fought every piece of the
  existing one-record-per-range machinery (Rejected alternatives).
- **Entries share their consolidation's range.** Finer ranges per entry
  would need the model to cite message boundaries. Everything that uses
  ranges (window exclusion, `recall_raw`, redact, reconsolidation) works
  at range level.
- **The turn trigger counts window tokens, CONTEXT excluded, and doesn't
  cut.** N bounds how much a consolidation normally covers; entry size is
  the instruction's job.
- **4000 tokens default.** A few minutes of active work. A guess, to be
  revised against the query set.
- **The prompt changes for every trigger**, so records don't come in two
  styles depending on which trigger wrote them.
- **Importance comes from the fork**, so the turn trigger doesn't add a
  rubric call per consolidation on a single-slot server.
- **The window exclusion is a `retrieve()` parameter**, applied in the
  candidate queries, so it acts before the cut to k and only passive
  retrieval uses it.
- **The funnel change is funnel-wide**, a one-line semantic change.
- **Lex's memory budget grows with the rest of this, not on its own.**
  With today's records a bigger budget admits more centroids.

## Rejected alternatives

- **Cut ranges into token spans, one fork per span.** The previous draft
  of this design. Span boundaries inside tool loops broke the range
  sentence, spans of pure tool work had no defined record, and the
  cutting had to spread into idle, compaction's stash and every commit
  path. Two design reviews failed on it.
- **Compact more often.** Same small records, but every compaction
  re-prefills the main window.
- **A wall-clock timer** ("once a minute"). Fires inside tool loops and on
  quiet channels.
- **Firing inside the tool loop** when the backlog passes N. Requirement 4
  rules out the turn trigger firing inside a tool loop.
- **A larger memory budget on its own.** Admits more of today's records.
- **Narrating again until it fits** (split a span, narrate each half).
  A retry loop on the response path at compaction, and smaller spans
  fragment the story.
- **Admitting a one-line gist with an id**, full narrative behind a tool.
  Depends on the model making the call; the gist is a summary again.
- **Truncating at admission.** Loses whatever is past the cut.
- **A rebuilt-view fallback when forks are off.** Full prefill per
  record; rejected for least mechanism. With forks off, compaction falls
  through to the catch-up view and idle skips an in-window range with a
  warning, as today, both under the new instruction.
- **Narrowing retrieval by channel.** Principle 14 is settled; the
  exclusion removes only rows this window holds.

## Deferred

- **Compaction's fork off the response path**, by forking against a
  snapshot of the pre-compaction window in a background task. Needs the
  fork API to accept a rendered message list.
- **Compaction waiting for an in-flight turn task** on the same range
  instead of forking it again. Both run and `_commit` keeps one reply.
  The cost is duplicate decode on the response path, and, when the turn
  commit lands between the compaction task's watermark read and its
  commit, the compaction reply is discarded and the rows past the turn's
  range (now out of the window) wait for the next idle or compaction.
  The same race exists between idle and compaction today; the turn
  trigger makes it more frequent.
- ~~**Exclusion after a cross-channel merge.**~~ Done in #627: the
  survivor is marked `merged_foreign` and never window-excluded. Originally: A cross-channel near-dup
  merge replaces the other channel's older record with one in the current
  channel whose range is in the window, so the older content is excluded
  until those rows leave. Rare; small fix.
- **Forking with a `cut` just past `range_end`** for turn and idle, so a
  message processed between the hook and the fork's render isn't
  narrated too and then again by the next consolidation. Idle has the
  same window today; near-dup merge is the backstop.
- **Backoff for a fork that keeps failing.** Each turn past N re-forks a
  growing range with a warning until compaction takes it to catch-up.
- **Dedupe on summary text, not the whole line**, so a record isn't
  re-admitted when its age string changes (existing behaviour, more
  frequent with many recent records).
- **`redact memory --dry-run` reports one affected memory** while the real
  run cascades to siblings (existing behaviour).
- **A default memory budget derived from the window** (say 1.5% of
  `max_context_tokens`) instead of a fixed 512.
- **Per-entry message ranges**, if reconsolidation or `recall_raw` turns
  out to need them.
- **Reconsolidating existing history** with the new instruction, and the
  operator command for it (the plan in memory-retrieval-fidelity.md).
- **Retiring obsolete memories.** Lex's local plugins have related work;
  none of it is in main.
- **Relative banding (#409)** and a reranker. Tune against the new records.
- **How much action detail to keep.** "I listed the directory, found
  agent.yaml, added the log level" is fine for now.
- **Catch-up segment budget** decoupled from `max_context_tokens`. Belongs
  with reconsolidation.
- **The catch-up raw-transcript fallback.** After two unusable replies,
  `_view_record` stores the raw transcript as one record of up to a
  segment's size. A known degradation, rare; splitting it is small.
- **Funnel entries larger than the whole budget** are re-queued by the
  deferred drain forever (existing behaviour). Dropping them is small.

Implementation notes for the Green gate: test stubs assigned to
`memory.importance_prior` (e.g. `tests/cognition/test_memory_dedup.py`,
`test_memory_consolidation.py`, `test_memory_idle_clock.py`) need to
accept the new `hint` argument; `harness/consolidate_replay.py` reads the
reply shape (`summary`, `topic_tags`, `participants`) and needs updating
to `entries`.

## PRD changes

Each needs the owner's sign-off before implementation.

- **§3.7 Memory 1** names compaction and idle as the triggers. Add the
  turn trigger: at the end of a turn with no tool calls pending, when the
  channel's un-consolidated window messages (CONTEXT excluded) reach
  `memory.consolidate_after_tokens`; live-window work that forks or skips
  per §3.8.5; needs forks for consolidation. A range that has left the
  window is left to compaction and idle, which reach the view call as
  today.
- **§3.7 Memory 2**: a consolidation writes one or more records; "a
  first-person summary" becomes "a self-contained first-person narrative
  entry about one stretch or topic"; records of one consolidation share
  its range, importance and stamp; the stamp includes the trigger. The
  default importance prior uses the consolidation's own rating when it
  gave one, and the cheap-model rubric otherwise.
- **§3.7 Memory 3**: the near-duplicate check skips records of the same
  consolidation, and "Overlapping triggers on one range keep one record"
  becomes "keep one consolidation's records".
- **§3.8 item 5** says a live-window background stage that can't run
  skips and logs a warning. Add that the turn trigger is not attempted at
  all when consolidation doesn't fork, so it neither skips nor warns;
  compaction and idle keep the warning.
- **§3.7 Memory 4**: passive retrieval excludes the current channel's
  records whose whole range is still in its live window, before
  candidates are cut to k. Say explicitly that this is not a retrieval scope, so it
  reads as consistent with principle 14.
- **§3.7 Funnel**: "greedy, drops logged" becomes "in priority order,
  skipping entries that don't fit, drops logged".
- **§3.7 Memory 7** needs no change; the instruction keeps topics apart
  by construction now, one entry per topic.

## Corrections to memory-retrieval-fidelity.md

- **Finding 1 misreads #409.** The right Yoz record ranked first for its
  own query (0.427). The 0.463 is the top hit for a different query,
  sourdough. What #409 shows is that no fixed threshold separates those
  numbers, which is finding 3, not a ranking failure. The centroid
  argument holds, but on the compression ratio, not on that evidence.
- **The compression ratio is understated.** The doc puts default config
  at about 10k tokens per record; it is closer to 7k (0.8 × 24k compacts
  down to a 12k retain budget). On Lex it is about 66k.
- **The 0.75/0.60 thresholds** are the code defaults. The `[weak]` Yoz
  result came from Lex's `band_on: similarity` at 0.60/0.50.
- **"The distribution is free"** needs a code change: `retrieve()` keeps
  k = 8 candidates of a k × 4 KNN, not every distance.
- **Step 1 needs more than labelling.** `scripts/eval_memory.py --live`
  drives the real retrieval path, but only over fixtures it ingests into
  a scratch database; scoring Lex's real records needs a snapshot mode.
  That mode now exists: `scripts/eval_memory.py --snapshot` (#658).
- **Prompt variant B is smaller than it looks.** The current instruction
  already asks for names, decisions, commitments, corrections and
  outcomes; the limit is "a few sentences".
- **Omitted:** `recency_from: last_used` already exists and softens
  "old means weak" for memories that get used.
