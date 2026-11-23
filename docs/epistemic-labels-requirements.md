# Epistemic labels: requirements

A claims plugin records what the agent has asserted, one claim per row, and
labels each row with the evidence that backs it. The label is computed from
pointers the store can check. Nobody is asked how sure they are.

**Status: draft, not accepted.** Schuyler owns acceptance. Nothing here is
built, and no issue is `ready`. This follows the multi-phase critique work
(#1169, #1179, #1149, #1075) in the roadmap.

It started as Lex's sketch of 2026-11-22 (with Schuyler: rubric phrasing,
trust tiers, staging). The sketch's requirement numbers (R1 to R8) are cited
where this document keeps, changes or drops one. Code and doc references
were read against `main` at `98ec754`.

## Why

The agent said it had "grepped the record" seconds after no grep ran. It was
not lying; confidence leaked into the report. Four premises follow from that
and from the other incidents in [Fixtures](#fixtures):

1. **Self-grades don't work.** Any mechanism that asks the claimant "how
   sure are you?" inherits the failure. A label is audit evidence, never
   self-assessment.
2. **A trusted instrument is an unchecked one.** A critique lens once quoted
   a line that was not in the record it cited. No check here may rest on
   believing a judge, this system's own included.
3. **Corroboration counts failure families, not instances.** Two statements
   by people fail the same way. A live probe fails differently: it is down,
   not confabulating.
4. **There is no stable bottom, and that is accepted.** A label names what
   would dissent, and staleness says when that naming last got tested.

## Decisions

Schuyler, live session 2026-11-23. Each row is a choice between options
whose costs were stated at the time; the cost of the one chosen is kept
here because the requirements inherit it.

| # | Decision | Cost accepted |
|---|---|---|
| D1 | Claims live in a new plugin with its own table. Narrative `memory` records stay unlabelled. | A second store: its own retrieval, rendering, redaction and retirement. |
| D2 | An idle lens writes claims after the exchange. An agent tool, consolidation, and appraisal's `past_claims` remain possible writers; none is required. | The writer is a model reconstructing pointers, so the store's checks carry the weight. |
| D3 | `observed` requires a verbatim quote found in the tool-result row it points at. **Held lightly.** | A claim that summarises or computes from output has no quote and lands `reported`. E12 measures how often. |
| D4 | The agent runs falsifiers, in a heartbeat or idle turn, with its ordinary tools. Code never executes a stored falsifier. | A model turn per falsifier. Most rows will be stale most of the time. |
| D5 | The store derives `family` from what a pointer resolves to. The lens may only move it down. | The patch for self-corroboration depends on the lens noticing. |
| D6 | Only a re-run falsifier flips a claim to `contested`, for now. A judged flip is a later stage. | "You said X, now you say not-X" is not caught until then. |
| D7 | Claims reach the window as a recall source with its own funnel budget. | A second retrieval per user turn, more context, and the same fact can appear twice. |
| D8 | Delivery is not a schema special case. "Published" is an ordinary claim with its own falsifier. | Nothing forces "it's published" to be split into stages. |
| D9 | Stakes-matching stays on #1149. This document defines only the lookup. | Until #1149 lands, nothing enforces labels on outgoing replies. |

## Terms

- **Claim**: one assertion, in one row, in the claims table.
- **Receipt**: a pointer from a claim to a `message_log` row, with a quote.
- **Family**: how a receipt can fail. `probe` (a tool result), `testimony`
  (something a participant said), `model` (the agent's own output, or no
  receipt at all). `record` is reserved; see open question 1.
- **Basis**: the label. `observed`, `attested`, `reported`, `prior`, plus
  the states `stale` and `contested` that render beside it.
- **Falsifier**: a command, the string its output must contain, and a
  sentence saying what a miss would mean.

## Requirements

### E1. One claim per row, in its own plugin

A claims plugin under `packages/` owns a claims table and a receipts table
in `sessions.db`. A row holds one assertion, the id of the trace it was
made in, its channel, and when. `memory` rows are not labelled and not
changed. (R1, narrowed by D1.)

Accept: disabling the plugin leaves memory, critique and the kernel suite
unchanged.

### E2. Receipts are checked, not believed

A receipt names a `message_log` id and carries a quote. The store accepts
it only when the row exists, is not a redaction tombstone, and contains the
quote verbatim. A receipt that fails is stored as failed with the reason,
and counts for nothing. (R2, R6; strengthened by D3: a pointer that
resolves is not enough, because a writer can cite a real row that says
something else.)

Accept: a claim citing a real tool row with a quote that row does not
contain is stored `reported` or `prior`, and the failure is queryable.

### E3. Family is derived

The store sets a receipt's family from the row it points at: a tool-result
row is `probe`, a user message is `testimony`, an assistant, summary or
CONTEXT row is `model`. The writer cannot set it. The writer can lower a
`probe` to `model`, to mark a tool call that only read back the agent's own
writing. No input raises a family. (D5; replaces R1's declared `family`.)

Accept: one query lists every receipt whose stored family differs from the
one its row implies, and every difference is a downgrade.

### E4. Basis is computed and capped

The store computes the highest basis a claim's valid receipts support, and
stores the lower of that and what the writer proposed. There is no numeric
confidence anywhere. (R1, R8.)

- `prior`: no valid receipt.
- `reported`: valid receipts, none of them a `probe`. Any number of
  `testimony` receipts is still `reported` (R3: repetition upgrades
  nothing).
- `observed`: a valid `probe` receipt.
- `attested`: valid receipts from two or more distinct families, `model`
  not counted. Promotion is a logged event on the row. (R5.)

Consequence, stated because it departs from R3: a document a participant
forwards is `testimony` however reputable its author, until a second family
corroborates it. The store sees who delivered a row, not who wrote what was
in it.

Accept: a writer proposing `observed` with no receipt gets `prior`. That is
the grep-claim fixture.

### E5. Recall never promotes

Retrieval count, recency of retrieval and repetition in conversation are
not inputs to basis. (R5's "highest-value single clause".)

Accept: retrieving a `reported` claim any number of times leaves its basis
and its receipts unchanged.

### E6. Who wrote the row

Each claim records the trace, channel and writer that produced it, and the
writer's stamp (prompt hash and model, as `memory` records carry). This is
the witness axis R8 asks the schema not to preclude: heartbeat, idle and
channel turns have different track records.

### E7. The writer is an idle lens

A lens runs after an exchange, off the main turn, and submits the claims
the reply made, with receipts drawn from the exchange's own rows. It never
delays a reply. Other writers may be added without changing the store,
because E2 to E4 apply to every submission. (D2.)

Accept: a store-level test submits claims with no lens loaded.

### E8. Falsifiers

A claim may carry a falsifier: a command, an `expect` string, and a
sentence saying what a miss means. A falsifier is stored text and is
treated as data everywhere. A claim with no falsifier can never rise above
`reported`. (R2.)

The first run sets `verified_at`. A falsifier whose first run does not
contain `expect` is rejected, since it would have refuted the claim it was
written for.

### E9. Staleness and re-runs

A claim with a falsifier is `stale` once `verified_at` is older than a
configured age. Staleness is the normal state of most rows (D4) and is
never an alarm.

In a heartbeat or idle turn the plugin shows the agent a few stale claims
with their falsifiers, fenced as data. If the agent runs one, the store
finds the tool call in that turn's rows whose command equals the stored
command, and:

- output contains `expect`: `verified_at` refreshes and the tool row
  becomes a new receipt;
- output lacks `expect`: the claim becomes `contested`, with the tool row
  as the dissenting receipt. (D6.)

No matching tool call means nothing changes, and the skip is logged. The
agent saying it ran the falsifier changes nothing. Code never executes a
falsifier. (R5; open question 1 of the sketch is answered: falsifiers run
on staleness and audit, never on admission.)

Accept: the self-healing demo of R5. A stale claim is re-run and clears; a
second is re-run against changed state and renders `contested`.

### E10. Audit

The same re-run path also takes a random sample of claims that are not
stale. The share of re-runs that end `contested` is the subsystem's health
metric, and it is one query. Receipts are also re-checked (E2) each time a
claim is rendered, so a row redacted or a quote invalidated after the write
demotes the claim then. (R6.)

Accept: a fixture plants a receipt whose quote is absent from its row; the
render path demotes the claim and logs it. An audit that only reports on
honest rows fails this.

### E11. Rendering

The plugin is a recall source (`recall.py`). Each hit renders as one line
whose label is built by code from the stored row:

```
[observed: probe msg 74301, verified 3d] the staging deploy returned 200
[reported: stated by Schuyler, msg 74268] the licence allows redistribution
[attested: 2 families, stale: falsifier not run since 11-02] ...
[contested: falsifier failed msg 75112] ...
```

A demoted, stale or contested claim differs in the label, not in wording
the agent chooses. How the agent hedges on reading a label stays a prompt
matter (PRD Memory 9). (R7, D7.)

### E12. Every rule has a one-query probe

For each of E2 to E5 and E9 the plugin's docs give one SQL query that
counts violations, and a test runs it. A rule without one is not
implemented. The docs also give the query for D3's cost: the share of
claims proposed `observed` that were stored lower for want of a verbatim
quote. D3 is revisited when that number exists. (R8.)

### E13. Lookup for other plugins

Another plugin can ask what backs a claim and get the basis, state and
receipts. Under the plugin-boundaries rule (#1178) that is a hook the
claims plugin owns, not a held reference. The pre-send check (#1149) is the
expected first caller; what it does with the answer is decided there. (D9.)

## Not building

- Numeric confidence, or any continuous trust dial. (R8.)
- Trust weighting between sources. Who said it is #138's axis; what backs
  it is this one. Neither replaces the other.
- Quarantine by default for external material. (R8.)
- Code that executes stored commands. (D4.)
- A contradiction judge, in the first stages. (D6.)
- `composed` and `delivery_ref` fields. (D8; drops R4's schema, keeps its
  fixture.)
- Stakes-matching and register tags. (D9.)
- Migration. No belief rows exist to migrate (sketch open question 4).

## Against the PRD

Read against `docs/prd.md` at `98ec754`. The PRD does not mention a claims
store, so its tools, config keys, prompts and storage are plugin-defined,
with this document as their provenance. Two items need the owner.

| PRD | Reading |
|---|---|
| Principle 1 (plugins; no undeclared dependency) | Optional plugin. E13 is a hook, per #1178. |
| Principle 3 (background work off the main turn) | The lens (E7) and re-runs (E9) are idle work. |
| Principle 4 (append-only log) | Receipts point at `message_log`; nothing rewrites it. A tombstoned row fails E2 and the claim demotes. |
| Principle 7 (epistemic checks never rest on generator confidence) | Agrees. Basis is computed from receipts. |
| Principle 8 (one SQLite file) | Two tables in `sessions.db`. |
| Principle 9 (everyone else is text) | Claims and falsifiers distilled from untrusted text are data: rendered through the funnel's framing, fenced in the idle turn. Needs a `docs/threat-model.md` entry. |
| Principle 14 (one actor, many channels) | Claims and receipts span channels, as memory does. |
| Principle 16 (compute is finite) | Re-verification is rationed by model time (D4). |
| §3.7 Memory 9 (calibration fragment never auto-injected) | Labels are structure. Hedging vocabulary stays persona. |
| §3.7 Semantic facts ("a self-distilled fact never satisfies the provenance gate") | Same rule, mechanical: `model` receipts never count toward `attested`. |

**For the owner:**

1. **Should the provenance check read claims?** PRD §3.7 Critique 2 fixes
   its evidence as the window plus a keyword search of raw messages. Giving
   it claims as evidence changes a PRD requirement. This document does not
   ask for it.
2. **The idle-turn seam.** E9 needs a plugin to put a section into a
   heartbeat or idle turn. `grep -rn collect_heartbeat corvidae packages`
   finds no such hook at `98ec754`. If the after-send idle lenses of #1149
   add one, E9 uses it; otherwise this is a hook request, and the kernel
   takes a hook only on two use cases.

## Open questions

1. **`record` receipts.** A commit or a file is a third family, but a
   receipt can only point at `message_log`, where a `git log` result is
   just a `probe`. Supporting `record` means the store resolves a commit
   sha or a path and content hash itself. Proposed: reserve the value,
   build it in stage 3.
2. **Is `expect` as a substring enough?** It is mechanical and matches E2,
   but it can't express "exit status 0" or "returns nothing". The sketch
   wants an accepted null logged as `observed: probe returned nothing`.
3. **What counts as "read back the agent's own writing"** (E3) beyond the
   lens's say-so. Operator-listed paths were offered and not chosen.
4. **How many stale claims per idle turn**, and which first: cheapest
   falsifier, oldest, or most retrieved.
5. **Claims and `corvidae redact`.** A receipt's quote is a copy of
   conversation text outside the redaction cascade, which is on hold
   (#1005).
6. **Retirement.** A `contested` claim stays retrievable. Whether it ever
   leaves retrieval is #629's question, for claims as for records.
7. **Duplicates.** The same assertion made in ten exchanges is ten rows or
   one row with ten receipts. E5 holds either way; rendering does not.

## Stages

1. **Store and labels.** E1 to E7, E11, E12, E13. Claims are written,
   checked, labelled and rendered. No falsifiers: nothing goes stale or
   contested yet, and nothing exceeds what its receipts support.
2. **Falsifiers.** E8 to E10. Needs the idle-turn seam.
3. **Later.** A judged contradiction flip that logs both receipts so the
   call can be checked by reading two rows (D6); `record` receipts; any
   writer beyond the idle lens.

## Fixtures

Production incidents, to be used as acceptance tests.

| Incident | Tests |
|---|---|
| "Grepped the record" when no grep ran (11-22) | E2, E4: `observed` with no valid receipt is stored `prior` |
| "His error list", "posts on his shelf" (11-22) | E4: a claim about someone else's state is `reported` without a probe |
| Critique lens quoted an absent line (11-21) | E2, E10: a quote missing from its row fails, at write and at render |
| "Post is done" for posts never pushed (11-18, 11-19) | E8, D8: committed, pushed and served are three claims with three falsifiers |
| Fabricated licence grant, retrieved again as fact (11-22) | E5: retrieval does not promote |
| Water-droplet paper returned for "footing" (11-22) | Out of scope (retrieval ranking, #674); see open question 2 for the accepted null |

## Related issues

- **Callers of the lookup:** #1149 (pre-send check), #1075 (pre-tool-use).
- **The existing confabulation check:** #478, #162, #387, #158.
- **The other axis:** #138 (trust tagging and quarantine).
- **Lifecycle:** #629 (retirement), #327 (reflection).
- **Semantic facts, the other claim-shaped store:** #137, #143.
- **Schema changes:** #14 (migrations).
- **Retrieval and rendering:** #670, #674.
- **Plugin boundaries:** #1178, #1179, #1169, #1175.

## Not read

Two companions to the sketch could not be reached when this was written:
the wiki note `Research/Epistemic-Trust-Hierarchies` and Lex's
`values/corvidae.md` and `values/epistemics.md`. Anything they settle that
this document contradicts is an error here.
