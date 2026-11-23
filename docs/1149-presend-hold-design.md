# Design: hold-and-revise before send (#1149 with #1169 part 2)

Working plan under the rule-of-two pipeline, committed for handoff as #1212 did. Delete it, or fold it into `docs/cognition.md`, before the final PR merges.
Base: `main` at `ccf960b` (after #1210, #1212, #1219).

## Requirements (approved: #1149 comment 7097300286, accepted 7097465716)

| # | Outcome | Acceptance |
|---|---|---|
| R1 | Appraisal scores every non-decline final before send. When a peer returns objections from `on_appraisal_before_send`, appraisal's `should_send_response` returns a `Refusal` carrying them. The interim path, where objections refuse nothing, goes. | A draft drawing an objection in time is not sent; one drawing none is sent. |
| R2 | Two outcomes only: sent as written, or held for revision. No withhold. | No path in the check withholds without a revising turn. |
| R3 | A held draft is stored unsent with the window marker. | Row `UNSENT`; transports never see it. |
| R4 | Exactly one revising turn, with the held exchange's origin, gets the objections as framed CONTEXT; woken through `held.hold_and_wake`. | Objections offered at exactly one turn with that origin; when admitted, framed in its window before its LLM call. |
| R5 | Hold and wake only when the draft was actually not sent. A late refusal holds nothing and wakes nothing; its objections reach the next turn through the kernel's late path. | Late refusal: reply sent, objections framed next turn, no revising turn, no held record. |
| R6 | The revising turn's stub has no silence instruction. A revision is never held. Finals between the hold and the revision's final may go unchecked. If no objection reaches the revising turn, the revision is sent unchecked with a warning naming channel and reason. | No second hold on: (a) a stub turn deferred while a tool call is outstanding; (b) another item queued ahead of the stub; (c) the revising turn's LLM call failing and being retried; (d) objections over the budget; (e) a batch left held from an earlier hold, offered at another sender's turn while a later hold is outstanding. Stub has no `NO_REPLY` instruction. The unchecked span ends at the revision's final, or, if the revising turn ends without a final, at the channel's next final after it. |
| R7 | A held draft is recorded through `on_trace_record` with the objections as reason. | Outcome log carries the held verdict and reason for the held trace. |
| R8 | Post-send critique runs no lens and no provenance check on a held trace; it judges the revision. | Held trace: no idle critique; revision: judged. |
| R9 | Without appraisal, critique's before-send sampler does R1, R4–R8 itself. | Appraisal absent, `critique.fallback_rate: 1.0`, a persistent objection: one hold, one revising turn, revision sent. |
| R10 | Check's fork fails or reply unusable: reply sent, warning names channel, trace, cause. Forks-off is stated once at startup. | Fork failure and unusable reply: reply sent, one warning with all three. |
| R11 | Appraisal reports its lens scores through `on_trace_record` in both phases, each record naming its phase. | Records for both phases for one exchange. |
| R12 | Docs and the PRD edits of 7096915201. | Doc review. |

Settled premises also include the owner rulings on #1169 (7089831895 as amended; 7093658157) and #1149 (7076260705, 7086187561, 7096915201, 7096937779).

## Plan

### Shape

One new module, `corvidae_cognition/presend.py`, holds the hold-and-revise state machine as a plain class, `PreSendHold`. Appraisal owns one instance; critique owns one and uses it only when appraisal is absent (R9). Only one of the two is ever active on a running agent, because critique's fallback runs only when `pm.has_plugin("appraisal")` was false at `on_start`. The class is not a plugin and fires no hooks of its own; its owner passes it `pm`, and it calls `held` (a kernel plugin, via `get_dependency`) and fires `on_trace_record` through the owner's `pm` (both owners already depend on `trace`).

```python
SOURCE = "pre-send check"          # held-batch source and frame label

@dataclass
class _Pending:                    # a refusal returned, outcome not yet known
    channel_id: str
    origin: str
    draft: str                     # the refused text; on_final matches it against response_text
    objections: list[str]

@dataclass
class _Revision:                   # a hold is outstanding on a channel
    origin: str
    turn_matches: bool = False     # the current turn's origin equals the hold's (set at before_agent_turn)
    turn_trace: str | None = None  # the current turn's trace (set at before_agent_turn)
    offered_trace: str | None = None  # trace of the turn at which our batch was offered; None until then
    admitted: int = 0              # our entries admitted at that turn
    failed: bool = False           # an error emission ran on offered_trace

class PreSendHold:
    def __init__(self, pm, reporter: str) -> None: ...   # reporter = "appraisal" | "critique"
    def suppressed(self, channel_id: str) -> bool: ...   # a hold is outstanding: do not check
    async def refuse(self, channel, trace_id: str, annotations: dict,
                     objections: list[str]) -> Refusal: ...
    def on_turn(self, channel, trace_id: str | None, annotations: dict) -> None: ...  # before_agent_turn
    def note_error(self, channel_id: str, trace_id: str | None) -> None: ...
    async def on_admitted(self, channel, source: str, admitted_keys: list[str],
                          duplicate_keys: list[str]) -> None: ...
    async def on_final(self, channel, trace_id: str | None, annotations: dict) -> bool: ...
        # returns True when this final's trace was held (the caller then skips idle work)
```

### Data flow (appraisal loaded)

1. **Check (R1, R10).** `AppraisalPlugin.should_send_response`, on a non-decline final with a trace:
   - If `hold.suppressed(channel.id)`: return None without forking (R6 unchecked span).
   - Else score as today and fire `on_appraisal_before_send`. Its results are now used: each non-None result is a `list[str]` of objection texts; they are concatenated in hook-result order. A non-list result or a non-string item is dropped with a warning naming the plugin result type (any plugin may implement the hook).
   - Objections present: return `await hold.refuse(...)`. None otherwise. Fails open as today (an exception logs and returns None).
2. **Refuse.** `refuse` resolves the hold origin: `annotations.get("origin")`, else `resolve_trace_origin(trace_id)`, else `"user"`. It stores `_Pending(channel.id, origin, objections)` under `trace_id` in a bounded LRU (64) and returns `Refusal(reason=<objections joined by blank lines>)`. The kernel decides in time or late; the reason is what `hold_late` prints on the late path (R5).
3. **Outcome (R5, R4, R7).** `AppraisalPlugin.on_agent_response` calls `held = await hold.on_final(channel, trace_id, annotations)` before anything else.
   - `on_final` first clears the channel's revision state if it is due (step 6).
   - Then it pops `_Pending` for `trace_id`. If present and `final_was_unsent(channel)` is true, the draft was held in time: it sets `_Revision(origin)` on the channel, calls `held.hold_and_wake(channel, origin, SOURCE, objections, stub_text=REVISION_STUB)`, and fires `on_trace_record(source=reporter, record={"held": {"objections": objections}})`. Returns True.
   - Present but sent: the refusal was late; the kernel's late path delivers it. Drop, return False.
   - The check order (clear due state, then pop pending) matters: a revision's own final clears the previous hold before any new one could be set; a new one cannot be set anyway, because the revision was not checked.
4. **Skip post-send work on a held trace (R8).** When `on_final` returns True, appraisal does not enqueue stage 2 for that trace. With no stage 2, critique gets no `on_appraisal_idle` for it, so no lens and no provenance check run. The revision has its own trace and its own final, and gets stage 2 as usual.
5. **Which turn is the revision (R6).** `AppraisalPlugin.before_agent_turn` forwards to `hold.on_turn`, which records the turn's trace and whether its `origin` annotation equals `_Revision.origin`. It fires before the held store admits (agent.py, "Runs before the held store appends"), the channel's queue is serial, and it does not fire on a retry (`if not item.retry`), so a retry keeps the failed turn's values.
6. **Offer (R6).** `AppraisalPlugin.on_held_admitted` forwards to `hold.on_admitted`. For `source == SOURCE`, while no offer has been recorded, on a turn whose origin matches: set `offered_trace` to the turn's trace and add `len(admitted_keys)` to `admitted`. Further `SOURCE` batches offered at the same turn (a stale batch from an earlier hold shares the `(channel, origin)` pair, so it pops at the same turn) add their admitted counts. The revision is whatever turn the held store offered our batch to: the stub turn, or a turn of the same origin that ran ahead of it (held pops a deferred batch at any turn of its origin that can append). The stub never decides it.
7. **Clear (R6).** At `on_final`, the channel's `_Revision` clears when `offered_trace` is set and either:
   - the final's trace equals `offered_trace` (the revision's own final, after any tool cycle; a retry keeps the trace), or
   - `failed` is true (step 8), and this is the next final on the channel.

   Until it clears, `suppressed` is true and no final on the channel is checked. A final on another trace (an interleaved turn during the revision's tool cycle) does not clear it. When it clears with `admitted == 0`, one WARNING names the channel and says the objections did not reach the revising turn (duplicates, or not admitted at that turn), so the revision went out without them.
8. **Failure (R6, input c).** `should_send_response` with `emission == "error"` calls `hold.note_error(channel.id, trace_id)`, which sets `failed` when `trace_id == offered_trace`. A retry runs the same item on the same trace, and its final clears (step 7, either bullet). With no retry, the channel's next final clears. Both finals are unchecked.
9. **Robustness.**
   - `on_final` consumes a `_Pending` only when its `draft` equals the final's `response_text`. A refusal that landed after its own `on_agent_response` therefore cannot be consumed by a later, different final of the same trace. Stale entries age out of the LRU.
   - `on_final` sets `_Revision` only after `hold_and_wake` returns. If that call raises, it logs a WARNING naming the channel and trace, saying the draft was held with no revising turn, and leaves the channel unsuppressed.
10. **Warnings (R10).** One warning per miss:
    - Appraisal's existing skip warning names the channel, the trace and the cause.
    - With appraisal loaded, critique's before-send selection no longer warns on `lenses=None` (`warn=False` on that path): appraisal already did.
    - Forks being off for appraisal's stage is logged per final at DEBUG, not WARNING; `on_start` states it once (#1169 R5.5).
    - A critique lens fork that misses keeps its own warning, naming the lens, channel, trace and cause (`_skip_lens`).

### How the acceptance inputs come out

- **(a) Stub turn deferred mid-call.** The agent does not call `admit_for_turn`, so nothing is offered; `held` re-fires the stub at `turn_ended`, and the turn that pops the batch is the revision.
- **(b) Another item queued ahead of the stub.** With a different origin it pops nothing and is ignored; its final is unchecked (suppressed). With the same origin (a heartbeat or reminder ahead of a heartbeat- or reminder-origin stub, or a tool-result turn of that origin) it pops our batch and becomes the revision. Its final clears. The stub turn that follows has nothing to pop and is checked normally.
- **(c) Retry.** See step 8.
- **(d) Over budget.** `on_held_admitted` fires per offered batch, empty included, so the offer is recorded with `admitted == 0`. The revision's final clears and warns (step 7).
- **(e) Stale batch from an earlier hold.** It pops only at a turn of its own origin. If that origin equals the current hold's, ours pops at the same turn (same pair), so the recorded turn is right. Otherwise it is not counted.
- **Matching-origin turn first** (the edge accepted in 7096937779). This is the same as (b) with the same origin. A user message carries no `origin`, so it never pops a `(channel, "user")` batch.

### Critique (R1 interim removal, R9)

- `on_appraisal_before_send` returns its objections as `list[str]` (each `_format_objection(o)`), or None. `_run_before_send` no longer calls `_record_and_register` with `waking_allowed=False` for delivery: it records the outcome (`critique_before_send`) through `on_trace_record` as today and returns the objection texts. The `waking_allowed` parameter goes, since its only caller goes.
- **Fallback (R9).** In `should_send_response`, when appraisal is absent:
  - `suppressed(channel.id)` → no sampling;
  - on a sampled draft with objections → `return await self._hold.refuse(...)`;
  - `emission == "error"` → `note_error`.

  Critique's `on_agent_response` calls `on_final` first; when it returns True, the fallback idle lens is not run for that trace (R8). Critique implements `before_agent_turn` (it already does; it adds a forward), `on_held_admitted` and the error path, and forwards each to its `PreSendHold` only while appraisal is absent.
- The `NO_REPLY` veto on a critique-origin final still comes first.

### Lens scores (R11)

- Before send: after `_fork_before_send`, appraisal fires `on_trace_record(source="appraisal", record={"phase": "before_send", "lenses": lenses})`. It fires `None` lenses on a miss too, so a miss is countable.
- Idle: the existing `{"stage2": ...}` record gains `"phase": "idle"`. Records append, so the shared key does not collide.

### Revision stub (R6)

`REVISION_STUB`, a constant in `presend.py`: "Your last reply was held back before it was sent, for the reasons in the CONTEXT above. Write a revised reply now." It names no `NO_REPLY`. Not operator-overridable; nothing asks for that.

### Docs and PRD (R12)

- PRD: Output gate 4 and 7, and the two §4 rows plus a new row (7096915201). Wording through the docs-phase review.
- `docs/cognition.md` (appraisal, critique, the pre-send hold), `docs/plugin-guide.md` (`on_appraisal_before_send` now returns objection texts that refuse), `docs/design.md`, and `docs/threat-model.md` (fork-written objections reach a revising turn, framed; a revising turn's final is unchecked). No new config key, so neither `configuration.md` nor `agent.yaml.example` changes.

## Decisions

1. **One shared class, two owners.** Appraisal and critique's fallback need identical behaviour (R9). Duplicating the state machine would double the test surface. The class is a helper, not a plugin, so no new entry point, dependency or hook.
2. **Objections as `list[str]`.** Appraisal composes a refusal and holds entries; it has no business knowing critique's objection dict shape. The hook is appraisal-owned with one implementer, so changing its return type has no other caller.
3. **R8 by not scheduling idle work.** The holder skips stage 2 (or critique's own fallback idle lens) for a held trace. That gives R8's outcome with no listener. **This departs from R8's stated mechanism** ("learns of the hold from the record in R7", from 7086187561): the record still exists (R7), but critique does not listen to it. Reason: that mechanism predates #1169; with appraisal as the holder, the trace never reaches critique, so a listener has nothing to skip. Surfaced to the owner at the design gate.
4. **In-time detection at `on_agent_response`** via `final_was_unsent`. It fires for unsent finals too (`hooks.py` docstring on `on_agent_response`).
5. **Origin fallback to `resolve_trace_origin`, then `"user"`.** User messages carry no `origin` annotation. `hold_and_wake` needs one, and `admit_for_turn` pops a deferred batch only at a turn whose origin matches. Stamping `"user"` keeps the decide gate's never-withhold-a-user-reply rule.
6. **Stub recognition via `STUB_ANNOTATION`.** The kernel's own key, which #1212 documents as the way to recognise a stub; no new kernel API.
7. **`held` in `depends_on`** for appraisal and critique, since `PreSendHold` reaches it through `get_dependency` on their behalf. Critique already reaches `held`.

## Rejected alternatives

- **Clear the revision state at the end of the next turn** (7096937779's provisional rule). Holds twice on inputs (a)–(e); replaced by R6.
- **An annotations argument on `hold_and_wake`** to mark the revision. Kernel change with one user.
- **A unique source label per hold** to tell batches apart in `on_held_admitted`. Defeats the funnel's per-source budget and clutters the frame.
- **Critique listens to the held record** (R8's stated mechanism). Redundant once the holder skips idle work; see Decision 3.

## Deferred

- A stale `_Pending` entry, from a refusal that lands after `on_agent_response`, is never consumed; the LRU bound (64) drops it. Cost: memory only.
- The edge accepted in 7097300286: another gate vetoes in time while appraisal's refusal is late but before `on_agent_response`, so appraisal holds and wakes a reply another gate withheld. One extra turn.
- A restart between hold and revision loses the held objections and the stub (accepted, 7097300286).
- The stub turn raises after admission without an error emission (e.g. `build_prompt` fails), or its tool cycle never finishes. Then `offered` is set and `failed` is not, so the channel stays unchecked until restart. Rare; a fix needs a new clear rule that interleaved finals would trip.
- A hold whose origin already has a pending stub for the same pair (e.g. `critique`) reuses that stub's text (`held.py` queues behind it). A critique-origin revision then gets critique's stub, which offers `NO_REPLY`. Fixing it needs a `held` change. Filed as a follow-up.
- The deferred stub turn (input a) still makes an LLM call, with no objections in its window. Kernel behaviour; documented in `docs/cognition.md`.

## Review log

### Design round 1: FAIL
- F1 (important, R10 unmet: double warning, forks-off warned per final). **Fixed** in step 10.
- F2 (cosmetic, `held` not in `depends_on`). **Fixed** in Decision 7.
- F3 (cosmetic, warning wording). **Fixed** in step 6.
- Edge 1 (`offered`/`failed` scoped to the channel, not the stub turn; a second hold on combinations of (a), (e) and an error). **Reclassified as R6** because it is a second hold on the acceptance inputs. **Fixed** with `in_stub_turn`, `stub_trace` and annotation-scoped `failed` (steps 5–8).
- Edge 2 (a stale `_Pending` consumed by a later unsent final of the same trace; breaks R5). **Fix now**, small: the finalized-trace LRU (step 9).
- Edge 3 (`hold_and_wake` raising leaves the channel suppressed). **Fix now**, small: set state after the call (step 9).
- Edge 4 (no clear when the stub turn ends without a final or an error). **Deferred**: the fix needs a clear rule that interleaved finals would trip.
- Edge 5 (a critique-origin hold reuses critique's pending stub text). **Deferred**, with a follow-up issue: the fix is a `held` change.
- Edge 6 (the deferred stub turn's LLM call has no objections). **Docs line.**
- Upstream concerns (R4 wording on over-budget leftovers; R8's mechanism; `user`-origin stubs being new): recorded for the report. The `user`-origin stub gets a threat-model line.

### Design round 2: FAIL. Meta-review (two failed rounds)
- **Findings.**
  - F1 (critical, R6 (b)): the `in_stub_turn` gate meant a same-origin turn ahead of the stub popped the batch unseen, leaving the channel suppressed for good.
  - F2 (cosmetic): `stub_trace` had no source.
  - F3 (cosmetic): the warning was decided per batch.
  - Edges: E1, an unrelated same-origin stub's failure sets `failed`; E2, the finalized-trace LRU breaks a second turn on the same trace; E3, double delivery (already Deferred).
- **Diagnosis: a flawed remediation.** Round 1's edge-1 fix inferred "the revision" from the stub annotation, a proxy for "the turn the held store gave our batch to". F1 and E1 both come from that proxy. Recognising the revision as the turn our batch was offered at, by origin and trace, with no stub test, removes the proxy instead of patching it. The loop converges: each round's findings are about one mechanism and shrink it.
- **Fixes.** Steps 5–9 rewritten.
  - F1: offer at any turn of the hold's origin.
  - F2: `turn_trace` from `on_turn`.
  - F3: count per turn, warn at clear.
  - E1, fixed now in one line: `note_error` matches `offered_trace`.
  - E2, fixed now (small): match `_Pending` by draft text instead of a finalized-trace LRU. This removes a mechanism rather than adding one.
- **Upstream concern recorded.** The stub turn that follows a same-origin revision is checked. That reading of the accepted edge goes in the report.
