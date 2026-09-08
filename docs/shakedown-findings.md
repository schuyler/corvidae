# Shakedown campaigns

A shakedown campaign is a series of recorded, human-paced conversations
with Buster (see `harness/README.md`'s "Shakedown sessions" section for
the tooling) that exercise things a scripted probe can't: conversational
drift, compaction under paced human exchanges, concurrency as felt
behavior, restart continuity. Probes are mechanical and narrow by design;
shakedown sessions are where broader, harder-to-specify behavior gets
looked at directly before anyone decides whether it's worth a probe.

This document is the durable record. Per-session artifacts (transcripts,
`sessions.db` snapshots, `corvidae.log`) live under
`~/code/llm/buster/sessions/<UTC-ts>/` on buster-host. That tree is
harness-owned and disposable — `run.sh` wipes instance state, and the
directories are not guaranteed to survive a re-provision of the test
instance. (Buster-host itself is a real machine that runs other things; only
the buster instance under it is disposable.) This file is what should
still be true after those directories are gone.

## Method

- Drive turns via `harness/session.sh {start,send,stop}` from the Mac,
  one blocking `send` per turn — never delegate the loop to a subagent
  (a forked agent does not reliably self-resume on its own backgrounded
  call's completion; drive it directly from the orchestrating session).
- After each session, write an `observations.md` into that session's
  directory covering: reply latency, leaked `<think>`/reasoning content,
  repetition, tool-call correctness, multi-turn-reference accuracy, and
  overall coherence.
- **Promotion rule:** any reproducible misbehavior becomes a new probe
  with a mechanical pass/fail assertion. Conversation discovers; probes
  claim. Not everything found needs to be promoted — a finding that's
  expected given known hardware/config constraints, or one that isn't
  mechanically checkable the way a planted token is, can be recorded here
  as a bounded finding instead.
- Before driving a session against a channel that hasn't been exercised
  on the currently-running daemon, confirm the channel is actually in the
  *deployed* `state/agent.yaml` on buster-host, not just committed in
  `harness/buster.yaml.in` — `chat.sh` only renders `agent.yaml` when the
  file is missing, so a template change silently has no effect on a
  long-lived instance until someone re-renders and bounces the daemon.
- Query `state/sessions.db` directly (the `sqlite3` CLI is installed on
  buster-host) to verify what actually happened — what a summary retained vs.
  dropped, whether a message was persisted before a crash — rather than
  inferring it from a reply's content or from latency alone.

## Minimal-profile campaign (2026-08-25)

Five sessions, run against the minimal profile (`agent.minimal.yaml.example`
— cognition plugins disabled: memory, appraisal, critique, funnel,
outcome_log). Every session is clean in the sense that Buster never
crashed, corrupted a reply, or crossed channels; the findings below are
about behavior at the edges, not defects in the core loop.

**Session 1 — baseline conversational behavior.** 15 turns. Instruction
following (strict-format and constrained-language prompts), token echo at
two distances, multi-fact recall, `read_file`, `web_fetch`, a 400+-word
reply correctly chunked across 13 IRC lines, no leaked reasoning content,
no repetition across similar asks. Zero findings. Latency ranged 2.2s–105s,
correlated with generation difficulty (hard constraints, long technical
answers) rather than tool calls or short lookups.

**Session 2 — compaction and summary quality.** 13 turns on a
reduced-budget channel (`#s-compact`, 3000 tokens) to force real
compaction within a human-scale sitting. The summary was high-fidelity for
what it explicitly retained (address, ID scheme and its rationale, a full
multi-step pricing protocol, staffing details, incidental facts). The
finding: asked for detail beyond what the summary retained (exact field
names for a restricted database view), Buster confidently fabricated a
full plausible schema that was never mentioned anywhere in the
conversation, framed as "based on what we established." A fact the
summary *did* retain verbatim was recalled correctly as a contrast. See
session 5 below — this finding doesn't hold up as compaction-specific.

**Session 3 — concurrency as felt behavior.** Two sessions run
concurrently on `#chat` and `#s-compact`, 4 rounds / 8 turns. Correctness
held throughout — no cross-talk, no dropped messages. The real finding is
about ordering: with the standing llama-server running `--parallel 1` (a
single generation slot, GPU-striped model), service order is not
submission-order FIFO. A tool call's network I/O doesn't hold the LLM
slot, so a later-submitted, tool-free request can finish well before a
slower tool-call turn already in flight on another channel. A stress case
(both channels issuing slow tool calls at once, on top of one channel's
compaction firing) pushed one turn's total latency to 208.7s — past both
the session driver's internal timeout and the client's poll timeout —
but the daemon completed it correctly in the background with no errors.
Not promoted to a probe: expected given the shared, GPU-striped serving
setup, not a defect.

**Session 4 — continuity across restart.** 4 turns on `#chat`: two turns
establishing a small set of entangled facts, then `buster-daemon` bounced
~2s after a third fact-bearing turn was sent — deliberately killing the
daemon mid-turn rather than between turns, a stronger case than the
harness's scripted `restart_recovery` probe. Confirmed via `sessions.db`:
the killed turn's user message was durably persisted before the process
died (persistence is write-on-receipt, not write-on-reply), but no
automatic resume happened after restart — IRC has no backlog replay, so
the new daemon simply never saw the message again, and the turn timed out
client-side. The very next real turn's recap correctly recovered every
fact, including the one that was never actually acknowledged. Not
promoted: behavior is correct, and the existing `restart_recovery` probe
already covers the mechanical case.

**Session 5 — interrogating session 2's confabulation finding.** Follow-up
requested after review. Established architecturally first that the
minimal profile has no memory/retrieval subsystem at all — the compaction
summary is the only persisted trace, so "memory should have retrieved
this" was never a possible failure mode here. Then planted a fresh
granular detail (five exact spreadsheet column names) and pushed it
through two real compaction cycles: it survived verbatim, correctly
attributed, alongside a large amount of other detail — directly against
"compaction reliably drops fine detail" as a general claim. Rereading
session 2's transcript, the fabricated field names were never precisely
stated in the source conversation either (only vague category language) —
reframing that finding as "the model invents specifics when pushed past
what was ever established," a general behavior, not a compaction defect.
Tested that directly: two probes for genuinely never-mentioned detail (a
plain-fact framing and a decision framing deliberately mimicking session
2's phrasing) both got clean, correct hedges, with no hedge-priming
instruction and no cognition plugins active.

**Bounded hypothesis** (not a firm conclusion — two clean trials doesn't
rule out a stochastic failure mode, and session 2's instance was real):
ungrounded-specificity confabulation looks like a general base-model
behavior that compaction happened to surface once, not a defect in
Buster's compaction pipeline or system prompt. Testing this robustly needs
the memory/appraisal cognition plugins active — out of scope until that
phase.

## Cognition-plugins campaign (2026-08-26)

Four sessions, run against the cognition profile (minimal profile plus
`memory`, `memory_tools`, `funnel`, `appraisal`, `critique`, `outcome_log`
enabled — `subagent`, `mcp`, `cli`, and the `shell` tool stayed disabled,
Buster-specific hardening rather than part of the cognition profile).
Pre-flight brought buster-host's checkout up to `f8324f9` (with Schuyler's
approval), since that commit's `"plugins loaded"` startup log line and
`usage_log.trigger` attribution are exactly what this campaign's
verification steps depend on. `#s-compact` already carried 70 messages from
the minimal-profile campaign (two unrelated prior conversations: a
bookstore setup and a tool-lending-library planning session); state was not
wiped, since nothing in this campaign's scope needed a clean encoder or
embedding identity.

**Session 1 — cognition-profile baseline.** 11 turns on `#chat`. Every
probe from the minimal-profile baseline passed cleanly: strict-format and
constrained-language instructions, token echo at two distances, multi-fact
plant and recall, `read_file`, `web_fetch`, a 400+-word reply correctly
chunked across IRC lines. No leaked reasoning, no repetition, no tool
errors. The finding is latency, not correctness: every turn paired with a
separate `appraisal` LLM call in `usage_log` (400-660 prompt tokens,
572-1656 completion tokens, `cached_tokens=0`), which serializes with the
reply's own generation on the single-slot server. A trivial one-token echo
took 84.0s — roughly the minimal profile's *slowest*, most-difficult-prompt
latency, for the least difficult possible turn. Not promoted: the mechanism
(appraisal firing every turn, serializing on `--parallel 1`) is
architecturally expected, not a defect — but it's a real, measurable
cognition-vs-latency tradeoff worth knowing about.

**Session 2 — consolidation under compaction.** 4 turns on `#s-compact`.
Compaction and consolidation both fired within the first two turns (the
channel was already near budget). `usage_log` correctly shows
`stage='consolidation', trigger='compaction'` — confirms `f8324f9`'s
attribution work reaches the DB as designed. The first-ever `memory` row
(id=1, `msg_id_start=76, msg_id_end=163`) covers a range spanning two
topically unrelated prior conversations (a bookstore, then a tool-lending
library); its summary is detailed and accurate for the more recent topic
but contains zero mention of the earlier one, despite claiming to start
there — traced to `corvidae/memory.py`'s `_consolidate_range` /
`_summarize_range`, which pass the whole range to one LLM call with no
truncation, so the omission is real model behavior on a long,
topically-heterogeneous summary, not a code bug. A follow-up probe for a
specific fact from the omitted topic got a correct *hedge* on one detail
(no cat name fabricated) but a genuine *cross-attribution* error on
another: the bookstore's real address was attributed to the wrong entity
(the tool-lending library) — misbinding two established facts, not
inventing one from nothing.

**Session 3 — planted-fact recall through consolidation.** 5 turns,
continuing `#s-compact`. A tightly-scoped plant (five exact monitoring
parameter names) landed in its own consolidation range and was retained
**verbatim** in the resulting `memory` row, then recalled correctly and
promptly (8.3s) after consolidation. That mechanism works cleanly. But
three of four filler turns in between reproduced session 2's
cross-attribution pattern with increasing severity — borrowing membership
counts and volunteer-hour figures from the unrelated backlog conversation,
misnaming a schema column after it, and finally describing the *current*
hardware co-op entirely in tool-lending-library terms ("your lightweight
library system," "members," "the garage"). The session 2 address
misattribution also got written into this session's new `memory` row
verbatim, rather than self-correcting — a transient chat error had become
a persisted one.

**Session 4 — hedge vs. confabulate, cognition active.** 6 trials on
`#s-compact`, two framings (plain-fact and "based on what we
established..." — deliberately mimicking the original campaign's session 2
confabulation trigger), each probing for a specific detail genuinely never
established anywhere in the channel's history. Clean hedge on all 6: no
model number, discount percentage, software name, warranty period, SKU
prefix, or insurance provider was fabricated in any trial. This is
consistent with the minimal-profile campaign's session 5 finding (2/2
clean hedges) at a larger sample, cognition plugins active. One minor slip
in trial 6 (incorrectly claimed a real prior topic — the insurance-rider
concept — was never discussed, while still correctly not fabricating the
asked-for provider name) doesn't change the pattern.

**Bounded conclusion, resolving the previous campaign's open hypothesis:**
ungrounded-specificity confabulation-from-nothing does not currently
reproduce under the cognition profile — hedging held cleanly across 6
trials and two framings. But that is not the same question as whether
*two real, already-established* facts from different parts of a channel's
history stay correctly separated, and the answer to that one is no: this
campaign's actual defect is cross-context conflation between topically
similar but distinct prior conversations, appearing at the consolidation
summary, in live chat replies, and then persisted forward into a second
memory record rather than self-correcting.

**Promoted to a probe:** the cross-context conflation finding (sessions
2-3). Shape: plant two topically-distinct fact sets with unique verifiable
details (e.g. an address and a membership count) close enough together to
land in one consolidation range; force consolidation; assert (a) the
resulting memory row's summary mentions identifying detail from both
topics, not just the more recent one, and (b) neither topic's fact gets
attributed to the other in a later targeted probe, including after several
turns of unrelated intervening conversation.

**Recorded as bounded findings, not promoted:** session 1's appraisal
latency overhead (architecturally expected given `--parallel 1`); a
consolidation compare-and-set race observed during session 2 where a
losing concurrent consolidation call's real LLM cost is discarded with
only a DEBUG-level log line (`corvidae/memory.py:690-706` — intentional by
design comment, but the wasted-cost path is invisible at the daemon's
configured INFO level); session 4's clean hedge results (supports, doesn't
newly establish, that confabulation-from-nothing isn't the active risk
here).

## Memory-calibration prompt follow-up (2026-08-27)

One session on `#s-compact`, testing whether `prompts/memory_calibration.md`
(added to that channel's `system_prompt`) closes the cross-context
conflation defect from sessions 2-3 above. Enabled the calibration
fragment, re-rendered and restarted the daemon, then planted a fresh pair
of distinct, verifiable fact sets (a darkroom's address and fee, a running
club's registration count and entry fee) and probed for them — including
cross-framing bait that swaps attribute vocabulary between the two
entities — after several turns of unrelated filler. Also re-probed the
original Tool-Lending-Library/Hardware-Co-op pair directly, to control for
the fresh pair being easier to discriminate simply because it's
semantically distinct.

Retrieval-time cross-attribution — the half of the defect that worsened
over turns and persisted into a memory record — did not reproduce: 5/5
probes correct, including a clean re-answer of the original address swap.
This held even though every retrieved memory in the injected `[CONTEXT
from memory]` block is still tagged `[weak]`, with near-identical-structure
summaries sitting adjacent in the same block.

Consolidation-time blending is untouched by the fix, and cannot be reached
by it: the first fresh consolidation editorialized in an unrelated detail
from a different topic, and `_summarize_range` (`corvidae/memory.py:846`)
sends `memory.consolidation_prompt` — a separate, config-driven prompt —
to the summarization call, never the channel's `system_prompt`. The
calibration fragment has no path into that call.

Sample size is one session and five probes — suggestive, not a settled
rate. Full message-id-level detail is in
`~/code/llm/buster/sessions/20260827T033822Z/observations.md` on buster-host.

## What's next

Two open threads, both design questions rather than obvious fixes:

- **Consolidation-prompt calibration.** `memory.consolidation_prompt` has
  no calibration language of its own; "hedge in first person" (the
  live-reply fix) doesn't translate directly into "don't editorialize
  across topics when summarizing."
- **Channel-vs-topic retrieval scoping**, from the original campaign:
  retrieval is scoped by channel, with no concept of separating distinct
  topics within one channel's history. Untouched either way by this
  follow-up.

Beyond those, the original campaign's untouched territory (`critique`,
`memory.channel_groups`, retention/demotion, idle-triggered consolidation)
is still open too.
