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
`~/code/llm/buster/sessions/<UTC-ts>/` on buster-host, which is a disposable,
harness-owned box — those directories are not guaranteed to survive a
re-provision. This file is what should still be true after they're gone.

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

## What's next

Structured cognition protocols — planted-fact recall, salience contrasts,
engagement calibration, and a robust hedge-vs-confabulate test — wait for
the cognition-plugins phase, since they score against appraisal/memory
tables the minimal profile disables. When that phase starts, a new
campaign should be planned the same way this one was: a short method
section, one session per behavior category, a promotion rule, and this
file kept current with the results.
