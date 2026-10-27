# Retrospective: web transport session (#87, PR #88)

A blameless look at one long Claude Code session that took the web
transport from design revision 7 to a green PR, run under the rule-of-two
process skill. It's for a later session to reconcile against the skill's
wording: what the process did, where it helped, where it cost more than it
bought, and what might change. It isn't a record of who got what wrong.
The process produced the outcomes below, and the proposals are about the
process.

Delete this file once the skill wording has been reconciled against it.

## Starting point

- Requirements R1–R15, approved by the user before the session.
- `plans/web-transport.md` at revision 7, after six cold design reviews.
  Rev 7 folded in an unreviewed fix for a shutdown-bound problem.
- The instruction: one more cold design review, then red, green and docs
  phases, each behind a review gate. Stop and diagnose after two failed
  gates.

## What happened

Roughly in order:

1. **Design review 7** failed with two critical findings. A handler stuck
   in a gate wasn't cancelled at shutdown: aiohttp drops its own
   reference to the handler task once the connection is lost. And
   client-chosen send ids collided across tabs. Both were fixed in rev 8.
2. **Design review 8** failed. Shutdown still missed a handler whose
   client the backpressure path had dropped, and rev 8 had added a
   `client_id` field nothing consumed.
3. That was two failures, so the process called for a **meta-review**.
   Its diagnosis: one structural flaw, patched case by case. `_clients`
   was serving both as the fan-out set and as shutdown's only registry
   of connections. The user chose the structural fix: connections owned
   by their handler tasks (rev 9).
4. **Design review 9** found two line-level defects: `asyncio.wait` on an
   empty set raises, and a `finally` aborted the pre-`prepare` 403. The
   user then asked why the design process had been so hard, agreed it
   was specifying individual lines, and passed the design gate,
   directing that remaining detail be settled in code and tests.
5. **Red phase.** About 2,000 lines of tests (107 functions), written by
   working through the design's 13-part Tests section. Implementation
   started in parallel; all tests passed on the first green run.
6. **Red review** failed: a subscribe race in raw-socket tests, and four
   assertions that couldn't fail.
7. **Browser smoke test** (local Playwright) found two real bugs no
   review or unit test had: Chrome rejects `ws://[::1]` as a CSP source,
   and a fixture mistake hid that the remote-image check never ran.
   Fixing the race-prone tests then exposed a real hang. Aiohttp's close
   can "succeed" having only called `transport.close()`, which never
   finishes against a peer that stopped reading. That's also true of a
   heartbeat timeout. The abort rule was settled in code, with a test.
8. **Docs** were written in parallel with the green review.
9. The user asked how to stop the red phase being over-specified. The
   **gating bar was narrowed** mid-review to blocker classes: security
   hole, data loss, hang or crash, a wrong result the user sees.
10. The user asked why the diff was 4,000 lines. Half was tests. The user
    asked for a **trim**: 107 → 64 test functions. The author
    mutation-checked the result (13 deliberate breaks, all caught after
    restoring one test), and a review confirmed no requirement lost
    coverage.
11. **Green review** found a "lost send on a half-open socket" and rated
    it critical data loss. It was anchored in the premise "history
    favors duplicates over losses". Two rounds of client-side recovery
    each failed re-review. At the second failure the user was asked, and
    chose to **drop the feature** and document it as a limitation. The
    user later noted that it had never been a requirement.
12. PR #88 opened; CI green.

## What worked

- **Cold reviewers found real bugs.** Among them: the stuck handler at
  shutdown, the dropped-client gap, the empty-`wait` crash, the 403 abort,
  the subscribe race, and identity-matching gaps in the unsent-message
  tests.
- **The fail-twice meta-review** turned a string of patches into one
  structural fix (connection ownership), which held up through every
  later review.
- **Probing third-party behaviour** settled questions prose couldn't:
  aiohttp's close and heartbeat paths, apluggy's context propagation,
  Chrome's CSP grammar.
- **Mutation checks** were cheap and decisive. Breaking a line on purpose
  and seeing whether a test fails found two weak tests and proved the
  trimmed suite still guarded what mattered.
- **The browser smoke test** caught what neither unit tests nor reviewers
  did.
- **Parallel streams.** The frontend was delegated to a cheaper model
  while tests were reviewed, and docs were written while the code was
  reviewed. Partitioning the work by file kept the streams from
  colliding.

## What cost more than it bought

### Severity ignored consequence

The rubric fails a gate on any reachable path that leaves a requirement
unmet, with no weighting for how much the outcome matters. Design reviews
7–9 and the first red review kept failing on edge cases with trivial
outcomes. After the bar was narrowed to blocker classes, findings dropped
to ones that mattered.

### The design was specified at the level of code

By rev 9 the plan dictated the order of statements in a `finally` block.
At that depth the design is an untested program and each reviewer acts as
its test runner. Every fix added prose, which gave the next reviewer more
to find. The decisive facts (aiohttp's close semantics) came from probes,
not from reading the design.

### Red-phase tests came from the design, not the requirements

The tests were derived from the design's Tests section, so they inherited
its detail. The trim showed 64 functions covered every requirement that
107 had. The brief to the red reviewer also said "uncovered design
behaviour is important" and asked for a wrong implementation per test.
That invited exhaustive coverage. Narrowing the brief, not the tests,
was what changed the reviews' output.

### Fixes grew into features

Twice a fix added a capability no requirement asked for: `client_id` in
rev 8, and client-side send recovery in green. Each opened new surface
that review then had to cover. Recovery failed two re-reviews before
being dropped. Both times the trigger was a real finding, but the
response was to build more instead of asking whether the case was in
scope.

### A premise was stretched to cover a new case

The send-recovery finding cited "history favors duplicates over losses",
a premise about replaying history, to rate a send-path case as data
loss. Nobody checked that the premise covered it. The premises file is
meant to stop reviewers relitigating settled decisions, but here a
premise widened what a gate would fail on instead.

### Reviewer-suggested fixes went unreviewed

Two of the last failures came from implementing a reviewer's own fix
advice ("name the other conversation in the notice"). A reviewer's
suggestion is a design decision made without review. The process
requires the author to name the class of bug and list its instances
before editing, but that wasn't applied to suggested fixes.

### Re-review and token cost

Each re-review re-read the whole artifact, including a 750-line plan, and
the session ran about ten reviewer passes across seven reviewer agents. The whole-work rule did
catch things a diff-only re-review would have missed (review 8's
dropped-client gap was in unchanged text that the fix made relevant). So
the cost wasn't simply waste, but it was high for what the later rounds
found.

### Contributing factors outside the process

This task was unusually dependent on a library's runtime behaviour:
aiohttp shutdown, cancellation and buffering. Some of the difficulty
would have shown up under any process. A spike, meaning throwaway code
that tests the risky mechanism before committing to a design, would have
found the same facts earlier.

## Proposals to reconcile against the skill

Ranked by how directly this session's evidence supports them.

1. **Weight severity by consequence.** A finding fails a gate only if it
   is a blocker: a security hole, data loss, a hang or crash, or a wrong
   result the user sees. An important finding is an acceptance criterion
   with no test that could catch a failure, a blocker guard that can't
   fail against a plausible bug, or code nothing reaches. Everything
   else goes on a "follow-ups (not gating)" list.
2. **Derive red-phase tests from acceptance criteria and named blocker
   risks**, not from the design's test list. The author mutation-checks
   the blocker guards and hands the list to the reviewer, whose question
   becomes "does each guard fail when it should, and is any blocker
   unguarded?"
3. **Route new capabilities through requirements.** When a fix or a
   finding needs something no approved requirement covers, ask the user
   whether it's in scope before building. That applies when a reviewer
   anchors a finding in a premise rather than a numbered requirement
   too: check that the premise covers the case.
4. **Keep the design at the level of contracts.** It states requirements,
   invariants and risks. Where correctness depends on a library's runtime
   behaviour, spike that part in code before the design gate. Statement
   order and similar detail belong to the red and green phases.
5. **Treat a reviewer's suggested fix as a hint.** The author still
   names the class of bug and lists every instance before editing,
   whether or not the reviewer supplied a fix.
6. **Run the meta-review at the first sign of patching.** If a fix only
   addresses the case the reviewer named while other cases of the same
   kind are plausible, step back then instead of waiting for the second
   failure.
7. **Scope fix-round re-reviews** (lower confidence). Keep whole-work
   re-review for the design. For code fix rounds, review the diff plus
   the modules it touches. Weigh this against review 8's catch, which a
   diff-only review would have missed.

## Open questions for the reconciliation

- How should a skill brief state the blocker classes so they apply to
  any project, not just this transport's (auth, sockets)?
- Is "mutation-check the guards" reasonable to require when there is no
  quick way to break code, as in some languages or build setups?
- When the user passes a gate by decision, as with the design here,
  should the skill record that as a distinct outcome from a review pass?
