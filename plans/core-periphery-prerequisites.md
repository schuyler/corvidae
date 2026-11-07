<!-- Brief for the agent managing the prerequisites of the core/periphery
     rework (docs/prd-core-periphery-supplement.md). Written 2026-11-06;
     retire it when #364 lands and the TRD is drafted. -->

# Brief: core/periphery prerequisites

You are managing the kernel work that has to land before the core/periphery
technical reference document (TRD) can be drafted. You are not writing the
TRD, not designing the core/periphery strategy, and not writing phase or
roadmap documents. Each ticket below is a practical outcome with its own
acceptance tests; land them, in order, and stop.

## Read first

- `AGENTS.md` and `CLAUDE.md` (TDD, exception discipline, docs current
  before done, the PRD is the reference).
- `docs/prd.md` §3.8 (forks) and §3.13 (resource pools and leases).
- `docs/prd-core-periphery-supplement.md` §3.3.6–7 and §3.5 (the fork
  rule), and `docs/core-periphery-open-questions.md` gap 12. The
  supplement is a draft and binds nothing, but the fork rule in §3.5.7 is
  the one place it changes what you build.
- GitHub #296 (the lease design and its five comments), then #16, #362,
  #363, #364, #367.

## The work, in order

1. **#16: split `CoreToolsPlugin` and rename every kernel-as-"core" name.**
   The ticket's latest comment has the inventory and the notes. No
   compatibility alias for `core_tools`; the owner accepts the config
   break. Add the guard test so the names can't creep back.
2. **#362: declare pools and their warm heads.** Parse and expose only; no
   scheduling.
3. **#363: lease scopes, so every LLM call knows its queue.** Carry the key
   through the attribution ContextVar. Every built-in call site is covered
   by a test. Leave room for a fork's scope to nest inside its parent's:
   a blocking fork runs under its parent's lease (supplement §3.5.7).
4. **#364 with #367: the lease manager, drain-or-yield, FIFO, metrics.**
   Before building, verify the ticket's own open item: whether today's
   code enforces that an exchange's own background work finishes before
   its channel's next main turn, and where. The acceptance test is
   amended by the owner's comment on #364 (2026-11-06): the one-head
   case holds because critique *blocks by default*, a per-stage default
   the operator can change, not a kernel rule; the non-blocking case,
   where B's turn runs, A's critique misses, and the miss is logged, is
   added. Implement to the amended test.

Stop there. **#365 and #366 are on hold**: the supplement replaces their
semantics (overdue notification instead of yield at a latency bound;
pauses that do not lapse), and the TRD will re-specify them. Do not build
them as written. #368's harness scenario is the TRD's probe and waits too.

## Rules

- One PR per ticket, on a branch named for it, red tests first. Run
  `uv run pytest` and the three CI jobs' subsets before pushing (see
  `CLAUDE.md`). Lint and type checks are advisory; don't block on them.
- Post the requirements, the PRD comparison and the plan as a comment on
  each ticket before implementing, as `AGENTS.md` requires. Where a ticket
  and the PRD disagree, stop and ask the owner; don't pick a side.
- Docs are part of done: `docs/design.md`, `docs/configuration.md`,
  `docs/plugin-guide.md`, `agent.yaml.example` as each ticket touches them.
- Don't widen a ticket. If you find a bug on the way, file it and move on
  unless it blocks the ticket.
- Don't touch `docs/prd.md` or the supplement. If the work shows either
  is wrong, say so on the ticket and wait.

## When you are done

Report, per ticket: the PR, what the tests prove, what you verified by
hand, what you did not verify, and anything the TRD should know (a
primitive that turned out to be missing, a measurement worth taking, a
place where the supplement's wording didn't survive contact with the
code). That report is the input to taking stock before the TRD.
