# Implementation plans

Executable work packages derived from `plans/bootstrap-mapping.md` (the plan
of record, converged after six rounds of code-grounded adversarial review).
Each phase document is written to be implemented by a coding agent without
architectural judgment calls: exact signatures, DDL, ordered work packages,
red tests first, and explicit trap warnings.

| Order | Document | Scope | Effort |
|---|---|---|---|
| 1 | `phase-0.md` | Observability hooks, attribution, outcome-log schema, eval-harness foundations | M–L |
| 2 | `phase-1a.md` | MemoryPlugin core: schema, rowid threading, embeddings, consolidation, retrieval, admission funnel | L |
| 3 | `phase-1b.md` | Reconsolidation/demotion, redact, memory tools, fixture evals | M |

Phases 2+ are deliberately not planned at this level yet — per
`bootstrap-mapping.md` §6, their detail depends on what the Phase 1 evals
show. Plan them when Phase 1 is green.

## Shared conventions (apply to every phase)

- **Red/green TDD** (AGENTS.md): write the failing tests named in each work
  package before the implementation. All pytest runs use timeouts
  (`pytest-timeout` is a dev dependency; `asyncio_mode = "auto"`).
- **Never swallow exceptions** — log, pass through, or re-raise (AGENTS.md).
  "Fail-soft" in these documents always means: catch, `logger.warning(...,
  exc_info=True)`, continue.
- **Comment every block** with its purpose; idiomatic Python with type hints
  (AGENTS.md).
- **Package management with `uv`**; run tests as `uv run pytest`.
- **The append-only invariant**: `message_log` rows are never deleted and
  their ordering is never rewritten. The only sanctioned content mutation is
  the Phase 1b `redact` tombstone.
- **The KV-cache discipline** (`bootstrap-mapping.md` §2.2): the context
  window grows at the tail and truncates at the head. Never insert, remove,
  or mutate mid-window. `ContextWindow.remove_by_type` is legacy — do not
  add call sites.
- **Update `docs/`** before declaring a phase complete (AGENTS.md):
  `docs/design.md` for architecture changes, `docs/plugin-guide.md` for new
  hooks, `docs/configuration.md` for new config keys.
- When a work package says a hook or table is specified in
  `bootstrap-mapping.md` §N, that section is normative — if this document
  and the mapping disagree, the mapping wins and the discrepancy should be
  reported.
