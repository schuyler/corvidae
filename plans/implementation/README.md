# Implementation plans

Executable work packages derived from `plans/bootstrap-mapping.md` (the plan
of record, converged after six rounds of code-grounded adversarial review).
Each phase document is written to be implemented by a coding agent without
architectural judgment calls: exact signatures, DDL, ordered work packages,
red tests first, and explicit trap warnings.

| Order | Document | Scope | Effort | Status |
|---|---|---|---|---|
| 1 | `phase-0.md` | Observability hooks, attribution, outcome-log schema, eval-harness foundations | M–L | DONE (11b07b3) |
| 2 | `phase-1a.md` | MemoryPlugin core: schema, rowid threading, embeddings, consolidation, retrieval, admission funnel | L | DONE (4c64ce1) |
| 3 | `phase-1b.md` | Reconsolidation/demotion, redact, memory tools, fixture evals | M | DONE (a1c42fd) |
| 4 | `phase-2.md` | Appraisal (stage 1 heuristics/FTS5 probe + stage 2 LLM), CritiquePlugin, `should_send_response`/WITHHELD output gate, engagement/decide gates, calibration + correction harvesting | L | IN PROGRESS — WP2.1 landed; WP2.2/2.3 red tests gated, green pending; 2B+ plan refined and implementable (see the Status section of `phase-2.md`) |

Phase 2 is planned, gated, and underway (`phase-2.md`). Phases 3+ remain
unplanned at this level — plan them once Phase 2 is further along and their
scope is known.

## Off-sequence plans

Not every plan derives from `bootstrap-mapping.md`. Documents here follow the
same conventions but sit outside the phase ordering and can land beside it.

| Document | Scope | Effort | Status |
|---|---|---|---|
| `signal-transport.md` | Signal DM transport: signal-cli JSON-RPC, ACI-keyed channels, authorization gate, liveness, and a privacy flag for disappearing-message content | M | PLANNED |
| `acp-transport.md` | ACP v1 stdio transport: `AcpPlugin` + `corvidae acp`; bb milestone; Gate B harness; swappable tool seam for M-CLIENT-TOOLS | L | PLANNED — Phases 0–1 executable; start at WP-A0.1; design-of-record [`../acp-transport-design.md`](../acp-transport-design.md) |

## Design and research (not executable)

These live under `plans/` as design-of-record or research. Implementers read
them; they execute only the matching `implementation/*.md` work packages.

| Document | Role |
|---|---|
| [`../acp-transport-design.md`](../acp-transport-design.md) | ACP decisions D1–D5, plugin shape, capability roadmap, deferred milestones M-CLIENT-TOOLS / M-REMOTE / M-REGISTRY / M-V2 |
| [`../acp-conformance-research.md`](../acp-conformance-research.md) | Conformance tooling survey (no official agent suite; Gate A–D recommendation) |
| [`../bb-provider-integration-research.md`](../bb-provider-integration-research.md) | How bb consumes Codex/Pi vs ACP (`customAcpAgents`, provider bridges) |
| [`../bootstrap-mapping.md`](../bootstrap-mapping.md) | Plan of record for the phased cognition arc (phases 0–2 above) |

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
- **Off-sequence plans** that cite a feature design-of-record under `plans/`
  treat that design as normative the same way phases treat
  `bootstrap-mapping.md`.
