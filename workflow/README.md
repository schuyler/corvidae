# Semi-Deterministic Workflow Engine

A Python orchestrator that drives an LLM through a multi-stage workflow,
with deterministic control flow and focused LLM invocations at each stage.

This directory contains an in-repo prototype. It will likely be extracted
into its own package once the design stabilises.

## Why This Exists

Giving an LLM a complete multi-stage workflow in a single prompt fails in
practice. As context grows, the model skips stages, forgets instructions,
and drifts from the intended process. The workflow engine solves this by
keeping the LLM's job small and focused:

- **Deterministic code** controls the workflow — which stage runs, in what
  order, and when to loop back.
- **The LLM** does what it's good at — reasoning, writing, deciding — within
  a single stage at a time, with a focused prompt and limited tool set.

## Core Principles

### Stage isolation

Each LLM invocation starts fresh. No accumulated conversation history from
previous stages. The LLM receives only what it needs for the current stage:
a system prompt, the stage instruction, and concise context from prior
stages.

This prevents context drift and means each stage is predictable and
testable in isolation.

### Prefer deterministic code; the LLM explains reasoning

Where possible, move responsibility for any work to deterministic code
rather than the LLM. The LLM should only do things that require
judgment. If it's unclear whether a task needs the LLM, prefer
deterministic code and check with a human when output quality might
suffer.

Stage output has two channels:

1. **LLM summary** — decisions made, reasoning behind them, approaches
   tried and rejected. Only things that require *judgment*.
2. **Orchestrator-gathered facts** — file diffs, test results, branch
   names, GitHub state, and the inputs and outputs of tool calls made
   during the stage. Anything observable programmatically.

The LLM never redundantly reports what the orchestrator can verify
directly. This keeps summaries small and prevents inaccurate
second-hand reporting of facts.

### Linear backbone with escape hatches

Workflows proceed forward through a defined sequence of stages. Specific
stages may "escape" backward to an earlier point when conditions warrant
(e.g., implementation reveals scope ambiguity → back to issue commenting
to ask questions).

This is not a general state machine. Every stage has a default forward
path, and only explicitly defined escape targets are allowed.

### Tool scoping per stage

Each stage only receives the tools it needs. The implementation stage gets
shell and file tools; the PR stage gets GitHub API tools; the "comment on
issue" stage gets GitHub comment tools. This reduces the LLM's decision
space and prevents it from taking actions outside its current
responsibility.

### Three-layer stage exit

Within a stage, the LLM runs in a tool-calling loop (like Corvidae's
`run_agent_loop`). It may call tools multiple times — write files, run
shell commands, post comments — before signalling completion. The loop
ends when the LLM calls the `stage_complete` tool.

A stage completes when:

1. The LLM calls a structured `stage_complete(outcome, summary)` tool
2. The orchestrator independently verifies the stage's exit criteria
   (tests exist, comment was posted, files changed, etc.)
3. If configured, a human approves before proceeding

All three must pass. If exit criteria fail, the orchestrator tells the
LLM what's missing and lets it retry within the same stage invocation.

### Structured outcomes

The LLM signals completion with an explicit outcome:

- `proceed` — stage is done, move to the next one
- `need_clarification` — loop back to an earlier stage (escape hatch)
- `scope_changed` — the task is different than expected; may loop back
- `blocked` — something unexpected prevents progress; needs human input

The orchestrator maps outcomes to transitions (including escape targets)
using the workflow definition. The LLM never chooses the next stage
directly.

## Relationship to Corvidae

The workflow engine is a separate program that uses Corvidae as an LLM
invocation layer. It imports Corvidae's primitives directly:

- `run_agent_turn` — single LLM invocation with tool calling
- `LLMClient` — HTTP client for the LLM API
- `ToolRegistry` / `Tool` — tool definitions and schema generation

The engine reads `agent.yaml` for LLM configuration (model, base URL,
API key) so there's a single config file for both systems.

This "thin client" approach (Approach A) is the starting point. As
patterns emerge, the engine may evolve into a declarative framework
(Approach B) or a Corvidae plugin (Approach C).

## Example Workflow: GitHub Issue Implementation

The motivating workflow. Stages:

1. **Comment on issue** — LLM writes a comment describing its
   implementation plan. *Tools: GitHub comments API.*
2. **Add "in progress" label** — Mechanical. No LLM.
3. **Create branch and worktree** — Mostly mechanical. LLM may suggest
   a branch name. *Tools: git.*
4. **Gate: ready to implement?** — LLM decides whether it has enough
   information, or needs to go back and ask more questions in the issue.
   *Escape target: stage 1.*
5. **Write tests (red)** — LLM writes failing tests. *Tools: file write,
   shell.*
6. **Implement (green)** — LLM writes code to pass the tests. *Tools:
   file write, shell, file read.*
7. **Refactor** — LLM cleans up. *Tools: file write, shell, file read.*
8. **Gate: docs needed?** — LLM decides whether documentation updates
   would add value.
9. **Update docs** — LLM writes documentation. *Tools: file write.*
10. **Submit PR** — LLM writes PR description. Mechanical PR creation.
    *Tools: GitHub PR API.*
11. **Respond to PR feedback** — LLM reads review comments and makes
    changes. May loop. *Tools: file write, shell, GitHub API.*
    *Escape target: stage 7 (refactor) or stage 5 (rewrite tests).*
12. **Close issue** — Mechanical. Post a comment linking to the merged PR.

Human gates are configurable per stage. Likely candidates: before stage 5
(ready to implement?), before stage 10 (ready to submit PR?).

## Evolution Path

1. **Now** — Standalone Python script importing Corvidae primitives.
    One workflow (GitHub issue implementation), defined in code.
2. **Near future** — Extract reusable stage machinery: prompt building,
    tool scoping, exit verification, summary accumulation. Still one
    workflow, but the orchestration pattern is a clean abstraction.
3. **When needed** — Declarative workflow definitions (Python decorators
    or YAML) so new workflows can be defined without writing orchestration
    code. Multiple workflows (issue implementation, code review, bug
    triage, release preparation).
4. **If needed** — Corvidae plugin integration, so workflows can be
    triggered from any channel (IRC message, CLI command, webhook).

## Key Decisions for Future Agents

- **Never give the LLM the full workflow.** It sees one stage at a time.
  The orchestrator is the authority on what happens next.
- **Never let the LLM choose the next stage.** It signals an outcome;
  the orchestrator maps it to a transition.
- **Keep summaries about reasoning, not facts.** File diffs, test
  results, and GitHub state come from the orchestrator. The LLM explains
  *why*, not *what*.
- **Include rejected approaches in summaries.** Future stages and human
  reviewers benefit from knowing what was tried and abandoned.
- **Verify independently.** Don't trust the LLM's claim that it's done.
  Check exit criteria programmatically.
- **Prefer deterministic code.** If a task can be done without the LLM,
  do it in code. If it's unclear, prefer code and ask a human when output
  quality might suffer. The LLM is expensive and unreliable; Python is
  cheap and deterministic.
- **Capture tool I/O.** The orchestrator logs tool call inputs and
  outputs as facts. The LLM doesn't need to summarise what it did — the
  orchestrator already knows.
- **Start thin, evolve when patterns are clear.** Don't build a
  framework until you've built at least two workflows and can see what's
  reusable.

## Implementation Status

**Approach A** (standalone script) is implemented. The workflow engine
lives in `workflow/` and imports Corvidae primitives directly.

### What's built

- **Core engine**: Stage, StageRunner, WorkflowEngine with linear backbone
  and escape hatches
- **stage_complete tool**: Three-layer exit (LLM signal → exit criteria
  verification → human gate)
- **Tool scoping**: Each stage receives only the tools it needs
- **WorkflowContext**: Two-channel accumulation (summaries + facts)
- **GitHub issue workflow**: 10 of 12 stages implemented (stages 1–10;
  PR feedback and issue close deferred)

### Usage

```bash
# Run the issue implementation workflow
uv run python -m workflow implement --issue https://github.com/owner/repo/issues/42

# Specify a local repo path
uv run python -m workflow implement --issue https://github.com/owner/repo/issues/42 --repo-path /path/to/repo
```

### Prerequisites

- Python 3.13+
- `gh` CLI installed and authenticated
- `agent.yaml` in the working directory with `llm.main` config

### What's deferred

- **Stage 11** (respond to PR feedback): Requires waiting for external
  events. Will need a pause/resume mechanism.
- **Stage 12** (close issue): Requires waiting for PR merge.
- **Framework extraction** (Approach B/C): Wait until a second workflow
  exists to identify reusable patterns.
