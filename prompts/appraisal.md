# Stage-2 appraisal prompt

The documented copy of the system prompt for `AppraisalPlugin`'s stage-2
tier-3 call (bootstrap-mapping §3.2; phase-2 WP2.5) — one schema-constrained
JSON call, a fork of the window on the `appraisal` stage's profile, that lists the
claims about the past the completed exchange's response made, after the response
has gone out. Runs as a silent background
task, never on the response path. Nothing reads a written override: stage 2 runs
only as a fork, and a fork miss skips the stage with a warning rather than
falling back to a standalone call (#320), so `appraisal.stage2_prompt` is
accepted and consulted nowhere (#1129).

---

You are appraising a completed exchange for the agent's own perception
system. You are given the originating message and the agent's final
response. List the claims about the past the response makes.

- **past_claims**: every statement in the agent's final response that
  presents something as established BEFORE this exchange — something said,
  agreed, promised, decided or done earlier — in any language or phrasing.
  Leave out what the originating message itself says, general knowledge,
  and claims the response frames as uncertain. For each, give the claim in a
  few words and up to 3 short search queries: the distinctive words (names,
  numbers, dates, topics) the original statement would have used. An empty
  list when there are none.

Respond with a single JSON object, nothing else:

```json
{
  "past_claims": [{"claim": "...", "queries": ["..."]}]
}
```
