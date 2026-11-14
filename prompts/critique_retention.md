# Critique prompt — retention lens

The documented copy of `DEFAULT_PROMPTS["retention"]` in
`packages/corvidae-cognition/corvidae_cognition/critique.py`, and the shape a
`critique.prompt.retention` override should keep: the lens asks whether the
exchange carried a durable fact, belief or behaviour, whether the knowledge
base already holds it, and — only when it does not — which file it belongs in
and the exact content to append.

With `agent.forks.enabled` the lens runs as a fork and uses
[critique_retention_fork.md](critique_retention_fork.md) instead; this text
describes the same check for a deployment with no fork for the stage.

Output schema, the same as every other lens: a single JSON object with an
`objections` list, each entry carrying `claim`, `objection`, `severity` and —
for this lens only — `file` and `content`.
