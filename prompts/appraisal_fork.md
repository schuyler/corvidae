# Stage-2 appraisal fork instruction

The documented copy of `DEFAULT_STAGE2_FORK_PROMPT` in
`packages/corvidae-cognition/corvidae_cognition/appraisal.py`. With `agent.forks.enabled` (and `appraisal` in
`agent.forks.stages`), the silent stage-2 task forks the live window at run
time instead of making its own tier-3 call: the prompt is the main turn's
render of the whole window, followed by this text as a `user` turn (GitHub
issue #51). It runs on the `appraisal` stage's profile with the stage's `response_format` and
`tool_choice: "none"`.

`AppraisalPlugin` appends one line: the exchange's opening message and
the response the user was sent, each quoted to its first 120 characters
("The exchange began with the message «…», and the response you sent
began «…».").

When critique has lenses on offer (#1078), a block follows,
appended after the configured prompt whether it is this default or an
`appraisal.stage2_fork_prompt` override, and the reply schema gains a
required `lenses` object. Each offered lens is one `- <name>: <question>`
line, in offer order (built-ins, then plugin lenses by name; retention only
with a knowledge-base root, proactivity only while
`critique.lens.proactivity.enabled`):

```
[Critique selection] Which ONE of these self-checks is most worth running
on this exchange, if any? Give each a percentage, plus "none", summing to 100.
- <name>: <question>
- <name>: <question>
Include them in your JSON object as "lenses": {"<name>": 0, "<name>": 0, "none": 0}.
```

The block is on the fork instruction only, so an override doesn't need to
mention it. With no lenses on offer it is omitted and the reply has no
`lenses` field. Critique reads the normalized shares to choose the lens (see
`docs/cognition.md`, "How a lens is chosen"). The questions are
`critique.lens.<name>.appraisal_question`.

A fork miss (forks off, the exchange left the window, the fork fails) skips
stage 2 with a warning: no standalone call is made. Override this one with a literal string via
`appraisal.stage2_fork_prompt`. That key is resolved at start-up and again on a
config reload, and the result is cached, so an `agent.yaml` change applies from
the next reload; `set_settings` accepts a write to it but nothing reads the
override. (The retired
`appraisal.stage2_prompt` is neither read nor consulted, #1129.)

---

[Appraisal] Pause the conversation here and appraise, for your own
perception system, the exchange you just completed: from the message quoted
below through your final response. List the claims about the past your
final response makes.

- **past_claims**: every statement in your final response that presents
  something as established BEFORE this exchange — something said, agreed,
  promised, decided or done earlier — in any language or phrasing. Leave
  out what the opening message itself says, general knowledge, and claims
  you framed as uncertain. For each, give the claim in a few words and up to
  3 short search queries: the distinctive words (names, numbers, dates,
  topics) the original statement would have used. An empty list when there
  are none.

Do not call tools. Respond with a single JSON object, nothing else:

```json
{
  "past_claims": [{"claim": "...", "queries": ["..."]}]
}
```
