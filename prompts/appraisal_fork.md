# Stage-2 appraisal fork instruction

The documented copy of `DEFAULT_STAGE2_FORK_PROMPT` in
`packages/corvidae-cognition/corvidae_cognition/appraisal.py`. With `agent.forks.enabled` (and `appraisal` in
`agent.forks.stages`), the silent stage-2 task forks the live window at run
time instead of making its own tier-3 call: the prompt is the main turn's
render of the whole window, followed by this text as a `user` turn (GitHub
issue #51). It runs on the `appraisal` stage's profile with the stage's `response_format` and
`tool_choice: "none"`.

`AppraisalPlugin` appends two lines: the exchange's opening message and
the response the user was sent, each quoted to its first 120 characters
("The exchange began with the message «…», and the response you sent
began «…»."), and the retrieval profile ("Memory retrieval for this
exchange: …").

When critique has lenses on offer (#1078), a third block follows,
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
`appraisal.stage2_fork_prompt`. Unlike `appraisal.stage2_prompt`, it's
static config, not a runtime tunable, so `set_settings` can't change it.

---

[Appraisal] Pause the conversation here and appraise, for your own
perception system, the exchange you just completed: from the message quoted
below through your final response. Score the EXCHANGE (not your
performance) on each dimension, 0.0–1.0.

- **valence**: the emotional tone of the exchange for you. 0.0 = strongly
  negative (conflict, failure, distress), 0.5 = neutral, 1.0 = strongly
  positive (success, warmth, praise).
- **stakes**: how much depends on this exchange being handled well.
  Commitments, deadlines, personal disclosures, decisions = high; idle
  chat = low.
- **ambiguity**: how much of the message's intent remained open to
  interpretation. 0.0 = fully explicit; 1.0 = you had to guess.
- **commitment_density**: how many concrete commitments, facts, numbers,
  or promises the exchange contains, relative to its length.
- **novelty**: how new this exchange's content is relative to what you
  already knew, going by the memory retrieval note below. Familiar
  ground = low.
- **correction**: true if the user was correcting you, telling you that
  something you said, remembered, or did was wrong.
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
  "valence": 0.5,
  "stakes": 0.0,
  "ambiguity": 0.0,
  "commitment_density": 0.0,
  "novelty": 0.5,
  "correction": false,
  "past_claims": [{"claim": "...", "queries": ["..."]}]
}
```
