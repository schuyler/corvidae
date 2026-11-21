# Critique fork instruction: retention

The documented copy of `DEFAULT_FORK_PROMPTS["retention"]` in
`packages/corvidae-cognition/corvidae_cognition/critique.py`. With `agent.forks.enabled` (and `critique` in
`agent.forks.stages`), the lens runs as a fork of the live window instead
of a separate critic call: the main turn's render of the whole window,
followed by this text as a `user` turn, and then — appended by the plugin,
not part of this text — the knowledge base root: `critique.retention.root`,
or markdown_recall's root when that key is unset (#742). It runs on the `critique` stage's profile
with the objections `response_format` and `tool_choice: "none"`.

Chosen like the other lenses (the appraisal fork's critique vector, or the
`critique.sample_below_rate` draw; its question is
`critique.lens.retention.appraisal_question`, #1078), and offered only when a
root is known (`critique.retention.root`, else markdown_recall's): without a knowledge base there is nothing to
check the exchange against and the lens never fires.

The lens flags; it does not file. Its objections name a file under the root and
the content to append, and the agent files them in the critique turn, after
reading the file and deciding whether the fact is really missing.

`critique.lens.retention.fork_prompt` overrides this text from static config, so
`set_settings` cannot soften the lens it is checked by.
