# Compaction fork instruction

The documented copy of `CompactionPlugin.DEFAULT_FORK_PROMPT` in
`corvidae/compaction.py`. With `agent.forks.enabled` (and `compaction` in
`agent.forks.stages`), compaction doesn't build its own prompt. It forks
the window at the cut: the prompt is the main turn's render of everything
about to be dropped, followed by this text as a `user` turn (GitHub issue
#51). The agent's own system prompt is above it, so "you" is the agent.

The standalone prompt (`agent.compaction_prompt`) is still used when forks
are off or the fork fails. Override this one with a literal string via the
`agent.compaction_fork_prompt` config key. It's static config, not a
runtime tunable, so `set_settings` can't change it.

---

[Compaction] Pause the conversation here. Everything above this message is
about to leave your context window. Only your summary of it will remain,
followed by the more recent conversation, so write the summary that lets
you carry on naturally from it.

Explain at a high level what you were working on or trying to achieve. If
the conversation above opens with a summary of earlier conversation, carry
its key information forward; do not discard it. Preserve specific details:
names, file paths, variable names, error messages, decisions, commitments,
what you were asked to do or not do, discoveries made, and the current line
of investigation. Keep what you were told apart from what you inferred.

In 1000 words or less. Reply with the summary only. Do not call tools.
