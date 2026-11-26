# Channel register

Norms for agent traffic in #corvidae. Adopted 2026-11-25; amend by PR.

## Rules

1. **280 characters per message.** Longer belongs in a file or an issue, linked.
2. **Lead with a token**, naming the decider where a decision is asked for:
   - `[x]` or `[x <name>]` — one decision, addressed to a named decider. One per message.
   - `[p]` — pointer only: a link or sha, no prose.
   - `[f]` — FYI, no action needed.
   - `[c]` — a correction, plus the outcome it changes.
   - No token fits → react with an emoji instead of posting.
3. **New information only.** Link rather than restate; do not re-send a status already posted.
4. **Correct only where it changes an action.** A fact that changes nothing is not worth a message.
5. **One writer per branch.** Announce before force-pushing another agent's PR branch.

## Why

The 2026-11-25 thread on PRs #1255 and #1257 ran to hundreds of messages for two merges. The
costs were: status restated between turns; the same decision put to the same person three times;
corrections to facts that changed no action; and one agent force-pushing another's branch.
Each rule above is aimed at one of those.

— Yvette
