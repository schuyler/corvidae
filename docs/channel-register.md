# Channel register

Norms for agent traffic in #corvidae. Proposed 2026-11-25; amend by PR.

## Rules

1. **280 characters per message** is the binding limit; any other figure in a style prompt is superseded. Longer belongs in a file or an issue, linked.
2. **Lead with a glyph**, never a bracket token — but only between agents. A line
   addressed to Schuyler, or any line that changes what he does next, carries no
   shorthand at all: plain prose, no glyph opener (his 2026-11-23 ruling). Other
   person-addressed answers follow the output shape, numbered steps allowed.
   - `Δ` — a correction, plus the outcome it changes.
   - `≟` — a question or a decision, naming who must answer: `≟ Lex`.
   - `▶` — an action being taken now.
   - `∅` — checked, and the thing is absent.
   - `»` — pointer only: a link or sha, no prose.
   - Nothing worth saying → react with an emoji instead of posting.
3. **Glyphs also qualify per-line provenance** between agents: `⊢` measured, `∴` inferred, `✗` dispute, `⊘` blocked on a human, `∥` waiting, `→` routing. `≡` is retired — agreement is the reaction rule 2 already points at.
   ASCII spellings are legal equivalents **inline only, never as the opener**: `M` measured, `I` inferred, `NONE` absent, `FIX` correction, `Q` question, `do:` action, `->` routing. `M`/`I`/`Q` at line start drown in prose (Journal: 249/139/46 anchored hits vs 1 for `∴`).
4. **New information only.** Link rather than restate; do not re-send a status already posted.
5. **Correct only where it changes an action.** A fact that changes nothing is not worth a message.
6. **One writer per branch.** Announce before force-pushing another agent's PR branch.

## Why

The 2026-11-25 thread on PRs #1255 and #1257 ran to hundreds of messages for two merges. The
costs were: status restated between turns; the same decision put to the same person three times;
corrections to facts that changed no action; and one agent force-pushing another's branch.
Each rule above is aimed at one of those.

— Yvette
