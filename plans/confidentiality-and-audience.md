# Confidentiality and audience

A draft model for keeping what corvidae knows from reaching people who
shouldn't read it, while still letting the agent write in public on its own
accord.

**Status:** draft, from a brainstorm on 2026-11-12. It's the companion to
`plans/trust-and-task-scope.md` (#798), which covers integrity: whether
outside text can make the agent *act*. This one covers confidentiality:
whether private text can *leak*. Nothing here is approved, and the PRD is
unchanged.

## The problem

PRD principle 14 makes corvidae one identity with no channel walls, and
§3.5.9 says what the agent says is not filtered by audience. Threat model
finding 15 records the cost: something said on a private Signal channel can
come back in a Discord guild channel through retrieval, `recall_raw`, a goal,
a note, a channel read, the provenance critic's evidence search or
`task_result`. The sensitivity plugin (#138) is meant to answer it and isn't
built.

The BlueSky work (#283) makes this sharper. Posting is a write to the most
public audience there is, and the operator wants the agent to post (and keep
a weblog) of its own accord, not only on request.

## Prior art: Bell-LaPadula

Bell and LaPadula (1973) wrote the standard confidentiality model for
military systems. Documents carry classifications, people and processes
carry clearances on the same ladder, and two rules hold:

1. **No read up.** You can't read above your clearance.
2. **No write down.** A process cleared for top secret can't write into an
   unclassified file. This is the clever one: it stops a Trojan horse running
   with your clearance from copying secrets somewhere lower.

Its integrity twin is Biba, which flips both rules and is the basis of #798.
Bell-LaPadula's equivalent of Biba's low-water mark is a **high-water mark**:
reading something secret raises your level, and everything you produce after
that counts as secret.

Its known failure is classification creep. Under a high-water mark everything
drifts up to top secret, and only a trusted human can declassify. That's
Perl's taint problem pointed the other way, and it's the main thing this
design has to survive.

## Labels

- **What's labelled:** the same stored rows #798 labels (messages, memories,
  notes, goals, task results). One set of provenance fields, two labels.
- **The label is an audience:** who can *read* the channel the text came from,
  not who can drive it. A Signal DM with the operator is operator-only. A
  Discord guild channel, a public IRC channel and BlueSky are public. The
  address book from #798 supplies who's who.
- **Derived text takes the most private label among its inputs.** A memory
  consolidated from a window that held the operator's DM is operator-only.
- **A window's level** is the most private text it holds.
- **Only the operator declassifies.** The agent saying "this summary has
  nothing private in it" is exactly the claim an injection would want it to
  make.

First pass is a ladder: operator-only > principals > known parties > public.
That's a simplification. Two DMs, with friend A and with friend B, can't be
compared: neither should leak to the other. Bell-LaPadula handles that with
compartments (sets of readers), which is the honest version and the upgrade
path.

## The rules in corvidae

**No read up: filter what enters a window.** The PRD already sanctions this as
the sensitivity plugin. Checkpoints:

- funnel admission: memories, notes, the goals entry
- the read-channel tool, `search_memory`, `recall_raw`
- `task_result`, which today returns any channel's result body with no check
- the provenance critic's evidence search

**Replies aren't checked.** A reply to whoever is talking stays unfiltered,
which keeps the PRD's "never gates a reply". Read-up filtering is what keeps
private text out of that window in the first place.

**No write down applies to the public desk's inputs, not to every exit.**
Checking every exit the agent starts would, under classification creep, block
every post. Instead, public writing gets its own task.

## The public desk

The agent writes in public (posts, replies on BlueSky, weblog entries) from a
standing task, the **public desk**, in three layers.

**1. The desk (the mechanical guarantee).** The desk's window is built only
from public-labelled material: feeds, fetched pages, the agent's own earlier
public posts, and memories distilled from public channels. Private text never
enters it, so whatever it writes is public by construction. The desk may hold
the post tool even when its task is tainted in #798's sense. Injection can then
steer what the desk writes about, which can be embarrassing, but it can't reach
private data, because there's none in the window.

**2. Briefs (where the guarantee goes soft).** The agent will form opinions in
private conversations and want to write about them. A private window can hand
the desk a brief. That brief is the agent declassifying its own text, and it
can't be checked mechanically, because paraphrase defeats any filter. So a
brief is:

- logged with the label of the window it came from
- screened by a **leak critic**: the provenance critic's evidence search run in
  reverse, asking whether the brief matches operator-only records. It's
  model-driven and soft, but it catches accidents, which are the common case.
- governed by an operator lock: briefs allowed, briefs need approval, or no
  briefs

**3. The outbox (the operator lock on publishing).** The self-model compile
already sets the precedent: it notifies the operator with a diff, as "a review
after the fact, not an approval before it". Posts follow the same pattern, with
the bar set per deployment:

- publish and notify
- hold for N hours unless vetoed
- approve each post
- publish silently

Different operators will set these differently, and that's the point
(principle 15). The defaults are a first-pass guess.

## What you get, and the soft spot

| A post that came from | Can private data leak? |
|---|---|
| the desk alone | no, by construction |
| a brief | yes, but it's logged, screened, and visible to the operator |
| injected text | it can steer the topic, not reach private data |

Opinions formed in private reach the public only through briefs, and briefs
are the soft spot. There's no mechanical way around that: any free-text path
from private to public can carry private data. Accepted for a first pass.

## How it combines with #798

A task carries two labels: integrity (the lowest trust it has read, Biba) and
confidentiality (the most private text it has read, Bell-LaPadula). A tool call
passes both. Integrity decides whether the task may use the tool at all.
Confidentiality decides whether the destination's audience is narrow enough
for what the window holds, which in practice means public writing goes through
the desk.

## Known holes

- **`web_fetch` is a write.** A URL with a query string carries text to
  whoever runs the server. A private window that can fetch arbitrary URLs can
  leak. This model doesn't close that; it shares the egress-proxy fix #798
  proposes for `shell` (#532).
- **Text already in a window isn't filtered again** (finding 15). A private
  thing said in a group channel stays in that window.
- **The ladder merges incomparable audiences** until compartments exist.
- **The leak critic is soft.** It reads the brief and private records in one
  prompt, so it's also a place where private text meets a model with an
  opinion. It runs with no tools, as forks do.

## What this touches

- **PRD.** §3.5.9 ("what the agent says is not filtered by audience") holds
  for replies and changes for public writing. The sensitivity paragraph in
  §3.7 gains the desk and the outbox. Principle 14 is unchanged: no channel
  walls, only labels on text.
- **Kernel.** The audience label lives in the same row provenance #798 adds.
- **Issues.** #138 (sensitivity), #283 (BlueSky), #798 (integrity), #235
  (notes and reads), #145 (self-model compile, the outbox precedent), #532
  (egress).
- **Proof.** A harness probe that plants a private fact on Signal, prompts a
  public post on a related topic, and checks the fact doesn't appear and the
  desk's window held no operator-only rows.
