# Trust and task scope

A draft model for deciding whether a piece of text can make corvidae act,
based on who wrote it rather than which pipe it arrived through.

**Status:** draft. It came out of a brainstorm on 2026-11-12, prompted by
the RSS (#778) and BlueSky (#283) work. Nothing here is approved, the PRD
is unchanged, and the wording will move. It's here so it can be revised in
review rather than in chat scrollback.

## The problem

Corvidae decides "is this an instruction?" in about six places, each with
its own rule:

1. Transport allowlists decide who can start a turn at all. Binary.
2. The kernel header and fence (#548) treat a person's message as the
   instruction channel and fence everything else as data.
3. Funnel CONTEXT says "data, not instructions" but arrives in the `system`
   role (threat model finding 3).
4. Tools carry one bit, `returns_untrusted`.
5. Graded trust (#138) and a per-call tool veto (#148) are designed but not
   built.
6. Per-channel `tools.disabled` (#149) is the only hard control that
   exists.

They share an assumption: passing the allowlist means you get to give
instructions. Feeds break it. Everyone who writes a feed item is a stranger,
and a feed-woken turn today would read that text with `shell`,
`web_fetch` and cross-channel send in reach and nobody watching.

Text also outlives its origin. A stranger's words can become a reminder, a
goal, a memory or a compaction summary and come back later with no record
of who wrote them (findings 7, 10, 11).

## Prior art

- **Perl's taint mode.** Outside data is tainted, taint propagates through
  expressions, dangerous sinks refuse tainted arguments, and a regex capture
  untaints. It mostly went unused for two reasons worth learning from:
  untainting was too cheap (`/(.*)/`), and it was coarse enough that people
  turned it off.
- **The Biba integrity model** (1977). Its *low-water-mark* policy, where a
  subject's integrity drops to the lowest level it has read, is the model
  below almost exactly. Its *no write up* rule closes the filesystem
  laundering hole.
- **Willison's dual-LLM pattern and DeepMind's CaMeL.** Both track taint per
  value by keeping the privileged model away from untrusted text. Finer
  grained than what's proposed here, at the cost of the agent not being able
  to read what it's reacting to. Not a fit for an agent whose job is reading
  and reacting, but the "narrow the type" trick (below) borrows from them.

The important difference from Perl: there, the program is trusted and only
values are tainted. Here the model *reads* the tainted text, and that text
can steer it, so taint attaches to the whole task, not to individual values.

## Definitions

**Identity.** A person, linked across transports by identifiers the
transport can verify, in an **address book** the operator owns. A YAML file
is fine to start. Each entry carries a trust level. An identifier the
transport can't verify, or one the address book doesn't list, is a
**stranger**. Trust is capped by how strongly the transport proves who's
talking: an unauthenticated IRC nick can't claim to be the operator, whatever
it's called (this is already #138's rule).

**Levels.** Operator > principal > known party > stranger. The middle rungs
are tentative.

**Source.** Anything the address book can vouch for besides a person: a
directory in the agent's workspace, a repository, a feed. "This directory is
principal-level" is a legitimate entry. A tool result's author is its source;
a source the address book doesn't list is a stranger.

**Task.** The unit of authorization. A task records who authorized it, the
act that authorized it (a message, a GitHub label, an operator config job),
the goal, and its current scope. Every turn belongs to the task that started
it. Each principal channel has a standing *converse* task authorized by
whoever is speaking. Each agency source (heartbeat, a feed, reminders) has a
standing task with a scope the operator configured, and turning the source on
is the authorization.

**Scope.** The set of tools available to the agent during a turn in the
task. Each identity has a **ceiling**, the largest scope it can authorize,
set in the address book.

## Rules

1. A task's scope is the lowest ceiling among the identities and sources of
   everything the task has taken in: the message that started it, messages
   added later, retrieved memories, tool results. It can drop during the
   task.
2. Scope only goes down. Nothing in a task raises it.
3. The operator raises scope by starting a new task. Context the operator
   explicitly carries into it from an earlier task counts as operator-vouched.
   That is the only untaint operation, and only a human can do it.
4. Every message, inbound and outbound, is logged with its task id and scope.
   So is every artifact derived from messages: memories, summaries, goals,
   reminders, notes and skills. Including a past item in an active task can
   lower that task's scope.
5. A task may write into a vouched directory only if its current scope is at
   least that directory's level (Biba's no write up). Otherwise a task that
   has read stranger text could write a file that a principal-level task later
   reads as principal text. A task whose scope has dropped can still write to
   an unvouched scratch area.
6. Retrieval can be scope-aware: it can leave out a memory below the current
   scope rather than include it and lower the task.
7. Enforcement happens at dispatch, and the tool schemas stay stable so the
   KV cache survives (PRD principle 2). The header shows the model its current
   scope, so it doesn't spend turns calling tools that will be refused.

## How the use cases come out

**Feeds.** Feed items are never instructions, including from accounts the
operator follows. BlueSky reading arrives as notifications, the same as RSS;
posting is a separate tool. A feed's standing task is "notice and report",
whose scope includes telling the operator about an item, since sending to the
operator isn't exfiltration. A post can't widen that, so it can't reach
`shell`. An optional refinement: a quarantined fork reads the item and hands
the main window only fields that can't carry instructions (the URL verbatim,
a relevance number, a topic from a fixed list), which keeps the window clean
at the cost of the agent not seeing the post's wording. Reactions already do
this for emoji (finding 18).

**Autonomous coding.** Lex picks tickets by label, works unattended and takes
the change to a PR. Two things keep that working:

- The label is the authorizing act. On GitHub only collaborators with triage
  rights can apply labels, so a principal accepted the ticket even if a
  stranger wrote it. The label vouches for the issue *as it read when it was
  labelled*; later edits and comments are stranger text.
- The workspace repo directory is a vouched source, so reading the code
  doesn't lower the task. Dependencies (`.venv`, `node_modules`) sit inside it
  at the directory's level. That's honest, because the agent already executes
  that code with full privileges, so reading it grants nothing new.

**Chat.** The converse task's scope is the speaker's ceiling. In a group
channel, the person whose message started the turn authorizes it, and
everyone else's text is input at their own level. Ask for something in a
guild channel and the task is yours, but other members' text in the window
lowers it.

**Across channels.** PRD §3.5.10 already makes a note-woken turn carry the
origin of the chain that started it. Scope can ride the same path: a feed task
that leaves a note on Signal wakes a turn that is still at feed scope, not at
the operator's Signal ceiling.

## The operator holds every key

Every restriction here is on by default, and the operator can open it (PRD
principle 15). That has a consequence worth stating outright: **task scoping
protects against outside intervention, not against the agent.** Lex has
`sudo` in its own container on purpose. Anything with `shell` can edit the
address book and rewrite `sessions.db`, including the scope labels rule 4
depends on.

So there are two layers:

- **Task scoping** keeps stranger text away from identity data. A task that
  has read stranger text has lost `shell`; a task that still has it has only
  read principal-level text or above. What's left is an injection hidden in a
  vouched source, such as a contributor's file or a dependency, which can
  already do the same damage by running.
- **The orchestration layer** protects the agent's identity data from the
  agent. Mount the address book (and, if wanted, an audit copy of the log)
  read-only from the host. A read-only bind mount holds against root inside
  the container as long as the container lacks `CAP_SYS_ADMIN`, which Docker
  drops by default. The operator opens this lock by changing the mount.

Defaults that follow:

- Only the operator's ceiling includes `shell` and `write_file`. The operator
  can grant them to a principal. That principal can then edit identity data,
  because tool-level scope can't say "`shell` except that file".
- A containerized runtime is the documented default, with the read-only mount
  documented alongside it.

## Known holes

- **`shell` breaks attribution.** `curl`, a `git pull` from a stranger's fork
  or a `cat` of any path brings in text the kernel can't attribute, and
  treating unknown as stranger would make every `shell` call lower the task.
  First pass: attribute `shell` output by its working directory and accept
  network reads through `shell` as a hole (`shortcut: shell network reads are
  unattributed; upgrade when tools run behind an egress proxy`). Planned fix:
  send `shell`'s network traffic through an egress proxy that records hosts,
  and attribute output by host from the address book. That's the sandboxing
  seam in #532.
- **Tool-level scope is coarse.** With scope as a set of tools, `shell` is
  effectively everything. Limits on arguments (paths, hosts, git remotes,
  target channels) would be finer and are left for later.
- **Coarseness is the Perl risk.** If too many tasks end up at stranger scope,
  the operator will turn this off. Scope-aware retrieval (rule 6) and vouched
  sources are what keep that from happening; whether they're enough is an
  empirical question for the harness.

## Not covered

**Confidentiality.** Everything here is integrity: can outside text make the
agent *do* things? The opposite question, whether private text can leak to
the wrong audience, is a separate model (Bell-LaPadula rather than Biba). In
corvidae that's the sensitivity plugin (#138) and threat model finding 15.
It deserves its own pass.

## What this touches

- **PRD.** Principle 9 says the model's authority is unbounded by decision;
  this bounds it, as a restriction the operator applies (principle 15).
  Principle 14 puts the tool-set bound on per-channel settings; this moves it
  to the person, which follows them across channels. §3.3.11 (the header)
  would carry the scope. These are owner decisions.
- **Kernel.** Task id and scope on every stored row, and the propagation rule,
  are kernel changes, not something a plugin can add.
- **Issues.** #138 (trust), #148 (tool veto), #499 and #507 (cross-channel
  trust), #283 (BlueSky), #778 (RSS), #137 and #317 (people records, which the
  address book overlaps), #532 (tool sandboxing).
- **Proof.** A harness probe that plants an instruction in a feed item and
  checks that no tool outside the feed's scope runs and that the downgrade is
  logged (PRD principle 12).
