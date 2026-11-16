# Redaction

**Status: on hold (#1005). Nothing in this document is a requirement.**
The PRD keeps one binding stub (§3.7 Memory 8): `corvidae redact` is the
operator-only deletion path, it tombstones `message_log` content in place,
and it is never a registered tool or reachable from a hook the model can
influence. Everything else a redaction does, or was once required to do, is
recorded here so the overhaul can start from it. This page is not a
requirements document. It says why redaction is on hold, what the shipped
command does and doesn't reach, what may change while the hold lasts, and
the requirements as they stood before the hold.

## Why it is on hold

Redaction can't be made complete in this framework. That isn't a matter of
finding the remaining bugs: the design can't get there.

**The cascade is opt-in and open-ended.** `corvidae redact` tombstones the
source row, then chases copies it knows about: the memory records on the
message's channel, both FTS indexes, and whatever tables plugins have
registered a scrubber for under the `corvidae.redact` entry-point group.
Nothing makes a plugin register one, and every new feature that stores text
is a new place for a copy to sit. The same leak has turned up again and
again:

- `exchange_log.past_claims` wasn't reached until a scrubber was added (#160).
- A message coalesced into a burst leaves its `past_claims` on the turn that
  answered it, which the scrubber doesn't find (#423).
- Proactivity objection text in `exchange_log.outcomes` isn't scrubbed (#1003).
- Goal text has no scrubber (threat model, finding 10).

**The model restates what it reads.** Once a message has been in the
window, any later reply, memory record, goal, note, critique objection or
cross-channel send can say the same thing in different words. A scrubber
can only find copies it can recognise, and a paraphrase isn't one. The only
complete answer is to treat everything derived from any window that held
the message as tainted. That set grows with every turn after the message,
and because the agent is one identity across channels (PRD principle 14) it
crosses channels too.

The former requirement's completeness check makes the gap concrete: it
searches the two FTS indexes for one sample word from the redacted text.
That finds a verbatim copy in those two indexes and nothing else.

## What the shipped command does

The mechanics are in "Operator redaction" in [design.md](design.md#operator-redaction-corvidae-redact).
In short, `corvidae redact` reaches:

- the `message_log` rows, tombstoned in place (row and id remain);
- `message_fts` and `memory_fts`, when present, through their update triggers;
- memory records on each redacted message's own channel whose range covers
  it: summary tombstoned, `redacted = 1`, removed from the index, vector
  deleted;
- every table a plugin has registered a scrubber for. The cognition package
  registers `exchange_log` (`past_claims` and the critique `outcomes`
  detail; `provenance_snapshot` is dropped) and `retrieval_candidate`.

Every completed run, real or `--dry-run`, ends with a notice naming what it
does not reach and pointing here.

## What it does not reach

- The JSONL conversation logs (`daemon.jsonl_log_dir`, when enabled) and the
  daemon log. These are
  separate files; delete or edit them yourself.
- Backups, calibration exports and any other copy taken out of `sessions.db`.
- Copies on other channels: a read of the redacted message, a note or send
  that restates it, and whatever those channels consolidated from them.
- The running daemon's context window. It still holds the original text
  until compaction or restart, and a compaction or live-window consolidation
  could write it back into new rows. Stop the daemon before redacting.
- Restatements: later replies, memory records outside the covered range,
  goals, notes and critique objections that paraphrase the message.
- `exchange_log` rows whose `message_rowid` is NULL.
- Proactivity objection detail in `exchange_log.outcomes` (#1003).
- The `past_claims` of the turn that answered a coalesced burst, when an
  earlier message in the burst is redacted (#423).
- Goal text (`goals` table).
- Any plugin table with no registered scrubber.

[threat-model.md](threat-model.md) carries the same list from the security
side (finding 10 for goals; "Smaller things" for the rest).

## What may change during the hold

- **Narrow leak fixes go ahead.** A fix that closes a known gap in the
  shipped command without changing its design is fine: #1003, which adds a
  missing section to an existing scrubber, is the model. Each one still
  gets a review loop.
- **Redesigns wait for the overhaul.** Anything that needs a new kernel
  contract, a new hook, or a new way of finding derived copies waits. #423
  is one: fixing it needs a link from an absorbed message's trace to the
  trace that answered it, which the kernel doesn't record.
- **Prevention is not redaction.** #508 stops secrets before they are
  written anywhere, at the one point a tool result passes through before it
  enters the conversation (`transform_tool_result`). Because a secret is
  copied verbatim or not at all, an exact-match filter there doesn't have
  the paraphrase problem. It falls outside the hold.
- **Scrubbers keep working.** A plugin that copies conversation text into
  its own table should still register one ([plugin-guide.md](plugin-guide.md#registering-a-redact-scrubber));
  a scrubber narrows the gap even though it can't close it.

## Directions for the overhaul

These are candidates, not decisions. None has been designed or tested.

- **Record where derived rows came from.** If every derived row named the
  message ids it was built from, as a kernel contract, redact could follow
  those links instead of relying on plugins to register. This catches
  copies, not paraphrase, and the provenance of a reply is "everything in
  the window", which is most of the channel.
- **Re-derive instead of scrubbing.** Tombstone the source, then rebuild
  what was derived after it from the tombstoned log. Related to the
  operator re-consolidation in #932. Expensive, and it reaches only what can
  be rebuilt.
- **Stop it at the door.** Filter before persistence (#508). Complete for
  verbatim secrets, useless for anything the operator only later decides is
  private.
- **Narrow the promise.** State that redaction removes the source and its
  verbatim copies, and that derived knowledge isn't guaranteed forgotten,
  the way a person who heard something can't unhear it. This is true today;
  the question is whether it is enough.

Open questions for the overhaul:

- Should the deletion path's safety properties (assert WAL, probe for
  schema, never create schema, report found-of-requested) go back into the
  PRD with the binding stub? They were moved out with the rest of Memory 8;
  they concern safety, not completeness.
- What does an operator need redaction for in practice: a secret pasted
  into a channel, a conversation someone asked to have forgotten, or legal
  erasure? Each sets a different bar.

## The requirement as it stood before the hold

PRD §3.7 Memory 8, verbatim, before #1005:

> 8. `corvidae redact` (`message`, `memory`, `range` forms, `--dry-run`)
>    tombstones `message_log` content in place and cascades to every derived
>    surface on the same channel: memory summaries, flags, vectors, both FTS
>    indexes. It reaches only the channel it is run on: copies on other
>    channels (reads, notes and sends, §3.5.9–11) are not followed. It
>    asserts WAL, probes for schema before touching it, never creates schema,
>    reports found-of-requested, and verifies its own completeness with an
>    FTS probe for a sample token. It is never a registered tool and never
>    reachable from a hook the model can influence.

The rest of the PRD's references to redaction stay, because they hold under
the stub: principle 4 (the tombstone is the log's one sanctioned mutation),
principle 8 (the redact CLI is a second process, so WAL is required), the
operator-control floor (`redact` is never a tool), §3.7 Memory 4 (retrieval
skips redacted records), the subcommand list, and the decisions-table row.
