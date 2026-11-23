"""Outcome log — the per-trace record log and the trace-origin index.

``OutcomeLogPlugin`` listens on ``on_trace_record`` (owned by the ``trace``
protocol plugin) and appends one JSON line per record to
``<session db stem>.outcomes.jsonl``, beside the session DB. It writes its
own records the same way from the message lifecycle: admitted, rejected,
persisted (with the originating message rowid). Reporters (appraisal,
critique, gates, memory, silence) never call it; they fire the hook.

It also answers ``resolve_trace_origin`` from a bounded in-memory index fed
by those records, first origin per trace winning. The index is lost on
restart.

The log accumulates every record about an exchange so retrieval quality,
lens scores, critique and gate outcomes can be correlated offline, and is
training data for a lens-selection head (#141). A trace has many lines;
readers group by ``trace_id`` and take the last line per top-level key of a
source's records.

``exchange_log``, the table this plugin wrote before #1179, is no longer
created or written. Old rows stay in old databases as history, and the
``exchange_log`` redact scrubber below still reaches them.
"""

import collections
import json
import logging
import os
import time
from pathlib import Path

import aiosqlite

from corvidae.hooks import CorvidaePlugin, get_dependency, hookimpl

# Named explicitly: corvidae/logging.py configures the "corvidae" logger,
# and __name__ would be corvidae_cognition.outcome_log.
logger = logging.getLogger("corvidae.outcome_log")

# How many traces the origin index remembers; the oldest is evicted first.
# A trace older than this resolves to None (gates then treats it as "task").
ORIGIN_INDEX_MAX = 65536

# The source name outcome_log's own lifecycle records carry.
SOURCE = "outcome_log"


async def outcomes_path(db: aiosqlite.Connection) -> Path | None:
    """The outcomes JSONL file for ``db``: beside the main database file,
    named ``<stem>.outcomes.jsonl``. None for a database with no file
    (``:memory:`` or a temporary DB).

    Used by outcome_log and by the redact scrubber, which only has the
    connection, so both find the same file.
    """
    # PRAGMA database_list rows are (seq, name, file); "main" is ours.
    async with db.execute("PRAGMA database_list") as cursor:
        rows = await cursor.fetchall()
    for _seq, name, file in rows:
        if name == "main":
            return Path(file).with_suffix(".outcomes.jsonl") if file else None
    return None


class OutcomeLogPlugin(CorvidaePlugin):
    """Appends trace records to JSONL and remembers each trace's origin."""

    depends_on = frozenset({"persistence", "trace"})

    def __init__(self, pm=None) -> None:
        if pm is not None:
            self.pm = pm
        # trace_id -> origin, oldest first, bounded by ORIGIN_INDEX_MAX.
        self._origins: collections.OrderedDict[str, str] = collections.OrderedDict()
        # The connection the cached path was resolved from, and the path
        # (None for a DB with no file). Keyed on the connection so a
        # reopened DB is resolved again.
        self._path_for: tuple[aiosqlite.Connection, Path | None] | None = None
        # The "nowhere to write" warning is said once.
        self._warned_no_file = False
        self._db_path = "sessions.db"

    @hookimpl
    async def on_start(self, config: dict) -> None:
        # Remember the configured path so the "nowhere to write" warning
        # can name it.
        self._db_path = config.get("daemon", {}).get("session_db", "sessions.db")

    def _channel_is_known(self, channel_id: str, trace_id: str) -> bool:
        """Return False when the registry never held channel_id (#261).

        A transport gate refuses strangers on a transient channel the
        registry never held. A record keyed on that id would grow the log
        without bound and keep a trace of a refused sender, and it feeds no
        calibration. With no registry plugin, fail open and record.
        """
        registry = self.pm.get_plugin("registry")
        if registry is not None and registry.get(channel_id) is None:
            logger.debug(
                "outcome_log: no record for unregistered channel %s (trace %s)",
                channel_id, trace_id,
            )
            return False
        return True

    def _index(self, trace_id: str, origin: str | None) -> str | None:
        """Record ``origin`` for ``trace_id`` unless one is already known
        (first wins, as the table's INSERT OR IGNORE did); return the
        trace's origin after this step."""
        known = self._origins.get(trace_id)
        if known is not None:
            self._origins.move_to_end(trace_id)
            return known
        if origin is None:
            return None
        self._origins[trace_id] = origin
        # Evict the least recently used traces past the bound.
        while len(self._origins) > ORIGIN_INDEX_MAX:
            self._origins.popitem(last=False)
        return origin

    async def _path(self) -> Path | None:
        """The file to append to, or None when there is nowhere to write.

        Caches only a result read from an open DB; while persistence has not
        opened its DB yet nothing is cached, so a later record retries.
        """
        from corvidae.persistence import PersistencePlugin

        db = get_dependency(self.pm, "persistence", PersistencePlugin).db
        if db is None:
            return None
        if self._path_for is None or self._path_for[0] is not db:
            self._path_for = (db, await outcomes_path(db))
        return self._path_for[1]

    async def _append(
        self, trace_id: str, channel_id: str, source: str,
        origin: str | None, record: dict,
    ) -> None:
        """Index the origin and append one line, unless the channel is a
        refused stranger's. A failed write is logged, never raised: the
        reporter's turn is not the place to fail."""
        if not self._channel_is_known(channel_id, trace_id):
            return
        origin = self._index(trace_id, origin)
        path = await self._path()
        if path is None:
            # Nowhere to write: an in-memory session DB, or persistence not
            # open yet. Say so once; the origin index still works.
            if not self._warned_no_file:
                self._warned_no_file = True
                logger.warning(
                    "outcome_log: trace records are not written: the session "
                    "DB has no file to write beside (daemon.session_db=%r); "
                    "set daemon.session_db to a file path to keep them",
                    self._db_path,
                )
            return
        line = json.dumps({
            "ts": time.time(), "trace_id": trace_id, "channel_id": channel_id,
            "source": source, "origin": origin, "record": record,
        })
        try:
            # Opened per line, so a redact rewrite (os.replace) never leaves
            # the daemon appending to the replaced file.
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError as exc:
            logger.warning(
                "outcome_log: could not append the %s record for trace %s to "
                "%s (%s); check disk space and permissions on that path",
                source, trace_id, path, type(exc).__name__,
                exc_info=True, extra={"trace_id": trace_id, "channel": channel_id},
            )

    # ------------------------------------------------------------------
    # The trace hooks
    # ------------------------------------------------------------------

    @hookimpl
    async def on_trace_record(
        self, trace_id: str, channel_id: str, source: str,
        origin: str | None, record: dict,
    ) -> None:
        """Append a reporter's record."""
        await self._append(trace_id, channel_id, source, origin, record)

    @hookimpl
    async def resolve_trace_origin(self, trace_id: str) -> str | None:
        """The first origin recorded for ``trace_id``, or None."""
        return self._origins.get(trace_id)

    # ------------------------------------------------------------------
    # Lifecycle records. Observe hooks fail soft in dispatch, so these
    # need no try/except of their own.
    # ------------------------------------------------------------------

    @hookimpl
    async def on_message_admitted(self, channel, trace_id: str, sender: str, text: str) -> None:
        await self._append(trace_id, channel.id, SOURCE, "user", {"event": "admitted"})

    @hookimpl
    async def on_message_rejected(self, channel, trace_id: str, sender: str, text: str) -> None:
        # Unregistered (transport-refused) channels are skipped in _append.
        await self._append(trace_id, channel.id, SOURCE, "user", {"event": "rejected"})

    @hookimpl
    async def on_message_persisted(
        self, channel, trace_id: str, rowid: int | None, text: str, annotations: dict,
    ) -> None:
        # A notification-born exchange has no admission record, so its
        # origin lands here; annotations["origin"] defaults to "task" for
        # background work. A user trace already holds "user" (first wins).
        origin = annotations.get("origin") or "task"
        await self._append(
            trace_id, channel.id, SOURCE, origin,
            {"event": "persisted", "message_rowid": rowid},
        )


# ------------------------------------------------------------------
# Operator redaction: the exchange_log scrubber for `corvidae redact`
# ------------------------------------------------------------------


def _scrub_past_claims(appraisal_json: str | None, tombstone: str) -> str | None:
    """Return appraisal JSON with stage2.past_claims text tombstoned, or None
    when there is nothing to change.

    Each claim and each of its queries becomes ``tombstone``, so the number
    of claims and queries survives for calibration while the words do not.
    No appraisal, no stage 2, or claims already tombstoned → None.
    """
    # NULL is the common case: most exchanges never get a stage-2 appraisal.
    # The column is written by outcome_log's JSON merge, so it parses, and
    # appraisal normalizes past_claims to a list of {claim, queries}.
    if appraisal_json is None:
        return None
    appraisal = json.loads(appraisal_json)
    stage2 = appraisal.get("stage2") or {}
    claims = stage2.get("past_claims")
    if not claims:
        return None
    scrubbed = [
        {"claim": tombstone, "queries": [tombstone] * len(claim["queries"])}
        for claim in claims
    ]
    # Already scrubbed by an earlier redaction: report no change.
    if scrubbed == claims:
        return None
    stage2["past_claims"] = scrubbed
    return json.dumps(appraisal)


def _scrub_outcomes_detail(outcomes_json: str | None, tombstone: str) -> str | None:
    """Return outcomes JSON with detail[*].claim and detail[*].objection
    tombstoned, or None when there is nothing to change.

    The critique stores objection objects (claim text + reason) in the
    outcomes envelope's provenance.detail and critique.detail lists.
    Redacting a message should reach that text; docs/redaction.md lists
    this surface among those redact covers. Counts and lens names
    survive; the words do not.

    No outcomes, no detail, or detail already tombstoned \u2192 None.
    """
    if outcomes_json is None:
        return None
    try:
        outcomes = json.loads(outcomes_json)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(outcomes, dict):
        return None

    changed = False
    for section in ("provenance", "critique"):
        detail = (outcomes.get(section) or {}).get("detail")
        if not detail or not isinstance(detail, list):
            continue
        scrubbed = []
        for entry in detail:
            if not isinstance(entry, dict):
                scrubbed.append(entry)
                continue
            e = dict(entry)
            if e.get("claim") and e["claim"] != tombstone:
                e["claim"] = tombstone
                changed = True
            if e.get("objection") and e["objection"] != tombstone:
                e["objection"] = tombstone
                changed = True
            scrubbed.append(e)
        if changed:
            outcomes[section]["detail"] = scrubbed

    if not changed:
        return None
    return json.dumps(outcomes)


async def redact_exchange_log(
    db: aiosqlite.Connection,
    channel_id: str,
    msg_ids: list[int],
    dry_run: bool,
    tombstone: str,
) -> int | None:
    """Scrub exchange_log text derived from redacted messages in one channel.

    For each redacted id X: the exchange that owns X (in the channel, the
    largest message_rowid <= X) gets its stage-2 past_claims and its
    outcomes objection detail tombstoned.
    Exchanges with a NULL message_rowid are not reached.

    Registered as the ``exchange_log`` scrubber in the ``corvidae.redact``
    entry-point group, which ``corvidae redact`` runs per channel over the
    ids it tombstoned (#160). Returns the number of exchanges changed (or
    that would be, in dry-run mode), or None when the table is absent.
    """
    # A DB whose daemon never ran outcome_log has no table to scrub.
    async with db.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'exchange_log'"
    ) as cursor:
        if await cursor.fetchone() is None:
            return None
    if not msg_ids:
        return 0
    changed: set[str] = set()

    # Find each redacted id's owning exchange, once per exchange.
    owners: set[str] = set()
    for msg_id in msg_ids:
        async with db.execute(
            "SELECT trace_id FROM exchange_log "
            "WHERE channel_id = ? AND message_rowid IS NOT NULL "
            "AND message_rowid <= ? ORDER BY message_rowid DESC LIMIT 1",
            (channel_id, msg_id),
        ) as cursor:
            row = await cursor.fetchone()
        if row is not None:
            owners.add(row[0])

    # Tombstone the owners' past_claims and outcomes detail text.
    # Sorted so the order is stable: a corrupt row must not hide the ones
    # after it.
    for trace_id in sorted(owners):
        async with db.execute(
            # CAST AS BLOB so invalid UTF-8 stored as TEXT fails at this
            # row's decode, not inside fetchone() (#232, #311).
            "SELECT CAST(appraisal AS BLOB), CAST(outcomes AS BLOB) "
            "FROM exchange_log WHERE trace_id = ?",
            (trace_id,),
        ) as cursor:
            row = await cursor.fetchone()
        # A corrupt value (bad JSON, invalid UTF-8, deep nesting, JSON that
        # is not an envelope object, or a claims entry missing its keys)
        # skips with one WARNING; the cascade completes over the other
        # exchanges (#311). The row is left as-is — only out-of-band writes
        # can produce one, and the operator is told where it is.
        try:
            scrubbed_appraisal = _scrub_past_claims(row[0], tombstone)
        except (ValueError, TypeError, RecursionError, AttributeError,
                KeyError):
            logger.warning(
                "skipping corrupt appraisal for exchange %s on channel %s; "
                "its past_claims are left unscrubbed — inspect or correct "
                "the row",
                trace_id, channel_id,
                exc_info=True,
                extra={"channel": channel_id, "trace_id": trace_id},
            )
            scrubbed_appraisal = None
        try:
            scrubbed_outcomes = _scrub_outcomes_detail(row[1], tombstone)
        except (ValueError, TypeError, RecursionError, AttributeError,
                KeyError):
            logger.warning(
                "skipping corrupt outcomes for exchange %s on channel %s; "
                "its detail is left unscrubbed — inspect or correct "
                "the row",
                trace_id, channel_id,
                exc_info=True,
                extra={"channel": channel_id, "trace_id": trace_id},
            )
            scrubbed_outcomes = None
        if scrubbed_appraisal is None and scrubbed_outcomes is None:
            continue
        changed.add(trace_id)
        if not dry_run:
            if scrubbed_appraisal is not None:
                await db.execute(
                    "UPDATE exchange_log SET appraisal = ? WHERE trace_id = ?",
                    (scrubbed_appraisal, trace_id),
                )
            if scrubbed_outcomes is not None:
                await db.execute(
                    "UPDATE exchange_log SET outcomes = ? WHERE trace_id = ?",
                    (scrubbed_outcomes, trace_id),
                )

    # Persist the scrubbed rows in one transaction.
    if not dry_run:
        await db.commit()
    return len(changed)


async def redact_outcomes_jsonl(
    db: aiosqlite.Connection,
    channel_id: str,
    msg_ids: list[int],
    dry_run: bool,
    tombstone: str,
) -> int | None:
    """Scrub text derived from redacted messages out of the outcomes JSONL.

    For each redacted id X: the trace that owns X (on the channel, the
    largest persisted ``message_rowid`` <= X, as for exchange_log) gets the
    claim text in its appraisal records (``stage2.past_claims``) and the
    objection text in its critique records (``provenance``/``critique``
    ``detail``) tombstoned; counts survive. The file is rewritten through a
    temporary file and ``os.replace``; a dry run counts and writes nothing.

    Registered as the ``outcome_log_jsonl`` scrubber in the
    ``corvidae.redact`` entry-point group. Returns the number of traces
    changed (or that would be), or None when there is no file for this DB.
    """
    path = await outcomes_path(db)
    if path is None or not path.exists():
        return None
    if not msg_ids:
        return 0
    raw_lines = path.read_text(encoding="utf-8").splitlines(keepends=True)

    # Parse every line once. A line that does not parse is kept as it is,
    # with one WARNING naming where it is (the corrupt-row rule, #311).
    parsed: list[dict | None] = []
    for number, raw in enumerate(raw_lines, start=1):
        try:
            entry = json.loads(raw)
            if not isinstance(entry, dict):
                raise ValueError("not a JSON object")
        except (ValueError, RecursionError):
            logger.warning(
                "skipping unparseable line %d of %s during redaction; it is "
                "left as-is — inspect or delete it by hand",
                number, path, exc_info=True, extra={"channel": channel_id},
            )
            entry = None
        parsed.append(entry)

    # The persisted message rowid of each trace on this channel.
    rowids = {
        e["trace_id"]: e["record"]["message_rowid"]
        for e in parsed
        if e is not None and e.get("channel_id") == channel_id
        and e.get("source") == SOURCE
        and isinstance(e.get("record"), dict)
        and e["record"].get("event") == "persisted"
        and e["record"].get("message_rowid") is not None
    }
    # Each redacted id's owner: the trace with the largest rowid <= the id.
    owners: set[str] = set()
    for msg_id in msg_ids:
        candidates = [(rowid, t) for t, rowid in rowids.items() if rowid <= msg_id]
        if candidates:
            owners.add(max(candidates)[1])

    # Tombstone the owners' claim and objection text, line by line.
    changed: set[str] = set()
    for index, entry in enumerate(parsed):
        if entry is None or entry.get("trace_id") not in owners:
            continue
        scrub = {"appraisal": _scrub_past_claims,
                 "critique": _scrub_outcomes_detail}.get(entry.get("source"))
        if scrub is None:
            continue
        try:
            scrubbed = scrub(json.dumps(entry.get("record")), tombstone)
        except (ValueError, TypeError, AttributeError, KeyError):
            logger.warning(
                "skipping malformed %s record on line %d of %s for trace %s; "
                "its text is left unscrubbed — inspect or correct the line",
                entry.get("source"), index + 1, path, entry.get("trace_id"),
                exc_info=True, extra={"channel": channel_id},
            )
            continue
        if scrubbed is None:
            continue
        changed.add(entry["trace_id"])
        entry["record"] = json.loads(scrubbed)
        raw_lines[index] = json.dumps(entry) + "\n"

    # Rewrite the file in one step so a reader never sees half of it.
    if changed and not dry_run:
        temp = path.with_name(path.name + ".tmp")
        temp.write_text("".join(raw_lines), encoding="utf-8")
        os.replace(temp, path)
    return len(changed)
