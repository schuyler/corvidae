"""The content record of every fork: inputs, findings, verdict, label (#1204).

``usage_log`` counts what a fork spent; this table records what it saw and
produced, so a tuning query or a distillation corpus can join the two by
``trace_id``.

Capture is content-addressed, not a snapshot. A fork's prompt is a list of
rendered messages, so the row keeps ``prompt_sha256`` over that list plus a
reference to the window rows it came from; the message text itself is already
durable in ``message_log``. The one rendered input that is not durable -- the
system prompt, which lives only in the live window -- is stored once per
version in ``fork_block``, keyed by its own hash. A sampled fraction of forks
also stores the whole rendered list there, as a check on the reconstruction.

Storage is bounded by configuration: :meth:`ForkLog.evict` drops text in a
fixed order and never drops a row.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import zlib
from typing import Any, Iterable, Sequence

import aiosqlite

from corvidae.persistence import DEFAULT_BUSY_TIMEOUT_MS

logger = logging.getLogger("corvidae.fork_log")

SCHEMA = """
CREATE TABLE IF NOT EXISTS fork_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    stage TEXT NOT NULL,
    point TEXT,
    channel_id TEXT NOT NULL,
    trace_id TEXT,
    lens_id TEXT,
    profile TEXT,
    model TEXT,
    question_sha256 BLOB,
    question_text TEXT,
    instruction TEXT,
    window_first_rowid INTEGER,
    window_last_rowid INTEGER,
    window_cut INTEGER,
    prompt_sha256 BLOB NOT NULL,
    block_count INTEGER NOT NULL DEFAULT 0,
    renderer_sha TEXT,
    findings_json TEXT,
    verdict TEXT,
    label TEXT,
    label_reason TEXT,
    latency_ms REAL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    cached_tokens INTEGER,
    snapshot_sha256 TEXT,
    system_sha256 TEXT,
    chain_parent INTEGER,
    chain_from INTEGER NOT NULL DEFAULT 0,
    reconstructable INTEGER NOT NULL DEFAULT 1
);
-- Only the block digests a fork appends to its parent's list: repeating the
-- whole array per fork costs more than the row (measured, #1204).
CREATE TABLE IF NOT EXISTS fork_digest (
    fork_id INTEGER NOT NULL,
    ordinal INTEGER NOT NULL,
    digest BLOB NOT NULL,
    PRIMARY KEY (fork_id, ordinal)
);

-- Rendered text that has no durable home elsewhere, compressed, by hash.
CREATE TABLE IF NOT EXISTS fork_block (
    sha256 TEXT PRIMARY KEY,
    bytes INTEGER NOT NULL,
    created_at REAL NOT NULL,
    text BLOB NOT NULL
);
"""

#: Built after the migration: an index over a late column cannot exist before
#: that column does.
INDEXES = """
CREATE INDEX IF NOT EXISTS fork_log_ts ON fork_log (ts);
CREATE INDEX IF NOT EXISTS fork_log_stage_ts ON fork_log (stage, ts);
CREATE INDEX IF NOT EXISTS fork_log_label ON fork_log (label, stage);
"""

#: Labels a finding can carry. ``ignored`` is deliberately absent: no timer
#: writes a label, so a finding nobody resolved stays unlabeled and the census
#: reports its age instead of a rate.
LABELS = frozenset({"accepted", "declined", "superseded"})


def prompt_sha256(messages: Sequence[dict]) -> bytes:
    """The hash of a rendered prompt: the messages, in order, as sent.

    Raw 32 bytes, not 64 hex characters: the hex form doubles the field on
    every row, and there is one row per fork (#1204).
    """
    blob = json.dumps(
        messages, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(blob).digest()


def window_blocks(channel) -> list[dict]:
    """The window a fork renders, as blocks; empty when there is none."""
    conv = getattr(channel, "conversation", None)
    return list(getattr(conv, "messages", None) or [])


def digest_of(block: dict | str) -> bytes:
    """A rendered block's 16-byte digest.

    The whole block, not just its text: two sends can share ``content`` and
    differ in ``role`` or ``tool_calls``, and a digest list that cannot tell
    them apart does not identify the prompt (#1204, found in review).
    """
    if isinstance(block, str):
        payload = block.encode()
    else:
        payload = json.dumps(
            block, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    return hashlib.blake2b(payload, digest_size=16).digest()


def question_sha256(question: str | None) -> bytes | None:
    """The hash of the lens question, so rows join by what asked them (#1128)."""
    if not question:
        return None
    return hashlib.sha256(question.strip().encode()).digest()


def _dumps(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


class ForkLog:
    """One connection to the fork log; open lazily, close on shutdown."""

    def __init__(
        self,
        db_path: str,
        *,
        sample_every: int = 50,
        busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
    ) -> None:
        self._db_path = db_path
        self._sample_every = max(1, sample_every)
        self._busy_timeout_ms = busy_timeout_ms
        self._seen = 0
        #: Forks whose row could not be written. Non-zero means the corpus has
        #: a hole the queries cannot see, so the caller escalates it.
        self.write_failures = 0
        self._db: aiosqlite.Connection | None = None
        #: Held by every write from its first statement to its commit or
        #: rollback. Writers share one connection, so without it one call's
        #: rollback would discard another call's uncommitted rows, and one
        #: call's commit would publish another's half-written record (#1211).
        self._write_lock = asyncio.Lock()

    async def open(self) -> None:
        if self._db is not None:
            return
        db = await aiosqlite.connect(self._db_path)
        db.row_factory = aiosqlite.Row
        # Shares the session database with the conversation writer; without a
        # bound of its own this writer waits sqlite3's 5.0 s default and drops
        # fork rows whenever another connection holds the write lock
        # (corvidae#1211).
        await db.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}")
        await db.executescript(SCHEMA)
        await self._migrate(db)
        await db.executescript(INDEXES)
        await db.commit()
        self._db = db

    #: Columns added after the first release. ``CREATE TABLE IF NOT EXISTS``
    #: does nothing to a table that already exists, so a database written by an
    #: older build never gains them and every write fails with
    #: "no column named ..." -- silently, because writers swallow errors.
    LATE_COLUMNS = {
        "point": "TEXT",
        "trace_id": "TEXT",
        "block_count": "INTEGER NOT NULL DEFAULT 0",
        "lens_id": "TEXT",
        "profile": "TEXT",
        "model": "TEXT",
        "question_sha256": "BLOB",
        "question_text": "TEXT",
        "instruction": "TEXT",
        "window_first_rowid": "INTEGER",
        "window_last_rowid": "INTEGER",
        "window_cut": "INTEGER",
        "renderer_sha": "TEXT",
        "findings_json": "TEXT",
        "verdict": "TEXT",
        "label": "TEXT",
        "label_reason": "TEXT",
        "latency_ms": "REAL",
        "prompt_tokens": "INTEGER",
        "completion_tokens": "INTEGER",
        "cached_tokens": "INTEGER",
        "snapshot_sha256": "TEXT",
        "system_sha256": "TEXT",
        "chain_parent": "INTEGER",
        "chain_from": "INTEGER NOT NULL DEFAULT 0",
        "reconstructable": "INTEGER NOT NULL DEFAULT 1",
    }

    async def _migrate(self, db: aiosqlite.Connection) -> None:
        async with db.execute("PRAGMA table_info(fork_log)") as cur:
            have = {row[1] for row in await cur.fetchall()}
        for name, decl in self.LATE_COLUMNS.items():
            if name not in have:
                await db.execute(f"ALTER TABLE fork_log ADD COLUMN {name} {decl}")
                logger.info("fork_log: added column %s to an existing table", name)
        # A defaulted column lies about rows written before it existed: they
        # claim zero blocks, which the length check would read as damage. This
        # runs every open so a rewound row is repaired too; it is idempotent.
        await db.execute(
            "UPDATE fork_log SET block_count = coalesce(chain_from, 0) +"
            " (SELECT count(*) FROM fork_digest"
            "  WHERE fork_digest.fork_id = fork_log.id)"
            " WHERE block_count = 0"
        )

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def _conn(self) -> aiosqlite.Connection:
        await self.open()
        assert self._db is not None
        return self._db

    async def store_block(self, text: str) -> str:
        """Store rendered text once, by content hash; returns the hash."""
        sha = hashlib.sha256(text.encode()).hexdigest()
        db = await self._conn()
        async with self._write_lock:
            try:
                await db.execute(
                    "INSERT OR IGNORE INTO fork_block (sha256, bytes, created_at, text)"
                    " VALUES (?, ?, ?, ?)",
                    (sha, len(text), time.time(), zlib.compress(text.encode(), 6)),
                )
                await db.commit()
            except BaseException:
                # Same reason as _insert_fork: a cancelled store must not
                # leave its transaction holding the write lock (#1211).
                await db.rollback()
                raise
        return sha

    async def read_block(self, sha: str) -> str | None:
        db = await self._conn()
        async with db.execute(
            "SELECT text FROM fork_block WHERE sha256 = ?", (sha,)
        ) as cur:
            row = await cur.fetchone()
        return None if row is None else zlib.decompress(row["text"]).decode()

    async def record(
        self,
        *,
        stage: str,
        channel_id: str,
        point: str | None = None,
        messages: Sequence[dict],
        instruction: str | None = None,
        question: str | None = None,
        findings: Iterable[dict] | None = None,
        verdict: str | None = None,
        trace_id: str | None = None,
        lens_id: str | None = None,
        profile: str | None = None,
        model: str | None = None,
        window: tuple[int, int] | None = None,
        cut: int | None = None,
        renderer_sha: str | None = None,
        latency_ms: float | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        cached_tokens: int | None = None,
        system_prompt: str | None = None,
        store_snapshot: bool = False,
        digests: Sequence[bytes] | None = None,
        chain_parent: int | None = None,
        chain_from: int = 0,
    ) -> int:
        """Write one fork row; returns its id.

        ``messages`` is the rendered prompt; only its hash is written unless
        ``store_snapshot`` asks for the text too. ``system_prompt`` is stored
        once per version, deduped by hash, because it is the one rendered
        input with no durable home.
        """
        db = await self._conn()
        # Every Nth fork keeps its rendered prompt: the sampling rate is the
        # only thing that costs text, and nothing else reads it.
        self._seen += 1
        if self._seen % self._sample_every == 0:
            store_snapshot = True
        snapshot_sha = None
        system_sha = None
        if system_prompt:
            system_sha = await self.store_block(system_prompt)
        if store_snapshot:
            snapshot_sha = await self.store_block(
                json.dumps(list(messages), ensure_ascii=False)
            )
        # No digests means the send cannot be identified later: say so on the
        # row instead of letting an empty chain look like an empty prompt.
        reconstructable = 0 if digests is None else 1
        first, last = (None, None) if window is None else window
        return await self._insert_fork(
            db,
            (
                time.time(),
                stage,
                point,
                channel_id,
                trace_id,
                lens_id,
                profile,
                model,
                question_sha256(question),
                question,
                instruction,
                first,
                last,
                cut,
                prompt_sha256(messages),
                len(messages),
                renderer_sha,
                _dumps(list(findings) if findings is not None else None),
                verdict,
                None,
                latency_ms,
                prompt_tokens,
                completion_tokens,
                cached_tokens,
                snapshot_sha,
                system_sha,
                chain_parent,
                chain_from,
                reconstructable,
            ),
            messages=messages,
            digests=digests,
            chain_parent=chain_parent,
            chain_from=chain_from,
        )

    async def _insert_fork(
        self,
        db: aiosqlite.Connection,
        row: tuple,
        *,
        messages: Sequence[dict],
        digests: Sequence[bytes] | None,
        chain_parent: int | None,
        chain_from: int,
    ) -> int:
        """Insert one fork row and its digests, then commit; returns its id."""
        async with self._write_lock:
            try:
                cur = await db.execute(
                    """INSERT INTO fork_log (
                        ts, stage, point, channel_id, trace_id, lens_id, profile, model,
                        question_sha256, question_text, instruction,
                        window_first_rowid, window_last_rowid, window_cut,
                        prompt_sha256, block_count, renderer_sha, findings_json, verdict, label,
                        latency_ms, prompt_tokens, completion_tokens, cached_tokens,
                        snapshot_sha256, system_sha256, chain_parent, chain_from,
                        reconstructable
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    row,
                )
                fork_id = int(cur.lastrowid)
                if digests:
                    digests = await self._check_digests(
                        messages, digests, chain_parent=chain_parent, chain_from=chain_from
                    )
                    await db.executemany(
                        "INSERT OR REPLACE INTO fork_digest (fork_id, ordinal, digest)"
                        " VALUES (?, ?, ?)",
                        [(fork_id, chain_from + i, d) for i, d in enumerate(digests)],
                    )
                await db.commit()
                return fork_id
            except BaseException:
                # The INSERT opened a write transaction; anything that fails
                # before the commit (a refused digest chain, a cancelled task)
                # must end it here. Left open it holds the database's write
                # lock until this connection's next commit, and every other
                # writer on the file -- the conversation log included -- fails
                # with "database is locked" meanwhile (#1211). The write lock
                # keeps this rollback to this call's own rows.
                await db.rollback()
                raise

    async def _check_digests(
        self,
        messages: Sequence[dict],
        digests: Sequence[bytes],
        *,
        chain_parent: int | None,
        chain_from: int,
    ) -> Sequence[bytes]:
        """Check a digest list, returning the tail this row stores."""
        prefix: list[bytes] = []
        if chain_from:
            if chain_parent is None:
                raise ValueError("fork_log: chain_from without a chain_parent")
            parent = await self.resolve_chain(chain_parent)
            if parent is None or len(parent) != chain_from:
                raise ValueError("fork_log: chain_parent does not resolve")
            prefix = parent
        expected = [digest_of(block) for block in messages]
        # Two shapes are legal: the tail this row appends, or the whole list the
        # caller already has. Anything else is a length mistake, not a content
        # mistake, and the two deserve different messages.
        if len(digests) == len(expected):
            digests = digests[chain_from:]
        elif len(digests) != len(expected) - chain_from:
            raise ValueError(
                "fork_log: digests have the wrong length: pass the blocks this"
                f" row appends ({len(expected) - chain_from}) or the whole"
                f" prompt ({len(expected)}), got {len(digests)}"
            )
        if [*prefix, *digests] != expected:
            raise ValueError(
                "fork_log: digests do not describe the rendered blocks"
            )
        return digests

    async def resolve_chain(
        self, fork_id: int, _memo: dict[int, list[bytes]] | None = None
    ) -> list[bytes] | None:
        """The fork's full block list, walked back through its parents.

        None when the chain cannot be rebuilt: a missing parent, a row marked
        ``reconstructable = 0``, or a parent whose length does not match the
        offset this row claims. A partial list would describe a different
        prompt, so it is not returned.
        """
        db = await self._conn()
        memo = {} if _memo is None else _memo
        if fork_id in memo:
            return memo[fork_id]
        async with db.execute(
            "SELECT chain_parent, chain_from, reconstructable FROM fork_log"
            " WHERE id = ?",
            (fork_id,),
        ) as cur:
            row = await cur.fetchone()
        if row is None or not row["reconstructable"]:
            return None
        async with db.execute(
            "SELECT ordinal, digest FROM fork_digest WHERE fork_id = ?"
            " ORDER BY ordinal",
            (fork_id,),
        ) as cur:
            own_rows = list(await cur.fetchall())
        own = [bytes(r["digest"]) for r in own_rows]
        # The stored ordinals must be contiguous from the offset the row claims:
        # checking only the first would rebuild a short chain out of a gap.
        if own_rows and [r["ordinal"] for r in own_rows] != list(
            range(row["chain_from"], row["chain_from"] + len(own_rows))
        ):
            return None
        base, parent = row["chain_from"], row["chain_parent"]
        if base == 0:
            blocks = [] if parent is None else None
        elif parent is None:
            blocks = None
        else:
            inherit = await self.resolve_chain(parent, memo)
            blocks = None if inherit is None or len(inherit) != base else inherit
        if blocks is None:
            return None
        resolved = [*blocks, *own]
        async with db.execute(
            "SELECT block_count FROM fork_log WHERE id = ?", (fork_id,)
        ) as cur:
            claim = await cur.fetchone()
        if claim is not None and claim["block_count"] != len(resolved):
            return None
        memo[fork_id] = resolved
        return resolved

    async def mark_unreconstructable(self, fork_ids: Iterable[int]) -> None:
        db = await self._conn()
        async with self._write_lock:
            await db.executemany(
                "UPDATE fork_log SET reconstructable = 0 WHERE id = ?",
                [(i,) for i in fork_ids],
            )
            await db.commit()

    async def set_label(
        self, fork_id: int, label: str, *, reason: str | None = None
    ) -> None:
        """Set a terminal label. Refuses anything not in :data:`LABELS`."""
        if label not in LABELS:
            raise ValueError(
                f"fork_log: unknown label {label!r}; one of {sorted(LABELS)}"
            )
        db = await self._conn()
        async with self._write_lock:
            await db.execute(
                "UPDATE fork_log SET label = ?, label_reason = ? WHERE id = ?",
                (label, reason, fork_id),
            )
            await db.commit()

    async def counts(self, since: float | None = None) -> list[aiosqlite.Row]:
        """Rows per stage and label, with the oldest unlabeled row's age."""
        db = await self._conn()
        sql = (
            "SELECT stage, coalesce(label, 'unlabeled') AS label, count(*) AS n,"
            " min(CASE WHEN label IS NULL THEN ts END) AS oldest_unlabeled"
            " FROM fork_log"
        )
        args: tuple = ()
        if since is not None:
            sql += " WHERE ts >= ?"
            args = (since,)
        sql += " GROUP BY stage, label ORDER BY stage, n DESC"
        async with db.execute(sql, args) as cur:
            return list(await cur.fetchall())

    async def bytes_used(self) -> int:
        db = await self._conn()
        total = 0
        for sql in (
            "SELECT coalesce(sum(length(findings_json)), 0) +"
            " coalesce(sum(length(instruction)), 0) +"
            " coalesce(sum(length(question_text)), 0) AS n FROM fork_log",
            "SELECT coalesce(sum(length(text)), 0) AS n FROM fork_block",
        ):
            async with db.execute(sql) as cur:
                row = await cur.fetchone()
                total += int(row["n"] or 0)
        return total

    async def evict(self, max_bytes: int) -> int:
        """Drop text until the log fits ``max_bytes``; returns bytes freed.

        Fixed order, and rows are never dropped: sampled snapshots first, then
        the remaining block text oldest-first. Labels, hashes, findings and
        question text stay.
        """
        # One lock for the whole pass: _drop_block writes on the shared
        # connection too and must not interleave with a record's rollback.
        async with self._write_lock:
            db = await self._conn()
            used = await self.bytes_used()
            if used <= max_bytes:
                return 0
            freed = 0
            async with db.execute(
                "SELECT id, snapshot_sha256 FROM fork_log"
                " WHERE snapshot_sha256 IS NOT NULL ORDER BY ts"
            ) as cur:
                snapshots = list(await cur.fetchall())
            for row in snapshots:
                if used <= max_bytes:
                    break
                dropped = await self._drop_block(row["snapshot_sha256"])
                await db.execute(
                    "UPDATE fork_log SET snapshot_sha256 = NULL WHERE id = ?",
                    (row["id"],),
                )
                await db.commit()
                used -= dropped
                freed += dropped
            if used > max_bytes:
                async with db.execute(
                    "SELECT sha256 FROM fork_block ORDER BY created_at"
                ) as cur:
                    blocks = list(await cur.fetchall())
                for row in blocks:
                    if used <= max_bytes:
                        break
                    dropped = await self._drop_block(row["sha256"])
                    if not dropped:
                        continue
                    used -= dropped
                    freed += dropped
                    # Only the rows that named this text lose reconstructability.
                    await db.execute(
                        "UPDATE fork_log SET reconstructable = 0"
                        " WHERE system_sha256 = ?",
                        (row["sha256"],),
                    )
                    await db.commit()
            return freed

    async def _drop_block(self, sha: str) -> int:
        """Delete one stored block; the caller holds ``_write_lock``."""
        db = await self._conn()
        async with db.execute(
            "SELECT bytes, text FROM fork_block WHERE sha256 = ?", (sha,)
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return 0
        await db.execute("DELETE FROM fork_block WHERE sha256 = ?", (sha,))
        await db.commit()
        return len(row["text"])


class ForkRecorder:
    """A plugin's view of the fork log: one connection, per-channel chains.

    The chaining state is per recorder because a chain only extends within one
    conversation; the failure counter is here so a caller can escalate.
    """

    def __init__(
        self,
        db_path: str,
        *,
        sample_every: int = 50,
        busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
    ) -> None:
        self.log = ForkLog(
            db_path, sample_every=sample_every, busy_timeout_ms=busy_timeout_ms
        )
        self._last: dict[str, tuple[int, list[bytes]]] = {}

    @property
    def write_failures(self) -> int:
        return self.log.write_failures

    async def close(self) -> None:
        await self.log.close()

    async def record(
        self,
        *,
        channel,
        stage: str,
        point: str | None,
        blocks: list[dict],
        lens_id: str | None = None,
        instruction: str,
        trace_id: str | None = None,
        result=None,
        question: str | None = None,
        findings: list[dict] | None = None,
    ) -> int | None:
        """Record one fork; returns the row id, or None when nothing was written.

        Never raises: a corpus hole must not change what the fork returned.
        """
        if not blocks:
            return None
        try:
            own = [digest_of(b) for b in blocks]
            parent_id, parent_blocks = self._last.get(channel.id, (0, []))
            shared = 0
            while (
                shared < len(parent_blocks)
                and shared < len(own)
                and parent_blocks[shared] == own[shared]
            ):
                shared += 1
            chained = bool(parent_blocks) and shared == len(parent_blocks)
            conv = getattr(channel, "conversation", None)
            fork_id = await self.log.record(
                stage=stage,
                point=point,
                channel_id=channel.id,
                messages=blocks,
                digests=own,
                chain_parent=parent_id if chained else None,
                chain_from=shared if chained else 0,
                instruction=instruction,
                lens_id=lens_id,
                question=question,
                findings=findings,
                verdict=getattr(result, "text", None),
                trace_id=trace_id,
                prompt_tokens=getattr(result, "prompt_tokens", None),
                completion_tokens=getattr(result, "completion_tokens", None),
                system_prompt=getattr(conv, "system_prompt", None),
            )
            self._last[channel.id] = (fork_id, own)
            return fork_id
        except Exception:
            self.log.write_failures += 1
            logger.log(
                logging.ERROR if self.log.write_failures == 1 else logging.WARNING,
                "fork_log: could not record a %s fork for %s, %s write failure(s)"
                " so far (#1204)",
                stage, getattr(channel, "id", "?"), self.log.write_failures,
                exc_info=True,
            )
            return None
