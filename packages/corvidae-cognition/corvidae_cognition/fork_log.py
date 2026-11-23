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

import hashlib
import json
import logging
import time
import zlib
from typing import Any, Iterable, Sequence

import aiosqlite

logger = logging.getLogger("corvidae.fork_log")

SCHEMA = """
CREATE TABLE IF NOT EXISTS fork_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    stage TEXT NOT NULL,
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
CREATE INDEX IF NOT EXISTS fork_log_ts ON fork_log (ts);
CREATE INDEX IF NOT EXISTS fork_log_stage_ts ON fork_log (stage, ts);
CREATE INDEX IF NOT EXISTS fork_log_label ON fork_log (label, stage);

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

    def __init__(self, db_path: str, *, sample_every: int = 50) -> None:
        self._db_path = db_path
        self._sample_every = max(1, sample_every)
        self._db: aiosqlite.Connection | None = None

    async def open(self) -> None:
        if self._db is not None:
            return
        db = await aiosqlite.connect(self._db_path)
        db.row_factory = aiosqlite.Row
        await db.executescript(SCHEMA)
        await db.commit()
        self._db = db

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
        await db.execute(
            "INSERT OR IGNORE INTO fork_block (sha256, bytes, created_at, text)"
            " VALUES (?, ?, ?, ?)",
            (sha, len(text), time.time(), zlib.compress(text.encode(), 6)),
        )
        await db.commit()
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
        cur = await db.execute(
            """INSERT INTO fork_log (
                ts, stage, channel_id, trace_id, lens_id, profile, model,
                question_sha256, question_text, instruction,
                window_first_rowid, window_last_rowid, window_cut,
                prompt_sha256, renderer_sha, findings_json, verdict, label,
                latency_ms, prompt_tokens, completion_tokens, cached_tokens,
                snapshot_sha256, system_sha256, chain_parent, chain_from,
                reconstructable
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                time.time(),
                stage,
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
        if own_rows and own_rows[0]["ordinal"] != row["chain_from"]:
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
        memo[fork_id] = [*blocks, *own]
        return memo[fork_id]

    async def mark_unreconstructable(self, fork_ids: Iterable[int]) -> None:
        db = await self._conn()
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
