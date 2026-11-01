"""Operator CLI for memory record stamps (issue #299).

Forms:
    corvidae memory --db sessions.db stamps
    corvidae memory --db sessions.db reconsolidate STAMP --config agent.yaml
    corvidae memory --db sessions.db reconsolidate STAMP --dry-run

``stamps`` lists records grouped by how they were made (instruction hash,
mode, model), plus a ``direct`` group (goal retirement; no consolidation call
produced them) and an ``unknown`` group (written before stamping).
``reconsolidate`` rebuilds a stamp's records from ``message_log`` with the
current standalone instructions: the new record supersedes the old one (demote,
never delete) and inherits its retention state and retrieval statistics.

Second-process discipline: ``stamps`` and ``--dry-run`` open the database
read-only and never migrate it. A live run asserts WAL, uses its own
``MemoryPlugin`` write connection, never runs ``memory.on_start`` (no
retention, no backfill, no idle watches), drops ``llm.embedding`` so it can
neither create ``memory_vec`` nor write ``embedding_meta``, never embeds (new
records are stored ``embedded=0`` for the daemon's retention backfill), and
loads sqlite-vec only to delete a superseded record's vector. Logs go to
stderr, never the daemon's log file. This is an operator tool, not an agent
tool: nothing here is registered with the tool registry.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.metadata
import logging
import re
import sqlite3
import sys
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite
import click
import yaml

from corvidae.hooks import create_plugin_manager
from corvidae.llm_plugin import LLMPlugin
from corvidae.persistence import PersistencePlugin
from corvidae_cognition.appraisal import AppraisalPlugin
from corvidae_cognition.memory import (
    MemoryPlugin,
    ReconsolidateSkipped,
    prompt_hash,
    resolve_consolidation_prompts,
)

logger = logging.getLogger("corvidae.memory_command")

# A stamp argument is a hex prefix of a hash (lower-cased before matching).
_HEX_PREFIX = re.compile(r"^[0-9a-f]{1,64}$")
# Matches the daemon's write busy timeout.
_BUSY_TIMEOUT_MS = 5000


# ---------------------------------------------------------------------------
# Schema probe and selection (read-only; tolerant of pre-change databases)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemorySchema:
    """Which of the stamp columns a ``memory`` table has.

    Checked separately: the three ALTERs are separate statements, so an
    interrupted migration can leave only some of them.
    """

    has_prompt_hash: bool
    has_mode: bool
    has_model: bool


async def probe_memory_schema(db: aiosqlite.Connection) -> MemorySchema | None:
    """The stamp columns of ``memory``, or None when there is no such table."""
    async with db.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'memory'"
    ) as cursor:
        if await cursor.fetchone() is None:
            return None
    async with db.execute("PRAGMA table_info(memory)") as cursor:
        columns = {row[1] for row in await cursor.fetchall()}
    return MemorySchema(
        "prompt_hash" in columns, "mode" in columns, "model" in columns
    )


def _col(schema: MemorySchema, name: str) -> str:
    """The SQL for a stamp column: its name, or the literal NULL when absent.

    ``name`` is always a code constant, never user text.
    """
    present = {
        "prompt_hash": schema.has_prompt_hash,
        "mode": schema.has_mode,
        "model": schema.has_model,
    }[name]
    return name if present else "NULL"


@dataclass
class Selection:
    """Records picked by a stamp: live ones to rebuild, plus skip counts."""

    # (memory id, channel id) of every live or demoted record, id order.
    selected: list[tuple[int, str]] = field(default_factory=list)
    # Records matching the stamp but not eligible: reason -> count.
    skipped: dict[str, int] = field(default_factory=dict)

    @property
    def matched(self) -> int:
        """Every record the stamp matched, selected or skipped."""
        return len(self.selected) + sum(self.skipped.values())


async def select_records(
    db: aiosqlite.Connection, schema: MemorySchema, stamp: str
) -> Selection:
    """Partition the records under ``stamp`` into selected and skipped (R8).

    ``stamp`` is ``unknown`` (no hash and no mode) or a hash prefix. Neither
    can match a ``direct`` row. Superseded and redacted records are counted
    and skipped; demoted (``indexed=0``) ones are kept.
    """
    if stamp == "direct":
        raise ValueError("the 'direct' group is never re-consolidated")
    params: tuple = ()
    if stamp == "unknown":
        # Each term is dropped when its column is absent: on a pre-change
        # table every row is unknown.
        terms = [
            f"{name} IS NULL"
            for name, present in (
                ("prompt_hash", schema.has_prompt_hash), ("mode", schema.has_mode)
            )
            if present
        ]
        where = " AND ".join(terms) or "1 = 1"
    elif schema.has_prompt_hash:
        # Bound parameters, never LIKE: no user text is a pattern.
        where = "substr(prompt_hash, 1, ?) = ?"
        params = (len(stamp), stamp)
    else:
        # No hash column, so no hash can match.
        return Selection()
    selection = Selection()
    async with db.execute(
        "SELECT id, channel_id, superseded_by, redacted FROM memory "
        f"WHERE {where} ORDER BY id", params,
    ) as cursor:
        rows = await cursor.fetchall()
    for memory_id, channel_id, superseded_by, redacted in rows:
        if superseded_by is not None:
            selection.skipped["superseded"] = selection.skipped.get("superseded", 0) + 1
        elif redacted:
            selection.skipped["redacted"] = selection.skipped.get("redacted", 0) + 1
        else:
            selection.selected.append((memory_id, channel_id))
    return selection


async def _resolve_stamp(
    db: aiosqlite.Connection, schema: MemorySchema, stamp: str, db_path: str
) -> str:
    """The full hash a stamp prefix names, or ``unknown`` unchanged.

    Raises ClickException for no match or an ambiguous prefix.
    """
    if stamp == "unknown":
        return stamp
    hashes: list[str] = []
    if schema.has_prompt_hash:
        async with db.execute(
            "SELECT DISTINCT prompt_hash FROM memory "
            "WHERE substr(prompt_hash, 1, ?) = ? ORDER BY 1", (len(stamp), stamp),
        ) as cursor:
            hashes = [row[0] for row in await cursor.fetchall()]
    if not hashes:
        raise click.ClickException(_no_match(stamp, db_path))
    if len(hashes) > 1:
        shown = ", ".join(h[: len(stamp) + 4] + "…" for h in hashes)
        raise click.ClickException(
            f"Stamp prefix '{stamp}' matches {len(hashes)} stamps ({shown}); "
            "give more characters"
        )
    return hashes[0]


def _no_match(stamp: str, db_path: str) -> str:
    """The message for a stamp that matches no record."""
    return (
        f"No records under stamp '{stamp}' in {db_path}; "
        f"run 'corvidae memory --db {db_path} stamps' to list stamps"
    )


# ---------------------------------------------------------------------------
# Config loading (a small copy of Runtime.start steps 1-3)
# ---------------------------------------------------------------------------


def _config_error(
    detail: str, config_path: str, cause: BaseException | None = None
) -> click.ClickException:
    """A human-facing config failure naming the file.

    When ``cause`` is given, the original exception is logged at DEBUG with
    its traceback (the caller re-raises ``from`` it). That record is visible
    only once logging is configured (``logging.level: DEBUG``); errors raised
    before then carry their detail in the Config error text itself.
    """
    if cause is not None:
        # DEBUG: stderr is the operator's terminal, and a traceback there
        # would bury the one-line Config error.
        logger.debug(
            "config error in %s: %s (%s)", config_path, detail,
            type(cause).__name__, exc_info=cause,
        )
    return click.ClickException(f"Config error: {detail} — check {config_path}")


def _read_yaml(config_path: str) -> dict:
    """The YAML mapping at ``config_path``; ClickException when unusable."""
    try:
        with open(config_path, encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise _config_error(
            getattr(exc, "strerror", None) or str(exc), config_path, exc
        ) from exc
    except UnicodeDecodeError as exc:
        # A file saved in another encoding (e.g. Latin-1) fails while PyYAML
        # reads it, outside YAMLError; report it, not a traceback.
        raise _config_error(
            f"file is not valid UTF-8 ({exc.reason} at byte {exc.start}); "
            "re-save it as UTF-8", config_path, exc,
        ) from exc
    except RecursionError as exc:
        # Pathologically deep nesting exhausts PyYAML's recursive composer.
        raise _config_error("YAML is nested too deeply", config_path, exc) from exc
    if not isinstance(config, dict):
        raise _config_error("file is empty or not a mapping", config_path)
    return config


def _memory_section(config: dict, config_path: str) -> dict:
    """The ``memory`` mapping of a config (empty when absent)."""
    section = config.get("memory") or {}
    if not isinstance(section, dict):
        raise _config_error("memory must be a mapping", config_path)
    return section


def _current_prompt_hash(config: dict, config_path: str) -> str:
    """Hash of the config's standalone consolidation prompt (``H_cur``)."""
    try:
        standalone, _fork = resolve_consolidation_prompts(
            _memory_section(config, config_path)
        )
    except ValueError as exc:
        raise _config_error(str(exc), config_path, exc) from exc
    return prompt_hash(standalone)


# shortcut: duplicated config load; extract a shared loader from Runtime.start if a third offline command needs it.
def load_offline_config(config_path: str) -> dict:
    """Load the daemon config for an offline run, validated for this command.

    Drops ``llm.embedding`` (the command never embeds, and must not create
    ``memory_vec`` or write ``embedding_meta``) and checks every non-null
    ``llm.<role>`` has ``base_url`` and ``model``. A role set to null is
    skipped, as the daemon does.
    """
    config = dict(_read_yaml(config_path))
    config["_base_dir"] = Path(config_path).parent
    config["_config_path"] = Path(config_path)
    llm = config.get("llm") or {}
    if not isinstance(llm, dict):
        raise _config_error("llm must be a mapping of roles", config_path)
    llm = {role: cfg for role, cfg in llm.items() if role != "embedding"}
    for role, cfg in llm.items():
        if cfg is None:
            continue
        for key in ("base_url", "model"):
            if not isinstance(cfg, dict) or key not in cfg:
                raise _config_error(
                    f"missing required key llm.{role}.{key}", config_path
                )
    config["llm"] = llm
    _current_prompt_hash(config, config_path)
    return config


# ---------------------------------------------------------------------------
# Re-consolidation core (separate from the click glue so tests can drive it)
# ---------------------------------------------------------------------------


@dataclass
class ReconsolidateReport:
    """What a re-consolidation run did."""

    # Skip counts by reason: superseded, redacted, direct, no dialog.
    skipped: dict[str, int] = field(default_factory=dict)
    # (old id, channel, new id) of every rebuilt record.
    done: list[tuple[int, str, int]] = field(default_factory=list)
    # (id, channel, exception type name) of every failed record.
    failed: list[tuple[int, str, str]] = field(default_factory=list)
    # Per-record output lines, in order.
    lines: list[str] = field(default_factory=list)


async def reconsolidate(
    memory: MemoryPlugin, selection: Selection
) -> ReconsolidateReport:
    """Re-consolidate each selected record; a per-record failure is skipped.

    Failures are logged with their traceback and reported by id, channel and
    exception type; the run continues with the next record (R9).
    """
    report = ReconsolidateReport(
        skipped={k: v for k, v in selection.skipped.items() if v}
    )
    for memory_id, channel_id in selection.selected:
        try:
            new_id = await memory.reconsolidate_record(memory_id)
        except ReconsolidateSkipped as exc:
            # Changed since selection (or direct): counted, no LLM call made.
            report.skipped[exc.reason] = report.skipped.get(exc.reason, 0) + 1
            report.lines.append(
                f"memory #{memory_id} ({channel_id}) skipped ({exc.reason})"
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "re-consolidation failed for memory %d on %s",
                memory_id, channel_id, exc_info=True,
            )
            report.failed.append((memory_id, channel_id, type(exc).__name__))
            report.lines.append(
                f"memory #{memory_id} ({channel_id}) failed ({type(exc).__name__})"
            )
        else:
            report.done.append((memory_id, channel_id, new_id))
            report.lines.append(f"memory #{memory_id} ({channel_id}) -> #{new_id}")
    return report


def _skip_clause(skipped: dict[str, int], prefix: str) -> str:
    """'<prefix> 1 superseded, 1 redacted' for the non-zero skip counts, or ''."""
    parts = [f"{count} {reason}" for reason, count in skipped.items() if count]
    return f"{prefix} {', '.join(parts)}" if parts else ""


# ---------------------------------------------------------------------------
# Offline wiring
# ---------------------------------------------------------------------------


def _log_to_stderr(config: dict, config_path: str) -> None:
    """Send logs to stderr at ``logging.level`` (default INFO).

    ``logging.file`` is ignored on purpose, so this process never writes into
    the daemon's log file. A single handler on the root logger; the corvidae
    loggers inherit it.
    """
    # A non-mapping logging block is a config error, not an AttributeError.
    logging_cfg = config.get("logging") or {}
    if not isinstance(logging_cfg, dict):
        raise _config_error(
            f"logging must be a mapping (e.g. {{level: INFO}}); got {logging_cfg!r}",
            config_path,
        )
    level_name = str(logging_cfg.get("level", "INFO")).upper()
    level = logging.getLevelName(level_name)
    if not isinstance(level, int):
        raise _config_error(
            f"logging.level must be one of DEBUG, INFO, WARNING, ERROR, "
            f"CRITICAL; got {level_name!r}", config_path,
        )
    handler = logging.StreamHandler(sys.stderr)
    handler._corvidae_memory_cli = True  # type: ignore[attr-defined]
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"
    ))
    root = logging.getLogger()
    # Replace a handler from an earlier call instead of stacking duplicates.
    for old in [h for h in root.handlers if getattr(h, "_corvidae_memory_cli", False)]:
        root.removeHandler(old)
    root.addHandler(handler)
    root.setLevel(level)
    # The corvidae logger may carry the daemon's own handlers if configured
    # earlier in this process; let it propagate to the stderr handler.
    logging.getLogger("corvidae").setLevel(level)


def _disabled_plugins(config: dict, config_path: str) -> list[str]:
    """``plugins.disabled`` as a list of names; anything else is a config error."""
    plugins = config.get("plugins")
    disabled = plugins.get("disabled") if isinstance(plugins, dict) else None
    if plugins is not None and not isinstance(plugins, dict):
        raise _config_error(
            f"plugins must be a mapping with a disabled list; got {plugins!r}",
            config_path,
        )
    if disabled is None:
        return []
    if not isinstance(disabled, list) or not all(isinstance(n, str) for n in disabled):
        raise _config_error(
            f"plugins.disabled must be a list of plugin names; got {disabled!r}",
            config_path,
        )
    return disabled


def _load_usage_log_plugin() -> object | None:
    """The usage-log plugin, loaded through its ``corvidae`` entry point.

    None (with a WARNING) when the entry point is missing: the run goes on
    but its LLM calls are not metered.
    """
    for entry in importlib.metadata.entry_points(group="corvidae"):
        if entry.name == "usage_log":
            return entry.load()()
    logger.warning(
        "no 'usage_log' entry point in the corvidae group; this run's LLM "
        "calls won't be metered in usage_log"
    )
    return None


@contextlib.asynccontextmanager
async def _open_offline_memory(
    db_path: str, config_path: str
) -> AsyncIterator[MemoryPlugin]:
    """Start just enough plugins to run re-consolidation; yield the memory plugin.

    Loads the config, logs to stderr, asserts WAL, builds persistence (on an
    injected connection), llm, usage_log, memory and (unless disabled)
    appraisal. Never runs ``memory.on_start``.
    """
    config = load_offline_config(config_path)
    _log_to_stderr(config, config_path)

    db = await aiosqlite.connect(db_path)
    memory: MemoryPlugin | None = None
    llm: LLMPlugin | None = None
    appraisal: AppraisalPlugin | None = None
    try:
        await db.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
        # WAL is asserted inline so the message is this command's own.
        async with db.execute("PRAGMA journal_mode") as cursor:
            row = await cursor.fetchone()
        mode = str(row[0]).lower() if row else "unknown"
        if mode != "wal":
            raise click.ClickException(
                f"{db_path} is in '{mode}' journal mode; corvidae memory "
                "reconsolidate needs WAL, which the daemon sets when it starts. "
                "Start the daemon once on this database, then retry. "
                "Nothing was changed."
            )
        async with db.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'memory_vec'"
        ) as cursor:
            has_vec = await cursor.fetchone() is not None

        # Register the plugins; persistence gets the open connection.
        pm = create_plugin_manager()
        persistence = PersistencePlugin()
        persistence.db = db
        pm.register(persistence, name="persistence")
        llm = LLMPlugin()
        pm.register(llm, name="llm")
        usage_log = _load_usage_log_plugin()
        if usage_log is not None:
            pm.register(usage_log, name="usage_log")
        memory = MemoryPlugin()
        pm.register(memory, name="memory")
        disabled = _disabled_plugins(config, config_path)
        # Appraisal installs the daemon's importance prior on memory.
        if "appraisal" not in disabled:
            appraisal = AppraisalPlugin()
            pm.register(appraisal, name="appraisal")

        # on_init for everyone, on_start only where needed (not memory's).
        try:
            await pm.ahook.on_init(pm=pm, config=config)
            for plugin in (llm, usage_log, appraisal):
                if plugin is not None:
                    await plugin.on_start(config=config)
        except KeyError as exc:
            key = str(exc.args[0]).split()[0] if exc.args else "?"
            raise _config_error(
                f"missing required key {key}", config_path, exc
            ) from exc
        except Exception as exc:
            # Any other plugin init/start failure is a config problem too
            # (e.g. memory.retrieval: 5); CancelledError is not an Exception.
            raise _config_error(
                str(exc) or type(exc).__name__, config_path, exc
            ) from exc

        # Open the schema (idempotent column upgrade) and the write
        # connection; load sqlite-vec on it only when there is a vector table.
        memory._load_vec_on_write = has_vec
        await memory._ensure_schema()
        conn = await memory._write_conn()
        if has_vec and conn is not None:
            try:
                async with conn.execute("SELECT 1 FROM memory_vec LIMIT 0"):
                    pass
            except sqlite3.OperationalError as exc:
                if "no such module" not in str(exc):
                    raise
                raise click.ClickException(
                    f"{db_path} has memory_vec, which needs the sqlite-vec "
                    "extension to delete superseded vectors, and it could not "
                    "be loaded; no records were changed. Install the cognition "
                    "extra (uv sync --extra cognition), or run corvidae memory "
                    "in the daemon's environment."
                )
        yield memory
    finally:
        # Teardown in reverse order of startup.
        if memory is not None:
            await memory.on_stop()
        if appraisal is not None:
            await appraisal.on_stop()
        if llm is not None:
            await llm.on_stop()
        await db.close()


# ---------------------------------------------------------------------------
# Click glue
# ---------------------------------------------------------------------------


def _ro_uri(db_path: str) -> str:
    """A read-only SQLite URI safe for relative paths and special characters."""
    return Path(db_path).absolute().as_uri() + "?mode=ro"


@contextlib.asynccontextmanager
async def _read_only(db_path: str) -> AsyncIterator[aiosqlite.Connection]:
    """A read-only connection: a write through it is impossible."""
    db = await aiosqlite.connect(_ro_uri(db_path), uri=True)
    try:
        await db.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
        yield db
    finally:
        await db.close()


@click.group("memory")
@click.option("--db", "db_path", required=True, help="Path to sessions.db")
@click.pass_context
def memory_command(ctx: click.Context, db_path: str) -> None:
    """Operator tool for how memory records were made."""
    # Check the path first: aiosqlite would silently create an empty database.
    if not Path(db_path).is_file():
        raise click.ClickException(f"Database not found: {db_path}")
    ctx.ensure_object(dict)
    ctx.obj["db_path"] = db_path


def _fmt_date(timestamp: float | None) -> str:
    """UTC calendar date of an epoch timestamp."""
    if timestamp is None:
        return "?"
    return datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%d")


async def _stamp_lines(db: aiosqlite.Connection, db_path: str) -> list[str]:
    """One line per stamp group: hash groups, then direct, then unknown."""
    schema = await probe_memory_schema(db)
    if schema is None:
        raise click.ClickException(f"No memory table in {db_path}; nothing to list")
    hash_col, mode_col, model_col = (
        _col(schema, "prompt_hash"), _col(schema, "mode"), _col(schema, "model")
    )
    key = f"COALESCE({hash_col}, CASE WHEN {mode_col} = 'direct' THEN 'direct' END)"
    async with db.execute(
        f"SELECT {key}, COUNT(*), GROUP_CONCAT(DISTINCT channel_id), "
        f"GROUP_CONCAT(DISTINCT {mode_col}), GROUP_CONCAT(DISTINCT {model_col}), "
        f"MIN(created_at), MAX(created_at) FROM memory GROUP BY {key}"
    ) as cursor:
        groups = await cursor.fetchall()

    def order(group: tuple) -> tuple:
        """Hash groups first (by hash), then direct, then unknown."""
        label = group[0]
        return (2, "") if label is None else (1, "") if label == "direct" else (0, label)

    lines = []
    for label, count, channels, modes, models, first, last in sorted(groups, key=order):
        name = "unknown" if label is None else "direct" if label == "direct" else label[:12]
        lines.append(
            f"{name:<12}  {_join(modes):<10}  {_join(models):<12}  {count} records  "
            f"{_join(channels)}  {_fmt_date(first)}..{_fmt_date(last)}"
        )
    return lines


def _join(concat: str | None) -> str:
    """A sorted, comma-separated view of a GROUP_CONCAT value ('-' when NULL)."""
    return ", ".join(sorted(concat.split(","))) if concat else "-"


@memory_command.command("stamps")
@click.pass_context
def _stamps_cmd(ctx: click.Context) -> None:
    """List records grouped by the instructions that produced them."""
    db_path = ctx.obj["db_path"]

    async def _run() -> list[str]:
        async with _read_only(db_path) as db:
            return await _stamp_lines(db, db_path)

    for line in asyncio.run(_run()):
        click.echo(line)


def _validate_stamp(stamp: str, db_path: str) -> str:
    """The normalized STAMP, or a UsageError naming the rejected value."""
    value = stamp.lower()
    if value in ("unknown", "direct") or _HEX_PREFIX.match(value):
        return value
    raise click.UsageError(
        "STAMP must be 'unknown' or a hex prefix of a stamp hash (characters "
        f"0-9, a-f); got '{stamp}'. Run 'corvidae memory --db {db_path} stamps' "
        "to list stamps"
    )


def _refuse_current(resolved: str, h_cur: str, config_path: str) -> None:
    """Refuse a stamp that is the current standalone prompt's: nothing would change."""
    if resolved == h_cur:
        raise click.ClickException(
            f"Stamp {resolved[:12]} is the current memory.consolidation_prompt; "
            "its records are already under the current instructions and "
            "re-consolidating would change nothing. To rebuild them, change "
            f"memory.consolidation_prompt in {config_path} first. "
            "Nothing was changed."
        )


@memory_command.command("reconsolidate")
@click.argument("stamp")
@click.option("--config", "config_path", default=None,
              help="The daemon's agent.yaml (its LLM settings); required unless --dry-run")
@click.option("--dry-run", is_flag=True, default=False,
              help="Report what would be re-consolidated; change nothing")
@click.pass_context
def _reconsolidate_cmd(
    ctx: click.Context, stamp: str, config_path: str | None, dry_run: bool
) -> None:
    """Rebuild a stamp's records from message_log with the current instructions."""
    db_path = ctx.obj["db_path"]
    stamp = _validate_stamp(stamp, db_path)
    if stamp == "direct":
        raise click.ClickException(
            "Records in the 'direct' group weren't written by a consolidation "
            "call and are never re-consolidated; pick a stamp from "
            f"'corvidae memory --db {db_path} stamps'"
        )
    if config_path is None and not dry_run:
        raise click.UsageError(
            "reconsolidate needs --config <path to the daemon's agent.yaml> "
            "(its LLM settings); add --dry-run to preview without one"
        )

    # The current prompt's hash: from the config when one is given.
    h_cur: str | None = None
    if config_path is not None:
        raw = _read_yaml(config_path)
        h_cur = _current_prompt_hash(raw, config_path)
        if not dry_run:
            load_offline_config(config_path)

    async def _resolve() -> tuple[str, Selection]:
        """Probe, resolve, select and refuse: read-only, before any plugin starts."""
        async with _read_only(db_path) as db:
            schema = await probe_memory_schema(db)
            if schema is None:
                raise click.ClickException(
                    f"No memory table in {db_path}; nothing to re-consolidate"
                )
            resolved = await _resolve_stamp(db, schema, stamp, db_path)
            if h_cur is not None:
                _refuse_current(resolved, h_cur, config_path or "")
            # No matching rows at all gets the no-match message before any
            # plugin starts or migration commits.
            selection = await select_records(db, schema, resolved)
            if selection.matched == 0:
                raise click.ClickException(_no_match(stamp, db_path))
            return resolved, selection

    resolved, preview = asyncio.run(_resolve())

    if dry_run:
        _print_dry_run(preview, resolved)
        return
    assert config_path is not None
    code = asyncio.run(_live_run(db_path, config_path, stamp, resolved, h_cur or ""))
    if code:
        ctx.exit(code)


def _print_dry_run(selection: Selection, resolved: str) -> None:
    """Print what a live run would re-consolidate (nothing is changed)."""
    listed = ", ".join(f"#{i} ({ch})" for i, ch in selection.selected)
    head = (
        f"Would re-consolidate {len(selection.selected)} records "
        f"under stamp {resolved[:12]}"
    )
    skip = _skip_clause(selection.skipped, "would skip")
    click.echo(
        head + (f": {listed}" if listed else "")
        + (f"; {skip}" if skip else "") + ". Nothing was changed."
    )


async def _live_run(
    db_path: str, config_path: str, stamp: str, resolved: str, h_cur: str
) -> int:
    """Re-consolidate the selection with real clients; returns the exit code."""
    async with _open_offline_memory(db_path, config_path) as memory:
        db = memory._resolve_db()
        assert db is not None
        # Re-probe after the migration so the new columns exist.
        schema = await probe_memory_schema(db)
        assert schema is not None
        selection = await select_records(db, schema, resolved)
        if selection.matched == 0:
            raise click.ClickException(_no_match(stamp, db_path))
        if not selection.selected:
            click.echo(
                f"All {selection.matched} records under {stamp} are already "
                "superseded or redacted; nothing to do."
            )
            return 0
        report = await reconsolidate(memory, selection)
    # Output: one line per record, a totals line, and the retry line.
    for line in report.lines:
        click.echo(line)
    skip = _skip_clause(report.skipped, "skipped")
    click.echo(
        f"Re-consolidated {len(report.done)} records"
        + (f", {skip}" if skip else "")
        + (f", {len(report.failed)} failed" if report.failed else "") + "."
    )
    if report.failed:
        click.echo(
            f"Re-run 'corvidae memory --db {db_path} reconsolidate {resolved[:12]} "
            f"--config {config_path}' to retry the {len(report.failed)} failed "
            "records. The ones that succeeded are superseded and their "
            f"replacements carry the current stamp {h_cur[:12]}, so a re-run "
            f"selects only the records still under {resolved[:12]}."
        )
        return 1
    return 0
