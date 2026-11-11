"""The markdown-folder recall source plugin (#719).

A recall source (``corvidae_cognition.recall.RecallSourceMixin``) that searches
a folder of markdown files for the turn the agent is about to take and hands
the hits to the funnel as framed CONTEXT. The index is a cheap-to-rebuild cache
over the folder; the files are the source of truth. Startup and idle reconcile
keep it current (the inotify directory watcher is a follow-up — it needs the
#723 constant fix, extraction from the ping plugin, and recursion).

The plugin owns the wiring (config, the embedding client + encoder identity,
the reconcile triggers); the index owns the SQLite state; the mixin owns the
gate, cooldown, timeout and fail-soft.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from pathlib import Path
from typing import Sequence

from corvidae.hooks import CorvidaePlugin, get_dependency, hookimpl
from corvidae.tuning import config_value
from corvidae_cognition.markdown_recall.index import Hit, MarkdownIndex, _EncoderIdentity
from corvidae_cognition.recall import RecallContext, RecallSourceMixin

logger = logging.getLogger("corvidae.markdown_recall")

DEFAULT_ROOT = "state/markdown_recall"
DEFAULT_DB_PATH = "state/markdown_recall/index.db"
DEFAULT_MAX_HITS = 5
# How quiet the system must be before the idle reconcile runs.
IDLE_QUIET_SECONDS = 30.0


class MarkdownRecallPlugin(RecallSourceMixin, CorvidaePlugin):
    """Recalls from a folder of markdown files on every plain user turn.

    The folder is untrusted text under the funnel's framing (PRD principle 9);
    the funnel is the only route its content reaches the window.
    """

    # Hard deps: the embedding client (llm) and the idle tracker (idle). The
    # funnel is an optional dep declared by the mixin (a source without it does
    # nothing, logged at DEBUG, not a stray hook).
    depends_on = frozenset({"llm", "idle"})
    # The funnel's frame label and the funnel.budgets key.
    source_name = "markdown_recall"
    # The agent.yaml block: root, db_path, max_hits, and the mixin's
    # cooldown/timeout.
    config_key = "markdown_recall"

    def __init__(self) -> None:
        self._pm = None
        self._index: MarkdownIndex | None = None
        self._root: Path | None = None
        self._max_hits = DEFAULT_MAX_HITS
        self._startup_task: asyncio.Task | None = None

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)  # stores self.pm and self.config
        self._pm = pm
        # The recall gate (cooldown/timeout) is always initialized, even
        # when the plugin is inert: the mixin's on_funnel_ingest runs on
        # every user turn and raises if _recall_gate is None (#739). An
        # inert source's search() returns None, so the gate is never
        # exercised.
        self.init_recall(config)
        # Inert when the operator has not configured a markdown_recall
        # block: no index, no reconcile, no search. The plugin is loaded
        # by entry point in every cognition install; the config block is
        # the opt-in.
        if config_value(config, "markdown_recall", None) is None:
            logger.debug("markdown_recall: no config block; plugin is inert")
            return
        # Multi-root: accept roots (list) or root (single, backward compat).
        roots_raw = config_value(config, "markdown_recall.roots", None)
        if roots_raw is not None:
            if not isinstance(roots_raw, list) or not roots_raw:
                raise ValueError(
                    "Config error: markdown_recall.roots must be a non-empty list of strings"
                )
            if not all(isinstance(r, str) for r in roots_raw):
                raise ValueError(
                    "Config error: markdown_recall.roots must be a list of strings"
                )
            self._root = [Path(r) for r in roots_raw]
        else:
            self._root = Path(config_value(config, "markdown_recall.root", DEFAULT_ROOT))
        db_path = Path(config_value(config, "markdown_recall.db_path", DEFAULT_DB_PATH))
        raw_max = config_value(config, "markdown_recall.max_hits", DEFAULT_MAX_HITS)
        if type(raw_max) is not int or raw_max < 1:
            raise ValueError(
                f"Config error: markdown_recall.max_hits must be a positive integer, "
                f"got {raw_max!r} — check markdown_recall.max_hits in agent.yaml"
            )
        self._max_hits = raw_max
        self._index = MarkdownIndex(
            root=self._root,
            db_path=db_path,
            get_client=self._embedding_client,
            get_identity=self._encoder_identity,
        )

    @hookimpl
    async def on_start(self, config: dict) -> None:
        """Build the index at startup and re-sync it when the system is quiet."""
        if self._index is None:
            return
        # The first build embeds every chunk over HTTP, so run it in the
        # background; a search before it finishes resolves to None (the index
        # is empty), never a block.
        import asyncio
        self._startup_task = asyncio.ensure_future(self._reconcile())
        # The idle tracker is a kernel component; if it is absent (a hand-built
        # test manager) the startup reconcile still ran, so the source works —
        # it just won't pick up out-of-band edits until the next startup.
        idle = self.pm.get_plugin("idle")
        if idle is None:
            logger.debug(
                "markdown recall: no idle tracker registered; out-of-band file "
                "changes are picked up at the next startup only"
            )
            return
        idle.watch_system(
            IDLE_QUIET_SECONDS, self._reconcile, name="markdown_recall.reconcile"
        )

    @hookimpl
    async def on_stop(self) -> None:
        """Cancel the startup task and close the index connection so the
        aiosqlite worker thread can exit. Without this, threading._shutdown
        blocks at interpreter exit and the daemon hangs (#738)."""
        task = self._startup_task
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        if self._index is not None:
            await self._index.close()

    async def _reconcile(self) -> None:
        """Re-sync the index to the folder; the index serializes on its own lock."""
        if self._index is None:
            return
        try:
            await self._index.reconcile()
        except Exception:
            # A failed reconcile keeps the old index; the next trigger retries.
            logger.warning("markdown recall: reconcile failed", exc_info=True)

    # -- the mixin contract -----------------------------------------------

    async def search(self, query: str, ctx: RecallContext) -> Sequence[Hit] | None:
        """The top hits for ``query``, in fused rank order (or None)."""
        if self._index is None:
            return None
        return await self._index.search(query, k=self._max_hits)

    def render(self, hit: Hit) -> str:
        """Render one hit as the line the funnel admits.

        ``hit.text`` already carries the chunker's ``[section_path]`` prefix, so
        it is the line as-is (adding the prefix again would double it). Stable
        for a given chunk's content: the funnel dedupes on whole text, so a line
        that changes between turns would defeat it. An edit changes the text and
        so the line — the new content re-surfaces.
        """
        return hit.text

    # -- the embedding stage ----------------------------------------------

    def _llm(self):
        """The LLMPlugin instance (profile-client access)."""
        from corvidae.llm_plugin import LLMPlugin
        return get_dependency(self.pm, "llm", LLMPlugin)

    def _embedding_client(self):
        """The ``llm`` stanza's ``embedding`` stage client."""
        return self._llm().get_client("embedding")

    def _encoder_identity(self) -> _EncoderIdentity | None:
        """The embedding profile's identity, or None when it is not configured.

        A None identity makes the index FTS5-only (no vector surface).
        """
        from corvidae.llm_plugin import parse_llm_config
        try:
            llm_config = parse_llm_config(self.config.get("llm"))
        except ValueError:
            # No llm stanza, or no resolvable default profile: no embedding
            # stage, so the index is FTS5-only.
            return None
        profile_name = llm_config.profile_for("embedding", "embedding")
        if not profile_name:
            return None
        profile = llm_config.profiles[profile_name]
        model = profile.get("model")
        dimensions = profile.get("dimensions")
        if model is None or dimensions is None:
            return None
        return _EncoderIdentity(
            model=str(model),
            dimensions=int(dimensions),
            document_prefix=str(profile.get("document_prefix", "")),
            query_prefix=str(profile.get("query_prefix", "")),
        )
