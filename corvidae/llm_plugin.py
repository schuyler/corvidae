"""LLMPlugin — owns LLM client lifecycle and configuration.

Parses every role under llm: in config, creates one LLMClient per role,
and manages their aiohttp session lifecycle. Other plugins access clients
via get_dependency(pm, "llm", LLMPlugin).get_client(role).

Config:
    llm:
      main:           # required
        base_url: ...
        model: ...
        api_key: ...        # optional
        extra_body: ...     # optional
        max_retries: 3      # optional
        retry_base_delay: 2.0
        retry_max_delay: 60.0
        model_not_loaded_wait: 60.0  # seconds to wait out an unloaded model
        timeout: ...        # optional
      background:     # optional — absent means use llm.main
        (same keys)
      embedding:      # optional — the /embeddings role (bootstrap-mapping §4.3)
        (same keys, plus required dimensions: int)

Roles are open-ended: any key under llm: becomes a client, and
get_client(role) falls back to main for unconfigured roles. Only llm.main
is hot-swapped by on_config_reload; other roles are restart-only for now.
"""
import asyncio
import logging
import math

import aiohttp

from corvidae.attribution import get_attribution
from corvidae.hooks import CorvidaePlugin, hookimpl
from corvidae.llm import LLMClient, LLMHTTPError, is_model_not_loaded

logger = logging.getLogger(__name__)

# Actionable hints for LLM HTTP failures. Unknown statuses still get the code.
_HTTP_STATUS_HINTS = {
    400: "the request was rejected",
    401: "the API key was rejected — check llm.main.api_key",
    403: "the API key is not allowed to use this model — check the provider account",
    404: "the endpoint or model was not found — check llm.main.base_url and llm.main.model",
    429: "the provider is rate-limiting or the account is out of quota",
}
# Provider error bodies are untrusted and can be long; show one short line.
_ERROR_BODY_PREVIEW_CHARS = 200


def _one_line(text: str, limit: int = _ERROR_BODY_PREVIEW_CHARS) -> str:
    """Collapse whitespace and truncate a provider error body for display."""
    collapsed = " ".join(text.split())
    return collapsed[:limit] + ("..." if len(collapsed) > limit else "")


class _HookObserver:
    """Bridges LLMClient observer callbacks to the plugin hook system.

    Defined here (not in corvidae.llm) so the LLM client module stays free
    of plugin-system knowledge. Carries the role the client was created
    for and snapshots the attribution contextvar at call time.
    """

    def __init__(self, pm, role: str, model: str) -> None:
        self._pm = pm
        self._role = role
        self._model = model

    async def request(
        self, model: str | None = None, **kwargs
    ) -> None:
        """Fire on_llm_request with role and attribution filled in."""
        await self._pm.ahook.on_llm_request(
            role=self._role,
            model=model or self._model,
            attribution=get_attribution(),
            **kwargs,
        )

    async def response(
        self, model: str | None = None, **kwargs
    ) -> None:
        """Fire on_llm_response with role and attribution filled in."""
        await self._pm.ahook.on_llm_response(
            role=self._role,
            model=model or self._model,
            attribution=get_attribution(),
            **kwargs,
        )


def _model_not_loaded_wait(cfg: dict, role: str) -> float:
    """Validate llm.<role>.model_not_loaded_wait: a finite, non-negative number of seconds."""
    value = cfg.get("model_not_loaded_wait", 60.0)
    # bool is an int subclass; reject it along with None and strings. inf
    # would wait forever and nan would silently disable the wait.
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(
            f"llm.{role}.model_not_loaded_wait must be a finite, non-negative number "
            f"of seconds, got {value!r}"
        )
    return float(value)


async def _close_after(client: LLMClient, delay: float = 5.0) -> None:
    """Close an old LLM client after a delay, allowing in-flight requests to finish."""
    await asyncio.sleep(delay)
    try:
        await client.stop()
    except Exception:
        logger.warning("failed to close old LLM client", exc_info=True)


class LLMPlugin(CorvidaePlugin):
    """Plugin that owns LLM client instances and their lifecycle.

    Attributes:
        embedding_dimensions: The vector dimension of the embedding role,
            or None when llm.embedding is not configured. The vec table
            schema (MemoryPlugin) needs a fixed dimension, so it is a
            required, startup-validated key.
    """

    depends_on = frozenset()

    def __init__(self, pm=None) -> None:
        if pm is not None:
            self.pm = pm
        # role -> LLMClient; populated in on_start from every key under llm:
        self._clients: dict[str, LLMClient | None] = {}
        # role -> config dict as parsed in on_init
        self._role_configs: dict[str, dict] = {}
        self.embedding_dimensions: int | None = None

    # ------------------------------------------------------------------
    # Legacy accessors — kept as properties over the role dict so existing
    # callers and tests (which read and assign these directly) keep working.
    # ------------------------------------------------------------------

    @property
    def main_client(self) -> LLMClient | None:
        """The llm.main client (None before on_start)."""
        return self._clients.get("main")

    @main_client.setter
    def main_client(self, client: LLMClient | None) -> None:
        self._clients["main"] = client

    @property
    def background_client(self) -> LLMClient | None:
        """The llm.background client, or None when not configured."""
        return self._clients.get("background")

    @background_client.setter
    def background_client(self, client: LLMClient | None) -> None:
        self._clients["background"] = client

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)
        # Every key under llm: is a role config (main, background,
        # embedding, and future roles alike).
        self._role_configs = dict(config.get("llm", {}) or {})

    @hookimpl
    async def on_start(self, config: dict) -> None:
        role_configs = self._role_configs
        if not role_configs:
            # Backward compat: read from config if on_init was not called
            role_configs = dict(config.get("llm", {}) or {})
        if role_configs.get("main") is None:
            raise KeyError("llm.main config is required")

        # Validate the embedding role before creating any client: the vec
        # table needs a fixed dimension, so a missing key is a config error.
        embedding_cfg = role_configs.get("embedding")
        if embedding_cfg is not None:
            if "dimensions" not in embedding_cfg:
                raise ValueError(
                    "llm.embedding requires a 'dimensions' key (the fixed "
                    "vector dimension of the embedding model)"
                )
            self.embedding_dimensions = int(embedding_cfg["dimensions"])

        # One client per configured role.
        for role, cfg in role_configs.items():
            if cfg is None:
                continue
            client = self._create_client(cfg, role=role)
            await client.start()
            self._clients[role] = client

    @hookimpl
    async def on_stop(self) -> None:
        for client in self._clients.values():
            if client is not None:
                await client.stop()

    @hookimpl
    async def on_config_reload(self, config: dict) -> None:
        """Re-read llm config and swap the main client if it changed.

        Only llm.main is hot-swapped: a new LLMClient is created, started,
        and swapped in, with the old one closed after a delay so in-flight
        requests complete. Other roles (background, embedding, ...) are
        restart-only for now — changes to them are logged and ignored.
        """
        llm_config = config.get("llm", {}) or {}
        new_main_config = llm_config.get("main")
        if new_main_config is None:
            # Validation should have caught this; log and skip.
            logger.error("on_config_reload: llm.main missing from new config, skipping LLM swap")
            return

        # Non-main role changes are restart-only; surface them honestly.
        for role, cfg in llm_config.items():
            if role != "main" and cfg != self._role_configs.get(role):
                logger.warning(
                    "on_config_reload: llm.%s changed; role reconfiguration "
                    "is restart-only, ignoring", role,
                )

        # Compare new config with stored config; skip swap if unchanged.
        if new_main_config == self._role_configs.get("main"):
            return

        logger.info("on_config_reload: llm.main changed, creating new LLM client")

        # Create and start new client before swapping so that an error during
        # start does not leave the plugin with a None client.
        new_client = self._create_client(new_main_config, role="main")
        await new_client.start()

        # Swap the reference and store the updated config.
        old_client = self._clients.get("main")
        self._clients["main"] = new_client
        self._role_configs["main"] = new_main_config

        # Schedule deferred closure of the old client so in-flight requests finish.
        if old_client is not None:
            asyncio.create_task(_close_after(old_client, delay=5.0))

    @hookimpl(trylast=True)
    async def resolve_llm_error(self, channel, error: Exception) -> str | None:
        """Explain LLM failures instead of falling back to the generic text.

        Covers an unloaded backend model (a llama-server router with
        --no-models-autoload rejects requests for an unloaded model, GitHub
        issue #55), HTTP rejections that name the status and any actionable
        cause (bad key, quota, wrong endpoint), and connection/timeout
        failures that point at the configured endpoint.

        trylast so another plugin's message takes precedence, and defers with
        None for errors this plugin cannot explain.
        """
        # Unloaded model first: its 400 body also matches the HTTP branch.
        if is_model_not_loaded(error):
            return (
                f"The model `{error.model}` isn't loaded on the LLM server "
                "(another model may be swapped in). Load it and send your "
                "message again."
            )
        # Provider returned a non-2xx response: name the status and hint.
        if isinstance(error, LLMHTTPError):
            hint = _HTTP_STATUS_HINTS.get(
                error.status, "the provider rejected the request"
            )
            message = f"LLM request failed with HTTP {error.status}: {hint}."
            if error.body:
                message += f" Provider said: {_one_line(error.body)}"
            return message
        # Never reached the provider: point at the endpoint configuration.
        if isinstance(error, asyncio.TimeoutError):
            return (
                "The LLM request timed out. Check llm.main.timeout and the "
                "endpoint's load; the full error is in the log."
            )
        if isinstance(error, aiohttp.ClientConnectionError):
            return (
                "Could not reach the LLM endpoint — check llm.main.base_url "
                "and that the server is running. The full error is in the log."
            )
        return None

    def get_client(self, role: str = "main") -> LLMClient:
        """Return the client for the given role.

        Unconfigured and unknown roles fall back to main
        (bootstrap-mapping §4.3).
        """
        return self._clients.get(role) or self._clients.get("main")

    def _create_client(self, cfg: dict, role: str = "main") -> LLMClient:
        """Construct an LLMClient from config and inject its hook observer.

        The observer bridges client callbacks to on_llm_request/on_llm_response
        with the given role. Injection is skipped when pm is unavailable
        (e.g. a plugin constructed bare in tests). Role-specific keys that
        LLMClient does not take (embedding's ``dimensions``) are consumed
        here, not passed through.
        """
        client = LLMClient(
            base_url=cfg["base_url"],
            model=cfg["model"],
            api_key=cfg.get("api_key"),
            extra_body=cfg.get("extra_body"),
            max_retries=cfg.get("max_retries", 3),
            retry_base_delay=cfg.get("retry_base_delay", 2.0),
            retry_max_delay=cfg.get("retry_max_delay", 60.0),
            timeout=cfg.get("timeout"),
            document_prefix=cfg.get("document_prefix", ""),
            query_prefix=cfg.get("query_prefix", ""),
            model_not_loaded_wait=_model_not_loaded_wait(cfg, role),
        )
        pm = getattr(self, "pm", None)
        if pm is not None:
            client.observer = _HookObserver(pm, role, cfg["model"])
        return client
