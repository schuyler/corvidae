"""LLMPlugin — owns LLM client lifecycle and configuration.

Parses every profile under llm: in config, creates one LLMClient per profile,
and manages their aiohttp session lifecycle. Other plugins access clients
via get_dependency(pm, "llm", LLMPlugin).get_client(stage), where stage is a
model-calling stage the plugin registered through register_llm_stages (None
for main turns).

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
        raw_reasoning: false  # endpoint returns <think>…</think> inline in content
        reasoning_reopened: truncate  # truncate | rerun: a reply that reopens reasoning after its answer
        timeout: ...        # optional
      embedding:      # optional — the /embeddings profile (bootstrap-mapping §4.3)
        (same keys, plus required dimensions: int)

Profiles are open-ended: any key under llm: becomes a client. A stage
resolves to its kind's default profile (DEFAULT_PROFILES): chat stages to
main, embedding stages to embedding. Only llm.main is hot-swapped by
on_config_reload; other profiles are restart-only for now.
"""
import asyncio
import logging
import math

import aiohttp

from corvidae.attribution import get_attribution
from corvidae.envref import EnvRefError, expand_env_refs
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
    of plugin-system knowledge. Carries the profile the client was created
    for and snapshots the attribution contextvar at call time.
    """

    def __init__(self, pm, profile: str, model: str) -> None:
        self._pm = pm
        self._profile = profile
        self._model = model

    async def request(
        self, model: str | None = None, **kwargs
    ) -> None:
        """Fire on_llm_request with profile and attribution filled in."""
        await self._pm.ahook.on_llm_request(
            profile=self._profile,
            model=model or self._model,
            attribution=get_attribution(),
            **kwargs,
        )

    async def response(
        self, model: str | None = None, **kwargs
    ) -> None:
        """Fire on_llm_response with profile and attribution filled in."""
        await self._pm.ahook.on_llm_response(
            profile=self._profile,
            model=model or self._model,
            attribution=get_attribution(),
            **kwargs,
        )


def _model_not_loaded_wait(cfg: dict, profile: str) -> float:
    """Validate llm.<profile>.model_not_loaded_wait: a finite, non-negative number of seconds."""
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
            f"llm.{profile}.model_not_loaded_wait must be a finite, non-negative number "
            f"of seconds, got {value!r}"
        )
    return float(value)


def _raw_reasoning(cfg: dict, profile: str) -> bool:
    """Validate llm.<profile>.raw_reasoning: a boolean, default false."""
    value = cfg.get("raw_reasoning", False)
    # A string such as "false" would otherwise read as true.
    if not isinstance(value, bool):
        raise ValueError(
            f"llm.{profile}.raw_reasoning must be true or false, got {value!r}"
        )
    return value


def _reasoning_reopened(cfg: dict, profile: str) -> str:
    """Validate llm.<profile>.reasoning_reopened: "truncate" (default) or "rerun".

    An invalid value never fails startup: it is logged and read as truncate.
    """
    value = cfg.get("reasoning_reopened", "truncate")
    # Only the two known strings pass; anything else (incl. non-strings) falls back.
    if value not in ("truncate", "rerun") or not isinstance(value, str):
        logger.warning(
            "llm.%s.reasoning_reopened is %r; accepted values are "
            "truncate, rerun; falling back to truncate — fix the value in "
            "agent.yaml and reload or restart",
            profile, value,
        )
        return "truncate"
    return value


async def _close_after(client: LLMClient, delay: float = 5.0) -> None:
    """Close an old LLM client after a delay, allowing in-flight requests to finish."""
    await asyncio.sleep(delay)
    try:
        await client.stop()
    except Exception:
        logger.warning("failed to close old LLM client", exc_info=True)


#: The profile a stage of each kind resolves to.
DEFAULT_PROFILES: dict[str, str] = {"chat": "main", "embedding": "embedding"}


class LLMPlugin(CorvidaePlugin):
    """Plugin that owns LLM client instances and their lifecycle.

    Attributes:
        embedding_dimensions: The vector dimension of the embedding profile,
            or None when llm.embedding is not configured. The vec table
            schema (MemoryPlugin) needs a fixed dimension, so it is a
            required, startup-validated key.
    """

    depends_on = frozenset()

    def __init__(self, pm=None) -> None:
        if pm is not None:
            self.pm = pm
        # profile -> LLMClient; populated in on_start from every key under llm:
        self._clients: dict[str, LLMClient | None] = {}
        # profile -> config dict as parsed in on_init
        self._profile_configs: dict[str, dict] = {}
        # stage -> kind; collected from register_llm_stages in on_start
        self._stages: dict[str, str] = {}
        self.embedding_dimensions: int | None = None

    # ------------------------------------------------------------------
    # Legacy accessors — kept as properties over the profile dict so existing
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
        # Every key under llm: is a profile config (main, embedding, and
        # future profiles alike).
        self._profile_configs = dict(config.get("llm", {}) or {})

    @hookimpl
    async def on_start(self, config: dict) -> None:
        # Collect stages first so the table doesn't depend on a client starting.
        if getattr(self, "pm", None) is not None:
            self.pm.hook.register_llm_stages(stages=self._stages)

        profile_configs = self._profile_configs
        if not profile_configs:
            # Backward compat: read from config if on_init was not called
            profile_configs = dict(config.get("llm", {}) or {})
        if profile_configs.get("main") is None:
            raise KeyError("llm.main config is required")

        # Validate the embedding profile before creating any client: the vec
        # table needs a fixed dimension, so a missing key is a config error.
        embedding_cfg = profile_configs.get("embedding")
        if embedding_cfg is not None:
            if "dimensions" not in embedding_cfg:
                raise ValueError(
                    "llm.embedding requires a 'dimensions' key (the fixed "
                    "vector dimension of the embedding model)"
                )
            self.embedding_dimensions = int(embedding_cfg["dimensions"])

        # One client per configured profile.
        for profile, cfg in profile_configs.items():
            if cfg is None:
                continue
            client = self._create_client(cfg, profile=profile)
            await client.start()
            self._clients[profile] = client

        # R5: where each stage's traffic goes, in one line.
        logger.info(
            "llm stages resolved: %s",
            ", ".join(
                f"{stage}→{self.resolve_profile(stage) or '(none)'}"
                for stage in sorted(self._stages)
            ),
        )

    @hookimpl
    async def on_stop(self) -> None:
        for client in self._clients.values():
            if client is not None:
                await client.stop()

    @hookimpl(tryfirst=True)
    async def on_config_reload(self, config: dict) -> None:
        """Re-read llm config and swap the main client if it changed.

        Only llm.main is hot-swapped: a new LLMClient is created, started,
        and swapped in, with the old one closed after a delay so in-flight
        requests complete. Other profiles (embedding, ...) are
        restart-only for now — changes to them are logged and ignored.

        ``tryfirst`` because Agent.on_config_reload re-borrows the client
        from this plugin: the swap has to land before anything reads it, and
        ConfigWatcher dispatches in registration order, which is not
        deterministic (#382).
        """
        llm_config = config.get("llm", {}) or {}
        new_main_config = llm_config.get("main")
        if new_main_config is None:
            # Validation should have caught this; log and skip.
            logger.error("on_config_reload: llm.main missing from new config, skipping LLM swap")
            return

        # Non-main profile changes are restart-only; surface them honestly.
        for profile, cfg in llm_config.items():
            if profile != "main" and cfg != self._profile_configs.get(profile):
                logger.warning(
                    "on_config_reload: llm.%s changed; profile reconfiguration "
                    "is restart-only, ignoring", profile,
                )

        # Compare new config with stored config; skip swap if unchanged.
        if new_main_config == self._profile_configs.get("main"):
            return

        logger.info("on_config_reload: llm.main changed, creating new LLM client")

        # Create and start new client before swapping so that an error during
        # start does not leave the plugin with a None client.
        new_client = self._create_client(new_main_config, profile="main")
        await new_client.start()

        # Swap the reference and store the updated config.
        old_client = self._clients.get("main")
        self._clients["main"] = new_client
        self._profile_configs["main"] = new_main_config

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

    def resolve_profile(self, stage: str | None = None) -> str | None:
        """The configured profile ``stage`` resolves to, or None when its
        default profile isn't configured.

        ``stage=None`` is a main turn. A registered stage resolves to its
        kind's default profile; any other name raises ValueError.
        """
        if stage is None:
            profile = DEFAULT_PROFILES["chat"]
        elif stage in self._stages:
            profile = DEFAULT_PROFILES[self._stages[stage]]
        else:
            raise ValueError(
                f"unregistered llm stage {stage!r}; registered stages: "
                f"{', '.join(sorted(self._stages)) or '(none)'}"
            )
        return profile if self._clients.get(profile) is not None else None

    def get_client(self, stage: str | None = None) -> LLMClient | None:
        """Return the client for ``stage`` (None for main turns).

        None when the stage's default profile isn't configured (including
        main turns before on_start). An unregistered stage raises ValueError.
        """
        profile = self.resolve_profile(stage)
        return self._clients[profile] if profile is not None else None

    def _create_client(self, cfg: dict, profile: str = "main") -> LLMClient:
        """Construct an LLMClient from config and inject its hook observer.

        The observer bridges client callbacks to on_llm_request/on_llm_response
        with the given profile. Injection is skipped when pm is unavailable
        (e.g. a plugin constructed bare in tests). Profile-specific keys that
        LLMClient does not take (embedding's ``dimensions``) are consumed
        here, not passed through.
        """
        # Resolve ${VAR} in api_key from the environment. The resolved key
        # lives only on the client, never in _profile_configs, so reload
        # comparison and logs see the raw reference.
        api_key = cfg.get("api_key")
        if isinstance(api_key, str):
            try:
                api_key = expand_env_refs(api_key)
            except EnvRefError as exc:
                where = f"llm.{profile}.api_key"
                if exc.reason == "multiline":
                    raise ValueError(
                        f"{where} references ${{{exc.var}}}, whose value "
                        f"contains a newline or NUL (often a trailing "
                        f"newline); fix {exc.var} in the daemon's environment "
                        "and restart."
                    ) from None
                raise ValueError(
                    f"{where} references ${{{exc.var}}}, which is not set in "
                    f"the daemon's environment; set {exc.var} in the "
                    "environment corvidae runs in and restart."
                ) from None
        client = LLMClient(
            base_url=cfg["base_url"],
            model=cfg["model"],
            api_key=api_key,
            extra_body=cfg.get("extra_body"),
            max_retries=cfg.get("max_retries", 3),
            retry_base_delay=cfg.get("retry_base_delay", 2.0),
            retry_max_delay=cfg.get("retry_max_delay", 60.0),
            timeout=cfg.get("timeout"),
            document_prefix=cfg.get("document_prefix", ""),
            query_prefix=cfg.get("query_prefix", ""),
            model_not_loaded_wait=_model_not_loaded_wait(cfg, profile),
            raw_reasoning=_raw_reasoning(cfg, profile),
            reasoning_reopened=_reasoning_reopened(cfg, profile),
        )
        pm = getattr(self, "pm", None)
        if pm is not None:
            client.observer = _HookObserver(pm, profile, cfg["model"])
        return client
