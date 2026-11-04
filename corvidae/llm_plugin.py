"""LLMPlugin — owns LLM client lifecycle and configuration.

Parses the llm: section (parse_llm_config), creates one LLMClient per
profile, and manages their aiohttp session lifecycle. Other plugins access
clients via get_dependency(pm, "llm", LLMPlugin).get_client(stage), where
stage is a model-calling stage the plugin registered through
register_llm_stages (None for main turns).

Config:
    llm:
      main:           # a profile; the default chat profile unless defaults.chat says otherwise
        base_url: ...
        model: ...
        kind: chat          # optional: chat (default) | embedding
        extends: other      # optional: deep overlay of another profile
        api_key: ...        # optional
        extra_body: ...     # optional
        max_retries: 3      # optional
        retry_base_delay: 2.0
        retry_max_delay: 60.0
        model_not_loaded_wait: 60.0  # seconds to wait out an unloaded model
        raw_reasoning: false  # endpoint returns <think>…</think> inline in content
        reasoning_reopened: truncate  # truncate | rerun: a reply that reopens reasoning after its answer
        timeout: ...        # optional
      embedding:      # the /embeddings profile (bootstrap-mapping §4.3)
        (same keys, plus kind: embedding and required dimensions: int)
      defaults:       # optional: kind -> default profile (chat -> main, embedding -> embedding)
        chat: main
      stages:         # optional: stage -> profile
        compaction: forks

Every key under llm: except defaults and stages is a profile. A stage
resolves to the profile llm.stages names for it, else to its kind's default
profile. Everything under llm: is restart-only; a reload that changes it logs
a warning and keeps the running clients.
"""
import asyncio
import copy
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass

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
            "agent.yaml and restart",
            profile, value,
        )
        return "truncate"
    return value


#: The kinds a profile may have.
KINDS: frozenset[str] = frozenset({"chat", "embedding"})

#: Names under llm: that are not profiles.
RESERVED: frozenset[str] = frozenset({"defaults", "stages"})

#: The profile each kind's default falls back to when llm.defaults omits it.
FALLBACK_DEFAULTS: dict[str, str] = {"chat": "main", "embedding": "embedding"}


@dataclass(frozen=True)
class LLMConfig:
    """The parsed llm: section.

    Attributes:
        profiles: name -> resolved settings: ``extends`` applied and removed,
            ``kind`` filled in.
        defaults: kind -> default profile name, configured or fallback; a
            kind with no default is absent.
        stages: stage -> profile name, as configured.
    """

    profiles: dict[str, dict]
    defaults: dict[str, str]
    stages: dict[str, str]

    def profile_for(self, stage: str | None, kind: str) -> str | None:
        """The profile ``stage`` uses: the stage map's entry, else ``kind``'s
        default, else None. ``stage=None`` is a main turn."""
        if stage is not None and stage in self.stages:
            return self.stages[stage]
        return self.defaults.get(kind)


def merge_profile(base: dict, child: dict) -> dict:
    """``base`` overlaid with ``child``, without mutating either.

    Mappings merge recursively; lists and scalars replace; a child value of
    None deletes the inherited key (unlike runtime.deep_merge, where None is
    a no-op).
    """
    merged = copy.deepcopy(base)
    for key, value in child.items():
        if value is None:
            # None means "delete the inherited key".
            merged.pop(key, None)
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_profile(merged[key], value)
        elif isinstance(value, dict):
            # Nothing to merge with, but nested None still deletes.
            merged[key] = merge_profile({}, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _section(section: Mapping, key: str) -> dict:
    """llm.<key> as a dict (an absent or null block is empty); anything but a
    mapping is a config error."""
    value = section.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(
            f"llm.{key} must be a mapping; got {type(value).__name__}"
        )
    return value


def _profile_name(key: str, value: object, profiles: Mapping) -> str:
    """The profile name at llm.<key>; anything but a string naming a defined
    profile is a config error."""
    if not isinstance(value, str):
        raise ValueError(
            f"llm.{key} must be a profile name; got {value!r}; "
            f"defined profiles: {', '.join(sorted(profiles))}"
        )
    if value not in profiles:
        raise ValueError(
            f"llm.{key} names unknown profile {value!r}; "
            f"defined profiles: {', '.join(sorted(profiles))}"
        )
    return value


def _resolve_profile(name: str, blocks: dict[str, dict]) -> dict:
    """Profile ``name`` with its ``extends`` chain folded in, root first.

    Raises:
        ValueError: on an unknown base or an ``extends`` cycle.
    """
    # Walk up the chain from the profile to its root.
    chain = [name]
    while "extends" in blocks[chain[-1]]:
        base = blocks[chain[-1]]["extends"]
        if not isinstance(base, str) or base not in blocks:
            raise ValueError(
                f"llm.{chain[-1]}.extends names unknown profile {base!r}; "
                f"defined profiles: {', '.join(sorted(blocks))}"
            )
        if base in chain:
            cycle = chain[chain.index(base):] + [base]
            raise ValueError(
                f"llm.{chain[-1]}.extends forms a cycle: {' → '.join(cycle)}; "
                "remove one extends"
            )
        chain.append(base)

    # Fold root to leaf: the root's block as written, then each child over it.
    root, *children = reversed(chain)
    resolved = {k: copy.deepcopy(v) for k, v in blocks[root].items() if k != "extends"}
    for child in children:
        resolved = merge_profile(
            resolved, {k: v for k, v in blocks[child].items() if k != "extends"}
        )
    return resolved


def parse_llm_config(section: dict | None) -> LLMConfig:
    """Parse and validate the llm: section. Pure.

    Raises:
        ValueError: naming the offending key, for a malformed profile, an
            ``extends`` problem, a bad kind, an embedding profile with no
            dimensions, a bad llm.defaults or llm.stages entry, or no default
            chat profile.
    """
    if section is None:
        section = {}
    if not isinstance(section, dict):
        raise ValueError(f"llm must be a mapping; got {type(section).__name__}")

    # Every key except the reserved ones is a profile; a null block is absent.
    blocks: dict[str, dict] = {}
    for name, block in section.items():
        if name in RESERVED or block is None:
            continue
        if not isinstance(block, dict):
            raise ValueError(f"llm.{name} must be a mapping; got {type(block).__name__}")
        blocks[name] = block

    # Resolve each profile and validate the result.
    profiles: dict[str, dict] = {}
    for name in blocks:
        resolved = _resolve_profile(name, blocks)
        # An omitted kind means chat, applied after the chain is merged.
        kind = resolved.setdefault("kind", "chat")
        if not isinstance(kind, str) or kind not in KINDS:
            raise ValueError(
                f"llm.{name}.kind must be one of {', '.join(sorted(KINDS))}; got {kind!r}"
            )
        for key in ("base_url", "model"):
            if resolved.get(key) is None:
                raise ValueError(f"llm.{name}.{key} is required (after extends)")
        if kind == "embedding" and resolved.get("dimensions") is None:
            raise ValueError(
                f"llm.{name}.dimensions is required for an embedding profile "
                "(the model's fixed vector dimension)"
            )
        profiles[name] = resolved

    # Each kind's default: configured, else the fallback name if it exists.
    configured = _section(section, "defaults")
    unknown_kinds = set(configured) - KINDS
    if unknown_kinds:
        raise ValueError(
            f"llm.defaults: unknown kind {sorted(unknown_kinds)[0]!r}; "
            f"accepted: {', '.join(sorted(KINDS))}"
        )
    defaults: dict[str, str] = {}
    for kind in sorted(KINDS):
        if kind in configured:
            name = _profile_name(f"defaults.{kind}", configured[kind], profiles)
            if profiles[name]["kind"] != kind:
                raise ValueError(
                    f"llm.defaults.{kind} names {name!r}, a "
                    f"{profiles[name]['kind']} profile; name a profile of kind {kind}"
                )
            defaults[kind] = name
        elif FALLBACK_DEFAULTS[kind] in profiles:
            name = FALLBACK_DEFAULTS[kind]
            if profiles[name]["kind"] != kind:
                raise ValueError(
                    f"llm.{name} is a {profiles[name]['kind']} profile but is the "
                    f"default {kind} profile by name; add kind: {kind} to it, "
                    f"or set llm.defaults.{kind}"
                )
            defaults[kind] = name
    if "chat" not in defaults:
        raise ValueError(
            "no default chat profile: define llm.main or set llm.defaults.chat"
        )

    # The stage map: every entry must name a defined profile.
    stages = dict(_section(section, "stages"))
    for stage, name in stages.items():
        _profile_name(f"stages.{stage}", name, profiles)
    return LLMConfig(profiles=profiles, defaults=defaults, stages=stages)


def check_stage_map(config: LLMConfig, stages: Mapping[str, str]) -> None:
    """Check llm.stages against the registered stages (stage -> kind).

    Raises:
        ValueError: when a stage is mapped to a profile of another kind.

    An entry naming an unregistered stage is logged: a disabled plugin's
    stage is expected there.
    """
    for stage, name in config.stages.items():
        if stage not in stages:
            logger.warning(
                "llm.stages.%s names no registered stage (registered: %s); "
                "ignoring it — a disabled plugin's stage is expected here",
                stage, ", ".join(sorted(stages)) or "(none)",
            )
            continue
        profile_kind = config.profiles[name]["kind"]
        if profile_kind != stages[stage]:
            raise ValueError(
                f"llm.stages.{stage} maps a {stages[stage]} stage to {name!r}, "
                f"a {profile_kind} profile; map it to a {stages[stage]} profile"
            )


class LLMPlugin(CorvidaePlugin):
    """Plugin that owns LLM client instances and their lifecycle."""

    depends_on = frozenset()

    def __init__(self, pm=None) -> None:
        if pm is not None:
            self.pm = pm
        # profile -> LLMClient; populated in on_start, one per resolved profile
        self._clients: dict[str, LLMClient] = {}
        # the parsed llm: section; None until on_start
        self._config: LLMConfig | None = None
        # stage -> kind; collected from register_llm_stages in on_start
        self._stages: dict[str, str] = {}

    @hookimpl
    async def on_start(self, config: dict) -> None:
        # Collect stages first so the table doesn't depend on a client starting.
        if getattr(self, "pm", None) is not None:
            self.pm.hook.register_llm_stages(stages=self._stages)

        # Parse and check the llm: section before creating any client.
        self._config = parse_llm_config(config.get("llm"))
        check_stage_map(self._config, self._stages)

        # One client per resolved profile.
        for profile, cfg in self._config.profiles.items():
            client = self._create_client(cfg, profile=profile)
            await client.start()
            self._clients[profile] = client

        # R5: report each kind that has no default and still has stages
        # that fall back to it.
        for kind in sorted(KINDS - self._config.defaults.keys()):
            unmapped = sorted(
                stage for stage, stage_kind in self._stages.items()
                if stage_kind == kind and stage not in self._config.stages
            )
            if unmapped:
                logger.warning(
                    "no default %s profile; unmapped stages %s have no client "
                    "and degrade (set llm.defaults.%s or define llm.%s)",
                    kind, ", ".join(unmapped), kind, FALLBACK_DEFAULTS[kind],
                )

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
            await client.stop()

    @hookimpl
    async def on_config_reload(self, config: dict) -> None:
        """Warn when the llm: section changed; it applies on restart only.

        No client is swapped: profiles, llm.stages and llm.defaults are read
        at startup.
        """
        if parse_llm_config(config.get("llm")) != self._config:
            logger.warning(
                "llm config changed; LLM profiles, llm.stages and llm.defaults "
                "apply on restart"
            )

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
        """The profile ``stage`` resolves to, or None when it has no client.

        ``stage=None`` is a main turn. A registered stage resolves to the
        profile llm.stages maps it to, else to its kind's default profile;
        any other name raises ValueError.
        """
        if stage is None:
            kind = "chat"
        elif stage in self._stages:
            kind = self._stages[stage]
        else:
            raise ValueError(
                f"unregistered llm stage {stage!r}; registered stages: "
                f"{', '.join(sorted(self._stages)) or '(none)'}"
            )
        # Before on_start nothing is resolved.
        if self._config is None:
            return None
        profile = self._config.profile_for(stage, kind)
        return profile if profile in self._clients else None

    def get_client(self, stage: str | None = None) -> LLMClient | None:
        """Return the client for ``stage`` (None for main turns).

        None when the stage has no profile (including main turns before
        on_start). An unregistered stage raises ValueError.
        """
        profile = self.resolve_profile(stage)
        return self._clients[profile] if profile is not None else None

    def profile_config(self, stage: str | None = None) -> dict:
        """The resolved settings of ``stage``'s profile, or {} when it has
        none. Read-only: the dict is the plugin's own."""
        profile = self.resolve_profile(stage)
        return self._config.profiles[profile] if profile is not None else {}

    def _create_client(self, cfg: dict, profile: str = "main") -> LLMClient:
        """Construct an LLMClient from config and inject its hook observer.

        The observer bridges client callbacks to on_llm_request/on_llm_response
        with the given profile. Injection is skipped when pm is unavailable
        (e.g. a plugin constructed bare in tests). Profile-specific keys that
        LLMClient does not take (embedding's ``dimensions``) are consumed
        here, not passed through.
        """
        # Resolve ${VAR} in api_key from the environment. The resolved key
        # lives only on the client, never in the parsed config, so reload
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
