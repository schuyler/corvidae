"""LLM client for OpenAI-compatible Chat Completions API.

This module provides an async HTTP client for LLM chat completions. It handles:
- Session lifecycle (start/stop with aiohttp)
- Request/response logging with timing and token usage
- Error handling and propagation

Logging:
    - INFO: client started/stopped, chat completion returned with metadata
    - DEBUG: chat completion request (message count, tool count)
    - ERROR: HTTP errors from the API

Token usage is extracted from the response's `usage` field. If the API doesn't
provide usage data, the fields are logged as `None`.
"""

import asyncio
import logging
import time
import uuid

import aiohttp

TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}
ERROR_BODY_CHARS = 2000
# Minimum sleep between "model is not loaded" retries, so a zero backoff
# config cannot spin without spending the wait budget.
MODEL_NOT_LOADED_MIN_DELAY = 1.0
# llama-server router's 400 body when the requested model is unloaded and
# --no-models-autoload is set (tools/server/server-models.cpp,
# router_validate_model).
MODEL_NOT_LOADED_MARKER = "model is not loaded"

logger = logging.getLogger(__name__)


class LLMHTTPError(aiohttp.ClientResponseError):
    """A non-2xx LLM API response, carrying the response body and model.

    Subclasses ClientResponseError so existing handlers keep working; the
    extra fields let error handlers (on_llm_error) tell failures apart.
    """

    def __init__(self, cause: aiohttp.ClientResponseError, body: str, model: str) -> None:
        super().__init__(
            request_info=cause.request_info,
            history=cause.history,
            status=cause.status,
            message=cause.message,
            headers=cause.headers,
        )
        self.body = body
        self.model = model


def _body_says_not_loaded(status: int, body: str) -> bool:
    """Whether a response is the router's 'model is not loaded' 400."""
    return status == 400 and isinstance(body, str) and MODEL_NOT_LOADED_MARKER in body


def is_model_not_loaded(error: BaseException) -> bool:
    """Whether error is an LLMHTTPError for an unloaded backend model."""
    return isinstance(error, LLMHTTPError) and _body_says_not_loaded(error.status, error.body)


class LLMClient:
    """Async client for OpenAI-compatible Chat Completions API.

    The client maintains an aiohttp session with optional Bearer auth.
    All requests include timing and token usage logging.

    Attributes:
        base_url: API base URL (e.g., "http://localhost:8080")
        model: Model name for requests (e.g., "llama3.1")
        api_key: Optional API key for Bearer auth
        session: aiohttp session (created by start(), closed by stop())
        observer: Optional observation seam — any object with async
            ``request(**kwargs)`` and ``response(**kwargs)`` methods.
            Default None (behavior is byte-for-byte unobserved). This
            module stays free of plugin-system imports; LLMPlugin injects
            an observer that bridges to the hook system. Observer failures
            are logged and never break an LLM call. Exactly one response
            callback fires per chat() call (retried transient attempts do
            not fire it).
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        extra_body: dict | None = None,
        max_retries: int = 3,
        retry_base_delay: float = 2.0,
        retry_max_delay: float = 60.0,
        timeout: float | None = None,
        document_prefix: str = "",
        query_prefix: str = "",
        model_not_loaded_wait: float = 60.0,
    ) -> None:
        """Initialize the LLM client.

        Args:
            base_url: API base URL (trailing slash will be stripped)
            model: Model identifier to use in requests
            api_key: Optional Bearer token for authentication
            extra_body: Optional dict of extra fields merged into every
                        chat request payload (instance-level defaults).
                        Call-level extra_body passed to chat() overrides
                        these when keys conflict.
            max_retries: Max retry attempts on transient errors (default 3).
                         Set to 0 to disable retries.
            retry_base_delay: Base delay in seconds for exponential backoff.
            retry_max_delay: Maximum delay cap in seconds.
            timeout: Total HTTP timeout in seconds per LLM request.
                     None uses aiohttp's session-level default (300s).
            document_prefix: String prepended to every text when kind='document'.
                             Defaults to '' (no prefix).
            query_prefix: String prepended to every text when kind='query'.
                          Defaults to '' (no prefix).
            model_not_loaded_wait: Total seconds to keep retrying while the
                          server reports the model is not loaded (a
                          llama-server router with --no-models-autoload).
                          Independent of max_retries. 0 disables.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.extra_body = extra_body
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        self.timeout = (
            aiohttp.ClientTimeout(total=timeout) if timeout is not None else None
        )
        self.document_prefix = document_prefix
        self.query_prefix = query_prefix
        self.model_not_loaded_wait = model_not_loaded_wait
        self.session: aiohttp.ClientSession | None = None
        self.observer: object | None = None

    async def _fire_observer(self, method: str, **kwargs) -> None:
        """Invoke an observer callback, swallowing (but logging) its errors.

        A metering bug must never take down an LLM call, so each observer
        invocation is individually wrapped.
        """
        if self.observer is None:
            return
        try:
            await getattr(self.observer, method)(**kwargs)
        except Exception:
            logger.warning(
                "LLM observer %s callback failed", method, exc_info=True
            )

    def _retry_delay(
        self, attempt: int, retry_after: str | None = None
    ) -> float:
        """Compute retry delay with exponential backoff.

        Honors Retry-After header (integer/float seconds) when present.
        Falls back to exponential backoff for unparseable values.
        """
        if retry_after is not None:
            try:
                return min(float(retry_after), self.retry_max_delay)
            except ValueError:
                pass
        # Cap the exponent: past 2**30 the delay is pinned at the max anyway,
        # and an unbounded one overflows float conversion on long waits.
        return min(self.retry_base_delay * (2 ** min(attempt, 30)), self.retry_max_delay)

    async def start(self) -> None:
        """Create the aiohttp session with optional auth header.

        Logs an INFO message with base_url and model.
        """
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self.session = aiohttp.ClientSession(headers=headers)

        logger.info(
            "client started",
            extra={"base_url": self.base_url, "model": self.model},
        )

    async def stop(self) -> None:
        """Close the aiohttp session.

        Logs an INFO message. Safe to call if session is None.
        """
        if self.session:
            await self.session.close()

        logger.info("client stopped")

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        extra_body: dict | None = None,
    ) -> dict:
        """Send a chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            tools: Optional list of tool schemas for function calling
            extra_body: Optional dict of extra fields to merge into request payload

        Returns:
            Full response dict from the API (includes choices, usage, etc.)

        Raises:
            RuntimeError: If start() was not called
            aiohttp.ClientResponseError: On non-2xx HTTP status

        Logs:
            DEBUG: Request metadata (message count, tool count)
            INFO: Response metadata (latency_ms, token counts, model)
            ERROR: HTTP failure with status code
        """
        if self.session is None:
            raise RuntimeError("LLMClient.start() must be called before chat()")

        logger.debug(
            "chat completion request",
            extra={
                "message_count": len(messages),
                "tool_count": len(tools) if tools else 0,
            },
        )

        payload = {
            "model": self.model,
            "messages": messages,
        }
        if tools:
            payload["tools"] = tools
        if self.extra_body:
            payload.update(self.extra_body)  # instance defaults
        if extra_body:
            payload.update(extra_body)  # call-level override wins

        # Mint a per-call id pairing the observer request with its response,
        # and announce the call (summarized — counts, not payloads).
        request_id = uuid.uuid4().hex[:12]
        await self._fire_observer(
            "request",
            model=self.model,
            request_id=request_id,
            message_count=len(messages),
            tool_count=len(tools) if tools else 0,
        )

        response, elapsed = await self._post_with_retries(
            "/chat/completions", payload, request_id, label="chat completion"
        )

        usage = response.get("usage", {})
        logger.info(
            "chat completion returned",
            extra={
                "model": self.model,
                "latency_ms": round(elapsed * 1000, 1),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            },
        )

        return response

    async def embed(
        self, texts: list[str], kind: str
    ) -> list[list[float]]:
        """POST {base_url}/embeddings (OpenAI-compatible).

        Args:
            texts: Input strings to embed.
            kind: Embedding kind — 'document' prepends document_prefix,
                  'query' prepends query_prefix. Unknown values raise ValueError.

        Returns one vector per input text, in order. Raises on terminal
        failure — callers own their degradation (bootstrap-mapping §3.1).
        """
        if self.session is None:
            raise RuntimeError("LLMClient.start() must be called before embed()")

        if kind == "document":
            prefix = self.document_prefix
        elif kind == "query":
            prefix = self.query_prefix
        else:
            raise ValueError(
                f"embed kind must be 'document' or 'query', got {kind!r}"
            )
        if prefix:
            texts = [prefix + t for t in texts]

        payload = {"model": self.model, "input": texts}

        # One observer request/response pair per embed call, same as chat().
        request_id = uuid.uuid4().hex[:12]
        await self._fire_observer(
            "request",
            model=self.model,
            request_id=request_id,
            message_count=len(texts),
            tool_count=0,
        )

        response, elapsed = await self._post_with_retries(
            "/embeddings", payload, request_id, label="embeddings"
        )

        # Unpack one vector per input, honoring the index field when the
        # provider supplies it (the order is not guaranteed by every server).
        data = sorted(response["data"], key=lambda d: d.get("index", 0))
        vectors = [d["embedding"] for d in data]

        logger.info(
            "embeddings returned",
            extra={
                "model": self.model,
                "latency_ms": round(elapsed * 1000, 1),
                "input_count": len(texts),
                "vector_count": len(vectors),
            },
        )
        return vectors

    async def _post_with_retries(
        self, path: str, payload: dict, request_id: str, label: str
    ) -> tuple[dict, float]:
        """POST with transient-error retries; shared by chat() and embed().

        Fires exactly one observer ``response`` callback per call — with
        usage on success, with the error string on terminal failure.

        Returns:
            (parsed JSON response, elapsed seconds of the final attempt).
        """
        start = time.monotonic()
        # Transient-error attempts, and the separate not-loaded wait budget.
        attempt = 0
        not_loaded_attempt = 0
        not_loaded_waited = 0.0
        try:
            while True:
                try:
                    start = time.monotonic()
                    async with self.session.post(
                        f"{self.base_url}{path}",
                        json=payload,
                        timeout=self.timeout,
                    ) as resp:
                        if (
                            resp.status in TRANSIENT_STATUS_CODES
                            and attempt < self.max_retries
                        ):
                            retry_after = resp.headers.get("Retry-After")
                            delay = self._retry_delay(attempt, retry_after)
                            logger.warning(
                                "transient LLM error, retrying",
                                extra={
                                    "status": resp.status,
                                    "attempt": attempt + 1,
                                    "delay": delay,
                                },
                            )
                            await asyncio.sleep(delay)
                            attempt += 1
                            continue

                        error_body = ""
                        if resp.status >= 400:
                            # raise_for_status() releases the connection; read
                            # the body first or lose it.
                            error_body = (await resp.text())[:ERROR_BODY_CHARS]

                        # Model unloaded on a router (GitHub issue #55): it may
                        # be swapped back shortly, so wait and retry within
                        # model_not_loaded_wait, clamping the last sleep.
                        remaining = self.model_not_loaded_wait - not_loaded_waited
                        if _body_says_not_loaded(resp.status, error_body) and remaining > 0:
                            delay = min(
                                max(self._retry_delay(not_loaded_attempt), MODEL_NOT_LOADED_MIN_DELAY),
                                remaining,
                            )
                            logger.warning(
                                "LLM model not loaded, waiting",
                                extra={
                                    "model": self.model,
                                    "attempt": not_loaded_attempt + 1,
                                    "delay": delay,
                                },
                            )
                            await asyncio.sleep(delay)
                            not_loaded_waited += delay
                            not_loaded_attempt += 1
                            continue

                        try:
                            resp.raise_for_status()
                            response = await resp.json()
                        except aiohttp.ClientResponseError as exc:
                            logger.error(
                                "%s failed", label,
                                extra={"status": resp.status, "body": error_body},
                            )
                            # A 2xx with an unparseable body (ContentTypeError)
                            # is not an HTTP failure; propagate it unchanged.
                            if resp.status < 400:
                                raise
                            # Name the model this request asked for: extra_body
                            # can override the client's configured model.
                            raise LLMHTTPError(
                                exc, body=error_body, model=payload.get("model", self.model)
                            ) from exc

                except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as exc:
                    if attempt < self.max_retries:
                        delay = self._retry_delay(attempt)
                        logger.warning(
                            "transient LLM connection error, retrying",
                            extra={
                                "error": str(exc),
                                "attempt": attempt + 1,
                                "delay": delay,
                            },
                        )
                        await asyncio.sleep(delay)
                        attempt += 1
                        continue
                    raise

                break  # success
        except Exception as exc:
            # Terminal (non-retried) failure about to propagate: fire the
            # single response callback with the error before re-raising.
            await self._fire_observer(
                "response",
                model=self.model,
                request_id=request_id,
                usage=None,
                latency_ms=(time.monotonic() - start) * 1000,
                error=str(exc),
            )
            raise

        elapsed = time.monotonic() - start

        # Success: fire the single response callback with usage verbatim
        # (None when the API omitted it) and the already-computed latency.
        await self._fire_observer(
            "response",
            model=self.model,
            request_id=request_id,
            usage=response.get("usage"),
            latency_ms=elapsed * 1000,
            error=None,
        )

        return response, elapsed
