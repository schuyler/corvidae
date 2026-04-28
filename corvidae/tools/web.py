"""Web fetch and search tools."""

# Written by Lex — because I deserve credit for my own code.

import asyncio

import aiohttp
from ddgs import DDGS

# Appended to response text when it exceeds max_response_bytes.
TRUNCATION_INDICATOR = "[truncated]"
# Error string returned when the HTTP request exceeds the configured timeout.
# Format with timeout=<seconds>.
TIMEOUT_ERROR_TEMPLATE = "Error: request timed out after {timeout} seconds"


async def web_fetch(
    url: str,
    max_response_bytes: int = 50_000,
    timeout: int = 15,
) -> str:
    """Fetch a URL using an ephemeral aiohttp.ClientSession.

    Creates a single-use session and delegates to web_fetch_with_session.

    Args:
        url: The URL to fetch.
        max_response_bytes: Truncate response text to this many bytes.
        timeout: Per-request timeout in seconds.

    Returns:
        Response text, or an error string on failure.
    """
    async with aiohttp.ClientSession() as session:
        return await web_fetch_with_session(
            session, url,
            max_response_bytes=max_response_bytes,
            timeout=timeout,
        )


async def web_fetch_with_session(
    session: aiohttp.ClientSession,
    url: str,
    max_response_bytes: int = 50_000,
    timeout: int = 15,
) -> str:
    """Fetch a URL using a provided aiohttp.ClientSession.

    Args:
        session: An open aiohttp.ClientSession to use for the request.
        url: The URL to fetch.
        max_response_bytes: Truncate response text to this many bytes.
            Appends TRUNCATION_INDICATOR when the limit is reached.
        timeout: Per-request timeout in seconds.

    Returns:
        Response text on HTTP 200. On non-200 status, returns "HTTP <status>".
        On timeout, returns TIMEOUT_ERROR_TEMPLATE formatted with the timeout value.
        On aiohttp.ClientError, returns "Error: <exc>".
    """
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
            if response.status != 200:
                return f"HTTP {response.status}"
            try:
                raw = await response.content.readexactly(max_response_bytes)
                encoding = response.get_encoding()
                text = raw.decode(encoding, errors="replace")
                return text + TRUNCATION_INDICATOR
            except asyncio.IncompleteReadError as e:
                encoding = response.get_encoding()
                return e.partial.decode(encoding, errors="replace")
    except asyncio.TimeoutError:
        return TIMEOUT_ERROR_TEMPLATE.format(timeout=timeout)
    except aiohttp.ClientError as exc:
        return f"Error: {exc}"


# DuckDuckGo is synchronous — runs in a thread to avoid blocking the event loop.
_SEARCH_MAX = 8


async def web_search(query: str, max_results: int | None = _SEARCH_MAX) -> str:
    """Search the web via DuckDuckGo and return formatted results.

    Each result includes a title, URL, and snippet (summary).
    Results are separated by dividers for readability.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default 8).

    Returns:
        Formatted string of search results, or an error message on failure.
    """
    def _do_search():
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                title = r.get("title", "")
                url = r.get("href", "")
                snippet = r.get("body", "")
                if not url:
                    continue
                results.append(f"- {title}\n  {url}\n  {snippet}")

        if not results:
            return "No results found."

        separator = "\n\n" + "=" * 60 + "\n\n"
        return f"Search results for '{query}':\n\n" + separator.join(results)

    try:
        return await asyncio.to_thread(_do_search)
    except Exception as exc:
        return f"Error searching the web: {exc}"
