"""Web fetch tool."""

import asyncio

import aiohttp

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
    """Fetch a URL and return its text content."""
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    try:
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            async with session.get(url) as response:
                if response.status != 200:
                    return f"HTTP {response.status}"
                text = await response.text()
                if len(text) > max_response_bytes:
                    return text[:max_response_bytes] + TRUNCATION_INDICATOR
                return text
    except asyncio.TimeoutError:
        return TIMEOUT_ERROR_TEMPLATE.format(timeout=timeout)
    except aiohttp.ClientError as exc:
        return f"Error: {exc}"


async def web_fetch_with_session(
    session: aiohttp.ClientSession,
    url: str,
    max_response_bytes: int = 50_000,
    timeout: int = 15,
) -> str:
    """Fetch a URL using a pre-existing aiohttp.ClientSession."""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
            if response.status != 200:
                return f"HTTP {response.status}"
            text = await response.text()
            if len(text) > max_response_bytes:
                return text[:max_response_bytes] + TRUNCATION_INDICATOR
            return text
    except asyncio.TimeoutError:
        return TIMEOUT_ERROR_TEMPLATE.format(timeout=timeout)
    except aiohttp.ClientError as exc:
        return f"Error: {exc}"
