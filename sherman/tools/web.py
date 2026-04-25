"""Web fetch tool."""

import asyncio

import aiohttp


async def web_fetch(url: str) -> str:
    """Fetch a URL and return its text content."""
    timeout = aiohttp.ClientTimeout(total=15)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                if response.status != 200:
                    return f"HTTP {response.status}"
                text = await response.text()
                if len(text) > 50000:
                    return text[:50000] + "[truncated]"
                return text
    except asyncio.TimeoutError:
        return f"Error: request timed out after 15 seconds"
    except aiohttp.ClientError as exc:
        return f"Error: {exc}"


async def web_fetch_with_session(session: aiohttp.ClientSession, url: str) -> str:
    """Fetch a URL using a pre-existing aiohttp.ClientSession."""
    try:
        async with session.get(url) as response:
            if response.status != 200:
                return f"HTTP {response.status}"
            text = await response.text()
            if len(text) > 50000:
                return text[:50000] + "[truncated]"
            return text
    except asyncio.TimeoutError:
        return f"Error: request timed out after 15 seconds"
    except aiohttp.ClientError as exc:
        return f"Error: {exc}"
