"""Shared helpers for building OpenAI-shaped LLM response dicts in tests."""

import json


def _usage(prompt: int = 100, completion: int = 20, cached: int = 0) -> dict:
    """Top-level `usage` envelope, shaped like llama-server's response.

    Real responses carry `usage` at the top level, never on the message —
    fixtures attach it there by default so tests exercise the real shape.
    """
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_tokens_details": {"cached_tokens": cached},
    }


def _make_text_response(
    text: str, reasoning: str | None = None, include_usage: bool = True
) -> dict:
    msg: dict = {"role": "assistant", "content": text}
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    response: dict = {"choices": [{"message": msg}]}
    if include_usage:
        response["usage"] = _usage()
    return response


def _make_tool_call_response(calls: list[dict], include_usage: bool = True) -> dict:
    response: dict = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": calls,
                }
            }
        ]
    }
    if include_usage:
        response["usage"] = _usage()
    return response


def _make_tool_call(call_id: str, name: str, args: dict) -> dict:
    return {
        "id": call_id,
        "function": {
            "name": name,
            "arguments": json.dumps(args),
        },
    }


def _make_mixed_response(
    text: str, calls: list[dict], include_usage: bool = True
) -> dict:
    """Response with both text content and tool calls."""
    response: dict = {
        "choices": [
            {
                "message": {
                    "content": text,
                    "tool_calls": calls,
                }
            }
        ]
    }
    if include_usage:
        response["usage"] = _usage()
    return response


def _make_null_content_tool_call_response(
    calls: list[dict], include_usage: bool = True
) -> dict:
    """Response with content=null and tool calls — as some LLMs emit."""
    response: dict = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": calls,
                }
            }
        ]
    }
    if include_usage:
        response["usage"] = _usage()
    return response


def _make_tool_call_malformed_args(call_id: str, name: str, raw_args: str) -> dict:
    """Build a tool call dict with raw (potentially invalid) JSON arguments."""
    return {
        "id": call_id,
        "function": {
            "name": name,
            "arguments": raw_args,
        },
    }
