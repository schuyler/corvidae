"""Shared helpers for building OpenAI-shaped LLM response dicts in tests."""

import json


def _make_text_response(text: str, reasoning: str | None = None) -> dict:
    msg: dict = {"role": "assistant", "content": text}
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    return {"choices": [{"message": msg}]}


def _make_tool_call_response(calls: list[dict]) -> dict:
    return {
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


def _make_tool_call(call_id: str, name: str, args: dict) -> dict:
    return {
        "id": call_id,
        "function": {
            "name": name,
            "arguments": json.dumps(args),
        },
    }


def _make_mixed_response(text: str, calls: list[dict]) -> dict:
    """Response with both text content and tool calls."""
    return {
        "choices": [
            {
                "message": {
                    "content": text,
                    "tool_calls": calls,
                }
            }
        ]
    }


def _make_null_content_tool_call_response(calls: list[dict]) -> dict:
    """Response with content=null and tool calls — as some LLMs emit."""
    return {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": calls,
                }
            }
        ]
    }


def _make_tool_call_malformed_args(call_id: str, name: str, raw_args: str) -> dict:
    """Build a tool call dict with raw (potentially invalid) JSON arguments."""
    return {
        "id": call_id,
        "function": {
            "name": name,
            "arguments": raw_args,
        },
    }
