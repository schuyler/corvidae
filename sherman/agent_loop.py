import inspect
import json
import re
from typing import Callable

from pydantic import create_model

from sherman.llm import LLMClient


async def run_agent_loop(
    client: LLMClient,
    messages: list[dict],
    tools: dict[str, Callable],
    tool_schemas: list[dict],
    max_turns: int = 10,
) -> str:
    """Run the agent loop to completion. Mutates messages in place."""
    for _ in range(max_turns):
        response = await client.chat(messages, tools=tool_schemas or None)
        msg = response["choices"][0]["message"]
        msg.setdefault("role", "assistant")
        messages.append(msg)

        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            return msg.get("content", "")

        for call in tool_calls:
            call_id = call["id"]
            fn_name = call["function"]["name"]
            args = json.loads(call["function"]["arguments"])

            if fn_name not in tools:
                content = f"Error: unknown tool '{fn_name}'"
            else:
                try:
                    content = await tools[fn_name](**args)
                except Exception as exc:
                    content = f"Error: {exc}"

            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": str(content),
            })

    return "(max tool-calling rounds reached)"


def tool_to_schema(fn: Callable) -> dict:
    """Generate a Chat Completions tool schema from a typed function."""
    sig = inspect.signature(fn)
    fields = {}
    for param_name, param in sig.parameters.items():
        annotation = param.annotation if param.annotation is not inspect.Parameter.empty else str
        fields[param_name] = (annotation, ...)

    model = create_model(fn.__name__, **fields)
    schema = model.model_json_schema()

    # Strip top-level title
    schema.pop("title", None)

    # Strip title from each property
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)

    description = ""
    if fn.__doc__:
        description = fn.__doc__.strip().splitlines()[0]

    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": description,
            "parameters": schema,
        },
    }


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from text."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
