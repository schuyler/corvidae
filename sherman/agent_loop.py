import inspect
import json
import logging
import re
from typing import Callable

from pydantic import create_model

from sherman.llm import LLMClient

logger = logging.getLogger(__name__)


def _truncate(s: str, maxlen: int = 200) -> str:
    """Truncate a string to maxlen characters, appending '...' if truncated."""
    return s[:maxlen] + "..." if len(s) > maxlen else s


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
        logger.debug(
            "LLM response received",
            extra={
                "role": msg.get("role"),
                "tool_calls_count": len(tool_calls) if tool_calls else 0,
            },
        )

        if not tool_calls:
            return msg.get("content", "")

        for call in tool_calls:
            call_id = call["id"]
            fn_name = call["function"]["name"]
            args = json.loads(call["function"]["arguments"])

            logger.debug(
                "tool call dispatched",
                extra={"tool": fn_name, "arg_keys": list(args.keys())},
            )

            if fn_name not in tools:
                logger.warning("unknown tool called: %s", fn_name)
                content = f"Error: unknown tool '{fn_name}'"
            else:
                try:
                    content = await tools[fn_name](**args)
                except Exception:
                    logger.warning(
                        "tool %s raised exception", fn_name, exc_info=True
                    )
                    content = f"Error: unknown error"

            logger.debug(
                "tool call result",
                extra={"tool": fn_name, "result_length": len(str(content))},
            )

            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": str(content),
            })

    logger.warning("max tool-calling rounds reached")
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


def strip_reasoning_content(messages: list[dict]) -> None:
    """Remove reasoning_content from assistant messages in place."""
    for msg in messages:
        if msg.get("role") == "assistant":
            msg.pop("reasoning_content", None)
