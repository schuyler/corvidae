#!/usr/bin/env python3
"""
Smoke test for a hand-rolled agent loop against llama-server running Qwen3-35B-A3B.
Exits 0 on success, 1 on any critical failure.
"""

import asyncio
import json
import os
import re
import sys
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import aiohttp

BASE_URL = os.environ.get("BASE_URL", "http://192.168.1.88:8080").rstrip("/")
MODEL = os.environ.get("MODEL", None)  # discovered from /v1/models if not set

TIMEOUT = aiohttp.ClientTimeout(total=60)


def header(n, description):
    print(f"\n{'='*60}")
    print(f"Step {n}: {description}")
    print(f"{'='*60}")


def get_current_time(timezone: str) -> str:
    try:
        tz = ZoneInfo(timezone)
    except ZoneInfoNotFoundError:
        return f"ERROR: unknown timezone {timezone!r}"
    now = datetime.now(tz=tz)
    return now.isoformat()


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current time in a given timezone",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "IANA timezone name, e.g. Asia/Tokyo",
                    }
                },
                "required": ["timezone"],
            },
        },
    }
]


async def chat(session: aiohttp.ClientSession, payload: dict) -> dict:
    url = f"{BASE_URL}/v1/chat/completions"
    async with session.post(url, json=payload) as resp:
        resp.raise_for_status()
        return await resp.json()


def extract_message(response: dict) -> dict:
    return response["choices"][0]["message"]


def inspect_thinking(label: str, response: dict):
    msg = extract_message(response)
    content = msg.get("content") or ""
    has_think_tag = "<think>" in content
    has_reasoning_content = "reasoning_content" in msg
    has_tool_calls = bool(msg.get("tool_calls"))
    interleaved = has_think_tag and has_tool_calls

    print(f"\n  [Thinking inspection: {label}]")
    print(f"    <think> in content:         {has_think_tag}")
    print(f"    reasoning_content field:    {has_reasoning_content}")
    if has_reasoning_content:
        snippet = str(msg["reasoning_content"])[:120]
        print(f"    reasoning_content snippet: {snippet!r}")
    print(f"    tool_calls present:         {has_tool_calls}")
    print(f"    thinking interleaved with tool calls: {interleaved}")
    if interleaved:
        print("    WARNING: thinking tokens appear interleaved with tool call response")


def _print_summary(results):
    print(f"\n{'Step':<50} {'Verdict'}")
    print("-" * 60)
    for name, verdict in results:
        print(f"{name:<50} {verdict}")


async def main():
    results = []  # list of (step_name, verdict)
    critical_failure = False

    async with aiohttp.ClientSession(timeout=TIMEOUT) as session:

        # ------------------------------------------------------------------
        # Step 1: Connectivity
        # ------------------------------------------------------------------
        header(1, "Connectivity — GET /v1/models")
        global MODEL
        try:
            url = f"{BASE_URL}/v1/models"
            async with session.get(url) as resp:
                resp.raise_for_status()
                models_data = await resp.json()
            print(json.dumps(models_data, indent=2))
            model_ids = [m["id"] for m in models_data.get("data", [])]
            if MODEL is None:
                if model_ids:
                    MODEL = model_ids[0]
                else:
                    print("FAIL: no models returned from /v1/models")
                    results.append(("Step 1: Connectivity", "FAIL"))
                    critical_failure = True
            if not critical_failure:
                print(f"\nUsing model: {MODEL}")
                results.append(("Step 1: Connectivity", "PASS"))
        except aiohttp.ClientConnectorError as e:
            print(f"FAIL: connection refused or unreachable: {BASE_URL}")
            print(f"  Error: {e}")
            print("  Suggestion: check that llama-server is running and BASE_URL is correct")
            results.append(("Step 1: Connectivity", "FAIL"))
            critical_failure = True
        except aiohttp.ClientError as e:
            print(f"FAIL: HTTP error: {e}")
            results.append(("Step 1: Connectivity", "FAIL"))
            critical_failure = True

        if critical_failure:
            _print_summary(results)
            sys.exit(1)

        # ------------------------------------------------------------------
        # Step 2: Simple completion
        # ------------------------------------------------------------------
        header(2, "Simple completion — no tools")
        step2_response = None
        try:
            payload = {
                "model": MODEL,
                "messages": [
                    {"role": "user", "content": "Say hello in exactly three words."}
                ],
            }
            step2_response = await chat(session, payload)
            print("Full raw response:")
            print(json.dumps(step2_response, indent=2))
            raw_content = extract_message(step2_response).get("content") or ""
            visible_content = strip_thinking(raw_content)
            if visible_content.strip():
                print(f"\nVERDICT: PASS (non-empty content after stripping thinking tokens)")
                results.append(("Step 2: Simple completion", "PASS"))
            else:
                print("\nVERDICT: FAIL (empty content after stripping thinking tokens)")
                results.append(("Step 2: Simple completion", "FAIL"))
                critical_failure = True
        except aiohttp.ClientError as e:
            print(f"FAIL: {e}")
            results.append(("Step 2: Simple completion", "FAIL"))
            critical_failure = True

        if critical_failure:
            _print_summary(results)
            sys.exit(1)

        # ------------------------------------------------------------------
        # Step 3: Tool call
        # ------------------------------------------------------------------
        header(3, "Tool call — 'What time is it in Tokyo?'")
        step3_response = None
        step3_tool_args = None
        try:
            payload = {
                "model": MODEL,
                "messages": [
                    {"role": "user", "content": "What time is it in Tokyo?"}
                ],
                "tools": TOOL_SCHEMA,
                "tool_choice": "auto",
            }
            print("Request payload:")
            print(json.dumps(payload, indent=2))
            step3_response = await chat(session, payload)
            print("\nFull raw response:")
            print(json.dumps(step3_response, indent=2))

            msg = extract_message(step3_response)
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                print("\nVERDICT: FAIL — no tool_calls in response")
                print("  Suggestion: check that llama-server is started with the --jinja flag")
                results.append(("Step 3: Tool call", "FAIL"))
                critical_failure = True
            else:
                tc = tool_calls[0]
                fn_name = tc.get("function", {}).get("name")
                raw_args = tc.get("function", {}).get("arguments", "{}")
                try:
                    step3_tool_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    print(f"\nVERDICT: FAIL — malformed tool call arguments: {raw_args!r}")
                    results.append(("Step 3: Tool call", "FAIL"))
                    critical_failure = True
                else:
                    if fn_name == "get_current_time" and "timezone" in step3_tool_args:
                        print(f"\nVERDICT: PASS (function={fn_name}, args={step3_tool_args})")
                        results.append(("Step 3: Tool call", "PASS"))
                    else:
                        print(f"\nVERDICT: FAIL — unexpected function name or missing args: name={fn_name!r}, args={step3_tool_args}")
                        results.append(("Step 3: Tool call", "FAIL"))
                        critical_failure = True
        except aiohttp.ClientError as e:
            print(f"FAIL: {e}")
            results.append(("Step 3: Tool call", "FAIL"))
            critical_failure = True

        if critical_failure:
            _print_summary(results)
            sys.exit(1)

        # ------------------------------------------------------------------
        # Step 4: Tool result round-trip
        # ------------------------------------------------------------------
        header(4, "Tool result round-trip")
        step4_response = None
        try:
            tc = extract_message(step3_response)["tool_calls"][0]
            tool_call_id = tc.get("id")
            if not tool_call_id:
                print("  WARNING: tool call has no id field, using positional fallback")
                tool_call_id = "call_0"
            timezone = step3_tool_args["timezone"]
            time_result = get_current_time(timezone)
            print(f"Executing get_current_time({timezone!r}) → {time_result}")

            messages = [
                {"role": "user", "content": "What time is it in Tokyo?"},
                extract_message(step3_response),
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": time_result,
                },
            ]
            payload = {
                "model": MODEL,
                "messages": messages,
                "tools": TOOL_SCHEMA,
                "tool_choice": "auto",
            }
            step4_response = await chat(session, payload)
            print("\nFull raw response:")
            print(json.dumps(step4_response, indent=2))

            msg = extract_message(step4_response)
            has_tool_calls = bool(msg.get("tool_calls"))
            raw_content = msg.get("content") or ""
            visible_content = strip_thinking(raw_content)
            if has_tool_calls:
                print("\nVERDICT: FAIL — model made another tool call instead of responding")
                results.append(("Step 4: Tool result round-trip", "FAIL"))
            elif visible_content.strip():
                print(f"\nVERDICT: PASS (final text response received)")
                results.append(("Step 4: Tool result round-trip", "PASS"))
            else:
                print("\nVERDICT: FAIL — empty content in final response")
                results.append(("Step 4: Tool result round-trip", "FAIL"))
        except aiohttp.ClientError as e:
            print(f"FAIL: {e}")
            results.append(("Step 4: Tool result round-trip", "FAIL"))

        # ------------------------------------------------------------------
        # Step 5: Thinking token investigation (INFO only)
        # ------------------------------------------------------------------
        header(5, "Thinking token investigation (INFO)")
        print("Examining raw responses from steps 2, 3, and 4:")
        if step2_response:
            inspect_thinking("Step 2: simple completion", step2_response)
        if step3_response:
            inspect_thinking("Step 3: tool call request", step3_response)
        if step4_response:
            inspect_thinking("Step 4: tool result response", step4_response)
        results.append(("Step 5: Thinking token investigation", "INFO"))

        # ------------------------------------------------------------------
        # Step 6: Multi-turn with thinking preserved in history
        # ------------------------------------------------------------------
        header(6, "Multi-turn — thinking preserved in history")
        step6_ok = False
        try:
            messages = [{"role": "user", "content": "What time is it in Tokyo?"}]
            payload = {"model": MODEL, "messages": messages, "tools": TOOL_SCHEMA, "tool_choice": "auto"}
            r1 = await chat(session, payload)
            print("Turn 1 response:")
            print(json.dumps(r1, indent=2))

            msg1 = extract_message(r1)
            tool_calls_1 = msg1.get("tool_calls") or []
            if not tool_calls_1:
                print("FAIL: no tool call in turn 1")
                results.append(("Step 6: Multi-turn (thinking kept)", "FAIL"))
            else:
                tc1 = tool_calls_1[0]
                args1 = json.loads(tc1["function"]["arguments"])
                result1 = get_current_time(args1["timezone"])
                tc1_id = tc1.get("id")
                if not tc1_id:
                    print("  WARNING: tool call has no id field, using positional fallback")
                    tc1_id = "call_0"
                messages.append(msg1)
                messages.append({"role": "tool", "tool_call_id": tc1_id, "content": result1})

                payload = {"model": MODEL, "messages": messages, "tools": TOOL_SCHEMA, "tool_choice": "auto"}
                r2 = await chat(session, payload)
                print("\nTurn 2 response (after tool result):")
                print(json.dumps(r2, indent=2))

                msg2 = extract_message(r2)
                if msg2.get("tool_calls"):
                    print("FAIL: turn 2 produced another tool call instead of a final response")
                    results.append(("Step 6: Multi-turn (thinking kept)", "FAIL"))
                else:
                    messages.append(msg2)
                    messages.append({"role": "user", "content": "And what time is it in London?"})

                    payload = {"model": MODEL, "messages": messages, "tools": TOOL_SCHEMA, "tool_choice": "auto"}
                    r3 = await chat(session, payload)
                    print("\nTurn 3 response (London question):")
                    print(json.dumps(r3, indent=2))

                    msg3 = extract_message(r3)
                    tool_calls_3 = msg3.get("tool_calls") or []
                    if not tool_calls_3:
                        print("FAIL: no tool call in turn 3 for London")
                        results.append(("Step 6: Multi-turn (thinking kept)", "FAIL"))
                    else:
                        tc3 = tool_calls_3[0]
                        args3 = json.loads(tc3["function"]["arguments"])
                        result3 = get_current_time(args3["timezone"])
                        tc3_id = tc3.get("id")
                        if not tc3_id:
                            print("  WARNING: tool call has no id field, using positional fallback")
                            tc3_id = "call_1"
                        messages.append(msg3)
                        messages.append({"role": "tool", "tool_call_id": tc3_id, "content": result3})

                        payload = {"model": MODEL, "messages": messages, "tools": TOOL_SCHEMA, "tool_choice": "auto"}
                        r4 = await chat(session, payload)
                        print("\nTurn 4 response (final):")
                        print(json.dumps(r4, indent=2))

                        msg4 = extract_message(r4)
                        content4 = strip_thinking(msg4.get("content") or "")
                        if content4.strip() and not msg4.get("tool_calls"):
                            print("\nVERDICT: PASS")
                            results.append(("Step 6: Multi-turn (thinking kept)", "PASS"))
                            step6_ok = True
                        else:
                            print("\nVERDICT: FAIL — no coherent final response")
                            results.append(("Step 6: Multi-turn (thinking kept)", "FAIL"))
        except (aiohttp.ClientError, KeyError, json.JSONDecodeError) as e:
            print(f"FAIL: {e}")
            results.append(("Step 6: Multi-turn (thinking kept)", "FAIL"))

        # ------------------------------------------------------------------
        # Step 7: Multi-turn with thinking stripped
        # ------------------------------------------------------------------
        header(7, "Multi-turn — thinking stripped from history")
        step7_ok = False
        try:
            messages = [{"role": "user", "content": "What time is it in Tokyo?"}]
            payload = {"model": MODEL, "messages": messages, "tools": TOOL_SCHEMA, "tool_choice": "auto"}
            r1 = await chat(session, payload)
            print("Turn 1 response:")
            print(json.dumps(r1, indent=2))

            msg1 = extract_message(r1)
            tool_calls_1 = msg1.get("tool_calls") or []
            if not tool_calls_1:
                print("FAIL: no tool call in turn 1")
                results.append(("Step 7: Multi-turn (thinking stripped)", "FAIL"))
            else:
                tc1 = tool_calls_1[0]
                args1 = json.loads(tc1["function"]["arguments"])
                result1 = get_current_time(args1["timezone"])
                tc1_id = tc1.get("id")
                if not tc1_id:
                    print("  WARNING: tool call has no id field, using positional fallback")
                    tc1_id = "call_0"

                # strip thinking from msg1 content before appending
                msg1_stripped = dict(msg1)
                if msg1_stripped.get("content"):
                    msg1_stripped["content"] = strip_thinking(msg1_stripped["content"])

                messages.append(msg1_stripped)
                messages.append({"role": "tool", "tool_call_id": tc1_id, "content": result1})

                payload = {"model": MODEL, "messages": messages, "tools": TOOL_SCHEMA, "tool_choice": "auto"}
                r2 = await chat(session, payload)
                print("\nTurn 2 response (after tool result):")
                print(json.dumps(r2, indent=2))

                msg2 = extract_message(r2)
                if msg2.get("tool_calls"):
                    print("FAIL: turn 2 produced another tool call instead of a final response")
                    results.append(("Step 7: Multi-turn (thinking stripped)", "FAIL"))
                else:
                    msg2_stripped = dict(msg2)
                    if msg2_stripped.get("content"):
                        msg2_stripped["content"] = strip_thinking(msg2_stripped["content"])

                    messages.append(msg2_stripped)
                    messages.append({"role": "user", "content": "And what time is it in London?"})

                    payload = {"model": MODEL, "messages": messages, "tools": TOOL_SCHEMA, "tool_choice": "auto"}
                    r3 = await chat(session, payload)
                    print("\nTurn 3 response (London question):")
                    print(json.dumps(r3, indent=2))

                    msg3 = extract_message(r3)
                    tool_calls_3 = msg3.get("tool_calls") or []
                    if not tool_calls_3:
                        print("FAIL: no tool call in turn 3 for London")
                        results.append(("Step 7: Multi-turn (thinking stripped)", "FAIL"))
                    else:
                        tc3 = tool_calls_3[0]
                        args3 = json.loads(tc3["function"]["arguments"])
                        result3 = get_current_time(args3["timezone"])
                        tc3_id = tc3.get("id")
                        if not tc3_id:
                            print("  WARNING: tool call has no id field, using positional fallback")
                            tc3_id = "call_1"

                        msg3_stripped = dict(msg3)
                        if msg3_stripped.get("content"):
                            msg3_stripped["content"] = strip_thinking(msg3_stripped["content"])

                        messages.append(msg3_stripped)
                        messages.append({"role": "tool", "tool_call_id": tc3_id, "content": result3})

                        payload = {"model": MODEL, "messages": messages, "tools": TOOL_SCHEMA, "tool_choice": "auto"}
                        r4 = await chat(session, payload)
                        print("\nTurn 4 response (final):")
                        print(json.dumps(r4, indent=2))

                        msg4 = extract_message(r4)
                        content4 = strip_thinking(msg4.get("content") or "")
                        if content4.strip() and not msg4.get("tool_calls"):
                            print("\nVERDICT: PASS")
                            results.append(("Step 7: Multi-turn (thinking stripped)", "PASS"))
                            step7_ok = True
                        else:
                            print("\nVERDICT: FAIL — no coherent final response")
                            results.append(("Step 7: Multi-turn (thinking stripped)", "FAIL"))
        except (aiohttp.ClientError, KeyError, json.JSONDecodeError) as e:
            print(f"FAIL: {e}")
            results.append(("Step 7: Multi-turn (thinking stripped)", "FAIL"))

        # ------------------------------------------------------------------
        # Step 8: Summary
        # ------------------------------------------------------------------
        header(8, "Summary")
        _print_summary(results)

        # Recommendation
        print("\nRecommendation for keep_thinking_in_history:")
        if step6_ok and step7_ok:
            print("  Both kept and stripped succeeded. Default: keep_thinking_in_history = False")
            print("  (Stripping is safer — avoids inflating context length with reasoning tokens.)")
        elif step6_ok and not step7_ok:
            print("  Kept succeeded, stripped failed. Default: keep_thinking_in_history = True")
            print("  (The model may rely on its own reasoning trace for coherent multi-turn behavior.)")
        elif not step6_ok and step7_ok:
            print("  Stripped succeeded, kept failed. Default: keep_thinking_in_history = False")
            print("  (Thinking tokens in history may confuse the model.)")
        else:
            print("  Both failed. Multi-turn tool use is unreliable. Investigate tool call issues first.")

    # Exit code
    CRITICAL_STEPS = {"Step 1: Connectivity", "Step 2: Simple completion", "Step 3: Tool call"}
    failed_critical = [name for name, verdict in results if verdict == "FAIL" and name in CRITICAL_STEPS]
    if failed_critical:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
