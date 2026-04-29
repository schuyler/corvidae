#!/usr/bin/env python3
"""Evaluation framework for compaction prompt quality.

Uses the real death-spiral conversation (extracted to a static fixture) to
compare compaction prompts. Scores summaries on whether a fresh LLM session
can identify the bug being investigated and the discoveries made.

Usage:
    python scripts/eval_compaction.py [--fixture tests/fixtures/death_spiral_compaction.json] [--api-key KEY] [--base-url URL] [--model MODEL]
    python scripts/eval_compaction.py --segment 2   # evaluate the death-spiral segment specifically

Requirements:
    - aiohttp (pip install aiohttp)
    - Access to an OpenAI-compatible LLM API
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Fixture loading
# ---------------------------------------------------------------------------

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
DEFAULT_FIXTURE = FIXTURE_DIR / "death_spiral_compaction.json"


def load_fixture(fixture_path: str | Path) -> dict:
    """Load the compaction fixture.

    Returns dict with keys:
        - description: human-readable description of the fixture
        - source: where the data came from
        - compaction_segments: list of segment dicts, each with:
            - segment_id: identifier
            - description: what this segment represents
            - compacted: list of message dicts that were replaced
            - summary: the summary message that was produced
            - retained: list of messages kept after compaction
    """
    with open(fixture_path) as f:
        return json.load(f)


def _truncate_for_display(msgs: list[dict], max_content: int = 500) -> list[dict]:
    """Truncate message content for display/serialization."""
    result = []
    for m in msgs:
        entry = dict(m)
        if len(entry.get("content", "")) > max_content:
            entry["content"] = entry["content"][:max_content] + "...[truncated]"
        result.append(entry)
    return result


# ---------------------------------------------------------------------------
# 2. Prompt definitions
# ---------------------------------------------------------------------------

CURRENT_PROMPT = (
    "Summarize the following conversation concisely, "
    "preserving key facts, decisions, and context that "
    "would be needed to continue the conversation."
)

IMPROVED_PROMPT = """\
Summarize this conversation for an AI agent that will continue it. Your summary will be the ONLY record of what happened — the original messages will be deleted.

You MUST include:

1. **User's requests and instructions**: What did the user explicitly ask for? Quote key phrases.
2. **Bug/task being investigated**: What problem is being solved? What specific symptoms or errors?
3. **Discoveries so far**: What has been found? Specific code locations, root causes, hypotheses.
4. **Current state of work**: What was the agent actively doing when compaction triggered? What was it about to do?
5. **Decisions made**: What choices were agreed on? What was rejected?
6. **Files/code examined**: Which files were read and what was found in each?

Critical: If the user gave instructions, they MUST appear in the summary. Do NOT say "no user instructions" — check carefully for user messages.

Format: Use clear sections with headers. Be specific (file names, line numbers, function names). Omit conversational filler."""


# ---------------------------------------------------------------------------
# 3. LLM interaction
# ---------------------------------------------------------------------------

async def call_llm(client_session, base_url: str, model: str, api_key: str, messages: list[dict]) -> str:
    """Call the LLM API and return the response content."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,
    }

    url = f"{base_url.rstrip('/')}/chat/completions"
    async with client_session.post(url, headers=headers, json=payload) as resp:
        if resp.status != 200:
            text = await resp.text()
            raise RuntimeError(f"LLM API error {resp.status}: {text[:500]}")
        data = await resp.json()
        return data["choices"][0]["message"]["content"]


async def generate_summary(client_session, base_url: str, model: str, api_key: str,
                           prompt: str, messages: list[dict]) -> str:
    """Generate a summary of messages using the given prompt."""
    # Serialize messages for the summarizer
    # Keep them compact — role + content snippet
    serialized = []
    for m in messages:
        entry = {"role": m["role"]}
        content = m.get("content", "")
        # Truncate very long tool outputs
        if len(content) > 500:
            content = content[:500] + "...[truncated]"
        entry["content"] = content
        serialized.append(entry)

    llm_messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(serialized, indent=2)},
    ]
    return await call_llm(client_session, base_url, model, api_key, llm_messages)


# ---------------------------------------------------------------------------
# 4. Evaluation scoring
# ---------------------------------------------------------------------------

EVAL_PROMPT = """\
You are evaluating a conversation summary. The summary was produced by compacting (replacing) earlier messages in an ongoing conversation. A fresh AI agent will receive ONLY this summary plus a few recent messages.

Based on the summary below, answer these questions. For each, score 1 (missing/wrong), 2 (partial), or 3 (complete and accurate):

Q1. Does the summary identify what the user asked the agent to do?
Q2. Does the summary identify the specific bug being investigated (duplicate LLM responses / re-entrant tool completion)?
Q3. Does the summary mention the root cause or key discoveries (e.g., re-entrant _process_queue_item, pending_tool_call_ids)?
Q4. Does the summary mention what code changes were made or planned?
Q5. Does the summary preserve the user's instructions (NOT say "no user instructions")?

Also answer:
- What bug is being investigated?
- What has been discovered so far?
- What was the agent about to do next?

SUMMARY:
{summary}

RETAINED MESSAGES (still visible to agent):
{retained}

Respond in this JSON format:
{{
    "scores": {{
        "user_requests": <1-3>,
        "bug_identification": <1-3>,
        "discoveries": <1-3>,
        "code_changes": <1-3>,
        "user_instructions_preserved": <1-3>
    }},
    "total": <5-15>,
    "bug_identified": "<what bug>",
    "discoveries": "<what was found>",
    "next_action": "<what was about to happen>",
    "critical_failures": ["<list of serious omissions>"]
}}"""


async def score_summary(client_session, base_url: str, model: str, api_key: str,
                        summary: str, retained: list[dict]) -> dict:
    """Score a summary using an LLM evaluator."""
    retained_text = "\n".join(
        f"  [{m['role']}] {m.get('content', '')[:200]}"
        for m in retained[:10]  # Cap at 10 retained messages for the evaluator
    )

    eval_messages = [
        {
            "role": "user",
            "content": EVAL_PROMPT.format(
                summary=summary,
                retained=retained_text or "(none)",
            ),
        }
    ]

    response = await call_llm(client_session, base_url, model, api_key, eval_messages)

    # Parse JSON response — handle markdown code fences
    response = response.strip()
    if response.startswith("```"):
        response = response.split("\n", 1)[1] if "\n" in response else response[3:]
    if response.endswith("```"):
        response = response[:-3]
    response = response.strip()
    if response.startswith("json"):
        response = response[4:].strip()

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {
            "scores": {},
            "total": 0,
            "parse_error": True,
            "raw_response": response[:500],
        }


# ---------------------------------------------------------------------------
# 5. Main evaluation loop
# ---------------------------------------------------------------------------

async def run_evaluation(args):
    """Run the full evaluation: load fixture → summarize → score."""
    import aiohttp

    # Load fixture
    print(f"📖 Loading fixture from {args.fixture}...")
    fixture = load_fixture(args.fixture)
    segments = fixture["compaction_segments"]
    print(f"   {fixture['description']}")
    print(f"   Source: {fixture['source']}")
    print(f"   Found {len(segments)} compaction segments")

    # Select segment
    seg_idx = args.segment - 1  # 1-indexed
    if seg_idx < 0 or seg_idx >= len(segments):
        print(f"❌ Segment {args.segment} not found (available: 1-{len(segments)})")
        return

    segment = segments[seg_idx]
    compacted = segment["compacted"]
    retained = segment["retained"]
    original_summary = segment["summary"]["content"]

    print(f"\n📋 Evaluating segment: {segment['segment_id']}")
    if segment.get("description"):
        print(f"   {segment['description']}")
    print(f"   Compacted messages: {len(compacted)}")
    print(f"   Retained messages: {len(retained)}")
    print(f"   Original summary length: {len(original_summary)} chars")

    # Show ground truth: what user messages are in the compacted segment
    user_msgs = [m for m in compacted if m["role"] == "user"]
    print(f"\n   USER MESSAGES IN COMPACTED SEGMENT (these must be preserved):")
    for um in user_msgs:
        print(f"     #{um['id']}: {um['content'][:100]}...")

    # Show key discoveries in compacted segment
    assistant_msgs = [m for m in compacted if m["role"] == "assistant"]
    print(f"\n   ASSISTANT MESSAGES WITH DISCOVERIES:")
    for am in assistant_msgs:
        if any(kw in am.get("content", "").lower() for kw in ["bug", "root cause", "re-entrant", "duplicate", "_process_queue_item"]):
            print(f"     #{am['id']}: {am['content'][:150]}...")

    # Generate summaries with both prompts
    async with aiohttp.ClientSession() as session:
        results = {}

        for label, prompt in [("CURRENT", CURRENT_PROMPT), ("IMPROVED", IMPROVED_PROMPT)]:
            print(f"\n{'='*70}")
            print(f"🤖 Generating summary with {label} prompt...")
            start = time.time()
            try:
                summary = await generate_summary(
                    session, args.base_url, args.model, args.api_key,
                    prompt, compacted,
                )
                elapsed = time.time() - start
                print(f"   Generated in {elapsed:.1f}s ({len(summary)} chars)")

                # Score the summary
                print(f"   Scoring summary...")
                scores = await score_summary(
                    session, args.base_url, args.model, args.api_key,
                    summary, retained,
                )

                results[label] = {
                    "summary": summary,
                    "scores": scores,
                    "elapsed": elapsed,
                    "summary_length": len(summary),
                }
            except Exception as e:
                print(f"   ❌ Error: {e}")
                results[label] = {"error": str(e)}

        # Also score the ORIGINAL summary from the actual death spiral
        print(f"\n{'='*70}")
        print(f"📜 Scoring ORIGINAL summary (from the actual death spiral)...")
        # Strip the "[Summary of earlier conversation]\n" prefix for evaluation
        orig_clean = original_summary
        if orig_clean.startswith("[Summary of earlier conversation]"):
            orig_clean = orig_clean.split("\n", 1)[1] if "\n" in orig_clean else orig_clean

        try:
            orig_scores = await score_summary(
                session, args.base_url, args.model, args.api_key,
                orig_clean, retained,
            )
            results["ORIGINAL"] = {
                "summary": original_summary,
                "scores": orig_scores,
                "summary_length": len(original_summary),
            }
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results["ORIGINAL"] = {"error": str(e)}

    # Print comparison
    print(f"\n{'='*70}")
    print("📊 EVALUATION RESULTS")
    print(f"{'='*70}")

    for label in ["ORIGINAL", "CURRENT", "IMPROVED"]:
        r = results.get(label, {})
        if "error" in r:
            print(f"\n{label}: ERROR - {r['error']}")
            continue

        scores = r.get("scores", {})
        total = scores.get("total", 0)
        individual = scores.get("scores", {})

        print(f"\n{label} (total: {total}/15):")
        if individual:
            for k, v in individual.items():
                bar = "█" * v + "░" * (3 - v)
                print(f"  {k:30s} {bar} {v}/3")

        failures = scores.get("critical_failures", [])
        if failures:
            print(f"  ⚠️  Critical failures:")
            for f in failures:
                print(f"     - {f}")

        bug = scores.get("bug_identified", "?")
        print(f"  Bug identified: {bug[:100]}")

    # Summary comparison
    print(f"\n{'='*70}")
    print("📝 SUMMARY TEXTS (first 300 chars each)")
    print(f"{'='*70}")
    for label in ["ORIGINAL", "CURRENT", "IMPROVED"]:
        r = results.get(label, {})
        summary = r.get("summary", "(no summary)")
        print(f"\n{label}:")
        print(f"  {summary[:300]}...")
        print(f"  [{len(summary)} total chars]")

    # Save full results to JSON
    output_path = Path(args.fixture).parent / "eval_results.json"
    with open(output_path, "w") as f:
        # Make results serializable
        clean_results = {}
        for label, r in results.items():
            clean = dict(r)
            if "elapsed" in clean:
                clean["elapsed"] = round(clean["elapsed"], 2)
            clean_results[label] = clean
        json.dump(clean_results, f, indent=2, default=str)
    print(f"\n💾 Full results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate compaction prompt quality")
    parser.add_argument(
        "--fixture", default=str(DEFAULT_FIXTURE),
        help=f"Path to compaction fixture JSON (default: {DEFAULT_FIXTURE})",
    )
    parser.add_argument(
        "--segment", type=int, default=1,
        help="Compaction segment to evaluate (1-indexed, default: 1)",
    )
    parser.add_argument("--base-url", default=None, help="LLM API base URL")
    parser.add_argument("--model", default=None, help="LLM model name")
    parser.add_argument("--api-key", default=None, help="LLM API key")
    args = parser.parse_args()

    # Load defaults from agent.yaml if not specified
    if not args.base_url or not args.model:
        config_path = Path(__file__).parent.parent / "agent.yaml"
        if config_path.exists():
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
            llm_config = config.get("llm", {}).get("main", {})
            if not args.base_url:
                args.base_url = llm_config.get("base_url")
            if not args.model:
                args.model = llm_config.get("model")
            if not args.api_key:
                args.api_key = llm_config.get("api_key")

    if not args.base_url or not args.model:
        print("❌ Must specify --base-url and --model (or have agent.yaml)")
        sys.exit(1)

    print(f"🌐 LLM: {args.model} @ {args.base_url}")
    asyncio.run(run_evaluation(args))


if __name__ == "__main__":
    main()
