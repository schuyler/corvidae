#!/usr/bin/env python3
"""llama-server cache-ram swap-cost probe (#231).

Measures what it costs to switch the single slot between prompts when
llama-server keeps the displaced prompt in its host-RAM prompt cache
(--cache-ram): restore time against prompt size in both directions, how many
prompts of each size stay resident, catch-up after a restore, and the first
switch after compaction. Talks to llama-server alone (the router's /models
endpoints plus /v1/chat/completions); no corvidae daemon is involved.

It unloads and reloads the model before each unit, so the agent using that
server must be stopped first. Writes run.json, requests.jsonl and report.md
into --out. See harness/README.md.

    python -m harness.cache_swap --base-url http://127.0.0.1:8080 \
        --model swift-1.5 --out DIR
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

UNIT_GROUPS = ("r1-standalone", "r1-effort", "r2", "r3", "r4")
CATCHUP_N = (1, 5, 20)

DEFAULT_SIZES = "16384,65536,131072,160000"
STANDALONE_TOKENS = 4096
R3_TOKENS = 65536
R4_TOKENS = 131072
R4_RETENTION = 0.40
R4_SUMMARY_WORDS = 1500

# A cold prefill of ~160k tokens can take several minutes.
HTTP_TIMEOUT_S = 3600
# Bounds the wait for the router to finish an unload or a load.
RESET_TIMEOUT_S = 1800
POLL_INTERVAL_S = 2

USER_WORDS = 110
ASSISTANT_WORDS = 260

WORDS = (
    "river stone lantern copper window garden orbit felt harbor maple "
    "signal ember valley compass thread marble quiet anchor meadow kettle "
    "falcon prism cedar ladder velvet summit pebble tundra beacon saddle "
    "willow cobalt furnace mosaic trellis granite lagoon bramble cipher "
    "juniper quarry sparrow timber vessel walnut zephyr almond bronze "
    "canyon dune"
).split()


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def parse_cache_ram(args: list[str]) -> int:
    """Return the --cache-ram limit in MiB from the model's launch args."""
    # Both spellings mean the same thing, so neither may be reported as
    # "missing from the preset" — that sends the operator to the wrong file.
    raw: str | None = None
    for a in args:
        if a == "--cache-ram":
            i = args.index(a)
            if i + 1 >= len(args):
                raise ValueError(
                    "--cache-ram requires a value (a positive MiB limit), but it "
                    "is the last argument"
                )
            raw = args[i + 1]
            break
        if a.startswith("--cache-ram="):
            raw = a.split("=", 1)[1]
            break
    if raw is None:
        raise ValueError(
            "--cache-ram is missing from the model's status.args; set it to a "
            "positive MiB limit in the model preset (the server default is "
            "not reported, and residency arithmetic needs a finite limit)"
        )
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(
            f"--cache-ram is {raw!r}, not an integer; set a positive MiB "
            "limit in the model preset"
        ) from None
    if value <= 0:
        raise ValueError(
            f"--cache-ram is {value}; set a positive MiB limit in the model "
            "preset (the residency arithmetic needs a finite limit)"
        )
    return value


def check_single_slot(props: dict) -> None:
    """Refuse to run unless the server has exactly one slot."""
    slots = props.get("total_slots")
    if slots != 1:
        raise ValueError(
            f"/props total_slots is {slots!r}; set parallel = 1 in the model "
            "preset (other slots would hold prompts and skew every count)"
        )


def parse_units(spec: str | None) -> list[str]:
    """Return the unit groups to run, in UNIT_GROUPS order."""
    if spec is None:
        return list(UNIT_GROUPS)
    wanted = [s.strip() for s in spec.split(",") if s.strip()]
    for group in wanted:
        if group not in UNIT_GROUPS:
            raise ValueError(
                f"--units group {group!r} is unknown; use a comma-separated "
                f"subset of: {', '.join(UNIT_GROUPS)}"
            )
    return [g for g in UNIT_GROUPS if g in wanted]


def residency_fill_count(cache_ram_mib: int, size_tokens: int,
                         kib_per_token: float) -> int:
    """Number of prompts to fill so the fill is expected to evict once."""
    return max(2, math.floor(cache_ram_mib * 1024 / (size_tokens * kib_per_token)) + 2)


def residency_bounds(cache_ram_mib: int, hits: int, missed: bool) -> dict:
    """Resident count and entry-size bounds from the newest-first probe."""
    c = hits + 1
    upper = cache_ram_mib / c
    return {
        "C": c,
        "lower_mib": cache_ram_mib / (c + 1) if missed else None,
        "upper_mib": upper,
        "k2_mib": 2 * upper,
    }


def derive_row(row: dict) -> dict:
    """Add overhead_ms and hit to a requests.jsonl row."""
    out = dict(row)
    out["overhead_ms"] = row["wall_ms"] - row["prompt_ms"] - row["predicted_ms"]
    expected = row["expected_cached"]
    out["hit"] = None if expected == 0 else row["cached_tokens"] >= expected / 2
    return out


def _filler(key: str, words: int) -> str:
    """Deterministic filler text of `words` words for a key."""
    rng = random.Random(key)
    return " ".join(rng.choice(WORDS) for _ in range(words))


def build_messages(seed: str, n_turns: int, tail_words: int) -> list[dict]:
    """System message, n_turns user/assistant pairs, then a final user turn."""
    # The seed line comes first so distinct prompts diverge early.
    msgs = [{
        "role": "system",
        "content": f"Seed {seed}.\nYou are a careful assistant in a long "
                   "conversation. " + _filler(f"{seed}:system", 60),
    }]
    # Alternating filler turns.
    for i in range(n_turns):
        msgs.append({"role": "user",
                     "content": _filler(f"{seed}:u{i}", USER_WORDS)})
        msgs.append({"role": "assistant",
                     "content": _filler(f"{seed}:a{i}", ASSISTANT_WORDS)})
    # The last turn's length is the fine-sizing knob.
    msgs.append({"role": "user",
                 "content": _filler(f"{seed}:tail", tail_words)})
    return msgs


def fit_messages(seed: str, target_tokens: int,
                 count_tokens: Callable[[list[dict]], int]) -> list[dict]:
    """Build messages whose counted size is within 1% of target_tokens."""
    # Largest turn count that stays at or under the target.
    lo, hi = 0, 1
    while count_tokens(build_messages(seed, hi, USER_WORDS)) <= target_tokens:
        lo, hi = hi, hi * 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if count_tokens(build_messages(seed, mid, USER_WORDS)) <= target_tokens:
            lo = mid
        else:
            hi = mid
    n_turns = lo
    # Largest tail length that stays at or under the target.
    t_lo, t_hi = 1, USER_WORDS + ASSISTANT_WORDS + USER_WORDS + 1
    while t_hi - t_lo > 1:
        mid = (t_lo + t_hi) // 2
        if count_tokens(build_messages(seed, n_turns, mid)) <= target_tokens:
            t_lo = mid
        else:
            t_hi = mid
    msgs = build_messages(seed, n_turns, t_lo)
    got = count_tokens(msgs)
    if abs(got - target_tokens) > target_tokens * 0.01:
        raise ValueError(
            f"fit_messages seed {seed!r} reached {got} tokens for target "
            f"{target_tokens}, outside 1%; check that count_tokens grows "
            "with message length"
        )
    return msgs


def extend_messages(messages: list[dict], n_turns: int, seed: str) -> list[dict]:
    """Append n_turns (assistant, user) pairs, leaving the input prefix intact."""
    out = [dict(m) for m in messages]
    for i in range(n_turns):
        out.append({"role": "assistant",
                    "content": _filler(f"{seed}:ea{i}", ASSISTANT_WORDS)})
        out.append({"role": "user",
                    "content": _filler(f"{seed}:eu{i}", USER_WORDS)})
    return out


def compacted_messages(messages: list[dict], retention: float,
                       summary_words: int, seed: str) -> list[dict]:
    """System message, a summary user turn, then the last share of the turns."""
    turns = messages[1:]
    keep = round(retention * len(turns))
    # The tail must start on an assistant turn so roles keep alternating
    # after the summary (a user turn).
    while keep > 0 and turns[len(turns) - keep]["role"] != "assistant":
        keep -= 1
    summary = {"role": "user",
               "content": f"Summary {seed}. " + _filler(f"{seed}:summary", summary_words)}
    tail = turns[len(turns) - keep:] if keep else []
    return [messages[0], summary] + [dict(m) for m in tail]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _num(x: float) -> str:
    """Plain number: no separators, no decimals."""
    return f"{x:.0f}"


def _table(headers: list[str], body: list[list[str]]) -> list[str]:
    """Markdown table lines."""
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return lines


def _units(rows: list[dict], group: str) -> dict[int, dict[str, dict]]:
    """Rows of a group as {size_or_n: {label: derived row}}, key ascending."""
    out: dict[int, dict[str, dict]] = {}
    for row in rows:
        g, _, key = row["unit"].rpartition("-")
        if g == group:
            out.setdefault(int(key), {}).setdefault(row["label"], row)
    return dict(sorted(out.items()))


def _r2_units(rows: list[dict]) -> dict[int, list[dict]]:
    """R2 rows per size, in request order."""
    out: dict[int, list[dict]] = {}
    for row in rows:
        g, _, key = row["unit"].rpartition("-")
        if g == "r2":
            out.setdefault(int(key), []).append(row)
    return dict(sorted(out.items()))


def _r1_section(rows: list[dict], kind: str) -> list[str]:
    """R1 table for one kind of other prompt."""
    title = {"standalone": "R1 restore cost, standalone other prompt (~4k)",
             "effort": "R1 restore cost, effort-switch other prompt (medium vs xhigh)"}[kind]
    body = []
    for size, labels in _units(rows, f"r1-{kind}").items():
        for label, cold_label, direction in (
                ("restore_A", "cold_A", "main > other > main"),
                ("restore_B", "cold_B", "reverse")):
            if label not in labels or cold_label not in labels:
                continue
            r, cold = labels[label], labels[cold_label]
            body.append([
                str(size), direction, _num(r["wall_ms"]), _num(r["prompt_ms"]),
                _num(r["overhead_ms"]),
                f"{r['cached_tokens']}/{r['prompt_tokens']}", str(r["hit"]),
                _num(cold["prompt_ms"]), _num(cold["wall_ms"]),
            ])
    headers = ["size", "direction", "wall ms", "prompt_ms", "overhead ms",
               "cached/prompt tokens", "hit", "cold prompt_ms", "cold wall ms"]
    return [f"## {title}", ""] + _table(headers, body) + [""]


def _r2_section(run: dict, rows: list[dict]) -> list[str]:
    """R2 table and per-size notes."""
    cache_ram = run["cache_ram_mib"]
    body, notes = [], []
    for size, unit_rows in _r2_units(rows).items():
        d = [derive_row(r) for r in unit_rows]
        fills = [r for r in d if r["label"].startswith("fill_")]
        probes = [r for r in d if r["label"].startswith("probe_")]
        hits = sum(1 for r in probes if r["hit"])
        miss = next((r for r in probes if r["hit"] is False), None)
        b = residency_bounds(cache_ram, hits, miss is not None)
        m = len(fills)
        # A group can hold a probe and no fill row (interrupted or partially
        # skipped run); dividing here would discard the whole run at print time.
        cold_mean_cell = _num(sum(r["prompt_ms"] for r in fills) / m) if m else "-"
        if miss is None:
            bounds = f"<= {_num(b['upper_mib'])}"
            notes.append(
                f"Size {size}: all probes hit, so >= {m} resident (a lower "
                f"bound). --kib-per-token was too high for this size; rerun "
                "with a smaller value.")
        else:
            bounds = f"{_num(b['lower_mib'])} < entry <= {_num(b['upper_mib'])}"
        body.append([
            str(size), str(m), str(b["C"]), bounds, _num(b["k2_mib"]),
            _num(miss["prompt_ms"]) if miss else "-",
            _num(miss["wall_ms"]) if miss else "-",
            cold_mean_cell,
        ])
    headers = ["size", "m", "C", "entry MiB", "cache-ram MiB for k = 2",
               "evicted prompt_ms", "evicted wall ms", "cold prompt_ms (mean)"]
    notes += ["Prompt sizes vary by up to 1% within a unit, so the bounds are "
              "approximate to that degree.",
              "A re-saved entry can gain a context checkpoint while its tail "
              "is reprocessed; at a tight fit that can evict one resident "
              "entry and undercount C by 1."]
    return (["## R2 residency per size", ""] + _table(headers, body) + [""]
            + [f"- {n}" for n in notes] + [""])


def _r3_section(rows: list[dict]) -> list[str]:
    """R3 catch-up table."""
    body = []
    for n, labels in _units(rows, "r3").items():
        if not {"cold_A0", "catchup", "cold_AN"} <= labels.keys():
            continue
        cu = derive_row(labels["catchup"])
        cold = labels["cold_AN"]
        body.append([
            str(n), str(cu["prompt_n"]), _num(cu["wall_ms"]),
            _num(cu["prompt_ms"]), str(cu["cached_tokens"]), str(cu["hit"]),
            _num(cold["prompt_ms"]), _num(cold["wall_ms"]),
        ])
    headers = ["N", "new tokens", "catch-up wall ms", "catch-up prompt_ms",
               "cached_tokens", "hit", "cold prompt_ms", "cold wall ms"]
    return ["## R3 catch-up after a restore", ""] + _table(headers, body) + [""]


def _r4_section(rows: list[dict]) -> list[str]:
    """R4 post-compaction table."""
    body = []
    for size, labels in _units(rows, "r4").items():
        if not {"post_compact", "cold_post"} <= labels.keys():
            continue
        pc, cold = labels["post_compact"], labels["cold_post"]
        body.append([
            str(size), _num(pc["wall_ms"]), _num(pc["prompt_ms"]),
            str(pc["cached_tokens"]), _num(cold["wall_ms"]),
            _num(cold["prompt_ms"]),
        ])
    headers = ["window size", "post-compaction wall ms",
               "post-compaction prompt_ms", "cached_tokens",
               "cold wall ms", "cold prompt_ms"]
    return (["## R4 first switch after compaction", ""]
            + _table(headers, body) + [""])


def _window_line(run: dict, rows: list[dict]) -> str | None:
    """The yes/no line for two near-full windows, from R1-effort or R2."""
    effort = _units(rows, "r1-effort")
    if effort:
        size = max(effort)
        restore = effort[size].get("restore_A")
        if restore is not None:
            answer = "yes" if derive_row(restore)["hit"] else "no"
            return (f"Do two near-full windows both fit in the cache at "
                    f"{size} tokens? {answer} (from r1-effort restore_A).")
    r2 = _r2_units(rows)
    if r2:
        size = max(r2)
        probes = [derive_row(r) for r in r2[size] if r["label"].startswith("probe_")]
        c = sum(1 for r in probes if r["hit"]) + 1
        answer = "yes" if c >= 2 else "no"
        return (f"Do two near-full windows both fit in the cache at {size} "
                f"tokens? {answer} (from r2 resident count C = {c}).")
    return None


def render_report(run: dict, rows: list[dict]) -> str:
    """Markdown report: a server line and one table per group with rows."""
    lines = [
        "# Cache-ram swap-cost probe", "",
        f"Server: {run['base_url']}, model {run['model']}, build "
        f"{run['build_info']}, cache-ram {run['cache_ram_mib']} MiB, "
        f"started {run['started_at']}.", "",
    ]
    # One section per group that has rows.
    derived = [derive_row(r) for r in rows]
    present = {r["unit"].rpartition("-")[0] for r in rows}
    if "r1-standalone" in present:
        lines += _r1_section(derived, "standalone")
    if "r1-effort" in present:
        lines += _r1_section(derived, "effort")
    if "r2" in present:
        lines += _r2_section(run, rows)
    if "r3" in present:
        lines += _r3_section(rows)
    if "r4" in present:
        lines += _r4_section(rows)
    window = _window_line(run, rows)
    if window:
        lines += ["## Near-full windows", "", window, ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTTP and the run
# ---------------------------------------------------------------------------


def http_json(method: str, url: str, body: dict | None = None) -> dict:
    """One JSON request; a non-2xx status raises with the URL and body."""
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        text = e.read().decode(errors="replace")
        raise RuntimeError(
            f"{method} {url} returned HTTP {e.code}: {text}; check that "
            "--base-url is the llama-server router and --model is a model "
            "it knows") from e
    except urllib.error.URLError as e:
        # Connection refused, DNS failure or timeout: no HTTP status to show.
        raise RuntimeError(
            f"{method} {url} failed ({e.reason}); check that llama-server "
            "is running and --base-url points at it") from e


class Probe:
    """Holds the connection details and output files for one run."""

    def __init__(self, base_url: str, model: str, out: Path,
                 kib_per_token: float) -> None:
        self.base = base_url.rstrip("/")
        self.model = model
        self.out = out
        self.kib_per_token = kib_per_token
        self.run: dict | None = None
        self.rows: list[dict] = []
        out.mkdir(parents=True, exist_ok=True)
        self.requests_file = (out / "requests.jsonl").open("w")

    def models_entry(self) -> dict:
        """The router's /models entry for the model."""
        for entry in http_json("GET", f"{self.base}/models")["data"]:
            if entry["id"] == self.model:
                return entry
        raise ValueError(
            f"model {self.model!r} is not in GET {self.base}/models; pass "
            "--model with an id the router lists")

    def wait_for(self, state: str) -> None:
        """Poll /models until the model's status.value is `state`."""
        deadline = time.monotonic() + RESET_TIMEOUT_S
        while self.models_entry()["status"]["value"] != state:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"model {self.model!r} did not reach {state!r} within "
                    f"{RESET_TIMEOUT_S} s; check the router log")
            time.sleep(POLL_INTERVAL_S)

    def reset(self) -> None:
        """Unload and reload the model so the prompt cache starts empty."""
        # The router refuses to unload a model that isn't running (e.g. after
        # a run that died mid-reset), so unload only when it is.
        if self.models_entry()["status"]["value"] != "unloaded":
            http_json("POST", f"{self.base}/models/unload", {"model": self.model})
            self.wait_for("unloaded")
        http_json("POST", f"{self.base}/models/load", {"model": self.model})
        self.wait_for("loaded")

    def capture_run(self) -> None:
        """Write run.json from the freshly loaded model (R5)."""
        props = http_json(
            "GET", f"{self.base}/props?model={urllib.parse.quote(self.model)}")
        check_single_slot(props)
        entry = self.models_entry()
        self.run = {
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "base_url": self.base,
            "model": self.model,
            "props": props,
            "models_entry": entry,
            "cache_ram_mib": parse_cache_ram(entry["status"]["args"]),
            "build_info": props["build_info"],
        }
        (self.out / "run.json").write_text(json.dumps(self.run, indent=2))

    def count_tokens(self, messages: list[dict]) -> int:
        """Rendered prompt length at xhigh, via /apply-template and /tokenize."""
        prompt = http_json("POST", f"{self.base}/apply-template", {
            "model": self.model, "messages": messages,
            "reasoning_effort": "xhigh"})["prompt"]
        tokens = http_json("POST", f"{self.base}/tokenize", {
            "model": self.model, "content": prompt})["tokens"]
        return len(tokens)

    def fit(self, seed: str, target: int) -> list[dict]:
        """Messages sized to target tokens."""
        return fit_messages(seed, target, self.count_tokens)

    def complete(self, messages: list[dict], effort: str,
                 cache_prompt: bool = True) -> tuple[dict, float]:
        """One completion with a minimal decode; returns (response, wall_ms)."""
        t0 = time.perf_counter()
        resp = http_json("POST", f"{self.base}/v1/chat/completions", {
            "model": self.model, "messages": messages, "max_tokens": 1,
            "reasoning_effort": effort, "stream": False,
            "cache_prompt": cache_prompt})
        return resp, (time.perf_counter() - t0) * 1000

    def begin(self, unit: str) -> None:
        """Start a unit: reset, capture run.json once, unlogged warm-up."""
        print(f"unit {unit}", flush=True)
        self.reset()
        if self.run is None:
            self.capture_run()
        # Distinct prefix at medium effort, so it shares nothing with xhigh prompts.
        self.complete([{"role": "user", "content": "warm-up ping"}], "medium")

    def request(self, unit: str, label: str, messages: list[dict],
                expected_cached: int, effort: str = "xhigh",
                cache_prompt: bool = True) -> dict:
        """Send one measured request and append its row to requests.jsonl."""
        resp, wall_ms = self.complete(messages, effort, cache_prompt)
        timings = resp["timings"]
        row = {
            "unit": unit, "label": label,
            "prompt_tokens": resp["usage"]["prompt_tokens"],
            "expected_cached": expected_cached,
            "cached_tokens": resp["usage"]["prompt_tokens_details"]["cached_tokens"],
            "cache_n": timings["cache_n"], "prompt_n": timings["prompt_n"],
            "prompt_ms": timings["prompt_ms"],
            "predicted_ms": timings["predicted_ms"],
            "wall_ms": wall_ms, "effort": effort, "cache_prompt": cache_prompt,
        }
        self.rows.append(row)
        self.requests_file.write(json.dumps(row) + "\n")
        self.requests_file.flush()
        return row

    def unit_r1(self, kind: str, size: int) -> None:
        """R1: cold A, cold B, then restore A and restore B."""
        unit = f"r1-{kind}-{size}"
        self.begin(unit)
        a = self.fit(f"{unit}-A", size)
        if kind == "effort":
            b, b_effort = a, "medium"
        else:
            b, b_effort = self.fit(f"{unit}-B", STANDALONE_TOKENS), "xhigh"
        cold_a = self.request(unit, "cold_A", a, 0)
        cold_b = self.request(unit, "cold_B", b, 0, b_effort)
        self.request(unit, "restore_A", a, cold_a["prompt_tokens"])
        self.request(unit, "restore_B", b, cold_b["prompt_tokens"], b_effort)

    def unit_r2(self, size: int) -> None:
        """R2: fill m prompts, then probe newest-first until the first miss."""
        unit = f"r2-{size}"
        self.begin(unit)
        m = residency_fill_count(self.run["cache_ram_mib"], size, self.kib_per_token)
        prompts = [self.fit(f"{unit}-P{i}", size) for i in range(1, m + 1)]
        fills = [self.request(unit, f"fill_{i}", p, 0)
                 for i, p in enumerate(prompts, 1)]
        # Probe Pm-1 down to P1; Pm is resident by construction.
        for i in range(m - 1, 0, -1):
            row = self.request(unit, f"probe_{i}", prompts[i - 1],
                               fills[i - 1]["prompt_tokens"])
            if derive_row(row)["hit"] is False:
                break

    def unit_r3(self, n: int) -> None:
        """R3: restore A0 extended by n turn pairs, against a cold prefill."""
        unit = f"r3-{n}"
        self.begin(unit)
        a0 = self.fit(f"{unit}-A0", R3_TOKENS)
        b = self.fit(f"{unit}-B", STANDALONE_TOKENS)
        an = extend_messages(a0, n, f"{unit}-ext")
        cold_a0 = self.request(unit, "cold_A0", a0, 0)
        self.request(unit, "cold_B", b, 0)
        self.request(unit, "catchup", an, cold_a0["prompt_tokens"])
        self.request(unit, "cold_AN", an, 0, cache_prompt=False)

    def unit_r4(self) -> None:
        """R4: switch back to a compacted window, against a cold prefill."""
        unit = f"r4-{R4_TOKENS}"
        self.begin(unit)
        a = self.fit(f"{unit}-A", R4_TOKENS)
        b = self.fit(f"{unit}-B", STANDALONE_TOKENS)
        post = compacted_messages(a, R4_RETENTION, R4_SUMMARY_WORDS, f"{unit}-summary")
        self.request(unit, "cold_A", a, 0)
        self.request(unit, "cold_B", b, 0)
        self.request(unit, "post_compact", post, 0)
        self.request(unit, "cold_post", post, 0, cache_prompt=False)


def main() -> None:
    """Parse arguments, run the selected units, write report.md."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--sizes", default=DEFAULT_SIZES,
                    help="comma-separated prompt sizes in tokens")
    ap.add_argument("--kib-per-token", type=float, default=120,
                    help="sizes the residency fill only")
    ap.add_argument("--units", default=None,
                    help=f"comma-separated subset of {', '.join(UNIT_GROUPS)}")
    args = ap.parse_args()

    groups = parse_units(args.units)
    sizes = [int(s) for s in args.sizes.split(",")]
    probe = Probe(args.base_url, args.model, args.out, args.kib_per_token)

    # Run each selected group in design order.
    for group in groups:
        if group in ("r1-standalone", "r1-effort"):
            for size in sizes:
                probe.unit_r1(group.removeprefix("r1-"), size)
        elif group == "r2":
            for size in sizes:
                probe.unit_r2(size)
        elif group == "r3":
            for n in CATCHUP_N:
                probe.unit_r3(n)
        else:
            probe.unit_r4()

    probe.requests_file.close()
    (args.out / "report.md").write_text(render_report(probe.run, probe.rows))
    print(f"wrote {args.out / 'report.md'}")


if __name__ == "__main__":
    main()
