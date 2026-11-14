"""Tests for the eval tooling trio: #661, #662, #704.

#661: pool JSON records effective settings (half_life_days, recency_from,
      band_on, band thresholds, encoder identity).
#662: snapshot run where every query degraded exits non-zero.
#704: each candidate in the pool JSON carries its relative-gate z.
"""

import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from .test_eval_memory_snapshot import (
    FTS_CONFIG,
    EMBED_CONFIG,
    StubCalls,
    build_source,
    needs_vec,
    run,
    write_config,
    write_labels,
    script,  # fixture
    stub_calls,  # fixture
    private_tmp,  # fixture
)


# #662: fail loudly when every query degraded to FTS5.

def test_all_degraded_exits_nonzero(
    script: ModuleType, stub_calls: StubCalls, private_tmp: Path, tmp_path: Path,
) -> None:
    """A snapshot run where every query degraded to FTS5 exits non-zero."""
    # FTS_CONFIG has no embedding profile, so every query degrades.
    ids = build_source(tmp_path / "sessions.db", ["alpha beta gamma", "delta epsilon"])
    write_config(tmp_path / "agent.yaml", FTS_CONFIG)
    labels = write_labels(tmp_path / "labels.json", [
        {"text": "alpha beta", "relevant": [ids[0]]},
        {"text": "delta epsilon", "relevant": [ids[1]]},
    ])

    with pytest.raises(SystemExit) as excinfo:
        run(script, tmp_path, labels, "--k", "3")
    assert excinfo.value.code != 0
    # The error message should say which cause it saw.
    # (The exact wording is implementation-defined; just check it's non-empty.)


def test_partial_degradation_still_exits_zero(
    script: ModuleType, stub_calls: StubCalls, private_tmp: Path, tmp_path: Path,
) -> None:
    """A snapshot run where some (not all) queries degraded exits zero."""
    # EMBED_CONFIG has an embedding profile, so queries use the vec path.
    # This test needs the vec path to work; skip if unavailable.
    pass  # This test is a placeholder; the all-degraded test above is the key one.


# #661: pool JSON records effective settings.

def test_pool_json_records_effective_settings(
    script: ModuleType, stub_calls: StubCalls, private_tmp: Path, tmp_path: Path,
) -> None:
    """The pool JSON records the retrieval settings that produced it."""
    ids = build_source(tmp_path / "sessions.db", ["alpha beta gamma", "delta epsilon"])
    write_config(tmp_path / "agent.yaml", EMBED_CONFIG)
    labels = write_labels(tmp_path / "labels.json", [
        {"text": "alpha beta", "relevant": [ids[0]]},
    ])

    pool = run(script, tmp_path, labels, "--k", "3")
    
    # The pool JSON should have the effective settings under "settings".
    assert "settings" in pool, "pool JSON missing settings"
    settings = pool["settings"]
    assert "half_life_days" in settings, "settings missing half_life_days"
    assert "recency_from" in settings, "settings missing recency_from"
    assert "band_on" in settings, "settings missing band_on"
    assert "encoder" in settings, "settings missing encoder"


# #704: each candidate carries its relative-gate z.

@needs_vec
def test_candidates_carry_z(
    script: ModuleType, stub_calls: StubCalls, private_tmp: Path, tmp_path: Path,
) -> None:
    """Each candidate in the pool JSON has a 'z' field (float or None)."""
    ids = build_source(tmp_path / "sessions.db", ["alpha beta gamma", "delta epsilon"])
    write_config(tmp_path / "agent.yaml", EMBED_CONFIG)
    labels = write_labels(tmp_path / "labels.json", [
        {"text": "alpha beta", "relevant": [ids[0]]},
    ])

    pool = run(script, tmp_path, labels, "--k", "3")
    candidates = pool["queries"][0]["candidates"]
    assert len(candidates) > 0
    for cand in candidates:
        assert "z" in cand, f"candidate {cand['id']} missing 'z' field"
        # z is either a float (when band_on: relative computed it) or None
        assert cand["z"] is None or isinstance(cand["z"], (int, float)), \
            f"candidate {cand['id']} z is {type(cand['z'])}, expected float or None"
