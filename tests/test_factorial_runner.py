"""The paired add-on preserves the fourth-arm archive and first-round design."""

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from factorial_runner import LLM_ARMS, run
from llm_agent import _ordered_menu
from nonbio import write_json

BASELINE_SMOKE = Path(__file__).resolve().parents[1] / "evidence" / "smoke"
UNIT_MODEL = "unit-test-selector"


def test_factorial_runs_only_paired_llm_arms(tmp_path):
    baseline_files = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in BASELINE_SMOKE.rglob("*") if path.is_file()
    }
    freeze = tmp_path / "fixture-protocol.md"
    freeze.write_text("Offline test fixture with a mock policy.\n", encoding="utf-8")
    chosen = {}

    def fixture_selector(*, eligible, observed, scores, uncertainty, k, seed, round_number, augmented, log_path):
        menu = _ordered_menu(eligible, scores, uncertainty, seed, round_number)
        assert all(set(row["inputs"]).isdisjoint({"strength", "replicates"}) for row in menu)
        candidates = (menu[-k:] if augmented else menu[:k])
        picked = [row["id"] for row in candidates]
        chosen[augmented] = picked
        write_json(log_path, {
            "status": "ok", "model": UNIT_MODEL, "seed": seed, "round": round_number,
            "budget": k, "knowledge_augmented": augmented, "menu": menu, "decision": {"selected_ids": picked},
        })
        return picked

    evidence = tmp_path / "factorial"
    status = run(
        evidence, baseline=BASELINE_SMOKE, selector=fixture_selector, treatment_model=UNIT_MODEL,
        freeze_path=freeze, smoke=True, workers=2,
    )
    assert status == {"status": "complete", "mode": "smoke", "arms": 2, "measurements": 10}
    assert (evidence / "source.json").read_bytes() == (BASELINE_SMOKE / "source.json").read_bytes()
    assert (evidence / "splits" / "seed-7.json").read_bytes() == (BASELINE_SMOKE / "splits" / "seed-7.json").read_bytes()
    config = json.loads((evidence / "config.json").read_text())
    assert config["model"] == UNIT_MODEL
    assert config["frozen_protocol_sha256"] == hashlib.sha256(freeze.read_bytes()).hexdigest()
    assert tuple(config["arms"]) == LLM_ARMS
    assert len(chosen) == 2
    assert set(chosen[False]).isdisjoint(chosen[True])
    assert pd.read_csv(evidence / "selections.csv").groupby("arm").size().to_dict() == dict.fromkeys(LLM_ARMS, 5)
    assert {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in BASELINE_SMOKE.rglob("*") if path.is_file()
    } == baseline_files


def test_factorial_rejects_split_mismatch_before_invoking_selector(tmp_path):
    baseline = tmp_path / "archive"
    shutil.copytree(BASELINE_SMOKE, baseline)
    exact_split = baseline / "splits" / "seed-7.json"
    split = json.loads(exact_split.read_text())
    split["pool"][0], split["test"][0] = split["test"][0], split["pool"][0]
    write_json(exact_split, split)
    freeze = tmp_path / "fixture-protocol.md"
    freeze.write_text("Offline fixture.\n", encoding="utf-8")
    evidence = tmp_path / "factorial"

    def uncalled(**kwargs):
        pytest.fail("Broken paired fixture reached the selector.")

    with pytest.raises(ValueError, match="split differs"):
        run(
            evidence, baseline=baseline, selector=uncalled, treatment_model=UNIT_MODEL,
            freeze_path=freeze, smoke=True,
        )
    assert not evidence.exists()
