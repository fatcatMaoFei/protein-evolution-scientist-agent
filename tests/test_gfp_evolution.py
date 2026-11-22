"""Guard GFP acquisition boundaries with lightweight synthetic measurement fixtures."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pandas as pd
import pytest

from gfp_data import GFP_WT, GFPDataset, make_variant
from gfp_evolution import LLM_ARMS, MeasuredGFPOracle, run_evolution


@pytest.fixture(scope="module")
def fixture_catalog() -> GFPDataset:
    """Fabricated scores only for testing state transitions, never for study results."""
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    substitutions = (
        f"{before}{position}{after}"
        for position, before in enumerate(GFP_WT, 1)
        for after in alphabet if before != after
    )
    variants = {name: make_variant(name) for name in itertools.chain(("WT",), itertools.islice(substitutions, 750))}
    scores = {key: 2.0 + (index % 19) / 10 for index, key in enumerate(variants)}
    scores["WT"] = 3.7
    return GFPDataset(GFP_WT, variants, scores, {key: int(x > 2.5) for key, x in scores.items()},
                      {"domain": "test-fixture-only", "fixture": True})


def test_oracle_rejects_invalid_batches_without_consuming_queries() -> None:
    measurements = {"p": 2.1, "q": 3.2, "r": 4.0}
    left, right = (MeasuredGFPOracle(measurements, budget=2) for _ in range(2))
    for invalid in (["p"], ["p", "p"], ["p", "outside"]):
        with pytest.raises(ValueError):
            left.reveal(invalid)
    assert left.reveal(["p", "q"]) == {"p": 2.1, "q": 3.2}
    with pytest.raises(ValueError):
        left.reveal(["p", "r"])
    assert right.reveal(["r", "p"]) == {"r": 4.0, "p": 2.1}


def test_four_arms_keep_independent_observations(tmp_path: Path, fixture_catalog: GFPDataset) -> None:
    seen = {}

    def select(request, *, log_path):
        menu = request["menu"]
        visible = {row["id"]: row["fitness"] for row in request["observed"]}
        assert len(menu) == 48
        assert all("fitness" not in row for row in menu)
        assert all(value == fixture_catalog.scores[key] for key, value in visible.items())
        assert not set(visible) & {row["id"] for row in menu}
        ordered = sorted(menu, key=lambda row: row["rank"], reverse=request["knowledge"])
        selected = [row["id"] for row in ordered[:request["budget"]]]
        seen[(request["knowledge"], request["round_number"])] = {
            "ids": set(visible), "menu": menu, "selected": set(selected),
            "feedback": set(request["latest_feedback_ids"]),
        }
        return selected

    output = tmp_path / "matched"
    status = run_evolution(fixture_catalog, output, selector=select, seeds=(7,), rounds=2, budget=2, workers=2)
    assert status["status"] == "complete"
    assert status["measurements"] == 16
    assert status["arms"] == 4
    assert seen[False, 1]["ids"] == seen[True, 1]["ids"]
    assert seen[False, 1]["menu"] == seen[True, 1]["menu"]
    for knowledge in (False, True):
        before, after = seen[knowledge, 1], seen[knowledge, 2]
        assert after["ids"] - before["ids"] == before["selected"] == after["feedback"]
        assert not seen[not knowledge, 1]["selected"] & after["ids"]

    split = json.loads((output / "splits/seed-7.json").read_text())
    assert sum(map(len, split.values())) == len(set().union(*map(set, split.values())))
    measured = pd.read_csv(output / "selections.csv")
    assert set(measured.id) <= set(split["pool"])
    assert all(row.fitness == fixture_catalog.scores[row.id] for row in measured.itertuples())
    predictions = pd.read_csv(output / "predictive.csv")
    assert len(predictions[predictions.stage == "final_test"]) == 8


def test_failed_second_round_preserves_only_already_committed_feedback(
    tmp_path: Path, fixture_catalog: GFPDataset,
) -> None:
    def select(request, *, log_path):
        if request["knowledge"] and request["round_number"] == 2:
            return ["outside", request["menu"][0]["id"]]
        return [item["id"] for item in request["menu"][:request["budget"]]]

    output = tmp_path / "failed"
    result = run_evolution(
        fixture_catalog, output, selector=select, arms=LLM_ARMS, seeds=(7,), rounds=2, budget=2, workers=2,
    )
    assert result["status"] == "partial"
    assert result["measurements"] == 4
    assert result["partial_measurements"] == 2
    failed_arm = output / "arms/seed-7/D_knowledge_llm"
    assert (failed_arm / "round-01.json").is_file()
    assert not (failed_arm / "round-02.json").exists()
    assert (pd.read_csv(output / "partial_selections.csv")["round"] == 1).all()
