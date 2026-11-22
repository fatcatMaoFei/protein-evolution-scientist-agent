"""Regression checks for the treatment and measurement boundaries."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import nonbio


def test_pinned_data_and_disjoint_split():
    table, source = nonbio.read_dataset(nonbio.DEFAULT_DATA)
    assert source["sha256"] == nonbio.SOURCE_SHA256
    assert source["raw_rows"] == 1030
    assert source["rows_at_28_days"] == 425
    assert len(table) == 416
    assert set(table.age) == {28}
    assert len(table.drop_duplicates(list(nonbio.FEATURES))) == len(table)
    split = nonbio.make_split(list(table.index), 17)
    assert tuple(map(len, split.values())) == (48, 96, 96, 176)
    assert len({recipe_id for ids in split.values() for recipe_id in ids}) == len(table)
    assert split == nonbio.make_split(list(reversed(table.index)), 17)


def test_oracle_failure_is_atomic_and_arm_ledgers_are_independent():
    values = {"a": 80.0, "b": 47.0, "c": 29.0}
    a = nonbio.Oracle(values, budget=2)
    b = nonbio.Oracle(values, budget=2)
    with pytest.raises(ValueError):
        a.reveal(["a", "validation"])
    with pytest.raises(ValueError):
        a.reveal(["a", "a"])
    assert a.reveal(["a", "b"]) == {"a": 80.0, "b": 47.0}
    with pytest.raises(ValueError):
        a.reveal(["a", "c"])
    assert b.reveal(["c", "a"]) == {"c": 29.0, "a": 80.0}


def test_controlled_callback_sees_only_its_own_observed_labels(tmp_path: Path):
    states = {}

    def test_selector(*, eligible, observed, scores, uncertainty, k, seed, round_number, augmented, log_path):
        assert set(observed).isdisjoint(eligible)
        assert set(scores) == set(eligible)
        assert set(uncertainty) == set(eligible)
        assert all(value >= 0 for value in uncertainty.values())
        assert all(set(row) == set(nonbio.FEATURES) for row in eligible.values())
        assert all(set(row) == {*nonbio.FEATURES, "strength"} for row in observed.values())
        assert len(observed) == nonbio.INITIAL_SIZE + (round_number - 1) * k
        ids = sorted(eligible, reverse=augmented)[:k]
        states[augmented, round_number] = (set(eligible), set(observed), set(ids))
        return ids

    output = tmp_path / "unit_fixture"
    metadata = nonbio.run(
        output, selector=test_selector, seeds=(7,), rounds=2, budget=2, workers=2, mode="unit-fixture",
    )
    assert metadata == {"status": "complete", "mode": "unit-fixture", "arms": 4, "measurements": 16}
    split = json.loads((output / "splits" / "seed-7.json").read_text())
    assert states[False, 1][:2] == states[True, 1][:2] == (set(split["pool"]), set(split["initial"]))
    for arm in (False, True):
        selected = states[arm, 1][2]
        other_selected = states[not arm, 1][2]
        assert selected <= states[arm, 2][1]
        assert other_selected <= states[arm, 2][0]

    measured = pd.read_csv(output / "selections.csv")
    table, _ = nonbio.read_dataset(nonbio.DEFAULT_DATA)
    assert set(measured.id) <= set(split["pool"])
    assert all(row.strength_mpa == table.loc[row.id, "strength"] for row in measured.itertuples())
    assert (measured.groupby("arm").id.nunique() == 4).all()
    prediction = pd.read_csv(output / "predictive.csv")
    assert (prediction[prediction.stage == "final_test"].n == 96).all()
    assert len(prediction[prediction.stage == "final_test"]) == len(nonbio.ARMS) * 2
