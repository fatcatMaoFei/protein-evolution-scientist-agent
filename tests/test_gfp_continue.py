import csv
import json
from types import SimpleNamespace

import pytest

from gfp_continue import load_csv, run_user_flow
from gfp_data import GFP_WT


def mutant(position):
    residue = GFP_WT[position - 1]
    return f"{residue}{position}{'A' if residue != 'A' else 'G'}"


@pytest.fixture
def inputs(tmp_path):
    initial = tmp_path / "initial.csv"
    initial.write_text("mutant,fitness\n" + "".join(f"{mutant(i)},{3+i/10}\n" for i in range(1, 5)))
    candidates = tmp_path / "candidates.csv"
    candidates.write_text(f"mutant\n{mutant(5)}\n{mutant(6)}\n")
    return initial, candidates


def test_user_workflow_leaves_unmeasured_scores_empty(tmp_path, inputs):
    class ForbiddenCatalogLabels:
        def __getitem__(self, key):
            raise AssertionError("Researcher workflow must not query catalog labels")

    dataset = SimpleNamespace(wild_type=GFP_WT, scores=ForbiddenCatalogLabels())
    initial, candidates = inputs
    first = tmp_path / "first"
    run_user_flow(dataset, first, stage="init", measured_csv=initial, prior_state=None,
                  batch_csv=None, candidate_csv=candidates, backend="deepseek", use_llm=False)
    rows = list(csv.DictReader((first / "recommendations.csv").open()))
    assert len(rows) == 2
    assert all(row["measured_fitness"] == "" and row["status"] == "prediction_only" for row in rows)

    feedback = tmp_path / "returned.csv"
    feedback.write_text(f"mutant,fitness\n{mutant(5)},3.9\n")
    next_pool = tmp_path / "next.csv"
    next_pool.write_text(f"mutant\n{mutant(6)}\n{mutant(7)}\n")
    second = tmp_path / "second"
    run_user_flow(dataset, second, stage="continue", measured_csv=None, prior_state=first / "state.json",
                  batch_csv=feedback, candidate_csv=next_pool, backend="deepseek", use_llm=False)
    state = json.loads((second / "state.json").read_text())
    assert state["new_measurements"] == [mutant(5)]
    assert state["proposed"] == [mutant(7)]
    assert set(state["outstanding"]) == {mutant(6), mutant(7)}
    assert len(state["observations"]) == 5


def test_csv_rejects_mismatch_and_duplicate(tmp_path):
    table = tmp_path / "measurements.csv"
    table.write_text(f"mutant,fitness,sequence\n{mutant(1)},4,{GFP_WT}\n")
    with pytest.raises(ValueError, match="Sequence does not match"):
        load_csv(table, measured=True)
    table.write_text(f"mutant,fitness\n{mutant(1)},3\n{mutant(1)},4\n")
    with pytest.raises(ValueError, match="Duplicate"):
        load_csv(table, measured=True)
