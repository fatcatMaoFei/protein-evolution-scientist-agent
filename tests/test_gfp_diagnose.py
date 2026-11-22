import json

import pytest

from gfp_diagnose import diagnose_arm, pool_diagnostics, random_hit_probability


def test_headroom_and_exact_random_baseline_use_strict_thresholds():
    scores = {"WT": 3.0, "incumbent": 3.8, "tie": 3.8, "better": 4.0, "dim": 2.0}
    result, hits = pool_diagnostics(scores, {"initial": ["WT", "incumbent"], "pool": ["tie", "better", "dim"]}, 7, 2)
    assert hits == {"better"}
    assert result["above_wt_count"] == 2
    assert result["max_incumbent_gain"] == pytest.approx(0.2)
    assert result["random_hit_probability"] == pytest.approx(2 / 3)
    result, hits = pool_diagnostics(scores, {"initial": ["WT", "better"], "pool": ["tie", "incumbent", "dim"]}, 7, 2)
    assert not hits
    assert result["pool_best_minus_initial"] == pytest.approx(-0.2)
    assert result["max_incumbent_gain"] == result["random_hit_probability"] == 0
    assert random_hit_probability(3, 2, 2) == 1
    with pytest.raises(ValueError, match="fit the candidate pool"):
        random_hit_probability(3, 0, 4)


def test_distinguishes_not_offered_from_offered_but_rejected(tmp_path):
    scores = {"WT": 3.0, "chosen": 3.5, "visible": 4.0, "hidden": 4.2}
    records = {
        "round-01-selection.json": {"selected_ids": ["chosen"], "model_only_ids": ["chosen"],
                                    "eligible_rf_ranks": {"chosen": 1, "visible": 2, "hidden": 3}},
        "round-01.json": {"seed": 7, "round": 1, "arm": "C_llm", "selected": [{"id": "chosen", "fitness": 3.5}]},
        "round-01-request.json": {"menu": [{"id": "chosen", "source": "rf_elite"},
                                            {"id": "visible", "source": "seeded_non_elite"}]},
    }
    for filename, record in records.items():
        (tmp_path / filename).write_text(json.dumps(record))
    inputs = {"pool": {"chosen", "visible", "hidden"}, "hits": {"visible", "hidden"},
              "headroom": {"initial_best": 3.8, "wild_type": 3.0}, "backend": "demo", "arm": "C_llm",
              "seed": 7, "rounds": 1, "budget": 1}
    row, rounds = diagnose_arm(scores, tmp_path, **inputs)
    assert row["above_initial_not_accessible_to_rule"] == 1
    assert row["above_initial_accessible_but_unselected"] == 1
    assert row["distinct_above_initial_in_llm_rf_elite"] == 0
    assert row["best_rf_rank_among_above_initial"] == 2
    assert row["incumbent_gain"] == 0 and row["above_wt_hit_rate"] == 1
    assert rounds[0]["batch_best"] == rounds[0]["batch_mean"] == 3.5
    with pytest.raises(ValueError, match="Round artifacts disagree"):
        diagnose_arm({**scores, "chosen": 3.4}, tmp_path, **inputs)
