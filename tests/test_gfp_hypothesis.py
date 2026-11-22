import copy

import pytest

from gfp_hypothesis import evaluate_hypotheses, make_hypothesis_plan


@pytest.fixture
def trial():
    observed = {"WT": 3.7192121319, "E5G:T108I": 3.18966331858}
    decision = {
        "hypothesis": "The tested variants will exceed wild type fluorescence.",
        "evidence_ids": ["WT", "E5G:T108I"],
        "selected_candidates": [
            {"id": "I47V:N144S:I161V", "prediction": 3.85},
            {"id": "I47V:I161V:Y200C", "prediction": 3.55},
            {"id": "I47V:M88V:I161V", "prediction": 2.85},
        ],
        "hypothesis_test": {"reference_id": "WT", "candidate_ids": ["I47V:N144S:I161V", "I47V:I161V:Y200C"],
                            "min_exceed": 1},
    }
    return decision, observed


def build(decision, observed):
    return make_hypothesis_plan(round_number=3, decision=decision, observations=observed, decision_sha256="d" * 64)


def test_partial_trial_keeps_unmeasured_values_pending_even_if_target_already_met(trial):
    decision, observed = trial
    plan = build(decision, observed)
    saved = copy.deepcopy(plan)
    updated = observed | {"I47V:N144S:I161V": 3.95}
    [result] = evaluate_hypotheses([plan], updated)
    assert result["status"] == "pending"
    assert result["n_measured"] == result["n_exceed"] == 1
    first, second = result["candidate_results"]
    assert first["residual"] == pytest.approx(0.1)
    assert first["forecast_outcome"] == "agreement"
    assert second["measured_fitness"] is second["residual"] is second["forecast_outcome"] is None
    assert plan == saved


def test_completed_trial_support_and_residual_directions_use_frozen_predictions(trial):
    decision, observed = trial
    plan = build(decision, observed)
    # The first candidate equals the reference, so only the second exceeds it.
    updated = observed | {"I47V:N144S:I161V": observed["WT"], "I47V:I161V:Y200C": 3.9}
    [result] = evaluate_hypotheses([plan], updated)
    assert result["status"] == "supported"
    assert result["n_measured"] == 2
    assert result["n_exceed"] == 1
    first, second = result["candidate_results"]
    assert first["forecast_outcome"] == "false_positive"
    assert first["residual"] < 0
    assert second["forecast_outcome"] == "false_negative"
    assert second["residual"] > 0
    assert result["threshold"] == observed["WT"]


def test_completed_trial_refutes_when_no_planned_measurement_exceeds_reference(trial):
    decision, observed = trial
    plan = build(decision, observed)
    updated = observed | {"I47V:N144S:I161V": 3.3, "I47V:I161V:Y200C": 3.55, "I47V:M88V:I161V": 4.5}
    [result] = evaluate_hypotheses([plan], updated)
    assert result["status"] == "refuted"
    assert result["n_exceed"] == 0
    assert "I47V:M88V:I161V" not in {row["id"] for row in result["candidate_results"]}


@pytest.mark.parametrize(("edit", "error"), [
    ({"reference_id": "I47V"}, "observed and cited"),
    ({"candidate_ids": ["I47V"]}, "selected and unmeasured"),
    ({"min_exceed": 3}, "between one"),
    ({"min_exceed": True}, "between one"),
    ({"candidate_ids": ["I47V:N144S:I161V"] * 2}, "distinct canonical"),
])
def test_plan_rejects_impossible_or_untraceable_criteria(trial, edit, error):
    decision, observed = trial
    decision["hypothesis_test"].update(edit)
    with pytest.raises(ValueError, match=error):
        build(decision, observed)


def test_plan_rejects_already_measured_candidates_and_uncited_reference(trial):
    decision, observed = trial
    with pytest.raises(ValueError, match="selected and unmeasured"):
        build(decision, observed | {"I47V:N144S:I161V": 4.0})
    decision["evidence_ids"].remove("WT")
    with pytest.raises(ValueError, match="observed and cited"):
        build(decision, observed)


def test_audit_rejects_changed_threshold_missing_forecast_and_repeated_round(trial):
    decision, observed = trial
    plan = build(decision, observed)
    with pytest.raises(ValueError, match="threshold differs"):
        evaluate_hypotheses([plan | {"threshold": 3.0}], observed)
    with pytest.raises(ValueError, match="threshold differs"):
        evaluate_hypotheses([plan], observed | {"WT": 3.0})
    with pytest.raises(ValueError, match="original fitted prediction"):
        evaluate_hypotheses([plan | {"predictions": {}}], observed)
    with pytest.raises(ValueError, match="unique and chronological"):
        evaluate_hypotheses([plan, plan], observed)
