from types import SimpleNamespace

import pytest

from gfp_v2 import decision_diagnostics, feedback_audit, trials


def test_archive_oracle_does_not_read_uncommitted_or_duplicate_choices(tmp_path):
    queried = []

    def score(key):
        queried.append(key)
        return 2.1

    ids = ["v" + str(number) for number in range(6)]
    oracle = trials.ArchiveOracle(ids, score)
    invalid, valid = tmp_path / "invalid.json", tmp_path / "valid.json"
    trials.write_json(invalid, {"policy": {"selected_ids": ids[:4] + ["outside"]}})
    with pytest.raises(FileNotFoundError):
        oracle.reveal(valid)
    with pytest.raises(ValueError, match="distinct, eligible"):
        oracle.reveal(invalid)
    assert queried == []
    trials.write_json(valid, {"policy": {"selected_ids": ids[:5]}})
    assert oracle.reveal(valid) == dict.fromkeys(ids[:5], 2.1)
    with pytest.raises(ValueError, match="distinct, eligible"):
        oracle.reveal(valid)
    assert queried == ids[:5]


def test_arm_fits_only_prior_own_reveals_and_stores_commit_before_lookup(tmp_path, monkeypatch):
    available = [f"gfp-{number}" for number in range(16)]
    initial = ["WT", "initial"]
    output = tmp_path / "arm"
    train_inputs = []
    archive_queries = []

    class Measured:
        def __getitem__(self, key):
            if key in initial:
                return 3.0
            number = len(archive_queries) // trials.BUDGET + 1
            commit = trials.read_json(output / f"round-{number:02d}-commit.json")
            assert key in commit["policy"]["selected_ids"]
            assert not (output / f"round-{number:02d}-measurement.json").exists()
            archive_queries.append(key)
            return 3.1 + (len(archive_queries) % 5) * 0.1

    def fake_forecast(dataset, observed, candidates, model, seed):
        assert set(observed) == set(initial + archive_queries)
        assert all(key not in candidates for key in observed)
        train_inputs.append(observed.copy())
        return [{"id": key, "prediction": float(index), "risk": 0.0} for index, key in enumerate(candidates)]

    monkeypatch.setattr(trials, "forecast", fake_forecast)
    dataset = SimpleNamespace(scores=Measured(), wild_type_id="WT")
    totals, batches, selection = trials.run_arm(
        dataset, {"id": "2201", "initial": initial, "pool": available}, "B_predictor", "ridge", output,
    )
    assert totals["budget_used"] == 15 and len(batches) == 3 and len(selection) == 15
    assert len(set(archive_queries)) == 15
    assert [len(rows) for rows in train_inputs] == [2, 7, 12]
    for number in range(1, trials.ROUNDS + 1):
        stem = output / f"round-{number:02d}"
        recorded = trials.read_json(stem.with_name(f"{stem.name}-measurement.json"))
        assert recorded["commit_sha256"] == trials.digest(stem.with_name(f"{stem.name}-commit.json"))


def test_audit_redacts_previous_truth_in_both_keeps_candidate_rows_and_history():
    wt = {"id": "WT", "fitness": 3.7, "round": 0}
    recent = {"id": "B", "fitness": 4.0, "round": 1}
    candidates = [{"id": "C", "prediction": 3.1, "risk": 0.2}]
    original = {
        "observed": [wt, recent], "latest_feedback": [recent], "current_best": 4.0,
        "candidates": candidates, "prior_decisions": [{"round": 1, "selected_ids": ["B"],
                                                       "verdicts": [{"id": "B", "fitness": 4.0}]}],
    }
    audit = feedback_audit.matched_requests(original)
    no, yes = audit["without"], audit["with"]
    assert no["candidates"] == yes["candidates"] == candidates
    assert no["prior_decisions"] == yes["prior_decisions"]
    assert all("verdicts" not in row for row in no["prior_decisions"])
    assert no["observed"] == [wt] and no["current_best"] == wt["fitness"] and no["latest_feedback"] == []
    assert yes["latest_feedback"] == [recent] and yes["observed"] == [wt, recent]
    assert "verdicts" in original["prior_decisions"][0]


def test_primary_contrast_uses_stronger_realized_non_llm_per_block():
    def row(block, arm, score):
        return {"block": block, "arm": arm, "new_best": score, "new_mean": score,
                "above_wt": 0, "incumbent_gain": 0}

    runs = [row("2201", "B_predictor", 3.8), row("2201", "C_explore", 3.2), row("2201", "D_agent", 3.4),
            row("2202", "B_predictor", 3.1), row("2202", "C_explore", 3.9), row("2202", "D_agent", 3.9)]
    paired, summary = trials.paired_statistics(runs, (("D_agent", "max_non_llm"),))
    assert [result["paired_arm"] for result in paired] == ["B_predictor", "C_explore"]
    assert summary[0]["new_best"] == pytest.approx(-0.2)
    assert summary[0]["new_best_ties"] == 1 and summary[0]["new_best_losses"] == 1


def test_manifest_rejects_shared_non_wild_type_ids_across_studies():
    init = ["WT"] + [f"i{number}" for number in range(95)]
    pool = [f"p{number}" for number in range(96)]
    data = SimpleNamespace(wild_type_id="WT", variants=dict.fromkeys(init + pool))
    overlapping = {"acquisition": [{"id": 2101, "initial": init, "pool": pool}],
                   "agent": [{"id": 2201, "initial": init, "pool": pool}]}
    with pytest.raises(ValueError, match="overlapping sealed block agent/2201"):
        trials.validated_blocks(overlapping, data)


@pytest.mark.parametrize(("agent", "greedy", "possible", "label"), [
    (3.1, 3.0, 3.6, "pool_ceiling"),
    (3.8, 3.7, 3.9, "agent_rescue"),
    (3.1, 3.8, 3.9, "agent_missed_greedy_opportunity"),
    (3.1, 3.2, 3.9, "both_missed_eligible_opportunity"),
    (3.9, 3.8, 3.9, "both_hit"),
])
def test_decision_diagnostics_distinguish_search_scoring_and_selection(agent, greedy, possible, label):
    assert decision_diagnostics.cutoff_attribution([agent], [greedy], possible, 3.7) == label


def test_hypothesis_counts_keep_forecast_and_measured_outcome_distinct():
    record = {"status": "SUPPORTED", "comparator_id": "WT", "forecast_exceeds_comparator": False,
              "above_WT": True, "improves_incumbent": False}
    counts = decision_diagnostics.hypothesis_counts([record])
    assert counts["forecast_not_above_comparator"] == counts["SUPPORTED"] == 1
    assert counts["measured_above_wt"] == 1 and counts["measured_above_incumbent"] == 0
