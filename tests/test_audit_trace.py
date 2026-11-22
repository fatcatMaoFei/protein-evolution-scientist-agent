"""Check timeline linkage without opening any in-progress study archives."""

import csv
import json

import pytest

from audit_trace import NOT_EVALUABLE, _reconstruct_arm, reconstruct_decisions, summarize_decisions


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_round(path, step, candidates, evidence, known, draft_statement):
    menu = [
        {"alias": f"C{index:02d}", "id": mix, "rank": index, "source": "top_prediction", "prediction": 43 + index}
        for index, mix in enumerate(candidates, 1)
    ]
    selected = [menu[0], menu[2]]
    measured = [48.0 + step, 52.0 + step]
    proposal = {
        "analyst": "Summarized known examples.", "hypothesis": draft_statement,
        "evidence_ids": evidence, "candidate_ids": [entry["alias"] for entry in menu],
    }
    decision = {
        **proposal, "candidate_ids": [item["alias"] for item in selected],
        "selected_ids": [item["id"] for item in selected],
        "critic": "Validated eligibility.", "selection_rationale": "Evaluate both records.",
    }
    report = [
        {"alias": item["alias"], "eligible": True, "prediction_rank": item["rank"],
         "prediction_mpa": item["prediction"], "rf_tree_disagreement_mpa": 2.1,
         "nearest_observed_example": next(iter(known))}
        for item in menu
    ]
    model_trace = {
        "model": "deepseek-fixture", "status": "ok", "mode": "live",
        "seed": 17, "round": step, "budget": 2,
        "knowledge_augmented": False, "decision": decision, "menu": menu,
        "tool_calls": [
            {"tool": "data_analyst", "trigger": "pipeline", "result": {
                "observations": len(known), "strongest": {mix: {"strength": y} for mix, y in known.items()},
                "weakest": {},
            }},
            {"tool": "evaluate_concrete_shortlist", "trigger": "model_tool_call",
             "arguments": proposal, "result": report},
        ],
    }
    stem = f"round-{step:02d}"
    write_json(path / f"{stem}-agent.json", model_trace)
    write_json(
        path / f"{stem}.json",
        {"seed": 17, "arm": "C_llm", "round": step, "budget": 2,
         "contemporaneous_greedy_ids": [menu[0]["id"], menu[1]["id"]],
         "greedy_overlap_at_k": 0.5, "number_observed_after_refit": len(known) + 2,
         "round_best_mpa": max(measured), "best_so_far_mpa": max(*known.values(), *measured),
         "top_cutoff_mpa": 50.0, "round_top_hits": sum(y >= 50 for y in measured)},
    )
    rows = [
        {"seed": 17, "arm": "C_llm", "round": step, "batch_position": position,
         "experiment_number": (step - 1) * 2 + position, "id": item["id"],
         "strength_mpa": y, "prediction_mpa": item["prediction"], "rf_rank": item["rank"],
         "top_hit": int(y >= 50), "is_myopic_greedy_pick": int(position == 1)}
        for position, (item, y) in enumerate(zip(selected, measured, strict=True), 1)
    ]
    with (path / f"{stem}-selections.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return {**known, **{item["id"]: y for item, y in zip(selected, measured, strict=True)}}


@pytest.fixture
def arm_archive(tmp_path):
    path = tmp_path / "concrete" / "arms" / "seed-17" / "C_llm"
    known = {"mix-001": 41.0, "mix-002": 39.0}
    write_json(path / "initial_observed.json", {mix: {"strength": y} for mix, y in known.items()})
    pool = [f"mix-{index:03d}" for index in range(3, 11)]
    write_json(tmp_path / "concrete" / "splits" / "seed-17.json", {"initial": list(known), "pool": pool})
    known = write_round(path, 1, pool[:4], ["mix-001"], known, "Estimated higher strength in this batch.")
    write_round(path, 2, pool[4:], ["mix-005"], known, "Explore near the new high result.")
    return path


def test_decision_history_preserves_when_each_observation_was_revealed(tmp_path, arm_archive):
    first, second = _reconstruct_arm(tmp_path, "concrete", "deepseek", 17, "C_llm", 2, 2)
    assert [row["recipe_id"] for row in first["observed_before"]] == ["mix-001", "mix-002"]
    assert "strength_mpa" not in first["proposed"][1]
    assert first["hypothesis"]["outcome_status"] == NOT_EVALUABLE
    assert first["same_state_greedy"]["overlap_at_k"] == 0.5
    cited = second["hypothesis"]["cited_evidence"]
    assert [(item["recipe_id"], item["revealed_round"]) for item in cited] == [("mix-005", 1)]
    assert cited[0]["source"]["file"].endswith("round-01-selections.csv")
    assert cited[0]["source"]["data_row"] == 2
    assert first["next_decision_change"]["newly_revealed_and_cited"] == cited
    assert second["next_decision_change"] is None
    summary = summarize_decisions([first, second])
    assert summary["revealed_measurements"] == 4
    assert summary["hypothesis_outcomes"] == {NOT_EVALUABLE: 2}
    assert summary["transitions_citing_newly_revealed_observations"] == 1


def test_rejects_future_measurement_as_cited_prior_evidence(tmp_path, arm_archive):
    trace_file = arm_archive / "round-01-agent.json"
    trace = json.loads(trace_file.read_text(encoding="utf-8"))
    trace["decision"]["evidence_ids"] = ["mix-005"]
    trace["tool_calls"][1]["arguments"]["evidence_ids"] = ["mix-005"]
    write_json(trace_file, trace)
    with pytest.raises(ValueError, match="cited evidence was unavailable"):
        _reconstruct_arm(tmp_path, "concrete", "deepseek", 17, "C_llm", 2, 2)


def test_study_reader_refuses_to_read_rounds_until_all_models_are_complete(tmp_path):
    for folder in ("concrete", "sonnet", "gpt", "qwen"):
        status = "running" if folder == "qwen" else "complete"
        write_json(tmp_path / folder / "status.json", {"status": status, "mode": "formal"})
    with pytest.raises(ValueError, match="complete formal evidence for qwen"):
        reconstruct_decisions(tmp_path)
