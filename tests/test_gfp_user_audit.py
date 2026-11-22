import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from gfp_user_audit import run_user_audit


EVIDENCE = Path(__file__).resolve().parents[1] / "evidence" / "gfp"


def _read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _manifest_round(source, destination, prior=None):
    shutil.copytree(source, destination)
    state_path = destination / "state.json"
    state = json.loads(state_path.read_text())
    if prior is not None:
        state["prior_state_path"] = os.path.relpath(prior, destination)
        state["prior_state_sha256"] = hashlib.sha256(prior.read_bytes()).hexdigest()
    else:
        state["prior_state_path"] = None
    measurements = destination / "imported_measurements.csv"
    values = {row["mutant"]: row["fitness"] for row in state["observations"]}
    with measurements.open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("mutant", "fitness"))
        writer.writerows((key, values[key]) for key in state["new_measurements"])
    state["artifacts"] = {name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
                          for name in ("recommendations.csv", measurements.name)}
    state_path.write_text(json.dumps(state))
    return state_path


def test_audit_reconstructs_actual_api_demo_without_causal_claim(tmp_path):
    state = EVIDENCE / "user-api-demo" / "round-02" / "state.json"
    original = hashlib.sha256(state.read_bytes()).hexdigest()
    output = tmp_path / "audit"
    summary = run_user_audit(state, output)
    assert summary["status"] == "complete"
    assert summary["rounds"] == [1, 2]
    assert summary["returned_recommendations"] == summary["prediction_residuals"] == 5
    assert summary["verified_citations"] == 6
    assert summary["recommendation_sets_changed"] == 1
    assert summary["paired_status"] == "NOT EVALUABLE"
    assert summary["feedback_gain"] is None
    first, second = _read_lines(output / "decision_traces.jsonl")
    initial_scores = {row["mutation"]: row["predicted_fitness"] for row in first["recommendations"]}
    for record in second["imported_measurements"]:
        assert record["predicted_fitness"] == initial_scores[record["mutant"]]
        assert record["residual"] == record["fitness"] - record["predicted_fitness"]
    assert all(row["measured_fitness"] is None for row in second["recommendations"])
    assert hashlib.sha256(state.read_bytes()).hexdigest() == original


def test_audit_links_rounds_across_folders_and_rejects_modified_ingest(tmp_path):
    source = EVIDENCE / "user-demo"
    first = _manifest_round(source / "round-01", tmp_path / "history" / "first")
    second = _manifest_round(source / "round-02", tmp_path / "other" / "second", prior=first)
    assert run_user_audit(second, tmp_path / "audit")["rounds"] == [1, 2]
    assert all(trace["import_receipt_verified"] for trace in _read_lines(tmp_path / "audit" / "decision_traces.jsonl"))
    with (first.parent / "imported_measurements.csv").open("a") as stream:
        stream.write("extra,4.0\n")
    with pytest.raises(ValueError, match="artifact digest does not match"):
        run_user_audit(second, tmp_path / "tampered")
    assert not (tmp_path / "tampered").exists()


def test_audit_marks_a_missing_predecessor_and_keeps_later_origin_unknown(tmp_path):
    source = EVIDENCE / "user-api-demo" / "round-02"
    destination = tmp_path / "isolated"
    shutil.copytree(source, destination)
    state = destination / "state.json"
    with pytest.raises(ValueError, match="outside every audited round"):
        run_user_audit(state, destination / "audit")
    output = tmp_path / "audit"
    summary = run_user_audit(state, output)
    assert summary["status"] == "partial"
    assert summary["history_gap"].startswith("Missing predecessor for round 2")
    assert summary["recommendation_comparisons"] == summary["returned_recommendations"] == 0
    trace, = _read_lines(output / "decision_traces.jsonl")
    assert all(record["kind"] == "origin_unavailable" for record in trace["imported_measurements"])


def test_audit_rejects_evidence_from_outside_research_history(tmp_path):
    source = EVIDENCE / "user-api-demo"
    destination = tmp_path / "history"
    shutil.copytree(source, destination)
    log = destination / "round-02" / "agent.json"
    trace = json.loads(log.read_text())
    first_citation = trace["decision"]["evidence_aliases"][0]
    trace["observed_evidence"][first_citation]["fitness"] += 1.0
    log.write_text(json.dumps(trace))
    with pytest.raises(ValueError, match="cited measured evidence does not match"):
        run_user_audit(destination / "round-02" / "state.json", tmp_path / "audit")
