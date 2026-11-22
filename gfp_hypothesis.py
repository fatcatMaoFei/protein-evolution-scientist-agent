"""Lock prospective GFP assay criteria and reconcile them with imported results."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping

from gfp_data import make_variant


_SHA256 = re.compile(r"[0-9a-f]{64}")
_TEST_FIELDS = {"reference_id", "candidate_ids", "min_exceed"}


def _ids(value, label: str) -> list[str]:
    if (not isinstance(value, list) or not value
            or any(not isinstance(key, str) or make_variant(key).id != key for key in value)
            or len(set(value)) != len(value)):
        raise ValueError(f"GFP {label} must be distinct canonical variant IDs")
    return value.copy()


def _finite(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"GFP {label} must be a finite score")
    return float(value)


def _validated_plan(plan: Mapping, observations: Mapping[str, float]) -> dict:
    number = plan.get("round")
    statement = plan.get("hypothesis")
    digest = plan.get("decision_sha256")
    if type(number) is not int or number < 1:
        raise ValueError("GFP hypothesis round must be a positive integer")
    if not isinstance(statement, str) or not statement.strip():
        raise ValueError("GFP hypothesis must have a statement")
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise ValueError("GFP hypothesis must identify the accepted decision SHA-256")

    candidates = _ids(plan.get("candidate_ids"), "test candidates")
    evidence = _ids(plan.get("evidence_ids"), "hypothesis evidence")
    reference = plan.get("reference_id")
    if reference not in evidence or not set(evidence) <= observations.keys():
        raise ValueError("GFP hypothesis reference must be an observed and cited variant")
    if set(candidates) & set(evidence):
        raise ValueError("GFP test candidates must be distinct from prior evidence")
    threshold = _finite(plan.get("threshold"), "hypothesis threshold")
    if threshold != _finite(observations[reference], "reference measurement"):
        raise ValueError("GFP hypothesis threshold differs from the cited reference measurement")

    minimum = plan.get("min_exceed")
    if type(minimum) is not int or not 1 <= minimum <= len(candidates):
        raise ValueError("GFP min_exceed must be between one and the number of test candidates")
    predictions = plan.get("predictions")
    if not isinstance(predictions, dict) or predictions.keys() != set(candidates):
        raise ValueError("Each GFP test candidate requires its original fitted prediction")
    return {"round": number, "hypothesis": statement, "reference_id": reference, "threshold": threshold,
            "candidate_ids": candidates, "min_exceed": minimum, "evidence_ids": evidence,
            "predictions": {key: _finite(predictions[key], "original prediction") for key in candidates},
            "decision_sha256": digest}


def make_hypothesis_plan(*, round_number: int, decision: Mapping,
                         observations: Mapping[str, float], decision_sha256: str) -> dict:
    """Bind a model's stated test to pre-assay evidence, selected IDs and forecasts."""
    test = decision.get("hypothesis_test")
    if not isinstance(test, dict) or set(test) != _TEST_FIELDS:
        raise ValueError("GFP decision requires reference_id, candidate_ids and min_exceed for its hypothesis test")
    reference = test["reference_id"]
    if not isinstance(reference, str) or reference not in observations:
        raise ValueError("GFP hypothesis reference must be an observed and cited variant")

    tested = _ids(test["candidate_ids"], "test candidates")
    selected = {row["id"]: row["prediction"] for row in decision["selected_candidates"]}
    if not set(tested) <= selected.keys() or set(tested) & observations.keys():
        raise ValueError("GFP hypothesis test candidates must be selected and unmeasured at decision time")
    plan = {**test, "round": round_number, "hypothesis": decision["hypothesis"],
            "threshold": observations[reference], "evidence_ids": decision["evidence_ids"],
            "predictions": {key: selected[key] for key in tested}, "decision_sha256": decision_sha256}
    return _validated_plan(plan, observations)


def evaluate_hypotheses(hypotheses: Iterable[Mapping], observations: Mapping[str, float]) -> list[dict]:
    """Evaluate fixed criteria using only measurements in the current imported ledger."""
    results = []
    for hypothesis in hypotheses:
        plan = _validated_plan(hypothesis, observations)
        if results and plan["round"] <= results[-1]["round"]:
            raise ValueError("GFP hypothesis rounds must be unique and chronological")
        threshold = plan["threshold"]
        rows = []
        for key, prediction in plan["predictions"].items():
            actual = _finite(observations[key], "test measurement") if key in observations else None
            predicted_exceed = prediction > threshold
            observed_exceed = actual > threshold if actual is not None else None
            error = None
            if observed_exceed is not None:
                error = "agreement"
                if predicted_exceed != observed_exceed:
                    error = "false_positive" if predicted_exceed else "false_negative"
            rows.append({"id": key, "predicted_fitness": prediction, "measured_fitness": actual,
                         "residual": actual - prediction if actual is not None else None,
                         "forecast_exceeds_threshold": predicted_exceed,
                         "measurement_exceeds_threshold": observed_exceed, "forecast_outcome": error})
        n_measured = sum(row["measured_fitness"] is not None for row in rows)
        n_exceed = sum(row["measurement_exceeds_threshold"] is True for row in rows)
        status = "pending"
        if n_measured == len(rows):
            status = "supported" if n_exceed >= plan["min_exceed"] else "refuted"
        results.append({**plan, "status": status, "n_measured": n_measured, "n_exceed": n_exceed,
                        "candidate_results": rows})
    return results
