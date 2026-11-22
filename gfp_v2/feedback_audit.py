"""Withhold only newly revealed GFP evidence while holding each fitted forecast fixed."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

import numpy as np

from gfp_v2.trials import (
    MAX_OUTER_ATTEMPTS,
    check_choices,
    digest,
    preserve,
    read_json,
    transport_spec,
    write_csv,
    write_json,
)

BLOCKS = ("2201", "2202")
ARMS = ("D_agent", "E_knowledge")
CONDITIONS = ("with", "without")
REPLICATES = (1, 2)
ROUND = 2


def matched_requests(original: dict) -> dict[str, dict]:
    """Retain the same candidate features, predictions, and previous selections."""
    latest = original["latest_feedback"]
    if not latest or len({row["id"] for row in latest}) != len(latest):
        raise ValueError("Audit requires distinct previously revealed feedback")
    omit = {row["id"] for row in latest}
    past = [row for row in original["observed"] if row["id"] not in omit]
    if len(past) + len(latest) != len(original["observed"]):
        raise ValueError("The feedback rows must have been present in the original observations")
    common = {**original, "prior_decisions": [
        {"round": prior["round"], "selected_ids": prior["selected_ids"], "strategy": prior.get("strategy")}
        for prior in original["prior_decisions"]
    ]}
    return {"with": common, "without": {**common, "observed": past, "latest_feedback": [],
                                         "current_best": max(row["fitness"] for row in past)}}


def decide(case: dict, condition: str, replicate: int, output: Path, transport, replay: Path | None):
    from gfp_v2.policy import select

    request = case["requests"][condition]
    case_path = Path("cases") / case["block"] / case["arm"]
    target = output / case_path
    stem = f"{condition}-{replicate}"
    commit = target / f"{stem}-commit.json"
    if commit.is_file():
        return read_json(commit)
    attempts = sorted(target.glob(f"{stem}-model-*.json"))
    if len(attempts) >= MAX_OUTER_ATTEMPTS:
        raise ValueError(f"Policy attempts exhausted for {case_path}/{stem}")
    name = f"{stem}-model-{len(attempts) + 1}.json"
    archive = None
    if replay:
        old = replay / case_path
        name = read_json(old / commit.name)["model_log"]
        if Path(name).name != name:
            raise ValueError("Archived model log must refer to this audit case only")
        archive = old / name
    decision = select(request=request, transport=transport, knowledge=case["arm"] == "E_knowledge",
                      log_path=target / name, replay_path=archive, repair_aliases=True)
    check_choices(decision["selected_ids"], [row["id"] for row in request["candidates"]])
    result = {"block": case["block"], "arm": case["arm"], "condition": condition, "replicate": replicate,
              "model_log": name, "policy": decision,
              "request_sha256": digest(target / f"{condition}-request.json")}
    write_json(commit, result)
    return result


def jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right)


def recover_case_results(decisions, dataset, cases):
    """Read selected archival values only after every fork decision has been committed."""
    indexed = {(row["block"], row["arm"], row["condition"], row["replicate"]): row for row in decisions}
    measurements, comparisons = [], []
    wt = dataset.scores[dataset.wild_type_id]
    for case in cases:
        scores = {}
        batch = {}
        for condition in CONDITIONS:
            for rep in REPLICATES:
                decision = indexed[case["block"], case["arm"], condition, rep]
                picked = decision["policy"]["selected_ids"]
                truth = {key: dataset.scores[key] for key in picked}
                scores[condition, rep] = max(truth.values()), mean(truth.values())
                batch[condition, rep] = set(picked)
                measurements.append({
                    "block": case["block"], "arm": case["arm"], "condition": condition, "replicate": rep,
                    "picked_ids": ";".join(picked), "new_best": scores[condition, rep][0],
                    "new_mean": scores[condition, rep][1], "above_wt": sum(value > wt for value in truth.values()),
                    "recent_evidence_cited": len(decision["policy"].get("recent_evidence_used", [])),
                    "truth_ids": ";".join(f"{key}={value!r}" for key, value in truth.items()),
                })
        cases_score = {"block": case["block"], "arm": case["arm"]}
        cases_score["between_jaccard"] = mean(jaccard(batch["with", a], batch["without", b])
                                              for a in REPLICATES for b in REPLICATES)
        for condition in CONDITIONS:
            a, b = REPLICATES
            cases_score[f"within_{condition}_jaccard"] = jaccard(batch[condition, a], batch[condition, b])
            cases_score[f"within_{condition}_best_abs_delta"] = abs(scores[condition, a][0] - scores[condition, b][0])
        for index, metric in enumerate(("best", "mean")):
            cases_score[f"feedback_{metric}_delta"] = mean(scores["with", rep][index] for rep in REPLICATES) - mean(
                scores["without", rep][index] for rep in REPLICATES
            )
        comparisons.append(cases_score)
    points = [(row["feedback_best_delta"], row["feedback_mean_delta"]) for row in comparisons]
    # D/E within a block share pool and initial observations; resample whole blocks.
    grouped = np.array([np.mean([p for row, p in zip(comparisons, points, strict=True) if row["block"] == block],
                               axis=0) for block in BLOCKS])
    samples = np.random.default_rng(668021).integers(len(BLOCKS), size=(10_000, len(BLOCKS)))
    quantiles = np.quantile(grouped[samples].mean(axis=1), [0.025, 0.975], axis=0)
    summary = {
        "contexts": len(cases),
        "repeat_draws_per_condition": len(REPLICATES),
        "mean_between_jaccard": mean(row["between_jaccard"] for row in comparisons),
        "mean_within_with_jaccard": mean(row["within_with_jaccard"] for row in comparisons),
        "mean_within_without_jaccard": mean(row["within_without_jaccard"] for row in comparisons),
        "mean_feedback_best_delta": mean(row["feedback_best_delta"] for row in comparisons),
        "feedback_best_ci_low": float(quantiles[0, 0]), "feedback_best_ci_high": float(quantiles[1, 0]),
        "mean_feedback_mean_delta": mean(row["feedback_mean_delta"] for row in comparisons),
        "feedback_mean_ci_low": float(quantiles[0, 1]), "feedback_mean_ci_high": float(quantiles[1, 1]),
        "with_feedback_recent_citation_count": sum(row["recent_evidence_cited"] > 0 and row["condition"] == "with"
                                                   for row in measurements),
        "interpretation": ("Decision channel conditional on identical round-two forecasts. Two independent archive "
                           "blocks and two repeated responses per condition provide descriptive noise context."),
    }
    return measurements, comparisons, summary


def run_audit(dataset, blocks: list[dict], agent_run: Path, output: Path, *, transport, profile, replay=None,
              resume=False, workers=4, hashes=None):
    config = read_json(agent_run / "config.json")
    status = read_json(agent_run / "status.json")
    if (
        status["status"] != "complete" or config["study"] != "agent" or config["blocks"] != blocks
        or config["requested_model"] != transport.model or config["hashes"] != hashes
    ):
        raise ValueError("Feedback audit requires the same complete predeclared agent study and model")
    if not resume and output.exists():
        raise FileExistsError(f"Use a fresh feedback audit directory: {output}")
    output.mkdir(parents=True, exist_ok=resume)
    if (output / "status.json").exists():
        raise FileExistsError("Feedback audit is already finalized")
    preserve(output / "config.json", {
        "agent_run_sha256": digest(agent_run / "config.json"), "model": transport.model, "profile": profile,
        "hashes": hashes, "blocks": list(BLOCKS), "arms": list(ARMS), "round": ROUND,
        "conditions": list(CONDITIONS), "repeats": list(REPLICATES), "outer_attempt_limit": MAX_OUTER_ATTEMPTS,
        "profile_transport": transport_spec(transport, profile),
        "presentation": "All forks use the same lossless numeric aliases for the full closed pool.",
    })
    cases = []
    for block in BLOCKS:
        for arm in ARMS:
            old = agent_run / "arms" / block / arm / f"round-{ROUND:02d}-request.json"
            requests = matched_requests(read_json(old))
            target = output / "cases" / block / arm
            case = {"block": block, "arm": arm, "requests": requests}
            for condition, request in requests.items():
                preserve(target / f"{condition}-request.json", request)
            preserve(target / "scope.json", {
                "original_request_sha256": digest(old),
                "conditioned_on_fitted_scores": True,
                "same_candidate_roster_and_forecasts": requests["with"]["candidates"] == requests["without"]["candidates"],
                "held_out_measurements": [row["id"] for row in requests["with"]["latest_feedback"]],
                "redacted_prior_verdicts_in_both": True,
            })
            cases.append(case)
    decisions, errors = [], []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(decide, case, condition, rep, output, transport, replay):
            (case["block"], case["arm"], condition, rep)
            for case in cases for condition in CONDITIONS for rep in REPLICATES
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                decisions.append(future.result())
            except Exception as exc:
                errors.append({"case": list(key), "error": f"{type(exc).__name__}: {exc}"})
    if errors:
        number = len(list(output.glob("failures-*.json"))) + 1
        write_json(output / f"failures-{number:02d}.json", sorted(errors, key=lambda row: row["case"]))
        raise RuntimeError(f"{len(errors)} audit forks failed. Consult {output}; no audit scores have been opened.")
    measurements, comparisons, summary = recover_case_results(decisions, dataset, cases)
    write_csv(output / "measurements.csv", measurements)
    write_csv(output / "comparisons.csv", comparisons)
    write_json(output / "summary.json", summary)
    result = {"status": "complete", "contexts": len(cases), "policy_decisions": len(decisions),
              "api_mode": "replay" if replay is not None else "live"}
    write_json(output / "status.json", result)
    return result
