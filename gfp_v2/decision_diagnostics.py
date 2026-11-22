"""Inspect closed GFP V2 decisions after the protected study unlock."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from statistics import mean

from gfp_data import load_dataset
from gfp_v2.trials import BUDGET, MODEL_ARMS, ROUNDS, check_choices, digest, read_json, write_csv, write_json


def cutoff_attribution(agent_values: list[float], greedy_values: list[float],
                       possible_best: float, threshold: float) -> str:
    """Assign failures based on the actual shared roster and its fixed ranking."""
    if possible_best <= threshold:
        return "pool_ceiling"
    agent_hit, greedy_hit = max(agent_values) > threshold, max(greedy_values) > threshold
    if agent_hit and greedy_hit:
        return "both_hit"
    if agent_hit:
        return "agent_rescue"
    if greedy_hit:
        return "agent_missed_greedy_opportunity"
    return "both_missed_eligible_opportunity"


def hypothesis_counts(verdicts: list[dict]) -> dict:
    counts = Counter(total=len(verdicts))
    for row in verdicts:
        counts[row["status"]] += 1
        counts["comparator_wt" if row["comparator_id"] == "WT" else "comparator_incumbent"] += 1
        counts["forecast_exceeds_comparator"] += int(row["forecast_exceeds_comparator"])
        counts["forecast_not_above_comparator"] += int(not row["forecast_exceeds_comparator"])
        counts["measured_above_wt"] += int(row["above_WT"] is True)
        counts["measured_above_incumbent"] += int(row["improves_incumbent"] is True)
    return dict(counts)


def inspect_case(dataset, request: dict, commit: dict, measurement: dict) -> tuple[dict, dict]:
    policy = commit["policy"]
    candidates = request["candidates"]
    eligible = {row["id"] for row in candidates}
    if len(eligible) != len(candidates):
        raise ValueError("Archive contains repeated candidate identifiers")
    selected = check_choices(policy["selected_ids"], eligible)
    agent = {row["id"]: row["measured_fitness"] for row in measurement["measurements"]}
    if list(agent) != selected or any(dataset.scores[key] != value for key, value in agent.items()):
        raise ValueError("The archive decision and true measured batch differ")
    greedy = sorted(candidates, key=lambda row: (-row["prediction"], row["id"]))[:BUDGET]
    greedy_ids = [row["id"] for row in greedy]
    greedy_values = [dataset.scores[key] for key in greedy_ids]
    best_eligible = max(dataset.scores[key] for key in eligible)
    wt = request["wild_type_fitness"]
    incumbent = request["current_best"]
    recent_ids = {row["id"] for row in request["latest_feedback"]}
    observed_ids = {row["id"] for row in request["observed"]}
    cited_ids = {row["id"] for row in policy["evidence"]}
    if sorted(recent_ids & cited_ids) != policy["recent_evidence_used"] or not cited_ids <= observed_ids:
        raise ValueError("Knowledge audit disagrees with archived evidence IDs")
    row = {"block": request["block"], "arm": request["arm"], "round": request["round"],
           "above_wt_attribution": cutoff_attribution(list(agent.values()), greedy_values, best_eligible, wt),
           "incumbent_attribution": cutoff_attribution(list(agent.values()), greedy_values, best_eligible, incumbent),
           "best_available_measured": best_eligible, "agent_best": max(agent.values()),
           "same_request_greedy_best": max(greedy_values),
           "agent_minus_same_request_greedy_best": max(agent.values()) - max(greedy_values),
           "agent_minus_same_request_greedy_mean": mean(agent.values()) - mean(greedy_values),
           "agent_above_wt": sum(value > wt for value in agent.values()),
           "same_request_greedy_above_wt": sum(value > wt for value in greedy_values),
           "overlap_with_same_request_greedy": len(set(greedy_ids) & set(selected)),
           "recent_evidence_available": len(recent_ids), "recent_evidence_cited": len(cited_ids & recent_ids),
           "strategy": policy["strategy"], "adaptation": policy["adaptation"]}
    detail = {"round": request["round"], "selection": selected,
              "next_decision_can_see": sorted(agent), "measurements": measurement["measurements"],
              "cited_evidence": policy["evidence"], "used_recent": policy["recent_evidence_used"],
              "hypotheses": policy["hypotheses"], "verdicts": measurement["verdicts"],
              "strategy": policy["strategy"], "adaptation": policy["adaptation"]}
    return row, detail


def run(dataset, source: Path, output: Path) -> dict:
    config = read_json(source / "config.json")
    status = read_json(source / "status.json")
    if config["study"] != "agent" or status["status"] != "complete":
        raise ValueError("Retrospective decision diagnostics require completed agent trials")
    cases, fixed_example, example = [], [], None
    verdicts = {arm: [] for arm in MODEL_ARMS}
    for block in config["blocks"]:
        for arm in sorted(MODEL_ARMS):
            path = source / "arms" / block["id"] / arm
            previous = None
            for number in range(1, ROUNDS + 1):
                base = path / f"round-{number:02d}"
                request = base.with_name(f"{base.name}-request.json")
                commit = base.with_name(f"{base.name}-commit.json")
                measured = read_json(base.with_name(f"{base.name}-measurement.json"))
                raw_commit = read_json(commit)
                if measured["commit_sha256"] != digest(commit) or raw_commit["request_sha256"] != digest(request):
                    raise ValueError("Retrospective scoring expects committed measurement and request hashes")
                case, detail = inspect_case(dataset, read_json(request), raw_commit, measured)
                cases.append(case)
                verdicts[arm].extend(detail["verdicts"])
                if block["id"] == config["blocks"][0]["id"] and arm == "D_agent":
                    fixed_example.append(detail)
                if example is None and previous and detail["adaptation"] not in ("continue", "no_new_feedback"):
                    rejected = {test["candidate_id"] for test in previous["verdicts"] if test["status"] == "REFUTED"}
                    cited = sorted(rejected & set(detail["used_recent"]))
                    if cited:
                        example = {"block": block["id"], "arm": arm, "cited_refuted": cited,
                                   "previous": previous, "following": detail,
                                   "scope": "Descriptive selected trace, not a causal counterfactual."}
                previous = detail
    summary = {}
    for arm in sorted(MODEL_ARMS):
        rows = [case for case in cases if case["arm"] == arm]
        followups = [case for case in rows if case["recent_evidence_available"]]
        counts = {"decisions": len(rows),
                  "recent_mentions": sum(row["recent_evidence_cited"] > 0 for row in followups),
                  "recent_is_available": len(followups),
                  "mean_agent_minus_same_request_greedy_best":
                      mean(row["agent_minus_same_request_greedy_best"] for row in rows)}
        for field in ("above_wt_attribution", "incumbent_attribution"):
            counts[field] = {label: sum(row[field] == label for row in rows)
                             for label in sorted({row[field] for row in rows})}
        summary[arm] = counts
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "decision_attribution.csv", cases)
    write_json(output / "case-first-block-D.json", fixed_example)
    write_json(output / "hypothesis-accounting.json",
               {arm: hypothesis_counts(verdicts[arm]) for arm in sorted(verdicts)})
    if example is not None:
        write_json(output / "case-refuted-and-cited.json", example)
    repaired_blocks = {entry.split("/", 1)[0] for entry in status.get("alias_recovered_decisions", [])}
    with (source / "paired.csv").open(newline="", encoding="utf-8") as stream:
        paired = [row for row in csv.DictReader(stream) if row["control"] == "max_non_llm"]
    subset_results = []
    for arm in sorted(MODEL_ARMS):
        for repaired in (False, True):
            group = [row for row in paired if row["treatment"] == arm and (row["block"] in repaired_blocks) == repaired]
            if group:
                subset_results.append({"arm": arm, "recovery_present_in_block": repaired,
                                       "blocks": len(group), "new_best_delta": mean(float(row["new_best"]) for row in group)})
    write_csv(output / "schema_recovery_sensitivity.csv", subset_results)
    result = {"status": "complete", "cases": len(cases), "by_arm": summary,
              "recovery_blocks": sorted(repaired_blocks),
              "scope": (
                  "Retrospective common-roster ranking diagnostic, not a new acquisition policy run "
                  "or causal estimate. Equal requests do not necessarily imply equal downstream states."
              )}
    write_json(output / "summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-cache", type=Path)
    args = parser.parse_args()
    print(run(load_dataset(args.data_cache), args.agent_run, args.output))
