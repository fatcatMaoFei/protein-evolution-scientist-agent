"""Independently check assay inputs, budgets, native call provenance and replay outputs."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from nonbio import ARMS, DEFAULT_DATA, read_dataset

TABLES = ("selections", "rounds", "methods", "method_summary", "comparisons", "predictive", "failures")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, msg: str) -> None:
    if not condition:
        raise ValueError(msg)


def verify_concrete(folder: Path, *, expected_seeds: list[int], expected_rounds: int) -> dict:
    data, info = read_dataset(DEFAULT_DATA)
    config = read(folder / "config.json")
    status = read(folder / "status.json")
    require(config["seeds"] == expected_seeds and config["rounds"] == expected_rounds and config["budget"] == 5,
            "Unexpected seed, round or budget")
    require(status["status"] == "complete", "The study is incomplete")
    require(read(folder / "source.json")["sha256"] == info["sha256"], "Source identity mismatch")
    selections = pd.read_csv(folder / "selections.csv")
    rounds = pd.read_csv(folder / "rounds.csv")
    counts = Counter()
    model_ids = set()
    per_arm = {arm: Counter() for arm in ARMS[2:]}
    for seed in expected_seeds:
        split = read(folder / "splits" / f"seed-{seed}.json")
        require(tuple(map(len, split.values())) == (48, 96, 96, 176), "Incorrect split sizes")
        require(len(set().union(*map(set, split.values()))) == len(data), "Splits overlap or omit recipes")
        cutoff = np.quantile(data.loc[split["pool"], "strength"], 0.9)
        initial = data.loc[split["initial"], "strength"]
        for arm in ARMS:
            path = folder / "arms" / f"seed-{seed}" / arm
            observed = set(split["initial"])
            initial_rows = read(path / "initial_observed.json")
            require(set(initial_rows) == observed, "Initial observations differ from split")
            require(all(np.isclose(row["strength"], initial.loc[mix]) for mix, row in initial_rows.items()),
                    "Incorrect initial measurement")
            incumbent = initial.max()
            total_hits = 0
            first_hit = None
            for step in range(1, expected_rounds + 1):
                take = selections[(selections.seed == seed) & (selections.arm == arm) & (selections["round"] == step)]
                take = take.sort_values("batch_position")
                ids = take.id.tolist()
                require(len(ids) == len(set(ids)) == 5 and set(ids) <= (set(split["pool"]) - observed),
                        "A query repeats a mix or targets a disallowed partition")
                target = data.loc[ids, "strength"].to_numpy()
                require(np.allclose(target, take.strength_mpa), "Oracle measurement differs from archived data")
                hits = target >= cutoff
                require(np.array_equal(hits.astype(int), take.top_hit), "Incorrect threshold hits")
                require(take.experiment_number.tolist() == list(range((step - 1) * 5 + 1, step * 5 + 1)),
                        "Incorrect experiment order")
                if hits.any() and first_hit is None:
                    first_hit = int(take.experiment_number.iloc[np.argmax(hits)])
                incumbent = max(incumbent, target.max())
                total_hits += hits.sum()
                metric = rounds[(rounds.seed == seed) & (rounds.arm == arm) & (rounds["round"] == step)].iloc[0]
                expected = (target.max(), incumbent, incumbent - initial.max(), incumbent - initial.iloc[0], total_hits)
                recorded = tuple(metric[key] for key in (
                    "round_best_mpa", "best_so_far_mpa", "gain_vs_initial_best_mpa", "gain_vs_reference_mpa", "top_hits"
                ))
                require(np.allclose(expected, recorded), "Round summary cannot be reconstructed")
                require((first_hit is None and pd.isna(metric.first_hit_tests)) or first_hit == metric.first_hit_tests,
                        "First-hit effort mismatch")
                if arm in per_arm:
                    trace = read(path / f"round-{step:02d}-agent.json")
                    require(trace["status"] == "ok" and trace["mode"] == "live", "Missing live agent decision")
                    require(trace["decision"]["selected_ids"] == ids, "Agent decision differs from oracle ledger")
                    require(trace["knowledge_augmented"] == (arm == ARMS[3]), "Mismatched treatment")
                    model_calls = [t for t in trace["tool_calls"] if t["trigger"] == "model_tool_call"]
                    require(len(model_calls) == 1, "Missing native evaluator call")
                    for call in trace["calls"]:
                        require(call["response"]["response_id"], "Missing provider response identifier")
                        model_ids.add(call["response"]["model"])
                    require(trace["live_generation_count"] == len(trace["calls"]) >= 2, "Incorrect live API count")
                    menu = {mix["id"]: mix["source"] for mix in trace["menu"]}
                    per_arm[arm]["selected"] += len(ids)
                    per_arm[arm]["selected_outside_greedy_top_five"] += int((take.is_myopic_greedy_pick == 0).sum())
                    per_arm[arm]["selected_from_non_elite_menu"] += sum(menu[mix] == "seeded_sample" for mix in ids)
                    counts.update(trace["usage"])
                    counts["http_responses"] += trace["live_generation_count"]
                    counts["decisions"] += 1
                observed.update(ids)
    expected_count = len(expected_seeds) * len(ARMS) * expected_rounds * 5
    require(status["measurements"] == len(selections) == expected_count, "Incorrect experiment count")
    return {"measured_queries": expected_count, "codeflow_response_models": sorted(model_ids),
            "llm_usage": dict(counts), "selection_diagnostics": {key: dict(value) for key, value in per_arm.items()}}


def compare_replay(original: Path, replay: Path) -> None:
    for stage in ("smoke", "concrete"):
        for table in TABLES:
            observed, regenerated = (pd.read_csv(root / stage / f"{table}.csv") for root in (original, replay))
            pd.testing.assert_frame_equal(observed, regenerated, check_exact=False, rtol=1e-10, atol=1e-10)
    for path in ("metrics", "error_strata", "position_ablation", "validation_grid", "exploratory_single_effect_transfer"):
        observed, regenerated = (pd.read_csv(root / "gb1" / f"{path}.csv") for root in (original, replay))
        pd.testing.assert_frame_equal(observed, regenerated, check_exact=False, rtol=1e-10, atol=1e-10)
    print("Archived decisions replayed and aggregate outputs reproduced.")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--evidence", type=Path, default=Path("evidence"))
    args = cli.parse_args()
    result = {
        "smoke": verify_concrete(args.evidence / "smoke", expected_seeds=[7], expected_rounds=1),
        "formal": verify_concrete(args.evidence / "concrete", expected_seeds=[17, 29, 43], expected_rounds=3),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
