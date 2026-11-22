"""Separate prediction-only ablation with matched budgets and measured outcomes."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from factorial_runner import BUDGET, FORMAL_SEEDS, _archive_config, _load
from nonbio import FEATURES, DEFAULT_DATA, Oracle, feature_view, read_dataset, run_arm, train, write_csv, write_json

POLICIES = ("rf_greedy", "rf_ucb", "ensemble_greedy", "ensemble_ucb")


def select(policy, *, eligible, observed, scores, uncertainty, k, seed, log_path, **_):
    ids = sorted(eligible)
    if policy.startswith("ensemble"):
        model = train(observed, seed)["Ridge"]
        features = np.array([[eligible[key][field] for field in FEATURES] for key in ids])
        ridge = dict(zip(ids, model.predict(features), strict=True))
        predicted = {key: float((scores[key] + ridge[key]) / 2) for key in ids}
    else:
        predicted = scores
    beta = 1.0 if policy.endswith("ucb") else 0.0
    utility = {key: predicted[key] + beta * uncertainty[key] for key in ids}
    selected = sorted(ids, key=lambda key: (-utility[key], key))[:k]
    write_json(log_path, {
        "mode": "deterministic_prediction_only", "policy": policy, "uncertainty_multiplier": beta,
        "selected": [
            {"id": key, "acquisition_score": utility[key], "prediction_mpa": predicted[key],
             "rf_disagreement_mpa": uncertainty[key]} for key in selected
        ],
    })
    return selected


def run(baseline: Path, output: Path):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    config = _archive_config(baseline, FORMAL_SEEDS, 3, False)
    table, source = read_dataset(DEFAULT_DATA)
    if source != _load(baseline / "source.json"):
        raise ValueError("Ablation source differs from main study")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", {
        **config, "mode": "exploratory", "arms": POLICIES, "model_weight_rf": 0.5,
        "model_weight_ridge": 0.5, "ucb_uncertainty_multiplier": 1.0,
    })
    write_json(output / "source.json", source)
    features = feature_view(table)
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = []
        for seed in FORMAL_SEEDS:
            split = _load(baseline / "splits" / f"seed-{seed}.json")
            write_json(output / "splits" / f"seed-{seed}.json", split)
            initial = {
                key: {**features[key], "strength": float(table.loc[key, "strength"])}
                for key in split["initial"]
            }
            pool = {key: features[key] for key in split["pool"]}
            actual = {key: float(table.loc[key, "strength"]) for key in pool}
            for policy in POLICIES:
                futures.append(executor.submit(
                    run_arm, seed=seed, arm=policy, rounds=3, budget=BUDGET,
                    selector=partial(select, policy), output=output, pool=pool, observed_initial=initial,
                    validation=table.loc[split["validation"]], test=table.loc[split["test"]],
                    oracle=Oracle(actual, BUDGET), high_cutoff=float(np.quantile(list(actual.values()), 0.9)),
                ))
        results = [task.result() for task in futures]
    for name in ("selections", "rounds", "methods"):
        write_csv(output / f"{name}.csv", [row for result in results for row in result[name]])
    original = pd.read_csv(baseline / "selections.csv")
    original = original[original.arm == "B_greedy"]
    selected = pd.read_csv(output / "selections.csv")
    if original.id.tolist() != selected[selected.arm == "rf_greedy"].id.tolist():
        raise ValueError("Exploratory RF greedy failed to reproduce the archived comparator")
    summary = [
        {
            "arm": policy, "mean_final_gain_mpa": float(np.mean([
                result["methods"][0]["gain_vs_initial_best_mpa"] for result in results
                if result["methods"][0]["arm"] == policy
            ])),
            "total_top_decile_hits": int(selected.loc[selected.arm == policy, "top_hit"].sum()),
            "measured_queries": int((selected.arm == policy).sum()),
        }
        for policy in POLICIES
    ]
    write_csv(output / "summary.csv", summary)
    write_json(output / "status.json", {"status": "complete", "measurements": len(selected), "mode": "exploratory"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=Path("evidence/concrete"), type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.baseline, args.output)
