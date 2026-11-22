"""Run the prospectively frozen, zero-API GFP acquisition ablation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from statistics import mean

import numpy as np

from gfp_data import fit_predictor, load_dataset
from gfp_evolution import MeasuredGFPOracle

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "gfp_quality_protocol.md"
BLOCKS = range(101, 113)
POLICIES = ("random", "rf", "ridge")
INITIAL_SIZE = 95
POOL_SIZE = 384
ROUNDS = 3
BUDGET = 5


def partition(keys, excluded, wt, *, blocks=BLOCKS):
    ids = sorted(set(keys) - set(excluded) - {wt})
    random.Random(713803).shuffle(ids)
    width = INITIAL_SIZE + POOL_SIZE
    if len(ids) < len(blocks) * width:
        raise ValueError("Not enough unexposed GFP IDs for the frozen supplemental blocks")
    return {
        block: {"initial": [wt, *ids[start:start + INITIAL_SIZE]],
                "pool": ids[start + INITIAL_SIZE:start + width]}
        for block, start in zip(blocks, range(0, len(blocks) * width, width), strict=True)
    }


def original_ids(root: Path = ROOT) -> set[str]:
    folders = [root / "evidence/gfp" / stage / "deepseek" / "splits" for stage in ("formal", "smoke")]
    folders.append(root / "evidence/gfp/configured-smoke/splits")
    union = set()
    for folder in folders:
        files = sorted(folder.glob("seed-*.json"))
        if not files:
            raise ValueError(f"Missing exclusion split manifest: {folder}")
        for file in files:
            roles = json.loads(file.read_text())
            if set(roles) != {"initial", "validation", "pool", "test"}:
                raise ValueError(f"Invalid original split roles: {file}")
            union.update(key for ids in roles.values() for key in ids)
    return union


def run_policy(dataset, block: int, split: dict, policy: str):
    observed = {key: dataset.scores[key] for key in split["initial"]}
    wt = observed[dataset.wild_type_id]
    initial_best = max(observed.values())
    eligible = list(split["pool"])
    oracle = MeasuredGFPOracle({key: dataset.scores[key] for key in eligible}, BUDGET)
    rng = random.Random(block)
    selections, rounds, rankings = [], [], []
    for number in range(1, ROUNDS + 1):
        scores = {}
        if policy == "random":
            batch = rng.sample(eligible, BUDGET)
        else:
            fit = fit_predictor([dataset.variants[key] for key in observed], list(observed.values()), block, model=policy)
            scores = dict(zip(eligible, map(float, fit.predict([dataset.variants[key] for key in eligible])), strict=True))
            ranked = sorted(scores, key=lambda key: (-scores[key], key))
            batch = ranked[:BUDGET]
            rankings.extend({"block": block, "policy": policy, "round": number, "rank": rank,
                             "mutant": key, "prediction": scores[key]}
                            for rank, key in enumerate(ranked, 1))
        feedback = oracle.reveal(batch)
        batch_values = list(feedback.values())
        rounds.append({"block": block, "policy": policy, "round": number, "batch_best": max(batch_values),
                       "batch_mean": mean(batch_values), "above_wt": sum(value > wt for value in batch_values)})
        selections.extend({"block": block, "policy": policy, "round": number, "position": position,
                           "mutant": key, "prediction": scores.get(key), "fitness": value}
                          for position, (key, value) in enumerate(feedback.items(), 1))
        observed.update(feedback)
        eligible = [key for key in eligible if key not in feedback]
    values = [row["fitness"] for row in selections]
    method = {"block": block, "policy": policy, "initial_best": initial_best, "new_best": max(values),
              "new_mean": mean(values), "above_wt": sum(value > wt for value in values),
              "incumbent_gain": max(0.0, max(values) - initial_best)}
    return method, rounds, selections, rankings


def paired_summary(methods: list[dict]) -> tuple[list[dict], list[dict]]:
    by_pair = {(row["policy"], row["block"]): row for row in methods}
    blocks = sorted({row["block"] for row in methods})
    metrics = ("new_best", "new_mean", "above_wt", "incumbent_gain")
    generator = np.random.default_rng(80177)
    resamples = generator.integers(len(blocks), size=(10_000, len(blocks)))
    paired, summary = [], []
    for policy, control in (("ridge", "rf"), ("rf", "random"), ("ridge", "random")):
        pairs = [
            {"policy": policy, "control": control, "block": block,
             **{metric: by_pair[policy, block][metric] - by_pair[control, block][metric] for metric in metrics}}
            for block in blocks
        ]
        values = np.array([row["new_best"] for row in pairs])
        low, high = np.quantile(values[resamples].mean(axis=1), [0.025, 0.975])
        summary.append({"policy": policy, "control": control, "blocks": len(blocks),
                        **{metric: mean(row[metric] for row in pairs) for metric in metrics},
                        "new_best_ci_low": float(low), "new_best_ci_high": float(high),
                        "new_best_wins": int((values > 0).sum()), "new_best_losses": int((values < 0).sum()),
                        "new_best_ties": int((values == 0).sum())})
        paired.extend(pairs)
    return paired, summary


def write_csv(path: Path, records: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(output: Path, *, cache=None) -> None:
    dataset = load_dataset(cache)
    excluded = original_ids()
    split = partition(dataset.variants, excluded, dataset.wild_type_id)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "splits.json", split)
    write_json(output / "config.json", {"protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
                 "blocks": list(BLOCKS), "policies": POLICIES, "initial_size": INITIAL_SIZE + 1,
                 "pool_size": POOL_SIZE, "rounds": ROUNDS, "budget": BUDGET,
                 "excluded_original_ids": len(excluded), "dataset": dataset.provenance})
    methods, rounds, selections, rankings = [], [], [], []
    for block in BLOCKS:
        for policy in POLICIES:
            method, round_items, picked, scores = run_policy(dataset, block, split[block], policy)
            methods.append(method)
            rounds.extend(round_items)
            selections.extend(picked)
            rankings.extend(scores)
    pairing, contrasts = paired_summary(methods)
    headroom = []
    for block, roles in split.items():
        baseline = max(dataset.scores[key] for key in roles["initial"])
        values = [dataset.scores[key] for key in roles["pool"]]
        headroom.append({"block": block, "initial_best": baseline, "pool_best": max(values),
                         "incumbent_exceeding_count": sum(value > baseline for value in values)})
    for filename, records in (("methods", methods), ("rounds", rounds), ("selections", selections),
                              ("rankings", rankings), ("paired", pairing), ("contrasts", contrasts),
                              ("headroom", headroom)):
        write_csv(output / f"{filename}.csv", records)
    write_json(output / "status.json", {"status": "complete", "queries": len(selections), "api_calls": 0,
                                        "paired_blocks": len(split)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-cache", type=Path)
    args = parser.parse_args()
    run(args.output, cache=args.data_cache)
