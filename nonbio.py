"""A closed-loop benchmark over measured 28-day concrete strengths."""

from __future__ import annotations

import hashlib
import io
import itertools
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable
from urllib.request import urlopen

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SOURCE = "https://archive.ics.uci.edu/static/public/165/data.csv"
SOURCE_SHA256 = "8d4b15b6fc68cd932d745cbd663d5ceae66dd54422e99c1e4865f2936ab7e2af"
DEFAULT_DATA = Path(__file__).resolve().parent / "data" / "concrete.csv"
DATASET_DOI = "https://doi.org/10.24432/C5PK67"
DATASET_LICENSE = "https://creativecommons.org/licenses/by/4.0/"
RAW_COLUMNS = (
    "Cement", "Blast Furnace Slag", "Fly Ash", "Water", "Superplasticizer",
    "Coarse Aggregate", "Fine Aggregate", "Age", "Concrete compressive strength",
)
FEATURES = (
    "cement", "slag", "fly_ash", "water", "superplasticizer",
    "coarse_aggregate", "fine_aggregate", "age",
)
AGE = 28
INITIAL_SIZE, VALIDATION_SIZE, TEST_SIZE = 48, 96, 96
PREDICTOR = "RandomForest"
ARMS = ("A_random", "B_greedy", "C_llm", "D_knowledge_llm")
RIDGE_ALPHA = 10.0
RF_PARAMETERS = {
    "n_estimators": 128, "min_samples_leaf": 2, "max_features": 0.8, "n_jobs": 1,
}
SELECTOR = Callable[..., list[str]]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")


def read_dataset(path: Path) -> tuple[pd.DataFrame, dict]:
    if path.exists():
        blob = path.read_bytes()
    else:
        with urlopen(SOURCE, timeout=30) as response:
            blob = response.read()
        if hashlib.sha256(blob).hexdigest() != SOURCE_SHA256:
            raise ValueError("Downloaded Concrete bytes do not match the pinned hash.")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as dest:
            dest.write(blob)
    digest = hashlib.sha256(blob).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(f"Unexpected Concrete SHA-256 for {path}: {digest}")

    raw = pd.read_csv(io.BytesIO(blob))
    if tuple(raw.columns) != RAW_COLUMNS or len(raw) != 1030:
        raise ValueError("Unexpected Concrete data header or row count.")
    raw.columns = (*FEATURES, "strength")
    if not np.isfinite(raw.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite value in Concrete input.")
    if (raw[list(FEATURES)] < 0).any().any() or (raw.strength <= 0).any():
        raise ValueError("Invalid ingredient or target value in Concrete input.")
    selected = raw.loc[raw.age == AGE]
    table = (
        selected.groupby(list(FEATURES), sort=True, as_index=False)
        .agg(strength=("strength", "median"), replicates=("strength", "size"))
    )
    if len(selected) != 425 or len(table) != 416:
        raise ValueError("Unexpected number of 28-day recipes.")
    table.index = [f"mix-{i:03d}" for i in range(1, len(table) + 1)]
    return table, {
        "title": "Concrete Compressive Strength",
        "creator": "I-Cheng Yeh",
        "doi": DATASET_DOI,
        "source_url": SOURCE,
        "license": "CC BY 4.0",
        "license_url": DATASET_LICENSE,
        "sha256": digest,
        "size_bytes": len(blob),
        "source_columns": list(RAW_COLUMNS),
        "raw_rows": len(raw),
        "missing_cells": int(raw.isna().sum().sum()),
        "age_constraint_days": AGE,
        "rows_at_28_days": len(selected),
        "exact_duplicate_rows_at_28_days": int(selected.duplicated().sum()),
        "recipe_groups_with_replicates": int((table.replicates > 1).sum()),
        "extra_rows_within_recipe_groups": len(selected) - len(table),
        "unique_recipes_at_28_days": len(table),
        "target": "28-day compressive strength",
        "unit": "MPa",
        "direction": "maximize",
        "group_target": "median of rows with identical recipe and age",
        "limitations": "Archived measured strengths for a finite catalog. Source laboratory or study identifiers are unavailable.",
    }


def make_split(recipe_ids: list[str], seed: int) -> dict[str, list[str]]:
    if len(set(recipe_ids)) != len(recipe_ids):
        raise ValueError("Expected unique recipes before splitting.")
    ordered = np.random.default_rng(seed).permutation(sorted(recipe_ids)).tolist()
    end_validation = INITIAL_SIZE + VALIDATION_SIZE
    end_test = end_validation + TEST_SIZE
    if len(ordered) <= end_test:
        raise ValueError("Insufficient unique recipes for requested split sizes.")
    return {
        "initial": ordered[:INITIAL_SIZE],
        "validation": ordered[INITIAL_SIZE:end_validation],
        "test": ordered[end_validation:end_test],
        "pool": ordered[end_test:],
    }


class Oracle:
    """Single-arm budgeted reveal over exactly the preregistered candidate pool."""

    def __init__(self, measurements: dict[str, float], budget: int):
        self._measurements = measurements.copy()
        self._remaining = set(measurements)
        self._budget = budget

    def reveal(self, recipe_ids: list[str]) -> dict[str, float]:
        # Validate the complete batch before changing anything.
        if len(recipe_ids) != self._budget or len(set(recipe_ids)) != self._budget:
            raise ValueError("A reveal requires exactly one batch of distinct recipe IDs.")
        if not set(recipe_ids) <= self._remaining:
            raise ValueError("Attempt to reveal a recipe outside this arm's untested pool.")
        measurements = {recipe_id: self._measurements[recipe_id] for recipe_id in recipe_ids}
        self._remaining.difference_update(recipe_ids)
        return measurements


def train(observed: dict[str, dict], seed: int) -> dict:
    rows = list(observed.values())
    x = np.array([[row[name] for name in FEATURES] for row in rows])
    y = np.array([row["strength"] for row in rows])
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA)).fit(x, y),
        PREDICTOR: RandomForestRegressor(random_state=seed, **RF_PARAMETERS).fit(x, y),
    }


def predictive_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    cutoff = float(np.quantile(actual, 0.9))
    high = actual >= cutoff
    k = math.ceil(len(actual) / 10)
    best = np.argsort(-predicted, kind="stable")[:k]
    true_positive = int(high[best].sum())
    if np.ptp(predicted) == 0 or np.ptp(actual) == 0:
        pearson, spearman = None, None
    else:
        pearson = float(pearsonr(actual, predicted).statistic)
        spearman = float(spearmanr(actual, predicted).statistic)
    return {
        "n": len(actual),
        "pearson": pearson,
        "spearman": spearman,
        "mse_mpa_squared": float(mean_squared_error(actual, predicted)),
        "high_fitness_cutoff_mpa": cutoff,
        "average_precision": float(average_precision_score(high, predicted)),
        "precision_at_k": true_positive / k,
        "recall_at_k": true_positive / int(high.sum()),
        "k": k,
        "actual_high_count": int(high.sum()),
    }


def evaluate_models(models: dict, subset: pd.DataFrame, *, seed: int, arm: str, stage: str, round_number: int) -> list[dict]:
    features = subset[list(FEATURES)].to_numpy()
    actual = subset.strength.to_numpy()
    return [
        {
            "seed": seed, "arm": arm, "stage": stage, "round": round_number, "model": model_name,
            **predictive_metrics(actual, model.predict(features)),
        }
        for model_name, model in models.items()
    ]


def feature_view(table: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        recipe_id: dict(zip(FEATURES, map(float, row), strict=True))
        for recipe_id, row in table[list(FEATURES)].iterrows()
    }


def failure_examples(selections: list[dict]) -> list[dict]:
    over = min(selections, key=lambda row: row["actual_minus_predicted_mpa"])
    under = max(selections, key=lambda row: row["actual_minus_predicted_mpa"])
    examples = []
    if over["actual_minus_predicted_mpa"] < 0:
        examples.append({"failure_type": "largest_overestimate", **over})
    if under["actual_minus_predicted_mpa"] > 0:
        examples.append({"failure_type": "largest_underestimate", **under})
    return examples


def run_arm(
    *, seed: int, arm: str, rounds: int, budget: int, selector: SELECTOR,
    output: Path, pool: dict, observed_initial: dict, validation: pd.DataFrame,
    test: pd.DataFrame, oracle: Oracle, high_cutoff: float,
) -> dict:
    arm_output = output / "arms" / f"seed-{seed}" / arm
    arm_output.mkdir(parents=True, exist_ok=True)
    observed = {recipe_id: row.copy() for recipe_id, row in observed_initial.items()}
    reference_id = next(iter(observed))
    reference = observed[reference_id]["strength"]
    initial_best = max(row["strength"] for row in observed.values())
    all_visible_features = [
        [row[name] for name in FEATURES] for row in itertools.chain(pool.values(), observed.values())
    ]
    scale = np.std(np.array(all_visible_features), axis=0)
    scale[scale == 0] = 1
    write_json(arm_output / "initial_observed.json", observed)

    rng = np.random.default_rng(seed)
    round_rows, selection_rows, diagnostics = [], [], []
    first_hit = None
    tested = hit_count = 0
    best_so_far = initial_best
    models = train(observed, seed)
    diagnostics.extend(evaluate_models(models, validation, seed=seed, arm=arm, stage="validation", round_number=0))

    for round_number in range(1, rounds + 1):
        started = time.perf_counter()
        eligible = {recipe_id: pool[recipe_id] for recipe_id in sorted(pool) if recipe_id not in observed}
        ordered_ids = list(eligible)
        x = np.array([[eligible[recipe_id][name] for name in FEATURES] for recipe_id in ordered_ids])
        scores = dict(zip(ordered_ids, map(float, models[PREDICTOR].predict(x)), strict=True))
        tree_predictions = np.array([tree.predict(x) for tree in models[PREDICTOR].estimators_])
        disagreement = dict(zip(ordered_ids, map(float, np.std(tree_predictions, axis=0)), strict=True))
        ranked = sorted(eligible, key=lambda recipe_id: (-scores[recipe_id], recipe_id))
        ranks = {recipe_id: i for i, recipe_id in enumerate(ranked, 1)}
        if arm == ARMS[0]:
            selected = rng.choice(ordered_ids, size=budget, replace=False).tolist()
        elif arm == ARMS[1]:
            selected = ranked[:budget]
        else:
            selected = selector(
                eligible={recipe_id: row.copy() for recipe_id, row in eligible.items()},
                observed={recipe_id: row.copy() for recipe_id, row in observed.items()},
                scores=scores.copy(),
                uncertainty=disagreement.copy(),
                k=budget,
                seed=seed,
                round_number=round_number,
                augmented=arm == ARMS[3],
                log_path=arm_output / f"round-{round_number:02d}-agent.json",
            )
        reveal_started = time.perf_counter()
        batch = oracle.reveal(selected)
        observed_x = np.array([[row[name] for name in FEATURES] for row in observed.values()])
        batch_rows = []
        for batch_position, (recipe_id, strength) in enumerate(batch.items(), 1):
            tested += 1
            high = strength >= high_cutoff
            hit_count += int(high)
            if high and first_hit is None:
                first_hit = tested
            candidate = eligible[recipe_id]
            candidate_x = np.array([candidate[name] for name in FEATURES])
            binder = candidate["cement"] + candidate["slag"] + candidate["fly_ash"]
            batch_rows.append({
                "seed": seed, "arm": arm, "round": round_number, "batch_position": batch_position,
                "experiment_number": tested, "id": recipe_id, "strength_mpa": strength,
                "prediction_mpa": scores[recipe_id], "rf_rank": ranks[recipe_id], "eligible_count": len(eligible),
                "rf_tree_disagreement_mpa": disagreement[recipe_id],
                "actual_minus_predicted_mpa": strength - scores[recipe_id],
                "top_hit": int(high), "is_myopic_greedy_pick": int(recipe_id in ranked[:budget]),
                "nearest_observed_distance": float(np.linalg.norm((observed_x - candidate_x) / scale, axis=1).min()),
                "water_to_binder": candidate["water"] / binder,
                **candidate,
            })

        round_best = max(batch.values())
        best_so_far = max(best_so_far, round_best)
        observed.update({recipe_id: {**eligible[recipe_id], "strength": strength} for recipe_id, strength in batch.items()})
        models = train(observed, seed)
        diagnostics.extend(evaluate_models(models, validation, seed=seed, arm=arm, stage="validation", round_number=round_number))
        metrics = {
            "seed": seed, "arm": arm, "round": round_number, "budget": budget, "tested": tested,
            "round_best_mpa": round_best, "best_so_far_mpa": best_so_far,
            "reference_id": reference_id, "reference_mpa": reference,
            "initial_best_mpa": initial_best, "gain_vs_reference_mpa": best_so_far - reference,
            "gain_vs_initial_best_mpa": best_so_far - initial_best,
            "gain_vs_initial_best_pct": 100 * (best_so_far / initial_best - 1),
            "top_cutoff_mpa": high_cutoff, "round_top_hits": sum(row["top_hit"] for row in batch_rows),
            "top_hits": hit_count, "first_hit_tests": first_hit, "top_hit_rate": hit_count / tested,
            "greedy_overlap_at_k": len(set(selected).intersection(ranked[:budget])) / budget,
            "mean_rf_rank": float(np.mean([row["rf_rank"] for row in batch_rows])),
            "mean_rf_tree_disagreement_mpa": float(np.mean([row["rf_tree_disagreement_mpa"] for row in batch_rows])),
            "mean_nearest_observed_distance": float(np.mean([row["nearest_observed_distance"] for row in batch_rows])),
            "mean_water_to_binder": float(np.mean([row["water_to_binder"] for row in batch_rows])),
            "mean_abs_prediction_error_mpa": float(np.mean([abs(row["actual_minus_predicted_mpa"]) for row in batch_rows])),
        }
        selection_rows.extend(batch_rows)
        round_rows.append(metrics)
        write_csv(arm_output / f"round-{round_number:02d}-selections.csv", batch_rows)
        write_json(arm_output / f"round-{round_number:02d}.json", {
            **metrics, "pre_reveal_seconds": reveal_started - started,
            "post_reveal_seconds": time.perf_counter() - reveal_started,
            "number_observed_after_refit": len(observed), "contemporaneous_greedy_ids": ranked[:budget],
        })

    diagnostics.extend(evaluate_models(models, test, seed=seed, arm=arm, stage="final_test", round_number=rounds))
    write_csv(arm_output / "predictive.csv", diagnostics)
    return {
        "rounds": round_rows, "selections": selection_rows, "predictive": diagnostics,
        "failures": failure_examples(selection_rows),
        "methods": [{
            **round_rows[-1],
            "first_hit_censored": first_hit is None,
            "mean_overlap_all_rounds": float(np.mean([row["greedy_overlap_at_k"] for row in round_rows])),
            "mean_rf_rank_all_rounds": float(np.mean([row["mean_rf_rank"] for row in round_rows])),
            "mean_distance_all_rounds": float(np.mean([row["mean_nearest_observed_distance"] for row in round_rows])),
            "mean_water_to_binder_all_rounds": float(np.mean([row["mean_water_to_binder"] for row in round_rows])),
        }],
    }


def paired_comparisons(methods: list[dict]) -> list[dict]:
    indexed = {(row["seed"], row["arm"]): row for row in methods}
    comparisons = []
    for seed in sorted({row["seed"] for row in methods}):
        for earlier, later in itertools.combinations(ARMS, 2):
            left, right = indexed[seed, earlier], indexed[seed, later]
            both_hit = left["first_hit_tests"] is not None and right["first_hit_tests"] is not None
            comparisons.append({
                "seed": seed, "left_arm": earlier, "right_arm": later,
                "right_minus_left_best_mpa": right["best_so_far_mpa"] - left["best_so_far_mpa"],
                "right_minus_left_top_hits": right["top_hits"] - left["top_hits"],
                "right_minus_left_tests_to_hit": right["first_hit_tests"] - left["first_hit_tests"] if both_hit else None,
                "left_censored": left["first_hit_censored"], "right_censored": right["first_hit_censored"],
                "right_minus_left_mean_water_to_binder": (
                    right["mean_water_to_binder_all_rounds"] - left["mean_water_to_binder_all_rounds"]
                ),
            })
    return comparisons


def method_summary(methods: list[dict]) -> list[dict]:
    rows = []
    for arm in ARMS:
        group = [row for row in methods if row["arm"] == arm]
        completed_hits = [row["first_hit_tests"] for row in group if not row["first_hit_censored"]]
        hits = sum(row["top_hits"] for row in group)
        tests = sum(row["tested"] for row in group)
        rows.append({
            "arm": arm, "seeds": len(group), "total_tests": tests, "total_top_hits": hits,
            "hit_rate": hits / tests,
            "seeds_with_hit": len(completed_hits),
            "median_tests_to_hit_among_hit_seeds": float(np.median(completed_hits)) if completed_hits else None,
            "mean_final_best_mpa": float(np.mean([row["best_so_far_mpa"] for row in group])),
            "mean_gain_vs_initial_best_mpa": float(np.mean([row["gain_vs_initial_best_mpa"] for row in group])),
            "mean_overlap_all_rounds": float(np.mean([row["mean_overlap_all_rounds"] for row in group])),
            "mean_rf_rank_all_rounds": float(np.mean([row["mean_rf_rank_all_rounds"] for row in group])),
            "mean_distance_all_rounds": float(np.mean([row["mean_distance_all_rounds"] for row in group])),
            "mean_water_to_binder_all_rounds": float(np.mean([row["mean_water_to_binder_all_rounds"] for row in group])),
        })
    return rows


def run(
    output: Path, *, selector: SELECTOR, data_path: Path = DEFAULT_DATA,
    seeds: tuple[int, ...] = (17, 29, 43), rounds: int = 3, budget: int = 5,
    workers: int = 4, mode: str,
) -> dict:
    """Run arms in parallel with immutable shared splits and per-arm checkpoints."""
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("Supply distinct nonnegative seeds.")
    if rounds < 1 or budget < 1 or workers < 1:
        raise ValueError("Rounds, budget and workers must be positive.")
    table, source = read_dataset(data_path)
    if rounds * budget > len(table) - INITIAL_SIZE - VALIDATION_SIZE - TEST_SIZE:
        raise ValueError("The total budget per arm must fit inside the candidate pool.")
    features = feature_view(table)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    # Exclusive creation prevents two runners from accidentally sharing one result path.
    with (output / "status.json").open("x", encoding="utf-8") as status_file:
        json.dump({"status": "running"}, status_file)
    write_json(output / "source.json", source)
    write_json(output / "config.json", {
        "mode": mode, "seeds": seeds, "rounds": rounds, "budget": budget, "arms": ARMS,
        "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
        "predictor": PREDICTOR, "rf_parameters": RF_PARAMETERS, "ridge_alpha": RIDGE_ALPHA,
        "target_age_days": AGE, "primary_endpoint": "final incumbent strength minus shared initial best in MPa",
        "high_threshold": "linear 0.9 quantile within the initial eligible pool for each seed, used only in evaluation",
        "first_hit_effort": "count through the first threshold hit in recorded batch order, right censored at total budget",
        "stopping_condition": "fixed complete rounds for every arm",
        "rank_ties": "lexical mix ID",
    })
    futures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for seed in seeds:
            split = make_split(list(table.index), seed)
            write_json(output / "splits" / f"seed-{seed}.json", split)
            initial = {
                recipe_id: {**features[recipe_id], "strength": float(table.loc[recipe_id, "strength"])}
                for recipe_id in split["initial"]
            }
            pool = {recipe_id: features[recipe_id] for recipe_id in split["pool"]}
            pool_measurements = {recipe_id: float(table.loc[recipe_id, "strength"]) for recipe_id in pool}
            high_cutoff = float(np.quantile(list(pool_measurements.values()), 0.9))
            validation = table.loc[split["validation"]]
            test = table.loc[split["test"]]
            for arm in ARMS:
                futures.append(executor.submit(
                    run_arm, seed=seed, arm=arm, rounds=rounds, budget=budget, selector=selector,
                    output=output, pool=pool, observed_initial=initial, validation=validation,
                    test=test, oracle=Oracle(pool_measurements, budget), high_cutoff=high_cutoff,
                ))

    try:
        results = [future.result() for future in futures]
    except Exception as error:
        write_json(output / "status.json", {
            "status": "failed", "mode": mode, "error_type": type(error).__name__,
            "completed_arms": sum(future.exception() is None for future in futures),
        })
        raise
    for name in ("rounds", "selections", "predictive", "failures", "methods"):
        write_csv(output / f"{name}.csv", [row for arm in results for row in arm[name]])
    methods = [row for arm in results for row in arm["methods"]]
    write_csv(output / "comparisons.csv", paired_comparisons(methods))
    write_csv(output / "method_summary.csv", method_summary(methods))
    status = {"status": "complete", "mode": mode, "arms": len(results), "measurements": len(results) * rounds * budget}
    write_json(output / "status.json", status)
    return status
