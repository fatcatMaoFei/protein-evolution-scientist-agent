"""Run a second LLM family against the frozen DeepSeek materials benchmark."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path

import numpy as np

from nonbio import (
    AGE,
    ARMS,
    DEFAULT_DATA,
    PREDICTOR,
    RF_PARAMETERS,
    RIDGE_ALPHA,
    SELECTOR,
    Oracle,
    feature_view,
    make_split,
    read_dataset,
    run_arm,
    write_csv,
    write_json,
)

LLM_ARMS = ARMS[2:]
FORMAL_SEEDS = (17, 29, 43)
SMOKE_SEEDS = (7,)
BUDGET = 5


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _archive_config(baseline: Path, seeds: tuple[int, ...], rounds: int, smoke: bool) -> dict:
    config = _load(baseline / "config.json")
    mode = "smoke" if smoke else "formal"
    status = _load(baseline / "status.json")
    if status != {
        "status": "complete",
        "mode": mode,
        "arms": len(seeds) * len(ARMS),
        "measurements": len(seeds) * len(ARMS) * rounds * BUDGET,
    }:
        raise ValueError("Paired baseline does not contain the complete original benchmark.")
    expected = {
        "mode": mode, "seeds": list(seeds), "rounds": rounds, "budget": BUDGET, "arms": list(ARMS),
        "predictor": PREDICTOR, "rf_parameters": RF_PARAMETERS, "ridge_alpha": RIDGE_ALPHA,
        "target_age_days": AGE,
        "numpy": version("numpy"), "pandas": version("pandas"),
        "scipy": version("scipy"), "scikit_learn": version("scikit-learn"),
    }
    for field, value in expected.items():
        if config.get(field) != value:
            raise ValueError(f"Paired baseline configuration differs on {field}.")
    return config


def _paired_menu(
    baseline: Path, seed: int, observed: dict, budget: int,
) -> tuple[list[dict], dict[str, str], str]:
    manifests = {}
    menus = []
    models = set()
    for arm in ARMS:
        arm_path = baseline / "arms" / f"seed-{seed}" / arm
        observations_file = arm_path / "initial_observed.json"
        if list(_load(observations_file).items()) != list(observed.items()):
            raise ValueError(f"Paired baseline initial observations differ for seed {seed}, arm {arm}.")
        manifests[str(observations_file.relative_to(baseline))] = _sha256(observations_file)
        if arm in LLM_ARMS:
            trace_file = arm_path / "round-01-agent.json"
            trace = _load(trace_file)
            if (
                trace.get("status") != "ok"
                or trace.get("mode") != "live"
                or trace.get("seed") != seed
                or trace.get("round") != 1
                or trace.get("budget") != budget
                or trace.get("knowledge_augmented") != (arm == LLM_ARMS[1])
            ):
                raise ValueError(f"Invalid paired first-round baseline trace for seed {seed}, arm {arm}.")
            models.add(trace["model"])
            manifests[str(trace_file.relative_to(baseline))] = _sha256(trace_file)
            menus.append(trace["menu"])
    if len(models) != 1 or menus[0] != menus[1]:
        raise ValueError(f"Paired first-round menus or model differ across archived arms for seed {seed}.")
    return menus[0], manifests, models.pop()


def _same_menu(expected: list[dict], actual: list[dict]) -> bool:
    if len(expected) != len(actual):
        return False
    for source, reproduced in zip(expected, actual, strict=True):
        for field in ("alias", "id", "source", "rank", "inputs"):
            if source[field] != reproduced[field]:
                return False
        for field in ("prediction", "rf_tree_disagreement"):
            if not math.isclose(source[field], reproduced[field], rel_tol=0, abs_tol=1e-10):
                return False
    return True


def _summary(methods: list[dict]) -> list[dict]:
    rows = []
    for arm in LLM_ARMS:
        group = [row for row in methods if row["arm"] == arm]
        confirmed_hits = [row["first_hit_tests"] for row in group if not row["first_hit_censored"]]
        hits = sum(row["top_hits"] for row in group)
        tests = sum(row["tested"] for row in group)
        rows.append({
            "arm": arm,
            "seeds": len(group),
            "total_tests": tests,
            "total_top_hits": hits,
            "hit_rate": hits / tests,
            "seeds_with_hit": len(confirmed_hits),
            "median_tests_to_hit_among_hit_seeds": float(np.median(confirmed_hits)) if confirmed_hits else None,
            "mean_final_best_mpa": float(np.mean([row["best_so_far_mpa"] for row in group])),
            "mean_gain_vs_initial_best_mpa": float(np.mean([row["gain_vs_initial_best_mpa"] for row in group])),
            **{
                field: float(np.mean([row[field] for row in group]))
                for field in (
                    "mean_overlap_all_rounds", "mean_rf_rank_all_rounds", "mean_distance_all_rounds",
                    "mean_water_to_binder_all_rounds",
                )
            },
        })
    return rows


def run(
    output: Path, *, baseline: Path, selector: SELECTOR, treatment_model: str, freeze_path: Path,
    smoke: bool = False, workers: int = 2, data_path: Path = DEFAULT_DATA,
) -> dict:
    """Match a completed baseline before querying a second model on either knowledge arm."""
    output = output.resolve()
    baseline = baseline.resolve()
    if output == baseline or baseline in output.parents:
        raise ValueError("Factorial output must be outside the original archive.")
    if workers < 1:
        raise ValueError("Workers must be positive.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    freeze_content = freeze_path.read_bytes()
    if not freeze_content or not treatment_model.strip():
        raise ValueError("A frozen protocol and an explicit treatment model are required.")

    seeds = SMOKE_SEEDS if smoke else FORMAL_SEEDS
    rounds = 1 if smoke else 3
    config = _archive_config(baseline, seeds, rounds, smoke)
    table, source = read_dataset(data_path)
    if _load(baseline / "source.json") != source:
        raise ValueError("Paired baseline source audit differs from the measured catalog.")
    features = feature_view(table)
    manifests = {name: _sha256(baseline / name) for name in ("config.json", "source.json", "status.json")}
    states, archived_models = {}, set()
    for seed in seeds:
        split_name = f"splits/seed-{seed}.json"
        split = make_split(list(table.index), seed)
        if _load(baseline / split_name) != split:
            raise ValueError(f"Paired baseline split differs for seed {seed}.")
        manifests[split_name] = _sha256(baseline / split_name)
        initial = {
            recipe_id: {**features[recipe_id], "strength": float(table.loc[recipe_id, "strength"])}
            for recipe_id in split["initial"]
        }
        menu, arm_manifests, model = _paired_menu(baseline, seed, initial, BUDGET)
        manifests.update(arm_manifests)
        archived_models.add(model)
        states[seed] = (split, initial, menu)
    if len(archived_models) != 1 or treatment_model in archived_models:
        raise ValueError("The paired comparison requires one archived model and a distinct treatment model.")

    output.mkdir(parents=True, exist_ok=True)
    mode = "smoke" if smoke else "formal"
    status_path = output / "status.json"
    with status_path.open("x", encoding="utf-8") as stream:
        json.dump({"status": "running", "mode": mode}, stream)
    write_json(output / "source.json", source)
    freeze_digest = hashlib.sha256(freeze_content).hexdigest()
    freeze_name = "protocol-freeze" + freeze_path.suffix
    (output / freeze_name).write_bytes(freeze_content)
    write_json(output / "config.json", {
        **config, "arms": LLM_ARMS, "python": sys.version.split()[0],
        "study": "paired LLM family by knowledge factorial",
        "model": treatment_model,
        "baseline_model": archived_models.pop(),
        "baseline_evidence": os.path.relpath(baseline, output),
        "baseline_file_sha256": manifests,
        "frozen_protocol_file": freeze_name,
        "frozen_protocol_sha256": freeze_digest,
    })

    def matched_selector(**kwargs) -> list[str]:
        # Compare the first-round treatment context against a label-free archived menu.
        if kwargs["round_number"] == 1:
            expected = states[kwargs["seed"]][2]
            if any(
                kwargs["eligible"].get(row["id"]) != row["inputs"]
                or not math.isclose(kwargs["scores"][row["id"]], row["prediction"], rel_tol=0, abs_tol=1e-10)
                or not math.isclose(
                    kwargs["uncertainty"][row["id"]], row["rf_tree_disagreement"], rel_tol=0, abs_tol=1e-10,
                )
                for row in expected
            ):
                raise ValueError("Current first-round RF features or scores differ from the original run.")
        selected = selector(**kwargs)
        recorded = _load(kwargs["log_path"])
        contract = {
            "status": "ok", "model": treatment_model, "seed": kwargs["seed"], "round": kwargs["round_number"],
            "budget": kwargs["k"], "knowledge_augmented": kwargs["augmented"],
        }
        if any(recorded.get(key) != value for key, value in contract.items()):
            raise ValueError("Selector evidence does not match the frozen treatment model or arm.")
        if recorded.get("decision", {}).get("selected_ids") != selected:
            raise ValueError("Selector result does not match its recorded decision.")
        if kwargs["round_number"] == 1 and not _same_menu(expected, recorded["menu"]):
            raise ValueError("Treatment's first-round menu differs from the paired baseline.")
        return selected

    futures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for seed, (split, initial, _) in states.items():
            write_json(output / "splits" / f"seed-{seed}.json", split)
            pool = {recipe_id: features[recipe_id] for recipe_id in split["pool"]}
            measurements = {recipe_id: float(table.loc[recipe_id, "strength"]) for recipe_id in pool}
            high_cutoff = float(np.quantile(list(measurements.values()), 0.9))
            for arm in LLM_ARMS:
                futures.append(executor.submit(
                    run_arm, seed=seed, arm=arm, rounds=rounds, budget=BUDGET, selector=matched_selector,
                    output=output, pool=pool, observed_initial=initial, validation=table.loc[split["validation"]],
                    test=table.loc[split["test"]], oracle=Oracle(measurements, BUDGET), high_cutoff=high_cutoff,
                ))
    try:
        results = [future.result() for future in futures]
        for name in ("rounds", "selections", "predictive", "failures", "methods"):
            write_csv(output / f"{name}.csv", [row for arm in results for row in arm[name]])
        write_csv(output / "method_summary.csv", _summary([row for arm in results for row in arm["methods"]]))
    except Exception as error:
        write_json(status_path, {
            "status": "failed", "mode": mode, "error_type": type(error).__name__,
            "completed_arms": sum(future.exception() is None for future in futures),
        })
        raise
    status = {
        "status": "complete", "mode": mode, "arms": len(results),
        "measurements": len(results) * rounds * BUDGET,
    }
    write_json(status_path, status)
    return status
