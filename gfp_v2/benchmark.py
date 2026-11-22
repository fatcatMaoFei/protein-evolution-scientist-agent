"""Select on development folds, then evaluate the recorded choice on sealed GFP folds."""

from __future__ import annotations

import argparse
import csv
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score

from gfp_data import WT_ID, load_dataset
from gfp_v2 import predictor


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARTITION = ROOT / "evidence/gfp-v2/partition.json"
DEFAULT_OUTPUT = ROOT / "evidence/gfp-v2/prediction"
BUDGETS = ("low", "ample")
ACCURACY_METRICS = ("spearman", "pearson", "mse", "mae", "wt_ap", "decile_ap", "wt_precision_5",
                    "wt_precision_10", "decile_precision_5", "decile_precision_10", "top_5_measured_best",
                    "top_10_measured_best")
RISK_METRICS = ("risk_error_spearman", "low_risk_quartile_mae", "high_risk_quartile_mae",
                "calibrated_90_coverage", "calibrated_90_mean_width")


def _write_json(path: Path, data: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _corr(function, first, second):
    return float(function(first, second).statistic) if len(first) > 1 and np.ptp(first) and np.ptp(second) else None


def _precision_at_k(predictions, positive, k: int) -> float:
    k = min(k, len(predictions))
    cutoff = np.partition(predictions, len(predictions) - k)[-k]
    ahead, tied = predictions > cutoff, predictions == cutoff
    expected_hits = positive[ahead].sum() + (k - ahead.sum()) * positive[tied].mean()
    return float(expected_hits / k)


def accuracy(ids, measured, predicted, wt: float) -> dict:
    """Use separate wild-type and within-fold upper-decile classifications."""
    measured, predicted = np.asarray(measured), np.asarray(predicted)
    if measured.shape != predicted.shape or measured.ndim != 1 or len(ids) != len(measured) or not len(ids):
        raise ValueError("Expected aligned nonempty GFP observations and predictions")
    if not np.isfinite(measured).all() or not np.isfinite(predicted).all():
        raise ValueError("Nonfinite value in GFP predictions or observations")
    difference = measured - predicted
    ranking = np.lexsort((np.asarray(ids), -predicted))
    decile_cutoff = float(np.quantile(measured, 0.90))
    result = {
        "n": len(ids), "wt_threshold": wt, "decile_threshold": decile_cutoff,
        "spearman": _corr(spearmanr, measured, predicted), "pearson": _corr(pearsonr, measured, predicted),
        "mse": float(np.mean(difference ** 2)), "mae": float(np.mean(np.abs(difference))),
    }
    for name, positive in (("wt", measured > wt), ("decile", measured >= decile_cutoff)):
        result[f"{name}_positive_n"] = int(positive.sum())
        result[f"{name}_ap"] = float(average_precision_score(positive, predicted)) if positive.any() else None
        for k in (5, 10):
            result[f"{name}_precision_{k}"] = _precision_at_k(predicted, positive, k)
    for k in (5, 10):
        best = ranking[:k]
        result[f"top_{k}_measured_best"] = float(measured[best].max())
        result[f"top_{k}_measured_mean"] = float(measured[best].mean())
        result[f"top_{k}_observed_wt_hits"] = int(np.sum(measured[best] > wt))
    return result


def _risk_weight(risk, scale):
    return np.clip(risk / scale, 0.5, 2.0)


def calibrate(risk, residual) -> dict:
    """Learn descriptive interval widths on development predictions only."""
    risk, residual = np.asarray(risk), np.asarray(residual)
    if risk.shape != residual.shape or not len(risk) or not np.isfinite(risk).all() or not np.isfinite(residual).all():
        raise ValueError("Calibration needs aligned finite development predictions and errors")
    scale = max(float(np.median(risk)), 1e-6)
    normalized_residual = np.abs(residual) / _risk_weight(risk, scale)
    index = min(len(residual) - 1, int(np.ceil(0.90 * (len(residual) + 1))) - 1)
    quantile = float(np.partition(normalized_residual, index)[index])
    return {"risk_scale": scale, "interval_multiplier": quantile, "calibration_n": len(residual)}


def risk_metrics(risk, residual, calibration) -> dict:
    risk, absolute = np.asarray(risk), np.abs(residual)
    if risk.shape != absolute.shape or not np.isfinite(risk).all() or np.any(risk < 0):
        raise ValueError("Each prediction needs a finite nonnegative risk estimate")
    order = np.argsort(risk, kind="stable")
    quartile = max(1, len(risk) // 4)
    half_width = _risk_weight(risk, calibration["risk_scale"]) * calibration["interval_multiplier"]
    return {
        "risk_error_spearman": _corr(spearmanr, risk, absolute),
        "low_risk_quartile_mae": float(np.mean(absolute[order[:quartile]])),
        "high_risk_quartile_mae": float(np.mean(absolute[order[-quartile:]])),
        "calibrated_90_coverage": float(np.mean(absolute <= half_width)),
        "calibrated_90_mean_width": float(np.mean(2 * half_width)),
    }


def _blocks(partition: dict, role: str) -> list[dict]:
    records = partition[f"prediction_{'development' if role == 'validation' else 'test'}"]
    for record in records:
        low, ample, holdout = (set(record[field]) for field in ("train_low", "train_ample", role))
        if not low <= ample or holdout & ample or any(
            len(record[name]) != len(set(record[name])) for name in ("train_low", "train_ample", role)
        ):
            raise ValueError("Prediction holdouts must be disjoint from nested training observations")
    return records


def _predictions(dataset, partition: dict, role: str, models: dict[str, tuple[str, ...]]) -> tuple[list[dict], dict]:
    rows, groups = [], {}
    for block in _blocks(partition, role):
        ids = block[role]
        query = [dataset.variants[key] for key in ids]
        labels = np.array([dataset.scores[key] for key in ids])
        seed = int(block["id"])
        for budget in BUDGETS:
            train_ids = block[f"train_{budget}"]
            train = [dataset.variants[key] for key in train_ids]
            targets = np.array([dataset.scores[key] for key in train_ids])
            observed_edits = {edit for variant in train for edit in variant.edits}
            observed_sites = {edit[1:-1] for edit in observed_edits}
            for name in models[budget]:
                model = predictor.fit(train, targets, seed, name)
                predicted, risk = model.estimate(query)
                groups[budget, name, block["id"]] = (ids, labels, predicted, risk)
                rows.extend({
                    "block": block["id"], "budget": budget, "train_size": len(train), "model": name,
                    "variant_id": key, "measured": float(label), "predicted": float(mean), "risk": float(width),
                    "mutation_count": len(variant.edits),
                    "unseen_edits": sum(edit not in observed_edits for edit in variant.edits),
                    "unseen_sites": sum(edit[1:-1] not in observed_sites for edit in variant.edits),
                } for key, label, mean, width, variant in zip(ids, labels, predicted, risk, query, strict=True))
    return rows, groups


def _means(rows: list[dict], fields) -> list[dict]:
    groups = {}
    for row in rows:
        groups.setdefault((row["budget"], row["model"]), []).append(row)
    summary = []
    for (budget, model), group in groups.items():
        result = {"budget": budget, "model": model, "blocks": len(group)}
        for field in fields:
            values = [row[field] for row in group if row[field] is not None]
            result[field] = float(np.mean(values)) if values else None
        summary.append(result)
    return summary


def _choose(summary, budget, names):
    candidates = [row for row in summary if row["budget"] == budget and row["model"] in names]
    order = ("wt_ap", "spearman", "decile_ap")
    return max(candidates, key=lambda row: (*(row[key] if row[key] is not None else -np.inf for key in order),
                                            -predictor.MODELS.index(row["model"])))["model"]


def _fingerprints(partition: Path):
    return {"partition_sha256": _digest(partition), "predictor_sha256": _digest(Path(predictor.__file__)),
            "benchmark_sha256": _digest(Path(__file__))}


def select(dataset, partition_path: Path, output: Path) -> dict:
    """Access development data only, serialize all scores and lock the winner."""
    partition = _read_json(partition_path)
    rows, groups = _predictions(dataset, partition, "validation", dict.fromkeys(BUDGETS, predictor.MODELS))
    metrics, calibration = [], {}
    wt = dataset.scores[WT_ID]
    for (budget, model, block), (ids, measured, predicted, _) in groups.items():
        metrics.append({"budget": budget, "model": model, "block": block, **accuracy(ids, measured, predicted, wt)})
    summary = _means(metrics, ACCURACY_METRICS)
    for budget in BUDGETS:
        calibration[budget] = {}
        for model in predictor.MODELS:
            data = [item for key, item in groups.items() if key[:2] == (budget, model)]
            calibration[budget][model] = calibrate(np.concatenate([item[3] for item in data]),
                                                   np.concatenate([item[1] - item[2] for item in data]))
    locked = {**_fingerprints(partition_path), "model_order": list(predictor.MODELS),
              "selection_rule": "Mean development WT AP, then Spearman, then upper-decile AP, then fixed model order",
              "selected": {budget: _choose(summary, budget, predictor.MODELS) for budget in BUDGETS},
              "best_baseline": {budget: _choose(summary, budget, predictor.BASELINES) for budget in BUDGETS},
              "best_enhanced": {budget: _choose(summary, budget, predictor.ENHANCED) for budget in BUDGETS},
              "development_calibration": calibration}
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "validation_predictions.csv", rows)
    _write_csv(output / "validation_metrics.csv", metrics)
    _write_csv(output / "validation_summary.csv", summary)
    locked["validation_artifact_sha256"] = {path.name: _digest(path) for path in output.glob("validation*.csv")}
    _write_json(output / "selection.json", locked)
    return locked


def _paired(metrics: list[dict], locked: dict) -> list[dict]:
    results = []
    rng = np.random.default_rng(7318)
    for budget in BUDGETS:
        winner = locked["selected"][budget]
        for left in dict.fromkeys((winner, locked["best_enhanced"][budget])):
            grouped = {(row["model"], row["block"]): row for row in metrics if row["budget"] == budget}
            blocks = [block for name, block in grouped if name == left]
            for right in predictor.BASELINES:
                for metric in ACCURACY_METRICS:
                    values = np.array([grouped[left, block][metric] - grouped[right, block][metric]
                                       for block in blocks if grouped[left, block][metric] is not None
                                       and grouped[right, block][metric] is not None])
                    if len(values):
                        simulations = rng.choice(values, (5000, len(values)), replace=True).mean(axis=1)
                        low, high = np.quantile(simulations, (0.025, 0.975))
                        results.append({"budget": budget, "model": left, "baseline": right, "metric": metric,
                                        "blocks": len(values), "mean_difference": float(values.mean()),
                                        "descriptive_bootstrap_low": float(low), "descriptive_bootstrap_high": float(high),
                                        "block_values_json": json.dumps(values.tolist())})
    return results


def _within_holdout_ap(groups, locked, wt):
    """Paired genotype sampling conditional on the fitted models and these four test blocks."""
    rng = np.random.default_rng(98851)
    results = []
    for budget in BUDGETS:
        winner, baseline = locked["selected"][budget], locked["best_baseline"][budget]
        deltas = []
        for (regime, model, block), (ids, measured, predicted, _) in groups.items():
            if (regime, model) != (budget, winner):
                continue
            baseline_ids, _, prediction_v1, _ = groups[budget, baseline, block]
            if ids != baseline_ids:
                raise ValueError("Paired AP needs the same holdout genotypes")
            positive = measured > wt
            samples = rng.integers(0, len(ids), size=(1000, len(ids)))
            observed_delta = []
            for sample in samples:
                truth = positive[sample]
                observed_delta.append(
                    float(average_precision_score(truth, predicted[sample])
                          - average_precision_score(truth, prediction_v1[sample])) if truth.any() else 0.0
                )
            deltas.append(observed_delta)
        lower, upper = np.quantile(np.mean(deltas, axis=0), (0.025, 0.975))
        results.append({"budget": budget, "model": winner, "baseline": baseline, "metric": "wt_ap",
                        "bootstrap_low": float(lower), "bootstrap_high": float(upper),
                        "replicates": 1000, "scope": "conditional on these fitted models and fixed test blocks"})
    return results


def evaluate(dataset, partition_path: Path, output: Path) -> None:
    """Verify selection artifacts before reading holdout labels."""
    locked = _read_json(output / "selection.json")
    if any(locked[key] != value for key, value in _fingerprints(partition_path).items()) or any(
        _digest(output / name) != expected for name, expected in locked["validation_artifact_sha256"].items()
    ):
        raise ValueError("Frozen validation inputs changed after model selection")
    if (output / "test_predictions.csv").exists():
        raise FileExistsError("The sealed test output already exists")
    partition = _read_json(partition_path)
    if _keys(_blocks(partition, "validation"), "validation") & _keys(_blocks(partition, "test"), "test") - {WT_ID}:
        raise ValueError("Development and test blocks must be disjoint except the common WT reference")
    models = {budget: tuple(dict.fromkeys((*predictor.BASELINES, locked["selected"][budget],
                                          locked["best_enhanced"][budget]))) for budget in BUDGETS}
    rows, groups = _predictions(dataset, partition, "test", models)
    metrics = []
    wt = dataset.scores[WT_ID]
    for (budget, model, block), (ids, measured, predicted, risk) in groups.items():
        metrics.append({"budget": budget, "model": model, "block": block, **accuracy(ids, measured, predicted, wt),
                        **risk_metrics(risk, measured - predicted, locked["development_calibration"][budget][model])})
    _write_csv(output / "test_predictions.csv", rows)
    _write_csv(output / "test_metrics.csv", metrics)
    _write_csv(output / "test_summary.csv", _means(metrics, (*ACCURACY_METRICS, *RISK_METRICS)))
    _write_csv(output / "test_paired.csv", _paired(metrics, locked))
    _write_csv(output / "test_within_holdout_ap.csv", _within_holdout_ap(groups, locked, wt))


def _keys(blocks, role):
    return {key for block in blocks for field in ("train_ample", role) for key in block[field]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("select", "test"))
    parser.add_argument("--partition", type=Path, default=DEFAULT_PARTITION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-cache", type=Path)
    args = parser.parse_args()
    dataset = load_dataset(args.data_cache)
    function = select if args.phase == "select" else evaluate
    result = function(dataset, args.partition, args.output)
    if result:
        print(json.dumps({name: result[name] for name in ("selected", "best_baseline", "best_enhanced")}))


if __name__ == "__main__":
    main()
