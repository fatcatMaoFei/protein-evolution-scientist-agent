"""Validate the two frozen GFP predictors without querying the acquisition or test splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from gfp_data import WT_ID, fit_predictor, load_dataset, metrics

ARCHIVE = Path(__file__).resolve().parent / "evidence/gfp/benchmark"
SCORE_COLUMNS = ("spearman", "pearson", "mse", "top_k_precision", "top_k_recall", "ap_high")
VALIDATION_BUDGETS = (5, 10)
RETAINED_FRACTIONS = (0.25, 0.50, 0.75, 1.0)


def risk_at_coverage(spread, absolute_error, retained_fraction: float) -> dict:
    """Mean absolute error among the smallest spreads, with random tie breaking."""
    spread, absolute_error = np.asanyarray(spread, dtype=float), np.asanyarray(absolute_error, dtype=float)
    if (spread.ndim != 1 or absolute_error.shape != spread.shape or not spread.size
            or not np.isfinite(spread).all() or not np.isfinite(absolute_error).all()
            or np.any(spread < 0) or np.any(absolute_error < 0) or not 0 < retained_fraction <= 1):
        raise ValueError("Expected matching, nonnegative spread and error vectors with valid retention")
    retained = max(1, round(spread.size * retained_fraction))
    cutoff = np.partition(spread, retained - 1)[retained - 1]
    lower = spread < cutoff
    ties = spread == cutoff
    risk = (absolute_error[lower].sum() + (retained - lower.sum()) * absolute_error[ties].mean()) / retained
    return {"retained_fraction": retained / spread.size, "retained": retained, "mae": float(risk)}


def spread_diagnostics(spread, measured, predicted) -> dict:
    """Describe spread versus realized error without constructing a confidence interval."""
    spread, measured, predicted = (np.asarray(values, dtype=float) for values in (spread, measured, predicted))
    if measured.shape != predicted.shape:
        raise ValueError("Expected matching measured and predicted vectors")
    absolute_error = np.abs(measured - predicted)
    coverages = [risk_at_coverage(spread, absolute_error, retained) for retained in RETAINED_FRACTIONS]
    correlation = None
    if np.ptp(spread) > 0 and np.ptp(absolute_error) > 0:
        correlation = float(spearmanr(spread, absolute_error).statistic)
    return {
        "spread_abs_error_spearman": correlation,
        "median_spread": float(np.median(spread)),
        "mae": float(absolute_error.mean()),
        "fraction_abs_error_within_one_spread": float(np.mean(absolute_error <= spread)),
        "fraction_abs_error_within_1_96_spreads": float(np.mean(absolute_error <= 1.96 * spread)),
        "risk_at_coverage": coverages,
    }


def validate_models(dataset, splits: dict[int, dict[str, list[str]]]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fit fixed model settings on initial observations and score only validation labels."""
    if not splits:
        raise ValueError("At least one archived validation split is required")
    records, predictions, diagnostics = [], [], []
    fold_ids = []
    wild_type = dataset.scores[WT_ID]
    for seed, split in sorted(splits.items()):
        assigned = [variant for role in split.values() for variant in role]
        if len(set(assigned)) != len(assigned) or WT_ID not in split["initial"]:
            raise ValueError(f"Invalid or overlapping GFP split at seed {seed}")
        initial, validation = split["initial"], split["validation"]
        fold_ids.extend(validation)
        targets = [dataset.scores[variant] for variant in initial]
        actual = [dataset.scores[variant] for variant in validation]
        training = [dataset.variants[variant] for variant in initial]
        held_out = [dataset.variants[variant] for variant in validation]
        for name in ("ridge", "rf"):
            model = fit_predictor(training, targets, seed, model=name)
            predicted = model.predict(held_out)
            for criterion, threshold in (("above_wild_type", wild_type), ("upper_decile", None)):
                for budget in VALIDATION_BUDGETS:
                    records.append({"seed": seed, "model": name, "criterion": criterion,
                                    **metrics(actual, predicted, top_k=budget, threshold=threshold)})
            spread = model.uncertainty(held_out) if name == "rf" else [None] * len(validation)
            if name == "rf":
                diagnostics.append({"seed": seed, **spread_diagnostics(spread, actual, predicted)})
            for variant, measured, score, width in zip(validation, actual, predicted, spread, strict=True):
                predictions.append({"seed": seed, "model": name, "variant_id": variant, "measured": measured,
                                    "predicted": score, "absolute_error": abs(measured - score), "tree_spread": width})
    return pd.DataFrame(records), pd.DataFrame(predictions), {
        "validation_rows": len(fold_ids),
        "unique_validation_variants": len(set(fold_ids)),
        "wild_type_measurement": wild_type,
        "rf_spread": diagnostics,
        "ridge_spread": "unavailable; the Ridge API's zeros are placeholders, not predictive uncertainty",
    }


def run(archive: Path, output: Path, cache: Path | None) -> None:
    splits = {int(path.stem.removeprefix("seed-")): json.loads(path.read_text(encoding="utf-8"))
              for path in (archive / "splits").glob("seed-*.json")}
    dataset = load_dataset(cache)
    comparison, predictions, summary = validate_models(dataset, splits)
    archived = pd.read_csv(archive / "metrics.csv")
    key = ["seed", "model", *SCORE_COLUMNS]
    original = archived.loc[archived["split"] == "validation", key].sort_values(["seed", "model"])
    reproduced = comparison.loc[
        (comparison["criterion"] == "above_wild_type") & (comparison["top_k"] == 5), key
    ].sort_values(["seed", "model"])
    pd.testing.assert_frame_equal(original.reset_index(drop=True), reproduced.reset_index(drop=True),
                                  check_exact=False, rtol=1e-10, atol=1e-10)
    output.mkdir(parents=True, exist_ok=False)
    comparison.to_csv(output / "validation_by_seed.csv", index=False, lineterminator="\n")
    grouped = comparison.groupby(["criterion", "top_k", "model"])[list(SCORE_COLUMNS)].mean().reset_index()
    grouped.to_csv(output / "validation_means.csv", index=False, lineterminator="\n")
    predictions.to_csv(output / "validation_predictions.csv", index=False, lineterminator="\n")
    summary.update({"status": "complete", "seeds": sorted(splits), "archived_validation_reproduced": True,
                    "test_labels_used": False, "acquisition_labels_used": False, "api_calls": 0,
                    "assay_sha256": dataset.provenance["assay_sha256"]})
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-cache", type=Path)
    args = parser.parse_args()
    run(args.archive, args.output, args.data_cache)


if __name__ == "__main__":
    main()
