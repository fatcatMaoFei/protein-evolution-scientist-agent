"""Run one fixed, aggregate-only evaluation on the published FLIP GB1 split."""

import argparse
from collections import Counter
import csv
from hashlib import sha256
from io import BytesIO
from itertools import combinations
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, mean_squared_error
from threadpoolctl import threadpool_limits


REVISION = "62cace8735f5610e2743cf06ce0f944b37fffaa6"
URL = (
    f"https://raw.githubusercontent.com/J-SNACKKB/FLIP/{REVISION}/splits/gb1/splits.zip"
)
ARCHIVE_SHA256 = "00a3c3d5be56ce35adfbec838db8ff549be369525e151a6978133da34b9fd86f"
MEMBER = "splits/three_vs_rest.csv"
MEMBER_SHA256 = "b8aa60cc061013a5c2e6f4070593c82976b1d9e5c9058a66ad57c6e9c3cec023"
POSITIONS = (39, 40, 41, 54)
CONSTRUCT_LENGTH = 265
ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
WT_FITNESS = 1.0
TOP_FRACTION = 0.05
SEED = 1701
RIDGE_GRID = (0.1, 1.0, 10.0, 100.0)
FOREST_GRID = (1, 3, 8)
N_ESTIMATORS = 160
MAX_FEATURES = 0.7


def require(condition, description):
    if not condition:
        raise ValueError(description)


def verified_member(cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "splits.zip"
    was_cached = cache_file.exists()
    retrieved_at = None
    if was_cached:
        archive = cache_file.read_bytes()
    else:
        with urlopen(URL, timeout=45) as response:
            archive = response.read()
            retrieved_at = response.headers.get("Date")
    require(sha256(archive).hexdigest() == ARCHIVE_SHA256, "FLIP ZIP checksum mismatch")
    if not was_cached:
        cache_file.write_bytes(archive)
    with ZipFile(BytesIO(archive)) as source:
        member = source.read(MEMBER)
    require(sha256(member).hexdigest() == MEMBER_SHA256, "FLIP CSV checksum mismatch")
    provenance = {
        "source_url": URL,
        "source_revision": REVISION,
        "archive_file": "splits.zip",
        "archive_sha256": ARCHIVE_SHA256,
        "archive_bytes": len(archive),
        "csv_member": MEMBER,
        "csv_sha256": MEMBER_SHA256,
        "csv_bytes": len(member),
        "retrieval": "verified cached archive" if was_cached else "downloaded and verified",
        "retrieval_server_date": retrieved_at,
        "license_flip": "AFL-3.0",
        "license_original": "CC-BY-4.0",
    }
    return member, provenance


def audit_table(member):
    frame = pd.read_csv(BytesIO(member), dtype=str, keep_default_na=False)
    require(
        list(frame.columns) == ["sequence", "target", "set", "validation"],
        "Unexpected FLIP fields",
    )
    sequences = frame["sequence"]
    missing_sequence = int(sequences.eq("").sum())
    malformed_sequence = int(
        (~sequences.str.fullmatch(f"[{ALPHABET}]{{{CONSTRUCT_LENGTH}}}")).sum()
    )
    duplicate_sequences = int(sequences.duplicated().sum())
    target = pd.to_numeric(frame["target"], errors="coerce").to_numpy(dtype=float)
    missing_or_invalid_target = int((~np.isfinite(target)).sum())
    counts = (missing_sequence, malformed_sequence, duplicate_sequences, missing_or_invalid_target)
    require(not any(counts), f"FLIP sequence or target audit failed: {counts}")
    require(np.all(target >= 0), "Negative assay fitness")
    require(set(frame["set"]) == {"train", "test"}, "Unexpected split labels")
    require(set(frame["validation"]) == {"", "True"}, "Unexpected validation labels")

    is_source_train = frame["set"].eq("train").to_numpy()
    validation = frame["validation"].eq("True").to_numpy()
    test = ~is_source_train
    train = is_source_train & ~validation
    require(not (test & validation).any(), "Validation row also marked as test")
    partitions = {"train": train, "validation": validation, "test": test}
    require(all(mask.any() for mask in partitions.values()), "Empty split partition")

    residues = np.asarray([list(sequence) for sequence in sequences], dtype="U1")
    changed_columns = np.flatnonzero(np.any(residues != residues[0], axis=0)) + 1
    require(tuple(changed_columns) == POSITIONS, "Unexpected variable construct coordinates")
    encoded = np.column_stack(
        [np.searchsorted(list(ALPHABET), residues[:, position - 1]) for position in POSITIONS]
    )
    # The assay's WT normalization anchors the sole score-one training row.
    # The edit-depth boundary checks this inferred reference independently.
    wt_rows = train & (target == WT_FITNESS)
    require(int(wt_rows.sum()) == 1, "Expected one WT-score reference in training")
    wt = encoded[wt_rows][0]
    depth = (encoded != wt).sum(axis=1)
    require(
        (depth[test] == len(POSITIONS)).all() and (depth[~test] < len(POSITIONS)).all(),
        "Observed split disagrees with the source mutation-depth definition",
    )

    audit = {
        "fields": list(frame.columns),
        "rows": len(target),
        "sequence_length": CONSTRUCT_LENGTH,
        "variable_construct_coordinates_1_based": list(POSITIONS),
        "observed_residue_alphabet_size": int(np.unique(residues).size),
        "duplicate_sequences": duplicate_sequences,
        "missing_sequences": missing_sequence,
        "malformed_sequences": malformed_sequence,
        "missing_or_nonfinite_targets": missing_or_invalid_target,
        "negative_targets": int((target < 0).sum()),
        "zero_targets": int((target == 0).sum()),
        "wt_reference_fitness": WT_FITNESS,
        "target_direction": "higher is greater assay enrichment relative to WT",
        "partition_intersection_count": 0,
        "partitions": {
            key: {
                "rows": int(mask.sum()),
                "above_wt": int((target[mask] > WT_FITNESS).sum()),
                "edits": dict(sorted(Counter(depth[mask].tolist()).items())),
                "source_row_index_sha256": sha256(
                    np.flatnonzero(mask).astype("<i8").tobytes()
                ).hexdigest(),
            }
            for key, mask in partitions.items()
        },
    }
    return encoded, target, depth, wt, partitions, audit


def encode(residues, pairwise=False):
    alphabet_size = len(ALPHABET)
    width = residues.shape[1]
    columns = [residues + np.arange(width) * alphabet_size]
    n_features = width * alphabet_size
    if pairwise:
        for left, right in combinations(range(width), 2):
            columns.append(
                (n_features + alphabet_size * residues[:, left] + residues[:, right])[:, None]
            )
            n_features += alphabet_size**2
    indices = np.column_stack(columns).ravel()
    entries_per_row = len(indices) // len(residues)
    return csr_matrix(
        (np.ones(len(indices)), indices, np.arange(len(residues) + 1) * entries_per_row),
        shape=(len(residues), n_features),
    )


def precision_at_k(binary, score, k):
    cutoff = np.partition(score, len(score) - k)[len(score) - k]
    above = score > cutoff
    tied = score == cutoff
    expected = float(binary[above].sum())
    expected += (k - int(above.sum())) * float(binary[tied].mean())
    return expected / k, expected


def metrics(target, score):
    positive = target > WT_FITNESS
    k = int(np.ceil(TOP_FRACTION * len(target)))
    precision, hits = precision_at_k(positive, score, k)
    correlations_undefined = np.ptp(score) == 0 or np.ptp(target) == 0
    return {
        "n": len(target),
        "spearman": None if correlations_undefined else float(spearmanr(target, score).statistic),
        "pearson": None if correlations_undefined else float(pearsonr(target, score).statistic),
        "mse": float(mean_squared_error(target, score)),
        "target_above_wt_prevalence": float(positive.mean()),
        "ap_above_wt": float(average_precision_score(positive, score)),
        "top_k": k,
        "top_k_precision_above_wt": precision,
        "top_k_recall_above_wt": hits / int(positive.sum()) if positive.any() else None,
    }


def forest(min_samples_leaf):
    return RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_features=MAX_FEATURES,
        min_samples_leaf=min_samples_leaf,
        random_state=SEED,
        n_jobs=1,
    )


def fit_and_select(train, target_train, validation, target_val, make_model, grid, model_name):
    trials = []
    for index, setting in enumerate(grid):
        model = make_model(setting).fit(train, target_train)
        scores = metrics(target_val, model.predict(validation))
        row = {"model": model_name, "partition": "validation", "setting": setting, **scores}
        trials.append((model, row, index))
    selected, best, _ = min(
        trials,
        key=lambda entry: (
            -entry[1]["spearman"] if entry[1]["spearman"] is not None else np.inf,
            entry[1]["mse"],
            entry[2],
        ),
    )
    return selected, best, [row for _, row, _ in trials]


def errors_by_score_band(model_name, target, score):
    rows = []
    for band, take in {
        "zero": target == 0,
        "positive_at_or_below_wt": (target > 0) & (target <= WT_FITNESS),
        "above_wt": target > WT_FITNESS,
    }.items():
        errors = score[take] - target[take]
        rows.append(
            {
                "model": model_name,
                "partition": "test",
                "measured_score_band": band,
                "n": len(errors),
                "mean_error_predicted_minus_measured": float(errors.mean()),
                "mae": float(np.abs(errors).mean()),
                "abs_error_p90": float(np.quantile(np.abs(errors), 0.9)),
            }
        )
    return rows


def single_effect_transfer(residues, target, depth, wt, train, evaluated, min_group_size=1):
    one_effect = {}
    for codes, score in zip(residues[train & (depth == 1)], target[train & (depth == 1)]):
        position = int(np.flatnonzero(codes != wt)[0])
        one_effect[position, int(codes[position])] = score / WT_FITNESS
    observed, expected = [], []
    for codes, score in zip(residues[evaluated], target[evaluated]):
        components = [
            one_effect.get((int(position), int(codes[position])))
            for position in np.flatnonzero(codes != wt)
        ]
        if all(component is not None for component in components):
            observed.append(score)
            expected.append(WT_FITNESS * np.prod(components))
    observed = np.asarray(observed)
    expected = np.asarray(expected)
    truth_above = observed > WT_FITNESS
    expected_above = expected > WT_FITNESS
    n = len(observed)
    evaluation_count = int(evaluated.sum())
    summarize = n >= min_group_size
    return {
        "evaluated_rows": evaluation_count,
        "covered_by_training_singles": n,
        "coverage": n / evaluation_count if evaluation_count else None,
        "training_singles": len(one_effect),
        "multiplicative_assumption": "WT times product of constituent training single scores / WT",
        "mean_abs_observed_minus_multiplicative": (
            float(np.abs(observed - expected).mean()) if summarize else None
        ),
        "observed_above_wt_expected_not_above_wt": (
            int((truth_above & ~expected_above).sum()) if summarize else None
        ),
        "observed_not_above_wt_expected_above_wt": (
            int((~truth_above & expected_above).sum()) if summarize else None
        ),
        "both_above_wt": int((truth_above & expected_above).sum()) if summarize else None,
        "both_not_above_wt": int((~truth_above & ~expected_above).sum()) if summarize else None,
    }


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def write_rows(path, rows):
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("evidence/gb1"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/gb1"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    member, provenance = verified_member(args.cache)
    residues, target, depth, wt, partitions, audit = audit_table(member)
    train, validation, test = (partitions[key] for key in ("train", "validation", "test"))

    write_json(args.output / "provenance.json", provenance)
    write_json(args.output / "audit.json", audit)
    write_json(
        args.output / "training_config.json",
        {
            "seed": SEED,
            "threads": 1,
            "ridge_alpha": list(RIDGE_GRID),
            "ridge_solver": "lsqr",
            "random_forest_min_samples_leaf": list(FOREST_GRID),
            "random_forest_n_estimators": N_ESTIMATORS,
            "random_forest_max_features": MAX_FEATURES,
            "model_selection": "validation Spearman descending, MSE ascending, grid order",
            "fit_partition": "train",
            "selection_partition": "validation",
            "evaluation_partition": "test",
            "top_fraction": TOP_FRACTION,
            "high_fitness": "assay score > 1 (wild type)",
        },
    )
    families = [
        ("ridge_additive", lambda alpha: Ridge(alpha=alpha, solver="lsqr"), RIDGE_GRID, False),
        ("ridge_pairwise", lambda alpha: Ridge(alpha=alpha, solver="lsqr"), RIDGE_GRID, True),
        ("random_forest", forest, FOREST_GRID, False),
    ]
    results, grids, error_rows, ablations = [], [], [], []
    with threadpool_limits(limits=1):
        selected_models = []
        for name, factory, grid, pairwise in families:
            fit_x = encode(residues[train], pairwise)
            val_x = encode(residues[validation], pairwise)
            model, best_val, all_val = fit_and_select(
                fit_x, target[train], val_x, target[validation], factory, grid, name
            )
            results.append(best_val)
            grids.extend(all_val)
            selected_models.append((name, model, best_val, pairwise))

        baseline = float(target[train].mean())
        for split in ("validation", "test"):
            actual = target[partitions[split]]
            constant = np.full(len(actual), baseline)
            results.append(
                {
                    "model": "train_mean",
                    "partition": split,
                    "setting": None,
                    **metrics(actual, constant),
                }
            )
            if split == "test":
                error_rows.extend(errors_by_score_band("train_mean", actual, constant))

        for name, model, best_val, pairwise in selected_models:
            prediction = model.predict(encode(residues[test], pairwise))
            test_scores = metrics(target[test], prediction)
            results.append(
                {"model": name, "partition": "test", "setting": best_val["setting"], **test_scores}
            )
            error_rows.extend(errors_by_score_band(name, target[test], prediction))
            if name == "random_forest":
                for omitted_index, position in enumerate(POSITIONS):
                    keep = [index for index in range(len(POSITIONS)) if index != omitted_index]
                    reduced_train = encode(residues[train][:, keep])
                    reduced_val = encode(residues[validation][:, keep])
                    reduced = forest(best_val["setting"]).fit(reduced_train, target[train])
                    reduced_metrics = metrics(target[validation], reduced.predict(reduced_val))
                    ablations.append(
                        {
                            "omitted_position": position,
                            "spearman_change_from_full": (
                                reduced_metrics["spearman"] - best_val["spearman"]
                            ),
                            "mse_change_from_full": reduced_metrics["mse"] - best_val["mse"],
                            **reduced_metrics,
                        }
                    )

    write_rows(args.output / "metrics.csv", results)
    write_rows(args.output / "validation_grid.csv", grids)
    write_rows(args.output / "error_strata.csv", error_rows)
    write_rows(args.output / "position_ablation.csv", ablations)
    write_json(
        args.output / "single_effect_transfer.json",
        {
            "partition": "test",
            "mutation_depth": len(POSITIONS),
            **single_effect_transfer(residues, target, depth, wt, train, test),
        },
    )
    write_rows(
        args.output / "exploratory_single_effect_transfer.csv",
        [
            {
                "partition": split,
                "mutation_depth": edits,
                **single_effect_transfer(
                    residues, target, depth, wt, train, partitions[split] & (depth == edits), 10
                ),
            }
            for split in ("train", "validation")
            for edits in (2, 3)
        ],
    )
    print(f"FLIP GB1 aggregate evaluation completed in {args.output}")


if __name__ == "__main__":
    main()
