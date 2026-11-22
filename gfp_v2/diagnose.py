"""Inspect only archived V1 genotypes, predictions and selector traces."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from gfp_data import FLUORESCENT_THRESHOLD, WT_ID, load_dataset
from gfp_v2.benchmark import ROOT, _corr, _write_json
from gfp_v2.partition import historical_ids
from scipy.stats import spearmanr


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _support_records(dataset, archive: Path) -> tuple[pd.DataFrame, dict]:
    validation = pd.read_csv(archive / "quality/model_validation/validation_predictions.csv")
    grouped = []
    collision_counts = {}
    for seed, records in validation.groupby("seed"):
        split = _read(archive / f"formal/deepseek/splits/seed-{seed}.json")
        counts = Counter(edit for key in split["initial"] for edit in dataset.variants[key].edits)
        site_counts = Counter(edit[1:-1] for edit in counts)
        coobserved = {pair for key in split["initial"] for pair in combinations(dataset.variants[key].edits, 2)}
        signature_counts = Counter()
        for row in records.to_dict("records"):
            variant = dataset.variants[row["variant_id"]]
            missing = [edit for edit in variant.edits if counts[edit] == 0]
            row["substitutions"] = len(variant.edits)
            row["missing_alleles"] = len(missing)
            row["missing_sites"] = sum(site_counts[edit[1:-1]] == 0 for edit in variant.edits)
            row["all_edits_unseen"] = len(missing) == len(variant.edits)
            row["any_allele_pair_seen"] = any(pair in coobserved for pair in combinations(variant.edits, 2))
            row["allele_pair_count"] = len(tuple(combinations(variant.edits, 2)))
            row["phenotype"] = ("above_wt" if row["measured"] > dataset.scores[WT_ID]
                                else "fluorescent_below_wt" if row["measured"] > FLUORESCENT_THRESHOLD else "dim")
            if row["model"] == "ridge":
                signature = (len(variant.edits), tuple(edit[1:-1] for edit in variant.edits if edit[1:-1] in site_counts),
                             tuple(edit for edit in variant.edits if counts[edit]))
                signature_counts[signature] += 1
            grouped.append(row)
        collision_counts[int(seed)] = {
            "rows": sum(signature_counts.values()), "distinct_v1_feature_vectors": len(signature_counts),
            "rows_sharing_a_v1_feature_vector": sum(size for size in signature_counts.values() if size > 1),
        }
    return pd.DataFrame(grouped), collision_counts


def _error_groups(validation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    outcomes = []
    for family in ("substitutions", "missing_alleles", "all_edits_unseen", "phenotype", "any_allele_pair_seen"):
        view = validation[validation["substitutions"] > 1] if family == "any_allele_pair_seen" else validation
        for (model, level), records in view.groupby(["model", family]):
            bias = records["predicted"].to_numpy() - records["measured"].to_numpy()
            outcomes.append({"grouping": family, "level": str(level), "model": model, "rows": len(records),
                             "mae": float(np.abs(bias).mean()), "mse": float((bias ** 2).mean()),
                             "pred_minus_measured": float(bias.mean()), "large_error_count": int(np.sum(np.abs(bias) > 1)),
                             "spearman": _corr(spearmanr, records["predicted"].to_numpy(), records["measured"].to_numpy())})
    ranks = []
    for (seed, model), records in validation.groupby(["seed", "model"]):
        edit_count = records["substitutions"].to_numpy()
        measured = records["measured"].to_numpy()
        predicted = records["predicted"].to_numpy()
        for name, mask in (("all", np.ones(len(records), dtype=bool)), ("fluorescent", measured > FLUORESCENT_THRESHOLD),
                           *((f"{count}_edits", edit_count == count) for count in range(1, 5))):
            ranks.append({"seed": seed, "model": model, "subset": name, "rows": int(mask.sum()),
                          "spearman": _corr(spearmanr, measured[mask], predicted[mask]),
                          "score_vs_edit_count": _corr(spearmanr, predicted[mask], edit_count[mask]),
                          "prediction_sd": float(predicted[mask].std()) if mask.any() else None})
    return pd.DataFrame(outcomes), pd.DataFrame(ranks)


def _misses(dataset, archive: Path) -> list[dict]:
    rows = []
    for path in sorted((archive / "formal/deepseek/splits").glob("seed-*.json")):
        seed = int(path.stem.split("-")[-1])
        data = _read(path)
        incumbent = max(dataset.scores[key] for key in data["initial"])
        counts = Counter(edit for key in data["initial"] for edit in dataset.variants[key].edits)
        pairs = {pair for key in data["initial"] for pair in combinations(dataset.variants[key].edits, 2)}
        qualifying = [key for key in data["pool"] if dataset.scores[key] > incumbent]
        for key in qualifying:
            ranks, menu_exposures, selected = [], [], []
            for backend in sorted(path.name for path in (archive / "formal").iterdir() if path.is_dir()):
                for trace in sorted((archive / "formal" / backend / "arms" / f"seed-{seed}").glob("*/round-*-selection.json")):
                    record = _read(trace)
                    if key in record["eligible_rf_ranks"]:
                        ranks.append(record["eligible_rf_ranks"][key])
                    request = trace.with_name(trace.name.replace("-selection.json", "-request.json"))
                    if request.exists() and key in {candidate["id"] for candidate in _read(request)["menu"]}:
                        menu_exposures.append(f"{backend}/{trace.parent.name}/{trace.name[:8]}")
                    if key in record["selected_ids"]:
                        selected.append(f"{backend}/{trace.parent.name}/{trace.name[:8]}")
            edits = dataset.variants[key].edits
            rows.append({"seed": seed, "variant_id": key, "measured": dataset.scores[key], "initial_best": incumbent,
                         "unseen_initial_alleles": sum(counts[edit] == 0 for edit in edits), "mutations": len(edits),
                         "initially_observed_allele_pairs": sum(pair in pairs for pair in combinations(edits, 2)),
                         "best_archived_rf_rank": min(ranks), "worst_archived_rf_rank": max(ranks),
                         "rf_top5_appearances": sum(rank <= 5 for rank in ranks), "evaluated_in_rounds": len(ranks),
                         "llm_menu_exposures": len(menu_exposures), "llm_menu_paths": ";".join(menu_exposures),
                         "actual_query_paths": ";".join(selected)})
    return rows


def _retrospective_nonadditivity(dataset) -> tuple[list[dict], dict]:
    retired = historical_ids() & dataset.variants.keys()
    singles = {key: dataset.scores[key] for key in retired if len(dataset.variants[key].edits) == 1}
    wt = dataset.scores[WT_ID]
    observations = []
    for key in sorted(retired):
        edits = dataset.variants[key].edits
        if len(edits) < 2 or not set(edits) <= singles.keys():
            continue
        additive = wt + sum(singles[edit] - wt for edit in edits)
        actual = dataset.scores[key]
        observations.append({"variant_id": key, "edits": len(edits), "additive_log_fluorescence": additive,
                             "measured_log_fluorescence": actual, "measured_minus_additive": actual - additive,
                             "dimmer_than_fluorescence_threshold_despite_additive_bright": (
                                 additive > FLUORESCENT_THRESHOLD >= actual)})
    residuals = [item["measured_minus_additive"] for item in observations]
    stats = {"retired_measured_singles": len(singles), "retired_multi_genotypes_matchable_to_all_singles": len(residuals),
             "median_measured_minus_independent_additive": float(np.median(residuals)) if residuals else None,
             "mae_independent_additive": float(np.mean(np.abs(residuals))) if residuals else None,
             "fraction_below_independent_additive": float(np.mean(np.array(residuals) < 0)) if residuals else None,
             "additive_bright_observed_dim_n": sum(item["dimmer_than_fluorescence_threshold_despite_additive_bright"]
                                                   for item in observations)}
    return observations, stats


def run(dataset, output: Path) -> None:
    archive = ROOT / "evidence/gfp"
    validation, collision_counts = _support_records(dataset, archive)
    errors, ranking = _error_groups(validation)
    misses = _misses(dataset, archive)
    deviations, interaction = _retrospective_nonadditivity(dataset)
    output.mkdir(parents=True, exist_ok=False)
    validation.to_csv(output / "v1_validation_annotated.csv", index=False, lineterminator="\n")
    errors.to_csv(output / "v1_error_groups.csv", index=False, lineterminator="\n")
    ranking.to_csv(output / "v1_conditional_ranking.csv", index=False, lineterminator="\n")
    pd.DataFrame(misses).to_csv(output / "v1_incumbent_misses.csv", index=False, lineterminator="\n")
    pd.DataFrame(deviations).to_csv(output / "v1_additive_deviations.csv", index=False, lineterminator="\n")
    multimutant = validation[(validation.model == "ridge") & (validation.substitutions > 1)]
    unseen = validation[(validation.model == "ridge") & (validation.missing_alleles > 0)]
    overall = ranking[ranking.subset == "all"].groupby("model").spearman.mean().to_dict()
    bright = ranking[ranking.subset == "fluorescent"].groupby("model").spearman.mean().to_dict()
    diagnostics = {"v1_only": True, "score_units": "log10 fluorescence", "v1_validation_rows": int(len(validation) // 2),
                   "fraction_with_unseen_training_allele": len(unseen) / (len(validation) // 2),
                   "fraction_multimutants_with_any_initial_coobserved_pair": float(multimutant.any_allele_pair_seen.mean()),
                   "feature_collisions_by_seed": collision_counts, "mean_heldout_spearman": overall,
                   "mean_heldout_spearman_among_fluorescent": bright, "original_qualifying_pool_rows": len(misses),
                   "missed_best_rf_rank_range": [min(row["best_archived_rf_rank"] for row in misses),
                                                 max(row["best_archived_rf_rank"] for row in misses)],
                   "qualifying_with_unknown_training_allele": sum(row["unseen_initial_alleles"] > 0 for row in misses),
                   "qualifying_above_incumbent_rf_top5_appearances": sum(row["rf_top5_appearances"] for row in misses),
                   "nonadditivity": interaction}
    _write_json(output / "v1_summary.json", diagnostics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "evidence/gfp-v2/diagnosis")
    parser.add_argument("--data-cache", type=Path)
    args = parser.parse_args()
    run(load_dataset(args.data_cache), args.output)


if __name__ == "__main__":
    main()
