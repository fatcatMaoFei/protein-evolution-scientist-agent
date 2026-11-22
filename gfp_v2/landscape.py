"""Supplementary GFP archive analyses; no API calls or frozen-policy refits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gfp_data import FLUORESCENT_THRESHOLD, GFP_WT, WT_ID, load_dataset
from gfp_v2.benchmark import ROOT, _digest, accuracy
from gfp_v2.structure import DEFAULT_PDB, read_mapping, variant_descriptors


EVIDENCE = ROOT / "evidence/gfp-v2"
DEFAULT_OUTPUT = EVIDENCE / "supplement/landscape"
DEGREES = range(1, 5)


def archive_frame(dataset) -> pd.DataFrame:
    return pd.DataFrame({"id": ids, "mutation_count": len(variant.edits), "fitness": dataset.scores[ids],
                         "above_wt": dataset.scores[ids] > dataset.scores[WT_ID],
                         "fluorescent": dataset.scores[ids] > FLUORESCENT_THRESHOLD}
                        for ids, variant in sorted(dataset.variants.items()) if ids != WT_ID)


def distributions(archive: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for degree, frame in archive.groupby("mutation_count", sort=True):
        values = frame.fitness
        rows.append({"mutation_count": degree, "variants": len(frame), "mean": values.mean(),
                     "median": values.median(), "p10": values.quantile(0.10), "p90": values.quantile(0.90),
                     "above_wt_n": int(frame.above_wt.sum()), "above_wt_rate": frame.above_wt.mean(),
                     "fluorescent_rate": frame.fluorescent.mean()})
    return pd.DataFrame(rows)


def nonadditivity(dataset) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match each multi-edit genotype to its component single-genotype measurements."""
    wt = dataset.scores[WT_ID]
    observations, denominators = [], {degree: 0 for degree in range(2, 5)}
    for key, variant in sorted(dataset.variants.items()):
        edits = variant.edits
        degree = len(edits)
        if degree < 2:
            continue
        denominators[degree] += 1
        if not all(edit in dataset.scores and len(dataset.variants[edit].edits) == 1 for edit in edits):
            continue
        additive = float(wt + sum(dataset.scores[edit] - wt for edit in edits))
        measured = dataset.scores[key]
        observations.append({"id": key, "mutation_count": degree, "single_ids": ";".join(edits),
                             "component_single_scores_json": json.dumps([dataset.scores[edit] for edit in edits]),
                             "additive_log_brightness": additive, "measured_log_brightness": measured,
                             "measured_minus_additive": measured - additive,
                             "additive_above_wt": additive > wt, "measured_above_wt": measured > wt,
                             "additive_bright_measured_dim": additive > FLUORESCENT_THRESHOLD >= measured})
    frame = pd.DataFrame(observations)
    summaries = []
    for degree, group in frame.groupby("mutation_count", sort=True):
        delta = group.measured_minus_additive
        summaries.append({"mutation_count": degree, "total_genotypes": denominators[degree],
                          "fully_matched_genotypes": len(group), "matched_fraction": len(group) / denominators[degree],
                          "median_measured_minus_additive": delta.median(), "additive_mae": delta.abs().mean(),
                          "fraction_below_additive": (delta < 0).mean(),
                          "additive_above_wt_n": int(group.additive_above_wt.sum()),
                          "measured_above_wt_n": int(group.measured_above_wt.sum()),
                          "additive_bright_measured_dim_n": int(group.additive_bright_measured_dim.sum())})
    return frame, pd.DataFrame(summaries)


def prediction_strata(predictions: pd.DataFrame, wt: float) -> pd.DataFrame:
    rows = []
    for (budget, model, degree), frame in predictions.groupby(["budget", "model", "mutation_count"], sort=True):
        errors = frame.predicted - frame.measured
        by_block = [accuracy(group.variant_id.tolist(), group.measured.to_numpy(), group.predicted.to_numpy(), wt)
                    for _, group in frame.groupby("block", sort=True)]
        size = len(frame)
        rank = spearmanr(frame.measured, frame.predicted).statistic if size >= 10 and (
            frame.measured.nunique() > 1 and frame.predicted.nunique() > 1
        ) else np.nan
        rows.append({"budget": budget, "model": model, "mutation_count": degree, "test_rows": size,
                     "test_blocks": frame.block.nunique(), "mae": errors.abs().mean(), "mse": errors.pow(2).mean(),
                     "bias_predicted_minus_measured": errors.mean(), "spearman_pooled": rank,
                     "above_wt_test_rate": (frame.measured > wt).mean(),
                     "mean_within_degree_top5_above_wt_precision": float(np.mean([item["wt_precision_5"] for item in by_block])),
                     "mean_within_degree_top5_best": float(np.mean([item["top_5_measured_best"] for item in by_block]))})
    return pd.DataFrame(rows)


def position_matrix(keys: list[str], dataset) -> np.ndarray:
    matrix = np.zeros((len(keys), len(GFP_WT)), dtype=bool)
    for i, key in enumerate(keys):
        positions = [int(edit[1:-1]) - 1 for edit in dataset.variants[key].edits]
        matrix[i, positions] = True
    return matrix


def ratio(observed, expected):
    observed, expected = np.asarray(observed), np.asarray(expected)
    return np.divide(observed, expected, out=np.full(expected.shape, np.nan, dtype=float), where=expected > 0)


def archived_positions(dataset, archive: pd.DataFrame, predictions: pd.DataFrame, winner: str) -> pd.DataFrame:
    """Contrast above-WT site incidence to expected incidence conditional on edit count."""
    pool_counts = np.zeros(len(GFP_WT), dtype=int)
    hit_counts = pool_counts.copy()
    expected_hits = np.zeros(len(GFP_WT))
    for _, frame in archive.groupby("mutation_count", sort=True):
        presence = position_matrix(frame.id.tolist(), dataset)
        hits = frame.above_wt.to_numpy()
        pool_counts += presence.sum(0)
        hit_counts += presence[hits].sum(0)
        expected_hits += hits.mean() * presence.sum(0)
    low = predictions[(predictions.budget == "low") & (predictions.model == winner)]
    present = position_matrix(low.variant_id.tolist(), dataset)
    residual = low.predicted - low.measured
    count = present.sum(0)
    return pd.DataFrame({"position": np.arange(1, len(GFP_WT) + 1), "assay_residue": list(GFP_WT),
                         "archive_genotypes": pool_counts, "archive_above_wt_genotypes": hit_counts,
                         "archive_above_wt_fraction": ratio(hit_counts, pool_counts),
                         "mutation_count_adjusted_expected_archive_above_wt": expected_hits,
                         "archive_adjusted_above_wt_enrichment": ratio(hit_counts, expected_hits),
                         "low_budget_heldout_n": count,
                         "low_budget_heldout_mae": ratio(present.T @ residual.abs().to_numpy(), count),
                         "low_budget_heldout_bias_predicted_minus_measured": ratio(present.T @ residual.to_numpy(), count)})


def _arm_counts() -> tuple[np.ndarray, np.ndarray]:
    # Rows aggregate by selection category, columns capture uniform and degree-matched controls.
    return np.zeros((4, 6), dtype=float), np.zeros((len(GFP_WT), 6), dtype=float)


def selection_preference(dataset, trial_root: Path, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use each original public feature-only roster and its selected IDs; score prevalence offline."""
    selections = pd.read_csv(trial_root / "selections.csv")
    arm_totals = {}
    wt = dataset.scores[WT_ID]
    for (block, arm, round_number), group in selections.groupby(["block", "arm", "round"], sort=True):
        record_path = trial_root / "arms" / str(block) / arm / f"round-{round_number:02d}-request.json"
        request = json.loads(record_path.read_text())
        ids = [candidate["id"] for candidate in request["candidates"]]
        if len(ids) != len(set(ids)) or not set(group.id) <= set(ids) or len(group) != request["budget"]:
            raise ValueError(f"Measured decisions do not match their frozen rosters: {arm} block {block} round {round_number}")
        if not all(np.isclose(dataset.scores[row.id], row.measured_fitness) for row in group.itertuples()):
            raise ValueError(f"Mismatch against pinned archived truth: {record_path}")
        counts, sites = arm_totals.setdefault(arm, _arm_counts())
        pool_presence = position_matrix(ids, dataset)
        picked_presence = position_matrix(group.id.tolist(), dataset)
        pool_degree = pool_presence.sum(1)
        picked_degree = picked_presence.sum(1)
        pool_high = np.array([dataset.scores[key] > wt for key in ids])
        picked_high = group.measured_fitness.to_numpy() > wt
        sites[:, 0] += picked_presence.sum(0)
        sites[:, 1] += len(group) * pool_presence.mean(0)
        sites[:, 3] += picked_presence[picked_high].sum(0)
        sites[:, 5] += pool_presence.sum(0)
        for d in DEGREES:
            eligible_degree, chosen_degree = pool_degree == d, picked_degree == d
            available, selected = eligible_degree.sum(), chosen_degree.sum()
            expected_choose = len(group) * available / len(ids)
            above_wt = int(np.sum(chosen_degree & picked_high))
            expected_above_wt = selected * pool_high[eligible_degree].mean() if available else 0.0
            counts[d - 1] += (selected, expected_choose, above_wt, expected_above_wt,
                              group.measured_fitness.to_numpy()[chosen_degree].sum(), available)
            if available:
                sites[:, 2] += selected * pool_presence[eligible_degree].mean(0)
                sites[:, 4] += selected * pool_presence[eligible_degree & pool_high].sum(0) / available

    degree_records, site_records = [], []
    for arm, (counts, sites) in arm_totals.items():
        for d, (observed, expected, hits, expected_hits, measured_sum, eligible) in zip(DEGREES, counts, strict=True):
            degree_records.append({"study": name, "arm": arm, "mutation_count": d, "selected": int(observed),
                                   "expected_uniform_selected": expected, "selected_over_expected": observed / expected if expected else np.nan,
                                   "above_wt_hits": int(hits), "expected_above_wt_with_same_degree_counts": expected_hits,
                                   "above_wt_rate": hits / observed if observed else np.nan,
                                   "mean_measured_fitness": measured_sum / observed if observed else np.nan,
                                   "eligible_candidate_round_exposures": int(eligible)})
        for i, (observed, expected, degree_expected, hits, expected_hits, exposed) in enumerate(sites, 1):
            site_records.append({"study": name, "arm": arm, "position": i, "assay_residue": GFP_WT[i - 1],
                                 "selected": int(observed), "expected_uniform_selected": expected,
                                 "expected_uniform_given_selected_edit_counts": degree_expected,
                                 "selected_over_edit_count_matched_expected": observed / degree_expected if degree_expected else np.nan,
                                 "above_wt_hits": int(hits), "expected_above_wt_hits_matching_edit_counts": expected_hits,
                                 "above_wt_fraction_of_selected": hits / observed if observed else np.nan,
                                 "eligible_candidate_round_exposures": int(exposed)})
    return pd.DataFrame(degree_records), pd.DataFrame(site_records)


def _structure_frames(development, test, dataset, rows, coordinates):
    keys = sorted(set(development.variant_id) | set(test.variant_id))
    descriptions = pd.DataFrame({"variant_id": key, **variant_descriptors(dataset.variants[key], rows, coordinates)}
                                for key in keys)
    return (development.merge(descriptions, on="variant_id", validate="many_to_one"),
            test.merge(descriptions, on="variant_id", validate="many_to_one"), descriptions)


def structural_extension(development, test, selections, dataset, rows, coordinates) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit a dev-only residual correction. Held-out scores cannot fit scaler, imputer or regression."""
    if set(development.variant_id) & set(test.variant_id):
        raise ValueError("Structural residual calibration may only use disjoint development genotypes")
    thresholds = dataset.scores[WT_ID]
    development, test, descriptions = _structure_frames(development, test, dataset, rows, coordinates)
    base_features = ["mutation_count", "unseen_edits", "unseen_sites", "predicted", "risk"]
    structural_features = list(descriptions.columns.drop("variant_id"))
    results = []
    for budget, model in selections.items():
        dev = development[(development.budget == budget) & (development.model == model)]
        evaluation = test[(test.budget == budget) & (test.model == model)].copy()
        evaluation["frozen_prediction"] = evaluation.predicted
        for label, extras in (("control_residual", []), ("structure_residual", structural_features)):
            features = [*base_features, *extras]
            estimator = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), Ridge(alpha=32.0))
            estimator.fit(dev[features], dev.measured - dev.predicted)
            evaluation[label] = evaluation.predicted + estimator.predict(evaluation[features])
        for block, frame in evaluation.groupby("block", sort=True):
            for arm in ("frozen_prediction", "control_residual", "structure_residual"):
                results.append({"budget": budget, "selected_model": model, "block": block, "method": arm,
                                **accuracy(frame.variant_id.tolist(), frame.measured.to_numpy(), frame[arm].to_numpy(), thresholds)})
    return pd.DataFrame(results), descriptions


def structure_associations(descriptions, test, selections) -> pd.DataFrame:
    evaluation = test.merge(descriptions, on="variant_id", validate="many_to_one")
    rows = []
    for budget, model in selections.items():
        data = evaluation[(evaluation.budget == budget) & (evaluation.model == model)].copy()
        data["absolute_error"] = (data.predicted - data.measured).abs()
        for feature in ("mean_ca_contacts", "min_ca_to_centroid_A", "minimum_pair_ca_distance_A"):
            for degree, frame in data.groupby("mutation_count", sort=True):
                clean = frame[[feature, "absolute_error", "measured"]].dropna()
                if len(clean) < 10 or clean[feature].nunique() <= 1:
                    continue
                rows.append({"budget": budget, "model": model, "mutation_count": degree,
                             "descriptor": feature, "complete_genotypes": len(clean),
                             "spearman_to_absolute_error": spearmanr(clean[feature], clean.absolute_error).statistic,
                             "spearman_to_measured_fitness": spearmanr(clean[feature], clean.measured).statistic})
    return pd.DataFrame(rows)


def _csv(output, name, frame):
    frame.to_csv(output / f"{name}.csv", index=False, lineterminator="\n", mode="x")


def plot_summary(output, histogram, degree_summary, sites, means):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(10.7, 7.4), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    colors = ("#2b657a", "#c86e4b", "#728936", "#6f6594")
    axes[0, 0].bar(histogram.mutation_count, 100 * histogram.above_wt_rate, color=colors[0])
    for row in histogram.itertuples():
        axes[0, 0].text(row.mutation_count, 100 * row.above_wt_rate + 0.45, f"n={row.variants:,}",
                        ha="center", fontsize=8)
    axes[0, 0].set(title="Measured archive", ylabel="Above WT (%)", xlabel="Substitutions", xticks=list(DEGREES), ylim=(0, 25))

    agent = degree_summary[degree_summary.study == "agent"].pivot(index="arm", columns="mutation_count", values="selected")
    names = [name.split("_")[0] for name in agent.index]
    bottom = np.zeros(len(agent))
    for color, degree in zip(colors, DEGREES, strict=True):
        axes[0, 1].bar(names, agent[degree], bottom=bottom, label=str(degree), color=color)
        bottom += agent[degree].to_numpy()
    axes[0, 1].set(title="Closed-loop selection by policy", xlabel="A random · B score · C risk · D LLM · E knowledge",
                  ylabel="Selections of 120")
    axes[0, 1].legend(title="Substitutions", fontsize=8, ncol=2, loc="center right")

    agent_sites = sites[(sites.study == "agent") & sites.arm.str.startswith(("D_", "E_"))]
    top = agent_sites.groupby("position", sort=True)[["selected", "expected_uniform_given_selected_edit_counts"]].sum()
    top = top.sort_values("selected", ascending=False, kind="stable").head(8).sort_values("selected", kind="stable")
    y = np.arange(len(top))
    axes[1, 0].barh(y - 0.16, top.selected, height=0.31, label="D + E selected", color=colors[0])
    axes[1, 0].barh(y + 0.16, top.expected_uniform_given_selected_edit_counts, height=0.31,
                    label="Degree-matched roster", color=colors[1])
    axes[1, 0].set(title="Most selected LLM sites", xlabel="Number of candidate selections",
                  yticks=y, yticklabels=[f"{GFP_WT[i - 1]}{i}" for i in top.index], xlim=(0, 18))
    axes[1, 0].legend(fontsize=8)

    ranked = means.pivot(index="budget", columns="method", values="wt_ap").loc[["low", "ample"]]
    x = np.arange(len(ranked))
    methods = (("frozen_prediction", "Frozen model"), ("control_residual", "Residual controls"),
               ("structure_residual", "Controls + structure"))
    for i, ((key, label), color) in enumerate(zip(methods, colors[:3], strict=True)):
        axes[1, 1].bar(x + (i - 1) * 0.24, ranked[key], width=0.23, label=label, color=color)
    axes[1, 1].set(title="Post-hoc structure increment", ylabel="Above-WT AP", xlabel="Training observations",
                  xticks=x, xticklabels=["96", "2,048"], ylim=(0, 0.5))
    axes[1, 1].legend(fontsize=8)
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output / "landscape_overview.png", dpi=180)


def run(dataset, evidence: Path, output: Path, pdb: Path) -> None:
    mapped, metadata, coordinates = read_mapping(pdb)
    rows = {entry["position"]: entry for entry in mapped}
    prediction = evidence / "prediction"
    selection = json.loads((prediction / "selection.json").read_text())["selected"]
    development = pd.read_csv(prediction / "validation_predictions.csv")
    test = pd.read_csv(prediction / "test_predictions.csv")
    archive = archive_frame(dataset)
    output.mkdir(parents=True, exist_ok=False)
    histogram = distributions(archive)
    _csv(output, "archive_by_mutation_count", histogram)
    deviations, deltas = nonadditivity(dataset)
    _csv(output, "single_multi_nonadditivity", deviations)
    _csv(output, "single_multi_summary", deltas)
    _csv(output, "prediction_error_by_mutation_count", prediction_strata(test, dataset.scores[WT_ID]))
    _csv(output, "archive_and_holdout_sites", archived_positions(dataset, archive, test, selection["low"]))
    degrees, sites = [], []
    for trial in ("acquisition", "agent"):
        degree, site = selection_preference(dataset, evidence / trial, trial)
        degrees.append(degree)
        sites.append(site)
    degree_summary, site_summary = pd.concat(degrees, ignore_index=True), pd.concat(sites, ignore_index=True)
    _csv(output, "selection_by_mutation_count", degree_summary)
    _csv(output, "selection_sites_adjusted", site_summary)
    _csv(output, "structure_position_map", pd.DataFrame(mapped))
    augmented, descriptors = structural_extension(development, test, selection, dataset, rows, coordinates)
    _csv(output, "structure_fold_comparison", augmented)
    key_metrics = ["spearman", "pearson", "mse", "mae", "wt_ap", "wt_precision_5", "wt_precision_10"]
    summary = augmented.groupby(["budget", "method"], sort=True)[key_metrics].mean().reset_index()
    _csv(output, "structure_comparison_means", summary)
    _csv(output, "structure_variant_features", descriptors)
    _csv(output, "structure_stratified_correlations", structure_associations(descriptors, test, selection))
    plot_summary(output, histogram, degree_summary, site_summary, summary)
    metadata.update({"assay_sha256": dataset.provenance["assay_sha256"],
                     "frozen_prediction_inputs_sha256": {name: _digest(prediction / name)
                         for name in ("selection.json", "validation_predictions.csv", "test_predictions.csv")},
                     "risk_and_error_analysis": "post hoc using pre-existing held-out predictions",
                     "residual_extension": "supplementary fixed alpha=32 Ridge, trained on separate validation residuals",
                     "unit": "log10 fluorescence", "wild_type": dataset.scores[WT_ID], "api_calls": 0,
                     "archive_genotypes_in_distribution": len(archive), "fully_paired_single_multi_genotypes": len(deviations),
                     "status": "complete"})
    (output / "provenance.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=EVIDENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pdb", type=Path, default=DEFAULT_PDB)
    parser.add_argument("--data-cache", type=Path)
    args = parser.parse_args()
    run(load_dataset(args.data_cache), args.evidence, args.output, args.pdb)


if __name__ == "__main__":
    main()
