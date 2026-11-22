"""Render a retrospective GB1 report from aggregate evidence only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate

from reporting import Manuscript

ROOT = Path(__file__).resolve().parent
DISPLAY_NAMES = {
    "ridge_additive": "Ridge, additive",
    "ridge_pairwise": "Ridge, pairwise",
    "random_forest": "Random Forest",
    "train_mean": "Training mean",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(value: float) -> str:
    return "NA" if pd.isna(value) else f"{value:.3f}"


def render(evidence: Path, receipt: Path, output: Path) -> None:
    audit = read_json(evidence / "audit.json")
    benchmark = pd.read_csv(evidence / "metrics.csv").set_index(["partition", "model"])
    test = benchmark.loc["test"]
    validation = benchmark.loc["validation"]
    errors = pd.read_csv(evidence / "error_strata.csv")
    ablations = pd.read_csv(evidence / "position_ablation.csv")
    transfer = read_json(evidence / "single_effect_transfer.json")
    exploratory = pd.read_csv(evidence / "exploratory_single_effect_transfer.csv")
    pairwise = test.loc["ridge_pairwise"]
    additive = test.loc["ridge_additive"]
    sources = read_json(evidence / "provenance.json")
    layout = Manuscript()
    layout.para("Research assessment evidence · November 2026", "deck")
    layout.para("GB1 retrospective fitness<br/>prediction on the FLIP split", "headline")
    layout.para(
        "How well do models fit on lower-order substitutions rank four-substitution GB1 measurements? "
        f"Pairwise Ridge reached test Spearman {pairwise.spearman:.3f}, compared with "
        f"{additive.spearman:.3f} for additive Ridge and "
        f"{test.loc['random_forest', 'spearman']:.3f} for Random Forest."
    )
    layout.para("What the archive measures", "section")
    layout.para(
        "Wu et al. measured selection enrichment relative to wild type in a GB1 binding assay. "
        "The score reflects binding and the fraction of folded protein. The wild-type reference "
        "is 1, and larger values indicate greater relative enrichment [1]. FLIP packages these "
        "measurements into an official three-versus-rest split [2]. Its sequence field uses a "
        "265-residue representation derived from chain A of PDB 5LDE. RCSB annotates the "
        "entity as chimeric. The benchmark WT anchor is the unique score-one training record [3]."
    )
    layout.para("Fixed prediction design", "section")
    layout.para(
        f"The source contains {audit['rows']:,} records with zero duplicate sequences and zero missing "
        f"or nonfinite targets. The {audit['partitions']['train']['rows']:,} training and "
        f"{audit['partitions']['validation']['rows']:,} validation records have at most three substitutions. "
        f"The held-out {audit['partitions']['test']['rows']:,} records each have four. "
        "The validation split chose hyperparameters. Final test scores use weights fit on training only."
    )
    layout.table(
        ["Model", "Spearman", "Pearson", "MSE", "AP", f"P@{int(pairwise.top_k)}"],
        [
            [DISPLAY_NAMES[name], *(metric(row[column]) for column in (
                "spearman", "pearson", "mse", "ap_above_wt", "top_k_precision_above_wt"
            ))]
            for name, row in test.iterrows()
        ],
        [145, 76, 67, 67, 67, 71],
    )
    layout.para(
        f"<b>Table 1.</b> Official held-out test. AP and precision mark measured fitness above 1 as "
        f"positive ({pairwise.target_above_wt_prevalence:.1%} prevalence). Precision is measured among "
        "the top 5% by model score, with fractional credit for ties at the threshold. "
        "Constant predictions have undefined correlations. Source: evidence/gb1/metrics.csv.", "note"
    )
    layout.para(
        "The additive model uses position-specific residue indicators. Pairwise Ridge adds cross-position "
        "indicators. This improves ranking on the official split and allows the two model families to differ "
        "in their representation of nonadditivity."
    )

    layout.page()
    layout.para("Generalization and error", "headline")
    layout.para(
        "Pairwise Ridge improved test Spearman by "
        f"{pairwise.spearman - additive.spearman:+.3f}, increased average precision from "
        f"{additive.ap_above_wt:.3f} to {pairwise.ap_above_wt:.3f}, and reduced MSE from "
        f"{additive.mse:.3f} to {pairwise.mse:.3f}. Each model was tuned independently on validation."
    )
    fitted = [name for name in DISPLAY_NAMES if name != "train_mean"]
    stratified = errors[errors.measured_score_band == "above_wt"].set_index("model")
    layout.table(
        ["Model", "Validation rho", "Test rho", "Above-WT MAE", "Above-WT signed error"],
        [[DISPLAY_NAMES[name], metric(validation.loc[name, "spearman"]), metric(test.loc[name, "spearman"]),
          metric(stratified.loc[name, "mae"]),
          f"{stratified.loc[name, 'mean_error_predicted_minus_measured']:+.3f}"] for name in fitted],
        [130, 83, 78, 92, 110],
    )
    layout.para(
        "<b>Table 2.</b> Rank transfer and subgroup calibration. Error uses prediction minus "
        "measurement on held-out records above WT. Source: evidence/gb1/error_strata.csv.", "note"
    )
    layout.para("Position dependence", "section")
    drops = ablations.spearman_change_from_full.abs().sort_values()
    layout.para(
        f"With a fixed forest configuration, leaving one of the four variable positions out reduced validation "
        f"Spearman by {drops.iloc[0]:.3f} to {drops.iloc[-1]:.3f}. This measures reliance of the fitted forest "
        "on sequence information under this split."
    )
    layout.para("Limits of single-substitution evidence", "section")
    layout.para(
        f"The training set has {transfer['training_singles']} measured single substitutions. "
        f"The predeclared multiplicative baseline covered {transfer['covered_by_training_singles']:,} of "
        f"{transfer['evaluated_rows']:,} four-substitution test records. Its test accuracy is unestimable "
        "because each covered record requires all four constituent training singles."
    )
    val_lower = exploratory[exploratory.partition == "validation"]
    layout.table(
        ["Validation edit depth", "Covered / total", "Overpredicted above WT"],
        [[str(int(row.mutation_depth)),
          f"{int(row.covered_by_training_singles)} / {int(row.evaluated_rows)}",
          str(int(row.observed_not_above_wt_expected_above_wt))] for row in val_lower.itertuples()],
        [170, 145, 178],
    )
    layout.para(
        "<b>Table 3.</b> Exploratory check on the lower-order validation records covered by the same "
        "multiplicative baseline. The subsets are small and selected by availability of component "
        "measurements. Source: evidence/gb1/exploratory_single_effect_transfer.csv.", "note"
    )

    layout.page()
    layout.para("Reproducibility and acceptance", "headline")
    layout.para(
        "The locked FLIP snapshot is read from the official repository. The benchmark validates archive "
        "and member hashes, fields, alphabet, construct length, duplicate records and partition membership. "
        "Training, validation and test form disjoint row sets. The source records "
        f"{sources['license_flip']} for the FLIP split and {sources['license_original']} for the original data. "
        "The benchmark writes aggregate "
        "prediction metrics, residual summaries and position-ablation results."
    )
    layout.para("Reproduction", "section")
    layout.para(
        "<font name='Bold'>python run.py gb1 --output runs/gb1-review</font><br/>"
        "Use Python 3.12 and the existing dependency lock. The benchmark command needs data access on "
        "first use. The report reads the archived aggregate tables. Use a fresh output directory."
    )
    if receipt.exists():
        verified = read_json(receipt)
        if verified.get("benchmark_reproduced"):
            comparison = verified["artifact_comparison"]
            layout.para(
                f"The verification run reproduced {comparison['byte_identical_files']} of "
                f"{comparison['files_checked']} archived files byte for byte. Only the retrieval "
                "metadata differs because the rerun verified a cached download."
            )
        tests = verified.get("local_tests", {})
        if tests.get("exit_code") == 0 and isinstance(tests.get("passed"), int):
            layout.para(f"The local suite passed {tests['passed']} tests. Source: evidence/gb1-review/verification.json.")
    layout.para("Research and engineering scope", "section")
    layout.para(
        "GB1 acceptance covers historical assay input, fixed-split prediction, aggregate validation and "
        "reproduction. A GB1 agent experiment, candidate sequence recommendations, iterative "
        "optimization and a feedback audit remain outside this delivery. The historical materials study "
        "has its own archive and separate scientific conclusions."
    )
    layout.para("AI use and provenance", "section")
    layout.para(
        "OpenAI Codex implemented the aggregate benchmark, verified the frozen evidence and drafted "
        "this report. GB1 model fitting uses conventional regressors. This GB1 verification made "
        "zero runtime LLM calls. The materials agents' historical use is disclosed separately."
    )
    layout.para("Primary references", "section")
    for reference in (
        '[1] Wu et al. (2016), <i>Adaptation in protein fitness landscapes is facilitated by indirect paths</i>. '
        '<link href="https://doi.org/10.7554/eLife.16965">eLife 5:e16965</link>.',
        '[2] Dallago et al. (2021), <i>FLIP: Benchmark tasks in fitness landscape inference for proteins</i>. '
        '<link href="https://doi.org/10.1101/2021.11.09.467890">FLIP paper</link>.',
        '[3] <link href="https://www.rcsb.org/structure/5LDE">RCSB PDB 5LDE</link>. '
        'Source metadata and redistribution terms are recorded in evidence/gb1/provenance.json.',
    ):
        layout.para(reference, "note")

    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output), pagesize=A4, leftMargin=51, rightMargin=51, topMargin=48, bottomMargin=54,
        title="GB1 retrospective fitness prediction on the FLIP split",
        author="RA technical assessment | author review pending",
    )

    def footer(canvas, document):
        canvas.setStrokeColor(layout.gray)
        canvas.line(51, 42, 544, 42)
        canvas.setFont("Sans", 8)
        canvas.setFillColor(layout.gray)
        canvas.drawString(51, 29, "GB1 retrospective benchmark")
        canvas.drawRightString(544, 29, str(document.page))

    doc.build(layout.story, onFirstPage=footer, onLaterPages=footer)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=ROOT / "evidence/gb1")
    parser.add_argument("--receipt", type=Path, default=ROOT / "evidence/gb1-review/verification.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.evidence, args.receipt, args.output)


if __name__ == "__main__":
    main()
