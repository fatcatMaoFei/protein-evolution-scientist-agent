"""Render the independent five-page GFP V2 acceptance report."""
from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Image, SimpleDocTemplate, Spacer

from reporting import Manuscript


def _table(book, head, rows, widths):
    book.table(head, rows, widths)


def _section(book, title, content):
    book.para(title, "section")
    for paragraph in content:
        book.para(escape(paragraph))


def _paired_figure(contrasts):
    image = BytesIO()
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False}):
        figure, ax = plt.subplots(figsize=(7.6, 2.65), layout="constrained")
        for position, row in enumerate(contrasts):
            estimate = row["mean"]
            ax.scatter(row["blocks"], [position] * len(row["blocks"]), color="#8eabb1", s=15, alpha=0.65)
            ax.errorbar(estimate, position, xerr=[[estimate - row["low"]], [row["high"] - estimate]],
                        fmt="D", color="#175b6d", capsize=3, zorder=3, markersize=5)
        ax.set_yticks(range(len(contrasts)), [row["label"] for row in contrasts])
        ax.invert_yaxis()
        ax.axvline(0, color="#657789", linestyle="--", linewidth=0.8)
        ax.grid(axis="x", color="#e6ecef", linewidth=0.6)
        ax.set_xlabel("Best newly measured fitness difference (log10 fluorescence)")
        figure.savefig(image, format="png", dpi=180)
        plt.close(figure)
    image.seek(0)
    return image


def _trajectory_figure(series):
    image = BytesIO()
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8.0, "axes.spines.top": False,
                         "axes.spines.right": False}):
        figure, ax = plt.subplots(figsize=(7.6, 2.25), layout="constrained")
        markers = ("s", "^", "o", "v", "D")
        colors = ("#778596", "#175b6d", "#65823b", "#a34c36", "#77528c")
        for row, marker, color in zip(series, markers, colors, strict=True):
            ax.plot(row["rounds"], row["new_best"], marker=marker, color=color, label=row["policy"], markersize=4)
        ax.set(xticks=(1, 2, 3), xlabel="Trial round", ylabel="Best new measurement")
        ax.grid(axis="y", color="#e6ecef", linewidth=0.6)
        figure.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=7.4,
                      ncol=5, loc="outside upper center")
        figure.savefig(image, format="png", dpi=180)
        plt.close(figure)
    image.seek(0)
    return image


def render(evidence: Path, output: Path) -> None:
    studies = json.loads((evidence / "report_inputs.json").read_text(encoding="utf-8"))
    book = Manuscript()

    book.para("RA technical assessment | GFP V2", "deck")
    book.para("GFP fitness prediction and scientific decisions", "headline")
    book.para("Independent comparisons using published fluorescence measurements", "deck")
    _section(book, "What changed", studies["overview"])
    _table(book, ["Gate", "Verdict", "Evidence supporting the verdict"], studies["gates"], [50, 83, 360])
    _section(book, "V1 evidence and limits", studies["v1"])
    _table(book, ["Official V1 arm", "Final best", "Initial-best gain", "> WT"], studies["v1_comparators"],
           [190, 96, 108, 99])
    book.para("See the frozen V1 report and the V2 protocol for the complete study boundaries.", "note")

    book.page()
    book.para("Prediction on unseen archive genotypes", "headline")
    _section(book, "Model selection and independent evaluation", studies["prediction_design"])
    _table(book, ["Training / model", "WT AP", "Spearman", "Pearson", "MSE", "MAE", "Tail P@5", "Tail P@10"],
           studies["prediction"], [142, 44, 56, 50, 44, 43, 56, 58])
    book.para("Test means across held-out blocks. WT AP labels fluorescence above the measured wild type. "
              "Tail precision uses the top measured decile. Full metrics and per-variant results are in the supplement.", "note")
    _section(book, "Error, feature limits and risk", studies["prediction_results"])

    book.page()
    book.para("Does stronger prediction improve acquisition?", "headline")
    _section(book, "Independent query cohorts", studies["acquisition_design"])
    _table(book, ["Policy", "Queries", "New best", "New mean", "> WT", "Gain"], studies["acquisition"],
           [163, 58, 69, 69, 64, 70])
    book.para("Best, mean and gain summarize paired blocks in log10 fluorescence. Gain is the difference from the starting incumbent. "
              "The WT column counts genuinely revealed measurements above the measured reference.", "note")
    _section(book, "Observed trade-offs", studies["acquisition_results"])
    book.story.append(Image(_trajectory_figure(studies["trajectory"]), width=480, height=142))
    book.para("Figure 1. Agent study trajectories. Each point is the mean across eight blocks of the best measurement in that round, in log10 fluorescence.", "note")

    book.page()
    book.para("Do LLM decisions add measured value?", "headline")
    _section(book, "Fair comparison", studies["agent_design"])
    _table(book, ["Policy", "Queries", "New best", "New mean", "> WT", "Gain"], studies["agent"],
           [163, 58, 69, 69, 64, 70])
    if studies["contrasts"]:
        book.story.append(Spacer(1, 8))
        book.story.append(Image(_paired_figure(studies["contrasts"]), width=480, height=167))
        book.para("Figure 2. Dots are paired blocks. Diamonds and bars show sample means and descriptive 95% paired intervals.", "note")
    _section(book, "Measured decisions and audited feedback", studies["agent_results"])

    book.page()
    book.para("Scientific workflow and reproducibility", "headline")
    _section(book, "One prospective decision", studies["case"])
    _section(book, "Failure boundaries", studies["limitations"])
    _section(book, "Run with your own API and GFP data", studies["reproduction"])
    book.para("Primary data: Sarkisyan et al., Nature 2016, doi:10.1038/nature17995. "
              "Protocol, archive inputs, manifests, tests, raw tool responses and archived decisions are recorded in the repository.", "note")

    def footer(canvas, document):
        canvas.setStrokeColor(book.gray)
        canvas.line(51, 42, 544, 42)
        canvas.setFont("Sans", 8)
        canvas.setFillColor(book.gray)
        canvas.drawString(51, 29, "GFP fluorescence directed evolution | V2")
        canvas.drawRightString(544, 29, str(document.page))

    output.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=51, rightMargin=51,
                                 topMargin=48, bottomMargin=54, title="GFP Predictor and Scientific Agent V2",
                                 author="RA technical assessment", invariant=1)
    document.build(book.story, onFirstPage=footer, onLaterPages=footer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=Path("evidence/gfp-v2"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.evidence, args.output)


if __name__ == "__main__":
    main()
