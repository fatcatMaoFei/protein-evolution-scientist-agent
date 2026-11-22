"""Build an evidence-linked four-page report from the two completed studies."""
from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

LABEL = {
    "A_random": "Random", "B_greedy": "RF greedy", "C_llm": "LLM",
    "D_knowledge_llm": "LLM + knowledge",
}
PALETTE = dict(zip(LABEL, ("#778596", "#175B6D", "#A34C36", "#77528C"), strict=True))
NAMES = {
    "train_mean": "Training mean", "ridge_additive": "Ridge, additive",
    "ridge_pairwise": "Ridge, pairwise", "random_forest": "Random Forest",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def num(value: float, digits: int = 3) -> str:
    return "NA" if pd.isna(value) else f"{value:.{digits}f}"


def signed(value: float) -> str:
    return f"{value:+.2f}"


def charts(rounds: pd.DataFrame, gb1: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#A8B6C0", "text.color": "#213447",
        "axes.labelcolor": "#213447", "xtick.color": "#213447", "ytick.color": "#213447",
    })
    fig, axes = plt.subplots(1, 3, figsize=(8, 2.7), sharey=True, layout="constrained")
    for axis, (seed, rows) in zip(axes, rounds.groupby("seed"), strict=True):
        axis.grid(axis="y", color="#E6ECF0", linewidth=0.8)
        for arm, label in LABEL.items():
            study = rows[rows.arm == arm].sort_values("round")
            axis.step([0, *study.tested], [0, *study.gain_vs_initial_best_mpa], where="post",
                      marker="o", markersize=3, color=PALETTE[arm], linewidth=1.6, label=label)
        axis.set(title=f"Seed {seed}", xticks=[0, 5, 10, 15], xlabel="Measurements")
    axes[0].set_ylabel("Best gain (MPa)")
    fig.savefig(output / "concrete_trajectories.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 2.2), layout="constrained")
    order = gb1[gb1.model != "train_mean"]
    x = [NAMES[name] for name in order.model]
    ax.plot(x, order.spearman, "o-", label="Spearman", color="#175B6D")
    ax.plot(x, order.ap_above_wt, "o-", label="Average precision", color="#A34C36")
    ax.axhline(gb1.target_above_wt_prevalence.iloc[0], color="#778596", linestyle=":",
               label="Above-WT prevalence")
    ax.grid(axis="y", color="#E6ECF0", linewidth=0.8)
    ax.set(ylim=(0, 1), ylabel="Test metric")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.savefig(output / "gb1_prediction.png", dpi=200)
    plt.close(fig)


class Manuscript:
    def __init__(self):
        fonts = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        for font, filename in (("Sans", "DejaVuSans"), ("Bold", "DejaVuSans-Bold"), ("Serif", "DejaVuSerif")):
            pdfmetrics.registerFont(TTFont(font, str(fonts / (filename + ".ttf"))))
        pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Bold", italic="Sans", boldItalic="Bold")
        navy, teal, gray = map(colors.HexColor, ("#182B3B", "#175B6D", "#526675"))
        self.gray = gray
        opts = {
            "headline": ("Serif", 25, 30, navy, 12),
            "deck": ("Sans", 9.6, 14, teal, 12),
            "section": ("Bold", 14, 19, navy, 6),
            "body": ("Sans", 9.2, 14, navy, 9),
            "note": ("Sans", 7.7, 11.2, gray, 8),
            "cell": ("Sans", 8.1, 11, navy, 0),
            "cell_right": ("Sans", 8.1, 11, navy, 0),
        }
        self.styles = {
            name: ParagraphStyle(name, fontName=font, fontSize=size, leading=leading,
                                 textColor=color, spaceAfter=after,
                                 alignment=TA_RIGHT if name == "cell_right" else 0,
                                 keepWithNext=name in ("headline", "section"))
            for name, (font, size, leading, color, after) in opts.items()
        }
        self.story = []

    def para(self, value, style="body"):
        self.story.append(Paragraph(value, self.styles[style]))

    def table(self, headers, rows, widths):
        values = [
            [Paragraph(escape(str(cell)), self.styles["cell" if i == 0 else "cell_right"])
             for i, cell in enumerate(row)]
            for row in (headers, *rows)
        ]
        table = Table(values, colWidths=widths, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE9ED")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F7F8")]),
            ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.HexColor("#C7D4DB")),
        ]))
        self.story.extend([table, Spacer(1, 8)])

    def page(self):
        self.story.append(PageBreak())

    def footer(self, canvas, doc):
        canvas.setStrokeColor(colors.HexColor("#CED9E0"))
        canvas.line(51, 42, 544, 42)
        canvas.setFont("Sans", 8)
        canvas.setFillColor(self.gray)
        canvas.drawString(51, 29, "Scientific agent assessment")
        canvas.drawRightString(544, 29, f"{doc.page}")

    def build(self, output):
        document = SimpleDocTemplate(
            str(output / "scientific-agent-assessment.pdf"), pagesize=A4, leftMargin=51,
            rightMargin=51, topMargin=48, bottomMargin=54,
            title="Fixed-budget scientific search with measured feedback",
            author="RA technical assessment | author review pending",
        )
        document.build(self.story, onFirstPage=self.footer, onLaterPages=self.footer)


def build_report(evidence: Path, output: Path) -> None:
    concrete, gb = evidence / "concrete", evidence / "gb1"
    for folder, expected in ((concrete, 180), (evidence / "smoke", 20)):
        status = load_json(folder / "status.json")
        if (status.get("status"), status.get("measurements")) != ("complete", expected):
            raise ValueError(f"Expected {expected} completed measurements at {folder}")
    rounds, measures, methods, pairs, failures, predict = [
        pd.read_csv(concrete / f"{name}.csv")
        for name in ("rounds", "selections", "methods", "comparisons", "failures", "predictive")
    ]
    gb1 = pd.read_csv(gb / "metrics.csv")
    gbtest = gb1[gb1.partition == "test"]
    interaction = load_json(gb / "single_effect_transfer.json")
    check = load_json(evidence / "verification.json") if (evidence / "verification.json").exists() else None
    small = pd.read_csv(gb / "exploratory_single_effect_transfer.csv")
    small = small[small.partition == "validation"].sort_values("mutation_depth")
    transfer_errors = list(zip(small.observed_not_above_wt_expected_above_wt, small.covered_by_training_singles))
    audit = load_json(gb / "audit.json")
    ablation = pd.read_csv(gb / "position_ablation.csv")
    errors = pd.read_csv(gb / "error_strata.csv")
    output.mkdir(parents=True, exist_ok=True)
    charts(rounds, gbtest, output)
    m = Manuscript()

    def deltas(left, right):
        subset = pairs[(pairs.left_arm == left) & (pairs.right_arm == right)].sort_values("seed")
        return subset.right_minus_left_best_mpa.tolist()

    c_b = deltas("B_greedy", "C_llm")
    d_c = deltas("C_llm", "D_knowledge_llm")
    leader = gbtest[gbtest.spearman.notna()].sort_values("spearman").iloc[-1]
    scores = gbtest.set_index("model").spearman

    m.para("RA technical assessment · November 2026", "deck")
    m.para("Fixed-budget scientific search<br/>with measured feedback", "headline")
    m.para("Retrospective FLIP GB1 prediction and a four-policy materials replay", "deck")
    m.para(
        f"In a three-seed replay using measured 28-day concrete strengths, the LLM's final best minus RF greedy "
        f"was {', '.join(map(signed, c_b))} MPa. Adding knowledge changed that final best by "
        f"{', '.join(map(signed, d_c))} MPa. In the separate GB1 task, {NAMES[leader.model]} "
        f"reached the highest held-out Spearman ({leader.spearman:.3f})."
    )
    m.para("Four-substitution GB1 prediction", "section")
    m.para(
        "Wu et al. experimentally characterized the relative enrichment of a four-site GB1 library [1]. "
        "FLIP turned these measurements into prediction splits [2]. The official four-substitution test has "
        f'{audit["partitions"]["test"]["rows"]:,} records. Training used {audit["partitions"]["train"]["rows"]:,} '
        f'lower-depth records, with {audit["partitions"]["validation"]["rows"]:,} for hyperparameter validation. '
        "One-hot features preserve residue identity by position. Pairwise Ridge adds categories for each of the "
        "six position pairs. Test scores are computed after selecting hyperparameters on validation."
    )
    m.table(
        ["Model", "Spearman", "Pearson", "MSE", "AP", f"P@{int(leader.top_k)}"],
        [[NAMES[r.model], num(r.spearman), num(r.pearson), num(r.mse),
          num(r.ap_above_wt), num(r.top_k_precision_above_wt)] for r in gbtest.itertuples()],
        [145, 76, 67, 67, 67, 67],
    )
    m.para(
        f"<b>Table 1.</b> Four-substitution prediction. AP and precision mark measured enrichment "
        f"above the wild-type score of 1 as positive ({leader.target_above_wt_prevalence:.1%} prevalence). "
        "The constant model has undefined correlations. Evidence: GB1 metrics.csv.", "note"
    )
    m.para(
        f"Pairwise features changed Spearman by {scores.ridge_pairwise-scores.ridge_additive:+.3f} "
        "relative to additive Ridge. The direct multiplicative baseline could score "
        f"{interaction['covered_by_training_singles']:,}/{audit['partitions']['test']['rows']:,} four-substitution records "
        "from the available training singles. This coverage gap leaves direct single-effect transfer to the "
        "four-substitution test unestimated."
    )
    m.para(
        "In a separate exploratory check on covered validation records, the multiplicative estimate incorrectly "
        f"placed {transfer_errors[0][0]}/{transfer_errors[0][1]} doubles and "
        f"{transfer_errors[1][0]}/{transfer_errors[1][1]} triples above WT. Missing constituent single effects make these "
        "small subsets selective. Evidence: GB1 exploratory_single_effect_transfer.csv.", "note"
    )
    m.page()

    m.para("Measured acquisition under equal budgets", "headline")
    m.para(
        "The catalog contains 416 unique 28-day concrete mixes [3]. Repeated observations of a mix "
        "contribute their median MPa. Each paired split has 48 shared initial observations, 96 validation "
        "mixes, 176 replay candidates and 96 final prediction-test mixes. Each arm submits five IDs for "
        "three rounds. The evaluator releases only the submitted measurements to that arm before refitting."
    )
    m.para(
        "The random arm samples from the pool. RF greedy takes the highest model scores. Both LLM arms use "
        "CodeFlow DeepSeek V4.1 Flash and see 24 RF leaders plus 24 seeded non-elite candidates each round. "
        "The knowledge arm also receives materials guidance and summaries from its own measured history."
    )
    m.story.append(Image(str(output / "concrete_trajectories.png"), width=482, height=163))
    m.story.append(Spacer(1, 4))
    m.para(
        "<b>Figure 1.</b> Gain over the initial best, by budget and seed. "
        '<font color="#778596">Random</font> · <font color="#175B6D">RF greedy</font> · '
        '<font color="#A34C36">LLM</font> · <font color="#77528C">LLM + knowledge</font>. '
        "Coincident curves represent equal observed gains. Evidence: Concrete rounds.csv.", "note"
    )
    rows = []
    for arm, label in LABEL.items():
        arm_rows = methods[methods.arm == arm].sort_values("seed")
        hit_positions = ", ".join("-" if pd.isna(v) else str(int(v)) for v in arm_rows.first_hit_tests)
        gains = ", ".join(f"{v:.2f}" for v in arm_rows.gain_vs_initial_best_mpa)
        hits = int(arm_rows.top_hits.sum())
        rows.append([label, gains, f"{hits}/45 ({hits/45:.0%})", hit_positions])
    m.table(["Policy", "Final gain (MPa)", "Top-decile hits", "Queries to first hit"],
            rows, [125, 118, 105, 145])
    m.para(
        "<b>Table 2.</b> Gains and first hits are ordered by seeds 17, 29, 43. The hit cutoff is each "
        "seed's pre-query pool 90th percentile, with ties included. First-hit counts follow recorded batch "
        "order. A dash means censored after 15 queries. Evidence: Concrete methods.csv.", "note"
    )
    m.para(
        f"The mean paired C minus B endpoint difference was {signed(sum(c_b)/len(c_b))} MPa. "
        f"The D minus C difference was {signed(sum(d_c)/len(d_c))} MPa. "
        "An arm may repeatedly hit strong mixes while failing to improve on a strong initial incumbent. "
        "The seed-level endpoint and hit counts therefore capture distinct outcomes."
    )
    m.page()

    m.para("Where the policies departed from greedy", "headline")
    rows = []
    for arm, label in LABEL.items():
        selected = measures[measures.arm == arm]
        rows.append([
            label, f"{selected.is_myopic_greedy_pick.sum()}/{len(selected)}",
            num(selected.rf_rank.mean(), 1), num(selected.nearest_observed_distance.mean(), 2),
            num(selected.water_to_binder.mean()),
            num(selected.actual_minus_predicted_mpa.abs().mean(), 1),
        ])
    m.table(["Policy", "Greedy picks", "Mean RF rank", "Distance", "Water/binder", "MAE (MPa)"],
            rows, [114, 70, 74, 69, 78, 88])
    m.para(
        "<b>Table 3.</b> Rank and greediness use the arm's own contemporaneous RF. Distance is standardized "
        "Euclidean distance to the nearest observed mix. MAE uses only queried recipes. "
        "Evidence: Concrete selections.csv.", "note"
    )
    llm = measures[measures.arm.isin(("C_llm", "D_knowledge_llm"))]
    exploratory = llm[llm.is_myopic_greedy_pick == 0]
    m.para(
        f"The two LLM arms selected {len(exploratory)}/{len(llm)} mixes beyond their own greedy top five. "
        f"{int(exploratory.top_hit.sum())} cleared the top-decile threshold. This establishes that the LLM "
        "policies changed what was tested, while the paired endpoint differences quantify realized improvement."
    )
    if check is not None:
        sampled = sum(d["selected_from_non_elite_menu"] for d in check["formal"]["selection_diagnostics"].values())
        m.para(
            f"The seeded non-elite menu contributed {sampled}/{len(llm)} LLM picks. Departures from the "
            "greedy top five remained within the model's high-ranked menu in this run. Evidence: per-round "
            "agent traces and verification.json."
        )
    m.para("Examples of prediction error at selection", "section")
    llm_fail = failures[failures.arm.isin(("C_llm", "D_knowledge_llm"))]
    low = llm_fail.sort_values("actual_minus_predicted_mpa").iloc[-1]
    high = llm_fail.sort_values("actual_minus_predicted_mpa").iloc[0]
    for description, row in (("Largest underestimate", low), ("Largest overestimate", high)):
        m.para(
            f"<b>{description}.</b> {LABEL[row.arm]} queued {row.id} at RF rank {row.rf_rank} "
            f"(seed {row.seed}, round {row['round']}). Predicted {row.prediction_mpa:.2f} MPa, "
            f"measured {row.strength_mpa:.2f} MPa. Evidence: Concrete failures.csv."
        )
    pred = predict[predict.stage == "final_test"].groupby("model")[["spearman", "mse_mpa_squared"]].mean()
    m.para(
        f"Across the 12 final arm evaluations, RF and Ridge averaged "
        f"{pred.loc['RandomForest','spearman']:.3f} and {pred.loc['Ridge','spearman']:.3f} Spearman. "
        f"Their MSEs were {pred.loc['RandomForest','mse_mpa_squared']:.1f} and "
        f"{pred.loc['Ridge','mse_mpa_squared']:.1f} MPa squared. Global prediction results cover "
        "the held-out 96 mixes per seed. Acquisition error covers a policy-selected subset. "
        "Evidence: Concrete predictive.csv."
    )
    m.para("GB1 site dependence and upper-tail error", "section")
    top = ablation.sort_values("spearman_change_from_full").iloc[0]
    rf_high = errors[(errors.model == "random_forest") & (errors.measured_score_band == "above_wt")].iloc[0]
    m.para(
        f"On validation, removing construct coordinate {int(top.omitted_position)} reduced RF Spearman "
        f"by {abs(top.spearman_change_from_full):.3f}, the largest of the four one-site ablations. "
        f"On held-out records above WT, RF mean prediction error was "
        f"{rf_high.mean_error_predicted_minus_measured:+.3f}. "
        "These diagnostics describe predictive reliance and calibration. "
        "Evidence: GB1 position_ablation.csv and error_strata.csv."
    )
    m.page()

    m.para("Interpretation and reproducibility", "headline")
    c_hits = int(methods.loc[methods.arm == "C_llm", "top_hits"].sum())
    d_hits = int(methods.loc[methods.arm == "D_knowledge_llm", "top_hits"].sum())
    m.para(
        f"The knowledge arm measured {d_hits}/45 top-decile hits versus {c_hits}/45 for the plain LLM. "
        f"Their terminal best-strength differences by seed were {', '.join(map(signed, d_c))} MPa. "
        "The hit count and the top outcome can therefore respond differently to a policy change."
    )
    m.para(
        "The measured replay estimates performance of complete acquisition policies for three paired splits "
        "of a finite catalog. The LLM policy combines model inference, an explicit exploration menu, and "
        "tool-returned evidence. Knowledge augmentation combines fixed materials guidance with "
        "observation-derived summaries. Its two ingredients would need separate ablations to identify their "
        "effects. Seed-level comparisons remain descriptive because the splits can reuse catalog entries."
    )
    m.para(
        "GB1 shows that pairwise information helps transfer scores to four-substitution profiles in the FLIP "
        "split. That prediction benchmark addresses a different question from the fixed-budget search on "
        "concrete. The policy results apply to the measured concrete catalog."
    )
    m.para("Execution and measurement interface", "section")
    m.para(
        "The validator checks source hashes and split assignments. Each concrete arm owns its experimental "
        "state and budget. The policy builds a shortlist and an observed-data summary. DeepSeek issues a "
        "native score tool call and then submits an ordered JSON decision. The critic checks order, "
        "IDs, membership and budget before the oracle returns labels."
    )
    m.para(
        "An instrument-facing extension can accept independently approved measurement records with sample "
        "IDs, units, replicates, protocol version and timestamps. An append-only import can validate "
        "provenance and preserve the distinction between predictions and measured results."
    )
    m.para("AI use and author review", "section")
    m.para(
        "OpenAI Codex specified the concrete replay under the assessment brief, implemented and ran both "
        "studies, and drafted this report. The two experimental agent arms used CodeFlow DeepSeek V4.1 Flash. "
        "Each log records the exact non-secret request, returned model ID, tool interactions, "
        "validation, timing and token use. Human author review remains pending. "
        "See protocol.md and disclosure.md."
    )
    m.para("Primary sources", "section")
    for reference in (
        '[1] Wu et al. (2016), <i>Adaptation in protein fitness landscapes is facilitated by indirect paths</i>. '
        '<link href="https://doi.org/10.7554/eLife.16965">eLife 5:e16965</link>.',
        '[2] Dallago et al. (2021), <i>FLIP: Benchmark tasks in fitness landscape inference for proteins</i>. '
        '<link href="https://doi.org/10.1101/2021.11.09.467890">doi:10.1101/2021.11.09.467890</link>.',
        '[3] Yeh, <i>Concrete Compressive Strength</i>. '
        '<link href="https://doi.org/10.24432/C5PK67">UCI Machine Learning Repository</link>, CC BY 4.0.',
        '[4] <link href="https://api-docs.deepseek.com/guides/tool_calls">DeepSeek function calling</link> '
        'and the <link href="https://codeflow.sh">CodeFlow</link> gateway. '
        'Endpoint and observed model IDs are in the execution traces.',
    ):
        m.para(reference, "note")
    m.build(output)
