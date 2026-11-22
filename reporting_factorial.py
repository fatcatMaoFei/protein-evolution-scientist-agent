"""Render the matched four-model analysis in the scientific report."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from reportlab.platypus import Image, Spacer

LABEL = {"deepseek": "DeepSeek", "sonnet": "Sonnet", "gpt": "GPT-6 Luna", "qwen": "Qwen3-4B", "greedy": "RF greedy"}
COLORS = {"deepseek": "#9C4C38", "sonnet": "#1A6274", "gpt": "#6E4C87", "qwen": "#507038", "greedy": "#71808D"}


def add_factorial_page(manuscript, evidence, output):
    folder = evidence / "across-models"
    cells = pd.read_csv(folder / "cell_summary.csv", keep_default_na=False)
    rounds = pd.read_csv(folder / "round_cells.csv", keep_default_na=False)
    paired = pd.read_csv(folder / "contrasts_by_seed.csv")
    usage = pd.read_csv(folder / "usage.csv")
    models = [key for key in LABEL if key != "greedy" and key in set(cells.model)]

    figure_file = output / "factorial_trajectories.png"
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.4), sharey=True, layout="constrained")
    for ax, condition in zip(axes, ("off", "on"), strict=True):
        ax.grid(axis="y", color="#E6ECF0", linewidth=0.8)
        for model in ("greedy", *models):
            level = "na" if model == "greedy" else condition
            block = rounds[(rounds.model == model) & (rounds.knowledge == level)]
            curve = block.groupby("round").gain_vs_initial_best_mpa.mean()
            ax.step([0, *[5 * number for number in curve.index]], [0, *curve], where="post",
                    label=LABEL[model], color=COLORS[model], marker="o", markersize=2.8, linewidth=1.4,
                    linestyle=":" if model == "greedy" else "-")
        ax.set(xticks=[0, 5, 10, 15], xlabel="Measurements", title=f"Knowledge {condition.upper()}")
    axes[0].set_ylabel("Mean best gain (MPa)")
    axes[0].legend(loc="upper left", frameon=False, fontsize=7, ncol=2)
    fig.savefig(figure_file, dpi=200)
    plt.close(fig)

    m = manuscript
    m.para("Model-by-knowledge robustness", "headline")
    m.para(
        "Models receive the same first-round menu and run the same evaluator and critic. Native transports and "
        "decoding settings differ across models. Knowledge context is the sole assigned change within a model."
    )
    summary = []
    for model in models:
        off, on = [
            cells[(cells.model == model) & (cells.knowledge == condition)].iloc[0]
            for condition in ("off", "on")
        ]
        summary.append([
            LABEL[model], f"{off.gain_vs_initial_best_mpa:.2f} / {on.gain_vs_initial_best_mpa:.2f}",
            f"{off.top_hits * off.n_seeds:.0f} / {on.top_hits * on.n_seeds:.0f}",
            f"{off.mean_greedy_overlap:.0%} / {on.mean_greedy_overlap:.0%}",
        ])
    m.table(["Model", "Mean gain, MPa", "Hits / 45", "Own top-five overlap"],
            summary, [126, 118, 97, 152])
    greedy = cells[cells.model == "greedy"].iloc[0]
    m.para(
        f"<b>Table 4.</b> Knowledge OFF / ON. Shared RF greedy gained {greedy.gain_vs_initial_best_mpa:.2f} MPa "
        f"and hit {greedy.top_hits * greedy.n_seeds:.0f}/45. Evidence: across-models/cell_summary.csv.", "note"
    )
    m.story.extend([Image(str(figure_file), width=482, height=137), Spacer(1, 4)])
    m.para(
        "<b>Figure 2.</b> Mean incumbent gain across the same three seeds at each round. "
        "The RF control is shared. Coincident trajectories indicate equal mean gains.", "note"
    )

    def deltas(contrast, metric, digits):
        rows = paired[paired.contrast == contrast].sort_values("seed")
        return ", ".join(f"{value:+.{digits}f}" for value in rows[metric])

    rows = []
    for model in models:
        simple = f"{model}_knowledge_on_minus_off"
        interaction = f"interaction_{model}_minus_deepseek"
        rows.append([
            LABEL[model], deltas(simple, "gain_vs_initial_best_mpa", 2), deltas(simple, "top_hits", 0),
            "-" if model == "deepseek" else deltas(interaction, "gain_vs_initial_best_mpa", 2),
            "-" if model == "deepseek" else deltas(interaction, "top_hits", 0),
        ])
    m.table(["Model", "Knowledge Δ MPa", "Knowledge Δ hits", "Interaction MPa", "Interaction hits"],
            rows, [97, 112, 88, 108, 88])
    m.para(
        "<b>Table 5.</b> Each triple gives seeds 17, 29, 43. Knowledge effect is ON minus OFF. Interaction "
        "subtracts DeepSeek's knowledge effect. Evidence: across-models/contrasts_by_seed.csv.", "note"
    )
    seeded = sum(
        45 * float(row.seeded_menu_fraction) for row in cells.itertuples() if row.model in models
    )
    m.para(
        f"The {2 * len(models)} LLM cells selected {int(round(seeded))}/{90 * len(models)} mixes "
        "from the seeded lower-ranked half of the menus. The top-five overlap reports reranking "
        "relative to each model's contemporaneous fitted predictor."
    )
    m.para(
        f"The {len(models)} models completed {int(usage.decisions.sum())} formal decisions in "
        f"{int(usage.recorded_model_responses.sum())} logged model responses. "
        "Usage is recorded per native tokenizer and includes cache accounting where applicable.", "note"
    )
