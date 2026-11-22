"""Render the GFP scientific assessment from archived measurements and decision logs."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from statistics import mean
from xml.sax.saxutils import escape

import matplotlib
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Image, SimpleDocTemplate, Spacer

from reporting import Manuscript

BACKENDS = {
    "deepseek": "DeepSeek", "sonnet": "Claude Sonnet", "gpt": "GPT", "qwen": "Qwen (local)",
}
ARMS = {
    "A_random": "Random", "B_model_only": "Model-Only",
    "C_llm": "LLM", "D_knowledge_llm": "Knowledge-LLM",
}
COLORS = dict(zip(ARMS, ("#778596", "#175B6D", "#A34C36", "#77528C"), strict=True))
LINE_STYLES = dict(zip(ARMS, ((0, (1, 2)), (0, (4, 2)), "-", (0, (7, 2))), strict=True))
MARKERS = dict(zip(ARMS, ("s", "^", "o", "D"), strict=True))


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _frame(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _number(value, digits: int = 3) -> str:
    return "NA" if value is None or pd.isna(value) else f"{value:.{digits}f}"


def _delta(value, digits: int = 3) -> str:
    return "NA" if value is None or pd.isna(value) else f"{value:+.{digits}f}"


def _percent(value) -> str:
    return "NA" if value is None or pd.isna(value) else f"{value:.0%}"


def _label(arm: str) -> str:
    return ARMS.get(arm, arm)


def _cohort(means: pd.DataFrame, backend: str, arm: str) -> dict:
    if means.empty:
        return {}
    found = means[(means["backend"] == backend) & (means["arm"] == arm)]
    return found.iloc[0].to_dict() if len(found) else {}


def _matched_batches(rounds: pd.DataFrame) -> pd.DataFrame:
    metrics = ("batch_best", "batch_mean")
    base = rounds.loc[rounds["arm"] == "B_model_only", ["seed", "round", *metrics]]
    comparisons = rounds.merge(base, on=["seed", "round"], suffixes=("", "_rf"),
                               validate="many_to_one", how="left")
    if comparisons[[f"{metric}_rf" for metric in metrics]].isna().any().any():
        raise ValueError("A GFP batch is missing its matched seed and round Model-Only result")
    for metric in metrics:
        comparisons[f"{metric}_delta"] = comparisons[metric] - comparisons[f"{metric}_rf"]
    return comparisons.groupby(["arm", "round"])[
        [*metrics, "batch_best_delta", "batch_mean_delta"]
    ].mean()


def _batch_figure(batches: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#B0BEC5",
    })
    figure, axes = plt.subplots(1, 2, figsize=(7.75, 2.2), layout="constrained")
    handles = [
        Line2D([0], [0], color=COLORS[arm], marker=MARKERS[arm], linestyle=LINE_STYLES[arm], label=name)
        for arm, name in ARMS.items()
    ]
    for axis, metric, title in zip(axes, ("batch_best", "batch_mean"),
                                   ("Best new measurement", "Mean of five new measurements"), strict=True):
        for arm in ARMS:
            part = batches.loc[arm]
            axis.plot(part.index, part[metric], color=COLORS[arm], linewidth=1.4,
                      marker=MARKERS[arm], markersize=4, linestyle=LINE_STYLES[arm])
        axis.grid(axis="y", color="#E6ECF0", linewidth=0.7)
        axis.set(title=title, xticks=[1, 2, 3], xlabel="Round")
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
    axes[0].set_ylabel("Measured fluorescence (log10)")
    figure.legend(handles=handles, loc="outside upper center", frameon=False, ncol=4, fontsize=8)
    figure.savefig(output, dpi=190)
    plt.close(figure)


def _observed_counts(source: dict) -> str:
    return (
        f"{source['assay_rows']:,} public mutant records, "
        f"{source['eligible_measured_variants_excluding_wild_type']:,} at one to four substitutions"
    )


def _page_benchmark(book: Manuscript, root: Path, source: dict, summary: pd.DataFrame) -> None:
    benchmark = _frame(root / "benchmark/metrics.csv")
    if benchmark.empty:
        raise ValueError("The GFP report requires archived measured prediction metrics")
    test = benchmark[benchmark["split"] == "test"]
    fold_means = test.groupby("model").mean(numeric_only=True)
    configs = _json(root / "formal/deepseek/config.json")
    split = configs.get("split_counts", {})
    primary = _cohort(summary, "deepseek", "C_llm")
    greedy = _cohort(summary, "deepseek", "B_model_only")
    headline = "The main contrast compares fluorescence discoveries with a fixed measured-query budget."
    if primary.get("seeds") == greedy.get("seeds") == 3:
        headline = (
            f"Across three matched initial states, mean final incumbent fluorescence was "
            f"{_number(primary['mean_best_so_far'], 4)} for the plain DeepSeek agent and "
            f"{_number(greedy['mean_best_so_far'], 4)} for direct RF recommendation. "
            "High starting incumbents left few observed improvements under the 15-query budget. "
            "Newly measured batch scores expose differences hidden by the terminal endpoint."
        )
    book.para("RA technical assessment | November 2026", "deck")
    book.para("Scientific agent for GFP<br/>fluorescence evolution", "headline")
    book.para("Controlled replay of archived Aequorea victoria GFP measurements", "deck")
    book.para(
        f"{headline} The program links reconstructed GFP variants, a trained predictor, native tool calls "
        "and measured feedback. This assessment tests isolated GFP fluorescence in one public archive."
    )
    book.para("Reference and measured phenotype", "section")
    book.para(
        f"The {escape(source['release'])} release reports {_observed_counts(source)}. The pinned "
        f"reference has {len(source['wild_type_sequence'])} residues. "
        f"The source labels are {escape(source['score_scale'])}, with larger scores indicating brighter GFP. "
        f"WT = {_number(source['wild_type_measurement'], 7)}, from the original study's measured WT aggregate. "
        "Substitutions were parsed and reconstructed against the 1-indexed reference. Every official variant "
        "reconstructed to its published full sequence. Source and file hashes are archived [1-3]."
    )
    wt = source["wild_type_sequence"]
    lines = "<br/>".join(wt[index:index + 54] for index in range(0, len(wt), 54))
    book.para(f"<font name='Courier' size='7.4'>{lines}</font>", "note")
    book.para("Independent prediction holdout", "section")
    book.para(
        f"Each seed has {split.get('initial', 96)} initial observations including WT, "
        f"{split.get('validation', 120)} validation records, {split.get('pool', 384)} searchable measurements "
        f"and {split.get('test', 120)} held-out prediction records. All roles are disjoint within seed. "
        "The baseline prediction table uses the shared initial measurements only. The predictors "
        "encode edit count, positions and exact alleles."
    )
    which_k = int(test.iloc[0]["top_k"])
    columns = ("spearman", "pearson", "mse", "top_k_precision", "top_k_recall", "ap_high", "ap_fluorescent")
    book.table(
        ["Model", "Spearman", "Pearson", "MSE", f"P@{which_k}", f"R@{which_k}", "AP>WT", "AP class"],
        [["Ridge" if name == "ridge" else "Random Forest",
          *(_number(row[column]) for column in columns)] for name, row in fold_means.iterrows()],
        [106, 65, 57, 55, 52, 52, 51, 55],
    )
    book.para(
        f"<b>Table 1.</b> Seed means across {int(test['seed'].nunique())} sealed test folds. "
        f"P@{which_k}, R@{which_k} and AP&gt;WT use measured fluorescence above WT as the positive class. "
        f"AP class uses the archive threshold {source['fluorescent_threshold']}. "
        "Source: evidence/gfp/benchmark/metrics.csv.", "note",
    )


def _page_experiment(book: Manuscript, root: Path, source: dict, figures: Path) -> None:
    primary = root / "formal/deepseek"
    config = _json(primary / "config.json")
    state = _json(primary / "status.json")
    rounds = _frame(primary / "rounds.csv")
    means = _frame(primary / "method_summary.csv")
    budget = config.get("budget", 5)
    n_rounds = config.get("rounds", 3)

    book.page()
    book.para("Measured acquisition under equal budgets", "headline")
    book.para(
        f"Three paired seed blocks use {n_rounds} rounds of {budget} measurements per arm. Random samples uniformly "
        "from the current pool. Model-Only takes the highest Random Forest scores. Two DeepSeek arms share "
        "the current predictor, scoring tool, schema and critic. Knowledge ON adds amino-acid property changes "
        "and summaries of that arm's already measured results."
    )
    book.para(
        "The native tool reconstructs each substitution design against the reference and scores it. "
        "The menu holds 24 RF leaders and 24 seeded lower-ranked variants. The virtual oracle reveals "
        "the selected batch after validation, then updates that arm alone."
    )
    if not rounds.empty:
        initial = rounds[["seed", "initial_best"]].drop_duplicates().sort_values("seed")
        entries = ", ".join(f"{row.seed} at {_number(row.initial_best, 4)}" for row in initial.itertuples())
        improved = rounds[rounds["batch_best"] > rounds["initial_best"]]
        observation = ""
        if len(improved) == 1:
            row = improved.iloc[0]
            observation = (f" The exception was {_label(row['arm'])}, seed {row['seed']}, round {row['round']}, "
                           f"at {_number(row['batch_best'], 4)}.")
        book.para(
            f"The best initial observations were already high, with seed {entries}. "
            f"Only {len(improved)}/{len(rounds)} core batches exceeded their starting incumbent." + observation
        )
        batches = _matched_batches(rounds)
        plot = figures / "gfp-measured-batches.png"
        _batch_figure(batches, plot)
        book.story.append(Image(str(plot), width=486, height=137))
        book.story.append(Spacer(1, 4))
        book.para(
            "<b>Figure 1.</b> Three-seed means for each round of new measurements, with different vertical scales. "
            "Source: evidence/gfp/formal/deepseek/rounds.csv.", "note",
        )
        batch_rows = []
        for arm in ARMS:
            part = batches.loc[arm]
            batch_rows.append([_label(arm), *(
                f"{_number(row.batch_best)} / {_number(row.batch_mean)}" for row in part.itertuples()
            ), f"{_delta(part['batch_best_delta'].mean(), 4)} / {_delta(part['batch_mean_delta'].mean(), 4)}"])
        book.table(["Policy", "R1 best/mean", "R2 best/mean", "R3 best/mean", "vs RF best/mean"],
                   batch_rows, [105, 91, 91, 91, 115])
        book.para(
            "<b>Table 2.</b> Each cell averages the per-batch best and mean across matched seeds. "
            "The final column averages paired differences from Model-Only at the same seed and round. "
            "These contrasts are descriptive.", "note",
        )
    rows = []
    for arm in ARMS:
        row = _cohort(means, "deepseek", arm)
        if row:
            rows.append([
                _label(arm), f"{int(row['measurements'])}", _number(row["mean_best_so_far"], 4),
                _delta(row["mean_gain_vs_initial_best"], 4),
                f"{int(row['above_wild_type_hits'])}/{int(row['measurements'])}",
                _percent(row["mean_mean_greedy_overlap_at_k"]),
            ])
        else:
            rows.append([_label(arm), "NA", "NA", "NA", "NA", "NA"])
    book.table(["Policy", "Queries", "Final best", "Gain", "Hits > WT", "Own RF overlap"], rows,
               [134, 68, 76, 67, 67, 81])
    book.para(
        f"<b>Table 3.</b> Final-best and gain values are seed means on the log10 score scale. "
        f"Hits are distinct queries above WT ({_number(source['wild_type_measurement'], 4)}) within their arms. "
        f"Completed histories {_number(state.get('arms'), 0)}/{_number(state.get('expected_arms'), 0)}. "
        "Source: evidence/gfp/formal/deepseek/method_summary.csv.", "note",
    )
    if not rounds.empty:
        knowledge = batches.loc["D_knowledge_llm"]
        book.para(
            f"Knowledge-LLM found a stronger best per batch than Model-Only on average "
            f"({_delta(knowledge['batch_best_delta'].mean(), 4)}), while its five-candidate mean was lower "
            f"({_delta(knowledge['batch_mean_delta'].mean(), 4)}). The initial incumbent held in every "
            "Knowledge-LLM history.", "note",
        )


def _page_factorial(book: Manuscript, root: Path, summary: pd.DataFrame) -> None:
    book.page()
    book.para("Model and knowledge contrasts", "headline")
    book.para(
        "Each provider has the same two knowledge conditions and paired initial states. Tool calls use "
        "the same GFP design contract. Provider and decoding behavior are part of the tested policy. "
        "DeepSeek, Claude Sonnet and GPT use remote APIs. Qwen runs locally through MLX."
    )
    rows = []
    for backend, name in BACKENDS.items():
        for arm, condition in (("C_llm", "OFF"), ("D_knowledge_llm", "ON")):
            row = _cohort(summary, backend, arm)
            rows.append([
                f"{name} {condition}", str(int(row["seeds"])) if row else "NA",
                _number(row.get("mean_best_so_far"), 4), _delta(row.get("mean_gain_vs_initial_best"), 4),
                f"{int(row['above_wild_type_hits'])}/{int(row['measurements'])}" if row else "NA",
                _percent(row.get("mean_mean_greedy_overlap_at_k")),
            ])
    book.table(["Provider / knowledge", "Seeds", "Final best", "Gain", "Hits > WT", "Own RF overlap"], rows,
               [155, 48, 72, 67, 67, 84])
    book.para(
        "<b>Table 4.</b> Completed histories only. The two direct controls are shared across the "
        "provider comparison. Mean best and gain are in log10 assay units. "
        "Source: evidence/gfp/aggregate/method_summary.csv.", "note",
    )

    pairwise = _frame(root / "aggregate/paired_effects.csv")
    comparisons = []
    for backend, name in BACKENDS.items():
        subset = pairwise[
            (pairwise["backend"] == backend) &
            (pairwise["baseline_arm"] == "C_llm") &
            (pairwise["treatment_arm"] == "D_knowledge_llm")
        ].sort_values("seed") if not pairwise.empty else pd.DataFrame()
        effects = {row.seed: row.delta_best for row in subset.itertuples()}
        comparisons.append([name, *(_delta(effects.get(seed), 4) for seed in (17, 29, 43)),
                            _delta(mean(effects.values()), 4) if effects else "NA"])
    book.para("Matched change from adding knowledge", "section")
    book.table(["Provider", "Seed 17", "Seed 29", "Seed 43", "Mean"], comparisons, [161, 83, 83, 83, 83])
    book.para(
        "<b>Table 5.</b> Each entry is Knowledge ON minus OFF in final measured best. A positive "
        "difference favors knowledge on the tested seed. Source: evidence/gfp/aggregate/paired_effects.csv.",
        "note",
    )
    chosen = _frame(root / "aggregate/selections.csv")
    if not chosen.empty:
        llm = chosen[chosen["arm"].isin(("C_llm", "D_knowledge_llm"))]
        if len(llm):
            outside = int((llm["menu_source"] == "seeded_non_elite").sum())
            book.para(
                f"LLM histories queried {outside}/{len(llm)} candidates from the seeded lower-ranked menu. "
                "These were genuine departures from the 24 leaders. The seed-level differences are "
                "descriptive for this archive and budget, with three matched initial states per provider. "
                "Possible model pretraining exposure to the public archive was unmeasured."
            )
    overlap = _frame(root / "aggregate/knowledge_selection_overlap.csv")
    if not overlap.empty:
        first = overlap[overlap["round"] == 1]
        book.para(
            f"At the shared initial state, knowledge changed the chosen set in "
            f"{int(first['set_changed'].sum())}/{len(first)} matched comparisons. "
            f"The mean overlap was {first['overlap_at_five'].mean():.0%}. All knowledge-enabled arms retained "
            "their starting incumbents. The changed choices establish a policy difference under this menu."
        )
    landscape = _json(root / "aggregate/landscape_summary.json")
    if landscape.get("combinations_with_all_components_measured"):
        positions = ", ".join(str(row["position"]) for row in landscape["most_frequent_selected_positions"][:3])
        book.para(
            f"In a descriptive archive check, {landscape['combinations_with_all_components_measured']:,} "
            "combinations had measurements for all constituent single substitutions. The median absolute "
            f"departure from log-additivity was {_number(landscape['median_absolute_deviation'])}. "
            f"The most frequent positions among distinct LLM selections were {positions}. "
            "Frequencies describe selections, with no inference of per-site causal effect.",
            "note",
        )
    status = _json(root / "aggregate/status.json")
    incomplete = [key for key, item in status.get("providers", {}).items() if item["status"] != "complete"]
    if incomplete:
        book.para(
            "Incomplete formal cells remain separate. See failures.json in the affected run directories "
            f"for the retained traces ({', '.join(map(escape, incomplete))}).", "note",
        )


def _page_audit(book: Manuscript, root: Path, audit: dict) -> None:
    book.page()
    book.para("Feedback audit and measured examples", "headline")
    trace = audit.get("trajectory", {})
    if trace:
        book.para(
            f"The trajectory audit reconstructed {trace['decisions']} logged rounds. In "
            f"{trace['feedback_cited_in_next']}/{trace['transitions']} transitions the next decision "
            "cited a newly revealed measurement. Free-text hypotheses without locked success criteria "
            "were logged as not evaluable. Citation counts quantify traceability."
        )
    else:
        book.para("No completed trajectory-audit output is available for this archive.")
    book.para(
        "The paired perturbation fixes the RF weights, eligible candidates, menu, provider and budget. "
        "The only information intervention is showing or withholding the measured previous batch. "
        "Both conditions exclude the previously selected candidates. Archived scores are revealed after each "
        "condition's batch is accepted."
    )
    paired = _frame(root / "audit/paired_feedback_results.csv")
    result = audit.get("paired", {})
    if audit.get("paired_status") == "complete" and not paired.empty:
        book.table(
            ["Checkpoint", "No feedback", "With feedback", "Model-Only", "Overlap"],
            [[f"{row.seed} / {_label(row.arm)}", _number(row.no_feedback_best, 4),
              _number(row.with_feedback_best, 4), _number(row.model_only_best, 4),
              _percent(row.top_k_overlap)] for row in paired.itertuples()],
            [165, 80, 83, 84, 81],
        )
        book.para(
            "<b>Table 6.</b> Best measured score in each matched five-candidate batch. Overlap is "
            "the fraction of IDs shared by the two feedback conditions. "
            "Source: evidence/gfp/audit/paired_feedback_results.csv.", "note",
        )
        message = (
            f"Feedback changed {result['sets_changed']}/{result['pairs']} sets, with mean overlap "
            f"{_percent(result['mean_overlap'])}. Best-measured score changed by "
            f"{_delta(result['mean_best_delta'], 4)} on average and batch mean by "
            f"{_delta(result['mean_batch_delta'], 4)}. With Feedback led Model-Only by "
            f"{_delta(result['mean_best_vs_greedy'], 4)} on the mean of paired bests. "
            f"The best score improved at {int((paired['best_delta'] > 0).sum())} of {len(paired)} checkpoints. "
            "Each checkpoint has one draw per condition."
        )
        book.para(message)
    else:
        status = escape(str(audit.get("paired_status", "not archived")))
        reason = escape(str(audit.get("paired_reason", "")))
        book.para(f"The paired audit is {status}. {reason}", "note")

    chosen = _frame(root / "aggregate/selections.csv")
    if not chosen.empty:
        llm = chosen[chosen["arm"].isin(("C_llm", "D_knowledge_llm"))]
        winners = llm.sort_values(["fitness", "prediction"], ascending=[False, False]).drop_duplicates("id").head(3)
        if not winners.empty:
            book.para("GFP candidates measured during replay", "section")
            book.table(
                ["Substitutions", "Decision", "Prediction", "Measured"],
                [[row.mutations, f"{BACKENDS[row.backend]} / {row.seed} / R{row.round}",
                  _number(row.prediction, 4), _number(row.fitness, 4)] for row in winners.itertuples()],
                [177, 158, 81, 77],
            )
            book.para(
                "<b>Table 7.</b> Top distinct LLM-selected variants by measured fluorescence. The matched full "
                "GFP sequences, all rounds, forecaster estimates and revealed scores are in "
                "evidence/gfp/aggregate/selections.csv. Agent traces retain rationales and evidence IDs.", "note",
            )


def _logs(root: Path, backend: str) -> list[dict]:
    return [_json(trace) for trace in sorted((root / f"formal/{backend}/arms").rglob("*-agent.json"))]


def _page_reproduction(book: Manuscript, root: Path) -> None:
    book.page()
    book.para("Reproduction and laboratory handoff", "headline")
    workflow = "user-api-demo" if (root / "user-api-demo/status.json").exists() else "user-demo"
    user = _json(root / workflow / "status.json")
    if user.get("status") == "complete":
        book.para(
            f"The documented user workflow loads {user['initial_observations']} published measurements as "
            f"a researcher CSV and imports {user['imported_after_selection']} returned assay values. "
            f"The state grows to {user['total_observations']} observed entries. It then exports "
            f"{user['second_batch_prediction_only']} predicted GFP candidates with the measured fields empty. "
            f"The full sequences, CSV inputs, recommendations and state are in evidence/gfp/{workflow}/."
        )
    book.para(
        "A laboratory integration can submit a batch ID and sequence checksum to a review queue, track "
        "replicates and return the assessed fluorescence value after review. The import step validates the "
        "GFP reference, canonical edits, uniqueness and the measured field before updating the predictor."
    )
    examples = _frame(root / "aggregate/failure_examples.csv")
    if not examples.empty:
        extrema = examples.loc[[examples["residual"].idxmin(), examples["residual"].idxmax()]]
        book.para("Forecast failures among measured selections", "section")
        book.table(
            ["Variant / decision", "Predicted", "Measured", "Residual"],
            [[f"{row.mutations} / {BACKENDS[row.backend]} {row.seed} R{row.round}",
              _number(row.prediction, 4), _number(row.fitness, 4), _delta(row.residual, 4)]
             for row in extrema.itertuples()], [274, 73, 73, 73],
        )
        book.para(
            "<b>Table 8.</b> Residual is measured minus predicted. Both extremes are from Knowledge-LLM "
            "batches. Source: evidence/gfp/aggregate/failure_examples.csv.", "note",
        )
    book.para("Executable and model receipts", "section")
    book.para(
        "<font name='Bold'>python run.py setup</font> tests a configured endpoint's native tools. "
        "<font name='Bold'>python run.py demo --dataset gfp --model-profile default</font> runs the CSV workflow. "
        "<font name='Bold'>python run.py gfp replay --from-run evidence/gfp --output runs/gfp-replay</font> "
        "reconstructs frozen research results. The repository README documents the live controls, native "
        "provider settings and paired audit. The assay archive and checksums are in gfp-source.md."
    )
    usage = []
    for backend, name in BACKENDS.items():
        traces = _logs(root, backend)
        if traces:
            ids = sorted({
                str(call["response"].get("model") or log.get("model"))
                for log in traces for call in log.get("calls", []) if "response" in call
            })
            valid = sum(log.get("status") == "ok" for log in traces)
            generations = sum(log.get("live_generation_count", 0) for log in traces)
            usage.append([name, ", ".join(ids), f"{valid}/{len(traces)}", str(generations)])
    if usage:
        book.table(["Provider", "Returned model identifier", "Valid decisions", "Live outputs"], usage,
                   [88, 244, 86, 75])
        book.para(
            "<b>Table 9.</b> Formal response accounting from each backend's native traces, which "
            "retain request hashes, tool outputs and token usage. Graphics and report drafts used "
            "Codex; the four named models generated the corresponding logged proposals. "
            "All measurements come from public GFP archives. No wet-lab experiment was performed.",
            "note",
        )
    _primary_sources(book)


def _primary_sources(book: Manuscript) -> None:
    book.para("Primary sources", "section")
    for entry in (
        '[1] Sarkisyan et al. (2016), <i>Local fitness landscape of the green fluorescent protein</i>. '
        '<link href="https://doi.org/10.1038/nature17995">Nature, doi:10.1038/nature17995</link>.',
        '[2] <link href="https://doi.org/10.6084/m9.figshare.3102154.v1">Original GFP amino-acid level '
        'measurements, Figshare v1</link>. Data license CC BY 4.0.',
        '[3] <link href="https://doi.org/10.5281/zenodo.15293562">ProteinGym substitutions v1.3</link>. '
        'Assay GFP_AEQVI_Sarkisyan_2016.',
    ):
        book.para(entry, "note")


def _quality_headroom(book: Manuscript, root: Path, summary: pd.DataFrame) -> None:
    diagnostic = _json(root / "quality/diagnosis/summary.json")
    rounds = pd.read_csv(root / "formal/deepseek/rounds.csv")
    book.page()
    book.para("Three rounds from the starting best", "headline")
    book.para(
        f"The original study completed {diagnostic['queries']} measured archive queries. Within each seed, "
        "all methods started with the same measurements and queried five variants per round. Model-Only "
        "took the top five RF scores. The LLM menu contained 24 RF-ranked and 24 seeded alternatives. "
        "Pool headroom was calculated after the formal run."
    )
    book.table(
        ["Seed", "Initial best", "Pool best", "> Initial", "> WT", "Random hit in 15"],
        [[str(row["seed"]), _number(row["initial_best"], 4), _number(row["pool_best"], 4),
          f"{row['above_initial_count']}/{row['pool_count']}", str(row["above_wt_count"]),
          f"{row['random_hit_probability']:.2%}"] for row in diagnostic["seed_headroom"]],
        [61, 89, 85, 80, 64, 114],
    )
    book.para(
        "<b>Table 2.</b> Random hit probability is exact for uniform sampling without replacement. "
        "The three original candidate pools share some IDs across seeds. "
        "Source: evidence/gfp/quality/diagnosis/seed_headroom.csv.", "note",
    )
    book.para(
        f"One of {diagnostic['arms']} arms improved its starting best. No improving variant appeared in an "
        "RF top-five or RF-elite list along the logged decisions. Seeded alternatives exposed an eligible "
        f"improvement to {diagnostic['llm_arms_exposed_to_improvement']}/{diagnostic['llm_arms']} LLM arms, "
        "and none selected one. These opportunities depend on each arm's past decisions."
    )
    book.story.append(Image(_incumbent_trajectories(rounds), width=486, height=154))
    book.story.append(Spacer(1, 4))
    book.para(
        "<b>Figure 1.</b> Original four-policy trajectories of the best measured gain from baseline. "
        "The 5, 10 and 15 measurements are the ends of rounds 1, 2 and 3. Overlapping curves show ties. "
        "Source: evidence/gfp/formal/deepseek/rounds.csv.", "note",
    )
    book.para("What the newly queried measurements showed", "section")
    batches = _matched_batches(rounds)
    rows = []
    for arm, label in ARMS.items():
        part = batches.loc[arm]
        method = _cohort(summary, "deepseek", arm)
        rows.append([label, *(
            f"{_number(row.batch_best)} / {_number(row.batch_mean)}" for row in part.itertuples()
        ), f"{_delta(part['batch_best_delta'].mean())} / {_delta(part['batch_mean_delta'].mean())}",
                     _delta(method["mean_gain_vs_initial_best"], 4)])
    book.table(["Policy", "R1 best/mean", "R2 best/mean", "R3 best/mean", "vs RF best/mean", "Final gain"],
               rows, [95, 80, 80, 80, 96, 62])
    book.para(
        "<b>Table 3.</b> New measured batch best and mean, averaged over three seeds. The RF contrast matches "
        "seed and round. Final gain compares the final incumbent with the initial best. "
        "Source: evidence/gfp/formal/deepseek/rounds.csv.", "note",
    )
    book.para(
        "The scarce above-incumbent candidates limited separation on final gain. The DeepSeek knowledge "
        "arm had a higher average batch best and a lower average batch mean than Model-Only. These are "
        "distinct, descriptive findings under the archived budget."
    )


def _incumbent_trajectories(rounds: pd.DataFrame) -> BytesIO:
    image = BytesIO()
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#B0BEC5",
    }):
        figure, axes = plt.subplots(1, 3, figsize=(7.75, 2.45), sharey=True, layout="constrained")
        handles = [
            Line2D([0], [0], color=COLORS[arm], marker=MARKERS[arm], linestyle=LINE_STYLES[arm], label=name)
            for arm, name in ARMS.items()
        ]
        for seed, axis in zip((17, 29, 43), axes, strict=True):
            seed_rows = rounds[rounds["seed"] == seed]
            for arm in ARMS:
                part = seed_rows[seed_rows["arm"] == arm].sort_values("round")
                if not part.empty:
                    axis.step([0, *part["n_revealed_total"]], [0, *part["gain_vs_initial_best"]],
                              where="post", color=COLORS[arm], alpha=0.91, linewidth=1.3,
                              marker=MARKERS[arm], markersize=3.4, linestyle=LINE_STYLES[arm])
            axis.grid(axis="y", color="#E6ECF0", linewidth=0.7)
            axis.set(title=f"Seed {seed}", xticks=[0, 5, 10, 15], xlabel="Measured variants")
            axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
        axes[0].set_ylabel("Incumbent gain (log10 score)")
        figure.legend(handles=handles, loc="outside upper center", frameon=False, ncol=4, fontsize=8)
        figure.savefig(image, format="png", dpi=190)
        plt.close(figure)
    image.seek(0)
    return image


def _quality_validation(book: Manuscript, root: Path, summary: pd.DataFrame) -> None:
    validation = root / "quality/model_validation"
    diagnostics = _json(validation / "summary.json")
    means = pd.read_csv(validation / "validation_means.csv")
    ranks = means.set_index(["model", "criterion", "top_k"]).sort_index()

    book.page()
    book.para("Where the predictor and the policy differ", "headline")
    book.para(
        f"Post hoc validation refit the two existing estimators on each original initial set. Scoring used "
        f"{diagnostics['validation_rows']} validation rows ({diagnostics['unique_validation_variants']} "
        "unique variants). The acquisition pools and final holdouts provided no labels for this analysis."
    )
    rows = []
    for model, label in (("rf", "Random Forest"), ("ridge", "Ridge")):
        wt = ranks.loc[model, "above_wild_type"]
        decile = ranks.loc[model, "upper_decile"]
        rows.append([label, *(_number(wt.loc[5, metric]) for metric in ("spearman", "pearson", "mse")),
                     *[f"{_number(values.loc[5, 'top_k_precision'])} / {_number(values.loc[10, 'top_k_precision'])}"
                       for values in (wt, decile)], _number(decile.loc[5, "ap_high"])])
    book.table(["Predictor", "Spearman", "Pearson", "MSE", ">WT P@5/10", "Top 10% P@5/10", "Top 10% AP"],
               rows, [92, 62, 57, 53, 74, 81, 74])
    book.para(
        "<b>Table 4.</b> Precision at K uses uniform allocation of tied predictions at the selection cutoff. "
        "AP is average precision. Source: evidence/gfp/quality/model_validation/validation_means.csv.", "note",
    )
    spread = diagnostics["rf_spread"]
    low_mae = mean(next(row["mae"] for row in fold["risk_at_coverage"] if row["retained_fraction"] == 0.25)
                   for fold in spread)
    full_mae = mean(fold["mae"] for fold in spread)
    coverage = mean(fold["fraction_abs_error_within_1_96_spreads"] for fold in spread)
    book.para(
        f"Ridge ranked better globally and at P@5. RF ranked better at P@10 for both thresholds. For RF, "
        f"the lowest-disagreement quarter had MAE {_number(low_mae)} versus {_number(full_mae)} overall. "
        f"Mean prediction ±1.96 tree SD covered {coverage:.1%}, so the interval has no demonstrated "
        "95% calibration. This screen measures predictive error, leaving exploration value untested."
    )
    book.para("What changed when knowledge was enabled", "section")
    rows = []
    for backend, name in BACKENDS.items():
        off = _cohort(summary, backend, "C_llm")
        on = _cohort(summary, backend, "D_knowledge_llm")
        rows.append([name, f"{int(off['above_wild_type_hits'])}/{int(off['measurements'])}",
                     f"{int(on['above_wild_type_hits'])}/{int(on['measurements'])}",
                     _delta(on["mean_best_so_far"] - off["mean_best_so_far"], 4)])
    book.table(["Backend", "Hits > WT, OFF", "Hits > WT, ON", "Final best ON - OFF"],
               rows, [135, 108, 108, 142])
    overlap = pd.read_csv(root / "aggregate/knowledge_selection_overlap.csv")
    first = overlap[overlap["round"] == 1]
    book.para(
        f"<b>Table 5.</b> First-round top-five membership changed in {int(first['set_changed'].sum())}/{len(first)} "
        "same-menu comparisons. Terminal best stayed unchanged across all four models. "
        "Sources: evidence/gfp/aggregate/method_summary.csv and knowledge_selection_overlap.csv.", "note",
    )
    failures = pd.read_csv(root / "aggregate/failure_examples.csv")
    over, under = (failures.loc[index] for index in (failures["residual"].idxmin(), failures["residual"].idxmax()))
    book.para(
        f"Among measured recommendations, <b>{escape(over['mutations'])}</b> scored "
        f"{_number(over['fitness'], 4)} after a {_number(over['prediction'], 4)} forecast. "
        f"<b>{escape(under['mutations'])}</b> scored {_number(under['fitness'], 4)} after a "
        f"{_number(under['prediction'], 4)} forecast. Source: evidence/gfp/aggregate/failure_examples.csv.", "note",
    )


def _quality_contrasts_figure(paired: pd.DataFrame, contrasts: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.5, 2.1), layout="constrained")
    styles = {"ridge": "Ridge", "rf": "RF", "random": "Random"}
    labels = []
    for index, row in enumerate(contrasts.itertuples()):
        labels.append(f"{styles[row.policy]} - {styles[row.control]}")
        points = paired[(paired["policy"] == row.policy) & (paired["control"] == row.control)].sort_values("block")
        positions = [index + 0.13 + (offset % 3 - 1) * 0.09 for offset in range(len(points))]
        axis.scatter(points["new_best"], positions, s=13, color="#78949E", alpha=0.78, zorder=3)
        axis.hlines(index - 0.18, row.new_best_ci_low, row.new_best_ci_high, color="#175B6D", linewidth=2)
        axis.plot(row.new_best, index - 0.18, marker="D", color="#175B6D", markersize=4)
    axis.axvline(0, color="#A34C36", linewidth=0.9, linestyle="--")
    axis.set(yticks=range(len(labels)), yticklabels=labels,
             xlabel="Difference in best new measured fluorescence (log10)")
    axis.invert_yaxis()
    axis.grid(axis="x", color="#E6ECF0", linewidth=0.7)
    axis.spines[["right", "top"]].set_visible(False)
    axis.tick_params(axis="both", labelsize=8)
    figure.savefig(output, dpi=200)
    plt.close(figure)


def _quality_acquisition(book: Manuscript, root: Path, figures: Path) -> None:
    folder = root / "quality/acquisition"
    status, config = (_json(folder / name) for name in ("status.json", "config.json"))
    if status.get("status") != "complete" or status.get("api_calls") != 0:
        raise ValueError("The quality report requires the completed zero-API supplemental acquisition")
    methods = pd.read_csv(folder / "methods.csv")
    contrasts = pd.read_csv(folder / "contrasts.csv")
    book.page()
    book.para("Prospective predictor acquisition comparison", "headline")
    book.para(
        f"A separate protocol fixed {status['paired_blocks']} paired blocks before execution. Except shared "
        "WT, variants are disjoint across blocks and from the original study splits. Blocks were "
        f"assigned without fluorescence labels. Each policy began with {config['initial_size']} observations "
        f"and had {config['rounds']} rounds of {config['budget']} measured queries from a {config['pool_size']}-item "
        f"pool. All {status['queries']} queries were completed with zero API calls. Estimator parameters stayed fixed."
    )
    means = methods.groupby("policy").mean(numeric_only=True)
    book.table(
        ["Policy", "Best new", "Mean new", "Hits > WT", "Incumbent gain"],
        [[label, *(_number(means.loc[key, metric], 4) for metric in ("new_best", "new_mean")),
          _number(means.loc[key, "above_wt"], 2), _delta(means.loc[key, "incumbent_gain"], 4)]
         for key, label in (("random", "Random"), ("rf", "Random Forest"), ("ridge", "Ridge"))],
        [133, 91, 91, 84, 94],
    )
    book.para(
        "<b>Table 6.</b> Means across blocks. The primary outcome is the best of 15 <i>new</i> measured "
        "variants. Incumbent gain is shown separately. Source: evidence/gfp/quality/acquisition/methods.csv.", "note",
    )
    plot = figures / "gfp-acquisition-contrasts.png"
    _quality_contrasts_figure(pd.read_csv(folder / "paired.csv"), contrasts, plot)
    book.story.append(Image(str(plot), width=486, height=136))
    book.story.append(Spacer(1, 4))
    book.para(
        "<b>Figure 2.</b> Dots are paired blocks. Diamonds and lines show the mean difference and descriptive "
        "95% percentile intervals from 10,000 block resamples. "
        "Source: evidence/gfp/quality/acquisition/paired.csv.", "note",
    )
    labels = {"ridge": "Ridge", "rf": "RF", "random": "Random"}
    book.table(
        ["Paired contrast", "Best new difference [95% interval]", "Win/loss/tie", "Mean new diff."],
        [[f"{labels[row.policy]} - {labels[row.control]}",
          f"{_delta(row.new_best, 4)} [{_delta(row.new_best_ci_low, 4)}, {_delta(row.new_best_ci_high, 4)}]",
          f"{row.new_best_wins}/{row.new_best_losses}/{row.new_best_ties}", _delta(row.new_mean, 4)]
         for row in contrasts.itertuples()], [120, 180, 95, 98],
    )
    book.para(
        "<b>Table 7.</b> Contrasts are within blocks. Intervals quantify variation over the sampled blocks in "
        "one assay. Freeze: gfp_quality_protocol.md. Source: evidence/gfp/quality/acquisition/contrasts.csv.", "note",
    )
    book.para(
        "Both predictive policies improved the mean new score versus Random in these blocks. The primary "
        "best-new intervals all span zero. The study measures predictive ranking policies and contains no LLM arm."
    )


def _quality_audit(book: Manuscript, root: Path, audit: dict) -> None:
    book.page()
    book.para("Feedback, hypothesis tests and claims", "headline")
    paired = audit["paired"]
    book.para("Archived feedback perturbation", "section")
    book.para(
        f"The four fixed-condition checkpoints changed {paired['sets_changed']}/{paired['pairs']} top-five sets. "
        f"With-feedback minus no-feedback averaged {_delta(paired['mean_best_delta'], 4)} in batch best. "
        f"The with-feedback arm averaged {_delta(paired['mean_best_vs_greedy'], 4)} versus fixed Model-Only. "
        "Improvement versus the no-feedback copy occurred at one checkpoint. Each condition used one draw per checkpoint. "
        "Source: evidence/gfp/audit/paired_feedback_results.csv."
    )
    _quality_hypothesis(book, root / "quality/hypothesis", _json(root / "benchmark/source.json")["wild_type_measurement"])
    book.para("Reproduction and scope", "section")
    book.para(
        "The verified archive supplies all replay measurements. The four LLM backends made actual native "
        "tool calls in the original study. The supplemental comparison and post hoc validation used no "
        "model APIs. The project has no wet-lab results. Codex assisted with code, analysis and this report."
    )
    book.para(
        "Render this report from archived results with "
        "<font name='Bold'>python run.py gfp report --quality --run-dir evidence/gfp "
        "--output runs/gfp-quality-report.pdf</font>. The README documents the assay checksum, single-provider "
        "CLI, CSV imports and audits. Hypothesis records link the pre-measurement decision hash to the threshold, "
        "predictions and imported assay values.",
        "note",
    )
    _primary_sources(book)


def _quality_hypothesis(book: Manuscript, folder: Path, wild_type: float) -> None:
    book.para("Prospective test in the user workflow", "section")
    trail = folder / "audit/decision_traces.jsonl"
    traces = [json.loads(line) for line in trail.read_text(encoding="utf-8").splitlines()]
    evaluations = traces[-1]["hypothesis_evaluations"]
    finished = next(test for test in evaluations if test["status"] in ("supported", "refuted"))
    origin = next(row for row in traces if row["round"] == finished["round"])
    pending = next(test for test in origin["hypothesis_evaluations"] if test["round"] == finished["round"])
    if pending["status"] != "pending" or pending["decision_sha256"] != finished["decision_sha256"]:
        raise ValueError("The GFP quality report requires a verified pending-to-measured hypothesis trace")
    book.para(
        f"The configured model ({escape(origin['decision']['model'])}) locked a round {finished['round']} "
        f"criterion of at least {finished['min_exceed']} of {len(finished['candidate_ids'])} designs "
        f"exceeding the low fluorescence of the measured comparator {escape(finished['reference_id'])} "
        f"({_number(finished['threshold'], 4)}). The test was pending before measurement. On import, "
        f"{finished['n_exceed']}/{finished['n_measured']} exceeded that threshold. The archived status "
        f"<b>{escape(finished['status'])}</b> applies only to this predeclared fluorescence comparison."
    )
    book.table(
        ["GFP variant", "Predicted", "Measured", "Residual", "Forecast outcome"],
        [[row["id"], _number(row["predicted_fitness"], 4), _number(row["measured_fitness"], 4),
          _delta(row["residual"], 4), row["forecast_outcome"].replace("_", " ")]
         for row in finished["candidate_results"]], [172, 71, 71, 69, 110],
    )
    book.para(
        "<b>Table 8.</b> Forecast outcome compares prediction and measurement with the same locked threshold. "
        "Residual is measured minus predicted. Only the selected archive comparison was imported. "
        "Source: evidence/gfp/quality/hypothesis/audit/decision_traces.jsonl.", "note",
    )
    best_measured = max(row["measured_fitness"] for row in finished["candidate_results"])
    relation = "below" if best_measured < wild_type else "equal to" if best_measured == wild_type else "above"
    pending_rounds = ", ".join(str(row["round"]) for row in evaluations if row["status"] == "pending")
    book.para(
        f"The measured value {_number(best_measured, 4)} was {relation} wild type ({_number(wild_type, 4)}). "
        "The proposed biochemical mechanism remains untested. "
        f"The round {pending_rounds} hypothesis is pending. "
        "Feedback gain in the user workflow is <b>NOT EVALUABLE</b> without a matched counterfactual."
    )


def render(run_dir: Path, output: Path, *, quality: bool = False) -> None:
    """Create a five-page research PDF and its figure from evidence on disk."""
    source = _json(run_dir / "benchmark/source.json")
    if source.get("assay") != "GFP_AEQVI_Sarkisyan_2016":
        raise ValueError("Expected the verified GFP benchmark source manifest")
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = _frame(run_dir / "aggregate/method_summary.csv")
    if summary.empty:
        summary = _frame(run_dir / "formal/deepseek/method_summary.csv")
    audit = _json(run_dir / "audit/audit_summary.json")
    book = Manuscript()
    _page_benchmark(book, run_dir, source, summary)
    if quality:
        book.para("Agent workflow", "section")
        book.para(
            "The agent proposes explicit substitutions and calls the scoring tool, which validates and reconstructs "
            "each sequence and returns fitted predictions. The agent chooses a batch from scored variants. "
            "The replay releases archive fluorescence after commitment. The next round fits the predictor "
            "to each arm's own observations."
        )
        book.para(
            "Evidence in this edition includes the original frozen study, follow-up diagnosis and validation, "
            "and a prospectively frozen supplemental exploration of predictive acquisition policies. "
            "All assay comparisons use measured log10 fluorescence.", "note",
        )
        _quality_headroom(book, run_dir, summary)
        _quality_validation(book, run_dir, summary)
        _quality_acquisition(book, run_dir, output.parent)
        _quality_audit(book, run_dir, audit)
    else:
        _page_experiment(book, run_dir, source, output.parent)
        _page_factorial(book, run_dir, summary)
        _page_audit(book, run_dir, audit)
        _page_reproduction(book, run_dir)

    def footer(canvas, document):
        canvas.setStrokeColor(book.gray)
        canvas.line(51, 42, 544, 42)
        canvas.setFont("Sans", 8)
        canvas.setFillColor(book.gray)
        canvas.drawString(51, 29, "GFP fluorescence directed evolution")
        canvas.drawRightString(544, 29, str(document.page))

    doc = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=51, rightMargin=51, topMargin=48, bottomMargin=54,
                            title="Scientific agent for GFP fluorescence evolution",
                            author="RA technical assessment")
    doc.build(book.story, onFirstPage=footer, onLaterPages=footer)
