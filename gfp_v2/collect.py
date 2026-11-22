"""Collect V2 report tables and acceptance gates from the committed evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


LABELS = {"A_random": "A Random", "B_rf": "B Random Forest", "C_ridge": "C Ridge",
          "D_v2": "D Predictor V2", "B_predictor": "B Predictor V2", "C_explore": "C Predicted + risk",
          "D_agent": "D LLM", "E_knowledge": "E Knowledge-LLM", "max_non_llm": "Better of B/C"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value, digits=3):
    return f"{value:.{digits}f}" if pd.notna(value) else "NA"


def contrast(row, key):
    return f"{row[key]:+.4f} [{row[key + '_ci_low']:+.4f}, {row[key + '_ci_high']:+.4f}]"


def gate(rows):
    if all(row["low"] > 0 for row in rows):
        return "GO"
    if any(row["mean"] > 0 for row in rows):
        return "MODIFY"
    return "NO-GO"


def summarize_methods(path):
    methods = pd.read_csv(path / "methods.csv")
    table = []
    for arm, rows in methods.groupby("arm", sort=True):
        table.append([LABELS[arm], int(rows["budget_used"].sum()), fmt(rows["new_best"].mean()),
                      fmt(rows["new_mean"].mean()), int(rows["above_wt"].sum()), fmt(rows["incumbent_gain"].mean())])
    return table, pd.read_csv(path / "contrasts.csv").set_index(["treatment", "control"])


def collect(evidence: Path) -> dict:
    lock = read(evidence / "prediction/selection.json")
    test = pd.read_csv(evidence / "prediction/test_summary.csv").set_index(["budget", "model"])
    paired = pd.read_csv(evidence / "prediction/test_paired.csv").set_index(["budget", "model", "baseline", "metric"])
    predictor, delta = [], {}
    for budget, label in (("low", "96"), ("ample", "2,048")):
        chosen = lock["selected"][budget]
        baseline = lock["best_baseline"][budget]
        pair = paired.loc[(budget, chosen, baseline, "wt_ap")]
        delta[budget] = {"mean": pair.mean_difference, "low": pair.descriptive_bootstrap_low,
                         "high": pair.descriptive_bootstrap_high}
        for model in dict.fromkeys(("ridge", "rf", chosen)):
            row = test.loc[budget, model]
            predictor.append([f"{label} / {model}", *[fmt(row[k]) for k in (
                "wt_ap", "spearman", "pearson", "mse", "mae", "decile_precision_5", "decile_precision_10")]])

    acquisition, ac = summarize_methods(evidence / "acquisition")
    ac_rf, ac_ridge = (ac.loc[("D_v2", policy)] for policy in ("B_rf", "C_ridge"))
    agent, ag = summarize_methods(evidence / "agent")
    archived_rounds = pd.read_csv(evidence / "agent/rounds.csv")
    rounds = archived_rounds.groupby(["round", "arm"])["new_best"].mean().unstack()
    trajectory = [{"policy": LABELS[arm], "rounds": rounds.index.tolist(), "new_best": rounds[arm].tolist()}
                  for arm in rounds.columns]
    old = pd.read_csv(evidence.parent / "gfp/formal/deepseek/method_summary.csv")
    old_labels = {"A_random": "Random", "B_model_only": "Model-Only", "C_llm": "LLM", "D_knowledge_llm": "Knowledge-LLM"}
    v1_comparators = [[old_labels[row.arm], fmt(row.mean_best_so_far, 4),
                       fmt(row.mean_gain_vs_initial_best, 4), f"{row.above_wild_type_hits}/{row.measurements}"]
                      for row in old.itertuples()]
    block_data = pd.read_csv(evidence / "agent/paired.csv")
    pairs = [("D_agent", "max_non_llm"), ("E_knowledge", "max_non_llm"), ("E_knowledge", "D_agent")]
    figures = []
    for policy, control in pairs:
        row = ag.loc[policy, control]
        blocks = block_data.loc[(block_data.treatment == policy) & (block_data.control == control), "new_best"]
        figures.append({"label": f"{LABELS[policy]} - {LABELS[control]}", "blocks": blocks.tolist(),
                        "mean": row.new_best, "low": row.new_best_ci_low, "high": row.new_best_ci_high})

    case_folder = evidence / "agent/arms/2208/D_agent"
    measured = read(case_folder / "round-01-measurement.json")
    prior = read(case_folder / "round-02-request.json")
    next_decision = read(case_folder / "round-02-commit.json")["policy"]
    second_verdict = read(case_folder / "round-02-measurement.json")["verdicts"][0]
    verdict = measured["verdicts"][0]
    target = verdict["candidate_id"]
    feedback_received = any(row["id"] == target for row in prior["latest_feedback"])
    feedback_cited = target in next_decision["recent_evidence_used"]
    smoke = read(evidence / "verification.json")
    audit = read(evidence / "audit/summary.json")
    notes = read(evidence / "analysis_notes.json")
    g2 = [{"mean": row.new_best, "low": row.new_best_ci_low} for row in (ac_rf, ac_ridge)]
    low_ap = delta["low"]
    ample_ap = delta["ample"]
    gates = [
        ["G1", gate([low_ap, ample_ap]), f"Test WT AP change vs selected baseline: 96 {low_ap['mean']:+.3f}; 2,048 {ample_ap['mean']:+.3f}."],
        ["G2", gate(g2), f"Predictor V2 - RF best new {ac_rf.new_best:+.4f}; vs Ridge {ac_ridge.new_best:+.4f}."],
        ["G3", "GO" if feedback_received and feedback_cited else "MODIFY",
         f"Block 2208 D: {target}, {verdict['status']}; next decision cites its revealed score."],
        ["G4", gate(figures[:2]), f"D and E vs best realized non-LLM per block: {figures[0]['mean']:+.4f}, {figures[1]['mean']:+.4f}."],
        ["G5", smoke["third_party_verdict"], smoke["third_party_summary"]],
    ]
    test_risk = []
    for budget, number in (("low", "96"), ("ample", "2,048")):
        risk = test.loc[budget, lock["selected"][budget]]
        change = delta[budget]
        test_risk.append(
            f"{number} observations: WT AP change {change['mean']:+.3f} [{change['low']:+.3f}, "
            f"{change['high']:+.3f}], block bootstrap. Risk/error Spearman {fmt(risk.risk_error_spearman)}. "
            f"Observed coverage of validation-scaled 90% intervals {risk.calibrated_90_coverage:.1%}."
        )
    return {
        "overview": ["V1 established a measured three-round GFP replay, a four-model knowledge comparison, "
                     "a paired audit, and 540 supplemental queries. V2 uses a separate preregistered 96/2,048-observation "
                     "benchmark, a new acquisition ablation, and one instrumented DeepSeek workflow.", *notes["overview"]],
        "gates": gates,
        "v1": notes["v1"],
        "v1_comparators": v1_comparators,
        "prediction_design": ["The reference scaffold and fitness target were fixed before model selection. The 23,528 non-WT "
                              "genotypes are disjoint across assignment blocks and absent from the retired V1 evidence. "
                              f"Validation WT AP selected {lock['selected']['low']} at 96 observations and {lock['selected']['ample']} "
                              "at 2,048. All independent rankings are from the same public log10 fluorescence assay."],
        "prediction": predictor,
        "prediction_results": test_risk + notes["prediction"],
        "acquisition_design": ["Conventional ablation: eight paired archive blocks, equal starting data and full candidate pool, "
                               "three rounds of five queries. An oracle reveals only committed IDs and each policy refits from its own feedback."],
        "acquisition": acquisition,
        "acquisition_results": [f"V2 - RF paired new best {contrast(ac_rf, 'new_best')}. V2 - Ridge "
                                f"{contrast(ac_ridge, 'new_best')}. Intervals are descriptive across archive blocks.", *notes["acquisition"]],
        "agent_design": ["All five arms see the same full 96-variant feature roster at inception and have 15 measured queries. "
                         "B ranks by prediction. C adds a risk bonus. D and E use identical scoring with the same model; E receives "
                         "additional observed summaries and coarse chemistry. The LLM tool scores explicit substitution strings."],
        "agent": agent,
        "trajectory": trajectory,
        "contrasts": figures,
        "agent_results": [f"D - max(B,C) {contrast(ag.loc['D_agent', 'max_non_llm'], 'new_best')}. "
                          f"E - max(B,C) {contrast(ag.loc['E_knowledge', 'max_non_llm'], 'new_best')}.",
                          *notes["agent"],
                          f"Feedback audit: {audit['contexts']} contexts, twice per condition. Mean set overlap was "
                          f"{audit['mean_between_jaccard']:.2f} across conditions and {audit['mean_within_with_jaccard']:.2f}/"
                          f"{audit['mean_within_without_jaccard']:.2f} within conditions. Feedback changed the best measured score "
                          f"by {audit['mean_feedback_best_delta']:+.4f} [{audit['feedback_best_ci_low']:+.4f}, "
                          f"{audit['feedback_best_ci_high']:+.4f}] on fixed forecasts. Only two independent blocks contribute."],
        "case": [f"Illustrative archived case, block 2208 D, round 1: {target} was declared before reveal against "
                 f"{verdict['comparator_id']} ({verdict['comparator_fitness']:.4f}). The fitted score was "
                 f"{verdict['forecast_fitness']:.4f}; the new measured score was {verdict['observed_fitness']:.4f}. "
                 f"Verdict {verdict['status']}; prediction residual {verdict['prediction_residual']:+.4f}.",
                 f"Round 2 cited the measurement and recorded '{next_decision['adaptation'].replace('_', ' ')}'. "
                 f"Its new test {second_verdict['candidate_id']} was {second_verdict['status']} at measured "
                 f"{second_verdict['observed_fitness']:.4f}. Both forecasts were already below WT. This is evidence "
                 "of a fulfilled decision loop and a useful challenge for scientific calibration.",
                 *notes["case"]],
        "limitations": notes["limitations"],
        "reproduction": notes["reproduction"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=Path("evidence/gfp-v2"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(collect(args.evidence), stream, indent=2, ensure_ascii=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
