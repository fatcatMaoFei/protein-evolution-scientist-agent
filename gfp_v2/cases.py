"""Extract reproducible illustrative cases after the V2 main study is complete."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gfp_data import make_variant


WT = 3.7192121319


def prediction_cases(frame):
    pool = frame.loc[(frame["model"] == "blend") & (frame["budget"] == "ample")].copy()
    pool["observed_minus_prediction"] = pool.measured - pool.predicted
    pool["error"] = pool.observed_minus_prediction.abs()
    high = pool.loc[(pool.measured > WT) & (pool.predicted > WT)]
    cases = {
        "accurate_above_wt": high.sort_values(["error", "variant_id"]).head(1),
        "largest_overestimate": pool.sort_values(["observed_minus_prediction", "variant_id"]).head(1),
        "largest_underestimate": pool.sort_values(["observed_minus_prediction", "variant_id"],
                                                ascending=[False, True]).head(1),
    }
    return {name: (rows.to_dict("records")[0] if not rows.empty else None) for name, rows in cases.items()}


def decision_cases(pairs, measured):
    paired = pairs.loc[pairs.control == "B_predictor"]
    wins, losses = paired.loc[paired.new_best > 0], paired.loc[paired.new_best < 0]
    cases = {"agent_beats_direct": wins.sort_values(["new_best", "block", "treatment"],
                                                   ascending=[False, True, True]).head(1),
             "agent_loses_to_direct": losses.sort_values(["new_best", "block", "treatment"]).head(1)}
    result = {}
    for label, table in cases.items():
        if table.empty:
            result[label] = None
            continue
        row = table.to_dict("records")[0]
        relevant = measured.loc[measured.block == row["block"]]
        for key in ("treatment", "control"):
            observations = relevant.loc[relevant.arm == row[key]]
            selected = observations.sort_values(["measured_fitness", "id"], ascending=[False, True]).iloc[0]
            row[f"{key}_best"] = {"id": selected.id, "round": int(selected["round"]),
                                  "prediction": float(selected.prediction), "measurement": float(selected.measured_fitness),
                                  "sequence": make_variant(selected.id).sequence}
        result[label] = row
    return result


def extract(root: Path) -> dict:
    examples = prediction_cases(pd.read_csv(root / "prediction" / "test_predictions.csv"))
    for item in examples.values():
        if item:
            item["sequence"] = make_variant(item["variant_id"]).sequence
    return {"scope": "Retrospective examples; illustrative extrema do not estimate frequency or causation.",
            "prediction": examples,
            "decision": decision_cases(pd.read_csv(root / "agent/paired.csv"),
                                       pd.read_csv(root / "agent/selections.csv")),
            "prospective_case": "agent-diagnostics/case-refuted-and-cited.json"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=Path("evidence/gfp-v2"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = extract(args.evidence)
    args.output.parent.mkdir(exist_ok=True, parents=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
