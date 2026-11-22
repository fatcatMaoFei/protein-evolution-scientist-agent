"""Use a verified compatible LLM profile and personal GFP assays without archive lookups."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from gfp_continue import ASSAY, load_csv
from gfp_data import GFP_WT, make_variant
from gfp_setup import load_openai_profile
from gfp_v2.policy import evaluate_hypotheses, select
from gfp_v2.predictor import MODELS, fit


STATE_VERSION = "gfp-v2-open-1"
REFERENCE_SHA256 = hashlib.sha256(GFP_WT.encode()).hexdigest()


def _write_json(path: Path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False, ensure_ascii=True)
        stream.write("\n")


def _prior_state(path: Path | None):
    if not path:
        return None, None
    content = path.read_bytes()
    state = json.loads(content)
    if state["format"] != STATE_VERSION or state["assay"] != ASSAY or state["reference_sha256"] != REFERENCE_SHA256:
        raise ValueError("Use a V2 GFP state with the pinned assay and reporter reference")
    trace_path = path.parent / "agent.json"
    trace_bytes = trace_path.read_bytes()
    if hashlib.sha256(trace_bytes).hexdigest() != state["trace_sha256"]:
        raise ValueError("The previous agent trace is missing or changed")
    trace = json.loads(trace_bytes)
    observed = state["observed"]
    history = state["history"]
    scores = {row["id"]: row["fitness"] for row in observed}
    if (trace["status"] != "accepted" or not history or state["round"] != len(history)
            or history[-1]["decision"] != trace["decision"] or history[-1]["round"] != state["round"]
            or observed != trace["input"]["observed"] or history[-1]["prior_best"] != trace["input"]["current_best"]
            or _reconcile(history[:-1], observed, scores["WT"]) != trace["input"]["prior_decisions"]):
        raise ValueError("The previous V2 observation or hypothesis ledger does not match its accepted trace")
    pending = {key for entry in history for key in entry["decision"]["selected_ids"]} - scores.keys()
    if sorted(pending) != state["pending_ids"]:
        raise ValueError("The previous V2 pending IDs do not match its accepted decision history")
    return state, hashlib.sha256(content).hexdigest()


def _reconcile(history, observed, wild_type):
    measured = {row["id"]: row["fitness"] for row in observed}
    return [
        {"round": prior["round"], "selected_ids": prior["decision"]["selected_ids"],
         "strategy": prior["decision"]["strategy"],
         "verdicts": evaluate_hypotheses(prior["decision"], measurements=measured,
                                         wild_type_fitness=wild_type, prior_best=prior["prior_best"])}
        for prior in history
    ]


def run_open(*, measurements: Path, out: Path, model: str, transport, prior_path: Path | None = None,
             candidate_path: Path | None = None, budget: int = 5, seed: int = 17, replay_path: Path | None = None):
    """Fit to the import ledger, score designs, and preserve the next pending decisions."""
    prior, prior_sha256 = _prior_state(prior_path)
    number = prior["round"] + 1 if prior else 1
    if prior and (prior["predictor"] != model or prior["seed"] != seed or prior["budget"] != budget):
        raise ValueError("Keep the same predictor, seed and budget when continuing the V2 trajectory")
    observed = list(prior["observed"]) if prior else []
    imported = load_csv(measurements, measured=True)
    if set(imported) & {row["id"] for row in observed}:
        raise ValueError("A new measurement duplicates an observed GFP variant")
    observed.extend({"id": key, "fitness": value.fitness, "round": number - 1} for key, value in imported.items())
    scores = {row["id"]: row["fitness"] for row in observed}
    if "WT" not in scores or len(scores) < 4:
        raise ValueError("Provide WT fluorescence and at least three other distinct measured GFP variants")
    wt, baseline = scores["WT"], max(scores.values())
    history = list(prior["history"]) if prior else []
    pending = set(prior["pending_ids"]) - set(imported) if prior else set()
    prior_decisions = _reconcile(history, observed, wt)
    estimates = fit([make_variant(key) for key in scores], list(scores.values()), seed, model)

    def predict(variants):
        mean, risk = estimates.estimate(variants)
        return {variant.id: {"id": variant.id, "prediction": float(prediction), "risk": float(uncertainty)}
                for variant, prediction, uncertainty in zip(variants, mean, risk, strict=True)}

    pool = load_csv(candidate_path, measured=False) if candidate_path else {}
    current = predict([variant for key, variant in pool.items() if key not in scores and key not in pending])
    request = {
        "experiment_id": "user_gfp_v2", "arm": "open", "block": "user", "round": number, "budget": budget,
        "predictor_name": model, "wild_type_fitness": wt, "current_best": baseline,
        "observed": observed, "candidates": list(current.values()), "pending_ids": sorted(pending),
        "latest_feedback": [row for row in observed if row["round"] == number - 1] if prior else [],
        "prior_decisions": prior_decisions,
    }
    out.mkdir(parents=True, exist_ok=False)
    trace = out / "agent.json"
    decision = select(request=request, transport=transport, knowledge=True, log_path=trace, replay_path=replay_path,
                      score_new=predict)
    history.append({"round": number, "prior_best": baseline, "decision": decision})
    table = {row["id"]: row for row in decision["designs"]}
    pending.update(decision["selected_ids"])
    state = {"format": STATE_VERSION, "assay": ASSAY, "reference_sha256": REFERENCE_SHA256, "round": number,
             "seed": seed, "predictor": model, "budget": budget, "observed": observed,
             "prior_state_sha256": prior_sha256, "pending_ids": sorted(pending), "history": history,
             "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest()}
    _write_json(out / "state.json", state)
    _write_json(out / "hypothesis_audit.json", _reconcile(history, observed, wt))
    with (out / "recommendations.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["mutant", "sequence", "prediction", "risk", "measured_fitness", "status", "source"])
        for key in decision["selected_ids"]:
            row = table[key]
            writer.writerow([key, row["sequence"], row["prediction"], row["risk"], "", "PENDING", row["source"]])
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("init", "continue"))
    parser.add_argument("--measurements", required=True, type=Path, help="CSV with mutant,fitness and optional sequence")
    parser.add_argument("--candidates", type=Path, help="Optional CSV with mutant and optional sequence, no fitness")
    parser.add_argument("--state", type=Path, help="State JSON from the previous V2 recommendation round")
    parser.add_argument("--model", choices=MODELS, required=True, help="The locked Predictor V2 recipe")
    parser.add_argument("--model-profile", default="default", help="Verified using run.py setup")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--budget", default=5, type=int)
    parser.add_argument("--seed", default=17, type=int)
    parser.add_argument("--replay", type=Path, help="Original native agent JSON for this exact decision")
    args = parser.parse_args(argv)
    if bool(args.state) != (args.stage == "continue"):
        parser.error("--state is required only for continue")
    transport = load_openai_profile(args.model_profile, require_key=args.replay is None)
    state = run_open(measurements=args.measurements, candidate_path=args.candidates, out=args.output, model=args.model,
                     transport=transport, prior_path=args.state, budget=args.budget, seed=args.seed, replay_path=args.replay)
    print(f"Prepared {state['budget']} pending GFP recommendations for round {state['round']}: {args.output}")


if __name__ == "__main__":
    main()
