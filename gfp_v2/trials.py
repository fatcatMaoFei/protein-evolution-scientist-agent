"""Run independent GFP V2 acquisition blocks and closed-pool agent trials."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

import numpy as np

from gfp_data import load_dataset
from gfp_setup import DEFAULT_PROFILE_DIR, OpenAICompatibleTransport, load_openai_profile
from llm_agent import AgentError

ROUNDS = 3
BUDGET = 5
INITIAL_COUNT = 96
POOL_COUNT = 96
RISK_WEIGHT = 0.5
BOOT_REP = 10_000
MAX_OUTER_ATTEMPTS = 2
ACQUISITION_ARMS = ("A_random", "B_rf", "C_ridge", "D_v2")
AGENT_ARMS = ("A_random", "B_predictor", "C_explore", "D_agent", "E_knowledge")
MODEL_ARMS = {"D_agent", "E_knowledge"}
DEFAULT_MANIFEST = Path("evidence/gfp-v2/partition.json")
DEFAULT_SELECTION = Path("evidence/gfp-v2/prediction/selection.json")


def read_json(path: Path) -> dict:
    return json.loads(path.read_bytes())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def deterministic_seed(*parts) -> int:
    key = json.dumps(list(parts), separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        with path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


def preserve(path: Path, value) -> None:
    if path.is_file():
        if read_json(path) != value:
            raise ValueError(f"Frozen artifact differs from regenerated state: {path}")
    else:
        write_json(path, value)


def transport_spec(transport, profile: str) -> dict:
    return {"model_id": transport.model, "base_url": transport.endpoint, "profile": profile}


def archived_transport(source: Path):
    spec = read_json(source / "config.json")["profile_transport"]
    return OpenAICompatibleTransport(**spec, api_key=None), spec["profile"]


def validated_blocks(manifest: dict, dataset) -> dict[str, list[dict]]:
    seen = set()
    studies = {}
    wt = dataset.wild_type_id
    for study in ("acquisition", "agent"):
        blocks = manifest[study]
        block_names = set()
        studies[study] = []
        for block in blocks:
            identity = str(block["id"])
            init, pool = block["initial"], block["pool"]
            used = (set(init) | set(pool)) - {wt}
            if (
                not identity.isdigit() or identity in block_names
                or len(init) != INITIAL_COUNT or len(set(init)) != INITIAL_COUNT
                or len(pool) != POOL_COUNT or len(set(pool)) != POOL_COUNT
                or wt not in init or set(init) & set(pool)
                or used & seen or not used <= dataset.variants.keys()
            ):
                raise ValueError(f"Invalid or overlapping sealed block {study}/{identity}")
            block_names.add(identity)
            seen.update(used)
            studies[study].append({"id": identity, "initial": list(init), "pool": list(pool)})
        if not studies[study]:
            raise ValueError(f"Study has no blocks: {study}")
    return studies


def forecast(dataset, observed: dict[str, float], available: list[str], model: str, seed: int) -> list[dict]:
    from gfp_v2.predictor import fit

    fitted = fit([dataset.variants[key] for key in observed], list(observed.values()), seed, model)
    means, risks = fitted.estimate([dataset.variants[key] for key in available])
    candidates = []
    for key, mu, risk in zip(available, means, risks, strict=True):
        mu, risk = float(mu), float(risk)
        if not math.isfinite(mu) or not math.isfinite(risk) or risk < 0:
            raise ValueError("A candidate must have a finite estimate and a nonnegative risk")
        candidates.append({"id": key, "prediction": mu, "risk": risk})
    return candidates


def choose_simple(arm: str, candidates: list[dict], seed: int) -> list[str]:
    if arm == "A_random":
        return random.Random(seed).sample([row["id"] for row in candidates], BUDGET)
    weight = RISK_WEIGHT if arm == "C_explore" else 0
    ranked = sorted(candidates, key=lambda row: (-(row["prediction"] + weight * row["risk"]), row["id"]))
    return [row["id"] for row in ranked[:BUDGET]]


def check_choices(choice, eligible) -> list[str]:
    if (
        not isinstance(choice, list)
        or len(choice) != BUDGET
        or not all(isinstance(key, str) for key in choice)
        or len(set(choice)) != BUDGET
        or not set(choice) <= set(eligible)
    ):
        raise ValueError("Precommit must contain exactly the budget of distinct, eligible variant IDs")
    return choice


class ArchiveOracle:
    """Look up only committed IDs, without materializing the pool's measurements."""

    def __init__(self, pool, score):
        self.available = set(pool)
        self._score = score

    def reveal(self, commit: Path) -> dict[str, float]:
        chosen = check_choices(read_json(commit)["policy"]["selected_ids"], self.available)
        measurements = {key: self._score(key) for key in chosen}
        if not all(math.isfinite(value) for value in measurements.values()):
            raise ValueError("Committed acquisition selects a non-finite archival measurement")
        self.available.difference_update(chosen)
        return measurements


def quantities(rows: list[dict], wt: float, baseline: float) -> dict:
    values = [row["measured_fitness"] for row in rows]
    return {
        "new_best": max(values),
        "new_mean": mean(values),
        "above_wt": sum(value > wt for value in values),
        "incumbent_gain": max(0.0, max(values) - baseline),
    }


def run_arm(dataset, block: dict, arm: str, model: str, output: Path, *, transport=None, replay=None,
            record_adaptation=True):
    """Commit each validated batch to disk before the isolated oracle can reveal it."""
    output.mkdir(parents=True, exist_ok=True)
    block_id = block["id"]
    observed = {key: dataset.scores[key] for key in block["initial"]}
    wt = observed[dataset.wild_type_id]
    baseline = max(observed.values())
    initial = [{"id": key, "fitness": value, "round": 0} for key, value in observed.items()]
    preserve(output / "initial.json", {"model": model, "observed": initial})
    history, latest, prior = list(initial), [], []
    available = list(block["pool"])
    oracle = ArchiveOracle(available, dataset.scores.__getitem__)
    rows, batches = [], []
    for number in range(1, ROUNDS + 1):
        stage = f"round-{number:02d}"
        common_seed = deterministic_seed("gfp-v2", block_id, number)
        candidates = forecast(dataset, observed, available, model, common_seed)
        request = {
            "experiment_id": "gfp-v2", "arm": arm, "block": block_id, "round": number, "budget": BUDGET,
            "wild_type_fitness": wt, "current_best": max(observed.values()),
            "candidates": candidates, "observed": history, "latest_feedback": latest, "prior_decisions": prior,
        }
        request_path = output / f"{stage}-request.json"
        preserve(request_path, request)
        decision_path = output / f"{stage}-commit.json"
        if decision_path.exists():
            decision = read_json(decision_path)
            if decision["request_sha256"] != digest(request_path):
                raise ValueError("A resumed selection does not match the held request")
        else:
            if arm in MODEL_ARMS:
                from gfp_v2.policy import select

                previous = sorted(output.glob(f"{stage}-model-*.json"))
                if len(previous) >= MAX_OUTER_ATTEMPTS:
                    raise ValueError(f"Policy attempts exhausted for {block_id}/{arm}/{stage}")
                if replay is not None:
                    old_commit = read_json(replay / f"{stage}-commit.json")
                    name = old_commit["model_log"]
                    if Path(name).name != name:
                        raise ValueError("Archived model log must refer to this arm only")
                    policy = select(request=request, transport=transport, knowledge=arm == "E_knowledge",
                                    log_path=output / name, replay_path=replay / name)
                else:
                    for attempt in range(len(previous) + 1, MAX_OUTER_ATTEMPTS + 1):
                        name = f"{stage}-model-{attempt}.json"
                        try:
                            policy = select(request=request, transport=transport, knowledge=arm == "E_knowledge",
                                            log_path=output / name, repair_aliases=attempt > 1)
                        except AgentError:
                            if attempt == MAX_OUTER_ATTEMPTS:
                                raise
                        else:
                            break
                decision = {"policy": policy, "model_log": name}
            else:
                decision = {"policy": {"selected_ids": choose_simple(arm, candidates, common_seed)}}
            check_choices(decision["policy"]["selected_ids"], available)
            decision["request_sha256"] = digest(request_path)
            write_json(decision_path, decision)
        choice = check_choices(decision["policy"]["selected_ids"], available)
        feedback = oracle.reveal(decision_path)
        if arm in MODEL_ARMS:
            from gfp_v2.policy import evaluate_hypotheses

            verdicts = evaluate_hypotheses(decision["policy"], measurements=feedback,
                                           wild_type_fitness=wt, prior_best=max(observed.values()))
        else:
            verdicts = []
        candidate_by_id = {row["id"]: row for row in candidates}
        batch = [
            {"block": block_id, "arm": arm, "round": number, "position": rank, "id": key,
             "prediction": candidate_by_id[key]["prediction"], "risk": candidate_by_id[key]["risk"],
             "measured_fitness": value}
            for rank, (key, value) in enumerate(feedback.items(), 1)
        ]
        batch_result = {"block": block_id, "arm": arm, "round": number, **quantities(batch, wt, baseline)}
        batch_result["running_incumbent"] = max(baseline, max(row["measured_fitness"] for row in rows + batch))
        preserve(output / f"{stage}-measurement.json", {
            "commit_sha256": digest(decision_path), "measurements": batch, "verdicts": verdicts,
        })
        latest = [{"id": key, "fitness": value, "round": number} for key, value in feedback.items()]
        next_prior = {"round": number, "selected_ids": choice, "verdicts": verdicts,
                      "strategy": decision["policy"].get("strategy")}
        if record_adaptation:
            next_prior["adaptation"] = decision["policy"].get("adaptation")
        history = history + latest
        prior = prior + [next_prior]
        observed.update(feedback)
        available = [key for key in available if key not in feedback]
        batches.append(batch_result)
        rows.extend(batch)
    totals = {"block": block_id, "arm": arm, "model": model, "initial_best": baseline,
              "budget_used": len(rows), **quantities(rows, wt, baseline)}
    preserve(output / "complete.json", totals)
    return totals, batches, rows


def paired_statistics(arms: list[dict], contrasts: tuple[tuple[str, str], ...]) -> tuple[list[dict], list[dict]]:
    blocks = sorted(set(row["block"] for row in arms))
    indexed = {(row["block"], row["arm"]): row for row in arms}
    metrics = ("new_best", "new_mean", "above_wt", "incumbent_gain")
    pairing, stats = [], []
    samples = np.random.default_rng(668021).integers(len(blocks), size=(BOOT_REP, len(blocks)))
    for treatment, control in contrasts:
        diffs = []
        for block in blocks:
            if control == "max_non_llm":
                greedy, explore = indexed[block, "B_predictor"], indexed[block, "C_explore"]
                comparison = greedy if greedy["new_best"] >= explore["new_best"] else explore
            else:
                comparison = indexed[block, control]
            active = indexed[block, treatment]
            diffs.append({
                "block": block, "treatment": treatment, "control": control,
                "paired_arm": comparison["arm"],
                **{metric: active[metric] - comparison[metric] for metric in metrics},
            })
        summary = {"treatment": treatment, "control": control, "blocks": len(blocks)}
        for metric in metrics:
            values = np.array([row[metric] for row in diffs], dtype=float)
            bootstrap = values[samples].mean(axis=1)
            low, high = np.quantile(bootstrap, [0.025, 0.975])
            summary.update({metric: float(values.mean()), f"{metric}_ci_low": float(low),
                            f"{metric}_ci_high": float(high)})
        values = [row["new_best"] for row in diffs]
        summary.update({"new_best_wins": sum(value > 0 for value in values),
                        "new_best_losses": sum(value < 0 for value in values),
                        "new_best_ties": sum(value == 0 for value in values)})
        pairing.extend(diffs)
        stats.append(summary)
    return pairing, stats


def run_study(study: str, dataset, blocks: list[dict], selection: dict, output: Path, *, transport=None,
              profile=None, replay=None, workers=4, hashes=None, resume=False) -> dict:
    if not resume and output.exists():
        raise FileExistsError(f"Use a fresh study output directory: {output}")
    output.mkdir(parents=True, exist_ok=resume)
    selected = selection["selected"]["low"]
    if study == "acquisition":
        challenger = selected if selected not in {"ridge", "rf"} else selection["best_enhanced"]["low"]
        methods = dict(zip(ACQUISITION_ARMS, ("rf", "rf", "ridge", challenger), strict=True))
        contrasts = (("D_v2", "B_rf"), ("D_v2", "C_ridge"), ("D_v2", "A_random"),
                     ("B_rf", "A_random"), ("C_ridge", "A_random"))
    elif study == "agent":
        if transport is None:
            raise ValueError("Agent trials require one configured model profile")
        methods = dict.fromkeys(AGENT_ARMS, selected)
        contrasts = (
            ("D_agent", "max_non_llm"), ("E_knowledge", "max_non_llm"),
            ("D_agent", "B_predictor"), ("D_agent", "C_explore"),
            ("E_knowledge", "B_predictor"), ("E_knowledge", "C_explore"),
            ("E_knowledge", "D_agent"), ("B_predictor", "A_random"), ("C_explore", "A_random"),
        )
    else:
        raise ValueError(f"Unknown study: {study}")
    preserve(output / "config.json", {
        "study": study, "blocks": blocks, "methods": methods, "rounds": ROUNDS, "round_budget": BUDGET,
        "selected_low": selected, "profile": profile, "requested_model": getattr(transport, "model", None),
        "risk_weight": RISK_WEIGHT, "chronology": "validated precommit before isolated archive reveal",
        "hashes": hashes or {}, "outer_attempt_limit": MAX_OUTER_ATTEMPTS,
        "paired_bootstrap": {"seed": 668021, "replicates": BOOT_REP, "quantiles": [0.025, 0.975]},
        **({"profile_transport": transport_spec(transport, profile)} if transport is not None else {}),
    })
    if (output / "status.json").exists():
        raise FileExistsError("Study is already finalized")
    all_results, errors = {}, []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for block in blocks:
            for arm, model in methods.items():
                relative = Path("arms") / block["id"] / arm
                source = replay / relative if replay and arm in MODEL_ARMS else None
                future = executor.submit(run_arm, dataset, block, arm, model, output / relative,
                                         transport=transport, replay=source, record_adaptation=study == "agent")
                futures[future] = (block["id"], arm)
        for future in as_completed(futures):
            key = futures[future]
            try:
                all_results[key] = future.result()
            except Exception as exc:
                errors.append({"block": key[0], "arm": key[1], "error": f"{type(exc).__name__}: {exc}"})
    if errors:
        number = len(list(output.glob("failures-*.json"))) + 1
        write_json(output / f"failures-{number:02d}.json", sorted(errors, key=lambda row: (row["block"], row["arm"])))
        raise RuntimeError(f"{len(errors)} arms failed. Inspect {output}; accepted rounds remain recorded.")
    ordered = [all_results[block["id"], arm] for block in blocks for arm in methods]
    totals = [entry[0] for entry in ordered]
    batches = [row for entry in ordered for row in entry[1]]
    measurements = [row for entry in ordered for row in entry[2]]
    diagnostics = []
    wt = dataset.scores[dataset.wild_type_id]
    for block in blocks:
        baseline = max(dataset.scores[key] for key in block["initial"])
        truth = [dataset.scores[key] for key in block["pool"]]
        diagnostics.append({"block": block["id"], "incumbent": baseline, "pool_best": max(truth),
                            "count_above_incumbent": sum(value > baseline for value in truth),
                            "count_above_wt": sum(value > wt for value in truth)})
    paired, summary = paired_statistics(totals, contrasts)
    for name, rows in (("methods", totals), ("rounds", batches), ("selections", measurements),
                       ("paired", paired), ("contrasts", summary), ("pool_diagnostics", diagnostics)):
        write_csv(output / f"{name}.csv", rows)
    served_models, alias_repairs = set(), []
    if study == "agent":
        for path in (output / "arms").glob("*/*/round-*-commit.json"):
            commit = read_json(path)
            if "model_log" not in commit:
                continue
            trace = read_json(path.parent / commit["model_log"])
            if trace.get("alias_repair"):
                alias_repairs.append("/".join([path.parent.parent.name, path.parent.name, path.name.split("-commit")[0]]))
            for call in trace["calls"]:
                reported = call.get("response", {}).get("model")
                if reported:
                    served_models.add(reported)
    result = {"status": "complete", "paired_blocks": len(blocks), "queries": len(measurements),
              "policy_decisions": len(blocks) * ROUNDS * len(MODEL_ARMS) if study == "agent" else 0,
              "api_mode": "replay" if replay is not None else "live" if study == "agent" else "none",
              "served_models": sorted(served_models)}
    if study == "agent":
        result["alias_recovered_decisions"] = sorted(alias_repairs)
    write_json(output / "status.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("acquisition", "agent", "audit"), required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-cache", type=Path)
    parser.add_argument("--model-profile")
    parser.add_argument("--profile-dir", type=Path, default=DEFAULT_PROFILE_DIR)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--agent-run", type=Path)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1 or (args.resume and args.replay):
        parser.error("Positive worker count required; pass either resume or replay")
    dataset = load_dataset(args.data_cache)
    hashes = {"manifest_sha256": digest(args.manifest), "selection_sha256": digest(args.selection)}
    if args.protocol:
        hashes["protocol_sha256"] = digest(args.protocol)
    selection = read_json(args.selection)
    blocks = validated_blocks(read_json(args.manifest), dataset)
    profile = args.model_profile
    if profile:
        transport = load_openai_profile(profile, profile_dir=args.profile_dir, require_key=args.replay is None)
    elif args.replay is not None and args.study != "acquisition":
        transport, profile = archived_transport(args.replay)
    else:
        transport = None
    if args.study == "audit":
        from gfp_v2.feedback_audit import run_audit

        if args.agent_run is None or transport is None:
            parser.error("Feedback audit requires --agent-run and --model-profile")
        result = run_audit(dataset, blocks["agent"], args.agent_run, args.output, transport=transport,
                           profile=profile, replay=args.replay, resume=args.resume,
                           workers=args.workers, hashes=hashes)
    else:
        result = run_study(args.study, dataset, blocks[args.study], selection, args.output,
                           transport=transport, profile=profile, replay=args.replay,
                           resume=args.resume, workers=args.workers, hashes=hashes)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
