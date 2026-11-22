"""Command-line interface for the measured GFP fluorescence experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "gfp_protocol.md"
BACKENDS = ("deepseek", "sonnet", "gpt", "qwen")
SEEDS = (17, 29, 43)
SMOKE_SEED = 7
BUDGET = 5


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")


def status(folder: Path) -> dict:
    return json.loads((folder / "status.json").read_text(encoding="utf-8"))


def run_benchmark(dataset, output: Path) -> None:
    from gfp_data import WT_ID, fit_predictor, make_split, metrics

    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(output)
    records = []
    for seed in SEEDS:
        split = make_split(dataset, seed)
        write_json(output / "splits" / f"seed-{seed}.json", split)
        train = [dataset.variants[key] for key in split["initial"]]
        y = [dataset.scores[key] for key in split["initial"]]
        for name in ("ridge", "rf"):
            model = fit_predictor(train, y, seed, model=name)
            for section in ("validation", "test"):
                requested = split[section]
                predicted = model.predict([dataset.variants[key] for key in requested])
                actual = [dataset.scores[key] for key in requested]
                records.append({"seed": seed, "model": name, "split": section, "n": len(requested),
                                **metrics(actual, predicted, top_k=BUDGET, threshold=dataset.scores[WT_ID])})
    pd.DataFrame(records).to_csv(output / "metrics.csv", index=False, lineterminator="\n")
    write_json(output / "source.json", dataset.provenance)
    write_json(output / "status.json", {"status": "complete", "seeds": list(SEEDS), "benchmark_rows": len(records)})


def bind_selector(backend: str, output: Path, archived: Path | None = None, transport=None):
    from gfp_agent import select_candidates

    def selector(request, *, log_path):
        original = None if archived is None else archived / Path(log_path).resolve().relative_to(output.resolve())
        return select_candidates(request, log_path=log_path, backend=backend, replay_path=original, transport=transport)

    return selector


def run_backend(dataset, output: Path, *, backend: str, smoke: bool, workers: int, archived: Path | None = None,
                transport=None, baseline_only: bool = False, with_controls: bool = False):
    from gfp_evolution import ARMS, run_evolution

    name = "openai_compatible" if transport else backend
    arms = ARMS[:2] if baseline_only else ARMS if backend == "deepseek" or transport or with_controls else ARMS[2:]
    selector = None if baseline_only else bind_selector(name, output, archived, transport)
    return run_evolution(
        dataset, output, selector=selector, backend=name,
        seeds=(SMOKE_SEED,) if smoke else SEEDS, rounds=1 if smoke else 3, budget=BUDGET,
        arms=arms, mode="smoke" if smoke else "formal",
        workers=1 if backend in ("qwen", "gpt") else workers, protocol_path=PROTOCOL,
    )


def verify_replay(original: Path, replayed: Path) -> dict:
    compared = []
    for backend in BACKENDS:
        for stage in ("smoke", "formal"):
            for name in ("rounds.csv", "selections.csv", "methods.csv", "method_summary.csv", "predictive.csv"):
                filename = Path(stage, backend, name)
                lhs = pd.read_csv(original / filename)
                rhs = pd.read_csv(replayed / filename)
                stable = [column for column in lhs if column not in ("elapsed_s", "time_s", "total_elapsed_s")]
                pd.testing.assert_frame_equal(lhs[stable], rhs[stable], check_exact=False, atol=1e-10, rtol=1e-10)
                compared.append(str(filename))
    return {"matched": compared, "count": len(compared)}


def finish_aggregate(dataset, formal: Path, output: Path) -> dict:
    from gfp_evolution import aggregate_studies

    result = aggregate_studies(formal / "deepseek", {name: formal / name for name in BACKENDS[1:]}, output)
    if result["status"] != "complete":
        raise ValueError(f"Incomplete GFP cells; see {output / 'status.json'}")
    rows = []
    for backend in BACKENDS:
        parent = formal / backend
        records = pd.read_csv(parent / "selections.csv")
        for seed in SEEDS:
            def round_rows(arm, round_number):
                return records[(records.seed == seed) & (records.arm == arm) &
                               (records["round"] == round_number)]

            for round_number in (1, 2, 3):
                left = round_rows("C_llm", round_number)
                right = round_rows("D_knowledge_llm", round_number)
                overlap = len(set(left.id) & set(right.id))
                if len(left) != BUDGET or len(right) != BUDGET:
                    raise ValueError("Incomplete GFP knowledge pair")
                menus = [
                    json.loads((parent / "arms" / f"seed-{seed}" / arm / f"round-{round_number:02d}-request.json")
                               .read_text(encoding="utf-8"))["menu"]
                    for arm in ("C_llm", "D_knowledge_llm")
                ]
                if round_number == 1 and menus[0] != menus[1]:
                    raise ValueError("First-round knowledge comparison lacks an identical menu")
                rows.append({
                    "backend": backend, "seed": seed, "round": round_number,
                    "shared_initial_state": round_number == 1, "same_menu": menus[0] == menus[1],
                    "overlap_at_five": overlap / BUDGET, "set_changed": overlap < BUDGET,
                    "batch_best_delta_on_minus_off": right.fitness.max() - left.fitness.max(),
                })
    pd.DataFrame(rows).to_csv(output / "knowledge_selection_overlap.csv", index=False, lineterminator="\n")
    from gfp_data import WT_ID

    wt = dataset.scores[WT_ID]
    combinations = [variant for variant in dataset.variants.values() if len(variant.edits) >= 2]
    deviations = [
        dataset.scores[variant.id] - (wt + sum(dataset.scores[edit] - wt for edit in variant.edits))
        for variant in combinations if all(edit in dataset.scores for edit in variant.edits)
    ]
    combined = pd.read_csv(output / "selections.csv")
    unique_selected = combined[combined.arm.isin(("C_llm", "D_knowledge_llm"))].id.unique()
    positions = Counter(int(edit[1:-1]) for key in unique_selected for edit in dataset.variants[key].edits)
    write_json(output / "landscape_summary.json", {
        "role": "descriptive after all formal runs; no model selection or candidate feedback",
        "additive_reference": "WT log score plus each measured single-edit difference from WT; no clipping",
        "eligible_archive_combinations": len(combinations),
        "combinations_with_all_components_measured": len(deviations),
        "median_signed_deviation": float(np.median(deviations)) if deviations else None,
        "median_absolute_deviation": float(np.median(np.abs(deviations))) if deviations else None,
        "unique_llm_selected_variants": len(unique_selected),
        "most_frequent_selected_positions": [
            {"position": position, "distinct_selections": count}
            for position, count in sorted(positions.items(), key=lambda item: (-item[1], item[0]))[:5]
        ],
    })
    return result


def run_all(dataset, output: Path, *, workers: int, archived: Path | None = None) -> None:
    write_json(output / "design.json", {"protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
                                       "seeds": list(SEEDS), "rounds": 3, "budget": BUDGET,
                                       "backends": list(BACKENDS), "replay": archived is not None})
    run_benchmark(dataset, output / "benchmark")

    def run_study(backend):
        for stage in ("smoke", "formal"):
            directory = Path(stage, backend)
            result = run_backend(dataset, output / directory, backend=backend, smoke=stage == "smoke", workers=workers,
                                 archived=archived / directory if archived else None)
            if result["status"] != "complete":
                raise ValueError(f"Incomplete GFP {backend} {stage} archive: {output / directory}")

    with ThreadPoolExecutor(max_workers=min(workers, len(BACKENDS))) as executor:
        list(executor.map(run_study, BACKENDS))
    formal = output / "formal"
    finish_aggregate(dataset, formal, output / "aggregate")
    if archived:
        write_json(output / "replay_check.json", verify_replay(archived, output))

    from gfp_audit import run_audit
    from gfp_continue import run_user_flow
    from gfp_report import render

    audit = run_audit(dataset, output, output / "audit", paired=archived is None,
                      replay=archived / "audit" if archived else None, workers=workers)
    if audit["paired_status"] != "complete":
        raise ValueError(f"GFP audit remains incomplete: {output / 'audit/audit_summary.json'}")
    run_user_flow(dataset, output / "user-demo", stage="demo", measured_csv=None, prior_state=None,
                  batch_csv=None, candidate_csv=None, backend="deepseek", use_llm=False)
    render(output, output / "report/gfp-research.pdf")


def setup_cli(argv: list[str] | None = None) -> None:
    from gfp_setup import DEFAULT_PROFILE_DIR, setup_openai_profile

    parser = argparse.ArgumentParser(description="Configure and test a native-tool OpenAI-compatible endpoint")
    parser.add_argument("--profile")
    parser.add_argument("--base-url")
    parser.add_argument("--model-id")
    parser.add_argument("--key-env")
    parser.add_argument("--profile-dir", type=Path, default=DEFAULT_PROFILE_DIR)
    args = parser.parse_args(argv)
    profile = args.profile or input("Profile [default]: ").strip() or "default"
    base_url = args.base_url or input("OpenAI-compatible API base URL: ").strip()
    model_id = args.model_id or input("Model ID: ").strip()
    key_env = args.key_env
    if key_env is None and sys.stdin.isatty():
        key_env = input("API key environment variable name [Enter for private local key]: ").strip() or None
    result = setup_openai_profile(profile, base_url=base_url, model_id=model_id, key_env=key_env,
                                  profile_dir=args.profile_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main(argv: list[str] | None = None) -> None:
    args_in = sys.argv[1:] if argv is None else argv
    if args_in and args_in[0] == "setup":
        setup_cli(args_in[1:])
        return

    from gfp_data import load_dataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("benchmark", "smoke", "evolve", "aggregate", "audit",
                                          "init", "continue", "demo", "full", "replay", "report"))
    parser.add_argument("--dataset", choices=("gfp",), default="gfp")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-cache", type=Path)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "evidence/gfp")
    parser.add_argument("--backend", choices=BACKENDS, default="deepseek")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--replay-audit", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--measurements", type=Path)
    parser.add_argument("--candidate-file", type=Path)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--from-run", type=Path)
    parser.add_argument("--model-profile")
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--with-controls", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args(args_in)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.output is None and args.stage != "demo":
        parser.error("--output is required outside the demo")
    default_output = ROOT / "runs" / f"gfp-demo-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%f}"
    output, run_dir = (args.output or default_output).resolve(), args.run_dir.resolve()
    transport = None
    if args.model_profile:
        if args.stage not in ("smoke", "evolve", "init", "continue", "demo") or args.baseline_only:
            parser.error("--model-profile is intended for single-model runs")
        from gfp_setup import DEFAULT_PROFILE_DIR, load_openai_profile
        transport = load_openai_profile(args.model_profile, profile_dir=args.profile_dir or DEFAULT_PROFILE_DIR,
                                        require_key=args.from_run is None)

    if args.stage == "report":
        from gfp_report import render
        render(run_dir, output)
        return
    if args.stage == "aggregate":
        finish_aggregate(load_dataset(args.data_cache), run_dir / "formal", output)
        return
    dataset = load_dataset(args.data_cache)
    if args.stage == "benchmark":
        run_benchmark(dataset, output)
    elif args.stage in ("smoke", "evolve"):
        result = run_backend(dataset, output, backend=args.backend, smoke=args.stage == "smoke", workers=args.workers,
                             transport=transport, archived=args.from_run, with_controls=args.with_controls,
                             baseline_only=args.baseline_only)
        print(json.dumps(result, ensure_ascii=False))
        if result["status"] != "complete":
            raise SystemExit(1)
    elif args.stage in ("full", "replay"):
        if output.exists() and any(output.iterdir()):
            parser.error(f"output directory must be empty: {output}")
        archived = (args.from_run or run_dir).resolve() if args.stage == "replay" else None
        run_all(dataset, output, workers=args.workers, archived=archived)
    elif args.stage == "audit":
        from gfp_audit import run_audit
        run_audit(dataset, run_dir, output, paired=args.paired, replay=args.replay_audit, workers=args.workers)
    else:
        from gfp_continue import run_user_flow
        run_user_flow(dataset, output, stage=args.stage, measured_csv=args.input, prior_state=args.state,
                      batch_csv=args.measurements, candidate_csv=args.candidate_file, backend=args.backend,
                      use_llm=args.use_llm or transport is not None, transport=transport)
        print(output)


if __name__ == "__main__":
    main()
