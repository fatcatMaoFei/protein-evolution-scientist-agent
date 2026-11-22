"""Run the archival GB1 prediction and measured concrete comparisons."""
from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKENDS = {
    "sonnet": ("sonnet_transport", "SONNET_MODEL", "protocol-factorial.md"),
    "gpt": ("gpt_transport", "GPT_MODEL", "protocol-four-model.md"),
    "qwen": ("qwen_transport", "QWEN_MODEL", "protocol-qwen-structural.md"),
}


def run_gb1(output: Path) -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "gb1_audit.py"), "--output", str(output)],
        cwd=ROOT, check=True,
    )


def run_concrete(
    output: Path, *, smoke: bool, workers: int, replay: Path | None = None
) -> None:
    from llm_agent import choose_candidates
    from nonbio import run

    def select(**kwargs):
        if replay is None:
            return choose_candidates(**kwargs)
        return choose_candidates(**kwargs, replay_path=replay / kwargs["log_path"].relative_to(output))

    mode = "smoke" if smoke else "formal"
    run(output, seeds=(7,) if smoke else (17, 29, 43), rounds=1 if smoke else 3, budget=5,
        selector=select, workers=workers, mode=mode if replay is None else f"replay-{mode}")


def run_addon(model, output, baseline, *, smoke, workers, archive=None):
    from factorial_runner import run

    module, model_attr, protocol = BACKENDS[model]
    transport = importlib.import_module(module)
    adapter = getattr(transport, f"choose_{model}_candidates")

    def select(**kwargs):
        if archive is None:
            return adapter(**kwargs)
        return adapter(**kwargs, replay_path=archive / kwargs["log_path"].relative_to(output))

    run(output, baseline=baseline, selector=select, treatment_model=getattr(transport, model_attr),
        freeze_path=ROOT / protocol, smoke=smoke, workers=1 if model in ("gpt", "qwen") else workers)


def run_aggregate(output, archive=None):
    from across_models import analyze as analyze_models
    from exploratory import run as run_ablation
    from factorial_analysis import analyze as analyze_pair

    sources = archive if archive else output
    analyze_pair(sources / "concrete", sources / "sonnet", output / "factorial")
    analyze_models(sources / "concrete", {name: sources / name for name in BACKENDS}, output / "across-models")
    run_ablation(sources / "concrete", output / "exploratory")


def main() -> None:
    if sys.argv[1:2] in (["gfp"], ["setup"], ["demo"]):
        from gfp import main as run_gfp
        command, *args = sys.argv[1:]
        run_gfp(args if command == "gfp" else [command, *args])
        return
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("study", choices=("smoke", "gb1", "concrete", "report", "audit", "all", "full", "replay", "replay-full", *BACKENDS))
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--evidence", type=Path, default=ROOT / "evidence")
    cli.add_argument("--workers", type=int, default=4)
    cli.add_argument("--smoke", action="store_true", help="Use seed 7 and one round for a named add-on.")
    audit_mode = cli.add_mutually_exclusive_group()
    audit_mode.add_argument("--paired", action="store_true", help="Include the live paired feedback audit.")
    audit_mode.add_argument("--replay-audit", type=Path, help="Replay already archived audit decisions.")
    args = cli.parse_args()
    output, evidence = args.output.resolve(), args.evidence.resolve()
    if args.workers < 1:
        cli.error("--workers must be at least 1")
    if args.study == "report":
        from reporting import build_report
        build_report(evidence, output)
        return
    if args.study == "audit":
        from decision_audit import run as run_audit
        result = run_audit(evidence, output, paired=args.paired, replay=args.replay_audit)
        print(result["trajectory_status"], result["paired_status"], output)
        return
    if output.exists() and any(output.iterdir()):
        cli.error(f"output directory must be empty: {output}")

    if args.study in BACKENDS:
        run_addon(args.study, output, evidence / ("smoke" if args.smoke else "concrete"),
                  smoke=args.smoke, workers=args.workers)
    elif args.study in ("all", "replay", "full", "replay-full"):
        from reporting import build_report
        from verify import compare_replay, verify_concrete

        subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, check=True)
        archive = evidence if args.study.startswith("replay") else None
        for stage in ("smoke", "concrete"):
            if stage == "concrete":
                run_gb1(output / "gb1")
            run_concrete(output / stage, smoke=stage == "smoke", workers=args.workers,
                         replay=archive / stage if archive else None)
        if archive:
            compare_replay(archive, output)
        else:
            verify_concrete(output / "smoke", expected_seeds=[7], expected_rounds=1)
            verify_concrete(output / "concrete", expected_seeds=[17, 29, 43], expected_rounds=3)
        if args.study in ("full", "replay-full"):
            def run_model(model):
                for smoke in (True, False):
                    key = f"{model}-smoke" if smoke else model
                    baseline = (archive or output) / ("smoke" if smoke else "concrete")
                    run_addon(model, output / key, baseline, smoke=smoke, workers=args.workers,
                              archive=archive / key if archive else None)

            with ThreadPoolExecutor(max_workers=min(len(BACKENDS), args.workers)) as executor:
                list(executor.map(run_model, BACKENDS))
            if archive:
                compare_replay(archive, output, extra_models=tuple(BACKENDS))
            run_aggregate(output, archive)
        build_report(output, output / "report")
    elif args.study == "gb1":
        run_gb1(output)
    else:
        run_concrete(output, smoke=args.study == "smoke", workers=args.workers)


if __name__ == "__main__":
    main()
