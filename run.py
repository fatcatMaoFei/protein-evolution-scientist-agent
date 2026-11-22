"""Entry points for the two bounded studies and their evidence bundle."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_gb1(output: Path) -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "gb1_audit.py"), "--output", str(output)],
        cwd=ROOT,
        check=True,
    )


def run_concrete(
    output: Path, *, smoke: bool, workers: int, replay: Path | None = None
) -> None:
    from llm_agent import choose_candidates
    from nonbio import run

    def select(**kwargs):
        if replay is None:
            return choose_candidates(**kwargs)
        archived_log = replay / kwargs["log_path"].relative_to(output)
        return choose_candidates(**kwargs, replay_path=archived_log)

    mode = "smoke" if smoke else "formal"
    run(
        output,
        seeds=(7,) if smoke else (17, 29, 43),
        rounds=1 if smoke else 3,
        budget=5,
        selector=select,
        workers=workers,
        mode=mode if replay is None else f"replay-{mode}",
    )


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("study", choices=("smoke", "gb1", "concrete", "report", "all", "replay"))
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--evidence", type=Path, default=ROOT / "evidence")
    cli.add_argument("--workers", type=int, default=4)
    args = cli.parse_args()
    output = args.output.resolve()
    if args.workers < 1:
        cli.error("--workers must be at least 1")
    if args.study == "report":
        from reporting import build_report

        build_report(args.evidence.resolve(), output)
        return
    if output.exists() and any(output.iterdir()):
        cli.error(f"output directory must be empty: {output}")

    if args.study in ("all", "replay"):
        subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, check=True)
        replay = args.evidence.resolve() if args.study == "replay" else None
        run_concrete(output / "smoke", smoke=True, workers=args.workers,
                     replay=replay / "smoke" if replay else None)
        run_gb1(output / "gb1")
        run_concrete(output / "concrete", smoke=False, workers=args.workers,
                     replay=replay / "concrete" if replay else None)
        if replay is not None:
            from verify import compare_replay
            compare_replay(replay, output)
        from reporting import build_report

        build_report(output, output / "report")
    elif args.study == "gb1":
        run_gb1(output)
    else:
        run_concrete(output, smoke=args.study == "smoke", workers=args.workers)


if __name__ == "__main__":
    main()
