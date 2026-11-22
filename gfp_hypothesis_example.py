"""Run a prospectively specified GFP mixed-menu hypothesis example with a configured API."""
from __future__ import annotations

import argparse
from pathlib import Path

from gfp_continue import MEASURED_COLUMNS, _load_state, _propose, _write_csv, _write_json, load_csv
from gfp_data import load_dataset, make_split
from gfp_setup import load_openai_profile
from gfp_user_audit import run_user_audit

SEED = 211
MENU_COUNT = 3


def run(dataset, output: Path, transport) -> None:
    """Lock one measured comparison slot, while scoring independently designed candidates."""
    split = make_split(dataset, SEED)
    output.mkdir(parents=True, exist_ok=False)
    initial = output / "initial.csv"
    pool1 = output / "round-01-menu.csv"
    pool2 = output / "round-02-menu.csv"
    _write_json(output / "design.json", {"seed": SEED, "menu_size": MENU_COUNT, "budget": 5,
                 "criterion": "Agent-selected primary design strictly exceeds its cited measured reference",
                 "reveal": "Only the declared archived primary after state and plan persist. Other designs stay pending.",
                 "scope": "Two-round prospective software example, separate from the acquisition ablation"})
    _write_csv(initial, MEASURED_COLUMNS, [(key, dataset.scores[key]) for key in split["initial"]])
    _write_csv(pool1, ("mutant",), [(key,) for key in split["pool"][:MENU_COUNT]])
    _write_csv(pool2, ("mutant",), [(key,) for key in split["pool"][MENU_COUNT:2 * MENU_COUNT]])
    first = output / "round-01"
    state = _propose(dataset, first, load_csv(initial, measured=True), prior=None, candidates=pool1,
                     backend="deepseek", use_llm=True, transport=transport, allow_novel=True, primary_from_menu=True)
    finish(dataset, output, transport, state)


def finish(dataset, output: Path, transport, state: dict) -> None:
    if state["hypothesis_evaluations"][0]["status"] != "pending":
        raise ValueError("The primary GFP hypothesis must be pending before archive reveal")
    primary = state["hypotheses"][0]["candidate_ids"][0]
    if primary not in make_split(dataset, SEED)["pool"][:MENU_COUNT]:
        raise ValueError("The primary comparison must be selected from the committed archive roster")
    second = output / "round-02"
    feedback = output / "returned_primary.csv"
    archived_value = dataset.scores[primary]
    if not feedback.exists():
        _write_csv(feedback, MEASURED_COLUMNS, [(primary, archived_value)])
    imported = load_csv(feedback, measured=True)
    if set(imported) != {primary} or imported[primary].fitness != archived_value:
        raise ValueError("Recorded example feedback does not match the predeclared archive query")
    updated = _propose(dataset, second, imported, prior=output / "round-01/state.json", candidates=output / "round-02-menu.csv",
                       backend="deepseek", use_llm=True, transport=transport, allow_novel=True, primary_from_menu=True)
    trace = run_user_audit(second / "state.json", output / "audit")
    _write_json(output / "status.json", {"status": "complete", "primary": updated["hypothesis_evaluations"][0],
                 "second_hypothesis": updated["hypothesis_evaluations"][1], "audit": trace})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-profile", required=True)
    parser.add_argument("--data-cache", type=Path)
    parser.add_argument("--resume", action="store_true", help="Continue after the first accepted round; use an empty round-02 path")
    args = parser.parse_args()
    dataset = load_dataset(args.data_cache)
    transport = load_openai_profile(args.model_profile)
    if args.resume:
        state, _ = _load_state(args.output / "round-01/state.json")
        finish(dataset, args.output, transport, state)
    else:
        run(dataset, args.output, transport)
