"""Reconstruct the fixed four-model Concrete decision audit from archived evidence."""

from __future__ import annotations

from collections import Counter
import csv
import json
from math import isclose
from pathlib import Path


ARCHIVES = {
    "deepseek": "concrete",
    "sonnet": "sonnet",
    "gpt": "gpt",
    "qwen": "qwen",
}
ARMS = {"C_llm": "off", "D_knowledge_llm": "on"}
SEEDS = (17, 29, 43)
ROUNDS = 3
BUDGET = 5
NOT_EVALUABLE = "NOT EVALUABLE"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _check(condition, problem: str) -> None:
    if not condition:
        raise ValueError(problem)


def _location(root: Path, path: Path, **reference) -> dict:
    return {"file": path.relative_to(root).as_posix(), **reference}


def _unique_by(records: list[dict], field: str) -> dict:
    indexed = {entry[field]: (number, entry) for number, entry in enumerate(records)}
    _check(len(indexed) == len(records), f"Duplicate {field} in archive")
    return indexed


def _check_complete(run_dir: Path) -> None:
    statuses = {}
    for model, source in ARCHIVES.items():
        path = run_dir / source / "status.json"
        # Check only status metadata until every study is complete.
        statuses[model] = _json(path) if path.is_file() else {}
    for model, status in statuses.items():
        _check(
            status.get("status") == "complete" and status.get("mode") == "formal",
            f"Decision audit requires complete formal evidence for {model}",
        )
    for model, source in ARCHIVES.items():
        expected_arms = len(SEEDS) * (4 if model == "deepseek" else len(ARMS))
        status = statuses[model]
        config = _json(run_dir / source / "config.json")
        _check(
            tuple(config["seeds"]) == SEEDS and (config["rounds"], config["budget"]) == (ROUNDS, BUDGET)
            and status["arms"] == expected_arms and status["measurements"] == expected_arms * ROUNDS * BUDGET,
            f"Unexpected formal schedule in {model} evidence",
        )


def _proposals(root: Path, trace_path: Path, trace: dict, observed: dict) -> tuple[dict, list[dict]]:
    native = [(index, tool) for index, tool in enumerate(trace["tool_calls"])
              if tool["tool"] == "evaluate_concrete_shortlist" and tool["trigger"] == "model_tool_call"]
    _check(len(native) == 1, f"Expected one recorded evaluator call in {trace_path}")
    tool_index, evaluator = native[0]
    original = evaluator["arguments"]
    menu = _unique_by(trace["menu"], "alias")
    scored = _unique_by(evaluator["result"], "alias")
    aliases = original["candidate_ids"]
    _check(
        len(set(aliases)) == len(aliases) == min(trace["budget"] + 2, len(menu))
        and set(aliases) == set(scored) and set(aliases) <= set(menu),
        f"Proposal and evaluator disagree in {trace_path}",
    )
    found = []
    for alias in aliases:
        menu_index, entry = menu[alias]
        result_index, score = scored[alias]
        _check(
            score["eligible"] and score["prediction_rank"] == entry["rank"]
            and isclose(score["prediction_mpa"], entry["prediction"], rel_tol=0, abs_tol=0.000501)
            and entry["id"] not in observed and score["nearest_observed_example"] in observed,
            f"Invalid recorded evaluator result for {alias} in {trace_path}",
        )
        found.append({
            "alias": alias,
            "recipe_id": entry["id"],
            "prediction_mpa": entry["prediction"],
            "evaluator_prediction_mpa": score["prediction_mpa"],
            "rf_rank": entry["rank"],
            "rf_tree_disagreement_mpa": score["rf_tree_disagreement_mpa"],
            "menu_source": entry["source"],
            "nearest_measured_example": observed[score["nearest_observed_example"]],
            "sources": {
                "menu": _location(root, trace_path, json_pointer=f"/menu/{menu_index}"),
                "tool": _location(root, trace_path, json_pointer=f"/tool_calls/{tool_index}/result/{result_index}"),
            },
        })
    final = trace["decision"]
    cited = original["evidence_ids"]
    analyst = [tool["result"] for tool in trace["tool_calls"] if tool["tool"] == "data_analyst"]
    _check(len(analyst) == 1 and analyst[0]["observations"] == len(observed),
           f"Analyst observation count differs in {trace_path}")
    examples = {**analyst[0]["strongest"], **analyst[0]["weakest"]}
    _check(
        set(examples) <= set(observed)
        and all(isclose(entry["strength"], observed[mix]["strength_mpa"]) for mix, entry in examples.items()),
        f"Analyst examples differ from measured observations in {trace_path}",
    )
    _check(
        final["hypothesis"] == original["hypothesis"] and final["evidence_ids"] == cited
        and bool(cited) and len(cited) == len(set(cited)) and set(cited) <= set(examples),
        f"Final hypothesis or cited evidence was unavailable in {trace_path}",
    )
    hypothesis = {
        "statement": original["hypothesis"],
        "analyst_statement": original["analyst"],
        "cited_evidence": [observed[mix] for mix in cited],
        "outcome_status": NOT_EVALUABLE,
        "outcome_reason": "The archived free-text hypothesis has no preregistered structured endpoint, comparator and success rule.",
        "source": _location(root, trace_path, json_pointer=f"/tool_calls/{tool_index}/arguments"),
    }
    return hypothesis, found


def _round_observations(root: Path, ledger_path: Path, trace: dict, metric: dict,
                        proposed: list[dict], observed: dict, seed: int, arm: str,
                        step: int, budget: int) -> tuple[list[dict], list[dict], dict]:
    with ledger_path.open(encoding="utf-8", newline="") as ledger:
        batch = list(csv.DictReader(ledger))
    ids = [row["id"] for row in batch]
    final = trace["decision"]
    _check(
        len(ids) == len(set(ids)) == budget and set(ids).isdisjoint(observed)
        and ids == final["selected_ids"],
        f"Final selection differs from measurement ledger at {ledger_path}",
    )
    by_id = _unique_by(proposed, "recipe_id")
    _check(set(ids) <= set(by_id), f"Selection was not among proposed and scored recipes at {ledger_path}")
    _check([by_id[mix][1]["alias"] for mix in ids] == final["candidate_ids"],
           f"Final recipe IDs disagree with final aliases at {ledger_path}")
    greedy_ids = metric["contemporaneous_greedy_ids"]
    overlap = [mix for mix in ids if mix in greedy_ids]
    _check(
        len(greedy_ids) == len(set(greedy_ids)) == budget and set(greedy_ids).isdisjoint(observed)
        and isclose(len(overlap) / budget, metric["greedy_overlap_at_k"], abs_tol=1e-12),
        f"Same-state greedy reference disagrees at {ledger_path}",
    )
    chosen, revealed = [], []
    for position, row in enumerate(batch, 1):
        mix = row["id"]
        proposal = by_id[mix][1]
        experiment_number = (step - 1) * budget + position
        measured = float(row["strength_mpa"])
        _check(
            (int(row["seed"]), row["arm"], int(row["round"]), int(row["batch_position"])) ==
            (seed, arm, step, position)
            and int(row["experiment_number"]) == experiment_number
            and int(row["rf_rank"]) == proposal["rf_rank"]
            and isclose(float(row["prediction_mpa"]), proposal["prediction_mpa"], abs_tol=1e-10)
            and int(row["is_myopic_greedy_pick"]) == int(mix in greedy_ids)
            and int(row["top_hit"]) == int(measured >= metric["top_cutoff_mpa"]),
            f"Recorded iteration, score or outcome does not reconcile at {ledger_path} row {position}",
        )
        measurement = {
            "recipe_id": mix,
            "strength_mpa": measured,
            "revealed_round": step,
            "experiment_number": experiment_number,
            "source": _location(root, ledger_path, data_row=position),
        }
        revealed.append(measurement)
        chosen.append({
            **measurement,
            "alias": proposal["alias"],
            "prediction_mpa": proposal["prediction_mpa"],
            "rf_rank": proposal["rf_rank"],
            "actual_minus_predicted_mpa": measured - proposal["prediction_mpa"],
            "top_decile_hit": bool(int(row["top_hit"])),
            "in_same_state_greedy": mix in greedy_ids,
        })
    return chosen, revealed, {"ids": greedy_ids, "overlap_ids": overlap, "overlap_at_k": len(overlap) / budget}


def _reconstruct_arm(root: Path, archive: str, model: str, seed: int, arm: str,
                     rounds: int, budget: int) -> list[dict]:
    path = root / archive / "arms" / f"seed-{seed}" / arm
    initial_path = path / "initial_observed.json"
    split = _json(root / archive / "splits" / f"seed-{seed}.json")
    observed = {
        mix: {
            "recipe_id": mix,
            "strength_mpa": float(value["strength"]),
            "revealed_round": 0,
            "experiment_number": None,
            "source": _location(root, initial_path, json_pointer=f"/{mix}"),
        }
        for mix, value in _json(initial_path).items()
    }
    _check(bool(observed) and list(observed) == split["initial"],
           f"Initial observations differ from archived split at {initial_path}")
    pool = set(split["pool"])
    history = []
    for step in range(1, rounds + 1):
        stem = f"round-{step:02d}"
        trace_path = path / f"{stem}-agent.json"
        round_path = path / f"{stem}.json"
        ledger_path = path / f"{stem}-selections.csv"
        trace, metric = _json(trace_path), _json(round_path)
        _check(
            trace["status"] == "ok" and trace["mode"] == "live"
            and (trace["seed"], trace["round"], trace["budget"], trace["knowledge_augmented"]) ==
                (seed, step, budget, ARMS[arm] == "on")
            and (metric["seed"], metric["arm"], metric["round"], metric["budget"]) ==
                (seed, arm, step, budget),
            f"Archived arm or round mismatch at {trace_path}",
        )
        _check(
            {row["id"] for row in trace["menu"]} | set(metric["contemporaneous_greedy_ids"]) <= pool - set(observed),
            f"An eligible set contains unavailable recipes at {trace_path}",
        )
        hypothesis, proposed = _proposals(root, trace_path, trace, observed)
        chosen, revealed, greedy = _round_observations(
            root, ledger_path, trace, metric, proposed, observed, seed, arm, step, budget,
        )
        critic = trace["decision"]
        record = {
            "model": model,
            "model_id": trace["model"],
            "knowledge": ARMS[arm],
            "seed": seed,
            "round": step,
            "budget": budget,
            "observed_before": list(observed.values()),
            "hypothesis": hypothesis,
            "proposed": proposed,
            "chosen": chosen,
            "scientific_critic": critic["critic"],
            "selection_rationale": critic["selection_rationale"],
            "same_state_greedy": greedy,
            "newly_revealed": revealed,
            "round_best_mpa": max(row["strength_mpa"] for row in chosen),
            "incumbent_after_mpa": max(row["strength_mpa"] for row in [*observed.values(), *revealed]),
            "top_decile_cutoff_mpa": metric["top_cutoff_mpa"],
            "observed_after_count": len(observed) + len(revealed),
            "next_decision_change": None,
            "sources": {
                "model_decision": _location(root, trace_path, json_pointer="/decision"),
                "greedy_checkpoint": _location(root, round_path, json_pointer="/contemporaneous_greedy_ids"),
                "measurement_ledger": _location(root, ledger_path),
            },
        }
        _check(
            record["observed_after_count"] == metric["number_observed_after_refit"]
            and isclose(record["round_best_mpa"], metric["round_best_mpa"], abs_tol=1e-10)
            and isclose(record["incumbent_after_mpa"], metric["best_so_far_mpa"], abs_tol=1e-10)
            and sum(row["top_decile_hit"] for row in chosen) == metric["round_top_hits"],
            f"Round checkpoint differs from measured state at {round_path}",
        )
        if history:
            previous = history[-1]
            _link_next_decision(previous, record)
        observed.update((item["recipe_id"], item) for item in revealed)
        history.append(record)
    return history


def _link_next_decision(previous: dict, next_record: dict) -> None:
    before = previous["hypothesis"]
    after = next_record["hypothesis"]
    originally_cited = {item["recipe_id"] for item in before["cited_evidence"]}
    later_cited = {item["recipe_id"] for item in after["cited_evidence"]}
    previous["next_decision_change"] = {
        "round": next_record["round"],
        "hypothesis_wording_changed": before["statement"].split() != after["statement"].split(),
        "newly_cited_ids": sorted(later_cited - originally_cited),
        "no_longer_cited_ids": sorted(originally_cited - later_cited),
        "newly_revealed_and_cited": [row for row in previous["newly_revealed"] if row["recipe_id"] in later_cited],
        "greedy_overlap_at_k_change": (
            next_record["same_state_greedy"]["overlap_at_k"] - previous["same_state_greedy"]["overlap_at_k"]
        ),
        "next_selected_ids": [item["recipe_id"] for item in next_record["chosen"]],
        "interpretation": "Describes the recorded transition between rounds.",
        "next_decision_source": next_record["sources"]["model_decision"],
    }


def reconstruct_decisions(run_dir: Path) -> list[dict]:
    """Read completed frozen formal archives and reconstruct exactly 72 decisions."""
    run_dir = Path(run_dir).resolve()
    _check_complete(run_dir)
    return [
        record
        for model, archive in ARCHIVES.items()
        for seed in SEEDS
        for arm in ARMS
        for record in _reconstruct_arm(run_dir, archive, model, seed, arm, ROUNDS, BUDGET)
    ]


def summarize_decisions(records: list[dict]) -> dict:
    """Describe audited decisions, evidence provenance and next-round uptake."""
    cohort = Counter()
    outcomes = Counter()
    for record in records:
        label = record["model"], record["knowledge"]
        cohort[label] += 1
        outcomes[record["hypothesis"]["outcome_status"]] += 1
    transitions = [record["next_decision_change"] for record in records if record["next_decision_change"]]
    return {
        "audited_rounds": len(records),
        "recorded_proposals": sum(len(record["proposed"]) for record in records),
        "revealed_measurements": sum(len(record["newly_revealed"]) for record in records),
        "cited_observation_references": sum(len(record["hypothesis"]["cited_evidence"]) for record in records),
        "hypothesis_outcomes": dict(outcomes),
        "contemporaneous_greedy_selected": sum(len(record["same_state_greedy"]["overlap_ids"]) for record in records),
        "next_round_transitions": len(transitions),
        "transitions_citing_newly_revealed_observations": sum(bool(row["newly_revealed_and_cited"]) for row in transitions),
        "by_cell": [
            {"model": model, "knowledge": knowledge, "rounds": number}
            for (model, knowledge), number in cohort.items()
        ],
    }
