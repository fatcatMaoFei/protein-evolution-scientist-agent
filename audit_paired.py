"""P2: audit whether new concrete observations change a frozen DeepSeek policy."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from llm_agent import (
    MODEL,
    MODEL_CONFIG,
    TOOLS,
    AgentError,
    _digest,
    _ordered_menu,
    data_analyst,
    hypothesis_and_design_input,
    knowledge_context,
)
from nonbio import (
    ARMS,
    DEFAULT_DATA,
    FEATURES,
    PREDICTOR,
    RF_PARAMETERS,
    RIDGE_ALPHA,
    feature_view,
    make_split,
    read_dataset,
    train,
    write_csv,
    write_json,
)

SEEDS = (17, 29)
LLM_ARMS = ARMS[2:]
TARGET_ROUND = 2
BUDGET = 5
MENU_SIZE = 48


class AuditNotEvaluable(ValueError):
    """A design, source, decision, or scoring check failed."""


@dataclass(frozen=True)
class PairedRequest:
    seed: int
    arm: str
    eligible: dict
    initial: dict
    feedback: dict
    scores: dict
    uncertainty: dict
    menu: list[dict]
    config: dict
    archived_sha256: dict

    @property
    def augmented(self) -> bool:
        return self.arm == LLM_ARMS[1]

    @property
    def greedy(self) -> list[str]:
        return [item["id"] for item in self.menu[:BUDGET]]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditNotEvaluable(message)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_requests(baseline: Path, table: pd.DataFrame, source: dict) -> list[PairedRequest]:
    archive_config = _read(baseline / "config.json")
    status = _read(baseline / "status.json")
    expected_config = {
        "mode": "formal", "seeds": [17, 29, 43], "rounds": 3, "budget": BUDGET, "arms": list(ARMS),
        "predictor": PREDICTOR, "rf_parameters": RF_PARAMETERS, "ridge_alpha": RIDGE_ALPHA,
    }
    _require(all(archive_config.get(key) == value for key, value in expected_config.items()),
             "Archived model, seeds, arms, rounds, or budget differ from the registered benchmark.")
    _require(status == {"status": "complete", "mode": "formal", "arms": 12, "measurements": 180},
             "The original four-arm concrete benchmark is incomplete.")
    _require(_read(baseline / "source.json") == source, "Pinned measurement source differs from the archive.")
    features = feature_view(table)
    pairs = []
    for seed in SEEDS:
        split = make_split(list(table.index), seed)
        split_path = baseline / "splits" / f"seed-{seed}.json"
        _require(_read(split_path) == split, f"Archived split differs for seed {seed}.")
        initial = {
            key: {**features[key], "strength": float(table.loc[key, "strength"])}
            for key in split["initial"]
        }
        model = train(initial, seed)[PREDICTOR]
        for arm in LLM_ARMS:
            identity = f"seed {seed}, arm {arm}"
            folder = baseline / "arms" / f"seed-{seed}" / arm
            initial_path = folder / "initial_observed.json"
            _require(list(_read(initial_path).items()) == list(initial.items()),
                     f"Archived initial observations differ for {identity}.")
            feedback_path = folder / "round-01-selections.csv"
            trace_path = folder / "round-01-agent.json"
            log = _read(trace_path)
            measured = pd.read_csv(feedback_path).sort_values("batch_position")
            identifiers = measured.id.tolist()
            _require(
                len(measured) == BUDGET and len(set(identifiers)) == BUDGET
                and set(identifiers) <= set(split["pool"])
                and measured.batch_position.tolist() == list(range(1, BUDGET + 1))
                and set(measured.seed) == {seed} and set(measured.arm) == {arm} and set(measured["round"]) == {1},
                f"Archived first-round measured batch is invalid for {identity}.",
            )
            _require(
                log.get("status") == "ok" and log.get("mode") == "live" and log.get("model") == MODEL
                and log.get("seed") == seed and log.get("round") == 1 and log.get("budget") == BUDGET
                and log.get("knowledge_augmented") == (arm == LLM_ARMS[1])
                and log.get("decision", {}).get("selected_ids") == identifiers,
                f"Archived first-round live selection does not match measurements for {identity}.",
            )
            config = log["config"]
            _require(config.get("model") == MODEL and all(config.get(key) == value for key, value in MODEL_CONFIG.items()),
                     f"Archived DeepSeek settings differ from the shared agent for {identity}.")
            feedback = {
                row.id: {**features[row.id], "strength": float(row.strength_mpa)}
                for row in measured.itertuples(index=False)
            }
            # The five recorded feedback labels may be checked now. Next-round labels remain unused.
            _require(all(row["strength"] == float(table.loc[key, "strength"]) for key, row in feedback.items()),
                     f"Frozen feedback labels do not match the pinned measurements for {identity}.")
            eligible = {key: features[key] for key in sorted(set(split["pool"]).difference(feedback))}
            x = np.array([[row[name] for name in FEATURES] for row in eligible.values()])
            scores = dict(zip(eligible, map(float, model.predict(x)), strict=True))
            tree_predictions = np.array([tree.predict(x) for tree in model.estimators_])
            uncertainty = dict(zip(eligible, map(float, np.std(tree_predictions, axis=0)), strict=True))
            overlap = [item for item in log["menu"] if item["id"] in eligible]
            _require(
                bool(overlap) and all(
                    item["inputs"] == eligible[item["id"]] and math.isclose(item["prediction"], scores[item["id"]], abs_tol=1e-10, rel_tol=0)
                    and math.isclose(item["rf_tree_disagreement"], uncertainty[item["id"]], abs_tol=1e-10, rel_tol=0)
                    for item in overlap
                ),
                f"Frozen pre-feedback RF does not reproduce archived round-one estimates for {identity}.",
            )
            menu = _ordered_menu(eligible, scores, uncertainty, seed, TARGET_ROUND)
            _require(len(menu) == MENU_SIZE and {item["id"] for item in menu} <= set(eligible),
                     f"Shared shortlist is invalid for {identity}.")
            hashes = {
                str(path.relative_to(baseline)): _sha256(path)
                for path in (split_path, initial_path, feedback_path, trace_path)
            }
            pairs.append(PairedRequest(seed, arm, eligible, initial, feedback, scores, uncertainty, menu, config, hashes))
    return pairs


def _branch_path(output: Path, pair: PairedRequest, with_feedback: bool) -> Path:
    name = "with-feedback" if with_feedback else "no-feedback"
    return output / "pairs" / f"seed-{pair.seed}" / pair.arm / f"{name}-agent.json"


def _decide(
    output: Path, pair: PairedRequest, with_feedback: bool, selector: Callable[..., list[str]], replay: bool,
) -> dict:
    path = _branch_path(output, pair, with_feedback)
    observed = {**pair.initial, **pair.feedback} if with_feedback else pair.initial
    analysis = data_analyst(observed)
    knowledge = knowledge_context(observed, pair.menu) if pair.augmented else None
    expected = hypothesis_and_design_input(analysis, pair.menu, BUDGET, BUDGET + 2, TARGET_ROUND, knowledge)
    selected = selector(
        eligible=copy.deepcopy(pair.eligible), observed=copy.deepcopy(observed), scores=pair.scores.copy(),
        uncertainty=pair.uncertainty.copy(), k=BUDGET, seed=pair.seed, round_number=TARGET_ROUND,
        augmented=pair.augmented, log_path=path, model=pair.config["model"],
    )
    log = _read(path)
    identity = f"seed {pair.seed}, arm {pair.arm}, feedback={with_feedback}"
    _require(
        log.get("status") == "ok" and log.get("mode") == ("replay" if replay else "live")
        and log.get("seed") == pair.seed and log.get("round") == TARGET_ROUND
        and log.get("budget") == BUDGET and log.get("knowledge_augmented") == pair.augmented
        and log.get("model") == pair.config["model"] and log.get("config") == pair.config and log.get("menu") == pair.menu,
        f"Decision trace does not match the frozen request for {identity}.",
    )
    calls = log.get("calls", [])
    _require(
        len(calls) >= 2 and log.get("live_generation_count") == (0 if replay else len(calls))
        and calls[0].get("stage") == "proposal" and calls[-1].get("stage") == "final_selection"
        and calls[0].get("request", {}).get("messages") == expected and calls[0]["request"].get("tools") == TOOLS
        and all(
            call.get("request_sha256") == _digest(call.get("request"))
            and call.get("response_sha256") == _digest(call.get("response"))
            and call.get("response", {}).get("model") == pair.config["model"]
            and call.get("response", {}).get("response_id")
            for call in calls
        ),
        f"Requests, model provenance, or provider responses do not validate for {identity}.",
    )
    tools = log.get("tool_calls", [])
    _require(
        len(tools) == 3 and tools[0].get("result") == analysis
        and tools[1].get("tool") == "evaluate_concrete_shortlist" and tools[1].get("trigger") == "model_tool_call"
        and tools[2].get("result", {}).get("eligible_unique_budget_verified") is True,
        f"Analyst, native evaluator, or critic evidence does not validate for {identity}.",
    )
    decision = log.get("decision", {})
    _require(
        len(selected) == BUDGET and len(set(selected)) == BUDGET
        and set(selected) <= {item["id"] for item in pair.menu}
        and decision.get("selected_ids") == selected
        and set(decision.get("evidence_ids", [])) <= set(observed),
        f"Selected candidates or cited observations are invalid for {identity}.",
    )
    return {
        "selected": selected, "decision": decision, "trace": path.relative_to(output).as_posix(),
        "trace_sha256": _sha256(path), "first_request_sha256": calls[0]["request_sha256"],
    }


def _score(pair: PairedRequest, before: dict, after: dict, table: pd.DataFrame) -> tuple[dict, list[dict]]:
    # This function receives the hidden next-round labels only after eight accepted decisions exist.
    menu_ids = {item["id"] for item in pair.menu}
    measured = {key: float(table.loc[key, "strength"]) for key in menu_ids}
    menu_best = max(measured.values())
    before_ids, after_ids = before["selected"], after["selected"]
    before_best = max(measured[key] for key in before_ids)
    after_best = max(measured[key] for key in after_ids)
    greedy_best = max(measured[key] for key in pair.greedy)
    overlap = len(set(before_ids).intersection(after_ids))
    row = {
        "seed": pair.seed, "arm": pair.arm, "round": TARGET_ROUND, "budget": BUDGET,
        "eligible_n": len(pair.eligible), "menu_n": len(pair.menu),
        "no_feedback_observations": len(pair.initial),
        "with_feedback_observations": len(pair.initial) + len(pair.feedback),
        "selection_overlap": overlap,
        "selection_jaccard": overlap / (2 * BUDGET - overlap),
        "set_changed": overlap < BUDGET, "order_changed": before_ids != after_ids,
        "menu_best_mpa": menu_best, "model_only_best_mpa": greedy_best,
        "no_feedback_best_mpa": before_best, "with_feedback_best_mpa": after_best,
        "delta_best_mpa": after_best - before_best,
        "no_feedback_regret_mpa": menu_best - before_best,
        "with_feedback_regret_mpa": menu_best - after_best,
        "regret_reduction_mpa": after_best - before_best,
        "no_feedback_greedy_overlap": len(set(before_ids).intersection(pair.greedy)),
        "with_feedback_greedy_overlap": len(set(after_ids).intersection(pair.greedy)),
        "no_feedback_best_delta_vs_greedy_mpa": before_best - greedy_best,
        "with_feedback_best_delta_vs_greedy_mpa": after_best - greedy_best,
        "new_feedback_evidence_count": len(set(after["decision"]["evidence_ids"]).intersection(pair.feedback)),
        "no_feedback_evidence_ids": json.dumps(before["decision"]["evidence_ids"]),
        "with_feedback_evidence_ids": json.dumps(after["decision"]["evidence_ids"]),
        "no_feedback_hypothesis": before["decision"]["hypothesis"],
        "with_feedback_hypothesis": after["decision"]["hypothesis"],
        "no_feedback_trace": before["trace"], "with_feedback_trace": after["trace"],
        "no_feedback_trace_sha256": before["trace_sha256"], "with_feedback_trace_sha256": after["trace_sha256"],
        "no_feedback_request_sha256": before["first_request_sha256"],
        "with_feedback_request_sha256": after["first_request_sha256"],
        "frozen_menu_sha256": _digest(pair.menu), "frozen_model_config_sha256": _digest(pair.config),
    }
    selections = [
        {
            "seed": pair.seed, "arm": pair.arm, "feedback_branch": name,
            "batch_position": index, "id": key, "strength_mpa": measured[key], "prediction_mpa": pair.scores[key],
            "rf_rank": next(item["rank"] for item in pair.menu if item["id"] == key),
        }
        for name, identifiers in (("no_feedback", before_ids), ("with_feedback", after_ids), ("greedy", pair.greedy))
        for index, key in enumerate(identifiers, 1)
    ]
    return row, selections


def run(
    output: Path, *, baseline: Path, selector: Callable[..., list[str]], freeze_path: Path,
    data_path: Path = DEFAULT_DATA, workers: int = 4, replay: bool = False,
) -> dict:
    """Execute the locked paired audit, or emit NOT EVALUABLE and omit scored outcomes."""
    output, baseline = output.resolve(), baseline.resolve()
    if output == baseline or baseline in output.parents:
        raise ValueError("P2 output must be outside the original archive.")
    if workers < 1:
        raise ValueError("Workers must be positive.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    with status_path.open("x", encoding="utf-8") as stream:
        json.dump({"status": "running", "started_utc": _utc()}, stream)
    futures = {}
    stage = "preflight"
    try:
        freeze = freeze_path.read_bytes()
        _require(bool(freeze), "An immutable P2 protocol is required.")
        table, source = read_dataset(data_path)
        pairs = _build_requests(baseline, table, source)
        _require(len({json.dumps(pair.config, sort_keys=True) for pair in pairs}) == 1,
                 "Archived DeepSeek configuration differs across pairs.")
        freeze_name = "protocol-freeze" + freeze_path.suffix
        (output / freeze_name).write_bytes(freeze)
        write_json(output / "config.json", {
            "study": "P2 matched feedback intervention", "model": MODEL, "model_config": pairs[0].config,
            "seeds": SEEDS, "arms": LLM_ARMS, "round": TARGET_ROUND, "budget": BUDGET, "menu_size": MENU_SIZE,
            "retrains_between_branches": False, "protocol_file": freeze_name,
            "protocol_sha256": hashlib.sha256(freeze).hexdigest(), "source_sha256": source["sha256"],
            "mode": "replay" if replay else "live",
            "baseline_manifest": {
                name: _sha256(baseline / name) for name in ("config.json", "source.json", "status.json")
            },
        })
        for pair in pairs:
            stem = _branch_path(output, pair, False).parent
            write_json(stem / "design.json", {
                "seed": pair.seed, "arm": pair.arm, "no_feedback_observed_ids": list(pair.initial),
                "added_feedback_ids": list(pair.feedback), "common_eligible_ids": list(pair.eligible),
                "frozen_score_and_uncertainty_sha256": _digest([pair.scores, pair.uncertainty]),
                "shared_greedy_ids": pair.greedy, "frozen_menu": pair.menu, "frozen_menu_sha256": _digest(pair.menu),
                "baseline_input_sha256": pair.archived_sha256,
            })
        stage = "decision-validation"
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for pair in pairs:
                for branch in (False, True):
                    futures[pair.seed, pair.arm, branch] = executor.submit(_decide, output, pair, branch, selector, replay)
        accepted = {key: future.result() for key, future in futures.items()}
        _require(len(accepted) == 2 * len(pairs), "Every pair requires two accepted decisions.")
        stage = "independent-scoring"
        scored_at = _utc()
        outcomes = [
            _score(pair, accepted[pair.seed, pair.arm, False], accepted[pair.seed, pair.arm, True], table)
            for pair in pairs
        ]
        write_csv(output / "pairs.csv", [row for row, _ in outcomes])
        write_csv(output / "selections.csv", [row for _, selection in outcomes for row in selection])
        status = {
            "status": "complete", "pairs": len(pairs), "decisions": len(accepted),
            "scoring_started_utc": scored_at, "completed_utc": _utc(),
        }
    except (AgentError, OSError, ValueError, KeyError, IndexError, AttributeError, TypeError, RuntimeError) as error:
        status = {
            "status": "NOT EVALUABLE", "stage": stage, "reason": str(error),
            "validated_decisions": sum(future.exception() is None for future in futures.values()),
        }
    write_json(status_path, status)
    return status
