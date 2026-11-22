"""Audit an import-and-continue GFP research record without an oracle or API."""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

from gfp_continue import RECOMMENDATION_COLUMNS, _load_state, _sha, load_csv
from gfp_data import make_variant
from llm_agent import _digest

_ARTIFACTS = {"recommendations.csv", "imported_measurements.csv", "agent_request.json", "agent.json"}
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _ids(values: list[str]) -> set[str]:
    if (not isinstance(values, list) or not all(isinstance(key, str) and make_variant(key).id == key for key in values)
            or len(values) != len(set(values))):
        raise ValueError("Invalid or repeated variant in GFP state history")
    return set(values)


def _legacy_predecessor(path: Path, expected: str) -> Path | None:
    """Search immediate neighboring rounds for old states without a predecessor link."""
    for folder in sorted(path.parent.parent.iterdir()):
        candidate = folder / "state.json"
        if folder != path.parent and not folder.is_symlink() and candidate.is_file() and not candidate.is_symlink():
            if _sha(candidate.read_bytes()) == expected:
                return candidate
    return None


def _history(path: Path) -> tuple[list[tuple[Path, dict]], str | None]:
    states = []
    expected_hash = None
    expected_round = None
    visited = set()
    while path is not None:
        if path in visited:
            raise ValueError("Repeated path in GFP state lineage")
        visited.add(path)
        state, digest = _load_state(path)
        number = state["round"]
        if (expected_hash is not None and digest != expected_hash
                or expected_round is not None and number != expected_round):
            raise ValueError("GFP predecessor hash or round number does not match")
        if type(number) is not int or number < 1:
            raise ValueError("Invalid GFP research round number")
        states.append((path, state))
        expected_hash = state["prior_state_sha256"]
        expected_round = number - 1
        link = state.get("prior_state_path")
        if number == 1:
            if expected_hash is not None or link is not None:
                raise ValueError("An initial GFP state cannot have a predecessor")
            return states[::-1], None
        if not isinstance(expected_hash, str) or _SHA256.fullmatch(expected_hash) is None:
            raise ValueError("Invalid predecessor digest in GFP state")
        if link is None:
            path = _legacy_predecessor(path, expected_hash)
        else:
            if not isinstance(link, str) or Path(link).is_absolute() or Path(link).suffix != ".json":
                raise ValueError("Invalid relative predecessor path in GFP state")
            path = (path.parent / link).resolve()
            if not path.is_file():
                path = None
    return states[::-1], f"Missing predecessor for round {expected_round + 1} (SHA-256 {expected_hash})"


def _files(path: Path, state: dict) -> dict[str, Path]:
    recorded = state.get("artifacts")
    if recorded is None:
        return {name: path.parent / name for name in _ARTIFACTS
                if (path.parent / name).is_file() and not (path.parent / name).is_symlink()}
    if (not isinstance(recorded, dict) or not set(recorded) <= _ARTIFACTS
            or not {"recommendations.csv", "imported_measurements.csv"} <= set(recorded)
            or ("agent_request.json" in recorded) != ("agent.json" in recorded)):
        raise ValueError("Invalid artifact manifest in GFP state")
    files = {name: path.parent / name for name in recorded}
    for name, artifact in files.items():
        if not artifact.is_file() or artifact.is_symlink() or _sha(artifact.read_bytes()) != recorded[name]:
            raise ValueError(f"GFP artifact digest does not match for round {state['round']} {name}")
    return files


def _recommendations(path: Path, proposed: list[str]) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        schema = set(RECOMMENDATION_COLUMNS) | {"origin"}
        required = {"mutation", "sequence", "predicted_fitness", "measured_fitness", "status", "decision"}
        fields = reader.fieldnames or []
        if not required <= set(fields) <= schema or len(fields) != len(set(fields)):
            raise ValueError("Invalid recommendations header in GFP research record")
        records = list(reader)
    if [row["mutation"] for row in records] != proposed:
        raise ValueError("GFP recommendations differ from the order in the saved research state")
    for row in records:
        if (None in row or None in row.values() or row["sequence"] != make_variant(row["mutation"]).sequence
                or row["status"] != "prediction_only" or row["measured_fitness"] != ""):
            raise ValueError("Invalid sequence or measured field in GFP recommendation")
        score = float(row["predicted_fitness"])
        if not math.isfinite(score):
            raise ValueError("Nonfinite prediction in GFP recommendation")
        row["predicted_fitness"] = score
        row["measured_fitness"] = None
    return records


def _agent_trace(files: dict[str, Path], state: dict, observed: dict[str, float], recommendations: list[dict]) -> dict:
    if "agent.json" not in files:
        ran_baseline = recommendations and all(row["decision"] == "rf" for row in recommendations)
        return {"status": "MODEL_ONLY" if ran_baseline else "NOT AVAILABLE", "citations": []}
    trace = json.loads(files["agent.json"].read_bytes())
    decision = trace["decision"]
    feedback = set(state["new_measurements"]) if state["round"] > 1 else set()
    ids = decision["evidence_ids"]
    aliases = decision["evidence_aliases"]
    aliases_to_rows = trace["observed_evidence"]
    recorded_feedback = set(trace["latest_feedback_ids"])
    if (trace["status"] != "ok" or trace["round"] != state["round"]
            or trace["budget"] != len(state["proposed"]) or decision["selected_ids"] != state["proposed"]
            or not recorded_feedback <= feedback or not _ids(ids) <= observed.keys()
            or ids != [aliases_to_rows[alias]["id"] for alias in aliases]
            or any(row["id"] not in observed or row["fitness"] != observed[row["id"]]
                   for row in aliases_to_rows.values())):
        raise ValueError("GFP agent log or cited measured evidence does not match the saved state")
    selected = decision["selected_candidates"]
    if [choice["id"] for choice in selected] != state["proposed"] or (recommendations and any(
        (choice["sequence"], choice["prediction"]) != (row["sequence"], row["predicted_fitness"])
        for choice, row in zip(selected, recommendations, strict=True)
    )):
        raise ValueError("GFP logged agent choices differ from exported sequences or predictions")
    if "agent_request.json" in files:
        request = json.loads(files["agent_request.json"].read_bytes())
        request_observations = {row["id"]: row["fitness"] for row in request["observed"]}
        if (_digest(request) != trace["input_sha256"] or request_observations != observed
                or len(request["observed"]) != len(observed) or request["menu"] != trace["menu"]
                or request["latest_feedback_ids"] != trace["latest_feedback_ids"]
                or request["round_number"] != state["round"] or request["budget"] != trace["budget"]):
            raise ValueError("GFP pre-decision request does not match its log or visible observation history")
    scored_calls = [call["result"] for call in trace["tool_calls"] if call["tool"] == "score_gfp_designs"]
    if len(scored_calls) != 1:
        raise ValueError("Missing or repeated score tool record in GFP decision")
    scored = {row["alias"]: row for row in scored_calls[0]}
    if (any({"fitness", "measured_fitness"} & row.keys() for row in scored.values()) or any(
        choice["alias"] not in scored or any(choice[key] != scored[choice["alias"]][key]
                                              for key in ("mutations", "sequence", "prediction"))
        for choice in selected
    )):
        raise ValueError("GFP selected predictions differ from the native score tool record")
    return {
        "status": "VERIFIED_REQUEST" if "agent_request.json" in files else "VERIFIED_LOGGED_EVIDENCE",
        "model": trace["model"], "backend": trace["backend"],
        "hypothesis": decision["hypothesis"], "rationale": decision["selection_rationale"], "critic": decision["critic"],
        "citations": [{"mutant": key, "fitness": observed[key], "new_in_round": key in feedback} for key in ids],
        "feedback_in_prompt": trace["latest_feedback_ids"],
        "scored_designs": scored_calls[0],
    }


def _trace(states: list[tuple[Path, dict]]) -> list[dict]:
    previous = previous_observations = None
    pending_predictions = {}
    traces = []
    for path, state in states:
        observed = {row["mutant"]: row["fitness"] for row in state["observations"]}
        new = _ids(state["new_measurements"])
        proposed = _ids(state["proposed"])
        pending = _ids(state["outstanding"])
        if not new <= observed.keys() or pending & observed.keys() or not proposed <= pending:
            raise ValueError("GFP imports, observations and outstanding proposals are inconsistent")
        prior_pending = set(previous["outstanding"]) if previous else set()
        if previous:
            if (observed != (previous_observations | {key: observed[key] for key in new})
                    or new & previous_observations.keys() or proposed & prior_pending
                    or pending != ((prior_pending - new) | proposed)):
                raise ValueError("Invalid GFP import or outstanding state transition")
        elif state["round"] == 1 and (new != observed.keys() or pending != proposed):
            raise ValueError("Invalid initial GFP observations or outstanding state")
        files = _files(path, state)
        recommendations = _recommendations(files["recommendations.csv"], state["proposed"]) if "recommendations.csv" in files else []
        if "imported_measurements.csv" in files:
            receipt = load_csv(files["imported_measurements.csv"], measured=True)
            if {key: row.fitness for key, row in receipt.items()} != {key: observed[key] for key in new}:
                raise ValueError("GFP measurement receipt does not match the imported state delta")
        returned = []
        for key in state["new_measurements"]:
            record = {"mutant": key, "fitness": observed[key]}
            if state["round"] == 1:
                record["kind"] = "initial_observation"
            elif key in prior_pending:
                record["kind"] = "previously_recommended"
                forecast = pending_predictions.pop(key, None)
                if forecast is not None:
                    record.update(forecast)
                    record["residual"] = observed[key] - record["predicted_fitness"]
            else:
                record["kind"] = "external_measurement" if previous else "origin_unavailable"
            returned.append(record)
        for row in recommendations:
            pending_predictions[row["mutation"]] = {
                "recommended_round": state["round"], "predicted_fitness": row["predicted_fitness"],
            }
        prior_proposed = set(previous["proposed"]) if previous else set()
        comparison = ({"top_set_changed": proposed != prior_proposed,
                       "shared_count": len(proposed & prior_proposed),
                       "jaccard": len(proposed & prior_proposed) / len(proposed | prior_proposed)}
                      if previous and (proposed or prior_proposed) else None)
        traces.append({
            "round": state["round"], "observations": state["observations"],
            "imported_measurements": returned, "import_receipt_verified": "imported_measurements.csv" in files,
            "proposed_ids": state["proposed"], "recommendations": recommendations,
            "recommendation_log_available": "recommendations.csv" in files,
            "pending_measurements": state["outstanding"], "change_vs_previous_round": comparison,
            "decision": _agent_trace(files, state, observed, recommendations),
        })
        previous, previous_observations = state, observed
    return traces


def run_user_audit(state_path: Path, output: Path) -> dict:
    """Read the saved user history and write a separate descriptive decision audit."""
    states, missing = _history(state_path.resolve(strict=True))
    output = output.resolve()
    if any(output.is_relative_to(path.parent) for path, _ in states):
        raise ValueError("Audit output must be outside every audited round directory")
    traces = _trace(states)
    output.mkdir(parents=True, exist_ok=False)
    imports = [record for row in traces for record in row["imported_measurements"] if row["round"] > 1]
    comparisons = [row["change_vs_previous_round"] for row in traces if row["change_vs_previous_round"] is not None]
    complete = missing is None and all(row["recommendation_log_available"] and row["decision"]["status"] != "NOT AVAILABLE"
                                       for row in traces)
    result = {
        "status": "complete" if complete else "partial", "history_gap": missing,
        "rounds": [row["round"] for row in traces], "subsequent_measurements": len(imports),
        "returned_recommendations": sum(record["kind"] == "previously_recommended" for record in imports),
        "external_measurements": sum(record["kind"] == "external_measurement" for record in imports),
        "prediction_residuals": sum("residual" in record for record in imports),
        "verified_citations": sum(len(row["decision"]["citations"]) for row in traces),
        "recommendation_comparisons": len(comparisons),
        "recommendation_sets_changed": sum(row["top_set_changed"] for row in comparisons),
        "paired_status": "NOT EVALUABLE", "feedback_gain": None,
        "paired_reason": "No matched counterfactual selection and outcome recorded in the user workflow.",
    }
    with (output / "decision_traces.jsonl").open("x", encoding="utf-8") as stream:
        for trace in traces:
            stream.write(json.dumps(trace, ensure_ascii=False, allow_nan=False) + "\n")
    (output / "audit_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# GFP 用户研究决策审计", "",
        f"读取 {len(traces)} 轮。核对 {result['verified_citations']} 条有来源的证据引用，"
        f"追溯 {result['returned_recommendations']} 条已推荐变体的返回实测，"
        f"其中 {result['prediction_residuals']} 条具有原始预测和残差。", "",
        "| 轮次 | 累计已观测 | 本轮导入 | 推荐 | 引用 | 与上轮共享推荐 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for trace in traces:
        overlap = trace["change_vs_previous_round"]
        lines.append(
            f"| {trace['round']} | {len(trace['observations'])} | {len(trace['imported_measurements'])} | "
            f"{len(trace['proposed_ids'])} | {len(trace['decision']['citations'])} | "
            f"{overlap['shared_count'] if overlap is not None else '不可比'} |"
        )
    if not complete:
        lines.extend(["", "历史记录不完整。" + (missing or "部分轮次缺少推荐导出或决策日志。")])
    lines.extend([
        "", "反馈收益为 **NOT EVALUABLE**。记录中没有受控反事实和对应结果。连续推荐集会因为待测排除规则而改变，"
        "引用与推荐变化只描述记录中的行为。", "",
        "[逐轮轨迹](decision_traces.jsonl) · [审计汇总](audit_summary.json)",
    ])
    (output / "audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
