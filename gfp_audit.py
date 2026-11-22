"""Trace and experimentally audit feedback use in the frozen GFP replay."""
from __future__ import annotations

import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

import pandas as pd

from gfp_evolution import build_menu, make_agent_request, prediction_view
from llm_agent import _digest

BACKENDS = ("deepseek", "sonnet", "gpt", "qwen")
LLM_ARMS = ("C_llm", "D_knowledge_llm")
CHECKPOINTS = tuple((seed, arm) for seed in (17, 29) for arm in LLM_ARMS)
BUDGET = 5


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, content) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as out:
        json.dump(content, out, indent=2, ensure_ascii=False, allow_nan=False)
        out.write("\n")


def frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False)


def _arm(root: Path, seed: int, arm: str) -> Path:
    return root / "arms" / f"seed-{seed}" / arm


def _studies(run_dir: Path) -> dict[str, Path]:
    if (run_dir / "config.json").is_file():
        return {read(run_dir / "config.json")["backend"]: run_dir}
    return {backend: run_dir / "formal" / backend for backend in BACKENDS}


def _freeze(studies: dict[str, Path]) -> dict[str, str]:
    manifest = {}
    for backend, parent in sorted(studies.items()):
        config = read(parent / "config.json")
        if read(parent / "status.json")["status"] != "complete" or config["domain"] != "gfp_fluorescence":
            raise ValueError(f"GFP results incomplete for {backend}")
        for path in sorted(parent.rglob("*")):
            if path.is_file():
                key = str(Path(backend) / path.relative_to(parent))
                manifest[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return manifest


def _selected(data, seed: int, arm: str, number: int):
    subset = data[(data.seed == seed) & (data.arm == arm) & (data["round"] == number)].sort_values("batch_position")
    rows = subset.to_dict(orient="records")
    if len(rows) != BUDGET or len(set(row["id"] for row in rows)) != BUDGET:
        raise ValueError("Incomplete or repeated observed GFP batch")
    return rows


def trace_audit(dataset, studies: dict[str, Path]) -> list[dict]:
    trajectory = []
    for backend, parent in studies.items():
        config = read(parent / "config.json")
        data = frame(parent / "selections.csv")
        for seed in config["seeds"]:
            split = read(parent / "splits" / f"seed-{seed}.json")
            for arm in LLM_ARMS:
                if arm not in config["arms"]:
                    continue
                arm_dir = _arm(parent, seed, arm)
                observed = read(arm_dir / "initial_observed.json")
                if set(observed) != set(split["initial"]):
                    raise ValueError("Initial observation archive does not match the split")
                previous = set()
                prior_record = None
                for number in range(1, config["rounds"] + 1):
                    stem = f"round-{number:02d}"
                    request = read(arm_dir / f"{stem}-request.json")
                    agent = read(arm_dir / f"{stem}-agent.json")
                    if agent["status"] != "ok" or agent["input_sha256"] != _digest(request):
                        raise ValueError("GFP trace does not reproduce its archived pre-selection context")
                    if {row["id"]: row["fitness"] for row in request["observed"]} != observed:
                        raise ValueError("GFP request exposes an observation outside its own revealed history")
                    if set(row["id"] for row in request["menu"]) - (set(split["pool"]) - observed.keys()):
                        raise ValueError("GFP request includes an ineligible candidate")
                    if set(request.get("latest_feedback_ids", [])) != previous:
                        raise ValueError("GFP previous feedback IDs do not match the revealed batch")
                    picks = _selected(data, seed, arm, number)
                    ids = [row["id"] for row in picks]
                    if ids != agent["decision"]["selected_ids"]:
                        raise ValueError("GFP response and measurement ledger disagree")
                    if set(ids) & observed.keys() or any(
                        row["fitness"] != dataset.scores[row["id"]] for row in picks
                    ):
                        raise ValueError("Selected GFP labels do not reproduce from the measured archive")
                    citations = agent["decision"]["evidence_ids"]
                    if not set(citations) <= observed.keys():
                        raise ValueError("A GFP claim refers to unavailable measurements")
                    tool = next(item for item in agent["tool_calls"] if item["tool"] == "score_gfp_designs")
                    row = {
                        "backend": backend, "seed": seed, "arm": arm, "round": number,
                        "observations_before": len(observed), "prior_feedback_ids": sorted(previous),
                        "cited_measurements": [{"id": key, "fitness": observed[key]} for key in citations],
                        "cited_new_feedback": sorted(previous & set(citations)),
                        "hypothesis": agent["decision"]["hypothesis"],
                        "hypothesis_test_status": "NOT EVALUABLE: free-text hypothesis has no locked success threshold",
                        "proposals_and_scores": tool["result"],
                        "selected": picks, "criticism": agent["decision"]["critic"],
                        "selection_rationale": agent["decision"]["selection_rationale"],
                        "greedy_overlap": sum(bool(row["is_contemporaneous_greedy"]) for row in picks) / BUDGET,
                        "archive_trace": str(Path(backend) / (arm_dir / f"{stem}-agent.json").relative_to(parent)),
                    }
                    if prior_record is not None:
                        prior_record["next_decision"] = {
                            "ids": ids, "hypothesis": row["hypothesis"],
                            "referenced_this_feedback": row["cited_new_feedback"],
                        }
                    trajectory.append(row)
                    prior_record = row
                    previous = set(ids)
                    observed.update({record["id"]: record["fitness"] for record in picks})
    return trajectory


def _checkpoints(dataset, parent: Path):
    data = frame(parent / "selections.csv")
    points = []
    for seed, arm in CHECKPOINTS:
        split = read(parent / "splits" / f"seed-{seed}.json")
        initial = read(_arm(parent, seed, arm) / "initial_observed.json")
        actual = _selected(data, seed, arm, 1)
        measured = {row["id"]: row["fitness"] for row in actual}
        eligible = {key: dataset.variants[key] for key in split["pool"] if key not in measured}
        scores, uncertainty = prediction_view(dataset, initial, eligible, seed)
        menu = build_menu(eligible, scores, uncertainty, seed=seed, round_number=2)
        ranked = [row["id"] for row in sorted(menu, key=lambda row: row["rank"])]
        no_feedback = make_agent_request(dataset, initial, menu, seed=seed, round_number=2,
                                         budget=BUDGET, knowledge=arm == LLM_ARMS[1])
        with_feedback = make_agent_request(dataset, initial | measured, menu, seed=seed, round_number=2,
                                           budget=BUDGET, knowledge=arm == LLM_ARMS[1],
                                           latest_feedback_ids=list(measured))
        fixed = {key: value for key, value in no_feedback.items() if key not in ("observed", "latest_feedback_ids")}
        if fixed != {key: value for key, value in with_feedback.items() if key not in ("observed", "latest_feedback_ids")}:
            raise ValueError("The paired GFP intervention changed fixed inputs")
        points.append({"seed": seed, "arm": arm, "no_feedback": no_feedback, "with_feedback": with_feedback,
                       "model_only": ranked[:BUDGET], "menu_sha256": _digest(menu)})
    return points


def _submit(point, branch: str, output: Path, replay: Path | None):
    from gfp_agent import select_candidates

    relative = Path("pairs") / f"seed-{point['seed']}" / point["arm"] / (branch + ".json")
    ids = select_candidates(point[branch], log_path=output / relative, backend="deepseek",
                            replay_path=replay / relative if replay else None)
    selected = read(output / relative)
    menu = {row["id"] for row in point[branch]["menu"]}
    if len(ids) != BUDGET or len(set(ids)) != BUDGET or not set(ids) <= menu:
        raise ValueError("Paired GFP decision is outside its frozen roster")
    return ids, selected


def _score_pair(dataset, point, choices) -> tuple[dict, list[dict]]:
    menu = {row["id"]: row for row in point["no_feedback"]["menu"]}
    best_available = max(dataset.scores[key] for key in menu)
    no, with_new = choices
    baseline = point["model_only"]
    measurements = []
    stats = {}
    for branch, selected in (("no_feedback", no[0]), ("with_feedback", with_new[0]), ("model_only", baseline)):
        values = [dataset.scores[key] for key in selected]
        stats[branch] = (max(values), mean(values))
        measurements.extend(
            {"seed": point["seed"], "arm": point["arm"], "condition": branch,
             "position": index, "id": key, "mutations": menu[key]["mutations"],
             "sequence": menu[key]["sequence"], "prediction": menu[key]["prediction"], "fitness": value}
            for index, (key, value) in enumerate(zip(selected, values, strict=True), 1)
        )
    feedback_cites = set(with_new[1]["decision"]["evidence_ids"]) & set(point["with_feedback"]["latest_feedback_ids"])
    overlap = len(set(no[0]) & set(with_new[0]))
    summary = {
        "seed": point["seed"], "arm": point["arm"], "menu_sha256": point["menu_sha256"],
        "set_changed": overlap < BUDGET, "top_k_overlap": overlap / BUDGET,
        "order_changed": no[0] != with_new[0],
        "best_delta": stats["with_feedback"][0] - stats["no_feedback"][0],
        "mean_delta": stats["with_feedback"][1] - stats["no_feedback"][1],
        "best_versus_model_only": stats["with_feedback"][0] - stats["model_only"][0],
        "feedback_model_only_overlap": len(set(with_new[0]) & set(baseline)) / BUDGET,
        "cited_new_feedback": ",".join(sorted(feedback_cites)),
        "best_available": best_available,
    }
    for branch, (best, avg) in stats.items():
        summary[f"{branch}_best"] = best
        summary[f"{branch}_mean"] = avg
        summary[f"{branch}_regret"] = best_available - best
    return summary, measurements


def _table(path: Path, records: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def _plot_pairs(output: Path, pairs: pd.DataFrame) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(7.5, 3.2), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    columns = ("no_feedback_best", "with_feedback_best", "model_only_best")
    for row in pairs.itertuples():
        label = "Knowledge ON" if row.arm == LLM_ARMS[1] else "Knowledge OFF"
        axes.plot(range(3), [getattr(row, key) for key in columns], marker="o", label=f"Seed {row.seed} · {label}")
    axes.set_xticks(range(3), ["No feedback", "Real feedback", "Model-Only"])
    axes.set_ylabel("Best measured fluorescence (log10 score)")
    axes.grid(axis="y", alpha=0.3)
    axes.legend(loc="best", fontsize=8)
    destination = output / "figures" / "paired-feedback.png"
    destination.parent.mkdir(exist_ok=True)
    figure.savefig(destination, dpi=160)


def _report(output: Path, result: dict) -> None:
    row = result["trajectory"]
    lines = ["# GFP 科研决策审计", "",
             f"共重建 {row['decisions']} 轮。{row['feedback_cited_in_next']} / {row['transitions']} 次轮间过渡"
             "中，下一轮理由引用了可追溯的新测量。引用本身只证明可追溯性。",
             f"LLM 选择与同轮 RF Top-5 的平均重叠为 {row['own_greedy_overlap']:.1%}。"
             "自由文本假设缺少预先固定的成败判据，判为 NOT EVALUABLE。", ""]
    if result["paired_status"] == "complete":
        p = result["paired"]
        pairs = frame(output / "paired_feedback_results.csv")
        improved = int((pairs.best_delta > 0).sum())
        lowest_mean = pairs.loc[pairs.mean_delta.idxmin()]
        label = "知识开启" if lowest_mean.arm == LLM_ARMS[1] else "知识关闭"
        _plot_pairs(output, pairs)
        lines.extend([
            f"固定 RF 权重、候选菜单、预算和模型，在 {p['pairs']} 个检查点中，"
            f"{p['sets_changed']} 个 Top-5 集合改变，平均重叠比例 {p['mean_overlap']:.1%}。",
            f"有反馈减无反馈的单批最佳实测值平均为 {p['mean_best_delta']:+.4f} 分，"
            f"整批平均实测值的差为 {p['mean_batch_delta']:+.4f} 分。"
            f"有反馈相对 Model-Only 的最佳值差为 {p['mean_best_vs_greedy']:+.4f} 分。",
            f"单批最佳值在 {improved}/{p['pairs']} 个检查点提高。"
            f"种子 {lowest_mean.seed}（{label}）的批次均值差为 {lowest_mean.mean_delta:+.4f} 分。"
            "每个检查点每个显示条件各一次决策，差值包含可能的单次响应波动。",
            "", "[配对图](figures/paired-feedback.png) · [配对表](paired_feedback_results.csv) · "
            "[选择与测量](paired_measurements.csv)",
        ])
    else:
        lines.append(f"配对审计：{result['paired_status']}。{result.get('paired_reason', '')}")
    lines.extend(["", "[逐轮轨迹](decision_traces.jsonl) · [审计收据](audit_summary.json)"])
    (output / "audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(dataset, run_dir: Path, output: Path, *, paired: bool, replay: Path | None, workers: int) -> dict:
    studies = _studies(run_dir.resolve())
    if any(output.resolve().is_relative_to(path) for path in studies.values()):
        raise ValueError("Audit must write outside the frozen runs")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    before = _freeze(studies)
    trajectory = trace_audit(dataset, studies)
    if not trajectory:
        raise ValueError("Trace audit requires at least one recorded LLM arm")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "decision_traces.jsonl").open("x", encoding="utf-8") as out:
        for row in trajectory:
            out.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    transitions = [row for row in trajectory if row["round"] > 1]
    result = {"status": "complete", "trajectory": {"decisions": len(trajectory),
                                                   "transitions": len(transitions),
                                                   "feedback_cited_in_next": sum(bool(row["cited_new_feedback"]) for row in transitions),
                                                   "own_greedy_overlap": mean(row["greedy_overlap"] for row in trajectory)},
              "paired_mode": "replay" if replay else "live" if paired else "trace_only",
              "paired_status": "not_run"}
    if paired or replay:
        try:
            if "deepseek" not in studies:
                raise ValueError("The frozen paired intervention is predefined for DeepSeek only")
            points = _checkpoints(dataset, studies["deepseek"])
            write_json(output / "checkpoint_inputs.json", points)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                jobs = [[executor.submit(_submit, point, branch, output, replay)
                         for branch in ("no_feedback", "with_feedback")] for point in points]
            decisions = [[job.result() for job in pair] for pair in jobs]
            measured = [_score_pair(dataset, point, picks) for point, picks in zip(points, decisions, strict=True)]
            rows, selections = zip(*measured, strict=True)
            _table(output / "paired_feedback_results.csv", list(rows))
            _table(output / "paired_measurements.csv", [row for batch in selections for row in batch])
            if replay:
                for file in ("paired_feedback_results.csv", "paired_measurements.csv"):
                    pd.testing.assert_frame_equal(frame(output / file), frame(replay / file),
                                                  check_exact=False, rtol=1e-10, atol=1e-10)
            traces = [trace for pair in decisions for _, trace in pair]
            result.update(paired_status="complete", paired={
                "pairs": len(rows), "sets_changed": sum(row["set_changed"] for row in rows),
                "mean_overlap": mean(row["top_k_overlap"] for row in rows),
                "mean_best_delta": mean(row["best_delta"] for row in rows),
                "best_delta_range": [min(row["best_delta"] for row in rows), max(row["best_delta"] for row in rows)],
                "mean_batch_delta": mean(row["mean_delta"] for row in rows),
                "mean_best_vs_greedy": mean(row["best_versus_model_only"] for row in rows),
                "new_feedback_cited": sum(bool(row["cited_new_feedback"]) for row in rows),
                "live_generations": sum(log["live_generation_count"] for log in traces),
                "input_tokens": sum(log["usage"]["input_tokens"] for log in traces),
                "output_tokens": sum(log["usage"]["output_tokens"] for log in traces),
            })
        except (ValueError, RuntimeError, OSError, KeyError, AssertionError) as exc:
            result.update(paired_status="NOT EVALUABLE", paired_reason=str(exc))
    if before != _freeze(studies):
        raise ValueError("Frozen GFP archive changed while the audit was running")
    result["main_archive_unchanged"] = True
    result["main_archive_manifest_sha256"] = _digest(before)
    write_json(output / "audit_summary.json", result)
    _report(output, result)
    return result
