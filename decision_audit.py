"""Audit frozen measured decision trajectories and optionally test a feedback intervention."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_DIRS = ("concrete", "sonnet", "gpt", "qwen")
SOURCE_DIRS = (*MODEL_DIRS, "across-models")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _frozen_manifest(root: Path) -> dict[str, str]:
    for stage in MODEL_DIRS:
        status = _read(root / stage / "status.json")
        if status["status"] != "complete" or status["mode"] != "formal":
            raise ValueError(f"{stage} has no frozen formal archive")
    if set(_read(root / "across-models/validation.json")["models"]) != {"deepseek", "sonnet", "gpt", "qwen"}:
        raise ValueError("Four matched completed models are required")
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for stage in SOURCE_DIRS
        for path in sorted((root / stage).rglob("*"))
        if path.is_file()
    }


def _paired_figure(frame: pd.DataFrame, figure_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.7, 3.8), layout="constrained")
    cells = ["No new feedback", "With real feedback", "Model only"]
    colors = ("#776587", "#326477", "#A55840", "#57794B")
    for color, marker, style, row in zip(colors, ("o", "D", "x", "s"), ("-", "--", ":", "-"),
                                         frame.itertuples(), strict=True):
        ax.plot(cells, [row.no_feedback_best_mpa, row.with_feedback_best_mpa, row.model_only_best_mpa],
                marker=marker, linestyle=style, color=color,
                label=f"Seed {row.seed}, {'knowledge ON' if row.arm.startswith('D') else 'knowledge OFF'}")
    ax.set_ylabel("Best measured strength (MPa)")
    ax.grid(axis="y", color="#E5E9EC")
    ax.spines[["right", "top"]].set_visible(False)
    fig.legend(loc="outside upper center", frameon=False, ncol=2, fontsize=8)
    fig.savefig(figure_dir / "paired-feedback.png", dpi=180)
    plt.close(fig)


def _check_replay(original_dir: Path, output: Path, replayed: pd.DataFrame) -> None:
    original = pd.read_csv(original_dir / "paired_feedback_results.csv")
    for row in original.itertuples():
        for label in ("no_feedback", "with_feedback"):
            source_dir = (original_dir / "paired").resolve()
            source_trace = (source_dir / getattr(row, f"{label}_trace")).resolve()
            if not source_trace.is_relative_to(source_dir):
                raise ValueError("Archived trace reference escapes the audit")
            expected = getattr(row, f"{label}_trace_sha256")
            if hashlib.sha256(source_trace.read_bytes()).hexdigest() != expected:
                raise ValueError("Archived trace digest differs from its recorded digest")
    # Replay logs have different execution timestamps and execution mode. Response digests are
    # checked in the policy replay, and the selected data and all paired outcomes must agree.
    log_hash_columns = ["no_feedback_trace_sha256", "with_feedback_trace_sha256"]
    pd.testing.assert_frame_equal(original.drop(columns=log_hash_columns), replayed.drop(columns=log_hash_columns),
                                  check_exact=False, rtol=1e-10, atol=1e-10)
    pd.testing.assert_frame_equal(pd.read_csv(original_dir / "paired/selections.csv"),
                                  pd.read_csv(output / "paired/selections.csv"),
                                  check_exact=False, rtol=1e-10, atol=1e-10)


def _batch_mean_comparison(output: Path) -> dict:
    selections = pd.read_csv(output / "paired/selections.csv")
    means = selections.groupby(["seed", "arm", "feedback_branch"]).strength_mpa.mean().unstack()
    gain = means.with_feedback - means.no_feedback
    return {
        "mean_batch_strength_gain_mpa": float(gain.mean()),
        "mean_batch_strength_gain_range_mpa": [float(gain.min()), float(gain.max())],
        "mean_batch_strength_gain_versus_model_only_mpa": float((means.with_feedback - means.greedy).mean()),
    }


def _write_report(output: Path, summary: dict, records: list[dict], paired: pd.DataFrame | None) -> None:
    trace_status = summary["trajectory_status"]
    pair_status = summary["paired_status"]
    sections = [
        "# 科研决策审计",
        "",
        "审计对象为实测混凝土闭环。审计记录、临时决策和测量查阅均使用独立目录。",
        "",
        "## 归档决策轨迹",
        "",
    ]
    if trace_status == "complete":
        feedback_rounds = [record for record in records if int(record["round"]) > 1]
        # P1 only uses the traces of actions that were actually taken. P2 supplies matched alternatives.
        sections += [
            f"从已完成的模型运行重建了 {len(records)} 轮，{len(feedback_rounds)} 轮发生在首次反馈以后。",
            f"已测方案中，有 {sum(len(record['same_state_greedy']['overlap_ids']) for record in records)}"
            f"/{sum(len(record['chosen']) for record in records)} 个与该轮直接 RF 前五重合。",
            "每轮记录假设、证据引用、提议候选、评分、实际选择、真实测量及下一轮行动。",
            f"{summary['trajectory']['next_round_transitions']} 次轮间过渡中有 "
            f"{summary['trajectory']['transitions_citing_newly_revealed_observations']} 次"
            "引用了上一轮刚测得的可追溯证据。",
            "轨迹追溯说明证据可用或被引用。反馈自身的选择效应由下面的配对检查测量。",
            "自由文本假设缺少预先规定的可检验判据时，其支持或反驳状态为 NOT EVALUABLE。",
            "",
            "主档案的模型贪心重叠及各轮测量在 [决策轨迹](decision_traces.jsonl)。",
        ]
    else:
        sections += [f"**NOT EVALUABLE**。{summary.get('trajectory_reason', '决策轨迹不完整。')}"]
    sections += ["", "## 固定检查点的反馈比较", ""]
    if pair_status == "complete" and paired is not None:
        changed = int(paired.set_changed.sum())
        delta = paired.delta_best_mpa
        direct = paired.with_feedback_best_mpa - paired.model_only_best_mpa
        eligible = len(paired)
        secondary = summary["paired_metrics"]
        selections = pd.read_csv(output / "paired/selections.csv")
        sections += [
            f"在 RF 权重、候选菜单、模型和预算相同的 {eligible} 个配对检查点，"
            f"提供上一轮真实测量后，{changed}/{eligible} 个 Top-5 集合发生变化。"
            f"平均集合重叠为 {paired.selection_overlap.mean()/5:.0%}。",
            f"有反馈减无反馈的最佳实测强度平均差为 {delta.mean():+.2f} MPa，"
            f"范围 [{delta.min():+.2f}, {delta.max():+.2f}] MPa。"
            f"有反馈减直接 RF 推荐的平均差为 {direct.mean():+.2f} MPa。"
            "每个条件只测一次模型决策，差异可能含单次响应变动。",
            f"作为次要指标，整批平均强度的反馈差为 {secondary['mean_batch_strength_gain_mpa']:+.2f} MPa，"
            f"相对 RF 直排的差值为 {secondary['mean_batch_strength_gain_versus_model_only_mpa']:+.2f} MPa。",
            f"有反馈的推荐理由在 {int((paired.new_feedback_evidence_count > 0).sum())}/{eligible} 个检查点"
            "引用了可追溯的新测量。"
            f"有反馈 Agent 与直接 RF 直排平均重叠 {paired.with_feedback_greedy_overlap.mean()/5:.0%}。"
            f"无反馈和有反馈的平均遗憾值分别为 {paired.no_feedback_regret_mpa.mean():.2f} 和 "
            f"{paired.with_feedback_regret_mpa.mean():.2f} MPa。",
            "",
            "| 种子 | 知识 | 无新反馈 MPa | 有真实反馈 MPa | RF 直排 MPa | 反馈差 MPa | 重叠 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for row in paired.itertuples():
            sections.append(
                f"| {row.seed} | {'开' if row.arm.startswith('D') else '关'} | "
                f"{row.no_feedback_best_mpa:.2f} | {row.with_feedback_best_mpa:.2f} | "
                f"{row.model_only_best_mpa:.2f} | {row.delta_best_mpa:+.2f} | {row.selection_overlap}/5 |"
            )
        weakest = paired.assign(gap=direct).sort_values(["gap", "seed"]).iloc[0]
        case = selections[(selections.seed == weakest.seed) & (selections.arm == weakest.arm)]
        greedy = case[case.feedback_branch == "greedy"].sort_values("strength_mpa").iloc[-1]
        picked = case[case.feedback_branch == "with_feedback"]
        sections += [
            "",
            f"有反馈相对 RF 最弱的检查点是种子 {weakest.seed}、知识"
            f"{'开' if weakest.arm.startswith('D') else '关'}，差值 {weakest.gap:+.2f} MPa。",
        ]
        if greedy.id not in set(picked.id):
            sections.append(
                f"有反馈的选择漏掉了 RF 排名第 {greedy.rf_rank} 的 {greedy.id}，"
                f"其存档实测为 {greedy.strength_mpa:.2f} MPa。"
            )
        sections += [
            "",
            "[配对结果与遗憾值](paired_feedback_results.csv) · [逐候选测量](paired/selections.csv) · "
            "[配对曲线](figures/paired-feedback.png)",
        ]
    else:
        sections += [f"**{pair_status}**。{summary.get('paired_reason', '本次仅运行了归档轨迹。')}"]
    sections += [
        "",
        "## 输入与出处",
        "",
        f"运行 ID {summary['run_id']}。"
        + ("冻结主实验文件的前后哈希一致性已校验。" if summary["frozen_inputs_unchanged"] else "冻结主实验档案未完成校验。"),
        "方法沿用 LLMDictator 的共用检查点、独立证据分支和实测终局思路。",
        "",
        "[冻结审计协议](audit_protocol.md) · [摘要与来源哈希](audit_summary.json)",
    ]
    (output / "audit_report.md").write_text("\n".join(sections) + "\n", encoding="utf-8")


def run(run_dir: Path, output: Path, *, paired: bool = False, replay: Path | None = None) -> dict:
    from audit_trace import reconstruct_decisions, summarize_decisions

    run_dir, output = run_dir.resolve(), output.resolve()
    if output == run_dir or any(output.is_relative_to(run_dir / stage) for stage in SOURCE_DIRS):
        raise ValueError("Use a new audit folder outside the frozen input directories")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    figure_dir = output / "figures"
    figure_dir.mkdir()
    protocol = ROOT / "audit_protocol.md"
    (output / "audit_protocol.md").write_bytes(protocol.read_bytes())
    started = datetime.now(timezone.utc)
    summary = {"run_id": f"decision-audit-{started.strftime('%Y%m%dT%H%M%S%fZ')}",
               "started_utc": started.isoformat(), "paired_mode": "replay" if replay else ("live" if paired else "none"),
               "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest()}
    records, table, before = [], None, None
    try:
        before = _frozen_manifest(run_dir)
        records = reconstruct_decisions(run_dir)
        with (output / "decision_traces.jsonl").open("x", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        summary.update(trajectory_status="complete", trajectory=summarize_decisions(records))
    except (ValueError, OSError, KeyError) as exc:
        summary.update(trajectory_status="NOT EVALUABLE", trajectory_reason=str(exc))

    if summary["trajectory_status"] == "complete" and (paired or replay):
        from audit_paired import run as run_pairs
        from llm_agent import choose_candidates

        pair_dir = output / "paired"

        def select(**kwargs):
            if replay is None:
                return choose_candidates(**kwargs)
            source = replay.resolve() / "paired" / kwargs["log_path"].relative_to(pair_dir)
            return choose_candidates(**kwargs, replay_path=source)

        result = run_pairs(pair_dir, baseline=run_dir / "concrete", selector=select, freeze_path=protocol,
                           replay=replay is not None)
        summary["paired_status"] = result["status"]
        summary["paired_result"] = result
        if result["status"] == "complete":
            table = pd.read_csv(pair_dir / "pairs.csv")
            if replay:
                _check_replay(replay, output, table)
            table.to_csv(output / "paired_feedback_results.csv", index=False, lineterminator="\n")
            delta = table.delta_best_mpa
            summary["paired_metrics"] = {
                "n_pairs": len(table), "recommendation_change_rate": float(table.set_changed.mean()),
                "ordered_change_rate": float(table.order_changed.mean()),
                "mean_top_k_overlap": float(table.selection_overlap.mean() / 5),
                "mean_realized_gain_mpa": float(delta.mean()), "gain_range_mpa": [float(delta.min()), float(delta.max())],
                "gain_positive_tied_negative": [int((delta > 1e-10).sum()), int((delta.abs() <= 1e-10).sum()), int((delta < -1e-10).sum())],
                "mean_no_feedback_regret_mpa": float(table.no_feedback_regret_mpa.mean()),
                "mean_with_feedback_regret_mpa": float(table.with_feedback_regret_mpa.mean()),
                "mean_model_only_agreement_no_feedback": float(table.no_feedback_greedy_overlap.mean() / 5),
                "mean_model_only_agreement_with_feedback": float(table.with_feedback_greedy_overlap.mean() / 5),
                "mean_gain_versus_model_only_mpa": float((table.with_feedback_best_mpa - table.model_only_best_mpa).mean()),
                "pairs_with_traceable_new_evidence_in_reasoning": int((table.new_feedback_evidence_count > 0).sum()),
            }
            summary["paired_metrics"].update(_batch_mean_comparison(output))
            _paired_figure(table, figure_dir)
        else:
            summary["paired_reason"] = result.get("reason", "配对决策没有通过校验。")
    else:
        summary["paired_status"] = "NOT EVALUABLE" if paired or replay else "未运行"

    if before is not None and before != _frozen_manifest(run_dir):
        raise ValueError("Frozen evidence changed during audit")
    summary["frozen_input_sha256"] = before or {}
    summary["frozen_inputs_unchanged"] = before is not None
    _save(output / "audit_summary.json", summary)
    _write_report(output, summary, records, table)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "evidence")
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--paired", action="store_true", help="Run the preselected live DeepSeek checks.")
    mode.add_argument("--replay-audit", type=Path, help="Verify and replay an archived audit without API calls.")
    args = parser.parse_args()
    result = run(args.run_dir, args.output, paired=args.paired, replay=args.replay_audit)
    print(result["trajectory_status"], result["paired_status"], args.output)


if __name__ == "__main__":
    main()
