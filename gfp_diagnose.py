"""Diagnose discovery opportunity and selection losses from frozen GFP logs."""
from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path
from statistics import fmean

from gfp_data import ASSAY, DEFAULT_CACHE_DIR, PROJECT, load_dataset


def random_hit_probability(pool_size: int, hits: int, budget: int) -> float:
    """Uniform distinct draws, with no replacement across rounds."""
    if not 0 <= hits <= pool_size or not 0 <= budget <= pool_size:
        raise ValueError("Hit count and query budget must both fit the candidate pool")
    return 1 - math.comb(pool_size - hits, budget) / math.comb(pool_size, budget)


def _read(path: Path) -> dict:
    return json.loads(path.read_bytes())


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def pool_diagnostics(scores: dict, split: dict, seed: int, budget: int) -> tuple[dict, set[str]]:
    initial_best = max(scores[key] for key in split["initial"])
    pool = split["pool"]
    wild_type = scores["WT"]
    pool_best = max(scores[key] for key in pool)
    hits = {key for key in pool if scores[key] > initial_best}
    return {
        "seed": seed, "initial_count": len(split["initial"]), "pool_count": len(pool),
        "initial_best": initial_best, "pool_best": pool_best, "wild_type": wild_type,
        "pool_best_minus_initial": pool_best - initial_best,
        "max_incumbent_gain": max(0.0, pool_best - initial_best),
        "above_initial_count": len(hits), "above_wt_count": sum(scores[key] > wild_type for key in pool),
        "random_query_budget": budget,
        "random_hit_probability": random_hit_probability(len(pool), len(hits), budget),
    }, hits


def diagnose_arm(scores, path: Path, *, pool, hits, headroom, backend, arm, seed, rounds, budget):
    eligible = set(pool)
    cohort = {"backend": backend, "arm": arm, "seed": seed}
    visited = set()
    exposed = set()
    elite_exposed = set()
    rf_top = set()
    all_outcomes = []
    observed_hits = set()
    best_hit_rank = None
    round_rows = []
    llm = arm in {"C_llm", "D_knowledge_llm"}
    for number in range(1, rounds + 1):
        prefix = f"round-{number:02d}"
        selection = _read(path / f"{prefix}-selection.json")
        data = _read(path / f"{prefix}.json")
        chosen = selection["selected_ids"]
        ranks = selection["eligible_rf_ranks"]
        top = set(selection["model_only_ids"])
        available_hits = hits & eligible
        menu = _read(path / f"{prefix}-request.json")["menu"] if llm else []
        menu_ids = {row["id"] for row in menu}
        if (len(chosen) != budget or len(set(chosen)) != budget or not set(chosen) <= eligible
                or set(ranks) != eligible or len(top) != budget or not top <= eligible
                or not menu_ids <= eligible or (llm and not set(chosen) <= menu_ids)
                or data["seed"] != seed or data["round"] != number or data["arm"] != arm
                or [row["id"] for row in data["selected"]] != chosen
                or any(row["fitness"] != scores[row["id"]] for row in data["selected"])):
            raise ValueError(f"Round artifacts disagree for {path}/{prefix}")
        rf_top.update(top & hits)
        visible_hits = menu_ids & hits
        exposed.update(visible_hits)
        elite_exposed.update(row["id"] for row in menu if row["source"] == "rf_elite" and row["id"] in hits)
        if available_hits:
            current_rank = min(ranks[k] for k in available_hits)
            best_hit_rank = current_rank if best_hit_rank is None else min(best_hit_rank, current_rank)
        count_hits = len(set(chosen) & hits)
        observed_hits.update(set(chosen) & hits)
        eligible.difference_update(chosen)
        visited.update(chosen)
        outcomes = [scores[key] for key in chosen]
        all_outcomes.extend(outcomes)
        round_rows.append({
            **cohort, "round": number, "n_measured": len(chosen), "batch_best": max(outcomes),
            "batch_mean": fmean(outcomes), "above_initial_hits": count_hits,
            "above_wt_hits": sum(value > headroom["wild_type"] for value in outcomes),
            "available_above_initial": len(available_hits), "rf_top5_above_initial": len(top & hits),
            "displayed_above_initial": len(visible_hits) if llm else None,
        })
    if len(visited) != rounds * budget:
        raise ValueError(f"Unexpected distinct query count for {path}")
    attainable = exposed if llm else rf_top if arm == "B_model_only" else hits
    above_wt = sum(row["above_wt_hits"] for row in round_rows)
    result = {
        **cohort, "n_measured": len(all_outcomes), "incumbent_gain": max(0.0, max(all_outcomes) - headroom["initial_best"]),
        "best_new_measurement": max(all_outcomes), "mean_new_measurement": fmean(all_outcomes),
        "above_wt_hits": above_wt, "above_wt_hit_rate": above_wt / len(all_outcomes),
        "above_initial_hits": len(observed_hits), "above_initial_not_accessible_to_rule": len(hits - attainable),
        "above_initial_accessible_but_unselected": len(attainable - observed_hits),
        "distinct_above_initial_in_rf_top5": len(rf_top),
        "distinct_above_initial_in_llm_menu": len(exposed) if llm else None,
        "distinct_above_initial_in_llm_rf_elite": len(elite_exposed) if llm else None,
        "best_rf_rank_among_above_initial": best_hit_rank,
    }
    return result, round_rows


def diagnose(archive: Path, output: Path, scores) -> dict:
    config = _read(archive / "deepseek/config.json")
    rounds, budget = config["rounds"], config["budget"]
    folders = sorted(path for path in archive.iterdir() if (path / "config.json").is_file())
    configs = {path: _read(path / "config.json") for path in folders}
    pools, arms, observed_rounds = [], [], []
    pool_ids = {}
    for seed in config["seeds"]:
        split_file = f"splits/seed-{seed}.json"
        split = _read(archive / "deepseek" / split_file)
        if (len(set().union(*map(set, split.values()))) != sum(map(len, split.values()))
                or "WT" not in split["initial"]):
            raise ValueError(f"Frozen split is inconsistent for seed {seed}")
        pool_ids[seed] = set(split["pool"])
        headroom, hits = pool_diagnostics(scores, split, seed, rounds * budget)
        pools.append(headroom)
        for folder, local in configs.items():
            if (local["rounds"] != rounds or local["budget"] != budget or local["seeds"] != config["seeds"]
                    or _read(folder / split_file) != split):
                raise ValueError(f"Provider did not use the shared frozen GFP split and budget: {folder.name}")
            for arm in local["arms"]:
                result, round_rows = diagnose_arm(
                    scores, folder / "arms" / f"seed-{seed}" / arm, pool=pool_ids[seed], hits=hits,
                    headroom=headroom, backend=local["backend"], arm=arm, seed=seed, rounds=rounds, budget=budget,
                )
                arms.append(result)
                observed_rounds.extend(round_rows)

    llms = [r for r in arms if r["distinct_above_initial_in_llm_menu"] is not None]
    summary = {
        "scope": "retrospective diagnosis of frozen GFP decisions; no LLM calls or model training",
        "budget_per_arm": rounds * budget,
        "seed_headroom": pools,
        "overlapping_pool_ids": {f"{a}/{b}": len(pool_ids[a] & pool_ids[b]) for a, b in combinations(pool_ids, 2)},
        "arms": len(arms), "rounds": len(observed_rounds), "queries": sum(row["n_measured"] for row in arms),
        "arms_that_improved": sum(row["above_initial_hits"] > 0 for row in arms),
        "arms_with_above_initial_in_random_forest_top5": sum(row["distinct_above_initial_in_rf_top5"] > 0 for row in arms),
        "llm_arms": len(llms), "llm_arms_exposed_to_improvement": sum(row["distinct_above_initial_in_llm_menu"] > 0 for row in llms),
        "llm_arms_exposed_via_rf_elite": sum(row["distinct_above_initial_in_llm_rf_elite"] > 0 for row in llms),
        "llm_arms_that_converted_exposure": sum(row["above_initial_hits"] > 0 for row in llms),
        "source": {"assay": ASSAY, "frozen_protocol_sha256": config["protocol_sha256"]},
    }
    output.mkdir(parents=True, exist_ok=False)
    _write_csv(output / "seed_headroom.csv", pools)
    _write_csv(output / "arm_metrics.csv", arms)
    _write_csv(output / "round_diagnostics.csv", observed_rounds)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_report(output, summary)
    return summary


def _write_report(output: Path, result: dict) -> None:
    lines = [
        "# 冻结 GFP 实验的胜出机会与选择损失", "",
        "| 种子 | 初始最佳 | 候选池最佳 | 初始最佳以上 | 高于野生型 | 最大可提高 | 随机 15 查至少命中一次 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["seed_headroom"]:
        lines.append(f"| {row['seed']} | {row['initial_best']:.4f} | {row['pool_best']:.4f} | "
                     f"{row['above_initial_count']}/{row['pool_count']} | {row['above_wt_count']} | "
                     f"{row['max_incumbent_gain']:+.4f} | {row['random_hit_probability']:.2%} |")
    lines.extend([
        "", "随机命中概率按同一 384 项池中的 15 次均匀、不放回抽样计算。荧光分数使用档案 log10 尺度。",
        "", f"{result['arms']} 个实验臂中，只有 {result['arms_that_improved']} 个提高了初始最佳。"
        f"出现过初始最佳以上候选的 RF Top-5 次数覆盖 {result['arms_with_above_initial_in_random_forest_top5']} 个臂。"
        f"{result['llm_arms']} 个 LLM 臂中有 {result['llm_arms_exposed_to_improvement']} 个曾在菜单看到可提高候选，"
        f"其中通过 RF 精英区看到的臂数为 {result['llm_arms_exposed_via_rf_elite']}，"
        f"最终选中的臂数为 {result['llm_arms_that_converted_exposure']}。",
        "", "终局最佳回答是否刷新已有最高值，受起点与池内上限限制。批内最佳反映该轮找到的最高新测量，"
        "批内均值反映同一测量预算下五条候选的整体表现。高于野生型命中率回答较宽阈值下亮变体的发现频率。"
        "这些指标可以同时变化，含义各自独立。",
        "", "三个候选池中有一个在给定初始最佳下无法刷新，另外两个只有少数可刷新候选。"
        "这使终局最佳事件稀少，难以用三个种子区分策略。菜单暴露和未选诊断是已发生轨迹上的归因描述；"
        "更换前序推荐会改变后续训练记录和菜单，故这里没有估计改用其他策略的反事实收益。",
        "", "[候选池数据](seed_headroom.csv) · [分臂损失](arm_metrics.csv) · [逐轮数据](round_diagnostics.csv) · [汇总](summary.json)",
    ])
    (output / "diagnosis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=PROJECT / "evidence/gfp/formal")
    parser.add_argument("--data-cache", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(not (args.data_cache / name).is_file() for name in (f"{ASSAY}.csv", "sarkisyan-brightness.tsv")):
        parser.error("Diagnosis requires the cached, hash-checked GFP assay files")
    report = diagnose(args.archive, args.output, load_dataset(args.data_cache).scores)
    print(f"{report['queries']} frozen queries diagnosed in {args.output}")


if __name__ == "__main__":
    main()
