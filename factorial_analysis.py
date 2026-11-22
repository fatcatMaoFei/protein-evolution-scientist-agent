"""Verify and summarize paired model-by-knowledge contrasts on measured Concrete."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
from math import isclose
from pathlib import Path
from statistics import mean, median

import numpy as np

from factorial_runner import _same_menu
from llm_agent import TOP_MENU, data_analyst
from nonbio import ARMS, DEFAULT_DATA, FEATURES, make_split, read_dataset, write_csv, write_json
from verify import read as read_json, require


MODELS = ("deepseek", "sonnet")
LLM_ARMS = ARMS[2:]
CONDITIONS = tuple((model, knowledge) for model in MODELS for knowledge in ("off", "on"))
CONTROLS = ("random", "greedy")
SCHEDULE_FIELDS = (
    "seeds", "rounds", "budget", "predictor", "rf_parameters", "ridge_alpha",
    "python", "numpy", "pandas", "scipy", "scikit_learn", "target_age_days",
    "primary_endpoint", "high_threshold", "first_hit_effort", "stopping_condition", "rank_ties",
)
OUTCOMES = (
    "gain_vs_initial_best_mpa", "top_hits", "mean_greedy_overlap", "seeded_menu_fraction",
)
USAGE_FIELDS = (
    "input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens",
    "provider_primary_input_tokens",
)


def load_records(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def match_schedule(baseline: dict, sonnet: dict) -> dict:
    for field in SCHEDULE_FIELDS:
        require(
            field in baseline and field in sonnet and baseline[field] == sonnet[field],
            f"Archived controls differ or are missing for {field}",
        )
    return {field: baseline[field] for field in SCHEDULE_FIELDS}


def condition_for(model: str, arm: str) -> tuple[str, str]:
    if arm in ARMS[:2]:
        return CONTROLS[ARMS.index(arm)], "na"
    return model, "on" if arm == LLM_ARMS[1] else "off"


def strict_digest(value) -> str:
    data = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(data.encode()).hexdigest()


def verify_frozen_inputs(baseline: Path, sonnet: Path, sonnet_config: dict, seeds: list[int]) -> None:
    ancestors = {"config.json", "source.json", "status.json"}
    for seed in seeds:
        ancestors.add(f"splits/seed-{seed}.json")
        for arm in ARMS:
            prefix = f"arms/seed-{seed}/{arm}"
            ancestors.add(f"{prefix}/initial_observed.json")
            if arm in LLM_ARMS:
                ancestors.add(f"{prefix}/round-01-agent.json")
    recorded = sonnet_config["baseline_file_sha256"]
    require(set(recorded) == ancestors, "Frozen Sonnet baseline evidence manifest is incomplete")
    for relative, digest in recorded.items():
        require(sha256((baseline / relative).read_bytes()).hexdigest() == digest,
                f"Original frozen baseline input has changed: {relative}")
    freeze_file = (sonnet / sonnet_config["frozen_protocol_file"]).resolve()
    require(freeze_file.is_relative_to(sonnet), "Frozen protocol must be local to the Sonnet evidence")
    require(sha256(freeze_file.read_bytes()).hexdigest() == sonnet_config["frozen_protocol_sha256"],
            "Frozen Sonnet protocol digest mismatch")


def canonical_proposal(request):
    """Compare instructions and functional tool definitions across native formats."""
    if "instructions" in request:
        return {
            "messages": [
                {"role": "system", "content": request["instructions"]},
                *request["input"],
            ],
            "tools": [
                {field: tool[field] for field in ("name", "description", "parameters")}
                for tool in request["tools"]
            ],
        }
    if "system" not in request:
        return {
            "messages": request["messages"],
            "tools": [tool["function"] for tool in request["tools"]],
        }
    return {
        "messages": [
            {"role": "system", "content": request["system"]},
            *[
                {"role": message["role"], "content": "\n".join(block["text"] for block in message["content"])}
                for message in request["messages"]
            ],
        ],
        "tools": [
            {"name": tool["name"], "description": tool["description"], "parameters": tool["input_schema"]}
            for tool in request["tools"]
        ],
    }


def cached_usage(response: dict) -> dict:
    provider = response["provider_usage"]
    base_tokens = response["usage"]["input_tokens"]
    if "prompt_tokens" in provider:
        require(base_tokens == provider["prompt_tokens"], "Reported Chat Completions input differs")
        cache_read = (provider.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        return {
            "cache_read_input_tokens": cache_read,
            "provider_primary_input_tokens": base_tokens,
        }
    if "input_tokens_details" in provider:
        require(base_tokens == provider["input_tokens"], "Reported Responses input differs")
        cache_read = (provider["input_tokens_details"] or {}).get("cached_tokens") or 0
        return {
            "cache_read_input_tokens": cache_read,
            "provider_primary_input_tokens": base_tokens,
        }
    primary_input = provider["input_tokens"]
    cache_read = provider.get("cache_read_input_tokens") or 0
    cache_creation = provider.get("cache_creation_input_tokens") or 0
    require(base_tokens == primary_input + cache_read + cache_creation,
            "Normalized input must include all reported cache categories")
    return {
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": cache_creation,
        "provider_primary_input_tokens": primary_input,
    }


def verify_trace(trace: dict, ids: list[str], *, model: str, augmented: bool,
                 seed: int, step: int, budget: int) -> dict:
    require(
        trace["status"] == "ok" and trace["mode"] == "live"
        and trace["knowledge_augmented"] is augmented
        and (trace["seed"], trace["round"], trace["budget"]) == (seed, step, budget),
        f"Wrong or incomplete live trace for {model}/{augmented}/{seed}/{step}",
    )
    require(trace["decision"]["selected_ids"] == ids, "Agent decision differs from measured ledger")
    require(len([tool for tool in trace["tool_calls"] if tool["trigger"] == "model_tool_call"
                 and tool["tool"] == "evaluate_concrete_shortlist"]) == 1,
            "Expected exactly one native evaluator call")
    calls = trace["calls"]
    require(len(calls) == trace["live_generation_count"] >= 2, "Incorrect live generation count")

    usage, cache_usage, requested, returned, runtimes = Counter(), Counter(), set(), set(), set()
    for call in calls:
        request, response = call["request"], call["response"]
        require(strict_digest(request) == call["request_sha256"], "Request digest mismatch")
        require(strict_digest(response) == call["response_sha256"], "Response digest mismatch")
        expected_model = response["model"]
        require(response["response_id"] and model in expected_model.lower(),
                f"Unexpected actual model for {model}: {expected_model}")
        require(
            request["model"] == trace["model"] and model in trace["model"].lower(),
            f"Wrong requested model for {model}",
        )
        if model == "qwen":
            require(trace["config"]["revision"] == request["revision"] == response["revision"],
                    "Local Qwen revision drift")
            record = [response["revision"], response["rendered_prompt"], response["raw_output"]]
            expected_id = "local-" + sha256(json.dumps(record, ensure_ascii=True).encode()).hexdigest()
            require(response["response_id"] == expected_id, "Local Qwen generation identity mismatch")
        requested.add(request["model"])
        returned.add(expected_model)
        runtimes.add(json.dumps(response.get("runtime", {}), sort_keys=True))
        usage.update(response["usage"])
        cache_usage.update(cached_usage(response))
    require(dict(usage) == trace["usage"], "Trace token totals differ from provider responses")
    menu = {mix["id"]: mix for mix in trace["menu"]}
    require(len(menu) == len(trace["menu"]) and set(ids) <= set(menu), "Ineligible or duplicate menu")
    require(
        all(entry["source"] == ("top_prediction" if entry["rank"] <= TOP_MENU else "seeded_sample")
            for entry in menu.values()),
        "Menu sources disagree with the fixed model-rank rule",
    )
    proposal = next(call["request"] for call in calls if call["stage"] == "proposal")
    return {
        "greedy_rank_picks": sum(menu[mix]["rank"] <= budget for mix in ids),
        "other_high_rank_picks": sum(budget < menu[mix]["rank"] <= TOP_MENU for mix in ids),
        "seeded_picks": sum(menu[mix]["source"] == "seeded_sample" for mix in ids),
        "menu": trace["menu"],
        "proposal_digest": strict_digest(canonical_proposal(proposal)),
        "usage": usage | cache_usage,
        "requested_models": requested,
        "returned_models": returned,
        "runtime_versions": runtimes,
        "generation_settings": {json.dumps({
            key: value for key, value in trace["config"].items() if key not in ("model", "api_base")
        }, sort_keys=True)},
        "calls": len(calls),
    }


def audit_arm(folder: Path, *, seed: int, arm: str, model: str, schedule: dict, split: dict,
              table, rounds: dict, batches: dict) -> tuple[dict, dict, dict]:
    arm_path = folder / "arms" / f"seed-{seed}" / arm
    initial = read_json(arm_path / "initial_observed.json")
    require(list(initial) == split["initial"], "Ordered initial rows differ from matched split")
    for mix_id, record in initial.items():
        require(all(isclose(record[field], table.at[mix_id, field]) for field in (*FEATURES, "strength")),
                "Initial observation differs from raw dataset")
    initial_best = max(row["strength"] for row in initial.values())
    threshold = float(np.quantile(table.loc[split["pool"], "strength"], 0.9))
    budget = schedule["budget"]
    eligible = set(split["pool"])
    observed = initial.copy()
    incumbent, total_hits, first_hit = initial_best, 0, None
    selections, overlaps, traces = [], [], []
    for step in range(1, schedule["rounds"] + 1):
        key = seed, arm, step
        batch = sorted(batches[key], key=lambda row: int(row["batch_position"]))
        ids = [row["id"] for row in batch]
        require(len(ids) == len(set(ids)) == budget and set(ids) <= eligible,
                f"Budget, partition or duplicate mismatch for {key}")
        require([int(row["batch_position"]) for row in batch] == list(range(1, budget + 1)),
                f"Within-batch order mismatch for {key}")
        require([int(row["experiment_number"]) for row in batch]
                == list(range((step - 1) * budget + 1, step * budget + 1)),
                f"Experimental order mismatch for {key}")
        for row in batch:
            expected = float(table.at[row["id"], "strength"])
            require(isclose(float(row["strength_mpa"]), expected)
                    and int(row["top_hit"]) == int(expected >= threshold),
                    f"Archived selection differs from measured source for {key}")
            if first_hit is None and int(row["top_hit"]):
                first_hit = int(row["experiment_number"])

        measured = table.loc[ids, "strength"]
        total_hits += sum(int(row["top_hit"]) for row in batch)
        incumbent = max(incumbent, float(measured.max()))
        checkpoint = read_json(arm_path / f"round-{step:02d}.json")
        greedy_ids = set(checkpoint["contemporaneous_greedy_ids"])
        require(len(greedy_ids) == budget and greedy_ids <= eligible, f"Ineligible greedy reference for {key}")
        require(checkpoint["number_observed_after_refit"] == len(initial) + step * budget,
                f"Wrong checkpoint observation count for {key}")
        require(all(int(row["is_myopic_greedy_pick"]) == int(row["id"] in greedy_ids) for row in batch),
                f"Wrong greedy overlap in selection ledger for {key}")
        greedy_overlap = len(set(ids) & greedy_ids) / budget
        round_row = rounds[key]
        for metric, expected in {
            "budget": budget, "tested": step * budget, "top_cutoff_mpa": threshold,
            "round_best_mpa": measured.max(), "best_so_far_mpa": incumbent,
            "top_hits": total_hits, "greedy_overlap_at_k": greedy_overlap,
            "gain_vs_initial_best_mpa": incumbent - initial_best,
        }.items():
            require(isclose(float(round_row[metric]), expected, abs_tol=1e-10),
                    f"Cannot reconstruct {metric} for {key}")
        actual_first_hit = int(float(round_row["first_hit_tests"])) if round_row["first_hit_tests"] else None
        require(actual_first_hit == first_hit, f"First-hit ledger mismatch for {key}")
        if arm in LLM_ARMS:
            agent_trace = read_json(arm_path / f"round-{step:02d}-agent.json")
            trace = verify_trace(
                agent_trace, ids, model=model, augmented=arm == LLM_ARMS[1],
                seed=seed, step=step, budget=budget,
            )
            analyst = [tool["result"] for tool in agent_trace["tool_calls"] if tool["tool"] == "data_analyst"]
            require(analyst == [data_analyst(observed)], f"Analyst used different observations for {key}")
            require(set(agent_trace["decision"]["evidence_ids"]) <= set(observed),
                    f"Agent cited evidence unavailable before this round for {key}")
            menu_ranks = {entry["id"]: entry["rank"] for entry in trace["menu"]}
            require(all(int(row["rf_rank"]) == menu_ranks[row["id"]] for row in batch),
                    f"Menu ranks differ from selected ledger for {key}")
            traces.append(trace)
        if arm == ARMS[1]:
            require(greedy_overlap == 1.0, f"Greedy control selections fail rank check for {key}")
        selections.append(set(ids))
        overlaps.append(greedy_overlap)
        eligible.difference_update(ids)
        observed.update({mix: {field: float(table.at[mix, field]) for field in (*FEATURES, "strength")}
                         for mix in ids})

    model, knowledge = condition_for(model, arm)
    measurements = schedule["rounds"] * budget
    cell = {
        "seed": seed, "model": model, "knowledge": knowledge, "measurements": measurements,
        "initial_best_mpa": initial_best, "final_best_mpa": incumbent,
        "gain_vs_initial_best_mpa": incumbent - initial_best, "top_hits": total_hits,
        "top_hit_rate": total_hits / measurements,
        "first_hit_tests": first_hit, "first_hit_censored": first_hit is None,
        "mean_greedy_overlap": mean(overlaps),
        "seeded_menu_fraction": sum(trace["seeded_picks"] for trace in traces) / measurements if traces else None,
    }
    if not traces:
        return cell, {"selections": selections}, {}
    usage = {
        "decisions": len(traces),
        "provider_responses": sum(trace["calls"] for trace in traces),
        **{key: sum(trace["usage"][key] for trace in traces) for key in USAGE_FIELDS},
        "requested_models": set.union(*(trace["requested_models"] for trace in traces)),
        "returned_models": set.union(*(trace["returned_models"] for trace in traces)),
        "generation_settings": set.union(*(trace["generation_settings"] for trace in traces)),
        "runtime_versions": set.union(*(trace["runtime_versions"] for trace in traces)),
    }
    selection = {
        "selections": selections,
        "initial": traces[0],
        "menu_picks": [
            {key: trace[key] for key in ("greedy_rank_picks", "other_high_rank_picks", "seeded_picks")}
            for trace in traces
        ],
    }
    return cell, selection, usage


def load_study(folder, *, model, arms, table, schedule, source_hash):
    status = read_json(folder / "status.json")
    require(status["status"] == "complete", f"{model} run is incomplete ({status['status']})")
    require(read_json(folder / "source.json")["sha256"] == source_hash,
            f"{model} raw data hash does not match")
    require(tuple(read_json(folder / "config.json")["arms"]) == arms, f"Incorrect arms for {model}")
    rounds, batches = {}, defaultdict(list)
    for row in load_records(folder / "rounds.csv"):
        key = (int(row["seed"]), row["arm"], int(row["round"]))
        require(key not in rounds, f"Duplicate round metric for {model} {key}")
        rounds[key] = row
    for row in load_records(folder / "selections.csv"):
        batches[int(row["seed"]), row["arm"], int(row["round"])].append(row)
    keys = {(seed, arm, step) for seed in schedule["seeds"] for arm in arms
            for step in range(1, schedule["rounds"] + 1)}
    require(set(rounds) == set(batches) == keys, f"Missing or unexpected scheduled rounds for {model}")
    require(status["measurements"] == len(keys) * schedule["budget"], f"Wrong measurement count for {model}")
    require(status["arms"] == len(schedule["seeds"]) * len(arms), f"Wrong completed arm count for {model}")

    splits, cells, selections, usages = {}, {}, {}, {}
    for seed in schedule["seeds"]:
        split = read_json(folder / "splits" / f"seed-{seed}.json")
        require(split == make_split(list(table.index), seed), f"Split manifest does not reconstruct for {model}/{seed}")
        splits[seed] = strict_digest(split)
        for arm in arms:
            cell, selection, usage = audit_arm(
                folder, seed=seed, arm=arm, model=model, schedule=schedule, split=split,
                table=table, rounds=rounds, batches=batches,
            )
            key = seed, cell["model"], cell["knowledge"]
            cells[key], selections[key] = cell, selection
            if usage:
                usages[key] = usage
    return splits, cells, selections, usages


def build_contrasts(by_condition: dict) -> dict:
    ds_off, ds_on, son_off, son_on = (by_condition[key] for key in CONDITIONS)
    return {
        "knowledge_on_minus_off_deepseek": ds_on - ds_off,
        "knowledge_on_minus_off_sonnet": son_on - son_off,
        "sonnet_minus_deepseek_knowledge_off": son_off - ds_off,
        "sonnet_minus_deepseek_knowledge_on": son_on - ds_on,
        "interaction_sonnet_minus_deepseek": (son_on - son_off) - (ds_on - ds_off),
    }


def summarize_by_seed(cells, seeds):
    row_data = {}
    for seed in seeds:
        for outcome in OUTCOMES:
            values = {key: cells[(seed, *key)][outcome] for key in CONDITIONS}
            contrasts = build_contrasts(values)
            if outcome != "seeded_menu_fraction":
                greedy = cells[seed, "greedy", "na"][outcome]
                contrasts.update({
                    f"{model}_{knowledge}_minus_shared_greedy": values[model, knowledge] - greedy
                    for model, knowledge in CONDITIONS
                })
            for contrast, value in contrasts.items():
                row_data.setdefault((seed, contrast), {"seed": seed, "contrast": contrast})[outcome] = value
    return [
        {**row, **{outcome: row.get(outcome) for outcome in OUTCOMES}}
        for row in row_data.values()
    ]


def summarize_contrasts(paired):
    grouped = defaultdict(list)
    for row in paired:
        for outcome in OUTCOMES:
            if row[outcome] is not None:
                grouped[row["contrast"], outcome].append(row[outcome])
    return [
        {"contrast": contrast, "outcome": metric, "n_paired_seeds": len(values),
         "mean": mean(values), "median": median(values), "min": min(values), "max": max(values),
         "positive": sum(value > 1e-10 for value in values),
         "tied": sum(abs(value) <= 1e-10 for value in values),
         "negative": sum(value < -1e-10 for value in values)}
        for (contrast, metric), values in grouped.items()
    ]


def compare_selections(selections, seeds, budget):
    for seed in seeds:
        initial_menus = [selections[(seed, *cell)]["initial"]["menu"] for cell in CONDITIONS]
        require(all(_same_menu(initial_menus[0], menu) for menu in initial_menus[1:]),
                f"Initial candidate menus differ for seed {seed}")
        for knowledge in ("off", "on"):
            prompts = {selections[seed, model, knowledge]["initial"]["proposal_digest"] for model in MODELS}
            require(len(prompts) == 1, f"Initial proposal instructions differ for {knowledge=} {seed=}")
        require(
            selections[seed, MODELS[0], "off"]["initial"]["proposal_digest"]
            != selections[seed, MODELS[0], "on"]["initial"]["proposal_digest"],
            f"Knowledge treatment did not change the first-round proposal context for {seed=}",
        )
        for left, right in (
            (CONDITIONS[0], CONDITIONS[1]), (CONDITIONS[2], CONDITIONS[3]),
            (CONDITIONS[0], CONDITIONS[2]), (CONDITIONS[1], CONDITIONS[3]),
        ):
            left_rounds = selections[(seed, *left)]["selections"]
            right_rounds = selections[(seed, *right)]["selections"]
            for step, (left_round, right_round) in enumerate(zip(left_rounds, right_rounds, strict=True), 1):
                yield {
                    "seed": seed, "round": step, "left_model": left[0], "left_knowledge": left[1],
                    "right_model": right[0], "right_knowledge": right[1],
                    "common_selections": len(left_round & right_round), "batch_overlap": len(left_round & right_round) / budget,
                    "jaccard": len(left_round & right_round) / len(left_round | right_round),
                }


def summarize_usage(usage, conditions=CONDITIONS):
    result = []
    for model, knowledge in conditions:
        runs = [value for (_, backend, kb), value in usage.items() if (backend, kb) == (model, knowledge)]
        row = {"model": model, "knowledge": knowledge}
        for key in ("decisions", "provider_responses", *USAGE_FIELDS):
            row[key] = sum(run[key] for run in runs)
        if model in ("deepseek", "gpt"):
            row["cache_creation_input_tokens"] = None
        for key in ("requested_models", "returned_models", "generation_settings", "runtime_versions"):
            row[key] = " | ".join(sorted(set.union(*(run[key] for run in runs))))
        result.append(row)
    return result


def analyze(baseline: Path, sonnet: Path, output: Path, data_path: Path = DEFAULT_DATA) -> dict:
    baseline, sonnet, output = (path.resolve() for path in (baseline, sonnet, output))
    for source in (baseline, sonnet):
        require(output != source and not output.is_relative_to(source), "Output would modify input evidence")
    configs = [read_json(path / "config.json") for path in (baseline, sonnet)]
    schedule = match_schedule(*configs)
    verify_frozen_inputs(baseline, sonnet, configs[1], schedule["seeds"])
    table, dataset = read_dataset(data_path)
    source_hash = dataset["sha256"]
    deepseek_splits, cells, selections, usage = load_study(
        baseline, model=MODELS[0], arms=ARMS, table=table, schedule=schedule, source_hash=source_hash,
    )
    sonnet_splits, other_cells, other_selections, other_usage = load_study(
        sonnet, model=MODELS[1], arms=LLM_ARMS, table=table, schedule=schedule, source_hash=source_hash,
    )
    require(deepseek_splits == sonnet_splits, "Effective assay splits differ")
    cells.update(other_cells)
    selections.update(other_selections)
    usage.update(other_usage)
    seeds = schedule["seeds"]
    paired = summarize_by_seed(cells, seeds)
    overlap = list(compare_selections(selections, seeds, schedule["budget"]))

    status = {
        "status": "complete",
        "baseline_mode": configs[0]["mode"],
        "sonnet_mode": configs[1]["mode"],
        "dataset_sha256": source_hash,
        "ordered_split_sha256_by_seed": deepseek_splits,
        "matched_schedule": schedule,
        "source_status": {model: read_json(path / "status.json") for model, path in zip(MODELS, (baseline, sonnet))},
        "source_file_sha256": {
            model: {
                name: sha256((path / name).read_bytes()).hexdigest()
                for name in ("config.json", "source.json", "selections.csv", "rounds.csv", "status.json")
            }
            for model, path in zip(MODELS, (baseline, sonnet))
        },
        "checked": ["frozen baseline manifest and Sonnet protocol digest",
                    "matched recorded assay conditions and splits", "ordered initial observations",
                    "ledger target, budget, threshold and incumbent", "live response model and usage",
                    "common round-1 menu across four cells", "matching round-1 messages and tools within knowledge level"],
        "inference": "Descriptive paired contrasts across seed blocks of the same finite recipe catalog",
        "interaction_direction": "(Sonnet on - Sonnet off) - (DeepSeek on - DeepSeek off)",
        "usage_provenance": {
            "responses": "Recorded provider response usage across all calls in each completed decision.",
            "input_tokens": "Cache-inclusive normalized input tokens.",
            "provider_primary_input_tokens": "Native prompt_tokens for Chat Completions and input_tokens for Messages.",
            "chat_completions": "Prompt tokens include cache hits. Cache creation is not reported.",
            "messages": "Total input is input_tokens plus cache_creation_input_tokens plus cache_read_input_tokens.",
            "reference_chat_completions": "https://api-docs.deepseek.com/api/create-chat-completion/",
            "reference_messages": "https://platform.claude.com/docs/en/api/typescript/messages",
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    require(not any(output.iterdir()), f"Output directory must be empty: {output}")
    conditions = [(control, "na") for control in CONTROLS] + list(CONDITIONS)
    write_csv(output / "cells.csv", [cells[(seed, *key)] for seed in seeds for key in conditions])
    write_csv(output / "cell_summary.csv", [
        {"model": model, "knowledge": knowledge, "n_seeds": len(seeds),
         **{
             outcome: mean(cells[seed, model, knowledge][outcome] for seed in seeds)
             for outcome in ("final_best_mpa", *OUTCOMES) if cells[seeds[0], model, knowledge][outcome] is not None
         }}
        for model, knowledge in conditions
    ])
    write_csv(output / "contrasts_by_seed.csv", paired)
    write_csv(output / "contrast_summary.csv", summarize_contrasts(paired))
    write_csv(output / "selection_overlap.csv", overlap)
    write_csv(output / "usage.csv", summarize_usage(usage))
    write_json(output / "validation.json", status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("evidence/concrete"))
    parser.add_argument("--sonnet", type=Path, default=Path("evidence/sonnet"))
    parser.add_argument("--output", type=Path, default=Path("evidence/factorial"))
    args = parser.parse_args()
    analyze(args.baseline, args.sonnet, args.output)
    print(f"Verified and summarized the matched factorial experiment in {args.output}")


if __name__ == "__main__":
    main()
