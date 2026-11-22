"""Verify and analyze model and knowledge effects in matched Concrete runs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from statistics import mean, median

from factorial_analysis import (
    CONTROLS, LLM_ARMS, OUTCOMES, condition_for, load_records, load_study, match_schedule,
    summarize_contrasts, summarize_usage, verify_frozen_inputs,
)
from factorial_runner import _same_menu
from nonbio import ARMS, DEFAULT_DATA, read_dataset, write_csv, write_json
from verify import read, require


REFERENCE = "deepseek"
DEFAULT_ADDONS = {
    "sonnet": Path("evidence/sonnet"),
    "gpt": Path("evidence/gpt"),
    "qwen": Path("evidence/qwen"),
}
KNOWLEDGE = ("off", "on")
DISTRIBUTION_FIELDS = (
    "rf_rank", "rf_tree_disagreement_mpa", "nearest_observed_distance", "water_to_binder",
    "prediction_mpa", "strength_mpa",
)
ROUND_FIELDS = (
    "tested", "round_best_mpa", "best_so_far_mpa", "gain_vs_initial_best_mpa",
    "round_top_hits", "top_hits", "first_hit_tests", "greedy_overlap_at_k",
)


def contrasts_for_seed(cells, seed, models):
    by_contrast = {}
    for outcome in OUTCOMES:
        values = {(model, kb): cells[seed, model, kb][outcome] for model in models for kb in KNOWLEDGE}
        knowledge_effects = {
            model: values[model, "on"] - values[model, "off"] for model in models
        }
        contrasts = {f"{model}_knowledge_on_minus_off": value for model, value in knowledge_effects.items()}
        contrasts.update({
            f"{right}_minus_{left}_knowledge_{kb}": values[right, kb] - values[left, kb]
            for left, right in combinations(models, 2) for kb in KNOWLEDGE
        })
        contrasts.update({
            f"interaction_{model}_minus_{REFERENCE}": knowledge_effects[model] - knowledge_effects[REFERENCE]
            for model in models if model != REFERENCE
        })
        if outcome != "seeded_menu_fraction":
            greedy = cells[seed, "greedy", "na"][outcome]
            contrasts.update({
                f"{model}_{kb}_minus_shared_greedy": value - greedy
                for (model, kb), value in values.items()
            })
        for contrast, value in contrasts.items():
            by_contrast.setdefault(contrast, dict.fromkeys(OUTCOMES))[outcome] = value
    return [
        {"seed": seed, "contrast": name, **values}
        for name, values in by_contrast.items()
    ]


def validate_selection_contexts(selections, seeds, models, budget):
    conditions = [(model, kb) for model in models for kb in KNOWLEDGE]
    for seed in seeds:
        reference = selections[seed, REFERENCE, "off"]["initial"]["menu"]
        for model, kb in conditions:
            current = selections[seed, model, kb]["initial"]
            expected = selections[seed, REFERENCE, kb]["initial"]
            require(_same_menu(reference, current["menu"]),
                    f"Initial candidate menu mismatch for {model}/{kb}/{seed}")
            require(current["proposal_digest"] == expected["proposal_digest"],
                    f"Initial proposal context mismatch for {model}/{kb}/{seed}")
        require(
            selections[seed, REFERENCE, "off"]["initial"]["proposal_digest"] !=
            selections[seed, REFERENCE, "on"]["initial"]["proposal_digest"],
            f"Knowledge switch did not change initial model input for {seed}",
        )
        for left, right in combinations(conditions, 2):
            if left[0] != right[0] and left[1] != right[1]:
                continue
            lhs = selections[(seed, *left)]["selections"]
            rhs = selections[(seed, *right)]["selections"]
            for step, (lhs_ids, rhs_ids) in enumerate(zip(lhs, rhs, strict=True), 1):
                shared = len(lhs_ids & rhs_ids)
                yield {
                    "seed": seed, "round": step, "left_model": left[0], "left_knowledge": left[1],
                    "right_model": right[0], "right_knowledge": right[1], "same_recipe_count": shared,
                    "batch_overlap": shared / budget, "jaccard": shared / len(lhs_ids | rhs_ids),
                }


def summarize_cells(cells, conditions, seeds):
    summary = []
    for model, kb in conditions:
        group = [cells[seed, model, kb] for seed in seeds]
        n_tests = sum(row["measurements"] for row in group)
        n_hits = sum(row["top_hits"] for row in group)
        first_hits = [row["first_hit_tests"] for row in group if row["first_hit_tests"] is not None]
        summary.append({
            "model": model, "knowledge": kb, "n_seeds": len(group), "total_measurements": n_tests,
            "total_top_hits": n_hits, "pooled_top_hit_rate": n_hits / n_tests,
            "seeds_with_hit": len(first_hits), "seeds_right_censored": len(group) - len(first_hits),
            "median_first_hit_among_hit_seeds": median(first_hits) if first_hits else None,
            **{
                metric: mean(row[metric] for row in group) if group[0][metric] is not None else None
                for metric in ("final_best_mpa", *OUTCOMES)
            },
        })
    return summary


def describe_distributions(paths):
    groups = defaultdict(list)
    for source, folder in paths.items():
        for row in load_records(folder / "selections.csv"):
            model, kb = condition_for(source, row["arm"])
            groups[int(row["seed"]), model, kb].append(row)
    return [
        {
            "seed": seed, "model": model, "knowledge": kb, "n_selections": len(rows),
            **{f"mean_{field}": mean(float(row[field]) for row in rows) for field in DISTRIBUTION_FIELDS},
            "mean_absolute_prediction_error_mpa": mean(
                abs(float(row["strength_mpa"]) - float(row["prediction_mpa"])) for row in rows
            ),
        }
        for (seed, model, kb), rows in groups.items()
    ]


def archive_rounds(paths):
    for model, path in paths.items():
        for row in load_records(path / "rounds.csv"):
            cohort, kb = condition_for(model, row["arm"])
            yield {
                "seed": row["seed"], "model": cohort, "knowledge": kb, "round": row["round"],
                **{metric: row[metric] for metric in ROUND_FIELDS},
            }


def analyze(baseline: Path, addons: dict[str, Path], output: Path, data_path: Path = DEFAULT_DATA) -> dict:
    require(addons and set(addons) <= set(DEFAULT_ADDONS), "Supply recognized and distinct add-on models")
    paths = {REFERENCE: baseline.resolve(), **{model: path.resolve() for model, path in addons.items()}}
    output = output.resolve()
    require(len(set(paths.values())) == len(paths), "A folder cannot count as two distinct model runs")
    require(all(not output.is_relative_to(path) for path in paths.values()),
            "Output directory cannot modify input evidence")
    baseline = paths[REFERENCE]
    baseline_config = read(baseline / "config.json")
    table, dataset = read_dataset(data_path)
    source_hash = dataset["sha256"]
    splits, cells, selections, usage = load_study(
        baseline, model=REFERENCE, arms=ARMS, table=table, schedule=baseline_config, source_hash=source_hash,
    )
    seeds = baseline_config["seeds"]
    for model, path in addons.items():
        path = path.resolve()
        config = read(path / "config.json")
        match_schedule(baseline_config, config)
        verify_frozen_inputs(baseline, path, config, seeds)
        matched_splits, matched_cells, matched_selections, matched_usage = load_study(
            path, model=model, arms=LLM_ARMS, table=table, schedule=baseline_config, source_hash=source_hash,
        )
        require(matched_splits == splits, f"{model} has unmatched assay splits")
        cells.update(matched_cells)
        selections.update(matched_selections)
        usage.update(matched_usage)
    models = tuple(paths)
    llm_conditions = [(model, kb) for model in models for kb in KNOWLEDGE]
    conditions = [(control, "na") for control in CONTROLS] + llm_conditions
    overlap = list(validate_selection_contexts(selections, seeds, models, baseline_config["budget"]))
    contrasts = [row for seed in seeds for row in contrasts_for_seed(cells, seed, models)]
    accounting = summarize_usage(usage, conditions=llm_conditions)
    for row in accounting:
        row["recorded_model_responses"] = row.pop("provider_responses")
    for model in models:
        used = [row for row in accounting if row["model"] == model]
        for field in ("requested_models", "returned_models", "generation_settings"):
            values = {row[field] for row in used}
            require(len(values) == 1 and " | " not in next(iter(values)),
                    f"Model or transport settings differ between {model} KB cells: {field}")
        if model != REFERENCE:
            require(used[0]["requested_models"] == read(paths[model] / "config.json")["model"],
                    f"{model} traces do not use the explicitly frozen requested model")
    output.mkdir(parents=True, exist_ok=True)
    require(not any(output.iterdir()), f"Output directory must be empty: {output}")
    status = {
        "status": "complete",
        "models": list(models),
        "knowledge_levels": list(KNOWLEDGE),
        "measured_dataset_sha256": source_hash,
        "ordered_splits_sha256": splits,
        "initial_menu_match": "same ordered candidate IDs, covariates, ranks and model estimates",
        "initial_prompt_and_tool_match": "equal instructions and input schema within each knowledge condition after transport normalization",
        "shared_controls": list(CONTROLS),
        "seeds": seeds,
        "rounds_per_arm": baseline_config["rounds"],
        "budget_per_round": baseline_config["budget"],
        "primary_endpoint": baseline_config["primary_endpoint"],
        "interaction_sign": "(add-on on - add-on off) - (DeepSeek on - DeepSeek off)",
        "inference_scope": "Paired descriptive comparisons across seed blocks of one finite catalog. Model levels include their native transports, generation settings and runtime.",
        "source_status": {model: read(path / "status.json") for model, path in paths.items()},
        "source_file_sha256": {
            model: {name: sha256((path / name).read_bytes()).hexdigest()
                    for name in ("config.json", "rounds.csv", "selections.csv", "status.json")}
            for model, path in paths.items()
        },
        "usage_units": {
            "input_tokens": "Cache-inclusive model input as normalized by each runtime.",
            "provider_primary_input_tokens": "Native primary input counter, excluding cache additions for Messages.",
            "recorded_model_responses": "Logged live remote responses or local generations.",
            "references": [
                "https://platform.claude.com/docs/en/api/typescript/messages",
                "https://api-docs.deepseek.com/api/create-chat-completion/",
            ],
        },
    }
    write_csv(output / "cells.csv", [cells[seed, *key] for seed in seeds for key in conditions])
    write_csv(output / "cell_summary.csv", summarize_cells(cells, conditions, seeds))
    write_csv(output / "round_cells.csv", list(archive_rounds(paths)))
    write_csv(output / "contrasts_by_seed.csv", contrasts)
    write_csv(output / "contrast_summary.csv", summarize_contrasts(contrasts))
    write_csv(output / "selection_overlap.csv", overlap)
    write_csv(output / "candidate_distribution.csv", describe_distributions(paths))
    write_csv(output / "menu_usage.csv", [
        {"seed": seed, "model": model, "knowledge": kb, "round": step, **picks}
        for seed in seeds for model, kb in llm_conditions
        for step, picks in enumerate(selections[seed, model, kb]["menu_picks"], 1)
    ])
    write_csv(output / "usage.csv", accounting)
    write_json(output / "validation.json", status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("evidence/concrete"))
    parser.add_argument("--addon", metavar="MODEL=PATH", action="append",
                        help="Specify complete add-on run(s). Defaults to Sonnet, GPT, and Qwen.")
    parser.add_argument("--output", type=Path, default=Path("evidence/across-models"))
    args = parser.parse_args()
    addons = DEFAULT_ADDONS.copy()
    if args.addon:
        addons = {}
        for spec in args.addon:
            model, marker, path = spec.partition("=")
            if not marker or not path or model not in DEFAULT_ADDONS or model in addons:
                parser.error(f"Invalid or repeated add-on: {spec}")
            addons[model] = Path(path)
    status = analyze(args.baseline, addons, args.output)
    print(f"Verified {len(status['models'])} matched model backends in {args.output}")


if __name__ == "__main__":
    main()
