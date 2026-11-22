"""Run and compare fixed-budget acquisition policies for measured GFP fluorescence."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Mapping, Sequence

import pandas as pd

from gfp_data import GFPDataset, Variant, fit_predictor, make_split, metrics

SEEDS = (17, 29, 43)
ROUNDS = 3
BUDGET = 5
MAX_MUTATIONS = 4
ELITE_MENU_SIZE = 24
EXPLORATORY_MENU_SIZE = 24
DOMAIN = "gfp_fluorescence"
ARMS = ("A_random", "B_model_only", "C_llm", "D_knowledge_llm")
CONTROL_ARMS = ARMS[:2]
LLM_ARMS = ARMS[2:]
SPLIT_COUNTS = {"initial": 96, "validation": 120, "pool": 384, "test": 120}
Selector = Callable[..., Sequence[str]]


class MeasuredGFPOracle:
    """Expose fluorescence for complete, distinct batches from one arm's measured pool."""

    def __init__(self, measurements: Mapping[str, float], budget: int) -> None:
        if budget < 1:
            raise ValueError("GFP oracle budget must be positive")
        if any(not math.isfinite(value) for value in measurements.values()):
            raise ValueError("The GFP oracle requires finite measured values")
        self._measurements = dict(measurements)
        self._budget = budget

    def reveal(self, batch: Sequence[str]) -> dict[str, float]:
        if len(batch) != self._budget or len(set(batch)) != self._budget:
            raise ValueError("The GFP oracle requires one complete batch of distinct IDs")
        if not set(batch) <= self._measurements.keys():
            raise ValueError("A requested GFP candidate is unavailable or was already measured in this arm")
        return {candidate: self._measurements.pop(candidate) for candidate in batch}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")


def _open_run_directory(output: Path) -> None:
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Use a fresh GFP run directory: {output}")
    output.mkdir(parents=True, exist_ok=True)


def _variant_row(variant: Variant) -> dict:
    return {
        "id": variant.id,
        "mutations": variant.mutant,
        "sequence": variant.sequence,
        "edits": list(variant.edits),
    }


def _rank(eligible: Mapping[str, Variant], scores: Mapping[str, float]) -> list[str]:
    return sorted(eligible, key=lambda candidate: (-scores[candidate], candidate))


def build_menu(
    eligible: Mapping[str, Variant],
    scores: Mapping[str, float],
    uncertainty: Mapping[str, float],
    *,
    seed: int,
    round_number: int,
) -> list[dict]:
    """Create a label-free roster from an arm's observable, contemporaneous fit."""
    if set(scores) != set(eligible) or set(uncertainty) != set(eligible):
        raise ValueError("Every eligible GFP variant requires exactly one score and uncertainty")
    if any(not math.isfinite(value) for value in scores.values()):
        raise ValueError("Candidate predictions must be finite")
    if any(not math.isfinite(value) or value < 0 for value in uncertainty.values()):
        raise ValueError("Candidate uncertainty must be finite and nonnegative")
    ranked = _rank(eligible, scores)
    ranks = {candidate: rank for rank, candidate in enumerate(ranked, 1)}
    elite = ranked[:ELITE_MENU_SIZE]
    remainder = ranked[ELITE_MENU_SIZE:]
    randomizer = random.Random(f"gfp-menu:{seed}:{round_number}")
    exploratory = randomizer.sample(remainder, min(EXPLORATORY_MENU_SIZE, len(remainder)))
    displayed = elite + exploratory
    randomizer.shuffle(displayed)
    return [
        {
            **_variant_row(eligible[candidate]),
            "alias": f"C{position:02d}",
            "prediction": scores[candidate],
            "uncertainty": uncertainty[candidate],
            "rank": ranks[candidate],
            "source": "rf_elite" if ranks[candidate] <= ELITE_MENU_SIZE else "seeded_non_elite",
        }
        for position, candidate in enumerate(displayed, 1)
    ]


def make_agent_request(
    dataset: GFPDataset,
    observed: Mapping[str, float],
    menu: list[dict],
    *,
    seed: int,
    round_number: int,
    budget: int,
    knowledge: bool,
    latest_feedback_ids: Sequence[str] = (),
) -> dict:
    roster_ids = {row["id"] for row in menu}
    if len(roster_ids) != len(menu) or roster_ids & observed.keys():
        raise ValueError("A GFP roster must exclude all observations and duplicate IDs")
    if any(candidate not in observed for candidate in latest_feedback_ids):
        raise ValueError("Recent GFP feedback must belong to this arm's revealed ledger")
    allowed_menu_fields = {"id", "mutations", "sequence", "edits", "alias", "prediction", "uncertainty", "rank", "source"}
    if any(set(row) != allowed_menu_fields for row in menu):
        raise ValueError("GFP menu has unexpected fields")
    return {
        "domain": DOMAIN,
        "reference_sequence": dataset.wild_type,
        "max_mutations": MAX_MUTATIONS,
        "seed": seed,
        "round_number": round_number,
        "budget": budget,
        "knowledge": knowledge,
        "fitness_direction": "maximize",
        "latest_feedback_ids": list(latest_feedback_ids),
        "observed": [
            {**_variant_row(dataset.variants[candidate]), "fitness": value}
            for candidate, value in observed.items()
        ],
        "menu": menu,
    }


def prediction_view(
    dataset: GFPDataset, observed: Mapping[str, float], eligible: Mapping[str, Variant], seed: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """Infer contender values from an explicitly supplied GFP observation ledger."""
    predictor = fit_predictor(
        [dataset.variants[candidate] for candidate in observed],
        list(observed.values()), seed, model="rf",
    )
    candidates = list(eligible.values())
    return (
        dict(zip(eligible, map(float, predictor.predict(candidates)), strict=True)),
        dict(zip(eligible, map(float, predictor.uncertainty(candidates)), strict=True)),
    )


def _prediction_metrics(
    dataset: GFPDataset,
    observed: Mapping[str, float],
    target: list[str],
    *,
    seed: int,
    arm: str,
    stage: str,
) -> list[dict]:
    features = [dataset.variants[candidate] for candidate in target]
    truth = [dataset.scores[candidate] for candidate in target]
    rows = []
    for name in ("ridge", "rf"):
        model = fit_predictor(
            [dataset.variants[candidate] for candidate in observed], list(observed.values()), seed, model=name,
        )
        rows.append({
            "seed": seed, "arm": arm, "stage": stage, "predictor": name, "n": len(target),
            **metrics(truth, model.predict(features), top_k=min(10, len(target))),
        })
    return rows


def _checked_batch(batch: Sequence[str], allowed: set[str], budget: int) -> list[str]:
    if isinstance(batch, str):
        raise ValueError("GFP recommendations must be a sequence of candidate IDs")
    selected = list(batch)
    if len(selected) != budget or any(not isinstance(candidate, str) for candidate in selected):
        raise ValueError("GFP selection does not match the batch budget")
    if len(set(selected)) != budget or not set(selected) <= allowed:
        raise ValueError("GFP selection contains a duplicate or outside-roster ID")
    return selected


def _run_arm(
    dataset: GFPDataset,
    split: dict[str, list[str]],
    arm_path: Path,
    *,
    seed: int,
    arm: str,
    rounds: int,
    budget: int,
    backend: str,
    selector: Selector | None,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    observed = {candidate: float(dataset.scores[candidate]) for candidate in split["initial"]}
    wild_type_id = dataset.wild_type_id
    if wild_type_id not in observed:
        raise ValueError("The measured GFP wild-type reference must be in initial observations")
    threshold = observed[wild_type_id]
    incumbent_id = max(observed, key=observed.__getitem__)
    incumbent = observed[incumbent_id]
    initial_best, initial_best_id = incumbent, incumbent_id
    available = {candidate: dataset.variants[candidate] for candidate in split["pool"]}
    oracle = MeasuredGFPOracle({candidate: float(dataset.scores[candidate]) for candidate in available}, budget)
    _write_json(arm_path / "initial_observed.json", observed)
    prediction_rows = _prediction_metrics(
        dataset, observed, split["validation"], seed=seed, arm=arm, stage="initial_validation",
    )

    round_rows: list[dict] = []
    selections: list[dict] = []
    latest_feedback: list[str] = []
    for round_number in range(1, rounds + 1):
        scores, uncertainty = prediction_view(dataset, observed, available, seed)
        ranked = _rank(available, scores)
        ranks = {candidate: rank for rank, candidate in enumerate(ranked, 1)}
        greedy = ranked[:budget]
        menu = build_menu(available, scores, uncertainty, seed=seed, round_number=round_number)
        menu_by_id = {row["id"]: row for row in menu}
        prefix = f"round-{round_number:02d}"
        if arm == CONTROL_ARMS[0]:
            rng = random.Random(f"gfp-random:{seed}:{round_number}")
            proposed = rng.sample(sorted(available), budget)
        elif arm == CONTROL_ARMS[1]:
            proposed = greedy
        else:
            if selector is None:
                raise ValueError("LLM GFP arms require a verified selector")
            request = make_agent_request(
                dataset, observed, menu, seed=seed, round_number=round_number, budget=budget,
                knowledge=arm == LLM_ARMS[1], latest_feedback_ids=latest_feedback,
            )
            _write_json(arm_path / f"{prefix}-request.json", request)
            proposed = selector(request, log_path=arm_path / f"{prefix}-agent.json")
        allowed = set(menu_by_id) if arm in LLM_ARMS else set(available)
        selected = _checked_batch(proposed, allowed, budget)
        _write_json(arm_path / f"{prefix}-selection.json", {
            "selected_ids": selected, "model_only_ids": greedy, "eligible_rf_ranks": ranks,
        })
        revealed = oracle.reveal(selected)
        chosen = []
        for position, candidate in enumerate(selected, 1):
            observed_fitness = revealed[candidate]
            variant = available[candidate]
            row = {
                "seed": seed, "arm": arm, "backend": backend, "round": round_number,
                "batch_position": position, "query_number": (round_number - 1) * budget + position,
                **_variant_row(variant), "mutation_count": len(variant.edits),
                "nearest_observed_edits": min(
                    sum(old != new for old, new in zip(dataset.variants[prior].sequence, variant.sequence, strict=True))
                    for prior in observed
                ),
                "prediction": scores[candidate], "uncertainty": uncertainty[candidate], "rank": ranks[candidate],
                "menu_source": menu_by_id[candidate]["source"] if candidate in menu_by_id else "outside_llm_roster",
                "fitness": observed_fitness, "measurement_source": "pinned_gfp_archive",
                "above_wild_type": observed_fitness > threshold, "residual": observed_fitness - scores[candidate],
                "is_contemporaneous_greedy": candidate in greedy,
            }
            if observed_fitness > incumbent:
                incumbent = observed_fitness
                incumbent_id = candidate
            chosen.append(row)
        observed.update(revealed)
        for candidate in selected:
            del available[candidate]
        latest_feedback = selected
        selections.extend(chosen)
        round_row = {
            "seed": seed, "arm": arm, "backend": backend, "round": round_number,
            "n_measured": len(chosen), "n_revealed_total": len(selections), "n_observed": len(observed),
            "batch_best": max(revealed.values()), "batch_mean": sum(revealed.values()) / len(revealed),
            "best_so_far": incumbent, "incumbent_id": incumbent_id,
            "initial_best": initial_best, "initial_best_id": initial_best_id, "wild_type_fitness": threshold,
            "gain_vs_initial_best": incumbent - initial_best, "gain_vs_wild_type": incumbent - threshold,
            "above_wild_type_hits": sum(row["above_wild_type"] for row in chosen),
            "greedy_overlap_at_k": sum(row["is_contemporaneous_greedy"] for row in chosen) / budget,
        }
        round_rows.append(round_row)
        _write_json(arm_path / f"{prefix}.json", {
            **round_row, "selected": chosen, "contemporaneous_greedy_ids": greedy,
        })

    prediction_rows.extend(_prediction_metrics(
        dataset, observed, split["test"], seed=seed, arm=arm, stage="final_test",
    ))
    first_hit = next((row["query_number"] for row in selections if row["above_wild_type"]), None)
    method = {
        **round_rows[-1],
        "above_wild_type_hits": sum(row["above_wild_type"] for row in selections),
        "first_above_wild_type_query": first_hit,
        "mean_greedy_overlap_at_k": sum(row["greedy_overlap_at_k"] for row in round_rows) / rounds,
        "rank_outside_elite": sum(row["rank"] > ELITE_MENU_SIZE for row in selections),
    }
    _write_json(arm_path / "status.json", {"status": "complete", "measurements": len(selections), "rounds": rounds})
    return round_rows, selections, prediction_rows, method


def _method_summary(methods: list[dict]) -> list[dict]:
    summary = []
    for backend, arm in sorted({(row["backend"], row["arm"]) for row in methods}):
        found = [row for row in methods if (row["backend"], row["arm"]) == (backend, arm)]
        fields = ("best_so_far", "gain_vs_initial_best", "gain_vs_wild_type", "mean_greedy_overlap_at_k")
        total = sum(row["n_revealed_total"] for row in found)
        hits = sum(row["above_wild_type_hits"] for row in found)
        summary.append({
            "backend": backend, "arm": arm, "seeds": len(found), "measurements": total,
            **{f"mean_{field}": sum(row[field] for row in found) / len(found) for field in fields},
            "above_wild_type_hits": hits, "above_wild_type_hit_rate": hits / total,
            "rank_outside_elite": sum(row["rank_outside_elite"] for row in found),
        })
    return summary


def _contrasts(methods: list[dict]) -> list[dict]:
    lookup = {(row["seed"], row["backend"], row["arm"]): row for row in methods}
    rows = []
    for seed in sorted({row["seed"] for row in methods}):
        found = [row for row in methods if row["seed"] == seed]
        for backend in sorted({row["backend"] for row in found}):
            for left, right in (
                (CONTROL_ARMS[0], CONTROL_ARMS[1]),
                (CONTROL_ARMS[1], LLM_ARMS[0]),
                (CONTROL_ARMS[1], LLM_ARMS[1]),
                LLM_ARMS,
            ):
                a = lookup.get((seed, backend, left))
                if a is None and left in CONTROL_ARMS:
                    a = next((row for row in found if row["arm"] == left), None)
                b = lookup.get((seed, backend, right))
                if a is None or b is None:
                    continue
                rows.append({
                    "seed": seed, "backend": backend, "baseline_backend": a["backend"],
                    "baseline_arm": left, "treatment_arm": right,
                    "delta_best": b["best_so_far"] - a["best_so_far"],
                    "delta_gain_vs_initial_best": b["gain_vs_initial_best"] - a["gain_vs_initial_best"],
                    "delta_above_wild_type_hits": b["above_wild_type_hits"] - a["above_wild_type_hits"],
                })
    return rows


def _prediction_errors(selections: list[dict]) -> list[dict]:
    examples = []
    for seed, backend, arm in sorted({(row["seed"], row["backend"], row["arm"]) for row in selections}):
        rows = [row for row in selections if (row["seed"], row["backend"], row["arm"]) == (seed, backend, arm)]
        for direction, chooser in (("largest_underestimate", max), ("largest_overestimate", min)):
            row = chooser(rows, key=lambda item: item["residual"])
            examples.append({"example": direction, **row})
    return examples


def run_evolution(
    dataset: GFPDataset,
    output: Path,
    *,
    selector: Selector | None = None,
    backend: str = "deepseek",
    seeds: tuple[int, ...] = SEEDS,
    rounds: int = ROUNDS,
    budget: int = BUDGET,
    arms: tuple[str, ...] = ARMS,
    mode: str = "formal",
    workers: int = 6,
    protocol_path: Path | None = None,
) -> dict:
    """Execute independent GFP histories with shared, seed-specific starting observations."""
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("Use distinct nonnegative GFP split seeds")
    if not arms or len(set(arms)) != len(arms) or not set(arms) <= set(ARMS):
        raise ValueError("Unknown or duplicate GFP arms")
    if workers < 1 or rounds < 1 or budget < 1 or budget > ELITE_MENU_SIZE:
        raise ValueError("Invalid GFP worker count, schedule or budget")
    if set(arms) & set(LLM_ARMS) and selector is None:
        raise ValueError("A selector is required for GFP LLM arms")
    _open_run_directory(output)
    split_keys = {seed: make_split(dataset, seed, **SPLIT_COUNTS) for seed in seeds}
    for seed, split in split_keys.items():
        reported = {key: len(value) for key, value in split.items()}
        if reported != SPLIT_COUNTS:
            raise ValueError(f"Unexpected frozen GFP split sizes for seed {seed}: {reported}")
        joined = list(itertools.chain.from_iterable(split.values()))
        if len(joined) != len(set(joined)) or not set(joined) <= dataset.variants.keys():
            raise ValueError(f"Frozen GFP split is not disjoint or has unknown records for seed {seed}")
        if rounds * budget > len(split["pool"]):
            raise ValueError("Total budget exceeds the measured GFP pool")
        _write_json(output / "splits" / f"seed-{seed}.json", split)
    config = {
        "domain": DOMAIN, "backend": backend, "seeds": list(seeds), "rounds": rounds, "budget": budget,
        "arms": list(arms), "mode": mode, "split_counts": SPLIT_COUNTS, "max_mutations": MAX_MUTATIONS,
        "menu": {"rf_elite": ELITE_MENU_SIZE, "seeded_non_elite": EXPLORATORY_MENU_SIZE},
        "metric": "maximum measured GFP assay score", "primary": "gain in incumbent from shared initial best",
        "secondary_hit": "measured assay score strictly higher than the measured wild-type anchor",
        "predictor": "fixed random forest refit on each arm's own observations before every choice",
        "validation": "initial diagnostic; held-out final scoring only after acquisition",
        "prediction_top_k": "precision and recall at 10 for the measured upper decile of the target split",
    }
    if protocol_path is not None:
        protocol_bytes = protocol_path.read_bytes()
        (output / "protocol-freeze.md").write_bytes(protocol_bytes)
        config["protocol_sha256"] = hashlib.sha256(protocol_bytes).hexdigest()
    _write_json(output / "config.json", config)
    _write_json(output / "source.json", dataset.provenance)
    _write_json(output / "status.json", {"status": "running", "mode": mode})
    results: dict[tuple[int, str], tuple] = {}
    failures = []
    partial = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            (seed, arm): executor.submit(
                _run_arm, dataset, split_keys[seed], output / "arms" / f"seed-{seed}" / arm,
                seed=seed, arm=arm, rounds=rounds, budget=budget, backend=backend, selector=selector,
            )
            for seed in seeds for arm in arms
        }
        for (seed, arm), future in futures.items():
            try:
                results[seed, arm] = future.result()
            except Exception as exc:
                failure = {"seed": seed, "arm": arm, "error_type": type(exc).__name__, "message": str(exc)}
                failures.append(failure)
                arm_path = output / "arms" / f"seed-{seed}" / arm
                for checkpoint in sorted(arm_path.glob("round-[0-9][0-9].json")):
                    partial.extend(_read_json(checkpoint)["selected"])
                _write_json(arm_path / "status.json", {"status": "failed", **failure})
    rounds_out = [row for details in results.values() for row in details[0]]
    selections = [row for details in results.values() for row in details[1]]
    predictions = [row for details in results.values() for row in details[2]]
    methods = [details[3] for details in results.values()]
    _write_csv(output / "rounds.csv", rounds_out)
    _write_csv(output / "selections.csv", selections)
    _write_csv(output / "predictive.csv", predictions)
    _write_csv(output / "methods.csv", methods)
    _write_csv(output / "method_summary.csv", _method_summary(methods))
    _write_csv(output / "comparisons.csv", _contrasts(methods))
    _write_csv(output / "failure_examples.csv", _prediction_errors(selections))
    if partial:
        _write_csv(output / "partial_selections.csv", partial)
    _write_json(output / "failures.json", failures)
    status = {
        "status": "partial" if failures else "complete", "mode": mode, "arms": len(results),
        "expected_arms": len(seeds) * len(arms), "measurements": len(selections),
        "partial_measurements": len(partial), "failed": len(failures),
    }
    _write_json(output / "status.json", status)
    return status


def aggregate_studies(primary_dir: Path, extra_dirs: Mapping[str, Path], output: Path) -> dict:
    """Check matched GFP setups and aggregate complete provider evidence."""
    base = _read_json(primary_dir / "config.json")
    expected = _read_json(primary_dir / "source.json")
    baseline_name = base["backend"]
    if baseline_name in extra_dirs:
        raise ValueError("The primary GFP run cannot be replaced by an extra provider")
    _open_run_directory(output)
    methods = []
    all_rounds = []
    all_selections = []
    provider_status = {}
    for backend, directory in {baseline_name: primary_dir, **extra_dirs}.items():
        config = _read_json(directory / "config.json")
        if config["backend"] != backend:
            raise ValueError(f"Unexpected GFP run identity for {backend}")
        for key in ("domain", "seeds", "rounds", "budget", "split_counts", "max_mutations", "menu"):
            if config[key] != base[key]:
                raise ValueError(f"GFP run differs in {key} for {backend}")
        if _read_json(directory / "source.json") != expected:
            raise ValueError(f"GFP source metadata differs for {backend}")
        for seed in base["seeds"]:
            relative = f"splits/seed-{seed}.json"
            if _read_json(directory / relative) != _read_json(primary_dir / relative):
                raise ValueError(f"GFP split differs for {backend} seed {seed}")
        provider_status[backend] = _read_json(directory / "status.json")
        if provider_status[backend]["arms"]:
            methods.extend(pd.read_csv(directory / "methods.csv").to_dict("records"))
            all_rounds.extend(pd.read_csv(directory / "rounds.csv").to_dict("records"))
            all_selections.extend(pd.read_csv(directory / "selections.csv").to_dict("records"))
    contrasts = _contrasts(methods)
    primary_knowledge = {
        row["seed"]: row["delta_best"] for row in contrasts if
        (row["backend"], row["baseline_arm"], row["treatment_arm"]) == (baseline_name, *LLM_ARMS)
    }
    interactions = [
        {
            "seed": row["seed"], "backend": row["backend"], "reference_backend": baseline_name,
            "delta_knowledge_best": row["delta_best"],
            "reference_delta_knowledge_best": primary_knowledge[row["seed"]],
            "difference_in_differences": row["delta_best"] - primary_knowledge[row["seed"]],
        }
        for row in contrasts if row["backend"] != baseline_name and
        (row["baseline_arm"], row["treatment_arm"]) == LLM_ARMS and row["seed"] in primary_knowledge
    ]
    _write_csv(output / "methods.csv", methods)
    _write_csv(output / "rounds.csv", all_rounds)
    _write_csv(output / "selections.csv", all_selections)
    _write_csv(output / "method_summary.csv", _method_summary(methods))
    _write_csv(output / "paired_effects.csv", contrasts)
    _write_csv(output / "knowledge_interactions.csv", interactions)
    _write_csv(output / "failure_examples.csv", _prediction_errors(all_selections))
    status = {
        "status": "complete" if all(row["status"] == "complete" for row in provider_status.values()) else "partial",
        "providers": provider_status, "records": len(all_selections),
        "shared_config": {key: base[key] for key in ("seeds", "rounds", "budget", "split_counts", "menu")},
    }
    _write_json(output / "status.json", status)
    return status
