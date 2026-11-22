"""Import measurements of the fixed GFP scaffold and issue the next GFP batch."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

from gfp_data import GFP_WT, make_variant

ASSAY = "GFP_AEQVI_Sarkisyan_2016"
MEASURED_COLUMNS = ("mutant", "fitness")
RECOMMENDATION_COLUMNS = ("mutation", "sequence", "predicted_fitness", "measured_fitness", "status", "decision")


@dataclass(frozen=True)
class Measurement:
    mutant: str
    fitness: float


def load_csv(path: Path, *, measured: bool) -> dict:
    required = set(MEASURED_COLUMNS if measured else ("mutant",))
    allowed = required | {"sequence"}
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = csv.DictReader(f)
        fields = rows.fieldnames or []
        if not required <= set(fields) <= allowed or len(fields) != len(set(fields)):
            raise ValueError(f"Expected columns {sorted(required)} with optional sequence: {path}")
        records = {}
        for row in rows:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed CSV row {rows.line_num}")
            variant = make_variant(row["mutant"].strip())
            if "sequence" in row and row["sequence"].strip() != variant.sequence:
                raise ValueError(f"Sequence does not match the mutation annotation at row {rows.line_num}")
            if variant.id in records:
                raise ValueError(f"Duplicate or equivalent variant on row {rows.line_num}")
            if measured:
                fitness = float(row["fitness"])
                if not math.isfinite(fitness):
                    raise ValueError(f"Nonfinite fluorescence on row {rows.line_num}")
                records[variant.id] = Measurement(variant.id, fitness)
            else:
                records[variant.id] = variant
    if not records:
        raise ValueError(f"Empty input: {path}")
    return records


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write_json(path: Path, obj) -> None:
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")


def _write_csv(path: Path, headers, rows) -> None:
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(headers)
        writer.writerows(rows)


def _load_state(path: Path) -> tuple[dict, str]:
    content = path.read_bytes()
    state = json.loads(content)
    if state["format"] != 1 or state["reference_sha256"] != _sha(GFP_WT.encode()) or state["assay"] != ASSAY:
        raise ValueError("State is incompatible with the GFP reference or score scale")
    observations = state["observations"]
    if (len({row["mutant"] for row in observations}) != len(observations)
            or not all(make_variant(row["mutant"]).id == row["mutant"] and math.isfinite(row["fitness"])
                       for row in observations)):
        raise ValueError("State contains invalid or duplicate GFP observations")
    return state, _sha(content)


def _propose(dataset, output: Path, observations: dict[str, Measurement], *, prior: Path | None,
             candidates: Path | None, backend: str, use_llm: bool, transport=None) -> dict:
    from gfp_data import fit_predictor
    from gfp_evolution import build_menu, make_agent_request

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a fresh output directory: {output}")
    previous, prior_sha = (_load_state(prior) if prior else (None, None))
    existing = ({row["mutant"]: Measurement(**row) for row in previous["observations"]} if previous else {})
    if set(existing) & set(observations):
        raise ValueError("An imported measurement duplicates an existing observation")
    visible = existing | observations
    if len(visible) < 4:
        raise ValueError("At least four distinct measured GFP variants are required")
    outstanding = set(previous["outstanding"]) if previous else set()
    outstanding.difference_update(observations)
    features = load_csv(candidates, measured=False) if candidates else dataset.variants
    excluded = set(visible) | outstanding
    eligible = {key: value for key, value in features.items() if key not in excluded}
    budget = min(5, len(eligible))
    if budget == 0:
        raise ValueError("No new legal GFP candidates are eligible")
    model = fit_predictor([make_variant(k) for k in visible], [item.fitness for item in visible.values()],
                          seed=17, model="rf")
    keys = list(eligible)
    variants = list(eligible.values())
    scores = dict(zip(keys, map(float, model.predict(variants)), strict=True))
    uncertainty = dict(zip(keys, map(float, model.uncertainty(variants)), strict=True))
    round_number = (previous["round"] + 1) if previous else 1
    output.mkdir(parents=True, exist_ok=True)
    if use_llm:
        from gfp_agent import select_candidates

        menu = build_menu(eligible, scores, uncertainty, seed=17, round_number=round_number)
        novel_observed = {key: make_variant(key) for key in visible if key not in dataset.variants}
        descriptors = replace(dataset, variants={**dataset.variants, **novel_observed}) if novel_observed else dataset
        request = make_agent_request(
            descriptors, {key: value.fitness for key, value in visible.items()}, menu,
            seed=17, round_number=round_number, budget=budget, knowledge=True,
            latest_feedback_ids=list(observations) if previous else [],
        )
        picks = select_candidates(request, backend="openai_compatible" if transport else backend,
                                  log_path=output / "agent.json", transport=transport)
    else:
        picks = sorted(eligible, key=lambda key: (-scores[key], key))[:budget]
    if len(picks) != budget or len(set(picks)) != budget or not set(picks) <= eligible.keys():
        raise ValueError("The selector returned an invalid next batch")
    _write_csv(output / "recommendations.csv", RECOMMENDATION_COLUMNS,
               [(key, eligible[key].sequence, scores[key], "", "prediction_only", backend if use_llm else "rf")
                for key in picks])
    state = {
        "format": 1,
        "assay": ASSAY,
        "reference_sha256": _sha(dataset.wild_type.encode()),
        "round": round_number,
        "prior_state_sha256": prior_sha,
        "observations": [{"mutant": key, "fitness": item.fitness} for key, item in visible.items()],
        "new_measurements": list(observations),
        "outstanding": sorted(outstanding | set(picks)),
        "proposed": picks,
        "score_units": "ProteinGym DMS_score on the Sarkisyan fluorescence scale",
    }
    _write_json(output / "state.json", state)
    return state


def _demo(dataset, output: Path, *, backend: str, use_llm: bool, transport=None) -> None:
    from gfp_data import make_split

    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(output)
    split = make_split(dataset, 7)
    _write_csv(output / "researcher_initial.csv", MEASURED_COLUMNS,
               [(key, dataset.scores[key]) for key in split["initial"]])
    _write_csv(output / "researcher_candidates.csv", ("mutant",), [(key,) for key in split["pool"]])
    first = _propose(
        dataset, output / "round-01", load_csv(output / "researcher_initial.csv", measured=True),
        prior=None, candidates=output / "researcher_candidates.csv", backend=backend, use_llm=use_llm, transport=transport,
    )
    _write_csv(output / "researcher_returned_measurements.csv", MEASURED_COLUMNS,
               [(key, dataset.scores[key]) for key in first["proposed"]])
    second = _propose(
        dataset, output / "round-02", load_csv(output / "researcher_returned_measurements.csv", measured=True),
        prior=output / "round-01/state.json", candidates=output / "researcher_candidates.csv",
        backend=backend, use_llm=use_llm, transport=transport,
    )
    _write_json(output / "status.json", {
        "status": "complete", "scope": "published measurements formatted as user data",
        "assay": ASSAY, "initial_observations": len(first["observations"]),
        "imported_after_selection": len(second["new_measurements"]),
        "total_observations": len(second["observations"]), "second_batch_prediction_only": len(second["proposed"]),
        "backend": "openai_compatible" if transport else backend if use_llm else "rf",
    })


def run_user_flow(dataset, output: Path, *, stage: str, measured_csv: Path | None,
                  prior_state: Path | None, batch_csv: Path | None, candidate_csv: Path | None,
                  backend: str, use_llm: bool, transport=None) -> None:
    if stage == "demo":
        _demo(dataset, output, backend=backend, use_llm=use_llm, transport=transport)
        return
    source = measured_csv if stage == "init" else batch_csv
    if source is None or (stage == "continue" and prior_state is None):
        raise ValueError("Supply --input for init or --state and --measurements for continue")
    _propose(dataset, output, load_csv(source, measured=True),
             prior=prior_state if stage == "continue" else None, candidates=candidate_csv,
             backend=backend, use_llm=use_llm)
