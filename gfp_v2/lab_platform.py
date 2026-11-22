"""Local GFP batch exchange for simulation and user-supplied assay files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path

from gfp_continue import ASSAY, load_csv
from gfp_data import GFP_WT, make_variant

SCALE = "log10_fluorescence"
REFERENCE_HASH = hashlib.sha256(GFP_WT.encode()).hexdigest()
SOURCES = {"public_archive", "external_assay", "simulation"}
MEASURED_SOURCES = SOURCES - {"simulation"}
RESULT_COLUMNS = ("batch_id", "sample_id", "mutant", "sequence", "status", "fitness", "scale", "source",
                  "source_ref")
LEDGER = "ledger.json"


def check_reference(path: Path | None) -> None:
    if path is None:
        return
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if (len(lines) < 2 or not lines[0].startswith(">") or any(line.startswith(">") for line in lines[1:])
            or "".join(lines[1:]).upper() != GFP_WT):
        raise ValueError("Provide one FASTA record matching the supported, pinned avGFP reference exactly")


def read_csv(path: Path, required: set[str]) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if not required <= set(fields) or len(fields) != len(set(fields)):
            raise ValueError(f"Missing or repeated CSV columns in {path}; required {sorted(required)}")
        rows = list(reader)
    if not rows or any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Empty or malformed CSV: {path}")
    return rows


def write_csv(path: Path, columns, rows) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def digest(items) -> str:
    encoded = json.dumps(items, sort_keys=True, allow_nan=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def save(directory: Path, ledger: dict) -> None:
    """Commit state and its ordered audit trail together."""
    with tempfile.NamedTemporaryFile(dir=directory, mode="w", encoding="utf-8", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(ledger, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.replace(temporary, directory / LEDGER)
        finally:
            temporary.unlink(missing_ok=True)


def sample_id(batch: str, mutant: str) -> str:
    return "GFP-S-" + digest([batch, mutant])[:24]


def batch_id(namespace: str, mutants: list[str]) -> str:
    return "GFP-B-" + digest([ASSAY, SCALE, REFERENCE_HASH, namespace, sorted(mutants)])[:24]


def canonical(mutant: str, sequence: str | None):
    variant = make_variant(mutant)
    if variant.id != mutant or (sequence is not None and sequence != variant.sequence):
        raise ValueError(f"Canonical one-based mutation annotation and pinned sequence do not agree: {mutant!r}")
    return variant


def load(directory: Path) -> dict:
    ledger = json.loads((directory / LEDGER).read_text())
    if (ledger["assay"] != ASSAY or ledger["scale"] != SCALE or ledger["reference_sha256"] != REFERENCE_HASH
            or ledger["batch_id"] != batch_id(ledger["namespace"], [row["mutant"] for row in ledger["samples"]])):
        raise ValueError("Batch metadata does not match the pinned GFP assay or its stable ID")
    for sample in ledger["samples"]:
        canonical(sample["mutant"], sample["sequence"])
        if sample["sample_id"] != sample_id(ledger["batch_id"], sample["mutant"]):
            raise ValueError("Sample identity changed after batch creation")
    return ledger


def event(ledger: dict, operation: str, details: dict) -> None:
    ledger["events"].append({"number": len(ledger["events"]) + 1, "operation": operation, **details})


def create(candidates: Path, directory: Path, *, namespace: str, reference: Path | None = None) -> dict:
    check_reference(reference)
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,80}", namespace):
        raise ValueError("Batch namespace must be a stable 1–80 character project or run token")
    rows = read_csv(candidates, {"mutant"})
    variants = [canonical(row["mutant"], row.get("sequence")) for row in rows]
    mutants = [variant.id for variant in variants]
    if len(mutants) != len(set(mutants)):
        raise ValueError("Duplicate mutants cannot share a batch")
    batch = batch_id(namespace, mutants)
    ledger = {
        "format": "gfp-v2-local-platform-1", "mode": "local_file_exchange_no_hardware",
        "assay": ASSAY, "scale": SCALE, "reference_sha256": REFERENCE_HASH,
        "namespace": namespace, "batch_id": batch, "samples": [], "events": [],
    }
    for variant in variants:
        ledger["samples"].append({"sample_id": sample_id(batch, variant.id), "mutant": variant.id,
                                  "sequence": variant.sequence, "status": "pending"})
    event(ledger, "create", {"input_sha256": hashlib.sha256(candidates.read_bytes()).hexdigest(),
                            "samples": [row["sample_id"] for row in ledger["samples"]]})
    directory.mkdir(parents=True, exist_ok=False)
    save(directory, ledger)
    return ledger


def submit(directory: Path, ids: list[str] | None = None) -> dict:
    ledger = load(directory)
    samples = {row["sample_id"]: row for row in ledger["samples"]}
    ids = ids or [key for key, row in samples.items() if row["status"] == "pending"]
    if not ids or len(ids) != len(set(ids)) or any(key not in samples or samples[key]["status"] != "pending" for key in ids):
        raise ValueError("Submit one or more distinct pending samples from this batch")
    for key in ids:
        samples[key]["status"] = "submitted"
    event(ledger, "submit", {"samples": ids})
    save(directory, ledger)
    return ledger


def export(directory: Path, output: Path) -> None:
    ledger = load(directory)
    write_csv(output, ("batch_id", "sample_id", "mutant", "sequence", "status", "scale"),
              [{**row, "batch_id": ledger["batch_id"], "scale": SCALE} for row in ledger["samples"]])


def ingest(directory: Path, results: Path) -> dict:
    ledger = load(directory)
    incoming = read_csv(results, set(RESULT_COLUMNS))
    if len({row["sample_id"] for row in incoming}) != len(incoming):
        raise ValueError("Duplicate sample results")
    samples = {row["sample_id"]: row for row in ledger["samples"]}
    for row in incoming:
        sample = samples.get(row["sample_id"])
        if row["batch_id"] != ledger["batch_id"] or not sample or sample["status"] != "submitted":
            raise ValueError("Results must refer to submitted samples in this batch")
        canonical(row["mutant"], row["sequence"])
        if (row["mutant"] != sample["mutant"] or row["sequence"] != sample["sequence"]
                or row["scale"] != SCALE or row["source"] not in SOURCES or not row["source_ref"].strip()):
            raise ValueError("Result identity, log10 fluorescence scale or provenance does not match this batch")
        if row["status"] == "returned":
            fitness = float(row["fitness"])
            if not math.isfinite(fitness):
                raise ValueError("Returned fluorescence must be finite")
        elif row["status"] == "failed" and not row["fitness"].strip():
            fitness = None
        else:
            raise ValueError("Report status returned with a measurement, or failed with empty fitness")
        sample.update(status=row["status"], fitness=fitness, source=row["source"], source_ref=row["source_ref"])
    event(ledger, "ingest", {"input_sha256": hashlib.sha256(results.read_bytes()).hexdigest(),
                             "result_rows": [{key: row[key] for key in RESULT_COLUMNS if key != "sequence"}
                                             for row in incoming]})
    save(directory, ledger)
    return ledger


def normalize(directory: Path, output: Path) -> int:
    ledger = load(directory)
    rows = [sample for sample in ledger["samples"]
            if sample["status"] == "returned" and sample["source"] in MEASURED_SOURCES]
    if not rows:
        raise ValueError("No returned external-assay or archive measurement can enter a V2 continuation")
    write_csv(output, ("mutant", "fitness", "sequence"), rows)
    event(ledger, "normalize", {"samples": [row["sample_id"] for row in rows],
                                "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest()})
    save(directory, ledger)
    return len(rows)


def simulate(directory: Path, output: Path, dataset) -> None:
    """Build a simulated platform return from genuine records after submission."""
    ledger = load(directory)
    if dataset.wild_type_id != "WT" or dataset.variants["WT"].sequence != GFP_WT:
        raise ValueError("Archive does not match the pinned GFP reference")
    rows = []
    for sample in ledger["samples"]:
        if sample["status"] != "submitted":
            continue
        key = sample["mutant"]
        score = dataset.scores[key] if key in dataset.scores else None
        source_ref = ("ProteinGym 1.3 GFP_AEQVI_Sarkisyan_2016; DOI 10.5281/zenodo.15293562"
                      if score is not None else "Simulator: submitted genotype has no matching archive record")
        rows.append({**sample, "batch_id": ledger["batch_id"], "status": "returned" if score is not None else "failed",
                     "scale": SCALE, "source": "public_archive" if score is not None else "simulation",
                     "source_ref": source_ref, "fitness": score})
    if not rows:
        raise ValueError("No submitted candidates can be serviced by the simulator")
    write_csv(output, RESULT_COLUMNS, rows)


def forecast(measurements: Path, candidates: Path, output: Path, *, model: str,
             top: int = 5, reference: Path | None = None, state_path: Path | None = None) -> int:
    """Score user-provided candidate genotypes using imported observations without a model API."""
    from gfp_v2.open_world import _prior_state
    from gfp_v2.predictor import fit

    check_reference(reference)
    if top < 1:
        raise ValueError("Top K must be positive")
    state, _ = _prior_state(state_path)
    observed = {row["id"]: row["fitness"] for row in state["observed"]} if state else {}
    imported = load_csv(measurements, measured=True)
    if observed.keys() & imported.keys() or (state and state["predictor"] != model):
        raise ValueError("New measurements cannot repeat prior observations or change a locked predictor")
    observed.update({key: entry.fitness for key, entry in imported.items()})
    if "WT" not in observed or len(observed) < 4:
        raise ValueError("Provide WT and three or more other distinct measured GFP records")
    pending = set(state["pending_ids"]) - set(imported) if state else set()
    eligible = [canonical(row["mutant"], row.get("sequence")) for row in read_csv(candidates, {"mutant"})]
    if len(eligible) != len({row.id for row in eligible}):
        raise ValueError("Candidate preview does not accept repeated variants")
    candidates_to_score = [candidate for candidate in eligible if candidate.id not in observed and candidate.id not in pending]
    if not candidates_to_score:
        raise ValueError("No eligible unseen and nonpending candidate remains")
    predictor = fit([make_variant(key) for key in observed], list(observed.values()), state["seed"] if state else 17, model)
    means, risks = predictor.estimate(candidates_to_score)
    records = [{"mutant": candidate.id, "sequence": candidate.sequence, "prediction": score,
                "risk": risk, "status": "prediction_only"}
               for candidate, score, risk in zip(candidates_to_score, means, risks, strict=True)]
    ranked = sorted(records, key=lambda row: (-row["prediction"], row["mutant"]))[:top]
    write_csv(output, ("rank", "mutant", "sequence", "prediction", "risk", "status"),
              [{"rank": index, **row} for index, row in enumerate(ranked, 1)])
    return len(ranked)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("create", "submit", "export", "ingest", "normalize", "simulate"):
        command = sub.add_parser(action)
        command.add_argument("--batch", type=Path, required=True)
        if action == "create":
            command.add_argument("--candidates", type=Path, required=True)
            command.add_argument("--namespace", required=True)
            command.add_argument("--reference", type=Path)
        elif action == "submit":
            command.add_argument("--sample", action="append")
        elif action == "ingest":
            command.add_argument("--results", type=Path, required=True)
        else:
            command.add_argument("--output", type=Path, required=True)
            if action == "simulate":
                command.add_argument("--data-cache", type=Path)
    preview = sub.add_parser("forecast", help="Rank personal GFP candidates without an API")
    preview.add_argument("--measurements", type=Path, required=True)
    preview.add_argument("--candidates", type=Path, required=True)
    preview.add_argument("--output", type=Path, required=True)
    preview.add_argument("--model", required=True)
    preview.add_argument("--top", type=int, default=5)
    preview.add_argument("--reference", type=Path)
    preview.add_argument("--state", type=Path)
    args = parser.parse_args(argv)
    if args.action == "forecast":
        print(f"Generated {forecast(args.measurements, args.candidates, args.output, model=args.model, top=args.top, reference=args.reference, state_path=args.state)} prediction-only recommendations")
    elif args.action == "create":
        print(create(args.candidates, args.batch, namespace=args.namespace, reference=args.reference)["batch_id"])
    elif args.action == "submit":
        submit(args.batch, args.sample)
    elif args.action == "ingest":
        ingest(args.batch, args.results)
    elif args.action == "export":
        export(args.batch, args.output)
    elif args.action == "normalize":
        print(f"Normalized {normalize(args.batch, args.output)} measured GFP observations")
    else:
        from gfp_data import load_dataset

        simulate(args.batch, args.output, load_dataset(args.data_cache))


if __name__ == "__main__":
    main()
