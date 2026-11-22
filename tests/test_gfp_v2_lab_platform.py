import csv
from types import SimpleNamespace

import pytest

from gfp_continue import load_csv
from gfp_data import GFP_WT, make_variant
from gfp_v2 import lab_platform as lab


def setup_batch(tmp_path):
    candidates = tmp_path / "candidates.csv"
    candidates.write_text("mutant\nK3R\nG4A\n")
    reference = tmp_path / "reference.fasta"
    reference.write_text(f">fixed\n{GFP_WT[:100]}\n{GFP_WT[100:]}\n")
    batch = tmp_path / "batch"
    ledger = lab.create(candidates, batch, namespace="study-round-one", reference=reference)
    return batch, ledger


def row(ledger, index=0, **overrides):
    return {
        **ledger["samples"][index], "batch_id": ledger["batch_id"], "fitness": "3.81",
        "status": "returned", "scale": lab.SCALE, "source": "external_assay", "source_ref": "fixture",
        **overrides,
    }


def test_lab_identity_is_stable_for_order_but_scoped_to_namespace(tmp_path):
    batch, ledger = setup_batch(tmp_path)
    reverse = tmp_path / "reverse.csv"
    reverse.write_text("mutant\nG4A\nK3R\n")
    same = lab.create(reverse, tmp_path / "same", namespace="study-round-one")
    other = lab.create(reverse, tmp_path / "other", namespace="study-round-two")
    assert ledger["batch_id"] == same["batch_id"] != other["batch_id"]
    assert {r["mutant"]: r["sample_id"] for r in ledger["samples"]} == {
        r["mutant"]: r["sample_id"] for r in same["samples"]
    }
    assert all(s["status"] == "pending" for s in lab.load(batch)["samples"])


@pytest.mark.parametrize("overrides", [
    {"batch_id": "wrong"}, {"sample_id": "wrong"}, {"sequence": "AAAA"},
    {"fitness": "nan"}, {"scale": "linear_fluorescence"}, {"source": "unregistered"},
    {"source_ref": ""}, {"status": "failed", "fitness": "3.2"},
])
def test_lab_rejects_bad_result_without_partial_mutation(tmp_path, overrides):
    batch, ledger = setup_batch(tmp_path)
    lab.submit(batch)
    ledger_before = (batch / lab.LEDGER).read_bytes()
    csv_path = tmp_path / "return.csv"
    lab.write_csv(csv_path, lab.RESULT_COLUMNS, [row(ledger, 0), row(ledger, 1, **overrides)])
    with pytest.raises(ValueError):
        lab.ingest(batch, csv_path)
    assert (batch / lab.LEDGER).read_bytes() == ledger_before


def test_lab_duplicate_and_resubmission_are_rejected(tmp_path):
    batch, ledger = setup_batch(tmp_path)
    lab.submit(batch)
    with pytest.raises(ValueError, match="pending"):
        lab.submit(batch, [ledger["samples"][0]["sample_id"]])
    duplicates = tmp_path / "duplicate.csv"
    lab.write_csv(duplicates, lab.RESULT_COLUMNS, [row(ledger), row(ledger)])
    with pytest.raises(ValueError, match="Duplicate"):
        lab.ingest(batch, duplicates)


def test_lab_simulator_never_fabricates_a_value_and_normalizes_real_archive_only(tmp_path):
    batch, ledger = setup_batch(tmp_path)
    lab.submit(batch)
    dataset = SimpleNamespace(wild_type_id="WT", variants={"WT": make_variant("WT")}, scores={"K3R": 3.8})
    results = tmp_path / "results.csv"
    lab.simulate(batch, results, dataset)
    updated = lab.ingest(batch, results)
    returned, failed = updated["samples"]
    assert returned["source"] == "public_archive" and returned["fitness"] == 3.8
    assert failed["source"] == "simulation" and failed["fitness"] is None
    normalized = tmp_path / "normalized.csv"
    assert lab.normalize(batch, normalized) == 1
    assert load_csv(normalized, measured=True)["K3R"].fitness == 3.8
    assert [event["operation"] for event in lab.load(batch)["events"]] == ["create", "submit", "ingest", "normalize"]
    assert len(lab.load(batch)["events"][-1]["output_sha256"]) == 64


def test_lab_normalization_rejects_synthetic_values_even_if_returned(tmp_path):
    batch, ledger = setup_batch(tmp_path)
    lab.submit(batch)
    fake = tmp_path / "synthetic.csv"
    lab.write_csv(fake, lab.RESULT_COLUMNS, [row(ledger, source="simulation")])
    lab.ingest(batch, fake)
    with pytest.raises(ValueError, match="No returned"):
        lab.normalize(batch, tmp_path / "normalized.csv")


def test_reference_validation_rejects_non_avGFP(tmp_path):
    bad = tmp_path / "wrong.fasta"
    bad.write_text(">different\nAAAA\n")
    with pytest.raises(ValueError, match="avGFP"):
        lab.check_reference(bad)


def test_no_api_forecast_exports_valid_sequences_and_prediction_only(tmp_path):
    observations = tmp_path / "observed.csv"
    observations.write_text("mutant,fitness\nWT,3.7\nK3R,3.0\nG4A,3.4\nE6D,3.5\n")
    candidates = tmp_path / "candidates.csv"
    candidates.write_text("mutant\nE5D\nL7A\nF8S\n")
    output = tmp_path / "predicted.csv"
    assert lab.forecast(observations, candidates, output, model="pair_ridge", top=2) == 2
    with output.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["rank"] for row in rows] == ["1", "2"]
    assert float(rows[0]["prediction"]) >= float(rows[1]["prediction"])
    for candidate in rows:
        assert candidate["status"] == "prediction_only" and "fitness" not in candidate
        assert candidate["sequence"] == make_variant(candidate["mutant"]).sequence
