import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gfp_data import AMINO_ACIDS, GFP_WT, WT_ID, make_variant
from gfp_v2 import benchmark, predictor


def _example():
    return [make_variant(WT_ID), *[make_variant(f"{source}{position}{next(aa for aa in AMINO_ACIDS if aa != source)}")
                                  for position, source in enumerate(GFP_WT[:15], 1)]]


@pytest.mark.parametrize("model", predictor.MODELS)
def test_fit_uses_only_supplied_scores_and_exposes_finite_risk(model):
    variants = _example()
    observed = np.linspace(1.3, 3.9, 12)
    first = predictor.fit(variants[:12], observed, 3, model)
    other = predictor.fit(variants[:12], observed + 0.01, 3, model)
    predicted, risk = first.estimate(variants[12:])
    assert len(predicted) == len(risk) == 4
    assert np.isfinite(predicted).all() and np.isfinite(risk).all() and np.all(risk >= 0)
    np.testing.assert_allclose(other.predict(variants[12:]), predicted + 0.01, atol=1e-5)
    assert first.estimate([])[0].size == 0


def test_pairwise_encoder_preserves_real_edit_pairs_and_distinguishes_chemistry():
    first = _example()[1].edits[0]
    partner = _example()[5].edits[0]
    variant = make_variant(f"{first}:{partner}")
    main = predictor._features(variant, False)
    paired = predictor._features(variant, True)
    assert main.items() <= paired.items()
    assert paired[f"pair_alleles:{first}|{partner}"] == 0.5
    assert any(name.startswith("pair_chemistry:") for name in paired)
    unknown = make_variant(f"{first[0]}{first[1:-1]}W")
    training = predictor.fit(_example()[:5], [3.7, 3.0, 2.0, 1.3, 3.3], 5, "pair_ridge")
    assert training.uncertainty([make_variant(first)])[0] < training.uncertainty([unknown])[0]


def test_wild_type_and_upper_decile_use_distinct_rules_and_tie_handling():
    results = benchmark.accuracy(list("abcdefghij"), np.arange(10), [9.0] * 10, 4.5)
    assert results["wt_positive_n"] == 5
    assert results["decile_positive_n"] == 1
    assert results["wt_precision_5"] == 0.5
    assert results["decile_precision_10"] == 0.1
    assert results["top_5_measured_best"] == 4  # Realized tie order uses the ID.
    assert results["spearman"] is None


def test_calibrated_risk_is_measured_against_independent_residuals():
    risk = np.arange(1, 11) / 10.0
    calibration = benchmark.calibrate(risk, risk / 2)
    summary = benchmark.risk_metrics(risk, risk + 0.5, calibration)
    assert summary["risk_error_spearman"] == pytest.approx(1)
    assert summary["high_risk_quartile_mae"] > summary["low_risk_quartile_mae"]
    assert summary["calibrated_90_coverage"] < 1


def test_model_choice_only_reads_development_labels_and_lock_precedes_test(tmp_path: Path):
    variants = _example()
    ids = [variant.id for variant in variants]
    values = dict(zip(ids, np.linspace(1.4, 3.9, len(ids)), strict=True))
    values[WT_ID] = 3.7
    development, test = ids[1:9], ids[9:]
    partition = {"prediction_development": [{"id": 1, "train_low": [WT_ID, *development[:3]],
                  "train_ample": [WT_ID, *development[:5]], "validation": development[5:]}],
                 "prediction_test": [{"id": 2, "train_low": [WT_ID, *test[:2]],
                  "train_ample": [WT_ID, *test[:4]], "test": test[4:]}]}
    partition_path = tmp_path / "splits.json"
    partition_path.write_text(json.dumps(partition))
    output = tmp_path / "evidence"
    genotype_features = {variant.id: variant for variant in variants}

    class Labels:
        def __init__(self, allowed):
            self.allowed = set(allowed) | {WT_ID}

        def __getitem__(self, key):
            assert key in self.allowed, key
            return values[key]

    data = SimpleNamespace(variants=genotype_features, scores=Labels(development))
    selection = benchmark.select(data, partition_path, output)
    assert set(selection["selected"]) == {"low", "ample"}
    assert (output / "selection.json").exists() and not (output / "test_predictions.csv").exists()
    data.scores = Labels(test)
    benchmark.evaluate(data, partition_path, output)
    assert (output / "test_metrics.csv").exists()
    with pytest.raises(FileExistsError):
        benchmark.evaluate(data, partition_path, output)
    (output / "validation_summary.csv").write_text("altered")
    with pytest.raises(ValueError, match="Frozen validation"):
        benchmark.evaluate(data, partition_path, output)
