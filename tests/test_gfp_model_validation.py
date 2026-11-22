import pytest

from gfp_data import GFP_WT, GFPDataset, make_variant
from gfp_model_validation import risk_at_coverage, spread_diagnostics, validate_models


def test_rf_risk_at_coverage_has_tie_aware_error_and_true_full_coverage():
    assert risk_at_coverage([0.1, 0.1, 0.5, 0.6], [1, 3, 9, 7], 0.25) == {
        "retained_fraction": 0.25, "retained": 1, "mae": 2.0,
    }
    assert risk_at_coverage([0.2, 0.1], [2, 8], 1)["mae"] == 5
    with pytest.raises(ValueError, match="matching"):
        risk_at_coverage([-0.3], [1], 0.5)


def test_zero_tree_spread_never_claims_uncertainty_ranking():
    summary = spread_diagnostics([0.0, 0.0], [2, 4], [2.5, 2.5])
    assert summary["spread_abs_error_spearman"] is None
    assert summary["fraction_abs_error_within_1_96_spreads"] == 0


def test_validation_only_uses_initial_and_validation_labels():
    split = {
        "initial": ["WT", "K3R", "K3Q", "V55A"],
        "validation": ["V55K", "K3R:V55A", "K3Q:V55T"],
        "pool": ["K3Q:V55A"],
        "test": ["V55Q"],
    }
    permitted = [*split["initial"], *split["validation"]]
    variants = {variant: make_variant(variant) for variant in permitted}
    scores = dict(zip(permitted, [3.72, 3.91, 3.52, 1.33, 1.50, 3.74, 1.31], strict=True))
    dataset = GFPDataset(GFP_WT, variants, scores, {}, {})
    results, predictions, diagnostics = validate_models(dataset, {17: split})
    assert len(results) == 8
    assert set(predictions.variant_id) == set(split["validation"])
    assert predictions.loc[predictions.model == "ridge", "tree_spread"].isna().all()
    assert diagnostics["unique_validation_variants"] == 3
    overlapping_split = {**split, "test": ["V55K"]}
    with pytest.raises(ValueError, match="overlapping"):
        validate_models(dataset, {17: overlapping_split})
