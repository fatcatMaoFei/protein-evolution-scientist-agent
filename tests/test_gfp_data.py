import json

import numpy as np
import pytest

from gfp_data import (
    DEFAULT_CACHE_DIR,
    GFP_WT,
    GFPDataset,
    fit_predictor,
    load_dataset,
    make_split,
    make_variant,
    metrics,
    parse_mutations,
)


def test_constructs_canonical_gfp_substitution_and_rejects_bad_inputs():
    variant = make_variant("V55A:K3R")
    assert variant.id == "K3R:V55A"
    assert variant.edits == ("K3R", "V55A")
    assert len(variant.sequence) == len(GFP_WT) == 238
    assert variant.sequence[2] == "R"
    assert variant.sequence[54] == "A"
    assert parse_mutations("WT") == ()
    assert make_variant("WT").sequence == GFP_WT
    for edits in ("A3R", "K3K", "K3X", "K0R", "K239R", "K3R:K3Q", "K3R:"):
        with pytest.raises(ValueError):
            make_variant(edits)
    with pytest.raises(ValueError, match="pinned"):
        make_variant("WT", "AAAA")
    with pytest.raises(ValueError, match="edit budget"):
        make_variant("WT", max_mutations=5)


def test_split_roles_are_disjoint_and_identical_when_truth_changes():
    records = [make_variant("WT")] + [
        make_variant(f"{residue}{index}A")
        for index, residue in enumerate(GFP_WT[:50], 1)
        if residue != "A"
    ]
    variants = {item.id: item for item in records}
    scores = {item.id: float(i) for i, item in enumerate(records)}
    one = GFPDataset(GFP_WT, variants, scores, {}, {})
    two = GFPDataset(GFP_WT, variants, {key: -value for key, value in scores.items()}, {}, {})
    sizes = dict(initial=8, validation=6, pool=16, test=8)
    split = make_split(one, 17, **sizes)
    assert split == make_split(two, 17, **sizes)
    assert split != make_split(two, 29, **sizes)
    assert {role: len(ids) for role, ids in split.items()} == sizes
    assert split["initial"][0] == "WT"
    flat = [identity for ids in split.values() for identity in ids]
    assert len(flat) == len(set(flat))
    assert "fitness" not in variants["WT"].__slots__


def test_predictor_interface_and_tie_aware_metrics():
    samples = [make_variant(edit) for edit in ("WT", "K3R", "K3Q", "V55A", "K3R:V55A", "V55T")]
    measured = [3.72, 3.77, 1.91, 3.18, 2.62, 3.49]
    for model in ("ridge", "rf"):
        regressor = fit_predictor(samples, measured, 17, model)
        assert regressor.predict(samples).shape == (6,)
        assert np.isfinite(regressor.predict([make_variant("V55A:K3H")])).all()
        assert (regressor.uncertainty(samples) >= 0).all()
    with pytest.raises(ValueError):
        fit_predictor(samples, [1, 2], 17)
    tied = metrics([1.0, 3.0, 4.0, 5.0], [3.0, 3.0, 3.0, 3.0], top_k=2, threshold=2.5)
    assert tied["spearman"] is None and tied["pearson"] is None
    assert tied["top_k_precision"] == 0.75
    assert tied["top_k_recall"] == 0.5
    assert json.loads(json.dumps(tied, allow_nan=False)) == tied


@pytest.mark.skipif(
    not (DEFAULT_CACHE_DIR / "GFP_AEQVI_Sarkisyan_2016.csv").exists()
    or not (DEFAULT_CACHE_DIR / "sarkisyan-brightness.tsv").exists(),
    reason="Official GFP archive not in the local cache",
)
def test_verified_real_gfp_archive():
    data = load_dataset()
    assert data.provenance["assay_rows"] == 51_714
    assert len(data.variants) == len(data.scores) == 35_585
    assert data.scores["WT"] == pytest.approx(3.7192121319)
    assert data.variants["WT"].sequence == GFP_WT
    assert data.binary["WT"] == 1
    split = make_split(data, 17)
    assert {key: len(value) for key, value in split.items()} == {
        "initial": 96, "validation": 120, "pool": 384, "test": 120
    }
    assert "WT" not in split["test"]
