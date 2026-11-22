import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gfp_data import GFP_WT, make_variant
from gfp_v2.landscape import nonadditivity, selection_preference, structural_extension
from gfp_v2.structure import DEFAULT_PDB, read_mapping, variant_descriptors


def _dataset():
    scores = {"WT": 3.7, "M1A": 3.5, "S2A": 3.6, "K3T": 3.9,
              "M1A:S2A": 1.3, "M1A:K3T": 3.8, "S2A:K3T": 3.4, "K3T:G4A": 3.5}
    return SimpleNamespace(scores=scores, variants={key: make_variant(key) for key in scores})


def test_coordinate_snapshot_alignment_and_unavailable_positions(tmp_path):
    rows, source, coordinates = read_mapping(DEFAULT_PDB)
    assert len(rows) == len(GFP_WT) == 238
    assert source["mapped_residue_count"] == 227
    assert [row["position"] for row in source["reference_differences"]] == [1, 64, 80]
    assert source["absent_ca_positions"] == list(range(231, 239))
    mapped = {row["position"]: row for row in rows}
    assert mapped[64]["ca_to_ca65_67_centroid_A"] is None
    variant = make_variant("M1A:K3T:E235G")
    information = variant_descriptors(variant, mapped, coordinates)
    assert information["unmapped_positions"] == 2
    assert np.isnan(information["minimum_pair_ca_distance_A"])
    corrupt = tmp_path / "corrupt.pdb"
    corrupt.write_bytes(DEFAULT_PDB.read_bytes()[:-1])
    with pytest.raises(ValueError, match="Checksum"):
        read_mapping(corrupt)


def test_matched_single_mutants_are_used_and_incomplete_pairs_remain_unmatched():
    effects, summary = nonadditivity(_dataset())
    dim = effects.set_index("id").loc["M1A:S2A"]
    assert dim.additive_log_brightness == pytest.approx(3.4)
    assert dim.measured_minus_additive == pytest.approx(-2.1)
    assert bool(dim.additive_bright_measured_dim)
    assert "K3T:G4A" not in effects.id.tolist()
    assert summary.iloc[0].fully_matched_genotypes == 3
    assert summary.iloc[0].total_genotypes == 4


def test_site_prevalence_conditions_on_selected_mutation_count(tmp_path):
    dataset = _dataset()
    pd.DataFrame([{"block": 2201, "arm": "D_agent", "round": 1, "id": "M1A:K3T", "measured_fitness": 3.8},
                  {"block": 2201, "arm": "D_agent", "round": 1, "id": "K3T", "measured_fitness": 3.9}]).to_csv(
        tmp_path / "selections.csv", index=False)
    folder = tmp_path / "arms/2201/D_agent"
    folder.mkdir(parents=True)
    (folder / "round-01-request.json").write_text(json.dumps({"budget": 2, "candidates": [{"id": key} for key in (
        "M1A", "K3T", "M1A:S2A", "M1A:K3T", "S2A:K3T"
    )]}))
    degrees, sites = selection_preference(dataset, tmp_path, "agent")
    d2 = degrees.set_index("mutation_count").loc[2]
    assert d2.expected_uniform_selected == pytest.approx(1.2)
    assert d2.expected_above_wt_with_same_degree_counts == pytest.approx(1 / 3)
    k3 = sites.set_index("position").loc[3]
    assert k3.selected == 2 and k3.above_wt_hits == 2
    assert k3.expected_uniform_given_selected_edit_counts == pytest.approx(1 / 2 + 2 / 3)
    assert k3.expected_above_wt_hits_matching_edit_counts == pytest.approx(1 / 2 + 1 / 3)


def test_residual_extension_rejects_cross_role_genotypes():
    shared = pd.DataFrame({"variant_id": ["M1A"]})
    with pytest.raises(ValueError, match="disjoint"):
        structural_extension(shared, shared, {}, None, {}, {})
