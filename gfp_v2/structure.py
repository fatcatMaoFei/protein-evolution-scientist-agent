"""Verified Cα descriptors from the public 1GFL wild-type avGFP structure."""

from __future__ import annotations

import gzip
import hashlib
from itertools import combinations
from pathlib import Path

import numpy as np

from gfp_data import GFP_WT
from gfp_v2.benchmark import ROOT


ENTRY = "1GFL"
CHAIN = "A"
PDB_URL = "https://files.rcsb.org/download/1GFL.pdb"
PDB_SHA256 = "60eb370ad1a2c7c63382399d8655e0448fc29acc2a436e1a6112a5d599390d6f"
DEFAULT_PDB = ROOT / "evidence/gfp-v2/supplement/structure/1GFL.pdb.gz"
THREE_TO_ONE = dict(zip("ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split(),
                        "ACDEFGHIKLMNPQRSTVWY", strict=True))
EXPECTED_DIFFERENCES = {1: ("M", "A"), 64: ("L", "F"), 80: ("Q", "R")}
CONTACT_RADIUS = 8.0
CHROMOPHORE_REFERENCE_POSITIONS = (65, 66, 67)


def read_mapping(pdb: Path) -> tuple[list[dict], dict, dict[int, np.ndarray]]:
    """Check all sequence identities, reject mismatches except the documented mapped differences."""
    content = pdb.read_bytes()
    if content.startswith(b"\x1f\x8b"):
        try:
            content = gzip.decompress(content)
        except (OSError, EOFError) as exc:
            raise ValueError("Checksum cannot be verified for corrupted PDB snapshot") from exc
    sha = hashlib.sha256(content).hexdigest()
    if sha != PDB_SHA256:
        raise ValueError(f"Checksum does not match official {ENTRY} coordinate snapshot")
    lines = content.decode("ascii").splitlines()
    sequence = "".join(THREE_TO_ONE[aa] for line in lines if line.startswith("SEQRES") and line[11] == CHAIN
                       for aa in line[19:].split())
    if len(sequence) != len(GFP_WT):
        raise ValueError(f"{ENTRY} chain {CHAIN} SEQRES cannot be aligned one-to-one to the assay reference")
    differences = {i: (assay, pdb_aa) for i, (assay, pdb_aa) in enumerate(zip(GFP_WT, sequence, strict=True), 1)
                   if assay != pdb_aa}
    if differences != EXPECTED_DIFFERENCES:
        raise ValueError(f"Unexpected chain {CHAIN} sequence differences: {differences}")

    coordinates = {}
    for line in lines:
        if not (line.startswith("ATOM  ") and line[21] == CHAIN and line[12:16].strip() == "CA"
                and line[16] in (" ", "A")):
            continue
        position = int(line[22:26])
        if (line[26] != " " or position in coordinates or not 1 <= position <= len(sequence)
                or THREE_TO_ONE.get(line[17:20]) != sequence[position - 1]):
            raise ValueError(f"Unexpected or non-unique chain {CHAIN} residue {position}")
        coordinates[position] = np.array([float(line[i:i + 8]) for i in (30, 38, 46)])

    if not all(i in coordinates and i not in differences for i in CHROMOPHORE_REFERENCE_POSITIONS):
        raise ValueError("Verified Cα coordinates for residues 65–67 are required")
    centroid = np.mean([coordinates[i] for i in CHROMOPHORE_REFERENCE_POSITIONS], axis=0)
    positions = []
    for i, (assay, pdb_aa) in enumerate(zip(GFP_WT, sequence, strict=True), 1):
        usable = i in coordinates and i not in differences
        if usable:
            xyz = coordinates[i]
            contacts = sum(abs(i - j) > 2 and np.linalg.norm(xyz - other) <= CONTACT_RADIUS
                           for j, other in coordinates.items())
            distance = float(np.linalg.norm(xyz - centroid))
        else:
            contacts, distance = None, None
        positions.append({"position": i, "assay_residue": assay, "pdb_seqres_residue": pdb_aa,
                          "mapping_status": "mapped" if usable else (
                              "reference_difference" if i in differences else "no_ca_coordinates"),
                          "ca_neighbors_within_8A_excluding_two_sequence_neighbors": contacts,
                          "ca_to_ca65_67_centroid_A": distance})
    metadata = {"entry": ENTRY, "chain": CHAIN, "download_url": PDB_URL, "sha256": sha,
                "structure_doi": "10.2210/pdb1GFL/pdb", "publication_doi": "10.1038/nbt1096-1246",
                "resolution_angstrom": 1.90, "assay_reference_length": len(GFP_WT),
                "pdb_seqres_length": len(sequence), "pdb_ca_count": len(coordinates),
                "mapped_residue_count": sum(row["mapping_status"] == "mapped" for row in positions),
                "reference_differences": [{"position": i, "assay": pair[0], "pdb": pair[1]}
                                          for i, pair in differences.items()],
                "absent_ca_positions": sorted(set(range(1, len(GFP_WT) + 1)) - coordinates.keys()),
                "descriptor_definitions": {"contact": "Cα distance ≤8 Å and sequence separation >2",
                                           "centroid": "mean of the deposited Cα coordinates at positions 65–67",
                                           "structure_prediction": "static reference only; no mutant structure estimated"}}
    return positions, metadata, {i: xyz for i, xyz in coordinates.items() if i not in differences}


def variant_descriptors(variant, rows: dict[int, dict], coordinates) -> dict:
    edits = [int(edit[1:-1]) for edit in variant.edits]
    mapped = [i for i in edits if i in coordinates]
    contacts = [rows[i]["ca_neighbors_within_8A_excluding_two_sequence_neighbors"] for i in mapped]
    distances = [rows[i]["ca_to_ca65_67_centroid_A"] for i in mapped]
    pair_distances = [float(np.linalg.norm(coordinates[a] - coordinates[b])) for a, b in combinations(mapped, 2)]
    return {"unmapped_positions": len(edits) - len(mapped),
            "mean_ca_contacts": float(np.mean(contacts)) if contacts else np.nan,
            "maximum_ca_contacts": max(contacts) if contacts else np.nan,
            "min_ca_to_centroid_A": min(distances) if distances else np.nan,
            "mean_ca_to_centroid_A": float(np.mean(distances)) if distances else np.nan,
            "minimum_pair_ca_distance_A": min(pair_distances) if pair_distances else np.nan}
