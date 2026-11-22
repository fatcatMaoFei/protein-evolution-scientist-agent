"""Freeze label-independent GFP V2 partitions against the recorded V1 commit."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
from pathlib import Path

from gfp_data import ASSAY_SHA256, WT_ID, load_dataset


ROOT = Path(__file__).resolve().parents[1]
BASE = "23adae08bb4929af2c82490ad010c5aa4e263200"
DRAW_SEED = 49730217
LOW, AMPLE, HOLDOUT, POOL = 96, 2048, 512, 96
ROLE_BLOCKS = {"prediction_development": range(2001, 2005), "prediction_test": range(2011, 2015),
               "acquisition": range(2101, 2109), "agent": range(2201, 2209)}
AA = b"[ACDEFGHIKLMNPQRSTVWY]"
EDIT = AA + b"[1-9][0-9]{0,2}" + AA
VARIANT = re.compile(rb"(?<![A-Za-z0-9:])" + EDIT + rb"(?:\:" + EDIT + rb")*(?![A-Za-z0-9:])")


def historical_ids(root: Path = ROOT) -> set[str]:
    """Retire any mutation token or genotype present in V1 GFP evidence bytes."""
    entries = subprocess.check_output(["git", "ls-tree", "-r", "-z", BASE, "--", "evidence/gfp"], cwd=root)
    blobs = [entry.split(b"\t", 1)[0].split()[-1] for entry in entries.split(b"\0") if entry]
    if not blobs:
        raise ValueError("V1 GFP evidence baseline is unavailable")
    payload = subprocess.check_output(["git", "cat-file", "--batch"], input=b"\n".join(blobs) + b"\n", cwd=root)
    candidates, offset = set(), 0
    for blob in blobs:
        stop = payload.index(b"\n", offset)
        token, kind, size = payload[offset:stop].split()
        if token != blob or kind != b"blob":
            raise ValueError("Invalid baseline evidence object")
        offset = stop + 1
        stop = offset + int(size)
        candidates.update(match.decode("ascii") for match in VARIANT.findall(payload[offset:stop]))
        offset = stop + 1
    if offset != len(payload):
        raise ValueError("Baseline evidence scan was incomplete")
    return candidates


def create(keys, historical) -> dict:
    keys = set(keys)
    excluded = keys & set(historical)
    available = sorted(keys - excluded - {WT_ID})
    random.Random(DRAW_SEED).shuffle(available)
    stream = iter(available)
    partitions = {}

    def take(n):
        chosen = [next(stream, None) for _ in range(n)]
        if None in chosen:
            raise ValueError("The V2 manifest requires more fresh archive genotypes")
        return chosen

    for role, ids in ROLE_BLOCKS.items():
        rows = []
        for block in ids:
            if role.startswith("prediction_"):
                train = [WT_ID, *take(AMPLE - 1)]
                holdout = "validation" if role.endswith("development") else "test"
                rows.append({"id": block, "train_low": train[:LOW], "train_ample": train, holdout: take(HOLDOUT)})
            else:
                rows.append({"id": block, "initial": [WT_ID, *take(LOW - 1)], "pool": take(POOL)})
        partitions[role] = rows
    result = {"v1_commit": BASE, "draw_seed": DRAW_SEED, "assay_sha256": ASSAY_SHA256,
              "excluded_v1_genotypes": len(excluded - {WT_ID}), **partitions}
    result["used_new_genotypes"] = verify(result, keys, excluded)
    return result


def verify(manifest, keys, historical) -> int:
    """Verify nested train budgets and disjoint non-WT blocks, including from V1."""
    used = set()
    keys = set(keys)
    excluded = set(historical) - {WT_ID}
    for role, ids in ROLE_BLOCKS.items():
        rows = manifest[role]
        if [row["id"] for row in rows] != list(ids):
            raise ValueError(f"Incorrect frozen block IDs for {role}")
        for row in rows:
            if role.startswith("prediction_"):
                train = row["train_ample"]
                if len(train) != AMPLE or row["train_low"] != train[:LOW]:
                    raise ValueError("Prediction budgets are not correctly nested")
                holdout = row["validation" if role.endswith("development") else "test"]
                expected_holdout = HOLDOUT
            else:
                train, holdout = row["initial"], row["pool"]
                if len(train) != LOW:
                    raise ValueError("Acquisition start has the wrong budget")
                expected_holdout = POOL
            if train[0] != WT_ID or len(holdout) != expected_holdout:
                raise ValueError("Missing WT training anchor or incorrect holdout size")
            other = train[1:] + holdout
            grouped = set(other)
            if len(grouped) != len(other) or WT_ID in grouped or grouped & (used | excluded) or not grouped <= keys:
                raise ValueError("Non-WT overlap, duplicate or unknown genotype in frozen partition")
            used.update(grouped)
    return len(used)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-cache", type=Path)
    args = parser.parse_args()
    data = load_dataset(args.data_cache)
    manifest = create(data.variants, historical_ids())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Frozen V2 manifest with {manifest['used_new_genotypes']} disjoint non-WT genotypes at {args.output}")


if __name__ == "__main__":
    main()
