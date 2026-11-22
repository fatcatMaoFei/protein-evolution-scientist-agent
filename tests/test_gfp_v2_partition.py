import copy
import json
from pathlib import Path

import pytest

from gfp_v2 import partition


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_partition_is_label_independent_and_excludes_v1():
    saved = json.loads((ROOT / "evidence/gfp-v2/partition.json").read_text())
    historical = partition.historical_ids()
    all_new = set().union(*(
        set(row.get("train_ample", row.get("initial", [])))
        | set(row.get("validation", row.get("test", row.get("pool", []))))
        for role in partition.ROLE_BLOCKS for row in saved[role]
    ))
    assert partition.verify(saved, all_new, historical) == saved["used_new_genotypes"] == 23528
    assert not ((all_new - {partition.WT_ID}) & historical)


def test_partition_creation_is_deterministic_and_rejects_cross_split_duplicates(monkeypatch):
    monkeypatch.setattr(partition, "LOW", 3)
    monkeypatch.setattr(partition, "AMPLE", 5)
    monkeypatch.setattr(partition, "HOLDOUT", 3)
    monkeypatch.setattr(partition, "POOL", 3)
    monkeypatch.setattr(partition, "ROLE_BLOCKS", {"prediction_development": range(1, 2),
                         "prediction_test": range(2, 3), "acquisition": range(3, 4), "agent": range(4, 5)})
    keys = [partition.WT_ID, *(f"id-{i}" for i in range(40))]
    retired = {"id-0", "id-1"}
    generated = partition.create(keys, retired)
    assert partition.create(reversed(keys), retired) == generated
    corrupted = copy.deepcopy(generated)
    corrupted["agent"][0]["pool"][0] = generated["prediction_test"][0]["test"][0]
    with pytest.raises(ValueError, match="overlap"):
        partition.verify(corrupted, keys, retired)
