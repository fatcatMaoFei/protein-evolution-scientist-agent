from types import SimpleNamespace

import pytest

import gfp_acquisition_ablation as study


def test_partition_is_label_free_deterministic_and_disjoint():
    keys = [f"id-{i}" for i in range(1000)] + ["WT"]
    excluded = {"id-0", "id-1"}
    blocks = study.partition(keys, excluded, "WT", blocks=(101, 102))
    assert study.partition(reversed(keys), excluded, "WT", blocks=(101, 102)) == blocks
    seen = set(excluded)
    for split in blocks.values():
        assert split["initial"][0] == "WT"
        initial, pool = set(split["initial"][1:]), set(split["pool"])
        assert len(initial) == 95 and len(pool) == 384
        assert not (initial & pool or (initial | pool) & seen)
        seen.update(initial | pool)


def test_model_only_refits_on_its_own_reveals(monkeypatch):
    scores = {"WT": -2, "initial": -1} | {f"v{i:02d}": float(i) for i in range(20)}
    dataset = SimpleNamespace(wild_type_id="WT", scores=scores, variants={key: key for key in scores})
    train = []

    def fit(variants, observed, block, *, model):
        assert model == "ridge"
        assert list(observed) == [scores[key] for key in variants]
        train.append(list(variants))
        return SimpleNamespace(predict=lambda pool: [float(key[1:]) for key in pool])

    monkeypatch.setattr(study, "fit_predictor", fit)
    split = {"initial": ["WT", "initial"], "pool": [f"v{i:02d}" for i in range(20)]}
    summary, rounds, selections, rankings = study.run_policy(dataset, 101, split, "ridge")
    assert train[0] == split["initial"]
    assert [len(ids) for ids in train] == [2, 7, 12]
    assert train[2] == split["initial"] + [row["mutant"] for row in selections[:10]]
    assert [row["batch_best"] for row in rounds] == [19, 14, 9]
    assert len(selections) == 15 and len({row["mutant"] for row in selections}) == 15
    assert all("fitness" not in row for row in rankings)
    assert summary["new_best"] == 19 and summary["new_mean"] == 12


def test_paired_contrasts_report_all_blocks_and_ties():
    methods = [{"policy": policy, "block": block, "new_best": val, "new_mean": val,
                "above_wt": 0, "incumbent_gain": 0}
               for policy, values in {"ridge": [3, 5], "rf": [2, 5], "random": [1, 6]}.items()
               for block, val in enumerate(values, 101)]
    paired, summary = study.paired_summary(methods)
    assert len(paired) == 6
    ridge_rf = summary[0]
    assert ridge_rf["new_best"] == pytest.approx(0.5)
    assert ridge_rf["new_best_wins"] == ridge_rf["new_best_ties"] == 1
    assert ridge_rf["new_best_losses"] == 0
