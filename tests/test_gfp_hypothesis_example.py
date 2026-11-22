from types import SimpleNamespace

import gfp_hypothesis_example as example


def test_example_reveals_only_primary_after_recorded_decision(tmp_path, monkeypatch):
    split = {"initial": ["WT"], "pool": [f"candidate-{i}" for i in range(6)]}
    accessed = []
    rounds = []

    class Scores:
        def __getitem__(self, key):
            accessed.append(key)
            if key == "WT":
                return 3.7
            assert rounds == [1] and key == "candidate-1"
            return 1.8

    def propose(dataset, output, measured, **options):
        assert options["primary_from_menu"] and options["allow_novel"]
        if not rounds:
            assert accessed == ["WT"]
            rounds.append(1)
            return {"hypothesis_evaluations": [{"status": "pending"}],
                    "hypotheses": [{"candidate_ids": ["candidate-1"]}]}
        assert accessed == ["WT", "candidate-1"] and measured["candidate-1"].fitness == 1.8
        rounds.append(2)
        return {"hypothesis_evaluations": [{"status": "refuted"}, {"status": "pending"}]}

    monkeypatch.setattr(example, "make_split", lambda dataset, seed: split)
    def load(path, **kwargs):
        key, value = path.read_text().splitlines()[1].split(",")
        return {key: SimpleNamespace(fitness=float(value))}

    monkeypatch.setattr(example, "load_csv", load)
    monkeypatch.setattr(example, "_propose", propose)
    monkeypatch.setattr(example, "run_user_audit", lambda state, output: {"status": "complete"})
    example.run(SimpleNamespace(scores=Scores()), tmp_path / "example", object())
    assert rounds == [1, 2] and accessed == ["WT", "candidate-1"]
