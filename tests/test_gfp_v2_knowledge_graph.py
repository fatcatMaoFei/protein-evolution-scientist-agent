import json
from types import SimpleNamespace

import pytest

from gfp_data import GFP_WT
from gfp_v2 import knowledge_graph as kg


def edit(pos: int, preference="V") -> str:
    ref = GFP_WT[pos - 1]
    return f"{ref}{pos}{preference if ref != preference else 'A'}"


def observed(key, fitness):
    return {"id": key, "fitness": fitness, "source": "local assay fixture"}


def test_graph_updates_measured_support_and_keeps_multi_observational():
    first, second = edit(3), edit(4)
    graph = kg.ObservedGraph([observed("WT", 3.7), observed(first, 3.9), observed(f"{first}:{second}", 4.1)])
    before = graph.query([3, 4])
    assert before["positions"][0]["single"]["above_WT"] == 1
    assert before["positions"][0]["multi_association"]["above_WT"] == 1
    assert before["positions"][1]["single"]["n"] == 0
    assertions = [row for row in graph.edges if row["relation"] == "observed_above_measured_reference"]
    assert [row["source"] for row in assertions] == [f"mutation:{first}"]
    assert assertions[0]["reference"] == "observation:WT"
    assert {"mutation", "position", "variant", "fitness_observation", "amino_acid", "property"} <= {row["kind"] for row in graph.nodes.values()}
    rules = graph.query([3, 4], rule_only=True)
    assert all("single" not in site and "multi_association" not in site for site in rules["positions"])
    assert set(rules["positions"][0]["reference_properties"]) == {"hydropathy", "charge", "size", "polarity"}
    with pytest.raises(ValueError, match="unique"):
        graph.update([observed(second, 3.8), observed(first, 1.0)])
    assert second not in graph.observed
    graph.update([observed(second, 3.8)])
    assert graph.query([4])["positions"][0]["single"]["above_WT"] == 1


def test_native_query_case_and_replay_release_only_after_both_decisions(tmp_path, monkeypatch):
    class FakeTransport:
        model, endpoint = "test", "local:test"

        def encode(self, request):
            return request

        def generate(self, _, request):
            if request["tool_choice"] != "none":
                return {"finish_reason": "tool_calls", "tool_calls": [{"id": "test", "type": "function", "function": {
                    "name": "query_gfp_knowledge", "arguments": '{"positions":[3]}'}}], "text": ""}
            return {"finish_reason": "stop", "tool_calls": [], "text": json.dumps({
                "selected_aliases": [f"C{i:03d}" for i in range(1, 6)], "primary_alias": "C001",
                "evidence_ids": [edit(3, "R")], "basis": "observed_evidence"})}

    asked = []
    live = tmp_path / "out"
    replay = tmp_path / "replay"

    class Archive:
        def __getitem__(self, key):
            assert all((live / name / "commit.json").exists() for name in ("rules", "graph"))
            assert key in roster
            asked.append(key)
            return 3.8

    roster = [edit(pos) for pos in range(3, 8)]
    monkeypatch.setattr(kg, "load_dataset", lambda: SimpleNamespace(scores=Archive()))
    request = {"observed": [{"id": "WT", "fitness": 3.7, "round": 0}, {"id": edit(3, "R"), "fitness": 3.9, "round": 1}],
               "candidates": [{"id": key, "prediction": 3.8, "risk": .15} for key in roster]}
    input_path = tmp_path / "request.json"
    input_path.write_text(json.dumps(request))
    kg.case(input_path, live, FakeTransport())
    result = json.loads((live / "summary.json").read_text())
    assert result["graph"]["primary_evidence_links"][0]["retrieved_by_graph"]
    assert not result["rules"]["primary_evidence_links"][0]["retrieved_by_graph"]
    assert result["graph"]["primary_test"] == "SUPPORTED" and result["new_measured_count_graph"] == 5
    assert result["graph_minus_rules_best"] == 0.0 and len(asked) == 10
    provider = FakeTransport()
    monkeypatch.setattr(provider, "generate", lambda *_: pytest.fail("Replay attempted a provider call"))
    kg.case(input_path, replay, provider, replay=live)
    assert (replay / "summary.json").read_bytes() == (live / "summary.json").read_bytes()
