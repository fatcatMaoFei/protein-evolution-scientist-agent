import csv
import json

import pytest

from gfp_data import GFP_WT
from gfp_v2.open_world import run_open
from gfp_v2.policy import TOOL, evaluate_hypotheses, select
from llm_agent import AgentError


def _mutation(position):
    residue = GFP_WT[position - 1]
    return f"{residue}{position}{'V' if residue != 'V' else 'A'}"


def _case():
    known = [{"id": "WT", "fitness": 3.72, "round": 0}, {"id": _mutation(4), "fitness": 4.0, "round": 0},
             {"id": _mutation(5), "fitness": 2.5, "round": 0}, {"id": _mutation(6), "fitness": 3.1, "round": 0}]
    rows = [{"id": _mutation(pos), "prediction": 3.6 + i / 100, "risk": .4}
            for i, pos in enumerate(range(10, 18))]
    return {"budget": 2, "round": 1, "wild_type_fitness": 3.72, "current_best": 4.0,
            "observed": known, "candidates": rows, "latest_feedback": [], "prior_decisions": []}


class FakeTransport:
    model = "unit-test-model"
    endpoint = "test-endpoint"
    config = {"backend": "fake"}

    def __init__(self, proposals, final):
        self.proposals = proposals
        self.final = final
        self.generated = []

    def encode(self, request):
        return request

    def generate(self, endpoint, request):
        self.generated.append(request)
        response = {"model": self.model, "usage": {"input_tokens": 5, "output_tokens": 3}}
        if request["tool_choice"] != "none":
            return {**response, "finish_reason": "tool_calls", "text": "",
                    "tool_calls": [{"id": "call1", "type": "function", "function": {
                        "name": TOOL, "arguments": json.dumps({"designs": self.proposals}),
                    }}]}
        return {**response, "finish_reason": "stop", "text": json.dumps(self.final), "tool_calls": []}


def _legal(identities, evidence=("WT",), adaptation="no_new_feedback"):
    return {"selected_ids": identities[:2], "primary_id": identities[0], "comparator": "WT",
            "evidence_ids": list(evidence), "strategy": "mixed", "adaptation": adaptation}


@pytest.mark.parametrize("knowledge", [False, True])
def test_closed_exposes_complete_pool_and_verifies_offline_archive(tmp_path, knowledge):
    request = _case()
    ids = [row["id"] for row in request["candidates"]]
    adapter = FakeTransport(ids[:4], _legal(ids))
    output = tmp_path / "live.json"
    decision = select(request=request, transport=adapter, knowledge=knowledge, log_path=output)
    assert decision["selected_ids"] == ids[:2]
    assert len(decision["designs"][0]["sequence"]) == len(GFP_WT)
    user = adapter.generated[0]["messages"][1]["content"]
    payload = json.loads(user.partition("\n")[2])
    assert {row[0] for row in payload["all_current_candidates"]} == set(ids)
    assert ("observed_knowledge" in payload) == knowledge
    assert ("chemistry_class_changes" in decision["designs"][0]) == knowledge
    assert decision["mechanism_validation"] == "NOT EVALUABLE"
    assert decision["hypotheses"][0]["status"] == "PENDING"
    adapter.generated.clear()
    replayed = select(request=request, transport=adapter, knowledge=knowledge, log_path=tmp_path / "replay.json", replay_path=output)
    assert replayed == decision and not adapter.generated
    record = json.loads(output.read_text())
    record["calls"][0]["response"]["text"] = "manual change"
    output.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="response hash mismatch"):
        select(request=request, transport=adapter, knowledge=knowledge, log_path=tmp_path / "bad.json", replay_path=output)


def test_reference_rejects_measured_claims_off_menu_and_unsupported_fields(tmp_path):
    request = _case()
    ids = [row["id"] for row in request["candidates"]]
    fake = FakeTransport(ids[:4], _legal(ids))
    invalid = {**request, "candidates": [{**request["candidates"][0], "fitness": 5.0}, *request["candidates"][1:]]}
    with pytest.raises(AgentError, match="only IDs"):
        select(request=invalid, transport=fake, knowledge=False, log_path=tmp_path / "leak.json")
    fake.proposals[-1] = _mutation(20)
    with pytest.raises(AgentError, match="displayed pool"):
        select(request=request, transport=fake, knowledge=False, log_path=tmp_path / "off.json")
    fake.proposals = ids[:4]
    fake.final = _legal(ids, evidence=[ids[0]])
    with pytest.raises(AgentError, match="observed GFP IDs"):
        select(request=request, transport=fake, knowledge=False, log_path=tmp_path / "unsupported.json")
    fake.final = {**_legal(ids), "mechanism": "buried rescue"}
    with pytest.raises(AgentError, match="Unexpected fields"):
        select(request=request, transport=fake, knowledge=False, log_path=tmp_path / "mechanism.json")


def test_alias_retry_preserves_full_roster_and_replays_without_explicit_mode_flag(tmp_path):
    request = _case()
    aliases = [f"C{index:03d}" for index in range(1, 5)]
    provider = FakeTransport(aliases, _legal(aliases))
    source = tmp_path / "alias.json"
    online = select(request=request, transport=provider, knowledge=False, log_path=source, repair_aliases=True)
    assert online["selected_ids"] == [row["id"] for row in request["candidates"][:2]]
    assert online["hypotheses"][0]["candidate_id"] == online["selected_ids"][0]
    decoded = json.loads(provider.generated[0]["messages"][1]["content"].partition("\n")[2])
    assert len(decoded["all_current_candidates"]) == len(request["candidates"])
    assert decoded["all_current_candidates"][0][0] == "C001"
    provider.generated.clear()
    replay = select(request=request, transport=provider, knowledge=False, log_path=tmp_path / "replay.json", replay_path=source)
    assert replay == online and not provider.generated


def test_prospective_threshold_support_is_distinct_from_incumbent_gain(tmp_path):
    request = _case()
    ids = [row["id"] for row in request["candidates"]]
    decision = select(request=request, transport=FakeTransport(ids[:4], _legal(ids)), knowledge=True, log_path=tmp_path / "pick.json")
    args = {"wild_type_fitness": request["wild_type_fitness"], "prior_best": request["current_best"]}
    assert evaluate_hypotheses(decision, measurements={}, **args)[0]["status"] == "PENDING"
    support = evaluate_hypotheses(decision, measurements={ids[0]: 3.8}, **args)[0]
    assert support["status"] == "SUPPORTED" and support["above_WT"] and not support["improves_incumbent"]
    assert support["mechanism_validation"] == support["feedback_gain"] == "NOT EVALUABLE"
    assert evaluate_hypotheses(decision, measurements={ids[0]: 3.72}, **args)[0]["status"] == "REFUTED"


def test_open_designs_wait_for_import_and_continue_with_refutation(tmp_path):
    initial = tmp_path / "initial.csv"
    initial.write_text("mutant,fitness\nWT,3.72\n" + "".join(f"{_mutation(i)},{2.2+i*.2}\n" for i in range(4, 7)))
    ids = [_mutation(pos) for pos in range(10, 14)]
    first_dir = tmp_path / "first"
    unordered = f"{_mutation(15)}:{ids[0]}"
    canonical = f"{ids[0]}:{_mutation(15)}"
    ids[0] = canonical
    one = run_open(measurements=initial, out=first_dir, model="pair_ridge", budget=2,
                   transport=FakeTransport([unordered, *ids[1:]], _legal(ids)))
    with (first_dir / "recommendations.csv").open() as stream:
        output = list(csv.DictReader(stream))
    assert all(row["source"] == "off_menu_design" and row["status"] == "PENDING" and row["measured_fitness"] == "" for row in output)
    assert one["pending_ids"] == sorted(ids[:2])
    feedback = tmp_path / "feedback.csv"
    feedback.write_text(f"mutant,fitness\n{ids[0]},2.01\n")
    roster = tmp_path / "roster.csv"
    roster.write_text("mutant\n" + "".join(f"{_mutation(i)}\n" for i in (20, 21, 22, 23)))
    novel = _mutation(24)
    next_ids = ["C001", "C002", "C003", f"NEW {novel}"]
    second_dir = tmp_path / "second"
    two = run_open(measurements=feedback, out=second_dir, model="pair_ridge", budget=2,
                   transport=FakeTransport(next_ids, _legal(["N004", "C001"], evidence=(ids[0],), adaptation="shift_to_exploration")),
                   prior_path=first_dir / "state.json", candidate_path=roster)
    audit = json.loads((second_dir / "hypothesis_audit.json").read_text())
    assert len(two["observed"]) == 5 and ids[0] not in two["pending_ids"] and ids[1] in two["pending_ids"]
    assert audit[0]["verdicts"][0]["status"] == "REFUTED"
    assert audit[1]["verdicts"][0]["status"] == "PENDING"
    assert audit[1]["verdicts"][0]["candidate_id"] == novel
    assert two["history"][1]["decision"]["recent_evidence_used"] == [ids[0]]
    edited = json.loads((first_dir / "state.json").read_text())
    edited["history"][0]["decision"]["hypotheses"][0]["comparator_fitness"] = 1.3
    (first_dir / "state.json").write_text(json.dumps(edited))
    with pytest.raises(ValueError, match="hypothesis ledger"):
        run_open(measurements=feedback, out=tmp_path / "tampered", model="pair_ridge", budget=2,
                 transport=FakeTransport([], {}), prior_path=first_dir / "state.json", candidate_path=roster)
