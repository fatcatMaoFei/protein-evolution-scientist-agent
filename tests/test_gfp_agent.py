import copy
import json
from pathlib import Path

import pytest

import gfp_agent
from gfp_data import GFP_WT, make_variant
from llm_agent import AgentError


def measured_case():
    ids = []
    for pos in range(3, 16):
        source = GFP_WT[pos - 1]
        mutant = "A" if source != "A" else "G"
        variant = make_variant(f"{source}{pos}{mutant}")
        ids.append({"id": variant.id, "mutations": variant.mutant, "sequence": variant.sequence,
                    "edits": list(variant.edits)})
    observed = [{**row, "fitness": 2.0 + i / 10} for i, row in enumerate(ids[:8])]
    menu = [
        {**row, "prediction": 2.5 + i / 100, "uncertainty": 0.1, "rank": i + 1,
         "source": "leader", "alias": f"C{i + 1:02d}"}
        for i, row in enumerate(ids[8:])
    ]
    return {
        "domain": "gfp_fluorescence", "reference_sequence": GFP_WT, "max_mutations": 4,
        "budget": 2, "seed": 7, "round_number": 2, "knowledge": True,
        "fitness_direction": "maximize", "observed": observed, "menu": menu,
        "latest_feedback_ids": [observed[3]["id"]],
    }


class _TestOnlyNativeAdapter:
    model = "fixture-only"
    endpoint = "local:unit-tests"
    config = {"backend": "fixture-only"}

    def __init__(self, case):
        self.case = case
        self.calls = 0

    def encode(self, request, *, native_continuation):
        return request

    def generate(self, endpoint, request):
        self.calls += 1
        reply = {"usage": {"input_tokens": 10, "output_tokens": 20}, "model": self.model,
                 "response_id": f"test-response-{self.calls}"}
        if request["tool_choice"] == "none":
            return {**reply, "finish_reason": "stop", "tool_calls": [],
                    "text": json.dumps({"candidate_ids": ["C01", "C02"], "critic": "Modeled only.",
                                         "selection_rationale": "Compare a short batch."})}
        arguments = {
            "analyst": "Compare the selected variants.",
            "hypothesis": "Observed brightness can vary with context.",
            "evidence_ids": ["O1"],
            "designs": [f"{row['alias']} {row['mutations']}" for row in self.case["menu"][:3]],
        }
        return {**reply, "finish_reason": "tool_calls", "text": "", "tool_calls": [
            {"type": "function", "id": "unit-tool-call", "function": {
                "name": gfp_agent.TOOL, "arguments": json.dumps(arguments),
            }}
        ]}


def test_native_tool_contract_and_digest_replay_measured_only(tmp_path, monkeypatch):
    case = measured_case()
    adapter = _TestOnlyNativeAdapter(case)
    monkeypatch.setattr(gfp_agent, "_transport", lambda _: adapter)
    live = tmp_path / "live.json"
    replay = tmp_path / "replay.json"
    expected = [row["id"] for row in case["menu"][:2]]
    assert gfp_agent.select_candidates(case, log_path=live) == expected
    assert adapter.calls == 2
    assert gfp_agent.select_candidates(case, log_path=replay, replay_path=live) == expected
    assert adapter.calls == 2
    record = json.loads(live.read_text())
    assert record["tool_calls"][0]["trigger"] == "model_tool_call"
    assert record["tool_calls"][0]["result"][0]["sequence"] == case["menu"][0]["sequence"]
    assert "fitness" not in record["tool_calls"][0]["result"][0]
    assert record["decision"]["evidence_ids"][0] == case["observed"][-1]["id"]
    assert set(record["latest_feedback_ids"]) <= {row["id"] for row in record["observed_evidence"].values()}
    assert json.loads(replay.read_text())["live_generation_count"] == 0
    record["calls"][1]["response"]["text"] += " "
    live.write_text(json.dumps(record))
    with pytest.raises(AgentError, match="digest"):
        gfp_agent.select_candidates(case, log_path=tmp_path / "corrupt.json", replay_path=live)


def test_reject_non_gfp_target_and_hidden_labels(tmp_path):
    case = measured_case()
    other = copy.deepcopy(case)
    other["reference_sequence"] = "A" + GFP_WT[1:]
    with pytest.raises(AgentError, match="pinned"):
        gfp_agent.select_candidates(other, log_path=tmp_path / "other.json")
    case["menu"][0]["fitness"] = 3.02
    with pytest.raises(AgentError, match="unexpected fields"):
        gfp_agent.select_candidates(case, log_path=tmp_path / "hidden.json")


def test_novel_prompt_does_not_change_archived_qwen_roster_wire():
    arm = Path(__file__).resolve().parents[1] / "evidence/gfp/formal/qwen/arms/seed-17/C_llm"
    case = json.loads((arm / "round-01-request.json").read_text())
    recorded = json.loads((arm / "round-01-agent.json").read_text())
    adapter = gfp_agent.QwenGFP()
    menu, evidence, shortlist = gfp_agent._prepare(case)
    request = {
        "model": adapter.model, "messages": gfp_agent._messages(case, menu, evidence, shortlist),
        "tools": gfp_agent.TOOLS, "tool_choice": {"type": "function", "function": {"name": gfp_agent.TOOL}},
        "max_tokens": recorded["config"]["proposal_max_tokens"],
    }
    assert adapter.encode(request, native_continuation=None) == recorded["calls"][0]["request"]

    case["pending_ids"] = []
    request["messages"] = gfp_agent._messages(case, menu, evidence, shortlist, allow_novel=True)
    assert "NEW followed by legal off-menu edits" in adapter.encode(request, native_continuation=None)["structural_contract"]


def test_tool_rejects_candidate_when_explicit_edits_disagree():
    case = measured_case()
    menu, evidence, size = gfp_agent._prepare(case)
    encoded = _TestOnlyNativeAdapter(case).generate("local:unit-tests", {"tool_choice": {}})
    function = encoded["tool_calls"][0]["function"]
    edited = json.loads(function["arguments"])
    edited["designs"][0] = f"C01 {case['menu'][1]['mutations']}"
    function["arguments"] = json.dumps(edited)
    with pytest.raises(AgentError, match="do not reconstruct"):
        gfp_agent._proposal(encoded, menu, evidence, size, limit=4)


def test_fenced_final_retains_selection_checks():
    body = json.dumps({"candidate_ids": ["C01", "C02"], "critic": "Modeled only.",
                       "selection_rationale": "Diverse shortlist."})
    fenced = "`" * 3 + "json\n" + body + "\n" + "`" * 3
    assert gfp_agent._parse_gfp_final(fenced, {"C01", "C02"}, 2)["candidate_ids"] == ["C01", "C02"]
    with pytest.raises(AgentError, match="unique"):
        gfp_agent._parse_gfp_final(fenced, {"C01", "C03"}, 2)


def test_rate_limit_backoff_is_archived_and_replayed_without_waits(tmp_path, monkeypatch):
    case = measured_case()
    adapter = _TestOnlyNativeAdapter(case)
    generate = adapter.generate
    real_attempts = []
    sleeps = []

    def limited(endpoint, request):
        real_attempts.append(request)
        if len(real_attempts) <= 2:
            raise AgentError("CodeFlow HTTP status 429")
        return generate(endpoint, request)

    monkeypatch.setattr(adapter, "generate", limited)
    monkeypatch.setattr(gfp_agent, "_transport", lambda _: adapter)
    monkeypatch.setattr(gfp_agent.time, "sleep", sleeps.append)
    live = tmp_path / "limited.json"
    selected = gfp_agent.select_candidates(case, log_path=live)
    assert sleeps == [15, 35]
    assert real_attempts[0] == real_attempts[1] == real_attempts[2]
    assert gfp_agent.select_candidates(case, log_path=tmp_path / "checked.json", replay_path=live) == selected
    assert len(real_attempts) == 4
    assert sleeps == [15, 35]
    assert json.loads(live.read_text())["calls"][1]["retry_after_s"] == 35
