"""P2 preserves the pair design and withholds scoring until decision acceptance."""

import json
from pathlib import Path

import pandas as pd

import audit_paired
import llm_agent

BASELINE = Path(__file__).resolve().parents[1] / "evidence" / "concrete"


def _fixture_provider(request):
    context = json.loads(request["messages"][1]["content"].split("\n", 1)[1])
    analysis = context["observation_summary"]
    has_feedback = analysis["observations"] == 53
    model_reply = {
        "model": llm_agent.MODEL, "response_id": "provider_fixture_not_live",
        "usage": {"input_tokens": 40, "output_tokens": 20},
    }
    if request["tool_choice"] != "none":
        return {
            **model_reply, "text": "", "finish_reason": "tool_calls",
            "tool_calls": [{
                "type": "function", "id": "fixture-call",
                "function": {"name": "evaluate_concrete_shortlist", "arguments": json.dumps({
                    "analyst": "The observations constrain a local comparison.",
                    "hypothesis": "Try some of the frozen top alternatives.",
                    "evidence_ids": [next(iter(analysis["strongest"]))],
                    "candidate_ids": [f"C{i:02}" for i in range(1, 8)],
                })},
            }],
        }
    return {
        **model_reply, "tool_calls": [], "finish_reason": "stop",
        "text": json.dumps({
            "candidate_ids": [f"C{i:02}" for i in (range(3, 8) if has_feedback else range(1, 6))],
            "critic": "The fixed predictor is unchanged by newly supplied measurements.",
            "selection_rationale": "Spend the batch budget in the same catalog.",
        }),
    }


def test_paired_audit_freezes_context_and_scores_after_all_offline_fixtures(monkeypatch, tmp_path):
    freeze = tmp_path / "protocol.md"
    freeze.write_text("Unit-test fixture. No remote model inference.\n", encoding="utf-8")
    output = tmp_path / "p2"
    calls = []

    def provider_fixture(endpoint, request):
        calls.append(request)
        return _fixture_provider(request)

    score = audit_paired._score

    def verify_scoring_barrier(*args):
        assert len(calls) == 16
        return score(*args)

    monkeypatch.setattr(llm_agent, "_generate", provider_fixture)
    monkeypatch.setattr(audit_paired, "_score", verify_scoring_barrier)
    result = audit_paired.run(output, baseline=BASELINE, freeze_path=freeze, selector=llm_agent.choose_candidates)
    assert result["status"] == "complete"
    assert result["pairs"] == 4 and result["decisions"] == 8
    pairs = pd.read_csv(output / "pairs.csv")
    assert pairs.selection_overlap.tolist() == [3] * 4
    assert pairs.set_changed.all()
    assert (pairs.no_feedback_greedy_overlap == 5).all()
    assert (pairs.no_feedback_best_mpa == pairs.model_only_best_mpa).all()
    assert (pairs.delta_best_mpa.round(10) == pairs.regret_reduction_mpa.round(10)).all()
    assert (pairs.with_feedback_regret_mpa >= 0).all()
    assert len(pd.read_csv(output / "selections.csv")) == 60

    for design_file in output.glob("pairs/*/*/design.json"):
        design = json.loads(design_file.read_text())
        excluded = set(design["added_feedback_ids"])
        assert len(excluded) == 5 and len(design["no_feedback_observed_ids"]) == 48
        assert not excluded.intersection(design["common_eligible_ids"])
        assert len(design["frozen_menu"]) == 48
        assert all("strength" not in entry["inputs"] for entry in design["frozen_menu"])
        no = json.loads(design_file.with_name("no-feedback-agent.json").read_text())
        yes = json.loads(design_file.with_name("with-feedback-agent.json").read_text())
        assert no["menu"] == yes["menu"] == design["frozen_menu"]
        no_call, yes_call = no["calls"][0]["request"], yes["calls"][0]["request"]
        assert {k: v for k, v in no_call.items() if k != "messages"} == {
            k: v for k, v in yes_call.items() if k != "messages"
        }
        for branch in (no_call, yes_call):
            payload = json.loads(branch["messages"][1]["content"].split("\n", 1)[1])
            assert len(payload["menu"]) == 48
        assert no_call["messages"] != yes_call["messages"]


def test_one_unaccepted_branch_blocks_every_outcome(monkeypatch, tmp_path):
    freeze = tmp_path / "protocol.md"
    freeze.write_text("Unit-test fixture.\n", encoding="utf-8")
    output = tmp_path / "p2"
    monkeypatch.setattr(llm_agent, "_generate", lambda _, request: _fixture_provider(request))

    def broken_selector(**kwargs):
        if kwargs["seed"] == 29 and len(kwargs["observed"]) == 53 and not kwargs["augmented"]:
            raise llm_agent.AgentError("Fixture deliberately rejects one branch.")
        return llm_agent.choose_candidates(**kwargs)

    result = audit_paired.run(output, baseline=BASELINE, freeze_path=freeze, selector=broken_selector)
    assert result["status"] == "NOT EVALUABLE"
    assert result["stage"] == "decision-validation"
    assert result["validated_decisions"] == 7
    assert not (output / "pairs.csv").exists()
    assert not (output / "selections.csv").exists()
