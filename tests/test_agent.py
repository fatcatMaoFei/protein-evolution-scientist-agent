"""Offline contracts for the CodeFlow tool-calling experiment."""

import json

import pytest

import llm_agent as agent


def recipe(cement=300):
    return dict(zip(agent.FEATURES, [cement, 0, 0, 160, 5, 900, 750, 28], strict=True))


@pytest.fixture
def views():
    eligible = {f"mix-{index:02d}": recipe(300 + index) for index in range(8)}
    observed = {"strong": {**recipe(300), "strength": 67}, "weak": {**recipe(260), "strength": 24}}
    return dict(
        eligible=eligible, observed=observed,
        scores={name: 40.0 - index for index, name in enumerate(eligible)},
        uncertainty={name: 2.0 for name in eligible}, k=2, seed=7, round_number=1,
    )


def proposal(first="C01"):
    draft = {
        "analyst": "Observed strengths vary across ingredient compositions.",
        "hypothesis": "A low predicted-strength miss is plausible when binder ratios differ.",
        "evidence_ids": ["strong"], "candidate_ids": [first, "C02", "C03", "C04"],
    }
    return {
        **generation(""), "finish_reason": "tool_calls",
        "tool_calls": [{"id": "call-example", "type": "function", "function": {"name": "evaluate_concrete_shortlist", "arguments": json.dumps(draft)}}],
    }


def generation(text):
    return {"text": text, "finish_reason": "stop", "usage": {"input_tokens": 100, "output_tokens": 40}}


def test_model_tool_call_executes_and_replays_without_inference(monkeypatch, tmp_path, views):
    received = []

    def synthetic_response(endpoint, request):
        assert endpoint == agent.DEFAULT_BASE
        received.append(request)
        if request["tool_choice"] != "none":
            return proposal()
        tool_reply = next(msg for msg in request["messages"] if msg["role"] == "tool")
        assert tool_reply["tool_call_id"] == "call-example"
        evaluator = json.loads(tool_reply["content"])
        assert evaluator[0]["alias"] == "C01"
        assert evaluator[0]["prediction_mpa"] == 40.0
        assert evaluator[0]["rf_tree_disagreement_mpa"] == 2.0
        assert "strength" not in evaluator[0]["recipe"]
        return generation(json.dumps({
            "candidate_ids": ["C03", "C01"], "critic": "Ingredient changes are confounded.",
            "selection_rationale": "C03 tests a lower ranked recipe alongside a high estimate.",
        }))

    monkeypatch.setattr(agent, "_generate", synthetic_response)
    live = tmp_path / "live.json"
    replay = tmp_path / "replay.json"
    selected = agent.choose_candidates(**views, augmented=True, log_path=live)
    assert selected == ["mix-02", "mix-00"]
    ledger = json.loads(live.read_text())
    assert ledger["mode"] == "live"
    assert ledger["usage"] == {"input_tokens": 200, "output_tokens": 80}
    assert ledger["tool_calls"][1]["trigger"] == "model_tool_call"
    assert not any("strength" in row["inputs"] for row in ledger["menu"])
    payload = json.loads(received[0]["messages"][-1]["content"].split("\n", 1)[1])
    assert payload["knowledge"]["observed_ratio_and_superplasticizer_split"]["groups"]
    assert agent.choose_candidates(**views, augmented=True, log_path=replay, replay_path=live) == selected
    assert json.loads(replay.read_text())["live_generation_count"] == 0
    assert len(received) == 2
    with pytest.raises(agent.AgentError, match="does not match current evidence"):
        agent.choose_candidates(**views, augmented=False, log_path=tmp_path / "wrong.json", replay_path=live)


def test_unavailable_alias_retries_then_fails_closed(monkeypatch, tmp_path, views):
    monkeypatch.setattr(agent, "_generate", lambda endpoint, request: proposal("C99"))
    trace = tmp_path / "failed.json"
    with pytest.raises(agent.AgentError, match="unique eligible aliases"):
        agent.choose_candidates(**views, augmented=False, log_path=trace)
    ledger = json.loads(trace.read_text())
    assert ledger["status"] == "error"
    assert "decision" not in ledger
    assert [row["attempt"] for row in ledger["calls"]] == [1, 2, 3]
    message = ledger["calls"][0]["request"]["messages"][-1]["content"]
    assert "knowledge" not in json.loads(message.split("\n", 1)[1])


def test_hidden_label_cannot_enter_candidate_interface(tmp_path, views):
    views["eligible"]["mix-01"]["strength"] = 99
    with pytest.raises(agent.AgentError, match="only the defined concrete columns"):
        agent.choose_candidates(**views, augmented=True, log_path=tmp_path / "invalid.json")
