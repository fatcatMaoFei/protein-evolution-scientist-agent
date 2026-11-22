"""Exercise Responses-native tool continuation and replay without network access."""

import json
from types import SimpleNamespace

import pytest

import gpt_transport as gpt
from llm_agent import AgentError, FEATURES, MAX_TOKENS


def _response(output, *, model=gpt.GPT_MODEL, status="completed"):
    response = pytest.importorskip("openai.types.responses").Response
    return response.model_validate({
        "object": "response", "created_at": 0, "id": "resp_offline", "status": status,
        "model": model, "parallel_tool_calls": False, "tool_choice": "none", "tools": [], "output": output,
        "usage": {
            "input_tokens": 23, "output_tokens": 42, "total_tokens": 65,
            "input_tokens_details": {"cached_tokens": 7}, "output_tokens_details": {"reasoning_tokens": 0},
        },
    })


def _views():
    recipe = dict(zip(FEATURES, [300, 0, 0, 160, 5, 900, 750, 28], strict=True))
    eligible = {f"mix-{n}": {**recipe, "cement": 300 + n} for n in range(1, 7)}
    return {
        "eligible": eligible, "observed": {"known": {**recipe, "strength": 54.2}},
        "scores": {key: 50.0 - n for n, key in enumerate(eligible)},
        "k": 2, "seed": 7, "round_number": 1, "augmented": True,
    }


def test_native_responses_tool_continuation_and_credential_free_replay(monkeypatch, tmp_path):
    tool_args = {
        "analyst": "The observed recipe is a local reference.",
        "hypothesis": "Nearby binder changes could improve measured strength.",
        "evidence_ids": ["known"], "candidate_ids": ["C01", "C02", "C03", "C04"],
    }
    native_call = {
        "id": "fc_concrete", "call_id": "call_concrete",
        "type": "function_call", "name": "evaluate_concrete_shortlist",
        "arguments": json.dumps(tool_args), "status": "completed",
    }
    calls = []

    def offline_create(**request):
        calls.append(request)
        assert request["model"] == gpt.GPT_MODEL
        assert request["max_output_tokens"] == MAX_TOKENS
        assert request["reasoning"] == {"effort": "none"}
        assert request["store"] is False and request["parallel_tool_calls"] is False
        assert request["tools"][0]["strict"] is True
        assert request["tools"][0]["parameters"]["additionalProperties"] is False
        assert len(request["instructions"]) > 0
        assert request["input"][0]["role"] == "user"
        if len(calls) == 1:
            assert request["tool_choice"] == {"type": "function", "name": "evaluate_concrete_shortlist"}
            return _response([native_call])
        assert request["tool_choice"] == "none"
        assert request["text"]["format"]["schema"] == gpt.FINAL_SCHEMA
        assert request["text"]["format"]["strict"] is True
        assert request["input"][1] == native_call
        tool_result = request["input"][2]
        assert tool_result["type"] == "function_call_output" and tool_result["call_id"] == "call_concrete"
        assert [row["alias"] for row in json.loads(tool_result["output"])] == tool_args["candidate_ids"]
        return _response([{
            "id": "msg_concrete", "type": "message", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "annotations": [], "text": json.dumps({
                "candidate_ids": ["C02", "C04"], "critic": "One reference is insufficient to infer interaction effects.",
                "selection_rationale": "Include two nearby high-ranked recipes.",
            })}],
        }])

    monkeypatch.setattr(gpt, "_client", lambda endpoint: SimpleNamespace(responses=SimpleNamespace(create=offline_create)))
    original = tmp_path / "gpt.json"
    outcome = gpt.choose_gpt_candidates(**_views(), log_path=original)
    assert outcome == ["mix-2", "mix-4"]
    trace = json.loads(original.read_text())
    assert trace["tool_calls"][1]["call_id"] == "call_concrete"
    assert trace["usage"] == {"input_tokens": 46, "output_tokens": 84}
    assert trace["calls"][0]["response"]["native_content"] == [native_call]
    assert trace["calls"][0]["response"]["provider_usage"]["input_tokens_details"]["cached_tokens"] == 7

    replay = tmp_path / "replay.json"
    assert gpt.choose_gpt_candidates(**_views(), log_path=replay, replay_path=original) == outcome
    assert len(calls) == 2
    assert json.loads(replay.read_text())["live_generation_count"] == 0


def test_gpt_endpoint_restriction_and_model_provenance(monkeypatch):
    monkeypatch.setenv("CODEFLOW_GPT_BASE_URL", "https://codeflow.sh/v1/responses")
    assert gpt.GPTTransport().endpoint == gpt.DEFAULT_BASE
    monkeypatch.setenv("CODEFLOW_GPT_BASE_URL", "https://example.com/v1/responses")
    with pytest.raises(AgentError, match="requires the CodeFlow"):
        _ = gpt.GPTTransport().endpoint
    assert gpt._matches_model(gpt.GPT_MODEL, "gpt-6-luna-2026-11-18")
    assert not gpt._matches_model(gpt.GPT_MODEL, "gpt-6-sol")
