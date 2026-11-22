"""Native Messages transport and trace replay tests with offline responses."""

import json
from types import SimpleNamespace

import pytest

import sonnet_transport as sonnet
from llm_agent import FEATURES, MAX_TOKENS, AgentError


def _response(content, stop):
    message_type = pytest.importorskip("anthropic.types").Message
    return message_type.model_validate({
        "id": "msg_offline", "role": "assistant", "type": "message",
        "model": sonnet.SONNET_MODEL, "content": content, "stop_reason": stop,
        "usage": {"input_tokens": 23, "cache_creation_input_tokens": 7, "cache_read_input_tokens": 5, "output_tokens": 42},
    })


def _views():
    recipe = dict(zip(FEATURES, [300, 0, 0, 160, 5, 900, 750, 28], strict=True))
    eligible = {f"mix-{n}": {**recipe, "cement": 300 + n} for n in range(1, 7)}
    return {
        "eligible": eligible, "observed": {"known": {**recipe, "strength": 54.2}},
        "scores": {key: 50.0 - n for n, key in enumerate(eligible)},
        "k": 2, "seed": 7, "round_number": 1, "augmented": True,
    }


def test_native_tool_continuation_json_and_credential_free_replay(monkeypatch, tmp_path):
    pytest.importorskip("anthropic")
    tool_args = {
        "analyst": "The one observed mix is the reference.", "hypothesis": "Nearby binder change might help.",
        "evidence_ids": ["known"], "candidate_ids": ["C01", "C02", "C03", "C04"],
    }
    native_assistant = [
        {"type": "thinking", "thinking": "", "signature": "native-signed-block"},
        {"type": "tool_use", "id": "toolu_concrete", "name": "evaluate_concrete_shortlist", "input": tool_args},
    ]
    calls = []

    def offline_create(**request):
        calls.append(request)
        assert request["thinking"] == {"type": "between_tools"}
        assert request["max_tokens"] == MAX_TOKENS
        assert "temperature" not in request
        assert request["model"] == sonnet.SONNET_MODEL
        assert request["tools"][0]["input_schema"]["additionalProperties"] is False
        if len(calls) == 1:
            assert request["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
            return _response(native_assistant, "tool_use")
        assert request["tool_choice"] == {"type": "none"}
        assert request["output_config"]["format"]["schema"] == sonnet.FINAL_SCHEMA
        assert request["messages"][1] == {"role": "assistant", "content": native_assistant}
        tool_output = request["messages"][2]["content"][0]
        assert tool_output["type"] == "tool_result"
        assert tool_output["tool_use_id"] == "toolu_concrete"
        assert [row["alias"] for row in json.loads(tool_output["content"])] == tool_args["candidate_ids"]
        return _response([{"type": "text", "text": json.dumps({
            "candidate_ids": ["C02", "C04"], "critic": "The reference alone does not isolate the slope.",
            "selection_rationale": "Probe two high-ranked recipes.",
        })}], "end_turn")

    monkeypatch.setattr(sonnet, "_client", lambda endpoint: SimpleNamespace(messages=SimpleNamespace(create=offline_create)))
    original = tmp_path / "sonnet.json"
    outcome = sonnet.choose_sonnet_candidates(**_views(), log_path=original)
    assert outcome == ["mix-2", "mix-4"]
    recorded = json.loads(original.read_text())
    assert recorded["tool_calls"][1]["call_id"] == "toolu_concrete"
    assert recorded["usage"] == {"input_tokens": 70, "output_tokens": 84}
    assert recorded["calls"][1]["request"]["model"] == sonnet.SONNET_MODEL
    assert recorded["calls"][0]["response"]["provider_stop_reason"] == "tool_use"
    assert recorded["calls"][0]["response"]["provider_usage"]["cache_read_input_tokens"] == 5
    assert recorded["calls"][0]["response"]["native_content"] == native_assistant

    replay = tmp_path / "replay.json"
    assert sonnet.choose_sonnet_candidates(**_views(), log_path=replay, replay_path=original) == outcome
    assert len(calls) == 2
    assert json.loads(replay.read_text())["live_generation_count"] == 0


def test_sonnet_endpoint_is_restricted(monkeypatch):
    monkeypatch.setenv("CODEFLOW_SONNET_BASE_URL", "https://codeflow.sh/v1/messages")
    assert sonnet.SonnetTransport().endpoint == "https://codeflow.sh"
    monkeypatch.setenv("CODEFLOW_SONNET_BASE_URL", "https://example.com/v1/messages")
    with pytest.raises(AgentError, match="requires the CodeFlow"):
        _ = sonnet.SonnetTransport().endpoint
