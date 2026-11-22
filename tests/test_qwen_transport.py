"""Verify the local model trace and native tool round trip without loading weights."""

import json
from types import SimpleNamespace

import qwen_transport as qwen
from llm_agent import FEATURES


def test_qwen_native_output_and_replay(monkeypatch, tmp_path):
    recipe = dict(zip(FEATURES, [300, 0, 0, 160, 5, 900, 750, 28], strict=True))
    eligible = {f"mix-{n}": {**recipe, "cement": 300 + n} for n in range(1, 7)}
    views = {
        "eligible": eligible, "observed": {"known": {**recipe, "strength": 54.2}},
        "scores": {key: 50.0 - n for n, key in enumerate(eligible)},
        "k": 2, "seed": 7, "round_number": 1, "augmented": False,
    }
    arguments = {
        "analyst": "The one observed mix is the reference.", "hypothesis": "Nearby binder change may help.",
        "evidence_ids": ["known"], "candidate_ids": ["C01", "C02", "C03", "C04"],
    }
    raw_proposal = json.dumps(arguments) + '}\n</tool_call>'
    raw_final = json.dumps({
        "candidate_ids": ["C02", "C04"], "critic": "A single measured example gives limited evidence.",
        "selection_rationale": "Probe two high-ranked recipes.",
    })
    generated = []

    def local_generation(request):
        generated.append(request)
        messages = qwen._rendered_messages(request)
        if len(generated) == 1:
            assert request["tools"][0]["function"]["name"] == "evaluate_concrete_shortlist"
            assert request["assistant_prefill"] == qwen.TOOL_PREFILL
            assert "MEASURED recipe IDs from this set: known" in request["structural_contract"]
            assert "exactly 4 distinct UNMEASURED candidate aliases" in request["structural_contract"]
            raw = raw_proposal
        else:
            assert request["tools"] is None
            assert "exactly 2 distinct aliases from the evaluated set: C01, C02, C03, C04" in request["structural_contract"]
            call = next(message for message in messages if message["role"] == "assistant")["tool_calls"][0]
            assert call["function"]["arguments"] == arguments
            result = next(message for message in messages if message["role"] == "tool")
            assert result["tool_call_id"].startswith("qwen-")
            assert all("strength" not in row["recipe"] for row in json.loads(result["content"]))
            raw = raw_final
        return "rendered prompt " + str(len(generated)) + request["assistant_prefill"], raw, SimpleNamespace(prompt_tokens=41, generation_tokens=53, finish_reason="stop")

    monkeypatch.setattr(qwen, "_generate_local", local_generation)
    monkeypatch.setattr(qwen, "version", lambda _: "offline-test")
    live = tmp_path / "live.json"
    assert qwen.choose_qwen_candidates(**views, log_path=live) == ["mix-2", "mix-4"]
    trace = json.loads(live.read_text())
    assert trace["config"]["revision"] == qwen.QWEN_REVISION
    assert trace["config"]["structural_contract"] == qwen.CONTRACT_VERSION
    first = trace["calls"][0]["response"]
    assert first["raw_output"] == raw_proposal and first["rendered_prompt"] == "rendered prompt 1" + qwen.TOOL_PREFILL
    assert first["response_id"].startswith("local-") and first["finish_reason"] == "tool_calls"
    assert trace["usage"] == {"input_tokens": 82, "output_tokens": 106}
    replay = tmp_path / "replay.json"
    assert qwen.choose_qwen_candidates(**views, log_path=replay, replay_path=live) == ["mix-2", "mix-4"]
    assert len(generated) == 2 and json.loads(replay.read_text())["live_generation_count"] == 0


def test_qwen_rejects_plain_or_multiple_tool_responses():
    one = '<tool_call>{"name":"f","arguments":{}}</tool_call>'
    assert qwen._parse_native_call(one)[0]["function"]["name"] == "f"
    assert qwen._parse_native_call(one + one) == []
    assert qwen._parse_native_call('{"name":"f","arguments":{}}') == []
