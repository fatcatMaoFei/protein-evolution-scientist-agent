from pathlib import Path

import audit_paired
import audit_trace
import decision_audit
import llm_agent


def test_audit_entry_forwards_frozen_model_once(monkeypatch, tmp_path):
    received = []

    def choose(**kwargs):
        received.append(kwargs)
        return ["recipe-1"]

    def run_pairs(output, *, baseline, selector, freeze_path, replay):
        assert baseline == tmp_path / "archive/concrete"
        assert freeze_path.name == "audit_protocol.md"
        assert not replay
        assert selector(model=llm_agent.MODEL, log_path=output / "agent.json") == ["recipe-1"]
        return {"status": "NOT EVALUABLE", "reason": "Fixture has no paired outcomes"}

    monkeypatch.setattr(decision_audit, "_frozen_manifest", lambda _: {"file": "frozen"})
    monkeypatch.setattr(audit_trace, "reconstruct_decisions", lambda _: [])
    monkeypatch.setattr(audit_paired, "run", run_pairs)
    monkeypatch.setattr(llm_agent, "choose_candidates", choose)
    out = tmp_path / "audited"
    result = decision_audit.run(Path(tmp_path / "archive"), out, paired=True)
    assert len(received) == 1
    assert received[0]["model"] == llm_agent.MODEL
    assert result["paired_status"] == "NOT EVALUABLE"
    assert result["frozen_inputs_unchanged"]
    assert not (out / "paired_feedback_results.csv").exists()
