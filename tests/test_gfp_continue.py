import csv
import json
from types import SimpleNamespace

import pytest

from gfp_continue import load_csv, run_user_flow
from gfp_data import GFP_WT, GFPDataset, make_variant


def mutant(position):
    residue = GFP_WT[position - 1]
    return f"{residue}{position}{'A' if residue != 'A' else 'G'}"


@pytest.fixture
def inputs(tmp_path):
    initial = tmp_path / "initial.csv"
    initial.write_text("mutant,fitness\n" + "".join(f"{mutant(i)},{3+i/10}\n" for i in range(1, 5)))
    candidates = tmp_path / "candidates.csv"
    candidates.write_text(f"mutant\n{mutant(5)}\n{mutant(6)}\n")
    return initial, candidates


def test_user_workflow_leaves_unmeasured_scores_empty(tmp_path, inputs):
    class ForbiddenCatalogLabels:
        def __getitem__(self, key):
            raise AssertionError("Researcher workflow must not query catalog labels")

    dataset = SimpleNamespace(wild_type=GFP_WT, scores=ForbiddenCatalogLabels())
    initial, candidates = inputs
    first = tmp_path / "first"
    run_user_flow(dataset, first, stage="init", measured_csv=initial, prior_state=None,
                  batch_csv=None, candidate_csv=candidates, backend="deepseek", use_llm=False)
    rows = list(csv.DictReader((first / "recommendations.csv").open()))
    assert len(rows) == 2
    assert all(row["measured_fitness"] == "" and row["status"] == "prediction_only" for row in rows)

    feedback = tmp_path / "returned.csv"
    feedback.write_text(f"mutant,fitness\n{mutant(5)},3.9\n")
    next_pool = tmp_path / "next.csv"
    next_pool.write_text(f"mutant\n{mutant(6)}\n{mutant(7)}\n")
    second = tmp_path / "second"
    run_user_flow(dataset, second, stage="continue", measured_csv=None, prior_state=first / "state.json",
                  batch_csv=feedback, candidate_csv=next_pool, backend="deepseek", use_llm=False)
    state = json.loads((second / "state.json").read_text())
    assert state["new_measurements"] == [mutant(5)]
    assert state["proposed"] == [mutant(7)]
    assert set(state["outstanding"]) == {mutant(6), mutant(7)}
    assert len(state["observations"]) == 5


def test_csv_rejects_mismatch_and_duplicate(tmp_path):
    table = tmp_path / "measurements.csv"
    table.write_text(f"mutant,fitness,sequence\n{mutant(1)},4,{GFP_WT}\n")
    with pytest.raises(ValueError, match="Sequence does not match"):
        load_csv(table, measured=True)
    table.write_text(f"mutant,fitness\n{mutant(1)},3\n{mutant(1)},4\n")
    with pytest.raises(ValueError, match="Duplicate"):
        load_csv(table, measured=True)


def test_configured_init_and_continue_score_open_designs_without_archive_truth(tmp_path, inputs, monkeypatch):
    from gfp import main
    import gfp_data
    import gfp_setup

    class ForbiddenCatalogLabels:
        def __getitem__(self, key):
            raise AssertionError("Off-menu prediction must not query archive labels")

    class Model:
        model = "test-tool-model"
        endpoint = "unit-test-endpoint"
        config = {}

        def __init__(self):
            self.round = 1
            self.calls = 0

        def encode(self, request, *, native_continuation):
            return request

        def generate(self, endpoint, request):
            self.calls += 1
            base = {"usage": {"input_tokens": 1, "output_tokens": 1}, "model": self.model}
            if request["tool_choice"] == "none":
                scores = json.loads(next(row["content"] for row in request["messages"] if row["role"] == "tool"))
                assert len(scores) == 6 and all("fitness" not in row for row in scores)
                assert all(not row["roster_match"] for row in scores)
                self.round += 1
                return {**base, "finish_reason": "stop", "tool_calls": [], "text": json.dumps({
                    "candidate_ids": [f"N{i:02}" for i in range(1, 6)], "critic": "Held out pending data.",
                    "selection_rationale": "Prioritize the first five scored designs.",
                })}
            designs = [f"NEW {mutant(position)}" for position in range(10 * self.round, 10 * self.round + 6)]
            draft = {"analyst": "Review known fluorescence.", "hypothesis": "Substitutions differ in brightness.",
                     "evidence_ids": ["O1"], "designs": designs}
            return {**base, "finish_reason": "tool_calls", "text": "", "tool_calls": [
                {"type": "function", "id": "unit-novel-tool", "function": {
                    "name": "score_gfp_designs", "arguments": json.dumps(draft),
                }},
            ]}

    initial, _ = inputs
    # A candidate file consisting entirely of observed sequences leaves the
    # displayed menu empty. Open mode can still score and select a full batch.
    candidates = tmp_path / "exhausted-menu.csv"
    candidates.write_text("mutant\n" + "".join(f"{mutant(i)}\n" for i in range(1, 5)))
    adapter = Model()
    monkeypatch.setattr(gfp_data, "load_dataset", lambda _: GFPDataset(GFP_WT, {}, ForbiddenCatalogLabels(), {}, {}))
    monkeypatch.setattr(gfp_setup, "load_openai_profile", lambda name, **kwargs: adapter)
    flags = ["--model-profile", "my-api", "--allow-novel", "--candidate-file", str(candidates)]
    first, second = tmp_path / "first", tmp_path / "second"
    main(["init", "--input", str(initial), "--output", str(first), *flags])
    with (first / "recommendations.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["mutation"] for row in rows] == [mutant(i) for i in range(10, 15)]
    assert all(row["origin"] == "llm_off_menu" and row["decision"] == "openai_compatible" for row in rows)
    assert all(row["sequence"] == make_variant(row["mutation"]).sequence and row["measured_fitness"] == "" for row in rows)

    returned = tmp_path / "returned.csv"
    returned.write_text(f"mutant,fitness\n{mutant(10)},3.9\n")
    main(["continue", "--state", str(first / "state.json"), "--measurements", str(returned),
          "--output", str(second), *flags])
    state = json.loads((second / "state.json").read_text())
    request = json.loads((second / "agent_request.json").read_text())
    assert request["latest_feedback_ids"] == [mutant(10)]
    assert request["pending_ids"] == sorted(mutant(i) for i in range(11, 15))
    assert state["new_measurements"] == [mutant(10)]
    assert state["prior_state_path"] == "../first/state.json"
    assert set(state["artifacts"]) == {"agent.json", "agent_request.json", "imported_measurements.csv", "recommendations.csv"}
    assert adapter.calls == 4


@pytest.mark.parametrize("bad_design", ["NEW W1G", "NEW WT", "NEW A238D", "NEW " + mutant(10)])
def test_open_mode_rejects_wrong_reference_and_excluded_designs(bad_design):
    from gfp_agent import TOOL, _proposal
    from llm_agent import AgentError

    arguments = {"analyst": "Analyze GFP.", "hypothesis": "Test GFP brightness.",
                 "evidence_ids": ["O1"], "designs": [bad_design]}
    response = {"tool_calls": [{"type": "function", "id": "test", "function": {
        "name": TOOL, "arguments": json.dumps(arguments),
    }}]}
    with pytest.raises(AgentError, match="NEW design"):
        _proposal(response, {}, {"O1": {}}, 1, 4, excluded={mutant(10)})
