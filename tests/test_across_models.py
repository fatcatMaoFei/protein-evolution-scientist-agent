"""Check four-model contrast construction and the Responses-native tool shape."""

from across_models import contrasts_for_seed
from factorial_analysis import OUTCOMES, cached_usage, canonical_proposal


def test_all_model_effects_paired_to_the_same_seed_and_greedy_arm():
    models = ("deepseek", "sonnet", "gpt", "qwen")
    endpoints = {
        ("deepseek", "off"): 1, ("deepseek", "on"): 3,
        ("sonnet", "off"): 1, ("sonnet", "on"): 4,
        ("gpt", "off"): 3, ("gpt", "on"): 1,
        ("qwen", "off"): 1, ("qwen", "on"): 2,
        ("greedy", "na"): 4,
    }
    cells = {(17, *key): dict.fromkeys(OUTCOMES, score) for key, score in endpoints.items()}
    contrasts = {
        row["contrast"]: row["gain_vs_initial_best_mpa"]
        for row in contrasts_for_seed(cells, 17, models)
    }
    assert contrasts["gpt_knowledge_on_minus_off"] == -2
    assert contrasts["gpt_minus_deepseek_knowledge_off"] == 2
    assert contrasts["qwen_minus_gpt_knowledge_off"] == -2
    assert contrasts["interaction_qwen_minus_deepseek"] == -1
    assert contrasts["qwen_on_minus_shared_greedy"] == -2
    assert len(contrasts) == 27


def test_responses_tool_rendering_matches_the_common_payload():
    content = [{"role": "user", "content": "test"}]
    function = {"name": "tool", "description": "Check", "parameters": {"type": "object"}}
    generated = {
        "instructions": "Shared instructions", "input": content,
        "tools": [{"type": "function", "strict": True, **function}],
    }
    reference = {
        "messages": [{"role": "system", "content": "Shared instructions"}, *content],
        "tools": [{"type": "function", "function": function}],
    }
    assert canonical_proposal(generated) == canonical_proposal(reference)


def test_responses_input_count_includes_observed_cache_hits():
    record = {
        "usage": {"input_tokens": 15, "output_tokens": 3},
        "provider_usage": {"input_tokens": 15, "input_tokens_details": {"cached_tokens": 7}},
    }
    assert cached_usage(record) == {"cache_read_input_tokens": 7, "provider_primary_input_tokens": 15}
