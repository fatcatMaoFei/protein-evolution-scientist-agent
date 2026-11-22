"""Validate the paired contrasts and cross-transport equivalence checks."""

from pathlib import Path

import pytest

from factorial_analysis import (
    CONDITIONS, SCHEDULE_FIELDS, build_contrasts, cached_usage, canonical_proposal, match_schedule,
)
from verify import read


def test_paired_interaction_is_difference_of_knowledge_effects():
    paired_values = dict(zip(CONDITIONS, (2, 6, 5, 8), strict=True))
    assert build_contrasts(paired_values) == {
        "knowledge_on_minus_off_deepseek": 4,
        "knowledge_on_minus_off_sonnet": 3,
        "sonnet_minus_deepseek_knowledge_off": 3,
        "sonnet_minus_deepseek_knowledge_on": 2,
        "interaction_sonnet_minus_deepseek": -1,
    }


def test_run_pair_requires_the_same_budget():
    left = dict.fromkeys(SCHEDULE_FIELDS, 1)
    right = {**left, "budget": 2}
    with pytest.raises(ValueError, match="budget"):
        match_schedule(left, right)


def test_native_transports_share_the_same_first_decision_prompt_and_tool_specification():
    path = Path(__file__).resolve().parents[1] / "evidence"
    relative = Path("arms/seed-7/C_llm/round-01-agent.json")
    requests = [read(path / stage / relative)["calls"][0]["request"]
                for stage in ("smoke", "sonnet-smoke")]
    assert canonical_proposal(requests[0]) == canonical_proposal(requests[1])


def test_native_token_accounting_distinguishes_cache_subtotals():
    provider = {"usage": {"input_tokens": 12}}
    openai_style = {**provider, "provider_usage": {"prompt_tokens": 12,
                                                   "prompt_tokens_details": {"cached_tokens": 5}}}
    native = {"usage": {"input_tokens": 24}, "provider_usage": {
        "input_tokens": 12, "cache_read_input_tokens": 5, "cache_creation_input_tokens": 7,
    }}
    assert cached_usage(openai_style)["provider_primary_input_tokens"] == 12
    assert cached_usage(native)["provider_primary_input_tokens"] == 12
    assert cached_usage(native)["cache_creation_input_tokens"] == 7
