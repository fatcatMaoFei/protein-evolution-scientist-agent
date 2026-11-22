"""Audited CodeFlow tool-calling agent for measured 28-day concrete recipes."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from statistics import mean, median
from urllib.parse import urlsplit


FEATURES = (
    "cement", "slag", "fly_ash", "water", "superplasticizer",
    "coarse_aggregate", "fine_aggregate", "age",
)
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
DEFAULT_BASE = "https://codeflow.sh/v1"
TOP_MENU, RANDOM_MENU = 24, 24
MAX_TOKENS = 512
MODEL_CONFIG = {"backend": "CodeFlow Chat Completions", "temperature": 0.0, "max_tokens": MAX_TOKENS, "thinking": {"type": "disabled"}}
SYSTEM = (
    "You are running a retrospective decision experiment on measured 28-day concrete mixtures. "
    "Maximize measured compressive strength (MPa) under a fixed small batch budget. "
    "Treat model predictions as estimates and only reported observations as measured. "
    "You may choose high predictions, or use a plausible and evidence-linked reason to explore. "
    "Candidate aliases only identify entries in the displayed menu. Never invent recipes or references. "
    "Do not use remembered values for candidate recipes. Mention both uncertainty and likely upside."
)
KNOWLEDGE = {
    "fixed_materials_context": [
        "Water to total cementitious mass is water/(cement + slag + fly_ash). Lower ratios often correspond to higher compressive strength, conditional on workable mixes and comparability of binder, aggregates, and curing.",
        "Superplasticizer can permit workable mixes at lower water content. Its effect need not be additive with water content.",
        "Replacing cement mass with slag or fly ash changes binder hydration. Total binder mass alone cannot identify the effect of replacement at a fixed curing age.",
        "Single-column summaries confound the other ingredients, and evidence across a combination does not identify the independent effect of each change.",
    ],
    "sources": [
        "FHWA-HRT-04-150 chapter 9, https://www.fhwa.dot.gov/publications/research/infrastructure/pavements/pccp/04150/chapt9.cfm",
        "FHWA-04-122 chapter 5, https://www.fhwa.dot.gov/publications/research/infrastructure/pavements/pccp/05042/05.cfm",
    ],
}
ARRAY_STR = {"type": "array", "items": {"type": "string"}}
STRING = {"type": "string"}
PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "analyst": STRING, "hypothesis": STRING,
        "evidence_ids": ARRAY_STR, "candidate_ids": ARRAY_STR,
    },
    "required": ["analyst", "hypothesis", "evidence_ids", "candidate_ids"],
    "additionalProperties": False,
}
TOOLS = [{
    "type": "function",
    "function": {
        "name": "evaluate_concrete_shortlist",
        "description": "Return model estimates and check recipe eligibility for the proposed concrete candidate aliases. Carries your analyst findings, testable hypothesis, and cited measured example IDs.",
        "parameters": PROPOSAL_SCHEMA,
    },
}]
ID_RE = re.compile(r"[A-Za-z0-9_.-]{1,48}\Z")


class AgentError(ValueError):
    """No valid agent decision was available for the oracle."""


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def _digest(value) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _check_inputs(eligible, observed, scores, uncertainty, k):
    if not 1 <= k <= len(eligible) or k > TOP_MENU + RANDOM_MENU or not observed:
        raise AgentError("Candidate budget or observation count is invalid")
    if eligible.keys() != scores.keys() or eligible.keys() & observed.keys():
        raise AgentError("Candidate, score and observation identifiers are inconsistent")
    for rows, expected in ((eligible, set(FEATURES)), (observed, set(FEATURES) | {"strength"})):
        for key, row in rows.items():
            if not isinstance(key, str) or not ID_RE.fullmatch(key) or set(row) != expected:
                raise AgentError("Every recipe must use only the defined concrete columns and an opaque ID")
            if any(not _number(value) or value < 0 for value in row.values()) or row["age"] != 28:
                raise AgentError("Recipes must have finite nonnegative fields and a fixed 28-day age")
            if row["cement"] + row["slag"] + row["fly_ash"] <= 0:
                raise AgentError("Recipes must contain a positive mass of cementitious materials")
    if any(not _number(v) for v in scores.values()):
        raise AgentError("Model estimates must be numeric and finite")
    if uncertainty is not None and (uncertainty.keys() != scores.keys() or any(
        not _number(value) or value < 0 for value in uncertainty.values()
    )):
        raise AgentError("RF tree disagreement must be finite and aligned with the score view")


def _ordered_menu(eligible, scores, uncertainty, seed, round_number):
    ranked = sorted(scores, key=lambda key: (-scores[key], key))
    count = min(RANDOM_MENU, len(ranked) - min(len(ranked), TOP_MENU))
    rng = random.Random(f"concrete-menu:{seed}:{round_number}")
    selected = ranked[:TOP_MENU] + rng.sample(ranked[TOP_MENU:], count)
    ranks = {key: rank for rank, key in enumerate(ranked, 1)}
    return [
        {
            "alias": f"C{index:02d}", "id": key,
            "source": "top_prediction" if index <= TOP_MENU else "seeded_sample",
            "prediction": scores[key], "rank": ranks[key], "inputs": dict(eligible[key]),
            "rf_tree_disagreement": uncertainty[key] if uncertainty is not None else None,
        }
        for index, key in enumerate(selected, 1)
    ]


def data_analyst(observed):
    """Provide both LLM arms the same compact observed-only descriptive tool output."""
    ordered = sorted(observed, key=lambda key: (-observed[key]["strength"], key))
    example_count = min(4, len(ordered))
    strongest = ordered[:example_count]
    weakest = [key for key in reversed(ordered) if key not in strongest][:example_count]
    contrasts = {}
    for feature in FEATURES[:-1]:
        cut = median(row[feature] for row in observed.values())
        low = [row["strength"] for row in observed.values() if row[feature] <= cut]
        high = [row["strength"] for row in observed.values() if row[feature] > cut]
        contrasts[feature] = [cut, len(low), round(mean(low), 2), len(high), round(mean(high), 2) if high else None]
    return {
        "observations": len(ordered),
        "strongest": {key: dict(observed[key]) for key in strongest},
        "weakest": {key: dict(observed[key]) for key in weakest},
        "descriptive_associations_columns": ["feature_cut", "n_at_or_below", "mean_at_or_below_mpa", "n_above", "mean_above_mpa"],
        "descriptive_associations": contrasts,
    }


def _water_ratio(inputs):
    return inputs["water"] / (inputs["cement"] + inputs["slag"] + inputs["fly_ash"])


def knowledge_context(observed, menu):
    """Compute a descriptive 2x2 contrast with observed measurements exclusively."""
    values = list(observed.values())
    water_cut = median(map(_water_ratio, values))
    plasticizer_cut = median(row["superplasticizer"] for row in values)
    groups = {}
    for lower_ratio in (True, False):
        for lower_plasticizer in (True, False):
            strength = [
                row["strength"] for row in values
                if (_water_ratio(row) <= water_cut) == lower_ratio
                and (row["superplasticizer"] <= plasticizer_cut) == lower_plasticizer
            ]
            label = ("lower_ratio" if lower_ratio else "higher_ratio") + "_" + ("lower_superplasticizer" if lower_plasticizer else "higher_superplasticizer")
            groups[label] = [len(strength), round(mean(strength), 2) if strength else None]
    return {
        **KNOWLEDGE,
        "observed_ratio_and_superplasticizer_split": {
            "ratio_cut": round(water_cut, 4), "superplasticizer_cut": plasticizer_cut,
            "columns": ["n", "mean_strength_mpa"], "groups": groups,
            "interpretation": "All groups use only currently revealed measurements. Descriptive, confounded comparison with potentially small cells.",
        },
        "menu_water_to_total_binder": {row["alias"]: round(_water_ratio(row["inputs"]), 4) for row in menu},
    }


def hypothesis_and_design_input(analysis, menu, k, shortlist_n, round_number, knowledge):
    columns = ["alias", "model_rank", "model_prediction_mpa", "rf_tree_sd_mpa", *FEATURES]
    rows = [
        [row["alias"], row["rank"], round(row["prediction"], 3),
         round(row["rf_tree_disagreement"], 3) if row["rf_tree_disagreement"] is not None else None,
         *[row["inputs"][col] for col in FEATURES]]
        for row in menu
    ]
    payload = {"round": round_number, "budget": k, "observation_summary": analysis, "menu_columns": columns, "menu": rows}
    if knowledge is not None:
        payload["knowledge"] = knowledge
    instructions = (
        "Act as Data Analyst, Hypothesis Generator and Recipe Designer. Write one short analysis sentence and one short, testable hypothesis. "
        "Cite 1 to 3 IDs from the observed strongest/weakest examples. "
        f"Then call evaluate_concrete_shortlist with {shortlist_n} distinct candidate aliases like C01 to C48. "
        "Do not output any content outside the single native tool call. Ingredients are kg/m3 and age is days. "
        "RF tree SD measures model disagreement and is not a calibrated confidence interval. "
        "After receiving your tool output, act as Scientific Critic. Explain one plausible limitation or combination risk and why your final choices merit testing. "
        "Return one JSON object only, without markdown: {\"candidate_ids\":[...],\"critic\":\"...\",\"selection_rationale\":\"...\"}. "
        f"Then use exactly {k} distinct aliases from the evaluator output and order by test priority. Keep both text fields to one short sentence."
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": instructions + "\n" + _json(payload)}]


def scientific_critic(candidate_ids, available, *, count):
    if not isinstance(candidate_ids, list) or any(not isinstance(key, str) for key in candidate_ids):
        raise AgentError("Recipe aliases must be a list of strings")
    if len(candidate_ids) != count or len(set(candidate_ids)) != count or not set(candidate_ids) <= available:
        raise AgentError(f"Expected {count} unique eligible aliases from the permitted universe")


def fitness_evaluator(candidate_ids, menu, observed):
    """Resolve exact live RF scores and measure input novelty without oracle access."""
    menu_by_alias = {row["alias"]: row for row in menu}
    result = []
    for alias in candidate_ids:
        entry = menu_by_alias[alias]
        inputs = entry["inputs"]
        distances = [
            (sum(inputs[col] != seen[col] for col in FEATURES[:-1]), key)
            for key, seen in observed.items()
        ]
        changes, nearest = min(distances)
        result.append({
            "alias": alias, "eligible": True,
            "prediction_mpa": round(entry["prediction"], 3), "prediction_rank": entry["rank"],
            "rf_tree_disagreement_mpa": round(entry["rf_tree_disagreement"], 3) if entry["rf_tree_disagreement"] is not None else None,
            "nearest_observed_example": nearest, "changed_ingredients_vs_nearest": changes,
            "recipe": inputs,
        })
    return result


def _endpoint():
    endpoint = os.environ.get("CODEFLOW_BASE_URL", DEFAULT_BASE).rstrip("/")
    if endpoint.endswith("/chat/completions"):
        endpoint = endpoint.removesuffix("/chat/completions")
    parsed = urlsplit(endpoint)
    if parsed.scheme != "https" or parsed.hostname != "codeflow.sh" or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise AgentError("The experiment needs an HTTPS CodeFlow API base URL without credentials or query text")
    return endpoint


@lru_cache(maxsize=1)
def _client(endpoint):
    from openai import OpenAI

    secret = os.environ.get("CODEFLOW_API_KEY")
    if not secret:
        raise AgentError("Set CODEFLOW_API_KEY for the live CodeFlow experiment")
    return OpenAI(base_url=endpoint, api_key=secret, max_retries=0, timeout=90.0)


def _generate(endpoint, request):
    from openai import APIError, APIStatusError

    request = dict(request)
    thinking = request.pop("thinking")
    try:
        completion = _client(endpoint).chat.completions.create(**request, extra_body={"thinking": thinking})
    except APIStatusError as exc:
        error_code = getattr(exc, "code", None)
        code = f" ({error_code})" if isinstance(error_code, str) and error_code in {
            "invalid_api_key", "model_not_found", "invalid_request_error", "rate_limit_exceeded",
            "insufficient_quota", "server_error", "permission_denied",
        } else ""
        raise AgentError(f"CodeFlow HTTP status {exc.status_code}{code}") from None
    except APIError as exc:
        raise AgentError(f"CodeFlow connection failed ({type(exc).__name__})") from None
    choice = completion.choices[0] if len(completion.choices) == 1 else None
    usage = completion.usage
    if choice is None or usage is None or not completion.model or not completion.id:
        raise AgentError("CodeFlow response omitted one choice, usage, response ID or the actual model")
    return {
        "text": choice.message.content or "", "tool_calls": [item.model_dump() for item in choice.message.tool_calls or []],
        "finish_reason": choice.finish_reason, "model": completion.model, "response_id": completion.id,
        "request_id": getattr(completion, "_request_id", None), "created": completion.created,
        "usage": {"input_tokens": usage.prompt_tokens, "output_tokens": usage.completion_tokens},
        "provider_usage": usage.model_dump(), "runtime": {"openai": version("openai")},
    }


def _parse_object(text):
    try:
        candidate = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise AgentError("Malformed model JSON") from exc
    if not isinstance(candidate, dict):
        raise AgentError("Expected a JSON object from the model")
    return candidate


def _check_fields(obj, keys, text_fields):
    if set(obj) != set(keys):
        raise AgentError("Unexpected or missing structured output fields")
    for key in text_fields:
        if not isinstance(obj[key], str) or not 1 <= len(obj[key].strip()) <= 600:
            raise AgentError(f"Expected concise nonempty text in {key}")


def _proposal(response, available, examples, shortlist_n):
    tools = response.get("tool_calls", [])
    if len(tools) != 1 or tools[0].get("type") != "function" or not tools[0].get("id"):
        raise AgentError("Model did not emit exactly one native function tool call")
    tool_call = tools[0]["function"]
    if tool_call["name"] != "evaluate_concrete_shortlist":
        raise AgentError("Model did not call the authorized concrete evaluator")
    draft = _parse_object(tool_call["arguments"])
    _check_fields(draft, PROPOSAL_SCHEMA["required"], ("analyst", "hypothesis"))
    scientific_critic(draft["candidate_ids"], available, count=shortlist_n)
    evidence = draft["evidence_ids"]
    if not isinstance(evidence, list) or not 1 <= len(evidence) <= 3:
        raise AgentError("Proposal must link to one to three observed examples")
    scientific_critic(evidence, examples, count=len(evidence))
    return draft


def _parse_final(text, available, k):
    final = _parse_object(text)
    _check_fields(final, ("candidate_ids", "critic", "selection_rationale"), ("critic", "selection_rationale"))
    scientific_critic(final["candidate_ids"], available, count=k)
    return final


def choose_candidates(
    *, eligible: Mapping[str, dict], observed: Mapping[str, dict], scores: Mapping[str, float],
    k: int, seed: int, round_number: int, augmented: bool, log_path: Path,
    uncertainty: Mapping[str, float] | None = None, model: str | None = None,
    replay_path: Path | None = None, transport=None,
) -> list[str]:
    """Return validated IDs chosen by a real LLM, or replay and verify a recorded decision."""
    _check_inputs(eligible, observed, scores, uncertainty, k)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        raise FileExistsError(log_path)
    previous = json.loads(Path(replay_path).read_text()) if replay_path is not None else None
    if previous is not None and (previous.get("status") != "ok" or previous.get("mode") != "live"):
        raise AgentError("Replay requires a successful original live trace")
    default_model = os.environ.get("CODEFLOW_MODEL", MODEL) if transport is None else transport.model
    requested_model = model or (previous["model"] if previous is not None else default_model)
    if previous is None:
        endpoint = _endpoint() if transport is None else transport.endpoint
    else:
        endpoint = previous["config"]["api_base"]
    model_config = MODEL_CONFIG if transport is None else transport.config
    config = {**model_config, "api_base": endpoint, "model": requested_model}
    menu = _ordered_menu(eligible, scores, uncertainty, seed, round_number)
    aliases = {row["alias"]: row["id"] for row in menu}
    analysis = data_analyst(observed)
    examples = analysis["strongest"].keys() | analysis["weakest"].keys()
    knowledge = knowledge_context(observed, menu) if augmented else None
    log = {
        "format_version": 2, "model": requested_model, "config": config,
        "mode": "live" if previous is None else "replay", "seed": seed, "round": round_number,
        "knowledge_augmented": augmented, "budget": k, "menu": menu,
        "calls": [], "tool_calls": [{"tool": "data_analyst", "trigger": "pipeline", "result": analysis}],
        "usage": {"input_tokens": 0, "output_tokens": 0}, "live_generation_count": 0,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    started = time.monotonic()

    def complete(stage, messages, parser, native_continuation=None):
        for attempt in (1, 2, 3):
            request = {
                "model": requested_model, "messages": messages, "tools": TOOLS,
                "temperature": MODEL_CONFIG["temperature"], "max_tokens": MAX_TOKENS,
                "thinking": MODEL_CONFIG["thinking"],
            }
            if stage == "proposal":
                request["tool_choice"] = {"type": "function", "function": {"name": "evaluate_concrete_shortlist"}}
            else:
                request["tool_choice"] = "none"
                request["response_format"] = {"type": "json_object"}
            if transport is not None:
                request = transport.encode(request, native_continuation=native_continuation)
            key = _digest(request)
            call = {"stage": stage, "attempt": attempt, "request_sha256": key, "request": request}
            log["calls"].append(call)
            mark = time.monotonic()
            if previous is None:
                response = _generate(endpoint, request) if transport is None else transport.generate(endpoint, request)
                log["live_generation_count"] += 1
            else:
                index = len(log["calls"]) - 1
                if index >= len(previous["calls"]):
                    raise AgentError("Replay is missing a required request")
                original = previous["calls"][index]
                if original.get("request_sha256") != key or _digest(original.get("request")) != key:
                    raise AgentError("Replay request does not match current evidence and configuration")
                response = original["response"]
                if original.get("response_sha256") != _digest(response):
                    raise AgentError("Replay response digest does not match its recorded generation")
            call.update(response=response, response_sha256=_digest(response), elapsed_s=round(time.monotonic() - mark, 3))
            for token_type in log["usage"]:
                log["usage"][token_type] += response["usage"][token_type]
            try:
                target_finish = "tool_calls" if stage == "proposal" else "stop"
                if response["finish_reason"] != target_finish:
                    raise AgentError(f"Model response did not complete: {response['finish_reason']}")
                call["parsed"] = parser(response)
                return call["parsed"], response
            except AgentError as exc:
                call["parse_error"] = str(exc)
                if attempt == 3:
                    raise
                messages = [*messages, {"role": "user", "content": f"Repair your previous response. Validation error: {exc}. Follow the original instructions exactly."}]
        raise AssertionError("unreachable")

    try:
        shortlist_n = min(k + 2, len(menu))
        messages = hypothesis_and_design_input(analysis, menu, k, shortlist_n, round_number, knowledge)
        draft, proposal = complete("proposal", messages, lambda reply: _proposal(reply, set(aliases), examples, shortlist_n))
        evaluations = fitness_evaluator(draft["candidate_ids"], menu, observed)
        tool_call = proposal["tool_calls"][0]
        tool_id = tool_call["id"]
        log["tool_calls"].append({"tool": "evaluate_concrete_shortlist", "trigger": "model_tool_call", "call_id": tool_id, "arguments": draft, "result": evaluations})
        final_messages = [
            *messages,
            {"role": "assistant", "content": proposal["text"] or None, "tool_calls": [tool_call]},
            {"role": "tool", "tool_call_id": tool_id, "content": _json(evaluations)},
        ]
        final, _ = complete(
            "final_selection", final_messages, lambda reply: _parse_final(reply["text"], set(draft["candidate_ids"]), k),
            native_continuation=proposal.get("native_content"),
        )
        resolved = [aliases[key] for key in final["candidate_ids"]]
        scientific_critic(resolved, set(eligible), count=k)
        log["tool_calls"].append({"tool": "scientific_critic", "trigger": "pipeline", "result": {"eligible_unique_budget_verified": True, "evidence_ids": draft["evidence_ids"]}})
        log["decision"] = {**draft, **final, "selected_ids": resolved}
        if previous is not None and (len(log["calls"]) != len(previous["calls"]) or previous["decision"] != log["decision"]):
            raise AgentError("Replay does not reproduce the recorded final decision")
        log["status"] = "ok"
        return resolved
    except (AgentError, OSError, ImportError, RuntimeError) as exc:
        log.update(status="error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        log["elapsed_s"] = round(time.monotonic() - started, 3)
        with log_path.open("x") as stream:
            json.dump(log, stream, indent=2, ensure_ascii=True, allow_nan=False)
            stream.write("\n")
