"""Native-tool GFP fluorescence decision policy with verifiable archived replay."""
from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from gfp_data import GFP_WT, MAX_MUTATIONS, make_variant
from gpt_transport import GPTTransport
from llm_agent import (
    ARRAY_STR,
    MODEL,
    MODEL_CONFIG,
    STRING,
    AgentError,
    _check_fields,
    _digest,
    _endpoint,
    _generate,
    _json,
    _number,
    _parse_final,
    _parse_object,
    scientific_critic,
)
from qwen_transport import QwenTransport
from sonnet_transport import SonnetTransport


TOOL = "score_gfp_designs"
SCHEMA = {
    "type": "object",
    "properties": {
        "analyst": STRING,
        "hypothesis": STRING,
        "evidence_ids": ARRAY_STR,
        "designs": ARRAY_STR,
    },
    "required": ["analyst", "hypothesis", "evidence_ids", "designs"],
    "additionalProperties": False,
}
TOOLS = [{
    "type": "function",
    "function": {
        "name": TOOL,
        "description": (
            "Construct and score candidate GFP fluorescence sequences. Each design string is "
            "a menu alias followed by one space and its explicit colon-separated substitutions."
        ),
        "parameters": SCHEMA,
    },
}]
SYSTEM = (
    "You are analyzing historical measurements for avGFP fluorescence. Larger log brightness is better. "
    "Only observations have measured fluorescence. Menu values and tool scores are fitted predictions. "
    "Use the observed evidence, state a testable short hypothesis, and specify edits that match displayed "
    "candidate identities on the pinned GFP reference. All eligible candidates are from a fixed menu. "
    "Do not estimate or cite measurements for candidates. Keep all text fields concise and evidence-specific."
)
AA_CLASSES = {
    "aliphatic nonpolar": "AVILM",
    "aromatic": "FWY",
    "polar uncharged": "STNQ",
    "acidic": "DE",
    "basic or protonation dependent": "KRH",
    "special": "CGP",
}
AA_GROUP = {aa: group for group, letters in AA_CLASSES.items() for aa in letters}
FINAL_TOKEN_CAP = 280
PROPOSAL_TOKEN_CAP = 440
REMOTE_FINAL_TOKEN_CAP = 840
DEEPSEEK_PROPOSAL_TOKEN_CAP = 720
RATE_LIMIT_BACKOFF_S = (15, 35)
BACKENDS = {"deepseek", "sonnet", "gpt", "qwen"}


class DeepSeekGFP:
    model = MODEL
    config = {**MODEL_CONFIG, "scope": "GFP fluorescence"}

    @property
    def endpoint(self):
        return _endpoint()

    def encode(self, request, *, native_continuation):
        return request

    def generate(self, endpoint, request):
        return _generate(endpoint, request)


class GPTGFP(GPTTransport):
    def encode(self, request, *, native_continuation):
        encoded = super().encode(request, native_continuation=native_continuation)
        if "text" in encoded:
            encoded["text"]["format"]["name"] = "gfp_final"
        return encoded


class QwenGFP(QwenTransport):
    def __init__(self):
        super().__init__(explicit_contract=False)
        self.config["structural_contract"] = "gfp-explicit-edits-v1"

    def encode(self, request, *, native_continuation):
        encoded = super().encode(request, native_continuation=native_continuation)
        if request["tool_choice"] == "none":
            tool = next(item for item in request["messages"] if item["role"] == "tool")
            aliases = [row["alias"] for row in json.loads(tool["content"])]
            budget = _payload(request)["budget"]
            encoded["structural_contract"] = (
                f"Return JSON only. Fields: candidate_ids, critic, selection_rationale. "
                f"candidate_ids must have exactly {budget} distinct aliases from {', '.join(aliases)}. "
                "Do not append a tool call or Markdown."
            )
            return encoded
        payload = _payload(request)
        known = [row[0] for row in payload["observations"]]
        format_instruction = "one C alias, a space and the exact substitutions from its menu row"
        if "pending_ids" in payload:
            format_instruction += " or NEW followed by legal off-menu edits, including at least one NEW design"
        encoded["assistant_prefill"] = f'<tool_call>\n{{"name":"{TOOL}","arguments":'
        encoded["structural_contract"] = (
            "Write the four argument fields analyst, hypothesis, evidence_ids, designs. "
            f"evidence_ids must be one to three distinct O aliases from {', '.join(known)}. "
            f"designs must be exactly {payload['shortlist_size']} distinct strings, each consisting of "
            f"{format_instruction}. Do not put a C alias in evidence_ids. "
            "Finish the native tool call with </tool_call>."
        )
        return encoded


def _payload(request):
    content = next(item["content"] for item in request["messages"] if item["role"] == "user")
    return json.loads(content.split("\n", 1)[1])


def _transport(backend):
    transports = {"deepseek": DeepSeekGFP, "sonnet": SonnetTransport, "gpt": GPTGFP, "qwen": QwenGFP}
    try:
        return transports[backend]()
    except KeyError as exc:
        raise ValueError(f"Unrecognized GFP backend {backend!r}") from exc


def _validate_record(row, *, observed, limit):
    fields = {"id", "mutations", "sequence", "edits"}
    required = fields | ({"fitness"} if observed else {"prediction", "uncertainty", "rank", "source", "alias"})
    if not isinstance(row, dict) or set(row) != required:
        raise AgentError("A GFP record has unexpected fields")
    variant = make_variant(row["mutations"], max_mutations=limit)
    if row["sequence"] != variant.sequence or tuple(row["edits"]) != variant.edits:
        raise AgentError("GFP sequence, canonical mutation text and edit tokens must all match")
    if not isinstance(row["id"], str) or not row["id"].strip():
        raise AgentError("Missing GFP record identifier")
    if observed:
        if not _number(row["fitness"]):
            raise AgentError("Observed GFP fitness must be finite")
    elif (not _number(row["prediction"]) or not _number(row["uncertainty"])
          or row["uncertainty"] < 0
          or not re.fullmatch(r"C[0-9]{2}", row["alias"])):
        raise AgentError("Invalid GFP candidate alias or estimated score")


def _prepare(request):
    budget = request["budget"]
    limit = request["max_mutations"]
    if (request.get("domain") != "gfp_fluorescence" or request["reference_sequence"] != GFP_WT
            or request.get("fitness_direction") != "maximize"
            or isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_MUTATIONS):
        raise AgentError("This agent only accepts the pinned Sarkisyan avGFP reference and fluorescence objective")
    if isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= 5:
        raise AgentError("GFP budget must be an integer from 1 through 5")
    if not isinstance(request["knowledge"], bool):
        raise AgentError("The GFP knowledge flag must be Boolean")
    observed, menu = request["observed"], request["menu"]
    allows_novel = "pending_ids" in request
    if not observed or len(menu) > 48 or (not allows_novel and len(menu) < budget):
        raise AgentError("Insufficient GFP observation or menu data")
    for row in observed:
        _validate_record(row, observed=True, limit=limit)
    for row in menu:
        _validate_record(row, observed=False, limit=limit)
    ids = [row["id"] for row in observed + menu]
    sequences = {row["sequence"] for row in observed + menu}
    if len(ids) != len(set(ids)) or len(ids) != len(sequences):
        raise AgentError("Repeated or previously observed GFP variant in candidate menu")
    by_alias = {row["alias"]: row for row in menu}
    if len(by_alias) != len(menu):
        raise AgentError("Duplicate menu aliases")
    observed_by_id = {row["id"]: row for row in observed}
    latest_ids = request.get("latest_feedback_ids", [])
    if (len(latest_ids) > 5 or len(latest_ids) != len(set(latest_ids))
            or not set(latest_ids) <= observed_by_id.keys()):
        raise AgentError("Latest GFP feedback must be a subset of observed measurements")
    ranked = sorted(observed, key=lambda row: (row["fitness"], row["id"]))
    display_ids = dict.fromkeys([*(row["id"] for row in ranked[-3:][::-1]),
                                  *(row["id"] for row in ranked[:2]), *latest_ids])
    evidence = {f"O{i}": observed_by_id[key] for i, key in enumerate(display_ids, 1)}
    return by_alias, evidence, budget + 1 if allows_novel else min(budget + 1, len(menu))


def _knowledge(observed, menu):
    requested_edits = {edit for row in menu.values() for edit in row["edits"]}
    single_scores, combination_counts = {}, {}
    for row in observed:
        edits = row["edits"]
        for edit in requested_edits.intersection(edits):
            if len(edits) == 1:
                single_scores[edit] = round(row["fitness"], 4)
            else:
                combination_counts[edit] = combination_counts.get(edit, 0) + 1
    conservative = sorted(
        edit for edit in requested_edits
        if AA_GROUP.get(edit[0]) == AA_GROUP.get(edit[-1]) != "special"
    )
    return {
        "amino_acid_classes": AA_CLASSES,
        "coarse_conservative_edits_in_menu": conservative,
        "observed_single_scores_matching_menu": single_scores,
        "number_of_observed_combinations_sharing_each_listed_edit": combination_counts,
        "interpretation": (
            "Conservative means the same coarse nonspecial class, not a fitness prediction. "
            "Histidine charge is pH-dependent. Scores for multi-edit "
            "GFP are context dependent. A count of overlapping edits does not give a causal effect."
        ),
    }


def _messages(request, menu, evidence, shortlist_size, *, allow_novel=False):
    payload = {
        "budget": request["budget"],
        "shortlist_size": shortlist_size,
        "round": request["round_number"],
        "mutation_limit": request["max_mutations"],
        "observations": [
            [alias, row["mutations"], round(row["fitness"], 4)] for alias, row in evidence.items()
        ],
        "latest_feedback_aliases": [
            alias for alias, row in evidence.items() if row["id"] in request.get("latest_feedback_ids", [])
        ],
        "menu": [[alias, row["mutations"], round(row["prediction"], 3)] for alias, row in menu.items()],
    }
    if request["knowledge"]:
        payload["gfp_knowledge"] = _knowledge(request["observed"], menu)
    instructions = (
        "Each O row is alias, edits, measured value. Each C row is alias, edits, rounded model estimate. "
        f"Call {TOOL} with a short analyst summary, a concrete hypothesis, 1–3 O evidence IDs "
        f"and exactly {shortlist_size} designs, e.g. a C alias and a space followed by its exact "
        "colon-delimited edits from the menu. The returned estimates resolve exact values and tool checks."
    )
    system = SYSTEM
    if allow_novel:
        system = SYSTEM.replace(
            "All eligible candidates are from a fixed menu.",
            "The reference menu is optional. The scoring tool can construct and score legal off-menu edits."
        )
        instructions = (
            f"Call {TOOL} immediately with a short analyst summary, a short hypothesis, 1–3 observed O aliases "
            f"and exactly {shortlist_size} designs. Write each design as a C alias followed by its exact edits, "
            "or NEW followed by a space and 1–4 colon-separated substitutions. Include at least one NEW design. "
            "Exclude observed IDs and pending_ids from NEW. The tool assigns N aliases to valid off-menu edits. "
            "Keep the analyst and hypothesis under 15 words each. "
            f"GFP reference: {GFP_WT}."
        )
        payload["pending_ids"] = request["pending_ids"]
    return [{"role": "system", "content": system},
            {"role": "user", "content": instructions + "\n" + _json(payload)}]


def _proposal(response, menu, evidence, count, limit, *, excluded=None):
    calls = response.get("tool_calls", [])
    if (len(calls) != 1 or calls[0].get("type") != "function" or not calls[0].get("id")
            or calls[0]["function"]["name"] != TOOL):
        raise AgentError("Expected one native score_gfp_designs call")
    draft = _parse_object(calls[0]["function"]["arguments"])
    _check_fields(draft, SCHEMA["required"], ("analyst", "hypothesis"))
    references = draft["evidence_ids"]
    if not isinstance(references, list) or not 1 <= len(references) <= 3:
        raise AgentError("Cite one to three measured O aliases")
    scientific_critic(references, set(evidence), count=len(references))
    designs = draft["designs"]
    if not isinstance(designs, list) or len(designs) != count:
        raise AgentError(f"Expected exactly {count} design strings")
    aliases = []
    novel = {}
    novel_ids = set()
    for record in designs:
        if excluded is not None and isinstance(record, str) and (match := re.fullmatch(r"NEW +([^\s]+)", record.strip())):
            try:
                variant = make_variant(match[1], max_mutations=limit)
            except ValueError as exc:
                raise AgentError("A NEW design must use legal substitutions on the pinned GFP sequence") from exc
            if not variant.edits or variant.id in excluded or variant.id in novel_ids:
                raise AgentError("A NEW design must be distinct and outside the menu, observations and pending batches")
            alias = f"N{len(novel) + 1:02d}"
            novel[alias] = variant
            novel_ids.add(variant.id)
            aliases.append(alias)
            continue
        matched = re.fullmatch(r"(C[0-9]{2})(?: +|: *)([^\s]+)", record.strip()) if isinstance(record, str) else None
        if matched is None:
            if excluded is not None:
                raise AgentError("Each design needs a C alias and exact edits or NEW followed by off-menu edits")
            raise AgentError("Each design needs a C alias and the exact edit string, separated by a space or colon")
        alias, mutations = matched.groups()
        if alias not in menu:
            raise AgentError("Design used an alias outside the GFP menu")
        try:
            variant = make_variant(mutations, max_mutations=limit)
        except ValueError as exc:
            raise AgentError("Malformed GFP edits in a design") from exc
        if variant.sequence != menu[alias]["sequence"] or variant.edits != tuple(menu[alias]["edits"]):
            raise AgentError(f"Explicit edits do not reconstruct the displayed candidate {alias}")
        aliases.append(alias)
    if excluded is not None and not novel:
        raise AgentError("Include at least one validated NEW design in the shortlist")
    scientific_critic(aliases, menu.keys() | novel.keys(), count=count)
    return draft, aliases, novel


def _parse_gfp_final(text, choices, budget):
    if isinstance(text, str):
        wrapped = re.fullmatch(r"\s*```(?:json)?\s*(\{.*\})\s*```\s*", text, re.DOTALL | re.IGNORECASE)
        if wrapped:
            text = wrapped[1]
    return _parse_final(text, choices, budget)


def select_candidates(request: dict, *, log_path: Path, backend="deepseek", replay_path=None,
                      transport=None, score_novel=None) -> list[str]:
    """Use native tool calls or verify an archived GFP decision without provider use."""
    menu, evidence, shortlist_size = _prepare(request)
    allows_novel = score_novel is not None
    if allows_novel != ("pending_ids" in request) or (allows_novel and replay_path is not None):
        raise AgentError("Off-menu scoring requires a user research request and a fitted predictor")
    tools = TOOLS
    if allows_novel:
        function = {**TOOLS[0]["function"], "description": (
            "Build and predict GFP variants from C aliases and their exact edits, or NEW followed by "
            "off-menu colon-separated edits. The tool assigns N aliases to accepted new designs."
        )}
        tools = [{"type": "function", "function": function}]
    adapter = _transport(backend) if transport is None else transport
    log_path = Path(log_path)
    if log_path.exists():
        raise FileExistsError(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    archive = json.loads(Path(replay_path).read_text()) if replay_path is not None else None
    if archive is not None and (
        archive.get("mode") != "live" or archive.get("status") != "ok"
        or archive.get("backend") != backend or archive.get("input_sha256") != _digest(request)
    ):
        raise AgentError("Archived GFP decision does not match backend or frozen inputs")
    endpoint = adapter.endpoint if archive is None else archive["config"]["api_base"]
    proposal_cap = (
        1024 if transport is not None else DEEPSEEK_PROPOSAL_TOKEN_CAP if backend == "deepseek" else PROPOSAL_TOKEN_CAP
    )
    final_cap = 840 if transport is not None or backend in {"deepseek", "sonnet"} else FINAL_TOKEN_CAP
    if allows_novel:
        proposal_cap, final_cap = (768, 440) if backend == "qwen" else (2304, 1400)
    log = {
        "format_version": 1, "mode": "replay" if archive else "live", "backend": backend,
        "model": adapter.model, "domain": "gfp_fluorescence",
        "config": {**adapter.config, "api_base": endpoint, "proposal_max_tokens": proposal_cap,
                   "final_max_tokens": final_cap, "rate_limit_backoff_s": RATE_LIMIT_BACKOFF_S},
        "seed": request["seed"], "round": request["round_number"], "budget": request["budget"],
        "knowledge_augmented": request["knowledge"], "input_sha256": _digest(request),
        "reference_sha256": hashlib.sha256(GFP_WT.encode("ascii")).hexdigest(),
        "menu": request["menu"], "observed_evidence": evidence,
        "latest_feedback_ids": request.get("latest_feedback_ids", []),
        "calls": [], "tool_calls": [], "usage": {"input_tokens": 0, "output_tokens": 0},
        "live_generation_count": 0, "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    started = time.monotonic()

    def complete(stage, base_messages, parser, native=None):
        messages = base_messages
        for attempt in range(1, 4):
            final = stage == "final_selection"
            native_choice = "none" if final else {"type": "function", "function": {"name": TOOL}}
            canonical = {
                "model": adapter.model, "messages": messages, "tools": tools, "tool_choice": native_choice,
                "temperature": 0.0, "max_tokens": final_cap if final else proposal_cap,
                "thinking": {"type": "disabled"},
            }
            if final:
                canonical["response_format"] = {"type": "json_object"}
            wire = adapter.encode(canonical, native_continuation=native)
            call = {"stage": stage, "attempt": attempt, "request": wire, "request_sha256": _digest(wire)}
            log["calls"].append(call)
            mark = time.monotonic()
            if archive is not None:
                offset = len(log["calls"]) - 1
                if offset >= len(archive["calls"]):
                    raise AgentError("Archived GFP calls are incomplete")
                source = archive["calls"][offset]
                if source.get("request_sha256") != call["request_sha256"] or _digest(source["request"]) != call["request_sha256"]:
                    raise AgentError("Archived GFP provider request failed digest verification")
                if "transport_error" in source:
                    call["transport_error"] = source["transport_error"]
                    if "retry_after_s" in source and attempt < 3:
                        call["retry_after_s"] = source["retry_after_s"]
                        continue
                    raise AgentError(call["transport_error"])
                response = source["response"]
                if source.get("response_sha256") != _digest(response):
                    raise AgentError("Archived GFP provider response failed digest verification")
            else:
                try:
                    response = adapter.generate(endpoint, wire)
                except AgentError as exc:
                    call["transport_error"] = str(exc)
                    matched = re.search(r"status (429|500|502|503|504)\b", str(exc))
                    if attempt < 3 and matched:
                        delay = RATE_LIMIT_BACKOFF_S[attempt - 1] if matched[1] == "429" else attempt * 3
                        call["retry_after_s"] = delay
                        time.sleep(delay)
                        continue
                    raise
                log["live_generation_count"] += 1
            call.update(response=response, response_sha256=_digest(response),
                        elapsed_s=round(time.monotonic() - mark, 3))
            for field in log["usage"]:
                log["usage"][field] += response["usage"][field]
            try:
                completion = "stop" if final else "tool_calls"
                if response["finish_reason"] != completion:
                    raise AgentError(f"Expected provider completion {completion}, saw {response['finish_reason']}")
                value = parser(response)
                call["validated"] = True
                return value, response
            except AgentError as exc:
                call["parse_error"] = str(exc)
                if attempt == 3:
                    raise
                messages = base_messages + [{"role": "user", "content": f"Repair your response. {exc}."}]
        raise AssertionError("unreachable")

    try:
        messages = _messages(request, menu, evidence, shortlist_size, allow_novel=allows_novel)
        excluded = None
        if allows_novel:
            excluded = {row["id"] for row in request["observed"] + request["menu"]} | set(request["pending_ids"])
        (draft, aliases, novel), native = complete("proposal", messages, lambda response: _proposal(
            response, menu, evidence, shortlist_size, request["max_mutations"], excluded=excluded
        ))
        candidates = dict(menu)
        if novel:
            for (alias, variant), (score, spread) in zip(novel.items(), score_novel(list(novel.values())), strict=True):
                if not _number(score) or not _number(spread) or spread < 0:
                    raise AgentError("The fitted GFP predictor returned an invalid off-menu score")
                candidates[alias] = {"alias": alias, "id": variant.id, "mutations": variant.mutant,
                                     "sequence": variant.sequence, "edits": list(variant.edits), "prediction": score,
                                     "uncertainty": spread, "rank": None, "source": "llm_off_menu"}
        scored = [
            {"alias": key, "mutations": candidates[key]["mutations"], "prediction": candidates[key]["prediction"],
             "uncertainty": candidates[key]["uncertainty"], "roster_match": key in menu} for key in aliases
        ]
        tool_call = native["tool_calls"][0]
        log["tool_calls"].append({
            "tool": TOOL, "trigger": "model_tool_call", "call_id": tool_call["id"],
            "arguments": draft, "result": [{**row, "sequence": candidates[row["alias"]]["sequence"]} for row in scored],
        })
        ending = " Reply with bare JSON. Limit critic and selection_rationale to 25 words each." if transport is not None or backend in {"deepseek", "sonnet"} else ""
        final_messages = [
            *messages,
            {"role": "assistant", "content": native["text"] or None, "tool_calls": [tool_call]},
            {"role": "tool", "tool_call_id": tool_call["id"], "content": _json(scored)},
            {"role": "user", "content": (
                f"Return JSON with candidate_ids, critic and selection_rationale. Select exactly {request['budget']} "
                "evaluated aliases ordered by test priority. Give reasons that refer to the estimates and identify a "
                "weakness. New fluorescence measurements are unavailable until after submission." + ending
            )},
        ]
        final, _ = complete(
            "final_selection", final_messages,
            lambda response: _parse_gfp_final(response["text"], set(aliases), request["budget"]),
            native=native.get("native_content"),
        )
        selected = [candidates[alias] for alias in final["candidate_ids"]]
        resolved = [row["id"] for row in selected]
        log["tool_calls"].append({"tool": "gfp_critic", "result": {"legal": True, "unique": True, "n": len(selected)}})
        log["decision"] = {
            **draft, **final,
            "evidence_aliases": draft["evidence_ids"],
            "evidence_ids": [evidence[alias]["id"] for alias in draft["evidence_ids"]],
            "selected_ids": resolved, "selected_candidates": selected,
        }
        if archive is not None and (len(archive["calls"]) != len(log["calls"]) or archive["decision"] != log["decision"]):
            raise AgentError("Replay failed to reproduce the archived GFP decision")
        log["status"] = "ok"
        return resolved
    except (AgentError, OSError, ImportError, RuntimeError) as exc:
        log.update(status="error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        log["elapsed_s"] = round(time.monotonic() - started, 3)
        with log_path.open("x") as out:
            json.dump(log, out, indent=2, ensure_ascii=True, allow_nan=False)
            out.write("\n")
