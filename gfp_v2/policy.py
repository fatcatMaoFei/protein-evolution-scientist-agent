"""Native-tool decisions and preregistered fluorescence checks for GFP V2."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import time
from urllib.parse import urlsplit

from gfp_data import GFP_WT, make_variant
from llm_agent import AgentError


TOOL = "score_gfp_v2"
MAX_ATTEMPTS = 3
PROPOSAL_TOKENS = 3600
DECISION_TOKENS = 3600
RETRY_DELAYS = (4, 12)
STRATEGIES = ("high_prediction", "uncertainty_exploration", "substitution_diversity", "evidence_replication", "mixed")
ADAPTATIONS = ("no_new_feedback", "continue", "shift_to_predicted_gain", "shift_to_exploration", "avoid_recent_edit_pattern")
CHEMISTRY = ("AVILM", "FWY", "STNQ", "DE", "KRH", "CGP")
CHEMISTRY_NAMES = ("aliphatic", "aromatic", "polar uncharged", "acidic", "basic or pH-dependent", "special")
CHEMISTRY_GROUP = {aa: name for group, name in zip(CHEMISTRY, CHEMISTRY_NAMES, strict=True) for aa in group}
_STRINGS = {"type": "array", "items": {"type": "string"}}
TOOL_SPEC = {"type": "function", "function": {
    "name": TOOL,
    "description": "Validate canonical avGFP mutation IDs and retrieve exact Predictor V2 scores and uncertainty.",
    "parameters": {"type": "object", "properties": {"designs": _STRINGS}, "required": ["designs"],
                   "additionalProperties": False},
}}
SYSTEM = (
    "You prioritize fluorescence of the isolated fixed avGFP reporter. Use the measured observations, "
    "the declared predictor, and the candidate sequences. A prediction or uncertainty is not a measurement. "
    "A substitution pattern or amino-acid class is not proof of structure or biochemical mechanism. "
    "Make a prospective fluorescence comparison against measured WT or the current observed incumbent. "
    "Produce the requested structured output only. Higher measured scores mean brighter fluorescence."
)


def _json(value) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def digest(value) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _object(value, fields, context):
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError) as exc:
        raise AgentError(f"Invalid JSON in {context}") from exc
    if not isinstance(parsed, dict) or set(parsed) != set(fields):
        raise AgentError(f"Unexpected fields in {context}")
    return parsed


def _finite(value) -> bool:
    return type(value) in (float, int) and math.isfinite(value)


def _canonical(key):
    try:
        variant = make_variant(key)
    except (ValueError, TypeError) as exc:
        raise AgentError("Invalid substitutions against the pinned avGFP reference") from exc
    if variant.id != key:
        raise AgentError("Use the canonical GFP mutation ID, with edits ordered by position")
    return variant


def _request(request):
    budget = request["budget"]
    if type(budget) is not int or not 1 <= budget <= 8 or type(request["round"]) is not int or request["round"] < 1:
        raise AgentError("Invalid V2 round or measurement budget")
    measured = {}
    for row in request["observed"]:
        if set(row) != {"id", "fitness", "round"} or not _finite(row["fitness"]) or row["id"] in measured:
            raise AgentError("Duplicate or invalid observed GFP measurement")
        if type(row["round"]) is not int or not 0 <= row["round"] < request["round"]:
            raise AgentError("Observed GFP measurement must predate the decision round")
        _canonical(row["id"])
        measured[row["id"]] = row
    if "WT" not in measured or request["wild_type_fitness"] != measured["WT"]["fitness"]:
        raise AgentError("The WT comparator must have its measured value in the current observations")
    incumbent = min(measured, key=lambda key: (-measured[key]["fitness"], key))
    if request["current_best"] != measured[incumbent]["fitness"]:
        raise AgentError("The requested current best differs from the observed incumbent")
    recent = []
    for row in request.get("latest_feedback", []):
        if row.get("id") not in measured or row != measured[row["id"]] or row["round"] == 0 or row["id"] in recent:
            raise AgentError("Recent feedback must match a newly revealed observation")
        recent.append(row["id"])
    candidates = {}
    for row in request["candidates"]:
        if set(row) != {"id", "prediction", "risk"}:
            raise AgentError("Candidate rows must contain only IDs, predictions and risk")
        key = row["id"]
        if not _finite(row["prediction"]) or not _finite(row["risk"]) or row["risk"] < 0 or key in candidates or key in measured:
            raise AgentError("Duplicate, already measured or invalid predicted GFP candidate")
        _canonical(key)
        candidates[key] = row
    return measured, incumbent, recent, candidates


def _history_knowledge(measured, wild_type):
    scores = defaultdict(list)
    for key, row in measured.items():
        for edit in make_variant(key).edits:
            scores[int(edit[1:-1])].append(row["fitness"])
    return {
        "chemistry_classes": dict(zip(CHEMISTRY_NAMES, CHEMISTRY, strict=True)),
        "position_counts": [
            [site, len(values), sum(value > wild_type for value in values), round(sum(values) / len(values), 4)]
            for site, values in sorted(scores.items())
        ],
        "position_columns": ["substitution_position", "observed_count", "observed_above_WT", "mean_fitness"],
        "interpretation": "Marginal patterns pool combinations and backgrounds. They do not establish effects or mechanisms.",
    }


def _proposal_message(request, candidates, open_world, count, knowledge, aliases=None):
    measured = request["observed"]
    payload = {
        "observed_columns": ["id", "measured_fitness", "observed_round"],
        "observed": [[row["id"], row["fitness"], row["round"]] for row in measured],
        "candidate_columns": ["id", "prediction", "risk"],
        "all_current_candidates": [[key, round(row["prediction"], 5), round(row["risk"], 5)] for key, row in candidates.items()],
        "risk_note": request.get("risk_note", "A relative predictive-risk score. It is not a calibrated interval."),
        "new_feedback_ids": [row["id"] for row in request.get("latest_feedback", [])],
        "prior_decisions": request.get("prior_decisions", []),
        "budget": request["budget"],
    }
    if knowledge:
        payload["observed_knowledge"] = _history_knowledge({row["id"]: row for row in measured}, request["wild_type_fitness"])
    instructions = f"Call {TOOL} with exactly {count} distinct candidate IDs. "
    if open_world:
        payload["avGFP_reference"] = GFP_WT
        payload["pending_ids"] = request.get("pending_ids", [])
        instructions += ("Include at least one newly designed one-to-four-substitution ID outside the displayed pool, "
                         "observations and pending IDs. Never pick an observed or pending ID. Use exact wild-type "
                         "residues and positions from the avGFP reference. After making a new design, fill remaining "
                         "slots from other legal designs or from all_current_candidates. ")
        if payload["pending_ids"] and len(candidates) >= count - 1:
            instructions += (f"For continuation, construct the call in two parts: first copy {count - 1} distinct IDs "
                             "from all_current_candidates, then add one legal NEW genotype. Earlier selected IDs and "
                             "all observed IDs provide evidence only and cannot be reused in this call. ")
    else:
        instructions += "Use the complete candidate pool below. The closed archive experiment cannot evaluate other designs. "
    instructions += "Consider prediction, risk, edit diversity, and relevant observed results.\n"
    if aliases:
        payload["candidate_columns"] = ["tool_alias", "id", "prediction", "risk"]
        payload["all_current_candidates"] = [[alias, key, round(candidates[key]["prediction"], 5),
                                               round(candidates[key]["risk"], 5)] for alias, key in aliases.items()]
        instructions = (f"Call {TOOL} with exactly {count} distinct entries using aliases (for example C001) from "
                        "all_current_candidates. Use the alias rather than retyping a genotype. The observed rows "
                        "and previous decisions give evidence, not available choices. ")
        if open_world:
            instructions += (f"Use {count - 1} shown aliases and add one NEW design as the string NEW followed by a "
                             "space and one-to-four colon-separated legal substitutions against avGFP_reference. "
                             "The new design must be different from all observed, pending and shown IDs. ")
        instructions += "Consider prediction, risk, edit diversity, and relevant observed results.\n"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": instructions + _json(payload)}]


def _scored_tool(response, candidates, measured, pending, score_new, count, knowledge, aliases=None):
    calls = response.get("tool_calls")
    if response.get("finish_reason") != "tool_calls" or not isinstance(calls, list) or len(calls) != 1:
        raise AgentError("The proposal must use exactly one native scoring tool call")
    call = calls[0]
    function = call.get("function", {})
    if call.get("type") != "function" or function.get("name") != TOOL or not isinstance(call.get("id"), str):
        raise AgentError("Unexpected native scoring tool call")
    designs = _object(function.get("arguments"), {"designs"}, "scoring tool")["designs"]
    if not isinstance(designs, list) or len(designs) != count or not all(isinstance(x, str) for x in designs):
        raise AgentError(f"Use exactly {count} canonical IDs in the scoring tool")
    if len(designs) != len(set(designs)):
        raise AgentError("Do not repeat candidate IDs")
    if aliases:
        translated = []
        for key in designs:
            if key in aliases:
                translated.append(aliases[key])
            elif score_new is not None and key.startswith("NEW "):
                translated.append(key[4:])
            else:
                raise AgentError("Use exact shown tool aliases instead of writing mutation IDs")
        designs = translated
        if len(designs) != len(set(designs)):
            raise AgentError("Different aliases or entries denote the same GFP design")
    if score_new is None:
        variants = [_canonical(key) for key in designs]
    else:
        try:
            variants = [make_variant(key) for key in designs]
        except ValueError as exc:
            raise AgentError(f"Invalid open GFP design: {exc}") from exc
        if len(set(v.id for v in variants)) != count:
            raise AgentError("Different edit orders still denote the same GFP design")
    unseen = [variant for variant in variants if variant.id not in candidates]
    if unseen and score_new is None:
        raise AgentError("A closed trial candidate must occur in its displayed pool")
    if score_new is not None:
        disallowed = [v.id for v in unseen if v.id in measured or v.id in pending or v.id == "WT"]
        if disallowed:
            raise AgentError(f"Exclude these already observed or pending GFP IDs: {', '.join(disallowed)}")
        if not unseen:
            raise AgentError("Score at least one legal GFP design outside the displayed candidate pool")
    new_scores = score_new(unseen) if unseen else {}
    if set(new_scores) != {variant.id for variant in unseen}:
        raise AgentError("The scorer must return a prediction and risk for every new design")
    scores = {**candidates, **new_scores}
    rows = []
    by_id = {key: alias for alias, key in aliases.items()} if aliases else {}
    for variant in variants:
        score = scores[variant.id]
        if not _finite(score.get("prediction")) or not _finite(score.get("risk")) or score["risk"] < 0:
            raise AgentError("The predictor returned invalid scores")
        row = {"id": variant.id, "sequence": variant.sequence, "substitutions": list(variant.edits),
               "prediction": score["prediction"], "risk": score["risk"],
               "source": "shared_pool" if variant.id in candidates else "off_menu_design"}
        if aliases:
            row["alias"] = by_id.get(variant.id, f"N{len(rows) + 1:03d}")
        if knowledge:
            row["chemistry_class_changes"] = [[edit, CHEMISTRY_GROUP[edit[0]], CHEMISTRY_GROUP[edit[-1]]]
                                               for edit in variant.edits]
        rows.append(row)
    return call, rows


_FINAL_FIELDS = {"selected_ids", "primary_id", "comparator", "evidence_ids", "strategy", "adaptation"}


def _decision(response, scored, measured, incumbent, recent, budget):
    if response.get("finish_reason") != "stop" or response.get("tool_calls"):
        raise AgentError("Return only the final JSON decision after scoring")
    final = _object(response.get("text"), _FINAL_FIELDS, "final decision")
    selected = final["selected_ids"]
    scored_by_id = {row["id"]: row for row in scored}
    aliases = {row["alias"]: row["id"] for row in scored if "alias" in row}
    primary = final["primary_id"]
    if aliases:
        if isinstance(selected, list):
            selected = [aliases.get(key, key) if isinstance(key, str) else key for key in selected]
        primary = aliases.get(primary, primary) if isinstance(primary, str) else primary
    if (not isinstance(selected, list) or len(selected) != budget
            or not all(isinstance(x, str) and x in scored_by_id for x in selected)
            or len(set(selected)) != budget or primary not in selected):
        raise AgentError(f"Choose exactly {budget} distinct scored IDs, including the primary ID")
    if final["comparator"] not in ("WT", "incumbent"):
        raise AgentError("Use measured WT or the observed incumbent as comparator")
    if final["strategy"] not in STRATEGIES or final["adaptation"] not in ADAPTATIONS:
        raise AgentError("Use the enumerated strategy and adaptation values")
    if not recent and final["adaptation"] != "no_new_feedback":
        raise AgentError("Use no_new_feedback when no newly observed measurements were supplied")
    if recent and final["adaptation"] == "no_new_feedback":
        raise AgentError("Record how the policy responds when newly observed measurements were supplied")
    evidence = final["evidence_ids"]
    if (not isinstance(evidence, list) or not 1 <= len(evidence) <= 4
            or not all(isinstance(key, str) and key in measured for key in evidence) or len(set(evidence)) != len(evidence)):
        raise AgentError("Cite one to four distinct, previously observed GFP IDs")
    comparator = "WT" if final["comparator"] == "WT" else incumbent
    threshold = measured[comparator]["fitness"]
    target = scored_by_id[primary]
    return {
        "selected_ids": selected,
        "hypotheses": [{"candidate_id": target["id"], "comparator_id": comparator, "comparator_fitness": threshold,
                         "direction": "above", "evidence_ids": evidence, "forecast_fitness": target["prediction"],
                         "forecast_exceeds_comparator": target["prediction"] > threshold,
                         "statement": f"{target['id']} will have fluorescence strictly above measured {comparator}.",
                         "scope": "fluorescence comparison only", "status": "PENDING"}],
        "designs": scored,
        "evidence": [measured[key] for key in evidence],
        "recent_evidence_used": sorted(set(evidence) & set(recent)),
        "strategy": final["strategy"], "adaptation": final["adaptation"],
        "mechanism_validation": "NOT EVALUABLE",
        "prediction_only": sorted(key for key in selected if scored_by_id[key]["source"] == "off_menu_design"),
    }


def _message(received, scored, budget, recent):
    call = received["tool_calls"][0]
    # Sequence stays in the trace; canonical edit IDs identify every design in the prompt.
    compact = [{key: value for key, value in row.items() if key != "sequence"} for row in scored]
    return [
        {"role": "assistant", "content": received.get("text") or None, "tool_calls": [call]},
        {"role": "tool", "tool_call_id": call["id"], "content": _json(compact)},
        {"role": "user", "content": (
            f"Choose {budget} scored IDs. Return a JSON object with selected_ids (array), primary_id (included), "
            "comparator (WT or incumbent), and evidence_ids (1 through 4 observed IDs). "
            "The primary candidate asserts a prospective comparison strictly above that measured comparator. "
            f"strategy is one of {_json(STRATEGIES)}. adaptation is one of {_json(ADAPTATIONS if recent else ('no_new_feedback',))}. "
            + ("Consider and cite relevant newly available observations. " if recent else "")
            + ("Use the tool's scored aliases in selected_ids and primary_id, and measured mutation IDs in evidence_ids. "
               if any("alias" in row for row in scored) else "")
            + "No extra fields. A comparator check cannot validate a biochemical mechanism."
        )},
    ]


def _read_archive(record, signature):
    if record is None:
        return None
    if record["input_sha256"] != signature or record["status"] != "accepted":
        raise AgentError("Replay requires the matching accepted native-tool decision trace")
    return record


def select(*, request: dict, transport, knowledge: bool, log_path: Path, replay_path: Path | None = None,
           score_new=None, repair_aliases: bool = False) -> dict:
    """Choose after native score calls. No assay, truth table or future measurements are accepted."""
    measured, incumbent, recent, candidates = _request(request)
    budget = request["budget"]
    open_world = score_new is not None
    pending = set(request.get("pending_ids", ()))
    if any(_canonical(key).id != key or key in measured for key in pending):
        raise AgentError("Pending designs must be distinct from measured evidence")
    if len(candidates) < budget and not open_world:
        raise AgentError("Closed candidate pool is smaller than the shared measurement budget")
    if open_world and len(candidates.keys() & pending):
        raise AgentError("Pending designs cannot appear in the displayed menu")
    count = budget + 2 if open_world else min(len(candidates), budget + 2)
    raw_archive = json.loads(Path(replay_path).read_text()) if replay_path else None
    use_aliases = repair_aliases or (open_world and bool(pending) and len(candidates) >= count - 1)
    if raw_archive is not None:
        use_aliases = raw_archive.get("alias_repair", False)
    aliases = {f"C{index:03d}": key for index, key in enumerate(candidates, 1)} if use_aliases else None
    identity = {"request": request, "knowledge": knowledge, "model": transport.model, "open_world": open_world}
    if use_aliases:
        identity["alias_repair"] = True
    signature = digest(identity)
    archive = _read_archive(raw_archive, signature)
    messages = _proposal_message(request, candidates, open_world, count, knowledge, aliases)
    trace = {"input_sha256": signature, "model": transport.model, "knowledge": knowledge, "open_world": open_world,
             "input": request, "backend_config": transport.config, "status": "failed", "calls": []}
    if use_aliases:
        trace["alias_repair"] = True
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def invoke(stage, prompt, validate):
        proposal = stage == "proposal"
        specification = TOOL_SPEC
        if aliases:
            description = "Score exact shown aliases, such as C001, without retyping the genotype."
            if open_world:
                description += " You may also submit a legal unobserved design in the format NEW followed by its genotype."
            specification = {**TOOL_SPEC, "function": {**TOOL_SPEC["function"], "description": description}}
        base = {"model": transport.model, "messages": prompt, "temperature": 0,
                "max_tokens": PROPOSAL_TOKENS if proposal else DECISION_TOKENS,
                "tools": [specification]}
        if proposal:
            base["tool_choice"] = {"type": "function", "function": {"name": TOOL}}
        else:
            base.update(tool_choice="none", response_format={"type": "json_object"})
        for attempt in range(MAX_ATTEMPTS):
            wire = transport.encode(base)
            if (urlsplit(transport.endpoint).hostname == "codeflow.sh"
                    and transport.model == "deepseek-ai/DeepSeek-V4.1-Flash"):
                # Match the primary V1 provider's nonthinking configuration. Otherwise the
                # gateway can spend the entire response budget on hidden reasoning.
                wire = {**wire, "thinking": {"type": "disabled"}}
            call_index = len(trace["calls"])
            call = {"stage": stage, "attempt": attempt + 1, "request_sha256": digest(wire), "wire": wire}
            trace["calls"].append(call)
            replay_call = archive["calls"][call_index] if archive is not None and call_index < len(archive["calls"]) else None
            if archive is not None and (replay_call is None or replay_call["request_sha256"] != digest(wire)
                                        or digest(replay_call["wire"]) != replay_call["request_sha256"]):
                raise AgentError("Archived native request did not match the replayed request")
            try:
                if replay_call is not None:
                    if "transport_error" in replay_call:
                        raise AgentError(replay_call["transport_error"])
                    response = replay_call["response"]
                    if digest(response) != replay_call["response_sha256"]:
                        raise RuntimeError("Archived native response hash mismatch")
                else:
                    response = transport.generate(transport.endpoint, wire)
            except AgentError as exc:
                call["transport_error"] = str(exc)
                retryable = any(reason in str(exc) for reason in ("status 429", "status 500", "status 502", "status 503", "status 504", "connection failed"))
                if not retryable or attempt + 1 == MAX_ATTEMPTS:
                    raise
                if replay_call is None:
                    time.sleep(RETRY_DELAYS[attempt])
                continue
            call.update(response=response, response_sha256=digest(response))
            try:
                return validate(response), response
            except AgentError as exc:
                call["validation_error"] = str(exc)
                if attempt + 1 == MAX_ATTEMPTS:
                    raise
                base["messages"] = [*prompt, {"role": "user", "content": f"Retry following the requested format. {exc}"}]
        raise RuntimeError("V2 call retry state did not terminate")

    with log_path.open("x") as stream:
        try:
            outcome, response = invoke("proposal", messages, lambda reply: _scored_tool(
                reply, candidates, measured, pending, score_new, count, knowledge, aliases,
            ))
            call, scored = outcome
            trace.update(tool_call=call, tool_results=scored, tool_results_sha256=digest(scored))
            if archive is not None and (archive["tool_results_sha256"] != digest(scored)
                                        or digest(archive["tool_results"]) != archive["tool_results_sha256"]):
                raise AgentError("Archived tool results did not match the scored designs")
            outcome, _ = invoke("decision", [*messages, *_message(response, scored, budget, recent)], lambda reply: _decision(
                reply, scored, measured, incumbent, recent, budget,
            ))
            if archive is not None and (len(trace["calls"]) != len(archive["calls"])
                                        or digest(archive["decision"]) != archive["decision_sha256"]
                                        or digest(outcome) != archive["decision_sha256"]):
                raise AgentError("Archived V2 selection differed from the verified replay")
            trace.update(status="accepted", decision=outcome, decision_sha256=digest(outcome))
            return outcome
        finally:
            counts = Counter()
            for call in trace["calls"]:
                counts.update(call.get("response", {}).get("usage", {}))
            trace["usage"] = dict(counts)
            trace["replayed"] = archive is not None
            json.dump(trace, stream, ensure_ascii=True, allow_nan=False, indent=2)
            stream.write("\n")


def evaluate_hypotheses(decision: dict, *, measurements: dict, wild_type_fitness: float, prior_best: float) -> list[dict]:
    """Evaluate exactly the preregistered comparisons among newly available observations."""
    outcomes = []
    for test in decision.get("hypotheses", ()):
        key = test["candidate_id"]
        tested = key in measurements
        value = measurements[key] if tested else None
        if tested and not _finite(value):
            raise AgentError("GFP fluorescence for a completed hypothesis must be finite")
        threshold = test["comparator_fitness"]
        status = ("SUPPORTED" if value > threshold else "REFUTED") if tested else "PENDING"
        outcomes.append({**test, "status": status, "observed_fitness": value,
                         "prediction_residual": value - test["forecast_fitness"] if tested else None,
                         "above_WT": value > wild_type_fitness if tested else None,
                         "improves_incumbent": value > prior_best if tested else None,
                         "mechanism_validation": "NOT EVALUABLE", "feedback_gain": "NOT EVALUABLE"})
    return outcomes
