"""Observed-only avGFP knowledge graph and a native-tool graph-versus-rules exercise."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
from statistics import mean
import time
from urllib.parse import urlsplit

from gfp_continue import load_csv
from gfp_data import GFP_WT, load_dataset, make_variant
from gfp_setup import load_openai_profile
from gfp_v2.policy import digest
from gfp_v2.predictor import CHARGE, CHEMISTRY, HYDROPATHY, fit
from llm_agent import AgentError


SIDE_CHAIN_HEAVY_ATOMS = dict(zip("ACDEFGHIKLMNPQRSTVWY", (1, 2, 4, 5, 7, 0, 6, 4, 5, 4, 4, 4, 3, 5, 7, 2, 3, 3, 10, 8), strict=True))
PROPERTY_SOURCE = {
    "hydropathy": "Kyte & Doolittle, J Mol Biol (1982), DOI:10.1016/0022-2836(82)90515-0",
    "charge": "Nominal acidic/basic side-chain charge. Histidine shown as a neutral approximation.",
    "size": "Non-hydrogen atom count in the side chain; size proxy, not molecular volume.",
    "polarity": "Coarse side-chain categories from the V2 sequence feature rules; special residues kept separate.",
}
TOOL = {
    "type": "function", "function": {"name": "query_gfp_knowledge", "description": "Retrieve evidence for one to three positions on the fixed GFP reference before deciding",
                                      "parameters": {"type": "object", "properties": {"positions": {"type": "array", "items": {"type": "integer"}}}, "required": ["positions"], "additionalProperties": False}},
}


def property_values(aa: str) -> dict:
    return {"hydropathy": HYDROPATHY[aa], "charge": CHARGE.get(aa, 0), "size": SIDE_CHAIN_HEAVY_ATOMS[aa], "polarity": CHEMISTRY[aa]}


def classify_edit(edit: str) -> str:
    """Coarse class changes, used to contextualize predictions rather than impose a fitness prior."""
    variant = make_variant(edit)
    if len(variant.edits) != 1:
        raise ValueError("Expected one legal substitution against the GFP reference")
    a, b = variant.edits[0][0], variant.edits[0][-1]
    return "conservative" if CHEMISTRY[a] == CHEMISTRY[b] and CHEMISTRY[a] != "special" else "radical_or_special"


class ObservedGraph:
    """Exact WT-anchored relations over imported observations only."""

    def __init__(self, observations):
        self.nodes = {}
        self.edges = []
        self.observed = {}
        self.positions = {}
        for aa in HYDROPATHY:
            self.nodes[f"aa:{aa}"] = {"kind": "amino_acid", "residue": aa}
            for name, value in property_values(aa).items():
                attribute = f"property:{name}:{value}"
                self.nodes[attribute] = {"kind": "property", "name": name, "value": value, "source": PROPERTY_SOURCE[name]}
                self._edge(f"aa:{aa}", "has_property", attribute)
        rows = list(observations)
        anchors = [row for row in rows if row["id"] == "WT"]
        if len(anchors) != 1:
            raise ValueError("The observed ledger must contain exactly one measured WT reference")
        self.update(anchors + [row for row in rows if row["id"] != "WT"])

    def _edge(self, source, relation, target, **metadata):
        self.edges.append({"source": source, "relation": relation, "target": target, **metadata})

    def update(self, observations) -> None:
        """Add one released batch, rejecting duplicate IDs and invalid phenotypes atomically."""
        rows = list(observations)
        variants = []
        keys = set(self.observed)
        for row in rows:
            if set(row) != {"id", "fitness", "source"} or not row["source"] or type(row["fitness"]) not in (int, float) or not math.isfinite(row["fitness"]):
                raise ValueError("An observed row requires a canonical ID, finite fitness, and source")
            variant = make_variant(row["id"])
            if variant.id != row["id"] or variant.id in keys:
                raise ValueError("Graph evidence requires unique canonical GFP IDs")
            variants.append(variant)
            keys.add(variant.id)
        for variant, row in zip(variants, rows, strict=True):
            self.observed[variant.id] = dict(row)
            variant_id, obs_id = f"variant:{variant.id}", f"observation:{variant.id}"
            self.nodes[variant_id] = {"kind": "variant", "id": variant.id, "substitution_count": len(variant.edits)}
            self.nodes[obs_id] = {"kind": "fitness_observation", "fitness": row["fitness"], "source": row["source"]}
            self._edge(variant_id, "has_measurement", obs_id)
            for edit in variant.edits:
                mutation_id = f"mutation:{edit}"
                position = int(edit[1:-1])
                site_id = f"position:{position}"
                if mutation_id not in self.nodes:
                    self.nodes.setdefault(site_id, {"kind": "position", "number": position, "reference": GFP_WT[position - 1]})
                    self.nodes[mutation_id] = {"kind": "mutation", "id": edit, "classification": classify_edit(edit)}
                    self._edge(mutation_id, "occurs_at", site_id)
                    self._edge(mutation_id, "changes_from", f"aa:{edit[0]}")
                    self._edge(mutation_id, "changes_to", f"aa:{edit[-1]}")
                self._edge(variant_id, "contains", mutation_id)
                self.positions.setdefault(position, []).append((variant.id, edit, len(variant.edits)))
            if variant.id == "WT":
                continue
            delta = row["fitness"] - self.observed["WT"]["fitness"]
            scope = "single_substitution_on_WT_observational_contrast" if len(variant.edits) == 1 else "whole_variant_association_no_site_attribution"
            self._edge(obs_id, "compared_with", "observation:WT", delta_fitness=delta, scope=scope)
            if delta > 0 and len(variant.edits) == 1:
                self._edge(f"mutation:{variant.edits[0]}", "observed_above_measured_reference", obs_id,
                           reference="observation:WT", delta_fitness=delta, scope=scope)

    def query(self, positions, *, rule_only: bool = False, sample_limit: int | None = 6):
        if not 1 <= len(positions) <= len(GFP_WT) or len(set(positions)) != len(positions) or any(type(p) is not int or not 1 <= p <= len(GFP_WT) for p in positions):
            raise ValueError("Query distinct positions within the GFP reference")
        wt = self.observed["WT"]["fitness"]
        response = {"reference": {"id": "WT", "fitness": wt, "source": self.observed["WT"]["source"]}, "positions": []}
        for pos in positions:
            site = {"position": pos, "reference_residue": GFP_WT[pos - 1], "reference_properties": property_values(GFP_WT[pos - 1])}
            if not rule_only:
                evidence = []
                for key, edit, count in self.positions.get(pos, []):
                    record = self.observed[key]
                    evidence.append({"id": key, "edit": edit, "fitness": record["fitness"], "source": record["source"],
                                     "WT_delta_fitness": record["fitness"] - wt, "scope": "single" if count == 1 else "multi_association",
                                     "classification": classify_edit(edit)})
                evidence.sort(key=lambda row: (row["scope"] != "single", -abs(row["WT_delta_fitness"]), row["id"]))
                for scope in ("single", "multi_association"):
                    group = [row for row in evidence if row["scope"] == scope]
                    site[scope] = {"n": len(group), "above_WT": sum(row["WT_delta_fitness"] > 0 for row in group),
                                   "mean_WT_delta_fitness": mean(row["WT_delta_fitness"] for row in group) if group else None,
                                   "observations": group[:sample_limit]}
                site["interpretation"] = "Single contrasts are WT-background measurements; multi associations cannot assign site effects."
            response["positions"].append(site)
        return response


def write(path: Path, body):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(body, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write("\n")


def _native_decision(request, graph, transport, *, rule_only, output, replay=None, focus=None):
    # All candidates and observed outcomes are identical across the two modes. Only the query result changes.
    candidates = {f"C{i:03d}": row for i, row in enumerate(request["candidates"], 1)}
    evidence_hints = {pos: len(rows) for pos, rows in sorted(graph.positions.items())}
    graph_wt = graph.observed["WT"]["fitness"]
    user = {"WT": graph_wt, "candidate_columns": ["alias", "GFP mutations", "prediction", "risk"],
            "candidate_rows": [[alias, row["id"], round(row["prediction"], 5), round(row["risk"], 5)]
                               for alias, row in candidates.items()],
            "observed_context": [[row["id"], row["fitness"]] for row in request["observed"]],
            "support_counts_by_position": evidence_hints}
    messages = [{"role": "system", "content": "You design isolated avGFP fluorescence comparisons. Only the supplied observations are measured. Distinguish single substitution WT-background observations from multi-position associations. Use query_gfp_knowledge before selecting five candidate aliases. Higher fluorescence is better."},
                {"role": "user", "content": "Call the native query for 1-3 positions that occur in candidates you are considering. Favor positions whose support counts warrant review.\n" + json.dumps(user)}]
    if focus:
        focus_aliases = [alias for alias, row in candidates.items() if set(focus["positions"]) & {
            int(edit[1:-1]) for edit in make_variant(row["id"]).edits}]
        if not focus_aliases:
            raise ValueError("No eligible archived candidate overlaps the predeclared focus positions")
        messages[1]["content"] += (f"\nThe preceding primary {focus['id']} has been measured at {focus['fitness']} versus WT {graph_wt}. "
                                   f"Query its exact positions {focus['positions']} to plan an alternative comparison. The new "
                                   f"primary must use one of these focus-compatible aliases {focus_aliases} and cite the returned "
                                   "preceding measurement as observed evidence.")
    trace = {"mode": "rules" if rule_only else "graph", "input_sha256": digest(request), "model": transport.model,
             "status": "failed", "calls": []}

    def call(prompt, stage, validate):
        base = {"model": transport.model, "messages": prompt, "tools": [TOOL], "max_tokens": 3600, "temperature": 0}
        if stage == "query":
            base["tool_choice"] = {"type": "function", "function": {"name": TOOL["function"]["name"]}}
        else:
            base.update(tool_choice="none", response_format={"type": "json_object"})
        for attempt in range(3):
            wire = transport.encode(base)
            if urlsplit(transport.endpoint).hostname == "codeflow.sh" and transport.model == "deepseek-ai/DeepSeek-V4.1-Flash":
                wire["thinking"] = {"type": "disabled"}
            entry = {"stage": stage, "attempt": attempt + 1, "request_sha256": digest(wire), "wire": wire}
            index = len(trace["calls"])
            trace["calls"].append(entry)
            try:
                if replay is None:
                    response = transport.generate(transport.endpoint, wire)
                else:
                    cached = replay["calls"][index]
                    expected = cached["request_sha256"]
                    if expected != digest(wire):
                        raise RuntimeError("Graph replay request did not match")
                    if digest(cached["wire"]) != expected:
                        # Initial evidence captured the first wire's message list by reference;
                        # its digest predates the three appended tool-continuation messages.
                        cached_messages = cached["wire"]["messages"]
                        prefix = {**cached["wire"], "messages": cached_messages[:2]}
                        if (stage != "query" or index != 0 or len(cached_messages) != 5
                                or [row["role"] for row in cached_messages[2:]] != ["assistant", "tool", "user"]
                                or digest(prefix) != expected):
                            raise RuntimeError("Graph replay archived wire failed digest checks")
                        entry["verified_origin"] = "original two-message query prefix"
                    if "transport_error" in cached:
                        raise AgentError(cached["transport_error"])
                    response = cached["response"]
                    if digest(response) != cached["response_sha256"]:
                        raise RuntimeError("Graph replay response digest failed")
            except AgentError as exc:
                entry["transport_error"] = str(exc)
                if attempt < 2 and any(text in str(exc) for text in ("429", "500", "502", "503", "504", "connection failed")):
                    if replay is None:
                        time.sleep((4, 12)[attempt])
                    continue
                raise
            entry.update(response=response, response_sha256=digest(response))
            try:
                return validate(response)
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                entry["validation_error"] = str(exc)
                if attempt == 2:
                    raise
                base["messages"] = [*prompt, {"role": "user", "content": f"Repair the output to meet this error: {exc}"}]
        raise RuntimeError("Graph tool retry did not stop correctly")

    def valid_query(response):
        calls = response["tool_calls"]
        if response["finish_reason"] != "tool_calls" or len(calls) != 1 or calls[0]["function"]["name"] != TOOL["function"]["name"]:
            raise ValueError("Produce one native position query call")
        obj = json.loads(calls[0]["function"]["arguments"])
        if set(obj) != {"positions"} or not 1 <= len(obj["positions"]) <= 3:
            raise ValueError("Provide no more than three distinct reference positions")
        if focus and set(obj["positions"]) != set(focus["positions"]):
            raise ValueError("Query all predeclared positions of the measured preceding primary")
        return calls[0], graph.query(obj["positions"], rule_only=rule_only, sample_limit=None if focus else 6)

    output.mkdir(parents=True, exist_ok=False)
    try:
        tool_call, returned = call(messages, "query", valid_query)
        trace.update(tool_call=tool_call, tool_result=returned, tool_result_sha256=digest(returned))
        queried_sites = {record["position"] for record in returned["positions"]}
        linked = {obs["id"] for site in returned["positions"] for scope in ("single", "multi_association")
                  for obs in site.get(scope, {}).get("observations", [])}
        messages = [*messages, {"role": "assistant", "content": None, "tool_calls": [tool_call]},
                     {"role": "tool", "tool_call_id": tool_call["id"], "content": json.dumps(returned)},
                     {"role": "user", "content": (
                         "Return JSON with selected_aliases (exactly 5 distinct C aliases), primary_alias (one of those 5, chosen before the WT comparison), "
                         "evidence_ids (0 to 3 existing observed GFP IDs), and basis (one of observed_evidence, physicochemical_rule, predictor). "
                         "For observed_evidence, include at least one measured ID sharing a mutation position with the primary candidate. "
                         "Prefer a linked record returned by the tool when it is relevant. Do not invent observations or mechanistic claims. "
                         "The precommitted primary hypothesis is strictly higher fluorescence than measured WT."
                     )}]
        if focus:
            messages[-1]["content"] += (f" The primary_alias must be one of {focus_aliases}. Include that alias among the five "
                                        f"selected_aliases and cite the graph-returned {focus['id']} as observed_evidence.")

        def final(response):
            if response["finish_reason"] != "stop" or response["tool_calls"]:
                raise ValueError("Return the final decision as JSON only")
            verdict = json.loads(response["text"])
            if set(verdict) != {"selected_aliases", "primary_alias", "evidence_ids", "basis"}:
                raise ValueError("Use the four exact final selection fields")
            ids = verdict["selected_aliases"]
            evidence = verdict["evidence_ids"]
            if not isinstance(ids, list) or len(ids) != 5 or len(set(ids)) != 5 or not set(ids) <= candidates.keys() or verdict["primary_alias"] not in ids:
                raise ValueError("Use exactly five available distinct C aliases with a primary among them")
            if focus and verdict["primary_alias"] not in focus_aliases:
                raise ValueError(f"For primary_alias, use one of {focus_aliases} and include it in selected_aliases")
            if (not isinstance(evidence, list) or len(evidence) > 3 or len(set(evidence)) != len(evidence)
                    or not set(evidence) <= graph.observed.keys() or verdict["basis"] not in ("observed_evidence", "physicochemical_rule", "predictor")
                    or (verdict["basis"] == "observed_evidence" and not evidence)):
                raise ValueError("Cite only measured observation IDs and use an allowed evidence category")
            primary = candidates[verdict["primary_alias"]]["id"]
            primary_edits = set(make_variant(primary).edits)
            primary_positions = {int(edit[1:-1]) for edit in primary_edits}
            links = []
            for observed_id in evidence:
                edits = set(make_variant(observed_id).edits)
                links.append({"observed_id": observed_id, "fitness": graph.observed[observed_id]["fitness"],
                              "source": graph.observed[observed_id]["source"], "retrieved_by_graph": observed_id in linked,
                              "shared_exact_edits": sorted(primary_edits & edits),
                              "shared_positions": sorted(primary_positions & {int(edit[1:-1]) for edit in edits}),
                              "scope": "single_WT_contrast" if len(edits) == 1 else "whole_variant_association" if edits else "WT"})
            if verdict["basis"] == "observed_evidence" and not any(link["shared_positions"] for link in links):
                raise ValueError("Observed evidence must share a position with the primary design")
            if focus and (verdict["basis"] != "observed_evidence" or not any(
                    link["observed_id"] == focus["id"] and link["retrieved_by_graph"] and link["shared_positions"] for link in links)):
                raise ValueError("The focused primary must cite the graph-returned preceding measurement and share its position")
            return {**verdict,
                    "selected": [{**candidates[alias], "sequence": make_variant(candidates[alias]["id"]).sequence,
                                  "substitution_rules": {edit: classify_edit(edit) for edit in make_variant(candidates[alias]["id"]).edits}}
                                 for alias in ids],
                    "primary_id": primary, "comparator": "WT", "threshold": graph_wt,
                    "direction": "strictly_above", "empirical_claim": "PENDING", "mechanism_validation": "NOT EVALUABLE",
                    "queried_primary_positions": sorted(queried_sites & primary_positions), "primary_evidence_links": links}

        decision = call(messages, "decision", final)
        trace.update(status="accepted", decision=decision, decision_sha256=digest(decision))
        if replay is not None and (trace["tool_result_sha256"] != replay["tool_result_sha256"]
                                   or trace["decision_sha256"] != replay["decision_sha256"]
                                   or len(trace["calls"]) != len(replay["calls"])):
            raise ValueError("Graph replay did not reproduce tool and decision hashes")
        write(output / "commit.json", decision)
    finally:
        trace["replayed"] = replay is not None
        tokens = Counter()
        for entry in trace["calls"]:
            tokens.update(entry.get("response", {}).get("usage", {}))
        trace["usage"] = dict(tokens)
        write(output / "agent.json", trace)


def case(request_path: Path, output: Path, transport, *, replay: Path | None = None):
    request = json.loads(request_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    observations = [{"id": row["id"], "fitness": row["fitness"], "source": f"frozen request {request_path.as_posix()} observed round {row['round']}"}
                    for row in request["observed"]]
    graph = ObservedGraph(observations)
    write(output / "graph_before.json", {"nodes": graph.nodes, "edges": graph.edges})
    support = graph.query(sorted(graph.positions), sample_limit=None)["positions"]
    write(output / "observed_positions.json", support)
    with (output / "observed_positions.csv").open("x", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["position", "WT_residue", "single_n", "single_above_WT", "single_mean_delta_WT", "multi_n", "multi_above_WT", "multi_mean_delta_WT"])
        for site in support:
            writer.writerow([site["position"], site["reference_residue"], *(site[scope][metric] for scope in ("single", "multi_association") for metric in ("n", "above_WT", "mean_WT_delta_fitness"))])
    def run(rule):
        name = "rules" if rule else "graph"
        archived = json.loads((replay / name / "agent.json").read_text()) if replay else None
        _native_decision(request, graph, transport, rule_only=rule, output=output / name, replay=archived)
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(run, (False, True)))
    # Selectors and graph above received no unrevealed fitness. Only accepted commit IDs cross this boundary.
    archive = load_dataset()
    summary = {}
    for name in ("graph", "rules"):
        folder = output / name
        commit = json.loads((folder / "commit.json").read_text())
        eligible = {row["id"] for row in request["candidates"]}
        picks = [row["id"] for row in commit["selected"]]
        if len(picks) != 5 or len(set(picks)) != 5 or not set(picks) <= eligible:
            raise ValueError("Invalid committed graph exercise candidates")
        rows = [{"id": key, "fitness": archive.scores[key], "source": f"{name} committed retrospective archive query"} for key in picks]
        write(folder / "measurements.json", {"commit_sha256": digest(commit), "rows": rows})
        values = {row["id"]: row["fitness"] for row in rows}
        score = values[commit["primary_id"]]
        summary[name] = {"selected_ids": picks, "best": max(values.values()), "mean": mean(values.values()),
                         "above_WT": sum(value > commit["threshold"] for value in values.values()),
                         "primary_id": commit["primary_id"], "primary_fitness": score,
                         "primary_threshold": commit["threshold"], "primary_test": "SUPPORTED" if score > commit["threshold"] else "REFUTED",
                         "primary_evidence_links": commit["primary_evidence_links"], "queried_primary_positions": commit["queried_primary_positions"]}
        if name == "graph":
            query = json.loads((folder / "agent.json").read_text())["tool_result"]
            positions = [site["position"] for site in query["positions"]]
            graph.update(rows)
            write(folder / "updated_query.json", graph.query(positions))
            write(output / "graph_after.json", {"nodes": graph.nodes, "edges": graph.edges})
    summary["graph_minus_rules_best"] = summary["graph"]["best"] - summary["rules"]["best"]
    summary["new_measured_count_graph"] = len(graph.observed) - len(observations)
    summary["scope"] = "One paired archived workflow comparison; additional measured decision value NOT EVALUABLE"
    write(output / "summary.json", summary)


def followup(request_path, prior_case, output, transport, *, replay=None):
    """Inspect the failed earlier primary's positions in the updated graph and run one new comparison."""
    request = json.loads(request_path.read_text())
    prior = json.loads((prior_case / "graph/commit.json").read_text())
    feedback = json.loads((prior_case / "graph/measurements.json").read_text())
    if feedback["commit_sha256"] != digest(prior) or set(row["id"] for row in feedback["rows"]) != set(row["id"] for row in prior["selected"]):
        raise ValueError("Follow-up graph update requires released feedback linked to its actual immutable selection")
    graph = ObservedGraph({"id": row["id"], "fitness": row["fitness"], "source": f"frozen request {request_path.as_posix()} observed round {row['round']}"}
                          for row in request["observed"])
    graph.update(feedback["rows"])
    key = prior["primary_id"]
    focus = {"id": key, "fitness": graph.observed[key]["fitness"],
             "positions": sorted(int(edit[1:-1]) for edit in make_variant(key).edits)}
    if focus["fitness"] > prior["threshold"] or not 1 <= len(focus["positions"]) <= 3:
        raise ValueError("This prerecorded follow-up exercises a refuted primary with up to three substitutions")
    remaining = [row["id"] for row in request["candidates"] if row["id"] not in graph.observed]
    model = fit([make_variant(i) for i in graph.observed], [row["fitness"] for row in graph.observed.values()], 2201, "blend")
    forecasts, risks = model.estimate([make_variant(i) for i in remaining])
    updated = {"observed": list(graph.observed.values()), "candidates": [{"id": i, "prediction": float(y), "risk": float(risk)}
               for i, y, risk in zip(remaining, forecasts, risks, strict=True)]}
    previous = json.loads((replay / "agent.json").read_text()) if replay else None
    _native_decision(updated, graph, transport, rule_only=False, output=output, replay=previous, focus=focus)
    commit = json.loads((output / "commit.json").read_text())
    # This retrospective exercise queries the archive only for the new batch, after the graph-based decision is written.
    archive = load_dataset()
    returned = {row["id"]: archive.scores[row["id"]] for row in commit["selected"]}
    score = returned[commit["primary_id"]]
    write(output / "updated_request.json", updated)
    write(output / "measurements.json", {"commit_sha256": digest(commit), "measurements": returned})
    write(output / "summary.json", {"first_refuted": focus, "new_primary": commit["primary_id"],
          "new_threshold": commit["threshold"], "new_primary_score": score, "new_primary_test": "SUPPORTED" if score > commit["threshold"] else "REFUTED",
          "best_in_five": max(returned.values()), "evidence_links": commit["primary_evidence_links"],
          "causal_feedback_gain": "NOT EVALUABLE", "mechanism_validation": "NOT EVALUABLE"})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    query = commands.add_parser("query", help="Inspect positions using only a measured GFP CSV")
    query.add_argument("--measurements", type=Path, required=True)
    query.add_argument("--positions", type=int, nargs="+", required=True)
    query.add_argument("--reference", type=Path, help="Optional reference FASTA; must exactly match the pinned GFP sequence")
    exercise = commands.add_parser("case", help="Run the fixed observed-context graph/rule exercise")
    exercise.add_argument("--request", type=Path, required=True)
    exercise.add_argument("--output", type=Path, required=True)
    exercise.add_argument("--model-profile", default="default")
    exercise.add_argument("--replay", type=Path)
    follow = commands.add_parser("followup", help="Target the measured refutation using an updated graph and refit predictor")
    follow.add_argument("--request", type=Path, required=True)
    follow.add_argument("--prior-case", type=Path, required=True)
    follow.add_argument("--output", type=Path, required=True)
    follow.add_argument("--model-profile", default="default")
    follow.add_argument("--replay", type=Path)
    args = parser.parse_args(argv)
    if args.stage == "query":
        if args.reference is not None:
            lines = args.reference.read_text().splitlines()
            headers = [row for row in lines if row.startswith(">")]
            sequence = "".join(row.strip() for row in lines if not row.startswith(">"))
            if len(headers) > 1 or sequence != GFP_WT:
                raise ValueError("Only the pinned isolated avGFP amino-acid reference is supported")
        rows = load_csv(args.measurements, measured=True)
        graph = ObservedGraph({"id": key, "fitness": row.fitness, "source": str(args.measurements)} for key, row in rows.items())
        print(json.dumps(graph.query(args.positions), indent=2))
    else:
        adapter = load_openai_profile(args.model_profile, require_key=args.replay is None)
        if args.stage == "case":
            case(args.request, args.output, adapter, replay=args.replay)
        else:
            followup(args.request, args.prior_case, args.output, adapter, replay=args.replay)


if __name__ == "__main__":
    main()
