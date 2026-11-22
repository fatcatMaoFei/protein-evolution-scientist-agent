# GFP follow-up knowledge graph exercise

The fixed inputs are `evidence/gfp-v2/agent/arms/2201/E_knowledge/round-02-request.json` (first V2 agent block, first decision after initial measured returns). This is an archived follow-up demonstration, outside the frozen V2 experiment. Both treatments receive the same predictor estimates and risk values, full eligible pool, released observations, evidence-count hints, and instruction. Using separate native DeepSeek calls, each requests 1–3 WT-reference positions and then commits five distinct candidate aliases, one primary under the strict measured-WT comparison, and relevant observed evidence links. The rules-only tool returns static properties. The graph tool also returns links, phenotype contrasts, and counts derived solely from the released measurement ledger. Both may cite existing observations. Citing graph-returned evidence and overlap of sites or exact edits are logged separately.

The archived fluorescence for selected IDs is read only after both decisions have been written. Report paired batch maximum, mean, and above-WT counts as descriptive demonstration results, and retain any failures or ties. Add the accepted graph batch measurements to a separate graph snapshot, then rerun the same query and show how evidence updates. The individual outcomes do not resolve a prospective graph improvement or validate a biochemical mechanism. Single-substitution contrasts describe measured differences on the WT background under this assay. Combinations remain whole-variant observations.

Chemistry sources appear on the property nodes. Residue size is represented by non-hydrogen atoms in the side chain. Conservative/radical describes broad class changes, not a fitness estimate or evidence of structure. The accepted reporter input is amino acid sequence against the pinned avGFP reference and at most four substitutions. DNA stop-codon checking requires a nucleotide coding sequence and is outside this interface.

## Follow-up after the initial comparison

The first graph primary was refuted. The next exercise uses its measured editing positions as a fixed focus, regardless of later outcomes. Incorporate the five graph-arm returns, refit the selected blend on the enlarged ledger with seed 2201, and display all remaining eligible features without fitness. Require the native call to query the earlier primary's positions and the next primary to share a queried position and cite that exact graph-returned outcome. Precommit another WT comparison, save the decision, then reveal only its five selected archive values. This deliberately tests whether feedback can be retrieved, validated and used in a later recommendation. A focus-matched forced citation and one follow-up batch cannot estimate a treatment effect.

## Results

The observed ledger had 101 measurements including WT, 149 sites with multi-substitution observations, and no measured single-substitution variant. A separate [query on the documented V1 user-measurement ledger](user-ledger-single-example.json) shows a measured single-substitution contrast at position 221: L221R scored 3.72440350818 versus WT 3.7192121319, a difference of about 0.0052. This small single contrast supplies weak observed support and is not attributed to structure or a validated mechanism. The native exercise used its own ledger throughout.

In the [initial native pair](case/summary.json), the graph returned site-linked evidence and the batches shared four of five candidates. The graph and rule batches each found one above-WT result and shared a best value of 3.72035125546. The graph batch mean was 3.581824326454 versus 3.489928947470 for the rule batch. Both chose Y39H:S175G as primary, with measured 3.63329668426 below WT. Their explanations used common-ledger site evidence, and the primary shared no site with the graph query. The differing batches show different choices in one pair; the mechanism of that choice and a reproducible graph advantage remain unresolved.

In the [actual native follow-up](followup/summary.json), the graph grew from 101 to 106 observations, the predictor refit, and site-39 combination support increased from two to three observations. DeepSeek queried 39 and 175, cited the new graph-returned Y39H:S175G observation, and retained its exact Y39H edit in E32G:Y39H:H169R:I188F. Its refit forecast was 2.69179005664 and its measured fluorescence was 1.30103001866, which refuted the precommitted WT comparison of 3.7192121319. The first attempt repeatedly nominated a primary outside the tested sites and is preserved in `followup-preflight-alias-failure`. The accepted prompt spelled out the focus-eligible alias. Only one eligible alternative shared a focus position, so the follow-up demonstrates auditable evidence retrieval and a refuted follow-up, with no basis for assigning a graph treatment gain.

## Reproduction

```bash
PY=.venv/bin/python
BASE=evidence/gfp-v2/supplement/knowledge_graph
R=evidence/gfp-v2/agent/arms/2201/E_knowledge/round-02-request.json
M=gfp-third-party
$PY -m gfp_v2.knowledge_graph query --measurements evidence/gfp/quality/hypothesis/round-01/imported_measurements.csv --positions 221
$PY -m gfp_v2.knowledge_graph case --request "$R" --model-profile "$M" --replay "$BASE/case" --output runs/gfp-kg-replay/case
$PY -m gfp_v2.knowledge_graph followup --request "$R" --prior-case runs/gfp-kg-replay/case --model-profile "$M" --replay "$BASE/followup" --output runs/gfp-kg-replay/followup
```

Use the verified served-model profile. Replay reproduced tool and decision hashes and the committed summaries without API calls. The original recorded query SHA predates a mutable-list capture issue; replay reconstructs and verifies exactly the original two-message query prefix for those unchanged logs. [Verification receipt](receipt.json).
