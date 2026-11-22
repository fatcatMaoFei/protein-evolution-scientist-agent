# Scientific Decision Auditor protocol

The auditor uses frozen 28-day concrete runs. It has no connection to the archival GB1 predictor, and it does not run acquisition rounds. It may start once the original four model groups have finished. Results and new traces write to a new audit path. Main inputs are hashed before and after the audit.

## Archived trajectory

Reconstruct every model, knowledge state, seed and round from the tool trace, batch ledger, round diagnostics and initial observations. Preserve hypothesis, proposed and selected IDs, predicted and measured outcomes, declared evidence and its observed provenance. Record the next decision and recomputed overlap with that arm's contemporaneous RF top five. A trace can establish that evidence was present or cited. It cannot identify the standalone effect of feedback. An unstructured hypothesis without an explicit pre-measurement decision rule receives NOT EVALUABLE for support or refutation.

## Preselected matched comparison

Use target round 2, seeds 17 and 29, and the plain and knowledge-augmented DeepSeek arms. Choose all four slots by their fixed IDs, with no screening on outcomes, recommendation differences, or agent reasoning. Pair one real request chain per information condition. The primary paired endpoint is the best measured strength among the five selected mixes. Also report set and order changes, overlap, mean measured strength, regret, the comparator and verified evidence citations.

Reconstruct the RF fitted on the original 48 initial observations, before round 1 feedback, and freeze it across both information conditions. Use the original round 2 eligible-pool rule. The five previously tested mixes stay outside both candidate pools. The same fixed scores and RF tree disagreement generate exactly the same 24 RF leaders and 24 seeded alternatives for both conditions. Use the same model, system/tool schema, treatment-appropriate knowledge setting, seed, output budget and validators. Pass the extra round 1 measurements only as observed data in With Real Feedback. Any resulting analyst, knowledge or novelty summaries are downstream of those added observations. A direct comparator uses the five highest frozen RF scores on the same menu.

After the paired decisions are recorded, a separate evaluator reads the measured archive for their selections. It also derives the maximum over the fixed 48-entry menu for regret. The menu maximum and untested candidate strengths never enter the policy inputs. No new physical measurement takes place. An unavailable or broken checkpoint is NOT EVALUABLE and is not replaced by a selected alternative.

The evidence allows a descriptive within-checkpoint contrast under controlled inputs. With one model response chain per condition, differences can include response variation. Report four paired values, their mean, range and direction counts. Treat seeds as shared blocks and avoid an independence-based precision claim. A finding that evidence was visible, a changed action and a gain in measured outcome are separate results.

## Research source

The method adapts the common-prefix and matched-terminal-decision evaluation in LLMDictator, commit b5e221cb69d7c395d47f6f19e7a0f89bc62d6894. The cited work separates the acquisition of evidence from execution and net yield. Its historical policies and outcomes are outside this benchmark.

[LLMDictator study](https://github.com/fatcatMaoFei/LLMDictator/blob/b5e221cb69d7c395d47f6f19e7a0f89bc62d6894/research_exports/llm_position_mechanisms_20261121/api_revalidation/three_candidate/information_state/cross_stage/npm_conditional_verification/REPORT_CN.md)
