# Experiment protocol

## Questions

Can a tool-backed language model improve a fixed-budget search over measured candidates when it receives the same model scores available to a greedy policy? Does adding documented domain knowledge and observation-derived summaries alter what it tests and what it discovers? In a separate retrospective task, how well do position-aware sequence models predict four-substitution GB1 records from lower-order examples?

The four-arm sequential study uses 28-day concrete strength. The GB1 study estimates prediction performance on historical measurements and operates independently of the concrete study. Neither dataset provides evidence about the other task's decision policies.

## GB1 prediction

Use the pinned FLIP official `three_vs_rest` benchmark. Use FLIP's designated fitting and validation labels, and evaluate the final test split separately. Compare predeclared Ridge with position one-hot encoding, Ridge with additional position-pair indicators, and Random Forest with position one-hot encoding. Report validation and test Spearman, Pearson, MSE, and discrimination above wild-type relative binding fitness of 1. The official split contains no virtual experiment pool for this study. Aggregate residuals and contrasts in measured lower-order combinations describe prediction failure and nonadditivity, respectively. Predictive benefit from interaction features and direct empirical nonadditivity answer different questions. Use aggregate output and avoid publishing variant-level scores, design recommendations, or trained models.

## Four-arm measured sequential study

- Source is Yeh's UCI Concrete Compressive Strength. Restrict to 28-day measurements to hold cure age fixed. Collapse repeated identical ingredient compositions to their median measured MPa before assigning each unique mix to a split. The source contains 425 rows at age 28 and 416 unique 28-day compositions.
- Predeclared paired split seeds are 17, 29, and 43. Each seed partitions mixes into 48 initial observations, 96 validation mixes, 176 eligible replay mixes and 96 final prediction-test mixes. Assignments are paired across policies. Use the first seeded initial mix as a reference and the strongest initial observation as the main improvement baseline.
- Each arm starts with an independent copy of the same initial observations and the same eligible IDs for the seed. Run exactly three rounds of five distinct eligible mixes per arm. After an entire accepted batch is selected, only its labels are released to that arm. Refit the fixed Random Forest on its new observations before the next decision. Retain Ridge as a prediction reference. Compute validation and final-test metrics outside the policy view. Stop after 15 queries, or record a failed arm explicitly if data validation, model execution or the decision contract prevents the planned run.
- A uniformly samples unseen eligible IDs. B takes the five best Random Forest predictions. C and D receive the live rank and uncertainty plus measured observations, with a menu cut to the 24 highest RF ranks and 24 seeded non-elite candidates. Both use the same decision schema and checks. D also receives materials knowledge and summaries derived only from its own exposed observations. The shortlist contains the arm's own greedy baseline and may diverge across arms after the first round.
- The primary outcome is gain in the running best strength over the initial best at 15 queries. Interpret C minus B as the incremental effect of the complete LLM policy relative to greedy. Interpret D minus C as the effect of the combined knowledge context under the tested agent. Report all seed-paired deltas, including unfavorable and zero differences.
- Secondary outcomes are per-round realized best and running best, improvement over the reference, hits and hit fraction above the pool-specific 90th percentile of measured strength, and queries to first hit (right-censored if none). Compute thresholds outside policy execution and exclude them from decision context. Selection overlap with the arm's own greedy top five and mean rank reveal deviations from exploitation. Distance to nearest arm-observed mix and water-to-binder ratio describe selected candidate distribution. Predeclared failure analysis uses measured-minus-predicted residual extremes.
- Scientific interpretation is descriptive with three paired seeds from one archival dataset, which can reuse mixes across different seeds. Fixed suggestions are scored by logged measured outcomes. Knowledge can plausibly improve, hurt, or leave performance unchanged. Neither a lower rank overlap nor a positive difference on one seed independently identifies the mechanism.
- A fast run uses one non-formal seed and one round. Confirm data checks, real structured LLM calls and validation behavior before the formal run. Its outputs stay separate from the formal evidence.

## Research governance

The assessment owner supplied the research questions and required comparisons. Codex adapted scope, specified operational decisions above, and implemented the study. Human review and approval of scientific claims and submission wording are pending. See [AI disclosure](disclosure.md).

## Execution adapter

Both policy arms use CodeFlow's OpenAI-compatible Chat Completions at `https://codeflow.sh/v1` with the requested model alias `deepseek-ai/DeepSeek-V4.1-Flash`. Inference temperature is zero and explicit thinking is disabled. The LLM issues a native function request to evaluate its proposed shortlist, then returns the final ordered batch with evidence references. The same policy checks and bounded validation retries apply in both arms. Calls, usage, response model ID and tool results are recorded per round. A separate smoke run established successful live execution before this protocol was frozen for formal acquisition.
