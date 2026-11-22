# AI use and source provenance

The assessment brief specified the GB1 prediction task, the four comparators, fixed-budget measured feedback, and scientific questions around exploitation, interaction, and exploration. OpenAI Codex proposed and implemented the separate concrete replay, operationalized the splits and metrics, wrote and tested code, generated figures, and drafted the report and interpretation. Codex used parallel Codex subagents for the GB1 analysis, the replay engine and the LLM policy. No external human review is recorded.

The LLM policy itself is an experimental intervention. The requested CodeFlow alias is `deepseek-ai/DeepSeek-V4.1-Flash`. A preliminary local backend probe preceded the switch to CodeFlow and contributed no experimental outcomes. The first CodeFlow transport probe used an incomplete alias and returned HTTP 404. The smoke and formal runs use the qualified alias. Its execution logs record exact prompts and decisions, model/provider identification and execution metadata. The closed-loop LLM receives public concrete features and the history of its own measured queries. The GB1 analysis is deterministic model fitting and evaluation with no runtime LLM calls.

## Sources

- My Paper governance, commit `6e094da3616f44d8bae2e650c8c83ca5ba748670`. [Mentor Playbook V3.0](https://github.com/fatcatMaoFei/mypaper/blob/6e094da3616f44d8bae2e650c8c83ca5ba748670/governance/MENTOR_PLAYBOOK_V3_0.md), [human-led research and AI disclosure](https://github.com/fatcatMaoFei/mypaper/blob/6e094da3616f44d8bae2e650c8c83ca5ba748670/governance/HUMAN_LED_RESEARCH_AI_ROLES_AND_DISCLOSURE.md), [writing style V7.0](https://github.com/fatcatMaoFei/mypaper/blob/6e094da3616f44d8bae2e650c8c83ca5ba748670/governance/ACADEMIC_WRITING_STYLE_V7_0.md).
- Christian Dallago et al., *FLIP: Benchmark tasks in fitness landscape inference for proteins*. [Paper](https://doi.org/10.1101/2021.11.09.467890), [official benchmark](https://github.com/J-SNACKKB/FLIP).
- Nicholas C. Wu et al. (2016), *Adaptation in protein fitness landscapes is facilitated by indirect paths*. [eLife](https://doi.org/10.7554/eLife.16965).
- I-Cheng Yeh (1998), *Modeling of strength of high-performance concrete using artificial neural networks*. [Cement and Concrete Research](https://doi.org/10.1016/S0008-8846(98)00165-3), [archival dataset and license](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength).
- DeepSeek. [Native tool calling](https://api-docs.deepseek.com/guides/tool_calls/), [Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion/), [CodeFlow gateway](https://codeflow.sh). Gateway-returned model IDs are recorded directly. A weight snapshot is unavailable.

Before submission, the human author should review protocol choices, validate the result-to-claim links, and approve the interpretation and disclosure. This document records actual participation and makes no claim of completed author review.

## Added validation and execution history

The user requested additional models after the primary DeepSeek run. The two-model extension introduced Claude Sonnet 5.5 through CodeFlow's Anthropic Messages endpoint. A subsequent extension introduced CodeFlow GPT-6 Luna through OpenAI Responses and a local Qwen3-4B-4bit MLX checkpoint. Original seeds, splits, budgets, Python tools and comparison endpoints were retained. Per-model source and protocol freezes precede their own formal execution. Relative chronology is recorded in the Git history and each evidence directory.

CodeFlow throttled GPT late in the initial formal run. The failed attempt is archived under evidence/gpt-rate-limited. The recovery ledger identifies successful rounds and provider responses reused under matching digests and the missing calls that ran live. The accepted formal records incorporate each provider response once. Qwen's unforced seed-7 variant failed its tool contract in one arm. The corrected smoke used the prefilled native evaluator envelope in protocol-qwen.md. The first formal attempt then stopped at another schema failure. The transport amendment in protocol-qwen-structural.md adds field-domain restatements before restarting the formal arms. The generated tool arguments and decisions remain model outputs.

The endpoint identifies the served model alias. The gateway does not provide independently checkable remote weights. Qwen's local weights use a pinned model revision and verified file hash. Token usage follows each model/runtime's native tokenizer and cache accounting. No cross-provider dollar cost is inferred from token counts.

- Anthropic [native tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools) and [Messages API](https://platform.claude.com/docs/en/api/messages).
- OpenAI [GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna) and [Responses API](https://developers.openai.com/api/reference/resources/responses/methods/create).
- Alibaba Qwen [base Qwen3-4B checkpoint](https://huggingface.co/Qwen/Qwen3-4B), [pinned MLX 4-bit conversion](https://huggingface.co/mlx-community/Qwen3-4B-4bit/tree/4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25) (Apache-2.0), and [MLX LM](https://github.com/ml-explore/mlx-lm).
- Federal Highway Administration materials heuristics [FHWA-HRT-04-150 Chapter 9](https://www.fhwa.dot.gov/publications/research/infrastructure/pavements/pccp/04150/chapt9.cfm) and [FHWA-HRT-04-122 Chapter 5](https://www.fhwa.dot.gov/publications/research/infrastructure/pavements/pccp/05042/05.cfm). Observation-derived knowledge is recalculated from each arm's revealed history.

## Scientific Decision Auditor

The user requested a separate audit after the main model experiments. Codex implemented the archive-linked trace reconstruction, four paired DeepSeek feedback checkpoints and a model-only comparator. The auditor adapts the separation of acquired information, action changes and realized outcome studied in [LLMDictator](https://github.com/fatcatMaoFei/LLMDictator/tree/b5e221cb69d7c395d47f6f19e7a0f89bc62d6894). The audit uses the concrete benchmark's frozen artifacts and independent tool traces. Its requests and measurements are recorded separately from the 450 formal main-study queries.

## GB1 retrospective verification at review baseline 0e52dd8

Codex audited the fixed FLIP data provenance, reran the existing regression benchmark, and drafted a standalone GB1 retrospective report from aggregate outputs. The verification ran no LLM providers and collected no new measurements. All previous model-adapter logs, materials results, decision audit outputs, and GB1 benchmark artifacts were retained. The acceptance record specifies the boundary of the GB1 evidence and distinguishes it from the historical materials results. [Frozen baseline](evidence/gb1-review/base_snapshot.json) · [Verification receipt](evidence/gb1-review/verification.json)

## GFP fluorescence study and compatible API onboarding

The main agent experiment now uses archived GFP fluorescence measurements. The protocol fixes the GFP reference, score, paired splits, budgets, accepted outputs, and model cells. In the measured study, model output supplies explicit mutation proposals and calls native scoring tools. The fitted predictor returns estimates and the replay oracle releases only measurements for submitted IDs. Remote native model trials retain DeepSeek, Claude Sonnet and GPT served IDs; local Qwen records its pinned weight revision and actual local responses. The configured-api setup and CSV demonstration use a separately tested OpenAI-compatible native-tool transport. Example researcher input and returned values come from the public fluorescence archive. No new laboratory measurements were made.

The compatibility profile used for the demonstration is stored locally. Its public verification receipt shows served model and response identifiers and confirms tool execution. It contains no API key. The formal measurements, compatibility demonstration and separate paired feedback audit have separate directories and logs. Frozen formal code hashes refer to the native experiment implementation in commit 47d1b789, prior to the optional adapter's increased response budget.

- [GFP primary study](https://doi.org/10.1038/nature17995) and [original data](https://doi.org/10.6084/m9.figshare.3102154.v1).
- [ProteinGym 1.3 frozen release](https://doi.org/10.5281/zenodo.15293562).
