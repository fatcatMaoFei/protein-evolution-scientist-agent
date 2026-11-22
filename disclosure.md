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
