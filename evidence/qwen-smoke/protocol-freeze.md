# Four-model robustness extension

This extension retains the archived DeepSeek primary study and Sonnet validation. Add CodeFlow GPT-6 Luna and a pinned local Qwen3-4B model, each with knowledge OFF and ON. Three independent paired seeds compare eight model-by-knowledge cells plus the same shared random and RF-greedy controls. This extension was specified after both earlier studies and before the two new formal runs.

The fixed controls in [protocol-factorial.md](protocol-factorial.md) apply to both new models. Use the same 3 × 5 budget, fitted RF and disagreement, split identities, deterministic shortlist rules, observed-evidence formats, score-tool semantics and output critic. Use seed 7 for separate smoke runs. Foreground further exploratory analyses only if their results are explicitly labeled.

Report each model's per-seed ON minus OFF primary gain and top-decile hit count. Compare every cell with shared RF greedy, and report each model's knowledge interaction relative to DeepSeek as a difference of the within-model knowledge effects. Include paired-seed ranges and exact observations. Inspect whether departures from each arm's current top five use model-rank 6–24 or the seeded lower-rank menu. The decoder, tokenizer, function framing and provider route are bundled with model identity. Therefore between-model comparisons identify performance of tested deployments.

The GPT arm targets CodeFlow `gpt-6-luna` using the OpenAI SDK Responses endpoint and native function calls. The proposal requests the same score-tool schema. The final decision uses the same fields as the main agent. Set reasoning effort to none and the common 512-token maximum. Preserve request and response usage.

The Qwen arm targets `mlx-community/Qwen3-4B-4bit` at `4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25`, with MLX LM 0.31.2, temperature zero, disabled thinking and a 512-token maximum per generation. The native template and tool-call envelope are part of the recorded transport. Inference is local. Record local tokenizer input and generated token counts with the exact rendered messages and raw output.
