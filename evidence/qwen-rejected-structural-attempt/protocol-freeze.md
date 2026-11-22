# Qwen native transport freeze

This supplement inherits the four-model protocol, data, split, tool contract and endpoints in protocol-four-model.md. It records the Qwen transport refinement after the separate seed-7 smoke attempt and before formal seeds 17, 29, and 43.

The first smoke implementation permitted Qwen to omit the native tool. The revised transport opens the native named evaluator envelope through an assistant prefill. Qwen generates the full analyst, hypothesis, evidence references, and candidate arguments. Log the prompt prefill separately from generated tokens. On a valid tool call, execute the common Python evaluator on the input aliases and return its actual result. Request the final ordered batch through a second MLX generation using the shared critic. Both knowledge conditions use the same template, model revision, generation limit and checks. The first attempt and its failure remain archived.

The fixed Qwen checkpoint is mlx-community/Qwen3-4B-4bit at revision 4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25. Its model.safetensors SHA-256 is e240c0bdc0ebb0681bf0da0f98d9719fd6ebe269a3633f81542c13e81345651d. Use 512 generated tokens per inference, temperature zero, thinking disabled and one MLX generation at a time. The native request and raw generated token trace are retained for exact replay.
