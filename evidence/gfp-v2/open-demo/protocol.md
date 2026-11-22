# GFP V2 open-world workflow demonstration

This is a software and provenance check using the archived V1 user-workflow fixture. It is outside the sealed V2 scientific experiments. The demonstration deliberately uses the fixed `pair_ridge` recipe and the verified compatible model profile already used for GFP. It does not choose a predictor based on demo outcomes.

Use the first-round historical WT and 95 other measurements from `evidence/gfp/quality/hypothesis/round-01/imported_measurements.csv`, and an ID-only reference pool of the first 20 unobserved IDs by SHA-256 of `gfp-v2-open-demo|<id>`. Scores are fitted exclusively to user-imported measurements. The first decision is saved before any follow-up measurements are read. For the continuation fixture, look up the first recommended ID in output order for which the archive already reports a measurement. If none are available, take the first unobserved ID from the ID-only reference pool and import it as an external observation. Never assign an archive measurement to an absent variant. Only explicit returned measurements are assimilated. All other forecasts remain pending.

The files record actual native model output. Offline verification uses the frozen provider traces and validates request, response, tool and decision hashes. This demonstration does not establish design improvement or a feedback treatment effect.

## Observed delivery

Both rounds used the served `deepseek-ai/DeepSeek-V4.1-Flash` model and generated five recommendations each. All first-round recommendations were legal off-menu designs. The fixed archive fixture had no measurements for the five selections, so the follow-up imported `T38S:F99S:K126R` (3.54678934565) as an external observation. The 96 observation ledger grew to 97 and the primary first-round comparison against an incumbent of 3.89651251107 stayed pending. In round two, the agent continued its high-prediction strategy, scored a new variant, and selected five from the displayed menu. Its citations did not include the newly imported row. Both realized scientific gain and effects of feedback are unestablished in this demonstration.

The failed preflight traces preserve a hidden reasoning cap failure, a proposal with noncanonical edit order, and continuation attempts that reused previously observed or pending IDs. The accepted continuation used exact aliases for the losslessly represented candidate menu. It preserves a separate `NEW` path for new genotypes.

## Offline replay

From the repository root, have a verified compatible profile for the trace's served model and CodeFlow endpoint. Pass its name as `--model-profile`. Replay does not require the API key. Run into new directories.

```bash
PY=.venv/bin/python
D=evidence/gfp-v2/open-demo
I=evidence/gfp/quality/hypothesis/round-01/imported_measurements.csv
M=gfp-third-party
$PY -m gfp_v2.open_world init --measurements "$I" --candidates "$D/candidate_features.csv" --model pair_ridge --model-profile "$M" --replay "$D/round-01/agent.json" --output runs/gfp-v2-open-replay/round-01
$PY -m gfp_v2.open_world continue --measurements "$D/returned_measurement.csv" --state runs/gfp-v2-open-replay/round-01/state.json --candidates "$D/candidate_features.csv" --model pair_ridge --model-profile "$M" --replay "$D/round-02/agent.json" --output runs/gfp-v2-open-replay/round-02
```

The accepted observations, immutable decisions and field-level hashes are in [the replay receipt](receipt.json).
