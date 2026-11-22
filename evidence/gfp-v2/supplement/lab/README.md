# GFP V2 local assay-platform handoff

The file exchange starts from the two live V2 recommendation lists. The exercise uses a simulated platform with archive-backed returns. No laboratory hardware was connected.

| Batch | Result | Evidence |
|---|---|---|
| Round 1 novel variants | Five submitted. The archive has no matching fluorescence records, so the simulator marked all five failed with empty fitness fields. | [Submitted CSV](novel/submitted.csv) · [Return CSV](novel/platform_return.csv) · [Audit ledger](novel/ledger.json) |
| Round 2 shared pool | Two submitted and returned from ProteinGym 1.3. `E235K` scored 3.6528 and `K3E:A206T` scored 3.7228. Three samples remain pending. | [Submitted CSV](archived/submitted.csv) · [Return CSV](archived/platform_return.csv) · [Audit ledger](archived/ledger.json) |
| Continued V2 decisions | The exact normalized assay CSV refitted pairwise Ridge on 99 observations. It generated a no-API Top-5 preview and separately went through live DeepSeek tool scoring and selection. | [Normalized CSV](archived/normalized.csv) · [Top 5](top5-after-return.csv) · [Agent trace](agent-after-return/agent.json) · [Hypothesis audit](agent-after-return/hypothesis_audit.json) |

## Third-party workflow

Use the project's Python environment. The fast demonstration and personal candidate forecast run without an API key. A user can configure a live model profile with `run.py setup` and use that profile for V2 continuation.

```bash
python run.py demo --dataset gfp --output runs/gfp-first
python run.py setup --profile my-model
python -m gfp_v2.lab_platform forecast \
  --reference my-gfp.fasta --measurements my-measurements.csv \
  --candidates my-variants.csv --model pair_ridge --top 5 --output runs/my-top5.csv
```

The reference FASTA must match the pinned avGFP sequence. The measurement file uses `mutant,fitness` with optional `sequence` and requires WT and at least three further measurements. Candidate files require canonical one-based `mutant` and accept optional `sequence`. Prediction-only output contains full sequences and ranks.

```bash
python -m gfp_v2.open_world init --measurements my-measurements.csv \
  --model pair_ridge --model-profile my-model --candidates my-variants.csv --output runs/round-01
python -m gfp_v2.lab_platform create --reference my-gfp.fasta \
  --candidates runs/round-01/recommendations.csv --namespace my-study-round-01 --batch runs/my-batch
python -m gfp_v2.lab_platform submit --batch runs/my-batch
python -m gfp_v2.lab_platform export --batch runs/my-batch --output runs/submitted.csv
python -m gfp_v2.lab_platform ingest --batch runs/my-batch --results my-assay-return.csv
python -m gfp_v2.lab_platform normalize --batch runs/my-batch --output runs/new-measurements.csv
python -m gfp_v2.open_world continue --state runs/round-01/state.json \
  --model pair_ridge --model-profile my-model --measurements runs/new-measurements.csv \
  --candidates my-variants.csv --output runs/round-02
```

The return file columns are `batch_id,sample_id,mutant,sequence,status,fitness,scale,source,source_ref`. Use `log10_fluorescence` on the pinned assay scale. `returned` requires a finite value. `failed` requires an empty value and provenance. The declared `source` is `external_assay`, `public_archive`, or `simulation`. Normalization exports only returned assay or archive rows. A value labeled `simulation` is excluded from the measured continuation CSV. The CLI checks identity, consistency and declared provenance. Independent verification of an external laboratory measurement requires its source record.

For the checked-in simulation, use `simulate --batch runs/my-batch --output runs/returned.csv` after submission. This route only returns values matched to the fixed published archive. Repeat live continuation with a configured profile, or use `--replay agent-after-return/agent.json` with that archived decision's verified profile and exact input files to reconstruct the recorded run without API calls. The [verification receipt](receipt.json) reports the archive hashes and repeat checks.
