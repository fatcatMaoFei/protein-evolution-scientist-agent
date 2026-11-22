# Mutation count, site preference, and measured GFP geometry

This retrospective supplement reads the archived V2 predictions and completed trial records. The original model choice, decision logs, assay outcomes, and protocol remain frozen. The structure correction is an exploratory comparison with an additional calibration fit. It does not select a replacement model or establish a prospective acquisition benefit. [Reproducible tables and provenance](landscape/) · [Four-panel overview](landscape/landscape_overview.png)

## Single, double, and higher-order substitutions

The same pinned assay gives the following observed distributions. The 96-observation model column uses sealed V2 test genotypes within each edit count. All measurements are log10 fluorescence. [Archive distributions](landscape/archive_by_mutation_count.csv) · [Error and Top-5 metrics by count and model](landscape/prediction_error_by_mutation_count.csv)

| Substitutions | Archive genotypes | Measured median | Above WT | V2 test genotypes | V2 low-budget MAE | V2 ample-budget MAE |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1,084 | 3.649 | 21.0% | 19 | 0.332 | 0.184 |
| 2 | 12,777 | 3.593 | 18.4% | 741 | 0.423 | 0.241 |
| 3 | 12,336 | 3.499 | 12.1% | 714 | 0.617 | 0.332 |
| 4 | 9,387 | 3.144 | 6.7% | 574 | 0.773 | 0.376 |

Single-edit test performance has limited support. Only 19 singles remained in the sealed test blocks after V1 evidence retirement, so its MAE and precision estimates are less representative than the multi-edit strata. Archive rates describe an observational genotype distribution and may reflect both mutation burden and library composition. They are not a matched estimate of the effect of making one additional mutation.

For multi-edit genotypes with all component singles measured, the independent-additive reference was WT plus the sum of each single's measured log-fluorescence change relative to WT. Across 11,954 doubles, 11,041 triples and 7,896 quadruples, MAE against the additive reference rose from 0.145 to 0.273 to 0.492. Median observed-minus-additive changes were −0.011, −0.028 and −0.047. There were 203, 743 and 1,115 cases respectively that the additive model classified as fluorescent (greater than 2.5) and the experiment measured as dim. This supports phenotypic nonadditivity relative to the stated benchmark. The assay floor, saturation, shared mutations and ascertainment of matchable singles limit mechanistic interpretation. [Matched data and expected values](landscape/single_multi_nonadditivity.csv) · [Matched summaries and denominators](landscape/single_multi_summary.csv)

## Site prevalence, preferences, and true outcomes

Uniform expectations are calculated within each original eligible round roster. For site comparisons, the stricter control preserves the number of variants selected in each mutation-count stratum and samples uniformly within that count. Archive enrichment also conditions on mutation count. These expectations are descriptive and may still reflect within-count variant composition, co-edits, predictors, and prior arm decisions. The peer designs and genotype frequencies are archived independently of measured outcomes.

| Closed policy | Doubles selected | Expected under uniform roster sampling | Above-WT hits among selected doubles |
|---|---:|---:|---:|
| A random | 47 / 120 | 50.1 | 11 |
| B greedy predictor | 99 / 120 | 47.1 | 27 |
| C risk acquisition | 99 / 120 | 47.3 | 22 |
| D LLM | 102 / 120 | 47.3 | 28 |
| E knowledge LLM | 99 / 120 | 47.3 | 24 |

The LLM double-edit preference largely matched both non-random model-based selectors. In the combined D and E selections, K26 was most frequent, appearing 15 times versus a degree-matched roster expectation of 8.84. Three selected variants containing K26 exceeded WT, while 65 of 678 archive variants containing K26 exceeded WT, 0.80 times the degree-matched archive expectation. K158 appeared 10 times versus 5.47 expected. Seven of those selections exceeded WT, and 240 of 910 archive variants containing K158 exceeded WT, 2.22 times its degree-matched expectation. M153 also appeared 10 times and none of those selections exceeded WT, despite positive archive enrichment. These three examples separate recommendation frequency, realized hits, and background archive association. They leave the residues' causal functional importance unresolved. [Per-policy strata and real hits](landscape/selection_by_mutation_count.csv) · [Site preferences for every strategy](landscape/selection_sites_adjusted.csv) · [Archive site rates and held-out prediction error](landscape/archive_and_holdout_sites.csv)

## Structure source and incremental diagnostic

The official **1GFL** deposit is a 1.90 Å crystal structure of recombinant *Aequorea victoria* GFP. Chain A has a 238-residue SEQRES alignment to the assay reference. Differences were verified at positions 1 (assay M, structure A), 64 (L, F) and 80 (Q, R). Those sites and the eight positions without Cα coordinates (231–238) were excluded from direct geometric features. The remaining 227 sites have matching sequence and coordinates. The coordinate snapshot is deposited alongside the analysis as gzip preserving the original official file bytes. Exact uncompressed checksum and mapping are recorded in the provenance. [RCSB record](https://www.rcsb.org/structure/1GFL) · [Primary structure paper](https://doi.org/10.1038/nbt1096-1246) · [wwPDB CC0 terms](https://www.wwpdb.org/about/usage-policies) · [Position map](landscape/structure_position_map.csv) · [Provenance](landscape/provenance.json)

Features are Cα neighbors within 8 Å excluding immediately adjacent sequence positions (separation ≤2), distance to the Cα centroid of the deposited 65–67 residues, and minimum Cα separation between edited sites. They describe the static deposited geometry and do not provide solvent accessibility, mutant structures, biochemical attribution, or a classification of a site as buried. Within triple-edit 96-observation test genotypes, mean neighbor count had Spearman 0.182 with absolute prediction error and −0.285 with measured fluorescence. Among quadruples, minimum distance to the 65–67 centroid had Spearman −0.202 with error. These weak associations describe the measured cohort. [Structural associations within degree](landscape/structure_stratified_correlations.csv)

To test incremental usefulness, the fixed Ridge residual correction (α=32) used only development prediction errors. It compared frozen prediction/risk, edit count, and unseen-edit/site controls with or without the geometry columns. The imputer and scaler fit development data only. The correction was evaluated on the same four sealed test blocks used by the already-finished V2 evaluation. This is exploratory reuse of existing test outcomes, with no independent confirmation of the auxiliary result. No acquisition policy or frozen Predictor V2 was changed. [Individual fold results](landscape/structure_fold_comparison.csv) · [Model comparison means](landscape/structure_comparison_means.csv)

| Training budget | Residual controls AP | Plus structure AP | Residual controls MSE | Plus structure MSE | WT P@5 controls / structure |
|---:|---:|---:|---:|---:|---:|
| 96 | 0.2152 | 0.2430 | 0.6098 | 0.5643 | 0.40 / 0.35 |
| 2,048 | 0.3795 | 0.3754 | 0.2067 | 0.2063 | 0.50 / 0.50 |

The 96-observation AP difference was positive in all four blocks (+0.007 to +0.045), while Top-5 degraded and the 2,048-observation AP contrast was slightly negative. This supports further prospective study of geometry under sparse labels. It leaves consistent Top-K or decision-level benefit unproven.

## Offline reproduction

```bash
python -m gfp_v2.landscape --output runs/gfp-v2-landscape
python -m pytest -q tests/test_gfp_v2_landscape.py
```

The tool revalidates the pinned coordinate snapshot, reads frozen V2 rosters and predictions, and writes all tables and the figure to a new directory. It uses no network or provider calls with the assay cache available.
