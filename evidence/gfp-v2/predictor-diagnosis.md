# GFP Predictor V2

## Why V1 struggled

**Sparse measured support and a strong edit-count shortcut.** The V1 dictionary encoder kept edit count, mutated positions and observed residue substitutions. An unseen substitution could retain a site indicator, but lost the identity and chemistry of the replacement. In the 360 archived validation rows, 330 had at least one substitution absent from that fold's training observations. The old encoder collapsed six rows into shared feature vectors across two folds. Only four of 355 multi-edit validation genotypes had an exact substitution pair co-observed in that fold's training data. No structural, evolutionary, chemical or explicit pairwise descriptors were available to either old estimator. [Support analysis](diagnosis/v1_validation_annotated.csv)

**Ridge ranked better with these inputs.** The mean across-seed Spearman was 0.430 for Ridge and 0.342 for RF. Restricting evaluation to fluorescent genotypes gave 0.217 and 0.184. For two-edit and three-edit variants, where edit count is constant, Ridge generally ranked variants better. RF's predicted scores correlated strongly with edit count across the full splits, between −0.913 and −0.636. This is consistent with RF using count as an easy early partition and sparse one-hot indicators limiting leaf support. Ridge pools additive evidence across edits through shrinkage. The observed contrasts support these explanations without assigning a causal share to the model architecture, the sample size, or their interaction. [Conditional ranking](diagnosis/v1_conditional_ranking.csv)

**The tail and the dim mode have different errors.** Among 93 dim validation genotypes (log fluorescence ≤2.5), both models systematically overpredicted. Mean absolute errors were 1.35 for RF and 1.30 for Ridge. Among 48 above-WT observations, both systematically underpredicted with MAEs of 0.55 and 0.66. Four-edit errors were larger than two-edit errors for both models. For RF, MAE was 0.47 when no substitution was unknown and 0.92 when three were unknown. Mutation count, phenotype and training support covary here, so these groups describe where error concentrates. [Error strata](diagnosis/v1_error_groups.csv)

**Additivity misses real fluorescence outcomes.** For 6,659 retired V1 multi-edit genotypes whose component single mutants all had archived measurements, summing log-fluorescence changes from WT yielded a median residual (observed minus additive) of −0.026 and MAE 0.272. In 418 cases classified as fluorescent by that additive calculation, measured fluorescence fell at or below 2.5. These retrospective phenotype deviations can reflect the known nonlinear fluorescence landscape and measurement effects. They do not show direct physical contacts or confirm a molecular mechanism. [Observed additive deviations](diagnosis/v1_additive_deviations.csv) · [Primary avGFP study](https://doi.org/10.1038/nature17995)

**High incumbents and the selection rule limited discovery.** The three archived pools contained 4, 0 and 3 candidates better than their starting incumbents. All seven lacked exact pairwise training support and six had at least one unknown substitution. The best RF ranks recorded for those seven ranged from 30 to 359. None entered Top-5 or the RF elite menu. Three LLM trajectories saw a qualifying variant through seeded menu exploration and did not select it. The random arm measured one such variant. The archived observations locate failures at opportunity, representation, ranking and menu access. They cannot uniquely identify why an individual forest assigned its rank. [Ranks and exposures](diagnosis/v1_incumbent_misses.csv) · [Archived V1 opportunity analysis](../gfp/quality/diagnosis/summary.json)

## Frozen comparison and independent evaluation

Ridge and RF use their original recipes. The new descriptors preserve the original site and substitution indicators and add substitution chemistry, normalized Kyte–Doolittle hydropathy changes, approximate charge changes, and coarse sequence-position summaries. Pair features encode exact allele/position pairs and chemical changes grouped by sequence distance. Histogram boosting uses at most 192 features chosen by training support. The hurdle first estimates the probability of fluorescence above 2.5 and then predicts the conditional mean in each mode. The fixed blend averages hurdle and pairwise Ridge. It uses no structures or external pretrained embeddings. [Hydropathy source](https://pubmed.ncbi.nlm.nih.gov/7108955/) · [Frozen experiment](../../gfp_v2/PROTOCOL.md)

Validation selected the blend in both budgets under mean above-WT AP and Ridge as the best legacy baseline. Validation AP was 0.246 versus 0.220 at 96 observations and 0.398 versus 0.383 at 2,048. The selection file records the code, partition and validation result hashes. The test command rejects changes before scoring the test genotypes. [Selection](prediction/selection.json) · [All development candidates](prediction/validation_summary.csv)

| Training observations | Model | Spearman | Pearson | MSE / MAE | Above WT AP | Above WT P@5 / P@10 | Upper-decile P@5 / P@10 | Best observed in Top-5 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 96 | Ridge | 0.400 | 0.447 | 0.715 / 0.666 | 0.197 | 0.150 / 0.200 | 0.100 / 0.125 | 3.758 |
| 96 | RF | 0.287 | 0.311 | 0.803 / 0.665 | 0.175 | 0.200 / 0.250 | 0.100 / 0.175 | 3.724 |
| 96 | Blend | 0.460 | 0.552 | 0.637 / 0.588 | 0.208 | 0.400 / 0.225 | 0.300 / 0.150 | 3.807 |
| 2,048 | Ridge | 0.780 | 0.833 | 0.279 / 0.389 | 0.394 | 0.600 / 0.550 | 0.600 / 0.525 | 3.862 |
| 2,048 | RF | 0.673 | 0.720 | 0.425 / 0.432 | 0.332 | 0.600 / 0.525 | 0.500 / 0.475 | 3.876 |
| 2,048 | Blend | 0.795 | 0.872 | 0.213 / 0.310 | 0.391 | 0.350 / 0.450 | 0.300 / 0.425 | 3.814 |

Values are means across four disjoint 512-genotype test folds. Precision averages tied ranks. Best measured score uses deterministic genotype-ID tie ordering. Both measures are predictive holdout outcomes, separate from iterative fixed-budget acquisition. [Full test table](prediction/test_summary.csv) · [Fold scores](prediction/test_metrics.csv)

**G1 MODIFY.** The blend improved regression error at both assay budgets. Clearer top-five enrichment at 96 did not establish superiority on the primary above-WT AP endpoint. Blend minus Ridge AP was +0.0115 with descriptive paired-block interval [−0.0230, +0.0393]. At 2,048, the difference was −0.0027 [−0.0386, +0.0284], and Ridge yielded better observed Top-5. Bootstrap resampling of genotypes within fixed test blocks also gives intervals crossing zero at both sizes. Greater data availability substantially improved both overall ranking and WT AP for all three evaluated predictors. The extra sequence features and hurdle fit helped regression, especially at 2,048, while high-tail ordering still depends on the objective and training budget. [Paired block contrasts](prediction/test_paired.csv) · [Conditional genotype comparisons](prediction/test_within_holdout_ap.csv)

**Risk is descriptive.** V2 risk tracks allele, site and exact-pair training support, scaled by training fluorescence standard deviation. RF uses tree disagreement. The blend's risk/error Spearman was 0.238 and 0.304, weaker than RF (0.463 and 0.711). Intervals calibrated on development residuals covered 90.9% and 91.3% of test outcomes at the two sizes, with mean widths of 3.015 and 1.549 log units. Wide intervals leave limited practical certainty around close calls, and this empirical coverage is no guarantee for new data or generated sequences. [Test risk metrics](prediction/test_summary.csv)

## Repeat evaluation without API calls

Run to fresh directories. Development selection must finish before the test command.

```sh
python -m gfp_v2.diagnose --output runs/gfp-v2-v1-diagnosis
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m gfp_v2.benchmark select --output runs/gfp-v2-prediction
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m gfp_v2.benchmark test --output runs/gfp-v2-prediction
```
