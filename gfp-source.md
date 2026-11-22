# GFP assay and prediction evidence

## Assay and pinned sources

The target is the native fluorescence of isolated Aequorea victoria GFP. Sarkisyan et al. measured fluorescence across an avGFP substitution library. Each archived score is a measured fluorescence-derived label. Predictions from this project remain a distinct field.

| Input | Pinned version or identifier | SHA-256 | Terms |
|---|---|---|---|
| ProteinGym substitutions ZIP | 1.3, Zenodo 15293562 | `3a83766254ac9ac9984ec25cb73c6e010ea4418f5e35f143933e6b6e6473b921` | MIT, as recorded on the release |
| GFP assay member | `DMS_ProteinGym_substitutions/GFP_AEQVI_Sarkisyan_2016.csv` | `05b97d36a23363331b271edc56f1496b9a539109fd006e65532f91e9903a0838` | ProteinGym release |
| Substitution reference catalog | `DMS_substitutions.csv`, same release | `a8f498011532a74aa9fe556a50555a75e928c5837d19c06a87592ae04049b308` | ProteinGym release |
| Original amino acid measurements | Figshare v1 file 4820647 | `5581230b9888f9960708a4769ce1c9b4453561f0b5f0374525445b4a984d54d5` | CC BY 4.0 |

[Sarkisyan et al. 2016](https://doi.org/10.1038/nature17995) · [Original data v1](https://doi.org/10.6084/m9.figshare.3102154.v1) · [ProteinGym v1.3](https://doi.org/10.5281/zenodo.15293562) · [ProteinGym reference schema](https://github.com/OATML-Markslab/ProteinGym#dms-assays)

The release has 51,714 GFP variant rows with no missing fields, duplicated mutation strings, or duplicate full sequences. The ProteinGym reference defines a 238-amino-acid GFP sequence, with 1-indexed positions. Reconstructing every listed variant from its substitution string reproduces its archived full sequence exactly. The project limits active candidates to 1–4 substitutions, leaving 35,584 eligible variant measurements. Its wild-type identity is `WT`. The original data's single aggregate WT row supplies the measured reference score, 3.7192121319 on the same log fluorescence scale. WT contributes one observation to every initial fold.

ProteinGym's `DMS_score` is the curated log-transformed fluorescence measurement. The benchmark phenotype is `mean_medianBrightness_per_aaseq` with orientation +1. Its archived fluorescence class uses `DMS_score > 2.5`, verified against every row. The primary optimization outcome is the continuous value. For a provenance cross-check, the original table's `S`-prefixed substitution tokens map to ProteinGym positions with offset +2. All 51,714 variant identifiers match after normalization, and 51,054 numeric scores match within 1e-9. Use the published ProteinGym values for its mutant rows and the original source only for WT. The code does not reconcile the 660 nonidentical scores into replacement labels.

## Prediction and split

Each seed draws a label-independent random permutation of a sorted variant identity list. The formal fold uses 96 initial observations including WT, 120 validation records, 384 query-pool records and 120 independent final prediction records. Fold roles never overlap within a seed. Policies for a seed share the initial observations and candidate pool, and maintain separate measurement ledgers. Each seed is analyzed as an independent paired comparison.

The predictors encode edit count, substituted-position indicators and exact allele indicators through a sparse vectorizer fitted to each training set. Ridge uses alpha 8 and the LSQR solver. Random Forest uses 128 trees, minimum leaf size 2, full feature search and one worker. Its uncertainty proxy is the standard deviation across tree predictions. Predictions use the archived log-fluorescence scale.

Report Spearman, Pearson, MSE, precision and recall at k for a measured top-decile definition, plus average precision for both high fluorescence and the archive's fluorescence class. At the predicted cutoff, compute expected positive hits under uniform tie-breaking. Rank correlations are undefined when either input is constant. The split-specific top decile is computed by the evaluation function from held-out measurements for reporting.
