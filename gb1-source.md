# FLIP GB1 retrospective protocol

**Question.** How accurately do additive and non-additive regressors predict the
published GB1 assay score when fit on variants with at most three substitutions
and evaluated on the official held-out four-substitution variants?

**Hypothesis.** A pairwise residue model or a random forest may outperform
an additive ridge model on positional extrapolation. Comparison is descriptive
for this one benchmark split. It is separate from the closed-loop study.

**Source.** Dallago et al., *FLIP: Benchmark tasks in fitness landscape inference
for proteins*, NeurIPS Datasets and Benchmarks (2021). Use the official archive
from revision `62cace8735f5610e2743cf06ce0f944b37fffaa6`, member
`splits/three_vs_rest.csv`. The pinned archive defines the split and exact row
counts. The scored assay in Wu et al. (2016) measures enrichment relative to
the wild type, with a wild-type score of one and larger values indicating
greater enrichment. Source measurements reflect binding and protein stability.
The CSV uses the 265-residue FLIP construct representation derived from
PDB 5LDE chain A. The model encodes only the four published variable coordinates.
[FLIP paper](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/2b44928ae11fb9384c4cf38708677c48-Abstract-round2.html)
[FLIP GB1 source and license](https://github.com/J-SNACKKB/FLIP/tree/62cace8735f5610e2743cf06ce0f944b37fffaa6/splits/gb1)
[Wu et al. assay](https://doi.org/10.7554/eLife.16965)
[FLIP license](https://github.com/J-SNACKKB/FLIP/blob/62cace8735f5610e2743cf06ce0f944b37fffaa6/LICENSE)
[PDB 5LDE construct record](https://www.rcsb.org/structure/5LDE)

The original Wu et al. data are CC BY 4.0. The prepared FLIP files are
AFL-3.0. The unchanged ZIP is downloaded and verified by SHA-256. It lives in
the local cache, and the evidence directory contains aggregate analysis only.
Counts refer to the pinned artifact and may differ from the publication's
summary table.

## Fixed analysis

- Train on source `train` rows lacking a validation marker. Use source
  validation rows for selection within each model family. Evaluate the selected
  models once on the source test split, using the weights from the training
  rows. Audit schema, residues, coordinate variation, duplicates, partition
  overlap, target missingness and observed mutation depth. Feature coordinates
  and alphabet are fixed in advance.
- Compare a train-mean baseline, additive Ridge on position-specific
  amino-acid indicators, pairwise Ridge with six position-pair indicator
  blocks, and Random Forest on the additive indicators. Ridge uses
  `alpha ∈ {0.1, 1, 10, 100}`. Forest uses 160 trees, seed 1701,
  `max_features = 0.7`, and `min_samples_leaf ∈ {1, 3, 8}`. Select
  by maximum validation Spearman, then minimum validation MSE, then grid
  order. Both Ridge variants use the sparse LSQR solver.
- Report Spearman, Pearson, MSE, AP for assay score strictly above one,
  and precision and recall for the top 5% by model score. Compute each
  top-5% measure as an expectation under uniform tie-breaking at the cutoff.
  Report positive prevalence, so AP has a chance reference. Summarize
  test residuals by measured score band and the error rate among the highest
  predicted 5%. Make no further model changes based on the test results.
- To test whether knowledge of individual substitutions transfers, compare
  measured four-substitution outcomes with the product of measured training
  singles, restricted to test rows whose four singles are present in
  training. Disclose coverage, absolute errors, and observed versus expected
  direction relative to wild type. Treat a support-conditioned subset as
  descriptive. A nonzero discrepancy can reflect interaction and assay noise.
- Measure positional predictive importance by retraining the selected forest
  with each site omitted, holding its settings fixed. Compare validation
  metrics on the original measured rows. This importance is model-dependent.

## Exploratory support check added after the test run

The locked four-edit transfer check returned zero support. To establish
whether this is specific to the official test partition, apply the same
training-single reference to the measured two- and three-edit rows in train
and validation, by depth and partition. This check leaves the formal
comparison and four-edit estimate as executed. The lower-depth groups are
descriptive. Fewer than ten covered rows are reported as a coverage count
with outcome summaries suppressed.
