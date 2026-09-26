# Experiment 1.1 — A=0.4 GI without NOW, minimum gain zero

This sensitivity run removes the NOW anchor and changes minimum impurity
decrease from 0.0005 to 0.0 relative to the final Experiment 1 GI configuration
saved in:

`results/experiment_1_5000V3clean__moreOFFSETS2_patients_A=0.4_minS20_minL10_minGAIN0.0005/`

From the repository root, launch training explicitly:

```bash
python3 experiments/experiment_1_1_gi_without_now.py --run
```

The separate output directory is:

`experiments/results/experiment_1_1_gi_without_now_A0.4_minGAIN0/`

Existing directories, including partial runs, are rejected. The earlier
`experiment_1_1_gi_without_now/` and `experiment_1_1_gi_without_now_minGAIN0/`
results remain untouched, as do the partial results in
`experiment_1_1_gi_without_now_A0.4_minGAIN0.0005/`.

## Configuration

- Candidate sampling rate: **0.4**.
- Maximum depth: **3**; minimum split size: **20**; minimum leaf size: **10**.
- Minimum impurity decrease: **0.0**; criterion: **MSE**, regression mode.
- Random state and split-sampling seed: **42**, inherited from the saved run.
- Dynamic anchors and learned missing directions: disabled; verbosity: **1**.
- Aggregations: **mean, sum, max, count**, reused from the saved GI model.
- Birth Date offsets: `[None, 0.7, 1, 2, 3, 4, 5, 6, 12, 18]`.
- NOW is removed only from `anchors_list` and `anchor_offset_map`. Full Timeline
  remains available once per feature/aggregation; the input data are unchanged.
- V3-clean-FA longitudinal data and GI labels are loaded from the baseline's
  saved paths and verified against its input hashes.
- The exact saved 3,500-training / 1,500-test patient split is reused, including
  the original first-seen patient order through Experiment 1's subsetting helper.

Baseline configuration, split, fitted GI model, and GI summary are pinned by
SHA-256. All constructor parameters are checked against the saved model. The
candidate-window validation temporarily disables sampling on a disposable probe
so it can inspect every available window; the training model retains **A=0.4**.

## Evaluation and outputs

The runner calls Experiment 1's existing `run_outcome` for GI only, using the
saved decision threshold (**0.5**) and explainability bins. This preserves its
metrics, training-patient temporal explainability, runtime and candidate counts,
model serialization, tree export, predictions, and ROC plotting procedures.

Standard Experiment 1 GI artifacts are saved under `gi/`, including `model.dill`,
`tree.json`, `tree.png`, `test_predictions.csv`, `roc_curve.png`,
`metrics_and_temporal_summary.json`, and `temporal_explainability/` outputs.
The run root contains `experiment_config.json`, the copied
`shared_patient_split.csv`, `comparison_with_original_gi.csv`, the existing
comparison figures in `figures/`, and `run_status.json` after all exports succeed.
The earlier flat export helper remains available for existing callers; new runs
use the standard Experiment 1 output layout.

The GI ground truth still depends on the last month before NOW. Alternative
windows do not represent recovery of that temporal mechanism. The supplemental
comparison marks NOW as unavailable; standard GI artifacts retain Experiment 1's
unchanged summary format. No statistical significance is inferred from this run.

Optional manual validation (reads inputs and probes windows, but does not fit or
write experiment outputs):

```bash
python3 experiments/experiment_1_1_gi_without_now.py --validate-only
```

Use an environment compatible with the saved dill model. No validation, tests,
or training were executed while preparing this update.
