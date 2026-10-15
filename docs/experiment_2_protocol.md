# Experiment 2 Protocol

Run these commands from the repository root after [installation](installation.md).
Use the included synthetic data and new output directories. For the central
training/comparison entry point, see [reproducibility.md](reproducibility.md).

## Experiment 2: Dynamic Anchors ON vs. OFF

Test whether dynamically created event references improve FA prediction on Noisy
Matched V2. Both trees use the same 4,000/1,000 patient block split, depth 3,
split/leaf minima 100/50, A=0.4, seed 42, minimum gain 0, and SUM aggregation.
Only the dynamic-anchor switch differs.

```bash
python -B experiments/experiment_2_noisy_v2.py --data-dir data/fish_intro_noisy_matched_v2 --output experiments/results/experiment_2_new_attempt
```

This trains two models and exports `dynamic_on/`, `dynamic_off/`, a configuration,
comparison metrics/figures, and anchor-usage analyses. The raw cohort is shared
through `configuration.json`; it is not duplicated in each result directory.
[Saved Experiment 2 results](../experiments/results/experiment_2_noisy_matched_v2/)
include model pickles, JSON/text trees, patient scores, diagnostics, anchor traces,
ROC/tree comparisons, and temporal-reference figures.

## Experiment 2.1: Candidate Sampling Robustness

Repeat paired Dynamic ON/OFF fits for sampling seeds 43–46 on the same fixed
cohort, holding every other setting fixed. Seed 42 is a separate checkpoint;
seed variability is not patient uncertainty or cross-validation. Seed 47 is
excluded from future fits by author request; its saved results are preserved.

This runner requires POSIX locking and the checkpoint's Python/package versions.
It verifies dataset/source hashes against the saved baseline and
[seed-42 checkpoint](../experiments/results/experiment_2_1_seed42_checkpoint/).

```bash
python -B -m experiments.experiment_2_1_candidate_sampling_robustness run --output experiments/results/experiment_2_1_new_attempt
```

A new output trains eight models, with individual files under
`seeds/seed_43/` through `seeds/seed_46/` and aggregates under `summary/`.
Repeating the command verifies and skips completed fits. An incomplete attempt
stops the run unless `--retry-incomplete` is explicitly supplied; corrupted
completion hashes stop execution.

[Saved robustness results](../experiments/results/experiment_2_1_candidate_sampling_robustness/)
retain the original frozen five-seed `protocol.json`, ten completed runs
(43–47), per-seed scores/trees, paired AUC tables, temporal stability summaries,
and figures. The four-seed runner requires a new output directory and does not
resume or modify this archive. To rebuild only the historical five-seed
summaries from those saved runs:

```bash
python -B -m experiments.experiment_2_1_candidate_sampling_robustness summarize
```

This performs no fitting, prediction, or model loading. It writes a new `summary/`
and archives the previous one under `history/`; to inspect existing results
without writes, open the [saved AUC summary](../experiments/results/experiment_2_1_candidate_sampling_robustness/summary/auc_summary.csv).

## CART baseline comparison

Compare a conventional `DecisionTreeClassifier` with the saved Dynamic ON TAT
scores. CART uses full-history sums of eight event features, gini, depth 3,
split/leaf minima 100/50, and seed 42. It uses the identical patient split and
labels; dates, anchors, IDs, and latent risk values are excluded from its matrix.

```bash
python -B experiments/experiment_2_cart_baseline.py --recorded-run experiments/results/experiment_2_noisy_matched_v2
```

This trains CART only, then creates a uniquely named
`experiments/results/experiment_2_cart_baseline/run_<timestamp>_<id>/` with
`results.csv`, `test_predictions.csv`, `run_metadata.json`, `cart_tree.png`, and
`roc_cart_vs_dynamic_on.png`. It does not retrain TAT or regenerate the cohort.
The [selected saved CART run](../experiments/results/experiment_2_cart_baseline/run_20261014T110752808924Z_5f1eab64/)
uses the final 100/50 configuration.
