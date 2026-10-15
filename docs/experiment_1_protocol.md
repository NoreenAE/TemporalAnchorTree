# Experiment 1 Protocol

Run these commands from the repository root after [installation](installation.md).
Use the included synthetic data and new output directories. For the central
training/comparison entry point, see [reproducibility.md](reproducibility.md).

## Experiment 1: FA vs. GI temporal ground-truth recovery

Compare FA and GI using the same observations, patient split, and model settings.
The stored protocol uses A=0.4, depth 3, split/leaf minima 20/10, minimum gain
0.0005, mean/sum/max/count aggregations, and a 70/30 joint-stratified split.
Dynamic anchors are disabled.

**Reproduce the recorded seed-42 settings:** the existing A wrapper restores the
saved A=1 reference configuration and split, changing A to 0.4:

```bash
python -B experiments/finding_best_A_for_exp_1.py --A 0.4
```

This trains both outcomes and writes
`supporting_analyses/candidate_sampling/results/A_0.4/`. The wrapper also accepts
A=0.2 or 0.8 for separate experiments; these are not the final A=0.4 result.
Its required reference is
[supporting_analyses/candidate_sampling/baseline_A1/](../supporting_analyses/candidate_sampling/baseline_A1/).

**Run the main runner with the same seed-42 model settings:**

```bash
python -B experiments/experiment_1_fa_vs_gi.py --validate-only
python -B experiments/experiment_1_fa_vs_gi.py --output experiments/results/experiment_1_new_attempt
```

The first command validates inputs and a shared split without training or writing.
The second trains both outcomes in a new directory. Both `RANDOM_STATE` and
`SPLIT_SAMPLING_SEED` now use 42, matching the recorded thesis settings. The central
[reproduction entry point](reproducibility.md) also verifies that the generated split is identical
to the stored split.

[Saved Experiment 1 results](../experiments/results/experiment_1_5000V3clean__moreOFFSETS2_patients_A=0.4_minS20_minL10_minGAIN0.0005/)
include `config.json`, `shared_patient_split.csv`, outcome predictions/metrics/JSON
trees, combined ROC/tree figures, and temporal-usage figures and tables.

## Experiment 1.1: GI without NOW

Remove NOW from the candidate references to assess its contribution to GI
prediction. Both variants use the final Experiment 1 cohort and recorded seed-42
split; the retained minimum gains are 0 and 0.0005.

Validate both without training:

```bash
python -B experiments/experiment_1_1_gi_without_now.py --validate-only --min-gain 0
python -B experiments/experiment_1_1_gi_without_now.py --validate-only --min-gain 0.0005
```

Train each variant into a separate new directory:

```bash
python -B experiments/experiment_1_1_gi_without_now.py --run --min-gain 0 --output experiments/results/gi_without_now_gain0_new
python -B experiments/experiment_1_1_gi_without_now.py --run --min-gain 0.0005 --output experiments/results/gi_without_now_gain0005_new
```

New runs export their configuration, split, GI predictions/metrics/tree,
temporal analysis, and comparison figures. Existing archives are
[gain 0](../experiments/results/experiment_1_1_gi_without_now_A0.4_minGAIN0/) and
[gain 0.0005](../experiments/results/experiment_1_1_gi_without_now_A0.4_minGAIN0.0005/).
The gain-0.0005 archive currently contains only its configuration, split,
predictions, and temporal path-usage table; its metrics/tree/figure outputs are
not present. The configuration can still be selected for a new full run.

## Supporting analyses

The [V2 hyperparameter search](../supporting_analyses/v2_hyperparameter_search/README.md)
is an earlier Experiment 1 dataset/protocol. Its summaries and top-three results
are retained **as results only**. The old dataset and search runner are not included,
so this search cannot be rerun from the public snapshot.
