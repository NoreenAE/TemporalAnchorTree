# Method and API Guide

## Project overview

A conventional tree usually receives one fixed feature vector per patient. TAT
instead searches feature–window–aggregation combinations directly over repeated,
timestamped observations. Each split remains a readable threshold rule, such as
“the number of GI events within one month of fish introduction is at most 3.”

The experiments ask whether these rules recover known temporal relationships,
how predictions change when a relevant reference is removed, and whether dynamic
anchors improve prediction when ordinary summaries cannot capture the relationship.
FA denotes the synthetic food-allergy outcome; GI denotes the synthetic
gastrointestinal outcome or symptom feature, depending on the experiment.

## Key features and how TAT works

1. **Predefined patient-specific anchors:** dates such as `Birth Date Anchor` and
   `NOW Anchor` are supplied in the observations. NOW is the patient's recorded
   reference date, not the computer's current date.
2. **Temporal windows:** offsets define intervals before or after an anchor.
   Integer offsets use calendar months; fractional offsets use rounded 30-day
   fractions. Window boundaries are inclusive. `None` selects the full timeline.
   `anchor_offset_map` overrides offsets for particular anchors.
3. **Aggregation and split search:** candidate rules aggregate a feature with
   functions such as mean, sum, max, or count, then compare the aggregate with a
   learned threshold. `candidate_sampling_rate` is the independent probability
   that a feature–window–aggregation group is explored; `1.0` searches all groups.
4. **Dynamic anchors:** eligible splits create patient-specific references for
   descendant nodes. Eligibility is configured with `feature_semantics`; enabling
   `use_dynamic_anchors` alone does not define it. Timestamp selection depends on
   the aggregation and semantics: for example, binary-event SUM selects the
   earliest positive event in the evaluated window. A patient without an eligible
   timestamp has an unavailable anchor.
5. **Predictions:** classification trees provide labels with `predict` and class
   proportions with `predict_proba`. Regression trees return leaf means with
   `predict`. The thesis runners fit regression trees to binary labels and evaluate
   these continuous scores; their scores are not calibrated clinical probabilities.

For example, a split on `FishIntroduction` can establish a child's introduction
date. A descendant can then count GI or vomiting events relative to that date.
This permits different children to use different temporal windows under the same
readable rule.

## Quick start — running TAT

After installation, execute the following Python code from the repository root,
for example in a Python session or a notebook cell. It loads the included
Experiment 2 data, preserves the saved split for 20 matched blocks, fits a small
classification tree, and evaluates 40 held-out patients. It writes no files.

This is an API demonstration with **160 training patients and 40 test patients**,
not a reproduction of the thesis regression-tree configuration.

```python
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from TAT import TemporalTree
from experiments.experiment_2_noisy_v2 import load_dataset
from generate_fish_intro_noisy_v2 import EVENT_FEATURES

# Load the included cohort and verify its version, hashes and saved split.
X, patients, _ = load_dataset(Path("data/fish_intro_noisy_matched_v2"))

# Use 20 whole matched blocks for a small demonstration, retaining their split.
blocks = patients["block"].drop_duplicates().iloc[:20]
patients = patients.loc[patients["block"].isin(blocks)]
labels = patients.set_index("patient_id")["FA"]
train_ids = patients.loc[patients["partition"].eq("train"), "patient_id"]
test_ids = patients.loc[patients["partition"].eq("test"), "patient_id"]
X_train = X.loc[X["patient_id"].isin(train_ids)]
X_test = X.loc[X["patient_id"].isin(test_ids)]

# TAT aligns labels with patient appearance order in the longitudinal table.
y_train = labels.loc[X_train["patient_id"].unique()]
y_test = labels.loc[X_test["patient_id"].unique()]

model = TemporalTree(
    max_depth=2,
    min_samples_split=20,
    min_samples_leaf=10,
    anchors_list=["Birth Date Anchor"],
    anchor_offset_list=[1],
    anchor_offset_map={"Birth Date Anchor": [1, 6, 12, None]},
    aggregation_funcs={"sum": np.sum},
    is_classification=True,
    criterion="gini",
    feature_semantics={name: "binary_event_indicator" for name in EVENT_FEATURES},
    use_dynamic_anchors=True,
    candidate_sampling_rate=1.0,
    random_state=42,
    split_sampling_seed=42,
    verbose=0,
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
positive_column = np.flatnonzero(model.classes_ == 1).item()
positive_scores = probabilities[:, positive_column]

print(f"Patients: {len(y_train)} train / {len(y_test)} test")
print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_test, positive_scores):.3f}")
```

### Input format and label alignment

`X` is a pandas DataFrame with one row per observation:

| Column | Meaning |
| --- | --- |
| `patient_id` | Patient identifier; repeated across observations |
| `time` | Observation timestamp |
| `feature_name` | Feature name or anchor name |
| `value` | Numeric observation, or a timestamp for a predefined anchor |

Each configured predefined anchor has one row per patient, with its date in
`value`. `load_dataset` performs the required date conversion for Experiment 2.
Keep outcomes, latent risk values, and cohort metadata outside the predictor table.

Pass one label per patient, **in `X.patient_id.unique()` order**. Predictions use
that same patient appearance order. Probability columns follow `model.classes_`,
so the example locates class `1` explicitly. `predict_proba` requires
`is_classification=True`; for the thesis regression trees, use `predict` to obtain
continuous scores.

## Repository structure

The following shows the main files in the current checkout; result directories
contain additional artifacts listed below.

```text
.
├── TAT/
│   ├── __init__.py
│   ├── temporal_tree.py
│   ├── anchor_utils.py
│   ├── temporal_tree_ensemble.py
│   ├── temporal_explainability.py
│   └── eval_utils.py
├── settree.py
├── project_paths.py
├── requirements.txt
├── README.md
├── LICENSE
├── create_temporal_benchmark_v3_clean_fa.py
├── generate_fish_intro_noisy_v2.py
├── plot_true_probability_histograms.py
├── generators/
│   ├── experiment1_longitudinal.py
│   └── experiment1_gi.py
├── data/
│   ├── temporal_benchmark_v3_clean_fa/
│   └── fish_intro_noisy_matched_v2/
├── experiments/
│   ├── reproduce_thesis_results.py
│   ├── experiment_1_fa_vs_gi.py
│   ├── finding_best_A_for_exp_1.py
│   ├── experiment_1_1_gi_without_now.py
│   ├── experiment_2_noisy_v2.py
│   ├── experiment_2_1_candidate_sampling_robustness.py
│   ├── experiment_2_cart_baseline.py
│   ├── experiment_2_fish_introduction_dynamic_anchors.py
│   ├── experiment_utils.py
│   ├── experiment_2_outputs.py
│   ├── experiment_2_temporal_explainability.py
│   ├── experiment_2_anchor_transition.py
│   ├── visualize_experiment_1_timelines.py
│   ├── visualize_experiment_2_noisy_matched_v2_patients.py
│   ├── plot_experiment_2_combined_roc.py
│   ├── EDA_experiment_2_noisy_v2.ipynb
│   ├── EDA_experiment_2_cart_inputs.ipynb
│   └── results/
├── notebooks/
│   └── EDA_experiment_1_v3_clean_fa.ipynb
├── figures/
├── supporting_analyses/
│   ├── candidate_sampling/baseline_A1/
│   └── v2_hyperparameter_search/



```

- [TAT/](../TAT/) contains the algorithm, anchor handling, optional regression
  ensembles, evaluation utilities, and temporal explainability.
- [generators/](../generators/) supplies the shared Experiment 1 simulation components;
  the two root-level generator scripts build the final cohorts.
- [experiments/](../experiments/) contains runnable studies, shared scientific
  helpers, post-processing, and the Experiment 2 notebooks.
- [data/](../data/), [experiments/results/](../experiments/results/), and [figures/](../figures/)
  hold the included cohorts, saved runs, and supplementary illustrations.
- [supporting_analyses/](../supporting_analyses/) holds the A=1 reference used by the
  Experiment 1 reproduction wrapper and results from the earlier V2 search.
- [docs/](../docs/) explains the method, datasets, protocols and reproduction
  workflow. [tests/](../tests/) contains lightweight algorithm and dependency checks.
- [project_paths.py](../project_paths.py) resolves project-relative paths;
  [settree.py](../settree.py) keeps historical model imports compatible with `TAT`.
