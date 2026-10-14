# Experiment 2 dynamic-anchor demonstration

Run manually from the repository root:

```bash
cd /home/noreena/TemporalDecisionTree_clean
python experiments/demo_experiment_2_dynamic_anchors.py
```

This demonstration reuses the fish-introduction Experiment 2 in
`experiment_2_fish_introduction_dynamic_anchors.py`, as identified by the
repository's experiment naming documentation. It imports its model factory,
shared configuration, dataset loader, and evaluation helper. It does not modify
the original experiment, generator, or tree implementation.

By default, the existing `generate_fish_intro_dataset.generate_data()` function
generates and validates its fixed 5,000-patient synthetic cohort in memory.
The generator has a fixed cohort size; retaining it avoids duplicating its
temporal rules or changing its validation. Only 110 patients are retained for
this demonstration. There is one training run and one prediction call.

To reuse an existing dataset produced by that generator instead:

```bash
python experiments/demo_experiment_2_dynamic_anchors.py --data-dir data/fish_intro_dataset_distractors_v2
```

The script selects ten complete blocks from the original training partition
(100 patients) and one complete block from the original test partition
(10 patients), with selection seed 42. Blocks and patient IDs cannot overlap.
Selection does not use outcomes. Keeping all ten matched profiles together
preserves the synthetic design and avoids splitting matched patients between
training and testing. Observations retain their original order, which matters
for candidate sampling and tied split scores.

Only Dynamic ON runs. Maximum depth remains 3. The only model changes are
`min_samples_split=10` and `min_samples_leaf=5`: nodes with ten patients can
split while each child must contain at least five patients. These represent
10% and 5% of the training cohort and permit meaningful deeper construction.
Small leaves will have less stable probability estimates, so the output is
an illustration rather than a performance benchmark.

All other settings are copied from Experiment 2: sum aggregation, binary-event
feature semantics, Birth Date windows of 1–22 calendar months and full follow-up,
dynamic windows of 1–4 calendar months, candidate sampling rate 0.4, fixed model
and sampling seeds 42, MSE criterion, and missing values routed right. Regression
on binary FA labels yields leaf means interpreted as positive-class probabilities.
The demonstration applies a probability threshold of 0.5 to report class labels.

## What the records mean

Temporary observers on this model instance call the original `_get_node` and
`_prepare_branch_anchor` methods and return their results unchanged. They record
the actual incoming anchor references and patient timestamps for every node,
including leaves, and the actual anchor lists passed to both children. Leaves
have no selected split; some stop before candidate search, while others may
result from finding no valid split. The observers are removed after fitting.

`available_as_reference` means that an anchor name belongs to the node's
candidate reference list. It does not mean every patient has a timestamp.
`patient_timestamps` records either a singleton timestamp list or `[]`, with
defined/undefined counts. Predefined and dynamic references are labelled
separately. A newly selected split cannot use its own generated anchor; that
reference is passed to descendants. Both branches receive the same logical
anchor name but separate patient-specific rows.

Display node IDs use preorder traversal. `source_node_id` uses the algorithm's
persisted breadth-first split IDs and determines the dynamic anchor name.
These two ID conventions are deliberately recorded separately.

During the actual held-out `model.predict` call, an observer delegates to the
original `_append_prediction_branch_anchor`. The same production aggregation
method is called again solely to annotate the value and check the observed
direction; split selection and routing remain in the original tree. Anchor
dates are read from the rows returned by the production helper. No training
timestamps are copied into test observations.

The annotation normalizes both scalar and singleton-array aggregate values.
This accommodates the production helper's direct-array return when a feature
has no observations, while preserving its missing value and routing behavior.

For these binary events with sum aggregation, the production helper chooses
the earliest eligible positive event in the selected temporal window, or `[]`
when no such event exists. The split threshold routes the patient; it is not
reapplied to each raw event when selecting an anchor timestamp.

The prediction PNG pairs the complete trained tree with a prediction-step table.
The left panel reuses Experiment 2's existing layout and the same clean node
labels as the training view. It highlights visited nodes and edges in blue and
the final leaf in green; unvisited branches remain visible in muted colors.
No patient-specific information is added inside the tree nodes.

The right panel has one row per actual visited split in traversal order:
node ID, learned condition, patient aggregate, YES/NO/Missing decision and
destination, reference anchor with timestamp, and newly generated dynamic anchor
with timestamp. Its final row shows the leaf, predicted class, probability,
and true class. Missing aggregates follow the recorded missing direction;
they are not interpreted as a failed comparison. Full-timeline windows ignore
the stored anchor. Undefined timestamps are shown as `[]`, distinct from an
absent reference.

Dynamic anchors use matching pastel cell colors and `[Dk]` identifiers in the
generation and reference columns. `k` is the persisted source split ID already
embedded in the anchor name, while the Node ID column uses preorder display
IDs. `[P]` identifies a predefined reference. The caption explains that the
learned structure and conditions stay fixed, timestamps come from the new
patient's observations using the training rules, and a generated reference
becomes available only to subsequent nodes.

Patient selection requires an actual nonempty timestamp generated earlier in
the same trace and reused later as the same dynamic reference with the same
timestamp. If none of the ten test patients qualifies, the script reports an
error rather than plotting a fallback. There is no seed search, refitting, or
forced tree construction. Before saving, the table checks its path and routing
against the tree, the order and consistency of dynamic reference use, and the
final probability and class against the recorded prediction.

## Outputs

All generated outputs go into
`experiments/results/demo_experiment_2_dynamic_anchors/`:

- `configuration.json`: effective parameters, source validation, and selection.
- `patients_and_split.csv`, `observations.csv`: the selected 110-patient cohort.
- `training_anchor_summary.json`: actual tree, selected splits, incoming anchors,
  training timestamps, and outgoing branch state, also summarized on stdout.
- `test_predictions.csv`: probabilities, labels, and true labels for ten patients.
- `test_metrics.json`: accuracy, precision, recall, F1, confusion matrix,
  ROC-AUC when both labels occur, Brier score, and Experiment 2 calibration bins.
- `test_prediction_paths.json`: captured split visits for all ten test patients.
- `selected_patient.json`: highlighted patient, selection reason, and leaf ID.
- `prediction_path.png`: the combined highlighted-tree and prediction-step
  table figure, saved at 300 DPI.
- `learned_tree_anchors.png`: the complete learned tree, with final selected
  splits and captured available anchors **before** each split inside one box
  per node. Each internal node shows the selected feature, reference, temporal
  window, aggregation, threshold, and incoming anchor list. `[P]` identifies
  predefined references and `[D]` dynamic references; the ` Anchor` suffix is
  omitted for compact display while source-node IDs remain intact. The current
  node's generated anchor is excluded from its list; only ancestor-generated
  references are available. Leaves show only patient count and positive-class
  probability. Missing routing is indicated on the corresponding edge. The PNG
  is saved at 300 DPI using the existing tree layout helper.

To draw this training view from the already saved tree, without generating
data, training, or predicting again, run:

```bash
python experiments/demo_experiment_2_dynamic_anchors.py --plot-training-summary experiments/results/demo_experiment_2_dynamic_anchors_fixed/training_anchor_summary.json --overwrite-plot
```

This mode writes only `learned_tree_anchors.png` beside the supplied JSON.
`--overwrite-plot` explicitly replaces the previous visualization and is valid
only in a plot-only mode. Without it, an existing PNG is preserved. The regular
demonstration also creates this training view immediately after saving its
training summary. Incoming references are taken directly from each node's
`available_anchors` record, not from `children_anchor_state`. Plotting validates
that a node's own generated reference is absent from its incoming list and
that a selected finite-window reference belongs to that list. Being listed
does not guarantee that an individual patient has a defined timestamp.

To regenerate only the combined prediction figure from existing results:

```bash
python experiments/demo_experiment_2_dynamic_anchors.py --plot-prediction-results experiments/results/demo_experiment_2_dynamic_anchors_fixed --overwrite-plot
```

This mode reads `training_anchor_summary.json`, `test_prediction_paths.json`,
and `test_predictions.csv`, selects a qualifying patient from those ten saved
test traces, and replaces only `prediction_path.png`. It does not generate data,
load a serialized model, train, predict, or modify saved numerical results.
The regular demonstration creates the same combined figure after its single
prediction call.

The script refuses to overwrite an existing output directory. For a subsequent
manual run, choose a new location with `--output PATH`.

No scripts, tests, training, prediction, or plot generation were executed while
implementing this demonstration. Runtime results remain to be verified by the
manual run.
