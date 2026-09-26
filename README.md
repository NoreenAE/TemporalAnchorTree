# Set-Tree
### Extending decision trees to process sets
This is the official repository for the paper: "Trees with Attention for Set Prediction Tasks" (ICML21).
[http://proceedings.mlr.press/v139/hirsch21a.html](http://proceedings.mlr.press/v139/hirsch21a.html)

This repository contains a prototypical implementaion of Set-Tree and GBeST (Gradient Boosted Set-Tree) algorithms

## Getting Started
The Set-Tree package can be downloaded from PIP:
`pip install settree`

We also supply the code and datasets for reproducing our experimetns under `exps` folder.

## Background and motivation
In many machine learning applications, each record represents a set of items. A set is an unordered group of items, the number of items may differ between different sets. Problems comprised from sets of items are present in diverse fields, from particle physics and cosmology to statistics and computer graphics. In this work, we present a novel tree-based algorithm for processing sets.

![set_problems](images/set_problems.PNG)

## The model
Set-Tree model comprised from two components:
1) **Set-compatible split criteria**: we specifically support the familly of split criteria defined by the following equation and parametrized by alpha and beta.
2) **Attention-Sets**: a mechanism for allplying the split criteria to subsets of the input. The attention-sets are derived forn previous split-criteria and allows the model to learn more complex set-functions.


<img src="https://github.com/TAU-MLwell/Set-Tree/blob/main/images/model.PNG" width="600" align="center">

## Implementation

The current implementation is based on Sklean's `BaseEstimator` class and is fully compatible with Sklearn.
It contains two main components: `SetDataset` object, that receives `records` as attribute. `records` is a list of numpy arrays, each array with shape `(n_i, d)` represents a single record (set). `d` is the dimention of each item in the record and `n_i` is the number of items in the i's record and may differ between records. The second componnent is `SetTree` model inherited from Sklean's `BaseEstimator` and has simillar attributes.

When configuring Set-Tree one should also configure:
- `operations` : list of the operations to be used
- `use_attention_set` : binary flag for activating the attention-sets mechanism 
- `attention_set_limit` :  the number of ancestors levels to derive attention-sets from
- `use_attention_set_comp` : binary flag for activating the attention-sets compatibility option

A simplified code snippet for training Set-Tree:
```
import settree
import numpy as np

set_data = settree.SetDataset(records=[np.random.randn(2,5) for _ in range(10)])
labels = np.random.randn(10) >= 0.5
set_tree_model = settree.SetTree(classifier=True,
                                 criterion='entropy',
                                 operations=settree.OPERATIONS,
                                 use_attention_set=True,
                                 use_attention_set_comp=True,
                                 attention_set_limit=5,
                                 max_depth=10)
set_tree_model.fit(set_data, labels)
```

A simplified code snippet for training GBeST:
```
import settree
import numpy as np

set_data = settree.SetDataset(records=[np.random.randn(2,5) for _ in range(10)])
labels = np.random.randn(10) >= 0.5
gbest_model = settree.GradientBoostedSetTreeClassifier(learning_rate=0.1, 
                                                       n_estimators=10,
                                                       criterion='mse',
                                                       operations=settree.OPERATIONS,
                                                       use_attention_set=True,
                                                       use_attention_set_comp=True,
                                                       attention_set_limit=5,
                                                       max_depth=10)
gbest_model.fit(set_data, labels)
```

For further details and examples see: `example.ipynb`.

## Temporal experiment ROC figures

Future runs save a 300 dpi ROC figure using the continuous evaluation scores
and the ROC-AUC already calculated by the evaluation pipeline:

- `main.py`: `roc_curve.png` in the supplied run output directory.
- Experiment 1: `fa/roc_curve.png`, `gi/roc_curve.png`, and
  `roc_curves_fa_vs_gi.png` in the parent result directory after both outcomes.
- Ensemble Robustness: `FA_roc_curve.png` and `GI_roc_curve.png` in each seed attempt
  directory, alongside the existing outcome-prefixed artifacts.
- Experiment 1 V2 search: `roc_curve.png` per outcome evaluation, including
  validation candidates; final test candidates also receive the combined
  `roc_curves_fa_vs_gi.png`. Validation plots use validation scores, while final
  plots use test scores. ROC saving is independent of the search's temporal
  usage `save_plots` option.

For a completed Experiment 1, the plotting-only helper
`experiments.experiment_1_fa_vs_gi.create_roc_comparison(Path(result_directory))`
reads each outcome's `test_predictions.csv` (`label` and `score` columns) and
`metrics_and_temporal_summary.json`. It writes only the combined PNG and never
loads a model or calls prediction/training. Missing prediction artifacts cause
an error; aggregate AUC values alone cannot supply a ROC curve.

## Experiment 1 thesis tree figures

After both outcomes have saved their trees, Experiment 1 automatically adds
`fa/fa_tree_thesis.png`, `gi/gi_tree_thesis.png`, and the stacked comparison
`learned_trees_fa_vs_gi_thesis.png` in the parent result directory. The existing
`tree.json` and `tree.png` artifacts are retained. All three new figures use
300 dpi and consistent node styling, with separate `(a)` FA and `(b)` GI panels.

Internal nodes show the feature, temporal window, aggregation, and threshold;
leaves show training patient counts and positive-class probability. Thresholds
are displayed with six significant digits and probabilities with three decimal
places. Integer offsets denote calendar months; fractional offsets are shown
as days using the model's rounded 30-day conversion. Branch labels describe
the comparison for observed aggregates; missing-value routing is not recorded
in the existing tree JSON and is not inferred by this visualization.

To generate these figures from a completed run without loading or executing a
model, call
`experiments.experiment_1_fa_vs_gi.create_thesis_tree_figures(Path(result_directory))`.
This helper reads only the two `tree.json` files and writes only the three PNGs.
It rejects incomplete child links, cycles, and unreachable nodes.

## Citation
If you use Set-Tree in your work, please cite:
```
@InProceedings{pmlr-v139-hirsch21a,
  title = 	 {Trees with Attention for Set Prediction Tasks},
  author =       {Hirsch, Roy and Gilad-Bachrach, Ran},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {4250--4261},
  year = 	 {2021},
  editor = 	 {Meila, Marina and Zhang, Tong},
  volume = 	 {139},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {18--24 Jul},
  publisher =    {PMLR}
}
```

## License
Set-Tree is MIT licensed, as found in the [LICENSE](https://github.com/TAU-MLwell/Set-Tree/blob/main/LICENSE) file.







## Experiment naming

- Experiment 1 — FA vs. GI: existing files and configuration unchanged.
- Experiment 2 — Fish Introduction – Dynamic Anchors:
  `experiments/experiment_2_fish_introduction_dynamic_anchors.py`.
- Ensemble Robustness:
  `experiments/ensemble_robustness.py`.
- Saved-tree ensemble analysis:
  `experiments/ensemble_robustness_from_saved_trees.py`;
  documentation: `experiments/ensemble_robustness_from_saved_trees.md`.

Fish data generation remains in `generate_fish_intro_dataset.py`;
the dataset remains in `data/fish_intro_dataset_distractors/`.

Historical result directories, result-path references, and saved-run identifiers
retain their original names. This naming-only change does not move results or
change existing output defaults.
