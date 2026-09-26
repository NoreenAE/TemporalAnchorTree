# Ensemble Robustness ensemble from saved trees

This isolated analysis reconstructs the completed Ensemble Robustness FA and GI trees
for seeds 42 through 51. It never calls `TemporalTree.fit()` or
`TemporalTreeEnsemble.fit()`. It restores each JSON tree, uses the existing
`TemporalTree.predict()`, validates all twenty saved ROC-AUC values, then
averages the ten aligned patient scores separately for FA and GI.

Run from the repository root after approval:

```bash
python experiments/ensemble_robustness_from_saved_trees.py
```

The source run is read from `experiments/results/experiment_2_robustness_v3/`.
The only outputs are written below
`experiments/results/experiment_2_ensemble_from_saved_trees/`:

- `reconstruction_validation.csv`
- `FA_ensemble_predictions.csv`
- `GI_ensemble_predictions.csv`
- `ensemble_summary.csv`

The validation CSV is written before the strict gate. If any reconstructed AUC
differs from the saved AUC by more than `1e-12`, the command raises an error and
writes no ensemble predictions or summary. Ensemble AUC is calculated from the
mean patient score, not from the mean of individual AUCs.
