import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from experiments import ensemble_robustness_from_saved_trees as analysis


def records():
    return [
        {"node_id": 0, "type": "split", "depth": 0, "num_patients": 2,
         "feature": "Measurement", "anchor": None, "offset": None,
         "is_full_timeline": True, "aggregation": "sum", "threshold": 1.0,
         "missing_direction": "right", "left_child": 1, "right_child": 2},
        {"node_id": 1, "type": "leaf", "depth": 1, "num_patients": 1,
         "prediction": 0.1},
        {"node_id": 2, "type": "leaf", "depth": 1, "num_patients": 1,
         "prediction": 0.9},
    ]


def data():
    rows = []
    for patient_id, value in [(1, 0.0), (2, 2.0)]:
        rows.extend([
            {"patient_id": patient_id, "time": pd.Timestamp("2024-01-01"),
             "feature_name": "Measurement", "value": value},
        ])
    return pd.DataFrame(rows)


class EnsembleAnalysisTests(unittest.TestCase):
    def test_reconstructs_json_tree_and_predicts_without_fit(self):
        model = analysis.reconstruct_tree(records())
        with patch.object(model, "fit", side_effect=AssertionError("fit called")):
            np.testing.assert_allclose(model.predict(data()), [0.1, 0.9])

    def test_invalid_tree_is_rejected(self):
        broken = records()
        broken[0]["right_child"] = 99
        with self.assertRaises(ValueError):
            analysis.reconstruct_tree(broken)

    def test_patient_id_alignment(self):
        table = analysis.prediction_table(
            np.array([2, 1]), pd.Series([1, 0]), {42: np.array([0.9, 0.1])}
        )
        self.assertEqual(table.patient_id.tolist(), [2, 1])
        self.assertEqual(table.seed_42_score.tolist(), [0.9, 0.1])
        with self.assertRaises(ValueError):
            analysis.prediction_table([1, 1], pd.Series([0, 1]), {42: [0.1, 0.9]})

    def test_averaging_scores_and_ensemble_auc(self):
        table = analysis.prediction_table(
            [1, 2, 3, 4], pd.Series([0, 0, 1, 1]),
            {42: np.array([0.1, 0.2, 0.8, 0.9]),
             43: np.array([0.4, 0.3, 0.6, 0.7])},
        )
        scores = table[["seed_42_score", "seed_43_score"]].mean(axis=1)
        self.assertEqual(scores.tolist(), [0.25, 0.25, 0.7, 0.8])
        self.assertEqual(roc_auc_score(table.y_true, scores), 1.0)

    def test_saved_auc_gate_rejects_any_mismatch(self):
        validation = pd.DataFrame({"validation_passed": [True, False]})
        with self.assertRaises(RuntimeError):
            if not validation.validation_passed.all():
                raise RuntimeError("A reconstructed AUC did not match the saved AUC.")

    def test_no_fit_in_analysis_path(self):
        with patch.object(analysis.TemporalTree, "fit", side_effect=AssertionError("fit called")):
            model = analysis.reconstruct_tree(json.loads(json.dumps(records())))
            model.predict(data())


if __name__ == "__main__":
    unittest.main()
