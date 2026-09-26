"""No-fit checks for the A=0.4, minimum-gain-zero NOW-anchor ablation."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments import experiment_1_1_gi_without_now as experiment
from settree.temporal_tree import TemporalLeaf, TemporalSplitNode, TemporalTree


class TestGIWithoutNow(unittest.TestCase):
    def setUp(self):
        no_fit = patch.object(
            TemporalTree, "fit", side_effect=AssertionError("No fitting")
        )
        no_fit.start()
        self.addCleanup(no_fit.stop)
        self.config = json.loads(
            (experiment.BASELINE / "config.json").read_text()
        )
        self.params = experiment.ablated_parameters(self.config)
        self.aggregations = experiment.original.aggregation_functions()

    def test_only_anchor_availability_and_gain_change_from_experiment_1(self):
        before = deepcopy(self.config)
        windows = experiment.validate_parameters(
            self.config, self.params, self.aggregations
        )
        self.assertEqual(before, self.config)
        self.assertEqual(self.params["anchors_list"], [experiment.BIRTH])
        self.assertNotIn(experiment.NOW, self.params["anchor_offset_map"])
        self.assertFalse(
            any(anchor == experiment.NOW for anchor, _ in windows)
        )
        self.assertEqual(windows.count((None, None)), 1)
        self.assertEqual(len(windows), 10)
        self.assertEqual(self.params["min_impurity_decrease"], 0.0)
        self.assertEqual(self.params["candidate_sampling_rate"], 0.4)
        for key, value in self.config["model_hyperparameters"].items():
            if key not in (
                "anchors_list", "anchor_offset_map", "min_impurity_decrease"
            ):
                self.assertEqual(self.params[key], value)

    def test_primary_output_is_rejected_without_modification(self):
        def snapshot():
            return {
                str(path.relative_to(experiment.PRIMARY)): (
                    experiment.original.sha256_file(path), path.stat().st_mtime_ns
                )
                for path in experiment.PRIMARY.rglob("*") if path.is_file()
            }

        before = snapshot()
        with patch.object(experiment, "OUTPUT", experiment.PRIMARY):
            with self.assertRaises(ValueError):
                experiment.run({})
        self.assertEqual(snapshot(), before)

    def test_parameter_drift_is_rejected(self):
        for key, value in [
            ("max_depth", 4),
            ("use_dynamic_anchors", True),
            ("candidate_sampling_rate", 0.5),
            ("min_impurity_decrease", 0.0005),
            ("anchors_list", [experiment.BIRTH, experiment.NOW]),
        ]:
            with self.subTest(key=key):
                params = deepcopy(self.params)
                params[key] = value
                with self.assertRaises(ValueError):
                    experiment.validate_parameters(
                        self.config, params, self.aggregations
                    )
        params = deepcopy(self.params)
        params["anchor_offset_map"][experiment.BIRTH].append(24)
        with self.assertRaises(ValueError):
            experiment.validate_parameters(
                self.config, params, self.aggregations
            )
        with self.assertRaises(ValueError):
            experiment.validate_parameters(
                self.config, self.params, {"mean": np.mean}
            )

    def test_probe_observes_actual_now_candidates_if_reintroduced(self):
        model = TemporalTree(
            **self.config["model_hyperparameters"],
            aggregation_funcs=self.aggregations
        )
        with patch.object(
            TemporalTree, "fit", side_effect=AssertionError("No fitting")
        ):
            windows = experiment.probe_candidate_windows(model)
        self.assertIn((experiment.NOW, -1), windows)
        self.assertEqual(windows.count((None, None)), 1)

    def test_exact_saved_membership_and_labels(self):
        reference = pd.read_csv(
            experiment.BASELINE / "shared_patient_split.csv"
        )
        data = reference[["patient_id"]].copy()
        labels = reference[["patient_id", "gi_label"]].rename(
            columns={"gi_label": "label"}
        )
        experiment.validate_split(reference.copy(), data, labels, reference)
        changed = reference.copy()
        train_index = changed.index[changed.split.eq("train")][0]
        test_index = changed.index[changed.split.eq("test")][0]
        changed.loc[[train_index, test_index], "split"] = ["test", "train"]
        with self.assertRaises(AssertionError):
            experiment.validate_split(changed, data, labels, reference)
        labels.loc[0, "label"] = 1 - labels.loc[0, "label"]
        with self.assertRaises(ValueError):
            experiment.validate_split(reference, data, labels, reference)

    def test_output_isolation_and_existing_directory(self):
        for path in (
            experiment.BASELINE,
            experiment.PRIMARY,
            experiment.ROOT / "data",
            experiment.ROOT / "experiments/results/experiment_2_robustness_v3",
        ):
            with self.assertRaises(ValueError):
                experiment.validate_output(path)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = (
                root / "experiments/results/experiment_1_1_gi_without_now_A0.4_minGAIN0"
            )
            expected.parent.mkdir(parents=True)
            with patch.object(experiment, "ROOT", root):
                experiment.validate_output(expected)
                expected.symlink_to(
                    experiment.BASELINE, target_is_directory=True
                )
                with self.assertRaises(ValueError):
                    experiment.validate_output(expected)
                expected.unlink()
                expected.mkdir()
                with self.assertRaises(ValueError):
                    experiment.validate_output(expected)
                (expected / "run_status.json").write_text(
                    '{"status": "completed"}'
                )
                with self.assertRaises(ValueError):
                    experiment.validate_output(expected)

    def test_validation_mode_never_runs_training(self):
        with patch.object(
            experiment.sys, "argv", ["experiment", "--validate-only"]
        ), patch.object(
            experiment, "prepare", return_value={"provenance": {}}
        ), patch.object(
            experiment, "run"
        ) as run, patch(
            "builtins.print"
        ):
            experiment.main()
            run.assert_not_called()

    def test_exports_and_figures_with_hand_built_tree(self):
        model = TemporalTree(
            **self.params, aggregation_funcs=self.aggregations
        )
        model.tree = TemporalSplitNode(
            feature="VomitingYN",
            threshold=0.2,
            aggregation="mean",
            time_window=(experiment.BIRTH, 12),
            num_patients=2,
            left=TemporalLeaf(0.1, 1, 1),
            right=TemporalLeaf(0.8, 1, 1),
        )
        for name in (
            "generated_groups",
            "sampled_groups",
            "sampling_skipped_groups",
            "generated_threshold_candidates",
            "valid_threshold_candidates",
            "invalid_threshold_candidates",
            "evaluated_threshold_candidates",
            "fallback_evaluated_threshold_candidates",
        ):
            setattr(model, name, 0)
        paths = pd.DataFrame(
            {
                "anchor": [experiment.BIRTH, None],
                "stored_offset": [12, np.nan],
                "is_full_timeline": [False, True],
            }
        )
        baseline = json.loads(
            (
                experiment.BASELINE / "gi/metrics_and_temporal_summary.json"
            ).read_text()
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(
                TemporalTree, "fit", side_effect=AssertionError("No fitting")
            ):
                result = experiment.export_results(
                    model,
                    np.array(["patient_b", "patient_a"]),
                    pd.Series([0, 1]),
                    np.array([0.1, 0.8]),
                    paths,
                    {"training": 0},
                    baseline,
                    output,
                )
            self.assertEqual(result["performance"]["roc_auc"], 1)
            predictions = pd.read_csv(output / "gi_test_predictions.csv")
            self.assertEqual(
                predictions.patient_id.tolist(), ["patient_b", "patient_a"]
            )
            self.assertEqual(predictions.y_true.tolist(), [0, 1])
            self.assertEqual(predictions.y_score.tolist(), [0.1, 0.8])
            tree = json.loads((output / "gi_tree.json").read_text())
            self.assertEqual(len(tree), 3)
            self.assertEqual(tree[0]["left_child"], 1)
            self.assertEqual(tree[2]["prediction"], 0.8)
            usage = pd.read_csv(output / "gi_temporal_window_usage.csv")
            self.assertEqual(usage.raw_usage_count.sum(), 2)
            self.assertEqual(usage.usage_proportion.sum(), 1)
            comparison = pd.read_csv(
                output / "comparison_with_original_gi.csv"
            )
            self.assertEqual(
                comparison.iloc[1].now_usage_percent, experiment.REMOVED
            )
            self.assertIn("min_gain=0 (", comparison.iloc[1].experiment)
            self.assertIn("A=0.4", comparison.iloc[1].experiment)
            self.assertEqual(result["experiment_name"], experiment.NAME)
            self.assertEqual(result["min_impurity_decrease"], 0.0)
            self.assertGreater(float(comparison.iloc[0].now_usage_percent), 0)
            self.assertNotIn("now_usage_proportion", result["temporal_usage"])
            flat = pd.read_csv(output / "per_run_results.csv")
            self.assertEqual(flat.iloc[0].birth_usage_percent, 50)
            for filename in (
                "roc_auc_original_vs_without_now.png",
                "temporal_usage_original_vs_without_now.png",
                "gi_tree_thesis.png",
            ):
                self.assertGreater(
                    (output / "figures" / filename).stat().st_size, 1000
                )

    def test_leaf_only_usage_and_unexpected_anchor(self):
        empty = pd.DataFrame(
            columns=["anchor", "stored_offset", "is_full_timeline"]
        )
        summary, distribution = experiment.usage_summary(empty)
        self.assertEqual(summary["total_patient_path_split_usages"], 0)
        self.assertTrue(distribution.empty)
        invalid = pd.DataFrame(
            {
                "anchor": [experiment.NOW],
                "stored_offset": [-1],
                "is_full_timeline": [False],
            }
        )
        with self.assertRaises(ValueError):
            experiment.usage_summary(invalid)


if __name__ == "__main__":
    unittest.main()
