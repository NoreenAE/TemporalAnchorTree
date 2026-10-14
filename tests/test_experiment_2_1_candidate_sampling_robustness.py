"""Experiment 2.1 safeguards and analysis checks; no fitting or data generation.

Run with the research interpreter, without requiring pytest::

    python -m unittest discover -s tests \
        -p 'test_experiment_2_1_candidate_sampling_robustness.py' -v
"""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from settree.temporal_tree import TemporalTree

from experiments import experiment_2_1_candidate_sampling_robustness as exp


def predictions():
    return pd.DataFrame({
        "patient_id": ["a", "b", "c", "d", "train"],
        "partition": ["test"] * 4 + ["train"],
        "documented": [True, True, True, False, True],
        "FA": [0, 1, 0, 1, 1], "block": [1, 1, 2, 2, 3],
        "score": [.1, .9, .2, .4, .8],
        "fish_anchor_created": [True, True, False, False, True],
        "symptom_split_uses_fish_anchor": [True, True, True, False, True],
        "available_fish_anchor_used": [True, True, False, False, True],
    }).set_index("patient_id")


def artifacts(folder, enabled, seed=42):
    folder.mkdir(parents=True, exist_ok=True)
    frame = predictions()
    if not enabled:
        frame["score"] = [.8, .7, .6, .3, .1]
        frame[list(exp.PREDICTION_FLAGS)] = False
    frame.to_csv(folder / "patient_predictions.csv")
    anchor = "Node 0 | SUM(FishIntroduction) Anchor"
    tree = [dict(node_id=0, type="split", feature="FishIntroduction",
                 aggregation="sum", window=["Birth Date Anchor", 22]),
            dict(node_id=1, type="split", feature="GI", aggregation="sum",
                 window=[anchor if enabled else "Birth Date Anchor", 1]),
            dict(node_id=2, type="split", feature="Vomiting", aggregation="sum",
                 window=[anchor if enabled else "Birth Date Anchor", 1])]
    exp.write_json(folder / "tree.json", tree)
    (folder / "tree.txt").write_text("fixture tree")
    (folder / "model.pkl").write_bytes(b"fixture only; never unpickle")
    traces = []
    if enabled:
        for pid in ("a", "b", "train"):
            traces.append(dict(patient_id=pid, node_id=0,
                               anchor_name=anchor, feature="FishIntroduction",
                               fish_anchor_created=True, evaluated_windows=[],
                               source_anchor="Birth Date Anchor", checks={"ok": True}))
        # Both symptoms and duplicate visits for a; c has no available window.
        for pid, node in (("a", 1), ("a", 1), ("a", 2), ("b", 2),
                          ("c", 1), ("train", 1)):
            traces.append(dict(patient_id=pid, node_id=node, anchor_name=None,
                               feature="GI" if node == 1 else "Vomiting",
                               fish_anchor_created=False, source_anchor=anchor,
                               checks={"ok": True},
                               evaluated_windows=[] if pid == "c" else [[0, 1]]))
    exp.write_json(folder / "anchor_trace.json", traces)
    metrics = {row["population"]: dict(roc_auc=row["auc_on"], n=row["n"],
                                       FA_cases=row["cases"])
               for row in exp.paired_auc(seed, dict(dynamic_on=frame, dynamic_off=frame))}
    exp.write_json(folder / "metrics.json", metrics)
    diag = dict(training_runtime_seconds=1.0, mechanism_checks_passed=True,
                audit_failures=[], coverage={"test": {"documented_patients": 3}})
    exp.write_json(folder / "diagnostics.json", diag)
    saved = exp.read_json(exp.BASELINE / "configuration.json")
    config = exp.model_config(saved, seed)
    exp.write_json(folder / "effective_configuration.json",
                   exp.effective_config(config, enabled))
    return diag


class Experiment21Tests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.saved = exp.read_json(exp.BASELINE / "configuration.json")
        for method in ("fit", "predict"):
            guard = patch.object(TemporalTree, method, side_effect=AssertionError(
                "Lightweight tests must never execute a model"))
            guard.start()
            self.addCleanup(guard.stop)

    def tearDown(self):
        self.temp.cleanup()

    def test_config_only_varies_sampling_seed_and_switch(self):
        original = deepcopy(self.saved)
        for seed in (42, *exp.PLANNED_SEEDS):
            config = exp.model_config(self.saved, seed)
            params = deepcopy(config)
            del params["aggregation_funcs"]
            expected = dict(self.saved["parameters"], split_sampling_seed=seed)
            self.assertEqual(params, expected)
            on, off = (exp.effective_config(config, flag) for flag in (True, False))
            self.assertTrue(on.pop("use_dynamic_anchors"))
            self.assertFalse(off.pop("use_dynamic_anchors"))
            self.assertEqual(on, off)
        self.assertEqual(self.saved, original)

    def test_configuration_drift_is_rejected(self):
        self.saved["parameters"]["max_depth"] = 4
        with self.assertRaisesRegex(ValueError, "differ"):
            exp.model_config(self.saved, 42)

    def test_input_bytes_including_row_order_must_match(self):
        source, data = self.root / "source", self.root / "data"
        source.mkdir()
        data.mkdir()
        for name in exp.baseline.DATASET_FILES:
            (source / name).write_text("first\nsecond\n")
            (data / name).write_text("first\nsecond\n")
        self.assertEqual(len(exp.verify_inputs(source, data)), 5)
        (data / "observations.csv").write_text("second\nfirst\n")
        with self.assertRaisesRegex(ValueError, "observations.csv"):
            exp.verify_inputs(source, data)

    def test_prediction_comparison_detects_patient_level_changes(self):
        frame = predictions()
        self.assertTrue(exp.prediction_comparison(frame, frame.copy())["passed"])
        changed = frame.copy()
        changed.loc["a", "score"] += .01
        report = exp.prediction_comparison(frame, changed)
        self.assertFalse(report["passed"])
        self.assertEqual(report["scores_beyond_tolerance"], 1)
        self.assertFalse(exp.prediction_comparison(frame, frame.iloc[::-1])["passed"])
        changed = frame.copy()
        changed.loc["a", "documented"] = False
        self.assertFalse(exp.prediction_comparison(frame, changed)["passed"])

    def test_pairing_rejects_labels_subset_and_partition_changes(self):
        for column, value in (("FA", 1), ("documented", False),
                              ("partition", "train")):
            off = predictions()
            off.loc["a", column] = value
            with self.assertRaisesRegex(ValueError, "metadata differ"):
                exp.paired_auc(43, {"dynamic_on": predictions(), "dynamic_off": off})

    def test_pairing_uses_fixed_test_subsets_and_continuous_scores(self):
        on = predictions()
        off = on.copy()
        off["score"] = 1 - on.score
        rows = exp.paired_auc(43, {"dynamic_on": on, "dynamic_off": off})
        self.assertEqual([row["n"] for row in rows], [4, 3])
        self.assertEqual([row["cases"] for row in rows], [2, 1])
        self.assertEqual([row["delta"] for row in rows], [1.0, 1.0])
        off.loc["train", "score"] = 0
        self.assertEqual(rows, exp.paired_auc(43, {"dynamic_on": on, "dynamic_off": off}))

    def test_window_presence_and_available_unique_patient_coverage(self):
        folder = self.root / "on"
        artifacts(folder, True)
        result = exp.temporal_structure(folder, predictions())
        self.assertTrue(result["gi_one_month_present"])
        self.assertTrue(result["vomiting_one_month_present"])
        self.assertEqual(result["gi_one_month_used_n"], 1)
        self.assertEqual(result["vomiting_one_month_used_n"], 2)
        self.assertEqual(result["either_one_month_used_n"], 2)
        self.assertEqual(result["either_one_month_coverage"], 2 / 3)
        self.assertEqual(result["fish_anchor_used_n"], 2)
        self.assertEqual(result["fish_anchor_usage_coverage"], 2 / 3)
        tree = exp.read_json(folder / "tree.json")
        tree[1]["window"][1] = 2
        exp.write_json(folder / "tree.json", tree)
        result = exp.temporal_structure(folder, predictions())
        self.assertFalse(result["gi_one_month_present"])
        self.assertEqual(result["gi_one_month_used_n"], 0)
        artifacts(self.root / "off", False)
        self.assertFalse(exp.temporal_structure(self.root / "off", predictions())[
            "fish_anchor_creator_present"])

    def test_json_comparison_detects_routing_and_audit_changes(self):
        self.assertTrue(exp.differences({"score": .5}, {"score": .5 + 1e-14})[
            "within_tolerance"])
        self.assertFalse(exp.differences({"checks": [True]}, {"checks": [False]})[
            "within_tolerance"])
        self.assertFalse(exp.differences({"left": 1}, {"left": 2})[
            "within_tolerance"])

    def test_five_seed_summary_uses_paired_sample_sd_and_checks_cohort(self):
        pairs = {}
        for seed in exp.PLANNED_SEEDS:
            pairs[seed] = self.root / str(seed)
            for run in exp.RUNS:
                artifacts(pairs[seed] / run, run == "dynamic_on", seed)
            off = exp.load_predictions(pairs[seed] / "dynamic_off")
            if seed % 2:
                off["score"] = predictions().score
                off.to_csv(pairs[seed] / "dynamic_off/patient_predictions.csv")
        table, summary, structures = exp.summarize_seeds(pairs)
        self.assertEqual(len(table), 10)
        self.assertEqual(len(structures), 10)
        figure_path = self.root / "paired_auc.png"
        exp.plot_paired_auc(table, figure_path)
        self.assertGreater(figure_path.stat().st_size, 1000)
        with self.assertRaises(FileExistsError):
            exp.plot_paired_auc(table, figure_path)
        for row in summary.to_dict("records"):
            deltas = table.loc[table.population.eq(row["population"]), "delta"]
            self.assertAlmostEqual(row["delta_sd"], np.std(deltas, ddof=1))
            self.assertGreater(row["delta_sd"], 0)
        with self.assertRaisesRegex(ValueError, "exactly"):
            exp.summarize_seeds({42: self.root})
        for run in exp.RUNS:
            path = pairs[47] / run
            frame = exp.load_predictions(path)
            frame.loc["a", "block"] = 99
            frame.to_csv(path / "patient_predictions.csv")
        with self.assertRaisesRegex(ValueError, "Cohort differs"):
            exp.summarize_seeds(pairs)

    def test_checkpoint_runs_only_42_and_preserves_baseline(self):
        source, data = self.root / "source", self.root / "data"
        source.mkdir()
        data.mkdir()
        exp.write_json(source / "configuration.json", self.saved)
        for name in exp.baseline.DATASET_FILES:
            for folder in (source, data):
                (folder / name).write_text("immutable input")
        for run in exp.RUNS:
            artifacts(source / run, run == "dynamic_on")
        before = {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()}
        observed = []
        observations = pd.DataFrame({"feature_name": ["first", "second"]})
        patients = predictions()

        def fake_run(enabled, actual_data, actual_patients, folder, *, model_config):
            self.assertIs(actual_data, observations)
            self.assertIs(actual_patients, patients)
            self.assertEqual(model_config["split_sampling_seed"], 42)
            observed.append(enabled)
            diag = artifacts(folder, enabled)
            diag["training_runtime_seconds"] = 2.0
            exp.write_json(folder / "diagnostics.json", diag)
            return diag

        output = self.root / "new"
        with patch.object(exp.baseline, "load_dataset", return_value=(
            observations, patients, {"feature_order": ["first", "second"]}
        )), patch.object(exp, "run_model", side_effect=fake_run):
            report = exp.run_checkpoint(output, source, data)
            self.assertTrue(report["passed"])
            with self.assertRaises(FileExistsError):
                exp.run_checkpoint(output, source, data)
        self.assertEqual(observed, [True, False])
        self.assertEqual(exp.read_json(output / "status.json")["new_seeds_executed"], [])
        self.assertEqual(before, {str(p): p.read_bytes() for p in source.rglob("*")
                                  if p.is_file()})


    def mock_sweep(self):
        """Install an immutable tiny cohort and artifact-writing training stub."""
        source = self.root / "baseline"
        source.mkdir()
        exp.write_json(source / "configuration.json", self.saved)
        for run in exp.RUNS:
            artifacts(source / run, run == "dynamic_on")
        frozen = dict(baseline_configuration=self.saved,
                      feature_order=["first", "second"],
                      sd_definition="sample SD across seeds (ddof=1)")
        observations = pd.DataFrame({"feature_name": ["first", "second"]})
        patients = predictions()
        for target, value in (("BASELINE", source),):
            patcher = patch.object(exp, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        for target, value in ((exp, "protocol"), (exp.baseline, "load_dataset")):
            result = frozen if value == "protocol" else (
                observations, patients, {"feature_order": ["first", "second"]})
            patcher = patch.object(target, value, return_value=result)
            patcher.start()
            self.addCleanup(patcher.stop)
        calls = []

        def fake_fit(enabled, actual_data, actual_patients, folder, *, model_config):
            self.assertIs(actual_data, observations)
            self.assertIs(actual_patients, patients)
            seed = model_config["split_sampling_seed"]
            self.assertIn(seed, exp.PLANNED_SEEDS)
            self.assertEqual(model_config["random_state"], 42)
            calls.append((seed, enabled))
            return artifacts(folder, enabled, seed)

        return frozen, calls, fake_fit

    def test_manual_sweep_resume_summary_and_completion_recovery(self):
        frozen, calls, fake_fit = self.mock_sweep()
        output = self.root / "sweep"
        with patch.object(exp, "run_model", side_effect=fake_fit):
            summary = exp.run_sweep(output)
        self.assertEqual(calls, [(s, enabled) for s in exp.PLANNED_SEEDS
                                 for enabled in (True, False)])
        self.assertEqual(len(pd.read_csv(summary / "results_table.csv")), 6)
        stability = pd.read_csv(summary / "temporal_stability_summary.csv")
        self.assertEqual(stability.loc[stability.run == "dynamic_on",
                                      "gi_one_month_present_seeds"].item(), 5)
        self.assertTrue((summary / "paired_roc_auc.png").is_file())
        self.assertTrue((summary / "paired_roc_auc.pdf").is_file())
        before = {str(p): p.read_bytes() for p in (output / "seeds").rglob("*")
                  if p.is_file()}
        # Simulate interruption after completion marker, before publication.
        folder = output / "seeds/seed_47/dynamic_off"
        folder.rename(folder.with_name("dynamic_off.incomplete"))
        with patch.object(exp, "run_model", side_effect=AssertionError("No refits")):
            exp.run_sweep(output)
            exp.summarize_only(output)
        after = {str(p): p.read_bytes() for p in (output / "seeds").rglob("*")
                 if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(len(list((output / "history").iterdir())), 2)
        self.assertEqual(exp.read_json(output / "protocol.json"), frozen)

    def test_interrupted_attempt_requires_opt_in_and_preserves_completed_pair_member(self):
        frozen, calls, fake_fit = self.mock_sweep()
        output = self.root / "sweep"

        def interrupted(enabled, data, patients, folder, *, model_config):
            if not enabled:
                (folder / "partial.txt").write_text("preserve this")
                raise KeyboardInterrupt()
            return fake_fit(enabled, data, patients, folder, model_config=model_config)

        with patch.object(exp, "run_model", side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                exp.run_sweep(output)
        complete = output / "seeds/seed_43/dynamic_on"
        original = {p.name: p.read_bytes() for p in complete.iterdir()}
        with patch.object(exp, "run_model", side_effect=AssertionError("No refits")):
            with self.assertRaisesRegex(ValueError, "Incomplete run"):
                exp.run_sweep(output)
            with self.assertRaises(FileNotFoundError):
                exp.summarize_only(output)
        with patch.object(exp, "run_model", side_effect=fake_fit):
            exp.run_sweep(output, retry_incomplete=True)
        self.assertEqual(len(calls), 10)
        self.assertEqual(calls.count((43, True)), 1)
        self.assertEqual(original, {p.name: p.read_bytes() for p in complete.iterdir()})
        archived = list((output / "seeds/seed_43/interrupted").glob("*/partial.txt"))
        self.assertEqual(len(archived), 1)
        self.assertEqual(archived[0].read_text(), "preserve this")

    def test_corrupted_completion_and_protocol_changes_never_refit(self):
        frozen, calls, fake_fit = self.mock_sweep()
        output = self.root / "sweep"
        with patch.object(exp, "run_model", side_effect=fake_fit):
            exp.run_sweep(output)
        with patch.object(exp, "run_model", side_effect=AssertionError("No refits")):
            with patch.object(exp, "protocol", return_value=dict(frozen, changed=True)):
                with self.assertRaisesRegex(ValueError, "refusing mixed runs"):
                    exp.run_sweep(output)
            prediction_file = output / "seeds/seed_43/dynamic_on/patient_predictions.csv"
            prediction_file.write_text(prediction_file.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "artifacts changed"):
                exp.run_sweep(output, retry_incomplete=True)
            with self.assertRaisesRegex(ValueError, "artifacts changed"):
                exp.summarize_only(output)

    def test_protected_paths_unknown_directories_and_concurrent_writer(self):
        for protected in (exp.DATA, exp.BASELINE, exp.CHECKPOINT):
            with self.assertRaisesRegex(ValueError, "protected"):
                exp.protect_output(protected / "new")
        with exp.output_lock(self.root):
            with self.assertRaisesRegex(RuntimeError, "Another process"):
                with exp.output_lock(self.root):
                    self.fail("Second writer acquired the lock")
        with patch.object(exp, "protocol", return_value={}):
            with self.assertRaisesRegex(ValueError, "lacks an Experiment"):
                exp.run_sweep(self.root)


if __name__ == "__main__":
    unittest.main()
