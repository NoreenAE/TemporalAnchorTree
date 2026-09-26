import json

import numpy as np
import pandas as pd

from experiments import experiment_1_fa_vs_gi as experiment


def _write_saved_usage(directory, birth_count, now_count, full_count):
    directory.mkdir(parents=True)
    total = birth_count + now_count + full_count
    pd.DataFrame({
        "anchor": ["Birth Date Anchor", "NOW Anchor"],
        "bin_start_offset": [0.0, -1.0],
        "bin_end_offset": [1.0, 0.0],
        "raw_usage_count": [birth_count, now_count],
        "normalized_usage_per_patient": [
            birth_count / 2,
            now_count / 2,
        ],
    }).to_csv(directory / "aggregated_temporal_usage.csv", index=False)
    (directory / "temporal_usage_summary.json").write_text(
        json.dumps({
            "evaluated_patient_count": 2,
            "temporal_split_usages": total,
            "full_timeline_usages": full_count,
        }),
        encoding="utf-8",
    )


def test_experiment_configuration_records_minimum_impurity_decrease():
    configuration = experiment.experiment_configuration({})

    assert configuration["min_impurity_decrease"] == (
        experiment.MIN_IMPURITY_DECREASE
    )
    assert configuration["model_hyperparameters"][
        "min_impurity_decrease"
    ] == experiment.MIN_IMPURITY_DECREASE


def test_experiment_comparison_is_created_from_saved_outcome_artifacts(
    tmp_path,
):
    _write_saved_usage(
        tmp_path / "fa" / "temporal_explainability",
        birth_count=6,
        now_count=1,
        full_count=3,
    )
    _write_saved_usage(
        tmp_path / "gi" / "temporal_explainability",
        birth_count=1,
        now_count=8,
        full_count=1,
    )

    output_path = experiment.create_temporal_usage_comparison(tmp_path)

    assert output_path == (
        tmp_path / "temporal_usage_comparison_fa_vs_gi.png"
    )
    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_run_experiment_creates_comparison_after_both_outcomes(
    monkeypatch,
    tmp_path,
):
    input_paths = {}
    for name in ("x", "fa", "gi"):
        path = tmp_path / f"{name}.pkl"
        path.write_bytes(name.encode("utf-8"))
        input_paths[name] = path
    output_dir = tmp_path / "results"
    events = []

    monkeypatch.setattr(experiment, "INPUT_PATHS", input_paths)
    monkeypatch.setattr(experiment, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(
        experiment,
        "load_inputs",
        lambda: (pd.DataFrame(), {"fa": pd.DataFrame(), "gi": pd.DataFrame()}),
    )
    monkeypatch.setattr(
        experiment,
        "create_shared_split",
        lambda data, labels: (
            np.asarray(["P1"]),
            np.asarray(["P2"]),
            pd.DataFrame({"patient_id": ["P1", "P2"]}),
        ),
    )

    def fake_run_outcome(**kwargs):
        outcome = kwargs["outcome"]
        events.append(outcome)
        return {
            "performance": {"roc_auc": 0.5},
            "runtime_seconds": {"training": 0.0, "total": 0.0},
            "candidate_evaluations": {"total_evaluated_candidates": 0},
            "root_split": {"anchor": None, "offset": None},
            "temporal_usage": {
                "birth_date_usage_proportion": 0.0,
                "now_usage_proportion": 0.0,
                "full_timeline_usage_proportion": 1.0,
            },
            "window_distribution": [],
        }

    def fake_create_comparison(run_output_dir):
        assert events == ["fa", "gi"]
        events.append("comparison")
        output = run_output_dir / experiment.TEMPORAL_USAGE_COMPARISON_FILENAME
        output.write_bytes(b"png")
        return output

    monkeypatch.setattr(experiment, "run_outcome", fake_run_outcome)
    monkeypatch.setattr(
        experiment,
        "create_temporal_usage_comparison",
        fake_create_comparison,
    )
    def fake_create_roc_comparison(run_output_dir, results):
        assert events == ["fa", "gi", "comparison"]
        assert set(results) == {"fa", "gi"}
        assert run_output_dir == output_dir
        events.append("roc")

    monkeypatch.setattr(
        experiment, "create_roc_comparison", fake_create_roc_comparison
    )
    def fake_create_thesis_trees(run_output_dir):
        assert events == ["fa", "gi", "comparison", "roc"]
        assert run_output_dir == output_dir
        events.append("trees")

    monkeypatch.setattr(
        experiment, "create_thesis_tree_figures", fake_create_thesis_trees
    )

    result = experiment.run_experiment()

    assert events == ["fa", "gi", "comparison", "roc", "trees"]
    assert result["artifacts"]["temporal_usage_comparison"] == str(
        output_dir / experiment.TEMPORAL_USAGE_COMPARISON_FILENAME
    )


def test_roc_comparison_uses_saved_continuous_scores(monkeypatch, tmp_path):
    from matplotlib.figure import Figure
    from settree.eval_utils import save_roc_curves

    captured = []
    original_savefig = Figure.savefig

    def capture_figure(figure, *args, **kwargs):
        captured.append(figure.axes[0])
        return original_savefig(figure, *args, **kwargs)

    def forbid_model_use(*args, **kwargs):
        raise AssertionError("Plotting must not fit or predict.")

    monkeypatch.setattr(Figure, "savefig", capture_figure)
    monkeypatch.setattr(experiment.TemporalTree, "fit", forbid_model_use)
    monkeypatch.setattr(experiment.TemporalTree, "predict", forbid_model_use)
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.4, 0.35, 0.8]
    for outcome in ("fa", "gi"):
        directory = tmp_path / outcome
        directory.mkdir()
        pd.DataFrame({"label": labels, "score": scores}).to_csv(
            directory / "test_predictions.csv", index=False
        )
        (directory / "metrics_and_temporal_summary.json").write_text(
            json.dumps({"performance": {"roc_auc": 0.75}}),
            encoding="utf-8",
        )

    output = experiment.create_roc_comparison(tmp_path)
    assert output == tmp_path / "roc_curves_fa_vs_gi.png"
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    axis = captured[-1]
    assert axis.get_title() == "ROC Curves for FA and GI"
    assert axis.get_xlabel() == "False Positive Rate"
    assert axis.get_ylabel() == "True Positive Rate"
    assert axis.get_legend_handles_labels()[1] == [
        "FA (AUC = 0.750)", "GI (AUC = 0.750)"
    ]
    for line in axis.lines[:2]:
        np.testing.assert_allclose(line.get_xdata(), [0, 0, 0.5, 0.5, 1])
        np.testing.assert_allclose(line.get_ydata(), [0, 0.5, 0.5, 1, 1])
    np.testing.assert_allclose(axis.lines[2].get_xdata(), [0, 1])
    np.testing.assert_allclose(axis.lines[2].get_ydata(), [0, 1])

    save_roc_curves({"ROC": (labels, scores, 0.75)}, tmp_path / "roc_curve.png")
    assert captured[-1].get_legend_handles_labels()[1] == ["ROC (AUC = 0.750)"]


def test_thesis_trees_render_saved_rules_without_model_use(monkeypatch, tmp_path):
    from matplotlib.figure import Figure
    from settree.eval_utils import _thesis_tree_label

    nodes = [
        {"node_id": 0, "type": "split", "feature": "FormulaYN",
         "anchor": "Birth Date Anchor", "offset": 6, "aggregation": "mean",
         "threshold": 0.42, "left_child": 1, "right_child": 2},
        {"node_id": 1, "type": "leaf", "num_patients": 20, "prediction": 0.1},
        {"node_id": 2, "type": "leaf", "num_patients": 30, "prediction": 0.8},
    ]
    originals = {}
    for outcome, records in (("fa", nodes), ("gi", [nodes[1]])):
        directory = tmp_path / outcome
        directory.mkdir()
        path = directory / "tree.json"
        path.write_text(json.dumps(records), encoding="utf-8")
        originals[path] = path.read_bytes()

    def forbid_model_use(*args, **kwargs):
        raise AssertionError("Tree plotting must not execute a model.")

    monkeypatch.setattr(experiment.TemporalTree, "fit", forbid_model_use)
    monkeypatch.setattr(experiment.TemporalTree, "predict", forbid_model_use)
    captured = []
    original_savefig = Figure.savefig

    def inspect_figure(figure, *args, **kwargs):
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        for axis in figure.axes:
            boxes = [
                text.get_bbox_patch().get_window_extent(renderer)
                for text in axis.texts if text.get_zorder() == 3
            ]
            for index, box in enumerate(boxes):
                assert not any(box.overlaps(other) for other in boxes[index + 1:])
                assert not box.overlaps(axis._left_title.get_window_extent(renderer))
            captured.append(([text.get_text() for text in axis.texts],
                             axis.get_title(loc="left")))
        return original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", inspect_figure)
    paths = experiment.create_thesis_tree_figures(tmp_path)
    assert paths == {
        "fa": tmp_path / "fa" / "fa_tree_thesis.png",
        "gi": tmp_path / "gi" / "gi_tree_thesis.png",
        "combined": tmp_path / "learned_trees_fa_vs_gi_thesis.png",
    }
    assert all(path.read_bytes().startswith(b"\x89PNG") for path in paths.values())
    assert all(path.read_bytes() == content for path, content in originals.items())
    assert "FormulaYN\nBirth Date: 0–6 months\nMean ≤ 0.42" in captured[0][0]
    assert "Patients = 20\nP(positive) = 0.100" in captured[0][0]
    assert captured[-2][1] == "(a) FA learned tree"
    assert captured[-1][1] == "(b) GI learned tree"
    assert "NOW: −1 to 0 months" in _thesis_tree_label(
        dict(nodes[0], anchor="NOW Anchor", offset=-1)
    )
    assert "Full timeline" in _thesis_tree_label(dict(nodes[0], offset=None))
    assert "0–21 days" in _thesis_tree_label(dict(nodes[0], offset=0.7))
