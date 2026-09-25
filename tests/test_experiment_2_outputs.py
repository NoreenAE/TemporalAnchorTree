"""Saved-artifact post-processing checks; no dataset generation or training."""
import json

import pandas as pd
import pytest

from experiments.experiment_2_outputs import save_experiment_outputs


def _saved_run(path, run, labels=(0, 1, 0)):
    folder = path / run
    folder.mkdir()
    pd.DataFrame({
        "patient_id": ["a", "b", "c"], "partition": ["test"] * 3,
        "FA": labels, "score": [.1, .8, .3],
        "documented": [True, True, False], "birth": ["2000-01-01"] * 3,
        "fish_time": ["2001-01-01", "2001-02-01", None],
    }).to_csv(folder / "patient_predictions.csv", index=False)
    tree = [dict(node_id=0, source_node_id=0, patients=10, type="split",
                 feature="MinorInjury", aggregation="sum", window=["fish anchor", 1],
                 threshold=1, missing_direction="right", left=1, right=2),
            dict(node_id=1, patients=5, type="leaf", score=.1),
            dict(node_id=2, patients=5, type="leaf", score=.8)]
    (folder / "tree.json").write_text(json.dumps(tree))
    traces = []
    if run == "dynamic_on":
        for pid in ("a", "b"):
            traces.append(dict(patient_id=pid, node_id=3, source_node_id=3,
                               feature="FishIntroduction", anchor_name="fish anchor",
                               anchor_timestamps=["2001-01-01"], source_anchor=None))
        # A distractor-only visit counts as use. No symptom event is required.
        for pid, windows in (("a", [["2001-01-01", "2001-02-01"]]), ("c", [])):
            traces.append(dict(patient_id=pid, node_id=0, source_node_id=0,
                               feature="MinorInjury", anchor_name=None, source_anchor="fish anchor",
                               evaluated_windows=windows, offset=1, direction="right",
                               eligible_symptom_count=None))
    (folder / "anchor_trace.json").write_text(json.dumps(traces))


def test_exports_and_combined_figures(tmp_path):
    for run in ("dynamic_on", "dynamic_off"):
        _saved_run(tmp_path, run)
    save_experiment_outputs(tmp_path)
    summary = json.loads((tmp_path / "anchor_usage_test_summary.json").read_text())
    counts = summary["runs"]["dynamic_on"]["cohorts"]["documented"]
    assert counts["denominator"] == 2
    assert counts["fish_anchor_created"] == 2
    assert counts["fish_anchor_used"] == 1
    assert counts["fish_anchor_used_for_symptom"] == 0
    assert counts["created_not_used"] == 1
    windows = pd.read_csv(tmp_path / "anchor_usage_test_windows.csv")
    assert windows.visited.all()
    assert windows.anchor_available.tolist() == [True, False]
    assert windows.iloc[0].aggregation == "sum"
    assert windows.iloc[1].isna().window_start
    for stem in ("roc_on_vs_off", "decision_trees_on_vs_off"):
        for ext in ("png", "pdf"):
            assert (tmp_path / f"{stem}.{ext}").stat().st_size > 1000


def test_mismatched_test_labels_rejected(tmp_path):
    _saved_run(tmp_path, "dynamic_on")
    _saved_run(tmp_path, "dynamic_off", labels=(1, 0, 0))
    with pytest.raises(ValueError, match="held-out IDs and labels"):
        save_experiment_outputs(tmp_path)
    assert not (tmp_path / "roc_on_vs_off.png").exists()


@pytest.mark.parametrize("feature", [
    "FishIntroduction", "GI", "Vomiting", "MinorInjury", "Cough", "RoutineVaccination",
    "SleepDisturbance", "DoctorVisit",
])
def test_every_anchor_feature_is_exported_and_highlighted(tmp_path, feature):
    from matplotlib.colors import to_hex
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.pyplot as plt
    from experiments.experiment_2_outputs import _anchor_export, _tree_figure

    for run in ("dynamic_on", "dynamic_off"):
        _saved_run(tmp_path, run)
    folder = tmp_path / "dynamic_on"
    traces = json.loads((folder / "anchor_trace.json").read_text())
    name = f"Node 3 | SUM({feature}) Anchor"
    for trace in traces:
        if trace["anchor_name"]:
            trace["anchor_name"] = name
            trace["feature"] = feature
        if trace["source_anchor"] == "fish anchor":
            trace["source_anchor"] = name
    # A training-only creation must not appear in the held-out export.
    traces.append(dict(traces[0], patient_id="training_child"))
    (folder / "anchor_trace.json").write_text(json.dumps(traces))
    predictions, trees = {}, {}
    for run in ("dynamic_on", "dynamic_off"):
        predictions[run] = pd.read_csv(
            tmp_path / run / "patient_predictions.csv").set_index("patient_id")
        trees[run] = json.loads((tmp_path / run / "tree.json").read_text())
    trees["dynamic_on"][0]["window"] = [name, 1]
    trees["dynamic_off"][0]["window"] = ["Birth Date Anchor", 1]
    _anchor_export(tmp_path, predictions, trees)
    creations = pd.read_csv(tmp_path / "anchor_usage_test_creations.csv")
    assert len(creations) == 2
    assert set(creations.anchor_feature) == {feature}
    assert set(creations.patient_id) == {"a", "b"}
    windows = pd.read_csv(tmp_path / "anchor_usage_test_windows.csv")
    assert set(windows.anchor_feature) == {feature}
    summary = json.loads((tmp_path / "anchor_usage_test_summary.json").read_text())
    on = summary["runs"]["dynamic_on"]
    assert on["dynamic_anchors"][0]["created_children"] == 2
    assert on["dynamic_anchors"][0]["used_children"] == 1
    assert on["dynamic_anchors"][0]["created_not_used_children"] == 1
    assert on["cohorts"]["all_test"]["dynamic_anchor_used"] == 1
    assert on["cohorts"]["all_test"]["fish_anchor_used"] == int(feature == "FishIntroduction")
    figure = _tree_figure(trees)
    try:
        for axis, expected in zip(figure.axes, ("#f1eaf6", "#eaf3f8")):
            boxes = [p for p in axis.patches if isinstance(p, FancyBboxPatch)]
            assert to_hex(boxes[0].get_facecolor()) == expected
        assert any(f"after\n{feature}" in text.get_text()
                   for text in figure.axes[0].texts)
    finally:
        plt.close(figure)
