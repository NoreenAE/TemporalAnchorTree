"""Saved-path usage tests; no model construction, fitting, or prediction."""
import json

import pandas as pd
import pytest

from experiments import experiment_2_temporal_explainability as usage
from experiments import experiment_2_noisy_v2 as runner


def saved_artifacts(folder):
    """Small artifact fixture testing changed offsets, full and missing windows."""
    folder.mkdir()
    on = folder / "dynamic_on"
    on.mkdir()
    patients = pd.DataFrame({
        "patient_id": ["a", "b", "c"], "partition": ["test"] * 3,
        "FA": [1, 0, 0], "score": [.8, .2, .1],
        "documented": [True, False, True],
    })
    patients.drop(columns="score").to_csv(
        folder / "patients_and_split.csv", index=False)
    patients.to_csv(on / "patient_predictions.csv", index=False)
    (folder / "configuration.json").write_text(json.dumps({
        "dataset_version": "fish_intro_noisy_matched_v2",
        "parameters": {"anchor_offset_list": [1],
                       "anchor_offset_map": {"Birth Date Anchor": [1, None]}},
    }))
    nodes = [
        dict(node_id=0, type="split", feature="FishIntroduction",
             window=["Birth Date Anchor", 3], left=1, right=4),
        dict(node_id=1, type="split", feature="GI",
             window=["saved fish anchor", 2], left=2, right=3),
        dict(node_id=2, type="leaf", score=.8),
        dict(node_id=3, type="leaf", score=.1),
        dict(node_id=4, type="split", feature="DoctorVisit",
             window=[None, None], left=5, right=6),
        dict(node_id=5, type="leaf", score=.2),
        dict(node_id=6, type="leaf", score=.3),
    ]
    (on / "tree.json").write_text(json.dumps(nodes))
    traces = []
    for pid in ("a", "b", "c"):
        traces.append(dict(
            patient_id=pid, node_id=0, feature="FishIntroduction",
            source_anchor="Birth Date Anchor", offset=3,
            direction="right" if pid == "b" else "left",
            anchor_name="saved fish anchor",
            anchor_timestamps=["2020-02-01"] if pid == "a" else [],
            evaluated_windows=[["2020-01-01", "2020-04-01"]],
            checks={"passed": True},
        ))
        traces.append(dict(
            patient_id=pid, node_id=4 if pid == "b" else 1,
            feature="DoctorVisit" if pid == "b" else "GI",
            source_anchor=None if pid == "b" else "saved fish anchor",
            offset=None if pid == "b" else 2,
            direction="right" if pid == "c" else "left", anchor_name=None,
            evaluated_windows=[["2020-02-01", "2020-04-01"]]
            if pid == "a" else [], checks={"passed": True},
        ))
    (on / "anchor_trace.json").write_text(json.dumps(traces))


def test_overlap_denominator_source_anchor_and_future_windows(tmp_path):
    saved_artifacts(tmp_path / "run")
    table, summary, edges = usage.calculate_usage(tmp_path / "run")
    assert summary["temporal_split_usages"] == 6
    assert summary["evaluated_patient_count"] == 3
    assert summary["reference_visit_counts"] == {
        usage.FISH: 2, usage.BIRTH: 3, usage.FULL: 1}
    assert summary["unavailable_reference_visits"] == {usage.FISH: 1}
    assert edges[usage.FISH] == [0, 1, 2]
    assert edges[usage.BIRTH] == [0, 1, 2, 3]
    assert table.loc[table.reference_type.eq(usage.FISH),
                     "visit_count"].tolist() == [1, 1]
    assert table.loc[table.reference_type.eq(usage.BIRTH),
                     "visit_count"].tolist() == [3, 3, 3]
    assert table.loc[table.reference_type.eq(usage.FULL),
                     "visit_count"].item() == 1
    assert table.usage_percentage.tolist() == pytest.approx(
        [100 / 6, 100 / 6, 50, 50, 50, 100 / 6])


def test_incomplete_trace_rejected_before_outputs(tmp_path):
    run = tmp_path / "run"
    saved_artifacts(run)
    trace = run / "dynamic_on/anchor_trace.json"
    trace.write_text(json.dumps(json.loads(trace.read_text())[:-1]))
    with pytest.raises(ValueError, match="does not reach a leaf"):
        usage.save_temporal_reference_usage(run)
    assert not (run / "dynamic_on/temporal_explainability_transition").exists()


def test_current_audit_does_not_force_expected_counts(tmp_path):
    run = tmp_path / "run"
    saved_artifacts(run)
    with pytest.raises(ValueError, match="Current saved-run audit mismatch"):
        usage.save_temporal_reference_usage(run, verify_current=True)
    assert not (run / "dynamic_on/temporal_explainability_transition").exists()


def test_transition_excludes_noncreated_patients_and_creation_from_after(tmp_path):
    from experiments.experiment_2_anchor_transition import (
        AFTER, BEFORE, calculate_transition,
    )
    run = tmp_path / "run"
    saved_artifacts(run)
    table, summary = calculate_transition(run)
    assert summary["anchor_created_test_patients"] == 1
    assert [p["patient_id"] for p in summary["patients"]] == ["a"]
    assert summary["stages"][BEFORE]["total_split_visits"] == 1
    assert summary["stages"][AFTER]["total_split_visits"] == 1
    assert summary["stages"][BEFORE]["reference_counts"]["Birth Date"] == 1
    assert summary["stages"][AFTER]["reference_counts"]["FishIntroduction"] == 1
    assert all(row["stage"] == BEFORE for row in summary["stage_assignments"]
               if row["is_first_fish_creation"])
    for stage in (BEFORE, AFTER):
        assert table.loc[table.stage.eq(stage), "usage_percentage"].sum() == 100


def test_transition_finds_nonroot_creation_and_preserves_other_reference(tmp_path):
    from experiments.experiment_2_anchor_transition import (
        AFTER, BEFORE, calculate_transition,
    )
    run = tmp_path / "run"
    saved_artifacts(run)
    tree_path = run / "dynamic_on/tree.json"
    tree = json.loads(tree_path.read_text())
    tree[0].update(feature="DoctorVisit", window=["NOW Anchor", 3])
    tree[1].update(feature="FishIntroduction", window=["Birth Date Anchor", 3])
    tree[2] = dict(node_id=2, type="split", feature="GI",
                   window=["saved fish anchor", 2], left=7, right=8)
    tree.extend([dict(node_id=7, type="leaf", score=.8),
                 dict(node_id=8, type="leaf", score=.9)])
    tree_path.write_text(json.dumps(tree))
    trace_path = run / "dynamic_on/anchor_trace.json"
    traces = json.loads(trace_path.read_text())
    for trace in traces:
        if trace["node_id"] == 0:
            trace.update(feature="DoctorVisit", source_anchor="NOW Anchor",
                         anchor_name=None, anchor_timestamps=[])
        elif trace["node_id"] == 1:
            trace.update(feature="FishIntroduction", source_anchor="Birth Date Anchor",
                         offset=3, evaluated_windows=[["2020-01-01", "2020-04-01"]],
                         anchor_name="saved fish anchor",
                         anchor_timestamps=["2020-02-01"]
                         if trace["patient_id"] == "a" else [])
    traces.insert(2, dict(
        patient_id="a", node_id=2, feature="GI", source_anchor="saved fish anchor",
        offset=2, direction="left", anchor_name=None,
        evaluated_windows=[["2020-02-01", "2020-04-01"]], checks={"passed": True}))
    trace_path.write_text(json.dumps(traces))
    table, summary = calculate_transition(run)
    assert not summary["all_creation_nodes_are_root"]
    assert summary["patients"][0]["creation_visit_index"] == 1
    assert summary["stages"][BEFORE]["total_split_visits"] == 2
    assert summary["stages"][AFTER]["total_split_visits"] == 1
    assert summary["unexpected_reference_categories"] == ["Other: NOW Anchor"]
    assert summary["stages"][BEFORE]["reference_percentages"]["Other: NOW Anchor"] == 50
    assert summary["stages"][AFTER]["reference_percentages"]["FishIntroduction"] == 100
    assert table.groupby("stage").usage_percentage.sum().tolist() == [100, 100]


def test_transition_no_downstream_visits_has_undefined_percentages(tmp_path):
    from experiments.experiment_2_anchor_transition import AFTER, calculate_transition
    run = tmp_path / "run"
    saved_artifacts(run)
    tree_path = run / "dynamic_on/tree.json"
    tree = json.loads(tree_path.read_text())
    tree[1] = dict(node_id=1, type="leaf", score=.8)
    # Keep one selected patient and the root/leaf structure for this edge case.
    tree = [tree[0], tree[1], tree[4], tree[5], tree[6]]
    tree_path.write_text(json.dumps(tree))
    for name in ("patients_and_split.csv", "dynamic_on/patient_predictions.csv"):
        path = run / name
        data = pd.read_csv(path)
        data.loc[data.patient_id.eq("a")].to_csv(path, index=False)
    trace_path = run / "dynamic_on/anchor_trace.json"
    trace_path.write_text(json.dumps(json.loads(trace_path.read_text())[:1]))
    table, summary = calculate_transition(run)
    assert summary["stages"][AFTER]["total_split_visits"] == 0
    assert table.loc[table.stage.eq(AFTER), "usage_percentage"].isna().all()
    assert summary["patient_diagnostics"]["no_post_split"] == 1


def test_progression_uses_distinct_units_and_reports_repeated_visits(tmp_path):
    run = tmp_path / "run"
    saved_artifacts(run)
    tree_path = run / "dynamic_on/tree.json"
    tree = json.loads(tree_path.read_text())
    # Patient a evaluates the same feature/window at two distinct split nodes.
    tree[2] = dict(node_id=2, type="split", feature="GI",
                   window=["saved fish anchor", 2], left=7, right=8)
    tree.extend([dict(node_id=7, type="leaf", score=.8),
                 dict(node_id=8, type="leaf", score=.9)])
    tree_path.write_text(json.dumps(tree))
    trace_path = run / "dynamic_on/anchor_trace.json"
    traces = json.loads(trace_path.read_text())
    traces.insert(2, dict(traces[1], node_id=2))
    trace_path.write_text(json.dumps(traces))
    tables, summary = usage.calculate_progression(run)
    stages = tables["fish_anchor_patient_usage"]
    assert stages.patient_count.tolist() == [2, 1, 1]
    assert stages.documented_test_patients.tolist() == [2, 2, 2]
    assert stages.usage_percentage.tolist() == [100, 50, 50]
    symptoms = tables["fish_relative_symptom_usage"]
    gi = symptoms.loc[symptoms.feature.eq("GI")]
    assert gi.visit_count.tolist() == [2, 2]
    assert gi.unique_patient_count.tolist() == [1, 1]
    assert gi.repeated_patient_contributions.tolist() == [1, 1]
    assert symptoms.loc[symptoms.feature.eq("Vomiting"),
                        "visit_count"].tolist() == [0, 0]
    assert not summary["symptom_counts_equal_unique_patients"]
    assert summary["symptom_selected_windows"] == [dict(
        feature="GI", offset_months=2, visit_count=2, unique_patient_count=1)]
    assert summary["full_timeline_usages"] == 1
    assert tables["birth_date_temporal_usage"].total_split_visits.tolist() == [7]*3


def test_other_fish_relative_features_are_audited_not_added_to_symptoms(tmp_path):
    run = tmp_path / "run"
    saved_artifacts(run)
    for name in ("tree.json", "anchor_trace.json"):
        path = run / "dynamic_on" / name
        rows = json.loads(path.read_text())
        for row in rows:
            if row.get("feature") == "GI":
                row["feature"] = "Cough"
        path.write_text(json.dumps(rows))
    tables, summary = usage.calculate_progression(run)
    assert set(tables["fish_relative_symptom_usage"].feature) == {"GI", "Vomiting"}
    assert summary["other_available_fish_relative_feature_visits"] == {"Cough": 1}
    assert summary["fish_anchor_available_at_downstream_split_patients"] == 1
    assert summary["fish_anchor_used_for_symptom_patients"] == 0


@pytest.mark.parametrize("plot_fails", [False, True])
def test_future_hook_after_on_and_failure_keeps_off_run(
        monkeypatch, tmp_path, plot_fails):
    source = tmp_path / "input"
    source.mkdir()
    for name in runner.DATASET_FILES:
        (source / name).write_text("{}")
    monkeypatch.setattr(runner, "load_dataset",
                        lambda path: (object(), object(), {"config": {}}))
    calls = []

    def saved_run(enabled, data, patients, output, **kwargs):
        calls.append("on" if enabled else "off")
        return {"saved": True}

    def postprocess(results):
        assert calls == ["on"]
        calls.append("plot")
        if plot_fails:
            raise ValueError("fixture post-processing failure")

    monkeypatch.setattr(runner, "run_model", saved_run)
    monkeypatch.setattr(usage, "save_temporal_reference_usage", postprocess)
    monkeypatch.setattr(runner, "save_experiment_outputs", lambda path: None)
    output = tmp_path / "output"
    monkeypatch.setattr("sys.argv", ["runner", "--data-dir", str(source),
                                     "--output", str(output)])
    if plot_fails:
        with pytest.warns(RuntimeWarning, match="scientific results are retained"):
            runner.main()
    else:
        runner.main()
    assert calls == ["on", "plot", "off"]
    assert json.loads((output / "comparison.json").read_text()) == {
        "dynamic_on": {"saved": True}, "dynamic_off": {"saved": True}}
