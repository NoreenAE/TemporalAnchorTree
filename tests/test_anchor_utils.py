import numpy as np
import pandas as pd
import pytest

from settree.anchor_utils import create_branch_anchors
from settree.temporal_tree import TemporalSplitNode, TemporalTree


def test_create_branch_anchors_handles_scalar_list_empty_and_nan_source_anchors():
    source_anchor = "source Anchor"
    output_anchor = "output Anchor"
    rows = [
        {"patient_id": "scalar", "time": pd.Timestamp("2020-01-01"), "feature_name": source_anchor, "value": pd.Timestamp("2020-01-01")},
        {"patient_id": "scalar", "time": pd.Timestamp("2020-02-01"), "feature_name": "FormulaYN", "value": 1.0},
        {"patient_id": "scalar", "time": pd.Timestamp("2020-04-01"), "feature_name": "FormulaYN", "value": 1.0},
        {"patient_id": "one_list", "time": pd.NaT, "feature_name": source_anchor, "value": [pd.Timestamp("2020-03-01")]},
        {"patient_id": "one_list", "time": pd.Timestamp("2020-04-01"), "feature_name": "FormulaYN", "value": 1.0},
        {"patient_id": "one_list", "time": pd.Timestamp("2020-06-01"), "feature_name": "FormulaYN", "value": 1.0},
        {"patient_id": "multi_list", "time": pd.NaT, "feature_name": source_anchor, "value": [pd.Timestamp("2016-03-26"), pd.Timestamp("2016-05-28")]},
        {"patient_id": "multi_list", "time": pd.Timestamp("2016-04-15"), "feature_name": "FormulaYN", "value": 1.0},
        {"patient_id": "multi_list", "time": pd.Timestamp("2016-06-15"), "feature_name": "FormulaYN", "value": 1.0},
        {"patient_id": "multi_list", "time": pd.Timestamp("2016-09-01"), "feature_name": "FormulaYN", "value": 1.0},
        {"patient_id": "empty_list", "time": pd.NaT, "feature_name": source_anchor, "value": []},
        {"patient_id": "empty_list", "time": pd.Timestamp("2020-02-01"), "feature_name": "FormulaYN", "value": 1.0},
        {"patient_id": "nan_anchor", "time": pd.NaT, "feature_name": source_anchor, "value": np.nan},
        {"patient_id": "nan_anchor", "time": pd.Timestamp("2020-02-01"), "feature_name": "FormulaYN", "value": 1.0},
    ]
    X_branch = pd.DataFrame(rows)

    result = create_branch_anchors(
        patient_ids=["scalar", "one_list", "multi_list", "empty_list", "nan_anchor"],
        X_branch=X_branch,
        anchor_name=output_anchor,
        direction="right",
        feature="FormulaYN",
        threshold=0.4,
        time_window=(source_anchor, 2),
    )

    values = result.set_index("patient_id")["value"].to_dict()
    assert values["scalar"] == [pd.Timestamp("2020-02-01")]
    assert values["one_list"] == [pd.Timestamp("2020-04-01")]
    assert values["multi_list"] == [pd.Timestamp("2016-04-15"), pd.Timestamp("2016-06-15")]
    assert values["empty_list"] == []
    assert values["nan_anchor"] == []


def test_create_branch_anchors_handles_numpy_array_source_anchors_like_lists():
    source_anchor = "source Anchor"
    rows = [
        {"patient_id": "list_anchor", "time": pd.NaT, "feature_name": source_anchor, "value": [pd.Timestamp("2016-03-26"), pd.Timestamp("2016-05-28")]},
        {"patient_id": "list_anchor", "time": pd.Timestamp("2016-04-15"), "feature_name": "FormulaYN", "value": 1.0},
        {"patient_id": "list_anchor", "time": pd.Timestamp("2016-06-15"), "feature_name": "FormulaYN", "value": 1.0},
        {"patient_id": "array_anchor", "time": pd.NaT, "feature_name": source_anchor, "value": np.array([pd.Timestamp("2016-03-26"), pd.Timestamp("2016-05-28")])},
        {"patient_id": "array_anchor", "time": pd.Timestamp("2016-04-15"), "feature_name": "FormulaYN", "value": 1.0},
        {"patient_id": "array_anchor", "time": pd.Timestamp("2016-06-15"), "feature_name": "FormulaYN", "value": 1.0},
        {"patient_id": "one_array", "time": pd.NaT, "feature_name": source_anchor, "value": np.array([pd.Timestamp("2020-03-01")])},
        {"patient_id": "one_array", "time": pd.Timestamp("2020-04-01"), "feature_name": "FormulaYN", "value": 1.0},
        {"patient_id": "empty_array", "time": pd.NaT, "feature_name": source_anchor, "value": np.array([])},
        {"patient_id": "empty_array", "time": pd.Timestamp("2020-04-01"), "feature_name": "FormulaYN", "value": 1.0},
    ]
    X_branch = pd.DataFrame(rows)

    result = create_branch_anchors(
        patient_ids=["list_anchor", "array_anchor", "one_array", "empty_array"],
        X_branch=X_branch,
        anchor_name="output Anchor",
        direction="right",
        feature="FormulaYN",
        threshold=0.4,
        time_window=(source_anchor, 2),
    )

    values = result.set_index("patient_id")["value"].to_dict()
    assert values["array_anchor"] == values["list_anchor"]
    assert values["one_array"] == [pd.Timestamp("2020-04-01")]
    assert values["empty_array"] == []


def test_create_branch_anchors_full_timeline_right_and_left():
    X_branch = pd.DataFrame([
        {"patient_id": "P1", "time": pd.Timestamp("2020-01-01"), "feature_name": "FormulaYN", "value": 1.0},
        {"patient_id": "P1", "time": pd.Timestamp("2020-03-01"), "feature_name": "FormulaYN", "value": 0.0},
        {"patient_id": "P1", "time": pd.Timestamp("2020-05-01"), "feature_name": "FormulaYN", "value": 1.0},
        {"patient_id": "P1", "time": pd.Timestamp("2020-08-01"), "feature_name": "FormulaYN", "value": 0.0},
    ])

    right = create_branch_anchors(["P1"], X_branch, "right Anchor", "right", "FormulaYN", 0.4, (None, None))
    left = create_branch_anchors(["P1"], X_branch, "left Anchor", "left", "FormulaYN", 0.4, (None, None))

    assert right.loc[0, "value"] == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-05-01")]
    assert left.loc[0, "value"] == [pd.Timestamp("2020-03-01"), pd.Timestamp("2020-08-01")]


def test_prediction_full_timeline_anchor_matches_training_anchor():
    model = TemporalTree()
    node = TemporalSplitNode(feature="FormulaYN", threshold=0.4, aggregation="mean", time_window=(None, None))
    row = pd.DataFrame([
        {"patient_id": "P1", "time": pd.Timestamp("2020-01-01"), "feature_name": "FormulaYN", "value": 1.0},
        {"patient_id": "P1", "time": pd.Timestamp("2020-03-01"), "feature_name": "FormulaYN", "value": 0.0},
        {"patient_id": "P1", "time": pd.Timestamp("2020-05-01"), "feature_name": "FormulaYN", "value": 1.0},
        {"patient_id": "P1", "time": pd.Timestamp("2020-08-01"), "feature_name": "FormulaYN", "value": 0.0},
    ])
    anchor_name = model._format_dynamic_anchor_name(node, "right")
    training_anchor = create_branch_anchors(["P1"], row, anchor_name, "right", "FormulaYN", 0.4, (None, None))
    prediction_row = model._append_prediction_branch_anchor(row, node, "right")
    prediction_anchor = prediction_row[prediction_row["feature_name"] == anchor_name]

    assert len(prediction_anchor) == 1
    assert training_anchor.loc[0, "feature_name"] == prediction_anchor.iloc[0]["feature_name"]
    assert training_anchor.loc[0, "value"] == prediction_anchor.iloc[0]["value"]


def test_dynamic_anchor_name_uses_full_timeline_display_only_for_none_window():
    model = TemporalTree()
    full_timeline_node = TemporalSplitNode(
        feature="FormulaYN",
        threshold=0.0,
        aggregation="mean",
        time_window=(None, None),
    )
    normal_node = TemporalSplitNode(
        feature="FormulaYN",
        threshold=0.5,
        aggregation="mean",
        time_window=("Birth Date Anchor", 3),
    )

    assert (
        model._format_dynamic_anchor_name(full_timeline_node, "right")
        == "FormulaYN > 0.000 | Full Timeline Anchor"
    )
    assert (
        model._format_dynamic_anchor_name(normal_node, "left")
        == "FormulaYN <= 0.500 | Birth Date Anchor +3m Anchor"
    )


def test_dynamic_anchor_key_keeps_exact_none_window_identity():
    model = TemporalTree()
    node = TemporalSplitNode(
        feature="FormulaYN",
        threshold=0.0,
        aggregation="mean",
        time_window=(None, None),
    )

    assert model._dynamic_anchor_key(node, "right") == (
        "FormulaYN",
        "right",
        0.0,
        None,
        None,
    )


def test_repeated_full_timeline_anchor_reuses_same_semantic_key():
    model = TemporalTree()
    node = TemporalSplitNode(
        feature="FormulaYN",
        threshold=0.0,
        aggregation="mean",
        time_window=(None, None),
    )
    X_branch = pd.DataFrame([
        {"patient_id": "P1", "time": pd.Timestamp("2020-01-01"), "feature_name": "FormulaYN", "value": 1.0},
    ])
    anchor_registry = {}

    anchor_name, anchors, anchor_df, anchor_registry = model._prepare_branch_anchor(
        node=node,
        patient_ids=["P1"],
        X_branch=X_branch,
        anchors=[],
        anchor_registry=anchor_registry,
        direction="right",
    )
    reused_name, reused_anchors, reused_df, reused_registry = model._prepare_branch_anchor(
        node=node,
        patient_ids=["P1"],
        X_branch=X_branch,
        anchors=anchors,
        anchor_registry=anchor_registry,
        direction="right",
    )

    assert anchor_name == "FormulaYN > 0.000 | Full Timeline Anchor"
    assert reused_name == anchor_name
    assert reused_anchors == anchors
    assert not anchor_df.empty
    assert reused_df.empty
    assert reused_registry == anchor_registry


def test_dynamic_anchor_key_distinguishes_exact_thresholds_that_display_same():
    model = TemporalTree()
    first_node = TemporalSplitNode(
        feature="FormulaYN",
        threshold=0.0001,
        aggregation="mean",
        time_window=(None, None),
    )
    second_node = TemporalSplitNode(
        feature="FormulaYN",
        threshold=0.0002,
        aggregation="mean",
        time_window=(None, None),
    )

    assert model._format_dynamic_anchor_name(first_node, "right") == (
        model._format_dynamic_anchor_name(second_node, "right")
    )
    assert model._dynamic_anchor_key(first_node, "right") != (
        model._dynamic_anchor_key(second_node, "right")
    )


def test_prediction_numpy_array_source_anchor_matches_training_anchor():
    model = TemporalTree()
    source_anchor = "source Anchor"
    node = TemporalSplitNode(feature="FormulaYN", threshold=0.4, aggregation="mean", time_window=(source_anchor, 2))
    row = pd.DataFrame([
        {"patient_id": "P1", "time": pd.NaT, "feature_name": source_anchor, "value": np.array([pd.Timestamp("2016-03-26"), pd.Timestamp("2016-05-28")])},
        {"patient_id": "P1", "time": pd.Timestamp("2016-04-15"), "feature_name": "FormulaYN", "value": 1.0},
        {"patient_id": "P1", "time": pd.Timestamp("2016-06-15"), "feature_name": "FormulaYN", "value": 1.0},
        {"patient_id": "P1", "time": pd.Timestamp("2016-09-01"), "feature_name": "FormulaYN", "value": 1.0},
    ])
    anchor_name = model._format_dynamic_anchor_name(node, "right")
    training_anchor = create_branch_anchors(["P1"], row, anchor_name, "right", "FormulaYN", 0.4, (source_anchor, 2))
    prediction_row = model._append_prediction_branch_anchor(row, node, "right")
    prediction_anchor = prediction_row[prediction_row["feature_name"] == anchor_name]

    assert len(prediction_anchor) == 1
    assert training_anchor.loc[0, "value"] == prediction_anchor.iloc[0]["value"]


def test_create_branch_anchors_non_none_window_unchanged():
    X_branch = pd.DataFrame([
        {"patient_id": "P1", "time": pd.Timestamp("2020-01-01"), "feature_name": "Birth Date Anchor", "value": pd.Timestamp("2020-01-01")},
        {"patient_id": "P1", "time": pd.Timestamp("2020-02-01"), "feature_name": "FormulaYN", "value": 1.0},
        {"patient_id": "P1", "time": pd.Timestamp("2020-08-01"), "feature_name": "FormulaYN", "value": 1.0},
    ])

    result = create_branch_anchors(["P1"], X_branch, "right Anchor", "right", "FormulaYN", 0.4, ("Birth Date Anchor", 3))

    assert result.loc[0, "value"] == [pd.Timestamp("2020-02-01")]


def test_create_branch_anchors_empty_feature_rows_full_timeline():
    X_branch = pd.DataFrame([
        {"patient_id": "P1", "time": pd.Timestamp("2020-01-01"), "feature_name": "OtherFeature", "value": 1.0},
    ])

    result = create_branch_anchors(["P1"], X_branch, "right Anchor", "right", "FormulaYN", 0.4, (None, None))

    assert result.loc[0, "value"] == []


def _aggregation_frame(anchor_rows):
    rows = [
        {"patient_id": patient_id, "time": pd.Timestamp("2020-02-01"), "feature_name": "FormulaYN", "value": 1.0}
        for patient_id in ["A", "B", "C"]
    ]
    rows.extend(anchor_rows)
    return pd.DataFrame(rows)


def test_aggregate_anchor_validation_correct_case_passes():
    model = TemporalTree()
    X = _aggregation_frame([
        {"patient_id": "A", "time": pd.NaT, "feature_name": "Some Dynamic Anchor", "value": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-15")]},
        {"patient_id": "B", "time": pd.NaT, "feature_name": "Some Dynamic Anchor", "value": [pd.Timestamp("2020-01-01")]},
        {"patient_id": "C", "time": pd.NaT, "feature_name": "Some Dynamic Anchor", "value": [pd.Timestamp("2020-01-01")]},
    ])

    values, _ = model._aggregate_feature_over_time(X, "FormulaYN", ("Some Dynamic Anchor", 2), "count")

    assert values.tolist() == [1, 1, 1]


def test_aggregate_feature_over_time_handles_numpy_array_anchor_value():
    model = TemporalTree()
    list_anchor = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-15")]
    array_anchor = np.array(list_anchor)
    X_list = pd.DataFrame([
        {"patient_id": "A", "time": pd.Timestamp("2020-02-01"), "feature_name": "FormulaYN", "value": 1.0},
        {"patient_id": "A", "time": pd.NaT, "feature_name": "Some Dynamic Anchor", "value": list_anchor},
    ])
    X_array = pd.DataFrame([
        {"patient_id": "A", "time": pd.Timestamp("2020-02-01"), "feature_name": "FormulaYN", "value": 1.0},
        {"patient_id": "A", "time": pd.NaT, "feature_name": "Some Dynamic Anchor", "value": array_anchor},
    ])

    list_values, _ = model._aggregate_feature_over_time(X_list, "FormulaYN", ("Some Dynamic Anchor", 2), "count")
    array_values, _ = model._aggregate_feature_over_time(X_array, "FormulaYN", ("Some Dynamic Anchor", 2), "count")

    assert array_values.tolist() == list_values.tolist()


@pytest.mark.parametrize("anchor_rows", [
    [
        {"patient_id": "A", "time": pd.NaT, "feature_name": "Some Dynamic Anchor", "value": [pd.Timestamp("2020-01-01")]},
        {"patient_id": "B", "time": pd.NaT, "feature_name": "Some Dynamic Anchor", "value": [pd.Timestamp("2020-01-01")]},
    ],
    [
        {"patient_id": "A", "time": pd.NaT, "feature_name": "Some Dynamic Anchor", "value": [pd.Timestamp("2020-01-01")]},
        {"patient_id": "B", "time": pd.NaT, "feature_name": "Some Dynamic Anchor", "value": [pd.Timestamp("2020-01-01")]},
        {"patient_id": "B", "time": pd.NaT, "feature_name": "Some Dynamic Anchor", "value": [pd.Timestamp("2020-01-15")]},
        {"patient_id": "C", "time": pd.NaT, "feature_name": "Some Dynamic Anchor", "value": [pd.Timestamp("2020-01-01")]},
    ],
    [
        {"patient_id": "A", "time": pd.NaT, "feature_name": "Some Dynamic Anchor", "value": [pd.Timestamp("2020-01-01")]},
        {"patient_id": "A", "time": pd.NaT, "feature_name": "Some Dynamic Anchor", "value": [pd.Timestamp("2020-01-15")]},
        {"patient_id": "B", "time": pd.NaT, "feature_name": "Some Dynamic Anchor", "value": [pd.Timestamp("2020-01-01")]},
    ],
])
def test_aggregate_anchor_validation_rejects_missing_duplicate_and_compensating_errors(anchor_rows):
    model = TemporalTree()
    X = _aggregation_frame(anchor_rows)

    with pytest.raises(ValueError, match="exactly one Some Dynamic Anchor"):
        model._aggregate_feature_over_time(X, "FormulaYN", ("Some Dynamic Anchor", 2), "count")


def test_aggregate_anchor_validation_is_branch_local():
    model = TemporalTree()
    X = pd.DataFrame([
        {"patient_id": "A", "time": pd.Timestamp("2020-02-01"), "feature_name": "FormulaYN", "value": 1.0},
        {"patient_id": "A", "time": pd.NaT, "feature_name": "Some Dynamic Anchor", "value": [pd.Timestamp("2020-01-01")]},
    ])

    values, _ = model._aggregate_feature_over_time(X, "FormulaYN", ("Some Dynamic Anchor", 2), "count")

    assert values.tolist() == [1]


def test_create_branch_anchors_rejects_missing_and_duplicate_source_anchor_rows():
    missing_source = pd.DataFrame([
        {"patient_id": "A", "time": pd.Timestamp("2020-02-01"), "feature_name": "FormulaYN", "value": 1.0},
    ])
    duplicate_source = pd.DataFrame([
        {"patient_id": "A", "time": pd.Timestamp("2020-02-01"), "feature_name": "FormulaYN", "value": 1.0},
        {"patient_id": "A", "time": pd.NaT, "feature_name": "Some Dynamic Anchor", "value": [pd.Timestamp("2020-01-01")]},
        {"patient_id": "A", "time": pd.NaT, "feature_name": "Some Dynamic Anchor", "value": [pd.Timestamp("2020-01-15")]},
    ])

    with pytest.raises(ValueError, match="exactly one Some Dynamic Anchor"):
        create_branch_anchors(["A"], missing_source, "out Anchor", "right", "FormulaYN", 0.4, ("Some Dynamic Anchor", 2))
    with pytest.raises(ValueError, match="exactly one Some Dynamic Anchor"):
        create_branch_anchors(["A"], duplicate_source, "out Anchor", "right", "FormulaYN", 0.4, ("Some Dynamic Anchor", 2))
