import numpy as np
import pandas as pd
import pytest

from TAT.anchor_utils import create_branch_anchors
from TAT.temporal_tree import TemporalSplitNode, TemporalTree


def test_create_branch_anchors_handles_scalar_list_empty_and_nan_source_anchors():
    source_anchor = "source Anchor"
    output_anchor = "output Anchor"
    rows = [
        {
            "patient_id": "scalar",
            "time": pd.Timestamp("2020-01-01"),
            "feature_name": source_anchor,
            "value": pd.Timestamp("2020-01-01"),
        },
        {
            "patient_id": "scalar",
            "time": pd.Timestamp("2020-02-01"),
            "feature_name": "FormulaYN",
            "value": 1.0,
        },
        {
            "patient_id": "scalar",
            "time": pd.Timestamp("2020-04-01"),
            "feature_name": "FormulaYN",
            "value": 1.0,
        },
        {
            "patient_id": "one_list",
            "time": pd.NaT,
            "feature_name": source_anchor,
            "value": [pd.Timestamp("2020-03-01")],
        },
        {
            "patient_id": "one_list",
            "time": pd.Timestamp("2020-04-01"),
            "feature_name": "FormulaYN",
            "value": 1.0,
        },
        {
            "patient_id": "one_list",
            "time": pd.Timestamp("2020-06-01"),
            "feature_name": "FormulaYN",
            "value": 1.0,
        },
        {
            "patient_id": "multi_list",
            "time": pd.NaT,
            "feature_name": source_anchor,
            "value": [pd.Timestamp("2016-03-26"), pd.Timestamp("2016-05-28")],
        },
        {
            "patient_id": "multi_list",
            "time": pd.Timestamp("2016-04-15"),
            "feature_name": "FormulaYN",
            "value": 1.0,
        },
        {
            "patient_id": "multi_list",
            "time": pd.Timestamp("2016-06-15"),
            "feature_name": "FormulaYN",
            "value": 1.0,
        },
        {
            "patient_id": "multi_list",
            "time": pd.Timestamp("2016-09-01"),
            "feature_name": "FormulaYN",
            "value": 1.0,
        },
        {
            "patient_id": "empty_list",
            "time": pd.NaT,
            "feature_name": source_anchor,
            "value": [],
        },
        {
            "patient_id": "empty_list",
            "time": pd.Timestamp("2020-02-01"),
            "feature_name": "FormulaYN",
            "value": 1.0,
        },
        {
            "patient_id": "nan_anchor",
            "time": pd.NaT,
            "feature_name": source_anchor,
            "value": np.nan,
        },
        {
            "patient_id": "nan_anchor",
            "time": pd.Timestamp("2020-02-01"),
            "feature_name": "FormulaYN",
            "value": 1.0,
        },
    ]
    X_branch = pd.DataFrame(rows)

    result = create_branch_anchors(
        patient_ids=["scalar", "one_list", "multi_list", "empty_list", "nan_anchor"],
        X_branch=X_branch,
        anchor_name=output_anchor,
        direction="left",
        feature="FormulaYN",
        threshold=0.4,
        time_window=(source_anchor, 2),
    )

    values = result.set_index("patient_id")["value"].to_dict()
    assert values["scalar"] == [pd.Timestamp("2020-02-01")]
    assert values["one_list"] == [pd.Timestamp("2020-04-01")]
    assert values["multi_list"] == [pd.Timestamp("2016-04-15")]
    assert values["empty_list"] == []
    assert values["nan_anchor"] == []


def test_create_branch_anchors_handles_numpy_array_source_anchors_like_lists():
    source_anchor = "source Anchor"
    rows = [
        {
            "patient_id": "list_anchor",
            "time": pd.NaT,
            "feature_name": source_anchor,
            "value": [pd.Timestamp("2016-03-26"), pd.Timestamp("2016-05-28")],
        },
        {
            "patient_id": "list_anchor",
            "time": pd.Timestamp("2016-04-15"),
            "feature_name": "FormulaYN",
            "value": 1.0,
        },
        {
            "patient_id": "list_anchor",
            "time": pd.Timestamp("2016-06-15"),
            "feature_name": "FormulaYN",
            "value": 1.0,
        },
        {
            "patient_id": "array_anchor",
            "time": pd.NaT,
            "feature_name": source_anchor,
            "value": np.array([pd.Timestamp("2016-03-26"), pd.Timestamp("2016-05-28")]),
        },
        {
            "patient_id": "array_anchor",
            "time": pd.Timestamp("2016-04-15"),
            "feature_name": "FormulaYN",
            "value": 1.0,
        },
        {
            "patient_id": "array_anchor",
            "time": pd.Timestamp("2016-06-15"),
            "feature_name": "FormulaYN",
            "value": 1.0,
        },
        {
            "patient_id": "one_array",
            "time": pd.NaT,
            "feature_name": source_anchor,
            "value": np.array([pd.Timestamp("2020-03-01")]),
        },
        {
            "patient_id": "one_array",
            "time": pd.Timestamp("2020-04-01"),
            "feature_name": "FormulaYN",
            "value": 1.0,
        },
        {
            "patient_id": "empty_array",
            "time": pd.NaT,
            "feature_name": source_anchor,
            "value": np.array([]),
        },
        {
            "patient_id": "empty_array",
            "time": pd.Timestamp("2020-04-01"),
            "feature_name": "FormulaYN",
            "value": 1.0,
        },
    ]
    X_branch = pd.DataFrame(rows)

    result = create_branch_anchors(
        patient_ids=["list_anchor", "array_anchor", "one_array", "empty_array"],
        X_branch=X_branch,
        anchor_name="output Anchor",
        direction="left",
        feature="FormulaYN",
        threshold=0.4,
        time_window=(source_anchor, 2),
    )

    values = result.set_index("patient_id")["value"].to_dict()
    assert values["array_anchor"] == values["list_anchor"]
    assert values["one_array"] == [pd.Timestamp("2020-04-01")]
    assert values["empty_array"] == []


def test_create_branch_anchors_full_timeline_left_and_right():
    X_branch = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "time": pd.Timestamp("2020-01-01"),
                "feature_name": "FormulaYN",
                "value": 1.0,
            },
            {
                "patient_id": "P1",
                "time": pd.Timestamp("2020-03-01"),
                "feature_name": "FormulaYN",
                "value": 0.0,
            },
            {
                "patient_id": "P1",
                "time": pd.Timestamp("2020-05-01"),
                "feature_name": "FormulaYN",
                "value": 1.0,
            },
            {
                "patient_id": "P1",
                "time": pd.Timestamp("2020-08-01"),
                "feature_name": "FormulaYN",
                "value": 0.0,
            },
        ]
    )

    left = create_branch_anchors(
        ["P1"], X_branch, "left Anchor", "left", "FormulaYN", 0.4, (None, None)
    )
    right = create_branch_anchors(
        ["P1"], X_branch, "right Anchor", "right", "FormulaYN", 0.4, (None, None)
    )

    assert left.loc[0, "value"] == [pd.Timestamp("2020-01-01")]
    assert right.loc[0, "value"] == [pd.Timestamp("2020-01-01")]


def test_prediction_full_timeline_anchor_matches_training_anchor():
    model = TemporalTree(feature_semantics={"FormulaYN": "binary_event_indicator"},)
    node = TemporalSplitNode(
        source_node_id=0,
        feature="FormulaYN",
        threshold=0.4,
        aggregation="mean",
        time_window=(None, None),
    )
    row = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "time": pd.Timestamp("2020-01-01"),
                "feature_name": "FormulaYN",
                "value": 1.0,
            },
            {
                "patient_id": "P1",
                "time": pd.Timestamp("2020-03-01"),
                "feature_name": "FormulaYN",
                "value": 0.0,
            },
            {
                "patient_id": "P1",
                "time": pd.Timestamp("2020-05-01"),
                "feature_name": "FormulaYN",
                "value": 1.0,
            },
            {
                "patient_id": "P1",
                "time": pd.Timestamp("2020-08-01"),
                "feature_name": "FormulaYN",
                "value": 0.0,
            },
        ]
    )
    anchor_name = model._format_dynamic_anchor_name(node, "left")
    training_anchor = create_branch_anchors(
        ["P1"],
        row,
        anchor_name,
        "left",
        "FormulaYN",
        0.4,
        (None, None),
        aggregation="mean",
        anchor_selection="all_valid",
    )
    prediction_row = model._append_prediction_branch_anchor(row, node, "left")
    prediction_anchor = prediction_row[prediction_row["feature_name"] == anchor_name]

    assert len(prediction_anchor) == 1
    assert (
        training_anchor.loc[0, "feature_name"]
        == prediction_anchor.iloc[0]["feature_name"]
    )
    assert training_anchor.loc[0, "value"] == prediction_anchor.iloc[0]["value"]


def test_dynamic_anchor_name_is_branch_independent():
    model = TemporalTree(feature_semantics={"FormulaYN": "binary_event_indicator"},)
    full_timeline_node = TemporalSplitNode(
        source_node_id=0,
        feature="FormulaYN",
        threshold=0.0,
        aggregation="mean",
        time_window=(None, None),
    )
    normal_node = TemporalSplitNode(
        source_node_id=0,
        feature="FormulaYN",
        threshold=0.5,
        aggregation="mean",
        time_window=("Birth Date Anchor", 3),
    )

    assert (
        model._format_dynamic_anchor_name(full_timeline_node, "left")
        == "Node 0 | MEAN(FormulaYN) Anchor"
    )
    assert (
        model._format_dynamic_anchor_name(normal_node, "right")
        == "Node 0 | MEAN(FormulaYN) Anchor"
    )


def test_dynamic_anchor_key_uses_source_node_identity():
    model = TemporalTree(feature_semantics={"FormulaYN": "binary_event_indicator"},)
    node = TemporalSplitNode(
        source_node_id=0,
        feature="FormulaYN",
        threshold=0.0,
        aggregation="mean",
        time_window=(None, None),
    )

    assert model._dynamic_anchor_key(node, "left") == 0
    assert model._dynamic_anchor_key(node, "right") == 0


def test_repeated_full_timeline_anchor_reuses_same_semantic_key():
    model = TemporalTree(feature_semantics={"FormulaYN": "binary_event_indicator"},)
    node = TemporalSplitNode(
        source_node_id=0,
        feature="FormulaYN",
        threshold=0.0,
        aggregation="mean",
        time_window=(None, None),
    )
    X_branch = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "time": pd.Timestamp("2020-01-01"),
                "feature_name": "FormulaYN",
                "value": 1.0,
            },
        ]
    )
    anchor_registry = {}

    anchor_name, anchors, anchor_df, anchor_registry = model._prepare_branch_anchor(
        node=node,
        patient_ids=["P1"],
        X_branch=X_branch,
        anchors=[],
        anchor_registry=anchor_registry,
        direction="left",
    )
    (
        reused_name,
        reused_anchors,
        reused_df,
        reused_registry,
    ) = model._prepare_branch_anchor(
        node=node,
        patient_ids=["P1"],
        X_branch=X_branch,
        anchors=anchors,
        anchor_registry=anchor_registry,
        direction="left",
    )

    assert anchor_name == "Node 0 | MEAN(FormulaYN) Anchor"
    assert reused_name == anchor_name
    assert reused_anchors == anchors
    assert not anchor_df.empty
    assert reused_df.empty
    assert reused_registry == anchor_registry


def test_different_source_nodes_have_distinct_names_despite_rounded_thresholds():
    model = TemporalTree(feature_semantics={"FormulaYN": "binary_event_indicator"},)
    first_node = TemporalSplitNode(
        source_node_id=0,
        feature="FormulaYN",
        threshold=0.0001,
        aggregation="mean",
        time_window=(None, None),
    )
    second_node = TemporalSplitNode(
        source_node_id=1,
        feature="FormulaYN",
        threshold=0.0002,
        aggregation="mean",
        time_window=(None, None),
    )

    assert model._format_dynamic_anchor_name(first_node, "left") != (
        model._format_dynamic_anchor_name(second_node, "left")
    )
    assert model._dynamic_anchor_key(first_node, "left") != (
        model._dynamic_anchor_key(second_node, "left")
    )


def test_dynamic_anchor_identity_distinguishes_aggregations():
    model = TemporalTree(feature_semantics={"FormulaYN": "binary_event_indicator"},)
    mean_node = TemporalSplitNode(
        source_node_id=0,
        feature="FormulaYN",
        threshold=0.5,
        aggregation="mean",
        time_window=("Birth Date Anchor", 3),
    )
    max_node = TemporalSplitNode(
        source_node_id=1,
        feature="FormulaYN",
        threshold=0.5,
        aggregation="max",
        time_window=("Birth Date Anchor", 3),
    )

    assert model._format_dynamic_anchor_name(mean_node, "left") == (
        "Node 0 | MEAN(FormulaYN) Anchor"
    )
    assert model._format_dynamic_anchor_name(max_node, "left") == (
        "Node 1 | MAX(FormulaYN) Anchor"
    )
    assert model._dynamic_anchor_key(mean_node, "left") != (
        model._dynamic_anchor_key(max_node, "left")
    )


def test_prediction_numpy_array_source_anchor_matches_training_anchor():
    model = TemporalTree(feature_semantics={"FormulaYN": "binary_event_indicator"},)
    source_anchor = "source Anchor"
    node = TemporalSplitNode(
        source_node_id=0,
        feature="FormulaYN",
        threshold=0.4,
        aggregation="mean",
        time_window=(source_anchor, 2),
    )
    row = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "time": pd.NaT,
                "feature_name": source_anchor,
                "value": np.array(
                    [pd.Timestamp("2016-03-26"), pd.Timestamp("2016-05-28")]
                ),
            },
            {
                "patient_id": "P1",
                "time": pd.Timestamp("2016-04-15"),
                "feature_name": "FormulaYN",
                "value": 1.0,
            },
            {
                "patient_id": "P1",
                "time": pd.Timestamp("2016-06-15"),
                "feature_name": "FormulaYN",
                "value": 1.0,
            },
            {
                "patient_id": "P1",
                "time": pd.Timestamp("2016-09-01"),
                "feature_name": "FormulaYN",
                "value": 1.0,
            },
        ]
    )
    anchor_name = model._format_dynamic_anchor_name(node, "left")
    training_anchor = create_branch_anchors(
        ["P1"],
        row,
        anchor_name,
        "left",
        "FormulaYN",
        0.4,
        (source_anchor, 2),
        aggregation="mean",
        anchor_selection="all_valid",
    )
    prediction_row = model._append_prediction_branch_anchor(row, node, "left")
    prediction_anchor = prediction_row[prediction_row["feature_name"] == anchor_name]

    assert len(prediction_anchor) == 1
    assert training_anchor.loc[0, "value"] == prediction_anchor.iloc[0]["value"]


def test_create_branch_anchors_non_none_window_unchanged():
    X_branch = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "time": pd.Timestamp("2020-01-01"),
                "feature_name": "Birth Date Anchor",
                "value": pd.Timestamp("2020-01-01"),
            },
            {
                "patient_id": "P1",
                "time": pd.Timestamp("2020-02-01"),
                "feature_name": "FormulaYN",
                "value": 1.0,
            },
            {
                "patient_id": "P1",
                "time": pd.Timestamp("2020-08-01"),
                "feature_name": "FormulaYN",
                "value": 1.0,
            },
        ]
    )

    result = create_branch_anchors(
        ["P1"],
        X_branch,
        "left Anchor",
        "left",
        "FormulaYN",
        0.4,
        ("Birth Date Anchor", 3),
    )

    assert result.loc[0, "value"] == [pd.Timestamp("2020-02-01")]


def test_create_branch_anchors_empty_feature_rows_full_timeline():
    X_branch = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "time": pd.Timestamp("2020-01-01"),
                "feature_name": "OtherFeature",
                "value": 1.0,
            },
        ]
    )

    result = create_branch_anchors(
        ["P1"], X_branch, "left Anchor", "left", "FormulaYN", 0.4, (None, None)
    )

    assert result.loc[0, "value"] == []


@pytest.mark.parametrize(
    ("aggregation", "direction", "threshold", "values", "expected_indexes"),
    [
        ("sum", "right", 4.0, [1, 1, 0, 1, 1, 1], [0]),
        ("sum", "left", 4.0, [1, 0, 1, 0, 1], [0]),
        ("mean", "right", 0.5, [1, 0, 1], [0]),
        ("mean", "left", 0.5, [0, 0, 1], [0]),
        ("count", "right", 4.0, [0, 1, 0, 1, 1], [0]),
        ("count", "left", 4.0, [0, 1, 0], [0]),
        ("max", "right", 0.8, [0.2, 0.9, 0.4, 0.9], [1]),
        ("max", "left", 0.8, [0.2, 0.7, 0.4], [1]),
    ],
)
def test_aggregation_aware_anchors_match_training_and_prediction(
    aggregation, direction, threshold, values, expected_indexes,
):
    times = list(pd.date_range("2020-01-01", periods=len(values), freq="D"))
    rows = [
        {
            "patient_id": "P1",
            "time": times[index],
            "feature_name": "VomitingYN",
            "value": values[index],
        }
        for index in reversed(range(len(values)))
    ]
    patient_data = pd.DataFrame(rows)
    node = TemporalSplitNode(
        source_node_id=0,
        feature="VomitingYN",
        threshold=threshold,
        aggregation=aggregation,
        time_window=(None, None),
    )
    if aggregation == "count":
        feature_type = "event_occurrence"
    elif aggregation == "max":
        feature_type = "state_measurement"
    else:
        feature_type = "binary_event_indicator"
    model = TemporalTree(feature_semantics={"VomitingYN": feature_type},)
    aggregate = model._aggregate_feature_over_time(
        patient_data, node.feature, node.time_window, node.aggregation,
    )[0][0]
    assert model._get_prediction_direction(aggregate, node) == direction

    anchor_name = model._format_dynamic_anchor_name(node, direction)
    training_anchor = create_branch_anchors(
        ["P1"],
        patient_data,
        anchor_name,
        direction,
        node.feature,
        node.threshold,
        node.time_window,
        aggregation=node.aggregation,
        anchor_selection=model._dynamic_anchor_selection(node),
    )
    prediction_data = model._append_prediction_branch_anchor(
        patient_data, node, direction,
    )
    prediction_anchor = prediction_data.loc[
        prediction_data["feature_name"].eq(anchor_name), "value"
    ].iloc[0]
    expected = [times[index] for index in expected_indexes]

    assert training_anchor.loc[0, "value"] == expected
    assert prediction_anchor == expected


def test_max_anchor_selects_one_timestamp_for_repeated_ties():
    repeated_time = pd.Timestamp("2020-01-02")
    patient_data = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "time": repeated_time,
                "feature_name": "signal",
                "value": 0.9,
            },
            {
                "patient_id": "P1",
                "time": pd.Timestamp("2020-01-01"),
                "feature_name": "signal",
                "value": 0.2,
            },
            {
                "patient_id": "P1",
                "time": repeated_time,
                "feature_name": "signal",
                "value": 0.9,
            },
        ]
    )

    anchor = create_branch_anchors(
        ["P1"],
        patient_data,
        "MAX(signal) <= 0.800 | Full Timeline Anchor",
        "left",
        "signal",
        0.8,
        (None, None),
        aggregation="max",
    )

    assert anchor.loc[0, "value"] == [repeated_time]


@pytest.mark.parametrize("aggregation", ["sum", "mean", "max", "count"])
@pytest.mark.parametrize("direction", ["right", "left"])
def test_aggregation_aware_anchors_are_empty_without_valid_observations(
    aggregation, direction,
):
    patient_data = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "time": pd.Timestamp("2020-01-01"),
                "feature_name": "VomitingYN",
                "value": np.nan,
            },
            {
                "patient_id": "P1",
                "time": pd.Timestamp("2020-01-02"),
                "feature_name": "VomitingYN",
                "value": "not numeric",
            },
        ]
    )
    node = TemporalSplitNode(
        source_node_id=0,
        feature="VomitingYN",
        threshold=0.5,
        aggregation=aggregation,
        time_window=(None, None),
    )
    feature_type = (
        "event_occurrence" if aggregation == "count" else "binary_event_indicator"
    )
    model = TemporalTree(feature_semantics={"VomitingYN": feature_type},)
    anchor_name = model._format_dynamic_anchor_name(node, direction)
    training_anchor = create_branch_anchors(
        ["P1"],
        patient_data,
        anchor_name,
        direction,
        node.feature,
        node.threshold,
        node.time_window,
        aggregation=node.aggregation,
        anchor_selection=model._dynamic_anchor_selection(node),
    )
    prediction_data = model._append_prediction_branch_anchor(
        patient_data, node, direction,
    )
    prediction_anchor = prediction_data.loc[
        prediction_data["feature_name"].eq(anchor_name), "value"
    ].iloc[0]

    assert training_anchor.loc[0, "value"] == []
    assert prediction_anchor == []


def _aggregation_frame(anchor_rows):
    rows = [
        {
            "patient_id": patient_id,
            "time": pd.Timestamp("2020-02-01"),
            "feature_name": "FormulaYN",
            "value": 1.0,
        }
        for patient_id in ["A", "B", "C"]
    ]
    rows.extend(anchor_rows)
    return pd.DataFrame(rows)


def test_aggregate_anchor_validation_correct_case_passes():
    model = TemporalTree()
    X = _aggregation_frame(
        [
            {
                "patient_id": "A",
                "time": pd.NaT,
                "feature_name": "Some Dynamic Anchor",
                "value": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-15")],
            },
            {
                "patient_id": "B",
                "time": pd.NaT,
                "feature_name": "Some Dynamic Anchor",
                "value": [pd.Timestamp("2020-01-01")],
            },
            {
                "patient_id": "C",
                "time": pd.NaT,
                "feature_name": "Some Dynamic Anchor",
                "value": [pd.Timestamp("2020-01-01")],
            },
        ]
    )

    values, _ = model._aggregate_feature_over_time(
        X, "FormulaYN", ("Some Dynamic Anchor", 2), "count"
    )

    assert values.tolist() == [1, 1, 1]


def test_aggregate_feature_over_time_handles_numpy_array_anchor_value():
    model = TemporalTree()
    list_anchor = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-15")]
    array_anchor = np.array(list_anchor)
    X_list = pd.DataFrame(
        [
            {
                "patient_id": "A",
                "time": pd.Timestamp("2020-02-01"),
                "feature_name": "FormulaYN",
                "value": 1.0,
            },
            {
                "patient_id": "A",
                "time": pd.NaT,
                "feature_name": "Some Dynamic Anchor",
                "value": list_anchor,
            },
        ]
    )
    X_array = pd.DataFrame(
        [
            {
                "patient_id": "A",
                "time": pd.Timestamp("2020-02-01"),
                "feature_name": "FormulaYN",
                "value": 1.0,
            },
            {
                "patient_id": "A",
                "time": pd.NaT,
                "feature_name": "Some Dynamic Anchor",
                "value": array_anchor,
            },
        ]
    )

    list_values, _ = model._aggregate_feature_over_time(
        X_list, "FormulaYN", ("Some Dynamic Anchor", 2), "count"
    )
    array_values, _ = model._aggregate_feature_over_time(
        X_array, "FormulaYN", ("Some Dynamic Anchor", 2), "count"
    )

    assert array_values.tolist() == list_values.tolist()


@pytest.mark.parametrize(
    "anchor_rows",
    [
        [
            {
                "patient_id": "A",
                "time": pd.NaT,
                "feature_name": "Some Dynamic Anchor",
                "value": [pd.Timestamp("2020-01-01")],
            },
            {
                "patient_id": "B",
                "time": pd.NaT,
                "feature_name": "Some Dynamic Anchor",
                "value": [pd.Timestamp("2020-01-01")],
            },
        ],
        [
            {
                "patient_id": "A",
                "time": pd.NaT,
                "feature_name": "Some Dynamic Anchor",
                "value": [pd.Timestamp("2020-01-01")],
            },
            {
                "patient_id": "B",
                "time": pd.NaT,
                "feature_name": "Some Dynamic Anchor",
                "value": [pd.Timestamp("2020-01-01")],
            },
            {
                "patient_id": "B",
                "time": pd.NaT,
                "feature_name": "Some Dynamic Anchor",
                "value": [pd.Timestamp("2020-01-15")],
            },
            {
                "patient_id": "C",
                "time": pd.NaT,
                "feature_name": "Some Dynamic Anchor",
                "value": [pd.Timestamp("2020-01-01")],
            },
        ],
        [
            {
                "patient_id": "A",
                "time": pd.NaT,
                "feature_name": "Some Dynamic Anchor",
                "value": [pd.Timestamp("2020-01-01")],
            },
            {
                "patient_id": "A",
                "time": pd.NaT,
                "feature_name": "Some Dynamic Anchor",
                "value": [pd.Timestamp("2020-01-15")],
            },
            {
                "patient_id": "B",
                "time": pd.NaT,
                "feature_name": "Some Dynamic Anchor",
                "value": [pd.Timestamp("2020-01-01")],
            },
        ],
    ],
)
def test_aggregate_anchor_validation_rejects_missing_duplicate_and_compensating_errors(
    anchor_rows,
):
    model = TemporalTree()
    X = _aggregation_frame(anchor_rows)

    with pytest.raises(ValueError, match="exactly one Some Dynamic Anchor"):
        model._aggregate_feature_over_time(
            X, "FormulaYN", ("Some Dynamic Anchor", 2), "count"
        )


def test_aggregate_anchor_validation_is_branch_local():
    model = TemporalTree()
    X = pd.DataFrame(
        [
            {
                "patient_id": "A",
                "time": pd.Timestamp("2020-02-01"),
                "feature_name": "FormulaYN",
                "value": 1.0,
            },
            {
                "patient_id": "A",
                "time": pd.NaT,
                "feature_name": "Some Dynamic Anchor",
                "value": [pd.Timestamp("2020-01-01")],
            },
        ]
    )

    values, _ = model._aggregate_feature_over_time(
        X, "FormulaYN", ("Some Dynamic Anchor", 2), "count"
    )

    assert values.tolist() == [1]


def test_create_branch_anchors_rejects_missing_and_duplicate_source_anchor_rows():
    missing_source = pd.DataFrame(
        [
            {
                "patient_id": "A",
                "time": pd.Timestamp("2020-02-01"),
                "feature_name": "FormulaYN",
                "value": 1.0,
            },
        ]
    )
    duplicate_source = pd.DataFrame(
        [
            {
                "patient_id": "A",
                "time": pd.Timestamp("2020-02-01"),
                "feature_name": "FormulaYN",
                "value": 1.0,
            },
            {
                "patient_id": "A",
                "time": pd.NaT,
                "feature_name": "Some Dynamic Anchor",
                "value": [pd.Timestamp("2020-01-01")],
            },
            {
                "patient_id": "A",
                "time": pd.NaT,
                "feature_name": "Some Dynamic Anchor",
                "value": [pd.Timestamp("2020-01-15")],
            },
        ]
    )

    with pytest.raises(ValueError, match="exactly one Some Dynamic Anchor"):
        create_branch_anchors(
            ["A"],
            missing_source,
            "out Anchor",
            "left",
            "FormulaYN",
            0.4,
            ("Some Dynamic Anchor", 2),
        )
    with pytest.raises(ValueError, match="exactly one Some Dynamic Anchor"):
        create_branch_anchors(
            ["A"],
            duplicate_source,
            "out Anchor",
            "left",
            "FormulaYN",
            0.4,
            ("Some Dynamic Anchor", 2),
        )
