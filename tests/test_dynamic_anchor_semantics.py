from types import MethodType

import numpy as np
import pandas as pd
import pytest

from settree.temporal_tree import TemporalSplitNode, TemporalTree


def _patient_data(values):
    times = list(pd.date_range("2020-01-01", periods=len(values), freq="D"))
    rows = [
        {
            "patient_id": "P1",
            "time": times[index],
            "feature_name": "signal",
            "value": values[index],
        }
        for index in reversed(range(len(values)))
    ]
    return pd.DataFrame(rows), times


def _prepare(model, node, data, direction="left", anchors=None, registry=None):
    return model._prepare_branch_anchor(
        node,
        data["patient_id"].unique(),
        data,
        anchors or [],
        registry or {},
        direction,
    )


@pytest.mark.parametrize("direction", ["right", "left"])
@pytest.mark.parametrize(
    "feature_type",
    [
        "binary_event_indicator",
        "state_measurement",
        "event_magnitude",
        "event_occurrence",
        {"type": "state_measurement", "count_anchor": False},
        {"type": "state_measurement", "count_anchor": True},
    ],
)
@pytest.mark.parametrize("aggregation", ["count", "sum", "max", "mean"])
def test_single_timestamp_rules_match_training_and_prediction(
    direction, feature_type, aggregation,
):
    binary = feature_type == "binary_event_indicator"
    data, times = _patient_data([0, 1, 1] if binary else [0.0, 8.0, 8.0])
    model = TemporalTree(feature_semantics={"signal": feature_type})
    node = TemporalSplitNode(
        source_node_id=0,
        feature="signal",
        threshold=10,
        aggregation=aggregation,
        time_window=(None, None),
    )
    name, anchors, rows, registry = _prepare(model, node, data, direction)
    prediction = model._append_prediction_branch_anchor(data, node, direction)
    expected_index = (
        1 if aggregation == "max" or (binary and aggregation == "sum") else 0
    )
    expected = [times[expected_index]]
    assert rows.iloc[0]["value"] == expected
    assert prediction.loc[prediction.feature_name.eq(name), "value"].iloc[0] == expected
    assert anchors == [name]
    assert registry[model._dynamic_anchor_key(node, direction)] == name


@pytest.mark.parametrize(
    ("aggregation", "expected_index"),
    [("count", 0), ("sum", None), ("max", 0), ("mean", 0),],
)
def test_all_zero_binary_observations(aggregation, expected_index):
    data, times = _patient_data([0, 0])
    model = TemporalTree(feature_semantics={"signal": "binary_event_indicator"})
    node = TemporalSplitNode(
        source_node_id=0,
        feature="signal",
        threshold=10,
        aggregation=aggregation,
        time_window=(None, None),
    )
    aggregate = model._aggregate_feature_over_time(
        data, "signal", node.time_window, aggregation
    )[0][0]
    assert model._get_prediction_direction(aggregate, node) == "left"
    name, _, rows, _ = _prepare(model, node, data)
    expected = [] if expected_index is None else [times[expected_index]]
    assert rows.iloc[0]["value"] == expected
    prediction = model._append_prediction_branch_anchor(data, node, "left")
    assert prediction.loc[prediction.feature_name.eq(name), "value"].iloc[0] == expected


@pytest.mark.parametrize("offset", [-0.1, 0.1, -1, 1, None])
@pytest.mark.parametrize("aggregation", ["count", "sum", "max", "mean"])
def test_only_contributing_window_observations_select_timestamp(offset, aggregation):
    data, times = _patient_data([99, 0, 3, 3, 99])
    data["time"] = data["time"].map(
        {
            times[0]: pd.Timestamp("2019-01-01"),
            times[1]: pd.Timestamp("2020-01-01"),
            times[2]: pd.Timestamp("2020-01-02"),
            times[3]: pd.Timestamp("2020-01-03"),
            times[4]: pd.Timestamp("2021-01-01"),
        }
    )
    reference = pd.Timestamp("2020-01-03" if offset and offset < 0 else "2020-01-01")
    # Multiple reference timestamps, including an overlapping window.
    data.loc[len(data)] = [
        "P1",
        pd.NaT,
        "source Anchor",
        [reference, reference],
    ]
    model = TemporalTree(feature_semantics={"signal": "state_measurement"})
    node = TemporalSplitNode(
        source_node_id=0,
        feature="signal",
        threshold=10,
        aggregation=aggregation,
        time_window=("source Anchor", offset),
    )
    name, _, rows, _ = _prepare(model, node, data)
    expected = pd.Timestamp(
        "2019-01-01"
        if offset is None
        else "2020-01-02"
        if aggregation == "max"
        else "2020-01-01"
    )
    assert rows.iloc[0]["value"] == [expected]
    prediction = model._append_prediction_branch_anchor(data, node, "left")
    assert prediction.loc[prediction.feature_name.eq(name), "value"].iloc[0] == [
        expected
    ]


@pytest.mark.parametrize("values", [[np.nan, "invalid"], [0, 0]])
def test_empty_anchor_preserves_routing_and_cannot_define_a_window(values):
    data, _ = _patient_data(values)
    model = TemporalTree(
        feature_semantics={"signal": "binary_event_indicator"}, verbose=0
    )
    node = TemporalSplitNode(
        source_node_id=0,
        feature="signal",
        threshold=10,
        aggregation="sum",
        time_window=(None, None),
        missing_direction="left",
    )
    aggregate = model._aggregate_feature_over_time(
        data, "signal", node.time_window, "sum"
    )[0][0]
    assert model._get_prediction_direction(aggregate, node) == "left"
    name, _, rows, _ = _prepare(model, node, data)
    assert rows.iloc[0]["value"] == []
    prediction = model._append_prediction_branch_anchor(data, node, "left")
    for offset in [-1, 1]:
        result = model._aggregate_feature_over_time(
            prediction, "signal", (name, offset), "count"
        )[0]
        assert np.isnan(result).all()


def test_missing_feature_and_missing_timestamp_have_no_anchor_date():
    data, _ = _patient_data([1])
    data["time"] = pd.NaT
    model = TemporalTree(feature_semantics={"signal": "binary_event_indicator"})
    node = TemporalSplitNode(
        source_node_id=0,
        feature="signal",
        threshold=0,
        aggregation="sum",
        time_window=(None, None),
    )
    for frame in [data, data.assign(feature_name="other")]:
        name, _, rows, _ = _prepare(model, node, frame)
        assert rows.iloc[0]["value"] == []
        prediction = model._append_prediction_branch_anchor(frame, node, "left")
        assert prediction.loc[prediction.feature_name.eq(name), "value"].iloc[0] == []


@pytest.mark.parametrize("direction", ["right", "left"])
def test_existing_path_anchors_are_preserved(direction):
    data, times = _patient_data([0, 1])
    data.loc[len(data)] = ["P1", pd.NaT, "prior Anchor", [times[0]]]
    model = TemporalTree(feature_semantics={"signal": "binary_event_indicator"})
    node = TemporalSplitNode(
        source_node_id=0,
        feature="signal",
        threshold=0.5,
        aggregation="max",
        time_window=("prior Anchor", 1),
    )
    existing = ["prior Anchor"]
    registry = {("prior",): "prior Anchor"}
    name, anchors, _, next_registry = _prepare(
        model, node, data, direction, existing, registry
    )
    prediction = model._append_prediction_branch_anchor(data, node, direction)
    pd.testing.assert_frame_equal(prediction.iloc[: len(data)], data)
    assert existing == ["prior Anchor"]
    assert registry == {("prior",): "prior Anchor"}
    assert anchors == existing + [name]
    assert next_registry[("prior",)] == "prior Anchor"


@pytest.mark.parametrize("enabled", [False, True])
def test_build_and_prediction_keep_patient_and_path_specific_anchors(enabled):
    # Force a small known topology to inspect actual BFS propagation and all
    # prediction entry points independently of candidate selection.
    data, times = _patient_data([0, 1])
    frames = []
    for patient, values in [
        ("A", [1, 1]),
        ("B", [0, 1]),
        ("C", [1, 0]),
        ("D", [np.nan, np.nan]),
    ]:
        frame, _ = _patient_data(values)
        frame["patient_id"] = patient
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    model = TemporalTree(
        feature_semantics={"signal": "binary_event_indicator"},
        use_dynamic_anchors=enabled,
        is_classification=False,
        criterion="mse",
        verbose=0,
    )
    root = TemporalSplitNode(
        source_node_id=0,
        feature="signal",
        threshold=1,
        aggregation="sum",
        time_window=(None, None),
        missing_direction="left",
    )
    left_name = model._format_dynamic_anchor_name(root, "left")
    seen = []

    def get_node(self, X, y, depth, anchors_list):
        if depth == 0:
            return root
        seen.append((X.copy(), list(anchors_list)))
        if depth == 1 and "B" in X.patient_id.values:
            return TemporalSplitNode(
                source_node_id=0,
                feature="signal",
                threshold=1,
                aggregation="count",
                time_window=(left_name, -0.1) if enabled else (None, None),
            )
        return self._create_leaf(float(y.mean()), len(y), depth)

    model._get_node = MethodType(get_node, model)
    model.fit(data, pd.Series([0.0, 1.0, 2.0, 3.0]))
    left_data, left_anchors = seen[0]
    right_data, right_anchors = seen[1]
    if enabled:
        assert left_name in right_anchors
        assert right_data.loc[right_data.feature_name.eq(left_name), "value"].iloc[
            0
        ] == [times[0]]
        assert left_name in left_anchors
        by_patient = (
            left_data[left_data.feature_name.eq(left_name)]
            .set_index("patient_id")
            .value
        )
        assert by_patient.to_dict() == {
            "B": [times[1]],
            "C": [times[0]],
            "D": [],
        }
        # The parent's LEFT anchors survive the next split on either branch.
        for child_data, child_anchors in seen[2:]:
            assert left_name in child_anchors
            inherited = (
                child_data[child_data.feature_name.eq(left_name)]
                .set_index("patient_id")
                .value
            )
            assert inherited.to_dict() == by_patient.loc[inherited.index].to_dict()
    else:
        assert all(anchors == model.anchors_list for _, anchors in seen)
        assert all(
            not X.feature_name.str.contains("SUM(", regex=False).any() for X, _ in seen
        )
    predictions, leaf_ids = model.predict_and_apply(data)
    np.testing.assert_array_equal(predictions, model.predict(data))
    np.testing.assert_array_equal(leaf_ids, model.apply(data))
    # Compare prediction routing to the patients actually supplied to each leaf.
    for patient, prediction in zip(data.patient_id.unique(), predictions):
        leaf_frame = (
            next(X for X, _ in seen[2:] if patient in X.patient_id.values)
            if patient != "A"
            else right_data
        )
        expected = {"A": 0.0, "B": 1.0, "C": 2.0, "D": 3.0}
        assert prediction == np.mean(
            [expected[pid] for pid in leaf_frame.patient_id.unique()]
        )


def test_missing_feature_semantics_disables_dynamic_anchor_generation():
    patient_data, _ = _patient_data([0, 1])
    model = TemporalTree()
    node = TemporalSplitNode(
        source_node_id=0,
        feature="signal",
        threshold=0.5,
        aggregation="max",
        time_window=(None, None),
    )

    (
        anchor_name,
        next_anchors,
        anchor_data,
        next_registry,
    ) = model._prepare_branch_anchor(
        node=node,
        patient_ids=["P1"],
        X_branch=patient_data,
        anchors=["Birth Date Anchor"],
        anchor_registry={},
        direction="left",
    )

    assert anchor_name is None
    assert next_anchors == ["Birth Date Anchor"]
    assert anchor_data.empty
    assert next_registry == {}


@pytest.mark.parametrize(
    "feature_semantics",
    [
        [],
        {"signal": "unknown"},
        {"signal": {"type": "state_measurement", "count_anchor": "yes"}},
        {"signal": {"type": "event_occurrence", "count_anchor": True}},
    ],
)
def test_invalid_feature_semantics_fail_during_fit(feature_semantics):
    patient_data, _ = _patient_data([0, 1])
    labels = pd.Series([1.0])
    model = TemporalTree(
        feature_semantics=feature_semantics,
        is_classification=False,
        criterion="mse",
        verbose=0,
    )

    with pytest.raises(ValueError):
        model.fit(patient_data, labels)


@pytest.mark.parametrize("direction", ["right", "left"])
def test_repeated_split_reuses_anchor_or_placeholder(direction):
    data, _ = _patient_data([0, 1])
    model = TemporalTree(feature_semantics={"signal": "binary_event_indicator"})
    node = TemporalSplitNode(
        source_node_id=0,
        feature="signal",
        threshold=0.5,
        aggregation="max",
        time_window=(None, None),
    )
    name, anchors, rows, registry = _prepare(model, node, data, direction)
    combined = pd.concat([data, rows], ignore_index=True)
    reused_name, reused_anchors, reused_rows, _ = _prepare(
        model, node, combined, direction, anchors, registry,
    )
    assert reused_name == name
    assert reused_anchors == anchors
    assert reused_rows.empty
    prediction = model._append_prediction_branch_anchor(combined, node, direction)
    pd.testing.assert_frame_equal(prediction, combined)


@pytest.mark.parametrize("aggregation", ["count", "sum", "max", "mean"])
def test_invalid_values_do_not_supply_anchor_timestamps(aggregation):
    data, times = _patient_data([np.nan, "invalid", -3, -2])
    model = TemporalTree(feature_semantics={"signal": "state_measurement"})
    node = TemporalSplitNode(
        source_node_id=0,
        feature="signal",
        threshold=10,
        aggregation=aggregation,
        time_window=(None, None),
    )
    name, _, rows, _ = _prepare(model, node, data)
    expected = [times[3] if aggregation == "max" else times[2]]
    assert rows.iloc[0]["value"] == expected
    prediction = model._append_prediction_branch_anchor(data, node, "left")
    assert prediction.loc[prediction.feature_name.eq(name), "value"].iloc[0] == expected


@pytest.mark.parametrize("offset", [-0.1, 0.1])
def test_direct_timestamp_window_matches_split_observations(offset):
    data, times = _patient_data([1, 2, 3])
    model = TemporalTree(feature_semantics={"signal": "state_measurement"})
    node = TemporalSplitNode(
        source_node_id=0,
        feature="signal",
        threshold=0,
        aggregation="max",
        time_window=(times[1], offset),
    )
    name, _, rows, _ = _prepare(model, node, data)
    expected = [times[1] if offset < 0 else times[2]]
    assert rows.iloc[0]["value"] == expected
    prediction = model._append_prediction_branch_anchor(data, node, "left")
    assert prediction.loc[prediction.feature_name.eq(name), "value"].iloc[0] == expected


@pytest.mark.parametrize("offset", [None, 1])
def test_single_symptom_requires_a_valid_split_before_creating_anchors(offset):
    """LEFT can anchor observed symptoms, but cannot split a constant value."""
    birth = pd.Timestamp("2020-01-01")
    rows = []
    for patient, days in enumerate([5, 10, 60, 65]):
        rows.extend(
            [
                {
                    "patient_id": patient,
                    "time": birth,
                    "feature_name": "Birth Date Anchor",
                    "value": birth,
                },
                {
                    "patient_id": patient,
                    "time": birth + pd.Timedelta(days=days),
                    "feature_name": "symptom",
                    "value": 1,
                },
            ]
        )
    data = pd.DataFrame(rows)
    model = TemporalTree(
        max_depth=1,
        min_samples_split=2,
        min_samples_leaf=1,
        anchors_list=["Birth Date Anchor"],
        anchor_offset_list=[offset],
        aggregation_funcs={"sum": np.sum},
        feature_semantics={"symptom": "binary_event_indicator"},
        is_classification=True,
        criterion="gini",
        candidate_sampling_rate=1.0,
        min_impurity_decrease=0.0,
        verbose=0,
    ).fit(data, pd.Series([0, 1, 0, 1]))
    if offset is None:
        assert model.tree.is_leaf()
        return

    root = model.tree
    assert not root.is_leaf()
    assert root.feature == "symptom"
    assert root.threshold == 1
    assert root.missing_direction == "right"
    assert root.left.num_patients == root.right.num_patients == 2
    for patient, patient_data in data.groupby("patient_id", sort=False):
        aggregate = model._aggregate_feature_over_time(
            patient_data, root.feature, root.time_window, root.aggregation,
        )[0][0]
        direction = model._get_prediction_direction(aggregate, root)
        assert direction == ("left" if patient < 2 else "right")
        name, anchors, created, _ = _prepare(model, root, patient_data, direction,)
        prediction_data = model._append_prediction_branch_anchor(
            patient_data, root, direction,
        )
        expected = (
            patient_data.loc[patient_data.feature_name.eq("symptom"), "time"].tolist()
            if patient < 2
            else []
        )
        assert created.iloc[0]["value"] == expected
        assert anchors == [name]
        assert (
            prediction_data.loc[prediction_data.feature_name.eq(name), "value"].iloc[0]
            == expected
        )


def test_identical_splits_at_distinct_nodes_do_not_reuse_anchors():
    data, times = _patient_data([0, 1])
    model = TemporalTree(feature_semantics={"signal": "binary_event_indicator"})
    first = TemporalSplitNode(
        source_node_id=0,
        feature="signal",
        threshold=0.5,
        aggregation="sum",
        time_window=(None, None),
    )
    second = TemporalSplitNode(
        source_node_id=1,
        feature="signal",
        threshold=0.5,
        aggregation="sum",
        time_window=(None, None),
    )
    name, anchors, rows, registry = _prepare(model, first, data, "right")
    combined = pd.concat([data, rows], ignore_index=True)
    next_name, next_anchors, next_rows, _ = _prepare(
        model, second, combined, "left", anchors, registry,
    )
    assert name != next_name
    assert next_anchors == [name, next_name]
    assert rows.iloc[0]["value"] == next_rows.iloc[0]["value"] == [times[1]]
    predicted = model._append_prediction_branch_anchor(combined, second, "left")
    assert predicted.loc[predicted.feature_name.eq(name), "value"].iloc[0] == [times[1]]
    assert predicted.loc[predicted.feature_name.eq(next_name), "value"].iloc[0] == [
        times[1]
    ]
    assert model._format_dynamic_anchor_name(first, "left") == name
