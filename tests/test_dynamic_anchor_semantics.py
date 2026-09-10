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


@pytest.mark.parametrize("direction", ["left", "right"])
@pytest.mark.parametrize(
    (
        "feature_semantics",
        "aggregation",
        "values",
        "expected_indexes",
    ),
    [
        ("binary_event_indicator", "sum", [1, 0, 1], [0, 2]),
        ("binary_event_indicator", "mean", [1, 0, 1], [0, 2]),
        ("binary_event_indicator", "max", [1, 0, 1], [0, 2]),
        ("state_measurement", "max", [2.0, 8.0, 8.0], [1, 2]),
        ("event_occurrence", "count", [0.0, 1.0, 2.0], [0, 1, 2]),
        ("event_magnitude", "sum", [3.0, 0.0, 4.0], [0, 2]),
        (
            {"type": "state_measurement", "count_anchor": True},
            "count",
            [2.0, 8.0, 4.0],
            [0, 1, 2],
        ),
    ],
)
def test_eligible_semantics_match_training_and_prediction_on_both_branches(
    direction,
    feature_semantics,
    aggregation,
    values,
    expected_indexes,
):
    patient_data, times = _patient_data(values)
    model = TemporalTree(feature_semantics={"signal": feature_semantics})
    node = TemporalSplitNode(
        feature="signal",
        threshold=0.5,
        aggregation=aggregation,
        time_window=(None, None),
    )

    anchor_name, next_anchors, anchor_data, _ = model._prepare_branch_anchor(
        node=node,
        patient_ids=["P1"],
        X_branch=patient_data,
        anchors=["Birth Date Anchor"],
        anchor_registry={},
        direction=direction,
    )
    prediction_data = model._append_prediction_branch_anchor(
        patient_data,
        node,
        direction,
    )
    prediction_anchor = prediction_data.loc[
        prediction_data["feature_name"].eq(anchor_name), "value"
    ].iloc[0]
    expected = [times[index] for index in expected_indexes]

    assert next_anchors == ["Birth Date Anchor", anchor_name]
    assert anchor_data.loc[0, "value"] == expected
    assert prediction_anchor == expected


@pytest.mark.parametrize("direction", ["left", "right"])
@pytest.mark.parametrize(
    ("feature_semantics", "aggregation"),
    [
        ("binary_event_indicator", "count"),
        ("state_measurement", "sum"),
        ("state_measurement", "mean"),
        ("state_measurement", "count"),
        ("event_occurrence", "mean"),
        ("event_magnitude", "max"),
    ],
)
def test_ineligible_semantics_inherit_existing_anchors_unchanged(
    direction,
    feature_semantics,
    aggregation,
):
    patient_data, _ = _patient_data([0.0, 1.0, 2.0])
    model = TemporalTree(feature_semantics={"signal": feature_semantics})
    node = TemporalSplitNode(
        feature="signal",
        threshold=0.5,
        aggregation=aggregation,
        time_window=(None, None),
    )
    existing_anchors = ["Birth Date Anchor"]

    anchor_name, next_anchors, anchor_data, next_registry = (
        model._prepare_branch_anchor(
            node=node,
            patient_ids=["P1"],
            X_branch=patient_data,
            anchors=existing_anchors,
            anchor_registry={},
            direction=direction,
        )
    )
    prediction_data = model._append_prediction_branch_anchor(
        patient_data,
        node,
        direction,
    )

    assert anchor_name is None
    assert next_anchors == existing_anchors
    assert anchor_data.empty
    assert next_registry == {}
    pd.testing.assert_frame_equal(prediction_data, patient_data)


@pytest.mark.parametrize("direction", ["left", "right"])
def test_binary_max_with_only_zero_values_creates_an_empty_anchor(direction):
    patient_data, _ = _patient_data([0, 0])
    model = TemporalTree(
        feature_semantics={"signal": "binary_event_indicator"},
    )
    node = TemporalSplitNode(
        feature="signal",
        threshold=0.5,
        aggregation="max",
        time_window=(None, None),
    )

    anchor_name, next_anchors, anchor_data, _ = model._prepare_branch_anchor(
        node=node,
        patient_ids=["P1"],
        X_branch=patient_data,
        anchors=[],
        anchor_registry={},
        direction=direction,
    )

    assert next_anchors == [anchor_name]
    assert anchor_data.loc[0, "value"] == []


def test_missing_feature_semantics_disables_dynamic_anchor_generation():
    patient_data, _ = _patient_data([0, 1])
    model = TemporalTree()
    node = TemporalSplitNode(
        feature="signal",
        threshold=0.5,
        aggregation="max",
        time_window=(None, None),
    )

    anchor_name, next_anchors, anchor_data, next_registry = (
        model._prepare_branch_anchor(
            node=node,
            patient_ids=["P1"],
            X_branch=patient_data,
            anchors=["Birth Date Anchor"],
            anchor_registry={},
            direction="right",
        )
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
