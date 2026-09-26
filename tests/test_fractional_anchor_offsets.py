from types import MethodType

import numpy as np
import pandas as pd
import pytest

from settree.anchor_utils import _offset_to_dateoffset, create_branch_anchors
from settree.temporal_explainability import (
    derive_monthly_bin_edges,
    run_temporal_explainability,
)
from settree.temporal_tree import TemporalSplitNode, TemporalTree


ANCHOR = "Birth Date Anchor"


def _fractional_training_data():
    birth_date = pd.Timestamp("2026-01-10")
    rows = []
    labels = []
    for patient_index in range(8):
        patient_id = f"P{patient_index:02d}"
        is_late = patient_index >= 4
        event_time = birth_date + pd.Timedelta(days=22 if is_late else 21)
        rows.extend(
            [
                {
                    "patient_id": patient_id,
                    "time": birth_date,
                    "feature_name": ANCHOR,
                    "value": birth_date,
                },
                {
                    "patient_id": patient_id,
                    "time": event_time,
                    "feature_name": "signal",
                    "value": 1.0,
                },
            ]
        )
        labels.append(float(is_late))
    return pd.DataFrame(rows), pd.Series(labels)


def _fractional_tree(**kwargs):
    parameters = {
        "max_depth": 1,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "anchors_list": [ANCHOR],
        "anchor_offset_list": [1],
        "anchor_offset_map": {ANCHOR: [0.7, 1]},
        "aggregation_funcs": {"max": np.max},
        "is_classification": False,
        "criterion": "mse",
        "learn_missing_direction": False,
        "candidate_sampling_rate": 1.0,
        "use_dynamic_anchors": False,
        "verbose": 0,
    }
    parameters.update(kwargs)
    return TemporalTree(**parameters)


@pytest.mark.parametrize(
    ("offset", "expected"),
    [
        (1, pd.Timestamp("2026-02-10")),
        (-1, pd.Timestamp("2025-12-10")),
        (0.7, pd.Timestamp("2026-01-31")),
        (-0.7, pd.Timestamp("2025-12-20")),
        (1.5, pd.Timestamp("2026-02-24")),
        (-1.5, pd.Timestamp("2025-11-26")),
    ],
)
def test_offset_conversion_preserves_calendar_months_and_signed_fractional_days(
    offset, expected,
):
    anchor = pd.Timestamp("2026-01-10")

    assert anchor + _offset_to_dateoffset(offset) == expected


def test_fractional_aggregation_and_full_timeline_boundaries():
    birth_date = pd.Timestamp("2026-01-10")
    X = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "time": birth_date,
                "feature_name": ANCHOR,
                "value": birth_date,
            },
            {
                "patient_id": "P1",
                "time": pd.Timestamp("2026-01-31"),
                "feature_name": "signal",
                "value": 1.0,
            },
            {
                "patient_id": "P1",
                "time": pd.Timestamp("2026-02-01"),
                "feature_name": "signal",
                "value": 3.0,
            },
        ]
    )
    model = TemporalTree(verbose=0)

    fractional, _ = model._aggregate_feature_over_time(
        X, "signal", (ANCHOR, 0.7), "sum",
    )
    calendar_month, _ = model._aggregate_feature_over_time(
        X, "signal", (ANCHOR, 1), "sum",
    )
    full_timeline, resolved_anchor = model._aggregate_feature_over_time(
        X, "signal", (None, None), "sum",
    )

    np.testing.assert_array_equal(fractional, [1.0])
    np.testing.assert_array_equal(calendar_month, [4.0])
    np.testing.assert_array_equal(full_timeline, [4.0])
    assert resolved_anchor is None


def test_fractional_candidate_is_generated_selected_and_reused_for_prediction():
    X, y = _fractional_training_data()
    model = _fractional_tree()
    evaluated_windows = []
    aggregate = model._aggregate_feature_over_time

    def recording_aggregate(self, data, feature, tau, aggregation, **kwargs):
        if kwargs.get("aggregation_result_cache") is not None:
            evaluated_windows.append(tau)
        return aggregate(data, feature, tau, aggregation, **kwargs)

    model._aggregate_feature_over_time = MethodType(recording_aggregate, model)
    model.fit(X, y)

    assert (ANCHOR, 0.7) in evaluated_windows
    assert model.tree.time_window == (ANCHOR, 0.7)
    np.testing.assert_array_equal(model.predict(X), y.to_numpy())


def test_fractional_dynamic_anchor_boundary_matches_training_and_prediction():
    birth_date = pd.Timestamp("2026-01-10")
    row = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "time": birth_date,
                "feature_name": ANCHOR,
                "value": birth_date,
            },
            {
                "patient_id": "P1",
                "time": pd.Timestamp("2026-01-31"),
                "feature_name": "signal",
                "value": 1.0,
            },
            {
                "patient_id": "P1",
                "time": pd.Timestamp("2026-02-01"),
                "feature_name": "signal",
                "value": 1.0,
            },
        ]
    )
    node = TemporalSplitNode(
        source_node_id=0,
        feature="signal",
        threshold=0.5,
        aggregation="max",
        time_window=(ANCHOR, 0.7),
    )
    model = TemporalTree(feature_semantics={"signal": "state_measurement"},)
    anchor_name = model._format_dynamic_anchor_name(node, "left")

    training_anchor = create_branch_anchors(
        ["P1"],
        row,
        anchor_name,
        "left",
        "signal",
        0.5,
        (ANCHOR, 0.7),
        aggregation="max",
    )
    prediction_data = model._append_prediction_branch_anchor(row, node, "left")
    prediction_anchor = prediction_data.loc[
        prediction_data["feature_name"].eq(anchor_name), "value"
    ].iloc[0]

    assert training_anchor.loc[0, "value"] == [pd.Timestamp("2026-01-31")]
    assert prediction_anchor == training_anchor.loc[0, "value"]
    assert anchor_name == "Node 0 | MAX(signal) Anchor"


@pytest.mark.parametrize("invalid_offset", [True, "0.7", np.nan, np.inf, -np.inf])
def test_invalid_offsets_still_fail_clearly(invalid_offset):
    X, y = _fractional_training_data()
    model = _fractional_tree(anchor_offset_map={ANCHOR: [invalid_offset]})

    with pytest.raises(ValueError, match="finite real numbers or None"):
        model.fit(X, y)


def test_fractional_explainability_preserves_audit_offset_and_monthly_bins(tmp_path,):
    X, y = _fractional_training_data()
    model = _fractional_tree().fit(X, y)
    bin_edges = derive_monthly_bin_edges({ANCHOR: [None, 0.7]})

    result = run_temporal_explainability(model, X, tmp_path, bin_edges)
    path_table = result["path_table"]
    usage_table = result["usage_table"]

    assert bin_edges == {ANCHOR: [0, 1]}
    assert set(path_table["stored_offset"]) == {0.7}
    assert set(path_table["effective_start_offset"]) == {0}
    assert set(path_table["effective_end_offset"]) == {0.7}
    assert usage_table[["bin_start_offset", "bin_end_offset"]].values.tolist() == [
        [0.0, 1.0]
    ]
    assert usage_table["raw_usage_count"].tolist() == [len(y)]
    assert result["histogram_path"].is_file()
