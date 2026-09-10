from types import MethodType

import numpy as np
import pandas as pd

from settree.temporal_tree import TemporalTree


FIXED_ANCHORS = ["Birth Date Anchor", "NOW Anchor"]


def _training_data(num_patients=8):
    rows = []
    labels = []
    for patient_index in range(num_patients):
        patient_id = f"P{patient_index:03d}"
        birth_date = pd.Timestamp("2020-01-01")
        now = pd.Timestamp("2021-01-01")
        rows.extend(
            [
                {
                    "patient_id": patient_id,
                    "time": birth_date,
                    "feature_name": "Birth Date Anchor",
                    "value": birth_date,
                },
                {
                    "patient_id": patient_id,
                    "time": now,
                    "feature_name": "NOW Anchor",
                    "value": now,
                },
                {
                    "patient_id": patient_id,
                    "time": birth_date + pd.Timedelta(days=15),
                    "feature_name": "signal",
                    "value": float(patient_index),
                },
                {
                    "patient_id": patient_id,
                    "time": now - pd.Timedelta(days=15),
                    "feature_name": "signal",
                    "value": float(patient_index),
                },
            ]
        )
        labels.append(float(patient_index))
    return pd.DataFrame(rows), pd.Series(labels)


def _tree(**kwargs):
    parameters = {
        "max_depth": 1,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "anchors_list": FIXED_ANCHORS.copy(),
        "anchor_offset_list": [-1, 1],
        "aggregation_funcs": {"sum": np.sum},
        "feature_semantics": {"signal": "event_magnitude"},
        "is_classification": False,
        "criterion": "mse",
        "candidate_sampling_rate": 1.0,
        "verbose": 0,
    }
    parameters.update(kwargs)
    return TemporalTree(**parameters)


def _record_training_windows(model):
    windows = []
    aggregate = model._aggregate_feature_over_time

    def recording_aggregate(self, X, feature, tau, aggregation, **kwargs):
        if kwargs.get("aggregation_result_cache") is not None:
            windows.append(tau)
        return aggregate(X, feature, tau, aggregation, **kwargs)

    model._aggregate_feature_over_time = MethodType(recording_aggregate, model)
    return windows


def _tree_signature(node):
    if node.is_leaf():
        return ("leaf", node.value, node.num_patients, node.depth)
    return (
        "split",
        node.feature,
        node.threshold,
        node.aggregation,
        node.time_window,
        node.missing_direction,
        _tree_signature(node.left),
        _tree_signature(node.right),
    )


def test_none_map_exactly_matches_omitted_map_behavior():
    X, y = _training_data()
    omitted_model = _tree().fit(X, y)
    none_model = _tree(anchor_offset_map=None).fit(X, y)

    assert _tree_signature(omitted_model.tree) == _tree_signature(none_model.tree)
    np.testing.assert_array_equal(
        omitted_model.predict(X),
        none_model.predict(X),
    )
    assert omitted_model.generated_groups == none_model.generated_groups
    assert omitted_model.generated_threshold_candidates == (
        none_model.generated_threshold_candidates
    )
    assert omitted_model.evaluated_threshold_candidates == (
        none_model.evaluated_threshold_candidates
    )


def test_mapped_anchor_uses_override_and_absent_anchor_uses_general_offsets():
    X, y = _training_data()
    model = _tree(
        anchor_offset_list=[-1],
        anchor_offset_map={"Birth Date Anchor": [1]},
        use_dynamic_anchors=False,
    )
    training_windows = _record_training_windows(model)

    model.fit(X, y)

    assert training_windows == [
        ("Birth Date Anchor", 1),
        ("NOW Anchor", -1),
    ]


def test_dynamic_anchors_absent_from_map_fall_back_to_general_offsets():
    X, y = _training_data()
    model = _tree(
        max_depth=2,
        anchors_list=["Birth Date Anchor"],
        anchor_offset_list=[2],
        anchor_offset_map={"Birth Date Anchor": [1]},
        use_dynamic_anchors=True,
    )
    training_windows = _record_training_windows(model)

    model.fit(X, y)

    dynamic_windows = [
        (anchor, offset)
        for anchor, offset in training_windows
        if anchor != "Birth Date Anchor"
    ]
    assert dynamic_windows
    assert all(offset == 2 for _, offset in dynamic_windows)


def test_fixed_anchor_mode_uses_only_each_fixed_anchor_override():
    X, y = _training_data()
    model = _tree(
        max_depth=2,
        anchor_offset_map={
            "Birth Date Anchor": [1],
            "NOW Anchor": [-1],
        },
        use_dynamic_anchors=False,
    )
    training_windows = _record_training_windows(model)

    model.fit(X, y)

    assert set(training_windows) == {
        ("Birth Date Anchor", 1),
        ("NOW Anchor", -1),
    }
    assert model.predict(X).shape == (len(y),)


def test_legacy_model_state_defaults_to_no_offset_overrides():
    model = _tree(anchor_offset_map={"Birth Date Anchor": [1]})
    state = model.__dict__.copy()
    del state["anchor_offset_map"]

    restored = TemporalTree.__new__(TemporalTree)
    restored.__setstate__(state)

    assert restored.anchor_offset_map is None
    assert restored.get_params()["anchor_offset_map"] is None
