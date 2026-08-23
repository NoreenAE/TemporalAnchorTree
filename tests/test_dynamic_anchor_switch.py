from types import MethodType

import dill
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
            ]
        )
        labels.append(float(patient_index))
    return pd.DataFrame(rows), pd.Series(labels)


def _tree(use_dynamic_anchors=None):
    parameters = {
        "max_depth": 2,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "anchors_list": FIXED_ANCHORS.copy(),
        "anchor_offset_list": [1],
        "aggregation_funcs": {"mean": np.mean},
        "is_classification": False,
        "criterion": "mse",
        "candidate_sampling_rate": 1.0,
        "verbose": 0,
    }
    if use_dynamic_anchors is not None:
        parameters["use_dynamic_anchors"] = use_dynamic_anchors
    return TemporalTree(**parameters)


def _record_candidate_anchor_lists(model):
    observed_anchor_lists = []
    find_best_split = model._find_best_split

    def recording_find_best_split(self, X, y, anchors_list=None):
        observed_anchor_lists.append(tuple(anchors_list))
        return find_best_split(X, y, anchors_list=anchors_list)

    model._find_best_split = MethodType(recording_find_best_split, model)
    return observed_anchor_lists


def _split_anchors(node):
    if node.is_leaf():
        return []
    return [
        node.time_window[0],
        *_split_anchors(node.left),
        *_split_anchors(node.right),
    ]


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


def test_default_behavior_keeps_dynamic_anchors_available_to_descendants():
    X, y = _training_data()
    model = _tree()
    observed_anchor_lists = _record_candidate_anchor_lists(model)

    model.fit(X, y)

    assert model.use_dynamic_anchors is True
    assert observed_anchor_lists[0] == tuple(FIXED_ANCHORS)
    assert any(
        len(anchor_list) > len(FIXED_ANCHORS)
        and any(anchor not in FIXED_ANCHORS for anchor in anchor_list)
        for anchor_list in observed_anchor_lists[1:]
    )
    assert model.predict(X).shape == (len(y),)


def test_explicit_true_matches_the_omitted_default_exactly():
    X, y = _training_data()
    default_model = _tree().fit(X, y)
    explicit_model = _tree(use_dynamic_anchors=True).fit(X, y)

    assert _tree_signature(default_model.tree) == _tree_signature(
        explicit_model.tree
    )
    np.testing.assert_array_equal(
        default_model.predict(X),
        explicit_model.predict(X),
    )
    assert default_model.generated_groups == explicit_model.generated_groups
    assert default_model.evaluated_threshold_candidates == (
        explicit_model.evaluated_threshold_candidates
    )


def test_legacy_model_state_without_flag_preserves_dynamic_behavior(tmp_path):
    X, y = _training_data()
    model = _tree().fit(X, y)
    expected_predictions = model.predict(X)
    del model.use_dynamic_anchors

    model_path = tmp_path / "legacy-model.dill"
    with model_path.open("wb") as model_file:
        dill.dump(model, model_file)
    with model_path.open("rb") as model_file:
        model = dill.load(model_file)

    assert model.use_dynamic_anchors is True
    assert model.get_params()["use_dynamic_anchors"] is True
    np.testing.assert_array_equal(model.predict(X), expected_predictions)
    observed_anchor_lists = _record_candidate_anchor_lists(model)
    model.fit(X, y)
    assert any(
        len(anchor_list) > len(FIXED_ANCHORS)
        for anchor_list in observed_anchor_lists[1:]
    )


def test_disabling_dynamic_anchors_keeps_every_descendant_on_fixed_anchors():
    X, y = _training_data()
    model = _tree(use_dynamic_anchors=False)
    observed_anchor_lists = _record_candidate_anchor_lists(model)

    model.fit(X, y)

    assert len(observed_anchor_lists) > 1
    assert all(
        anchor_list == tuple(FIXED_ANCHORS)
        for anchor_list in observed_anchor_lists
    )
    assert set(_split_anchors(model.tree)).issubset(FIXED_ANCHORS)


def test_prediction_and_leaf_routing_work_in_both_anchor_modes():
    X, y = _training_data()
    for use_dynamic_anchors in (True, False):
        model = _tree(use_dynamic_anchors=use_dynamic_anchors).fit(X, y)

        predictions = model.predict(X)
        combined_predictions, combined_leaf_ids = model.predict_and_apply(X)
        leaf_ids = model.apply(X)

        np.testing.assert_array_equal(predictions, combined_predictions)
        np.testing.assert_array_equal(leaf_ids, combined_leaf_ids)
        assert predictions.shape == leaf_ids.shape == (len(y),)
