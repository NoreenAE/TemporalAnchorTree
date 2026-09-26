"""Class distributions and label mapping for TemporalTree predictions."""

import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from settree.temporal_tree import TemporalTree


def _data(values):
    return pd.DataFrame(
        {
            "patient_id": [f"P{i}" for i in range(len(values))],
            "time": pd.Timestamp("2020-01-01"),
            "feature_name": "signal",
            "value": values,
        }
    )


def _model(**overrides):
    parameters = dict(
        max_depth=2,
        min_samples_split=2,
        min_samples_leaf=1,
        anchor_offset_list=[None],
        aggregation_funcs={"sum": np.sum},
        is_classification=True,
        criterion="gini",
        verbose=0,
        candidate_sampling_rate=1.0,
        feature_semantics={"signal": "state_measurement"},
    )
    parameters.update(overrides)
    return TemporalTree(**parameters)


def _check_consistency(model, data):
    probabilities = model.predict_proba(data)
    assert probabilities.shape == (
        data.patient_id.nunique(),
        len(model.classes_),
    )
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    expected = model.classes_[probabilities.argmax(axis=1)]
    np.testing.assert_array_equal(model.predict(data), expected)
    predictions, leaves = model.predict_and_apply(data)
    np.testing.assert_array_equal(predictions, expected)
    np.testing.assert_array_equal(leaves, model.apply(data))
    return probabilities


@pytest.mark.parametrize("dynamic", [False, True])
def test_pure_positive_and_negative_leaves_and_auc(dynamic):
    data = _data([0, 0, 1, 1])
    labels = pd.Series([0, 0, 1, 1])
    model = _model(use_dynamic_anchors=dynamic).fit(data, labels)
    np.testing.assert_array_equal(model.classes_, [0, 1])
    np.testing.assert_array_equal(model.tree.left.value, [1, 0])
    np.testing.assert_array_equal(model.tree.right.value, [0, 1])
    probabilities = _check_consistency(model, data)
    np.testing.assert_array_equal(model.predict(data), labels)
    positive_column = np.flatnonzero(model.classes_ == 1)[0]
    assert roc_auc_score(labels, probabilities[:, positive_column]) == 1
    assert "Probabilities" in model.tree.right.describe()


@pytest.mark.parametrize("stop", ["depth", "size", "no_split", "gain"])
def test_mixed_leaf_retains_distribution_for_every_stopping_path(stop):
    parameters = {
        "depth": {"max_depth": 0},
        "size": {"min_samples_split": 5},
        "no_split": {},
        "gain": {"min_impurity_decrease": 1.0},
    }[stop]
    data = _data([1, 1, 1, 1] if stop == "no_split" else [0, 1, 2, 3])
    model = _model(**parameters).fit(data, pd.Series([0, 1, 1, 1]))
    assert model.tree.is_leaf()
    np.testing.assert_allclose(model.tree.value, [0.25, 0.75])
    np.testing.assert_allclose(
        _check_consistency(model, data), [[0.25, 0.75]] * 4
    )
    np.testing.assert_array_equal(model.predict(data), [1] * 4)


@pytest.mark.parametrize("label", [0, 1, "positive"])
def test_single_class_fit_has_one_column_and_returns_that_class(label):
    data = _data([0, 1, 2])
    model = _model().fit(data, pd.Series([label] * 3))
    np.testing.assert_array_equal(model.classes_, [label])
    np.testing.assert_array_equal(
        _check_consistency(model, data), np.ones((3, 1))
    )
    np.testing.assert_array_equal(model.predict(data), [label] * 3)


@pytest.mark.parametrize("labels", [[20, 10, 30], ["z", "a", "m"]])
def test_nonconsecutive_and_string_classes_use_sorted_global_columns(labels):
    data = _data([0, 1, 2])
    model = _model().fit(data, pd.Series(labels))
    np.testing.assert_array_equal(model.classes_, sorted(labels))
    _check_consistency(model, data)
    np.testing.assert_array_equal(model.predict(data), labels)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(
        restored.predict_proba(data), model.predict_proba(data)
    )
    np.testing.assert_array_equal(restored.classes_, model.classes_)


def test_mixed_child_uses_global_classes_and_ties_choose_first_class():
    data = _data([0, 0, 1, 1])
    model = _model(max_depth=1).fit(data, pd.Series([20, 10, 30, 30]))
    np.testing.assert_array_equal(model.classes_, [10, 20, 30])
    np.testing.assert_allclose(model.tree.left.value, [0.5, 0.5, 0])
    np.testing.assert_allclose(model.tree.right.value, [0, 0, 1])
    _check_consistency(model, data)
    np.testing.assert_array_equal(model.predict(data), [10, 10, 30, 30])


def test_refit_replaces_class_order():
    data = _data([0, 1])
    model = _model().fit(data, pd.Series([0, 1]))
    model.fit(data, pd.Series([9, 4]))
    np.testing.assert_array_equal(model.classes_, [4, 9])
    _check_consistency(model, data)
    np.testing.assert_array_equal(model.predict(data), [9, 4])


def test_regression_leaf_stays_scalar_mean():
    data = _data([0, 1, 2, 3])
    model = _model(is_classification=False, criterion="mse", max_depth=0)
    model.fit(data, pd.Series([0.0, 1.0, 1.0, 1.0]))
    assert np.isscalar(model.tree.value)
    assert model.tree.value == 0.75
    np.testing.assert_array_equal(model.predict(data), [0.75] * 4)
    assert model.tree.describe() == "Leaf: Predict = 0.75 (patients: 4)"
