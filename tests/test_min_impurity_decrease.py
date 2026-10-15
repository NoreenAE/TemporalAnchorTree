import dill
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from TAT.temporal_tree import TemporalLeaf, TemporalSplitNode, TemporalTree


def _known_gain_data(num_features=1):
    patient_ids = ["P0", "P1", "P2", "P3"]
    rows = []
    for feature_index in range(num_features):
        for patient_id, value in zip(patient_ids, [0.0, 0.0, 1.0, 1.0]):
            rows.append({
                "patient_id": patient_id,
                "time": pd.Timestamp("2020-01-01"),
                "feature_name": f"signal_{feature_index}",
                "value": value,
            })
    return pd.DataFrame(rows), pd.Series([0.0, 0.0, 1.0, 1.0])


def _tree(**kwargs):
    parameters = {
        "max_depth": 1,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "anchors_list": ["Birth Date Anchor"],
        "anchor_offset_list": [None],
        "aggregation_funcs": {"mean": np.mean},
        "is_classification": False,
        "criterion": "mse",
        "candidate_sampling_rate": 1.0,
        "split_sampling_seed": 42,
        "verbose": 0,
        "use_dynamic_anchors": False,
    }
    parameters.update(kwargs)
    return TemporalTree(**parameters)


def _split_signature(model):
    root = model.tree
    if isinstance(root, TemporalLeaf):
        return ("leaf", root.value, root.num_patients, root.depth)
    return (
        "split",
        root.feature,
        root.threshold,
        root.aggregation,
        root.time_window,
        root.missing_direction,
        root.impurity,
        root.left.value,
        root.right.value,
    )


def _candidate_counters(model):
    names = (
        "generated_groups",
        "sampled_groups",
        "sampling_skipped_groups",
        "generated_threshold_candidates",
        "invalid_threshold_candidates",
        "valid_threshold_candidates",
        "evaluated_threshold_candidates",
        "fallback_evaluated_groups",
        "fallback_generated_threshold_candidates",
        "fallback_valid_threshold_candidates",
        "fallback_evaluated_threshold_candidates",
        "nodes_requiring_sampling_fallback",
    )
    return {name: getattr(model, name) for name in names}


def test_zero_exactly_preserves_existing_split_behavior():
    X, y = _known_gain_data()
    omitted = _tree().fit(X, y)
    explicit_zero = _tree(min_impurity_decrease=0.0).fit(X, y)

    assert _split_signature(explicit_zero) == _split_signature(omitted)
    assert np.array_equal(explicit_zero.predict(X), omitted.predict(X))
    assert _candidate_counters(explicit_zero) == _candidate_counters(omitted)


def test_large_minimum_gain_prevents_split_and_uses_existing_leaf_value():
    X, y = _known_gain_data()
    model = _tree(min_impurity_decrease=1.0).fit(X, y)

    assert isinstance(model.tree, TemporalLeaf)
    assert model.tree.value == pytest.approx(y.mean())
    assert model.evaluated_threshold_candidates > 0


@pytest.mark.parametrize("minimum_gain", [0.249, 0.25])
def test_minimum_at_or_below_known_gain_allows_split(minimum_gain):
    X, y = _known_gain_data()
    model = _tree(min_impurity_decrease=minimum_gain).fit(X, y)

    assert isinstance(model.tree, TemporalSplitNode)
    assert model.tree.impurity == pytest.approx(0.0)


def test_minimum_above_known_gain_blocks_split():
    X, y = _known_gain_data()
    model = _tree(min_impurity_decrease=0.251).fit(X, y)

    assert isinstance(model.tree, TemporalLeaf)


def test_gain_gate_does_not_change_sampling_or_candidate_search():
    X, y = _known_gain_data(num_features=4)
    baseline = _tree(candidate_sampling_rate=0.5).fit(X, y)
    blocked = _tree(
        candidate_sampling_rate=0.5,
        min_impurity_decrease=1.0,
    ).fit(X, y)

    assert _candidate_counters(blocked) == _candidate_counters(baseline)
    assert isinstance(baseline.tree, TemporalSplitNode)
    assert isinstance(blocked.tree, TemporalLeaf)


@pytest.mark.parametrize(
    "invalid_value",
    [-0.1, np.nan, np.inf, True, "0.1"],
)
def test_minimum_gain_validation(invalid_value):
    X, y = _known_gain_data()

    with pytest.raises(ValueError, match="min_impurity_decrease"):
        _tree(min_impurity_decrease=invalid_value).fit(X, y)


def test_parameter_is_cloneable_and_serializable():
    model = _tree(min_impurity_decrease=0.125)

    assert clone(model).min_impurity_decrease == 0.125
    assert dill.loads(dill.dumps(model)).min_impurity_decrease == 0.125


def test_legacy_serialized_state_defaults_to_zero():
    model = _tree(min_impurity_decrease=0.125)
    state = model.__dict__.copy()
    del state["min_impurity_decrease"]
    restored = TemporalTree.__new__(TemporalTree)

    restored.__setstate__(state)

    assert restored.min_impurity_decrease == 0.0
