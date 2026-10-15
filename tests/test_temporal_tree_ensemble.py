import dill
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from TAT.temporal_tree import TemporalTree
from TAT.temporal_tree_ensemble import (
    MAX_SPLIT_SAMPLING_SEED,
    TemporalTreeEnsemble,
)


def _data(num_patients=8):
    patient_ids = [f"P{index:03d}" for index in range(num_patients)]
    rows = []
    for feature_index in range(4):
        for patient_index, patient_id in enumerate(patient_ids):
            rows.append(
                {
                    "patient_id": patient_id,
                    "time": pd.Timestamp("2020-01-01"),
                    "feature_name": f"feature_{feature_index}",
                    "value": float(
                        (patient_index + feature_index) % num_patients
                    ),
                }
            )
    X = pd.DataFrame(rows)
    y = pd.Series(np.linspace(0.0, 1.0, num_patients))
    return X, y


def _base_tree(rate=0.5, verbose=0):
    return TemporalTree(
        max_depth=1,
        min_samples_leaf=1,
        anchors_list=["Birth Date Anchor"],
        anchor_offset_list=[None],
        aggregation_funcs={"mean": np.mean},
        is_classification=False,
        criterion="mse",
        random_state=17,
        candidate_sampling_rate=rate,
        split_sampling_seed=999,
        verbose=verbose,
    )


def _tree_signature(model):
    root = model.tree
    return (
        root.feature,
        root.threshold,
        root.aggregation,
        root.time_window,
        root.missing_direction,
        root.left.value,
        root.right.value,
    )


def test_one_estimator_exactly_matches_direct_temporal_tree():
    X, y = _data()
    direct = clone(_base_tree()).set_params(split_sampling_seed=42)
    ensemble = TemporalTreeEnsemble(
        estimator=_base_tree(),
        n_estimators=1,
        base_split_sampling_seed=42,
    )

    direct.fit(X, y)
    ensemble.fit(X, y)

    member = ensemble.estimators_[0]
    assert ensemble.split_sampling_seeds_ == (42,)
    assert _tree_signature(member) == _tree_signature(direct)
    assert np.array_equal(ensemble.predict(X), direct.predict(X))
    assert member.generated_groups == direct.generated_groups
    assert member.sampled_groups == direct.sampled_groups
    assert member.evaluated_threshold_candidates == (
        direct.evaluated_threshold_candidates
    )


def test_generated_and_explicit_seed_resolution():
    generated = TemporalTreeEnsemble(
        estimator=_base_tree(),
        n_estimators=5,
        base_split_sampling_seed=42,
    )
    explicit = TemporalTreeEnsemble(
        estimator=_base_tree(),
        n_estimators=3,
        base_split_sampling_seed="ignored",
        split_sampling_seeds=[9, 4, 7],
    )

    assert generated._resolve_split_sampling_seeds() == (42, 43, 44, 45, 46)
    assert explicit._resolve_split_sampling_seeds() == (9, 4, 7)


def test_explicit_seed_length_and_duplicates_are_rejected():
    with pytest.raises(ValueError, match="must equal n_estimators"):
        TemporalTreeEnsemble(
            estimator=_base_tree(),
            n_estimators=2,
            split_sampling_seeds=[42],
        )._resolve_split_sampling_seeds()

    with pytest.raises(ValueError, match="must not contain duplicates"):
        TemporalTreeEnsemble(
            estimator=_base_tree(),
            n_estimators=2,
            split_sampling_seeds=[42, 42],
        )._resolve_split_sampling_seeds()


@pytest.mark.parametrize(
    "seed",
    [True, 1.5, "42", -1, MAX_SPLIT_SAMPLING_SEED + 1],
)
def test_invalid_explicit_seed_type_or_range_is_rejected(seed):
    ensemble = TemporalTreeEnsemble(
        estimator=_base_tree(),
        n_estimators=1,
        split_sampling_seeds=[seed],
    )

    with pytest.raises(ValueError, match="split_sampling_seeds"):
        ensemble._resolve_split_sampling_seeds()


@pytest.mark.parametrize("n_estimators", [True, 0, -1, 1.5])
def test_invalid_n_estimators_is_rejected(n_estimators):
    with pytest.raises(ValueError, match="n_estimators"):
        TemporalTreeEnsemble(
            estimator=_base_tree(),
            n_estimators=n_estimators,
        )._resolve_split_sampling_seeds()


def test_generated_seed_overflow_is_rejected():
    with pytest.raises(ValueError, match="exceeds"):
        TemporalTreeEnsemble(
            estimator=_base_tree(),
            n_estimators=2,
            base_split_sampling_seed=MAX_SPLIT_SAMPLING_SEED,
        )._resolve_split_sampling_seeds()


def test_members_are_independent_clones_and_only_split_seed_changes():
    X, y = _data()
    base = _base_tree()
    ensemble = TemporalTreeEnsemble(
        estimator=base,
        n_estimators=3,
        split_sampling_seeds=[42, 43, 44],
    ).fit(X, y)

    assert len({id(member) for member in ensemble.estimators_}) == 3
    assert all(member is not base for member in ensemble.estimators_)
    base_parameters = base.get_params(deep=False)
    for seed, member in zip(
        ensemble.split_sampling_seeds_,
        ensemble.estimators_,
    ):
        member_parameters = member.get_params(deep=False)
        assert member_parameters.pop("split_sampling_seed") == seed
        expected_parameters = base_parameters.copy()
        expected_parameters.pop("split_sampling_seed")
        assert member_parameters == expected_parameters


def test_predictions_are_reproducible_aligned_ordered_and_meaned():
    X, y = _data()
    patient_order = ["P005", "P002", "P007", "P000"]
    prediction_X = pd.concat(
        [X[X["patient_id"] == patient_id] for patient_id in patient_order],
        ignore_index=True,
    )
    ensemble = TemporalTreeEnsemble(
        estimator=_base_tree(),
        n_estimators=3,
        split_sampling_seeds=[42, 43, 44],
    )

    ensemble.fit(X, y)
    first_individual = ensemble.predict_individual(prediction_X)
    first_mean = ensemble.predict(prediction_X)
    direct_member_predictions = np.vstack(
        [member.predict(prediction_X) for member in ensemble.estimators_]
    )

    assert first_individual.shape == (3, 4)
    assert np.array_equal(first_individual, direct_member_predictions)
    assert np.array_equal(first_mean, first_individual.mean(axis=0))

    ensemble.fit(X, y)
    assert np.array_equal(
        ensemble.predict_individual(prediction_X),
        first_individual,
    )
    assert np.array_equal(ensemble.predict(prediction_X), first_mean)


def test_refit_replaces_fitted_members_and_metadata():
    X, y = _data()
    ensemble = TemporalTreeEnsemble(
        estimator=_base_tree(),
        n_estimators=2,
        split_sampling_seeds=[42, 43],
    ).fit(X, y)
    original_member_ids = [id(member) for member in ensemble.estimators_]

    ensemble.set_params(split_sampling_seeds=[44, 45])
    ensemble.fit(X, y)

    assert ensemble.split_sampling_seeds_ == (44, 45)
    assert [id(member) for member in ensemble.estimators_] != original_member_ids
    assert len(ensemble.fit_times_) == ensemble.n_estimators_ == 2
    assert ensemble.fit_time_ >= 0


def test_member_failure_fails_atomically_with_member_context(monkeypatch):
    X, y = _data()
    fit = TemporalTree.fit

    def failing_fit(self, member_X, member_y):
        if self.split_sampling_seed == 43:
            raise ValueError("deliberate failure")
        return fit(self, member_X, member_y)

    monkeypatch.setattr(TemporalTree, "fit", failing_fit)
    ensemble = TemporalTreeEnsemble(
        estimator=_base_tree(),
        n_estimators=2,
        split_sampling_seeds=[42, 43],
    )

    with pytest.raises(
        RuntimeError,
        match="estimator_index=1, split_sampling_seed=43",
    ):
        ensemble.fit(X, y)
    assert not hasattr(ensemble, "estimators_")
    assert not hasattr(ensemble, "fit_times_")


def test_exhaustive_members_warn_and_classification_is_rejected():
    X, y = _data()
    exhaustive = TemporalTreeEnsemble(
        estimator=_base_tree(rate=1.0),
        n_estimators=2,
        split_sampling_seeds=[42, 43],
    )

    with pytest.warns(UserWarning, match="not expected to create"):
        exhaustive.fit(X, y)

    classification_tree = _base_tree().set_params(
        is_classification=True,
        criterion="gini",
    )
    with pytest.raises(ValueError, match="is_classification=False"):
        TemporalTreeEnsemble(
            estimator=classification_tree,
            n_estimators=1,
        ).fit(X, y)


def test_empty_valid_prediction_input_and_not_fitted_behavior():
    X, y = _data()
    ensemble = TemporalTreeEnsemble(
        estimator=_base_tree(),
        n_estimators=2,
        split_sampling_seeds=[42, 43],
    )

    with pytest.raises(NotFittedError):
        ensemble.predict(X)

    ensemble.fit(X, y)
    empty_X = X.iloc[:0].copy()
    assert ensemble.predict_individual(empty_X).shape == (2, 0)
    assert ensemble.predict(empty_X).shape == (0,)


def test_sklearn_clone_get_params_and_set_params_compatibility():
    ensemble = TemporalTreeEnsemble(
        estimator=_base_tree(),
        n_estimators=3,
        base_split_sampling_seed=42,
    )
    cloned = clone(ensemble)

    assert cloned.get_params()["n_estimators"] == 3
    assert cloned.get_params()["estimator__max_depth"] == 1
    cloned.set_params(estimator__max_depth=2, n_estimators=4)
    assert cloned.estimator.max_depth == 2
    assert cloned.n_estimators == 4


def test_dill_round_trip_preserves_members_and_predictions(tmp_path):
    X, y = _data()
    ensemble = TemporalTreeEnsemble(
        estimator=_base_tree(),
        n_estimators=2,
        split_sampling_seeds=[42, 43],
    ).fit(X, y)
    expected = ensemble.predict_individual(X)
    model_path = tmp_path / "ensemble.dill"

    with model_path.open("wb") as model_file:
        dill.dump(ensemble, model_file)
    with model_path.open("rb") as model_file:
        restored = dill.load(model_file)

    assert restored.split_sampling_seeds_ == (42, 43)
    assert len(restored.estimators_) == 2
    assert np.array_equal(restored.predict_individual(X), expected)
