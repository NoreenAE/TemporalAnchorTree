from types import MethodType

import numpy as np
import pandas as pd
import pytest

from TAT.temporal_tree import TemporalTree


def _training_data(num_patients=100, num_features=1):
    patient_ids = [f"P{index:04d}" for index in range(num_patients)]
    rows = []
    for feature_index in range(num_features):
        for patient_index, patient_id in enumerate(patient_ids):
            rows.append(
                {
                    "patient_id": patient_id,
                    "time": pd.Timestamp("2020-01-01"),
                    "feature_name": f"signal_{feature_index}",
                    "value": float(patient_index + feature_index),
                }
            )
    X = pd.DataFrame(rows)
    y = pd.Series(np.arange(num_patients, dtype=float), index=patient_ids)
    return X, y


def _tree(
    candidate_sampling_rate,
    split_sampling_seed=42,
    verbose=0,
    min_samples_leaf=1,
    aggregation_funcs=None,
    **kwargs,
):
    return TemporalTree(
        max_depth=1,
        min_samples_leaf=min_samples_leaf,
        anchors_list=["Birth Date Anchor"],
        anchor_offset_list=[None],
        aggregation_funcs=aggregation_funcs or {"mean": np.mean},
        is_classification=False,
        criterion="mse",
        candidate_sampling_rate=candidate_sampling_rate,
        split_sampling_seed=split_sampling_seed,
        verbose=verbose,
        **kwargs,
    )


def _record_search_aggregations(model):
    calls = []
    aggregate = model._aggregate_feature_over_time

    def recording_aggregate(self, X, feature, tau, aggregation, **kwargs):
        if kwargs.get("aggregation_result_cache") is not None:
            calls.append((feature, tau, aggregation))
        return aggregate(X, feature, tau, aggregation, **kwargs)

    model._aggregate_feature_over_time = MethodType(recording_aggregate, model)
    return calls


@pytest.mark.parametrize(
    "invalid_rate",
    [0, -0.1, 1.1, np.nan, np.inf, True, "0.1"],
)
def test_candidate_sampling_rate_validation(invalid_rate):
    X, y = _training_data(4)

    with pytest.raises(ValueError, match="candidate_sampling_rate"):
        _tree(invalid_rate).fit(X, y)


def test_exhaustive_mode_preserves_all_groups_and_thresholds_without_rng_draws(
    monkeypatch,
):
    class NoDrawGenerator:
        def random(self):
            raise AssertionError("Exhaustive mode must not draw from the RNG")

    monkeypatch.setattr(np.random, "default_rng", lambda seed: NoDrawGenerator())
    X, y = _training_data(50)
    model = _tree(1.0)

    model.fit(X, y)

    assert model.generated_groups == 1
    assert model.sampled_groups == 1
    assert model.sampling_skipped_groups == 0
    assert model.generated_threshold_candidates == 50
    assert model.invalid_threshold_candidates == 1
    assert model.valid_threshold_candidates == 49
    assert model.evaluated_threshold_candidates == 49
    assert model.fallback_evaluated_groups == 0
    assert model.fallback_evaluated_threshold_candidates == 0
    assert model.nodes_requiring_sampling_fallback == 0
    assert model.tree.feature == "signal_0"
    assert model.tree.threshold == 24.0
    assert model.tree.aggregation == "mean"
    assert model.tree.time_window == (None, None)
    assert model.tree.missing_direction == "right"


def test_ten_percent_sampling_has_expected_group_rate_and_counter_identities():
    X, y = _training_data(num_patients=4, num_features=1000)
    model = _tree(0.1, split_sampling_seed=42)

    model.fit(X, y)

    observed_rate = model.sampled_groups / model.generated_groups
    assert 0.07 <= observed_rate <= 0.13
    assert model.generated_groups == 1000
    assert model.generated_groups == (
        model.sampled_groups + model.sampling_skipped_groups
    )
    assert model.generated_threshold_candidates == (
        model.invalid_threshold_candidates + model.valid_threshold_candidates
    )
    assert (
        model.valid_threshold_candidates
        == model.evaluated_threshold_candidates
    )


def test_rejected_group_never_calls_aggregation_and_retained_group_is_exhaustive(
    monkeypatch,
):
    class RecordingGenerator:
        def __init__(self):
            self.draws = iter([0.9, 0.1])
            self.draw_count = 0

        def random(self):
            self.draw_count += 1
            return next(self.draws)

    generator = RecordingGenerator()
    monkeypatch.setattr(np.random, "default_rng", lambda seed: generator)
    X, y = _training_data(num_patients=4, num_features=2)
    model = _tree(0.5)
    aggregate_calls = _record_search_aggregations(model)

    model.fit(X, y)

    assert generator.draw_count == 2
    assert aggregate_calls == [("signal_1", (None, None), "mean")]
    assert model.generated_groups == 2
    assert model.sampled_groups == 1
    assert model.sampling_skipped_groups == 1
    assert model.generated_threshold_candidates == 4
    assert model.valid_threshold_candidates == 3
    assert model.evaluated_threshold_candidates == 3
    assert model.fallback_evaluated_groups == 0


def test_repeated_fits_with_same_seed_repeat_sampled_group_sequence():
    X, y = _training_data(num_patients=5, num_features=50)
    model = _tree(0.1, split_sampling_seed=42)
    aggregate_calls = _record_search_aggregations(model)

    model.fit(X, y)
    first_calls = aggregate_calls.copy()
    first_threshold = model.tree.threshold
    first_counts = (
        model.generated_groups,
        model.sampled_groups,
        model.sampling_skipped_groups,
        model.generated_threshold_candidates,
        model.evaluated_threshold_candidates,
        model.fallback_evaluated_groups,
    )

    aggregate_calls.clear()
    model.fit(X, y)

    assert aggregate_calls == first_calls
    assert model.tree.threshold == first_threshold
    assert (
        model.generated_groups,
        model.sampled_groups,
        model.sampling_skipped_groups,
        model.generated_threshold_candidates,
        model.evaluated_threshold_candidates,
        model.fallback_evaluated_groups,
    ) == first_counts


def test_different_seeds_normally_change_sampled_group_sequence():
    X, y = _training_data(num_patients=5, num_features=50)
    first_model = _tree(0.1, split_sampling_seed=42)
    second_model = _tree(0.1, split_sampling_seed=43)
    first_calls = _record_search_aggregations(first_model)
    second_calls = _record_search_aggregations(second_model)

    first_model.fit(X, y)
    second_model.fit(X, y)

    assert first_calls != second_calls


def test_split_sampling_does_not_modify_numpy_global_random_state():
    X, y = _training_data(20)
    state_before = np.random.get_state()

    _tree(0.1, split_sampling_seed=42).fit(X, y)

    state_after = np.random.get_state()
    assert state_after[0] == state_before[0]
    assert np.array_equal(state_after[1], state_before[1])
    assert state_after[2:] == state_before[2:]


def test_fallback_replays_first_rejected_group_exhaustively_without_new_draws(
    monkeypatch,
):
    class RejectingGenerator:
        def __init__(self):
            self.draw_count = 0

        def random(self):
            self.draw_count += 1
            return 1.0

    generator = RejectingGenerator()
    monkeypatch.setattr(np.random, "default_rng", lambda seed: generator)
    X, y = _training_data(20)
    model = _tree(0.1)

    model.fit(X, y)

    assert generator.draw_count == 1
    assert model.generated_groups == 1
    assert model.sampled_groups == 0
    assert model.sampling_skipped_groups == 1
    assert model.generated_threshold_candidates == 0
    assert model.evaluated_threshold_candidates == 0
    assert model.fallback_evaluated_groups == 1
    assert model.fallback_generated_threshold_candidates == 20
    assert model.fallback_valid_threshold_candidates == 19
    assert model.fallback_evaluated_threshold_candidates == 19
    assert model.nodes_requiring_sampling_fallback == 1
    assert model.tree.threshold == 9.0


def test_fallback_stops_after_first_rejected_group_with_a_valid_threshold(
    monkeypatch,
):
    class RejectingGenerator:
        def __init__(self):
            self.draw_count = 0

        def random(self):
            self.draw_count += 1
            return 1.0

    generator = RejectingGenerator()
    monkeypatch.setattr(np.random, "default_rng", lambda seed: generator)
    patient_ids = [f"P{index:04d}" for index in range(4)]
    rows = []
    for feature, values in (
        ("constant", [1.0, 1.0, 1.0, 1.0]),
        ("valid", [0.0, 1.0, 2.0, 3.0]),
        ("must_not_replay", [3.0, 2.0, 1.0, 0.0]),
    ):
        for patient_id, value in zip(patient_ids, values):
            rows.append(
                {
                    "patient_id": patient_id,
                    "time": pd.Timestamp("2020-01-01"),
                    "feature_name": feature,
                    "value": value,
                }
            )
    X = pd.DataFrame(rows)
    y = pd.Series(np.arange(4, dtype=float), index=patient_ids)
    model = _tree(0.1)
    aggregate_calls = _record_search_aggregations(model)

    model.fit(X, y)

    assert generator.draw_count == 3
    assert model.generated_groups == model.sampling_skipped_groups == 3
    assert model.sampled_groups == 0
    assert model.fallback_evaluated_groups == 2
    assert aggregate_calls == [
        ("constant", (None, None), "mean"),
        ("valid", (None, None), "mean"),
    ]
    assert model.fallback_generated_threshold_candidates == 5
    assert model.fallback_valid_threshold_candidates == 3
    assert model.fallback_evaluated_threshold_candidates == 3
    assert model.tree.feature == "valid"


def test_missing_direction_expands_thresholds_inside_one_group():
    X, y = _training_data(4)
    X = X.iloc[:3].copy()
    X = pd.concat(
        [
            X,
            pd.DataFrame(
                [
                    {
                        "patient_id": "P0003",
                        "time": pd.Timestamp("2020-01-01"),
                        "feature_name": "Birth Date Anchor",
                        "value": pd.Timestamp("2000-01-01"),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    model = _tree(1.0, learn_missing_direction=True)

    model.fit(X, y)

    assert model.generated_groups == 1
    assert model.sampled_groups == 1
    assert model.generated_threshold_candidates == 6
    assert model.invalid_threshold_candidates == 1
    assert model.valid_threshold_candidates == 5
    assert model.evaluated_threshold_candidates == 5


def test_invalid_threshold_is_not_scored_or_resampled(monkeypatch):
    class KeepGenerator:
        def __init__(self):
            self.draw_count = 0

        def random(self):
            self.draw_count += 1
            return 0.1

    generator = KeepGenerator()
    monkeypatch.setattr(np.random, "default_rng", lambda seed: generator)
    X, y = _training_data(4)
    model = _tree(0.5, min_samples_leaf=2)

    model.fit(X, y)

    assert generator.draw_count == 1
    assert model.generated_groups == model.sampled_groups == 1
    assert model.generated_threshold_candidates == 4
    assert model.invalid_threshold_candidates == 3
    assert model.valid_threshold_candidates == 1
    assert model.evaluated_threshold_candidates == 1
    assert model.fallback_evaluated_groups == 0


def test_old_counter_names_remain_read_only_compatibility_aliases():
    X, y = _training_data(4)
    model = _tree(1.0)

    model.fit(X, y)

    assert model.generated_candidate_opportunities == model.generated_groups
    assert model.sampled_candidates == model.sampled_groups
    assert model.sampling_skipped_candidates == model.sampling_skipped_groups
    assert model.invalid_sampled_candidates == model.invalid_threshold_candidates
    assert model.evaluated_candidates == (
        model.evaluated_threshold_candidates
        + model.fallback_evaluated_threshold_candidates
    )


def test_group_sampling_summary_is_printed_once(capsys):
    X, y = _training_data(10)
    model = _tree(1.0, verbose=1)

    model.fit(X, y)

    output = capsys.readouterr().out
    assert output.count(
        "=== Feature-Window-Aggregation Group Sampling ==="
    ) == 1
    assert "Group sampling probability (A): 1.000" in output
    assert "Generated groups: 1" in output
    assert "Sampled groups: 1" in output
    assert "Skipped groups: 0" in output
    assert "Observed group sampling rate: 100.00%" in output
    assert "Threshold candidates generated from sampled groups: 10" in output
    assert "Invalid threshold candidates from sampled groups: 1" in output
    assert "Evaluated threshold candidates from sampled groups: 9" in output
    assert "Fallback-evaluated groups: 0" in output
