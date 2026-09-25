"""Focused checks; generators run in memory only, with no model training."""
import numpy as np
import pandas as pd
import pytest

from experiments.experiment_2_fish_introduction_dynamic_anchors import (
    load_dataset, make_model,
)
from generate_fish_intro_dataset import (
    BIRTH, DEFAULT_DATA_DIR, DISTRACTORS, FEATURE_ORDER, FISH, ROOT, SYMPTOMS,
    generate_data, validate_observations,
)


@pytest.fixture(scope="module")
def generated():
    return generate_data()


def test_observed_truth_matching_and_independent_labels(generated):
    data, patients, validation = generated
    assert validation["passed"]
    assert patients.groupby("block").R.sum().eq(4).all()
    assert patients.groupby("block").documented.sum().eq(8).all()
    assert patients.groupby("block").partition.nunique().eq(1).all()
    assert patients.partition.value_counts().to_dict() == {"train": 4000, "test": 1000}
    assert patients.scenario.value_counts().to_dict() == {
        "ordinary": 4000, "endpoint": 500, "boundary": 500}
    assert patients.fa_probability.eq(np.where(patients.R.eq(1), .9, .05)).all()
    # FA must be a separate Bernoulli draw, not a copy of R or a quota.
    assert (patients.FA != patients.R).any()
    assert data.feature_name.eq(BIRTH).sum() == 5000
    assert data.feature_name.eq(FISH).sum() == 4000
    repeated = generate_data()
    pd.testing.assert_frame_equal(data, repeated[0])
    pd.testing.assert_frame_equal(patients, repeated[1])


def test_background_in_either_candidate_window_rejected(generated):
    data, patients, _ = generated
    child = patients.iloc[0]
    # The later window does not change this child's positive R, but must still
    # be forbidden to preserve every pairing's background exclusion.
    extra = pd.DataFrame([dict(patient_id=child.patient_id, feature_name="GI",
                               time=child.b + pd.Timedelta(seconds=2), value=1)])
    with pytest.raises(ValueError, match="background"):
        validate_observations(pd.concat([data, extra], ignore_index=True), patients)


def test_primary_comparison_changes_only_switch():
    on, off = make_model(True).get_params(), make_model(False).get_params()
    assert on.pop("use_dynamic_anchors") is True
    assert off.pop("use_dynamic_anchors") is False
    assert on == off
    assert on["anchors_list"] == [BIRTH]
    assert on["anchor_offset_map"] == {BIRTH: [*range(1, 23), None]}
    assert on["anchor_offset_list"] == [1, 2, 3, 4]
    assert on["candidate_sampling_rate"] == 0.4
    assert on["feature_semantics"] == {
        feature: "binary_event_indicator"
        for feature in (FISH, *SYMPTOMS, *DISTRACTORS)
    }
    assert on["max_depth"] == 3
    assert set(on["aggregation_funcs"]) == {"sum"}
    assert on["criterion"] == "mse"
    assert not on["is_classification"]


@pytest.mark.parametrize("feature", (FISH, *SYMPTOMS, *DISTRACTORS))
def test_every_event_feature_has_positive_event_anchor_policy(feature):
    from types import SimpleNamespace

    node = SimpleNamespace(feature=feature, aggregation="sum")
    assert make_model(True)._dynamic_anchor_selection(node) == "positive_events"


def test_revised_feature_schema_and_event_presence(generated):
    data, patients, validation = generated
    assert set(data.columns) == {"patient_id", "time", "feature_name", "value"}
    expected = {FISH, *SYMPTOMS, *DISTRACTORS}
    assert len(expected) == 8
    assert {"MinorInjury", "RoutineVaccination"}.issubset(expected)
    assert {"Fever", "Rash"}.isdisjoint(expected)
    assert set(data.feature_name) == expected | {BIRTH}
    fish_counts = data.loc[data.feature_name.eq(FISH)].groupby("patient_id").size()
    actual = fish_counts.reindex(patients.patient_id, fill_value=0).to_numpy()
    np.testing.assert_array_equal(actual, patients.documented.astype(int))
    assert (~patients.documented).sum() == 1000
    assert validation["distractors"]["poisson_mean"] == 6
    assert validation["distractors"]["features"] == list(DISTRACTORS)
    assert DEFAULT_DATA_DIR.name == "fish_intro_dataset_distractors_v2"
    assert tuple(data.feature_name.unique()) == FEATURE_ORDER


def test_rename_matches_historical_numerical_snapshot(generated):
    """Compare every row, label, and partition; never overwrite the snapshot."""
    historical = ROOT / "data/fish_intro_dataset_distractors"
    if not (historical / "observations.csv").exists():
        pytest.skip("Historical local snapshot is not distributed with the code")
    data, patients, _ = generated
    old = pd.read_csv(historical / "observations.csv", parse_dates=["time"],
                      dtype={"patient_id": str, "value": object})
    birth = old.feature_name.eq(BIRTH)
    old.loc[birth, "value"] = pd.to_datetime(old.loc[birth, "value"])
    old.loc[~birth, "value"] = pd.to_numeric(old.loc[~birth, "value"])
    old["feature_name"] = old.feature_name.replace(
        {"Fever": "MinorInjury", "Rash": "RoutineVaccination"})
    # Includes row/feature order: no new sampling or tie-breaking order.
    pd.testing.assert_frame_equal(data, old)
    old_patients = pd.read_csv(
        historical / "patients_and_split.csv",
        parse_dates=["birth", "followup_end", "a", "b", "signal_time", "fish_time"],
        dtype={"patient_id": str, "documented": bool},
    )
    pd.testing.assert_frame_equal(patients, old_patients)


def test_historical_names_rejected_by_revised_runner():
    historical = ROOT / "data/fish_intro_dataset_distractors"
    if not (historical / "observations.csv").exists():
        pytest.skip("Historical local snapshot is not distributed with the code")
    with pytest.raises(ValueError, match="revised Experiment 2 feature names"):
        load_dataset(historical)
