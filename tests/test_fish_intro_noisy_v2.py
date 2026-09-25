"""Fast in-memory generator/loader checks. No fitting, prediction, or figures."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from generate_fish_intro_noisy_v2 import (
    BIRTH, DISTRACTORS, EVENT_FEATURES, FEATURE_ORDER, FISH, SYMPTOMS,
    GeneratorConfig, descriptive_audit, distractor_history, generate_data,
    qualifying_counts, recompute_r, save_dataset, validate_dataset,
)
from experiments.experiment_2_noisy_v2 import load_dataset
from experiments.experiment_2_fish_introduction_dynamic_anchors import make_model


@pytest.fixture(scope="module")
def generated():
    return generate_data(GeneratorConfig(blocks=20))


def test_truth_all_events_and_matching(generated):
    data, patients, validation = generated
    assert validation["passed"]
    pd.testing.assert_series_equal(recompute_r(data, patients), patients.R)
    # Independent scalar oracle, including ALL symptoms, not a designated one.
    records = {pid: rows for pid, rows in data.groupby("patient_id")}
    for child in patients.itertuples(index=False):
        rows = records[child.patient_id]
        fish = rows.loc[rows.feature_name.eq(FISH), "time"]
        assert len(fish) == int(child.documented)
        symptom = rows.loc[rows.feature_name.isin(SYMPTOMS), "time"]
        count = 0 if fish.empty else sum(
            fish.iloc[0] < t <= fish.iloc[0] + pd.DateOffset(months=1) for t in symptom)
        assert child.R == int(count > 0)
    assert patients.groupby("block").R.sum().eq(4).all()
    assert patients.groupby("block").documented.sum().eq(8).all()
    assert set(data.feature_name) == {BIRTH, *EVENT_FEATURES}
    assert tuple(data.feature_name.unique()) == FEATURE_ORDER
    assert not set(patients.columns).difference({"patient_id"}).intersection(
        set(data.columns).difference({"time", "value", "feature_name"}))


def test_calendar_boundaries_and_extra_background_changes_truth():
    fish = pd.Timestamp("2020-01-31 12:00:00")
    end = fish + pd.DateOffset(months=1)
    history = [("GI", fish), ("Vomiting", end),
               ("GI", end + pd.Timedelta(seconds=1))]
    assert qualifying_counts(history, [fish]).tolist() == [1]
    patients = pd.DataFrame({"patient_id": ["a", "b"]})
    data = pd.DataFrame([("a", fish, FISH, 1), ("a", fish, "GI", 1),
                         ("b", end, "GI", 1)],
                        columns=["patient_id", "time", "feature_name", "value"])
    assert recompute_r(data, patients).tolist() == [0, 0]
    data.loc[len(data)] = ["a", end, "Vomiting", 1]
    assert recompute_r(data, patients).tolist() == [1, 0]


def test_background_inside_windows_and_no_forced_labels(generated):
    data, patients, _ = generated
    assert "signal_time" not in patients  # all symptoms are unrestricted background
    counts = []
    for pid, rows in data.groupby("patient_id"):
        child = patients.set_index("patient_id").loc[pid]
        history = list(zip(rows.feature_name, rows.time))
        candidates = qualifying_counts(history, [child.a, child.b])
        assert candidates.max() >= 1
        assert candidates.min() == 0
        counts.append(candidates.max())
    assert max(counts) > 1
    altered = patients.copy()
    altered.loc[altered.R.eq(1).idxmax(), "R"] = 0
    with pytest.raises(ValueError, match="R disagrees"):
        validate_dataset(data, altered)


def test_label_stream_and_reproducibility(generated):
    data, patients, _ = generated
    config = GeneratorConfig(blocks=20)
    labels = np.random.default_rng(np.random.SeedSequence(config.seed).spawn(6)[5])
    np.testing.assert_array_equal(patients.FA, labels.binomial(1, patients.fa_probability))
    np.testing.assert_array_equal(patients.fa_probability, np.where(patients.R.eq(1), .9, .05))
    repeated = generate_data(config)
    pd.testing.assert_frame_equal(data, repeated[0])
    pd.testing.assert_frame_equal(patients, repeated[1])


def test_distractor_mechanisms_and_no_target_inputs():
    import inspect
    assert set(inspect.signature(distractor_history).parameters) == {"rng", "birth", "end", "config"}
    config = GeneratorConfig()
    birth = pd.Timestamp("2010-01-01")
    end = birth + pd.DateOffset(months=30)
    rng = np.random.default_rng(183)
    samples = [distractor_history(rng, birth, end, config) for _ in range(400)]
    means = {f: np.mean([sum(feature == f for feature, _ in sample) for sample in samples])
             for f in DISTRACTORS}
    assert means["MinorInjury"] < means["DoctorVisit"] < means["Cough"]
    assert means["MinorInjury"] < means["SleepDisturbance"]
    assert 4 < means["RoutineVaccination"] < 6
    milestones = [birth + pd.DateOffset(months=m) for m in config.vaccination_months]
    for sample in samples:
        for feature, time in sample:
            assert birth <= time <= end
            if feature == "RoutineVaccination":
                assert min(abs(time-m) for m in milestones) <= pd.Timedelta(days=10)
    # Changing label/matching-only assumptions cannot alter a distractor draw.
    same = replace(config, gi_mean=99, candidate_times=10)
    assert distractor_history(np.random.default_rng(5), birth, end, config) == \
        distractor_history(np.random.default_rng(5), birth, end, same)


def test_audit_and_roundtrip_same_split_for_both_runs(generated, tmp_path):
    data, patients, validation = generated
    audit = descriptive_audit(data, patients)
    for cohort in audit["cohorts"].values():
        assert cohort["fixed_age_single_score_R_auc_range"] == pytest.approx([.5, .5])
        assert list(cohort["count_age_density_R_auc"].values()) == pytest.approx(
            [.5] * len(cohort["count_age_density_R_auc"]))
        joint = cohort["joint_fixed_age_signatures"]["nonfish"]
        assert joint["children_with_opposite_R_signature"] == joint["denominator"]
    output = tmp_path / "new_dataset"
    save_dataset(output, data, patients, validation, audit)
    on_data, on_patients, _ = load_dataset(output)
    off_data, off_patients, _ = load_dataset(output)
    pd.testing.assert_frame_equal(on_data, off_data)
    pd.testing.assert_frame_equal(on_patients, off_patients)
    assert on_patients.groupby("block").partition.nunique().eq(1).all()
    assert len(on_patients.loc[on_patients.partition.eq("test")]) == 40
    on, off = make_model(True).get_params(), make_model(False).get_params()
    assert on.pop("use_dynamic_anchors") and not off.pop("use_dynamic_anchors")
    assert on == off
    assert set(on["feature_semantics"]) == set(EVENT_FEATURES)
    with pytest.raises(FileExistsError):
        save_dataset(output, data, patients, validation, audit)
    # Hash checking detects accidental metadata changes before any model runs.
    with (output / "patients_and_split.csv").open("a") as file:
        file.write("\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_dataset(output)


def test_validator_rejects_block_leakage_and_metadata_predictors(generated):
    data, patients, _ = generated
    changed = patients.copy()
    changed.loc[0, "partition"] = "test" if changed.loc[0, "partition"] == "train" else "train"
    with pytest.raises(ValueError, match="split leakage"):
        validate_dataset(data, changed)
    with pytest.raises(ValueError, match="four observation columns"):
        validate_dataset(data.assign(R=0), patients)
