import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from generate_temporal_benchmark import (
    FEEDING_STATES,
    FEEDING_TEMPLATES,
    FEEDING_TRAJECTORY_COUNTS,
    FEEDING_TRAJECTORY_OR,
    FEEDING_TRAJECTORY_PROBS,
    GROUP_ORDER,
    INCONSISTENT_FEEDING_TEMPLATES,
    LABEL_COLUMNS,
    LEAKAGE_FEATURES,
    LONGITUDINAL_COLUMNS,
    N_EXPERIMENT_3,
    REGULAR_BOUNDARY_SIGNATURES,
    PUBLICATION_V2_PROFILE,
    VALIDATED_V1_PROFILE,
    BenchmarkConfig,
    _publication_food_probability,
    artifact_paths,
    build_trajectory_spec,
    calibrate_fa_intercept,
    calculate_fa_probabilities,
    extract_observable_boundary_signatures,
    feeding_state_at,
    generate_benchmark,
    sample_feeding_groups,
    save_benchmark,
)


@pytest.fixture(scope="module")
def structural_result():
    config = BenchmarkConfig(n_patients=22, master_seed=101)
    groups = list(GROUP_ORDER) * 2
    return generate_benchmark(
        config,
        assigned_groups=groups,
        require_empirical_distribution=False,
    )


@pytest.fixture(scope="module")
def publication_v2_result():
    config = BenchmarkConfig(
        n_patients=220,
        master_seed=303,
        output_dir=Path("data/temporal_benchmark_v2"),
        generation_profile=PUBLICATION_V2_PROFILE,
    )
    return generate_benchmark(
        config,
        assigned_groups=list(GROUP_ORDER) * 20,
        require_empirical_distribution=False,
    )


def test_empirical_counts_and_probabilities_are_exactly_derived():
    assert sum(FEEDING_TRAJECTORY_COUNTS.values()) == N_EXPERIMENT_3
    assert set(FEEDING_TRAJECTORY_COUNTS) == set(FEEDING_TRAJECTORY_OR)
    assert math.isclose(
        sum(FEEDING_TRAJECTORY_PROBS.values()),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    for group in GROUP_ORDER:
        assert FEEDING_TRAJECTORY_PROBS[group] == (
            FEEDING_TRAJECTORY_COUNTS[group] / N_EXPERIMENT_3
        )


def test_empirical_categorical_sampling_is_deterministic_and_reasonable():
    first_rng = np.random.default_rng(42)
    second_rng = np.random.default_rng(42)
    first = sample_feeding_groups(20_000, first_rng)
    second = sample_feeding_groups(20_000, second_rng)

    np.testing.assert_array_equal(first, second)
    realized = pd.Series(first).value_counts(normalize=True)
    assert set(realized.index) == set(GROUP_ORDER)
    assert max(
        abs(realized[group] - FEEDING_TRAJECTORY_PROBS[group])
        for group in GROUP_ORDER
    ) < 0.015


def test_all_defined_trajectory_templates_match_approved_states():
    assert FEEDING_TEMPLATES == {
        "2.0": ("F", "F", "F", "F"),
        "2.1": ("M", "F", "F", "F"),
        "2.2": ("M", "M", "F", "F"),
        "2.3": ("M", "M", "M", "F"),
        "2.4": ("M", "M", "M", "M"),
        "3": ("B", "B", "B", "B"),
        "4": ("F", "B", "B", "B"),
        "5.1": ("B", "F", "F", "F"),
        "5.2": ("B", "B", "F", "F"),
        "5.3": ("B", "B", "B", "F"),
    }
    assert FEEDING_STATES == {
        "F": (1.0, 0.0),
        "B": (0.0, 1.0),
        "M": (1.0, 1.0),
    }


@pytest.mark.parametrize(
    ("group", "before_state", "after_state"),
    [("2.1", "M", "F"), ("4", "F", "B")],
)
def test_exact_21_day_trajectory_boundary(group, before_state, after_state):
    birth = pd.Timestamp("2026-01-10")
    spec = build_trajectory_spec(group, birth, np.random.default_rng(7))
    transition = birth + pd.Timedelta(days=21)

    assert spec["transition_times"] == (transition,)
    assert feeding_state_at(
        transition - pd.Timedelta(days=1),
        spec["initial_state"],
        spec["transition_times"],
        spec["transition_states"],
    ) == before_state
    assert feeding_state_at(
        transition,
        spec["initial_state"],
        spec["transition_times"],
        spec["transition_states"],
    ) == after_state


@pytest.mark.parametrize(
    ("group", "lower", "upper"),
    [
        ("2.2", "21d", 2),
        ("2.3", 2, 4),
        ("5.1", "21d", 2),
        ("5.2", 2, 4),
        ("5.3", 4, 6),
    ],
)
def test_sampled_transition_times_stay_in_approved_ranges(group, lower, upper):
    birth = pd.Timestamp("2026-01-10")
    rng = np.random.default_rng(123)
    for _ in range(100):
        transition = build_trajectory_spec(group, birth, rng)[
            "transition_times"
        ][0]
        lower_time = (
            birth + pd.Timedelta(days=21)
            if lower == "21d"
            else birth + pd.DateOffset(months=lower)
        )
        upper_time = birth + pd.DateOffset(months=upper)
        assert lower_time < transition <= upper_time


def test_group_6_templates_are_unique_inconsistent_and_nonduplicating():
    defined_templates = set(FEEDING_TEMPLATES.values())
    assert len(set(INCONSISTENT_FEEDING_TEMPLATES)) == len(
        INCONSISTENT_FEEDING_TEMPLATES
    )
    for template in INCONSISTENT_FEEDING_TEMPLATES:
        assert template not in defined_templates
        assert sum(a != b for a, b in zip(template, template[1:])) >= 2

    birth = pd.Timestamp("2026-01-10")
    for seed in range(20):
        spec = build_trajectory_spec("6", birth, np.random.default_rng(seed))
        assert tuple(spec["conceptual_states"]
                     ) in INCONSISTENT_FEEDING_TEMPLATES
        assert len(spec["transition_times"]) >= 2


def test_generation_is_deterministic_under_master_seed():
    config = BenchmarkConfig(n_patients=22, master_seed=2026)
    groups = list(GROUP_ORDER) * 2
    first = generate_benchmark(
        config,
        assigned_groups=groups,
        require_empirical_distribution=False,
    )
    second = generate_benchmark(
        config,
        assigned_groups=groups,
        require_empirical_distribution=False,
    )

    pd.testing.assert_frame_equal(first.data, second.data)
    pd.testing.assert_frame_equal(first.fa_labels, second.fa_labels)
    pd.testing.assert_frame_equal(first.ground_truth, second.ground_truth)
    assert first.metadata == second.metadata


def test_anchor_now_and_temporal_integrity(structural_result):
    data = structural_result.data
    ground_truth = structural_result.ground_truth
    for anchor_name in ("Birth Date Anchor", "NOW Anchor"):
        anchors = data.loc[data["feature_name"].eq(anchor_name)]
        assert anchors.groupby("patient_id").size().eq(1).all()
        assert anchors["value"].map(
            lambda value: isinstance(value, pd.Timestamp)).all()

    now_by_patient = ground_truth.set_index("patient_id")["now"]
    assert (data["time"] <= data["patient_id"].map(now_by_patient)).all()
    assert ground_truth["now_age_months"].between(18, 24).all()
    assert ground_truth["now_age_months"].nunique() > 1
    assert (
        ground_truth["birth_date"] + pd.DateOffset(months=6)
        < ground_truth["now"] - pd.DateOffset(months=6)
    ).all()


def test_every_patient_has_exactly_one_private_feeding_group(
    structural_result,
):
    ground_truth = structural_result.ground_truth

    assert len(ground_truth) == ground_truth["patient_id"].nunique()
    assert ground_truth["feeding_group"].notna().all()
    assert ground_truth.groupby("patient_id")["feeding_group"].nunique().eq(
        1
    ).all()
    assert len(ground_truth) == sum(
        ground_truth["feeding_group"].value_counts()
    )


def test_exact_boundaries_make_regular_groups_observably_exclusive(
    structural_result,
):
    ground_truth = structural_result.ground_truth
    signatures = extract_observable_boundary_signatures(
        structural_result.data,
        ground_truth,
    )

    assert set(signatures) == set(ground_truth["patient_id"])
    for row in ground_truth.itertuples(index=False):
        signature = signatures[row.patient_id]
        matching_regular_groups = [
            group
            for group, expected in REGULAR_BOUNDARY_SIGNATURES.items()
            if signature == expected
        ]
        if row.feeding_group == "6":
            assert matching_regular_groups == []
        else:
            assert matching_regular_groups == [row.feeding_group]


def test_exclusivity_validation_checks_pass(structural_result):
    checks = structural_result.metadata["validation"]["checks"]
    expected_checks = {
        "one_private_ground_truth_row_per_patient",
        "exactly_one_private_feeding_group_per_patient",
        "no_multiple_trajectory_assignments",
        "feeding_groups_cover_full_cohort",
        "exact_feeding_boundary_observations_present",
        "no_observable_regular_group_overlap",
        "observable_history_matches_single_assigned_group",
        "group_6_exclusive_from_regular_groups",
    }

    assert expected_checks.issubset(checks)
    assert all(checks[check] for check in expected_checks)


def test_fa_calibration_beta_conversion_and_bernoulli_output(
    structural_result,
):
    groups = structural_result.ground_truth["feeding_group"]
    intercept = calibrate_fa_intercept(groups, 0.10)
    probabilities = calculate_fa_probabilities(groups, intercept)

    assert math.isclose(probabilities.mean(), 0.10, abs_tol=1e-10)
    expected_betas = groups.map(
        {group: math.log(value)
         for group, value in FEEDING_TRAJECTORY_OR.items()}
    )
    np.testing.assert_allclose(
        structural_result.ground_truth["beta_j"],
        expected_betas,
    )
    assert structural_result.fa_labels["label"].isin([0, 1]).all()


def test_output_schema_no_leakage_and_metadata_completeness(structural_result):
    assert list(structural_result.data.columns) == LONGITUDINAL_COLUMNS
    assert list(structural_result.fa_labels.columns) == LABEL_COLUMNS
    normalized_features = {
        feature.replace(" ", "").replace("_", "").lower()
        for feature in structural_result.data["feature_name"].unique()
    }
    normalized_leakage = {
        feature.replace(" ", "").replace("_", "").lower()
        for feature in LEAKAGE_FEATURES
    }
    assert normalized_features.isdisjoint(normalized_leakage)

    metadata = structural_result.metadata
    assert metadata["gi_label_status"] == "not_implemented"
    assert metadata["feeding"]["empirical_counts"] == FEEDING_TRAJECTORY_COUNTS
    assert metadata["feeding"]["sampling_probabilities"] == (
        FEEDING_TRAJECTORY_PROBS
    )
    assert metadata["fa"]["target_prevalence"] == 0.10
    assert set(metadata["validation"]["observation_density"]) == {
        "birth_0_to_6m",
        "now_minus_6m_to_now",
        "now_minus_1m_to_now",
    }
    assert all(
        metadata["validation"]["checks"][check]
        for check in (
            "feeding_observations_match_ground_truth",
            "transition_ranges_valid",
            "no_leakage_features",
        )
    )


def test_artifacts_use_dedicated_directory_and_refuse_overwrite(
    structural_result,
    tmp_path,
):
    output_dir = tmp_path / "data" / "temporal_benchmark"
    config = BenchmarkConfig(
        n_patients=22,
        master_seed=101,
        output_dir=output_dir,
    )
    paths = artifact_paths(config)

    saved_paths = save_benchmark(structural_result, config)

    assert saved_paths == paths
    assert all(path.parent == output_dir for path in paths.values())
    assert all(path.is_file() for path in paths.values())
    assert json.loads(paths["metadata"].read_text(encoding="utf-8")) == (
        structural_result.metadata
    )
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        save_benchmark(structural_result, config)


def test_validated_v1_remains_the_default_profile(structural_result):
    assert BenchmarkConfig().generation_profile == VALIDATED_V1_PROFILE
    assert "generation_profile" not in structural_result.metadata
    assert set(structural_result.metadata["rng_streams"]) == {
        "cohort",
        "trajectory",
        "feeding_observations",
        "distractors",
        "missingness",
        "fa_labels",
    }


def test_publication_v2_now_is_day_level_and_strictly_separated(
    publication_v2_result,
):
    ground_truth = publication_v2_result.ground_truth
    earliest = ground_truth["birth_date"] + pd.DateOffset(months=18)
    latest = ground_truth["birth_date"] + pd.DateOffset(months=24)

    assert ground_truth["now"].between(earliest, latest).all()
    assert ground_truth["now_age_days"].nunique() >= 100
    assert (
        ground_truth["birth_date"] + pd.DateOffset(months=6)
        < ground_truth["now"] - pd.DateOffset(months=6)
    ).all()
    assert publication_v2_result.metadata["validation"]["checks"][
        "day_level_now_distribution"
    ]


def test_publication_v2_food_probability_increases_with_age():
    probabilities = [
        _publication_food_probability(age, 180.0, 0.90)
        for age in (60, 120, 180, 270, 450)
    ]

    assert probabilities == sorted(probabilities)
    assert probabilities[0] < 0.01
    assert probabilities[-1] > 0.89


def test_publication_v2_growth_is_routine_and_longitudinally_coherent(
    publication_v2_result,
):
    audit = publication_v2_result.metadata["validation"][
        "publication_v2_covariate_audit"
    ]["growth"]

    for feature in ("LengthCm", "WeightKg"):
        feature_audit = audit[feature]
        assert 7 <= feature_audit["observations_per_patient"]["min"]
        assert feature_audit["observations_per_patient"]["max"] <= 9
        assert feature_audit["measurement_gap_days"]["min"] >= 20
        assert feature_audit["observed_decrease_rate"] <= 0.01


def test_publication_v2_uses_separate_food_growth_and_label_rng_streams(
    publication_v2_result,
):
    streams = publication_v2_result.metadata["rng_streams"]

    assert streams["fa_labels"]["spawn_key"] == [5]
    assert streams["food_exposure"]["spawn_key"] == [7]
    assert streams["growth"]["spawn_key"] == [8]
