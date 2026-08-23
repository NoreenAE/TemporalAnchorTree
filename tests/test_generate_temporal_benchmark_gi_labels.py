import inspect
import json
import math

import numpy as np
import pandas as pd
import pytest

from generate_temporal_benchmark_gi_labels import (
    GI_EXCLUDED_DISTRACTORS,
    GI_OR,
    GI_RNG_SPAWN_KEY,
    GIConfig,
    calculate_gi_probabilities,
    calibrate_gi_intercept,
    extract_gi_predictors,
    generate_gi_benchmark,
    gi_artifact_paths,
    make_gi_rng,
    sample_gi_labels,
    save_gi_benchmark,
    sha256_file,
)


@pytest.fixture
def longitudinal_x():
    records = []
    patterns = [
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 1),
        (1, 1, 1),
    ]
    for index, pattern in enumerate(patterns):
        patient_id = f"P{index}"
        birth = pd.Timestamp("2020-01-01") + pd.DateOffset(days=index)
        now = pd.Timestamp("2021-03-31") + pd.DateOffset(days=index)
        window_start = now - pd.DateOffset(months=1)
        records.extend(
            [
                (patient_id, birth, "Birth Date Anchor", birth),
                (patient_id, now, "NOW Anchor", now),
            ]
        )
        for feature, value in zip(GI_OR, pattern):
            records.extend(
                [
                    (
                        patient_id,
                        window_start - pd.Timedelta(days=1),
                        feature,
                        1 - value,
                    ),
                    (patient_id, window_start, feature, value),
                    (patient_id, now, feature, 0),
                ]
            )
        records.extend(
            [
                (patient_id, window_start, "FoodExposureYN", 1),
                (patient_id, window_start, "ClinicalVisitYN", 1),
            ]
        )
    return pd.DataFrame(
        records,
        columns=["patient_id", "time", "feature_name", "value"],
    )


@pytest.fixture
def gi_result(longitudinal_x):
    return generate_gi_benchmark(
        longitudinal_x,
        GIConfig(n_patients=4, master_seed=123),
    )


def test_extracts_exact_three_predictors_with_calendar_month_boundary(
    longitudinal_x,
):
    predictors = extract_gi_predictors(longitudinal_x)

    assert list(predictors.columns) == [
        "patient_id",
        "V_i",
        "D_i",
        "G_i",
        "n_active_predictors",
    ]
    assert predictors[["V_i", "D_i", "G_i"]].to_numpy().tolist() == [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 1],
        [1, 1, 1],
    ]


def test_observations_before_final_calendar_month_do_not_contribute(
    longitudinal_x,
):
    original = extract_gi_predictors(longitudinal_x)
    changed = longitudinal_x.copy(deep=True)
    now = changed.loc[
        changed["feature_name"].eq("NOW Anchor")
    ].set_index("patient_id")["value"]
    outside = changed["time"].lt(
        changed["patient_id"].map(now) - pd.DateOffset(months=1)
    ) & changed["feature_name"].isin(GI_OR)
    changed.loc[outside, "value"] = 99

    pd.testing.assert_frame_equal(original, extract_gi_predictors(changed))


@pytest.mark.parametrize("feature", GI_EXCLUDED_DISTRACTORS)
def test_excluded_distractors_cannot_affect_gi_probability(
    longitudinal_x,
    feature,
):
    config = GIConfig(n_patients=4, master_seed=123)
    original = generate_gi_benchmark(longitudinal_x, config)
    changed = longitudinal_x.copy(deep=True)
    changed.loc[changed["feature_name"].eq(feature), "value"] = 999
    regenerated = generate_gi_benchmark(changed, config)

    pd.testing.assert_frame_equal(original.labels, regenerated.labels)
    pd.testing.assert_frame_equal(
        original.ground_truth,
        regenerated.ground_truth,
    )


def test_coefficients_are_exact_logs_of_approved_synthetic_ors(gi_result):
    metadata = gi_result.metadata

    assert set(metadata["predictors"]) == set(GI_OR)
    for feature, odds_ratio in GI_OR.items():
        assert metadata["predictors"][feature]["coefficient"] == math.log(
            odds_ratio
        )


def test_intercept_calibration_matches_target(longitudinal_x):
    predictors = extract_gi_predictors(longitudinal_x)
    intercept = calibrate_gi_intercept(predictors, 0.12)
    probabilities = calculate_gi_probabilities(predictors, intercept)

    assert math.isclose(probabilities.mean(), 0.12, abs_tol=1e-10)


def test_bernoulli_labels_use_dedicated_deterministic_rng(gi_result):
    probabilities = gi_result.ground_truth["gi_probability"]
    first = sample_gi_labels(probabilities, 123)
    second = sample_gi_labels(probabilities, 123)
    fa_stream = np.random.default_rng(
        np.random.SeedSequence(123, spawn_key=(5,))
    ).binomial(1, probabilities)

    np.testing.assert_array_equal(first, second)
    assert GI_RNG_SPAWN_KEY != (5,)
    assert make_gi_rng(123).bit_generator.__class__.__name__ == "PCG64"
    assert first.dtype.kind in "iu"
    assert set(first).issubset({0, 1})
    assert not np.array_equal(
        make_gi_rng(123).random(20),
        np.random.default_rng(
            np.random.SeedSequence(123, spawn_key=(5,))
        ).random(20),
    )
    assert fa_stream.shape == first.shape


def test_generation_has_no_fa_or_feeding_ground_truth_input(gi_result):
    parameters = inspect.signature(generate_gi_benchmark).parameters

    assert set(parameters) == {"data", "config"}
    assert all(
        not value
        for value in gi_result.metadata[
            "external_ground_truth_inputs"
        ].values()
    )


def test_generation_does_not_mutate_longitudinal_x(longitudinal_x):
    original = longitudinal_x.copy(deep=True)

    generate_gi_benchmark(
        longitudinal_x,
        GIConfig(n_patients=4, master_seed=123),
    )

    pd.testing.assert_frame_equal(longitudinal_x, original)


def test_gi_schema_and_metadata_are_complete(gi_result):
    assert list(gi_result.labels.columns) == ["patient_id", "label"]
    assert gi_result.labels["label"].isin([0, 1]).all()
    assert list(gi_result.ground_truth.columns) == [
        "patient_id",
        "V_i",
        "D_i",
        "G_i",
        "n_active_predictors",
        "gi_probability",
        "gi_label",
    ]
    assert gi_result.metadata["true_temporal_window"] == {
        "anchor": "NOW Anchor",
        "start_offset": "-1 calendar month",
        "end_offset": "0 months",
        "bounds": "inclusive",
    }
    assert gi_result.metadata["excluded_model_visible_distractors"] == [
        "FoodExposureYN",
        "ClinicalVisitYN",
    ]
    assert all(gi_result.metadata["validation"]["checks"].values())


def test_saving_gi_artifacts_leaves_source_files_unchanged(
    longitudinal_x,
    gi_result,
    tmp_path,
):
    source_x = tmp_path / "source_x.pkl"
    source_fa = tmp_path / "source_fa.pkl"
    longitudinal_x.to_pickle(source_x)
    pd.DataFrame(
        {"patient_id": ["P0", "P1", "P2", "P3"], "label": [0, 1, 0, 1]}
    ).to_pickle(source_fa)
    hashes_before = (sha256_file(source_x), sha256_file(source_fa))
    config = GIConfig(
        n_patients=4,
        master_seed=123,
        output_dir=tmp_path / "gi",
    )

    paths = save_gi_benchmark(gi_result, config)

    assert hashes_before == (sha256_file(source_x), sha256_file(source_fa))
    assert paths == gi_artifact_paths(config)
    assert all(path.is_file() for path in paths.values())
    assert json.loads(paths["metadata"].read_text()) == gi_result.metadata
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        save_gi_benchmark(gi_result, config)
