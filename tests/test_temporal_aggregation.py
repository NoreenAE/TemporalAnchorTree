import numpy as np
import pandas as pd

from TAT.temporal_tree import TemporalTree


def _aggregation_caches():
    return {
        "anchor_reference_cache": {},
        "feature_subset_cache": {},
        "feature_anchor_merge_cache": {},
        "time_window_mask_cache": {},
        "filtered_numeric_cache": {},
        "aggregation_result_cache": {},
    }


def _missing_value_frame():
    rows = [
        {
            "patient_id": "absent",
            "time": pd.Timestamp("2020-01-01"),
            "feature_name": "other",
            "value": 10.0,
        },
        {
            "patient_id": "mixed",
            "time": pd.Timestamp("2020-01-01"),
            "feature_name": "signal",
            "value": 3.0,
        },
        {
            "patient_id": "mixed",
            "time": pd.Timestamp("2020-01-02"),
            "feature_name": "signal",
            "value": np.nan,
        },
        {
            "patient_id": "all_nan",
            "time": pd.Timestamp("2020-01-01"),
            "feature_name": "signal",
            "value": np.nan,
        },
        {
            "patient_id": "valid",
            "time": pd.Timestamp("2020-01-01"),
            "feature_name": "signal",
            "value": 1.0,
        },
        {
            "patient_id": "valid",
            "time": pd.Timestamp("2020-01-02"),
            "feature_name": "signal",
            "value": 2.0,
        },
        {
            "patient_id": "single",
            "time": pd.Timestamp("2020-01-01"),
            "feature_name": "signal",
            "value": 7.0,
        },
    ]
    return pd.DataFrame(rows)


def test_cached_aggregations_preserve_missing_values_and_patient_order():
    X = _missing_value_frame()
    model = TemporalTree(verbose=0)
    caches = _aggregation_caches()
    expected = {
        "mean": [np.nan, 3.0, np.nan, 1.5, 7.0],
        "sum": [np.nan, 3.0, np.nan, 3.0, 7.0],
        "max": [np.nan, 3.0, np.nan, 2.0, 7.0],
        "count": [np.nan, 1.0, 0.0, 2.0, 1.0],
    }

    for aggregation, expected_values in expected.items():
        values, resolved_anchor = model._aggregate_feature_over_time(
            X,
            "signal",
            (None, None),
            aggregation,
            **caches,
        )

        assert resolved_anchor is None
        np.testing.assert_array_equal(values, expected_values)
        assert len(caches["aggregation_result_cache"]) == 1

    cached_result = next(iter(caches["aggregation_result_cache"].values()))
    assert list(cached_result.columns) == ["mean", "sum", "max", "count"]


def test_empty_integer_window_preserves_cached_sum_dtype():
    X = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "time": pd.Timestamp("2019-01-01"),
                "feature_name": "signal",
                "value": 1,
            },
            {
                "patient_id": "P1",
                "time": pd.NaT,
                "feature_name": "Birth Date Anchor",
                "value": pd.Timestamp("2020-01-01"),
            },
        ]
    )
    model = TemporalTree(verbose=0)
    caches = _aggregation_caches()

    values, _ = model._aggregate_feature_over_time(
        X,
        "signal",
        ("Birth Date Anchor", 1),
        "sum",
        **caches,
    )

    cached_result = next(iter(caches["aggregation_result_cache"].values()))
    assert cached_result["sum"].dtype == np.dtype("float64")
    assert np.isnan(values[0])


def test_custom_aggregation_path_is_unchanged():
    X = _missing_value_frame()
    model = TemporalTree(verbose=0)

    values, _ = model._aggregate_feature_over_time(
        X,
        "signal",
        (None, None),
        "median",
    )

    np.testing.assert_array_equal(
        values,
        [np.nan, 3.0, np.nan, 1.5, 7.0],
    )
