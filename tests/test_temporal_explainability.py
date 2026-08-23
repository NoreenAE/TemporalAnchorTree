from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from settree.temporal_explainability import (
    PATH_TABLE_COLUMNS,
    aggregate_temporal_usage,
    derive_monthly_bin_edges,
    run_temporal_explainability,
    trace_temporal_decision_paths,
)
from settree.temporal_tree import TemporalLeaf, TemporalSplitNode, TemporalTree
from settree.temporal_tree_ensemble import TemporalTreeEnsemble


BIN_EDGES = {
    "Birth Date Anchor": list(range(0, 7)),
    "NOW Anchor": list(range(-6, 1)),
}


def _leaf(value, leaf_id, depth=2):
    leaf = TemporalLeaf(value=value, num_patients=1, depth=depth)
    leaf.sn = leaf_id
    return leaf


def _fitted_manual_tree():
    full_timeline_node = TemporalSplitNode(
        feature="other",
        threshold=0.5,
        aggregation="mean",
        time_window=(None, None),
        missing_direction="right",
        depth=1,
        left=_leaf(0.1, 0),
        right=_leaf(0.2, 1),
    )
    zero_width_node = TemporalSplitNode(
        feature="recent",
        threshold=0.5,
        aggregation="mean",
        time_window=("NOW Anchor", 0),
        missing_direction="left",
        depth=1,
        left=_leaf(0.8, 2),
        right=_leaf(0.9, 3),
    )
    root = TemporalSplitNode(
        feature="signal",
        threshold=0.5,
        aggregation="mean",
        time_window=("Birth Date Anchor", 2),
        missing_direction="right",
        depth=0,
        left=full_timeline_node,
        right=zero_width_node,
    )
    model = TemporalTree(
        is_classification=False,
        use_dynamic_anchors=False,
        verbose=0,
    )
    model.tree = root
    return model


def _patient_data():
    birth_date = pd.Timestamp("2020-01-01")
    now = pd.Timestamp("2021-01-01")
    return pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "time": birth_date,
                "feature_name": "Birth Date Anchor",
                "value": birth_date,
            },
            {
                "patient_id": "P1",
                "time": now,
                "feature_name": "NOW Anchor",
                "value": now,
            },
            {
                "patient_id": "P1",
                "time": birth_date + pd.DateOffset(months=1),
                "feature_name": "signal",
                "value": 0.1,
            },
            {
                "patient_id": "P1",
                "time": birth_date + pd.DateOffset(months=3),
                "feature_name": "other",
                "value": 0.1,
            },
            {
                "patient_id": "P2",
                "time": birth_date,
                "feature_name": "Birth Date Anchor",
                "value": birth_date,
            },
            {
                "patient_id": "P2",
                "time": now,
                "feature_name": "NOW Anchor",
                "value": now,
            },
        ]
    )


def test_derive_monthly_bins_covers_complete_configured_ranges():
    edges = derive_monthly_bin_edges(
        {
            "Birth Date Anchor": [1, 2, 3, 4, 5, 6],
            "NOW Anchor": [-1, -2, -3, -4, -5, -6],
        }
    )

    assert edges == BIN_EDGES


def test_trace_matches_apply_and_records_missing_full_and_zero_width_visits():
    model = _fitted_manual_tree()
    X = _patient_data()

    path_table = trace_temporal_decision_paths(model, X)
    applied_leaf_ids = dict(zip(X["patient_id"].unique(), model.apply(X)))
    traced_leaf_ids = path_table.groupby("patient_id", sort=False)[
        "terminal_leaf_id"
    ].first().to_dict()

    assert list(path_table.columns) == PATH_TABLE_COLUMNS
    assert len(path_table) == 4
    assert traced_leaf_ids == applied_leaf_ids
    assert path_table.groupby("patient_id").size().to_dict() == {"P1": 2, "P2": 2}
    assert int(path_table["missing"].sum()) == 2
    assert int(path_table["is_full_timeline"].sum()) == 1
    assert int(path_table["is_zero_width"].sum()) == 1
    assert set(path_table["evaluated_patient_count"]) == {2}
    assert set(path_table["raw_usage_weight"]) == {1}
    assert set(path_table["per_patient_usage_weight"]) == {0.5}


def test_usage_bins_count_overlapping_and_repeated_path_rows_additively():
    path_table = trace_temporal_decision_paths(
        _fitted_manual_tree(),
        _patient_data(),
    )
    repeated_row = path_table.iloc[[0]].copy()
    repeated_row["node_id"] = 99
    repeated_row["stored_offset"] = 1
    repeated_row["effective_end_offset"] = 1
    path_table = pd.concat([path_table, repeated_row], ignore_index=True)

    usage_table, summary = aggregate_temporal_usage(
        path_table,
        BIN_EDGES,
        evaluated_patient_count=2,
    )

    birth_usage = usage_table.loc[
        usage_table["anchor"] == "Birth Date Anchor"
    ]
    now_usage = usage_table.loc[usage_table["anchor"] == "NOW Anchor"]
    assert birth_usage["raw_usage_count"].tolist() == [3, 2, 0, 0, 0, 0]
    assert birth_usage["normalized_usage_per_patient"].tolist() == [
        1.5,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    assert now_usage["raw_usage_count"].tolist() == [0, 0, 0, 0, 0, 0]
    assert summary["full_timeline_usages"] == 1
    assert summary["zero_width_usages"] == 1
    assert summary["temporal_split_usages"] == 5


def test_run_saves_exactly_the_path_csv_and_histogram(tmp_path):
    result = run_temporal_explainability(
        _fitted_manual_tree(),
        _patient_data(),
        tmp_path,
        BIN_EDGES,
    )

    assert result["path_table_path"] == tmp_path / "temporal_path_usage.csv"
    assert result["histogram_path"] == tmp_path / "temporal_usage_histogram.png"
    assert result["path_table_path"].is_file()
    assert result["histogram_path"].is_file()
    assert result["histogram_path"].stat().st_size > 0
    assert {path.name for path in Path(tmp_path).iterdir()} == {
        "temporal_path_usage.csv",
        "temporal_usage_histogram.png",
    }
    saved_table = pd.read_csv(result["path_table_path"])
    assert list(saved_table.columns) == PATH_TABLE_COLUMNS
    assert len(saved_table) == 4


def test_run_rejects_ensemble_models_clearly(tmp_path):
    with pytest.raises(TypeError, match="ensemble models are not supported"):
        run_temporal_explainability(
            TemporalTreeEnsemble(),
            _patient_data(),
            tmp_path,
            BIN_EDGES,
        )
