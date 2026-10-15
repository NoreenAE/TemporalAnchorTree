import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from TAT.temporal_explainability import (
    PATH_TABLE_COLUMNS,
    aggregate_temporal_usage,
    calculate_temporal_usage_percentages,
    derive_monthly_bin_edges,
    load_temporal_usage_artifacts,
    plot_temporal_usage_comparison,
    run_temporal_explainability,
    trace_temporal_decision_paths,
)
from TAT.temporal_tree import TemporalLeaf, TemporalSplitNode, TemporalTree
from TAT.temporal_tree_ensemble import TemporalTreeEnsemble


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


def test_usage_percentages_use_all_patient_path_split_usages():
    usage_table = pd.DataFrame({
        "anchor": ["Birth Date Anchor", "NOW Anchor"],
        "bin_start_offset": [0.0, -1.0],
        "bin_end_offset": [1.0, 0.0],
        "raw_usage_count": [6, 2],
        "normalized_usage_per_patient": [3.0, 1.0],
    })
    summary = {
        "evaluated_patient_count": 2,
        "temporal_split_usages": 10,
        "full_timeline_usages": 3,
    }

    percentages, full_percentage = calculate_temporal_usage_percentages(
        usage_table,
        summary,
    )

    assert percentages["usage_percentage"].tolist() == [60.0, 20.0]
    assert full_percentage == 30.0


def test_load_and_plot_comparison_use_saved_aggregate_artifacts(tmp_path):
    usage_tables = {
        "fa": pd.DataFrame({
            "anchor": [
                "Birth Date Anchor",
                "Birth Date Anchor",
                "NOW Anchor",
            ],
            "bin_start_offset": [0.0, 1.0, -1.0],
            "bin_end_offset": [1.0, 2.0, 0.0],
            "raw_usage_count": [6, 4, 0],
            "normalized_usage_per_patient": [3.0, 2.0, 0.0],
        }),
        "gi": pd.DataFrame({
            "anchor": ["Birth Date Anchor", "NOW Anchor"],
            "bin_start_offset": [0.0, -1.0],
            "bin_end_offset": [1.0, 0.0],
            "raw_usage_count": [1, 7],
            "normalized_usage_per_patient": [0.5, 3.5],
        }),
    }
    summaries = {
        "fa": {
            "evaluated_patient_count": 2,
            "temporal_split_usages": 10,
            "full_timeline_usages": 4,
        },
        "gi": {
            "evaluated_patient_count": 2,
            "temporal_split_usages": 10,
            "full_timeline_usages": 2,
        },
    }
    # The comparison highlights complete monthly ground-truth regions. Fill
    # unused bins with zeros, as aggregate_temporal_usage does for real exports.
    monthly_bins = pd.MultiIndex.from_tuples(
        [
            (anchor, float(start), float(end))
            for anchor, edges in BIN_EDGES.items()
            for start, end in zip(edges[:-1], edges[1:])
        ],
        names=["anchor", "bin_start_offset", "bin_end_offset"],
    )
    for outcome, table in usage_tables.items():
        usage_tables[outcome] = (
            table.set_index(monthly_bins.names)
            .reindex(monthly_bins, fill_value=0)
            .reset_index()
        )
    loaded = {}
    for outcome in ("fa", "gi"):
        directory = tmp_path / outcome
        directory.mkdir()
        usage_tables[outcome].to_csv(
            directory / "aggregated_temporal_usage.csv",
            index=False,
        )
        (directory / "temporal_usage_summary.json").write_text(
            json.dumps(summaries[outcome]),
            encoding="utf-8",
        )
        loaded[outcome] = load_temporal_usage_artifacts(directory)
        pd.testing.assert_frame_equal(
            loaded[outcome]["usage_table"], usage_tables[outcome]
        )

    output_path = plot_temporal_usage_comparison(
        loaded,
        tmp_path / "temporal_usage_comparison_fa_vs_gi.png",
        subtitle="test configuration",
    )

    assert output_path.is_file()
    assert output_path.stat().st_size > 0


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
