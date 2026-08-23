"""Post-training temporal decision-path explainability utilities."""

from numbers import Integral
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .temporal_tree import TemporalTree


PATH_TABLE_COLUMNS = [
    "patient_id",
    "node_id",
    "depth",
    "feature",
    "aggregation",
    "anchor",
    "stored_offset",
    "effective_start_offset",
    "effective_end_offset",
    "aggregated_value",
    "missing",
    "direction",
    "terminal_leaf_id",
    "is_full_timeline",
    "is_zero_width",
    "raw_usage_weight",
    "per_patient_usage_weight",
    "evaluated_patient_count",
]

USAGE_TABLE_COLUMNS = [
    "anchor",
    "bin_start_offset",
    "bin_end_offset",
    "raw_usage_count",
    "normalized_usage_per_patient",
]


def derive_monthly_bin_edges(
    anchor_offset_map: Mapping,
) -> Dict:
    """Derive consecutive monthly bin edges from anchor-specific offsets.

    Each anchor's bins span the complete range between zero and its most
    extreme configured non-zero offset. ``None`` and zero-width offsets do not
    create bins.
    """
    if not isinstance(anchor_offset_map, Mapping):
        raise ValueError("anchor_offset_map must be a mapping.")

    bin_edges_by_anchor = {}
    for anchor, offsets in anchor_offset_map.items():
        if not isinstance(offsets, Sequence) or isinstance(offsets, (str, bytes)):
            raise ValueError(f"Offsets for {anchor!r} must be a sequence.")

        numeric_offsets = []
        for offset in offsets:
            if offset is None or offset == 0:
                continue
            if isinstance(offset, bool) or not isinstance(offset, Integral):
                raise ValueError(
                    f"Monthly offsets for {anchor!r} must be integers or None."
                )
            numeric_offsets.append(int(offset))

        if not numeric_offsets:
            raise ValueError(
                f"At least one non-zero monthly offset is required for {anchor!r}."
            )

        first_edge = min(0, min(numeric_offsets))
        last_edge = max(0, max(numeric_offsets))
        bin_edges_by_anchor[anchor] = list(range(first_edge, last_edge + 1))

    return bin_edges_by_anchor


def _build_preorder_node_id_map(root) -> Dict[int, int]:
    """Map node object identities to deterministic pre-order integer IDs."""
    node_ids = {}

    def visit(node) -> None:
        if node is None:
            return
        node_ids[id(node)] = len(node_ids)
        if not node.is_leaf():
            visit(node.left)
            visit(node.right)

    visit(root)
    return node_ids


def _effective_offsets(anchor, stored_offset) -> Tuple[Optional[int], Optional[int]]:
    """Return the ordered start and end offsets represented by a node."""
    if anchor is None and stored_offset is None:
        return None, None
    if stored_offset is None:
        return None, None
    return min(0, stored_offset), max(0, stored_offset)


def trace_temporal_decision_paths(
    model: TemporalTree,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Trace every patient once and return one row per visited split node.

    Routing reuses the fitted tree's production aggregation, missing-direction,
    and branch-anchor methods. Patient ordering matches ``TemporalTree.apply``.
    """
    if not isinstance(model, TemporalTree):
        raise TypeError(
            "Temporal explainability version 1 supports a single "
            "fitted TemporalTree only."
        )
    if model.tree is None:
        raise ValueError("Model not fitted. Call `.fit()` first.")
    required_columns = {"patient_id", "time", "feature_name", "value"}
    missing_columns = required_columns.difference(X.columns)
    if missing_columns:
        raise ValueError(
            "X is missing required columns: "
            f"{sorted(missing_columns)}."
        )

    traced_data = X.copy()
    traced_data["time"] = pd.to_datetime(traced_data["time"])
    patient_groups = traced_data.groupby("patient_id", sort=False)
    evaluated_patient_count = patient_groups.ngroups
    per_patient_weight = (
        1.0 / evaluated_patient_count if evaluated_patient_count else 0.0
    )
    node_ids = _build_preorder_node_id_map(model.tree)
    records = []

    for patient_id, patient_data in patient_groups:
        node = model.tree
        patient_records = []

        while not node.is_leaf():
            aggregated_value = model._aggregate_feature_over_time(
                patient_data,
                node.feature,
                node.time_window,
                node.aggregation,
            )[0]
            missing = bool(pd.isna(aggregated_value))
            direction = model._get_prediction_direction(aggregated_value, node)
            recorded_value = (
                aggregated_value.item()
                if isinstance(aggregated_value, np.ndarray)
                and aggregated_value.size == 1
                else aggregated_value
            )
            anchor, stored_offset = node.time_window
            effective_start, effective_end = _effective_offsets(
                anchor,
                stored_offset,
            )
            is_full_timeline = anchor is None and stored_offset is None
            is_zero_width = stored_offset == 0 and not is_full_timeline

            patient_records.append({
                "patient_id": patient_id,
                "node_id": node_ids[id(node)],
                "depth": node.depth,
                "feature": node.feature,
                "aggregation": node.aggregation,
                "anchor": anchor,
                "stored_offset": stored_offset,
                "effective_start_offset": effective_start,
                "effective_end_offset": effective_end,
                "aggregated_value": recorded_value,
                "missing": missing,
                "direction": direction,
                "terminal_leaf_id": None,
                "is_full_timeline": is_full_timeline,
                "is_zero_width": is_zero_width,
                "raw_usage_weight": 1,
                "per_patient_usage_weight": per_patient_weight,
                "evaluated_patient_count": evaluated_patient_count,
            })

            patient_data = model._append_prediction_branch_anchor(
                patient_data,
                node,
                direction,
            )
            node = node.left if direction == "left" else node.right

        terminal_leaf_id = getattr(node, "sn", -1)
        for record in patient_records:
            record["terminal_leaf_id"] = terminal_leaf_id
        records.extend(patient_records)

    return pd.DataFrame.from_records(records, columns=PATH_TABLE_COLUMNS)


def _validate_bin_edges(
    bin_edges_by_anchor: Mapping,
) -> Dict:
    """Validate and copy configured monthly histogram edges."""
    if not isinstance(bin_edges_by_anchor, Mapping) or not bin_edges_by_anchor:
        raise ValueError("bin_edges_by_anchor must be a non-empty mapping.")

    validated_edges = {}
    for anchor, edges in bin_edges_by_anchor.items():
        edge_array = np.asarray(edges)
        if edge_array.ndim != 1 or len(edge_array) < 2:
            raise ValueError(f"At least two bin edges are required for {anchor!r}.")
        if not np.issubdtype(edge_array.dtype, np.number):
            raise ValueError(f"Bin edges for {anchor!r} must be numeric.")
        edge_array = edge_array.astype(float)
        if not np.isfinite(edge_array).all() or not np.all(np.diff(edge_array) > 0):
            raise ValueError(
                f"Bin edges for {anchor!r} must be finite and strictly increasing."
            )
        if not np.allclose(np.diff(edge_array), 1.0):
            raise ValueError(f"Bin edges for {anchor!r} must use monthly spacing.")
        validated_edges[anchor] = edge_array
    return validated_edges


def aggregate_temporal_usage(
    path_table: pd.DataFrame,
    bin_edges_by_anchor: Mapping,
    evaluated_patient_count: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Aggregate audited path rows into additive monthly usage bins."""
    missing_columns = set(PATH_TABLE_COLUMNS).difference(path_table.columns)
    if missing_columns:
        raise ValueError(
            "path_table is missing required columns: "
            f"{sorted(missing_columns)}."
        )
    if (
        isinstance(evaluated_patient_count, bool)
        or not isinstance(evaluated_patient_count, Integral)
        or evaluated_patient_count < 1
    ):
        raise ValueError("evaluated_patient_count must be a positive integer.")

    edges_by_anchor = _validate_bin_edges(bin_edges_by_anchor)
    usage_records = []

    eligible_rows = path_table.loc[
        ~path_table["is_full_timeline"] & ~path_table["is_zero_width"]
    ]
    for anchor, edges in edges_by_anchor.items():
        anchor_rows = eligible_rows.loc[eligible_rows["anchor"] == anchor]
        bin_starts = edges[:-1]
        bin_ends = edges[1:]

        if anchor_rows.empty:
            raw_counts = np.zeros(len(bin_starts), dtype=int)
        else:
            window_starts = anchor_rows["effective_start_offset"].to_numpy(
                dtype=float,
            )
            window_ends = anchor_rows["effective_end_offset"].to_numpy(
                dtype=float,
            )
            raw_counts = np.fromiter(
                (
                    np.count_nonzero(
                        (window_starts < bin_end)
                        & (window_ends > bin_start)
                    )
                    for bin_start, bin_end in zip(bin_starts, bin_ends)
                ),
                dtype=int,
                count=len(bin_starts),
            )

        for bin_start, bin_end, raw_count in zip(
            bin_starts,
            bin_ends,
            raw_counts,
        ):
            usage_records.append({
                "anchor": anchor,
                "bin_start_offset": bin_start,
                "bin_end_offset": bin_end,
                "raw_usage_count": int(raw_count),
                "normalized_usage_per_patient": (
                    float(raw_count) / int(evaluated_patient_count)
                ),
            })

    full_timeline_usages = int(path_table["is_full_timeline"].sum())
    zero_width_usages = int(path_table["is_zero_width"].sum())
    summary = {
        "evaluated_patient_count": int(evaluated_patient_count),
        "temporal_split_usages": int(len(path_table)),
        "full_timeline_usages": full_timeline_usages,
        "zero_width_usages": zero_width_usages,
        "normalized_full_timeline_usages": (
            full_timeline_usages / int(evaluated_patient_count)
        ),
        "normalized_zero_width_usages": (
            zero_width_usages / int(evaluated_patient_count)
        ),
    }
    usage_table = pd.DataFrame.from_records(
        usage_records,
        columns=USAGE_TABLE_COLUMNS,
    )
    return usage_table, summary


def plot_temporal_usage(
    usage_table: pd.DataFrame,
    summary: Mapping[str, float],
    bin_edges_by_anchor: Mapping,
    output_path,
) -> Path:
    """Save a multi-panel normalized temporal split-usage histogram."""
    edges_by_anchor = _validate_bin_edges(bin_edges_by_anchor)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    anchors = list(edges_by_anchor)
    figure, axes = plt.subplots(
        1,
        len(anchors),
        figsize=(7 * len(anchors), 5),
        squeeze=False,
        sharey=True,
    )

    for axis, anchor in zip(axes[0], anchors):
        anchor_usage = usage_table.loc[usage_table["anchor"] == anchor]
        positions = np.arange(len(anchor_usage))
        heights = anchor_usage["normalized_usage_per_patient"].to_numpy()
        labels = [
            f"{start:g} to {end:g}"
            for start, end in zip(
                anchor_usage["bin_start_offset"],
                anchor_usage["bin_end_offset"],
            )
        ]
        axis.bar(positions, heights, width=0.8)
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=35, ha="right")
        axis.set_title(f"{anchor} — Temporal Split Usage")
        axis.set_xlabel("Months Relative to Anchor")
        axis.set_ylabel(
            "Mean Patient-Path Temporal Split Usages per Patient"
        )
        axis.grid(axis="y", alpha=0.25)

    figure.text(
        0.5,
        0.01,
        "Evaluated patients: "
        f"{int(summary['evaluated_patient_count'])} | "
        f"Full-timeline usages: {int(summary['full_timeline_usages'])} | "
        f"Zero-width usages: {int(summary['zero_width_usages'])}",
        ha="center",
    )
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def run_temporal_explainability(
    model: TemporalTree,
    X: pd.DataFrame,
    output_dir,
    bin_edges_by_anchor: Mapping,
) -> Dict:
    """Trace training paths and save the version-one CSV and histogram."""
    if not isinstance(model, TemporalTree):
        raise TypeError(
            "Temporal explainability version 1 supports single-tree "
            "TemporalTree models only; ensemble models are not supported."
        )

    evaluated_patient_count = int(X["patient_id"].nunique())
    if evaluated_patient_count < 1:
        raise ValueError("X must contain at least one patient.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path_table = trace_temporal_decision_paths(model, X)
    usage_table, summary = aggregate_temporal_usage(
        path_table,
        bin_edges_by_anchor,
        evaluated_patient_count,
    )

    path_table_path = output_dir / "temporal_path_usage.csv"
    histogram_path = output_dir / "temporal_usage_histogram.png"
    path_table.to_csv(path_table_path, index=False)
    plot_temporal_usage(
        usage_table,
        summary,
        bin_edges_by_anchor,
        histogram_path,
    )

    return {
        "path_table": path_table,
        "usage_table": usage_table,
        "summary": summary,
        "path_table_path": path_table_path,
        "histogram_path": histogram_path,
    }
