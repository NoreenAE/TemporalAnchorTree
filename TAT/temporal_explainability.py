"""Post-training temporal decision-path explainability utilities."""

import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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

TEMPORAL_USAGE_PERCENTAGE_COLUMN = "usage_percentage"
TEMPORAL_USAGE_ANCHORS = ("Birth Date Anchor", "NOW Anchor")
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
            if (
                isinstance(offset, bool)
                or not isinstance(offset, Real)
                or not np.isfinite(offset)
            ):
                raise ValueError(
                    f"Offsets for {anchor!r} must be finite real numbers or None."
                )
            numeric_offsets.append(float(offset))

        if not numeric_offsets:
            raise ValueError(
                f"At least one non-zero monthly offset is required for {anchor!r}."
            )

        first_edge = math.floor(min(0, min(numeric_offsets)))
        last_edge = math.ceil(max(0, max(numeric_offsets)))
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


def _effective_offsets(anchor, stored_offset) -> Tuple[Optional[Real], Optional[Real]]:
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
    """Save the individual three-panel temporal-usage percentage figure."""
    edges_by_anchor = _validate_bin_edges(bin_edges_by_anchor)
    percentage_table, full_timeline_percentage = (
        calculate_temporal_usage_percentages(usage_table, summary)
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    anchors = list(edges_by_anchor)
    figure, axes = plt.subplots(
        1,
        len(anchors) + 1,
        figsize=(15.5, 5.3),
        sharey=True,
        gridspec_kw={
            "width_ratios": [3.2] * len(anchors) + [1.2],
        },
    )

    for axis, anchor in zip(axes[:-1], anchors):
        anchor_usage = percentage_table.loc[
            percentage_table["anchor"].eq(anchor)
        ].sort_values(["bin_start_offset", "bin_end_offset"])
        expected_bins = len(edges_by_anchor[anchor]) - 1
        if len(anchor_usage) != expected_bins:
            raise ValueError(
                f"Expected {expected_bins} monthly bins for {anchor!r}; "
                f"found {len(anchor_usage)}."
            )
        positions = np.arange(len(anchor_usage))
        heights = anchor_usage[TEMPORAL_USAGE_PERCENTAGE_COLUMN].to_numpy()
        labels = [
            f"{start:g}–{end:g}"
            for start, end in zip(
                anchor_usage["bin_start_offset"],
                anchor_usage["bin_end_offset"],
            )
        ]
        bars = axis.bar(positions, heights, width=0.8)
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=35, ha="right")
        axis.set_title(anchor)
        axis.set_xlabel("Months Relative to Anchor", labelpad=8)
        axis.set_ylim(0, 100)
        axis.grid(axis="y", alpha=0.25)
        _annotate_nonzero_percentages(axis, bars, heights)

    full_timeline_axis = axes[-1]
    full_timeline_bar = full_timeline_axis.bar(
        [0],
        [full_timeline_percentage],
        width=0.58,
    )
    full_timeline_axis.set_title("Full Timeline")
    full_timeline_axis.set_xticks([0])
    full_timeline_axis.set_xticklabels([""])
    full_timeline_axis.set_xlabel("Full Timeline", labelpad=8)
    full_timeline_axis.set_ylim(0, 100)
    full_timeline_axis.grid(axis="y", alpha=0.25)
    _annotate_nonzero_percentages(
        full_timeline_axis,
        full_timeline_bar,
        [full_timeline_percentage],
    )

    figure.text(
        0.014,
        0.55,
        "Temporal split usage (%)",
        rotation="vertical",
        va="center",
    )
    figure.text(
        0.5,
        0.035,
        f"Evaluated patients: {int(summary['evaluated_patient_count'])}",
        ha="center",
    )
    figure.subplots_adjust(
        left=0.07,
        right=0.985,
        bottom=0.25,
        top=0.90,
        wspace=0.16,
    )
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def calculate_temporal_usage_percentages(
    usage_table: pd.DataFrame,
    summary: Mapping[str, float],
) -> Tuple[pd.DataFrame, float]:
    """Convert saved raw usage counts to percentages of all split usages."""
    missing_columns = {
        "anchor",
        "bin_start_offset",
        "bin_end_offset",
        "raw_usage_count",
    }.difference(usage_table.columns)
    if missing_columns:
        raise ValueError(
            "usage_table is missing required columns: "
            f"{sorted(missing_columns)}."
        )
    missing_summary = {
        "evaluated_patient_count",
        "temporal_split_usages",
        "full_timeline_usages",
    }.difference(summary)
    if missing_summary:
        raise ValueError(
            "summary is missing required fields: "
            f"{sorted(missing_summary)}."
        )
    total_usages = int(summary["temporal_split_usages"])
    full_timeline_usages = int(summary["full_timeline_usages"])
    if total_usages <= 0:
        raise ValueError("temporal_split_usages must be positive.")
    if not 0 <= full_timeline_usages <= total_usages:
        raise ValueError(
            "full_timeline_usages must be between zero and total usages."
        )
    raw_counts = pd.to_numeric(
        usage_table["raw_usage_count"],
        errors="coerce",
    )
    if raw_counts.isna().any() or raw_counts.lt(0).any():
        raise ValueError("raw_usage_count must contain non-negative numbers.")
    if raw_counts.gt(total_usages).any():
        raise ValueError(
            "A monthly-bin usage count cannot exceed total split usages."
        )

    percentages = usage_table.copy()
    percentages[TEMPORAL_USAGE_PERCENTAGE_COLUMN] = (
        raw_counts / total_usages * 100.0
    )
    full_timeline_percentage = (
        full_timeline_usages / total_usages * 100.0
    )
    return percentages, float(full_timeline_percentage)


def load_temporal_usage_artifacts(output_dir) -> dict:
    """Load already-computed temporal usage counts for plotting only."""
    output_dir = Path(output_dir)
    usage_path = output_dir / "aggregated_temporal_usage.csv"
    summary_path = output_dir / "temporal_usage_summary.json"
    if not usage_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(
            "Saved temporal usage artifacts are required: "
            f"{usage_path} and {summary_path}."
        )
    usage_table = pd.read_csv(usage_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    calculate_temporal_usage_percentages(usage_table, summary)
    return {
        "usage_table": usage_table,
        "summary": summary,
        "usage_path": usage_path,
        "summary_path": summary_path,
    }


def _informative_annotation_indices(heights, *, required=()):
    """Return a small set of bar indices that carry useful labels."""
    values = np.asarray(heights, dtype=float)
    nonzero = np.flatnonzero(values > 0)
    if not len(nonzero):
        return set()

    selected = {int(nonzero[np.argmax(values[nonzero])])}
    if len(values) > 1:
        change_index = int(np.argmax(np.abs(np.diff(values))))
        selected.update((change_index, change_index + 1))
    selected.update(int(index) for index in required if 0 <= index < len(values))
    return selected & set(int(index) for index in nonzero)


def _annotate_selected_percentages(axis, bars, heights, indices):
    """Annotate only selected non-zero bars to keep comparison readable."""
    for index in sorted(indices):
        bar = bars[index]
        height = float(heights[index])
        axis.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            clip_on=False,
        )


def plot_temporal_usage_comparison(
    usage_by_outcome: Mapping[str, Mapping],
    output_path,
    subtitle: Optional[str] = None,
) -> Path:
    """Save a three-panel FA-versus-GI temporal usage comparison."""
    normalized = {
        str(outcome).upper(): artifacts
        for outcome, artifacts in usage_by_outcome.items()
    }
    if set(normalized) != {"FA", "GI"}:
        raise ValueError("usage_by_outcome must contain exactly FA and GI.")

    prepared = {}
    evaluated_patient_counts = set()
    for outcome in ("FA", "GI"):
        artifacts = normalized[outcome]
        if not {"usage_table", "summary"}.issubset(artifacts):
            raise ValueError(
                f"{outcome} artifacts require usage_table and summary."
            )
        percentage_table, full_percentage = (
            calculate_temporal_usage_percentages(
                artifacts["usage_table"],
                artifacts["summary"],
            )
        )
        evaluated_patient_counts.add(
            int(artifacts["summary"]["evaluated_patient_count"])
        )
        prepared[outcome] = {
            "usage_table": percentage_table,
            "full_percentage": full_percentage,
        }
    if len(evaluated_patient_counts) != 1:
        raise ValueError("FA and GI evaluated-patient counts must match.")
    evaluated_patient_count = evaluated_patient_counts.pop()

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][:2]
    outcome_colors = dict(zip(("FA", "GI"), colors))
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(17, 7),
        sharey=True,
        gridspec_kw={"width_ratios": [3.2, 3.2, 1.25]},
    )
    bar_width = 0.36
    for axis, anchor in zip(axes[:2], TEMPORAL_USAGE_ANCHORS):
        bins = sorted({
            (float(row.bin_start_offset), float(row.bin_end_offset))
            for outcome in ("FA", "GI")
            for row in prepared[outcome]["usage_table"].loc[
                prepared[outcome]["usage_table"]["anchor"].eq(anchor)
            ].itertuples(index=False)
        })
        if not bins:
            raise ValueError(f"No saved monthly bins found for {anchor!r}.")
        positions = np.arange(len(bins))
        if anchor == "Birth Date Anchor":
            region_start, region_end = 0, 6
        else:
            region_start, region_end = -1, 0
        first_region_bin = bins.index((float(region_start), float(region_start + 1)))
        last_region_bin = bins.index((float(region_end - 1), float(region_end)))
        axis.axvspan(
            first_region_bin - 0.5,
            last_region_bin + 0.5,
            color="#d8b365",
            alpha=0.18,
            zorder=0,
        )
        for outcome_index, outcome in enumerate(("FA", "GI")):
            anchor_usage = prepared[outcome]["usage_table"].loc[
                prepared[outcome]["usage_table"]["anchor"].eq(anchor)
            ].set_index(["bin_start_offset", "bin_end_offset"])
            heights = np.asarray([
                float(
                    anchor_usage.loc[
                        bin_pair,
                        TEMPORAL_USAGE_PERCENTAGE_COLUMN,
                    ]
                )
                if bin_pair in anchor_usage.index
                else 0.0
                for bin_pair in bins
            ])
            offset = (outcome_index - 0.5) * bar_width
            bars = axis.bar(
                positions + offset,
                heights,
                width=bar_width,
                color=outcome_colors[outcome],
                zorder=2,
            )
            required = ()
            if outcome == "GI" and anchor == "NOW Anchor":
                required = (bins.index((-1.0, 0.0)),)
            _annotate_selected_percentages(
                axis,
                bars,
                heights,
                _informative_annotation_indices(heights, required=required),
            )
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [f"{start:g}–{end:g}" for start, end in bins],
            rotation=35,
            ha="right",
        )
        axis.set_title(anchor, fontsize=14, pad=12)
        axis.set_xlabel("Months Relative to Anchor", labelpad=8, fontsize=12)
        axis.tick_params(axis="both", labelsize=10)
        axis.set_ylim(0, 100)
        axis.grid(axis="y", alpha=0.25, zorder=1)

    full_timeline_axis = axes[2]
    full_heights = [
        prepared[outcome]["full_percentage"] for outcome in ("FA", "GI")
    ]
    full_bars = full_timeline_axis.bar(
        np.arange(2),
        full_heights,
        width=0.62,
        color=[outcome_colors[outcome] for outcome in ("FA", "GI")],
    )
    full_timeline_axis.set_xticks(np.arange(2))
    full_timeline_axis.set_xticklabels(["FA", "GI"])
    full_timeline_axis.set_title("Full Timeline", fontsize=14, pad=12)
    full_timeline_axis.set_ylim(0, 100)
    full_timeline_axis.tick_params(axis="both", labelsize=10)
    full_timeline_axis.grid(axis="y", alpha=0.25, zorder=1)
    _annotate_selected_percentages(
        full_timeline_axis,
        full_bars,
        full_heights,
        _informative_annotation_indices(full_heights),
    )
    for axis, label in zip(axes, ("(a)", "(b)", "(c)")):
        axis.text(
            -0.02,
            1.02,
            label,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            clip_on=False,
        )

    figure.text(
        0.014,
        0.51,
        "Temporal usage (%)",
        rotation="vertical",
        va="center",
        fontsize=13,
    )
    figure.suptitle("Temporal Split Usage for FA and GI", y=0.97, fontsize=19)
    figure.legend(
        handles=[
            Patch(facecolor=outcome_colors[outcome], label=outcome)
            for outcome in ("FA", "GI")
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=2,
        frameon=False,
        fontsize=12,
    )
    figure.subplots_adjust(
        left=0.07,
        right=0.985,
        bottom=0.23,
        top=0.82,
        wspace=0.16,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def _annotate_nonzero_percentages(axis, bars, percentages) -> None:
    """Label non-zero percentage bars without cluttering empty bins."""
    for bar, percentage in zip(bars, percentages):
        if percentage <= 0:
            continue
        axis.annotate(
            f"{percentage:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


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
