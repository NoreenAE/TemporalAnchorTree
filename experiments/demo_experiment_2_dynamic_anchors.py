"""Show real training anchor state and one held-out prediction path.

Run from the repository root:
    python experiments/demo_experiment_2_dynamic_anchors.py

No work occurs on import. Only one Dynamic ON tree is fitted. The original
Experiment 2 generator/settings/helpers are reused without modifying them.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import textwrap
from types import MethodType

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.controlled_dynamic_anchors_100 import (  # noqa: E402
    anchor_values, require, tree_records, write_json,
)
from experiments.experiment_2_fish_introduction_dynamic_anchors import (  # noqa: E402
    SHARED_MODEL_CONFIG, evaluate, load_dataset, make_model,
)
from experiments.experiment_2_outputs import _tree_layout  # noqa: E402
from generate_fish_intro_dataset import SEED, generate_data  # noqa: E402
from settree.temporal_tree import TemporalTree  # noqa: E402

DEFAULT_OUTPUT = ROOT / "experiments/results/demo_experiment_2_dynamic_anchors"
LABEL_THRESHOLD = 0.5


def select_patients(patients: pd.DataFrame) -> pd.DataFrame:
    """Select 10 training blocks and one test block, without outcome selection."""
    rng = np.random.default_rng(SEED)
    selected = []
    for partition, count in (("train", 10), ("test", 1)):
        pool = patients.loc[patients.partition.eq(partition)]
        blocks = rng.choice(sorted(pool.block.unique()), count, replace=False)
        selected.append(pool.loc[pool.block.isin(blocks)].copy())
    cohort = pd.concat(selected, ignore_index=True)
    require(cohort.patient_id.is_unique, "Training/test patient IDs overlap")
    require(cohort.groupby("block").partition.nunique().eq(1).all(),
            "Training/test blocks overlap")
    require(cohort.groupby(["partition", "block"]).size().eq(10).all(),
            "Expected complete ten-patient matched blocks")
    require(cohort.partition.value_counts().to_dict() == {"train": 100, "test": 10},
            "Expected exactly 100 training and 10 test patients")
    return cohort


def timestamp_list(value) -> list:
    """Normalize a predefined timestamp or dynamic singleton/empty list."""
    if isinstance(value, (list, np.ndarray)):
        return list(value)
    return [] if value is None or pd.isna(value) else [value]


def describe_anchors(data: pd.DataFrame, names: list, predefined: list) -> list[dict]:
    """Separate candidate reference names from each patient's defined dates."""
    patients = list(data.patient_id.unique())
    descriptions = []
    for name in names:
        rows = data.loc[data.feature_name.eq(name)]
        require(not rows.patient_id.duplicated().any(), "Duplicate anchor rows")
        values = dict(zip(rows.patient_id, rows.value))
        require(set(values) == set(patients), f"Missing anchor rows for {name}")
        dates = {pid: timestamp_list(values[pid]) for pid in patients}
        descriptions.append(dict(
            name=name, kind="predefined" if name in predefined else "dynamic",
            available_as_reference=True,
            timestamp_defined_count=sum(bool(value) for value in dates.values()),
            timestamp_undefined_count=sum(not value for value in dates.values()),
            patient_timestamps=dates,
        ))
    return descriptions


def fit_with_anchor_capture(model: TemporalTree, training: pd.DataFrame,
                            labels: pd.Series) -> list[dict]:
    """Observe production node creation and branch preparation, including leaves."""
    node_states, branch_states = {}, {}
    get_node = model._get_node
    prepare = model._prepare_branch_anchor

    def capture_node(self, X, y, depth, anchors_list):
        node = get_node(X, y, depth, anchors_list)
        node_states[id(node)] = describe_anchors(X, anchors_list, self.anchors_list)
        return node

    def capture_branch(self, **kwargs):
        result = prepare(**kwargs)
        name, available, created, _ = result
        branch_states[(id(kwargs["node"]), kwargs["direction"])] = dict(
            generated_anchor=name, available_anchors=list(available),
            patient_timestamps={
                row.patient_id: timestamp_list(row.value)
                for row in created.itertuples(index=False)
            },
        )
        return result

    model._get_node = MethodType(capture_node, model)
    model._prepare_branch_anchor = MethodType(capture_branch, model)
    try:
        model.fit(training, labels)
    finally:
        del model._get_node
        del model._prepare_branch_anchor

    records, ids = tree_records(model.tree)

    def attach(node):
        record = records[ids[id(node)]]
        record["available_anchors"] = node_states[id(node)]
        record["generated_anchor"] = None
        record["children_anchor_state"] = {}
        if node.is_leaf():
            return
        record["source_node_id"] = node.source_node_id
        for direction in ("left", "right"):
            state = branch_states[(id(node), direction)]
            record["children_anchor_state"][direction] = dict(
                child_node_id=record[direction], **state,
            )
            record["generated_anchor"] = state["generated_anchor"]
            attach(getattr(node, direction))

    attach(model.tree)
    return records


def predict_with_capture(model: TemporalTree, test: pd.DataFrame,
                         records: list[dict]) -> tuple[np.ndarray, dict]:
    """Capture values and regenerated dates from the actual predict call."""
    _, ids = tree_records(model.tree)
    traces = {pid: [] for pid in test.patient_id.unique()}
    append = model._append_prediction_branch_anchor

    def capture(self, row, node, direction):
        pid = row.patient_id.iloc[0]
        before = anchor_values(row)
        # Annotation only: use the same production aggregate on the same row.
        value = self._aggregate_feature_over_time(
            row, node.feature, node.time_window, node.aggregation,
        )[0]
        # Production returns an array directly when the feature has no rows,
        # and (array, anchor) otherwise. After [0], one patient's aggregate is
        # therefore either a scalar or a singleton array, including NaN.
        value = np.asarray(value).item()
        require(direction == self._get_prediction_direction(value, node),
                "Captured prediction direction differs from production routing")
        outgoing = append(row, node, direction)
        after = anchor_values(outgoing)
        record = records[ids[id(node)]]
        name = record["generated_anchor"]
        source, offset = node.time_window
        uses_dynamic = offset is not None and source not in self.anchors_list
        source_dates = timestamp_list(before.get(source))
        missing = bool(pd.isna(value))
        traces[pid].append(dict(
            node_id=record["node_id"], depth=node.depth,
            aggregated_value=None if missing else float(value), missing=missing,
            condition_satisfied=None if missing else bool(value <= node.threshold),
            direction=direction, next_node_id=record[direction],
            source_anchor=source, full_timeline=offset is None,
            uses_dynamic_anchor=uses_dynamic,
            source_anchor_timestamps=source_dates,
            source_timestamp_defined=bool(source_dates),
            generated_anchor=name,
            generated_anchor_timestamps=timestamp_list(after.get(name)),
        ))
        return outgoing

    model._append_prediction_branch_anchor = MethodType(capture, model)
    try:
        scores = model.predict(test)
    finally:
        del model._append_prediction_branch_anchor
    return scores, traces


def print_training_summary(records: list) -> None:
    """Print every actual node and its incoming/outgoing anchor state."""
    print("\nTRAINING: anchor references and patient-specific timestamp coverage")
    for node in records:
        print(f"\nNode {node['node_id']} | depth {node['depth']} | "
              f"{node['patients']} patients | {node['type']}")
        if node["type"] == "split":
            print(f"  Split: {node['aggregation']}({node['feature']}) "
                  f"<= {node['threshold']:g}; window={node['window']}; "
                  f"missing -> {node['missing_direction']}")
            print(f"  Anchor source ID: {node['source_node_id']}")
        else:
            print(f"  P(FA=1)={node['score']:.4f}; no selected split")
        for anchor in node["available_anchors"]:
            print(f"  [{anchor['kind']}] {anchor['name']}: available reference; "
                  f"timestamps defined for {anchor['timestamp_defined_count']}/"
                  f"{node['patients']} patients")
        print(f"  Generated logical anchor: {node['generated_anchor'] or 'none'}")
        for direction, state in node["children_anchor_state"].items():
            print(f"  -> {direction} child {state['child_node_id']}: "
                  + "; ".join(state["available_anchors"]))


def choose_patient(traces: dict) -> tuple[str, str]:
    """Require creation and later reuse of the same defined patient timestamp."""
    for pid, path in traces.items():
        created = {}
        for visit in path:
            source_dates = [pd.Timestamp(date) for date in
                            visit["source_anchor_timestamps"]]
            if (visit["uses_dynamic_anchor"] and source_dates
                    and created.get(visit["source_anchor"]) == source_dates):
                return pid, (
                    "Patient creates a valid dynamic anchor and reuses "
                    "its timestamp at a subsequent split."
                )
            name = visit["generated_anchor"]
            dates = [pd.Timestamp(date) for date in
                     visit["generated_anchor_timestamps"]]
            if name and dates:
                created[name] = dates
    raise ValueError(
        "No held-out patient trace demonstrates dynamic anchor creation and "
        "subsequent reuse with a valid timestamp. No fallback path is plotted."
    )


def plot_prediction_steps(axis: Axes, records: list[dict], path: list[dict],
                          patient: pd.Series, leaf_id: int) -> None:
    """Explain actual split visits with consistent creation/reference colors."""
    nodes = {node["node_id"]: node for node in records}
    creators = {
        node["generated_anchor"]: node for node in records
        if node["generated_anchor"]
    }
    palette = ("#E4D9F3", "#FBE3CE", "#D4EDE5", "#F3D6E0")
    generated = [visit["generated_anchor"] for visit in path
                 if visit["generated_anchor"]]
    colors = {name: palette[index % len(palette)]
              for index, name in enumerate(generated)}

    def anchor_text(name: str | None, dates: list) -> str:
        if name is None:
            return "None"
        if name in creators:
            creator = creators[name]
            label = (f"[D{creator['source_node_id']}] "
                     f"{creator['aggregation'].upper()}({creator['feature']})")
        else:
            label = "[P] " + name.removesuffix(" Anchor")
        timestamps = "\n".join(
            pd.Timestamp(date).strftime("%Y-%m-%d\n%H:%M:%S") for date in dates
        ) if dates else "Timestamp undefined []"
        return f"{label}\n{timestamps}"

    rows, row_colors = [], []
    created = {}
    expected_node = records[0]["node_id"]
    for visit in path:
        require(visit["node_id"] == expected_node, "Trace is not a connected path")
        node = nodes[visit["node_id"]]
        require(node["type"] == "split", "Trace visit must identify a split node")
        source, offset = node["window"]
        require(source == visit["source_anchor"], "Trace reference differs from tree")
        require(offset is None or source in {
            anchor["name"] for anchor in node["available_anchors"]
        }, "Selected reference is absent from this node's incoming anchors")
        require(visit["generated_anchor"] == node["generated_anchor"],
                "Trace generated anchor differs from tree")
        dates = visit["source_anchor_timestamps"]
        require(bool(dates) == visit["source_timestamp_defined"],
                "Trace timestamp availability is inconsistent")
        if visit["uses_dynamic_anchor"]:
            require(source in created, "Dynamic reference used before its creation")
            require([pd.Timestamp(date) for date in dates] == created[source],
                    "Reused timestamp differs from this patient's generated timestamp")
        value = visit["aggregated_value"]
        if visit["missing"]:
            decision, aggregate = "Missing", "Missing"
            direction = node["missing_direction"]
        else:
            satisfied = value <= node["threshold"]
            require(satisfied == visit["condition_satisfied"],
                    "Trace condition differs from the learned threshold")
            decision = "YES" if satisfied else "NO"
            aggregate = f"{value:g}"
            direction = "left" if satisfied else "right"
        require(direction == visit["direction"], "Trace routing differs from tree")
        expected_node = node[direction]
        require(expected_node == visit["next_node_id"], "Trace edge differs from tree")
        window = "full timeline" if offset is None else (
            f"{min(0, offset):g} to {max(0, offset):g} months from reference"
        )
        condition = (f"{node['aggregation'].upper()}({node['feature']}) "
                     f"<= {node['threshold']:g}\n{window}")
        name = visit["generated_anchor"]
        new_dates = visit["generated_anchor_timestamps"]
        reference = (
            "None (full timeline)" if offset is None else anchor_text(source, dates)
        )
        rows.append([
            str(node["node_id"]), condition, aggregate,
            f"{decision}\n-> {direction}\nNode {expected_node}",
            reference, anchor_text(name, new_dates),
        ])
        row_colors.append([
            "#EEF6FC", "white", "white", "white",
            colors.get(source, "#F3F3F3") if offset is not None else "white",
            colors.get(name, "white"),
        ])
        if name:
            created[name] = [pd.Timestamp(date) for date in new_dates]

    leaf = nodes[leaf_id]
    require(expected_node == leaf_id and leaf["type"] == "leaf",
            "Trace does not end at the recorded leaf")
    require(np.isclose(leaf["score"], patient["score"]),
            "Highlighted leaf probability differs from actual prediction")
    require(int(leaf["score"] >= LABEL_THRESHOLD) == int(patient["predicted_label"]),
            "Recorded class differs from the leaf probability threshold")
    rows.append([
        f"Leaf {leaf_id}", "Final prediction",
        f"P(positive)\n{patient['score']:.3f}",
        f"Class {int(patient['predicted_label'])}",
        f"True class: {int(patient['FA'])}", "No further split",
    ])
    row_colors.append(["#D1EEDB"] * 6)
    widths = (8, 26, 12, 14, 27, 27)
    rows = [[
        "\n".join(textwrap.fill(line, width=width)
                  for line in str(cell).splitlines())
        for cell, width in zip(row, widths)
    ] for row in rows]
    row_heights = [max(len(cell.splitlines()) for cell in row) * .19 + .25
                   for row in rows]
    axis_height = axis.get_position().height * axis.figure.get_figheight()
    table_height = (.65 + sum(row_heights)) / axis_height
    bottom = .94 - table_height
    require(bottom > .18, "Prediction table exceeds figure height")
    axis.axis("off")
    axis.set_title("B. Patient prediction steps", loc="left", fontsize=12, pad=16)
    table = axis.table(
        cellText=rows, cellColours=row_colors,
        colLabels=["Node ID", "Learned split\ncondition", "Patient\naggregate",
                   "Decision", "Reference anchor\nand patient timestamp",
                   "New dynamic anchor\nand patient timestamp"],
        colColours=["#DEE8EF"] * 6,
        colWidths=[.06, .23, .10, .11, .25, .25],
        cellLoc="left", bbox=(0, bottom, 1, table_height),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    heights = [.65, *row_heights]
    for (row, column), cell in table.get_celld().items():
        cell.set_height(heights[row] / axis_height)
        cell.set_edgecolor("#C4CDD4")
        cell.set_linewidth(.6)
        cell.PAD = .06
        if row == 0 or row == len(rows):
            cell.get_text().set_fontweight("bold")
    axis.text(
        0, bottom - .04,
        "[P] predefined; [Dk] dynamic anchor from persisted source split k.\n"
        "Matching identifiers and cell colors connect creation to later use.\n"
        "New timestamps come from this patient's eligible observations.\n"
        "A generated reference becomes available only at subsequent nodes.\n"
        "Undefined []: reference exists, but this patient has no timestamp.\n"
        "Missing aggregates follow the tree's stored missing direction.\n"
        f"Class threshold: P(positive) >= {LABEL_THRESHOLD:g}.",
        transform=axis.transAxes, va="top", fontsize=9, linespacing=1.4,
    )


def training_node_label(node: dict) -> str:
    """Show a learned split and its captured incoming candidate references."""
    if node["type"] == "leaf":
        return (f"Patients = {node['patients']}\n"
                f"P(positive) = {node['score']:.3f}")

    anchors = node["available_anchors"]
    names = {anchor["name"] for anchor in anchors}
    require(node["generated_anchor"] not in names,
            f"Node {node['node_id']} includes its own anchor in incoming references")

    def compact_anchor(name: str, kind: str) -> str:
        label = name.removesuffix(" Anchor")
        return f"[{'P' if kind == 'predefined' else 'D'}] {label}"

    source, offset = node["window"]
    reference = next((anchor for anchor in anchors if anchor["name"] == source), None)
    require(offset is None or reference is not None,
            f"Node {node['node_id']} uses an unavailable reference")
    lines = [f"Node {node['node_id']} | Patients = {node['patients']}",
             f"Feature: {node['feature']}"]
    if offset is None:
        lines.extend(("Reference: none (full timeline)", "Window: full timeline"))
    else:
        lines.extend((
            "Reference: " + compact_anchor(source, reference["kind"]),
            f"Window: {min(0, offset):g} to {max(0, offset):g} months",
        ))
    lines.extend((f"Aggregation: {node['aggregation'].upper()}",
                  f"Threshold: <= {node['threshold']:g}",
                  "Available anchors before split:"))
    lines.extend(compact_anchor(anchor["name"], anchor["kind"]) for anchor in anchors)
    return "\n".join(textwrap.fill(line, width=46) for line in lines)


def plot_prediction_path(records: list, path: list, patient: pd.Series | None,
                         reason: str, output: Path, *,
                         training_only: bool = False) -> int | None:
    """Render the clean trained tree and, for prediction, a separate step table."""
    visits = {visit["node_id"]: visit for visit in path}
    leaf_id = None if training_only else (
        path[-1]["next_node_id"] if path else records[0]["node_id"]
    )
    visited = (
        {n["node_id"] for n in records} if training_only else set(visits) | {leaf_id}
    )
    path_edges = {(v["node_id"], v["next_node_id"]) for v in path}
    labels = {n["node_id"]: training_node_label(n) for n in records}
    line_height = .22
    sizes = {
        kind: (4.8 if kind == "split" else 2.5, max(
            (len(labels[n["node_id"]].splitlines()) * line_height + .4
             for n in records if n["type"] == kind), default=.8,
        )) for kind in ("split", "leaf")
    }
    nodes, positions, contours = _tree_layout(records, sizes)
    left = min(span[0] for span in contours)
    right = max(span[1] for span in contours)
    step = max(size[1] for size in sizes.values()) + .8
    depth = max(n["depth"] for n in records)
    height = depth * step + sizes["split"][1] + 2
    tree_width = right - left + 1
    if training_only:
        figure, axis = plt.subplots(figsize=(tree_width, height))
    else:
        require(patient is not None, "Prediction view requires a selected patient")
        figure, (axis, steps_axis) = plt.subplots(
            1, 2, figsize=(tree_width + 14.5, height),
            gridspec_kw={"width_ratios": [tree_width, 14.5]},
        )
    try:
        for node_id, node in nodes.items():
            x, d = positions[node_id]
            y = -d * step
            w, h = sizes[node["type"]]
            active = node_id in visited
            axis.add_patch(FancyBboxPatch(
                (x - w / 2, y - h / 2), w, h,
                boxstyle="round,pad=0,rounding_size=0.06",
                facecolor=("#D1EEDB" if node_id == leaf_id or
                           (training_only and node["type"] == "leaf")
                           else "#DCEEFF")
                if active else "#F2F2F2",
                edgecolor="#176B9C" if active else "#B8B8B8",
                linewidth=1 if training_only else (2 if active else .8), zorder=3,
            ))
            axis.text(x, y, labels[node_id], ha="center", va="center",
                      fontsize=10, linespacing=1.25,
                      fontweight="bold" if active and not training_only else "normal",
                      color="#143C53" if active else "#808080", zorder=4)
            if node["type"] == "leaf":
                continue
            for direction in ("left", "right"):
                child = node[direction]
                cx, cd = positions[child]
                start = (x, y - h / 2)
                end = (cx, -cd * step + sizes[nodes[child]["type"]][1] / 2)
                active_edge = (node_id, child) in path_edges
                color = "#176B9C" if active_edge or training_only else "#C6C6C6"
                axis.add_patch(FancyArrowPatch(
                    start, end, arrowstyle="-", linewidth=3 if active_edge else .8,
                    color=color, zorder=1,
                ))
                edge_label = "<=" if direction == "left" else ">"
                if direction == node["missing_direction"]:
                    edge_label += " / missing"
                axis.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2,
                          edge_label, fontsize=10,
                          color=color, ha="center",
                          bbox=dict(facecolor="white", edgecolor="none"), zorder=2)
        axis.set_xlim(left - .5, right + .5)
        axis.set_ylim(-depth * step - max(h for _, h in sizes.values()) / 2 - .3,
                      sizes["split"][1] / 2 + .3)
        axis.axis("off")
        title = (
            "Learned Dynamic ON tree — final selected splits\n"
            "Available anchors BEFORE split: [P] predefined, [D] dynamic"
        ) if training_only else (
            f"Prediction for new test patient {patient['patient_id']}\n"
            "Tree structure and split conditions are fixed after training; "
            "timestamps are generated from this patient's observations "
            "using the training rules."
        )
        figure.suptitle(title, fontsize=11)
        figure.subplots_adjust(top=.91, bottom=.035, left=.015, right=.985,
                               wspace=.07)
        if not training_only:
            axis.set_title("A. Fixed trained tree and patient path", loc="left",
                           fontsize=12, pad=16)
            plot_prediction_steps(steps_axis, records, path, patient, leaf_id)
        filename = (
            "learned_tree_anchors.png" if training_only else "prediction_path.png"
        )
        figure.savefig(output / filename, dpi=300,
                       bbox_inches="tight",
                       facecolor="white")
    finally:
        plt.close(figure)
    return leaf_id


def main() -> None:
    """Prepare 110 patients, fit once, predict once, and save the demonstration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        help="Reuse a saved original Experiment 2 v2 dataset")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    plot_mode = parser.add_mutually_exclusive_group()
    plot_mode.add_argument("--plot-training-summary", type=Path,
                          help="Plot an existing training_anchor_summary.json only; "
                          "save beside it, without training or prediction")
    plot_mode.add_argument("--plot-prediction-results", type=Path,
                          help="Plot tree plus patient steps from a saved results "
                          "directory, without training or prediction")
    parser.add_argument("--overwrite-plot", action="store_true",
                        help="Replace the requested PNG in either plot-only mode")
    args = parser.parse_args()
    if args.overwrite_plot and not (
        args.plot_training_summary or args.plot_prediction_results
    ):
        parser.error("--overwrite-plot requires a plot-only mode")
    if args.plot_prediction_results:
        output = args.plot_prediction_results
        if (output / "prediction_path.png").exists() and not args.overwrite_plot:
            parser.error("prediction_path.png already exists; "
                         "use --overwrite-plot to replace it")
        records = json.loads(
            (output / "training_anchor_summary.json").read_text()
        )["nodes"]
        traces = json.loads((output / "test_prediction_paths.json").read_text())
        predictions = pd.read_csv(output / "test_predictions.csv",
                                  dtype={"patient_id": str})
        require(len(predictions) == 10 and predictions.patient_id.is_unique,
                "Expected predictions for exactly ten distinct test patients")
        require(set(predictions.patient_id) == set(traces),
                "Prediction patients and recorded traces differ")
        pid, reason = choose_patient(traces)
        patient = predictions.set_index("patient_id", drop=False).loc[pid]
        plot_prediction_path(records, traces[pid], patient, reason, output)
        print(f"Highlighted patient: {pid}. {reason}")
        print(f"Saved: {(output / 'prediction_path.png').resolve()}")
        return
    if args.plot_training_summary:
        output = args.plot_training_summary.parent
        if (output / "learned_tree_anchors.png").exists() and not args.overwrite_plot:
            parser.error("learned_tree_anchors.png already exists; "
                         "use --overwrite-plot to replace it")
        records = json.loads(args.plot_training_summary.read_text())["nodes"]
        plot_prediction_path(records, [], None, "", output, training_only=True)
        print(f"Saved: {(output / 'learned_tree_anchors.png').resolve()}")
        return
    if args.output.exists():
        parser.error("Output already exists; use --output with a new directory")

    data, patients, validation = (
        load_dataset(args.data_dir) if args.data_dir else generate_data()
    )
    cohort = select_patients(patients)
    data = data.loc[data.patient_id.isin(cohort.patient_id)].copy()
    training = data.loc[data.patient_id.isin(
        cohort.loc[cohort.partition.eq("train"), "patient_id"])]
    test = data.loc[data.patient_id.isin(
        cohort.loc[cohort.partition.eq("test"), "patient_id"])]
    labels = cohort.set_index("patient_id").loc[
        training.patient_id.unique(), "FA"
    ].reset_index(drop=True)
    config = deepcopy(SHARED_MODEL_CONFIG)
    config.update(min_samples_split=10, min_samples_leaf=5)
    model = make_model(True, config)
    args.output.mkdir(parents=True, exist_ok=False)
    cohort.to_csv(args.output / "patients_and_split.csv", index=False)
    data.to_csv(args.output / "observations.csv", index=False)
    effective = deepcopy(config)
    effective["aggregation_funcs"] = {"sum": "numpy.sum"}
    write_json(args.output / "configuration.json", dict(
        model=effective, use_dynamic_anchors=True, selection_seed=SEED,
        source=str(args.data_dir.resolve()) if args.data_dir else
        "generate_fish_intro_dataset.generate_data (fixed seed 42)",
        source_validation=validation,
        train_patients=100, test_patients=10, label_threshold=LABEL_THRESHOLD,
        selection="10 complete training blocks and 1 complete original test block",
        min_samples_reason="10 permits splitting nodes with 10% of training patients; "
        "5 prevents children with fewer than 5% (five patients).",
    ))
    records = fit_with_anchor_capture(model, training, labels)
    print_training_summary(records)
    write_json(args.output / "training_anchor_summary.json", dict(
        node_id_convention="Preorder IDs; source_node_id is the separate persisted "
        "BFS split ID used inside dynamic anchor names.",
        semantics="Available anchor names are candidate references. [] means no "
        "timestamp for that patient. Leaf entries record incoming references; "
        "they have no selected split (stopping or no valid candidate).",
        nodes=records,
    ))
    plot_prediction_path(records, [], None, "", args.output, training_only=True)
    scores, traces = predict_with_capture(model, test, records)
    predictions = cohort.set_index("patient_id").loc[
        test.patient_id.unique(), ["FA"]
    ].reset_index()
    predictions["score"] = scores
    predictions["predicted_label"] = (scores >= LABEL_THRESHOLD).astype(int)
    predictions.to_csv(args.output / "test_predictions.csv", index=False)
    metrics = evaluate(predictions)
    truth, predicted = predictions.FA, predictions.predicted_label
    metrics.update(
        accuracy=float(accuracy_score(truth, predicted)),
        precision=float(precision_score(truth, predicted, zero_division=0)),
        recall=float(recall_score(truth, predicted, zero_division=0)),
        f1=float(f1_score(truth, predicted, zero_division=0)),
        confusion_matrix=confusion_matrix(truth, predicted, labels=[0, 1]).tolist(),
        confusion_matrix_order="rows=true, columns=predicted, labels=[0, 1]",
        label_threshold=LABEL_THRESHOLD,
    )
    write_json(args.output / "test_metrics.json", metrics)
    write_json(args.output / "test_prediction_paths.json", traces)
    print("\nTEST PREDICTIONS (score = P(FA=1))")
    print(predictions.to_string(index=False))
    print("\nTEST METRICS")
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "brier",
                "confusion_matrix"):
        print(f"  {key}: {metrics[key]}")
    pid, reason = choose_patient(traces)
    patient = predictions.set_index("patient_id", drop=False).loc[pid]
    leaf_id = plot_prediction_path(records, traces[pid], patient, reason, args.output)
    require(np.isclose(records[leaf_id]["score"], patient["score"]),
            "Highlighted leaf probability differs from actual prediction")
    write_json(args.output / "selected_patient.json", dict(
        patient_id=pid, reason=reason, leaf_node_id=leaf_id,
        true_label=int(patient["FA"]), predicted_label=int(patient["predicted_label"]),
        probability=float(patient["score"]), path=traces[pid],
    ))
    print(f"\nHighlighted patient: {pid}. {reason}")
    print(f"Outputs: {args.output.resolve()}")


if __name__ == "__main__":
    main()
