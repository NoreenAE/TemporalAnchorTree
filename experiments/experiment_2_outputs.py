"""Post-process saved Experiment 2 artifacts without loading or running a model."""
from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from TAT.eval_utils import save_roc_curves

RUNS = ("dynamic_on", "dynamic_off")
LABELS = ("Dynamic Anchoring ON", "Dynamic Anchoring OFF")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _tree_label(node: dict) -> str:
    """Display saved conditions without exposing internal anchor identifiers."""
    if node["type"] == "leaf":
        return (f"Patients = {node['patients']:,}\n"
                f"P(positive) = {node['score']:.3f}")
    anchor, offset = node["window"]
    if offset is None:
        window = "Full timeline"
    else:
        match = re.fullmatch(r"Node \d+ \| [^(]+\((.*)\) Anchor", str(anchor))
        name = match.group(1) if match else str(anchor)
        if not match and name.endswith(" Anchor"):
            name = name[:-len(" Anchor")]
        extent = str(int(offset)) if float(offset).is_integer() else str(offset)
        interval = f"0–{extent}" if offset >= 0 else f"{extent}–0"
        relation = "after" if offset >= 0 else "relative to"
        window = (f"Window: {interval} months {relation}\n"
                  + textwrap.fill(name, 30))
    threshold = str(node["threshold"])
    if float(node["threshold"]).is_integer():
        threshold = str(int(node["threshold"]))
    return (textwrap.fill(node["feature"], 30) + "\n"
            + window + "\n"
            + f"{node['aggregation'].capitalize()} ≤ {threshold}")


def _tree_layout(records: list[dict], sizes: dict) -> tuple:
    """Pack subtree contours in physical inches, preserving every saved edge."""
    nodes = {n["node_id"]: n for n in records}
    children = {n[k] for n in records if n["type"] == "split"
                for k in ("left", "right")}
    roots = set(nodes) - children
    if (len(nodes) != len(records) or len(roots) != 1
            or not children.issubset(nodes)):
        raise ValueError("Expected one complete saved tree")
    visited = set()

    def visit(node_id):
        if node_id in visited:
            raise ValueError("Cycle or shared child in saved tree")
        visited.add(node_id)
        node = nodes[node_id]
        half = sizes[node["type"]][0] / 2
        if node["type"] == "leaf":
            return {node_id: (0, 0)}, [(-half, half)]
        left, lc = visit(node["left"])
        right, rc = visit(node["right"])
        distance = max(a[1] - b[0] + .28 for a, b in zip(lc, rc))
        positions = {node_id: (0, 0)}
        for branch, shift in ((left, -distance / 2), (right, distance / 2)):
            positions.update({key: (x + shift, depth + 1)
                              for key, (x, depth) in branch.items()})
        contours = [(-half, half)]
        for depth in range(max(len(lc), len(rc))):
            spans = []
            if depth < len(lc):
                spans.append(tuple(x - distance / 2 for x in lc[depth]))
            if depth < len(rc):
                spans.append(tuple(x + distance / 2 for x in rc[depth]))
            contours.append((min(x[0] for x in spans), max(x[1] for x in spans)))
        return positions, contours

    positions, contours = visit(next(iter(roots)))
    if visited != set(nodes):
        raise ValueError("Unreachable saved tree nodes")
    return nodes, positions, contours


def _tree_figure(trees: dict):
    """Build the combined figure with measured boxes and boundary connectors."""
    labels = {run: {n["node_id"]: _tree_label(n) for n in trees[run]} for run in RUNS}
    # Measure actual font extents; use the same dimensions per node type in
    # both panels. Physical-inch coordinates prevent panel-dependent scaling.
    probe = plt.figure(figsize=(2, 2))
    sizes = {kind: [0., 0.] for kind in ("split", "leaf")}
    try:
        renderer = probe.canvas.get_renderer()
        for run in RUNS:
            for node in trees[run]:
                text = probe.text(0, 0, labels[run][node["node_id"]],
                                  fontsize=11, linespacing=1.25)
                box = text.get_window_extent(renderer)
                size = sizes[node["type"]]
                size[0] = max(size[0], box.width / probe.dpi + .24)
                size[1] = max(size[1], box.height / probe.dpi + .20)
                text.remove()
    finally:
        plt.close(probe)
    layouts = [_tree_layout(trees[run], sizes) for run in RUNS]
    step = max(size[1] for size in sizes.values()) + .65
    widths = [max(b for a, b in c) - min(a for a, b in c) for _, _, c in layouts]
    heights = [(len(c) - 1) * step + max(s[1] for s in sizes.values())
               for _, _, c in layouts]
    width = max(widths) + .6
    height = sum(heights) + 1.8
    figure = plt.figure(figsize=(width, height))
    top = height - .65
    for index, (run, layout, panel_height) in enumerate(zip(RUNS, layouts, heights)):
        nodes, positions, contours = layout
        axis = figure.add_axes([0, (top - panel_height) / height,
                                1, panel_height / height])
        low = min(a for a, b in contours)
        high = max(b for a, b in contours)
        center = (low + high) / 2
        axis.set_xlim(center - width / 2, center + width / 2)
        half_height = max(s[1] for s in sizes.values()) / 2
        axis.set_ylim(-panel_height + half_height, half_height)
        axis.axis("off")
        axis.set_title(f"({'ab'[index]}) {LABELS[index]}", loc="left",
                       x=.02, fontsize=14, fontweight="bold", pad=17)
        for node_id, node in nodes.items():
            x, depth = positions[node_id]
            y = -depth * step
            w, h = sizes[node["type"]]
            color = "#EAF3F8" if node["type"] == "split" else "#EDF4E9"
            # Dynamic-anchor names follow TemporalTree's persisted naming
            # contract. Classify the window reference, not the split feature.
            if node["type"] == "split":
                anchor, offset = node["window"]
                if offset is not None and re.fullmatch(
                    r"Node \d+ \| [^(]+\(.*\) Anchor", str(anchor)
                ):
                    color = "#F1EAF6"
            axis.add_patch(FancyBboxPatch(
                (x - w / 2, y - h / 2), w, h,
                boxstyle="round,pad=0,rounding_size=0.06",
                facecolor=color,
                edgecolor="#455A64", linewidth=.9, zorder=3, clip_on=False))
            axis.text(x, y, labels[run][node_id], ha="center", va="center",
                      fontsize=11, linespacing=1.25, zorder=4)
            if node["type"] == "leaf":
                continue
            for key, sign in (("left", "≤"), ("right", ">")):
                child_id = node[key]
                cx, cd = positions[child_id]
                start = (x, y - h / 2)
                end = (cx, -cd * step + sizes[nodes[child_id]["type"]][1] / 2)
                axis.add_patch(FancyArrowPatch(
                    start, end, arrowstyle="-", shrinkA=2, shrinkB=2,
                    linewidth=1, color="#66757F", zorder=1))
                dx, dy = end[0] - start[0], end[1] - start[1]
                length = (dx * dx + dy * dy) ** .5
                side = 1 if key == "left" else -1
                offset = (side * 11 * dy / length, -side * 11 * dx / length)
                axis.annotate(sign, ((start[0] + end[0]) / 2,
                                     (start[1] + end[1]) / 2),
                              xytext=offset,
                              textcoords="offset points", ha="center", va="center",
                              fontsize=11, zorder=2)
        top -= panel_height + .9
    return figure


def _save_trees(trees: dict, output: Path) -> None:
    with plt.rc_context({"pdf.fonttype": 42, "font.family": "DejaVu Sans"}):
        figure = _tree_figure(trees)
        try:
            for extension in ("png", "pdf"):
                figure.savefig(output / f"decision_trees_on_vs_off.{extension}",
                               dpi=300, bbox_inches="tight", pad_inches=.15,
                               facecolor="white")
        finally:
            plt.close(figure)


def save_tree_comparison(output: Path) -> None:
    """Refresh only the combined PNG/PDF from saved tree JSON files."""
    output = Path(output)
    trees = {run: _read_json(output / run / "tree.json") for run in RUNS}
    _save_trees(trees, output)


def _anchor_export(output: Path, predictions: dict, trees: dict) -> None:
    """Export all held-out dynamic anchors, retaining fish-specific summaries."""
    patient_rows, visits, creations = [], [], []
    summary = {
        "schema_version": 2,
        "definitions": {
            "scope": "Creation/window rows include every dynamic-anchor feature. "
                     "Filter anchor_feature=FishIntroduction for fish-only rows.",
            "legacy_cohort_fields": "created_not_used, documented_no_anchor and "
                                    "no_anchor_created remain FishIntroduction-specific.",
            "dynamic_anchor_used": "An actual downstream visit has an available "
                                   "dynamic-anchor window, even if it contains no events.",
            "created": "At least one nonempty FishIntroduction-derived anchor timestamp.",
            "used": "A visited downstream split references a fish-derived anchor "
                    "and has a nonempty evaluated_windows list; no event need occur in it.",
            "symptom_used": "Used as above, restricted to GI or Vomiting.",
            "visit_rows": "Only actual held-out prediction visits; one row per "
                          "evaluated window, or one blank-window row if unavailable.",
        },
        "observations_file": "observations.csv",
        "patient_metadata_file": "patients_and_split.csv",
        "dataset_reference": "configuration.json:dataset_directory",
        "join_key": "patient_id",
        "runs": {},
    }
    visit_columns = ["run", "patient_id", "node_id", "source_node_id", "feature",
                     "aggregation", "source_anchor", "offset_months", "direction",
                     "visited", "anchor_available", "window_start", "window_end",
                     "eligible_symptom_count", "anchor_feature"]
    creation_columns = ["run", "patient_id", "node_id", "source_node_id",
                        "anchor_name", "anchor_timestamp", "anchor_feature"]
    for run in RUNS:
        patients = predictions[run]
        ids = set(patients.index)
        nodes = {n["node_id"]: n for n in trees[run]}
        traces = [t for t in _read_json(output / run / "anchor_trace.json")
                  if t["patient_id"] in ids]
        anchor_features = {t["anchor_name"]: t["feature"] for t in traces
                           if t["anchor_name"]}
        all_created, all_used = set(), set()
        created_pairs, used_pairs = set(), set()
        created, used, symptom_used = set(), set(), set()
        usage_groups = {}
        for trace in traces:
            pid = trace["patient_id"]
            if trace["anchor_name"]:
                for timestamp in trace["anchor_timestamps"]:
                    all_created.add(pid)
                    created_pairs.add((pid, trace["anchor_name"]))
                    if trace["feature"] == "FishIntroduction":
                        created.add(pid)
                    creations.append(dict(run=run, patient_id=pid,
                                          node_id=trace["node_id"],
                                          source_node_id=trace["source_node_id"],
                                          anchor_name=trace["anchor_name"],
                                          anchor_timestamp=timestamp,
                                          anchor_feature=trace["feature"]))
            if trace["source_anchor"] not in anchor_features:
                continue
            anchor_feature = anchor_features[trace["source_anchor"]]
            windows = trace["evaluated_windows"]
            aggregation = nodes[trace["node_id"]]["aggregation"]
            if windows:
                all_used.add(pid)
                used_pairs.add((pid, trace["source_anchor"]))
                if anchor_feature == "FishIntroduction":
                    used.add(pid)
                    if trace["feature"] in ("GI", "Vomiting"):
                        symptom_used.add(pid)
                key = (trace["source_anchor"], trace["feature"], aggregation,
                       trace["offset"])
                usage_groups.setdefault(key, set()).add(pid)
            for start, end in windows or [(None, None)]:
                visits.append(dict(
                    run=run, patient_id=pid, node_id=trace["node_id"],
                    anchor_feature=anchor_feature,
                    source_node_id=trace["source_node_id"], feature=trace["feature"],
                    aggregation=aggregation, source_anchor=trace["source_anchor"],
                    offset_months=trace["offset"], direction=trace["direction"],
                    visited=True, anchor_available=bool(windows),
                    window_start=start, window_end=end,
                    eligible_symptom_count=trace["eligible_symptom_count"],
                ))
        if not used_pairs.issubset(created_pairs):
            raise ValueError(f"{run}: dynamic anchor use without recorded creation")
        if not used.issubset(created):
            raise ValueError(f"{run}: fish anchor use without recorded creation")
        cohort = patients[["documented", "birth", "fish_time"]].copy()
        cohort["dynamic_anchor_created"] = cohort.index.isin(all_created)
        cohort["dynamic_anchor_used"] = cohort.index.isin(all_used)
        cohort["fish_anchor_created"] = cohort.index.isin(created)
        cohort["fish_anchor_used"] = cohort.index.isin(used)
        cohort["fish_anchor_used_for_symptom"] = cohort.index.isin(symptom_used)
        cohort["created_not_used"] = cohort.index.isin(created - used)
        cohort["documented_no_anchor"] = cohort.documented & ~cohort.fish_anchor_created
        cohort.insert(0, "run", run)
        patient_rows.extend(cohort.reset_index().to_dict("records"))
        groups = {}
        for name, subset in (("all_test", cohort),
                             ("documented", cohort[cohort.documented]),
                             ("undocumented", cohort[~cohort.documented])):
            groups[name] = {"denominator": len(subset)}
            for flag in ("fish_anchor_created", "fish_anchor_used",
                         "fish_anchor_used_for_symptom", "created_not_used",
                         "documented_no_anchor", "dynamic_anchor_created",
                         "dynamic_anchor_used"):
                groups[name][flag] = int(subset[flag].sum())
            groups[name]["no_anchor_created"] = int((~subset.fish_anchor_created).sum())
        summary["runs"][run] = dict(
            cohorts=groups,
            dynamic_anchors=[dict(
                anchor_name=name, anchor_feature=feature,
                created_children=sum(anchor == name for _, anchor in created_pairs),
                used_children=sum(anchor == name for _, anchor in used_pairs),
                created_not_used_children=len(
                    {(pid, anchor) for pid, anchor in created_pairs - used_pairs
                     if anchor == name}),
                denominator=len(cohort),
            ) for name, feature in anchor_features.items()],
            downstream_usage=[dict(source_anchor=k[0], feature=k[1],
                                   anchor_feature=anchor_features[k[0]],
                                   aggregation=k[2], offset_months=k[3],
                                   unique_test_children=len(v),
                                   denominator=len(cohort))
                              for k, v in usage_groups.items()],
        )
    pd.DataFrame(patient_rows).to_csv(output / "anchor_usage_test_patients.csv", index=False)
    pd.DataFrame(creations, columns=creation_columns).to_csv(
        output / "anchor_usage_test_creations.csv", index=False)
    pd.DataFrame(visits, columns=visit_columns).to_csv(
        output / "anchor_usage_test_windows.csv", index=False)
    (output / "anchor_usage_test_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_experiment_outputs(output: Path) -> None:
    """Save combined figures and anchor exports after both model runs finish.

    Reads only CSV/JSON artifacts. Never unpickles models or predicts again.
    Patient windows and anchor timestamps join to the existing observations.csv
    using patient_id; metadata and generating-rule fields remain in the snapshot.
    """
    output = Path(output)
    predictions, trees = {}, {}
    for run in RUNS:
        frame = pd.read_csv(output / run / "patient_predictions.csv",
                            dtype={"patient_id": str, "documented": bool})
        predictions[run] = frame.loc[frame.partition.eq("test")].set_index(
            "patient_id", verify_integrity=True).sort_index()
        trees[run] = _read_json(output / run / "tree.json")
    on, off = (predictions[run] for run in RUNS)
    if on.empty or not on.index.equals(off.index) or not on.FA.equals(off.FA):
        raise ValueError("ON/OFF must have identical nonempty held-out IDs and labels")
    curves = {label: (predictions[run].FA, predictions[run].score,
                      roc_auc_score(predictions[run].FA, predictions[run].score))
              for run, label in zip(RUNS, LABELS)}
    _anchor_export(output, predictions, trees)
    for extension in ("png", "pdf"):
        save_roc_curves(curves, output / f"roc_on_vs_off.{extension}",
                        title="Experiment 2 — Held-out ROC comparison")
    _save_trees(trees, output)
