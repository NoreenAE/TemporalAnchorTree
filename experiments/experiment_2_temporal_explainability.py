"""Saved-artifact explainability entry point for final Experiment 2.

No model is deserialized or invoked. Run the current-result audit with:
    python -B -m experiments.experiment_2_temporal_explainability --verify-current

Automatic generation now compares source references before/at and after first
fish-anchor creation in anchor-created TEST patients. Earlier monthly histogram
calculation helpers remain callable for reproducibility.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from TAT.temporal_explainability import (
    PATH_TABLE_COLUMNS, aggregate_temporal_usage,
    calculate_temporal_usage_percentages, derive_monthly_bin_edges,
)
from experiments.experiment_utils import artifact_path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "experiments/results/experiment_2_noisy_matched_v2"
FISH = "FishIntroduction dynamic anchor"
BIRTH = "Birth Date anchor"
FULL = "Full Timeline"
SYMPTOMS = ("GI", "Vomiting")
INPUTS = (
    "configuration.json", "patients_and_split.csv",
    "dynamic_on/tree.json", "dynamic_on/patient_predictions.csv",
    "dynamic_on/anchor_trace.json",
)


def require(condition: bool, message: str) -> None:
    """Reject incomplete or unsupported evidence before publishing outputs."""
    if not condition:
        raise ValueError(message)


def read_json(path: Path):
    """Read a saved JSON artifact without loading executable objects."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_test_visits(results: Path, *, allow_other: bool = False
                     ) -> tuple[list[dict], dict, int]:
    """Verify complete saved TEST paths and classify window source anchors."""
    missing = [str(artifact_path(results, name)) for name in INPUTS
               if not artifact_path(results, name).is_file()]
    if missing:
        raise FileNotFoundError("Missing saved artifacts:\n" + "\n".join(missing))
    configuration = read_json(results / "configuration.json")
    require(configuration["dataset_version"] == "fish_intro_noisy_matched_v2",
            "Expected the final noisy matched V2 experiment")
    patients = pd.read_csv(artifact_path(results, "patients_and_split.csv"),
                           dtype={"patient_id": str})
    predictions = pd.read_csv(
        results / "dynamic_on/patient_predictions.csv",
        dtype={"patient_id": str}, float_precision="round_trip",
    )
    metadata = patients.set_index("patient_id", verify_integrity=True)
    scores = predictions.set_index("patient_id", verify_integrity=True)
    require(set(metadata.index) == set(scores.index), "Prediction cohort mismatch")
    for column in ("partition", "FA", "documented"):
        require(metadata[column].sort_index().equals(scores[column].sort_index()),
                f"Saved prediction/metadata mismatch: {column}")
    test_ids = set(metadata.index[metadata.partition.eq("test")])
    require(bool(test_ids), "No held-out TEST patients")
    tree = read_json(results / "dynamic_on/tree.json")
    nodes = {node["node_id"]: node for node in tree}
    require(len(nodes) == len(tree), "Duplicate saved tree node IDs")
    children = {node[side] for node in tree if node["type"] == "split"
                for side in ("left", "right")}
    roots = set(nodes) - children
    require(len(roots) == 1 and children.issubset(nodes), "Invalid saved tree")
    root = next(iter(roots))
    traces = read_json(results / "dynamic_on/anchor_trace.json")
    creators = {}
    for trace in traces:
        name = trace.get("anchor_name")
        if name:
            require(name not in creators or creators[name] == trace["feature"],
                    f"Conflicting saved anchor source: {name}")
            creators[name] = trace["feature"]
    require(metadata.documented.isin([True, False]).all(),
            "Missing or invalid documentation status")
    paths = {pid: [] for pid in sorted(test_ids)}
    for trace in traces:
        if trace["patient_id"] in paths:
            paths[trace["patient_id"]].append(trace)
    visits = []
    for pid, path in paths.items():
        expected, seen, created_anchors = root, set(), {}
        for trace in path:
            node_id = trace["node_id"]
            require(node_id == expected and node_id not in seen,
                    f"Incomplete or duplicate saved path: {pid}")
            seen.add(node_id)
            node = nodes[node_id]
            require(node["type"] == "split"
                    and node["feature"] == trace["feature"]
                    and node["window"] == [trace["source_anchor"], trace["offset"]],
                    f"Saved trace/tree mismatch: {pid}/{node_id}")
            require(all(trace["checks"].values()),
                    f"Saved prediction audit failed: {pid}/{node_id}")
            anchor, offset = node["window"]
            if offset is None:
                require(anchor is None, "Expected canonical full-timeline window")
                reference = FULL
            elif anchor == "Birth Date Anchor":
                reference = BIRTH
            elif creators.get(anchor) == "FishIntroduction":
                reference = FISH
            elif allow_other:
                reference = f"Other: {anchor}"
            else:
                raise ValueError(
                    f"Unsupported window source {anchor!r}; cannot represent it "
                    "honestly in the requested three reference panels")
            windows = trace["evaluated_windows"]
            for start, end in windows:
                require(pd.Timestamp(end) == pd.Timestamp(start)
                        + pd.DateOffset(months=offset),
                        f"Saved calendar-window mismatch: {pid}/{node_id}")
            available = reference == FULL or bool(windows)
            if reference == FISH and available:
                require(anchor in created_anchors and all(
                    pd.Timestamp(start) in created_anchors[anchor]
                    for start, _ in windows),
                    f"Fish anchor use without prior saved creation: {pid}/{node_id}")
            fish_created = (trace["feature"] == "FishIntroduction"
                            and bool(trace.get("anchor_timestamps")))
            require(trace.get("fish_anchor_created", fish_created) == fish_created,
                    f"Inconsistent saved fish creation flag: {pid}/{node_id}")
            if fish_created:
                require(bool(metadata.loc[pid, "documented"]),
                        f"Fish anchor created for undocumented patient: {pid}")
                require(bool(trace.get("anchor_name")), "Missing created anchor name")
                created_anchors[trace["anchor_name"]] = {
                    pd.Timestamp(time) for time in trace["anchor_timestamps"]}
            visits.append(dict(
                patient_id=pid, node_id=node_id, anchor=reference,
                direction=trace["direction"],
                feature=trace["feature"],
                source_anchor=anchor,
                created_anchor_name=trace.get("anchor_name"),
                created_anchor_timestamps=trace.get("anchor_timestamps", []),
                documented=bool(metadata.loc[pid, "documented"]),
                fish_anchor_created=fish_created,
                stored_offset=offset,
                effective_start_offset=min(0, offset)
                if offset is not None and available else np.nan,
                effective_end_offset=max(0, offset)
                if offset is not None and available else np.nan,
                is_full_timeline=reference == FULL,
                is_zero_width=offset == 0, anchor_available=available,
            ))
            require(trace["direction"] in ("left", "right"), "Invalid direction")
            expected = node[trace["direction"]]
        require(nodes[expected]["type"] == "leaf",
                f"Saved trace does not reach a leaf: {pid}")
        require(nodes[expected]["score"] == float(scores.loc[pid, "score"]),
                f"Terminal leaf differs from saved prediction: {pid}")
    return visits, configuration["parameters"], len(test_ids)


def calculate_usage(results: Path) -> tuple[pd.DataFrame, dict, dict]:
    """Reuse Experiment 1's overlap counting and all-split-visit denominator."""
    visits, parameters, patient_count = load_test_visits(results)
    return aggregate_reference_usage(visits, parameters, patient_count)


def aggregate_reference_usage(visits: list[dict], parameters: dict,
                              patient_count: int) -> tuple[pd.DataFrame, dict, dict]:
    """Aggregate validated reference visits without reading artifacts twice."""
    require(bool(visits), "No split visits: percentage denominator is undefined")
    offsets = {
        FISH: list(parameters["anchor_offset_list"]),
        BIRTH: list(parameters["anchor_offset_map"]["Birth Date Anchor"]),
    }
    # Include observed offsets as well as configured candidate ranges. Future
    # saved trees therefore need not share the current selected windows.
    for visit in visits:
        if not visit["is_full_timeline"]:
            offsets[visit["anchor"]].append(visit["stored_offset"])
    edges = derive_monthly_bin_edges(offsets)
    # Only the aggregation-consumed fields are populated. Unused path-schema
    # fields remain NA; no missing model values or patient paths are invented.
    paths = pd.DataFrame(visits).reindex(columns=PATH_TABLE_COLUMNS)
    usage, summary = aggregate_temporal_usage(paths, edges, patient_count)
    percentages, full_percentage = calculate_temporal_usage_percentages(
        usage, summary)
    rows = [dict(
        reference_type=row.anchor,
        temporal_bin=f"{row.bin_start_offset:g}–{row.bin_end_offset:g}",
        bin_start_months=row.bin_start_offset,
        bin_end_months=row.bin_end_offset,
        visit_count=int(row.raw_usage_count),
        total_split_visits=summary["temporal_split_usages"],
        usage_percentage=row.usage_percentage,
    ) for row in percentages.itertuples(index=False)]
    rows.append(dict(
        reference_type=FULL, temporal_bin=FULL,
        bin_start_months=None, bin_end_months=None,
        visit_count=summary["full_timeline_usages"],
        total_split_visits=summary["temporal_split_usages"],
        usage_percentage=full_percentage,
    ))
    summary.update(
        cohort="test", model="Dynamic Anchoring ON",
        reference_visit_counts={reference: sum(
            visit["anchor"] == reference for visit in visits)
            for reference in (FISH, BIRTH, FULL)},
        unavailable_reference_visits=dict(Counter(
            visit["anchor"] for visit in visits
            if not visit["anchor_available"])),
        definition="Each visited TEST patient–split contributes once to every "
                   "monthly bin overlapped by its available window; percentages "
                   "divide by all TEST patient–split visits. Empty-event windows "
                   "count; unavailable references do not overlap bins.",
        experiment_1_comparison="Same counting rule and denominator definition; "
                                "Experiment 1 FA used training patients.",
    )
    require(sum(summary["reference_visit_counts"].values())
            == summary["temporal_split_usages"], "Reference accounting mismatch")
    return pd.DataFrame(rows), summary, edges


def calculate_progression(results: Path) -> tuple[dict[str, pd.DataFrame], dict]:
    """Keep the three panels' distinct units and denominators explicit."""
    visits, parameters, patient_count = load_test_visits(results)
    reference_table, summary, edges = aggregate_reference_usage(
        visits, parameters, patient_count)
    birth = reference_table.loc[
        reference_table.reference_type.eq(BIRTH)].copy()
    birth["unit"] = "percent of all TEST patient–split visits"
    documented = {v["patient_id"] for v in visits if v["documented"]}
    created = {v["patient_id"] for v in visits if v["fish_anchor_created"]}
    fish_visits = [v for v in visits if v["anchor"] == FISH
                   and v["anchor_available"]]
    available = {v["patient_id"] for v in fish_visits}
    used = {v["patient_id"] for v in fish_visits if v["feature"] in SYMPTOMS}
    require(used.issubset(created) and created.issubset(documented),
            "Invalid documented/created/symptom-used patient progression")
    require(bool(documented), "No documented TEST patients: Panel B undefined")
    stages = pd.DataFrame([dict(
        stage=stage, patient_count=len(ids),
        documented_test_patients=len(documented),
        usage_percentage=100 * len(ids) / len(documented),
        unit="unique patients; percent of documented TEST patients",
    ) for stage, ids in (
        ("Documented FishIntroduction", documented),
        ("Anchor created", created),
        ("Anchor used for symptom split", used),
    )])
    symptom_rows = []
    for feature in SYMPTOMS:
        for start, end in zip(edges[FISH][:-1], edges[FISH][1:]):
            eligible = [v for v in fish_visits if v["feature"] == feature
                        and v["effective_start_offset"] < end
                        and v["effective_end_offset"] > start]
            unique_patients = {v["patient_id"] for v in eligible}
            symptom_rows.append(dict(
                feature=feature, temporal_bin=f"{start:g}–{end:g}",
                bin_start_months=start, bin_end_months=end,
                visit_count=len(eligible), unique_patient_count=len(unique_patients),
                repeated_patient_contributions=len(eligible) - len(unique_patients),
                unit="patient–split visits", denominator="none; absolute count",
            ))
    symptoms = pd.DataFrame(symptom_rows)
    other_features = Counter(v["feature"] for v in fish_visits
                             if v["feature"] not in SYMPTOMS)
    selected_windows = []
    for feature, offset in sorted({(v["feature"], v["stored_offset"])
                                   for v in fish_visits
                                   if v["feature"] in SYMPTOMS}):
        window_visits = [v for v in fish_visits if v["feature"] == feature
                         and v["stored_offset"] == offset]
        selected_windows.append(dict(
            feature=feature, offset_months=offset, visit_count=len(window_visits),
            unique_patient_count=len({v["patient_id"] for v in window_visits}),
        ))
    summary.update(
        reference_counting_definition=summary["definition"],
        definition="Panel A: temporal split usage as a percentage of all TEST "
                   "split visits. Panel B: unique-patient stages as percentages "
                   "of documented TEST patients. Panel C: absolute symptom "
                   "patient–split visit counts; no percentage denominator.",
        experiment_1_comparison="Panel A uses the Experiment 1 counting rule and "
                                "denominator definition; Experiment 1 FA used "
                                "training patients. Panels B/C use distinct units.",
        documented_fish_introduction_patients=len(documented),
        fish_anchor_created_patients=len(created),
        fish_anchor_available_at_downstream_split_patients=len(available),
        fish_anchor_used_for_symptom_patients=len(used),
        patient_progression_unit="unique documented TEST patients",
        patient_progression_denominator=len(documented),
        symptom_features_displayed=list(SYMPTOMS),
        other_available_fish_relative_feature_visits=dict(other_features),
        symptom_window_usage=symptom_rows,
        symptom_selected_windows=selected_windows,
        symptom_counts_equal_unique_patients=bool(
            symptoms.repeated_patient_contributions.eq(0).all()),
        panel_definitions={
            "A": "Birth Date window overlap counts / all TEST split visits × 100",
            "B": "Unique patients / documented FishIntroduction TEST patients × 100; "
                 "used means a later available fish-relative GI or Vomiting split, "
                 "including windows containing no symptom events",
            "C": "Absolute GI/Vomiting patient–split visits overlapping each monthly "
                 "fish-relative bin; a patient may contribute multiple visits",
        },
    )
    return {"birth_date_temporal_usage": birth,
            "fish_anchor_patient_usage": stages,
            "fish_relative_symptom_usage": symptoms}, summary


def plot_usage(tables: dict[str, pd.DataFrame], summary: dict, output: Path) -> None:
    """Show absolute windows, patient anchoring, and subsequent symptom windows."""
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42}):
        figure, axes = plt.subplots(
            1, 3, figsize=(19, 5.8), gridspec_kw={"width_ratios": [1.5, 1, 1]},
        )
        try:
            birth = tables["birth_date_temporal_usage"]
            axes[0].bar(np.arange(len(birth)), birth.usage_percentage,
                        color="#517C9B", width=.8)
            axes[0].set_xticks(np.arange(len(birth)))
            axes[0].set_xticklabels(birth.temporal_bin, rotation=55,
                                   ha="right", fontsize=8)
            axes[0].set_ylim(0, 100)
            axes[0].set_xlabel("Age in calendar months")
            axes[0].set_ylabel("Temporal split usage (%)")
            stages = tables["fish_anchor_patient_usage"]
            bars = axes[1].bar(np.arange(len(stages)), stages.usage_percentage,
                               color=["#517C9B", "#419A85", "#419A85"], width=.6)
            axes[1].set_xticks(np.arange(len(stages)))
            axes[1].set_xticklabels([
                "Documented\nFishIntroduction", "Anchor\ncreated",
                "Anchor used for\nsymptom split"], fontsize=9)
            axes[1].set_ylabel("Documented TEST patients (%)")
            axes[1].set_ylim(0, 115)
            axes[1].set_yticks([0, 20, 40, 60, 80, 100])
            for bar, row in zip(bars, stages.itertuples(index=False)):
                axes[1].annotate(
                    f"{row.patient_count:,} ({row.usage_percentage:g}%)",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=9,
                )
            symptoms = tables["fish_relative_symptom_usage"]
            bins = symptoms.loc[symptoms.feature.eq(SYMPTOMS[0]), "temporal_bin"]
            x = np.arange(len(bins))
            for index, (feature, color) in enumerate(zip(
                    SYMPTOMS, ("#2166AC", "#B35806"))):
                values = symptoms.loc[symptoms.feature.eq(feature), "visit_count"]
                bars = axes[2].bar(x + (index - .5) * .34, values,
                                   width=.34, color=color, label=feature)
                for bar, value in zip(bars, values):
                    if value:
                        axes[2].annotate(
                            str(value), (bar.get_x() + bar.get_width() / 2, value),
                            xytext=(0, 4), textcoords="offset points",
                            ha="center", fontsize=9,
                        )
            axes[2].set_xticks(x)
            axes[2].set_xticklabels(bins, fontsize=9)
            axes[2].set_xlabel("Months relative to FishIntroduction")
            axes[2].set_ylabel("Patient–split visits")
            axes[2].set_ylim(0, max(1, symptoms.visit_count.max()) * 1.2)
            axes[2].legend(frameon=False, fontsize=9)
            titles = (
                "A. Birth Date-relative temporal usage",
                "B. FishIntroduction anchor creation and use",
                "C. Symptom usage relative to FishIntroduction",
            )
            for axis, title in zip(axes, titles):
                axis.set_title(title, fontsize=11, loc="left", pad=14)
                axis.grid(axis="y", alpha=.2)
                axis.set_axisbelow(True)
                for spine in ("top", "right"):
                    axis.spines[spine].set_visible(False)
            figure.text(
                .5, .03,
                f"Held-out TEST patients: {summary['evaluated_patient_count']:,}"
                f"  |  Panel A: {summary['temporal_split_usages']:,} split visits"
                f"  |  Panel B: {summary['patient_progression_denominator']:,}"
                " documented patients  |  Panel C: absolute visit counts",
                ha="center", fontsize=10,
            )
            figure.subplots_adjust(left=.05, right=.985, bottom=.25,
                                   top=.88, wspace=.32)
            for extension in ("png", "pdf"):
                figure.savefig(output / f"temporal_anchor_progression.{extension}",
                               dpi=300, facecolor="white")
        finally:
            plt.close(figure)


def save_temporal_reference_usage(results: Path, output: Path = None,
                                  *, verify_current: bool = False) -> dict:
    """Generate the current before/after-creation analysis from saved paths.

    Kept as the runner's public entry point. Earlier calculation helpers remain
    callable, but automatic generation now uses the stage-specific analysis.
    """
    from experiments.experiment_2_anchor_transition import save_anchor_transition

    return save_anchor_transition(results, output, verify_current=verify_current)


def main() -> None:
    """Post-process saved final Experiment 2 artifacts without a model run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-current", action="store_true")
    args = parser.parse_args()
    summary = save_temporal_reference_usage(
        args.results, args.output, verify_current=args.verify_current)
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in ("patients", "stage_assignments")},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
