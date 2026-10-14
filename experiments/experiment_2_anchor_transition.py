"""Compare saved temporal references before and after fish-anchor creation.

Analysis is restricted to anchor-created TEST patients. The first recorded
FishIntroduction anchor creation defines each patient's boundary in traversal
order. The creating split is included before/at creation, never after it.
No trained model is deserialized or invoked.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.experiment_2_temporal_explainability import (
    BIRTH, FISH, FULL, INPUTS, load_test_visits, require,
)

BEFORE = "before_at_creation"
AFTER = "after_creation"
LABELS = {BIRTH: "Birth Date", FISH: "FishIntroduction", FULL: "Full Timeline"}


def split_creation_stages(visits: list[dict]) -> tuple[list[dict], list[dict]]:
    """Assign every included visit once using each patient's first creation."""
    paths = defaultdict(list)
    for visit in visits:
        paths[visit["patient_id"]].append(visit)
    records, patients = [], []
    for pid, path in sorted(paths.items()):
        creation_indices = [i for i, visit in enumerate(path)
                            if visit["fish_anchor_created"]]
        if not creation_indices:
            continue
        boundary = creation_indices[0]
        creation = path[boundary]
        require(creation["documented"], "Included patient lacks documented fish")
        post_references = sorted({LABELS.get(v["anchor"], v["anchor"])
                                  for v in path[boundary + 1:]})
        patients.append(dict(
            patient_id=pid, partition="test", documented=True,
            creation_visit_index=boundary, creation_node_id=creation["node_id"],
            creation_is_root=boundary == 0,
            created_anchor_name=creation["created_anchor_name"],
            created_anchor_timestamps=creation["created_anchor_timestamps"],
            fish_creation_visit_indices=creation_indices,
            before_at_creation_visit_count=boundary + 1,
            after_creation_visit_count=len(path) - boundary - 1,
            pre_anchor_path=[dict(node_id=v["node_id"], direction=v["direction"])
                             for v in path[:boundary + 1]],
            post_anchor_references=post_references,
        ))
        for index, visit in enumerate(path):
            records.append(dict(
                patient_id=pid, node_id=visit["node_id"], traversal_index=index,
                stage=BEFORE if index <= boundary else AFTER,
                is_first_fish_creation=index == boundary,
                reference_type=LABELS.get(visit["anchor"], visit["anchor"]),
                source_anchor=visit["source_anchor"],
                anchor_available=visit["anchor_available"],
            ))
    require(bool(patients), "No FishIntroduction-anchor-created TEST patients")
    require(len({(v["patient_id"], v["node_id"]) for v in records}) == len(records),
            "Duplicate included patient–split visit")
    require(all(v["stage"] == BEFORE for v in records
                if v["is_first_fish_creation"]), "Creation split assigned after creation")
    return records, patients


def calculate_transition(results: Path) -> tuple[pd.DataFrame, dict]:
    """Count mutually exclusive source-reference categories within each stage."""
    visits, _, test_count = load_test_visits(Path(results), allow_other=True)
    records, patients = split_creation_stages(visits)
    references = ["Birth Date", "FishIntroduction", "Full Timeline"]
    references.extend(sorted({v["reference_type"] for v in records} - set(references)))
    rows, stage_summary = [], {}
    for stage in (BEFORE, AFTER):
        stage_records = [v for v in records if v["stage"] == stage]
        counts = Counter(v["reference_type"] for v in stage_records)
        total = len(stage_records)
        percentages = {ref: 100 * counts[ref] / total if total else None
                       for ref in references}
        require(sum(counts.values()) == total, "Stage reference accounting mismatch")
        if total:
            require(np.isclose(sum(percentages.values()), 100),
                    "Stage percentages do not sum to 100")
        stage_summary[stage] = dict(
            total_split_visits=total,
            reference_counts={ref: counts[ref] for ref in references},
            reference_percentages=percentages,
            population_patient_count=len(patients),
            patients_with_stage_visits=len({v["patient_id"] for v in stage_records}),
        )
        rows.extend(dict(
            stage=stage, reference_type=ref, visit_count=counts[ref],
            total_stage_split_visits=total, usage_percentage=percentages[ref],
            anchor_created_test_patients=len(patients),
            unit="percent of patient–split visits within stage",
        ) for ref in references)
    reference_sets = [set(p["post_anchor_references"]) for p in patients]
    pre_paths = {tuple((v["node_id"], v["direction"])
                      for v in p["pre_anchor_path"]) for p in patients}
    summary = dict(
        analysis="Temporal reference before/at versus after first fish-anchor creation",
        test_patient_count=test_count, anchor_created_test_patients=len(patients),
        population_definition="Same TEST patients with documented FishIntroduction "
                              "and saved nonempty fish-anchor creation in both panels",
        creation_boundary_rule="First nonempty FishIntroduction-derived anchor "
                               "creation per patient, in decision-path traversal order; "
                               "the creating split belongs to before_at_creation",
        metric="One count per patient–split visit, classified by its window source "
               "reference; each stage has its own split-visit denominator",
        stages=stage_summary,
        all_creation_nodes_are_root=all(p["creation_is_root"] for p in patients),
        creation_node_patient_counts=dict(Counter(p["creation_node_id"] for p in patients)),
        distinct_pre_anchor_paths=len(pre_paths),
        all_patients_share_pre_anchor_path=len(pre_paths) == 1,
        patients_with_multiple_fish_creation_visits=sum(
            len(p["fish_creation_visit_indices"]) > 1 for p in patients),
        patient_diagnostics=dict(
            at_least_one_post_split=sum(bool(refs) for refs in reference_sets),
            at_least_one_post_fish_split=sum("FishIntroduction" in refs
                                           for refs in reference_sets),
            only_fish=sum(refs == {"FishIntroduction"} for refs in reference_sets),
            fish_and_birth=sum(refs == {"FishIntroduction", "Birth Date"}
                               for refs in reference_sets),
            only_birth=sum(refs == {"Birth Date"} for refs in reference_sets),
            any_full_timeline=sum("Full Timeline" in refs for refs in reference_sets),
            any_other_reference=sum(bool(refs - set(LABELS.values()))
                                    for refs in reference_sets),
            no_post_split=sum(not refs for refs in reference_sets),
        ),
        unexpected_reference_categories=references[3:],
        unavailable_reference_visits=sum(not v["anchor_available"] for v in records),
        classification_policy="Visited source references are counted even for empty "
                              "or unavailable windows; availability is audited separately",
        validation_passed=True, model_loaded=False, predictions_generated=False,
        model_trained_or_refit=False, patients=patients,
        stage_assignments=records,
    )
    require(sum(s["total_split_visits"] for s in stage_summary.values())
            == sum(p["before_at_creation_visit_count"] + p["after_creation_visit_count"]
                   for p in patients), "Included path accounting mismatch")
    return pd.DataFrame(rows), summary


def audit_current_transition(summary: dict) -> None:
    """Check the existing final run only; expectations never drive counting."""
    before, after = summary["stages"][BEFORE], summary["stages"][AFTER]
    require(summary["test_patient_count"] == 1000
            and summary["anchor_created_test_patients"] == 720
            and before["total_split_visits"] == 720
            and before["reference_counts"]
            == {"Birth Date": 720, "FishIntroduction": 0, "Full Timeline": 0}
            and after["total_split_visits"] == 1440
            and after["reference_counts"]
            == {"Birth Date": 224, "FishIntroduction": 1216, "Full Timeline": 0},
            "Current saved-run audit mismatch: " + json.dumps(summary["stages"]))


def plot_transition(table: pd.DataFrame, summary: dict, output: Path) -> None:
    """Plot stage percentages without asserting a causal or performance claim."""
    colors = {"Birth Date": "#517C9B", "FishIntroduction": "#419A85",
              "Full Timeline": "#999999"}
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42}):
        figure, axes = plt.subplots(1, 2, figsize=(15, 5.6), sharey=True)
        try:
            for panel, axis, stage in zip("AB", axes, (BEFORE, AFTER)):
                values = table.loc[table.stage.eq(stage)]
                bars = axis.bar(np.arange(len(values)),
                                values.usage_percentage.fillna(0), width=.6,
                                color=[colors.get(ref, "#9370A8")
                                       for ref in values.reference_type])
                axis.set_xticks(np.arange(len(values)))
                axis.set_xticklabels(values.reference_type, fontsize=10)
                axis.set_ylim(0, 110)
                axis.set_yticks(np.arange(0, 101, 20))
                axis.grid(axis="y", alpha=.2)
                axis.set_axisbelow(True)
                for spine in ("top", "right"):
                    axis.spines[spine].set_visible(False)
                title = ("Before/at FishIntroduction anchor creation" if stage == BEFORE
                         else "After FishIntroduction anchor creation")
                axis.set_title(f"{panel}. Temporal reference\n{title}",
                               loc="left", fontsize=12, pad=15)
                total = summary["stages"][stage]["total_split_visits"]
                axis.set_xlabel(f"Patient–split visits in this stage: {total:,}",
                                labelpad=12)
                if not total:
                    axis.text(.5, .5, "No downstream split visits; percentages undefined",
                              transform=axis.transAxes, ha="center", fontsize=10)
                else:
                    for bar, row in zip(bars, values.itertuples(index=False)):
                        axis.annotate(
                            f"{row.usage_percentage:.1f}%\n(n={row.visit_count:,})",
                            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 5), textcoords="offset points",
                            ha="center", fontsize=10,
                        )
            axes[0].set_ylabel("Temporal-reference usage within stage (%)")
            figure.text(.5, .035,
                        "Same anchor-created held-out TEST population in both panels: "
                        f"{summary['anchor_created_test_patients']:,} patients",
                        ha="center", fontsize=11)
            figure.subplots_adjust(left=.07, right=.985, bottom=.21,
                                   top=.80, wspace=.16)
            for extension in ("png", "pdf"):
                figure.savefig(output / f"temporal_reference_transition.{extension}",
                               dpi=300, facecolor="white")
        finally:
            plt.close(figure)


def save_anchor_transition(results: Path, output: Path = None,
                           *, verify_current: bool = False) -> dict:
    """Audit first, then save new figure/data files without touching inputs."""
    results = Path(results)
    output = (Path(output) if output is not None
              else results / "dynamic_on/temporal_explainability_transition")
    require(not output.exists(), f"Output exists; refusing overwrite: {output}")
    table, summary = calculate_transition(results)
    if verify_current:
        audit_current_transition(summary)
    summary["source_sha256"] = {
        name: hashlib.sha256((results / name).read_bytes()).hexdigest()
        for name in INPUTS}
    summary["postprocessing_source_sha256"] = {
        str(path.relative_to(Path(__file__).resolve().parents[1])):
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__), Path(__file__).with_name(
            "experiment_2_temporal_explainability.py"))}
    output.mkdir(parents=True, exist_ok=False)
    plot_transition(table, summary, output)
    table.to_csv(output / "temporal_reference_transition.csv", index=False)
    (output / "temporal_reference_transition_audit.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary
