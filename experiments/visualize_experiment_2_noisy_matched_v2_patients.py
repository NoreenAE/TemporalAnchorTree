"""Plot two held-out patients using saved noisy matched V2 artifacts only.

Run from the repository root:
    python experiments/visualize_experiment_2_noisy_matched_v2_patients.py

Select the smallest patient ID in each requested label/symptom category.
Scores never affect selection. No model is loaded, trained, or invoked.
The output directory must be new; saved results are never overwritten.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments/results/experiment_2_noisy_matched_v2"
OUTPUT = ROOT / "figures/experiment_2_noisy_matched_v2_patient_timelines"
FISH = "FishIntroduction"
BIRTH = "Birth Date Anchor"
SYMPTOMS = ("GI", "Vomiting")
COLORS = {FISH: "#333333", "GI": "#2166AC", "Vomiting": "#B35806"}
MARKERS = {FISH: "D", "GI": "o", "Vomiting": "^"}
WINDOW_COLOR = "#65B99A"
FILES = (
    "observations.csv", "patients_and_split.csv", "manifest.json",
    "dynamic_on/patient_predictions.csv", "dynamic_on/tree.json",
    "anchor_usage_test_creations.csv", "anchor_usage_test_windows.csv",
)


def require(condition: bool, message: str) -> None:
    """Stop rather than publish a figure with inconsistent saved evidence."""
    if not condition:
        raise ValueError(message)


def source_hashes() -> dict[str, str]:
    """Fingerprint every saved artifact consumed by this visualization."""
    return {name: hashlib.sha256((RESULTS / name).read_bytes()).hexdigest()
            for name in FILES}


def load_artifacts() -> tuple:
    """Load one saved run and verify its immutable dataset snapshots."""
    missing = [str(RESULTS / name) for name in FILES
               if not (RESULTS / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing saved artifacts:\n" + "\n".join(missing))
    manifest = json.loads((RESULTS / "manifest.json").read_text())
    hashes = source_hashes()
    for name in ("observations.csv", "patients_and_split.csv"):
        require(hashes[name] == manifest[name], f"Snapshot hash mismatch: {name}")
    observations = pd.read_csv(
        RESULTS / "observations.csv", parse_dates=["time"],
        dtype={"patient_id": str, "value": str},
    )
    date_columns = ["birth", "followup_end", "fish_time"]
    patients = pd.read_csv(RESULTS / "patients_and_split.csv",
                           parse_dates=date_columns, dtype={"patient_id": str})
    predictions = pd.read_csv(
        RESULTS / "dynamic_on/patient_predictions.csv", parse_dates=date_columns,
        dtype={"patient_id": str}, float_precision="round_trip",
    )
    require(not patients.patient_id.duplicated().any(), "Duplicate metadata IDs")
    require(not predictions.patient_id.duplicated().any(), "Duplicate prediction IDs")
    patients = patients.set_index("patient_id").sort_index()
    predictions = predictions.set_index("patient_id").sort_index()
    require(patients.index.equals(predictions.index), "Prediction cohort mismatch")
    for column in ("partition", "documented", "FA", *date_columns):
        require(patients[column].equals(predictions[column]),
                f"Metadata/prediction mismatch: {column}")
    creations = pd.read_csv(RESULTS / "anchor_usage_test_creations.csv",
                            parse_dates=["anchor_timestamp"])
    windows = pd.read_csv(RESULTS / "anchor_usage_test_windows.csv",
                          parse_dates=["window_start", "window_end"])
    tree = json.loads((RESULTS / "dynamic_on/tree.json").read_text())
    return observations, patients, predictions, creations, windows, tree, hashes


def qualifying_events(rows: pd.DataFrame, fish: pd.Timestamp) -> pd.DataFrame:
    """Apply the dataset's strict-after, inclusive-end calendar-month rule."""
    end = fish + pd.DateOffset(months=1)
    return rows.loc[rows.feature_name.isin(SYMPTOMS)
                    & rows.time.gt(fish) & rows.time.le(end)]


def validate_patient(patient_id: str, label: int, artifacts: tuple) -> dict:
    """Cross-check observations, saved probabilities, and actual anchor visits."""
    observations, patients, predictions, creations, windows, tree, _ = artifacts
    patient, prediction = patients.loc[patient_id], predictions.loc[patient_id]
    rows = observations.loc[observations.patient_id.eq(patient_id)]
    require(patient.partition == prediction.partition == "test", "Expected TEST")
    require(bool(patient.documented), "FishIntroduction must be documented")
    fish_rows = rows.loc[rows.feature_name.eq(FISH)]
    require(len(fish_rows) == 1, "Expected exactly one observed FishIntroduction")
    fish = fish_rows.time.iloc[0]
    require(fish == patient.fish_time, "FishIntroduction timestamp mismatch")
    birth = rows.loc[rows.feature_name.eq(BIRTH)]
    require(len(birth) == 1 and birth.time.iloc[0] == patient.birth
            and pd.Timestamp(birth.value.iloc[0]) == patient.birth,
            "Observed birth anchor mismatch")
    events = rows.loc[~rows.feature_name.eq(BIRTH)]
    require(pd.to_numeric(events.value).eq(1).all(), "Expected unit-valued events")
    require(events.time.between(patient.birth, patient.followup_end).all(),
            "Observed event outside follow-up")
    require(patient.followup_end == patient.birth + pd.DateOffset(months=30),
            "Expected 30-calendar-month follow-up")
    # The tree audit uses an inclusive start. The dataset prevents coincident
    # symptoms, so its strict-after rule and audited counts must agree here.
    require(not events.loc[events.feature_name.isin(SYMPTOMS), "time"].eq(fish).any(),
            "Coincident symptom makes strict-after truth differ from model window")
    created = creations.loc[
        creations.run.eq("dynamic_on") & creations.patient_id.eq(patient_id)
        & creations.anchor_feature.eq(FISH) & creations.anchor_timestamp.eq(fish)
    ]
    require(not created.empty and bool(prediction.fish_anchor_created),
            "No saved FishIntroduction anchor creation evidence")
    available = windows.loc[
        windows.run.eq("dynamic_on") & windows.patient_id.eq(patient_id)
        & windows.anchor_feature.eq(FISH) & windows.feature.isin(SYMPTOMS)
        & windows.visited.eq(True) & windows.anchor_available.eq(True)
        & windows.source_anchor.isin(created.anchor_name)
        & windows.offset_months.eq(1) & windows.window_start.eq(fish)
        & windows.window_end.eq(fish + pd.DateOffset(months=1))
    ]
    require(not available.empty and bool(prediction.available_fish_anchor_used),
            "No saved available FishIntroduction-relative symptom window")
    qualifying = qualifying_events(rows, fish)
    require(int(patient.FA) == label and bool(len(qualifying)) == bool(label),
            "Patient does not satisfy requested label/symptom category")
    require(np.isfinite(prediction.score) and 0 <= prediction.score <= 1,
            "Invalid saved prediction probability")
    nodes = {node["node_id"]: node for node in tree}
    visits = []
    for visit in available.itertuples(index=False):
        node = nodes[visit.node_id]
        require(node["feature"] == visit.feature
                and node["window"] == [visit.source_anchor, 1],
                "Saved window audit differs from saved tree")
        count = int(qualifying.feature_name.eq(visit.feature).sum())
        require(count == visit.eligible_symptom_count, "Audited symptom count mismatch")
        visits.append(dict(node_id=int(visit.node_id), feature=visit.feature,
                           aggregation=node["aggregation"], threshold=node["threshold"],
                           direction=visit.direction, eligible_symptom_count=count,
                           source_anchor=visit.source_anchor))
    return dict(
        patient_id=patient_id, partition="test", true_FA=label,
        saved_dynamic_on_prediction_probability=float(prediction.score),
        fish_introduction_timestamp=fish.isoformat(),
        window_end_timestamp=(fish + pd.DateOffset(months=1)).isoformat(),
        fish_anchor_created=True, fish_anchor_available=True,
        qualifying_GI_events=[t.isoformat() for t in qualifying.loc[
            qualifying.feature_name.eq("GI"), "time"].sort_values()],
        qualifying_Vomiting_events=[t.isoformat() for t in qualifying.loc[
            qualifying.feature_name.eq("Vomiting"), "time"].sort_values()],
        saved_symptom_window_visits=visits,
        reason=(f"Documented introduction and saved created/available fish anchor; "
                f"FA={label}; {len(qualifying)} qualifying GI/Vomiting event(s)."),
    )


def select_patients(artifacts: tuple) -> tuple[list[dict], dict[str, int]]:
    """Select lexicographic minima without using scores or prediction accuracy."""
    observations, patients, _, creations, _, _, _ = artifacts
    created_ids = set(creations.loc[
        creations.run.eq("dynamic_on") & creations.anchor_feature.eq(FISH)
        & creations.anchor_timestamp.notna(), "patient_id"])
    groups = {pid: rows for pid, rows in observations.groupby("patient_id")}
    candidates = {1: [], 0: []}
    for pid, patient in patients.loc[
        patients.partition.eq("test") & patients.documented.eq(True)
        & patients.index.isin(created_ids)
    ].iterrows():
        fish_rows = groups[pid].loc[groups[pid].feature_name.eq(FISH)]
        require(len(fish_rows) == 1, f"Ambiguous documented introduction: {pid}")
        has_symptom = bool(len(qualifying_events(groups[pid], fish_rows.time.iloc[0])))
        label = int(patient.FA)
        if has_symptom == bool(label):
            candidates[label].append(pid)
    for label in (1, 0):
        require(bool(candidates[label]), f"No eligible example for FA={label}")
    # Validation fails loudly if the saved available-window evidence is absent.
    selected = [validate_patient(min(candidates[label]), label, artifacts)
                for label in (1, 0)]
    return selected, {str(label): len(ids) for label, ids in candidates.items()}


def draw_figure(selected: list[dict], observations: pd.DataFrame,
                patients: pd.DataFrame):
    """Show every observed event with exact calendar-month shading endpoints."""
    features = [FISH, *SYMPTOMS, *sorted(
        set(observations.feature_name) - {BIRTH, FISH, *SYMPTOMS})]
    figure, axes = plt.subplots(2, 1, figsize=(11, 7.6), sharex=True)
    # Keep full-precision saved scores; round only the displayed panel titles.
    titles = [
        f"({panel}) True FA = {example['true_FA']}, model score = "
        f"{example['saved_dynamic_on_prediction_probability']:.3f}"
        for panel, example in zip("ab", selected)
    ]
    for axis, example, title in zip(axes, selected, titles):
        pid = example["patient_id"]
        birth = patients.loc[pid, "birth"]
        edges = pd.DatetimeIndex([
            birth + pd.DateOffset(months=m) for m in range(32)]).to_numpy()

        def age(times) -> np.ndarray:
            values = pd.DatetimeIndex(times).to_numpy()
            index = np.searchsorted(edges, values, side="right") - 1
            require(((index >= 0) & (index < len(edges) - 1)).all(),
                    "Timestamp outside calendar plotting coordinates")
            return index + (values - edges[index]) / (edges[index + 1] - edges[index])

        start, end = age([example["fish_introduction_timestamp"],
                          example["window_end_timestamp"]])
        axis.axvspan(start, end, color=WINDOW_COLOR, alpha=.22, linewidth=0)
        axis.axvline(start, color=COLORS[FISH], linestyle="--", linewidth=1)
        rows = observations.loc[observations.patient_id.eq(pid)]
        for y, feature in enumerate(features):
            events = rows.loc[rows.feature_name.eq(feature)]
            axis.scatter(age(events.time), np.full(len(events), y),
                         marker=MARKERS.get(feature, "|"),
                         color=COLORS.get(feature, "#9A9A9A"),
                         s=32 if feature in COLORS else 22,
                         linewidths=.7, zorder=3)
        axis.set_yticks(range(len(features)))
        axis.set_yticklabels(features, fontsize=9)
        axis.set_ylim(len(features) - .5, -.6)
        axis.set_title(title, loc="left", fontsize=12, pad=12)
        axis.grid(axis="x", color="#E9E9E9", linewidth=.5)
        axis.tick_params(axis="y", length=0)
        for spine in ("top", "right", "left"):
            axis.spines[spine].set_visible(False)
        axis.spines["bottom"].set_color("#BBBBBB")
    axes[-1].set_xlim(-.15, 30.15)
    axes[-1].set_xticks(range(0, 31, 2))
    axes[-1].set_xlabel("Age in calendar months", fontsize=11, labelpad=8)
    handles = [
        Line2D([], [], color=COLORS[FISH], linestyle="--", marker="D",
               markersize=5, linewidth=1, label="Fish introduction"),
        Patch(facecolor=WINDOW_COLOR, alpha=.22,
              label="0–1 month post-introduction window"),
        *[Line2D([], [], color=COLORS[f], marker=MARKERS[f], linestyle="none",
                 markersize=5, label=f) for f in SYMPTOMS],
    ]
    figure.legend(handles=handles, loc="lower center", ncol=4,
                  frameon=False, fontsize=9, bbox_to_anchor=(.54, .015))
    figure.subplots_adjust(left=.19, right=.985, top=.95, bottom=.14, hspace=.35)
    return figure


def main() -> None:
    """Validate both examples before creating new publication outputs."""
    require(not OUTPUT.exists(), f"Output exists; refusing to overwrite: {OUTPUT}")
    artifacts = load_artifacts()
    selected, counts = select_patients(artifacts)
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42}):
        figure = draw_figure(selected, artifacts[0], artifacts[1])
        try:
            require(source_hashes() == artifacts[-1], "Saved artifacts changed during plotting")
            OUTPUT.mkdir(parents=True, exist_ok=False)
            for extension in ("png", "pdf"):
                figure.savefig(OUTPUT / f"patient_timelines.{extension}", dpi=400,
                               facecolor="white", bbox_inches="tight", pad_inches=.15)
        finally:
            plt.close(figure)
    provenance = dict(
        saved_run=str(RESULTS.relative_to(ROOT)), model="Dynamic Anchoring ON",
        source_sha256=artifacts[-1],
        visualization_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        selection_rule="Smallest patient_id in each TEST, documented, fish-anchor-created "
                       "label/symptom category; saved available windows validated afterwards. "
                       "No scores, correctness, or visual criteria used.",
        eligible_candidate_counts=counts,
        classification_threshold=None,
        classification_threshold_defined=False,
        classification_threshold_applied=False,
        score_interpretation="Experiment 2 evaluated continuous model scores; "
                             "it did not define or apply a classification threshold "
                             "or produce predicted FA class labels. Scores are rounded "
                             "to three decimals only in the figure titles.",
        symptom_window_rule="FishIntroduction < event <= FishIntroduction + DateOffset(months=1)",
        age_coordinates="Linear interpolation between birth + DateOffset(months=m) ticks; "
                        "window endpoints transformed independently from exact timestamps.",
        validation_passed=True, model_loaded=False, model_trained_or_refit=False,
        predictions_recomputed=False, patients=selected,
    )
    (OUTPUT / "provenance.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
