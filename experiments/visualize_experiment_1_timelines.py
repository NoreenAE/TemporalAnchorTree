"""Plot saved Experiment 1 mechanisms; no data generation or model execution.

Run from the repository: python experiments/visualize_experiment_1_timelines.py
Patients are selected deterministically by feeding group / active GI predictors
and observed label, then patient ID. These are illustrations, not a prevalence
sample. All plotted points and probabilities come from existing artifacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/temporal_benchmark_v3_clean_fa"
FA_FEATURES = ["FormulaYN", "BreastMilkYN"]
GI_FEATURES = ["VomitingYN", "DiarrheaYN", "GISymptomYN"]
BLUE, RED, GRAY = "#2166AC", "#B53A3A", "#9B9B9B"


def calendar_coordinate(times, origin, first: int, last: int) -> np.ndarray:
    """Interpolate between exact calendar-month ticks for plotting only."""
    edges = pd.DatetimeIndex([
        origin + pd.DateOffset(months=m) for m in range(first, last + 2)
    ]).to_numpy()
    values = pd.DatetimeIndex(times).to_numpy()
    indices = np.searchsorted(edges, values, side="right") - 1
    indices = np.clip(indices, 0, len(edges) - 2)
    return first + indices + (values - edges[indices]) / (edges[indices + 1] - edges[indices])


def pick_patient(table: pd.DataFrame, **criteria) -> pd.Series:
    """Choose the first saved patient satisfying the stated display criteria."""
    selected = table
    for column, value in criteria.items():
        selected = selected.loc[selected[column].eq(value)]
    if selected.empty:
        raise ValueError(f"No saved patient meets selection criteria: {criteria}")
    return selected.sort_values("patient_id").iloc[0]


def draw_patient(ax, observations, patient, mechanism: str) -> None:
    """Separate relevant values 1/0 and observations outside the true window."""
    is_fa = mechanism == "FA"
    features = FA_FEATURES if is_fa else GI_FEATURES
    origin = patient.birth_date if is_fa else patient.now
    first, last = (0, 12) if is_fa else (-3, 0)
    low = origin if is_fa else origin - pd.DateOffset(months=1)
    high = origin + pd.DateOffset(months=6) if is_fa else origin
    color = BLUE if is_fa else RED
    window = (0, 6) if is_fa else (-1, 0)
    ax.axvspan(*window, color=color, alpha=.09, zorder=0)
    ax.axvline(0, color="#555555", linewidth=.8, linestyle="--", zorder=1)
    rows = observations.loc[observations.patient_id.eq(patient.patient_id)]
    # The saved anchors, not a common invented origin, define each panel.
    for feature, expected in [("Birth Date Anchor", patient.birth_date), ("NOW Anchor", patient.now)]:
        anchor = rows.loc[rows.feature_name.eq(feature), "value"]
        if len(anchor) != 1 or pd.Timestamp(anchor.iloc[0]) != expected:
            raise ValueError(f"Inconsistent saved anchor for {patient.patient_id}")
    display_low = origin + pd.DateOffset(months=first)
    display_high = origin + pd.DateOffset(months=last)
    for y, feature in enumerate(features):
        events = rows.loc[rows.feature_name.eq(feature) & rows.time.between(display_low, display_high)]
        values = pd.to_numeric(events.value)
        if not values.isin([0, 1]).all():
            raise ValueError(f"Nonbinary observation: {patient.patient_id}/{feature}")
        inside = events.time.between(low, high)
        x = calendar_coordinate(events.time, origin, first, last)
        for relevant, value, marker, size in [
            (False, 0, "o", 12), (False, 1, "x", 20),
            (True, 0, "o", 18), (True, 1, "o" if is_fa else "*", 22 if is_fa else 65),
        ]:
            mask = (inside.eq(relevant) & values.eq(value)).to_numpy()
            shade = color if relevant else GRAY
            options = dict(color=shade) if marker == "x" else dict(
                facecolors=shade if value else "white", edgecolors=shade)
            ax.scatter(x[mask], np.full(mask.sum(), y), marker=marker, s=size,
                       linewidths=.65, zorder=3 if relevant else 2, **options)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels([f.replace("YN", "").replace("BreastMilk", "Breast milk")
                        .replace("GISymptom", "GI symptom") for f in features])
    ax.set_ylim(len(features) - .5, -.7)
    ax.set_xlim(first - .10 if is_fa else first - .025, last + (.15 if is_fa else .025))
    ax.set_xticks([0, 2, 4, 6, 8, 10, 12] if is_fa else [-3, -2, -1, 0])
    ax.set_xticklabels(["Birth", "2", "4", "6", "8", "10", "12"]
                       if is_fa else ["−3", "−2", "−1", "NOW"])
    probability = patient.fa_probability if is_fa else patient.gi_probability
    label = patient.fa_label if is_fa else patient.gi_label
    ax.set_title(f"{patient.patient_id}   |   P({mechanism}=1) = {probability:.1%}   |   observed {mechanism} = {int(label)}",
                 loc="left", fontsize=10, pad=22)
    detail = f"{patient.trajectory_description} (group {patient.feeding_group})" if is_fa else (
        f"Final-month positives: vomiting={int(patient.V_i)}, diarrhea={int(patient.D_i)}, "
        f"GI symptom={int(patient.G_i)}")
    ax.text(0, 1.07, detail, transform=ax.transAxes, fontsize=8.5, color="#555555")
    ax.grid(axis="x", color="#EEEEEE", linewidth=.5, zorder=0)
    ax.tick_params(length=2.5, labelsize=8)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#BBBBBB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "figures/experiment_1_ground_truth_timelines.png")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace only the requested PNG and matching PDF.")
    args = parser.parse_args()
    pdf_path = args.output.with_suffix(".pdf")
    if args.output.suffix.lower() != ".png":
        raise ValueError("Use a .png output path; the matching PDF is also saved.")
    if not args.overwrite and (args.output.exists() or pdf_path.exists()):
        raise FileExistsError("Figure already exists; choose a different --output path.")
    observations = pd.read_pickle(DATA_DIR / "temporal_benchmark_5000.pkl")
    observations = observations.copy()
    observations["time"] = pd.to_datetime(observations.time)
    patients = pd.read_pickle(DATA_DIR / "temporal_benchmark_ground_truth_5000.pkl")
    gi = pd.read_pickle(DATA_DIR / "temporal_benchmark_gi_ground_truth_5000.pkl").drop(columns="gi_label")
    patients = patients.merge(gi, on="patient_id", validate="one_to_one")
    for outcome in ("fa", "gi"):
        labels = pd.read_pickle(DATA_DIR / f"temporal_benchmark_{outcome}_labels_5000.pkl")
        patients = patients.merge(labels.rename(columns={"label": f"{outcome}_label"}),
                                  on="patient_id", validate="one_to_one")
    patients["birth_date"] = pd.to_datetime(patients.birth_date)
    patients["now"] = pd.to_datetime(patients.now)
    patients["feeding_group"] = patients.feeding_group.astype(str)
    choices = [("2.0", 0, "Formula only"), ("2.2", 1, "Mixed → formula between 21 days and 2 months"),
               ("5.2", 0, "Breast milk → formula between 2 and 4 months"), ("3", 1, "Breast milk only")]
    fa_examples = []
    for group, label, description in choices:
        patient = pick_patient(patients, feeding_group=group, fa_label=label).copy()
        patient["trajectory_description"] = description
        fa_examples.append(patient)
    gi_examples = [pick_patient(patients, n_active_predictors=n, gi_label=label)
                   for n, label in [(0, 1), (1, 0), (2, 1), (3, 0)]]
    # Validate selected GI summaries against real positive observations.
    metadata = json.loads((DATA_DIR / "temporal_benchmark_gi_metadata_5000.json").read_text())
    for patient in gi_examples:
        rows = observations.loc[observations.patient_id.eq(patient.patient_id)]
        recent = rows.loc[rows.time.between(patient.now - pd.DateOffset(months=1), patient.now)]
        active = [int(pd.to_numeric(recent.loc[recent.feature_name.eq(f), "value"]).max())
                  for f in GI_FEATURES]
        if active != [int(patient.V_i), int(patient.D_i), int(patient.G_i)]:
            raise ValueError("Saved GI ground truth differs from observations")
        logit = metadata["calibrated_intercept"] + np.dot(active, np.log([3, 2.5, 2]))
        if not np.isclose(1 / (1 + np.exp(-logit)), patient.gi_probability):
            raise ValueError("Saved GI probability differs from its mechanism")
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white",
                         "axes.facecolor": "white", "pdf.fonttype": 42})
    fig, axes = plt.subplots(4, 2, figsize=(15, 11))
    fig.subplots_adjust(left=.075, right=.985, bottom=.18, top=.835, hspace=.78, wspace=.24)
    for i in range(4):
        draw_patient(axes[i, 0], observations, fa_examples[i], "FA")
        draw_patient(axes[i, 1], observations, gi_examples[i], "GI")
    axes[-1, 0].set_xlabel("Age in calendar months · first 12 months shown", fontsize=10)
    axes[-1, 1].set_xlabel("Calendar months relative to NOW · final 3 months shown", fontsize=10)
    fig.suptitle("Experiment 1: two independent temporal ground-truth mechanisms", fontsize=17, y=.97)
    fig.text(.075, .923, "A   Food allergy: early feeding trajectory", fontsize=13, weight="bold", color=BLUE)
    fig.text(.075, .899, "Shaded window: Birth to +6 calendar months (inclusive)", fontsize=10)
    fig.text(.58, .923, "B   GI: recent symptom indicators", fontsize=13, weight="bold", color=RED)
    fig.text(.58, .899, "Shaded window: NOW −1 calendar month to NOW (inclusive)", fontsize=10)
    handles = [
        Line2D([], [], marker="o", color=BLUE, linestyle="none", label="FA window: value 1"),
        Line2D([], [], marker="*", markersize=9, color=RED, linestyle="none", label="GI window: value 1"),
        Line2D([], [], marker="o", markerfacecolor="white", color="#555555", linestyle="none", label="Open circle: recorded value 0"),
        Line2D([], [], marker="x", color=GRAY, linestyle="none", label="Gray: outside outcome's window"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .086), ncol=4,
               frameon=False, fontsize=9, handletextpad=.5, columnspacing=1.4)
    fig.text(.5, .052, "Probabilities describe synthetic risk; observed labels are sampled Bernoulli outcomes, not deterministic rule results.",
             ha="center", fontsize=10)
    fig.text(.5, .027, "Saved V3 clean-FA patients, purposefully selected for illustration—not a prevalence sample. No histories or labels were generated.",
             ha="center", fontsize=9, color="#555555")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)
    print("FA patients:", ", ".join(p.patient_id for p in fa_examples))
    print("GI patients:", ", ".join(p.patient_id for p in gi_examples))
    print(args.output)
    print(pdf_path)


if __name__ == "__main__":
    main()
