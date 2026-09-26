"""Create publication figures from saved Experiment 2 artifacts only.

This script never calls TemporalTree, regenerates data, or changes saved results.
Run from the repository root with:
    python figures/experiment_2_fish_introduction_dynamic_anchors/visualize_saved_results.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/results/experiment_2_fish_introduction_dynamic_anchors"
DATASET = ROOT / "data/fish_intro_dataset_distractors"
OUTPUT = ROOT / "figures/experiment_2_fish_introduction_dynamic_anchors"
FISH, GI, VOM = "FishIntroduction", "GI", "Vomiting"
DISTRACTORS = {"Fever", "Cough", "Rash", "SleepDisturbance", "DoctorVisit"}
BLUE, RED, GRAY, GREEN = "#2166AC", "#B53A3A", "#777777", "#2A9D8F"


def load_artifacts():
    """Load only persisted CSV/JSON artifacts used by the figures."""
    observations = pd.read_csv(DATASET / "observations.csv", parse_dates=["time"])
    patients = pd.read_csv(
        DATASET / "patients_and_split.csv",
        parse_dates=["birth", "followup_end", "a", "b", "signal_time", "fish_time"],
        dtype={"patient_id": str, "documented": bool},
    )
    saved = pd.read_csv(RESULTS / "patients_and_split.csv", parse_dates=["birth", "followup_end", "a", "b", "signal_time", "fish_time"], dtype={"patient_id": str, "documented": bool})
    if not patients.equals(saved):
        raise ValueError("Experiment results and saved dataset have different patient tables.")
    comparison = json.loads((RESULTS / "comparison.json").read_text())
    on_tree = json.loads((RESULTS / "dynamic_on/tree.json").read_text())
    on_diagnostics = json.loads((RESULTS / "dynamic_on/diagnostics.json").read_text())
    off_diagnostics = json.loads((RESULTS / "dynamic_off/diagnostics.json").read_text())
    return observations, patients, comparison, on_tree, on_diagnostics, off_diagnostics


def calendar_age(times, birth):
    """Return exact-calendar-month plotting coordinates from a birth timestamp."""
    values = pd.DatetimeIndex(times).to_numpy()
    edges = pd.DatetimeIndex([birth + pd.DateOffset(months=i) for i in range(32)]).to_numpy()
    indices = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, 30)
    return indices + (values - edges[indices]) / (edges[indices + 1] - edges[indices])


def relative_months(times, origin):
    """Return calendar-month coordinates relative to a patient-specific origin."""
    values = pd.DatetimeIndex(times).to_numpy()
    origin = pd.Timestamp(origin)
    result = []
    for value in values:
        value = pd.Timestamp(value)
        if value >= origin:
            whole = (value.year - origin.year) * 12 + value.month - origin.month
            start = origin + pd.DateOffset(months=whole)
            result.append(whole + (value - start) / (start + pd.DateOffset(months=1) - start))
        else:
            whole = (origin.year - value.year) * 12 + origin.month - value.month
            start = origin - pd.DateOffset(months=whole)
            result.append(-whole + (value - start) / (origin - start))
    return np.asarray(result, dtype=float)


def patient_rows(observations, patient_id):
    return observations.loc[observations.patient_id.eq(patient_id)].copy()


def choose_matched_pair(patients):
    """Select the first block containing documented R=1 and R=0 children."""
    for block in sorted(patients.block.unique()):
        group = patients.loc[patients.block.eq(block) & patients.documented]
        positives = group.loc[group.R.eq(1)].sort_values("patient_id")
        negatives = group.loc[group.R.eq(0)].sort_values("patient_id")
        if len(positives) and len(negatives):
            return positives.iloc[0], negatives.iloc[0]
    raise ValueError("No matched documented R=1/R=0 block found.")


def draw_rule_timeline(ax, observations, patient, label, title):
    features = [FISH, GI, VOM]
    rows = patient_rows(observations, patient.patient_id)
    fish = rows.loc[rows.feature_name.eq(FISH), "time"]
    if len(fish) != 1:
        raise ValueError("Selected patient does not have one fish timestamp.")
    fish_time = fish.iloc[0]
    end = fish_time + pd.DateOffset(months=1)
    show = rows.loc[rows.feature_name.isin(features)].copy()
    show["x"] = calendar_age(show.time, patient.birth)
    origin_age = float(calendar_age([fish_time], patient.birth)[0])
    window_end_age = float(calendar_age([end], patient.birth)[0])
    ax.axvspan(origin_age, window_end_age, color=GREEN, alpha=.16)
    ax.axvline(origin_age, color="#222222", linestyle="--", linewidth=.8)
    for y, feature in enumerate(features):
        part = show.loc[show.feature_name.eq(feature)]
        value = pd.to_numeric(part.value)
        qualifying = part.time.gt(fish_time) & part.time.le(end) & value.eq(1)
        for mask, marker, color, size in [
            (value.eq(0), "o", "white", 28),
            (value.eq(1) & ~qualifying, "x", GRAY, 28),
            (qualifying, "*", RED, 105),
        ]:
            ax.scatter(part.loc[mask, "x"], np.full(mask.sum(), y), marker=marker,
                       s=size, facecolors=color if marker == "o" else None,
                       edgecolors=GRAY if marker == "o" else color, linewidths=.8)
    ax.set_yticks(range(3), ["Fish introduction", "GI", "Vomiting"])
    ax.set_xlim(max(0, origin_age - 4), min(30, window_end_age + 8))
    ax.set_xlabel("Age in calendar months")
    ax.set_title(f"{title}: {patient.patient_id} | R={int(patient.R)} | sampled FA={int(patient.FA)}", loc="left", fontsize=10)
    ax.text(.01, 1.06, f"FA probability={patient.fa_probability:.1%}; shaded region is [fish, fish + 1 calendar month]",
            transform=ax.transAxes, fontsize=8.5, color="#555555")
    ax.grid(axis="x", color="#EEEEEE", linewidth=.5)
    ax.set_ylim(2.5, -.5)


def panel_a(ax, observations, patients):
    positive, negative = choose_matched_pair(patients)
    draw_rule_timeline(ax[0], observations, positive, positive.FA, "R = 1 · within-window symptom")
    draw_rule_timeline(ax[1], observations, negative, negative.FA, "R = 0 · symptom before/after window")
    ax[0].text(-.12, 1.18, "A", transform=ax[0].transAxes, fontsize=15, fontweight="bold")
    ax[0].text(.0, 1.22, "Ground-truth temporal rule", transform=ax[0].transAxes, fontsize=12, fontweight="bold")
    ax[1].text(.01, -0.38, "Patients are matched within block; FA is sampled independently from R.", transform=ax[1].transAxes, fontsize=8.5, color="#555555")


def panel_b(ax, tree):
    nodes = {n["node_id"]: n for n in tree}
    fish_node = nodes[0]
    downstream = [n for n in tree if n.get("type") == "split" and n.get("window", [None])[0] == "Node 0 | SUM(FishIntroduction) Anchor" and n["feature"] in (GI, VOM)]
    ax.axis("off")
    ax.text(.0, .98, "B", transform=ax.transAxes, fontsize=15, fontweight="bold")
    ax.text(.06, .98, "Actual learned dynamic-anchor mechanism (saved ON tree)", transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
    def details(node):
        anchor, offset = node["window"]
        window = "full timeline" if anchor is None else f"{anchor} + {offset} month(s)"
        return f"node {node['node_id']} · {node['aggregation'].upper()}({node['feature']}) ≤ {node['threshold']:.0f}\nwindow: {window} · LEFT {node['left']} / RIGHT {node['right']}"
    positions = [(.08, .67), (.56, .78), (.56, .48)]
    labels = [details(fish_node)] + [details(n) for n in downstream[:2]]
    colors = [BLUE, RED, RED]
    for (x, y), text, color in zip(positions, labels, colors):
        ax.text(x, y, text, transform=ax.transAxes, bbox=dict(boxstyle="round,pad=.55", facecolor="white", edgecolor=color, linewidth=2), fontsize=9, va="center")
    ax.add_patch(FancyArrowPatch((.39, .69), (.55, .75), transform=ax.transAxes, arrowstyle="-|>", mutation_scale=14, color="#555555"))
    ax.add_patch(FancyArrowPatch((.39, .66), (.55, .52), transform=ax.transAxes, arrowstyle="-|>", mutation_scale=14, color="#555555"))
    ax.text(.06, .19, "Node 0 creates the patient-specific fish anchor.\nNodes 1 and 4 then evaluate GI/Vomiting within +1 calendar month of that anchor.\nOther saved splits include distractors; this panel reports only nodes that use the fish-derived anchor.", transform=ax.transAxes, fontsize=9, color="#444444", va="top")


def panel_c(ax, comparison):
    metrics = ["test_overall", "test_documented"]
    labels = ["Overall\n(n=1,000)", "Documented\n(n=800)"]
    x = np.arange(2); width = .34
    on = [comparison["dynamic_on"]["metrics"][m]["roc_auc"] for m in metrics]
    off = [comparison["dynamic_off"]["metrics"][m]["roc_auc"] for m in metrics]
    ax.bar(x - width/2, on, width, color=BLUE, label="Dynamic ON")
    ax.bar(x + width/2, off, width, color=GRAY, label="Dynamic OFF")
    for positions, values in ((x-width/2, on), (x+width/2, off)):
        for position, value in zip(positions, values): ax.text(position, value+.018, f"{value:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, labels); ax.set_ylim(0, 1.08); ax.set_ylabel("Held-out ROC-AUC")
    ax.set_title("C   Predictive performance", loc="left", fontsize=12, fontweight="bold")
    ax.text(0, -.22, "Brier score: ON 0.074 overall / 0.080 documented; OFF 0.218 / 0.261", transform=ax.transAxes, fontsize=8.5, color="#555555")
    ax.legend(frameon=False, loc="upper left"); ax.grid(axis="y", color="#EEEEEE")


def panel_d(ax, on_diag):
    cov = on_diag["coverage"]
    labels = ["Train\n(n=3,200)", "Test\n(n=800)"]
    x = np.arange(2); width = .36
    created = [cov[p]["fish_anchor_created"] for p in ("train", "test")]
    used = [cov[p]["available_fish_anchor_used"] for p in ("train", "test")]
    ax.bar(x-width/2, created, width, color=BLUE, label="Anchor created")
    ax.bar(x+width/2, used, width, color=GREEN, label="Available anchor used")
    for positions, values in ((x-width/2, created), (x+width/2, used)):
        for position, value in zip(positions, values): ax.text(position, value+35, str(value), ha="center", fontsize=8)
    ax.set_xticks(x, labels); ax.set_ylim(0, 3500); ax.set_ylabel("Documented-introduction children")
    ax.set_title("D   Fish-anchor coverage", loc="left", fontsize=12, fontweight="bold")
    ax.text(0, -.22, "Usage counts refer to children with an available fish anchor on a later path.", transform=ax.transAxes, fontsize=8.5, color="#555555")
    ax.legend(frameon=False, loc="upper left"); ax.grid(axis="y", color="#EEEEEE")


def figure_one(obs, patients, comparison, tree, on_diag):
    fig = plt.figure(figsize=(15, 12), facecolor="white")
    grid = fig.add_gridspec(4, 2, height_ratios=[1.08, 1.08, .9, .9], hspace=.9, wspace=.3)
    panel_a([fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, :])], obs, patients)
    panel_b(fig.add_subplot(grid[2, :]), tree)
    panel_c(fig.add_subplot(grid[3, 0]), comparison)
    panel_d(fig.add_subplot(grid[3, 1]), on_diag)
    fig.suptitle("Experiment 2 — Fish Introduction – Dynamic Anchors", fontsize=17, y=.98)
    fig.text(.5, .015, "Saved artifacts only. R is the timestamp rule; FA is an independently sampled probabilistic outcome. Metrics are held-out test results.", ha="center", fontsize=9, color="#555555")
    return fig


def choose_timeline_sample(patients):
    """Choose two documented children per block from first qualifying matched blocks."""
    selected = []
    for block in sorted(patients.block.unique()):
        group = patients.loc[patients.block.eq(block) & patients.documented]
        r1, r0 = group.loc[group.R.eq(1)].sort_values("patient_id"), group.loc[group.R.eq(0)].sort_values("patient_id")
        if len(r1) and len(r0):
            selected.extend([r1.iloc[0], r0.iloc[0]])
        if len(selected) >= 8: break
    return pd.DataFrame(selected[:8]).reset_index(drop=True)


def figure_two(obs, patients):
    sample = choose_timeline_sample(patients)
    features = [FISH, GI, VOM]
    fig, axes = plt.subplots(len(sample), 2, figsize=(14, 16), sharey="row")
    for row, patient in enumerate(sample.itertuples(index=False)):
        data = obs.loc[obs.patient_id.eq(patient.patient_id) & obs.feature_name.isin(features)]
        fish = data.loc[data.feature_name.eq(FISH), "time"].iloc[0]
        end = fish + pd.DateOffset(months=1)
        for col, aligned in enumerate(("birth", "fish")):
            ax = axes[row, col]
            origin = patient.birth if aligned == "birth" else fish
            x = calendar_age(data.time, patient.birth) if aligned == "birth" else relative_months(data.time, fish)
            if aligned == "birth":
                start, stop, window = 0, 30, (float(calendar_age([fish], patient.birth)[0]), float(calendar_age([end], patient.birth)[0]))
            else:
                start, stop, window = -6, 6, (0, float(relative_months([end], fish)[0]))
            ax.axvspan(*window, color=GREEN, alpha=.14)
            ax.axvline(0 if aligned == "fish" else window[0], color="#222222", linestyle="--", linewidth=.8)
            for y, feature in enumerate(features):
                part = data.loc[data.feature_name.eq(feature)].copy(); values = pd.to_numeric(part.value)
                qualifying = part.time.gt(fish) & part.time.le(end) & values.eq(1)
                for mask, marker, color, size in [(values.eq(0), "o", "white", 18), (values.eq(1) & ~qualifying, "x", GRAY, 24), (qualifying, "*", RED, 75)]:
                    mask = mask.to_numpy(); xx = x[data.feature_name.eq(feature).to_numpy()]
                    ax.scatter(xx[mask], np.full(mask.sum(), y), marker=marker, s=size, facecolors=color if marker == "o" else None, edgecolors=GRAY if marker == "o" else color, linewidths=.7)
            ax.set_yticks(range(3), ["Fish introduction", "GI", "Vomiting"])
            ax.set_xlim(start, stop); ax.set_ylim(2.5, -.5); ax.grid(axis="x", color="#EEEEEE", linewidth=.5)
            ax.tick_params(labelsize=8)
            if row == 0: ax.set_title("Birth-aligned" if aligned == "birth" else "Fish-aligned", fontsize=12, fontweight="bold")
            if row == len(sample)-1: ax.set_xlabel("Age (calendar months)" if aligned == "birth" else "Months relative to fish introduction")
        axes[row, 0].text(-.18, .5, f"{patient.patient_id}\nR={int(patient.R)}\nFA={int(patient.FA)}", transform=axes[row, 0].transAxes, ha="right", va="center", fontsize=8)
    handles = [Line2D([], [], marker="D", color="#222222", linestyle="none", label="Fish introduction"), Line2D([], [], marker="*", color=RED, linestyle="none", markersize=9, label="GI/vomiting inside [fish, fish+1 month]"), Line2D([], [], marker="x", color=GRAY, linestyle="none", label="Outside ground-truth window"), Line2D([], [], marker="o", markerfacecolor="white", color=GRAY, linestyle="none", label="Recorded value 0")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .025), ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Figure 2 — Birth-aligned versus fish-aligned patient timelines", fontsize=17, y=.995)
    fig.text(.5, .002, "Sample rule: first four saved blocks containing documented R=1 and R=0 children; the same two children per block are shown in both columns. R uses actual timestamps; FA is sampled.", ha="center", fontsize=8.5, color="#555555")
    fig.subplots_adjust(left=.16, right=.98, top=.96, bottom=.08, hspace=.65, wspace=.12)
    return fig


def main():
    obs, patients, comparison, tree, on_diag, _ = load_artifacts()
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 120, "pdf.fonttype": 42, "axes.facecolor": "white"})
    fig1 = figure_one(obs, patients, comparison, tree, on_diag)
    fig2 = figure_two(obs, patients)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig1.savefig(OUTPUT / "figure_1_from_exposure_to_prediction.png", dpi=300, bbox_inches="tight")
    fig1.savefig(OUTPUT / "figure_1_from_exposure_to_prediction.pdf", bbox_inches="tight")
    fig2.savefig(OUTPUT / "figure_2_birth_vs_fish_aligned_timelines.png", dpi=300, bbox_inches="tight")
    fig2.savefig(OUTPUT / "figure_2_birth_vs_fish_aligned_timelines.pdf", bbox_inches="tight")
    plt.close(fig1); plt.close(fig2)
    print("Saved four figure files under", OUTPUT)


if __name__ == "__main__":
    main()
