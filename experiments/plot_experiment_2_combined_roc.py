"""Plot seeds 42–47 from saved Experiment 2/2.1 scores, without model calls.

Run from the repository root::

    python -m experiments.plot_experiment_2_combined_roc

Only a new PNG is written; existing figures and result artifacts are read-only.
The population matches Experiment 2's ROC: all held-out test patients.
"""
from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve

RESULTS = Path(__file__).resolve().parent / "results"
BASELINE = RESULTS / "experiment_2_noisy_matched_v2"
SWEEP = RESULTS / "experiment_2_1_candidate_sampling_robustness" / "seeds"
OUTPUT = BASELINE / "roc_on_vs_off_across_six_seeds.png"
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")


def save_combined_roc(output: Path | None = None) -> Path:
    """Save twelve ROC curves using only persisted labels and model scores.

    Require identical held-out IDs and labels across all runs. Exclusive file
    creation prevents replacement of an existing combined figure.
    """
    output = Path(output) if output is not None else OUTPUT
    if output.exists():
        raise FileExistsError(output)
    curves = []
    reference = None
    for seed, color in zip(range(42, 48), COLORS):
        folder = BASELINE if seed == 42 else SWEEP / f"seed_{seed}"
        for run, style, name in (
            ("dynamic_on", "-", "Dynamic ON"),
            ("dynamic_off", "--", "Dynamic OFF"),
        ):
            frame = pd.read_csv(
                folder / run / "patient_predictions.csv",
                usecols=["patient_id", "partition", "FA", "score"],
                dtype={"patient_id": str},
            )
            test = frame.loc[frame.partition.eq("test")].set_index(
                "patient_id", verify_integrity=True).sort_index()
            if test.empty or set(test.FA) != {0, 1}:
                raise ValueError(f"Seed {seed}/{run}: expected both test classes")
            if test.score.isna().any() or not test.score.between(0, 1).all():
                raise ValueError(f"Seed {seed}/{run}: invalid saved scores")
            if reference is None:
                reference = test.FA
            elif not test.FA.equals(reference):
                raise ValueError(f"Seed {seed}/{run}: test IDs or labels differ")
            fpr, tpr, _ = roc_curve(test.FA, test.score)
            label = f"{name} – Seed 42" if seed == 42 else "_nolegend_"
            curves.append((fpr, tpr, color, style, label))

    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
    }):
        figure, axis = plt.subplots(figsize=(7, 7), facecolor="white")
        try:
            for fpr, tpr, color, style, label in curves:
                axis.plot(fpr, tpr, color=color, linestyle=style,
                          linewidth=1.8, label=label)
            axis.plot([0, 1], [0, 1], color="0.5", linestyle=":",
                      linewidth=1, label="_nolegend_", zorder=0)
            axis.set(
                xlabel="False Positive Rate", ylabel="True Positive Rate",
                title="ROC Curves – Dynamic ON vs OFF Across Seeds",
                xlim=(0, 1), ylim=(0, 1), aspect="equal",
            )
            axis.set_axisbelow(True)
            axis.grid(color="0.9", linewidth=0.6)
            axis.legend(loc="lower right", frameon=False)
            figure.tight_layout()
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("xb") as handle:
                figure.savefig(handle, format="png", dpi=300,
                               bbox_inches="tight", facecolor="white")
        finally:
            plt.close(figure)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    print(save_combined_roc(parser.parse_args().output))
