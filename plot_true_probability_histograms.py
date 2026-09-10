"""Plot saved true-probability distributions for the synthetic benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_DATA_DIR = Path("data/temporal_benchmark_v3")
FA_GROUND_TRUTH = "temporal_benchmark_ground_truth_5000.pkl"
GI_GROUND_TRUTH = "temporal_benchmark_gi_ground_truth_5000.pkl"
FA_FIGURE = "true_fa_probability_histogram.png"
GI_FIGURE = "true_gi_probability_histogram.png"
COMBINED_FIGURE = "true_probability_histogram_fa_vs_gi.png"


def _load_probabilities(path: Path, column: str) -> pd.Series:
    """Load and validate a saved patient-level probability column."""
    ground_truth = pd.read_pickle(path)
    if column not in ground_truth:
        raise ValueError(f"Missing {column!r} in {path}.")

    probabilities = ground_truth[column]
    if probabilities.isna().any() or not probabilities.between(0, 1).all():
        raise ValueError(f"{column!r} in {path} must contain values in [0, 1].")
    return probabilities


def _save_histogram(
    probabilities: pd.Series,
    output_path: Path,
    xlabel: str,
    title: str,
    color: str,
) -> None:
    """Save a clean, publication-quality true-probability histogram."""
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.hist(
        probabilities,
        bins=np.linspace(0, 1, 31),
        color=color,
        edgecolor="white",
        linewidth=0.7,
    )
    axis.set_xlim(0, 1)
    axis.set_xlabel(xlabel, fontsize=12)
    axis.set_ylabel("Number of patients", fontsize=12)
    axis.set_title(title, fontsize=14, pad=12)
    axis.grid(axis="y", alpha=0.25, linewidth=0.8)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def create_probability_histograms(data_dir: Path) -> tuple[Path, Path]:
    """Create FA and GI histograms from existing saved probabilities."""
    fa_probabilities = _load_probabilities(
        data_dir / FA_GROUND_TRUTH,
        "fa_probability",
    )
    gi_probabilities = _load_probabilities(
        data_dir / GI_GROUND_TRUTH,
        "gi_probability",
    )

    fa_output = data_dir / FA_FIGURE
    gi_output = data_dir / GI_FIGURE
    _save_histogram(
        fa_probabilities,
        fa_output,
        "True P(FA)",
        "Distribution of True FA Probabilities",
        "#4472A5",
    )
    _save_histogram(
        gi_probabilities,
        gi_output,
        "True P(GI)",
        "Distribution of True GI Probabilities",
        "#C46A3A",
    )
    return fa_output, gi_output


def create_combined_probability_histogram(data_dir: Path) -> Path:
    """Overlay saved FA and GI probabilities using identical bins."""
    fa_probabilities = _load_probabilities(
        data_dir / FA_GROUND_TRUTH,
        "fa_probability",
    )
    gi_probabilities = _load_probabilities(
        data_dir / GI_GROUND_TRUTH,
        "gi_probability",
    )

    output_path = data_dir / COMBINED_FIGURE
    bins = np.linspace(0, 1, 31)
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.hist(
        fa_probabilities,
        bins=bins,
        color="green",
        alpha=0.55,
        label="FA",
    )
    axis.hist(
        gi_probabilities,
        bins=bins,
        color="red",
        alpha=0.55,
        label="GI",
    )
    axis.set_xlim(0, 1)
    axis.set_xlabel("True Probability", fontsize=12)
    axis.set_ylabel("Number of patients", fontsize=12)
    axis.set_title(
        "Distribution of True Outcome Probabilities",
        fontsize=14,
        pad=12,
    )
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25, linewidth=0.8)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Save one overlaid FA-versus-GI histogram.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Create both true-probability histograms."""
    args = parse_args(argv)
    if args.combined:
        print(create_combined_probability_histogram(args.data_dir).resolve())
    else:
        for output_path in create_probability_histograms(args.data_dir):
            print(output_path.resolve())


if __name__ == "__main__":
    main()
