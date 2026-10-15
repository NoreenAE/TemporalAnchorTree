"""Run one candidate-sampling rate using the saved final Experiment 1.

From the repository root: python3 experiments/finding_best_A_for_exp_1.py --A 0.8
Only A changes scientifically; each invocation trains FA and GI. The saved
baseline's actual aggregations include count, omitted from its config metadata.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Iterator, Optional, Sequence
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import experiment_1_fa_vs_gi as experiment  # noqa: E402
from project_paths import resolve_path  # noqa: E402

BASELINE = ROOT / "supporting_analyses/candidate_sampling/baseline_A1"
RESULTS_ROOT = ROOT / "supporting_analyses/candidate_sampling/results"
ALLOWED_RATES = (0.8, 0.4, 0.2)


@contextmanager
def configured_experiment(rate: float) -> Iterator[Path]:
    """Temporarily configure the existing pipeline from the saved baseline.

    Input hashes and patient membership are checked before the reused runner
    creates outputs or fits a model. No baseline artifact is written.
    """
    if rate not in ALLOWED_RATES:
        raise ValueError(f"A must be one of {ALLOWED_RATES}.")
    output = RESULTS_ROOT / f"A_{rate:.1f}"
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite an existing run: {output}")
    if output.resolve() != output:
        raise ValueError("The output path must not redirect through symlinks.")

    config = json.loads((BASELINE / "config.json").read_text(encoding="utf-8"))
    parameters = deepcopy(config["model_hyperparameters"])
    parameters["candidate_sampling_rate"] = rate
    input_paths = {key: resolve_path(value) for key, value in config["input_paths"].items()}
    load_inputs = experiment.load_inputs
    create_split = experiment.create_shared_split

    def checked_inputs():
        for key, path in input_paths.items():
            if experiment.sha256_file(path) != config["input_sha256"][key]:
                raise ValueError(f"Input differs from final Experiment 1: {path}")
        return load_inputs()

    def checked_split(data, labels):
        train_ids, test_ids, split = create_split(data, labels)
        saved = experiment.pd.read_csv(BASELINE / "shared_patient_split.csv")
        experiment.pd.testing.assert_frame_equal(split, saved)
        return train_ids, test_ids, split

    # Preserve the actual fitted-model behavior, including count. Do not use
    # AGGREGATION_NAMES from the old metadata to filter the model's functions.
    aggregation_names = ["mean", "sum", "max", "count"]
    if list(experiment.aggregation_functions()) != aggregation_names:
        raise ValueError("Experiment 1's aggregation functions have changed.")

    overrides = {
        "EXPERIMENT_NAME": f"finding_best_A_for_exp_1_A_{rate:.1f}",
        "EXPERIMENT_DESCRIPTION": (
            f"Final Experiment 1 with candidate_sampling_rate={rate:.1f}; "
            "all other training and evaluation settings are preserved."
        ),
        "OUTPUT_DIR": output,
        "INPUT_PATHS": input_paths,
        "OUTCOMES": deepcopy(config["outcomes"]),
        "MODEL_HYPERPARAMETERS": parameters,
        "ANCHORS_LIST": parameters["anchors_list"],
        "ANCHOR_OFFSET_LIST": parameters["anchor_offset_list"],
        "ANCHOR_OFFSET_MAP": parameters["anchor_offset_map"],
        "AGGREGATION_NAMES": aggregation_names,
        "TEMPORAL_EXPLAINABILITY_BIN_EDGES": config[
            "temporal_explainability_bin_edges"
        ],
        "RANDOM_STATE": config["random_state"],
        "SPLIT_SAMPLING_SEED": config["split_sampling_seed"],
        "CANDIDATE_SAMPLING_RATE": rate,
        "MIN_IMPURITY_DECREASE": config["min_impurity_decrease"],
        "USE_DYNAMIC_ANCHORS": config["use_dynamic_anchors"],
        "VERBOSE": parameters["verbose"],
        "TEST_SIZE": config["test_size"],
        "DECISION_THRESHOLD": config["decision_threshold"],
        "load_inputs": checked_inputs,
        "create_shared_split": checked_split,
    }
    with patch.multiple(experiment, **overrides):
        yield output


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Require exactly one of the three new sampling rates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--A", type=float, choices=ALLOWED_RATES, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run both outcomes for the single manually selected A value."""
    args = parse_args(argv)
    with configured_experiment(args.A):
        experiment.run_experiment()


if __name__ == "__main__":
    main()
