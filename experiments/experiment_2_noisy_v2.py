"""Run the saved noisy V2 benchmark using the unchanged Experiment 2 ON/OFF code.

Dataset generation is separate. Importing this module never trains or predicts.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from generate_fish_intro_noisy_v2 import (
    BIRTH, OBSERVATION_COLUMNS, VERSION, validate_dataset,
)
from experiments.experiment_2_fish_introduction_dynamic_anchors import (
    run_model, write_json,
)
from experiments.experiment_2_outputs import save_experiment_outputs

# Experiment 2 V2 configuration: edit experiment settings here.
# Partitions and labels come from the saved dataset; no new split is sampled.
DEFAULT_DATA_DIR = ROOT / "data/fish_intro_noisy_matched_v2"
DEFAULT_OUTPUT = ROOT / "experiments/results/experiment_2_noisy_matched_v2"
MODEL_RUNS = (
    dict(name="dynamic_on", USE_DYNAMIC_ANCHORS=True),
    dict(name="dynamic_off", USE_DYNAMIC_ANCHORS=False),
)
MODEL_CONFIG = dict(
    max_depth=3,
    min_samples_split=100,
    min_samples_leaf=50,
    min_impurity_decrease=0.0,
    is_classification=False,  # Regression on binary FA; leaf means are probabilities.
    criterion="mse",
    candidate_sampling_rate=0.4,
    random_state=42,
    split_sampling_seed=42,
    anchors_list=["Birth Date Anchor"],
    anchor_offset_list=[1, 2, 3, 4],  # Calendar months for dynamic anchors.
    anchor_offset_map={"Birth Date Anchor": [*range(1, 23), None]},
    # None means full timeline. Preserve sum-only: the shared trace audit
    # assumes unit-valued event counts; other aggregations need audit review.
    aggregation_funcs={"sum": np.sum},
    feature_semantics={
        feature: "binary_event_indicator"
        for feature in (
            "FishIntroduction", "GI", "Vomiting", "MinorInjury", "Cough",
            "RoutineVaccination", "SleepDisturbance", "DoctorVisit",
        )
    },
    learn_missing_direction=False,  # Empty aggregates route right.
    verbose=0,
)
# Evaluation remains the shared protocol: ROC-AUC, Brier, and 10-bin calibration.
# End of editable experiment configuration.

DATASET_FILES = ("observations.csv", "patients_and_split.csv", "validation.json",
                 "descriptive_audit.json", "manifest.json")


def load_dataset(directory: Path) -> tuple:
    """Load one immutable cohort for both switches; validate schema and truth."""
    directory = Path(directory)
    validation = json.loads((directory / "validation.json").read_text())
    if validation.get("version") != VERSION:
        raise ValueError("Expected noisy matched V2 data, not a historical renamed dataset")
    manifest = json.loads((directory / "manifest.json").read_text())
    for name in DATASET_FILES[:-1]:
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != manifest[name]:
            raise ValueError(f"Dataset hash mismatch: {name}")
    data = pd.read_csv(directory / "observations.csv", parse_dates=["time"],
                       dtype={"patient_id": str, "value": object})
    if list(data.columns) != OBSERVATION_COLUMNS:
        raise ValueError("Unexpected predictor columns")
    birth = data.feature_name.eq(BIRTH)
    data.loc[birth, "value"] = pd.to_datetime(data.loc[birth, "value"])
    data.loc[~birth, "value"] = pd.to_numeric(data.loc[~birth, "value"])
    patients = pd.read_csv(directory / "patients_and_split.csv",
                           parse_dates=["birth", "followup_end", "fish_time", "a", "b"],
                           dtype={"patient_id": str, "documented": bool})
    validate_dataset(data, patients)
    return data, patients, validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; select a new directory")
    data, patients, validation = load_dataset(args.data_dir)
    args.output.mkdir(parents=True, exist_ok=False)
    for name in DATASET_FILES:
        shutil.copyfile(args.data_dir / name, args.output / name)
    parameters = deepcopy(MODEL_CONFIG)
    parameters.pop("aggregation_funcs")
    sources = (Path(__file__), ROOT / "generate_fish_intro_noisy_v2.py",
               ROOT / "experiments/experiment_2_fish_introduction_dynamic_anchors.py",
               ROOT / "experiments/experiment_2_outputs.py",
               ROOT / "settree/temporal_tree.py", ROOT / "settree/anchor_utils.py")
    write_json(args.output / "configuration.json", dict(
        dataset_version=VERSION, parameters=parameters, aggregations=list(MODEL_CONFIG["aggregation_funcs"]),
        runs=MODEL_RUNS, dataset_directory=str(args.data_dir.resolve()),
        dataset_config=validation["config"],
        split="80/20 blocks by default; both runs use identical saved assignments",
        sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sources},
    ))
    results = {}
    for run in MODEL_RUNS:
        output = args.output / run["name"]
        output.mkdir()
        results[run["name"]] = run_model(
            run["USE_DYNAMIC_ANCHORS"], data, patients, output,
            model_config=MODEL_CONFIG,
        )
    write_json(args.output / "comparison.json", results)
    save_experiment_outputs(args.output)


if __name__ == "__main__":
    main()
