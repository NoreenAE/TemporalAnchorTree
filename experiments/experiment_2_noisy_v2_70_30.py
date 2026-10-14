"""Run Experiment 2 ON/OFF on the unchanged cohort with a 70/30 block split.

Run from the repository root with the research Python environment::

    python -m experiments.experiment_2_noisy_v2_70_30

Use --prepare-only to validate and save the new split without training.
Existing data/results are never overwritten. Only partition assignments change;
the original observations, labels, model parameters and seed 42 are retained.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from experiments import experiment_2_noisy_v2 as baseline
from generate_fish_intro_noisy_v2 import (
    descriptive_audit, save_dataset, validate_dataset,
)

DATA = baseline.ROOT / "data/fish_intro_noisy_matched_v2_70_30"
OUTPUT = baseline.ROOT / "experiments/results/experiment_2_noisy_matched_v2_70_30"


def sha256(path: Path) -> str:
    """Return the checksum of a saved research artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_split() -> Path:
    """Validate and save a new partition of the original 500 matched blocks."""
    if DATA.exists():
        raise FileExistsError(DATA)
    source = baseline.DEFAULT_OUTPUT
    saved = json.loads((source / "configuration.json").read_text())
    current = deepcopy(baseline.MODEL_CONFIG)
    current.pop("aggregation_funcs")
    if (current != saved["parameters"] or saved["aggregations"] != ["sum"]
            or list(baseline.MODEL_RUNS) != saved["runs"]):
        raise ValueError("Model parameters differ from the saved Experiment 2")
    data, patients, original_validation = baseline.load_dataset(source)
    config = deepcopy(original_validation["config"])
    # Reuse exactly the generator's independent split RNG stream, not its
    # observation/label generators. Reproduce 80/20 before drawing 70/30.
    blocks = np.arange(config["blocks"])
    if patients.block.nunique() != len(blocks):
        raise ValueError("Unexpected matched block count")
    split_stream = np.random.SeedSequence(config["seed"]).spawn(6)[4]
    original_test = np.random.default_rng(split_stream).choice(
        blocks, round(len(blocks) * config["test_fraction"]), replace=False)
    if set(original_test) != set(patients.loc[patients.partition.eq("test"), "block"]):
        raise ValueError("Cannot reproduce the original split")
    test_blocks = np.random.default_rng(split_stream).choice(
        blocks, round(len(blocks) * 0.3), replace=False)
    changed = patients.copy()
    changed["partition"] = np.where(changed.block.isin(test_blocks), "test", "train")
    pd.testing.assert_frame_equal(
        patients.drop(columns="partition"), changed.drop(columns="partition"))
    if changed.partition.value_counts().to_dict() != {"train": 3500, "test": 1500}:
        raise ValueError("Expected 3,500 train and 1,500 test children")
    config["test_fraction"] = 0.3
    validation = validate_dataset(data, changed)
    validation.update(version=original_validation["version"], config=config)
    validation["split_provenance"] = {
        "source": str(source),
        "source_sha256": {name: sha256(source / name) for name in baseline.DATASET_FILES},
        "script_sha256": sha256(Path(__file__)),
        "changed_fields": ["partition"],
        "split_rng": "SeedSequence(seed=42).spawn(6)[4]; choice without replacement",
        "train_children": 3500, "test_children": 1500,
        "original_test_blocks_retained": len(set(original_test) & set(test_blocks)),
    }
    audit = descriptive_audit(data, changed)
    save_dataset(DATA, data, changed, validation, audit)
    # Preserve exact original observation bytes, including CSV formatting.
    shutil.copyfile(source / "observations.csv", DATA / "observations.csv")
    manifest = json.loads((DATA / "manifest.json").read_text())
    manifest["observations.csv"] = sha256(DATA / "observations.csv")
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    reloaded_data, reloaded_patients, _ = baseline.load_dataset(DATA)
    pd.testing.assert_frame_equal(data, reloaded_data)
    pd.testing.assert_frame_equal(changed, reloaded_patients)
    return DATA


def main() -> None:
    """Prepare a separate dataset and optionally run the existing ON/OFF runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if OUTPUT.exists():
        parser.error(f"Output already exists: {OUTPUT}")
    directory = prepare_split()
    print(f"Prepared {directory}: 3,500 train / 1,500 test", flush=True)
    if not args.prepare_only:
        subprocess.run(
            [sys.executable, "-B", "-u", "-m", "experiments.experiment_2_noisy_v2",
             "--data-dir", str(directory), "--output", str(OUTPUT)],
            cwd=baseline.ROOT, check=True,
        )


if __name__ == "__main__":
    main()
