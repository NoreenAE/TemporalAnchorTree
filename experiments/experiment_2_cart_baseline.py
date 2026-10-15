"""Static CART baseline for the recorded Experiment 2 Dynamic ON comparison.

Run manually from the repository root:
python experiments/experiment_2_cart_baseline.py

Reuse the immutable noisy matched V2 artifacts copied into the recorded TAT
result directory. The outcome is FA; GI is an observed clinical event feature.
Preprocessing in this file sums each event feature over the entire available
history. Unit-valued event rows give counts; an absent event gives zero. Dates,
birth anchors, event order, windows, dynamic anchors, private risk variables,
documentation metadata, and patient/block IDs never enter the feature matrix.

Each run creates a new directory under experiments/results/experiment_2_cart_baseline.
Training is performed only when main() is called, not when this file is imported.
Tree depth and minimum split/leaf sample counts match the recorded TAT settings.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
from uuid import uuid4

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier, plot_tree

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.experiment_2_noisy_v2 import load_dataset  # noqa: E402
from project_paths import portable_metadata  # noqa: E402
from experiments.experiment_utils import dataset_directory  # noqa: E402


RECORDED_RUN = ROOT / "experiments/results/experiment_2_noisy_matched_v2"
OUTPUT_ROOT = ROOT / "experiments/results/experiment_2_cart_baseline"
EVENT_FEATURES = (
    "FishIntroduction", "GI", "Vomiting", "MinorInjury", "Cough",
    "RoutineVaccination", "SleepDisturbance", "DoctorVisit",
)


def file_sha256(path: Path) -> str:
    """Hash an input artifact without changing it."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate_events(data: pd.DataFrame, patient_ids: pd.Index) -> pd.DataFrame:
    """Return one full-history event count per feature and patient.

    The benchmark contains event-occurrence rows with value=1, not repeated
    zero/one assessments. Consequently a mean of present rows would be one.
    Sum preserves the observed clinical counts without using chronology. Zero
    means no recorded occurrence, including absent FishIntroduction records;
    it is not a mean-based imputation learned from other patients.
    """
    if set(data["feature_name"]) != set(EVENT_FEATURES) | {"Birth Date Anchor"}:
        raise ValueError("Unexpected feature schema; inspect potential leakage.")
    events = data.loc[
        data["feature_name"].isin(EVENT_FEATURES),
        ["patient_id", "feature_name", "value"],
    ].copy()
    events["value"] = pd.to_numeric(events["value"], errors="raise")
    if not events["value"].eq(1).all():
        raise ValueError("Expected unit-valued event occurrences.")
    matrix = events.groupby(["patient_id", "feature_name"])["value"].sum().unstack()
    matrix = matrix.reindex(index=patient_ids, columns=EVENT_FEATURES).fillna(0)
    matrix.columns = [f"{name}__sum" for name in matrix.columns]
    if not matrix.index.is_unique or not np.isfinite(matrix.to_numpy()).all():
        raise ValueError("Invalid patient-level clinical matrix.")
    return matrix.astype(float)


def prepare_inputs(recorded_run: Path) -> tuple:
    """Validate the exact cohort, block split, labels, and saved Dynamic ON scores."""
    configuration = json.loads((recorded_run / "configuration.json").read_text())
    if not any(
        run["name"] == "dynamic_on" and run["USE_DYNAMIC_ANCHORS"]
        for run in configuration["runs"]
    ):
        raise ValueError("The reference run must contain Dynamic ON TAT.")
    data, patients, _ = load_dataset(recorded_run)
    info = patients.set_index("patient_id", verify_integrity=True)
    if set(data["patient_id"]) != set(info.index):
        raise ValueError("Observations and labels have different patients.")
    if not info["partition"].isin(["train", "test"]).all():
        raise ValueError("Invalid saved patient partitions.")
    if info.groupby("block")["partition"].nunique().ne(1).any():
        raise ValueError("Matched blocks cross the train/test boundary.")
    matrix = aggregate_events(data, info.index)
    train_ids = info.index[info["partition"].eq("train")]
    test_ids = info.index[info["partition"].eq("test")]
    if not len(train_ids) or not len(test_ids) or set(train_ids) & set(test_ids):
        raise ValueError("Expected nonempty, disjoint saved partitions.")

    reference = pd.read_csv(
        recorded_run / "dynamic_on/patient_predictions.csv",
        usecols=["patient_id", "partition", "FA", "score"],
        dtype={"patient_id": str},
    ).set_index("patient_id", verify_integrity=True)
    if set(reference.index) != set(info.index):
        raise ValueError("Saved TAT predictions have different patients.")
    reference = reference.reindex(info.index)
    if not reference["partition"].eq(info["partition"]).all():
        raise ValueError("CART and recorded TAT would use different patient splits.")
    if not reference["FA"].eq(info["FA"]).all():
        raise ValueError("Saved TAT labels differ from the benchmark labels.")
    if reference["score"].isna().any() or not reference["score"].between(0, 1).all():
        raise ValueError("Saved TAT scores are invalid.")
    for ids in (train_ids, test_ids):
        if info.loc[ids, "FA"].nunique() != 2:
            raise ValueError("Each partition must contain both FA classes.")
    print(f"Patients: {len(info)}; static CART features: {matrix.shape[1]}")
    print(f"Feature columns: {matrix.columns.tolist()}")
    for name, ids in (("train", train_ids), ("test", test_ids)):
        prevalence = info.loc[ids, "FA"].mean()
        print(f"{name}: {len(ids)} patients; FA prevalence={prevalence:.4f}")
    print("Verified: identical saved patients, block split, FA labels, and TAT scores.")
    print("Verified: only clinical event sums enter CART; no dates/anchors/labels/IDs.")
    return matrix, info, train_ids, test_ids, reference, configuration


def main(recorded_run: Path = RECORDED_RUN) -> None:
    """Fit one CART and save a separate comparison run without overwriting TAT."""
    recorded_run = recorded_run.resolve()
    data_dir = dataset_directory(recorded_run)
    sources = {
        "observations": data_dir / "observations.csv",
        "patients_and_split": data_dir / "patients_and_split.csv",
        "manifest": data_dir / "manifest.json",
        "configuration": recorded_run / "configuration.json",
        "dynamic_on_predictions": recorded_run / "dynamic_on/patient_predictions.csv",
        "baseline_script": Path(__file__),
    }
    hashes = {name: file_sha256(path) for name, path in sources.items()}
    matrix, info, train_ids, test_ids, reference, config = prepare_inputs(recorded_run)
    parameters = {
        "max_depth": config["parameters"]["max_depth"],
        "min_samples_split": config["parameters"]["min_samples_split"],
        "min_samples_leaf": config["parameters"]["min_samples_leaf"],
        "criterion": "gini",
        "random_state": config["parameters"]["random_state"],
    }
    print(f"CART model parameters: {parameters}")
    train, test = matrix.loc[train_ids], matrix.loc[test_ids]
    y_train, y_test = info.loc[train_ids, "FA"], info.loc[test_ids, "FA"]
    model = DecisionTreeClassifier(**parameters)
    model.fit(train, y_train)
    train_scores = model.predict_proba(train)[:, 1]
    test_scores = model.predict_proba(test)[:, 1]
    cart_train_auc = float(roc_auc_score(y_train, train_scores))
    cart_test_auc = float(roc_auc_score(y_test, test_scores))
    tat_train_auc = float(roc_auc_score(y_train, reference.loc[train_ids, "score"]))
    tat_test_auc = float(roc_auc_score(y_test, reference.loc[test_ids, "score"]))
    results = pd.DataFrame([
        {"outcome": "FA", "model": "CART", "train_roc_auc": cart_train_auc,
         "test_roc_auc": cart_test_auc},
        {"outcome": "FA", "model": "TAT_dynamic_on_saved",
         "train_roc_auc": tat_train_auc, "test_roc_auc": tat_test_auc},
    ])
    for name, path in sources.items():
        if file_sha256(path) != hashes[name]:
            raise RuntimeError(f"{name} changed during the run; outputs not saved.")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = OUTPUT_ROOT / f"run_{stamp}_{uuid4().hex[:8]}"
    output.mkdir(exist_ok=False)
    results.to_csv(output / "results.csv", index=False)
    pd.DataFrame({
        "patient_id": test_ids, "FA_label": y_test.to_numpy(),
        "CART_probability": test_scores,
        "TAT_dynamic_on_probability": reference.loc[test_ids, "score"].to_numpy(),
    }).to_csv(output / "test_predictions.csv", index=False)

    figure, axis = plt.subplots()
    for name, scores, auc in (
        ("CART", test_scores, cart_test_auc),
        ("TAT Dynamic ON", reference.loc[test_ids, "score"], tat_test_auc),
    ):
        fpr, tpr, _ = roc_curve(y_test, scores)
        axis.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")
    axis.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
    axis.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
             title="Experiment 2 – CART vs TAT Dynamic ON")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "roc_cart_vs_dynamic_on.png", dpi=300)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(20, 10))
    plot_tree(model, feature_names=matrix.columns.tolist(),
              class_names=["FA=0", "FA=1"],
              filled=True, rounded=True, precision=2, fontsize=8, ax=axis)
    axis.set_title("Experiment 2 – Static CART – FA")
    figure.tight_layout()
    figure.savefig(output / "cart_tree.png", dpi=300, bbox_inches="tight")
    plt.close(figure)
    metadata = {
        "outcome": "FA", "aggregation": "one full-history sum per clinical event",
        "model_parameters": parameters, "effective_tree_parameters": model.get_params(),
        "reference": "saved Dynamic ON scores; TAT is not retrained",
        "evaluation_population": "all saved test patients, including undocumented",
        "train_patient_ids": train_ids.tolist(), "test_patient_ids": test_ids.tolist(),
        "feature_columns": matrix.columns.tolist(),
        "python_executable": Path(sys.executable).name, "python_version": sys.version,
        "package_versions": {name: version(name) for name in
                             ("numpy", "pandas", "scikit-learn", "matplotlib")},
        "input_paths": {name: str(path) for name, path in sources.items()},
        "input_sha256": hashes,
        "results": results.to_dict("records"),
    }
    (output / "run_metadata.json").write_text(
        json.dumps(portable_metadata(metadata), indent=2, sort_keys=True,
                   allow_nan=False) + "\n"
    )
    print("\nExperiment 2 – CART vs TAT Dynamic ON (FA)")
    for result in results.to_dict("records"):
        print(f"{result['model']}: train AUC={result['train_roc_auc']:.2f}; "
              f"test AUC={result['test_roc_auc']:.2f}")
    print(f"Saved new run: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recorded-run", type=Path, default=RECORDED_RUN)
    main(recorded_run=parser.parse_args().recorded_run)
