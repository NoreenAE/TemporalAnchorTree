"""Static FA/GI CART (sklearn DecisionTreeClassifier) baseline for Experiment 1.

Run from the repository root:
python experiments/experiment_1_xgboost_baseline/experiment1_cart_baseline.py
Add --verify-reproducibility to fit each model twice and compare predictions.
Requires the existing project dependencies. The ROC PNG, results
CSV, run_metadata JSON, and separate FA/GI tree PNGs are written beside the
script when main() runs. Output filenames start with experiment1_cart_baseline.

Clinical values use one mean per feature over every available observation, without
dates, observation order, windows, anchors, or observation-count features.
These summaries can retain clinical associations with age, but contain no
explicit timing information. No ground-truth probabilities or groups are read.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
from typing import Dict, Tuple

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier, plot_tree

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Direct execution puts this experiment folder on sys.path, not the repository.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments import experiment_1_fa_vs_gi as experiment  # noqa: E402


OUTPUT_DIR = Path(__file__).resolve().parent
ANCHORS = {"Birth Date Anchor", "NOW Anchor"}
# Explicitly audited Experiment 1 schema: reject additions rather than silently
# feeding a new label, date, offset, or categorical representation to the model.
NUMERIC_FEATURES = {
    "ApgarFiveMinutes", "BirthOrder", "BirthWeightKg", "GestationalAgeWeeks",
    "LengthCm", "MaternalAgeYears", "SocioEconomicScore", "WeightKg",
}
BINARY_FEATURES = {
    "BreastMilkYN", "ClinicalVisitYN", "DiarrheaYN", "FoodExposureYN",
    "FormulaYN", "GISymptomYN", "VomitingYN",
}
# Match the Experiment 1 TAT depth; preserve the baseline's existing model seed.
MODEL_PARAMETERS = {
    "max_depth": experiment.MODEL_HYPERPARAMETERS["max_depth"],
    "criterion": "gini",
    "random_state": 42,
}


def load_patient_split(
    data: pd.DataFrame, labels: Dict[str, pd.DataFrame]
) -> Tuple[np.ndarray, np.ndarray]:
    """Reuse recorded patient membership; reproduce source logic if absent.

    The saved run used seed 42; the current source specifies seed 48. Thus
    rerunning the current split function would change the published comparison.
    Saved membership takes precedence, with input hashes and labels checked.
    Both original label files are validated to preserve the historical
    joint-stratified split, and each outcome is trained independently.
    """
    split_path = experiment.OUTPUT_DIR / "shared_patient_split.csv"
    if not split_path.exists():
        train_ids, test_ids, _ = experiment.create_shared_split(data, labels)
        print(
            "No recorded split found: reusing Experiment 1 create_shared_split "
            f"(seed={experiment.RANDOM_STATE}, test_size={experiment.TEST_SIZE})."
        )
        return train_ids, test_ids

    config_path = experiment.OUTPUT_DIR / "config.json"
    config = json.loads(config_path.read_text())
    for name, path in experiment.INPUT_PATHS.items():
        if experiment.sha256_file(path) != config["input_sha256"][name]:
            raise ValueError(
                f"{name} input differs from the recorded Experiment 1 run; "
                "stop to resolve dataset provenance before comparison."
            )
    split = pd.read_csv(
        split_path, dtype={"patient_id": str, "joint_stratum": str}
    )
    required = {"patient_id", "fa_label", "gi_label", "joint_stratum", "split"}
    if set(split.columns) != required or split.isna().any().any():
        raise ValueError("Recorded split has an unexpected schema or missing data.")
    if split["patient_id"].duplicated().any():
        raise ValueError("Recorded split contains duplicate patients.")
    if set(split["patient_id"]) != set(data["patient_id"]):
        raise ValueError("Recorded split and X do not contain identical patients.")
    if not split["split"].isin(["train", "test"]).all():
        raise ValueError("Recorded split contains invalid partition names.")
    for outcome in ("fa", "gi"):
        expected = split["patient_id"].map(
            labels[outcome].set_index("patient_id")["label"]
        )
        if not split[f"{outcome}_label"].eq(expected).all():
            raise ValueError(f"Recorded {outcome.upper()} labels differ from input.")
    joint = split["fa_label"].astype(str) + "_" + split["gi_label"].astype(str)
    if not split["joint_stratum"].eq(joint).all():
        raise ValueError("Recorded joint FA/GI strata are inconsistent.")
    train_ids = split.loc[split["split"].eq("train"), "patient_id"].to_numpy()
    test_ids = split.loc[split["split"].eq("test"), "patient_id"].to_numpy()
    if not len(train_ids) or not len(test_ids):
        raise ValueError("Both recorded partitions must contain patients.")
    print(f"Exact recorded Experiment 1 split: {split_path}")
    print(
        f"Recorded seed={config['random_state']}; "
        f"current source seed={experiment.RANDOM_STATE}; input hashes verified."
    )
    return train_ids, test_ids


def aggregate_clinical_values(
    data: pd.DataFrame, train_ids: np.ndarray
) -> pd.DataFrame:
    """Return one full-timeline mean per clinical feature for each patient.

    All clinical features in this dataset are numeric; there are no categorical
    strings to encode. Numeric measurements use their arithmetic mean; binary
    indicators use the proportion of nonmissing observations that are positive.
    Keep exactly one column per clinical feature, including the constant
    ClinicalVisitYN mean. No feature selection, counts, or chronology are used.
    Means skip missing observations; all-missing/absent values stay NaN, which
    CART (sklearn DecisionTreeClassifier) handles without imputation in the
    project's sklearn 1.6.1 environment. train_ids only validates membership;
    each patient's aggregation is independent of every other patient's values.
    """
    expected = NUMERIC_FEATURES | BINARY_FEATURES | ANCHORS
    if set(data["feature_name"]) != expected:
        raise ValueError(
            "Unexpected feature schema: stop and inspect possible temporal "
            "or label leakage before updating the clinical allowlist."
        )
    # Select only these three fields BEFORE aggregation: time and anchor values
    # never reach the aggregation, nor any fitted estimator.
    clinical = data.loc[
        data["feature_name"].isin(NUMERIC_FEATURES | BINARY_FEATURES),
        ["patient_id", "feature_name", "value"],
    ].copy()
    clinical["value"] = pd.to_numeric(clinical["value"], errors="raise")
    if np.isinf(clinical["value"].to_numpy(dtype=float)).any():
        raise ValueError("Clinical values contain infinity.")
    binary_values = clinical.loc[
        clinical["feature_name"].isin(BINARY_FEATURES), "value"
    ].dropna()
    if not binary_values.isin([0, 1]).all():
        raise ValueError("A binary clinical feature contains a non-binary value.")
    matrix = clinical.groupby(["patient_id", "feature_name"])["value"].mean().unstack(
        "feature_name"
    )
    matrix.columns = [f"{feature}__mean" for feature in matrix.columns]
    matrix = matrix.reindex(sorted(data["patient_id"].unique()))
    matrix = matrix.sort_index(axis=1).astype(float)
    if not set(train_ids).issubset(matrix.index):
        raise ValueError("Training patients are missing from the static matrix.")
    if not matrix.index.is_unique or matrix.shape[1] == 0:
        raise ValueError("Static matrix must have unique patients and features.")
    if any(
        column.split("__")[0] not in NUMERIC_FEATURES | BINARY_FEATURES
        for column in matrix
    ):
        raise ValueError("Unexpected model input: possible temporal/target leakage.")
    print("Aggregation: one full-timeline mean per clinical feature.")
    return matrix


def prepare_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.Series]]:
    """Load the exact cohort, historical split, clinical matrix, and FA/GI targets."""
    data, labels = experiment.load_inputs()
    train_ids, test_ids = load_patient_split(data, labels)
    matrix = aggregate_clinical_values(data, train_ids)
    train = matrix.loc[train_ids]
    test = matrix.loc[test_ids]
    targets = {}
    print(f"Patients: {len(matrix)}; CART features: {matrix.shape[1]}")
    print(f"First columns: {matrix.columns[:8].tolist()}")
    print(f"Train patients: {len(train)}; test patients: {len(test)}")
    for outcome in ("fa", "gi"):
        target = labels[outcome].set_index("patient_id")["label"]
        targets[outcome] = target.reindex(matrix.index)
        if targets[outcome].isna().any():
            raise ValueError(f"Missing {outcome.upper()} patient labels.")
        for name, frame in (("train", train), ("test", test)):
            aligned = targets[outcome].loc[frame.index]
            if not aligned.index.equals(frame.index) or aligned.nunique() != 2:
                raise ValueError("Labels must align and contain both classes.")
            print(f"{outcome.upper()} {name} positive prevalence: {aligned.mean():.4f}")
    print("Verified: one row per patient; FA/GI labels aligned by patient ID.")
    print(
        "Verified: no time/date, Birth Date, NOW, anchors, temporal offsets, "
        "windows, observation counts, patient IDs, or target labels in model X."
    )
    return train, test, targets


def dataframe_sha256(frame: pd.DataFrame) -> str:
    """Hash values, patient order, column order, and dtypes for run comparison."""
    schema = json.dumps({
        "columns": frame.columns.tolist(),
        "dtypes": frame.dtypes.astype(str).tolist(),
    }, sort_keys=True).encode()
    digest = hashlib.sha256(schema)
    digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return digest.hexdigest()


def save_tree_figure(
    model: DecisionTreeClassifier,
    feature_names: list,
    outcome: str,
    path: Path,
) -> None:
    """Plot one fitted CART tree with feature names and outcome class labels."""
    figure, axis = plt.subplots(figsize=(20, 10))
    try:
        plot_tree(
            model,
            feature_names=feature_names,
            class_names=[f"{outcome.upper()}={int(label)}" for label in model.classes_],
            filled=True,
            rounded=True,
            precision=2,
            fontsize=8,
            ax=axis,
        )
        axis.set_title(f"Experiment 1 – CART – {outcome.upper()}")
        figure.tight_layout()
        figure.savefig(path, dpi=300, bbox_inches="tight")
    finally:
        plt.close(figure)


def main(verify_reproducibility: bool = False) -> None:
    """Fit FA/GI separately and save train/test AUCs, ROC, trees, and provenance."""
    provenance_paths = {
        **experiment.INPUT_PATHS,
        "baseline_script": Path(__file__),
        "experiment_1_script": Path(experiment.__file__),
    }
    split_path = experiment.OUTPUT_DIR / "shared_patient_split.csv"
    if split_path.exists():
        provenance_paths["patient_split"] = split_path
    input_hashes = {
        name: experiment.sha256_file(path)
        for name, path in provenance_paths.items()
    }
    train, test, targets = prepare_inputs()
    metadata = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "package_versions": {
            name: version(name)
            for name in ("numpy", "pandas", "scikit-learn", "matplotlib")
        },
        "model_parameters": MODEL_PARAMETERS,
        "aggregation": "one full-timeline mean per clinical feature",
        "feature_columns": train.columns.tolist(),
        "train_patient_count": len(train),
        "test_patient_count": len(test),
        "matrix_sha256": {
            "train": dataframe_sha256(train),
            "test": dataframe_sha256(test),
        },
        "input_paths": {
            name: str(path.resolve()) for name, path in provenance_paths.items()
        },
        "input_sha256": input_hashes,
        "repeated_fit_verification_requested": verify_reproducibility,
        "outcomes": {},
    }
    results = []
    models = {}
    figure, axis = plt.subplots()
    print("\n================================")
    print("CART (sklearn DecisionTreeClassifier) static baseline – FA/GI")
    print("================================")
    print("Training FA/GI separately with the original Experiment 1 patient split.")
    print(f"Python: {sys.executable}; scikit-learn: {version('scikit-learn')}")
    print(f"Model parameters: {MODEL_PARAMETERS}")
    for outcome in ("fa", "gi"):
        model = DecisionTreeClassifier(**MODEL_PARAMETERS)
        # Fixed tree settings; fit only the original training patients.
        y_train = targets[outcome].loc[train.index]
        model.fit(train, y_train)
        models[outcome] = model
        train_probabilities = model.predict_proba(train)[:, 1]
        train_auc = float(roc_auc_score(y_train, train_probabilities))
        probabilities = model.predict_proba(test)[:, 1]
        if verify_reproducibility:
            repeated = DecisionTreeClassifier(**MODEL_PARAMETERS)
            repeated.fit(train, y_train)
            repeated_probabilities = repeated.predict_proba(test)[:, 1]
            if not np.array_equal(probabilities, repeated_probabilities):
                raise RuntimeError(
                    f"{outcome.upper()} predictions differ between identical fits."
                )
            print(f"{outcome.upper()}: repeated-fit predictions match exactly.")
        y_test = targets[outcome].loc[test.index]
        auc = float(roc_auc_score(y_test, probabilities))
        metadata["outcomes"][outcome.upper()] = {
            "roc_auc": auc,
            "train_roc_auc": train_auc,
            "test_roc_auc": auc,
            "train_labels_sha256": dataframe_sha256(y_train.to_frame()),
            "test_labels_sha256": dataframe_sha256(y_test.to_frame()),
            "test_probabilities_sha256": hashlib.sha256(
                probabilities.tobytes()
            ).hexdigest(),
            "test_probabilities_dtype": str(probabilities.dtype),
            "effective_tree_parameters": model.get_params(),
            "repeated_fit_predictions_match": (
                True if verify_reproducibility else None
            ),
        }
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        print(
            f"{outcome.upper()} ROC-AUC: train={train_auc:.2f}; test={auc:.2f}"
        )
        results.append(
            {"outcome": outcome.upper(),
             "model": "CART (sklearn DecisionTreeClassifier)", "roc_auc": auc,
             "train_roc_auc": train_auc, "test_roc_auc": auc}
        )
        axis.plot(
            fpr, tpr,
            label=f"CART (sklearn DecisionTreeClassifier) – "
                  f"{outcome.upper()} (AUC = {auc:.2f})",
        )
    axis.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
    axis.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Experiment 1 – Static CART (sklearn DecisionTreeClassifier) – FA/GI",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.legend()
    figure.tight_layout()
    figure_path = OUTPUT_DIR / "experiment1_cart_baseline_roc.png"
    results_path = OUTPUT_DIR / "experiment1_cart_baseline_results.csv"
    metadata_path = OUTPUT_DIR / "experiment1_cart_baseline_run_metadata.json"
    for name, path in provenance_paths.items():
        if experiment.sha256_file(path) != input_hashes[name]:
            raise RuntimeError(
                f"{name} changed during this run; results were not saved."
            )
    figure.savefig(figure_path, dpi=300)
    plt.close(figure)
    for outcome, model in models.items():
        tree_path = OUTPUT_DIR / f"experiment1_cart_baseline_{outcome}_tree.png"
        save_tree_figure(model, train.columns.tolist(), outcome, tree_path)
        metadata["outcomes"][outcome.upper()]["tree_figure"] = str(tree_path)
    # Retain full precision in the machine-readable CSV for later comparison.
    pd.DataFrame(results).to_csv(results_path, index=False)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(f"Saved: {figure_path}\nSaved: {results_path}\nSaved: {metadata_path}")
    print("\nFinal CART ROC-AUC summary:")
    for result in results:
        print(
            f"{result['outcome']}: train={result['train_roc_auc']:.2f}; "
            f"test={result['test_roc_auc']:.2f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-reproducibility",
        action="store_true",
        help="Fit each outcome twice and require identical test probabilities.",
    )
    main(verify_reproducibility=parser.parse_args().verify_reproducibility)
