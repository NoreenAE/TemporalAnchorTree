"""Experiment 1: compare FA and GI temporal usage on one shared X."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Dict, Mapping, Optional, Sequence, Tuple

import dill
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from TAT.eval_utils import (  # noqa: E402
    plot_tree_structure, save_roc_curves, save_thesis_tree_figures,
)
from TAT.temporal_explainability import (  # noqa: E402
    derive_monthly_bin_edges,
    load_temporal_usage_artifacts,
    plot_temporal_usage_comparison,
    run_temporal_explainability,
)
from TAT.temporal_tree import TemporalTree  # noqa: E402
from project_paths import portable_metadata, resolve_path  # noqa: E402


EXPERIMENT_NAME = "experiment_1_fa_vs_gi_on_5000V3clean_patients_A=0.4_minS20_minL10_more_offsets2_minGAIN0.0005"
EXPERIMENT_DESCRIPTION = (
    "min GAIN, 0.0005 ,wider offset ranges, coinFlip A=0.4, Same validated longitudinal X (5000) v2 and identical TemporalTree settings; "
    "only the FA versus GI outcome label changes."
)
RANDOM_STATE = 42
TEST_SIZE = 0.30
DECISION_THRESHOLD = 0.50
CANDIDATE_SAMPLING_RATE = 0.4
SPLIT_SAMPLING_SEED = 42
MIN_IMPURITY_DECREASE = 0.0005 # 0 = off
USE_DYNAMIC_ANCHORS = False
VERBOSE = 1

DATA_DIR = REPOSITORY_ROOT / "data" / "temporal_benchmark_v3_clean_fa"
INPUT_PATHS = {
    "x": DATA_DIR / "temporal_benchmark_5000.pkl",
    "fa": DATA_DIR / "temporal_benchmark_fa_labels_5000.pkl",
    "gi": DATA_DIR / "temporal_benchmark_gi_labels_5000.pkl",
}
OUTPUT_DIR = (
    REPOSITORY_ROOT / "experiments" / "results" / "experiment_1_5000V3clean__moreOFFSETS2_patients_A=0.4_minS20_minL10_minGAIN0.0005"
)

ANCHOR_OFFSET_MAP = {
    "Birth Date Anchor": [None, 0.7, 1, 2, 3, 4, 5, 6, 12, 18], #חצי שנה, שנה, שנה וחצי
    "NOW Anchor": [None, -1, -2, -3, -4, -5, -6, -12, -18],
}
ANCHORS_LIST = list(ANCHOR_OFFSET_MAP)
ANCHOR_OFFSET_LIST = [None]
TEMPORAL_EXPLAINABILITY_BIN_EDGES = derive_monthly_bin_edges(
    ANCHOR_OFFSET_MAP
)

AGGREGATION_NAMES = ["mean", "sum", "max" ] #"count"
MODEL_HYPERPARAMETERS = {
    "max_depth": 3,
    "min_samples_split": 20,
    "min_samples_leaf": 10,
    "anchors_list": ANCHORS_LIST,
    "anchor_offset_list": ANCHOR_OFFSET_LIST,
    "is_classification": False,
    "criterion": "mse",
    "learn_missing_direction": False,
    "random_state": RANDOM_STATE,
    "candidate_sampling_rate": CANDIDATE_SAMPLING_RATE,
    "split_sampling_seed": SPLIT_SAMPLING_SEED,
    "min_impurity_decrease": MIN_IMPURITY_DECREASE,
    "verbose": VERBOSE,
    "use_dynamic_anchors": USE_DYNAMIC_ANCHORS,
    "anchor_offset_map": ANCHOR_OFFSET_MAP,
}
OUTCOMES = {
    "fa": {
        "label_path": INPUT_PATHS["fa"],
        "true_region": "Birth Date Anchor: 0 to +6 calendar months",
    },
    "gi": {
        "label_path": INPUT_PATHS["gi"],
        "true_region": "NOW Anchor: -1 calendar month to NOW",
    },
}
TEMPORAL_USAGE_COMPARISON_FILENAME = "temporal_split_usage_fa_vs_gi_thesis.png"


def aggregation_functions() -> Dict[str, object]:
    """Return a fresh explicit aggregation mapping for each model."""
    return {
        "mean": np.mean,
        "sum": np.sum,
        "max": np.max,
        "count": lambda values: values.size,
    }


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one immutable input artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def effective_candidate_windows() -> list:
    """Mirror TemporalTree's current Full Timeline deduplication semantics."""
    windows = []
    seen_full_timeline = False
    for anchor, offsets in ANCHOR_OFFSET_MAP.items():
        for offset in offsets:
            if offset is None:
                if not seen_full_timeline:
                    windows.append((None, None))
                    seen_full_timeline = True
            else:
                windows.append((anchor, offset))
    return windows


def load_longitudinal_x(path: Path) -> pd.DataFrame:
    """Load the validated X using the repository's anchor conversion rule."""
    data = pd.read_pickle(resolve_path(path)).copy()
    required_columns = ["patient_id", "time", "feature_name", "value"]
    if list(data.columns) != required_columns:
        raise ValueError(
            f"Unexpected X schema: {list(data.columns)!r}; "
            f"expected {required_columns!r}."
        )
    data["time"] = pd.to_datetime(data["time"], errors="coerce")
    if data["time"].isna().any():
        raise ValueError("X contains invalid timestamps.")
    anchor_mask = data["feature_name"].str.endswith("Anchor", na=False)
    data.loc[anchor_mask, "value"] = pd.to_datetime(
        data.loc[anchor_mask, "value"],
        errors="coerce",
    )
    if data.loc[anchor_mask, "value"].isna().any():
        raise ValueError("X contains invalid anchor timestamps.")
    data.loc[~anchor_mask, "value"] = pd.to_numeric(
        data.loc[~anchor_mask, "value"],
        errors="coerce",
    )
    return data


def load_labels(path: Path, outcome: str) -> pd.DataFrame:
    """Load and validate one binary patient-level outcome artifact."""
    labels = pd.read_pickle(resolve_path(path)).copy()
    if list(labels.columns) != ["patient_id", "label"]:
        raise ValueError(f"Unexpected {outcome} label schema.")
    if labels["patient_id"].duplicated().any():
        raise ValueError(f"{outcome} labels contain duplicate patients.")
    if not labels["label"].isin([0, 1]).all():
        raise ValueError(f"{outcome} labels must be binary.")
    return labels


def load_inputs() -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Load X once and both outcomes independently."""
    data = load_longitudinal_x(INPUT_PATHS["x"])
    labels = {
        outcome: load_labels(settings["label_path"], outcome)
        for outcome, settings in OUTCOMES.items()
    }
    x_patient_ids = set(data["patient_id"])
    for outcome, outcome_labels in labels.items():
        if set(outcome_labels["patient_id"]) != x_patient_ids:
            raise ValueError(
                f"X and {outcome} labels do not contain identical patients."
            )
    return data, labels


def create_shared_split(
    data: pd.DataFrame,
    labels: Mapping[str, pd.DataFrame],
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Create one joint-stratified patient split reused by both outcomes."""
    patient_ids = np.asarray(sorted(data["patient_id"].unique()))
    split_table = pd.DataFrame({"patient_id": patient_ids})
    for outcome in OUTCOMES:
        label_series = labels[outcome].set_index("patient_id")["label"]
        split_table[f"{outcome}_label"] = split_table["patient_id"].map(
            label_series
        )
    if split_table.isna().any().any():
        raise ValueError("The shared split table contains missing labels.")
    split_table["joint_stratum"] = (
        split_table["fa_label"].astype(str)
        + "_"
        + split_table["gi_label"].astype(str)
    )
    stratum_counts = split_table["joint_stratum"].value_counts()
    if stratum_counts.lt(2).any():
        raise ValueError(
            "Joint FA/GI stratification requires at least two patients in "
            "every observed label combination."
        )

    train_ids, test_ids = train_test_split(
        patient_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=split_table["joint_stratum"],
    )
    train_set = set(train_ids)
    test_set = set(test_ids)
    if train_set & test_set or train_set | test_set != set(patient_ids):
        raise RuntimeError("Shared patient split is not a disjoint partition.")
    split_table["split"] = split_table["patient_id"].map(
        lambda patient_id: "train" if patient_id in train_set else "test"
    )
    return np.asarray(train_ids), np.asarray(test_ids), split_table


def subset_for_patients(
    data: pd.DataFrame,
    labels: pd.DataFrame,
    patient_ids: Sequence[str],
) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Subset X and align y to TemporalTree's first-seen patient order."""
    subset = data.loc[data["patient_id"].isin(patient_ids)].copy()
    patient_order = subset["patient_id"].drop_duplicates().to_numpy()
    label_series = labels.set_index("patient_id")["label"]
    missing = set(patient_order) - set(label_series.index)
    if missing:
        raise ValueError(f"Missing labels for {len(missing)} patients.")
    aligned_labels = label_series.reindex(patient_order).reset_index(drop=True)
    return subset, aligned_labels, patient_order


def build_model() -> TemporalTree:
    """Build either outcome's model from exactly one shared configuration."""
    return TemporalTree(
        **MODEL_HYPERPARAMETERS,
        aggregation_funcs=aggregation_functions(),
    )


def evaluate_scores(
    y_true: pd.Series,
    scores: np.ndarray,
) -> Tuple[dict, np.ndarray]:
    """Calculate imbalanced binary metrics from continuous leaf scores."""
    if len(y_true) != len(scores):
        raise ValueError("Predictions and labels have different lengths.")
    predictions = (scores >= DECISION_THRESHOLD).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, predictions)),
        "confusion_matrix": matrix.astype(int).tolist(),
        "decision_threshold": DECISION_THRESHOLD,
    }
    return metrics, predictions


def root_split_summary(model: TemporalTree) -> dict:
    """Return a JSON-safe description of the fitted root node."""
    root = model.tree
    if root is None:
        raise ValueError("Model has not been fitted.")
    if root.is_leaf():
        return {"is_leaf": True, "prediction": root.value}
    anchor, offset = root.time_window
    return {
        "is_leaf": False,
        "feature": root.feature,
        "aggregation": root.aggregation,
        "threshold": float(root.threshold),
        "anchor": anchor,
        "offset": offset,
        "is_full_timeline": anchor is None and offset is None,
    }


def export_tree_json(model: TemporalTree, path: Path) -> None:
    """Save the fitted tree as a deterministic pre-order JSON table."""
    nodes = []

    def visit(node, depth: int) -> Optional[int]:
        if node is None:
            return None
        node_id = len(nodes)
        if node.is_leaf():
            nodes.append({
                "node_id": node_id,
                "type": "leaf",
                "depth": depth,
                "prediction": node.value,
                "num_patients": node.num_patients,
            })
            return node_id
        record = {
            "node_id": node_id,
            "type": "split",
            "depth": depth,
            "feature": node.feature,
            "aggregation": node.aggregation,
            "anchor": node.time_window[0],
            "offset": node.time_window[1],
            "threshold": float(node.threshold),
            "num_patients": node.num_patients,
            "left_child": None,
            "right_child": None,
        }
        nodes.append(record)
        record["left_child"] = visit(node.left, depth + 1)
        record["right_child"] = visit(node.right, depth + 1)
        return node_id

    visit(model.tree, 0)
    write_json(nodes, path)


def temporal_usage_summary(
    path_table: pd.DataFrame,
) -> Tuple[dict, pd.DataFrame]:
    """Summarize patient-path split usage by anchor and effective window."""
    total = int(len(path_table))
    birth_count = int(path_table["anchor"].eq("Birth Date Anchor").sum())
    now_count = int(path_table["anchor"].eq("NOW Anchor").sum())
    full_count = int(path_table["is_full_timeline"].sum())
    summary = {
        "total_patient_path_split_usages": total,
        "birth_date_anchor_usages": birth_count,
        "now_anchor_usages": now_count,
        "full_timeline_usages": full_count,
        "birth_date_usage_proportion": birth_count / total if total else 0.0,
        "now_usage_proportion": now_count / total if total else 0.0,
        "full_timeline_usage_proportion": full_count / total if total else 0.0,
    }

    distribution = path_table.copy()
    distribution["window"] = distribution.apply(
        lambda row: (
            "Full Timeline"
            if row["is_full_timeline"]
            else f"{row['anchor']} | offset={row['stored_offset']}"
        ),
        axis=1,
    )
    distribution = (
        distribution.groupby("window", dropna=False)
        .size()
        .rename("raw_usage_count")
        .reset_index()
    )
    distribution["usage_proportion"] = (
        distribution["raw_usage_count"] / total if total else 0.0
    )
    return summary, distribution


def candidate_evaluation_summary(model: TemporalTree) -> dict:
    """Collect the implementation's training-wide candidate counters."""
    return {
        "generated_groups": int(model.generated_groups),
        "sampled_groups": int(model.sampled_groups),
        "skipped_groups": int(model.sampling_skipped_groups),
        "generated_threshold_candidates": int(
            model.generated_threshold_candidates
        ),
        "valid_threshold_candidates": int(model.valid_threshold_candidates),
        "invalid_threshold_candidates": int(
            model.invalid_threshold_candidates
        ),
        "evaluated_threshold_candidates": int(
            model.evaluated_threshold_candidates
        ),
        "fallback_evaluated_threshold_candidates": int(
            model.fallback_evaluated_threshold_candidates
        ),
        "total_evaluated_candidates": int(model.evaluated_candidates),
    }


def run_outcome(
    outcome: str,
    data: pd.DataFrame,
    labels: pd.DataFrame,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
    output_dir: Path,
) -> dict:
    """Train, evaluate, explain, and save one outcome with shared settings."""
    outcome_start = perf_counter()
    output_dir.mkdir(parents=True, exist_ok=False)
    train_data, y_train, train_order = subset_for_patients(
        data,
        labels,
        train_ids,
    )
    test_data, y_test, test_order = subset_for_patients(
        data,
        labels,
        test_ids,
    )
    if set(train_order) & set(test_order):
        raise RuntimeError("Patient leakage detected between train and test.")

    model = build_model()
    training_start = perf_counter()
    model.fit(train_data, y_train)
    training_seconds = perf_counter() - training_start

    evaluation_start = perf_counter()
    test_scores = model.predict(test_data)
    performance, test_predictions = evaluate_scores(y_test, test_scores)
    evaluation_seconds = perf_counter() - evaluation_start
    predictions = pd.DataFrame({
        "patient_id": test_order,
        "label": y_test.to_numpy(dtype=int),
        "score": test_scores,
        "prediction": test_predictions,
    })
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    explainability_dir = output_dir / "temporal_explainability"
    explainability_start = perf_counter()
    explainability = run_temporal_explainability(
        model=model,
        X=train_data,
        output_dir=explainability_dir,
        bin_edges_by_anchor=TEMPORAL_EXPLAINABILITY_BIN_EDGES,
    )
    explainability_seconds = perf_counter() - explainability_start
    explainability["usage_table"].to_csv(
        explainability_dir / "aggregated_temporal_usage.csv",
        index=False,
    )
    write_json(
        explainability["summary"],
        explainability_dir / "temporal_usage_summary.json",
    )
    usage_summary, window_distribution = temporal_usage_summary(
        explainability["path_table"]
    )
    window_distribution.to_csv(
        explainability_dir / "temporal_window_distribution.csv",
        index=False,
    )

    model_path = output_dir / "model.dill"
    with model_path.open("wb") as handle:
        dill.dump(model, handle)
    export_tree_json(model, output_dir / "tree.json")
    figure = plot_tree_structure(
        model.tree,
        title=f"Experiment 1 — {outcome.upper()}",
        save_path=output_dir / "tree.png",
    )
    plt.close(figure)

    result = {
        "outcome": outcome,
        "true_region": OUTCOMES[outcome]["true_region"],
        "performance": performance,
        "n_train_patients": int(len(train_order)),
        "n_test_patients": int(len(test_order)),
        "train_positive_prevalence": float(y_train.mean()),
        "test_positive_prevalence": float(y_test.mean()),
        "runtime_seconds": {
            "training": training_seconds,
            "evaluation": evaluation_seconds,
            "explainability": explainability_seconds,
            "total": perf_counter() - outcome_start,
        },
        "candidate_evaluations": candidate_evaluation_summary(model),
        "root_split": root_split_summary(model),
        "temporal_usage": usage_summary,
        "window_distribution": window_distribution.to_dict(orient="records"),
        "artifacts": {
            "model": str(model_path),
            "tree_json": str(output_dir / "tree.json"),
            "tree_png": str(output_dir / "tree.png"),
            "predictions": str(output_dir / "test_predictions.csv"),
            "path_table": str(explainability["path_table_path"]),
            "aggregated_usage": str(
                explainability_dir / "aggregated_temporal_usage.csv"
            ),
            "histogram": str(explainability["histogram_path"]),
        },
    }
    write_json(result, output_dir / "metrics_and_temporal_summary.json")
    save_roc_curves(
        {"ROC": (y_test, test_scores, performance["roc_auc"])},
        output_dir / "roc_curve.png",
    )
    return result


def experiment_configuration(input_hashes: Mapping[str, str]) -> dict:
    """Return the complete JSON-safe shared experiment configuration."""
    windows = effective_candidate_windows()
    full_timeline_count = sum(window == (None, None) for window in windows)
    if full_timeline_count != 1:
        raise RuntimeError("Full Timeline must occur exactly once.")
    return {
        "experiment_name": EXPERIMENT_NAME,
        "description": EXPERIMENT_DESCRIPTION,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "split_strategy": "one joint-stratified FA/GI patient split",
        "decision_threshold": DECISION_THRESHOLD,
        "candidate_sampling_rate": CANDIDATE_SAMPLING_RATE,
        "split_sampling_seed": SPLIT_SAMPLING_SEED,
        "min_impurity_decrease": MIN_IMPURITY_DECREASE,
        "use_dynamic_anchors": USE_DYNAMIC_ANCHORS,
        "anchor_offset_map": ANCHOR_OFFSET_MAP,
        "effective_candidate_windows": windows,
        "effective_full_timeline_window_count": full_timeline_count,
        "temporal_explainability_bin_edges": (
            TEMPORAL_EXPLAINABILITY_BIN_EDGES
        ),
        "aggregations": AGGREGATION_NAMES,
        "effective_aggregations": list(aggregation_functions()),
        "model_hyperparameters": MODEL_HYPERPARAMETERS,
        "input_paths": {name: str(path) for name, path in INPUT_PATHS.items()},
        "input_sha256": dict(input_hashes),
        "output_directory": str(OUTPUT_DIR),
        "outcomes": OUTCOMES,
    }


def comparison_summary(results: Mapping[str, dict]) -> dict:
    """Create the requested side-by-side FA-versus-GI summary."""
    fields = {}
    for outcome, result in results.items():
        fields[outcome] = {
            **result["performance"],
            "runtime_seconds": result["runtime_seconds"],
            "candidate_evaluations": result["candidate_evaluations"],
            "root_split": result["root_split"],
            "temporal_usage": result["temporal_usage"],
            "window_distribution": result["window_distribution"],
        }
    return {"experiment_name": EXPERIMENT_NAME, "outcomes": fields}


def create_temporal_usage_comparison(output_dir: Path) -> Path:
    """Plot FA versus GI from their already-saved explainability counts."""
    usage_by_outcome = {
        outcome: load_temporal_usage_artifacts(
            output_dir / outcome / "temporal_explainability"
        )
        for outcome in OUTCOMES
    }
    return plot_temporal_usage_comparison(
        usage_by_outcome=usage_by_outcome,
        output_path=output_dir / TEMPORAL_USAGE_COMPARISON_FILENAME,
    )


def create_thesis_tree_figures(output_dir: Path) -> dict:
    """Render the two saved trees and their combined panel without model use."""
    return save_thesis_tree_figures(
        {outcome: output_dir / outcome / "tree.json" for outcome in OUTCOMES},
        output_dir,
    )


def create_roc_comparison(
    output_dir: Path,
    results: Optional[Mapping[str, dict]] = None,
    predictions_filename: str = "test_predictions.csv",
) -> Path:
    """Save FA/GI ROC curves using saved predictions, without loading models.

    Reuse evaluation AUCs from results, or from saved outcome summaries when
    plotting a completed Experiment 1. Missing artifacts raise an error.
    """
    curves = {}
    for outcome in OUTCOMES:
        outcome_dir = output_dir / outcome
        predictions = pd.read_csv(outcome_dir / predictions_filename)
        if results is None:
            result = json.loads(
                (outcome_dir / "metrics_and_temporal_summary.json").read_text(
                    encoding="utf-8"
                )
            )
        else:
            result = results[outcome]
        curves[outcome.upper()] = (
            predictions["label"],
            predictions["score"],
            result["performance"]["roc_auc"],
        )
    return save_roc_curves(
        curves,
        output_dir / "roc_curves_fa_vs_gi.png",
        title="ROC Curves for FA and GI",
    )


def write_json(value, path: Path) -> None:
    """Write indented JSON, converting NumPy and Path values safely."""
    path.write_text(
        json.dumps(portable_metadata(value), indent=2,
                   default=_json_default, sort_keys=True),
        encoding="utf-8",
    )


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON.")


def validate_only() -> dict:
    """Validate inputs and shared split without fitting or writing outputs."""
    for name, path in INPUT_PATHS.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name} input artifact: {path}")
    data, labels = load_inputs()
    train_ids, test_ids, split_table = create_shared_split(data, labels)
    windows = effective_candidate_windows()
    non_anchor_features = data.loc[
        ~data["feature_name"].str.endswith("Anchor", na=False),
        "feature_name",
    ].nunique()
    result = {
        "status": "VALIDATED_WITHOUT_TRAINING",
        "n_patients": int(data["patient_id"].nunique()),
        "n_longitudinal_rows": int(len(data)),
        "n_model_features": int(non_anchor_features),
        "n_effective_windows": len(windows),
        "n_full_timeline_windows": sum(
            window == (None, None) for window in windows
        ),
        "feature_window_aggregation_groups_per_node": int(
            non_anchor_features * len(windows) * len(AGGREGATION_NAMES)
        ),
        "n_train_patients": int(len(train_ids)),
        "n_test_patients": int(len(test_ids)),
        "joint_stratum_counts": {
            str(key): int(value)
            for key, value in split_table["joint_stratum"]
            .value_counts()
            .sort_index()
            .items()
        },
        "prevalence": {
            outcome: {
                split: float(
                    split_table.loc[
                        split_table["split"].eq(split),
                        f"{outcome}_label",
                    ].mean()
                )
                for split in ("train", "test")
            }
            for outcome in OUTCOMES
        },
    }
    return result


def run_experiment() -> dict:
    """Execute both models only after explicit user approval."""
    for name, path in INPUT_PATHS.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name} input artifact: {path}")
    input_hashes_before = {
        name: sha256_file(path) for name, path in INPUT_PATHS.items()
    }
    data, labels = load_inputs()
    train_ids, test_ids, split_table = create_shared_split(data, labels)

    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=False)
    write_json(
        experiment_configuration(input_hashes_before),
        OUTPUT_DIR / "config.json",
    )
    split_table.to_csv(OUTPUT_DIR / "shared_patient_split.csv", index=False)

    results = {}
    for outcome in OUTCOMES:
        results[outcome] = run_outcome(
            outcome=outcome,
            data=data,
            labels=labels[outcome],
            train_ids=train_ids,
            test_ids=test_ids,
            output_dir=OUTPUT_DIR / outcome,
        )
    comparison_figure_path = create_temporal_usage_comparison(OUTPUT_DIR)
    comparison = comparison_summary(results)
    comparison["artifacts"] = {
        "temporal_usage_comparison": str(comparison_figure_path),
    }
    write_json(comparison, OUTPUT_DIR / "comparison_summary.json")
    pd.DataFrame({
        outcome: {
            **result["performance"],
            "training_runtime_seconds": result["runtime_seconds"]["training"],
            "total_runtime_seconds": result["runtime_seconds"]["total"],
            "evaluated_candidates": result["candidate_evaluations"][
                "total_evaluated_candidates"
            ],
            "root_anchor": result["root_split"].get("anchor"),
            "root_offset": result["root_split"].get("offset"),
            "birth_date_usage_proportion": result["temporal_usage"][
                "birth_date_usage_proportion"
            ],
            "now_usage_proportion": result["temporal_usage"][
                "now_usage_proportion"
            ],
            "full_timeline_usage_proportion": result["temporal_usage"][
                "full_timeline_usage_proportion"
            ],
        }
        for outcome, result in results.items()
    }).T.to_csv(OUTPUT_DIR / "comparison_summary.csv")

    input_hashes_after = {
        name: sha256_file(path) for name, path in INPUT_PATHS.items()
    }
    if input_hashes_after != input_hashes_before:
        raise RuntimeError(
            "An immutable benchmark input changed during the run."
        )
    create_roc_comparison(OUTPUT_DIR, results)
    create_thesis_tree_figures(OUTPUT_DIR)
    return comparison


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate configuration, inputs, and shared split without "
            "training."
        ),
    )
    parser.add_argument("--output", type=Path,
                        help="New result directory; saved thesis results are never overwritten.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    global OUTPUT_DIR
    args = parse_args(argv)
    if args.output is not None:
        OUTPUT_DIR = resolve_path(args.output)
    if args.validate_only:
        print(json.dumps(validate_only(), indent=2, sort_keys=True))
        return
    run_experiment()


if __name__ == "__main__":
    main()
