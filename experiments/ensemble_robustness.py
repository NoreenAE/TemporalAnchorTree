"""Ensemble Robustness: robustness to stochastic candidate-split sampling on V2.

The experiment fits the same TemporalTree configuration for FA and GI using
ten candidate-sampling seeds.  The sampling seed is the only experimental
variable. Specify model hyperparameters directly and manually in
SELECTED_MODEL_CONFIGURATION before validation or training.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
import traceback
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple
from uuid import uuid4

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from settree.eval_utils import save_roc_curves  # noqa: E402
from settree.temporal_explainability import (  # noqa: E402
    trace_temporal_decision_paths,
)
from settree.temporal_tree import TemporalTree  # noqa: E402


EXPERIMENT_NAME = "experiment_2_robustness_v3"
DATA_DIR = REPOSITORY_ROOT / "data" / "temporal_benchmark_v3_clean_fa"
INPUT_PATHS = {
    "x": DATA_DIR / "temporal_benchmark_5000.pkl",
    "fa": DATA_DIR / "temporal_benchmark_fa_labels_5000.pkl",
    "gi": DATA_DIR / "temporal_benchmark_gi_labels_5000.pkl",
}
OUTPUT_DIR = (
    REPOSITORY_ROOT
    / "experiments"
    / "results"
    / "experiment_2_robustness_v3"
)

TRAIN_TEST_SPLIT = {
    "test_size": 0.30,
    "seed": 42,
    "stratification": "joint_fa_gi",
}
SPLIT_SAMPLING_SEEDS = (42, 43, 44, 45, 46, 47, 48, 49, 50, 51)
ANCHOR_OFFSET_MAP = {
    "Birth Date Anchor": [None, 0.7, 1, 2, 3, 4, 5, 6, 12, 18],
    "NOW Anchor": [None, -1, -2, -3, -4, -5, -6, -12, -18],
}
AGGREGATION_NAMES = ("mean", "sum", "max", "count")

# Specify the model hyperparameters directly; no search result is required.
SELECTED_MODEL_CONFIGURATION = {
    "candidate_sampling_rate": 0.4,
    "max_depth": 3,
    "min_samples_split": 20,
    "min_samples_leaf": 10,
    "min_impurity_decrease": 0.0005,
}

FIXED_MODEL_CONFIGURATION = {
    "anchors_list": ["Birth Date Anchor", "NOW Anchor"],
    "anchor_offset_list": [None],
    "anchor_offset_map": ANCHOR_OFFSET_MAP,
    "is_classification": False,
    "criterion": "mse",
    "learn_missing_direction": False,
    "random_state": 42,
    "use_dynamic_anchors": False,
    "verbose": 1,
}

OUTCOMES = ("fa", "gi")
EXPECTED_OFFSET_MAP = {
    "Birth Date Anchor": [None, 0.7, 1, 2, 3, 4, 5, 6, 12, 18],
    "NOW Anchor": [None, -1, -2, -3, -4, -5, -6, -12, -18],
}
REQUIRED_SELECTED_FIELDS = {
    "candidate_sampling_rate",
    "max_depth",
    "min_samples_split",
    "min_samples_leaf",
    "min_impurity_decrease",
}
PER_SEED_RESULTS_FILENAME = "per_seed_results.csv"
MANIFEST_FILENAME = "run_manifest.json"


def utc_now() -> str:
    """Return a stable UTC timestamp for the central run manifest."""
    return datetime.now(timezone.utc).isoformat()


def aggregation_functions() -> Dict[str, object]:
    """Build the callable mapping from the names recorded in provenance."""
    available = {
        "mean": np.mean,
        "sum": np.sum,
        "max": np.max,
        "count": lambda values: values.size,
    }
    return {name: available[name] for name in AGGREGATION_NAMES}


def sha256_file(path: Path) -> str:
    """Calculate the SHA256 digest of a file without loading it at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    if pd.isna(value):
        return None
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON.")


def _json_bytes(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")


def configuration_fingerprint(value) -> str:
    """Return a deterministic fingerprint for scientific configuration."""
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def atomic_write_json(value, path: Path) -> None:
    """Atomically replace one central JSON artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_csv(table: pd.DataFrame, path: Path) -> None:
    """Atomically replace one central CSV artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    table.to_csv(temporary, index=False)
    os.replace(temporary, path)


def validate_selected_configuration(
    selected: Mapping = SELECTED_MODEL_CONFIGURATION,
) -> dict:
    """Reject incomplete or scientifically inconsistent model settings."""
    missing = REQUIRED_SELECTED_FIELDS.difference(selected)
    extra = set(selected).difference(REQUIRED_SELECTED_FIELDS)
    if missing or extra:
        raise ValueError(
            "SELECTED_MODEL_CONFIGURATION fields differ from the required "
            f"schema; missing={sorted(missing)}, extra={sorted(extra)}."
        )
    placeholder_fields = [
        name
        for name, value in selected.items()
        if value is None
        or (
            isinstance(value, str)
            and ("REPLACE" in value.upper() or "PLACEHOLDER" in value.upper())
        )
    ]
    if placeholder_fields:
        raise ValueError(
            "Fill SELECTED_MODEL_CONFIGURATION before validation or "
            f"training; placeholders remain in {placeholder_fields}."
        )
    sampling_rate = selected["candidate_sampling_rate"]
    if (
        isinstance(sampling_rate, bool)
        or not isinstance(sampling_rate, (int, float))
        or not np.isfinite(sampling_rate)
        or not 0 < sampling_rate <= 1
    ):
        raise ValueError("candidate_sampling_rate must be in (0, 1].")
    for name in ("max_depth", "min_samples_split", "min_samples_leaf"):
        value = selected[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if selected["min_samples_split"] < 2 * selected["min_samples_leaf"]:
        raise ValueError(
            "min_samples_split must be at least twice min_samples_leaf."
        )
    gain = selected["min_impurity_decrease"]
    if (
        isinstance(gain, bool)
        or not isinstance(gain, (int, float))
        or not np.isfinite(gain)
        or gain < 0
    ):
        raise ValueError("min_impurity_decrease must be non-negative.")
    return dict(selected)


def validate_experiment_design(
    selected: Mapping = SELECTED_MODEL_CONFIGURATION,
    seeds: Sequence[int] = SPLIT_SAMPLING_SEEDS,
    anchor_offset_map: Mapping = ANCHOR_OFFSET_MAP,
    fixed: Mapping = FIXED_MODEL_CONFIGURATION,
) -> dict:
    """Validate all preregistered invariants before any output is created."""
    validated = validate_selected_configuration(selected)
    if tuple(seeds) != tuple(range(42, 52)) or len(set(seeds)) != 10:
        raise ValueError("Sampling seeds must be exactly 42 through 51.")
    if dict(anchor_offset_map) != EXPECTED_OFFSET_MAP:
        raise ValueError("The broad temporal search space changed.")
    if fixed.get("use_dynamic_anchors") is not False:
        raise ValueError("use_dynamic_anchors must remain False.")
    if tuple(fixed.get("anchors_list", ())) != tuple(ANCHOR_OFFSET_MAP):
        raise ValueError("anchors_list must match the fixed anchor map.")
    if tuple(AGGREGATION_NAMES) != ("mean", "sum", "max", "count"):
        raise ValueError("The aggregation set changed.")
    return validated


def model_parameters(
    split_sampling_seed: int,
    selected: Mapping = SELECTED_MODEL_CONFIGURATION,
) -> dict:
    """Build kwargs whose only between-run change is the sampling seed."""
    selected = validate_experiment_design(selected=selected)
    if split_sampling_seed not in SPLIT_SAMPLING_SEEDS:
        raise ValueError("Unexpected split_sampling_seed.")
    return {
        "max_depth": selected["max_depth"],
        "min_samples_split": selected["min_samples_split"],
        "min_samples_leaf": selected["min_samples_leaf"],
        **FIXED_MODEL_CONFIGURATION,
        "candidate_sampling_rate": selected["candidate_sampling_rate"],
        "split_sampling_seed": int(split_sampling_seed),
        "min_impurity_decrease": selected["min_impurity_decrease"],
    }


def build_model(split_sampling_seed: int) -> TemporalTree:
    """Construct one model from the fixed configuration."""
    return TemporalTree(
        **model_parameters(split_sampling_seed),
        aggregation_funcs=aggregation_functions(),
    )


def effective_windows(
    anchor_offset_map: Mapping = ANCHOR_OFFSET_MAP,
) -> list[dict]:
    """Return every numeric window plus one deduplicated Full Timeline."""
    windows = []
    full_added = False
    for anchor, offsets in anchor_offset_map.items():
        for offset in offsets:
            if offset is None:
                if not full_added:
                    windows.append({
                        "anchor": None,
                        "offset": None,
                        "is_full_timeline": True,
                    })
                    full_added = True
                continue
            windows.append({
                "anchor": anchor,
                "offset": offset,
                "is_full_timeline": False,
            })
    return windows


def load_longitudinal_x(path: Path) -> pd.DataFrame:
    """Load V2 X and apply the repository's established type conversion."""
    data = pd.read_pickle(path).copy()
    required = ["patient_id", "time", "feature_name", "value"]
    if list(data.columns) != required:
        raise ValueError(f"Unexpected V2 X schema: {list(data.columns)!r}.")
    data["time"] = pd.to_datetime(data["time"], errors="coerce")
    if data["time"].isna().any():
        raise ValueError("V2 X contains invalid timestamps.")
    anchor_mask = data["feature_name"].str.endswith("Anchor", na=False)
    data.loc[anchor_mask, "value"] = pd.to_datetime(
        data.loc[anchor_mask, "value"], errors="coerce"
    )
    if data.loc[anchor_mask, "value"].isna().any():
        raise ValueError("V2 X contains invalid anchor timestamps.")
    data.loc[~anchor_mask, "value"] = pd.to_numeric(
        data.loc[~anchor_mask, "value"], errors="coerce"
    )
    return data


def load_labels(path: Path, outcome: str) -> pd.DataFrame:
    """Load and validate one patient-level binary label table."""
    labels = pd.read_pickle(path).copy()
    if list(labels.columns) != ["patient_id", "label"]:
        raise ValueError(f"Unexpected {outcome.upper()} label schema.")
    if labels["patient_id"].duplicated().any():
        raise ValueError(
            f"{outcome.upper()} labels contain duplicate patients."
        )
    if not labels["label"].isin([0, 1]).all():
        raise ValueError(f"{outcome.upper()} labels must be binary.")
    return labels


def load_inputs() -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Load the immutable V2 data and require identical patient sets."""
    data = load_longitudinal_x(INPUT_PATHS["x"])
    labels = {
        outcome: load_labels(INPUT_PATHS[outcome], outcome)
        for outcome in OUTCOMES
    }
    patients = set(data["patient_id"])
    for outcome, table in labels.items():
        if set(table["patient_id"]) != patients:
            raise ValueError(f"V2 X and {outcome.upper()} labels differ.")
    return data, labels


def create_shared_split(
    data: pd.DataFrame,
    labels: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Create the one joint-stratified patient split used by all fits."""
    patient_ids = np.asarray(sorted(data["patient_id"].unique()))
    split = pd.DataFrame({"patient_id": patient_ids})
    for outcome in OUTCOMES:
        by_patient = labels[outcome].set_index("patient_id")["label"]
        split[f"{outcome}_label"] = split["patient_id"].map(by_patient)
    if split.isna().any().any():
        raise ValueError("The shared split contains missing labels.")
    split["joint_stratum"] = (
        split["fa_label"].astype(str) + "_" + split["gi_label"].astype(str)
    )
    if split["joint_stratum"].value_counts().lt(2).any():
        raise ValueError("Every observed joint stratum needs two patients.")
    train_ids, test_ids = train_test_split(
        patient_ids,
        test_size=TRAIN_TEST_SPLIT["test_size"],
        random_state=TRAIN_TEST_SPLIT["seed"],
        stratify=split["joint_stratum"],
    )
    train_set = set(train_ids)
    test_set = set(test_ids)
    if train_set & test_set or train_set | test_set != set(patient_ids):
        raise RuntimeError("The patient split is not a disjoint partition.")
    split["split"] = split["patient_id"].map(
        lambda patient_id: "train" if patient_id in train_set else "test"
    )
    return split


def split_patient_ids(split: pd.DataFrame, name: str) -> np.ndarray:
    """Return patient IDs assigned to one saved partition."""
    return split.loc[split["split"].eq(name), "patient_id"].to_numpy()


def subset_for_patients(
    data: pd.DataFrame,
    labels: pd.DataFrame,
    patient_ids: Iterable,
) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Subset X and align labels to TemporalTree's first-seen order."""
    requested = set(patient_ids)
    subset = data.loc[data["patient_id"].isin(requested)].copy()
    order = subset["patient_id"].drop_duplicates().to_numpy()
    if set(order) != requested:
        raise ValueError("A requested patient is absent from V2 X.")
    by_patient = labels.set_index("patient_id")["label"]
    aligned = by_patient.reindex(order)
    if aligned.isna().any():
        raise ValueError("A requested patient is missing an outcome label.")
    return subset, aligned.reset_index(drop=True), order


def temporal_window_usage(
    path_table: pd.DataFrame,
    test_patient_ids: Sequence,
    split_sampling_seed: Optional[int] = None,
    outcome: Optional[str] = None,
) -> pd.DataFrame:
    """Return zero-filled path usage for every effective temporal window."""
    total = int(len(path_table))
    patient_total = int(len(test_patient_ids))
    rows = []
    for window in effective_windows():
        if window["is_full_timeline"]:
            mask = path_table["is_full_timeline"].astype(bool)
        else:
            mask = (
                ~path_table["is_full_timeline"].astype(bool)
                & path_table["anchor"].eq(window["anchor"])
                & pd.to_numeric(
                    path_table["stored_offset"], errors="coerce"
                ).eq(float(window["offset"]))
            )
        count = int(mask.sum())
        patient_count = int(path_table.loc[mask, "patient_id"].nunique())
        rows.append({
            "split_sampling_seed": split_sampling_seed,
            "outcome": outcome,
            **window,
            "path_usage_count": count,
            "path_usage_percentage": count / total * 100.0 if total else 0.0,
            "patient_count": patient_count,
            "patient_coverage_percentage": (
                patient_count / patient_total * 100.0 if patient_total else 0.0
            ),
        })
    return pd.DataFrame(rows)


def anchor_usage(path_table: pd.DataFrame) -> dict:
    """Calculate mutually exclusive Birth, NOW, and Full usage percentages."""
    total = int(len(path_table))
    birth = int(path_table["anchor"].eq("Birth Date Anchor").sum())
    now = int(path_table["anchor"].eq("NOW Anchor").sum())
    full = int(path_table["is_full_timeline"].astype(bool).sum())
    if birth + now + full != total:
        raise ValueError(
            "A path usage is outside the fixed temporal categories."
        )
    denominator = total if total else 1
    return {
        "total_path_split_usages": total,
        "birth_usage_percentage": birth / denominator * 100.0,
        "now_usage_percentage": now / denominator * 100.0,
        "full_timeline_usage_percentage": full / denominator * 100.0,
    }


def expected_anchor_dominance(
    usage: Mapping[str, float],
    outcome: str,
) -> bool:
    """Apply the preregistered strict anchor-dominance definition."""
    if usage["total_path_split_usages"] == 0:
        return False
    birth = usage["birth_usage_percentage"]
    now = usage["now_usage_percentage"]
    full = usage["full_timeline_usage_percentage"]
    if outcome == "fa":
        return bool(birth > now and birth > full)
    if outcome == "gi":
        return bool(now > birth and now > full)
    raise ValueError(f"Unknown outcome: {outcome!r}.")


def dominant_anchor(usage: Mapping[str, float]) -> Optional[str]:
    """Return the unique dominant category, or None for ties/no splits."""
    if usage["total_path_split_usages"] == 0:
        return None
    values = {
        "Birth Date Anchor": usage["birth_usage_percentage"],
        "NOW Anchor": usage["now_usage_percentage"],
        "Full Timeline": usage["full_timeline_usage_percentage"],
    }
    maximum = max(values.values())
    winners = [name for name, value in values.items() if value == maximum]
    return winners[0] if len(winners) == 1 else None


def iter_split_nodes(root):
    """Yield internal nodes in deterministic pre-order."""
    if root is None or root.is_leaf():
        return
    yield root
    yield from iter_split_nodes(root.left)
    yield from iter_split_nodes(root.right)


def root_split_information(model: TemporalTree) -> dict:
    """Return the approved root fields in a JSON-safe form."""
    root = model.tree
    if root is None:
        raise ValueError("The model has not been fitted.")
    if root.is_leaf():
        return {
            "root_is_leaf": True,
            "root_feature": None,
            "root_aggregation": None,
            "root_anchor": None,
            "root_offset": None,
            "root_is_full_timeline": False,
            "root_threshold": None,
        }
    anchor, offset = root.time_window
    return {
        "root_is_leaf": False,
        "root_feature": root.feature,
        "root_aggregation": root.aggregation,
        "root_anchor": anchor,
        "root_offset": offset,
        "root_is_full_timeline": anchor is None and offset is None,
        "root_threshold": float(root.threshold),
    }


def now_minus_one_behavior(
    model: TemporalTree,
    path_table: pd.DataFrame,
    test_patient_count: int,
) -> dict:
    """Describe structural and held-out path use of the GI NOW -1 window."""
    present = any(
        node.time_window[0] == "NOW Anchor"
        and node.time_window[1] is not None
        and float(node.time_window[1]) == -1.0
        for node in iter_split_nodes(model.tree)
    )
    offsets = pd.to_numeric(path_table["stored_offset"], errors="coerce")
    mask = path_table["anchor"].eq("NOW Anchor") & offsets.eq(-1.0)
    count = int(mask.sum())
    patient_count = int(path_table.loc[mask, "patient_id"].nunique())
    total = int(len(path_table))
    return {
        "now_minus_1_present_in_tree": bool(present),
        "now_minus_1_used_on_test_paths": bool(count > 0),
        "now_minus_1_path_usage_count": count,
        "now_minus_1_path_usage_percentage": (
            count / total * 100.0 if total else 0.0
        ),
        "now_minus_1_patient_coverage_percentage": (
            patient_count / test_patient_count * 100.0
            if test_patient_count
            else 0.0
        ),
    }


def candidate_sampling_counters(model: TemporalTree) -> dict:
    """Collect every candidate-sampling counter exposed by TemporalTree."""
    generated = int(model.generated_groups)
    sampled = int(model.sampled_groups)
    evaluated = int(model.evaluated_threshold_candidates)
    fallback_evaluated = int(model.fallback_evaluated_threshold_candidates)
    return {
        "generated_groups": generated,
        "sampled_groups": sampled,
        "skipped_groups": int(model.sampling_skipped_groups),
        "observed_group_sampling_rate": (
            sampled / generated if generated else 0.0
        ),
        "generated_threshold_candidates": int(
            model.generated_threshold_candidates
        ),
        "invalid_threshold_candidates": int(
            model.invalid_threshold_candidates
        ),
        "valid_threshold_candidates": int(model.valid_threshold_candidates),
        "evaluated_threshold_candidates": evaluated,
        "fallback_evaluated_groups": int(model.fallback_evaluated_groups),
        "fallback_generated_threshold_candidates": int(
            model.fallback_generated_threshold_candidates
        ),
        "fallback_valid_threshold_candidates": int(
            model.fallback_valid_threshold_candidates
        ),
        "fallback_evaluated_threshold_candidates": fallback_evaluated,
        "nodes_requiring_sampling_fallback": int(
            model.nodes_requiring_sampling_fallback
        ),
        "total_evaluated_candidates": evaluated + fallback_evaluated,
    }


def export_tree_json(model: TemporalTree, path: Path) -> None:
    """Export the complete fitted tree as a deterministic pre-order table."""
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
        anchor, offset = node.time_window
        record = {
            "node_id": node_id,
            "type": "split",
            "depth": depth,
            "feature": node.feature,
            "aggregation": node.aggregation,
            "anchor": anchor,
            "offset": offset,
            "is_full_timeline": anchor is None and offset is None,
            "threshold": float(node.threshold),
            "missing_direction": node.missing_direction,
            "num_patients": node.num_patients,
            "impurity": getattr(node, "impurity", None),
            "left_child": None,
            "right_child": None,
        }
        nodes.append(record)
        record["left_child"] = visit(node.left, depth + 1)
        record["right_child"] = visit(node.right, depth + 1)
        return node_id

    visit(model.tree, 0)
    atomic_write_json(nodes, path)


def run_outcome(
    outcome: str,
    split_sampling_seed: int,
    data: pd.DataFrame,
    labels: pd.DataFrame,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
    attempt_dir: Path,
) -> Tuple[dict, pd.DataFrame]:
    """Fit, evaluate, and export one outcome for one sampling seed."""
    started = perf_counter()
    train_data, y_train, train_order = subset_for_patients(
        data, labels, train_ids
    )
    test_data, y_test, test_order = subset_for_patients(data, labels, test_ids)
    if set(train_order) & set(test_order):
        raise RuntimeError("Patient leakage detected.")

    model = build_model(split_sampling_seed)
    training_started = perf_counter()
    model.fit(train_data, y_train)
    training_seconds = perf_counter() - training_started

    evaluation_started = perf_counter()
    scores = np.asarray(model.predict(test_data), dtype=float)
    roc_auc = float(roc_auc_score(y_test, scores))
    evaluation_seconds = perf_counter() - evaluation_started

    temporal_started = perf_counter()
    paths = trace_temporal_decision_paths(model, test_data)
    usage = anchor_usage(paths)
    windows = temporal_window_usage(
        paths,
        test_order,
        split_sampling_seed=split_sampling_seed,
        outcome=outcome,
    )
    temporal_seconds = perf_counter() - temporal_started

    prefix = outcome.upper()
    export_tree_json(model, attempt_dir / f"{prefix}_tree.json")
    windows.to_csv(
        attempt_dir / f"{prefix}_temporal_window_usage.csv", index=False
    )

    row = {
        "split_sampling_seed": int(split_sampling_seed),
        "outcome": outcome.upper(),
        "roc_auc": roc_auc,
        **usage,
        "dominant_anchor": dominant_anchor(usage),
        "expected_anchor_dominant": expected_anchor_dominance(usage, outcome),
        **root_split_information(model),
        "training_runtime_seconds": training_seconds,
        "evaluation_runtime_seconds": evaluation_seconds,
        "temporal_analysis_runtime_seconds": temporal_seconds,
        "total_runtime_seconds": perf_counter() - started,
        **candidate_sampling_counters(model),
        "now_minus_1_present_in_tree": None,
        "now_minus_1_used_on_test_paths": None,
        "now_minus_1_path_usage_count": None,
        "now_minus_1_path_usage_percentage": None,
        "now_minus_1_patient_coverage_percentage": None,
    }
    if outcome == "gi":
        row.update(now_minus_one_behavior(model, paths, len(test_order)))
    save_roc_curves(
        {"ROC": (y_test, scores, roc_auc)},
        attempt_dir / f"{prefix}_roc_curve.png",
    )
    return row, windows


def sample_sd(values: Sequence[float]) -> float:
    """Return the sample SD used in all cross-seed summaries."""
    return float(np.std(np.asarray(values, dtype=float), ddof=1))


def root_signature(row: Mapping) -> str:
    """Create the threshold-free categorical root signature."""
    signature = {
        key: None if pd.isna(row[key]) else row[key]
        for key in (
            "root_is_leaf",
            "root_feature",
            "root_aggregation",
            "root_anchor",
            "root_offset",
            "root_is_full_timeline",
        )
    }
    return json.dumps(signature, sort_keys=True, default=_json_default)


def _seed_rows_fingerprint(rows: pd.DataFrame) -> str:
    """Fingerprint the two central result rows after CSV normalization."""
    ordered = rows.sort_values("outcome").reset_index(drop=True)
    serialized = ordered.to_json(
        orient="records",
        double_precision=15,
        date_format="iso",
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def summarize_per_seed(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Create the tidy thesis summary from exactly ten seeds per outcome."""
    records = []

    def add(outcome, section, metric, statistic, value, category=None):
        records.append({
            "outcome": outcome,
            "section": section,
            "metric": metric,
            "category": category,
            "statistic": statistic,
            "value": value,
            "n_seeds": 10,
        })

    for outcome in ("FA", "GI"):
        rows = per_seed.loc[per_seed["outcome"].eq(outcome)].copy()
        if set(rows["split_sampling_seed"]) != set(SPLIT_SAMPLING_SEEDS):
            raise ValueError(f"{outcome} does not contain all ten seeds.")
        auc = rows["roc_auc"].astype(float)
        for statistic, value in (
            ("mean", auc.mean()),
            ("sd", sample_sd(auc)),
            ("min", auc.min()),
            ("max", auc.max()),
        ):
            add(outcome, "predictive", "roc_auc", statistic, float(value))
        for column, metric in (
            ("birth_usage_percentage", "birth_usage_percentage"),
            ("now_usage_percentage", "now_usage_percentage"),
            (
                "full_timeline_usage_percentage",
                "full_timeline_usage_percentage",
            ),
        ):
            values = rows[column].astype(float)
            add(outcome, "anchor_usage", metric, "mean", float(values.mean()))
            add(outcome, "anchor_usage", metric, "sd", sample_sd(values))
        dominant_count = int(
            rows["expected_anchor_dominant"].astype(bool).sum()
        )
        add(
            outcome,
            "dominance",
            "expected_anchor_dominant",
            "count",
            dominant_count,
        )
        add(
            outcome,
            "dominance",
            "expected_anchor_dominant",
            "percentage",
            dominant_count / 10 * 100.0,
        )

        signatures = rows.apply(root_signature, axis=1)
        counts = signatures.value_counts(sort=False)
        for signature, count in sorted(counts.items()):
            add(
                outcome,
                "root",
                "root_signature",
                "count",
                int(count),
                signature,
            )
            add(
                outcome,
                "root",
                "root_signature",
                "percentage",
                int(count) / 10 * 100.0,
                signature,
            )
        modal_signature = sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )[0]
        add(outcome, "root", "distinct_root_signatures", "count", len(counts))
        add(
            outcome,
            "root",
            "modal_root_signature",
            "count",
            int(modal_signature[1]),
            modal_signature[0],
        )
        add(
            outcome,
            "root",
            "modal_root_signature",
            "percentage",
            int(modal_signature[1]) / 10 * 100.0,
            modal_signature[0],
        )
        add(
            outcome,
            "root",
            "no_split_roots",
            "count",
            int(rows["root_is_leaf"].astype(bool).sum()),
        )

        if outcome == "GI":
            for column, metric in (
                ("now_minus_1_present_in_tree", "now_minus_1_structural"),
                ("now_minus_1_used_on_test_paths", "now_minus_1_test_path"),
            ):
                count = int(rows[column].astype(bool).sum())
                add(outcome, "now_minus_1", metric, "count", count)
                add(
                    outcome,
                    "now_minus_1",
                    metric,
                    "percentage",
                    count / 10 * 100.0,
                )
            for column, metric in (
                (
                    "now_minus_1_path_usage_percentage",
                    "now_minus_1_path_usage_percentage",
                ),
                (
                    "now_minus_1_patient_coverage_percentage",
                    "now_minus_1_patient_coverage_percentage",
                ),
            ):
                values = rows[column].astype(float)
                add(
                    outcome,
                    "now_minus_1",
                    metric,
                    "mean",
                    float(values.mean()),
                )
                add(outcome, "now_minus_1", metric, "sd", sample_sd(values))
    return pd.DataFrame(records)


def summarize_temporal_windows(window_rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate zero-filled per-window usage over all ten seeds."""
    records = []
    grouping = ["outcome", "anchor", "offset", "is_full_timeline"]
    normalized = window_rows.copy()
    normalized["anchor"] = normalized["anchor"].fillna("Full Timeline")
    normalized["offset"] = normalized["offset"].astype(object).where(
        normalized["offset"].notna(), "Full Timeline"
    )
    for keys, rows in normalized.groupby(grouping, sort=False, dropna=False):
        if set(rows["split_sampling_seed"]) != set(SPLIT_SAMPLING_SEEDS):
            raise ValueError(
                f"Window {keys!r} does not contain all ten seeds."
            )
        usage = rows["path_usage_percentage"].astype(float)
        coverage = rows["patient_coverage_percentage"].astype(float)
        used = int(usage.gt(0).sum())
        records.append({
            "outcome": keys[0],
            "anchor": keys[1],
            "offset": keys[2],
            "is_full_timeline": bool(keys[3]),
            "mean_usage_percentage": float(usage.mean()),
            "sd_usage_percentage": sample_sd(usage),
            "seeds_where_used": used,
            "percentage_of_seeds_where_used": used / 10 * 100.0,
            "mean_patient_coverage_percentage": float(coverage.mean()),
            "sd_patient_coverage_percentage": sample_sd(coverage),
        })
    return pd.DataFrame(records)


def _ordered_window_labels() -> list[str]:
    labels = [
        f"Birth +{offset:g}"
        for offset in ANCHOR_OFFSET_MAP["Birth Date Anchor"]
        if offset is not None
    ]
    labels.extend(
        f"NOW {offset:g}"
        for offset in sorted(
            value
            for value in ANCHOR_OFFSET_MAP["NOW Anchor"]
            if value is not None
        )
    )
    labels.append("Full Timeline")
    return labels


def _window_label(row: Mapping) -> str:
    if bool(row["is_full_timeline"]):
        return "Full Timeline"
    if row["anchor"] == "Birth Date Anchor":
        return f"Birth +{float(row['offset']):g}"
    return f"NOW {float(row['offset']):g}"


def _plot_seed_distribution(
    axis, x, values, color: str, *,
    jitter_bounds: Tuple[float, float] = (-0.28, 0.04),
    mean_offset: float = 0.20,
    summary_label: str = "Mean ± sample SD",
) -> None:
    """Show seed observations and mean ± sample SD at categorical positions.

    Horizontal offsets only separate overlapping observations; they carry no
    scientific meaning and do not consume any model random-number state.
    ``values`` has one row per seed and one column per category, including zeros.
    """
    offsets = np.linspace(*jitter_bounds, len(values))
    axis.scatter(
        (np.asarray(x)[None, :] + offsets[:, None]).ravel(),
        values.ravel(),
        s=18,
        color=color,
        alpha=0.45,
        edgecolors="none",
        label="Individual seeds",
        zorder=3,
    )
    axis.errorbar(
        np.asarray(x) + mean_offset,
        values.mean(axis=0),
        yerr=values.std(axis=0, ddof=1),
        fmt="D",
        linestyle="none",
        markersize=5,
        color=color,
        markeredgecolor="white",
        markeredgewidth=0.7,
        elinewidth=1.4,
        capsize=3,
        label=summary_label,
        zorder=4,
    )


def create_auc_figure(per_seed: pd.DataFrame, path: Path) -> None:
    """Plot seed ROC-AUC observations grouped by outcome, with mean ± SD."""
    figure, axis = plt.subplots(figsize=(6, 5.2))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][:2]
    for position, (outcome, color) in enumerate(zip(("FA", "GI"), colors)):
        rows = per_seed.loc[per_seed["outcome"].eq(outcome)].sort_values(
            "split_sampling_seed"
        )
        _plot_seed_distribution(
            axis, [position], rows["roc_auc"].to_numpy(float)[:, None], color,
            jitter_bounds=(-0.025, 0.025),
            mean_offset=0.0,
            summary_label="Mean ± SD",
        )
    axis.set_xticks([0, 1])
    axis.set_xticklabels(["FA", "GI"])
    axis.set_xlim(-0.6, 1.6)
    axis.set_ylabel("ROC-AUC")
    axis.set_title("Predictive robustness")
    axis.grid(axis="y", alpha=0.22)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    handles, labels = axis.get_legend_handles_labels()
    legend_entries = dict(zip(labels, handles))
    axis.legend(legend_entries.values(), legend_entries.keys(), frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)


def _temporal_plot_profiles(window_rows: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Validate complete saved observations and order them without zero filling."""
    labels = _ordered_window_labels()
    prepared = window_rows.copy()
    prepared["outcome"] = prepared["outcome"].str.strip().str.upper()
    valid_anchors = prepared["anchor"].isin(ANCHOR_OFFSET_MAP)
    if not (valid_anchors | prepared["is_full_timeline"].eq(True)).all():
        raise ValueError("Unexpected temporal window anchor.")
    prepared["window_label"] = prepared.apply(_window_label, axis=1)
    keys = ["outcome", "split_sampling_seed", "window_label"]
    expected = pd.MultiIndex.from_product(
        [("FA", "GI"), SPLIT_SAMPLING_SEEDS, labels], names=keys
    )
    indexed = prepared.set_index(keys)["path_usage_percentage"]
    if (
        not indexed.index.is_unique
        or len(expected.difference(indexed.index))
        or len(indexed.index.difference(expected))
    ):
        raise ValueError(
            "Temporal plot requires exactly one explicit usage row for every "
            "outcome, seed, and window; missing/duplicate/unexpected rows found."
        )
    values = pd.to_numeric(indexed.reindex(expected), errors="raise").to_numpy(float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 100)).any():
        raise ValueError("Temporal usage percentages must be finite and in [0, 100].")
    profiles = values.reshape(2, len(SPLIT_SAMPLING_SEEDS), len(labels))
    return dict(zip(("FA", "GI"), profiles))


def create_temporal_usage_figure(
    window_rows: pd.DataFrame,
    path: Path,
) -> None:
    """Plot categorical window usage observations and mean ± sample SD.

    Shading identifies Birth windows contained in the predefined 0–6 month
    region (FA), or the NOW −1 to NOW window (GI). Nested Birth windows remain
    separate categories; their usage is never combined into a region metric.
    """
    labels = _ordered_window_labels()
    profiles = _temporal_plot_profiles(window_rows)
    figure, axes = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True, sharey=True
    )
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][:2]
    x = np.arange(len(labels))
    birth_count = sum(label.startswith("Birth") for label in labels)
    now_end = len(labels) - 1
    for panel, axis, outcome, color in zip("ab", axes, ("FA", "GI"), colors):
        if outcome == "FA":
            highlighted = [
                labels.index(f"Birth +{offset:g}")
                for offset in ANCHOR_OFFSET_MAP["Birth Date Anchor"]
                if offset is not None and 0 < offset <= 6
            ]
            region_label = "Birth windows within 0–6 months"
        else:
            highlighted = [labels.index("NOW -1")]
            region_label = "NOW −1 to NOW"
        axis.axvspan(
            min(highlighted) - 0.5,
            max(highlighted) + 0.5,
            color="#d8b365",
            alpha=0.18,
            label=region_label,
            zorder=0,
        )
        _plot_seed_distribution(axis, x, profiles[outcome], color)
        for boundary in (birth_count - 0.5, now_end - 0.5):
            axis.axvline(boundary, color="0.5", linewidth=0.8, alpha=0.45)
        for center, group in (
            ((birth_count - 1) / 2, "Birth Date Anchor"),
            ((birth_count + now_end - 1) / 2, "NOW Anchor"),
            (now_end, "Full Timeline"),
        ):
            axis.text(
                center, 1.02, group, transform=axis.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=10,
            )
        axis.set_ylabel("Temporal usage (%)")
        axis.set_title(f"({panel}) {outcome}", loc="left", pad=30)
        axis.grid(axis="y", alpha=0.22)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(
            loc="upper right", bbox_to_anchor=(1, 1.22), ncol=3,
            frameon=False, fontsize=9,
        )
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=45, ha="right")
    axes[-1].set_xlabel("Temporal window (categorical; offsets in months)")
    axes[-1].set_xlim(-0.6, len(labels) - 0.4)
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)


def _package_versions() -> dict:
    versions = {}
    for package in ("numpy", "pandas", "scikit-learn", "matplotlib", "dill"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _git_metadata() -> dict:
    def run(*args):
        result = subprocess.run(
            ["git", *args],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty_worktree": bool(status) if status is not None else None,
    }


def source_metadata() -> dict:
    """Record compact code and environment provenance centrally."""
    paths = {
        "experiment": Path(__file__).resolve(),
        "temporal_tree": REPOSITORY_ROOT / "settree" / "temporal_tree.py",
        "temporal_explainability": (
            REPOSITORY_ROOT / "settree" / "temporal_explainability.py"
        ),
    }
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": _package_versions(),
        "git": _git_metadata(),
        "source_sha256": {
            name: sha256_file(path) for name, path in paths.items()
        },
    }


def scientific_configuration(
    selected: Mapping,
    input_hashes: Mapping[str, str],
    split_hash: str,
) -> dict:
    """Return the configuration that must not change on resume."""
    return {
        "experiment_name": EXPERIMENT_NAME,
        "input_paths": {name: str(path) for name, path in INPUT_PATHS.items()},
        "input_sha256": dict(input_hashes),
        "patient_split": {**TRAIN_TEST_SPLIT, "sha256": split_hash},
        "selected_model_configuration": dict(selected),
        "fixed_model_configuration": FIXED_MODEL_CONFIGURATION,
        "split_sampling_seeds": list(SPLIT_SAMPLING_SEEDS),
        "only_experimental_variable": "split_sampling_seed",
        "anchor_offset_map": ANCHOR_OFFSET_MAP,
        "effective_windows": effective_windows(),
        "aggregations": list(AGGREGATION_NAMES),
        "predictive_metric": "roc_auc",
        "temporal_usage_population": "fixed held-out test patients",
        "temporal_usage_unit": "one test-patient visit to one internal split",
        "dominance_definition": {
            "fa": "Birth percentage strictly exceeds NOW and Full Timeline",
            "gi": "NOW percentage strictly exceeds Birth and Full Timeline",
            "ties_and_no_split": False,
        },
    }


def _initial_manifest(
    fingerprint: str,
    input_hashes: Mapping,
    split_hash: str,
) -> dict:
    return {
        "experiment_name": EXPERIMENT_NAME,
        "configuration_fingerprint": fingerprint,
        "input_sha256": dict(input_hashes),
        "shared_patient_split_sha256": split_hash,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "seeds": {
            str(seed): {"status": "not_started"}
            for seed in SPLIT_SAMPLING_SEEDS
        },
    }


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_per_seed_results(output_dir: Path) -> pd.DataFrame:
    path = output_dir / PER_SEED_RESULTS_FILENAME
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def _replace_seed_rows(
    table: pd.DataFrame,
    seed: int,
    rows: Sequence[Mapping],
) -> pd.DataFrame:
    replacement = pd.DataFrame(rows)
    if table.empty:
        combined = replacement
    else:
        retained = table.loc[
            pd.to_numeric(table["split_sampling_seed"]).ne(seed)
        ]
        combined = pd.concat([retained, replacement], ignore_index=True)
    return combined.sort_values(
        ["split_sampling_seed", "outcome"]
    ).reset_index(drop=True)


def _seed_artifact_paths(output_dir: Path, seed: int) -> list[Path]:
    seed_dir = output_dir / "seeds" / f"seed_{seed}"
    return [
        seed_dir / "FA_tree.json",
        seed_dir / "FA_temporal_window_usage.csv",
        seed_dir / "GI_tree.json",
        seed_dir / "GI_temporal_window_usage.csv",
    ]


def verify_completed_seed(
    output_dir: Path,
    seed: int,
    manifest_entry: Mapping,
    per_seed: pd.DataFrame,
) -> bool:
    """Verify central rows and hashes before skipping a completed seed."""
    if manifest_entry.get("status") != "complete":
        return False
    rows = (
        per_seed.loc[
            pd.to_numeric(per_seed["split_sampling_seed"]).eq(seed)
        ]
        if not per_seed.empty
        else pd.DataFrame()
    )
    if set(rows.get("outcome", pd.Series(dtype=str))) != {"FA", "GI"}:
        raise RuntimeError(
            f"Completed seed {seed} lacks its two central rows."
        )
    if manifest_entry.get("result_rows_fingerprint") != (
        _seed_rows_fingerprint(rows)
    ):
        raise RuntimeError(
            f"Completed seed {seed} has changed central result rows."
        )
    expected_hashes = manifest_entry.get("artifact_sha256", {})
    for path in _seed_artifact_paths(output_dir, seed):
        if not path.is_file():
            raise RuntimeError(
                f"Completed seed {seed} is missing {path.name}."
            )
        relative = str(path.relative_to(output_dir))
        if expected_hashes.get(relative) != sha256_file(path):
            raise RuntimeError(
                f"Completed seed {seed} has a changed artifact."
            )
    return True


def _finalize_seed(
    output_dir: Path,
    seed: int,
    manifest: dict,
) -> pd.DataFrame:
    """Idempotently publish a computed seed from central pending state."""
    entry = manifest["seeds"][str(seed)]
    if entry.get("status") != "finalizing":
        raise RuntimeError(f"Seed {seed} is not ready to finalize.")
    attempt_dir = Path(entry["attempt_directory"])
    final_dir = output_dir / "seeds" / f"seed_{seed}"
    if final_dir.exists() and attempt_dir.exists():
        raise RuntimeError(
            f"Seed {seed} has both final and pending artifact directories."
        )
    if not final_dir.exists():
        if not attempt_dir.is_dir():
            raise RuntimeError(
                f"Seed {seed} has no recoverable attempt directory."
            )
        os.replace(attempt_dir, final_dir)
    per_seed = _read_per_seed_results(output_dir)
    per_seed = _replace_seed_rows(per_seed, seed, entry["pending_rows"])
    atomic_write_csv(per_seed, output_dir / PER_SEED_RESULTS_FILENAME)
    published_rows = per_seed.loc[
        pd.to_numeric(per_seed["split_sampling_seed"]).eq(seed)
    ]
    artifact_hashes = {
        str(path.relative_to(output_dir)): sha256_file(path)
        for path in _seed_artifact_paths(output_dir, seed)
    }
    manifest["seeds"][str(seed)] = {
        "status": "complete",
        "completed_at": utc_now(),
        "artifact_sha256": artifact_hashes,
        "result_rows_fingerprint": _seed_rows_fingerprint(published_rows),
    }
    manifest["updated_at"] = utc_now()
    atomic_write_json(manifest, output_dir / MANIFEST_FILENAME)
    return per_seed


def _prepare_run(
    selected: Mapping,
    input_hashes: Mapping[str, str],
    split: pd.DataFrame,
    output_dir: Path,
) -> Tuple[dict, str]:
    """Create or verify all central pre-training artifacts."""
    config_path = output_dir / "experiment_config.json"
    if (
        output_dir.exists()
        and any(output_dir.iterdir())
        and not config_path.is_file()
    ):
        raise RuntimeError(
            "The Ensemble Robustness output directory is non-empty but has no "
            "experiment_config.json."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = output_dir / "shared_patient_split.csv"
    if split_path.exists():
        saved = pd.read_csv(split_path)
        pd.testing.assert_frame_equal(
            saved.reset_index(drop=True),
            split.reset_index(drop=True),
            check_dtype=False,
        )
    else:
        atomic_write_csv(split, split_path)
    split_hash = sha256_file(split_path)
    scientific = scientific_configuration(selected, input_hashes, split_hash)
    fingerprint = configuration_fingerprint(scientific)
    if config_path.exists():
        saved_config = _read_json(config_path)
        if saved_config.get("configuration_fingerprint") != fingerprint:
            raise RuntimeError("Existing Ensemble Robustness configuration differs.")
    else:
        atomic_write_json(
            {
                **scientific,
                "configuration_fingerprint": fingerprint,
                "provenance": source_metadata(),
            },
            config_path,
        )
    manifest_path = output_dir / MANIFEST_FILENAME
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        if manifest.get("configuration_fingerprint") != fingerprint:
            raise RuntimeError(
                "Existing run manifest has a different configuration."
            )
    else:
        manifest = _initial_manifest(fingerprint, input_hashes, split_hash)
        atomic_write_json(manifest, manifest_path)
    (output_dir / "seeds").mkdir(exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    return manifest, fingerprint


def _collect_window_rows(output_dir: Path) -> pd.DataFrame:
    tables = []
    for seed in SPLIT_SAMPLING_SEEDS:
        for outcome in ("FA", "GI"):
            path = (
                output_dir
                / "seeds"
                / f"seed_{seed}"
                / f"{outcome}_temporal_window_usage.csv"
            )
            if not path.is_file():
                raise RuntimeError(
                    f"Missing completed window artifact: {path}"
                )
            tables.append(pd.read_csv(path))
    return pd.concat(tables, ignore_index=True)


def write_final_outputs(output_dir: Path, per_seed: pd.DataFrame) -> None:
    """Write the three cross-seed artifacts and two approved figures."""
    if len(per_seed) != 20:
        raise RuntimeError("Final summaries require exactly 20 outcome rows.")
    summary = summarize_per_seed(per_seed)
    windows = _collect_window_rows(output_dir)
    window_summary = summarize_temporal_windows(windows)
    atomic_write_csv(summary, output_dir / "robustness_summary.csv")
    atomic_write_csv(
        window_summary, output_dir / "temporal_window_stability.csv"
    )
    create_auc_figure(
        per_seed,
        output_dir / "figures" / "auc_across_seeds.png",
    )
    create_temporal_usage_figure(
        windows, output_dir / "figures" / "temporal_usage_across_seeds.png"
    )


def validate_only() -> dict:
    """Validate configuration, inputs, and split without fitting or writing."""
    selected = validate_experiment_design()
    for name, path in INPUT_PATHS.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing V2 {name} artifact: {path}")
    input_hashes = {
        name: sha256_file(path) for name, path in INPUT_PATHS.items()
    }
    data, labels = load_inputs()
    split = create_shared_split(data, labels)
    return {
        "status": "validated_without_training",
        "selected_model_configuration": selected,
        "input_sha256": input_hashes,
        "n_patients": int(split["patient_id"].nunique()),
        "n_train_patients": int(split["split"].eq("train").sum()),
        "n_test_patients": int(split["split"].eq("test").sum()),
        "split_sampling_seeds": list(SPLIT_SAMPLING_SEEDS),
        "planned_tree_fits": 20,
        "effective_window_count": len(effective_windows()),
    }


def run_experiment(
    retry_failed_seed: Optional[int] = None,
    output_dir: Path = OUTPUT_DIR,
) -> dict:
    """Run or resume the complete robustness experiment."""
    selected = validate_experiment_design()
    if (
        retry_failed_seed is not None
        and retry_failed_seed not in SPLIT_SAMPLING_SEEDS
    ):
        raise ValueError(
            "retry_failed_seed is not one of the configured seeds."
        )
    for name, path in INPUT_PATHS.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing V2 {name} artifact: {path}")
    input_hashes_before = {
        name: sha256_file(path) for name, path in INPUT_PATHS.items()
    }
    data, labels = load_inputs()
    split = create_shared_split(data, labels)
    train_ids = split_patient_ids(split, "train")
    test_ids = split_patient_ids(split, "test")
    manifest, _ = _prepare_run(
        selected, input_hashes_before, split, Path(output_dir)
    )
    manifest_path = Path(output_dir) / MANIFEST_FILENAME

    # Finish any seed interrupted during the small atomic-publication phase.
    for seed in SPLIT_SAMPLING_SEEDS:
        if manifest["seeds"][str(seed)].get("status") == "finalizing":
            _finalize_seed(Path(output_dir), seed, manifest)

    per_seed = _read_per_seed_results(Path(output_dir))
    for seed in SPLIT_SAMPLING_SEEDS:
        entry = manifest["seeds"][str(seed)]
        if entry.get("status") == "complete":
            verify_completed_seed(Path(output_dir), seed, entry, per_seed)
            print(f"[RESUME] Skipping completed seed {seed}")
            continue
        if entry.get("status") in {"running", "failed"}:
            if retry_failed_seed != seed:
                print(
                    f"[RESUME] Seed {seed} is {entry.get('status')}; "
                    "an explicit retry is required."
                )
                continue
        elif retry_failed_seed == seed:
            raise RuntimeError(
                f"Seed {seed} has not failed and needs no retry."
            )

        attempt_dir = (
            Path(output_dir)
            / "seeds"
            / f".seed_{seed}.{uuid4().hex}.tmp"
        )
        attempt_dir.mkdir(parents=False, exist_ok=False)
        manifest["seeds"][str(seed)] = {
            "status": "running",
            "started_at": utc_now(),
            "attempt_directory": str(attempt_dir),
        }
        manifest["updated_at"] = utc_now()
        atomic_write_json(manifest, manifest_path)
        try:
            rows = []
            for outcome in OUTCOMES:
                print(f"[RUN] seed={seed} outcome={outcome.upper()}")
                row, _ = run_outcome(
                    outcome=outcome,
                    split_sampling_seed=seed,
                    data=data,
                    labels=labels[outcome],
                    train_ids=train_ids,
                    test_ids=test_ids,
                    attempt_dir=attempt_dir,
                )
                rows.append(row)
            manifest["seeds"][str(seed)] = {
                "status": "finalizing",
                "attempt_directory": str(attempt_dir),
                "pending_rows": rows,
            }
            manifest["updated_at"] = utc_now()
            atomic_write_json(manifest, manifest_path)
            per_seed = _finalize_seed(Path(output_dir), seed, manifest)
        except Exception as error:
            if manifest["seeds"][str(seed)].get("status") == "finalizing":
                # The pending rows are already durable in the central manifest.
                # A later run can safely finish this idempotent publication.
                raise
            manifest["seeds"][str(seed)] = {
                "status": "failed",
                "failed_at": utc_now(),
                "attempt_directory": str(attempt_dir),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            manifest["updated_at"] = utc_now()
            atomic_write_json(manifest, manifest_path)
            raise

    per_seed = _read_per_seed_results(Path(output_dir))
    all_complete = all(
        manifest["seeds"][str(seed)].get("status") == "complete"
        for seed in SPLIT_SAMPLING_SEEDS
    )
    if all_complete:
        write_final_outputs(Path(output_dir), per_seed)
    input_hashes_after = {
        name: sha256_file(path) for name, path in INPUT_PATHS.items()
    }
    if input_hashes_after != input_hashes_before:
        raise RuntimeError("A V2 input changed during Ensemble Robustness.")
    return {
        "status": "complete" if all_complete else "incomplete",
        "completed_seeds": sum(
            manifest["seeds"][str(seed)].get("status") == "complete"
            for seed in SPLIT_SAMPLING_SEEDS
        ),
        "output_directory": str(output_dir),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate configuration, V2 inputs, and split without writing.",
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="Run or safely resume the approved 20-fit experiment.",
    )
    parser.add_argument(
        "--retry-failed-seed",
        type=int,
        help="Explicitly retry one failed or interrupted configured seed.",
    )
    args = parser.parse_args(argv)
    if args.retry_failed_seed is not None and not args.run:
        parser.error("--retry-failed-seed requires --run")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.validate_only:
        print(json.dumps(validate_only(), indent=2, sort_keys=True))
        return
    result = run_experiment(retry_failed_seed=args.retry_failed_seed)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
