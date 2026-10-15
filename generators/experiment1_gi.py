"""Shared GI outcome mechanisms for final Experiment 1 generation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit


MASTER_SEED = 20260823
N_PATIENTS = 1000
TARGET_GI_PREVALENCE = 0.12
OUTPUT_DIR = Path(".generated_base")

GI_OR = {
    "VomitingYN": 3.00,
    "DiarrheaYN": 2.50,
    "GISymptomYN": 2.00,
}
GI_PREDICTOR_COLUMNS = {
    "VomitingYN": "V_i",
    "DiarrheaYN": "D_i",
    "GISymptomYN": "G_i",
}
GI_EXCLUDED_DISTRACTORS = ("FoodExposureYN", "ClinicalVisitYN")
GI_RNG_SPAWN_KEY = (6,)
LABEL_COLUMNS = ["patient_id", "label"]
GI_GROUND_TRUTH_COLUMNS = [
    "patient_id",
    "V_i",
    "D_i",
    "G_i",
    "n_active_predictors",
    "gi_probability",
    "gi_label",
]


@dataclass(frozen=True)
class GIConfig:
    """Configuration for GI-label derivation from an immutable X artifact."""

    n_patients: int = N_PATIENTS
    master_seed: int = MASTER_SEED
    target_prevalence: float = TARGET_GI_PREVALENCE
    output_dir: Path = OUTPUT_DIR
    benchmark_version: str = "phase_2_gi_v1"
    overwrite: bool = False


@dataclass
class GIBenchmarkResult:
    """In-memory GI labels, private ground truth, and audit metadata."""

    labels: pd.DataFrame
    ground_truth: pd.DataFrame
    metadata: dict


def validate_gi_configuration(config: GIConfig) -> None:
    """Validate the fixed scientific specification and run configuration."""
    if config.n_patients < 1:
        raise ValueError("n_patients must be positive.")
    if not 0 < config.target_prevalence < 1:
        raise ValueError(
            "target_prevalence must lie strictly between 0 and 1."
        )
    if set(GI_OR) != set(GI_PREDICTOR_COLUMNS):
        raise ValueError("GI ORs and predictor mappings must match exactly.")
    if set(GI_OR) & set(GI_EXCLUDED_DISTRACTORS):
        raise ValueError("Excluded distractors cannot enter the GI mechanism.")
    if any(value <= 0 for value in GI_OR.values()):
        raise ValueError("Every GI odds ratio must be positive.")


def _extract_anchor_times(data: pd.DataFrame, anchor_name: str) -> pd.Series:
    anchors = data.loc[
        data["feature_name"].eq(anchor_name),
        ["patient_id", "value"],
    ].copy()
    counts = anchors.groupby("patient_id").size()
    patient_ids = pd.Index(data["patient_id"].unique())
    counts = counts.reindex(patient_ids, fill_value=0)
    if not counts.eq(1).all():
        raise ValueError(
            f"Every patient must have exactly one {anchor_name} row."
        )
    anchors["value"] = pd.to_datetime(anchors["value"], errors="coerce")
    if anchors["value"].isna().any():
        raise ValueError(f"Every {anchor_name} value must be a timestamp.")
    return anchors.set_index("patient_id")["value"].reindex(patient_ids)


def extract_gi_predictors(data: pd.DataFrame) -> pd.DataFrame:
    """Extract approved binary predictors from each patient's final month."""
    required_columns = {"patient_id", "time", "feature_name", "value"}
    if not required_columns.issubset(data.columns):
        missing = sorted(required_columns - set(data.columns))
        raise ValueError(f"Longitudinal X is missing columns: {missing}.")

    patient_ids = pd.Index(
        sorted(data["patient_id"].unique()),
        name="patient_id",
    )
    now_by_patient = _extract_anchor_times(data, "NOW Anchor").reindex(
        patient_ids
    )
    features = data.loc[
        data["feature_name"].isin(GI_PREDICTOR_COLUMNS),
        ["patient_id", "time", "feature_name", "value"],
    ].copy()
    features["time"] = pd.to_datetime(features["time"], errors="coerce")
    features["now"] = features["patient_id"].map(now_by_patient)
    features["window_start"] = features["now"] - pd.DateOffset(months=1)
    in_window = (
        features["time"].ge(features["window_start"])
        & features["time"].le(features["now"])
    )
    final_month = features.loc[in_window].copy()
    final_month["value"] = pd.to_numeric(
        final_month["value"],
        errors="coerce",
    )
    if final_month["value"].isna().any():
        raise ValueError(
            "GI predictor values in the true window must be numeric."
        )
    if not final_month["value"].isin([0, 1]).all():
        raise ValueError("GI predictor values must be binary.")

    coverage = final_month.groupby(["patient_id", "feature_name"]).size()
    expected_index = pd.MultiIndex.from_product(
        [patient_ids, GI_PREDICTOR_COLUMNS],
        names=["patient_id", "feature_name"],
    )
    if not coverage.reindex(expected_index, fill_value=0).gt(0).all():
        raise ValueError(
            "Every patient must have observations for every approved GI "
            "predictor in the final month."
        )

    maxima = final_month.pivot_table(
        index="patient_id",
        columns="feature_name",
        values="value",
        aggfunc="max",
    ).reindex(patient_ids)
    predictors = maxima.rename(columns=GI_PREDICTOR_COLUMNS).reset_index()
    predictor_names = list(GI_PREDICTOR_COLUMNS.values())
    predictors = predictors[["patient_id", *predictor_names]]
    predictors[predictor_names] = predictors[predictor_names].astype(int)
    predictors["n_active_predictors"] = predictors[predictor_names].sum(
        axis=1
    )
    return predictors


def calibrate_gi_intercept(
    predictors: pd.DataFrame,
    target_prevalence: float,
) -> float:
    """Calibrate alpha0 for the realized approved-predictor distribution."""
    linear_effect = _calculate_linear_effect(predictors)

    def objective(intercept: float) -> float:
        return (
            float(expit(intercept + linear_effect).mean())
            - target_prevalence
        )

    return float(brentq(objective, -30.0, 30.0, xtol=1e-12, rtol=1e-12))


def _calculate_linear_effect(predictors: pd.DataFrame) -> np.ndarray:
    effect = np.zeros(len(predictors), dtype=float)
    for feature, predictor_name in GI_PREDICTOR_COLUMNS.items():
        effect += predictors[predictor_name].to_numpy(dtype=float) * math.log(
            GI_OR[feature]
        )
    return effect


def calculate_gi_probabilities(
    predictors: pd.DataFrame,
    intercept: float,
) -> np.ndarray:
    """Calculate theoretical GI probabilities from approved predictors only."""
    return expit(intercept + _calculate_linear_effect(predictors))


def make_gi_rng(master_seed: int) -> np.random.Generator:
    """Create the dedicated GI-label RNG without reusing the FA stream."""
    sequence = np.random.SeedSequence(
        entropy=master_seed,
        spawn_key=GI_RNG_SPAWN_KEY,
    )
    return np.random.default_rng(sequence)


def sample_gi_labels(
    probabilities: Sequence[float],
    master_seed: int,
) -> np.ndarray:
    """Sample reproducible Bernoulli labels from theoretical probabilities."""
    return make_gi_rng(master_seed).binomial(1, probabilities).astype(int)


def generate_gi_benchmark(
    data: pd.DataFrame,
    config: GIConfig = GIConfig(),
) -> GIBenchmarkResult:
    """Derive and validate GI labels without modifying longitudinal X."""
    validate_gi_configuration(config)
    original_data = data.copy(deep=True)
    predictors = extract_gi_predictors(data)
    if len(predictors) != config.n_patients:
        raise ValueError(
            f"Expected {config.n_patients} patients, found {len(predictors)}."
        )

    intercept = calibrate_gi_intercept(
        predictors,
        config.target_prevalence,
    )
    probabilities = calculate_gi_probabilities(predictors, intercept)
    sampled_labels = sample_gi_labels(probabilities, config.master_seed)

    labels = predictors[["patient_id"]].copy()
    labels["label"] = sampled_labels
    ground_truth = predictors.copy()
    ground_truth["gi_probability"] = probabilities
    ground_truth["gi_label"] = sampled_labels
    ground_truth = ground_truth[GI_GROUND_TRUTH_COLUMNS]

    validation = validate_gi_benchmark(
        data,
        labels,
        ground_truth,
        intercept,
        config,
    )
    if not data.equals(original_data):
        raise RuntimeError("GI generation mutated longitudinal X in memory.")

    predictor_columns = list(GI_PREDICTOR_COLUMNS.values())
    probability_series = ground_truth["gi_probability"]
    metadata = {
        "benchmark_version": config.benchmark_version,
        "source_x_status": "read_only_unchanged",
        "outcome_interpretation": (
            "Acute GI/stomach-upset outcome assessed immediately after the "
            "patient-specific NOW observation cutoff."
        ),
        "cohort_size": config.n_patients,
        "master_seed": config.master_seed,
        "true_temporal_window": {
            "anchor": "NOW Anchor",
            "start_offset": "-1 calendar month",
            "end_offset": "0 months",
            "bounds": "inclusive",
        },
        "predictors": {
            feature: {
                "private_variable": GI_PREDICTOR_COLUMNS[feature],
                "aggregation": "max (any positive event)",
                "odds_ratio": float(GI_OR[feature]),
                "coefficient": float(math.log(GI_OR[feature])),
                "effect_size_status": "synthetic benchmark effect size",
            }
            for feature in GI_OR
        },
        "excluded_model_visible_distractors": list(
            GI_EXCLUDED_DISTRACTORS
        ),
        "external_ground_truth_inputs": {
            "feeding_group": False,
            "feeding_trajectory": False,
            "fa_probability": False,
            "fa_label": False,
            "early_feeding_state": False,
        },
        "target_prevalence": config.target_prevalence,
        "calibrated_intercept": intercept,
        "mean_theoretical_probability": float(probability_series.mean()),
        "realized_prevalence": float(labels["label"].mean()),
        "positive_count": int(labels["label"].sum()),
        "probability_distribution": _describe_numeric(probability_series),
        "predictor_distribution": {
            predictor: {
                "positive_count": int(ground_truth[predictor].sum()),
                "positive_proportion": float(ground_truth[predictor].mean()),
            }
            for predictor in predictor_columns
        },
        "active_predictor_distribution": {
            str(count): int(number)
            for count, number in ground_truth["n_active_predictors"]
            .value_counts()
            .sort_index()
            .items()
        },
        "probability_by_active_predictor_count": {
            str(count): {
                "n_patients": int(len(group)),
                "mean": float(group["gi_probability"].mean()),
                "min": float(group["gi_probability"].min()),
                "max": float(group["gi_probability"].max()),
            }
            for count, group in ground_truth.groupby("n_active_predictors")
        },
        "rng": {
            "name": "gi_labels",
            "entropy": config.master_seed,
            "spawn_key": list(GI_RNG_SPAWN_KEY),
            "bit_generator": make_gi_rng(
                config.master_seed
            ).bit_generator.__class__.__name__,
            "separate_from_fa_label_stream": True,
        },
        "label_schema": LABEL_COLUMNS,
        "private_ground_truth_schema": GI_GROUND_TRUTH_COLUMNS,
        "validation": validation,
    }
    return GIBenchmarkResult(labels, ground_truth, metadata)


def validate_gi_benchmark(
    data: pd.DataFrame,
    labels: pd.DataFrame,
    ground_truth: pd.DataFrame,
    intercept: float,
    config: GIConfig,
) -> dict:
    """Validate structural, mechanistic, and temporal GI invariants."""
    x_patient_ids = set(data["patient_id"])
    label_patient_ids = set(labels["patient_id"])
    ground_truth_patient_ids = set(ground_truth["patient_id"])
    extracted = extract_gi_predictors(data)
    predictor_columns = list(GI_PREDICTOR_COLUMNS.values())
    expected_probabilities = calculate_gi_probabilities(extracted, intercept)

    ordered_ground_truth = ground_truth.set_index("patient_id").reindex(
        extracted["patient_id"]
    )
    checks = {
        "cohort_size": len(x_patient_ids) == config.n_patients,
        "same_x_gi_patient_ids": (
            x_patient_ids == label_patient_ids == ground_truth_patient_ids
        ),
        "unique_gi_labels": labels["patient_id"].is_unique,
        "gi_label_schema": list(labels.columns) == LABEL_COLUMNS,
        "binary_gi_labels": labels["label"].isin([0, 1]).all(),
        "approved_predictors_only": set(GI_OR) == {
            "VomitingYN",
            "DiarrheaYN",
            "GISymptomYN",
        },
        "food_exposure_excluded": "FoodExposureYN" not in GI_OR,
        "clinical_visit_excluded": "ClinicalVisitYN" not in GI_OR,
        "predictors_match_exact_final_month_extraction": bool(
            ordered_ground_truth[predictor_columns]
            .reset_index(drop=True)
            .equals(extracted[predictor_columns].reset_index(drop=True))
        ),
        "active_count_matches_predictors": bool(
            ordered_ground_truth["n_active_predictors"].to_numpy().tolist()
            == extracted["n_active_predictors"].to_numpy().tolist()
        ),
        "probabilities_match_approved_equation": bool(
            np.allclose(
                ordered_ground_truth["gi_probability"],
                expected_probabilities,
                atol=1e-12,
                rtol=0.0,
            )
        ),
        "mean_probability_matches_target": math.isclose(
            float(ground_truth["gi_probability"].mean()),
            config.target_prevalence,
            abs_tol=1e-10,
            rel_tol=0.0,
        ),
        "no_external_ground_truth_dependency": True,
        "dedicated_gi_rng_stream": GI_RNG_SPAWN_KEY != (5,),
        "deterministic_label_sampling": bool(
            np.array_equal(
                labels["label"],
                sample_gi_labels(expected_probabilities, config.master_seed),
            )
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            "GI benchmark validation failed: " + ", ".join(sorted(failed))
        )
    return {"checks": {name: bool(value) for name, value in checks.items()}}


def temporal_specificity_audit(data: pd.DataFrame) -> dict:
    """Summarize approved predictor occurrence across temporal regions."""
    now_by_patient = _extract_anchor_times(data, "NOW Anchor")
    birth_by_patient = _extract_anchor_times(data, "Birth Date Anchor")
    patient_ids = pd.Index(sorted(data["patient_id"].unique()))
    observations = data.loc[
        data["feature_name"].isin(GI_OR),
        ["patient_id", "time", "feature_name", "value"],
    ].copy()
    observations["time"] = pd.to_datetime(observations["time"])
    observations["value"] = pd.to_numeric(observations["value"])
    observations["now"] = observations["patient_id"].map(now_by_patient)
    observations["birth"] = observations["patient_id"].map(birth_by_patient)

    window_masks = {
        "birth_0_to_6m": (
            observations["time"].ge(observations["birth"])
            & observations["time"].le(
                observations["birth"] + pd.DateOffset(months=6)
            )
        ),
        "middle_background": (
            observations["time"].gt(
                observations["birth"] + pd.DateOffset(months=6)
            )
            & observations["time"].lt(
                observations["now"] - pd.DateOffset(months=6)
            )
        ),
        "now_minus_6m_to_1m": (
            observations["time"].ge(
                observations["now"] - pd.DateOffset(months=6)
            )
            & observations["time"].lt(
                observations["now"] - pd.DateOffset(months=1)
            )
        ),
        "now_minus_1m_to_now": (
            observations["time"].ge(
                observations["now"] - pd.DateOffset(months=1)
            )
            & observations["time"].le(observations["now"])
        ),
    }
    regions = {
        name: _event_summary(observations.loc[mask], patient_ids)
        for name, mask in window_masks.items()
    }

    final_indicators = _window_any_indicators(
        observations,
        patient_ids,
        months=1,
    )
    nested = {}
    for months in (1, 2, 3, 6):
        indicators = _window_any_indicators(
            observations,
            patient_ids,
            months=months,
        )
        nested[f"now_minus_{months}m_to_now"] = {
            feature: {
                "positive_proportion": float(indicators[feature].mean()),
                "correlation_with_true_1m": _safe_correlation(
                    indicators[feature],
                    final_indicators[feature],
                ),
            }
            for feature in GI_OR
        }
    full_indicators = observations.pivot_table(
        index="patient_id",
        columns="feature_name",
        values="value",
        aggfunc="max",
    ).reindex(patient_ids, fill_value=0)
    nested["full_timeline"] = {
        feature: {
            "positive_proportion": float(full_indicators[feature].mean()),
            "correlation_with_true_1m": _safe_correlation(
                full_indicators[feature],
                final_indicators[feature],
            ),
        }
        for feature in GI_OR
    }
    return {"regions": regions, "nested_now_windows": nested}


def _window_any_indicators(
    observations: pd.DataFrame,
    patient_ids: pd.Index,
    months: int,
) -> pd.DataFrame:
    mask = (
        observations["time"].ge(
            observations["now"] - pd.DateOffset(months=months)
        )
        & observations["time"].le(observations["now"])
    )
    return observations.loc[mask].pivot_table(
        index="patient_id",
        columns="feature_name",
        values="value",
        aggfunc="max",
    ).reindex(patient_ids, fill_value=0)


def _event_summary(
    observations: pd.DataFrame,
    patient_ids: pd.Index,
) -> dict:
    summary = {}
    for feature in GI_OR:
        counts = observations.loc[
            observations["feature_name"].eq(feature)
        ].groupby("patient_id")["value"].sum().reindex(
            patient_ids,
            fill_value=0,
        )
        summary[feature] = {
            "positive_proportion": float(counts.gt(0).mean()),
            "mean_event_count": float(counts.mean()),
            "median_event_count": float(counts.median()),
            "min_event_count": int(counts.min()),
            "max_event_count": int(counts.max()),
            "zero_proportion": float(counts.eq(0).mean()),
        }
    return summary


def independence_audit(
    result: GIBenchmarkResult,
    fa_labels: pd.DataFrame,
    fa_ground_truth: pd.DataFrame,
) -> dict:
    """Measure finite-sample GI associations without using FA in generation."""
    audit = result.ground_truth.merge(
        fa_labels.rename(columns={"label": "fa_label"}),
        on="patient_id",
        validate="one_to_one",
    ).merge(
        fa_ground_truth[["patient_id", "feeding_group", "fa_probability"]],
        on="patient_id",
        validate="one_to_one",
    )
    group_means = audit.groupby("feeding_group")["gi_probability"].mean()
    return {
        "correlation_gi_probability_with_fa_probability": _safe_correlation(
            audit["gi_probability"],
            audit["fa_probability"],
        ),
        "correlation_gi_probability_with_fa_label": _safe_correlation(
            audit["gi_probability"],
            audit["fa_label"],
        ),
        "correlation_gi_label_with_fa_label": _safe_correlation(
            audit["gi_label"],
            audit["fa_label"],
        ),
        "gi_fa_both_positive_count": int(
            (audit["gi_label"].eq(1) & audit["fa_label"].eq(1)).sum()
        ),
        "mean_gi_probability_by_feeding_group": {
            str(group): float(value) for group, value in group_means.items()
        },
        "mean_gi_probability_group_range": {
            "min": float(group_means.min()),
            "max": float(group_means.max()),
        },
        "generation_used_fa_or_feeding_inputs": False,
    }


def _safe_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if np.std(left_array) == 0 or np.std(right_array) == 0:
        return float("nan")
    return float(np.corrcoef(left_array, right_array)[0, 1])


def _describe_numeric(values: pd.Series) -> dict:
    return {
        "min": float(values.min()),
        "q25": float(values.quantile(0.25)),
        "median": float(values.median()),
        "q75": float(values.quantile(0.75)),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
    }


def sha256_file(path: Path) -> str:
    """Calculate a source-artifact hash without changing the file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gi_artifact_paths(config: GIConfig) -> Dict[str, Path]:
    """Return separate GI artifact paths, leaving Phase-1 files untouched."""
    suffix = str(config.n_patients)
    return {
        "labels": (
            config.output_dir / f"temporal_benchmark_gi_labels_{suffix}.pkl"
        ),
        "metadata": (
            config.output_dir
            / f"temporal_benchmark_gi_metadata_{suffix}.json"
        ),
        "ground_truth": (
            config.output_dir
            / f"temporal_benchmark_gi_ground_truth_{suffix}.pkl"
        ),
    }


def save_gi_benchmark(
    result: GIBenchmarkResult,
    config: GIConfig,
) -> Dict[str, Path]:
    """Atomically save only new GI artifacts."""
    paths = gi_artifact_paths(config)
    existing = [path for path in paths.values() if path.exists()]
    if existing and not config.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing GI artifacts: "
            + ", ".join(str(path) for path in existing)
        )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".temporal_benchmark_gi_",
        dir=config.output_dir,
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        temporary_paths = {
            name: temporary_root / path.name for name, path in paths.items()
        }
        result.labels.to_pickle(temporary_paths["labels"])
        result.ground_truth.to_pickle(temporary_paths["ground_truth"])
        temporary_paths["metadata"].write_text(
            json.dumps(result.metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        for name, path in paths.items():
            temporary_paths[name].replace(path)
    return paths
