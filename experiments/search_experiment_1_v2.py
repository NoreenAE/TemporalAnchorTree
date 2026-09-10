"""Smart, resumable TemporalTree configuration search for Experiment 1 V2.

The search deliberately separates model selection from final evaluation:

1. Eight scientifically motivated hyperparameter configurations are screened
   on one shared validation split with sampling seed 42. Every configuration
   receives the same broad temporal search space, including irrelevant
   alternative windows.
2. Four finalists (the top three combined scores plus the strongest temporal
   recovery candidate) are evaluated with seeds 42, 43, and 44.
3. The three most stable finalists are retrained on the complete outer
   training set and evaluated once on the untouched outer test set.

No benchmark data are written by this script. Existing Experiment 1 helpers
are reused for loading, prediction metrics, tree export, and explainability.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter
import traceback
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from experiments import experiment_1_fa_vs_gi as experiment_1
from settree.temporal_explainability import (
    aggregate_temporal_usage,
    derive_monthly_bin_edges,
    load_temporal_usage_artifacts,
    plot_temporal_usage,
    plot_temporal_usage_comparison,
    trace_temporal_decision_paths,
)
from settree.temporal_tree import TemporalTree


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPOSITORY_ROOT / "data" / "temporal_benchmark_v2"
INPUT_PATHS = {
    "x": DATA_DIR / "temporal_benchmark_5000.pkl",
    "fa": DATA_DIR / "temporal_benchmark_fa_labels_5000.pkl",
    "gi": DATA_DIR / "temporal_benchmark_gi_labels_5000.pkl",
}
OUTPUT_DIR = (
    REPOSITORY_ROOT / "experiments" / "results" / "v2_hyperparameter_search"
)

OUTER_TEST_SIZE = 0.30
OUTER_SPLIT_SEED = 42
VALIDATION_FRACTION_OF_OUTER_TRAIN = 0.25
INNER_SPLIT_SEED = 137
SCREENING_SEED = 42
STABILITY_SEEDS = (42, 43, 44)
N_FINALISTS = 4
N_TOP_CANDIDATES = 3
AGGREGATION_NAMES = ("mean", "sum", "max" ) #"count"

BROAD_ANCHOR_OFFSET_MAP = {
    "Birth Date Anchor": [None, 0.7, 1, 2, 3, 4, 5, 6, 12, 18],
    "NOW Anchor": [None, -1, -2, -3, -4, -5, -6, -12, -18],
}

EXPECTED_TEMPORAL_REGIONS = {
    "fa": {
        "anchor": "Birth Date Anchor",
        "start_offset": 0.0,
        "end_offset": 6.0,
    },
    "gi": {
        "anchor": "NOW Anchor",
        "start_offset": -1.0,
        "end_offset": 0.0,
    },
}

SCORING_SPECIFICATION = {
    "temporal_recovery": (
        "Mean per-split overlap fraction with the outcome's known temporal "
        "region on the correct anchor. For a used interval W and true interval "
        "T, relevance = length(W intersection T) / length(W). Wrong anchors, "
        "Full Timeline, and zero-width windows receive 0."
    ),
    "auc_skill": "clip(2 * (ROC_AUC - 0.5), 0, 1)",
    "average_precision_skill": (
        "clip((Average Precision - evaluation prevalence) / "
        "(1 - evaluation prevalence), 0, 1)"
    ),
    "combined_score": (
        "Unweighted mean of FA AUC skill, FA AP skill, FA temporal recovery, "
        "GI AUC skill, GI AP skill, and GI temporal recovery."
    ),
    "finalist_rank": (
        "Descending mean combined validation score minus one sample SD across "
        "sampling seeds; mean combined score and candidate_id break ties."
    ),
    "pareto_reporting": (
        "A finalist is Pareto-optimal when no other finalist is at least as "
        "good on all six mean normalized components and strictly better on "
        "at least one. Pareto status is reported but does not replace the "
        "predeclared stability-adjusted ranking."
    ),
}

PREVIOUS_RESULT_EVIDENCE = {
    "v2_a_0_1": {
        "candidate_sampling_rate": 0.1,
        "fa_roc_auc": 0.6017,
        "gi_roc_auc": 0.6215,
        "conclusion": "Excluded because it was clearly weaker than A=0.4.",
    },
    "v2_a_0_4_reference": {
        "candidate_sampling_rate": 0.4,
        "observed_pair_training_minutes": 39.7,
        "conclusion": "Retained as the evidence-based baseline.",
    },
    "v2_wide_offsets": {
        "observed_pair_training_minutes_range": [53.7, 65.4],
        "conclusion": (
            "The broad 0.7/1-6/12/18 and -1-6/-12/-18 search space is fixed "
            "for every candidate. Its distractor usage and higher runtime are "
            "part of the revised scientific stress test, not parameters to "
            "optimize away."
        ),
    },
}

ESTIMATED_RUNTIME = {
    "maximum_model_pairs": 19,
    "maximum_individual_tree_fits": 38,
    "estimated_wall_clock_hours_range": [18, 30],
    "basis": (
        "Saved V2 A=0.4 broad-window runs required about 54 minutes per FA/GI "
        "pair at depth three on 3,500 training patients. Screening uses 2,625 "
        "training patients, but depth-four/five trees, A=0.6, multi-seed "
        "confirmation, and final full-training refits add substantial overhead."
    ),
}


@dataclass(frozen=True)
class SearchCandidate:
    """One deliberately selected TemporalTree configuration."""

    candidate_id: str
    candidate_sampling_rate: float
    max_depth: int
    min_samples_split: int
    min_samples_leaf: int
    min_impurity_decrease: float
    rationale: str

    def __post_init__(self) -> None:
        if self.min_samples_split < 2 * self.min_samples_leaf:
            raise ValueError(
                "min_samples_split must be at least twice min_samples_leaf."
            )

    @property
    def anchor_offset_map(self) -> Dict[str, list]:
        """Return a fresh copy of the fixed broad temporal search space."""
        return {
            anchor: list(offsets)
            for anchor, offsets in BROAD_ANCHOR_OFFSET_MAP.items()
        }

    def model_parameters(self, split_sampling_seed: int) -> dict:
        """Return the complete existing TemporalTree parameter mapping."""
        anchor_offset_map = self.anchor_offset_map
        return {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "anchors_list": list(anchor_offset_map),
            "anchor_offset_list": [None],
            "is_classification": False,
            "criterion": "mse",
            "learn_missing_direction": False,
            "random_state": OUTER_SPLIT_SEED,
            "candidate_sampling_rate": self.candidate_sampling_rate,
            "split_sampling_seed": int(split_sampling_seed),
            "min_impurity_decrease": self.min_impurity_decrease,
            "verbose": 1,
            "use_dynamic_anchors": False,
            "anchor_offset_map": anchor_offset_map,
        }

    def to_dict(self) -> dict:
        """Return a complete JSON-safe candidate description."""
        return {
            "candidate_id": self.candidate_id,
            "candidate_sampling_rate": self.candidate_sampling_rate,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "min_impurity_decrease": self.min_impurity_decrease,
            "temporal_search_space": "broad_fixed",
            "anchor_offset_map": self.anchor_offset_map,
            "aggregations": list(AGGREGATION_NAMES),
            "use_dynamic_anchors": False,
            "rationale": self.rationale,
        }


SCREENING_CANDIDATES = (
    SearchCandidate(
        "C01",
        0.4,
        3,
        20,
        10,
        0.0,
        "Broad-space baseline with moderate sample regularization and no gain "
        "threshold.",
    ),
    SearchCandidate(
        "C02",
        0.4,
        3,
        40,
        20,
        0.0,
        "Isolates stronger sample regularization from the baseline while "
        "leaving the gain threshold disabled.",
    ),
    SearchCandidate(
        "C03",
        0.4,
        3,
        20,
        10,
        0.0005,
        "Isolates the smallest gain threshold from C01 at the same depth and "
        "sample constraints.",
    ),
    SearchCandidate(
        "C04",
        0.4,
        4,
        40,
        20,
        0.0005,
        "Adds one level of capacity with stronger sample regularization and a "
        "small gain threshold to suppress broad-space distractors.",
    ),
    SearchCandidate(
        "C05",
        0.4,
        4,
        40,
        20,
        0.001,
        "Strengthens only the gain requirement relative to C04 at depth four.",
    ),
    SearchCandidate(
        "C06",
        0.4,
        4,
        40,
        20,
        0.002,
        "Tests the strongest planned gain threshold at depth four without "
        "crossing a full grid.",
    ),
    SearchCandidate(
        "C07",
        0.4,
        5,
        60,
        30,
        0.001,
        "Tests depth five only with stronger split, leaf, and gain "
        "regularization appropriate to the broad search space.",
    ),
    SearchCandidate(
        "C08",
        0.6,
        4,
        40,
        20,
        0.001,
        "Tests whether denser candidate sampling improves broad-space recovery "
        "using the balanced depth-four regularization setting from C05.",
    ),
)

COMPONENT_NAMES = (
    "fa_auc_skill",
    "fa_average_precision_skill",
    "fa_temporal_recovery_score",
    "gi_auc_skill",
    "gi_average_precision_skill",
    "gi_temporal_recovery_score",
)


def _json_bytes(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=experiment_1._json_default,
    ).encode("utf-8")


def configuration_fingerprint(value) -> str:
    """Return a stable SHA-256 identifier for a JSON-safe configuration."""
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def search_plan() -> dict:
    """Return the complete predeclared search design."""
    return {
        "search_name": "experiment_1_v2_smart_hyperparameter_search",
        "dataset_version": "V2",
        "scientific_question": (
            "Which TemporalTree hyperparameter configuration provides the "
            "best balance of predictive performance and temporal recovery "
            "when the model is exposed to a broad set of candidate temporal "
            "windows, including irrelevant alternatives?"
        ),
        "input_paths": {name: str(path) for name, path in INPUT_PATHS.items()},
        "split_protocol": {
            "outer_test_size": OUTER_TEST_SIZE,
            "outer_split_seed": OUTER_SPLIT_SEED,
            "validation_fraction_of_outer_train": (
                VALIDATION_FRACTION_OF_OUTER_TRAIN
            ),
            "inner_split_seed": INNER_SPLIT_SEED,
            "joint_stratification": "FA label + GI label",
            "selection_uses_test_set": False,
        },
        "screening_seed": SCREENING_SEED,
        "stability_seeds": list(STABILITY_SEEDS),
        "n_screening_configurations": len(SCREENING_CANDIDATES),
        "n_finalists": N_FINALISTS,
        "n_top_candidates": N_TOP_CANDIDATES,
        "candidate_configurations": [
            candidate.to_dict() for candidate in SCREENING_CANDIDATES
        ],
        "fixed_temporal_search_space": {
            "optimized_during_search": False,
            "anchor_offset_map": {
                anchor: list(offsets)
                for anchor, offsets in BROAD_ANCHOR_OFFSET_MAP.items()
            },
            "full_timeline_behavior": (
                "None retains TemporalTree's existing deduplicated Full "
                "Timeline behavior."
            ),
        },
        "finalist_selection": (
            "Top three screening combined scores plus the highest mean FA/GI "
            "temporal recovery candidate; duplicates are filled by combined "
            "score until four finalists are selected."
        ),
        "representative_seed": (
            "Sampling seed whose combined validation score is closest to the "
            "candidate's median across seeds; lower seed breaks exact ties."
        ),
        "scoring": SCORING_SPECIFICATION,
        "previous_result_evidence": PREVIOUS_RESULT_EVIDENCE,
        "estimated_runtime": ESTIMATED_RUNTIME,
    }


def validate_search_plan(candidates: Sequence[SearchCandidate]) -> None:
    """Reject accidental grids, duplicate IDs, or invalid shared settings."""
    if len(candidates) != 8:
        raise ValueError("The bounded screening design must contain 8 candidates.")
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Candidate IDs must be unique.")
    observed_gains = {candidate.min_impurity_decrease for candidate in candidates}
    if observed_gains != {0.0, 0.0005, 0.001, 0.002}:
        raise ValueError("The predeclared gain range changed unexpectedly.")
    if any(candidate.candidate_sampling_rate < 0.4 for candidate in candidates):
        raise ValueError("Known-weak candidate sampling rates must be excluded.")
    if any(candidate.max_depth not in {3, 4, 5} for candidate in candidates):
        raise ValueError("Only the predeclared depth range 3/4/5 is allowed.")
    if any(
        candidate.anchor_offset_map != BROAD_ANCHOR_OFFSET_MAP
        for candidate in candidates
    ):
        raise ValueError(
            "Every candidate must use the identical broad temporal search space."
        )
    if any(
        candidate.model_parameters(42)["use_dynamic_anchors"]
        for candidate in candidates
    ):
        raise ValueError("Dynamic anchors must remain disabled.")


def load_v2_inputs() -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Load and validate the three immutable V2 model input artifacts."""
    data = experiment_1.load_longitudinal_x(INPUT_PATHS["x"])
    labels = {
        outcome: experiment_1.load_labels(INPUT_PATHS[outcome], outcome)
        for outcome in ("fa", "gi")
    }
    patient_ids = set(data["patient_id"])
    for outcome, outcome_labels in labels.items():
        if set(outcome_labels["patient_id"]) != patient_ids:
            raise ValueError(f"V2 X and {outcome} labels have different patients.")
    return data, labels


def create_shared_search_splits(
    data: pd.DataFrame,
    labels: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Create the unchanged outer test split plus an inner validation split."""
    patient_ids = np.asarray(sorted(data["patient_id"].unique()))
    split_table = pd.DataFrame({"patient_id": patient_ids})
    for outcome in ("fa", "gi"):
        by_patient = labels[outcome].set_index("patient_id")["label"]
        split_table[f"{outcome}_label"] = split_table["patient_id"].map(
            by_patient
        )
    if split_table.isna().any().any():
        raise ValueError("The shared split table contains missing labels.")
    split_table["joint_stratum"] = (
        split_table["fa_label"].astype(str)
        + "_"
        + split_table["gi_label"].astype(str)
    )
    if split_table["joint_stratum"].value_counts().lt(2).any():
        raise ValueError("Every joint stratum requires at least two patients.")

    outer_train_ids, test_ids = train_test_split(
        patient_ids,
        test_size=OUTER_TEST_SIZE,
        random_state=OUTER_SPLIT_SEED,
        stratify=split_table["joint_stratum"],
    )
    strata_by_patient = split_table.set_index("patient_id")["joint_stratum"]
    search_train_ids, validation_ids = train_test_split(
        outer_train_ids,
        test_size=VALIDATION_FRACTION_OF_OUTER_TRAIN,
        random_state=INNER_SPLIT_SEED,
        stratify=strata_by_patient.reindex(outer_train_ids),
    )
    assignments = {
        **{patient_id: "search_train" for patient_id in search_train_ids},
        **{patient_id: "validation" for patient_id in validation_ids},
        **{patient_id: "test" for patient_id in test_ids},
    }
    if len(assignments) != len(patient_ids):
        raise RuntimeError("Patient splits are not a disjoint partition.")
    split_table["split"] = split_table["patient_id"].map(assignments)
    return split_table


def split_patient_ids(
    split_table: pd.DataFrame,
    split_names: Iterable[str],
) -> np.ndarray:
    """Return IDs belonging to one or more named split partitions."""
    names = set(split_names)
    return split_table.loc[
        split_table["split"].isin(names), "patient_id"
    ].to_numpy()


def temporal_recovery_score(path_table: pd.DataFrame, outcome: str) -> float:
    """Score temporal recovery through overlap with known V2 ground truth."""
    if outcome not in EXPECTED_TEMPORAL_REGIONS:
        raise ValueError(f"Unknown outcome: {outcome}")
    if path_table.empty:
        return 0.0
    required = {
        "anchor",
        "effective_start_offset",
        "effective_end_offset",
        "is_full_timeline",
        "is_zero_width",
    }
    missing = required.difference(path_table.columns)
    if missing:
        raise ValueError(f"Path table is missing columns: {sorted(missing)}")

    expected = EXPECTED_TEMPORAL_REGIONS[outcome]
    weights = []
    for row in path_table.itertuples(index=False):
        if (
            row.is_full_timeline
            or row.is_zero_width
            or row.anchor != expected["anchor"]
        ):
            weights.append(0.0)
            continue
        window_start = float(row.effective_start_offset)
        window_end = float(row.effective_end_offset)
        window_length = window_end - window_start
        if window_length <= 0:
            weights.append(0.0)
            continue
        overlap = max(
            0.0,
            min(window_end, expected["end_offset"])
            - max(window_start, expected["start_offset"]),
        )
        weights.append(overlap / window_length)
    return float(np.mean(weights))


def predictive_skill_scores(
    roc_auc: float,
    average_precision: float,
    prevalence: float,
) -> Tuple[float, float]:
    """Normalize predictive metrics to transparent no-skill baselines."""
    if not 0 <= prevalence < 1:
        raise ValueError("Evaluation prevalence must be in [0, 1).")
    auc_skill = float(np.clip(2.0 * (roc_auc - 0.5), 0.0, 1.0))
    ap_skill = float(np.clip(
        (average_precision - prevalence) / (1.0 - prevalence),
        0.0,
        1.0,
    ))
    return auc_skill, ap_skill


def add_combined_score(pair_result: dict) -> dict:
    """Attach six normalized components and their equal-weight mean."""
    components = {}
    for outcome in ("fa", "gi"):
        result = pair_result["outcomes"][outcome]
        auc_skill, ap_skill = predictive_skill_scores(
            result["performance"]["roc_auc"],
            result["performance"]["average_precision"],
            result["evaluation_positive_prevalence"],
        )
        components[f"{outcome}_auc_skill"] = auc_skill
        components[f"{outcome}_average_precision_skill"] = ap_skill
        components[f"{outcome}_temporal_recovery_score"] = result[
            "temporal_recovery_score"
        ]
    pair_result["score_components"] = components
    pair_result["combined_score"] = float(np.mean([
        components[name] for name in COMPONENT_NAMES
    ]))
    return pair_result


def build_model(candidate: SearchCandidate, split_sampling_seed: int) -> TemporalTree:
    """Construct an unchanged TemporalTree from one search configuration."""
    return TemporalTree(
        **candidate.model_parameters(split_sampling_seed),
        aggregation_funcs=experiment_1.aggregation_functions(),
    )


def _save_no_split_plot(path: Path, title: str) -> None:
    """Save an honest placeholder when a fitted tree contains no split."""
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.axis("off")
    axis.text(
        0.5,
        0.5,
        "No temporal splits were used",
        ha="center",
        va="center",
        fontsize=14,
    )
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _save_combined_usage_plot(pair_directory: Path) -> Path:
    """Create the required FA-versus-GI usage figure from saved artifacts."""
    output_path = pair_directory / "temporal_usage_comparison_fa_vs_gi.png"
    artifacts = {
        outcome: load_temporal_usage_artifacts(
            pair_directory / outcome / "temporal_explainability"
        )
        for outcome in ("fa", "gi")
    }
    if all(
        int(value["summary"]["temporal_split_usages"]) > 0
        for value in artifacts.values()
    ):
        return plot_temporal_usage_comparison(artifacts, output_path)
    _save_no_split_plot(output_path, "FA vs GI Temporal Usage Comparison")
    return output_path


def run_outcome(
    candidate: SearchCandidate,
    split_sampling_seed: int,
    outcome: str,
    data: pd.DataFrame,
    labels: pd.DataFrame,
    train_ids: np.ndarray,
    evaluation_ids: np.ndarray,
    output_dir: Path,
    save_plots: bool,
) -> dict:
    """Fit and evaluate one outcome without changing Experiment 1 metrics."""
    outcome_start = perf_counter()
    output_dir.mkdir(parents=True, exist_ok=False)
    train_data, y_train, train_order = experiment_1.subset_for_patients(
        data, labels, train_ids
    )
    evaluation_data, y_evaluation, evaluation_order = (
        experiment_1.subset_for_patients(data, labels, evaluation_ids)
    )
    if set(train_order) & set(evaluation_order):
        raise RuntimeError("Patient leakage detected during search evaluation.")

    model = build_model(candidate, split_sampling_seed)
    training_start = perf_counter()
    model.fit(train_data, y_train)
    training_seconds = perf_counter() - training_start

    evaluation_start = perf_counter()
    scores = model.predict(evaluation_data)
    performance, predictions = experiment_1.evaluate_scores(
        y_evaluation, scores
    )
    evaluation_seconds = perf_counter() - evaluation_start
    pd.DataFrame({
        "patient_id": evaluation_order,
        "label": y_evaluation.to_numpy(dtype=int),
        "score": scores,
        "prediction": predictions,
    }).to_csv(output_dir / "predictions.csv", index=False)

    explainability_start = perf_counter()
    path_table = trace_temporal_decision_paths(model, evaluation_data)
    bin_edges = derive_monthly_bin_edges(candidate.anchor_offset_map)
    usage_table, explainability_summary = aggregate_temporal_usage(
        path_table,
        bin_edges,
        int(len(evaluation_order)),
    )
    usage_summary, window_distribution = experiment_1.temporal_usage_summary(
        path_table
    )
    recovery_score = temporal_recovery_score(path_table, outcome)
    explainability_seconds = perf_counter() - explainability_start

    explainability_dir = output_dir / "temporal_explainability"
    explainability_dir.mkdir()
    path_table.to_csv(
        explainability_dir / "temporal_path_usage.csv", index=False
    )
    usage_table.to_csv(
        explainability_dir / "aggregated_temporal_usage.csv", index=False
    )
    experiment_1.write_json(
        explainability_summary,
        explainability_dir / "temporal_usage_summary.json",
    )
    window_distribution.to_csv(
        explainability_dir / "temporal_window_distribution.csv", index=False
    )
    histogram_path = explainability_dir / "temporal_usage_histogram.png"
    if save_plots:
        if int(explainability_summary["temporal_split_usages"]) > 0:
            plot_temporal_usage(
                usage_table,
                explainability_summary,
                bin_edges,
                histogram_path,
            )
        else:
            _save_no_split_plot(
                histogram_path,
                f"{outcome.upper()} Temporal Usage",
            )

    tree_path = output_dir / "tree.json"
    experiment_1.export_tree_json(model, tree_path)
    return {
        "outcome": outcome,
        "performance": performance,
        "temporal_recovery_score": recovery_score,
        "temporal_recovery_definition": SCORING_SPECIFICATION[
            "temporal_recovery"
        ],
        "n_train_patients": int(len(train_order)),
        "n_evaluation_patients": int(len(evaluation_order)),
        "train_positive_prevalence": float(y_train.mean()),
        "evaluation_positive_prevalence": float(y_evaluation.mean()),
        "runtime_seconds": {
            "training": training_seconds,
            "evaluation": evaluation_seconds,
            "explainability": explainability_seconds,
            "total": perf_counter() - outcome_start,
        },
        "candidate_evaluations": experiment_1.candidate_evaluation_summary(
            model
        ),
        "root_split": experiment_1.root_split_summary(model),
        "temporal_usage": usage_summary,
        "window_distribution": window_distribution.to_dict(orient="records"),
        "artifacts": {
            "tree_json": str(tree_path),
            "predictions": str(output_dir / "predictions.csv"),
            "path_table": str(
                explainability_dir / "temporal_path_usage.csv"
            ),
            "aggregated_usage": str(
                explainability_dir / "aggregated_temporal_usage.csv"
            ),
            "window_distribution": str(
                explainability_dir / "temporal_window_distribution.csv"
            ),
            "histogram": str(histogram_path) if save_plots else None,
        },
    }


def _completed_result(run_dir: Path, expected_fingerprint: str) -> Optional[dict]:
    """Load a verified completed run or reject unsafe partial reuse."""
    status_path = run_dir / "status.json"
    result_path = run_dir / "result.json"
    if not run_dir.exists():
        return None
    if not status_path.is_file():
        raise RuntimeError(
            f"Existing run directory has no status file: {run_dir}"
        )
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("configuration_fingerprint") != expected_fingerprint:
        raise RuntimeError(f"Configuration mismatch in existing run: {run_dir}")
    if status.get("status") != "complete" or not result_path.is_file():
        raise RuntimeError(
            f"Run is {status.get('status', 'incomplete')!r}; it will not be "
            f"silently overwritten: {run_dir}"
        )
    print(f"[RESUME] Skipping completed run: {run_dir}")
    return json.loads(result_path.read_text(encoding="utf-8"))


def run_pair(
    candidate: SearchCandidate,
    split_sampling_seed: int,
    stage: str,
    evaluation_split: str,
    data: pd.DataFrame,
    labels: Mapping[str, pd.DataFrame],
    train_ids: np.ndarray,
    evaluation_ids: np.ndarray,
    run_dir: Path,
    save_plots: bool = False,
) -> dict:
    """Run one paired FA/GI configuration with resumable status markers."""
    run_configuration = {
        "candidate": candidate.to_dict(),
        "split_sampling_seed": int(split_sampling_seed),
        "stage": stage,
        "evaluation_split": evaluation_split,
        "train_patient_count": int(len(train_ids)),
        "evaluation_patient_count": int(len(evaluation_ids)),
    }
    fingerprint = configuration_fingerprint(run_configuration)
    completed = _completed_result(run_dir, fingerprint)
    if completed is not None:
        return completed

    run_dir.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(exist_ok=False)
    status_path = run_dir / "status.json"
    experiment_1.write_json(
        {
            "status": "running",
            "configuration_fingerprint": fingerprint,
            "configuration": run_configuration,
        },
        status_path,
    )
    started = perf_counter()
    try:
        outcomes = {
            outcome: run_outcome(
                candidate=candidate,
                split_sampling_seed=split_sampling_seed,
                outcome=outcome,
                data=data,
                labels=labels[outcome],
                train_ids=train_ids,
                evaluation_ids=evaluation_ids,
                output_dir=run_dir / outcome,
                save_plots=save_plots,
            )
            for outcome in ("fa", "gi")
        }
        result = add_combined_score({
            **run_configuration,
            "configuration_fingerprint": fingerprint,
            "outcomes": outcomes,
            "pair_runtime_seconds": perf_counter() - started,
        })
        experiment_1.write_json(result, run_dir / "result.json")
        experiment_1.write_json(
            {
                "status": "complete",
                "configuration_fingerprint": fingerprint,
                "result_path": str(run_dir / "result.json"),
            },
            status_path,
        )
        return result
    except Exception as error:
        experiment_1.write_json(
            {
                "status": "failed",
                "configuration_fingerprint": fingerprint,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
            status_path,
        )
        raise


def select_finalist_ids(screening_results: Sequence[dict]) -> List[str]:
    """Select top predictive-temporal candidates plus a temporal safeguard."""
    if len(screening_results) < N_FINALISTS:
        raise ValueError("Not enough completed screening candidates.")
    ordered = sorted(
        screening_results,
        key=lambda result: (
            -result["combined_score"],
            result["candidate"]["candidate_id"],
        ),
    )
    selected = [
        result["candidate"]["candidate_id"] for result in ordered[:3]
    ]
    temporal_best = max(
        screening_results,
        key=lambda result: (
            np.mean([
                result["outcomes"][outcome]["temporal_recovery_score"]
                for outcome in ("fa", "gi")
            ]),
            result["combined_score"],
        ),
    )["candidate"]["candidate_id"]
    if temporal_best not in selected:
        selected.append(temporal_best)
    for result in ordered:
        candidate_id = result["candidate"]["candidate_id"]
        if candidate_id not in selected:
            selected.append(candidate_id)
        if len(selected) == N_FINALISTS:
            break
    return selected


def _sample_standard_deviation(values: Sequence[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def summarize_candidate_runs(results: Sequence[dict]) -> List[dict]:
    """Aggregate validation runs by configuration and add stability ranking."""
    grouped: Dict[str, List[dict]] = {}
    for result in results:
        candidate_id = result["candidate"]["candidate_id"]
        grouped.setdefault(candidate_id, []).append(result)

    summaries = []
    for candidate_id, candidate_results in grouped.items():
        component_means = {
            f"{name}_mean": float(np.mean([
                result["score_components"][name]
                for result in candidate_results
            ]))
            for name in COMPONENT_NAMES
        }
        combined_values = [
            result["combined_score"] for result in candidate_results
        ]
        raw_means = {}
        raw_sds = {}
        for outcome in ("fa", "gi"):
            for metric in ("roc_auc", "average_precision"):
                values = [
                    result["outcomes"][outcome]["performance"][metric]
                    for result in candidate_results
                ]
                raw_means[f"{outcome}_{metric}_mean"] = float(np.mean(values))
                raw_sds[f"{outcome}_{metric}_sd"] = _sample_standard_deviation(
                    values
                )
            temporal_values = [
                result["outcomes"][outcome]["temporal_recovery_score"]
                for result in candidate_results
            ]
            raw_means[f"{outcome}_temporal_recovery_mean"] = float(
                np.mean(temporal_values)
            )
            raw_sds[f"{outcome}_temporal_recovery_sd"] = (
                _sample_standard_deviation(temporal_values)
            )
            for usage_name in (
                "birth_date_usage_proportion",
                "now_usage_proportion",
                "full_timeline_usage_proportion",
            ):
                usage_values = [
                    result["outcomes"][outcome]["temporal_usage"][usage_name]
                    for result in candidate_results
                ]
                raw_means[f"{outcome}_{usage_name}_mean"] = float(
                    np.mean(usage_values)
                )
                raw_sds[f"{outcome}_{usage_name}_sd"] = (
                    _sample_standard_deviation(usage_values)
                )

            window_maps = [
                {
                    row["window"]: float(row["usage_proportion"])
                    for row in result["outcomes"][outcome][
                        "window_distribution"
                    ]
                }
                for result in candidate_results
            ]
            windows = sorted({
                window for window_map in window_maps for window in window_map
            })
            raw_means[f"{outcome}_window_usage_stability"] = [
                {
                    "window": window,
                    "mean_usage_proportion": float(np.mean([
                        window_map.get(window, 0.0)
                        for window_map in window_maps
                    ])),
                    "sd_usage_proportion": _sample_standard_deviation([
                        window_map.get(window, 0.0)
                        for window_map in window_maps
                    ]),
                }
                for window in windows
            ]
            root_signatures = [
                json.dumps(
                    result["outcomes"][outcome]["root_split"],
                    sort_keys=True,
                )
                for result in candidate_results
            ]
            raw_means[f"{outcome}_root_stability"] = [
                {
                    "root_split": json.loads(signature),
                    "count": root_signatures.count(signature),
                    "proportion": root_signatures.count(signature)
                    / len(root_signatures),
                }
                for signature in sorted(set(root_signatures))
            ]
        combined_mean = float(np.mean(combined_values))
        combined_sd = _sample_standard_deviation(combined_values)
        summaries.append({
            "candidate_id": candidate_id,
            "candidate": candidate_results[0]["candidate"],
            "seeds": sorted(
                int(result["split_sampling_seed"])
                for result in candidate_results
            ),
            "n_seeds": len(candidate_results),
            **raw_means,
            **raw_sds,
            **component_means,
            "combined_score_mean": combined_mean,
            "combined_score_sd": combined_sd,
            "stability_adjusted_score": combined_mean - combined_sd,
            "fa_outcome_score_mean": float(np.mean([
                component_means["fa_auc_skill_mean"],
                component_means["fa_average_precision_skill_mean"],
                component_means["fa_temporal_recovery_score_mean"],
            ])),
            "gi_outcome_score_mean": float(np.mean([
                component_means["gi_auc_skill_mean"],
                component_means["gi_average_precision_skill_mean"],
                component_means["gi_temporal_recovery_score_mean"],
            ])),
        })

    _add_pareto_status(summaries)
    summaries.sort(key=lambda summary: (
        -summary["stability_adjusted_score"],
        -summary["combined_score_mean"],
        summary["candidate_id"],
    ))
    for rank, summary in enumerate(summaries, start=1):
        summary["rank"] = rank
    return summaries


def _add_pareto_status(summaries: Sequence[dict]) -> None:
    """Mark non-dominated summaries across the six normalized components."""
    for candidate in summaries:
        values = np.asarray([
            candidate[f"{name}_mean"] for name in COMPONENT_NAMES
        ])
        dominated = False
        for other in summaries:
            if other is candidate:
                continue
            other_values = np.asarray([
                other[f"{name}_mean"] for name in COMPONENT_NAMES
            ])
            if np.all(other_values >= values) and np.any(other_values > values):
                dominated = True
                break
        candidate["pareto_optimal"] = not dominated


def representative_seed(candidate_results: Sequence[dict]) -> int:
    """Choose the seed closest to median validation performance."""
    if not candidate_results:
        raise ValueError("At least one candidate result is required.")
    median = float(np.median([
        result["combined_score"] for result in candidate_results
    ]))
    selected = min(
        candidate_results,
        key=lambda result: (
            abs(result["combined_score"] - median),
            int(result["split_sampling_seed"]),
        ),
    )
    return int(selected["split_sampling_seed"])


def _flatten_result(result: dict, rank_by_candidate: Mapping[str, int]) -> dict:
    candidate = result["candidate"]
    row = {
        "candidate_id": candidate["candidate_id"],
        "stage": result["stage"],
        "evaluation_split": result["evaluation_split"],
        "split_sampling_seed": result["split_sampling_seed"],
        "candidate_sampling_rate": candidate["candidate_sampling_rate"],
        "max_depth": candidate["max_depth"],
        "min_samples_split": candidate["min_samples_split"],
        "min_samples_leaf": candidate["min_samples_leaf"],
        "min_impurity_decrease": candidate["min_impurity_decrease"],
        "temporal_search_space": candidate["temporal_search_space"],
        "anchor_offset_map_json": json.dumps(candidate["anchor_offset_map"]),
        "aggregations_json": json.dumps(candidate["aggregations"]),
        "use_dynamic_anchors": candidate["use_dynamic_anchors"],
        "pair_runtime_seconds": result["pair_runtime_seconds"],
        "combined_score": result["combined_score"],
        "rank": rank_by_candidate.get(candidate["candidate_id"]),
    }
    for outcome in ("fa", "gi"):
        outcome_result = result["outcomes"][outcome]
        usage = outcome_result["temporal_usage"]
        row.update({
            f"{outcome}_roc_auc": outcome_result["performance"]["roc_auc"],
            f"{outcome}_average_precision": outcome_result["performance"][
                "average_precision"
            ],
            f"{outcome}_temporal_recovery_score": outcome_result[
                "temporal_recovery_score"
            ],
            f"{outcome}_birth_usage_proportion": usage[
                "birth_date_usage_proportion"
            ],
            f"{outcome}_now_usage_proportion": usage["now_usage_proportion"],
            f"{outcome}_full_timeline_usage_proportion": usage[
                "full_timeline_usage_proportion"
            ],
            f"{outcome}_window_distribution_json": json.dumps(
                outcome_result["window_distribution"]
            ),
            f"{outcome}_training_runtime_seconds": outcome_result[
                "runtime_seconds"
            ]["training"],
            f"{outcome}_root_json": json.dumps(outcome_result["root_split"]),
        })
    row.update(result["score_components"])
    return row


def write_all_results(
    output_dir: Path,
    results: Sequence[dict],
    summaries: Sequence[dict],
) -> None:
    """Write auditable CSV and nested JSON snapshots of every completed run."""
    rank_by_candidate = {
        summary["candidate_id"]: summary["rank"] for summary in summaries
    }
    rows = [_flatten_result(result, rank_by_candidate) for result in results]
    pd.DataFrame(rows).to_csv(
        output_dir / "all_candidates_results.csv", index=False
    )
    experiment_1.write_json(
        {
            "scoring": SCORING_SPECIFICATION,
            "candidate_summaries": list(summaries),
            "runs": list(results),
        },
        output_dir / "all_candidates_results.json",
    )


def _reason_for_rank(summary: Mapping) -> str:
    return (
        f"Ranked on stability-adjusted validation score "
        f"{summary['stability_adjusted_score']:.4f} "
        f"(mean {summary['combined_score_mean']:.4f}, "
        f"SD {summary['combined_score_sd']:.4f}); mean FA/GI temporal "
        f"recovery was {summary['fa_temporal_recovery_mean']:.4f}/"
        f"{summary['gi_temporal_recovery_mean']:.4f}, with mean FA/GI "
        f"ROC-AUC {summary['fa_roc_auc_mean']:.4f}/"
        f"{summary['gi_roc_auc_mean']:.4f}."
    )


def run_final_candidate(
    rank: int,
    candidate: SearchCandidate,
    validation_summary: dict,
    validation_results: Sequence[dict],
    data: pd.DataFrame,
    labels: Mapping[str, pd.DataFrame],
    outer_train_ids: np.ndarray,
    test_ids: np.ndarray,
    split_details: Mapping,
    top_directory: Path,
) -> dict:
    """Refit one selected candidate and evaluate the sealed test set once."""
    selected_seed = representative_seed(validation_results)
    candidate_dir = top_directory / f"candidate_{rank}"
    final_configuration = {
        "rank": rank,
        "candidate": candidate.to_dict(),
        "model_parameters": candidate.model_parameters(selected_seed),
        "stability_seeds": list(STABILITY_SEEDS),
        "representative_seed": selected_seed,
        "representative_seed_rule": search_plan()["representative_seed"],
        "validation_summary": validation_summary,
        "per_seed_validation_runs": [
            {
                "split_sampling_seed": result["split_sampling_seed"],
                "combined_score": result["combined_score"],
                "fa_tree_json": result["outcomes"]["fa"]["artifacts"][
                    "tree_json"
                ],
                "gi_tree_json": result["outcomes"]["gi"]["artifacts"][
                    "tree_json"
                ],
            }
            for result in sorted(
                validation_results,
                key=lambda item: item["split_sampling_seed"],
            )
        ],
        "split_details": dict(split_details),
        "ranking_does_not_use_test_metrics": True,
    }
    fingerprint = configuration_fingerprint(final_configuration)
    completed = _completed_result(candidate_dir, fingerprint)
    if completed is not None:
        return completed

    candidate_dir.parent.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(exist_ok=False)
    status_path = candidate_dir / "status.json"
    experiment_1.write_json(
        {
            "status": "running",
            "configuration_fingerprint": fingerprint,
        },
        status_path,
    )
    experiment_1.write_json(final_configuration, candidate_dir / "config.json")
    started = perf_counter()
    try:
        outcomes = {
            outcome: run_outcome(
                candidate=candidate,
                split_sampling_seed=selected_seed,
                outcome=outcome,
                data=data,
                labels=labels[outcome],
                train_ids=outer_train_ids,
                evaluation_ids=test_ids,
                output_dir=candidate_dir / outcome,
                save_plots=True,
            )
            for outcome in ("fa", "gi")
        }
        test_result = add_combined_score({
            "candidate": candidate.to_dict(),
            "split_sampling_seed": selected_seed,
            "stage": "final_test",
            "evaluation_split": "test",
            "outcomes": outcomes,
            "pair_runtime_seconds": perf_counter() - started,
        })
        comparison_path = _save_combined_usage_plot(candidate_dir)
        metrics = {
            "rank": rank,
            "ranking_basis": "validation only",
            "validation": validation_summary,
            "test": test_result,
            "combined_usage_figure": str(comparison_path),
        }
        experiment_1.write_json(metrics, candidate_dir / "metrics.json")
        summary = {
            "rank": rank,
            "candidate_id": candidate.candidate_id,
            "why_ranked_highly": _reason_for_rank(validation_summary),
            "representative_seed": selected_seed,
            "validation_stability_adjusted_score": validation_summary[
                "stability_adjusted_score"
            ],
            "test_combined_score_not_used_for_ranking": test_result[
                "combined_score"
            ],
            "artifacts": {
                "config": str(candidate_dir / "config.json"),
                "metrics": str(candidate_dir / "metrics.json"),
                "fa_tree_json": outcomes["fa"]["artifacts"]["tree_json"],
                "gi_tree_json": outcomes["gi"]["artifacts"]["tree_json"],
                "fa_temporal_usage": outcomes["fa"]["artifacts"][
                    "aggregated_usage"
                ],
                "gi_temporal_usage": outcomes["gi"]["artifacts"][
                    "aggregated_usage"
                ],
                "fa_temporal_histogram": outcomes["fa"]["artifacts"][
                    "histogram"
                ],
                "gi_temporal_histogram": outcomes["gi"]["artifacts"][
                    "histogram"
                ],
                "combined_temporal_usage": str(comparison_path),
            },
        }
        experiment_1.write_json(
            summary, candidate_dir / "candidate_summary.json"
        )
        result = {
            **test_result,
            "validation_summary": validation_summary,
            "top_rank": rank,
            "representative_seed": selected_seed,
            "configuration_fingerprint": fingerprint,
        }
        experiment_1.write_json(result, candidate_dir / "result.json")
        experiment_1.write_json(
            {
                "status": "complete",
                "configuration_fingerprint": fingerprint,
                "result_path": str(candidate_dir / "result.json"),
            },
            status_path,
        )
        return result
    except Exception as error:
        experiment_1.write_json(
            {
                "status": "failed",
                "configuration_fingerprint": fingerprint,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
            status_path,
        )
        raise


def _split_details(
    split_table: pd.DataFrame,
    split_path: Path,
    input_hashes: Mapping[str, str],
) -> dict:
    return {
        "patient_split_file": str(split_path),
        "patient_split_sha256": experiment_1.sha256_file(split_path),
        "patient_counts": {
            name: int(split_table["split"].eq(name).sum())
            for name in ("search_train", "validation", "test")
        },
        "outer_test_size": OUTER_TEST_SIZE,
        "outer_split_seed": OUTER_SPLIT_SEED,
        "validation_fraction_of_outer_train": (
            VALIDATION_FRACTION_OF_OUTER_TRAIN
        ),
        "inner_split_seed": INNER_SPLIT_SEED,
        "input_sha256": dict(input_hashes),
    }


def _prepare_output_directory(plan: Mapping) -> None:
    """Create or verify the search root without silently replacing a search."""
    plan_fingerprint = configuration_fingerprint(plan)
    plan_path = OUTPUT_DIR / "search_plan.json"
    if OUTPUT_DIR.exists():
        if not plan_path.is_file():
            raise RuntimeError(
                f"Existing output directory has no search plan: {OUTPUT_DIR}"
            )
        saved_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if configuration_fingerprint(saved_plan) != plan_fingerprint:
            raise RuntimeError(
                "Existing output directory belongs to a different search plan."
            )
        print(f"[RESUME] Verified existing search directory: {OUTPUT_DIR}")
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    experiment_1.write_json(plan, plan_path)


def _write_or_verify_split(split_table: pd.DataFrame) -> Path:
    split_path = OUTPUT_DIR / "shared_patient_splits.csv"
    if split_path.exists():
        saved = pd.read_csv(split_path)
        pd.testing.assert_frame_equal(
            saved.reset_index(drop=True),
            split_table.reset_index(drop=True),
            check_dtype=False,
        )
    else:
        split_table.to_csv(split_path, index=False)
    return split_path


def run_search() -> dict:
    """Execute the bounded search; this is intentionally not run on import."""
    validate_search_plan(SCREENING_CANDIDATES)
    for name, path in INPUT_PATHS.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing V2 {name} artifact: {path}")
    hashes_before = {
        name: experiment_1.sha256_file(path)
        for name, path in INPUT_PATHS.items()
    }
    plan = {**search_plan(), "input_sha256": hashes_before}
    _prepare_output_directory(plan)
    search_started = perf_counter()
    data, labels = load_v2_inputs()
    split_table = create_shared_search_splits(data, labels)
    split_path = _write_or_verify_split(split_table)
    split_details = _split_details(split_table, split_path, hashes_before)
    experiment_1.write_json(
        split_details, OUTPUT_DIR / "split_details.json"
    )

    search_train_ids = split_patient_ids(split_table, ["search_train"])
    validation_ids = split_patient_ids(split_table, ["validation"])
    outer_train_ids = split_patient_ids(
        split_table, ["search_train", "validation"]
    )
    test_ids = split_patient_ids(split_table, ["test"])
    candidate_by_id = {
        candidate.candidate_id: candidate
        for candidate in SCREENING_CANDIDATES
    }

    validation_results = []
    for candidate in SCREENING_CANDIDATES:
        print(f"[SCREEN] {candidate.candidate_id} seed={SCREENING_SEED}")
        validation_results.append(run_pair(
            candidate=candidate,
            split_sampling_seed=SCREENING_SEED,
            stage="screening",
            evaluation_split="validation",
            data=data,
            labels=labels,
            train_ids=search_train_ids,
            evaluation_ids=validation_ids,
            run_dir=(
                OUTPUT_DIR / "validation_runs" / candidate.candidate_id
                / f"seed_{SCREENING_SEED}"
            ),
        ))
        partial_summaries = summarize_candidate_runs(validation_results)
        write_all_results(OUTPUT_DIR, validation_results, partial_summaries)

    finalist_ids = select_finalist_ids(validation_results)
    experiment_1.write_json(
        {
            "finalist_ids": finalist_ids,
            "selection_rule": search_plan()["finalist_selection"],
        },
        OUTPUT_DIR / "selected_finalists.json",
    )
    for candidate_id in finalist_ids:
        candidate = candidate_by_id[candidate_id]
        for seed in STABILITY_SEEDS:
            if seed == SCREENING_SEED:
                continue
            print(f"[STABILITY] {candidate_id} seed={seed}")
            validation_results.append(run_pair(
                candidate=candidate,
                split_sampling_seed=seed,
                stage="stability",
                evaluation_split="validation",
                data=data,
                labels=labels,
                train_ids=search_train_ids,
                evaluation_ids=validation_ids,
                run_dir=(
                    OUTPUT_DIR / "validation_runs" / candidate_id
                    / f"seed_{seed}"
                ),
            ))
            partial_summaries = summarize_candidate_runs(validation_results)
            write_all_results(
                OUTPUT_DIR, validation_results, partial_summaries
            )

    finalist_results = [
        result for result in validation_results
        if result["candidate"]["candidate_id"] in finalist_ids
    ]
    finalist_summaries = summarize_candidate_runs(finalist_results)
    top_summaries = finalist_summaries[:N_TOP_CANDIDATES]
    final_test_results = []
    for rank, validation_summary in enumerate(top_summaries, start=1):
        candidate_id = validation_summary["candidate_id"]
        candidate_results = [
            result for result in finalist_results
            if result["candidate"]["candidate_id"] == candidate_id
        ]
        print(f"[FINAL TEST] rank={rank} candidate={candidate_id}")
        final_test_results.append(run_final_candidate(
            rank=rank,
            candidate=candidate_by_id[candidate_id],
            validation_summary=validation_summary,
            validation_results=candidate_results,
            data=data,
            labels=labels,
            outer_train_ids=outer_train_ids,
            test_ids=test_ids,
            split_details=split_details,
            top_directory=OUTPUT_DIR / "top_3_candidates",
        ))

    all_results = validation_results + final_test_results
    all_summaries = summarize_candidate_runs(validation_results)
    write_all_results(OUTPUT_DIR, all_results, all_summaries)

    best_fa = max(
        finalist_summaries, key=lambda item: item["fa_outcome_score_mean"]
    )
    best_gi = max(
        finalist_summaries, key=lambda item: item["gi_outcome_score_mean"]
    )
    most_stable = min(
        finalist_summaries,
        key=lambda item: (item["combined_score_sd"], item["candidate_id"]),
    )
    hashes_after = {
        name: experiment_1.sha256_file(path)
        for name, path in INPUT_PATHS.items()
    }
    if hashes_after != hashes_before:
        raise RuntimeError("An immutable V2 input changed during the search.")

    summary = {
        "status": "complete",
        "number_of_configurations_tested": len(SCREENING_CANDIDATES),
        "number_of_validation_model_pairs": len(validation_results),
        "number_of_final_test_model_pairs": len(final_test_results),
        "total_model_pairs": len(all_results),
        "total_runtime_seconds": perf_counter() - search_started,
        "search_strategy": search_plan(),
        "ranking_method": SCORING_SPECIFICATION,
        "top_3_configurations": [
            {
                **validation_summary,
                "why_ranked_highly": _reason_for_rank(validation_summary),
                "final_artifact_directory": str(
                    OUTPUT_DIR / "top_3_candidates" / f"candidate_{rank}"
                ),
            }
            for rank, validation_summary in enumerate(top_summaries, start=1)
        ],
        "best_fa_candidate": best_fa["candidate_id"],
        "best_gi_candidate": best_gi["candidate_id"],
        "best_balanced_fa_gi_candidate": top_summaries[0]["candidate_id"],
        "most_stable_candidate": most_stable["candidate_id"],
        "performance_temporal_tradeoff": {
            summary["candidate_id"]: {
                "fa_predictive_score": float(np.mean([
                    summary["fa_auc_skill_mean"],
                    summary["fa_average_precision_skill_mean"],
                ])),
                "fa_temporal_recovery": summary[
                    "fa_temporal_recovery_mean"
                ],
                "gi_predictive_score": float(np.mean([
                    summary["gi_auc_skill_mean"],
                    summary["gi_average_precision_skill_mean"],
                ])),
                "gi_temporal_recovery": summary[
                    "gi_temporal_recovery_mean"
                ],
            }
            for summary in finalist_summaries
        },
        "input_sha256_before_and_after": {
            "before": hashes_before,
            "after": hashes_after,
            "unchanged": True,
        },
    }
    experiment_1.write_json(
        summary, OUTPUT_DIR / "hyperparameter_search_summary.json"
    )
    pd.DataFrame(summary["top_3_configurations"]).to_csv(
        OUTPUT_DIR / "hyperparameter_search_summary.csv", index=False
    )
    print("\nSearch complete")
    print(f"Top candidates: {[item['candidate_id'] for item in top_summaries]}")
    print(f"Results: {OUTPUT_DIR}")
    return summary


def validate_only() -> dict:
    """Validate inputs and splits without training or writing search outputs."""
    validate_search_plan(SCREENING_CANDIDATES)
    hashes = {
        name: experiment_1.sha256_file(path)
        for name, path in INPUT_PATHS.items()
    }
    data, labels = load_v2_inputs()
    split_table = create_shared_search_splits(data, labels)
    return {
        "status": "validated_without_training",
        "input_sha256": hashes,
        "patient_counts": split_table["split"].value_counts().to_dict(),
        "search_plan": search_plan(),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print the predeclared plan without loading data or training.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate V2 inputs and patient splits without training or writing.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.plan_only:
        validate_search_plan(SCREENING_CANDIDATES)
        print(json.dumps(search_plan(), indent=2, sort_keys=True))
        return
    if args.validate_only:
        print(json.dumps(validate_only(), indent=2, sort_keys=True))
        return
    run_search()


if __name__ == "__main__":
    main()
