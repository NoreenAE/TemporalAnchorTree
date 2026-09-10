"""Reconstruct V2 split selection without fitting a TemporalTree.

This diagnostic is intentionally limited to Steps 1--4 of the approved
Experiment 1 investigation.  It loads immutable inputs and completed search
artifacts, reconstructs node populations, replays feature-window-aggregation
sampling, and independently scores the original threshold candidates.

The script never calls ``TemporalTree.fit`` and writes only to the isolated
``experiments/results/v2_split_diagnostics`` directory.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from experiments import experiment_1_fa_vs_gi as experiment_1
from experiments import search_experiment_1_v2 as search
from settree.temporal_tree import TemporalTree


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SEARCH_RESULTS = (
    REPOSITORY_ROOT / "experiments" / "results" / "v2_hyperparameter_search"
)
OUTPUT_DIR = (
    REPOSITORY_ROOT / "experiments" / "results" / "v2_split_diagnostics"
)
AGGREGATIONS = ("mean", "sum", "max", "count")
ABSOLUTE_NEAR_TIE = 1e-4
SECONDARY_ABSOLUTE_NEAR_TIE = 5e-4
RELATIVE_NEAR_TIE_FRACTION = 0.05
FLOAT_TOLERANCE = 1e-12


@dataclass(frozen=True)
class NodeSpec:
    """One preregistered saved node to audit."""

    audit_id: str
    result_path: str
    outcome: str
    node_id: int
    role: str


NODE_SPECS = (
    NodeSpec(
        "c02_validation_seed44_fa_root",
        "validation_runs/C02/seed_44/result.json",
        "fa",
        0,
        "problematic",
    ),
    NodeSpec(
        "c02_final_fa_root_control",
        "top_3_candidates/candidate_1/result.json",
        "fa",
        0,
        "positive_control",
    ),
    NodeSpec(
        "c02_final_gi_root_control",
        "top_3_candidates/candidate_1/result.json",
        "gi",
        0,
        "positive_control",
    ),
    NodeSpec(
        "c02_final_fa_node1_n992",
        "top_3_candidates/candidate_1/result.json",
        "fa",
        1,
        "problematic",
    ),
    NodeSpec(
        "c02_final_gi_node8_n801",
        "top_3_candidates/candidate_1/result.json",
        "gi",
        8,
        "problematic",
    ),
    NodeSpec(
        "c01_final_gi_node2_n1935",
        "top_3_candidates/candidate_2/result.json",
        "gi",
        2,
        "regularization_comparison",
    ),
    NodeSpec(
        "c02_final_gi_node2_n1935",
        "top_3_candidates/candidate_1/result.json",
        "gi",
        2,
        "regularization_comparison",
    ),
    NodeSpec(
        "c02_final_gi_node12_n86",
        "top_3_candidates/candidate_1/result.json",
        "gi",
        12,
        "problematic",
    ),
    NodeSpec(
        "c02_final_fa_node9_n1673",
        "top_3_candidates/candidate_1/result.json",
        "fa",
        9,
        "problematic",
    ),
)


@dataclass
class NodePopulation:
    """Longitudinal data and aligned labels reaching one saved node."""

    data: pd.DataFrame
    labels: np.ndarray
    patient_ids: np.ndarray


@dataclass
class GroupEvaluation:
    """Threshold audit for one feature-window-aggregation group."""

    thresholds: np.ndarray
    valid_mask: np.ndarray
    scores: np.ndarray
    gains: np.ndarray
    left_sizes: np.ndarray
    right_sizes: np.ndarray
    best_index: Optional[int]

    @property
    def valid_count(self) -> int:
        return int(self.valid_mask.sum())

    @property
    def invalid_count(self) -> int:
        return int(len(self.thresholds) - self.valid_count)


@dataclass
class RunContext:
    """Reconstructed state for one saved outcome model."""

    result_path: Path
    result: dict
    candidate: dict
    outcome: str
    tree_path: Path
    nodes: list
    node_by_id: dict
    model: TemporalTree
    training: Dict[int, NodePopulation]
    held_out: Dict[int, NodePopulation]
    train_total: int
    held_out_total: int
    sampling_by_node: Dict[int, list]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mse(labels: np.ndarray) -> float:
    if len(labels) == 0:
        return 0.0
    mean = float(np.mean(labels))
    return mean * (1.0 - mean)


def _json_numbers(values: Iterable) -> str:
    return json.dumps([float(value) for value in values], separators=(",", ":"))


def _offset_equal(first, second) -> bool:
    if first is None or second is None:
        return first is None and second is None
    return bool(np.isclose(float(first), float(second), atol=0.0, rtol=0.0))


def _snap_saved_threshold_to_observed_value(
    values: np.ndarray,
    saved_threshold: float,
) -> float:
    """Resolve sub-ULP JSON/environment drift at observed split boundaries.

    Candidate thresholds originate from observed aggregate values.  One saved
    V2 threshold differs from its recomputed observed value by roughly
    1.8e-15 (9.524999999999999 versus 9.525), which changes one ``<=`` branch
    if the serialized float is applied literally.  Snapping only when the
    nearest observed value is within the reconstruction tolerance restores
    the original candidate boundary; larger differences remain untouched and
    will fail the saved child-size assertion.
    """
    observed = np.unique(values[~pd.isna(values)].astype(float))
    if len(observed) == 0:
        return float(saved_threshold)
    nearest = float(observed[np.argmin(np.abs(observed - saved_threshold))])
    if abs(nearest - float(saved_threshold)) <= FLOAT_TOLERANCE:
        return nearest
    return float(saved_threshold)


def _build_model(candidate: Mapping, split_sampling_seed: int) -> TemporalTree:
    """Create an unfitted model solely to reuse exact aggregation semantics."""
    return TemporalTree(
        anchors_list=["Birth Date Anchor", "NOW Anchor"],
        anchor_offset_list=[None],
        anchor_offset_map=candidate["anchor_offset_map"],
        aggregation_funcs=experiment_1.aggregation_functions(),
        candidate_sampling_rate=candidate["candidate_sampling_rate"],
        criterion="mse",
        is_classification=False,
        learn_missing_direction=False,
        max_depth=candidate["max_depth"],
        min_impurity_decrease=candidate["min_impurity_decrease"],
        min_samples_leaf=candidate["min_samples_leaf"],
        min_samples_split=candidate["min_samples_split"],
        random_state=42,
        split_sampling_seed=split_sampling_seed,
        use_dynamic_anchors=False,
        verbose=0,
    )


def _route_saved_tree(
    model: TemporalTree,
    nodes: Sequence[dict],
    data: pd.DataFrame,
    labels: pd.Series,
) -> Dict[int, NodePopulation]:
    """Route a cohort through saved splits using TemporalTree aggregation."""
    node_by_id = {int(node["node_id"]): node for node in nodes}
    patient_ids = data["patient_id"].drop_duplicates().to_numpy()
    aligned_labels = labels.to_numpy(dtype=float)
    if len(patient_ids) != len(aligned_labels):
        raise RuntimeError("Patient and label order are not aligned.")

    populations = {
        0: NodePopulation(data, aligned_labels, patient_ids),
    }
    queue = deque([0])
    while queue:
        node_id = queue.popleft()
        node = node_by_id[node_id]
        if node["type"] == "leaf":
            continue
        population = populations[node_id]
        observed_order = population.data["patient_id"].drop_duplicates().to_numpy()
        if not np.array_equal(observed_order, population.patient_ids):
            raise RuntimeError(f"Patient order changed at node {node_id}.")
        values, _ = model._aggregate_feature_over_time(
            population.data,
            node["feature"],
            (node["anchor"], node["offset"]),
            node["aggregation"],
        )
        values = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy()
        missing = pd.isna(values)
        effective_threshold = _snap_saved_threshold_to_observed_value(
            values,
            float(node["threshold"]),
        )
        with np.errstate(invalid="ignore"):
            left_mask = (~missing) & (values <= effective_threshold)
        right_mask = ~left_mask
        for child_key, mask in (("left_child", left_mask), ("right_child", right_mask)):
            child_id = int(node[child_key])
            child_ids = population.patient_ids[mask]
            child_data = population.data.loc[
                population.data["patient_id"].isin(child_ids)
            ].copy()
            child_labels = population.labels[mask]
            child_order = child_data["patient_id"].drop_duplicates().to_numpy()
            if not np.array_equal(child_order, child_ids):
                raise RuntimeError(f"Child order changed at node {child_id}.")
            populations[child_id] = NodePopulation(
                child_data,
                child_labels,
                child_ids,
            )
            queue.append(child_id)
    return populations


def _time_windows(candidate: Mapping) -> list:
    windows = []
    full_seen = False
    for anchor in ("Birth Date Anchor", "NOW Anchor"):
        for offset in candidate["anchor_offset_map"][anchor]:
            if offset is None:
                if full_seen:
                    continue
                windows.append((None, None))
                full_seen = True
            else:
                windows.append((anchor, offset))
    return windows


def _candidate_groups(population: NodePopulation, candidate: Mapping) -> list:
    groups = []
    generation_index = 0
    for feature in population.data["feature_name"].unique():
        if feature == "Birth Date" or str(feature).endswith("Anchor"):
            continue
        for anchor, offset in _time_windows(candidate):
            for aggregation in AGGREGATIONS:
                groups.append({
                    "generation_index": generation_index,
                    "feature": feature,
                    "anchor": anchor,
                    "offset": offset,
                    "aggregation": aggregation,
                })
                generation_index += 1
    return groups


def _internal_bfs_order(nodes: Sequence[dict]) -> list:
    node_by_id = {int(node["node_id"]): node for node in nodes}
    order = []
    queue = deque([0])
    while queue:
        node_id = queue.popleft()
        node = node_by_id[node_id]
        if node["type"] == "leaf":
            continue
        order.append(node_id)
        queue.append(int(node["left_child"]))
        queue.append(int(node["right_child"]))
    return order


def _replay_sampling(
    result: Mapping,
    outcome: str,
    nodes: Sequence[dict],
    populations: Mapping[int, NodePopulation],
) -> Dict[int, list]:
    candidate = result["candidate"]
    rate = float(candidate["candidate_sampling_rate"])
    rng = np.random.default_rng(int(result["split_sampling_seed"]))
    replay = {}
    total_generated = 0
    total_sampled = 0
    for node_id in _internal_bfs_order(nodes):
        node_groups = []
        for group in _candidate_groups(populations[node_id], candidate):
            draw = None if rate == 1.0 else float(rng.random())
            sampled = True if draw is None else draw < rate
            group = dict(group)
            group["sampling_draw"] = draw
            group["sampled"] = sampled
            node_groups.append(group)
            total_generated += 1
            total_sampled += int(sampled)
        replay[node_id] = node_groups

    saved = result["outcomes"][outcome]["candidate_evaluations"]
    if total_generated != int(saved["generated_groups"]):
        raise RuntimeError(
            f"Generated-group replay mismatch: {total_generated} != "
            f"{saved['generated_groups']}"
        )
    if total_sampled != int(saved["sampled_groups"]):
        raise RuntimeError(
            f"Sampled-group replay mismatch: {total_sampled} != "
            f"{saved['sampled_groups']}"
        )
    if int(saved["fallback_evaluated_threshold_candidates"]) != 0:
        raise RuntimeError("Targeted run used fallback sampling; replay is incomplete.")
    return replay


def _result_tree_path(result_path: Path, result: Mapping, outcome: str) -> Path:
    saved = Path(result["outcomes"][outcome]["artifacts"]["tree_json"])
    if saved.is_file():
        return saved
    return result_path.parent / outcome / "tree.json"


def _load_context(
    result_path: Path,
    outcome: str,
    data: pd.DataFrame,
    labels_by_outcome: Mapping[str, pd.DataFrame],
    split_table: pd.DataFrame,
) -> RunContext:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    candidate = result["candidate"]
    model = _build_model(candidate, int(result["split_sampling_seed"]))
    tree_path = _result_tree_path(result_path, result, outcome)
    nodes = json.loads(tree_path.read_text(encoding="utf-8"))
    node_by_id = {int(node["node_id"]): node for node in nodes}

    if result["evaluation_split"] == "validation":
        train_splits = ("search_train",)
        held_out_splits = ("validation",)
    elif result["evaluation_split"] == "test":
        train_splits = ("search_train", "validation")
        held_out_splits = ("test",)
    else:
        raise ValueError(f"Unsupported evaluation split: {result['evaluation_split']}")

    train_ids = search.split_patient_ids(split_table, train_splits)
    held_out_ids = search.split_patient_ids(split_table, held_out_splits)
    train_data, train_labels, _ = experiment_1.subset_for_patients(
        data, labels_by_outcome[outcome], train_ids
    )
    held_out_data, held_out_labels, _ = experiment_1.subset_for_patients(
        data, labels_by_outcome[outcome], held_out_ids
    )
    training = _route_saved_tree(model, nodes, train_data, train_labels)
    held_out = _route_saved_tree(model, nodes, held_out_data, held_out_labels)

    for node_id, population in training.items():
        saved_n = int(node_by_id[node_id]["num_patients"])
        if len(population.patient_ids) != saved_n:
            raise RuntimeError(
                f"Saved node size mismatch at node {node_id}: "
                f"{len(population.patient_ids)} != {saved_n}"
            )

    sampling = _replay_sampling(result, outcome, nodes, training)
    return RunContext(
        result_path=result_path,
        result=result,
        candidate=candidate,
        outcome=outcome,
        tree_path=tree_path,
        nodes=nodes,
        node_by_id=node_by_id,
        model=model,
        training=training,
        held_out=held_out,
        train_total=len(training[0].patient_ids),
        held_out_total=len(held_out[0].patient_ids),
        sampling_by_node=sampling,
    )


def _evaluate_values(
    values: np.ndarray,
    labels: np.ndarray,
    min_samples_leaf: int,
    parent_mse: float,
) -> GroupEvaluation:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy()
    non_missing = ~pd.isna(numeric)
    observed_values = numeric[non_missing].astype(float)
    observed_labels = labels[non_missing].astype(float)
    thresholds = np.unique(observed_values)
    if len(thresholds) == 0:
        empty_float = np.asarray([], dtype=float)
        empty_int = np.asarray([], dtype=int)
        return GroupEvaluation(
            empty_float,
            np.asarray([], dtype=bool),
            empty_float,
            empty_float,
            empty_int,
            empty_int,
            None,
        )

    order = np.argsort(observed_values, kind="stable")
    sorted_values = observed_values[order]
    sorted_labels = observed_labels[order]
    cumulative_positives = np.cumsum(sorted_labels)
    left_sizes = np.searchsorted(sorted_values, thresholds, side="right")
    left_positives = cumulative_positives[left_sizes - 1]
    total_size = len(labels)
    total_positives = float(np.sum(labels))
    right_sizes = total_size - left_sizes
    right_positives = total_positives - left_positives
    valid = (
        (left_sizes >= min_samples_leaf)
        & (right_sizes >= min_samples_leaf)
    )

    scores = np.full(len(thresholds), np.nan, dtype=float)
    if np.any(valid):
        left_p = left_positives[valid] / left_sizes[valid]
        right_p = right_positives[valid] / right_sizes[valid]
        scores[valid] = (
            left_sizes[valid] * left_p * (1.0 - left_p)
            + right_sizes[valid] * right_p * (1.0 - right_p)
        ) / total_size
    gains = parent_mse - scores
    valid_indices = np.flatnonzero(valid)
    best_index = None
    if len(valid_indices):
        best_index = int(valid_indices[np.argmin(scores[valid_indices])])
    return GroupEvaluation(
        thresholds=thresholds,
        valid_mask=valid,
        scores=scores,
        gains=gains,
        left_sizes=left_sizes.astype(int),
        right_sizes=right_sizes.astype(int),
        best_index=best_index,
    )


def _evaluate_group(
    context: RunContext,
    population: NodePopulation,
    group: Mapping,
    caches: Mapping[str, dict],
    parent_mse: float,
) -> GroupEvaluation:
    values, _ = context.model._aggregate_feature_over_time(
        population.data,
        group["feature"],
        (group["anchor"], group["offset"]),
        group["aggregation"],
        anchor_reference_cache=caches["anchor_reference"],
        feature_subset_cache=caches["feature_subset"],
        feature_anchor_merge_cache=caches["feature_anchor_merge"],
        time_window_mask_cache=caches["time_window_mask"],
        filtered_numeric_cache=caches["filtered_numeric"],
        aggregation_result_cache=caches["aggregation_result"],
    )
    return _evaluate_values(
        values,
        population.labels,
        int(context.candidate["min_samples_leaf"]),
        parent_mse,
    )


def _new_caches() -> dict:
    return {
        "anchor_reference": {},
        "feature_subset": {},
        "feature_anchor_merge": {},
        "time_window_mask": {},
        "filtered_numeric": {},
        "aggregation_result": {},
    }


def _group_signature(group: Mapping) -> Tuple:
    return (
        group["feature"],
        group["anchor"],
        group["offset"],
        group["aggregation"],
    )


def _expected_groups(outcome: str) -> list:
    if outcome == "fa":
        feature_windows = (
            ("FormulaYN", "Birth Date Anchor", 6),
            ("BreastMilkYN", "Birth Date Anchor", 6),
        )
    elif outcome == "gi":
        feature_windows = (
            ("VomitingYN", "NOW Anchor", -1),
            ("DiarrheaYN", "NOW Anchor", -1),
            ("GISymptomYN", "NOW Anchor", -1),
        )
    else:
        raise ValueError(outcome)
    return [
        {
            "feature": feature,
            "anchor": anchor,
            "offset": offset,
            "aggregation": aggregation,
        }
        for feature, anchor, offset in feature_windows
        for aggregation in AGGREGATIONS
    ]


def _best_fields(evaluation: GroupEvaluation) -> dict:
    if evaluation.best_index is None:
        return {
            "best_valid_threshold": None,
            "best_left_size": None,
            "best_right_size": None,
            "best_score": None,
            "best_gain": None,
        }
    index = evaluation.best_index
    return {
        "best_valid_threshold": float(evaluation.thresholds[index]),
        "best_left_size": int(evaluation.left_sizes[index]),
        "best_right_size": int(evaluation.right_sizes[index]),
        "best_score": float(evaluation.scores[index]),
        "best_gain": float(evaluation.gains[index]),
    }


def _audit_node(spec: NodeSpec, context: RunContext) -> Tuple[dict, list, list]:
    node = context.node_by_id[spec.node_id]
    if node["type"] != "split":
        raise ValueError(f"Target node {spec.audit_id} is not a split.")
    population = context.training[spec.node_id]
    held_out_population = context.held_out[spec.node_id]
    parent_mse = _mse(population.labels)
    parent_prevalence = float(np.mean(population.labels))
    caches = _new_caches()
    groups = context.sampling_by_node[spec.node_id]

    ranking_rows = []
    evaluations = {}
    threshold_gains = []
    reconstructed_winner = None
    reconstructed_best_score = float("inf")
    for group in groups:
        if not group["sampled"]:
            continue
        evaluation = _evaluate_group(
            context, population, group, caches, parent_mse
        )
        evaluations[_group_signature(group)] = evaluation
        threshold_gains.extend(
            evaluation.gains[evaluation.valid_mask].astype(float).tolist()
        )
        fields = _best_fields(evaluation)
        row = {
            "audit_id": spec.audit_id,
            "candidate_id": context.candidate["candidate_id"],
            "stage": context.result["stage"],
            "split_sampling_seed": context.result["split_sampling_seed"],
            "outcome": context.outcome,
            "node_id": spec.node_id,
            "depth": node["depth"],
            "node_sample_size": len(population.patient_ids),
            "generation_index": group["generation_index"],
            "sampling_draw": group["sampling_draw"],
            "sampled": True,
            "feature": group["feature"],
            "anchor": group["anchor"],
            "offset": group["offset"],
            "aggregation": group["aggregation"],
            "threshold_count": len(evaluation.thresholds),
            "valid_threshold_count": evaluation.valid_count,
            "invalid_threshold_count": evaluation.invalid_count,
            **fields,
        }
        ranking_rows.append(row)
        if (
            evaluation.best_index is not None
            and float(evaluation.scores[evaluation.best_index])
            < reconstructed_best_score
        ):
            reconstructed_best_score = float(
                evaluation.scores[evaluation.best_index]
            )
            reconstructed_winner = (group, evaluation)

    if reconstructed_winner is None:
        raise RuntimeError(f"No valid sampled candidate at {spec.audit_id}.")
    winner_group, winner_evaluation = reconstructed_winner
    winner_index = winner_evaluation.best_index
    winner_gain = float(winner_evaluation.gains[winner_index])

    if (
        winner_group["feature"] != node["feature"]
        or winner_group["anchor"] != node["anchor"]
        or not _offset_equal(winner_group["offset"], node["offset"])
        or winner_group["aggregation"] != node["aggregation"]
        or not np.isclose(
            float(winner_evaluation.thresholds[winner_index]),
            float(node["threshold"]),
            rtol=0.0,
            atol=FLOAT_TOLERANCE,
        )
    ):
        raise RuntimeError(
            f"Winner reconstruction failed at {spec.audit_id}: "
            f"{_group_signature(winner_group)} / "
            f"{winner_evaluation.thresholds[winner_index]} versus saved "
            f"{node['feature'], node['anchor'], node['offset'], node['aggregation']} "
            f"/ {node['threshold']}"
        )

    left = context.training[int(node["left_child"])]
    right = context.training[int(node["right_child"])]
    reconstructed_sizes = (
        int(winner_evaluation.left_sizes[winner_index]),
        int(winner_evaluation.right_sizes[winner_index]),
    )
    saved_sizes = (len(left.patient_ids), len(right.patient_ids))
    if reconstructed_sizes != saved_sizes:
        raise RuntimeError(
            f"Child-size reconstruction failed at {spec.audit_id}: "
            f"{reconstructed_sizes} != {saved_sizes}"
        )

    weighted_child_mse = (
        len(left.labels) * _mse(left.labels)
        + len(right.labels) * _mse(right.labels)
    ) / len(population.labels)
    direct_gain = parent_mse - weighted_child_mse
    if not np.isclose(direct_gain, winner_gain, rtol=0.0, atol=FLOAT_TOLERANCE):
        raise RuntimeError(
            f"Gain reconstruction failed at {spec.audit_id}: "
            f"{direct_gain} != {winner_gain}"
        )

    valid_rows = [row for row in ranking_rows if row["best_gain"] is not None]
    valid_rows.sort(key=lambda row: (-row["best_gain"], row["generation_index"]))
    for rank, row in enumerate(valid_rows, start=1):
        row["group_rank"] = rank
        row["gain_gap_from_winner"] = winner_gain - row["best_gain"]
        row["within_abs_1e_4_of_winner"] = (
            row["gain_gap_from_winner"] <= ABSOLUTE_NEAR_TIE + FLOAT_TOLERANCE
        )
        row["within_abs_5e_4_of_winner"] = (
            row["gain_gap_from_winner"]
            <= SECONDARY_ABSOLUTE_NEAR_TIE + FLOAT_TOLERANCE
        )
        row["within_5_percent_of_winner_gain"] = (
            row["best_gain"]
            >= winner_gain * (1.0 - RELATIVE_NEAR_TIE_FRACTION)
            - FLOAT_TOLERANCE
        )
    invalid_rows = [row for row in ranking_rows if row["best_gain"] is None]
    for row in invalid_rows:
        row.update({
            "group_rank": None,
            "gain_gap_from_winner": None,
            "within_abs_1e_4_of_winner": False,
            "within_abs_5e_4_of_winner": False,
            "within_5_percent_of_winner_gain": False,
        })
    ranking_rows = valid_rows + invalid_rows
    rank_by_signature = {
        _group_signature(row): row["group_rank"]
        for row in valid_rows
    }

    replay_by_signature = {
        _group_signature(group): group for group in groups
    }
    expected_rows = []
    for expected in _expected_groups(context.outcome):
        signature = _group_signature(expected)
        replay = replay_by_signature.get(signature)
        generated = replay is not None
        evaluation = evaluations.get(signature)
        if evaluation is None:
            evaluation = _evaluate_group(
                context, population, expected, caches, parent_mse
            )
        fields = _best_fields(evaluation)
        sampled = bool(replay["sampled"]) if generated else False
        if not generated:
            reason = "not_generated"
            classification = "insufficient"
        elif not sampled:
            reason = "sampled_group_skipped"
            classification = "A"
        elif evaluation.best_index is None:
            reason = "all_thresholds_invalid"
            classification = "B"
        elif fields["best_gain"] > winner_gain + FLOAT_TOLERANCE:
            reason = "sampled_expected_candidate_better_than_saved_winner"
            classification = "D"
        elif signature == _group_signature(winner_group):
            reason = "expected_candidate_selected"
            classification = "expected_selected"
        else:
            reason = "sampled_valid_but_lower_or_equal_empirical_gain"
            classification = "C"
        expected_rows.append({
            "audit_id": spec.audit_id,
            "candidate_id": context.candidate["candidate_id"],
            "stage": context.result["stage"],
            "split_sampling_seed": context.result["split_sampling_seed"],
            "outcome": context.outcome,
            "node_id": spec.node_id,
            "depth": node["depth"],
            "node_sample_size": len(population.patient_ids),
            "feature": expected["feature"],
            "anchor": expected["anchor"],
            "offset": expected["offset"],
            "aggregation": expected["aggregation"],
            "generated": generated,
            "generation_index": replay["generation_index"] if generated else None,
            "sampling_draw": replay["sampling_draw"] if generated else None,
            "sampled": sampled,
            "threshold_count": len(evaluation.thresholds),
            "valid_threshold_count": evaluation.valid_count,
            "invalid_threshold_count": evaluation.invalid_count,
            "thresholds_considered_json": _json_numbers(evaluation.thresholds),
            "valid_thresholds_json": _json_numbers(
                evaluation.thresholds[evaluation.valid_mask]
            ),
            "invalid_thresholds_json": _json_numbers(
                evaluation.thresholds[~evaluation.valid_mask]
            ),
            **fields,
            "sampled_group_rank": rank_by_signature.get(signature),
            "winner_gain": winner_gain,
            "gain_gap_winner_minus_expected": (
                None if fields["best_gain"] is None
                else winner_gain - fields["best_gain"]
            ),
            "counterfactual_better_than_winner": (
                fields["best_gain"] is not None
                and fields["best_gain"] > winner_gain + FLOAT_TOLERANCE
            ),
            "sampled_expected_better_than_winner": (
                sampled
                and fields["best_gain"] is not None
                and fields["best_gain"] > winner_gain + FLOAT_TOLERANCE
            ),
            "classification": classification,
            "reason": reason,
        })

    sampled_expected_valid = [
        row for row in expected_rows
        if row["sampled"] and row["best_gain"] is not None
    ]
    all_expected_valid = [
        row for row in expected_rows if row["best_gain"] is not None
    ]
    if any(row["sampled_expected_better_than_winner"] for row in expected_rows):
        node_classification = "D"
    elif any(row["reason"] == "expected_candidate_selected" for row in expected_rows):
        node_classification = "expected_selected"
    elif not any(row["sampled"] for row in expected_rows):
        node_classification = "A"
    elif not sampled_expected_valid:
        node_classification = "B"
    else:
        node_classification = "C"

    best_sampled_expected = max(
        sampled_expected_valid,
        key=lambda row: row["best_gain"],
        default=None,
    )
    best_counterfactual_expected = max(
        all_expected_valid,
        key=lambda row: row["best_gain"],
        default=None,
    )
    min_leaf = int(context.candidate["min_samples_leaf"])
    minimum_child = min(len(left.patient_ids), len(right.patient_ids))
    threshold_gains_array = np.asarray(threshold_gains, dtype=float)
    inventory = {
        "audit_id": spec.audit_id,
        "role": spec.role,
        "candidate_id": context.candidate["candidate_id"],
        "stage": context.result["stage"],
        "evaluation_split": context.result["evaluation_split"],
        "split_sampling_seed": context.result["split_sampling_seed"],
        "outcome": context.outcome,
        "node_id": spec.node_id,
        "depth": node["depth"],
        "train_total": context.train_total,
        "parent_sample_size": len(population.patient_ids),
        "training_node_coverage": len(population.patient_ids) / context.train_total,
        "held_out_total": context.held_out_total,
        "held_out_node_sample_size": len(held_out_population.patient_ids),
        "held_out_node_coverage": (
            len(held_out_population.patient_ids) / context.held_out_total
        ),
        "parent_positive_count": int(np.sum(population.labels)),
        "parent_prevalence": parent_prevalence,
        "parent_mse": parent_mse,
        "selected_feature": node["feature"],
        "selected_anchor": node["anchor"],
        "selected_offset": node["offset"],
        "selected_aggregation": node["aggregation"],
        "selected_threshold": float(node["threshold"]),
        "left_size": len(left.patient_ids),
        "right_size": len(right.patient_ids),
        "left_prevalence": float(np.mean(left.labels)),
        "right_prevalence": float(np.mean(right.labels)),
        "left_mse": _mse(left.labels),
        "right_mse": _mse(right.labels),
        "weighted_child_mse": weighted_child_mse,
        "selected_gain": direct_gain,
        "min_samples_leaf": min_leaf,
        "minimum_child_size": minimum_child,
        "minimum_child_margin_above_min_leaf": minimum_child - min_leaf,
        "minimum_child_to_min_leaf_ratio": minimum_child / min_leaf,
        "minimum_child_at_constraint": minimum_child == min_leaf,
        "generated_group_count_at_node": len(groups),
        "sampled_group_count_at_node": sum(group["sampled"] for group in groups),
        "valid_sampled_group_count": len(valid_rows),
        "evaluated_valid_threshold_count": len(threshold_gains),
        "selected_group_rank": rank_by_signature[_group_signature(winner_group)],
        "groups_within_abs_1e_4": sum(
            row["within_abs_1e_4_of_winner"] for row in valid_rows
        ),
        "groups_within_abs_5e_4": sum(
            row["within_abs_5e_4_of_winner"] for row in valid_rows
        ),
        "groups_within_5_percent": sum(
            row["within_5_percent_of_winner_gain"] for row in valid_rows
        ),
        "thresholds_within_abs_1e_4": int(np.sum(
            winner_gain - threshold_gains_array
            <= ABSOLUTE_NEAR_TIE + FLOAT_TOLERANCE
        )),
        "thresholds_within_abs_5e_4": int(np.sum(
            winner_gain - threshold_gains_array
            <= SECONDARY_ABSOLUTE_NEAR_TIE + FLOAT_TOLERANCE
        )),
        "thresholds_within_5_percent": int(np.sum(
            threshold_gains_array
            >= winner_gain * (1.0 - RELATIVE_NEAR_TIE_FRACTION)
            - FLOAT_TOLERANCE
        )),
        "best_sampled_expected_feature": (
            best_sampled_expected["feature"] if best_sampled_expected else None
        ),
        "best_sampled_expected_aggregation": (
            best_sampled_expected["aggregation"] if best_sampled_expected else None
        ),
        "best_sampled_expected_gain": (
            best_sampled_expected["best_gain"] if best_sampled_expected else None
        ),
        "best_sampled_expected_rank": (
            best_sampled_expected["sampled_group_rank"]
            if best_sampled_expected else None
        ),
        "winner_minus_best_sampled_expected_gain": (
            None if best_sampled_expected is None
            else winner_gain - best_sampled_expected["best_gain"]
        ),
        "best_counterfactual_expected_feature": (
            best_counterfactual_expected["feature"]
            if best_counterfactual_expected else None
        ),
        "best_counterfactual_expected_aggregation": (
            best_counterfactual_expected["aggregation"]
            if best_counterfactual_expected else None
        ),
        "best_counterfactual_expected_gain": (
            best_counterfactual_expected["best_gain"]
            if best_counterfactual_expected else None
        ),
        "any_sampled_expected_better_than_selected": any(
            row["sampled_expected_better_than_winner"] for row in expected_rows
        ),
        "reconstruction_verified": True,
        "node_primary_classification": node_classification,
    }
    return inventory, expected_rows, ranking_rows


def main() -> None:
    """Run the approved read-only reconstruction and write isolated tables."""
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing diagnostics: {OUTPUT_DIR}"
        )

    data, labels_by_outcome = search.load_v2_inputs()
    split_table = pd.read_csv(SEARCH_RESULTS / "shared_patient_splits.csv")
    contexts = {}
    inventory_rows = []
    expected_rows = []
    ranking_rows = []
    input_artifacts = set()

    for spec in NODE_SPECS:
        print(f"[AUDIT] {spec.audit_id}", flush=True)
        result_path = SEARCH_RESULTS / spec.result_path
        key = (str(result_path), spec.outcome)
        if key not in contexts:
            contexts[key] = _load_context(
                result_path,
                spec.outcome,
                data,
                labels_by_outcome,
                split_table,
            )
        context = contexts[key]
        inventory, expected, ranking = _audit_node(spec, context)
        print(
            f"[VERIFIED] {spec.audit_id}: "
            f"gain={inventory['selected_gain']:.9f}",
            flush=True,
        )
        inventory_rows.append(inventory)
        expected_rows.extend(expected)
        ranking_rows.extend(ranking)
        input_artifacts.update((result_path, context.tree_path))

    if any(row["any_sampled_expected_better_than_selected"] for row in inventory_rows):
        raise RuntimeError(
            "A sampled expected candidate beat a saved winner; diagnostics "
            "must be reviewed before outputs are trusted."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(inventory_rows).to_csv(
        OUTPUT_DIR / "node_inventory.csv", index=False
    )
    pd.DataFrame(expected_rows).to_csv(
        OUTPUT_DIR / "expected_candidate_audit.csv", index=False
    )
    pd.DataFrame(ranking_rows).to_csv(
        OUTPUT_DIR / "candidate_group_ranking.csv", index=False
    )

    manifest = {
        "diagnostic_scope": "Approved Steps 1-4 only",
        "training_performed": False,
        "temporal_tree_modified": False,
        "synthetic_data_modified": False,
        "node_specs": [asdict(spec) for spec in NODE_SPECS],
        "near_tie_definitions": {
            "absolute_primary": ABSOLUTE_NEAR_TIE,
            "absolute_secondary": SECONDARY_ABSOLUTE_NEAR_TIE,
            "relative_fraction_below_winner": RELATIVE_NEAR_TIE_FRACTION,
        },
        "reconstruction_acceptance": {
            "required_exact_fields": [
                "feature",
                "anchor",
                "offset",
                "aggregation",
                "threshold",
                "left_size",
                "right_size",
                "gain",
            ],
            "floating_absolute_tolerance": FLOAT_TOLERANCE,
            "saved_threshold_boundary_rule": (
                "A saved threshold is snapped to the nearest recomputed "
                "observed aggregate only when their absolute difference is "
                "at most the floating tolerance."
            ),
            "all_nodes_verified": all(
                row["reconstruction_verified"] for row in inventory_rows
            ),
        },
        "inputs": {
            str(path.relative_to(REPOSITORY_ROOT)): _sha256(path)
            for path in sorted(input_artifacts)
        },
        "immutable_data_hashes": json.loads(
            (SEARCH_RESULTS / "split_details.json").read_text(encoding="utf-8")
        )["input_sha256"],
        "shared_patient_splits_sha256": _sha256(
            SEARCH_RESULTS / "shared_patient_splits.csv"
        ),
        "outputs": [
            "node_inventory.csv",
            "expected_candidate_audit.csv",
            "candidate_group_ranking.csv",
        ],
    }
    (OUTPUT_DIR / "diagnostic_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
