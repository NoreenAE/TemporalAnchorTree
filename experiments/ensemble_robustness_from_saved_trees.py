"""Reconstruct Ensemble Robustness trees and calculate FA/GI ensemble AUCs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import ensemble_robustness as original
from settree.temporal_tree import TemporalLeaf, TemporalSplitNode, TemporalTree

SOURCE = ROOT / "experiments/results/experiment_2_robustness_v3"
OUTPUT = ROOT / "experiments/results/experiment_2_ensemble_from_saved_trees"
SEEDS = tuple(range(42, 52))
OUTCOMES = ("FA", "GI")
AUC_TOLERANCE = 1e-12


def reconstruct_tree(records: Sequence[Mapping]) -> TemporalTree:
    """Restore a fitted inference tree from one Ensemble Robustness tree JSON file."""
    nodes = {record["node_id"]: record for record in records}
    if not records or len(nodes) != len(records) or 0 not in nodes:
        raise ValueError("Tree JSON must contain a unique node list rooted at 0.")
    visited = set()

    def visit(node_id: int, depth: int):
        if node_id in visited or node_id not in nodes:
            raise ValueError("Tree JSON contains a cycle or missing child.")
        visited.add(node_id)
        record = nodes[node_id]
        if record.get("depth") != depth or record.get("num_patients", 0) < 1:
            raise ValueError("Tree JSON contains invalid node metadata.")
        if record["type"] == "leaf":
            return TemporalLeaf(record["prediction"], record["num_patients"], depth)
        if record["type"] != "split":
            raise ValueError("Tree JSON contains an unknown node type.")
        anchor = record["anchor"]
        offset = record["offset"]
        if record["is_full_timeline"] != (anchor is None and offset is None):
            raise ValueError("Tree JSON contains an invalid timeline flag.")
        left = visit(record["left_child"], depth + 1)
        right = visit(record["right_child"], depth + 1)
        if left.num_patients + right.num_patients != record["num_patients"]:
            raise ValueError("Tree JSON child counts do not match the parent.")
        return TemporalSplitNode(
            feature=record["feature"],
            threshold=record["threshold"],
            aggregation=record["aggregation"],
            time_window=(anchor, offset),
            missing_direction=record["missing_direction"],
            num_patients=record["num_patients"],
            depth=depth,
            impurity=record.get("impurity"),
            left=left,
            right=right,
        )

    tree = visit(0, 0)
    if visited != set(nodes):
        raise ValueError("Tree JSON contains unreachable nodes.")
    model = TemporalTree(is_classification=False, use_dynamic_anchors=False)
    model.tree = tree
    return model


def load_saved_inputs():
    """Load the exact data and patient partition recorded by Ensemble Robustness."""
    config = json.loads((SOURCE / "experiment_config.json").read_text())
    if tuple(config["split_sampling_seeds"]) != SEEDS:
        raise ValueError("The saved run does not contain seeds 42 through 51.")
    split = pd.read_csv(SOURCE / "shared_patient_split.csv")
    data = original.load_longitudinal_x(original.INPUT_PATHS["x"])
    labels = {
        outcome: original.load_labels(original.INPUT_PATHS[outcome.lower()], outcome.lower())
        for outcome in OUTCOMES
    }
    return data, labels, split


def prediction_table(patient_ids, labels, scores) -> pd.DataFrame:
    """Align labels and all member scores by the saved patient order."""
    table = pd.DataFrame({"patient_id": patient_ids, "y_true": labels.to_numpy()})
    for seed, values in scores.items():
        if len(values) != len(table):
            raise ValueError("A member returned the wrong number of scores.")
        table[f"seed_{seed}_score"] = values
    if table.patient_id.duplicated().any() or table.isna().any().any():
        raise ValueError("Patient IDs or scores are not aligned.")
    return table


def validate_and_ensemble(output: Path = OUTPUT) -> pd.DataFrame:
    """Validate all members, then calculate the two score-level ensembles."""
    data, labels, split = load_saved_inputs()
    test_ids = split.loc[split["split"].eq("test"), "patient_id"].to_numpy()
    saved = pd.read_csv(SOURCE / "per_seed_results.csv")
    validation_rows = []
    prediction_tables = {}

    for outcome in OUTCOMES:
        test_data, test_labels, patient_order = original.subset_for_patients(
            data, labels[outcome], test_ids
        )
        scores = {}
        for seed in SEEDS:
            model = reconstruct_tree(
                json.loads(
                    (SOURCE / f"seeds/seed_{seed}/{outcome}_tree.json").read_text()
                )
            )
            member_scores = np.asarray(model.predict(test_data), dtype=float)
            reconstructed_auc = float(roc_auc_score(test_labels, member_scores))
            saved_auc = float(
                saved.loc[
                    saved.outcome.eq(outcome)
                    & saved.split_sampling_seed.eq(seed),
                    "roc_auc",
                ].iloc[0]
            )
            difference = abs(reconstructed_auc - saved_auc)
            validation_rows.append(
                {
                    "outcome": outcome,
                    "split_sampling_seed": seed,
                    "saved_auc": saved_auc,
                    "reconstructed_auc": reconstructed_auc,
                    "absolute_difference": difference,
                    "validation_passed": difference <= AUC_TOLERANCE,
                }
            )
            scores[seed] = member_scores
        prediction_tables[outcome] = prediction_table(
            patient_order, test_labels, scores
        )

    validation = pd.DataFrame(validation_rows)
    output.mkdir(parents=True, exist_ok=True)
    validation.to_csv(output / "reconstruction_validation.csv", index=False)
    if not validation.validation_passed.all():
        raise RuntimeError("A reconstructed AUC did not match the saved AUC.")

    summary_rows = []
    for outcome, table in prediction_tables.items():
        score_columns = [f"seed_{seed}_score" for seed in SEEDS]
        table["ensemble_score"] = table[score_columns].mean(axis=1)
        ensemble_auc = float(roc_auc_score(table.y_true, table.ensemble_score))
        table.to_csv(output / f"{outcome}_ensemble_predictions.csv", index=False)
        aucs = validation.loc[validation.outcome.eq(outcome), "reconstructed_auc"]
        summary_rows.append(
            {
                "outcome": outcome,
                **{f"auc_seed_{seed}": float(value) for seed, value in zip(SEEDS, aucs)},
                "individual_auc_mean": float(aucs.mean()),
                "individual_auc_sample_sd": float(aucs.std(ddof=1)),
                "individual_auc_min": float(aucs.min()),
                "individual_auc_max": float(aucs.max()),
                "ensemble_auc": ensemble_auc,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "ensemble_summary.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    validate_and_ensemble(args.output_dir)


if __name__ == "__main__":
    main()
