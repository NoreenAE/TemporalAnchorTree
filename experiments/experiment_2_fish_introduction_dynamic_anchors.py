"""Experiment 2: Fish Introduction – Dynamic Anchors.

Run: python experiments/experiment_2_fish_introduction_dynamic_anchors.py
Load data/fish_intro_dataset_distractors_v2/; never regenerate observations,
labels or splits. All eight event features can create dynamic anchors.
Generate a new dataset separately with generate_fish_intro_dataset.py.
Birth windows are cumulative [birth, birth + m months], m=1..22, plus
full follow-up. Dynamic windows extend 1, 2, 3 or 4 calendar months.
Fish introductions after month 22 remain accessible through full follow-up.
No symptoms coincide with introduction, so the strict-after truth agrees.
After both runs finish, saved artifacts produce combined ROC and tree figures
(roc_on_vs_off and decision_trees_on_vs_off, PNG/PDF) in --output, plus
anchor_usage_test_summary.json and anchor_usage_test_{patients,creations,windows}.csv.
Usage exports join to observations.csv by patient_id; only actual held-out
prediction visits are exported for every dynamic-anchor feature, with separate
FishIntroduction summaries. Creation and downstream use remain distinct.
Outputs are never overwritten. No execution occurs on import.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
import platform
import shutil
import sys
from types import MethodType
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.controlled_dynamic_anchors_100 import (  # noqa: E402
    anchor_values, require, tree_records, write_json,
)
from settree.temporal_tree import TemporalTree  # noqa: E402
from generate_fish_intro_dataset import (  # noqa: E402
    BIRTH, DEFAULT_DATA_DIR, DISTRACTORS, FISH, SEED, SYMPTOMS,
)

# The only run-specific setting is USE_DYNAMIC_ANCHORS.
MODEL_RUNS = (
    dict(name="dynamic_on", USE_DYNAMIC_ANCHORS=True),
    dict(name="dynamic_off", USE_DYNAMIC_ANCHORS=False),
)
SHARED_MODEL_CONFIG = dict(
    is_classification=False, criterion="mse", max_depth=3,
    min_samples_split=100, min_samples_leaf=50, min_impurity_decrease=0.0,
    anchors_list=[BIRTH], anchor_offset_list=[1, 2, 3, 4],
    anchor_offset_map={BIRTH: [*range(1, 23), None]},
    aggregation_funcs={"sum": np.sum},
    feature_semantics={
        f: "binary_event_indicator" for f in (FISH, *SYMPTOMS, *DISTRACTORS)
    },
    candidate_sampling_rate=0.4, random_state=SEED, split_sampling_seed=SEED,
    learn_missing_direction=False, verbose=0,
)
DATASET_FILES = ("observations.csv", "patients_and_split.csv", "validation.json")


def load_dataset(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Restore saved types without generating events, labels, or partitions."""
    for name in DATASET_FILES:
        require((directory / name).is_file(), f"Missing dataset file: {directory / name}")
    data = pd.read_csv(directory / "observations.csv", parse_dates=["time"],
                       dtype={"patient_id": str, "value": object})
    birth_rows = data.feature_name.eq(BIRTH)
    data.loc[birth_rows, "value"] = pd.to_datetime(data.loc[birth_rows, "value"])
    data.loc[~birth_rows, "value"] = pd.to_numeric(data.loc[~birth_rows, "value"])
    require(set(data.feature_name) == {BIRTH, FISH, *SYMPTOMS, *DISTRACTORS},
            "Expected revised Experiment 2 feature names; use a separately "
            "generated v2 dataset, not a historical dataset")
    patients = pd.read_csv(
        directory / "patients_and_split.csv",
        parse_dates=["birth", "followup_end", "a", "b", "signal_time", "fish_time"],
        dtype={"patient_id": str, "documented": bool},
    )
    validation = json.loads((directory / "validation.json").read_text())
    return data, patients, validation


def make_model(enabled: bool, model_config: dict = None) -> TemporalTree:
    """Isolate mutable settings while changing only the dynamic-anchor switch."""
    config = SHARED_MODEL_CONFIG if model_config is None else model_config
    return TemporalTree(use_dynamic_anchors=enabled, **deepcopy(config))


def evaluate(group: pd.DataFrame) -> dict:
    """Report probability metrics and fixed, equal-width calibration bins."""
    bins = np.minimum((group.score.to_numpy() * 10).astype(int), 9)
    calibration = []
    for index in range(10):
        subset = group.loc[bins == index]
        calibration.append(dict(
            lower=index / 10, upper=(index + 1) / 10, n=len(subset),
            mean_prediction=float(subset.score.mean()) if len(subset) else None,
            observed_FA=float(subset.FA.mean()) if len(subset) else None,
        ))
    return dict(n=len(group), FA_cases=int(group.FA.sum()),
                roc_auc=float(roc_auc_score(group.FA, group.score))
                if group.FA.nunique() == 2 else None,
                brier=float(brier_score_loss(group.FA, group.score)),
                mean_prediction=float(group.score.mean()),
                prevalence=float(group.FA.mean()), calibration=calibration)


def run_model(enabled: bool, data: pd.DataFrame, patients: pd.DataFrame,
              output: Path, *, model_config: dict = None) -> dict:
    """Fit normally; audit actual prediction calls against training anchors."""
    model = make_model(enabled, model_config)
    if model_config is not None:
        effective = model.get_params(deep=False)
        effective["aggregation_funcs"] = {
            name: f"{function.__module__}.{function.__qualname__}"
            for name, function in effective["aggregation_funcs"].items()
        }
        write_json(output / "effective_configuration.json", effective)
    info = patients.set_index("patient_id")
    training = data.loc[data.patient_id.isin(
        patients.loc[patients.partition.eq("train"), "patient_id"])]
    captured = {}
    prepare = model._prepare_branch_anchor

    def capture(self, **kwargs):
        result = prepare(**kwargs)
        for row in result[2].itertuples(index=False):
            captured[(kwargs["node"].source_node_id, row.patient_id)] = (
                kwargs["direction"], row.value)
        return result

    model._prepare_branch_anchor = MethodType(capture, model)
    print(f"Fitting dynamic anchors {'ON' if enabled else 'OFF'}", flush=True)
    # Measure fit only, including the existing training-anchor capture.
    fit_started = perf_counter()
    try:
        model.fit(training, info.loc[training.patient_id.unique(), "FA"].reset_index(drop=True))
        training_runtime_seconds = perf_counter() - fit_started
    finally:
        del model._prepare_branch_anchor
    records, ids = tree_records(model.tree)
    creators = {}

    def identify(node):
        if node.is_leaf():
            return
        records[ids[id(node)]]["source_node_id"] = node.source_node_id
        records[ids[id(node)]]["is_distractor_split"] = node.feature in DISTRACTORS
        if enabled and model._dynamic_anchor_selection(node) is not None:
            creators[model._format_dynamic_anchor_name(node, "left")] = node.feature
        identify(node.left)
        identify(node.right)

    identify(model.tree)
    write_json(output / "tree.json", records)
    (output / "tree.txt").write_text("\n".join(
        "  " * record["depth"] + str(record) for record in records) + "\n")
    traces, errors = [], []
    append = model._append_prediction_branch_anchor

    def audit(self, path, node, direction):
        pid = path.patient_id.iloc[0]
        before = anchor_values(path)
        source, offset = node.time_window
        references = [] if offset is None else before[source]
        if isinstance(references, pd.Timestamp):
            references = [references]
        times = path.loc[path.feature_name.eq(node.feature), "time"]
        windows = [(start, start + pd.DateOffset(months=offset)) for start in references]
        eligible = times if offset is None else times.loc[
            times.map(lambda time: any(start <= time <= end for start, end in windows))]
        expected = [eligible.min()] if len(eligible) else []
        aggregate = float(len(eligible)) if len(eligible) else np.nan
        outgoing = append(path, node, direction)
        after = anchor_values(outgoing)
        anchor_eligible = model._dynamic_anchor_selection(node) is not None
        creates_anchor = enabled and anchor_eligible
        name = model._format_dynamic_anchor_name(node, direction) if creates_anchor else None
        selected = after.get(name, []) if creates_anchor else []
        checks = dict(
            timestamp_correct=selected == (expected if creates_anchor else []),
            routing_correct=direction == model._get_prediction_direction(aggregate, node),
            inherited_anchors_preserved=all(after.get(k) == v for k, v in before.items()),
            path_isolated=set(after) == set(before) | ({name} if creates_anchor else set()),
        )
        if enabled and info.loc[pid, "partition"] == "train":
            checks["training_prediction_consistent"] = captured.get(
                (node.source_node_id, pid)) == (
                    (direction, selected) if creates_anchor else None
                )
        errors.extend(f"{pid}/source {node.source_node_id}: {key}"
                      for key, passed in checks.items() if not passed)
        uses_fish = node.feature in SYMPTOMS and creators.get(source) == FISH
        traces.append(dict(
            patient_id=pid, node_id=ids[id(node)], source_node_id=node.source_node_id,
            feature=node.feature, direction=direction, source_anchor=source,
            offset=offset, evaluated_windows=windows,
            eligible_symptom_count=len(eligible) if node.feature in SYMPTOMS else None,
            anchor_eligible=anchor_eligible,
            anchor_name=name, anchor_timestamps=selected,
            dynamic_anchor_created=bool(creates_anchor and selected),
            source_anchor_feature=creators.get(source),
            available_dynamic_anchor_used=bool(creators.get(source) and references),
            fish_anchor_created=bool(creates_anchor and node.feature == FISH and selected),
            symptom_uses_fish_anchor=uses_fish,
            fish_anchor_available_for_symptom=bool(uses_fish and references), checks=checks,
        ))
        return outgoing

    print("Auditing actual prediction paths for all 5,000 children", flush=True)
    model._append_prediction_branch_anchor = MethodType(audit, model)
    try:
        scores = model.predict(data)
    finally:
        del model._append_prediction_branch_anchor
    predictions = patients.merge(pd.DataFrame(
        {"patient_id": data.patient_id.unique(), "score": scores}), on="patient_id",
        validate="one_to_one")
    for column, flag in (("fish_anchor_created", "fish_anchor_created"),
                         ("symptom_split_uses_fish_anchor", "symptom_uses_fish_anchor"),
                         ("available_fish_anchor_used", "fish_anchor_available_for_symptom")):
        pids = {row["patient_id"] for row in traces if row[flag]}
        predictions[column] = predictions.patient_id.isin(pids)
    predictions.to_csv(output / "patient_predictions.csv", index=False)
    write_json(output / "anchor_trace.json", traces)
    test = predictions.loc[predictions.partition.eq("test")]
    metrics = {"test_overall": evaluate(test),
               "test_documented": evaluate(test.loc[test.documented])}
    write_json(output / "metrics.json", metrics)
    coverage = {}
    for partition in ("train", "test"):
        subset = predictions.loc[predictions.partition.eq(partition) & predictions.documented]
        coverage[partition] = dict(
            documented_patients=len(subset),
            fish_anchor_created=int(subset.fish_anchor_created.sum()),
            fish_anchor_coverage=float(subset.fish_anchor_created.mean()),
            available_fish_anchor_used=int(subset.available_fish_anchor_used.sum()),
        )
    selected_splits = [record for record in records if record["type"] == "split"]
    selected_features = sorted({record["feature"] for record in selected_splits})
    summary = dict(
        training_runtime_seconds=training_runtime_seconds,
        training_runtime_scope="model.fit including training-anchor capture; excludes prediction audit",
        selected_features=selected_features,
        selected_feature_split_counts={
            feature: sum(record["feature"] == feature for record in selected_splits)
            for feature in selected_features
        },
        distractor_split_counts={
            feature: sum(record["feature"] == feature for record in selected_splits)
            for feature in DISTRACTORS
        },
        distractor_splits=[record for record in selected_splits
                           if record["is_distractor_split"]],
        mechanism_checks_passed=not errors, audit_failures=errors,
        audited_split_visits=len(traces), coverage=coverage,
        fish_anchor_creation_by_branch={branch: sum(
            row["fish_anchor_created"] and row["direction"] == branch for row in traces)
            for branch in ("left", "right")},
        symptom_nodes_using_fish_anchor=sorted({row["source_node_id"] for row in traces
                                              if row["symptom_uses_fish_anchor"]}),
        metrics=metrics,
    )
    write_json(output / "diagnostics.json", summary)
    with (output / "model.pkl").open("wb") as file:
        pickle.dump(model, file)
    require(not errors, f"Mechanism audit failed; see {output / 'diagnostics.json'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "experiments/results/experiment_2_fish_introduction_dynamic_anchors_v2")
    args = parser.parse_args()
    require(not args.output.exists(), "Output exists; choose another --output")
    data, patients, validation = load_dataset(args.data_dir)
    args.output.mkdir(parents=True)
    # Preserve the existing output layout with byte-for-byte input snapshots.
    for name in DATASET_FILES:
        shutil.copyfile(args.data_dir / name, args.output / name)
    params = make_model(True).get_params()
    params.pop("aggregation_funcs")
    write_json(args.output / "configuration.json", dict(
        seed=SEED, parameters=params, aggregations=["sum"],
        dataset_directory=str(args.data_dir.resolve()),
        dataset_sha256={name: hashlib.sha256((args.data_dir / name).read_bytes()).hexdigest()
                        for name in DATASET_FILES},
        FA_probabilities={"R1": .90, "R0": .05},
        split="400 train / 100 test blocks, stratified by scenario and signal type",
        birth_dates="Uniform days from 2010-01-01 through 2014-12-31",
        calibration="10 fixed equal-width bins; last bin includes 1",
        versions=dict(python=platform.python_version(), numpy=np.__version__,
                      pandas=pd.__version__, sklearn=sklearn.__version__),
        sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (Path(__file__).resolve(), ROOT / "generate_fish_intro_dataset.py",
                             ROOT / "settree/temporal_tree.py",
                             ROOT / "settree/anchor_utils.py",
                             ROOT / "experiments/controlled_dynamic_anchors_100.py")},
    ))
    print(f"Loaded dataset; saved validation: {validation}", flush=True)
    results = {}
    for run in MODEL_RUNS:
        output = args.output / run["name"]
        output.mkdir()
        results[run["name"]] = run_model(
            run["USE_DYNAMIC_ANCHORS"], data, patients, output
        )
    write_json(args.output / "comparison.json", results)
    from experiments.experiment_2_outputs import save_experiment_outputs

    save_experiment_outputs(args.output)
    print(f"Completed. Results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
