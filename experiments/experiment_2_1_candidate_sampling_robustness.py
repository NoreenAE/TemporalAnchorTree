"""Experiment 2.1: five paired candidate-sampling seeds on Noisy Matched V2.

From the repository root, using the research Python environment::

    python -m experiments.experiment_2_1_candidate_sampling_robustness run
    python -m experiments.experiment_2_1_candidate_sampling_robustness summarize

Only ``run`` trains models, for seeds 43--47 only. Repeating that command
verifies and skips completed runs. An incomplete attempt stops execution;
``run --retry-incomplete`` archives it before retrying that model. A completed
attempt awaiting publication is recovered without training. Failed completion
hash checks always stop execution. An OS lock prevents concurrent writers and
is released automatically when a process exits, including after interruption.

The successful seed-42 checkpoint is read-only. All new artifacts live under
experiments/results/experiment_2_1_candidate_sampling_robustness by default.
``summarize`` never fits, predicts, or unpickles models. It requires all ten
verified runs, rebuilds summary/, and preserves older summaries in history/.
Mean/SD describe seed variability on a fixed cohort, not patient uncertainty.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from uuid import uuid4
from time import perf_counter
from typing import Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import experiment_2_noisy_v2 as baseline
from experiments.experiment_2_fish_introduction_dynamic_anchors import (
    make_model, run_model, write_json,
)

BASELINE = ROOT / "experiments/results/experiment_2_noisy_matched_v2"
DATA = ROOT / "data/fish_intro_noisy_matched_v2"
CHECKPOINT = ROOT / "experiments/results/experiment_2_1_seed42_checkpoint"
OUTPUT = ROOT / "experiments/results/experiment_2_1_candidate_sampling_robustness"
RUN_ARTIFACTS = (
    "effective_configuration.json", "patient_predictions.csv", "metrics.json",
    "tree.json", "tree.txt", "anchor_trace.json", "diagnostics.json", "model.pkl",
)
CHECKPOINT_SEED = 42
PLANNED_SEEDS = (43, 44, 45, 46, 47)
RUNS = ("dynamic_on", "dynamic_off")
POPULATIONS = ("test_overall", "test_documented")
ATOL = 1e-12
PREDICTION_FLAGS = (
    "fish_anchor_created", "symptom_split_uses_fish_anchor",
    "available_fish_anchor_used",
)


def read_json(path: Path):
    """Read an existing JSON artifact."""
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    """Hash exact bytes, including saved row ordering."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_config(saved: dict, seed: int) -> dict:
    """Restore saved parameters, changing only the candidate-sampling seed."""
    if saved["dataset_version"] != "fish_intro_noisy_matched_v2":
        raise ValueError("Expected the saved Noisy Matched V2 baseline")
    expected = deepcopy(baseline.MODEL_CONFIG)
    expected.pop("aggregation_funcs")
    if saved["parameters"] != expected or saved["aggregations"] != ["sum"]:
        raise ValueError("Current defaults differ from the approved saved baseline")
    if saved["runs"] != list(baseline.MODEL_RUNS):
        raise ValueError("Saved ON/OFF switches differ from the baseline")
    config = deepcopy(saved["parameters"])
    config["aggregation_funcs"] = {"sum": np.sum}
    config["split_sampling_seed"] = seed
    return config


def effective_config(config: dict, enabled: bool) -> dict:
    """Resolve all constructor settings without fitting a model."""
    params = make_model(enabled, config).get_params(deep=False)
    params["aggregation_funcs"] = {"sum": "numpy.sum"}
    return params


def verify_inputs(source: Path, data_dir: Path) -> dict:
    """Require byte-identical baseline inputs and preserve the saved partition."""
    hashes = {}
    for name in baseline.DATASET_FILES:
        actual, saved = data_dir / name, source / name
        if not actual.is_file() or not saved.is_file():
            raise ValueError(f"Missing baseline input: {name}")
        if sha256(actual) != sha256(saved):
            raise ValueError(f"Baseline input differs: {name}")
        hashes[name] = sha256(actual)
    return hashes


def differences(expected, actual) -> dict:
    """Report exact and tolerance-based JSON differences; integers stay exact."""
    count, examples = 0, []

    def visit(left, right, path):
        nonlocal count
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right)):
                if key not in left or key not in right:
                    record(path + "." + key, left.get(key), right.get(key))
                else:
                    visit(left[key], right[key], path + "." + key)
        elif isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                record(path + ".length", len(left), len(right))
            for index, (a, b) in enumerate(zip(left, right)):
                visit(a, b, f"{path}[{index}]")
        elif isinstance(left, float) and isinstance(right, (int, float)):
            if not np.isclose(left, right, rtol=0, atol=ATOL):
                record(path, left, right)
        elif type(left) is not type(right) or left != right:
            record(path, left, right)

    def record(path, left, right):
        nonlocal count
        count += 1
        if len(examples) < 20:
            examples.append(dict(path=path, original=left, reproduced=right))

    visit(expected, actual, "root")
    return dict(exact_equal=expected == actual, within_tolerance=count == 0,
                differences_beyond_tolerance=count, examples=examples)


def load_predictions(folder: Path) -> pd.DataFrame:
    """Load saved scores with unique patient IDs and lossless float parsing."""
    frame = pd.read_csv(
        folder / "patient_predictions.csv", float_precision="round_trip",
        dtype={"patient_id": str, "documented": bool},
    ).set_index("patient_id", verify_integrity=True)
    if not np.isfinite(frame.score).all() or not frame.score.between(0, 1).all():
        raise ValueError("Predictions must be finite probabilities")
    if not frame.FA.isin([0, 1]).all():
        raise ValueError("Expected binary FA labels")
    return frame


def prediction_comparison(original: pd.DataFrame, reproduced: pd.DataFrame) -> dict:
    """Compare all patients and all metadata, not merely aggregate AUC."""
    same_ids = set(original.index) == set(reproduced.index)
    report = dict(patient_ids_equal=same_ids,
                  patient_order_equal=original.index.equals(reproduced.index))
    if not same_ids:
        return dict(report, passed=False)
    right = reproduced.reindex(original.index)
    metadata_equal = original.drop(columns="score").equals(right.drop(columns="score"))
    delta = right.score.to_numpy() - original.score.to_numpy()
    report.update(
        metadata_and_anchor_flags_equal=metadata_equal,
        n=len(original), exact_score_matches=int(np.sum(delta == 0)),
        scores_beyond_tolerance=int(np.sum(np.abs(delta) > ATOL)),
        max_absolute_score_difference=float(np.abs(delta).max()),
        passed=bool(metadata_equal and np.all(np.abs(delta) <= ATOL)
                    and report["patient_order_equal"]),
    )
    return report


def paired_auc(seed: int, predictions: Mapping[str, pd.DataFrame]) -> list[dict]:
    """Calculate two population-specific paired AUC differences from scores."""
    on, off = (predictions[run].sort_index() for run in RUNS)
    metadata = [name for name in on if name not in ("score", *PREDICTION_FLAGS)]
    if not on[metadata].equals(off[metadata]):
        raise ValueError("ON/OFF patient IDs, labels or cohort metadata differ")
    rows = []
    for population in POPULATIONS:
        mask = on.partition.eq("test")
        if population == "test_documented":
            mask &= on.documented
        labels = on.loc[mask, "FA"]
        if labels.nunique() != 2:
            raise ValueError(f"Both classes required for {population}")
        auc_on = float(roc_auc_score(labels, on.loc[mask, "score"]))
        auc_off = float(roc_auc_score(labels, off.loc[mask, "score"]))
        rows.append(dict(seed=seed, population=population, n=len(labels),
                         cases=int(labels.sum()), auc_on=auc_on, auc_off=auc_off,
                         delta=auc_on - auc_off))
    return rows


def temporal_structure(folder: Path, predictions: pd.DataFrame) -> dict:
    """Describe fish-relative one-month splits and unique held-out path usage.

    Presence is assessed anywhere in the tree. Coverage uses the fixed
    documented test denominator and requires an available evaluated window,
    even when that window contains no symptom events. Node IDs are local;
    comparisons across seeds use source feature and window semantics.
    """
    tree = read_json(folder / "tree.json")
    traces = read_json(folder / "anchor_trace.json")
    documented = set(predictions.index[
        predictions.partition.eq("test") & predictions.documented])
    if not documented:
        raise ValueError("Documented test population is empty")
    source_features = {
        row["anchor_name"]: row["feature"] for row in traces
        if row["anchor_name"]
    }
    matching = {
        symptom: {node["node_id"] for node in tree
                  if node["type"] == "split" and node["feature"] == symptom
                  and node["aggregation"] == "sum" and node["window"][1] == 1
                  and source_features.get(node["window"][0]) == "FishIntroduction"}
        for symptom in ("GI", "Vomiting")
    }
    used = {
        symptom: {row["patient_id"] for row in traces
                  if row["patient_id"] in documented
                  and row["node_id"] in nodes and row["evaluated_windows"]}
        for symptom, nodes in matching.items()
    }
    created = {row["patient_id"] for row in traces
               if row["patient_id"] in documented and row["fish_anchor_created"]}
    fish_used = {row["patient_id"] for row in traces
                 if row["patient_id"] in documented
                 and source_features.get(row.get("source_anchor")) == "FishIntroduction"
                 and row["evaluated_windows"]}
    symptom_used = {row["patient_id"] for row in traces
                    if row["patient_id"] in fish_used
                    and row["feature"] in ("GI", "Vomiting")
                    and source_features.get(row.get("source_anchor")) == "FishIntroduction"
                    and row["evaluated_windows"]}
    covered = used["GI"] | used["Vomiting"]
    root = tree[0]
    return dict(
        root_feature=root.get("feature"), root_window=root.get("window"),
        fish_anchor_creator_present="FishIntroduction" in source_features.values(),
        gi_one_month_present=bool(matching["GI"]),
        vomiting_one_month_present=bool(matching["Vomiting"]),
        documented_test_n=len(documented), fish_anchor_created_n=len(created),
        fish_anchor_creation_coverage=len(created) / len(documented),
        fish_anchor_used_n=len(fish_used),
        fish_anchor_usage_coverage=len(fish_used) / len(documented),
        fish_anchor_used_for_symptom_n=len(symptom_used),
        gi_one_month_used_n=len(used["GI"]),
        vomiting_one_month_used_n=len(used["Vomiting"]),
        either_one_month_used_n=len(covered),
        either_one_month_coverage=len(covered) / len(documented),
    )


def summarize_seeds(pairs: Mapping[int, Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare the approved five-seed tables from complete saved pairs only.

    This function never fits models. Seed 42 is excluded from the new-seed
    summary. The same cohort is required across seeds as well as within pairs.
    """
    if set(pairs) != set(PLANNED_SEEDS):
        raise ValueError("Require exactly the prespecified seeds 43--47")
    rows, structures, reference = [], [], None
    for seed in PLANNED_SEEDS:
        predictions = {run: load_predictions(pairs[seed] / run) for run in RUNS}
        metadata = predictions["dynamic_on"].drop(
            columns=["score", *PREDICTION_FLAGS]).sort_index()
        if reference is not None and not metadata.equals(reference):
            raise ValueError("Cohort differs across seeds")
        reference = metadata
        rows.extend(paired_auc(seed, predictions))
        for run in RUNS:
            config = read_json(pairs[seed] / run / "effective_configuration.json")
            expected = effective_config(
                model_config(read_json(BASELINE / "configuration.json"), seed),
                run == "dynamic_on",
            )
            if config != expected:
                raise ValueError(f"Unexpected effective configuration: {seed}/{run}")
            diagnostics = read_json(pairs[seed] / run / "diagnostics.json")
            if not diagnostics["mechanism_checks_passed"] or diagnostics["audit_failures"]:
                raise ValueError(f"Failed dynamic-anchor audit: {seed}/{run}")
            structures.append(dict(seed=seed, run=run, **temporal_structure(
                pairs[seed] / run, predictions[run])))
    table = pd.DataFrame(rows)
    summary = []
    for population, group in table.groupby("population", sort=False):
        row = dict(population=population, seeds=len(group),
                   positive_pairs=int(group.delta.gt(0).sum()),
                   minimum_delta=float(group.delta.min()))
        for metric in ("auc_on", "auc_off", "delta"):
            row[metric + "_mean"] = float(group[metric].mean())
            row[metric + "_sd"] = float(group[metric].std(ddof=1))
        summary.append(row)
    return table, pd.DataFrame(summary), pd.DataFrame(structures)


def plot_paired_auc(table: pd.DataFrame, destination: Path) -> None:
    """Plot centered seed dots and sample-SD summaries on a labeled truncated axis."""
    if destination.exists():
        raise FileExistsError(destination)
    figure, axes = plt.subplots(1, 2, figsize=(12, 6.5), sharey=True)
    colors = plt.get_cmap("tab10")(np.arange(len(PLANNED_SEEDS)))
    ticks = np.arange(45, 101, 5) / 100
    try:
        for axis, population, title in zip(
            axes, POPULATIONS, ("(A) Full test population", "(B) Documented fish introduction")
        ):
            group = table.loc[table.population.eq(population)].sort_values("seed")
            if tuple(group.seed) != PLANNED_SEEDS:
                raise ValueError("Plot requires all five prespecified seed pairs")
            values = group[["auc_off", "auc_on"]].to_numpy()
            for seed, pair, color in zip(group.seed, values, colors):
                label = f"Seed {seed} (OFF: {pair[0]:.4f}, ON: {pair[1]:.4f})"
                axis.scatter([0, 1], pair, color=color, s=36, alpha=.8,
                             label=label, zorder=3)
            axis.errorbar([0, 1], values.mean(axis=0),
                          yerr=values.std(axis=0, ddof=1), fmt="D", color="black",
                          capsize=4, label="Mean ± sample SD", zorder=5)
            axis.set_xticks([0, 1])
            axis.set_xticklabels(["OFF", "ON"])
            axis.set(xlim=(-.25, 1.25), ylim=(.45, 1), title=title, ylabel="ROC-AUC")
            axis.set_yticks(ticks)
            axis.tick_params(axis="y", labelleft=True)
            axis.set_axisbelow(True)
            axis.grid(axis="y", color="0.85", linewidth=.6, alpha=.7)
            axis.legend(loc="upper center", bbox_to_anchor=(.5, -.20), fontsize=8)
        figure.suptitle("Dynamic anchoring performance across candidate-sampling seeds")
        figure.subplots_adjust(left=.07, right=.98, top=.86, bottom=.36, wspace=.22)
        figure.text(.5, .025,
                    "Y-axes are truncated to 0.45–1.00 in both panels. "
                    "Black diamonds show mean ± sample SD across five seeds.",
                    ha="center", fontsize=9)
        figure.savefig(destination, dpi=300, bbox_inches="tight")
    finally:
        plt.close(figure)


def compare_checkpoint(source: Path, output: Path, config: dict) -> dict:
    """Compare configurations, all predictions, trees, metrics and path audits."""
    report = dict(absolute_tolerance=ATOL, relative_tolerance=0, runs={})
    for run in RUNS:
        original, reproduced = source / run, output / run
        pred = prediction_comparison(load_predictions(original), load_predictions(reproduced))
        checks = {
            "configuration": differences(effective_config(config, run == "dynamic_on"),
                                         read_json(reproduced / "effective_configuration.json")),
            "predictions": pred,
        }
        for name in ("metrics", "tree", "anchor_trace", "diagnostics"):
            left, right = (read_json(p / f"{name}.json") for p in (original, reproduced))
            if name == "diagnostics":
                checks["fit_runtime_seconds"] = dict(
                    original=left.pop("training_runtime_seconds"),
                    reproduced=right.pop("training_runtime_seconds"))
            checks[name] = differences(left, right)
        checks["auc"] = {
            population: dict(
                original=read_json(original / "metrics.json")[population]["roc_auc"],
                reproduced=read_json(reproduced / "metrics.json")[population]["roc_auc"],
            ) for population in POPULATIONS
        }
        checks["passed"] = pred["passed"] and all(
            checks[name]["within_tolerance"]
            for name in ("configuration", "metrics", "tree", "anchor_trace", "diagnostics")
        )
        report["runs"][run] = checks
    report["passed"] = all(run["passed"] for run in report["runs"].values())
    return report


def run_checkpoint(output: Path, source: Path = BASELINE, data_dir: Path = DATA) -> dict:
    """Run only seed 42 in a new directory, save comparison, then stop."""
    output, source, data_dir = map(Path, (output, source, data_dir))
    if output.exists():
        raise FileExistsError(output)
    for protected in (source.resolve(), data_dir.resolve()):
        if protected == output.resolve() or protected in output.resolve().parents:
            raise ValueError("Output must be separate from baseline results and data")
    saved = read_json(source / "configuration.json")
    config = model_config(saved, CHECKPOINT_SEED)
    input_hashes = verify_inputs(source, data_dir)
    data, patients, validation = baseline.load_dataset(data_dir)
    if list(data.feature_name.unique()) != validation["feature_order"]:
        raise ValueError("Saved feature ordering was not preserved")
    parameters = deepcopy(config)
    parameters.pop("aggregation_funcs")
    sources = set(saved["sha256"]) | {
        str(Path(__file__).resolve().relative_to(ROOT)),
        "experiments/controlled_dynamic_anchors_100.py",
    }
    provenance = dict(
        baseline_directory=str(source.resolve()), data_directory=str(data_dir.resolve()),
        seed=CHECKPOINT_SEED, planned_seeds=list(PLANNED_SEEDS),
        parameters=parameters, aggregations=["sum"], input_sha256=input_hashes,
        feature_order=list(data.feature_name.unique()),
        baseline_configuration_sha256=sha256(source / "configuration.json"),
        baseline_source_sha256=saved["sha256"],
        source_sha256={name: sha256(ROOT / name) for name in sorted(sources)},
        git_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        versions=dict(python=platform.python_version(), numpy=np.__version__,
                      pandas=pd.__version__, sklearn=sklearn.__version__,
                      matplotlib=matplotlib.__version__),
        scope="Seed-42 reproduction only; stop for approval regardless of outcome",
    )
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "configuration.json", provenance)
    write_json(output / "status.json", dict(status="running", seed=42))
    started = perf_counter()
    try:
        results = {}
        for run in RUNS:
            folder = output / run
            folder.mkdir()
            results[run] = run_model(run == "dynamic_on", data, patients, folder,
                                     model_config=deepcopy(config))
        write_json(output / "comparison.json", results)
        predictions = {run: load_predictions(output / run) for run in RUNS}
        pd.DataFrame(paired_auc(42, predictions)).to_csv(
            output / "paired_auc.csv", index=False)
        write_json(output / "temporal_structure.json", {
            run: temporal_structure(output / run, predictions[run]) for run in RUNS})
        report = compare_checkpoint(source, output, config)
        report["input_hashes_unchanged"] = verify_inputs(source, data_dir) == input_hashes
        report["passed"] &= report["input_hashes_unchanged"]
        write_json(output / "reproduction_report.json", report)
        write_json(output / "status.json", dict(
            status="checkpoint_complete_awaiting_approval", passed=report["passed"],
            seed=42, total_runtime_seconds=perf_counter() - started,
            new_seeds_executed=[],
        ))
        return report
    except Exception as error:
        write_json(output / "status.json", dict(
            status="failed", seed=42, error=f"{type(error).__name__}: {error}",
            total_runtime_seconds=perf_counter() - started, new_seeds_executed=[],
        ))
        raise



def atomic_json(path: Path, value: dict) -> None:
    """Publish a small manifest only after its entire JSON has been written."""
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    write_json(temporary, value)
    temporary.replace(path)


def protocol() -> dict:
    """Freeze inputs, source versions and settings against the passed checkpoint."""
    saved = read_json(BASELINE / "configuration.json")
    model_config(saved, CHECKPOINT_SEED)
    checkpoint = read_json(CHECKPOINT / "configuration.json")
    report = read_json(CHECKPOINT / "reproduction_report.json")
    status = read_json(CHECKPOINT / "status.json")
    if not report["passed"] or not status["passed"]:
        raise ValueError("A successful seed-42 checkpoint is required")
    inputs = verify_inputs(BASELINE, DATA)
    if inputs != checkpoint["input_sha256"]:
        raise ValueError("Inputs differ from the successful checkpoint")
    source_hashes = {
        name: sha256(ROOT / name) for name in checkpoint["source_sha256"]
    }
    this_source = str(Path(__file__).resolve().relative_to(ROOT))
    for name, digest in source_hashes.items():
        if name != this_source and digest != checkpoint["source_sha256"][name]:
            raise ValueError(f"Source changed since checkpoint: {name}")
    versions = dict(python=platform.python_version(), numpy=np.__version__,
                    pandas=pd.__version__, sklearn=sklearn.__version__,
                    matplotlib=matplotlib.__version__)
    if versions != checkpoint["versions"]:
        raise ValueError("Use the checkpoint's Python/package versions")
    return dict(
        schema_version=1, experiment="experiment_2_1_candidate_sampling_robustness",
        seeds=list(PLANNED_SEEDS), runs=list(RUNS),
        baseline_configuration=saved, input_sha256=inputs,
        feature_order=checkpoint["feature_order"], versions=versions,
        source_sha256=source_hashes,
        checkpoint_report_sha256=sha256(CHECKPOINT / "reproduction_report.json"),
        baseline_prediction_sha256={
            run: sha256(BASELINE / run / "patient_predictions.csv") for run in RUNS
        },
        sd_definition="sample SD across seeds (ddof=1), not patient uncertainty",
    )


def protocol_hash(value: dict) -> str:
    """Fingerprint the full immutable experiment protocol."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def protect_output(output: Path) -> Path:
    """Reject outputs inside existing datasets, baselines or the checkpoint."""
    output = output.resolve()
    for protected in (ROOT / "data", BASELINE, CHECKPOINT):
        protected = protected.resolve()
        if (output == protected or protected in output.parents
                or output in protected.parents):
            raise ValueError(f"Output overlaps protected location: {protected}")
    return output


@contextmanager
def output_lock(output: Path):
    """Hold a process lock; abrupt process exit releases it without cleanup."""
    with (output / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process is using this output directory") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def check_run(folder: Path, seed: int, run: str, frozen: dict) -> None:
    """Validate model artifacts by reading files only; never unpickle or predict."""
    for name in RUN_ARTIFACTS:
        if not (folder / name).is_file() or not (folder / name).stat().st_size:
            raise ValueError(f"Missing or empty run artifact: {folder / name}")
    expected = effective_config(
        model_config(frozen["baseline_configuration"], seed), run == "dynamic_on")
    if read_json(folder / "effective_configuration.json") != expected:
        raise ValueError(f"Configuration mismatch: {folder}")
    diagnostics = read_json(folder / "diagnostics.json")
    if not diagnostics["mechanism_checks_passed"] or diagnostics["audit_failures"]:
        raise ValueError(f"Dynamic-anchor audit failed: {folder}")
    frame = load_predictions(folder)
    original = load_predictions(BASELINE / run)
    metadata = [c for c in original if c not in ("score", *PREDICTION_FLAGS)]
    if not frame[metadata].equals(original[metadata]):
        raise ValueError(f"Patient metadata or row ordering changed: {folder}")
    metrics = read_json(folder / "metrics.json")
    for row in paired_auc(seed, dict(dynamic_on=frame, dynamic_off=frame)):
        saved = metrics[row["population"]]
        if (saved["n"] != row["n"] or saved["FA_cases"] != row["cases"]
                or not np.isclose(saved["roc_auc"], row["auc_on"], rtol=0, atol=ATOL)):
            raise ValueError(f"Saved AUC disagrees with predictions: {folder}")
    # Validate that the full JSON artifacts can be read before marking completion.
    read_json(folder / "tree.json")
    traces = read_json(folder / "anchor_trace.json")
    if not all(all(row["checks"].values()) for row in traces):
        raise ValueError(f"Prediction-path checks failed: {folder}")


def completed_run(folder: Path, seed: int, run: str, frozen: dict) -> None:
    """Reject a corrupt or incompatible completed run instead of retraining it."""
    marker = read_json(folder / "complete.json")
    expected = dict(seed=seed, run=run, protocol_sha256=protocol_hash(frozen))
    if any(marker.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Completion marker mismatch: {folder}")
    hashes = {name: sha256(folder / name) for name in RUN_ARTIFACTS}
    if marker["artifact_sha256"] != hashes:
        raise ValueError(f"Completed artifacts changed: {folder}; refusing to overwrite")
    check_run(folder, seed, run, frozen)


def prepare_run(folder: Path, seed: int, run: str, frozen: dict,
                retry_incomplete: bool) -> bool:
    """Return whether fitting is needed, recovering published completion safely."""
    partial = folder.with_name(folder.name + ".incomplete")
    if folder.exists():
        completed_run(folder, seed, run, frozen)
        if partial.exists():
            raise ValueError(f"Conflicting completed and incomplete runs: {folder}")
        return False
    if partial.exists():
        if (partial / "complete.json").exists():
            completed_run(partial, seed, run, frozen)
            partial.rename(folder)
            return False
        if not retry_incomplete:
            raise ValueError(
                f"Incomplete run: {partial}. Rerun with --retry-incomplete to "
                "preserve this attempt and restart only incomplete models.")
    return True


def fit_pending(folder: Path, seed: int, run: str, frozen: dict,
                data: pd.DataFrame, patients: pd.DataFrame) -> None:
    """Fit one explicitly requested missing run, never overwriting an attempt."""
    partial = folder.with_name(folder.name + ".incomplete")
    folder.parent.mkdir(parents=True, exist_ok=True)
    if partial.exists():
        history = folder.parent / "interrupted"
        history.mkdir(exist_ok=True)
        partial.rename(history / f"{run}_{uuid4().hex}")
    partial.mkdir()
    started = perf_counter()
    try:
        config = model_config(frozen["baseline_configuration"], seed)
        print(f"Seed {seed}: {run}", flush=True)
        run_model(run == "dynamic_on", data, patients, partial, model_config=config)
        check_run(partial, seed, run, frozen)
        atomic_json(partial / "complete.json", dict(
            seed=seed, run=run, protocol_sha256=protocol_hash(frozen),
            artifact_sha256={name: sha256(partial / name) for name in RUN_ARTIFACTS},
            completed_utc=datetime.now(timezone.utc).isoformat(),
            total_runtime_seconds=perf_counter() - started,
        ))
        partial.rename(folder)
    except BaseException as error:
        if partial.exists():
            atomic_json(partial / "interruption.json", dict(
                seed=seed, run=run, error=f"{type(error).__name__}: {error}",
                elapsed_seconds=perf_counter() - started,
            ))
        raise


def write_summary(output: Path, frozen: dict) -> Path:
    """Publish thesis tables/figure from ten verified runs; preserve older summaries."""
    pairs = {seed: output / "seeds" / f"seed_{seed}" for seed in PLANNED_SEEDS}
    for seed, pair in pairs.items():
        for run in RUNS:
            completed_run(pair / run, seed, run, frozen)
    table, summary, temporal = summarize_seeds(pairs)
    destination = output / f".summary_{uuid4().hex}"
    destination.mkdir()
    table.to_csv(destination / "auc_per_seed.csv", index=False)
    summary.to_csv(destination / "auc_summary.csv", index=False)
    compact = []
    for seed in PLANNED_SEEDS:
        row = dict(seed=str(seed))
        for population, prefix in zip(POPULATIONS, ("full", "documented")):
            values = table.loc[(table.seed == seed) & (table.population == population)].iloc[0]
            for metric in ("auc_on", "auc_off", "delta"):
                row[f"{prefix}_{metric}"] = f"{values[metric]:.4f}"
        compact.append(row)
    row = dict(seed="Mean ± SD")
    for population, prefix in zip(POPULATIONS, ("full", "documented")):
        values = summary.loc[summary.population == population].iloc[0]
        for metric in ("auc_on", "auc_off", "delta"):
            row[f"{prefix}_{metric}"] = (
                f"{values[metric + '_mean']:.4f} ± {values[metric + '_sd']:.4f}")
    pd.DataFrame([*compact, row]).to_csv(destination / "results_table.csv", index=False)
    temporal.to_csv(destination / "temporal_per_seed.csv", index=False)
    stability = []
    for run, group in temporal.groupby("run", sort=False):
        row = dict(run=run, seeds=len(group),
                   documented_test_n=int(group.documented_test_n.iloc[0]))
        for flag in ("fish_anchor_creator_present", "gi_one_month_present",
                     "vomiting_one_month_present"):
            row[flag + "_seeds"] = int(group[flag].sum())
        for column in ("fish_anchor_created_n", "fish_anchor_used_n",
                       "fish_anchor_used_for_symptom_n", "gi_one_month_used_n",
                       "vomiting_one_month_used_n", "either_one_month_used_n",
                       "fish_anchor_creation_coverage", "fish_anchor_usage_coverage",
                       "either_one_month_coverage"):
            row[column + "_mean"] = float(group[column].mean())
            row[column + "_sd"] = float(group[column].std(ddof=1))
        stability.append(row)
    pd.DataFrame(stability).to_csv(destination / "temporal_stability_summary.csv", index=False)
    for extension in ("png", "pdf"):
        plot_paired_auc(table, destination / f"paired_roc_auc.{extension}")
    write_json(destination / "summary_manifest.json", dict(
        protocol_sha256=protocol_hash(frozen), seeds=list(PLANNED_SEEDS),
        sd_definition=frozen["sd_definition"],
        coverage_definition="Unique documented test patients; available anchor required; "
                            "an evaluated window may contain zero events",
        completed_run_markers={
            f"{seed}/{run}": sha256(pair / run / "complete.json")
            for seed, pair in pairs.items() for run in RUNS
        },
    ))
    target = output / "summary"
    if target.exists():
        history = output / "history"
        history.mkdir(exist_ok=True)
        target.rename(history / f"summary_{uuid4().hex}")
    destination.rename(target)
    return target


def run_sweep(output: Path = OUTPUT, retry_incomplete: bool = False) -> Path:
    """Execute the five manual seed pairs, or resume them, then aggregate."""
    output = protect_output(Path(output))
    frozen = protocol()
    if output.exists() and not (output / "protocol.json").is_file():
        raise ValueError(
            "Existing output lacks an Experiment 2.1 protocol; use a new directory")
    output.mkdir(parents=True, exist_ok=True)
    with output_lock(output):
        path = output / "protocol.json"
        if path.exists():
            if read_json(path) != frozen:
                raise ValueError(
                    "Protocol, inputs, software or source changed; refusing mixed runs")
        else:
            atomic_json(path, frozen)
        pending = []
        # Inspect every existing run before starting any new fits.
        for seed in PLANNED_SEEDS:
            for run in RUNS:
                folder = output / "seeds" / f"seed_{seed}" / run
                if prepare_run(folder, seed, run, frozen, retry_incomplete):
                    pending.append((folder, seed, run))
                else:
                    print(f"Verified complete; skip seed {seed}: {run}", flush=True)
        if pending:
            data, patients, validation = baseline.load_dataset(DATA)
            if (list(data.feature_name.unique()) != frozen["feature_order"]
                    or validation["feature_order"] != frozen["feature_order"]):
                raise ValueError("Feature order differs from the checkpoint")
            for folder, seed, run in pending:
                fit_pending(folder, seed, run, frozen, data, patients)
        if protocol() != frozen:
            raise ValueError("Inputs or source changed during execution; refusing aggregation")
        return write_summary(output, frozen)


def summarize_only(output: Path = OUTPUT) -> Path:
    """Regenerate derived summaries without running or loading any model."""
    output = protect_output(Path(output))
    frozen = read_json(output / "protocol.json")
    if frozen != protocol():
        raise ValueError("Protocol, inputs, software or source changed")
    with output_lock(output):
        return write_summary(output, frozen)


def main() -> None:
    """Expose manual training/resumption and strictly saved-artifact aggregation."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run/resume seeds 43--47; never seed 42")
    run.add_argument("--output", type=Path, default=OUTPUT)
    run.add_argument("--retry-incomplete", action="store_true",
                     help="Archive incomplete attempts and restart only those models")
    summary = commands.add_parser("summarize", help="Rebuild summary from saved runs only")
    summary.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.command == "run":
        destination = run_sweep(args.output, args.retry_incomplete)
    else:
        destination = summarize_only(args.output)
    print(f"Summary saved to {destination}")


if __name__ == "__main__":
    main()
