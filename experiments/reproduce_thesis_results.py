"""Audit saved thesis results or explicitly refit the recorded experiments.

Run from the repository root::

    python -B -m experiments.reproduce_thesis_results check-saved
    python -B -m experiments.reproduce_thesis_results rerun --experiment cart \
        --output experiments/results/reproduction_example

``check-saved`` never fits, predicts or unpickles models. It writes only an
explicitly requested report; dataset pickles are read as inputs.
``rerun`` uses existing runners and adjacent directories with a shared prefix.
The central directory contains logs and a JSON report.
Exit codes: 0 = passed, 1 = mismatch/error, 2 = partial reference or dry run.
No data generation, implicit model conversion, or reference replacement occurs.

To run experiment 2 run the following command: 
python -B experiments/reproduce_thesis_results.py rerun --experiment experiment_2 --output experiments/results/reproduction_exp2_seed42
To run experiment 1 run the following command: 
python -B experiments/reproduce_thesis_results.py rerun --experiment experiment_1 --output experiments/results/reproduction_exp1_seed42"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
from functools import lru_cache
from importlib.metadata import version
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Optional, Sequence
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import experiment_1_fa_vs_gi as exp1
from experiments import experiment_1_1_gi_without_now as gi
from experiments import experiment_2_noisy_v2 as exp2
from experiments.experiment_2_fish_introduction_dynamic_anchors import evaluate
from project_paths import portable_metadata, relative_path, resolve_path

RESULTS = ROOT / "experiments/results"
REFERENCES = {
    "experiment_1": gi.BASELINE,
    "gi_gain0": RESULTS / "experiment_1_1_gi_without_now_A0.4_minGAIN0",
    "gi_gain0005": RESULTS
    / "experiment_1_1_gi_without_now_A0.4_minGAIN0.0005",
    "experiment_2": exp2.DEFAULT_OUTPUT,
    "experiment_2_1": RESULTS / "experiment_2_1_candidate_sampling_robustness",
    "cart": RESULTS
    / ("experiment_2_cart_baseline/run_20261014T110752808924Z_5f1eab64"),
}
ATOL = 1e-12
GAINS = {"gi_gain0": 0.0, "gi_gain0005": 0.0005}


def read_json(path: Path) -> Any:
    """Read a JSON artifact without executing serialized model code."""
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    """Hash file contents without loading large files into memory."""
    return exp1.sha256_file(path)


def require(condition: bool, message: str) -> None:
    """Reject a failed scientific or output-isolation check."""
    if not condition:
        raise ValueError(message)


def compare_values(expected: Any, actual: Any, name: str = "value") -> None:
    """Compare JSON values, keeping counts exact and floats within ATOL."""
    if isinstance(expected, dict):
        require(
            isinstance(actual, dict) and expected.keys() == actual.keys(),
            f"{name}: dictionary keys differ",
        )
        for key in expected:
            compare_values(expected[key], actual[key], f"{name}.{key}")
    elif isinstance(expected, (list, tuple)):
        require(
            isinstance(actual, (list, tuple)) and len(expected) == len(actual),
            f"{name}: sequence lengths differ",
        )
        for index, (left, right) in enumerate(zip(expected, actual)):
            compare_values(left, right, f"{name}[{index}]")
    elif isinstance(expected, (float, np.floating)):
        require(
            isinstance(actual, (float, int, np.number))
            and not isinstance(actual, (bool, np.bool_))
            and np.isfinite(expected)
            and np.isfinite(actual)
            and np.isclose(expected, actual, rtol=0, atol=ATOL),
            f"{name}: expected {expected!r}, obtained {actual!r}",
        )
    else:
        require(
            expected == actual and type(expected) is type(actual),
            f"{name}: expected {expected!r}, obtained {actual!r}",
        )


def read_table(path: Path, *, predictions: bool = False) -> pd.DataFrame:
    """Read CSVs with lossless float parsing and string patient IDs."""
    frame = pd.read_csv(
        path, float_precision="round_trip", dtype={"patient_id": str}
    )
    if predictions:
        frame = frame.set_index("patient_id", verify_integrity=True)
    return frame


def compare_tables(expected: pd.DataFrame, actual: pd.DataFrame) -> None:
    """Compare labels, table order, missingness and numeric values."""
    pd.testing.assert_frame_equal(
        expected,
        actual,
        check_dtype=False,
        check_exact=False,
        rtol=0,
        atol=ATOL,
    )


def validate_scores(frame: pd.DataFrame, label: str, score: str) -> None:
    """Require unique patients, binary labels and finite scores in [0, 1]."""
    require(
        len(frame) > 0 and frame.index.is_unique,
        "Empty or duplicate prediction patients",
    )
    require(frame[label].isin([0, 1]).all(), "Labels are not binary 0/1")
    require(
        pd.api.types.is_numeric_dtype(frame[score]),
        "Score columns must be numeric",
    )
    require(
        np.isfinite(frame[score]).all() and frame[score].between(0, 1).all(),
        "Scores must be finite and in [0, 1]",
    )


def compare_predictions(
    reference: Path, candidate: Path, score: str = "score"
) -> dict:
    """Compare individual scores and all associated patient metadata."""
    left, right = (
        read_table(p, predictions=True) for p in (reference, candidate)
    )
    require(
        left.index.equals(right.index), "Prediction IDs or row order differ"
    )
    compare_tables(left.drop(columns=score), right.drop(columns=score))
    require(np.isfinite(right[score]).all(), "Nonfinite reproduced scores")
    delta = np.abs(left[score].to_numpy() - right[score].to_numpy())
    count = int(np.sum(delta > ATOL))
    require(
        count == 0, f"{count} patient scores differ; max delta={delta.max()}"
    )
    return dict(
        patients=len(left),
        exact_score_matches=int(np.sum(delta == 0)),
        maximum_absolute_score_difference=float(delta.max()),
    )


@lru_cache(maxsize=1)
def experiment1_inputs() -> tuple:
    """Validate the recorded cohort and recreate its seed-42 shared split."""
    config = read_json(REFERENCES["experiment_1"] / "config.json")
    compare_values(
        config["model_hyperparameters"],
        exp1.MODEL_HYPERPARAMETERS,
        "Experiment 1 model parameters",
    )
    for key, value in (
        ("random_state", exp1.RANDOM_STATE),
        ("split_sampling_seed", exp1.SPLIT_SAMPLING_SEED),
        ("test_size", exp1.TEST_SIZE),
        ("decision_threshold", exp1.DECISION_THRESHOLD),
    ):
        compare_values(config[key], value, key)
    require(
        exp1.RANDOM_STATE == exp1.SPLIT_SAMPLING_SEED == 42,
        "The recorded Experiment 1 protocol requires both seeds to be 42",
    )
    for key, digest in config["input_sha256"].items():
        path = resolve_path(config["input_paths"][key])
        require(
            path.resolve() == exp1.INPUT_PATHS[key].resolve(),
            f"Experiment 1 input location differs: {key}",
        )
        require(
            sha256(path) == digest, f"Experiment 1 input hash differs: {key}"
        )
    data, labels = exp1.load_inputs()
    _, _, split = exp1.create_shared_split(data, labels)
    compare_tables(
        read_table(REFERENCES["experiment_1"] / "shared_patient_split.csv"),
        split,
    )
    return config, data, labels, split


def check_outcome(
    directory: Path,
    outcome: str,
    split: pd.DataFrame,
    data: pd.DataFrame,
    missing: list,
) -> dict:
    """Recalculate held-out metrics against independent cohort labels/split."""
    frame = read_table(directory / "test_predictions.csv", predictions=True)
    validate_scores(frame, "label", "score")
    test = split.loc[split.split.eq("test")].set_index("patient_id")
    order = data.loc[data.patient_id.isin(test.index), "patient_id"].unique()
    require(
        frame.index.tolist() == list(order), "Wrong TEST membership or order"
    )
    require(
        np.array_equal(frame.label, test.loc[order, f"{outcome}_label"]),
        "Prediction labels differ from the included cohort",
    )
    performance, predictions = exp1.evaluate_scores(frame.label, frame.score)
    require(
        np.array_equal(frame.prediction, predictions),
        "Stored class predictions differ from the recorded threshold",
    )
    metric_path = directory / "metrics_and_temporal_summary.json"
    if metric_path.is_file():
        compare_values(
            read_json(metric_path)["performance"],
            performance,
            f"{outcome} TEST metrics",
        )
    else:
        missing.append(relative_path(metric_path))
    if not (directory / "tree.json").is_file():
        missing.append(relative_path(directory / "tree.json"))
    return dict(
        patients=len(frame),
        roc_auc=performance["roc_auc"],
        recalculated_performance=performance,
    )


def check_experiment1(study: str, directory: Path, missing: list) -> dict:
    """Check FA/GI or either NOW ablation using the original shared cohort."""
    config, data, _, split = experiment1_inputs()
    compare_tables(split, read_table(directory / "shared_patient_split.csv"))
    if study == "experiment_1":
        saved = read_json(directory / "config.json")
        compare_values(
            config["model_hyperparameters"], saved["model_hyperparameters"]
        )
        outcomes = ("fa", "gi")
    else:
        saved = read_json(directory / "experiment_config.json")
        with gi.configured_run(GAINS[study]):
            expected = gi.ablated_parameters(config)
        compare_values(
            expected, saved["model_hyperparameters"], "GI parameters"
        )
        outcomes = ("gi",)
    compare_values(
        list(exp1.aggregation_functions()),
        saved.get("effective_aggregations", saved["aggregations"]),
        "Actual aggregations",
    )
    for key, digest in saved["input_sha256"].items():
        require(
            sha256(resolve_path(saved["input_paths"][key])) == digest,
            f"Input hash differs: {key}",
        )
    metrics = {
        outcome: check_outcome(
            directory / outcome, outcome, split, data, missing
        )
        for outcome in outcomes
    }
    if study == "experiment_1":
        comparison = read_json(directory / "comparison_summary.json")[
            "outcomes"
        ]
        for outcome, values in metrics.items():
            actual = values["recalculated_performance"]
            compare_values(
                {key: comparison[outcome][key] for key in actual},
                actual,
                f"{outcome} comparison summary",
            )
    elif (directory / "comparison_with_original_gi.csv").is_file():
        comparison = read_table(directory / "comparison_with_original_gi.csv")
        compare_values(
            float(comparison.iloc[1].roc_auc),
            metrics["gi"]["roc_auc"],
            "GI ablation comparison AUC",
        )
    return metrics


@lru_cache(maxsize=1)
def experiment2_inputs() -> tuple:
    """Validate the included V2 cohort once, retaining its block partition."""
    return exp2.load_dataset(exp2.DEFAULT_DATA_DIR)


def check_fish_model(directory: Path) -> dict:
    """Recalculate both populations' AUC, Brier and calibration from saved
    scores.
    """
    _, patients, _ = experiment2_inputs()
    frame = pd.read_csv(
        directory / "patient_predictions.csv",
        float_precision="round_trip",
        dtype={"patient_id": str, "documented": bool},
        parse_dates=["birth", "followup_end", "fish_time", "a", "b"],
    ).set_index("patient_id", verify_integrity=True)
    validate_scores(frame, "FA", "score")
    compare_tables(
        patients.set_index("patient_id"),
        frame[patients.columns.drop("patient_id")],
    )
    test = frame.loc[frame.partition.eq("test")]
    metrics = dict(
        test_overall=evaluate(test),
        test_documented=evaluate(test.loc[test.documented]),
    )
    compare_values(
        read_json(directory / "metrics.json"), metrics, "V2 metrics"
    )
    diagnostics = read_json(directory / "diagnostics.json")
    require(
        diagnostics["mechanism_checks_passed"]
        and not diagnostics["audit_failures"],
        "Dynamic-anchor audit failed",
    )
    traces = read_json(directory / "anchor_trace.json")
    require(
        all(all(row["checks"].values()) for row in traces),
        "A stored anchor trace has failed checks",
    )
    read_json(directory / "tree.json")
    return {
        name: {key: value[key] for key in ("n", "roc_auc", "brier")}
        for name, value in metrics.items()
    }


def check_experiment2(directory: Path) -> dict:
    """Check recorded ON/OFF settings and both saved models."""
    config = read_json(directory / "configuration.json")
    parameters = {
        key: value
        for key, value in exp2.MODEL_CONFIG.items()
        if key != "aggregation_funcs"
    }
    compare_values(parameters, config["parameters"], "V2 model parameters")
    compare_values(
        list(exp2.MODEL_CONFIG["aggregation_funcs"]),
        config["aggregations"],
        "V2 aggregations",
    )
    compare_values(list(exp2.MODEL_RUNS), config["runs"], "ON/OFF switches")
    require(
        config["dataset_version"] == "fish_intro_noisy_matched_v2",
        "Wrong Experiment 2 dataset version",
    )
    return {
        run["name"]: check_fish_model(directory / run["name"])
        for run in exp2.MODEL_RUNS
    }


def check_robustness(directory: Path) -> dict:
    """Reuse frozen-protocol and completed-run checks without writing."""
    from experiments import (
        experiment_2_1_candidate_sampling_robustness as robust,
    )

    frozen = read_json(directory / "protocol.json")
    seeds = robust.validate_saved_protocol(frozen)
    for run in robust.RUNS:
        check_fish_model(robust.CHECKPOINT / run)
        compare_predictions(
            robust.BASELINE / run / "patient_predictions.csv",
            robust.CHECKPOINT / run / "patient_predictions.csv",
        )
    pairs = {
        seed: directory / "seeds" / f"seed_{seed}"
        for seed in seeds
    }
    for seed, pair in pairs.items():
        for run in robust.RUNS:
            robust.completed_run(pair / run, seed, run, frozen)
            check_fish_model(pair / run)
    per_seed, summary, temporal = robust.summarize_seeds(pairs, seeds=seeds)
    for name, table in (
        ("auc_per_seed.csv", per_seed),
        ("auc_summary.csv", summary),
        ("temporal_per_seed.csv", temporal),
    ):
        serialized = pd.read_csv(
            io.StringIO(table.to_csv(index=False)),
            float_precision="round_trip",
        )
        compare_tables(read_table(directory / "summary" / name), serialized)
    return dict(
        completed_runs=len(seeds) * len(robust.RUNS),
        sampling_seeds=list(seeds),
        checkpoint_models_verified=2,
        seed42="separate recorded checkpoint",
        auc_summary=summary.to_dict(orient="records"),
    )


def check_cart(directory: Path) -> dict:
    """Check CART inputs, TEST labels/scores and reported TEST AUCs."""
    from experiments import experiment_2_cart_baseline as cart

    matrix, info, train, test, reference, config = cart.prepare_inputs(
        cart.RECORDED_RUN
    )
    metadata = read_json(directory / "run_metadata.json")
    expected = {
        key: config["parameters"][key]
        for key in (
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "random_state",
        )
    }
    expected["criterion"] = "gini"
    compare_values(
        expected, metadata["model_parameters"], "Final CART parameters"
    )
    compare_values(
        list(matrix.columns), metadata["feature_columns"], "CART features"
    )
    compare_values(
        list(train), metadata["train_patient_ids"], "CART TRAIN IDs"
    )
    compare_values(list(test), metadata["test_patient_ids"], "CART TEST IDs")
    frame = read_table(directory / "test_predictions.csv", predictions=True)
    validate_scores(frame, "FA_label", "CART_probability")
    require(frame.index.tolist() == list(test), "CART TEST IDs/order differ")
    require(
        np.array_equal(frame.FA_label, info.loc[test, "FA"]),
        "CART TEST labels differ from the cohort",
    )
    np.testing.assert_allclose(
        frame.TAT_dynamic_on_probability,
        reference.loc[test, "score"],
        rtol=0,
        atol=ATOL,
    )
    results = read_table(directory / "results.csv")
    compare_values(
        metadata["results"],
        results.to_dict(orient="records"),
        "CART metadata/results consistency",
    )
    table = results.set_index("model", verify_integrity=True)
    metrics = {}
    for model, column in (
        ("CART", "CART_probability"),
        ("TAT_dynamic_on_saved", "TAT_dynamic_on_probability"),
    ):
        auc = float(roc_auc_score(frame.FA_label, frame[column]))
        compare_values(float(table.loc[model, "test_roc_auc"]), auc, model)
        metrics[model] = auc
    return dict(
        test_patients=len(frame),
        test_roc_auc=metrics,
        note=(
            "Saved CSV supports TEST recalculation; a refit also "
            "compares TRAIN AUC."
        ),
    )


def reference_integrity(directory: Path, missing: list) -> int:
    """Check original hashes and report absent reference artifacts."""
    ledger = read_json(ROOT / "docs/preserved_artifacts.json")[
        "unchanged_byte_hashes"
    ]
    count = 0
    prefix = relative_path(directory) + "/"
    for name, digest in ledger.items():
        if not name.startswith(prefix):
            continue
        path = ROOT / name
        if not path.is_file():
            missing.append(name)
        else:
            require(
                sha256(path) == digest,
                f"Original artifact hash differs: {name}",
            )
            count += 1
    return count


def check_study(study: str, directory: Optional[Path] = None) -> dict:
    """Return a per-experiment report, distinguishing partial archives from
    PASS.
    """
    reference = REFERENCES[study]
    path = directory or reference
    result = dict(
        experiment=study, directory=relative_path(path), status="failed"
    )
    missing: list[str] = []
    try:
        if directory is None:
            result["original_artifact_hashes_checked"] = reference_integrity(
                path, missing
            )
        with redirect_stdout(io.StringIO()):
            if study == "experiment_1" or study in GAINS:
                result["metrics"] = check_experiment1(study, path, missing)
            elif study == "experiment_2":
                result["metrics"] = check_experiment2(path)
            elif study == "experiment_2_1":
                result["metrics"] = check_robustness(path)
            else:
                result["metrics"] = check_cart(path)
        require(
            not missing or directory is None,
            "Reproduced run is missing expected metric/tree exports",
        )
        result["status"] = "partial" if missing else "passed"
    except (
        OSError,
        ValueError,
        AssertionError,
        KeyError,
        ImportError,
    ) as error:
        result["error"] = str(error).replace(str(ROOT), ".")
    if missing:
        key = (
            "missing_reference_artifacts"
            if directory is None
            else "missing_artifacts"
        )
        result[key] = sorted(set(missing))
    return result


def scientific_summary(value: dict) -> dict:
    """Exclude output provenance and runtime measurements from comparison."""
    return {
        key: item
        for key, item in value.items()
        if key
        not in ("artifacts", "runtime_seconds", "training_runtime_seconds")
    }


def compare_artifacts(
    reference: Path, candidate: Path, pairs: list, missing: list
) -> list:
    """Compare scientific exports and report missing references."""
    reports = []
    for relative, score in pairs:
        left, right = reference / relative, candidate / relative
        if not left.is_file():
            missing.append(relative_path(left))
            continue
        require(right.is_file(), f"Reproduced artifact missing: {relative}")
        if score:
            detail = compare_predictions(left, right, score)
        elif left.suffix == ".csv":
            compare_tables(read_table(left), read_table(right))
            detail = dict(equal_within_tolerance=True)
        else:
            a, b = read_json(left), read_json(right)
            if left.name == "metrics_and_temporal_summary.json":
                a, b = scientific_summary(a), scientific_summary(b)
            compare_values(a, b, relative)
            detail = dict(equal_within_tolerance=True)
        reports.append(dict(artifact=relative, **detail))
    return reports


def compare_reproduction(study: str, candidate: Path) -> dict:
    """Audit a fresh run and compare its predictions and trees."""
    result = check_study(study, candidate)
    if result["status"] == "failed":
        return result
    missing = list(result.get("missing_reference_artifacts", []))
    pairs = []
    if study == "experiment_1" or study in GAINS:
        pairs.append(("shared_patient_split.csv", None))
        outcomes = ("fa", "gi") if study == "experiment_1" else ("gi",)
        for outcome in outcomes:
            pairs.extend(
                (f"{outcome}/{name}", score)
                for name, score in (
                    ("test_predictions.csv", "score"),
                    ("tree.json", None),
                    ("metrics_and_temporal_summary.json", None),
                    ("temporal_explainability/temporal_path_usage.csv", None),
                )
            )
    elif study == "cart":
        pairs = [
            ("test_predictions.csv", "CART_probability"),
            ("results.csv", None),
        ]
    else:
        if study == "experiment_2":
            prefixes = [""]
        else:
            from experiments import (
                experiment_2_1_candidate_sampling_robustness as robust,
            )

            prefixes = [f"seeds/seed_{seed}/" for seed in robust.PLANNED_SEEDS]
        for prefix in prefixes:
            for run in ("dynamic_on", "dynamic_off"):
                pairs.extend(
                    (f"{prefix}{run}/{name}", score)
                    for name, score in (
                        ("patient_predictions.csv", "score"),
                        ("tree.json", None),
                        ("metrics.json", None),
                        ("anchor_trace.json", None),
                    )
                )
    try:
        result["comparisons"] = compare_artifacts(
            REFERENCES[study], candidate, pairs, missing
        )
        result["status"] = "partial" if missing else "passed"
    except (OSError, ValueError, AssertionError, KeyError) as error:
        result.update(
            status="failed", error=str(error).replace(str(ROOT), ".")
        )
    if missing:
        result["missing_reference_artifacts"] = sorted(set(missing))
    return result


def selected_studies(name: str) -> tuple:
    """Expand the documented all-experiments selection in execution order."""
    return tuple(REFERENCES) if name == "all" else (name,)


def output_plan(output: Path, studies: Sequence[str]) -> dict:
    """Reserve new flat destinations for existing runners."""
    output = resolve_path(output).absolute()
    require(not output.is_symlink(), "Output must not be a symlink")
    require(
        output.parent.resolve() == RESULTS,
        "--output must be directly under experiments/results",
    )
    output = output.parent.resolve() / output.name
    paths = {
        name: output.with_name(f"{output.name}_{name}") for name in studies
    }
    for path in (output, *paths.values()):
        require(
            not path.exists() and not path.is_symlink(),
            f"Output exists; refusing overwrite: {relative_path(path)}",
        )
    return paths


def runner_command(study: str, destination: Path) -> Optional[list]:
    """Build existing CLI calls, using the Python entry point for CART."""
    modules = {
        "experiment_1": "experiment_1_fa_vs_gi",
        "experiment_2": "experiment_2_noisy_v2",
        "experiment_2_1": "experiment_2_1_candidate_sampling_robustness",
    }
    if study == "cart":
        return None
    if study in GAINS:
        args = [
            "-m",
            "experiments.experiment_1_1_gi_without_now",
            "--run",
            "--min-gain",
            f"{GAINS[study]:g}",
        ]
    else:
        args = ["-m", f"experiments.{modules[study]}"]
        if study == "experiment_2_1":
            args.append("run")
    return [sys.executable, "-B", *args, "--output", str(destination)]


def run_study(study: str, destination: Path, log: Path) -> Path:
    """Invoke an existing runner with its scientific and safety checks."""
    require(
        not destination.exists() and not destination.is_symlink(),
        "Runner destination already exists; refusing overwrite",
    )
    command = runner_command(study, destination)
    with log.open("x", encoding="utf-8") as handle:
        if command is not None:
            subprocess.run(
                command,
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
            return destination
        from experiments import experiment_2_cart_baseline as cart

        with (
            redirect_stdout(handle),
            patch.object(cart, "OUTPUT_ROOT", destination),
        ):
            cart.main(cart.RECORDED_RUN)
    children = [path for path in destination.iterdir() if path.is_dir()]
    require(
        len(children) == 1, "Expected exactly one newly generated CART run"
    )
    return children[0]


def exit_status(results: list) -> int:
    """Prioritize failure and never label a partial audit PASS."""
    if any(result["status"] == "failed" for result in results):
        return 1
    return 2 if any(result["status"] == "partial" for result in results) else 0


def write_report(path: Path, report: dict) -> None:
    """Create a new JSON report, refusing existing files and symlinks."""
    with path.open("x", encoding="utf-8") as handle:
        json.dump(portable_metadata(report), handle, indent=2, allow_nan=False)
        handle.write("\n")


def report_context() -> dict:
    """Record the verification environment and code without changing
    provenance.
    """
    return dict(
        generated_utc=datetime.now(timezone.utc).isoformat(),
        versions={
            name: version(name)
            for name in ("numpy", "pandas", "scikit-learn", "matplotlib")
        },
        python=platform.python_version(),
        absolute_tolerance=ATOL,
        verification_script_sha256=sha256(Path(__file__)),
    )


def save_progress(output: Path, report: dict) -> None:
    """Atomically update only this invocation's newly reserved report
    directory.
    """
    temporary = output / "report.json.tmp"
    temporary.write_text(
        json.dumps(portable_metadata(report), indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(output / "report.json")


def rerun(studies: Sequence[str], output: Path, dry_run: bool = False) -> dict:
    """Preflight references and refit only on explicit request."""
    plan = output_plan(output, studies)
    checks = [check_study(study) for study in studies]
    report = dict(
        mode="rerun",
        **report_context(),
        output_plan={key: relative_path(path) for key, path in plan.items()},
        reference_checks=checks,
        planned_studies=list(studies),
        results=[],
    )
    if exit_status(checks) == 1:
        report.update(status="failed", results=checks)
        return report
    if dry_run:
        report.update(status="dry_run", results=checks)
        return report
    output = RESULTS / resolve_path(output).name
    output.mkdir(exist_ok=False)
    reference_hashes = {
        str(path): sha256(path)
        for study in studies
        for path in REFERENCES[study].rglob("*")
        if path.is_file()
    }
    report["status"] = "running"
    save_progress(output, report)
    for study in studies:
        log_name = relative_path(output / f"{study}.log")
        print(
            f"Reproducing {study}; log: {log_name}",
            file=sys.stderr,
            flush=True,
        )
        try:
            candidate = run_study(study, plan[study], output / f"{study}.log")
            result = compare_reproduction(study, candidate)
        except Exception as error:
            result = dict(
                experiment=study,
                status="failed",
                error=str(error).replace(str(ROOT), "."),
            )
        report["results"].append(result)
        # A partially completed batch must never look like a completed PASS.
        report["status"] = "running"
        save_progress(output, report)
    unchanged = all(
        Path(path).is_file() and sha256(Path(path)) == digest
        for path, digest in reference_hashes.items()
    )
    report["references_unchanged"] = unchanged
    if not unchanged:
        report["results"].append(
            dict(
                experiment="reference_integrity",
                status="failed",
                error="A reference artifact changed during execution",
            )
        )
    report["status"] = {0: "passed", 1: "failed", 2: "partial"}[
        exit_status(report["results"])
    ]
    save_progress(output, report)
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Require an explicit audit or training mode; audits are read-only by
    default.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser(
        "check-saved", help="Read-only metrics and integrity audit"
    )
    audit.add_argument(
        "--experiment", choices=(*REFERENCES, "all"), default="all"
    )
    audit.add_argument("--report", type=Path, help="Optional NEW report file")
    train = commands.add_parser(
        "rerun", help="Refit with recorded seeds/configuration"
    )
    train.add_argument(
        "--experiment", choices=(*REFERENCES, "all"), default="all"
    )
    train.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New report directory directly under experiments/results",
    )
    train.add_argument(
        "--dry-run",
        action="store_true",
        help="Preflight without training/writes",
    )
    return parser.parse_args(argv)


def execute(argv: Optional[Sequence[str]] = None) -> int:
    """Print a machine-readable report and return an honest verification
    status.
    """
    args = parse_args(argv)
    studies = selected_studies(args.experiment)
    if args.command == "check-saved":
        report = dict(
            mode="check-saved",
            **report_context(),
            results=[check_study(study) for study in studies],
        )
        report["status"] = {0: "passed", 1: "failed", 2: "partial"}[
            exit_status(report["results"])
        ]
        if args.report is not None:
            path = resolve_path(args.report)
            # A requested report must not be placed in any historical artifact
            # directory.
            for protected in (
                ROOT / "data",
                ROOT / "supporting_analyses",
                *REFERENCES.values(),
            ):
                require(
                    protected.resolve() not in path.resolve().parents,
                    "Report must be outside the reference/data directories",
                )
            write_report(path, report)
    else:
        report = rerun(studies, args.output, args.dry_run)
    print(json.dumps(portable_metadata(report), indent=2, allow_nan=False))
    return (
        2
        if report.get("status") == "dry_run"
        else exit_status(report["results"])
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Report rejected paths and setup failures without hiding their failure
    code.
    """
    try:
        return execute(argv)
    except (OSError, ValueError) as error:
        message = (
            str(error)
            .replace(str(ROOT), ".")
            .replace(sys.executable, "python")
        )
        print(json.dumps(dict(status="failed", error=message), indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
