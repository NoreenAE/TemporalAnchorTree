"""Experiment 1.1 GI without NOW, at either recorded minimum-gain setting."""
from __future__ import annotations

import argparse
from copy import deepcopy
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter
from typing import Iterator, Optional
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import experiment_1_fa_vs_gi as original  # noqa: E402
from TAT.eval_utils import (
    _draw_thesis_tree,
    _load_thesis_tree,
)  # noqa: E402
from TAT.temporal_tree import TemporalTree  # noqa: E402
from project_paths import resolve_path  # noqa: E402

plt = original.plt
BASELINE = (
    ROOT
    / "experiments/results/experiment_1_5000V3clean__moreOFFSETS2_patients_A=0.4_minS20_minL10_minGAIN0.0005"
)
OUTPUT = (
    ROOT / "experiments/results/experiment_1_1_gi_without_now_A0.4_minGAIN0"
)
ALLOWED_OUTPUT = None
PRIMARY = ROOT / "experiments/results/experiment_1_1_gi_without_now"
NAME = "Experiment 1.1 – GI Without NOW Anchor – A=0.4, min gain 0"
MIN_GAIN = 0.0
REMOVED = "not available / removed from search space"
BIRTH = "Birth Date Anchor"
NOW = "NOW Anchor"
# Pin the inspected final run, including its exact patient membership.
BASELINE_HASHES = {
    "config.json": (
        "5867fd41b350baec93240ce71ddf68bf0af5e5e44251a95ae75b1dfef42e69ef"
    ),
    "shared_patient_split.csv": (
        "59b8a1d7a12ac8d839611e3ba517dd9f6f4d9eaca439b85324de8f787708a900"
    ),
    "gi/metrics_and_temporal_summary.json": (
        "3466d40634d63e2afb89d209465d075749583a20f2c5e7f33fdeb2e34b9149e1"
    ),
}


def require(condition: bool, message: str) -> None:
    """Fail validation explicitly, including when Python assertions are disabled."""
    if not condition:
        raise ValueError(message)


def validate_output(output: Path) -> None:
    """Allow only the dedicated new destination; reject existing paths/symlinks."""
    expected = ALLOWED_OUTPUT or ROOT / (
        f"experiments/results/experiment_1_1_gi_without_now_A0.4_minGAIN{MIN_GAIN:g}"
    )
    require(
        output.absolute() == expected,
        "Only the isolated output directory is allowed.",
    )
    require(
        not output.exists() and not output.is_symlink(),
        "Output already exists; refusing to overwrite even a partial run.",
    )
    require(
        output.parent.resolve() == ROOT / "experiments/results",
        "Output parent must not redirect through a symlink.",
    )


def ablated_parameters(configuration: dict) -> dict:
    """Remove NOW using the selected recorded minimum-gain setting."""
    params = deepcopy(configuration["model_hyperparameters"])
    params["anchors_list"] = [BIRTH]
    params["anchor_offset_map"] = {BIRTH: params["anchor_offset_map"][BIRTH]}
    params["min_impurity_decrease"] = MIN_GAIN
    return params


def probe_candidate_windows(model: TemporalTree) -> list:
    """Exercise the real candidate generator without data aggregation or fitting.

    A synthetic feature and both raw anchor names test candidate exclusion.
    Empty aggregation results prevent threshold evaluation and tree learning.
    Only this disposable probe model is modified. Sampling is temporarily
    disabled so validation observes every available group, even when A < 1.
    """
    calls = []

    def capture(data, feature, window, aggregation, **kwargs):
        calls.append((feature, window, aggregation))
        return np.array([np.nan]), window[0]

    model.generated_groups = 0
    model.sampled_groups = 0
    data = pd.DataFrame({"feature_name": ["probe", BIRTH, NOW]})
    with patch.object(
        model, "_aggregate_feature_over_time", side_effect=capture
    ), patch.object(model, "_log"), patch.object(
        model, "candidate_sampling_rate", 1.0
    ):
        model._find_best_split(data, pd.Series([0]))
    require(
        all(feature == "probe" for feature, _, _ in calls),
        "Raw anchors entered the feature search space.",
    )
    windows = list(dict.fromkeys(window for _, window, _ in calls))
    for window in windows:
        require(
            [agg for _, candidate, agg in calls if candidate == window]
            == list(model.aggregation_funcs),
            "Candidate aggregation mismatch.",
        )
    return windows


def validate_parameters(
    config: dict, params: dict, aggregations: dict
) -> list:
    """Check the NOW ablation, selected gain cutoff, and candidate windows."""
    require(
        params == ablated_parameters(config),
        "Unexpected model parameter change.",
    )
    required = {
        "candidate_sampling_rate": 0.4,
        "max_depth": 3,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "min_impurity_decrease": MIN_GAIN,
        "use_dynamic_anchors": False,
    }
    for key, value in required.items():
        require(params[key] == value, f"Incorrect {key}.")
    require(params["anchors_list"] == [BIRTH], "Only Birth may be an anchor.")
    require(
        list(params["anchor_offset_map"]) == [BIRTH],
        "Unexpected anchor offsets.",
    )
    require(
        list(aggregations) == ["mean", "sum", "max", "count"],
        "Aggregations differ from the saved fitted GI model.",
    )
    windows = probe_candidate_windows(
        TemporalTree(**params, aggregation_funcs=aggregations)
    )
    expected = [
        tuple(w) for w in config["effective_candidate_windows"] if w[0] != NOW
    ]
    require(
        windows == expected,
        "Effective candidate windows changed unexpectedly.",
    )
    require(
        windows.count((None, None))
        == config["effective_full_timeline_window_count"]
        == 1,
        "Full Timeline must remain one effective window.",
    )
    return windows


def validate_split(
    split: pd.DataFrame,
    data: pd.DataFrame,
    labels: pd.DataFrame,
    reference: pd.DataFrame,
) -> None:
    """Verify saved membership, labels, disjointness and complete coverage."""
    require(
        not split["patient_id"].duplicated().any(), "Duplicate split patients."
    )
    pd.testing.assert_frame_equal(
        split.sort_values("patient_id").reset_index(drop=True),
        reference.sort_values("patient_id").reset_index(drop=True),
    )
    require(
        split["split"].value_counts().to_dict()
        == {"train": 3500, "test": 1500},
        "Expected exactly 3500 training and 1500 test patients.",
    )
    require(
        set(split.patient_id)
        == set(data.patient_id)
        == set(labels.patient_id),
        "Input patient coverage differs from the saved split.",
    )
    aligned = labels.set_index("patient_id").label.reindex(split.patient_id)
    require(
        np.array_equal(aligned.to_numpy(), split.gi_label.to_numpy()),
        "GI labels differ from saved split labels.",
    )


def prepare(*, validate_destination: bool = True) -> dict:
    """Read and validate immutable inputs and baseline; never train or write."""
    if validate_destination:
        validate_output(OUTPUT)
    for relative, digest in BASELINE_HASHES.items():
        require(
            original.sha256_file(BASELINE / relative) == digest,
            f"Final baseline artifact changed: {relative}",
        )
    config = json.loads((BASELINE / "config.json").read_text())
    baseline_result = json.loads(
        (BASELINE / "gi/metrics_and_temporal_summary.json").read_text()
    )
    require(baseline_result["outcome"] == "gi", "Outcome must be GI only.")
    require(
        config["decision_threshold"] == original.DECISION_THRESHOLD,
        "Evaluation threshold changed.",
    )
    # The public release records the checked model configuration in JSON.
    # This mapping was verified against the original fitted GI model before
    # excluding its serialized local code paths from the public snapshot.
    aggregations = original.aggregation_functions()
    require(list(aggregations) == config["aggregations"],
            "Saved baseline aggregation order differs from the verified mapping.")
    params = ablated_parameters(config)
    windows = validate_parameters(config, params, aggregations)
    paths = {name: resolve_path(config["input_paths"][name]) for name in ("x", "gi")}
    for name, path in paths.items():
        require(
            original.sha256_file(path) == config["input_sha256"][name],
            f"Input hash mismatch: {name}",
        )
    data = original.load_longitudinal_x(paths["x"])
    labels = original.load_labels(paths["gi"], "gi")
    reference = pd.read_csv(BASELINE / "shared_patient_split.csv")
    split = reference.copy(deep=True)
    validate_split(split, data, labels, reference)
    provenance = {
        "experiment_name": NAME,
        "analysis_type": f"NOW-anchor ablation with min gain {MIN_GAIN:g}",
        "sensitivity_change": {
            "parameter": "min_impurity_decrease",
            "baseline_value": config["model_hyperparameters"]["min_impurity_decrease"],
            "sensitivity_value": params["min_impurity_decrease"],
        },
        "outcome": "gi",
        "intended_ablation": "NOW Anchor removed from candidate search space.",
        "true_region": config["outcomes"]["gi"]["true_region"],
        "interpretation": "Alternative windows are not recovery of the true GI temporal ground truth.",
        "baseline_directory": str(BASELINE),
        "baseline_sha256": BASELINE_HASHES,
        "split_source": str(BASELINE / "shared_patient_split.csv"),
        "input_paths": {k: str(v) for k, v in paths.items()},
        "input_sha256": {k: config["input_sha256"][k] for k in paths},
        "model_hyperparameters": params,
        "aggregations": list(aggregations),
        "aggregation_provenance": "Explicit mapping verified against the original fitted GI model.",
        "baseline_declared_aggregations": config["aggregations"],
        "effective_candidate_windows": windows,
        "effective_full_timeline_window_count": 1,
        "decision_threshold": config["decision_threshold"],
        "temporal_explainability_bin_edges": config[
            "temporal_explainability_bin_edges"
        ],
        "temporal_usage_population": "training patients, production path tracing",
        "n_train_patients": 3500,
        "n_test_patients": 1500,
        "now_usage_status": REMOVED,
        "output_directory": str(OUTPUT),
    }
    return dict(
        data=data,
        labels=labels,
        split=split,
        params=params,
        aggregations=aggregations,
        provenance=provenance,
        baseline_result=baseline_result,
        configuration=config,
    )


def usage_summary(paths: pd.DataFrame) -> tuple:
    """Summarize only available windows; never report NOW dominance as zero."""
    require(
        not paths["anchor"].eq(NOW).any(), "NOW appeared in learned paths."
    )
    total = len(paths)
    full = paths["is_full_timeline"].astype(bool)
    birth = paths["anchor"].eq(BIRTH) & ~full
    require(
        int(birth.sum() + full.sum()) == total, "Unexpected learned anchor."
    )
    summary = {
        "total_patient_path_split_usages": total,
        "birth_date_usage_proportion": float(birth.sum() / total)
        if total
        else 0.0,
        "full_timeline_usage_proportion": float(full.sum() / total)
        if total
        else 0.0,
    }
    windows = pd.Series(
        [
            "Full Timeline"
            if row.is_full_timeline
            else f"{row.anchor} | offset={row.stored_offset:g}"
            for row in paths.itertuples()
        ],
        dtype=str,
    )
    distribution = (
        windows.value_counts()
        .rename_axis("window")
        .reset_index(name="raw_usage_count")
    )
    distribution["usage_proportion"] = (
        distribution.raw_usage_count / total if total else 0.0
    )
    return summary, distribution


def comparison_table(baseline: dict, result: dict) -> pd.DataFrame:
    """Compare descriptive results, explicitly marking unavailable NOW usage."""
    rows = []
    for name, record, removed in [
        ("Original GI (with NOW)", baseline, False),
        (f"GI without NOW, A=0.4, min_gain={MIN_GAIN:g} (Experiment 1.1)", result, True),
    ]:
        root = record["root_split"]
        usage = record["temporal_usage"]
        window = (
            "Leaf"
            if root["is_leaf"]
            else "Full Timeline"
            if root["is_full_timeline"]
            else f"{root['anchor']} | offset={root['offset']}"
        )
        rows.append(
            {
                "experiment": name,
                "roc_auc": record["performance"]["roc_auc"],
                "root_feature": root.get("feature"),
                "root_temporal_window": window,
                "birth_usage_percent": 100
                * usage["birth_date_usage_proportion"],
                "now_usage_percent": REMOVED
                if removed
                else 100 * usage["now_usage_proportion"],
                "full_timeline_usage_percent": 100
                * usage["full_timeline_usage_proportion"],
                "major_temporal_windows": json.dumps(
                    sorted(
                        record["window_distribution"],
                        key=lambda w: -w["usage_proportion"],
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def save_figures(
    comparison: pd.DataFrame, output: Path, tree_path: Optional[Path] = None
) -> None:
    """Save descriptive comparisons and the existing thesis-style GI tree."""
    figures = output / "figures"
    figures.mkdir()
    colors = ["#4477AA", "#EE7733"]
    labels = ["Original GI\nwith NOW", f"GI without NOW\nA=0.4, min_gain={MIN_GAIN:g}"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, comparison.roc_auc, color=colors, width=0.55)
    for bar, value in zip(bars, comparison.roc_auc):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
        )
    ax.set(
        ylim=(0, 1), ylabel="Test ROC-AUC",
        title=f"Experiment 1.1 — GI without NOW, A=0.4, min gain {MIN_GAIN:g}",
    )
    fig.text(
        0.5,
        0.01,
        "Single fixed split; no statistical significance inferred.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(figures / "roc_auc_original_vs_without_now.png", dpi=300)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(3)
    for index, row in comparison.iterrows():
        positions = x + (index - 0.5) * 0.34
        values = [
            row.birth_usage_percent,
            row.now_usage_percent,
            row.full_timeline_usage_percent,
        ]
        for position, value in zip(positions, values):
            if isinstance(value, str):
                ax.text(
                    position,
                    4,
                    "Unavailable\nNOW removed",
                    ha="center",
                    fontsize=9,
                    rotation=90,
                    color=colors[index],
                )
            else:
                ax.bar(position, value, width=0.32, color=colors[index])
        ax.bar(
            [], [], color=colors[index], label=labels[index].replace("\n", " ")
        )
    ax.set(
        xticks=x,
        xticklabels=["Birth Anchor", "NOW Anchor", "Full Timeline"],
        ylim=(0, 105),
        ylabel="Training patient-path split usage (%)",
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(
        figures / "temporal_usage_original_vs_without_now.png", dpi=300
    )
    plt.close(fig)
    tree = _load_thesis_tree(tree_path or output / "gi_tree.json")
    height = max(3.5, 2 + 1.45 * max(-y for _, y in tree[1].values()))
    fig, ax = plt.subplots(figsize=(max(10, tree[3] * 1.9), height))
    _draw_thesis_tree(
        ax, tree, f"GI learned tree — NOW unavailable, A=0.4, min gain {MIN_GAIN:g}"
    )
    fig.tight_layout()
    fig.savefig(figures / "gi_tree_thesis.png", dpi=300, facecolor="white")
    plt.close(fig)


def export_results(
    model: TemporalTree,
    test_order: np.ndarray,
    y_test: pd.Series,
    scores: np.ndarray,
    paths: pd.DataFrame,
    timing: dict,
    baseline: dict,
    output: Path,
) -> dict:
    """Retain the earlier flat export API for existing callers and tests.

    New runs use Experiment 1's standard output pipeline in ``run``.
    """
    performance, _ = original.evaluate_scores(y_test, scores)
    pd.DataFrame(
        {
            "patient_id": test_order,
            "y_true": y_test.to_numpy(),
            "y_score": scores,
        }
    ).to_csv(output / "gi_test_predictions.csv", index=False)
    original.export_tree_json(model, output / "gi_tree.json")
    paths.to_csv(output / "gi_temporal_path_usage.csv", index=False)
    usage, distribution = usage_summary(paths)
    distribution.to_csv(output / "gi_temporal_window_usage.csv", index=False)
    result = {
        "experiment_name": NAME,
        "min_impurity_decrease": model.min_impurity_decrease,
        "outcome": "gi",
        "performance": performance,
        "runtime_seconds": timing,
        "root_split": original.root_split_summary(model),
        "temporal_usage": usage,
        "candidate_evaluations": original.candidate_evaluation_summary(model),
        "window_distribution": distribution.to_dict("records"),
    }
    original.write_json(
        result, output / "gi_metrics_and_temporal_summary.json"
    )
    flat = pd.json_normalize(
        {k: v for k, v in result.items() if k != "window_distribution"}
    )
    flat["birth_usage_percent"] = 100 * usage["birth_date_usage_proportion"]
    flat["full_timeline_usage_percent"] = (
        100 * usage["full_timeline_usage_proportion"]
    )
    flat.to_csv(output / "per_run_results.csv", index=False)
    comparison = comparison_table(baseline, result)
    comparison.to_csv(output / "comparison_with_original_gi.csv", index=False)
    save_figures(comparison, output)
    return result


def run(prepared: dict) -> None:
    """Fit once, using only the validated saved GI inputs and patient split."""
    start = perf_counter()
    validate_output(OUTPUT)
    OUTPUT.mkdir(exist_ok=False)
    original.write_json(
        prepared["provenance"], OUTPUT / "experiment_config.json"
    )
    shutil.copyfile(
        BASELINE / "shared_patient_split.csv",
        OUTPUT / "shared_patient_split.csv",
    )
    split = prepared["split"]
    config = prepared["configuration"]
    # Reuse the final GI pipeline with its saved evaluation settings, while
    # removing NOW and applying the selected recorded minimum-gain cutoff.
    with patch.multiple(
        original,
        MODEL_HYPERPARAMETERS=prepared["params"],
        DECISION_THRESHOLD=config["decision_threshold"],
        TEMPORAL_EXPLAINABILITY_BIN_EDGES=config[
            "temporal_explainability_bin_edges"
        ],
        OUTCOMES=config["outcomes"],
    ), patch.object(
        original, "aggregation_functions", return_value=prepared["aggregations"]
    ):
        result = original.run_outcome(
            outcome="gi",
            data=prepared["data"],
            labels=prepared["labels"],
            train_ids=split.loc[split.split.eq("train"), "patient_id"].to_numpy(),
            test_ids=split.loc[split.split.eq("test"), "patient_id"].to_numpy(),
            output_dir=OUTPUT / "gi",
        )
    comparison = comparison_table(prepared["baseline_result"], result)
    comparison.to_csv(OUTPUT / "comparison_with_original_gi.csv", index=False)
    save_figures(comparison, OUTPUT, tree_path=OUTPUT / "gi/tree.json")
    original.write_json(
        {"status": "completed", "total_seconds": perf_counter() - start},
        OUTPUT / "run_status.json",
    )


@contextmanager
def configured_run(
    min_gain: float, output: Optional[Path] = None,
) -> Iterator[None]:
    """Select either recorded GI ablation without changing its other settings."""
    if min_gain not in (0.0, 0.0005):
        raise ValueError("Only recorded min gains 0 and 0.0005 are supported.")
    destination = output or ROOT / (
        f"experiments/results/experiment_1_1_gi_without_now_A0.4_minGAIN{min_gain:g}"
    )
    destination = resolve_path(destination)
    if destination.parent.resolve() != ROOT / "experiments/results":
        raise ValueError("Output must be a new directory under experiments/results.")
    if destination == PRIMARY or destination == BASELINE:
        raise ValueError("Output cannot replace a protected baseline.")
    with patch.multiple(
        sys.modules[__name__], MIN_GAIN=min_gain, OUTPUT=destination,
        ALLOWED_OUTPUT=destination,
        NAME=f"Experiment 1.1 – GI Without NOW Anchor – A=0.4, min gain {min_gain:g}",
    ):
        yield


def main() -> None:
    """Require an explicit validation or training mode; no implicit fitting."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--min-gain", type=float, choices=(0.0, 0.0005), default=0.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with configured_run(args.min_gain, args.output):
        prepared = prepare(validate_destination=args.run)
        print(json.dumps(original.portable_metadata(prepared["provenance"]), indent=2))
        if args.run:
            run(prepared)
        else:
            print("Validation passed. No training performed; no output artifacts written.")


if __name__ == "__main__":
    main()
