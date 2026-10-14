"""Evaluate saved Experiment 2 scores among documented-fish test patients.

Use precisely candidate-sampling seeds 42--46; seed 42 is the reference run.
No estimator is fitted or loaded. The raw-event subgroup is fixed before any
prediction values are read. Every execution saves a new analysis directory.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import experiment_2_cart_baseline as baseline  # noqa: E402

SEEDS = (42, 43, 44, 45, 46)
ROBUSTNESS = ROOT / "experiments/results/experiment_2_1_candidate_sampling_robustness"
CART_RUN = ROOT / (
    "experiments/results/experiment_2_cart_baseline/"
    "run_20261014T093121848027Z_9a8a1270"
)
OUTPUT_ROOT = ROOT / "experiments/results/experiment_2_1_documented_fish_subgroup"


def read_json(path: Path) -> dict:
    """Read a saved configuration or provenance record."""
    return json.loads(path.read_text())


def require(condition: bool, message: str) -> None:
    """Stop on invalid inputs, including when Python assertions are disabled."""
    if not condition:
        raise ValueError(message)


def define_subgroup(cart_run: Path) -> tuple:
    """Validate cohort identity and define membership solely from raw events."""
    metadata_path = cart_run / "run_metadata.json"
    metadata = read_json(metadata_path)
    paths = {name: Path(value) for name, value in metadata["input_paths"].items()}
    tracked = {metadata_path: baseline.file_sha256(metadata_path)}
    for name, path in paths.items():
        require(baseline.file_sha256(path) == metadata["input_sha256"][name],
                f"CART input hash changed: {path}")
        tracked[path] = metadata["input_sha256"][name]
    recorded = paths["observations"].parent
    manifest = read_json(paths["manifest"])
    # The manifest also holds generator provenance; only these keys name data files.
    for name in ("observations.csv", "patients_and_split.csv", "validation.json",
                 "descriptive_audit.json"):
        expected = manifest[name]
        path = recorded / name
        require(baseline.file_sha256(path) == expected, f"Dataset hash changed: {path}")
        tracked[path] = expected
    raw = pd.read_csv(paths["observations"], dtype={"patient_id": str, "value": object})
    info = pd.read_csv(paths["patients_and_split"],
                       usecols=["patient_id", "partition", "FA", "block"],
                       dtype={"patient_id": str}).set_index("patient_id", verify_integrity=True)
    require(set(raw.patient_id) == set(info.index), "Raw cohort and label IDs differ")
    require(info.partition.isin(["train", "test"]).all(), "Invalid partitions")
    require(info.FA.isin([0, 1]).all(), "FA must be binary with positive class 1")
    require(not info[["partition", "FA", "block"]].isna().any().any(), "Missing labels/split")
    require(info.groupby("block").partition.nunique().eq(1).all(), "Block leakage")
    train_ids = info.index[info.partition.eq("train")]
    test_ids = info.index[info.partition.eq("test")]
    require(metadata["train_patient_ids"] == train_ids.tolist(), "CART train split differs")
    require(metadata["test_patient_ids"] == test_ids.tolist(), "CART test split differs")
    # Membership uses only the raw feature name and the original test IDs.
    fish_ids = set(raw.loc[raw.feature_name.eq("FishIntroduction"), "patient_id"])
    subgroup = test_ids[test_ids.isin(fish_ids)]
    require(len(subgroup) > 0, "Empty documented-fish subgroup")
    require(info.loc[subgroup, "FA"].nunique() == 2, "Subgroup needs both classes")
    matrix = baseline.aggregate_events(raw, info.index)
    require(metadata["feature_columns"] == matrix.columns.tolist(), "CART feature schema differs")
    require(metadata["aggregation"] == "one full-history sum per clinical event",
            "CART aggregation differs")
    require(matrix.loc[subgroup, "FishIntroduction__sum"].ge(1).all(),
            "Raw subgroup and clinical counts disagree")
    return recorded, info, test_ids, subgroup, tracked


def load_tat_scores(recorded: Path, robustness: Path, info: pd.DataFrame,
                    test_ids: pd.Index, tracked: dict) -> dict:
    """Align every saved ON/OFF score by ID and validate its seed/configuration."""
    configuration = read_json(recorded / "configuration.json")
    protocol_path = robustness / "protocol.json"
    protocol = read_json(protocol_path)
    tracked[protocol_path] = baseline.file_sha256(protocol_path)
    require(protocol["baseline_configuration"] == configuration, "Robustness baseline differs")
    for name, expected in protocol["input_sha256"].items():
        require(baseline.file_sha256(recorded / name) == expected,
                f"Robustness dataset differs: {name}")
    scores = {}
    for seed in SEEDS:
        for mode in ("dynamic_on", "dynamic_off"):
            folder = recorded / mode if seed == 42 else robustness / "seeds" / f"seed_{seed}" / mode
            if seed == 42:
                require(configuration["parameters"]["split_sampling_seed"] == seed,
                        "Reference candidate seed is not 42")
                require(any(run["name"] == mode and run["USE_DYNAMIC_ANCHORS"] ==
                            (mode == "dynamic_on") for run in configuration["runs"]),
                        "Reference dynamic switch differs")
            else:
                effective_path = folder / "effective_configuration.json"
                effective = read_json(effective_path)
                tracked[effective_path] = baseline.file_sha256(effective_path)
                for name, value in configuration["parameters"].items():
                    expected = seed if name == "split_sampling_seed" else value
                    require(effective[name] == expected, f"Seed {seed} setting differs: {name}")
                require(effective["use_dynamic_anchors"] == (mode == "dynamic_on"),
                        f"Seed {seed} dynamic switch differs")
                require(effective["aggregation_funcs"] == {"sum": "numpy.sum"},
                        f"Seed {seed} aggregation differs")
            path = folder / "patient_predictions.csv"
            tracked[path] = baseline.file_sha256(path)
            prediction = pd.read_csv(path, usecols=["patient_id", "partition", "FA", "score"],
                                     dtype={"patient_id": str}).set_index(
                                         "patient_id", verify_integrity=True)
            require(set(prediction.index) == set(info.index), f"Missing/extra {mode} seed {seed} IDs")
            prediction = prediction.reindex(info.index)
            require(prediction.partition.equals(info.partition), f"Split differs: {mode}, {seed}")
            require(prediction.FA.equals(info.FA), f"Labels differ: {mode}, {seed}")
            require(prediction.score.notna().all() and prediction.score.between(0, 1).all(),
                    f"Invalid probability scores: {mode}, {seed}")
            scores[("TAT Dynamic ON" if mode == "dynamic_on" else "TAT Dynamic OFF", seed)] = (
                prediction.loc[test_ids, "score"])
    return scores


def load_cart_scores(cart_run: Path, info: pd.DataFrame, test_ids: pd.Index,
                     tracked: dict, reference_scores: pd.Series) -> pd.Series:
    """Use validated saved CART positive-class probabilities without fitting."""
    path = cart_run / "test_predictions.csv"
    tracked[path] = baseline.file_sha256(path)
    prediction = pd.read_csv(path, dtype={"patient_id": str}).set_index(
        "patient_id", verify_integrity=True)
    require(set(prediction.index) == set(test_ids), "CART test IDs differ")
    prediction = prediction.reindex(test_ids)
    require(np.array_equal(prediction.FA_label, info.loc[test_ids, "FA"]), "CART labels differ")
    require(prediction.CART_probability.notna().all() and
            prediction.CART_probability.between(0, 1).all(), "Invalid CART probabilities")
    require(np.array_equal(prediction.TAT_dynamic_on_probability,
                           reference_scores.loc[test_ids]), "CART reference ON scores differ")
    return prediction.CART_probability


def evaluate(scores: dict, info: pd.DataFrame, ids: pd.Index) -> pd.DataFrame:
    """Compute probability-based AUC for one fixed population."""
    labels = info.loc[ids, "FA"]
    rows = []
    for (model, seed), values in scores.items():
        require(set(ids).issubset(values.index), f"Missing {model} subgroup predictions")
        rows.append(dict(model=model, seed=seed, subgroup_n=len(ids),
                         positives=int(labels.sum()), negatives=int(len(ids)-labels.sum()),
                         roc_auc=float(roc_auc_score(labels, values.loc[ids]))))
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> dict:
    """Summarize seed AUCs using sample SD (ddof=1), not ensemble scores."""
    summaries = {}
    for model in ("TAT Dynamic ON", "TAT Dynamic OFF", "CART"):
        values = results.loc[results.model.eq(model), "roc_auc"]
        summaries[model] = dict(mean=float(values.mean()),
                                sd=float(values.std(ddof=1)) if len(values) > 1 else None,
                                runs=len(values))
    return summaries


def paired_bootstrap(scores: dict, info: pd.DataFrame, subgroup: pd.Index,
                     repeats: int, seed: int) -> dict:
    """Resample documented patients in blocks, sharing draws across all scores.

    Compute the mean of five seed AUCs within every resample, rather than the
    AUC of averaged predictions. Models and the seed set stay fixed.
    """
    rng = np.random.default_rng(seed)
    labels = info.loc[subgroup, "FA"].to_numpy()
    block_ids = info.loc[subgroup, "block"].to_numpy()
    blocks = [np.flatnonzero(block_ids == block) for block in np.unique(block_ids)]
    arrays = {key: values.loc[subgroup].to_numpy() for key, values in scores.items()}
    draws = []
    for _ in range(repeats):
        indices = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        if len(np.unique(labels[indices])) < 2:
            continue
        aucs = {key: roc_auc_score(labels[indices], values[indices]) for key, values in arrays.items()}
        on = np.mean([aucs[("TAT Dynamic ON", s)] for s in SEEDS])
        off = np.mean([aucs[("TAT Dynamic OFF", s)] for s in SEEDS])
        cart = aucs[("CART", "NA")]
        draws.append([on, off, cart, on-cart, on-off, aucs[("TAT Dynamic ON", 42)]-cart])
    require(bool(draws), "No valid bootstrap draws")
    names = ("ON_mean", "OFF_mean", "CART", "ON_mean_minus_CART",
             "ON_mean_minus_OFF_mean", "ON_reference42_minus_CART")
    return dict(seed=seed, requested_repeats=repeats, valid_repeats=len(draws),
                test_blocks=len(blocks), method="paired percentile block bootstrap",
                interpretation="Conditional on saved models, fixed seed set, and observed subgroup; "
                               "does not resample training data or candidate-sampling seeds.",
                intervals_95=dict(zip(names, np.quantile(draws, [.025, .975], axis=0).T.tolist())))


def save_figure(summary: dict, output: Path) -> None:
    """Plot seed mean/sample SD and the single saved CART value."""
    fig, axis = baseline.plt.subplots(figsize=(8, 5))
    for position, (model, values) in enumerate(summary.items()):
        axis.errorbar(position, values["mean"], yerr=values["sd"], fmt="o",
                      capsize=6, markersize=8)
        label = f"{values['mean']:.2f}"
        if values["sd"] is not None:
            label += f" ± {values['sd']:.2f}"
        axis.annotate(label, (position, values["mean"] + (values["sd"] or 0)), xytext=(0, 12),
                      textcoords="offset points", ha="center")
    axis.axhline(.5, linestyle="--", color="gray", label="Chance (0.50)")
    axis.set(xticks=range(3), xticklabels=list(summary), ylabel="ROC-AUC",
             ylim=(.4, 1), title="Experiment 2.1 – Documented FishIntroduction Subgroup")
    axis.text(.5, -.16, "TAT: seeds 42–46, mean ± sample SD; CART: single saved run",
              transform=axis.transAxes, ha="center", fontsize=9)
    axis.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output / "experiment_2_1_documented_fish_comparison.png", dpi=300)
    baseline.plt.close(fig)


def main(cart_run: Path, robustness: Path, output_root: Path,
         bootstrap_repeats: int, bootstrap_seed: int) -> None:
    """Analyze saved scores and save a new result set without altering inputs."""
    require(bootstrap_repeats > 0, "Bootstrap repeats must be positive")
    recorded, info, test_ids, subgroup, tracked = define_subgroup(cart_run)
    print(f"Test patients: {len(test_ids)}; documented: {len(subgroup)}; "
          f"undocumented: {len(test_ids)-len(subgroup)}")
    print(f"Subgroup positives: {int(info.loc[subgroup, 'FA'].sum())}; "
          f"negatives: {int(len(subgroup)-info.loc[subgroup, 'FA'].sum())}")
    scores = load_tat_scores(recorded, robustness, info, test_ids, tracked)
    scores[("CART", "NA")] = load_cart_scores(
        cart_run, info, test_ids, tracked, scores[("TAT Dynamic ON", 42)])
    print("SUBGROUP IDS IDENTICAL ACROSS MODELS: YES")
    documented_results = evaluate(scores, info, subgroup)
    full_results = evaluate(scores, info, test_ids)
    summary, full_summary = summarize(documented_results), summarize(full_results)
    confidence = paired_bootstrap(scores, info, subgroup, bootstrap_repeats, bootstrap_seed)
    for path, expected in tracked.items():
        require(baseline.file_sha256(path) == expected, f"Input changed during analysis: {path}")
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = output_root / f"run_{stamp}_{uuid4().hex[:8]}"
    output.mkdir(exist_ok=False)
    info.loc[subgroup, ["block", "FA"]].reset_index().to_csv(
        output / "documented_fish_subgroup_patient_ids.csv", index=False)
    documented_results.to_csv(output / "experiment_2_1_documented_fish_results.csv", index=False,
                              float_format="%.17g")
    full_results.to_csv(output / "experiment_2_1_full_test_reference_results.csv", index=False,
                       float_format="%.17g")
    rows = [dict(model=model, mean_roc_auc=values["mean"], sample_sd=values["sd"],
                 thesis_display=(f"{values['mean']:.2f} ± {values['sd']:.2f}" if
                                 values["sd"] is not None else f"{values['mean']:.2f}"))
            for model, values in summary.items()]
    pd.DataFrame(rows).to_csv(output / "documented_fish_primary_comparison.csv", index=False,
                              float_format="%.17g", na_rep="NA")
    save_figure(summary, output)
    report = dict(subgroup_definition="Original test patients with >=1 raw FishIntroduction row",
                  subgroup_defined_before_predictions=True, total_test_n=len(test_ids),
                  subgroup_n=len(subgroup), undocumented_n=len(test_ids)-len(subgroup),
                  positives=int(info.loc[subgroup, "FA"].sum()),
                  negatives=int(len(subgroup)-info.loc[subgroup, "FA"].sum()),
                  seeds=list(SEEDS), sd_ddof=1, documented_summary=summary,
                  full_test_summary=full_summary, paired_bootstrap=confidence,
                  validation=dict(same_ids=True, same_labels=True, same_split=True,
                                  block_leakage=False, missing_predictions=0,
                                  scores="saved FA=1 probabilities/leaf means",
                                  cart_features="eight original full-history clinical event sums",
                                  models_retrained=False, original_inputs_unchanged=True),
                  input_sha256={str(path): expected for path, expected in tracked.items()},
                  analysis_script_sha256=baseline.file_sha256(Path(__file__)))
    (output / "analysis_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(documented_results.to_string(index=False))
    print("\nDocumented subgroup:")
    print(pd.DataFrame(rows)[["model", "thesis_display"]].to_string(index=False))
    print("\nFull test reference (same seeds 42–46):")
    print(json.dumps(full_summary, indent=2))
    print("\nSupporting paired block-bootstrap intervals:")
    print(json.dumps(confidence, indent=2))
    print(f"Saved analysis: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cart-run", type=Path, default=CART_RUN)
    parser.add_argument("--robustness", type=Path, default=ROBUSTNESS)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261014)
    args = parser.parse_args()
    main(args.cart_run, args.robustness, args.output_root,
         args.bootstrap_repeats, args.bootstrap_seed)
