"""Read-only audit of the saved Experiment 2 CART comparison.

Refits CART in memory, independently replays saved TAT trees from raw events,
and prints JSON. Never writes experiment artifacts or regenerates the cohort.
Run with the same Python environment used for the original baseline.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import experiment_2_cart_baseline as baseline  # noqa: E402

DEFAULT_CART_RUN = ROOT / (
    "experiments/results/experiment_2_cart_baseline/"
    "run_20261014T093121848027Z_9a8a1270"
)


def digest(path: Path) -> str:
    """Compute immutable artifact identity."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def raw_histories(raw: pd.DataFrame) -> dict:
    """Index only patient IDs, feature names, and raw observation timestamps."""
    histories = defaultdict(dict)
    for (patient, feature), group in raw.groupby(["patient_id", "feature_name"]):
        histories[patient][feature] = sorted(group.time.tolist())
    return histories


def raw_rule(histories: dict, ids: pd.Index) -> pd.Series:
    """Reconstruct (fish, fish + one calendar month] symptom presence."""
    result = {}
    for patient in ids:
        history = histories[patient]
        fish = history.get("FishIntroduction", [])
        assert len(fish) <= 1
        result[patient] = int(bool(fish) and any(
            fish[0] < time <= fish[0] + pd.DateOffset(months=1)
            for feature in ("GI", "Vomiting")
            for time in history.get(feature, [])
        ))
    return pd.Series(result).reindex(ids)


def replay_tree(histories: dict, ids: pd.Index, records: list,
                dynamic: bool) -> tuple:
    """Replay saved splits independently; no labels/private metadata are read."""
    nodes = {node["node_id"]: node for node in records}
    scores, paths, empty, anchors = {}, defaultdict(list), Counter(), Counter()
    for patient in ids:
        history = histories[patient]
        registry = {"Birth Date Anchor": history["Birth Date Anchor"]}
        node = nodes[0]
        while True:
            paths[node["node_id"]].append(patient)
            if node["type"] == "leaf":
                scores[patient] = node["score"]
                break
            reference, offset = node["window"]
            events = history.get(node["feature"], [])
            windows = [(start, start + pd.DateOffset(months=offset))
                       for start in registry.get(reference, [])] if offset is not None else []
            eligible = events if offset is None else [time for time in events if any(
                min(start, end) <= time <= max(start, end)
                for start, end in windows
            )]
            aggregate = len(eligible) if eligible else np.nan
            if pd.isna(aggregate):
                direction = node.get("missing_direction", "right")
                empty[node["node_id"]] += 1
            else:
                direction = "left" if aggregate <= node["threshold"] else "right"
            if dynamic:
                name = f"Node {node['source_node_id']} | SUM({node['feature']}) Anchor"
                registry[name] = [min(eligible)] if eligible else []
                anchors[node["node_id"]] += bool(eligible)
            node = nodes[node[direction]]
    return pd.Series(scores).reindex(ids), dict(paths), dict(empty), dict(anchors)


def fit_cart(matrix, info, train, test, parameters):
    """Refit on the recorded training IDs and select positive-class scores."""
    model = DecisionTreeClassifier(**parameters)
    model.fit(matrix.loc[train], info.loc[train, "FA"])
    positive = list(model.classes_).index(1)
    return model, model.predict_proba(matrix.loc[test])[:, positive]


def audit(cart_run: Path, bootstrap_repeats: int, permutations: int,
          child: bool = False) -> dict:
    """Validate saved identities, model behavior, and conditional uncertainty."""
    metadata = read_json(cart_run / "run_metadata.json")
    recorded = Path(metadata["input_paths"]["observations"]).parent
    tracked = list(recorded.rglob("*")) + list(cart_run.rglob("*"))
    before = {str(path): digest(path) for path in tracked if path.is_file()}
    with contextlib.redirect_stdout(io.StringIO()):
        matrix, info, train, test, reference, config = baseline.prepare_inputs(recorded)
    raw = pd.read_csv(recorded / "observations.csv", parse_dates=["time"],
                      dtype={"patient_id": str, "value": str})
    saved = pd.read_csv(cart_run / "test_predictions.csv").set_index(
        "patient_id", verify_integrity=True).reindex(test)
    assert metadata["train_patient_ids"] == train.tolist()
    assert metadata["test_patient_ids"] == test.tolist()
    assert set(raw.patient_id) == set(matrix.index) == set(info.index)
    assert saved.FA_label.equals(info.loc[test, "FA"].rename("FA_label"))
    model, scores = fit_cart(matrix, info, train, test, metadata["model_parameters"])
    assert np.allclose(scores, saved.CART_probability, rtol=0, atol=1e-15)
    assert np.allclose(reference.loc[test, "score"], saved.TAT_dynamic_on_probability,
                       rtol=0, atol=1e-15)
    result = {"cart_auc": roc_auc_score(info.loc[test, "FA"], scores),
              "cart_prediction_sha256": hashlib.sha256(scores.tobytes()).hexdigest()}
    if child:
        return result
    result.update(cart_parameters=model.get_params(), tat_parameters=config["parameters"],
                  source_dataset_bytes_match={name: digest(recorded / name) == digest(
                      Path(config["dataset_directory"]) / name)
                      for name in ("observations.csv", "patients_and_split.csv")},
                  input_hashes_match={name: digest(Path(metadata["input_paths"][name])) == value
                                      for name, value in metadata["input_sha256"].items()},
                  tat_current_source_hash_matches={name: digest(ROOT / name) == value
                                                   for name, value in config["sha256"].items()},
                  cohort_patients=len(info), raw_observations=len(raw),
                  raw_feature_counts=raw.feature_name.value_counts().to_dict(),
                  train_patients=len(train), test_patients=len(test),
                  train_blocks=info.loc[train, "block"].nunique(),
                  test_blocks=info.loc[test, "block"].nunique(),
                  train_test_overlap=len(set(train) & set(test)),
                  crossing_blocks=int(info.groupby("block").partition.nunique().gt(1).sum()),
                  label_examples=info.FA.sample(6, random_state=42).to_dict(),
                  classes=model.classes_.tolist(),
                  cart_hard_label_auc=roc_auc_score(info.loc[test, "FA"], model.predict(matrix.loc[test])),
                  cart_train_auc=roc_auc_score(info.loc[train, "FA"], model.predict_proba(matrix.loc[train])[:, 1]),
                  tat_train_auc=roc_auc_score(info.loc[train, "FA"], reference.loc[train, "score"]))
    histories = raw_histories(raw)
    rule = raw_rule(histories, info.index)
    assert rule.eq(info.R).all()  # Comparison only AFTER raw reconstruction.
    result["oracle_auc"] = roc_auc_score(info.loc[test, "FA"], rule.loc[test])
    result["partition_mechanism"] = {}
    for name, ids in (("all", info.index), ("train", train), ("test", test)):
        group = info.loc[ids]
        result["partition_mechanism"][name] = dict(
            n=len(ids), positives=int(group.FA.sum()), prevalence=float(group.FA.mean()),
            documented=int(group.documented.sum()),
            relationship_present=int(rule.loc[ids].sum()),
            positive_given_present=float(group.loc[rule.loc[ids].eq(1), "FA"].mean()),
            positive_given_absent=float(group.loc[rule.loc[ids].eq(0), "FA"].mean()))
    result["tat_replay"] = {}
    all_scores = {}
    for name, dynamic in (("dynamic_on", True), ("dynamic_off", False)):
        records = read_json(recorded / name / "tree.json")
        replay, paths, empty, created = replay_tree(histories, info.index, records, dynamic)
        predictions = pd.read_csv(recorded / name / "patient_predictions.csv").set_index(
            "patient_id", verify_integrity=True).reindex(info.index)
        assert predictions.FA.eq(info.FA).all()
        assert predictions.partition.eq(info.partition).all()
        assert np.allclose(replay, predictions.score, rtol=0, atol=1e-15)
        for node in records:
            training_ids = [pid for pid in paths[node["node_id"]] if pid in set(train)]
            assert len(training_ids) == node["patients"]
            if node["type"] == "leaf":
                assert abs(info.loc[training_ids, "FA"].mean() - node["score"]) < 1e-15
        all_scores[name] = replay.loc[test]
        result["tat_replay"][name] = dict(all_predictions_match=True,
            training_leaf_means_match=True, test_auc=roc_auc_score(info.loc[test, "FA"], replay.loc[test]),
            empty_window_routes=empty, created_anchors=created)
    result["univariate_test"] = {}
    diagnostics = matrix.copy()
    diagnostics["total_observations"] = raw.groupby("patient_id").size().reindex(info.index)
    diagnostics["raw_observation_span_days"] = raw.groupby("patient_id").time.agg(
        lambda values: (values.max() - values.min()).total_seconds() / 86400).reindex(info.index)
    diagnostics["scheduled_followup_days"] = (
        pd.to_datetime(info.followup_end) - pd.to_datetime(info.birth)).dt.days
    for column in diagnostics:
        auc = roc_auc_score(info.loc[test, "FA"], diagnostics.loc[test, column])
        result["univariate_test"][column] = dict(auc=auc, direction_adjusted_auc=max(auc, 1-auc),
            means_by_label=diagnostics.loc[test].groupby(info.loc[test, "FA"])[column].mean().to_dict())
    result["subgroups"] = {}
    for documented in (True, False):
        ids = test[info.loc[test, "documented"].eq(documented)]
        labels = info.loc[ids, "FA"]
        result["subgroups"][str(documented)] = dict(n=len(ids), positives=int(labels.sum()),
            cart_auc=roc_auc_score(labels, saved.loc[ids, "CART_probability"]),
            **{name: roc_auc_score(labels, values.loc[ids]) for name, values in all_scores.items()})
    result["label_examples_both_pipelines"] = {
        pid: {"CART_label": int(info.loc[pid, "FA"]),
              "TAT_label": int(reference.loc[pid, "FA"])}
        for pid in result["label_examples"]}
    example_ids = [matrix.index[matrix.FishIntroduction__sum.gt(0)][0],
                   matrix.index[matrix.FishIntroduction__sum.eq(0)][0],
                   matrix.index[(matrix.GI__sum + matrix.Vomiting__sum).gt(0)][0]]
    no_symptoms = matrix.index[(matrix.GI__sum + matrix.Vomiting__sum).eq(0)]
    result["patients_without_any_symptoms"] = len(no_symptoms)
    if len(no_symptoms):
        example_ids.append(no_symptoms[0])
    for feature in ("GI__sum", "Vomiting__sum"):
        absent = matrix.index[matrix[feature].eq(0)]
        if len(absent):
            example_ids.append(absent[0])
    no_post_fish = info.index[info.documented & rule.eq(0)]
    result["example_no_post_fish_symptoms"] = str(no_post_fish[0])
    result["manual_counts"] = {}
    for pid in dict.fromkeys(example_ids):
        counts = {f"{feature}__sum": len(histories[pid].get(feature, []))
                  for feature in baseline.EVENT_FEATURES}
        assert counts == matrix.loc[pid].to_dict()
        result["manual_counts"][pid] = counts
    signatures = defaultdict(set)
    for pid, history in histories.items():
        signature = tuple((feature, tuple(times)) for feature, times in sorted(history.items()))
        signatures[signature].add(info.loc[pid, "partition"])
    result["full_raw_history_signatures_crossing_split"] = sum(len(v) > 1 for v in signatures.values())
    nonfish_signatures = defaultdict(set)
    for pid, history in histories.items():
        signature = tuple((feature, tuple(times)) for feature, times in sorted(history.items())
                          if feature != "FishIntroduction")
        nonfish_signatures[signature].add(info.loc[pid, "partition"])
    result["nonfish_raw_history_signatures_crossing_split"] = sum(
        len(v) > 1 for v in nonfish_signatures.values())
    result["duplicate_raw_records"] = int(raw.duplicated().sum())
    rng = np.random.default_rng(20261014)
    permutation_aucs = []
    for _ in range(permutations):
        permuted = DecisionTreeClassifier(**metadata["model_parameters"])
        permuted.fit(matrix.loc[train], rng.permutation(info.loc[train, "FA"].to_numpy()))
        permutation_aucs.append(roc_auc_score(info.loc[test, "FA"],
                                             permuted.predict_proba(matrix.loc[test])[:, 1]))
    result["train_label_permutation_aucs"] = permutation_aucs
    rng = np.random.default_rng(20261014)
    blocks = [np.flatnonzero(info.loc[test, "block"].to_numpy() == block)
              for block in sorted(info.loc[test, "block"].unique())]
    draws = []
    labels = info.loc[test, "FA"].to_numpy()
    tat_scores = all_scores["dynamic_on"].to_numpy()
    for _ in range(bootstrap_repeats):
        indices = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        if len(np.unique(labels[indices])) != 2:
            continue
        cart_auc = roc_auc_score(labels[indices], scores[indices])
        tat_auc = roc_auc_score(labels[indices], tat_scores[indices])
        draws.append((cart_auc, tat_auc, tat_auc-cart_auc))
    result["paired_block_bootstrap"] = dict(seed=20261014, repeats=len(draws),
        intervals_95=dict(zip(("cart", "tat", "difference"),
                             np.quantile(draws, [.025, .975], axis=0).T.tolist())))
    result["roc_coordinates"] = {}
    for name, values in (("CART", scores), ("TAT Dynamic ON", tat_scores)):
        fpr, tpr, thresholds = roc_curve(labels, values)
        result["roc_coordinates"][name] = dict(fpr=fpr.tolist(), tpr=tpr.tolist(),
            thresholds=[float(x) if np.isfinite(x) else "infinity" for x in thresholds])
    fig, axis = baseline.plt.subplots()
    for name, values in (("CART", scores), ("TAT Dynamic ON", tat_scores)):
        fpr, tpr, _ = roc_curve(labels, values)
        axis.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(labels, values):.2f})")
    axis.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
    axis.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
             title="Experiment 2 – CART vs TAT Dynamic ON")
    axis.legend()
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=300)
    baseline.plt.close(fig)
    buffer.seek(0)
    result["roc_figure_pixels_identical"] = np.array_equal(
        baseline.plt.imread(buffer), baseline.plt.imread(cart_run / "roc_cart_vs_dynamic_on.png"))
    process = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                              "--cart-run", str(cart_run), "--child"],
                             check=True, capture_output=True, text=True)
    result["fresh_process"] = json.loads(process.stdout)
    assert result["fresh_process"]["cart_prediction_sha256"] == result["cart_prediction_sha256"]
    assert all(digest(Path(path)) == value for path, value in before.items())
    assert set(before) == {str(path) for folder in (recorded, cart_run)
                           for path in folder.rglob("*") if path.is_file()}
    result["experiment_artifacts_unchanged"] = True
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cart-run", type=Path, default=DEFAULT_CART_RUN)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--permutations", type=int, default=10)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    print(json.dumps(audit(arguments.cart_run, arguments.bootstrap_repeats,
                           arguments.permutations, arguments.child), indent=2))
