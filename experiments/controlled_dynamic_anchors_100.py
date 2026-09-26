"""Controlled both-branch anchor comparison; execute explicitly to generate results.

25 matched quartets preserve symptom and treatment marginal distributions by
label. Each patient has exactly one observation per feature (value 1). Fixed
windows distinguish early/late cohorts, never members of a quartet. Dynamic
windows may distinguish the paired versus crossed event relationships.

No experiment executes on import. Outputs are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from types import MethodType

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from settree.temporal_tree import TemporalTree  # noqa: E402

BIRTH = "Birth Date Anchor"
NOW = "NOW Anchor"
SEED = 42


def write_json(path: Path, value) -> None:
    """Write JSON with native NumPy and timestamp conversion."""

    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, pd.Timestamp):
            return item.isoformat()
        raise TypeError(f"Cannot serialize {type(item)}")

    path.write_text(
        json.dumps(value, indent=2, default=convert, allow_nan=False) + "\n"
    )


def generate_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return 100 four-row patients and a separate ground-truth/split table."""
    rng = np.random.default_rng(SEED)
    # One held-out quartet per boundary scenario: three early and two late.
    test_groups = {
        int(
            rng.choice(
                [g for g in range(25) if g % 5 == case and (g < 15) == (case < 3)]
            )
        )
        for case in range(5)
    }
    rows, patients = [], []
    birth, now = pd.Timestamp("2020-01-01"), pd.Timestamp("2022-01-01")
    second = pd.Timedelta(seconds=1)
    for group in range(25):
        early = group < 15
        origin = pd.Timestamp("2020-03-31" if early else "2021-03-31")
        a = origin + pd.Timedelta(days=5 * (group if early else group - 15))
        end = a + pd.DateOffset(months=1)
        case = group % 5
        if case == 0:
            ta, b = a + pd.Timedelta(days=7), a + pd.DateOffset(months=2)
            tb = b + pd.Timedelta(days=7)
        elif case == 1:
            ta, b = end, end + pd.Timedelta(days=1)
            tb = b + pd.Timedelta(days=7)
        elif case == 2:
            ta, b, tb = end - second, end, end + second
        elif case == 3:
            ta, b = a + second, a + pd.DateOffset(months=2)
            tb = b + pd.DateOffset(months=1)
        else:
            ta = a + pd.Timedelta(days=7)
            b, tb = ta + second, end + second
        # Labels are calculated from dates, never chosen to favor a model.
        for member, (symptom, treatment) in enumerate(
            [(a, ta), (b, tb), (a, tb), (b, ta)]
        ):
            pid = f"P{group * 4 + member:03d}"
            endpoint = symptom + pd.DateOffset(months=1)
            label = int(symptom < treatment <= endpoint)
            patients.append(
                dict(
                    patient_id=pid,
                    group=group,
                    scenario=case,
                    cohort="early" if early else "late",
                    partition="test" if group in test_groups else "train",
                    symptom_time=symptom,
                    treatment_time=treatment,
                    window_end=endpoint,
                    label=label,
                    relation="within"
                    if label
                    else "before"
                    if treatment < symptom
                    else "late",
                )
            )
            # Feature order is explicit: equal-score candidates retain the first.
            for feature, time, value in [
                ("symptom", symptom, 1),
                ("treatment", treatment, 1),
                (BIRTH, birth, birth),
                (NOW, now, now),
            ]:
                rows.append(
                    dict(patient_id=pid, time=time, feature_name=feature, value=value)
                )
    data, truth = pd.DataFrame(rows), pd.DataFrame(patients)
    require(
        len(truth) == 100 and truth.label.sum() == 50, "Expected 100 balanced patients"
    )
    require(truth.groupby("group").label.sum().eq(2).all(), "Unbalanced quartet")
    for _, group_data in truth.groupby("group"):
        for feature in ["symptom_time", "treatment_time"]:
            require(
                sorted(group_data.loc[group_data.label.eq(0), feature])
                == sorted(group_data.loc[group_data.label.eq(1), feature]),
                "Event-time marginal imbalance",
            )
    require(truth.partition.eq("test").sum() == 20, "Expected an 80/20 grouped split")
    require(
        data.groupby("patient_id").size().eq(4).all(), "Observation density differs"
    )
    # Every quartet member has the same signature under all fixed windows.
    for _, quartet in truth.groupby("group"):
        for feature in ["symptom_time", "treatment_time"]:
            for source, offset in [(birth, 12), (birth, 24), (now, -12), (now, -24)]:
                end = source + pd.DateOffset(months=offset)
                membership = quartet[feature].between(
                    min(source, end), max(source, end)
                )
                require(
                    membership.nunique() == 1,
                    "Fixed windows distinguish quartet members",
                )
    return data, truth


def require(condition: bool, message: str) -> None:
    """Validate the experiment design, even when Python assertions are disabled."""
    if not condition:
        raise ValueError(message)


def make_model(enabled: bool) -> TemporalTree:
    """Identical exhaustive MSE settings except for the dynamic-anchor switch."""
    return TemporalTree(
        is_classification=False,
        criterion="mse",
        max_depth=3,
        min_samples_split=4,
        min_samples_leaf=2,
        min_impurity_decrease=0.0,
        anchors_list=[BIRTH, NOW],
        anchor_offset_list=[1],
        anchor_offset_map={BIRTH: [12, 24], NOW: [-12, -24]},
        aggregation_funcs={"sum": np.sum},
        feature_semantics={
            "symptom": "binary_event_indicator",
            "treatment": "binary_event_indicator",
        },
        use_dynamic_anchors=enabled,
        candidate_sampling_rate=1.0,
        random_state=SEED,
        split_sampling_seed=SEED,
        learn_missing_direction=False,
        verbose=0,
    )


def anchor_values(data: pd.DataFrame) -> dict:
    """Extract path anchor rows without changing the event data."""
    anchors = data.loc[data.feature_name.str.endswith("Anchor")]
    require(
        not anchors.feature_name.duplicated().any(), "Duplicate patient anchor rows"
    )
    return dict(zip(anchors.feature_name, anchors.value))


def eligible_time(data: pd.DataFrame, node):
    """Independent single-event window oracle, not the production selector."""
    observation = data.loc[data.feature_name.eq(node.feature), "time"]
    require(len(observation) == 1, "Audit expects one observation per feature")
    time = observation.iloc[0]
    source, offset = node.time_window
    if source is None or offset is None:
        return time
    reference = anchor_values(data)[source]
    references = reference if isinstance(reference, list) else [reference]
    for start in references:
        stop = start + pd.DateOffset(months=offset)
        if min(start, stop) <= time <= max(start, stop):
            return time
    return None


def same_number(a, b) -> bool:
    """Compare aggregate outputs including missing windows."""
    return bool((pd.isna(a) and pd.isna(b)) or a == b)


def tree_records(root) -> tuple[list[dict], dict]:
    """Export stable preorder IDs, including leaf means and missing routing."""
    records, ids = [], {}

    def visit(node):
        index = len(records)
        ids[id(node)] = index
        record = dict(node_id=index, depth=node.depth, patients=node.num_patients)
        records.append(record)
        if node.is_leaf():
            record.update(type="leaf", score=float(node.value))
        else:
            record.update(
                type="split",
                feature=node.feature,
                aggregation=node.aggregation,
                window=node.time_window,
                threshold=float(node.threshold),
                missing_direction=node.missing_direction,
                available_anchors=node.anchors_list,
            )
            record["left"] = visit(node.left)
            record["right"] = visit(node.right)
        return index

    visit(root)
    return records, ids


def run_model(
    enabled: bool, data: pd.DataFrame, truth: pd.DataFrame, output: Path
) -> dict:
    """Fit normally while recording actual branch preparation; audit predictions."""
    model = make_model(enabled)
    training = truth.loc[truth.partition.eq("train")]
    train_data = data.loc[data.patient_id.isin(training.patient_id)]
    # Labels must follow first-occurrence patient order, rather than row indices.
    labels = (
        truth.set_index("patient_id")
        .label.loc[train_data.patient_id.unique()]
        .reset_index(drop=True)
    )
    captured = {}
    original = model._prepare_branch_anchor

    def record_prepare(self, **kwargs):
        result = original(**kwargs)
        name, available, created, _ = result
        for pid, incoming in kwargs["X_branch"].groupby("patient_id", sort=False):
            additions = (
                created.loc[created.patient_id.eq(pid)]
                if not created.empty
                else created
            )
            outgoing = (
                incoming
                if additions.empty
                else pd.concat([incoming, additions], ignore_index=True)
            )
            captured[(id(kwargs["node"]), pid)] = dict(
                direction=kwargs["direction"],
                available=list(available),
                before=anchor_values(incoming),
                after=anchor_values(outgoing),
            )
        return result

    model._prepare_branch_anchor = MethodType(record_prepare, model)
    try:
        model.fit(train_data, labels)
    finally:
        del model._prepare_branch_anchor
    nodes, node_ids = tree_records(model.tree)
    write_json(output / "tree.json", nodes)
    lines = []
    for n in nodes:
        detail = (
            f"score={n['score']:.4f}"
            if n["type"] == "leaf"
            else f"{n['aggregation']}({n['feature']}) <= {n['threshold']} "
            f"window={n['window']} LEFT={n['left']} RIGHT={n['right']} "
            f"missing={n['missing_direction']}"
        )
        lines.append(
            f"{'  ' * n['depth']}node {n['node_id']} n={n['patients']}: {detail}"
        )
    (output / "tree.txt").write_text("\n".join(lines) + "\n")

    audits, patient_rows, errors = [], [], []
    for pid, patient_data in data.groupby("patient_id", sort=False):
        info = truth.set_index("patient_id").loc[pid]
        node, path = model.tree, patient_data.copy()
        available, creators = list(model.anchors_list), {}
        symptom_dates, reasons = [], []
        while not node.is_leaf():
            node_id = node_ids[id(node)]
            before = anchor_values(path)
            eligible = eligible_time(path, node)
            aggregate = float(
                model._aggregate_feature_over_time(
                    path, node.feature, node.time_window, node.aggregation
                )[0][0]
            )
            direction = model._get_prediction_direction(aggregate, node)
            name = model._format_dynamic_anchor_name(node, direction)
            next_path = model._append_prediction_branch_anchor(path, node, direction)
            after = anchor_values(next_path)
            expected = (
                [eligible]
                if enabled and eligible is not None
                else []
            )
            expected_available = list(available)
            if enabled and name not in expected_available:
                expected_available.append(name)
            actual = after.get(name, [])
            reason = (
                "disabled"
                if not enabled
                else "no_eligible_timestamp"
                if eligible is None
                else "reused"
                if name in before
                else "created"
            )
            checks = dict(
                aggregate_correct=same_number(
                    aggregate, 1.0 if eligible is not None else np.nan
                ),
                selected_timestamp_correct=actual == expected,
                earlier_anchors_preserved=all(
                    key in after and after[key] == value
                    for key, value in before.items()
                ),
                node_candidates_match_path=list(node.anchors_list) == available,
                no_unexpected_anchor_rows=set(after)
                == set(before) | ({name} if enabled else set()),
                unavailable_anchor_has_no_timestamp=eligible is not None or actual == [],
            )
            training_record = captured.get((id(node), pid))
            if info.partition == "train":
                checks["training_prediction_consistent"] = bool(
                    training_record is not None
                    and training_record["direction"] == direction
                    and training_record["before"] == before
                    and training_record["after"] == after
                    and training_record["available"] == expected_available
                )
            source = node.time_window[0]
            creator = creators.get(source)
            symptom_window = bool(
                node.feature == "treatment"
                and creator is not None
                and creator[1] == "symptom"
            )
            record = dict(
                patient_id=pid,
                partition=info.partition,
                node_id=node_id,
                feature=node.feature,
                aggregation=node.aggregation,
                source_anchor=source,
                offset=node.time_window[1],
                threshold=node.threshold,
                aggregate=None if pd.isna(aggregate) else aggregate,
                direction=direction,
                anchor_name=name if enabled else None,
                anchor_timestamps=actual,
                reason=reason,
                available_before=available,
                available_after=expected_available,
                source_created_at_node=creator[0] if creator else None,
                treatment_uses_symptom_anchor=symptom_window,
                relative_window_contains_treatment=bool(aggregate == 1)
                if symptom_window
                else None,
                checks=checks,
            )
            audits.append(record)
            errors.extend(
                f"{pid}/node {node_id}: {key}" for key, ok in checks.items() if not ok
            )
            if enabled and name not in before:
                creators[name] = (node_id, node.feature)
            if node.feature == "symptom":
                reasons.append(reason)
                if actual:
                    symptom_dates.extend(actual)
            available, path = expected_available, next_path
            node = node.left if direction == "left" else node.right
        score = float(model.predict(patient_data)[0])
        leaf_id = int(model.apply(patient_data)[0])
        if score != float(node.value) or leaf_id != node.sn:
            errors.append(f"{pid}: prediction/leaf audit mismatch")
        patient_trace = [r for r in audits if r["patient_id"] == pid]
        actual_windows = [
            r["relative_window_contains_treatment"]
            for r in patient_trace
            if r["treatment_uses_symptom_anchor"]
        ]
        symptom_creators = [
            r["node_id"]
            for r in patient_trace
            if r["feature"] == "symptom" and r["anchor_timestamps"]
        ]
        patient_rows.append(
            dict(
                patient_id=pid,
                score=score,
                predicted_label=int(score >= 0.5),
                anchor_timestamp=symptom_dates[0] if symptom_dates else None,
                symptom_anchor_creator_nodes=json.dumps(symptom_creators),
                actual_relative_window_results=json.dumps(actual_windows),
                symptom_anchor_reason=";".join(reasons)
                if reasons
                else "no_valid_initial_split"
                if model.tree.is_leaf()
                else "no_symptom_split_on_path",
                expected_relative_window=bool(
                    info.symptom_time < info.treatment_time <= info.window_end
                ),
                actual_treatment_window_evaluated=bool(actual_windows),
            )
        )
    audit_table = truth.merge(
        pd.DataFrame(patient_rows), on="patient_id", validate="one_to_one"
    )
    audit_table.to_csv(output / "patient_audit.csv", index=False)
    write_json(output / "anchor_trace.json", audits)
    print(
        audit_table.loc[
            audit_table.partition.eq("test"),
            [
                "patient_id",
                "symptom_time",
                "treatment_time",
                "label",
                "predicted_label",
                "anchor_timestamp",
                "symptom_anchor_creator_nodes",
                "actual_relative_window_results",
            ],
        ].to_string(index=False)
    )
    metrics = {}
    for partition, group in audit_table.groupby("partition"):
        metrics[partition] = dict(
            n=len(group),
            roc_auc=roc_auc_score(group.label, group.score),
            accuracy=accuracy_score(group.label, group.predicted_label),
            precision=precision_score(
                group.label, group.predicted_label, zero_division=0
            ),
            recall=recall_score(group.label, group.predicted_label, zero_division=0),
            f1=f1_score(group.label, group.predicted_label, zero_division=0),
            confusion_matrix=confusion_matrix(
                group.label, group.predicted_label, labels=[0, 1]
            ).tolist(),
        )
    write_json(output / "metrics.json", metrics)
    summary = dict(
        dynamic_anchors=enabled,
        mechanism_checks_passed=not errors,
        mechanism_status="checked" if audits else "not_exercised_no_split",
        failures=errors,
        visited_split_count=len(audits),
        patients_with_symptom_anchor=int(audit_table.anchor_timestamp.notna().sum()),
        treatment_nodes_using_symptom_anchor=sorted(
            {r["node_id"] for r in audits if r["treatment_uses_symptom_anchor"]}
        ),
        test_patients_with_relative_treatment_split=int(
            audit_table.loc[
                audit_table.partition.eq("test"), "actual_treatment_window_evaluated"
            ].sum()
        ),
        metrics=metrics,
    )
    write_json(output / "diagnostics.json", summary)
    return summary


def main() -> None:
    """Generate, validate, fit, and audit only when the user runs this script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/results/controlled_dynamic_anchors_100",
    )
    args = parser.parse_args()
    require(not args.output.exists(), "Output exists; choose a new --output directory")
    data, truth = generate_data()
    # Preflight the actual root candidate. This is evaluated only on execution.
    probe = make_model(False)
    training = truth.loc[truth.partition.eq("train")]
    train_data = data.loc[data.patient_id.isin(training.patient_id)]
    aggregates = probe._aggregate_feature_over_time(
        train_data, "symptom", (BIRTH, 12), "sum"
    )[0]
    require(
        int((aggregates <= 1).sum()) == 48 and int(pd.isna(aggregates).sum()) == 32,
        "Expected valid 48/32 root split with missing values routed RIGHT",
    )
    args.output.mkdir(parents=True)
    data.to_csv(args.output / "observations.csv", index=False)
    truth.to_csv(args.output / "patients_and_split.csv", index=False)
    write_json(
        args.output / "configuration.json",
        dict(
            seed=SEED,
            parameters={
                k: v
                for k, v in make_model(True).get_params().items()
                if k != "aggregation_funcs"
            },
            aggregations=["sum"],
            decision_threshold=0.5,
            versions=dict(
                python=platform.python_version(),
                numpy=np.__version__,
                pandas=pd.__version__,
                sklearn=sklearn.__version__,
            ),
            sha256={
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [
                    Path(__file__).resolve(),
                    ROOT / "settree/temporal_tree.py",
                    ROOT / "settree/anchor_utils.py",
                ]
            },
        ),
    )
    summaries = {}
    for enabled in [True, False]:
        name = "dynamic_on" if enabled else "dynamic_off"
        output = args.output / name
        output.mkdir()
        summaries[name] = run_model(enabled, data, truth, output)
        print(name, json.dumps(summaries[name]["metrics"]["test"], indent=2))
        print((output / "tree.txt").read_text())
    write_json(args.output / "comparison.json", summaries)
    diagnostic = [
        "A. Anchor correctness: see mechanism_checks_passed and failures in diagnostics.json.",
        "Checks compare independent event/window expectations, recorded training branch outputs,",
        "prediction outputs, inherited rows, and the actual next-node candidate lists.",
        "B. Learning: see treatment_nodes_using_symptom_anchor and patient coverage.",
        "A correct but unused anchor does not establish that the relationship was learned.",
        "C. Performance: metrics.json reports train/test metrics; score >= 0.5 predicts 1.",
        "Confusion matrices use rows=true, columns=predicted, class order [0, 1].",
        "Design: 25 matched quartets, 50 positive / 25 before / 25 late; grouped 80/20 split.",
        "The coarse fixed windows cannot distinguish quartet members. Root candidates have zero",
        "immediate gain, permitted by min_impurity_decrease=0. Symptom is first in tie order.",
        "LEFT receives early-cohort symptom anchors. The late RIGHT cohort can remain a mixed",
        "leaf because its remaining fixed-window aggregates are constant. This is a limitation",
        "of greedy training with post-split anchors, not evidence of incorrect propagation.",
        "Calendar windows include both endpoints. No simultaneous symptom/treatment pairs exist.",
        "No test labels are used for fitting, threshold choice, or hyperparameter tuning.",
    ]
    for name, summary in summaries.items():
        diagnostic.append(
            f"{name}: checks={summary['mechanism_checks_passed']}; "
            f"symptom-anchor patients={summary['patients_with_symptom_anchor']}; "
            f"relative treatment nodes={summary['treatment_nodes_using_symptom_anchor']}; "
            f"test accuracy={summary['metrics']['test']['accuracy']:.3f}"
        )
    (args.output / "diagnostic_summary.txt").write_text("\n".join(diagnostic) + "\n")
    print(f"Saved results to {args.output}")
    if any(not s["mechanism_checks_passed"] for s in summaries.values()):
        raise SystemExit(
            "Audit failures recorded; inspect anchor_trace.json and diagnostics.json"
        )


if __name__ == "__main__":
    main()
