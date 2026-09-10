import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments import experiment_2_robustness_v2 as experiment
from settree.temporal_tree import TemporalLeaf, TemporalSplitNode, TemporalTree


VALID_CONFIGURATION = {
    "candidate_id": "C02",
    "candidate_sampling_rate": 0.4,
    "max_depth": 3,
    "min_samples_split": 40,
    "min_samples_leaf": 20,
    "min_impurity_decrease": 0.0,
}


def _path_table(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "patient_id",
            "anchor",
            "stored_offset",
            "is_full_timeline",
        ],
    )


def _split_tree(anchor="NOW Anchor", offset=-1):
    root = TemporalSplitNode(
        feature="signal",
        threshold=2.5,
        aggregation="mean",
        time_window=(anchor, offset),
        missing_direction="right",
        num_patients=10,
        depth=0,
        impurity=0.2,
    )
    root.left = TemporalLeaf(value=0.1, num_patients=5, depth=1)
    root.right = TemporalLeaf(value=0.8, num_patients=5, depth=1)
    model = TemporalTree()
    model.tree = root
    return model


def _per_seed_table():
    rows = []
    for outcome in ("FA", "GI"):
        for index, seed in enumerate(experiment.SPLIT_SAMPLING_SEEDS):
            is_fa = outcome == "FA"
            rows.append(
                {
                    "split_sampling_seed": seed,
                    "outcome": outcome,
                    "roc_auc": 0.60 + index / 100,
                    "birth_usage_percentage": 70.0 if is_fa else 10.0,
                    "now_usage_percentage": 20.0 if is_fa else 80.0,
                    "full_timeline_usage_percentage": 10.0,
                    "expected_anchor_dominant": True,
                    "root_is_leaf": False,
                    "root_feature": "signal",
                    "root_aggregation": "mean",
                    "root_anchor": (
                        "Birth Date Anchor" if is_fa else "NOW Anchor"
                    ),
                    "root_offset": 1 if is_fa else -1,
                    "root_is_full_timeline": False,
                    "now_minus_1_present_in_tree": not is_fa,
                    "now_minus_1_used_on_test_paths": not is_fa,
                    "now_minus_1_path_usage_percentage": (
                        np.nan if is_fa else float(index)
                    ),
                    "now_minus_1_patient_coverage_percentage": (
                        np.nan if is_fa else float(index * 2)
                    ),
                }
            )
    return pd.DataFrame(rows)


def _window_rows():
    rows = []
    for outcome in ("FA", "GI"):
        for seed in experiment.SPLIT_SAMPLING_SEEDS:
            for window in experiment.effective_windows():
                used = (
                    window["anchor"] == "Birth Date Anchor"
                    and window["offset"] == 1
                    and outcome == "FA"
                ) or (
                    window["anchor"] == "NOW Anchor"
                    and window["offset"] == -1
                    and outcome == "GI"
                    and seed < 47
                )
                rows.append(
                    {
                        "split_sampling_seed": seed,
                        "outcome": outcome,
                        **window,
                        "path_usage_count": int(used),
                        "path_usage_percentage": 20.0 if used else 0.0,
                        "patient_count": int(used),
                        "patient_coverage_percentage": (
                            10.0 if used else 0.0
                        ),
                    }
                )
    return pd.DataFrame(rows)


def test_configuration_is_explicit_and_placeholders_block_execution():
    assert experiment.SELECTED_MODEL_CONFIGURATION == {
        "candidate_id": "REPLACE_AFTER_SEARCH_COMPLETES",
        "candidate_sampling_rate": None,
        "max_depth": None,
        "min_samples_split": None,
        "min_samples_leaf": None,
        "min_impurity_decrease": None,
    }

    with pytest.raises(ValueError, match="placeholders remain"):
        experiment.validate_selected_configuration()


def test_design_has_exact_seeds_offsets_aggregations_and_fixed_anchors():
    experiment.validate_experiment_design(selected=VALID_CONFIGURATION)

    assert experiment.SPLIT_SAMPLING_SEEDS == tuple(range(42, 52))
    assert experiment.ANCHOR_OFFSET_MAP == experiment.EXPECTED_OFFSET_MAP
    assert experiment.AGGREGATION_NAMES == ("mean", "sum", "max", "count")
    assert experiment.FIXED_MODEL_CONFIGURATION["use_dynamic_anchors"] is False
    assert len(experiment.effective_windows()) == 18
    assert sum(
        window["is_full_timeline"]
        for window in experiment.effective_windows()
    ) == 1


@pytest.mark.parametrize(
    "change",
    [
        {"candidate_sampling_rate": 0},
        {"max_depth": 0},
        {"min_samples_split": 10, "min_samples_leaf": 20},
        {"min_impurity_decrease": -0.1},
    ],
)
def test_invalid_selected_model_values_are_rejected(change):
    configuration = {**VALID_CONFIGURATION, **change}

    with pytest.raises(ValueError):
        experiment.validate_selected_configuration(configuration)


def test_model_kwargs_differ_only_by_split_sampling_seed():
    first = experiment.model_parameters(42, selected=VALID_CONFIGURATION)
    second = experiment.model_parameters(51, selected=VALID_CONFIGURATION)

    differing = {
        key for key in first if first[key] != second[key]
    }
    assert differing == {"split_sampling_seed"}
    assert first["candidate_sampling_rate"] == 0.4
    assert first["anchor_offset_map"] == experiment.ANCHOR_OFFSET_MAP


def test_shared_split_is_deterministic_disjoint_and_joint_stratified():
    patients = [f"P{index:03d}" for index in range(40)]
    data = pd.DataFrame({"patient_id": patients})
    labels = {
        "fa": pd.DataFrame(
            {
                "patient_id": patients,
                "label": [index % 2 for index in range(40)],
            }
        ),
        "gi": pd.DataFrame(
            {
                "patient_id": patients,
                "label": [(index // 2) % 2 for index in range(40)],
            }
        ),
    }

    first = experiment.create_shared_split(data, labels)
    second = experiment.create_shared_split(data, labels)
    pd.testing.assert_frame_equal(first, second)

    train = set(experiment.split_patient_ids(first, "train"))
    test = set(experiment.split_patient_ids(first, "test"))
    assert not train & test
    assert train | test == set(patients)
    assert len(train) == 28
    assert len(test) == 12
    assert set(first["joint_stratum"]) == {"0_0", "0_1", "1_0", "1_1"}


def test_subset_aligns_labels_to_first_seen_patient_order():
    data = pd.DataFrame(
        {
            "patient_id": ["P2", "P2", "P1", "P1"],
            "value": [1, 2, 3, 4],
        }
    )
    labels = pd.DataFrame(
        {"patient_id": ["P1", "P2"], "label": [0, 1]}
    )

    _, aligned, order = experiment.subset_for_patients(
        data, labels, ["P1", "P2"]
    )

    assert order.tolist() == ["P2", "P1"]
    assert aligned.tolist() == [1, 0]


def test_anchor_usage_window_zero_fill_and_strict_dominance():
    paths = _path_table(
        [
            ("P1", "Birth Date Anchor", 1, False),
            ("P1", "Birth Date Anchor", 1, False),
            ("P2", "NOW Anchor", -1, False),
            ("P2", None, None, True),
        ]
    )

    usage = experiment.anchor_usage(paths)
    windows = experiment.temporal_window_usage(paths, ["P1", "P2"], 42, "FA")

    assert usage["birth_usage_percentage"] == 50.0
    assert usage["now_usage_percentage"] == 25.0
    assert usage["full_timeline_usage_percentage"] == 25.0
    assert experiment.expected_anchor_dominance(usage, "fa") is True
    assert experiment.expected_anchor_dominance(usage, "gi") is False
    assert experiment.dominant_anchor(usage) == "Birth Date Anchor"
    assert len(windows) == 18
    unused = windows.loc[
        windows["anchor"].eq("Birth Date Anchor")
        & pd.to_numeric(windows["offset"]).eq(18)
    ].iloc[0]
    assert unused["path_usage_count"] == 0
    assert unused["path_usage_percentage"] == 0

    tie = {
        "total_path_split_usages": 2,
        "birth_usage_percentage": 50.0,
        "now_usage_percentage": 50.0,
        "full_timeline_usage_percentage": 0.0,
    }
    assert experiment.expected_anchor_dominance(tie, "fa") is False
    assert experiment.dominant_anchor(tie) is None


def test_no_split_usage_is_zero_and_not_dominant():
    paths = _path_table([])
    usage = experiment.anchor_usage(paths)

    assert usage == {
        "total_path_split_usages": 0,
        "birth_usage_percentage": 0.0,
        "now_usage_percentage": 0.0,
        "full_timeline_usage_percentage": 0.0,
    }
    assert experiment.expected_anchor_dominance(usage, "fa") is False
    assert experiment.expected_anchor_dominance(usage, "gi") is False


def test_now_minus_one_distinguishes_tree_presence_and_test_path_use():
    model = _split_tree()
    paths = _path_table(
        [
            ("P1", "NOW Anchor", -1, False),
            ("P2", "NOW Anchor", -1, False),
            ("P2", "Birth Date Anchor", 1, False),
            ("P3", None, None, True),
        ]
    )

    result = experiment.now_minus_one_behavior(model, paths, 4)

    assert result["now_minus_1_present_in_tree"] is True
    assert result["now_minus_1_used_on_test_paths"] is True
    assert result["now_minus_1_path_usage_count"] == 2
    assert result["now_minus_1_path_usage_percentage"] == 50.0
    assert result["now_minus_1_patient_coverage_percentage"] == 50.0

    no_test_use = experiment.now_minus_one_behavior(
        model, paths.iloc[0:0], 4
    )
    assert no_test_use["now_minus_1_present_in_tree"] is True
    assert no_test_use["now_minus_1_used_on_test_paths"] is False


def test_root_information_and_tree_export_are_complete(tmp_path):
    model = _split_tree()
    output = tmp_path / "tree.json"

    root = experiment.root_split_information(model)
    experiment.export_tree_json(model, output)
    nodes = json.loads(output.read_text(encoding="utf-8"))

    assert root["root_feature"] == "signal"
    assert root["root_anchor"] == "NOW Anchor"
    assert root["root_offset"] == -1
    assert root["root_threshold"] == 2.5
    assert [node["type"] for node in nodes] == ["split", "leaf", "leaf"]
    assert nodes[0]["left_child"] == 1
    assert nodes[0]["right_child"] == 2
    assert nodes[0]["missing_direction"] == "right"


def test_candidate_sampling_counter_extraction_includes_fallbacks():
    model = TemporalTree()
    values = {
        "generated_groups": 100,
        "sampled_groups": 40,
        "sampling_skipped_groups": 60,
        "generated_threshold_candidates": 1000,
        "invalid_threshold_candidates": 100,
        "valid_threshold_candidates": 900,
        "evaluated_threshold_candidates": 900,
        "fallback_evaluated_groups": 2,
        "fallback_generated_threshold_candidates": 20,
        "fallback_valid_threshold_candidates": 18,
        "fallback_evaluated_threshold_candidates": 18,
        "nodes_requiring_sampling_fallback": 2,
    }
    for name, value in values.items():
        setattr(model, name, value)

    counters = experiment.candidate_sampling_counters(model)

    assert counters["observed_group_sampling_rate"] == 0.4
    assert counters["total_evaluated_candidates"] == 918
    assert counters["fallback_evaluated_groups"] == 2


def test_cross_seed_summary_has_only_approved_descriptive_results():
    summary = experiment.summarize_per_seed(_per_seed_table())

    auc = summary.loc[
        summary["outcome"].eq("FA")
        & summary["metric"].eq("roc_auc")
    ].set_index("statistic")["value"]
    assert auc["mean"] == pytest.approx(0.645)
    assert auc["sd"] == pytest.approx(
        np.std(np.arange(0.60, 0.70, 0.01), ddof=1)
    )
    assert auc["min"] == pytest.approx(0.60)
    assert auc["max"] == pytest.approx(0.69)
    assert "score" not in set(summary["section"])
    assert set(summary["n_seeds"]) == {10}
    gi_metrics = set(summary.loc[summary["outcome"].eq("GI"), "metric"])
    assert "now_minus_1_structural" in gi_metrics
    assert "modal_root_signature" in gi_metrics


def test_window_summary_counts_unused_seeds_as_zero():
    summary = experiment.summarize_temporal_windows(_window_rows())
    now_minus_one = summary.loc[
        summary["outcome"].eq("GI")
        & summary["anchor"].eq("NOW Anchor")
        & pd.to_numeric(summary["offset"], errors="coerce").eq(-1)
    ].iloc[0]

    assert now_minus_one["mean_usage_percentage"] == 10.0
    assert now_minus_one["sd_usage_percentage"] == pytest.approx(
        np.std([20.0] * 5 + [0.0] * 5, ddof=1)
    )
    assert now_minus_one["seeds_where_used"] == 5
    assert now_minus_one["percentage_of_seeds_where_used"] == 50.0


def test_figures_are_limited_to_the_two_approved_summaries(tmp_path):
    per_seed = _per_seed_table()
    windows = _window_rows()
    auc_path = tmp_path / "auc_across_seeds.png"
    usage_path = tmp_path / "temporal_usage_across_seeds.png"

    experiment.create_auc_figure(per_seed, auc_path)
    experiment.create_temporal_usage_figure(windows, usage_path)

    assert auc_path.is_file() and auc_path.stat().st_size > 0
    assert usage_path.is_file() and usage_path.stat().st_size > 0
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "auc_across_seeds.png",
        "temporal_usage_across_seeds.png",
    ]


def test_central_manifest_finalizes_and_verifies_completed_seed(tmp_path):
    output_dir = tmp_path / "experiment"
    seeds_dir = output_dir / "seeds"
    attempt_dir = seeds_dir / ".seed_42.attempt.tmp"
    attempt_dir.mkdir(parents=True)
    for name in (
        "FA_tree.json",
        "FA_temporal_window_usage.csv",
        "GI_tree.json",
        "GI_temporal_window_usage.csv",
    ):
        (attempt_dir / name).write_text("artifact", encoding="utf-8")
    rows = [
        {"split_sampling_seed": 42, "outcome": "FA", "roc_auc": 0.6},
        {"split_sampling_seed": 42, "outcome": "GI", "roc_auc": 0.7},
    ]
    manifest = {
        "seeds": {
            str(seed): {"status": "not_started"}
            for seed in experiment.SPLIT_SAMPLING_SEEDS
        }
    }
    manifest["seeds"]["42"] = {
        "status": "finalizing",
        "attempt_directory": str(attempt_dir),
        "pending_rows": rows,
    }

    table = experiment._finalize_seed(output_dir, 42, manifest)

    assert set(table["outcome"]) == {"FA", "GI"}
    assert manifest["seeds"]["42"]["status"] == "complete"
    assert not attempt_dir.exists()
    reloaded = pd.read_csv(output_dir / experiment.PER_SEED_RESULTS_FILENAME)
    assert experiment.verify_completed_seed(
        output_dir, 42, manifest["seeds"]["42"], reloaded
    )

    artifact = output_dir / "seeds" / "seed_42" / "FA_tree.json"
    artifact.write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed artifact"):
        experiment.verify_completed_seed(
            output_dir, 42, manifest["seeds"]["42"], reloaded
        )


def test_nonempty_unrecognized_output_directory_is_rejected(tmp_path):
    output_dir = tmp_path / "experiment"
    output_dir.mkdir()
    (output_dir / "unrelated.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(RuntimeError, match="non-empty"):
        experiment._prepare_run(
            VALID_CONFIGURATION,
            {"x": "x", "fa": "fa", "gi": "gi"},
            pd.DataFrame({"patient_id": ["P1"], "split": ["train"]}),
            output_dir,
        )


def test_cli_requires_explicit_mode_and_retry_requires_run():
    with pytest.raises(SystemExit):
        experiment.parse_args([])
    with pytest.raises(SystemExit):
        experiment.parse_args(["--validate-only", "--retry-failed-seed", "42"])

    args = experiment.parse_args(["--run", "--retry-failed-seed", "42"])
    assert args.run is True
    assert args.retry_failed_seed == 42


def test_placeholder_blocks_before_loading_or_writing(
    monkeypatch,
    tmp_path,
):
    called = False

    def forbidden_load():
        nonlocal called
        called = True
        raise AssertionError(
            "Inputs must not load while configuration is incomplete"
        )

    monkeypatch.setattr(experiment, "load_inputs", forbidden_load)

    with pytest.raises(ValueError, match="placeholders remain"):
        experiment.run_experiment(output_dir=tmp_path / "results")

    assert called is False
    assert not (tmp_path / "results").exists()


def test_mocked_full_run_writes_only_approved_outputs_and_resumes(
    monkeypatch,
    tmp_path,
):
    input_paths = {}
    for name in ("x", "fa", "gi"):
        path = tmp_path / f"{name}.pkl"
        path.write_bytes(name.encode("utf-8"))
        input_paths[name] = path
    output_dir = tmp_path / "experiment_2"
    split = pd.DataFrame(
        {
            "patient_id": ["P1", "P2"],
            "fa_label": [0, 1],
            "gi_label": [1, 0],
            "joint_stratum": ["0_1", "1_0"],
            "split": ["train", "test"],
        }
    )
    fake_data = pd.DataFrame({"patient_id": ["P1", "P2"]})
    fake_labels = {
        outcome: pd.DataFrame(
            {"patient_id": ["P1", "P2"], "label": [0, 1]}
        )
        for outcome in experiment.OUTCOMES
    }
    calls = []

    monkeypatch.setattr(experiment, "INPUT_PATHS", input_paths)
    monkeypatch.setattr(
        experiment,
        "validate_experiment_design",
        lambda: dict(VALID_CONFIGURATION),
    )
    monkeypatch.setattr(
        experiment, "load_inputs", lambda: (fake_data, fake_labels)
    )
    monkeypatch.setattr(
        experiment, "create_shared_split", lambda data, labels: split.copy()
    )
    monkeypatch.setattr(experiment, "source_metadata", lambda: {"test": True})

    def fake_run_outcome(
        outcome,
        split_sampling_seed,
        data,
        labels,
        train_ids,
        test_ids,
        attempt_dir,
    ):
        calls.append((split_sampling_seed, outcome))
        prefix = outcome.upper()
        (attempt_dir / f"{prefix}_tree.json").write_text(
            "[]", encoding="utf-8"
        )
        windows = _window_rows()
        windows = windows.loc[
            windows["split_sampling_seed"].eq(split_sampling_seed)
            & windows["outcome"].eq(prefix)
        ].copy()
        windows.to_csv(
            attempt_dir / f"{prefix}_temporal_window_usage.csv",
            index=False,
        )
        row = _per_seed_table().loc[
            _per_seed_table()["split_sampling_seed"].eq(split_sampling_seed)
            & _per_seed_table()["outcome"].eq(prefix)
        ].iloc[0].to_dict()
        row.update(
            {
                "total_path_split_usages": 10,
                "dominant_anchor": (
                    "Birth Date Anchor" if outcome == "fa" else "NOW Anchor"
                ),
                "root_is_full_timeline": False,
                "root_threshold": 1.5,
                "training_runtime_seconds": 1.0,
                "evaluation_runtime_seconds": 0.1,
                "temporal_analysis_runtime_seconds": 0.1,
                "total_runtime_seconds": 1.2,
                "now_minus_1_path_usage_count": (
                    None if outcome == "fa" else 2
                ),
            }
        )
        for column in (
            "generated_groups",
            "sampled_groups",
            "skipped_groups",
            "generated_threshold_candidates",
            "invalid_threshold_candidates",
            "valid_threshold_candidates",
            "evaluated_threshold_candidates",
            "fallback_evaluated_groups",
            "fallback_generated_threshold_candidates",
            "fallback_valid_threshold_candidates",
            "fallback_evaluated_threshold_candidates",
            "nodes_requiring_sampling_fallback",
            "total_evaluated_candidates",
        ):
            row[column] = 1
        row["observed_group_sampling_rate"] = 0.4
        return row, windows

    monkeypatch.setattr(experiment, "run_outcome", fake_run_outcome)

    first = experiment.run_experiment(output_dir=output_dir)
    second = experiment.run_experiment(output_dir=output_dir)

    assert first["status"] == second["status"] == "complete"
    assert len(calls) == 20
    assert len(pd.read_csv(output_dir / "per_seed_results.csv")) == 20
    assert (output_dir / "robustness_summary.csv").is_file()
    assert (output_dir / "temporal_window_stability.csv").is_file()
    figure_names = sorted(
        path.name for path in (output_dir / "figures").iterdir()
    )
    assert figure_names == [
        "auc_across_seeds.png",
        "temporal_usage_across_seeds.png",
    ]
    for seed in experiment.SPLIT_SAMPLING_SEEDS:
        seed_files = sorted(
            path.name
            for path in (output_dir / "seeds" / f"seed_{seed}").iterdir()
        )
        assert seed_files == [
            "FA_temporal_window_usage.csv",
            "FA_tree.json",
            "GI_temporal_window_usage.csv",
            "GI_tree.json",
        ]


def test_source_contains_only_approved_predictive_and_resume_concepts():
    source = Path(experiment.__file__).read_text(encoding="utf-8").lower()

    assert "roc_auc_score" in source
    assert "combined_score" not in source
    assert "best_seed" not in source
    assert "jaccard" not in source
