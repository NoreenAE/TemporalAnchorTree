import json

import numpy as np
import pandas as pd
import pytest

from experiments import experiment_1_fa_vs_gi as experiment_1
from experiments import search_experiment_1_v2 as search


def _candidate_result(
    candidate,
    seed,
    combined_score,
    fa_temporal=0.5,
    gi_temporal=0.5,
):
    outcomes = {}
    components = {}
    for outcome, temporal in (("fa", fa_temporal), ("gi", gi_temporal)):
        outcomes[outcome] = {
            "performance": {
                "roc_auc": 0.65,
                "average_precision": 0.20,
            },
            "evaluation_positive_prevalence": 0.10,
            "temporal_recovery_score": temporal,
            "temporal_usage": {
                "birth_date_usage_proportion": 0.5,
                "now_usage_proportion": 0.25,
                "full_timeline_usage_proportion": 0.25,
            },
            "window_distribution": [{
                "window": f"{outcome}_window",
                "raw_usage_count": 2,
                "usage_proportion": 0.5,
            }],
            "root_split": {
                "is_leaf": False,
                "anchor": "Birth Date Anchor" if outcome == "fa" else "NOW Anchor",
                "offset": 1 if outcome == "fa" else -1,
            },
        }
        components[f"{outcome}_auc_skill"] = 0.3
        components[f"{outcome}_average_precision_skill"] = 1 / 9
        components[f"{outcome}_temporal_recovery_score"] = temporal
    return {
        "candidate": candidate.to_dict(),
        "split_sampling_seed": seed,
        "stage": "screening" if seed == 42 else "stability",
        "evaluation_split": "validation",
        "outcomes": outcomes,
        "score_components": components,
        "combined_score": combined_score,
        "pair_runtime_seconds": 1.0,
    }


def _path_table(rows):
    return pd.DataFrame(rows, columns=[
        "anchor",
        "effective_start_offset",
        "effective_end_offset",
        "is_full_timeline",
        "is_zero_width",
    ])


def test_screening_design_is_bounded_and_contains_exact_planned_ranges():
    search.validate_search_plan(search.SCREENING_CANDIDATES)

    assert [candidate.candidate_id for candidate in search.SCREENING_CANDIDATES] == [
        "C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08"
    ]
    sampling_rates = {
        candidate.candidate_sampling_rate
        for candidate in search.SCREENING_CANDIDATES
    }
    assert sampling_rates == {
        0.4, 0.6
    }
    assert {candidate.max_depth for candidate in search.SCREENING_CANDIDATES} == {
        3, 4, 5
    }
    impurity_decreases = {
        candidate.min_impurity_decrease
        for candidate in search.SCREENING_CANDIDATES
    }
    assert impurity_decreases == {
        0.0, 0.0005, 0.001, 0.002
    }
    assert all(
        candidate.min_samples_split >= 2 * candidate.min_samples_leaf
        for candidate in search.SCREENING_CANDIDATES
    )
    assert all(
        candidate.model_parameters(42)["use_dynamic_anchors"] is False
        for candidate in search.SCREENING_CANDIDATES
    )
    assert search.ESTIMATED_RUNTIME["maximum_model_pairs"] == 19


def test_search_plan_and_fingerprint_are_deterministic():
    first = search.search_plan()
    second = search.search_plan()

    assert first == second
    assert search.configuration_fingerprint(first) == (
        search.configuration_fingerprint(second)
    )
    assert first["n_screening_configurations"] == 8
    assert first["split_protocol"]["selection_uses_test_set"] is False


def test_outer_test_partition_exactly_matches_experiment_1_logic():
    patient_ids = [f"P{index:03d}" for index in range(40)]
    data = pd.DataFrame({"patient_id": patient_ids})
    labels = {
        "fa": pd.DataFrame({
            "patient_id": patient_ids,
            "label": [index % 2 for index in range(40)],
        }),
        "gi": pd.DataFrame({
            "patient_id": patient_ids,
            "label": [(index // 2) % 2 for index in range(40)],
        }),
    }

    _, experiment_test_ids, _ = experiment_1.create_shared_split(data, labels)
    split_table = search.create_shared_search_splits(data, labels)
    search_test_ids = search.split_patient_ids(split_table, ["test"])

    assert set(search_test_ids) == set(experiment_test_ids)
    assert split_table["split"].value_counts().to_dict() == {
        "search_train": 21,
        "test": 12,
        "validation": 7,
    }


def test_temporal_recovery_uses_geometric_overlap_and_penalizes_distractors():
    fa_paths = _path_table([
        ("Birth Date Anchor", 0.0, 6.0, False, False),
        ("Birth Date Anchor", 0.0, 12.0, False, False),
        ("NOW Anchor", -1.0, 0.0, False, False),
        (None, np.nan, np.nan, True, False),
    ])
    gi_paths = _path_table([
        ("NOW Anchor", -1.0, 0.0, False, False),
        ("NOW Anchor", -2.0, 0.0, False, False),
        ("Birth Date Anchor", 0.0, 1.0, False, False),
        (None, np.nan, np.nan, True, False),
    ])

    assert search.temporal_recovery_score(fa_paths, "fa") == pytest.approx(0.375)
    assert search.temporal_recovery_score(gi_paths, "gi") == pytest.approx(0.375)
    assert search.temporal_recovery_score(fa_paths.iloc[0:0], "fa") == 0.0


def test_predictive_metrics_are_normalized_against_no_skill_baselines():
    auc_skill, ap_skill = search.predictive_skill_scores(
        roc_auc=0.65,
        average_precision=0.19,
        prevalence=0.10,
    )

    assert auc_skill == pytest.approx(0.30)
    assert ap_skill == pytest.approx(0.10)
    assert search.predictive_skill_scores(0.4, 0.05, 0.10) == (0.0, 0.0)


def test_finalist_selection_protects_best_temporal_candidate():
    results = []
    for index, candidate in enumerate(search.SCREENING_CANDIDATES[:6]):
        temporal = 0.99 if candidate.candidate_id == "C06" else 0.2
        results.append(_candidate_result(
            candidate,
            seed=42,
            combined_score=0.9 - index * 0.1,
            fa_temporal=temporal,
            gi_temporal=temporal,
        ))

    assert search.select_finalist_ids(results) == ["C01", "C02", "C03", "C06"]


def test_multi_seed_summary_records_variability_roots_windows_and_rank():
    first = search.SCREENING_CANDIDATES[0]
    second = search.SCREENING_CANDIDATES[1]
    results = [
        _candidate_result(first, 42, 0.50),
        _candidate_result(first, 43, 0.60),
        _candidate_result(first, 44, 0.55),
        _candidate_result(second, 42, 0.40),
        _candidate_result(second, 43, 0.40),
        _candidate_result(second, 44, 0.40),
    ]

    summaries = search.summarize_candidate_runs(results)
    first_summary = next(
        summary for summary in summaries if summary["candidate_id"] == "C01"
    )

    assert first_summary["n_seeds"] == 3
    assert first_summary["combined_score_mean"] == pytest.approx(0.55)
    assert first_summary["combined_score_sd"] == pytest.approx(0.05)
    assert first_summary["fa_birth_date_usage_proportion_sd"] == 0.0
    assert first_summary["fa_window_usage_stability"][0][
        "mean_usage_proportion"
    ] == 0.5
    assert first_summary["fa_root_stability"][0]["proportion"] == 1.0
    assert {summary["rank"] for summary in summaries} == {1, 2}


def test_representative_seed_is_closest_to_median_score():
    candidate = search.SCREENING_CANDIDATES[0]
    results = [
        _candidate_result(candidate, 42, 0.70),
        _candidate_result(candidate, 43, 0.50),
        _candidate_result(candidate, 44, 0.60),
    ]

    assert search.representative_seed(results) == 44


def test_completed_run_is_resumed_but_partial_run_is_not_overwritten(tmp_path):
    configuration = {"candidate": "C01", "seed": 42}
    fingerprint = search.configuration_fingerprint(configuration)
    complete_dir = tmp_path / "complete"
    complete_dir.mkdir()
    (complete_dir / "status.json").write_text(json.dumps({
        "status": "complete",
        "configuration_fingerprint": fingerprint,
    }))
    (complete_dir / "result.json").write_text(json.dumps({"value": 7}))

    assert search._completed_result(complete_dir, fingerprint) == {"value": 7}

    partial_dir = tmp_path / "partial"
    partial_dir.mkdir()
    (partial_dir / "status.json").write_text(json.dumps({
        "status": "running",
        "configuration_fingerprint": fingerprint,
    }))
    with pytest.raises(RuntimeError, match="silently overwritten"):
        search._completed_result(partial_dir, fingerprint)
