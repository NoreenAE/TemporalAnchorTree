import json

import numpy as np
import pandas as pd

from experiments import experiment_1_fa_vs_gi as experiment


def _write_saved_usage(directory, birth_count, now_count, full_count):
    directory.mkdir(parents=True)
    total = birth_count + now_count + full_count
    pd.DataFrame({
        "anchor": ["Birth Date Anchor", "NOW Anchor"],
        "bin_start_offset": [0.0, -1.0],
        "bin_end_offset": [1.0, 0.0],
        "raw_usage_count": [birth_count, now_count],
        "normalized_usage_per_patient": [
            birth_count / 2,
            now_count / 2,
        ],
    }).to_csv(directory / "aggregated_temporal_usage.csv", index=False)
    (directory / "temporal_usage_summary.json").write_text(
        json.dumps({
            "evaluated_patient_count": 2,
            "temporal_split_usages": total,
            "full_timeline_usages": full_count,
        }),
        encoding="utf-8",
    )


def test_experiment_configuration_records_minimum_impurity_decrease():
    configuration = experiment.experiment_configuration({})

    assert configuration["min_impurity_decrease"] == (
        experiment.MIN_IMPURITY_DECREASE
    )
    assert configuration["model_hyperparameters"][
        "min_impurity_decrease"
    ] == experiment.MIN_IMPURITY_DECREASE


def test_experiment_comparison_is_created_from_saved_outcome_artifacts(
    tmp_path,
):
    _write_saved_usage(
        tmp_path / "fa" / "temporal_explainability",
        birth_count=6,
        now_count=1,
        full_count=3,
    )
    _write_saved_usage(
        tmp_path / "gi" / "temporal_explainability",
        birth_count=1,
        now_count=8,
        full_count=1,
    )

    output_path = experiment.create_temporal_usage_comparison(tmp_path)

    assert output_path == (
        tmp_path / "temporal_usage_comparison_fa_vs_gi.png"
    )
    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_run_experiment_creates_comparison_after_both_outcomes(
    monkeypatch,
    tmp_path,
):
    input_paths = {}
    for name in ("x", "fa", "gi"):
        path = tmp_path / f"{name}.pkl"
        path.write_bytes(name.encode("utf-8"))
        input_paths[name] = path
    output_dir = tmp_path / "results"
    events = []

    monkeypatch.setattr(experiment, "INPUT_PATHS", input_paths)
    monkeypatch.setattr(experiment, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(
        experiment,
        "load_inputs",
        lambda: (pd.DataFrame(), {"fa": pd.DataFrame(), "gi": pd.DataFrame()}),
    )
    monkeypatch.setattr(
        experiment,
        "create_shared_split",
        lambda data, labels: (
            np.asarray(["P1"]),
            np.asarray(["P2"]),
            pd.DataFrame({"patient_id": ["P1", "P2"]}),
        ),
    )

    def fake_run_outcome(**kwargs):
        outcome = kwargs["outcome"]
        events.append(outcome)
        return {
            "performance": {"roc_auc": 0.5},
            "runtime_seconds": {"training": 0.0, "total": 0.0},
            "candidate_evaluations": {"total_evaluated_candidates": 0},
            "root_split": {"anchor": None, "offset": None},
            "temporal_usage": {
                "birth_date_usage_proportion": 0.0,
                "now_usage_proportion": 0.0,
                "full_timeline_usage_proportion": 1.0,
            },
            "window_distribution": [],
        }

    def fake_create_comparison(run_output_dir):
        assert events == ["fa", "gi"]
        events.append("comparison")
        output = run_output_dir / experiment.TEMPORAL_USAGE_COMPARISON_FILENAME
        output.write_bytes(b"png")
        return output

    monkeypatch.setattr(experiment, "run_outcome", fake_run_outcome)
    monkeypatch.setattr(
        experiment,
        "create_temporal_usage_comparison",
        fake_create_comparison,
    )

    result = experiment.run_experiment()

    assert events == ["fa", "gi", "comparison"]
    assert result["artifacts"]["temporal_usage_comparison"] == str(
        output_dir / experiment.TEMPORAL_USAGE_COMPARISON_FILENAME
    )
