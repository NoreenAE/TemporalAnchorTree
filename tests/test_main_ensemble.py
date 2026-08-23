import numpy as np
import pandas as pd
import pytest

import main
from settree.temporal_tree import TemporalTree
from settree.temporal_tree_ensemble import TemporalTreeEnsemble


def _small_data(num_patients=10):
    patient_ids = [f"P{index:03d}" for index in range(num_patients)]
    X = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "time": pd.Timestamp("2020-01-01"),
            "feature_name": "signal",
            "value": np.arange(num_patients, dtype=float),
        }
    )
    y = pd.Series([0, 1] * (num_patients // 2))
    return X, y


def _configure_fast_tree(monkeypatch):
    config = {
        "max_depth": 1,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "anchors_list": ["Birth Date Anchor"],
        "anchor_offset_list": [None],
        "aggregation_funcs": {"mean": np.mean},
        "is_classification": False,
        "criterion": "mse",
        "learn_missing_direction": False,
        "random_state": 17,
        "candidate_sampling_rate": 0.5,
        "split_sampling_seed": 42,
    }
    monkeypatch.setattr(main, "MODEL_CONFIG", config)
    monkeypatch.setattr(main, "VERBOSE", 0)
    monkeypatch.setattr(main, "ANCHOR_OFFSET_MAP", None)
    return config


def test_single_tree_main_switch_preserves_single_tree_configuration(monkeypatch):
    config = _configure_fast_tree(monkeypatch)
    monkeypatch.setattr(main, "USE_ENSEMBLE", False)
    monkeypatch.setattr(main, "USE_DYNAMIC_ANCHORS", False)

    model = main.build_model()

    assert type(model) is TemporalTree
    assert model.get_params(deep=False) == {
        **config,
        "verbose": 0,
        "use_dynamic_anchors": False,
        "anchor_offset_map": None,
    }
    assert main.MODEL_CONFIG == config


def test_ensemble_main_switch_builds_generated_seed_configuration(monkeypatch):
    config = _configure_fast_tree(monkeypatch)
    monkeypatch.setattr(main, "USE_ENSEMBLE", True)
    monkeypatch.setattr(main, "USE_DYNAMIC_ANCHORS", True)
    monkeypatch.setattr(main, "N_ESTIMATORS", 5)
    monkeypatch.setattr(main, "BASE_ENSEMBLE_SPLIT_SAMPLING_SEED", 42)
    monkeypatch.setattr(main, "ENSEMBLE_SPLIT_SAMPLING_SEEDS", None)

    model = main.build_model()

    assert isinstance(model, TemporalTreeEnsemble)
    assert model.estimator.get_params(deep=False) == {
        **config,
        "verbose": 0,
        "use_dynamic_anchors": True,
        "anchor_offset_map": None,
    }
    assert model._resolve_split_sampling_seeds() == (42, 43, 44, 45, 46)


def test_ensemble_main_switch_uses_explicit_seed_configuration(monkeypatch):
    _configure_fast_tree(monkeypatch)
    monkeypatch.setattr(main, "USE_ENSEMBLE", True)
    monkeypatch.setattr(main, "N_ESTIMATORS", 3)
    monkeypatch.setattr(main, "BASE_ENSEMBLE_SPLIT_SAMPLING_SEED", 999)
    monkeypatch.setattr(main, "ENSEMBLE_SPLIT_SAMPLING_SEEDS", [8, 3, 12])

    model = main.build_model()

    assert model._resolve_split_sampling_seeds() == (8, 3, 12)


def test_switching_model_mode_does_not_change_train_test_split(monkeypatch):
    patient_ids = [f"P{index:03d}" for index in range(20)]
    data = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "time": pd.Timestamp("2020-01-01"),
            "feature_name": "signal",
            "value": np.arange(20, dtype=float),
        }
    )
    labels = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "label": [0, 1] * 10,
        }
    )

    monkeypatch.setattr(main, "USE_ENSEMBLE", False)
    first = main.split_data(data, labels)
    monkeypatch.setattr(main, "USE_ENSEMBLE", True)
    second = main.split_data(data, labels)

    for first_value, second_value in zip(first, second):
        if isinstance(first_value, pd.DataFrame):
            pd.testing.assert_frame_equal(first_value, second_value)
        else:
            pd.testing.assert_series_equal(first_value, second_value)


def test_ensemble_evaluation_and_export_do_not_assume_one_model_tree(
    monkeypatch,
    tmp_path,
    capsys,
):
    _configure_fast_tree(monkeypatch)
    monkeypatch.setattr(main, "USE_ENSEMBLE", True)
    monkeypatch.setattr(main, "N_ESTIMATORS", 2)
    monkeypatch.setattr(main, "ENSEMBLE_SPLIT_SAMPLING_SEEDS", [42, 43])
    X, y = _small_data()
    model = main.build_model().fit(X, y)

    monkeypatch.setattr(
        main,
        "evaluate_TemporalTree",
        lambda y_real, y_predicted, y_probabilities: {"accuracy": 1.0},
    )
    monkeypatch.setattr(
        main,
        "plot_tree_structure",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("single-tree plotting must not run")
        ),
    )

    results = main.evaluate_model(model, X, y, tmp_path, plot_tree=True)
    export_path = main.export_model_tree_structure(model, tmp_path)

    assert results == {"accuracy": 1.0}
    assert export_path is None
    output = capsys.readouterr().out
    assert "Tree plotting is skipped" in output
    assert "Single-tree JSON export is skipped" in output


def test_main_dill_save_load_supports_ensemble(monkeypatch, tmp_path):
    _configure_fast_tree(monkeypatch)
    monkeypatch.setattr(main, "USE_ENSEMBLE", True)
    monkeypatch.setattr(main, "N_ESTIMATORS", 2)
    monkeypatch.setattr(main, "ENSEMBLE_SPLIT_SAMPLING_SEEDS", [42, 43])
    X, y = _small_data()
    model = main.build_model().fit(X, y)
    expected = model.predict(X)

    model_path = main.save_model(model, tmp_path)
    restored = main.load_model(model_path)

    assert isinstance(restored, TemporalTreeEnsemble)
    assert restored.split_sampling_seeds_ == (42, 43)
    assert len(restored.estimators_) == 2
    assert np.array_equal(restored.predict(X), expected)


def test_build_config_records_active_ensemble_configuration(monkeypatch):
    _configure_fast_tree(monkeypatch)
    monkeypatch.setattr(main, "USE_ENSEMBLE", True)
    monkeypatch.setattr(main, "N_ESTIMATORS", 3)
    monkeypatch.setattr(main, "BASE_ENSEMBLE_SPLIT_SAMPLING_SEED", 42)
    monkeypatch.setattr(main, "ENSEMBLE_SPLIT_SAMPLING_SEEDS", [7, 8, 9])

    config = main.build_config("test", "data.pkl")

    assert config["use_ensemble"] is True
    assert config["ensemble"] == {
        "n_estimators": 3,
        "base_split_sampling_seed": 42,
        "split_sampling_seeds": [7, 8, 9],
    }


def test_build_config_and_model_record_anchor_offset_map(monkeypatch):
    _configure_fast_tree(monkeypatch)
    anchor_offset_map = {
        "Birth Date Anchor": [1, 2, 3],
        "NOW Anchor": [-1, -2, -3],
    }
    monkeypatch.setattr(main, "ANCHOR_OFFSET_MAP", anchor_offset_map)
    monkeypatch.setattr(main, "USE_ENSEMBLE", False)

    model = main.build_model()
    config = main.build_config("test", "data.pkl")

    assert model.anchor_offset_map == anchor_offset_map
    assert config["anchor_offset_map"] == anchor_offset_map


def test_build_config_records_temporal_explainability_settings(monkeypatch):
    _configure_fast_tree(monkeypatch)
    bin_edges = {"Birth Date Anchor": [0, 1, 2]}
    monkeypatch.setattr(main, "RUN_TEMPORAL_EXPLAINABILITY", True)
    monkeypatch.setattr(main, "TEMPORAL_EXPLAINABILITY_BIN_EDGES", bin_edges)

    config = main.build_config("test", "data.pkl")

    assert config["run_temporal_explainability"] is True
    assert config["temporal_explainability_bin_edges"] == bin_edges


def test_disabled_temporal_explainability_does_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "RUN_TEMPORAL_EXPLAINABILITY", False)
    monkeypatch.setattr(
        main,
        "run_temporal_explainability",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("disabled explainability must not run")
        ),
    )

    result = main.run_training_temporal_explainability(
        object(),
        pd.DataFrame(),
        tmp_path,
    )

    assert result is None
    assert list(tmp_path.iterdir()) == []


def test_enabled_temporal_explainability_uses_training_data(
    monkeypatch,
    tmp_path,
):
    X, _ = _small_data()
    model = TemporalTree(is_classification=False)
    bin_edges = {"Birth Date Anchor": [0, 1]}
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {
            "path_table_path": tmp_path / "temporal_path_usage.csv",
            "histogram_path": tmp_path / "temporal_usage_histogram.png",
        }

    monkeypatch.setattr(main, "RUN_TEMPORAL_EXPLAINABILITY", True)
    monkeypatch.setattr(main, "TEMPORAL_EXPLAINABILITY_BIN_EDGES", bin_edges)
    monkeypatch.setattr(main, "run_temporal_explainability", fake_run)

    result = main.run_training_temporal_explainability(model, X, tmp_path)

    assert result is not None
    assert captured["model"] is model
    assert captured["X"] is X
    assert captured["output_dir"] == tmp_path
    assert captured["bin_edges_by_anchor"] == bin_edges


def test_enabled_temporal_explainability_rejects_ensemble(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(main, "RUN_TEMPORAL_EXPLAINABILITY", True)

    with pytest.raises(ValueError, match="supports single-tree models only"):
        main.run_training_temporal_explainability(
            TemporalTreeEnsemble(),
            pd.DataFrame(),
            tmp_path,
        )
