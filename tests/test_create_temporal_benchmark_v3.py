import json
from pathlib import Path

import pandas as pd
import pytest

from create_temporal_benchmark_v3 import (
    ARTIFACT_NAMES,
    BYTE_IDENTICAL_ARTIFACTS,
    artifact_paths,
    create_v3,
    derive_v3_data,
    sha256_file,
    validate_v3_data,
)


def _row(patient_id, time, feature_name, value):
    return {
        "patient_id": patient_id,
        "time": pd.Timestamp(time),
        "feature_name": feature_name,
        "value": value,
    }


@pytest.fixture
def v2_data():
    rows = []
    patients = {
        "P1": (pd.Timestamp("2020-01-15"), pd.Timestamp("2022-01-15")),
        "P2": (pd.Timestamp("2020-02-20"), pd.Timestamp("2021-10-20")),
    }
    target_months = (0, 1, 2, 4, 12)
    for patient_id, (birth, now) in patients.items():
        rows.extend([
            _row(patient_id, birth, "Birth Date Anchor", birth),
            _row(patient_id, now, "NOW Anchor", now),
            _row(patient_id, birth, "GestationalAgeWeeks", 39.0),
        ])
        for month in target_months:
            time = birth + pd.DateOffset(months=month)
            if month:
                time += pd.Timedelta(days=1 if patient_id == "P1" else -2)
            rows.append(
                _row(patient_id, time, "LengthCm", 50.0 + month)
            )
        if patient_id == "P1":
            rows.append(_row(patient_id, now, "LengthCm", 74.0))

        redundant_time = birth + pd.DateOffset(months=3)
        rows.extend([
            _row(patient_id, redundant_time, "LengthCm", 53.0),
            _row(patient_id, redundant_time, "WeightKg", 6.5),
            _row(patient_id, redundant_time, "FormulaYN", 1.0),
        ])
    return pd.DataFrame(rows)


def _write_source_artifacts(source_dir: Path, data: pd.DataFrame) -> None:
    source_dir.mkdir()
    paths = artifact_paths(source_dir)
    data.to_pickle(paths["data"])
    patients = pd.DataFrame({
        "patient_id": ["P1", "P2"],
        "birth_date": pd.to_datetime(["2020-01-15", "2020-02-20"]),
        "now": pd.to_datetime(["2022-01-15", "2021-10-20"]),
    })
    fa_labels = pd.DataFrame({"patient_id": ["P1", "P2"], "label": [1, 0]})
    gi_labels = pd.DataFrame({"patient_id": ["P1", "P2"], "label": [0, 1]})
    fa_labels.to_pickle(paths["fa_labels"])
    gi_labels.to_pickle(paths["gi_labels"])
    patients.assign(feeding_group=["2.0", "3"]).to_pickle(
        paths["fa_ground_truth"]
    )
    patients[["patient_id"]].assign(
        V_i=[0, 1],
        D_i=[0, 0],
        G_i=[0, 1],
        gi_probability=[0.1, 0.4],
    ).to_pickle(paths["gi_ground_truth"])
    paths["fa_metadata"].write_text(
        json.dumps({
            "benchmark_version": "V2",
            "output_directory": str(source_dir),
            "fa": {"calibrated_intercept": -2.4},
            "validation": {"checks": {"valid": True}},
        }),
        encoding="utf-8",
    )
    paths["gi_metadata"].write_text(
        json.dumps({
            "benchmark_version": "V2",
            "calibrated_intercept": -2.9,
            "source_artifacts": {"longitudinal_x": "v2"},
            "validation": {"checks": {"valid": True}},
        }),
        encoding="utf-8",
    )


def test_derivation_changes_only_length_rows(v2_data):
    v3_data, selection = derive_v3_data(v2_data)

    validation = validate_v3_data(v2_data, v3_data, selection)
    assert all(validation["checks"].values())
    pd.testing.assert_frame_equal(
        v2_data.loc[v2_data["feature_name"].ne("LengthCm")]
        .sort_values(["patient_id", "time", "feature_name"])
        .reset_index(drop=True),
        v3_data.loc[v3_data["feature_name"].ne("LengthCm")]
        .sort_values(["patient_id", "time", "feature_name"])
        .reset_index(drop=True),
        check_exact=True,
    )
    assert len(v3_data) < len(v2_data)


def test_shared_timestamp_rows_survive_length_deletion(v2_data):
    v3_data, _ = derive_v3_data(v2_data)
    redundant_times = {
        "P1": pd.Timestamp("2020-04-15"),
        "P2": pd.Timestamp("2020-05-20"),
    }
    for patient_id, time in redundant_times.items():
        shared = v3_data.loc[
            v3_data["patient_id"].eq(patient_id)
            & v3_data["time"].eq(time)
        ]
        assert set(shared["feature_name"]) == {"FormulaYN", "WeightKg"}


def test_retained_length_rows_are_unchanged_existing_rows(v2_data):
    v3_data, selection = derive_v3_data(v2_data)
    retained = v3_data.loc[v3_data["feature_name"].eq("LengthCm")]
    source_rows = v2_data.loc[selection["source_index"].sort_values()]

    pd.testing.assert_frame_equal(
        source_rows.reset_index(drop=True),
        retained.reset_index(drop=True),
        check_exact=True,
    )
    assert retained.groupby("patient_id").size().to_dict() == {
        "P1": 6,
        "P2": 5,
    }
    assert selection["distance_days"].max() <= 7


def test_complete_v3_is_atomic_and_preserves_source(tmp_path, v2_data):
    source_dir = tmp_path / "v2"
    output_dir = tmp_path / "v3"
    _write_source_artifacts(source_dir, v2_data)
    source_paths = artifact_paths(source_dir)
    hashes_before = {
        name: sha256_file(path) for name, path in source_paths.items()
    }
    mtimes_before = {
        name: path.stat().st_mtime_ns for name, path in source_paths.items()
    }

    output_paths = create_v3(source_dir, output_dir)

    assert set(output_paths) == set(ARTIFACT_NAMES)
    assert all(path.is_file() for path in output_paths.values())
    assert hashes_before == {
        name: sha256_file(path) for name, path in source_paths.items()
    }
    assert mtimes_before == {
        name: path.stat().st_mtime_ns for name, path in source_paths.items()
    }
    for name in BYTE_IDENTICAL_ARTIFACTS:
        assert sha256_file(source_paths[name]) == sha256_file(
            output_paths[name]
        )

    fa_metadata = json.loads(output_paths["fa_metadata"].read_text())
    gi_metadata = json.loads(output_paths["gi_metadata"].read_text())
    assert fa_metadata["benchmark_version"] == "V3"
    assert gi_metadata["benchmark_version"] == "V3"
    assert fa_metadata["parent_benchmark"] == "V2"
    assert fa_metadata["fa"]["calibrated_intercept"] == -2.4
    assert gi_metadata["calibrated_intercept"] == -2.9
    assert all(
        fa_metadata["validation"]["artifact_identity"][outcome][
            "checks"
        ].values()
        for outcome in ("fa", "gi")
    )


def test_existing_output_directory_is_never_overwritten(tmp_path, v2_data):
    source_dir = tmp_path / "v2"
    output_dir = tmp_path / "v3"
    _write_source_artifacts(source_dir, v2_data)
    output_dir.mkdir()
    sentinel = output_dir / "keep.txt"
    sentinel.write_text("untouched", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        create_v3(source_dir, output_dir)

    assert sentinel.read_text(encoding="utf-8") == "untouched"
