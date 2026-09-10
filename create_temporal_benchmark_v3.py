"""Derive Benchmark V3 by pruning only LengthCm rows from validated V2."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from generate_temporal_benchmark import _observation_density


SOURCE_DIR = Path("data/temporal_benchmark_v2")
OUTPUT_DIR = Path("data/temporal_benchmark_v3")
COHORT_SIZE = 5000
LENGTH_FEATURE = "LengthCm"
TARGET_AGE_MONTHS = (0, 1, 2, 4, 12, 24)
TARGET_TOLERANCE_DAYS = 7

ARTIFACT_NAMES = {
    "data": "temporal_benchmark_5000.pkl",
    "fa_labels": "temporal_benchmark_fa_labels_5000.pkl",
    "fa_metadata": "temporal_benchmark_metadata_5000.json",
    "fa_ground_truth": "temporal_benchmark_ground_truth_5000.pkl",
    "gi_labels": "temporal_benchmark_gi_labels_5000.pkl",
    "gi_metadata": "temporal_benchmark_gi_metadata_5000.json",
    "gi_ground_truth": "temporal_benchmark_gi_ground_truth_5000.pkl",
}
BYTE_IDENTICAL_ARTIFACTS = (
    "fa_labels",
    "fa_ground_truth",
    "gi_labels",
    "gi_ground_truth",
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_paths(directory: Path) -> Dict[str, Path]:
    """Return all expected benchmark paths below one version directory."""
    return {
        name: directory / filename
        for name, filename in ARTIFACT_NAMES.items()
    }


def dataframe_sha256(data: pd.DataFrame) -> str:
    """Hash dataframe contents, column order, dtypes, and row order."""
    digest = hashlib.sha256()
    digest.update("|".join(map(str, data.columns)).encode("utf-8"))
    digest.update("|".join(map(str, data.dtypes)).encode("utf-8"))
    digest.update(
        pd.util.hash_pandas_object(data, index=True).to_numpy().tobytes()
    )
    return digest.hexdigest()


def label_identity_report(
    source_paths: Mapping[str, Path],
    output_paths: Mapping[str, Path],
) -> dict:
    """Verify copied artifacts and their outcome counts are unchanged."""
    report = {}
    for outcome in ("fa", "gi"):
        label_key = f"{outcome}_labels"
        ground_truth_key = f"{outcome}_ground_truth"
        source_labels = pd.read_pickle(source_paths[label_key])
        output_labels = pd.read_pickle(output_paths[label_key])
        pd.testing.assert_frame_equal(
            source_labels,
            output_labels,
            check_dtype=True,
            check_exact=True,
        )
        source_positive_count = int(source_labels["label"].sum())
        output_positive_count = int(output_labels["label"].sum())
        source_prevalence = float(source_labels["label"].mean())
        output_prevalence = float(output_labels["label"].mean())
        label_hash = sha256_file(source_paths[label_key])
        ground_truth_hash = sha256_file(source_paths[ground_truth_key])
        checks = {
            "label_sha256_identical": (
                label_hash == sha256_file(output_paths[label_key])
            ),
            "ground_truth_sha256_identical": (
                ground_truth_hash
                == sha256_file(output_paths[ground_truth_key])
            ),
            "positive_count_identical": (
                source_positive_count == output_positive_count
            ),
            "prevalence_identical": source_prevalence == output_prevalence,
        }
        if not all(checks.values()):
            raise ValueError(f"{outcome.upper()} identity validation failed.")
        report[outcome] = {
            "checks": checks,
            "label_sha256": label_hash,
            "ground_truth_sha256": ground_truth_hash,
            "positive_count": source_positive_count,
            "prevalence": source_prevalence,
        }
    return report


def extract_anchors(data: pd.DataFrame) -> pd.DataFrame:
    """Extract exactly one valid Birth Date and NOW timestamp per patient."""
    anchor_names = ("Birth Date Anchor", "NOW Anchor")
    anchors = data.loc[
        data["feature_name"].isin(anchor_names),
        ["patient_id", "feature_name", "value"],
    ].copy()
    counts = anchors.groupby(["patient_id", "feature_name"]).size().unstack(
        fill_value=0
    ).reindex(columns=anchor_names, fill_value=0)
    if not counts.eq(1).all().all():
        raise ValueError("Every patient must have exactly one of each anchor.")
    wide = anchors.pivot(
        index="patient_id",
        columns="feature_name",
        values="value",
    ).rename(columns={
        "Birth Date Anchor": "birth_date",
        "NOW Anchor": "now",
    })
    wide["birth_date"] = pd.to_datetime(wide["birth_date"], errors="coerce")
    wide["now"] = pd.to_datetime(wide["now"], errors="coerce")
    if wide[["birth_date", "now"]].isna().any().any():
        raise ValueError("Every anchor value must be a valid timestamp.")
    return wide


def select_length_rows(
    data: pd.DataFrame,
    tolerance_days: int = TARGET_TOLERANCE_DAYS,
) -> Tuple[pd.Index, pd.DataFrame]:
    """Select existing LengthCm rows nearest the approved target ages."""
    if tolerance_days < 0:
        raise ValueError("tolerance_days must be non-negative.")
    anchors = extract_anchors(data)
    length_rows = data.loc[data["feature_name"].eq(LENGTH_FEATURE)].copy()
    length_rows["source_index"] = length_rows.index
    selected_indices = []
    selection_rows = []
    used_indices = set()

    for target_month in TARGET_AGE_MONTHS:
        targets = anchors[["birth_date", "now"]].copy()
        targets["target_time"] = (
            targets["birth_date"] + pd.DateOffset(months=target_month)
        )
        candidates = length_rows.merge(
            targets[["target_time", "now"]],
            left_on="patient_id",
            right_index=True,
            how="inner",
            validate="many_to_one",
        )
        candidates = candidates.loc[
            candidates["time"].le(candidates["now"])
            & ~candidates["source_index"].isin(used_indices)
        ].copy()
        candidates["distance_days"] = (
            candidates["time"] - candidates["target_time"]
        ).abs().dt.days
        candidates = candidates.loc[
            candidates["distance_days"].le(tolerance_days)
        ].sort_values(
            ["patient_id", "distance_days", "time", "source_index"],
            kind="mergesort",
        )
        chosen = candidates.groupby("patient_id", sort=False).head(1)
        used_indices.update(chosen["source_index"].tolist())
        selected_indices.extend(chosen["source_index"].tolist())
        selection_rows.extend({
            "patient_id": row.patient_id,
            "source_index": int(row.source_index),
            "target_month": int(target_month),
            "target_time": row.target_time,
            "retained_time": row.time,
            "distance_days": int(row.distance_days),
        } for row in chosen.itertuples(index=False))

    selected = pd.Index(sorted(selected_indices), dtype="int64")
    if selected.duplicated().any():
        raise RuntimeError("A LengthCm row was selected more than once.")
    return selected, pd.DataFrame.from_records(selection_rows)


def derive_v3_data(
    v2_data: pd.DataFrame,
    tolerance_days: int = TARGET_TOLERANCE_DAYS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return V2 with only redundant LengthCm rows removed."""
    selected, selection = select_length_rows(v2_data, tolerance_days)
    is_length = v2_data["feature_name"].eq(LENGTH_FEATURE)
    keep = ~is_length | v2_data.index.isin(selected)
    return v2_data.loc[keep].reset_index(drop=True), selection


def _deterministic_non_length(data: pd.DataFrame) -> pd.DataFrame:
    rows = data.loc[~data["feature_name"].eq(LENGTH_FEATURE)].copy()
    return rows.sort_values(
        ["patient_id", "time", "feature_name"],
        kind="mergesort",
    ).reset_index(drop=True)


def length_pruning_report(
    v2_data: pd.DataFrame,
    v3_data: pd.DataFrame,
    selection: pd.DataFrame,
) -> dict:
    """Summarize the exact LengthCm-only V2-to-V3 transformation."""
    v2_length = v2_data.loc[v2_data["feature_name"].eq(LENGTH_FEATURE)]
    v3_length = v3_data.loc[v3_data["feature_name"].eq(LENGTH_FEATURE)]
    patient_ids = pd.Index(v2_data["patient_id"].unique(), name="patient_id")
    v2_counts = v2_length.groupby("patient_id").size().reindex(
        patient_ids, fill_value=0
    )
    v3_counts = v3_length.groupby("patient_id").size().reindex(
        patient_ids, fill_value=0
    )
    removed = len(v2_length) - len(v3_length)
    count_distribution = v3_counts.value_counts().sort_index()
    selection_by_target = selection.groupby("target_month").agg(
        retained_rows=("patient_id", "size"),
        mean_absolute_distance_days=("distance_days", "mean"),
        median_absolute_distance_days=("distance_days", "median"),
        maximum_absolute_distance_days=("distance_days", "max"),
    )
    gaps = v3_length.sort_values(["patient_id", "time"]).groupby(
        "patient_id"
    )["time"].diff().dt.days.dropna()
    anchors = extract_anchors(v2_data)
    ages = v3_length.copy()
    ages["birth_date"] = ages["patient_id"].map(anchors["birth_date"])
    ages["age_days"] = (ages["time"] - ages["birth_date"]).dt.days
    ages["age_months"] = ages["age_days"] / 30.4375
    return {
        "tolerance_days": int(TARGET_TOLERANCE_DAYS),
        "target_age_months": list(TARGET_AGE_MONTHS),
        "v2_length_rows": int(len(v2_length)),
        "v3_length_rows": int(len(v3_length)),
        "removed_length_rows": int(removed),
        "removed_percentage": float(100.0 * removed / len(v2_length)),
        "measurements_per_patient": {
            "v2": {
                "mean": float(v2_counts.mean()),
                "median": float(v2_counts.median()),
                "min": int(v2_counts.min()),
                "max": int(v2_counts.max()),
            },
            "v3": {
                "mean": float(v3_counts.mean()),
                "median": float(v3_counts.median()),
                "min": int(v3_counts.min()),
                "max": int(v3_counts.max()),
            },
        },
        "v3_patient_count_distribution": {
            str(count): {
                "patients": int(number),
                "percentage": float(100.0 * number / len(v3_counts)),
            }
            for count, number in count_distribution.items()
        },
        "v3_frequency_categories": {
            "fewer_than_5": int(v3_counts.lt(5).sum()),
            "exactly_5": int(v3_counts.eq(5).sum()),
            "exactly_6": int(v3_counts.eq(6).sum()),
            "more_than_6": int(v3_counts.gt(6).sum()),
        },
        "selection_by_target": {
            str(month): {
                key: (
                    int(value) if key in {
                        "retained_rows",
                        "maximum_absolute_distance_days",
                    } else float(value)
                )
                for key, value in row.items()
            }
            for month, row in selection_by_target.to_dict("index").items()
        },
        "retained_age_days": {
            "min": int(ages["age_days"].min()),
            "median": float(ages["age_days"].median()),
            "max": int(ages["age_days"].max()),
        },
        "retained_age_months": {
            "min": float(ages["age_months"].min()),
            "median": float(ages["age_months"].median()),
            "max": float(ages["age_months"].max()),
        },
        "retained_gap_days": {
            "min": int(gaps.min()),
            "q25": float(gaps.quantile(0.25)),
            "median": float(gaps.median()),
            "q75": float(gaps.quantile(0.75)),
            "max": int(gaps.max()),
        },
    }


def validate_v3_data(
    v2_data: pd.DataFrame,
    v3_data: pd.DataFrame,
    selection: pd.DataFrame,
) -> dict:
    """Fail loudly unless V3 equals V2 minus selected LengthCm rows."""
    v2_non_length = _deterministic_non_length(v2_data)
    v3_non_length = _deterministic_non_length(v3_data)
    pd.testing.assert_frame_equal(
        v2_non_length,
        v3_non_length,
        check_dtype=True,
        check_exact=True,
    )
    selected_indices = pd.Index(selection["source_index"].sort_values())
    expected_length = v2_data.loc[selected_indices].reset_index(drop=True)
    actual_length = v3_data.loc[
        v3_data["feature_name"].eq(LENGTH_FEATURE)
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        expected_length,
        actual_length,
        check_dtype=True,
        check_exact=True,
    )
    anchors = extract_anchors(v3_data)
    row_now = actual_length["patient_id"].map(anchors["now"])
    selection_is_chronological = bool(
        selection.sort_values(["patient_id", "target_month"])
        .groupby("patient_id")["retained_time"]
        .apply(lambda times: times.is_monotonic_increasing)
        .all()
    )
    checks = {
        "non_length_rows_exactly_equal": True,
        "non_length_row_counts_equal": (
            len(v2_non_length) == len(v3_non_length)
        ),
        "non_length_dtypes_equal": v2_non_length.dtypes.equals(
            v3_non_length.dtypes
        ),
        "retained_length_rows_are_exact_v2_rows": True,
        "no_new_length_rows": len(actual_length) <= int(
            v2_data["feature_name"].eq(LENGTH_FEATURE).sum()
        ),
        "retained_length_timestamps_unchanged": True,
        "retained_length_values_unchanged": True,
        "no_length_observation_after_now": bool(
            actual_length["time"].le(row_now).all()
        ),
        "retained_length_schedule_is_chronological": (
            selection_is_chronological
        ),
        "same_patient_population": set(v2_data["patient_id"])
        == set(v3_data["patient_id"]),
        "only_rows_deleted": len(v3_data) <= len(v2_data),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("V3 validation failed: " + ", ".join(failed))
    return {
        "checks": checks,
        "non_length_dataframe_sha256": dataframe_sha256(v3_non_length),
    }


def _build_provenance(
    parent_metadata_hash: str,
    report: Mapping,
    validation: Mapping,
) -> dict:
    return {
        "benchmark_version": "V3",
        "parent_benchmark": "V2",
        "parent_metadata_sha256": parent_metadata_hash,
        "derivation": "V2 minus selected redundant LengthCm rows only",
        "only_model_visible_change": "Reduced LengthCm observation frequency",
        "target_age_months": list(TARGET_AGE_MONTHS),
        "maximum_tolerance_days": TARGET_TOLERANCE_DAYS,
        "closest_existing_row_only": True,
        "new_length_rows_generated": False,
        "length_values_modified": False,
        "length_timestamps_modified": False,
        "non_length_rows_modified": False,
        "fa_labels_regenerated": False,
        "gi_labels_regenerated": False,
        "fa_intercept_recalibrated": False,
        "gi_intercept_recalibrated": False,
        "scientific_ground_truth_changed": False,
        "length_pruning_report": dict(report),
        "v3_identity_validation": dict(validation),
    }


def _updated_metadata(
    parent_metadata: Mapping,
    parent_metadata_hash: str,
    output_dir: Path,
    report: Mapping,
    validation: Mapping,
    observation_density: Mapping,
    artifact_identity: Optional[Mapping] = None,
) -> dict:
    metadata = deepcopy(dict(parent_metadata))
    parent_validation = metadata.pop("validation", None)
    metadata["benchmark_version"] = "V3"
    metadata["parent_benchmark"] = "V2"
    if "output_directory" in metadata:
        metadata["output_directory"] = str(output_dir)
    metadata["parent_validation"] = parent_validation
    metadata["v3_provenance"] = _build_provenance(
        parent_metadata_hash,
        report,
        validation,
    )
    metadata["validation"] = {
        **dict(validation),
        "length_pruning": dict(report),
        "observation_density": dict(observation_density),
    }
    if artifact_identity is not None:
        metadata["validation"]["artifact_identity"] = dict(
            artifact_identity
        )
    return metadata


def create_v3(
    source_dir: Path = SOURCE_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> Dict[str, Path]:
    """Create a self-contained V3 atomically and preserve V2 exactly."""
    if source_dir.resolve() == output_dir.resolve():
        raise ValueError("V2 source and V3 output directories must differ.")
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite V3 directory: {output_dir}"
        )
    source_paths = artifact_paths(source_dir)
    missing = [path for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing V2 artifacts: {missing}")
    source_hashes_before = {
        name: sha256_file(path) for name, path in source_paths.items()
    }
    source_mtimes_before = {
        name: path.stat().st_mtime_ns for name, path in source_paths.items()
    }

    v2_data = pd.read_pickle(source_paths["data"])
    v3_data, selection = derive_v3_data(v2_data)
    validation = validate_v3_data(v2_data, v3_data, selection)
    report = length_pruning_report(v2_data, v3_data, selection)
    fa_ground_truth = pd.read_pickle(source_paths["fa_ground_truth"])
    observation_density = _observation_density(v3_data, fa_ground_truth)
    fa_metadata = json.loads(source_paths["fa_metadata"].read_text())
    gi_metadata = json.loads(source_paths["gi_metadata"].read_text())
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=".temporal_benchmark_v3_",
        dir=output_dir.parent,
    ) as temporary_directory:
        stage = Path(temporary_directory) / output_dir.name
        stage.mkdir()
        stage_paths = artifact_paths(stage)
        v3_data.to_pickle(stage_paths["data"])
        for name in BYTE_IDENTICAL_ARTIFACTS:
            shutil.copyfile(source_paths[name], stage_paths[name])

        copied_hashes = {
            name: sha256_file(stage_paths[name])
            for name in BYTE_IDENTICAL_ARTIFACTS
        }
        if any(
            copied_hashes[name] != source_hashes_before[name]
            for name in BYTE_IDENTICAL_ARTIFACTS
        ):
            raise RuntimeError(
                "A copied label or ground-truth artifact changed."
            )
        artifact_identity = label_identity_report(source_paths, stage_paths)

        fa_metadata_v3 = _updated_metadata(
            fa_metadata,
            source_hashes_before["fa_metadata"],
            output_dir,
            report,
            validation,
            observation_density,
            artifact_identity,
        )
        gi_metadata_v3 = _updated_metadata(
            gi_metadata,
            source_hashes_before["gi_metadata"],
            output_dir,
            report,
            validation,
            observation_density,
            artifact_identity,
        )
        gi_metadata_v3["parent_source_artifacts"] = gi_metadata.get(
            "source_artifacts"
        )
        gi_metadata_v3["source_x_status"] = (
            "V2-derived; only neutral LengthCm rows were deleted"
        )
        gi_metadata_v3["source_artifacts"] = {
            "longitudinal_x": {
                "path": str(output_dir / ARTIFACT_NAMES["data"]),
                "sha256": sha256_file(stage_paths["data"]),
            },
            "fa_labels": {
                "path": str(output_dir / ARTIFACT_NAMES["fa_labels"]),
                "sha256": copied_hashes["fa_labels"],
            },
            "fa_ground_truth": {
                "path": str(output_dir / ARTIFACT_NAMES["fa_ground_truth"]),
                "sha256": copied_hashes["fa_ground_truth"],
            },
        }
        stage_paths["fa_metadata"].write_text(
            json.dumps(fa_metadata_v3, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        stage_paths["gi_metadata"].write_text(
            json.dumps(gi_metadata_v3, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        reloaded = pd.read_pickle(stage_paths["data"])
        pd.testing.assert_frame_equal(v3_data, reloaded, check_exact=True)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        stage.replace(output_dir)

    source_hashes_after = {
        name: sha256_file(path) for name, path in source_paths.items()
    }
    source_mtimes_after = {
        name: path.stat().st_mtime_ns for name, path in source_paths.items()
    }
    if source_hashes_after != source_hashes_before:
        raise RuntimeError("A V2 source artifact changed during V3 creation.")
    if source_mtimes_after != source_mtimes_before:
        raise RuntimeError("A V2 source modification timestamp changed.")

    output_paths = artifact_paths(output_dir)
    print("TEMPORAL BENCHMARK V3 — LengthCm SUBSELECTION")
    print("=" * 58)
    print(json.dumps(report, indent=2, sort_keys=True))
    print("\nIdentity validation:")
    for name, passed in validation["checks"].items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    print("\nCopied label and private-ground-truth identity:")
    for outcome, outcome_report in artifact_identity.items():
        print(
            f"  {outcome.upper()}: positive_count="
            f"{outcome_report['positive_count']}, prevalence="
            f"{outcome_report['prevalence']:.6f}, SHA-256 checks=PASS"
        )
    print("\nSaved artifacts:")
    for path in output_paths.values():
        print(f"  {path.resolve()}")
    return output_paths


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    create_v3(args.source_dir, args.output_dir)


if __name__ == "__main__":
    main()
