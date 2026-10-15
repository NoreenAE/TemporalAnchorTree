"""Generate final Experiment 1 V3 Clean FA data, without training a model.

The unchanged longitudinal/FA/GI components live in generators/. Intermediate
artifacts are staged temporarily; no historical dataset directory is required.
An explicit --source-dir may instead reuse compatible existing base artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from project_paths import portable_metadata

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data/temporal_benchmark_v3_clean_fa"
X_NAME = "temporal_benchmark_5000.pkl"
SEED = 20260909
PROTECTED = (
    "Birth Date Anchor",
    "NOW Anchor",
    "FormulaYN",
    "BreastMilkYN",
    "VomitingYN",
    "DiarrheaYN",
    "GISymptomYN",
)
JOINT = (
    "ApgarFiveMinutes",
    "BirthOrder",
    "BirthWeightKg",
    "GestationalAgeWeeks",
    "MaternalAgeYears",
    "SocioEconomicScore",
    "LengthCm",
    "WeightKg",
)
MODIFIABLE = JOINT + ("FoodExposureYN", "ClinicalVisitYN")
LIMITS = {
    "label_r": 0.10,
    "smd": 0.20,
    "auc_distance": 0.06,
    "probability_r": 0.10,
    "group_eta": 0.15,
}
BIRTH_OFFSETS = (0.7, 1, 2, 3, 4, 5, 6, 12, 18)
NOW_OFFSETS = (-1, -2, -3, -4, -5, -6, -12, -18)


def sha256(path: Path) -> str:
    """Hash exact file bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot(directory: Path) -> dict:
    """Record every file's relative path, digest, and modification time."""
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return {
        str(p.relative_to(directory)): {
            "sha256": sha256(p),
            "mtime_ns": p.stat().st_mtime_ns,
        }
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


def frame_hash(frame: pd.DataFrame) -> str:
    """Hash protected rows including index, column order, and dtypes."""
    digest = hashlib.sha256(
        str(list(zip(frame.columns, frame.dtypes))).encode()
    )
    digest.update(
        pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    )
    return digest.hexdigest()


def transform(
    data: pd.DataFrame, seed: int = SEED
) -> tuple[pd.DataFrame, dict]:
    """Permute complete value sequences; never accept labels or ground truth.

    Shared demographic/growth donors retain birth-weight/growth coherence.
    Equal row-count strata avoid interpolation, truncation, or invented values.
    """
    features = set(data.feature_name.unique())
    if features != set(PROTECTED + MODIFIABLE):
        raise ValueError(f"Unexpected feature schema: {features}")
    rng = np.random.Generator(np.random.PCG64(seed))
    result = data.copy(deep=True)
    maps = {}
    for block, names, reference in (
        ("demographic_growth", JOINT, "LengthCm"),
        ("food", ("FoodExposureYN",), "FoodExposureYN"),
    ):
        counts = (
            data.loc[data.feature_name.eq(reference)]
            .groupby("patient_id")
            .size()
        )
        donors = {}
        for count in sorted(counts.unique()):
            ids = sorted(counts.index[counts.eq(count)])
            donors.update(zip(ids, rng.permutation(ids).tolist()))
        maps[block] = donors
        for feature in names:
            rows = data.loc[data.feature_name.eq(feature)].sort_values(
                ["patient_id", "time"], kind="stable"
            )
            groups = {
                pid: group.index.to_numpy()
                for pid, group in rows.groupby("patient_id", sort=True)
            }
            recipient_indices = []
            donor_indices = []
            for recipient, donor in donors.items():
                if len(groups[recipient]) != len(groups[donor]):
                    raise ValueError("Incompatible donor sequence lengths")
                recipient_indices.extend(groups[recipient])
                donor_indices.extend(groups[donor])
            result.loc[recipient_indices, "value"] = data.loc[
                donor_indices, "value"
            ].to_numpy()
    return result, maps


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Return Pearson correlation, or NaN when it is undefined."""
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def statistics(values: pd.Series, targets: pd.DataFrame) -> dict:
    """Compute effect sizes on observed summaries, with explicit support counts."""
    values = values.reindex(targets.index).astype(float)
    valid = np.isfinite(values)
    x = values[valid].to_numpy()
    target = targets.loc[valid]
    y = target.label.to_numpy()
    a, b = x[y == 1], x[y == 0]
    result = {
        "n": len(x),
        "n_positive": len(a),
        "n_negative": len(b),
        "label_r": correlation(x, y),
        "probability_r": correlation(x, target.fa_probability.to_numpy()),
        "smd": np.nan,
        "auc": np.nan,
        "auc_distance": np.nan,
        "group_eta": np.nan,
    }
    if len(a) and len(b):
        ranks = rankdata(x)
        auc = (ranks[y == 1].sum() - len(a) * (len(a) + 1) / 2) / (
            len(a) * len(b)
        )
        result.update(auc=float(auc), auc_distance=float(abs(auc - 0.5)))
        if len(a) > 1 and len(b) > 1:
            scale = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
            result["smd"] = (
                float((a.mean() - b.mean()) / scale)
                if scale
                else (0.0 if a.mean() == b.mean() else float("inf"))
            )
    if len(x) > 1 and np.var(x) > 0:
        groups = target.feeding_group.to_numpy()
        between = sum(
            np.sum(groups == group)
            * (x[groups == group].mean() - x.mean()) ** 2
            for group in np.unique(groups)
        )
        result["group_eta"] = float(
            np.sqrt(between / np.sum((x - x.mean()) ** 2))
        )
    return result


def audit(
    data: pd.DataFrame, targets: pd.DataFrame, phase: str
) -> pd.DataFrame:
    """Audit inclusive Birth/NOW windows, full timeline, and strictly late feeding.

    Mean/max exclude absent observations; sum/count are zero for empty windows.
    Missing-summary indicators are additionally audited for every aggregation.
    Count means nonmissing values; event_count independently audits row density.
    """
    births = targets.birth_date
    nows = targets.now
    windows = [("Full Timeline", None, births, nows)]
    for offset in BIRTH_OFFSETS:
        delta = (
            pd.DateOffset(months=int(offset))
            if float(offset).is_integer()
            else pd.DateOffset(days=round(offset * 30))
        )
        windows.append(("Birth", offset, births, births + delta))
    for offset in NOW_OFFSETS:
        windows.append(
            ("NOW", offset, nows + pd.DateOffset(months=offset), nows)
        )
    records = []
    for feature in MODIFIABLE + ("FormulaYN", "BreastMilkYN"):
        rows = data.loc[
            data.feature_name.eq(feature), ["patient_id", "time", "value"]
        ].copy()
        rows["value"] = rows.value.astype(float)
        is_feeding = feature in ("FormulaYN", "BreastMilkYN")
        regions = ["all", "post_6_months"] if is_feeding else ["all"]
        for region in regions:
            selected = rows
            if region == "post_6_months":
                selected = rows.loc[
                    rows.time.gt(
                        rows.patient_id.map(births + pd.DateOffset(months=6))
                    )
                ]
            for anchor, offset, start, end in windows:
                window = selected.loc[
                    selected.time.ge(selected.patient_id.map(start))
                    & selected.time.le(selected.patient_id.map(end))
                ]
                grouped = window.groupby("patient_id").value
                summaries = grouped.agg(
                    ["mean", "sum", "max", "count"]
                ).reindex(targets.index)
                summaries["event_count"] = grouped.size().reindex(
                    targets.index, fill_value=0
                )
                summaries[["sum", "count"]] = summaries[
                    ["sum", "count"]
                ].fillna(0)
                for aggregation in summaries:
                    for kind, values in [
                        ("value", summaries[aggregation]),
                        (
                            "missing_indicator",
                            summaries[aggregation].isna().astype(float),
                        ),
                    ]:
                        metrics = statistics(values, targets)
                        exceeded = [
                            name
                            for name, limit in LIMITS.items()
                            if abs(metrics[name]) >= limit
                        ]
                        records.append(
                            {
                                "phase": phase,
                                "feature": feature,
                                "region": region,
                                "anchor": anchor,
                                "offset": offset,
                                "aggregation": aggregation,
                                "summary_kind": kind,
                                **metrics,
                                "exceeded": "|".join(exceeded),
                                "strong": bool(exceeded),
                            }
                        )
    return pd.DataFrame(records)


def strongest_table(audits: pd.DataFrame) -> pd.DataFrame:
    """Compare maxima on a common scale: fraction of each predefined limit."""
    records = []
    for feature in MODIFIABLE:
        row = {"feature": feature}
        for phase in ("before", "after"):
            subset = audits.loc[
                audits.feature.eq(feature) & audits.phase.eq(phase)
            ]
            scores = subset[list(LIMITS)].abs().div(pd.Series(LIMITS))
            stacked = scores.stack()
            if stacked.empty:
                continue
            index, statistic = stacked.idxmax()
            best = subset.loc[index]
            row.update(
                {
                    f"{phase}_{key}": best[key]
                    for key in (
                        "anchor",
                        "offset",
                        "aggregation",
                        "summary_kind",
                        "n",
                        "n_positive",
                        "n_negative",
                    )
                }
            )
            row[f"{phase}_statistic"] = statistic
            row[f"{phase}_absolute_value"] = abs(best[statistic])
            row[f"{phase}_limit_ratio"] = stacked.max()
        records.append(row)
    return pd.DataFrame(records)


def assert_exact_frame(left: pd.DataFrame, right: pd.DataFrame) -> None:
    """Check exact values, indexes and dtypes without scalar diagnostic overhead."""
    if not left.equals(right) or not left.dtypes.equals(right.dtypes):
        raise AssertionError("DataFrames differ")


def validate(source: pd.DataFrame, result: pd.DataFrame) -> dict:
    """Assert exact structural identity and preserved marginal value multisets."""
    assert_exact_frame(
        source.drop(columns="value"), result.drop(columns="value"),
    )
    protected = source.feature_name.isin(PROTECTED + ("ClinicalVisitYN",))
    assert_exact_frame(source.loc[protected], result.loc[protected])
    changes = {}
    for feature in MODIFIABLE:
        before = source.loc[source.feature_name.eq(feature), "value"].astype(
            float
        )
        after = result.loc[result.feature_name.eq(feature), "value"].astype(
            float
        )
        np.testing.assert_array_equal(np.sort(before), np.sort(after))
        changes[feature] = int(
            (~(before.eq(after) | (before.isna() & after.isna()))).sum()
        )
    ordered = result.sort_values(
        ["patient_id", "time", "feature_name"], kind="stable"
    )
    assert_exact_frame(result, ordered)
    return {
        "patient_count": int(source.patient_id.nunique()),
        "same_ids_anchors_timestamps_rows_dtypes": True,
        "protected_rows_identical": True,
        "marginal_values_and_missing_counts_identical": True,
        "temporal_order_unchanged": True,
        "changed_value_counts": changes,
        "protected_rows_sha256_before": frame_hash(source.loc[protected]),
        "protected_rows_sha256_after": frame_hash(result.loc[protected]),
    }


def create(
    source: Path | None = None, output: Path = OUTPUT, old_v3: Path | None = None
) -> dict:
    """Generate from scratch, or transform explicitly supplied base artifacts.

    An optional old_v3 directory is snapshotted only for historical audits;
    neither it nor a saved intermediate dataset is required by default.
    """
    output = output.resolve()
    if source is None:
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".experiment1_base_", dir=output.parent,
        ) as temporary:
            source = Path(temporary) / "base"
            generate_base_artifacts(source)
            return create(source=source, output=output, old_v3=old_v3)
    source = source.resolve()
    old_v3 = old_v3.resolve() if old_v3 is not None else None
    protected = {"v2": source}
    if old_v3 is not None:
        protected["old_v3"] = old_v3
    protected_paths = tuple(protected.values())
    if output.exists() or any(
        output == p or p in output.parents or output in p.parents
        for p in protected_paths
    ):
        raise FileExistsError(
            "Output must be new and separate from V2 and old V3"
        )
    initial = {name: snapshot(path) for name, path in protected.items()}
    data = pd.read_pickle(source / X_NAME)
    ground = pd.read_pickle(
        source / "temporal_benchmark_ground_truth_5000.pkl"
    ).set_index("patient_id")
    labels = pd.read_pickle(
        source / "temporal_benchmark_fa_labels_5000.pkl"
    ).set_index("patient_id")
    if (
        not ground.index.is_unique
        or not labels.index.is_unique
        or set(ground.index) != set(labels.index)
    ):
        raise ValueError(
            "Ground truth and labels require identical unique IDs"
        )
    targets = ground.join(labels)
    if len(targets) != 5000 or set(data.patient_id) != set(targets.index):
        raise ValueError("Expected the original 5000-patient cohort")
    print("Applying the single predefined permutation", flush=True)
    result, donors = transform(data)
    validation = validate(data, result)
    print("Auditing V2", flush=True)
    before = audit(data, targets, "before")
    print("Auditing transformed data", flush=True)
    after = audit(result, targets, "after")
    audits = pd.concat([before, after], ignore_index=True)
    eligible = after.loc[after.feature.isin(MODIFIABLE)]
    late = after.loc[after.region.eq("post_6_months")]
    validation.update(
        distractor_acceptance_passed=not bool(eligible.strong.any()),
        distractor_flagged_summaries=int(eligible.strong.sum()),
        late_feeding_flagged_summaries=int(late.strong.sum()),
    )
    metadata = {
        "source": "generated intermediate base" if source.parent.name.startswith(".experiment1_base_") else str(source),
        "seed": SEED,
        "rng": "PCG64",
        "protected_features": PROTECTED,
        "modifiable_features": MODIFIABLE,
        "limits": LIMITS,
        "donors": donors,
        "method": "Joint demographics/growth permutation within growth row count; food sequence permutation within food row count; recipient timestamps unchanged.",
        "label_inputs_to_transformation": False,
        "seed_attempts": 1,
        "clinical_visit_values": "unchanged; already constant one",
        "copied_metadata_scope": "Original V2 provenance, copied verbatim; see this file for derivation.",
        "acceptance_scope": "Univariate effect-size screens, including missingness and density; not a guarantee of multivariate independence.",
        "maximum_definition": "absolute statistic divided by its predefined limit",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".clean_fa_", dir=output.parent
    ) as temporary:
        stage = Path(temporary) / output.name
        stage.mkdir()
        result.to_pickle(stage / X_NAME)
        copied = {}
        for path in sorted(source.iterdir()):
            if path.is_file() and path.name != X_NAME:
                shutil.copyfile(path, stage / path.name)
                copied[path.name] = sha256(stage / path.name)
                if copied[path.name] != initial["v2"][path.name]["sha256"]:
                    raise AssertionError("Copied protected artifact differs")
        assert_exact_frame(result, pd.read_pickle(stage / X_NAME))
        final = {name: snapshot(path) for name, path in protected.items()}
        if final != initial:
            raise AssertionError("A protected source directory changed")
        validation.update(
            source_directories_untouched=True,
            copied_artifact_sha256=copied,
            source_snapshots_before=initial,
            source_snapshots_after=final,
            transformed_x_sha256=sha256(stage / X_NAME),
        )
        audits.to_csv(stage / "distractor_association_audit.csv", index=False)
        strongest_table(audits).to_csv(
            stage / "before_after_audit.csv", index=False
        )
        for name, content in [
            ("transformation_metadata.json", metadata),
            ("validation_summary.json", validation),
        ]:
            (stage / name).write_text(
                json.dumps(portable_metadata(content), indent=2, allow_nan=False) + "\n"
            )
        stage.replace(output)
    print(
        json.dumps(
            {
                k: v
                for k, v in validation.items()
                if not k.startswith("source_snapshots")
            },
            indent=2,
        )
    )
    return validation


def generate_base_artifacts(output: Path) -> None:
    """Stage the original 5000-patient base with unchanged scientific settings."""
    from generators.experiment1_longitudinal import (
        BenchmarkConfig, PUBLICATION_V2_PROFILE, generate_benchmark, save_benchmark,
    )
    from generators.experiment1_gi import (
        GIConfig, generate_gi_benchmark, save_gi_benchmark,
        independence_audit, temporal_specificity_audit, sha256_file,
    )

    config = BenchmarkConfig(
        n_patients=5000, output_dir=output, generation_profile=PUBLICATION_V2_PROFILE,
    )
    base = generate_benchmark(config)
    paths = save_benchmark(base, config)
    gi_config = GIConfig(
        n_patients=5000, output_dir=output, benchmark_version="publication_v2_gi_v1",
    )
    gi = generate_gi_benchmark(base.data, gi_config)
    gi.metadata["source_artifacts"] = {
        key: {"path": path.name, "sha256": sha256_file(path)}
        for key, path in {
            "longitudinal_x": paths["data"], "fa_labels": paths["fa_labels"],
            "fa_ground_truth": paths["ground_truth"],
        }.items()
    }
    gi.metadata["temporal_specificity_audit"] = temporal_specificity_audit(base.data)
    gi.metadata["independence_audit"] = independence_audit(
        gi, base.fa_labels, base.ground_truth,
    )
    gi.metadata["validation"]["checks"].update(
        source_artifact_hashes_unchanged=True, same_x_fa_gi_patient_ids=True,
    )
    # Only output-location metadata changes; all data/label values are unchanged.
    base.metadata["output_directory"] = "generated intermediate base"
    paths["metadata"].write_text(json.dumps(base.metadata, indent=2, sort_keys=True))
    save_gi_benchmark(gi, gi_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=None,
                        help="Optional existing base artifacts; default regenerates them.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument(
        "--old-v3-dir", type=Path, default=None,
        help="Optionally verify that an existing old-V3 directory is untouched.",
    )
    args = parser.parse_args()
    create(args.source_dir, args.output_dir, args.old_v3_dir)
