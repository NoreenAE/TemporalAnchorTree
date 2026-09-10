"""Identity, permutation, and analytical audit checks for the clean FA derivative."""
import json

import numpy as np
import pandas as pd
import pytest

from create_temporal_benchmark_v3_clean_fa import (
    JOINT,
    MODIFIABLE,
    OLD_V3,
    OUTPUT,
    PROTECTED,
    SOURCE,
    X_NAME,
    assert_exact_frame,
    audit,
    create,
    sha256,
    snapshot,
    statistics,
    transform,
    validate,
)


@pytest.fixture(scope="module")
def source():
    return pd.read_pickle(SOURCE / X_NAME)


def test_reproducible_value_only_permutation(source):
    first, donors = transform(source)
    second, repeated = transform(source)
    assert_exact_frame(first, second)
    assert donors == repeated
    checks = validate(source, first)
    assert checks["patient_count"] == 5000
    assert (
        checks["protected_rows_sha256_before"]
        == checks["protected_rows_sha256_after"]
    )
    assert checks["changed_value_counts"]["ClinicalVisitYN"] == 0
    for block in donors.values():
        assert sorted(block) == sorted(block.values())
    # A demographic and both growth trajectories must share the same donor.
    for feature in ("BirthWeightKg", "LengthCm", "WeightKg"):
        original = source.loc[source.feature_name.eq(feature)].groupby(
            "patient_id"
        )
        changed = first.loc[first.feature_name.eq(feature)].groupby(
            "patient_id"
        )
        for recipient in list(donors["demographic_growth"])[:30]:
            donor = donors["demographic_growth"][recipient]
            np.testing.assert_array_equal(
                changed.get_group(recipient).value.astype(float),
                original.get_group(donor).value.astype(float),
            )


def test_saved_artifact_identity_and_source_snapshots(source):
    report = json.loads((OUTPUT / "validation_summary.json").read_text())
    assert snapshot(SOURCE) == report["source_snapshots_before"]["v2"]
    assert snapshot(OLD_V3) == report["source_snapshots_before"]["old_v3"]
    assert (
        report["source_snapshots_before"] == report["source_snapshots_after"]
    )
    for name, digest in report["copied_artifact_sha256"].items():
        assert sha256(SOURCE / name) == sha256(OUTPUT / name) == digest
    result = pd.read_pickle(OUTPUT / X_NAME)
    validate(source, result)
    expected, _ = transform(source)
    assert_exact_frame(expected, result)
    # Exact GI feature and artifact identity protects the entire GI mechanism.
    for feature in PROTECTED:
        mask = source.feature_name.eq(feature)
        assert_exact_frame(source.loc[mask], result.loc[mask])


def test_refuses_existing_output():
    with pytest.raises(FileExistsError):
        create(output=OLD_V3)
    with pytest.raises(FileExistsError):
        create(output=SOURCE)


def test_statistics_known_effects_and_missingness():
    target = pd.DataFrame(
        {
            "label": [0, 0, 1, 1],
            "fa_probability": [0.1, 0.1, 0.9, 0.9],
            "feeding_group": ["a", "a", "b", "b"],
        }
    )
    perfect = statistics(pd.Series([0.0, 0.0, 1.0, 1.0]), target)
    assert perfect["auc"] == 1
    assert perfect["label_r"] == pytest.approx(1)
    assert perfect["probability_r"] == pytest.approx(1)
    assert perfect["group_eta"] == pytest.approx(1)
    reverse = statistics(pd.Series([1.0, 1.0, 0.0, 0.0]), target)
    assert reverse["auc"] == 0
    assert reverse["auc_distance"] == 0.5
    constant = statistics(pd.Series([2.0, 2.0, 2.0, 2.0]), target)
    assert constant["auc"] == 0.5
    assert np.isnan(constant["label_r"])
    assert statistics(pd.Series([np.nan] * 4), target)["n"] == 0


def test_audit_inclusive_calendar_fractional_and_late_boundaries():
    birth = pd.Timestamp("2020-01-31")
    target = pd.DataFrame(
        {
            "birth_date": [birth] * 4,
            "now": [pd.Timestamp("2021-08-01")] * 4,
            "label": [0, 0, 1, 1],
            "fa_probability": [0.1, 0.1, 0.9, 0.9],
            "feeding_group": ["a", "a", "b", "b"],
        },
        index=["a", "b", "c", "d"],
    )
    rows = []
    for pid in target.index:
        for feature in MODIFIABLE + ("FormulaYN", "BreastMilkYN"):
            for time in [
                birth + pd.Timedelta(days=21),
                birth + pd.DateOffset(months=1),
                birth + pd.DateOffset(months=6),
            ]:
                rows.append((pid, time, feature, 1.0))
    data = pd.DataFrame(
        rows, columns=["patient_id", "time", "feature_name", "value"]
    )
    report = audit(data, target, "before")
    assert len(report) == (len(MODIFIABLE) + 4) * 18 * 5 * 2
    early = report.loc[
        report.anchor.eq("Birth")
        & report.offset.eq(0.7)
        & report.aggregation.eq("mean")
        & report.summary_kind.eq("value")
        & report.region.eq("all")
    ]
    assert early.n.eq(4).all()
    late = report.loc[
        report.region.eq("post_6_months")
        & report.aggregation.eq("mean")
        & report.summary_kind.eq("value")
    ]
    assert late.n.eq(
        0
    ).all()  # Exactly month six remains protected early feeding.
