from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


SEED = 2026
N_PATIENTS = 5000
OUTPUT_DIR = Path(".")
DATA_OUTPUT = OUTPUT_DIR / "synthetic_temporal_unit_test_5000.pkl"
LABELS_OUTPUT = OUTPUT_DIR / "synthetic_temporal_unit_test_labels_5000.pkl"

FEATURES = [
    "% Weight Gain",
    "ApgarFiveMinutes",
    "ApgarOneMinute",
    "Birth Date Anchor",
    "BirthOrder",
    "BirthType_cesarean",
    "BirthType_instrumental",
    "BirthType_spontaneous",
    "BirthWeight",
    "FoodsStartAgeMonths",
    "FormulaStartAgeMonths",
    "FormulaYN",
    "HGB",
    "Length",
    "Mother education level",
    "Num older siblings",
    "Nursing End Age before 1 month",
    "NursingEndAgeMonths",
    "PregnancyWeek",
    "SocioEconomic_score",
    "is_jew",
    "vaccinations",
]


def _make_background_feature_value(name: str, rng: np.random.Generator, birth_date: pd.Timestamp) -> tuple[float, str]:
    """Generate one plausible numeric or binary feature value without leaking the label."""
    if name == "% Weight Gain":
        return float(rng.normal(180, 30)), "numeric"
    if name == "ApgarFiveMinutes":
        return float(rng.integers(5, 10)), "numeric"
    if name == "ApgarOneMinute":
        return float(rng.integers(3, 9)), "numeric"
    if name == "BirthOrder":
        return float(rng.integers(1, 4)), "numeric"
    if name == "BirthType_cesarean":
        return float(rng.choice([0, 1], p=[0.75, 0.25])), "binary"
    if name == "BirthType_instrumental":
        return float(rng.choice([0, 1], p=[0.9, 0.1])), "binary"
    if name == "BirthType_spontaneous":
        return float(rng.choice([0, 1], p=[0.8, 0.2])), "binary"
    if name == "BirthWeight":
        return float(rng.normal(3.2, 0.5)), "numeric"
    if name == "FoodsStartAgeMonths":
        return float(rng.uniform(3, 12)), "numeric"
    if name == "FormulaStartAgeMonths":
        # Intentionally non-informative for the ground-truth timing rule.
        return float(rng.uniform(1, 12)), "numeric"
    if name == "FormulaYN":
        return float(0), "binary"
    if name == "HGB":
        return float(rng.normal(12.5, 1.2)), "numeric"
    if name == "Length":
        return float(rng.normal(49.5, 2.2)), "numeric"
    if name == "Mother education level":
        return float(rng.integers(1, 6)), "numeric"
    if name == "Num older siblings":
        return float(rng.poisson(0.8)), "numeric"
    if name == "Nursing End Age before 1 month":
        return float(rng.choice([0, 1], p=[0.65, 0.35])), "binary"
    if name == "NursingEndAgeMonths":
        return float(rng.uniform(0.5, 7.0)), "numeric"
    if name == "PregnancyWeek":
        return float(rng.normal(39.0, 1.5)), "numeric"
    if name == "SocioEconomic_score":
        return float(rng.normal(0, 1)), "numeric"
    if name == "is_jew":
        return float(rng.choice([0, 1], p=[0.9, 0.1])), "binary"
    if name == "vaccinations":
        return float(rng.integers(0, 6)), "numeric"
    raise ValueError(f"Unsupported feature: {name}")


def _random_visit_age_months(rng: np.random.Generator) -> float:
    # Irregular but sufficiently informative visit pattern around the critical window.
    base = rng.uniform(0.4, 8.0)
    jitter = rng.uniform(-0.35, 0.35)
    return float(max(0.2, base + jitter))


def generate_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    rows = []
    labels = []

    for pid in range(N_PATIENTS):
        patient_id = f"P{pid:05d}"
        birth_date = pd.Timestamp("2018-01-01") + pd.to_timedelta(int(rng.integers(0, 365 * 3)), unit="D")

        # Create one anchor row per patient.
        rows.append({
            "patient_id": patient_id,
            "time": birth_date,
            "feature_name": "Birth Date Anchor",
            "value": birth_date,
        })

        # Create a controlled timing group for FormulaYN.
        timing_group = rng.choice(["before", "during_2_4", "after", "never"], p=[0.25, 0.25, 0.25, 0.25])

        # Create one background observation for each non-Formula feature.
        for feature in FEATURES:
            if feature == "Birth Date Anchor":
                continue
            if feature == "FormulaYN":
                continue

            age_months = _random_visit_age_months(rng)
            event_time = birth_date + pd.to_timedelta(age_months * 30, unit="D")
            value, _ = _make_background_feature_value(feature, rng, birth_date)
            rows.append({
                "patient_id": patient_id,
                "time": event_time,
                "feature_name": feature,
                "value": value,
            })

        # Create FormulaYN observations with known temporal structure.
        # The critical rule is whether FormulaYN=1 occurs during months 2-4.
        if timing_group == "before":
            formula_time = birth_date + pd.to_timedelta(rng.uniform(0.6, 1.8) * 30, unit="D")
            formula_value = 1.0
        elif timing_group == "during_2_4":
            formula_time = birth_date + pd.to_timedelta(rng.uniform(2.0, 4.0) * 30, unit="D")
            formula_value = 1.0
        elif timing_group == "after":
            formula_time = birth_date + pd.to_timedelta(rng.uniform(5.0, 8.0) * 30, unit="D")
            formula_value = 1.0
        else:
            formula_time = None
            formula_value = 0.0

        if formula_time is not None:
            rows.append({
                "patient_id": patient_id,
                "time": formula_time,
                "feature_name": "FormulaYN",
                "value": formula_value,
            })

        # Generate label based on the temporal presence of FormulaYN in the 2-4 month window.
        # Strong signal but not deterministic.
        if timing_group == "during_2_4":
            p_pos = 0.70
        else:
            p_pos = 0.05

        # Add a little overlap/noise to avoid perfect separability.
        p_pos = min(0.95, max(0.01, p_pos + rng.normal(0.0, 0.02)))
        label = int(rng.random() < p_pos)
        labels.append({"patient_id": patient_id, "label": label})

    data = pd.DataFrame(rows)
    labels_df = pd.DataFrame(labels)

    # Ensure the long-format structure and data quality expectations.
    data = data.sort_values(["patient_id", "time", "feature_name"]).reset_index(drop=True)
    labels_df = labels_df.sort_values("patient_id").reset_index(drop=True)
    return data, labels_df


def validate_dataset(data: pd.DataFrame, labels_df: pd.DataFrame) -> None:
    print("====================================================")
    print("DATASET VALIDATION")
    print("====================================================")
    print(f"Rows in data: {len(data)}")
    print(f"Patients: {data['patient_id'].nunique()}")
    print(f"Features present: {sorted(data['feature_name'].unique().tolist())}")
    print(f"Number of unique features: {data['feature_name'].nunique()}")
    print(f"Exact duplicate rows: {data.duplicated().sum()}")
    print(f"Birth anchor rows: {(data['feature_name'] == 'Birth Date Anchor').sum()}")
    print(f"Labels rows: {len(labels_df)}")
    print(f"Label values: {sorted(labels_df['label'].unique().tolist())}")
    print(f"Positive labels: {(labels_df['label'] == 1).sum()}")
    print(f"Negative labels: {(labels_df['label'] == 0).sum()}")
    print(f"Overall prevalence: {labels_df['label'].mean():.3f}")

    # Validate the intended temporal signal.
    data_with_formula = data[data['feature_name'] == 'FormulaYN'].copy()
    data_with_formula = data_with_formula.merge(labels_df, on='patient_id', how='left')
    data_with_formula['time_in_months'] = (
        data_with_formula['time'].dt.to_pydatetime() - data_with_formula['time'].dt.to_pydatetime()
    )
    # The above line is intentionally unused; the temporal pattern is derived from the generated groups in the data itself.
    # We re-create the group logic from the generated rows directly.
    data_with_formula['month'] = data_with_formula['time'].dt.year * 12 + data_with_formula['time'].dt.month

    # Since we did not persist group membership, derive the intended ground-truth by checking whether a FormulaYN=1 row exists during 2-4 months after birth.
    birth_rows = data[data['feature_name'] == 'Birth Date Anchor'][['patient_id', 'time']].rename(columns={'time': 'birth_time'})
    formula_rows = data[data['feature_name'] == 'FormulaYN'][['patient_id', 'time', 'value']]
    formula_with_birth = formula_rows.merge(birth_rows, on='patient_id', how='left')
    formula_with_birth['months_since_birth'] = (formula_with_birth['time'] - formula_with_birth['birth_time']).dt.days / 30.4375
    has_mid_window = (
        (formula_with_birth['value'] == 1.0) &
        (formula_with_birth['months_since_birth'] >= 2.0) &
        (formula_with_birth['months_since_birth'] <= 4.0)
    ).groupby(formula_with_birth['patient_id']).any()

    signal_present = pd.Series(has_mid_window, index=has_mid_window.index).rename('mid_window')
    labels_with_signal = labels_df.merge(signal_present.reset_index(), on='patient_id', how='left')
    labels_with_signal['mid_window'] = labels_with_signal['mid_window'].fillna(False)

    print("Temporal pattern present count:", int(labels_with_signal['mid_window'].sum()))
    print("Temporal pattern absent count:", int((~labels_with_signal['mid_window']).sum()))
    print("Prevalence with 2-4 month pattern:", labels_with_signal.loc[labels_with_signal['mid_window'], 'label'].mean())
    print("Prevalence without pattern:", labels_with_signal.loc[~labels_with_signal['mid_window'], 'label'].mean())

    # Timing subgroup breakdown using the same rule based on the observed formula row timing.
    formula_with_birth['timing_group'] = 'never'
    formula_with_birth.loc[(formula_with_birth['value'] == 1.0) & (formula_with_birth['months_since_birth'] < 2.0), 'timing_group'] = 'before'
    formula_with_birth.loc[(formula_with_birth['value'] == 1.0) & (formula_with_birth['months_since_birth'] >= 2.0) & (formula_with_birth['months_since_birth'] <= 4.0), 'timing_group'] = 'during_2_4'
    formula_with_birth.loc[(formula_with_birth['value'] == 1.0) & (formula_with_birth['months_since_birth'] > 4.0), 'timing_group'] = 'after'
    timing_summary = formula_with_birth.groupby('patient_id')['timing_group'].last().reset_index()
    labels_with_timing = labels_df.merge(timing_summary, on='patient_id', how='left')
    labels_with_timing['timing_group'] = labels_with_timing['timing_group'].fillna('never')
    for group in ['before', 'during_2_4', 'after', 'never']:
        subset = labels_with_timing.loc[labels_with_timing['timing_group'] == group]
        if len(subset) > 0:
            print(f"Prevalence for {group}: {subset['label'].mean():.3f}")

    birth_times = data.loc[data['feature_name'] == 'Birth Date Anchor', ['patient_id', 'time']].rename(columns={'time': 'birth_time'})
    data_with_birth = data.merge(birth_times, on='patient_id', how='left')
    assert not ((data_with_birth['time'] < data_with_birth['birth_time']).any())

    assert len(labels_df) == N_PATIENTS
    assert data['patient_id'].nunique() == N_PATIENTS
    assert data['feature_name'].nunique() == len(FEATURES)
    assert data.duplicated().sum() == 0
    assert ((data['feature_name'] == 'Birth Date Anchor').sum()) == N_PATIENTS
    assert labels_df['label'].isin([0, 1]).all()
    assert labels_df['patient_id'].is_unique

    print("Validation passed.")


def main() -> None:
    data, labels_df = generate_dataset()
    data.to_pickle(DATA_OUTPUT)
    labels_df.to_pickle(LABELS_OUTPUT)
    validate_dataset(data, labels_df)
    print("Saved data to:", DATA_OUTPUT)
    print("Saved labels to:", LABELS_OUTPUT)
    print("Known ground-truth rule: FormulaYN=1 during the 2-4 month window after birth should increase the probability of the label to about 70%, while FormulaYN outside that window should not.")


if __name__ == "__main__":
    main()
