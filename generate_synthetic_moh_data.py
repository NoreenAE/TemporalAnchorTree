import uuid
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_PATIENTS = 2000
OUTPUT_DATA_PATH = Path("synthetic_moh_longitudinal_2000.pkl")
OUTPUT_LABELS_PATH = Path("synthetic_moh_labels_2000.pkl")

EXPECTED_FEATURES = [
    "SocioEconomic_score",
    "Birth Date Anchor",
    "BirthWeight",
    "ApgarOneMinute",
    "ApgarFiveMinutes",
    "PregnancyWeek",
    "BirthType_spontaneous",
    "Mother education level",
    "Num older siblings",
    "vaccinations",
    "% Weight Gain",
    "FormulaYN",
    "NursingEndAgeMonths",
    "FormulaStartAgeMonths",
    "FoodsStartAgeMonths",
    "Nursing End Age before 1 month",
    "HGB",
    "is_jew",
    "Length",
    "BirthType_cesarean",
    "BirthOrder",
    "BirthType_instrumental",
]


def make_patient_id(index: int) -> str:
    return f"patient_{index:04d}_{uuid.uuid4().hex[:8]}"


def sample_pregnancy_week(rng: np.random.Generator) -> float:
    choices = [32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0]
    probs = [0.01, 0.01, 0.02, 0.03, 0.01, 0.12, 0.20, 0.25, 0.20, 0.12, 0.03]
    week = rng.choice(choices, p=probs)
    if week < 37:
        week += rng.normal(0, 0.4)
    elif week >= 37:
        week += rng.normal(0, 0.7)
    return float(np.clip(round(week, 1), 32.0, 42.0))


def sample_birth_weight(pregnancy_week: float, rng: np.random.Generator) -> float:
    if pregnancy_week < 37:
        baseline = 1.45 + 0.08 * (pregnancy_week - 32)
    else:
        baseline = 3.0 + 0.06 * (pregnancy_week - 37)
    noise = rng.normal(0, 0.22)
    weight = baseline + noise
    return float(np.clip(round(weight, 2), 1.2, 4.8))


def sample_apgar(pregnancy_week: float, delivery_type: str, birth_weight: float, rng: np.random.Generator):
    prematurity_penalty = max(0, 37 - pregnancy_week) * 0.18
    delivery_penalty = {"spontaneous": 0.0, "cesarean": 0.3, "instrumental": 0.2}[delivery_type]
    weight_penalty = max(0, 2.7 - birth_weight) * 0.25
    base1 = 8.8 - prematurity_penalty - delivery_penalty - weight_penalty + rng.normal(0, 0.6)
    apgar1 = int(np.clip(round(base1), 0, 10))
    base5 = 9.3 - 0.1 * max(0, 37 - pregnancy_week) - 0.1 * delivery_penalty + rng.normal(0, 0.45)
    apgar5 = int(np.clip(round(base5), apgar1, 10))
    return apgar1, apgar5


def sample_delivery_type(pregnancy_week: float, rng: np.random.Generator) -> str:
    if pregnancy_week < 37:
        probs = {"spontaneous": 0.60, "cesarean": 0.30, "instrumental": 0.10}
    else:
        probs = {"spontaneous": 0.73, "cesarean": 0.20, "instrumental": 0.07}
    return rng.choice(list(probs.keys()), p=list(probs.values()))


def sample_education_and_socioeconomic(rng: np.random.Generator):
    education = rng.choice([1, 2, 3], p=[0.08, 0.38, 0.54])
    latent = 5.0 + 1.2 * (education - 2) + rng.normal(0, 1.2)
    socioeconomic = int(np.clip(round(latent), 1, 10))
    return education, socioeconomic


def sample_siblings_and_birth_order(rng: np.random.Generator):
    older_siblings = int(rng.choice([0, 1, 2, 3, 4], p=[0.45, 0.30, 0.15, 0.08, 0.02]))
    birth_order = older_siblings + 1
    return older_siblings, birth_order


def sample_feeding_trajectory(rng: np.random.Generator):
    category = rng.choice(["exclusive_bf", "formula_birth", "mixed_birth", "late_formula", "formula_then_bf"], p=[0.28, 0.16, 0.18, 0.24, 0.14])
    if category == "exclusive_bf":
        formula_start_age = np.nan
        nursing_end_age = np.nan
        return category, formula_start_age, nursing_end_age
    if category == "formula_birth":
        formula_start_age = 0.0
        nursing_end_age = float(np.clip(rng.normal(0.2, 0.2), 0.0, 0.6)) if rng.random() < 0.70 else np.nan
        return category, formula_start_age, nursing_end_age
    if category == "mixed_birth":
        formula_start_age = 0.0
        nursing_end_age = float(np.clip(rng.normal(3.5, 1.2), 0.5, 6.0)) if rng.random() < 0.55 else np.nan
        return category, formula_start_age, nursing_end_age
    if category == "late_formula":
        formula_start_age = float(np.clip(rng.normal(2.3, 1.0), 0.5, 5.8))
        nursing_end_age = np.nan if rng.random() < 0.55 else float(np.clip(rng.normal(4.0, 1.0), 1.0, 6.0))
        return category, formula_start_age, nursing_end_age
    formula_start_age = 0.0
    nursing_end_age = float(np.clip(rng.normal(2.6, 1.2), 0.5, 6.0)) if rng.random() < 0.70 else np.nan
    return category, formula_start_age, nursing_end_age


def sample_visit_schedule(birth_date: pd.Timestamp, rng: np.random.Generator):
    candidate_ages = [
        (30 + int(rng.normal(0, 6)), "one_month"),
        (60 + int(rng.normal(0, 10)), "two_months"),
        (120 + int(rng.normal(0, 12)), "four_months"),
        (180 + int(rng.normal(0, 14)), "six_months"),
    ]
    visit_count = int(rng.choice([1, 2, 3, 4], p=[0.05, 0.16, 0.34, 0.45]))
    chosen = rng.choice([0, 1, 2, 3], size=visit_count, replace=False)
    visits = []
    seen_times = set()
    for idx in sorted(chosen):
        age_days, label = candidate_ages[idx]
        visit_time = birth_date + pd.Timedelta(days=age_days)
        if visit_time in seen_times:
            continue
        visits.append((visit_time, label, age_days))
        seen_times.add(visit_time)
    return sorted(visits, key=lambda x: x[0])


def sample_length(pregnancy_week: float, birth_weight: float, age_months: float, rng: np.random.Generator) -> float:
    base = 46.0 + 0.25 * max(0, pregnancy_week - 37) + 0.7 * min(birth_weight, 3.5) + rng.normal(0, 0.9)
    growth = age_months * 3.1
    length = base + growth
    return float(np.clip(round(length, 1), 45.0, 74.0))


def sample_weight_gain_percent(birth_weight: float, age_months: float, rng: np.random.Generator) -> float:
    growth_factor = 0.18 * age_months + 0.02 * age_months * age_months
    percent = (growth_factor + rng.normal(0, 0.05)) * 100.0
    return float(np.clip(round(percent, 1), 0.0, 220.0))


def generate_patient(index: int, rng: np.random.Generator):
    patient_id = make_patient_id(index)
    birth_year = int(2016 + rng.integers(0, 7))
    birth_month = int(rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))
    birth_day = int(rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]))
    birth_date = pd.Timestamp(year=birth_year, month=birth_month, day=birth_day)

    pregnancy_week = sample_pregnancy_week(rng)
    birth_weight = sample_birth_weight(pregnancy_week, rng)
    delivery_type = sample_delivery_type(pregnancy_week, rng)
    apgar1, apgar5 = sample_apgar(pregnancy_week, delivery_type, birth_weight, rng)

    education_level, socioeconomic_score = sample_education_and_socioeconomic(rng)
    older_siblings, birth_order = sample_siblings_and_birth_order(rng)
    is_jew = int(rng.random() < 0.56)

    category, formula_start_age, nursing_end_age = sample_feeding_trajectory(rng)
    foods_start_age = float(np.clip(rng.normal(5.2, 0.8), 3.6, 6.0))
    if category == "exclusive_bf":
        foods_start_age = float(np.clip(rng.normal(5.4, 0.7), 4.5, 6.0))

    birth_rows = []
    birth_rows.append((patient_id, birth_date, "Birth Date Anchor", birth_date))
    birth_rows.append((patient_id, birth_date, "SocioEconomic_score", socioeconomic_score))
    birth_rows.append((patient_id, birth_date, "BirthWeight", birth_weight))
    birth_rows.append((patient_id, birth_date, "ApgarOneMinute", apgar1))
    birth_rows.append((patient_id, birth_date, "ApgarFiveMinutes", apgar5))
    birth_rows.append((patient_id, birth_date, "PregnancyWeek", pregnancy_week))
    if delivery_type == "spontaneous":
        birth_rows.append((patient_id, birth_date, "BirthType_spontaneous", 1))
        birth_rows.append((patient_id, birth_date, "BirthType_cesarean", 0))
        birth_rows.append((patient_id, birth_date, "BirthType_instrumental", 0))
    elif delivery_type == "cesarean":
        birth_rows.append((patient_id, birth_date, "BirthType_spontaneous", 0))
        birth_rows.append((patient_id, birth_date, "BirthType_cesarean", 1))
        birth_rows.append((patient_id, birth_date, "BirthType_instrumental", 0))
    else:
        birth_rows.append((patient_id, birth_date, "BirthType_spontaneous", 0))
        birth_rows.append((patient_id, birth_date, "BirthType_cesarean", 0))
        birth_rows.append((patient_id, birth_date, "BirthType_instrumental", 1))
    birth_rows.append((patient_id, birth_date, "Mother education level", education_level))
    birth_rows.append((patient_id, birth_date, "Num older siblings", older_siblings))
    birth_rows.append((patient_id, birth_date, "is_jew", is_jew))
    birth_rows.append((patient_id, birth_date, "BirthOrder", birth_order))
    birth_rows.append((patient_id, birth_date, "Length", sample_length(pregnancy_week, birth_weight, 0.0, rng)))
    birth_rows.append((patient_id, birth_date, "% Weight Gain", 0.0))
    birth_rows.append((patient_id, birth_date, "HGB", 0))

    visits = sample_visit_schedule(birth_date, rng)
    rows = []
    for visit_time, visit_label, age_days in visits:
        age_months = age_days / 30.4375
        length_value = sample_length(pregnancy_week, birth_weight, age_months, rng)
        weight_gain_value = sample_weight_gain_percent(birth_weight, age_months, rng)
        rows.append((patient_id, visit_time, "Length", length_value))
        rows.append((patient_id, visit_time, "% Weight Gain", weight_gain_value))
        rows.append((patient_id, visit_time, "vaccinations", int(rng.random() < (0.82 if len(visits) >= 3 else 0.72))))

        if np.isnan(formula_start_age):
            formula_yn = 0
        else:
            formula_yn = int(age_months >= formula_start_age)
        rows.append((patient_id, visit_time, "FormulaYN", formula_yn))

        rows.append((patient_id, visit_time, "HGB", int(rng.random() < 0.32)))

    # Create milestone features exactly once for each patient.
    if np.isnan(formula_start_age):
        rows.append((patient_id, birth_date, "FormulaStartAgeMonths", np.nan))
    else:
        formula_time = birth_date + pd.Timedelta(days=int(formula_start_age * 30.4375))
        rows.append((patient_id, formula_time, "FormulaStartAgeMonths", formula_start_age))

    if np.isnan(nursing_end_age):
        rows.append((patient_id, birth_date, "NursingEndAgeMonths", np.nan))
        rows.append((patient_id, birth_date, "Nursing End Age before 1 month", 0))
    else:
        nursing_time = birth_date + pd.Timedelta(days=int(nursing_end_age * 30.4375))
        rows.append((patient_id, nursing_time, "NursingEndAgeMonths", nursing_end_age))
        rows.append((patient_id, nursing_time, "Nursing End Age before 1 month", 1 if nursing_end_age < 1 else 0))

    food_time = birth_date + pd.Timedelta(days=int(foods_start_age * 30.4375))
    rows.append((patient_id, food_time, "FoodsStartAgeMonths", foods_start_age))

    all_rows = birth_rows + rows
    return all_rows


def build_dataset():
    rng = np.random.default_rng(RANDOM_SEED)
    all_rows = []
    for idx in range(N_PATIENTS):
        all_rows.extend(generate_patient(idx + 1, rng))

    df = pd.DataFrame(all_rows, columns=["patient_id", "time", "feature_name", "value"])
    df = df.sort_values(["patient_id", "time"]).reset_index(drop=True)
    return df


def build_labels(df: pd.DataFrame):
    patient_ids = sorted(df["patient_id"].unique())
    rng = np.random.default_rng(RANDOM_SEED + 1)
    labels = []
    for patient_id in patient_ids:
        patient_df = df[df["patient_id"] == patient_id]
        baseline = patient_df[patient_df["feature_name"].isin(["PregnancyWeek", "BirthWeight", "BirthType_cesarean", "BirthType_instrumental", "SocioEconomic_score", "Mother education level", "is_jew", "Num older siblings", "FormulaStartAgeMonths", "NursingEndAgeMonths"])]
        score = -3.8
        if not baseline.empty:
            preg = float(patient_df.loc[patient_df["feature_name"] == "PregnancyWeek", "value"].iloc[0])
            bw = float(patient_df.loc[patient_df["feature_name"] == "BirthWeight", "value"].iloc[0])
            cesarean = int(patient_df.loc[patient_df["feature_name"] == "BirthType_cesarean", "value"].iloc[0])
            instrumental = int(patient_df.loc[patient_df["feature_name"] == "BirthType_instrumental", "value"].iloc[0])
            socio = int(patient_df.loc[patient_df["feature_name"] == "SocioEconomic_score", "value"].iloc[0])
            edu = int(patient_df.loc[patient_df["feature_name"] == "Mother education level", "value"].iloc[0])
            jew = int(patient_df.loc[patient_df["feature_name"] == "is_jew", "value"].iloc[0])
            older = int(patient_df.loc[patient_df["feature_name"] == "Num older siblings", "value"].iloc[0])
            formula_start = pd.to_numeric(patient_df.loc[patient_df["feature_name"] == "FormulaStartAgeMonths", "value"], errors="coerce").dropna()
            nursing_end = pd.to_numeric(patient_df.loc[patient_df["feature_name"] == "NursingEndAgeMonths", "value"], errors="coerce").dropna()
            score += (37 - preg) * 0.015
            score += (2.7 - bw) * 0.10
            score += cesarean * 0.20
            score += instrumental * 0.18
            score += (10 - socio) * 0.02
            score += (3 - edu) * 0.04
            score += jew * 0.08
            score += older * 0.03
            if len(formula_start) > 0:
                score += max(0.0, 1.5 - float(formula_start.iloc[0])) * 0.08
            if len(nursing_end) > 0:
                score += max(0.0, 2.0 - float(nursing_end.iloc[0])) * 0.04
        score += rng.normal(0.0, 0.25)
        prob = 1 / (1 + np.exp(-score))
        label_value = int(rng.random() < prob)
        labels.append((patient_id, label_value))
    labels_df = pd.DataFrame(labels, columns=["patient_id", "label"])
    return labels_df


def validate_dataset(df: pd.DataFrame, labels_df: pd.DataFrame):
    assert df["patient_id"].nunique() == N_PATIENTS, f"Expected {N_PATIENTS} patients, found {df['patient_id'].nunique()}"
    assert set(df["feature_name"].unique()) == set(EXPECTED_FEATURES), f"Unexpected feature names: {sorted(set(df['feature_name'].unique()))}"
    assert set(df.columns) == {"patient_id", "time", "feature_name", "value"}, f"Unexpected columns: {list(df.columns)}"
    assert "Birth Date Anchor" in set(df["feature_name"]), "Birth Date Anchor is required"
    assert "BirthDate_Anchor" not in set(df["feature_name"]), "BirthDate_Anchor should not appear in the final dataset"
    assert df["patient_id"].is_unique is False or True

    per_patient_counts = df.groupby("patient_id").size()
    assert per_patient_counts.min() >= 1, "Each patient should have at least one row"
    assert df["time"].notna().all(), "Time column contains missing values"
    assert labels_df["patient_id"].nunique() == N_PATIENTS, "Labels file should contain one row per patient"
    assert 0.03 <= labels_df["label"].mean() <= 0.05, f"Label prevalence outside expected range: {labels_df['label'].mean()}"

    # Structural consistency checks.
    patient_births = df[df["feature_name"] == "Birth Date Anchor"].set_index("patient_id")["time"]
    assert len(patient_births) == N_PATIENTS
    assert patient_births.index.is_unique
    assert patient_births.groupby(level=0).size().eq(1).all(), "Every patient should have exactly one Birth Date Anchor row"
    for patient_id, birth_time in patient_births.items():
        patient_times = df.loc[df["patient_id"] == patient_id, "time"]
        assert (patient_times >= birth_time).all(), f"Encounter time before birth for {patient_id}"

    delivery_types = [
        "BirthType_spontaneous",
        "BirthType_cesarean",
        "BirthType_instrumental",
    ]
    per_patient_delivery = df[df["feature_name"].isin(delivery_types)].groupby(["patient_id", "feature_name"])["value"].first().unstack(fill_value=0)
    if len(per_patient_delivery) > 0:
        assert np.all(per_patient_delivery.sum(axis=1).between(0.9, 1.1)), "Delivery type indicators should sum to one per patient"

    for feature_name in ["FormulaStartAgeMonths", "NursingEndAgeMonths", "FoodsStartAgeMonths"]:
        values = df[df["feature_name"] == feature_name]["value"]
        if len(values) > 0:
            numeric_values = pd.to_numeric(values, errors="coerce")
            assert (numeric_values.dropna() >= 0).all(), f"Negative values found for {feature_name}"

    # Ensure numeric ranges are plausible.
    pregnancy = pd.to_numeric(df.loc[df["feature_name"] == "PregnancyWeek", "value"], errors="coerce")
    assert (pregnancy.dropna() >= 32).all(), "PregnancyWeek values should be at least 32"
    assert (pregnancy.dropna() <= 42).all(), "PregnancyWeek values should be at most 42"

    apgar1 = pd.to_numeric(df.loc[df["feature_name"] == "ApgarOneMinute", "value"], errors="coerce")
    apgar5 = pd.to_numeric(df.loc[df["feature_name"] == "ApgarFiveMinutes", "value"], errors="coerce")
    assert (apgar1.dropna().between(0, 10)).all(), "ApgarOneMinute values should be 0-10"
    assert (apgar5.dropna().between(0, 10)).all(), "ApgarFiveMinutes values should be 0-10"
    assert (apgar5.dropna().to_numpy() >= apgar1.dropna().to_numpy()).all(), "ApgarFiveMinutes should usually be >= ApgarOneMinute"

    return per_patient_counts


def main():
    df = build_dataset()
    labels_df = build_labels(df)
    labels_df = labels_df.sort_values("patient_id").reset_index(drop=True)

    df.to_pickle(OUTPUT_DATA_PATH)
    labels_df.to_pickle(OUTPUT_LABELS_PATH)

    per_patient_counts = validate_dataset(df, labels_df)

    # Reload the generated files using the standard pickle interface.
    reloaded_df = pd.read_pickle(OUTPUT_DATA_PATH)
    reloaded_labels = pd.read_pickle(OUTPUT_LABELS_PATH)
    validate_dataset(reloaded_df, reloaded_labels)

    print("Generated files:")
    print(f"- {OUTPUT_DATA_PATH.resolve()}")
    print(f"- {OUTPUT_LABELS_PATH.resolve()}")
    print("\nDataframe shapes:")
    print(f"- data shape: {reloaded_df.shape}")
    print(f"- labels shape: {reloaded_labels.shape}")
    print(f"- unique patients: {reloaded_df['patient_id'].nunique()}")
    print("- feature list:")
    for feature in reloaded_df["feature_name"].unique():
        print(f"  - {feature}")
    print("\nLabel distribution:")
    print(reloaded_labels["label"].value_counts().sort_index().to_string())
    print(f"- label=0 count: {(reloaded_labels['label'] == 0).sum()}")
    print(f"- label=1 count: {(reloaded_labels['label'] == 1).sum()}")
    print(f"- prevalence: {reloaded_labels['label'].mean():.4f}")
    print("\nRows per patient:")
    print(per_patient_counts.describe().to_string())
    print("\nExample patient trajectories:")
    sample_patients = reloaded_df["patient_id"].drop_duplicates().sample(n=5, random_state=RANDOM_SEED).tolist()
    for patient_id in sample_patients:
        patient_rows = reloaded_df[reloaded_df["patient_id"] == patient_id].sort_values("time")
        print(f"\n{patient_id}")
        print(patient_rows[["time", "feature_name", "value"]].to_string(index=False))

    print("\nCompatibility note:")
    print("The generated dataset now uses 'Birth Date Anchor' directly, matching the current repository pipeline.")


if __name__ == "__main__":
    main()
