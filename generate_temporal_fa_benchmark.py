from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier


SEED = 20260809
N_PATIENTS = 5000
OUTPUT_DIR = Path(".")
DATA_OUTPUT = OUTPUT_DIR / "temporal_fa_benchmark_5000.pkl"
LABELS_OUTPUT = OUTPUT_DIR / "temporal_fa_benchmark_labels_5000.pkl"

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

WINDOWS = {
    "0_2m": (0.20, 1.80),
    "2_4m": (2.25, 3.75),
    "4_6m": (4.25, 5.75),
}

PATTERNS = {
    "exclusive_bf": {
        "states": [0.0, 0.0, 0.0],
        "probability": 0.12,
    },
    "continuous_formula": {
        "states": [1.0, 1.0, 1.0],
        "probability": 0.20,
    },
    "early_formula_then_bf": {
        "states": [1.0, 0.0, 0.0],
        "probability": 0.04,
    },
    "early_formula_then_stop_late": {
        "states": [1.0, 1.0, 0.0],
        "probability": 0.12,
    },
    "bf_formula_before_2m": {
        "states": [1.0, 1.0, 1.0],
        "probability": 0.17,
    },
    "bf_formula_2_4m": {
        "states": [0.0, 1.0, 1.0],
        "probability": 0.17,
    },
    "bf_formula_4_6m": {
        "states": [0.0, 0.0, 1.0],
        "probability": 0.12,
    },
    "inconsistent": {
        "states": [1.0, 0.0, 1.0],
        "probability": 0.06,
    },
}


def clipped_normal(
    rng: np.random.Generator,
    mean: float,
    sd: float,
    lower: float,
    upper: float,
) -> float:
    return float(np.clip(rng.normal(mean, sd), lower, upper))


def maybe_missing(value, probability: float, rng: np.random.Generator):
    return np.nan if rng.random() < probability else value


def make_patient_id(index: int) -> str:
    return f"TFA_{index:05d}"


def sample_birth_date(rng: np.random.Generator) -> pd.Timestamp:
    offset_days = int(rng.integers(0, 365 * 4))
    return pd.Timestamp("2016-01-01") + pd.Timedelta(days=offset_days)


def sample_pregnancy_week(rng: np.random.Generator) -> float:
    return round(clipped_normal(rng, 39.0, 1.45, 32.0, 42.0), 1)


def sample_delivery_type(pregnancy_week: float, rng: np.random.Generator) -> str:
    if pregnancy_week < 37:
        return str(rng.choice(
            ["spontaneous", "cesarean", "instrumental"],
            p=[0.54, 0.36, 0.10],
        ))
    return str(rng.choice(
        ["spontaneous", "cesarean", "instrumental"],
        p=[0.68, 0.24, 0.08],
    ))


def sample_birth_weight(pregnancy_week: float, rng: np.random.Generator) -> float:
    if pregnancy_week < 37:
        mean = 2.35 + 0.13 * (pregnancy_week - 34.0)
    else:
        mean = 3.25 + 0.05 * (pregnancy_week - 39.0)
    return round(clipped_normal(rng, mean, 0.38, 1.2, 4.8), 2)


def sample_apgar(
    pregnancy_week: float,
    delivery_type: str,
    birth_weight: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    prematurity_penalty = max(0.0, 37.0 - pregnancy_week) * 0.25
    low_weight_penalty = max(0.0, 2.5 - birth_weight) * 0.55
    delivery_penalty = 0.35 if delivery_type in {"cesarean", "instrumental"} else 0.0
    apgar1 = int(round(clipped_normal(
        rng,
        8.2 - prematurity_penalty - low_weight_penalty - delivery_penalty,
        1.0,
        1.0,
        10.0,
    )))
    apgar5 = int(round(clipped_normal(
        rng,
        max(apgar1, 8.9 - 0.40 * prematurity_penalty - 0.30 * low_weight_penalty),
        0.55,
        apgar1,
        10.0,
    )))
    return float(apgar1), float(apgar5)


def sample_siblings_and_birth_order(rng: np.random.Generator) -> tuple[int, int]:
    older_siblings = int(rng.choice([0, 1, 2, 3, 4], p=[0.42, 0.32, 0.16, 0.08, 0.02]))
    return older_siblings, older_siblings + 1


def sample_pattern(rng: np.random.Generator) -> str:
    names = list(PATTERNS)
    probabilities = [PATTERNS[name]["probability"] for name in names]
    return str(rng.choice(names, p=probabilities))


def sample_window_age_months(window_name: str, rng: np.random.Generator) -> float:
    lower, upper = WINDOWS[window_name]
    return float(np.clip(rng.uniform(lower, upper) + rng.normal(0.0, 0.04), lower, upper))


def formula_start_age_months(states: list[float], rng: np.random.Generator) -> float:
    formula_indices = [index for index, state in enumerate(states) if state == 1.0]
    if not formula_indices:
        return np.nan if rng.random() < 0.78 else round(float(rng.uniform(4.0, 8.5)), 2)

    first_index = formula_indices[0]
    lower, upper = list(WINDOWS.values())[first_index]
    if rng.random() < 0.58:
        return np.nan
    noisy_start = rng.uniform(lower, upper) + rng.normal(0.0, 0.75)
    return round(float(np.clip(noisy_start, 0.0, 8.0)), 2)


def nursing_end_age_months(states: list[float], rng: np.random.Generator) -> float:
    if states == [1.0, 0.0, 0.0]:
        return np.nan if rng.random() < 0.72 else round(float(rng.uniform(6.0, 13.0)), 2)
    if states == [0.0, 0.0, 0.0]:
        return np.nan if rng.random() < 0.82 else round(float(rng.uniform(6.0, 14.0)), 2)
    if states[0] == 0.0 and any(state == 1.0 for state in states[1:]):
        return np.nan if rng.random() < 0.65 else round(float(rng.uniform(2.0, 10.0)), 2)
    if states[0] == 1.0 and any(state == 1.0 for state in states[1:]):
        return np.nan if rng.random() < 0.58 else round(float(rng.uniform(0.8, 9.0)), 2)
    return np.nan if rng.random() < 0.75 else round(float(rng.uniform(4.0, 12.0)), 2)


def sample_length(
    birth_weight: float,
    pregnancy_week: float,
    age_months: float,
    rng: np.random.Generator,
) -> float:
    birth_length = 48.5 + 0.35 * (pregnancy_week - 37.0) + 0.8 * (birth_weight - 3.0)
    length = birth_length + 3.0 * age_months + rng.normal(0.0, 0.9)
    return round(clipped_normal(rng, length, 0.25, 43.0, 72.0), 1)


def sample_weight_gain(age_months: float, rng: np.random.Generator) -> float:
    gain = 12.0 + 20.5 * age_months + 1.5 * age_months**2 + rng.normal(0.0, 16.0)
    return round(float(np.clip(gain, 0.0, 230.0)), 1)


def compute_window_formula_states(data: pd.DataFrame) -> pd.DataFrame:
    birth_rows = data[data["feature_name"] == "Birth Date Anchor"][
        ["patient_id", "time"]
    ].rename(columns={"time": "birth_time"})
    formula_rows = data[data["feature_name"] == "FormulaYN"][
        ["patient_id", "time", "value"]
    ].copy()
    formula_rows = formula_rows.merge(birth_rows, on="patient_id", how="left")
    formula_rows["age_months"] = (
        (formula_rows["time"] - formula_rows["birth_time"]).dt.days / 30.4375
    )
    patients = pd.Index(data["patient_id"].unique(), name="patient_id")
    result = pd.DataFrame(index=patients)
    for window_name, (lower, upper) in WINDOWS.items():
        mask = formula_rows["age_months"].between(lower, upper, inclusive="both")
        window_values = formula_rows.loc[mask].groupby("patient_id")["value"].max()
        result[f"formula_{window_name}"] = window_values.reindex(patients).fillna(0.0)
    return result.reset_index()


def static_feature_matrix(data: pd.DataFrame) -> pd.DataFrame:
    static_rows = data[~data["feature_name"].isin(["FormulaYN", "Birth Date Anchor"])].copy()
    static_rows["value"] = pd.to_numeric(static_rows["value"], errors="coerce")
    return static_rows.pivot_table(
        index="patient_id",
        columns="feature_name",
        values="value",
        aggfunc="mean",
    )


def formula_summaries(data: pd.DataFrame) -> pd.DataFrame:
    formula_rows = data[data["feature_name"] == "FormulaYN"][["patient_id", "value"]].copy()
    formula_rows["value"] = pd.to_numeric(formula_rows["value"], errors="coerce")
    return formula_rows.groupby("patient_id")["value"].agg(
        FormulaYN_mean="mean",
        FormulaYN_sum="sum",
        FormulaYN_count="count",
        FormulaYN_max="max",
        Formula_ever="max",
    )


def visit_and_missingness_summaries(data: pd.DataFrame) -> pd.DataFrame:
    numeric_data = data.copy()
    numeric_data["is_missing_value"] = numeric_data["value"].isna().astype(float)
    return numeric_data.groupby("patient_id").agg(
        visit_row_count=("feature_name", "size"),
        missing_value_count=("is_missing_value", "sum"),
        missing_value_rate=("is_missing_value", "mean"),
    )


def make_patient(index: int, rng: np.random.Generator) -> tuple[list[dict], dict]:
    patient_id = make_patient_id(index)
    birth_date = sample_birth_date(rng)
    pattern = sample_pattern(rng)
    states = list(PATTERNS[pattern]["states"])

    pregnancy_week = sample_pregnancy_week(rng)
    delivery_type = sample_delivery_type(pregnancy_week, rng)
    birth_weight = sample_birth_weight(pregnancy_week, rng)
    apgar1, apgar5 = sample_apgar(pregnancy_week, delivery_type, birth_weight, rng)
    older_siblings, birth_order = sample_siblings_and_birth_order(rng)
    socioeconomic_score = int(rng.choice([1, 2, 3, 4, 5], p=[0.16, 0.24, 0.31, 0.20, 0.09]))
    mother_education = int(rng.choice([1, 2, 3, 4, 5], p=[0.10, 0.22, 0.35, 0.23, 0.10]))
    is_jew = int(rng.random() < 0.56)
    foods_start_age = round(clipped_normal(rng, 5.4, 0.8, 3.5, 8.5), 2)
    formula_start = formula_start_age_months(states, rng)
    nursing_end = nursing_end_age_months(states, rng)
    nursing_end_before_1m = 0.0 if pd.isna(nursing_end) else float(nursing_end < 1.0)

    rows = [{
        "patient_id": patient_id,
        "time": birth_date,
        "feature_name": "Birth Date Anchor",
        "value": birth_date,
    }]

    delivery_values = {
        "BirthType_spontaneous": float(delivery_type == "spontaneous"),
        "BirthType_cesarean": float(delivery_type == "cesarean"),
        "BirthType_instrumental": float(delivery_type == "instrumental"),
    }
    birth_features = {
        "PregnancyWeek": pregnancy_week,
        "BirthWeight": birth_weight,
        "ApgarOneMinute": apgar1,
        "ApgarFiveMinutes": apgar5,
        "SocioEconomic_score": socioeconomic_score,
        "Mother education level": mother_education,
        "is_jew": is_jew,
        "Num older siblings": older_siblings,
        "BirthOrder": birth_order,
        "FoodsStartAgeMonths": maybe_missing(foods_start_age, 0.08, rng),
        "FormulaStartAgeMonths": formula_start,
        "NursingEndAgeMonths": nursing_end,
        "Nursing End Age before 1 month": nursing_end_before_1m,
        **delivery_values,
    }

    for feature, value in birth_features.items():
        rows.append({
            "patient_id": patient_id,
            "time": birth_date + pd.Timedelta(days=int(rng.integers(0, 4))),
            "feature_name": feature,
            "value": value,
        })

    for window_index, window_name in enumerate(WINDOWS):
        age_months = sample_window_age_months(window_name, rng)
        event_time = birth_date + pd.Timedelta(days=int(round(age_months * 30.4375)))
        formula_value = states[window_index]

        if rng.random() >= 0.03:
            rows.append({
                "patient_id": patient_id,
                "time": event_time,
                "feature_name": "FormulaYN",
                "value": formula_value,
            })

        rows.extend([
            {
                "patient_id": patient_id,
                "time": event_time + pd.Timedelta(days=1),
                "feature_name": "Length",
                "value": maybe_missing(sample_length(birth_weight, pregnancy_week, age_months, rng), 0.08, rng),
            },
            {
                "patient_id": patient_id,
                "time": event_time + pd.Timedelta(days=1),
                "feature_name": "% Weight Gain",
                "value": maybe_missing(sample_weight_gain(age_months, rng), 0.08, rng),
            },
            {
                "patient_id": patient_id,
                "time": event_time + pd.Timedelta(days=2),
                "feature_name": "vaccinations",
                "value": maybe_missing(float(rng.integers(0, 4)), 0.04, rng),
            },
            {
                "patient_id": patient_id,
                "time": event_time + pd.Timedelta(days=2),
                "feature_name": "HGB",
                "value": maybe_missing(round(clipped_normal(rng, 11.6, 1.0, 7.5, 15.5), 1), 0.12, rng),
            },
        ])

    meta = {
        "patient_id": patient_id,
        "trajectory": pattern,
        "intended_states": tuple(states),
        "delivery_type": delivery_type,
        "socioeconomic_score": socioeconomic_score,
        "nursing_end_before_1m": nursing_end_before_1m,
    }
    return rows, meta


def assign_deterministic_labels(
    data: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    window_states = compute_window_formula_states(data)
    static_features = static_feature_matrix(data)
    rule_df = metadata_df.set_index("patient_id").join(
        window_states.set_index("patient_id"),
        how="left",
    ).join(
        static_features[[
            "SocioEconomic_score",
            "BirthType_cesarean",
            "BirthType_instrumental",
            "Nursing End Age before 1 month",
        ]],
        how="left",
        rsuffix="_observed",
    )

    f1 = rule_df["formula_0_2m"].eq(1.0)
    f2 = rule_df["formula_2_4m"].eq(1.0)
    f3 = rule_df["formula_4_6m"].eq(1.0)
    breastfeeding_continues = rule_df["Nursing End Age before 1 month"].eq(0.0)
    high_ses = rule_df["SocioEconomic_score"].ge(4)
    top_ses = rule_df["SocioEconomic_score"].eq(5)
    non_spontaneous = (
        rule_df["BirthType_cesarean"].eq(1.0)
        | rule_df["BirthType_instrumental"].eq(1.0)
    )

    rule_1 = f1 & ~f2 & ~f3 & breastfeeding_continues
    rule_2 = ~f1 & ~f2 & ~f3 & breastfeeding_continues & top_ses & non_spontaneous
    rule_3 = f1 & ~f2 & f3 & high_ses & non_spontaneous

    rule_df["rule_1_main_temporal_early_stop"] = rule_1
    rule_df["rule_2_exclusive_bf_ses_birth_type"] = rule_2
    rule_df["rule_3_inconsistent_ses_birth_type"] = rule_3
    rule_columns = [
        "rule_1_main_temporal_early_stop",
        "rule_2_exclusive_bf_ses_birth_type",
        "rule_3_inconsistent_ses_birth_type",
    ]
    rule_df["label"] = rule_df[rule_columns].any(axis=1).astype(int)
    rule_df["rule_overlap_count"] = rule_df[rule_columns].sum(axis=1)
    rule_df["fa_rule"] = "none"
    for rule_column in rule_columns:
        rule_df.loc[rule_df[rule_column], "fa_rule"] = rule_column

    labels_df = rule_df.reset_index()[["patient_id", "label"]].sort_values("patient_id")
    diagnostics_df = rule_df.reset_index().sort_values("patient_id")
    return labels_df.reset_index(drop=True), diagnostics_df.reset_index(drop=True)


def generate_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    rows = []
    metadata = []

    for index in range(N_PATIENTS):
        patient_rows, patient_metadata = make_patient(index, rng)
        rows.extend(patient_rows)
        metadata.append(patient_metadata)

    data = pd.DataFrame(rows, columns=["patient_id", "time", "feature_name", "value"])
    data = data.drop_duplicates()
    data = data.sort_values(["patient_id", "time", "feature_name"]).reset_index(drop=True)
    metadata_df = pd.DataFrame(metadata).sort_values("patient_id").reset_index(drop=True)
    labels_df, diagnostics_df = assign_deterministic_labels(data, metadata_df)
    return data, labels_df, diagnostics_df


def safe_auc(y_true: pd.Series, score: pd.Series) -> tuple[float, float]:
    valid = score.notna()
    if valid.sum() == 0 or y_true.loc[valid].nunique() < 2 or score.loc[valid].nunique() < 2:
        return np.nan, np.nan
    raw_auc = roc_auc_score(y_true.loc[valid], score.loc[valid])
    return float(raw_auc), float(max(raw_auc, 1.0 - raw_auc))


def shortcut_auc_table(
    data: pd.DataFrame,
    labels_df: pd.DataFrame,
) -> pd.DataFrame:
    y = labels_df.set_index("patient_id")["label"]
    static_features = static_feature_matrix(data)
    formula_features = formula_summaries(data)
    leakage_features = visit_and_missingness_summaries(data)
    feature_matrix = static_features.join(formula_features, how="outer").join(
        leakage_features,
        how="outer",
    ).reindex(y.index)

    features_to_check = [
        "FormulaYN_mean",
        "FormulaYN_sum",
        "FormulaYN_count",
        "FormulaYN_max",
        "Formula_ever",
        "FormulaStartAgeMonths",
        "NursingEndAgeMonths",
        "Nursing End Age before 1 month",
        "SocioEconomic_score",
        "BirthType_spontaneous",
        "BirthType_cesarean",
        "BirthType_instrumental",
        "HGB",
        "% Weight Gain",
        "visit_row_count",
        "missing_value_count",
        "missing_value_rate",
    ]

    rows = []
    for feature in features_to_check:
        raw_auc, oriented_auc = safe_auc(y, feature_matrix[feature])
        rows.append({
            "feature": feature,
            "raw_auc": raw_auc,
            "oriented_auc": oriented_auc,
            "non_missing": int(feature_matrix[feature].notna().sum()),
        })
    return pd.DataFrame(rows).sort_values("oriented_auc", ascending=False)


def static_shallow_tree_diagnostic(
    data: pd.DataFrame,
    labels_df: pd.DataFrame,
) -> tuple[float, pd.Series]:
    X = static_feature_matrix(data).reindex(labels_df["patient_id"])
    y = labels_df.set_index("patient_id").loc[X.index, "label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=SEED,
        stratify=y,
    )
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        DecisionTreeClassifier(max_depth=3, min_samples_leaf=50, random_state=SEED),
    )
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probabilities)
    classifier = model.named_steps["decisiontreeclassifier"]
    importances = pd.Series(classifier.feature_importances_, index=X.columns)
    return float(auc), importances.sort_values(ascending=False)


def validate_dataset(
    data: pd.DataFrame,
    labels_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
) -> str:
    print("A. DATASET STRUCTURE")
    print(f"Patients: {data['patient_id'].nunique()}")
    print(f"Rows: {len(data)}")
    print(f"Features: {data['feature_name'].nunique()}")
    print(f"Duplicate rows: {int(data.duplicated().sum())}")
    print(f"Label counts: {labels_df['label'].value_counts().sort_index().to_dict()}")
    print(f"FA prevalence: {labels_df['label'].mean():.4f}")
    birth_anchor_counts = data[data["feature_name"] == "Birth Date Anchor"].groupby("patient_id").size()
    print(f"One Birth Date Anchor per patient: {bool((birth_anchor_counts == 1).all())}")

    print("\nB. FA RULE DISTRIBUTION")
    rule_columns = [
        "rule_1_main_temporal_early_stop",
        "rule_2_exclusive_bf_ses_birth_type",
        "rule_3_inconsistent_ses_birth_type",
    ]
    for rule_column in rule_columns:
        print(f"{rule_column}: {int(diagnostics_df[rule_column].sum())}")
    print(f"Rule overlaps: {diagnostics_df['rule_overlap_count'].value_counts().sort_index().to_dict()}")
    print(f"Total FA: {int(labels_df['label'].sum())}")

    print("\nC. TRAJECTORY DISTRIBUTION")
    trajectory_summary = diagnostics_df.groupby("trajectory").agg(
        patients=("label", "count"),
        fa_cases=("label", "sum"),
        fa_prevalence=("label", "mean"),
    )
    print(trajectory_summary.to_string())

    print("\nD. TEMPORAL PATTERN VISIBILITY FROM SAVED ROWS")
    window_columns = [f"formula_{window_name}" for window_name in WINDOWS]
    print(diagnostics_df.groupby("trajectory")[window_columns].mean().to_string())

    print("\nH. UNIVARIATE SHORTCUT AUC TABLE")
    shortcut_table = shortcut_auc_table(data, labels_df)
    print(shortcut_table.to_string(index=False))

    print("\nI. STATIC SHALLOW-TREE DIAGNOSTIC")
    static_auc, importances = static_shallow_tree_diagnostic(data, labels_df)
    print(f"Static shallow tree ROC-AUC: {static_auc:.4f}")
    print("Top static tree features:")
    print(importances[importances > 0].head(10).to_string())

    print("\nJ. GROUND-TRUTH RECOVERABILITY CHECK")
    recomputed_labels, recomputed_diagnostics = assign_deterministic_labels(data, diagnostics_df[[
        "patient_id",
        "trajectory",
        "intended_states",
        "delivery_type",
        "socioeconomic_score",
        "nursing_end_before_1m",
    ]])
    labels_match = labels_df.sort_values("patient_id").reset_index(drop=True).equals(
        recomputed_labels.sort_values("patient_id").reset_index(drop=True)
    )
    print(f"Observable deterministic rules exactly recover labels: {labels_match}")
    print(f"Recovered FA count: {int(recomputed_labels['label'].sum())}")

    print("\nK. MISSINGNESS / VISIT-COUNT LEAKAGE CHECK")
    leakage = shortcut_table[
        shortcut_table["feature"].isin(["visit_row_count", "missing_value_count", "missing_value_rate"])
    ]
    print(leakage.to_string(index=False))

    print("\nVALIDATION ASSERTIONS")
    assert data["patient_id"].nunique() == N_PATIENTS
    assert len(labels_df) == N_PATIENTS
    assert set(data["feature_name"].unique()) == set(FEATURES)
    assert data["feature_name"].nunique() == len(FEATURES)
    assert labels_df["patient_id"].is_unique
    assert labels_df["label"].isin([0, 1]).all()
    assert set(data["patient_id"].unique()) == set(labels_df["patient_id"].unique())
    assert data.duplicated().sum() == 0
    assert (birth_anchor_counts == 1).all()
    assert labels_match
    data_with_birth = data.merge(
        data[data["feature_name"] == "Birth Date Anchor"][["patient_id", "time"]].rename(columns={"time": "birth_time"}),
        on="patient_id",
        how="left",
    )
    assert not (data_with_birth["time"] < data_with_birth["birth_time"]).any()
    birth_type = data[data["feature_name"].isin([
        "BirthType_spontaneous",
        "BirthType_cesarean",
        "BirthType_instrumental",
    ])].pivot(index="patient_id", columns="feature_name", values="value")
    assert (birth_type.sum(axis=1) == 1).all()
    print("All validation assertions passed.")
    max_univariate_auc = shortcut_table["oriented_auc"].max()
    if max_univariate_auc >= 0.80 or static_auc >= 0.85:
        return "NEEDS GENERATOR ADJUSTMENT"
    return "READY FOR TEMPORALTREE TRAINING"


def main() -> None:
    data, labels_df, diagnostics_df = generate_dataset()
    data.to_pickle(DATA_OUTPUT)
    labels_df.to_pickle(LABELS_OUTPUT)

    reloaded_data = pd.read_pickle(DATA_OUTPUT)
    reloaded_labels = pd.read_pickle(LABELS_OUTPUT)
    assert reloaded_data.equals(data)
    assert reloaded_labels.equals(labels_df)

    verdict = validate_dataset(data, labels_df, diagnostics_df)

    print("\nA. NEW FILES CREATED")
    print(f"Data: {DATA_OUTPUT.resolve()}")
    print(f"Labels: {LABELS_OUTPUT.resolve()}")
    print("\nM. VERDICT")
    print(verdict)


if __name__ == "__main__":
    main()
