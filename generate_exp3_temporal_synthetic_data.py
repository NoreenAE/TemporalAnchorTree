from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier


SEED = 20260808
N_PATIENTS = 5000
N_FA_POSITIVE = 250
OUTPUT_DIR = Path(".")
DATA_OUTPUT = OUTPUT_DIR / "synthetic_exp3_temporal_5000.pkl"
LABELS_OUTPUT = OUTPUT_DIR / "synthetic_exp3_temporal_labels_5000.pkl"

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
    "early_formula_then_bf": {
        "states": [1.0, 0.0, 0.0],
        "probability": 0.01,
        "temporal_risk_score": 4.0,
    },
    "late_formula": {
        "states": [0.0, 0.0, 1.0],
        "probability": 0.35,
        "temporal_risk_score": 1.2,
    },
    "formula_early_then_stop_late": {
        "states": [1.0, 1.0, 0.0],
        "probability": 0.20,
        "temporal_risk_score": 2.95,
    },
    "formula_later_continue": {
        "states": [0.0, 1.0, 1.0],
        "probability": 0.25,
        "temporal_risk_score": 0.8,
    },
    "continuous_formula": {
        "states": [1.0, 1.0, 1.0],
        "probability": 0.10,
        "temporal_risk_score": 0.5,
    },
    "exclusive_bf": {
        "states": [0.0, 0.0, 0.0],
        "probability": 0.01,
        "temporal_risk_score": 3.0,
    },
    "inconsistent": {
        "states": [1.0, 0.0, 1.0],
        "probability": 0.08,
        "temporal_risk_score": 2.0,
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


def make_patient_id(index: int) -> str:
    return f"EXP3_{index:05d}"


def sample_birth_date(rng: np.random.Generator) -> pd.Timestamp:
    offset_days = int(rng.integers(0, 365 * 4))
    return pd.Timestamp("2016-01-01") + pd.Timedelta(days=offset_days)


def sample_pregnancy_week(rng: np.random.Generator) -> float:
    return round(clipped_normal(rng, 39.0, 1.5, 32.0, 42.0), 1)


def sample_birth_weight(pregnancy_week: float, rng: np.random.Generator) -> float:
    if pregnancy_week < 37:
        mean = 2.35 + 0.13 * (pregnancy_week - 34)
    else:
        mean = 3.25 + 0.05 * (pregnancy_week - 39)
    return round(clipped_normal(rng, mean, 0.38, 1.2, 4.8), 2)


def sample_apgar(
    pregnancy_week: float,
    delivery_type: str,
    birth_weight: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    prematurity_penalty = max(0.0, 37.0 - pregnancy_week) * 0.25
    low_weight_penalty = max(0.0, 2.5 - birth_weight) * 0.55
    delivery_penalty = 0.45 if delivery_type in {"cesarean", "instrumental"} else 0.0
    apgar1 = int(round(clipped_normal(
        rng,
        8.1 - prematurity_penalty - low_weight_penalty - delivery_penalty,
        1.0,
        1,
        10,
    )))
    apgar5 = int(round(clipped_normal(
        rng,
        max(apgar1, 8.9 - 0.45 * prematurity_penalty - 0.35 * low_weight_penalty),
        0.55,
        apgar1,
        10,
    )))
    return float(apgar1), float(apgar5)


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


def sample_siblings_and_birth_order(rng: np.random.Generator) -> tuple[int, int]:
    older_siblings = int(rng.choice([0, 1, 2, 3, 4], p=[0.42, 0.32, 0.16, 0.08, 0.02]))
    return older_siblings, older_siblings + 1


def sample_pattern(rng: np.random.Generator) -> str:
    names = list(PATTERNS)
    probabilities = [PATTERNS[name]["probability"] for name in names]
    return str(rng.choice(names, p=probabilities))


def sample_window_age_months(
    window_name: str,
    rng: np.random.Generator,
) -> float:
    lower, upper = WINDOWS[window_name]
    return float(np.clip(rng.uniform(lower, upper) + rng.normal(0.0, 0.06), lower, upper))


def formula_start_age_months(
    states: list[float],
    rng: np.random.Generator,
) -> float:
    formula_indices = [index for index, state in enumerate(states) if state == 1.0]
    if not formula_indices:
        return np.nan if rng.random() < 0.78 else round(float(rng.uniform(4.0, 8.5)), 2)

    first_index = formula_indices[0]
    lower, upper = list(WINDOWS.values())[first_index]
    if rng.random() < 0.72:
        return np.nan
    noisy_start = rng.uniform(lower, upper) + rng.normal(0.0, 1.15)
    return round(float(np.clip(noisy_start, 0.0, 8.0)), 2)


def nursing_end_age_months(
    states: list[float],
    rng: np.random.Generator,
) -> float:
    if rng.random() < 0.84:
        return np.nan
    if states[0] == 1.0 and any(state == 1.0 for state in states[1:]):
        return round(float(rng.uniform(0.5, 8.0)), 2)
    if states[0] == 0.0 and any(state == 1.0 for state in states[1:]):
        return round(float(rng.uniform(2.0, 9.5)), 2)
    return round(float(rng.uniform(4.0, 12.0)), 2)


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


def maybe_missing(value, probability: float, rng: np.random.Generator):
    return np.nan if rng.random() < probability else value


def calculate_risk_score(
    pattern: str,
    pregnancy_week: float,
    birth_weight: float,
    socioeconomic_score: int,
) -> float:
    return PATTERNS[pattern]["temporal_risk_score"]


def make_patient(
    index: int,
    rng: np.random.Generator,
) -> tuple[list[dict], dict]:
    patient_id = make_patient_id(index)
    birth_date = sample_birth_date(rng)
    pattern = sample_pattern(rng)
    states = list(PATTERNS[pattern]["states"])

    pregnancy_week = sample_pregnancy_week(rng)
    delivery_type = sample_delivery_type(pregnancy_week, rng)
    birth_weight = sample_birth_weight(pregnancy_week, rng)
    apgar1, apgar5 = sample_apgar(pregnancy_week, delivery_type, birth_weight, rng)
    older_siblings, birth_order = sample_siblings_and_birth_order(rng)
    socioeconomic_score = int(rng.choice([1, 2, 3, 4, 5], p=[0.13, 0.22, 0.31, 0.23, 0.11]))
    mother_education = int(rng.choice([1, 2, 3, 4, 5], p=[0.10, 0.22, 0.35, 0.23, 0.10]))
    is_jew = int(rng.random() < 0.56)
    foods_start_age = round(clipped_normal(rng, 5.4, 0.8, 3.5, 8.5), 2)
    formula_start = formula_start_age_months(states, rng)
    nursing_end = nursing_end_age_months(states, rng)
    nursing_end_before_1m = (
        0.0
        if pd.isna(nursing_end)
        else float(nursing_end < 1.0)
    )
    risk_score = calculate_risk_score(
        pattern=pattern,
        pregnancy_week=pregnancy_week,
        birth_weight=birth_weight,
        socioeconomic_score=socioeconomic_score,
    )

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
        "states": tuple(states),
        "temporal_risk_score": PATTERNS[pattern]["temporal_risk_score"],
        "risk_score": risk_score,
        "formula_mean_truth": float(np.mean(states)),
        "formula_sum_truth": float(np.sum(states)),
        "formula_ever_truth": float(np.max(states)),
    }
    return rows, meta


def assign_deterministic_labels(metadata_df: pd.DataFrame) -> pd.DataFrame:
    ranked = metadata_df.sort_values(
        ["risk_score", "patient_id"],
        ascending=[False, True],
    ).reset_index(drop=True)
    positive_patients = set(ranked.head(N_FA_POSITIVE)["patient_id"])
    labels_df = metadata_df[["patient_id"]].copy()
    labels_df["label"] = labels_df["patient_id"].isin(positive_patients).astype(int)
    return labels_df.sort_values("patient_id").reset_index(drop=True)


def generate_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    rows = []
    metadata = []

    for index in range(N_PATIENTS):
        patient_rows, meta = make_patient(index, rng)
        rows.extend(patient_rows)
        metadata.append(meta)

    data = pd.DataFrame(rows, columns=["patient_id", "time", "feature_name", "value"])
    data = data.drop_duplicates()
    metadata_df = pd.DataFrame(metadata)
    labels_df = assign_deterministic_labels(metadata_df)

    data = data.sort_values(["patient_id", "time", "feature_name"]).reset_index(drop=True)
    labels_df = labels_df.sort_values("patient_id").reset_index(drop=True)
    metadata_df = metadata_df.sort_values("patient_id").reset_index(drop=True)
    return data, labels_df, metadata_df


def patient_formula_summaries(data: pd.DataFrame) -> pd.DataFrame:
    formula_rows = data[data["feature_name"] == "FormulaYN"][["patient_id", "value"]].copy()
    formula_rows["value"] = pd.to_numeric(formula_rows["value"], errors="coerce")
    return formula_rows.groupby("patient_id")["value"].agg(
        formula_mean="mean",
        formula_sum="sum",
        formula_count="count",
        formula_ever="max",
    ).reset_index()


def static_feature_matrix(data: pd.DataFrame) -> pd.DataFrame:
    static_rows = data[~data["feature_name"].isin(["FormulaYN", "Birth Date Anchor"])].copy()
    static_rows["value"] = pd.to_numeric(static_rows["value"], errors="coerce")
    return static_rows.pivot_table(
        index="patient_id",
        columns="feature_name",
        values="value",
        aggfunc="mean",
    )


def static_model_auc(data: pd.DataFrame, labels_df: pd.DataFrame) -> tuple[float, pd.Series]:
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
        DecisionTreeClassifier(max_depth=3, random_state=SEED),
    )
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    classifier = model.named_steps["decisiontreeclassifier"]
    importances = pd.Series(classifier.feature_importances_, index=X.columns).sort_values(ascending=False)
    return float(roc_auc_score(y_test, probabilities)), importances


def validate_dataset(
    data: pd.DataFrame,
    labels_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> None:
    labels_with_meta = labels_df.merge(metadata_df, on="patient_id", how="left")
    formula_summaries = patient_formula_summaries(data)
    shortcut_df = labels_with_meta.merge(formula_summaries, on="patient_id", how="left")

    feature_list = sorted(data["feature_name"].unique().tolist())
    birth_rows = data[data["feature_name"] == "Birth Date Anchor"][["patient_id", "time"]].rename(columns={"time": "birth_time"})
    data_with_birth = data.merge(birth_rows, on="patient_id", how="left")
    duplicate_count = int(data.duplicated().sum())
    missing_summary = data.groupby("feature_name")["value"].apply(lambda values: int(values.isna().sum()))

    print("====================================================")
    print("EXPERIMENT 3 TEMPORAL SYNTHETIC DATA VALIDATION")
    print("====================================================")
    print(f"Data shape: {data.shape}")
    print(f"Labels shape: {labels_df.shape}")
    print(f"Number of patients: {data['patient_id'].nunique()}")
    print(f"Number of features: {data['feature_name'].nunique()}")
    print("Feature list:")
    for feature in feature_list:
        print(f"  - {feature}")

    label_counts = labels_df["label"].value_counts().sort_index()
    print("\nFA label counts:")
    print(label_counts.to_string())
    print(f"Overall FA prevalence: {labels_df['label'].mean():.4f}")

    print("\nLatent trajectory distribution, risk score, and FA prevalence:")
    trajectory_summary = labels_with_meta.groupby("trajectory").agg(
        patients=("label", "count"),
        mean_risk_score=("risk_score", "mean"),
        min_risk_score=("risk_score", "min"),
        max_risk_score=("risk_score", "max"),
        fa_cases=("label", "sum"),
        fa_prevalence=("label", "mean"),
    )
    print(trajectory_summary.to_string())

    print("\nFormulaYN full-timeline summaries by trajectory:")
    print(shortcut_df.groupby("trajectory")[["formula_mean", "formula_sum", "formula_count", "formula_ever"]].mean().to_string())

    print("\nCritical matched-amount contrasts:")
    contrast_1 = shortcut_df[shortcut_df["trajectory"].isin(["early_formula_then_bf", "late_formula"])]
    contrast_2 = shortcut_df[shortcut_df["trajectory"].isin(["formula_early_then_stop_late", "formula_later_continue"])]
    print(contrast_1.groupby("trajectory")[["risk_score", "label", "formula_mean", "formula_sum", "formula_count", "formula_ever"]].mean().to_string())
    print(contrast_2.groupby("trajectory")[["risk_score", "label", "formula_mean", "formula_sum", "formula_count", "formula_ever"]].mean().to_string())

    print("\nShortcut diagnostic ROC-AUC for full-timeline FormulaYN summaries:")
    for feature in ["formula_mean", "formula_sum", "formula_count", "formula_ever"]:
        valid_rows = shortcut_df[feature].notna()
        auc = roc_auc_score(shortcut_df.loc[valid_rows, "label"], shortcut_df.loc[valid_rows, feature])
        print(f"{feature}: raw_auc={auc:.4f}, oriented_auc={max(auc, 1 - auc):.4f}")

    static_auc, importances = static_model_auc(data, labels_df)
    print("\nStatic non-temporal shallow DecisionTree diagnostic:")
    print(f"ROC-AUC: {static_auc:.4f}")
    print("Feature importances:")
    print(importances[importances > 0].to_string())

    print("\nDuplicate count:", duplicate_count)
    print("\nMissingness summary by feature:")
    print(missing_summary[missing_summary > 0].to_string())

    assert data["patient_id"].nunique() == N_PATIENTS
    assert len(labels_df) == N_PATIENTS
    assert int(labels_df["label"].sum()) == N_FA_POSITIVE
    assert int((labels_df["label"] == 0).sum()) == N_PATIENTS - N_FA_POSITIVE
    assert labels_df["label"].mean() == N_FA_POSITIVE / N_PATIENTS
    assert data["feature_name"].nunique() == len(FEATURES)
    assert set(feature_list) == set(FEATURES)
    assert (data["feature_name"] == "Birth Date Anchor").sum() == N_PATIENTS
    assert labels_df["patient_id"].is_unique
    assert labels_df["label"].isin([0, 1]).all()
    assert set(data["patient_id"].unique()) == set(labels_df["patient_id"].unique())
    assert duplicate_count == 0
    assert not (data_with_birth["time"] < data_with_birth["birth_time"]).any()

    birth_type = data[data["feature_name"].isin([
        "BirthType_spontaneous",
        "BirthType_cesarean",
        "BirthType_instrumental",
    ])].pivot_table(index="patient_id", columns="feature_name", values="value", aggfunc="first")
    assert (birth_type.sum(axis=1) == 1).all()

    order_rows = data[data["feature_name"].isin(["BirthOrder", "Num older siblings"])].pivot_table(
        index="patient_id",
        columns="feature_name",
        values="value",
        aggfunc="first",
    )
    assert (order_rows["BirthOrder"] == order_rows["Num older siblings"] + 1).all()

    apgar_rows = data[data["feature_name"].isin(["ApgarOneMinute", "ApgarFiveMinutes"])].pivot_table(
        index="patient_id",
        columns="feature_name",
        values="value",
        aggfunc="first",
    )
    assert (apgar_rows["ApgarFiveMinutes"] >= apgar_rows["ApgarOneMinute"]).all()

    reloaded_data = pd.read_pickle(DATA_OUTPUT)
    reloaded_labels = pd.read_pickle(LABELS_OUTPUT)
    assert reloaded_data.equals(data)
    assert reloaded_labels.equals(labels_df)

    print("\nOutput validation passed.")


def main() -> None:
    data, labels_df, metadata_df = generate_dataset()
    data.to_pickle(DATA_OUTPUT)
    labels_df.to_pickle(LABELS_OUTPUT)
    validate_dataset(data, labels_df, metadata_df)
    print("\nSaved data to:", DATA_OUTPUT.resolve())
    print("Saved labels to:", LABELS_OUTPUT.resolve())
    print("\nEmbedded deterministic temporal ground truth:")
    print("- The FA label is assigned to the top 250 patients by deterministic risk_score.")
    print("- risk_score is dominated by the temporal FormulaYN order across 0-2m, 2-4m, and 4-6m.")
    print("- Background variables are distractors and do not contribute to the FA label.")
    print("- [1,0,0] and [0,0,1] have the same full-timeline formula amount but different risk.")
    print("- [1,1,0] and [0,1,1] have the same full-timeline formula amount but different risk.")


if __name__ == "__main__":
    main()
