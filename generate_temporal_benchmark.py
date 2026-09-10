"""Generate the shared Phase-1 temporal benchmark and Food Allergy labels."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import tempfile
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit
from scipy.stats import chisquare


MASTER_SEED = 20260823
N_PATIENTS = 1000
OUTPUT_DIR = Path("data/temporal_benchmark")
NOW_AGE_MONTHS_MIN = 18
NOW_AGE_MONTHS_MAX = 24
TARGET_FA_PREVALENCE = 0.10
N_EXPERIMENT_3 = 208_951
VALIDATED_V1_PROFILE = "validated_v1"
PUBLICATION_V2_PROFILE = "publication_v2"
GENERATION_PROFILES = (VALIDATED_V1_PROFILE, PUBLICATION_V2_PROFILE)
PUBLICATION_V2_OUTPUT_DIR = Path("data/temporal_benchmark_v2")

ROUTINE_GROWTH_MONTHS = (0, 1, 2, 4, 6, 9, 12, 18, 24)
LENGTH_VELOCITY_RANGES = (
    (0.0, 3.0, 3.0, 3.5),
    (3.0, 6.0, 2.0, 2.5),
    (6.0, 12.0, 1.0, 1.5),
    (12.0, 24.0, 0.8, 1.0),
)
WEIGHT_VELOCITY_RANGES = (
    (0.0, 3.0, 0.7, 1.0),
    (3.0, 6.0, 0.5, 0.7),
    (6.0, 12.0, 0.25, 0.4),
    (12.0, 24.0, 0.15, 0.25),
)
FOOD_EXPOSURE_BASE_PROBABILITY = 0.005
FOOD_EXPOSURE_MAX_PROBABILITY_RANGE = (0.85, 0.95)
FOOD_EXPOSURE_MIDPOINT_DAYS_RANGE = (150, 210)
FOOD_EXPOSURE_LOGISTIC_SCALE_DAYS = 20.0

FEEDING_TRAJECTORY_COUNTS = {
    "2.0": 41_120,
    "2.1": 19_560,
    "2.2": 25_913,
    "2.3": 6_659,
    "2.4": 24_817,
    "3": 40_682,
    "4": 5_427,
    "5.1": 21_444,
    "5.2": 10_417,
    "5.3": 9_205,
    "6": 3_707,
}

FEEDING_TRAJECTORY_OR = {
    "2.0": 1.00,
    "2.1": 0.51,
    "2.2": 0.49,
    "2.3": 0.54,
    "2.4": 0.47,
    "3": 2.95,
    "4": 3.10,
    "5.1": 1.13,
    "5.2": 1.68,
    "5.3": 1.60,
    "6": 1.98,
}

GROUP_ORDER = tuple(FEEDING_TRAJECTORY_COUNTS)
FEEDING_TRAJECTORY_PROBS = {
    group: count / N_EXPERIMENT_3
    for group, count in FEEDING_TRAJECTORY_COUNTS.items()
}

FEEDING_STATES = {
    "F": (1.0, 0.0),
    "B": (0.0, 1.0),
    "M": (1.0, 1.0),
}

FEEDING_TEMPLATES = {
    "2.0": ("F", "F", "F", "F"),
    "2.1": ("M", "F", "F", "F"),
    "2.2": ("M", "M", "F", "F"),
    "2.3": ("M", "M", "M", "F"),
    "2.4": ("M", "M", "M", "M"),
    "3": ("B", "B", "B", "B"),
    "4": ("F", "B", "B", "B"),
    "5.1": ("B", "F", "F", "F"),
    "5.2": ("B", "B", "F", "F"),
    "5.3": ("B", "B", "B", "F"),
}

INCONSISTENT_FEEDING_TEMPLATES = (
    ("M", "F", "M", "F"),
    ("F", "B", "M", "F"),
    ("B", "M", "B", "M"),
    ("M", "B", "F", "M"),
)

# Model-visible signatures at: pre-21d, 21d, 2m, 4m, and 6m.  These
# signatures make adjacent transition intervals observable without changing
# the patient's single latent trajectory assignment.
REGULAR_BOUNDARY_SIGNATURES = {
    "2.0": ("F", "F", "F", "F", "F"),
    "2.1": ("M", "F", "F", "F", "F"),
    "2.2": ("M", "M", "F", "F", "F"),
    "2.3": ("M", "M", "M", "F", "F"),
    "2.4": ("M", "M", "M", "M", "M"),
    "3": ("B", "B", "B", "B", "B"),
    "4": ("F", "B", "B", "B", "B"),
    "5.1": ("B", "B", "F", "F", "F"),
    "5.2": ("B", "B", "B", "F", "F"),
    "5.3": ("B", "B", "B", "B", "F"),
}

GROUP_6_BOUNDARY_SIGNATURES = {
    (template[0], template[1], template[2], template[3], template[3])
    for template in INCONSISTENT_FEEDING_TEMPLATES
}

GI_DISTRACTOR_RATES = {
    "VomitingYN": 0.08,
    "DiarrheaYN": 0.10,
    "GISymptomYN": 0.12,
    "FoodExposureYN": 0.25,
}

LEAKAGE_FEATURES = {
    "feeding_group",
    "trajectory_id",
    "fa_probability",
    "gi_probability",
    "fa_risk_score",
    "gi_risk_score",
    "risk_score",
    "feeding_or",
    "transition_time",
    "formulastartagemonths",
    "nursingendagemonths",
    "label",
}

LONGITUDINAL_COLUMNS = ["patient_id", "time", "feature_name", "value"]
LABEL_COLUMNS = ["patient_id", "label"]


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for one generated benchmark cohort."""

    n_patients: int = N_PATIENTS
    master_seed: int = MASTER_SEED
    output_dir: Path = OUTPUT_DIR
    now_age_months_min: int = NOW_AGE_MONTHS_MIN
    now_age_months_max: int = NOW_AGE_MONTHS_MAX
    target_fa_prevalence: float = TARGET_FA_PREVALENCE
    continuous_missingness_rate: float = 0.05
    clinical_visit_interval_days: int = 14
    generation_profile: str = VALIDATED_V1_PROFILE
    overwrite: bool = False


@dataclass
class BenchmarkResult:
    """In-memory benchmark artifacts and validation metadata."""

    data: pd.DataFrame
    fa_labels: pd.DataFrame
    ground_truth: pd.DataFrame
    metadata: dict


def validate_configuration(config: BenchmarkConfig) -> None:
    """Fail clearly when scientific or generation configuration is invalid."""
    if config.n_patients < 1:
        raise ValueError("n_patients must be positive.")
    if config.now_age_months_min < 12:
        raise ValueError(
            "now_age_months_min must exceed 12 months so the two six-month "
            "anchor regions are strictly separated."
        )
    if config.now_age_months_max < config.now_age_months_min:
        raise ValueError("now_age_months_max must be >= now_age_months_min.")
    if not 0 < config.target_fa_prevalence < 1:
        raise ValueError(
            "target_fa_prevalence must lie strictly between 0 and 1.")
    if not 0 <= config.continuous_missingness_rate < 1:
        raise ValueError("continuous_missingness_rate must lie in [0, 1).")
    if config.clinical_visit_interval_days < 1:
        raise ValueError("clinical_visit_interval_days must be positive.")
    if config.generation_profile not in GENERATION_PROFILES:
        raise ValueError(
            "generation_profile must be one of: "
            + ", ".join(GENERATION_PROFILES)
        )
    if (
        config.generation_profile == PUBLICATION_V2_PROFILE
        and config.output_dir.resolve() == OUTPUT_DIR.resolve()
    ):
        raise ValueError(
            "publication_v2 artifacts require a versioned output directory."
        )
    if sum(FEEDING_TRAJECTORY_COUNTS.values()) != N_EXPERIMENT_3:
        raise ValueError("Experiment 3 feeding counts do not sum to 208951.")
    if set(FEEDING_TRAJECTORY_COUNTS) != set(FEEDING_TRAJECTORY_OR):
        raise ValueError("Feeding count and OR groups must match exactly.")
    if not math.isclose(
        sum(FEEDING_TRAJECTORY_PROBS.values()),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Derived feeding probabilities do not sum to one.")
    _validate_inconsistent_templates()


def _validate_inconsistent_templates() -> None:
    defined = set(FEEDING_TEMPLATES.values())
    for template in INCONSISTENT_FEEDING_TEMPLATES:
        if len(template) != 4 or any(
            state not in FEEDING_STATES for state in template
        ):
            raise ValueError(f"Invalid inconsistent template: {template!r}.")
        transitions = sum(left != right for left,
                          right in zip(template, template[1:]))
        if transitions < 2:
            raise ValueError(
                "Inconsistent template must contain at least two transitions: "
                f"{template!r}."
            )
        if template in defined:
            raise ValueError(
                "Inconsistent template duplicates a defined feeding group: "
                f"{template!r}."
            )
    if len(set(INCONSISTENT_FEEDING_TEMPLATES)) != len(
        INCONSISTENT_FEEDING_TEMPLATES
    ):
        raise ValueError("Inconsistent feeding templates must be unique.")
    regular_signatures = set(REGULAR_BOUNDARY_SIGNATURES.values())
    if len(regular_signatures) != len(REGULAR_BOUNDARY_SIGNATURES):
        raise ValueError(
            "Regular feeding signatures must be mutually exclusive."
        )
    if regular_signatures & GROUP_6_BOUNDARY_SIGNATURES:
        raise ValueError(
            "Group 6 signatures must be exclusive from regular groups."
        )


def _make_rng_streams(
    master_seed: int,
    generation_profile: str = VALIDATED_V1_PROFILE,
):
    names = [
        "cohort",
        "trajectory",
        "feeding_observations",
        "distractors",
        "missingness",
        "fa_labels",
    ]
    if generation_profile == PUBLICATION_V2_PROFILE:
        # Spawn key 6 is reserved for the established dedicated GI-label RNG.
        names.extend(["reserved_gi_labels", "food_exposure", "growth"])
    child_sequences = np.random.SeedSequence(master_seed).spawn(len(names))
    rngs = {
        name: np.random.default_rng(sequence)
        for name, sequence in zip(names, child_sequences)
        if name != "reserved_gi_labels"
    }
    stream_metadata = {
        name: {"spawn_key": list(sequence.spawn_key)}
        for name, sequence in zip(names, child_sequences)
        if name != "reserved_gi_labels"
    }
    return rngs, stream_metadata


def _make_patient_id(index: int) -> str:
    return f"TB_{index:05d}"


def generate_cohort(
    config: BenchmarkConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate patient identifiers, birth dates, and patient-specific NOW."""
    birth_origin = pd.Timestamp("2018-01-01")
    birth_offsets = rng.integers(0, 365 * 3, size=config.n_patients)
    records = []
    if config.generation_profile == VALIDATED_V1_PROFILE:
        now_ages = rng.integers(
            config.now_age_months_min,
            config.now_age_months_max + 1,
            size=config.n_patients,
        )
    else:
        now_ages = None

    for index, birth_offset in enumerate(birth_offsets):
        birth_date = birth_origin + pd.Timedelta(days=int(birth_offset))
        if now_ages is not None:
            now = birth_date + pd.DateOffset(months=int(now_ages[index]))
            age_fields = {"now_age_months": int(now_ages[index])}
        else:
            earliest_now = birth_date + pd.DateOffset(
                months=config.now_age_months_min
            )
            latest_now = birth_date + pd.DateOffset(
                months=config.now_age_months_max
            )
            available_days = (latest_now - earliest_now).days
            day_offset = int(rng.integers(0, available_days + 1))
            now = earliest_now + pd.Timedelta(days=day_offset)
            now_age_days = int((now - birth_date).days)
            age_fields = {
                "now_age_days": now_age_days,
                "now_age_months": now_age_days / 30.4375,
            }
        records.append({
            "patient_id": _make_patient_id(index),
            "birth_date": birth_date,
            "now": now,
            **age_fields,
        })
    return pd.DataFrame.from_records(records)


def sample_feeding_groups(
    n_patients: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample latent trajectories from the empirical probabilities."""
    probabilities = [
        FEEDING_TRAJECTORY_PROBS[group]
        for group in GROUP_ORDER
    ]
    return rng.choice(GROUP_ORDER, size=n_patients, p=probabilities)


def _sample_transition_time(
    lower_exclusive: pd.Timestamp,
    upper_inclusive: pd.Timestamp,
    rng: np.random.Generator,
) -> pd.Timestamp:
    first_allowed = lower_exclusive + pd.Timedelta(days=1)
    available_days = (upper_inclusive - first_allowed).days
    if available_days < 0:
        raise ValueError(
            "Transition interval contains no valid whole-day timestamps.")
    day_offset = int(rng.integers(0, available_days + 1))
    return first_allowed + pd.Timedelta(days=day_offset)


def build_trajectory_spec(
    group: str,
    birth_date: pd.Timestamp,
    rng: np.random.Generator,
) -> dict:
    """Resolve a trajectory into an initial state and timed transitions."""
    day_21 = birth_date + pd.Timedelta(days=21)
    month_2 = birth_date + pd.DateOffset(months=2)
    month_4 = birth_date + pd.DateOffset(months=4)

    if group == "6":
        template_index = int(rng.integers(
            0, len(INCONSISTENT_FEEDING_TEMPLATES)))
        template = INCONSISTENT_FEEDING_TEMPLATES[template_index]
        boundaries = (day_21, month_2, month_4)
        transitions = [
            (boundary, next_state)
            for previous_state, next_state, boundary in zip(
                template,
                template[1:],
                boundaries,
            )
            if previous_state != next_state
        ]
        template_name = f"inconsistent_{template_index + 1}"
    else:
        if group not in FEEDING_TEMPLATES:
            raise ValueError(f"Unknown feeding group: {group!r}.")
        template = FEEDING_TEMPLATES[group]
        template_name = ",".join(template)
        transitions = []
        if group == "2.1":
            transitions = [(day_21, "F")]
        elif group == "2.2":
            transitions = [(
                _sample_transition_time(day_21, month_2, rng),
                "F",
            )]
        elif group == "2.3":
            transitions = [(
                _sample_transition_time(month_2, month_4, rng),
                "F",
            )]
        elif group == "4":
            transitions = [(day_21, "B")]
        elif group == "5.1":
            transitions = [(
                _sample_transition_time(day_21, month_2, rng),
                "F",
            )]
        elif group == "5.2":
            transitions = [(
                _sample_transition_time(month_2, month_4, rng),
                "F",
            )]
        elif group == "5.3":
            transitions = [(
                _sample_transition_time(
                    month_4,
                    birth_date + pd.DateOffset(months=6),
                    rng,
                ),
                "F",
            )]

    return {
        "feeding_template": template_name,
        "conceptual_states": tuple(template),
        "initial_state": template[0],
        "transition_times": tuple(time for time, _ in transitions),
        "transition_states": tuple(state for _, state in transitions),
    }


def feeding_state_at(
    timestamp: pd.Timestamp,
    initial_state: str,
    transition_times: Sequence[pd.Timestamp],
    transition_states: Sequence[str],
) -> str:
    """Return the active feeding state at one observation timestamp."""
    state = initial_state
    for transition_time, next_state in zip(
        transition_times,
        transition_states,
    ):
        if timestamp >= transition_time:
            state = next_state
        else:
            break
    return state


def _jittered_timestamp(
    timestamp: pd.Timestamp,
    rng: np.random.Generator,
    jitter_days: int,
) -> pd.Timestamp:
    return timestamp + pd.Timedelta(
        days=int(rng.integers(-jitter_days, jitter_days + 1))
    )


def _early_feeding_assessment_times(
    birth_date: pd.Timestamp,
    transition_times: Sequence[pd.Timestamp],
    rng: np.random.Generator,
) -> List[pd.Timestamp]:
    six_months = birth_date + pd.DateOffset(months=6)
    planned = [
        birth_date + pd.Timedelta(days=day)
        for day in (1, 6, 12, 18, 24, 30, 45)
    ]
    planned.extend(
        [
            birth_date + pd.DateOffset(months=2) - pd.Timedelta(days=3),
            birth_date + pd.DateOffset(months=2) + pd.Timedelta(days=3),
            birth_date + pd.DateOffset(months=3),
            birth_date + pd.DateOffset(months=4) - pd.Timedelta(days=3),
            birth_date + pd.DateOffset(months=4) + pd.Timedelta(days=3),
            birth_date + pd.DateOffset(months=5),
            six_months - pd.Timedelta(days=2),
        ]
    )
    jittered = [_jittered_timestamp(time, rng, 2) for time in planned]
    for transition_time in transition_times:
        jittered.extend(
            [
                transition_time - pd.Timedelta(days=1),
                transition_time + pd.Timedelta(days=1),
            ]
        )
    mandatory_boundaries = {
        birth_date + pd.Timedelta(days=21),
        birth_date + pd.DateOffset(months=2),
        birth_date + pd.DateOffset(months=4),
        six_months,
    }
    valid = mandatory_boundaries | {
        max(birth_date, min(time, six_months))
        for time in jittered
    }
    return sorted(valid)


def generate_feeding_rows(
    patient_id: str,
    birth_date: pd.Timestamp,
    now: pd.Timestamp,
    trajectory_spec: Mapping,
    rng: np.random.Generator,
) -> List[tuple]:
    """Generate complete early feeding and independent later distractors."""
    rows = []
    for timestamp in _early_feeding_assessment_times(
        birth_date,
        trajectory_spec["transition_times"],
        rng,
    ):
        state = feeding_state_at(
            timestamp,
            trajectory_spec["initial_state"],
            trajectory_spec["transition_times"],
            trajectory_spec["transition_states"],
        )
        formula, breast_milk = FEEDING_STATES[state]
        rows.extend(
            [
                (patient_id, timestamp, "FormulaYN", formula),
                (patient_id, timestamp, "BreastMilkYN", breast_milk),
            ]
        )

    six_months = birth_date + pd.DateOffset(months=6)
    for age_months in range(8, 25, 2):
        timestamp = _jittered_timestamp(
            birth_date + pd.DateOffset(months=age_months),
            rng,
            4,
        )
        if timestamp <= six_months or timestamp > now:
            continue
        state = str(rng.choice(tuple(FEEDING_STATES)))
        formula, breast_milk = FEEDING_STATES[state]
        rows.extend(
            [
                (patient_id, timestamp, "FormulaYN", formula),
                (patient_id, timestamp, "BreastMilkYN", breast_milk),
            ]
        )
    return rows


def _maybe_missing(
    value: float,
    missingness_rate: float,
    rng: np.random.Generator,
):
    return np.nan if rng.random() < missingness_rate else value


def _piecewise_growth_value(
    age_months: float,
    baseline: float,
    velocities: Sequence[float],
    ranges: Sequence[tuple],
) -> float:
    """Integrate one patient's positive piecewise growth velocities."""
    value = baseline
    for velocity, (start, end, _, _) in zip(velocities, ranges):
        months_in_segment = max(0.0, min(age_months, end) - start)
        value += velocity * months_in_segment
    return float(value)


def _publication_growth_rows(
    patient_id: str,
    birth_date: pd.Timestamp,
    now: pd.Timestamp,
    birth_weight: float,
    missingness_rate: float,
    growth_rng: np.random.Generator,
    missingness_rng: np.random.Generator,
) -> List[tuple]:
    """Generate routine-schedule measurements from coherent latent curves."""
    birth_length = float(np.clip(growth_rng.normal(50.0, 1.8), 45.0, 55.0))
    length_velocities = [
        float(growth_rng.uniform(low, high))
        for _, _, low, high in LENGTH_VELOCITY_RANGES
    ]
    weight_velocities = [
        float(growth_rng.uniform(low, high))
        for _, _, low, high in WEIGHT_VELOCITY_RANGES
    ]

    rows = []
    for scheduled_month in ROUTINE_GROWTH_MONTHS:
        scheduled_time = birth_date + pd.DateOffset(months=scheduled_month)
        timestamp = (
            birth_date
            if scheduled_month == 0
            else _jittered_timestamp(scheduled_time, growth_rng, 3)
        )
        timestamp = max(birth_date, timestamp)
        if timestamp > now:
            continue
        age_months = (timestamp - birth_date).days / 30.4375
        latent_length = _piecewise_growth_value(
            age_months,
            birth_length,
            length_velocities,
            LENGTH_VELOCITY_RANGES,
        )
        latent_weight = _piecewise_growth_value(
            age_months,
            birth_weight,
            weight_velocities,
            WEIGHT_VELOCITY_RANGES,
        )
        observed_length = latent_length + float(
            np.clip(growth_rng.normal(0.0, 0.25), -0.5, 0.5)
        )
        observed_weight = latent_weight + float(
            np.clip(growth_rng.normal(0.0, 0.08), -0.2, 0.2)
        )
        rows.extend([
            (
                patient_id,
                timestamp,
                "WeightKg",
                _maybe_missing(
                    round(observed_weight, 2),
                    missingness_rate,
                    missingness_rng,
                ),
            ),
            (
                patient_id,
                timestamp,
                "LengthCm",
                _maybe_missing(
                    round(observed_length, 1),
                    missingness_rate,
                    missingness_rng,
                ),
            ),
        ])
    return rows


def _publication_food_probability(
    age_days: int,
    midpoint_days: float,
    maximum_probability: float,
) -> float:
    """Return an age-dependent neutral food-exposure probability."""
    logistic = float(expit(
        (age_days - midpoint_days) / FOOD_EXPOSURE_LOGISTIC_SCALE_DAYS
    ))
    return (
        FOOD_EXPOSURE_BASE_PROBABILITY
        + (maximum_probability - FOOD_EXPOSURE_BASE_PROBABILITY) * logistic
    )


def generate_distractor_rows(
    patient_id: str,
    birth_date: pd.Timestamp,
    now: pd.Timestamp,
    config: BenchmarkConfig,
    cohort_rng: np.random.Generator,
    distractor_rng: np.random.Generator,
    missingness_rng: np.random.Generator,
    food_exposure_rng: np.random.Generator,
    growth_rng: np.random.Generator,
) -> List[tuple]:
    """Generate FA-independent clinical and measurement distractors."""
    rows = []
    gestational_age = float(np.clip(cohort_rng.normal(39.0, 1.4), 34.0, 42.0))
    birth_weight = float(np.clip(cohort_rng.normal(3.25, 0.45), 1.5, 4.8))
    perinatal = {
        "GestationalAgeWeeks": round(gestational_age, 1),
        "BirthWeightKg": round(birth_weight, 2),
        "ApgarFiveMinutes": float(cohort_rng.integers(7, 11)),
        "BirthOrder": float(cohort_rng.integers(1, 5)),
        "MaternalAgeYears": float(cohort_rng.integers(20, 43)),
        "SocioEconomicScore": round(float(cohort_rng.normal(0.0, 1.0)), 3),
    }
    rows.extend(
        (patient_id, birth_date, feature, value)
        for feature, value in perinatal.items()
    )

    if config.generation_profile == PUBLICATION_V2_PROFILE:
        food_midpoint_days = float(food_exposure_rng.uniform(
            *FOOD_EXPOSURE_MIDPOINT_DAYS_RANGE
        ))
        food_maximum_probability = float(food_exposure_rng.uniform(
            *FOOD_EXPOSURE_MAX_PROBABILITY_RANGE
        ))
    else:
        food_midpoint_days = float("nan")
        food_maximum_probability = float("nan")

    def add_clinical_assessment(timestamp: pd.Timestamp) -> None:
        rows.append((patient_id, timestamp, "ClinicalVisitYN", 1.0))
        for feature, probability in GI_DISTRACTOR_RATES.items():
            if (
                config.generation_profile == PUBLICATION_V2_PROFILE
                and feature == "FoodExposureYN"
            ):
                continue
            value = float(distractor_rng.random() < probability)
            rows.append((patient_id, timestamp, feature, value))
        if config.generation_profile == PUBLICATION_V2_PROFILE:
            age_days = int((timestamp - birth_date).days)
            food_probability = _publication_food_probability(
                age_days,
                food_midpoint_days,
                food_maximum_probability,
            )
            food_value = float(
                food_exposure_rng.random() < food_probability
            )
            rows.append((
                patient_id,
                timestamp,
                "FoodExposureYN",
                food_value,
            ))

    planned = birth_date + pd.Timedelta(days=7)
    seen_clinical_times = set()
    while planned <= now:
        timestamp = _jittered_timestamp(planned, distractor_rng, 2)
        timestamp = max(birth_date, min(timestamp, now))
        if timestamp not in seen_clinical_times:
            seen_clinical_times.add(timestamp)
            add_clinical_assessment(timestamp)
        planned += pd.Timedelta(days=config.clinical_visit_interval_days)

    if (
        config.generation_profile == PUBLICATION_V2_PROFILE
        and now not in seen_clinical_times
    ):
        seen_clinical_times.add(now)
        add_clinical_assessment(now)

    if config.generation_profile == PUBLICATION_V2_PROFILE:
        rows.extend(_publication_growth_rows(
            patient_id,
            birth_date,
            now,
            birth_weight,
            config.continuous_missingness_rate,
            growth_rng,
            missingness_rng,
        ))
        return rows

    measurement_times = [
        birth_date + pd.DateOffset(months=month)
        for month in (1, 2, 4, 6)
    ]
    planned = birth_date + pd.DateOffset(months=6) + pd.Timedelta(days=7)
    while planned <= now:
        measurement_times.append(planned)
        planned += pd.Timedelta(days=config.clinical_visit_interval_days)

    for planned_time in measurement_times:
        timestamp = _jittered_timestamp(planned_time, distractor_rng, 2)
        timestamp = max(birth_date, min(timestamp, now))
        age_months = (timestamp - birth_date).days / 30.4375
        weight = 3.25 + 0.62 * age_months + distractor_rng.normal(0.0, 0.45)
        length = 50.0 + 2.15 * age_months + distractor_rng.normal(0.0, 1.4)
        rows.extend(
            [
                (
                    patient_id,
                    timestamp,
                    "WeightKg",
                    _maybe_missing(
                        round(float(weight), 2),
                        config.continuous_missingness_rate,
                        missingness_rng,
                    ),
                ),
                (
                    patient_id,
                    timestamp,
                    "LengthCm",
                    _maybe_missing(
                        round(float(length), 1),
                        config.continuous_missingness_rate,
                        missingness_rng,
                    ),
                ),
            ]
        )
    return rows


def calibrate_fa_intercept(
    groups: Sequence[str],
    target_prevalence: float,
) -> float:
    """Calibrate the FA intercept for the realized trajectory composition."""
    betas = np.asarray(
        [math.log(FEEDING_TRAJECTORY_OR[str(group)]) for group in groups],
        dtype=float,
    )

    def objective(intercept: float) -> float:
        return float(expit(intercept + betas).mean() - target_prevalence)

    return float(brentq(objective, -30.0, 30.0, xtol=1e-12, rtol=1e-12))


def calculate_fa_probabilities(
    groups: Sequence[str],
    intercept: float,
) -> np.ndarray:
    """Calculate patient-level theoretical FA probabilities."""
    betas = np.asarray(
        [math.log(FEEDING_TRAJECTORY_OR[str(group)]) for group in groups],
        dtype=float,
    )
    return expit(intercept + betas)


def _observation_density(
    data: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> dict:
    anchors = ground_truth.set_index("patient_id")[["birth_date", "now"]]
    observations = data.loc[
        ~data["feature_name"].str.endswith("Anchor", na=False)
    ].copy()
    observations["birth_date"] = observations["patient_id"].map(
        anchors["birth_date"]
    )
    observations["now"] = observations["patient_id"].map(anchors["now"])
    windows = {
        "birth_0_to_6m": (
            observations["time"] >= observations["birth_date"]
        ) & (
            observations["time"]
            <= observations["birth_date"] + pd.DateOffset(months=6)
        ),
        "now_minus_6m_to_now": (
            observations["time"]
            >= observations["now"] - pd.DateOffset(months=6)
        ) & (observations["time"] <= observations["now"]),
        "now_minus_1m_to_now": (
            observations["time"]
            >= observations["now"] - pd.DateOffset(months=1)
        ) & (observations["time"] <= observations["now"]),
    }
    patient_index = pd.Index(ground_truth["patient_id"], name="patient_id")
    summary = {}
    for window_name, mask in windows.items():
        window_rows = observations.loc[mask]
        row_counts = window_rows.groupby("patient_id").size().reindex(
            patient_index,
            fill_value=0,
        )
        timestamp_counts = (
            window_rows.groupby("patient_id")["time"]
            .nunique()
            .reindex(patient_index, fill_value=0)
        )
        summary[window_name] = {
            "long_format_rows_per_patient": _describe_counts(row_counts),
            "unique_timestamps_per_patient": _describe_counts(
                timestamp_counts
            ),
        }
    return summary


def _describe_counts(values: pd.Series) -> dict:
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": int(values.min()),
        "max": int(values.max()),
    }


def _publication_covariate_audit(
    data: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> dict:
    """Audit approved v2 NOW, food-exposure, and growth behavior."""
    private = ground_truth.set_index("patient_id")
    observations = data.loc[
        ~data["feature_name"].str.endswith("Anchor", na=False)
    ].copy()
    observations["birth_date"] = observations["patient_id"].map(
        private["birth_date"]
    )
    observations["age_months"] = (
        observations["time"] - observations["birth_date"]
    ).dt.days / 30.4375

    food = observations.loc[
        observations["feature_name"].eq("FoodExposureYN")
    ].copy()
    food["value"] = pd.to_numeric(food["value"], errors="coerce")
    food["age_band"] = pd.cut(
        food["age_months"],
        [-np.inf, 4.0, 6.0, 12.0, 24.0, np.inf],
        right=False,
        labels=["0_to_4m", "4_to_6m", "6_to_12m", "12_to_24m", "24m_plus"],
    )
    food_summary = {
        str(band): {
            "observations": int(len(group)),
            "positive_count": int(group["value"].sum()),
            "prevalence": float(group["value"].mean()),
        }
        for band, group in food.groupby("age_band", observed=True)
    }

    growth_summary = {}
    for feature in ("LengthCm", "WeightKg"):
        feature_rows = observations.loc[
            observations["feature_name"].eq(feature),
            ["patient_id", "time", "value", "age_months"],
        ].copy()
        counts = feature_rows.groupby("patient_id").size().reindex(
            ground_truth["patient_id"], fill_value=0
        )
        feature_rows["value"] = pd.to_numeric(
            feature_rows["value"], errors="coerce"
        )
        valid = feature_rows.dropna(subset=["value"]).sort_values(
            ["patient_id", "time"]
        )
        valid["gap_days"] = valid.groupby("patient_id")["time"].diff().dt.days
        valid["change"] = valid.groupby("patient_id")["value"].diff()
        decreases = -valid.loc[valid["change"].lt(0), "change"]
        gaps = valid["gap_days"].dropna()
        growth_summary[feature] = {
            "observations_per_patient": _describe_counts(counts),
            "missing_count": int(feature_rows["value"].isna().sum()),
            "missing_rate": float(feature_rows["value"].isna().mean()),
            "measurement_gap_days": {
                "min": int(gaps.min()),
                "median": float(gaps.median()),
                "max": int(gaps.max()),
            },
            "observed_decrease_count": int(len(decreases)),
            "observed_transition_count": int(valid["change"].notna().sum()),
            "observed_decrease_rate": float(
                len(decreases) / max(1, valid["change"].notna().sum())
            ),
            "decrease_magnitude": {
                "median": float(decreases.median()) if len(decreases) else 0.0,
                "max": float(decreases.max()) if len(decreases) else 0.0,
            },
        }

    exact_integer_month = np.zeros(len(ground_truth), dtype=bool)
    for month in range(NOW_AGE_MONTHS_MIN, NOW_AGE_MONTHS_MAX + 1):
        exact_integer_month |= ground_truth["now"].eq(
            ground_truth["birth_date"] + pd.DateOffset(months=month)
        ).to_numpy()
    now_summary = {
        "sampling": "uniform integer day from Birth+18m through Birth+24m",
        "age_days": _describe_counts(ground_truth["now_age_days"]),
        "unique_age_days": int(ground_truth["now_age_days"].nunique()),
        "exact_integer_calendar_month_count": int(exact_integer_month.sum()),
        "exact_integer_calendar_month_proportion": float(
            exact_integer_month.mean()
        ),
    }
    return {
        "now": now_summary,
        "food_exposure": food_summary,
        "growth": growth_summary,
    }


def _trajectory_distribution_summary(ground_truth: pd.DataFrame) -> dict:
    counts = ground_truth["feeding_group"].value_counts().reindex(
        GROUP_ORDER,
        fill_value=0,
    )
    realized = counts / len(ground_truth)
    expected_counts = np.asarray(
        [FEEDING_TRAJECTORY_PROBS[group] *
            len(ground_truth) for group in GROUP_ORDER]
    )
    observed_counts = counts.to_numpy(dtype=float)
    chi_square = chisquare(observed_counts, f_exp=expected_counts)
    standard_deviations = np.sqrt(
        expected_counts
        * (
            1.0
            - np.asarray(
                [FEEDING_TRAJECTORY_PROBS[g] for g in GROUP_ORDER]
            )
        )
    )
    standardized_residuals = (
        observed_counts - expected_counts) / standard_deviations
    return {
        "counts": {group: int(counts[group]) for group in GROUP_ORDER},
        "realized_probabilities": {
            group: float(realized[group]) for group in GROUP_ORDER
        },
        "expected_probabilities": dict(FEEDING_TRAJECTORY_PROBS),
        "expected_counts": {
            group: float(expected_counts[index])
            for index, group in enumerate(GROUP_ORDER)
        },
        "standardized_residuals": {
            group: float(standardized_residuals[index])
            for index, group in enumerate(GROUP_ORDER)
        },
        "max_absolute_standardized_residual": float(
            np.max(np.abs(standardized_residuals))
        ),
        "chi_square_statistic": float(chi_square.statistic),
        "chi_square_p_value": float(chi_square.pvalue),
    }


def _validate_feeding_observations(
    data: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> bool:
    feeding = data.loc[
        data["feature_name"].isin(["FormulaYN", "BreastMilkYN"])
    ].copy()
    birth_by_patient = ground_truth.set_index("patient_id")["birth_date"]
    feeding["birth_date"] = feeding["patient_id"].map(birth_by_patient)
    feeding = feeding.loc[
        feeding["time"] <= feeding["birth_date"] + pd.DateOffset(months=6)
    ]
    paired = feeding.pivot(
        index=["patient_id", "time"],
        columns="feature_name",
        values="value",
    )
    if paired[["FormulaYN", "BreastMilkYN"]].isna().any().any():
        return False

    specs = ground_truth.set_index("patient_id")
    for (patient_id, timestamp), values in paired.iterrows():
        spec = specs.loc[patient_id]
        expected_state = feeding_state_at(
            timestamp,
            spec["initial_state"],
            spec["transition_times"],
            spec["transition_states"],
        )
        if tuple(values[["FormulaYN", "BreastMilkYN"]]) != FEEDING_STATES[
            expected_state
        ]:
            return False
    return True


def _validate_transition_ranges(ground_truth: pd.DataFrame) -> bool:
    for row in ground_truth.itertuples(index=False):
        birth = row.birth_date
        transitions = tuple(row.transition_times)
        if row.feeding_group in {"2.0", "2.4", "3"}:
            if transitions:
                return False
        elif row.feeding_group in {"2.1", "4"}:
            if transitions != (birth + pd.Timedelta(days=21),):
                return False
        elif row.feeding_group in {"2.2", "5.1"}:
            if not (
                len(transitions) == 1
                and birth + pd.Timedelta(days=21) < transitions[0]
                <= birth + pd.DateOffset(months=2)
            ):
                return False
        elif row.feeding_group in {"2.3", "5.2"}:
            if not (
                len(transitions) == 1
                and birth + pd.DateOffset(months=2) < transitions[0]
                <= birth + pd.DateOffset(months=4)
            ):
                return False
        elif row.feeding_group == "5.3":
            if not (
                len(transitions) == 1
                and birth + pd.DateOffset(months=4) < transitions[0]
                <= birth + pd.DateOffset(months=6)
            ):
                return False
        elif row.feeding_group == "6":
            if len(transitions) < 2:
                return False
        else:
            return False
    return True


def extract_observable_boundary_signatures(
    data: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> Dict[str, Optional[tuple]]:
    """Extract each patient's state signature at defining boundaries.

    The first element is the unique observed state before 21 days. The
    remaining elements are the states observed exactly at 21 days and at
    two, four, and six calendar months. A ``None`` signature indicates
    incomplete, duplicated, or internally inconsistent feeding observations.
    """
    feeding = data.loc[
        data["feature_name"].isin(["FormulaYN", "BreastMilkYN"])
    ].copy()
    duplicate_feature_rows = feeding.duplicated(
        ["patient_id", "time", "feature_name"],
        keep=False,
    )
    invalid_patients = set(feeding.loc[duplicate_feature_rows, "patient_id"])
    feeding = feeding.loc[~feeding["patient_id"].isin(invalid_patients)]
    paired = feeding.pivot(
        index=["patient_id", "time"],
        columns="feature_name",
        values="value",
    ).reset_index()
    signatures: Dict[str, Optional[tuple]] = {}

    for row in ground_truth.itertuples(index=False):
        patient_id = row.patient_id
        if patient_id in invalid_patients:
            signatures[patient_id] = None
            continue
        observations = paired.loc[paired["patient_id"].eq(patient_id)].copy()
        if observations[["FormulaYN", "BreastMilkYN"]].isna().any().any():
            signatures[patient_id] = None
            continue
        observations["state"] = observations.apply(
            lambda observation: _feeding_state_from_values(
                observation["FormulaYN"],
                observation["BreastMilkYN"],
            ),
            axis=1,
        )
        if observations["state"].isna().any():
            signatures[patient_id] = None
            continue

        day_21 = row.birth_date + pd.Timedelta(days=21)
        boundary_times = (
            day_21,
            row.birth_date + pd.DateOffset(months=2),
            row.birth_date + pd.DateOffset(months=4),
            row.birth_date + pd.DateOffset(months=6),
        )
        pre_boundary_states = observations.loc[
            observations["time"] < day_21,
            "state",
        ].unique()
        if len(pre_boundary_states) != 1:
            signatures[patient_id] = None
            continue

        boundary_states = []
        for boundary_time in boundary_times:
            states = observations.loc[
                observations["time"].eq(boundary_time),
                "state",
            ]
            if len(states) != 1:
                boundary_states = []
                break
            boundary_states.append(states.iloc[0])
        signatures[patient_id] = (
            (str(pre_boundary_states[0]), *boundary_states)
            if len(boundary_states) == len(boundary_times)
            else None
        )
    return signatures


def _feeding_state_from_values(formula, breast_milk) -> Optional[str]:
    values = (float(formula), float(breast_milk))
    for state, expected_values in FEEDING_STATES.items():
        if values == expected_values:
            return state
    return None


def validate_benchmark(
    data: pd.DataFrame,
    fa_labels: pd.DataFrame,
    ground_truth: pd.DataFrame,
    config: BenchmarkConfig,
    require_empirical_distribution: bool = True,
) -> dict:
    """Validate structural and scientific invariants before artifact saving."""
    patient_ids = set(ground_truth["patient_id"])
    data_patient_ids = set(data["patient_id"])
    label_patient_ids = set(fa_labels["patient_id"])
    ground_truth_rows_per_patient = ground_truth.groupby("patient_id").size()
    private_groups_per_patient = ground_truth.groupby("patient_id")[
        "feeding_group"
    ].nunique(dropna=False)
    checks = {
        "cohort_size": len(patient_ids) == config.n_patients,
        "one_private_ground_truth_row_per_patient": bool(
            len(ground_truth) == config.n_patients
            and ground_truth_rows_per_patient.eq(1).all()
        ),
        "exactly_one_private_feeding_group_per_patient": bool(
            private_groups_per_patient.eq(1).all()
            and ground_truth["feeding_group"].isin(GROUP_ORDER).all()
        ),
        "no_multiple_trajectory_assignments": bool(
            ground_truth_rows_per_patient.eq(1).all()
            and private_groups_per_patient.eq(1).all()
        ),
        "feeding_groups_cover_full_cohort": bool(
            ground_truth["feeding_group"].isin(GROUP_ORDER).all()
            and ground_truth["feeding_group"].value_counts().sum()
            == config.n_patients
        ),
        "same_patient_population": (
            patient_ids == data_patient_ids == label_patient_ids
        ),
        "longitudinal_schema": list(data.columns) == LONGITUDINAL_COLUMNS,
        "label_schema": list(fa_labels.columns) == LABEL_COLUMNS,
        "unique_labels": fa_labels["patient_id"].is_unique,
        "binary_labels": fa_labels["label"].isin([0, 1]).all(),
    }

    for anchor_name in ("Birth Date Anchor", "NOW Anchor"):
        anchor_rows = data.loc[data["feature_name"].eq(anchor_name)]
        counts = anchor_rows.groupby("patient_id").size().reindex(
            ground_truth["patient_id"],
            fill_value=0,
        )
        normalized_anchor = anchor_name.lower().replace(" ", "_")
        checks[f"one_{normalized_anchor}_per_patient"] = bool(
            counts.eq(1).all()
        )
        checks[f"valid_{normalized_anchor}_timestamps"] = bool(
            anchor_rows["value"]
            .map(lambda value: isinstance(value, pd.Timestamp))
            .all()
        )

    now_by_patient = ground_truth.set_index("patient_id")["now"]
    row_now = data["patient_id"].map(now_by_patient)
    checks["no_observations_after_now"] = bool((data["time"] <= row_now).all())
    if config.generation_profile == PUBLICATION_V2_PROFILE:
        earliest_now = ground_truth["birth_date"] + pd.DateOffset(
            months=config.now_age_months_min
        )
        latest_now = ground_truth["birth_date"] + pd.DateOffset(
            months=config.now_age_months_max
        )
        checks["now_age_within_configured_range"] = bool(
            ground_truth["now"].between(earliest_now, latest_now).all()
        )
        exact_integer_month = np.zeros(len(ground_truth), dtype=bool)
        for month in range(
            config.now_age_months_min,
            config.now_age_months_max + 1,
        ):
            exact_integer_month |= ground_truth["now"].eq(
                ground_truth["birth_date"] + pd.DateOffset(months=month)
            ).to_numpy()
        checks["day_level_now_distribution"] = bool(
            ground_truth["now_age_days"].nunique()
            >= min(100, max(2, config.n_patients // 2))
            and exact_integer_month.mean() < 0.10
        )
    else:
        checks["now_age_within_configured_range"] = bool(
            ground_truth["now_age_months"].between(
                config.now_age_months_min,
                config.now_age_months_max,
            ).all()
        )
    checks["patient_specific_now_ages"] = bool(
        ground_truth["now_age_months"].nunique() > 1
        if config.n_patients > 1
        else True
    )
    checks["strict_temporal_separation"] = bool(
        (
            ground_truth["birth_date"] + pd.DateOffset(months=6)
            < ground_truth["now"] - pd.DateOffset(months=6)
        ).all()
    )

    distribution = _trajectory_distribution_summary(ground_truth)
    checks["all_11_trajectories_represented"] = bool(
        all(distribution["counts"][group] > 0 for group in GROUP_ORDER)
    )
    checks["trajectory_distribution_reasonable"] = bool(
        distribution["max_absolute_standardized_residual"] < 5.0
        and distribution["chi_square_p_value"] > 1e-6
    )
    checks["feeding_observations_match_ground_truth"] = (
        _validate_feeding_observations(data, ground_truth)
    )
    checks["transition_ranges_valid"] = _validate_transition_ranges(
        ground_truth)

    boundary_signatures = extract_observable_boundary_signatures(
        data,
        ground_truth,
    )
    checks["exact_feeding_boundary_observations_present"] = bool(
        len(boundary_signatures) == config.n_patients
        and all(
            signature is not None
            for signature in boundary_signatures.values()
        )
    )
    regular_match_counts = {}
    assigned_group_matches = {}
    for row in ground_truth.itertuples(index=False):
        signature = boundary_signatures.get(row.patient_id)
        matching_regular_groups = [
            group
            for group, expected in REGULAR_BOUNDARY_SIGNATURES.items()
            if signature == expected
        ]
        regular_match_counts[row.patient_id] = len(matching_regular_groups)
        assigned_group_matches[row.patient_id] = (
            matching_regular_groups == [row.feeding_group]
            if row.feeding_group != "6"
            else signature in GROUP_6_BOUNDARY_SIGNATURES
        )
    checks["no_observable_regular_group_overlap"] = bool(
        all(count <= 1 for count in regular_match_counts.values())
    )
    checks["observable_history_matches_single_assigned_group"] = bool(
        all(assigned_group_matches.values())
    )
    group_6_patient_ids = set(
        ground_truth.loc[ground_truth["feeding_group"].eq("6"), "patient_id"]
    )
    checks["group_6_exclusive_from_regular_groups"] = bool(
        all(regular_match_counts[patient_id] == 0
            for patient_id in group_6_patient_ids)
    )

    expected_beta = ground_truth["feeding_group"].map(
        {group: math.log(value)
         for group, value in FEEDING_TRAJECTORY_OR.items()}
    )
    checks["feeding_betas_match_log_ors"] = bool(
        np.allclose(ground_truth["beta_j"],
                    expected_beta, atol=1e-12, rtol=0.0)
    )
    checks["mean_fa_probability_matches_target"] = bool(
        math.isclose(
            float(ground_truth["fa_probability"].mean()),
            config.target_fa_prevalence,
            rel_tol=0.0,
            abs_tol=1e-10,
        )
    )
    group_probabilities = ground_truth.groupby("feeding_group")[
        "fa_probability"
    ].first()
    ordered_groups = sorted(GROUP_ORDER, key=FEEDING_TRAJECTORY_OR.get)
    checks["fa_probability_order_matches_ors"] = bool(
        group_probabilities.reindex(ordered_groups).is_monotonic_increasing
    )
    realized_prevalence = float(fa_labels["label"].mean())
    sampling_se = math.sqrt(
        config.target_fa_prevalence
        * (1.0 - config.target_fa_prevalence)
        / config.n_patients
    )
    checks["realized_fa_prevalence_reasonable"] = bool(
        abs(realized_prevalence - config.target_fa_prevalence)
        <= max(0.03, 5.0 * sampling_se)
    )

    normalized_features = {
        str(feature).replace(" ", "").replace("_", "").lower()
        for feature in data["feature_name"].unique()
    }
    normalized_leakage = {
        feature.replace(" ", "").replace("_", "").lower()
        for feature in LEAKAGE_FEATURES
    }
    checks["no_leakage_features"] = normalized_features.isdisjoint(
        normalized_leakage
    )

    density = _observation_density(data, ground_truth)
    checks["sufficient_birth_density"] = (
        density["birth_0_to_6m"]["long_format_rows_per_patient"]["min"]
        >= 20
        and density["birth_0_to_6m"]["unique_timestamps_per_patient"]["min"]
        >= 10
    )
    checks["sufficient_now_6m_density"] = (
        density["now_minus_6m_to_now"]["long_format_rows_per_patient"]["min"]
        >= 20
        and density["now_minus_6m_to_now"]
        ["unique_timestamps_per_patient"]["min"]
        >= 10
    )
    checks["sufficient_now_1m_density"] = (
        density["now_minus_1m_to_now"]["long_format_rows_per_patient"]["min"]
        >= 5
        and density["now_minus_1m_to_now"]
        ["unique_timestamps_per_patient"]["min"]
        >= 2
    )

    publication_audit = None
    if config.generation_profile == PUBLICATION_V2_PROFILE:
        publication_audit = _publication_covariate_audit(data, ground_truth)
        food = publication_audit["food_exposure"]
        intended_bands = ("0_to_4m", "4_to_6m", "6_to_12m", "12_to_24m")
        food_prevalence = [food[band]["prevalence"] for band in intended_bands]
        checks["food_exposure_increases_with_age"] = bool(
            all(left < right for left, right in zip(
                food_prevalence, food_prevalence[1:]
            )) if config.n_patients >= 500 else True
        )
        for feature, maximum_decrease in (
            ("LengthCm", 0.5),
            ("WeightKg", 0.3),
        ):
            audit = publication_audit["growth"][feature]
            normalized = feature.lower().replace("cm", "_cm").replace(
                "kg", "_kg"
            )
            checks[f"routine_{normalized}_measurement_frequency"] = bool(
                audit["observations_per_patient"]["min"] >= 7
                and audit["observations_per_patient"]["max"] <= 9
                and audit["measurement_gap_days"]["min"] >= 20
            )
            checks[f"coherent_{normalized}_trajectory"] = bool(
                audit["decrease_magnitude"]["max"] <= maximum_decrease
            )

    required_checks = dict(checks)
    if not require_empirical_distribution:
        required_checks.pop("trajectory_distribution_reasonable")
        required_checks.pop("all_11_trajectories_represented")
    failed = [name for name, passed in required_checks.items() if not passed]
    if failed:
        raise ValueError(
            "Benchmark validation failed: " + ", ".join(sorted(failed))
        )
    return {
        "checks": {name: bool(value) for name, value in checks.items()},
        "trajectory_distribution": distribution,
        "observation_density": density,
        "realized_fa_prevalence": realized_prevalence,
        "fa_positive_count": int(fa_labels["label"].sum()),
        **(
            {"publication_v2_covariate_audit": publication_audit}
            if publication_audit is not None
            else {}
        ),
    }


def generate_benchmark(
    config: BenchmarkConfig = BenchmarkConfig(),
    assigned_groups: Optional[Sequence[str]] = None,
    require_empirical_distribution: bool = True,
) -> BenchmarkResult:
    """Generate and validate an in-memory shared-X/FA benchmark."""
    validate_configuration(config)
    rngs, stream_metadata = _make_rng_streams(
        config.master_seed,
        config.generation_profile,
    )
    cohort = generate_cohort(config, rngs["cohort"])
    if assigned_groups is None:
        groups = sample_feeding_groups(config.n_patients, rngs["trajectory"])
    else:
        if len(assigned_groups) != config.n_patients:
            raise ValueError("assigned_groups length must equal n_patients.")
        if not set(assigned_groups).issubset(GROUP_ORDER):
            raise ValueError("assigned_groups contains an unknown trajectory.")
        groups = np.asarray(assigned_groups, dtype=object)

    rows = []
    ground_truth_rows = []
    for patient, group in zip(cohort.itertuples(index=False), groups):
        group = str(group)
        rows.extend(
            [
                (
                    patient.patient_id,
                    patient.birth_date,
                    "Birth Date Anchor",
                    patient.birth_date,
                ),
                (
                    patient.patient_id,
                    patient.now,
                    "NOW Anchor",
                    patient.now,
                ),
            ]
        )
        trajectory_spec = build_trajectory_spec(
            group,
            patient.birth_date,
            rngs["trajectory"],
        )
        rows.extend(
            generate_feeding_rows(
                patient.patient_id,
                patient.birth_date,
                patient.now,
                trajectory_spec,
                rngs["feeding_observations"],
            )
        )
        rows.extend(
            generate_distractor_rows(
                patient.patient_id,
                patient.birth_date,
                patient.now,
                config,
                rngs["cohort"],
                rngs["distractors"],
                rngs["missingness"],
                rngs.get("food_exposure", rngs["distractors"]),
                rngs.get("growth", rngs["distractors"]),
            )
        )
        private_row = {
            "patient_id": patient.patient_id,
            "birth_date": patient.birth_date,
            "now": patient.now,
            "now_age_months": patient.now_age_months,
            "feeding_group": group,
            **trajectory_spec,
            "feeding_or": float(FEEDING_TRAJECTORY_OR[group]),
            "beta_j": float(math.log(FEEDING_TRAJECTORY_OR[group])),
        }
        if config.generation_profile == PUBLICATION_V2_PROFILE:
            private_row["now_age_days"] = patient.now_age_days
        ground_truth_rows.append(private_row)

    data = pd.DataFrame.from_records(rows, columns=LONGITUDINAL_COLUMNS)
    data["time"] = pd.to_datetime(data["time"])
    data = data.sort_values(
        ["patient_id", "time", "feature_name"],
        kind="mergesort",
    ).reset_index(drop=True)
    ground_truth = pd.DataFrame.from_records(ground_truth_rows).sort_values(
        "patient_id"
    ).reset_index(drop=True)

    fa_intercept = calibrate_fa_intercept(
        ground_truth["feeding_group"],
        config.target_fa_prevalence,
    )
    fa_probabilities = calculate_fa_probabilities(
        ground_truth["feeding_group"],
        fa_intercept,
    )
    ground_truth["fa_probability"] = fa_probabilities
    fa_labels = ground_truth[["patient_id"]].copy()
    fa_labels["label"] = rngs["fa_labels"].binomial(
        1,
        fa_probabilities,
    ).astype(int)

    validation = validate_benchmark(
        data,
        fa_labels,
        ground_truth,
        config,
        require_empirical_distribution=require_empirical_distribution,
    )
    per_group_fa_probability = {
        group: float(
            expit(fa_intercept + math.log(FEEDING_TRAJECTORY_OR[group])))
        for group in GROUP_ORDER
    }
    if config.generation_profile == PUBLICATION_V2_PROFILE:
        now_age_metadata = validation[
            "publication_v2_covariate_audit"
        ]["now"]
        benchmark_version = "publication_v2_fa_v1"
    else:
        now_age_metadata = {
            "configured_min_months": config.now_age_months_min,
            "configured_max_months": config.now_age_months_max,
            "realized_counts": {
                str(age): int(count)
                for age, count in ground_truth["now_age_months"]
                .value_counts()
                .sort_index()
                .items()
            },
        }
        benchmark_version = "phase_1_fa_v1"
    metadata = {
        "benchmark_version": benchmark_version,
        "gi_label_status": "not_implemented",
        "cohort_size": config.n_patients,
        "master_seed": config.master_seed,
        "rng_streams": stream_metadata,
        "output_directory": str(config.output_dir),
        "longitudinal_schema": LONGITUDINAL_COLUMNS,
        "label_schema": LABEL_COLUMNS,
        "now_age_distribution": now_age_metadata,
        "feeding": {
            "experiment_3_population": N_EXPERIMENT_3,
            "empirical_counts": dict(FEEDING_TRAJECTORY_COUNTS),
            "sampling_probabilities": dict(FEEDING_TRAJECTORY_PROBS),
            "odds_ratios": dict(FEEDING_TRAJECTORY_OR),
            "log_odds_coefficients": {
                group: float(math.log(value))
                for group, value in FEEDING_TRAJECTORY_OR.items()
            },
            "conceptual_templates": {
                group: list(template)
                for group, template in FEEDING_TEMPLATES.items()
            },
            "inconsistent_templates": [
                list(template) for template in INCONSISTENT_FEEDING_TEMPLATES
            ],
            "transition_rules": {
                "2.1": "Mixed to formula at exactly 21 days",
                "2.2": (
                    "Mixed to formula uniformly in "
                    "(21 days, 2 calendar months]"
                ),
                "2.3": (
                    "Mixed to formula uniformly in (2, 4 calendar months]"
                ),
                "4": "Formula to EBF at exactly 21 days",
                "5.1": (
                    "EBF to formula uniformly in "
                    "(21 days, 2 calendar months]"
                ),
                "5.2": (
                    "EBF to formula uniformly in (2, 4 calendar months]"
                ),
                "5.3": (
                    "EBF to formula uniformly in (4, 6 calendar months]"
                ),
                "6": (
                    "Approved explicit templates transition at "
                    "21d, 2m, and 4m"
                ),
            },
            "post_6_month_behavior": (
                "Independent categorical F/B/M samples; independent of early "
                "trajectory and FA"
            ),
        },
        "fa": {
            "target_prevalence": config.target_fa_prevalence,
            "calibrated_intercept": fa_intercept,
            "per_group_probability": per_group_fa_probability,
            "mean_theoretical_probability": float(fa_probabilities.mean()),
            "realized_prevalence": validation["realized_fa_prevalence"],
            "positive_count": validation["fa_positive_count"],
        },
        "distractors": {
            "gi_feature_rates_per_assessment": {
                feature: rate
                for feature, rate in GI_DISTRACTOR_RATES.items()
                if not (
                    config.generation_profile == PUBLICATION_V2_PROFILE
                    and feature == "FoodExposureYN"
                )
            },
            "clinical_visit_interval_days": (
                config.clinical_visit_interval_days
            ),
            "gi_features_affect_fa": False,
        },
        "missingness": {
            "continuous_measurement_rate": (
                config.continuous_missingness_rate
            ),
            "anchors_missing": False,
            "feeding_assessments_missing": False,
            "depends_on_fa": False,
        },
        "ground_truth_windows": {
            "fa": "Birth Date Anchor 0 to +6 calendar months",
            "future_gi": "NOW Anchor -1 calendar month to NOW",
            "density_now": "NOW Anchor -6 calendar months to NOW",
        },
        "feature_names": sorted(data["feature_name"].unique().tolist()),
        "validation": validation,
    }
    if config.generation_profile == PUBLICATION_V2_PROFILE:
        metadata["generation_profile"] = config.generation_profile
        metadata["publication_v2_changes"] = {
            "now_sampling": (
                "Uniform integer day from patient-specific Birth+18 calendar "
                "months through Birth+24 calendar months, inclusive"
            ),
            "food_exposure": {
                "status": "neutral age-dependent distractor",
                "base_probability": FOOD_EXPOSURE_BASE_PROBABILITY,
                "maximum_probability_range": list(
                    FOOD_EXPOSURE_MAX_PROBABILITY_RANGE
                ),
                "patient_midpoint_days_range": list(
                    FOOD_EXPOSURE_MIDPOINT_DAYS_RANGE
                ),
                "logistic_scale_days": FOOD_EXPOSURE_LOGISTIC_SCALE_DAYS,
                "affects_fa": False,
                "affects_gi": False,
            },
            "clinical_assessments": {
                "background_interval_days": (
                    config.clinical_visit_interval_days
                ),
                "background_jitter_days": 2,
                "patient_specific_now_assessment": True,
            },
            "growth": {
                "routine_schedule_months": list(ROUTINE_GROWTH_MONTHS),
                "measurement_jitter_days": 3,
                "patient_specific_latent_trajectories": True,
                "length_velocity_ranges_cm_per_month": [
                    list(values) for values in LENGTH_VELOCITY_RANGES
                ],
                "weight_velocity_ranges_kg_per_month": [
                    list(values) for values in WEIGHT_VELOCITY_RANGES
                ],
                "affects_fa": False,
                "affects_gi": False,
            },
        }
    return BenchmarkResult(data, fa_labels, ground_truth, metadata)


def artifact_paths(config: BenchmarkConfig) -> Dict[str, Path]:
    """Return the four artifact paths for the configured cohort size."""
    suffix = str(config.n_patients)
    return {
        "data": config.output_dir / f"temporal_benchmark_{suffix}.pkl",
        "fa_labels": (
            config.output_dir / f"temporal_benchmark_fa_labels_{suffix}.pkl"
        ),
        "metadata": (
            config.output_dir / f"temporal_benchmark_metadata_{suffix}.json"
        ),
        "ground_truth": (
            config.output_dir / f"temporal_benchmark_ground_truth_{suffix}.pkl"
        ),
    }


def save_benchmark(
    result: BenchmarkResult,
    config: BenchmarkConfig,
) -> Dict[str, Path]:
    """Save validated artifacts atomically and refuse accidental overwrites."""
    paths = artifact_paths(config)
    existing = [path for path in paths.values() if path.exists()]
    if existing and not config.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing benchmark artifacts: "
            + ", ".join(str(path) for path in existing)
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".temporal_benchmark_",
        dir=config.output_dir,
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        temporary_paths = {
            name: temporary_root / path.name for name, path in paths.items()
        }
        result.data.to_pickle(temporary_paths["data"])
        result.fa_labels.to_pickle(temporary_paths["fa_labels"])
        result.ground_truth.to_pickle(temporary_paths["ground_truth"])
        temporary_paths["metadata"].write_text(
            json.dumps(result.metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        for name, path in paths.items():
            temporary_paths[name].replace(path)
    return paths


def print_summary(result: BenchmarkResult, paths: Mapping[str, Path]) -> None:
    """Print the concise scientific validation summary requested for review."""
    metadata = result.metadata
    distribution = metadata["validation"]["trajectory_distribution"]
    print("TEMPORAL BENCHMARK PHASE 1 — FA SANITY COHORT")
    print("=" * 55)
    print(f"Patients: {metadata['cohort_size']}")
    print("\nTrajectory distribution:")
    for group in GROUP_ORDER:
        count = distribution["counts"][group]
        realized = 100 * distribution["realized_probabilities"][group]
        expected = 100 * distribution["expected_probabilities"][group]
        print(
            f"  {group:>3}: {count:4d} | realized={realized:7.3f}% "
            f"| expected={expected:7.3f}%"
        )
    print(
        "\nCalibrated FA beta0: "
        f"{metadata['fa']['calibrated_intercept']:.12f}"
    )
    print("Theoretical P(FA) by trajectory:")
    for group, probability in metadata["fa"]["per_group_probability"].items():
        print(f"  {group:>3}: {probability:.8f}")
    print(
        "Mean theoretical P(FA): "
        f"{metadata['fa']['mean_theoretical_probability']:.12f}"
    )
    print(
        f"Realized FA prevalence: {metadata['fa']['realized_prevalence']:.6f} "
        f"({metadata['fa']['positive_count']} positives)"
    )
    print("\nObservation density (anchors excluded):")
    for window, statistics in metadata["validation"][
        "observation_density"
    ].items():
        row_mean = statistics["long_format_rows_per_patient"]["mean"]
        time_mean = statistics["unique_timestamps_per_patient"]["mean"]
        print(
            f"  {window}: rows={row_mean:.3f}, "
            f"unique_timestamps={time_mean:.3f}"
        )
    print("\nValidation checks:")
    for check, passed in metadata["validation"]["checks"].items():
        print(f"  {check}: {'PASS' if passed else 'FAIL'}")
    print("\nSaved artifacts:")
    for path in paths.values():
        print(f"  {path.resolve()}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-patients", type=int, default=N_PATIENTS)
    parser.add_argument("--master-seed", type=int, default=MASTER_SEED)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--generation-profile",
        choices=GENERATION_PROFILES,
        default=VALIDATED_V1_PROFILE,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly allow replacement of artifacts for this cohort size.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    config = BenchmarkConfig(
        n_patients=args.n_patients,
        master_seed=args.master_seed,
        output_dir=args.output_dir,
        generation_profile=args.generation_profile,
        overwrite=args.overwrite,
    )
    result = generate_benchmark(config)
    paths = save_benchmark(result, config)
    print_summary(result, paths)


if __name__ == "__main__":
    main()
