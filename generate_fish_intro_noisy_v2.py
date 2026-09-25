"""Noisy, exactly marginal-matched Experiment 2 V2; never fit a model.

Symptom histories are sampled without window exclusions. Matching selects
unaltered histories and fish timestamps with opposite response relationships.
See experiments/experiment_2_noisy_v2.md for conditioning and interpretation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from generate_fish_intro_dataset import BIRTH, FISH, SYMPTOMS, FEATURE_ORDER

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data/fish_intro_noisy_matched_v2"
VERSION = "fish_intro_noisy_matched_v2"
EVENT_FEATURES = tuple(f for f in FEATURE_ORDER if f != BIRTH)
DISTRACTORS = tuple(f for f in EVENT_FEATURES if f not in (FISH, *SYMPTOMS))
OBSERVATION_COLUMNS = ["patient_id", "time", "feature_name", "value"]


@dataclass(frozen=True)
class GeneratorConfig:
    """Synthetic assumptions over 30 months, NOT epidemiological estimates."""

    seed: int = 42
    blocks: int = 500
    test_fraction: float = .2
    gi_mean: float = 5.0
    vomiting_mean: float = 3.0
    fish_min_months: int = 6
    fish_max_months: int = 24
    candidate_times: int = 256
    max_attempts_per_block: int = 100
    cough_episode_mean: float = 3.0
    sleep_episode_mean: float = 2.0
    cluster_probability: float = .4
    cluster_max_extra: int = 2
    cluster_max_days: int = 3
    doctor_visit_mean: float = 2.0
    minor_injury_mean: float = .5
    vaccination_months: tuple = (2, 4, 6, 12, 18, 24)
    vaccination_probability: float = .85
    vaccination_jitter_days: int = 10

    def validate(self) -> None:
        """Reject configurations that cannot support this benchmark."""
        if self.blocks < 2 or not 0 < self.test_fraction < 1:
            raise ValueError("Need at least two blocks and a nontrivial split")
        if not 1 <= round(self.blocks * self.test_fraction) < self.blocks:
            raise ValueError("Train and test must both contain blocks")
        if not 0 < self.fish_min_months < self.fish_max_months <= 29:
            raise ValueError("Fish ages must leave one month of observed follow-up")
        if min(self.candidate_times, self.max_attempts_per_block) < 1:
            raise ValueError("Matching budgets must be positive")
        rates = (self.gi_mean, self.vomiting_mean, self.cough_episode_mean,
                 self.sleep_episode_mean, self.doctor_visit_mean,
                 self.minor_injury_mean)
        if any(not np.isfinite(rate) or rate < 0 for rate in rates):
            raise ValueError("Rates must be finite and nonnegative")
        if self.gi_mean + self.vomiting_mean == 0:
            raise ValueError("Cannot match positive windows without symptoms")
        if not all(0 <= p <= 1 for p in
                   (self.cluster_probability, self.vaccination_probability)):
            raise ValueError("Probabilities must be in [0, 1]")
        if (self.cluster_max_extra < 1 or self.cluster_max_days < 1
                or self.vaccination_jitter_days < 0):
            raise ValueError("Invalid cluster or jitter bounds")
        if not all(0 < month < 30 for month in self.vaccination_months):
            raise ValueError("Vaccination milestones must be inside follow-up")


def _uniform_times(rng, birth, end, count: int) -> list:
    seconds = int((end - birth).total_seconds())
    return list(birth + pd.to_timedelta(rng.integers(0, seconds + 1, count), unit="s"))


def symptom_history(rng, birth, end, config: GeneratorConfig) -> list:
    """Generate all symptoms over full follow-up, without reference to fish."""
    return [(feature, time) for feature, mean in
            zip(SYMPTOMS, (config.gi_mean, config.vomiting_mean))
            for time in _uniform_times(rng, birth, end, rng.poisson(mean))]


def distractor_history(rng, birth, end, config: GeneratorConfig) -> list:
    """Generate events from age, RNG, and assumptions only; no target inputs."""
    events = []
    for feature, mean in (("Cough", config.cough_episode_mean),
                          ("SleepDisturbance", config.sleep_episode_mean)):
        for time in _uniform_times(rng, birth, end, rng.poisson(mean)):
            events.append((feature, time))
            if rng.random() < config.cluster_probability:
                extras = rng.integers(1, config.cluster_max_extra + 1)
                for seconds in rng.integers(1, config.cluster_max_days * 86400 + 1, extras):
                    followup = time + pd.Timedelta(seconds=int(seconds))
                    if followup <= end:
                        events.append((feature, followup))
    for feature, mean in (("DoctorVisit", config.doctor_visit_mean),
                          ("MinorInjury", config.minor_injury_mean)):
        events.extend((feature, time) for time in
                      _uniform_times(rng, birth, end, rng.poisson(mean)))
    for month in config.vaccination_months:
        if rng.random() < config.vaccination_probability:
            bound = config.vaccination_jitter_days * 86400
            time = birth + pd.DateOffset(months=month) + pd.Timedelta(
                seconds=int(rng.integers(-bound, bound + 1)))
            if birth <= time <= end:
                events.append(("RoutineVaccination", time))
    return events


def qualifying_counts(history: list, fish_times: list) -> np.ndarray:
    """Count every symptom in (fish, fish + one calendar month]."""
    times = np.array([time.value for feature, time in history if feature in SYMPTOMS],
                     dtype=np.int64)
    starts = pd.DatetimeIndex(fish_times)
    ends = starts + pd.DateOffset(months=1)
    return ((times[:, None] > starts.asi8[None, :])
            & (times[:, None] <= ends.asi8[None, :])).sum(axis=0)


def _matched_histories(symptom_rng, fish_rng, birth, end, config):
    for attempt in range(1, config.max_attempts_per_block + 1):
        histories = [symptom_history(symptom_rng, birth, end, config) for _ in range(2)]
        candidates = _uniform_times(
            fish_rng, birth + pd.DateOffset(months=config.fish_min_months),
            birth + pd.DateOffset(months=config.fish_max_months), config.candidate_times)
        counts = [qualifying_counts(history, candidates) for history in histories]
        # Remove exact simultaneity, since the existing model's intervals are
        # inclusive at the start whereas synthetic R is strictly after fish.
        symptom_times = {time for history in histories for _, time in history}
        allowed = np.array([time not in symptom_times for time in candidates])
        first = np.flatnonzero((counts[0] > 0) & (counts[1] == 0) & allowed)
        second = np.flatnonzero((counts[0] == 0) & (counts[1] > 0) & allowed)
        if len(first) and len(second):
            a = candidates[int(fish_rng.choice(first))]
            b = candidates[int(fish_rng.choice(second))]
            return histories, (a, b), attempt
    raise ValueError("No matched block within fixed budget; do not alter labels or tune to AUC")


def recompute_r(observations: pd.DataFrame, patients: pd.DataFrame) -> pd.Series:
    """Derive R from observed fish and all symptom rows, never metadata labels."""
    fish = observations.loc[observations.feature_name.eq(FISH), ["patient_id", "time"]]
    if fish.patient_id.duplicated().any():
        raise ValueError("More than one fish event per child")
    fish_times = fish.set_index("patient_id").time
    symptoms = observations.loc[observations.feature_name.isin(SYMPTOMS)]
    starts = symptoms.patient_id.map(fish_times)
    ends = symptoms.patient_id.map(fish_times + pd.DateOffset(months=1))
    qualifying_ids = symptoms.loc[symptoms.time.gt(starts) & symptoms.time.le(ends),
                                 "patient_id"]
    return pd.Series(patients.patient_id.isin(qualifying_ids).astype(int).to_numpy(),
                     index=patients.index, name="R")


def _history_signatures(data: pd.DataFrame, patients: pd.DataFrame) -> dict:
    births = patients.set_index("patient_id").birth
    signatures = {pid: {f: [] for f in EVENT_FEATURES} for pid in patients.patient_id}
    for pid, time, feature, _ in data.itertuples(index=False, name=None):
        if feature != BIRTH:
            signatures[pid][feature].append(int((time - births[pid]).total_seconds()))
    return {pid: {f: tuple(sorted(times)) for f, times in features.items()}
            for pid, features in signatures.items()}


def validate_dataset(data: pd.DataFrame, patients: pd.DataFrame) -> dict:
    """Check truth, exact marginal/full non-fish-history matching and partitions."""
    if list(data.columns) != OBSERVATION_COLUMNS:
        raise ValueError("Predictor table must contain only the four observation columns")
    if not patients.patient_id.is_unique or set(data.patient_id) != set(patients.patient_id):
        raise ValueError("Patient IDs are inconsistent")
    if set(data.feature_name) != {BIRTH, *EVENT_FEATURES}:
        raise ValueError("Expected all eight event features and Birth Date Anchor")
    if set(patients.partition) != {"train", "test"}:
        raise ValueError("Expected nonempty train and test partitions")
    events = data.feature_name.ne(BIRTH)
    if not data.loc[events, "value"].eq(1).all():
        raise ValueError("Events must have value=1")
    meta = patients.set_index("patient_id")
    birth = data.loc[~events].set_index("patient_id")
    if (not birth.index.is_unique or set(birth.index) != set(meta.index)
            or not pd.to_datetime(birth.value).eq(meta.birth.reindex(birth.index)).all()
            or not birth.time.eq(meta.birth.reindex(birth.index)).all()):
        raise ValueError("Invalid birth anchors")
    if not (data.time.ge(data.patient_id.map(meta.birth))
            & data.time.le(data.patient_id.map(meta.followup_end))).all():
        raise ValueError("Event outside follow-up")
    if not meta.followup_end.eq(meta.birth + pd.DateOffset(months=30)).all():
        raise ValueError("Follow-up must be 30 calendar months")
    fish = data.loc[data.feature_name.eq(FISH)].set_index("patient_id").time
    if (not fish.index.is_unique or set(fish.index) != set(meta.index[meta.documented])
            or not fish.eq(meta.fish_time.reindex(fish.index)).all()
            or meta.loc[~meta.documented, "fish_time"].notna().any()):
        raise ValueError("Fish documentation/timestamps are inconsistent")
    truth = recompute_r(data, patients)
    if not truth.equals(patients.R):
        raise ValueError("R disagrees with observed events")
    if not patients.fa_probability.eq(np.where(truth.eq(1), .9, .05)).all():
        raise ValueError("Incorrect FA sampling probabilities")
    if not patients.FA.isin([0, 1]).all():
        raise ValueError("FA must be binary")
    symptoms = data.loc[data.feature_name.isin(SYMPTOMS)]
    if symptoms.time.eq(symptoms.patient_id.map(fish)).any():
        raise ValueError("Exact fish/symptom simultaneity is outside this benchmark")
    signatures = _history_signatures(data, patients)
    nonfish = tuple(f for f in EVENT_FEATURES if f != FISH)
    for _, group in patients.groupby("block", sort=False):
        documented = group.loc[group.documented]
        if (len(group) != 10 or len(documented) != 8 or documented.R.sum() != 4
                or group.partition.nunique() != 1 or group.birth.nunique() != 1):
            raise ValueError("Invalid matched block or split leakage")
        for feature in EVENT_FEATURES:
            hist = [Counter(signatures[pid][feature] for pid in
                            documented.loc[documented.R.eq(r), "patient_id"])
                    for r in (0, 1)]
            if hist[0] != hist[1]:
                raise ValueError(f"Marginal history mismatch: {feature}")
        histories = [Counter(tuple(signatures[pid][f] for f in nonfish) for pid in
                             documented.loc[documented.R.eq(r), "patient_id"])
                     for r in (0, 1)]
        if histories[0] != histories[1]:
            raise ValueError("Joint non-fish history mismatch")
        undoc = Counter(tuple(signatures[pid][f] for f in nonfish)
                        for pid in group.loc[~group.documented, "patient_id"])
        if Counter({key: n * 2 for key, n in undoc.items()}) != histories[0]:
            raise ValueError("Non-fish histories differ by documentation status")
    return dict(passed=True, patients=len(patients), blocks=patients.block.nunique(),
                observations=len(data), marginal_history_mismatches=0,
                joint_nonfish_history_mismatches=0,
                documentation_nonfish_history_mismatches=0)


def generate_data(config: GeneratorConfig = GeneratorConfig()) -> tuple:
    """Build matched blocks in memory; label only after all observations exist."""
    config.validate()
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(config.seed).spawn(6)]
    birth_rng, symptom_rng, fish_rng, distractor_rng, split_rng, label_rng = streams
    test_blocks = set(split_rng.choice(config.blocks,
                                      round(config.blocks * config.test_fraction), replace=False))
    rows, children, attempts = [], [], []
    for block in range(config.blocks):
        birth = pd.Timestamp("2010-01-01") + pd.Timedelta(days=int(birth_rng.integers(0, 1826)))
        end = birth + pd.DateOffset(months=30)
        histories, (a, b), attempt = _matched_histories(symptom_rng, fish_rng, birth, end, config)
        attempts.append(attempt)
        for history in histories:
            history.extend(distractor_history(distractor_rng, birth, end, config))
        # Every complete non-fish history occurs twice per R group and once
        # undocumented. No designated signal is added; no timestamp is edited.
        profiles = [(0, a), (1, b), (0, b), (1, a)] * 2 + [(0, None), (1, None)]
        # Avoid encoding R in the profile suffix or first-row ordering.
        for profile, index in enumerate(birth_rng.permutation(len(profiles)), 1):
            history_id, fish = profiles[index]
            pid = f"N{block:04d}_{profile:02d}"
            children.append(dict(patient_id=pid, block=block, profile=profile,
                                 history_id=history_id, partition="test" if block in test_blocks else "train",
                                 documented=fish is not None, birth=birth, followup_end=end,
                                 a=a, b=b, fish_time=fish))
            rows.append((pid, birth, BIRTH, birth))
            rows.extend((pid, time, feature, 1) for feature, time in histories[history_id])
            if fish is not None:
                rows.append((pid, fish, FISH, 1))
    data = pd.DataFrame(rows, columns=OBSERVATION_COLUMNS)
    rank = {feature: index for index, feature in enumerate(FEATURE_ORDER)}
    # Feature-major order also preserves enumeration when a rare feature is
    # absent from the first patient. Birth rows establish stable patient order.
    data = data.sort_values(["feature_name", "patient_id", "time"], kind="stable",
                            key=lambda s: s.map(rank) if s.name == "feature_name" else s).reset_index(drop=True)
    patients = pd.DataFrame(children)
    patients["R"] = recompute_r(data, patients)
    patients["fa_probability"] = np.where(patients.R.eq(1), .9, .05)
    patients["FA"] = label_rng.binomial(1, patients.fa_probability)
    validation = validate_dataset(data, patients)
    validation.update(version=VERSION, config=asdict(config),
                      feature_order=list(FEATURE_ORDER), matching=dict(
        attempted_history_pairs=sum(attempts), accepted_blocks=config.blocks,
        rejected_history_pairs=sum(attempts) - config.blocks,
        acceptance_fraction=config.blocks / sum(attempts),
        max_attempts_observed=max(attempts)),
        label_mechanism="Independent Bernoulli(0.90 if R=1 else 0.05)")
    return data, patients, validation


def _auc(labels, scores):
    return float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None


def descriptive_audit(data: pd.DataFrame, patients: pd.DataFrame) -> dict:
    """Describe marginal and joint fixed-age signals; no fitting or predictions."""
    meta = patients.set_index("patient_id")
    event = data.loc[data.feature_name.ne(BIRTH)].copy()
    event["age_days"] = (event.time - event.patient_id.map(meta.birth)).dt.total_seconds() / 86400
    counts = event.groupby(["patient_id", "feature_name"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=meta.index, columns=EVENT_FEATURES, fill_value=0).fillna(0)
    result = {"interpretation": "AUCs are raw descriptive scores, not fitted models. "
              "Joint signature checks use block membership only for auditing, not as a predictor.",
              "cohorts": {}}
    for partition in ("all", "train", "test"):
        part = meta if partition == "all" else meta.loc[meta.partition.eq(partition)]
        doc = part.loc[part.documented]
        groups = {}
        for r in (0, 1):
            ids = doc.index[doc.R.eq(r)]
            obs = event.loc[event.patient_id.isin(ids)]
            groups[str(r)] = dict(n=len(ids), feature_mean_counts=counts.loc[ids].mean().to_dict(),
                                 feature_count_sd=counts.loc[ids].std().to_dict(),
                                 feature_event_mean_age_days=obs.groupby("feature_name").age_days.mean().to_dict(),
                                 observation_density_mean=float(counts.loc[ids].sum(axis=1).mean()),
                                 followup_days_mean=float((doc.loc[ids].followup_end-doc.loc[ids].birth).dt.days.mean()))
        # Fixed-age cumulative counts accessible without dynamic anchors.
        records = event.loc[event.patient_id.isin(doc.index)]
        matrices = []
        for month in range(1, 23):
            ends = doc.birth + pd.DateOffset(months=month)
            selected = records.loc[records.time.le(records.patient_id.map(ends))]
            matrix = selected.groupby(["patient_id", "feature_name"]).size().unstack(fill_value=0)
            matrix = matrix.reindex(index=doc.index, columns=EVENT_FEATURES, fill_value=0).fillna(0)
            matrix.columns = [f"{f}@{month}" for f in matrix.columns]
            matrices.append(matrix)
        fixed_age = pd.concat(matrices, axis=1)
        joint = {}
        for name, columns in (("nonfish", [c for c in fixed_age if not c.startswith(FISH + "@")]),
                              ("fish_and_symptoms", [c for c in fixed_age if c.split("@")[0] in (FISH, *SYMPTOMS)]),
                              ("all_features", list(fixed_age))):
            signatures = fixed_age[columns].apply(lambda row: tuple(row), axis=1)
            table = pd.DataFrame(dict(block=doc.block, R=doc.R, signature=signatures))
            mixed = table.groupby(["block", "signature"]).R.transform("nunique").eq(2)
            joint[name] = dict(children_with_opposite_R_signature=int(mixed.sum()), denominator=len(doc))
        fish_age = (doc.fish_time-doc.birth).dt.total_seconds()/86400
        symptom = data.loc[data.feature_name.isin(SYMPTOMS) & data.patient_id.isin(doc.index)]
        starts = symptom.patient_id.map(doc.fish_time)
        ends = symptom.patient_id.map(doc.fish_time + pd.DateOffset(months=1))
        qualifying = symptom.loc[symptom.time.gt(starts) & symptom.time.le(ends)]
        qc = qualifying.groupby("patient_id").size().reindex(doc.index, fill_value=0)
        baseline = {f"count_{f}": _auc(doc.R, counts.loc[doc.index, f]) for f in EVENT_FEATURES}
        baseline.update(fish_age=_auc(doc.R, fish_age),
                        observation_density=_auc(doc.R, counts.loc[doc.index].sum(axis=1)))
        fixed_aucs = [_auc(doc.R, fixed_age[c]) for c in fixed_age]
        result["cohorts"][partition] = dict(
            groups=groups, count_age_density_R_auc=baseline,
            fixed_age_single_score_R_auc_range=[min(fixed_aucs), max(fixed_aucs)],
            joint_fixed_age_signatures=joint,
            background_response_windows=dict(positive_children=int(qc.gt(0).sum()),
                                             multiple_symptom_children=int(qc.gt(1).sum()),
                                             qualifying_symptom_rows=int(qc.sum())),
            documentation=dict(n=len(part), documented=int(part.documented.sum()),
                               FA_prevalence=part.groupby("documented").FA.mean().to_dict(),
                               FA_auc=_auc(part.FA, part.documented.astype(int)),
                               expected_documented_prevalence=.475,
                               expected_undocumented_prevalence=.05))
    return result


def save_dataset(output: Path, data, patients, validation, audit) -> None:
    """Create a new versioned directory; refuse to overwrite any directory."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    data.to_csv(output / "observations.csv", index=False)
    patients.to_csv(output / "patients_and_split.csv", index=False)
    for name, value in (("validation.json", validation), ("descriptive_audit.json", audit)):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    manifest = {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                for name in ("observations.csv", "patients_and_split.csv", "validation.json",
                             "descriptive_audit.json")}
    manifest["generator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--config", type=Path, help="JSON fields of GeneratorConfig")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; select a new directory")
    config = GeneratorConfig(**json.loads(args.config.read_text())) if args.config else GeneratorConfig()
    data, patients, validation = generate_data(config)
    audit = descriptive_audit(data, patients)
    save_dataset(args.output, data, patients, validation, audit)
    print(json.dumps(validation, indent=2))
    print(f"Saved {args.output}; no model was trained or evaluated.")


if __name__ == "__main__":
    main()
