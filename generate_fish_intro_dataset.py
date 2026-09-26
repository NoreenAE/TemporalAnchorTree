"""Generate the matched fish-introduction dataset; never train or evaluate.

500 blocks: 400 ordinary, 50 upper-endpoint, 50 one-second boundary cases.
Within each scenario, signal types are 40% GI, 40% vomiting, 20% both.
Each block contains four documented R=1, four documented R=0 and two
undocumented children. R is validated before independent FA sampling.
Five independent distractors use Poisson(6) counts and uniform event times
throughout follow-up. They do not contribute to R or FA sampling.
The revised v2 dataset renames Fever to MinorInjury and Rash to
RoutineVaccination only. Draw order and feature encounter order are preserved.
Shared background symptoms remain outside both candidate response windows:
allowing them inside would flip matched R=0 profiles under the unchanged rule.
Existing dataset files are never overwritten. No execution occurs on import.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data/fish_intro_dataset_distractors_v2"

BIRTH = "Birth Date Anchor"
FISH = "FishIntroduction"
SYMPTOMS = ("GI", "Vomiting")
# Keep positions unchanged: each feature receives the same RNG draws as before.
DISTRACTORS = (
    "MinorInjury", "Cough", "RoutineVaccination", "SleepDisturbance", "DoctorVisit",
)
# Preserve the former lexical encounter order under a naming-only revision.
# TemporalTree enumerates features in encounter order, affecting sampled groups
# and exact-score ties. This order is fixed independently of outcomes.
FEATURE_ORDER = (
    BIRTH, "Cough", "DoctorVisit", "MinorInjury", FISH, "GI",
    "RoutineVaccination", "SleepDisturbance", "Vomiting",
)
SEED = 42
SECOND = pd.Timedelta(seconds=1)


def require(condition: bool, message: str) -> None:
    """Validate the dataset even when Python assertions are disabled."""
    if not condition:
        raise ValueError(message)


def random_time(rng, start: pd.Timestamp, end: pd.Timestamp) -> pd.Timestamp:
    """Sample an integer second from the closed interval."""
    return start + pd.Timedelta(seconds=int(rng.integers(
        0, int((end - start).total_seconds()) + 1
    )))


def generate_distractors(patients: pd.DataFrame, rng) -> pd.DataFrame:
    """Sample independent binary events using only IDs and follow-up bounds."""
    rows = []
    for child in patients[["patient_id", "birth", "followup_end"]].itertuples(index=False):
        for feature in DISTRACTORS:
            for _ in range(int(rng.poisson(lam=6))):
                rows.append(dict(
                    patient_id=child.patient_id,
                    time=random_time(rng, child.birth, child.followup_end),
                    feature_name=feature, value=1,
                ))
    return pd.DataFrame(rows, columns=["patient_id", "time", "feature_name", "value"])


def validate_observations(data: pd.DataFrame, patients: pd.DataFrame) -> pd.Series:
    """Recompute truth from all events; fail rather than repair any mismatch."""
    require(len(patients) == 5000, "Expected 5,000 children")
    require(patients.patient_id.is_unique, "Duplicate child IDs")
    require(set(data.patient_id) == set(patients.patient_id), "Patient mismatch")
    require(set(data.feature_name) == {BIRTH, FISH, *SYMPTOMS, *DISTRACTORS},
            "Unexpected feature")
    info = patients.set_index("patient_id")
    relations = {}
    for pid, rows in data.groupby("patient_id", sort=False):
        child = info.loc[pid]
        birth = rows.loc[rows.feature_name.eq(BIRTH)]
        fish = rows.loc[rows.feature_name.eq(FISH), "time"]
        symptoms = rows.loc[rows.feature_name.isin(SYMPTOMS), "time"]
        require(len(birth) == 1 and birth.value.iloc[0] == child.birth,
                f"{pid}: invalid Birth anchor")
        require(rows.time.between(child.birth, child.followup_end).all(),
                f"{pid}: outside follow-up")
        require(len(fish) == int(child.documented), f"{pid}: fish count")
        require(rows.loc[~rows.feature_name.eq(BIRTH), "value"].eq(1).all(),
                f"{pid}: nonbinary event")
        require(all(rows.feature_name.eq(s).any() for s in SYMPTOMS),
                f"{pid}: missing background symptom type")
        if child.documented:
            require(fish.iloc[0] == child.fish_time, f"{pid}: fish profile mismatch")
        for symptom in SYMPTOMS:
            observed = rows.loc[rows.feature_name.eq(symptom), "time"]
            signal = observed.eq(child.signal_time)
            require(int(signal.sum()) == int(symptom in child.signal_type.split("+")),
                    f"{pid}: designated symptom mismatch")
            background = observed.loc[~signal]
            require(1 <= len(background) <= 3, f"{pid}: background event count")
            require(not (background.between(child.a, child.a + pd.DateOffset(months=1))
                         | background.between(child.b, child.b + pd.DateOffset(months=1))).any(),
                    f"{pid}: background inside introduction window")
        require(not symptoms.isin([child.a, child.b]).any(),
                f"{pid}: symptom coincides with introduction")
        relations[pid] = int(bool(len(fish)) and bool((
            (symptoms > fish.iloc[0])
            & (symptoms <= fish.iloc[0] + pd.DateOffset(months=1))
        ).any()))

    actual = patients.patient_id.map(relations).astype(int)
    for block, group in patients.assign(R=actual).groupby("block"):
        require(len(group) == 10 and sorted(group.profile) == list(range(1, 11)),
                f"Block {block}: profiles")
        require(group.partition.nunique() == 1, f"Block {block}: split leakage")
        require(group.loc[group.documented, "R"].sum() == 4
                and group.documented.sum() == 8
                and group.loc[~group.documented, "R"].sum() == 0,
                f"Block {block}: observed R violates 4/4/2")
        require(group.R.tolist() == [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                f"Block {block}: observed R disagrees with pairing")
        # Equal marginal timestamps/counts within documented R groups.
        block_rows = data.loc[data.patient_id.isin(group.patient_id)]
        for symptom in SYMPTOMS:
            backgrounds = []
            for child in group.itertuples(index=False):
                backgrounds.append(sorted(block_rows.loc[
                    block_rows.patient_id.eq(child.patient_id)
                    & block_rows.feature_name.eq(symptom)
                    & ~block_rows.time.eq(child.signal_time), "time"].tolist()))
            require(all(times == backgrounds[0] for times in backgrounds),
                    f"Block {block}: unmatched {symptom} background")
        for feature in (FISH, *SYMPTOMS):
            signatures = []
            for relation in (0, 1):
                ids = group.loc[group.documented & group.R.eq(relation), "patient_id"]
                signatures.append(sorted(block_rows.loc[
                    block_rows.patient_id.isin(ids)
                    & block_rows.feature_name.eq(feature), "time"
                ].tolist()))
            require(signatures[0] == signatures[1],
                    f"Block {block}: {feature} marginal imbalance")
    require(patients.partition.value_counts().to_dict() == {"train": 4000, "test": 1000},
            "Expected 400/100 block split")
    return actual


def generate_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Generate matched events, validate observed R, then independently draw FA."""
    # The first three child streams are unchanged; only child 3 is new.
    event_seed, split_seed, label_seed, distractor_seed = np.random.SeedSequence(SEED).spawn(4)
    rng, split_rng, label_rng = [np.random.default_rng(s) for s in
                               (event_seed, split_seed, label_seed)]
    distractor_rng = np.random.default_rng(distractor_seed)
    rows, children = [], []
    block = 0
    for scenario, total in (("ordinary", 400), ("endpoint", 50), ("boundary", 50)):
        for signal_types, fraction in ((("GI",), .4), (("Vomiting",), .4), (SYMPTOMS, .2)):
            count = int(total * fraction)
            test_blocks = set(split_rng.choice(range(block, block + count),
                                               count // 5, replace=False))
            for _ in range(count):
                birth = pd.Timestamp("2010-01-01") + pd.Timedelta(
                    days=int(rng.integers(0, 1826)))
                end = birth + pd.DateOffset(months=30)
                a = random_time(rng, birth + pd.Timedelta(days=15),
                                birth + pd.DateOffset(months=22))
                a_end = a + pd.DateOffset(months=1)
                b = a + pd.DateOffset(months=1 if scenario == "boundary" else 2)
                b_end = b + pd.DateOffset(months=1)
                if scenario == "boundary":
                    u, v = b - SECOND, b + SECOND
                elif scenario == "endpoint":
                    u, v = a_end, b_end
                else:
                    u = random_time(rng, a + SECOND, a_end - SECOND)
                    v = random_time(rng, b + SECOND, b_end - SECOND)
                require(a < u <= a_end < v and u < b < v <= b_end,
                        f"Block {block}: timing inequalities")
                background = []
                for feature in SYMPTOMS:
                    for _ in range(int(rng.integers(1, 4))):
                        while True:
                            time = random_time(rng, birth, end)
                            if not (a <= time <= a_end or b <= time <= b_end):
                                break
                        background.append((feature, time))
                profiles = [(a, u)] * 2 + [(b, v)] * 2 + [(a, v)] * 2 + [(b, u)] * 2
                profiles += [(None, u), (None, v)]
                for profile, (fish, signal) in enumerate(profiles, start=1):
                    pid = f"F{block:03d}_{profile:02d}"
                    children.append(dict(
                        patient_id=pid, block=block, profile=profile,
                        scenario=scenario, signal_type="+".join(signal_types),
                        partition="test" if block in test_blocks else "train",
                        documented=fish is not None, birth=birth, followup_end=end,
                        a=a, b=b, signal_time=signal, fish_time=fish,
                    ))
                    events = [(BIRTH, birth, birth)]
                    if fish is not None:
                        events.append((FISH, fish, 1))
                    events.extend((feature, time, 1) for feature, time in background)
                    events.extend((feature, signal, 1) for feature in signal_types)
                    for feature, time, value in events:
                        rows.append(dict(patient_id=pid, time=time,
                                         feature_name=feature, value=value))
                block += 1
    data, patients = pd.DataFrame(rows), pd.DataFrame(children)
    data = pd.concat([data, generate_distractors(patients, distractor_rng)],
                     ignore_index=True)
    # Sort using stable feature identities rather than the revised display names.
    feature_rank = {name: index for index, name in enumerate(FEATURE_ORDER)}
    data = data.sort_values(
        ["patient_id", "feature_name", "time"], kind="stable",
        key=lambda column: column.map(feature_rank)
        if column.name == "feature_name" else column,
    )
    data = data.reset_index(drop=True)
    patients["R"] = validate_observations(data, patients)
    patients["fa_probability"] = np.where(patients.R.eq(1), .90, .05)
    patients["FA"] = label_rng.binomial(1, patients.fa_probability)
    validation = dict(
        passed=True, patients=len(patients), observations=len(data), blocks=block,
        documented_R1=int((patients.documented & patients.R.eq(1)).sum()),
        documented_R0=int((patients.documented & patients.R.eq(0)).sum()),
        undocumented=int((~patients.documented).sum()),
        actual_FA_cases=int(patients.FA.sum()), expected_FA_cases=1950,
        distractors=dict(features=list(DISTRACTORS), poisson_mean=6,
                         timestamp_distribution="uniform seconds over full follow-up",
                         seed_spawn_key=list(distractor_seed.spawn_key)),
        checks=["actual timestamp R before FA sampling", "4/4/2 in every block",
                "matched documented marginal event times and counts",
                "no simultaneous fish/symptom events", "30-month follow-up",
                "background GI and vomiting in every child", "intact split blocks"],
    )
    return data, patients, validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    files = ("observations.csv", "patients_and_split.csv", "validation.json")
    require(not any((args.output / name).exists() for name in files),
            "Dataset files already exist; choose a new --output directory")
    data, patients, validation = generate_data()
    args.output.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output / "observations.csv", index=False)
    patients.to_csv(args.output / "patients_and_split.csv", index=False)
    (args.output / "validation.json").write_text(
        json.dumps(validation, indent=2, allow_nan=False) + "\n"
    )
    print(f"Validated dataset: {validation}", flush=True)
    print(f"Saved dataset: {args.output}", flush=True)


if __name__ == "__main__":
    main()
