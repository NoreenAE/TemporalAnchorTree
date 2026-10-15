# Reproducibility

## Audit or reproduce the thesis experiments

[experiments/reproduce_thesis_results.py](../experiments/reproduce_thesis_results.py)
provides one entry point for checking saved results or explicitly training new
runs. It reuses the existing experiment runners and included datasets.

Read-only verification of all saved studies:

```bash
python -B experiments/reproduce_thesis_results.py check-saved
```

This checks available original artifact hashes, patient membership and labels,
recorded configurations, and recalculates held-out metrics from the saved scores.
Experiment 2 includes Brier/calibration checks; robustness includes the frozen
protocol, all completion records, and recomputed summary tables (ten records for
the original five-seed archive; eight for new four-seed runs). No model is
trained, used for prediction, or loaded from a model pickle. Included DataFrame
pickles are still read as inputs. Add `--report reproduction_report.json` to write
a **new** JSON report outside reference/data directories; existing files are
never overwritten.

Inspect the preflight for Experiment 1 without training or creating outputs:

```bash
python -B experiments/reproduce_thesis_results.py rerun --experiment experiment_1 --output experiments/results/reproduction_seed42 --dry-run
```

Explicitly train and compare Experiment 1 using both seeds set to 42:

```bash
python -B experiments/reproduce_thesis_results.py rerun --experiment experiment_1 --output experiments/results/reproduction_seed42
```

Or run all final studies (potentially several hours):

```bash
python -B experiments/reproduce_thesis_results.py rerun --experiment all --output experiments/results/reproduction_all_seed42
```

Selections are `experiment_1`, `gi_gain0`, `gi_gain0005`, `experiment_2`,
`experiment_2_1`, `cart`, or `all`. Running `cart` trains CART only and compares it
with the stored TAT scores; it does not implicitly retrain TAT. Experiment 2.1
retains its POSIX and exact-environment requirements.

The output argument must name an unused directory directly under
`experiments/results`. To retain the existing runners' output safeguards, each
study writes to an **adjacent directory with the same prefix**. For example,
`reproduction_seed42/report.json` and `reproduction_seed42/experiment_1.log` hold
the report/log, while `reproduction_seed42_experiment_1/` holds that study's new
artifacts. All these paths are under `experiments/results/`. Original runs are
never overwritten. An existing central or selected study destination is rejected.

Fresh runs are checked against the recorded configuration/split, patient scores,
JSON trees and temporal path/anchor traces where available. CART also compares
the TRAIN/TEST AUC table; no original CART JSON tree is distributed. Numeric
comparisons use absolute tolerance `1e-12` and zero relative tolerance. Runtime,
figure bytes, pickle bytes, and output-location provenance are not equality
criteria. Any mismatch produces a failed report, not an updated reference.

Exit codes are **0** for passed checks, **1** for a mismatch/error, and **2** for a
partial reference archive or a dry run. The default all-study audit currently
returns 2: the gain-0.0005 GI archive lacks its metrics JSON and tree. Its labels,
split, scores and computed AUC remain checkable, but a full structural comparison
cannot be certified against missing artifacts. A fresh complete run never fills
in or replaces those missing original references.

Scientific dataset validation findings are recorded in
[validation.md](validation.md). Original scientific artifact hashes are stored
in [preserved_artifacts.json](preserved_artifacts.json), which is required by the
reproduction runner.

## Recorded settings and new runs

| Component | Data / split | Relevant seeds and configuration |
| --- | --- | --- |
| Experiment 1 | V3 Clean FA; shared 70/30 patient split | Saved split/random and sampling seeds 42; A=0.4; gain 0.0005. Main source defaults now also use 42. The central reproduction runner verifies the recorded configuration and split before training. |
| Experiment 1.1 | Same cohort and recorded patient split | Seeds 42; NOW removed; select gain 0 or 0.0005 explicitly. |
| Experiment 2 | Noisy Matched V2; saved 80/20 matched-block split | Data and baseline seeds 42; identical inputs for ON/OFF. |
| Experiment 2.1 | Same fixed Experiment 2 cohort/split | New fits: sampling seeds 43–46. Preserved archive: 43–47. Separate seed-42 checkpoint; all other settings fixed. |
| CART | Same Experiment 2 cohort/split | Seed 42; depth 3; split/leaf 100/50. |

The included data and saved predictions are the reference artifacts. Inspecting
or plotting them reproduces an analysis without training. Training into a new
output reproduces a protocol, and requires comparing its predictions, trees,
and metrics with the saved run before claiming identical results.

Experiment 1's `config.json` and `shared_patient_split.csv`, the GI variants'
`experiment_config.json`, Experiment 2's `configuration.json`, the robustness
`protocol.json` and completion markers, and CART's `run_metadata.json` record the
respective inputs and settings. See [protocols](../docs/protocols.md).

For Experiment 2.1, new training uses only seeds 43–46. Its original saved
five-seed archive (43–47) is retained. The runner checks each archive against
its recorded seed list and compares new patient scores against the matching
archived seeds; four-seed means are not compared to five-seed means. See
[experiment_2_protocol.md](experiment_2_protocol.md) for the saved protocol,
checkpoint and resumption requirements. The earlier V2 parameter search is
results-only; see [experiment_1_protocol.md](experiment_1_protocol.md).
