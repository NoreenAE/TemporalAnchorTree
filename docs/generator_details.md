# Generator Details

**Both final cohorts are already included.** Run the experiments on those files
for the stored thesis inputs. Regeneration is optional and does not train models.
Both generators require a new output directory and refuse to overwrite one that
already exists. The `_regenerated` paths below are new destinations, not included
input directories.

### Experiment 1: V3 Clean FA

[create_temporal_benchmark_v3_clean_fa.py](../create_temporal_benchmark_v3_clean_fa.py)
builds 5,000 patients with longitudinal feeding patterns, growth/distractor
measurements, and separate FA/GI labels. FA depends on feeding patterns in the
first six months after birth; GI depends on symptom summaries during the final
calendar month relative to NOW. The clean-FA transformation permutes specified
distractor values while preserving the protected features and outcomes.

```bash
python -B create_temporal_benchmark_v3_clean_fa.py --output-dir data/experiment1_regenerated
```

Default simulation master seed: `20260823` for the longitudinal and GI components;
transformation seed: `20260909`. These are fixed in the current code, not CLI seed
options. The generator stages its own intermediate cohort and removes it afterward;
it requires no saved V2/V3 dataset. `--source-dir` and `--old-v3-dir` are optional
advanced inputs for existing-base and historical-preservation checks.

The new directory contains:

- `temporal_benchmark_5000.pkl`: longitudinal predictors.
- `temporal_benchmark_fa_labels_5000.pkl` and
  `temporal_benchmark_gi_labels_5000.pkl`: patient labels.
- `temporal_benchmark_ground_truth_5000.pkl` and
  `temporal_benchmark_gi_ground_truth_5000.pkl`: latent simulation ground truth.
- `temporal_benchmark_metadata_5000.json`,
  `temporal_benchmark_gi_metadata_5000.json`, `transformation_metadata.json`,
  `validation_summary.json`, `distractor_association_audit.csv`, and
  `before_after_audit.csv`: provenance and validation/audit outputs.

Included data: [data/temporal_benchmark_v3_clean_fa/](../data/temporal_benchmark_v3_clean_fa/).
The included histograms are separate plotting outputs, not required model inputs.
See [validation.md](validation.md) for the saved distractor acceptance findings.

### Experiment 2: Noisy Matched V2

[generate_fish_intro_noisy_v2.py](../generate_fish_intro_noisy_v2.py) builds 500 matched
blocks containing 5,000 synthetic children. Matched histories distinguish temporal
relationships from ordinary event summaries. The latent response `R` records
whether a GI/vomiting event occurs strictly after fish introduction and within one
calendar month; FA is independently sampled with probability 0.90 if `R=1` and
0.05 otherwise. `R` is simulation metadata, never a predictor.

```bash
python -B generate_fish_intro_noisy_v2.py --output data/experiment2_regenerated
```

Defaults: seed `42`, 500 blocks, and a saved 80/20 block split. `--config` optionally
accepts a JSON file containing fields of `GeneratorConfig`; keep the defaults to
reproduce the final cohort. Output files are `observations.csv`,
`patients_and_split.csv`, `validation.json`, `descriptive_audit.json`, and
`manifest.json`. The patient file includes labels and partition assignments; the
manifest stores file hashes.

Included data: [data/fish_intro_noisy_matched_v2/](../data/fish_intro_noisy_matched_v2/).
The final runners default to the included datasets; creating another cohort does
not redirect the Experiment 1 or robustness runners to it automatically.
