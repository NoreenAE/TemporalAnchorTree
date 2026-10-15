# Dataset Validation

## Experiment 1: V3 Clean FA

The included [validation summary](../data/temporal_benchmark_v3_clean_fa/validation_summary.json)
and [distractor association audit](../data/temporal_benchmark_v3_clean_fa/distractor_association_audit.csv)
record the final synthetic dataset checks. The predefined distractor acceptance
screen remains false, with 12 flagged sparse summaries. “Clean FA” does not mean
that every independence screen passed.

The [generator verification report](generator_verification.json) records exact
DataFrame equality for all five Experiment 1 data/label/ground-truth files.
Regenerated pickle bytes differ. Recalculated audit effects can differ by
floating-point precision; the maximum recorded AUC difference is
0.000069757914, confined to mean/sum summaries of continuous LengthCm/WeightKg
values and sensitive to ties. Flagged rows and acceptance decisions are unchanged.
Use the included originals when exact artifact identity is required.

## Experiment 2: V2 Noisy Matched

The included [validation report](../data/fish_intro_noisy_matched_v2/validation.json),
[descriptive audit](../data/fish_intro_noisy_matched_v2/descriptive_audit.json), and
[manifest](../data/fish_intro_noisy_matched_v2/manifest.json) describe the final
synthetic cohort and its saved partition. The generator verification report
records exact DataFrame equality against the saved cohort, including labels and
partitions, and identical default-generator CSV bytes across the dependency
cleanup it checked.

## Checking the included results

The [reproduction entry point](../experiments/reproduce_thesis_results.py) checks
patient membership, labels, configurations, saved scores and derived metrics.
It reads [preserved_artifacts.json](preserved_artifacts.json) to verify original
scientific artifact hashes. This file is a required reproduction input.
See [reproducibility.md](reproducibility.md) for commands, comparison criteria and
partial-reference limitations.

These checks concern the included synthetic benchmarks. They do not establish
clinical validity or suitability for medical use.
