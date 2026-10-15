# Final Experiment 1 V3 Clean FA synthetic cohort

5,000 synthetic patients, shared longitudinal X, separate FA/GI labels, and latent
simulation ground truth. Experiment 1.1 reuses this cohort and the saved main-run
split. No real patient-level clinical records are present.

The final transformation preserves IDs, timestamps, feature schemas, marginal
values/missingness, labels, and protected feeding/GI rows. Joint demographic/growth
donor sequences and food donor sequences are permuted within compatible count
strata using PCG64 seed 20260909, without label inputs. The underlying synthetic
longitudinal and FA/GI streams use master seed 20260823.

## Acceptance result

The original predefined distractor screen **failed**, with 12 flagged sparse
conditional summaries. Those NOW -18-month summaries involve only 28 observed
patients (two FA positive). No post-hoc exemption or alternative seed was used.
Late-only feeding had zero flagged summaries. These univariate effect-size checks
are not a guarantee of multivariate independence. See the two audit CSV files and
`validation_summary.json` for the actual results.

## Reproduction

From the repository root, create a separate output directory:

```bash
python -B create_temporal_benchmark_v3_clean_fa.py --output-dir data/experiment1_regenerated
```

The final generator uses shared components under `generators/`, stages intermediate
base data, and removes the staging directory. No historical dataset is required.
The five data/label/ground-truth frames were compared exactly against these final
artifacts during release validation. Serialized pickle bytes differ despite
exact frame equality. Recomputed audit correlations differ at floating-point
precision, and tie-sensitive AUC summaries differ by at most 0.000069758. Every
flagged row and acceptance decision is unchanged; historical audit CSVs remain
unmodified. See docs/generator_verification.json for the measured differences. Output-location metadata
and snapshot modification times are provenance, not deterministic data values.

The metadata records the original staged derivation; historical source digests may
refer to inputs no longer distributed. `transformation_metadata.json` identifies
the final derivative. Labels and latent ground truth are not model predictors.
