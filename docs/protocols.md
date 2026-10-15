# Recorded Experiment Settings

| Component | Recorded protocol | Active source / reproduction |
| --- | --- | --- |
| Experiment 1 | V3 Clean FA; A=0.4; depth 3; split/leaf 20/10; min gain 0.0005; seeds 42; 70/30 joint split; mean/sum/max/count | Source defaults use the recorded seeds 42. The central reproduction runner checks the original configuration, split and outputs; the existing A wrapper remains available. Declared metadata in the main source omits count, although the model still receives it. |
| Experiment 1.1 | Same cohort and saved split; NOW removed; min gain 0 or 0.0005 | Explicit CLI selection; all other recorded model/evaluation settings unchanged. |
| Experiment 2 | Noisy matched V2; 500 blocks; 80/20 block split; seed 42; A=0.4; depth 3; split/leaf 100/50; min gain 0; sum only | Dynamic ON/OFF differ only by the anchor switch. |
| Experiment 2.1 | Original paired sampling seeds 43–47 on the same cohort, with seed-42 checkpoint | New fits use only 43–46. Original per-seed outputs, five-seed summaries and their protocol remain unchanged. New fits require a separate output directory. |
| CART | Static event sums; gini; depth 3; split/leaf 100/50; seed 42 | Selected updated CART run retained; original minimum-sample-default run excluded from final comparison. |
| V2 search | Earlier Experiment 1 V2 dataset; eight configurations, validation/stability/final-test stages | Results only by author decision; no old input dataset or rerunnable search is distributed. |

The clean-FA dataset's predefined association acceptance remains false with 12
flagged sparse summaries. This remains a limitation of the included synthetic benchmark.
Synthetic ground truth is retained for interpretation and generator validation,
never as a TAT predictor.

`experiments/reproduce_thesis_results.py check-saved` audits all saved studies
without fitting. Its explicit `rerun` mode writes new runs with recorded seeds,
compares individual predictions and scientific exports, and preserves references.
The gain-0.0005 GI archive lacks original metrics/tree exports, so verification
is reported as partial instead of PASS.

Experiment 2.1 saved-result checks validate the seed list recorded in each
protocol: five seeds in the original archive, four in new runs. Compatibility for
the exact original protocol permits only this runner's source fingerprint and
the planned seed list to differ from current execution; data, algorithm source,
settings and environment guards still apply. Historical completion hashes are
not rewritten. The reproduction runner fits and compares only seeds 43–46;
it checks individual predictions against the matching archived seeds, rather
than comparing four-seed averages with five-seed averages.
