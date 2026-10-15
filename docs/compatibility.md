# Compatibility and Known Limitations

## Loading saved models

Import the algorithm from `TAT`. Keep [settree.py](../settree.py) alongside it:
14 saved Experiment 2 models reference the former package path. This bridge
permits ordinary `pickle.load` without rewriting the saved models. New
serialization uses `TAT` paths.

## Experiment 1 and GI without NOW

Experiment 1 receives mean/sum/max/count aggregations, although the main source's
declared aggregation list and older metadata omit count. Configuration output
records `effective_aggregations`; the A wrapper preserves all four functions.

Experiment 1/1.1 include JSON trees and scores where available. Historical `.dill`
closures are not distributed. The gain-0.0005 GI archive is partial: its original
metrics/tree exports are absent. See [reproducibility.md](reproducibility.md) for
what can and cannot be compared against that archive.

The recorded Experiment 1 confusion matrices contain no positive predictions at
threshold 0.5; their ROC-AUC values assess continuous-score ranking.

## Generator reproducibility

Regenerated Experiment 1 DataFrames match the included originals, while pickle
bytes and some floating audit values differ. The 12 flagged distractor summaries
remain flagged. See [validation.md](validation.md) and the unchanged
[generator verification report](generator_verification.json) for the numerical
findings.

## Robustness environment and saved protocols

The robustness runner requires POSIX locking and the recorded Python/package
versions. Source/version guards reject changed model code or mixed environments.
A narrowly scoped compatibility check accepts the exact original five-seed
protocol for saved-result verification and summary reconstruction; it cannot be
used for new fits. New fits use seeds 43–46 in a separate output directory, while
the original 43–47 archive remains unchanged. See
[experiment_2_protocol.md](experiment_2_protocol.md) for execution requirements.
