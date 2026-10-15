# Temporal Anchor Tree (TAT)

Temporal Anchor Tree (TAT) is an interpretable decision-tree framework for
longitudinal data. TAT evaluates features within temporal windows defined relative
to fixed or patient-specific reference events.

This repository accompanies the MSc thesis and contains the TAT implementation,
synthetic-data generators, final synthetic datasets, experiment runners, EDA
notebooks, and saved experimental results.

All patient-level datasets included in this repository are synthetic.

## Repository Structure

```text
TemporalAnchorTree/
├── TAT/                  # Core TAT implementation
├── generators/           # Synthetic-data generators
├── data/                 # Final synthetic datasets
├── experiments/          # Experiment runners and saved results
├── notebooks/            # EDA notebooks
├── supporting_analyses/  # Additional thesis analyses
├── docs/                 # Detailed documentation
├── requirements.txt
└── README.md
```

## Installation

Python 3.9.5 is the verified research environment.

```bash
git clone https://github.com/NoreenAE/TemporalDecisionTree.git TemporalAnchorTree
cd TemporalAnchorTree
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

See [installation details](docs/installation.md) for platform-specific instructions.

## Synthetic Datasets

### Experiment 1 – V3 Clean FA

This dataset contains 5,000 synthetic patients with longitudinal feeding,
gastrointestinal, and background clinical observations. Separate food-allergy
(FA) and gastrointestinal (GI) outcomes depend on predefined temporal relationships.

Generator: [create_temporal_benchmark_v3_clean_fa.py](create_temporal_benchmark_v3_clean_fa.py)

Included data: [data/temporal_benchmark_v3_clean_fa/](data/temporal_benchmark_v3_clean_fa/)

### Experiment 2 – V2 Noisy Matched

This dataset contains 5,000 synthetic children in a matched design. The relevant
temporal signal depends on GI or vomiting events occurring relative to each
child's `FishIntroduction` event.

Generator: [generate_fish_intro_noisy_v2.py](generate_fish_intro_noisy_v2.py)

Included data: [data/fish_intro_noisy_matched_v2/](data/fish_intro_noisy_matched_v2/)

## Experiments

| Experiment | Purpose | Main runner |
| --- | --- | --- |
| Experiment 1 | Temporal ground-truth evaluation for FA and GI | [experiment_1_fa_vs_gi.py](experiments/experiment_1_fa_vs_gi.py) |
| Experiment 1.1 | GI prediction without the NOW anchor | [experiment_1_1_gi_without_now.py](experiments/experiment_1_1_gi_without_now.py) |
| Experiment 2 | Dynamic anchors ON vs. OFF | [experiment_2_noisy_v2.py](experiments/experiment_2_noisy_v2.py) |
| Experiment 2.1 | Candidate-sampling robustness | [experiment_2_1_candidate_sampling_robustness.py](experiments/experiment_2_1_candidate_sampling_robustness.py) |
| CART baseline | Conventional non-temporal tree comparison | [experiment_2_cart_baseline.py](experiments/experiment_2_cart_baseline.py) |

Saved outputs are available under [experiments/results/](experiments/results/).

## Main Results

Main saved results are available under [experiments/results/](experiments/results/).
The repository includes ROC curves, learned tree visualizations, temporal-reference
analyses, robustness summaries, and CART baseline results corresponding to the
thesis experiments. See [results and visualizations](docs/results.md) for details.

## Exploratory Data Analysis

EDA notebooks are provided for both final synthetic datasets:

- Experiment 1: [notebooks/EDA_experiment_1_v3_clean_fa.ipynb](notebooks/EDA_experiment_1_v3_clean_fa.ipynb)
- Experiment 2: [experiments/EDA_experiment_2_noisy_v2.ipynb](experiments/EDA_experiment_2_noisy_v2.ipynb)

## Reproducibility

Final synthetic datasets, train/test partitions, random seeds, experiment
configurations, and saved outputs are included in the repository. Use
[experiments/reproduce_thesis_results.py](experiments/reproduce_thesis_results.py)
to check saved results or rerun the thesis experiments; see the
[reproduction guide](docs/reproducibility.md) for instructions.

## Citation and Use

This repository accompanies the MSc thesis on the Temporal Anchor Tree framework.
Citation information will be added with the final thesis record.
All patient-level datasets included in this repository are synthetic.

This project is licensed under the [MIT License](LICENSE). See
[citation and licensing details](docs/citation_and_use.md).
