from datetime import datetime
import json
from pathlib import Path
import time

import dill
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from settree.eval_utils import evaluate_TemporalTree, plot_tree_structure
from settree.temporal_explainability import (
    derive_monthly_bin_edges,
    run_temporal_explainability,
)
from settree.temporal_tree import TemporalTree
from settree.temporal_tree_ensemble import TemporalTreeEnsemble


PLOT_TREE = False  # Set to True to plot the tree structure during evaluation
VERBOSE = 1
# Training verbosity:
# 0 = Silent: no training/debug output.
# 1 = Normal (recommended): node progress, selected splits, branch sizes, leaf reasons, and training summary.
# 2 = Progress monitoring: level 1 + split-search progress every 1,000 candidates.
# 3 = Full debug: level 2 + detailed internal messages, including aggregation [AGG] output.
RUN_NAME = "coin_flip_temporal_fa_benchmark_5000"
RUN_DESCRIPTION = " depth = 3 run benchmark on temporal_fa_benchmark_5000.pkl dataset with 3 windows , prove that the runtime optimization with coin flip works"
RANDOM_STATE = 42
USE_DYNAMIC_ANCHORS = False
ANCHOR_OFFSET_MAP = { # here we define the anhor names and offsets
    "Birth Date Anchor": [1, 2, 3, 4, 5, 6],
    # "NOW Anchor": [-1, -2, -3, -4, -5, -6],
}
RUN_TEMPORAL_EXPLAINABILITY = True
TEMPORAL_EXPLAINABILITY_BIN_EDGES = derive_monthly_bin_edges(
    ANCHOR_OFFSET_MAP,
)
# Single-tree split-group sampling configuration.
SPLIT_SAMPLING_SEED = 42

# Ensemble configuration. These values are used only when USE_ENSEMBLE is True.
USE_ENSEMBLE = False
N_ESTIMATORS = 5
BASE_ENSEMBLE_SPLIT_SAMPLING_SEED = 42
ENSEMBLE_SPLIT_SAMPLING_SEEDS = None

TEST_SIZE = 0.3
OUTPUT_ROOT = Path("outputs/runs")
DATA_PATH = Path("temporal_fa_benchmark_5000.pkl")
LABELS_PATH = Path("temporal_fa_benchmark_labels_5000.pkl")
MODEL_CONFIG = {
    "max_depth": 3,
    "min_samples_split": 1,
    "min_samples_leaf": 1,
    "anchors_list": list(ANCHOR_OFFSET_MAP.keys()),#["Birth Date Anchor"], #, "NOW Anchor"
    "anchor_offset_list": [None, 1, 2, 3, 4, 6, 9, 12],
    "is_classification": False,
    "criterion": "mse",
    "learn_missing_direction": False,
    "candidate_sampling_rate": 0.1, #A
    "split_sampling_seed": SPLIT_SAMPLING_SEED,
}


def make_run_dir(run_name, output_root=OUTPUT_ROOT):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_name = run_name or f"run_{timestamp}"
    run_dir = output_root / base_name
    if run_dir.exists():
        run_dir = output_root / f"{base_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def build_config(run_description, dataset_path):
    return {
        "description": run_description,
        "dataset": str(dataset_path),
        "random_state": RANDOM_STATE,
        "max_depth": MODEL_CONFIG["max_depth"],
        "min_samples_split": MODEL_CONFIG["min_samples_split"],
        "min_samples_leaf": MODEL_CONFIG["min_samples_leaf"],
        "anchors_list": MODEL_CONFIG["anchors_list"],
        "anchor_offset_list": MODEL_CONFIG["anchor_offset_list"],
        "anchor_offset_map": ANCHOR_OFFSET_MAP,
        "run_temporal_explainability": RUN_TEMPORAL_EXPLAINABILITY,
        "temporal_explainability_bin_edges": (
            TEMPORAL_EXPLAINABILITY_BIN_EDGES
        ),
        "criterion": MODEL_CONFIG["criterion"],
        "is_classification": MODEL_CONFIG["is_classification"],
        "use_dynamic_anchors": USE_DYNAMIC_ANCHORS,
        "candidate_sampling_rate": MODEL_CONFIG["candidate_sampling_rate"],
        "split_sampling_seed": MODEL_CONFIG["split_sampling_seed"],
        "use_ensemble": USE_ENSEMBLE,
        "ensemble": {
            "n_estimators": N_ESTIMATORS,
            "base_split_sampling_seed": BASE_ENSEMBLE_SPLIT_SAMPLING_SEED,
            "split_sampling_seeds": ENSEMBLE_SPLIT_SAMPLING_SEEDS,
        } if USE_ENSEMBLE else None,
        "test_size": TEST_SIZE,
    }


def write_json(data, path):
    path.write_text(json.dumps(data, indent=2, default=str))


def save_metrics(results, runtime_minutes, artifact_paths, output_dir):
    metrics = {
        "accuracy": float(results["accuracy"]),
        "precision": float(results["precision"]),
        "recall": float(results["recall"]),
        "auc_roc": float(results["auc_roc"]) if "auc_roc" in results else None,
        "runtime_minutes": runtime_minutes,
        "artifacts": {
            "model": str(artifact_paths["model"]),
            "tree_json": str(artifact_paths["tree_json"]) if artifact_paths["tree_json"] is not None else None,
            "tree_png": str(artifact_paths["tree_png"]) if artifact_paths["tree_png"] is not None else None,
            "config": str(artifact_paths["config"]),
        },
    }
    metrics_path = output_dir / "metrics.json"
    write_json(metrics, metrics_path)
    return metrics_path


def load_data(data_path, labels_path):
    if str(data_path).lower().endswith('.pkl'):
        data = pd.read_pickle(data_path)
    else:
        data = pd.read_csv(data_path)

    data["time"] = pd.to_datetime(data["time"], errors="coerce")
    invalid_time_count = data["time"].isna().sum()

    if invalid_time_count > 0:
        raise ValueError(
            f"Found {invalid_time_count} invalid timestamps in the 'time' column."
        )

    data.loc[data['feature_name'] == 'Birth Date', 'value'] = pd.to_datetime(
        data.loc[data['feature_name'] == 'Birth Date', 'value']
    )
    leak_features = [
        'Contact dermatitis',
        'Allergy education',
        'Application of moisturizer to skin',
        'Skin condition care',
        'Eczema'
    ]
    data.loc[data['feature_name'] == 'Birth Date', 'feature_name'] = 'Birth Date Anchor'

    data = data.loc[~data['feature_name'].isin(leak_features)]
    # Preserve birth-anchor values as timestamps while converting other features to numeric.
    anchor_mask = data['feature_name'].eq('Birth Date Anchor')
    data.loc[anchor_mask, 'value'] = pd.to_datetime(data.loc[anchor_mask, 'value'])
    data.loc[~anchor_mask, 'value'] = pd.to_numeric(data.loc[~anchor_mask, 'value'], errors='coerce')

    if str(labels_path).lower().endswith('.pkl'):
        labels_df = pd.read_pickle(labels_path)
    else:
        labels_df = pd.read_csv(labels_path)

    labels_df = labels_df[labels_df['patient_id'].isin(list(data['patient_id'].unique()))]
    return data, labels_df


def split_data(data, labels_df):
    n_all = len(data['patient_id'].unique())
    print(f'number of patients in the data: {n_all}')

    duplicate_label_patients = labels_df.loc[
        labels_df["patient_id"].duplicated(), "patient_id"
    ].unique()
    if len(duplicate_label_patients) > 0:
        raise ValueError(
            "labels_df must contain exactly one label per patient. "
            f"Found duplicate labels for {len(duplicate_label_patients)} patients."
        )

    data_patients = set(data["patient_id"].unique())
    label_patients = set(labels_df["patient_id"].unique())
    missing_data_labels = data_patients - label_patients
    if missing_data_labels:
        raise ValueError(
            "Every patient in the event data must have a label. "
            f"Missing labels for {len(missing_data_labels)} patients."
        )

    # Split train - test datasets.

    # Stratified split to maintain label distribution
    train_patients, test_patients = train_test_split(
        labels_df["patient_id"],
        test_size=TEST_SIZE,
        stratify=labels_df["label"],  # Ensures class balance
        random_state=RANDOM_STATE
    )
    labels_by_patient = labels_df.set_index("patient_id")

    assert set(train_patients).isdisjoint(test_patients)

    # Create train and test sets.
    train_data = data[data["patient_id"].isin(train_patients)].copy()
    test_data = data[data["patient_id"].isin(test_patients)].copy()

    train_patient_order = train_data["patient_id"].drop_duplicates().to_numpy()
    test_patient_order = test_data["patient_id"].drop_duplicates().to_numpy()

    for split_name, patient_order in {
        "train": train_patient_order,
        "test": test_patient_order,
    }.items():
        missing_labels = set(patient_order) - set(labels_by_patient.index)
        if missing_labels:
            raise ValueError(
                f"Every {split_name} patient must have a label. "
                f"Missing labels for {len(missing_labels)} {split_name} patients."
            )

    y_train = labels_by_patient.loc[train_patient_order, "label"].reset_index(drop=True)
    y_test = labels_by_patient.loc[test_patient_order, "label"].reset_index(drop=True)

    n_train = len(train_data['patient_id'].unique())
    print(f'number of patients in the train: {n_train}')
    print(f'Precentage of label 1 in train: {y_train.sum()*100/n_train}')
    n_test = len(test_data['patient_id'].unique())
    print(f'number of patients in the test: {n_test}')
    return train_data, test_data, y_train, y_test


def build_model():
    """Construct either the configured single tree or ensemble estimator."""
    base_tree = TemporalTree(
        **MODEL_CONFIG,
        verbose=VERBOSE,
        use_dynamic_anchors=USE_DYNAMIC_ANCHORS,
        anchor_offset_map=ANCHOR_OFFSET_MAP,
    )
    if not USE_ENSEMBLE:
        return base_tree

    return TemporalTreeEnsemble(
        estimator=base_tree,
        n_estimators=N_ESTIMATORS,
        base_split_sampling_seed=BASE_ENSEMBLE_SPLIT_SAMPLING_SEED,
        split_sampling_seeds=ENSEMBLE_SPLIT_SAMPLING_SEEDS,
    )


def train_model(train_data, y_train):
    model = build_model()
    model.fit(train_data, y_train)
    if VERBOSE >= 1:
        if y_train.nunique() < 2:
            print("[TRAIN] ROC-AUC: unavailable (training labels contain one class)")
        else:
            train_scores = model.predict(train_data)
            train_auc = roc_auc_score(y_train, train_scores)
            print(f"[TRAIN] ROC-AUC: {train_auc:.4f}")
    return model


def run_training_temporal_explainability(model, train_data, output_dir):
    """Run the configured version-one explainability stage on training data."""
    if not RUN_TEMPORAL_EXPLAINABILITY:
        return None
    if isinstance(model, TemporalTreeEnsemble):
        raise ValueError(
            "Temporal explainability version 1 supports single-tree models "
            "only. Disable RUN_TEMPORAL_EXPLAINABILITY when using an ensemble."
        )

    result = run_temporal_explainability(
        model=model,
        X=train_data,
        output_dir=output_dir,
        bin_edges_by_anchor=TEMPORAL_EXPLAINABILITY_BIN_EDGES,
    )
    print(
        "[EXPLAINABILITY] Saved training temporal paths and histogram\n"
        f"        Path table: {result['path_table_path']}\n"
        f"        Histogram: {result['histogram_path']}"
    )
    return result


def evaluate_model(model, test_data, y_test, output_dir=None, plot_tree=False):
    print('-------------evaluate_model() func-------------')
    predictions = model.predict(test_data)
    if len(predictions) != len(y_test):
        raise ValueError(
            "Test predictions and labels have different lengths."
        )

    # print('Predictions:', predictions)
    print('Threshold:', 0.5)
    y_predicted = (predictions >= 0.5).astype(int)
    # print('Predicted labels:', y_predicted)
    results = evaluate_TemporalTree(y_test, y_predicted, y_probabilities=predictions)
    if plot_tree and isinstance(model, TemporalTreeEnsemble):
        print(
            "[ENSEMBLE] Tree plotting is skipped because the current plot "
            "function supports one TemporalTree only. Individual trees remain "
            "available through model.estimators_."
        )
    elif plot_tree:
        save_path = None
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            save_path = output_dir / "tree.png"
        plot_tree_structure(model.tree, save_path=save_path)
    return results


def export_tree_structure(tree, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    nodes = []
    lines = []
    filename = "tree.json"
    
    def traverse(node, depth=0):
        node_id = len(nodes)
        if node.is_leaf():
            node_data = {
                "node_id": node_id,
                "type": "leaf",
                "depth": depth,
                "prediction": node.value,
                "num_patients": node.num_patients,
            }
            nodes.append(node_data)
            lines.append(
                f"Node {node_id} | Leaf | depth={depth} | "
                f"prediction={node.value} | patients={node.num_patients}"
            )
            return node_id

        node_data = {
            "node_id": node_id,
            "type": "split",
            "depth": depth,
            "feature": node.feature,
            "aggregation": node.aggregation,
            "anchor": node.time_window[0],
            "offset": node.time_window[1],
            "threshold": node.threshold,
            "num_patients": node.num_patients,
            "left_child": None,
            "right_child": None,
        }
        nodes.append(node_data)

        left_child = traverse(node.left, depth + 1) if node.left is not None else None
        right_child = traverse(node.right, depth + 1) if node.right is not None else None
        node_data["left_child"] = left_child
        node_data["right_child"] = right_child

        lines.append(
            f"Node {node_id} | Split | depth={depth} | "
            f"feature={node.feature} | aggregation={node.aggregation} | "
            f"anchor={node.time_window[0]} | offset={node.time_window[1]} | "
            f"threshold={node.threshold} | patients={node.num_patients} | "
            f"left_child={left_child} | right_child={right_child}"
        )
        return node_id

    traverse(tree)

    # text_path = output_dir / "temporal_tree_structure.txt"
    json_path = output_dir / filename

    # text_path.write_text("\n".join(lines) + "\n")
    json_path.write_text(json.dumps(nodes, indent=2, default=str))
    return json_path


def export_model_tree_structure(model, output_dir):
    """Export a single tree, or explicitly skip unsupported ensemble export."""
    if isinstance(model, TemporalTreeEnsemble):
        print(
            "[ENSEMBLE] Single-tree JSON export is skipped. Individual fitted "
            "trees are stored in the serialized ensemble's estimators_."
        )
        return None
    return export_tree_structure(model.tree, output_dir)


def save_model(model, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / "model.dill"

    with open(filename, "wb") as f:
        dill.dump(model, f)
    return filename


def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = dill.load(f)
    if isinstance(model, dict) and "model" in model:
        return model["model"]
    return model


def main():
    total_start_time = time.time()

    # # Existing CSV-based run remains unchanged.
    # legacy_data_path = Path('legacy/synthea_dataset_AtopicDermatitis_2.csv')
    # legacy_labels_path = Path('legacy/labels_AtopicDermatitis_synthea_2.csv')
    # output_dir = Path('outputs/models')

    # data, labels_df = load_data(legacy_data_path, legacy_labels_path)
    # train_data, test_data, y_train, y_test = split_data(data, labels_df)

    # start_time = time.time()
    # model = train_model(train_data, y_train)
    # end_time = time.time()

    # print(f"\nTraining time: {end_time - start_time:.2f} seconds")

    # results = evaluate_model(model, test_data, y_test)
    # print(results)

    # save_model(model, output_dir)
    # print(f"\nTraining time: {end_time - start_time:.2f} seconds")
    # print(f"Total runtime: {total_runtime_seconds / 60:.2f} minutes")


    # Additional synthetic-data run using the new PKL files.
    output_dir = make_run_dir(RUN_NAME)
    config_path = output_dir / "config.json"
    write_json(build_config(RUN_DESCRIPTION, DATA_PATH), config_path)
    #synthetic_moh_longitudinal_2000
    #synthetic_moh_labels_2000
    # synthetic_data_path = Path('synthetic_temporal_unit_test_5000.pkl') #
    # synthetic_labels_path = Path('synthetic_temporal_unit_test_labels_5000.pkl') #
    
    # synthetic_data_path = Path('synthetic_exp3_temporal_5000.pkl') #
    # synthetic_labels_path = Path('synthetic_exp3_temporal_labels_5000.pkl')
    
    # strong benchmark:
    synthetic_data_path = DATA_PATH
    synthetic_labels_path = LABELS_PATH
    
    synthetic_data, synthetic_labels_df = load_data(synthetic_data_path, synthetic_labels_path)
    (
        synthetic_train_data,
        synthetic_test_data,
        synthetic_y_train,
        synthetic_y_test
    ) = split_data(
        synthetic_data, synthetic_labels_df
    )

    synthetic_start_time = time.time()
    synthetic_model = train_model(synthetic_train_data, synthetic_y_train)
    synthetic_end_time = time.time()

    print(f"\nSynthetic training time: {synthetic_end_time - synthetic_start_time:.2f} seconds")

    run_training_temporal_explainability(
        synthetic_model,
        synthetic_train_data,
        output_dir,
    )

    synthetic_results = evaluate_model(
        synthetic_model,
        synthetic_test_data,
        synthetic_y_test,
        output_dir,
        PLOT_TREE
    )
    print("Synthetic results:")
    print(synthetic_results)

    tree_json_path = export_model_tree_structure(synthetic_model, output_dir)
    model_path = save_model(synthetic_model, output_dir)

    total_runtime_seconds = time.time() - total_start_time
    artifact_paths = {
        "model": model_path,
        "tree_json": tree_json_path,
        "tree_png": (
            output_dir / "tree.png"
            if PLOT_TREE and not isinstance(synthetic_model, TemporalTreeEnsemble)
            else None
        ),
        "config": config_path,
    }
    save_metrics(
        synthetic_results,
        total_runtime_seconds / 60,
        artifact_paths,
        output_dir,
    )
    print(f"\nTotal runtime: {total_runtime_seconds:.2f} seconds")
    print(f"Total runtime: {total_runtime_seconds / 60:.2f} minutes")


if __name__ == "__main__":
    print("Starting Temporal Tree training and evaluation...")
    main()
