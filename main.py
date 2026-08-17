from datetime import datetime
import json
from pathlib import Path
import time

import dill
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from settree.eval_utils import evaluate_TemporalTree, plot_tree_structure
from settree.temporal_tree import TemporalTree


TARGET_RECALL = 0.80
PLOT_TREE = False  # Set to True to plot the tree structure during evaluation


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

    #split train - validation - test datasets 

    # Stratified split to maintain label distribution
    train_val_patients, test_patients = train_test_split(
        labels_df["patient_id"],
        test_size=0.2,
        stratify=labels_df["label"],  # Ensures class balance
        random_state=42
    )
    labels_by_patient = labels_df.set_index("patient_id")
    train_val_labels = labels_by_patient.loc[train_val_patients, "label"]

    train_patients, val_patients = train_test_split(
        train_val_patients,
        test_size=0.25,
        stratify=train_val_labels,
        random_state=42
    )

    assert set(train_patients).isdisjoint(val_patients)
    assert set(train_patients).isdisjoint(test_patients)
    assert set(val_patients).isdisjoint(test_patients)

    # Create train, validation and test sets
    train_data = data[data["patient_id"].isin(train_patients)].copy()
    val_data = data[data["patient_id"].isin(val_patients)].copy()
    test_data = data[data["patient_id"].isin(test_patients)].copy()

    train_patient_order = train_data["patient_id"].drop_duplicates().to_numpy()
    val_patient_order = val_data["patient_id"].drop_duplicates().to_numpy()
    test_patient_order = test_data["patient_id"].drop_duplicates().to_numpy()

    for split_name, patient_order in {
        "train": train_patient_order,
        "validation": val_patient_order,
        "test": test_patient_order,
    }.items():
        missing_labels = set(patient_order) - set(labels_by_patient.index)
        if missing_labels:
            raise ValueError(
                f"Every {split_name} patient must have a label. "
                f"Missing labels for {len(missing_labels)} {split_name} patients."
            )

    y_train = labels_by_patient.loc[train_patient_order, "label"].reset_index(drop=True)
    y_val = labels_by_patient.loc[val_patient_order, "label"].reset_index(drop=True)
    y_test = labels_by_patient.loc[test_patient_order, "label"].reset_index(drop=True)

    n_train = len(train_data['patient_id'].unique())
    print(f'number of patients in the train: {n_train}')
    print(f'Precentage of label 1 in train: {y_train.sum()*100/n_train}')
    n_val = len(val_data['patient_id'].unique())
    print(f'number of patients in the validation: {n_val}')
    print(f'Precentage of label 1 in validation: {y_val.sum()*100/n_val}')
    n_test = len(test_data['patient_id'].unique())
    print(f'number of patients in the test: {n_test}')
    return train_data, val_data, test_data, y_train, y_val, y_test


def train_model(train_data, y_train):
    anchors_list = ['Birth Date Anchor']
    anchor_offset_list = [None,1, 2, 3, 4, 6, 9, 12, 12 * 5]#-12, -6, 12*5, 12*7] #months
    model = TemporalTree(max_depth=3,
                         min_samples_split=1,
                         min_samples_leaf = 7,
                         anchors_list=anchors_list,
                         anchor_offset_list=anchor_offset_list,
                         is_classification=False,
                         criterion='mse',
                        learn_missing_direction=False)
    model.fit(train_data, y_train)
    return model


def select_threshold_for_target_recall(y_val, val_scores, target_recall):
    print('\n-------------select_threshold_for_target_recall() func-------------')
    y_val = np.asarray(y_val)
    val_scores = np.asarray(val_scores)

    if not np.all(np.isfinite(val_scores)):
        raise ValueError(
            "Validation scores must be finite before threshold selection."
        )

    if len(val_scores) != len(y_val):
        raise ValueError(
            "Validation scores and labels have different lengths."
        )

    candidate_thresholds = np.unique(val_scores)

    threshold_results = []
    for threshold in candidate_thresholds:
        y_pred = (val_scores >= threshold).astype(int)
        true_positive = np.sum((y_pred == 1) & (y_val == 1))
        false_positive = np.sum((y_pred == 1) & (y_val == 0))
        false_negative = np.sum((y_pred == 0) & (y_val == 1))

        recall_denominator = true_positive + false_negative
        precision_denominator = true_positive + false_positive
        recall = true_positive / recall_denominator if recall_denominator else 0.0
        precision = true_positive / precision_denominator if precision_denominator else 0.0

        threshold_results.append((threshold, recall, precision))

    satisfying_thresholds = [
        result for result in threshold_results
        if result[1] >= target_recall
    ]

    if satisfying_thresholds:
        selected_threshold, validation_recall, validation_precision = max(
            satisfying_thresholds,
            key=lambda result: (result[2], result[1], result[0])
        )
        target_recall_achieved = True
    else:
        selected_threshold, validation_recall, validation_precision = max(
            threshold_results,
            key=lambda result: (result[1], result[2], result[0])
        )
        target_recall_achieved = False
        print(
            "WARNING: Target recall of "
            f"{target_recall:.2f} was not achievable on the validation set. "
            f"Best achieved validation recall was {validation_recall:.2f}. "
            f"Using fallback threshold {selected_threshold}."
        )

    return (
        selected_threshold,
        target_recall_achieved,
        validation_recall,
        validation_precision
    )


def evaluate_model(model, test_data, y_test, threshold, output_dir=None, plot_tree=False):
    print('-------------evaluate_model() func-------------')
    predictions = model.predict(test_data)
    if len(predictions) != len(y_test):
        raise ValueError(
            "Test predictions and labels have different lengths."
        )

    # print('Predictions:', predictions)
    print('Threshold:', threshold)
    y_predicted = (predictions >= threshold).astype(int)
    # print('Predicted labels:', y_predicted)
    results = evaluate_TemporalTree(y_test, y_predicted, y_probabilities=predictions)
    if plot_tree:
        save_path = None
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            save_path = output_dir / "temporal_tree_structure.png"
        plot_tree_structure(model.tree, save_path=save_path)
    return results


def export_tree_structure(tree, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    nodes = []
    lines = []
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f'temporal_tree_structure_{timestamp}.json'
    
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


def save_model(model, selected_threshold, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = output_dir / f'temporal_tree_{timestamp}.dill'
    model_package = {
        "model": model,
        "threshold": selected_threshold
    }

    with open(filename, "wb") as f:
        dill.dump(model_package, f)
    return filename


def load_model(model_path):
    with open(model_path, 'rb') as f:
        model_package = dill.load(f)
    model = model_package["model"]
    threshold = model_package["threshold"]
    return model, threshold


def main():
    total_start_time = time.time()

    # # Existing CSV-based run remains unchanged.
    # legacy_data_path = Path('legacy/synthea_dataset_AtopicDermatitis_2.csv')
    # legacy_labels_path = Path('legacy/labels_AtopicDermatitis_synthea_2.csv')
    # output_dir = Path('outputs/models')

    # data, labels_df = load_data(legacy_data_path, legacy_labels_path)
    # train_data, val_data, test_data, y_train, y_val, y_test = split_data(data, labels_df)

    # start_time = time.time()
    # model = train_model(train_data, y_train)
    # end_time = time.time()

    # print(f"\nTraining time: {end_time - start_time:.2f} seconds")

    # val_scores = model.predict(val_data)
    # selected_threshold, target_recall_achieved, validation_recall, validation_precision = (
    #     select_threshold_for_target_recall(y_val, val_scores, TARGET_RECALL)
    # )
    # print(f"Selected threshold: {selected_threshold}")
    # print(f"Target recall achieved: {target_recall_achieved}")
    # print(f"Validation recall: {validation_recall}")
    # print(f"Validation precision: {validation_precision}")
    # results = evaluate_model(model, test_data, y_test, selected_threshold)
    # print(results)

    # save_model(model, selected_threshold, output_dir)
    # print(f"\nTraining time: {end_time - start_time:.2f} seconds")
    # print(f"Total runtime: {total_runtime_seconds / 60:.2f} minutes")


    # Additional synthetic-data run using the new PKL files.
    output_dir = Path('outputs/models')
    #synthetic_moh_longitudinal_2000
    #synthetic_moh_labels_2000
    # synthetic_data_path = Path('synthetic_temporal_unit_test_5000.pkl') #
    # synthetic_labels_path = Path('synthetic_temporal_unit_test_labels_5000.pkl') #
    
    # synthetic_data_path = Path('synthetic_exp3_temporal_5000.pkl') #
    # synthetic_labels_path = Path('synthetic_exp3_temporal_labels_5000.pkl')
    
    # strong benchmark:
    synthetic_data_path = Path('temporal_fa_benchmark_5000.pkl') #
    synthetic_labels_path = Path('temporal_fa_benchmark_labels_5000.pkl')
    
    synthetic_data, synthetic_labels_df = load_data(synthetic_data_path, synthetic_labels_path)
    (
        synthetic_train_data,
        synthetic_val_data,
        synthetic_test_data,
        synthetic_y_train,
        synthetic_y_val,
        synthetic_y_test
    ) = split_data(
        synthetic_data, synthetic_labels_df
    )

    synthetic_start_time = time.time()
    synthetic_model = train_model(synthetic_train_data, synthetic_y_train)
    synthetic_end_time = time.time()

    print(f"\nSynthetic training time: {synthetic_end_time - synthetic_start_time:.2f} seconds")

    synthetic_val_scores = synthetic_model.predict(synthetic_val_data)
    (
        selected_threshold,
        target_recall_achieved,
        validation_recall,
        validation_precision
    ) = select_threshold_for_target_recall(
        synthetic_y_val,
        synthetic_val_scores,
        TARGET_RECALL
    )
    print(f"Selected threshold: {selected_threshold}")
    print(f"Target recall achieved: {target_recall_achieved}")
    print(f"Validation recall: {validation_recall}")
    print(f"Validation precision: {validation_precision}")

    synthetic_results = evaluate_model(
        synthetic_model,
        synthetic_test_data,
        synthetic_y_test,
        selected_threshold,
        output_dir,
        PLOT_TREE
    )
    print("Synthetic results:")
    print(synthetic_results)

    export_tree_structure(synthetic_model.tree, output_dir)
    save_model(synthetic_model,selected_threshold, output_dir)

    total_runtime_seconds = time.time() - total_start_time
    print(f"\nTotal runtime: {total_runtime_seconds:.2f} seconds")
    print(f"Total runtime: {total_runtime_seconds / 60:.2f} minutes")


if __name__ == "__main__":
    print("Starting Temporal Tree training and evaluation...")
    main()
