from datetime import datetime
from pathlib import Path
import time

import dill
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from settree.eval_utils import evaluate_TemporalTree, plot_tree_structure
from settree.temporal_tree import TemporalTree


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
    data.loc[~anchor_mask, 'value'] = data.loc[~anchor_mask, 'value'].fillna(0)

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

    #split train - test datasets 

    # Stratified split to maintain label distribution
    train_patients, test_patients = train_test_split(
        labels_df["patient_id"],
        test_size=0.3,
        stratify=labels_df["label"],  # Ensures class balance
        random_state=42
    )

    # Create train and test sets
    train_data = data[data["patient_id"].isin(train_patients)].copy()
    test_data = data[data["patient_id"].isin(test_patients)].copy()

    train_patient_order = train_data["patient_id"].drop_duplicates().to_numpy()
    test_patient_order = test_data["patient_id"].drop_duplicates().to_numpy()
    labels_by_patient = labels_df.set_index("patient_id")

    missing_train_labels = set(train_patient_order) - set(labels_by_patient.index)
    if missing_train_labels:
        raise ValueError(
            "Every train patient must have a label. "
            f"Missing labels for {len(missing_train_labels)} train patients."
        )

    missing_test_labels = set(test_patient_order) - set(labels_by_patient.index)
    if missing_test_labels:
        raise ValueError(
            "Every test patient must have a label. "
            f"Missing labels for {len(missing_test_labels)} test patients."
        )

    y_train = labels_by_patient.loc[train_patient_order, "label"].reset_index(drop=True)
    y_test = labels_by_patient.loc[test_patient_order, "label"].reset_index(drop=True)

    n_train = len(train_data['patient_id'].unique())
    print(f'number of patients in the train: {n_train}')
    print(f'Precentage of label 1 in train: {y_train.sum()*100/n_train}')
    n_test = len(test_data['patient_id'].unique())
    print(f'number of patients in the test: {n_test}')
    return train_data, test_data, y_train, y_test


def train_model(train_data, y_train):
    anchors_list = ['Birth Date Anchor']
    anchor_offset_list = [None, 6, 12, 12 * 5]#-12, -6, 12*5, 12*7] #months
    model = TemporalTree(max_depth=3,
                         min_samples_split=1,
                         min_samples_leaf = 1,
                         anchors_list=anchors_list,
                         anchor_offset_list=anchor_offset_list,
                         is_classification=False,
                         criterion='mse',
                        learn_missing_direction=False)
    model.fit(train_data, y_train)
    return model


def evaluate_model(model, test_data, y_test):
    print('-------------evaluate_model() func-------------')
    predictions = model.predict(test_data)
    print('Predictions:', predictions)
    y_predicted = (predictions >= 0.5).astype(int)
    print('Predicted labels:', y_predicted)
    results = evaluate_TemporalTree(y_test, y_predicted, y_probabilities=predictions)
    plot_tree_structure(model.tree)
    return results


def save_model(model, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = output_dir / f'temporal_tree_{timestamp}.dill'
    with open(filename, 'wb') as f:
        dill.dump(model, f)
    return filename


def load_model(model_path):
    with open(model_path, 'rb') as f:
        return dill.load(f)


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

    # Additional synthetic-data run using the new PKL files.
    output_dir = Path('outputs/models')
    #synthetic_moh_longitudinal_2000
    #synthetic_moh_labels_2000
    synthetic_data_path = Path('synthetic_temporal_unit_test_5000.pkl') #
    synthetic_labels_path = Path('synthetic_temporal_unit_test_labels_5000.pkl') #

    synthetic_data, synthetic_labels_df = load_data(synthetic_data_path, synthetic_labels_path)
    synthetic_train_data, synthetic_test_data, synthetic_y_train, synthetic_y_test = split_data(
        synthetic_data, synthetic_labels_df
    )

    synthetic_start_time = time.time()
    synthetic_model = train_model(synthetic_train_data, synthetic_y_train)
    synthetic_end_time = time.time()

    print(f"\nSynthetic training time: {synthetic_end_time - synthetic_start_time:.2f} seconds")

    synthetic_results = evaluate_model(synthetic_model, synthetic_test_data, synthetic_y_test)
    print("Synthetic results:")
    print(synthetic_results)

    save_model(synthetic_model, output_dir)

    total_runtime_seconds = time.time() - total_start_time
    print(f"\nTotal runtime: {total_runtime_seconds:.2f} seconds")
    print(f"Total runtime: {total_runtime_seconds / 60:.2f} minutes")


if __name__ == "__main__":
    print("Starting Temporal Tree training and evaluation...")
    main()
