#%%
# improved:
import numpy as np
import pandas as pd
import argparse
from typing import List, Tuple, Dict, Callable, Any
from itertools import product
from datetime import datetime, date, time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, 
    roc_auc_score, roc_curve, matthews_corrcoef
)
from pandas import Timestamp

class TemporalDecisionTreeRegressor:
    def __init__(self, max_depth: int = 3, min_samples_split: int = 1, anchors_list=['Birth Date Anchor'], anchor_offset_list = [None], time_windows: List[Tuple] = [(0,3),(-6,-3),(None,None)] , aggregation_funcs: Dict[str, Callable] = None, is_classification = True , criterion = "entropy"):
        self.max_depth = max_depth
        self.is_classification = is_classification
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.time_windows = time_windows # (None, None) means look at all data, no specified window time
        self.anchors_list = anchors_list
        self.anchor_offset_list = anchor_offset_list
        self.aggregation_funcs = aggregation_funcs if aggregation_funcs else {
            'mean': np.mean,
            'sum': np.sum,
            'max': np.max,
            'count': lambda x: x.size # count the number of events in the period
        }
        self.tree = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fits the regression tree to the data."""
        print("Fitting the model...")
        self._validate_input(X, y)
        self.tree = self._build_tree(X, y, depth=0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts the target values for the input data."""
        print('\n==========================================================================================')
        print("Predicting...")
        X = X.copy()
        X["time"] = pd.to_datetime(X["time"])

        # self._validate_input(X, pd.Series([0]*len(X)))  #Validate input for prediction, dummy y
        unique_patient_ids = X['patient_id'].unique()
        
        print(f'Predicting on {len(unique_patient_ids)} patients.')

        predictions = np.zeros(len(unique_patient_ids))
        for i, patient_id in enumerate(unique_patient_ids):
            # print(patient_id)
            patient_data = X[X['patient_id'] == patient_id]
            # print(patient_data)
            
            predictions[i] = self._predict_row(patient_data, self.tree)
        return predictions
    def evaluate(self,y_real, y_predicted, y_probabilities=None):
        """
        Evaluate classification performance with metrics suitable for imbalanced data.
        
        Parameters:
        - y_real: Array-like, true labels (ground truth).
        - y_predicted: Array-like, predicted labels.
        - y_probabilities: Array-like, predicted probabilities for positive class (needed for AUC-ROC).

        Returns:
        - A dictionary with accuracy, precision, recall, F1 score, MCC, AUC-ROC, and confusion matrix.
        - Displays an AUC-ROC curve plot.
        """
        # Compute traditional metrics
        
        accuracy = accuracy_score(y_real, y_predicted)
        precision = precision_score(y_real, y_predicted, average='binary', zero_division=0)
        recall = recall_score(y_real, y_predicted, average='binary', zero_division=0)
        f1 = f1_score(y_real, y_predicted, average='binary', zero_division=0)
        mcc = matthews_corrcoef(y_real, y_predicted)  # More balanced metric for imbalanced datasets
        conf_matrix = confusion_matrix(y_real, y_predicted)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'mcc': mcc,
            'confusion_matrix': conf_matrix
        }
        
        # AUC-ROC Calculation & Plot
        if y_probabilities is not None:
            auc_roc = roc_auc_score(y_real, y_probabilities)
            fpr, tpr, _ = roc_curve(y_real, y_probabilities)
            print(len(fpr))
            # Store AUC in results
            results['auc_roc'] = auc_roc

            # Plot ROC Curve
            plt.figure(figsize=(6, 6))
            plt.plot(fpr, tpr, color='blue', label=f'AUC-ROC = {auc_roc:.3f}')
            plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.grid()
            plt.show()
        return results

    def _validate_input(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Validates the input data."""
        print("Validating input...")
        if 'time' not in X.columns or 'patient_id' not in X.columns or 'feature_name' not in X.columns or 'value' not in X.columns :
            raise ValueError("Input data must contain 'time', 'patient_id', 'feature_name', 'value'  columns.")
        if len(X['patient_id'].unique()) != len(y):
            raise ValueError("The length of y must be the same as the number of patients.")
        if X.isnull().values.any() or y.isnull().values.any():
            raise ValueError("Input data contains missing values. Please handle them before fitting or predicting.")
        
        # Check if self.anchors_list is a list
        if not isinstance(self.anchors_list, list):
                raise ValueError("self.anchors_list must be a list.")

        # Check that all elements are either strings or pandas Timestamps
        if not all(isinstance(item, (str, Timestamp)) for item in self.anchors_list):
            raise ValueError("self.anchors_list must contain only strings or pandas Timestamps.")

        # If the element is a string, ensure it ends with 'Anchor'
        if not all(isinstance(item, Timestamp) or (isinstance(item, str) and item.endswith("Anchor")) for item in self.anchors_list):
            raise ValueError("All string elements in self.anchors_list must end with 'Anchor'.")

        # Check if self.anchor_offset_list is a list of integers
        if not isinstance(self.anchor_offset_list, list) or not all(isinstance(item, (int, type(None))) for item in self.anchor_offset_list):
            raise ValueError("self.anchor_offset_list must be a list containing only integers or None.")
    
    
    def _build_tree(self, X: pd.DataFrame, y: pd.Series, depth: int, anchors_list=None) -> Dict:
        print('-' * 40)
        print(f"Building tree at depth {depth}...")

        if anchors_list is None: # the root node will only choose from the given anchors
            anchors_list = self.anchors_list.copy()
        
        print(f'Anchors list: {anchors_list}')
        unique_patient_ids = X['patient_id'].unique()
        num_patients = len(unique_patient_ids)

        # Stopping conditions
        if num_patients <= self.min_samples_split or depth >= self.max_depth:
            print(f"Stop: (num_patients <= {self.min_samples_split}) or (depth >= {self.max_depth}) -> leaf node")
            if self.is_classification:
                most_common_label = np.bincount(y).argmax()
                return {'value': most_common_label, 'num_patients': num_patients}
            else:
                return {'value': y.mean(), 'num_patients': num_patients}

        # Check homogeneity
        if self.is_classification:
            if len(np.unique(y)) == 1:
                print(f"Homogeneous classification node ==> return: {y.iloc[0]}")
                return {'value': y.iloc[0]}
        else:
            if np.all(y == y.iloc[0]):
                print(f"Homogeneous regression node ==> return: {y.iloc[0]}")
                return {'value': y.iloc[0]}

        best_split = self._find_best_split(X, y, anchors_list=anchors_list)
        if not best_split:
            print('Stop, no valid split is found -> the node is a leaf node.')
            if self.is_classification:
                most_common_label = np.bincount(y).argmax()
                return {'value': most_common_label, 'num_patients': num_patients}
            else:
                return {'value': y.mean(), 'num_patients': num_patients}

        print(f'==> Best split is for: {best_split}')

        # Aggregate feature values
        aggregated_values, anchor = self._aggregate_feature_over_time(
            X, best_split['feature'], best_split['time_window'], best_split['aggregation']
        )
        # print(f'aggregated_values:\n{aggregated_values}')
        left_mask = aggregated_values <= best_split['threshold']
        right_mask = ~left_mask

        left_patient_ids = unique_patient_ids[left_mask]
        right_patient_ids = unique_patient_ids[right_mask]

        print(f"Splitting: {num_patients} patients -> Left: {len(left_patient_ids)}, Right: {len(right_patient_ids)}")

        # -------------- updated the propagation ----------------

        # Generate new anchors names based on the split direction
        left_anchor = f"{best_split['feature']} <= {np.round(best_split['threshold'])} Anchor"
        right_anchor = f"{best_split['feature']} > {np.round(best_split['threshold'])} Anchor"

        # Function to extract valid dates based on a mask
        def create_anchor_dates(patient_data, side_mask):
            return patient_data.loc[side_mask, 'time'].tolist()

        # # Filter data for the best split feature
        filtered_X = X[X['feature_name'] == best_split['feature']]
        left_anchor_data = []
        right_anchor_data = []
        # --------------------------------------------------------------------------
        # Create separate X for left and right subtrees with unique anchors
        X_left = X[X['patient_id'].isin(left_patient_ids)]
        X_right = X[X['patient_id'].isin(right_patient_ids)]
        
        filtered_X_left = X_left[X_left['feature_name'] == best_split['feature']]
        filtered_X_right = X_right[X_right['feature_name'] == best_split['feature']]
        
        # ----------------- assigning anchors based on the chosen feature in the best split --------------------------------------
        # add anchors for left patients
        for patient_id in left_patient_ids:
            patient_data = filtered_X_left[filtered_X_left['patient_id'] == patient_id].copy() # the patient data for the specific chosen feature
            # patient_data = X_left[X_left['patient_id'] == patient_id].copy() # the patient data for all features
            # feature_patient_data = patient_data[patient_data['feature_name'] == best_split['feature']]

            if patient_data.empty:
                # No data for this feature, assign empty anchor for this patients
                left_anchor_data.append({'patient_id': patient_id, 'time': np.nan, 'feature_name': left_anchor, 'value': []})
                continue
            
            # Step 1: Apply value threshold
            value_mask_l = patient_data['value'].astype(float) <= float(best_split['threshold']) # first mask is where the value of the feature exceeds a threshold

            # Step 2: Apply time window mask based on best_split time_window  
            # second mask is the time points on the first mask to fall within the time_window defined by the best split 
            prev_anchor_name = best_split['time_window'][0] # previous anchor name (that defines the previous time window)
            offset = best_split['time_window'][1] 
            # the time window we are searching is [prev_anchor_name +- offset]
            patient_data['time'] = pd.to_datetime(patient_data['time'])
            if isinstance(prev_anchor_name, str):
                # Get the anchor value for this patient (single row)
                full_patient_data = X_left[X_left['patient_id'] == patient_id]
                anchor_value = full_patient_data[full_patient_data['feature_name'] == prev_anchor_name]['value']

                if anchor_value.empty:
                    time_mask = pd.Series([False] * len(patient_data), index=patient_data.index)

                else:
                    anchor_raw = anchor_value.values[0]
                    if offset is None:
                        time_mask = pd.Series([False] * len(patient_data), index=patient_data.index)
                        prev_anchor_name = None  # Treat as no valid anchor

                    elif isinstance(anchor_raw, pd.Timestamp):
                        anchor_time = pd.to_datetime(anchor_raw)
                        offset_time = anchor_time + pd.DateOffset(months=offset)

                        if offset >= 0:
                            time_mask = (patient_data['time'] >= anchor_time) & (patient_data['time'] <= offset_time)
                        else:
                            time_mask = (patient_data['time'] <= anchor_time) & (patient_data['time'] >= offset_time)

                    elif isinstance(anchor_raw, list):
                        # Convert to datetime if not already
                        anchor_times = pd.to_datetime(anchor_raw)
                        offset_times = [dt + pd.DateOffset(months=offset) for dt in anchor_times]

                        def time_filter(time_val, ref_list, offset_list):
                            if offset >= 0:
                                return any((time_val <= o) and (r <= time_val) for r, o in zip(ref_list, offset_list))
                            else:
                                return any((time_val >= o) and (r >= time_val) for r, o in zip(ref_list, offset_list))

                        # Apply time mask using the list logic
                        time_mask = patient_data.apply(
                            lambda row: time_filter(row['time'], anchor_times, offset_times), axis=1
                        )

                    else:
                        raise ValueError(f"The value of {prev_anchor_name} Anchor must be a Timestamp or List of Timestamps.")

            elif isinstance(prev_anchor_name, pd.Timestamp):
                anchor_time = prev_anchor_name
                offset_time = anchor_time + pd.DateOffset(months=offset)

                if offset >= 0:
                    time_mask = (patient_data['time'] >= anchor_time) & (patient_data['time'] <= offset_time)
                else:
                    time_mask = (patient_data['time'] <= anchor_time) & (patient_data['time'] >= offset_time)

            elif prev_anchor_name is None:
                # No restriction on time
                time_mask = pd.Series([True] * len(patient_data), index=patient_data.index)

            else:
                raise ValueError(f"Unsupported anchor type: {type(prev_anchor_name)}")
            # Step 3: Combine masks (that later on will define the data points to join the multiple-anchor)
            mask_patient_l = value_mask_l & time_mask
            
            # print('------new mask-----------------')
            # Step 4: 
            # Create anchors based on the mask
            left_anchor_dates = create_anchor_dates(patient_data, mask_patient_l)
            left_anchor_data.append({'patient_id': patient_id, 'time': np.nan, 'feature_name': left_anchor, 'value': left_anchor_dates})
        
        # add anchors for right patients
        for patient_id in right_patient_ids:
            patient_data = filtered_X_right[filtered_X_right['patient_id'] == patient_id].copy() # the patient data for the specific chosen feature
            # patient_data = X_left[X_left['patient_id'] == patient_id].copy() # the patient data for all features
            # feature_patient_data = patient_data[patient_data['feature_name'] == best_split['feature']]

            if patient_data.empty:
                # No data for this feature, assign empty anchor for this patients
                right_anchor_data.append({'patient_id': patient_id, 'time': np.nan, 'feature_name': right_anchor, 'value': []})
                print('No data for this feature, assign empty anchor for this patients')
                continue
            
            # Step 1: Apply value threshold
            value_mask_r = patient_data['value'].astype(float) <= float(best_split['threshold']) # first mask is where the value of the feature exceeds a threshold

            # Step 2: Apply time window mask based on best_split time_window  
            # second mask is the time points on the first mask to fall within the time_window defined by the best split 
            prev_anchor_name = best_split['time_window'][0] # previous anchor name (that defines the previous time window)
            offset = best_split['time_window'][1] 
            # the time window we are searching is [prev_anchor_name +- offset]
            patient_data['time'] = pd.to_datetime(patient_data['time'])
            if isinstance(prev_anchor_name, str):
                # Get the anchor value for this patient (single row)
                full_patient_data = X_right[X_right['patient_id'] == patient_id]
                anchor_value = full_patient_data[full_patient_data['feature_name'] == prev_anchor_name]['value']

                if anchor_value.empty:
                    time_mask = pd.Series([False] * len(patient_data), index=patient_data.index)

                else:
                    anchor_raw = anchor_value.values[0]
                    if offset is None:
                        time_mask = pd.Series([False] * len(patient_data), index=patient_data.index)
                        prev_anchor_name = None  # Treat as no valid anchor

                    elif isinstance(anchor_raw, pd.Timestamp):
                        anchor_time = pd.to_datetime(anchor_raw)
                        offset_time = anchor_time + pd.DateOffset(months=offset)

                        if offset >= 0:
                            time_mask = (patient_data['time'] >= anchor_time) & (patient_data['time'] <= offset_time)
                        else:
                            time_mask = (patient_data['time'] <= anchor_time) & (patient_data['time'] >= offset_time)

                    elif isinstance(anchor_raw, list):
                        # Convert to datetime if not already
                        anchor_times = pd.to_datetime(anchor_raw)
                        offset_times = [dt + pd.DateOffset(months=offset) for dt in anchor_times]

                        def time_filter(time_val, ref_list, offset_list):
                            if offset >= 0:
                                return any((time_val <= o) and (r <= time_val) for r, o in zip(ref_list, offset_list))
                            else:
                                return any((time_val >= o) and (r >= time_val) for r, o in zip(ref_list, offset_list))

                        # Apply time mask using the list logic
                        time_mask = patient_data.apply(
                            lambda row: time_filter(row['time'], anchor_times, offset_times), axis=1
                        )

                    else:
                        raise ValueError(f"The value of {prev_anchor_name} Anchor must be a Timestamp or List of Timestamps.")

            elif isinstance(prev_anchor_name, pd.Timestamp):
                anchor_time = prev_anchor_name
                offset_time = anchor_time + pd.DateOffset(months=offset)

                if offset >= 0:
                    time_mask = (patient_data['time'] >= anchor_time) & (patient_data['time'] <= offset_time)
                else:
                    time_mask = (patient_data['time'] <= anchor_time) & (patient_data['time'] >= offset_time)

            elif prev_anchor_name is None:
                # No restriction on time
                time_mask = pd.Series([True] * len(patient_data), index=patient_data.index)

            else:
                raise ValueError(f"Unsupported anchor type: {type(prev_anchor_name)}")
            # Step 3: Combine masks
            mask_patient_r = value_mask_r & time_mask
            # Step 4: 
            # print('------new mask-----------------')
            # Create anchors based on the mask
            right_anchor_dates = create_anchor_dates(patient_data, mask_patient_r)
            right_anchor_data.append({'patient_id': patient_id, 'time': np.nan, 'feature_name': right_anchor, 'value': right_anchor_dates})

        # Create new DataFrames for left and right anchors
        left_anchors_df = pd.DataFrame(left_anchor_data)
        right_anchors_df = pd.DataFrame(right_anchor_data)           
        # from IPython.display import display
        # print(f'left_anchor_data:{len(left_anchors_df)}')
        # # # display(left_anchors_df)
        # print(f'right_anchor_data:{len(right_anchors_df)}')
        # # # display(right_anchors_df)
        # print(f'left_patient_ids:{len(left_patient_ids)}')
        # # display(left_patient_ids)
        # print(f'right_patient_ids: {len(right_patient_ids)}')
        # display(right_patient_ids)
 
        X_next_left = pd.concat([X_left, left_anchors_df], ignore_index=True)
        X_next_right = pd.concat([X_right, right_anchors_df], ignore_index=True)

        # print(f'\nX_next_left:')
        # display(X_next_left)
        # print(f'X_next_right:')
        # display(X_next_right)
        # print('\nCheck if all the patients in left has the new anchor:')
        # ids__ = X_next_left.loc[X_next_left['feature_name']==left_anchor,'patient_id'].unique()
        # print(f'len(ids): {len(ids__)}')
        # print(f'len left patients: {len(left_patient_ids)}')
        
        # Create separate anchor lists for left and right
        left_anchors_list = anchors_list + [left_anchor]
        right_anchors_list = anchors_list + [right_anchor]
        
        print('Entering recursive call')
        # Recursive calls with independent anchor lists
        left_tree = self._build_tree(X_next_left, y[left_mask], depth + 1, left_anchors_list)
        right_tree = self._build_tree(X_next_right, y[right_mask], depth + 1, right_anchors_list)

        # Return the split node
        return {
            'feature': best_split['feature'],
            'threshold': best_split['threshold'],
            'aggregation': best_split['aggregation'],
            'time_window': best_split['time_window'],
            'num_patients': num_patients,
            'anchors_list': anchors_list,  # Keep the current node's anchor list
            'left': left_tree,
            'right': right_tree
        }

    def _find_best_split(self, X: pd.DataFrame, y: pd.Series, anchors_list = None) -> Dict:
        """Finds the best split (feature, threshold, time and aggregation function) using temporal aggregations efficiently.
        X- raw EHR data
        y- target nd array
        """
        print("Finding best split...")

        best_split = None
        best_score= float('inf')
        num_samples = len(y)
        feature_names = X['feature_name'].unique()
        # print(f'Features names: {feature_names}')
        
        if anchors_list is None: # the root node will only choose from the given anchors
            anchors_list = self.anchors_list.copy()
        
        for feature, (anchor, offset), agg_name in product(feature_names, product(anchors_list, self.anchor_offset_list), self.aggregation_funcs.keys()):
            if (feature == 'Birth Date') or (feature.endswith('Anchor')):  # feature in self.anchors_list
                continue  # Skip anchor features and 'Birth Date'
            
            tau = (anchor, offset)  # Create tau tuple
            print(f'\ntesting {(feature, tau, agg_name)}')
            aggregated_values, anchor = self._aggregate_feature_over_time(X, feature, tau , agg_name)
            # print(f'aggregated_values:{aggregated_values}')

            thresh_values = np.unique(pd.Series(aggregated_values).dropna())
            for threshold in thresh_values:
                # print(f'  threshold:{threshold}')
                with np.errstate(invalid='ignore'):  # Suppress invalid comparison warnings
                    left_mask = aggregated_values <= threshold

                right_mask = ~left_mask
                # print(f'right_mask:\n{right_mask}')
                # print(f'left_mask ( <= ):\n{left_mask}')
                if np.sum(left_mask) < self.min_samples_split or np.sum(right_mask) < self.min_samples_split:
                    print(f'  This split has {np.sum(left_mask)} samples on the left, and {np.sum(right_mask)} on the right.')
                    print(f'  The split is skipped, it does not meet the minimum number of samples required for valid child nodes ({self.min_samples_split}).')
                    continue

                # score = self._calculate_mse(y[left_mask], y[right_mask])
                score = self._calculate_split_score(y[left_mask], y[right_mask], self.criterion, self.is_classification)
                # print(f'  Calculating {self.criterion} score for this combination with threshold {threshold}.')
                # print(f'{self.criterion} score:{score}. best score:{score}')
                
                if score < best_score:
                    best_score = score
                    best_split = {
                        'feature': feature,
                        'threshold': threshold,
                        'aggregation': agg_name,
                        'time_window': (anchor, offset) # (chosen anchor, chosen offset)
                    }
        return best_split


    def _aggregate_feature_over_time(self, X_original: pd.DataFrame, feature: str, tau: Tuple, aggregation: str) -> np.ndarray:
        """
        Aggregates feature values within a specified time window using a given aggregation function.
        """
        Anchor_feature_name, offset_from_anchor = tau
        X = X_original.copy()
        # Ensure "time" is datetime
        X["time"] = pd.to_datetime(X["time"])

        # Pre-filter the relevant feature
        X_feature = X[X["feature_name"] == feature].copy()
        # If X_feature is empty, return NaN for the patient
        if X_feature.empty:
            # Ensure all patient IDs are present
            # print('X_feature is empty, return Nan')
            final_df = pd.DataFrame({"patient_id": X["patient_id"].unique(),"value":np.array([np.nan] * len(X["patient_id"].unique()))})
            return final_df["value"].values
        if isinstance(Anchor_feature_name, str):
            # Check if Anchor_feature_name appears exactly once per patient
            reference_feature = X[X["feature_name"] == Anchor_feature_name][["patient_id", "value"]].copy()

            if len(reference_feature) != len(X["patient_id"].unique()):
                print(f'len of data with this anchor {len(reference_feature["patient_id"].unique())}')
                print(f'unique patients: {len(X["patient_id"].unique())}')
                raise ValueError(f"Each patient must have one {Anchor_feature_name} entry in 'feature_name'.")

            reference_feature.rename(columns={"value": "reference_dates_list"}, inplace=True)
            X_feature = X_feature.merge(reference_feature, on="patient_id", how="left")  
            if offset_from_anchor is None:
                   mask = np.ones(len(X_feature), dtype=bool)  # Select all rows
                   Anchor_feature_name = None # in the case where we are looking at the data of all times, the anchor name makes no sense.
            elif X_feature["reference_dates_list"].map(lambda x: isinstance(x, pd.Timestamp)).all():
                # Single timestamp case             
                X_feature["date_relative_offset"] = X_feature["reference_dates_list"] + pd.DateOffset(months=offset_from_anchor) 
                if offset_from_anchor >= 0:
                    # print(X_feature["time"].dtype)  
                    # print(X_feature["date_relative_offset"].dtype)  
                    # print(X_feature["reference_dates_list"].dtype)  
                    mask = (X_feature["time"] <= X_feature["date_relative_offset"]) & (X_feature["reference_dates_list"] <= X_feature["time"])
                else:
                    mask = (X_feature["time"] >= X_feature["date_relative_offset"]) & (X_feature["reference_dates_list"] >= X_feature["time"])

            elif X_feature["reference_dates_list"].map(lambda x: isinstance(x, list)).all():
                # Convert lists of timestamps to datetime
                X_feature["reference_dates_list"] = X_feature["reference_dates_list"].map(lambda lst: pd.to_datetime(lst))
                X_feature["date_relative_offset"] = X_feature["reference_dates_list"].map(lambda lst: [dt + pd.DateOffset(months=offset_from_anchor) for dt in lst])

                # Efficient vectorized filtering
                def time_filter(times, offsets, time_val):
                    if offset_from_anchor >= 0:
                        return any((time_val <= off) and (ref <= time_val) for ref, off in zip(times, offsets))
                    else:
                        return any((time_val >= off) and (ref >= time_val) for ref, off in zip(times, offsets))

                mask = X_feature.apply(lambda row: time_filter(row["reference_dates_list"], row["date_relative_offset"], row["time"]), axis=1)

            else:
                raise ValueError(f"The value of {Anchor_feature_name} Anchor should be pd.Timestamps or a List of pd.Timestamps or None")
        
        
        elif isinstance(Anchor_feature_name, pd.Timestamp):
            anchor_date = Anchor_feature_name
            X_feature["date_relative_offset"] = anchor_date + pd.DateOffset(months=offset_from_anchor)

            if offset_from_anchor >= 0:
                mask = (X_feature["time"] <= X_feature["date_relative_offset"]) & (anchor_date <= X_feature["time"])
            else:
                mask = (X_feature["time"] >= X_feature["date_relative_offset"]) & (anchor_date >= X_feature["time"])

        elif Anchor_feature_name is None: # if i want to look on all data, i give None in the list of anchors_list
                   mask = np.ones(len(X_feature), dtype=bool)
                   
        else:
            raise TypeError(f"Unsupported anchor or offset type: {type(tau[0]), type(tau[1])}")

        # Apply mask
        filtered_X = X_feature[mask].copy()
        
        # Ensure 'value' is numeric
        filtered_X["value"] = pd.to_numeric(filtered_X["value"], errors="coerce")

        # Aggregate values
        agg_result = filtered_X.groupby("patient_id")["value"].agg(aggregation).reset_index() if aggregation != "count" \
            else filtered_X.groupby("patient_id")["value"].count().reset_index()

        # Ensure all patient IDs are present
        final_df = pd.DataFrame({"patient_id": X["patient_id"].unique()}).merge(agg_result, on="patient_id", how="left")

        return final_df["value"].values, Anchor_feature_name

    # staticmethod
    def add_age_at_event(self,X):
        """
        Handles both cases where the birth date in found as a feature or the birth date fount as a seperate column.
        """
        if 'Birth Date' not in X['feature_name'].unique() and 'Birth Date' not in X.columns():
            raise ValueError("Input data must contain 'Birth Date' as a column or as a feature in 'feature_name' columns.")

        if 'Birth Date' in X['feature_name'].unique():
            birth_dates = X[X["feature_name"] == "Birth Date"].set_index("patient_id")["time"]
            # Merge birth dates into the dataframe based on patient_id
            X = X.merge(birth_dates, on="patient_id", how="left", suffixes=("", "_birth"))
            # Remove 'Birth Date' rows
            X = X[X["feature_name"] != "Birth Date"]
            # Ensure 'Birth Date' column is in datetime format
            X.rename(columns={'time_birth': 'Birth Date'}, inplace=True)
            X['Birth Date'] = pd.to_datetime(X['Birth Date'])
        elif 'Birth Date'  in X.columns():
            # Ensure 'Birth Date' column is in datetime format
            X['Birth Date'] = pd.to_datetime(X['Birth Date'])

        X['age_at_event(years)'] = (X['time'] - X['Birth Date']).dt.total_seconds() / (365.25 * 24 * 3600)
        # Calculate max age per patient
        max_ages = X.groupby("patient_id")["age_at_event(years)"].max()
        # Merge max age into dataset
        X = X.merge(max_ages.rename("max_age"), on="patient_id", how="left")
        return X

    def _calculate_mse(self, left_y: pd.Series, right_y: pd.Series) -> float:
        """Calculates the mean squared error for a given split."""
        left_mse = ((left_y - left_y.mean()) ** 2).mean() if len(left_y) > 0 else 0
        right_mse = ((right_y - right_y.mean()) ** 2).mean() if len(right_y) > 0 else 0
        return (left_mse * len(left_y) + right_mse * len(right_y)) / (len(left_y) + len(right_y))
    #
    def _calculate_split_score(self, left_y: pd.Series, right_y: pd.Series, criterion: str, is_classification: bool) -> float:
        """Calculates impurity score (MSE, MAE, Gini, Entropy, etc.) for a split."""

        if len(left_y) == 0 or len(right_y) == 0:
            return float('inf')  # Avoid empty splits

        total_size = len(left_y) + len(right_y)
        weight_left = len(left_y) / total_size
        weight_right = len(right_y) / total_size

        if not is_classification:
            # Regression-based criteria
            if criterion == "mse":
                mse_left = ((left_y - left_y.mean()) ** 2).mean()
                mse_right = ((right_y - right_y.mean()) ** 2).mean()
                return weight_left * mse_left + weight_right * mse_right

            elif criterion == "mae":
                mae_left = (left_y - left_y.median()).abs().mean()
                mae_right = (right_y - right_y.median()).abs().mean()
                return weight_left * mae_left + weight_right * mae_right


            else:
                raise ValueError(f"Unsupported regression criterion: {criterion}. Choose: 'mae' or 'mse'")

        else:
            # Classification-based criteria
            def gini_impurity(y: pd.Series) -> float:
                """Computes the Gini impurity of a set of class labels."""
                p = y.value_counts(normalize=True).to_numpy()  # Convert to NumPy array
                return float(1 - np.sum(p ** 2))  # Explicit float conversion

            def entropy(y: pd.Series) -> float:
                """Computes the entropy of a set of class labels."""
                p = y.value_counts(normalize=True).to_numpy()  # Convert to NumPy array
                return float(-np.sum(p * np.log2(p + 1e-9)))  # Explicit float conversion

            if criterion == "gini":
                return weight_left * gini_impurity(left_y) + weight_right * gini_impurity(right_y)

            elif criterion == "entropy":
                return weight_left * entropy(left_y) + weight_right * entropy(right_y)


            else:
                raise ValueError(f"Unsupported classification criterion: {criterion}. Choose 'gini' or 'entropy'")
            #

    def _get_time_window_mask(self, X: pd.DataFrame, feature: str, anchor_name, offset):
        """
        Returns a boolean mask for data['time'] being within the time window defined
        by anchor_feature_name ± offset_from_anchor (in months).
        
        Parameters:
            X (pd.DataFrame): The full dataset.
            feature (str): The feature to filter on.
            anchor_name (str, pd.Timestamp, or None): Anchor feature name or direct timestamp or None for no restriction.
            offset (int or None): Offset in months from anchor point.

        Returns:
            pd.Series: Boolean mask for X_feature with aligned index.
        """

        X_feature = X[X["feature_name"] == feature].copy()
        
        if X_feature.empty:
            return pd.Series([False] * len(X), index=X.index)

        if isinstance(anchor_name, str):
            reference_feature = X[X["feature_name"] == anchor_name][["patient_id", "value"]].copy()
            if len(reference_feature) != len(X["patient_id"].unique()):
                raise ValueError(f"Each patient must have one {anchor_name} entry in 'feature_name'.")
            
            reference_feature.rename(columns={"value": "reference_dates_list"}, inplace=True)
            X_feature = X_feature.merge(reference_feature, on="patient_id", how="left")

            if offset is None:
                return pd.Series([True] * len(X_feature), index=X_feature.index)

            if X_feature["reference_dates_list"].map(lambda x: isinstance(x, pd.Timestamp)).all():
                X_feature["date_relative_offset"] = X_feature["reference_dates_list"] + pd.DateOffset(months=offset)
                if offset >= 0:
                    mask = (X_feature["time"] <= X_feature["date_relative_offset"]) & (X_feature["time"] >= X_feature["reference_dates_list"])
                else:
                    mask = (X_feature["time"] >= X_feature["date_relative_offset"]) & (X_feature["time"] <= X_feature["reference_dates_list"])

            elif X_feature["reference_dates_list"].map(lambda x: isinstance(x, list)).all():
                X_feature["reference_dates_list"] = X_feature["reference_dates_list"].map(pd.to_datetime)
                X_feature["date_relative_offset"] = X_feature["reference_dates_list"].map(
                    lambda lst: [dt + pd.DateOffset(months=offset) for dt in lst]
                )
                def time_filter(times, offsets, time_val):
                    if offset >= 0:
                        return any((time_val <= off) and (ref <= time_val) for ref, off in zip(times, offsets))
                    else:
                        return any((time_val >= off) and (ref >= time_val) for ref, off in zip(times, offsets))
                mask = X_feature.apply(lambda row: time_filter(row["reference_dates_list"], row["date_relative_offset"], row["time"]), axis=1)

            else:
                raise ValueError("Anchor values should be Timestamps or lists of Timestamps.")

            return pd.Series(mask, index=X_feature.index)

        elif isinstance(anchor_name, pd.Timestamp):
            offset_time = anchor_name + pd.DateOffset(months=offset)
            if offset >= 0:
                mask = (X_feature["time"] >= anchor_name) & (X_feature["time"] <= offset_time)
            else:
                mask = (X_feature["time"] <= anchor_name) & (X_feature["time"] >= offset_time)
            return pd.Series(mask, index=X_feature.index)

        elif anchor_name is None:
            return pd.Series([True] * len(X_feature), index=X_feature.index)

        else:
            raise TypeError(f"Unsupported anchor or offset type: {type(anchor_name)}, {type(offset)}")

    def _predict_row(self, row: pd.DataFrame, tree: Dict) -> float:
        """Predicts the target value for a single row of data.""" # find something more effecient ........ !!!!!!!!!!
        
        # print(row)
        if 'value' in tree:
            # print(f'return: {tree["value"]}')
            return tree['value']
        anchor_name, offset = tree['time_window']
        if anchor_name is not None and isinstance(anchor_name, str) and ('<=' in anchor_name or '>' in anchor_name):
            # Dynamically reconstruct anchor from patient data
            feature = tree['feature']
            threshold = tree['threshold']

            patient_data = row[row['feature_name'] == feature]
            patient_data.loc[:, 'value'] = pd.to_numeric(patient_data['value'], errors='coerce')

            if '<=' in anchor_name:
                mask = patient_data['value'] <= threshold
            else:
                mask = patient_data['value'] > threshold

            anchor_dates = patient_data.loc[mask, 'time'].tolist()

            # Add new anchor row to patient data (just like in training) making sure anchors used in time_window are created if they’re missing from the test row.
            anchor_row = pd.DataFrame([{
                'patient_id': row['patient_id'].iloc[0],
                'time': np.nan,
                'feature_name': anchor_name,
                'value': anchor_dates
            }])

            row = pd.concat([row, anchor_row], ignore_index=True)

        aggregated_value = self._aggregate_feature_over_time(row, tree['feature'], tree['time_window'], tree['aggregation'])[0]
        # print(f'aggregated_value:{aggregated_value}')
        if aggregated_value <= tree['threshold']:
            return self._predict_row(row, tree['left'])
        else:
            return self._predict_row(row, tree['right'])

def split_data_(data,labels_df):
    n_all=len(data['patient_id'].unique())
    print(f'number of patients in the data: {n_all}')

    from sklearn.model_selection import train_test_split

    #split train - test datasets 

    # Stratified split to maintain label distribution
    train_patients, test_patients = train_test_split(
        labels_df["patient_id"], 
        test_size=0.3, 
        stratify=labels_df["label"],  # Ensures class balance
        random_state=42
    )

    # Create train and test sets
    train_data = data[data["patient_id"].isin(train_patients)]
    test_data = data[data["patient_id"].isin(test_patients)]

    train_labels = labels_df[labels_df["patient_id"].isin(train_patients)]
    test_labels = labels_df[labels_df["patient_id"].isin(test_patients)]

    y_train = train_labels['label']
    y_test = test_labels['label']

    n_train=len(train_data['patient_id'].unique())
    print(f'number of patients in the train: {n_train}')
    print(f'Precentage of label 1 in train: {y_train.sum()*100/n_train}')
    n_test=len(test_data['patient_id'].unique())
    print(f'number of patients in the test: {n_test}')
    return train_data, test_data, y_train, y_test
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data_balanced(data, labels_df):
    n_all = len(data['patient_id'].unique())
    print(f'Number of patients in the data: {n_all}')

    # Stratified split to maintain label distribution in the test set
    train_patients, test_patients = train_test_split(
        labels_df["patient_id"], 
        test_size=0.3, 
        stratify=labels_df["label"],  
        random_state=42
    )

    # Create train and test sets
    train_data = data[data["patient_id"].isin(train_patients)]
    test_data = data[data["patient_id"].isin(test_patients)]

    train_labels = labels_df[labels_df["patient_id"].isin(train_patients)]
    test_labels = labels_df[labels_df["patient_id"].isin(test_patients)]

    # Balance the training set (40% label 1)
    num_label_1 = train_labels[train_labels["label"] == 1].shape[0]
    total_train = num_label_1 / 0.4  # Total patients needed in train set
    target_label_1 = int(total_train * 0.4)
    target_label_0 = int(total_train * 0.6)

    # Oversample label 1 if needed
    if num_label_1 < target_label_1:
        label_1_patients = train_labels[train_labels["label"] == 1]
        oversampled_label_1 = label_1_patients.sample(n=target_label_1, replace=True, random_state=42)
    else:
        oversampled_label_1 = train_labels[train_labels["label"] == 1].sample(n=target_label_1, random_state=42)

    # Undersample label 0 to match 60% target
    undersampled_label_0 = train_labels[train_labels["label"] == 0].sample(n=target_label_0, random_state=42)

    # Combine to create a balanced train set
    balanced_train_labels = pd.concat([oversampled_label_1, undersampled_label_0])
    
    # Get final balanced train patient IDs
    balanced_train_patients = balanced_train_labels["patient_id"]

    # Filter the data to match the new train set
    balanced_train_data = data[data["patient_id"].isin(balanced_train_patients)]

    # Final labels
    y_train = balanced_train_labels['label']
    y_test = test_labels['label']

    # Print details
    n_train = len(balanced_train_data['patient_id'].unique())
    print(f'Number of patients in the balanced train: {n_train}')
    print(f'Percentage of label 1 in train: {y_train.mean() * 100:.2f}%')  # Should be close to 40%
    n_test = len(test_data['patient_id'].unique())
    print(f'Number of patients in the test: {n_test}')
    print(f'Percentage of label 1 in test: {y_test.mean() * 100:.2f}%')  # Should be close to 40%

    return balanced_train_data, test_data, y_train, y_test

#%% -----------------------------------------------------
#--------------------------------------------------------
# run on real data (Atopic Dermatitis form Synthea)

# def main():
# parser = argparse.ArgumentParser(description='Train Temporal Decision Tree Regressor')
# parser.add_argument('data_file', type=str, help='Path to the input data file')
# parser.add_argument('labels_file', type=str, help='Path to the input labels file')
# args = parser.parse_args()
import pandas as pd
# # Load data
data = pd.read_csv('synthea_dataset_AtopicDermatitis_9.csv')
data.loc[data['feature_name'] == 'Birth Date', 'value'] = pd.to_datetime(data.loc[data['feature_name'] == 'Birth Date', 'value'])
leak_features = ['Contact dermatitis', 'Application of moisturizer to skin','Skin condition care','Eczema']
data.loc[data['feature_name'] == 'Birth Date', 'feature_name'] = 'Birth Date Anchor'

data = data.loc[~data['feature_name'].isin(leak_features)]
# insure that the values are in the correct type (int)
for index, row in data.iterrows():
    if isinstance(row['value'], pd.Timestamp):
        continue  # Skip if it's already a timestamp
    try:
        data.at[index, 'value'] = pd.to_numeric(row['value'], errors='coerce')
    except:
        continue  # Skip if conversion fails

labels_df = pd.read_csv('labels_AtopicDermatitis_synthea_9.csv')

labels_20 = labels_df.head(50)
data_20 = data.loc[data['patient_id'].isin(labels_20['patient_id'])]

train_data, test_data, y_train, y_test = split_data_(data, labels_df)

#%% Run the model

anchors_list = ['Birth Date Anchor']
anchor_offset_list = [None, 6 , 12 ,-12, -6, 12*5] #12*5, 12*7, 
taus =[('Birth Date',12*5),(None,None),('Birth Date',12*7),('Birth Date',12*2)] # ,(None,None),(0,12), in each split, search whether to look into the first 3 months, the last 3 months, or None (all data points)
# Train model
model = TemporalDecisionTreeRegressor(max_depth=3,
                                    min_samples_split=0,
                                    anchors_list=anchors_list,
                                    anchor_offset_list=anchor_offset_list,
                                    time_windows=taus,
                                    is_classification=False,
                                    criterion='mse')
model.fit(train_data, y_train)

#%% Save the model
# import joblib
# joblib.dump(model, 'saved_models\model_with_attention.pkl')
import dill
from datetime import datetime

# Generate timestamped filename
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
filename = f'/home/noreena/TemporalDecisionTree/saved_models/model_with_attention_{timestamp}.dill'
# Save the model
with open(filename, 'wb') as f:
    dill.dump(model, f)
    
#%%
# Load the model
with open('/home/noreena/TemporalDecisionTree/saved_models/model_with_attention_2025-04-07_23-54-07.dill', 'rb') as f:
    model = dill.load(f)   
#%%
predictions = model.predict(test_data)

print(model.tree)
y_predicted = (predictions >= 0.5).astype(int)
results = model.evaluate(y_test,y_predicted,y_probabilities=predictions)
print(results)

# if __name__ == "__main__":
#     main()

#===================================================================================
# %% undersmapling for balanced data:
import pandas as pd
# # Load data
data = pd.read_csv('synthea_dataset_AtopicDermatitis_9.csv')
data.loc[data['feature_name'] == 'Birth Date', 'value'] = pd.to_datetime(data.loc[data['feature_name'] == 'Birth Date', 'value'])

labels_df = pd.read_csv('labels_AtopicDermatitis_synthea_9.csv')
# merged_df = data.merge(labels_df, on="patient_id")

label_counts = labels_df['label'].value_counts()

num_label_1 = int(label_counts.sum() * 0.4)  # 40% patients with label 1
num_label_0 = int(label_counts.sum() * 0.6)  # Remaining 60% with label 0
# Separate data by label
data_label_1 = labels_df[labels_df['label'] == 1]
data_label_0 = labels_df[labels_df['label'] == 0]
# Sample from label 0 to match the desired ratio
data_label_0_sampled = data_label_0.sample(n=num_label_0, random_state=42)
balanced_labels = pd.concat([data_label_1, data_label_0_sampled])
balanced_labels = balanced_labels.sample(frac=1, random_state=42).reset_index(drop=True)
balanced_labels['label'].value_counts()
balanced_data = data[data['patient_id'].isin(balanced_labels['patient_id'])]

# train_data_balanced, test_data_balanced, y_train_balanced, y_test_balanced = split_data_(balanced_data,balanced_labels)
train_data_balanced, test_data, y_train_balanced, y_test = split_data_balanced(balanced_data, balanced_labels)
# %%
taus =[('Birth Date',12*5),(None,None)] # ,(None,None),(0,12), in each split, search whether to look into the first 3 months, the last 3 months, or None (all data points)
# Train model
model_balanced = TemporalDecisionTreeRegressor(max_depth=3,
                                    min_samples_split=0,
                                    time_windows=taus,
                                    is_classification=False,
                                    criterion='mse')
model_balanced.fit(train_data_balanced, y_train_balanced)

predictions_balanced = model_balanced.predict(test_data)

print(model_balanced.tree)
y_predicted_balanced = (predictions_balanced >= 0.4).astype(int)
results = model_balanced.evaluate(y_test,y_predicted_balanced,y_probabilities=predictions_balanced)

print(results)
# =================================================================================
# %% run of hypethitical disease PIDS labels. 
# =================================================================================
import pandas as pd
# # Load data
data = pd.read_csv('synthea_dataset_AtopicDermatitis_9.csv')
data.loc[data['feature_name'] == 'Birth Date', 'value'] = pd.to_datetime(data.loc[data['feature_name'] == 'Birth Date', 'value'])
leak_features = ['Acute allergic reaction']
data_hypo = data.loc[~data['feature_name'].isin(leak_features)]
labels_df_hypothetical = pd.read_csv('hypothetical_PIDS_labels_synthea.csv')
labels_df_hypothetical.rename(columns={'PIDS_label':'label'},inplace=True)
train_data, test_data, y_train, y_test = split_data_(data_hypo,labels_df_hypothetical)

#%% Run the model
taus = [
    ('Birth Date', 12 * 3),   # First 3 years (critical for early-onset allergies)
    ('Birth Date', 12 * 5),   # First 5 years (immune development, infections)
    ('Birth Date', 12 * 7),   # First 7 years (metabolic issues, BMI, heart rate)
    ('Birth Date', 12 * 12),  # First 12 years (chronic conditions, hospital visits)
    (None, None),             # All data points (baseline case)
    # (0, 6),                   # First 6 months (early immune response assessment)
    # (-12, 0),                 # Last year (most recent health status before prediction)
]
# Train model
model_hypothetical = TemporalDecisionTreeRegressor(max_depth=3,
                                    min_samples_split=0,
                                    time_windows=taus,
                                    is_classification=False,
                                    criterion='mse')
model_hypothetical.fit(train_data, y_train)

predictions_hypothetical = model_hypothetical.predict(test_data)

print(model_hypothetical.tree)
y_predicted_hypothetical = (predictions_hypothetical >= 0.5).astype(int)
results_hypothetical = model_hypothetical.evaluate(y_test,y_predicted_hypothetical,y_probabilities=predictions_hypothetical)
print(results_hypothetical)

# %% new labels: 

import pandas as pd
# # Load data
data = pd.read_csv('synthea_dataset_AtopicDermatitis_9.csv')
data.loc[data['feature_name'] == 'Birth Date', 'value'] = pd.to_datetime(data.loc[data['feature_name'] == 'Birth Date', 'value'])
leak_features = ['Acute allergic reaction']
data_hypo = data.loc[~data['feature_name'].isin(leak_features)]
labels_df_hypothetical_2 = pd.read_csv('hypothetical_PIDS_labels_synthea_2_new.csv')
labels_df_hypothetical_2.rename(columns={'PIDS_label':'label'},inplace=True)
train_data, test_data, y_train, y_test = split_data_(data_hypo,labels_df_hypothetical_2)

#%% Run the model
taus = [
    ('Birth Date', 12 * 3),   # First 3 years (critical for early-onset allergies)
    ('Birth Date', 12 * 5),   # First 5 years (immune development, infections)
    ('Birth Date', 12 * 7),   # First 7 years (metabolic issues, BMI, heart rate)
    ('Birth Date', 12 * 12),  # First 12 years (chronic conditions, hospital visits)
    (None, None),             # All data points (baseline case)
    # (0, 6),                   # First 6 months (early immune response assessment)
    # (-12, 0),                 # Last year (most recent health status before prediction)
]
# Train model
model_hypothetical_2 = TemporalDecisionTreeRegressor(max_depth=3,
                                    min_samples_split=0,
                                    time_windows=taus,
                                    is_classification=False,
                                    criterion='mse')
model_hypothetical_2.fit(train_data, y_train)

predictions_hypothetical_2 = model_hypothetical_2.predict(test_data)

# print(model_hypothetical.tree)
y_predicted_hypothetical_2 = (predictions_hypothetical_2 >= 0.5).astype(int)
results_hypothetical_2 = model_hypothetical_2.evaluate(y_test,y_predicted_hypothetical_2,y_probabilities=predictions_hypothetical_2)
print(results_hypothetical_2)
# %% ===================== Hypothetical diabetes for synthea 9 

import pandas as pd
# # Load data
data = pd.read_csv('synthea_dataset_AtopicDermatitis_9.csv')

data.loc[data['feature_name'] == 'Birth Date', 'value'] = pd.to_datetime(data.loc[data['feature_name'] == 'Birth Date', 'value'])
# leak_features = ['Acute allergic reaction']
# data_hypo = data.loc[~data['feature_name'].isin(leak_features)]
labels_df_hypothetical_9 = pd.read_csv('hypothetical_diabetes_labels_synthea_9.csv')
labels_df_hypothetical_9.rename(columns={'diabetes_label':'label'},inplace=True)

train_data, test_data, y_train, y_test = split_data_(data, labels_df_hypothetical_9)

#%%Run the model
taus = [
    ('Birth Date', 12 * 3),   # First 3 years (critical for early-onset allergies)
    ('Birth Date', 12 * 5),   # First 5 years (immune development, infections)
    ('Birth Date', 12 * 7),   # First 7 years (metabolic issues, BMI, heart rate)
    ('Birth Date', 12 * 12),  # First 12 years (chronic conditions, hospital visits)
    (None, None),             # All data points (baseline case)
    # (0, 6),                   # First 6 months (early immune response assessment)
    # (-12, 0),                 # Last year (most recent health status before prediction)
]
# Train model
model_hypothetical_diabetes = TemporalDecisionTreeRegressor(max_depth=3,
                                    min_samples_split=0,
                                    time_windows=taus,
                                    is_classification=False,
                                    criterion='mse')
model_hypothetical_diabetes.fit(train_data, y_train)

predictions_hypothetical_9 = model_hypothetical_diabetes.predict(test_data)

# print(model_hypothetical.tree)
y_predicted_hypothetical_9 = (predictions_hypothetical_9 >= 0.5).astype(int)
results_hypothetical_9 = model_hypothetical_diabetes.evaluate(y_test,y_predicted_hypothetical_9,y_probabilities=predictions_hypothetical_9)
print(results_hypothetical_9)
# %%
