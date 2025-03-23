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


class TemporalDecisionTreeRegressor:
    def __init__(self, max_depth: int = 3, min_samples_split: int = 1, time_windows: List[Tuple] = [(0,3),(-6,-3),(None,None)] , aggregation_funcs: Dict[str, Callable] = None, is_classification = True , criterion = "entropy"):
        self.max_depth = max_depth
        self.is_classification = is_classification
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.time_windows = time_windows # (None, None) means look at all data, no specified window time
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

    def _build_tree(self, X: pd.DataFrame, y: pd.Series, depth: int) -> Dict:
        """
        Recursively builds the regression tree: recursively splits the data into smaller groups by finding the best split at each step

        Parameters:
            X: DataFrame - EHR Raw data with columns ['patient_id', 'time', 'feature_name', 'feature_value']. can contain multiple rows for one patient
            y: Series - Target variable (1,num_of_patients) - nd array
            time_col: str - Time column name
            depth: int - Current depth of the tree (default = 0)
            aggregation_cache: dict - Cache for storing computed feature aggregations (default = None)

        Returns:
            dict: A decision tree represented as a dictionary.
        """
        print('-'*40)
        print('-'*40)
        print(f"Building tree at depth {depth}...")
        # print(f'input data:\n{X}')
        # print(f'len of data:{len(X)}')

        # if len(X['patient_id'].unique()) <= self.min_samples_split or depth >= self.max_depth:
        #     print("Stop: (len(X['patient_id'].unique()) < self.min_samples_split) or (depth >= max_depth) -> leaf node")
        #     # print(f'return: {y.mean()}')
        #     if self.is_classification:
        #       most_common_label = np.bincount(y).argmax()  # Get the most frequent class
        #       print(f'classification task ==> return: {most_common_label}')
        #       return {'value': most_common_label}
        #     else:
        #       print(f'return: {y.mean()}')
        #       return {'value': y.mean()}
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
            if len(np.unique(y)) == 1:  # All labels are the same
                print(f"Homogeneous classification node ==> return: {y.iloc[0]}")
                return {'value': y.iloc[0]}

        else:
            if np.all(y == y.iloc[0]):  # All values are identical
                print(f"Homogeneous regression node ==> return: {y.iloc[0]}")
                return {'value': y.iloc[0]}

        best_split = self._find_best_split(X, y)
        if not best_split:
            print('Stop, no valid split is found -> the node is a leaf node.')
            if self.is_classification:
              most_common_label = np.bincount(y).argmax()  # Get the most frequent class
            #   print(f'return: {most_common_label}')
              return {'value': most_common_label, 'num_patients': num_patients}
            else:
            #   print(f'return: {y.mean()}')
              return {'value': y.mean(), 'num_patients': num_patients}

        print(f'==> Best split is for:{best_split}')
        unique_patient_ids = X['patient_id'].unique()
        aggregated_values = self._aggregate_feature_over_time(X, best_split['feature'], best_split['time_window'], best_split['aggregation'])
        # print(f'aggregated_values: {aggregated_values}')

        left_mask = aggregated_values <= best_split['threshold']
        right_mask = ~left_mask
        # print(f'left_mask:{left_mask}')
        # print(f'right_mask:{right_mask}')

        left_patient_ids = unique_patient_ids[left_mask]
        right_patient_ids = unique_patient_ids[right_mask]
        # print(f'left_patient_ids:{left_patient_ids}')
        # print(f'right_patient_ids:{right_patient_ids}')
        X_next_left = X[X['patient_id'].isin(left_patient_ids)]
        X_next_right = X[X['patient_id'].isin(right_patient_ids)]
        print(f"Splitting: {num_patients} patients -> Left: {len(left_patient_ids)}, Right: {len(right_patient_ids)}")

        print('entering recursive call')
        left_tree = self._build_tree(X_next_left, y[left_mask], depth + 1)
        print('finished the left side of the tree, strat the right side...')
        right_tree = self._build_tree(X_next_right, y[right_mask], depth + 1)

        # print(f'return: {{"feature": {best_split["feature"]}, "threshold": {best_split["threshold"]}, "left": {left_tree}, "right": {right_tree}}}')
        return {'feature': best_split['feature'],
                'threshold': best_split['threshold'],
                'aggregation': best_split['aggregation'],
                'time_window': best_split['time_window'],
                'num_patients': num_patients, # indicates how many unique patients reached this node.
                'left': left_tree, 'right': right_tree}

    def _find_best_split(self, X: pd.DataFrame, y: pd.Series) -> Dict:
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

        for feature, tau , agg_name in product(feature_names, self.time_windows, self.aggregation_funcs.keys()): # would change when changin (anchor,int) to 2 lists...
            if feature in ['Birth Date', 'age_at_event(years)','Pregnancy Start Date', 'max_age', 'age_start', 'age_end']: # or if feature in Anchor_list
              continue # here , skip "Anchor features" , ------------
            
            print(f'\ntesting {(feature, tau , agg_name)}')
            aggregated_values = self._aggregate_feature_over_time(X, feature, tau , agg_name)
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
                        'time_window': tau #Tuple (tau_start,tau_end)
                    }
        return best_split


    # def _aggregate_feature_over_time(self, X: pd.DataFrame, feature: str, tau: Tuple , aggregation: str) -> np.ndarray:
    #     """
    #     Aggregates feature values within a specified time window using a given aggregation function.
    #         -x :EHR raw data
    #         feature: a feature name on which we are aggregating
    #         tau: Tuple(Anchor_feature_name,offset_from_anchor) - the delta in months to define the time window
    #         aggregation: the aggregation function name wanted: min,max,avg,median,count ....
    #       """
    #     # print(f'Entered aggregate_features function.')

    #     Anchor_feature_name, offset_from_anchor = tau # [(0,3),(-60,-3),(None,None)]
    #     # ---- supoort different types of taus ----
    #     if isinstance(Anchor_feature_name, str):
    #         # print('str')

    #         if len(X[X['feature_name']==Anchor_feature_name])!=len(X['patient_id'].unique()) : #| (reference_feature_name not in X['feature_name'].unique())
    #           raise ValueError(f"Input data must contain {Anchor_feature_name} as an Anchor in 'feature_name' columns, once for each patient.")
            
    #         date_reference_column = X[X["feature_name"] == Anchor_feature_name][['patient_id','value']] #df
    #         date_reference_column.rename(columns={'value':'reference_dates_list'},inplace=True)
    #         X = X.merge(date_reference_column,on='patient_id',how='left')            
    #         X["time"] = pd.to_datetime(X['time'])
           
    #         if X['reference_dates_list'].apply(lambda x: isinstance(x, pd.Timestamp)).all():

    #             # X['reference_dates_list'] = pd.to_datetime(X['reference_dates_list'])
    #             X["date_relative_offset"] = X['reference_dates_list']  + pd.DateOffset(months=offset_from_anchor)
    #             # print(f'    Aggregating {feature} over {tau_end} months from {Anchor_feature_name}.')
    #             mask = (
    #                 (X["feature_name"] == feature) & (
    #                 ((offset_from_anchor >= 0) & (X["time"] <= X["date_relative_offset"]) & (X["reference_dates_list"] <= X["time"])) |
    #                 ((offset_from_anchor < 0) & (X["time"] >= X["date_relative_offset"]) & (X["reference_dates_list"] >= X["time"])))
    #             )

    #         elif X['reference_dates_list'].apply(lambda x: isinstance(x, list)).all(): # if all values in the column are lists. 

    #             # tau_end relatively to Anchor_feature_name
    #             # Adjust tau range dynamically based on each patient's date_reference
    #             # Convert reference_dates_list to datetime (ensuring each list entry is datetime)
    #             X['reference_dates_list'] = X['reference_dates_list'].apply(lambda lst: [pd.to_datetime(dt) for dt in lst])

    #             # Compute relative offsets for each timestamp in the list
    #             X["date_relative_offset"] = X['reference_dates_list'].apply(lambda lst: [dt + pd.DateOffset(months=offset_from_anchor) for dt in lst])

    #             # Function to check if any timestamp in the list meets the condition
    #             def check_time_conditions(row):
    #                 times = row["reference_dates_list"]
    #                 offsets = row["date_relative_offset"]
    #                 time_val = row["time"]
                    
    #                 if offset_from_anchor >= 0:
    #                     return any((time_val <= offset) and (ref_date <= time_val) for ref_date, offset in zip(times, offsets))
    #                 else:
    #                     return any((time_val >= offset) and (ref_date >= time_val) for ref_date, offset in zip(times, offsets))

    #             # Apply the filtering condition
    #             mask = (X["feature_name"] == feature) & X.apply(check_time_conditions, axis=1)
    #         else:
    #             raise ValueError(f"The value of {Anchor_feature_name} Anchor should be pd.Timestamps or a List of pd.Timestamps.")

    #     elif isinstance(Anchor_feature_name, pd.Timestamp):
    #         print("Anchor is a pandas Timestamp")
    #         # if tau_start.year != 0000:
    #         #     has_year = True # the date contains a specific year
    #         anchor_date = Anchor_feature_name
    #         mask = (
    #             (X["feature_name"] == feature) & (
    #             ((anchor_date >= 0) & (X["time"] <= anchor_date + pd.DateOffset(months=offset_from_anchor)) & (anchor_date <= X["time"])) |
    #             ((anchor_date < 0) & (X["time"] >= anchor_date + pd.DateOffset(months=offset_from_anchor)) & (anchor_date >= X["time"])))
    #         )
            

    #     # Compute row-wise age_start and age_end dynamically
    #     elif Anchor_feature_name is None and offset_from_anchor is None:
    #         mask = X["feature_name"] == feature  # No filtering, return all feature data

    #     elif isinstance(Anchor_feature_name,int) and isinstance(offset_from_anchor,int) :
    #         # Adjust tau range dynamically based on each patient's max_age
    #         X = self.add_age_at_event(X) #!!!
    #         X["age_start"] = np.where(
    #             Anchor_feature_name < 0, X["max_age"] + Anchor_feature_name / 12, Anchor_feature_name/12
    #         )
    #         X["age_end"] = np.where(
    #             offset_from_anchor < 0, X["max_age"] + offset_from_anchor / 12, offset_from_anchor/12
    #         )

    #         if Anchor_feature_name < 0 and offset_from_anchor > 0:
    #             raise ValueError("Invalid tau range: tau_start < 0 and offset_from_anchor > 0 is not supported.")

    #         mask = (
    #             (X["feature_name"] == feature) &
    #             (X["age_at_event(years)"] >= X["age_start"]) &
    #             (X["age_at_event(years)"] <= X["age_end"])
    #         )
    #     else:
    #       raise TypeError(f"time_windows input type is unspported ({type(tau[0]),type(tau[1])}).\ntime_windows input should be of type: Tuple(timestamps,int) or Tuple(int,int) or Tuple(str,int) or Tuple(None,None)")

    #     filtered_X = X[mask]
    #     filtered_X['value'] = pd.to_numeric(filtered_X['value'], errors='coerce')
    #     # print(f'filtered_X:\n{filtered_X}')
    #     if aggregation == "count":
    #         agg_result = filtered_X.groupby("patient_id")["value"].count().reset_index()
    #     else:
    #         agg_result = filtered_X.groupby("patient_id")["value"].agg(aggregation).reset_index()

    #     # print(f'agg_result:{agg_result}')

    #     # Ensure all patient IDs are present (even if they had no valid feature values)
    #     all_patients = pd.DataFrame({'patient_id': X['patient_id'].unique()})  # Get all unique patient IDs
    #     final_df = all_patients.merge(agg_result, on='patient_id', how='left')  # Left join to keep all patients
    #     # print('aggregated:')
    #     # print(final_df)
    #     result = final_df['value'].values

    #     return result


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
            print('X_feature is empty, return Nan')
            final_df = pd.DataFrame({"patient_id": X["patient_id"].unique(),"value":np.array([np.nan] * len(X["patient_id"].unique()))})
            return final_df["value"].values


        if isinstance(Anchor_feature_name, str):
            # Check if Anchor_feature_name appears exactly once per patient
            reference_feature = X[X["feature_name"] == Anchor_feature_name][["patient_id", "value"]].copy()

            if len(reference_feature) != len(X["patient_id"].unique()):
                raise ValueError(f"Each patient must have one {Anchor_feature_name} entry in 'feature_name'.")

            reference_feature.rename(columns={"value": "reference_dates_list"}, inplace=True)
            X_feature = X_feature.merge(reference_feature, on="patient_id", how="left")  
            if X_feature["reference_dates_list"].map(lambda x: isinstance(x, pd.Timestamp)).all():
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
                raise ValueError(f"The value of {Anchor_feature_name} Anchor should be pd.Timestamps or a List of pd.Timestamps.")

        elif isinstance(Anchor_feature_name, pd.Timestamp):
            anchor_date = Anchor_feature_name
            X_feature["date_relative_offset"] = anchor_date + pd.DateOffset(months=offset_from_anchor)

            if offset_from_anchor >= 0:
                mask = (X_feature["time"] <= X_feature["date_relative_offset"]) & (anchor_date <= X_feature["time"])
            else:
                mask = (X_feature["time"] >= X_feature["date_relative_offset"]) & (anchor_date >= X_feature["time"])

        elif Anchor_feature_name is None and offset_from_anchor is None:
            mask = np.ones(len(X_feature), dtype=bool)  # Select all rows

        elif isinstance(Anchor_feature_name, int) and isinstance(offset_from_anchor, int):
            X_feature = self.add_age_at_event(X_feature)  # Precompute age_at_event(years)
            X_feature["age_start"] = np.where(
                Anchor_feature_name < 0, X_feature["max_age"] + Anchor_feature_name / 12, Anchor_feature_name / 12
            )
            X_feature["age_end"] = np.where(
                offset_from_anchor < 0, X_feature["max_age"] + offset_from_anchor / 12, offset_from_anchor / 12
            )

            if Anchor_feature_name < 0 and offset_from_anchor > 0:
                raise ValueError("Invalid tau range: tau_start < 0 and offset_from_anchor > 0 is not supported.")

            mask = (X_feature["age_at_event(years)"].between(X_feature["age_start"], X_feature["age_end"]))

        else:
            raise TypeError(f"Unsupported tau type: {type(tau[0]), type(tau[1])}")

        # Apply mask
        filtered_X = X_feature[mask].copy()
        
        # Ensure 'value' is numeric
        filtered_X["value"] = pd.to_numeric(filtered_X["value"], errors="coerce")

        # Aggregate values
        agg_result = filtered_X.groupby("patient_id")["value"].agg(aggregation).reset_index() if aggregation != "count" \
            else filtered_X.groupby("patient_id")["value"].count().reset_index()

        # Ensure all patient IDs are present
        final_df = pd.DataFrame({"patient_id": X["patient_id"].unique()}).merge(agg_result, on="patient_id", how="left")

        return final_df["value"].values

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


    def _predict_row(self, row: pd.DataFrame, tree: Dict) -> float:
        """Predicts the target value for a single row of data.""" # find something more effecient ........ !!!!!!!!!!
        
        print("Predicting row...")
        # print(row)
        if 'value' in tree:
            # print(f'return: {tree["value"]}')
            return tree['value']
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
data = data.loc[~data['feature_name'].isin(leak_features)]
labels_df = pd.read_csv('labels_AtopicDermatitis_synthea_9.csv')
train_data, test_data, y_train, y_test = split_data_(data,labels_df)
#%% Run the model
taus =[('Birth Date',12*5),(None,None),('Birth Date',12*7),('Birth Date',12*2)] # ,(None,None),(0,12), in each split, search whether to look into the first 3 months, the last 3 months, or None (all data points)
# Train model
model = TemporalDecisionTreeRegressor(max_depth=3,
                                    min_samples_split=0,
                                    time_windows=taus,
                                    is_classification=False,
                                    criterion='mse')
model.fit(train_data, y_train)

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
