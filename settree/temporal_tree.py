#%%
import numpy as np
from collections import defaultdict
import math
import queue
from pandas import Timestamp
from itertools import product
from typing import Optional
import pandas as pd
from sklearn.utils import check_random_state
from sklearn.base import BaseEstimator
# from settree.set_data import SetDataset, OPERATIONS, _split_temporal_dataset, split_set_dataset, apply_operation, update_attention_set, mask_x_set
# from settree.splitters import SPLITTERS
from collections import deque
from typing import List, Tuple, Dict, Callable, Any
from .anchor_utils import create_branch_anchors, get_time_window_mask # my addition


#%%
class TemporalSplitNode:
    def __init__(
        self,
        feature: str = None,
        threshold: float = None,
        aggregation: str = None,
        time_window: tuple = None,
        missing_direction: str = "right",
        anchors_list: list = None,
        num_patients: int = 0,
        depth: int = 0,
        impurity: float = None, # value: float = None,  # ← NEW: For leaf prediction
        left=None,
        right=None
    ):
        
        """
        Represents a split node in a temporal decision tree.
        If `value` is not None, this node is a leaf.

        Parameters:
        - feature: Name of the feature used in the split.
        - threshold: Threshold for the split condition.
        - aggregation: Aggregation function used (e.g., 'mean', 'max').
        - time_window: (anchor_name, offset) defining the time window for aggregation.
        - missing_direction: Direction used for missing aggregated values.
        - anchors_list: List of anchors available at this node.
        - num_patients: Number of patients reaching this node.
        - depth: Depth of this node in the tree.
        - impurity: Optional impurity score at the node.
        - left: Left child node.
        - right: Right child node.
        """
        self.feature = feature
        self.threshold = threshold
        self.aggregation = aggregation
        self.time_window = time_window  # (anchor, offset)
        self.missing_direction = missing_direction
        self.anchors_list = anchors_list if anchors_list else []
        self.num_patients = num_patients
        self.depth = depth
        self.impurity = impurity
        # self.value = value  # ← For leaf nodes

        self.left = left
        self.right = right

    def is_leaf(self):
        return False

    def describe(self):
        anchor, offset = self.time_window
        offset_str = f"{offset} months" if offset is not None else "any time"
        return (
            f"[Depth {self.depth}] {self.feature} ({self.aggregation}) <= {self.threshold:.2f} "
            f"within {offset_str} of {anchor}"
        )

    def __repr__(self):
        return (
            f"SplitNode(feature={self.feature}, threshold={self.threshold}, "
            f"aggregation={self.aggregation}, time_window={self.time_window}, "
            f"missing_direction={self.missing_direction}, "
            f"depth={self.depth}, num_patients={self.num_patients})"
        )

class TemporalLeaf: 
    def __init__(self, value, num_patients, depth):
        self.value = value
        self.num_patients = num_patients
        self.depth = depth

    def is_leaf(self):
        return True

    def __repr__(self):
        return f"Leaf(value={self.value}, num_patients={self.num_patients})"
    
    def describe(self):
        return f"Leaf: Predict = {self.value:.2f} (patients: {self.num_patients})"

class TemporalTree(BaseEstimator):
    """
    TemporalTree base class
    Inherited from sklearn's BaseEstimator (recommended)
    """

    def __init__(self,
                 max_depth=3,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 anchors_list=None,
                 anchor_offset_list=None,
                 aggregation_funcs=None,
                 is_classification=True,
                 criterion="entropy",
                 learn_missing_direction=False,
                 random_state=None):

        """
        Parameters
        ----------
        classifier : bool
            Whether the tree used for classification or regression

        criterion : {"gini", "entropy", "mse"}, default="entropy"
            The function to measure the quality of a split.

        max_features : int, float, {"auto", "sqrt", "log2"} or None, default="auto"
            The number of features to consider when looking for the best split:
                - If int, then consider `max_features` features at each split.
                - If float, then `max_features` is a fraction and
                  `int(max_features * n_features)` features are considered at each
                  split.
                - If "auto", then `max_features=sqrt(n_features)`.
                - If "sqrt", then `max_features=sqrt(n_features)`.
                - If "log2", then `max_features=log2(n_features)`.
                - If None, then `max_features=n_features`.

        min_samples_split : int or float, default=2
            The minimum number of samples required to split an internal node:
            - If int, then consider `min_samples_split` as the minimum number.
            - If float, then `min_samples_split` is a fraction and
              `ceil(min_samples_split * n_samples)` are the minimum
              number of samples for each split

        max_depth : int, default=None
            The maximum depth of the tree. If None, then nodes are expanded until
            all leaves are pure or until all leaves contain less than
            min_samples_split samples.

        min_samples_leaf : int or float, default=1
            The minimum number of samples required to be at a leaf node.
            A split point at any depth will only be considered if it leaves at
            least ``min_samples_leaf`` training samples in each of the left and
            right branches.  This may have the effect of smoothing the model,
            especially in regression.
            - If int, then consider `min_samples_leaf` as the minimum number.
            - If float, then `min_samples_leaf` is a fraction and
              `ceil(min_samples_leaf * n_samples)` are the minimum
              number of samples for each node.

        random_state : int, RandomState instance or None, default=None
            Used to pick randomly the `max_features` used at each split.
        """

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.is_classification = is_classification
        self.criterion = criterion
        self.learn_missing_direction = learn_missing_direction
        self.random_state = random_state

        self.anchors_list = anchors_list or ['Birth Date Anchor']
        self.anchor_offset_list = anchor_offset_list or [None]
        self.aggregation_funcs = aggregation_funcs or {
            'mean': np.mean,
            'sum': np.sum,
            'max': np.max,
            'count': lambda x: x.size
        }
        self.tree = None
        
    def init(self, X, y):
        """
        Internal initialization at the beginning of .fit().
        Prepares tracking variables and validates core config.
        """
        # Set a safe default max_depth if None was passed
        self.max_depth_ = self.max_depth if self.max_depth is not None else np.inf

        # Set up random generator (for future random split / feature sampling)
        self._rng = check_random_state(self.random_state)

        # Dataset tracking
        self.n_samples_ = len(y)
        self.n_features_ = X['feature_name'].nunique()

        # Classification-specific
        if self.is_classification:
            self.n_classes_ = y.nunique()
        else:
            self.n_classes_ = None

        # Reset tree and leaf tracking
        self.tree = None
        self.n_leafs = 0
        self.leafs = []
        self.depth = 0

        if isinstance(self.min_samples_split, float):
            self.min_samples_split = math.ceil(self.min_samples_split * self.n_features)
    
    def _check_stopping_conditions(self, y, num_patients, depth) -> Optional[TemporalLeaf]:
        y_target = y["label"] if isinstance(y, pd.DataFrame) else y

        if num_patients < self.min_samples_split or depth >= self.max_depth_:
            print(f"Stopping: too few samples or max depth reached. samples:{num_patients}, depth:{depth}")
            prediction = y_target.mode()[0] if self.is_classification else y_target.mean()
            return self._create_leaf(prediction, num_patients, depth)

        if self.is_classification and y_target.nunique() == 1:
            print("Stopping: homogeneous classification label.")
            return self._create_leaf(y_target.iloc[0], num_patients, depth)

        elif not self.is_classification and y_target.nunique() == 1:
            print("Stopping: homogeneous regression value.")
            return self._create_leaf(y_target.iloc[0], num_patients, depth)

        return None
        
    # def _check_stopping_conditions(self, y, num_patients, depth) -> Optional[TemporalLeaf]:
    #     if num_patients <= self.min_samples_split or depth >= self.max_depth_:
    #         print(f"Stopping: too few samples or max depth reached. samples:{num_patients}, depth:{depth}")
    #         prediction = y.mode()[0] if self.is_classification else y.mean()
    #         return self._create_leaf(prediction, num_patients, depth)

    #     if self.is_classification and y.nunique() == 1:
    #         print("Stopping: homogeneous classification label.")
    #         return self._create_leaf(y.iloc[0], num_patients, depth)

    #     elif not self.is_classification and y.nunique() == 1:
    #         print("Stopping: homogeneous regression value.")
    #         return self._create_leaf(y.iloc[0], num_patients, depth)
    #     return None
    
    def _create_leaf(self, value, num_patients, depth): # __add_leaf
            """Create and register a leaf node with a unique ID (sn)."""
            leaf = TemporalLeaf(value=value, num_patients=num_patients, depth=depth)
            leaf.sn = self.n_leafs
            self.n_leafs += 1
            self.leafs.append(leaf)
            print(f'[_create_leaf] return at depth {depth} : \n{leaf} done with create leaf\n')
            return leaf
    
        # def __add_leaf(val, n_samples, curr_depth):
        #     l = TemporalLeaf(val, n_samples=n_samples, sn=self.n_leafs, depth=curr_depth)
        #     self.n_leafs += 1
        #     self.leafs.append(l)
        #     return l
        

    def _get_node(self, X, y, depth: int, anchors_list: list):
        """
        Constructs a node: returns a leaf if stopping condition is met,
        otherwise returns a split node (children not yet attached).
        """
        num_patients = len(y)

        stopping_node = self._check_stopping_conditions(y, num_patients, depth)
        if stopping_node: #stop expanding the tree at this point and return a leaf node if a stopping condition has been met.
            return stopping_node
        # print(f'[get_node] stopping_node: {stopping_node} . Continues to find the best split and grow the tree further.')
        print(f"[get_node] depth={depth}, num_patients={len(y)}, anchors={anchors_list}")

        best_split = self._find_best_split(X, y, anchors_list=anchors_list)

        if not best_split:
            print("Stopping: No valid split found.")
            prediction = y.mode()[0] if self.is_classification else y.mean()
            return self._create_leaf(prediction, num_patients, depth)

        return TemporalSplitNode(
            feature=best_split["feature"],
            threshold=best_split["threshold"],
            aggregation=best_split["aggregation"],
            time_window=best_split["time_window"],
            missing_direction=best_split.get("missing_direction", "right"),
            anchors_list=anchors_list,
            num_patients=num_patients,
            depth=depth,
            impurity=best_split.get("score", None)
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Train the temporal decision tree using queue-based BFS building.
        """
        print("Fitting TemporalDecisionTree...")
        self._validate_input(X, y)
        self.init(X, y)

        self.tree = self._build(X, y)
        return self

    def _build(self, X: pd.DataFrame, y: pd.DataFrame):
        """
        Builds the tree using BFS traversal and creates dynamic anchor rows
        after each split for temporal propagation.
        """
        anch_lst = self.anchors_list.copy()
        print(f'Inside _build - call _get_node with (x={len(X)}, y={len(y)}, {0}, anchors_list={anch_lst})')
        root = self._get_node(X, y, depth=0, anchors_list=self.anchors_list.copy())
        self.depth = 0
        print('_build function:')
        # print(f'Root: {root}')
        if root.is_leaf():
            return root

        queue = deque()
        queue.append((root, X, y, 0, self.anchors_list.copy()))
        # print(f'queue: {queue}')
        while queue:
            print(f'[build] depth={self.depth}')
            node, X_node, y_node, depth, anchors = queue.popleft()

            # --- 1. Aggregate for this split ---
            print('Inside _build - aggregate on best split')
            agg_values, _ = self._aggregate_feature_over_time(
                X_node, node.feature, node.time_window, node.aggregation
            )
            unique_patient_ids = X_node['patient_id'].unique()

            missing_mask = pd.isna(agg_values)
            with np.errstate(invalid='ignore'):
                non_missing_left_mask = (~missing_mask) & (agg_values <= node.threshold)
            if getattr(node, "missing_direction", "right") == "left":
                left_mask = non_missing_left_mask | missing_mask
            else:
                left_mask = non_missing_left_mask
            right_mask = ~left_mask

            y_left = y_node.loc[left_mask]
            y_right = y_node.loc[right_mask]

            # left_ids = y_left.index ############ bug
            # right_ids = y_right.index
            left_ids = unique_patient_ids[left_mask]
            right_ids = unique_patient_ids[right_mask]
        
            print(f'[_build] X_node len : {len(X_node)}')
            # print(f'[_build] X_node  : {X_node}')
            print(f'[_build] left_ids len : {len(left_ids)}')
            # print(f'[_build] y_left : {y_left}')
            # print(f'[_build] left_ids : {left_ids}')

            # split temporal data
            X_left = X_node[X_node['patient_id'].isin(left_ids)]
            X_right = X_node[X_node['patient_id'].isin(right_ids)]
            
            # print(f'[_build] X_left len : {len(X_left)}') 
            # print(f'[_build] X_right len : {len(X_right)}')

            # --- 2. Create new anchor names ---
            left_anchor = f"{node.feature} <= {np.round(node.threshold)} Anchor"
            right_anchor = f"{node.feature} > {np.round(node.threshold)} Anchor"
            left_anchors = anchors + [left_anchor]
            right_anchors = anchors + [right_anchor]

            # --- 3. Generate new anchor rows for each group ---
            left_anchor_df = create_branch_anchors(
                patient_ids=left_ids,
                X_branch=X_left,
                anchor_name=left_anchor,
                direction="left",
                feature=node.feature,
                threshold=node.threshold,
                time_window=node.time_window
            )
            right_anchor_df = create_branch_anchors(
                patient_ids=right_ids,
                X_branch=X_right,
                anchor_name=right_anchor,
                direction="right",
                feature=node.feature,
                threshold=node.threshold,
                time_window=node.time_window
            )

            # --- 4. Append anchor rows to event data ---
            X_left_next = pd.concat([X_left, left_anchor_df], ignore_index=True)
            X_right_next = pd.concat([X_right, right_anchor_df], ignore_index=True)

            # print(f'left_anchors: {left_anchor}')
            # --- 5. Create child nodes with updated data and anchors ---
            node.left = self._get_node(X_left_next, y_left, depth + 1, left_anchors)
            node.right = self._get_node(X_right_next, y_right, depth + 1, right_anchors)

            if not node.left.is_leaf():
                queue.append((node.left, X_left_next, y_left, depth + 1, left_anchors))
            if not node.right.is_leaf():
                queue.append((node.right, X_right_next, y_right, depth + 1, right_anchors))

            self.depth = max(self.depth, depth + 1)

        return root


    def compute_feature_importances(self, normalize: bool = True):
        """
        Computes feature importances based on total impurity reduction across the tree.

        Returns:
            Dict[str, float]: Mapping from feature name to importance score (normalized if requested).
        """
        importances = defaultdict(float)

        def _recurse(node):
            if node is None or node.is_leaf():
                return

            # Retrieve impurity and number of samples for parent and children
            parent_imp = node.impurity
            n = node.num_patients

            left = node.left
            right = node.right

            if left is None or right is None:
                return

            n_l = left.num_patients
            n_r = right.num_patients
            imp_l = getattr(left, 'impurity', 0)
            imp_r = getattr(right, 'impurity', 0)

            gain = (parent_imp * n) - (imp_l * n_l + imp_r * n_r)
            importances[node.feature] += gain

            _recurse(left)
            _recurse(right)

        _recurse(self.tree)

        # Convert to dict and normalize
        importances = dict(importances)

        if normalize:
            total = sum(importances.values())
            if total > 0:
                importances = {k: v / total for k, v in importances.items()}

        return importances

# here
    # def _traverse(self, X_set, node):

    #     # if reached a leaf - exit
    #     while not isinstance(node, Leaf):
    #         X, _, _ = apply_operation(X_set=X_set,
    #                                   mask_inds=[0],
    #                                   op=node.op,
    #                                   attention_set_level=node.use_attention_set,
    #                                   attention_set_comp=node.use_attention_set_comp)

    #         # record the current's node attention set
    #         update_attention_set(X_set=X_set,
    #                              mask_inds=[0],
    #                              op=node.op,
    #                              feat=node.feature,
    #                              thresh=node.threshold,
    #                              attention_set_level=node.use_attention_set,
    #                              attention_set_comp=node.use_attention_set_comp)

    #         # if empty - turn left
    #         if not len(X):
    #             node = node.left
    #         else:
    #             if X[0][node.feature] < node.threshold:
    #                 node = node.left
    #             else:
    #                 node = node.right
    #     return node

    # def _traverse_with_memory(self, X_set, node, nodes_out=False):

    #     # if reached a leaf - exit
    #     nodes = []
    #     while not isinstance(node, Leaf):
    #         nodes.append(node.__dict__)

    #         X, _, _ = apply_operation(X_set=X_set,
    #                                   mask_inds=[0],
    #                                   op=node.op,
    #                                   attention_set_level=node.use_attention_set,
    #                                   attention_set_comp=node.use_attention_set_comp)

    #         # record the current's node attention set
    #         update_attention_set(X_set=X_set,
    #                              mask_inds=[0],
    #                              op=node.op,
    #                              feat=node.feature,
    #                              thresh=node.threshold,
    #                              attention_set_level=node.use_attention_set,
    #                              attention_set_comp=node.use_attention_set_comp)

    #         # if empty - turn left
    #         if not len(X):
    #             node = node.left
    #         else:
    #             if X[0][node.feature] < node.threshold:
    #                 node = node.left
    #             else:
    #                 node = node.right

    #     if nodes_out:
    #         return nodes, X_set.attention_set[0]
    #     else:
    #         return X_set.attention_set[0]

    # def _get_leaf_val(self, X_set, node, prob=False):
    #     node = self._traverse(X_set, node)
    #     # whet reached a leaf - report its value
    #     if self.classifier:
    #         return node.value if prob else node.value.argmax()
    #     return node.value

    # def _get_leaf_ind(self, X_set, node):
    #     node = self._traverse(X_set, node)
    #     return node.sn

    # def _get_leaf_val_and_ind(self, X_set, node, prob=False):
    #     node = self._traverse(X_set, node)
    #     # whet reached a leaf - report its value
    #     if self.classifier:
    #         return node.value if prob else node.value.argmax()
    #     return node.value, node.sn

    # def compute_feature_importances(self, normalize=True):
    #     """Computes the importance of each feature (aka variable)."""

    #     def get_all_nodes(root, mem):
    #         if isinstance(root.right, SetSplitNode) and isinstance(root.left, SetSplitNode):
    #             mem.append(root)
    #         if isinstance(root.right, SetSplitNode):
    #             get_all_nodes(root.right, mem)
    #         if isinstance(root.left, SetSplitNode):
    #             get_all_nodes(root.left, mem)
    #         return mem

    #     importances = np.zeros((self.n_features,))
    #     nodes = get_all_nodes(self.tree_, mem=[])
    #     for split_node in nodes:
    #             importances[split_node.feature] += (split_node.weighted_n_node_samples * split_node.impurity -
    #                                                 split_node.left.weighted_n_node_samples * split_node.left.impurity -
    #                                                 split_node.right.weighted_n_node_samples * split_node.right.impurity)

    #     importances /= nodes[0].weighted_n_node_samples

    #     if normalize:
    #         normalizer = np.sum(importances)

    #         if normalizer > 0.0:
    #             # Avoid dividing by zero (e.g., when root is pure)
    #             importances /= normalizer

    #     return importances

    def _get_leaf_id(self, row: pd.DataFrame, node) -> int:
        """Returns the unique ID (sn) of the leaf node reached by the patient."""
        if node.is_leaf():
            return getattr(node, 'sn', -1)

        aggregated_value = self._aggregate_feature_over_time(
            row, node.feature, node.time_window, node.aggregation
        )[0]

        direction = self._get_prediction_direction(aggregated_value, node)
        row = self._append_prediction_branch_anchor(row, node, direction)
        next_node = node.left if direction == "left" else node.right

        return self._get_leaf_id(row, next_node)

    def _get_prediction_direction(self, aggregated_value, node) -> str:
        if pd.isna(aggregated_value):
            return "left" if getattr(node, "missing_direction", "right") == "left" else "right"
        return "left" if aggregated_value <= node.threshold else "right"

    def _append_prediction_branch_anchor(self, row: pd.DataFrame, node, direction: str) -> pd.DataFrame:
        patient_id = row["patient_id"].iloc[0]
        if direction == "left":
            anchor_name = f"{node.feature} <= {np.round(node.threshold)} Anchor"
        else:
            anchor_name = f"{node.feature} > {np.round(node.threshold)} Anchor"

        prev_anchor_name, offset = node.time_window
        patient_data = row[row["feature_name"] == node.feature]

        if patient_data.empty:
            anchor_dates = []
        else:
            value_mask = (
                patient_data["value"].astype(float) <= node.threshold
                if direction == "left"
                else patient_data["value"].astype(float) > node.threshold
            )

            anchor_value = row[row["feature_name"] == prev_anchor_name]["value"]
            if anchor_value.empty or pd.isnull(anchor_value.values[0]):
                time_mask = pd.Series([False] * len(patient_data), index=patient_data.index)
            else:
                anchor_raw = anchor_value.values[0]
                if isinstance(anchor_raw, pd.Timestamp):
                    anchor_time = pd.to_datetime(anchor_raw)
                    offset_time = anchor_time + pd.DateOffset(months=offset)
                    time_mask = (
                        (patient_data["time"] >= anchor_time) & (patient_data["time"] <= offset_time)
                        if offset >= 0 else
                        (patient_data["time"] <= anchor_time) & (patient_data["time"] >= offset_time)
                    )
                elif isinstance(anchor_raw, list):
                    anchor_times = pd.to_datetime(anchor_raw)
                    offset_times = [dt + pd.DateOffset(months=offset) for dt in anchor_times]

                    def time_filter(time_val):
                        return any(
                            (time_val <= o and r <= time_val) if offset >= 0 else (time_val >= o and r >= time_val)
                            for r, o in zip(anchor_times, offset_times)
                        )

                    time_mask = patient_data["time"].apply(time_filter)
                else:
                    time_mask = pd.Series([False] * len(patient_data), index=patient_data.index)

            combined_mask = value_mask & time_mask
            anchor_dates = patient_data.loc[combined_mask, "time"].tolist()

        anchor_df = pd.DataFrame([{
            "patient_id": patient_id,
            "time": np.nan,
            "feature_name": anchor_name,
            "value": anchor_dates
        }])
        return pd.concat([row, anchor_df], ignore_index=True)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.tree is None:
            raise ValueError("Model not fitted. Call `.fit()` first.")
        if self.is_classification:
            return np.argmax(self.predict_proba(X), axis=1)

        X = X.copy()
        X["time"] = pd.to_datetime(X["time"])

        return np.array([
            self._predict_row(patient_data, self.tree)
            for _, patient_data in X.groupby("patient_id", sort=False)
        ])
   
    def _predict_row(self, row: pd.DataFrame, node) -> float:
        """
        Recursively traverses the tree to predict a value for one patient's data.

        Parameters:
            row (pd.DataFrame): All events for a single patient (long-format).
            node (TemporalSplitNode or TemporalLeaf): Current node in the tree.

        Returns:
            float: The predicted value (regression or class label).
        """
        if node.is_leaf():
            return node.value if not self.is_classification else np.argmax(node.value)

        # Aggregate and make the decision
        aggregated_value = self._aggregate_feature_over_time(
            row, node.feature, node.time_window, node.aggregation
        )[0]

        direction = self._get_prediction_direction(aggregated_value, node)
        row = self._append_prediction_branch_anchor(row, node, direction)
        next_node = node.left if direction == "left" else node.right
        return self._predict_row(row, next_node)

    def apply(self, X: pd.DataFrame) -> np.ndarray:
        """Returns the ID of the leaf each patient lands in."""
        if self.tree is None:
            raise ValueError("Model not fitted.")
        
        X = X.copy()
        X["time"] = pd.to_datetime(X["time"])
        patient_ids = X['patient_id'].unique()
        leaf_ids = np.zeros(len(patient_ids), dtype=int)

        for i, pid in enumerate(patient_ids):
            patient_data = X[X['patient_id'] == pid]
            leaf_ids[i] = self._get_leaf_id(patient_data, self.tree)

        return leaf_ids

    def _predict_proba_row(self, row: pd.DataFrame, node) -> np.ndarray:
        if node.is_leaf():
            return node.value  # expected to be a probability vector

        aggregated_value = self._aggregate_feature_over_time(
            row, node.feature, node.time_window, node.aggregation
        )[0]

        direction = self._get_prediction_direction(aggregated_value, node)
        row = self._append_prediction_branch_anchor(row, node, direction)
        next_node = node.left if direction == "left" else node.right
        return self._predict_proba_row(row, next_node)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_classification:
            raise ValueError("`predict_proba` is only supported for classification trees.")

        if self.tree is None:
            raise ValueError("Model not fitted. Please train the model first.")

        X = X.copy()
        X["time"] = pd.to_datetime(X["time"])
        patient_ids = X['patient_id'].unique()
        probabilities = []

        for pid in patient_ids:
            patient_data = X[X['patient_id'] == pid]
            probabilities.append(self._predict_proba_row(patient_data, self.tree))

        return np.vstack(probabilities)

    def predict_and_apply(self, X: pd.DataFrame):
        """Returns both the prediction and the leaf ID for each patient."""
        if self.tree is None:
            raise ValueError("Model not fitted. Please train the model first.")
        
        X = X.copy()
        X["time"] = pd.to_datetime(X["time"])
        patient_ids = X['patient_id'].unique()
        predictions = []
        leaf_ids = []
        for pid in patient_ids:
            patient_data = X[X['patient_id'] == pid]
            predictions.append(self._predict_row(patient_data, self.tree))
            leaf_ids.append(self._get_leaf_id(patient_data, self.tree))
        return np.array(predictions), np.array(leaf_ids)

    # def _get_decision_path(self, X_set, node):
    #     m = self._traverse_with_memory(X_set, node, nodes_out=False)
    #     return m

    # def _get_detailed_decision_path(self, X_set, node):
    #     nodes, m = self._traverse_with_memory(X_set, node, nodes_out=True)
    #     return nodes, m

    # def decision_path(self, X_set):
    #     return [self._get_decision_path(SetDataset(records=[x]), self.tree_) for x in X_set.records]

    # def detailed_decision_path(self, X_set):
    #     return [self._get_detailed_decision_path(SetDataset(records=[x]), self.tree_) for x in X_set.records]

    @property
    def n_nodes(self):
        def count_nodes(root):
            if isinstance(root, TemporalLeaf):
                return 0

            res = 0
            if (root.left and root.right):
                res += 1
            res += (count_nodes(root.left) +
                    count_nodes(root.right))
            return res
        return count_nodes(self.tree_)

# ==================== add methods from my class

    def _find_best_split(self, X: pd.DataFrame, y: pd.Series, anchors_list = None):
        """ Finds the best split (feature, threshold, time and aggregation function) using temporal aggregations efficiently.
        X- raw EHR data
        y- target nd array
        """
        print("Finding best split...")

        best_split = None
        best_score= float('inf')
        num_samples = len(y)
        feature_names = X['feature_name'].unique()
        anchor_reference_cache = {}
        feature_subset_cache = {}
        feature_anchor_merge_cache = {}
        time_window_mask_cache = {}
        filtered_numeric_cache = {}
        aggregation_result_cache = {}
        # print(f'Features names: {feature_names}')
        
        if anchors_list is None: # the root node will only choose from the given anchors
            anchors_list = self.anchors_list.copy()

        time_windows = []
        seen_full_timeline_window = False
        for anchor, offset in product(anchors_list, self.anchor_offset_list):
            if offset is None:
                if seen_full_timeline_window:
                    continue
                time_windows.append((None, None))
                seen_full_timeline_window = True
            else:
                time_windows.append((anchor, offset))
        
        for feature, (anchor, offset), agg_name in product(feature_names,
                                                           time_windows,
                                                           self.aggregation_funcs.keys()):
            # print('best split entered loop')
            if (feature == 'Birth Date') or (feature.endswith('Anchor')):  # feature in self.anchors_list
                # print(f'skip {feature}')
                continue  # Skip anchor features and 'Birth Date'
            
            tau = (anchor, offset)  # Create tau tuple
            print(f"\nTesting: feature={feature}, time window={tau}, agg={agg_name}")
            aggregated_values, anchor = self._aggregate_feature_over_time(
                X,
                feature,
                tau,
                agg_name,
                anchor_reference_cache=anchor_reference_cache,
                feature_subset_cache=feature_subset_cache,
                feature_anchor_merge_cache=feature_anchor_merge_cache,
                time_window_mask_cache=time_window_mask_cache,
                filtered_numeric_cache=filtered_numeric_cache,
                aggregation_result_cache=aggregation_result_cache,
            )
            # print(f'aggregated_values:{aggregated_values}')

            thresh_values = np.unique(pd.Series(aggregated_values).dropna())
            for threshold in thresh_values:
                # print(f'  threshold:{threshold}')
                missing_mask = pd.isna(aggregated_values)
                with np.errstate(invalid='ignore'):  # Suppress invalid comparison warnings
                    non_missing_left_mask = (~missing_mask) & (aggregated_values <= threshold)

                missing_directions = (
                    ["right", "left"]
                    if self.learn_missing_direction and np.any(missing_mask)
                    else ["right"]
                )
                for missing_direction in missing_directions:
                    if missing_direction == "left":
                        left_mask = non_missing_left_mask | missing_mask
                    else:
                        left_mask = non_missing_left_mask

                    right_mask = ~left_mask
                    # print(f'right_mask:\n{right_mask}')
                    # print(f'left_mask ( <= ):\n{left_mask}')
                    if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                        print(f'  This split has {np.sum(left_mask)} samples on the left, and {np.sum(right_mask)} on the right.')
                        print(f'  The split is skipped, it does not meet the minimum number of samples required for valid leaf nodes ({self.min_samples_leaf}).')
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
                            'time_window': (anchor, offset), # (chosen anchor, chosen offset)
                            'missing_direction': missing_direction,
                            'score': score  
                        }
        print("Best split:", best_split)

        return best_split


    def _aggregate_feature_over_time(
        self,
        X_original: pd.DataFrame,
        feature: str,
        tau: Tuple,
        aggregation: str,
        anchor_reference_cache=None,
        feature_subset_cache=None,
        feature_anchor_merge_cache=None,
        time_window_mask_cache=None,
        filtered_numeric_cache=None,
        aggregation_result_cache=None,
    ) -> np.ndarray:
        """
        Aggregates feature values within a specified time window using a given aggregation function.
        """
        Anchor_feature_name, offset_from_anchor = tau

        # Pre-filter the relevant feature
        if feature_subset_cache is not None and feature in feature_subset_cache:
            X_feature = feature_subset_cache[feature].copy()
        else:
            X_feature = X_original[X_original["feature_name"] == feature].copy()
            # Ensure "time" is datetime
            X_feature["time"] = pd.to_datetime(X_feature["time"])

            if feature_subset_cache is not None:
                feature_subset_cache[feature] = X_feature.copy()
        # If X_feature is empty, return NaN for the patient
        if X_feature.empty:
            # Ensure all patient IDs are present
            # print('X_feature is empty, return Nan')
            final_df = pd.DataFrame({"patient_id": X_original["patient_id"].unique(),"value":np.array([np.nan] * len(X_original["patient_id"].unique()))})
            return final_df["value"].values
        if offset_from_anchor is None:
            mask = np.ones(len(X_feature), dtype=bool)  # Select all rows
            Anchor_feature_name = None # in the case where we are
        
        if isinstance(Anchor_feature_name, str):
            # Check if Anchor_feature_name appears exactly once per patient
            mask_cache_key = (feature, Anchor_feature_name, offset_from_anchor)
            if anchor_reference_cache is not None and Anchor_feature_name in anchor_reference_cache:
                reference_feature = anchor_reference_cache[Anchor_feature_name].copy()
            else:
                reference_feature = X_original[X_original["feature_name"] == Anchor_feature_name][["patient_id", "value"]].copy()
                reference_feature.rename(columns={"value": "reference_dates_list"}, inplace=True)

                if anchor_reference_cache is not None:
                    anchor_reference_cache[Anchor_feature_name] = reference_feature.copy()

            if len(reference_feature) != len(X_original["patient_id"].unique()):
                print(f'len of data with this anchor {len(reference_feature["patient_id"].unique())}')
                print(f'unique patients: {len(X_original["patient_id"].unique())}')
                raise ValueError(f"Each patient must have one {Anchor_feature_name} entry in 'feature_name'.")

            merge_cache_key = (feature, Anchor_feature_name)
            if feature_anchor_merge_cache is not None and merge_cache_key in feature_anchor_merge_cache:
                X_feature = feature_anchor_merge_cache[merge_cache_key].copy()
            else:
                X_feature = X_feature.merge(reference_feature, on="patient_id", how="left")

                if feature_anchor_merge_cache is not None:
                    feature_anchor_merge_cache[merge_cache_key] = X_feature.copy()

            if time_window_mask_cache is not None and mask_cache_key in time_window_mask_cache:
                mask = time_window_mask_cache[mask_cache_key].copy()
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
                if time_window_mask_cache is not None:
                    time_window_mask_cache[mask_cache_key] = mask.copy()

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
                if time_window_mask_cache is not None:
                    time_window_mask_cache[mask_cache_key] = mask.copy()

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
        filtered_cache_key = (feature, Anchor_feature_name, offset_from_anchor)
        if filtered_numeric_cache is not None and filtered_cache_key in filtered_numeric_cache:
            filtered_X = filtered_numeric_cache[filtered_cache_key]
        else:
            filtered_X = X_feature[mask].copy()
            
            # Ensure 'value' is numeric
            filtered_X["value"] = pd.to_numeric(filtered_X["value"], errors="coerce")

            if filtered_numeric_cache is not None:
                filtered_numeric_cache[filtered_cache_key] = filtered_X

        # Aggregate values
        cached_aggregation_names = {"mean", "sum", "max", "count"}
        if aggregation in cached_aggregation_names:
            if aggregation_result_cache is not None and filtered_cache_key in aggregation_result_cache:
                agg_results = aggregation_result_cache[filtered_cache_key]
            else:
                agg_results = filtered_X.groupby("patient_id")["value"].agg(
                    mean="mean",
                    sum=lambda values: values.sum(min_count=1),
                    max="max",
                    count="count",
                )

                if aggregation_result_cache is not None:
                    aggregation_result_cache[filtered_cache_key] = agg_results

            agg_result = agg_results[aggregation]
        else:
            agg_result = filtered_X.groupby("patient_id")["value"].agg(aggregation)
        
        print(f"[AGG] feature={feature}, anchor={Anchor_feature_name}, offset={offset_from_anchor}")
        # print("Non-null aggregated values:",agg_result)

        return agg_result.reindex(X_original["patient_id"].unique()).values, Anchor_feature_name

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

    def _validate_input(self, X: pd.DataFrame, y: pd.Series) -> None:
        """ Validates the input data."""
        print("Validating input...")
        if 'time' not in X.columns or 'patient_id' not in X.columns or 'feature_name' not in X.columns or 'value' not in X.columns :
            raise ValueError("Input data must contain 'time', 'patient_id', 'feature_name', 'value'  columns.")
        if len(X['patient_id'].unique()) != len(y):
            raise ValueError("The length of y must be the same as the number of patients.")
        required_columns = ["patient_id", "time", "feature_name"]
        if X[required_columns].isnull().values.any() or y.isnull().values.any():
            raise ValueError(
                "Input data contains missing values in required columns or labels. "
                "Missing values are allowed only in the 'value' column."
            )
        
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
    
  

# ============================================================= end of my methods 
def print_debug_error(mask_inds_l, mask_inds_r, y, args):
    print('There are {} left and {} right inds'.format(len(mask_inds_l), len(mask_inds_r)))

    if not len(mask_inds_l):
        print('Invalid data {} split'.format('left'))
        print('There are {} unique y(right inds) values'.format(len(set(y.take(mask_inds_r)))))
    else:
        print('Invalid data {} split'.format('right'))
        print('There are {} unique y(left inds) values'.format(len(set(y.take(mask_inds_l)))))

    if 'is_leaf' in args:
        print('args["is_leaf"] = {}'.format(args['is_leaf']))
