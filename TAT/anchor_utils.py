from numbers import Real

import numpy as np
import pandas as pd


def _offset_to_dateoffset(offset: Real) -> pd.DateOffset:
    """Use calendar months for integers and signed 30-day fractions otherwise."""
    if (
        isinstance(offset, bool)
        or not isinstance(offset, Real)
        or not np.isfinite(offset)
    ):
        raise ValueError("Anchor offsets must be finite real numbers.")

    numeric_offset = float(offset)
    if numeric_offset.is_integer():
        return pd.DateOffset(months=int(numeric_offset))
    return pd.DateOffset(days=round(numeric_offset * 30))


def _aggregation_anchor_dates(
    patient_data: pd.DataFrame,
    aggregation: str,
    anchor_selection: str = None,
) -> list:
    """Return the earliest eligible timestamp as a singleton list, or []."""
    numeric_values = pd.to_numeric(patient_data["value"], errors="coerce")
    valid_data = patient_data.loc[numeric_values.notna()].copy()
    if valid_data.empty:
        return []

    valid_data["_numeric_value"] = numeric_values.loc[valid_data.index]
    if anchor_selection is None:
        anchor_selection = (
            "maximum" if aggregation == "max" else "all_valid"
        )

    if anchor_selection == "maximum":
        maximum = valid_data["_numeric_value"].max()
        valid_data = valid_data[valid_data["_numeric_value"] == maximum]
    elif anchor_selection == "positive_events":
        valid_data = valid_data[valid_data["_numeric_value"] == 1]
    elif anchor_selection == "nonzero_contributions":
        valid_data = valid_data[valid_data["_numeric_value"] != 0]
    elif anchor_selection != "all_valid":
        raise ValueError(
            f"Unsupported dynamic-anchor selection rule: {anchor_selection}."
        )

    ordered_data = valid_data.sort_values(
        "time",
        kind="mergesort",
        na_position="last",
    )
    return ordered_data["time"].dropna().head(1).tolist()


def create_branch_anchors(patient_ids, X_branch, anchor_name, direction,
                          feature, threshold, time_window, aggregation="mean",
                          anchor_selection=None):
    """
    Generates anchor rows for patients after a temporal split.

    Parameters:
        patient_ids (list-like): Patient IDs in this branch.
        X_branch (pd.DataFrame): Long-format EHR data for this branch.
        anchor_name (str): Name to assign to the new anchor (e.g., "IgE <= 200 Anchor").
        direction (str): "left" or "right" branch already selected by routing.
        feature (str): Feature used in the split.
        threshold (float): Patient-level aggregate threshold used in the split.
        time_window (tuple): (anchor_feature_name, offset_in_months).
        aggregation (str): Patient-level aggregation used by the split.
        anchor_selection (str or None): Semantic timestamp-selection rule.

        Either branch may contain a timestamp when an eligible observation
        exists in the evaluated window. The threshold is not reapplied to raw values.
        An unavailable timestamp is represented by [], never a valid window.

    Returns:
        pd.DataFrame: New anchor rows, one per patient.
    """
    anchor_data = []
    grouped = X_branch.groupby("patient_id")
    prev_anchor_name, offset = time_window

    if isinstance(prev_anchor_name, str) and offset is not None:
        reference_feature = X_branch[X_branch['feature_name'] == prev_anchor_name][['patient_id', 'value']]
        anchor_counts = reference_feature.groupby('patient_id').size()
        expected_patients = pd.Index(patient_ids)
        counts_by_patient = anchor_counts.reindex(expected_patients, fill_value=0)
        missing_patients = counts_by_patient[counts_by_patient == 0]
        duplicate_patients = counts_by_patient[counts_by_patient > 1]
        if len(missing_patients) > 0 or len(duplicate_patients) > 0:
            raise ValueError(
                f"Each patient must have exactly one {prev_anchor_name} entry in 'feature_name'. "
                f"Missing anchor for {len(missing_patients)} patients "
                f"(examples: {missing_patients.index[:5].tolist()}); "
                f"duplicate anchor rows for {len(duplicate_patients)} patients "
                f"(examples: {duplicate_patients.index[:5].tolist()})."
            )

    for patient_id in patient_ids:
        try:
            full_patient_data = grouped.get_group(patient_id)
        except KeyError:
            anchor_data.append({
                'patient_id': patient_id,
                'time': np.nan,
                'feature_name': anchor_name,
                'value': []
            })
            continue

        patient_data = full_patient_data[full_patient_data['feature_name'] == feature]
        if patient_data.empty:
            anchor_data.append({
                'patient_id': patient_id,
                'time': np.nan,
                'feature_name': anchor_name,
                'value': []
            })
            continue

        # Retrieve anchor reference value (from previous anchor)
        if prev_anchor_name is None or offset is None:
            time_mask = pd.Series([True] * len(patient_data), index=patient_data.index)
        else:
            anchor_value = (
                pd.Series([prev_anchor_name])
                if isinstance(prev_anchor_name, pd.Timestamp)
                else full_patient_data.loc[
                    full_patient_data["feature_name"] == prev_anchor_name, "value"
                ]
            )
            if anchor_value.empty:
                time_mask = pd.Series([False] * len(patient_data), index=patient_data.index)
            else:
                anchor_raw = anchor_value.iloc[0]
                if isinstance(anchor_raw, (list, np.ndarray)) and len(anchor_raw) == 0:
                    time_mask = pd.Series([False] * len(patient_data), index=patient_data.index)
                elif not isinstance(anchor_raw, (list, np.ndarray)) and pd.isnull(anchor_raw):
                    time_mask = pd.Series([False] * len(patient_data), index=patient_data.index)
                elif isinstance(anchor_raw, pd.Timestamp):
                    anchor_time = pd.to_datetime(anchor_raw)
                    offset_time = anchor_time + _offset_to_dateoffset(offset)
                    time_mask = (
                        (patient_data['time'] >= anchor_time) & (patient_data['time'] <= offset_time)
                        if offset >= 0 else
                        (patient_data['time'] <= anchor_time) & (patient_data['time'] >= offset_time)
                    )
                elif isinstance(anchor_raw, (list, np.ndarray)):
                    anchor_times = pd.to_datetime(anchor_raw)
                    date_offset = _offset_to_dateoffset(offset)
                    offset_times = [dt + date_offset for dt in anchor_times]

                    def time_filter(time_val):
                        return any(
                            (time_val <= o and r <= time_val) if offset >= 0 else (time_val >= o and r >= time_val)
                            for r, o in zip(anchor_times, offset_times)
                        )

                    time_mask = patient_data['time'].apply(time_filter)
                else:
                    time_mask = pd.Series([False] * len(patient_data), index=patient_data.index)

        anchor_dates = _aggregation_anchor_dates(
            patient_data.loc[time_mask],
            aggregation,
            anchor_selection,
        )

        anchor_data.append({
            'patient_id': patient_id,
            'time': np.nan,
            'feature_name': anchor_name,
            'value': anchor_dates
        })

    return pd.DataFrame(anchor_data)



def get_time_window_mask(X: pd.DataFrame, feature: str, anchor_name, offset): # there is no use bentayim
        """
        Returns a boolean mask for data['time'] being within the time window defined
        by anchor_feature_name ± offset_from_anchor (in months).
        
        Parameters:
            X (pd.DataFrame): The full dataset.
            feature (str): The feature to filter on.
            anchor_name (str, pd.Timestamp, or None): Anchor feature name or direct timestamp or None for no restriction.
            offset (real number or None): Offset in months from anchor point.

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
                X_feature["date_relative_offset"] = (
                    X_feature["reference_dates_list"]
                    + _offset_to_dateoffset(offset)
                )
                if offset >= 0:
                    mask = (X_feature["time"] <= X_feature["date_relative_offset"]) & (X_feature["time"] >= X_feature["reference_dates_list"])
                else:
                    mask = (X_feature["time"] >= X_feature["date_relative_offset"]) & (X_feature["time"] <= X_feature["reference_dates_list"])

            elif X_feature["reference_dates_list"].map(lambda x: isinstance(x, list)).all():
                X_feature["reference_dates_list"] = X_feature["reference_dates_list"].map(pd.to_datetime)
                date_offset = _offset_to_dateoffset(offset)
                X_feature["date_relative_offset"] = X_feature["reference_dates_list"].map(
                    lambda lst: [dt + date_offset for dt in lst]
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
            offset_time = anchor_name + _offset_to_dateoffset(offset)
            if offset >= 0:
                mask = (X_feature["time"] >= anchor_name) & (X_feature["time"] <= offset_time)
            else:
                mask = (X_feature["time"] <= anchor_name) & (X_feature["time"] >= offset_time)
            return pd.Series(mask, index=X_feature.index)

        elif anchor_name is None:
            return pd.Series([True] * len(X_feature), index=X_feature.index)

        else:
            raise TypeError(f"Unsupported anchor or offset type: {type(anchor_name)}, {type(offset)}")
    
