import numbers

import pandas as pd

import main


def _load_pickled_data(tmp_path, rows):
    data_path = tmp_path / "data.pkl"
    labels_path = tmp_path / "labels.pkl"
    pd.DataFrame(rows).to_pickle(data_path)
    pd.DataFrame({"patient_id": ["P1"], "label": [0]}).to_pickle(labels_path)
    return main.load_data(data_path, labels_path)


def test_load_data_preserves_all_anchor_values_and_numeric_features(tmp_path):
    dynamic_anchor = "FormulaYN > 0.500 | Birth Date Anchor +3m Anchor"
    data, _ = _load_pickled_data(
        tmp_path,
        [
            {
                "patient_id": "P1",
                "time": "2020-01-01",
                "feature_name": "Birth Date Anchor",
                "value": "2020-01-01",
            },
            {
                "patient_id": "P1",
                "time": "2022-03-15",
                "feature_name": "NOW Anchor",
                "value": "2022-03-15",
            },
            {
                "patient_id": "P1",
                "time": "2020-04-01",
                "feature_name": dynamic_anchor,
                "value": "2020-04-01",
            },
            {
                "patient_id": "P1",
                "time": "2020-02-01",
                "feature_name": "FormulaYN",
                "value": "1.25",
            },
        ],
    )

    values = data.set_index("feature_name")["value"]
    assert values["Birth Date Anchor"] == pd.Timestamp("2020-01-01")
    assert values["NOW Anchor"] == pd.Timestamp("2022-03-15")
    assert values[dynamic_anchor] == pd.Timestamp("2020-04-01")
    assert not pd.isna(values["NOW Anchor"])
    assert isinstance(values["FormulaYN"], numbers.Number)
    assert values["FormulaYN"] == 1.25


def test_load_data_keeps_legacy_birth_anchor_behavior(tmp_path):
    data, labels = _load_pickled_data(
        tmp_path,
        [
            {
                "patient_id": "P1",
                "time": "2020-01-01",
                "feature_name": "Birth Date Anchor",
                "value": pd.Timestamp("2020-01-01"),
            },
            {
                "patient_id": "P1",
                "time": "2020-02-01",
                "feature_name": "FormulaYN",
                "value": "0",
            },
        ],
    )

    birth_value = data.loc[
        data["feature_name"].eq("Birth Date Anchor"), "value"
    ].iloc[0]
    formula_value = data.loc[data["feature_name"].eq("FormulaYN"), "value"].iloc[0]

    assert birth_value == pd.Timestamp("2020-01-01")
    assert isinstance(formula_value, numbers.Number)
    assert formula_value == 0
    assert labels.to_dict("records") == [{"patient_id": "P1", "label": 0}]
