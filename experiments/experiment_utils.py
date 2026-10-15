"""Shared serialization and anchor-audit helpers, without experiment execution."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from project_paths import portable_metadata, resolve_path


DATASET_FILENAMES = {
    "observations.csv", "patients_and_split.csv", "validation.json",
    "descriptive_audit.json", "manifest.json",
}


def dataset_directory(directory: Path) -> Path:
    """Locate one shared dataset from a data directory or a saved run config."""
    directory = Path(directory)
    if (directory / "observations.csv").is_file():
        return directory
    config = json.loads((directory / "configuration.json").read_text())
    return resolve_path(config["dataset_directory"])


def artifact_path(directory: Path, name: str) -> Path:
    """Resolve dataset artifacts through a run's shared-data reference."""
    path = Path(directory) / name
    if name in DATASET_FILENAMES and not path.is_file():
        return dataset_directory(Path(directory)) / name
    return path

def require(condition: bool, message: str) -> None:
    """Validate an invariant even when Python assertions are disabled."""
    if not condition:
        raise ValueError(message)


def write_json(path: Path, value: Any) -> None:
    """Write JSON with native NumPy and timestamp conversion."""
    def convert(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, pd.Timestamp):
            return item.isoformat()
        raise TypeError(f"Cannot serialize {type(item)}")

    path.write_text(
        json.dumps(portable_metadata(value), indent=2, default=convert,
                   allow_nan=False) + "\n"
    )


def anchor_values(data: pd.DataFrame) -> dict:
    """Extract path anchor rows without changing the event data."""
    anchors = data.loc[data.feature_name.str.endswith("Anchor")]
    require(
        not anchors.feature_name.duplicated().any(), "Duplicate patient anchor rows"
    )
    return dict(zip(anchors.feature_name, anchors.value))


def tree_records(root: Any) -> tuple[list[dict], dict]:
    """Export stable preorder IDs, including leaf means and missing routing."""
    records, ids = [], {}

    def visit(node: Any) -> int:
        index = len(records)
        ids[id(node)] = index
        record = dict(node_id=index, depth=node.depth, patients=node.num_patients)
        records.append(record)
        if node.is_leaf():
            record.update(type="leaf", score=float(node.value))
        else:
            record.update(
                type="split",
                feature=node.feature,
                aggregation=node.aggregation,
                window=node.time_window,
                threshold=float(node.threshold),
                missing_direction=node.missing_direction,
                available_anchors=node.anchors_list,
            )
            record["left"] = visit(node.left)
            record["right"] = visit(node.right)
        return index

    visit(root)
    return records, ids
