"""Resolve historical model module paths to the Temporal Anchor Tree package.

Existing pickles import ``settree.temporal_tree`` automatically. Alias the old
modules to the actual TAT modules so restored and newly created objects share
class identity. New serialization records the classes' TAT module paths.
"""

from importlib import import_module
import sys


_package = import_module("TAT")
for _module_name in (
    "temporal_tree",
    "temporal_tree_ensemble",
    "anchor_utils",
    "eval_utils",
    "temporal_explainability",
):
    sys.modules[f"{__name__}.{_module_name}"] = import_module(
        f"TAT.{_module_name}"
    )
sys.modules[__name__] = _package
