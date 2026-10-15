"""Runtime imports only; no dataset generation or saved-model execution."""

from importlib import import_module
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "module_name",
    ("numpy", "pandas", "scipy", "sklearn", "matplotlib", "dill",
     "IPython", "ipykernel"),
)
def test_required_dependencies_import(module_name: str) -> None:
    """Each direct runtime/notebook dependency can be imported."""
    import_module(module_name)


def test_saved_model_class_imports_work_in_a_fresh_process() -> None:
    """Resolve class paths used by included models without loading a model."""
    script = """
import pickle
import sys

assert 'TAT' not in sys.modules
for name in ('TemporalTree', 'TemporalSplitNode', 'TemporalLeaf'):
    payload = ('csettree.temporal_tree\\n' + name + '\\n.').encode()
    restored = pickle.loads(payload)
    from TAT import temporal_tree
    assert restored is getattr(temporal_tree, name)
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
