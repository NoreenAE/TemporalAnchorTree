# Installation

## 1. Install prerequisites and clone

Install Git and **Python 3.9**. The pinned scientific dependencies require Python
3.9 or newer; the recorded research environment is **Python 3.9.5**.
Use 3.9.5 for the strict Experiment 2.1 checkpoint checks, which require the
recorded Python and scientific-package versions. Other Python versions have not
been validated for thesis reproduction.

The clone command uses the currently configured Git origin and names the
checkout `TemporalAnchorTree`:

```bash
git clone https://github.com/NoreenAE/TemporalDecisionTree.git TemporalAnchorTree
cd TemporalAnchorTree
```

**Run all remaining commands from this repository root.** Python imports the local
`TAT` package directly; installing dependencies does not install TAT from PyPI.

## 2. Create and activate a virtual environment

Linux/macOS, in a bash-compatible shell:

```bash
python3.9 --version
python3.9 -m venv .venv
source .venv/bin/activate
```

Windows, in PowerShell with the Python launcher installed:

```powershell
py -3.9 --version
py -3.9 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell does not permit activation, use `.\.venv\Scripts\python.exe` in
place of `python` in subsequent commands. Experiment 2.1 imports `fcntl` for
process locking and requires a POSIX environment: on Windows, run that experiment
inside WSL using the Linux instructions. Its strict version checks also apply
on Linux and macOS.

## 3. Install dependencies

With the environment active:

```bash
python -m pip install -r requirements.txt
python -m pip check
```

The requirements pin NumPy, pandas, SciPy, scikit-learn, matplotlib, dill, IPython,
and the notebook kernel. They provide neither a Jupyter server nor pytest. To
use the EDA notebooks, open them in a Jupyter-capable frontend and select this
virtual environment's Python kernel.

## 4. Verify the installation

```bash
python -B -c "import sys, numpy, pandas, sklearn; from TAT import TemporalTree; print(sys.version.split()[0]); print('TAT import OK'); print(numpy.__version__, pandas.__version__, sklearn.__version__)"
```

With the recorded pins, the scientific versions printed are `1.26.4`, `1.5.3`,
and `1.6.1`.
Saved-model and environment requirements are described in
[compatibility.md](compatibility.md).
The method and runnable API example are in [method.md](method.md).

## 5. Lightweight tests

The public suite checks the TAT algorithm, its runtime dependencies, and required
saved-model import paths. It uses small synthetic fixtures and temporary files;
it does not regenerate the thesis datasets, read stored thesis artifacts, or train
thesis experiments. Tiny trees and ensembles are fitted only on test fixtures.

Install pytest separately from the runtime requirements, then run from the
repository root:

```bash
python -m pip install pytest==8.3.5
python -B -m pytest -q
```

The pytest configuration limits default collection to `tests/`.
