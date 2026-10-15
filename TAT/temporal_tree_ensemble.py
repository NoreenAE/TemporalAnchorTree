"""Sequential ensembles of stochastically sampled TemporalTree estimators."""

from numbers import Integral
from time import perf_counter
from typing import Optional, Sequence, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_is_fitted

from .temporal_tree import TemporalTree


MAX_SPLIT_SAMPLING_SEED = 2**32 - 1


class TemporalTreeEnsemble(BaseEstimator):
    """Average predictions from independently sampled regression trees.

    Every member is fitted on the same data with the same ``TemporalTree``
    parameters. Only ``split_sampling_seed`` changes between members. The
    current research use case fits regression trees to binary 0/1 targets and
    evaluates the resulting continuous mean score with ROC-AUC. The returned
    scores are not calibrated clinical probabilities.

    Parameters
    ----------
    estimator : TemporalTree or None, default=None
        Unfitted regression-style tree template. ``None`` resolves to
        ``TemporalTree(is_classification=False, criterion="mse")``.
    n_estimators : int, default=5
        Number of independent trees to fit.
    base_split_sampling_seed : int, default=42
        First seed in the deterministic consecutive sequence used when
        ``split_sampling_seeds`` is not supplied.
    split_sampling_seeds : sequence of int or None, default=None
        Explicit ordered member seeds. Its length must equal ``n_estimators``
        and all values must be unique. When supplied, this sequence is
        authoritative and ``base_split_sampling_seed`` is ignored.
    """

    def __init__(
        self,
        estimator: Optional[TemporalTree] = None,
        n_estimators: int = 5,
        base_split_sampling_seed: int = 42,
        split_sampling_seeds: Optional[Sequence[int]] = None,
    ):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.base_split_sampling_seed = base_split_sampling_seed
        self.split_sampling_seeds = split_sampling_seeds

    @staticmethod
    def _validate_seed(seed, parameter_name: str) -> int:
        """Return a validated portable NumPy seed."""
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise ValueError(
                f"{parameter_name} must contain integers in the range "
                f"[0, {MAX_SPLIT_SAMPLING_SEED}], excluding bool."
            )
        seed = int(seed)
        if not 0 <= seed <= MAX_SPLIT_SAMPLING_SEED:
            raise ValueError(
                f"{parameter_name} must contain integers in the range "
                f"[0, {MAX_SPLIT_SAMPLING_SEED}]."
            )
        return seed

    def _resolve_estimator(self) -> TemporalTree:
        """Return and validate the unfitted tree template."""
        estimator = self.estimator
        if estimator is None:
            estimator = TemporalTree(
                is_classification=False,
                criterion="mse",
            )
        if not isinstance(estimator, TemporalTree):
            raise TypeError("estimator must be a TemporalTree instance or None.")
        if estimator.is_classification:
            raise ValueError(
                "TemporalTreeEnsemble version one supports only "
                "TemporalTree(is_classification=False)."
            )
        return estimator

    def _resolve_split_sampling_seeds(self) -> Tuple[int, ...]:
        """Validate configuration and return ordered member seeds."""
        if isinstance(self.n_estimators, bool) or not isinstance(
            self.n_estimators,
            Integral,
        ):
            raise ValueError("n_estimators must be an integer greater than 0.")
        n_estimators = int(self.n_estimators)
        if n_estimators < 1:
            raise ValueError("n_estimators must be an integer greater than 0.")

        if self.split_sampling_seeds is not None:
            if isinstance(self.split_sampling_seeds, (str, bytes)):
                raise ValueError("split_sampling_seeds must be a sequence of integers.")
            try:
                supplied_seeds = tuple(self.split_sampling_seeds)
            except TypeError as exc:
                raise ValueError(
                    "split_sampling_seeds must be a sequence of integers."
                ) from exc
            if len(supplied_seeds) != n_estimators:
                raise ValueError(
                    "len(split_sampling_seeds) must equal n_estimators."
                )
            resolved_seeds = tuple(
                self._validate_seed(seed, "split_sampling_seeds")
                for seed in supplied_seeds
            )
            if len(set(resolved_seeds)) != len(resolved_seeds):
                raise ValueError("split_sampling_seeds must not contain duplicates.")
            return resolved_seeds

        base_seed = self._validate_seed(
            self.base_split_sampling_seed,
            "base_split_sampling_seed",
        )
        final_seed = base_seed + n_estimators - 1
        if final_seed > MAX_SPLIT_SAMPLING_SEED:
            raise ValueError(
                "The generated split-sampling seed sequence exceeds "
                f"{MAX_SPLIT_SAMPLING_SEED}."
            )
        return tuple(range(base_seed, final_seed + 1))

    def _clear_fitted_state(self) -> None:
        """Remove fitted attributes before starting a new atomic fit."""
        for attribute in (
            "estimators_",
            "split_sampling_seeds_",
            "fit_times_",
            "fit_time_",
            "n_estimators_",
        ):
            if hasattr(self, attribute):
                delattr(self, attribute)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit every member sequentially on exactly the same ``X`` and ``y``."""
        self._clear_fitted_state()
        estimator_template = self._resolve_estimator()
        resolved_seeds = self._resolve_split_sampling_seeds()
        n_estimators = len(resolved_seeds)
        verbose = getattr(estimator_template, "verbose", 0)

        if (
            n_estimators > 1
            and estimator_template.candidate_sampling_rate == 1.0
        ):
            warnings.warn(
                "candidate_sampling_rate=1.0 enables exhaustive candidate "
                "evaluation; changing only split_sampling_seed is not expected "
                "to create ensemble diversity.",
                UserWarning,
                stacklevel=2,
            )

        if verbose >= 1:
            print(
                "=== Temporal Tree Ensemble ===\n"
                f"Number of estimators: {n_estimators}\n"
                "Candidate sampling rate per tree: "
                f"{estimator_template.candidate_sampling_rate:.3f}\n"
                f"Split sampling seeds: {list(resolved_seeds)}\n"
                "=============================="
            )

        fitted_estimators = []
        fit_times = []
        ensemble_start = perf_counter()
        for index, seed in enumerate(resolved_seeds):
            if verbose >= 1:
                print(
                    f"[ENSEMBLE] Tree {index + 1}/{n_estimators} | "
                    f"split_sampling_seed={seed}"
                )

            member = clone(estimator_template)
            member.set_params(split_sampling_seed=seed)
            member_start = perf_counter()
            try:
                member.fit(X, y)
            except Exception as exc:
                self._clear_fitted_state()
                raise RuntimeError(
                    "TemporalTreeEnsemble member fit failed: "
                    f"estimator_index={index}, split_sampling_seed={seed}."
                ) from exc
            fit_times.append(perf_counter() - member_start)
            fitted_estimators.append(member)

        fit_time = perf_counter() - ensemble_start
        self.estimators_ = fitted_estimators
        self.split_sampling_seeds_ = resolved_seeds
        self.fit_times_ = tuple(fit_times)
        self.fit_time_ = fit_time
        self.n_estimators_ = n_estimators

        if verbose >= 1:
            for index, (seed, member_time) in enumerate(
                zip(self.split_sampling_seeds_, self.fit_times_),
            ):
                print(
                    f"[ENSEMBLE] Tree {index + 1}/{self.n_estimators_} | "
                    f"split_sampling_seed={seed} | fit_time={member_time:.2f}s"
                )
            print(f"[ENSEMBLE] Total fit time: {self.fit_time_:.2f}s")

        return self

    def predict_individual(self, X: pd.DataFrame) -> np.ndarray:
        """Return member predictions in first-occurrence patient order.

        Returns
        -------
        numpy.ndarray
            Array with shape ``(n_estimators, n_patients)``. Rows correspond
            exactly to ``estimators_``, ``split_sampling_seeds_``, and
            ``fit_times_``.
        """
        check_is_fitted(self, attributes=["estimators_", "n_estimators_"])
        required_columns = {"patient_id", "time", "feature_name", "value"}
        missing_columns = required_columns - set(X.columns)
        if missing_columns:
            raise ValueError(
                "Prediction data is missing required columns: "
                f"{sorted(missing_columns)}."
            )

        n_patients = X["patient_id"].drop_duplicates().size
        if n_patients == 0:
            return np.empty((self.n_estimators_, 0), dtype=float)

        predictions = []
        for index, estimator in enumerate(self.estimators_):
            member_predictions = np.asarray(estimator.predict(X))
            if member_predictions.ndim != 1:
                raise ValueError(
                    "TemporalTreeEnsemble expected one-dimensional member "
                    f"predictions, but estimator {index} returned shape "
                    f"{member_predictions.shape}."
                )
            if len(member_predictions) != n_patients:
                raise ValueError(
                    f"TemporalTreeEnsemble estimator {index} returned "
                    f"{len(member_predictions)} predictions for {n_patients} "
                    "patients."
                )
            predictions.append(member_predictions)

        return np.vstack(predictions)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return the arithmetic mean of all member regression predictions."""
        individual_predictions = self.predict_individual(X)
        if individual_predictions.shape[1] == 0:
            return np.empty(0, dtype=float)
        if self.n_estimators_ == 1:
            return individual_predictions[0].copy()
        return individual_predictions.mean(axis=0)
