# MIT License
#
# Adapted SimBA attack for tabular data: per-feature perturbations, projection to valid domain.
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from tqdm.auto import trange

from art.attacks.attack import EvasionAttack
from art.config import ART_NUMPY_DTYPE
from art.estimators.classification.classifier import ClassifierMixin
from art.estimators.estimator import BaseEstimator, NeuralNetworkMixin
from art.utils import is_probability

from src.attacks.projection import project_tabular_constraints

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

logger = logging.getLogger(__name__)


class SimBATabular(EvasionAttack):
    """
    Simple black-box attack for tabular data: perturbs one feature at a time (Linf/L2),
    accepts change if it improves the objective, and projects to valid tabular domain.
    """

    attack_params = EvasionAttack.attack_params + [
        "max_iter",
        "epsilon",
        "order",
        "targeted",
        "batch_size",
        "verbose",
        "feature_clip_min",
        "feature_clip_max",
        "integer_indices",
        "categorical_groups",
        "editable_mask",
        "norm",
        "only_increase_mask",
        "only_decrease_mask",
        "max_abs_step",
    ]

    _estimator_requirements = (BaseEstimator, ClassifierMixin, NeuralNetworkMixin)

    def __init__(
        self,
        classifier: "CLASSIFIER_TYPE",
        max_iter: int = 1000,
        epsilon: float = 0.1,
        order: str = "random",
        targeted: bool = False,
        batch_size: int = 1,
        verbose: bool = True,
        feature_clip_min: np.ndarray | None = None,
        feature_clip_max: np.ndarray | None = None,
        integer_indices: list[int] | None = None,
        categorical_groups: list[list[int]] | None = None,
        editable_mask: np.ndarray | None = None,
        norm: str = "inf",
        only_increase_mask: np.ndarray | None = None,
        only_decrease_mask: np.ndarray | None = None,
        max_abs_step: np.ndarray | None = None,
    ):
        super().__init__(estimator=classifier)
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.order = order
        self._targeted = targeted
        self.batch_size = batch_size
        self.verbose = verbose
        self.feature_clip_min = feature_clip_min
        self.feature_clip_max = feature_clip_max
        self.integer_indices = integer_indices or []
        self.categorical_groups = categorical_groups or []
        self.editable_mask = editable_mask
        self.norm = norm
        self.only_increase_mask = only_increase_mask
        self.only_decrease_mask = only_decrease_mask
        self.max_abs_step = max_abs_step
        self._check_params()

    def generate(self, x: np.ndarray, y: np.ndarray | None = None, **kwargs) -> np.ndarray:
        if x.ndim != 2:
            raise ValueError("SimBATabular expects inputs of shape (n_samples, n_features).")
        x = x.astype(ART_NUMPY_DTYPE)
        x_adv = x.copy()
        self._x_reference = x.copy()

        y_pred_raw = self.estimator.predict(x, batch_size=self.batch_size)
        y_prob_pred = self._to_probability(y_pred_raw)
        if self.estimator.nb_classes == 2 and y_prob_pred.shape[1] == 1:
            raise ValueError("Binary classifier with single output is not supported.")

        if y is None:
            if self.targeted:
                raise ValueError("Target labels `y` need to be provided for a targeted attack.")
            y_i = np.argmax(y_prob_pred, axis=1)
        else:
            if y.ndim == 1:
                y_i = y
            else:
                y_i = np.argmax(y, axis=1)

        clip_min, clip_max = self._init_bounds(x_adv)
        n_features = x.shape[1]
        # Map from categorical index to its group for quick lookup
        cat_index_to_group = {}
        for group in self.categorical_groups:
            for idx in group:
                cat_index_to_group[idx] = group

        for i_sample in trange(x.shape[0], desc="SimBA (tabular) - sample", disable=not self.verbose):
            desired_label = int(y_i[i_sample])
            current_probs = y_prob_pred[i_sample]
            current_label = int(np.argmax(current_probs))
            # If desired_label not in range (shouldn't happen), fall back to current_label
            if desired_label >= len(current_probs):
                desired_label = current_label
            last_prob = float(current_probs.reshape(-1)[desired_label])

            if self.order == "random":
                indices = np.random.permutation(n_features)
            elif self.order == "diag":
                indices = np.arange(n_features)
            else:
                raise ValueError("order must be `random` or `diag`.")

            nb_iter = 0
            term_flag = self._is_success(current_label, desired_label)
            while not term_flag and nb_iter < self.max_iter:
                idx = indices[nb_iter % n_features]
                # Categorical swap if idx belongs to a one-hot group
                if idx in cat_index_to_group:
                    group = cat_index_to_group[idx]
                    current_group_vals = x_adv[i_sample, group]
                    current_active = int(np.argmax(current_group_vals))
                    # pick an alternative category different from current
                    alternatives = [g for g in group if g != group[current_active]]
                    alt_idx = np.random.choice(alternatives)
                    x_candidate = x_adv[i_sample].copy()
                    x_candidate[group] = 0
                    x_candidate[alt_idx] = 1
                    x_candidate = self._project_tabular(x_candidate[None, :], clip_min, clip_max)[0]
                    probs = self._to_probability(self.estimator.predict(x_candidate[None, :], batch_size=self.batch_size))
                    prob_target = probs.reshape(-1)[desired_label]
                    if (self.targeted and prob_target > last_prob) or (not self.targeted and prob_target < last_prob):
                        x_adv[i_sample] = x_candidate
                        last_prob = prob_target
                        current_label = int(np.argmax(probs, axis=1)[0])
                else:
                    diff = np.zeros(n_features, dtype=ART_NUMPY_DTYPE)
                    if self.norm in [np.inf, "inf"]:
                        step = self.epsilon
                    elif self.norm == "2":
                        step = self.epsilon
                    else:
                        raise ValueError("Unsupported norm; use 'inf' or '2'.")
                    if idx in self.integer_indices:
                        step = max(1.0, np.ceil(self.epsilon))
                    diff[idx] = step

                    left = self._project_tabular((x_adv[[i_sample]] - diff), clip_min, clip_max)
                    right = self._project_tabular((x_adv[[i_sample]] + diff), clip_min, clip_max)

                    left_probs = self._to_probability(self.estimator.predict(left, batch_size=self.batch_size))
                    right_probs = self._to_probability(self.estimator.predict(right, batch_size=self.batch_size))
                    left_prob = left_probs.reshape(-1)[desired_label]
                    right_prob = right_probs.reshape(-1)[desired_label]

                    if self.targeted:
                        if left_prob > last_prob or right_prob > last_prob:
                            if left_prob >= right_prob:
                                x_adv[i_sample] = left[0]
                                last_prob = left_prob
                                current_label = int(np.argmax(left_probs, axis=1)[0])
                            else:
                                x_adv[i_sample] = right[0]
                                last_prob = right_prob
                                current_label = int(np.argmax(right_probs, axis=1)[0])
                    else:
                        if left_prob < last_prob or right_prob < last_prob:
                            if left_prob <= right_prob:
                                x_adv[i_sample] = left[0]
                                last_prob = left_prob
                                current_label = int(np.argmax(left_probs, axis=1)[0])
                            else:
                                x_adv[i_sample] = right[0]
                                last_prob = right_prob
                                current_label = int(np.argmax(right_probs, axis=1)[0])

                # keep within epsilon ball for L2
                if self.norm == "2":
                    delta = x_adv[i_sample] - x[i_sample]
                    norm_delta = np.linalg.norm(delta)
                    if norm_delta > self.epsilon:
                        x_adv[i_sample] = x[i_sample] + delta * (self.epsilon / norm_delta)
                        x_adv[i_sample] = self._project_tabular(x_adv[[i_sample]], clip_min, clip_max)[0]

                term_flag = self._is_success(current_label, desired_label)
                nb_iter += 1

            x_adv[i_sample] = self._project_tabular(x_adv[[i_sample]], clip_min, clip_max)[0]

        return x_adv

    def _is_success(self, current_label: int, desired_label: int) -> bool:
        return desired_label == current_label if self.targeted else desired_label != current_label

    def _init_bounds(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.feature_clip_min is not None and self.feature_clip_max is not None:
            return self.feature_clip_min, self.feature_clip_max
        if self.estimator.clip_values is not None:
            cmin, cmax = self.estimator.clip_values
            return np.ones_like(x[0]) * cmin, np.ones_like(x[0]) * cmax
        return np.min(x, axis=0), np.max(x, axis=0)

    def _project_tabular(self, x: np.ndarray, clip_min: np.ndarray, clip_max: np.ndarray) -> np.ndarray:
        return project_tabular_constraints(
            x_adv=x,
            x_orig=self._x_reference[: x.shape[0]],
            feature_clip_min=clip_min,
            feature_clip_max=clip_max,
            integer_indices=self.integer_indices,
            categorical_groups=self.categorical_groups,
            editable_mask=self.editable_mask,
            only_increase_mask=self.only_increase_mask,
            only_decrease_mask=self.only_decrease_mask,
            max_abs_step=self.max_abs_step,
        )

    def _check_params(self) -> None:
        if not isinstance(self.max_iter, int) or self.max_iter <= 0:
            raise ValueError("max_iter must be a positive integer.")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be > 0.")
        if self.order not in ("random", "diag"):
            raise ValueError("order must be `random` or `diag`.")
        if self.batch_size != 1:
            raise ValueError("batch_size must be 1 in this implementation.")
        if not isinstance(self.targeted, bool):
            raise ValueError("targeted must be bool.")
        if self.norm not in ["inf", np.inf, "2"]:
            raise ValueError("norm must be 'inf' or '2'.")
        if not isinstance(self.verbose, bool):
            raise ValueError("verbose must be bool.")

    def _to_probability(self, preds: np.ndarray) -> np.ndarray:
        if is_probability(preds[0]):
            return preds
        # Apply softmax to logits
        exp = np.exp(preds - np.max(preds, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)
