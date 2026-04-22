# MIT License
#
# Adapted SquareAttack for tabular data: perturb subsets of features, enforce per-feature clipping,
# integer rounding, one-hot projection, and optional editability mask.
from __future__ import annotations

import bisect
import logging
import math
import random
from typing import TYPE_CHECKING, Callable

import numpy as np
from tqdm.auto import trange

from art.attacks.attack import EvasionAttack
from art.config import ART_NUMPY_DTYPE
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import check_and_transform_label_format, get_labels_np_array

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

logger = logging.getLogger(__name__)


class SquareAttackTabular(EvasionAttack):
    """
    SquareAttack adapted to tabular vectors.
    Perturbs random subsets of features (Linf/L2), projects to valid tabular domain.
    """

    attack_params = EvasionAttack.attack_params + [
        "norm",
        "adv_criterion",
        "loss",
        "max_iter",
        "eps",
        "p_init",
        "nb_restarts",
        "batch_size",
        "verbose",
        "feature_clip_min",
        "feature_clip_max",
        "integer_indices",
        "categorical_groups",
        "editable_mask",
        "attack_name",
    ]

    _estimator_requirements = (BaseEstimator,)

    def __init__(
        self,
        estimator: "CLASSIFIER_TYPE",
        norm: int | float | str = np.inf,
        adv_criterion: Callable[[np.ndarray, np.ndarray], bool] | None = None,
        loss: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        max_iter: int = 100,
        eps: float = 0.3,
        p_init: float = 0.8,
        nb_restarts: int = 1,
        batch_size: int = 128,
        verbose: bool = True,
        feature_clip_min: np.ndarray | None = None,
        feature_clip_max: np.ndarray | None = None,
        integer_indices: list[int] | None = None,
        categorical_groups: list[list[int]] | None = None,
        editable_mask: np.ndarray | None = None,
        attack_name: str | None = None,  # accepted but unused; for config symmetry
    ):
        super().__init__(estimator=estimator)

        self.norm = norm

        if adv_criterion is not None:
            self.adv_criterion = adv_criterion
        elif isinstance(self.estimator, ClassifierMixin):
            self.adv_criterion = lambda y_pred, y: np.argmax(y_pred, axis=1) != np.argmax(y, axis=1)
        else:  # pragma: no cover
            raise ValueError("No acceptable adversarial criterion available.")

        if loss is not None:
            self.loss = loss
        elif isinstance(self.estimator, ClassifierMixin):
            self.loss = self._get_logits_diff
        else:  # pragma: no cover
            raise ValueError("No acceptable loss available.")

        self.max_iter = max_iter
        self.eps = eps
        self.p_init = p_init
        self.nb_restarts = nb_restarts
        self.batch_size = batch_size
        self.verbose = verbose

        self.feature_clip_min = feature_clip_min
        self.feature_clip_max = feature_clip_max
        self.integer_indices = integer_indices or []
        self.categorical_groups = categorical_groups or []
        self.editable_mask = editable_mask

        self._check_params()

    def _get_logits_diff(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        y_pred = self.estimator.predict(x, batch_size=self.batch_size)
        logit_correct = np.take_along_axis(y_pred, np.expand_dims(np.argmax(y, axis=1), axis=1), axis=1)
        logit_highest_incorrect = np.take_along_axis(
            y_pred, np.expand_dims(np.argsort(y_pred, axis=1)[:, -2], axis=1), axis=1
        )
        return (logit_correct - logit_highest_incorrect)[:, 0]

    def _get_percentage_of_elements(self, i_iter: int) -> float:
        i_p = i_iter / self.max_iter
        intervals = [0.001, 0.005, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
        p_ratio = [1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64, 1 / 128, 1 / 256, 1 / 512]
        i_ratio = bisect.bisect_left(intervals, i_p)
        return self.p_init * p_ratio[i_ratio]

    def _project_tabular(self, x: np.ndarray) -> np.ndarray:
        """Enforce per-feature clipping, integer rounding, one-hot, and editable mask."""
        if self.feature_clip_min is not None and self.feature_clip_max is not None:
            x = np.clip(x, self.feature_clip_min, self.feature_clip_max)

        if self.editable_mask is not None:
            x = x * self.editable_mask + (1 - self.editable_mask) * np.clip(x, self.feature_clip_min, self.feature_clip_max)

        if self.integer_indices:
            x[:, self.integer_indices] = np.round(x[:, self.integer_indices])

        for group in self.categorical_groups:
            sub = x[:, group]
            max_idx = np.argmax(sub, axis=1)
            sub[:] = 0
            rows = np.arange(sub.shape[0])
            sub[rows, max_idx] = 1
            x[:, group] = sub

        return x

    def _init_bounds(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.feature_clip_min is not None and self.feature_clip_max is not None:
            return self.feature_clip_min, self.feature_clip_max
        if self.estimator.clip_values is not None:
            cmin, cmax = self.estimator.clip_values
            return np.ones_like(x[0]) * cmin, np.ones_like(x[0]) * cmax
        return np.min(x, axis=0), np.max(x, axis=0)

    def generate(self, x: np.ndarray, y: np.ndarray | None = None, **kwargs) -> np.ndarray:
        if x.ndim != 2:
            raise ValueError("Tabular SquareAttack expects 2D inputs (n_samples, n_features).")

        x_adv = x.astype(ART_NUMPY_DTYPE)
        x_orig = x_adv.copy()

        if isinstance(self.estimator, ClassifierMixin):
            if y is not None:
                y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)

        if y is None:
            logger.info("Using model predictions as true labels.")
            y = self.estimator.predict(x, batch_size=self.batch_size)
            if isinstance(self.estimator, ClassifierMixin):
                y = get_labels_np_array(y)

        if isinstance(self.estimator, ClassifierMixin):
            if self.estimator.nb_classes == 2 and y.shape[1] == 1:  # pragma: no cover
                raise ValueError("Binary classifier with single output not supported.")

        clip_min, clip_max = self._init_bounds(x_adv)
        n_features = x.shape[1]

        for _ in trange(self.nb_restarts, desc="SquareAttack - restarts", disable=not self.verbose):
            y_pred = self.estimator.predict(x_adv, batch_size=self.batch_size)
            robust = np.logical_not(self.adv_criterion(y_pred, y))
            if np.sum(robust) == 0:
                break

            x_robust = x_adv[robust]
            y_robust = y[robust]
            loss_init = self.loss(x_robust, y_robust)

            # init perturbation: flip a random subset
            k0 = max(1, int(round(self.p_init * n_features)))
            x_orig_robust = x_orig[robust]
            for i in range(x_robust.shape[0]):
                idx = np.random.choice(n_features, size=k0, replace=False)
                delta = np.random.choice([-self.eps, self.eps], size=idx.shape[0])
                x_robust[i, idx] = x_robust[i, idx] + delta
            x_robust = np.clip(x_robust, clip_min, clip_max)
            if self.norm in [np.inf, "inf"]:
                x_robust = np.clip(x_robust, x_orig_robust - self.eps, x_orig_robust + self.eps)
            else:
                diff = x_robust - x_orig_robust
                norms = np.linalg.norm(diff, axis=1, keepdims=True)
                scale = np.minimum(1.0, self.eps / (norms + 1e-12))
                x_robust = x_orig_robust + diff * scale
            x_robust = self._project_tabular(x_robust)

            loss_new = self.loss(x_robust, y_robust)
            improved = (loss_new - loss_init) < 0.0
            x_adv[robust] = np.where(improved[:, None], x_robust, x_adv[robust])

            for i_iter in trange(self.max_iter, desc="SquareAttack - iterations", leave=False, disable=not self.verbose):
                p = self._get_percentage_of_elements(i_iter)
                k = max(1, int(round(p * n_features)))

                y_pred = self.estimator.predict(x_adv, batch_size=self.batch_size)
                robust = np.logical_not(self.adv_criterion(y_pred, y))
                if np.sum(robust) == 0:
                    break

                x_robust = x_adv[robust]
                y_robust = y[robust]
                loss_init = self.loss(x_robust, y_robust)

                if self.norm in [np.inf, "inf"]:
                    deltas = np.zeros_like(x_robust)
                    for i in range(x_robust.shape[0]):
                        idx = np.random.choice(n_features, size=k, replace=False)
                        delta = np.random.choice([-self.eps, self.eps], size=idx.shape[0])
                        deltas[i, idx] = delta
                    x_new = x_robust + deltas
                    x_new = np.clip(x_new, clip_min, clip_max)
                    x_new = self._project_tabular(x_new)
                elif self.norm == 2:
                    deltas = np.zeros_like(x_robust)
                    for i in range(x_robust.shape[0]):
                        idx = np.random.choice(n_features, size=k, replace=False)
                        noise = np.random.randn(k)
                        noise = noise / (np.linalg.norm(noise) + 1e-12) * self.eps
                        deltas[i, idx] = noise
                    x_new = x_robust + deltas
                    x_new = np.clip(x_new, clip_min, clip_max)
                    x_new = self._project_tabular(x_new)
                else:
                    raise ValueError("Only Linf and L2 norms are supported in tabular SquareAttack.")

                loss_new = self.loss(x_new, y_robust)
                improved = (loss_new - loss_init) < 0.0
                x_adv[robust] = np.where(improved[:, None], x_new, x_adv[robust])

                # keep within epsilon-ball around original for linf/l2
                if self.norm in [np.inf, "inf"]:
                    x_adv = np.clip(x_adv, x_orig - self.eps, x_orig + self.eps)
                else:
                    diff = x_adv - x_orig
                    norms = np.linalg.norm(diff, axis=1, keepdims=True)
                    scale = np.minimum(1.0, self.eps / (norms + 1e-12))
                    x_adv = x_orig + diff * scale
                x_adv = np.clip(x_adv, clip_min, clip_max)
                x_adv = self._project_tabular(x_adv)

        return x_adv

    def _check_params(self) -> None:
        if self.norm not in [2, np.inf, "inf"]:
            raise ValueError('The argument norm has to be either 2 or `inf`/np.inf in tabular variant.')
        if not isinstance(self.max_iter, int) or self.max_iter <= 0:
            raise ValueError("max_iter must be positive int.")
        if not isinstance(self.eps, (int, float)) or self.eps <= 0.0:
            raise ValueError("eps must be positive.")
        if not isinstance(self.p_init, (int, float)) or self.p_init <= 0.0 or self.p_init >= 1.0:
            raise ValueError("p_init must be in (0,1).")
        if not isinstance(self.nb_restarts, int) or self.nb_restarts <= 0:
            raise ValueError("nb_restarts must be positive int.")
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("batch_size must be positive int.")
        if not isinstance(self.verbose, bool):
            raise ValueError("verbose must be bool.")
