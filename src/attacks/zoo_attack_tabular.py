from __future__ import annotations

import logging

import numpy as np

from art.attacks.evasion import ZooAttack
from art.config import ART_NUMPY_DTYPE

from src.attacks.projection import project_tabular_constraints

logger = logging.getLogger(__name__)


class ZooAttackTabular(ZooAttack):
    """
    ZOO adapted to tabular feature vectors with projection back to a feasible domain.
    """

    attack_params = ZooAttack.attack_params + [
        "feature_clip_min",
        "feature_clip_max",
        "integer_indices",
        "categorical_groups",
        "editable_mask",
        "only_increase_mask",
        "only_decrease_mask",
        "max_abs_step",
        "attack_name",
    ]

    def __init__(
        self,
        classifier,
        confidence: float = 0.0,
        targeted: bool = False,
        learning_rate: float = 1e-2,
        max_iter: int = 10,
        binary_search_steps: int = 1,
        initial_const: float = 1e-3,
        abort_early: bool = True,
        use_resize: bool = False,
        use_importance: bool = False,
        nb_parallel: int = 128,
        batch_size: int = 1,
        variable_h: float = 1e-4,
        verbose: bool = True,
        feature_clip_min: np.ndarray | None = None,
        feature_clip_max: np.ndarray | None = None,
        integer_indices: list[int] | None = None,
        categorical_groups: list[list[int]] | None = None,
        editable_mask: np.ndarray | None = None,
        only_increase_mask: np.ndarray | None = None,
        only_decrease_mask: np.ndarray | None = None,
        max_abs_step: np.ndarray | None = None,
        attack_name: str | None = None,
    ):
        super().__init__(
            classifier=classifier,
            confidence=confidence,
            targeted=targeted,
            learning_rate=learning_rate,
            max_iter=max_iter,
            binary_search_steps=binary_search_steps,
            initial_const=initial_const,
            abort_early=abort_early,
            use_resize=use_resize,
            use_importance=use_importance,
            nb_parallel=nb_parallel,
            batch_size=batch_size,
            variable_h=variable_h,
            verbose=verbose,
        )
        self.feature_clip_min = feature_clip_min
        self.feature_clip_max = feature_clip_max
        self.integer_indices = integer_indices or []
        self.categorical_groups = categorical_groups or []
        self.editable_mask = editable_mask
        self.only_increase_mask = only_increase_mask
        self.only_decrease_mask = only_decrease_mask
        self.max_abs_step = max_abs_step
        self.attack_name = attack_name
        self._x_orig_batch: np.ndarray | None = None
        self._check_tabular_params()

    def _check_tabular_params(self) -> None:
        expected_size = int(np.prod(self.estimator.input_shape))
        if self.feature_clip_min is not None and self.feature_clip_max is not None:
            if self.feature_clip_min.shape != self.feature_clip_max.shape:
                raise ValueError("feature_clip_min and feature_clip_max must share the same shape.")
            if self.feature_clip_min.size != expected_size:
                raise ValueError("feature_clip_min/feature_clip_max must match estimator input size.")
        if self.editable_mask is not None and self.editable_mask.size != expected_size:
            raise ValueError("editable_mask must match estimator input size.")
        if self.only_increase_mask is not None and self.only_increase_mask.size != expected_size:
            raise ValueError("only_increase_mask must match estimator input size.")
        if self.only_decrease_mask is not None and self.only_decrease_mask.size != expected_size:
            raise ValueError("only_decrease_mask must match estimator input size.")
        if self.max_abs_step is not None and self.max_abs_step.size != expected_size:
            raise ValueError("max_abs_step must match estimator input size.")

    def _generate_bss(
        self, x_batch: np.ndarray, y_batch: np.ndarray, c_batch: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._x_orig_batch = x_batch.astype(ART_NUMPY_DTYPE)
        return super()._generate_bss(x_batch, y_batch, c_batch)

    def _optimizer(self, x: np.ndarray, targets: np.ndarray, c_batch: np.ndarray) -> np.ndarray:
        x_next = super()._optimizer(x, targets, c_batch)
        x_projected = self._project_tabular(x_next, x_reference=self._x_orig_batch)
        self._current_noise = x_projected - x
        return x_projected

    def generate(self, x: np.ndarray, y: np.ndarray | None = None, **kwargs) -> np.ndarray:
        x_adv = super().generate(x, y=y, **kwargs)
        return self._project_tabular(x_adv, x_reference=x.astype(ART_NUMPY_DTYPE))

    def _project_tabular(self, x_adv: np.ndarray, x_reference: np.ndarray | None) -> np.ndarray:
        if x_reference is None:
            x_reference = x_adv

        feature_clip_min, feature_clip_max = self._init_bounds(x_adv)
        return project_tabular_constraints(
            x_adv=x_adv,
            x_orig=x_reference,
            feature_clip_min=feature_clip_min,
            feature_clip_max=feature_clip_max,
            integer_indices=self.integer_indices,
            categorical_groups=self.categorical_groups,
            editable_mask=self.editable_mask,
            only_increase_mask=self.only_increase_mask,
            only_decrease_mask=self.only_decrease_mask,
            max_abs_step=self.max_abs_step,
        )

    def _init_bounds(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.feature_clip_min is not None and self.feature_clip_max is not None:
            return self.feature_clip_min.astype(ART_NUMPY_DTYPE), self.feature_clip_max.astype(ART_NUMPY_DTYPE)
        if self.estimator.clip_values is not None:
            cmin, cmax = self.estimator.clip_values
            return np.ones_like(x[0], dtype=ART_NUMPY_DTYPE) * cmin, np.ones_like(x[0], dtype=ART_NUMPY_DTYPE) * cmax
        return np.min(x, axis=0), np.max(x, axis=0)


class ZooAttackTabularFromDataset(ZooAttackTabular):
    """
    Convenience wrapper to configure ZOO with tabular constraints from TabularDataset.
    """

    def __init__(self, classifier, tab_dataset, **kwargs):
        feature_clip_min = tab_dataset.feature_ranges[:, 0].astype(np.float32)
        feature_clip_max = tab_dataset.feature_ranges[:, 1].astype(np.float32)
        data_min = tab_dataset.X_train.min(axis=0)
        data_max = tab_dataset.X_train.max(axis=0)
        feature_clip_min = np.where(np.isfinite(feature_clip_min), feature_clip_min, data_min)
        feature_clip_max = np.where(np.isfinite(feature_clip_max), feature_clip_max, data_max)

        super().__init__(
            classifier=classifier,
            feature_clip_min=feature_clip_min,
            feature_clip_max=feature_clip_max,
            integer_indices=tab_dataset.ordinal_indices.tolist(),
            categorical_groups=[group.tolist() for group in tab_dataset.one_hot_groups],
            **kwargs,
        )
