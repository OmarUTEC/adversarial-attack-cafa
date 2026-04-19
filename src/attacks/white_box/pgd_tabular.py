from __future__ import annotations

import numpy as np
from art.attacks.evasion import ProjectedGradientDescent

from src.attacks.projection import project_tabular_constraints


class PGDAttackTabular(ProjectedGradientDescent):
    """
    PGD wrapper that projects adversarial samples back to a feasible tabular domain.
    """

    attack_params = ProjectedGradientDescent.attack_params + [
        "feature_clip_min",
        "feature_clip_max",
        "integer_indices",
        "categorical_groups",
        "editable_mask",
        "only_increase_mask",
        "only_decrease_mask",
        "max_abs_step",
    ]

    def __init__(
        self,
        estimator,
        x_reference: np.ndarray,
        feature_clip_min: np.ndarray,
        feature_clip_max: np.ndarray,
        integer_indices: list[int] | None = None,
        categorical_groups: list[list[int]] | None = None,
        editable_mask: np.ndarray | None = None,
        only_increase_mask: np.ndarray | None = None,
        only_decrease_mask: np.ndarray | None = None,
        max_abs_step: np.ndarray | None = None,
        **kwargs,
    ):
        super().__init__(estimator=estimator, **kwargs)
        self.x_reference = np.array(x_reference, copy=True)
        self.feature_clip_min = np.array(feature_clip_min, copy=True)
        self.feature_clip_max = np.array(feature_clip_max, copy=True)
        self.integer_indices = integer_indices or []
        self.categorical_groups = categorical_groups or []
        self.editable_mask = editable_mask
        self.only_increase_mask = only_increase_mask
        self.only_decrease_mask = only_decrease_mask
        self.max_abs_step = max_abs_step

    def generate(self, x: np.ndarray, y: np.ndarray | None = None, **kwargs) -> np.ndarray:
        self.x_reference = np.array(x, copy=True)
        x_adv = super().generate(x=x, y=y, **kwargs)
        return project_tabular_constraints(
            x_adv=x_adv,
            x_orig=self.x_reference,
            feature_clip_min=self.feature_clip_min,
            feature_clip_max=self.feature_clip_max,
            integer_indices=self.integer_indices,
            categorical_groups=self.categorical_groups,
            editable_mask=self.editable_mask,
            only_increase_mask=self.only_increase_mask,
            only_decrease_mask=self.only_decrease_mask,
            max_abs_step=self.max_abs_step,
        )
