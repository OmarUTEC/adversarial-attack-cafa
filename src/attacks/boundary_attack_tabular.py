from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from art.attacks.evasion import BoundaryAttack as ARTBoundaryAttack
from art.estimators.classification import ClassifierMixin
from art.estimators.estimator import BaseEstimator

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

logger = logging.getLogger(__name__)

class BoundaryAttackTabular(ARTBoundaryAttack):
    """
    Boundary Attack (black-box) adapted for tabular data.
    """
    def __init__(
        self,
        estimator: CLASSIFIER_TYPE,
        feature_clip_min: np.ndarray | float | None = None,
        feature_clip_max: np.ndarray | float | None = None,
        integer_indices: list[int] | None = None,
        categorical_groups: list[list[int]] | None = None,
        targeted: bool = False,
        delta: float = 0.01,
        epsilon: float = 0.01,
        step_adapt: float = 0.667,
        max_iter: int = 100,
        num_trial: int = 25,
        sample_size: int = 20,
        init_size: int = 100,
        min_epsilon: float = 0.0,
        verbose: bool = True,
    ):
        super().__init__(
            estimator=estimator,
            targeted=targeted,
            delta=delta,
            epsilon=epsilon,
            step_adapt=step_adapt,
            max_iter=max_iter,
            num_trial=num_trial,
            sample_size=sample_size,
            init_size=init_size,
            min_epsilon=min_epsilon,
            verbose=verbose,
        )
        self.feature_clip_min = feature_clip_min
        self.feature_clip_max = feature_clip_max
        self.integer_indices = integer_indices
        self.categorical_groups = categorical_groups

    def generate(self, x: np.ndarray, y: np.ndarray | None = None, **kwargs) -> np.ndarray:
        # Fix input_shape: ART's BoundaryAttack needs a tuple for np.random.randn(*shape).
        original_input_shape = self.estimator.input_shape
        if isinstance(original_input_shape, int) and hasattr(self.estimator, '_input_shape'):
            self.estimator._input_shape = (original_input_shape,)

        # Wrap estimator.predict so every internal query from ART's boundary walk receives
        # projected (constraint-valid) inputs.  Without this, the attack explores and anchors
        # to invalid points (broken one-hot groups, out-of-range values) that only become
        # valid after the post-hoc projection, making the found boundary unreliable.
        original_predict = self.estimator.predict
        def _projecting_predict(x_input, **pkwargs):
            return original_predict(self._project_tabular(x_input), **pkwargs)
        self.estimator.predict = _projecting_predict

        try:
            x_adv = super().generate(x=x, y=y, **kwargs)
        finally:
            self.estimator.predict = original_predict
            if isinstance(original_input_shape, int) and hasattr(self.estimator, '_input_shape'):
                self.estimator._input_shape = original_input_shape

        return self._project_tabular(x_adv)

    def _project_tabular(self, x: np.ndarray) -> np.ndarray:
        x_proj = x.copy()
        
        # 1. Clip features
        if self.feature_clip_min is not None and self.feature_clip_max is not None:
            x_proj = np.clip(x_proj, self.feature_clip_min, self.feature_clip_max)
            
        # 2. Round integer/ordinal indices
        if self.integer_indices:
            x_proj[:, self.integer_indices] = np.rint(x_proj[:, self.integer_indices])
            
        # 3. Handle categorical (one-hot) groups
        if self.categorical_groups:
            for group in self.categorical_groups:
                # Find index with max value in the one-hot group
                group_data = x_proj[:, group]
                max_indices = np.argmax(group_data, axis=1)
                # Reset to zero and set the max to 1
                x_proj[:, group] = 0
                for i, idx in enumerate(max_indices):
                    x_proj[i, group[idx]] = 1
                    
        return x_proj
