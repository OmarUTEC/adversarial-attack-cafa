from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from tqdm.auto import tqdm

from art.attacks.evasion import SignOPTAttack
from art.config import ART_NUMPY_DTYPE
from art.utils import check_and_transform_label_format, compute_success, get_labels_np_array

from src.attacks.projection import project_tabular_constraints

logger = logging.getLogger(__name__)


class SignOPTAttackTabular(SignOPTAttack):
    """
    SignOPT adapted to tabular feature vectors with projection back to a feasible domain.
    """

    attack_params = SignOPTAttack.attack_params + [
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
        estimator,
        targeted: bool = False,
        epsilon: float = 0.001,
        num_trial: int = 100,
        max_iter: int = 1000,
        query_limit: int = 20000,
        k: int = 200,
        alpha: float = 0.2,
        beta: float = 0.001,
        eval_perform: bool = False,
        batch_size: int = 64,
        verbose: bool = False,
        feature_clip_min: np.ndarray | None = None,
        feature_clip_max: np.ndarray | None = None,
        integer_indices: list[int] | None = None,
        categorical_groups: list[list[int]] | None = None,
        editable_mask: np.ndarray | None = None,
        only_increase_mask: np.ndarray | None = None,
        only_decrease_mask: np.ndarray | None = None,
        max_abs_step: np.ndarray | None = None,
        attack_name: str | None = None,
    ) -> None:
        super().__init__(
            estimator=estimator,
            targeted=targeted,
            epsilon=epsilon,
            num_trial=num_trial,
            max_iter=max_iter,
            query_limit=query_limit,
            k=k,
            alpha=alpha,
            beta=beta,
            eval_perform=eval_perform,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.feature_clip_min = feature_clip_min
        self.feature_clip_max = feature_clip_max
        self.clip_min = feature_clip_min
        self.clip_max = feature_clip_max
        self.integer_indices = integer_indices or []
        self.categorical_groups = categorical_groups or []
        self.editable_mask = editable_mask
        self.only_increase_mask = only_increase_mask
        self.only_decrease_mask = only_decrease_mask
        self.max_abs_step = max_abs_step
        self.attack_name = attack_name
        self._current_x0: np.ndarray | None = None
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

    def generate(self, x: np.ndarray, y: np.ndarray | None = None, **kwargs) -> np.ndarray:
        if y is None:
            if self.targeted:
                raise ValueError("Target labels `y` need to be provided for a targeted attack.")
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))  # type: ignore

        targets = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes, return_one_hot=False)
        if targets is not None:
            targets = np.asarray(targets).reshape(-1).astype(int)

        if self.targeted and targets is None:
            raise ValueError("Target labels `y` need to be provided for a targeted attack.")

        x_init = kwargs.get("x_init")

        if self.clip_min is None and self.clip_max is None:
            inferred_min, inferred_max = self._init_bounds(x)
            self.clip_min, self.clip_max = inferred_min, inferred_max

        preds = np.argmax(self.estimator.predict(x, batch_size=self.batch_size), axis=1)
        x_adv = x.astype(ART_NUMPY_DTYPE)

        counter = 0
        for ind, val in enumerate(tqdm(x_adv, desc="Sign_OPT attack", disable=not self.verbose)):
            self._current_x0 = val.copy().astype(ART_NUMPY_DTYPE)
            if self.targeted:
                if targets[ind] == preds[ind]:
                    continue
                if x_init is None:
                    raise ValueError("`x_init` needs to be provided for a targeted attack.")
                x_adv[ind], diff, succeed = self._attack(
                    x_0=val,
                    y_0=preds[ind],
                    target=targets[ind],
                    x_init=x_init,
                )
            else:
                x_adv[ind], diff, succeed = self._attack(
                    x_0=val,
                    y_0=preds[ind],
                )
            if succeed and self.eval_perform and counter < 100:
                self.logs[counter] = np.linalg.norm(diff)
                counter += 1

        x_adv = self._project_tabular(x_adv, x_reference=x.astype(ART_NUMPY_DTYPE))

        if self.targeted is False:
            logger.info(
                "Success rate of Sign_OPT attack: %.2f%%",
                100 * compute_success(self.estimator, x, targets, x_adv, self.targeted, batch_size=self.batch_size),
            )

        return x_adv

    def _clip_value(self, x_0: np.ndarray) -> np.ndarray:
        x_proj = self._project_tabular(np.expand_dims(x_0, axis=0), x_reference=self._reference_for_single())
        return x_proj[0]

    def _is_label(self, x_0: np.ndarray, label: Optional[int]) -> bool:
        x_proj = self._clip_value(x_0)
        pred = self.estimator.predict(np.expand_dims(x_proj, axis=0), batch_size=self.batch_size)
        pred_y0 = np.argmax(pred)
        return pred_y0 == label

    def _reference_for_single(self) -> np.ndarray:
        if self._current_x0 is None:
            raise ValueError("Current original sample reference is not set.")
        return np.expand_dims(self._current_x0, axis=0)

    def _project_tabular(self, x_adv: np.ndarray, x_reference: np.ndarray) -> np.ndarray:
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


class SignOPTAttackTabularFromDataset(SignOPTAttackTabular):
    """
    Convenience wrapper to configure SignOPT with tabular constraints from TabularDataset.
    """

    def __init__(self, estimator, tab_dataset, **kwargs):
        feature_clip_min = tab_dataset.feature_ranges[:, 0].astype(np.float32)
        feature_clip_max = tab_dataset.feature_ranges[:, 1].astype(np.float32)
        data_min = tab_dataset.X_train.min(axis=0)
        data_max = tab_dataset.X_train.max(axis=0)
        feature_clip_min = np.where(np.isfinite(feature_clip_min), feature_clip_min, data_min)
        feature_clip_max = np.where(np.isfinite(feature_clip_max), feature_clip_max, data_max)

        super().__init__(
            estimator=estimator,
            feature_clip_min=feature_clip_min,
            feature_clip_max=feature_clip_max,
            integer_indices=tab_dataset.ordinal_indices.tolist(),
            categorical_groups=[group.tolist() for group in tab_dataset.one_hot_groups],
            **kwargs,
        )
