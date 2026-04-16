from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from tqdm.auto import tqdm, trange

from art.attacks.attack import EvasionAttack
from art.attacks.evasion import BoundaryAttack
from art.config import ART_NUMPY_DTYPE
from art.estimators.classification import ClassifierMixin
from art.estimators.estimator import BaseEstimator
from art.utils import check_and_transform_label_format, compute_success, get_labels_np_array, to_categorical

from src.attacks.projection import project_tabular_constraints

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

logger = logging.getLogger(__name__)


class BoundaryAttackTabular(BoundaryAttack):
    """
    BoundaryAttack adapted to tabular feature vectors with projection to a feasible domain.
    """

    attack_params = EvasionAttack.attack_params + [
        "targeted",
        "delta",
        "epsilon",
        "step_adapt",
        "max_iter",
        "num_trial",
        "sample_size",
        "init_size",
        "min_epsilon",
        "batch_size",
        "verbose",
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

    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(
        self,
        estimator: "CLASSIFIER_TYPE",
        batch_size: int = 64,
        targeted: bool = False,
        delta: float = 0.01,
        epsilon: float = 0.01,
        step_adapt: float = 0.667,
        max_iter: int = 5000,
        num_trial: int = 25,
        sample_size: int = 20,
        init_size: int = 100,
        min_epsilon: float = 0.0,
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
    ) -> None:
        super().__init__(
            estimator=estimator,
            batch_size=batch_size,
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
        self.integer_indices = integer_indices or []
        self.categorical_groups = categorical_groups or []
        self.editable_mask = editable_mask
        self.only_increase_mask = only_increase_mask
        self.only_decrease_mask = only_decrease_mask
        self.max_abs_step = max_abs_step
        self.attack_name = attack_name
        self._current_original: np.ndarray | None = None
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

        y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes, return_one_hot=False)

        clip_min, clip_max = self._init_bounds(x)
        preds = np.argmax(self.estimator.predict(x, batch_size=self.batch_size), axis=1)

        x_adv_init = kwargs.get("x_adv_init")
        if x_adv_init is not None:
            x_adv_init = self._project_tabular(x_adv_init, x_reference=x)
            init_preds = np.argmax(self.estimator.predict(x_adv_init, batch_size=self.batch_size), axis=1)
        else:
            init_preds = [None] * len(x)
            x_adv_init = [None] * len(x)

        x_adv = x.astype(ART_NUMPY_DTYPE)

        for ind, val in enumerate(tqdm(x_adv, desc="Boundary attack", disable=not self.verbose)):
            self._current_original = val.copy().astype(ART_NUMPY_DTYPE)
            if self.targeted:
                x_adv[ind] = self._perturb(
                    x=val,
                    y=int(y[ind]),
                    y_p=int(preds[ind]),
                    init_pred=init_preds[ind],
                    adv_init=x_adv_init[ind],
                    clip_min=clip_min,
                    clip_max=clip_max,
                )
            else:
                x_adv[ind] = self._perturb(
                    x=val,
                    y=-1,
                    y_p=int(preds[ind]),
                    init_pred=init_preds[ind],
                    adv_init=x_adv_init[ind],
                    clip_min=clip_min,
                    clip_max=clip_max,
                )

        y_onehot = to_categorical(y, self.estimator.nb_classes)
        logger.info(
            "Success rate of Boundary attack: %.2f%%",
            100 * compute_success(self.estimator, x, y_onehot, x_adv, self.targeted, batch_size=self.batch_size),
        )

        return x_adv

    def _perturb(
        self,
        x: np.ndarray,
        y: int,
        y_p: int,
        init_pred: int,
        adv_init: np.ndarray,
        clip_min: np.ndarray,
        clip_max: np.ndarray,
    ) -> np.ndarray:
        initial_sample = self._init_sample(x, y, y_p, init_pred, adv_init, clip_min, clip_max)
        if initial_sample is None:
            return x
        return self._attack(initial_sample[0], x, y_p, initial_sample[1], self.delta, self.epsilon, clip_min, clip_max)

    def _init_sample(
        self,
        x: np.ndarray,
        y: int,
        y_p: int,
        init_pred: int,
        adv_init: np.ndarray,
        clip_min: np.ndarray,
        clip_max: np.ndarray,
    ):
        nprd = np.random.RandomState()
        initial_sample = None

        if self.targeted:
            if y == y_p:
                return None
            if adv_init is not None and init_pred == y:
                return adv_init.astype(ART_NUMPY_DTYPE), init_pred

            for _ in range(self.init_size):
                random_img = nprd.uniform(clip_min, clip_max, size=x.shape).astype(x.dtype)
                random_img = self._project_single(random_img, x_reference=x)
                random_class = np.argmax(
                    self.estimator.predict(np.array([random_img]), batch_size=self.batch_size),
                    axis=1,
                )[0]
                if random_class == y:
                    initial_sample = random_img, random_class
                    logger.info("Found initial adversarial image for targeted attack.")
                    break
            else:
                logger.warning("Failed to draw a random image that is adversarial, attack failed.")

        else:
            if adv_init is not None and init_pred != y_p:
                return adv_init.astype(ART_NUMPY_DTYPE), init_pred

            for _ in range(self.init_size):
                random_img = nprd.uniform(clip_min, clip_max, size=x.shape).astype(x.dtype)
                random_img = self._project_single(random_img, x_reference=x)
                random_class = np.argmax(
                    self.estimator.predict(np.array([random_img]), batch_size=self.batch_size),
                    axis=1,
                )[0]
                if random_class != y_p:
                    initial_sample = random_img, random_class
                    logger.info("Found initial adversarial image for untargeted attack.")
                    break
            else:
                logger.warning("Failed to draw a random image that is adversarial, attack failed.")

        return initial_sample

    def _attack(
        self,
        initial_sample: np.ndarray,
        original_sample: np.ndarray,
        y_p: int,
        target: int,
        initial_delta: float,
        initial_epsilon: float,
        clip_min: np.ndarray,
        clip_max: np.ndarray,
    ) -> np.ndarray:
        x_adv = self._project_single(initial_sample, x_reference=original_sample)
        self.curr_delta = initial_delta
        self.curr_epsilon = initial_epsilon
        self.curr_adv = x_adv

        for _ in trange(self.max_iter, desc="Boundary attack - iterations", disable=not self.verbose):
            for _ in range(self.num_trial):
                potential_advs_list: list[np.ndarray] = []
                for _ in range(self.sample_size):
                    potential_adv = x_adv + self._orthogonal_perturb(self.curr_delta, x_adv, original_sample)
                    potential_adv = self._project_single(potential_adv, x_reference=original_sample)
                    potential_advs_list.append(potential_adv)

                preds = np.argmax(
                    self.estimator.predict(np.array(potential_advs_list), batch_size=self.batch_size),
                    axis=1,
                )

                satisfied = preds == target if self.targeted else preds != y_p
                delta_ratio = np.mean(satisfied)

                if delta_ratio < 0.2:
                    self.curr_delta *= self.step_adapt
                elif delta_ratio > 0.5:
                    self.curr_delta /= self.step_adapt

                if delta_ratio > 0:
                    x_advs = np.array(potential_advs_list)[np.where(satisfied)[0]]
                    break
            else:
                logger.warning("Adversarial example found but not optimal.")
                return x_adv

            for _ in range(self.num_trial):
                perturb = np.repeat(np.array([original_sample]), len(x_advs), axis=0) - x_advs
                perturb *= self.curr_epsilon
                potential_advs = x_advs + perturb
                potential_advs = self._project_tabular(potential_advs, x_reference=np.repeat(np.array([original_sample]), len(x_advs), axis=0))
                preds = np.argmax(
                    self.estimator.predict(potential_advs, batch_size=self.batch_size),
                    axis=1,
                )

                satisfied = preds == target if self.targeted else preds != y_p
                epsilon_ratio = np.mean(satisfied)

                if epsilon_ratio < 0.2:
                    self.curr_epsilon *= self.step_adapt
                elif epsilon_ratio > 0.5:
                    self.curr_epsilon /= self.step_adapt

                if epsilon_ratio > 0:
                    x_adv = self._best_adv(original_sample, potential_advs[np.where(satisfied)[0]])
                    x_adv = self._project_single(x_adv, x_reference=original_sample)
                    self.curr_adv = x_adv
                    break
            else:
                logger.warning("Adversarial example found but not optimal.")
                return self._project_single(self._best_adv(original_sample, x_advs), x_reference=original_sample)

            if self.curr_epsilon < self.min_epsilon:
                return x_adv

        return x_adv

    def _orthogonal_perturb(self, delta: float, current_sample: np.ndarray, original_sample: np.ndarray) -> np.ndarray:
        perturb = np.random.randn(*self.estimator.input_shape).astype(ART_NUMPY_DTYPE)
        perturb /= np.linalg.norm(perturb)
        perturb *= delta * np.linalg.norm(original_sample - current_sample)

        direction = original_sample - current_sample
        direction_flat = direction.flatten()
        perturb_flat = perturb.flatten()

        direction_flat /= np.linalg.norm(direction_flat)
        perturb_flat -= np.dot(perturb_flat, direction_flat.T) * direction_flat
        perturb = perturb_flat.reshape(self.estimator.input_shape)

        hypotenuse = np.sqrt(1 + delta**2)
        perturb = ((1 - hypotenuse) * (current_sample - original_sample) + perturb) / hypotenuse
        return perturb

    def _project_single(self, sample: np.ndarray, x_reference: np.ndarray) -> np.ndarray:
        return self._project_tabular(np.expand_dims(sample, axis=0), np.expand_dims(x_reference, axis=0))[0]

    def _project_tabular(self, x_adv: np.ndarray, x_reference: np.ndarray) -> np.ndarray:
        return project_tabular_constraints(
            x_adv=x_adv,
            x_orig=x_reference,
            feature_clip_min=self.feature_clip_min,
            feature_clip_max=self.feature_clip_max,
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


class BoundaryAttackTabularFromDataset(BoundaryAttackTabular):
    """
    Convenience wrapper to configure BoundaryAttack with tabular constraints from TabularDataset.
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
