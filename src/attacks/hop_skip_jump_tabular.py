# MIT License
#
# Adapted HopSkipJump attack for tabular data: adds per-feature clipping, one-hot/integer projection,
# and optional editability mask/decision threshold.
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from tqdm.auto import tqdm

from art.attacks.attack import EvasionAttack
from art.config import ART_NUMPY_DTYPE
from art.estimators.classification import ClassifierMixin
from art.estimators.estimator import BaseEstimator
from art.utils import check_and_transform_label_format, compute_success, get_labels_np_array, to_categorical

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

logger = logging.getLogger(__name__)


class HopSkipJump(EvasionAttack):
    """
    HopSkipJump attack (black-box) with tabular projection.
    """

    attack_params = EvasionAttack.attack_params + [
        "targeted",
        "norm",
        "max_iter",
        "max_eval",
        "init_eval",
        "init_size",
        "curr_iter",
        "batch_size",
        "verbose",
        "feature_clip_min",
        "feature_clip_max",
        "integer_indices",
        "categorical_groups",
        "editable_mask",
        "decision_threshold",
        "feature_importance",
        "attack_name",
        "max_eval_per_iter",
    ]
    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(
        self,
        classifier: "CLASSIFIER_TYPE",
        batch_size: int = 64,
        targeted: bool = False,
        norm: int | float | str = 2,
        max_iter: int = 50,
        max_eval: int = 10000,
        init_eval: int = 100,
        init_size: int = 100,
        verbose: bool = True,
        feature_clip_min: np.ndarray | None = None,
        feature_clip_max: np.ndarray | None = None,
        integer_indices: list[int] | None = None,
        categorical_groups: list[list[int]] | None = None,
        editable_mask: np.ndarray | None = None,
        decision_threshold: float | None = None,
        feature_importance: np.ndarray | None = None,
        attack_name: str | None = None,  # accepted but unused; kept for config symmetry
        max_eval_per_iter: int | None = 512,
    ) -> None:
        super().__init__(estimator=classifier)
        self._targeted = targeted
        self.norm = norm
        self.max_iter = max_iter
        self.max_eval = max_eval
        self.init_eval = init_eval
        self.init_size = init_size
        self.curr_iter = 0
        self.batch_size = batch_size
        self.verbose = verbose
        self.max_eval_per_iter = max_eval_per_iter if max_eval_per_iter is not None else max_eval
        # Tabular-specific configuration
        self.feature_clip_min = feature_clip_min
        self.feature_clip_max = feature_clip_max
        self.integer_indices = integer_indices or []
        self.categorical_groups = categorical_groups or []
        self.editable_mask = editable_mask
        self.decision_threshold = decision_threshold
        self.feature_importance = feature_importance
        self._check_params()
        self.curr_iter = 0

        # Set binary search threshold
        if norm == 2:
            self.theta = 0.01 / np.sqrt(np.prod(self.estimator.input_shape))
        else:
            self.theta = 0.01 / np.prod(self.estimator.input_shape)

    def generate(self, x: np.ndarray, y: np.ndarray | None = None, **kwargs) -> np.ndarray:
        mask = kwargs.get("mask")

        if y is None:
            if self.targeted:  # pragma: no cover
                raise ValueError("Target labels `y` need to be provided for a targeted attack.")
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))  # type: ignore

        y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)

        if self.estimator.nb_classes == 2 and y.shape[1] == 1:  # pragma: no cover
            raise ValueError(
                "This attack has not yet been tested for binary classification with a single output classifier."
            )

        resume = kwargs.get("resume")
        start = self.curr_iter if resume else 0

        if mask is not None:
            if len(mask.shape) == len(x.shape):
                mask = mask.astype(ART_NUMPY_DTYPE)
            else:
                mask = np.array([mask.astype(ART_NUMPY_DTYPE)] * x.shape[0])
        else:
            # Tabular: if editable_mask was provided in __init__, use it as default mask
            if self.editable_mask is not None:
                mask = np.array([self.editable_mask.astype(ART_NUMPY_DTYPE)] * x.shape[0])
            else:
                mask = np.array([None] * x.shape[0])

        if self.estimator.clip_values is not None:
            clip_min, clip_max = self.estimator.clip_values
        else:
            clip_min, clip_max = np.min(x), np.max(x)

        clip_min = self._expand_clip_value(clip_min, kind="min")
        clip_max = self._expand_clip_value(clip_max, kind="max")

        preds = np.argmax(self.estimator.predict(x, batch_size=self.batch_size), axis=1)

        x_adv_init = kwargs.get("x_adv_init")
        if x_adv_init is not None:
            for i in range(x.shape[0]):
                if mask[i] is not None:
                    x_adv_init[i] = x_adv_init[i] * mask[i] + x[i] * (1 - mask[i])
            init_preds = np.argmax(self.estimator.predict(x_adv_init, batch_size=self.batch_size), axis=1)
        else:
            init_preds = [None] * len(x)
            x_adv_init = [None] * len(x)

        if self.targeted and y is None:  # pragma: no cover
            raise ValueError("Target labels `y` need to be provided for a targeted attack.")

        x_adv = x.astype(ART_NUMPY_DTYPE)

        y = np.argmax(y, axis=1)

        for ind, val in enumerate(tqdm(x_adv, desc="HopSkipJump", disable=not self.verbose)):
            self.curr_iter = start
            if self.targeted:
                x_adv[ind] = self._perturb(
                    x=val,
                    y=y[ind],  # type: ignore
                    y_p=preds[ind],
                    init_pred=init_preds[ind],
                    adv_init=x_adv_init[ind],
                    mask=mask[ind],
                    clip_min=clip_min,
                    clip_max=clip_max,
                )
            else:
                x_adv[ind] = self._perturb(
                    x=val,
                    y=-1,
                    y_p=preds[ind],
                    init_pred=init_preds[ind],
                    adv_init=x_adv_init[ind],
                    mask=mask[ind],
                    clip_min=clip_min,
                    clip_max=clip_max,
                )

        y = to_categorical(y, self.estimator.nb_classes)  # type: ignore

        logger.info(
            "Success rate of HopSkipJump attack: %.2f%%",
            100 * compute_success(self.estimator, x, y, x_adv, self.targeted, batch_size=self.batch_size),
        )

        return x_adv

    def _perturb(
        self,
        x: np.ndarray,
        y: int,
        y_p: int,
        init_pred: int,
        adv_init: np.ndarray,
        mask: np.ndarray | None,
        clip_min: float,
        clip_max: float,
    ) -> np.ndarray:
        initial_sample = self._init_sample(x, y, y_p, init_pred, adv_init, mask, clip_min, clip_max)
        if initial_sample is None:
            return x
        x_adv = self._attack(initial_sample[0], x, initial_sample[1], mask, clip_min, clip_max)
        return x_adv

    def _init_sample(
        self,
        x: np.ndarray,
        y: int,
        y_p: int,
        init_pred: int,
        adv_init: np.ndarray,
        mask: np.ndarray | None,
        clip_min: float,
        clip_max: float,
    ) -> np.ndarray | tuple[np.ndarray, int] | None:
        nprd = np.random.RandomState()
        initial_sample = None

        if self.targeted:
            if y == y_p:
                return None
            if adv_init is not None and init_pred == y:
                return adv_init.astype(ART_NUMPY_DTYPE), init_pred

            for _ in range(self.init_size):
                random_img = nprd.uniform(clip_min, clip_max, size=x.shape).astype(x.dtype)
                if mask is not None:
                    random_img = random_img * mask + x * (1 - mask)
                random_img = self._project_tabular(random_img, clip_min, clip_max, mask)
                random_class = np.argmax(
                    self.estimator.predict(np.array([random_img]), batch_size=self.batch_size),
                    axis=1,
                )[0]

                if random_class == y:
                    random_img = self._binary_search(
                        current_sample=random_img,
                        original_sample=x,
                        target=y,
                        norm=2,
                        clip_min=clip_min,
                        clip_max=clip_max,
                        threshold=0.001,
                        mask=mask,
                    )
                    initial_sample = random_img, random_class
                    logger.info("Found initial adversarial image for targeted attack.")
                    break
            else:
                logger.warning("Failed to draw a random image that is adversarial, attack failed.")

        else:
            if adv_init is not None and init_pred != y_p:
                return adv_init.astype(ART_NUMPY_DTYPE), y_p

            for _ in range(self.init_size):
                random_img = nprd.uniform(clip_min, clip_max, size=x.shape).astype(x.dtype)
                if mask is not None:
                    random_img = random_img * mask + x * (1 - mask)
                random_img = self._project_tabular(random_img, clip_min, clip_max, mask)
                random_class = np.argmax(
                    self.estimator.predict(np.array([random_img]), batch_size=self.batch_size),
                    axis=1,
                )[0]

                if random_class != y_p:
                    random_img = self._binary_search(
                        current_sample=random_img,
                        original_sample=x,
                        target=y_p,
                        norm=2,
                        clip_min=clip_min,
                        clip_max=clip_max,
                        threshold=0.001,
                        mask=mask,
                    )
                    initial_sample = random_img, y_p
                    logger.info("Found initial adversarial image for untargeted attack.")
                    break
            else:
                logger.warning("Failed to draw a random image that is adversarial, attack failed.")

        return initial_sample

    def _attack(
        self,
        initial_sample: np.ndarray,
        original_sample: np.ndarray,
        target: int,
        mask: np.ndarray | None,
        clip_min: float,
        clip_max: float,
    ) -> np.ndarray:
        current_sample = initial_sample

        for _ in range(self.max_iter):
            delta = self._compute_delta(
                current_sample=current_sample,
                original_sample=original_sample,
                clip_min=clip_min,
                clip_max=clip_max,
            )

            current_sample = self._binary_search(
                current_sample=current_sample,
                original_sample=original_sample,
                norm=self.norm,
                target=target,
                clip_min=clip_min,
                clip_max=clip_max,
                mask=mask,
            )

            # Limit the number of model evaluations per iteration to control runtime
            num_eval = int(self.init_eval * np.sqrt(self.curr_iter + 1))
            num_eval = min(self.max_eval, num_eval)
            num_eval = min(self.max_eval_per_iter, num_eval)
            num_eval = max(self.init_eval, num_eval)

            update = self._compute_update(
                current_sample=current_sample,
                num_eval=num_eval,
                delta=delta,
                target=target,
                mask=mask,
                clip_min=clip_min,
                clip_max=clip_max,
            )

            if self.norm == 2:
                dist = np.linalg.norm(original_sample - current_sample)
            else:
                dist = np.max(abs(original_sample - current_sample))

            epsilon = 2.0 * dist / np.sqrt(self.curr_iter + 1)
            potential_sample = current_sample
            success = False

            # Binary search along update direction; cap steps to avoid infinite loops when no adversarial is found.
            for _ in range(50):
                epsilon /= 2.0
                potential_sample = current_sample + epsilon * update
                success = bool(
                    self._adversarial_satisfactory(  # type: ignore
                        samples=potential_sample[None],
                        target=target,
                        clip_min=clip_min,
                        clip_max=clip_max,
                    )[0]
                )
                if success:
                    break

            if not success:
                logger.debug("HopSkipJump: no successful step found after 50 halvings; stopping early for this sample.")
                break

            current_sample = np.clip(potential_sample, clip_min, clip_max)
            current_sample = self._project_tabular(current_sample, clip_min, clip_max, mask)

            self.curr_iter += 1

            if np.isnan(current_sample).any():  # pragma: no cover
                logger.debug("NaN detected in sample, returning original sample.")
                return original_sample

        return current_sample

    def _binary_search(
        self,
        current_sample: np.ndarray,
        original_sample: np.ndarray,
        target: int,
        norm: int | float | str,
        clip_min: float,
        clip_max: float,
        threshold: float | None = None,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        if norm == 2:
            (upper_bound, lower_bound) = (np.array(1.0), np.array(0.0))
            if threshold is None:
                threshold = self.theta
        else:
            (upper_bound, lower_bound) = (
                np.max(abs(original_sample - current_sample)),
                np.array(0.0),
            )
            if threshold is None:
                threshold = np.minimum(upper_bound * self.theta, self.theta)

        while (upper_bound - lower_bound) > threshold:
            alpha = (upper_bound + lower_bound) / 2.0
            interpolated_sample = self._interpolate(
                current_sample=current_sample,
                original_sample=original_sample,
                alpha=alpha,
                norm=norm,
            )

            satisfied = self._adversarial_satisfactory(
                samples=interpolated_sample[None],
                target=target,
                clip_min=clip_min,
                clip_max=clip_max,
            )[0]
            lower_bound = np.where(satisfied == 0, alpha, lower_bound)
            upper_bound = np.where(satisfied == 1, alpha, upper_bound)

        result = self._interpolate(
            current_sample=current_sample,
            original_sample=original_sample,
            alpha=float(upper_bound),
            norm=norm,
        )

        return self._project_tabular(result, clip_min, clip_max, mask)

    def _compute_delta(
        self,
        current_sample: np.ndarray,
        original_sample: np.ndarray,
        clip_min: float,
        clip_max: float,
    ) -> float:
        if self.curr_iter == 0:
            return 0.1 * (clip_max - clip_min)

        if self.norm == 2:
            dist = np.linalg.norm(original_sample - current_sample)
            delta = np.sqrt(np.prod(self.estimator.input_shape)) * self.theta * dist
        else:
            dist = np.max(abs(original_sample - current_sample))
            delta = np.prod(self.estimator.input_shape) * self.theta * dist

        return delta

    def _compute_update(
        self,
        current_sample: np.ndarray,
        num_eval: int,
        delta: float,
        target: int,
        mask: np.ndarray | None,
        clip_min: float,
        clip_max: float,
    ) -> np.ndarray:
        input_shape = self.estimator.input_shape
        input_dim = input_shape if isinstance(input_shape, int) else int(np.prod(input_shape))
        rnd_noise_shape = [num_eval, input_dim]
        if self.norm == 2:
            rnd_noise = np.random.randn(*rnd_noise_shape).astype(ART_NUMPY_DTYPE)
        else:
            rnd_noise = np.random.uniform(low=-1, high=1, size=rnd_noise_shape).astype(ART_NUMPY_DTYPE)

        if mask is not None:
            rnd_noise = rnd_noise * mask

        rnd_noise = rnd_noise / np.sqrt(
            np.sum(
                rnd_noise**2,
                axis=tuple(range(len(rnd_noise_shape)))[1:],
                keepdims=True,
            )
        )
        eval_samples = np.clip(current_sample + delta * rnd_noise, clip_min, clip_max)
        eval_samples = self._project_tabular(eval_samples, clip_min, clip_max, mask)
        rnd_noise = (eval_samples - current_sample) / delta

        satisfied = self._adversarial_satisfactory(
            samples=eval_samples, target=target, clip_min=clip_min, clip_max=clip_max
        )
        f_val = 2 * satisfied.reshape([num_eval] + [1]) - 1.0
        f_val = f_val.astype(ART_NUMPY_DTYPE)

        if np.mean(f_val) == 1.0:
            grad = np.mean(rnd_noise, axis=0)
        elif np.mean(f_val) == -1.0:
            grad = -np.mean(rnd_noise, axis=0)
        else:
            f_val -= np.mean(f_val)
            grad = np.mean(f_val * rnd_noise, axis=0)

        if self.norm == 2:
            result = grad / np.linalg.norm(grad)
        else:
            result = np.sign(grad)

        return result

    def _adversarial_satisfactory(
        self, samples: np.ndarray, target: int, clip_min: float, clip_max: float
    ) -> np.ndarray:
        samples = np.clip(samples, clip_min, clip_max)
        preds = np.argmax(self.estimator.predict(samples, batch_size=self.batch_size), axis=1)

        if self.decision_threshold is not None and self.estimator.nb_classes == 2:
            probas = self.estimator.predict(samples, batch_size=self.batch_size)
            pos_prob = probas[:, target]
            if self.targeted:
                result = pos_prob >= self.decision_threshold
            else:
                result = pos_prob < self.decision_threshold
        else:
            if self.targeted:
                result = preds == target
            else:
                result = preds != target

        return result

    @staticmethod
    def _interpolate(
        current_sample: np.ndarray, original_sample: np.ndarray, alpha: float, norm: int | float | str
    ) -> np.ndarray:
        if norm == 2:
            result = (1 - alpha) * original_sample + alpha * current_sample
        else:
            result = np.clip(current_sample, original_sample - alpha, original_sample + alpha)

        return result

    def _check_params(self) -> None:
        if self.norm not in [2, np.inf, "inf"]:
            raise ValueError('Norm order must be either 2, `np.inf` or "inf".')

        if not isinstance(self.max_iter, int) or self.max_iter < 0:
            raise ValueError("The number of iterations must be a non-negative integer.")

        if not isinstance(self.max_eval, int) or self.max_eval <= 0:
            raise ValueError("The maximum number of evaluations must be a positive integer.")

        if not isinstance(self.init_eval, int) or self.init_eval <= 0:
            raise ValueError("The initial number of evaluations must be a positive integer.")

        if self.init_eval > self.max_eval:
            raise ValueError("The maximum number of evaluations must be larger than the initial number of evaluations.")

        if not isinstance(self.init_size, int) or self.init_size <= 0:
            raise ValueError("The number of initial trials must be a positive integer.")

        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")

        # Tabular validations
        # Allow estimator.input_shape to be int or tuple; compare by total size
        expected_size = int(np.prod(self.estimator.input_shape))
        if self.feature_clip_min is not None and self.feature_clip_max is not None:
            if self.feature_clip_min.shape != self.feature_clip_max.shape:
                raise ValueError("feature_clip_min and feature_clip_max must share the same shape.")
            if self.feature_clip_min.size != expected_size:
                raise ValueError("feature_clip_min/feature_clip_max must match estimator input_size.")
        if self.editable_mask is not None and self.editable_mask.size != expected_size:
            raise ValueError("editable_mask must match estimator input_size.")
        if self.feature_importance is not None and self.feature_importance.size != expected_size:
            raise ValueError("feature_importance must match estimator input_size.")

    # ---------------------- tabular utilities ---------------------- #
    def _expand_clip_value(self, clip_value: float | np.ndarray, kind: str) -> np.ndarray:
        if kind == "min" and self.feature_clip_min is not None:
            return self.feature_clip_min.astype(ART_NUMPY_DTYPE)
        if kind == "max" and self.feature_clip_max is not None:
            return self.feature_clip_max.astype(ART_NUMPY_DTYPE)
        return np.ones(self.estimator.input_shape, dtype=ART_NUMPY_DTYPE) * clip_value

    def _project_tabular(
        self, sample: np.ndarray, clip_min: np.ndarray, clip_max: np.ndarray, mask: np.ndarray | None
    ) -> np.ndarray:
        if mask is not None:
            sample = sample * mask + (1 - mask) * np.clip(sample, clip_min, clip_max)
        sample = np.clip(sample, clip_min, clip_max)

        if self.integer_indices:
            sample[..., self.integer_indices] = np.round(sample[..., self.integer_indices])

        for group in self.categorical_groups:
            sub = sample[..., group]
            max_idx = np.argmax(sub, axis=-1)
            if sample.ndim == 1:
                sub[:] = 0
                sub[max_idx] = 1
            else:
                sub[:] = 0
                rows = np.arange(sub.shape[0])
                sub[rows, max_idx] = 1
            sample[..., group] = sub

        return sample


class HopSkipJumpTabular(HopSkipJump):
    """
    Convenience wrapper to configure HopSkipJump with tabular constraints from TabularDataset.
    """

    def __init__(self, estimator, tab_dataset, decision_threshold=None, **kwargs):
        # Build per-feature clip bounds; replace inf with empirical min/max to keep uniform sampling finite
        feature_clip_min = tab_dataset.feature_ranges[:, 0].astype(np.float32)
        feature_clip_max = tab_dataset.feature_ranges[:, 1].astype(np.float32)
        data_min = tab_dataset.X_train.min(axis=0)
        data_max = tab_dataset.X_train.max(axis=0)
        feature_clip_min = np.where(np.isfinite(feature_clip_min), feature_clip_min, data_min)
        feature_clip_max = np.where(np.isfinite(feature_clip_max), feature_clip_max, data_max)

        integer_indices = tab_dataset.ordinal_indices.tolist()
        categorical_groups = [group.tolist() for group in tab_dataset.one_hot_groups]
        editable_mask = kwargs.pop("editable_mask", None)  # optional external mask

        super().__init__(
            classifier=estimator,
            feature_clip_min=feature_clip_min,
            feature_clip_max=feature_clip_max,
            integer_indices=integer_indices,
            categorical_groups=categorical_groups,
            editable_mask=editable_mask,
            decision_threshold=decision_threshold,
            **kwargs,
        )
