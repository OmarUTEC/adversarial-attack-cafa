from __future__ import annotations

import numpy as np


def project_tabular_constraints(
    x_adv: np.ndarray,
    x_orig: np.ndarray,
    feature_clip_min: np.ndarray,
    feature_clip_max: np.ndarray,
    integer_indices: list[int] | None = None,
    categorical_groups: list[list[int]] | None = None,
    editable_mask: np.ndarray | None = None,
    only_increase_mask: np.ndarray | None = None,
    only_decrease_mask: np.ndarray | None = None,
    max_abs_step: np.ndarray | None = None,
) -> np.ndarray:
    """
    Project perturbed tabular samples back to a feasible domain.

    Supported constraints:
    - per-feature clipping
    - non-editable features
    - monotonic changes (only increase / only decrease)
    - per-feature maximum absolute change relative to x_orig
    - integer rounding
    - one-hot categorical projection
    """
    x_proj = np.array(x_adv, copy=True)
    x_orig = np.array(x_orig, copy=False)

    integer_indices = integer_indices or []
    categorical_groups = categorical_groups or []

    x_proj = np.clip(x_proj, feature_clip_min, feature_clip_max)

    if editable_mask is not None:
        x_proj = x_proj * editable_mask + x_orig * (1.0 - editable_mask)

    if only_increase_mask is not None:
        inc_mask = only_increase_mask.astype(bool)
        x_proj[..., inc_mask] = np.maximum(x_proj[..., inc_mask], x_orig[..., inc_mask])

    if only_decrease_mask is not None:
        dec_mask = only_decrease_mask.astype(bool)
        x_proj[..., dec_mask] = np.minimum(x_proj[..., dec_mask], x_orig[..., dec_mask])

    if max_abs_step is not None:
        lower = x_orig - max_abs_step
        upper = x_orig + max_abs_step
        x_proj = np.clip(x_proj, lower, upper)
        x_proj = np.clip(x_proj, feature_clip_min, feature_clip_max)

    if integer_indices:
        x_proj[..., integer_indices] = np.round(x_proj[..., integer_indices])

    for group in categorical_groups:
        sub = x_proj[..., group]
        max_idx = np.argmax(sub, axis=-1)
        if x_proj.ndim == 1:
            sub[:] = 0
            sub[max_idx] = 1
        else:
            sub[:] = 0
            rows = np.arange(sub.shape[0])
            sub[rows, max_idx] = 1
        x_proj[..., group] = sub

    return x_proj
