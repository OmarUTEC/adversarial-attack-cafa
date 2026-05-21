import numpy as np
from sklearn.metrics import confusion_matrix, recall_score
from art.estimators import NeuralNetworkMixin

from src.attacks.cafa import CaFA


def evaluate_crafted_samples(
        X_adv: np.ndarray,
        X_orig: np.ndarray,
        y: np.ndarray,
        classifier,
        tab_dataset,
):
    y_pred = classifier.predict(X_adv).argmax(axis=1)
    is_misclassified = y_pred != y

    l0_costs = CaFA.calc_l0_cost(X_orig, X_adv)
    stand_linf_costs = CaFA.calc_standard_linf_cost(
        X_orig, X_adv,
        standard_factors=tab_dataset.standard_factors,
        relevant_indices=tab_dataset.ordinal_indices.tolist() + tab_dataset.cont_indices.tolist())

    assert len(is_misclassified) == len(l0_costs) == len(stand_linf_costs) == len(X_adv)

    # Confusion matrix & fraud-specific metrics
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    fnr    = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        # Attack success:
        'is_misclassified_rate': is_misclassified.mean(),

        # Confusion matrix:
        'confusion_matrix': {'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn)},

        # Fraud-specific metrics:
        'fnr':          fnr,
        'recall_fraud': recall,

        # Costs — L0:
        'l0_costs_mean':        l0_costs.mean(),
        'l0_costs_on_mis_mean': l0_costs[is_misclassified].mean() if is_misclassified.any() else 0.0,

        # Costs — standardized L-inf:
        'stand_linf_costs_mean':        stand_linf_costs.mean(),
        'stand_linf_costs_on_mis_mean': stand_linf_costs[is_misclassified].mean() if is_misclassified.any() else 0.0,
    }
