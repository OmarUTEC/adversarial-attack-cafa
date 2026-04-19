import numpy as np
from art.estimators import NeuralNetworkMixin

from src.attacks.white_box.cafa import CaFA


def evaluate_crafted_samples(
        X_adv: np.ndarray,
        X_orig: np.ndarray,
        y: np.ndarray,
        classifier: NeuralNetworkMixin,
        tab_dataset,
):
    """
    Evaluates the crafted adversarial samples wrp to the given classifier.
    :param X_adv: Adversarial samples crafted by the attack, in format of model's input.
    :param X_orig: Samples on which the attack was applied, in format of model's input.
    :param y: labels of the samples.
    :param classifier: the targeted classifier.
    :param tab_dataset: the tabular dataset object of the model's input data.
    :return:
    """
    # Evaluate misclassification
    is_misclassified = classifier.predict(X_adv).argmax(axis=1) != y

    # Evaluate 'cost' metrics
    l0_costs = CaFA.calc_l0_cost(X_orig, X_adv)
    stand_linf_costs = CaFA.calc_standard_linf_cost(
        X_orig, X_adv,
        standard_factors=tab_dataset.standard_factors,
        relevant_indices=tab_dataset.ordinal_indices.tolist() + tab_dataset.cont_indices.tolist())

    assert len(is_misclassified) == len(l0_costs) == len(stand_linf_costs) == len(X_adv)

    return {
        # Attack success:
        'is_misclassified_rate': is_misclassified.mean(),

        # Costs:
        #  - L0:
        'l0_costs_mean': l0_costs.mean(),
        'l0_costs_on_mis_mean': l0_costs[is_misclassified].mean(),

        #  - Standardized-linf
        'stand_linf_costs_mean': stand_linf_costs.mean(),
        'stand_linf_costs_on_mis_mean': stand_linf_costs[is_misclassified].mean(),

    }
