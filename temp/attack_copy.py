import json
import logging
from typing import Dict
import os

from omegaconf import DictConfig, OmegaConf
import hydra
import torch
import numpy as np
from art.estimators.classification import PyTorchClassifier
from tqdm import tqdm

from src.attacks.cafa import CaFA
from src.attacks.hop_skip_jump_tabular import HopSkipJumpTabular
from src.attacks.square_attack_tabular import SquareAttackTabular
from src.attacks.simba_tabular import SimBATabular
from src.models.utils import load_trained_model
from src.utils import evaluate_crafted_samples
from src.datasets.load_tabular_data import TabularDataset

logger = logging.getLogger(__name__)


def _get_model_module(model_type: str):
    """
    Return the module containing train/grid_search_hyperparameters for the requested model type.
    """
    if model_type == "mlp":
        from src.models import mlp as module
    elif model_type == "logistic_regression":
        from src.models import logistic_regression as module
    elif model_type == "lstm_attention":
        from src.models import lstm_attention as module
    else:
        raise NotImplementedError(f"Unknown model type: {model_type}")
    return module


def _default_model_path(model_type: str, dataset_name: str) -> str:
    return f"trained-models/{dataset_name}-{model_type}.ckpt"


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info(f"Used config: {OmegaConf.to_yaml(cfg)}")
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # 1. Process data:
    tab_dataset = TabularDataset(**cfg.data.params)
    trainset, devset = tab_dataset.get_train_dev_sets(dev_set_proportion=0.15)

    # 2. Load model; optionally, re-train before:
    surrogate_model_type = cfg.transfer.surrogate_model if "transfer" in cfg and cfg.transfer else cfg.ml_model.model_type
    model_module = _get_model_module(surrogate_model_type)
    if cfg.ml_model.perform_training or cfg.ml_model.perform_grid_search_hparams:
        best_hparams = cfg.ml_model.default_hparams
        if cfg.ml_model.perform_grid_search_hparams:
            best_hparams = model_module.grid_search_hyperparameters(trainset=trainset,
                                                                    testset=devset,
                                                                    tab_dataset=tab_dataset)
        model_module.train(best_hparams, trainset=trainset, testset=devset, tab_dataset=tab_dataset,
                           model_artifact_path=cfg.ml_model.model_artifact_path)

    surrogate_model_path = cfg.ml_model.model_artifact_path if surrogate_model_type == cfg.ml_model.model_type else _default_model_path(surrogate_model_type, cfg.data.name)
    model = load_trained_model(surrogate_model_path, model_type=surrogate_model_type)

    # 3. Wrap the model to ART classifier, for executing the attack:
    classifier = PyTorchClassifier(
        model=model,
        loss=lambda output, target: torch.functional.F.cross_entropy(output, target.long()),
        input_shape=tab_dataset.n_features,
        nb_classes=tab_dataset.n_classes,
    )
    eval_params = dict(classifier=classifier, tab_dataset=tab_dataset)

    # 4. Evaluate before the attack:
    X, y = tab_dataset.X_test[:cfg.n_samples_to_attack], tab_dataset.y_test[:cfg.n_samples_to_attack]
    if cfg.data_split_to_attack == 'train':
        X, y = tab_dataset.X_train[:cfg.n_samples_to_attack], tab_dataset.y_train[:cfg.n_samples_to_attack]
    evaluations: Dict[str, Dict[str, float]] = {}

    evaluations['before-attack'] = evaluate_crafted_samples(X_adv=X, X_orig=X, y=y, **eval_params)
    
    # Before-attack evaluations on transfer target models (if any)
    if "transfer" in cfg and cfg.transfer and getattr(cfg.transfer, "target_models", []):
        for target_model_type in cfg.transfer.target_models:
            target_path = _default_model_path(target_model_type, cfg.data.name)
            target_model = load_trained_model(target_path, model_type=target_model_type)
            target_clf = PyTorchClassifier(
                model=target_model,
                loss=lambda output, target: torch.functional.F.cross_entropy(output, target.long()),
                input_shape=tab_dataset.n_features,
                nb_classes=tab_dataset.n_classes,
            )
            eval_params_target = dict(eval_params)
            eval_params_target["classifier"] = target_clf
            eval_key = f"before-attack-transfer-{target_model_type}"
            evaluations[eval_key] = evaluate_crafted_samples(X_adv=X, X_orig=X, y=y, **eval_params_target)
            logger.info(f"{eval_key}: {evaluations[eval_key]}")
    np.save(os.path.join(output_dir, "X.npy"), X)
    np.save(os.path.join(output_dir, "Y.npy"), y)
    logger.info(f"before-attack: {evaluations['before-attack']}")

    # 4. Attack:
    X_adv = None
    if cfg.perform_attack:
        attack_name = getattr(cfg.attack, "attack_name", "cafa")
        if attack_name == "cafa":
            logger.info("Executing CaFA attack.")
            attack = CaFA(estimator=classifier,
                          **tab_dataset.structure_constraints,
                          **cfg.attack)
            X_adv = attack.generate(x=X, y=y)
            evaluations['after-cafa'] = evaluate_crafted_samples(X_adv=X_adv, X_orig=X, y=y, **eval_params)
            logger.info(f"after-cafa: {evaluations['after-cafa']}")
        elif attack_name == "hop_skip_jump":
            logger.info("Executing HopSkipJump attack.")
            attack = HopSkipJumpTabular(
                estimator=classifier,
                tab_dataset=tab_dataset,
                **cfg.attack,
            )
            X_adv = attack.generate(x=X, y=y)
            evaluations['after-hop-skip-jump'] = evaluate_crafted_samples(X_adv=X_adv, X_orig=X, y=y, **eval_params)
            logger.info(f"after-hop-skip-jump: {evaluations['after-hop-skip-jump']}")
        elif attack_name == "square_attack":
            logger.info("Executing SquareAttack (tabular) attack.")
            attack_params = dict(cfg.attack)
            # Remove projection params coming from config to avoid duplicates; we pass actual values below.
            for key in ["feature_clip_min", "feature_clip_max", "integer_indices", "categorical_groups", "editable_mask"]:
                attack_params.pop(key, None)
            attack = SquareAttackTabular(
                estimator=classifier,
                feature_clip_min=tab_dataset.feature_ranges[:, 0].astype(np.float32),
                feature_clip_max=tab_dataset.feature_ranges[:, 1].astype(np.float32),
                integer_indices=tab_dataset.ordinal_indices.tolist(),
                categorical_groups=[grp.tolist() for grp in tab_dataset.one_hot_groups],
                **attack_params,
            )
            X_adv = attack.generate(x=X, y=y)
            evaluations['after-square-attack'] = evaluate_crafted_samples(X_adv=X_adv, X_orig=X, y=y, **eval_params)
            logger.info(f"after-square-attack: {evaluations['after-square-attack']}")
        elif attack_name == "simba":
            logger.info("Executing SimBA (tabular) attack.")
            attack_params = dict(cfg.attack)
            for key in ["feature_clip_min", "feature_clip_max", "integer_indices", "categorical_groups", "editable_mask", "attack_name"]:
                attack_params.pop(key, None)
            attack = SimBATabular(
                classifier=classifier,
                feature_clip_min=tab_dataset.feature_ranges[:, 0].astype(np.float32),
                feature_clip_max=tab_dataset.feature_ranges[:, 1].astype(np.float32),
                integer_indices=tab_dataset.ordinal_indices.tolist(),
                categorical_groups=[grp.tolist() for grp in tab_dataset.one_hot_groups],
                **attack_params,
            )
            X_adv = attack.generate(x=X, y=y)
            evaluations['after-simba'] = evaluate_crafted_samples(X_adv=X_adv, X_orig=X, y=y, **eval_params)
            logger.info(f"after-simba: {evaluations['after-simba']}")
        else:
            raise ValueError(f"Unsupported attack type: {attack_name}")

        np.save(os.path.join(output_dir, "X_adv.npy"), X_adv)
        
        # Transfer evaluation on additional target models if configured
        if "transfer" in cfg and cfg.transfer and getattr(cfg.transfer, "target_models", []):
            for target_model_type in cfg.transfer.target_models:
                
                # Carga el modelo objetivo
                target_path = _default_model_path(target_model_type, cfg.data.name)
                target_model = load_trained_model(target_path, model_type=target_model_type)
                
                # Envuelve el modelo objetivo en un clasificador ART
                target_clf = PyTorchClassifier(
                    model=target_model,
                    loss=lambda output, target: torch.functional.F.cross_entropy(output, target.long()),
                    input_shape=tab_dataset.n_features,
                    nb_classes=tab_dataset.n_classes,
                )

                # Prepara los parámetros de evaluación para el modelo objetivo                    
                eval_params_target = dict(eval_params)
                eval_params_target["classifier"] = target_clf
                eval_key = f"after-{attack_name}-transfer-{target_model_type}"

                # Realiza la evaluación de las muestras adversariales en el modelo objetivo
                evaluations[eval_key] = evaluate_crafted_samples(X_adv=X_adv, X_orig=X, y=y, **eval_params_target)

                # Registra los resultados de la evaluación
                logger.info(f"{eval_key}: {evaluations[eval_key]}")

    # 5. Log & save evaluations:
    logger.info(f"Evaluations: {evaluations}")
    with open(os.path.join(output_dir, "evaluations.json"), "w") as f:
        json.dump(evaluations, f, indent=4)
    logger.info(f"Finished run. results saved in {output_dir}")


if __name__ == "__main__":
    main()
