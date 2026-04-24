import json
import logging
from typing import Dict
import os
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from omegaconf import DictConfig, ListConfig, OmegaConf
import hydra
import torch
import numpy as np
from art.estimators.classification import PyTorchClassifier, XGBoostClassifier

from src.attacks.white_box.cafa import CaFA
from src.attacks.white_box.deepfool_tabular import DeepFoolAttackTabular
from src.attacks.white_box.fgm_tabular import FGMAttackTabular
from src.attacks.white_box.jsma_tabular import JSMAAttackTabular
from src.attacks.white_box.pgd_tabular import PGDAttackTabular
from src.attacks.black_box.square_attack_tabular import SquareAttackTabular
from src.attacks.black_box.simba_tabular import SimBATabular
from src.attacks.black_box.hop_skip_jump_tabular import HopSkipJumpTabular
from src.attacks.black_box.zoo_attack_tabular import ZooAttackTabularFromDataset
from src.attacks.black_box.sign_opt_tabular import SignOPTAttackTabularFromDataset
from src.attacks.black_box.boundary_attack_tabular import BoundaryAttackTabularFromDataset
from src.models.utils import load_trained_model
from src.utils import evaluate_crafted_samples
from src.datasets.load_tabular_data import TabularDataset
from src.models import mlp as mlp_models
from src.models import lstm_attention as lstm_models
from src.models import logistic_regression as logreg_models
from src.models import xgboost_model as xgboost_models

MODEL_TRAINERS = {
    'mlp': {
        'train': mlp_models.train,
        'grid_search': mlp_models.grid_search_hyperparameters,
    },
    'lstm_attention': {
        'train': lstm_models.train,
        'grid_search': lstm_models.grid_search_hyperparameters,
    },
    'logistic_regression': {
        'train': logreg_models.train,
        'grid_search': logreg_models.grid_search_hyperparameters,
    },
    'xgboost': {
        'train': xgboost_models.train,
        'grid_search': xgboost_models.grid_search_hyperparameters,
    },
}

logger = logging.getLogger(__name__)


def _art_input_shape(tab_dataset: TabularDataset) -> tuple[int]:
    return (tab_dataset.n_features,)


def _classification_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    ART may pass labels either as class indices or one-hot/probability vectors.
    Normalize both cases to class indices for cross-entropy.
    """
    if target.ndim > 1:
        target = target.argmax(dim=1)
    return torch.functional.F.cross_entropy(output, target.long())


def _get_model_module(model_type: str):
    """
    Return the module containing train/grid_search_hyperparameters for the requested model type.
    """
    if model_type == "mlp":
        return mlp_models
    if model_type == "logistic_regression":
        return logreg_models
    if model_type == "lstm_attention":
        return lstm_models
    if model_type == "xgboost":
        return xgboost_models
    raise NotImplementedError(f"Unknown model type: {model_type}")


def _default_model_path(model_type: str, dataset_name: str) -> str:
    if model_type == "xgboost":
        return f"trained-models/{dataset_name}-{model_type}.pkl"
    return f"trained-models/{dataset_name}-{model_type}.ckpt"


def _model_file_name(model_type: str) -> str:
    return "model.pkl" if model_type == "xgboost" else "model.ckpt"


def _create_training_artifact_dir(dataset_name: str, model_type: str) -> Path:
    run_id = f"{datetime.now().strftime('%Y-%m-%d-%H%M%S')}-{uuid4().hex[:8]}"
    artifact_dir = Path("trained-models") / f"{dataset_name}-{model_type}-{run_id}"
    artifact_dir.mkdir(parents=True, exist_ok=False)
    return artifact_dir


def _to_jsonable(value):
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _dataset_summary(tab_dataset: TabularDataset) -> Dict:
    summary = {
        "dataset_name": tab_dataset.data_parameters["dataset_name"],
        "n_features": tab_dataset.n_features,
        "n_classes": tab_dataset.n_classes,
        "feature_names": tab_dataset.feature_names.tolist(),
        "label_name": tab_dataset.label_name,
        "encoding_method": tab_dataset.cat_encoding_method,
        "train_shape": list(tab_dataset.X_train.shape),
        "test_shape": list(tab_dataset.X_test.shape),
        "train_positive_rate": float(tab_dataset.y_train.mean()),
        "test_positive_rate": float(tab_dataset.y_test.mean()),
    }
    if getattr(tab_dataset, "has_predefined_splits", False):
        summary["val_shape"] = list(tab_dataset.X_val.shape)
        summary["val_positive_rate"] = float(tab_dataset.y_val.mean())
    return summary


def _write_training_documentation(
    artifact_dir: Path,
    model_type: str,
    model_artifact_path: str,
    tab_dataset: TabularDataset,
    hyperparameters,
    training_results,
    cfg: DictConfig,
    hydra_output_dir: str,
) -> None:
    report = {
        "model_type": model_type,
        "model_artifact_path": model_artifact_path,
        "dataset": _dataset_summary(tab_dataset),
        "hyperparameters": _to_jsonable(hyperparameters),
        "training_results": _to_jsonable(training_results),
        "config": _to_jsonable(cfg),
        "hydra_output_dir": hydra_output_dir,
    }
    report_path = artifact_dir / "training_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    dataset_name = report["dataset"]["dataset_name"]
    results = report["training_results"] or {}
    validation_metrics = results.get("validation_metrics", {})
    if not validation_metrics and "best_val_pr_auc" in results:
        validation_metrics = {
            "precision": results.get("best_val_precision"),
            "recall": results.get("best_val_recall"),
            "f1": results.get("best_val_f1"),
            "pr_auc": results.get("best_val_pr_auc"),
            "roc_auc": results.get("best_val_roc_auc"),
            "confusion_matrix": results.get("best_val_confusion_matrix"),
        }

    lines = [
        f"# Training Artifact: {dataset_name} / {model_type}",
        "",
        f"- Model artifact: `{Path(model_artifact_path).name}`",
        f"- Dataset: `{dataset_name}`",
        f"- Features: `{report['dataset']['n_features']}`",
        f"- Classes: `{report['dataset']['n_classes']}`",
        f"- Train shape: `{report['dataset']['train_shape']}`",
        f"- Test shape: `{report['dataset']['test_shape']}`",
    ]
    if "val_shape" in report["dataset"]:
        lines.append(f"- Validation shape: `{report['dataset']['val_shape']}`")
    if validation_metrics:
        confusion_matrix = validation_metrics.get("confusion_matrix")
        if confusion_matrix is None and all(k in validation_metrics for k in ["tn", "fp", "fn", "tp"]):
            confusion_matrix = {
                "tn": validation_metrics["tn"],
                "fp": validation_metrics["fp"],
                "fn": validation_metrics["fn"],
                "tp": validation_metrics["tp"],
            }
        lines += [
            "",
            "## Validation Metrics",
            "",
            f"- Precision: `{validation_metrics.get('precision')}`",
            f"- Recall: `{validation_metrics.get('recall')}`",
            f"- F1-score: `{validation_metrics.get('f1')}`",
            f"- PR-AUC: `{validation_metrics.get('pr_auc')}`",
            f"- ROC-AUC: `{validation_metrics.get('roc_auc')}`",
            f"- Confusion matrix: `{confusion_matrix}`",
        ]
    lines += [
        "",
        "## Files",
        "",
        "- `training_report.json`: complete machine-readable training metadata.",
        f"- `{Path(model_artifact_path).name}`: serialized trained model.",
    ]
    (artifact_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_art_classifier(model, model_type: str, tab_dataset: TabularDataset):
    if model_type == "xgboost":
        classifier = XGBoostClassifier(
            model=model.model,
            nb_classes=tab_dataset.n_classes,
        )
        classifier._input_shape = _art_input_shape(tab_dataset)
        return classifier

    return PyTorchClassifier(
        model=model,
        loss=_classification_loss,
        input_shape=_art_input_shape(tab_dataset),
        nb_classes=tab_dataset.n_classes,
    )


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info(f"Used config: {OmegaConf.to_yaml(cfg)}")
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # 1. Process data:
    tab_dataset = TabularDataset(**cfg.data.params)
    trainset, devset = tab_dataset.get_train_dev_sets(dev_set_proportion=0.15)

    # 2. Load model; optionally, re-train before:
    surrogate_model_type = (
        cfg.transfer.surrogate_model
        if "transfer" in cfg and cfg.transfer and cfg.transfer.surrogate_model is not None
        else cfg.ml_model.model_type
    )
    surrogate_model_path = (
        cfg.ml_model.model_artifact_path
        if surrogate_model_type == cfg.ml_model.model_type
        else _default_model_path(surrogate_model_type, cfg.data.name)
    )
    if cfg.ml_model.perform_training or cfg.ml_model.perform_grid_search_hparams:
        model_module = _get_model_module(surrogate_model_type)
        best_hparams = cfg.ml_model.default_hparams
        if cfg.ml_model.perform_grid_search_hparams:
            best_hparams = model_module.grid_search_hyperparameters(trainset=trainset,
                                                                    testset=devset,
                                                                    tab_dataset=tab_dataset)
        artifact_dir = _create_training_artifact_dir(cfg.data.name, surrogate_model_type)
        surrogate_model_path = str(artifact_dir / _model_file_name(surrogate_model_type))
        training_results = model_module.train(
            best_hparams,
            trainset=trainset,
            testset=devset,
            tab_dataset=tab_dataset,
            model_artifact_path=surrogate_model_path,
        )
        _write_training_documentation(
            artifact_dir=artifact_dir,
            model_type=surrogate_model_type,
            model_artifact_path=surrogate_model_path,
            tab_dataset=tab_dataset,
            hyperparameters=best_hparams,
            training_results=training_results,
            cfg=cfg,
            hydra_output_dir=output_dir,
        )
        logger.info(f"Training artifact saved in {artifact_dir}")
    model = load_trained_model(surrogate_model_path, model_type=surrogate_model_type)

    # 3. Wrap the model to ART classifier, for executing the attack:
    classifier = _build_art_classifier(model, surrogate_model_type, tab_dataset)
    eval_params = dict(classifier=classifier, tab_dataset=tab_dataset)

    # 4. Evaluate before the attack:
    X, y = tab_dataset.X_test[:cfg.n_samples_to_attack], tab_dataset.y_test[:cfg.n_samples_to_attack]
    if cfg.data_split_to_attack == 'train':
        X, y = tab_dataset.X_train[:cfg.n_samples_to_attack], tab_dataset.y_train[:cfg.n_samples_to_attack]
    evaluations: Dict[str, Dict[str, float]] = {}

    evaluations['before-attack'] = evaluate_crafted_samples(X_adv=X, X_orig=X, y=y, **eval_params)
    if "transfer" in cfg and cfg.transfer and getattr(cfg.transfer, "target_models", []):
        for target_model_type in cfg.transfer.target_models:
            target_path = _default_model_path(target_model_type, cfg.data.name)
            target_model = load_trained_model(target_path, model_type=target_model_type)
            target_clf = _build_art_classifier(target_model, target_model_type, tab_dataset)
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
        elif attack_name == "square_attack":
            logger.info("Executing SquareAttack (tabular) attack.")
            attack_params = dict(cfg.attack)
            for key in ["feature_clip_min", "feature_clip_max", "integer_indices", "categorical_groups", "editable_mask", "attack_name"]:
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
        elif attack_name == "hop_skip_jump":
            logger.info("Executing HopSkipJump (tabular) attack.")
            attack_params = dict(cfg.attack)
            for key in ["feature_clip_min", "feature_clip_max", "integer_indices", "categorical_groups",
                        "editable_mask", "decision_threshold", "attack_name"]:
                attack_params.pop(key, None)
            attack = HopSkipJumpTabular(
                estimator=classifier,
                tab_dataset=tab_dataset,
                **attack_params,
            )
            X_adv = attack.generate(x=X, y=y)
            evaluations['after-hop-skip-jump'] = evaluate_crafted_samples(X_adv=X_adv, X_orig=X, y=y, **eval_params)
            logger.info(f"after-hop-skip-jump: {evaluations['after-hop-skip-jump']}")
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
        elif attack_name == "zoo_attack":
            logger.info("Executing ZOO (tabular) attack.")
            attack_params = dict(cfg.attack)
            for key in [
                "feature_clip_min",
                "feature_clip_max",
                "integer_indices",
                "categorical_groups",
                "editable_mask",
                "only_increase_mask",
                "only_decrease_mask",
                "max_abs_step",
                "attack_name",
            ]:
                attack_params.pop(key, None)
            attack = ZooAttackTabularFromDataset(
                classifier=classifier,
                tab_dataset=tab_dataset,
                **attack_params,
            )
            X_adv = attack.generate(x=X, y=y)
            evaluations['after-zoo-attack'] = evaluate_crafted_samples(X_adv=X_adv, X_orig=X, y=y, **eval_params)
            logger.info(f"after-zoo-attack: {evaluations['after-zoo-attack']}")
        elif attack_name == "sign_opt":
            logger.info("Executing SignOPT (tabular) attack.")
            attack_params = dict(cfg.attack)
            for key in [
                "feature_clip_min",
                "feature_clip_max",
                "integer_indices",
                "categorical_groups",
                "editable_mask",
                "only_increase_mask",
                "only_decrease_mask",
                "max_abs_step",
                "attack_name",
            ]:
                attack_params.pop(key, None)
            attack = SignOPTAttackTabularFromDataset(
                estimator=classifier,
                tab_dataset=tab_dataset,
                **attack_params,
            )
            X_adv = attack.generate(x=X, y=y)
            evaluations['after-sign-opt'] = evaluate_crafted_samples(X_adv=X_adv, X_orig=X, y=y, **eval_params)
            logger.info(f"after-sign-opt: {evaluations['after-sign-opt']}")
        elif attack_name == "boundary_attack":
            logger.info("Executing BoundaryAttack (tabular) attack.")
            attack_params = dict(cfg.attack)
            for key in [
                "feature_clip_min",
                "feature_clip_max",
                "integer_indices",
                "categorical_groups",
                "editable_mask",
                "only_increase_mask",
                "only_decrease_mask",
                "max_abs_step",
                "attack_name",
            ]:
                attack_params.pop(key, None)
            attack = BoundaryAttackTabularFromDataset(
                estimator=classifier,
                tab_dataset=tab_dataset,
                **attack_params,
            )
            X_adv = attack.generate(x=X, y=y)
            evaluations['after-boundary-attack'] = evaluate_crafted_samples(X_adv=X_adv, X_orig=X, y=y, **eval_params)
            logger.info(f"after-boundary-attack: {evaluations['after-boundary-attack']}")
        elif attack_name == "pgd":
            logger.info("Executing PGD (tabular) attack.")
            attack_params = dict(cfg.attack)
            for key in [
                "feature_clip_min",
                "feature_clip_max",
                "integer_indices",
                "categorical_groups",
                "editable_mask",
                "only_increase_mask",
                "only_decrease_mask",
                "max_abs_step",
                "attack_name",
            ]:
                attack_params.pop(key, None)
            attack = PGDAttackTabular(
                estimator=classifier,
                x_reference=X,
                feature_clip_min=tab_dataset.feature_ranges[:, 0].astype(np.float32),
                feature_clip_max=tab_dataset.feature_ranges[:, 1].astype(np.float32),
                integer_indices=tab_dataset.ordinal_indices.tolist(),
                categorical_groups=[grp.tolist() for grp in tab_dataset.one_hot_groups],
                **attack_params,
            )
            X_adv = attack.generate(x=X, y=y)
            evaluations["after-pgd"] = evaluate_crafted_samples(X_adv=X_adv, X_orig=X, y=y, **eval_params)
            logger.info(f"after-pgd: {evaluations['after-pgd']}")
        elif attack_name == "fgm":
            logger.info("Executing FGM/FGSM (tabular) attack.")
            attack_params = dict(cfg.attack)
            for key in [
                "feature_clip_min",
                "feature_clip_max",
                "integer_indices",
                "categorical_groups",
                "editable_mask",
                "only_increase_mask",
                "only_decrease_mask",
                "max_abs_step",
                "attack_name",
            ]:
                attack_params.pop(key, None)
            attack = FGMAttackTabular(
                estimator=classifier,
                x_reference=X,
                feature_clip_min=tab_dataset.feature_ranges[:, 0].astype(np.float32),
                feature_clip_max=tab_dataset.feature_ranges[:, 1].astype(np.float32),
                integer_indices=tab_dataset.ordinal_indices.tolist(),
                categorical_groups=[grp.tolist() for grp in tab_dataset.one_hot_groups],
                **attack_params,
            )
            X_adv = attack.generate(x=X, y=y)
            evaluations["after-fgm"] = evaluate_crafted_samples(X_adv=X_adv, X_orig=X, y=y, **eval_params)
            logger.info(f"after-fgm: {evaluations['after-fgm']}")
        elif attack_name == "deepfool":
            logger.info("Executing DeepFool (tabular) attack.")
            attack_params = dict(cfg.attack)
            for key in [
                "feature_clip_min",
                "feature_clip_max",
                "integer_indices",
                "categorical_groups",
                "editable_mask",
                "only_increase_mask",
                "only_decrease_mask",
                "max_abs_step",
                "attack_name",
            ]:
                attack_params.pop(key, None)
            attack = DeepFoolAttackTabular(
                classifier=classifier,
                x_reference=X,
                feature_clip_min=tab_dataset.feature_ranges[:, 0].astype(np.float32),
                feature_clip_max=tab_dataset.feature_ranges[:, 1].astype(np.float32),
                integer_indices=tab_dataset.ordinal_indices.tolist(),
                categorical_groups=[grp.tolist() for grp in tab_dataset.one_hot_groups],
                **attack_params,
            )
            X_adv = attack.generate(x=X, y=y)
            evaluations["after-deepfool"] = evaluate_crafted_samples(X_adv=X_adv, X_orig=X, y=y, **eval_params)
            logger.info(f"after-deepfool: {evaluations['after-deepfool']}")
        elif attack_name == "jsma":
            logger.info("Executing JSMA (tabular) attack.")
            attack_params = dict(cfg.attack)
            for key in [
                "feature_clip_min",
                "feature_clip_max",
                "integer_indices",
                "categorical_groups",
                "editable_mask",
                "only_increase_mask",
                "only_decrease_mask",
                "max_abs_step",
                "attack_name",
            ]:
                attack_params.pop(key, None)
            attack = JSMAAttackTabular(
                classifier=classifier,
                x_reference=X,
                feature_clip_min=tab_dataset.feature_ranges[:, 0].astype(np.float32),
                feature_clip_max=tab_dataset.feature_ranges[:, 1].astype(np.float32),
                integer_indices=tab_dataset.ordinal_indices.tolist(),
                categorical_groups=[grp.tolist() for grp in tab_dataset.one_hot_groups],
                **attack_params,
            )
            X_adv = attack.generate(x=X, y=y)
            evaluations["after-jsma"] = evaluate_crafted_samples(X_adv=X_adv, X_orig=X, y=y, **eval_params)
            logger.info(f"after-jsma: {evaluations['after-jsma']}")
        else:
            raise ValueError(f"Unsupported attack type: {attack_name}")

        np.save(os.path.join(output_dir, "X_adv.npy"), X_adv)
        if "transfer" in cfg and cfg.transfer and getattr(cfg.transfer, "target_models", []):
            for target_model_type in cfg.transfer.target_models:
                target_path = _default_model_path(target_model_type, cfg.data.name)
                target_model = load_trained_model(target_path, model_type=target_model_type)
                target_clf = _build_art_classifier(target_model, target_model_type, tab_dataset)
                eval_params_target = dict(eval_params)
                eval_params_target["classifier"] = target_clf
                eval_key = f"after-{attack_name}-transfer-{target_model_type}"
                evaluations[eval_key] = evaluate_crafted_samples(X_adv=X_adv, X_orig=X, y=y, **eval_params_target)
                logger.info(f"{eval_key}: {evaluations[eval_key]}")

    # 5. Log & save evaluations:
    logger.info(f"Evaluations: {evaluations}")
    with open(os.path.join(output_dir, "evaluations.json"), "w") as f:
        json.dump(evaluations, f, indent=4)
    logger.info(f"Finished run. results saved in {output_dir}")

if __name__ == "__main__":
    main()
