import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict
from uuid import uuid4

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

from src.datasets.load_tabular_data import TabularDataset
from src.models import logistic_regression as logreg_models
from src.models import lstm_attention as lstm_models
from src.models import mlp as mlp_models
from src.models import xgboost_model as xgboost_models
from src.models.metrics import compute_binary_classification_metrics, select_binary_threshold
from src.models.utils import load_trained_model

logger = logging.getLogger(__name__)

MODEL_TRAINERS = {
    "logistic_regression": logreg_models,
    "lstm_attention": lstm_models,
    "mlp": mlp_models,
    "xgboost": xgboost_models,
}


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
        "train_positive_count": int(tab_dataset.y_train.sum()),
        "train_negative_count": int((tab_dataset.y_train == 0).sum()),
        "train_positive_rate": float(tab_dataset.y_train.mean()),
        "test_shape": list(tab_dataset.X_test.shape),
        "test_positive_count": int(tab_dataset.y_test.sum()),
        "test_negative_count": int((tab_dataset.y_test == 0).sum()),
        "test_positive_rate": float(tab_dataset.y_test.mean()),
    }
    if getattr(tab_dataset, "has_predefined_splits", False):
        summary["val_shape"] = list(tab_dataset.X_val.shape)
        summary["val_positive_count"] = int(tab_dataset.y_val.sum())
        summary["val_negative_count"] = int((tab_dataset.y_val == 0).sum())
        summary["val_positive_rate"] = float(tab_dataset.y_val.mean())
    return summary


def _predict_scores(model, model_type: str, x: np.ndarray, batch_size: int = 65536) -> np.ndarray:
    if model_type == "xgboost":
        return model.model.predict_proba(x)[:, 1]

    scores = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            x_batch = torch.tensor(x[start:start + batch_size], dtype=torch.float32)
            logits = model(x_batch)
            if logits.ndim == 1 or logits.shape[1] == 1:
                y_score = torch.sigmoid(logits.reshape(-1))
            else:
                y_score = torch.softmax(logits, dim=1)[:, 1]
            scores.append(y_score.detach().cpu().numpy())
    return np.concatenate(scores)


def _build_readme(report: Dict) -> str:
    dataset = report["dataset"]
    val_metrics = report["validation_metrics"]
    test_metrics = report["test_metrics"]
    return "\n".join([
        f"# Training Artifact: {dataset['dataset_name']} / {report['model_type']}",
        "",
        f"- Model artifact: `{Path(report['model_artifact_path']).name}`",
        f"- Dataset: `{dataset['dataset_name']}`",
        f"- Features: `{dataset['n_features']}`",
        f"- Train records: `{dataset['train_shape'][0]}`",
        f"- Validation records: `{dataset.get('val_shape', ['n/a'])[0]}`",
        f"- Test records: `{dataset['test_shape'][0]}`",
        f"- Selected threshold: `{report['selected_threshold']}`",
        f"- Threshold strategy: `{report['threshold_selection'].get('threshold_strategy')}`",
        f"- Total elapsed seconds: `{report['elapsed_seconds']['total']}`",
        "",
        "## Validation Metrics",
        "",
        f"- Precision: `{val_metrics['precision']}`",
        f"- Recall: `{val_metrics['recall']}`",
        f"- F1-score: `{val_metrics['f1']}`",
        f"- PR-AUC: `{val_metrics['pr_auc']}`",
        f"- ROC-AUC: `{val_metrics['roc_auc']}`",
        f"- Confusion matrix: `tn={val_metrics['tn']}, fp={val_metrics['fp']}, fn={val_metrics['fn']}, tp={val_metrics['tp']}`",
        "",
        "## Test Metrics",
        "",
        f"- Precision: `{test_metrics['precision']}`",
        f"- Recall: `{test_metrics['recall']}`",
        f"- F1-score: `{test_metrics['f1']}`",
        f"- PR-AUC: `{test_metrics['pr_auc']}`",
        f"- ROC-AUC: `{test_metrics['roc_auc']}`",
        f"- Confusion matrix: `tn={test_metrics['tn']}, fp={test_metrics['fp']}, fn={test_metrics['fn']}, tp={test_metrics['tp']}`",
        "",
        "## Files",
        "",
        "- `training_report.json`: complete machine-readable training metadata.",
        f"- `{Path(report['model_artifact_path']).name}`: serialized trained model.",
        "",
    ])


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_started_at = datetime.now()
    run_start_time = time.perf_counter()
    logger.info(f"Used config: {OmegaConf.to_yaml(cfg)}")
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    tab_dataset = TabularDataset(**cfg.data.params)
    trainset, devset = tab_dataset.get_train_dev_sets(dev_set_proportion=0.15)

    model_type = cfg.ml_model.model_type
    if model_type not in MODEL_TRAINERS:
        raise NotImplementedError(f"Unknown model type: {model_type}")

    model_module = MODEL_TRAINERS[model_type]
    hyperparameters = cfg.ml_model.default_hparams
    if cfg.ml_model.perform_grid_search_hparams:
        hyperparameters = model_module.grid_search_hyperparameters(
            trainset=trainset,
            testset=devset,
            tab_dataset=tab_dataset,
        )

    artifact_dir = _create_training_artifact_dir(cfg.data.name, model_type)
    model_artifact_path = str(artifact_dir / _model_file_name(model_type))
    training_start_time = time.perf_counter()
    training_results = model_module.train(
        hyperparameters,
        trainset=trainset,
        testset=devset,
        tab_dataset=tab_dataset,
        model_artifact_path=model_artifact_path,
    )
    training_elapsed_seconds = time.perf_counter() - training_start_time

    model = load_trained_model(model_artifact_path, model_type=model_type)

    evaluation_start_time = time.perf_counter()
    x_val = tab_dataset.X_val if getattr(tab_dataset, "has_predefined_splits", False) else tab_dataset.X_test
    y_val = tab_dataset.y_val if getattr(tab_dataset, "has_predefined_splits", False) else tab_dataset.y_test
    val_scores = _predict_scores(model, model_type, x_val)
    test_scores = _predict_scores(model, model_type, tab_dataset.X_test)

    threshold_cfg = cfg.get("threshold_selection", {})
    strategy = threshold_cfg.get("strategy", "max_f1")
    min_precision = threshold_cfg.get("min_precision")
    min_recall = threshold_cfg.get("min_recall")

    threshold_selection = select_binary_threshold(
        y_val,
        val_scores,
        strategy=strategy,
        min_precision=min_precision,
        min_recall=min_recall,
    )
    selected_threshold = float(threshold_selection["threshold"])
    validation_metrics = compute_binary_classification_metrics(y_val, val_scores, threshold=selected_threshold)
    test_metrics = compute_binary_classification_metrics(tab_dataset.y_test, test_scores, threshold=selected_threshold)
    evaluation_elapsed_seconds = time.perf_counter() - evaluation_start_time
    total_elapsed_seconds = time.perf_counter() - run_start_time

    report = {
        "started_at": run_started_at.isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": {
            "training": training_elapsed_seconds,
            "evaluation": evaluation_elapsed_seconds,
            "total": total_elapsed_seconds,
        },
        "model_type": model_type,
        "model_artifact_path": model_artifact_path,
        "dataset": _dataset_summary(tab_dataset),
        "hyperparameters": _to_jsonable(hyperparameters),
        "training_results": _to_jsonable(training_results),
        "threshold_selection": _to_jsonable(threshold_selection),
        "selected_threshold": selected_threshold,
        "validation_metrics": _to_jsonable(validation_metrics),
        "test_metrics": _to_jsonable(test_metrics),
        "config": _to_jsonable(cfg),
        "hydra_output_dir": output_dir,
    }

    with (artifact_dir / "training_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    (artifact_dir / "README.md").write_text(_build_readme(report), encoding="utf-8")

    logger.info(f"Training artifact saved in {artifact_dir}")
    logger.info(f"Selected threshold: {selected_threshold:.8f}")
    logger.info(f"Validation metrics: {validation_metrics}")
    logger.info(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
