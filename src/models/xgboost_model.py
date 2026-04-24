import logging
import os
import pickle
from typing import Any, Dict

import torch
import xgboost as xgb

from src.datasets.load_tabular_data import TabularDataset
from src.models.metrics import compute_binary_classification_metrics

logger = logging.getLogger(__name__)


class XGBoostModel:
    def __init__(self, n_classes: int, hparams: Dict[str, Any] | None = None):
        self.n_classes = n_classes
        self.hparams = {
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "verbosity": 1,
            "random_state": 42,
            **(hparams or {}),
        }
        self.hparams.setdefault(
            "objective",
            "multi:softprob" if n_classes > 2 else "binary:logistic",
        )
        self.hparams.setdefault("eval_metric", "aucpr")
        if self.hparams.get("scale_pos_weight") == "auto":
            self.hparams.pop("scale_pos_weight")

        if n_classes > 2:
            self.hparams["num_class"] = n_classes

        self.model = xgb.XGBClassifier(**self.hparams)

    def fit(self, trainset, devset):
        x_train, y_train = trainset
        x_val, y_val = devset

        logger.info("Training XGBoost model.")
        if self.n_classes == 2 and self.hparams.get("scale_pos_weight") == "auto":
            raise ValueError("scale_pos_weight='auto' must be resolved before constructing XGBoostModel.")
        self.model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            verbose=False,
        )

        y_pred = self.model.predict(x_val)
        y_prob = self.model.predict_proba(x_val)
        if self.n_classes == 2:
            metrics = compute_binary_classification_metrics(y_val, y_prob[:, 1])
        else:
            raise ValueError("This fraud-detection pipeline currently expects binary classifiers.")
        metrics["predicted_positive_rate"] = float(y_pred.mean())
        self.last_validation_metrics = metrics
        logger.info(
            "Validation metrics: "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"f1={metrics['f1']:.4f}, "
            f"pr_auc={metrics['pr_auc']:.4f}, "
            f"roc_auc={metrics['roc_auc']:.4f}, "
            f"confusion_matrix={{tn={metrics['tn']}, fp={metrics['fp']}, "
            f"fn={metrics['fn']}, tp={metrics['tp']}}}"
        )
        return metrics

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Model saved to {path}")

    @staticmethod
    def load(path: str):
        with open(path, "rb") as f:
            return pickle.load(f)


def train(
    hyperparameters: Dict[str, Any],
    trainset: torch.utils.data.Dataset,
    testset: torch.utils.data.Dataset,
    tab_dataset: TabularDataset,
    model_artifact_path: str = None,
):
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)

    x_train, y_train = next(iter(trainloader))
    x_val, y_val = next(iter(testloader))

    resolved_hparams = dict(hyperparameters)
    if resolved_hparams.get("scale_pos_weight") == "auto":
        positives = float((y_train.numpy() == 1).sum())
        negatives = float((y_train.numpy() == 0).sum())
        resolved_hparams["scale_pos_weight"] = negatives / positives if positives > 0 else 1.0

    model = XGBoostModel(
        n_classes=tab_dataset.n_classes,
        hparams=resolved_hparams,
    )
    validation_metrics = model.fit((x_train.numpy(), y_train.numpy()), (x_val.numpy(), y_val.numpy()))

    if model_artifact_path is not None:
        model.save(model_artifact_path)

    return {
        "model_artifact_path": model_artifact_path,
        "resolved_hyperparameters": resolved_hparams,
        "validation_metrics": validation_metrics,
        "best_val_hp_metric": validation_metrics["pr_auc"],
    }


def grid_search_hyperparameters(
    trainset: torch.utils.data.Dataset,
    testset: torch.utils.data.Dataset,
    tab_dataset: TabularDataset,
):
    return {
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "verbosity": 1,
        "random_state": 42,
    }
