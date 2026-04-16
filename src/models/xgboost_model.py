import logging
import os
import pickle
from typing import Any, Dict

import torch
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score

from src.datasets.load_tabular_data import TabularDataset

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
        self.hparams.setdefault("eval_metric", "auc")
        if n_classes > 2:
            self.hparams["num_class"] = n_classes

        self.model = xgb.XGBClassifier(**self.hparams)

    def fit(self, trainset, devset):
        x_train, y_train = trainset
        x_val, y_val = devset

        logger.info("Training XGBoost model.")
        self.model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            verbose=False,
        )

        y_pred = self.model.predict(x_val)
        y_prob = self.model.predict_proba(x_val)
        acc = accuracy_score(y_val, y_pred)
        if self.n_classes == 2:
            auc = roc_auc_score(y_val, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_val, y_prob, multi_class="ovr")
        logger.info(f"Validation Accuracy: {acc:.4f}, AUC: {auc:.4f}")
        return auc

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

    model = XGBoostModel(
        n_classes=tab_dataset.n_classes,
        hparams=hyperparameters,
    )
    model.fit((x_train.numpy(), y_train.numpy()), (x_val.numpy(), y_val.numpy()))

    if model_artifact_path is not None:
        model.save(model_artifact_path)

    return model


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
