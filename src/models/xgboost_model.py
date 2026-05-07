import logging
import os
import pickle
from typing import Dict, Any

import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, recall_score
from src.datasets.load_tabular_data import TabularDataset

logger = logging.getLogger(__name__)

class XGBoostModel:
    def __init__(self, n_classes: int, hparams: Dict[str, Any] = None):
        self.n_classes = n_classes
        self.hparams = hparams or {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'objective': 'multi:softprob' if n_classes > 2 else 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': 42,
        }
        if n_classes > 2:
            self.hparams['num_class'] = n_classes
            
        self.model = xgb.XGBClassifier(**self.hparams)

    def fit(self, trainset, devset, early_stopping_rounds: int = 10):  # early_stopping_rounds unused (moved to constructor)
        X_train, y_train = trainset
        X_val, y_val = devset

        logger.info("Training XGBoost model...")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        
        y_pred = self.model.predict(X_val)
        y_prob = self.model.predict_proba(X_val)

        acc = accuracy_score(y_val, y_pred)
        if self.n_classes == 2:
            auc    = roc_auc_score(y_val, y_prob[:, 1])
            auc_pr = average_precision_score(y_val, y_prob[:, 1])
            f1     = f1_score(y_val, y_pred, pos_label=1, zero_division=0)
            recall = recall_score(y_val, y_pred, pos_label=1, zero_division=0)
            logger.info(f"Val — Accuracy: {acc:.4f}  AUC-ROC: {auc:.4f}  AUC-PR: {auc_pr:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")
        else:
            auc    = roc_auc_score(y_val, y_prob, multi_class='ovr')
            auc_pr = auc
            logger.info(f"Val — Accuracy: {acc:.4f}  AUC-ROC: {auc:.4f}")
        return auc_pr

    @staticmethod
    def _base_path(path: str) -> str:
        return path[:-4] if path.endswith('.pkl') else path

    def save(self, path: str):
        base = self._base_path(path)
        os.makedirs(os.path.dirname(base) or '.', exist_ok=True)
        self.model.save_model(base + '.json')
        with open(base + '.meta.pkl', 'wb') as f:
            pickle.dump({'n_classes': self.n_classes, 'hparams': self.hparams}, f)
        logger.info(f"Model saved to {base}.json")

    @staticmethod
    def load(path: str):
        base = XGBoostModel._base_path(path)
        try:
            with open(base + '.meta.pkl', 'rb') as f:
                meta = pickle.load(f)
        except FileNotFoundError:
            with open(path, 'rb') as f:
                return pickle.load(f)
        wrapper = XGBoostModel(n_classes=meta['n_classes'], hparams=meta['hparams'])
        wrapper.model.load_model(base + '.json')
        return wrapper

def train(hparams: Dict[str, Any], trainset, testset, tab_dataset: TabularDataset, model_artifact_path: str):
    import torch

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)
    test_loader  = torch.utils.data.DataLoader(testset,  batch_size=len(testset),  shuffle=False)

    X_train, y_train = next(iter(train_loader))
    X_val,   y_val   = next(iter(test_loader))
    X_train, y_train = X_train.numpy(), y_train.numpy()
    X_val,   y_val   = X_val.numpy(),   y_val.numpy()

    hparams_with_weight = dict(hparams)
    hparams_with_weight['scale_pos_weight'] = tab_dataset.class_weight_minority

    n_classes = tab_dataset.n_classes
    model_wrapper = XGBoostModel(n_classes=n_classes, hparams=hparams_with_weight)
    model_wrapper.fit((X_train, y_train), (X_val, y_val))
    model_wrapper.save(model_artifact_path)
    return model_wrapper

def grid_search_hyperparameters(trainset, testset, tab_dataset: TabularDataset):
    import torch

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)
    test_loader  = torch.utils.data.DataLoader(testset,  batch_size=len(testset),  shuffle=False)
    X_tr, y_tr = next(iter(train_loader))
    X_val, y_val = next(iter(test_loader))
    X_tr,  y_tr  = X_tr.numpy(),  y_tr.numpy()
    X_val, y_val = X_val.numpy(), y_val.numpy()

    weight = tab_dataset.class_weight_minority

    def objective(trial):
        params = {
            'max_depth':        trial.suggest_int('max_depth', 3, 10),
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators':     trial.suggest_int('n_estimators', 100, 600),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'objective':        'binary:logistic',
            'eval_metric':      'aucpr',
            'scale_pos_weight': weight,
            'random_state':     42,
            'verbosity':        0,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        y_prob = model.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, y_prob)

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=30, timeout=900, show_progress_bar=True)

    best = study.best_trial.params
    logger.info(f"XGBoost HPO done. Best AUC-PR: {study.best_value:.4f}  params: {best}")
    return best
