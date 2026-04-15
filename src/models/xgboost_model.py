import logging
import os
import pickle
from typing import Dict, Any

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score
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
            'use_label_encoder': False
        }
        if n_classes > 2:
            self.hparams['num_class'] = n_classes
            
        self.model = xgb.XGBClassifier(**self.hparams)

    def fit(self, trainset, devset):
        X_train, y_train = trainset
        X_val, y_val = devset
        
        logger.info("Training XGBoost model...")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=True
        )
        
        y_pred = self.model.predict(X_val)
        y_prob = self.model.predict_proba(X_val)
        
        acc = accuracy_score(y_val, y_pred)
        if self.n_classes == 2:
            auc = roc_auc_score(y_val, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_val, y_prob, multi_class='ovr')
            
        logger.info(f"Validation Accuracy: {acc:.4f}, AUC: {auc:.4f}")
        return auc

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Model saved to {path}")

    @staticmethod
    def load(path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)

def train(hparams: Dict[str, Any], trainset, testset, tab_dataset: TabularDataset, model_artifact_path: str):
    import torch
    
    # Extract data from Torch datasets
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)
    
    X_train, y_train = next(iter(train_loader))
    X_val, y_val = next(iter(test_loader))
    
    X_train, y_train = X_train.numpy(), y_train.numpy()
    X_val, y_val = X_val.numpy(), y_val.numpy()
    
    n_classes = tab_dataset.n_classes
    model_wrapper = XGBoostModel(n_classes=n_classes, hparams=hparams)
    model_wrapper.fit((X_train, y_train), (X_val, y_val))
    model_wrapper.save(model_artifact_path)
    return model_wrapper

def grid_search_hyperparameters(trainset, testset, tab_dataset: TabularDataset):
    # Simplified version for now, could use Optuna later
    return {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'random_state': 42
    }
