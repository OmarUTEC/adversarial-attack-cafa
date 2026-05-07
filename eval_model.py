"""
Evalua un modelo entrenado sobre el test set completo.
Uso: python eval_model.py <dataset> <model_type>
Ejemplo: python eval_model.py amlworld_hi logistic_regression
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, recall_score, confusion_matrix
)
import torch
from art.estimators.classification import PyTorchClassifier, XGBoostClassifier

sys.path.insert(0, '.')
from src.datasets.load_tabular_data import TabularDataset
from src.models.utils import load_trained_model

dataset_name = sys.argv[1] if len(sys.argv) > 1 else 'amlworld_hi'
model_type   = sys.argv[2] if len(sys.argv) > 2 else 'logistic_regression'

DATA_PATHS = {
    'amlworld_hi': (
        'data/amlworld/raw-data/HI-Small_Trans.csv',
        'data/amlworld/raw-data/amlworld.metadata.csv',
    ),
    'amlworld_li': (
        'data/amlworld/raw-data/LI-Small_Trans.csv',
        'data/amlworld/raw-data/amlworld.metadata.csv',
    ),
    'creditcard': (
        'data/creditcard/raw-data/creditcard_2023.csv',
        'data/creditcard/raw-data/creditcard.metadata.csv',
    ),
}

data_path, meta_path = DATA_PATHS[dataset_name]

print(f"Cargando {dataset_name}...")
ds = TabularDataset(
    dataset_name=dataset_name,
    data_file_path=data_path,
    metadata_file_path=meta_path,
    encoding_method='one_hot_encoding',
    random_seed=42,
    train_proportion=0.8,
)

model_path = f'trained-models/{dataset_name}-{model_type}.ckpt'
if model_type == 'xgboost':
    model_path = f'trained-models/{dataset_name}-{model_type}.pkl'

print(f"Cargando modelo {model_type} desde {model_path}...")
model = load_trained_model(model_path, model_type=model_type)

if model_type == 'xgboost':
    classifier = XGBoostClassifier(model=model.model, nb_classes=ds.n_classes)
    classifier._input_shape = (ds.n_features,)
    y_prob = model.model.predict_proba(ds.X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
else:
    classifier = PyTorchClassifier(
        model=model,
        loss=lambda o, t: torch.functional.F.cross_entropy(o, t.long()),
        input_shape=ds.n_features,
        nb_classes=ds.n_classes,
    )
    probs  = classifier.predict(ds.X_test)
    y_prob = probs[:, 1]
    y_pred = probs.argmax(axis=1)

y_test = ds.y_test.astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

print()
print(f"=== {dataset_name} — {model_type} ===")
print(f"Test set: {len(y_test):,} muestras  |  fraudes: {int(y_test.sum()):,}")
print()
print(f"  AUC-ROC:        {roc_auc_score(y_test, y_prob):.4f}")
print(f"  AUC-PR:         {average_precision_score(y_test, y_prob):.4f}")
print(f"  Recall fraude:  {recall_score(y_test, y_pred, pos_label=1, zero_division=0):.4f}")
print(f"  F1 fraude:      {f1_score(y_test, y_pred, pos_label=1, zero_division=0):.4f}")
print()
print(f"  Confusion matrix:")
print(f"    TP={tp:>6,}  FN={fn:>6,}   (fraudes detectados / perdidos)")
print(f"    FP={fp:>6,}  TN={tn:>9,}")
print(f"  class_weight usado: {ds.class_weight_minority:.2f}")
