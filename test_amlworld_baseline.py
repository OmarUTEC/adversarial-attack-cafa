"""
Quick baseline: XGBoost on AMLworld HI-Small
  - sin class weights
  - con class weights sqrt(ratio)
  - con class weights ratio completo
"""
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    recall_score, f1_score, average_precision_score,
    roc_auc_score, confusion_matrix
)
import xgboost as xgb

sys.path.insert(0, '.')
from src.datasets.load_tabular_data import TabularDataset

print("Cargando AMLworld HI-Small...")
ds = TabularDataset(
    dataset_name='amlworld_hi',
    data_file_path='data/amlworld/raw-data/HI-Small_Trans.csv',
    metadata_file_path='data/amlworld/raw-data/amlworld.metadata.csv',
    encoding_method='one_hot_encoding',
    random_seed=42,
    train_proportion=0.8,
)
print("  Dataset cargado.")

# Split directo con numpy — evita el bug O(n²) de get_train_dev_sets
X_tr, X_dev, y_tr, y_dev = train_test_split(
    ds.X_train, ds.y_train.astype(int),
    test_size=0.15, random_state=42, stratify=ds.y_train.astype(int)
)
X_test, y_test = ds.X_test, ds.y_test.astype(int)

fraud_tr = int((y_tr == 1).sum())
legit_tr = int((y_tr == 0).sum())
ratio    = legit_tr / fraud_tr
weight   = np.sqrt(ratio)

print(f"  Train efectivo: {len(X_tr):,}  |  fraude: {fraud_tr:,}  legítimo: {legit_tr:,}")
print(f"  Dev:            {len(X_dev):,}")
print(f"  Test:           {len(X_test):,}  |  fraude: {int(y_test.sum()):,}")
print(f"  Ratio: {ratio:.1f}  |  sqrt(ratio): {weight:.2f}")
print()

BASE_PARAMS = dict(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=200,
    objective='binary:logistic',
    eval_metric='aucpr',
    random_state=42,
    verbosity=0,
)


def evaluate(name, model):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    print(f"── {name} ──")
    print(f"  AUC-ROC:        {roc_auc_score(y_test, y_prob):.4f}")
    print(f"  AUC-PR:         {average_precision_score(y_test, y_prob):.4f}")
    print(f"  Recall fraude:  {recall_score(y_test, y_pred, pos_label=1, zero_division=0):.4f}")
    print(f"  F1 fraude:      {f1_score(y_test, y_pred, pos_label=1, zero_division=0):.4f}")
    print(f"  Confusion matrix:")
    print(f"    TP={tp}  FN={fn}   (fraudes detectados / perdidos)")
    print(f"    FP={fp}  TN={tn}")
    print()

print("Entrenando SIN class weights...")
m1 = xgb.XGBClassifier(**BASE_PARAMS)
m1.fit(X_tr, y_tr, eval_set=[(X_dev, y_dev)], verbose=False)
evaluate("SIN class weights", m1)

print(f"Entrenando CON class weights sqrt ({weight:.1f})...")
m2 = xgb.XGBClassifier(**BASE_PARAMS, scale_pos_weight=weight)
m2.fit(X_tr, y_tr, eval_set=[(X_dev, y_dev)],
       verbose=False)
evaluate(f"CON sqrt(ratio) = {weight:.1f}", m2)

print(f"Entrenando CON class weights ratio completo ({ratio:.1f})...")
m3 = xgb.XGBClassifier(**BASE_PARAMS, scale_pos_weight=ratio)
m3.fit(X_tr, y_tr, eval_set=[(X_dev, y_dev)],
       verbose=False)
evaluate(f"CON ratio completo = {ratio:.1f}", m3)
