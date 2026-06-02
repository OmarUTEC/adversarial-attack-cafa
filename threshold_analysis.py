"""
Análisis de umbral de decisión para MLP y LSTM-Attention en AMLworld HI-Small.
Evalúa Recall, Precisión y F1 con distintos umbrales (0.5 → 0.01).
"""
import sys
import numpy as np
import torch
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score

from src.models.utils import load_trained_model
from src.datasets.load_tabular_data import TabularDataset

THRESHOLDS = [0.50, 0.30, 0.20, 0.10, 0.05, 0.02, 0.01]

MODELS = {
    'MLP':        ('trained-models/amlworld_hi-mlp.ckpt',            'mlp'),
    'LSTM-Att.':  ('trained-models/amlworld_hi-lstm_attention.ckpt',  'lstm_attention'),
    'Log. Reg.':  ('trained-models/amlworld_hi-logistic_regression.ckpt', 'logistic_regression'),
    'XGBoost':    ('trained-models/amlworld_hi-xgboost',               'xgboost'),
}

DATA_PARAMS = {
    'dataset_name':       'amlworld_hi',
    'data_file_path':     'data/amlworld/raw-data/HI-Small_Trans.csv',
    'metadata_file_path': 'data/amlworld/raw-data/amlworld.metadata.csv',
    'encoding_method':    'one_hot_encoding',
    'random_seed':        42,
    'train_proportion':   0.8,
}


def get_probabilities(model, model_type, X_tensor):
    if model_type == 'xgboost':
        probs = model.model.predict_proba(X_tensor.numpy())[:, 1]
    else:
        with torch.no_grad():
            logits = model(X_tensor)
            probs = torch.softmax(logits, dim=1)[:, 1].numpy()
    return probs


def bar(val, width=20):
    filled = int(round(val * width))
    return '█' * filled + '░' * (width - filled)


def main():
    print('=' * 65)
    print('  ANÁLISIS DE UMBRAL — AMLworld HI-Small')
    print('=' * 65)
    print('\nCargando dataset...', end=' ', flush=True)

    tab = TabularDataset(**DATA_PARAMS)
    _, testset = tab.get_train_dev_sets()
    X_test = torch.stack([testset[i][0] for i in range(len(testset))]).cpu()
    y_test = np.array([testset[i][1].item() for i in range(len(testset))])

    n_fraud = y_test.sum()
    n_total = len(y_test)
    print(f'OK  ({n_total:,} muestras — {int(n_fraud):,} fraudes)')

    results = {}

    for model_name, (path, mtype) in MODELS.items():
        print(f'\n{"─"*65}')
        print(f'  Modelo: {model_name}')
        print(f'{"─"*65}')
        print(f'  Cargando modelo...', end=' ', flush=True)

        try:
            model = load_trained_model(path, model_type=mtype)
            if hasattr(model, 'cpu'):
                model = model.cpu()
            print('OK')
        except Exception as e:
            print(f'ERROR: {e}')
            continue

        print(f'  Calculando probabilidades...', end=' ', flush=True)
        probs = get_probabilities(model, mtype, X_test)
        auc = roc_auc_score(y_test, probs)
        print(f'OK  (AUC-ROC = {auc:.4f})\n')

        print(f'  {"Umbral":>8}  {"Recall":>8}  {"Precisión":>10}  {"F1":>8}  {"TP":>6}  {"FP":>6}')
        print(f'  {"─"*8}  {"─"*8}  {"─"*10}  {"─"*8}  {"─"*6}  {"─"*6}')

        model_results = []
        for thr in THRESHOLDS:
            preds = (probs >= thr).astype(int)
            recall = recall_score(y_test, preds, pos_label=1, zero_division=0)
            prec   = precision_score(y_test, preds, pos_label=1, zero_division=0)
            f1     = f1_score(y_test, preds, pos_label=1, zero_division=0)
            tp     = int(((preds == 1) & (y_test == 1)).sum())
            fp     = int(((preds == 1) & (y_test == 0)).sum())

            marker = ' ◄ default' if thr == 0.50 else ''
            print(f'  {thr:>8.2f}  {recall:>7.1%}  {prec:>10.1%}  {f1:>8.4f}  {tp:>6,}  {fp:>6,}{marker}')
            model_results.append((thr, recall, prec, f1, tp, fp))

        results[model_name] = model_results

    print(f'\n{"="*65}')
    print('  RESUMEN — Mejor F1 por modelo')
    print(f'{"="*65}')
    print(f'  {"Modelo":<12}  {"Umbral":>8}  {"Recall":>8}  {"Precisión":>10}  {"F1":>8}')
    print(f'  {"─"*12}  {"─"*8}  {"─"*8}  {"─"*10}  {"─"*8}')
    for model_name, res in results.items():
        best = max(res, key=lambda x: x[3])
        thr, recall, prec, f1, tp, fp = best
        print(f'  {model_name:<12}  {thr:>8.2f}  {recall:>7.1%}  {prec:>10.1%}  {f1:>8.4f}')

    print()


if __name__ == '__main__':
    main()
