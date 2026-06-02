"""
Evaluación del modelo LSTM-Attention entrenado con SMOTE sobre AMLworld HI-Small.
Guarda resultados en outputs/amlworld_hi/lstm_smote_evaluation.json
"""
import json
import os
import numpy as np
import torch
from sklearn.metrics import (roc_auc_score, recall_score,
                              f1_score, precision_score, confusion_matrix)

from src.models.utils import load_trained_model
from src.datasets.load_tabular_data import TabularDataset

MODEL_PATH = 'trained-models/amlworld_hi-lstm_attention-smote.ckpt'
OUT_PATH   = 'outputs/amlworld_hi/lstm_smote_evaluation.json'

DATA_PARAMS = {
    'dataset_name':       'amlworld_hi',
    'data_file_path':     'data/amlworld/raw-data/HI-Small_Trans.csv',
    'metadata_file_path': 'data/amlworld/raw-data/amlworld.metadata.csv',
    'encoding_method':    'one_hot_encoding',
    'random_seed':        42,
    'train_proportion':   0.8,
}


def main():
    if not os.path.exists(MODEL_PATH):
        print(f'ERROR: modelo no encontrado en {MODEL_PATH}')
        print('Asegúrate de haber completado el entrenamiento con train_smote_amlworld.py')
        return

    print('Cargando dataset...', end=' ', flush=True)
    tab = TabularDataset(**DATA_PARAMS)
    _, testset = tab.get_train_dev_sets()
    print('OK')

    print('Cargando modelo LSTM-Attention...', end=' ', flush=True)
    model = load_trained_model(MODEL_PATH, model_type='lstm_attention')
    model = model.cpu().eval()
    print('OK')

    print('Evaluando...', end=' ', flush=True)
    loader = torch.utils.data.DataLoader(testset, batch_size=2048, shuffle=False)
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            probs = torch.softmax(model(x.cpu()), dim=1)[:, 1].numpy()
            all_probs.extend(probs)
            all_labels.extend(y.numpy())
    print('OK')

    y_true = np.array(all_labels)
    probs  = np.array(all_probs)
    preds  = (probs >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()

    results = {
        'modelo':   'LSTM-Attention',
        'dataset':  'AMLworld HI-Small',
        'tecnica':  'SMOTE ratio 1:10',
        'metricas': {
            'AUC-ROC':   round(roc_auc_score(y_true, probs), 4),
            'Recall':    round(recall_score(y_true, preds, pos_label=1, zero_division=0), 4),
            'Precision': round(precision_score(y_true, preds, pos_label=1, zero_division=0), 4),
            'F1':        round(f1_score(y_true, preds, pos_label=1, zero_division=0), 4),
        },
        'confusion_matrix': {'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn)},
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(results, f, indent=4)

    print(f'\nGuardado en: {OUT_PATH}')
    print(json.dumps(results, indent=4))


if __name__ == '__main__':
    main()
