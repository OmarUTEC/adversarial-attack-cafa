"""
Entrenamiento de LSTM-Attention sobre AMLworld HI-Small
con balanceo SMOTE aplicado al conjunto de entrenamiento.
"""
import os
import numpy as np
import torch
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, recall_score, f1_score

from src.datasets.load_tabular_data import TabularDataset
from src.models.lstm_attention import train as train_fn
from src.models.utils import load_trained_model

DATA_PARAMS = {
    'dataset_name':       'amlworld_hi',
    'data_file_path':     'data/amlworld/raw-data/HI-Small_Trans.csv',
    'metadata_file_path': 'data/amlworld/raw-data/amlworld.metadata.csv',
    'encoding_method':    'one_hot_encoding',
    'random_seed':        42,
    'train_proportion':   0.8,
}

HPARAMS = {
    'sequence_length': 1,
    'hidden_dim':      64,
    'n_lstm_layers':   2,
    'dropout':         0.3,
    'lr':              1e-3,
    'weight_decay':    1e-5,
}

OUT_PATH  = 'trained-models/amlworld_hi-lstm_attention-smote.ckpt'
SMOTE_DIR = 'data/amlworld/smote'


def print_class_dist(y, label=''):
    n_fraud = int((y == 1).sum())
    n_legit = int((y == 0).sum())
    print(f'  {label}: {n_legit:,} legítimas | {n_fraud:,} fraudes | ratio 1:{n_legit//max(n_fraud,1)}')


def evaluate(model, testset):
    loader = torch.utils.data.DataLoader(testset, batch_size=2048, shuffle=False)
    all_probs, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            probs = torch.softmax(model(x.cpu()), dim=1)[:, 1].numpy()
            all_probs.extend(probs)
            all_labels.extend(y.numpy())
    y_true = np.array(all_labels)
    probs  = np.array(all_probs)
    preds  = (probs >= 0.5).astype(int)
    print(f'  AUC-ROC : {roc_auc_score(y_true, probs):.4f}')
    print(f'  Recall  : {recall_score(y_true, preds, pos_label=1, zero_division=0):.2%}')
    print(f'  F1      : {f1_score(y_true, preds, pos_label=1, zero_division=0):.4f}')


def main():
    print('=' * 60)
    print('  ENTRENAMIENTO LSTM-ATTENTION CON SMOTE — AMLworld HI')
    print('=' * 60)

    # Cargar dataset
    print('\nCargando dataset...', end=' ', flush=True)
    tab = TabularDataset(**DATA_PARAMS)
    trainset, testset = tab.get_train_dev_sets()
    print('OK')

    # Cargar SMOTE desde archivo si ya existe, sino generarlo
    bal_path = os.path.join(SMOTE_DIR, 'train_smote_1_10.npz')
    if os.path.exists(bal_path):
        print(f'\nCargando SMOTE desde {bal_path}...', end=' ', flush=True)
        data   = np.load(bal_path)
        X_bal  = data['X']
        y_bal  = data['y']
        print('OK')
    else:
        print('\nGenerando SMOTE (ratio 1:10)...', end=' ', flush=True)
        X_train = torch.stack([trainset[i][0] for i in range(len(trainset))]).numpy()
        y_train = np.array([trainset[i][1].item() for i in range(len(trainset))])
        smote   = SMOTE(sampling_strategy=0.1, random_state=42, k_neighbors=5)
        X_bal, y_bal = smote.fit_resample(X_train, y_train)
        os.makedirs(SMOTE_DIR, exist_ok=True)
        np.savez(bal_path, X=X_bal, y=y_bal)
        print('OK')

    print('\nDistribución del train set:')
    print_class_dist(y_bal, 'SMOTE')

    # Crear TensorDataset balanceado
    trainset_bal = torch.utils.data.TensorDataset(
        torch.tensor(X_bal, dtype=torch.float32),
        torch.tensor(y_bal, dtype=torch.long)
    )

    # Entrenar LSTM
    print(f'\n{"=" * 60}')
    print('  Entrenando LSTM-ATTENTION con SMOTE')
    print(f'{"=" * 60}')

    train_fn(
        hyperparameters=HPARAMS,
        trainset=trainset_bal,
        testset=testset,
        tab_dataset=tab,
        model_artifact_path=OUT_PATH,
    )

    # Evaluar
    print('\nEvaluación post-entrenamiento (lstm_attention):')
    model = load_trained_model(OUT_PATH, model_type='lstm_attention')
    model = model.cpu()
    evaluate(model, testset)

    print('\n' + '=' * 60)
    print('  LISTO — modelo guardado en:', OUT_PATH)
    print('=' * 60)


if __name__ == '__main__':
    main()
