"""
Entrenamiento de MLP y LSTM-Attention sobre AMLworld HI-Small
con balanceo SMOTE aplicado al conjunto de entrenamiento.

SMOTE se aplica SOLO al train set — el test set queda intacto
para evaluar en condiciones reales.
"""
import os
import shutil
import numpy as np
import torch
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, recall_score, f1_score

from src.datasets.load_tabular_data import TabularDataset
from src.models.mlp import train as train_mlp
from src.models.lstm_attention import train as train_lstm

DATA_PARAMS = {
    'dataset_name':       'amlworld_hi',
    'data_file_path':     'data/amlworld/raw-data/HI-Small_Trans.csv',
    'metadata_file_path': 'data/amlworld/raw-data/amlworld.metadata.csv',
    'encoding_method':    'one_hot_encoding',
    'random_seed':        42,
    'train_proportion':   0.8,
}

MODELS_TO_TRAIN = ['lstm_attention']

HPARAMS = {
    'mlp': {
        'hidden_dims': [128, 64],
        'dropout':     0.3,
        'lr':          1e-3,
        'weight_decay': 1e-5,
    },
    'lstm_attention': {
        'sequence_length': 1,
        'hidden_dim':      64,
        'n_lstm_layers':   2,
        'dropout':         0.3,
        'lr':              1e-3,
        'weight_decay':    1e-5,
    },
}


def print_class_dist(y, label=''):
    n_fraud = int((y == 1).sum())
    n_legit = int((y == 0).sum())
    ratio   = n_legit / max(n_fraud, 1)
    print(f'  {label}: {n_legit:,} legítimas | {n_fraud:,} fraudes | ratio 1:{ratio:.0f}')


def evaluate(model_type, model, testset):
    loader = torch.utils.data.DataLoader(testset, batch_size=2048, shuffle=False)
    all_probs, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.cpu())
            probs  = torch.softmax(logits, dim=1)[:, 1].numpy()
            all_probs.extend(probs)
            all_labels.extend(y.numpy())
    y_true  = np.array(all_labels)
    probs   = np.array(all_probs)
    preds   = (probs >= 0.5).astype(int)
    auc     = roc_auc_score(y_true, probs)
    recall  = recall_score(y_true, preds, pos_label=1, zero_division=0)
    f1      = f1_score(y_true, preds, pos_label=1, zero_division=0)
    print(f'  AUC-ROC : {auc:.4f}')
    print(f'  Recall  : {recall:.2%}')
    print(f'  F1      : {f1:.4f}')


def main():
    print('=' * 60)
    print('  ENTRENAMIENTO CON SMOTE — AMLworld HI-Small')
    print('=' * 60)

    print('\nCargando dataset...', end=' ', flush=True)
    tab = TabularDataset(**DATA_PARAMS)
    trainset, testset = tab.get_train_dev_sets()
    print('OK')

    # Extraer arrays del trainset
    X_train = torch.stack([trainset[i][0] for i in range(len(trainset))]).numpy()
    y_train = np.array([trainset[i][1].item() for i in range(len(trainset))])

    print('\nDistribución ANTES de SMOTE:')
    print_class_dist(y_train, 'Train')

    # Guardar dataset original (train)
    SMOTE_DIR = 'data/amlworld/smote'
    orig_path = os.path.join(SMOTE_DIR, 'train_original.npz')
    np.savez(orig_path, X=X_train, y=y_train)
    print(f'  Dataset original guardado en: {orig_path}')

    # Aplicar SMOTE — target: ratio 1:10 (manejable sin explotar memoria)
    print('\nAplicando SMOTE (ratio 1:10)...', end=' ', flush=True)
    smote = SMOTE(sampling_strategy=0.1, random_state=42, k_neighbors=5)
    X_bal, y_bal = smote.fit_resample(X_train, y_train)
    print('OK')

    # Guardar dataset balanceado
    bal_path = os.path.join(SMOTE_DIR, 'train_smote_1_10.npz')
    np.savez(bal_path, X=X_bal, y=y_bal)
    print(f'  Dataset SMOTE guardado en:    {bal_path}')

    print('\nDistribución DESPUÉS de SMOTE:')
    print_class_dist(y_bal, 'Train')

    # Crear nuevo dataset balanceado como TensorDataset
    X_tensor = torch.tensor(X_bal, dtype=torch.float32)
    y_tensor = torch.tensor(y_bal, dtype=torch.long)
    trainset_bal = torch.utils.data.TensorDataset(X_tensor, y_tensor)

    # Entrenar modelos
    for model_type in MODELS_TO_TRAIN:
        print(f'\n{"=" * 60}')
        print(f'  Entrenando {model_type.upper()} con SMOTE')
        print(f'{"=" * 60}')

        out_path = f'trained-models/amlworld_hi-{model_type}-smote.ckpt'

        hparams = {**HPARAMS[model_type]}

        if model_type == 'mlp':
            from src.models.mlp import train as train_fn
        else:
            from src.models.lstm_attention import train as train_fn

        train_fn(
            hyperparameters=hparams,
            trainset=trainset_bal,
            testset=testset,
            tab_dataset=tab,
            model_artifact_path=out_path,
        )

        print(f'\nEvaluación post-entrenamiento ({model_type}):')
        from src.models.utils import load_trained_model
        model = load_trained_model(out_path, model_type=model_type)
        model = model.cpu()
        evaluate(model_type, model, testset)

    print('\n' + '=' * 60)
    print('  LISTO')
    print('=' * 60)


if __name__ == '__main__':
    main()
