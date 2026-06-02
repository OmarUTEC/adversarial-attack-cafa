"""
Entrena Log. Reg. y XGBoost con SMOTE sobre AMLworld HI-Small.
MLP y LSTM se entrenan por separado con train_smote_amlworld.py / train_lstm_smote.py
"""
import os
import numpy as np
import torch
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, recall_score, f1_score

from src.datasets.load_tabular_data import TabularDataset

DATA_PARAMS = {
    'dataset_name':       'amlworld_hi',
    'data_file_path':     'data/amlworld/raw-data/HI-Small_Trans.csv',
    'metadata_file_path': 'data/amlworld/raw-data/amlworld.metadata.csv',
    'encoding_method':    'one_hot_encoding',
    'random_seed':        42,
    'train_proportion':   0.8,
}

SMOTE_DIR = 'data/amlworld/smote'
MODELS_TO_TRAIN = ['logistic_regression', 'xgboost']


def print_dist(y, label=''):
    nf = int((y==1).sum())
    nl = int((y==0).sum())
    print(f'  {label}: {nl:,} legítimas | {nf:,} fraudes | ratio 1:{nl//max(nf,1)}')


def main():
    print('=' * 60)
    print('  ENTRENAMIENTO LOG.REG. + XGBOOST CON SMOTE')
    print('=' * 60)

    print('\nCargando dataset...', end=' ', flush=True)
    tab = TabularDataset(**DATA_PARAMS)
    trainset, testset = tab.get_train_dev_sets()
    print('OK')

    # Cargar SMOTE si existe, sino generar
    bal_path = os.path.join(SMOTE_DIR, 'train_smote_1_10.npz')
    if os.path.exists(bal_path):
        print(f'Cargando SMOTE desde {bal_path}...', end=' ', flush=True)
        data  = np.load(bal_path)
        X_bal = data['X']
        y_bal = data['y']
        print('OK')
    else:
        print('Generando SMOTE...', end=' ', flush=True)
        X_tr = torch.stack([trainset[i][0] for i in range(len(trainset))]).numpy()
        y_tr = np.array([trainset[i][1].item() for i in range(len(trainset))])
        smote = SMOTE(sampling_strategy=0.1, random_state=42, k_neighbors=5)
        X_bal, y_bal = smote.fit_resample(X_tr, y_tr)
        os.makedirs(SMOTE_DIR, exist_ok=True)
        np.savez(bal_path, X=X_bal, y=y_bal)
        print('OK')

    print_dist(y_bal, 'SMOTE train')

    trainset_bal = torch.utils.data.TensorDataset(
        torch.tensor(X_bal, dtype=torch.float32),
        torch.tensor(y_bal, dtype=torch.long)
    )

    for model_type in MODELS_TO_TRAIN:
        print(f'\n{"=" * 60}')
        print(f'  Entrenando {model_type.upper()} con SMOTE')
        print(f'{"=" * 60}')

        out_path = f'trained-models/amlworld_hi-{model_type}-smote.ckpt'

        if model_type == 'logistic_regression':
            from src.models.logistic_regression import train as train_fn
            train_fn(
                hyperparameters={'lr': 1e-3, 'weight_decay': 1e-5},
                trainset=trainset_bal,
                testset=testset,
                tab_dataset=tab,
                model_artifact_path=out_path,
            )
        elif model_type == 'xgboost':
            from src.models.xgboost_model import train as train_fn
            out_path = f'trained-models/amlworld_hi-{model_type}-smote'
            train_fn(
                hparams={},
                trainset=trainset_bal,
                testset=testset,
                tab_dataset=tab,
                model_artifact_path=out_path,
            )

        print(f'\nModelo guardado en: {out_path}')

    print('\n' + '=' * 60)
    print('  LISTO')
    print('=' * 60)
    print('\nAhora corre los ataques con:')
    print('  bash run_smote_attacks.sh')


if __name__ == '__main__':
    main()
