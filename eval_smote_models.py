"""
Evalúa los modelos Logistic Regression y XGBoost entrenados con SMOTE.
Extrae métricas finales en el conjunto de prueba.
"""
import json
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, accuracy_score
from src.datasets.load_tabular_data import TabularDataset
from src.models.logistic_regression import LitLogisticRegression
from src.models.xgboost_model import XGBoostModel

DATA_PARAMS = {
    'dataset_name':       'amlworld_hi',
    'data_file_path':     'data/amlworld/raw-data/HI-Small_Trans.csv',
    'metadata_file_path': 'data/amlworld/raw-data/amlworld.metadata.csv',
    'encoding_method':    'one_hot_encoding',
    'random_seed':        42,
    'train_proportion':   0.8,
}

def eval_logistic_regression():
    print('\n' + '=' * 60)
    print('  EVALUACIÓN LOGISTIC REGRESSION SMOTE')
    print('=' * 60)
    
    # Load dataset & model
    tab = TabularDataset(**DATA_PARAMS)
    _, testset = tab.get_train_dev_sets()
    
    print('Cargando modelo...', end=' ', flush=True)
    model = LitLogisticRegression.load_from_checkpoint(
        'trained-models/amlworld_hi-logistic_regression-smote.ckpt'
    )
    model.eval()
    print('OK')
    
    # Prepare test data
    testloader = torch.utils.data.DataLoader(testset, batch_size=4096, shuffle=False)
    
    all_y = []
    all_probs = []
    
    print('Prediciendo en test set...', end=' ', flush=True)
    with torch.no_grad():
        for x, y in testloader:
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.append(probs)
            all_y.append(y.cpu().numpy())
    
    y_test = np.concatenate(all_y)
    y_prob = np.concatenate(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)
    print('OK')
    
    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_prob)
    auc_pr = average_precision_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    
    results = {
        'model': 'logistic_regression',
        'dataset': 'amlworld_hi',
        'training_method': 'smote',
        'accuracy': float(acc),
        'auc_roc': float(auc_roc),
        'auc_pr': float(auc_pr),
        'f1_fraud': float(f1),
        'recall_fraud': float(recall),
    }
    
    print(f'\nResultados LOGISTIC REGRESSION SMOTE:')
    print(f'  Accuracy:      {acc:.4f}')
    print(f'  AUC-ROC:       {auc_roc:.4f}')
    print(f'  AUC-PR:        {auc_pr:.4f}')
    print(f'  F1 (fraude):   {f1:.4f}')
    print(f'  Recall (fraude): {recall:.4f}')
    
    return results

def eval_xgboost():
    print('\n' + '=' * 60)
    print('  EVALUACIÓN XGBOOST SMOTE')
    print('=' * 60)
    
    # Load dataset & model
    tab = TabularDataset(**DATA_PARAMS)
    _, testset = tab.get_train_dev_sets()
    
    print('Cargando modelo...', end=' ', flush=True)
    model_wrapper = XGBoostModel.load('trained-models/amlworld_hi-xgboost-smote')
    print('OK')
    
    # Prepare test data
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)
    
    print('Prediciendo en test set...', end=' ', flush=True)
    x_test, y_test = next(iter(testloader))
    x_test = x_test.numpy()
    y_test = y_test.numpy()
    
    y_pred = model_wrapper.model.predict(x_test)
    y_prob = model_wrapper.model.predict_proba(x_test)[:, 1]
    print('OK')
    
    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_prob)
    auc_pr = average_precision_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    
    results = {
        'model': 'xgboost',
        'dataset': 'amlworld_hi',
        'training_method': 'smote',
        'accuracy': float(acc),
        'auc_roc': float(auc_roc),
        'auc_pr': float(auc_pr),
        'f1_fraud': float(f1),
        'recall_fraud': float(recall),
    }
    
    print(f'\nResultados XGBOOST SMOTE:')
    print(f'  Accuracy:      {acc:.4f}')
    print(f'  AUC-ROC:       {auc_roc:.4f}')
    print(f'  AUC-PR:        {auc_pr:.4f}')
    print(f'  F1 (fraude):   {f1:.4f}')
    print(f'  Recall (fraude): {recall:.4f}')
    
    return results

def main():
    print('=' * 60)
    print('  EVALUACIÓN DE MODELOS SMOTE')
    print('=' * 60)
    
    results_logreg = eval_logistic_regression()
    results_xgb = eval_xgboost()
    
    # Save results
    all_results = {
        'logistic_regression': results_logreg,
        'xgboost': results_xgb,
    }
    
    with open('outputs/amlworld_hi/smote_evaluation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print('\n' + '=' * 60)
    print('  RESUMEN COMPARATIVO')
    print('=' * 60)
    print(f'\n{"Métrica":<20} {"Logistic Reg":<15} {"XGBoost":<15}')
    print('-' * 50)
    for metric in ['accuracy', 'auc_roc', 'auc_pr', 'f1_fraud', 'recall_fraud']:
        val_lr = results_logreg[metric]
        val_xgb = results_xgb[metric]
        print(f'{metric:<20} {val_lr:<15.4f} {val_xgb:<15.4f}')
    
    print(f'\nResultados guardados en: outputs/amlworld_hi/smote_evaluation_results.json')

if __name__ == '__main__':
    main()
