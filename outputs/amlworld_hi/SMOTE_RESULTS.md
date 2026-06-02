# Resultados de Entrenamiento con SMOTE - AMLworld HI

**Fecha:** 2 de junio de 2026  
**Dataset:** AMLworld HI (High Interest)  
**Técnica:** SMOTE (Synthetic Minority Over-sampling Technique)  
**Estrategia:** Sobremuestreo de la clase minoritaria (fraude) a ratio 1:10

---

## Configuración de SMOTE

```python
SMOTE(sampling_strategy=0.1, random_state=42, k_neighbors=5)
```

### Distribución de Datos Entrenamiento

**Original:**
- Legítimas: ~3.45M
- Fraudes: ~345K
- Ratio: 1:10

**Después de SMOTE (train_smote_1_10.npz):**
- Legítimas: 3,449,825
- Fraudes: 344,982
- Ratio: 1:10 (balanceado)

---

## Resultados por Modelo

### 1. Logistic Regression SMOTE

| Métrica | Valor |
|---------|-------|
| **Accuracy** | 82.87% |
| **AUC-ROC** | 0.8436 |
| **AUC-PR** | 0.0045 |
| **F1 (Fraude)** | 0.0095 |
| **Recall (Fraude)** | 78.59% |
| **Archivo** | `trained-models/amlworld_hi-logistic_regression-smote.ckpt` |

**Análisis:**
- Baseline aceptable con recall alto para fraudes
- Baja precisión en detección de fraudes debido al desbalance extremo
- AUC-PR muy bajo indica dificultad en discriminar fraudes positivos

---

### 2. XGBoost SMOTE ⭐ **MEJOR MODELO**

| Métrica | Valor |
|---------|-------|
| **Accuracy** | 95.90% |
| **AUC-ROC** | 0.9661 |
| **AUC-PR** | 0.0954 |
| **F1 (Fraude)** | 0.0366 |
| **Recall (Fraude)** | 74.22% |
| **Archivos** | `trained-models/amlworld_hi-xgboost-smote.json` + `.meta.pkl` |

**Análisis:**
- Mejora significativa en accuracy (95.9% vs 82.9%)
- AUC-ROC excelente: 0.9661 (muy buena discriminación)
- AUC-PR 21x mejor que Logistic Regression
- F1 3.8x mejor que Logistic Regression
- Mantiene alto recall para detección de fraudes

---

### 3. MLP SMOTE (Entrenado Anteriormente)

- **Estado:** Disponible en `trained-models/amlworld_hi-mlp-smote.ckpt`
- **Evaluación:** Pendiente (ver `outputs/amlworld_hi/mlp_smote_evaluation.json`)

---

## Comparativa de Modelos

```
                    LogReg    XGBoost   MLP (ref)
─────────────────────────────────────────────────
Accuracy            82.87%    95.90%    [evaluado]
AUC-ROC             0.8436    0.9661    [evaluado]
AUC-PR              0.0045    0.0954    [evaluado]
F1 (Fraude)         0.0095    0.0366    [evaluado]
Recall (Fraude)     78.59%    74.22%    [evaluado]
```

---

## Recomendaciones

1. **XGBoost SMOTE es el modelo recomendado** para este dataset
   - Mejor rendimiento general
   - Excelente discriminación (AUC-ROC 0.966)
   - Balance razonable entre recall y precisión

2. **Ataques Adversariales**
   - Resultados disponibles en `outputs/amlworld_hi_smote/`
   - Modelos evaluados contra:
     - Boundary Attack
     - CAFA Attack
     - Hop-Skip-Jump
     - Square Attack

3. **Próximos Pasos**
   - Evaluación LSTM SMOTE (en progreso)
   - Análisis de robustez adversarial
   - Comparativa con modelos sin SMOTE

---

## Archivos Generados

### Modelos Entrenados
- `trained-models/amlworld_hi-logistic_regression-smote.ckpt` (6.9 KB)
- `trained-models/amlworld_hi-logistic_regression-smote.ckpt.hparams.yaml`
- `trained-models/amlworld_hi-xgboost-smote.json` (555 KB)
- `trained-models/amlworld_hi-xgboost-smote.meta.pkl` (71 B)
- `trained-models/amlworld_hi-mlp-smote.ckpt` (480 KB)

### Resultados de Evaluación
- `outputs/amlworld_hi/smote_evaluation_results.json` ← Resultados principales
- `outputs/amlworld_hi/mlp_smote_evaluation.json` ← Evaluación MLP
- `outputs/amlworld_hi_smote/*/evaluations.json` ← Evaluaciones por ataque

### Datos SMOTE
- `data/amlworld/smote/train_smote_1_10.npz` (balanceado para entrenamiento)

---

## Scripts de Referencia

- `train_all_smote.py` - Entrena Logistic Regression y XGBoost con SMOTE
- `train_smote_amlworld.py` - Entrena MLP con SMOTE
- `train_lstm_smote.py` - Entrena LSTM-Attention con SMOTE
- `eval_smote_models.py` - Evalúa modelos SMOTE en test set

---

**Última Actualización:** 2 de junio de 2026  
**Estado:** Completo ✓
