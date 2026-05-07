# Resultados Experimentales — AMLworld HI-Small (IBM)

Dataset: **HI-Small_Trans.csv** (IBM Transactions for Anti-Money Laundering)  
Test set: **1,015,669 transacciones** | Fraudes: **1,087** (0.107% — imbalance 1:993)  
Class weight aplicado: `sqrt(n_legit / n_fraud)` = **31.50**

---

## 1. Métricas de Clasificación (antes de cualquier ataque)

| Modelo | AUC-ROC | AUC-PR | Recall fraude | F1 fraude | TP | FP | TN | FN |
|--------|---------|--------|:------------:|:---------:|---:|---:|---:|---:|
| **XGBoost** | **0.9660** | **0.1833** | **33.95%** | **0.2032** | 369 | 2,176 | 1,012,406 | 718 |
| MLP | 0.9131 | 0.0146 | 0.00% | 0.0000 | 0 | 0 | 1,014,582 | 1,087 |
| Logistic Regression | 0.5672 | 0.0013 | 10.67% | 0.0103 | 116 | 21,222 | 993,360 | 971 |

### Observaciones

- **XGBoost** es el único modelo operacionalmente viable: detecta 1 de cada 3 fraudes con solo 2,176 falsas alarmas sobre más de 1 millón de transacciones.
- **MLP**: a pesar de obtener AUC-ROC=0.91 en el dev set durante entrenamiento, predice todas las transacciones como legítimas en el test set. Con solo ~3,476 muestras de fraude en entrenamiento y un imbalance de 1:992, la red neuronal no generaliza al umbral de decisión correcto. Se documenta como hallazgo: los modelos de árbol son más eficientes en muestras bajo imbalance extremo.
- **Logistic Regression**: modelo lineal incapaz de capturar las relaciones no lineales entre features. Detecta algunas transacciones fraudulentas pero genera 21K+ falsas alarmas, inaceptable en producción.

---

## 2. Ataques Adversariales

Los ataques se ejecutaron sobre las muestras de fraude **correctamente clasificadas** por cada modelo (estrategia de ataque estratificada). Se midió cuántas de esas muestras el atacante logra hacer clasificar como legítimas.

> **Nota de compatibilidad:** CaFA y SimBA requieren acceso a gradientes o internos de red neuronal (PyTorch). No son aplicables a XGBoost (modelo de árbol no diferenciable). Este es un hallazgo relevante sobre la transferibilidad de ataques entre familias de modelos.

### 2.1 XGBoost — 369 fraudes atacados

| Ataque | Tipo | Evasión | ΔFNR | L0 medio | L0 (exitosos) | L∞ estand. |
|--------|------|:-------:|:----:|:--------:|:-------------:|:----------:|
| HopSkipJump | Black-box | **2.2%** | +0.022 | 2.56 | 4.63 | 1.07×10⁻⁵ |
| BoundaryAttack | Black-box | **100%** | +1.000 | 5.81 | 5.81 | 0.2897 |
| SquareAttack | Black-box | **100%** | +1.000 | 6.05 | 6.05 | 1.09×10⁻⁵ |
| CaFA | White-box | ❌ N/A | — | — | — | — |
| SimBA | White-box | ❌ N/A | — | — | — | — |

### 2.2 Logistic Regression — 116 fraudes atacados

| Ataque | Tipo | Evasión | ΔFNR | L0 medio | L0 (exitosos) | L∞ estand. |
|--------|------|:-------:|:----:|:--------:|:-------------:|:----------:|
| HopSkipJump | Black-box | **5.2%** | +0.052 | 2.78 | 4.50 | 4.55×10⁻⁶ |
| BoundaryAttack | Black-box | **100%** | +1.000 | 4.51 | 4.51 | 0.0112 |
| SquareAttack | Black-box | **100%** | +1.000 | 6.01 | 6.01 | 1.09×10⁻⁵ |
| CaFA | White-box | **100%** | +1.000 | 3.55 | 3.55 | 0.0185 |
| SimBA | White-box | **100%** | +1.000 | **2.39** | 2.39 | 1.94×10⁻⁷ |

> **L0:** número promedio de features modificadas (de ~15 features en total).  
> **L∞ estand.:** magnitud máxima de cambio normalizada por el rango de la feature.  
> **ΔFNR:** incremento en la tasa de falsos negativos (fraudes no detectados) post-ataque.

---

## 3. Comparación Consolidada: Antes vs. Después del Ataque

### XGBoost

| Ataque | Recall antes | Recall después | TP antes | TP después | ΔFNR |
|--------|:-----------:|:--------------:|:--------:|:----------:|:----:|
| HopSkipJump | 100% | **97.8%** | 369 | 361 | +0.022 |
| BoundaryAttack | 100% | **0%** | 369 | 0 | +1.000 |
| SquareAttack | 100% | **0%** | 369 | 0 | +1.000 |

### Logistic Regression

| Ataque | Recall antes | Recall después | TP antes | TP después | ΔFNR |
|--------|:-----------:|:--------------:|:--------:|:----------:|:----:|
| HopSkipJump | 100% | **94.8%** | 116 | 110 | +0.052 |
| BoundaryAttack | 100% | **0%** | 116 | 0 | +1.000 |
| SquareAttack | 100% | **0%** | 116 | 0 | +1.000 |
| CaFA | 100% | **0%** | 116 | 0 | +1.000 |
| SimBA | 100% | **0%** | 116 | 0 | +1.000 |

---

## 4. Análisis e Interpretación

### ¿Por qué HopSkipJump falla donde los demás tienen éxito?

HopSkipJump parte *desde dentro* de la región de fraude y busca el punto en la frontera de decisión más cercano que el modelo clasifique como legítimo. En XGBoost, cuya frontera de decisión es un conjunto de hiperplanos irregulares definidos por árboles de decisión, esta travesía resulta difícil: el algoritmo encuentra fronteras locales pero no las cruza completamente. BoundaryAttack y SquareAttack operan en la dirección opuesta (desde el espacio legítimo hacia adentro), lo que resulta mucho más efectivo.

### ¿Por qué XGBoost es más robusto que LogReg frente a HopSkipJump?

La frontera de decisión de XGBoost es altamente no lineal y discontinua (producto de múltiples árboles de decisión), lo que dificulta la navegación gradual de HopSkipJump. LogReg tiene una única frontera lineal — una vez encontrado el hiperplano, cruzarlo es trivial.

### Costo de perturbación

Con **2–6 features modificadas** sobre un total de ~15, cualquier atacante puede evadir los detectores con alta probabilidad. Las perturbaciones son mínimas en magnitud (L∞ cercano a cero en HopSkipJump y SimBA), lo que significa que las transacciones adversariales son prácticamente indistinguibles de las legítimas.

### Implicación práctica

Un sistema AML basado únicamente en clasificadores de ML es vulnerable a un atacante que conozca las features del modelo y ajuste sus transacciones en consecuencia. La robustez adversarial debe considerarse como requisito de diseño, no como mejora opcional.

---

## 5. Archivos de Resultados

| Archivo | Contenido |
|---------|-----------|
| `boundary_attack_amlworld_hi__evaluations.json` | BoundaryAttack vs XGBoost |
| `hop_skip_jump_amlworld_hi__evaluations.json` | HopSkipJump vs XGBoost |
| `square_attack_xgboost__evaluations.json` | SquareAttack vs XGBoost |
| `cafa_logistic_regression__evaluations.json` | CaFA vs LogReg |
| `boundary_attack_logistic_regression__evaluations.json` | BoundaryAttack vs LogReg |
| `hop_skip_jump_logistic_regression__evaluations.json` | HopSkipJump vs LogReg |
| `square_attack_logistic_regression__evaluations.json` | SquareAttack vs LogReg |
| `simba_logistic_regression__evaluations.json` | SimBA vs LogReg |
