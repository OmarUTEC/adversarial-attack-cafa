# Robustez Adversarial en Modelos de Detección de Fraude Financiero

Framework sistemático, reproducible y extendible para evaluar la robustez adversarial de modelos de Machine Learning aplicados a la detección de fraude bancario en datos tabulares. Implementa ataques de caja blanca y caja negra sobre cuatro arquitecturas distintas, midiendo vulnerabilidad mediante métricas de discriminación (AUC), tasa de éxito del ataque (ASR) y costos de perturbación (L0, Lp).

## Algoritmos Implementados

### Modelos de Machine Learning

| Modelo | Tipo | HPO | Framework |
| :--- | :--- | :--- | :--- |
| Regresión Logística | Lineal (baseline) | Optuna | PyTorch Lightning |
| MLP | Red neuronal densa | Optuna | PyTorch Lightning |
| LSTM-Attention | Recurrente + atención aditiva | Optuna | PyTorch Lightning |
| XGBoost | Ensamble de árboles | Manual | XGBoost |

### Ataques Adversariales

Implementados sobre [ART (Adversarial Robustness Toolbox)](https://github.com/Trusted-AI/adversarial-robustness-toolbox), con soporte para restricciones de integridad de datos tabulares (rangos de features, variables categóricas, variables enteras).

| Ataque | Caja | Gradiente | Descripción |
| :--- | :--- | :--- | :--- |
| **CaFA** (Categorical Feature Attack) | Blanca | Si | Combina TabCW (L0-minimización) con TabPGD; maneja features categóricas, ordinales y continuas por separado |
| **HopSkipJump (HSJ)** | Negra | No | Búsqueda iterativa en la frontera de decisión mediante estimación de gradiente por signos |
| **Boundary Attack** | Negra | No | Paseo aleatorio sobre la frontera de decisión con proyección al dominio tabular |
| **SimBA** | Negra | No | Perturba una feature a la vez de forma greedy (L-inf) |
| **Square Attack** | Negra | No | Perturba subconjuntos aleatorios de features con múltiples reinicios |

## Datasets

| Dataset | Dominio | Features | Desafío principal |
| :--- | :--- | :--- | :--- |
| **Credit Card Fraud 2023** | Fraude bancario | 29 (PCA) + monto | Desbalance extremo, features anonimizadas por PCA |
| **Adult (Censo)** | Ingreso > 50K | 14 mixtas | Variables categóricas y continuas |
| **Bank Marketing** | Campaña bancaria | 20 mixtas | Variables ordinales y categóricas |
| **Phishing** | Detección web | 30 técnicas | Clasificación binaria en features técnicas |

### Partición del Dataset (Credit Card Fraud 2023)

El dataset se divide en tres subconjuntos con semilla fija (`random_seed=42`) para garantizar reproducibilidad:

| Subconjunto | Proporción | Transacciones | Uso |
| :--- | :---: | ---: | :--- |
| **Total** | 100% | 568,630 | — |
| **Train set** | 80% | 454,904 | Base de entrenamiento |
| — Training efectivo | 85% del train | 386,669 | Optimización de pesos del modelo |
| — Dev set | 15% del train | 68,235 | Validación durante entrenamiento (early stopping) |
| **Test set** | 20% | 113,726 | Evaluación final e igualdad de ataques |
| **Muestras atacadas** | 1.8% del test | 2,000 | Subconjunto representativo usado en los experimentos |

Los ataques adversariales se ejecutan exclusivamente sobre el **test set** — datos que el modelo nunca vio durante el entrenamiento — para garantizar que los resultados reflejan el comportamiento real del modelo en producción y no una memorización del conjunto de entrenamiento. El tamaño de 2,000 muestras está controlado por el parámetro `n_samples_to_attack` en `config/config.yaml`.

## Métricas de Evaluación

- **Integridad Base:** AUC-ROC sobre el conjunto de test limpio
- **Vulnerabilidad:** ASR — tasa de muestras adversariales que evaden el clasificador
- **Sigilo:** Costo L0 (número de features modificadas) y costo L-inf estandarizado (magnitud de perturbación normalizada por rango intercuantílico)

## Estructura del Proyecto

```text
.
├── attack.py               # Entry point principal (Hydra)
├── config/
│   ├── config.yaml         # Config global con defaults
│   ├── attack/             # Configs por ataque (cafa, hsj, boundary, simba, square)
│   ├── data/               # Configs por dataset
│   └── ml_model/           # Configs por modelo
├── src/
│   ├── attacks/            # Implementaciones de ataques (ART extensions)
│   ├── datasets/           # Loaders y preprocesamiento por dataset
│   ├── models/             # Arquitecturas (PyTorch Lightning + XGBoost)
│   └── utils.py            # Métricas: ASR, L0, Lp estandarizado
├── trained-models/         # Checkpoints entrenados (.ckpt / .pkl)
├── data/                   # Datasets crudos y metadata (Git LFS)
└── requirements.txt
```

## Matriz de Compatibilidad

| Modelo | CaFA (blanca) | HSJ (negra) | Boundary (negra) | SimBA (negra) | Square (negra) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Regresión Logística | Si | Si | Si | Si | Si |
| MLP | Si | Si | Si | Si | Si |
| LSTM-Attention | Si | Si | Si | Si | Si |
| XGBoost | No¹ | Si | Si | No² | Si |

¹ CaFA requiere `loss_gradient()` — el gradiente de entrada es nulo en árboles de decisión.  
² SimBA requiere `NeuralNetworkMixin` de ART, que `XGBoostClassifier` no implementa.

## Resultados Experimentales — Credit Card Fraud 2023 (n = 2000)

> ASR base = tasa de error natural del clasificador sin ataque. ΔASR = incremento atribuido al ataque.

### XGBoost

| Ataque | Tipo | ASR | ΔASR | L0 (evasiones) | L-inf estd. (evasiones) |
| :--- | :---: | ---: | ---: | ---: | ---: |
| Baseline | — | 0.05% | — | 0 | 0 |
| Square Attack | Negra | 50.6% | +50.6% | 21.76 | 0.736 |
| HopSkipJump | Negra | 99.95% | +99.9% | 29.97 | 71.41 |
| Boundary Attack | Negra | 99.95% | +99.9% | 29.85 | 88.80 |

*CaFA y SimBA no son compatibles con XGBoost (ver Matriz de Compatibilidad).*

### LSTM-Attention

| Ataque | Tipo | ASR | ΔASR | L0 (evasiones) | L-inf estd. (evasiones) |
| :--- | :---: | ---: | ---: | ---: | ---: |
| Baseline | — | 22.1% | — | 0 | 0 |
| SimBA | Negra | 22.1% | 0% | 0 | 0 |
| Square Attack | Negra | 22.6% | +0.5% | 0.34 | 0.0149 |
| CaFA | Blanca | 23.7% | +1.6% | 2.54 | 0.0031 |
| Boundary Attack | Negra | 52.1% | +30.0% | 28.75 | 67.92 |
| HopSkipJump | Negra | 77.9% | +55.8% | 29.98 | 45.49 |

### MLP

| Ataque | Tipo | ASR | ΔASR | L0 (evasiones) | L-inf estd. (evasiones) |
| :--- | :---: | ---: | ---: | ---: | ---: |
| Baseline | — | 0.1% | — | 0 | 0 |
| SimBA | Negra | 3.5% | +3.4% | 4.23 | 0.1106 |
| Boundary Attack | Negra | 48.0% | +47.9% | 29.81 | 102.25 |
| CaFA | Blanca | 52.9% | +52.7% | 22.58 | 0.0332 |
| HopSkipJump | Negra | 99.8% | +99.7% | 29.89 | 0.1493 |
| Square Attack | Negra | 100.0% | +99.9% | 11.92 | 0.6272 |

### Regresión Logística

| Ataque | Tipo | ASR | ΔASR | L0 (evasiones) | L-inf estd. (evasiones) |
| :--- | :---: | ---: | ---: | ---: | ---: |
| Baseline | — | 0.8% | — | 0 | 0 |
| SimBA | Negra | 5.5% | +4.7% | 1.44 | 0.0802 |
| Boundary Attack | Negra | 48.0% | +47.2% | 29.27 | 0.3570 |
| CaFA | Blanca | 78.5% | +77.7% | 12.25 | 0.0324 |
| HopSkipJump | Negra | 99.2% | +98.4% | 29.83 | 0.1588 |
| Square Attack | Negra | 100.0% | +99.2% | 15.58 | 0.7124 |

### Interpretación de Resultados

**Cómo leer las métricas:**
- **ASR (Attack Success Rate):** porcentaje de muestras que el modelo clasifica incorrectamente tras el ataque. Un ASR alto significa que el ataque logró engañar al clasificador.
- **ΔASR:** incremento real atribuible al ataque (ASR después − ASR base). Es la métrica principal de efectividad.
- **L0 (evasiones):** promedio de features modificadas en las muestras que sí lograron evadir. Mide la minimalidad del ataque — menos features = más sigiloso.
- **L-inf estd. (evasiones):** magnitud máxima de perturbación normalizada por el rango intercuantílico de cada feature. Valores cercanos a 0 indican perturbaciones imperceptibles; valores altos (>1) indican perturbaciones burdas y detectables.

**Conclusiones por modelo:**

- **Regresión Logística** es altamente vulnerable: CaFA logra 78.5% de ΔASR modificando solo 12.25 features con perturbaciones casi imperceptibles (L-inf 0.032), lo que refleja que la superficie de decisión lineal es fácilmente explotable con gradientes directos. Square Attack alcanza el 100% de evasión, confirmando que el modelo no puede resistir perturbaciones coordinadas en múltiples features. SimBA es el único ataque con impacto reducido (+4.7%), al necesitar más iteraciones para cruzar una frontera lineal feature por feature.

- **XGBoost** presenta vulnerabilidad extrema frente a HSJ y Boundary Attack (ΔASR ~99.9%), pero ambos ataques requieren perturbaciones masivas (L-inf ~71–89), lo que los hace fácilmente detectables por sistemas de monitoreo. Su resistencia frente a Square Attack es moderada (50.6%), con un costo de perturbación razonable. Al ser un modelo de árboles, solo es atacable en caja negra.

- **LSTM-Attention** es el modelo más robusto del experimento. SimBA no consigue ninguna evasión adicional sobre el baseline (ΔASR = 0%), y CaFA — el ataque más sofisticado — solo logra +1.6% con perturbaciones casi imperceptibles (L-inf 0.003). HSJ alcanza 77.9% de ASR pero a un costo de perturbación elevado (L-inf 45.49). El alto ASR base (22.1%) sugiere que parte de las "evasiones" provienen del error natural del modelo sobre transacciones ambiguas.

- **MLP** es el modelo más vulnerable entre las redes neuronales. Square Attack lo evade al 100% y HSJ al 99.8%, ambos con costos de perturbación moderados. CaFA logra 52.9% de evasión modificando en promedio 22.6 features con perturbaciones mínimas (L-inf 0.033), siendo el ataque más sigiloso sobre esta arquitectura. El MLP, al ser completamente diferenciable y con gradientes densos, ofrece una superficie de ataque amplia tanto para métodos de caja blanca como negra.

**Patrón general observado:** existe un trade-off claro entre efectividad y sigilo. Los ataques basados en frontera de decisión (HSJ, Boundary) logran ASR altos pero con perturbaciones detectables. CaFA representa el punto óptimo del trade-off: alta efectividad con mínima perturbación, gracias a su diseño específico para datos tabulares con restricciones de integridad. Square Attack es consistentemente devastador en modelos diferenciables (100% en LogReg y MLP) pero requiere perturbaciones mayores que CaFA.

## Instalación y Uso

```bash
# Instalar dependencias
pip install -r requirements.txt

# Entrenar un modelo
python attack.py data=creditcard ml_model=lstm_attention perform_attack=False ml_model.perform_training=True

# Ejecutar ataque (config por defecto: lstm_attention + cafa + creditcard)
python attack.py

# Combinaciones distintas
python attack.py data=creditcard ml_model=xgboost attack=hop_skip_jump
python attack.py data=adult ml_model=mlp attack=square_attack
python attack.py data=bank ml_model=logistic_regression attack=simba
```

Los resultados se guardan en `outputs/{timestamp}/` con `evaluations.json`, `X_adv.npy` y logs de TensorBoard.
