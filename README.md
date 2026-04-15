# Robustez Adversarial en Modelos de Detección de Fraude Financiero

Este repositorio contiene un framework sistemático y reproducible para evaluar la robustez de modelos de Machine Learning aplicados a la detección de fraude bancario utilizando datos tabulares.

## 🚀 Algoritmos Implementados

### Modelos de Machine Learning (Defensa)
Arquitecturas diseñadas para la detección de fraude transaccional:
*   **XGBoost (eXtreme Gradient Boosting):** Modelo de ensamble de árboles de decisión, estándar en la industria bancaria.
*   **MLP (Perceptrón Multicapa):** Red neuronal densa para captura de relaciones no lineales.
*   **LSTM-Attention:** Arquitectura recurrente con mecanismo de atención aditiva para capturar dependencias complejas.
*   **Regresión Logística (LogReg):** Modelo lineal base para comparativas de interpretabilidad.

### ⚔️ Ataques Adversariales (Evasión)
Framework de evaluación de vulnerabilidades utilizando ART (Adversarial Robustness Toolbox):
*   **CaFA (Categorical Feature Attack):** Ataque de Caja Blanca optimizado para datos tabulares con restricciones de integridad.
*   **HopSkipJump (HSJ):** Ataque de Caja Negra basado en el límite de decisión (Decision Boundary).
*   **Boundary Attack:** Genera ejemplos adversariales cercanos mediante procesos de búsqueda en la frontera de decisión.
*   **SimBA (Simple Black-box Attack):** Ataque basado en consultas que perturba características individuales de forma iterativa.
*   **Square Attack:** Ataque de Caja Negra basado en consultas, optimizado para no requerir gradientes.

## Datasets Utilizados
Los experimentos se centran en datos financieros reales y sintetizados con desafíos de desbalance y confidencialidad:
*   **Credit Card Fraud Detection (2023):** Transacciones bancarias transformadas por PCA para garantizar la privacidad de los usuarios, con un enfoque en la detección de anomalías.
*   **Adult (Censo):** Datos socioeconómicos utilizados para validar la generalización del framework en datos estructurados.
*   **Bank Marketing:** Datos de campañas bancarias para evaluar ataques en variables categóricas y continuas.
*   **Phishing Dataset:** Detección de sitios fraudulentos mediante características técnicas.

## Estructura del Proyecto
```text
.
├── config/             # Configuraciones Hydra (Ataques, Datos, Modelos)
├── data/               # Datasets crudos y restricciones minadas (Git LFS)
├── src/
│   ├── attacks/        # Implementación de ataques especializados
│   ├── datasets/       # Cargadores y preprocesamiento de datos
│   ├── models/         # Arquitecturas de ML (PyTorch & XGBoost)
│   └── utils.py        # Métricas (AUC, ASR, Costos L0/Lp)
├── trained-models/     # Checkpoints de los modelos entrenados
├── attack.py           # Script principal de ejecución
└── requirements.txt    # Dependencias del entorno
```

## � Matriz de Compatibilidad y Estado

A continuación se detalla la compatibilidad técnica entre los ataques implementados y las arquitecturas de modelos, así como el estado actual de su integración en el framework:

| Modelo | Arquitectura | **CaFA (PFC1)** | **Boundary Attack** | **HopSkipJump** | **SimBA / Square** | Estado |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **MLP** | Neuronal (Dif.) | Sí | Sí | Sí | Sí | Revisar |
| **LSTM-Attention** | Recurrente (Dif.) | Sí | Sí | Sí | Sí | Revisar |
| **Log. Regression** | Lineal (Dif.) | Sí | Sí | Sí | Sí | Revisar |
| **XGBoost** | Árboles (No-Dif.) | No* | Sí | Sí | Sí | Revisar |
| **TabTransformer** | Transformer (Dif.)| Sí | Sí | Sí | Sí | **En progreso** |

*\*XGBoost es inherentemente robusto contra ataques basados en gradientes (Caja Blanca) debido a su naturaleza no diferenciable. Requiere ataques de Caja Negra o de transferencia.*

## �🛠️ Instalación y Uso
(Instrucciones de instalación aquí...)

