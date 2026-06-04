# Resultados — AMLworld HI-Small (SMOTE)

Evaluación de ataques adversariales sobre 4 modelos entrenados con SMOTE (ratio 1:10) en el
dataset AMLworld HI-Small (IBM). El conjunto contiene transacciones financieras reales con
etiquetas de lavado de dinero, con una tasa de fraude extremadamente baja (~0.1%).
Todos los ataques se aplican exclusivamente sobre muestras de fraude del conjunto de test
que el modelo clasifica correctamente antes del ataque.

---

## 1. Métricas basales (sin ataque)

| Modelo | AUC-ROC | Recall fraude | F1 fraude |
|---|---|---|---|
| MLP | 0.935 | 82.3% | 0.018 |
| LSTM-Attention | 0.713 | 96.9% | 0.003 |
| Logistic Reg. | 0.844 | 78.6% | 0.010 |
| XGBoost | 0.966 | 74.2% | 0.037 |

**Interpretación:** El F1 bajo en todos los modelos es inherente al desbalance extremo del
dataset (~0.1% de fraudes): incluso con SMOTE, la clase mayoritaria domina la distribución
de test. El **recall de fraude** es la métrica operacionalmente crítica, pues mide cuántas
transacciones ilícitas logra detectar el sistema antes de ser atacado. El LSTM-Attention
obtiene el recall más alto (96.9%) a costa de un AUC-ROC más bajo (0.713), lo que indica
que el modelo tiende a clasificar más agresivamente como fraude pero con menor discriminación
global. XGBoost logra el mejor AUC-ROC (0.966), reflejando mayor poder discriminativo
general, aunque su recall de fraude (74.2%) es el más bajo entre los cuatro modelos.

---

## 2. Tasa de evasión por ataque (%)

| Modelo | CaFA | HopSkipJump | Boundary | Square |
|---|---|---|---|---|
| MLP | 98.8 | 95.1 | 100.0 | 100.0 |
| LSTM-Attention | **68.2** | 0.0 | 0.7 | 0.2 |
| Logistic Reg. | 100.0 | 32.2 | 99.8 | 89.6 |
| XGBoost | N/A | 99.3 | 100.0 | 100.0 |

> CaFA no se aplica a XGBoost porque requiere gradientes de la función de pérdida; XGBoost
> es un modelo de árboles de decisión que no expone derivadas continuas.

---

## 3. Costo de perturbación — L₀ medio sobre ejemplos evadidos

| Modelo | CaFA | HopSkipJump | Boundary | Square |
|---|---|---|---|---|
| MLP | 4.72 | 9.62 | 4.53 | 6.17 |
| LSTM-Attention | 3.69 | — | 8.29 | 6.50 |
| Logistic Reg. | 3.52 | 9.06 | 4.76 | 7.09 |
| XGBoost | N/A | 9.94 | 4.97 | 6.39 |

**Interpretación:** L₀ mide el número de features modificadas en cada transacción adversarial.
Valores bajos indican ataques más sigilosos: con pocas modificaciones el atacante logra
evadir el detector. CaFA destaca como el ataque de menor costo (3.5–4.7 features promedio),
lo que lo hace especialmente peligroso porque las perturbaciones son difíciles de detectar
manualmente o con sistemas de auditoría basados en umbrales simples. HopSkipJump requiere
modificar ~10 features en promedio, lo que podría activar alertas en sistemas de monitoreo
basados en reglas.

---

## 4. Análisis por modelo

### 4.1 MLP — Modelo más vulnerable

El MLP es el modelo que peor resiste los ataques adversariales: todos los ataques —incluyendo
los de caja negra sin acceso al gradiente— logran tasas de evasión de entre 95% y 100%.
Esto se explica porque las redes neuronales densas aprenden representaciones lineales por
partes con fronteras de decisión suaves, lo que facilita que perturbaciones continuas pequeñas
crucen dichas fronteras. La alta dimensionalidad combinada con la naturaleza suave del
espacio de decisión del MLP lo hace vulnerable tanto a ataques basados en gradiente (CaFA)
como a los basados en decisiones (HopSkipJump, Boundary, Square). En términos prácticos,
un adversario con acceso de caja negra al modelo puede evadir el detector con alta
probabilidad modificando menos de 10 features de la transacción.

### 4.2 LSTM-Attention — Modelo más robusto

El LSTM-Attention es el modelo más resistente del experimento: los tres ataques de caja
negra fallan casi completamente (HopSkipJump: 0.0%, Boundary: 0.7%, Square: 0.2%), mientras
que CaFA —el único ataque basado en gradiente— logra evadir el 68.2% de las muestras.
Esta brecha se explica por las propiedades de la arquitectura: (1) la capa de atención
crea una superficie de decisión altamente no lineal que es difícil de atravesar con búsquedas
basadas en decisión, y (2) la naturaleza recurrente del LSTM genera gradientes internos
complejos que no se transfieren eficientemente a perturbaciones de caja negra. El hecho de
que CaFA sí tenga éxito con 68.2% demuestra que la vulnerabilidad existe, pero solo puede
explotarse con acceso al gradiente del modelo, lo que representa un supuesto de amenaza más
restrictivo (adversario de caja blanca). En escenarios de despliegue real, donde el
atacante generalmente no tiene acceso al modelo interno, el LSTM-Attention es el más seguro.

### 4.3 Logistic Regression — Vulnerable a todos excepto HopSkipJump parcialmente

La regresión logística presenta el perfil de vulnerabilidad más desigual: CaFA y Boundary
Attack la evaden casi completamente (100% y 99.8%), Square Attack alcanza 89.6%, mientras
que HopSkipJump solo logra 32.2%. Esta resistencia parcial a HopSkipJump se explica porque
el algoritmo opera en norma L₂ y requiere muestras iniciales adversariales válidas; la
frontera de decisión lineal de LogReg en el espacio de features binarias (one-hot encoding)
genera una región de decisión con bordes abruptos que dificultan la interpolación L₂.
Sin embargo, Boundary Attack —que también opera sin gradientes— sí logra evadir 99.8%,
lo que indica que la frontera lineal es en realidad muy fácil de cruzar con métodos
que buscan directamente sobre la superficie de decisión. CaFA evade el 100% porque la
frontera lineal tiene un gradiente constante y bien definido, lo que hace trivial calcular
la dirección de perturbación óptima con muy pocas features modificadas (L₀ = 3.52).

### 4.4 XGBoost — Vulnerable a todos los ataques de caja negra

XGBoost no admite CaFA por su naturaleza de árbol de decisión no diferenciable. Sin embargo,
los tres ataques de caja negra lo evaden con tasas de 99.3%–100%. Esto ocurre porque las
fronteras de decisión de los árboles de boosting forman regiones constantes por partes
(superficies en escalón), que son más fáciles de cruzar mediante búsqueda por decisión
que las superficies suaves de redes neuronales. Boundary Attack logra 100% de evasión,
lo que indica que prácticamente toda transacción de fraude puede ser reformulada como una
transacción "legítima" desde la perspectiva del modelo con solo modificar ~5 features.
Esto es especialmente preocupante porque XGBoost tiene el mejor AUC-ROC basal (0.966)
pero es completamente vulnerable a adversarios de caja negra.

---

## 5. Hallazgos clave

### 5.1 SMOTE no mejora la robustez adversarial
El entrenamiento con SMOTE mejora el recall basal de los modelos al balancear las clases de
entrenamiento, pero no confiere resistencia frente a ataques adversariales. La vulnerabilidad
adversarial depende de la geometría de la frontera de decisión aprendida, no de la
distribución de los datos de entrenamiento. SMOTE no regulariza ni endurece esa frontera.

### 5.2 CaFA es el ataque más eficiente y sigiloso
Con L₀ de 3.5–4.7 features y perturbaciones L∞ inferiores al 2.4% del rango normalizado,
CaFA es a la vez el ataque más eficaz (68%–100% de evasión) y el más difícil de detectar.
Su capacidad para evadir tanto modelos lineales (LogReg) como no lineales (MLP, LSTM) con
un costo tan bajo lo convierte en la amenaza más relevante para sistemas de detección de
fraude financiero en producción.

### 5.3 La arquitectura determina la robustez frente a ataques de caja negra
El LSTM-Attention es el único modelo que resiste ataques sin gradiente. Esto sugiere que
arquitecturas recurrentes con mecanismos de atención crean espacios de decisión que son
inherentemente más difíciles de explorar mediante búsquedas heurísticas. En cambio, MLP,
LogReg y XGBoost —con superficies de decisión más regulares— son altamente susceptibles.

### 5.4 Alto AUC-ROC no implica robustez adversarial
XGBoost tiene el mejor AUC-ROC (0.966) pero es el más vulnerable a ataques de caja negra
(99.3%–100% de evasión). LSTM tiene el peor AUC-ROC (0.713) pero es el más robusto.
Esto demuestra que las métricas de clasificación estándar no son indicadores de seguridad
adversarial y que la evaluación de robustez requiere pruebas explícitas de ataque.

---

## 6. Gráficos generados

| Archivo | Descripción |
|---|---|
| `fig0_baseline_metrics.png` | AUC-ROC, Recall y F1 de los 4 modelos antes del ataque |
| `fig1_evasion_heatmap.png` | Heatmap modelos × ataques (rojo = alta evasión) |
| `fig2_cafa.png` | Scatter L₀ vs evasión — CaFA |
| `fig2_hopskipjump.png` | Scatter L₀ vs evasión — HopSkipJump |
| `fig2_boundaryattack.png` | Scatter L₀ vs evasión — BoundaryAttack |
| `fig2_squareattack.png` | Scatter L₀ vs evasión — SquareAttack |
| `fig3_recall_mlp.png` | Degradación recall bajo ataque — MLP |
| `fig3_recall_lstm-att.png` | Degradación recall bajo ataque — LSTM-Attention |
| `fig3_recall_log_reg.png` | Degradación recall bajo ataque — Logistic Regression |
| `fig3_recall_xgboost.png` | Degradación recall bajo ataque — XGBoost |
| `fig4_evasion_bars.png` | Barras agrupadas: todos los modelos y ataques |

---

## 7. Outputs de referencia

Los artefactos de cada ataque (`evaluations.json`, `X_adv.npy`, `attack.log`) se encuentran en:

```
outputs/amlworld_hi_smote/
  cafa_mlp/
  cafa_lstm_attention/
  cafa_logistic_regression/
  hop_skip_jump_mlp/
  hop_skip_jump_lstm_attention/
  hop_skip_jump_logistic_regression/
  hop_skip_jump_xgboost/
  boundary_attack_mlp/
  boundary_attack_lstm_attention/
  boundary_attack_logistic_regression/
  boundary_attack_xgboost/
  square_attack_mlp/
  square_attack_lstm_attention/
  square_attack_logistic_regression/
  square_attack_xgboost/
```
