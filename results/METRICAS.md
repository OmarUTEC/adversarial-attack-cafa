# Definición de Métricas

Métricas utilizadas en la evaluación experimental de modelos de clasificación y ataques adversariales para detección de fraude financiero.

---

## 1. Métricas de Clasificación del Modelo

Evalúan la capacidad del modelo para distinguir transacciones fraudulentas de legítimas **antes de cualquier ataque**.

---

### AUC-ROC
*(Area Under the Receiver Operating Characteristic Curve)*

Área bajo la curva que relaciona la Tasa de Verdaderos Positivos (TPR) contra la Tasa de Falsos Positivos (FPR) a distintos umbrales de decisión. Mide la capacidad del modelo para separar ambas clases de forma independiente al umbral elegido.

- **Rango:** 0 – 1
- **Óptimo:** 1.0 (separación perfecta)
- **Referencia aleatoria:** 0.5
- **Limitación:** En datasets con imbalance extremo puede ser optimista, ya que incluye los verdaderos negativos en su cálculo.

> `AUC-ROC = ∫ TPR d(FPR)`

---

### AUC-PR
*(Area Under the Precision-Recall Curve)*

Área bajo la curva que relaciona la Precisión contra el Recall a distintos umbrales. A diferencia del AUC-ROC, no incluye los verdaderos negativos en su cálculo, lo que la hace más informativa bajo imbalance extremo. Un modelo que predice siempre la clase mayoritaria tiene AUC-PR igual a la proporción de positivos en el dataset.

- **Rango:** 0 – 1
- **Óptimo:** 1.0
- **Referencia aleatoria:** igual a la prevalencia de fraude (ej. 0.001 para 0.1% de fraudes)
- **Ventaja:** Penaliza fuertemente los modelos que fallan en la clase minoritaria.

> `AUC-PR = ∫ Precisión d(Recall)`

---

### Recall
*(Sensibilidad / Tasa de Verdaderos Positivos)*

Proporción de transacciones fraudulentas que el modelo detecta correctamente sobre el total de fraudes existentes en el conjunto de prueba. Es la métrica operacional más crítica en detección de fraude: un Recall bajo implica fraudes que pasan desapercibidos y generan pérdidas directas.

- **Rango:** 0 – 1
- **Óptimo:** 1.0 (detecta todos los fraudes)

> `Recall = TP / (TP + FN)`

---

### F1-Score
*(F1)*

Media armónica entre la Precisión y el Recall. Penaliza los modelos que sacrifican una métrica en favor de la otra. Es útil cuando tanto los falsos positivos (falsas alarmas) como los falsos negativos (fraudes no detectados) tienen consecuencias operacionales relevantes.

- **Rango:** 0 – 1
- **Óptimo:** 1.0

> `F1 = 2 × (Precisión × Recall) / (Precisión + Recall)`

---

### TP — True Positives (Verdaderos Positivos)

Número de transacciones fraudulentas que el modelo identificó **correctamente** como fraude. Representa los fraudes efectivamente detectados.

---

### FP — False Positives (Falsos Positivos)

Número de transacciones **legítimas** que el modelo clasificó incorrectamente como fraude. En producción implica el bloqueo o revisión de operaciones válidas, generando fricción para el cliente y costo operativo.

---

### FN — False Negatives (Falsos Negativos)

Número de transacciones **fraudulentas** que el modelo clasificó incorrectamente como legítimas. Es el error más costoso en detección de fraude: cada FN representa una pérdida económica no interceptada.

---

### TN — True Negatives (Verdaderos Negativos)

Número de transacciones **legítimas** que el modelo identificó correctamente como legítimas. En datasets muy desbalanceados, este valor es muy alto y puede inflar métricas como la accuracy general.

---

### FNR — False Negative Rate (Tasa de Falsos Negativos)

Proporción de fraudes que el modelo **no detecta** sobre el total de fraudes existentes. Complemento directo del Recall. Es la métrica base para calcular el impacto de los ataques adversariales.

- **Rango:** 0 – 1
- **Óptimo:** 0.0 (ningún fraude escapa al detector)

> `FNR = FN / (FN + TP) = 1 − Recall`

---

### Tabla resumen — Métricas de clasificación

| Métrica | Rango | Óptimo | Sensible al imbalance |
|---------|:-----:|:------:|:---------------------:|
| AUC-ROC | 0 – 1 | 1.0 | No |
| AUC-PR  | 0 – 1 | 1.0 | Sí |
| Recall  | 0 – 1 | 1.0 | Sí |
| F1      | 0 – 1 | 1.0 | Sí |
| FNR     | 0 – 1 | 0.0 | Sí |
| TP / FP / FN / TN | ≥ 0 | — | Sí |

---

## 2. Métricas de Ataques Adversariales

Evalúan la efectividad del ataque y el costo de la perturbación aplicada **después de generar los ejemplos adversariales**.

---

### Tasa de Evasión
*(Evasion Rate)*

Proporción de ejemplos adversariales que logran ser clasificados como legítimos por el modelo objetivo tras ser perturbados. Mide directamente si el ataque funcionó.

- **Rango:** 0% – 100%
- **Óptimo para el defensor:** 0% (el modelo rechaza todos los intentos de evasión)
- **Óptimo para el atacante:** 100% (todos los fraudes perturbados pasan como legítimos)

> `Tasa de Evasión = muestras mal clasificadas post-ataque / total de muestras atacadas`

---

### ΔFNR — Delta False Negative Rate

Incremento en la tasa de falsos negativos provocado por el ataque. Mide el daño operacional real: cuántos fraudes adicionales deja de detectar el modelo como consecuencia directa de las perturbaciones adversariales.

- **Rango:** 0.0 – 1.0
- **Óptimo para el defensor:** 0.0 (el ataque no degrada la detección)
- **Valor máximo:** 1.0 (todos los fraudes que antes detectaba ahora escapan)

> `ΔFNR = FNR_post_ataque − FNR_pre_ataque`

---

### Recall post-ataque

Proporción de transacciones fraudulentas que el modelo **sigue detectando** correctamente después de recibir el ataque adversarial. Es el complemento directo del FNR post-ataque y representa la capacidad de detección residual del modelo bajo condiciones adversariales.

- **Rango:** 0% – 100%
- **Óptimo para el defensor:** 100% (el ataque no afecta la detección)

> `Recall_post = TP_post / (TP_post + FN_post)`

---

### L0 — Norma L0 (Costo de perturbación)

Número de características que el atacante modificó para generar el ejemplo adversarial. Mide la mínima cantidad de cambios necesarios para lograr la evasión. Un L0 bajo indica un ataque eficiente: pocas modificaciones bastan para engañar al modelo, lo que dificulta su detección por reglas de validación.

- **Rango:** 0 – número total de features
- **Óptimo para el defensor:** Alto (el atacante necesita modificar muchas features)
- **Óptimo para el atacante:** Bajo (pocas modificaciones logran la evasión)

> `L0 = |{ i : x_i ≠ x'_i }|`
>
> donde `x` es la muestra original y `x'` es la muestra adversarial.

**Estadísticas por muestra:**
- **L0 Media:** Promedio de características modificadas en todas las muestras que lograron evasión
- **L0 Mínimo:** Menor número de características modificadas en cualquier muestra exitosa (mejor caso atacante)
- **L0 Máximo:** Mayor número de características modificadas en cualquier muestra exitosa (peor caso atacante)
- **Desv. Estándar (σ):** Variabilidad en el número de cambios entre muestras

**Ejemplo de interpretación:**
- L0 Media = 5.81, Mín = 4, Máx = 8, σ = 1.2 → el ataque modifica consistentemente ~5-6 features
- L0 Media = 5.81, Mín = 1, Máx = 14, σ = 3.5 → el ataque es altamente variable (algunas muestras escapan con 1 cambio, otras necesitan 14)

---

### L∞ estandarizada — Norma L-infinito normalizada

Magnitud máxima de perturbación aplicada a cualquier característica, normalizada por el rango de valores válidos de esa característica. Mide qué tan "visible" o detectable es la perturbación dentro del espacio de valores posibles. Un valor cercano a 0 indica cambios imperceptibles dentro del rango normal de operación de la feature.

- **Rango:** 0 – ∞ (en la práctica 0 – 1 tras normalización)
- **Óptimo para el defensor:** Alto (los cambios son visibles y detectables)
- **Óptimo para el atacante:** Cercano a 0 (cambios imperceptibles)

> `L∞_std = max_i ( |x_i − x'_i| / rango_i )`

---

### Tabla resumen — Métricas adversariales

| Métrica | Rango | Óptimo (defensor) | Óptimo (atacante) |
|---------|:-----:|:-----------------:|:-----------------:|
| Tasa de Evasión | 0% – 100% | 0% | 100% |
| ΔFNR | 0.0 – 1.0 | 0.0 | 1.0 |
| Recall post-ataque | 0% – 100% | 100% | 0% |
| L0 (media) | 0 – N features | Alto | Bajo |
| L0 (distribución) | Mín–Máx, σ | Concentrado arriba | Variable abajo |
| L∞ estandarizada | 0 – ∞ | Alto | ~0 |

---

## 3. Estadísticas de L0 por Muestra — Experimentación Real

### Credit Card 2023 (29 features)

| Ataque | Modelo | Evasión | L0 Media | L0 Mín | L0 Máx | σ | Interpretación |
|--------|--------|---------|----------|--------|--------|---|----|
| **CaFA** | MLP | 52.9% | 11.94 | 10 | 14 | 1.1 | Consistente, modifica ~12 features |
| | Log. Reg. | 78.5% | 9.62 | 8 | 11 | 0.9 | Muy consistente, ~10 features |
| | LSTM-Att. | 23.7% | 0.60 | 0 | 2 | 0.7 | Extremadamente sigiloso |
| **HopSkipJump** | XGBoost | 99.95% | 29.97 | 29 | 30 | 0.2 | Casi todas las features (frontera lejana) |
| | MLP | 99.8% | 29.86 | 29 | 30 | 0.3 | Casi todas las features |
| | Log. Reg. | 99.2% | 29.83 | 29 | 30 | 0.2 | Casi todas las features |
| | LSTM-Att. | 77.9% | 29.98 | 29 | 30 | 0.2 | Casi todas (pero menor evasión) |
| **BoundaryAttack** | XGBoost | 99.95% | 29.85 | 29 | 30 | 0.2 | Casi todas las features |
| | MLP | 48.0% | 14.32 | 12 | 16 | 1.3 | Moderadamente variable |
| | Log. Reg. | 48.0% | 14.14 | 12 | 15 | 1.1 | Moderadamente variable |
| | LSTM-Att. | 52.1% | 20.95 | 18 | 22 | 1.4 | Mayor dispersión |
| **SquareAttack** | MLP | 100% | 11.92 | 10 | 13 | 0.8 | Consistente, muy sigiloso (zona roja) |
| | Log. Reg. | 100% | 15.58 | 14 | 17 | 1.0 | Consistente, moderadamente sigiloso |
| | LSTM-Att. | 22.6% | 0.42 | 0 | 1 | 0.5 | Extremadamente sigiloso pero poco efectivo |

### AMLworld HI-Small (15 features)

| Ataque | Modelo | Evasión | L0 Media | L0 Mín | L0 Máx | σ | Interpretación |
|--------|--------|---------|----------|--------|--------|---|----|
| **CaFA** | Log. Reg. | 100% | 3.55 | 3 | 4 | 0.4 | Muy consistente, modifica solo 3-4 features (zona roja) |
| **HopSkipJump** | XGBoost | 2.2% | 2.56 | 2 | 3 | 0.3 | Modifica pocas features pero sin éxito (único ataque resistido) |
| | Log. Reg. | 5.2% | 2.78 | 2 | 3 | 0.3 | Ídem |
| **BoundaryAttack** | XGBoost | 100% | 5.81 | 5 | 6 | 0.35 | Muy consistente, ~6 features (zona roja) |
| | Log. Reg. | 100% | 4.51 | 4 | 5 | 0.35 | Muy consistente, ~5 features (zona roja) |
| **SquareAttack** | XGBoost | 100% | 6.05 | 5 | 7 | 0.4 | Muy consistente, ~6 features (zona roja) |
| | Log. Reg. | 100% | 6.01 | 5 | 7 | 0.4 | Muy consistente, ~6 features (zona roja) |

---

### Interpretación de la Distribución L0

**Bajo σ (dispersión pequeña):**
- SquareAttack en Credit Card (σ=0.8) → consigue evasión con el MISMO número de features en casi todas las muestras
- CaFA en AMLworld (σ=0.4) → estrategia muy estable, predecible

**Alto σ (dispersión grande):**
- BoundaryAttack en Credit Card sobre MLP/LogReg (σ=1.3) → algunas muestras escapan con 12 features, otras necesitan 16
- Implica que ciertos puntos de fraude necesitan mayor perturbación que otros

**Interpretación para defensa:**
- **Baja σ:** Facil de detectar por reglas (p.ej. "alerta si 5-6 features cambian simultáneamente")
- **Alta σ:** Difícil de detectar (varía mucho, parecen cambios naturales)
