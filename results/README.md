# Resultados Experimentales — Robustez Adversarial en Detección de Fraude

Experimentos sobre dos datasets de detección de fraude financiero:
- **Credit Card 2023** — 568,630 transacciones, balance 50/50, 29 features
- **AMLworld HI-Small (IBM)** — 5,078,208 transacciones, imbalance 1:993, 15 features

Modelos evaluados: XGBoost, MLP, Logistic Regression, LSTM-Attention  
Ataques evaluados: CaFA, SimBA, HopSkipJump, BoundaryAttack, SquareAttack

---

## Conclusiones por Métrica

### Tasa de Evasión

En **Credit Card**, HopSkipJump y SquareAttack alcanzan evasión cercana al 100% en XGBoost, MLP y Logistic Regression. LSTM-Attention es el modelo más resistente en este dataset (evasión máxima del 52% con BoundaryAttack). CaFA y SimBA muestran efectividad moderada sobre modelos lineales pero fracasan contra LSTM.

En **AMLworld HI**, la tasa de evasión es extrema: BoundaryAttack, SquareAttack, CaFA y SimBA logran 100% de evasión con pocos intentos. La excepción es HopSkipJump, que solo alcanza 2.2% sobre XGBoost y 5.2% sobre Logistic Regression — el único ataque que los modelos resisten.

**Conclusión:** Ningún modelo evaluado es robusto ante todos los ataques. La tasa de evasión depende más de la estrategia del ataque que de la arquitectura del modelo.

---

### ΔFNR (Delta False Negative Rate)

En **Credit Card**, el daño operacional varía según el ataque. HopSkipJump produce ΔFNR cercano a +1.0 en todos los modelos. BoundaryAttack y CaFA generan ΔFNR de 0.47–0.79. SimBA tiene impacto bajo (ΔFNR < 0.05) salvo en Logistic Regression.

En **AMLworld HI**, el ΔFNR es binario: o el ataque falla completamente (HopSkipJump, ΔFNR ≈ 0.02–0.05) o produce ΔFNR = +1.0 (evasión total). No hay término medio, lo que evidencia la fragilidad de los modelos ante perturbaciones pequeñas en este dominio.

**Conclusión:** En detección de lavado de dinero (AMLworld), un ataque exitoso convierte el 100% de los fraudes detectados en no detectados. El impacto operacional es catastrófico — no gradual.

---

### Recall post-ataque

En **Credit Card**, el recall post-ataque mínimo observado fue 0% (Square y HopSkipJump vs MLP/LogReg). LSTM-Attention conserva recall del 47.8% incluso bajo BoundaryAttack, siendo el modelo con mayor recall residual.

En **AMLworld HI**, el recall post-ataque colapsa a 0% en todos los ataques exitosos. Solo HopSkipJump preserva recall del 97.8% (XGBoost) y 94.8% (Logistic Regression). XGBoost es el único modelo con recall residual significativo.

**Conclusión:** XGBoost en AMLworld y LSTM en Credit Card son los modelos con mayor recall residual tras el ataque. En ambos casos HopSkipJump es el ataque con menor impacto sobre el recall.

---

### L0 — Costo de perturbación (features modificadas)

En **Credit Card** (29 features), los ataques más costosos son HopSkipJump y BoundaryAttack, que modifican 20–30 features. CaFA y SimBA son los más eficientes, con 1–12 features modificadas. SquareAttack logra evasión total con 12–16 features.

En **AMLworld HI** (15 features), los ataques son notablemente más eficientes: SimBA elude el modelo con solo 2.39 features modificadas, CaFA con 3.55, BoundaryAttack con 4.51–5.81, y SquareAttack con ~6. HopSkipJump modifica pocas features (~2.6) pero rara vez logra la evasión.

**Conclusión:** En AMLworld, un atacante necesita modificar entre 2 y 6 características de una transacción para evadir completamente la detección. Esto es operacionalmente viable — equivale a ajustar el monto, la moneda y el formato de pago de una transferencia.

---

### L∞ estandarizada — Magnitud de los cambios

En **Credit Card**, las perturbaciones de HopSkipJump son las más grandes (L∞ hasta 88.76 en XGBoost), lo que indica cambios drásticos en los valores de las features. CaFA y SimBA producen perturbaciones mínimas (L∞ < 0.03), difícilmente detectables por reglas de validación.

En **AMLworld HI**, SimBA produce perturbaciones prácticamente imperceptibles (L∞ = 1.9×10⁻⁷), mientras BoundaryAttack sobre XGBoost requiere cambios más visibles (L∞ = 0.29). SquareAttack y HopSkipJump generan perturbaciones mínimas a pesar de modificar varias features.

**Conclusión:** Los ataques de gradiente (CaFA, SimBA) generan perturbaciones pequeñas y difíciles de detectar. Los ataques de frontera (HopSkipJump, BoundaryAttack) requieren cambios mayores pero son igualmente efectivos. La L∞ baja de SimBA lo convierte en el ataque más sigiloso del conjunto evaluado.

---

## Conclusiones por Dataset

### Credit Card 2023

El dataset balanceado permite que todos los modelos alcancen métricas de clasificación excelentes (AUC-ROC ≥ 0.82, Recall ≥ 94%). Sin embargo, esta alta capacidad clasificadora no implica robustez adversarial: todos los modelos son vulnerables a al menos tres de los cinco ataques evaluados. LSTM-Attention es el modelo más robusto relativamente, resistiendo mejor CaFA y SimBA debido a sus fronteras de decisión no lineales. XGBoost, a pesar de tener el mejor clasificador, es el más vulnerable a HopSkipJump y BoundaryAttack por la regularidad de sus fronteras de decisión en este espacio de features.

### AMLworld HI-Small (IBM)

El imbalance extremo (1:993) es el factor determinante en este dataset. MLP y LSTM no logran detectar fraudes (Recall = 0%), lo que los hace inútiles tanto para clasificación como para evaluación adversarial. Logistic Regression detecta una fracción pequeña (10.7%) pero es trivialmente atacable. XGBoost es el único modelo operacionalmente viable (Recall = 33.9%, AUC-ROC = 0.966) y el único con resultados adversariales significativos. Aun así, BoundaryAttack y SquareAttack logran 100% de evasión sobre XGBoost con ~6 features modificadas. Solo HopSkipJump es resistido, con apenas 2.2% de evasión.

---

## Hallazgo Principal

> **El imbalance extremo de datos es más determinante para la vulnerabilidad adversarial que la arquitectura del modelo.** En Credit Card (balanceado), todos los modelos detectan fraudes y todos son atacables. En AMLworld (imbalanceado), solo XGBoost detecta fraudes y los ataques son más eficientes con menor costo de perturbación. La robustez adversarial no puede evaluarse independientemente de la calidad del clasificador base.

---

## Archivos

| Archivo | Descripción |
|---------|-------------|
| `tabla1_baseline.png` | Métricas de clasificación por modelo y dataset |
| `tabla2_evasion.png` | Tasa de evasión por ataque, modelo y dataset |
| `tabla3_l0_cost.png` | Costo L0 por ataque, modelo y dataset |
| `tabla_metricas_ataques.png` | Definición de métricas adversariales |
| `tabla_metricas_valores.png` | Métricas con valores experimentales (mejor/peor caso) |
| `amlworld_hi/README.md` | Análisis detallado AMLworld HI-Small |
| `amlworld_hi/fig0_baseline_metrics.png` | Métricas basales AMLworld |
| `amlworld_hi/fig1_evasion_rate.png` | Tasa de evasión AMLworld |
| `amlworld_hi/fig2_cost_vs_evasion.png` | Costo vs evasión AMLworld |
| `amlworld_hi/fig3_recall_degradation.png` | Degradación del recall AMLworld |
