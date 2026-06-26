# Arquitectura Interna de los Ataques Adversariales
> Referencia para generación de diapositivas PPT

---

## 1. CaFA — Cost-aware Feasible Attacks (Caja Blanca)

**Tipo**: White-box | **Norma**: L0 | **Compatible con**: Log. Reg., MLP, LSTM-Att.

### Idea central
CaFA busca engañar al modelo modificando el **menor número posible de características**
(minimizar L0). Para lograrlo combina dos algoritmos anidados: TabCWL0 (exterior) y
TabPGD (interior).

---

### Flujo completo

```
INPUT
Muestra de fraude x (correctamente clasificada por el modelo)
        │
        ▼
Inicialización
  · Copiar x → x_adv
  · Máscara de features activas = todas habilitadas (1)
  · Inicializar perturbación aleatoria dentro de bola epsilon
        │
        ▼
╔══════════════════════════════════════════════════════╗
║       BUCLE EXTERNO — TabCWL0                        ║
║       Objetivo: minimizar L0 (features modificadas)  ║
║       Repetir hasta max_iter = 500                   ║
║                                                      ║
║   PASO 1 — Ejecutar TabPGD con la máscara actual     ║
║   ┌──────────────────────────────────────────────┐   ║
║   │   BUCLE INTERNO — TabPGD                     │   ║
║   │   Objetivo: cruzar la frontera de decisión   │   ║
║   │   Repetir hasta max_iter_tabpgd = 100 pasos  │   ║
║   │                                              │   ║
║   │   a) Calcular gradiente de la pérdida:       │   ║
║   │      grad = ∂Loss(x_adv, y) / ∂x_adv         │   ║
║   │      (requiere acceso a pesos del modelo)    │   ║
║   │                                              │   ║
║   │   b) Acumular gradiente: accum_grads += grad │   ║
║   │                                              │   ║
║   │   c) Calcular perturbación temporal:         │   ║
║   │      Δ = step_size × std_factors × sign(grad)│   ║
║   │                                              │   ║
║   │   d) Aplicar perturbación según tipo:        │   ║
║   │      · Continuas  → x_adv += Δ directo       │   ║
║   │      · Enteras    → x_adv += round(Δ)        │   ║
║   │      · Categóricas (one-hot)                 │   ║
║   │          cada 10 pasos usar accum_grads       │   ║
║   │          → cambiar a la categoría con        │   ║
║   │            mayor gradiente acumulado         │   ║
║   │                                              │   ║
║   │   e) Proyectar a bola epsilon:               │   ║
║   │      x_adv = clip(x_adv, x-eps, x+eps)      │   ║
║   │                                              │   ║
║   │   f) Respetar rangos válidos por feature     │   ║
║   │      x_adv = clip(x_adv, min_f, max_f)      │   ║
║   │                                              │   ║
║   │   g) Early stop: si modelo ya lo clasifica   │   ║
║   │      como legítimo → parar este loop         │   ║
║   └──────────────────────────────────────────────┘   ║
║                                                      ║
║   PASO 2 — Calcular score de importancia             ║
║      score_j = |grad_j × (x_adv_j - x_j)|           ║
║      (Carlini-Wagner: grad × perturbación actual)    ║
║      Feature con menor score = menos contribuye      ║
║                                                      ║
║   PASO 3 — Congelar feature menos importante         ║
║      máscara[feature_menos_util] = 0                 ║
║      (no se vuelve a tocar en iteraciones futuras)   ║
║                                                      ║
║   PASO 4 — Guardar mejor resultado                   ║
║      Si ataque exitoso Y L0 < L0_anterior:           ║
║        actualizar x_adv guardado                     ║
╚══════════════════════════════════════════════════════╝
        │
        ▼
OUTPUT
x_adv: ejemplo adversarial que evade el modelo
       con el menor número de features modificadas
```

### Parámetros clave
| Parámetro | Valor | Significado |
|---|---|---|
| `max_iter` | 500 | Iteraciones del bucle externo (TabCWL0) |
| `max_iter_tabpgd` | 100 | Pasos del bucle interno (TabPGD) |
| `step_size` | 0.0003 | Tamaño del paso por iteración |
| `eps` | 0.03 | Radio máximo de perturbación (bola L∞) |
| `perturb_categorical_each_steps` | 10 | Cada cuántos pasos se perturban categóricas |

---

## 2. HopSkipJump Attack — HSJ (Caja Negra)

**Tipo**: Black-box | **Norma**: L2 | **Compatible con**: todos los modelos

### Idea central
HSJ **nunca accede al interior del modelo**. Solo le pregunta: "¿esta transacción
es fraude o legítima?" (consulta binaria). Usando miles de estas preguntas, estima
la dirección del gradiente en la frontera de decisión y se va acercando poco a poco
al ejemplo original sin perder la evasión.

---

### Flujo completo

```
INPUT
Muestra de fraude x (correctamente clasificada por el modelo)
        │
        ▼
FASE 0 — Encontrar punto inicial adversarial
  Generar ruido aleatorio hasta encontrar x_init
  tal que modelo(x_init) = LEGÍTIMO
  (se parte desde el otro lado de la frontera)
        │
        ▼
╔══════════════════════════════════════════════════════╗
║       ITERACIÓN PRINCIPAL (max_iter = 50)            ║
║                                                      ║
║   FASE 1 — Estimar dirección del gradiente           ║
║            en la frontera de decisión                ║
║   ┌──────────────────────────────────────────────┐   ║
║   │  Lanzar N consultas aleatorias               │   ║
║   │  (N aumenta cada iteración: init_eval=100)   │   ║
║   │                                              │   ║
║   │  Para cada consulta:                         │   ║
║   │    · generar dirección aleatoria u           │   ║
║   │    · evaluar modelo(x_frontera + δ·u)        │   ║
║   │    · si cambia de clase → u apunta           │   ║
║   │      hacia el interior adversarial           │   ║
║   │                                              │   ║
║   │  Promediar todas las u que cambiaron clase   │   ║
║   │  → estimación del gradiente en frontera      │   ║
║   └──────────────────────────────────────────────┘   ║
║                                                      ║
║   FASE 2 — Paso hacia el ejemplo original            ║
║   ┌──────────────────────────────────────────────┐   ║
║   │  Interpolar entre x_adv y x original:        │   ║
║   │    x_nuevo = x_adv + α·(x - x_adv)           │   ║
║   │                                              │   ║
║   │  Buscar α máximo tal que:                    │   ║
║   │    modelo(x_nuevo) siga siendo LEGÍTIMO      │   ║
║   │  (búsqueda binaria sobre α)                  │   ║
║   │                                              │   ║
║   │  Actualizar x_adv ← x_nuevo                 │   ║
║   └──────────────────────────────────────────────┘   ║
║                                                      ║
║   PROYECCIÓN TABULAR (adaptación propia)             ║
║   · Clip a rangos válidos por feature                ║
║   · Redondear features enteras                       ║
║   · Proyectar one-hot encoding                       ║
╚══════════════════════════════════════════════════════╝
        │
        ▼
OUTPUT
x_adv: más cercano a x original que en la iteración anterior
       sigue siendo clasificado como LEGÍTIMO
```

### Parámetros clave
| Parámetro | Valor | Significado |
|---|---|---|
| `max_iter` | 50 | Número de iteraciones principales |
| `max_eval` | 10,000 | Total máximo de consultas al modelo |
| `init_eval` | 100 | Consultas en la primera iteración |
| `norm` | L2 | Distancia minimizada |

---

## 3. Boundary Attack (Caja Negra)

**Tipo**: Black-box | **Norma**: L2 | **Compatible con**: todos los modelos

### Idea central
Boundary Attack imagina la frontera de decisión como una "superficie" y camina sobre
ella en dos movimientos alternados: uno que lo mantiene sobre la superficie (ortogonal)
y otro que lo acerca al ejemplo original (hacia la fuente). Es más robusto que HSJ
en espacios con fronteras complejas o irregulares.

---

### Flujo completo

```
INPUT
Muestra de fraude x (correctamente clasificada por el modelo)
        │
        ▼
INICIALIZACIÓN
  Generar x_adv con mucho ruido aleatorio
  Verificar que modelo(x_adv) = LEGÍTIMO
  (punto de partida: lejos de x, con gran perturbación)
        │
        ▼
╔══════════════════════════════════════════════════════╗
║       ITERACIÓN PRINCIPAL (max_iter = 100)           ║
║                                                      ║
║   PASO A — Movimiento ORTOGONAL (delta)              ║
║   ┌──────────────────────────────────────────────┐   ║
║   │  Objetivo: moverse SOBRE la frontera         │   ║
║   │            sin acercarse ni alejarse         │   ║
║   │                                              │   ║
║   │  Generar perturbación η perpendicular        │   ║
║   │  a la dirección (x_adv → x_original)        │   ║
║   │                                              │   ║
║   │  x_candidate = x_adv + delta · η            │   ║
║   │                                              │   ║
║   │  Si modelo(x_candidate) = LEGÍTIMO:          │   ║
║   │    → aceptar (nos quedamos en zona de evasión)│  ║
║   │  Si modelo(x_candidate) = FRAUDE:            │   ║
║   │    → rechazar (cruzamos la frontera)         │   ║
║   └──────────────────────────────────────────────┘   ║
║                                                      ║
║   PASO B — Movimiento hacia la FUENTE (epsilon)      ║
║   ┌──────────────────────────────────────────────┐   ║
║   │  Objetivo: acercarse a x original            │   ║
║   │                                              │   ║
║   │  x_candidate = x_adv + eps · (x - x_adv)    │   ║
║   │  (interpolar un pequeño paso hacia x)        │   ║
║   │                                              │   ║
║   │  Si modelo(x_candidate) = LEGÍTIMO:          │   ║
║   │    → aceptar (más cerca sin perder evasión)  │   ║
║   └──────────────────────────────────────────────┘   ║
║                                                      ║
║   ADAPTACIÓN DE TAMAÑOS DE PASO                      ║
║   ┌──────────────────────────────────────────────┐   ║
║   │  tasa_éxito_ortogonal > 50%                  │   ║
║   │    → aumentar delta (explorar más)           │   ║
║   │  tasa_éxito_ortogonal < 50%                  │   ║
║   │    → reducir delta (moverse con cuidado)     │   ║
║   │  factor de ajuste: step_adapt = 0.667        │   ║
║   └──────────────────────────────────────────────┘   ║
║                                                      ║
║   PROYECCIÓN TABULAR (adaptación propia)             ║
║   · Clip a rangos válidos por feature                ║
║   · Redondear features enteras                       ║
║   · Proyectar one-hot encoding                       ║
╚══════════════════════════════════════════════════════╝
        │
        ▼
OUTPUT
x_adv: perturbación reducida progresivamente
       sigue siendo clasificado como LEGÍTIMO
```

### Parámetros clave
| Parámetro | Valor | Significado |
|---|---|---|
| `max_iter` | 100 | Iteraciones principales |
| `delta` | 0.01 | Tamaño inicial del paso ortogonal |
| `epsilon` | 0.01 | Tamaño inicial del paso hacia la fuente |
| `step_adapt` | 0.667 | Factor de ajuste de paso (< 1 = conservador) |
| `num_trial` | 25 | Intentos por iteración en paso ortogonal |

---

## 4. Square Attack (Caja Negra)

**Tipo**: Black-box | **Norma**: L∞ | **Compatible con**: todos los modelos

### Idea central
Square Attack es el más simple de los tres ataques de caja negra. No estima gradientes
ni camina sobre fronteras. Simplemente **propone cambios aleatorios en subconjuntos de
features** y los acepta si mejoran la evasión. Su nombre viene de que en imágenes aplica
perturbaciones en forma de cuadrado; en datos tabulares aplica cambios en grupos de columnas.

---

### Flujo completo

```
INPUT
Muestra de fraude x (correctamente clasificada por el modelo)
        │
        ▼
INICIALIZACIÓN
  Generar perturbación inicial aleatoria v
  dentro del presupuesto L∞: |v_j| ≤ eps para todo j
  x_adv = x + v
  score_0 = P(legítimo | x_adv)   ← probabilidad inicial
        │
        ▼
╔══════════════════════════════════════════════════════╗
║       ITERACIÓN (max_iter = 100)                     ║
║                                                      ║
║   PASO 1 — Seleccionar subconjunto de features       ║
║   ┌──────────────────────────────────────────────┐   ║
║   │  Tamaño del grupo: p × n_features            │   ║
║   │  p = p_init (empieza grande, decrece)        │   ║
║   │  Elegir columnas aleatoriamente              │   ║
║   └──────────────────────────────────────────────┘   ║
║                                                      ║
║   PASO 2 — Proponer cambio en ese subconjunto        ║
║   ┌──────────────────────────────────────────────┐   ║
║   │  Para cada feature j en el subconjunto:      │   ║
║   │    · si v_j = +eps → cambiar a -eps          │   ║
║   │    · si v_j = -eps → cambiar a +eps          │   ║
║   │    (alternancia para explorar ambos lados)   │   ║
║   │                                              │   ║
║   │  Aplicar proyección tabular:                 │   ║
║   │    · Clip a rangos válidos por feature       │   ║
║   │    · Redondear enteras                       │   ║
║   │    · Proyectar one-hot                       │   ║
║   └──────────────────────────────────────────────┘   ║
║                                                      ║
║   PASO 3 — Evaluar con el modelo                     ║
║   ┌──────────────────────────────────────────────┐   ║
║   │  score_nuevo = P(legítimo | x_adv_propuesto) │   ║
║   │                                              │   ║
║   │  Si score_nuevo > score_actual:              │   ║
║   │    → ACEPTAR: x_adv ← x_adv_propuesto       │   ║
║   │    → score_actual ← score_nuevo              │   ║
║   │                                              │   ║
║   │  Si score_nuevo ≤ score_actual:              │   ║
║   │    → RECHAZAR: mantener x_adv anterior       │   ║
║   └──────────────────────────────────────────────┘   ║
║                                                      ║
║   PASO 4 — Reducir tamaño del grupo                  ║
║   p decrece con la iteración:                        ║
║   más iteraciones → grupos más pequeños              ║
║   → refinamiento fino al final                       ║
╚══════════════════════════════════════════════════════╝
        │
        ▼
Verificar early stop:
  Si modelo(x_adv) = LEGÍTIMO → ataque exitoso, parar
        │
        ▼
OUTPUT
x_adv: ejemplo adversarial encontrado por búsqueda aleatoria
       perturbación L∞ ≤ eps en todas las features
```

### Parámetros clave
| Parámetro | Valor | Significado |
|---|---|---|
| `max_iter` | 100 | Total de propuestas de cambio |
| `eps` | depende del dataset | Presupuesto máximo L∞ |
| `p_init` | 0.8 | Proporción inicial de features a perturbar |
| `nb_restarts` | 1 | Reinicios aleatorios si no converge |

---

## Tabla comparativa completa

```
┌──────────────────┬───────────────┬──────────────┬──────────────┬──────────────┐
│                  │     CaFA      │ HopSkipJump  │   Boundary   │    Square    │
├──────────────────┼───────────────┼──────────────┼──────────────┼──────────────┤
│ Tipo             │ Caja blanca   │ Caja negra   │ Caja negra   │ Caja negra   │
│ Acceso modelo    │ Gradientes    │ Solo decisión│ Solo decisión│ Probabilidad │
│ Mecanismo        │ PGD + máscara │ Est. gradiente│ Pasos ortog. │ Búsqueda     │
│                  │ CW (L0)       │ en frontera  │ + hacia fuent│ aleatoria    │
│ Norma objetivo   │ L0 (mínimas   │ L2           │ L2           │ L∞           │
│                  │ features)     │              │              │              │
│ Consultas modelo │ Muchas        │ Muchas       │ Moderadas    │ Pocas        │
│                  │ (con grads)   │ (~10,000)    │ (~2,500)     │ (~100)       │
│ Punto de inicio  │ x original    │ Punto lejano │ Punto lejano │ x + ruido    │
│                  │               │ adversarial  │ adversarial  │ pequeño      │
│ XGBoost          │ NO            │ SI           │ SI           │ SI           │
│ Redes neuronales │ SI            │ SI           │ SI           │ SI           │
├──────────────────┼───────────────┼──────────────┼──────────────┼──────────────┤
│ Fortaleza        │ Mínimo cambio │ Robusto y    │ Muy robusto  │ Muy rápido,  │
│                  │ posible (L0)  │ convergente  │ en fronteras │ pocas consul.│
│ Debilidad        │ Solo modelos  │ Lento        │ Muy lento    │ No minimiza  │
│                  │ diferenciables│ (10k consult)│ (conv. lenta)│ L0           │
└──────────────────┴───────────────┴──────────────┴──────────────┴──────────────┘
```

---

## Notas para las diapositivas PPT

- **Colores sugeridos por tipo**:
  - Caja Blanca (CaFA): rojo/naranja — acceso total al modelo
  - Caja Negra (HSJ, Boundary, Square): azul — solo consultas externas

- **Icono diferenciador**:
  - CaFA: candado abierto (acceso interno)
  - HSJ / Boundary / Square: candado cerrado (solo preguntas)

- **Flujo común a todos**:
  `Muestra de fraude → Perturbación → Verificar evasión → Ajustar → Salida adversarial`

- **Lo que hace diferente a cada uno**:
  - CaFA: *cómo* perturba (con gradiente, feature por feature)
  - HSJ: *cómo* estima la dirección (preguntas aleatorias en la frontera)
  - Boundary: *cómo* se mueve (dos pasos alternados sobre la superficie)
  - Square: *cómo* elige qué cambiar (subconjuntos aleatorios, acepta/rechaza)
