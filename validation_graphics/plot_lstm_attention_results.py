"""
Adversarial Attack Results — LSTM-Attention / Credit Card Fraud 2023
Generates:
  1. Line chart  — cumulative evasions vs L-inf budget (similar to Biarese et al.)
  2. Bar chart   — ASR / L0 / L-inf per attack
  3. Summary table printed to console + saved as figure
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
BASE = os.path.join(os.path.dirname(__file__), "..", "outputs")
SAVE = os.path.dirname(__file__)

ATTACKS = {
    "CaFA":           "cafa_lstm_attention",
    "HopSkipJump":    "hop_skip_jump_lstm_attention",
    "Boundary Attack":"boundary_attack_lstm_attention",
    "SimBA":          "simba_lstm_attention",
    "Square Attack":  "square_attack_lstm_attention",
}

COLORS = {
    "CaFA":            "#9467bd",
    "HopSkipJump":     "#1f77b4",
    "Boundary Attack": "#ff7f0e",
    "SimBA":           "#d62728",
    "Square Attack":   "#2ca02c",
}

MARKERS = {
    "CaFA":            "D",
    "HopSkipJump":     "o",
    "Boundary Attack": "s",
    "SimBA":           "x",
    "Square Attack":   "^",
}

# ── load data ─────────────────────────────────────────────────────────────────
def load_attack(name, folder):
    path = os.path.join(BASE, folder)
    X     = np.load(os.path.join(path, "X.npy"))
    X_adv = np.load(os.path.join(path, "X_adv.npy"))
    Y     = np.load(os.path.join(path, "Y.npy"))
    with open(os.path.join(path, "evaluations.json")) as f:
        evals = json.load(f)
    return X, X_adv, Y, evals

data = {}
for name, folder in ATTACKS.items():
    try:
        data[name] = load_attack(name, folder)
        print(f"  Loaded {name}")
    except FileNotFoundError as e:
        print(f"  SKIP {name}: {e}")

# ── compute per-sample L-inf (absolute, raw feature space) ───────────────────
def linf_per_sample(X, X_adv):
    return np.max(np.abs(X_adv - X), axis=1)

# ── 1. LINE CHART — cumulative evasions vs L-inf budget ──────────────────────
fig1, ax1 = plt.subplots(figsize=(9, 5.5))

budgets = np.linspace(0, 15, 300)

for name, (X, X_adv, Y, evals) in data.items():
    key = [k for k in evals if k != "before-attack"][0]
    linf = linf_per_sample(X, X_adv)

    asr_total = evals[key]["is_misclassified_rate"]
    n = len(X)

    sorted_linf = np.sort(linf)
    n_evaded = int(round(asr_total * n))
    evaded_linf = sorted_linf[:n_evaded]

    cum_evasions = [np.sum(evaded_linf <= b) for b in budgets]

    ax1.plot(budgets, cum_evasions,
             label=name,
             color=COLORS[name],
             marker=MARKERS[name],
             markevery=30,
             linewidth=2.2,
             markersize=7)

ax1.set_xlabel("L∞ Budget (espacio de features)", fontsize=12)
ax1.set_ylabel("Evasiones acumuladas", fontsize=12)
ax1.set_title("LSTM-Attention — Evasiones acumuladas vs presupuesto L∞\nCreditcard Fraud 2023 (n=2000)", fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.yaxis.set_major_locator(mticker.MultipleLocator(200))
ax1.set_xlim(left=0)
ax1.set_ylim(bottom=0)
plt.tight_layout()
fig1.savefig(os.path.join(SAVE, "lstm_attention_cumulative_evasions.png"), dpi=150)
print("\nGuardado: lstm_attention_cumulative_evasions.png")

# ── 2. BAR CHART — ASR / L0 / L-inf estandarizado ───────────────────────────
attack_names = list(data.keys())
asr_vals, l0_vals, linf_vals = [], [], []

for name, (X, X_adv, Y, evals) in data.items():
    key = [k for k in evals if k != "before-attack"][0]
    asr_vals.append(evals[key]["is_misclassified_rate"] * 100)
    l0_vals.append(evals[key]["l0_costs_on_mis_mean"])
    linf_vals.append(evals[key]["stand_linf_costs_on_mis_mean"])

x = np.arange(len(attack_names))
width = 0.25

fig2, ax2 = plt.subplots(figsize=(10, 5.5))
bars1 = ax2.bar(x - width, asr_vals,  width, label="ASR (%)",              color="#1f77b4", alpha=0.85)
bars2 = ax2.bar(x,         l0_vals,   width, label="L0 (features mod.)",   color="#ff7f0e", alpha=0.85)
bars3 = ax2.bar(x + width, linf_vals, width, label="L∞ estd. (evasiones)", color="#2ca02c", alpha=0.85)

for bar in [*bars1, *bars2, *bars3]:
    h = bar.get_height()
    ax2.annotate(f"{h:.1f}",
                 xy=(bar.get_x() + bar.get_width() / 2, h),
                 xytext=(0, 3), textcoords="offset points",
                 ha="center", va="bottom", fontsize=8.5)

ax2.set_xticks(x)
ax2.set_xticklabels(attack_names, fontsize=11)
ax2.set_ylabel("Valor de la métrica", fontsize=12)
ax2.set_title("LSTM-Attention — Comparativa de métricas por ataque\nCreditcard Fraud 2023 (n=2000)", fontsize=13)
ax2.legend(fontsize=11)
ax2.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
fig2.savefig(os.path.join(SAVE, "lstm_attention_metrics_comparison.png"), dpi=150)
print("Guardado: lstm_attention_metrics_comparison.png")

# ── 3. SUMMARY TABLE ─────────────────────────────────────────────────────────
rows = []
for name, (X, X_adv, Y, evals) in data.items():
    key = [k for k in evals if k != "before-attack"][0]
    e = evals[key]
    rows.append({
        "Ataque":                 name,
        "ASR (%)":                f"{e['is_misclassified_rate']*100:.2f}",
        "L0 medio":               f"{e['l0_costs_mean']:.2f}",
        "L0 (evasiones)":         f"{e['l0_costs_on_mis_mean']:.2f}",
        "L∞ estd. medio":         f"{e['stand_linf_costs_mean']:.4f}",
        "L∞ estd. (evasiones)":   f"{e['stand_linf_costs_on_mis_mean']:.4f}",
    })

df = pd.DataFrame(rows)
print("\n── Tabla de resultados LSTM-Attention ───────────────────────────────")
print(df.to_string(index=False))

# ── 3b. TABLE AS FIGURE ───────────────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(12, 2.6))
ax3.axis("off")

tbl = ax3.table(
    cellText=df.values.tolist(),
    colLabels=list(df.columns),
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.6)

for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#f2f2f2")
    cell.set_edgecolor("#cccccc")

ax3.set_title("LSTM-Attention — Credit Card Fraud 2023  |  n = 2000 muestras",
              fontsize=12, fontweight="bold", pad=12)
plt.tight_layout()
fig3.savefig(os.path.join(SAVE, "lstm_attention_results_table.png"), dpi=150, bbox_inches="tight")
print("Guardado: lstm_attention_results_table.png")

plt.show()
