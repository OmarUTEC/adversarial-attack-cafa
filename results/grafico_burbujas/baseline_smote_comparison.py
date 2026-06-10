"""
Gráfico comparativo: métricas baseline SIN SMOTE vs CON SMOTE — AMLworld HI-Small
Modelos: XGBoost, MLP, Log. Reg.
Métricas: AUC-ROC, Recall, F1
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Datos ────────────────────────────────────────────────────
MODELS = ['XGBoost', 'MLP', 'Log. Reg.']

SIN_SMOTE = {
    'AUC-ROC': [0.9660, 0.9131, 0.5672],
    'Recall':  [0.3395, 0.0000, 0.1067],
    'F1':      [0.2032, 0.0000, 0.0103],
}

CON_SMOTE = {
    'AUC-ROC': [0.9661, 0.9351, 0.8436],
    'Recall':  [0.7422, 0.8234, 0.7859],
    'F1':      [0.0366, 0.0184, 0.0095],
}

METRICS   = ['AUC-ROC', 'Recall', 'F1']
COLORS    = {'Sin SMOTE': '#94A3B8', 'Con SMOTE': '#16A34A'}

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.25,
    'grid.linestyle':    '--',
    'figure.dpi':        160,
})

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

bar_w  = 0.32
x      = np.arange(len(MODELS))

for ax, metric in zip(axes, METRICS):
    v_sin  = SIN_SMOTE[metric]
    v_con  = CON_SMOTE[metric]

    b1 = ax.bar(x - bar_w/2, v_sin, width=bar_w,
                color=COLORS['Sin SMOTE'], edgecolor='white',
                linewidth=0.6, label='Sin SMOTE', alpha=0.9)
    b2 = ax.bar(x + bar_w/2, v_con, width=bar_w,
                color=COLORS['Con SMOTE'], edgecolor='white',
                linewidth=0.6, label='Con SMOTE', alpha=0.9)

    # Valores sobre las barras
    for bar, val in zip(b1, v_sin):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                    f'{val:.2f}', ha='center', va='bottom',
                    fontsize=8, color='#475569', fontweight='bold')

    for bar, val in zip(b2, v_con):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                    f'{val:.2f}', ha='center', va='bottom',
                    fontsize=8, color='#15803D', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda y, _: f'{y:.0%}' if metric != 'AUC-ROC' else f'{y:.2f}'))
    ax.set_title(metric, fontsize=13, fontweight='bold', pad=10)
    ax.tick_params(axis='x', length=0)

# Leyenda global
from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor=COLORS['Sin SMOTE'], label='Sin SMOTE (original)'),
    Patch(facecolor=COLORS['Con SMOTE'], label='Con SMOTE (1:10)'),
]
fig.legend(handles=legend_handles, loc='lower center',
           fontsize=10, framealpha=0.95,
           bbox_to_anchor=(0.5, -0.08), ncol=2)

fig.suptitle(
    'Métricas Baseline — Sin SMOTE vs Con SMOTE\nAMLworld HI-Small (IBM)',
    fontsize=13, fontweight='bold', color='#B45309', y=1.03)

fig.tight_layout(rect=[0, 0.06, 1, 1])
path = os.path.join(OUT_DIR, 'baseline_smote_comparison.png')
fig.savefig(path, bbox_inches='tight', dpi=160)
plt.close(fig)
print(f'Saved: {path}')
