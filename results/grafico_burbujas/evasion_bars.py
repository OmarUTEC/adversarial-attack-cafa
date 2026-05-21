"""
Gráfico de barras agrupadas — Tasa de Evasión por Modelo y Ataque
Una imagen con 2 subplots: Credit Card | AMLworld HI
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

NAN = float('nan')

MODEL_COLOR = {
    'XGBoost':   '#2563EB',
    'MLP':       '#16A34A',
    'Log. Reg.': '#D97706',
    'LSTM-Att.': '#9333EA',
}
MODELS  = ['XGBoost', 'MLP', 'Log. Reg.', 'LSTM-Att.']
ATTACKS = ['CaFA', 'HopSkipJump', 'BoundaryAttack', 'SquareAttack']
ATTACK_LABELS = ['CaFA', 'HopSkipJump', 'Boundary\nAttack', 'Square\nAttack']

EVASION = {
    'Credit Card': {
        'CaFA':           {'XGBoost': NAN,    'MLP': 52.9, 'Log. Reg.': 78.5, 'LSTM-Att.': 23.7},
        'HopSkipJump':    {'XGBoost': 99.95,  'MLP': 99.8, 'Log. Reg.': 99.2, 'LSTM-Att.': 77.9},
        'BoundaryAttack': {'XGBoost': 99.95,  'MLP': 48.0, 'Log. Reg.': 48.0, 'LSTM-Att.': 52.1},
        'SquareAttack':   {'XGBoost': NAN,    'MLP': 100.0,'Log. Reg.':100.0, 'LSTM-Att.': 22.6},
    },
    'AMLworld HI': {
        'CaFA':           {'XGBoost': NAN,   'MLP': NAN,  'Log. Reg.': 100.0, 'LSTM-Att.': NAN},
        'HopSkipJump':    {'XGBoost': 2.2,   'MLP': NAN,  'Log. Reg.':   5.2, 'LSTM-Att.': NAN},
        'BoundaryAttack': {'XGBoost': 100.0, 'MLP': NAN,  'Log. Reg.': 100.0, 'LSTM-Att.': NAN},
        'SquareAttack':   {'XGBoost': 100.0, 'MLP': NAN,  'Log. Reg.': 100.0, 'LSTM-Att.': NAN},
    },
}

plt.rcParams.update({
    'font.family':    'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':      True,
    'grid.alpha':     0.25,
    'grid.linestyle': '--',
    'figure.dpi':     160,
})

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

n_attacks = len(ATTACKS)
n_models  = len(MODELS)
bar_w     = 0.18
group_gap = 0.9   # espacio entre grupos de ataques

for ax, ds, ds_color in zip(axes,
                             ['Credit Card', 'AMLworld HI'],
                             ['#1D4ED8',     '#B45309']):

    for m_idx, model in enumerate(MODELS):
        xs     = []
        heights = []
        colors  = []

        for a_idx, atk in enumerate(ATTACKS):
            val = EVASION[ds][atk][model]
            x   = a_idx * group_gap + (m_idx - (n_models - 1) / 2) * bar_w
            if not np.isnan(val):
                xs.append(x)
                heights.append(val)
                colors.append(MODEL_COLOR[model])

        if xs:
            bars = ax.bar(xs, heights,
                          width=bar_w * 0.92,
                          color=MODEL_COLOR[model],
                          edgecolor='white',
                          linewidth=0.6,
                          zorder=3,
                          alpha=0.88)
            # Valores sobre las barras
            for bar, h in zip(bars, heights):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        h + 1.5,
                        f'{h:.0f}%',
                        ha='center', va='bottom',
                        fontsize=7.2,
                        color=MODEL_COLOR[model],
                        fontweight='bold')

    # Marcas "N/A" debajo del eje para combinaciones incompatibles
    for a_idx, atk in enumerate(ATTACKS):
        for m_idx, model in enumerate(MODELS):
            val = EVASION[ds][atk][model]
            if np.isnan(val):
                x = a_idx * group_gap + (m_idx - (n_models - 1) / 2) * bar_w
                ax.text(x, -7, 'N/A',
                        ha='center', va='top',
                        fontsize=6.0,
                        color=MODEL_COLOR[model],
                        alpha=0.55,
                        rotation=90)

    # Eje X con etiquetas de ataques
    xtick_pos = [i * group_gap for i in range(n_attacks)]
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(ATTACK_LABELS, fontsize=10, fontweight='bold')

    ax.set_ylim(-15, 115)
    ax.set_ylabel('Tasa de Evasión (%)', fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%' if y >= 0 else ''))
    ax.tick_params(axis='y', labelsize=9)
    ax.tick_params(axis='x', length=0)

    # Línea de referencia 100%
    ax.axhline(100, color='#EF4444', linestyle='--',
               linewidth=1.0, alpha=0.40, zorder=1)
    ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] else 3,
            101.5, '100%', fontsize=7.5,
            color='#EF4444', alpha=0.6, va='bottom', ha='right')

    ax.set_title(ds, fontsize=13, fontweight='bold',
                 color=ds_color, pad=12)

# Leyenda global de modelos
legend_handles = [
    mpatches.Patch(facecolor=MODEL_COLOR[m], edgecolor='white',
                   linewidth=0.8, label=m)
    for m in MODELS
]
fig.legend(handles=legend_handles,
           title='Modelo',
           loc='lower center',
           fontsize=9.5, title_fontsize=10,
           framealpha=0.95,
           bbox_to_anchor=(0.5, -0.08),
           ncol=4)

fig.suptitle(
    'Tasa de Evasión por Ataque y Modelo — Comparativa entre Datasets',
    fontsize=13, fontweight='bold', y=1.02)

fig.tight_layout(rect=[0, 0.06, 1, 1])
path = os.path.join(OUT_DIR, 'evasion_bars.png')
fig.savefig(path, bbox_inches='tight', dpi=160)
plt.close(fig)
print(f'Saved: {path}')
