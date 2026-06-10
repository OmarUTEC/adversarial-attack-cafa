"""
Gráficos de burbujas — AMLworld HI-Small
Incluye modelos originales (XGBoost, Log. Reg.) y MLP entrenado con SMOTE.
4 paneles: CaFA | HopSkipJump | BoundaryAttack | SquareAttack
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
NAN = float('nan')

MODEL_STYLE = {
    'XGBoost':          {'marker': 'o', 'color': '#2563EB'},
    'MLP (SMOTE)':      {'marker': '^', 'color': '#16A34A'},
    'Log. Reg.':        {'marker': 's', 'color': '#D97706'},
    'LSTM-Att.(SMOTE)': {'marker': 'D', 'color': '#9333EA'},
}

MODELS = ['XGBoost', 'MLP (SMOTE)', 'Log. Reg.', 'LSTM-Att.(SMOTE)']

ATTACK_TITLE_COLOR = {
    'CaFA':           '#EF4444',
    'HopSkipJump':    '#B45309',
    'BoundaryAttack': '#0369A1',
    'SquareAttack':   '#7C3AED',
}

EVASION = {
    'CaFA':           {'XGBoost': NAN,    'MLP (SMOTE)': 0.9876, 'Log. Reg.': 1.0000, 'LSTM-Att.(SMOTE)': 0.6858},
    'HopSkipJump':    {'XGBoost': 0.9938, 'MLP (SMOTE)': 0.9514, 'Log. Reg.': 0.2957, 'LSTM-Att.(SMOTE)': 0.0000},
    'BoundaryAttack': {'XGBoost': 1.0000, 'MLP (SMOTE)': 1.0000, 'Log. Reg.': 0.9989, 'LSTM-Att.(SMOTE)': 0.0048},
    'SquareAttack':   {'XGBoost': 1.0000, 'MLP (SMOTE)': 1.0000, 'Log. Reg.': 0.8953, 'LSTM-Att.(SMOTE)': 0.0019},
}

L0_DATA = {
    'CaFA':           {'XGBoost': NAN,  'MLP (SMOTE)': 4.724, 'Log. Reg.': 3.52,  'LSTM-Att.(SMOTE)': 2.48},
    'HopSkipJump':    {'XGBoost': 9.92, 'MLP (SMOTE)': 9.528, 'Log. Reg.': 5.89,  'LSTM-Att.(SMOTE)': 0.00},
    'BoundaryAttack': {'XGBoost': 5.01, 'MLP (SMOTE)': 4.529, 'Log. Reg.': 4.75,  'LSTM-Att.(SMOTE)': 0.04},
    'SquareAttack':   {'XGBoost': 6.58, 'MLP (SMOTE)': 6.166, 'Log. Reg.': 7.27,  'LSTM-Att.(SMOTE)': 0.78},
}

GROUPS = [
    ('CaFA',           'CaFA'),
    ('HopSkipJump',    'HopSkipJump'),
    ('BoundaryAttack', 'BoundaryAttack'),
    ('SquareAttack',   'SquareAttack'),
]

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.22,
    'grid.linestyle':    '--',
    'figure.dpi':        160,
})


def _spread(items, x_tol=0.5, y_tol=4.0):
    n      = len(items)
    coords = [[l0, ev] for l0, ev, *_ in items]
    done   = [False] * n
    for i in range(n):
        if done[i]:
            continue
        cluster = [i]
        for j in range(i + 1, n):
            if (not done[j]
                    and abs(coords[j][0] - coords[i][0]) < x_tol
                    and abs(coords[j][1] - coords[i][1]) < y_tol):
                cluster.append(j)
                done[j] = True
        if len(cluster) > 1:
            cx   = sum(coords[k][0] for k in cluster) / len(cluster)
            half = 0.55 * len(cluster)
            step = 2 * half / (len(cluster) - 1)
            for rank, k in enumerate(cluster):
                coords[k][0] = cx - half + rank * step
    return [(coords[k][0], coords[k][1], items[k][2], items[k][3])
            for k in range(n)]


def _place_labels(items, x_range, plot_w_in=5.2, plot_h_in=3.8):
    if not items:
        return []
    du_per_x_in = x_range / plot_w_in
    du_per_y_in = 132.0  / plot_h_in
    CHAR_W = 0.058 * du_per_x_in
    LINE_H = 0.140 * du_per_y_in
    DX     = 0.14  * du_per_x_in
    step   = LINE_H * 1.6
    dy_cands = [i * step for i in
                [1, 2, -1, 3, -2, 4, -3, 5, -4, 6, -5, 7, -6]]

    indexed  = sorted(enumerate(items), key=lambda t: -t[1][1])
    positions = [None] * len(items)
    placed    = []

    for orig_idx, (l0, ev, model, atk) in indexed:
        lw  = len(model) * CHAR_W
        lh  = LINE_H
        tx1 = l0 + DX
        tx2 = tx1 + lw
        chosen = dy_cands[0]
        for cdy in dy_cands:
            ty1 = ev + cdy - lh / 2
            ty2 = ev + cdy + lh / 2
            if not any(tx1 < bx2 and tx2 > bx1 and ty1 < by2 and ty2 > by1
                       for bx1, by1, bx2, by2 in placed):
                chosen = cdy
                placed.append((tx1, ty1, tx2, ty2))
                break
        else:
            chosen = dy_cands[-1]
            placed.append((tx1, ev+chosen-lh/2, tx2, ev+chosen+lh/2))
        positions[orig_idx] = (l0 + DX, ev + chosen)
    return positions


def draw_panel(ax, atk, panel_title, total_feat):
    raw = []
    for model in MODELS:
        ev = EVASION[atk][model]
        l0 = L0_DATA[atk][model]
        if not (np.isnan(ev) or np.isnan(l0)):
            raw.append((l0, ev * 100, model, atk))

    ax.axhline(100, color='#EF4444', linestyle='--',
               linewidth=0.9, alpha=0.35, zorder=1)

    x_range = max(total_feat + 4,
                  (max(l0 for l0, *_ in raw) + 4) if raw else total_feat + 4)

    if raw:
        items = _spread(raw)
        for (l0, ev, model, _) in items:
            s = MODEL_STYLE[model]
            ax.scatter(l0, ev, s=270,
                       marker=s['marker'],
                       color=s['color'],
                       edgecolors='white',
                       linewidths=1.8,
                       zorder=6, alpha=0.93)

        positions = _place_labels(items, x_range=x_range)
        for (l0, ev, model, _), (tx, ty) in zip(items, positions):
            ax.text(tx, ty, model,
                    ha='left', va='center',
                    fontsize=8.2,
                    color=MODEL_STYLE[model]['color'],
                    fontweight='bold',
                    zorder=7)

    ax.set_xlim(-0.3, x_range)
    ax.set_ylim(-13, 122)
    ax.set_xlabel('L0 — Features modificadas', fontsize=9)
    ax.set_ylabel('Tasa de evasión (%)', fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
    ax.tick_params(labelsize=8.5)
    ax.set_title(panel_title, fontsize=12, fontweight='bold',
                 color=ATTACK_TITLE_COLOR[atk], pad=10)


def build_legend(fig):
    handles = [
        Line2D([0], [0],
               marker=MODEL_STYLE[m]['marker'], color='w',
               markerfacecolor=MODEL_STYLE[m]['color'],
               markeredgecolor='white', markeredgewidth=1.2,
               markersize=13, label=m)
        for m in MODELS
    ]
    fig.legend(handles=handles,
               title='Modelo  (forma + color)',
               loc='lower center',
               fontsize=10, title_fontsize=10.5,
               framealpha=0.95,
               bbox_to_anchor=(0.5, -0.05),
               ncol=4)


fig, axes = plt.subplots(2, 2, figsize=(14, 11))
for ax, (title, atk) in zip(axes.flatten(), GROUPS):
    draw_panel(ax, atk, title, total_feat=15)

build_legend(fig)
fig.suptitle(
    'Eficiencia de ataques adversariales — AMLworld HI-Small (IBM)\n'
    'Eje X: features modificadas (L0)  ·  Eje Y: tasa de evasión\n'
    'MLP entrenado con SMOTE (ratio 1:10)',
    fontsize=12, fontweight='bold', color='#B45309', y=1.02)

fig.tight_layout(rect=[0, 0.06, 1, 1])
path = os.path.join(OUT_DIR, 'bubble_amlworld_smote.png')
fig.savefig(path, bbox_inches='tight', dpi=160)
plt.close(fig)
print(f'Saved: {path}')
