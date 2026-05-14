"""
Gráficos de burbujas — L0 vs Tasa de Evasión
4 cuadrantes: CaFA | HopSkipJump | BoundaryAttack | SquareAttack
  · Forma  = Modelo  (fija en todos los paneles)
  · Color  = Modelo  (fija en todos los paneles)
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────
# PALETA  —  cada modelo tiene forma y color únicos
# ─────────────────────────────────────────────────────────────
MODEL_STYLE = {
    'XGBoost':   {'marker': 'o', 'color': '#2563EB'},
    'MLP':       {'marker': '^', 'color': '#16A34A'},
    'Log. Reg.': {'marker': 's', 'color': '#D97706'},
    'LSTM-Att.': {'marker': 'D', 'color': '#9333EA'},
}
MODELS = ['XGBoost', 'MLP', 'Log. Reg.', 'LSTM-Att.']

ATTACK_TITLE_COLOR = {
    'CaFA':           '#EF4444',
    'HopSkipJump':    '#B45309',
    'BoundaryAttack': '#0369A1',
    'SquareAttack':   '#7C3AED',
}
DS_COLOR = {
    'Credit Card': '#1D4ED8',
    'AMLworld HI': '#B45309',
}

NAN = float('nan')

EVASION = {
    'Credit Card': {
        'CaFA':           {'XGBoost': NAN,    'MLP': 0.529, 'Log. Reg.': 0.785, 'LSTM-Att.': 0.237},
        'HopSkipJump':    {'XGBoost': 0.9995, 'MLP': 0.998, 'Log. Reg.': 0.992, 'LSTM-Att.': 0.779},
        'BoundaryAttack': {'XGBoost': 0.9995, 'MLP': 0.480, 'Log. Reg.': 0.480, 'LSTM-Att.': 0.521},
        'SquareAttack':   {'XGBoost': NAN,    'MLP': 1.000, 'Log. Reg.': 1.000, 'LSTM-Att.': 0.226},
    },
    'AMLworld HI': {
        'CaFA':           {'XGBoost': NAN,   'MLP': NAN,  'Log. Reg.': 1.000, 'LSTM-Att.': NAN},
        'HopSkipJump':    {'XGBoost': 0.022, 'MLP': NAN,  'Log. Reg.': 0.052, 'LSTM-Att.': NAN},
        'BoundaryAttack': {'XGBoost': 1.000, 'MLP': NAN,  'Log. Reg.': 1.000, 'LSTM-Att.': NAN},
        'SquareAttack':   {'XGBoost': 1.000, 'MLP': NAN,  'Log. Reg.': 1.000, 'LSTM-Att.': NAN},
    },
}
L0_DATA = {
    'Credit Card': {
        'CaFA':           {'XGBoost': NAN,   'MLP': 11.94, 'Log. Reg.': 9.62,  'LSTM-Att.': 0.60},
        'HopSkipJump':    {'XGBoost': 29.97, 'MLP': 29.86, 'Log. Reg.': 29.83, 'LSTM-Att.': 29.98},
        'BoundaryAttack': {'XGBoost': 29.85, 'MLP': 14.32, 'Log. Reg.': 14.14, 'LSTM-Att.': 20.95},
        'SquareAttack':   {'XGBoost': NAN,   'MLP': 11.92, 'Log. Reg.': 15.58, 'LSTM-Att.': 0.42},
    },
    'AMLworld HI': {
        'CaFA':           {'XGBoost': NAN,  'MLP': NAN, 'Log. Reg.': 3.55, 'LSTM-Att.': NAN},
        'HopSkipJump':    {'XGBoost': 2.56, 'MLP': NAN, 'Log. Reg.': 2.78, 'LSTM-Att.': NAN},
        'BoundaryAttack': {'XGBoost': 5.81, 'MLP': NAN, 'Log. Reg.': 4.51, 'LSTM-Att.': NAN},
        'SquareAttack':   {'XGBoost': 6.05, 'MLP': NAN, 'Log. Reg.': 6.01, 'LSTM-Att.': NAN},
    },
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


# ─────────────────────────────────────────────────────────────
# Dispersar puntos casi coincidentes
# ─────────────────────────────────────────────────────────────
def _spread(items, x_tol=0.9, y_tol=5.0):
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
            half = 0.65 * len(cluster)
            step = 2 * half / (len(cluster) - 1)
            for rank, k in enumerate(cluster):
                coords[k][0] = cx - half + rank * step
    return [(coords[k][0], coords[k][1], items[k][2], items[k][3])
            for k in range(n)]


# ─────────────────────────────────────────────────────────────
# Colocación de etiquetas en coordenadas de datos (sin conversión)
# ─────────────────────────────────────────────────────────────
def _place_labels(items, x_range, y_range=132,
                  plot_w_in=5.2, plot_h_in=3.8):
    """
    Trabaja directamente en unidades de datos para posicionar etiquetas.
    Convierte dimensiones de texto (pulgadas) a unidades de datos y
    detecta colisiones con bounding-boxes 2D.
    Retorna lista de (text_x, text_y) en coordenadas de datos.
    """
    if not items:
        return []

    du_per_x_in = x_range  / plot_w_in   # unidades de datos por pulgada (X)
    du_per_y_in = y_range  / plot_h_in   # unidades de datos por pulgada (Y)

    # Dimensiones de texto en unidades de datos (fuente ~8pt = ~0.11in alto)
    CHAR_W = 0.052 * du_per_x_in   # ~0.052in por carácter
    LINE_H = 0.115 * du_per_y_in   # ~0.115in por línea
    DX     = 0.10  * du_per_x_in   # offset horizontal desde el punto

    step   = LINE_H * 1.35          # separación mínima entre centros de etiqueta
    dy_candidates = [i * step for i in
                     [1, 2, -1, 3, -2, 4, -3, 5, -4, 6, -5, 7, -6, 8, -7]]

    indexed  = sorted(enumerate(items), key=lambda t: -t[1][1])
    positions = [None] * len(items)
    placed    = []   # (x1, y1, x2, y2) en coordenadas de datos

    for orig_idx, (l0, ev, model, atk) in indexed:
        lbl   = model
        nl    = 1
        lw    = len(lbl) * CHAR_W
        lh    = nl * LINE_H
        tx1   = l0 + DX
        tx2   = tx1 + lw

        chosen_dy = dy_candidates[0]
        for cdy in dy_candidates:
            ty1 = ev + cdy - lh / 2
            ty2 = ev + cdy + lh / 2
            if not any(tx1 < bx2 and tx2 > bx1 and ty1 < by2 and ty2 > by1
                       for bx1, by1, bx2, by2 in placed):
                chosen_dy = cdy
                placed.append((tx1, ty1, tx2, ty2))
                break
        else:
            chosen_dy = dy_candidates[-1]
            placed.append((tx1, ev + chosen_dy - lh/2,
                           tx2, ev + chosen_dy + lh/2))

        positions[orig_idx] = (l0 + DX, ev + chosen_dy)

    return positions


# ─────────────────────────────────────────────────────────────
# Panel individual
# ─────────────────────────────────────────────────────────────
def draw_panel(ax, ds, atk, panel_title, total_feat):
    raw = []
    for model in MODELS:
        ev = EVASION[ds][atk][model]
        l0 = L0_DATA[ds][atk][model]
        if not (np.isnan(ev) or np.isnan(l0)):
            raw.append((l0, ev * 100, model, atk))

    # Panel vacío
    if not raw:
        ax.text(0.5, 0.5, 'N/A\n(no compatible\ncon este ataque)',
                ha='center', va='center', fontsize=10,
                color='#9CA3AF', style='italic', transform=ax.transAxes)
        ax.set_facecolor('#F9FAFB')
        ax.set_title(panel_title, fontsize=12, fontweight='bold',
                     color='#6B7280', pad=10)
        ax.set_xlabel('L0 — Features modificadas', fontsize=9)
        ax.set_ylabel('Tasa de evasión (%)', fontsize=9)
        ax.set_xlim(0, total_feat + 1)
        ax.set_ylim(-8, 115)
        return

    items = _spread(raw)

    # Zona de alta amenaza
    threat_x = 8 if ds == 'AMLworld HI' else 16
    ax.add_patch(plt.Rectangle(
        (0, 75), threat_x, 42,
        facecolor='#FEE2E2', alpha=0.45, zorder=0,
        linestyle='--', edgecolor='#EF4444', linewidth=0.8))
    ax.text(0.5, 112, 'Alta amenaza', fontsize=7.5,
            color='#EF4444', alpha=0.80, style='italic')
    ax.axhline(100, color='#EF4444', linestyle='--',
               linewidth=0.9, alpha=0.35, zorder=1)

    # Puntos: forma + color = modelo
    for (l0, ev, model, _atk) in items:
        s = MODEL_STYLE[model]
        ax.scatter(l0, ev, s=270,
                   marker=s['marker'],
                   color=s['color'],
                   edgecolors='white',
                   linewidths=1.8,
                   zorder=4, alpha=0.93)

    # Etiquetas en coordenadas de datos
    x_range = max(total_feat + 10, max(l0 for l0, *_ in items) + 10)
    positions = _place_labels(items, x_range=x_range)

    for (l0, ev, model, _atk), (tx, ty) in zip(items, positions):
        ax.text(tx, ty, model,
                ha='left', va='center',
                fontsize=8.2,
                color=MODEL_STYLE[model]['color'],
                fontweight='bold',
                zorder=5)

    ax.set_xlim(-0.5, x_range)
    ax.set_ylim(-10, 122)
    ax.set_xlabel('L0 — Features modificadas', fontsize=9)
    ax.set_ylabel('Tasa de evasión (%)', fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
    ax.tick_params(labelsize=8.5)
    ax.set_title(panel_title, fontsize=12, fontweight='bold',
                 color=ATTACK_TITLE_COLOR[atk], pad=10)


# ─────────────────────────────────────────────────────────────
# Leyenda de modelos
# ─────────────────────────────────────────────────────────────
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


# ══════════════════════════════════════════════════════════════
# IMAGEN 1 — Credit Card
# ══════════════════════════════════════════════════════════════
def plot_creditcard():
    ds, tf = 'Credit Card', 29
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for ax, (title, atk) in zip(axes.flatten(), GROUPS):
        draw_panel(ax, ds, atk, title, tf)
    build_legend(fig)
    fig.suptitle(
        'Eficiencia de ataques adversariales — Credit Card 2023\n'
        'Eje X: features modificadas (L0)  ·  Eje Y: tasa de evasión  ·  '
        'Zona roja: alta amenaza',
        fontsize=13, fontweight='bold', color=DS_COLOR[ds], y=1.02)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    path = os.path.join(OUT_DIR, 'bubble_creditcard.png')
    fig.savefig(path, bbox_inches='tight', dpi=160)
    plt.close(fig)
    print(f'Saved: {path}')


# ══════════════════════════════════════════════════════════════
# IMAGEN 2 — AMLworld HI-Small
# ══════════════════════════════════════════════════════════════
def plot_amlworld():
    ds, tf = 'AMLworld HI', 15
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for ax, (title, atk) in zip(axes.flatten(), GROUPS):
        draw_panel(ax, ds, atk, title, tf)
    build_legend(fig)
    fig.suptitle(
        'Eficiencia de ataques adversariales — AMLworld HI-Small (IBM)\n'
        'Eje X: features modificadas (L0)  ·  Eje Y: tasa de evasión  ·  '
        'Zona roja: alta amenaza',
        fontsize=13, fontweight='bold', color=DS_COLOR[ds], y=1.02)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    path = os.path.join(OUT_DIR, 'bubble_amlworld.png')
    fig.savefig(path, bbox_inches='tight', dpi=160)
    plt.close(fig)
    print(f'Saved: {path}')


if __name__ == '__main__':
    print('Generando gráficos de burbujas...')
    plot_creditcard()
    plot_amlworld()
    print('Listo.')
