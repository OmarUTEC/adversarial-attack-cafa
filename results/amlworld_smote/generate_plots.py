"""
Genera gráficos de ataques adversariales sobre modelos entrenados con SMOTE
en AMLworld HI-Small (IBM). Modelos: MLP, LSTM-Att., Log. Reg., XGBoost.
Estilo consistente con bubble_plots.py (creditcard).
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────
# PALETA — idéntica a creditcard/bubble_plots.py
# ─────────────────────────────────────────────────────────────
MODEL_STYLE = {
    'XGBoost':   {'marker': 'o', 'color': '#2563EB'},
    'MLP':       {'marker': '^', 'color': '#16A34A'},
    'Log. Reg.': {'marker': 's', 'color': '#D97706'},
    'LSTM-Att.': {'marker': 'D', 'color': '#9333EA'},
}
MODELS = ['XGBoost', 'MLP', 'Log. Reg.', 'LSTM-Att.']

ATTACK_COLOR = {
    'CaFA':           '#EF4444',
    'HopSkipJump':    '#B45309',
    'BoundaryAttack': '#0369A1',
    'SquareAttack':   '#7C3AED',
}
ALL_ATTACKS = ['CaFA', 'HopSkipJump', 'BoundaryAttack', 'SquareAttack']

NAN = float('nan')

# ─────────────────────────────────────────────────────────────
# DATOS — SMOTE AMLworld HI-Small
# evasion [0-1], l0_mis = L0 sobre ejemplos evadidos
# ─────────────────────────────────────────────────────────────
EVASION = {
    'CaFA':           {'XGBoost': NAN,    'MLP': 0.9876, 'Log. Reg.': 1.0000, 'LSTM-Att.': 0.6820},
    'HopSkipJump':    {'XGBoost': 0.9925, 'MLP': 0.9514, 'Log. Reg.': 0.3222, 'LSTM-Att.': 0.0000},
    'BoundaryAttack': {'XGBoost': 1.0000, 'MLP': 1.0000, 'Log. Reg.': 0.9977, 'LSTM-Att.': 0.0067},
    'SquareAttack':   {'XGBoost': 1.0000, 'MLP': 1.0000, 'Log. Reg.': 0.8964, 'LSTM-Att.': 0.0019},
}
L0_DATA = {
    'CaFA':           {'XGBoost': NAN,   'MLP': 4.72,  'Log. Reg.': 3.52,  'LSTM-Att.': 3.69},
    'HopSkipJump':    {'XGBoost': 9.94,  'MLP': 9.62,  'Log. Reg.': 9.06,  'LSTM-Att.': NAN},
    'BoundaryAttack': {'XGBoost': 4.97,  'MLP': 4.53,  'Log. Reg.': 4.76,  'LSTM-Att.': 8.29},
    'SquareAttack':   {'XGBoost': 6.39,  'MLP': 6.17,  'Log. Reg.': 7.09,  'LSTM-Att.': 6.50},
}

# Métricas basales
BASELINE = {
    'MLP':       {'auc_roc': 0.9351, 'recall': 0.8234, 'f1': 0.0184},
    'LSTM-Att.': {'auc_roc': 0.7131, 'recall': 0.9688, 'f1': 0.0025},
    'Log. Reg.': {'auc_roc': 0.8436, 'recall': 0.7859, 'f1': 0.0096},
    'XGBoost':   {'auc_roc': 0.9661, 'recall': 0.7422, 'f1': 0.0366},
}

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
# Helpers anti-solapamiento (mismo algoritmo que bubble_plots.py)
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


def _place_labels(items, x_range, plot_w_in=5.2, plot_h_in=3.8):
    if not items:
        return []
    du_per_x_in = x_range / plot_w_in
    du_per_y_in = 132.0   / plot_h_in
    CHAR_W = 0.056 * du_per_x_in
    LINE_H = 0.130 * du_per_y_in
    DX     = 0.12  * du_per_x_in
    step   = LINE_H * 1.5
    dy_cands = [i * step for i in
                [1, 2, -1, 3, -2, 4, -3, 5, -4, 6, -5, 7, -6, 8, -7, 9, -8]]

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


def _legend_handles():
    return [
        Line2D([0], [0],
               marker=MODEL_STYLE[m]['marker'], color='w',
               markerfacecolor=MODEL_STYLE[m]['color'],
               markeredgecolor='white', markeredgewidth=1.2,
               markersize=13, label=m)
        for m in MODELS
    ]


# ══════════════════════════════════════════════════════════════
# FIG 0 — Métricas basales de los 4 modelos SMOTE
# ══════════════════════════════════════════════════════════════
def plot_baseline_metrics():
    fig, ax = plt.subplots(figsize=(10, 5.5))

    metrics = ['AUC-ROC', 'Recall fraude', 'F1 fraude']
    keys    = ['auc_roc', 'recall', 'f1']
    x       = np.arange(len(metrics))
    width   = 0.18
    offsets = np.linspace(-1.5, 1.5, 4) * width

    for model, offset in zip(MODELS, offsets):
        s    = MODEL_STYLE[model]
        vals = [BASELINE[model][k] for k in keys]
        bars = ax.bar(x + offset, vals, width, label=model,
                      color=s['color'], alpha=0.85,
                      edgecolor='white', linewidth=0.8)
        for bar, val in zip(bars, vals):
            if val > 0.005:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.012,
                        f'{val:.3f}', ha='center', va='bottom',
                        fontsize=7, color=s['color'], fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel('Valor de métrica', fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title('Métricas basales por modelo — AMLworld HI-Small (SMOTE)',
                 fontsize=13, fontweight='bold', pad=12)
    ax.legend(handles=_legend_handles(), fontsize=9, framealpha=0.9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))

    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig0_baseline_metrics.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


# ══════════════════════════════════════════════════════════════
# FIG 1 — Heatmap evasión (modelos × ataques)
# ══════════════════════════════════════════════════════════════
def plot_evasion_heatmap():
    data = np.full((len(MODELS), len(ALL_ATTACKS)), np.nan)
    for i, model in enumerate(MODELS):
        key = model  # same keys
        for j, atk in enumerate(ALL_ATTACKS):
            ev = EVASION[atk][model]
            if not np.isnan(ev):
                data[i, j] = ev * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.cm.RdYlGn_r
    cmap.set_bad('#E0E0E0')
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=100, aspect='auto')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Tasa de evasión (%)', fontsize=10)

    ax.set_xticks(range(len(ALL_ATTACKS)))
    ax.set_xticklabels(ALL_ATTACKS, fontsize=10)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS, fontsize=10)

    for i in range(len(MODELS)):
        for j in range(len(ALL_ATTACKS)):
            val = data[i, j]
            if not np.isnan(val):
                text_color = 'white' if (val > 65 or val < 15) else 'black'
                ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                        fontsize=9, fontweight='bold', color=text_color)
            else:
                ax.text(j, i, 'N/A', ha='center', va='center',
                        fontsize=9, color='#888888', style='italic')

    ax.set_title('Tasa de evasión (%) — AMLworld HI-Small (SMOTE)',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Ataque', fontsize=11)
    ax.set_ylabel('Modelo', fontsize=11)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig1_evasion_heatmap.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


# ══════════════════════════════════════════════════════════════
# FIG 2a-2d — Scatter L0 vs evasión, UNA figura por ataque
# ══════════════════════════════════════════════════════════════
def plot_scatter_per_attack():
    for atk in ALL_ATTACKS:
        fig, ax = plt.subplots(figsize=(7, 5))

        raw = []
        for model in MODELS:
            ev = EVASION[atk][model]
            l0 = L0_DATA[atk][model]
            if not (np.isnan(ev) or np.isnan(l0)) and ev > 0:
                raw.append((l0, ev * 100, model, atk))

        ax.axhline(100, color='#EF4444', linestyle='--',
                   linewidth=0.9, alpha=0.35, zorder=1)

        x_max = 14
        if raw:
            x_max = max(max(l0 for l0, *_ in raw) + 3, 14)
            items = _spread(raw)
            for (l0, ev, model, _) in items:
                s = MODEL_STYLE[model]
                ax.scatter(l0, ev, s=280,
                           marker=s['marker'],
                           color=s['color'],
                           edgecolors='white',
                           linewidths=1.8,
                           zorder=6, alpha=0.93)

            positions = _place_labels(items, x_range=x_max)
            for (l0, ev, model, _), (tx, ty) in zip(items, positions):
                ax.text(tx, ty, model,
                        ha='left', va='center',
                        fontsize=9, color=MODEL_STYLE[model]['color'],
                        fontweight='bold', zorder=7)

        # Modelos con evasión 0 o N/A → nota al pie del eje
        for model in MODELS:
            ev = EVASION[atk][model]
            l0 = L0_DATA[atk][model]
            if np.isnan(ev) or np.isnan(l0):
                label = f'{model}: N/A'
            elif ev == 0.0:
                label = f'{model}: 0% evasión'
            else:
                continue
            ax.text(0.02, 0.04 + MODELS.index(model) * 0.06,
                    label, transform=ax.transAxes,
                    fontsize=8, color=MODEL_STYLE[model]['color'],
                    alpha=0.7, style='italic')

        ax.set_xlim(-0.5, x_max)
        ax.set_ylim(-8, 115)
        ax.set_xlabel('L₀ — Features modificadas (sobre evadidos)', fontsize=10)
        ax.set_ylabel('Tasa de evasión (%)', fontsize=10)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
        ax.tick_params(labelsize=9)
        ax.set_title(f'{atk} — AMLworld HI-Small (SMOTE)',
                     fontsize=13, fontweight='bold',
                     color=ATTACK_COLOR[atk], pad=10)

        fig.legend(handles=_legend_handles(),
                   title='Modelo  (forma + color)',
                   loc='lower center', fontsize=9, title_fontsize=9.5,
                   framealpha=0.95, bbox_to_anchor=(0.5, -0.08), ncol=4)
        fig.tight_layout(rect=[0, 0.08, 1, 1])

        slug = atk.lower().replace(' ', '_')
        path = os.path.join(OUT_DIR, f'fig2_{slug}.png')
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {path}')


# ══════════════════════════════════════════════════════════════
# FIG 3 — Degradación del recall (barras horiz.), una por modelo
# ══════════════════════════════════════════════════════════════
def plot_recall_degradation():
    for model in MODELS:
        s = MODEL_STYLE[model]

        attacks_with_data = [atk for atk in ALL_ATTACKS
                             if not np.isnan(EVASION[atk][model])]
        recall_after = [1.0 - EVASION[atk][model] for atk in attacks_with_data]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        y      = np.arange(len(attacks_with_data))
        height = 0.38

        ax.barh(y + height / 2, [100.0] * len(attacks_with_data),
                height, color='#4CAF50', alpha=0.75, label='Antes del ataque')
        ax.barh(y - height / 2, [r * 100 for r in recall_after],
                height,
                color=[ATTACK_COLOR[a] for a in attacks_with_data],
                alpha=0.85, label='Después del ataque')

        for i, (ra, atk) in enumerate(zip(recall_after, attacks_with_data)):
            ax.text(101, i + height / 2, '100%',
                    va='center', fontsize=8, color='#2E7D32', fontweight='bold')
            label = f'{ra*100:.1f}%' if ra > 0 else '0%'
            xpos  = ra * 100 + 0.5 if ra > 0.05 else 1.5
            ax.text(xpos, i - height / 2, label,
                    va='center', fontsize=8,
                    color=ATTACK_COLOR[atk], fontweight='bold')

        ax.set_yticks(y)
        ax.set_yticklabels(attacks_with_data, fontsize=10)
        ax.set_xlim(0, 118)
        ax.set_xlabel('Recall fraude (%)', fontsize=10)
        ax.set_title(f'{model} — Degradación del recall\nAMLworld HI-Small (SMOTE)',
                     fontsize=12, fontweight='bold', color=s['color'], pad=10)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
        ax.axvline(100, color='green', linestyle='--', linewidth=0.8, alpha=0.4)

        green_patch  = mpatches.Patch(color='#4CAF50', alpha=0.75,
                                      label='Antes del ataque (recall = 100%)')
        attack_patch = mpatches.Patch(color='#888888', alpha=0.75,
                                      label='Después del ataque')
        ax.legend(handles=[green_patch, attack_patch],
                  loc='lower right', fontsize=8.5, framealpha=0.9)

        fig.tight_layout()
        slug = model.lower().replace('. ', '_').replace(' ', '_').replace('.', '')
        path = os.path.join(OUT_DIR, f'fig3_recall_{slug}.png')
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {path}')


# ══════════════════════════════════════════════════════════════
# FIG 4 — Barras agrupadas: todos los modelos y ataques
# ══════════════════════════════════════════════════════════════
def plot_evasion_bars():
    fig, ax = plt.subplots(figsize=(12, 5.5))

    x       = np.arange(len(ALL_ATTACKS))
    width   = 0.18
    offsets = np.linspace(-1.5, 1.5, 4) * width

    for model, offset in zip(MODELS, offsets):
        s    = MODEL_STYLE[model]
        vals = []
        xs   = []
        for i, atk in enumerate(ALL_ATTACKS):
            ev = EVASION[atk][model]
            if not np.isnan(ev):
                vals.append(ev * 100)
                xs.append(x[i] + offset)

        if xs:
            bars = ax.bar(xs, vals, width * 0.92,
                          color=s['color'], alpha=0.88,
                          edgecolor='white', linewidth=0.6, zorder=3)
            for bar, h in zip(bars, vals):
                if h >= 5:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            h + 1.5, f'{h:.0f}%',
                            ha='center', va='bottom', fontsize=7,
                            fontweight='bold', color=s['color'])

        # N/A
        for i, atk in enumerate(ALL_ATTACKS):
            if np.isnan(EVASION[atk][model]):
                ax.text(x[i] + offset, -7, 'N/A',
                        ha='center', va='top', fontsize=6,
                        color=s['color'], alpha=0.55, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(ALL_ATTACKS, fontsize=10, fontweight='bold')
    ax.set_ylabel('Tasa de evasión (%)', fontsize=11)
    ax.set_ylim(-15, 115)
    ax.set_title('Tasa de evasión por ataque y modelo — AMLworld HI-Small (SMOTE)',
                 fontsize=13, fontweight='bold', pad=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda y, _: f'{y:.0f}%' if y >= 0 else ''))
    ax.axhline(100, color='#EF4444', linestyle='--',
               linewidth=1.0, alpha=0.40, zorder=1)

    fig.legend(handles=_legend_handles(),
               title='Modelo', loc='lower center',
               fontsize=9.5, title_fontsize=10,
               framealpha=0.95, bbox_to_anchor=(0.5, -0.08), ncol=4)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    path = os.path.join(OUT_DIR, 'fig4_evasion_bars.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


if __name__ == '__main__':
    plot_baseline_metrics()
    plot_evasion_heatmap()
    plot_scatter_per_attack()
    plot_recall_degradation()
    plot_evasion_bars()
    print('\nTodos los gráficos generados correctamente.')
