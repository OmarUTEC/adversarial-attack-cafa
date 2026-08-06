"""
Versión a colores de los 3 gráficos de burbujas para LaTeX.
Misma estructura que gen_bubble_latex.py pero con paleta de colores.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'dashboard', 'data')
OUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'imagenes')

ATTACKS  = ['CaFA', 'HopSkipJump', 'BoundaryAttack', 'SquareAttack']
PELIGRO  = 70.0

ATTACK_COLOR = {
    'CaFA':           '#0C4A6E',
    'HopSkipJump':    '#0C4A6E',
    'BoundaryAttack': '#0C4A6E',
    'SquareAttack':   '#0C4A6E',
}

MODEL_STYLE = {
    'XGBoost':          {'marker': 'o', 'color': '#0C0C0C', 'label': 'XGBoost'},
    'MLP':              {'marker': '^', 'color': '#0369A1', 'label': 'MLP'},
    'Log. Reg.':        {'marker': 's', 'color': '#38BDF8', 'label': 'Log. Reg.'},
    'LSTM-Att.':        {'marker': 'D', 'color': '#64748B', 'label': 'LSTM-Att.'},
    'MLP (SMOTE)':      {'marker': '^', 'color': '#0369A1', 'label': 'MLP (S)'},
    'LSTM-Att.(SMOTE)': {'marker': 'D', 'color': '#64748B', 'label': 'LSTM-Att.(S)'},
}

plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'axes.facecolor':   'white',
    'figure.facecolor': 'white',
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.alpha':       0.25,
    'grid.linestyle':   ':',
    'grid.color':       '#BBBBBB',
    'figure.dpi':       160,
})


def load_dataset(folder):
    path = os.path.join(DATA_DIR, folder, 'attacks.json')
    d    = json.load(open(path))
    nf   = d['n_features']
    recs = []
    for r in d['records']:
        # Excluir modelos que no detectaban fraude en baseline (recall=0)
        if r.get('baseline', {}).get('recall', 1) == 0:
            continue
        recs.append({
            'attack':  r['attack'],
            'model':   r['model'],
            'evasion': r['evasion'],
            'l0_pct':  round(r['l0'] / nf * 100, 2),
        })
    return recs, nf


def _spread(items, x_tol=1.5, y_tol=4.0):
    coords = [[r['l0_pct'], r['evasion']] for r in items]
    done   = [False] * len(items)
    for i in range(len(items)):
        if done[i]:
            continue
        cluster = [i]
        for j in range(i + 1, len(items)):
            if (not done[j]
                    and abs(coords[j][0] - coords[i][0]) < x_tol
                    and abs(coords[j][1] - coords[i][1]) < y_tol):
                cluster.append(j)
                done[j] = True
        if len(cluster) > 1:
            cx   = sum(coords[k][0] for k in cluster) / len(cluster)
            half = 1.2 * len(cluster)
            step = 2 * half / (len(cluster) - 1)
            for rank, k in enumerate(cluster):
                coords[k][0] = cx - half + rank * step
    return [{**items[k], 'l0_pct': coords[k][0]} for k in range(len(items))]


def draw_panel(ax, pts, atk, x_max):
    panel   = [r for r in pts if r['attack'] == atk]
    l0_vals = [r['l0_pct'] for r in panel if r['l0_pct'] > 0]
    med_pct = float(np.median(l0_vals)) if l0_vals else x_max / 2

    # Zona de Peligro (celeste muy tenue)
    ax.axhspan(PELIGRO, 108, color='#E0F2FE', alpha=0.8, zorder=0)
    # Zona Crítica (celeste más marcado)
    ax.add_patch(mpatches.Rectangle(
        (-0.5, PELIGRO), med_pct + 0.5, 108 - PELIGRO,
        color='#7DD3FC', alpha=0.35, zorder=0
    ))

    # Líneas de referencia
    ax.axhline(100,     color='#334155', linestyle='--', linewidth=0.8, alpha=0.4, zorder=1)
    ax.axhline(PELIGRO, color='#0369A1', linestyle='-',  linewidth=1.3, alpha=0.9, zorder=2)
    ax.axvline(med_pct, color='#334155', linestyle=':',  linewidth=1.0, alpha=0.6, zorder=2)

    # Etiquetas de zona (encima del panel)
    ax.text(0.02, 1.01, 'Zona Crítica', transform=ax.transAxes,
            fontsize=9, color='#0369A1', va='bottom', ha='left',
            style='italic', clip_on=False)
    ax.text(0.99, 1.01, 'Zona de Peligro', transform=ax.transAxes,
            fontsize=9, color='#0369A1', va='bottom', ha='right',
            style='italic', clip_on=False)

    # Puntos
    if panel:
        for r in _spread(panel):
            s = MODEL_STYLE.get(r['model'])
            if s is None:
                continue
            ax.scatter(r['l0_pct'], r['evasion'], s=340,
                       marker=s['marker'],
                       color=s['color'],
                       edgecolors='white',
                       linewidths=1.0,
                       zorder=6)

    ax.set_xlim(-1, x_max)
    ax.set_ylim(-10, 108)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel('% Features modificadas', fontsize=11, color='#374151')
    ax.set_ylabel('Tasa de evasión (%)',     fontsize=11, color='#374151')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
    ax.tick_params(labelsize=10.5, colors='#374151')
    for sp in ax.spines.values():
        sp.set_color('#D1D5DB')
        sp.set_linewidth(0.8)
    ax.set_title(atk, fontsize=13, fontweight='bold',
                 color=ATTACK_COLOR.get(atk, 'black'), pad=8)


def build_legend(fig, models_in_data):
    model_handles = []
    seen = set()
    for m in models_in_data:
        s = MODEL_STYLE.get(m)
        if s and s['label'] not in seen:
            seen.add(s['label'])
            model_handles.append(
                Line2D([0], [0], marker=s['marker'], linestyle='None',
                       color=s['color'], markeredgecolor='white',
                       markeredgewidth=0.8, markersize=11, label=s['label'])
            )

    zone_handles = [
        mpatches.Patch(facecolor='#7DD3FC', alpha=0.5, edgecolor='#0369A1',
                       linewidth=0.8, label='Zona Crítica  (evasión ≥ 70 %, L₀ ≤ mediana)'),
        mpatches.Patch(facecolor='#E0F2FE', alpha=0.9, edgecolor='#0369A1',
                       linewidth=0.8, label='Zona de Peligro  (evasión ≥ 70 %)'),
    ]

    fig.legend(
        handles=model_handles + zone_handles,
        loc='lower center',
        fontsize=11,
        framealpha=1.0,
        edgecolor='#D1D5DB',
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        handlelength=2.0,
        handletextpad=0.7,
        columnspacing=2.5,
        labelspacing=1.1,
    )


def make_figure(folder, x_max_pct):
    pts, _  = load_dataset(folder)
    models  = list({r['model'] for r in pts})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.patch.set_facecolor('white')
    for ax, atk in zip(axes.flatten(), ATTACKS):
        draw_panel(ax, pts, atk, x_max_pct)
    build_legend(fig, models)
    fig.tight_layout(rect=[0, 0.11, 1, 1])
    return fig


if __name__ == '__main__':
    fig = make_figure('creditcard', x_max_pct=115)
    out = os.path.join(OUT_DIR, 'bubble_creditcard_color.png')
    fig.savefig(out, bbox_inches='tight', dpi=180, facecolor='white')
    plt.close(fig)
    print(f'Guardado: {out}')

    fig = make_figure('amlworld_sinsmote', x_max_pct=20)
    out = os.path.join(OUT_DIR, 'bubble_amlworld_color.png')
    fig.savefig(out, bbox_inches='tight', dpi=180, facecolor='white')
    plt.close(fig)
    print(f'Guardado: {out}')

    fig = make_figure('amlworld_smote', x_max_pct=30)
    out = os.path.join(OUT_DIR, 'bubble_amlworld_smote_color.png')
    fig.savefig(out, bbox_inches='tight', dpi=180, facecolor='white')
    plt.close(fig)
    print(f'Guardado: {out}')

    print('Listo.')
