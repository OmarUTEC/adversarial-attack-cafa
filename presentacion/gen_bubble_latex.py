"""
Genera 3 gráficos de burbujas para LaTeX — datos desde dashboard JSON.
Eje X: % features modificadas (L0/n_features×100)
B&W con Zona de Peligro y Zona Crítica.
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

ATTACKS = ['CaFA', 'HopSkipJump', 'BoundaryAttack', 'SquareAttack']
PELIGRO = 70.0

# Paleta B&W: forma + relleno
MODEL_STYLE = {
    'XGBoost':          {'marker': 'o', 'fc': 'black',   'ec': 'black',   'label': 'XGBoost'},
    'MLP':              {'marker': '^', 'fc': 'white',   'ec': 'black',   'label': 'MLP'},
    'Log. Reg.':        {'marker': 's', 'fc': '#555555', 'ec': 'black',   'label': 'Log. Reg.'},
    'LSTM-Att.':        {'marker': 'D', 'fc': '#AAAAAA', 'ec': 'black',   'label': 'LSTM-Att.'},
    'MLP (SMOTE)':      {'marker': '^', 'fc': 'white',   'ec': 'black',   'label': 'MLP (S)'},
    'LSTM-Att.(SMOTE)': {'marker': 'D', 'fc': '#AAAAAA', 'ec': 'black',   'label': 'LSTM-Att.(S)'},
}

plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'axes.facecolor':   'white',
    'figure.facecolor': 'white',
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'grid.linestyle':   ':',
    'grid.color':       '#999999',
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
            'evasion': r['evasion'],          # ya en %
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
    out = []
    for k, r in enumerate(items):
        out.append({**r, 'l0_pct': coords[k][0]})
    return out


def draw_panel(ax, pts, atk, x_max):
    panel = [r for r in pts if r['attack'] == atk]

    # Mediana L0% del panel (excluye l0=0)
    l0_vals = [r['l0_pct'] for r in panel if r['l0_pct'] > 0]
    med_pct  = float(np.median(l0_vals)) if l0_vals else x_max / 2

    # ── Zonas de fondo ──────────────────────────────────────
    ax.axhspan(PELIGRO, 108, color='#CCCCCC', alpha=0.28, zorder=0)
    ax.add_patch(mpatches.Rectangle(
        (-0.5, PELIGRO), med_pct + 0.5, 108 - PELIGRO,
        color='#888888', alpha=0.22, zorder=0
    ))

    # ── Líneas de referencia ────────────────────────────────
    ax.axhline(100,     color='black', linestyle='--', linewidth=0.8,  alpha=0.35, zorder=1)
    ax.axhline(PELIGRO, color='black', linestyle='-',  linewidth=1.1,  alpha=0.70, zorder=2)
    ax.axvline(med_pct, color='black', linestyle=':',  linewidth=0.9,  alpha=0.55, zorder=2)

    # ── Etiquetas de zona ───────────────────────────────────
    ax.text(0.02, 1.01, 'Zona Crítica', transform=ax.transAxes,
            fontsize=9, color='#333333', va='bottom', ha='left', style='italic',
            clip_on=False,
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.0))
    ax.text(0.99, 1.01, 'Zona de Peligro', transform=ax.transAxes,
            fontsize=9, color='#333333', va='bottom', ha='right', style='italic',
            clip_on=False,
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.0))

    # ── Puntos ──────────────────────────────────────────────
    if panel:
        spread = _spread(panel)
        for r in spread:
            s = MODEL_STYLE.get(r['model'])
            if s is None:
                continue
            ax.scatter(r['l0_pct'], r['evasion'], s=320,
                       marker=s['marker'],
                       facecolors=s['fc'],
                       edgecolors=s['ec'],
                       linewidths=1.2,
                       zorder=6)

    ax.set_xlim(-1, x_max)
    ax.set_ylim(-10, 108)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel('% Features modificadas', fontsize=11, color='#222222')
    ax.set_ylabel('Tasa de evasión (%)',            fontsize=11, color='#222222')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
    ax.tick_params(labelsize=10.5, colors='#333333')
    for sp in ax.spines.values():
        sp.set_color('#888888')
        sp.set_linewidth(0.8)
    ax.set_title(atk, fontsize=13, fontweight='bold', color='black', pad=8)


def build_legend(fig, models_in_data):
    # Solo modelos que aparecen en el dataset
    model_handles = []
    seen = set()
    for m in models_in_data:
        s = MODEL_STYLE.get(m)
        if s and s['label'] not in seen:
            seen.add(s['label'])
            model_handles.append(
                Line2D([0], [0], marker=s['marker'], linestyle='None',
                       markerfacecolor=s['fc'], markeredgecolor=s['ec'],
                       markeredgewidth=1.2, markersize=11, label=s['label'])
            )

    zone_handles = [
        mpatches.Patch(facecolor='#888888', alpha=0.45, edgecolor='black',
                       linewidth=0.5, label='Zona Crítica  (evasión ≥ 70 %, L₀ ≤ mediana)'),
        mpatches.Patch(facecolor='#CCCCCC', alpha=0.55, edgecolor='black',
                       linewidth=0.5, label='Zona de Peligro  (evasión ≥ 70 %)'),
    ]

    fig.legend(
        handles=model_handles + zone_handles,
        loc='lower center',
        fontsize=11,
        framealpha=1.0,
        edgecolor='#AAAAAA',
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        handlelength=2.0,
        handletextpad=0.7,
        columnspacing=2.5,
        labelspacing=1.1,
    )


def make_figure(folder, x_max_pct):
    pts, nf = load_dataset(folder)
    models   = list({r['model'] for r in pts})

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.patch.set_facecolor('white')
    for ax, atk in zip(axes.flatten(), ATTACKS):
        draw_panel(ax, pts, atk, x_max_pct)
    build_legend(fig, models)
    fig.tight_layout(rect=[0, 0.11, 1, 1])
    return fig


# ── Generar las 3 imágenes ─────────────────────────────────────
if __name__ == '__main__':
    # Credit Card: L0% puede llegar a ~104%, usar 115
    fig = make_figure('creditcard', x_max_pct=115)
    out = os.path.join(OUT_DIR, 'bubble_creditcard.png')
    fig.savefig(out, bbox_inches='tight', dpi=180, facecolor='white')
    plt.close(fig)
    print(f'Guardado: {out}')

    # AMLworld sin SMOTE: L0% máx ~14.7%, usar 20
    fig = make_figure('amlworld_sinsmote', x_max_pct=20)
    out = os.path.join(OUT_DIR, 'bubble_amlworld.png')
    fig.savefig(out, bbox_inches='tight', dpi=180, facecolor='white')
    plt.close(fig)
    print(f'Guardado: {out}')

    # AMLworld con SMOTE: L0% máx ~24.2%, usar 30
    fig = make_figure('amlworld_smote', x_max_pct=30)
    out = os.path.join(OUT_DIR, 'bubble_amlworld_smote.png')
    fig.savefig(out, bbox_inches='tight', dpi=180, facecolor='white')
    plt.close(fig)
    print(f'Guardado: {out}')

    print('Listo.')
