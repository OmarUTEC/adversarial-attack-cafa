"""
Gráficos para tesis — Robustez Adversarial en Detección de Fraude Financiero
Datasets: Credit Card 2023 | AMLworld HI-Small (IBM)
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────
# PALETA Y ESTILO GLOBAL
# ─────────────────────────────────────────────────────────────
PALETTE = {
    'XGBoost':   '#2563EB',   # azul
    'MLP':       '#16A34A',   # verde
    'Log. Reg.': '#D97706',   # naranja
    'LSTM-Att.': '#9333EA',   # violeta
}
ATTACK_PALETTE = {
    'CaFA':           '#EF4444',
    'SimBA':          '#F97316',
    'HopSkipJump':    '#EAB308',
    'BoundaryAttack': '#06B6D4',
    'SquareAttack':   '#8B5CF6',
}
DS_COLORS = {'Credit Card': '#1D4ED8', 'AMLworld HI': '#B45309'}

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.25,
    'grid.linestyle':    '--',
    'figure.dpi':        150,
})

MODELS  = ['XGBoost', 'MLP', 'Log. Reg.', 'LSTM-Att.']
ATTACKS = ['CaFA', 'SimBA', 'HopSkipJump', 'BoundaryAttack', 'SquareAttack']

# ─────────────────────────────────────────────────────────────
# DATOS
# ─────────────────────────────────────────────────────────────
BASELINE = {
    'Credit Card': {
        'XGBoost':   {'AUC-ROC': 1.0000, 'AUC-PR': 1.0000, 'Recall': 0.9997, 'F1': 0.9998},
        'MLP':       {'AUC-ROC': 1.0000, 'AUC-PR': 1.0000, 'Recall': 0.9998, 'F1': 0.9997},
        'Log. Reg.': {'AUC-ROC': 0.9914, 'AUC-PR': 0.9920, 'Recall': 0.9921, 'F1': 0.9926},
        'LSTM-Att.': {'AUC-ROC': 0.8183, 'AUC-PR': 0.7372, 'Recall': 0.9416, 'F1': 0.8194},
    },
    'AMLworld HI': {
        'XGBoost':   {'AUC-ROC': 0.9660, 'AUC-PR': 0.1833, 'Recall': 0.3395, 'F1': 0.2032},
        'MLP':       {'AUC-ROC': 0.9131, 'AUC-PR': 0.0146, 'Recall': 0.0000, 'F1': 0.0000},
        'Log. Reg.': {'AUC-ROC': 0.5672, 'AUC-PR': 0.0013, 'Recall': 0.1067, 'F1': 0.0103},
        'LSTM-Att.': {'AUC-ROC': None,   'AUC-PR': None,   'Recall': None,   'F1': None},
    },
}

NAN = float('nan')
EVASION = {
    'Credit Card': {
        'CaFA':           {'XGBoost': NAN,   'MLP': 0.529, 'Log. Reg.': 0.785, 'LSTM-Att.': 0.237},
        'SimBA':          {'XGBoost': NAN,   'MLP': 0.035, 'Log. Reg.': 0.055, 'LSTM-Att.': 0.221},
        'HopSkipJump':    {'XGBoost': 0.9995,'MLP': 0.998, 'Log. Reg.': 0.992, 'LSTM-Att.': 0.779},
        'BoundaryAttack': {'XGBoost': 0.9995,'MLP': 0.480, 'Log. Reg.': 0.480, 'LSTM-Att.': 0.521},
        'SquareAttack':   {'XGBoost': NAN,   'MLP': 1.000, 'Log. Reg.': 1.000, 'LSTM-Att.': 0.226},
    },
    'AMLworld HI': {
        'CaFA':           {'XGBoost': NAN,   'MLP': NAN,  'Log. Reg.': 1.000, 'LSTM-Att.': NAN},
        'SimBA':          {'XGBoost': NAN,   'MLP': NAN,  'Log. Reg.': 1.000, 'LSTM-Att.': NAN},
        'HopSkipJump':    {'XGBoost': 0.022, 'MLP': NAN,  'Log. Reg.': 0.052, 'LSTM-Att.': NAN},
        'BoundaryAttack': {'XGBoost': 1.000, 'MLP': NAN,  'Log. Reg.': 1.000, 'LSTM-Att.': NAN},
        'SquareAttack':   {'XGBoost': 1.000, 'MLP': NAN,  'Log. Reg.': 1.000, 'LSTM-Att.': NAN},
    },
}

L0 = {
    'Credit Card': {
        'CaFA':           {'XGBoost': NAN,  'MLP': 11.94,'Log. Reg.': 9.62, 'LSTM-Att.': 0.60},
        'SimBA':          {'XGBoost': NAN,  'MLP': 2.99, 'Log. Reg.': 1.37, 'LSTM-Att.': 0.08},
        'HopSkipJump':    {'XGBoost': 29.97,'MLP': 29.86,'Log. Reg.': 29.83,'LSTM-Att.': 29.98},
        'BoundaryAttack': {'XGBoost': 29.85,'MLP': 14.32,'Log. Reg.': 14.14,'LSTM-Att.': 20.95},
        'SquareAttack':   {'XGBoost': NAN,  'MLP': 11.92,'Log. Reg.': 15.58,'LSTM-Att.': 0.42},
    },
    'AMLworld HI': {
        'CaFA':           {'XGBoost': NAN,  'MLP': NAN, 'Log. Reg.': 3.55, 'LSTM-Att.': NAN},
        'SimBA':          {'XGBoost': NAN,  'MLP': NAN, 'Log. Reg.': 2.39, 'LSTM-Att.': NAN},
        'HopSkipJump':    {'XGBoost': 2.56, 'MLP': NAN, 'Log. Reg.': 2.78, 'LSTM-Att.': NAN},
        'BoundaryAttack': {'XGBoost': 5.81, 'MLP': NAN, 'Log. Reg.': 4.51, 'LSTM-Att.': NAN},
        'SquareAttack':   {'XGBoost': 6.05, 'MLP': NAN, 'Log. Reg.': 6.01, 'LSTM-Att.': NAN},
    },
}


# ══════════════════════════════════════════════════════════════
# GRÁFICO 1 — Métricas basales lado a lado
# ══════════════════════════════════════════════════════════════
def plot_baseline():
    metrics = ['AUC-ROC', 'AUC-PR', 'Recall', 'F1']
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=False)

    for ax, ds in zip(axes, ['Credit Card', 'AMLworld HI']):
        x     = np.arange(len(metrics))
        width = 0.18
        offsets = np.linspace(-(1.5*width), 1.5*width, 4)

        for i, (model, offset) in enumerate(zip(MODELS, offsets)):
            vals = [BASELINE[ds][model].get(m) for m in metrics]
            vals = [v if v is not None else 0 for v in vals]
            bars = ax.bar(x + offset, vals, width,
                          color=PALETTE[model], alpha=0.88,
                          label=model, edgecolor='white', linewidth=0.6)
            for bar, val in zip(bars, vals):
                if val > 0.02:
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + 0.01,
                            f'{val:.2f}', ha='center', va='bottom',
                            fontsize=6.8, color=PALETTE[model], fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=11)
        ax.set_ylim(0, 1.18)
        ax.set_ylabel('Valor', fontsize=11)
        ax.set_title(ds, fontsize=13, fontweight='bold',
                     color=DS_COLORS[ds], pad=10)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f'{y:.1f}'))

    handles = [mpatches.Patch(color=PALETTE[m], label=m) for m in MODELS]
    fig.legend(handles=handles, loc='lower center', ncol=4,
               fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle('Métricas de clasificación por modelo y dataset',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    _save(fig, 'g1_baseline_metrics.png')


# ══════════════════════════════════════════════════════════════
# GRÁFICO 2 — Heatmap de tasa de evasión
# ══════════════════════════════════════════════════════════════
def plot_evasion_heatmap():
    cmap = LinearSegmentedColormap.from_list(
        'rob', ['#1B5E20', '#FFEB3B', '#B71C1C'], N=256)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, ds in zip(axes, ['Credit Card', 'AMLworld HI']):
        matrix = []
        annot  = []
        for atk in ATTACKS:
            row_val = []
            row_ann = []
            for model in MODELS:
                v = EVASION[ds][atk][model]
                if np.isnan(v):
                    row_val.append(-0.05)
                    row_ann.append('N/A')
                else:
                    row_val.append(v)
                    row_ann.append(f'{v*100:.1f}%')
            matrix.append(row_val)
            annot.append(row_ann)

        mat = np.array(matrix, dtype=float)
        im  = ax.imshow(np.where(mat < 0, np.nan, mat),
                        cmap=cmap, vmin=0, vmax=1, aspect='auto')

        # Celdas N/A en gris
        gray = np.zeros((*mat.shape, 4))
        gray[mat < 0] = [0.85, 0.85, 0.85, 1.0]
        ax.imshow(gray, aspect='auto')

        for i in range(len(ATTACKS)):
            for j in range(len(MODELS)):
                txt   = annot[i][j]
                color = 'white' if (mat[i,j] > 0.55 or mat[i,j] < 0) else 'black'
                ax.text(j, i, txt, ha='center', va='center',
                        fontsize=9.5, fontweight='bold', color=color)

        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels(MODELS, fontsize=10)
        ax.set_yticks(range(len(ATTACKS)))
        ax.set_yticklabels(ATTACKS, fontsize=10)
        ax.set_title(ds, fontsize=13, fontweight='bold',
                     color=DS_COLORS[ds], pad=10)

        cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
        cb.set_label('Tasa de evasión', fontsize=9)
        cb.ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))

    fig.suptitle('Tasa de evasión por ataque y modelo',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    _save(fig, 'g2_evasion_heatmap.png')


# ══════════════════════════════════════════════════════════════
# GRÁFICO 3 — Radar de métricas basales
# ══════════════════════════════════════════════════════════════
def plot_radar():
    metrics = ['AUC-ROC', 'AUC-PR', 'Recall', 'F1']
    N       = len(metrics)
    angles  = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6),
                             subplot_kw=dict(polar=True))

    for ax, ds in zip(axes, ['Credit Card', 'AMLworld HI']):
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'],
                           fontsize=7.5, color='gray')
        ax.grid(color='gray', alpha=0.3)

        for model in MODELS:
            vals = [BASELINE[ds][model].get(m) or 0 for m in metrics]
            vals += vals[:1]
            ax.plot(angles, vals, linewidth=2,
                    color=PALETTE[model], label=model)
            ax.fill(angles, vals, alpha=0.10, color=PALETTE[model])

        ax.set_title(ds, fontsize=13, fontweight='bold',
                     color=DS_COLORS[ds], pad=18)

    handles = [mpatches.Patch(color=PALETTE[m], label=m) for m in MODELS]
    fig.legend(handles=handles, loc='lower center', ncol=4,
               fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle('Perfil de clasificación por modelo (radar)',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    _save(fig, 'g3_radar.png')


# ══════════════════════════════════════════════════════════════
# GRÁFICO 4 — Scatter por tipo de ataque (estilo paper)
# Forma = ataque | Color = modelo | Panel = dataset
# ══════════════════════════════════════════════════════════════
ATTACK_MARKERS = {
    'CaFA':           'o',    # círculo
    'SimBA':          '^',    # triángulo arriba
    'HopSkipJump':    's',    # cuadrado
    'BoundaryAttack': 'D',    # diamante
    'SquareAttack':   'P',    # cruz gruesa
}
ATTACK_LABELS = {
    'CaFA':           'CaFA',
    'SimBA':          'SimBA',
    'HopSkipJump':    'HopSkipJump',
    'BoundaryAttack': 'BoundaryAttack',
    'SquareAttack':   'SquareAttack',
}

def _smart_offset(x, y, x_range, y_range, all_points, idx):
    """Devuelve offset (dx, dy) evitando superposición con puntos cercanos."""
    candidates = [(8,4),(-55,4),(8,-14),(-55,-14),(8,14),(-55,14)]
    best = candidates[0]
    best_min_dist = 0
    for dx, dy in candidates:
        tx = x + dx / x_range * 0.05
        ty = y + dy / y_range * 0.05
        min_dist = min(
            ((tx - px)**2 + (ty - py)**2)**0.5
            for j, (px, py) in enumerate(all_points) if j != idx
        ) if len(all_points) > 1 else 999
        if min_dist > best_min_dist:
            best_min_dist = min_dist
            best = (dx, dy)
    return best

def plot_bubble():
    # 2 filas × 1 columna por dataset, pero cada fila tiene 5 paneles (uno por ataque)
    # Layout: 2 rows (datasets) × 5 cols (ataques), más columna de leyenda
    fig = plt.figure(figsize=(18, 9))
    gs  = fig.add_gridspec(2, 6, width_ratios=[1,1,1,1,1,0.45],
                           hspace=0.45, wspace=0.35)

    total_feat = {'Credit Card': 29, 'AMLworld HI': 15}

    for row, ds in enumerate(['Credit Card', 'AMLworld HI']):
        tf = total_feat[ds]
        for col, atk in enumerate(ATTACKS):
            ax = fig.add_subplot(gs[row, col])
            marker = ATTACK_MARKERS[atk]
            color_atk = ATTACK_PALETTE[atk]

            points = []
            labels_info = []
            for model in MODELS:
                ev = EVASION[ds][atk][model]
                l0 = L0[ds][atk][model]
                if np.isnan(ev) or np.isnan(l0):
                    continue
                points.append((l0, ev * 100))
                labels_info.append((l0, ev * 100, model))

            # Dibuja puntos
            for (l0, ev_pct, model) in labels_info:
                ax.scatter(l0, ev_pct,
                           s=180, marker=marker,
                           color=PALETTE[model],
                           edgecolors='white', linewidths=1.3,
                           zorder=4, alpha=0.92)

            # Etiquetas sin superposición
            for i, (l0, ev_pct, model) in enumerate(labels_info):
                all_xy = [(p[0], p[1]) for p in points]
                dx, dy = _smart_offset(l0, ev_pct, tf + 2, 115, all_xy, i)
                ax.annotate(
                    model,
                    (l0, ev_pct),
                    textcoords='offset points',
                    xytext=(dx, dy),
                    fontsize=7.2,
                    color=PALETTE[model],
                    fontweight='bold',
                    arrowprops=dict(
                        arrowstyle='-',
                        color=PALETTE[model],
                        lw=0.7, alpha=0.5
                    ) if abs(dx) > 10 else None,
                )

            # Zona de alta amenaza (bajo L0, alta evasión)
            threat_l0 = 6 if ds == 'AMLworld HI' else 15
            ax.add_patch(plt.Rectangle(
                (0, 75), threat_l0, 40,
                facecolor='red', alpha=0.06, zorder=0,
                linestyle='--', edgecolor='red', linewidth=0.5))

            ax.axhline(100, color='#EF4444', linestyle='--',
                       linewidth=0.8, alpha=0.35)
            ax.set_xlim(-0.5, tf + 1.5)
            ax.set_ylim(-8, 118)
            ax.set_xlabel('L0 (features)', fontsize=8)
            if col == 0:
                ax.set_ylabel(f'{ds}\nEvasión (%)', fontsize=8.5,
                              color=DS_COLORS[ds], fontweight='bold')
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
            ax.set_title(atk, fontsize=9, fontweight='bold',
                         color=color_atk, pad=6)
            ax.tick_params(labelsize=7.5)

            # Mensaje si no hay datos
            if not labels_info:
                ax.text(tf/2, 55, 'N/A\n(incompatible)',
                        ha='center', va='center', fontsize=8,
                        color='#9CA3AF', style='italic')
                ax.set_facecolor('#F9FAFB')

    # ── Leyenda en columna derecha ──
    ax_leg = fig.add_subplot(gs[:, 5])
    ax_leg.axis('off')

    # Por modelo (color)
    model_handles = [
        plt.scatter([], [], s=110, marker='o',
                    color=PALETTE[m], label=m, edgecolors='white', linewidths=1)
        for m in MODELS
    ]
    leg1 = ax_leg.legend(handles=model_handles,
                         title='Modelo\n(color)',
                         loc='upper center',
                         fontsize=8.5, title_fontsize=9,
                         framealpha=0.95,
                         bbox_to_anchor=(0.5, 1.0))
    ax_leg.add_artist(leg1)

    # Por ataque (forma)
    atk_handles = [
        plt.scatter([], [], s=110, marker=ATTACK_MARKERS[a],
                    color='#6B7280', label=a, edgecolors='white', linewidths=1)
        for a in ATTACKS
    ]
    ax_leg.legend(handles=atk_handles,
                  title='Ataque\n(forma)',
                  loc='lower center',
                  fontsize=8.5, title_fontsize=9,
                  framealpha=0.95,
                  bbox_to_anchor=(0.5, 0.38))

    fig.suptitle(
        'Eficiencia de ataques adversariales — L0 (features modificadas) vs Tasa de evasión\n'
        'Forma = tipo de ataque  |  Color = modelo  |  Zona roja = alta amenaza (bajo costo, alta evasión)',
        fontsize=11, fontweight='bold', y=1.01)

    _save(fig, 'g4_bubble_l0_evasion.png')


# ══════════════════════════════════════════════════════════════
# GRÁFICO 5 — Degradación del recall: antes vs después
# ══════════════════════════════════════════════════════════════
def plot_recall_before_after():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for ax, ds in zip(axes, ['Credit Card', 'AMLworld HI']):
        models_avail = [m for m in MODELS
                        if BASELINE[ds][m]['Recall'] is not None
                        and BASELINE[ds][m]['Recall'] > 0]
        attacks_plot = ATTACKS

        x      = np.arange(len(attacks_plot))
        n_m    = len(models_avail)
        width  = 0.7 / n_m

        for i, model in enumerate(models_avail):
            baseline_recall = BASELINE[ds][model]['Recall']
            recalls_after   = []
            for atk in attacks_plot:
                ev = EVASION[ds][atk][model]
                if np.isnan(ev):
                    recalls_after.append(None)
                else:
                    recalls_after.append((1 - ev) * baseline_recall)

            offset = (i - (n_m-1)/2) * width
            for j, ra in enumerate(recalls_after):
                if ra is None:
                    ax.bar(x[j] + offset, 0.02, width,
                           color='#E5E7EB', edgecolor='#9CA3AF',
                           linewidth=0.6, linestyle='--')
                else:
                    alpha = 0.55 + 0.4 * (1 - ra)
                    ax.bar(x[j] + offset, ra * 100, width,
                           color=PALETTE[model], alpha=alpha,
                           edgecolor='white', linewidth=0.5)

        # Línea de recall base
        for model in models_avail:
            br = BASELINE[ds][model]['Recall']
            if br:
                ax.axhline(br * 100, color=PALETTE[model],
                           linestyle=':', linewidth=1.2, alpha=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(attacks_plot, fontsize=9.5, rotation=15, ha='right')
        ax.set_ylim(0, 115)
        ax.set_ylabel('Recall de fraude post-ataque (%)', fontsize=10)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
        ax.set_title(ds, fontsize=13, fontweight='bold',
                     color=DS_COLORS[ds], pad=10)
        ax.text(len(attacks_plot)-0.5, 108,
                'Líneas punteadas = recall base',
                fontsize=7.5, color='gray', ha='right', style='italic')

    handles = [mpatches.Patch(color=PALETTE[m], label=m) for m in MODELS]
    fig.legend(handles=handles, loc='lower center', ncol=4,
               fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle('Recall de fraude post-ataque por modelo\n'
                 '(líneas punteadas = recall sin ataque)',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    _save(fig, 'g5_recall_after_attack.png')


# ══════════════════════════════════════════════════════════════
# GRÁFICO 6 — Comparación AUC-PR: el gran diferenciador
# ══════════════════════════════════════════════════════════════
def plot_aucpr_comparison():
    fig, ax = plt.subplots(figsize=(11, 5.5))

    datasets = ['Credit Card', 'AMLworld HI']
    models   = ['XGBoost', 'MLP', 'Log. Reg.', 'LSTM-Att.']
    x        = np.arange(len(models))
    width    = 0.35

    for i, (ds, offset) in enumerate(zip(datasets,
                                          [-width/2, width/2])):
        vals = [BASELINE[ds][m]['AUC-PR'] or 0 for m in models]
        bars = ax.bar(x + offset, vals, width,
                      color=DS_COLORS[ds], alpha=0.82,
                      label=ds, edgecolor='white', linewidth=0.8)
        for bar, val in zip(bars, vals):
            if val > 0.005:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom',
                        fontsize=8.5, fontweight='bold',
                        color=DS_COLORS[ds])

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('AUC-PR', fontsize=12)
    ax.set_title('AUC-PR por modelo y dataset\n'
                 '(métrica clave bajo imbalance extremo)',
                 fontsize=13, fontweight='bold', pad=12)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f'{y:.2f}'))

    # Anotación de diferencia
    ax.annotate('Caída dramática\npor imbalance 1:993',
                xy=(0, 0.1833), xytext=(0.6, 0.45),
                arrowprops=dict(arrowstyle='->', color='#6B7280'),
                fontsize=9, color='#374151',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='#FEF3C7', edgecolor='#D97706'))

    fig.tight_layout()
    _save(fig, 'g6_aucpr_comparison.png')


# ─────────────────────────────────────────────────────────────
def _save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Saved: {path}')


if __name__ == '__main__':
    print('Generando gráficos...')
    plot_baseline()
    plot_evasion_heatmap()
    plot_radar()
    plot_bubble()
    plot_recall_before_after()
    plot_aucpr_comparison()
    print('\nTodos los gráficos generados.')
