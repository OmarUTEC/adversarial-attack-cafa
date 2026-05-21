"""
Genera los 3 gráficos más representativos para análisis de ataques adversariales
sobre AMLworld HI-Small (IBM).
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ──────────────────────────────────────────────
# Datos experimentales
# ──────────────────────────────────────────────

# Métricas basales (test set completo)
BASELINE = {
    'XGBoost':  {'auc_roc': 0.9660, 'auc_pr': 0.1833, 'recall': 0.3395, 'f1': 0.2032},
    'MLP':      {'auc_roc': 0.9131, 'auc_pr': 0.0146, 'recall': 0.0000, 'f1': 0.0000},
    'Log. Reg.':{'auc_roc': 0.5672, 'auc_pr': 0.0013, 'recall': 0.1067, 'f1': 0.0103},
}

# Resultados de ataques: (evasion_rate, l0_mean, delta_fnr, recall_after)
ATTACKS = {
    'XGBoost': {
        'HopSkipJump':   {'evasion': 0.0217, 'l0': 2.56,  'delta_fnr': 0.022,  'recall_after': 0.9783},
        'BoundaryAttack':{'evasion': 1.0000, 'l0': 5.81,  'delta_fnr': 1.000,  'recall_after': 0.0000},
        'SquareAttack':  {'evasion': 1.0000, 'l0': 6.05,  'delta_fnr': 1.000,  'recall_after': 0.0000},
    },
    'Log. Reg.': {
        'HopSkipJump':   {'evasion': 0.0517, 'l0': 2.78,  'delta_fnr': 0.052,  'recall_after': 0.9483},
        'BoundaryAttack':{'evasion': 1.0000, 'l0': 4.51,  'delta_fnr': 1.000,  'recall_after': 0.0000},
        'SquareAttack':  {'evasion': 1.0000, 'l0': 6.01,  'delta_fnr': 1.000,  'recall_after': 0.0000},
        'CaFA':          {'evasion': 1.0000, 'l0': 3.55,  'delta_fnr': 1.000,  'recall_after': 0.0000},
        'SimBA':         {'evasion': 1.0000, 'l0': 2.39,  'delta_fnr': 1.000,  'recall_after': 0.0000},
    },
}

# ──────────────────────────────────────────────
# Paleta de colores consistente
# ──────────────────────────────────────────────
ATTACK_COLORS = {
    'HopSkipJump':    '#4C72B0',
    'BoundaryAttack': '#DD8452',
    'SquareAttack':   '#55A868',
    'CaFA':           '#C44E52',
    'SimBA':          '#8172B2',
}
MODEL_COLORS = {
    'XGBoost':   '#2196F3',
    'Log. Reg.': '#FF9800',
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 150,
})


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICO 1 — Tasa de evasión por ataque y modelo (grouped bar chart)
# El más usado en papers de adversarial ML para mostrar vulnerabilidad
# ══════════════════════════════════════════════════════════════════════════════
def plot_evasion_rate():
    fig, ax = plt.subplots(figsize=(10, 5.5))

    attacks_xgb  = list(ATTACKS['XGBoost'].keys())
    attacks_lr   = list(ATTACKS['Log. Reg.'].keys())
    all_attacks  = ['HopSkipJump', 'BoundaryAttack', 'SquareAttack', 'CaFA', 'SimBA']

    x = np.arange(len(all_attacks))
    width = 0.35

    xgb_vals = []
    lr_vals  = []
    for atk in all_attacks:
        xgb_vals.append(ATTACKS['XGBoost'].get(atk, {}).get('evasion', None))
        lr_vals.append(ATTACKS['Log. Reg.'].get(atk, {}).get('evasion', None))

    # Barras XGBoost
    for i, (val, atk) in enumerate(zip(xgb_vals, all_attacks)):
        if val is not None:
            bar = ax.bar(x[i] - width/2, val * 100, width,
                         color=MODEL_COLORS['XGBoost'], alpha=0.85,
                         edgecolor='white', linewidth=0.8)
            ax.text(x[i] - width/2, val * 100 + 1.5,
                    f'{val*100:.1f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold',
                    color=MODEL_COLORS['XGBoost'])
        else:
            ax.bar(x[i] - width/2, 0, width, color='#CCCCCC', alpha=0.4,
                   edgecolor='#AAAAAA', linewidth=0.8, linestyle='--')
            ax.text(x[i] - width/2, 2, 'N/A', ha='center', va='bottom',
                    fontsize=7.5, color='#888888', style='italic')

    # Barras LogReg
    for i, (val, atk) in enumerate(zip(lr_vals, all_attacks)):
        if val is not None:
            bar = ax.bar(x[i] + width/2, val * 100, width,
                         color=MODEL_COLORS['Log. Reg.'], alpha=0.85,
                         edgecolor='white', linewidth=0.8)
            ax.text(x[i] + width/2, val * 100 + 1.5,
                    f'{val*100:.1f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold',
                    color=MODEL_COLORS['Log. Reg.'])

    ax.set_xticks(x)
    ax.set_xticklabels(all_attacks, fontsize=10)
    ax.set_ylabel('Tasa de evasión (%)', fontsize=11)
    ax.set_title('Tasa de evasión por ataque y modelo\nAMLworld HI-Small', fontsize=13, fontweight='bold', pad=12)
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))

    legend_patches = [
        mpatches.Patch(color=MODEL_COLORS['XGBoost'],   label='XGBoost'),
        mpatches.Patch(color=MODEL_COLORS['Log. Reg.'], label='Logistic Regression'),
        mpatches.Patch(color='#CCCCCC', alpha=0.6, label='No aplicable (requiere gradientes)'),
    ]
    ax.legend(handles=legend_patches, loc='upper left', framealpha=0.9, fontsize=9)

    # Línea de referencia 100%
    ax.axhline(100, color='red', linestyle='--', linewidth=0.8, alpha=0.5)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig1_evasion_rate.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICO 2 — Costo de perturbación vs. tasa de evasión (scatter plot)
# Muestra la eficiencia de cada ataque: ¿cuántas features hay que cambiar
# para lograr la evasión? Usado en papers para comparar el "costo del ataque".
# ══════════════════════════════════════════════════════════════════════════════
def plot_cost_vs_evasion():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, (model, atk_data) in zip(axes, ATTACKS.items()):
        for atk_name, vals in atk_data.items():
            evasion = vals['evasion'] * 100
            l0      = vals['l0']
            color   = ATTACK_COLORS[atk_name]
            size    = 200 + evasion * 3

            ax.scatter(l0, evasion, s=size, color=color, alpha=0.85,
                       edgecolors='white', linewidths=1.5, zorder=3)
            # Etiqueta del ataque
            offset_y = 4 if evasion < 95 else -8
            ax.annotate(atk_name, (l0, evasion),
                        textcoords='offset points', xytext=(6, offset_y),
                        fontsize=8.5, color=color, fontweight='bold')

        ax.set_xlim(0, 8)
        ax.set_ylim(-5, 115)
        ax.set_xlabel('L₀ medio (features modificadas)', fontsize=10)
        ax.set_title(f'{model}', fontsize=12, fontweight='bold')
        ax.axhline(100, color='red', linestyle='--', linewidth=0.8, alpha=0.4, label='100% evasión')
        ax.axhline(0,   color='gray', linestyle='-',  linewidth=0.5, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))

        # Región de "bajo costo - alta evasión" (esquina superior izquierda = peor para el defensor)
        ax.add_patch(plt.Rectangle((0, 85), 4, 30, fill=True,
                                   facecolor='red', alpha=0.06, zorder=0))
        ax.text(0.15, 112, 'Alta amenaza', fontsize=7.5, color='red', alpha=0.7, style='italic')

    axes[0].set_ylabel('Tasa de evasión (%)', fontsize=10)

    # Leyenda global de ataques
    legend_patches = [mpatches.Patch(color=c, label=a) for a, c in ATTACK_COLORS.items()]
    fig.legend(handles=legend_patches, title='Ataque', loc='lower center',
               ncol=5, framealpha=0.9, fontsize=8.5, bbox_to_anchor=(0.5, -0.05))

    fig.suptitle('Eficiencia del ataque: costo de perturbación vs. tasa de evasión\nAMLworld HI-Small',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig2_cost_vs_evasion.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICO 3 — Degradación del recall antes vs. después del ataque
# Muestra el impacto real del atacante sobre la capacidad de detección de fraude.
# Usado en papers de seguridad en ML como métrica operacional principal.
# ══════════════════════════════════════════════════════════════════════════════
def plot_recall_degradation():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)

    for ax, (model, atk_data) in zip(axes, ATTACKS.items()):
        attacks = list(atk_data.keys())
        recall_before = [1.0] * len(attacks)
        recall_after  = [atk_data[a]['recall_after'] for a in attacks]

        y = np.arange(len(attacks))
        height = 0.38

        bars_before = ax.barh(y + height/2, [r * 100 for r in recall_before],
                               height, color='#4CAF50', alpha=0.75, label='Antes del ataque')
        bars_after  = ax.barh(y - height/2, [r * 100 for r in recall_after],
                               height,
                               color=[ATTACK_COLORS[a] for a in attacks],
                               alpha=0.85, label='Después del ataque')

        # Etiquetas de valor
        for i, (rb, ra) in enumerate(zip(recall_before, recall_after)):
            ax.text(rb * 100 + 0.5, i + height/2, f'{rb*100:.0f}%',
                    va='center', fontsize=8, color='#2E7D32', fontweight='bold')
            label = f'{ra*100:.1f}%' if ra > 0 else '0%'
            ax.text(ra * 100 + 0.5 if ra > 0.05 else 1.5,
                    i - height/2, label,
                    va='center', fontsize=8,
                    color=ATTACK_COLORS[attacks[i]], fontweight='bold')

        ax.set_yticks(y)
        ax.set_yticklabels(attacks, fontsize=10)
        ax.set_xlim(0, 112)
        ax.set_xlabel('Recall de fraude (%)', fontsize=10)
        ax.set_title(f'{model}', fontsize=12, fontweight='bold')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
        ax.axvline(100, color='green', linestyle='--', linewidth=0.8, alpha=0.4)

    axes[0].set_ylabel('Ataque', fontsize=10)

    # Leyenda
    green_patch  = mpatches.Patch(color='#4CAF50', alpha=0.75, label='Antes del ataque (recall = 100%)')
    attack_patch = mpatches.Patch(color='#888888', alpha=0.75, label='Después del ataque (por color = ataque)')
    fig.legend(handles=[green_patch, attack_patch], loc='lower center',
               ncol=2, framealpha=0.9, fontsize=9, bbox_to_anchor=(0.5, -0.05))

    fig.suptitle('Degradación del recall de fraude bajo ataque adversarial\nAMLworld HI-Small',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig3_recall_degradation.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


# ══════════════════════════════════════════════════════════════════════════════
# BONUS — Métricas basales de los 3 modelos (radar / grouped bar)
# Muestra la comparación de clasificación antes de cualquier ataque
# ══════════════════════════════════════════════════════════════════════════════
def plot_baseline_metrics():
    fig, ax = plt.subplots(figsize=(9, 5))

    metrics      = ['AUC-ROC', 'AUC-PR', 'Recall\nfraude', 'F1\nfraude']
    models       = list(BASELINE.keys())
    model_colors = ['#2196F3', '#9E9E9E', '#FF9800']
    keys         = ['auc_roc', 'auc_pr', 'recall', 'f1']

    x     = np.arange(len(metrics))
    width = 0.25

    for i, (model, color) in enumerate(zip(models, model_colors)):
        vals = [BASELINE[model][k] for k in keys]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=model,
                      color=color, alpha=0.85, edgecolor='white', linewidth=0.8)
        for bar, val in zip(bars, vals):
            if val > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=7.5,
                        color=color, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel('Valor de métrica', fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_title('Métricas de clasificación por modelo (sin ataque)\nAMLworld HI-Small',
                 fontsize=13, fontweight='bold', pad=12)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))

    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig0_baseline_metrics.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


if __name__ == '__main__':
    plot_baseline_metrics()
    plot_evasion_rate()
    plot_cost_vs_evasion()
    plot_recall_degradation()
    print('\nTodos los gráficos generados correctamente.')
