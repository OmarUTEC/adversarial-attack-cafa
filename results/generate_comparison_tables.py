"""
Genera tablas comparativas en PNG para ambos datasets.
Guarda los resultados en results/
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────
# DATOS
# ─────────────────────────────────────────────────────────────

BASELINE = {
    'Credit Card': {
        'XGBoost':    {'auc_roc': 1.0000, 'auc_pr': 1.0000, 'recall': 0.9997, 'f1': 0.9998},
        'MLP':        {'auc_roc': 1.0000, 'auc_pr': 1.0000, 'recall': 0.9998, 'f1': 0.9997},
        'Log. Reg.':  {'auc_roc': 0.9914, 'auc_pr': 0.9920, 'recall': 0.9921, 'f1': 0.9926},
        'LSTM-Att.':  {'auc_roc': 0.8183, 'auc_pr': 0.7372, 'recall': 0.9416, 'f1': 0.8194},
    },
    'AMLworld HI': {
        'XGBoost':    {'auc_roc': 0.9660, 'auc_pr': 0.1833, 'recall': 0.3395, 'f1': 0.2032},
        'MLP':        {'auc_roc': 0.9131, 'auc_pr': 0.0146, 'recall': 0.0000, 'f1': 0.0000},
        'Log. Reg.':  {'auc_roc': 0.5672, 'auc_pr': 0.0013, 'recall': 0.1067, 'f1': 0.0103},
        'LSTM-Att.':  {'auc_roc': None,   'auc_pr': None,   'recall': None,   'f1': None},
    },
}

# evasion = tasa de evasión post-ataque, l0 = features modificadas, N/A = no compatible
NA = None
ATTACKS = {
    'Credit Card': {
        # (evasion_after, l0_mean)
        'CaFA':          {'XGBoost': NA,            'MLP': (0.529, 11.94), 'Log. Reg.': (0.785, 9.62),  'LSTM-Att.': (0.237, 0.60)},
        'SimBA':         {'XGBoost': NA,            'MLP': (0.035, 2.99),  'Log. Reg.': (0.055, 1.37),  'LSTM-Att.': (0.221, 0.08)},
        'HopSkipJump':   {'XGBoost': (0.9995,29.97),'MLP': (0.998, 29.86),'Log. Reg.': (0.992, 29.83), 'LSTM-Att.': (0.779, 29.98)},
        'BoundaryAttack':{'XGBoost': (0.9995,29.85),'MLP': (0.480, 14.32),'Log. Reg.': (0.480, 14.14), 'LSTM-Att.': (0.521, 20.95)},
        'SquareAttack':  {'XGBoost': NA,            'MLP': (1.000, 11.92),'Log. Reg.': (1.000, 15.58), 'LSTM-Att.': (0.226, 0.42)},
    },
    'AMLworld HI': {
        'CaFA':          {'XGBoost': NA,           'MLP': NA,           'Log. Reg.': (1.000, 3.55),  'LSTM-Att.': None},
        'SimBA':         {'XGBoost': NA,           'MLP': NA,           'Log. Reg.': (1.000, 2.39),  'LSTM-Att.': None},
        'HopSkipJump':   {'XGBoost': (0.022, 2.56),'MLP': NA,          'Log. Reg.': (0.052, 2.78),  'LSTM-Att.': None},
        'BoundaryAttack':{'XGBoost': (1.000, 5.81),'MLP': NA,          'Log. Reg.': (1.000, 4.51),  'LSTM-Att.': None},
        'SquareAttack':  {'XGBoost': (1.000, 6.05),'MLP': NA,          'Log. Reg.': (1.000, 6.01),  'LSTM-Att.': None},
    },
}

MODELS   = ['XGBoost', 'MLP', 'Log. Reg.', 'LSTM-Att.']
ATTACKS_ = ['CaFA', 'SimBA', 'HopSkipJump', 'BoundaryAttack', 'SquareAttack']
DATASETS = ['Credit Card', 'AMLworld HI']

# ─────────────────────────────────────────────────────────────
# TABLA 1 — Métricas basales
# ─────────────────────────────────────────────────────────────
def plot_baseline_table():
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.axis('off')

    cols = ['Dataset', 'Modelo', 'AUC-ROC', 'AUC-PR', 'Recall', 'F1']
    rows = []
    for ds in DATASETS:
        for model, m in BASELINE[ds].items():
            auc_roc = f"{m['auc_roc']:.4f}" if m['auc_roc'] is not None else '—'
            auc_pr  = f"{m['auc_pr']:.4f}"  if m['auc_pr']  is not None else '—'
            recall  = f"{m['recall']*100:.2f}%" if m['recall'] is not None else '—'
            f1      = f"{m['f1']:.4f}"      if m['f1']      is not None else '—'
            rows.append([ds, model, auc_roc, auc_pr, recall, f1])

    table = ax.table(
        cellText=rows,
        colLabels=cols,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.7)

    # Estilo encabezado
    for j in range(len(cols)):
        table[0, j].set_facecolor('#1565C0')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Colores por dataset y destacar mejores
    ds_colors = {'Credit Card': '#E3F2FD', 'AMLworld HI': '#FFF3E0'}
    best_per_ds = {
        'Credit Card': {'XGBoost', 'MLP'},
        'AMLworld HI': {'XGBoost'},
    }
    for i, row in enumerate(rows):
        ds    = row[0]
        model = row[1]
        bg    = ds_colors[ds]
        for j in range(len(cols)):
            cell = table[i + 1, j]
            cell.set_facecolor(bg)
            cell.set_edgecolor('#BBDEFB')
        if model in best_per_ds[ds]:
            for j in range(len(cols)):
                table[i + 1, j].set_text_props(fontweight='bold')
            table[i + 1, 1].set_facecolor('#A5D6A7')  # verde para mejor modelo

    ax.set_title('Tabla 1 — Métricas de clasificación por modelo y dataset',
                 fontsize=13, fontweight='bold', pad=14)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'tabla1_baseline.png')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Saved: {path}')


# ─────────────────────────────────────────────────────────────
# TABLA 2 — Tasa de evasión por ataque, modelo y dataset
# ─────────────────────────────────────────────────────────────
def plot_attack_evasion_table():
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    for ax, ds in zip(axes, DATASETS):
        ax.axis('off')
        cols = ['Ataque'] + MODELS
        rows = []
        cell_colors = []

        for atk in ATTACKS_:
            row   = [atk]
            color = ['#ECEFF1']
            for model in MODELS:
                val = ATTACKS[ds][atk].get(model)
                if val is None and ds == 'AMLworld HI' and model in ('MLP', 'LSTM-Att.'):
                    row.append('Sin TP')
                    color.append('#F5F5F5')
                elif val is None:
                    row.append('N/A')
                    color.append('#FFECB3')
                else:
                    evasion = val[0]
                    row.append(f'{evasion*100:.1f}%')
                    if evasion >= 0.90:
                        color.append('#FFCDD2')   # rojo: alta evasión = peligroso
                    elif evasion >= 0.40:
                        color.append('#FFE0B2')   # naranja: media
                    elif evasion > 0.05:
                        color.append('#FFF9C4')   # amarillo: baja
                    else:
                        color.append('#C8E6C9')   # verde: resistente
            rows.append(row)
            cell_colors.append(color)

        table = ax.table(
            cellText=rows,
            colLabels=cols,
            cellLoc='center',
            loc='center',
            cellColours=cell_colors,
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9.5)
        table.scale(1, 1.9)

        for j in range(len(cols)):
            table[0, j].set_facecolor('#1565C0')
            table[0, j].set_text_props(color='white', fontweight='bold')

        ax.set_title(f'{ds}', fontsize=12, fontweight='bold', pad=10)

    # Leyenda de colores
    legend = [
        mpatches.Patch(color='#FFCDD2', label='≥ 90%  — Alta vulnerabilidad'),
        mpatches.Patch(color='#FFE0B2', label='40–90% — Vulnerabilidad media'),
        mpatches.Patch(color='#FFF9C4', label='5–40%  — Vulnerabilidad baja'),
        mpatches.Patch(color='#C8E6C9', label='< 5%   — Resistente'),
        mpatches.Patch(color='#FFECB3', label='N/A    — No compatible'),
    ]
    fig.legend(handles=legend, loc='lower center', ncol=5,
               fontsize=8.5, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle('Tabla 2 — Tasa de evasión (%) por ataque, modelo y dataset',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'tabla2_evasion.png')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Saved: {path}')


# ─────────────────────────────────────────────────────────────
# TABLA 3 — Costo de perturbación L0 (features modificadas)
# ─────────────────────────────────────────────────────────────
def plot_l0_table():
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    for ax, ds in zip(axes, DATASETS):
        ax.axis('off')
        cols = ['Ataque'] + MODELS
        rows = []
        cell_colors = []

        total_features = 29 if ds == 'Credit Card' else 15

        for atk in ATTACKS_:
            row   = [atk]
            color = ['#ECEFF1']
            for model in MODELS:
                val = ATTACKS[ds][atk].get(model)
                if val is None and ds == 'AMLworld HI' and model in ('MLP', 'LSTM-Att.'):
                    row.append('Sin TP')
                    color.append('#F5F5F5')
                elif val is None:
                    row.append('N/A')
                    color.append('#FFECB3')
                else:
                    l0 = val[1]
                    pct = l0 / total_features
                    row.append(f'{l0:.1f} / {total_features}')
                    if pct >= 0.80:
                        color.append('#FFCDD2')
                    elif pct >= 0.40:
                        color.append('#FFE0B2')
                    elif pct >= 0.15:
                        color.append('#FFF9C4')
                    else:
                        color.append('#C8E6C9')
            rows.append(row)
            cell_colors.append(color)

        table = ax.table(
            cellText=rows,
            colLabels=cols,
            cellLoc='center',
            loc='center',
            cellColours=cell_colors,
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9.5)
        table.scale(1, 1.9)

        for j in range(len(cols)):
            table[0, j].set_facecolor('#4A148C')
            table[0, j].set_text_props(color='white', fontweight='bold')

        ax.set_title(f'{ds}  (total features: {total_features})',
                     fontsize=12, fontweight='bold', pad=10)

    legend = [
        mpatches.Patch(color='#FFCDD2', label='≥ 80% features — Muy costoso'),
        mpatches.Patch(color='#FFE0B2', label='40–80%         — Costoso'),
        mpatches.Patch(color='#FFF9C4', label='15–40%         — Moderado'),
        mpatches.Patch(color='#C8E6C9', label='< 15%          — Eficiente (pocos cambios)'),
        mpatches.Patch(color='#FFECB3', label='N/A            — No compatible'),
    ]
    fig.legend(handles=legend, loc='lower center', ncol=5,
               fontsize=8.5, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle('Tabla 3 — Costo de perturbación L0 (features modificadas / total)',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'tabla3_l0_cost.png')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Saved: {path}')


if __name__ == '__main__':
    plot_baseline_table()
    plot_attack_evasion_table()
    plot_l0_table()
    print('\nTodas las tablas generadas.')
