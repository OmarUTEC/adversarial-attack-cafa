"""
Tabla 1 — Métricas baseline con SMOTE (pre-ataque)
Tabla 2 — Métricas de ataque sobre modelos SMOTE (post-ataque)
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Colores ────────────────────────────────────────────────
HDR_BG  = '#1E5FA8'
HDR_FG  = 'white'
ROW_A   = '#E8F4FD'   # azul muy claro
ROW_B   = '#FFFFFF'
BEST_BG = '#C8E6C9'   # verde claro — mejor valor
PEND_BG = '#FEF9C3'   # amarillo — pendiente
RED_FG  = '#DC2626'
GRN_FG  = '#15803D'


def make_table(ax, col_labels, rows, row_colors, title):
    ax.axis('off')
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        cellColours=row_colors,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.9)
    for j in range(len(col_labels)):
        table[0, j].set_facecolor(HDR_BG)
        table[0, j].set_text_props(color=HDR_FG, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold',
                 color='#1E3A5F', pad=14)
    return table


# ══════════════════════════════════════════════════════════════
# TABLA 1 — Baseline pre-ataque
# ══════════════════════════════════════════════════════════════
cols1 = ['Modelo', 'AUC-ROC', 'AUC-PR', 'Recall', 'F1']

rows1 = [
    ['XGBoost',        '0.9661', '0.0954', '74.22%', '3.66%'],
    ['MLP',            '0.9351', '—',      '82.34%', '1.84%'],
    ['Log. Reg.',      '0.8436', '0.0045', '78.59%', '0.95%'],
    ['LSTM-Att.',      '—',      '—',      '—',      '—'],
]

colors1 = [
    [ROW_A, ROW_A,  ROW_A,  ROW_A,  ROW_A],
    [ROW_B, ROW_B,  ROW_B,  BEST_BG, ROW_B],   # MLP mejor Recall
    [ROW_A, ROW_A,  ROW_A,  ROW_A,  ROW_A],
    [PEND_BG]*5,
]

fig1, ax1 = plt.subplots(figsize=(11, 3.2))
t1 = make_table(ax1, cols1, rows1, colors1,
                'Tabla 1 — Métricas Baseline con SMOTE (Pre-Ataque)\nAMLworld HI-Small')

# Resaltar valores
for i, r in enumerate(rows1, start=1):
    if r[3] != '—':
        val = float(r[3].replace('%',''))
        color = GRN_FG if val >= 70 else RED_FG if val == 0 else 'black'
        t1[i, 3].set_text_props(color=color, fontweight='bold')

legend = [
    mpatches.Patch(facecolor=BEST_BG, label='Mejor Recall'),
    mpatches.Patch(facecolor=PEND_BG, label='Pendiente'),
]
fig1.legend(handles=legend, loc='lower center',
            bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=9)

fig1.tight_layout()
path1 = os.path.join(OUT_DIR, 'tabla1_smote_baseline.png')
fig1.savefig(path1, bbox_inches='tight', dpi=160)
plt.close(fig1)
print(f'Saved: {path1}')


# ══════════════════════════════════════════════════════════════
# TABLA 2 — Resultados de ataques post-ataque
# ══════════════════════════════════════════════════════════════
cols2 = ['Modelo', 'Ataque', 'Evasión', 'Recall post', 'L0 medio', 'ΔFNR']

rows2 = [
    ['MLP',        'CaFA',          '98.76%', '1.24%',   '4.72 / 15', '0.988'],
    ['MLP',        'HopSkipJump',   '95.14%', '4.86%',   '9.53 / 15', '0.951'],
    ['MLP',        'BoundaryAttack','100.0%', '0.00%',   '4.53 / 15', '1.000'],
    ['MLP',        'SquareAttack',  '100.0%', '0.00%',   '6.17 / 15', '1.000'],
    ['XGBoost',    'HopSkipJump',   '99.38%', '0.62%',   '9.92 / 15', '0.994'],
    ['XGBoost',    'BoundaryAttack','100.0%', '0.00%',   '5.01 / 15', '1.000'],
    ['XGBoost',    'SquareAttack',  '100.0%', '0.00%',   '6.58 / 15', '1.000'],
    ['Log. Reg.',  'CaFA',          '100.0%', '0.00%',   '3.52 / 15', '1.000'],
    ['Log. Reg.',  'HopSkipJump',   '29.57%', '70.43%',  '5.89 / 15', '0.296'],
    ['Log. Reg.',  'BoundaryAttack','99.88%', '0.12%',   '4.75 / 15', '0.999'],
    ['Log. Reg.',  'SquareAttack',  '89.53%', '10.47%',  '7.27 / 15', '0.895'],
    ['LSTM-Att.',  'CaFA',          '68.58%', '31.42%',  '2.48 / 15', '0.686'],
    ['LSTM-Att.',  'HopSkipJump',   '0.00%',  '100.0%',  '0.00 / 15', '0.000'],
    ['LSTM-Att.',  'BoundaryAttack','0.48%',  '99.52%',  '0.04 / 15', '0.005'],
    ['LSTM-Att.',  'SquareAttack',  '0.19%',  '99.81%',  '0.78 / 15', '0.002'],
]

colors2 = []
for r in rows2:
    ev_str = r[2]
    if ev_str == '—':
        colors2.append([PEND_BG] * 6)
    elif ev_str == '100.0%':
        colors2.append(['#FEE2E2'] * 6)   # rojo claro — evasión total
    elif float(ev_str.replace('%', '')) < 5.0:
        colors2.append(['#C8E6C9'] * 6)   # verde — modelo robusto
    else:
        colors2.append([ROW_A if i % 2 == 0 else ROW_B for i in range(6)])

fig2, ax2 = plt.subplots(figsize=(13, 6))
t2 = make_table(ax2, cols2, rows2, colors2,
                'Tabla 2 — Métricas de Ataques con SMOTE (Post-Ataque)\nAMLworld HI-Small')

# Resaltar evasión 100%
for i, r in enumerate(rows2, start=1):
    if r[2] == '100.0%':
        t2[i, 2].set_text_props(color=RED_FG, fontweight='bold')
    if r[3] == '0.00%':
        t2[i, 3].set_text_props(color=RED_FG, fontweight='bold')

legend2 = [
    mpatches.Patch(facecolor='#FEE2E2', label='Evasión total (100%)'),
    mpatches.Patch(facecolor='#C8E6C9', label='Modelo robusto (evasión < 5%)'),
]
fig2.legend(handles=legend2, loc='lower center',
            bbox_to_anchor=(0.5, -0.06), ncol=2, fontsize=9)

fig2.tight_layout()
path2 = os.path.join(OUT_DIR, 'tabla2_smote_ataques.png')
fig2.savefig(path2, bbox_inches='tight', dpi=160)
plt.close(fig2)
print(f'Saved: {path2}')
