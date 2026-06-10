"""
Tabla comparativa de métricas baseline: Sin SMOTE vs Con SMOTE — AMLworld HI-Small
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Datos ────────────────────────────────────────────────────
rows = [
    # Modelo            Entren.       AUC-ROC   Recall    Prec.    F1
    ['XGBoost',        'Sin SMOTE',  '0.966',  '33.95%', '20.18%','20.32%'],
    ['XGBoost',        'Con SMOTE',  '0.966',  '74.22%', '1.95%', '3.66%'],
    ['MLP',            'Sin SMOTE',  '0.913',  '0.00%',  '0.00%', '0.00%'],
    ['MLP',            'Con SMOTE',  '0.935',  '82.34%', '0.93%', '1.84%'],
    ['Log. Reg.',      'Sin SMOTE',  '0.567',  '10.67%', '0.52%', '1.03%'],
    ['Log. Reg.',      'Con SMOTE',  '0.844',  '78.59%', '0.48%', '0.95%'],
    ['LSTM-Att.',      'Sin SMOTE',  '0.411',  '0.00%',  '0.00%', '0.00%'],
    ['LSTM-Att.',      'Con SMOTE',  '—',      '—',      '—',     '—'],
]

col_labels = ['Modelo', 'Entrenamiento', 'AUC-ROC', 'Recall', 'Precisión', 'F1']

# ── Colores por fila ─────────────────────────────────────────
row_colors = []
for r in rows:
    if r[1] == 'Con SMOTE' and r[2] != '—':
        row_colors.append(['#DCFCE7'] * len(col_labels))   # verde claro
    elif r[1] == 'Sin SMOTE':
        row_colors.append(['#F8FAFC'] * len(col_labels))   # gris muy claro
    else:
        row_colors.append(['#F1F5F9'] * len(col_labels))   # gris (pendiente)

fig, ax = plt.subplots(figsize=(13, 5))
ax.axis('off')

table = ax.table(
    cellText=rows,
    colLabels=col_labels,
    cellLoc='center',
    loc='center',
    rowLoc='center',
    cellColours=row_colors,
)

table.auto_set_font_size(False)
table.set_fontsize(10.5)
table.scale(1, 1.8)

# Estilo encabezado
for j in range(len(col_labels)):
    table[0, j].set_facecolor('#1E293B')
    table[0, j].set_text_props(color='white', fontweight='bold')

# Negrita para valores Recall con SMOTE
for i, r in enumerate(rows, start=1):
    if r[1] == 'Con SMOTE' and r[3] != '—':
        table[i, 3].set_text_props(fontweight='bold', color='#15803D')
    if r[1] == 'Sin SMOTE' and r[3] == '0.00%':
        table[i, 3].set_text_props(color='#DC2626', fontweight='bold')

fig.suptitle(
    'Métricas Baseline — Sin SMOTE vs Con SMOTE  |  AMLworld HI-Small',
    fontsize=12, fontweight='bold', color='#B45309', y=0.98)

# Leyenda
from matplotlib.patches import Patch
legend = [
    Patch(facecolor='#F8FAFC', edgecolor='#CBD5E1', label='Sin SMOTE'),
    Patch(facecolor='#DCFCE7', edgecolor='#86EFAC', label='Con SMOTE'),
    Patch(facecolor='#F1F5F9', edgecolor='#CBD5E1', label='Pendiente'),
]
ax.legend(handles=legend, loc='lower center', bbox_to_anchor=(0.5, -0.12),
          ncol=3, fontsize=9, framealpha=0.9)

path = os.path.join(OUT_DIR, 'tabla_smote_comparison.png')
fig.savefig(path, bbox_inches='tight', dpi=160)
plt.close(fig)
print(f'Saved: {path}')
