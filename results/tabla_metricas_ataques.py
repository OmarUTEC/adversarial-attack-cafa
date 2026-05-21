"""
Genera tabla de definición de métricas para ataques adversariales.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

def plot_metrics_table():
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.axis('off')

    cols = ['Métrica', 'Qué mide', 'Mejor valor\n(desde la perspectiva del defensor)']

    rows = [
        ['Tasa de Evasión',    'Proporción de ejemplos adversariales clasificados\ncomo legítimos tras la perturbación.',                     '0 %\n(modelo resistente)'],
        ['ΔFNR',               'Incremento en la tasa de fraudes no detectados\ninducido por el ataque.',                                    '0.0\n(sin degradación)'],
        ['Recall post-ataque', 'Proporción de fraudes que el modelo sigue\ndetectando correctamente tras el ataque.',                       '100 %\n(sin pérdida de detección)'],
        ['L0',                 'Número de características que el atacante\ndebe modificar para lograr la evasión.',                         'Alto\n(ataque costoso de ejecutar)'],
        ['L∞ estandarizada',   'Magnitud máxima de perturbación en cualquier\ncaracterística, normalizada por su rango válido.',            'Alto\n(cambios fácilmente detectables)'],
    ]

    cell_colors = []
    row_colors = ['#E3F2FD', '#E8F5E9', '#FFF3E0', '#F3E5F5', '#FCE4EC']
    for i, rc in enumerate(row_colors):
        cell_colors.append([rc] * len(cols))

    table = ax.table(
        cellText=rows,
        colLabels=cols,
        cellLoc='center',
        loc='center',
        cellColours=cell_colors,
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 3.2)

    # Encabezado
    for j in range(len(cols)):
        table[0, j].set_facecolor('#1565C0')
        table[0, j].set_text_props(color='white', fontweight='bold', fontsize=10)

    # Negrita en columna Métrica
    for i in range(1, len(rows) + 1):
        table[i, 0].set_text_props(fontweight='bold')

    ax.set_title('Métricas de evaluación para ataques adversariales',
                 fontsize=13, fontweight='bold', pad=16)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'tabla_metricas_ataques.png')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Saved: {path}')

if __name__ == '__main__':
    plot_metrics_table()
