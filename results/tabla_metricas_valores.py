"""
Tabla de métricas adversariales con valores reales de los experimentos.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

def plot():
    fig, ax = plt.subplots(figsize=(17, 6.5))
    ax.axis('off')

    cols = [
        'Métrica',
        'Mejor valor\n(defensor)',
        'Mejor resultado\nexperimental',
        'Peor resultado\nexperimental',
        'Ataque más\nresistido',
        'Ataque más\nefectivo',
    ]

    rows = [
        [
            'Tasa de Evasión',
            '0%',
            '2.2%\n(HSJ vs XGBoost — AMLworld)',
            '100%\n(Boundary/Square/CaFA/SimBA\nvs múltiples modelos)',
            'HopSkipJump',
            'BoundaryAttack\nSquareAttack\nCaFA / SimBA',
        ],
        [
            'ΔFNR',
            '0.0',
            '+0.022\n(HSJ vs XGBoost — AMLworld)',
            '+1.000\n(evasión total — AMLworld\ny Credit Card)',
            'HopSkipJump',
            'BoundaryAttack\nSquareAttack',
        ],
        [
            'Recall post-ataque',
            '100%',
            '97.8%\n(HSJ vs XGBoost — AMLworld)',
            '0%\n(evasión total en\nvarios ataques)',
            'HopSkipJump',
            'BoundaryAttack\nSquareAttack',
        ],
        [
            'L0\n(features modificadas)',
            'Alto\n(ataque costoso)',
            '0.42 features\n(Square vs LSTM — Credit Card)',
            '29.97 features\n(HSJ vs XGBoost — Credit Card)\n[29 features en total]',
            'SquareAttack\nvs LSTM-Att.',
            'HopSkipJump\n(modifica casi todo)',
        ],
        [
            'L∞ estandarizada\n(magnitud cambio)',
            'Alto\n(cambios visibles)',
            '~1.9×10⁻⁷\n(SimBA vs LogReg — AMLworld)',
            '88.76\n(HSJ vs XGBoost — Credit Card)',
            'SimBA\n(cambios mínimos)',
            'HopSkipJump\nBoundaryAttack',
        ],
    ]

    row_colors = [
        ['#E3F2FD'] * len(cols),
        ['#E8F5E9'] * len(cols),
        ['#FFF3E0'] * len(cols),
        ['#F3E5F5'] * len(cols),
        ['#FCE4EC'] * len(cols),
    ]

    table = ax.table(
        cellText=rows,
        colLabels=cols,
        cellLoc='center',
        loc='center',
        cellColours=row_colors,
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 4.5)

    # Encabezado
    for j in range(len(cols)):
        table[0, j].set_facecolor('#1565C0')
        table[0, j].set_text_props(color='white', fontweight='bold', fontsize=9)

    # Negrita columna métrica
    for i in range(1, len(rows) + 1):
        table[i, 0].set_text_props(fontweight='bold')
        # Resaltar mejor resultado (col 2) en verde
        table[i, 2].set_facecolor('#C8E6C9')
        # Resaltar peor resultado (col 3) en rojo claro
        table[i, 3].set_facecolor('#FFCDD2')

    ax.set_title(
        'Métricas adversariales — valores obtenidos en la experimentación\n'
        '(Credit Card 2023  |  AMLworld HI-Small)',
        fontsize=12, fontweight='bold', pad=14,
    )

    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'tabla_metricas_valores.png')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Saved: {path}')

if __name__ == '__main__':
    plot()
