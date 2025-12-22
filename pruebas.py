import matplotlib.pyplot as plt

# Datos para los métodos y ataques adversariales
methods = [
    'CAFA 10k (Logistic Regression)', 'HSJ 10k (Logistic Regression)',
    'CAFA 10k (LSTM-Attention)', 'HSJ 10k (LSTM-Attention)'
]

# Tasa de clasificación incorrecta antes del ataque
before_attack = [
    0.0075, 0.0075,  # CAFA y HSJ con Logistic Regression
    0.2126, 0.2126   # CAFA y HSJ con LSTM-Attention
]

# Tasa de clasificación incorrecta después de los ataques
after_attack_cafa_rl = [0.7077, None, None, None]  # CAFA para Logistic Regression
after_attack_hsj_rl = [None, 0.9925, None, None]  # Hop Skip Jump para Logistic Regression
after_attack_cafa_lstm = [None, None, 0.7874, None]  # CAFA para LSTM-Attention
after_attack_hsj_lstm = [None, None, None, 0.9925]  # Hop Skip Jump para LSTM-Attention

# Crear las gráficas para Logistic Regression
plt.figure(figsize=(12, 8))

# CAFA Logistic Regression
bars1 = plt.bar(methods[0], before_attack[0], width=0.4, label="Before Attack (CAFA LR)", align='center', color='blue')
bars2 = plt.bar(methods[0], after_attack_cafa_rl[0], width=0.4, label="After Attack (CAFA LR)", align='edge', color='orange')

# Hop Skip Jump Logistic Regression
bars3 = plt.bar(methods[1], before_attack[1], width=0.4, label="Before Attack (HSJ LR)", align='center', color='blue')
bars4 = plt.bar(methods[1], after_attack_hsj_rl[1], width=0.4, label="After Attack (HSJ LR)", align='edge', color='orange')

plt.xlabel("Métodos de Regresión Logística")
plt.ylabel("Tasa de Incorrectos (Misclassification Rate)")
#plt.title("Tasa de Incorrectos Antes y Después de los Ataques Adversariales para Regresión Logística")
plt.legend()

# Agregar los valores de los porcentajes sobre las barras
for bar in bars1 + bars2 + bars3 + bars4:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom', fontsize=10)

# Guardar la gráfica como un archivo .png
plt.tight_layout()
plt.savefig("misclassification_results_lr.png", format='png')

# Mostrar la gráfica
plt.show()

# Crear las gráficas para LSTM-Attention
plt.figure(figsize=(12, 8))

# Colores
color_pre ='blue'  # Azul profesional
color_post = 'orange' # Rojo suave (No tan agresivo como el 'red' puro)

# CAFA LSTM-Attention
bars5 = plt.bar(methods[2], before_attack[2], width=0.4, label="Before Attack (CAFA LSTM)", align='center', color=color_pre)
bars6 = plt.bar(methods[2], after_attack_cafa_lstm[2], width=0.4, label="After Attack (CAFA LSTM)", align='edge', color=color_post)

# Hop Skip Jump LSTM-Attention
bars7 = plt.bar(methods[3], before_attack[3], width=0.4, label="Before Attack (HSJ LSTM)", align='center', color=color_pre)
bars8 = plt.bar(methods[3], after_attack_hsj_lstm[3], width=0.4, label="After Attack (HSJ LSTM)", align='edge', color=color_post)

plt.xlabel("Métodos LSTM-Attention")
plt.ylabel("Tasa de Incorrectos (Misclassification Rate)")
#plt.title("Tasa de Incorrectos Antes y Después de los Ataques Adversariales para LSTM-Attention")
plt.legend()

# Agregar los valores de los porcentajes sobre las barras
for bar in bars5 + bars6 + bars7 + bars8:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom', fontsize=10)

# Guardar la gráfica como un archivo .png
plt.tight_layout()
plt.savefig("misclassification_results_lstm.png", format='png')

# Mostrar la gráfica
plt.show()
