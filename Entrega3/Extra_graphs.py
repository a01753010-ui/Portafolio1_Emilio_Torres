import numpy as np
import matplotlib.pyplot as plt

labels = ['Exp 1\n2 var → nuevo', 'Exp 2\n3 var afinado → nuevo', 'Exp 3\n4 var afinado → mismo']
acc  = [0.527, 0.512, 0.820]
base = [0.586, 0.586, 0.500]

x = np.arange(len(labels)); w = 0.38
plt.figure(figsize=(7.5, 3.8))
plt.bar(x - w/2, acc,  w, label='Accuracy del modelo', color='#4f81bd')
plt.bar(x + w/2, base, w, label='Baseline (clase mayoritaria)', color='#c0504d', alpha=0.7)
plt.xticks(x, labels); plt.ylabel('Accuracy'); plt.ylim(0, 1)
plt.title('Desempeño de clasificación en los tres experimentos')
for i, (a, b) in enumerate(zip(acc, base)):
    plt.text(i - w/2, a + 0.02, f'{a:.2f}', ha='center', fontsize=9)
    plt.text(i + w/2, b + 0.02, f'{b:.2f}', ha='center', fontsize=9)
plt.legend(); plt.tight_layout(); plt.show()


grupos = ['Modelo original\n(2 var)', 'Modelo afinado\n(3-4 var, relu)']
rmse_mismo = [7.87, 6.85]
rmse_nuevo = [12.11, 12.45]

x = np.arange(2); w = 0.38
plt.figure(figsize=(7, 3.6))
plt.bar(x - w/2, rmse_mismo, w, label='Mismo dataset (prueba)', color='#9bbb59')
plt.bar(x + w/2, rmse_nuevo, w, label='Dataset nuevo', color='#c0504d')
plt.xticks(x, grupos); plt.ylabel('RMSE (puntos)')
plt.title('RMSE: el afinado ayuda en el mismo dataset, no en el nuevo')
for i, (a, b) in enumerate(zip(rmse_mismo, rmse_nuevo)):
    plt.text(i - w/2, a + 0.15, f'{a:.1f}', ha='center', fontsize=9)
    plt.text(i + w/2, b + 0.15, f'{b:.1f}', ha='center', fontsize=9)
plt.legend(); plt.tight_layout(); plt.show()