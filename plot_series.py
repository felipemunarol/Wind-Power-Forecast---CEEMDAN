import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

plt.rcParams.update({
    'font.size': 16,        # tamanho geral da fonte
    'axes.titlesize': 20,   # título do gráfico
    'axes.labelsize': 18,   # rótulos dos eixos
    'xtick.labelsize': 14,  # ticks x
    'ytick.labelsize': 14,  # ticks y
    'legend.fontsize': 12   # legenda
})


# =========================
# 1. Load data
# =========================
y_test = np.loadtxt('y_test.txt')

dates = np.loadtxt('dates.txt', dtype=str)
# 1️Concatena data + hora
dates_str = np.char.add(dates[:,0], ' ')
dates_str = np.char.add(dates_str, dates[:,1])
dates = pd.to_datetime(dates_str)

# =========================
# 2. Load ALL predictions
# =========================
pred_files = sorted(glob.glob("y_pred*.txt"))

predictions = {}

for file in pred_files:
    name = os.path.splitext(os.path.basename(file))[0]  # nome sem .txt
    predictions[name] = np.loadtxt(file)

# =========================
# 3. Plot
# =========================
plt.figure(figsize=(12,6))

# Real
plt.plot(y_test, label='Real', linewidth=2, color='black')

# Predições
for name, y_pred in predictions.items():
    # if len(y_pred) == len(y_test):
    plt.plot(y_pred, linestyle='--', label=name)

plt.title('Wind Power Forecast')
plt.xlabel('Samples')
plt.ylabel('Power')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()