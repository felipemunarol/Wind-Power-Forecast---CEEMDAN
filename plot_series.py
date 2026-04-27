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

cv = False

# =========================
# 1. Load data
# =========================

if cv:
    dates = np.loadtxt('dates_cv.txt', dtype=str)
    y_test = np.loadtxt('y_test_cv.txt')
else:
    dates = np.loadtxt('dates.txt', dtype=str)
    y_test = np.loadtxt('y_test.txt')

dates = np.asarray(dates).reshape(-1)
dates = pd.to_datetime(dates, errors='coerce')

# =========================
# 2. Load ALL predictions
# =========================
pred_files = sorted(glob.glob("y_pred*.txt"))

predictions = {}

for file in sorted(pred_files):
    filename = os.path.basename(file)

    if cv:
        # pega só arquivos com _cv
        if not filename.endswith("_cv.txt"):
            continue
    else:
        # pega só arquivos SEM _cv
        if filename.endswith("_cv.txt"):
            continue

    name = os.path.splitext(filename)[0]
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