import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1. Load data
# =========================
y_test = np.loadtxt('y_test.txt')
y_pred = np.loadtxt('y_pred.txt')

dates = np.loadtxt('dates.txt', dtype=str)
dates = pd.to_datetime(dates)

# =========================
# 2. Plot
# =========================
plt.figure(figsize=(12,6))

plt.plot(dates, y_test, label='Real', linewidth=2)
plt.plot(dates, y_pred, label='Predicted', linestyle='--')

plt.title('Wind Power Forecast')
plt.xlabel('Time')
plt.ylabel('Power')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()