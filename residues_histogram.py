import  os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

DATA_DIR = "."  # pasta onde estão os txt

# =========================
# 1. Carregar dados
# =========================
y_test = np.loadtxt("y_test.txt")

pred_files = sorted(glob(os.path.join(DATA_DIR, "y_pred*.txt")))


for pred_file in pred_files:
    # =========================
    # 2. Resíduo
    # =========================
    y_pred = np.loadtxt(pred_file)
    residuo = y_test - y_pred

    mu = np.mean(residuo)
    std = np.std(residuo)

    # =========================
    # 3. Plot (paper style)
    # =========================
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
    })

    fig, ax = plt.subplots(figsize=(4.5,3.5))

    # histograma normalizado
    counts, bins, _ = ax.hist(
        residuo,
        bins=40,
        density=True,
        alpha=0.6
    )

    # curva normal ajustada
    x = np.linspace(bins.min(), bins.max(), 200)
    pdf = norm.pdf(x, mu, std)
    ax.plot(x, pdf, linewidth=2)

    # linhas de referência
    ax.axvline(mu, linestyle='--', linewidth=1)
    ax.axvline(0, linestyle=':', linewidth=1)

    # labels
    ax.set_xlabel("Residuals")
    ax.set_ylabel("Density")

    # anotação
    ax.text(
        0.95, 0.95,
        f"$\\mu={mu:.2f}$\n$\\sigma={std:.2f}$",
        transform=ax.transAxes,
        ha='right',
        va='top'
    )

    # grid leve
    ax.grid(alpha=0.3)

    plt.tight_layout()

    # =========================
    # 4. Salvar figura
    # =========================
    plt.savefig("residuals_histogram.png", dpi=300)
    plt.savefig("residuals_histogram.pdf")

    plt.show()