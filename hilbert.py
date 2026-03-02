import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert


def hilbert_spectrum(imfs, Fs, ER=1.0, B=512, fmax=None, plot=True):
    """
    Hilbert Spectrum + Marginal Hilbert Spectrum
    
    Parameters
    ----------
    imfs : ndarray (N, num_imfs)
        Matrix of IMFs (each column = one IMF)
    Fs : float
        Sampling frequency
    ER : float
        Scaling factor (Hz=1, rpm=60)
    B : int
        Number of frequency bins
    fmax : float
        Max frequency for plot
    plot : bool
        If True, generates plots
        
    Returns
    -------
    MHS : ndarray (B,)
        Marginal Hilbert Spectrum
    fp : ndarray (B,)
        Frequency axis
    """

    imfs = np.asarray(imfs)
    N, num_imfs = imfs.shape
    t = np.arange(N) / Fs
    tmax = N / Fs

    # =============================
    # 1) Hilbert Transform
    # =============================
    analytic = hilbert(imfs, axis=0)
    amplitude = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic), axis=0)

    # Instantaneous frequency
    inst_freq = (Fs / (2 * np.pi)) * np.diff(phase, axis=0)
    inst_freq = np.vstack([inst_freq, np.zeros((1, num_imfs))])
    inst_freq *= ER

    # =============================
    # 2) Frequency smoothing
    # =============================
    span = max(1, int(0.01 * Fs))
    kernel = np.ones(span) / span

    for i in range(num_imfs):
        inst_freq[:, i] = np.convolve(inst_freq[:, i], kernel, mode='same')

    # =============================
    # 3) Build Hilbert Spectrum
    # =============================
    f_abs = np.abs(inst_freq)

    f_min = np.min(f_abs)
    f_max_data = np.max(f_abs)

    mi = 0.5 / (f_max_data - f_min + 1e-12)

    partition = np.linspace(0, 0.5, B + 1)
    fp = partition[:-1] / mi

    HS = np.zeros((B, N - 1))

    # Quantization
    indices = np.floor(mi * f_abs[:-1] * B).astype(int)

    indices[indices >= B] = B - 1
    indices[indices < 0] = 0

    for n in range(N - 1):
        for k in range(num_imfs):
            bin_idx = indices[n, k]
            HS[bin_idx, n] += amplitude[n, k]

    # =============================
    # 4) Normalize and convert to dB
    # =============================
    HS = HS / (np.max(HS) + 1e-12)
    HS_dB = 20 * np.log10(HS + 1e-12)
    HS_dB[HS_dB < -80] = -80

    if fmax is None:
        fmax = np.max(fp)

    # =============================
    # 5) Marginal Hilbert Spectrum
    # =============================
    dt = 1 / Fs
    MHS = np.sum(HS, axis=1) * dt
    MHS = MHS / (np.max(MHS) + 1e-12)

    # =============================
    # 6) Plot
    # =============================
    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

        im = ax1.imshow(
            HS_dB.T,
            aspect='auto',
            origin='lower',
            extent=[0, fmax, 0, tmax],
            vmin=-60,
            vmax=0
        )
        ax1.set_ylabel("Time (s)")
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_title("Hilbert Spectrum (HS)")
        fig.colorbar(im, ax=ax1)

        ax2.plot(fp, 20 * np.log10(MHS + 1e-12), linewidth=0.8)
        ax2.set_xlim([0, fmax])
        ax2.set_ylim([-80, 0])
        ax2.grid(True)
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Normalized amplitude (dB)")
        ax2.set_title("Marginal Hilbert Spectrum (MHS)")

        plt.tight_layout()
        plt.show()

    return MHS, fp
