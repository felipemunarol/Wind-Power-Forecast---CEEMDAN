import numpy as np


def VMD(f, alpha, tau, K, DC=0, init=1, tol=1e-7, N_iter_max=500):
    """
    Variational Mode Decomposition (VMD)
    Dragomiretskiy & Zosso (2014)

    Parameters
    ----------
    f : 1D array
        Input signal
    alpha : float or array (K,)
        Bandwidth constraint
    tau : float
        Time-step of dual ascent (0 for noise-slack)
    K : int
        Number of modes
    DC : int
        1 = include DC mode
    init : int
        0 = zero, 1 = uniform, 2 = random
    tol : float
        Convergence tolerance
    N_iter_max : int
        Maximum iterations

    Returns
    -------
    u : ndarray (K, N)
        Modes in time domain
    u_hat : ndarray (K, T)
        Spectral modes
    omega : ndarray (K,)
        Center frequencies
    """

    f = np.asarray(f, dtype=float)
    f = f.flatten()
    N = len(f)

    # -----------------------------
    # Mirror signal
    # -----------------------------
    f_mirror = np.concatenate([f[::-1], f, f[::-1]])
    T = len(f_mirror)

    # -----------------------------
    # Frequency domain
    # -----------------------------
    freqs = np.arange(0, T) / T - 0.5
    freqs = np.fft.fftshift(freqs)

    f_hat = np.fft.fftshift(np.fft.fft(f_mirror))
    f_hat_plus = np.copy(f_hat)
    f_hat_plus[freqs < 0] = 0

    # -----------------------------
    # Initialization
    # -----------------------------
    u_hat_plus = np.zeros((K, T), dtype=complex)
    omega = np.zeros(K)

    if init == 1:
        omega = (0.5 / K) * np.arange(K)
    elif init == 2:
        omega = np.sort(np.random.rand(K) / 2)

    if DC:
        omega[0] = 0

    lambda_hat = np.zeros(T, dtype=complex)

    if np.isscalar(alpha):
        alpha = alpha * np.ones(K)

    uDiff = tol + np.finfo(float).eps
    n = 0

    # ===============================
    # Main loop
    # ===============================
    while (uDiff > tol) and (n < N_iter_max):

        u_hat_plus_prev = np.copy(u_hat_plus)

        for k in range(K):

            sum_uk = np.sum(u_hat_plus, axis=0) - u_hat_plus[k, :]

            u_hat_plus[k, :] = (
                f_hat_plus - sum_uk - lambda_hat / 2
            ) / (1 + alpha[k] * (freqs - omega[k]) ** 2)

            # Update center frequency
            if not (DC and k == 0):
                numerator = np.sum(freqs * np.abs(u_hat_plus[k, :]) ** 2)
                denominator = np.sum(np.abs(u_hat_plus[k, :]) ** 2) + 1e-12
                omega[k] = numerator / denominator

        # Dual ascent
        lambda_hat = lambda_hat + tau * (
            np.sum(u_hat_plus, axis=0) - f_hat_plus
        )

        # Convergence check
        uDiff = 0
        for k in range(K):
            uDiff += (1 / T) * np.sum(
                np.abs(u_hat_plus[k, :] - u_hat_plus_prev[k, :]) ** 2
            )

        n += 1

    # -----------------------------
    # Reconstruction
    # -----------------------------
    u_hat = u_hat_plus

    u_temp = np.real(
        np.fft.ifft(np.fft.ifftshift(u_hat, axes=1), axis=1)
    )

    # Remove mirror
    u = u_temp[:, N:2 * N]

    return u, u_hat, omega
