"""Utility functions for infrastructure.
"""

import numpy as np
from scipy.fft import fft, fftshift, ifft, ifftshift


def T(v):
    """Implement transpose for row vector by reshaping to len(v) x 1.

    Args:
        v (np.array): Row vector to be transposed.

    Returns:
        np.array: len(v) x 1 column vector of v^T.
    """
    return np.reshape(v, (len(v), 1))


def ctft(f, x_0, delta_x):
    """Approximate implementation of the "physical" Fourier transform
    (integral from -Inf to Inf).

    Args:
        f (np.array): Array to be transformed.
        x_0 (float): Leftmost sampling point.
        delta_x (float): Sampling interval.

    Returns:
        np.array: Array of length N of the "physical" Fourier transform of f.
    """

    N = len(f)
    n = np.arange(-np.floor(N / 2), np.ceil(N / 2))
    k_0 = 2 * np.pi / (N * delta_x)
    phase_fix = np.exp(-1j * k_0 * x_0 * n)

    return phase_fix * fftshift(fft(f)) * delta_x


def ctift(f, x_0, delta_x):
    """Approximate implementation of the "physical" inverse Fourier transform
    (integral from -Inf to Inf).

    Args:
        f (np.array): Array to be transformed.
        x_0 (float): Leftmost sampling point.
        delta_x (float): Sampling interval.

    Returns:
        np.array: Array of length N of the "physical" inv. Fourier transform
        of f.
    """

    N = len(f)
    n = np.arange(-np.floor(N / 2), np.ceil(N / 2))
    k_0 = 2 * np.pi / (N * delta_x)
    phase_fix = np.exp(1j * k_0 * x_0 * n)

    return N * phase_fix * ifftshift(ifft(f)) * delta_x
