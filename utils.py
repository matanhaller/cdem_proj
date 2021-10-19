"""Utility functions for infrastructure.
"""

import numpy as np
from scipy.fft import fft, fftshift, ifft, ifftshift


def ctft(f, x_0, delta_x, shift=True):
    """Approximate implementation of the "physical" Fourier transform
    (integral).

    Args:
        f (np.array): Array to be transformed.
        x_0 (float): Leftmost sampling point.
        delta_x (float): Sampling interval.
        shift (bool): Whether to shift zero-frequency component to center.

    Returns:
        np.array: Array of length N of the "physical" Fourier transform of f.
    """
    N = len(f)
    n = np.arange(-np.floor(N / 2), np.ceil(N / 2))
    k_0 = 2 * np.pi / (N * delta_x)
    phase_fix = np.exp(-1j * k_0 * x_0 * n)

    F = fft(f)
    if shift:
        F = fftshift(F)

    return (1 / (2 * np.pi)) * phase_fix * F * delta_x


def ctift(f, x_0, delta_x, shift=True):
    """Approximate implementation of the "physical" inverse Fourier transform
    (integral).

    Args:
        f (np.array): Array to be transformed.
        x_0 (float): Leftmost sampling point.
        delta_x (float): Sampling interval.
        shift (bool): Whether to shift zero-frequency component to center.

    Returns:
        np.array: Array of length N of the "physical" inv. Fourier transform
        of f.
    """
    N = len(f)
    n = np.arange(-np.floor(N / 2), np.ceil(N / 2))
    k_0 = 2 * np.pi / (N * delta_x)
    phase_fix = np.exp(1j * k_0 * x_0 * n)

    F = ifft(f)
    if shift:
        F = ifftshift(F)

    return N * phase_fix * F * delta_x
