"""Utility functions for infrastructure.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift, ifft, ifftshift
from scipy.integrate import trapezoid
from consts import *


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


def plot_EELS(ax, psi, omega_lst, delta_t_lst, title, color_min, color_max):
    """Plot EELS spectrum corresponding to wavefunction calculated via CDEM.
    The plot is a logarithmic-scaled heatmap of the wavefunction's squared
    modulus (corresponding to probability amplitude) of different energies and
    pump-probe delays.

    Args:
        ax (matplotlib.Axes): Axes object for plot.
        psi (np.ndarray): 2D array of coherent wavefunction values for
        different energies and pump-probe delays.
        omega_lst (np.array): List of frequencies corresponding to
        wavefunction.
        delta_t_lst (np.array): List of time delays.
        title (str): Plot title.
        color_min (int): Minimal value for colorbar.
        color_max (int): Maximal value for colorbar.
    """
    E_lst = omega_lst * hbar / e
    psi_plot = ax.pcolormesh(E_lst, delta_t_lst * 1e12,
                             np.log10(np.abs(psi[:-1, :-1]) ** 2),
                             cmap='hot', vmin=color_min, vmax=color_max)
    ax.set_xlabel('Energy Shift [eV]', fontsize=10)
    ax.set_ylabel('Time Delay [ps]', fontsize=10)
    ax.set_title(title, fontsize=14)
    plt.colorbar(psi_plot, ax=ax)


def plot_mean_energy_shift(ax, psi, omega_lst, delta_t_lst, title):
    """Plot mean energy shift of electron probe as a function of the pumb-probe
    delay.

    Args:
        ax (matplotlib.Axes): Axes object for plot.
        psi (np.ndarray): 2D array of coherent wavefunction values for
        different energies and pump-probe delays.
        omega_lst (np.array): List of frequencies corresponding to
        wavefunction.
        delta_t_lst (np.array): List of time delays.
        title (str): Plot title.
    """
    E_lst = omega_lst * hbar / e
    E_mean = trapezoid(E_lst * np.abs(psi) ** 2, E_lst, axis=1)

    ax.plot(delta_t_lst * 1e12, E_mean)
    ax.set_xlabel('Time Delay [ps]', fontsize=10)
    ax.set_ylabel('Mean Energy Shift [eV]', fontsize=10)
    ax.set_title(title, fontsize=14)
