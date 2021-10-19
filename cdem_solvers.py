"""Numerical solution of CDEM for various potentials.
"""

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import fftconvolve
from consts import *
from utils import ctft, ctift
import matplotlib.pyplot as plt


class CDEMSolver(object):

    def __init__(self, v, energy_fwhm, F_s, z_max, d_z, L, delta_t_lst):
        self._v = v
        self._sigma_t = hbar / \
            (energy_fwhm / (2 * np.sqrt(2 * np.log(2))) * e)
        self._F_s = F_s
        self._z_max = z_max
        self._d_z = d_z
        self._L = L

        self._d_t = 1 / F_s
        self._d_omega = 2 * np.pi * F_s / L

        self._omega_lst = np.arange(-0.5 * self._L, 0.5 * self._L) \
            * self._d_omega
        self._t_lst = np.arange(-0.5 * self._L, 0.5 * self._L) * self._d_t
        self._z_lst = np.arange(-0.5 * self._z_max,
                                0.5 * self._z_max, self._d_z)

        self._delta_t_lst = delta_t_lst

    def _V(self, z, omega):
        return 1

    def _beta(self):
        W, Z = np.meshgrid(self._omega_lst, self._z_lst)
        V_mat = self._V(Z, W)
        C = 1 / (hbar * self._v)
        return C * trapezoid(V_mat * np.exp(1j * np.outer(self._z_lst,
                                                          self._omega_lst)
                                            / self._v), dx=self._d_z, axis=0)

    def _f(self, beta):
        f_inner_integral = ctift(
            beta, x_0=self._omega_lst[0], delta_x=self._d_omega, shift=True)
        f_integrand = np.exp(-1j * f_inner_integral)

        C = 1 / (2 * np.pi)
        return C * ctift(f_integrand, x_0=self._t_lst[0],
                         delta_x=self._d_t, shift=True)

    def _phi_0(self):
        return 1 / np.sqrt(2 * np.pi * self._sigma_t ** 2) \
            * np.exp(-0.5 * self._sigma_t ** 2 * self._omega_lst ** 2)

    def _psi_coherent(self, f):
        phi_0 = self._phi_0()
        exp_d_t = np.exp(-1j * np.outer(self._delta_t_lst, self._omega_lst))

        f_rep = np.tile(f, (len(self._delta_t_lst), 1))

        psi_coherent = fftconvolve(phi_0 * exp_d_t, f_rep, mode='same', axes=1)

        # Normalizing w.r.t energy (E [eV] = hbar * omega / e)
        psi_coherent_norm = np.sqrt(trapezoid(np.abs(psi_coherent) ** 2,
                                              dx=self._d_omega, axis=1)
                                    * e / hbar)

        return psi_coherent / np.reshape(psi_coherent_norm,
                                         (len(psi_coherent_norm), 1))

    def solve_coherent(self, debug=False):

        beta = self._beta()
        if debug:
            plt.plot(self._omega_lst / (2 * np.pi), np.abs(beta))
            plt.xlabel('$f$ [Hz]', fontsize=14)
            plt.ylabel(r'$\beta_{\omega}$', fontsize=14)
            plt.show()
        f = self._f(beta)
        if debug:
            plt.plot(self._omega_lst / (2 * np.pi), np.abs(f))
            plt.xlabel('$f$ [Hz]', fontsize=14)
            plt.ylabel(r'$f_{\omega}$', fontsize=14)
            plt.show()

        return self._psi_coherent(f)


class CDEMDipole(CDEMSolver):

    def __init__(self, v, energy_fwhm, F_s, z_max, d_z, L, delta_t_lst,
                 orientation, d, d_min, N):
        super().__init__(v, energy_fwhm, F_s, z_max, d_z, L, delta_t_lst)
        self._orientation = orientation
        self._d = d
        self._d_min = d_min
        self._N = N

    def _X(self, z, omega):
        if self._orientation == 'z':
            R_minus = np.sqrt(self._d_min ** 2 + (z - self._d / 2) ** 2)
            R_plus = np.sqrt(self._d_min ** 2 + (z + self._d / 2) ** 2)

        elif self._orientation == 'xy':
            R_minus = np.sqrt((self._d_min - self._d / 2) ** 2 + z ** 2)
            R_plus = np.sqrt((self._d_min + self._d / 2) ** 2 + z ** 2)

        k = omega / c

        return -e ** 2 / (4 * np.pi * epsilon_0) * \
            (np.exp(1j * k * R_minus)
                / R_minus - np.exp(1j * k * R_plus) / R_plus)

    def _H(self):
        return 1

    def _V(self, z, omega):
        return self._N * self._X(z, omega) * self._H()


class CDEMDipoleHarmonic(CDEMDipole):

    def __init__(self, v, energy_fwhm, F_s, z_max, d_z, L, delta_t_lst,
                 orientation, d, d_min, N, f_d):
        super().__init__(v, energy_fwhm, F_s, z_max, d_z, L, delta_t_lst,
                         orientation, d, d_min, N)
        self._f_d = f_d

    def _H(self):
        h = np.cos(2 * np.pi * self._f_d * self._t_lst)
        H = ctft(h, x_0=self._t_lst[0], delta_x=self._d_t, shift=True)

        return H


class CDEMDipoleHarmonicGaussian(CDEMDipoleHarmonic):

    def __init__(self, v, energy_fwhm, F_s, z_max, d_z, L, delta_t_lst,
                 orientation, d, d_min, N, f_d, sigma_d):
        super().__init__(v, energy_fwhm, F_s, z_max, d_z, L, delta_t_lst,
                         orientation, d, d_min, N, f_d)
        self._sigma_d = sigma_d

    def _H(self):
        h = np.cos(2 * np.pi * self._f_d * self._t_lst) * \
            np.exp(-0.5 * self._t_lst ** 2 / self._sigma_d ** 2)
        H = ctft(h, x_0=self._t_lst[0], delta_x=self._d_t, shift=True)

        return H


class CDEMDipoleTransient(CDEMDipole):

    def __init__(self, v, energy_fwhm, F_s, z_max, d_z, L, delta_t_lst,
                 orientation, d, d_min, N, tau_r, tau_d):
        super().__init__(v, energy_fwhm, F_s, z_max, d_z, L, delta_t_lst,
                         orientation, d, d_min, N)
        self._tau_r = tau_r
        self._tau_d = tau_d

    def _H(self):
        h = np.zeros((int(self._L),))
        h[self._t_lst > 0] = (np.exp(-self._t_lst[self._t_lst > 0]
                                     / self._tau_d) - np.exp(
            -self._t_lst[self._t_lst > 0] / self._tau_r))
        h[self._t_lst < 0] = 0
        H = ctft(h, x_0=self._t_lst[0], delta_x=self._d_t, shift=True)

        return H
