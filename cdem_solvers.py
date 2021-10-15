"""Numerical solution of CDEM for various potentials.
"""

import numpy as np
from scipy.integrate import trapz
from np.image import convolve1d
from consts import *


class CDEMSolver(object):

    def __init__(self, v, energy_fwhm):
        self._v = v
        self._sigma_t = hbar / \
            (energy_fwhm / (2 * np.sqrt(2 * np.log2(2))) * e)

        self._z_min = -1
        self._z_max = 1
        self._k_z_sample = 1

        self._t_min = -1
        self._t_max = 1
        self._f_sample = 1

        self._Omega_max = 1
        self._tau_sample = 1

    def __V(self, z, omega):
        return -1

    def __beta(self, z_lst, Omega_lst):
        WO, Z = np.meshgrid(Omega_lst, z_lst)
        V_mat = self.__V(Z, WO)
        C = 1 / (hbar * v)
        return C * trapz(V_mat * np.exp(1j * T(Omega_lst) * z_lst / self._v),
                         z_lst, axis=0)

    def __f(self, t_lst, omega_lst, Omega_lst, beta):
        f_inner_integrand = np.real(np.exp(1j * T(Omega_lst) * t_lst) * beta)
        f_inner_integral = trapz(f_inner_integrand, Omega_lst, axis=0)
        f_integrand = np.exp(1j * (T(omega_lst) * t_lst - T(f_inner_integral)))
        C = 1 / (2 * np.pi)
        return C * trapz(f_integrand, t_lst, axis=0)

    def __phi_0(self, omega):
        return 1 / np.sqrt(2 * np.pi * self._sigma_t ** 2)
        * np.exp(-0.5 * self._sigma_t ** 2 * omega ** 2)

    def __psi_coherent(self, omega_lst, delta_t_lst, phi_0, f):
        C = hbar * self._v
        return C * np.abs(convolve1d(phi0 * exp(-1j * T(delta_t_lst)
                                                * omega_lst), f,
                                     mode='constant', axis=1)) ** 2

    def solve_coherent(self, omega_lst, delta_t_lst):
        delta_z = 1 / self._k_z_sample
        z_lst = np.arange(self._z_min, self._z_max, delta_z)

        delta_Omega = 1 / self._tau_sample
        Omega_lst = np.arange(0, self._Omega_max, delta_Omega)

        delta_t = 1 / self._f_sample
        t_lst = np.arange(self._t_min, self._t_max, delta_t)

        beta = self.__beta(z_lst, Omega_lst)
        f = self.__f(t_lst, omega_lst, Omega_lst, beta)
        phi_0 = self.__phi_0(omega_lst)

        return self.__psi_coherent(omega_lst, delta_t_lst, phi_0, f)
