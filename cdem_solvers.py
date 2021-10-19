"""Numerical solution of CDEM for various potentials.
"""

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import fftconvolve
from consts import *
from utils import ctft, ctift
import matplotlib.pyplot as plt


class CDEMSolver(object):

    """Abstract class for solving CDEM in various configurations.
    """

    def __init__(self, v, energy_fwhm, F_s, z_max, d_z, L, delta_t_lst):
        """Contructor method.

        Args:
            v (float): Electron's velocity (in m*s^-1).
            energy_fwhm (float): FWHM of the electron pulse (in eV).
            F_s (float): Sampling frequency to be used in simulation (in Hz).
            z_max (float): Maximal z value to consider in simulation (in m).
            d_z (float): Sampling interval for z (in m).
            L (int): Number of time/frequency variables to consider.
            delta_t_lst (np.array): List of time delays between electron probe.
            and excitations.
        """
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
        """Fourier representation of generalized CDEM potential (see theory).
        Should be overriden by subclasses!

        Args:
            z (float): Distance from sample in the z-direction.
            omega (float): Frequency.

        Returns:
            float: Pontential value V(z,omega).
        """
        return 1

    def _beta(self):
        """Calculating beta(omega) from the CDEM formailsm (see theory).

        Returns:
            float: Value of beta(omega).
        """
        W, Z = np.meshgrid(self._omega_lst, self._z_lst)
        V_mat = self._V(Z, W)
        C = 1 / (hbar * self._v)
        return C * trapezoid(V_mat * np.exp(1j * np.outer(self._z_lst,
                                                          self._omega_lst)
                                            / self._v), dx=self._d_z, axis=0)

    def _f(self, beta):
        """Calculating f(omega) from the CDEM formailsm (see theory).

        Args:
            beta (float): value of beta calculated beforehand.

        Returns:
            float: Value of f(omega).
        """
        f_inner_integral = ctift(
            beta, x_0=self._omega_lst[0], delta_x=self._d_omega, shift=True)
        f_integrand = np.exp(-1j * f_inner_integral)

        C = 1 / (2 * np.pi)
        return C * ctift(f_integrand, x_0=self._t_lst[0],
                         delta_x=self._d_t, shift=True)

    def _phi_0(self):
        """Gaussian of the electron's temporal pulse.

        Returns:
            float: Gaussian with temporal width sigma_t.
        """
        return 1 / np.sqrt(2 * np.pi * self._sigma_t ** 2) \
            * np.exp(-0.5 * self._sigma_t ** 2 * self._omega_lst ** 2)

    def _psi_coherent(self, f):
        """Calculating the coherent electron's wavefunction in the energy basis
        after interaction with the potential, for the different time delays.

        Args:
            f (float): Value of f(omega) calculated beforehand.

        Returns:
            np.ndarray: Coherent electron wavefunction.
        """
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
        """Solving the CDEM formalism for current configuration

        Args:
            debug (bool, optional): Debug flag (plots additional graphs).

        Returns:
            np.ndarray: Coherent electron wavefunction.
        """
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

    """Class for solving CDEM for dipole potential with an arbitrary temporal
    variation.
    """

    def __init__(self, v, energy_fwhm, F_s, z_max, d_z, L, delta_t_lst,
                 orientation, d, d_min, N):
        """Constructor method.

        Args:
            orientation (str): Dipole moment orientation (xy/z).
            d (float): Separation distance between the charges (in m).
            d_min (float): Minimal distance of electron probe from dipole.
            N (int): Number of dipoles.
        """
        super().__init__(v, energy_fwhm, F_s, z_max, d_z, L, delta_t_lst)
        self._orientation = orientation
        self._d = d
        self._d_min = d_min
        self._N = N

    def _X(self, z, omega):
        """Spatial component of the dipole potential (decays like r^-2).

        Args:
            z (float): z-coordinate.
            omega (float): Frequency.

        Returns:
            float: Spatial dipole potential.
        """
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
        """Fourier represntation of the temporal component of the potential.
        Should be overriden by subclasses!

        Returns:
            float: Calculated Fourier transform of temporal component.
        """
        return 1

    def _V(self, z, omega):
        """Calculating the dipole's CDEM potential, which is seperable and can
        be written as V(z,omega)=X(z,omega)*H(omega).

        Args:
            z (float): z-coordinate.
            omega (float): Frequency.

        Returns:
            float: Calculated potential.
        """
        return self._N * self._X(z, omega) * self._H()


class CDEMDipoleHarmonic(CDEMDipole):

    """Class for solving CDEM for harmonic-varying dipole potential.
    """

    def __init__(self, v, energy_fwhm, F_s, z_max, d_z, L, delta_t_lst,
                 orientation, d, d_min, N, f_d):
        """Constructor method.

        Args:
            f_d (float): Dipole's oscillation frequency (in Hz).
        """
        super().__init__(v, energy_fwhm, F_s, z_max, d_z, L, delta_t_lst,
                         orientation, d, d_min, N)
        self._f_d = f_d

    def _H(self):
        """Fourier transform of the dipole's periodic oscillation.

        Returns:
            float: Resultant H(omega).
        """
        h = np.cos(2 * np.pi * self._f_d * self._t_lst)
        H = ctft(h, x_0=self._t_lst[0], delta_x=self._d_t, shift=True)

        return H


class CDEMDipoleHarmonicGaussian(CDEMDipoleHarmonic):

    """Class for solving CDEM for harmonic-varying dipole potential w. Gaussian
    envelope.
    """

    def __init__(self, v, energy_fwhm, F_s, z_max, d_z, L, delta_t_lst,
                 orientation, d, d_min, N, f_d, sigma_d):
        """Constructor method.

        Args:
            sigma_d (float): Gaussian envelope temporal width (in s).
        """
        super().__init__(v, energy_fwhm, F_s, z_max, d_z, L, delta_t_lst,
                         orientation, d, d_min, N, f_d)
        self._sigma_d = sigma_d

    def _H(self):
        """Fourier transform of the dipole's periodic oscillation, multiplied
        by Gaussian envelope.

        Returns:
            float: Resultant H(omega).
        """
        h = np.cos(2 * np.pi * self._f_d * self._t_lst) * \
            np.exp(-0.5 * self._t_lst ** 2 / self._sigma_d ** 2)
        H = ctft(h, x_0=self._t_lst[0], delta_x=self._d_t, shift=True)

        return H


class CDEMDipoleTransient(CDEMDipole):

    """Class for solving CDEM for transient dipole potential, reflecting the
    behavior in several experiments (see Michael's research proposal).
    """

    def __init__(self, v, energy_fwhm, F_s, z_max, d_z, L, delta_t_lst,
                 orientation, d, d_min, N, tau_r, tau_d):
        """Constructor method.

        Args:
            tau_r (float): Rise time of the potential (in s).
            tau_d (float): Decay time of the potential (in s).
        """
        super().__init__(v, energy_fwhm, F_s, z_max, d_z, L, delta_t_lst,
                         orientation, d, d_min, N)
        self._tau_r = tau_r
        self._tau_d = tau_d

    def _H(self):
        """Fourier transform of the dipole potential's transient temporal
        variation.

        Returns:
            float: Resultant H(omega).
        """
        h = np.zeros((int(self._L),))
        h[self._t_lst > 0] = (np.exp(-self._t_lst[self._t_lst > 0]
                                     / self._tau_d) - np.exp(
            -self._t_lst[self._t_lst > 0] / self._tau_r))
        h[self._t_lst < 0] = 0
        H = ctft(h, x_0=self._t_lst[0], delta_x=self._d_t, shift=True)

        return H
