#
# TODO: cite the papers that form the basis of this code
#

from typing import Tuple, Optional, Dict, Callable

import logging
import numpy as np
import spectral_data
from deepdrr import geo
from deepdrr import vol
from rayleigh_form_factor_data import build_form_factor_func
import compton_data
from rita import RITA

import math


logger = logging.getLogger(__name__)


#
# USEFUL CONSTANTS
#

LIGHT_C = 2.99792458e08 # m/s
ELECTRON_MASS = 9.1093826e-31 # kg
ELECTRON_REST_ENERGY = 510998.918 # eV


def simulate_scatter_no_vr(
    volume: vol.Volume,
    source: geo.Point3D,
    output_shape: Tuple[int, int],
    spectrum: Optional[np.ndarray] = spectral_data.spectrums['90KV_AL40'], 
    photon_count: Optional[int] = 10000000, # 10^7
    E_abs: Optional[np.float32] = 1000 # TODO: check that the spectrum data uses keV as its units -- ask Benjamin about the spectral data source
) -> np.ndarray:
    """Produce a grayscale (intensity-based) image representing the photon scatter during an X-Ray, 
    without using VR (variance reduction) techniques.

    Args:
        volume (np.ndarray): the volume density data.
        source (Point3D): the source point for rays in the camera's IJK space
        spectrum (Optional[np.ndarray], optional): spectrum array.  Defaults to 90KV_AL40 spectrum.
        photon_count (Optional[int], optional): the number of photons simulated.  Defaults to 10^7 photons.
        E_abs (Optional[np.float32], optional): the energy (in keV) at or below which photons are assumed to be absorbed by the materials.  Defaults to 1000 (keV).
        output_shape (Tuple[int, int]): the {height}x{width} dimensions of the output image

    Returns:
        np.ndarray: intensity image of the photon scatter
    """
    count_milestones = [int(math.pow(10, i)) for i in range(int(1 + math.ceil(math.log10(photon_count))))] # [1, 10, 100, ..., 10^7] in default case

    accumulator = np.zeros(output_shape).astype(np.float32)

    N_vals = None
    sigma_C_vals = None
    sigma_R_vals = None

    rayleigh_samplers = {}
    for mat in volume.segmentation:
        pdf_func = build_form_factor_func(mat)
        maximum_spectrum_energy = spectrum[0,-1] # TODO: convert to eV
        maximum_kappa = maximum_spectrum_energy / ELECTRON_REST_ENERGY
        x_max = 20.6074 * 2 * maximum_kappa
        x_max2 = x_max * x_max
        rayleigh_samplers[mat] = RITA.from_pdf(0, x_max2, pdf_func) # uses default of 128 grid-points

    for i in range(photon_count):
        if (i+1) in count_milestones:
            print(f"Simulating photon history {i+1} / {photon_count}")
        BLAH = 0
        initial_dir = BLAH # Random sampling is a TODO
        initial_E = sample_initial_energy(spectrum)
        single_scatter = track_single_photon_no_vr(source, initial_dir, initial_E, E_abs, N_vals, sigma_C_vals, sigma_R_vals)
        accumulator = accumulator + single_scatter
    
    print(f"Finished simulating {photon_count} photon histories")
    return NotImplemented

def track_single_photon_no_vr(
    initial_pos: geo.Point3D,
    initial_dir: geo.Vector3D,
    initial_E: np.float32,
    E_abs: np.float32,
    N_vals: np.ndarray,
    sigma_C_vals: np.ndarray,
    sigma_R_vals: np.ndarray
) -> np.ndarray:
    """Produce a grayscale (intensity-based) image representing the photon scatter of a single photon 
    during an X-Ray, without using VR (variance reduction) techniques.

    Args:
        initial_pos (geo.Point3D): the initial position of the photon, which is the X-Ray source
        initial_dir (geo.Vector3D): the initial direction of travel of the photon
        initital_E (np.float32): the initial energy of the photon
        E_abs (np.float32): the energy (in keV) at or below which photons are assumed to be absorbed by the materials.  Defaults to 1000 (keV).
        N_vals (np.ndarray): a field over the volume of 'N', the number of molecules per unit volume.  Each cell in N_vals contains the information for the corresponding voxel.
        sigma_C_vals (np.ndarray): a field over the volume of 'sigma_C', the interactional cross-section for Compton scatter.  Each cell in sigma_C_vals contains the information for the corresponding voxel.
        sigma_R_vals (np.ndarray): a field over the volume of 'sigma_R', the interactional cross-section for Rayleigh scatter.  Each cell in sigma_R_vals contains the information for the corresponding voxel.

    Returns:
        np.ndarray: intensity image of the photon scatter
    """
    BLAH = 0
    # find the point on the boundary of the volume that the photon from the source first reaches
    pos = BLAH
    dir = initial_dir

    photon_energy = initial_E # tracker variable throughout the duration of the photon history

    while True: # emulate a do-while loop
        # simulate moving the photon
        sigma_T = BLAH # function of photon_energy
        lambda_T = BLAH # function of N_vals[pos] and sigma_T[pos]
        s = -1 * lambda_T * np.log(sample_U01())
        
        pos = pos + (s * dir)

        # Handle crossings of segementation boundary
        BLAH = 0

        will_exit_volume = BLAH
        if will_exit_volume:
            break

        # simulate the photon interaction
        prob_of_Compton = BLAH 
        theta, W = None, None
        if sample_U01() < prob_of_Compton:
            theta, W = sample_Compton_theta_E_prime(None, None)
        else:
            theta, W = sample_Rayleigh_theta(None, None)
        
        photon_energy = photon_energy - W
        if photon_energy <= E_abs:
            break
        
        phi = 2 * np.pi * sample_U01()
        dir = get_scattered_dir(dir, theta, phi)

        # END WHILE
    
    # final processing
    BLAH = BLAH

    return NotImplemented

def get_scattered_dir(
    dir: geo.Vector3D,
    theta: np.float32,
    phi: np.float32
) -> geo.Vector3D:
    """Determine the new direction of travel after getting scattered

    Args:
        dir (geo.Vector3D): the incoming direction of travel
        theta (np.float32): the polar scattering angle, i.e. the angle dir and dir_prime
        phi (np.float32): the azimuthal angle, i.e. how dir_prime is rotated about the axis 'dir'.

    Returns:
        geo.Vector3D: the outgoing direction of travel
    """
    return NotImplemented

def sample_U01() -> np.float32:
    """Returns a value uniformly sampled from the interval [0,1]"""
    return NotImplemented

def sample_initial_energy(spectrum: np.ndarray) -> np.float32:
    """Determine the energy (in keV) of a photon emitted by an X-Ray source with the given spectrum

    Args:
        spectrum (np.ndarray): the data associated with the spectrum.  Cross-reference spectral_data.py
    
    Returns:
        np.float32: the energy of a photon, in keV
    """
    return NotImplemented

def sample_Rayleigh_theta(
    mat: str,
    photon_energy: np.float32,
    rayleigh_samplers: Dict[str, RITA]
) -> np.float32:
    """Randomly sample values of theta and W for a given Rayleigh scatter interaction
    Based on page 49 of paper 'PENELOPE-2006: A Code System for Monte Carlo Simulation of Electron and Photon Transport'

    Args:
        mat (str): a string specifying the material at that position in the volume
        photon_energy (np.float32): the energy of the incoming photon
        rayleigh_samplers (Dict[str, RITA]): a dictionary associating materials to their respective RITA sampler objects 

    Returns:
        np.float32: cos(theta), where theta is the polar scattering angle 
    """
    kappa = photon_energy / ELECTRON_REST_ENERGY
    # Sample a random value of x^2 from the distribution pi(x^2), restricted to the interval (0, x_{max}^2)
    x_max = 20.6074 * 2 * kappa
    x_max2 = x_max * x_max
    sampler = rayleigh_samplers[mat]
    x2 = sampler.sample_rita()
    while (x2 > x_max2):
        # Resample until x^2 is in the interval (0, x_{max}^2)
        x2 = sampler.sample_rita()

    while True:
        # Set cos_theta
        cos_theta = 1 - (2 * x2 / x_max2)

        # Test cost_theta
        g = (1 + cos_theta * cos_theta) / 2

        if sample_U01() <= g:
            break

    return cos_theta

def sample_Compton_theta_E_prime(
    photon_energy: np.float32,
) -> np.float32:
    """Randomly sample values of theta and W for a given Compton scatter interaction

    Args:
        N_val: the number of molecules per unit volume at the location of the photon interaction
        sigma_val: the interactional cross-sectional area at the location of the photon interation

    Returns:
        np.float32: cos_theta, the polar scattering angle 
        np.float32: E_prime, the energy of the outgoing photon
    """
    kappa = photon_energy / ELECTRON_REST_ENERGY

    a_1 = np.log1p(2 * kappa)
    one_p2k = 1 + 2 * kappa
    a_2 = 2 * kappa * (1 + kappa) / (one_p2k * one_p2k)

    tau_min = 1 / one_p2k

    #
    # ---- Function definitions ----
    #

    def f_i(i):
        return compton_data.electron_counts[i]

    def U_i(i):
        return compton_data.ionization_energies[i]

    def p_i_max(i, E, cos_theta):
        """Computes p_{i,max}(E,\\theta)
        
        Args:
            i: the number of the electron shell, as per Table 6.2 in 'PENELOPE-2006' specifications
            E: the energy of the incoming photon
            cos_theta: the cosine of the scatter angle, theta
        
        Returns:
            the result of applying Eqn 2.45
        """
        U_i = compton_data.ionization_energies[i]
        tmp = E * (E - U_i) * (1 - cos_theta)
        numerator = tmp - ELECTRON_REST_ENERGY * U_i
        denom_term = 2 * tmp + U_i * U_i
        return numerator / (LIGHT_C * np.sqrt(denom_term))
    
    def n_i_A(i, p_z):
        """Implements Eqn 2.54 from 'PENELOPE-2006' specification"""
        d_1 = np.sqrt(0.5)
        d_2 = np.sqrt(2)
        J_i = compton_data.compton_profile[i]
        if p_z < 0:
            tmp = d_1 - (d_2 * J_i * p_z)
            return 0.5 * np.exp((d_1 * d_1) - (tmp * tmp))
        else:
            tmp = d_1 + (d_2 * J_i * p_z)
            return 1 - 0.5 * np.exp((d_1 * d_1) - (tmp * tmp))

    def heaviside(x):
        return 1 if x > 0 else 0
    
    def set_shell_scatter_vals(S_E_theta_vals, E, cos_theta, p_i_max_vals):
        """computes: f_{i} \\Theta(E - U_{i}) n_{i}(p_{i,max}) and stores the result in S_E_theta_vals"""
        assert len(S_E_theta_vals) == compton_data.NUM_SHELLS
        for i in range(compton_data.NUM_SHELLS):
            S_E_theta_vals[i] = f_i(i) * heaviside(E - U_i(i)) * n_i_A(i, p_i_max_vals[i])

    def S(E, cos_theta):
        return sum(
            [
                f_i(i) * heaviside(E - U_i(i)) * n_i_A(i, p_i_max(i, E, cos_theta))
                for i in range(compton_data.NUM_SHELLS)
            ]
        )
    
    def T(tau, cos_theta, p_i_max_vals, S_E_theta_vals):
        lt_numer = (1 - tau) * ((2 * kappa + 1) * tau - 1)
        lt_denom = kappa * kappa * tau * (1 + tau * tau)
        return (1 - lt_numer / lt_denom) * S_E_theta_vals[i] / S(photon_energy, np.pi)

    # Sample cos_theta
    # want local storage, since p_{i,max} values are reused in the E_prime sampling
    p_i_max_vals = [0 for i in range(compton_data.NUM_SHELLS)]
    S_E_theta_vals = [0 for i in range(compton_data.NUM_SHELLS)]

    while True:
        i = 1 if sample_U01() < (a_1 / (a_1 + a_2)) else 2
        tau = None
        if 1 == i:
            tau = np.power(tau_min, sample_U01())
        else:
            assert 2 == i
            rnd = sample_U01()
            tau = np.sqrt(rnd + tau_min * tau_min * (1 - rnd))
        
        cos_theta = 1 - (1 - tau) / (kappa * tau)

        for i in range(compton_data.NUM_SHELLS):
            p_i_max_vals[i] = p_i_max(i, photon_energy, cos_theta)
            set_shell_scatter_vals(S_E_theta_vals, photon_energy, cos_theta, p_i_max_vals)

        if sample_U01() <= T(tau, cos_theta, p_i_max_vals, S_E_theta_vals):
            break

    # cos_theta is now set

    # Select active electron shell
    def F(p_z, tau, cos_theta):
        beta = np.sqrt(1 + (tau * tau) - (2 * tau * cos_theta))
        tmp = tau * (tau - cos_theta) / (beta * beta)
        return 1 + beta * (1 + tmp) * p_z / (ELECTRON_MASS * LIGHT_C)
    
    F_max = F(0.2 * ELECTRON_MASS * LIGHT_C, tau, cos_theta)

    total_shell_prob = sum(S_E_theta_vals)
    while True:
        rnd = sample_U01()
        active_shell = 0
        for i in range(len(S_E_theta_vals)):
            if rnd < (S_E_theta_vals[i] / total_shell_prob):
                active_shell = i
                break
        
        A = sample_U01() * n_i_A(active_shell, p_i_max_vals[active_shell])
        d_1 = np.sqrt(0.5)
        d_2 = np.sqrt(2)
        J_i = compton_data.compton_profile[i]

        p_z = None
        if A < 0.5:
            p_z = (d_1 - np.sqrt(d_1 * d_1 - np.log(2 * A))) / (d_2 * J_i)
        else:
            p_z = (np.sqrt(d_1 * d_1 - np.log(2 * (1 - A))) - d_1) / (d_2 * J_i)

        if p_z >= (-1 * ELECTRON_MASS * LIGHT_C):
            break

        if sample_U01() * F_max < F(p_z, tau, cos_theta):
            break
    
    # p_z is set

    def sign(x):
        return 1 if x > 0 else (0 if x == 0 else -1)
    
    t = (p_z * p_z) / (ELECTRON_MASS * ELECTRON_REST_ENERGY)
    tmp1 = 1 - t * tau * cos_theta
    tmp2 = 1 - t * tau * tau

    # E_prime = E_ratio * photon_energy
    E_ratio = (tmp1 + sign(p_z) * np.sqrt(tmp1 * tmp1 - tmp2 * (1 - tau))) * tau / tmp2
    return cos_theta, E_ratio * photon_energy
