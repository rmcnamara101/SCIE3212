####################################################################################
####################################################################################
#
#
#   Cell Dynamics Model
#
#   This model computes the dynamics of the cell concentration fields
#   based on the solid velocity field and the mass flux, defined as follows:
#
#   dC = -∇·(u C) -  ∇·J
#   dt
#
#   where u is the solid velocity field, J is the mass flux, and M is the mobility.
#
#   The solid velocity is defined and computed using the following formulas::
#
#   u = - (∇p + ( δE ) ∇C_T)
#               (δC_T)
#
#    ∇²p = S_T - ∇·(( δE ) ∇C_T)
#                   (δC_T)
#
#   where p is the pressure field δE/δC_T is the variational derivative of the 
#   adhesion energy with respect to the total cell density C_T, S_T is the source
#   term for the total cell density.
#
#   The mass flux is defined as follows:
#
#   J = -M ∇( δE )
#           (δC_T)
#
#
#   This file should be used as an importable class, that will take a tumor growth 
#   model as an argument, will be able to update the concentration fields of the tumor
#   based on the solid velocity field and the mass flux.
#   This class should be used in the following way:
#
#
#   <code>
#
#   model = TumorGrowthModel()
#   cell_dynamics = CellDynamics(model)
#   cell_dynamics.compute_cell_dynamics()
#
#   </code>
#
#
#   This will return a set of derivatives of each of the volume fraction fields.
#
#
# Author:
#   - Riley Jae McNamara
#
# Date:
#   - 2025-02-19
#
####################################################################################
####################################################################################


import numpy as np
import numba as nb
from src.utils.utils import gradient, laplacian, divergence
from src.models.cell_production import compute_cell_sources


@nb.njit
def compute_cell_dynamics(phi_H, phi_P, phi_D, phi_N, nutrient, dx, gamma, epsilon, M, lambda_S, lambda_P, mu_S, mu_P, mu_D, alpha_D, p_0, p_1, gamma_N, n_S, n_P, n_D):
    """
    Compute dynamics derivatives: -∇·(u C) - ∇·J.
    Returns: dC_S, dC_P, dC_D, dC_N as NumPy arrays.
    """
    C_T = phi_H + phi_P + phi_D + phi_N
    ux, uy, uz = compute_solid_velocity(phi_H, phi_P, phi_D, phi_N, nutrient, dx, gamma, epsilon, lambda_S, lambda_P, mu_S, mu_P, mu_D, alpha_D, p_0, p_1, gamma_N, n_S, n_P, n_D)
    Jx_S, Jy_S, Jz_S = compute_mass_current(phi_H, C_T, dx, gamma, epsilon, M)
    Jx_P, Jy_P, Jz_P = compute_mass_current(phi_P, C_T, dx, gamma, epsilon, M)
    Jx_D, Jy_D, Jz_D = compute_mass_current(phi_D, C_T, dx, gamma, epsilon, M)
    Jx_N, Jy_N, Jz_N = compute_mass_current(phi_N, C_T, dx, gamma, epsilon, M)

    dC_S = -divergence(ux * phi_H, uy * phi_H, uz * phi_H, dx) - divergence(Jx_S, Jy_S, Jz_S, dx)
    dC_P = -divergence(ux * phi_P, uy * phi_P, uz * phi_P, dx) - divergence(Jx_P, Jy_P, Jz_P, dx)
    dC_D = -divergence(ux * phi_D, uy * phi_D, uz * phi_D, dx) - divergence(Jx_D, Jy_D, Jz_D, dx)
    dC_N = -divergence(ux * phi_N, uy * phi_N, uz * phi_N, dx) - divergence(Jx_N, Jy_N, Jz_N, dx)

    return dC_S, dC_P, dC_D, dC_N

@nb.njit
def compute_solid_velocity(phi_H, phi_P, phi_D, phi_N, nutrient, dx, gamma, epsilon, lambda_S, lambda_P, mu_S, mu_P, mu_D, alpha_D, p_0, p_1, gamma_N, n_S, n_P, n_D):
    C_T = phi_H + phi_P + phi_D + phi_N
    pressure = compute_internal_pressure(phi_H, phi_P, phi_D, phi_N, nutrient, dx, gamma, epsilon, lambda_S, lambda_P, mu_S, mu_P, mu_D, alpha_D, p_0, p_1, gamma_N, n_S, n_P, n_D)
    energy_deriv = compute_adhesion_energy_derivative(C_T, dx, gamma, epsilon)
    
    grad_C_x = gradient(C_T, dx, 0)
    grad_C_y = gradient(C_T, dx, 1)
    grad_C_z = gradient(C_T, dx, 2)
    grad_p_x = gradient(pressure, dx, 0)
    grad_p_y = gradient(pressure, dx, 1)
    grad_p_z = gradient(pressure, dx, 2)

    ux = -(grad_p_x + energy_deriv * grad_C_x)
    uy = -(grad_p_y + energy_deriv * grad_C_y)
    uz = -(grad_p_z + energy_deriv * grad_C_z)

    max_velocity = 10.0
    ux = np.clip(ux, -max_velocity, max_velocity)
    uy = np.clip(uy, -max_velocity, max_velocity)
    uz = np.clip(uz, -max_velocity, max_velocity)

    return ux, uy, uz

@nb.njit
def compute_internal_pressure(phi_H, phi_P, phi_D, phi_N, nutrient, dx, gamma, epsilon, lambda_S, lambda_P, mu_S, mu_P, mu_D, alpha_D, p_0, p_1, gamma_N, n_S, n_P, n_D):
    C_T = phi_H + phi_P + phi_D + phi_N
    src_S, src_P, src_D, src_N = compute_cell_sources(phi_H, phi_P, phi_D, phi_N, nutrient, n_S, n_P, n_D, lambda_S, lambda_P, mu_S, mu_P, mu_D, alpha_D, p_0, p_1, gamma_N)
    S_T = src_S + src_P + src_D + src_N

    energy_deriv = compute_adhesion_energy_derivative(C_T, dx, gamma, epsilon)
    laplace_C = laplacian(C_T, dx)
    grad_C_x = gradient(C_T, dx, 0)
    grad_C_y = gradient(C_T, dx, 1)
    grad_C_z = gradient(C_T, dx, 2)
    grad_energy_x = gradient(energy_deriv, dx, 0)
    grad_energy_y = gradient(energy_deriv, dx, 1)
    grad_energy_z = gradient(energy_deriv, dx, 2)

    divergence = grad_energy_x * grad_C_x + grad_energy_y * grad_C_y + grad_energy_z * grad_C_z + energy_deriv * laplace_C
    rhs = S_T - divergence
    rhs = np.clip(rhs, -1e3, 1e3)  # Stabilize RHS

    p = np.zeros_like(rhs)
    for _ in range(10):
        p_new = laplacian(p, dx) * dx * dx + rhs
        p = np.clip((p_new + p) / 2.0, -1e3, 1e3)  # Tighter bounds
    return p

@nb.njit
def compute_adhesion_energy_derivative(phi, dx, gamma, epsilon):
    phi = np.clip(phi, 1e-10, 1 - 1e-10)
    f_prime = 0.5 * phi * (1 - phi) * (2 * phi - 1)
    laplace_C = laplacian(phi, dx)
    energy_deriv = (gamma / epsilon) * f_prime - gamma * epsilon * laplace_C
    return np.clip(energy_deriv, -500.0, 500.0)

@nb.njit
def compute_mass_current(v_cell, phi_T, dx, gamma, epsilon, M):
    energy_deriv = compute_adhesion_energy_derivative(phi_T, dx, gamma, epsilon)
    grad_energy_x = gradient(energy_deriv, dx, 0)
    grad_energy_y = gradient(energy_deriv, dx, 1)
    grad_energy_z = gradient(energy_deriv, dx, 2)

    Jx = -M * grad_energy_x
    Jy = -M * grad_energy_y
    Jz = -M * grad_energy_z

    epsilon_small = 1e-10
    scaling = v_cell / (phi_T + epsilon_small)
    Jx = scaling * Jx
    Jy = scaling * Jy
    Jz = scaling * Jz

    return Jx, Jy, Jz


class DynamicsModel:
    def __init__(self, model):
        self.model = model

    def compute_cell_dynamics(self, phi_H, phi_P, phi_D, phi_N, nutrient, dx, params):
        """Wrapper to call the static Numba-optimized function."""
        return compute_cell_dynamics(
            phi_H, phi_P, phi_D, phi_N, nutrient, dx,
            params['gamma'], params['epsilon'], params['M'],
            params['lambda_S'], params['lambda_P'], params['mu_S'], params['mu_P'],
            params['mu_D'], params['alpha_D'], params['p_0'], params['p_1'], params['gamma_N'],
            self.model.n_H, self.model.n_P, self.model.n_D
        )