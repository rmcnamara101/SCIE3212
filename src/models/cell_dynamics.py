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

from src.models.cell_production import compute_cell_sources, compute_cell_sources_scie3121_model, compute_pressure_cell_sources
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import cg, LinearOperator
from src.utils.utils import SCIE3121_params


def laplacian(field, dx):
    grad_x = np.gradient(field, axis=0, edge_order=2) * (1/dx)
    grad_y = np.gradient(field, axis=1, edge_order=2) * (1/dx)
    grad_z = np.gradient(field, axis=2, edge_order=2) * (1/dx)
    return (np.gradient(grad_x, axis=0, edge_order=2) +
            np.gradient(grad_y, axis=1, edge_order=2) +
            np.gradient(grad_z, axis=2, edge_order=2)) * (1/dx)

def divergence(ux, uy, uz, dx):
    return np.gradient(ux, axis=0, edge_order=2) / dx + np.gradient(uy, axis=1, edge_order=2) / dx + np.gradient(uz, axis=2, edge_order=2) / dx

@nb.njit
def compute_cell_dynamics(phi_H, phi_P, phi_D, phi_N, nutrient, dx, gamma, epsilon, M, lambda_S, lambda_P, mu_S, mu_P, mu_D, alpha_D, p_0, p_1, gamma_N, n_S, n_P, n_D):
    """
    Compute dynamics derivatives: -∇·(u C) - ∇·J.
    Returns: dC_S, dC_P, dC_D, dC_N as NumPy arrays.
    """
    C_T = phi_H + phi_P + phi_D + phi_N
    ux, uy, uz = 0.5 * compute_solid_velocity(phi_H, phi_P, phi_D, phi_N, nutrient, dx, gamma, epsilon, lambda_S, lambda_P, mu_S, mu_P, mu_D, alpha_D, p_0, p_1, gamma_N, n_S, n_P, n_D)
    Jx_S, Jy_S, Jz_S = compute_mass_current(phi_H, C_T, dx, gamma, epsilon, M)
    Jx_P, Jy_P, Jz_P = compute_mass_current(phi_P, C_T, dx, gamma, epsilon, M)
    Jx_D, Jy_D, Jz_D = compute_mass_current(phi_D, C_T, dx, gamma, epsilon, M)
    Jx_N, Jy_N, Jz_N = compute_mass_current(phi_N, C_T, dx, gamma, epsilon, 0.000001)

    dC_S = -divergence(ux * phi_H, uy * phi_H, uz * phi_H, dx) - divergence(Jx_S, Jy_S, Jz_S, dx)
    dC_P = -divergence(ux * phi_P, uy * phi_P, uz * phi_P, dx) - divergence(Jx_P, Jy_P, Jz_P, dx)
    dC_D = -divergence(ux * phi_D, uy * phi_D, uz * phi_D, dx) - divergence(Jx_D, Jy_D, Jz_D, dx)
    dC_N = -divergence(ux * phi_N, uy * phi_N, uz * phi_N, dx) - divergence(Jx_N, Jy_N, Jz_N, dx)

    return dC_S, dC_P, dC_D, dC_N

def compute_cell_dynamics_scie3121_model(phi_H, phi_D, phi_N, nutrient, dx, gamma, epsilon, M, lambda_H, lambda_D, mu_H, mu_D, mu_N, p_H, p_D, n_H, n_D):
    """
    Compute dynamics derivatives: -∇·(u \phi) - ∇·J for SCIE3121 model.
    Returns: dphi_H, dphi_D, dphi_N as NumPy arrays.
    """
    phi_T = phi_H + phi_D + phi_N
    ux, uy, uz = compute_solid_velocity_scie3121_model(
        phi_H, phi_D, phi_N, nutrient, dx, gamma, epsilon,
        lambda_H, lambda_D, mu_H, mu_D, mu_N, p_H, p_D, n_H, n_D
    )
    Jx_S, Jy_S, Jz_S = compute_mass_current(phi_H, phi_T, dx, gamma, epsilon, M)
    Jx_D, Jy_D, Jz_D = compute_mass_current(phi_D, phi_T, dx, gamma, epsilon, M)
    Jx_N, Jy_N, Jz_N = compute_mass_current(phi_N, phi_T, dx, gamma, epsilon, 1e-6)

    dphi_H = -divergence(ux * phi_H, uy * phi_H, uz * phi_H, dx) - divergence(Jx_S, Jy_S, Jz_S, dx)
    dphi_D = -divergence(ux * phi_D, uy * phi_D, uz * phi_D, dx) - divergence(Jx_D, Jy_D, Jz_D, dx)
    dphi_N = -divergence(ux * phi_N, uy * phi_N, uz * phi_N, dx) - divergence(Jx_N, Jy_N, Jz_N, dx)

    # Debug: Check for NaN/Inf in derivatives
    if np.any(np.isnan(dphi_H)) or np.any(np.isinf(dphi_H)):
        raise ValueError(f"dphi_H contains NaN/Inf: max={np.max(dphi_H)}, min={np.min(dphi_H)}")
    if np.any(np.isnan(dphi_D)) or np.any(np.isinf(dphi_D)):
        raise ValueError(f"dphi_D contains NaN/Inf: max={np.max(dphi_D)}, min={np.min(dphi_D)}")
    if np.any(np.isnan(dphi_N)) or np.any(np.isinf(dphi_N)):
        raise ValueError(f"dphi_N contains NaN/Inf: max={np.max(dphi_N)}, min={np.min(dphi_N)}")

    return dphi_H, dphi_D, dphi_N

@nb.njit
def compute_solid_velocity(phi_H, phi_P, phi_D, phi_N, nutrient, dx, gamma, epsilon, lambda_S, lambda_P, mu_S, mu_P, mu_D, alpha_D, p_0, p_1, gamma_N, n_S, n_P, n_D):
    C_T = phi_H + phi_D + phi_N
    pressure = compute_internal_pressure(
        phi_H, phi_P, phi_D, phi_N, nutrient, dx,
        gamma, epsilon, 
        lambda_S, lambda_P, 
        mu_S, mu_P, mu_D, 
        alpha_D, p_0, p_1, gamma_N, n_S, n_P, n_D)
    energy_deriv = compute_adhesion_energy_derivative(C_T, dx, gamma, epsilon)
    
    grad_C_x = np.gradient(C_T, dx, 0)
    grad_C_y = np.gradient(C_T, dx, 1)
    grad_C_z = np.gradient(C_T, dx, 2)
    grad_p_x = np.gradient(pressure, dx, 0)
    grad_p_y = np.gradient(pressure, dx, 1)
    grad_p_z = np.gradient(pressure, dx, 2)

    ux = -(grad_p_x - energy_deriv * grad_C_x)
    uy = -(grad_p_y - energy_deriv * grad_C_y)
    uz = -(grad_p_z - energy_deriv * grad_C_z)

    #max_velocity = 10.0
    #ux = np.clip(ux, -max_velocity, max_velocity)
    #uy = np.clip(uy, -max_velocity, max_velocity)
    #uz = np.clip(uz, -max_velocity, max_velocity)

    return ux, uy, uz

def compute_solid_velocity_scie3121_model(phi_H, phi_D, phi_N, nutrient, dx, gamma, epsilon, lambda_H, lambda_D, mu_H, mu_D, mu_N, p_H, p_D, n_H, n_D):
    """
    Compute solid velocity: u = -(∇p + (δE/δφ_T) ∇φ_T).
    Returns: ux, uy, uz as NumPy arrays.
    """
    phi_T = phi_H + phi_D + phi_N
    pressure = compute_internal_pressure_scie3121_model(
        phi_H, phi_D, phi_N, nutrient, dx, 
        gamma, epsilon,
        lambda_H, lambda_D, 
        mu_H, mu_D, mu_N, 
        p_H, p_D, n_H, n_D
    )
    energy_deriv = compute_adhesion_energy_derivative(phi_T, dx, gamma, epsilon)
    
    grad_C_x = np.gradient(phi_T, axis=0) / dx
    grad_C_y = np.gradient(phi_T, axis=1) / dx
    grad_C_z = np.gradient(phi_T, axis=2) / dx
    grad_p_x = np.gradient(pressure, axis=0) / dx
    grad_p_y = np.gradient(pressure, axis=1) / dx
    grad_p_z = np.gradient(pressure, axis=2) / dx

    ux = -(grad_p_x + energy_deriv * grad_C_x)
    uy = -(grad_p_y + energy_deriv * grad_C_y)
    uz = -(grad_p_z + energy_deriv * grad_C_z)

    if np.any(np.isnan(ux)) or np.any(np.isinf(ux)):
        raise ValueError(f"ux contains NaN/Inf: max={np.max(ux)}, min={np.min(ux)}")
    
    return ux, uy, uz

@nb.njit
def compute_internal_pressure(phi_H, phi_P, phi_D, phi_N, nutrient, dx, gamma, epsilon, lambda_S, lambda_P, mu_S, mu_P, mu_D, alpha_D, p_0, p_1, gamma_N, n_S, n_P, n_D):
    C_T = phi_H + phi_P + phi_D + phi_N
    src_S, src_P, src_D, src_N = compute_cell_sources(phi_H, phi_P, phi_D, phi_N, nutrient, n_S, n_P, n_D, lambda_S, lambda_P, mu_S, mu_P, mu_D, alpha_D, p_0, p_1, gamma_N)
    S_T = src_S + src_P + src_D + src_N

    energy_deriv = compute_adhesion_energy_derivative(C_T, dx, gamma, epsilon)
    laplace_C = laplacian(C_T, dx)
    grad_C_x = np.gradient(C_T, dx, 0)
    grad_C_y = np.gradient(C_T, dx, 1)
    grad_C_z = np.gradient(C_T, dx, 2)
    grad_energy_x = np.gradient(energy_deriv, dx, 0)
    grad_energy_y = np.gradient(energy_deriv, dx, 1)
    grad_energy_z = np.gradient(energy_deriv, dx, 2)

    divergence = grad_energy_x * grad_C_x + grad_energy_y * grad_C_y + grad_energy_z * grad_C_z + energy_deriv * laplace_C
    rhs = S_T - divergence


    p = np.zeros_like(rhs)
    for _ in range(50):
        p_new = laplacian(p, dx) * dx * dx + rhs
        p = (p_new + p) / 2.0
    return p

def compute_pressure_with_debug(phi_H, phi_D, phi_N, nutrient, dx, gamma, epsilon, lambda_H, lambda_D, mu_H, mu_D, mu_N, p_H, p_D, n_H, n_D):
    p = compute_internal_pressure_scie3121_model(phi_H, phi_D, phi_N, nutrient, dx, gamma, epsilon, lambda_H, lambda_D, mu_H, mu_D, mu_N, p_H, p_D, n_H, n_D)
    print(f"Final Max pressure: {np.max(p)}, Min pressure: {np.min(p)}")
    return p   

def compute_internal_pressure_scie3121_model(phi_H, phi_D, phi_N, nutrient, dx, gamma, epsilon, lambda_H, lambda_D, mu_H, mu_D, mu_N, p_H, p_D, n_H, n_D):
    """
    Compute the internal pressure field for the SCIE3121 model using conjugate gradient.
    Args:
        phi_H, phi_D, phi_N: Volume fractions for healthy, dead, and nutrient cells.
        nutrient: Nutrient concentration field.
        dx: Grid spacing.
        gamma, epsilon, lambda_H, lambda_D, mu_H, mu_D, mu_N, p_H, p_D, n_H, n_D: Model parameters.
    Returns:
        pressure: 3D pressure field array.
    """
    phi_T = phi_H + phi_D + phi_N
    # Compute source terms (assuming compute_cell_sources_scie3121_model exists)
    src_H, src_D, src_N = compute_pressure_cell_sources(
        phi_H, phi_D, phi_N, nutrient, n_H, n_D, lambda_H, lambda_D, mu_H, mu_D, mu_N, p_H, p_D
    )
    S_T = src_H + src_D + src_N

    # Compute the right-hand side: S_T - ∇·(δE/δφ_T ∇φ_T)
    energy_deriv = compute_adhesion_energy_derivative(phi_T, dx, gamma, epsilon)
    grad_C_x = np.gradient(phi_T, axis=0) / dx
    grad_C_y = np.gradient(phi_T, axis=1) / dx
    grad_C_z = np.gradient(phi_T, axis=2) / dx
    grad_energy_x = np.gradient(energy_deriv, axis=0) / dx
    grad_energy_y = np.gradient(energy_deriv, axis=1) / dx
    grad_energy_z = np.gradient(energy_deriv, axis=2) / dx
    laplace_C = laplacian(phi_T, dx)
    divergence = grad_energy_x * grad_C_x + grad_energy_y * grad_C_y + grad_energy_z * grad_C_z + energy_deriv * laplace_C
    rhs = S_T - divergence

    # Solve ∇²p = rhs using conjugate gradient
    shape = phi_T.shape
    A = get_laplacian_operator(shape, dx)
    rhs_flat = rhs.flatten()
    p_flat, info = cg(A, rhs_flat, rtol=1e-6, maxiter=100)
    if info < 0:
        raise ValueError("Illegal input or breakdown in CG solver")
    pressure = p_flat.reshape(shape)

    # Debug output
    if np.any(np.isnan(pressure)) or np.any(np.isinf(pressure)):
        raise ValueError(f"Pressure contains NaN/Inf: max={np.max(pressure)}, min={np.min(pressure)}")
    
    return -pressure

def compute_adhesion_energy_derivative(phi, dx, gamma, epsilon):
    f_prime = 0.5 * phi * (1 - phi) * (2 * phi - 1)
    laplace_phi = laplacian(phi, dx)
    energy_deriv = ( gamma / epsilon ) * f_prime - 0.01 * gamma * epsilon * laplace_phi
    return energy_deriv

def compute_mass_current(v_cell, phi_T, dx, gamma, epsilon, M):
    """
    Compute mass flux: J = -M ∇(δE/δφ_T) scaled by v_cell/phi_T.
    Returns: Jx, Jy, Jz as NumPy arrays.
    """
    energy_deriv = compute_adhesion_energy_derivative(phi_T, dx, gamma, epsilon)
    grad_energy_x = np.gradient(energy_deriv, axis=0) / dx
    grad_energy_y = np.gradient(energy_deriv, axis=1) / dx
    grad_energy_z = np.gradient(energy_deriv, axis=2) / dx

    Jx = -M * grad_energy_x
    Jy = -M * grad_energy_y
    Jz = -M * grad_energy_z

    epsilon_small = 1e-6  # Increased for stability
    phi_T_clamped = np.where(phi_T < epsilon_small, epsilon_small, phi_T)
    scaling = v_cell / phi_T_clamped
    Jx = scaling * Jx
    Jy = scaling * Jy
    Jz = scaling * Jz

    return Jx, Jy, Jz


def laplacian_neumann(p_flat, shape, dx):
    """
    Compute the 3D Laplacian with Neumann boundary conditions.
    Args:
        p_flat: Flattened pressure array.
        shape: Tuple (nx, ny, nz) of the grid shape.
        dx: Grid spacing.
    Returns:
        Flattened Laplacian array (float64).
    """
    p = p_flat.reshape(shape)
    # Explicitly use float64 for lap_p to avoid dtype mismatch
    lap_p = np.zeros(shape, dtype=np.float64)
    # x-direction
    lap_p[1:-1, :, :] = (p[2:, :, :] - 2*p[1:-1, :, :] + p[:-2, :, :]) / dx**2
    lap_p[0, :, :] = 2*(p[1, :, :] - p[0, :, :]) / dx**2
    lap_p[-1, :, :] = 2*(p[-2, :, :] - p[-1, :, :]) / dx**2
    # y-direction
    lap_p[:, 1:-1, :] += (p[:, 2:, :] - 2*p[:, 1:-1, :] + p[:, :-2, :]) / dx**2
    lap_p[:, 0, :] += 2*(p[:, 1, :] - p[:, 0, :]) / dx**2
    lap_p[:, -1, :] += 2*(p[:, -2, :] - p[:, -1, :]) / dx**2
    # z-direction
    lap_p[:, :, 1:-1] += (p[:, :, 2:] - 2*p[:, :, 1:-1] + p[:, :, :-2]) / dx**2
    lap_p[:, :, 0] += 2*(p[:, :, 1] - p[:, :, 0]) / dx**2
    lap_p[:, :, -1] += 2*(p[:, :, -2] - p[:, :, -1]) / dx**2
    return lap_p.flatten()

def get_laplacian_operator(shape, dx):
    """
    Create a LinearOperator for the Laplacian with Neumann conditions.
    Args:
        shape: Tuple (nx, ny, nz) of the grid shape.
        dx: Grid spacing.
    Returns:
        LinearOperator representing the Laplacian (float64 output).
    """
    N = np.prod(shape)
    def matvec(p_flat):
        return laplacian_neumann(p_flat, shape, dx)
    # Specify dtype as float64 to ensure compatibility
    return LinearOperator((N, N), matvec=matvec, dtype=np.float64)

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
    

class SCIE3121_DynamicsModel(DynamicsModel):
    def __init__(self, model):
        super().__init__(model)
    
    def compute_cell_dynamics(self, phi_H, phi_D, phi_N, nutrient, dx, params):
        """
        Compute cell dynamics for SCIE3121 model.
        Returns: dphi_H, dphi_D, dphi_N as NumPy arrays.
        """
        phi_T = phi_H + phi_D + phi_N
        ux, uy, uz = compute_solid_velocity_scie3121_model(
            phi_H, phi_D, phi_N, nutrient, dx, 
            params['gamma'], params['epsilon'],
            params['lambda_H'], params['lambda_D'], params['mu_H'], params['mu_D'], params['mu_N'],
            params['p_H'], params['p_D'], self.model.n_H, self.model.n_D
        )
        # Assuming compute_mass_current remains the same
        Jx_H, Jy_H, Jz_H = compute_mass_current(phi_H, phi_T, dx, params['gamma'], params['epsilon'], params['M'])
        Jx_D, Jy_D, Jz_D = compute_mass_current(phi_D, phi_T, dx, params['gamma'], params['epsilon'], params['M'])
        Jx_N, Jy_N, Jz_N = compute_mass_current(phi_N, phi_T, dx, params['gamma'], params['epsilon'], 1e-6)

        # Compute divergence (reusing your divergence function)
        dphi_H = -divergence(ux * phi_H, uy * phi_H, uz * phi_H, dx) - divergence(Jx_H, Jy_H, Jz_H, dx)
        dphi_D = -divergence(ux * phi_D, uy * phi_D, uz * phi_D, dx) - divergence(Jx_D, Jy_D, Jz_D, dx)
        dphi_N = -divergence(ux * phi_N, uy * phi_N, uz * phi_N, dx) - divergence(Jx_N, Jy_N, Jz_N, dx)

        # Debug checks
        if np.any(np.isnan(dphi_H)) or np.any(np.isinf(dphi_H)):
            raise ValueError(f"dphi_H contains NaN/Inf: max={np.max(dphi_H)}, min={np.min(dphi_H)}")
        
        return dphi_H, dphi_D, dphi_N