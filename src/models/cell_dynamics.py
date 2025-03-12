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
from src.models.cell_production import compute_cell_sources, compute_cell_sources_scie3121_model
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import cg


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
    pressure = compute_internal_pressure(phi_H, phi_P, phi_D, phi_N, nutrient, dx, gamma, epsilon, lambda_S, lambda_P, mu_S, mu_P, mu_D, alpha_D, p_0, p_1, gamma_N, n_S, n_P, n_D)
    energy_deriv = compute_adhesion_energy_derivative(C_T, dx, gamma, epsilon)
    
    grad_C_x = gradient(C_T, dx, 0)
    grad_C_y = gradient(C_T, dx, 1)
    grad_C_z = gradient(C_T, dx, 2)
    grad_p_x = gradient(pressure, dx, 0)
    grad_p_y = gradient(pressure, dx, 1)
    grad_p_z = gradient(pressure, dx, 2)

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
        phi_H, phi_D, phi_N, nutrient, dx, gamma, epsilon,
        lambda_H, lambda_D, mu_H, mu_D, mu_N, p_H, p_D, n_H, n_D
    )
    energy_deriv = compute_adhesion_energy_derivative(phi_T, dx, gamma, epsilon)
    
    grad_C_x = gradient(phi_T, dx, 0)
    grad_C_y = gradient(phi_T, dx, 1)
    grad_C_z = gradient(phi_T, dx, 2)
    grad_p_x = gradient(pressure, dx, 0)
    grad_p_y = gradient(pressure, dx, 1)
    grad_p_z = gradient(pressure, dx, 2)

    ux = -(grad_p_x - energy_deriv * grad_C_x)
    uy = -(grad_p_y - energy_deriv * grad_C_y)
    uz = -(grad_p_z - energy_deriv * grad_C_z)

    #max_velocity = 10.0
    #ux = np.clip(ux, -max_velocity, max_velocity)
    #uy = np.clip(uy, -max_velocity, max_velocity)
    #uz = np.clip(uz, -max_velocity, max_velocity)

    # Debug: Check for NaN/Inf in velocities
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
    grad_C_x = gradient(C_T, dx, 0)
    grad_C_y = gradient(C_T, dx, 1)
    grad_C_z = gradient(C_T, dx, 2)
    grad_energy_x = gradient(energy_deriv, dx, 0)
    grad_energy_y = gradient(energy_deriv, dx, 1)
    grad_energy_z = gradient(energy_deriv, dx, 2)

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
    Solve the Poisson equation ∇²p = S_T - ∇·((δE/δφ_T) ∇φ_T) using Conjugate Gradient.
    Returns: pressure field as a NumPy array.
    """
    # Compute total cell volume fraction and source terms
    phi_T = phi_H + phi_D + phi_N
    src_H = lambda_H * nutrient * phi_H * (2 * p_H - 1) - mu_H * phi_H
    src_D = 2 * lambda_H * nutrient * (1 - p_H) * phi_H + lambda_D * nutrient * phi_D * (2 * p_D - 1)
    src_N = -mu_N * phi_N
    Src_T = src_H + src_D + src_N

    # Debug: Check source terms
    if np.any(np.isnan(Src_T)) or np.any(np.isinf(Src_T)):
        raise ValueError(f"Src_T contains NaN/Inf: max={np.max(Src_T)}, min={np.min(Src_T)}")

    # Compute adhesion energy derivative and spatial terms
    energy_deriv = compute_adhesion_energy_derivative(phi_T, dx, gamma, epsilon)
    laplace_C = laplacian(phi_T, dx)
    grad_C_x = gradient(phi_T, dx, 0)
    grad_C_y = gradient(phi_T, dx, 1)
    grad_C_z = gradient(phi_T, dx, 2)
    grad_energy_x = gradient(energy_deriv, dx, 0)
    grad_energy_y = gradient(energy_deriv, dx, 1)
    grad_energy_z = gradient(energy_deriv, dx, 2)

    # Debug: Check gradients
    if np.any(np.isnan(grad_C_x)) or np.any(np.isinf(grad_C_x)):
        raise ValueError(f"grad_C_x contains NaN/Inf: max={np.max(grad_C_x)}, min={np.min(grad_C_x)}")
    if np.any(np.isnan(grad_energy_x)) or np.any(np.isinf(grad_energy_x)):
        raise ValueError(f"grad_energy_x contains NaN/Inf: max={np.max(grad_energy_x)}, min={np.min(grad_energy_x)}")

    # Compute divergence term: ∇·((δE/δφ_T) ∇φ_T)
    divergence = (grad_energy_x * grad_C_x + grad_energy_y * grad_C_y + grad_energy_z * grad_C_z + 
                  energy_deriv * laplace_C)
    rhs = Src_T - divergence


    # Grid dimensions
    Nz, Ny, Nx = phi_T.shape
    N = Nx * Ny * Nz

    # Flatten rhs for solver
    rhs_flat = rhs.ravel()

    # Apply Dirichlet boundary conditions (p = 0 at boundaries)
    boundary_indices = np.unique(np.concatenate([
        np.arange(Nx * Ny),                        # z=0 plane
        np.arange(Nx * Ny * (Nz-1), Nx * Ny * Nz),  # z=Nz-1 plane
        np.arange(0, Nx * Ny * Nz, Nx * Ny),        # y=0 plane
        np.arange(Nx-1, Nx * Ny * Nz, Nx * Ny),     # y=Ny-1 plane
        np.arange(0, Nx * Ny * Nz, Nx),             # x=0 plane
        np.arange(Nx-1, Nx * Ny * Nz, Nx)           # x=Nx-1 plane
    ]))
    rhs_flat[boundary_indices] = 0.0

    # Build 3D Laplacian matrix
    main_diag = -6 * np.ones(N)  # 6 neighbors in 3D
    x_offsets = np.ones(N - 1)
    x_offsets[Nx-1::Nx] = 0  # Zero at row ends
    y_offsets = np.ones(N - Nx)
    y_offsets[Nx*(Ny-1)::Nx] = 0  # Zero at column ends
    z_offsets = np.ones(N - Nx * Ny)
    z_offsets[Nx*Ny*(Nz-1)::(Nx*Ny)] = 0  # Zero at layer ends

    diagonals = [
        x_offsets, y_offsets, z_offsets,  # Off-diagonals (+x, +y, +z)
        main_diag,                        # Main diagonal
        x_offsets, y_offsets, z_offsets   # Off-diagonals (-x, -y, -z)
    ]
    offsets = [-1, -Nx, -Nx*Ny, 0, 1, Nx, Nx*Ny]
    L = diags(diagonals, offsets, shape=(N, N)).tocsc() / (dx * dx)

    # Initial guess (proportional to phi_T)
    p_flat = phi_T.ravel() * 0.01

    # Solve using Conjugate Gradient
    p_flat, info = cg(L, -rhs_flat, x0=p_flat, atol=1e-8, maxiter=10000)
    if info != 0:
        raise ValueError(f"CG failed to converge, info={info}")

    # Reshape and validate
    p = p_flat.reshape((Nz, Ny, Nx))
    if np.any(np.isnan(p)) or np.any(np.isinf(p)):
        raise ValueError(f"Pressure contains NaN/Inf: max={np.max(p)}, min={np.min(p)}")

    return p

def compute_adhesion_energy_derivative(phi, dx, gamma, epsilon):
    f_prime = phi * (1 - phi)
    laplace_C = laplacian(phi, dx)
    energy_deriv = gamma * f_prime - 0.01 * gamma * epsilon * laplace_C
    return energy_deriv

def compute_mass_current(v_cell, phi_T, dx, gamma, epsilon, M):
    """
    Compute mass flux: J = -M ∇(δE/δφ_T) scaled by v_cell/phi_T.
    Returns: Jx, Jy, Jz as NumPy arrays.
    """
    energy_deriv = compute_adhesion_energy_derivative(phi_T, dx, gamma, epsilon)
    grad_energy_x = gradient(energy_deriv, dx, 0)
    grad_energy_y = gradient(energy_deriv, dx, 1)
    grad_energy_z = gradient(energy_deriv, dx, 2)

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
        return compute_cell_dynamics_scie3121_model(
            phi_H, phi_D, phi_N, nutrient, dx, 
            params['gamma'], params['epsilon'], params['M'],
            params['lambda_H'], params['lambda_D'], params['mu_H'], params['mu_D'], params['mu_N'],
            params['p_H'], params['p_D'], self.model.n_H, self.model.n_D
        )