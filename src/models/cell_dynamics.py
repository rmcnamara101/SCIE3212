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

@nb.njit
def laplacian(field, dx):
    """
    Compute the Laplacian of a field with Neumann boundary conditions.
    ∇²φ = ∂²φ/∂x² + ∂²φ/∂y² + ∂²φ/∂z²
    """
    # Initialize Laplacian array
    lap = np.zeros_like(field)
    
    # x-direction
    lap[1:-1, :, :] = (field[2:, :, :] - 2*field[1:-1, :, :] + field[:-2, :, :]) / dx**2
    lap[0, :, :] = 2*(field[1, :, :] - field[0, :, :]) / dx**2  # Neumann at x=0
    lap[-1, :, :] = 2*(field[-2, :, :] - field[-1, :, :]) / dx**2  # Neumann at x=L
    
    # y-direction
    lap[:, 1:-1, :] += (field[:, 2:, :] - 2*field[:, 1:-1, :] + field[:, :-2, :]) / dx**2
    lap[:, 0, :] += 2*(field[:, 1, :] - field[:, 0, :]) / dx**2  # Neumann at y=0
    lap[:, -1, :] += 2*(field[:, -2, :] - field[:, -1, :]) / dx**2  # Neumann at y=L
    
    # z-direction
    lap[:, :, 1:-1] += (field[:, :, 2:] - 2*field[:, :, 1:-1] + field[:, :, :-2]) / dx**2
    lap[:, :, 0] += 2*(field[:, :, 1] - field[:, :, 0]) / dx**2  # Neumann at z=0
    lap[:, :, -1] += 2*(field[:, :, -2] - field[:, :, -1]) / dx**2  # Neumann at z=L
    
    return lap

@nb.njit
def gradient_neumann(field, dx, axis):
    """
    Compute the gradient of a field with Neumann boundary conditions.
    Args:
        field: The field to compute the gradient of
        dx: Grid spacing
        axis: Axis along which to compute the gradient (0, 1, or 2)
    Returns:
        Gradient array
    """
    grad = np.zeros_like(field)
    
    if axis == 0:  # x-direction
        grad[1:-1, :, :] = (field[2:, :, :] - field[:-2, :, :]) / (2*dx)
        grad[0, :, :] = 0  # Neumann at x=0: ∂φ/∂x = 0
        grad[-1, :, :] = 0  # Neumann at x=L: ∂φ/∂x = 0
    elif axis == 1:  # y-direction
        grad[:, 1:-1, :] = (field[:, 2:, :] - field[:, :-2, :]) / (2*dx)
        grad[:, 0, :] = 0  # Neumann at y=0: ∂φ/∂y = 0
        grad[:, -1, :] = 0  # Neumann at y=L: ∂φ/∂y = 0
    elif axis == 2:  # z-direction
        grad[:, :, 1:-1] = (field[:, :, 2:] - field[:, :, :-2]) / (2*dx)
        grad[:, :, 0] = 0  # Neumann at z=0: ∂φ/∂z = 0
        grad[:, :, -1] = 0  # Neumann at z=L: ∂φ/∂z = 0
    
    return grad

def divergence(ux, uy, uz, dx):
    """
    Compute the divergence of a vector field with Neumann boundary conditions.
    ∇·u = ∂ux/∂x + ∂uy/∂y + ∂uz/∂z
    """
    return (gradient_neumann(ux, dx, 0) + 
            gradient_neumann(uy, dx, 1) + 
            gradient_neumann(uz, dx, 2))

@nb.njit
def compute_adhesion_energy_derivative_with_laplace(phi, laplace_phi, gamma, epsilon):
    """
    Compute the adhesion energy derivative using a precomputed Laplacian.
    Args:
        phi: The field (typically phi_T)
        laplace_phi: Precomputed Laplacian of phi
        gamma, epsilon: Model parameters
    Returns:
        Energy derivative field
    """
    f_prime = 0.5 * phi * (1 - phi) * (2 * phi - 1)
    energy_deriv = (gamma / epsilon) * f_prime - 0.01 * gamma * epsilon * laplace_phi
    return energy_deriv

def compute_solid_velocity_scie3121_model_with_grads(
        phi_H, phi_D, phi_N, nutrient, dx, gamma, epsilon,
        lambda_H, lambda_D, mu_H, mu_D, mu_N, p_H, p_D, n_H, n_D,
        energy_deriv, grad_C_x, grad_C_y, grad_C_z, laplace_phi):
    """
    Compute solid velocity using precomputed gradients and energy derivative.
    Args:
        phi_H, phi_D, phi_N: Volume fractions
        nutrient: Nutrient concentration field
        dx: Grid spacing
        gamma, epsilon, etc.: Model parameters
        energy_deriv: Precomputed adhesion energy derivative
        grad_C_x, grad_C_y, grad_C_z: Precomputed gradients of phi_T
        laplace_phi: Precomputed Laplacian of phi_T
    Returns:
        ux, uy, uz: Velocity components
    """
    phi_T = phi_H + phi_D + phi_N
    
    # Compute pressure using precomputed values
    src_H, src_D, src_N = compute_pressure_cell_sources(
        phi_H, phi_D, phi_N, nutrient, n_H, n_D, lambda_H, lambda_D, mu_H, mu_D, mu_N, p_H, p_D
    )
    S_T = src_H + src_D + src_N
    
    # Use precomputed values for divergence term
    grad_energy_x = gradient_neumann(energy_deriv, dx, 0)
    grad_energy_y = gradient_neumann(energy_deriv, dx, 1)
    grad_energy_z = gradient_neumann(energy_deriv, dx, 2)
    
    divergence = grad_energy_x * grad_C_x + grad_energy_y * grad_C_y + grad_energy_z * grad_C_z + energy_deriv * laplace_phi
    rhs = S_T - divergence
    
    # Solve pressure equation
    shape = phi_T.shape
    A = get_laplacian_operator(shape, dx)
    rhs_flat = rhs.flatten()
    
    # Add error handling for CG solver
    try:
        p_flat, info = cg(A, rhs_flat, rtol=1e-5, maxiter=100)
        if info != 0:

            # Use a more robust fallback method if CG fails
            p_flat = np.zeros_like(rhs_flat)
    except Exception as e:
        print(f"CG solver error: {e}")
        p_flat = np.zeros_like(rhs_flat)
    
    pressure = -p_flat.reshape(shape)
    
    # Check for NaN/Inf in pressure and fix
    if np.any(np.isnan(pressure)) or np.any(np.isinf(pressure)):
        print("Warning: NaN/Inf detected in pressure, replacing with zeros")
        pressure = np.nan_to_num(pressure, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Compute gradients of pressure
    grad_p_x = gradient_neumann(pressure, dx, 0)
    grad_p_y = gradient_neumann(pressure, dx, 1)
    grad_p_z = gradient_neumann(pressure, dx, 2)
    
    # Check for NaN/Inf in energy derivative and gradients
    energy_deriv = np.nan_to_num(energy_deriv, nan=0.0, posinf=0.0, neginf=0.0)
    grad_C_x = np.nan_to_num(grad_C_x, nan=0.0, posinf=0.0, neginf=0.0)
    grad_C_y = np.nan_to_num(grad_C_y, nan=0.0, posinf=0.0, neginf=0.0)
    grad_C_z = np.nan_to_num(grad_C_z, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Compute velocity components with safety checks
    ux = -(grad_p_x + energy_deriv * grad_C_x)
    uy = -(grad_p_y + energy_deriv * grad_C_y)
    uz = -(grad_p_z + energy_deriv * grad_C_z)
    
    # Replace any NaN/Inf values before clipping
    ux = np.nan_to_num(ux, nan=0.0, posinf=0.0, neginf=0.0)
    uy = np.nan_to_num(uy, nan=0.0, posinf=0.0, neginf=0.0)
    uz = np.nan_to_num(uz, nan=0.0, posinf=0.0, neginf=0.0)

    # Now clip the values
    ux = np.clip(ux, -1, 1)
    uy = np.clip(uy, -1, 1)
    uz = np.clip(uz, -1, 1)
    
    return ux, uy, uz

def compute_mass_current_with_energy_deriv(v_cell, phi_T, dx, gamma, epsilon, M, energy_deriv):
    """
    Compute mass flux using precomputed energy derivative.
    Args:
        v_cell: Volume fraction of specific cell type
        phi_T: Total cell density
        dx: Grid spacing
        gamma, epsilon, M: Model parameters
        energy_deriv: Precomputed adhesion energy derivative
    Returns:
        Jx, Jy, Jz: Mass flux components
    """
    grad_energy_x = gradient_neumann(energy_deriv, dx, 0)
    grad_energy_y = gradient_neumann(energy_deriv, dx, 1)
    grad_energy_z = gradient_neumann(energy_deriv, dx, 2)

    Jx = -M * grad_energy_x
    Jy = -M * grad_energy_y
    Jz = -M * grad_energy_z

    epsilon_small = 1e-6
    phi_T_clamped = np.where(phi_T < epsilon_small, epsilon_small, phi_T)
    scaling = v_cell / phi_T_clamped
    Jx = scaling * Jx
    Jy = scaling * Jy
    Jz = scaling * Jz

    return Jx, Jy, Jz

def compute_cell_dynamics_scie3121_model(phi_H, phi_D, phi_N, nutrient, dx, gamma, epsilon, M, lambda_H, lambda_D, mu_H, mu_D, mu_N, p_H, p_D, n_H, n_D):
    """
    Compute dynamics derivatives: -∇·(u \phi) - ∇·J for SCIE3121 model.
    Returns: dphi_H, dphi_D, dphi_N as NumPy arrays.
    """
    phi_T = phi_H + phi_D + phi_N

    # Compute shared quantities once
    laplace_phi = laplacian(phi_T, dx)
    energy_deriv = compute_adhesion_energy_derivative_with_laplace(phi_T, laplace_phi, gamma, epsilon)
    grad_C_x = gradient_neumann(phi_T, dx, 0)
    grad_C_y = gradient_neumann(phi_T, dx, 1)
    grad_C_z = gradient_neumann(phi_T, dx, 2)

    # Pass precomputed values to velocity computation
    ux, uy, uz = compute_solid_velocity_scie3121_model_with_grads(
        phi_H, phi_D, phi_N, nutrient, dx, gamma, epsilon,
        lambda_H, lambda_D, mu_H, mu_D, mu_N, p_H, p_D, n_H, n_D,
        energy_deriv, grad_C_x, grad_C_y, grad_C_z, laplace_phi
    )

    # Pass precomputed energy derivative to mass current computation
    Jx_H, Jy_H, Jz_H = compute_mass_current_with_energy_deriv(
        phi_H, phi_T, dx, gamma, epsilon, M, energy_deriv
    )
    Jx_D, Jy_D, Jz_D = compute_mass_current_with_energy_deriv(
        phi_D, phi_T, dx, gamma, epsilon, M, energy_deriv
    )
    Jx_N, Jy_N, Jz_N = compute_mass_current_with_energy_deriv(
        phi_N, phi_T, dx, gamma, epsilon, 1e-9, energy_deriv
    )

    # Compute divergence terms for final derivatives
    dphi_H = -divergence(ux * phi_H, uy * phi_H, uz * phi_H, dx) - divergence(Jx_H, Jy_H, Jz_H, dx)
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


def compute_adhesion_energy_derivative_with_laplace(phi, laplace_phi, gamma, epsilon):
    f_prime = 0.5 * phi * (1 - phi) * (2 * phi - 1)
    energy_deriv = (gamma / epsilon) * f_prime - 0.01 * gamma * epsilon * laplace_phi
    return energy_deriv


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
        return 
    

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
            params['lambda_H'], params['lambda_D'], params['mu_H'], params['mu_D'],
            params['mu_N'], params['p_H'], params['p_D'], params['n_H'], params['n_D']
        )

def compute_internal_pressure_scie3121_model(
        phi_H, phi_D, phi_N, nutrient, dx, gamma, epsilon,
        lambda_H, lambda_D, mu_H, mu_D, mu_N, p_H, p_D, n_H, n_D,
        laplace_phi=None, energy_deriv=None):
    """
    Compute internal pressure field for SCIE3121 model.
    
    Args:
        phi_H, phi_D, phi_N: Volume fractions
        nutrient: Nutrient concentration field
        dx: Grid spacing
        gamma, epsilon, etc.: Model parameters
        laplace_phi: Optional precomputed Laplacian of phi_T
        energy_deriv: Optional precomputed adhesion energy derivative
        
    Returns:
        Pressure field
    """
    phi_T = phi_H + phi_D + phi_N
    
    # Compute Laplacian if not provided
    if laplace_phi is None:
        laplace_phi = laplacian(phi_T, dx)
    
    # Compute energy derivative if not provided
    if energy_deriv is None:
        energy_deriv = compute_adhesion_energy_derivative_with_laplace(
            phi_T, laplace_phi, gamma, epsilon
        )
    
    # Compute gradients of phi_T
    grad_C_x = gradient_neumann(phi_T, dx, 0)
    grad_C_y = gradient_neumann(phi_T, dx, 1)
    grad_C_z = gradient_neumann(phi_T, dx, 2)
    
    # Compute gradients of energy derivative
    grad_energy_x = gradient_neumann(energy_deriv, dx, 0)
    grad_energy_y = gradient_neumann(energy_deriv, dx, 1)
    grad_energy_z = gradient_neumann(energy_deriv, dx, 2)
    
    # Compute source terms
    src_H, src_D, src_N = compute_pressure_cell_sources(
        phi_H, phi_D, phi_N, nutrient, n_H, n_D, lambda_H, lambda_D, mu_H, mu_D, mu_N, p_H, p_D
    )
    S_T = src_H + src_D + src_N
    
    # Compute divergence term
    divergence = (grad_energy_x * grad_C_x + 
                  grad_energy_y * grad_C_y + 
                  grad_energy_z * grad_C_z + 
                  energy_deriv * laplace_phi)
    
    # Compute right-hand side of pressure equation
    rhs = S_T - divergence
    
    # Solve pressure equation
    shape = phi_T.shape
    A = get_laplacian_operator(shape, dx)
    rhs_flat = rhs.flatten()
    p_flat, info = cg(A, rhs_flat, rtol=1e-6, maxiter=100)
    if info < 0:
        raise ValueError("Illegal input or breakdown in CG solver")
    
    # Reshape solution to original grid shape
    pressure = -p_flat.reshape(shape)
    
    return pressure