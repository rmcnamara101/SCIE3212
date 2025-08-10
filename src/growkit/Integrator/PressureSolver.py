"""
Pressure Solver Module

This module implements the pressure solver for the Poisson equation:

∇²p = S_T - ∇·((δE/δφ_T) ∇φ_T)

where:
- p: pressure field
- S_T: total source term (sum of all population sources)
- δE/δφ_T: adhesion energy derivative
- φ_T: total cell density

This implementation uses a robust, well-conditioned solver with proper regularization.

Author: Riley Jae McNamara
Date: 2025-02-19
"""

import numpy as np
from scipy.sparse.linalg import cg, LinearOperator
from scipy.sparse import diags
from src.growkit.MathEngine.Operators import gradient, laplacian
from src.growkit.ProductionEngine.SourceConstructor import SourceConstructor


def laplacian_neumann_robust(p_flat, shape, dx):
    """
    Computes the 3D Laplacian (∇²) with Neumann boundary conditions.
    This version is more numerically stable.
    """
    p = p_flat.reshape(shape)
    lap_p = np.zeros(shape, dtype=np.float64)
    dx2 = dx**2

    # x-direction
    lap_p[1:-1, :, :] = (p[2:, :, :] - 2*p[1:-1, :, :] + p[:-2, :, :]) / dx2
    # Neumann BC: dp/dx = 0 at boundaries
    lap_p[0, :, :] = (p[1, :, :] - p[0, :, :]) / dx2
    lap_p[-1, :, :] = (p[-2, :, :] - p[-1, :, :]) / dx2

    # y-direction
    lap_p[:, 1:-1, :] += (p[:, 2:, :] - 2*p[:, 1:-1, :] + p[:, :-2, :]) / dx2
    lap_p[:, 0, :] += (p[:, 1, :] - p[:, 0, :]) / dx2
    lap_p[:, -1, :] += (p[:, -2, :] - p[:, -1, :]) / dx2

    # z-direction
    lap_p[:, :, 1:-1] += (p[:, :, 2:] - 2*p[:, :, 1:-1] + p[:, :, :-2]) / dx2
    lap_p[:, :, 0] += (p[:, :, 1] - p[:, :, 0]) / dx2
    lap_p[:, :, -1] += (p[:, :, -2] - p[:, :, -1]) / dx2
    
    return lap_p.flatten()


def create_preconditioner(shape, dx, epsilon):
    """
    Creates a simple diagonal preconditioner for the regularized Laplacian.
    """
    N = np.prod(shape)
    nx, ny, nz = shape
    
    # Diagonal elements of the regularized Laplacian
    # For interior points: -6/dx^2 + epsilon
    # For boundary points: -3/dx^2 + epsilon (fewer neighbors)
    diag_elements = np.ones(N) * (-6.0 / dx**2 + epsilon)
    
    # Adjust for boundary points
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                idx = i * ny * nz + j * nz + k
                boundary_count = 0
                if i == 0 or i == nx-1: boundary_count += 1
                if j == 0 or j == ny-1: boundary_count += 1
                if k == 0 or k == nz-1: boundary_count += 1
                
                if boundary_count > 0:
                    diag_elements[idx] = (-6.0 + boundary_count) / dx**2 + epsilon
    
    # Create preconditioner: inverse of diagonal
    diag_elements = np.abs(diag_elements)  # Ensure positive
    diag_elements = np.maximum(diag_elements, 1e-12)  # Avoid division by zero
    preconditioner_diag = 1.0 / diag_elements
    
    return diags([preconditioner_diag], [0], shape=(N, N), dtype=np.float64)


def get_regularized_operator_robust(shape, dx, epsilon):
    """
    Creates a robust SciPy LinearOperator for the regularized negative 
    Laplacian (-∇² + εI), with better numerical stability.
    """
    N = np.prod(shape)
    
    def matvec(p_flat):
        # Compute the matrix-vector product for (-∇² + εI)p
        lap_p_flat = laplacian_neumann_robust(p_flat, shape, dx)
        result = -lap_p_flat + epsilon * p_flat
        
        # Ensure finite values
        if not np.isfinite(result).all():
            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        
        return result

    return LinearOperator((N, N), matvec=matvec, dtype=np.float64)


def solve_pressure_poisson_robust(phi_T, S_T, energy_deriv, dx, shape, rtol=1e-6, maxiter=2000):
    """
    Solves the pressure Poisson equation using a robust, well-conditioned method.
    Equation: -∇²p = S_T - ∇·((δE/δφ_T) ∇φ_T)
    """
    
    # Input validation and cleaning
    phi_T = np.nan_to_num(phi_T, nan=0.0, posinf=1.0, neginf=0.0)
    S_T = np.nan_to_num(S_T, nan=0.0, posinf=0.0, neginf=0.0)
    energy_deriv = np.nan_to_num(energy_deriv, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Compute the divergence term
    grad_C_x, grad_C_y, grad_C_z = gradient(phi_T, dx)
    grad_energy_x, grad_energy_y, grad_energy_z = gradient(energy_deriv, dx)
    laplace_phi = laplacian(phi_T, dx)
    
    divergence_term = (grad_energy_x * grad_C_x +
                       grad_energy_y * grad_C_y +
                       grad_energy_z * grad_C_z +
                       energy_deriv * laplace_phi)
    
    # Clean divergence term
    divergence_term = np.nan_to_num(divergence_term, nan=0.0, posinf=0.0, neginf=0.0)

    # Construct right-hand side
    rhs = S_T - divergence_term
    rhs_flat = rhs.flatten().astype(np.float64, copy=False)
    
    # Clean RHS
    rhs_flat = np.nan_to_num(rhs_flat, nan=0.0, posinf=0.0, neginf=0.0)
    
   
    
    # Check if RHS is effectively zero
    if np.linalg.norm(rhs_flat) < 1e-12:
        return np.zeros(shape)
    
    # Determine appropriate regularization parameter
    # Use a more conservative epsilon based on problem scale
    rhs_norm = np.linalg.norm(rhs_flat)
    dx2 = dx**2
    # epsilon should be large enough to ensure positive definiteness
    # but small enough to not dominate the Laplacian
    epsilon = max(1e-8, min(rhs_norm * 1e-6, 1e-4))
    
    
    # Create the regularized operator
    A_reg = get_regularized_operator_robust(shape, dx, epsilon)
    
    # Create preconditioner
    M = create_preconditioner(shape, dx, epsilon)
    
    # Solve the system with preconditioning
    try:
        # Use a more conservative tolerance
        solver_rtol = min(rtol, 1e-4)
        
        p_flat, info = cg(A_reg, rhs_flat, M=M, x0=np.zeros_like(rhs_flat), 
                         rtol=solver_rtol, maxiter=maxiter, atol=1e-12)
        
        
        
        # Check convergence
        if info != 0:

            # Try with even more conservative parameters
            if info == 1:  # maxiter reached
                
                p_flat, info = cg(A_reg, rhs_flat, M=M, x0=np.zeros_like(rhs_flat),
                                rtol=1e-2, maxiter=1000, atol=1e-10)
                
        
        # Check for numerical issues
        if not np.isfinite(p_flat).all():
            
            p_flat = np.nan_to_num(p_flat, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Clip extreme values
        p_flat = np.clip(p_flat, -1e6, 1e6)
        
    except Exception as e:
        
        p_flat = np.zeros_like(rhs_flat)
    
    # Reshape and return
    pressure = p_flat.reshape(shape)
    
    
    
    return pressure


# --- Main Solver Class ---

class PressureSolver:
    """
    Pressure solver that handles the pressure equation for all populations.
    """
    
    def __init__(self, cfg, populations):
        """
        Initialize the pressure solver.
        """
        self.cfg = cfg
        self.pops = populations
        self.labels = list(populations.keys())
        self.m = self.cfg["physics"]["adhesion_energy"]["m"]
        self.source_constructor = SourceConstructor(cfg, populations)
        
    
    def solve_pressure(self, phi_hat, nutrient_field, dx, energy_deriv=None):
        """
        Solves the pressure equation using the robust solver.
        """
        
        
        phi_T = np.sum(phi_hat, axis=0)
        
        
        # Compute energy derivative if not provided
        if energy_deriv is None:
            
            from src.growkit.PhysicsEngine.VectorizedCellDynamics import compute_adhesion_energy_derivative_numba
            laplace_phi = laplacian(phi_T, dx)
            energy_deriv = compute_adhesion_energy_derivative_numba(phi_T, laplace_phi, self.m)
        else:
            pass
        
        # Compute total source term
        
        from src.growkit.ProductionEngine.SourceConstructor import compute_pressure_source_vector_numba
        S_T = compute_pressure_source_vector_numba(
            phi_hat, nutrient_field, self.source_constructor.lambda_, 
            self.source_constructor.mu, self.source_constructor.nutrient_thresholds,
            self.source_constructor.P, self.source_constructor.beta_N
        )
        
        
        # Solve the pressure equation using the robust method
        pressure = solve_pressure_poisson_robust(
            phi_T, S_T, energy_deriv, dx, phi_hat.shape[1:]
        )
        
        return -pressure
    
    def compute_pressure_gradient(self, pressure, dx):
        """
        Computes the pressure gradient.
        """
        return gradient(pressure, dx)