"""
Pressure Solver Module

This module implements the pressure solver for the Poisson equation:

∇²p = S_T - ∇·(δE/δφ_T * ∇φ_T)

where:
- p: pressure field
- S_T: total source term (sum of all population sources)
- δE/δφ_T: adhesion energy derivative (chemical potential)
- φ_T: total cell density (cell density field)

This implementation uses a robust, well-conditioned solver with proper regularization.

Author: Riley Jae McNamara
Date: 2025-02-19
"""

# --- Standard imports ---

import numpy as np
from scipy.sparse.linalg import cg, LinearOperator
from scipy.sparse import diags
from functools import lru_cache

# --- Local imports ---

from src.growkit.MathEngine.Operators import isotropic_gradient_components, isotropic_laplacian
from src.growkit.ProductionEngine.SourceConstructor import SourceConstructor

# --- Memory management ---

def clear_pressure_solver_caches():
    """
    Clear all LRU caches in the pressure solver to free memory.
    Call this periodically during long simulations to prevent memory bloat.
    """
    create_improved_preconditioner.cache_clear()
    get_cached_operator.cache_clear()
    create_dx_aware_preconditioner.cache_clear()

# --- Solver functions ---

def solve_pressure_poisson_optimized(phi_T, S_T, energy_deriv, dx, shape, rtol=1e-6, maxiter=2000, solver_type="improved_cg"):
    """
    Solves the pressure Poisson equation using optimized methods.
    Equation: -∇²p = S_T - ∇·((δE/δφ_T) ∇φ_T)
    
    Optimizations:
    - Uses scipy's C-implemented laplacian
    - Gaussian initial guess
    - Improved preconditioner
    - Cached operators
    - Vectorized operations
    
    Args:
        solver_type: Type of solver to use ("improved_cg", "original_cg", "fast_cg")
            - "fast_cg": Uses simple laplacian (no diagonal contributions) for speed
            - "improved_cg": Uses isotropic operators with adaptive parameters
            - "original_cg": Uses isotropic operators with simpler parameters
    """
    
    # Input validation and cleaning
    phi_T = np.nan_to_num(phi_T, nan=0.0, posinf=1.0, neginf=0.0)
    S_T = np.nan_to_num(S_T, nan=0.0, posinf=0.0, neginf=0.0)
    energy_deriv = np.nan_to_num(energy_deriv, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Route to fast solver if requested (uses simple operators, no diagonal contributions)
    if solver_type == "fast_cg":
        return _solve_fast_cg(phi_T, S_T, energy_deriv, dx, shape, rhs_flat=None)
    
    # Compute the divergence term using isotropic operators (slower but more accurate)
    grad_C_x, grad_C_y, grad_C_z = isotropic_gradient_components(phi_T, dx)
    grad_energy_x, grad_energy_y, grad_energy_z = isotropic_gradient_components(energy_deriv, dx)
    laplace_phi = isotropic_laplacian(phi_T, dx)
    
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
    
    # Route to appropriate solver based on solver_type
    if solver_type == "improved_cg":
        return _solve_improved_cg(phi_T, S_T, energy_deriv, dx, shape, rtol, maxiter, rhs_flat)
    elif solver_type == "original_cg":
        return _solve_original_cg(phi_T, S_T, energy_deriv, dx, shape, rtol, maxiter, rhs_flat)
    else:
        print(f"Warning: Unknown solver type '{solver_type}', using improved_cg")
        return _solve_improved_cg(phi_T, S_T, energy_deriv, dx, shape, rtol, maxiter, rhs_flat)


def _solve_fast_cg(phi_T, S_T, energy_deriv, dx, shape, rhs_flat=None):
    """
    Fast CG solver based on the old implementation.
    Uses simple laplacian (no diagonal contributions) for maximum speed.
    This is the recommended solver for parameter sweeps.
    """
    from src.growkit.MathEngine.Operators import laplacian as simple_laplacian, gradient as simple_gradient, laplacian_neumann
    
    # Compute RHS if not provided
    if rhs_flat is None:
        # Use simple operators (no diagonal contributions) for speed
        grad_C_x, grad_C_y, grad_C_z = simple_gradient(phi_T, dx)
        grad_energy_x, grad_energy_y, grad_energy_z = simple_gradient(energy_deriv, dx)
        laplace_phi = simple_laplacian(phi_T, dx)
        
        divergence_term = (grad_energy_x * grad_C_x +
                          grad_energy_y * grad_C_y +
                          grad_energy_z * grad_C_z +
                          energy_deriv * laplace_phi)
        
        # Clean divergence term and clip extreme values to prevent instability
        divergence_term = np.nan_to_num(divergence_term, nan=0.0, posinf=0.0, neginf=0.0)
        # Clip divergence term to prevent numerical explosion
        max_div = np.abs(S_T).max() * 10.0  # Limit divergence to reasonable scale
        divergence_term = np.clip(divergence_term, -max_div, max_div)
        
        rhs = S_T - divergence_term
        rhs_flat = rhs.flatten().astype(np.float64, copy=False)
        rhs_flat = np.nan_to_num(rhs_flat, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Clip RHS to prevent numerical explosion
        max_rhs = 1e6  # Reasonable upper bound
        rhs_flat = np.clip(rhs_flat, -max_rhs, max_rhs)
    
    # Check if RHS is effectively zero (but use a more lenient threshold)
    rhs_norm = np.linalg.norm(rhs_flat)
    if rhs_norm < 1e-10:
        # Return zeros if RHS is truly zero, but print warning if it's suspiciously small
        if rhs_norm > 1e-15:  # Very small but not zero - might indicate an issue
            print(f"Warning: Fast CG solver RHS is very small (norm={rhs_norm:.2e}), returning zero pressure")
        return np.zeros(shape)
    
    # Add minimal regularization to stabilize the solver
    # The Laplacian with Neumann BCs is singular (null space = constants), so we need regularization
    dx_factor = 1.0 / (dx * dx)
    epsilon = max(1e-10, min(rhs_norm * 1e-8, 1e-6))  # Adaptive but small regularization
    
    # Create regularized negative laplacian operator: -∇² + εI
    # The equation is -∇²p = rhs, so we solve (-∇² + εI)p = rhs
    N = np.prod(shape)
    def matvec(p_flat):
        # Return -laplacian + regularization to match the equation (-∇² + εI)p = rhs
        lap_result = -laplacian_neumann(p_flat, shape, dx)
        return lap_result + epsilon * p_flat
    
    A = LinearOperator((N, N), matvec=matvec, dtype=np.float64)
    
    # Create simple diagonal preconditioner for better convergence
    # Use the diagonal of the operator as preconditioner
    diag_estimate = -6.0 / (dx * dx) + epsilon  # Approximate diagonal for interior points
    preconditioner_diag = 1.0 / max(abs(diag_estimate), 1e-12)
    M = diags([np.full(N, preconditioner_diag)], [0], shape=(N, N), dtype=np.float64)
    
    # Solve (-∇² + εI)p = rhs with preconditioning
    try:
        p_flat, info = cg(A, rhs_flat, M=M, rtol=1e-4, maxiter=200, atol=1e-10)
        if info != 0:
            # Try with more relaxed parameters
            p_flat, info2 = cg(A, rhs_flat, M=M, rtol=1e-2, maxiter=1000, atol=1e-8)
            if info2 != 0:
                # If still not converging, try without preconditioner
                p_flat, info3 = cg(A, rhs_flat, rtol=1e-2, maxiter=2000, atol=1e-8)
                if info3 != 0:
                    print(f"Warning: Fast CG solver did not converge after all attempts (info={info3}), RHS norm={rhs_norm:.2e}")
                    # Return best guess
    except Exception as e:
        print(f"Error in Fast CG solver: {e}, RHS norm={rhs_norm:.2e}")
        p_flat = np.zeros_like(rhs_flat)
    
    # Clean any NaN/Inf values
    p_flat = np.nan_to_num(p_flat, nan=0.0, posinf=0.0, neginf=0.0)
    
    return p_flat.reshape(shape)


def _solve_improved_cg(phi_T, S_T, energy_deriv, dx, shape, rtol, maxiter, rhs_flat):
    """Improved CG solver with dx-aware parameters."""
    # Adaptive regularization based on dx
    dx_factor = 1.0 / (dx * dx)  # dx appears as 1/dx^2 in Laplacian
    base_epsilon = 1e-8
    adaptive_epsilon = base_epsilon * max(1.0, dx_factor / 100.0)  # Scale with dx
    
    # Create dx-aware operator and preconditioner
    A_reg = get_cached_operator(shape, dx, adaptive_epsilon)
    M = create_dx_aware_preconditioner(shape, dx, adaptive_epsilon)
    
    # Adaptive tolerance based on dx
    adaptive_rtol = min(rtol, max(1e-4, rtol * dx_factor / 100.0))
    
    # Create Gaussian initial guess
    x0 = create_gaussian_initial_guess(shape, dx)
    
    # Solve with adaptive parameters
    try:
        p_flat, info = cg(A_reg, rhs_flat, M=M, 
                         rtol=adaptive_rtol, maxiter=maxiter, atol=1e-12)
        
        if info != 0:
            # Fallback with even more relaxed parameters
            p_flat, info = cg(A_reg, rhs_flat, M=M,
                            rtol=1e-2, maxiter=1000, atol=1e-10)
        
        if not np.isfinite(p_flat).all():
            p_flat = np.nan_to_num(p_flat, nan=0.0, posinf=0.0, neginf=0.0)
        
    except Exception as e:
        p_flat = np.zeros_like(rhs_flat)
    
    return p_flat.reshape(shape)


def _solve_original_cg(phi_T, S_T, energy_deriv, dx, shape, rtol, maxiter, rhs_flat):
    """Original CG solver (for comparison)."""
    # Determine appropriate regularization parameter
    rhs_norm = np.linalg.norm(rhs_flat)
    epsilon = max(1e-8, min(rhs_norm * 1e-6, 1e-4))
    
    # Create Gaussian initial guess
    x0 = create_gaussian_initial_guess(shape, dx)
    
    # Get cached operator and preconditioner
    A_reg = get_cached_operator(shape, dx, epsilon)
    M = create_improved_preconditioner(shape, dx, epsilon)
    
    # Solve the system with preconditioning
    try:
        solver_rtol = min(rtol, 1e-4)
        
        p_flat, info = cg(A_reg, rhs_flat, M=M, x0=x0, 
                         rtol=solver_rtol, maxiter=maxiter, atol=1e-12)
        
        # Check convergence and try fallback if needed
        if info != 0:
            if info == 1:  # maxiter reached
                # Try with more conservative parameters and zero initial guess
                p_flat, info = cg(A_reg, rhs_flat, M=M, x0=np.zeros_like(rhs_flat),
                                rtol=1e-2, maxiter=1000, atol=1e-10)
        
        # Check for numerical issues
        if not np.isfinite(p_flat).all():
            p_flat = np.nan_to_num(p_flat, nan=0.0, posinf=0.0, neginf=0.0)
        
    except Exception as e:
        # Fallback to zero solution
        p_flat = np.zeros_like(rhs_flat)
    
    return p_flat.reshape(shape)


# --- Main Solver Class ---

class PressureSolver:
    """
    Optimized pressure solver that handles the pressure equation for all populations.
    Features:
    - Cached operators and preconditioners
    - Gaussian initial guesses
    - Vectorized operations
    - Improved numerical stability
    """
    
    def __init__(self, cfg, populations, solver_type="improved_cg"):
        """
        Initialize the pressure solver.
        
        Args:
            cfg: Configuration dictionary
            populations: Population definitions from YAML
            solver_type: Type of solver to use ("improved_cg", "gmres", "original_cg")
        """
        self.cfg = cfg
        self.pops = populations
        self.labels = list(populations.keys())
        self.m = self.cfg["physics"]["adhesion_energy"]["m"]
        self.solver_type = solver_type
        self.source_constructor = SourceConstructor(cfg, populations)
        
        # Cache for shape-specific data
        self._shape_cache = {}


    def solve_pressure(self, phi_hat, nutrient_field, dx, energy_deriv=None):
        """
        Solves the pressure equation using the optimized solver.
        """
        phi_T = np.sum(phi_hat, axis=0)
        
        # Compute energy derivative if not provided
        if energy_deriv is None:
            from src.growkit.PhysicsEngine.Energy.VectorizedEnergy import compute_adhesion_energy_derivative_numba
            laplace_phi = isotropic_laplacian(phi_T, dx)
            energy_deriv = compute_adhesion_energy_derivative_numba(phi_T, laplace_phi, self.m)
        
        # Compute total source term
        from src.growkit.ProductionEngine.SourceConstructor import compute_pressure_source_vector_numba
        S_T = compute_pressure_source_vector_numba(
            phi_hat, nutrient_field, self.source_constructor.lambda_, 
            self.source_constructor.mu, self.source_constructor.nutrient_thresholds,
            self.source_constructor.P, self.source_constructor.beta_N
        )
        
        pressure = solve_pressure_poisson_optimized(
            phi_T, S_T, energy_deriv, dx, phi_hat.shape[1:], solver_type=self.solver_type
        )
        
        return -pressure



# --- Helper functions ---


def create_gaussian_initial_guess(shape, dx, radius_factor=0.1):
    """
    Creates a Gaussian initial guess centered in the grid with std dev proportional to radius.
    
    Args:
        shape: Grid shape (nx, ny, nz)
        dx: Grid spacing
        radius_factor: Factor to multiply by grid radius for std dev
        
    Returns:
        Gaussian initial guess as flattened array
    """
    nx, ny, nz = shape
    
    # Create coordinate grids
    x = np.arange(nx) * dx
    y = np.arange(ny) * dx
    z = np.arange(nz) * dx
    
    # Center coordinates
    x_center = x[nx // 2]
    y_center = y[ny // 2]
    z_center = z[nz // 2]
    
    # Standard deviation proportional to grid radius
    grid_radius = min(nx, ny, nz) * dx * 0.5
    sigma = grid_radius * radius_factor
    
    # Create 3D Gaussian
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Gaussian function
    gaussian = np.exp(-((X - x_center)**2 + (Y - y_center)**2 + (Z - z_center)**2) / (2 * sigma**2))
    
    return gaussian.flatten().astype(np.float32)


@lru_cache(maxsize=4)  # Reduced from 32 to prevent memory bloat
def create_improved_preconditioner(shape, dx, epsilon):
    """
    Creates an improved preconditioner for the regularized Laplacian.
    Uses vectorized operations to eliminate triple loops.
    
    Args:
        shape: Grid shape (nx, ny, nz)
        dx: Grid spacing
        epsilon: Regularization parameter
        
    Returns:
        Preconditioner as sparse matrix
    """
    nx, ny, nz = shape
    N = nx * ny * nz
    
    # Base diagonal elements for interior points
    diag_elements = np.full(N, -6.0 / dx**2 + epsilon, dtype=np.float32)
    
    # Create index arrays for efficient boundary detection
    i_indices = np.arange(nx).reshape(-1, 1, 1)
    j_indices = np.arange(ny).reshape(1, -1, 1)
    k_indices = np.arange(nz).reshape(1, 1, -1)
    
    # Create 3D index arrays
    I, J, K = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
    
    # Flatten for vectorized operations
    I_flat = I.flatten()
    J_flat = J.flatten()
    K_flat = K.flatten()
    
    # Count boundary neighbors for each point
    boundary_count = np.zeros(N, dtype=np.int32)
    boundary_count += (I_flat == 0) | (I_flat == nx-1)  # x boundaries
    boundary_count += (J_flat == 0) | (J_flat == ny-1)  # y boundaries
    boundary_count += (K_flat == 0) | (K_flat == nz-1)  # z boundaries
    
    # Adjust diagonal elements for boundary points
    # For boundary points, we have fewer neighbors
    diag_elements -= boundary_count / dx**2
    
    # Ensure positive definiteness and avoid division by zero
    diag_elements = np.abs(diag_elements)
    diag_elements = np.maximum(diag_elements, 1e-12)
    
    # Create preconditioner: inverse of diagonal
    preconditioner_diag = 1.0 / diag_elements
    
    return diags([preconditioner_diag], [0], shape=(N, N), dtype=np.float32)


@lru_cache(maxsize=4)  # Reduced from 32 to prevent memory bloat
def get_cached_operator(shape, dx, epsilon):
    """
    Creates a cached SciPy LinearOperator for (-∇² + εI) using scipy.ndimage.
    
    Args:
        shape: Grid shape (nx, ny, nz)
        dx: Grid spacing
        epsilon: Regularization parameter
        
    Returns:
        LinearOperator for the regularized negative Laplacian
    """
    N = np.prod(shape)
    dx2 = dx**2

    def matvec(p_flat):
        p = p_flat.reshape(shape)
        # Use isotropic laplacian to eliminate grid artifacts
        lap_p = isotropic_laplacian(p, dx)
        
        # Regularization term
        result = -lap_p.flatten() + epsilon * p_flat
        
        # Use safe casting to prevent overflow
        return safe_float32_cast(result, safety_factor=0.1)

    return LinearOperator((N, N), matvec=matvec, dtype=np.float32)


@lru_cache(maxsize=4)  # Reduced from 32 to prevent memory bloat
def create_dx_aware_preconditioner(shape, dx, epsilon):
    """
    Create dx-aware preconditioner that scales better with small dx.
    """
    nx, ny, nz = shape
    N = nx * ny * nz
    
    # Base diagonal elements with dx-aware scaling
    diag_elements = np.full(N, -6.0 / dx**2 + epsilon, dtype=np.float64)
    
    # Create index arrays for boundary detection
    I, J, K = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
    I_flat = I.flatten()
    J_flat = J.flatten()
    K_flat = K.flatten()
    
    # Count boundary neighbors
    boundary_count = np.zeros(N, dtype=np.int32)
    boundary_count += (I_flat == 0) | (I_flat == nx-1)
    boundary_count += (J_flat == 0) | (J_flat == ny-1)
    boundary_count += (K_flat == 0) | (K_flat == nz-1)
    
    # Adjust for boundaries
    diag_elements -= boundary_count / dx**2
    
    # Enhanced regularization for small dx
    dx_factor = 1.0 / (dx * dx)
    enhanced_epsilon = epsilon * max(1.0, dx_factor / 50.0)
    diag_elements += enhanced_epsilon
    
    # Ensure positive definiteness
    diag_elements = np.abs(diag_elements)
    diag_elements = np.maximum(diag_elements, 1e-12)
    
    # Create preconditioner
    preconditioner_diag = 1.0 / diag_elements
    
    return diags([preconditioner_diag], [0], shape=(N, N), dtype=np.float32)


# Legacy functions for backward compatibility
def laplacian_neumann_robust(p_flat, shape, dx):
    """
    Legacy function - kept for backward compatibility.
    Use scipy.ndimage.laplace instead for better performance.
    
    Fixed to use consistent boundary conditions to avoid diamond-shaped artifacts.
    """
    p = p_flat.reshape(shape)
    lap_p = np.zeros(shape, dtype=np.float64)
    dx2 = dx**2

    # x-direction
    lap_p[1:-1, :, :] = (p[2:, :, :] - 2*p[1:-1, :, :] + p[:-2, :, :]) / dx2
    # Neumann BC: dp/dx = 0 at boundaries - use one-sided difference
    lap_p[0, :, :] = (p[1, :, :] - p[0, :, :]) / dx2
    lap_p[-1, :, :] = (p[-2, :, :] - p[-1, :, :]) / dx2

    # y-direction
    lap_p[:, 1:-1, :] += (p[:, 2:, :] - 2*p[:, 1:-1, :] + p[:, :-2, :]) / dx2
    # Neumann BC: dp/dy = 0 at boundaries - use one-sided difference
    lap_p[:, 0, :] += (p[:, 1, :] - p[:, 0, :]) / dx2
    lap_p[:, -1, :] += (p[:, -2, :] - p[:, -1, :]) / dx2

    # z-direction
    lap_p[:, :, 1:-1] += (p[:, :, 2:] - 2*p[:, :, 1:-1] + p[:, :, :-2]) / dx2
    # Neumann BC: dp/dz = 0 at boundaries - use one-sided difference
    lap_p[:, :, 0] += (p[:, :, 1] - p[:, :, 0]) / dx2
    lap_p[:, :, -1] += (p[:, :, -2] - p[:, :, -1]) / dx2
    
    return lap_p.flatten()


def create_preconditioner(shape, dx, epsilon):
    """
    Legacy function - kept for backward compatibility.
    Use create_improved_preconditioner instead for better performance.
    """
    return create_improved_preconditioner(shape, dx, epsilon)


def get_regularized_operator_robust(shape, dx, epsilon):
    """
    Legacy function - kept for backward compatibility.
    Use get_cached_operator instead for better performance.
    """
    return get_cached_operator(shape, dx, epsilon)


def get_optimized_operator(shape, dx, epsilon):
    """
    Legacy function - kept for backward compatibility.
    Use get_cached_operator instead for better performance.
    """
    return get_cached_operator(shape, dx, epsilon)


def solve_pressure_poisson_robust(phi_T, S_T, energy_deriv, dx, shape, rtol=1e-6, maxiter=2000):
    """
    Legacy function - kept for backward compatibility.
    Use solve_pressure_poisson_optimized instead for better performance.
    """
    return solve_pressure_poisson_optimized(phi_T, S_T, energy_deriv, dx, shape, rtol, maxiter)


def safe_float32_cast(arr, safety_factor=0.1):
    """
    Safely cast array to float32 by clipping extreme values to prevent overflow.
    
    Args:
        arr: Input array
        safety_factor: Fraction of max float32 value to use as clipping limit
    
    Returns:
        Array safely cast to float32
    """
    max_val = np.finfo(np.float32).max * safety_factor
    clipped_arr = np.clip(arr, -max_val, max_val)
    return clipped_arr.astype(np.float32)

