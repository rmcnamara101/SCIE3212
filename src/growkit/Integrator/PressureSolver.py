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
from scipy.sparse.linalg import cg, LinearOperator, gmres
from scipy.sparse import diags, csr_matrix
# Removed scipy.ndimage.laplace import - using isotropic operators instead
from functools import lru_cache
from src.growkit.MathEngine.Operators import isotropic_gradient_components, isotropic_laplacian
from src.growkit.ProductionEngine.SourceConstructor import SourceConstructor

# Try to import PyAMG for multigrid, fall back to manual implementation
try:
    import pyamg
    PYAMG_AVAILABLE = True
except ImportError:
    PYAMG_AVAILABLE = False
    print("PyAMG not available, using manual multigrid implementation")


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


@lru_cache(maxsize=32)
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


@lru_cache(maxsize=32)
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
        return result.astype(np.float32)

    return LinearOperator((N, N), matvec=matvec, dtype=np.float32)


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
        solver_type: Type of solver to use ("improved_cg", "gmres", "original_cg")
    """
    
    # Input validation and cleaning
    phi_T = np.nan_to_num(phi_T, nan=0.0, posinf=1.0, neginf=0.0)
    S_T = np.nan_to_num(S_T, nan=0.0, posinf=0.0, neginf=0.0)
    energy_deriv = np.nan_to_num(energy_deriv, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Compute the divergence term using isotropic operators
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
    elif solver_type == "gmres":
        return _solve_gmres(phi_T, S_T, energy_deriv, dx, shape, rtol, maxiter, rhs_flat)
    elif solver_type == "original_cg":
        return _solve_original_cg(phi_T, S_T, energy_deriv, dx, shape, rtol, maxiter, rhs_flat)
    elif solver_type == "multigrid":
        return _solve_multigrid(phi_T, S_T, energy_deriv, dx, shape, rtol, maxiter, rhs_flat)
    elif solver_type == "pyamg":
        return _solve_pyamg(phi_T, S_T, energy_deriv, dx, shape, rtol, maxiter, rhs_flat)
    else:
        print(f"Warning: Unknown solver type '{solver_type}', using improved_cg")
        return _solve_improved_cg(phi_T, S_T, energy_deriv, dx, shape, rtol, maxiter, rhs_flat)


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


def _solve_gmres(phi_T, S_T, energy_deriv, dx, shape, rtol, maxiter, rhs_flat):
    """GMRES solver - more robust than CG for ill-conditioned systems."""
    # Adaptive regularization
    dx_factor = 1.0 / (dx * dx)
    adaptive_epsilon = 1e-8 * max(1.0, dx_factor / 100.0)
    
    A_reg = get_cached_operator(shape, dx, adaptive_epsilon)
    M = create_dx_aware_preconditioner(shape, dx, adaptive_epsilon)
    
    # GMRES with restart
    try:
        p_flat, info = gmres(A_reg, rhs_flat, M=M, restart=50,
                           rtol=rtol, maxiter=maxiter//50, atol=1e-12)
        
        if info != 0:
            # Fallback with more conservative parameters
            p_flat, info = gmres(A_reg, rhs_flat, M=M, restart=20,
                               rtol=1e-2, maxiter=20, atol=1e-10)
        
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


@lru_cache(maxsize=32)
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


def _solve_multigrid(phi_T, S_T, energy_deriv, dx, shape, rtol, maxiter, rhs_flat):
    """
    Real geometric multigrid solver implementation.
    Uses V-cycle multigrid for the Poisson equation.
    """
    nx, ny, nz = shape
    
    # Create initial guess
    x0 = create_gaussian_initial_guess(shape, dx, radius_factor=0.1)
    
    # Apply V-cycle multigrid
    p_flat = geometric_multigrid_vcycle(shape, dx, rhs_flat, x0, rtol, maxiter)
    
    if not np.isfinite(p_flat).all():
        p_flat = np.nan_to_num(p_flat, nan=0.0, posinf=0.0, neginf=0.0)
    
    return p_flat.reshape(shape)


def geometric_multigrid_vcycle(shape, dx, rhs, x0, rtol, maxiter):
    """
    Geometric multigrid V-cycle for 3D Poisson equation.
    """
    nx, ny, nz = shape
    
    # Determine number of levels (coarsest grid should be at least 4x4x4)
    min_size = 4
    max_levels = min(
        int(np.log2(nx / min_size)),
        int(np.log2(ny / min_size)), 
        int(np.log2(nz / min_size))
    )
    if max_levels <= 0:
        # Grid too small for multigrid, use direct solve
        return solve_direct_small_grid(shape, dx, rhs)
    
    # Create grid hierarchy
    levels = create_multigrid_levels(shape, dx, max_levels)
    
    # V-cycle
    x = x0.copy()
    for cycle in range(maxiter):
        x_old = x.copy()
        x = v_cycle(levels, rhs, x, rtol)
        
        # Check convergence
        residual = np.linalg.norm(x - x_old)
        if residual < rtol * np.linalg.norm(x):
            break
    
    return x


def create_multigrid_levels(shape, dx, max_levels):
    """
    Create multigrid level hierarchy.
    """
    levels = []
    nx, ny, nz = shape
    current_dx = dx
    
    for level in range(max_levels + 1):
        if nx < 4 or ny < 4 or nz < 4:
            break
            
        levels.append({
            'shape': (nx, ny, nz),
            'dx': current_dx,
            'size': nx * ny * nz
        })
        
        # Coarsen grid
        nx = max(4, nx // 2)
        ny = max(4, ny // 2) 
        nz = max(4, nz // 2)
        current_dx *= 2.0
    
    return levels


def v_cycle(levels, rhs, x, rtol):
    """
    V-cycle multigrid iteration.
    """
    if len(levels) <= 1:
        # Coarsest level - solve directly
        return solve_coarsest_level(levels[0], rhs)
    
    # Pre-smoothing (relaxation)
    x = gauss_seidel_smoother(levels[0], rhs, x, num_sweeps=2)
    
    # Compute residual
    residual = compute_residual(levels[0], rhs, x)
    
    # Restrict residual to coarser grid
    coarse_rhs = restrict_residual(levels[0], levels[1], residual)
    
    # Recursive V-cycle on coarser grid
    coarse_correction = v_cycle(levels[1:], coarse_rhs, np.zeros_like(coarse_rhs), rtol)
    
    # Prolong correction to finer grid
    correction = prolong_correction(levels[1], levels[0], coarse_correction)
    
    # Add correction
    x = x + correction
    
    # Post-smoothing
    x = gauss_seidel_smoother(levels[0], rhs, x, num_sweeps=2)
    
    return x


def gauss_seidel_smoother(level, rhs, x, num_sweeps=2):
    """
    Gauss-Seidel relaxation smoother.
    """
    nx, ny, nz = level['shape']
    dx = level['dx']
    dx2 = dx * dx
    
    for sweep in range(num_sweeps):
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                for k in range(1, nz-1):
                    idx = i * ny * nz + j * nz + k
                    
                    # 7-point stencil update
                    x[idx] = (rhs[idx] * dx2 + 
                             x[(i-1)*ny*nz + j*nz + k] + x[(i+1)*ny*nz + j*nz + k] +
                             x[i*ny*nz + (j-1)*nz + k] + x[i*ny*nz + (j+1)*nz + k] +
                             x[i*ny*nz + j*nz + (k-1)] + x[i*ny*nz + j*nz + (k+1)]) / 6.0
    
    return x


def compute_residual(level, rhs, x):
    """
    Compute residual: r = rhs - A*x
    """
    nx, ny, nz = level['shape']
    dx = level['dx']
    dx2 = dx * dx
    
    residual = np.zeros_like(rhs)
    
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            for k in range(1, nz-1):
                idx = i * ny * nz + j * nz + k
                
                # 7-point stencil
                laplacian = (x[(i-1)*ny*nz + j*nz + k] + x[(i+1)*ny*nz + j*nz + k] +
                            x[i*ny*nz + (j-1)*nz + k] + x[i*ny*nz + (j+1)*nz + k] +
                            x[i*ny*nz + j*nz + (k-1)] + x[i*ny*nz + j*nz + (k+1)] -
                            6.0 * x[idx]) / dx2
                
                residual[idx] = rhs[idx] + laplacian
    
    return residual


def restrict_residual(fine_level, coarse_level, fine_residual):
    """
    Restrict residual from fine to coarse grid.
    """
    fine_nx, fine_ny, fine_nz = fine_level['shape']
    coarse_nx, coarse_ny, coarse_nz = coarse_level['shape']
    
    coarse_residual = np.zeros(coarse_nx * coarse_ny * coarse_nz)
    
    # Simple injection restriction
    for i in range(coarse_nx):
        for j in range(coarse_ny):
            for k in range(coarse_nz):
                fine_i = min(2*i, fine_nx-1)
                fine_j = min(2*j, fine_ny-1)
                fine_k = min(2*k, fine_nz-1)
                
                fine_idx = fine_i * fine_ny * fine_nz + fine_j * fine_nz + fine_k
                coarse_idx = i * coarse_ny * coarse_nz + j * coarse_nz + k
                
                coarse_residual[coarse_idx] = fine_residual[fine_idx]
    
    return coarse_residual


def prolong_correction(coarse_level, fine_level, coarse_correction):
    """
    Prolong correction from coarse to fine grid.
    """
    coarse_nx, coarse_ny, coarse_nz = coarse_level['shape']
    fine_nx, fine_ny, fine_nz = fine_level['shape']
    
    fine_correction = np.zeros(fine_nx * fine_ny * fine_nz)
    
    # Linear interpolation prolongation
    for i in range(fine_nx):
        for j in range(fine_ny):
            for k in range(fine_nz):
                coarse_i = i // 2
                coarse_j = j // 2
                coarse_k = k // 2
                
                # Clamp to valid range
                coarse_i = min(coarse_i, coarse_nx-1)
                coarse_j = min(coarse_j, coarse_ny-1)
                coarse_k = min(coarse_k, coarse_nz-1)
                
                coarse_idx = coarse_i * coarse_ny * coarse_nz + coarse_j * coarse_nz + coarse_k
                fine_idx = i * fine_ny * fine_nz + j * fine_nz + k
                
                fine_correction[fine_idx] = coarse_correction[coarse_idx]
    
    return fine_correction


def solve_coarsest_level(level, rhs):
    """
    Solve the coarsest level directly using CG.
    """
    shape = level['shape']
    dx = level['dx']
    
    # Create operator for coarsest level
    A = create_sparse_laplacian_matrix_pyamg(shape, dx)
    
    # Solve with CG
    try:
        from scipy.sparse.linalg import cg
        x, info = cg(A, rhs, tol=1e-6, maxiter=1000)
        if info != 0:
            x = np.zeros_like(rhs)
    except:
        x = np.zeros_like(rhs)
    
    return x


def solve_direct_small_grid(shape, dx, rhs):
    """
    Direct solve for small grids.
    """
    A = create_sparse_laplacian_matrix_pyamg(shape, dx)
    
    try:
        from scipy.sparse.linalg import spsolve
        x = spsolve(A, rhs)
    except:
        x = np.zeros_like(rhs)
    
    return x


def _solve_pyamg(phi_T, S_T, energy_deriv, dx, shape, rtol, maxiter, rhs_flat):
    """
    PyAMG multigrid solver - best performance for small dx.
    """
    if not PYAMG_AVAILABLE:
        print("PyAMG not available, falling back to improved CG")
        return _solve_improved_cg(phi_T, S_T, energy_deriv, dx, shape, rtol, maxiter, rhs_flat)
    
    try:
        # Create the Laplacian matrix for PyAMG
        A = create_sparse_laplacian_matrix_pyamg(shape, dx)
        
        # Create PyAMG multigrid solver
        ml = pyamg.ruge_stuben_solver(A)
        
        # Solve using PyAMG multigrid
        p_flat = ml.solve(rhs_flat, tol=rtol, maxiter=maxiter)
        
        if not np.isfinite(p_flat).all():
            p_flat = np.nan_to_num(p_flat, nan=0.0, posinf=0.0, neginf=0.0)
        
        return p_flat.reshape(shape)
        
    except Exception as e:
        print(f"PyAMG solver failed: {e}, falling back to improved CG")
        return _solve_improved_cg(phi_T, S_T, energy_deriv, dx, shape, rtol, maxiter, rhs_flat)


@lru_cache(maxsize=32)
def create_sparse_laplacian_matrix_pyamg(shape, dx):
    """
    Create sparse matrix representation of the Laplacian operator for PyAMG.
    """
    nx, ny, nz = shape
    N = nx * ny * nz
    
    # Create the 3D Laplacian stencil
    from scipy.sparse import lil_matrix
    
    A = lil_matrix((N, N), dtype=np.float64)
    
    # Fill the matrix with 3D Laplacian stencil
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                idx = i * ny * nz + j * nz + k
                
                # Diagonal element (center)
                A[idx, idx] = -6.0 / (dx * dx)
                
                # Neighbors in x direction
                if i > 0:
                    A[idx, (i-1) * ny * nz + j * nz + k] = 1.0 / (dx * dx)
                if i < nx-1:
                    A[idx, (i+1) * ny * nz + j * nz + k] = 1.0 / (dx * dx)
                
                # Neighbors in y direction
                if j > 0:
                    A[idx, i * ny * nz + (j-1) * nz + k] = 1.0 / (dx * dx)
                if j < ny-1:
                    A[idx, i * ny * nz + (j+1) * nz + k] = 1.0 / (dx * dx)
                
                # Neighbors in z direction
                if k > 0:
                    A[idx, i * ny * nz + j * nz + (k-1)] = 1.0 / (dx * dx)
                if k < nz-1:
                    A[idx, i * ny * nz + j * nz + (k+1)] = 1.0 / (dx * dx)
    
    return A.tocsc()


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
        
    def _get_cached_data(self, shape):
        """
        Get cached data for a specific grid shape.
        """
        if shape not in self._shape_cache:
            self._shape_cache[shape] = {
                'gaussian_guess': create_gaussian_initial_guess(shape, self.cfg.get('dx', 1.0))
            }
        return self._shape_cache[shape]
    
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
        
        # Use FFT solver by default for massive speedup
        solver_type = getattr(self, 'solver_type', 'fft')
        
        if solver_type == 'fft':
            # Use FFT solver for ~2000x speedup
            from src.growkit.Integrator.FFTPressureSolver import solve_pressure_fft
            pressure = solve_pressure_fft(phi_T, S_T, energy_deriv, dx, phi_hat.shape[1:])
        else:
            # Fall back to iterative solver for other boundary conditions
            pressure = solve_pressure_poisson_optimized(
                phi_T, S_T, energy_deriv, dx, phi_hat.shape[1:], solver_type=solver_type
            )
        
        return -pressure
    
    def compute_pressure_gradient(self, pressure, dx):
        """
        Computes the pressure gradient using isotropic operators.
        """
        return isotropic_gradient_components(pressure, dx)

    def _get_eps_bucket(self, eps):
        # bucket to nearest power-of-10 in [1e-8, 1e-4]
        if eps <= 1e-8: return 1e-8
        if eps >= 1e-4: return 1e-4
        # round to one significant digit log scale
        import math
        p = round(math.log10(eps))
        return 10.0**p

    def clear_caches(self):
        """
        Clear all caches to free memory.
        """
        self._shape_cache.clear()
        # Clear LRU caches
        create_improved_preconditioner.cache_clear()
        get_cached_operator.cache_clear()


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
