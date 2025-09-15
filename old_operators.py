import numpy as np
import numba as nb
from scipy.sparse.linalg import cg, LinearOperator

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

@nb.njit
def gradient_isotropic(field, dx, axis):
    """
    Compute the gradient of a field using an isotropic approach to reduce directional bias.
    This implementation uses a weighted average of gradients in multiple directions.
    
    Args:
        field: The field to compute the gradient of
        dx: Grid spacing
        axis: Axis along which to compute the gradient (0, 1, or 2)
    Returns:
        Gradient array
    """
    grad = np.zeros_like(field)
    nx, ny, nz = field.shape
    
    if axis == 0:  # x-direction
        # Main x-direction gradient (70% weight)
        grad[1:-1, :, :] = 0.7 * (field[2:, :, :] - field[:-2, :, :]) / (2*dx)
        
        # Add diagonal contributions (30% weight)
        # We'll use a simplified approach that's more efficient
        # For each diagonal direction, we'll compute the gradient and add it with appropriate weight
        # This is more efficient than nested loops
        
        # Diagonal in xy plane
        grad[1:-1, 1:-1, :] += 0.075 * (field[2:, 2:, :] - field[:-2, :-2, :]) / (2*dx*np.sqrt(2))
        grad[1:-1, 1:-1, :] += 0.075 * (field[2:, :-2, :] - field[:-2, 2:, :]) / (2*dx*np.sqrt(2))
        
        # Diagonal in xz plane
        grad[1:-1, :, 1:-1] += 0.075 * (field[2:, :, 2:] - field[:-2, :, :-2]) / (2*dx*np.sqrt(2))
        grad[1:-1, :, 1:-1] += 0.075 * (field[2:, :, :-2] - field[:-2, :, 2:]) / (2*dx*np.sqrt(2))
        
        # Neumann boundary conditions
        grad[0, :, :] = 0
        grad[-1, :, :] = 0
        
    elif axis == 1:  # y-direction
        # Main y-direction gradient (70% weight)
        grad[:, 1:-1, :] = 0.7 * (field[:, 2:, :] - field[:, :-2, :]) / (2*dx)
        
        # Add diagonal contributions (30% weight)
        # Diagonal in xy plane
        grad[1:-1, 1:-1, :] += 0.075 * (field[2:, 2:, :] - field[:-2, :-2, :]) / (2*dx*np.sqrt(2))
        grad[1:-1, 1:-1, :] += 0.075 * (field[:-2, 2:, :] - field[2:, :-2, :]) / (2*dx*np.sqrt(2))
        
        # Diagonal in yz plane
        grad[:, 1:-1, 1:-1] += 0.075 * (field[:, 2:, 2:] - field[:, :-2, :-2]) / (2*dx*np.sqrt(2))
        grad[:, 1:-1, 1:-1] += 0.075 * (field[:, 2:, :-2] - field[:, :-2, 2:]) / (2*dx*np.sqrt(2))
        
        # Neumann boundary conditions
        grad[:, 0, :] = 0
        grad[:, -1, :] = 0
        
    elif axis == 2:  # z-direction
        # Main z-direction gradient (70% weight)
        grad[:, :, 1:-1] = 0.7 * (field[:, :, 2:] - field[:, :, :-2]) / (2*dx)
        
        # Add diagonal contributions (30% weight)
        # Diagonal in xz plane
        grad[1:-1, :, 1:-1] += 0.075 * (field[2:, :, 2:] - field[:-2, :, :-2]) / (2*dx*np.sqrt(2))
        grad[1:-1, :, 1:-1] += 0.075 * (field[:-2, :, 2:] - field[2:, :, :-2]) / (2*dx*np.sqrt(2))
        
        # Diagonal in yz plane
        grad[:, 1:-1, 1:-1] += 0.075 * (field[:, 2:, 2:] - field[:, :-2, :-2]) / (2*dx*np.sqrt(2))
        grad[:, 1:-1, 1:-1] += 0.075 * (field[:, :-2, 2:] - field[:, 2:, :-2]) / (2*dx*np.sqrt(2))
        
        # Neumann boundary conditions
        grad[:, :, 0] = 0
        grad[:, :, -1] = 0
    
    return grad

@nb.njit
def divergence(ux, uy, uz, dx):
    """
    Compute the divergence of a vector field with Neumann boundary conditions.
    ∇·u = ∂ux/∂x + ∂uy/∂y + ∂uz/∂z
    """
    return (gradient_neumann(ux, dx, 0) + 
            gradient_neumann(uy, dx, 1) + 
            gradient_neumann(uz, dx, 2))

@nb.njit
def divergence_isotropic(ux, uy, uz, dx):
    """
    Compute the divergence of a vector field using isotropic gradients.
    ∇·u = ∂ux/∂x + ∂uy/∂y + ∂uz/∂z
    """
    return (gradient_isotropic(ux, dx, 0) + 
            gradient_isotropic(uy, dx, 1) + 
            gradient_isotropic(uz, dx, 2))


def laplacian_neumann(p_flat, shape, dx):
    """
    Compute the 3D Laplacian with Neumann boundary conditions and improved isotropy.
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
    nx, ny, nz = shape
    dx2 = dx * dx
    
    # Interior points - use a more isotropic approach with vectorized operations
    # Standard Cartesian directions (70% weight)
    lap_p[1:-1, 1:-1, 1:-1] = 0.7 * (
        (p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1] + 
         p[1:-1, 2:, 1:-1] + p[1:-1, :-2, 1:-1] + 
         p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2] - 
         6*p[1:-1, 1:-1, 1:-1]) / dx2
    )
    
    # Diagonal directions (30% weight)
    # We'll use a simplified approach that's more efficient
    # For each diagonal direction, we'll compute the contribution and add it with appropriate weight
    
    # Diagonal in xy plane
    lap_p[1:-1, 1:-1, 1:-1] += 0.075 * (
        (p[2:, 2:, 1:-1] + p[:-2, :-2, 1:-1] - 2*p[1:-1, 1:-1, 1:-1]) / (dx2 * 2)
    )
    lap_p[1:-1, 1:-1, 1:-1] += 0.075 * (
        (p[2:, :-2, 1:-1] + p[:-2, 2:, 1:-1] - 2*p[1:-1, 1:-1, 1:-1]) / (dx2 * 2)
    )
    
    # Diagonal in xz plane
    lap_p[1:-1, 1:-1, 1:-1] += 0.075 * (
        (p[2:, 1:-1, 2:] + p[:-2, 1:-1, :-2] - 2*p[1:-1, 1:-1, 1:-1]) / (dx2 * 2)
    )
    lap_p[1:-1, 1:-1, 1:-1] += 0.075 * (
        (p[2:, 1:-1, :-2] + p[:-2, 1:-1, 2:] - 2*p[1:-1, 1:-1, 1:-1]) / (dx2 * 2)
    )
    
    # Diagonal in yz plane
    lap_p[1:-1, 1:-1, 1:-1] += 0.075 * (
        (p[1:-1, 2:, 2:] + p[1:-1, :-2, :-2] - 2*p[1:-1, 1:-1, 1:-1]) / (dx2 * 2)
    )
    lap_p[1:-1, 1:-1, 1:-1] += 0.075 * (
        (p[1:-1, 2:, :-2] + p[1:-1, :-2, 2:] - 2*p[1:-1, 1:-1, 1:-1]) / (dx2 * 2)
    )
    
    # X boundaries
    lap_p[0, :, :] = 2*(p[1, :, :] - p[0, :, :]) / dx2
    lap_p[-1, :, :] = 2*(p[-2, :, :] - p[-1, :, :]) / dx2
    
    # Y boundaries
    lap_p[:, 0, :] += 2*(p[:, 1, :] - p[:, 0, :]) / dx2
    lap_p[:, -1, :] += 2*(p[:, -2, :] - p[:, -1, :]) / dx2
    
    # Z boundaries
    lap_p[:, :, 0] += 2*(p[:, :, 1] - p[:, :, 0]) / dx2
    lap_p[:, :, -1] += 2*(p[:, :, -2] - p[:, :, -1]) / dx2
    
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


@nb.njit
def compute_pressure_cell_sources(phi_H, phi_D, phi_N, nutrient, n_H, n_D, lambda_H, lambda_D, mu_H, mu_D, p_H, p_D, mu_N):
    """
    Compute source terms for pressure calculation.
    """
    # Smooth gating sharpness
    k = 5.0  # Higher = sharper transition

    # Smooth gating functions for growth and death
    G_H = 0.5 * (1 + np.tanh(k * (nutrient - n_H)))  # Growth gate for healthy cells
    G_D = 0.5 * (1 + np.tanh(k * (nutrient - n_D)))  # Growth gate for diseased cells

    D_H = 0.5 * (1 - np.tanh(k * (nutrient - n_H)))  # Death gate for healthy cells
    D_D = 0.5 * (1 - np.tanh(k * (nutrient - n_D)))  # Death gate for diseased cells

    # Source terms
    src_H = lambda_H * nutrient * phi_H * (2 * p_H - 1) * G_H - mu_H * phi_H * D_H 
    src_D = 2 * lambda_H * nutrient * (1 - p_H) * phi_H * G_H + lambda_D * nutrient * phi_D * G_D - mu_D * phi_D * D_D
    src_N = -mu_N * phi_N
    
    return (src_H, src_D, src_N) 

@nb.njit
def compute_pressure_components(phi_H, phi_D, phi_N, nutrient, n_H, n_D, lambda_H, lambda_D, mu_H, mu_D, p_H, p_D, mu_N, energy_deriv, grad_C_x, grad_C_y, grad_C_z, laplace_phi, dx):
    """Compute the components needed for pressure calculation that can be Numba-accelerated"""
    phi_T = phi_H + phi_D + phi_N
    # Compute pressure using precomputed values
    src_H, src_D, src_N = compute_pressure_cell_sources(
        phi_H, phi_D, phi_N, nutrient, n_H, n_D, lambda_H, lambda_D, mu_H, mu_D, mu_N, p_H, p_D
    )
    S_T = src_H + src_D + src_N
    
    # Use precomputed values for divergence term
    grad_energy_x = gradient_isotropic(energy_deriv, dx, 0)
    grad_energy_y = gradient_isotropic(energy_deriv, dx, 1)
    grad_energy_z = gradient_isotropic(energy_deriv, dx, 2)
    
    divergence = grad_energy_x * grad_C_x + grad_energy_y * grad_C_y + grad_energy_z * grad_C_z + energy_deriv * laplace_phi
    rhs = S_T - divergence
    
    return rhs, phi_T.shape

def solve_pressure_poisson(rhs, shape, dx):
    """Solve the Poisson equation for pressure using scipy's conjugate gradient solver"""
    A = get_laplacian_operator(shape, dx)
    rhs_flat = rhs.flatten()
    
    try:
        p_flat, info = cg(A, rhs_flat, rtol=1e-3, maxiter=100)
        if info != 0:
            #print(f"CG solver did not converge (info={info}), using fallback")
            p_flat = np.zeros_like(rhs_flat)
    except Exception as e:
        print(f"CG solver error: {e}")
        p_flat = np.zeros_like(rhs_flat)
    
    return -p_flat.reshape(shape)

def compute_pressure(phi_H, phi_D, phi_N, nutrient, n_H, n_D, lambda_H, lambda_D, mu_H, mu_D, p_H, p_D, mu_N, energy_deriv, grad_C_x, grad_C_y, grad_C_z, laplace_phi, dx):
    """Main pressure computation function that combines Numba and non-Numba parts"""
    # Get components using Numba-accelerated function
    rhs, shape = compute_pressure_components(
        phi_H, phi_D, phi_N, nutrient, n_H, n_D, lambda_H, lambda_D, 
        mu_H, mu_D, p_H, p_D, mu_N, energy_deriv, 
        grad_C_x, grad_C_y, grad_C_z, laplace_phi, dx
    )
    
    # Solve the Poisson equation using scipy
    return solve_pressure_poisson(rhs, shape, dx)

@nb.njit
def gradient_magnitude_isotropic(field, dx):
    """
    Compute the gradient magnitude of a field using an isotropic approach in a single pass.
    This is more efficient than computing gradients in three directions separately.
    
    Args:
        field: The field to compute the gradient magnitude of
        dx: Grid spacing
    Returns:
        Gradient magnitude array
    """
    nx, ny, nz = field.shape
    grad_mag = np.zeros_like(field)
    
    # Main directional gradients (70% weight)
    # x-direction
    grad_x = np.zeros_like(field)
    grad_x[1:-1, :, :] = 0.7 * (field[2:, :, :] - field[:-2, :, :]) / (2*dx)
    
    # y-direction
    grad_y = np.zeros_like(field)
    grad_y[:, 1:-1, :] = 0.7 * (field[:, 2:, :] - field[:, :-2, :]) / (2*dx)
    
    # z-direction
    grad_z = np.zeros_like(field)
    grad_z[:, :, 1:-1] = 0.7 * (field[:, :, 2:] - field[:, :, :-2]) / (2*dx)
    
    # Add diagonal contributions (30% weight)
    # Diagonal in xy plane
    grad_xy = np.zeros_like(field)
    grad_xy[1:-1, 1:-1, :] = 0.075 * (field[2:, 2:, :] - field[:-2, :-2, :]) / (2*dx*np.sqrt(2))
    grad_xy[1:-1, 1:-1, :] += 0.075 * (field[2:, :-2, :] - field[:-2, 2:, :]) / (2*dx*np.sqrt(2))
    
    # Diagonal in xz plane
    grad_xz = np.zeros_like(field)
    grad_xz[1:-1, :, 1:-1] = 0.075 * (field[2:, :, 2:] - field[:-2, :, :-2]) / (2*dx*np.sqrt(2))
    grad_xz[1:-1, :, 1:-1] += 0.075 * (field[2:, :, :-2] - field[:-2, :, 2:]) / (2*dx*np.sqrt(2))
    
    # Diagonal in yz plane
    grad_yz = np.zeros_like(field)
    grad_yz[:, 1:-1, 1:-1] = 0.075 * (field[:, 2:, 2:] - field[:, :-2, :-2]) / (2*dx*np.sqrt(2))
    grad_yz[:, 1:-1, 1:-1] += 0.075 * (field[:, :-2, 2:] - field[:, 2:, :-2]) / (2*dx*np.sqrt(2))
    
    # Combine all gradients
    grad_x += grad_xy + grad_xz
    grad_y += grad_xy + grad_yz
    grad_z += grad_xz + grad_yz
    
    # Calculate gradient magnitude
    grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
    
    # Apply Neumann boundary conditions
    grad_mag[0, :, :] = 0
    grad_mag[-1, :, :] = 0
    grad_mag[:, 0, :] = 0
    grad_mag[:, -1, :] = 0
    grad_mag[:, :, 0] = 0
    grad_mag[:, :, -1] = 0
    
    return grad_mag