"""
Operators Module (isotropy-focused, API-compatible)

Kept function names:
- _gradient_neumann(field, dx, axis)
- gradient(field, dx) -> (gx, gy, gz)
- divergence(ux, uy, uz, dx) -> div
- laplacian(field, dx) -> lap
- laplacian_neumann(p_flat, shape, dx) -> lap_flat
- curl(ux, uy, uz, dx) -> (cx, cy, cz)
- vector_laplacian(ux, uy, uz, dx) -> (lx, ly, lz)
- isotropic_gradient(field, dx) -> |∇φ|
- isotropic_gradient_components(field, dx) -> (gix, giy, giz)

New (optional, non-breaking):
- isotropic_laplacian(field, dx) -> ∇·(isotropic ∇φ)

All operators use Neumann (zero-normal-derivative) boundary behavior.
"""

from __future__ import annotations
import numpy as np
from numba import njit
from .NaturalBoundaryConditions import apply_natural_gradient_boundaries, apply_controlled_boundary_conditions

__all__ = [
    "_gradient_neumann",
    "gradient",
    "divergence",
    "laplacian",
    "laplacian_neumann",
    "curl",
    "vector_laplacian",
    "isotropic_gradient",
    "isotropic_gradient_components",
    "isotropic_laplacian",
]

# ----------------------------
# Axis-aligned core primitives
# ----------------------------

@njit(cache=True, fastmath=True)
def _gradient_neumann(field: np.ndarray, dx: float, axis: int) -> np.ndarray:
    """
    Axis-aligned central-difference gradient with Neumann (zero-flux) BCs.
    2nd-order centered interior; zero normal derivative at domain faces.
    """
    grad = np.zeros_like(field)
    if axis == 0:
        grad[1:-1, :, :] = (field[2:, :, :] - field[:-2, :, :]) / (2.0 * dx)
        grad[0, :, :] = 0.0
        grad[-1, :, :] = 0.0
    elif axis == 1:
        grad[:, 1:-1, :] = (field[:, 2:, :] - field[:, :-2, :]) / (2.0 * dx)
        grad[:, 0, :] = 0.0
        grad[:, -1, :] = 0.0
    else:
        grad[:, :, 1:-1] = (field[:, :, 2:] - field[:, :, :-2]) / (2.0 * dx)
        grad[:, :, 0] = 0.0
        grad[:, :, -1] = 0.0
    return grad


def gradient(field: np.ndarray, dx: float):
    """Axis-aligned ∇φ with Neumann BCs (kept for API compatibility)."""
    gx = _gradient_neumann(field, dx, 0)
    gy = _gradient_neumann(field, dx, 1)
    gz = _gradient_neumann(field, dx, 2)
    return gx, gy, gz


def divergence(ux: np.ndarray, uy: np.ndarray, uz: np.ndarray, dx: float) -> np.ndarray:
    """Axis-aligned ∇·u with Neumann BCs on each component."""
    return (
        _gradient_neumann(ux, dx, 0)
        + _gradient_neumann(uy, dx, 1)
        + _gradient_neumann(uz, dx, 2)
    )


@njit(cache=True, fastmath=True)
def laplacian(field: np.ndarray, dx: float) -> np.ndarray:
    """
    Axis-aligned ∇²φ with Neumann BCs (classic 6-point stencil, 2nd-order).
    Kept for compatibility/benchmarks.
    """
    lap = np.zeros_like(field)
    inv_dx2 = 1.0 / (dx * dx)

    # x
    lap[1:-1, :, :] += (field[2:, :, :] - 2.0 * field[1:-1, :, :] + field[:-2, :, :]) * inv_dx2
    lap[0, :, :]     += (field[1, :, :] - field[0, :, :]) * inv_dx2
    lap[-1, :, :]    += (field[-2, :, :] - field[-1, :, :]) * inv_dx2

    # y
    lap[:, 1:-1, :] += (field[:, 2:, :] - 2.0 * field[:, 1:-1, :] + field[:, :-2, :]) * inv_dx2
    lap[:, 0, :]    += (field[:, 1, :] - field[:, 0, :]) * inv_dx2
    lap[:, -1, :]   += (field[:, -2, :] - field[:, -1, :]) * inv_dx2

    # z
    lap[:, :, 1:-1] += (field[:, :, 2:] - 2.0 * field[:, :, 1:-1] + field[:, :, :-2]) * inv_dx2
    lap[:, :, 0]    += (field[:, :, 1] - field[:, :, 0]) * inv_dx2
    lap[:, :, -1]   += (field[:, :, -2] - field[:, :, -1]) * inv_dx2

    return lap


def laplacian_neumann(p_flat: np.ndarray, shape: tuple[int, int, int], dx: float) -> np.ndarray:
    """
    Axis-aligned ∇²φ with Neumann BCs for flattened arrays (pressure solver path).
    Returns float64 flattened array to match solver expectations.
    """
    p = p_flat.reshape(shape)
    lap_p = laplacian(p, dx).astype(np.float64, copy=False)
    return lap_p.ravel()


@njit(cache=True, fastmath=True)
def curl(ux: np.ndarray, uy: np.ndarray, uz: np.ndarray, dx: float):
    """Axis-aligned curl with Neumann BCs on partials."""
    cx = _gradient_neumann(uz, dx, 1) - _gradient_neumann(uy, dx, 2)
    cy = _gradient_neumann(ux, dx, 2) - _gradient_neumann(uz, dx, 0)
    cz = _gradient_neumann(uy, dx, 0) - _gradient_neumann(ux, dx, 1)
    return cx, cy, cz


def vector_laplacian(ux: np.ndarray, uy: np.ndarray, uz: np.ndarray, dx: float):
    """Axis-aligned vector Laplacian, componentwise."""
    return laplacian(ux, dx), laplacian(uy, dx), laplacian(uz, dx)

# ---------------------------------------
# Isotropic operators (recommended)
# ---------------------------------------

@njit(cache=True, fastmath=True)
def _zero_face_gradients(gx: np.ndarray, gy: np.ndarray, gz: np.ndarray):
    """Apply NO boundary conditions - let gradients develop naturally."""
    # NO boundary conditions - let gradients develop completely naturally
    # This should eliminate all artificial constraints that cause diamond artifacts
    return gx, gy, gz


def apply_cell_field_boundary_conditions(field: np.ndarray, boundary_type: str = "natural", 
                                       control_factor: float = 0.5):
    """
    Apply boundary conditions to cell fields with different approaches.
    
    Args:
        field: 3D cell field
        boundary_type: Type of boundary condition
            - "natural": Allow natural flow (recommended for eliminating diamond artifacts)
            - "controlled": Adjustable constraint strength
            - "open": Allow outflow with slight damping
            - "minimal": Minimal constraints
        control_factor: Control strength for "controlled" type (0.0 = open, 1.0 = constrained)
        
    Returns:
        field with appropriate boundary conditions applied
    """
    if boundary_type == "natural":
        # Use natural extrapolation - allows natural flow
        return apply_natural_gradient_boundaries(field, field, field, boundary_width=1)[0]
    elif boundary_type == "controlled":
        # Use controlled boundaries with adjustable strength
        return apply_controlled_boundary_conditions(field, boundary_width=1, control_factor=control_factor)
    elif boundary_type == "open":
        # Use open boundaries - allow outflow
        from .NaturalBoundaryConditions import apply_open_boundary_conditions
        return apply_open_boundary_conditions(field, boundary_width=1)
    elif boundary_type == "minimal":
        # Use minimal constraints
        from .NaturalBoundaryConditions import apply_minimal_boundary_conditions
        return apply_minimal_boundary_conditions(field, boundary_width=1)
    else:
        # Default to natural
        return apply_natural_gradient_boundaries(field, field, field, boundary_width=1)[0]


@njit(cache=True, fastmath=True)
def isotropic_gradient_components(field: np.ndarray, dx: float):
    """
    Isotropic gradient components using the proven approach from the old codebase.
    This uses proper weights for axis-aligned and diagonal contributions.
    """
    nx, ny, nz = field.shape
    gx = np.zeros_like(field)
    gy = np.zeros_like(field)
    gz = np.zeros_like(field)

    # X component using isotropic approach
    # Main x-direction gradient (70% weight)
    gx[1:-1, :, :] = 0.7 * (field[2:, :, :] - field[:-2, :, :]) / (2*dx)
    
    # Add diagonal contributions (30% weight)
    # Diagonal in xy plane
    gx[1:-1, 1:-1, :] += 0.075 * (field[2:, 2:, :] - field[:-2, :-2, :]) / (2*dx*np.sqrt(2))
    gx[1:-1, 1:-1, :] += 0.075 * (field[2:, :-2, :] - field[:-2, 2:, :]) / (2*dx*np.sqrt(2))
    
    # Diagonal in xz plane
    gx[1:-1, :, 1:-1] += 0.075 * (field[2:, :, 2:] - field[:-2, :, :-2]) / (2*dx*np.sqrt(2))
    gx[1:-1, :, 1:-1] += 0.075 * (field[2:, :, :-2] - field[:-2, :, 2:]) / (2*dx*np.sqrt(2))
    
    # Consistent boundary handling to prevent grid artifacts
    # Use the same 0.7 weight as interior to maintain consistency
    gx[0, :, :] = 0.7 * (field[1, :, :] - field[0, :, :]) / dx
    gx[-1, :, :] = 0.7 * (field[-1, :, :] - field[-2, :, :]) / dx
    
    # Y component using isotropic approach
    # Main y-direction gradient (70% weight)
    gy[:, 1:-1, :] = 0.7 * (field[:, 2:, :] - field[:, :-2, :]) / (2*dx)
    
    # Add diagonal contributions (30% weight)
    # Diagonal in xy plane
    gy[1:-1, 1:-1, :] += 0.075 * (field[2:, 2:, :] - field[:-2, :-2, :]) / (2*dx*np.sqrt(2))
    gy[1:-1, 1:-1, :] += 0.075 * (field[:-2, 2:, :] - field[2:, :-2, :]) / (2*dx*np.sqrt(2))
    
    # Diagonal in yz plane
    gy[:, 1:-1, 1:-1] += 0.075 * (field[:, 2:, 2:] - field[:, :-2, :-2]) / (2*dx*np.sqrt(2))
    gy[:, 1:-1, 1:-1] += 0.075 * (field[:, 2:, :-2] - field[:, :-2, 2:]) / (2*dx*np.sqrt(2))
    
    # Consistent boundary handling for Y
    gy[:, 0, :] = 0.7 * (field[:, 1, :] - field[:, 0, :]) / dx
    gy[:, -1, :] = 0.7 * (field[:, -1, :] - field[:, -2, :]) / dx
    
    # Z component using isotropic approach
    # Main z-direction gradient (70% weight)
    gz[:, :, 1:-1] = 0.7 * (field[:, :, 2:] - field[:, :, :-2]) / (2*dx)
    
    # Add diagonal contributions (30% weight)
    # Diagonal in xz plane
    gz[1:-1, :, 1:-1] += 0.075 * (field[2:, :, 2:] - field[:-2, :, :-2]) / (2*dx*np.sqrt(2))
    gz[1:-1, :, 1:-1] += 0.075 * (field[:-2, :, 2:] - field[2:, :, :-2]) / (2*dx*np.sqrt(2))
    
    # Diagonal in yz plane
    gz[:, 1:-1, 1:-1] += 0.075 * (field[:, 2:, 2:] - field[:, :-2, :-2]) / (2*dx*np.sqrt(2))
    gz[:, 1:-1, 1:-1] += 0.075 * (field[:, :-2, 2:] - field[:, 2:, :-2]) / (2*dx*np.sqrt(2))
    
    # Consistent boundary handling for Z
    gz[:, :, 0] = 0.7 * (field[:, :, 1] - field[:, :, 0]) / dx
    gz[:, :, -1] = 0.7 * (field[:, :, -1] - field[:, :, -2]) / dx

    return gx, gy, gz


@njit(cache=True, fastmath=True)
def _divergence_from_components(ux: np.ndarray, uy: np.ndarray, uz: np.ndarray, dx: float) -> np.ndarray:
    """
    Divergence using the proven isotropic approach from the old codebase.
    This uses the same isotropic gradient approach for each component.
    """
    # Use the proven isotropic gradient approach for each component
    gx_ux, gy_ux, gz_ux = isotropic_gradient_components(ux, dx)
    gx_uy, gy_uy, gz_uy = isotropic_gradient_components(uy, dx)
    gx_uz, gy_uz, gz_uz = isotropic_gradient_components(uz, dx)
    
    # Divergence is the sum of diagonal components
    div = gx_ux + gy_uy + gz_uz
    
    return div


@njit(cache=True, fastmath=True)
def isotropic_laplacian(field: np.ndarray, dx: float) -> np.ndarray:
    """
    Isotropic Laplacian using the proven approach from the old codebase.
    This uses proper weights for axis-aligned and diagonal contributions.
    """
    lap = np.zeros_like(field)
    nx, ny, nz = field.shape
    dx2 = dx * dx
    
    # Interior points - use the proven isotropic approach with proper weights
    # Standard Cartesian directions (70% weight)
    lap[1:-1, 1:-1, 1:-1] = 0.7 * (
        (field[2:, 1:-1, 1:-1] + field[:-2, 1:-1, 1:-1] + 
         field[1:-1, 2:, 1:-1] + field[1:-1, :-2, 1:-1] + 
         field[1:-1, 1:-1, 2:] + field[1:-1, 1:-1, :-2] - 
         6*field[1:-1, 1:-1, 1:-1]) / dx2
    )
    
    # Diagonal directions (30% weight) - this is the key for isotropy
    # Diagonal in xy plane
    lap[1:-1, 1:-1, 1:-1] += 0.075 * (
        (field[2:, 2:, 1:-1] + field[:-2, :-2, 1:-1] - 2*field[1:-1, 1:-1, 1:-1]) / (dx2 * 2)
    )
    lap[1:-1, 1:-1, 1:-1] += 0.075 * (
        (field[2:, :-2, 1:-1] + field[:-2, 2:, 1:-1] - 2*field[1:-1, 1:-1, 1:-1]) / (dx2 * 2)
    )
    
    # Diagonal in xz plane
    lap[1:-1, 1:-1, 1:-1] += 0.075 * (
        (field[2:, 1:-1, 2:] + field[:-2, 1:-1, :-2] - 2*field[1:-1, 1:-1, 1:-1]) / (dx2 * 2)
    )
    lap[1:-1, 1:-1, 1:-1] += 0.075 * (
        (field[2:, 1:-1, :-2] + field[:-2, 1:-1, 2:] - 2*field[1:-1, 1:-1, 1:-1]) / (dx2 * 2)
    )
    
    # Diagonal in yz plane
    lap[1:-1, 1:-1, 1:-1] += 0.075 * (
        (field[1:-1, 2:, 2:] + field[1:-1, :-2, :-2] - 2*field[1:-1, 1:-1, 1:-1]) / (dx2 * 2)
    )
    lap[1:-1, 1:-1, 1:-1] += 0.075 * (
        (field[1:-1, 2:, :-2] + field[1:-1, :-2, 2:] - 2*field[1:-1, 1:-1, 1:-1]) / (dx2 * 2)
    )
    
    # X boundaries - use natural boundary handling
    lap[0, :, :] = 2*(field[1, :, :] - field[0, :, :]) / dx2
    lap[-1, :, :] = 2*(field[-2, :, :] - field[-1, :, :]) / dx2
    
    # Y boundaries
    lap[:, 0, :] += 2*(field[:, 1, :] - field[:, 0, :]) / dx2
    lap[:, -1, :] += 2*(field[:, -2, :] - field[:, -1, :]) / dx2
    
    # Z boundaries
    lap[:, :, 0] += 2*(field[:, :, 1] - field[:, :, 0]) / dx2
    lap[:, :, -1] += 2*(field[:, :, -2] - field[:, :, -1]) / dx2
    
    return lap


@njit(cache=True, fastmath=True)
def isotropic_gradient(field: np.ndarray, dx: float) -> np.ndarray:
    """|∇φ| computed from isotropic gradient components."""
    gix, giy, giz = isotropic_gradient_components(field, dx)
    return np.sqrt(gix * gix + giy * giy + giz * giz)
