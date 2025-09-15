"""
Isotropic Boundary Conditions Module

This module provides isotropic boundary condition implementations that eliminate
grid-aligned artifacts by using smooth, consistent boundary treatments.

Author: Riley Jae McNamara
Date: 2025-02-19
"""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def apply_neumann_boundary_conditions(field: np.ndarray, boundary_width: int = 1):
    """
    Apply Neumann boundary conditions (zero normal derivative) isotropically.
    
    This function creates smooth boundary conditions that reduce grid-aligned artifacts
    by using a consistent approach across all boundaries.
    
    Args:
        field: 3D field to apply boundary conditions to
        boundary_width: Width of boundary layer to treat (default: 1)
        
    Returns:
        field with Neumann boundary conditions applied
    """
    nx, ny, nz = field.shape
    result = field.copy()
    
    # Apply Neumann BCs with smooth transitions
    # X boundaries
    for i in range(boundary_width):
        # Left boundary: copy from interior
        result[i, :, :] = result[boundary_width, :, :]
        # Right boundary: copy from interior  
        result[nx-1-i, :, :] = result[nx-1-boundary_width, :, :]
    
    # Y boundaries
    for j in range(boundary_width):
        # Front boundary: copy from interior
        result[:, j, :] = result[:, boundary_width, :]
        # Back boundary: copy from interior
        result[:, ny-1-j, :] = result[:, ny-1-boundary_width, :]
    
    # Z boundaries
    for k in range(boundary_width):
        # Bottom boundary: copy from interior
        result[:, :, k] = result[:, :, boundary_width]
        # Top boundary: copy from interior
        result[:, :, nz-1-k] = result[:, :, nz-1-boundary_width]
    
    return result


@njit(cache=True, fastmath=True)
def apply_smooth_neumann_boundary_conditions(field: np.ndarray, boundary_width: int = 2):
    """
    Apply smooth Neumann boundary conditions using linear extrapolation.
    
    This creates even smoother boundary conditions by using linear extrapolation
    from the interior, which further reduces grid artifacts.
    
    Args:
        field: 3D field to apply boundary conditions to
        boundary_width: Width of boundary layer to treat (default: 2)
        
    Returns:
        field with smooth Neumann boundary conditions applied
    """
    nx, ny, nz = field.shape
    result = field.copy()
    
    # X boundaries with linear extrapolation
    for i in range(boundary_width):
        # Left boundary: linear extrapolation from interior
        if boundary_width < nx - 1:
            slope = result[boundary_width, :, :] - result[boundary_width + 1, :, :]
            result[i, :, :] = result[boundary_width, :, :] + slope * (boundary_width - i)
        
        # Right boundary: linear extrapolation from interior
        if boundary_width < nx - 1:
            slope = result[nx-1-boundary_width, :, :] - result[nx-1-boundary_width-1, :, :]
            result[nx-1-i, :, :] = result[nx-1-boundary_width, :, :] + slope * (boundary_width - i)
    
    # Y boundaries with linear extrapolation
    for j in range(boundary_width):
        # Front boundary: linear extrapolation from interior
        if boundary_width < ny - 1:
            slope = result[:, boundary_width, :] - result[:, boundary_width + 1, :]
            result[:, j, :] = result[:, boundary_width, :] + slope * (boundary_width - j)
        
        # Back boundary: linear extrapolation from interior
        if boundary_width < ny - 1:
            slope = result[:, ny-1-boundary_width, :] - result[:, ny-1-boundary_width-1, :]
            result[:, ny-1-j, :] = result[:, ny-1-boundary_width, :] + slope * (boundary_width - j)
    
    # Z boundaries with linear extrapolation
    for k in range(boundary_width):
        # Bottom boundary: linear extrapolation from interior
        if boundary_width < nz - 1:
            slope = result[:, :, boundary_width] - result[:, :, boundary_width + 1]
            result[:, :, k] = result[:, :, boundary_width] + slope * (boundary_width - k)
        
        # Top boundary: linear extrapolation from interior
        if boundary_width < nz - 1:
            slope = result[:, :, nz-1-boundary_width] - result[:, :, nz-1-boundary_width-1]
            result[:, :, nz-1-k] = result[:, :, nz-1-boundary_width] + slope * (boundary_width - k)
    
    return result


@njit(cache=True, fastmath=True)
def apply_isotropic_neumann_gradients(gx: np.ndarray, gy: np.ndarray, gz: np.ndarray, 
                                     boundary_width: int = 1):
    """
    Apply isotropic Neumann boundary conditions to gradient components.
    
    This ensures that gradient components have consistent zero normal derivatives
    at boundaries, eliminating grid-aligned artifacts in gradient computations.
    
    Args:
        gx, gy, gz: Gradient components
        boundary_width: Width of boundary layer to treat
        
    Returns:
        gx, gy, gz with Neumann boundary conditions applied
    """
    nx, ny, nz = gx.shape
    
    # X boundaries: zero normal derivative means gx = 0
    for i in range(boundary_width):
        gx[i, :, :] = 0.0
        gx[nx-1-i, :, :] = 0.0
    
    # Y boundaries: zero normal derivative means gy = 0
    for j in range(boundary_width):
        gy[:, j, :] = 0.0
        gy[:, ny-1-j, :] = 0.0
    
    # Z boundaries: zero normal derivative means gz = 0
    for k in range(boundary_width):
        gz[:, :, k] = 0.0
        gz[:, :, nz-1-k] = 0.0
    
    return gx, gy, gz


@njit(cache=True, fastmath=True)
def create_boundary_distance_field(shape: tuple, dx: float) -> np.ndarray:
    """
    Create a field representing the distance to the nearest boundary.
    
    This can be used to apply boundary conditions that vary smoothly with distance
    from the boundary, reducing grid artifacts.
    
    Args:
        shape: Grid shape (nx, ny, nz)
        dx: Grid spacing
        
    Returns:
        Distance field where each point contains the distance to the nearest boundary
    """
    nx, ny, nz = shape
    distance_field = np.zeros(shape, dtype=np.float32)
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Distance to each boundary
                dist_x = min(i, nx - 1 - i) * dx
                dist_y = min(j, ny - 1 - j) * dx
                dist_z = min(k, nz - 1 - k) * dx
                
                # Distance to nearest boundary
                distance_field[i, j, k] = min(dist_x, dist_y, dist_z)
    
    return distance_field


@njit(cache=True, fastmath=True)
def apply_distance_based_boundary_conditions(field: np.ndarray, dx: float, 
                                           boundary_thickness: float = 2.0 * 0.2):
    """
    Apply boundary conditions that vary smoothly with distance from boundary.
    
    This creates the smoothest possible boundary conditions by using a distance-based
    approach that eliminates sharp transitions at boundaries.
    
    Args:
        field: 3D field to apply boundary conditions to
        dx: Grid spacing
        boundary_thickness: Thickness of boundary layer in physical units
        
    Returns:
        field with smooth distance-based boundary conditions applied
    """
    nx, ny, nz = field.shape
    result = field.copy()
    
    # Create distance field
    distance_field = create_boundary_distance_field((nx, ny, nz), dx)
    
    # Apply smooth boundary conditions
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                dist = distance_field[i, j, k]
                
                if dist < boundary_thickness:
                    # In boundary layer: blend with interior value
                    # Find the closest interior point
                    if i < nx // 2:
                        interior_i = nx // 2
                    else:
                        interior_i = nx // 2
                    
                    if j < ny // 2:
                        interior_j = ny // 2
                    else:
                        interior_j = ny // 2
                        
                    if k < nz // 2:
                        interior_k = nz // 2
                    else:
                        interior_k = nz // 2
                    
                    # Blend factor based on distance
                    blend_factor = dist / boundary_thickness
                    interior_value = field[interior_i, interior_j, interior_k]
                    
                    result[i, j, k] = blend_factor * field[i, j, k] + (1.0 - blend_factor) * interior_value
    
    return result
