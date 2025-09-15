"""
Natural Boundary Conditions Module

This module provides natural boundary condition implementations that allow
cells to flow naturally without artificial constraints that cause grid artifacts.

Author: Riley Jae McNamara
Date: 2025-02-19
"""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def apply_natural_gradient_boundaries(gx: np.ndarray, gy: np.ndarray, gz: np.ndarray, 
                                    boundary_width: int = 1):
    """
    Apply natural boundary conditions to gradient components.
    
    Instead of forcing zero gradients at boundaries, this allows natural
    gradients to develop while preventing numerical artifacts.
    
    Args:
        gx, gy, gz: Gradient components
        boundary_width: Width of boundary layer to treat
        
    Returns:
        gx, gy, gz with natural boundary conditions applied
    """
    nx, ny, nz = gx.shape
    
    # Instead of forcing zero gradients, use natural extrapolation
    # This allows cells to flow naturally while maintaining stability
    
    # X boundaries: use natural extrapolation from interior
    for i in range(boundary_width):
        # Left boundary: extrapolate from interior
        if i < nx - 1:
            gx[i, :, :] = gx[boundary_width, :, :]
        # Right boundary: extrapolate from interior
        if nx - 1 - i > 0:
            gx[nx-1-i, :, :] = gx[nx-1-boundary_width, :, :]
    
    # Y boundaries: use natural extrapolation from interior
    for j in range(boundary_width):
        # Front boundary: extrapolate from interior
        if j < ny - 1:
            gy[:, j, :] = gy[:, boundary_width, :]
        # Back boundary: extrapolate from interior
        if ny - 1 - j > 0:
            gy[:, ny-1-j, :] = gy[:, ny-1-boundary_width, :]
    
    # Z boundaries: use natural extrapolation from interior
    for k in range(boundary_width):
        # Bottom boundary: extrapolate from interior
        if k < nz - 1:
            gz[:, :, k] = gz[:, :, boundary_width]
        # Top boundary: extrapolate from interior
        if nz - 1 - k > 0:
            gz[:, :, nz-1-k] = gz[:, :, nz-1-boundary_width]
    
    return gx, gy, gz


@njit(cache=True, fastmath=True)
def apply_minimal_boundary_conditions(field: np.ndarray, boundary_width: int = 1):
    """
    Apply minimal boundary conditions that don't create artificial constraints.
    
    This allows natural flow while preventing numerical instabilities.
    
    Args:
        field: 3D field to apply boundary conditions to
        boundary_width: Width of boundary layer to treat
        
    Returns:
        field with minimal boundary conditions applied
    """
    nx, ny, nz = field.shape
    result = field.copy()
    
    # Apply minimal constraints - just ensure stability without forcing physics
    # X boundaries: gentle extrapolation
    for i in range(boundary_width):
        if i < nx - 1:
            result[i, :, :] = result[boundary_width, :, :]
        if nx - 1 - i > 0:
            result[nx-1-i, :, :] = result[nx-1-boundary_width, :, :]
    
    # Y boundaries: gentle extrapolation
    for j in range(boundary_width):
        if j < ny - 1:
            result[:, j, :] = result[:, boundary_width, :]
        if ny - 1 - j > 0:
            result[:, ny-1-j, :] = result[:, ny-1-boundary_width, :]
    
    # Z boundaries: gentle extrapolation
    for k in range(boundary_width):
        if k < nz - 1:
            result[:, :, k] = result[:, :, boundary_width]
        if nz - 1 - k > 0:
            result[:, :, nz-1-k] = result[:, :, nz-1-boundary_width]
    
    return result


@njit(cache=True, fastmath=True)
def apply_open_boundary_conditions(field: np.ndarray, boundary_width: int = 1):
    """
    Apply open boundary conditions that allow natural outflow.
    
    This is the most natural approach - cells can flow out of the domain
    without artificial constraints.
    
    Args:
        field: 3D field to apply boundary conditions to
        boundary_width: Width of boundary layer to treat
        
    Returns:
        field with open boundary conditions applied
    """
    nx, ny, nz = field.shape
    result = field.copy()
    
    # Open boundaries: allow natural flow out of domain
    # This is the most physically natural approach
    
    # X boundaries: allow outflow (no artificial constraints)
    for i in range(boundary_width):
        if i < nx - 1:
            # Left boundary: allow natural flow
            result[i, :, :] = result[boundary_width, :, :] * 0.9  # Slight damping to prevent instability
        if nx - 1 - i > 0:
            # Right boundary: allow natural flow
            result[nx-1-i, :, :] = result[nx-1-boundary_width, :, :] * 0.9
    
    # Y boundaries: allow outflow
    for j in range(boundary_width):
        if j < ny - 1:
            result[:, j, :] = result[:, boundary_width, :] * 0.9
        if ny - 1 - j > 0:
            result[:, ny-1-j, :] = result[:, ny-1-boundary_width, :] * 0.9
    
    # Z boundaries: allow outflow
    for k in range(boundary_width):
        if k < nz - 1:
            result[:, :, k] = result[:, :, boundary_width] * 0.9
        if nz - 1 - k > 0:
            result[:, :, nz-1-k] = result[:, :, nz-1-boundary_width] * 0.9
    
    return result


@njit(cache=True, fastmath=True)
def apply_controlled_boundary_conditions(field: np.ndarray, boundary_width: int = 1, 
                                       control_factor: float = 0.5):
    """
    Apply controlled boundary conditions with adjustable constraint strength.
    
    This allows you to control how much constraint to apply at boundaries,
    from completely open (0.0) to completely constrained (1.0).
    
    Args:
        field: 3D field to apply boundary conditions to
        boundary_width: Width of boundary layer to treat
        control_factor: Control strength (0.0 = open, 1.0 = constrained)
        
    Returns:
        field with controlled boundary conditions applied
    """
    nx, ny, nz = field.shape
    result = field.copy()
    
    # Controlled boundaries: adjustable constraint strength
    # control_factor = 0.0: completely open (natural flow)
    # control_factor = 1.0: completely constrained (no flux)
    # control_factor = 0.5: balanced approach
    
    # X boundaries: controlled constraint
    for i in range(boundary_width):
        if i < nx - 1:
            # Left boundary: controlled extrapolation
            interior_value = result[boundary_width, :, :]
            result[i, :, :] = interior_value * (1.0 - control_factor) + result[i, :, :] * control_factor
        if nx - 1 - i > 0:
            # Right boundary: controlled extrapolation
            interior_value = result[nx-1-boundary_width, :, :]
            result[nx-1-i, :, :] = interior_value * (1.0 - control_factor) + result[nx-1-i, :, :] * control_factor
    
    # Y boundaries: controlled constraint
    for j in range(boundary_width):
        if j < ny - 1:
            interior_value = result[:, boundary_width, :]
            result[:, j, :] = interior_value * (1.0 - control_factor) + result[:, j, :] * control_factor
        if ny - 1 - j > 0:
            interior_value = result[:, ny-1-boundary_width, :]
            result[:, ny-1-j, :] = interior_value * (1.0 - control_factor) + result[:, ny-1-j, :] * control_factor
    
    # Z boundaries: controlled constraint
    for k in range(boundary_width):
        if k < nz - 1:
            interior_value = result[:, :, boundary_width]
            result[:, :, k] = interior_value * (1.0 - control_factor) + result[:, :, k] * control_factor
        if nz - 1 - k > 0:
            interior_value = result[:, :, nz-1-boundary_width]
            result[:, :, nz-1-k] = interior_value * (1.0 - control_factor) + result[:, :, nz-1-k] * control_factor
    
    return result
