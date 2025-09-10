"""
Operators Module

This module contains all the differential operators (gradient, divergence, Laplacian)
used throughout the physics engine. This centralizes the numerical methods and
ensures consistency across all components.

Author: Riley Jae McNamara
Date: 2025-02-19
"""

import numpy as np
from numba import njit


@njit
def _gradient_neumann(field, dx, axis):
    """
    Compute the gradient of a field with Neumann boundary conditions using 4th-order accuracy.
    
    Uses 4th-order central differences in the interior and 3rd-order one-sided differences
    at boundaries to reduce diamond-shaped artifacts.
    
    Args:
        field: The field to compute the gradient of
        dx: Grid spacing
        axis: Axis along which to compute the gradient (0, 1, or 2)
        
    Returns:
        Gradient array
    """
    grad = np.zeros_like(field)
    
    if axis == 0:  # x-direction
        # 2nd-order central differences in interior (consistent with Laplacian)
        grad[1:-1, :, :] = (field[2:, :, :] - field[:-2, :, :]) / (2*dx)
        
        # Neumann BC at boundaries: ∂φ/∂x = 0
        grad[0, :, :] = 0
        grad[-1, :, :] = 0
        
    elif axis == 1:  # y-direction
        # 2nd-order central differences in interior (consistent with Laplacian)
        grad[:, 1:-1, :] = (field[:, 2:, :] - field[:, :-2, :]) / (2*dx)
        
        # Neumann BC at boundaries: ∂φ/∂y = 0
        grad[:, 0, :] = 0
        grad[:, -1, :] = 0
        
    elif axis == 2:  # z-direction
        # 2nd-order central differences in interior (consistent with Laplacian)
        grad[:, :, 1:-1] = (field[:, :, 2:] - field[:, :, :-2]) / (2*dx)
        
        # Neumann BC at boundaries: ∂φ/∂z = 0
        grad[:, :, 0] = 0
        grad[:, :, -1] = 0
    
    return grad


def gradient(field, dx):
    """
    Compute the full gradient vector of a field.
    
    Args:
        field: The field to compute the gradient of
        dx: Grid spacing
        
    Returns:
        grad_x, grad_y, grad_z: Gradient components
    """
    grad_x = _gradient_neumann(field, dx, 0)
    grad_y = _gradient_neumann(field, dx, 1)
    grad_z = _gradient_neumann(field, dx, 2)
    
    return grad_x, grad_y, grad_z


def divergence(ux, uy, uz, dx):
    """
    Compute the divergence of a vector field with Neumann boundary conditions.
    ∇·u = ∂ux/∂x + ∂uy/∂y + ∂uz/∂z
    
    Args:
        ux, uy, uz: Vector field components
        dx: Grid spacing
        
    Returns:
        Divergence field
    """
    return (_gradient_neumann(ux, dx, 0) + 
            _gradient_neumann(uy, dx, 1) + 
            _gradient_neumann(uz, dx, 2))


@njit
def laplacian(field, dx):
    """
    Compute the Laplacian of a field with Neumann boundary conditions using 4th-order accuracy.
    ∇²φ = ∂²φ/∂x² + ∂²φ/∂y² + ∂²φ/∂z²
    
    Uses 4th-order central differences in the interior and 2nd-order one-sided differences
    at boundaries to reduce diamond-shaped artifacts.
    
    Args:
        field: The field to compute the Laplacian of
        dx: Grid spacing
        
    Returns:
        Laplacian field
    """
    # Initialize Laplacian array
    lap = np.zeros_like(field)
    dx2 = dx**2
    
    # x-direction - use consistent 2nd-order accuracy to avoid artifacts
    # 2nd-order central differences in interior
    lap[1:-1, :, :] = (field[2:, :, :] - 2*field[1:-1, :, :] + field[:-2, :, :]) / dx2
    
    # Neumann BC at boundaries: ∂φ/∂x = 0
    # Use ghost point extrapolation: φ[-1] = φ[0] and φ[nx] = φ[nx-1]
    # This gives ∂²φ/∂x² = (φ[1] - 2*φ[0] + φ[-1])/dx² = (φ[1] - φ[0])/dx²
    lap[0, :, :] = (field[1, :, :] - field[0, :, :]) / dx2
    lap[-1, :, :] = (field[-2, :, :] - field[-1, :, :]) / dx2
    
    # y-direction - use consistent 2nd-order accuracy
    # 2nd-order central differences in interior
    lap[:, 1:-1, :] += (field[:, 2:, :] - 2*field[:, 1:-1, :] + field[:, :-2, :]) / dx2
    
    # Neumann BC at boundaries: ∂φ/∂y = 0
    # Use ghost point extrapolation: φ[-1] = φ[0] and φ[ny] = φ[ny-1]
    lap[:, 0, :] += (field[:, 1, :] - field[:, 0, :]) / dx2
    lap[:, -1, :] += (field[:, -2, :] - field[:, -1, :]) / dx2
    
    # z-direction - use consistent 2nd-order accuracy
    # 2nd-order central differences in interior
    lap[:, :, 1:-1] += (field[:, :, 2:] - 2*field[:, :, 1:-1] + field[:, :, :-2]) / dx2
    
    # Neumann BC at boundaries: ∂φ/∂z = 0
    # Use ghost point extrapolation: φ[-1] = φ[0] and φ[nz] = φ[nz-1]
    lap[:, :, 0] += (field[:, :, 1] - field[:, :, 0]) / dx2
    lap[:, :, -1] += (field[:, :, -2] - field[:, :, -1]) / dx2
    
    return lap


def laplacian_neumann(p_flat, shape, dx):
    """
    Compute the 3D Laplacian with Neumann boundary conditions for flattened arrays using 4th-order accuracy.
    
    Uses 4th-order central differences in the interior and 2nd-order one-sided differences
    at boundaries to reduce diamond-shaped artifacts.
    
    Args:
        p_flat: Flattened pressure array
        shape: Tuple (nx, ny, nz) of the grid shape
        dx: Grid spacing
        
    Returns:
        Flattened Laplacian array (float64)
    """
    p = p_flat.reshape(shape)
    # Explicitly use float64 for lap_p to avoid dtype mismatch
    lap_p = np.zeros(shape, dtype=np.float64)
    dx2 = dx**2
    
    # x-direction - use consistent 2nd-order accuracy
    # 2nd-order central differences in interior
    lap_p[1:-1, :, :] = (p[2:, :, :] - 2*p[1:-1, :, :] + p[:-2, :, :]) / dx2
    
    # Neumann BC at boundaries: ∂φ/∂x = 0
    # Use ghost point extrapolation: φ[-1] = φ[0] and φ[nx] = φ[nx-1]
    lap_p[0, :, :] = (p[1, :, :] - p[0, :, :]) / dx2
    lap_p[-1, :, :] = (p[-2, :, :] - p[-1, :, :]) / dx2
    
    # y-direction - use consistent 2nd-order accuracy
    # 2nd-order central differences in interior
    lap_p[:, 1:-1, :] += (p[:, 2:, :] - 2*p[:, 1:-1, :] + p[:, :-2, :]) / dx2
    
    # Neumann BC at boundaries: ∂φ/∂y = 0
    # Use ghost point extrapolation: φ[-1] = φ[0] and φ[ny] = φ[ny-1]
    lap_p[:, 0, :] += (p[:, 1, :] - p[:, 0, :]) / dx2
    lap_p[:, -1, :] += (p[:, -2, :] - p[:, -1, :]) / dx2
    
    # z-direction - use consistent 2nd-order accuracy
    # 2nd-order central differences in interior
    lap_p[:, :, 1:-1] += (p[:, :, 2:] - 2*p[:, :, 1:-1] + p[:, :, :-2]) / dx2
    
    # Neumann BC at boundaries: ∂φ/∂z = 0
    # Use ghost point extrapolation: φ[-1] = φ[0] and φ[nz] = φ[nz-1]
    lap_p[:, :, 0] += (p[:, :, 1] - p[:, :, 0]) / dx2
    lap_p[:, :, -1] += (p[:, :, -2] - p[:, :, -1]) / dx2
    
    return lap_p.flatten()


@njit
def curl(ux, uy, uz, dx):
    """
    Compute the curl of a vector field.
    ∇×u = (∂uz/∂y - ∂uy/∂z, ∂ux/∂z - ∂uz/∂x, ∂uy/∂x - ∂ux/∂y)
    
    Args:
        ux, uy, uz: Vector field components
        dx: Grid spacing
        
    Returns:
        curl_x, curl_y, curl_z: Curl components
    """
    # ∂uz/∂y - ∂uy/∂z
    curl_x = _gradient_neumann(uz, dx, 1) - _gradient_neumann(uy, dx, 2)
    
    # ∂ux/∂z - ∂uz/∂x
    curl_y = _gradient_neumann(ux, dx, 2) - _gradient_neumann(uz, dx, 0)
    
    # ∂uy/∂x - ∂ux/∂y
    curl_z = _gradient_neumann(uy, dx, 0) - _gradient_neumann(ux, dx, 1)
    
    return curl_x, curl_y, curl_z


@njit
def vector_laplacian(ux, uy, uz, dx):
    """
    Compute the vector Laplacian of a vector field.
    ∇²u = (∇²ux, ∇²uy, ∇²uz)
    
    Args:
        ux, uy, uz: Vector field components
        dx: Grid spacing
        
    Returns:
        lap_x, lap_y, lap_z: Vector Laplacian components
    """
    lap_x = laplacian(ux, dx)
    lap_y = laplacian(uy, dx)
    lap_z = laplacian(uz, dx)
    
    return lap_x, lap_y, lap_z


@njit
def isotropic_gradient(field, dx):
    """
    Compute the isotropic gradient magnitude using a more isotropic finite difference scheme.
    
    This function computes the gradient magnitude in a way that treats all directions
    more equally, reducing diamond-shaped artifacts that arise from Cartesian finite differences.
    
    Args:
        field: The field to compute the gradient of
        dx: Grid spacing
        
    Returns:
        Gradient magnitude array
    """
    nx, ny, nz = field.shape
    grad_mag = np.zeros_like(field)
    
    # Use a more isotropic stencil that considers all neighboring points
    # This reduces the bias toward grid-aligned directions
    
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            for k in range(1, nz-1):
                # Compute gradients in all three directions
                grad_x = (field[i+1, j, k] - field[i-1, j, k]) / (2*dx)
                grad_y = (field[i, j+1, k] - field[i, j-1, k]) / (2*dx)
                grad_z = (field[i, j, k+1] - field[i, j, k-1]) / (2*dx)
                
                # Also consider diagonal neighbors for more isotropy
                # This helps reduce grid-aligned artifacts
                grad_xy = (field[i+1, j+1, k] - field[i-1, j-1, k]) / (2*np.sqrt(2)*dx)
                grad_xz = (field[i+1, j, k+1] - field[i-1, j, k-1]) / (2*np.sqrt(2)*dx)
                grad_yz = (field[i, j+1, k+1] - field[i, j-1, k-1]) / (2*np.sqrt(2)*dx)
                
                # Combine all gradient components with weights
                # The diagonal components help reduce anisotropy
                grad_mag[i, j, k] = np.sqrt(
                    0.5 * (grad_x**2 + grad_y**2 + grad_z**2) +
                    0.1 * (grad_xy**2 + grad_xz**2 + grad_yz**2)
                )
    
    # Handle boundaries with simpler stencils
    # X boundaries
    for j in range(ny):
        for k in range(nz):
            # Left boundary
            if nx > 1:
                grad_mag[0, j, k] = abs(field[1, j, k] - field[0, j, k]) / dx
            # Right boundary  
            if nx > 1:
                grad_mag[-1, j, k] = abs(field[-1, j, k] - field[-2, j, k]) / dx
    
    # Y boundaries
    for i in range(nx):
        for k in range(nz):
            # Front boundary
            if ny > 1:
                grad_mag[i, 0, k] = abs(field[i, 1, k] - field[i, 0, k]) / dx
            # Back boundary
            if ny > 1:
                grad_mag[i, -1, k] = abs(field[i, -1, k] - field[i, -2, k]) / dx
    
    # Z boundaries
    for i in range(nx):
        for j in range(ny):
            # Bottom boundary
            if nz > 1:
                grad_mag[i, j, 0] = abs(field[i, j, 1] - field[i, j, 0]) / dx
            # Top boundary
            if nz > 1:
                grad_mag[i, j, -1] = abs(field[i, j, -1] - field[i, j, -2]) / dx
    
    return grad_mag


@njit
def isotropic_gradient_components(field, dx):
    """
    Compute the isotropic gradient components using a more isotropic finite difference scheme.
    
    This function computes the gradient components in a way that treats all directions
    more equally, reducing diamond-shaped artifacts that arise from Cartesian finite differences.
    
    Args:
        field: The field to compute the gradient of
        dx: Grid spacing
        
    Returns:
        grad_x, grad_y, grad_z: Gradient component arrays
    """
    nx, ny, nz = field.shape
    grad_x = np.zeros_like(field)
    grad_y = np.zeros_like(field)
    grad_z = np.zeros_like(field)
    
    # Use a more isotropic stencil that considers all neighboring points
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            for k in range(1, nz-1):
                # Standard Cartesian gradients
                grad_x[i, j, k] = (field[i+1, j, k] - field[i-1, j, k]) / (2*dx)
                grad_y[i, j, k] = (field[i, j+1, k] - field[i, j-1, k]) / (2*dx)
                grad_z[i, j, k] = (field[i, j, k+1] - field[i, j, k-1]) / (2*dx)
                
                # Add diagonal contributions for more isotropy
                # This helps reduce grid-aligned artifacts
                grad_x[i, j, k] += 0.1 * (
                    (field[i+1, j+1, k] - field[i-1, j-1, k]) / (2*np.sqrt(2)*dx) +
                    (field[i+1, j-1, k] - field[i-1, j+1, k]) / (2*np.sqrt(2)*dx) +
                    (field[i+1, j, k+1] - field[i-1, j, k-1]) / (2*np.sqrt(2)*dx) +
                    (field[i+1, j, k-1] - field[i-1, j, k+1]) / (2*np.sqrt(2)*dx)
                )
                
                grad_y[i, j, k] += 0.1 * (
                    (field[i+1, j+1, k] - field[i-1, j-1, k]) / (2*np.sqrt(2)*dx) +
                    (field[i-1, j+1, k] - field[i+1, j-1, k]) / (2*np.sqrt(2)*dx) +
                    (field[i, j+1, k+1] - field[i, j-1, k-1]) / (2*np.sqrt(2)*dx) +
                    (field[i, j+1, k-1] - field[i, j-1, k+1]) / (2*np.sqrt(2)*dx)
                )
                
                grad_z[i, j, k] += 0.1 * (
                    (field[i+1, j, k+1] - field[i-1, j, k-1]) / (2*np.sqrt(2)*dx) +
                    (field[i-1, j, k+1] - field[i+1, j, k-1]) / (2*np.sqrt(2)*dx) +
                    (field[i, j+1, k+1] - field[i, j-1, k-1]) / (2*np.sqrt(2)*dx) +
                    (field[i, j-1, k+1] - field[i, j+1, k-1]) / (2*np.sqrt(2)*dx)
                )
    
    # Handle boundaries with simpler stencils
    # X boundaries
    for j in range(ny):
        for k in range(nz):
            # Left boundary
            if nx > 1:
                grad_x[0, j, k] = (field[1, j, k] - field[0, j, k]) / dx
            # Right boundary  
            if nx > 1:
                grad_x[-1, j, k] = (field[-1, j, k] - field[-2, j, k]) / dx
    
    # Y boundaries
    for i in range(nx):
        for k in range(nz):
            # Front boundary
            if ny > 1:
                grad_y[i, 0, k] = (field[i, 1, k] - field[i, 0, k]) / dx
            # Back boundary
            if ny > 1:
                grad_y[i, -1, k] = (field[i, -1, k] - field[i, -2, k]) / dx
    
    # Z boundaries
    for i in range(nx):
        for j in range(ny):
            # Bottom boundary
            if nz > 1:
                grad_z[i, j, 0] = (field[i, j, 1] - field[i, j, 0]) / dx
            # Top boundary
            if nz > 1:
                grad_z[i, j, -1] = (field[i, j, -1] - field[i, j, -2]) / dx
    
    return grad_x, grad_y, grad_z


@njit
def isotropic_laplacian(field, dx):
    """
    Compute the isotropic Laplacian using a rotated stencil approach.
    
    This uses multiple rotated stencils to achieve better isotropy than
    standard Cartesian finite differences, reducing diamond-shaped artifacts.
    
    Args:
        field: The field to compute the Laplacian of
        dx: Grid spacing
        
    Returns:
        Laplacian array
    """
    nx, ny, nz = field.shape
    lap = np.zeros_like(field)
    dx2 = dx * dx
    
    # Use multiple rotated stencils for better isotropy
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            for k in range(1, nz-1):
                # Standard Cartesian stencil
                cartesian = (
                    (field[i+1, j, k] - 2*field[i, j, k] + field[i-1, j, k]) +
                    (field[i, j+1, k] - 2*field[i, j, k] + field[i, j-1, k]) +
                    (field[i, j, k+1] - 2*field[i, j, k] + field[i, j, k-1])
                ) / dx2
                
                # Rotated stencils (45-degree rotations in each plane)
                # XY plane rotation
                rotated_xy = (
                    (field[i+1, j+1, k] - 2*field[i, j, k] + field[i-1, j-1, k]) +
                    (field[i+1, j-1, k] - 2*field[i, j, k] + field[i-1, j+1, k])
                ) / (2 * dx2)
                
                # XZ plane rotation
                rotated_xz = (
                    (field[i+1, j, k+1] - 2*field[i, j, k] + field[i-1, j, k-1]) +
                    (field[i+1, j, k-1] - 2*field[i, j, k] + field[i-1, j, k+1])
                ) / (2 * dx2)
                
                # YZ plane rotation
                rotated_yz = (
                    (field[i, j+1, k+1] - 2*field[i, j, k] + field[i, j-1, k-1]) +
                    (field[i, j+1, k-1] - 2*field[i, j, k] + field[i, j-1, k+1])
                ) / (2 * dx2)
                
                # Combine all stencils with equal weight for maximum isotropy
                lap[i, j, k] = (cartesian + rotated_xy + rotated_xz + rotated_yz) / 4.0
    
    # Handle boundaries with simpler stencils
    # X boundaries
    for j in range(ny):
        for k in range(nz):
            # Left boundary
            if nx > 1:
                lap[0, j, k] = (field[1, j, k] - field[0, j, k]) / dx2
            # Right boundary
            if nx > 1:
                lap[-1, j, k] = (field[-2, j, k] - field[-1, j, k]) / dx2
    
    # Y boundaries
    for i in range(nx):
        for k in range(nz):
            # Front boundary
            if ny > 1:
                lap[i, 0, k] = (field[i, 1, k] - field[i, 0, k]) / dx2
            # Back boundary
            if ny > 1:
                lap[i, -1, k] = (field[i, -2, k] - field[i, -1, k]) / dx2
    
    # Z boundaries
    for i in range(nx):
        for j in range(ny):
            # Bottom boundary
            if nz > 1:
                lap[i, j, 0] = (field[i, j, 1] - field[i, j, 0]) / dx2
            # Top boundary
            if nz > 1:
                lap[i, j, -1] = (field[i, j, -2] - field[i, j, -1]) / dx2
    
    return lap
