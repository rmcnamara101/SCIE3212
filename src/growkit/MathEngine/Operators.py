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
        # 4th-order central differences in interior
        grad[2:-2, :, :] = (-field[4:, :, :] + 8*field[3:-1, :, :] - 8*field[1:-3, :, :] + field[:-4, :, :]) / (12*dx)
        
        # 3rd-order one-sided differences near boundaries
        grad[1, :, :] = (-11*field[0, :, :] + 18*field[1, :, :] - 9*field[2, :, :] + 2*field[3, :, :]) / (6*dx)
        grad[-2, :, :] = (11*field[-1, :, :] - 18*field[-2, :, :] + 9*field[-3, :, :] - 2*field[-4, :, :]) / (6*dx)
        
        # Neumann BC at boundaries: ∂φ/∂x = 0
        grad[0, :, :] = 0
        grad[-1, :, :] = 0
        
    elif axis == 1:  # y-direction
        # 4th-order central differences in interior
        grad[:, 2:-2, :] = (-field[:, 4:, :] + 8*field[:, 3:-1, :] - 8*field[:, 1:-3, :] + field[:, :-4, :]) / (12*dx)
        
        # 3rd-order one-sided differences near boundaries
        grad[:, 1, :] = (-11*field[:, 0, :] + 18*field[:, 1, :] - 9*field[:, 2, :] + 2*field[:, 3, :]) / (6*dx)
        grad[:, -2, :] = (11*field[:, -1, :] - 18*field[:, -2, :] + 9*field[:, -3, :] - 2*field[:, -4, :]) / (6*dx)
        
        # Neumann BC at boundaries: ∂φ/∂y = 0
        grad[:, 0, :] = 0
        grad[:, -1, :] = 0
        
    elif axis == 2:  # z-direction
        # 4th-order central differences in interior
        grad[:, :, 2:-2] = (-field[:, :, 4:] + 8*field[:, :, 3:-1] - 8*field[:, :, 1:-3] + field[:, :, :-4]) / (12*dx)
        
        # 3rd-order one-sided differences near boundaries
        grad[:, :, 1] = (-11*field[:, :, 0] + 18*field[:, :, 1] - 9*field[:, :, 2] + 2*field[:, :, 3]) / (6*dx)
        grad[:, :, -2] = (11*field[:, :, -1] - 18*field[:, :, -2] + 9*field[:, :, -3] - 2*field[:, :, -4]) / (6*dx)
        
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
    
    # x-direction
    # 4th-order central differences in interior
    lap[2:-2, :, :] = (-field[4:, :, :] + 16*field[3:-1, :, :] - 30*field[2:-2, :, :] + 16*field[1:-3, :, :] - field[:-4, :, :]) / (12*dx2)
    
    # 2nd-order one-sided differences near boundaries
    lap[1, :, :] = (2*field[0, :, :] - 5*field[1, :, :] + 4*field[2, :, :] - field[3, :, :]) / dx2
    lap[-2, :, :] = (2*field[-1, :, :] - 5*field[-2, :, :] + 4*field[-3, :, :] - field[-4, :, :]) / dx2
    
    # Neumann BC at boundaries: ∂φ/∂x = 0, so use one-sided difference
    lap[0, :, :] = (field[1, :, :] - field[0, :, :]) / dx2
    lap[-1, :, :] = (field[-2, :, :] - field[-1, :, :]) / dx2
    
    # y-direction
    # 4th-order central differences in interior
    lap[:, 2:-2, :] += (-field[:, 4:, :] + 16*field[:, 3:-1, :] - 30*field[:, 2:-2, :] + 16*field[:, 1:-3, :] - field[:, :-4, :]) / (12*dx2)
    
    # 2nd-order one-sided differences near boundaries
    lap[:, 1, :] += (2*field[:, 0, :] - 5*field[:, 1, :] + 4*field[:, 2, :] - field[:, 3, :]) / dx2
    lap[:, -2, :] += (2*field[:, -1, :] - 5*field[:, -2, :] + 4*field[:, -3, :] - field[:, -4, :]) / dx2
    
    # Neumann BC at boundaries: ∂φ/∂y = 0, so use one-sided difference
    lap[:, 0, :] += (field[:, 1, :] - field[:, 0, :]) / dx2
    lap[:, -1, :] += (field[:, -2, :] - field[:, -1, :]) / dx2
    
    # z-direction
    # 4th-order central differences in interior
    lap[:, :, 2:-2] += (-field[:, :, 4:] + 16*field[:, :, 3:-1] - 30*field[:, :, 2:-2] + 16*field[:, :, 1:-3] - field[:, :, :-4]) / (12*dx2)
    
    # 2nd-order one-sided differences near boundaries
    lap[:, :, 1] += (2*field[:, :, 0] - 5*field[:, :, 1] + 4*field[:, :, 2] - field[:, :, 3]) / dx2
    lap[:, :, -2] += (2*field[:, :, -1] - 5*field[:, :, -2] + 4*field[:, :, -3] - field[:, :, -4]) / dx2
    
    # Neumann BC at boundaries: ∂φ/∂z = 0, so use one-sided difference
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
    
    # x-direction
    # 4th-order central differences in interior
    lap_p[2:-2, :, :] = (-p[4:, :, :] + 16*p[3:-1, :, :] - 30*p[2:-2, :, :] + 16*p[1:-3, :, :] - p[:-4, :, :]) / (12*dx2)
    
    # 2nd-order one-sided differences near boundaries
    lap_p[1, :, :] = (2*p[0, :, :] - 5*p[1, :, :] + 4*p[2, :, :] - p[3, :, :]) / dx2
    lap_p[-2, :, :] = (2*p[-1, :, :] - 5*p[-2, :, :] + 4*p[-3, :, :] - p[-4, :, :]) / dx2
    
    # Neumann BC at boundaries: ∂φ/∂x = 0, so use one-sided difference
    lap_p[0, :, :] = (p[1, :, :] - p[0, :, :]) / dx2
    lap_p[-1, :, :] = (p[-2, :, :] - p[-1, :, :]) / dx2
    
    # y-direction
    # 4th-order central differences in interior
    lap_p[:, 2:-2, :] += (-p[:, 4:, :] + 16*p[:, 3:-1, :] - 30*p[:, 2:-2, :] + 16*p[:, 1:-3, :] - p[:, :-4, :]) / (12*dx2)
    
    # 2nd-order one-sided differences near boundaries
    lap_p[:, 1, :] += (2*p[:, 0, :] - 5*p[:, 1, :] + 4*p[:, 2, :] - p[:, 3, :]) / dx2
    lap_p[:, -2, :] += (2*p[:, -1, :] - 5*p[:, -2, :] + 4*p[:, -3, :] - p[:, -4, :]) / dx2
    
    # Neumann BC at boundaries: ∂φ/∂y = 0, so use one-sided difference
    lap_p[:, 0, :] += (p[:, 1, :] - p[:, 0, :]) / dx2
    lap_p[:, -1, :] += (p[:, -2, :] - p[:, -1, :]) / dx2
    
    # z-direction
    # 4th-order central differences in interior
    lap_p[:, :, 2:-2] += (-p[:, :, 4:] + 16*p[:, :, 3:-1] - 30*p[:, :, 2:-2] + 16*p[:, :, 1:-3] - p[:, :, :-4]) / (12*dx2)
    
    # 2nd-order one-sided differences near boundaries
    lap_p[:, :, 1] += (2*p[:, :, 0] - 5*p[:, :, 1] + 4*p[:, :, 2] - p[:, :, 3]) / dx2
    lap_p[:, :, -2] += (2*p[:, :, -1] - 5*p[:, :, -2] + 4*p[:, :, -3] - p[:, :, -4]) / dx2
    
    # Neumann BC at boundaries: ∂φ/∂z = 0, so use one-sided difference
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
