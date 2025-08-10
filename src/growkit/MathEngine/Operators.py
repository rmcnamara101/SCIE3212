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
    Compute the Laplacian of a field with Neumann boundary conditions.
    ∇²φ = ∂²φ/∂x² + ∂²φ/∂y² + ∂²φ/∂z²
    
    Args:
        field: The field to compute the Laplacian of
        dx: Grid spacing
        
    Returns:
        Laplacian field
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


def laplacian_neumann(p_flat, shape, dx):
    """
    Compute the 3D Laplacian with Neumann boundary conditions for flattened arrays.
    
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
