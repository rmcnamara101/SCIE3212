"""
Vectorized Energy Module

This module computes the adhesion energy derivative field in a vectorized form.
The adhesion energy is defined as:

E = m * (f(φ) - 0.01 * ∇²φ)

where f(φ) = 0.5 * φ * (1 - φ) * (2φ - 1) is the double-well potential.

Author: Riley Jae McNamara
Date: 2025-02-19
"""

import numpy as np
from numba import njit
from src.growkit.MathEngine.Operators import isotropic_laplacian


def compute_adhesion_energy_derivative_numba(phi, laplace_phi, m):
    """
    Compute adhesion energy derivative using Numba optimization.
    
    Args:
        phi: Total cell density field
        laplace_phi: Precomputed Laplacian of phi
        m: Adhesion energy parameter
        
    Returns:
        Energy derivative field: δE/δφ
    """
    
    # Compute double-well potential derivative: f'(φ) = 0.5 * φ * (1 - φ) * (2φ - 1)
    f_prime = 0.5 * phi * (1 - phi) * (2 * phi - 1)
    
    # Compute energy derivative: δE/δφ = m * (f'(φ) - 0.01 * ∇²φ)
    energy_deriv = m * (f_prime - 0.01 * laplace_phi)
    
    
    return energy_deriv


class VectorizedEnergy:
    """
    Vectorized energy computer that handles adhesion energy derivatives.
    """
    
    def __init__(self, cfg, populations, field_manager=None):
        """
        Initialize the vectorized energy computer.
        
        Args:
            cfg: Configuration dictionary
            populations: Population definitions from YAML
            field_manager: Optional FieldManager instance for storing energy derivative
        """
        self.cfg = cfg
        self.pops = populations
        self.labels = list(populations.keys())
        self.M = len(self.labels)
        self.field_manager = field_manager
        
        # Extract energy parameters
        self._extract_energy_params()
    
    def _extract_energy_params(self):
        """Extract energy parameters from configuration."""
        self.m = self.cfg["physics"]["adhesion_energy"]["m"]
    
    def compute_energy_derivative(self, phi_T, dx):
        """
        Compute adhesion energy derivative for total cell density.
        
        Args:
            phi_T: Total cell density field
            dx: Grid spacing
            
        Returns:
            energy_deriv: Energy derivative field
        """
        # NO SMOOTHING - use original field directly to eliminate potential diamond artifacts
        # Gaussian smoothing might be creating grid-aligned artifacts
        # Use the original phi_T field without any smoothing
        laplace_phi = isotropic_laplacian(phi_T, dx)
        # Use original phi_T for both double-well derivative and curvature term
        energy_deriv = compute_adhesion_energy_derivative_numba(phi_T, laplace_phi, self.m)
    
        
        # Store energy derivative in field manager if available
        if self.field_manager is not None:
            self.field_manager.energy_derivative = energy_deriv
        
        return energy_deriv
    
    def compute_energy_derivative_from_phi_hat(self, phi_hat, dx):
        """
        Compute adhesion energy derivative from stacked population fields.
        
        Args:
            phi_hat: Stacked cell fraction fields (M, nx, ny, nz)
            dx: Grid spacing
            
        Returns:
            energy_deriv: Energy derivative field
        """
        # Compute total cell density
        phi_T = np.sum(phi_hat, axis=0)
        return self.compute_energy_derivative(phi_T, dx)
