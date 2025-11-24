"""
Vectorized Energy Module

This module computes the adhesion energy derivative field in a vectorized form.
The adhesion energy is defined as:

E = m * (f(φ) - 0.01 * ∇²φ)

For coagulation simulations, we use a simple quadratic potential:
f(φ) = -0.5 * φ²

This creates attractive forces that encourage cells to aggregate into a single
cohesive spheroid, rather than the phase separation behavior of double-well potentials.

Author: Riley Jae McNamara
Date: 2025-02-19
"""

import numpy as np
from numba import njit
from src.growkit.MathEngine.Operators import isotropic_laplacian


def compute_adhesion_energy_derivative_numba(phi, laplace_phi, m, k):
    """
    Compute adhesion energy derivative using Numba optimization.
    
    Args:
        phi: Total cell density field
        laplace_phi: Precomputed Laplacian of phi
        m: Adhesion energy parameter
        
    Returns:
        Energy derivative field: δE/δφ
    """
    
    # Promote to float64 for stability
    phi = phi.astype(np.float64, copy=False)
    laplace_phi = laplace_phi.astype(np.float64, copy=False)
    m = float(m)

    # Simple quadratic potential: f(φ) = -0.5 * φ * (1 - φ) * (2 * φ - 1)
    # This creates attractive forces that pull cells together
    f_prime = - (k / 4) * phi * (1 - phi) * (2 * phi - 1)
    
    # Compute energy derivative: δE/δφ = m * (f'(φ) - 0.01 * ∇²φ)
    # The negative f_prime creates attractive forces toward high cell density
    energy_deriv = f_prime - m * laplace_phi
    
    
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
        self.k = self.cfg["physics"]["adhesion_energy"]["k"]

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
        # Ensure float64 for operator math
        phi_T = phi_T.astype(np.float64, copy=False)
        laplace_phi = isotropic_laplacian(phi_T, dx)
        # Use original phi_T for both double-well derivative and curvature term
        energy_deriv = compute_adhesion_energy_derivative_numba(phi_T, laplace_phi, self.m, self.k)
        
        # Optional config-based clipping for very large adhesion m
        max_ed = (
            self.cfg.get("physics", {})
            .get("adhesion_energy", {})
            .get("max_energy_derivative", None)
        )
        if max_ed is not None and max_ed > 0:
            energy_deriv = np.clip(energy_deriv, -float(max_ed), float(max_ed))
        
        # Store as float64 internally; cast down only when storing to field manager
    
        
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
