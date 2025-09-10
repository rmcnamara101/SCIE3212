"""
Vectorized Mass Flux Module

This module computes the mass flux vector fields for all populations in a vectorized form.
The mass flux is defined as:

J_i = -M_i * ∇(δE/δφ_T) * (φ_i / φ_T)

where M_i is the mobility of population i, δE/δφ_T is the adhesion energy derivative,
φ_i is the volume fraction of population i, and φ_T is the total cell density.

Author: Riley Jae McNamara
Date: 2025-02-19
"""

import numpy as np
from numba import njit
from src.growkit.MathEngine.Operators import _gradient_neumann, isotropic_gradient_components


def compute_mass_fluxes_numba(phi_hat, phi_T, dx, energy_deriv, M_matrix):
    """
    Compute mass fluxes for all populations using Numba optimization.
    
    Args:
        phi_hat: Stacked cell fraction fields (M, nx, ny, nz)
        phi_T: Total cell density field
        dx: Grid spacing
        energy_deriv: Energy derivative field
        M_matrix: Mobility matrix (M, M)
        
    Returns:
        J_hat: Stacked mass flux fields (M, 3, nx, ny, nz)
    """
    M, nx, ny, nz = phi_hat.shape
    J_hat = np.zeros((M, 3, nx, ny, nz), dtype=np.float32)
    
    # Compute gradients of energy derivative using isotropic operators
    grad_energy_x, grad_energy_y, grad_energy_z = isotropic_gradient_components(energy_deriv, dx)
    
    # Compute mass fluxes for each population
    for i in range(M):
        mobility = M_matrix[i, i]
        
        # Base mass flux: -M * ∇(δE/δφ_T)
        Jx_base = -mobility * grad_energy_x
        Jy_base = -mobility * grad_energy_y
        Jz_base = -mobility * grad_energy_z
        
        # Scale by population fraction: (φ_i / φ_T)
        epsilon_small = 1e-6
        phi_T_clamped = np.where(phi_T < epsilon_small, epsilon_small, phi_T)
        scaling = phi_hat[i] / phi_T_clamped
        
        J_hat[i, 0] = scaling * Jx_base
        J_hat[i, 1] = scaling * Jy_base
        J_hat[i, 2] = scaling * Jz_base

        J_hat[i, 0] = J_hat[i, 0]
        J_hat[i, 1] = J_hat[i, 1]
        J_hat[i, 2] = J_hat[i, 2]
    
    return J_hat


class VectorizedMassFlux:
    """
    Vectorized mass flux computer that handles mass fluxes for all populations.
    """
    
    def __init__(self, cfg, populations, field_manager=None):
        """
        Initialize the vectorized mass flux computer.
        
        Args:
            cfg: Configuration dictionary
            populations: Population definitions from YAML
            field_manager: Optional FieldManager instance for storing mass flux
        """
        self.cfg = cfg
        self.pops = populations
        self.labels = list(populations.keys())
        self.M = len(self.labels)
        self.field_manager = field_manager
        
        # Extract mobility parameters
        self._extract_mobility_params()
        
        # Build mobility matrix M (diagonal matrix of mobilities)
        self.M_matrix = self._build_mobility_matrix()
    
    def _extract_mobility_params(self):
        """Extract mobility parameters from configuration."""
        # Mobility parameters for each population
        self.mobilities = np.array([
            p["dynamics"].get("mobility", 0.01) for p in self.pops.values()
        ], dtype=np.float32)
    
    def _build_mobility_matrix(self):
        """
        Build mobility matrix M (diagonal matrix).
        
        Returns:
            M_matrix: (M, M) diagonal mobility matrix
        """
        M_matrix = np.diag(self.mobilities)
        return M_matrix.astype(np.float32)
    
    def compute_mass_fluxes(self, phi_hat, dx, energy_deriv=None):
        """
        Compute mass fluxes for all populations.
        
        Args:
            phi_hat: Stacked cell fraction fields (M, nx, ny, nz)
            dx: Grid spacing
            energy_deriv: Optional precomputed energy derivative
            
        Returns:
            J_hat: Stacked mass flux fields (M, 3, nx, ny, nz) for (x, y, z) components
        """
        # Compute total cell density
        phi_T = np.sum(phi_hat, axis=0)
        
        # Compute energy derivative if not provided
        if energy_deriv is None:
            from src.growkit.PhysicsEngine.Energy.VectorizedEnergy import VectorizedEnergy
            energy_computer = VectorizedEnergy(self.cfg, self.pops)
            energy_deriv = energy_computer.compute_energy_derivative(phi_T, dx)
        
        # Compute mass fluxes using the vectorized approach
        J_hat = compute_mass_fluxes_numba(
            phi_hat, phi_T, dx, energy_deriv, self.M_matrix
        )
        
        # Store mass flux in field manager if available
        if self.field_manager is not None:
            self.field_manager.mass_flux = J_hat
        
        return J_hat
