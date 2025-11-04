"""
Vectorized Mass Flux Module

This module computes the mass flux vector fields for all populations in a vectorized form.
The mass flux is defined as:

J_i = φ_i * (-M_i * ∇(δE/δφ_T))

where M_i is the mobility of population i, δE/δφ_T is the adhesion energy derivative,
and φ_i is the local population density. The gating by φ_i ensures no flux in void regions,
preventing mass creation where no cells exist.

Author: Riley Jae McNamara
Date: 2025-02-19
"""

import numpy as np
from numba import njit
from src.growkit.MathEngine.Operators import isotropic_gradient_components

def safe_float32_cast(arr, safety_factor=0.1):
    """
    Safely cast array to float32 by clipping extreme values to prevent overflow.
    
    Args:
        arr: Input array
        safety_factor: Fraction of max float32 value to use as clipping limit
    
    Returns:
        Array safely cast to float32
    """
    max_val = np.finfo(np.float32).max * safety_factor
    clipped_arr = np.clip(arr, -max_val, max_val)
    return clipped_arr.astype(np.float32)


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
    # Use float64 for numerical stability during computation
    phi_hat = phi_hat.astype(np.float64, copy=False)
    phi_T = phi_T.astype(np.float64, copy=False)
    energy_deriv = energy_deriv.astype(np.float64, copy=False)
    M_matrix = M_matrix.astype(np.float64, copy=False)
    J_hat = np.zeros((M, 3, nx, ny, nz), dtype=np.float64)
    
    # Compute gradients of energy derivative using isotropic operators
    grad_energy_x, grad_energy_y, grad_energy_z = isotropic_gradient_components(energy_deriv, dx)
    
    # Apply boundary conditions to gradient components to ensure proper mass conservation
    from src.growkit.MathEngine.NaturalBoundaryConditions import apply_natural_gradient_boundaries
    grad_energy_x, grad_energy_y, grad_energy_z = apply_natural_gradient_boundaries(
        grad_energy_x, grad_energy_y, grad_energy_z, boundary_width=1
    )
    
    # Compute mass fluxes for each population
    for i in range(M):
        mobility = M_matrix[i, i]
        
        # Base mass flux: -M * ∇(δE/δφ_T)
        # Prevent overflow in intermediate calculations using safe casting
        Jx_base = safe_float32_cast(-mobility * grad_energy_x, safety_factor=0.01)
        Jy_base = safe_float32_cast(-mobility * grad_energy_y, safety_factor=0.01)
        Jz_base = safe_float32_cast(-mobility * grad_energy_z, safety_factor=0.01)

        # Gate mass flux by local population presence to prevent flux in voids
        # This ensures J_i = 0 where φ_i = 0, preventing mass creation in empty regions
        J_hat[i, 0] = Jx_base * phi_hat[i]
        J_hat[i, 1] = Jy_base * phi_hat[i]
        J_hat[i, 2] = Jz_base * phi_hat[i]
    
    
    # Sanitize any NaNs/Infs that may have arisen numerically
    J_hat = np.nan_to_num(J_hat, nan=0.0, posinf=0.0, neginf=0.0)
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
        
        # Compute mass fluxes using the vectorized approach (float64 internal)
        J_hat = compute_mass_fluxes_numba(
            phi_hat, phi_T, dx, energy_deriv, self.M_matrix
        )
        
        # Optional global scaling for mass flux magnitude to counteract large dx
        flux_scale = (
            self.cfg.get("physics", {})
            .get("mass_flux", {})
            .get("scale", 1.0)
        )
        if flux_scale != 1.0:
            J_hat = J_hat * float(flux_scale)
        
        # Optional clipping based on config to prevent overflow/explosions
        max_mag = (
            self.cfg.get("physics", {})
            .get("mass_flux", {})
            .get("max_magnitude", None)
        )
        if max_mag is not None and max_mag > 0:
            J_hat = np.clip(J_hat, -float(max_mag), float(max_mag))
        
        # Prevent overflow by clipping extreme flux values before casting to float32
        J_hat = safe_float32_cast(J_hat, safety_factor=0.1)
        
        # Store mass flux in field manager if available
        if self.field_manager is not None:
            self.field_manager.mass_flux = J_hat
        
        return J_hat
