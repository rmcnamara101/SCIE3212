"""
Vectorized Cell Dynamics Module

This module integrates the separated physics components (Energy, MassFlux, SolidVelocity)
to compute the complete cell dynamics in a vectorized form following the same pattern
as SourceConstructor and FieldManager. The dynamics equation is:

dφ_hat/dt = -∇·(u ⊗ φ_hat) - ∇·J_hat

where:
- φ_hat: stacked cell fraction fields for all M populations
- u: solid velocity field (same for all populations)
- J_hat: stacked mass flux fields for all M populations
- ⊗: outer product (velocity applied to each population)

Author: Riley Jae McNamara
Date: 2025-02-19
"""

import numpy as np
from numba import njit
from src.growkit.MathEngine.Operators import _gradient_neumann


class VectorizedCellDynamics:
    """
    Vectorized cell dynamics computer that handles all populations simultaneously.
    """
    
    def __init__(self, cfg, populations, field_manager=None):
        """
        Initialize the vectorized cell dynamics computer.
        
        Args:
            cfg: Configuration dictionary
            populations: Population definitions from YAML
            field_manager: Optional FieldManager instance for storing pressure/velocity
        """
        self.cfg = cfg
        self.pops = populations
        self.labels = list(populations.keys())
        self.M = len(self.labels)
        self.field_manager = field_manager
        
        # Initialize separated physics components
        from src.growkit.PhysicsEngine.Energy.VectorizedEnergy import VectorizedEnergy
        from src.growkit.PhysicsEngine.MassFlux.VectorizedMassFlux import VectorizedMassFlux
        from src.growkit.PhysicsEngine.SolidVelocity.VectorizedSolidVelocity import VectorizedSolidVelocity
        
        self.energy_computer = VectorizedEnergy(cfg, populations, field_manager)
        self.mass_flux_computer = VectorizedMassFlux(cfg, populations, field_manager)
        self.velocity_computer = VectorizedSolidVelocity(cfg, populations, field_manager)
    
    def compute_energy_derivative(self, phi_T, dx):
        """
        Compute adhesion energy derivative for total cell density.
        
        Args:
            phi_T: Total cell density field
            dx: Grid spacing
            
        Returns:
            energy_deriv: Energy derivative field
        """
        return self.energy_computer.compute_energy_derivative(phi_T, dx)
    
    def compute_solid_velocity(self, phi_hat, nutrient_field, dx):
        """
        Compute solid velocity field.
        
        Args:
            phi_hat: Stacked cell fraction fields (M, nx, ny, nz)
            nutrient_field: Nutrient concentration field
            dx: Grid spacing
            
        Returns:
            ux, uy, uz: Velocity components
        """
        return self.velocity_computer.compute_solid_velocity(phi_hat, nutrient_field, dx)
    
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
        return self.mass_flux_computer.compute_mass_fluxes(phi_hat, dx, energy_deriv)
    
    def compute_dynamics(self, phi_hat, nutrient_field, dx):
        """
        Compute cell dynamics derivatives for all populations.
        
        Args:
            phi_hat: Stacked cell fraction fields (M, nx, ny, nz)
            nutrient_field: Nutrient concentration field
            dx: Grid spacing
            
        Returns:
            dphi_hat: Derivatives of stacked cell fraction fields (M, nx, ny, nz)
        """
        # Compute solid velocity
        ux, uy, uz = self.compute_solid_velocity(phi_hat, nutrient_field, dx)
        ux, uy, uz = -ux, -uy, -uz
        
        # Compute mass fluxes
        J_hat = self.compute_mass_fluxes(phi_hat, dx)
        
        # Compute dynamics using the vectorized approach
        return compute_cell_dynamics_numba(phi_hat, ux, uy, uz, J_hat, dx)





@njit
def compute_cell_dynamics_numba(phi_hat, ux, uy, uz, J_hat, dx):
    """
    Compute cell dynamics derivatives using Numba optimization.
    
    Args:
        phi_hat: Stacked cell fraction fields (M, nx, ny, nz)
        ux, uy, uz: Velocity components
        J_hat: Stacked mass flux fields (M, 3, nx, ny, nz)
        dx: Grid spacing
        
    Returns:
        dphi_hat: Derivatives of stacked cell fraction fields (M, nx, ny, nz)
    """
    M, nx, ny, nz = phi_hat.shape
    dphi_hat = np.zeros_like(phi_hat, dtype=np.float32)
    
    # Compute dynamics for each population
    for i in range(M):
        # Advection term: -∇·(u φ_i)
        adv_x = ux * phi_hat[i]
        adv_y = uy * phi_hat[i]
        adv_z = uz * phi_hat[i]
        advection = -(_gradient_neumann(adv_x, dx, 0) + 
                      _gradient_neumann(adv_y, dx, 1) + 
                      _gradient_neumann(adv_z, dx, 2))
        
        # Mass flux term: -∇·J_i
        Jx = J_hat[i, 0]
        Jy = J_hat[i, 1]
        Jz = J_hat[i, 2]
        mass_flux = -(_gradient_neumann(Jx, dx, 0) + 
                      _gradient_neumann(Jy, dx, 1) + 
                      _gradient_neumann(Jz, dx, 2))
        
        # Total derivative
        dphi_hat[i] = advection + mass_flux
        
        # Clip for stability
        dphi_hat[i] = np.clip(dphi_hat[i], -1, 1)
    
    return dphi_hat
