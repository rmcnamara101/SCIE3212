"""
Vectorized Cell Dynamics Module

This module integrates the separated physics components (Energy, MassFlux, SolidVelocity)
to compute the complete cell dynamics in a vectorized form following the same pattern
as SourceConstructor and FieldManager. The dynamics equation is:

dφ_hat/dt = -∇·(u ⊗ φ_hat) - ∇·J_hat + A_hat

where:
- φ_hat: stacked cell fraction fields for all M populations
- u: solid velocity field (same for all populations)
- J_hat: stacked mass flux fields for all M populations
- A_hat: source terms (growth/death) for all populations
- ⊗: outer product (velocity applied to each population)

Author: Riley Jae McNamara
Date: 2025-02-19
"""

import numpy as np
from numba import njit
import time

from src.growkit.MathEngine.Operators import _divergence_from_components, isotropic_gradient_components
from src.growkit.MathEngine.NaturalBoundaryConditions import apply_natural_gradient_boundaries


class VectorizedCellDynamics:
    """
    Vectorized cell dynamics computer that handles all populations simultaneously.
    """
    
    def __init__(self, cfg, populations, field_manager=None, source_constructor=None):
        """
        Initialize the vectorized cell dynamics computer.
        
        Args:
            cfg: Configuration dictionary
            populations: Population definitions from YAML
            field_manager: Optional FieldManager instance for storing pressure/velocity
            source_constructor: Optional SourceConstructor instance for growth/death terms
        """
        self.cfg = cfg
        self.pops = populations
        self.labels = list(populations.keys())
        self.M = len(self.labels)
        self.field_manager = field_manager
        self.source_constructor = source_constructor
        
        # Initialize separated physics components
        from src.growkit.PhysicsEngine.Energy.VectorizedEnergy import VectorizedEnergy
        from src.growkit.PhysicsEngine.MassFlux.VectorizedMassFlux import VectorizedMassFlux
        from src.growkit.PhysicsEngine.SolidVelocity.VectorizedSolidVelocity import VectorizedSolidVelocity
        
        self.energy_computer = VectorizedEnergy(cfg, populations, field_manager)
        self.mass_flux_computer = VectorizedMassFlux(cfg, populations, field_manager)
        self.velocity_computer = VectorizedSolidVelocity(cfg, populations, field_manager)
        
        # Pre-compile Numba functions to avoid JIT compilation overhead
        self._precompile_numba_functions()
    
    def _precompile_numba_functions(self):
        """Pre-compile Numba functions to avoid JIT compilation overhead."""
        print("Pre-compiling Numba functions...")
        
        # Create dummy arrays for compilation
        dummy_shape = (3, 10, 10, 10)  # Small arrays for compilation
        dummy_phi_hat = np.random.random(dummy_shape).astype(np.float32)
        dummy_ux = np.random.random((10, 10, 10)).astype(np.float32)
        dummy_uy = np.random.random((10, 10, 10)).astype(np.float32)
        dummy_uz = np.random.random((10, 10, 10)).astype(np.float32)
        dummy_J_hat = np.random.random((3, 3, 10, 10, 10)).astype(np.float32)
        dummy_A_hat = np.random.random(dummy_shape).astype(np.float32)
        dummy_dx = 0.2
        
        # Compile the functions
        try:
            _ = compute_cell_dynamics_numba(dummy_phi_hat, dummy_ux, dummy_uy, dummy_uz, dummy_J_hat, dummy_A_hat, dummy_dx)
            print("Numba functions compiled successfully!")
        except Exception as e:
            print(f"Warning: Numba compilation failed: {e}")
    
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
    
    def compute_dynamics(self, phi_hat, nutrient_field, dx, source_terms=None):
        """
        Compute cell dynamics derivatives for all populations.
        
        Args:
            phi_hat: Stacked cell fraction fields (M, nx, ny, nz)
            nutrient_field: Nutrient concentration field
            dx: Grid spacing
            source_terms: Pre-computed source terms (optional, for efficiency)
            
        Returns:
            dphi_hat: Derivatives of stacked cell fraction fields (M, nx, ny, nz)
        """
        # Compute total cell density once
        phi_T = np.sum(phi_hat, axis=0)
        
        # Compute energy derivative once and reuse
        energy_deriv = self.compute_energy_derivative(phi_T, dx)
        
        # Compute solid velocity
        ux, uy, uz = self.compute_solid_velocity(phi_hat, nutrient_field, dx)
        
        # Compute mass fluxes (pass precomputed energy derivative)
        J_hat = self.compute_mass_fluxes(phi_hat, dx, energy_deriv)
        
        # Use provided source terms or compute them if not available
        if source_terms is None:
            if self.source_constructor is not None:
                source_terms = self.source_constructor.compute_source_vector_vectorized(phi_hat, nutrient_field)
            else:
                # If no source constructor, create zero source terms
                source_terms = np.zeros_like(phi_hat)
        
        # Compute dynamics using the vectorized approach
        result = compute_cell_dynamics_numba(phi_hat, ux, uy, uz, J_hat, source_terms, dx)
        
        return result
    
    def compute_source_terms(self, phi_hat, nutrient_field):
        """
        Compute source terms (growth/death) for all populations.
        This should be called once per time step, not for each RK4 coefficient.
        
        Args:
            phi_hat: Stacked cell fraction fields (M, nx, ny, nz)
            nutrient_field: Nutrient concentration field
            
        Returns:
            A_hat: Source terms for all populations (M, nx, ny, nz)
        """
        if self.source_constructor is not None:
            A_hat = self.source_constructor.compute_source_vector_vectorized(phi_hat, nutrient_field)
    
            return A_hat
        else:
            return np.zeros_like(phi_hat)





@njit
def _isotropic_upwind_divergence(ux, uy, uz, phi, dx):
    """
    Compute isotropic upwind divergence of u * phi using gradient-based approach.
    This eliminates grid-aligned artifacts by using isotropic gradient components.
    """
    nx, ny, nz = phi.shape
    div = np.zeros_like(phi, dtype=np.float32)
    
    # Compute gradients of phi using isotropic scheme
    # We'll use a simplified isotropic gradient computation here
    inv_2dx = 1.0 / (2.0 * dx)
    inv_2sqrt2dx = 1.0 / (2.0 * np.sqrt(2.0) * dx)
    w_axis = 1.0
    w_diag = 0.10  # diagonal blend weight
    
    # Compute isotropic gradient components
    grad_phi_x = np.zeros_like(phi)
    grad_phi_y = np.zeros_like(phi)
    grad_phi_z = np.zeros_like(phi)
    
    # Interior points with isotropic stencil
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            for k in range(1, nz-1):
                # Axis-aligned components
                gx_axis = (phi[i+1, j, k] - phi[i-1, j, k]) * inv_2dx
                gy_axis = (phi[i, j+1, k] - phi[i, j-1, k]) * inv_2dx
                gz_axis = (phi[i, j, k+1] - phi[i, j, k-1]) * inv_2dx
                
                # Diagonal components for isotropy
                gx_diag = (
                    (phi[i+1, j+1, k] - phi[i-1, j-1, k]) * inv_2sqrt2dx +
                    (phi[i+1, j-1, k] - phi[i-1, j+1, k]) * inv_2sqrt2dx +
                    (phi[i+1, j, k+1] - phi[i-1, j, k-1]) * inv_2sqrt2dx +
                    (phi[i+1, j, k-1] - phi[i-1, j, k+1]) * inv_2sqrt2dx
                ) * 0.25
                
                gy_diag = (
                    (phi[i+1, j+1, k] - phi[i-1, j-1, k]) * inv_2sqrt2dx +
                    (phi[i-1, j+1, k] - phi[i+1, j-1, k]) * inv_2sqrt2dx +
                    (phi[i, j+1, k+1] - phi[i, j-1, k-1]) * inv_2sqrt2dx +
                    (phi[i, j+1, k-1] - phi[i, j-1, k+1]) * inv_2sqrt2dx
                ) * 0.25
                
                gz_diag = (
                    (phi[i+1, j, k+1] - phi[i-1, j, k-1]) * inv_2sqrt2dx +
                    (phi[i-1, j, k+1] - phi[i+1, j, k-1]) * inv_2sqrt2dx +
                    (phi[i, j+1, k+1] - phi[i, j-1, k-1]) * inv_2sqrt2dx +
                    (phi[i, j-1, k+1] - phi[i, j+1, k-1]) * inv_2sqrt2dx
                ) * 0.25
                
                # Combine axis and diagonal components
                grad_phi_x[i, j, k] = w_axis * gx_axis + w_diag * gx_diag
                grad_phi_y[i, j, k] = w_axis * gy_axis + w_diag * gy_diag
                grad_phi_z[i, j, k] = w_axis * gz_axis + w_diag * gz_diag
    
    # Apply NO boundary conditions - let gradients develop completely naturally
    # This should eliminate all artificial constraints that cause diamond artifacts
    # grad_phi_x, grad_phi_y, grad_phi_z = apply_natural_gradient_boundaries(
    #     grad_phi_x, grad_phi_y, grad_phi_z, boundary_width=1
    # )
    
    # Compute divergence using isotropic approach
    # ∇·(u φ) = φ ∇·u + u·∇φ
    # For incompressible flow: ∇·u = 0, so ∇·(u φ) = u·∇φ
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Compute u·∇φ using isotropic gradients
                div[i, j, k] = (ux[i, j, k] * grad_phi_x[i, j, k] + 
                               uy[i, j, k] * grad_phi_y[i, j, k] + 
                               uz[i, j, k] * grad_phi_z[i, j, k])
    
    # Return -∇·(u φ)
    return -div


@njit
def compute_cell_dynamics_numba(phi_hat, ux, uy, uz, J_hat, A_hat, dx):
    """
    Compute cell dynamics derivatives using Numba optimization.
    
    Args:
        phi_hat: Stacked cell fraction fields (M, nx, ny, nz)
        ux, uy, uz: Velocity components
        J_hat: Stacked mass flux fields (M, 3, nx, ny, nz)
        A_hat: Source terms (growth/death) for all populations (M, nx, ny, nz)
        dx: Grid spacing
        
    Returns:
        dphi_hat: Derivatives of stacked cell fraction fields (M, nx, ny, nz)
    """
    M, nx, ny, nz = phi_hat.shape
    dphi_hat = np.zeros_like(phi_hat, dtype=np.float32)
    
    # Compute dynamics for each population
    for i in range(M):
        # Advection term: isotropic upwind divergence for -∇·(u φ_i)
        advection = _isotropic_upwind_divergence(ux, uy, uz, phi_hat[i], dx)
        
        # Mass flux term: -∇·J_i (using isotropic divergence)
        Jx = J_hat[i, 0]
        Jy = J_hat[i, 1]
        Jz = J_hat[i, 2]
        mass_flux = -_divergence_from_components(Jx, Jy, Jz, dx)
        
        # Source term: A_i (growth/death)
        source_term = A_hat[i]
        
        # Total derivative: dφ_i/dt = -∇·(u φ_i) - ∇·J_i + A_i
        dphi_hat[i] = advection + mass_flux + source_term
        
        # Apply conservative clipping to prevent unphysical values
        dphi_hat[i] = np.clip(dphi_hat[i], -1.0, 1.0)
        
    
    return dphi_hat
