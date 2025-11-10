"""
Vectorized Solid Velocity Module

This module computes the solid velocity field using the pressure gradient and adhesion energy.
The velocity is computed as:

u = -(∇p + (δE/δφ_T) ∇φ_T)

where:
- p: pressure field (solved from Poisson equation)
- δE/δφ_T: adhesion energy derivative
- φ_T: total cell density

Author: Riley Jae McNamara
Date: 2025-02-19
"""

import numpy as np
from functools import lru_cache
from src.growkit.MathEngine.Operators import isotropic_gradient, isotropic_gradient_components

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
from src.growkit.Integrator.PressureSolver import PressureSolver


class VectorizedSolidVelocity:
    """
    Vectorized solid velocity computer that handles all populations simultaneously.
    """
    
    def __init__(self, cfg, populations, field_manager=None):
        """
        Initialize the vectorized solid velocity computer.
        
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
        self.pressure_constant = cfg.get("physics", {}).get("pressure_constant", 1.0)
        # Check if pressure solver should be disabled
        self.disable_pressure = cfg.get("physics", {}).get("disable_pressure", False)
        
        # Initialize pressure solver only if not disabled
        if not self.disable_pressure:
            self.pressure_solver = self._create_pressure_solver(cfg)
        else:
            self.pressure_solver = None
            print("Pressure solver disabled - using adhesion-only mode")
        
        # Initialize gravity and bowl potential parameters
        physics_config = cfg.get("physics", {})
        self.gravity_enabled = physics_config.get("gravity_enabled", False)
        self.gravity_strength = physics_config.get("gravity_strength", 0.1)
        self.bowl_potential_enabled = physics_config.get("bowl_potential_enabled", False)
        self.bowl_potential_strength = physics_config.get("bowl_potential_strength", 0.05)
        
        # Get bowl center, default to domain center if not specified
        domain_shape = cfg.get("domain", {}).get("shape", 100)
        default_center = [domain_shape // 2, domain_shape // 2, domain_shape // 2]
        self.bowl_center = physics_config.get("bowl_center", default_center)
        
        # Cache for pressure solutions to avoid redundant solves during RK4
        self._pressure_cache = {}
    
    def _compute_state_hash(self, phi_hat, nutrient_field, energy_deriv):
        """
        Compute a hash of the current state for caching.
        Uses a simplified hash based on key features to avoid expensive full comparisons.
        """
        # Compute total cell density
        phi_T = np.sum(phi_hat, axis=0)
        
        # Use key statistics for hashing (much faster than full array comparison)
        phi_T_mean = np.mean(phi_T)
        phi_T_std = np.std(phi_T)
        phi_T_max = np.max(phi_T)
        nutrient_mean = np.mean(nutrient_field)
        energy_mean = np.mean(energy_deriv) if energy_deriv is not None else 0.0
        
        # Create a more sensitive hash to avoid over-caching
        # Use more decimal places to distinguish between similar states
        state_hash = hash((
            round(phi_T_mean, 6),  # Increased precision to reduce false cache hits
            round(phi_T_std, 6), 
            round(phi_T_max, 6),
            round(nutrient_mean, 6),
            round(energy_mean, 6)
        ))
        
        return state_hash
    
    def _compute_gravity_velocity(self, shape):
        """
        Compute gravity velocity field (uniform downward force in -z direction).
        
        Args:
            shape: Shape of the velocity field (nx, ny, nz)
            
        Returns:
            ux, uy, uz: Velocity components (ux=0, uy=0, uz=-gravity_strength)
        """
        ux = np.zeros(shape, dtype=np.float32)
        uy = np.zeros(shape, dtype=np.float32)
        uz = np.full(shape, -self.gravity_strength, dtype=np.float32)
        return ux, uy, uz
    
    def _compute_bowl_potential_velocity(self, shape, dx):
        """
        Compute bowl potential velocity (radial restoring force toward center).
        
        Args:
            shape: Shape of the velocity field (nx, ny, nz)
            dx: Grid spacing
            
        Returns:
            ux, uy, uz: Velocity components pointing toward bowl center
        """
        nx, ny, nz = shape
        
        # Create coordinate grids
        x = np.arange(nx) * dx
        y = np.arange(ny) * dx
        z = np.arange(nz) * dx
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Compute displacement from bowl center (in physical units)
        dx_field = X - self.bowl_center[0] * dx
        dy_field = Y - self.bowl_center[1] * dx
        dz_field = Z - self.bowl_center[2] * dx
        
        # Velocity = -k * displacement (restoring force toward center)
        ux = -self.bowl_potential_strength * dx_field
        uy = -self.bowl_potential_strength * dy_field
        uz = -self.bowl_potential_strength * dz_field
        
        return ux, uy, uz
    
    def compute_solid_velocity(self, phi_hat, nutrient_field, dx, energy_deriv=None):
        """
        Compute solid velocity field with caching for RK4 efficiency.
        
        Args:
            phi_hat: Stacked cell fraction fields (M, nx, ny, nz)
            nutrient_field: Nutrient concentration field
            dx: Grid spacing
            energy_deriv: Optional precomputed energy derivative
            
        Returns:
            ux, uy, uz: Velocity components
        """
        # Compute total cell density
        phi_T = np.sum(phi_hat, axis=0)
        
        # Compute energy derivative if not provided
        if energy_deriv is None:
            from src.growkit.PhysicsEngine.Energy.VectorizedEnergy import VectorizedEnergy
            energy_computer = VectorizedEnergy(self.cfg, self.pops, self.field_manager)
            energy_deriv = energy_computer.compute_energy_derivative(phi_T, dx)
        
        # Check if pressure solver is disabled
        if self.disable_pressure:
            # Adhesion-only mode: u = -(δE/δφ_T) ∇φ_T (no pressure term)
            # Compute gradients of total cell density using isotropic scheme
            grad_C_x, grad_C_y, grad_C_z = isotropic_gradient_components(phi_T, dx)
            
            # Compute velocity: u = -energy_deriv * ∇φ_T (adhesion only)
            ux = -energy_deriv * grad_C_x
            uy = -energy_deriv * grad_C_y
            uz = -energy_deriv * grad_C_z
            
            # Add gravity contribution if enabled
            if self.gravity_enabled:
                ux_grav, uy_grav, uz_grav = self._compute_gravity_velocity(phi_T.shape)
                ux += ux_grav
                uy += uy_grav
                uz += uz_grav

            # Add bowl potential contribution if enabled  
            if self.bowl_potential_enabled:
                ux_bowl, uy_bowl, uz_bowl = self._compute_bowl_potential_velocity(
                    phi_T.shape, dx
                )
                ux += ux_bowl
                uy += uy_bowl
                uz += uz_bowl
            
            # Apply boundary conditions to velocity field to ensure proper mass conservation
            # This is critical for preventing artificial mass creation in compaction mode
            from src.growkit.MathEngine.NaturalBoundaryConditions import apply_natural_gradient_boundaries
            ux, uy, uz = apply_natural_gradient_boundaries(ux, uy, uz, boundary_width=1)
            
            # Store zero pressure in field manager if available
            if self.field_manager is not None:
                self.field_manager.pressure = np.zeros_like(phi_T)
                self.field_manager.velocity[0] = ux
                self.field_manager.velocity[1] = uy
                self.field_manager.velocity[2] = uz
            
            return ux, uy, uz
        
        # Normal mode with pressure solver
        # Ensure we have at least 2 populations for the pressure solver
        if len(phi_hat) < 2:
            raise ValueError("Need at least 2 populations for pressure solver")
        
        # Check cache for similar state
        state_hash = self._compute_state_hash(phi_hat, nutrient_field, energy_deriv)
        
        # Temporary debug: disable caching to see if that fixes the issue
        use_cache = False  # Set to False to disable caching for debugging
        
        if use_cache and state_hash in self._pressure_cache:
            # Cache hit - reuse pressure solution
            pressure = self._pressure_cache[state_hash]
        else:
            # Cache miss - solve pressure equation
            pressure = self.pressure_solver.solve_pressure(
                phi_hat, nutrient_field, dx, energy_deriv=energy_deriv
            )
            
            # Cache the result (limit cache size to prevent memory issues)
            if use_cache and len(self._pressure_cache) < 10:  # Increased cache size for RK4
                self._pressure_cache[state_hash] = pressure
            elif use_cache and len(self._pressure_cache) >= 10:
                # Clear cache if it gets too large
                self._pressure_cache.clear()
                self._pressure_cache[state_hash] = pressure
        
        # Store pressure in field manager if available
        if self.field_manager is not None:
            self.field_manager.pressure = pressure
        
        # Compute pressure gradient using isotropic scheme
        grad_p_x, grad_p_y, grad_p_z = isotropic_gradient_components(pressure, dx)
        
        # Compute gradients of total cell density using isotropic scheme
        grad_C_x, grad_C_y, grad_C_z = isotropic_gradient_components(phi_T, dx)

        
        # Compute velocity: u = -(∇p + (δE/δφ_T) ∇φ_T)
        # Note: pressure returned by solver is -p, so grad_p_* = -∂p/∂*
        # Therefore u = grad_p_* - energy_deriv * grad_C_*
        ux = self.pressure_constant * (grad_p_x - energy_deriv * grad_C_x)
        uy = self.pressure_constant *(grad_p_y - energy_deriv * grad_C_y)
        uz = self.pressure_constant *(grad_p_z - energy_deriv * grad_C_z)
        
        # Add gravity contribution if enabled
        if self.gravity_enabled:
            ux_grav, uy_grav, uz_grav = self._compute_gravity_velocity(phi_T.shape)
            ux += ux_grav
            uy += uy_grav
            uz += uz_grav

        # Add bowl potential contribution if enabled  
        if self.bowl_potential_enabled:
            ux_bowl, uy_bowl, uz_bowl = self._compute_bowl_potential_velocity(
                phi_T.shape, dx
            )
            ux += ux_bowl
            uy += uy_bowl
            uz += uz_bowl
        
        # Store velocity in field manager if available
        if self.field_manager is not None:
            # Use safe casting to prevent overflow
            self.field_manager.velocity[0] = safe_float32_cast(ux, safety_factor=0.1)
            self.field_manager.velocity[1] = safe_float32_cast(uy, safety_factor=0.1)
            self.field_manager.velocity[2] = safe_float32_cast(uz, safety_factor=0.1)

        return ux, uy, uz
    
    def _create_pressure_solver(self, cfg):
        """
        Create pressure solver with flexible parameter extraction.
        
        Args:
            cfg: Configuration dictionary
            
        Returns:
            PressureSolver instance
        """
        # Extract physics parameters
        m = cfg["physics"]["adhesion_energy"]["m"]
        epsilon = cfg["physics"]["adhesion_energy"].get("epsilon", 0.1)
        
        # Extract population parameters dynamically
        populations = cfg["populations"]
        
        # Get the first two populations for lambda, mu, p, n parameters
        # (assuming we need at least 2 populations for the pressure solver)
        pop_names = list(populations.keys())
        
        if len(pop_names) < 2:
            raise ValueError("Need at least 2 populations for pressure solver")
        
        # Use first two populations for the required parameters
        pop1 = populations[pop_names[0]]
        pop2 = populations[pop_names[1]]
        
        # Extract parameters with defaults
        lambda_1 = pop1["dynamics"].get("lambda", 0.0)
        lambda_2 = pop2["dynamics"].get("lambda", 0.0)
        mu_1 = pop1["dynamics"].get("mu", 0.0)
        mu_2 = pop2["dynamics"].get("mu", 0.0)
        p_1 = pop1["dynamics"].get("p", 0.0)
        p_2 = pop2["dynamics"].get("p", 0.0)
        n_1 = pop1["dynamics"].get("n", 0.0)
        n_2 = pop2["dynamics"].get("n", 0.0)
        
        # For the third parameter (mu_3), use the third population if available, otherwise use mu_2
        if len(pop_names) >= 3:
            mu_3 = populations[pop_names[2]]["dynamics"].get("mu", 0.0)
        else:
            mu_3 = mu_2  # Use the same as second population
        
        # Get solver type from config, default to fft for massive speedup
        solver_type = self.cfg.get("physics", {}).get("pressure_solver", "fft")
        return PressureSolver(self.cfg, self.pops, solver_type=solver_type)
