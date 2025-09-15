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
        
        # Initialize pressure solver
        self.pressure_solver = self._create_pressure_solver(cfg)
        
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
        ux = (grad_p_x - energy_deriv * grad_C_x)
        uy = (grad_p_y - energy_deriv * grad_C_y)
        uz = (grad_p_z - energy_deriv * grad_C_z)
        # Store velocity in field manager if available
        if self.field_manager is not None:
            self.field_manager.velocity[0] = ux
            self.field_manager.velocity[1] = uy
            self.field_manager.velocity[2] = uz

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
        
        return PressureSolver(self.cfg, self.pops)
