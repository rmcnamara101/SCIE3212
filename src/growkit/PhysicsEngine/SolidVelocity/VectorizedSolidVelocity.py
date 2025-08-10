"""
Vectorized Solid Velocity Module

This module computes the solid velocity vector field in a vectorized form.
The solid velocity is defined as:

u = -(∇p + (δE/δφ_T) ∇φ_T)

where p is the pressure field, δE/δφ_T is the adhesion energy derivative,
and φ_T is the total cell density.

Author: Riley Jae McNamara
Date: 2025-02-19
"""

import numpy as np
from src.growkit.MathEngine.Operators import gradient
from src.growkit.Integrator.PressureSolver import PressureSolver


class VectorizedSolidVelocity:
    """
    Vectorized solid velocity computer that handles velocity field computation.
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
        
        # Initialize pressure solver with flexible parameter extraction
        self.pressure_solver = self._create_pressure_solver(cfg)
    
    def compute_solid_velocity(self, phi_hat, nutrient_field, dx, energy_deriv=None):
        """
        Compute solid velocity field.
        
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
        
        # Solve for pressure using the stacked phi_hat
        pressure = self.pressure_solver.solve_pressure(
            phi_hat, nutrient_field, dx, energy_deriv=energy_deriv
        )
        
        # Store pressure in field manager if available
        if self.field_manager is not None:
            self.field_manager.pressure = pressure
        
        # Compute pressure gradient
        grad_p_x, grad_p_y, grad_p_z = gradient(pressure, dx)
        
        # Compute gradients of total cell density
        grad_C_x, grad_C_y, grad_C_z = gradient(phi_T, dx)
        
        # Compute velocity: u = -(∇p + (δE/δφ_T) ∇φ_T)
        ux = -(grad_p_x + energy_deriv * grad_C_x)
        uy = -(grad_p_y + energy_deriv * grad_C_y)
        uz = -(grad_p_z + energy_deriv * grad_C_z)
        
        # Clip velocities for stability
        ux = np.clip(ux, -1, 1)
        uy = np.clip(uy, -1, 1)
        uz = np.clip(uz, -1, 1)
        
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
