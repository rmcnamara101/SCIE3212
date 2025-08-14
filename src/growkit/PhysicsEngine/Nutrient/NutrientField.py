"""
Nutrient Field Module

This module handles nutrient diffusion and consumption in the tumor growth simulation.
It provides a clean interface for updating the nutrient field based on cell populations.

Author: Riley Jae McNamara
Date: 2025-02-19
"""

import numpy as np
from numba import njit
from src.growkit.MathEngine.Operators import _gradient_neumann


class NutrientField:
    """
    Nutrient field manager that handles diffusion and consumption.
    """
    
    def __init__(self, cfg, populations):
        """
        Initialize the nutrient field manager.
        
        Args:
            cfg: Configuration dictionary
            populations: Population definitions from YAML
        """
        self.cfg = cfg
        self.pops = populations
        self.labels = list(populations.keys())
        self.M = len(self.labels)
        
        # Extract nutrient parameters from config
        self.diffusion_coeff = float(cfg["nutrient"]["dynamics"]["diffusion"])
        self.consumption_rate = [0.0] * self.M
        self.production_rate = [0.0] * self.M
        for pop in self.pops:
            self.consumption_rate[self.labels.index(pop)] = float(self.pops[pop].get('dynamics', {}).get('nutrient_consumption', 0.0))
            self.production_rate[self.labels.index(pop)] = float(self.pops[pop].get('dynamics', {}).get('nutrient_production', 0.0))
        self.boundary_value = float(cfg.get('nutrient', {}).get('boundary_value', 1.0))
        
        # Extract consumption rates for each population
        self.population_consumption_rates = {}
        for label, pop_data in populations.items():
            self.population_consumption_rates[label] = float(pop_data.get('dynamics', {}).get('nutrient_consumption', 0.1))
    
    def initialize_nutrient_field(self, shape):
        """
        Initialize nutrient field with uniform concentration of 1.0.
        
        Args:
            shape: Grid shape (nx, ny, nz)
            dx: Grid spacing
            
        Returns:
            nutrient_field: Initial nutrient concentration field (uniform at 1.0)
        """
        nx, ny, nz = shape
        
        # Always create uniform nutrient field with value 1.0
        nutrient_field = np.ones(shape, dtype=np.float32)
        
        return nutrient_field
    
    def compute_nutrient_update(self, phi_hat, nutrient_field, dx):
        """
        Compute nutrient field update due to diffusion and consumption.
        
        Args:
            phi_hat: Stacked cell fraction fields (M, nx, ny, nz)
            nutrient_field: Current nutrient concentration field
            dx: Grid spacing
            
        Returns:
            nutrient_field_new: Updated nutrient concentration field
        """
        
        # Extract consumption rates as a simple array for Numba compatibility
        consumption_rates = np.array([self.population_consumption_rates[label] for label in self.labels], dtype=np.float32)
        
        # Compute nutrient update using Numba-optimized function
        return compute_nutrient_update_numba(
            nutrient_field, phi_hat, dx,
            self.diffusion_coeff, np.array(self.production_rate, dtype=np.float32), consumption_rates
        )


@njit
def compute_nutrient_update_numba(nutrient_field, phi_hat, dx, 
                                diffusion_coeff, production_rate, consumption_rates):
    """
    Compute nutrient field update using Numba optimization. 
    
    Args:
        nutrient_field: Current nutrient concentration field
        phi_T: Total cell density field
        phi_hat: Stacked cell fraction fields (M, nx, ny, nz)
        dx: Grid spacing
        diffusion_coeff: Nutrient diffusion coefficient
        consumption_rate: Base nutrient consumption rate
        production_rate: Base nutrient production rate
        consumption_rates: Array of consumption rates per population
        
    Returns:
        nutrient_field_new: Updated nutrient concentration field
    """
    nx, ny, nz = nutrient_field.shape
    M = phi_hat.shape[0]
    
    # Initialize new nutrient field
    nutrient_field_new = np.zeros_like(nutrient_field, dtype=np.float32)
    
    # Compute diffusion term (Laplacian)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Interior points
                if 1 <= i < nx-1 and 1 <= j < ny-1 and 1 <= k < nz-1:
                    # Central difference for Laplacian
                    diff_x = (nutrient_field[i+1, j, k] - 2*nutrient_field[i, j, k] + nutrient_field[i-1, j, k]) / dx**2
                    diff_y = (nutrient_field[i, j+1, k] - 2*nutrient_field[i, j, k] + nutrient_field[i, j-1, k]) / dx**2
                    diff_z = (nutrient_field[i, j, k+1] - 2*nutrient_field[i, j, k] + nutrient_field[i, j, k-1]) / dx**2
                    diffusion = diffusion_coeff * (diff_x + diff_y + diff_z)
                else:
                    # Boundary points - use Neumann conditions (no flux)
                    diffusion = 0.0
                
                # Compute consumption term
                consumption = 0.0
                for m in range(M):
                    # Each population consumes nutrient based on its density and consumption rate
                    # Use a simple linear consumption model
                    consumption += consumption_rates[m] * phi_hat[m, i, j, k]

                # Compute production term
                production = 0.0
                for m in range(M):
                    production += production_rate[m] * phi_hat[m, i, j, k]
                
                # Update nutrient field 
                nutrient_field_new[i, j, k] = nutrient_field[i, j, k] + diffusion - consumption + production
                
                # Ensure non-negative nutrient concentration
                nutrient_field_new[i, j, k] = max(0.0, nutrient_field_new[i, j, k])
    
    return nutrient_field_new


def create_simple_nutrient_function(phi_hat, nutrient_field, dx):
    """
    Simple nutrient function for backward compatibility.
    
    Args:
        phi_hat: Stacked cell fraction fields
        nutrient_field: Current nutrient concentration field
        dx: Grid spacing
        
    Returns:
        nutrient_field_new: Updated nutrient concentration field
    """
    # Simple diffusion-consumption model
    nx, ny, nz = nutrient_field.shape
    nutrient_field_new = np.zeros_like(nutrient_field)
    
    # Diffusion coefficient and consumption rate
    D = 0.1
    alpha = 0.1
    
    # Compute total cell density
    phi_T = np.sum(phi_hat, axis=0)
    
    # Simple finite difference update
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            for k in range(1, nz-1):
                # Diffusion term
                diffusion = D * (
                    (nutrient_field[i+1, j, k] - 2*nutrient_field[i, j, k] + nutrient_field[i-1, j, k]) / dx**2 +
                    (nutrient_field[i, j+1, k] - 2*nutrient_field[i, j, k] + nutrient_field[i, j-1, k]) / dx**2 +
                    (nutrient_field[i, j, k+1] - 2*nutrient_field[i, j, k] + nutrient_field[i, j, k-1]) / dx**2
                )
                
                # Consumption term
                consumption = alpha * phi_T[i, j, k] * nutrient_field[i, j, k]
                
                # Update
                nutrient_field_new[i, j, k] = nutrient_field[i, j, k] + diffusion - consumption
                nutrient_field_new[i, j, k] = max(0.0, nutrient_field_new[i, j, k])
    
    # Set boundary conditions (Dirichlet)
    nutrient_field_new[0, :, :] = 1.0
    nutrient_field_new[-1, :, :] = 1.0
    nutrient_field_new[:, 0, :] = 1.0
    nutrient_field_new[:, -1, :] = 1.0
    nutrient_field_new[:, :, 0] = 1.0
    nutrient_field_new[:, :, -1] = 1.0
    
    return nutrient_field_new
