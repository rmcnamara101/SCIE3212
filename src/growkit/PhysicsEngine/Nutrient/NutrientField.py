"""
Nutrient Field Module

This module handles nutrient diffusion and consumption in the tumor growth simulation.
It provides a clean interface for updating the nutrient field based on cell populations.

Author: Riley Jae McNamara
Date: 2025-02-19
"""

import numpy as np
from numba import njit
from src.growkit.MathEngine.Operators import isotropic_laplacian


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
        self.nutrient_threshold = [0.0] * self.M
        for pop in self.pops:
            self.consumption_rate[self.labels.index(pop)] = float(self.pops[pop].get('dynamics', {}).get('nutrient_consumption', 0.0))
            self.production_rate[self.labels.index(pop)] = float(self.pops[pop].get('dynamics', {}).get('nutrient_production', 0.0))
            self.nutrient_threshold[self.labels.index(pop)] = float(self.pops[pop].get('dynamics', {}).get('nutrient_threshold', 0.0))
        self.boundary_value = float(cfg.get('nutrient', {}).get('boundary_value', 1.0))
        
        # Extract consumption rates for each population
        self.population_consumption_rates = {}
        for label, pop_data in populations.items():
            self.population_consumption_rates[label] = float(pop_data.get('dynamics', {}).get('nutrient_consumption', 0.1))
    
    def initialize_nutrient_field(self, shape, phi_hat=None, initialization_type="uniform"):
        """
        Initialize nutrient field with different strategies.
        
        Args:
            shape: Grid shape (nx, ny, nz)
            phi_hat: Cell fraction fields (M, nx, ny, nz) - required for natural initialization
            initialization_type: Type of initialization ("uniform", "natural", "gradient")
            
        Returns:
            nutrient_field: Initial nutrient concentration field
        """
        nx, ny, nz = shape
        
        if initialization_type == "uniform":
            # Always create uniform nutrient field with value 1.0
            nutrient_field = np.ones(shape, dtype=np.float32)
        elif initialization_type == "natural" and phi_hat is not None:
            # Create natural nutrient field based on cell density
            nutrient_field = self._create_natural_nutrient_field(phi_hat)
        elif initialization_type == "gradient":
            # Create a simple radial gradient
            nutrient_field = self._create_radial_gradient_field(shape)
        else:
            # Default to uniform
            nutrient_field = np.ones(shape, dtype=np.float32)
        
        return nutrient_field
    
    def _create_natural_nutrient_field(self, phi_hat):
        """
        Create a natural nutrient field that accounts for cell consumption.
        
        This creates a nutrient field where:
        - Boundary has maximum concentration (1.0)
        - Center has reduced concentration based on cell density
        - Accounts for diffusion and consumption balance
        
        Args:
            phi_hat: Cell fraction fields (M, nx, ny, nz)
            
        Returns:
            nutrient_field: Natural nutrient concentration field
        """
        nx, ny, nz = phi_hat.shape[1:]
        nutrient_field = np.ones((nx, ny, nz), dtype=np.float32)
        
        # Compute total cell density
        total_cell_density = np.sum(phi_hat, axis=0)
        
        # Create a more realistic nutrient field
        # Start with boundary conditions and solve a simplified steady-state
        # This approximates the balance between diffusion and consumption
        
        # Parameters for natural nutrient field - adjusted for stronger effect
        max_consumption_effect = 0.5  # Maximum reduction in center (50% reduction = 0.5 nutrient)
        consumption_length_scale = min(nx, ny, nz) / 4  # Shorter length scale for steeper gradient
        
        # Create distance field from center (inverse of distance from boundary)
        center = np.array([nx//2, ny//2, nz//2])
        max_radius = np.sqrt(np.sum(center**2))
        
        distance_from_center = np.zeros((nx, ny, nz), dtype=np.float32)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # Distance from center
                    dist = np.sqrt((i - center[0])**2 + (j - center[1])**2 + (k - center[2])**2)
                    distance_from_center[i, j, k] = dist
        
        # Create consumption mask based on cell density
        consumption_mask = total_cell_density > 0.1  # Only consider areas with significant cell density
        
        # Add spatial noise to break up linear patterns
        # Create noise field with spatial correlation
        noise_scale = 0.15  # Amplitude of noise (15% variation)
        noise_length_scale = min(nx, ny, nz) / 16  # Spatial correlation length
        
        # Generate correlated noise using simple spatial filtering
        raw_noise = np.random.normal(0, 1, (nx, ny, nz)).astype(np.float32)
        noise_field = np.zeros_like(raw_noise)
        
        # Apply simple spatial smoothing to create correlated noise
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # Simple 3D averaging kernel for spatial correlation
                    i_start = max(0, i - 2)
                    i_end = min(nx, i + 3)
                    j_start = max(0, j - 2)
                    j_end = min(ny, j + 3)
                    k_start = max(0, k - 2)
                    k_end = min(nz, k + 3)
                    
                    noise_field[i, j, k] = np.mean(raw_noise[i_start:i_end, j_start:j_end, k_start:k_end])
        
        # Normalize noise to [-1, 1] range
        noise_field = noise_field / (np.max(np.abs(noise_field)) + 1e-8)
        
        # Apply natural nutrient reduction with stronger gradient effect and noise
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if consumption_mask[i, j, k]:
                        # Calculate consumption effect based on cell density and distance from center
                        cell_density_factor = total_cell_density[i, j, k]
                        
                        # Create gradient: closer to center = more consumption
                        # Normalize distance from center (0 at center, 1 at boundary)
                        normalized_distance = distance_from_center[i, j, k] / max_radius
                        
                        # Apply stronger consumption in center with smooth gradient outward
                        # This creates ~0.5 nutrient at center, ~1.0 at boundaries
                        reduction = max_consumption_effect * cell_density_factor * (1.0 - normalized_distance)
                        
                        # Add noise to break up linear patterns
                        noise_factor = 1.0 + noise_scale * noise_field[i, j, k]
                        
                        # Apply noise to the reduction factor
                        nutrient_field[i, j, k] = max(0.1, 1.0 - reduction * noise_factor)
                    else:
                        # Areas without cells maintain full nutrient with slight noise
                        noise_factor = 1.0 + 0.05 * noise_field[i, j, k]  # Smaller noise for empty areas
                        nutrient_field[i, j, k] = max(0.8, 1.0 * noise_factor)
        
        return nutrient_field
    
    def _create_radial_gradient_field(self, shape):
        """
        Create a simple radial gradient nutrient field.
        
        Args:
            shape: Grid shape (nx, ny, nz)
            
        Returns:
            nutrient_field: Radial gradient nutrient field
        """
        nx, ny, nz = shape
        nutrient_field = np.ones((nx, ny, nz), dtype=np.float32)
        
        center = np.array([nx//2, ny//2, nz//2])
        max_radius = np.sqrt(np.sum(center**2))
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # Calculate distance from center
                    dist = np.sqrt((i - center[0])**2 + (j - center[1])**2 + (k - center[2])**2)
                    
                    # Create gradient: higher at boundaries, lower at center
                    gradient_factor = dist / max_radius
                    nutrient_field[i, j, k] = 0.2 + 0.8 * gradient_factor
        
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
        nutrient_threshold = np.array([self.nutrient_threshold[self.labels.index(label)] for label in self.labels], dtype=np.float32)
        production_rate = np.array([self.production_rate[self.labels.index(label)] for label in self.labels], dtype=np.float32)
        
        # Compute nutrient update using Numba-optimized function
        return compute_nutrient_update_numba(
            nutrient_field, phi_hat, dx,
            self.diffusion_coeff, production_rate, consumption_rates, nutrient_threshold
        )


@njit
def compute_nutrient_update_numba(nutrient_field, phi_hat, dx, 
                                diffusion_coeff, production_rate, consumption_rates, nutrient_threshold):
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
    phi_T = np.sum(phi_hat, axis=0)
    cutoff = 0.1
    penetration_depth = 0.03
    
    # Initialize new nutrient field
    nutrient_field_new = np.zeros_like(nutrient_field, dtype=np.float32)
    
    # Compute diffusion term using isotropic laplacian
    # Since this is a Numba function, we'll compute a simplified isotropic laplacian inline
    diffusion_field = np.zeros_like(nutrient_field, dtype=np.float32)
    
    # Compute isotropic laplacian for interior points
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            for k in range(1, nz-1):
                # Axis-aligned components
                lap_axis = (
                    (nutrient_field[i+1, j, k] - 2*nutrient_field[i, j, k] + nutrient_field[i-1, j, k]) +
                    (nutrient_field[i, j+1, k] - 2*nutrient_field[i, j, k] + nutrient_field[i, j-1, k]) +
                    (nutrient_field[i, j, k+1] - 2*nutrient_field[i, j, k] + nutrient_field[i, j, k-1])
                ) / dx**2
                
                # Diagonal components for isotropy (simplified)
                lap_diag = (
                    (nutrient_field[i+1, j+1, k] - 2*nutrient_field[i, j, k] + nutrient_field[i-1, j-1, k]) +
                    (nutrient_field[i+1, j-1, k] - 2*nutrient_field[i, j, k] + nutrient_field[i-1, j+1, k]) +
                    (nutrient_field[i+1, j, k+1] - 2*nutrient_field[i, j, k] + nutrient_field[i-1, j, k-1]) +
                    (nutrient_field[i+1, j, k-1] - 2*nutrient_field[i, j, k] + nutrient_field[i-1, j, k+1]) +
                    (nutrient_field[i, j+1, k+1] - 2*nutrient_field[i, j, k] + nutrient_field[i, j-1, k-1]) +
                    (nutrient_field[i, j+1, k-1] - 2*nutrient_field[i, j, k] + nutrient_field[i, j-1, k+1])
                ) / (2.0 * dx**2) * 0.1  # Small weight for diagonal terms
                
                diffusion_field[i, j, k] = diffusion_coeff * (lap_axis + lap_diag)
    
    # Boundary points have zero diffusion (Neumann conditions for diffusion operator)
    # This is correct - diffusion operator should have zero flux at boundaries
    diffusion_field[0, :, :] = 0.0
    diffusion_field[-1, :, :] = 0.0
    diffusion_field[:, 0, :] = 0.0
    diffusion_field[:, -1, :] = 0.0
    diffusion_field[:, :, 0] = 0.0
    diffusion_field[:, :, -1] = 0.0
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                diffusion = diffusion_field[i, j, k]
                
                # Compute consumption term
                consumption = 0.0
                for m in range(M):
                    # Each population consumes nutrient based on its density and consumption rate
                    # Use a simple linear consumption model
                    consumption += consumption_rates[m] * phi_hat[m, i, j, k]

                # Compute production term
                production = 0.0
                for m in range(M):
                    production += production_rate[m] * (1 - phi_T[i, j, k]) * (1 - nutrient_field[i, j, k])

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
    
    # Use isotropic laplacian for diffusion
    diffusion_field = isotropic_laplacian(nutrient_field, dx)
    
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            for k in range(1, nz-1):
                # Diffusion term using isotropic laplacian
                diffusion = D * diffusion_field[i, j, k]
                
                # Consumption term
                consumption = alpha * phi_T[i, j, k] * nutrient_field[i, j, k]
                
                # Update
                nutrient_field_new[i, j, k] = nutrient_field[i, j, k] + diffusion - consumption
                nutrient_field_new[i, j, k] = max(0.0, nutrient_field_new[i, j, k])
    
    # Set boundary conditions (Dirichlet) - this is the correct physics for nutrient fields
    # Nutrient fields should have fixed values at boundaries (source of nutrients)
    nutrient_field_new[0, :, :] = 1.0
    nutrient_field_new[-1, :, :] = 1.0
    nutrient_field_new[:, 0, :] = 1.0
    nutrient_field_new[:, -1, :] = 1.0
    nutrient_field_new[:, :, 0] = 1.0
    nutrient_field_new[:, :, -1] = 1.0
    
    return nutrient_field_new
