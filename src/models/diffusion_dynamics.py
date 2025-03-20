import numpy as np
import numba as nb
from scipy.ndimage import laplace

from src.utils.utils import laplacian

@nb.njit
def compute_nutrient_diffusion(C_S, C_P, C_D, C_N, nutrient, dx, D_n, boundary_value):
    consumption_rate = 2.0
    C_T = C_S + C_P + C_D + C_N
    
    # Create a copy of nutrient to avoid modifying the original
    nutrient_with_boundary = nutrient.copy()
    
    # Apply Dirichlet boundary conditions (fixed value at boundaries)
    nutrient_with_boundary[0, :] = boundary_value  # Top boundary
    nutrient_with_boundary[-1, :] = boundary_value  # Bottom boundary
    nutrient_with_boundary[:, 0] = boundary_value  # Left boundary
    nutrient_with_boundary[:, -1] = boundary_value  # Right boundary
    
    d_nutrient = D_n * laplacian(nutrient_with_boundary, dx) - consumption_rate * C_T * nutrient
    return np.clip(d_nutrient, -100.0, 100.0)  # Prevent extreme nutrient changes

@nb.njit
def compute_nutrient_diffusion_scie3121_model(phi_H, phi_D, phi_N, nutrient, dx, D_n, boundary_value):
    # Adjust these parameters - they may be causing issues
    consumption_rate = 0.1
    production_rate = 1.0
    microenvironment_nutrient_saturation = 1.0
    phi_T = phi_H + phi_D + phi_N  
    
    # Create a copy of nutrient to avoid modifying the original
    nutrient_with_boundary = nutrient.copy()
    
    # Apply Dirichlet boundary conditions (fixed value at boundaries)
    nutrient_with_boundary[0, :] = boundary_value  # Top boundary
    nutrient_with_boundary[-1, :] = boundary_value  # Bottom boundary
    nutrient_with_boundary[:, 0] = boundary_value  # Left boundary
    nutrient_with_boundary[:, -1] = boundary_value  # Right boundary
    
    # Modified equation to ensure nutrient doesn't drop too low in cell-dense regions
    d_nutrient = D_n * laplacian(nutrient_with_boundary, dx) - consumption_rate * phi_T * nutrient + production_rate * (1 - phi_T) * (microenvironment_nutrient_saturation - nutrient)
    
    # Add a minimum nutrient level to prevent complete depletion
    min_nutrient = 0.01
    d_nutrient = np.where(nutrient < min_nutrient, 
                          np.maximum(d_nutrient, 0.0),  # Only allow positive changes when nutrient is low
                          d_nutrient)
    
    return np.clip(d_nutrient, -10.0, 10.0)  # More conservative clipping


class DiffusionDynamics:
    def __init__(self, model):
        self.model = model

    def compute_nutrient_diffusion(self, phi_H, phi_D, phi_N, nutrient, params):
        """Wrapper to call the static Numba-optimized function."""
        boundary_value = params.get('boundary_nutrient', 1.0)  # Default to 1.0 if not specified
        return compute_nutrient_diffusion(phi_H, phi_D, phi_N, nutrient, self.model.dx, params['D_n'], boundary_value)
    

    def apply_nutrient_diffusion(self):
        """
        Apply nutrient diffusion to the nutrient field with Dirichlet boundary conditions.
        """
        nutrient = self.model.nutrient.copy()
        D = self.model.params['D_n']
        consumption_rate = 2.0
        boundary_value = self.model.params.get('boundary_nutrient', 1.0)  # Default to 1.0 if not specified
        
        # Apply Dirichlet boundary conditions
        nutrient_with_boundary = nutrient.copy()
        nutrient_with_boundary[0, :] = boundary_value  # Top boundary
        nutrient_with_boundary[-1, :] = boundary_value  # Bottom boundary
        nutrient_with_boundary[:, 0] = boundary_value  # Left boundary
        nutrient_with_boundary[:, -1] = boundary_value  # Right boundary
        
        # Compute Laplacian with boundary conditions applied
        nutrient_laplacian = laplace(nutrient_with_boundary)
        
        # Update interior points only (boundaries remain fixed)
        interior = np.ones_like(nutrient, dtype=bool)
        interior[0, :] = interior[-1, :] = interior[:, 0] = interior[:, -1] = False
        
        nutrient[interior] += self.model.dt * (D * nutrient_laplacian[interior] - 
                                     consumption_rate * self.model.C_T[interior] * nutrient[interior])
        
        self.model.nutrient = nutrient

class SCIE3121DiffusionModel(DiffusionDynamics):

    def __init__(self, model):
        super().__init__(model)

    def compute_nutrient_diffusion(self, phi_H, phi_D, phi_N, nutrient, params):
        boundary_value = params.get('boundary_nutrient', 1.0)  # Default to 1.0 if not specified
        return compute_nutrient_diffusion_scie3121_model(
            phi_H, phi_D, phi_N, nutrient, self.model.dx, params['D_n'], boundary_value
        )
    
    