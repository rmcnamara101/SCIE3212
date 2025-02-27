
import numpy as np
import numba as nb
from scipy.ndimage import laplace

from src.utils.utils import laplacian

@nb.njit
def compute_nutrient_diffusion(C_S, C_P, C_D, C_N, nutrient, dx, D_n):
    consumption_rate = 0.1
    C_T = C_S + C_P + C_D + C_N
    d_nutrient = D_n * laplacian(nutrient, dx) - consumption_rate * C_T * nutrient
    return np.clip(d_nutrient, -100.0, 100.0)  # Prevent extreme nutrient changes

class DiffusionDynamics:
    def __init__(self, model):
        self.model = model

    def compute_nutrient_diffusion(self, C_S, C_P, C_D, C_N, nutrient, params):
        """Wrapper to call the static Numba-optimized function."""
        return compute_nutrient_diffusion(C_S, C_P, C_D, C_N, nutrient, self.model.dx, params['D_n'])
    

    def apply_nutrient_diffusion(self):
        """
        Apply nutrient diffusion to the nutrient field.
        """

        nutrient = self.model.nutrient

        D = self.model.params['D_n']

        consumption_rate = 0.1
        
        nutrient += self.model.dt * (D * laplace(nutrient) - 
                                     consumption_rate * self.model.C_T * nutrient)
        
        self.model.nutrient = nutrient
