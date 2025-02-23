
import numpy as np
from scipy.ndimage import laplace

class DiffusionDynamics:
    def __init__(self, model):
        self.model = model


    def compute_nutrient_diffusion(self, state: dict = None) -> np.ndarray:
        """
        Compute the time derivative for the nutrient field due to diffusion and consumption.
        """
        if state is not None:
            nutrient = state['nutrient']
            # Total cell density from state (all cell types)
            C_T = state['C_S'] + state['C_P'] + state['C_D'] + state['C_N']
        else:
            nutrient = self.model.nutrient
            C_T = self.model.C_T

        D = self.model.params['D_n']
        consumption_rate = 0.1

        diffusion_deriv = D * laplace(nutrient) - consumption_rate * C_T * nutrient
        return diffusion_deriv
    

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
