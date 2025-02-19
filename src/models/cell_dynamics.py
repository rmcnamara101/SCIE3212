# models/cell_dynamics.py
import numpy as np

class CellDynamics:
    def __init__(self, model):
        self.model = model

    def update_cell_sources(self):
        """Compute source terms for cell populations based on growth dynamics."""
        params = self.model.params
        n = self.model.nutrient
        C_S, C_P, C_D = self.model.C_S, self.model.C_P, self.model.C_D

        # Stem Cell Source
        self.model.C_S += self.model.dt * (
            params['lambda_S'] * n * C_S * (2 * params['p_0'] - 1) -
            params['mu_S'] * np.heaviside(self.model.n_S - n, 0) * C_S
        )

        # Progenitor Cell Source
        self.model.C_P += self.model.dt * (
            params['lambda_S'] * n * 2 * (1 - params['p_0']) * C_S +
            params['lambda_P'] * n * C_P * (2 * params['p_1'] - 1) -
            params['mu_P'] * np.heaviside(self.model.n_P - n, 0) * C_P
        )

        # Differentiated Cell Source
        self.model.C_D += self.model.dt * (
            params['lambda_P'] * n * 2 * (1 - params['p_1']) * C_P -
            params['mu_D'] * np.heaviside(self.model.n_D - n, 0) * C_D -
            params['alpha_D'] * C_D
        )
