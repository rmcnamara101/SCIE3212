####################################################################################
####################################################################################
#
#
#   Cell Production Model
#
#   This model computes the production of the cell volume fraction fields
#   based on the nutrient concentration and the drug concentration.
#
#   The production of the cell concentration fields is defined as follows:
#
#   dC = Src_H + Src_P + Src_D + Src_N 
#   dt
#
#   where Src_h, Src_P, Src_D, and Src_N are the source terms for the stem cells,
#   progenitor cells, differentiated cells, and necrotic cells, respectively.
#
#   The source terms are defined and computed using the following formulas::
#
#   Src_H = λ_S n phi_H (2p_0 -1) - μ_S H(n_S - n) phi_H
#
#   Src_P = λ_S n 2(1 - p_0) phi_H + λ_P n phi_P (2p_1 - 1) - μ_P H(n_P - n) phi_P
#
#   Src_D = λ_P n 2(1 - p_1) phi_P - μ_D H(n_D - n) phi_D - α_D phi_D
#
#   Src_N = μ_S H(n_S - n) phi_H + μ_P H(n_P - n) phi_P + μ_D H(n_D - n) C_phi + α_D phi_D - γ_N phi_N
#
#   where: λ terms refer to cell proliferation, the μ terms refer to cell death,
#   the α terms refer to cell differentiation, and the γ term refers to necrotic cell dissolution.
#
#   The p_0 and p_1 parameters are refer to the probability of cell division, which can be easily 
#   implemented as a function of chemical, nutrient, and drug concentrations. But also naturally 
#   occuring inhibitors, promoters, and other cell cycle control mechanisms.
#
#
#   This file should be used as an importable class, that will take a tumor growth 
#   model as an argument, will be able to calculate the derivatives of the volume fraction
#   fields of the tumor based on the source terms.
#
#   This class should be used in the following way:
#
#   <code>
#
#   model = TumorGrowthModel()
#   cell_production = ProductionModel(model)
#   cell_production.compute_cell_sources()
#
#   </code>
#
#
#   This will return a set of derivatives that can be applied to the volume fraction fields.
#
#
# Author:
#   - Riley Jae McNamara
#
# Date:
#   - 2025-02-19
#
####################################################################################
####################################################################################

import numpy as np
import numba as nb

@nb.njit
def compute_cell_sources(phi_H, phi_P, phi_D, phi_N, nutrient, n_S, n_P, n_D, lambda_S, lambda_P, mu_S, mu_P, mu_D, alpha_D, p_0, p_1, gamma_N):
    H_S = np.where(n_S - nutrient > 0, 1.0, 0.0)
    H_P = np.where(n_P - nutrient > 0, 1.0, 0.0)
    H_D = np.where(n_D - nutrient > 0, 1.0, 0.0)

    src_S = lambda_S * nutrient * phi_H * (2 * p_0 - 1) - mu_S * H_S * phi_H
    src_P = lambda_S * nutrient * 2 * (1 - p_0) * phi_H + lambda_P * nutrient * phi_P * (2 * p_1 - 1) - mu_P * H_P * phi_P
    src_D = lambda_P * nutrient * 2 * (1 - p_1) * phi_P - mu_D * H_D * phi_D - alpha_D * phi_D
    src_N = mu_S * H_S * phi_H + mu_P * H_P * phi_P + mu_D * H_D * phi_D + alpha_D * phi_D - gamma_N * phi_N

    # Tighter clipping to prevent runaway growth
    return (
        np.clip(src_S, -50.0, 50.0),
        np.clip(src_P, -50.0, 50.0),
        np.clip(src_D, -50.0, 50.0),
        np.clip(src_N, -50.0, 50.0)
    )

@nb.njit
def compute_cell_sources_scie3121_model(phi_H, phi_D, phi_N, nutrient, n_H, n_D, lambda_H, lambda_D, mu_H, mu_D, p_H, p_D, mu_N):
    # Make sure these thresholds are appropriate
    H_H = np.where(n_H - nutrient > 0, 1.0, 0.0)
    H_D = np.where(n_D - nutrient > 0, 1.0, 0.0)

    
    # Ensure proper balance between growth and death
    src_H = lambda_H * nutrient * phi_H * (2 * p_H - 1) - mu_H * H_H * phi_H
    src_D = 2 * lambda_H * nutrient * (1 - p_H) * phi_H + lambda_D * nutrient * phi_D * (2 * p_D - 1) - mu_D * H_D * phi_D 
    
    # Make sure cells are properly transitioning to necrotic state
    src_N = mu_H * H_H * phi_H + mu_D * H_D * phi_D - mu_N * phi_N
    
    # Debug: print max values to check for extreme changes
    # print(f"Max src_H: {np.max(np.abs(src_H))}, Max src_D: {np.max(np.abs(src_D))}, Max src_N: {np.max(np.abs(src_N))}")
    
    # More conservative clipping
    return (
        src_H,
        src_D,
        src_N
    )

def compute_pressure_cell_sources(phi_H, phi_D, phi_N, nutrient, n_H, n_D, lambda_H, lambda_D, mu_H, mu_D, p_H, p_D, mu_N):
    
    H_H = np.where(n_H - nutrient > 0, 1.0, 0.0)
    H_D = np.where(n_D - nutrient > 0, 1.0, 0.0)

    src_H = lambda_H * nutrient * phi_H * (2 * p_H - 1)
    src_D = 2 * lambda_H * nutrient * (1 - p_H) * phi_H + lambda_D * nutrient * phi_D * (2 * p_D - 1)
    src_N = mu_N * phi_N
    
    return (src_H, src_D, src_N)
    
    

class ProductionModel:

    def __init__(self, model, drug_model = None):
        self.model = model
        self.drug_model = drug_model


    def compute_cell_sources(self, phi_H, phi_P, phi_D, phi_N, nutrient, n_S, n_P, n_D, params):
        """Wrapper to call the Numba-optimized function."""
        return compute_cell_sources(
            phi_H, phi_P, phi_D, phi_N, nutrient, n_S, n_P, n_D,
            params['lambda_S'], params['lambda_P'], params['mu_S'], params['mu_P'],
            params['mu_D'], params['alpha_D'], params['p_0'], params['p_1'], params['gamma_N']
        )

class SCIE3121_MODEL:

    def __init__(self, model, drug_model = None):
        self.model = model
        self.drug_model = drug_model
        self.params = self.model.params

    def compute_cell_sources(self, phi_H, phi_D, phi_N, nutrient, n_H, n_D, params):
        """Wrapper to call the Numba-optimized function."""
        return compute_cell_sources_scie3121_model(
            phi_H, phi_D, phi_N, nutrient,
            n_H, n_D, params['lambda_H'], params['lambda_D'],
            params['mu_H'], params['mu_D'], params['p_H'], params['p_D'],
            params['mu_N']
        )