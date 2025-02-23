####################################################################################
####################################################################################
#
#
#   Cell Production Model
#
#   This model computes the production of the cell concentration fields
#   based on the nutrient concentration and the drug concentration.
#
#   The production of the cell concentration fields is defined as follows:
#
#   dC = Src_S + Src_P + Src_D + Src_N 
#   dt
#
#   where Src_S, Src_P, Src_D, and Src_N are the source terms for the stem cells,
#   progenitor cells, differentiated cells, and necrotic cells, respectively.
#
#   The source terms are defined and computed using the following formulas::
#
#   Src_S = λ_S n C_S (2p_0 -1) - μ_S H(n_S - n) C_S
#
#   Src_P = λ_S n 2(1 - p_0) C_S + λ_P n C_P (2p_1 - 1) - μ_P H(n_P - n) C_P
#
#   Src_D = λ_P n 2(1 - p_1) C_P - μ_D H(n_D - n) C_D - α_D C_D
#
#   Src_N = μ_S H(n_S - n) C_S + μ_P H(n_P - n) C_P + μ_D H(n_D - n) C_D + α_D C_D - γ_N C_N
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
#   model as an argument, will be able to update the concentration fields of the tumor
#   based on the source terms.
#
#   This class should be used in the following way:
#
#   <code>
#
#   model = TumorGrowthModel()
#   cell_production = ProductionModel(model)
#   cell_production.update_cell_sources()
#
#   </code>
#
#
#   This will directly update the concentration fields of the tumor model.
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


class ProductionModel:

    def __init__(self, model, drug_model = None):
        self.model = model
        self.drug_model = drug_model


    def compute_cell_sources(self, state: dict = None) -> dict:
        """
        Compute the source term derivatives for each cell population.
        If a state dict is provided, use it instead of self.model fields.
        """
        if state is not None:
            n = state['nutrient']
            C_S = state['C_S']
            C_P = state['C_P']
            C_D = state['C_D']
            C_N = state['C_N']
        else:
            n = self.model.nutrient
            C_S = self.model.C_S
            C_P = self.model.C_P
            C_D = self.model.C_D
            C_N = self.model.C_N

        lambda_S = self.model.params['lambda_S']
        p_0 = self.model.params['p_0']
        mu_S = self.model.params['mu_S']
        src_S = lambda_S * n * C_S * (2 * p_0 - 1) - mu_S * np.heaviside(self.model.n_S - n, 0) * C_S

        lambda_P = self.model.params['lambda_P']
        p_1 = self.model.params['p_1']
        mu_P = self.model.params['mu_P']
        src_P = lambda_S * n * 2 * (1 - p_0) * C_S + lambda_P * n * C_P * (2 * p_1 - 1) - mu_P * np.heaviside(self.model.n_P - n, 0) * C_P

        mu_D = self.model.params['mu_D']
        alpha_D = self.model.params['alpha_D']
        src_D = lambda_P * n * 2 * (1 - p_1) * C_P - mu_D * np.heaviside(self.model.n_D - n, 0) * C_D - alpha_D * C_D

        gamma_N = self.model.params['gamma_N']
        src_N = (mu_S * np.heaviside(self.model.n_S - n, 0) * C_S +
                 mu_P * np.heaviside(self.model.n_P - n, 0) * C_P +
                 mu_D * np.heaviside(self.model.n_D - n, 0) * C_D +
                 alpha_D * C_D - gamma_N * C_N)
        src_N = 0

        # These are the time derivatives (dC/dt) for each field.
        return {'C_S': src_S, 'C_P': src_P, 'C_D': src_D, 'C_N': src_N}

    def apply_cell_sources(self):
        """
        Compute source terms for cell populations based on growth dynamics.
        """
   

        C_S, C_P, C_D, C_N = self.model.C_S, self.model.C_P, self.model.C_D, self.model.C_N

        # Stem Cell Source
        C_S += self.model.dt * (self._compute_src_S())

        # Progenitor Cell Source
        C_P += self.model.dt * (self._compute_src_P())

        # Differentiated Cell Source
        C_D += self.model.dt * (self._compute_src_D())

        # Necrotic Cell Source
        #C_N += self.model.dt * (self._compute_src_N())

        self.model.C_S = C_S
        self.model.C_P = C_P
        self.model.C_D = C_D
        self.model.C_N = C_N

   
    def _compute_src_S(self) -> np.ndarray:
        """
        Compute the source term for the stem cells.
        src_S = \lamda_S n C_S (2p_0 -1) - \mu_S H(\hat {n_S} - n) C_S
        """
        lambda_S = self.model.params['lambda_S']
        p_0 = self.model.params['p_0']
        mu_S = self.model.params['mu_S']

        n = self.model.nutrient
        C_S = self.model.C_S
        n_S = self.model.n_S
        src_S = lambda_S * n * C_S * (2 * p_0 - 1) - mu_S * np.heaviside(n_S - n, 0) * C_S    
        return src_S


    def _compute_src_P(self) -> np.ndarray:
        """
        Compute the source term for the prologenitor cells.
        src_P = \lambda_S n 2(1 - p_0) C_S + \lambda_P n C_P (2p_1 - 1) - \mu_P H(\hat {n_P} - n) C_P
        """
        lambda_S = self.model.params['lambda_S']
        lambda_P = self.model.params['lambda_P']
        n = self.model.nutrient
        p_0 = self.model.params['p_0']
        p_1 = self.model.params['p_1']
        C_S = self.model.C_S
        C_P = self.model.C_P
        mu_P = self.model.params['mu_P']
        n_P = self.model.n_P
        src_P = lambda_S * n * 2 * (1 - p_0) * C_S + lambda_P * n * C_P * (2 * p_1 - 1) - mu_P * np.heaviside(n_P - n, 0) * C_P
        return src_P


    def _compute_src_D(self) -> np.ndarray:
        """
        Compute the source term for the differentiated cells.
        src_D = \lambda_P n 2(1 - p_1) C_P - \mu_D H(\hat {n_D} - n) C_D - \alpha_D C_D
        """
        lambda_P = self.model.params['lambda_P']
        n = self.model.nutrient
        p_1 = self.model.params['p_1']
        C_P = self.model.C_P
        C_D = self.model.C_D
        mu_D = self.model.params['mu_D']
        n_D = self.model.n_D
        alpha_D = self.model.params['alpha_D']
        src_D = lambda_P * n * 2 * (1 - p_1) * C_P - mu_D * np.heaviside(n_D - n, 0) * C_D - alpha_D * C_D
        return src_D


    def _compute_src_N(self) -> np.ndarray:
        """
        Compute the source term for the nutrient.
        src_N = \mu_S H(\hat {n_S} - n) C_S + \mu_P H(\hat {n_P} - n) C_P + \mu_D H(\hat {n_D} - n) C_D + \alpha_D C_D - \gamma_N C_N
        """
        n_S = self.model.n_S
        n_P = self.model.n_P
        n_D = self.model.n_D

        n = self.model.nutrient

        C_S = self.model.C_S
        C_P = self.model.C_P
        C_D = self.model.C_D
        C_N = self.model.C_N
        mu_S = self.model.params['mu_S']
        mu_P = self.model.params['mu_P']
        mu_D = self.model.params['mu_D']
        alpha_D = self.model.params['alpha_D']
        gamma_N = self.model.params['gamma_N']
        src_N = mu_S * np.heaviside(n_S - n, 0) * C_S + mu_P * np.heaviside(n_P - n, 0) * C_P + mu_D * np.heaviside(n_D - n, 0) * C_D + alpha_D * C_D - gamma_N * C_N
        return src_N


    def _compute_src_T(self) -> np.ndarray:
        """
        Compute the source term for the total cell density.
        src_T = \lambda_S n C_S + \lambda_P n C_P + \gamma_N C_N
        """
        lambda_S = self.model.params['lambda_S']
        lambda_P = self.model.params['lambda_P']
        n = self.model.nutrient
        C_S = self.model.C_S
        C_P = self.model.C_P
        C_N = self.model.C_N
        gamma_N = self.model.params['gamma_N']
        src_T = lambda_S * n * C_S + lambda_P * n * C_P - gamma_N * C_N
        src_T = self._compute_src_N() + self._compute_src_S() + self._compute_src_P() + self._compute_src_D()
        return src_T


    def _compute_p0(self):
        """
        Compute the p0 parameter based on the nutrient concentration.
        """
        if self.drug_model is not None:
            # this is the area where the effects of the drug can be implemented
            pass
        else:
            # a more complex application of p_0 can be implemented here
            p_0 = self.model.params['p_0']
            return p_0


    def _compute_p1(self):
        """
        Compute the p1 parameter based on the nutrient concentration.
        """
        if self.drug_model is not None:
            # this is the area where the effects of the drug can be implemented
            pass
        else:
            # a more complex application of p_1 can be implemented here
            p_1 = self.model.params['p_1']
            return p_1

