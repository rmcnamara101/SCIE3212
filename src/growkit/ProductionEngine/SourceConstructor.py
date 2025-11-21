import numpy as np
from numba import njit

from src.growkit.ProductionEngine.MatrixConstructor import MatrixConstructor

class SourceConstructor:
    """
    Source vector constructor that implements the unified vectorized form:
    
    A_hat = G_N * n * P * alpha_hat - D * phi_hat
    
    where:
    - A_hat: source term vector for all populations
    - G_N: global necrotic feedback factor
    - n: local nutrient field
    - P: transfer matrix (column-stochastic)
    - alpha_hat: proliferation-weighted populations (alpha_hat_i = lambda_i * phi_i)
    - D: death matrix (nutrient-modulated death rates)
    - phi_hat: stacked cell fraction fields for all M populations
    """
    
    def __init__(self, cfg, populations):
        """
        Initialize the source constructor.
        
        Args:
            cfg: Configuration dictionary
            populations: Population definitions from YAML
            matrix_constructor: MatrixConstructor instance for transfer matrix P
        """
        self.cfg = cfg
        self.pops = populations
        self.labels = list(populations.keys())
        self.M = len(self.labels)
        self.matrix_constructor = MatrixConstructor(cfg, populations)
        
        # Extract population parameters
        self._extract_population_params()
        
        # Build transfer matrix P
        self.P, self.name_to_idx = self.matrix_constructor.build_transfer_matrix()
        
        # Build death matrix D
        self.D = self._build_death_matrix()
        
        # Necrotic feedback parameters
        self.beta_N = cfg["populations"]["Necrotic"]["dynamics"]["beta_N"]
    
    def _extract_population_params(self):
        """Extract proliferation and death parameters from population definitions."""
        self.lambda_ = np.array([p["dynamics"]["lambda"] for p in self.pops.values()], dtype=np.float32)
        self.mu = np.array([p["dynamics"]["mu"] for p in self.pops.values()], dtype=np.float32)
        self.nutrient_thresholds = np.array([p["dynamics"]["nutrient_threshold"] for p in self.pops.values()], dtype=np.float32)
    
    def _build_death_matrix(self):
        """
        Build death matrix D with nutrient-modulated death rates.
        
        D[i,i] = mu_i * Theta_i for each population
        D[M-1,i] = -mu_i * Theta_i for viable populations (mass flows into necrotic)
        D[i,j] = 0 otherwise
        
        Returns:
            D: (M, M) death matrix where M includes all populations including necrotic
        """
        M = self.M
        D = np.zeros((M, M), dtype=np.float32)
        
        # For each population, set death rates
        for i, (name, pop) in enumerate(self.pops.items()):
            mu_i = self.mu[i]
            
            # Death matrix entries for this population
            D[i, i] = mu_i  # Death from population i
            D[M-1, i] = -mu_i  # Mass flows into necrotic (last row)
        
        return D
    
    def _compute_nutrient_switches(self, nutrient_field):
        """
        Compute smooth nutrient switches Theta_i for each population.
        
        Args:
            nutrient_field: Local nutrient concentration field
            
        Returns:
            Theta: (M, nx, ny, nz) array of nutrient switches (death stops when nutrient > nutrient_threshold)
        """
        k = self.cfg["nutrient"]["dynamics"]["k"]  # Steepness parameter for smooth transition
        shape = nutrient_field.shape
        Theta = np.zeros((self.M, *shape), dtype=np.float32)
        
        # Use local nutrient levels for computing switches
        for i in range(self.M):
            # Death threshold: death stops when nutrient > nutrient_threshold
            n_thresh = self.nutrient_thresholds[i]
            Theta[i] = 0.5 * (1 + np.tanh(k * (nutrient_field - n_thresh)))
            
        return Theta
    
    def _compute_necrotic_feedback(self, phi_hat):
        """
        Compute global necrotic feedback factor G_N.
        
        G_N(t) = exp(-beta_N * V_N(t))
        where V_N is the total necrotic volume
        
        Args:
            phi_hat: Stacked cell fraction fields for all populations
            
        Returns:
            G_N: Global necrotic feedback factor
        """
        # Assuming necrotic is the last population
        V_N = np.sum(phi_hat[-1])  # Total necrotic volume
        G_N = np.exp(-self.beta_N * V_N)
        return G_N
    
    def _compute_proliferation_weighted_populations(self, phi_hat):
        """
        Compute proliferation-weighted populations alpha_hat.
        
        alpha_hat_i = lambda_i * phi_i
        
        Args:
            phi_hat: Stacked cell fraction fields for all populations
            
        Returns:
            alpha_hat: Proliferation-weighted populations
        """
        alpha_hat = np.zeros_like(phi_hat)
        for i in range(self.M):
            alpha_hat[i] = self.lambda_[i] * phi_hat[i]
        return alpha_hat
    
    def compute_source_vector(self, phi_hat, nutrient_field):
        """
        Compute the complete source vector A_hat.
        
        A_hat = G_N * n * P * alpha_hat - D * phi_hat
        
        Args:
            phi_hat: Stacked cell fraction fields for all M populations
            nutrient_field: Local nutrient concentration field
            
        Returns:
            A_hat: Source term vector for all populations (including necrotic)
        """
        # Compute nutrient switches
        Theta = self._compute_nutrient_switches(nutrient_field)
        
        # Compute necrotic feedback
        G_N = self._compute_necrotic_feedback(phi_hat)
        
        # Compute proliferation-weighted populations
        alpha_hat = self._compute_proliferation_weighted_populations(phi_hat)
        
        # Apply nutrient modulation to death matrix
        # Death should happen when nutrient is BELOW death threshold (inverse of switch)
        # So we use (1 - Theta) instead of Theta for death
        D_modulated = self.D.copy()
        for i in range(self.M):
            D_modulated[i, i] *= (1.0 - Theta[i])
            D_modulated[self.M-1, i] *= (1.0 - Theta[i])  # Also modulate flow into necrotic
        
        # Compute source vector components
        # Growth term: G_N * n * P * alpha_hat
        growth_term = G_N * nutrient_field * np.tensordot(self.P, alpha_hat, axes=([1], [0]))
        
        # Death term: D * phi_hat
        death_term = np.tensordot(D_modulated, phi_hat, axes=([1], [0]))
        
        # Combine terms
        A_hat = growth_term - death_term
        
        return A_hat
    
    def compute_pressure_source_vector(self, phi_hat, nutrient_field):
        """
        Compute the pressure source vector that excludes mass transfer terms.
        
        For the pressure equation, we only want absolute growth/death terms that
        actually change the total volume, not redistribution of existing mass.
        
        Args:
            phi_hat: Stacked cell fraction fields for all M populations
            nutrient_field: Local nutrient concentration field
            
        Returns:
            pressure_source: Source term for pressure equation (scalar field)
        """
        # Compute nutrient switches
        Theta = self._compute_nutrient_switches(nutrient_field)
        
        # Compute necrotic feedback
        G_N = self._compute_necrotic_feedback(phi_hat)
        
        # Compute proliferation-weighted populations
        alpha_hat = self._compute_proliferation_weighted_populations(phi_hat)
        
        # For pressure, we only want absolute growth/death, not mass transfer
        # Growth: sum over all populations of their absolute growth
        growth_terms = np.zeros_like(nutrient_field)
        for i in range(self.M):
            # Only count growth that stays in the same population (diagonal of P)
            if i < self.P.shape[0] and i < self.P.shape[1]:
                growth_terms += G_N * nutrient_field * self.P[i, i] * alpha_hat[i]
        
        # Death terms are excluded from pressure source because:
        # - All death from viable populations flows into necrotic (via death matrix D)
        # - The necrotic pathway is a transfer term (net mass conserved)
        # - Transfer terms should not contribute to pressure
        # Therefore, pressure source only includes growth terms
        
        pressure_source = growth_terms
        
        return pressure_source
    
    def compute_pressure_source_vector_vectorized(self, phi_hat, nutrient_field):
        """
        Vectorized version of pressure source computation for efficiency.
        
        Args:
            phi_hat: Stacked cell fraction fields for all M populations
            nutrient_field: Local nutrient concentration field
            
        Returns:
            pressure_source: Source term for pressure equation (scalar field)
        """
        # Compute nutrient switches (vectorized)
        k = self.cfg["nutrient"]["dynamics"]["k"]
        if nutrient_field.ndim == 3:
            nutrient_field_expanded = nutrient_field[None, :, :, :]
        else:
            nutrient_field_expanded = nutrient_field
            
        Theta = 0.5 * (1 + np.tanh(k * (nutrient_field_expanded[:, None, :, :, :] - self.nutrient_thresholds[None, :, None, None, None])))
        
        # Compute necrotic feedback
        V_N = np.sum(phi_hat[:, -1])  # Total necrotic volume
        G_N = np.exp(-self.beta_N * V_N)  # G_N is a scalar
        
        # Compute proliferation-weighted populations
        alpha_hat = self.lambda_[None, :, None, None, None] * phi_hat[:, :self.M]
        
        # For pressure, only count absolute growth/death, not mass transfer
        # Growth: only diagonal terms of P (growth that stays in same population)
        growth_terms = np.zeros_like(nutrient_field)
        for i in range(self.M):
            if i < self.P.shape[0] and i < self.P.shape[1]:
                growth_terms += G_N * nutrient_field * self.P[i, i] * alpha_hat[:, i]
        
        # Death terms are excluded from pressure source because:
        # - All death from viable populations flows into necrotic (via death matrix D)
        # - The necrotic pathway is a transfer term (net mass conserved)
        # - Transfer terms should not contribute to pressure
        # Therefore, pressure source only includes growth terms
        
        pressure_source = growth_terms
        
        return pressure_source.squeeze()
    
    def compute_source_vector_vectorized(self, phi_hat, nutrient_field):
        """
        Vectorized version of source vector computation for efficiency.
        
        Args:
            phi_hat: Stacked cell fraction fields for all M populations (M, nx, ny, nz)
            nutrient_field: Local nutrient concentration field (nx, ny, nz)
            
        Returns:
            A_hat: Source term vector for all populations (M, nx, ny, nz)
        """
        M = self.M
        
        # Compute nutrient switches (spatially varying)
        Theta = self._compute_nutrient_switches(nutrient_field)  # (M, nx, ny, nz)
        
        # Compute necrotic feedback - always a global scalar
        V_N = np.sum(phi_hat[-1])  # Total necrotic volume (assuming necrotic is last population)
        G_N = np.exp(-self.beta_N * V_N)  # G_N is a scalar
        
        # Compute proliferation-weighted populations
        alpha_hat = self.lambda_[:, None, None, None] * phi_hat  # (M, nx, ny, nz)
        
        # Apply nutrient modulation to death matrix (spatially varying)
        # Death should happen when nutrient is BELOW death threshold (inverse of switch)
        # So we use (1 - Theta) instead of Theta for death
        # D_modulated[i,j] = D[i,j] * (1 - Theta[j]) for each spatial location
        D_modulated = np.zeros((M, M, *nutrient_field.shape), dtype=np.float32)
        for i in range(M):
            for j in range(M):
                D_modulated[i, j] = self.D[i, j] * (1.0 - Theta[j])
        
        # Compute source vector
        # Growth term: G_N * n * P * alpha_hat
        # G_N is scalar, nutrient_field is (nx, ny, nz), alpha_hat is (M, nx, ny, nz)
        # P is (M, M), we want P * alpha_hat = (M, nx, ny, nz)
        growth_term = G_N * nutrient_field[None, :, :, :] * np.tensordot(self.P, alpha_hat, axes=([1], [0]))
        
        # Death term: D_modulated * phi_hat (spatially varying)
        # D_modulated is (M, M, nx, ny, nz), phi_hat is (M, nx, ny, nz)
        # We want to compute sum_j D_modulated[i,j] * phi_hat[j] for each i and spatial location
        death_term = np.zeros_like(phi_hat)
        for i in range(M):
            for j in range(M):
                death_term[i] += D_modulated[i, j] * phi_hat[j]
        
        # Combine terms
        A_hat = growth_term - death_term
        
        return A_hat
    
    def compute_growth_and_death_separately(self, phi_hat, nutrient_field):
        """
        Compute growth and death terms separately for diagnostic purposes.
        
        This method helps understand the contribution of growth vs death to the
        source terms, which is useful for debugging and analysis.
        
        Args:
            phi_hat: Stacked cell fraction fields for all M populations (M, nx, ny, nz)
            nutrient_field: Local nutrient concentration field (nx, ny, nz)
            
        Returns:
            dict with keys:
                - 'growth_term': (M, nx, ny, nz) growth contributions for each population
                - 'death_term': (M, nx, ny, nz) death contributions for each population
                - 'net_source': (M, nx, ny, nz) net source terms (growth - death)
                - 'net_source_viable': (nx, ny, nz) net source for viable populations only (excludes necrotic)
                - 'total_growth': (nx, ny, nz) total growth across all populations
                - 'total_death': (nx, ny, nz) total death across all populations
        """
        M = self.M
        
        # Compute nutrient switches (spatially varying)
        Theta = self._compute_nutrient_switches(nutrient_field)  # (M, nx, ny, nz)
        
        # Compute necrotic feedback
        V_N = np.sum(phi_hat[-1])  # Total necrotic volume
        G_N = np.exp(-self.beta_N * V_N)
        
        # Compute proliferation-weighted populations
        alpha_hat = self.lambda_[:, None, None, None] * phi_hat
        
        # Apply nutrient modulation to death matrix
        # Death should happen when nutrient is BELOW death threshold (inverse of switch)
        # So we use (1 - Theta) instead of Theta for death
        D_modulated = np.zeros((M, M, *nutrient_field.shape), dtype=np.float32)
        for i in range(M):
            for j in range(M):
                D_modulated[i, j] = self.D[i, j] * (1.0 - Theta[j])
        
        # Compute growth and death terms
        # Growth term: G_N * n * P * alpha_hat
        growth_term = G_N * nutrient_field[None, :, :, :] * np.tensordot(self.P, alpha_hat, axes=([1], [0]))
        
        death_term = np.zeros_like(phi_hat)
        for i in range(M):
            for j in range(M):
                death_term[i] += D_modulated[i, j] * phi_hat[j]
        
        # Compute net source
        net_source = growth_term - death_term
        
        # For analysis, exclude necrotic population (last population) when computing net viable source
        # This gives the net growth/death for viable cells only
        net_source_viable = np.sum(net_source[:-1], axis=0) if M > 1 else net_source[0]
        
        # Total growth and death (summed across all populations)
        total_growth = np.sum(growth_term, axis=0)
        total_death = np.sum(death_term, axis=0)
        
        return {
            'growth_term': growth_term,
            'death_term': death_term,
            'net_source': net_source,
            'net_source_viable': net_source_viable,
            'total_growth': total_growth,
            'total_death': total_death
        }


@njit
def compute_source_vector_numba(phi_hat, nutrient_field, lambda_, mu, nutrient_thresholds, 
                               P, D, beta_N):
    """
    Numba-optimized version of source vector computation.
    
    Args:
        phi_hat: Stacked cell fraction fields for all M populations
        nutrient_field: Local nutrient concentration field
        lambda_: Proliferation rates for each population
        mu: Death rates for each population
        nutrient_thresholds: Nutrient thresholds for each population
        P: Transfer matrix
        D: Death matrix
        beta_N: Necrotic feedback parameter
        
    Returns:
        A_hat: Source term vector for all populations
    """
    M = lambda_.shape[0]
    shape = phi_hat.shape[1:]  # Spatial dimensions
    
    # Initialize output
    A_hat = np.zeros((M+1, *shape), dtype=np.float32)
    
    # Compute nutrient switches (spatially varying, simplified for Numba)
    k = 5.0
    Theta = np.zeros((M, *shape), dtype=np.float32)
    for i in range(M):
        for x in range(shape[0]):
            for y in range(shape[1]):
                for z in range(shape[2]):
                    local_nutrient = nutrient_field[x, y, z]
                    Theta[i, x, y, z] = 0.5 * (1 + np.tanh(k * (local_nutrient - nutrient_thresholds[i])))
    
    # Compute necrotic feedback
    V_N = np.sum(phi_hat[M-1])  # Assuming necrotic is last population
    G_N = np.exp(-beta_N * V_N)
    
    # Compute proliferation-weighted populations
    alpha_hat = np.zeros_like(phi_hat[:M])
    for i in range(M):
        alpha_hat[i] = lambda_[i] * phi_hat[i]
    
    # Apply nutrient modulation to death matrix (spatially varying)
    # Compute source vector with spatially varying death rates
    for i in range(M+1):
        for j in range(M):
            for x in range(shape[0]):
                for y in range(shape[1]):
                    for z in range(shape[2]):
                        # Growth term contribution
                        if i < M:
                            A_hat[i, x, y, z] += G_N * nutrient_field[x, y, z] * P[i, j] * alpha_hat[j, x, y, z]
                        
                        # Death term contribution (spatially varying)
                        # Death should happen when nutrient is BELOW threshold (inverse of growth switch)
                        # So we use (1 - Theta) instead of Theta for death
                        A_hat[i, x, y, z] -= D[i, j] * (1.0 - Theta[j, x, y, z]) * phi_hat[j, x, y, z]
    
    return A_hat


@njit
def compute_pressure_source_vector_numba(phi_hat, nutrient_field, lambda_, mu, nutrient_thresholds, 
                                        P, beta_N):
    """
    Numba-optimized version of pressure source computation.
    
    For the pressure equation, only absolute growth/death terms are considered,
    excluding mass transfer between populations.
    
    Args:
        phi_hat: Stacked cell fraction fields for all M populations
        nutrient_field: Local nutrient concentration field
        lambda_: Proliferation rates for each population
        mu: Death rates for each population
        nutrient_thresholds: Nutrient thresholds for each population
        P: Transfer matrix
        beta_N: Necrotic feedback parameter
        
    Returns:
        pressure_source: Source term for pressure equation (scalar field)
    """
    M = lambda_.shape[0]
    shape = phi_hat.shape[1:]  # Spatial dimensions
    
    # Initialize output
    pressure_source = np.zeros(shape, dtype=np.float32)
    
    # Compute nutrient switches (spatially varying, simplified for Numba)
    k = 5.0
    Theta = np.zeros((M, *shape), dtype=np.float32)
    for i in range(M):
        for x in range(shape[0]):
            for y in range(shape[1]):
                for z in range(shape[2]):
                    local_nutrient = nutrient_field[x, y, z]
                    Theta[i, x, y, z] = 0.5 * (1 + np.tanh(k * (local_nutrient - nutrient_thresholds[i])))
    
    # Compute necrotic feedback
    V_N = np.sum(phi_hat[M-1])  # Assuming necrotic is last population
    G_N = np.exp(-beta_N * V_N)
    
    # Compute proliferation-weighted populations
    alpha_hat = np.zeros_like(phi_hat[:M])
    for i in range(M):
        alpha_hat[i] = lambda_[i] * phi_hat[i]
    
    # For pressure, only count absolute growth/death, not mass transfer
    # Growth: only diagonal terms of P (growth that stays in same population)
    for i in range(M):
        if i < P.shape[0] and i < P.shape[1]:
            for x in range(shape[0]):
                for y in range(shape[1]):
                    for z in range(shape[2]):
                        pressure_source[x, y, z] += G_N * nutrient_field[x, y, z] * P[i, i] * alpha_hat[i, x, y, z]
    
    # Death terms are excluded from pressure source because:
    # - All death from viable populations flows into necrotic (via death matrix D)
    # - The necrotic pathway is a transfer term (net mass conserved)
    # - Transfer terms should not contribute to pressure
    # Therefore, pressure source only includes growth terms
    
    return pressure_source
