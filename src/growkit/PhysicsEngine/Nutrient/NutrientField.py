"""
Nutrient Field Module

This module handles nutrient diffusion and consumption in the tumor growth simulation.
It provides a clean interface for updating the nutrient field based on cell populations.

Author: Riley Jae McNamara
Date: 2025-02-19
"""

import numpy as np
from numba import njit
from scipy.sparse.linalg import cg, LinearOperator
from scipy.sparse import diags
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
        self.k_switch = float(cfg.get('nutrient', {}).get('dynamics', {}).get('k', 1.0))
        self.lambda_rates = np.array([float(self.pops[pop].get('dynamics', {}).get('lambda', 0.0)) for pop in self.labels], dtype=np.float32)
        self.mu_rates = np.array([float(self.pops[pop].get('dynamics', {}).get('mu', 0.0)) for pop in self.labels], dtype=np.float32)
        
        # Extract consumption rates for each population
        self.population_consumption_rates = {}
        for label, pop_data in populations.items():
            self.population_consumption_rates[label] = float(pop_data.get('dynamics', {}).get('nutrient_consumption', 0.1))
        
        # Use quasi-steady-state solver by default (disabled due to numerical issues with large D)
        # Can be enabled via config, but may be slow/unstable with very large diffusion coefficients
        self.use_quasi_steady_state = cfg.get('nutrient', {}).get('dynamics', {}).get('quasi_steady_state', False)
    
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
    
    def compute_nutrient_update(self, phi_hat, nutrient_field, dx, dt):
        """
        Compute nutrient field update due to diffusion and consumption.
        
        Uses quasi-steady-state solver by default (solves D*∇²n = consumption - production)
        which is physically correct when diffusion is much faster than cell growth.
        This is also much faster than explicit time-stepping with sub-stepping.
        
        Args:
            phi_hat: Stacked cell fraction fields (M, nx, ny, nz)
            nutrient_field: Current nutrient concentration field
            dx: Grid spacing
            dt: Time step (not used in quasi-steady-state, but kept for API compatibility)
            
        Returns:
            nutrient_field_new: Updated nutrient concentration field
        """
        if self.use_quasi_steady_state:
            # Solve quasi-steady-state: D * ∇²n = consumption - production
            # This is physically correct when diffusion >> cell growth timescales
            return self._solve_quasi_steady_state(phi_hat, nutrient_field, dx)
        else:
            # Use explicit time-stepping (original method)
            # Check stability constraint for explicit diffusion
            # For 3D explicit diffusion: D * dt / dx^2 < 1/6 for stability
            max_stable_diffusion = (dx**2) / (6.0 * dt)
            
            if self.diffusion_coeff > max_stable_diffusion:
                # Use sub-stepping to maintain stability
                num_substeps = int(np.ceil(6.0 * self.diffusion_coeff * dt / (dx**2)))
                dt_substep = dt / num_substeps
                
                # Apply diffusion in multiple sub-steps
                current_nutrient = nutrient_field.copy()
                for substep in range(num_substeps):
                    consumption_rates = np.array([self.population_consumption_rates[label] for label in self.labels], dtype=np.float32)
                    nutrient_threshold = np.array([self.nutrient_threshold[self.labels.index(label)] for label in self.labels], dtype=np.float32)
                    production_rate = np.array([self.production_rate[self.labels.index(label)] for label in self.labels], dtype=np.float32)
                    
                    current_nutrient = compute_nutrient_update_numba(
                        current_nutrient, phi_hat, dx, dt_substep,
                        self.diffusion_coeff, production_rate, consumption_rates, nutrient_threshold,
                        self.lambda_rates, self.mu_rates, np.float32(self.k_switch)
                    )
                
                return current_nutrient
            else:
                # Diffusion is stable, proceed normally
                consumption_rates = np.array([self.population_consumption_rates[label] for label in self.labels], dtype=np.float32)
                nutrient_threshold = np.array([self.nutrient_threshold[self.labels.index(label)] for label in self.labels], dtype=np.float32)
                production_rate = np.array([self.production_rate[self.labels.index(label)] for label in self.labels], dtype=np.float32)
                
                return compute_nutrient_update_numba(
                    nutrient_field, phi_hat, dx, dt,
                    self.diffusion_coeff, production_rate, consumption_rates, nutrient_threshold,
                    self.lambda_rates, self.mu_rates, np.float32(self.k_switch)
                )
    
    def _solve_quasi_steady_state(self, phi_hat, nutrient_field, dx):
        """
        Solve the quasi-steady-state nutrient equation: D * ∇²n = consumption - production
        
        This assumes diffusion is much faster than cell growth, so the nutrient field
        reaches steady state instantly. This is physically correct and much faster than
        explicit time-stepping with sub-stepping.
        
        Args:
            phi_hat: Stacked cell fraction fields (M, nx, ny, nz)
            nutrient_field: Current nutrient concentration field (used as initial guess)
            dx: Grid spacing
            
        Returns:
            nutrient_field_new: Steady-state nutrient concentration field
        """
        shape = nutrient_field.shape
        nx, ny, nz = shape
        M = phi_hat.shape[0]
        
        # Compute consumption and production terms (source/sink)
        consumption_production = compute_consumption_production_numba(
            phi_hat, nutrient_field,
            np.array([self.population_consumption_rates[label] for label in self.labels], dtype=np.float32),
            np.array([self.production_rate[self.labels.index(label)] for label in self.labels], dtype=np.float32),
            np.array([self.nutrient_threshold[self.labels.index(label)] for label in self.labels], dtype=np.float32),
            np.float32(self.k_switch)
        )
        
        # Equation: D * ∇²n = consumption - production
        # We solve: -D * ∇²n = -(consumption - production)
        # This keeps the source term at full magnitude (not divided by D)
        
        # Create boundary mask
        boundary_mask = np.zeros(shape, dtype=bool)
        boundary_mask[0, :, :] = True
        boundary_mask[-1, :, :] = True
        boundary_mask[:, 0, :] = True
        boundary_mask[:, -1, :] = True
        boundary_mask[:, :, 0] = True
        boundary_mask[:, :, -1] = True
        
        # Create linear operator for -∇² with Dirichlet BCs
        # We solve for the full field, but enforce BCs in the operator
        N = np.prod(shape)
        dx2 = dx * dx
        
        def matvec(n_flat):
            """Apply -D*∇² operator with Dirichlet BCs"""
            n = n_flat.reshape(shape).copy()
            # Enforce Dirichlet BCs: n = boundary_value at boundaries
            n[boundary_mask] = self.boundary_value
            # Compute Laplacian and scale by D
            lap_n = isotropic_laplacian(n, dx)
            result = -self.diffusion_coeff * lap_n.flatten()
            # At boundaries, enforce identity (n = boundary_value)
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        if boundary_mask[i, j, k]:
                            idx = i * ny * nz + j * nz + k
                            result[idx] = n_flat[idx]  # Identity: n = n at boundaries
            return result.astype(np.float64)
        
        A = LinearOperator((N, N), matvec=matvec, dtype=np.float64)
        
        # Create preconditioner (scaled by D to match the operator)
        diag_elements = np.full(N, -self.diffusion_coeff * 6.0 / dx2, dtype=np.float64)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if boundary_mask[i, j, k]:
                        idx = i * ny * nz + j * nz + k
                        diag_elements[idx] = 1.0  # Identity at boundaries
        
        diag_elements = np.abs(diag_elements)
        diag_elements = np.maximum(diag_elements, 1e-12)
        preconditioner_diag = 1.0 / diag_elements
        M = diags([preconditioner_diag], [0], shape=(N, N), dtype=np.float64)
        
        # Initial guess: current nutrient field
        x0 = nutrient_field.flatten().astype(np.float64, copy=False)
        x0[boundary_mask.flatten()] = self.boundary_value  # Enforce BCs in initial guess
        
        # Adjust RHS: 
        # - At interior points: use the source term -(consumption - production)
        # - At boundaries: enforce Dirichlet BCs by setting RHS = boundary_value
        #   (the operator will enforce n = boundary_value, so we need RHS = boundary_value)
        rhs_adjusted = -consumption_production.flatten().astype(np.float64, copy=False)
        boundary_flat = boundary_mask.flatten()
        rhs_adjusted[boundary_flat] = self.boundary_value
        
        # Solve the system: -D*∇²n = -(consumption - production) with n = boundary_value at boundaries
        try:
            n_flat, info = cg(A, rhs_adjusted, M=M, x0=x0, rtol=1e-4, maxiter=200, atol=1e-8)
            if info != 0:
                # Try with more relaxed parameters
                n_flat, info = cg(A, rhs_adjusted, M=M, x0=x0, rtol=1e-2, maxiter=500, atol=1e-6)
        except Exception as e:
            print(f"Warning: Nutrient quasi-steady-state solver error: {e}, using current field")
            n_flat = x0
        
        # Reshape and enforce boundary conditions
        nutrient_field_new = n_flat.reshape(shape).astype(np.float32)
        nutrient_field_new[boundary_mask] = self.boundary_value  # Ensure exact boundary values
        
        # Ensure non-negative
        nutrient_field_new = np.maximum(nutrient_field_new, 0.0)
        
        return nutrient_field_new


@njit
def compute_consumption_production_numba(phi_hat, nutrient_field, consumption_rates, 
                                         production_rate, nutrient_threshold, k_switch):
    """
    Compute consumption - production terms for quasi-steady-state solver.
    
    Args:
        phi_hat: Stacked cell fraction fields (M, nx, ny, nz)
        nutrient_field: Current nutrient concentration field
        consumption_rates: Array of consumption rates per population
        production_rate: Array of production rates per population
        nutrient_threshold: Array of nutrient thresholds per population
        k_switch: Switch steepness parameter
        
    Returns:
        consumption_production: Field of (consumption - production) at each point
    """
    nx, ny, nz = nutrient_field.shape
    M = phi_hat.shape[0]
    consumption_production = np.zeros_like(nutrient_field, dtype=np.float32)
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                local_n = nutrient_field[i, j, k]
                consumption = 0.0
                production = 0.0
                
                for m in range(M):
                    consumption += consumption_rates[m] * phi_hat[m, i, j, k]
                    production += production_rate[m] * phi_hat[m, i, j, k] * (1.0 - local_n)
                
                consumption_production[i, j, k] = consumption - production
    
    return consumption_production


@njit
def compute_nutrient_update_numba(nutrient_field, phi_hat, dx, dt,
                                diffusion_coeff, production_rate, consumption_rates, nutrient_threshold,
                                lambda_rates, mu_rates, k_switch):
    """
    Compute nutrient field update using Numba optimization. 
    
    Args:
        nutrient_field: Current nutrient concentration field
        phi_T: Total cell density field
        phi_hat: Stacked cell fraction fields (M, nx, ny, nz)
        dx: Grid spacing
        dt: Time step
        diffusion_coeff: Nutrient diffusion coefficient (in units of length^2/time)
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
    # Since this is a Numba function, we'll compute the proper isotropic laplacian inline
    # This matches the implementation in Operators.isotropic_laplacian
    diffusion_field = np.zeros_like(nutrient_field, dtype=np.float32)
    dx2 = dx * dx
    
    # Compute isotropic laplacian for interior points using proper weights
    # Standard Cartesian directions (70% weight) + Diagonal directions (30% weight total, 0.075 each)
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            for k in range(1, nz-1):
                # Axis-aligned components (70% weight)
                lap_axis = 0.7 * (
                    (nutrient_field[i+1, j, k] + nutrient_field[i-1, j, k] +
                     nutrient_field[i, j+1, k] + nutrient_field[i, j-1, k] +
                     nutrient_field[i, j, k+1] + nutrient_field[i, j, k-1] -
                     6.0 * nutrient_field[i, j, k])
                ) / dx2
                
                # Diagonal components for isotropy (0.075 weight each, 6 diagonals = 0.45 total)
                # Diagonal in xy plane
                lap_diag = 0.075 * (
                    (nutrient_field[i+1, j+1, k] + nutrient_field[i-1, j-1, k] - 2.0 * nutrient_field[i, j, k]) / (dx2 * 2.0)
                )
                lap_diag += 0.075 * (
                    (nutrient_field[i+1, j-1, k] + nutrient_field[i-1, j+1, k] - 2.0 * nutrient_field[i, j, k]) / (dx2 * 2.0)
                )
                # Diagonal in xz plane
                lap_diag += 0.075 * (
                    (nutrient_field[i+1, j, k+1] + nutrient_field[i-1, j, k-1] - 2.0 * nutrient_field[i, j, k]) / (dx2 * 2.0)
                )
                lap_diag += 0.075 * (
                    (nutrient_field[i+1, j, k-1] + nutrient_field[i-1, j, k+1] - 2.0 * nutrient_field[i, j, k]) / (dx2 * 2.0)
                )
                # Diagonal in yz plane
                lap_diag += 0.075 * (
                    (nutrient_field[i, j+1, k+1] + nutrient_field[i, j-1, k-1] - 2.0 * nutrient_field[i, j, k]) / (dx2 * 2.0)
                )
                lap_diag += 0.075 * (
                    (nutrient_field[i, j+1, k-1] + nutrient_field[i, j-1, k+1] - 2.0 * nutrient_field[i, j, k]) / (dx2 * 2.0)
                )
                
                # Scale diffusion by dt for proper time integration
                diffusion_field[i, j, k] = diffusion_coeff * dt * (lap_axis + lap_diag)
    
    # Boundary points: compute diffusion but will enforce Dirichlet conditions after
    diffusion_field[0, :, :] = 0.0
    diffusion_field[-1, :, :] = 0.0
    diffusion_field[:, 0, :] = 0.0
    diffusion_field[:, -1, :] = 0.0
    diffusion_field[:, :, 0] = 0.0
    diffusion_field[:, :, -1] = 0.0
    
    # Boundary value for Dirichlet conditions
    boundary_value = 1.0
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Check if we're on a boundary
                is_boundary = (i == 0 or i == nx-1 or j == 0 or j == ny-1 or k == 0 or k == nz-1)
                
                if is_boundary:
                    # Dirichlet boundary condition: fixed nutrient concentration at boundaries
                    nutrient_field_new[i, j, k] = boundary_value
                else:
                    # Interior points: compute diffusion, consumption, and production
                    diffusion = diffusion_field[i, j, k]
                    
                    # Compute consumption term tied to proliferation activity
                    consumption = 0.0
                    production = 0.0
                    local_n = nutrient_field[i, j, k]
                    for m in range(M):
                        s_grow = 0.5 * (1.0 + np.tanh(k_switch * (local_n - nutrient_threshold[m])))
                        s_death = 1.0 - s_grow
                        consumption += consumption_rates[m] * phi_hat[m, i, j, k] 
                        production += production_rate[m] * phi_hat[m, i, j, k] * (1.0 - local_n) 

                    # Update nutrient field with proper time scaling
                    # All terms (diffusion, consumption, production) must be scaled by dt for consistency
                    nutrient_field_new[i, j, k] = nutrient_field[i, j, k] + diffusion - dt * consumption + dt * production
                    
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
