# field_manager.py
import numpy as np
from pathlib import Path
import yaml

from src.growkit.Fields.InitialConditions.InitialConditions import InitialConditions
from src.growkit.PhysicsEngine.VectorizedCellDynamics import VectorizedCellDynamics

class FieldManager:
    def __init__(self, cfg_path: str):

        self.cfg = yaml.safe_load(Path(cfg_path).read_text())

        # --- domain ---------------------------------------------------------
        self.grid = (self.cfg["domain"]["shape"], self.cfg["domain"]["shape"], self.cfg["domain"]["shape"])
        self.dx    = self.cfg["domain"]["dx"]
        self.dt    = self.cfg["time"]["dt"]
        self.steps = self.cfg["time"]["steps"]

        # --- populations ----------------------------------------------------
        pops = self.cfg["populations"]
        self.labels   = [p["label"] for p in pops.values()]
        self.M        = len(pops)
        
        # --- cell fields ----------------------------------------------------
        self.phi_hat = None  # Cell fraction fields (M, nx, ny, nz)
        self.nutrient_field = None  # Nutrient concentration field (nx, ny, nz)
        
        # --- physics fields -------------------------------------------------
        self.pressure = np.zeros(self.grid, dtype=np.float32)  # Pressure field
        self.velocity = np.zeros((3,) + self.grid, dtype=np.float32)  # Velocity field (ux, uy, uz)
        self.energy_derivative = np.zeros(self.grid, dtype=np.float32)  # Adhesion energy derivative field (scalar)
        self.mass_flux = np.zeros((self.M, 3) + self.grid, dtype=np.float32)  # Mass flux fields (M populations, 3 components each)
        self.source_terms = np.zeros((self.M,) + self.grid, dtype=np.float32)  # Source terms (growth/death) for all populations

    def initialize_fields(self, initial_conditions=None):
        """
        Initialize the simulation fields.
        
        Args:
            initial_conditions: Optional initial conditions dictionary (for backward compatibility)
        """
        if initial_conditions is None:
            # Use the new InitialConditions class
            ic_manager = InitialConditions(self.cfg)
            self.phi_hat, self.nutrient_field = ic_manager.initialize_cell_fields()
        else:
            # Backward compatibility: use provided initial conditions
            self.phi_hat = initial_conditions["phi_hat"].copy()
            if "nutrient" in initial_conditions:
                self.nutrient_field = initial_conditions["nutrient"].copy()
            else:
                # Initialize nutrient field using the new system
                ic_manager = InitialConditions(self.cfg)
                _, self.nutrient_field = ic_manager.initialize_cell_fields()

    def update_physics_fields(self, phi_hat, nutrient_field, cell_dynamics=None, source_terms=None):
        """
        Update physics fields based on current cell and nutrient fields.
        
        Args:
            phi_hat: Current cell fraction fields
            nutrient_field: Current nutrient field
            cell_dynamics: Optional pre-initialized VectorizedCellDynamics instance
            source_terms: Optional pre-computed source terms
        """
        # Update stored fields
        self.phi_hat = phi_hat
        self.nutrient_field = nutrient_field
        
        # Use provided cell_dynamics or create new one
        if cell_dynamics is None:
            cell_dynamics = VectorizedCellDynamics(self.cfg, self.cfg["populations"], self)
        
        # Compute energy derivative
        phi_T = np.sum(phi_hat, axis=0)
        self.energy_derivative = cell_dynamics.compute_energy_derivative(phi_T, self.dx)
        
        # Compute solid velocity (this will update pressure and velocity in field_manager)
        cell_dynamics.compute_solid_velocity(phi_hat, nutrient_field, self.dx)
        
        # Compute mass fluxes (this will update mass_flux in field_manager)
        cell_dynamics.compute_mass_fluxes(phi_hat, self.dx, self.energy_derivative)
        
        # Store source terms if provided
        if source_terms is not None:
            self.source_terms = source_terms

    def get_cell_fields(self):
        """
        Get current cell fields.
        
        Returns:
            phi_hat: Cell fraction fields
            nutrient_field: Nutrient field
        """
        return self.phi_hat, self.nutrient_field

    def set_cell_fields(self, phi_hat, nutrient_field):
        """
        Set cell fields.
        
        Args:
            phi_hat: Cell fraction fields
            nutrient_field: Nutrient field
        """
        self.phi_hat = phi_hat
        self.nutrient_field = nutrient_field

    def normalize_volume_fractions(self, add_host_field=False):
        """
        Normalize volume fractions to ensure they are bounded by 1 at each spatial point.
        Only normalizes regions where there are actually cells (total density > 0).
        
        Args:
            add_host_field: Whether to add a host field (currently not implemented)
            
        Returns:
            phi_hat_normalized: Normalized cell fraction fields
        """
        if self.phi_hat is None:
            return None
        
        # Ensure all fractions are non-negative
        phi_hat = np.clip(self.phi_hat, 0.0, 1.0)
        
        # Calculate total cell fraction at each spatial point
        phi_T = np.sum(phi_hat, axis=0)  # Shape: (nx, ny, nz)
        
        # Only normalize regions where there are actually cells (total density > 0)
        # and where the total density significantly exceeds 1 (to prevent over-saturation)
        cell_mask = phi_T > 0  # Regions with cells
        overflow_mask = phi_T > 1.01  # Regions where total density significantly exceeds 1 (1% tolerance)
        
        # Combine masks: only normalize where there are cells AND they overflow
        normalize_mask = cell_mask & overflow_mask
        
        if np.any(normalize_mask):
            # Create scaling factor to normalize to sum = 1 only where needed
            scaling = np.ones_like(phi_T)
            scaling[normalize_mask] = 1.0 / phi_T[normalize_mask]
            
            # Apply scaling to all cell types
            phi_hat_normalized = phi_hat * scaling[None, :, :, :]
        else:
            phi_hat_normalized = phi_hat.copy()
        
        # Update the stored fields
        self.phi_hat = phi_hat_normalized
        
        return phi_hat_normalized

    def check_volume_fraction_constraints(self, tolerance=1e-6):
        """
        Check if volume fraction constraints are satisfied.
        
        Args:
            tolerance: Tolerance for checking if sum is bounded by 1
            
        Returns:
            is_valid: Boolean indicating if constraints are satisfied
            max_deviation: Maximum deviation from sum = 1 (only for regions with cells)
            mean_deviation: Mean deviation from sum = 1 (only for regions with cells)
        """
        if self.phi_hat is None:
            return False, 0.0, 0.0
        
        phi_T = np.sum(self.phi_hat, axis=0)
        
        # Check if any values are negative
        has_negative = np.any(self.phi_hat < 0)
        
        # Check deviations from sum = 1 only in regions where there are cells
        cell_mask = phi_T > 0
        if np.any(cell_mask):
            deviations = np.abs(phi_T[cell_mask] - 1.0)
            max_deviation = np.max(deviations)
            mean_deviation = np.mean(deviations)
        else:
            max_deviation = 0.0
            mean_deviation = 0.0
        
        # Check if any regions significantly exceed 1 (overflow)
        has_overflow = np.any(phi_T > 1.01 + tolerance)
        
        is_valid = not has_negative and not has_overflow
        
        return is_valid, max_deviation, mean_deviation





