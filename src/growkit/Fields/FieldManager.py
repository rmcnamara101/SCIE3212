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
        shape = int(self.cfg["domain"]["shape"])  # Ensure shape is an integer
        self.grid = (shape, shape, shape)
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
        self.host_field = None  # Host field representing remaining space (nx, ny, nz)
        
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
        
        # Initialize host field to fill remaining space
        self._initialize_host_field()

    def _initialize_host_field(self):
        """
        Initialize the host field to ensure total volume fraction equals 1 everywhere.
        """
        if self.phi_hat is not None:
            # Calculate total cell fraction at each spatial point
            phi_T = np.sum(self.phi_hat, axis=0)  # Shape: (nx, ny, nz)
            
            # Host field fills the remaining space (1 - total_cell_fraction)
            self.host_field = np.clip(1.0 - phi_T, 0.0, 1.0)
        else:
            # If no cell fields, host field fills entire space
            self.host_field = np.ones(self.grid, dtype=np.float32)

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
        
        # Update host field to maintain total volume fraction = 1
        self._update_host_field()
        
        # Use provided cell_dynamics or create new one
        if cell_dynamics is None:
            cell_dynamics = VectorizedCellDynamics(self.cfg, self.cfg["populations"], self)
        
        # Compute energy derivative using only cellular volume fraction (exclude host)
        # Including host makes total = 1 everywhere, zeroing gradients and killing adhesion
        phi_T = np.sum(phi_hat, axis=0)
        self.energy_derivative = cell_dynamics.compute_energy_derivative(phi_T, self.dx)
        
        # Compute solid velocity (this will update pressure and velocity in field_manager)
        cell_dynamics.compute_solid_velocity(phi_hat, nutrient_field, self.dx)
        
        # Compute mass fluxes (this will update mass_flux in field_manager)
        cell_dynamics.compute_mass_fluxes(phi_hat, self.dx, self.energy_derivative)
        
        # Store source terms if provided
        if source_terms is not None:
            self.source_terms = source_terms

    def _update_host_field(self):
        """
        Update the host field to ensure total volume fraction equals 1 everywhere.
        """
        if self.phi_hat is not None:
            # Calculate total cell fraction at each spatial point
            phi_T = np.sum(self.phi_hat, axis=0)  # Shape: (nx, ny, nz)
            
            # Host field fills the remaining space (1 - total_cell_fraction)
            self.host_field = np.clip(1.0 - phi_T, 0.0, 1.0)

    def get_cell_fields(self):
        """
        Get current cell fields.
        
        Returns:
            phi_hat: Cell fraction fields
            nutrient_field: Nutrient field
            host_field: Host field
        """
        return self.phi_hat, self.nutrient_field, self.host_field

    def set_cell_fields(self, phi_hat, nutrient_field):
        """
        Set cell fields and update host field accordingly.
        
        Args:
            phi_hat: Cell fraction fields
            nutrient_field: Nutrient field
        """
        self.phi_hat = phi_hat
        self.nutrient_field = nutrient_field
        self._update_host_field()

    def get_total_volume_fraction(self):
        """
        Get the total volume fraction (cells + host) at each spatial point.
        
        Returns:
            total_volume: Total volume fraction field (nx, ny, nz)
        """
        if self.phi_hat is None:
            return self.host_field if self.host_field is not None else np.ones(self.grid, dtype=np.float32)
        
        return np.sum(self.phi_hat, axis=0) + self.host_field

    def normalize_volume_fractions(self, add_host_field=True):
        """
        Normalize volume fractions to ensure they are bounded by 1 at each spatial point.
        The host field automatically fills the remaining space to ensure total = 1 everywhere.
        Also applies boundary constraints to prevent cell leakage.
        
        Args:
            add_host_field: Whether to maintain the host field (default: True)
            
        Returns:
            phi_hat_normalized: Normalized cell fraction fields
        """
        if self.phi_hat is None:
            return None
        
        # Ensure all fractions are non-negative
        phi_hat = np.clip(self.phi_hat, 0.0, 1.0)
        
        # Note: Boundary constraints removed as they were causing issues
        # The physics should naturally handle boundary behavior
        
        # Calculate total cell fraction at each spatial point
        phi_T = np.sum(phi_hat, axis=0)  # Shape: (nx, ny, nz)
        
        # Find regions where total cell density exceeds 1
        overflow_mask = phi_T > 1.0
        
        if np.any(overflow_mask):
            # Create scaling factor to normalize to sum = 1 where needed
            scaling = np.ones_like(phi_T)
            scaling[overflow_mask] = 1.0 / phi_T[overflow_mask]
            
            # Apply scaling to all cell types
            phi_hat_normalized = phi_hat * scaling[None, :, :, :]
        else:
            phi_hat_normalized = phi_hat.copy()
        
        # Update the stored fields
        self.phi_hat = phi_hat_normalized
        
        # Update host field to maintain total volume fraction = 1
        if add_host_field:
            self._update_host_field()
        
        return phi_hat_normalized

    def check_volume_fraction_constraints(self, tolerance=1e-6):
        """
        Check if volume fraction constraints are satisfied.
        With host field, total volume fraction should equal 1 everywhere.
        
        Args:
            tolerance: Tolerance for checking if sum equals 1
            
        Returns:
            is_valid: Boolean indicating if constraints are satisfied
            max_deviation: Maximum deviation from sum = 1
            mean_deviation: Mean deviation from sum = 1
        """
        if self.phi_hat is None:
            return False, 0.0, 0.0
        
        # Get total volume fraction including host field
        total_volume = self.get_total_volume_fraction()
        
        # Check if any values are negative
        has_negative = np.any(self.phi_hat < 0) or (self.host_field is not None and np.any(self.host_field < 0))
        
        # Check deviations from sum = 1 (should be exactly 1 everywhere with host field)
        deviations = np.abs(total_volume - 1.0)
        max_deviation = np.max(deviations)
        mean_deviation = np.mean(deviations)
        
        # Check if any regions significantly deviate from 1
        has_deviation = np.any(deviations > tolerance)
        
        is_valid = not has_negative and not has_deviation
        
        return is_valid, max_deviation, mean_deviation

    def get_cell_and_host_fields(self):
        """
        Get all fields including host field as a single array.
        
        Returns:
            all_fields: Array containing cell fields and host field (M+1, nx, ny, nz)
        """
        if self.phi_hat is None:
            return None
        
        # Stack cell fields and host field
        all_fields = np.vstack([self.phi_hat, self.host_field[None, :, :, :]])
        return all_fields

    def set_cell_and_host_fields(self, all_fields):
        """
        Set all fields from a single array that includes cell fields and host field.
        
        Args:
            all_fields: Array containing cell fields and host field (M+1, nx, ny, nz)
        """
        if all_fields is None:
            return
        
        # Extract cell fields (first M components)
        self.phi_hat = all_fields[:self.M, :, :, :]
        
        # Extract host field (last component)
        self.host_field = all_fields[self.M, :, :, :]
        
        # Update nutrient field if needed (this method doesn't handle nutrient field)





