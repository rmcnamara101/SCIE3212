# field_manager.py
import numpy as np
from pathlib import Path
import yaml

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

    def initialize_fields(self, initial_conditions=None):
        """
        Initialize the simulation fields.
        
        Args:
            initial_conditions: Optional initial conditions dictionary (for backward compatibility)
        """
        if initial_conditions is None:
            # Use the new InitialConditions class
            from src.growkit.Fields.InitialConditions.InitialConditions import InitialConditions
            ic_manager = InitialConditions(self.cfg)
            self.phi_hat, self.nutrient_field = ic_manager.initialize_cell_fields()
        else:
            # Backward compatibility: use provided initial conditions
            self.phi_hat = initial_conditions["phi_hat"].copy()
            if "nutrient" in initial_conditions:
                self.nutrient_field = initial_conditions["nutrient"].copy()
            else:
                # Initialize nutrient field using the new system
                from src.growkit.Fields.InitialConditions.InitialConditions import InitialConditions
                ic_manager = InitialConditions(self.cfg)
                _, self.nutrient_field = ic_manager.initialize_cell_fields()

    def update_physics_fields(self, phi_hat, nutrient_field):
        """
        Update physics fields based on current cell and nutrient fields.
        
        Args:
            phi_hat: Current cell fraction fields
            nutrient_field: Current nutrient field
        """
        # Update stored fields
        self.phi_hat = phi_hat
        self.nutrient_field = nutrient_field
        
        # Compute physics fields using the physics engine
        from src.growkit.PhysicsEngine import VectorizedCellDynamics
        cell_dynamics = VectorizedCellDynamics(self.cfg, self.cfg["populations"], self)
        
        # Compute energy derivative
        phi_T = np.sum(phi_hat, axis=0)
        self.energy_derivative = cell_dynamics.compute_energy_derivative(phi_T, self.dx)
        
        # Compute solid velocity (this will update pressure and velocity in field_manager)
        cell_dynamics.compute_solid_velocity(phi_hat, nutrient_field, self.dx)
        
        # Compute mass fluxes (this will update mass_flux in field_manager)
        cell_dynamics.compute_mass_fluxes(phi_hat, self.dx, self.energy_derivative)

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





