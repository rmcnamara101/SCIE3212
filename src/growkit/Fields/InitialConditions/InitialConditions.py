import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import yaml


class InitialConditions:
    """
    Handles initialization of simulation fields based on configuration.
    Supports multiple initial condition types and configurable seeding densities.
    """
    
    def __init__(self, cfg: Dict):
        """
        Initialize the InitialConditions manager.
        
        Args:
            cfg: Configuration dictionary loaded from YAML
        """
        self.cfg = cfg
        self.grid = (cfg["domain"]["shape"], cfg["domain"]["shape"], cfg["domain"]["shape"])
        self.dx = cfg["domain"]["dx"]
        
        # Get initial conditions configuration
        self.ic_config = cfg.get("initial_conditions", {})
        self.ic_type = self.ic_config.get("type", "spherical")
        
        # Get population information
        self.populations = cfg["populations"]
        self.M = len(self.populations)
        self.labels = [p["label"] for p in self.populations.values()]
        
    def initialize_cell_fields(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize cell fraction fields based on configuration.
        
        Returns:
            phi_hat: Cell fraction fields (M, nx, ny, nz)
            nutrient_field: Nutrient field (nx, ny, nz)
        """
        nx, ny, nz = self.grid
        
        # Initialize cell fraction fields based on type
        if self.ic_type == "spherical":
            phi_hat = self._create_spherical_initial_conditions()
        elif self.ic_type == "gaussian_random_blob":
            phi_hat = self._create_deformed_blob_initial_conditions()
        elif self.ic_type == "multiple_spheres":
            phi_hat = self._create_multiple_spheres_initial_conditions()
        elif self.ic_type == "uniform":
            phi_hat = self._create_uniform_initial_conditions()
        elif self.ic_type == "custom":
            phi_hat = self._create_custom_initial_conditions()
        else:
            raise ValueError(f"Unknown initial condition type: {self.ic_type}")
        
        # Initialize nutrient field
        nutrient_field = self._initialize_nutrient_field()
        
        # Ensure physical constraints
        phi_hat = np.clip(phi_hat, 0.0, 1.0)
        nutrient_field = np.clip(nutrient_field, 0.0, 1.0)
        
        return phi_hat, nutrient_field
    
    def _create_spherical_initial_conditions(self) -> np.ndarray:
        """Create spherical initial conditions."""
        nx, ny, nz = self.grid
        phi_hat = np.zeros((self.M, nx, ny, nz), dtype=np.float32)
        
        # Get configuration parameters
        center = self.ic_config.get("center", [nx//2, ny//2, nz//2])
        radius = self.ic_config.get("radius", min(nx, ny, nz) // 4)
        seeding_densities = self.ic_config.get("seeding_densities", {})
        
        # Default seeding densities if not specified
        if not seeding_densities:
            seeding_densities = {label: 0.8 if i == 0 else 0.1 / (self.M - 1) 
                               for i, label in enumerate(self.labels)}
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    dist = np.sqrt((i - center[0])**2 + (j - center[1])**2 + (k - center[2])**2)
                    if dist <= radius:
                        for m, label in enumerate(self.labels):
                            phi_hat[m, i, j, k] = seeding_densities.get(label, 0.0)
        
        return phi_hat
    
    def _create_deformed_blob_initial_conditions(self) -> np.ndarray:
        """Create deformed blob initial conditions with hard boundaries."""
        nx, ny, nz = self.grid
        phi_hat = np.zeros((self.M, nx, ny, nz), dtype=np.float32)
        
        # Get configuration parameters
        center = self.ic_config.get("center", [nx//2, ny//2, nz//2])
        radius = self.ic_config.get("radius", min(nx, ny, nz) // 4)  # Base radius
        deformation_strength = self.ic_config.get("deformation_strength", 0.3)  # How much to deform
        seeding_densities = self.ic_config.get("seeding_densities", {})
        
        # Set random seed for reproducible deformations
        np.random.seed(self.cfg.get("simulation", {}).get("seed", 42))
        
        # Create deformation field using spherical harmonics or simple noise
        # This creates irregular, blobby shapes
        deformation_field = np.random.normal(0, 1, (nx, ny, nz))
        
        # Apply spatial smoothing to create correlated deformations
        from scipy.ndimage import gaussian_filter
        try:
            # Smooth the deformation field to create realistic blobby shapes
            smoothed_deformation = gaussian_filter(deformation_field, sigma=2.0)
            # Normalize to [-1, 1] range
            smoothed_deformation = (smoothed_deformation - smoothed_deformation.min()) / (smoothed_deformation.max() - smoothed_deformation.min()) * 2 - 1
        except ImportError:
            # Fallback if scipy is not available
            smoothed_deformation = deformation_field * 0.1
        
        # Create the deformed blob with hard boundaries
        blob_mask = np.zeros((nx, ny, nz), dtype=bool)
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # Calculate distance from center
                    dist = np.sqrt((i - center[0])**2 + (j - center[1])**2 + (k - center[2])**2)
                    
                    # Apply deformation to the radius
                    deformation = smoothed_deformation[i, j, k] * deformation_strength
                    deformed_radius = radius * (1 + deformation)
                    
                    # Hard boundary: inside or outside
                    if dist <= deformed_radius:
                        blob_mask[i, j, k] = True
        
        # Create density field with hard boundaries
        total_density = np.zeros((nx, ny, nz), dtype=np.float32)
        total_density[blob_mask] = 1.0  # Full density inside, zero outside
        
        # Distribute density among populations
        for m, label in enumerate(self.labels):
            density_fraction = seeding_densities.get(label, 1.0 / self.M)
            phi_hat[m, :, :, :] = total_density * density_fraction
        
        return phi_hat
    
    def _create_multiple_spheres_initial_conditions(self) -> np.ndarray:
        """Create multiple spheres initial conditions."""
        nx, ny, nz = self.grid
        phi_hat = np.zeros((self.M, nx, ny, nz), dtype=np.float32)
        
        spheres = self.ic_config.get("spheres", [])
        if not spheres:
            # Default: two spheres
            spheres = [
                {"center": [nx//3, ny//2, nz//2], "radius": min(nx, ny, nz) // 6, 
                 "population": 0, "density": 0.8},
                {"center": [2*nx//3, ny//2, nz//2], "radius": min(nx, ny, nz) // 6, 
                 "population": 1, "density": 0.8}
            ]
        
        for sphere in spheres:
            center = sphere["center"]
            radius = sphere["radius"]
            population = sphere["population"]
            density = sphere["density"]
            
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        dist = np.sqrt((i - center[0])**2 + (j - center[1])**2 + (k - center[2])**2)
                        if dist <= radius:
                            phi_hat[population, i, j, k] = density
        
        return phi_hat
    
    def _create_uniform_initial_conditions(self) -> np.ndarray:
        """Create uniform initial conditions."""
        nx, ny, nz = self.grid
        phi_hat = np.zeros((self.M, nx, ny, nz), dtype=np.float32)
        
        seeding_densities = self.ic_config.get("seeding_densities", {})
        if not seeding_densities:
            seeding_densities = {label: 0.1 for label in self.labels}
        
        for m, label in enumerate(self.labels):
            density = seeding_densities.get(label, 0.1)
            phi_hat[m, :, :, :] = density
        
        return phi_hat
    
    def _create_custom_initial_conditions(self) -> np.ndarray:
        """Create custom initial conditions using a function or predefined pattern."""
        nx, ny, nz = self.grid
        phi_hat = np.zeros((self.M, nx, ny, nz), dtype=np.float32)
        
        # This could be extended to support custom functions
        # For now, fall back to spherical
        print("Warning: Custom initial conditions not implemented, falling back to spherical")
        return self._create_spherical_initial_conditions()
    
    def _initialize_nutrient_field(self) -> np.ndarray:
        """Initialize nutrient field."""
        nx, ny, nz = self.grid
        
        # Check if nutrient initialization is specified in config
        nutrient_config = self.ic_config.get("nutrient", {})
        
        if nutrient_config.get("type") == "uniform":
            concentration = nutrient_config.get("concentration", 1.0)
            return np.full((nx, ny, nz), concentration, dtype=np.float32)
        elif nutrient_config.get("type") == "gradient":
            # Create a gradient from high to low concentration
            concentration_max = nutrient_config.get("concentration_max", 1.0)
            concentration_min = nutrient_config.get("concentration_min", 0.0)
            
            nutrient_field = np.zeros((nx, ny, nz), dtype=np.float32)
            for i in range(nx):
                # Linear gradient along x-axis
                concentration = concentration_max - (concentration_max - concentration_min) * i / nx
                nutrient_field[i, :, :] = concentration
            return nutrient_field
        else:
            # Default: use nutrient manager
            try:
                from src.growkit.PhysicsEngine.Nutrient.NutrientField import NutrientField
                nutrient_manager = NutrientField(self.cfg, self.cfg["populations"])
                return nutrient_manager.initialize_nutrient_field((nx, ny, nz), self.dx)
            except ImportError:
                # Fallback to uniform nutrient field
                return np.full((nx, ny, nz), 1.0, dtype=np.float32)
    
    def get_initial_conditions_summary(self) -> Dict:
        """Get a summary of the initial conditions configuration."""
        return {
            "type": self.ic_type,
            "grid_size": self.grid,
            "dx": self.dx,
            "num_populations": self.M,
            "population_labels": self.labels,
            "configuration": self.ic_config
        }
