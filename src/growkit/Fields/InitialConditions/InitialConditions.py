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
        nutrient_field = self._initialize_nutrient_field(phi_hat)
        
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
        
        # Get noise scale from config or use default
        noise_scale = self.ic_config.get("seeding_noise_scale", 0.2)
        
        # Create base density mask
        base_mask = np.zeros((nx, ny, nz), dtype=np.float32)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    dist = np.sqrt((i - center[0])**2 + (j - center[1])**2 + (k - center[2])**2)
                    if dist <= radius:
                        base_mask[i, j, k] = 1.0
        
        # Add noise to each population's seeding
        for m, label in enumerate(self.labels):
            base_density = base_mask * seeding_densities.get(label, 0.0)
            noisy_density = self._add_seeding_noise(base_density, noise_scale)
            phi_hat[m, :, :, :] = noisy_density
        
        return phi_hat
    
    def _add_seeding_noise(self, base_density: np.ndarray, noise_scale: float = 0.2) -> np.ndarray:
        """
        Add spatial noise to seeding density to create more realistic initial conditions.
        
        Args:
            base_density: Base density field (nx, ny, nz)
            noise_scale: Scale of noise (0.0 = no noise, 1.0 = 100% variation)
            
        Returns:
            noisy_density: Density field with added noise
        """
        nx, ny, nz = base_density.shape
        
        # Generate spatially correlated noise with multiple scales
        # This creates more realistic, multi-scale variations
        noise_field = np.zeros((nx, ny, nz))
        
        # Add noise at different spatial scales for more realistic patterns
        scales = [1, 2, 4]  # Different correlation lengths
        weights = [0.5, 0.3, 0.2]  # Weights for each scale
        
        for scale, weight in zip(scales, weights):
            # Generate noise at this scale
            raw_noise = np.random.normal(0, 1, (nx, ny, nz))
            
            # Apply spatial smoothing with different kernel sizes
            scale_noise = np.zeros_like(raw_noise)
            
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        # Larger kernel for larger scales
                        i_start = max(0, i - scale)
                        i_end = min(nx, i + scale + 1)
                        j_start = max(0, j - scale)
                        j_end = min(ny, j + scale + 1)
                        k_start = max(0, k - scale)
                        k_end = min(nz, k + scale + 1)
                        
                        scale_noise[i, j, k] = np.mean(raw_noise[i_start:i_end, j_start:j_end, k_start:k_end])
            
            # Add weighted contribution from this scale
            noise_field += weight * scale_noise
        
        # Normalize noise to [-1, 1] range
        noise_field = noise_field / (np.max(np.abs(noise_field)) + 1e-8)
        
        # Apply noise to base density with some additional randomness
        # Add both multiplicative and additive noise for more realistic variation
        multiplicative_noise = 1.0 + noise_scale * noise_field
        additive_noise = 0.1 * noise_scale * noise_field * (base_density > 0)  # Only add noise where there are cells
        
        noisy_density = base_density * multiplicative_noise + additive_noise
        
        # Ensure physical constraints
        noisy_density = np.clip(noisy_density, 0.0, 1.0)
        
        return noisy_density
    
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
        
        # Get noise scale from config or use default
        noise_scale = self.ic_config.get("seeding_noise_scale", 0.2)
        
        # Distribute density among populations with noise
        for m, label in enumerate(self.labels):
            density_fraction = seeding_densities.get(label, 1.0 / self.M)
            base_density = total_density * density_fraction
            noisy_density = self._add_seeding_noise(base_density, noise_scale)
            phi_hat[m, :, :, :] = noisy_density
        
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
        
        # Get noise scale from config or use default
        noise_scale = self.ic_config.get("seeding_noise_scale", 0.2)
        
        for sphere in spheres:
            center = sphere["center"]
            radius = sphere["radius"]
            population = sphere["population"]
            density = sphere["density"]
            
            # Create base density for this sphere
            sphere_density = np.zeros((nx, ny, nz), dtype=np.float32)
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        dist = np.sqrt((i - center[0])**2 + (j - center[1])**2 + (k - center[2])**2)
                        if dist <= radius:
                            sphere_density[i, j, k] = density
            
            # Add noise to this sphere's density
            noisy_density = self._add_seeding_noise(sphere_density, noise_scale)
            phi_hat[population, :, :, :] = noisy_density
        
        return phi_hat
    
    def _create_uniform_initial_conditions(self) -> np.ndarray:
        """Create uniform initial conditions."""
        nx, ny, nz = self.grid
        phi_hat = np.zeros((self.M, nx, ny, nz), dtype=np.float32)
        
        seeding_densities = self.ic_config.get("seeding_densities", {})
        if not seeding_densities:
            seeding_densities = {label: 0.1 for label in self.labels}
        
        # Get noise scale from config or use default
        noise_scale = self.ic_config.get("seeding_noise_scale", 0.2)
        
        for m, label in enumerate(self.labels):
            density = seeding_densities.get(label, 0.1)
            # Create uniform base density
            base_density = np.full((nx, ny, nz), density, dtype=np.float32)
            # Add noise to create spatial variation
            noisy_density = self._add_seeding_noise(base_density, noise_scale)
            phi_hat[m, :, :, :] = noisy_density
        
        return phi_hat
    
    def _create_custom_initial_conditions(self) -> np.ndarray:
        """Create custom initial conditions using a function or predefined pattern."""
        nx, ny, nz = self.grid
        phi_hat = np.zeros((self.M, nx, ny, nz), dtype=np.float32)
        
        # This could be extended to support custom functions
        # For now, fall back to spherical
        print("Warning: Custom initial conditions not implemented, falling back to spherical")
        return self._create_spherical_initial_conditions()
    
    def _initialize_nutrient_field(self, phi_hat=None) -> np.ndarray:
        """
        Initialize nutrient field based on configuration.
        
        Args:
            phi_hat: Cell fraction fields (M, nx, ny, nz) - required for natural initialization
            
        Returns:
            nutrient_field: Initial nutrient field
        """
        nx, ny, nz = self.grid
        
        from src.growkit.PhysicsEngine.Nutrient.NutrientField import NutrientField
        nutrient_manager = NutrientField(self.cfg, self.cfg["populations"])
        
        # Get nutrient initialization type from config
        nutrient_config = self.ic_config.get("nutrient", {})
        initialization_type = nutrient_config.get("type", "uniform")
        
        # Initialize nutrient field with the specified type
        if initialization_type == "natural" and phi_hat is not None:
            return nutrient_manager.initialize_nutrient_field(
                (nx, ny, nz), 
                phi_hat=phi_hat, 
                initialization_type="natural"
            )
        elif initialization_type == "gradient":
            return nutrient_manager.initialize_nutrient_field(
                (nx, ny, nz), 
                initialization_type="gradient"
            )
        else:
            # Default to uniform
            return nutrient_manager.initialize_nutrient_field(
                (nx, ny, nz), 
                initialization_type="uniform"
            )
    
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
