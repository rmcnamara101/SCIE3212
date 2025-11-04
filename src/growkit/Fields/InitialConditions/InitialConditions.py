"""

Initial Conditions Module

This module handles the initialization of simulation fields based on YAML configuration.
It supports multiple initial condition types and configurable seeding densities.

It supports the following initial condition types:
- spherical
- gaussian_random_blob
- multiple_spheres
- uniform
- organoid_settling
- main_spheroid_with_blobs
- custom

It also supports the following nutrient field initialization types:
- natural
- gradient
- uniform

It also supports the following seeding types:
- uniform
- gaussian noise

To adjust initial conditions, simply modify the YAML configuration file.

Example YAML configuration:

initial_conditions:
  type: gaussian_random_blob
  radius: 7
  center: [50, 50, 50]
  deformation_strength: 0.3
  seeding_densities:
    Tumour: 0.7
    Necrotic: 0.0
  nutrient:
    type: natural
    concentration: 1.0
    concentration_max: 1.0
    concentration_min: 0.0

Author: Riley Jae McNamara
Date: 2025-10-27

"""


# -- Standard imports --
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import yaml


# -- Module definition --  
class InitialConditions:
    """
    Handles initialization of simulation fields based on YAML configuration.
    Supports multiple initial condition types and configurable seeding densities.
    Supports multiple nutrient field initialization types.
    Supports multiple seeding types.
    Supports multiple noise scales.
    """
    
    def __init__(self, cfg: Dict):
        """
        Initialize the InitialConditions manager.
        
        Args:
            cfg: Configuration dictionary loaded from YAML
        """
        self.cfg = cfg
        self.grid = (cfg["domain"]["shape"], cfg["domain"]["shape"], cfg["domain"]["shape"])
        self.dx = np.float32(cfg["domain"]["dx"])
        
        # Get initial conditions configuration
        self.ic_config = cfg.get("initial_conditions", {})
        self.ic_type = self.ic_config.get("type", "spherical")
        
        # Get population information
        self.populations = cfg["populations"]
        self.M = len(self.populations)
        self.labels = [p["label"] for p in self.populations.values()]
        self.names = list(self.populations.keys())  # Population names (keys)
        
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
        elif self.ic_type == "organoid_settling":
            phi_hat = self._create_organoid_settling_initial_conditions()
        elif self.ic_type == "main_spheroid_with_blobs":
            phi_hat = self._create_main_spheroid_with_blobs_initial_conditions()
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
        """Create spherical initial conditions with smooth boundaries to reduce diamond artifacts."""
        nx, ny, nz = self.grid
        phi_hat = np.zeros((self.M, nx, ny, nz), dtype=np.float32)
        
        # Get configuration parameters
        center = self.ic_config.get("center", [nx//2, ny//2, nz//2])
        radius = self.ic_config.get("radius", min(nx, ny, nz) // 4)
        seeding_densities = self.ic_config.get("seeding_densities", {})
        
        # Default seeding densities if not specified
        if not seeding_densities:
            seeding_densities = {name: 0.8 if i == 0 else 0.1 / (self.M - 1) 
                               for i, name in enumerate(self.names)}
        
        # Get noise scale from config or use default
        noise_scale = self.ic_config.get("seeding_noise_scale", 0.2)
        
        # Create base density with GAUSSIAN profile to eliminate diamond artifacts
        # Gaussian profiles are naturally isotropic and should eliminate grid artifacts
        base_mask = np.zeros((nx, ny, nz), dtype=np.float32)
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    dist = np.sqrt((i - center[0])**2 + (j - center[1])**2 + (k - center[2])**2)
                    # Use Gaussian profile instead of step function
                    # This should eliminate diamond artifacts by being naturally isotropic
                    sigma = radius / 2.0  # Standard deviation controls width
                    density = np.exp(-(dist**2) / (2 * sigma**2))
                    base_mask[i, j, k] = density
        
        # Add noise to each population's seeding
        for m, name in enumerate(self.names):
            base_density = base_mask * seeding_densities.get(name, 0.0)
            # Only add noise if the base density is non-zero to avoid creating cells where none should exist
            if np.sum(base_density) > 0:
                noisy_density = self._add_seeding_noise(base_density, noise_scale)
            else:
                noisy_density = base_density  # Keep zero density populations at zero
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
        """Create deformed blob initial conditions with smooth boundaries to reduce diamond artifacts."""
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
        
        # Create the deformed blob with SMOOTH boundaries to reduce diamond artifacts
        total_density = np.zeros((nx, ny, nz), dtype=np.float32)
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # Calculate distance from center
                    dist = np.sqrt((i - center[0])**2 + (j - center[1])**2 + (k - center[2])**2)
                    
                    # Apply deformation to the radius
                    deformation = smoothed_deformation[i, j, k] * deformation_strength
                    deformed_radius = radius * (1 + deformation)
                    
                    # Use hard boundary (step function) instead of smooth transition
                    # This eliminates the hyperbolic tangent that may be causing circular artifacts
                    if dist <= deformed_radius:
                        density = 1.0
                    else:
                        density = 0.0
                    total_density[i, j, k] = density
        
        # Get noise scale from config or use default
        noise_scale = self.ic_config.get("seeding_noise_scale", 0.2)
        
        # Distribute density among populations with noise
        for m, name in enumerate(self.names):
            density_fraction = seeding_densities.get(name, 1.0 / self.M)
            base_density = total_density * density_fraction
            # Only add noise if the base density is non-zero to avoid creating cells where none should exist
            if np.sum(base_density) > 0:
                noisy_density = self._add_seeding_noise(base_density, noise_scale)
            else:
                noisy_density = base_density  # Keep zero density populations at zero
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
            seeding_densities = {name: 0.1 for name in self.names}
        
        # Get noise scale from config or use default
        noise_scale = self.ic_config.get("seeding_noise_scale", 0.2)
        
        for m, name in enumerate(self.names):
            density = seeding_densities.get(name, 0.1)
            # Create uniform base density
            base_density = np.full((nx, ny, nz), density, dtype=np.float32)
            # Only add noise if the base density is non-zero to avoid creating cells where none should exist
            if density > 0:
                noisy_density = self._add_seeding_noise(base_density, noise_scale)
            else:
                noisy_density = base_density  # Keep zero density populations at zero
            phi_hat[m, :, :, :] = noisy_density
        
        return phi_hat
    
    def _create_organoid_settling_initial_conditions(self) -> np.ndarray:
        """
        Create organoid settling initial conditions - a loose collection of cells 
        with decreasing density from center, simulating organoids settling under adhesion.
        """
        nx, ny, nz = self.grid
        phi_hat = np.zeros((self.M, nx, ny, nz), dtype=np.float32)
        
        # Get configuration parameters
        center = self.ic_config.get("center", [nx//2, ny//2, nz//2])
        core_radius = self.ic_config.get("core_radius", min(nx, ny, nz) // 8)  # Dense core radius
        outer_radius = self.ic_config.get("outer_radius", min(nx, ny, nz) // 3)  # Outer boundary
        core_density = self.ic_config.get("core_density", 0.7)  # Maximum density at center
        outer_density = self.ic_config.get("outer_density", 0.1)  # Minimum density at edge
        seeding_densities = self.ic_config.get("seeding_densities", {})
        
        # Set random seed for reproducible patterns
        np.random.seed(self.cfg.get("simulation", {}).get("seed", 42))
        
        # Default seeding densities if not specified
        if not seeding_densities:
            seeding_densities = {name: 0.8 if i == 0 else 0.1 / (self.M - 1) 
                               for i, name in enumerate(self.names)}
        
        # Create the density profile with multiple components
        total_density = np.zeros((nx, ny, nz), dtype=np.float32)
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # Calculate distance from center
                    dist = np.sqrt((i - center[0])**2 + (j - center[1])**2 + (k - center[2])**2)
                    
                    if dist <= outer_radius:
                        # Simple smooth falloff from center to edge
                        # Use a smooth exponential or power law decay
                        
                        # Normalize distance to [0, 1] range
                        normalized_dist = dist / outer_radius
                        
                        # Create smooth falloff using exponential decay
                        # This gives a natural organoid-like density distribution
                        falloff_factor = np.exp(-3.0 * normalized_dist)  # Adjust -3.0 to control steepness
                        
                        # Scale from core_density at center to outer_density at edge
                        density = outer_density + (core_density - outer_density) * falloff_factor
                        
                        # Add some randomness to create loose, irregular structure
                        random_factor = 0.9 + 0.2 * np.random.random()  # 0.9 to 1.1 (small variation)
                        density *= random_factor
                        
                        # Add small-scale spatial variations to break symmetry
                        # Create small clusters and voids typical of organoid settling
                        spatial_variation = 1.0 + 0.1 * np.sin(2 * np.pi * i / 6) * np.cos(2 * np.pi * j / 6) * np.sin(2 * np.pi * k / 6)
                        density *= spatial_variation
                        
                        # Ensure density is always positive
                        density = max(0.0, density)
                        
                        total_density[i, j, k] = density
        
        # Normalize to ensure maximum density doesn't exceed 1.0
        max_density = np.max(total_density)
        if max_density > 0:
            total_density = total_density / max_density * 0.8  # Scale to 80% max to leave room for other populations
        
        # Get noise scale from config or use default (lower for organoid settling)
        noise_scale = self.ic_config.get("seeding_noise_scale", 0.15)
        
        # Distribute density among populations with noise
        for m, name in enumerate(self.names):
            density_fraction = seeding_densities.get(name, 1.0 / self.M)
            base_density = total_density * density_fraction
            
            # Only add noise if the base density is non-zero
            if np.sum(base_density) > 0:
                noisy_density = self._add_seeding_noise(base_density, noise_scale)
            else:
                noisy_density = base_density
            
            phi_hat[m, :, :, :] = noisy_density
        
        return phi_hat
    
    def _create_main_spheroid_with_blobs_initial_conditions(self) -> np.ndarray:
        """
        Create initial conditions with one main spheroid and multiple smaller blobs around it.
        This is designed for testing coagulation dynamics where smaller blobs should
        aggregate toward the main spheroid under gravity and bowl potential forces.
        """
        nx, ny, nz = self.grid
        phi_hat = np.zeros((self.M, nx, ny, nz), dtype=np.float32)
        
        # Get configuration parameters
        center = self.ic_config.get("center", [nx//2, ny//2, nz//2])
        main_radius = self.ic_config.get("main_radius", min(nx, ny, nz) // 6)  # Main spheroid radius
        num_blobs = self.ic_config.get("num_blobs", 8)  # Number of smaller blobs
        blob_radius_range = self.ic_config.get("blob_radius_range", [2, 4])  # Min/max blob radius
        blob_distance_range = self.ic_config.get("blob_distance_range", [main_radius + 5, main_radius + 15])  # Distance from main spheroid
        seeding_densities = self.ic_config.get("seeding_densities", {})
        
        # Set random seed for reproducible patterns
        np.random.seed(self.cfg.get("simulation", {}).get("seed", 42))
        
        # Default seeding densities if not specified
        if not seeding_densities:
            seeding_densities = {name: 0.8 if i == 0 else 0.1 / (self.M - 1) 
                               for i, name in enumerate(self.names)}
        
        # Create the main spheroid at the center
        main_density = np.zeros((nx, ny, nz), dtype=np.float32)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    dist = np.sqrt((i - center[0])**2 + (j - center[1])**2 + (k - center[2])**2)
                    if dist <= main_radius:
                        # Use smooth falloff for the main spheroid
                        density = 1.0 - (dist / main_radius) * 0.3  # Slight falloff from center
                        main_density[i, j, k] = density
        
        # Create smaller blobs around the main spheroid
        blob_density = np.zeros((nx, ny, nz), dtype=np.float32)
        
        for blob_idx in range(num_blobs):
            # Generate random position around the main spheroid
            # Use spherical coordinates for better distribution
            theta = np.random.uniform(0, 2 * np.pi)  # Azimuthal angle
            phi = np.random.uniform(0, np.pi)  # Polar angle
            distance = np.random.uniform(blob_distance_range[0], blob_distance_range[1])
            
            # Convert to Cartesian coordinates
            blob_x = center[0] + distance * np.sin(phi) * np.cos(theta)
            blob_y = center[1] + distance * np.sin(phi) * np.sin(theta)
            blob_z = center[2] + distance * np.cos(phi)
            
            # Ensure blob is within domain bounds
            blob_x = np.clip(blob_x, 0, nx - 1)
            blob_y = np.clip(blob_y, 0, ny - 1)
            blob_z = np.clip(blob_z, 0, nz - 1)
            
            # Random blob radius
            blob_radius = np.random.uniform(blob_radius_range[0], blob_radius_range[1])
            
            # Create this blob
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        dist_to_blob = np.sqrt((i - blob_x)**2 + (j - blob_y)**2 + (k - blob_z)**2)
                        if dist_to_blob <= blob_radius:
                            # Use smooth falloff for blobs too
                            density = 1.0 - (dist_to_blob / blob_radius) * 0.2
                            blob_density[i, j, k] = max(blob_density[i, j, k], density)
        
        # Combine main spheroid and blobs
        total_density = main_density + blob_density
        
        # Normalize to prevent overflow
        max_density = np.max(total_density)
        if max_density > 1.0:
            total_density = total_density / max_density * 0.9  # Scale to 90% max
        
        # Get noise scale from config or use default
        noise_scale = self.ic_config.get("seeding_noise_scale", 0.1)  # Lower noise for cleaner blobs
        
        # Distribute density among populations with noise
        for m, name in enumerate(self.names):
            density_fraction = seeding_densities.get(name, 1.0 / self.M)
            base_density = total_density * density_fraction
            
            # Only add noise if the base density is non-zero
            if np.sum(base_density) > 0:
                noisy_density = self._add_seeding_noise(base_density, noise_scale)
            else:
                noisy_density = base_density
            
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
