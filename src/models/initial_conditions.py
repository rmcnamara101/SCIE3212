# File: src/models/initial_conditions.py

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from copy import deepcopy
import os

class InitialCondition(ABC):
    """
    Abstract base class for initial conditions.
    """

    def __init__(self, grid_shape: Tuple):
        self.grid_shape = grid_shape
        self.phi_H = np.zeros(grid_shape)
        self.phi_D = np.zeros(grid_shape)
        self.phi_N = np.zeros(grid_shape)
        self.nutrient = np.zeros(grid_shape)
        self.n_H = np.zeros(grid_shape)
        self.n_D = np.zeros(grid_shape)
        self.phi_h = np.zeros(grid_shape)
        
    @abstractmethod
    def initialize(self, params: dict):
        """
        Initialize the cell and nutrient fields based on the specific initial condition.
        """
        pass

    def visualize_3d(self, threshold=0.1):
        """
        Creates a 3D visualization of the tumor using matplotlib.
        
        Parameters:
        -----------
        threshold : float
            Value threshold for plotting cells
        """
        from mpl_toolkits.mplot3d import Axes3D
        
        # Combine cell types for visualization
        total_cells = self.phi_H + self.phi_D + self.phi_N
        
        # Get coordinates where cells exist
        coords = np.where(total_cells > threshold)
        
        # Create figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create color map based on cell types
        colors = np.zeros((len(coords[0]), 3))
        for i in range(len(coords[0])):
            z, y, x = coords[0][i], coords[1][i], coords[2][i]
            # RGB: healthy=red, dead=green, necrotic=blue
            colors[i, 0] = self.phi_H[z, y, x] / (total_cells[z, y, x] + 1e-6)
            colors[i, 1] = self.phi_D[z, y, x] / (total_cells[z, y, x] + 1e-6)
            colors[i, 2] = self.phi_N[z, y, x] / (total_cells[z, y, x] + 1e-6)
        
        # Plot the points
        ax.scatter(coords[2], coords[1], coords[0], c=colors, s=5, alpha=0.7)
        
        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Tumor Visualization')
        
        return fig

class SphericalTumor(InitialCondition):
    """
    Initial condition with a spherical tumor at the center of the grid.
    """

    def __init__(self, grid_shape: Tuple, radius: int = 5, nutrient_value: float = 1):
        super().__init__(grid_shape)
        self.radius = radius
        self.nutrient_value = nutrient_value
    
    def initialize(self, params):
        # Initialize nutrient field
        Nz, Ny, Nx = self.grid_shape
        self.nutrient = self.nutrient_value * np.ones(self.grid_shape)
        
        # Use broadcasting for parameter fields
        self.n_H = 0.2 * np.ones(self.grid_shape)
        self.n_D = 0.2 * np.ones(self.grid_shape)
        
        # Create slightly off-center coordinates
        center = np.array([
            Nz//2 ,  # Shift 2 cells in z direction
            Ny//2 ,  # Shift 1 cell in negative y direction
            Nx//2   # Shift 1 cell in x direction
        ])
        
        for k in range(Nz):
            for j in range(Ny):
                for i in range(Nx):
                    r = np.sqrt((k - center[0])**2 + (j - center[1])**2 + (i - center[2])**2)
                    if r < self.radius:  # Using the class radius parameter
                        self.phi_H[k, j, i] = 0.35 #* np.exp(-r**2 / 10)  # Gaussian tumor
                        self.phi_D[k, j, i] = 0.6 #* np.exp(-r**2 / 10)
                        self.phi_N[k, j, i] = 0.5# * np.exp(-r**2 / 10)

class EllipsoidTumor(InitialCondition):
    """
    Initial condition with an ellipsoidal tumor.
    This allows for different radii in each dimension.
    """

    def __init__(self, grid_shape: Tuple, radii: Tuple[int, int, int] = (7, 5, 3), nutrient_value: float = 1):
        super().__init__(grid_shape)
        self.radii = radii  # (z_radius, y_radius, x_radius)
        self.nutrient_value = nutrient_value
    
    def initialize(self, params):
        # Initialize nutrient field
        Nz, Ny, Nx = self.grid_shape
        self.nutrient = self.nutrient_value * np.ones(self.grid_shape)
        
        # Use broadcasting for parameter fields
        self.n_H = 0.2 * np.ones(self.grid_shape)
        self.n_D = 0.2 * np.ones(self.grid_shape)
        
        center = np.array([Nz//2, Ny//2, Nx//2])
        
        for k in range(Nz):
            for j in range(Ny):
                for i in range(Nx):
                    # Normalized distance for ellipsoid
                    dist = ((k - center[0])/self.radii[0])**2 + \
                           ((j - center[1])/self.radii[1])**2 + \
                           ((i - center[2])/self.radii[2])**2
                    
                    if dist < 1.0:  # Inside the ellipsoid
                        self.phi_H[k, j, i] = 0.35
                        self.phi_D[k, j, i] = 0.6
                        self.phi_N[k, j, i] = 0.5

class MultipleTumors(InitialCondition):
    """
    Initial condition with multiple spherical tumors at specified locations.
    """

    def __init__(self, grid_shape: Tuple, centers=None, radii=None, nutrient_value: float = 1):
        super().__init__(grid_shape)
        # Default: 2 tumors with radius 5
        self.centers = centers if centers is not None else [
            (0.3, 0.3, 0.3),  # Positions as fraction of grid dimensions
            (0.7, 0.7, 0.7)
        ]
        self.radii = radii if radii is not None else [5, 4]
        self.nutrient_value = nutrient_value
    
    def initialize(self, params):
        Nz, Ny, Nx = self.grid_shape
        self.nutrient = self.nutrient_value * np.ones(self.grid_shape)
        self.n_H = 0.2 * np.ones(self.grid_shape)
        self.n_D = 0.2 * np.ones(self.grid_shape)
        
        # Convert relative positions to grid indices
        grid_centers = []
        for center in self.centers:
            grid_centers.append((
                int(center[0] * Nz),
                int(center[1] * Ny),
                int(center[2] * Nx)
            ))
        
        for t, (center, radius) in enumerate(zip(grid_centers, self.radii)):
            for k in range(Nz):
                for j in range(Ny):
                    for i in range(Nx):
                        r = np.sqrt((k - center[0])**2 + 
                                    (j - center[1])**2 + 
                                    (i - center[2])**2)
                        if r < radius:
                            self.phi_H[k, j, i] = 0.35
                            self.phi_D[k, j, i] = 0.6
                            self.phi_N[k, j, i] = 0.5

class LayeredTumor(InitialCondition):
    """
    Initial condition with a tumor having concentric layers:
    - Outer layer: Healthy cells
    - Middle layer: Dead cells
    - Inner core: Necrotic cells
    """

    def __init__(self, grid_shape: Tuple, outer_radius: int = 8, 
                 middle_radius: int = 5, inner_radius: int = 3, 
                 nutrient_value: float = 1):
        super().__init__(grid_shape)
        self.outer_radius = outer_radius
        self.middle_radius = middle_radius
        self.inner_radius = inner_radius
        self.nutrient_value = nutrient_value
    
    def initialize(self, params):
        Nz, Ny, Nx = self.grid_shape
        self.nutrient = self.nutrient_value * np.ones(self.grid_shape)
        self.n_H = 0.2 * np.ones(self.grid_shape)
        self.n_D = 0.2 * np.ones(self.grid_shape)
        
        center = np.array([Nz//2, Ny//2, Nx//2])
        
        for k in range(Nz):
            for j in range(Ny):
                for i in range(Nx):
                    r = np.sqrt((k - center[0])**2 + (j - center[1])**2 + (i - center[2])**2)
                    
                    if r < self.inner_radius:
                        # Necrotic core
                        self.phi_N[k, j, i] = 0.9
                    elif r < self.middle_radius:
                        # Dead cell layer
                        self.phi_D[k, j, i] = 0.8
                    elif r < self.outer_radius:
                        # Healthy cell layer
                        self.phi_H[k, j, i] = 0.7

class InvasiveTumor(InitialCondition):
    """
    Initial condition with a tumor having finger-like projections
    that simulate invasive growth patterns.
    """

    def __init__(self, grid_shape: Tuple, base_radius: int = 5, 
                 num_projections: int = 6, projection_length: int = 4,
                 nutrient_value: float = 1):
        super().__init__(grid_shape)
        self.base_radius = base_radius
        self.num_projections = num_projections
        self.projection_length = projection_length
        self.nutrient_value = nutrient_value
    
    def initialize(self, params):
        Nz, Ny, Nx = self.grid_shape
        self.nutrient = self.nutrient_value * np.ones(self.grid_shape)
        self.n_H = 0.2 * np.ones(self.grid_shape)
        self.n_D = 0.2 * np.ones(self.grid_shape)
        
        center = np.array([Nz//2, Ny//2, Nx//2])
        
        # Generate random directions for projections
        np.random.seed(42)  # For reproducibility
        directions = []
        for _ in range(self.num_projections):
            # Random unit vector in 3D
            v = np.random.randn(3)
            v = v / np.linalg.norm(v)
            directions.append(v)
        
        for k in range(Nz):
            for j in range(Ny):
                for i in range(Nx):
                    pos = np.array([k, j, i])
                    r_base = np.linalg.norm(pos - center)
                    
                    # Check if in base tumor
                    in_tumor = r_base < self.base_radius
                    
                    # Check if in any projection
                    for dir_vec in directions:
                        # Project point onto direction vector
                        v = pos - center
                        proj_len = np.dot(v, dir_vec)
                        
                        # Distance from projection line
                        dist_from_line = np.linalg.norm(v - proj_len * dir_vec)
                        
                        # Check if point is in the projection
                        if (0 < proj_len < self.projection_length and 
                            dist_from_line < self.base_radius * 0.6):
                            in_tumor = True
                            break
                    
                    if in_tumor:
                        self.phi_H[k, j, i] = 0.35
                        self.phi_D[k, j, i] = 0.5
                        self.phi_N[k, j, i] = 0.2

class RandomBlobTumor(InitialCondition):
    """
    Initial condition with an irregular, blob-like tumor shape.
    Creates a non-symmetric tumor with random perturbations to match
    more realistic tumor morphologies seen in medical imaging.
    Uses constant cell density throughout the tumor.
    """

    def __init__(self, grid_shape: Tuple, base_radius: int = 6, 
                 irregularity: float = 0.5, spikiness: float = 0.3,
                 noise_scale: float = 2.0, nutrient_value: float = 1):
        super().__init__(grid_shape)
        self.base_radius = base_radius
        self.irregularity = irregularity  # 0-1, how irregular the blob is
        self.spikiness = spikiness  # 0-1, how spiky vs smooth the perturbations are
        self.noise_scale = noise_scale  # Scale of the noise features
        self.nutrient_value = nutrient_value
    
    def initialize(self, params):
        Nz, Ny, Nx = self.grid_shape
        self.nutrient = self.nutrient_value * np.ones(self.grid_shape)
        self.n_H = 0.2 * np.ones(self.grid_shape)
        self.n_D = 0.2 * np.ones(self.grid_shape)
        
        center = np.array([Nz//2, Ny//2, Nx//2])
        
        # Generate a noise field for perturbations
        # Using random seed for reproducibility
        np.random.seed(42)
        
        # Create a 3D coordinate grid for noise calculation
        noise_grid = np.zeros(self.grid_shape)
        
        # Fill the grid with random perturbation values
        raw_noise = np.random.randn(*self.grid_shape)
        
        # Simple gaussian blur to smooth the noise (3D convolution)
        from scipy.ndimage import gaussian_filter
        noise_grid = gaussian_filter(raw_noise, sigma=self.noise_scale * (1 - self.spikiness))
        
        # Normalize noise to [-1, 1] range
        noise_grid = noise_grid / np.max(np.abs(noise_grid))
        
        # Set constant cell density values
        cell_density_H = 0.35
        cell_density_D = 0.5
        cell_density_N = 0.2
        
        for k in range(Nz):
            for j in range(Ny):
                for i in range(Nx):
                    pos = np.array([k, j, i])
                    # Basic distance from center
                    r_base = np.linalg.norm(pos - center)
                    
                    # Get noise value at this point for radius perturbation
                    noise_val = noise_grid[k, j, i]
                    
                    # Calculate the perturbed radius threshold for this point
                    radius_perturbation = self.base_radius * self.irregularity * noise_val
                    radius_threshold = self.base_radius + radius_perturbation
                    
                    # If within the perturbed radius, it's part of the tumor
                    if r_base < radius_threshold:
                        # Use constant density values instead of varying with noise
                        self.phi_H[k, j, i] = cell_density_H
                        self.phi_D[k, j, i] = cell_density_D
                        self.phi_N[k, j, i] = cell_density_N

class PerlinBlobTumor(InitialCondition):
    """
    Initial condition generating a highly realistic, irregular tumor shape
    using multiple octaves of noise to create natural-looking perturbations.
    """
    
    def __init__(self, grid_shape: Tuple, base_radius: int = 6, 
                 irregularity: float = 0.5, octaves: int = 3, 
                 persistence: float = 0.5, nutrient_value: float = 1):
        super().__init__(grid_shape)
        self.base_radius = base_radius
        self.irregularity = irregularity
        self.octaves = octaves  # Number of noise layers to combine
        self.persistence = persistence  # How quickly amplitude decreases per octave
        self.nutrient_value = nutrient_value
    
    def initialize(self, params):
        Nz, Ny, Nx = self.grid_shape
        self.nutrient = self.nutrient_value * np.ones(self.grid_shape)
        self.n_H = 0.2 * np.ones(self.grid_shape)
        self.n_D = 0.2 * np.ones(self.grid_shape)
        
        center = np.array([Nz//2, Ny//2, Nx//2])
        
        # Generate noise field with multiple octaves
        np.random.seed(42)
        
        # Generate base noise grid
        noise_grid = np.zeros(self.grid_shape)
        
        from scipy.ndimage import gaussian_filter
        
        # Generate multiple octaves of noise
        max_amplitude = 0
        amplitude = 1.0
        frequency = 1.0
        
        for octave in range(self.octaves):
            # Generate new random noise at this frequency
            octave_noise = np.random.randn(*self.grid_shape)
            
            # Smooth by different amounts based on frequency
            sigma = 3.0 / frequency
            smoothed_noise = gaussian_filter(octave_noise, sigma=sigma)
            
            # Add to the total noise field with appropriate amplitude
            noise_grid += smoothed_noise * amplitude
            
            # Keep track of maximum possible amplitude for normalization later
            max_amplitude += amplitude
            
            # Update amplitude and frequency for next octave
            amplitude *= self.persistence
            frequency *= 2.0
        
        # Normalize the noise field
        noise_grid = noise_grid / max_amplitude
        
        for k in range(Nz):
            for j in range(Ny):
                for i in range(Nx):
                    pos = np.array([k, j, i])
                    r_base = np.linalg.norm(pos - center)
                    
                    # Get noise value for this position
                    noise_val = noise_grid[k, j, i]
                    
                    # Calculate perturbed radius
                    radius_perturbation = self.base_radius * self.irregularity * noise_val
                    radius_threshold = self.base_radius + radius_perturbation
                    
                    # Apply radial distance factor to make perturbations more pronounced at surface
                    # This creates a more blob-like appearance rather than just a noisy sphere
                    radial_factor = min(1.0, r_base / self.base_radius)
                    effective_threshold = self.base_radius + radius_perturbation * radial_factor
                    
                    if r_base < effective_threshold:
                        # Vary cell densities based on distance from center
                        center_factor = 1.0 - (r_base / self.base_radius) * 0.3
                        
                        self.phi_H[k, j, i] = 0.35 * center_factor
                        self.phi_D[k, j, i] = 0.5 * (1 + 0.2 * noise_val) * center_factor
                        self.phi_N[k, j, i] = 0.2 * center_factor

def create_3d_plot(ax, ic_obj, title=None, threshold=0.2, subsample=5):
    """
    Create a 3D plot of the initial condition on the given axis.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis to plot on
    ic_obj : InitialCondition
        The initial condition object
    title : str, optional
        Title for the plot
    threshold : float
        Value threshold for plotting cells (higher = fewer points)
    subsample : int
        Only plot every Nth point to reduce clutter (higher = fewer points)
    """
    # Combine cell types for visualization
    total_cells = ic_obj.phi_H + ic_obj.phi_D + ic_obj.phi_N
    
    # Get coordinates where cells exist with higher threshold
    coords = np.where(total_cells > threshold)
    
    if len(coords[0]) == 0:
        ax.text(0.5, 0.5, 0.5, "No cells found", transform=ax.transAxes, 
                ha='center', fontsize=12, color='red')
        ax.set_title(title)
        return
    
    # Subsample the points to reduce density
    coords_subsample = (
        coords[0][::subsample],
        coords[1][::subsample],
        coords[2][::subsample]
    )
    
    # Create color map based on cell types
    colors = np.zeros((len(coords_subsample[0]), 3))
    for i in range(len(coords_subsample[0])):
        z, y, x = coords_subsample[0][i], coords_subsample[1][i], coords_subsample[2][i]
        # RGB: healthy=red, dead=green, necrotic=blue
        colors[i, 0] = ic_obj.phi_H[z, y, x] / (total_cells[z, y, x] + 1e-6)
        colors[i, 1] = ic_obj.phi_D[z, y, x] / (total_cells[z, y, x] + 1e-6)
        colors[i, 2] = ic_obj.phi_N[z, y, x] / (total_cells[z, y, x] + 1e-6)
    
    # Plot the points with larger marker size
    ax.scatter(coords_subsample[2], coords_subsample[1], coords_subsample[0], 
              c=colors, s=15, alpha=0.8, edgecolors='none')
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set fixed limits based on grid shape to avoid excessive zoom
    grid_size = ic_obj.grid_shape
    ax.set_xlim(0, grid_size[2])
    ax.set_ylim(0, grid_size[1])
    ax.set_zlim(0, grid_size[0])
    
    # Add a subtitle with point count info
    point_count = len(coords_subsample[0])
    total_points = len(coords[0])
    if title:
        ax.set_title(f"{title}\n{point_count} points plotted of {total_points} total")

def visualize_all_shapes(grid_shape=(60, 60, 60), save_path=None, threshold=0.2, subsample=5):
    """
    Create and visualize all tumor initial conditions with a larger grid.
    
    Parameters:
    -----------
    grid_shape : tuple
        Shape of the grid to use for all initial conditions
    save_path : str, optional
        Path to save the figure. If None, the figure is displayed.
    threshold : float
        Value threshold for plotting cells (higher = fewer points)
    subsample : int
        Only plot every Nth point to reduce clutter
    """
    # Import required modules here for self-contained function
    from matplotlib.gridspec import GridSpec
    
    # Define all the initial condition classes and their parameters
    # Scale up the tumor sizes to match the larger grid
    ic_classes = [
        (SphericalTumor, {"radius": 12, "nutrient_value": 1.0}),
        (EllipsoidTumor, {"radii": (16, 10, 6), "nutrient_value": 1.0}),
        (MultipleTumors, {"centers": [(0.35, 0.35, 0.35), (0.65, 0.65, 0.65)], 
                          "radii": [10, 8], "nutrient_value": 1.0}),
        (LayeredTumor, {"outer_radius": 16, "middle_radius": 10, 
                        "inner_radius": 6, "nutrient_value": 1.0}),
        (InvasiveTumor, {"base_radius": 10, "num_projections": 6, 
                         "projection_length": 8, "nutrient_value": 1.0}),
        (RandomBlobTumor, {"base_radius": 12, "irregularity": 0.5, 
                           "spikiness": 0.3, "noise_scale": 2.0, "nutrient_value": 1.0}),
        (PerlinBlobTumor, {"base_radius": 12, "irregularity": 0.6, 
                           "octaves": 3, "persistence": 0.5, "nutrient_value": 1.0})
    ]
    
    # Set up the figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 3, figure=fig)
    
    # Dictionary to store all initial condition objects
    ic_objects = {}
    
    # Parameters for model initialization
    params = {}  # Empty parameters as our IC classes don't require much input
    
    # Initialize and visualize each initial condition
    for i, (ic_class, ic_params) in enumerate(ic_classes):
        class_name = ic_class.__name__
        print(f"Processing {class_name}...")
        
        # Create and initialize the IC object
        ic_obj = ic_class(grid_shape=grid_shape, **ic_params)
        ic_obj.initialize(params)
        ic_objects[class_name] = ic_obj
        
        # Create a subplot
        row, col = i // 3, i % 3
        ax = fig.add_subplot(gs[row, col], projection='3d')
        
        # Create the visualization with the specified threshold and subsampling
        create_3d_plot(ax, ic_obj, title=class_name, threshold=threshold, subsample=subsample)
        
        # Set better view angle for clarity
        ax.view_init(elev=30, azim=45)
    
    # Add an empty plot for the legend in the last position
    if len(ic_classes) < 9:
        ax = fig.add_subplot(gs[2, 2])
        ax.axis('off')
        ax.text(0.5, 0.9, 'Cell Types', fontsize=12, fontweight='bold', ha='center')
        ax.text(0.5, 0.7, 'Red: Healthy Cells', fontsize=10, color='red', ha='center')
        ax.text(0.5, 0.5, 'Green: Dead Cells', fontsize=10, color='green', ha='center')
        ax.text(0.5, 0.3, 'Blue: Necrotic Cells', fontsize=10, color='blue', ha='center')
        ax.text(0.5, 0.1, f'Threshold: {threshold}, Subsample: 1/{subsample}', 
                fontsize=9, ha='center', style='italic')
    
    # Set overall title
    fig.suptitle('Tumor Initial Condition Shapes', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save or display the figure
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    return ic_objects

def create_2d_slices(grid_shape=(60, 60, 60), save_path=None):
    """
    Create and visualize 2D slices through the center of each tumor shape.
    This can be helpful to see internal structures.
    
    Parameters are the same as visualize_all_shapes()
    """
    # Similar to visualize_all_shapes but with 2D slices instead
    ic_classes = [
        (SphericalTumor, {"radius": 12, "nutrient_value": 1.0}),
        (EllipsoidTumor, {"radii": (16, 10, 6), "nutrient_value": 1.0}),
        (MultipleTumors, {"centers": [(0.35, 0.35, 0.35), (0.65, 0.65, 0.65)], 
                          "radii": [10, 8], "nutrient_value": 1.0}),
        (LayeredTumor, {"outer_radius": 16, "middle_radius": 10, 
                        "inner_radius": 6, "nutrient_value": 1.0}),
        (InvasiveTumor, {"base_radius": 10, "num_projections": 6, 
                         "projection_length": 8, "nutrient_value": 1.0}),
        (RandomBlobTumor, {"base_radius": 12, "irregularity": 0.5, 
                           "spikiness": 0.3, "noise_scale": 2.0, "nutrient_value": 1.0}),
        (PerlinBlobTumor, {"base_radius": 12, "irregularity": 0.6, 
                           "octaves": 3, "persistence": 0.5, "nutrient_value": 1.0})
    ]
    
    fig, axs = plt.subplots(len(ic_classes), 3, figsize=(18, 4*len(ic_classes)))
    
    params = {}
    for i, (ic_class, ic_params) in enumerate(ic_classes):
        class_name = ic_class.__name__
        print(f"Processing 2D slices for {class_name}...")
        
        # Create and initialize the IC object
        ic_obj = ic_class(grid_shape=grid_shape, **ic_params)
        ic_obj.initialize(params)
        
        # Get the middle slice
        z_middle = grid_shape[0] // 2
        
        # Plot the three cell types
        im1 = axs[i, 0].imshow(ic_obj.phi_H[z_middle], cmap='Reds', vmin=0, vmax=1)
        axs[i, 0].set_title(f"{class_name} - Healthy Cells")
        plt.colorbar(im1, ax=axs[i, 0], fraction=0.046, pad=0.04)
        
        im2 = axs[i, 1].imshow(ic_obj.phi_D[z_middle], cmap='Greens', vmin=0, vmax=1)
        axs[i, 1].set_title(f"{class_name} - Dead Cells")
        plt.colorbar(im2, ax=axs[i, 1], fraction=0.046, pad=0.04)
        
        im3 = axs[i, 2].imshow(ic_obj.phi_N[z_middle], cmap='Blues', vmin=0, vmax=1)
        axs[i, 2].set_title(f"{class_name} - Necrotic Cells")
        plt.colorbar(im3, ax=axs[i, 2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # Save or display the figure
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"2D slices saved to {save_path}")
    else:
        plt.show()

# Add this to the end of your initial_conditions.py file to run the visualization
if __name__ == "__main__":
    # Create directory for saved figures
    output_dir = "outputs/visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Use a larger grid size (60x60x60)
    grid_size = (60, 60, 60)
    
    print("Generating 3D visualizations...")
    # Increase threshold and subsampling to reduce point density
    ic_objects = visualize_all_shapes(
        grid_shape=grid_size,
        threshold=0.25,  # Higher threshold = fewer points (was 0.1)
        subsample=6,     # Only plot every 6th point
        save_path=f"{output_dir}/initial_conditions_3d.png"
    )
    
    print("\nGenerating 2D slice visualizations...")
    create_2d_slices(
        grid_shape=grid_size,
        save_path=f"{output_dir}/initial_conditions_2d_slices.png"
    )
    
    # Add an additional visualization showing combined cell types
    print("\nGenerating combined cell type visualizations...")
    fig, axs = plt.subplots(3, 3, figsize=(18, 18))
    
    names = list(ic_objects.keys())
    for i in range(min(len(names), 9)):
        row, col = i // 3, i % 3
        ic = ic_objects[names[i]]
        
        # Middle slice
        z_middle = grid_size[0] // 2
        
        # Create RGB image with combined cell types
        rgb_img = np.zeros((*grid_size[1:], 3))
        rgb_img[..., 0] = ic.phi_H[z_middle]  # Red = Healthy
        rgb_img[..., 1] = ic.phi_D[z_middle]  # Green = Dead
        rgb_img[..., 2] = ic.phi_N[z_middle]  # Blue = Necrotic
        
        # Normalize if needed to ensure proper visibility
        max_val = np.max(rgb_img)
        if max_val > 0:
            rgb_img = rgb_img / max_val
        
        axs[row, col].imshow(rgb_img)
        axs[row, col].set_title(f"{names[i]}")
        axs[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/initial_conditions_combined.png", dpi=300, bbox_inches='tight')
    print(f"Combined visualization saved to {output_dir}/initial_conditions_combined.png")
    
    print("\nVisualization complete!")
