import os
import sys

# Add parent directory to Python path to allow imports from src/
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from src.models.SCIE3121_model import SCIE3121_MODEL
from src.models.initial_conditions import SphericalTumor
from src.utils.utils import SCIE3121_params

class VolumeFractionPlotter:
    def __init__(self, model=None, simulation_data=None):
        """
        Initialize the plotter with either a model or simulation data.
        
        Args:
            model: A SCIE3121_MODEL instance (optional)
            simulation_data: Path to a saved simulation file or a dictionary of simulation data (optional)
        """
        if model is not None:
            self.model = model
            self.simulation_history = model.get_history()
            self.dx = model.dx
            self.grid_shape = model.grid_shape
        elif simulation_data is not None:
            self.load_simulation(simulation_data)
        else:
            raise ValueError("Either model or simulation_data must be provided")

    def load_simulation(self, simulation_data):
        """
        Load simulation data from a file or dictionary.
        
        Args:
            simulation_data: Path to a saved simulation file or a dictionary of simulation data
        """
        if isinstance(simulation_data, str):
            # Load from file
            if simulation_data.endswith('.npz'):
                data = np.load(simulation_data, allow_pickle=True)
                self.simulation_history = {key: data[key] for key in data.files}
            else:
                raise ValueError(f"Unsupported file format: {simulation_data}")
        elif isinstance(simulation_data, dict):
            # Use provided dictionary
            self.simulation_history = simulation_data
        else:
            raise ValueError("simulation_data must be a file path or dictionary")
        
        # Extract dx and grid_shape from the data
        if 'dx' in self.simulation_history:
            self.dx = float(self.simulation_history['Simulation Metadata']['dx'])
        else:
            self.dx = 1.0  # Default value
            print("Warning: dx not found in simulation data, using default value of 1.0")
        
        # Determine grid shape from the first volume fraction array
        for key in self.simulation_history:
            if key.endswith('cell volume fraction') and isinstance(self.simulation_history[key], np.ndarray):
                if len(self.simulation_history[key].shape) > 3:  # Time series of 3D arrays
                    self.grid_shape = self.simulation_history[key][0].shape
                else:  # Single 3D array
                    self.grid_shape = self.simulation_history[key].shape
                break
        else:
            raise ValueError("Could not determine grid shape from simulation data")

    def plot_volume_fractions(self, step: int, density_factors=None, thresholds=None, zoom_factor=0.8, point_size_factors=None, glow_effect=True, glow_size=3):
        """
        Plot a 3D scatter plot of volume fractions for all cell types at a given step,
        styled to resemble a fluorescently stained organoid with a black background.
        
        Args:
            step: Time step to visualize
            density_factors: Dict with density factors for each cell type {'Healthy': 0.1, 'Diseased': 0.2, 'Necrotic': 0.3}
                            or a single float to use the same factor for all types
            thresholds: Dict with threshold values for each cell type {'Healthy': 0.1, 'Diseased': 0.05, 'Necrotic': 0.2}
                       or a single float to use the same threshold for all types
            zoom_factor: Factor to control zoom level (0-1, lower means more zoomed in)
            point_size_factors: Dict with size multipliers for each cell type {'Healthy': 40, 'Diseased': 40, 'Necrotic': 200}
                               or a single float to use the same size factor for all types
            glow_effect: Whether to add a glow effect to make points appear more fluorescent
            glow_size: Size of the glow effect (higher values create more diffuse glow)
        """
        # Create figure and 3D axes
        fig = plt.figure(figsize=(10, 10), dpi=150)  # Higher DPI for better quality
        ax = fig.add_subplot(111, projection='3d')
        
        # Define cell types with fluorescent colors - using brighter, more saturated colors
        cell_types = {
            'Necrotic': ('necrotic cell volume fraction', '#FF3333'),    # Bright red
            'Diseased': ('diseased cell volume fraction', '#3366FF'),    # Bright blue
            'Healthy': ('healthy cell volume fraction', '#33FFFF')       # Bright cyan
        }
        
        # Set default density factors if not provided
        if density_factors is None:
            density_factors = {'Healthy': 0.3, 'Diseased': 0.3, 'Necrotic': 0.1}
        elif isinstance(density_factors, (int, float)):
            # If a single value is provided, use it for all cell types
            density_factors = {name: density_factors for name in cell_types.keys()}
        
        # Set default thresholds if not provided
        if thresholds is None:
            thresholds = {'Healthy': 0.05, 'Diseased': 0.05, 'Necrotic': 0.05}
        elif isinstance(thresholds, (int, float)):
            # If a single value is provided, use it for all cell types
            thresholds = {name: thresholds for name in cell_types.keys()}
        
        # Set default point size factors if not provided
        if point_size_factors is None:
            point_size_factors = {'Healthy': 40, 'Diseased': 40, 'Necrotic': 40}
        elif isinstance(point_size_factors, (int, float)):
            # If a single value is provided, use it for all cell types
            point_size_factors = {name: point_size_factors for name in cell_types.keys()}
        
        # Track the center of mass for zooming
        total_points = 0
        center_x, center_y, center_z = 0, 0, 0
        
        # Plot scatter points for each cell type
        for name, (key, color) in cell_types.items():
            if key in self.simulation_history:
                field = self.simulation_history[key][step] if len(self.simulation_history[key].shape) > 3 else self.simulation_history[key]
                
                # Get threshold for this cell type
                threshold = thresholds.get(name, 0.05)  # Default to 0.05 if not specified
                
                # Apply threshold to filter out low values
                x, y, z = np.where(field > threshold)
                fractions = field[field > threshold]
                
                # Get density factor for this cell type
                density_factor = density_factors.get(name, 0.3)  # Default to 0.3 if not specified
                
                # Reduce density by random sampling
                if density_factor < 5.0 and len(x) > 0:
                    num_points = max(1, int(len(x) * density_factor))
                    indices = np.random.choice(len(x), num_points, replace=False)
                    x, y, z = x[indices], y[indices], z[indices]
                    fractions = fractions[indices]
                
                # Update center of mass calculation
                if len(x) > 0:
                    center_x += np.sum(x)
                    center_y += np.sum(y)
                    center_z += np.sum(z)
                    total_points += len(x)
                
                # Get point size factor for this cell type
                size_factor = point_size_factors.get(name, 40)  # Default to 40 if not specified
                
                if glow_effect and len(x) > 0:
                    # Add glow effect by plotting multiple layers with decreasing opacity
                    # First layer: Core points (brightest)
                    ax.scatter(x * self.dx, y * self.dx, z * self.dx, 
                            c=color, s=size_factor * fractions, alpha=0.9, 
                            edgecolors='none', label=f"{name} (n={len(x)})")
                    
                    # Second layer: First glow layer
                    ax.scatter(x * self.dx, y * self.dx, z * self.dx, 
                            c=color, s=size_factor * fractions * glow_size, alpha=0.3, 
                            edgecolors='none')
                    
                    # Third layer: Outer glow (most diffuse)
                    ax.scatter(x * self.dx, y * self.dx, z * self.dx, 
                            c=color, s=size_factor * fractions * glow_size * 2, alpha=0.1, 
                            edgecolors='none')
                else:
                    # Original plotting without glow effect
                    ax.scatter(x * self.dx, y * self.dx, z * self.dx, 
                            c=color, s=size_factor * fractions, alpha=0.7, 
                            label=f"{name} (n={len(x)})")
                
                # Print statistics for debugging
                if len(x) > 0:
                    print(f"{name} cells: {len(x)} points, min value: {np.min(fractions):.6f}, max value: {np.max(fractions):.6f}, mean value: {np.mean(fractions):.6f}")
                else:
                    print(f"{name} cells: No points found above threshold {threshold}")
        
        # Calculate center of mass for zooming
        if total_points > 0:
            center_x = (center_x / total_points) * self.dx
            center_y = (center_y / total_points) * self.dx
            center_z = (center_z / total_points) * self.dx
        else:
            # Default to center of grid if no points
            center_x = self.grid_shape[0] * self.dx / 2
            center_y = self.grid_shape[1] * self.dx / 2
            center_z = self.grid_shape[2] * self.dx / 2
        
        # Calculate zoom boundaries
        grid_size_x = self.grid_shape[0] * self.dx
        grid_size_y = self.grid_shape[1] * self.dx
        grid_size_z = self.grid_shape[2] * self.dx
        
        # Calculate half-width of view based on zoom factor
        half_width_x = grid_size_x * zoom_factor / 2
        half_width_y = grid_size_y * zoom_factor / 2
        half_width_z = grid_size_z * zoom_factor / 2
        
        # Set axis limits centered on center of mass
        ax.set_xlim([max(0, center_x - half_width_x), min(grid_size_x, center_x + half_width_x)])
        ax.set_ylim([max(0, center_y - half_width_y), min(grid_size_y, center_y + half_width_y)])
        ax.set_zlim([max(0, center_z - half_width_z), min(grid_size_z, center_z + half_width_z)])
        
        # Customize appearance for fluorescent microscopy style
        ax.set_facecolor('black')  # Set axes background to black
        fig.patch.set_facecolor('black')  # Set figure background to black
        ax.grid(False)  # Remove grid
        
        # Remove axes panes and lines - updated for newer matplotlib versions
        # Set pane colors to transparent
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        # Make pane edges invisible
        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')
        
        # Hide axis lines
        ax.xaxis._axinfo["grid"]['color'] = (0,0,0,0)
        ax.yaxis._axinfo["grid"]['color'] = (0,0,0,0)
        ax.zaxis._axinfo["grid"]['color'] = (0,0,0,0)
        
        # Hide axis ticks and labels for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        
        # Add legend with white text for visibility
        #ax.legend(loc='upper right', facecolor='black', edgecolor='none', labelcolor='white')
        
        # Apply a slight blur effect to the entire figure (post-processing)
        plt.tight_layout()
        plt.show()

    def plot_total_volume_evolution(self):
        """
        Plot the total volume occupied by all cells over time.
        """
        if 'total cell volume fraction' in self.simulation_history and 'step' in self.simulation_history:
            total_volume = np.sum(self.simulation_history['total cell volume fraction'], axis=(1, 2, 3)) * (self.dx ** 3)
            
            plt.figure(figsize=(10, 6))
            plt.plot(self.simulation_history['step'], total_volume, 'gray', linestyle='--', label='Total Volume')
            plt.xlabel('Time Step')
            plt.ylabel('Total Volume')
            plt.title('Total Tumor Volume Evolution')
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            print("Required data for volume evolution plot not found in simulation history")

    def plot_radius_evolution(self):
        """
        Plot the radius evolution of the tumor.
        """
        if 'radius' in self.simulation_history and 'step' in self.simulation_history:
            plt.figure(figsize=(10, 6))
            plt.plot(self.simulation_history['step'], self.simulation_history['radius'], 'purple', label='Radius')
            plt.xlabel('Time Step')
            plt.ylabel('Radius')
            plt.title('Tumor Radius Evolution')
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            print("Radius data not found in simulation history")

    def plot_isosurface(self, step: int, cell_type: str = 'total', threshold: float = 0.1, 
                        color=None, alpha: float = 0.7, smooth_sigma: float = 1.0):
        """
        Plot an isosurface of the tumor at a given step.
        
        Args:
            step: Time step to visualize
            cell_type: 'total', 'healthy', 'diseased', or 'necrotic'
            threshold: Isosurface threshold value
            color: Color for the isosurface (defaults based on cell type)
            alpha: Transparency of the isosurface
            smooth_sigma: Gaussian smoothing sigma for the volume fraction field
        """
        # Map cell type to data key and default color
        cell_type_map = {
            'total': ('total cell volume fraction', 'gray'),
            'healthy': ('healthy cell volume fraction', 'yellow'),
            'diseased': ('diseased cell volume fraction', 'cyan'),
            'necrotic': ('necrotic cell volume fraction', 'white')
        }
        
        if cell_type not in cell_type_map:
            raise ValueError(f"Invalid cell type: {cell_type}. Must be one of {list(cell_type_map.keys())}")
        
        key, default_color = cell_type_map[cell_type]
        if color is None:
            color = default_color
        
        # Get the volume fraction field
        if key in self.simulation_history:
            field = self.simulation_history[key][step] if len(self.simulation_history[key].shape) > 3 else self.simulation_history[key]
        else:
            raise ValueError(f"Data for {cell_type} cells not found in simulation history")
        
        # Apply Gaussian smoothing
        field_smooth = gaussian_filter(field, sigma=smooth_sigma)
        
        # Create figure and 3D axes
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Generate isosurface using marching cubes
        try:
            verts, faces, _, _ = marching_cubes(field_smooth, threshold)
            
            # Scale vertices by dx
            verts = verts * self.dx
            
            # Create mesh
            mesh = Poly3DCollection(verts[faces])
            mesh.set_facecolor(color)
            mesh.set_alpha(alpha)
            mesh.set_edgecolor('none')
            
            # Add mesh to plot
            ax.add_collection3d(mesh)
            
            # Set axis limits
            ax.set_xlim([0, self.grid_shape[0] * self.dx])
            ax.set_ylim([0, self.grid_shape[1] * self.dx])
            ax.set_zlim([0, self.grid_shape[2] * self.dx])
            
            # Set labels
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            # Set title
            ax.set_title(f'{cell_type.capitalize()} Cell Isosurface (Step {step})')
            
            plt.show()
            
        except Exception as e:
            print(f"Error generating isosurface: {e}")
            print("Try adjusting the threshold or smooth_sigma parameters.")

    def _draw_bounding_box(self, ax, xlim, ylim, zlim):
        """
        Draw a simple wireframe bounding box to help visualize the overall spatial domain.
        """
        corners = np.array([[xlim[0], ylim[0], zlim[0]],
                            [xlim[1], ylim[0], zlim[0]],
                            [xlim[1], ylim[1], zlim[0]],
                            [xlim[0], ylim[1], zlim[0]],
                            [xlim[0], ylim[0], zlim[1]],
                            [xlim[1], ylim[0], zlim[1]],
                            [xlim[1], ylim[1], zlim[1]],
                            [xlim[0], ylim[1], zlim[1]]])
        edges = [(0,1), (1,2), (2,3), (3,0),
                 (4,5), (5,6), (6,7), (7,4),
                 (0,4), (1,5), (2,6), (3,7)]
        for edge in edges:
            pts = corners[list(edge), :]
            ax.plot(pts[:,0], pts[:,1], pts[:,2], color='gray', linestyle='--', linewidth=1)

    def plot_volume_fraction_cross_section(self, step: int, axis='z', slice_pos=None, 
                                          cell_types=None, alpha=0.7, cmap='viridis', 
                                          figsize=(12, 10), show_colorbar=True):
        """
        Plot a 2D cross-section of volume fractions for specified cell types.
        
        Args:
            step: Time step to visualize
            axis: Axis along which to take the cross-section ('x', 'y', or 'z')
            slice_pos: Position along the axis to slice (if None, uses the middle)
            cell_types: List of cell types to plot, e.g. ['Healthy', 'Diseased', 'Necrotic']
                       If None, plots all available types
            alpha: Transparency of the plots
            cmap: Colormap to use for the plots
            figsize: Figure size as (width, height)
            show_colorbar: Whether to show colorbars for each plot
        """
        # Define default cell types with their keys and display names
        default_cell_types = {
            'Necrotic': ('necrotic cell volume fraction', 'Necrotic Cells'),
            'Diseased': ('diseased cell volume fraction', 'Diseased Cells'),
            'Healthy': ('healthy cell volume fraction', 'Healthy Cells'),
            'Total': ('total cell volume fraction', 'Total Cell Density')
        }
        
        # Use specified cell types or default to all
        if cell_types is None:
            cell_types = list(default_cell_types.keys())
        
        # Filter to only include cell types that exist in the data
        available_types = []
        for name in cell_types:
            if name in default_cell_types and default_cell_types[name][0] in self.simulation_history:
                available_types.append(name)
        
        if not available_types:
            print("No requested cell types found in simulation data")
            return
        
        # Get the data for the specified step
        data = {}
        for name in available_types:
            key, _ = default_cell_types[name]
            if len(self.simulation_history[key].shape) > 3:  # Time series of 3D arrays
                data[name] = self.simulation_history[key][step]
            else:  # Single 3D array
                data[name] = self.simulation_history[key]
        
        # Determine slice position if not specified
        if slice_pos is None:
            if axis == 'x':
                slice_pos = self.grid_shape[0] // 2
            elif axis == 'y':
                slice_pos = self.grid_shape[1] // 2
            else:  # default to z
                slice_pos = self.grid_shape[2] // 2
        
        # Extract slices based on the specified axis
        slices = {}
        for name in available_types:
            if axis == 'x':
                slices[name] = data[name][slice_pos, :, :]
            elif axis == 'y':
                slices[name] = data[name][:, slice_pos, :]
            else:  # default to z
                slices[name] = data[name][:, :, slice_pos]
        
        # Set up the figure and subplots
        n_plots = len(available_types)
        n_cols = min(2, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_plots == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Create plots
        for i, name in enumerate(available_types):
            ax = axes[i]
            im = ax.imshow(slices[name].T, origin='lower', cmap=cmap, alpha=alpha,
                          extent=[0, self.grid_shape[0]*self.dx, 0, self.grid_shape[1]*self.dx])
            
            # Add colorbar if requested
            if show_colorbar:
                plt.colorbar(im, ax=ax, label='Volume Fraction')
            
            # Set title and labels
            _, display_name = default_cell_types[name]
            ax.set_title(f"{display_name} ({axis.upper()}={slice_pos})")
            
            # Set axis labels based on the slice orientation
            if axis == 'x':
                ax.set_xlabel('Y Position')
                ax.set_ylabel('Z Position')
            elif axis == 'y':
                ax.set_xlabel('X Position')
                ax.set_ylabel('Z Position')
            else:  # z
                ax.set_xlabel('X Position')
                ax.set_ylabel('Y Position')
        
        # Hide any unused subplots
        for i in range(n_plots, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()

    def plot_volume_fractions_2d(self, step: int, axis='z', slice_pos=None, density_factors=None, 
                               thresholds=None, point_size_factors=None, background_color='black'):
        """
        Plot a 2D cross-section of volume fractions for all cell types at a given step,
        styled to resemble a fluorescently stained tissue slice with a black background.
        
        Args:
            step: Time step to visualize
            axis: Axis along which to take the cross-section ('x', 'y', or 'z')
            slice_pos: Position along the axis to slice (if None, uses the middle)
            density_factors: Dict with density factors for each cell type {'Healthy': 0.1, 'Diseased': 0.2, 'Necrotic': 0.3}
                            or a single float to use the same factor for all types
            thresholds: Dict with threshold values for each cell type {'Healthy': 0.1, 'Diseased': 0.05, 'Necrotic': 0.2}
                       or a single float to use the same threshold for all types
            point_size_factors: Dict with size multipliers for each cell type {'Healthy': 40, 'Diseased': 40, 'Necrotic': 200}
                               or a single float to use the same size factor for all types
            background_color: Color for the plot background
        """
        # Create figure and axes
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111)
        
        # Define cell types with fluorescent colors (same as in plot_volume_fractions)
        cell_types = {
            'Necrotic': ('necrotic cell volume fraction', 'red'),
            'Diseased': ('diseased cell volume fraction', 'blue'),
            'Healthy': ('healthy cell volume fraction', 'cyan')
        }
        
        # Set default density factors if not provided
        if density_factors is None:
            density_factors = {'Healthy': 0.3, 'Diseased': 0.3, 'Necrotic': 0.1}
        elif isinstance(density_factors, (int, float)):
            # If a single value is provided, use it for all cell types
            density_factors = {name: density_factors for name in cell_types.keys()}
        
        # Set default thresholds if not provided
        if thresholds is None:
            thresholds = {'Healthy': 0.05, 'Diseased': 0.05, 'Necrotic': 0.05}
        elif isinstance(thresholds, (int, float)):
            # If a single value is provided, use it for all cell types
            thresholds = {name: thresholds for name in cell_types.keys()}
        
        # Set default point size factors if not provided
        if point_size_factors is None:
            point_size_factors = {'Healthy': 40, 'Diseased': 40, 'Necrotic': 40}
        elif isinstance(point_size_factors, (int, float)):
            # If a single value is provided, use it for all cell types
            point_size_factors = {name: point_size_factors for name in cell_types.keys()}
        
        # Determine slice position if not specified
        if slice_pos is None:
            if axis == 'x':
                slice_pos = self.grid_shape[0] // 2
            elif axis == 'y':
                slice_pos = self.grid_shape[1] // 2
            else:  # default to z
                slice_pos = self.grid_shape[2] // 2
        
        # Extract the appropriate 2D coordinates based on the axis
        if axis == 'x':
            coord_labels = ('Y', 'Z')
        elif axis == 'y':
            coord_labels = ('X', 'Z')
        else:  # z
            coord_labels = ('X', 'Y')
        
        # Plot scatter points for each cell type
        for name, (key, color) in cell_types.items():
            if key in self.simulation_history:
                # Get the volume fraction field for this step
                field = self.simulation_history[key][step] if len(self.simulation_history[key].shape) > 3 else self.simulation_history[key]
                
                # Extract the 2D slice based on the specified axis
                if axis == 'x':
                    slice_data = field[slice_pos, :, :]
                    y, z = np.meshgrid(np.arange(self.grid_shape[1]), np.arange(self.grid_shape[2]), indexing='ij')
                    coords = (y, z)
                elif axis == 'y':
                    slice_data = field[:, slice_pos, :]
                    x, z = np.meshgrid(np.arange(self.grid_shape[0]), np.arange(self.grid_shape[2]), indexing='ij')
                    coords = (x, z)
                else:  # z
                    slice_data = field[:, :, slice_pos]
                    x, y = np.meshgrid(np.arange(self.grid_shape[0]), np.arange(self.grid_shape[1]), indexing='ij')
                    coords = (x, y)
                
                # Get threshold for this cell type
                threshold = thresholds.get(name, 0.05)  # Default to 0.05 if not specified
                
                # Apply threshold to filter out low values
                mask = slice_data > threshold
                if not np.any(mask):
                    print(f"{name} cells: No points found above threshold {threshold} in this slice")
                    continue
                
                # Get coordinates and values above threshold
                c1, c2 = coords
                c1_points = c1[mask] * self.dx
                c2_points = c2[mask] * self.dx
                fractions = slice_data[mask]
                
                # Get density factor for this cell type
                density_factor = density_factors.get(name, 0.3)  # Default to 0.3 if not specified
                
                # Reduce density by random sampling
                if density_factor < 1.0 and len(c1_points) > 0:
                    num_points = max(1, int(len(c1_points) * density_factor))
                    indices = np.random.choice(len(c1_points), num_points, replace=False)
                    c1_points, c2_points = c1_points[indices], c2_points[indices]
                    fractions = fractions[indices]
                
                # Get point size factor for this cell type
                size_factor = point_size_factors.get(name, 40)  # Default to 40 if not specified
                
                # Plot with adjusted point size
                ax.scatter(c1_points, c2_points, c=color, s=size_factor * fractions, alpha=0.3, 
                          label=f"{name} (n={len(c1_points)})")
                
                # Print statistics for debugging
                if len(c1_points) > 0:
                    print(f"{name} cells: {len(c1_points)} points, min value: {np.min(fractions):.6f}, "
                          f"max value: {np.max(fractions):.6f}, mean value: {np.mean(fractions):.6f}")
        
        # Set axis limits
        if axis == 'x':
            ax.set_xlim([0, self.grid_shape[1] * self.dx])
            ax.set_ylim([0, self.grid_shape[2] * self.dx])
        elif axis == 'y':
            ax.set_xlim([0, self.grid_shape[0] * self.dx])
            ax.set_ylim([0, self.grid_shape[2] * self.dx])
        else:  # z
            ax.set_xlim([0, self.grid_shape[0] * self.dx])
            ax.set_ylim([0, self.grid_shape[1] * self.dx])
        
        # Set labels
        ax.set_xlabel(f'{coord_labels[0]} Position')
        ax.set_ylabel(f'{coord_labels[1]} Position')
        ax.set_title(f'Cell Volume Fractions ({axis.upper()}={slice_pos})')
        
        # Customize appearance for microscopy style
        ax.set_facecolor(background_color)  # Set axes background
        fig.patch.set_facecolor(background_color)  # Set figure background
        
        # Add legend with white text for visibility if background is dark
        if background_color.lower() in ['black', '#000000', '#000']:
            ax.legend(loc='upper right', facecolor='black', edgecolor='none', labelcolor='white')
        else:
            ax.legend(loc='upper right')
        
        plt.show()

def main():
    # Uncomment the following lines and replace with your simulation file path
    saved_sim_path = '/Users/rileymcnamara/CODE/2025/SCIE3212/data/project_model_test_sim_data.npz'
    plotter = VolumeFractionPlotter(simulation_data=saved_sim_path)
    
    # Example 1: Different density factors and thresholds for each cell type
    plotter.plot_volume_fractions(
        step=-1, 
        density_factors={'Healthy': 0.1, 'Diseased': 0.1, 'Necrotic': 0.4}, 
        thresholds={'Healthy': 0.05, 'Diseased': 0.1, 'Necrotic': 0.1},
        zoom_factor=0.3,
        point_size_factors={'Healthy': 100, 'Diseased': 20, 'Necrotic': 20}
    )
    
    # Example: Plot 2D cross-section with the same styling as 3D plot
    #plotter.plot_volume_fractions_2d(
    #    step=-1,
    #    axis='z',
    #    slice_pos=25,
    #    density_factors={'Healthy': 0.2, 'Diseased': 0.2, 'Necrotic': 0.5},
    #    thresholds={'Healthy': 0.1, 'Diseased': 0.1, 'Necrotic': 0.01},
    #    point_size_factors={'Healthy': 40, 'Diseased': 40, 'Necrotic': 40}
    #)
    
    # Example 2: Same density factor and threshold for all cell types
    # plotter.plot_volume_fractions(step=-1, density_factors=0.05, thresholds=0.1, zoom_factor=0.5)
    
    #plotter.plot_isosurface(step=10, cell_type='total')
    #plotter.plot_total_volume_evolution()
    # plotter.plot_radius_evolution()

    # Plot middle z-slice of all cell types
    #plotter.plot_volume_fractions_2d(step=-1, axis='z', slice_pos=40)

    # Plot specific y-slice of just healthy and diseased cells
    #plotter.plot_volume_fraction_cross_section(
    #    step=10, 
    #    axis='y', 
    #    slice_pos=25, 
    #    cell_types=['Healthy', 'Diseased'],
    #    cmap='plasma'
    #)

if __name__ == "__main__":
    main()