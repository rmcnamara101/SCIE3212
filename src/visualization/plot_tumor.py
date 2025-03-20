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
from matplotlib import animation
from src.models.SCIE3121_model import SCIE3121_MODEL
from src.models.initial_conditions import SphericalTumor
from src.utils.utils import SCIE3121_params
import ipywidgets as widgets
from IPython.display import display, clear_output

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
            self.dx = self.simulation_history['Simulation Metadata']['dx']
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

    def plot_inking_3d(self, step: int, density_factors=None, thresholds=None, zoom_factor=0.8, point_size_factors=None):
        """
        Plot a 3D scatter plot of volume fractions styled as fluorescent microscopy imaging.
        
        Args:
            step: Time step to visualize
            density_factors: Dict with density factors for each cell type
            thresholds: Dict with threshold values for each cell type
            zoom_factor: Factor to control zoom level (0-1, lower means more zoomed in)
            point_size_factors: Dict with size multipliers for each cell type
        """
        # Create figure and 3D axes with dark background
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(12, 12), dpi=150)
        ax = fig.add_subplot(111, projection='3d')
        
        # Define cell types with realistic fluorescent protein colors
        cell_types = {
            'Necrotic': ('necrotic cell volume fraction', '#FF4444'),    # Red fluorescent protein
            'Diseased': ('diseased cell volume fraction', '#44FF44'),    # Green fluorescent protein
            'Healthy': ('healthy cell volume fraction', '#4444FF')       # Blue fluorescent protein
        }
        
        # Set default parameters if not provided
        if density_factors is None:
            density_factors = {'Healthy': 0.4, 'Diseased': 0.4, 'Necrotic': 0.4}
        elif isinstance(density_factors, (int, float)):
            density_factors = {name: density_factors for name in cell_types.keys()}
        
        if thresholds is None:
            thresholds = {'Healthy': 0.05, 'Diseased': 0.05, 'Necrotic': 0.05}
        elif isinstance(thresholds, (int, float)):
            thresholds = {name: thresholds for name in cell_types.keys()}
        
        if point_size_factors is None:
            point_size_factors = {'Healthy': 50, 'Diseased': 50, 'Necrotic': 50}
        elif isinstance(point_size_factors, (int, float)):
            point_size_factors = {name: point_size_factors for name in cell_types.keys()}
        
        # Track center of mass for zooming
        total_points = 0
        center_x, center_y, center_z = 0, 0, 0
        
        # Plot each cell type with multiple layers for realistic fluorescence
        for name, (key, base_color) in cell_types.items():
            if key in self.simulation_history:
                field = self.simulation_history[key][step] if len(self.simulation_history[key].shape) > 3 else self.simulation_history[key]
                
                # Get coordinates above threshold
                x, y, z = np.where(field > thresholds[name])
                fractions = field[field > thresholds[name]]
                
                if len(x) > 0:
                    # Apply density reduction
                    if density_factors[name] < 1.0:
                        num_points = max(1, int(len(x) * density_factors[name]))
                        indices = np.random.choice(len(x), num_points, replace=False)
                        x, y, z = x[indices], y[indices], z[indices]
                        fractions = fractions[indices]
                    
                    # Update center of mass
                    center_x += np.sum(x)
                    center_y += np.sum(y)
                    center_z += np.sum(z)
                    total_points += len(x)
                    
                    # Convert coordinates to physical units
                    x_pos = x * self.dx
                    y_pos = y * self.dx
                    z_pos = z * self.dx
                    
                    # Create layered glow effect
                    size_base = point_size_factors[name]
                    
                    # Layer 1: Core (brightest, smallest)
                    ax.scatter(x_pos, y_pos, z_pos,
                              c=base_color,
                              s=size_base * fractions,
                              alpha=0.9,
                              edgecolors='none')
                    
                    # Layer 2: Inner glow
                    ax.scatter(x_pos, y_pos, z_pos,
                              c=base_color,
                              s=size_base * fractions * 2,
                              alpha=0.3,
                              edgecolors='none')
                    
                    # Layer 3: Outer glow (most diffuse)
                    ax.scatter(x_pos, y_pos, z_pos,
                              c=base_color,
                              s=size_base * fractions * 4,
                              alpha=0.1,
                              edgecolors='none')
                    
                    # Layer 4: Ambient glow (very diffuse)
                    ax.scatter(x_pos, y_pos, z_pos,
                              c=base_color,
                              s=size_base * fractions * 8,
                              alpha=0.05,
                              edgecolors='none')
        
        # Set view limits based on center of mass
        if total_points > 0:
            center_x = (center_x / total_points) * self.dx
            center_y = (center_y / total_points) * self.dx
            center_z = (center_z / total_points) * self.dx
        else:
            center_x = self.grid_shape[0] * self.dx / 2
            center_y = self.grid_shape[1] * self.dx / 2
            center_z = self.grid_shape[2] * self.dx / 2
        
        # Calculate and set view boundaries
        grid_size = max(self.grid_shape) * self.dx
        half_width = grid_size * zoom_factor / 2
        
        ax.set_xlim([max(0, center_x - half_width), min(self.grid_shape[0] * self.dx, center_x + half_width)])
        ax.set_ylim([max(0, center_y - half_width), min(self.grid_shape[1] * self.dx, center_y + half_width)])
        ax.set_zlim([max(0, center_z - half_width), min(self.grid_shape[2] * self.dx, center_z + half_width)])
        
        # Style the plot for fluorescence microscopy look
        ax.set_facecolor('#000000')  # Pure black background
        fig.patch.set_facecolor('#000000')
        
        # Remove all axes elements for clean look
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        # Set optimal viewing angle for 3D visualization
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        plt.show()
        
        # Reset style for other plots
        plt.style.use('default')

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

    def plot_inking_2d(self, step: int, axis='z', slice_pos=None, density_factors=None, 
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

    def plot_combined_3d(self, step: int, nutrient_threshold: float = 0.1, **kwargs):
        """Plot 3D scatter of cell types with a nutrient isosurface overlay."""
        # First, plot cell types using existing function
        self.plot_inking_3d(step, **kwargs)
        
        # Add nutrient isosurface if available
        if 'nutrient' in self.simulation_history:
            nutrient_field = self.simulation_history['nutrient'][step]
            verts, faces, _, _ = marching_cubes(nutrient_field, nutrient_threshold)
            verts *= self.dx  # Scale to physical coordinates
            
            ax = plt.gca()
            mesh = Poly3DCollection(verts[faces], alpha=0.2, facecolor='green', edgecolor='none')
            ax.add_collection3d(mesh)
            
            # Ensure axes limits encompass both scatter and isosurface
            ax.set_xlim(0, self.grid_shape[0] * self.dx)
            ax.set_ylim(0, self.grid_shape[1] * self.dx)
            ax.set_zlim(0, self.grid_shape[2] * self.dx)
            plt.title(f"Step {step}: Cell Types with Nutrient Isosurface")
        else:
            print("Nutrient field not found in simulation history")
        plt.show()

    def animate_cross_sections(self, axis: str = 'z', slice_pos: int = None, cell_types: list = None, interval: int = 200, save_path: str = None):
        """Animate 2D cross-sections of cell volume fractions over time."""
        if slice_pos is None:
            slice_pos = self.grid_shape[{'x': 0, 'y': 1, 'z': 2}[axis]] // 2
        
        fig, ax = plt.subplots()
        
        def update(frame):
            ax.clear()
            self.plot_volume_fraction_cross_section(frame, axis, slice_pos, cell_types, show_colorbar=False)
            ax.set_title(f"Step {frame}")
        
        ani = animation.FuncAnimation(fig, update, frames=len(self.simulation_history['step']), interval=interval)
        
        if save_path:
            ani.save(save_path, writer='ffmpeg')
        else:
            plt.show()


    def plot_radial_distribution(self, step: int, bins: int = 20):
        """Plot average cell volume fractions vs. distance from organoid center."""
        # Calculate center of mass using total cell volume fraction
        total_field = self.simulation_history['healthy cell volume fraction'][step] + self.simulation_history['diseased cell volume fraction'][step] + self.simulation_history['necrotic cell volume fraction'][step]
        indices = np.indices(total_field.shape)
        center = np.array([np.sum(indices[i] * total_field) / np.sum(total_field) for i in range(3)]) * self.dx
        
        # Compute distances from center
        x, y, z = np.meshgrid(np.arange(self.grid_shape[0]), np.arange(self.grid_shape[1]), np.arange(self.grid_shape[2]), indexing='ij')
        coords = np.stack([x, y, z], axis=-1) * self.dx
        distances = np.linalg.norm(coords - center, axis=-1)
        
        # Define radial bins
        bin_edges = np.linspace(0, np.max(distances), bins + 1)
        
        # Plot for each cell type
        cell_types = ['healthy', 'diseased', 'necrotic']
        for ct in cell_types:
            key = f'{ct} cell volume fraction'
            if key in self.simulation_history:
                field = self.simulation_history[key][step]
                bin_indices = np.digitize(distances.ravel(), bin_edges)
                averages = [np.mean(field.ravel()[bin_indices == i]) for i in range(1, bins + 1)]
                plt.plot(bin_edges[:-1], averages, label=ct.capitalize())
        
        plt.xlabel('Distance from Center (units)')
        plt.ylabel('Average Volume Fraction')
        plt.legend()
        plt.title(f"Step {step}: Radial Distribution")
        plt.show()

    def plot_nutrient_vs_cell(self, step: int, cell_type: str = 'healthy', bins: int = 50):
        """Plot 2D histogram of nutrient level vs. cell volume fraction."""
        if 'nutrient' in self.simulation_history and f'{cell_type} cell volume fraction' in self.simulation_history:
            nutrient = self.simulation_history['nutrient'][step].ravel()
            cell_vf = self.simulation_history[f'{cell_type} cell volume fraction'][step].ravel()
            
            # Create figure with larger size
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Set colors and style
            background_color = '#1f2937'  # Dark blue-gray
            text_color = '#e5e7eb'  # Light gray
            spine_color = '#4b5563'  # Medium gray
            
            # Plot 2D histogram with improved styling
            hist = plt.hist2d(nutrient, cell_vf, 
                             bins=bins,
                             cmap='plasma',  # Using plasma colormap for better contrast on dark background
                             norm=plt.matplotlib.colors.LogNorm(),
                             density=True)
            
            # Customize the plot style
            ax.set_facecolor(background_color)
            fig.patch.set_facecolor(background_color)
            
            # Add colorbar with custom styling
            cbar = plt.colorbar(label='Normalized Density (log scale)')
            cbar.ax.yaxis.label.set_color(text_color)
            cbar.ax.tick_params(colors=text_color)
            
            # Improve labels and title formatting
            plt.xlabel('Nutrient Concentration', fontsize=12, color=text_color)
            plt.ylabel(f'{cell_type.capitalize()} Cell Volume Fraction', fontsize=12, color=text_color)
            plt.title(f"Step {step}: Nutrient vs. {cell_type.capitalize()} Cells", 
                     fontsize=14, color=text_color, pad=20)
            
            # Customize ticks
            ax.tick_params(colors=text_color, which='both')
            
            # Remove grid and style spines
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_color(spine_color)
                spine.set_linewidth(1.5)
            
            # Print statistics for debugging
            print(f"Number of data points: {len(nutrient)}")
            print(f"Nutrient range: [{nutrient.min():.3f}, {nutrient.max():.3f}]")
            print(f"Cell volume fraction range: [{cell_vf.min():.3f}, {cell_vf.max():.3f}]")
            
            plt.tight_layout()
            plt.show()
        else:
            print("Required data not found in simulation history")

    def plot_average_vf_over_time(self, threshold: float = 0.1):
        """Plot average volume fractions within organoid over time."""
        steps = self.simulation_history['step']
        cell_types = ['healthy', 'diseased', 'necrotic']
        averages = {ct: [] for ct in cell_types}
        
        for step in range(len(steps)):
            total_field = self.simulation_history['healthy cell volume fraction'][step] + self.simulation_history['diseased cell volume fraction'][step] + self.simulation_history['necrotic cell volume fraction'][step]
            mask = total_field > threshold
            for ct in cell_types:
                key = f'{ct} cell volume fraction'
                if key in self.simulation_history:
                    field = self.simulation_history[key][step]
                    avg = np.mean(field[mask]) if np.any(mask) else 0
                    averages[ct].append(avg)
        
        for ct in cell_types:
            plt.plot(steps, averages[ct], label=ct.capitalize())
        
        plt.xlabel('Time Step')
        plt.ylabel('Average Volume Fraction')
        plt.legend()
        plt.title('Average Volume Fractions Over Time')
        plt.show()

    def interactive_plot_3d(self):
        """Create an interactive 3D plot with adjustable parameters."""
        
        # Override the default figure size in plot_inking_3d
        def modified_plot_3d(**kwargs):
            plt.figure(figsize=(8, 8), dpi=100)  # Smaller figure size
            self.plot_inking_3d(**kwargs)
        
        # Create widgets for all adjustable parameters
        step_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=len(self.simulation_history['step'])-1,
            description='Time Step:',
            continuous_update=False,
            layout=widgets.Layout(width='300px')
        )
        
        zoom_slider = widgets.FloatSlider(
            value=0.8,
            min=0.1,
            max=1.0,
            step=0.1,
            description='Zoom:',
            continuous_update=False,
            layout=widgets.Layout(width='300px')
        )
        
        # Density factors for each cell type
        density_healthy = widgets.FloatSlider(
            value=0.4,
            min=0.1,
            max=1.0,
            step=0.1,
            description='Healthy Density:',
            continuous_update=False,
            layout=widgets.Layout(width='300px')
        )
        
        density_diseased = widgets.FloatSlider(
            value=0.4,
            min=0.1,
            max=1.0,
            step=0.1,
            description='Diseased Density:',
            continuous_update=False,
            layout=widgets.Layout(width='300px')
        )
        
        density_necrotic = widgets.FloatSlider(
            value=0.4,
            min=0.1,
            max=1.0,
            step=0.1,
            description='Necrotic Density:',
            continuous_update=False,
            layout=widgets.Layout(width='300px')
        )
        
        # Thresholds for each cell type
        threshold_healthy = widgets.FloatSlider(
            value=0.05,
            min=0.01,
            max=0.5,
            step=0.01,
            description='Healthy Threshold:',
            continuous_update=False,
            layout=widgets.Layout(width='300px')
        )
        
        threshold_diseased = widgets.FloatSlider(
            value=0.05,
            min=0.01,
            max=0.5,
            step=0.01,
            description='Diseased Threshold:',
            continuous_update=False,
            layout=widgets.Layout(width='300px')
        )
        
        threshold_necrotic = widgets.FloatSlider(
            value=0.05,
            min=0.01,
            max=0.5,
            step=0.01,
            description='Necrotic Threshold:',
            continuous_update=False,
            layout=widgets.Layout(width='300px')
        )
        
        # Point size factors for each cell type
        size_healthy = widgets.FloatSlider(
            value=50,
            min=10,
            max=200,
            step=10,
            description='Healthy Size:',
            continuous_update=False,
            layout=widgets.Layout(width='300px')
        )
        
        size_diseased = widgets.FloatSlider(
            value=50,
            min=10,
            max=200,
            step=10,
            description='Diseased Size:',
            continuous_update=False,
            layout=widgets.Layout(width='300px')
        )
        
        size_necrotic = widgets.FloatSlider(
            value=50,
            min=10,
            max=200,
            step=10,
            description='Necrotic Size:',
            continuous_update=False,
            layout=widgets.Layout(width='300px')
        )
        
        # View angle controls
        elev_slider = widgets.FloatSlider(
            value=20,
            min=0,
            max=90,
            step=5,
            description='Elevation:',
            continuous_update=False,
            layout=widgets.Layout(width='300px')
        )
        
        azim_slider = widgets.FloatSlider(
            value=45,
            min=0,
            max=360,
            step=5,
            description='Azimuth:',
            continuous_update=False,
            layout=widgets.Layout(width='300px')
        )
        
        # Toggle buttons for cell types
        show_healthy = widgets.ToggleButton(
            value=True,
            description='Show Healthy',
            layout=widgets.Layout(width='100px')
        )
        show_diseased = widgets.ToggleButton(
            value=True,
            description='Show Diseased',
            layout=widgets.Layout(width='100px')
        )
        show_necrotic = widgets.ToggleButton(
            value=True,
            description='Show Necrotic',
            layout=widgets.Layout(width='100px')
        )
        
        # Create output widget for the plot
        plot_output = widgets.Output()
        
        def update_plot(change):
            with plot_output:
                plot_output.clear_output(wait=True)
                
                # Collect parameters
                density_factors = {
                    'Healthy': density_healthy.value if show_healthy.value else 0,
                    'Diseased': density_diseased.value if show_diseased.value else 0,
                    'Necrotic': density_necrotic.value if show_necrotic.value else 0
                }
                
                thresholds = {
                    'Healthy': threshold_healthy.value,
                    'Diseased': threshold_diseased.value,
                    'Necrotic': threshold_necrotic.value
                }
                
                point_size_factors = {
                    'Healthy': size_healthy.value,
                    'Diseased': size_diseased.value,
                    'Necrotic': size_necrotic.value
                }
                
                # Plot with current parameters
                modified_plot_3d(
                    step=step_slider.value,
                    density_factors=density_factors,
                    thresholds=thresholds,
                    zoom_factor=zoom_slider.value,
                    point_size_factors=point_size_factors
                )
        
        # Create control panel layout with better organization
        controls = widgets.VBox([
            widgets.HBox([step_slider, zoom_slider], layout=widgets.Layout(justify_content='space-around')),
            widgets.HBox([show_healthy, show_diseased, show_necrotic], layout=widgets.Layout(justify_content='space-around')),
            widgets.HTML(value="<h4>Density Controls</h4>"),
            widgets.HBox([density_healthy, density_diseased, density_necrotic], layout=widgets.Layout(justify_content='space-around')),
            widgets.HTML(value="<h4>Threshold Controls</h4>"),
            widgets.HBox([threshold_healthy, threshold_diseased, threshold_necrotic], layout=widgets.Layout(justify_content='space-around')),
            widgets.HTML(value="<h4>Size Controls</h4>"),
            widgets.HBox([size_healthy, size_diseased, size_necrotic], layout=widgets.Layout(justify_content='space-around')),
            widgets.HTML(value="<h4>View Controls</h4>"),
            widgets.HBox([elev_slider, azim_slider], layout=widgets.Layout(justify_content='space-around'))
        ], layout=widgets.Layout(width='100%', padding='10px'))
        
        # Create the main layout
        main_layout = widgets.VBox([
            controls,
            plot_output
        ], layout=widgets.Layout(width='100%', align_items='center'))
        
        # Connect the update function to all controls
        for w in [step_slider, zoom_slider, 
                  density_healthy, density_diseased, density_necrotic,
                  threshold_healthy, threshold_diseased, threshold_necrotic,
                  size_healthy, size_diseased, size_necrotic,
                  elev_slider, azim_slider,
                  show_healthy, show_diseased, show_necrotic]:
            w.observe(update_plot, 'value')
        
        # Display the main layout
        display(main_layout)
        
        # Initial plot
        update_plot(None)

def main():
    # Uncomment the following lines and replace with your simulation file path
    saved_sim_path = '/Users/rileymcnamara/CODE/2025/SCIE3212/data/base_sim_data.npz'
    plotter = VolumeFractionPlotter(simulation_data=saved_sim_path)
    
    # Example 1: Different density factors and thresholds for each cell type
    #plotter.plot_inking_3d(
    #    step=0, 
    #    density_factors={'Healthy': 0.2, 'Diseased': 0.2, 'Necrotic': 0.2}, 
    #    thresholds={'Healthy': 0.1, 'Diseased': 0.1, 'Necrotic': 0.4},
    #    zoom_factor=0.3,
    #    point_size_factors={'Healthy': 20, 'Diseased': 20, 'Necrotic': 20}
    #)
    
    # Example: Plot 2D cross-section with the same styling as 3D plot
    #plotter.plot_inking_2d(
    #    step=-1,
    #    axis='z',
    #    slice_pos=plotter.grid_shape[2] // 2,
    #    density_factors={'Healthy': 0.2, 'Diseased': 0.2, 'Necrotic': 0.5},
    #    thresholds={'Healthy': 0.1, 'Diseased': 0.1, 'Necrotic': 0.5},
    #    point_size_factors={'Healthy': 30, 'Diseased': 30, 'Necrotic': 20}
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

    #plotter.plot_combined_3d(step=-1, density_factors={'Healthy': 0.05, 'Diseased': 0.05, 'Necrotic': 0.05}, thresholds={'Healthy': 0.1, 'Diseased': 0.1, 'Necrotic': 0.5}, point_size_factors={'Healthy': 30, 'Diseased': 30, 'Necrotic': 20})
    #plotter.animate_cross_sections(axis='z', cell_types=['Healthy', 'Diseased'], interval=200, save_path='animation.mp4')
    #plotter.plot_radial_distribution(step=-1, bins=100)
    #plotter.plot_nutrient_vs_cell(step=-1, cell_type='necrotic', bins=100)
    #plotter.plot_average_vf_over_time(threshold=0.1)

    plotter.interactive_plot_3d()

if __name__ == "__main__":
    main()