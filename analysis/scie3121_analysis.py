import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import marching_cubes
from scipy.signal import savgol_filter
from matplotlib.animation import FuncAnimation
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from copy import deepcopy

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.SCIE3121_model import SCIE3121_MODEL
from src.models.initial_conditions import SphericalTumor
from src.utils.utils import SCIE3121_params, gradient, laplacian, divergence

class scie3121SimulationAnalyzer:
    """Analyzer for tumor growth simulation data from NPZ files."""
    
    def __init__(self, filepath):
        """
        Initialize with the path to an NPZ file containing simulation history.
        """
        self.filepath = filepath
        self.history = load_simulation_history(filepath)
        self.volume_data = compute_total_volumes(self.history)
        self.radius_data = compute_total_radius(self.history, threshold=0.05)
        self.metadata = self.history['Simulation Metadata']

    def get_simulation_metadata(self):
        """Return metadata about the simulation."""
        return self.metadata

    def plot_volumes(self, smooth_window=5, normalize=None):
        """Plot total volumes of cell types over time with optional smoothing and normalization.
        
        Args:
            smooth_window (int): Window size for Savitzky-Golay smoothing (odd number).
            normalize (str, optional): 'initial' to normalize by initial total volume,
                                       'max' to normalize by maximum total volume.
        """
        steps, healthy, diseased, necrotic, total = self.volume_data
        
        # Ensure smooth_window is odd
        #if smooth_window % 2 == 0:
        #    smooth_window += 1
        
        # Apply smoothing if window > 1
        #if smooth_window > 1:
        #    healthy = savgol_filter(healthy, smooth_window, 3)
        #    progenitor = savgol_filter(progenitor, smooth_window, 3)
        #    differentiated = savgol_filter(differentiated, smooth_window, 3)
        #    necrotic = savgol_filter(necrotic, smooth_window, 3)
        #    total = savgol_filter(total, smooth_window, 3)
        
        # Normalize volumes
        #if normalize == 'initial':
        #    initial_total = total[0]
        #    if initial_total != 0:
        #        healthy /= initial_total
        #        progenitor /= initial_total
        #        differentiated /= initial_total
        #        necrotic /= initial_total
        #        total /= initial_total
        #    else:
        #        print("Warning: Initial total volume is zero, skipping normalization.")
        #elif normalize == 'max':
        #    max_total = np.max(total)
        #    if max_total != 0:
        #        healthy /= max_total
        #        progenitor /= max_total
        #        differentiated /= max_total
        #        necrotic /= max_total
        #        total /= max_total
        #    else:
        #        print("Warning: Maximum total volume is zero, skipping normalization.")
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, diseased, label='Differentiated')
        plt.plot(steps, necrotic, label='Necrotic')
        plt.plot(steps, healthy, label='Healthy')
        plt.plot(steps, total, label='Total', linestyle='--', color='black', alpha=0.7)
        plt.xlabel(f'Time step % 10 ')
        plt.ylabel('Volume (summed cell fraction)' if not normalize else 'Normalized Volume')
        plt.title('Tumor Cell Volume Evolution')
        plt.legend()
        plt.show()

    def plot_radius(self, smooth_window=5):
        """
        Plot tumor radius over time with optional smoothing.
        
        Args:
            smooth_window (int): Window size for Savitzky-Golay smoothing (odd number).
        """
        radii = self.radius_data
        steps = self.history['step']
        
        if smooth_window % 2 == 0:
            smooth_window += 1
        
        if smooth_window > 1:
            radii = savgol_filter(radii, smooth_window, 3)
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, radii, label='Tumor Radius', color='magenta')
        plt.xlabel('Time Step')
        plt.ylabel('Radius (in grid units)')
        plt.title('Tumor Radius Evolution')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

    def plot_3d_at_step(self, step_index, levels=None, colors=None, mode='isosurface'):
        """
        Plot 3D visualization of tumor fields at a specific time step with multiple modes.
        
        Args:
            step_index (int): Index of the time step to visualize.
            levels (dict, optional): Isosurface levels for each cell type. Defaults to 0.1.
            colors (dict, optional): Colors for each cell type. Defaults to preset values.
            mode (str): Visualization mode ('isosurface', 'voxel', 'scatter'). Defaults to 'isosurface'.
        """
        if levels is None:
            levels = {'healthy': 0.1, 'diseased': 0.1, 'necrotic': 0.1}
        if colors is None:
            colors = {'healthy': 'green', 'diseased': 'red', 'necrotic': 'black'}
        
        fields = {
            'healthy': self.history['healthy cell volume fraction'][step_index],
            'diseased': self.history['diseased cell volume fraction'][step_index],
            'necrotic': self.history['necrotic cell volume fraction'][step_index]
        }
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        if mode == 'isosurface':
            for cell_type, field in fields.items():
                level = levels[cell_type]
                if np.max(field) > level:
                    verts, faces, _, _ = marching_cubes(field, level=level)
                    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], 
                                   color=colors[cell_type], alpha=0.3, label=cell_type)
        
        elif mode == 'voxel':
            for cell_type, field in fields.items():
                mask = field > levels[cell_type]
                ax.voxels(mask, facecolors=colors[cell_type], edgecolors='k', alpha=0.5, label=cell_type)
        
        elif mode == 'scatter':
            for cell_type, field in fields.items():
                x, y, z = np.where(field > levels[cell_type])
                ax.scatter(x, y, z, c=colors[cell_type], alpha=0.5, s=10, label=cell_type)
        
        else:
            raise ValueError("Mode must be 'isosurface', 'voxel', or 'scatter'")
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'3D Tumor Visualization at Step {step_index} ({mode})')
        ax.legend()
        plt.show()

    def plot_cross_section(self, step_index, cell_type='healthy', plane='XY', index=None, smooth_sigma=2.0, 
                        vmin=None, vmax=None, levels=50, cmap='viridis'):
        """
        Plot 2D cross-sections of a specified cell type with smoothed contours and customizable scaling.
        
        Args:
            step_index (int): Index of the time step to visualize.
            cell_type (str): The cell type to plot ('healthy', 'progenitor', 'differentiated', 'necrotic').
            plane (str): Plane to slice ('XY', 'XZ', 'YZ'). Defaults to 'XY'.
            index (int, optional): Index along the perpendicular axis. Defaults to midpoint.
            smooth_sigma (float): Standard deviation for Gaussian smoothing. 0 for no smoothing.
            vmin (float, optional): Minimum value for colormap scaling. Defaults to data minimum.
            vmax (float, optional): Maximum value for colormap scaling. Defaults to data maximum.
            levels (int): Number of contour levels. Defaults to 10.
            cmap (str): Colormap name. Defaults to 'viridis'.
        """
        if cell_type not in ['healthy', 'diseased', 'necrotic', 'total']:
            raise ValueError("Cell type must be 'healthy', 'diseased', 'necrotic', or 'total'")
        
        fields = {
            'healthy': self.history['healthy cell volume fraction'][step_index],
            'diseased': self.history['diseased cell volume fraction'][step_index],
            'necrotic': self.history['necrotic cell volume fraction'][step_index],
            'total': (self.history['healthy cell volume fraction'] + 
                      self.history['necrotic cell volume fraction'] +
                      self.history['diseased cell volume fraction'])[step_index]
        }
        
        shape = fields['healthy'].shape
        if index is None:
            if plane == 'XY':
                index = shape[2] // 2
            elif plane == 'XZ':
                index = shape[1] // 2
            elif plane == 'YZ':
                index = shape[0] // 2
            else:
                raise ValueError("Plane must be 'XY', 'XZ', or 'YZ'")
        
        field = fields[cell_type]
        
        # Select the slice based on the plane
        if plane == 'XY':
            slice_ = field[:, :, index]
            x, y = np.arange(shape[0]), np.arange(shape[1])
            xlabel, ylabel = 'X', 'Y'
        elif plane == 'XZ':
            slice_ = field[:, index, :]
            x, y = np.arange(shape[0]), np.arange(shape[2])
            xlabel, ylabel = 'X', 'Z'
        elif plane == 'YZ':
            slice_ = field[index, :, :]
            x, y = np.arange(shape[1]), np.arange(shape[2])
            xlabel, ylabel = 'Y', 'Z'
        
        # Smooth the slice with Gaussian filter if sigma > 0
        if smooth_sigma > 0:
            slice_ = gaussian_filter(slice_, sigma=smooth_sigma)
        
        # Create finer grid for interpolation
        x_fine, y_fine = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
        X, Y = np.meshgrid(x, y)
        X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
        
        # Interpolate data onto finer grid
        slice_fine = griddata((X.flatten(), Y.flatten()), slice_.flatten(), 
                            (X_fine, Y_fine), method='cubic')
        
        # Determine colormap bounds dynamically if not provided
        if vmin is None:
            vmin = np.nanmin(slice_fine)  # Use min of interpolated data
        if vmax is None:
            vmax = np.nanmax(slice_fine)  # Use max of interpolated data
        
        # Ensure vmin and vmax are reasonable (avoid identical values)
        if vmin == vmax:
            vmin = max(0, vmin - 0.01)  # Small offset to avoid flat colormap
            vmax = vmin + 0.01
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(6, 5))
        contour = ax.contourf(X_fine, Y_fine, slice_fine, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(contour, ax=ax, label=f'{cell_type.capitalize()} Volume Fraction')
        
        # Set labels and titles
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{cell_type.capitalize()} Cross-Section ({plane} at index {index})')
        fig.suptitle(f'Step {step_index}', y=1.05)
        
        plt.show()

    def animate_cross_section(self, plane='XY', index=None, interval=200, save_as=None):
        """
        Animate 2D cross-sections of tumor fields over time.
        
        Args:
            plane (str): Plane to slice ('XY', 'XZ', 'YZ'). Defaults to 'XY'.
            index (int, optional): Index along the perpendicular axis. Defaults to midpoint.
            interval (int): Delay between frames in milliseconds. Defaults to 200.
            save_as (str, optional): File path to save animation (e.g., 'output.mp4').
        """
        # Get the shape from the first step to determine default index
        shape = self.history['healthy cell volume fraction'][0].shape
        if index is None:
            if plane == 'XY':
                index = shape[2] // 2
            elif plane == 'XZ':
                index = shape[1] // 2
            elif plane == 'YZ':
                index = shape[0] // 2
            else:
                raise ValueError("Plane must be 'XY', 'XZ', or 'YZ'")
        
        # Set up the figure and axes
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Set up the grid coordinates based on the plane
        if plane == 'XY':
            x, y = np.arange(shape[0]), np.arange(shape[1])
            xlabel, ylabel = 'X', 'Y'
        elif plane == 'XZ':
            x, y = np.arange(shape[0]), np.arange(shape[2])
            xlabel, ylabel = 'X', 'Z'
        elif plane == 'YZ':
            x, y = np.arange(shape[1]), np.arange(shape[2])
            xlabel, ylabel = 'Y', 'Z'
        
        X, Y = np.meshgrid(x, y)
        
        # Create a finer grid for smoother visualization
        x_fine, y_fine = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
        X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
        
        # Set up the contour plots for each cell type
        cell_types = ['healthy', 'diseased', 'necrotic', 'total']
        contour_plots = []
        
        for i, cell_type in enumerate(cell_types):
            ax = axes[i]
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f'{cell_type.capitalize()}')
            
            # Initialize with empty data
            contour = ax.contourf(X_fine, Y_fine, np.zeros((100, 100)), levels=50, cmap='viridis', vmin=0, vmax=1)
            fig.colorbar(contour, ax=ax, label=f'{cell_type.capitalize()} Volume Fraction')
            contour_plots.append(contour)
        
        def update(frame):
            # Get data for the current frame
            fields = {
                'healthy': self.history['healthy cell volume fraction'][frame],
                'diseased': self.history['diseased cell volume fraction'][frame],
                'necrotic': self.history['necrotic cell volume fraction'][frame],
            }
            # Calculate total field
            fields['total'] = fields['healthy'] + fields['diseased'] + fields['necrotic']
            
            for i, cell_type in enumerate(cell_types):
                ax = axes[i]
                
                # Extract the slice based on the plane
                if plane == 'XY':
                    slice_ = fields[cell_type][:, :, index]
                elif plane == 'XZ':
                    slice_ = fields[cell_type][:, index, :]
                elif plane == 'YZ':
                    slice_ = fields[cell_type][index, :, :]
                
                # Apply Gaussian smoothing
                slice_ = gaussian_filter(slice_, sigma=1.0)
                
                # Interpolate to finer grid
                slice_fine = griddata((X.flatten(), Y.flatten()), slice_.flatten(), 
                                    (X_fine, Y_fine), method='cubic')
                
                # Clear the previous contour plot
                ax.clear()
                
                # Create a new contour plot
                contour_plots[i] = ax.contourf(X_fine, Y_fine, slice_fine, levels=50, 
                                            cmap='viridis', vmin=0, vmax=1)
                
                # Reset labels and title
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_title(f'{cell_type.capitalize()}')
            
            fig.suptitle(f'Step {self.history["step"][frame]}')
            return contour_plots
        
        anim = FuncAnimation(fig, update, frames=len(self.history['step']), interval=interval, blit=False)
        
        if save_as:
            anim.save(save_as, writer='ffmpeg')  # Requires ffmpeg installed
        
        plt.tight_layout()
        plt.show()
        
        return anim  # Return the animation object to prevent garbage collection

    def plot_growth_rate(self):
        """Plot the tumor growth rate over time."""
        steps = self.history['step']
        total_volume = self.volume_data[-1]  # Total volume is the last element
        growth_rate = np.gradient(total_volume, steps)  # Numerical derivative
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, growth_rate, label='Growth Rate', color='teal')
        plt.xlabel('Time Step')
        plt.ylabel('Growth Rate (volume per step)')
        plt.title('Tumor Growth Rate')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()


    def plot_combined_cross_section(self, step_index, plane='XY', index=None, smooth_sigma=2.0, 
                                  levels=50, cmaps=None):
        """
        Plot 2D cross-sections of all fields (including nutrient) in a single figure.
        
        Args:
            step_index (int): Index of the time step to visualize.
            plane (str): Plane to slice ('XY', 'XZ', 'YZ'). Defaults to 'XY'.
            index (int, optional): Index along the perpendicular axis. Defaults to midpoint.
            smooth_sigma (float): Standard deviation for Gaussian smoothing. 0 for no smoothing.
            levels (int): Number of contour levels. Defaults to 50.
            cmaps (dict, optional): Custom colormaps for each field.
        """
        if 'nutrient concentration' not in self.history:
            raise ValueError("Nutrient concentration data not found in the simulation history.")
        
        if cmaps is None:
            cmaps = {
                'healthy': 'Greens',
                'diseased': 'Reds',
                'necrotic': 'Greys',
                'nutrient': 'Blues'
            }
        
        fields = {
            'healthy': self.history['healthy cell volume fraction'][step_index],
            'diseased': self.history['diseased cell volume fraction'][step_index],
            'necrotic': self.history['necrotic cell volume fraction'][step_index],
            'nutrient': self.history['nutrient concentration'][step_index]
        }
        
        shape = fields['healthy'].shape
        if index is None:
            if plane == 'XY':
                index = shape[2] // 2
            elif plane == 'XZ':
                index = shape[1] // 2
            elif plane == 'YZ':
                index = shape[0] // 2
            else:
                raise ValueError("Plane must be 'XY', 'XZ', or 'YZ'")
        
        # Set up the figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        # Set up the grid coordinates based on the plane
        if plane == 'XY':
            x, y = np.arange(shape[0]), np.arange(shape[1])
            xlabel, ylabel = 'X', 'Y'
        elif plane == 'XZ':
            x, y = np.arange(shape[0]), np.arange(shape[2])
            xlabel, ylabel = 'X', 'Z'
        elif plane == 'YZ':
            x, y = np.arange(shape[1]), np.arange(shape[2])
            xlabel, ylabel = 'Y', 'Z'
        
        X, Y = np.meshgrid(x, y)
        
        # Create finer grid for interpolation
        x_fine, y_fine = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
        X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
        
        # Plot each field
        for i, (field_name, field_data) in enumerate(fields.items()):
            ax = axes[i]
            
            # Select the slice based on the plane
            if plane == 'XY':
                slice_ = field_data[:, :, index]
            elif plane == 'XZ':
                slice_ = field_data[:, index, :]
            elif plane == 'YZ':
                slice_ = field_data[index, :, :]
            
            # Smooth the slice with Gaussian filter if sigma > 0
            if smooth_sigma > 0:
                slice_ = gaussian_filter(slice_, sigma=smooth_sigma)
            
            # Interpolate data onto finer grid
            slice_fine = griddata((X.flatten(), Y.flatten()), slice_.flatten(), 
                                (X_fine, Y_fine), method='cubic')
            
            # Determine colormap bounds
            vmin = 0
            vmax = np.nanmax(slice_fine) if np.nanmax(slice_fine) > 0 else 1.0
            
            # Create the contour plot
            contour = ax.contourf(X_fine, Y_fine, slice_fine, levels=levels, 
                                cmap=cmaps[field_name], vmin=vmin, vmax=vmax)
            fig.colorbar(contour, ax=ax, label=f'{field_name.capitalize()} Concentration')
            
            # Set labels and title
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f'{field_name.capitalize()}')
        
        fig.suptitle(f'Cross-Sections at Step {step_index} ({plane} Plane, Index {index})', y=1.02)
        plt.tight_layout()
        plt.show()

    def plot_population_radii(self, threshold=0.05, smooth_window=5):
        """
        Plot the radius of each cell population over time.
        
        Args:
            threshold (float): Threshold value for determining the boundary of each population.
            smooth_window (int): Window size for Savitzky-Golay smoothing (odd number).
        """
        steps = self.history['step']
        healthy_radii, diseased_radii, necrotic_radii = compute_population_radii(self.history, threshold)
        
        # Ensure smooth_window is odd
        if smooth_window % 2 == 0:
            smooth_window += 1
        
        # Apply smoothing if window > 1 and enough data points
        if smooth_window > 1 and len(steps) > smooth_window:
            healthy_radii = savgol_filter(healthy_radii, smooth_window, 3)
            diseased_radii = savgol_filter(diseased_radii, smooth_window, 3)
            necrotic_radii = savgol_filter(necrotic_radii, smooth_window, 3)
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, healthy_radii, label='Healthy', color='green')
        plt.plot(steps, diseased_radii, label='Diseased', color='red')
        plt.plot(steps, necrotic_radii, label='Necrotic', color='black')
        
        # Also plot the total radius for comparison
        plt.plot(steps, self.radius_data, label='Total Tumor', color='blue', linestyle='--')
        
        plt.xlabel('Time Step')
        plt.ylabel('Radius (in grid units)')
        plt.title(f'Cell Population Radii Evolution (threshold={threshold})')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

# Helper functions 
def load_simulation_history(npz_filename):
    """Load simulation history from an NPZ file."""
    data = np.load(npz_filename, allow_pickle=True)
    history = {key: data[key].item() if key == 'history' else data[key] for key in data}
    return history

def compute_total_volumes(history):
    """Compute total volumes for each cell type over time."""
    steps = history['step']
    healthy_volumes = [np.sum(phi) for phi in history['healthy cell volume fraction']]
    diseased_volumes = [np.sum(phi) for phi in history['diseased cell volume fraction']]  # Changed to match typo
    necrotic_volumes = [np.sum(phi) for phi in history['necrotic cell volume fraction']]
    total_volumes = [h + d + n for h, d, n in zip(healthy_volumes, diseased_volumes, necrotic_volumes)]  # Adjusted to 3 terms
    return steps, healthy_volumes, diseased_volumes, necrotic_volumes, total_volumes

def create_distance_grid_from_field(field):
    """Create a distance grid from the field center."""
    shape = field.shape
    center = np.array([s // 2 for s in shape])
    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    return np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)

def compute_total_radius(history, threshold=0.05):
    """Compute tumor radius over time based on a threshold."""
    radii = []
    first_phi = history['healthy cell volume fraction'][0]
    distance_grid = create_distance_grid_from_field(first_phi)
    
    for phi_H,  phi_D, phi_N in zip(
            history['healthy cell volume fraction'],
            history['diseased cell volume fraction'],
            history['necrotic cell volume fraction']):
        total_field = phi_H + phi_D + phi_N
        tumor_mask = total_field >= (threshold * np.max(total_field))
        radius = np.max(distance_grid[tumor_mask]) if np.any(tumor_mask) else 0
        radii.append(radius)
    return radii

def compute_population_radii(history, threshold=0.05):
    """Compute radius of each cell population over time based on a threshold."""
    healthy_radii = []
    diseased_radii = []
    necrotic_radii = []
    
    first_phi = history['healthy cell volume fraction'][0]
    distance_grid = create_distance_grid_from_field(first_phi)
    
    for phi_H, phi_D, phi_N in zip(
            history['healthy cell volume fraction'],
            history['diseased cell volume fraction'],
            history['necrotic cell volume fraction']):
        
        # Calculate radius for each population
        healthy_mask = phi_H >= (threshold * np.max(phi_H)) if np.max(phi_H) > 0 else np.zeros_like(phi_H, dtype=bool)
        diseased_mask = phi_D >= (threshold * np.max(phi_D)) if np.max(phi_D) > 0 else np.zeros_like(phi_D, dtype=bool)
        necrotic_mask = phi_N >= (threshold * np.max(phi_N)) if np.max(phi_N) > 0 else np.zeros_like(phi_N, dtype=bool)
        
        healthy_radius = np.max(distance_grid[healthy_mask]) if np.any(healthy_mask) else 0
        diseased_radius = np.max(distance_grid[diseased_mask]) if np.any(diseased_mask) else 0
        necrotic_radius = np.max(distance_grid[necrotic_mask]) if np.any(necrotic_mask) else 0
        
        healthy_radii.append(healthy_radius)
        diseased_radii.append(diseased_radius)
        necrotic_radii.append(necrotic_radius)
        
    return healthy_radii, diseased_radii, necrotic_radii

def main():

    model = SCIE3121_MODEL(
        grid_shape=(50, 50, 50),
        dx=1,
        dt=0.1,
        params=SCIE3121_params,
        initial_conditions=SphericalTumor(grid_shape=(30, 30, 30), radius=7, nutrient_value=1.0),
    )
    analyzer = scie3121SimulationAnalyzer(filepath='data/project_model_test_sim_data.npz') 
    #analyzer.visualize_velocity_field(model, step=0)
    #analyzer.animate_velocity_field(model)
    analyzer.animate_cross_section(plane='XY', interval=200)
    #alyzer.plot_combined_cross_section(step_index=0, plane='XY', index=None, smooth_sigma=2.0, levels=50, cmaps=None) 
    #analyzer.plot_nutrient_field(step_index=14, plane='XY', index=None, smooth_sigma=2.0, vmin=None, vmax=None, levels=50, cmap='viridis')
    #analyzer.animate_nutrient_field(plane='XY', index=None, interval=200, save_as=None)
    analyzer.plot_population_radii(threshold=0.05, smooth_window=5)
   
    
if __name__ == "__main__":
    main()