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
from src.utils.utils import SCIE3121_params
from src.models.cell_dynamics import compute_internal_pressure_scie3121_model, compute_solid_velocity_scie3121_model

class scie3121SimulationAnalyzer:
    """Analyzer for tumor growth simulation data from NPZ files."""
    
    def __init__(self, filepath):
        """
        Initialize with the path to an NPZ file containing simulation history.
        """
        self.filepath = filepath
        self.history = load_simulation_history(filepath)
        print(self.history.keys())
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

    def plot_3d_multiple_steps(self, step_indices, levels=None, colors=None):
        """
        Plot 3D isosurfaces for multiple time steps in one figure.
        
        Args:
            step_indices (list): List of time step indices to visualize.
            levels (dict, optional): Isosurface levels for each cell type.
            colors (dict, optional): Colors for each cell type.
        """
        if levels is None:
            levels = {'healthy': 0.1, 'diseased': 0.1, 'necrotic': 0.1}
        if colors is None:
            colors = {'healthy': 'green',  'differentiated': 'red', 'necrotic': 'black'}
        
        fig = plt.figure(figsize=(12, 10))
        
        for i, step_index in enumerate(step_indices, 1):
            ax = fig.add_subplot(1, len(step_indices), i, projection='3d')
            fields = {
                'healthy': self.history['healthy cell volume fraction'][step_index],
                'diseased': self.history['diseased cell volume fraction'][step_index],
                'necrotic': self.history['necrotic cell volume fraction'][step_index]
            }
            
            for cell_type, field in fields.items():
                level = levels[cell_type]
                if np.max(field) > level:
                    verts, faces, _, _ = marching_cubes(field, level=level)
                    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], 
                                   color=colors[cell_type], alpha=0.3, label=cell_type)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Step {step_index}')
        
        plt.suptitle('3D Tumor Evolution Across Multiple Steps')
        plt.tight_layout()
        plt.legend()
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

    def test_tumor_stability(self, steps=10):
        """Test if a spherical tumor maintains its shape without external forces."""
        # Create a small grid for faster testing
        grid_shape = (30, 30, 30)
        dx = 0.2
        dt = 0.001  # Very small time step for stability
        
        # Create initial conditions with a clear spherical tumor
        initial_conditions = SphericalTumor(grid_shape, radius=7, nutrient_value=0.1)
        
        # Create a modified parameter set with minimal diffusion
        test_params = deepcopy(SCIE3121_params)
        # Set diffusion coefficients to very small values
        # test_params['D_H'] = 0.001  # Adjust parameter names as needed
        
        # Initialize model
        model = SCIE3121_MODEL(
            grid_shape=grid_shape,
            dx=dx,
            dt=dt,
            params=test_params,
            initial_conditions=initial_conditions,
            save_steps=1
        )
        
        # Run for a few steps and visualize
        for step in range(steps):
            model._update(step)
            
            # Every few steps, visualize a slice
            if step % 2 == 0:
                plt.figure(figsize=(15, 5))
                
                # Plot healthy cells
                plt.subplot(131)
                plt.imshow(model.phi_H[:, grid_shape[1]//2, :])
                plt.title(f"Healthy Cells - Step {step}")
                plt.colorbar()
                
                # Plot diseased cells
                plt.subplot(132)
                plt.imshow(model.phi_D[:, grid_shape[1]//2, :])
                plt.title(f"Diseased Cells - Step {step}")
                plt.colorbar()
                
                # Plot total tumor
                plt.subplot(133)
                plt.imshow(model.phi_H[:, grid_shape[1]//2, :] + 
                          model.phi_D[:, grid_shape[1]//2, :] + 
                          model.phi_N[:, grid_shape[1]//2, :])
                plt.title(f"Total Tumor - Step {step}")
                plt.colorbar()
                
                plt.tight_layout()
                plt.show()

    
    def visualize_velocity_field(self, model, step=0):
        """
        Visualize the velocity field to identify leaking causes.
        
        Args:
            model: The simulation model object
            step (int): The time step to visualize
        """
        # Get a slice of the tumor
        slice_idx = model.grid_shape[1] // 2
        
        # Calculate velocity field for this state
        try:
            # Try to get data from model history first
            phi_H = model.history['healthy cell volume fraction'][step]
            phi_D = model.history['diseased cell volume fraction'][step]
            phi_N = model.history['necrotic cell volume fraction'][step]
            nutrient = model.history['nutrient concentration'][step]
        except (KeyError, AttributeError, IndexError):
            # Fallback to current model state if history access fails
            phi_H = model.phi_H
            phi_D = model.phi_D
            phi_N = model.phi_N
            nutrient = model.nutrient
        
        # Get parameters
        params = model.params
        dx = model.dx
        
        # Calculate pressure
        p = compute_internal_pressure_scie3121_model(
            phi_H, phi_D, phi_N, nutrient, dx, 
            params['gamma'], params['epsilon'], 
            params['lambda_H'], params['lambda_D'], 
            params['mu_H'], params['mu_D'], params['mu_N'], 
            params['p_H'], params['p_D'], params['n_H'], params['n_D']
        )
        
        # Calculate velocity components
        ux, uy, uz = compute_solid_velocity_scie3121_model(
            phi_H, phi_D, phi_N, nutrient, dx, 
            params['gamma'], params['epsilon'], 
            params['lambda_H'], params['lambda_D'], 
            params['mu_H'], params['mu_D'], params['mu_N'], 
            params['p_H'], params['p_D'], params['n_H'], params['n_D']
        )
        
        # Create the figure
        plt.figure(figsize=(12, 10))
        
        # Plot total tumor
        plt.subplot(221)
        total_tumor = phi_H[:, slice_idx, :] + phi_D[:, slice_idx, :] + phi_N[:, slice_idx, :]
        plt.imshow(total_tumor, origin='lower', cmap='viridis')
        plt.title("Total Tumor")
        plt.colorbar()
        
        # Plot pressure
        plt.subplot(222)
        plt.imshow(p[:, slice_idx, :], origin='lower', cmap='coolwarm')
        plt.title("Pressure Field")
        plt.colorbar()
        
        # Plot velocity magnitude
        plt.subplot(223)
        velocity_mag = np.sqrt(ux[:, slice_idx, :]**2 + uz[:, slice_idx, :]**2)
        plt.imshow(velocity_mag, origin='lower', cmap='plasma')
        plt.title("Velocity Magnitude")
        plt.colorbar()
        
        # Plot velocity vectors
        plt.subplot(224)
        
        # Create a downsampled grid for the quiver plot to avoid overcrowding
        # Determine appropriate downsampling factor based on grid size
        grid_size = min(model.grid_shape[0], model.grid_shape[2])
        downsample = max(1, grid_size // 20)  # Aim for ~20 arrows in each dimension
        
        # Create the grid for quiver plot
        x_indices = np.arange(0, model.grid_shape[0], downsample)
        z_indices = np.arange(0, model.grid_shape[2], downsample)
        X, Z = np.meshgrid(x_indices, z_indices)
        
        # Extract the velocity components at the downsampled grid points
        U = ux[::downsample, slice_idx, ::downsample]
        V = uz[::downsample, slice_idx, ::downsample]
        
        # Adjust the grid to match the downsampled velocity components
        X, Z = np.meshgrid(
            np.linspace(0, model.grid_shape[0] - 1, U.shape[0]),
            np.linspace(0, model.grid_shape[2] - 1, U.shape[1])
        )
        
        # Debugging: Print shapes to identify mismatch
        print(f"X shape: {X.shape}, Z shape: {Z.shape}")
        print(f"U shape: {U.shape}, V shape: {V.shape}")
        
        # Ensure the shapes match
        if U.shape != X.shape or V.shape != Z.shape:
            raise ValueError("Mismatch in shapes of velocity components and grid for quiver plot.")
        
        # Normalize the velocity vectors for better visualization
        # Only if there are non-zero velocities
        magnitude = np.sqrt(U**2 + V**2)
        max_mag = np.max(magnitude)
        
        if max_mag > 0:
            # Plot the quiver with normalized vectors
            plt.quiver(X, Z, U/max_mag, V/max_mag, magnitude, 
                      cmap='viridis', scale=25, pivot='mid')
            plt.colorbar(label='Velocity Magnitude')
        else:
            plt.text(0.5, 0.5, "No velocity vectors to display", 
                    ha='center', va='center', transform=plt.gca().transAxes)
        
        # Show the background tumor field as well
        plt.imshow(total_tumor, origin='lower', cmap='gray', alpha=0.3)
        plt.title("Velocity Field")
        
        plt.tight_layout()
        plt.show()

    def test_sharp_interface(self, steps=10):
        """Test if a spherical tumor maintains a sharp interface."""
        # Create a small grid for faster testing
        grid_shape = (30, 30, 30)
        dx = 0.2
        dt = 0.0001  # Very small time step for stability
        
        # Create initial conditions with a clear spherical tumor
        initial_conditions = SphericalTumor(grid_shape, radius=7, nutrient_value=0.1)
        
        # Create a modified parameter set with minimal diffusion and strong adhesion
        test_params = deepcopy(SCIE3121_params)
        # Increase surface tension
        test_params['gamma'] = test_params.get('gamma', 1.0) * 5.0
        # Decrease interface thickness
        test_params['epsilon'] = test_params.get('epsilon', 1.0) * 0.5
        # Decrease mobility
        if 'M_H' in test_params:
            test_params['M_H'] *= 0.1
        if 'M_D' in test_params:
            test_params['M_D'] *= 0.1
        
        # Initialize model
        model = SCIE3121_MODEL(
            grid_shape=grid_shape,
            dx=dx,
            dt=dt,
            params=test_params,
            initial_conditions=initial_conditions,
            save_steps=1
        )
        
        # Run for a few steps and visualize
        for step in range(steps):
            model._update(step)
            
            # Every few steps, visualize a slice
            if step % 2 == 0:
                plt.figure(figsize=(15, 5))
                
                # Plot healthy cells
                plt.subplot(131)
                plt.imshow(model.phi_H[:, grid_shape[1]//2, :])
                plt.title(f"Healthy Cells - Step {step}")
                plt.colorbar()
                
                # Plot diseased cells
                plt.subplot(132)
                plt.imshow(model.phi_D[:, grid_shape[1]//2, :])
                plt.title(f"Diseased Cells - Step {step}")
                plt.colorbar()
                
                # Plot total tumor
                plt.subplot(133)
                plt.imshow(model.phi_H[:, grid_shape[1]//2, :] + 
                          model.phi_D[:, grid_shape[1]//2, :] + 
                          model.phi_N[:, grid_shape[1]//2, :])
                plt.title(f"Total Tumor - Step {step}")
                plt.colorbar()
                
                plt.tight_layout()
                plt.show()

    def plot_nutrient_field(self, step_index, plane='XY', index=None, smooth_sigma=2.0, 
                        vmin=None, vmax=None, levels=50, cmap='viridis'):
        """
        Plot 2D cross-section of the nutrient field with smoothed contours and customizable scaling.
        
        Args:
            step_index (int): Index of the time step to visualize.
            plane (str): Plane to slice ('XY', 'XZ', 'YZ'). Defaults to 'XY'.
            index (int, optional): Index along the perpendicular axis. Defaults to midpoint.
            smooth_sigma (float): Standard deviation for Gaussian smoothing. 0 for no smoothing.
            vmin (float, optional): Minimum value for colormap scaling. Defaults to data minimum.
            vmax (float, optional): Maximum value for colormap scaling. Defaults to data maximum.
            levels (int): Number of contour levels. Defaults to 50.
            cmap (str): Colormap name. Defaults to 'viridis'.
        """
        if 'nutrient concentration' not in self.history:
            raise ValueError("Nutrient concentration data not found in the simulation history.")
        
        nutrient_field = self.history['nutrient concentration'][step_index]
        
        shape = nutrient_field.shape
        if index is None:
            if plane == 'XY':
                index = shape[2] // 2
            elif plane == 'XZ':
                index = shape[1] // 2
            elif plane == 'YZ':
                index = shape[0] // 2
            else:
                raise ValueError("Plane must be 'XY', 'XZ', or 'YZ'")
        
        # Select the slice based on the plane
        if plane == 'XY':
            slice_ = nutrient_field[:, :, index]
            x, y = np.arange(shape[0]), np.arange(shape[1])
            xlabel, ylabel = 'X', 'Y'
        elif plane == 'XZ':
            slice_ = nutrient_field[:, index, :]
            x, y = np.arange(shape[0]), np.arange(shape[2])
            xlabel, ylabel = 'X', 'Z'
        elif plane == 'YZ':
            slice_ = nutrient_field[index, :, :]
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
        fig.colorbar(contour, ax=ax, label='Nutrient Concentration')
        
        # Set labels and titles
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'Nutrient Cross-Section ({plane} at index {index})')
        fig.suptitle(f'Step {step_index}', y=1.05)
        
        plt.show()

    def plot_nutrient_evolution(self, smooth_window=5):
        """
        Plot the evolution of total nutrient concentration over time.
        
        Args:
            smooth_window (int): Window size for Savitzky-Golay smoothing (odd number).
        """
        if 'nutrient concentration' not in self.history:
            raise ValueError("Nutrient concentration data not found in the simulation history.")
        
        steps = self.history['step']
        nutrient_total = [np.sum(n) for n in self.history['nutrient concentration']]
        
        # Apply smoothing if window > 1
        if smooth_window > 1:
            if smooth_window % 2 == 0:
                smooth_window += 1  # Ensure odd window size
            nutrient_total = savgol_filter(nutrient_total, smooth_window, 3)
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, nutrient_total, label='Total Nutrient', color='blue')
        plt.xlabel('Time Step')
        plt.ylabel('Total Nutrient Concentration')
        plt.title('Nutrient Evolution Over Time')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

    def animate_nutrient_field(self, plane='XY', index=None, interval=200, save_as=None):
        """
        Animate 2D cross-sections of the nutrient field over time.
        
        Args:
            plane (str): Plane to slice ('XY', 'XZ', 'YZ'). Defaults to 'XY'.
            index (int, optional): Index along the perpendicular axis. Defaults to midpoint.
            interval (int): Delay between frames in milliseconds. Defaults to 200.
            save_as (str, optional): File path to save animation (e.g., 'nutrient_animation.mp4').
        """
        if 'nutrient concentration' not in self.history:
            raise ValueError("Nutrient concentration data not found in the simulation history.")
        
        # Get the shape from the first step to determine default index
        shape = self.history['nutrient concentration'][0].shape
        if index is None:
            if plane == 'XY':
                index = shape[2] // 2
            elif plane == 'XZ':
                index = shape[1] // 2
            elif plane == 'YZ':
                index = shape[0] // 2
            else:
                raise ValueError("Plane must be 'XY', 'XZ', or 'YZ'")
        
        # Set up the figure and axis
        fig, ax = plt.subplots(figsize=(10, 8))
        
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
        
        # Get initial slice for setting up the plot
        if plane == 'XY':
            initial_slice = self.history['nutrient concentration'][0][:, :, index]
        elif plane == 'XZ':
            initial_slice = self.history['nutrient concentration'][0][:, index, :]
        elif plane == 'YZ':
            initial_slice = self.history['nutrient concentration'][0][index, :, :]
        
        # Apply Gaussian smoothing
        initial_slice = gaussian_filter(initial_slice, sigma=1.0)
        
        # Interpolate to finer grid
        initial_slice_fine = griddata((X.flatten(), Y.flatten()), initial_slice.flatten(), 
                                    (X_fine, Y_fine), method='cubic')
        
        # Find global min/max for consistent colormap
        all_nutrient = self.history['nutrient concentration']
        vmin = 0
        vmax = max(1.0, np.max([np.max(n) for n in all_nutrient]))
        
        # Create initial contour plot
        contour = ax.contourf(X_fine, Y_fine, initial_slice_fine, levels=50, 
                            cmap='viridis', vmin=vmin, vmax=vmax)
        fig.colorbar(contour, ax=ax, label='Nutrient Concentration')
        
        # Set labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'Nutrient Field ({plane} Plane, Step {self.history["step"][0]})')
        
        def update(frame):
            """Update function for animation"""
            # Get current slice data from history
            if plane == 'XY':
                slice_ = self.history['nutrient concentration'][frame][:, :, index]
            elif plane == 'XZ':
                slice_ = self.history['nutrient concentration'][frame][:, index, :]
            elif plane == 'YZ':
                slice_ = self.history['nutrient concentration'][frame][index, :, :]
            
            # Apply Gaussian smoothing
            slice_ = gaussian_filter(slice_, sigma=1.0)
            
            # Interpolate to finer grid
            slice_fine = griddata((X.flatten(), Y.flatten()), slice_.flatten(), 
                                (X_fine, Y_fine), method='cubic')
            
            # Clear the axis and create a new contour plot
            ax.clear()
            contour = ax.contourf(X_fine, Y_fine, slice_fine, levels=50, 
                                cmap='viridis', vmin=vmin, vmax=vmax)
            
            # Reset labels and title
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f'Nutrient Field ({plane} Plane, Step {self.history["step"][frame]})')
            
            return [contour]
        
        anim = FuncAnimation(fig, update, frames=len(self.history['step']), interval=interval, blit=False)
        
        if save_as:
            anim.save(save_as, writer='ffmpeg')  # Requires ffmpeg installed
        
        plt.tight_layout()
        plt.show()
        
        return anim  # Return the animation object to prevent garbage collection

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

def main():

    model = SCIE3121_MODEL(
        grid_shape=(50, 50, 50),
        dx=0.2,
        dt=0.001,
        params=SCIE3121_params,
        initial_conditions=SphericalTumor(grid_shape=(30, 30, 30), radius=7, nutrient_value=1.0),
    )
    analyzer = scie3121SimulationAnalyzer(filepath='data/project_model_test_sim_data.npz') 
    analyzer.visualize_velocity_field(model, step=70)
    #analyzer.animate_cross_section(plane='XY', interval=200)
    #analyzer.plot_combined_cross_section(step_index=0, plane='XY', index=None, smooth_sigma=2.0, levels=50, cmaps=None) 
    #analyzer.plot_nutrient_field(step_index=14, plane='XY', index=None, smooth_sigma=2.0, vmin=None, vmax=None, levels=50, cmap='viridis')
    #analyzer.animate_nutrient_field(plane='XY', index=None, interval=200, save_as=None)

if __name__ == "__main__":
    main()