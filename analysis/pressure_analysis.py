import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes

from analysis.scie3121_analysis import scie3121SimulationAnalyzer
from src.models.cell_dynamics import (
    compute_internal_pressure_scie3121_model,
    compute_adhesion_energy_derivative_with_laplace,
    laplacian,
    gradient_neumann
)
from src.models.SCIE3121_model import SCIE3121_MODEL
from src.models.initial_conditions import SphericalTumor
from src.utils.utils import SCIE3121_params
from matplotlib.animation import FuncAnimation

class PressureAnalyzer(scie3121SimulationAnalyzer):
    def __init__(self, filepath):
        super().__init__(filepath)

    
    def visualize_pressure_field(self, model, step, mode='3d', plane='XY', index=None, 
                                    levels=50, cmap='coolwarm', show_tumor_outline=True,
                                    show_derivatives=False, threshold=0.1):
        """
        Visualize the pressure field at a specific time step in multiple ways.
        
        Args:
            model: The simulation model object
            step (int): The time step to visualize
            mode (str): Visualization mode - '3d', 'slice', 'contour', 'surface', or 'all'
            plane (str): Plane to slice ('XY', 'XZ', 'YZ') for slice and contour modes
            index (int, optional): Index along the perpendicular axis. Defaults to midpoint.
            levels (int): Number of contour levels for contour plots
            cmap (str): Colormap for visualization
            show_tumor_outline (bool): Whether to overlay tumor outline on plots
            show_derivatives (bool): Whether to show pressure derivatives/gradients
            threshold (float): Threshold for tumor outline
        """
        # Get the pressure field for the specified step
        try:
            phi_H = self.history['healthy cell volume fraction'][step]
            phi_D = self.history['diseased cell volume fraction'][step]
            phi_N = self.history['necrotic cell volume fraction'][step]
            nutrient = self.history['nutrient concentration'][step]
        except (KeyError, IndexError) as e:
            print(f"Error accessing history data at step {step}: {e}")
            return
        
        # Compute phi_T
        phi_T = phi_H + phi_D + phi_N
        
        # Compute shared quantities once (following the optimization pattern)
        laplace_phi = laplacian(phi_T, model.dx)
        
        # Compute energy derivative using the optimized function if needed
        if show_derivatives:
            energy_deriv = compute_adhesion_energy_derivative_with_laplace(
                phi_T, laplace_phi, model.params['gamma'], model.params['epsilon']
            )
        
        # Calculate pressure field
        p = compute_internal_pressure_scie3121_model(
            phi_H, phi_D, phi_N, nutrient, model.dx, 
            model.params['gamma'], model.params['epsilon'],
            model.params['lambda_H'], model.params['lambda_D'], 
            model.params['mu_H'], model.params['mu_D'], model.params['mu_N'],
            model.params['p_H'], model.params['p_D'], model.params['n_H'], model.params['n_D']
        )
        
        # Get shape and determine default index if not provided
        shape = p.shape
        if index is None:
            if plane == 'XY':
                index = shape[2] // 2
            elif plane == 'XZ':
                index = shape[1] // 2
            elif plane == 'YZ':
                index = shape[0] // 2
        
        # Extract slice based on the plane
        if plane == 'XY':
            p_slice = p[:, :, index]
            tumor_slice = phi_T[:, :, index]
            x_label, y_label = 'X', 'Y'
            grad_y, grad_x = np.gradient(p_slice, model.dx)  # axis 0 is y, axis 1 is x
        elif plane == 'XZ':
            p_slice = p[:, index, :]
            tumor_slice = phi_T[:, index, :]
            x_label, y_label = 'X', 'Z'
            grad_z, grad_x = np.gradient(p_slice, model.dx)
        elif plane == 'YZ':
            p_slice = p[index, :, :]
            tumor_slice = phi_T[index, :, :]
            x_label, y_label = 'Y', 'Z'
            grad_z, grad_y = np.gradient(p_slice, model.dx)
                
        # Calculate pressure gradient magnitude for the slice
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Determine visualization mode
        if mode == 'all':
            fig = plt.figure(figsize=(18, 12))
            gs = plt.GridSpec(2, 3, figure=fig)
            
            # 3D isosurface
            ax1 = fig.add_subplot(gs[0, 0], projection='3d')
            self._plot_pressure_3d(p, phi_T, ax1, threshold, cmap)
            
            # 2D slice
            ax2 = fig.add_subplot(gs[0, 1])
            self._plot_pressure_slice(p_slice, tumor_slice, ax2, x_label, y_label, 
                                     show_tumor_outline, threshold, cmap)
            
            # Contour plot
            ax3 = fig.add_subplot(gs[0, 2])
            self._plot_pressure_contour(p_slice, tumor_slice, ax3, x_label, y_label, 
                                       levels, show_tumor_outline, threshold, cmap)
            
            # Gradient magnitude
            ax4 = fig.add_subplot(gs[1, 0])
            self._plot_gradient_magnitude(grad_magnitude, tumor_slice, ax4, x_label, y_label, 
                                         show_tumor_outline, threshold)
            
            # Gradient vector field
            ax5 = fig.add_subplot(gs[1, 1])
            self._plot_gradient_vectors(grad_x, grad_y, tumor_slice, ax5, x_label, y_label, 
                                       show_tumor_outline, threshold)
            
            # 3D pressure surface
            ax6 = fig.add_subplot(gs[1, 2], projection='3d')
            self._plot_pressure_surface(p_slice, tumor_slice, ax6, x_label, y_label, cmap)
            
            plt.tight_layout()
            plt.suptitle(f'Pressure Field Analysis - Step {step}', fontsize=16, y=1.02)
            
        elif mode == '3d':
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            self._plot_pressure_3d(p, phi_T, ax, threshold, cmap)
            plt.title(f'3D Pressure Field - Step {step}')
            
        elif mode == 'slice':
            fig, ax = plt.subplots(figsize=(10, 8))
            self._plot_pressure_slice(p_slice, tumor_slice, ax, x_label, y_label, 
                                     show_tumor_outline, threshold, cmap)
            plt.title(f'Pressure Field {plane}-Slice at index {index} - Step {step}')
            
        elif mode == 'contour':
            fig, ax = plt.subplots(figsize=(10, 8))
            self._plot_pressure_contour(p_slice, tumor_slice, ax, x_label, y_label, 
                                       levels, show_tumor_outline, threshold, cmap)
            plt.title(f'Pressure Contours {plane}-Slice at index {index} - Step {step}')
            
        elif mode == 'surface':
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            self._plot_pressure_surface(p_slice, tumor_slice, ax, x_label, y_label, cmap)
            plt.title(f'Pressure Surface {plane}-Slice at index {index} - Step {step}')
            
        elif mode == 'derivatives':
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Gradient magnitude
            self._plot_gradient_magnitude(grad_magnitude, tumor_slice, axes[0], 
                                         x_label, y_label, show_tumor_outline, threshold)
            axes[0].set_title('Pressure Gradient Magnitude')
            
            # Gradient vector field
            self._plot_gradient_vectors(grad_x, grad_y, tumor_slice, axes[1], 
                                       x_label, y_label, show_tumor_outline, threshold)
            axes[1].set_title('Pressure Gradient Vectors')
            
            # 1D profile
            self._plot_pressure_profile(p_slice, tumor_slice, axes[2], plane)
            axes[2].set_title('Pressure Profile Along Center')
            
            plt.suptitle(f'Pressure Derivatives Analysis - Step {step}', fontsize=16)
            plt.tight_layout()
            
        plt.show()
        
        # Return the pressure field for further analysis if needed
        #return p
    
    def _plot_pressure_3d(self, p, phi_T, ax, threshold, cmap):
        """Helper method to plot 3D pressure isosurfaces."""
        # Find reasonable isosurface levels
        p_min, p_max = np.min(p), np.max(p)
        levels = np.linspace(p_min, p_max, 5)
        
        # Plot pressure isosurfaces
        for level in levels:
            if np.any(p > level):
                try:
                    verts, faces, _, _ = marching_cubes(p, level=level)
                    # Fix deprecated get_cmap call
                    colormap = plt.colormaps[cmap]
                    color = colormap((level - p_min) / (p_max - p_min))
                    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], 
                                   color=color, alpha=0.3)
                except:
                    pass  # Skip if marching cubes fails for this level
        
        # Plot tumor outline
        if np.any(phi_T > threshold):
            try:
                verts, faces, _, _ = marching_cubes(phi_T, level=threshold)
                ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], 
                               color='black', alpha=0.1)
            except:
                pass
        
        plt.title('Pressure 3D')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    def _plot_pressure_slice(self, p_slice, tumor_slice, ax, x_label, y_label, 
                            show_tumor_outline, threshold, cmap):
        """Helper method to plot 2D pressure slice."""
        im = ax.imshow(p_slice, origin='lower', cmap=cmap)
        plt.colorbar(im, ax=ax, label='Pressure')
        plt.title('Pressure Slice')
        
        # Overlay tumor outline if requested
        if show_tumor_outline:
            ax.contour(tumor_slice, levels=[threshold], colors='black', linewidths=1.5)
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
    
    def _plot_pressure_contour(self, p_slice, tumor_slice, ax, x_label, y_label, 
                              levels, show_tumor_outline, threshold, cmap):
        """Helper method to plot pressure contours."""
        # Apply Gaussian smoothing for better contours
        p_smooth = gaussian_filter(p_slice, sigma=1.0)
        
        # Create contour plot
        contour = ax.contourf(p_smooth, levels=levels, cmap=cmap)
        plt.colorbar(contour, ax=ax, label='Pressure')
        plt.title('Pressure Contours')
        # Add contour lines
        ax.contour(p_smooth, levels=levels, colors='black', linewidths=0.5, alpha=0.5)
        
        # Overlay tumor outline if requested
        if show_tumor_outline:
            ax.contour(tumor_slice, levels=[threshold], colors='white', linewidths=2)
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
    
    def _plot_gradient_magnitude(self, grad_magnitude, tumor_slice, ax, x_label, y_label, 
                                show_tumor_outline, threshold):
        """Helper method to plot pressure gradient magnitude."""
        # Apply Gaussian smoothing for better visualization
        grad_smooth = gaussian_filter(grad_magnitude, sigma=1.0)
        
        im = ax.imshow(grad_smooth, origin='lower', cmap='viridis')
        plt.colorbar(im, ax=ax, label='Gradient Magnitude')
        plt.title('Pressure Gradient Magnitude')
        # Overlay tumor outline if requested
        if show_tumor_outline:
            ax.contour(tumor_slice, levels=[threshold], colors='white', linewidths=1.5)
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
    
    def _plot_gradient_vectors(self, grad_x, grad_y, tumor_slice, ax, x_label, y_label, 
                              show_tumor_outline, threshold):
        """Helper method to plot pressure gradient vectors."""
        # Downsample for clearer vector field
        shape = grad_x.shape
        downsample = max(1, min(shape) // 20)
        
        x_indices = np.arange(0, shape[0], downsample)
        y_indices = np.arange(0, shape[1], downsample)
        X, Y = np.meshgrid(x_indices, y_indices)
        
        # Extract downsampled gradients
        U = grad_x[x_indices][:, y_indices]
        V = grad_y[x_indices][:, y_indices]
        
        # Normalize vectors for better visualization
        magnitude = np.sqrt(U**2 + V**2)
        max_mag = np.max(magnitude) if np.max(magnitude) > 0 else 1.0
        U = U / max_mag
        V = V / max_mag
        
        # Plot background tumor slice for context
        ax.imshow(tumor_slice, origin='lower', cmap='gray', alpha=0.3)
        
        # Plot vector field
        quiver = ax.quiver(X, Y, U, V, magnitude, cmap='viridis', 
                          scale=20, width=0.006, pivot='mid')
        plt.colorbar(quiver, ax=ax, label='Gradient Strength')
        plt.title('Pressure Gradient Vectors')
        
        # Overlay tumor outline if requested
        if show_tumor_outline:
            ax.contour(tumor_slice, levels=[threshold], colors='white', linewidths=1.5)
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
    
    def _plot_pressure_profile(self, p_slice, tumor_slice, ax, plane):
        """Helper method to plot 1D pressure profile along center."""
        # Get center indices
        center_x = p_slice.shape[0] // 2
        center_y = p_slice.shape[1] // 2
        
        # Extract profiles along both axes
        profile_x = p_slice[center_x, :]
        profile_y = p_slice[:, center_y]
        
        # Extract tumor profiles for context
        tumor_x = tumor_slice[center_x, :]
        tumor_y = tumor_slice[:, center_y]
        
        # Plot profiles
        x_coords = np.arange(len(profile_x))
        y_coords = np.arange(len(profile_y))
        
        ax.plot(x_coords, profile_x, 'b-', label=f'{plane[1]}-axis profile')
        ax.plot(y_coords, profile_y, 'r-', label=f'{plane[0]}-axis profile')
        
        # Plot tumor profiles on secondary y-axis
        ax2 = ax.twinx()
        ax2.plot(x_coords, tumor_x, 'b--', alpha=0.5, label=f'{plane[1]}-axis tumor')
        ax2.plot(y_coords, tumor_y, 'r--', alpha=0.5, label=f'{plane[0]}-axis tumor')
        ax2.set_ylabel('Tumor Volume Fraction', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
        
        # Add labels and legend
        ax.set_xlabel('Position (grid units)')
        ax.set_ylabel('Pressure')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        # Add grid for readability
        ax.grid(True, linestyle='--', alpha=0.7)

    def _plot_pressure_surface(self, p_slice, tumor_slice, ax, x_label, y_label, cmap):
        """Helper method to plot 3D pressure surface focused on a 20x20 box around the tumor center."""
        # Find the center of the tumor (weighted by tumor density)
        if np.any(tumor_slice > 0):
            # Use center of mass if tumor is present
            from scipy import ndimage
            center_y, center_x = ndimage.center_of_mass(tumor_slice)
        else:
            # Otherwise use the middle of the grid
            center_y, center_x = tumor_slice.shape[0] // 2, tumor_slice.shape[1] // 2
        
        # Convert to integers
        center_x, center_y = int(center_x), int(center_y)
        
        # Define the 20x20 box around the center
        box_size = 100
        half_size = box_size // 2
        
        # Calculate box boundaries with bounds checking
        x_min = max(0, center_x - half_size)
        x_max = min(tumor_slice.shape[1], center_x + half_size)
        y_min = max(0, center_y - half_size)
        y_max = min(tumor_slice.shape[0], center_y + half_size)
        
        # Extract the region of interest
        p_roi = p_slice[y_min:y_max, x_min:x_max]
        tumor_roi = tumor_slice[y_min:y_max, x_min:x_max]
        
        # Apply Gaussian smoothing for better visualization
        p_smooth = gaussian_filter(p_roi, sigma=1.0)
        
        # Create coordinate grids for the ROI
        x = np.arange(x_min, x_max)
        y = np.arange(y_min, y_max)
        X, Y = np.meshgrid(x, y)
        
        # Create the surface plot
        surf = ax.plot_surface(X, Y, p_smooth, cmap=cmap, 
                              linewidth=0, antialiased=True, alpha=0.8)
        
        # Add a color bar
        fig = ax.figure
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Pressure')
        
        # Add tumor outline as a contour at the bottom of the plot
        if np.any(tumor_roi > 0):
            # Find the minimum pressure value for the base of the plot
            z_base = np.min(p_smooth)
            
            # Create a contour of the tumor at the base
            tumor_smooth = gaussian_filter(tumor_roi, sigma=1.0)
            threshold = 0.1 * np.max(tumor_smooth)
            
            # Find the contour using scikit-image's find_contours
            from skimage import measure
            contours = measure.find_contours(tumor_smooth, threshold)
            
            # Plot each contour at the base of the 3D plot
            for contour in contours:
                # Contour points are (row, column), need to swap for (x, y)
                # Also need to offset by the ROI boundaries
                ax.plot(contour[:, 1] + x_min, contour[:, 0] + y_min, 
                       z_base * np.ones(contour.shape[0]), 
                       'k-', linewidth=2, alpha=0.7)
        
        # Set labels and adjust view
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel('Pressure')
        
        # Set a good viewing angle
        ax.view_init(elev=30, azim=90)
        
        # Add a grid for better depth perception
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Set axis limits to show only the ROI
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # Add a title indicating this is a zoomed view
        ax.set_title(f"Pressure Surface")

    def animate_pressure_field(self, model, plane='XY', index=None, interval=200, save_as=None, show_tumor_outline=True, threshold=0.1, cmap='coolwarm'):
        """
        Animate the pressure field over time.
        
        Args:
            model (SCIE3121_MODEL): The model instance containing grid parameters
            plane (str): Plane to visualize ('XY', 'XZ', or 'YZ')
            index (int, optional): Index for the slice. If None, uses middle slice
            interval (int): Interval between frames in milliseconds
            save_as (str, optional): Filename to save animation. If None, displays instead
            show_tumor_outline (bool): Whether to show tumor boundary contour
            threshold (float): Threshold for tumor outline
            cmap (str): Colormap to use for pressure visualization
        """
        # Set up the figure and axis
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Get total number of timesteps
        num_steps = len(self.history['step'])
        
        # Initialize with first frame
        im = self._plot_pressure_slice(model, step=0, plane=plane, index=index, 
                                    ax=ax, show_tumor_outline=show_tumor_outline, 
                                    threshold=threshold, cmap=cmap)
        
        def update(frame):
            ax.clear()
            im = self._plot_pressure_slice(model, step=frame, plane=plane, index=index, 
                                        ax=ax, show_tumor_outline=show_tumor_outline, 
                                        threshold=threshold, cmap=cmap)
            ax.set_title(f'Pressure Field - Step {frame}')
            return [im]
        
        # Create animation
        anim = FuncAnimation(fig, update, frames=num_steps, 
                            interval=interval, blit=True)
        
        # Save or display
        if save_as:
            anim.save(save_as, writer='pillow')
        else:
            plt.show()
            
        return anim

def main():
    model = SCIE3121_MODEL(
        grid_shape=(50, 50, 50),
        dx=0.2,
        dt=0.001,
        params=SCIE3121_params,
        initial_conditions=SphericalTumor(grid_shape=(30, 30, 30), radius=7, nutrient_value=1.0),)
    
    analyzer = PressureAnalyzer(filepath='data/project_model_test_sim_data.npz')
    analyzer.visualize_pressure_field(model, step=-1, mode='all', plane='XY', index=None, levels=50, cmap='coolwarm', show_tumor_outline=True, show_derivatives=False, threshold=0.1)
    #analyzer.animate_pressure_field(model, plane='XY', index=None, interval=200, save_as=None, show_tumor_outline=True, threshold=0.1, cmap='coolwarm')

if __name__ == "__main__":
    main()