import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes

from analysis.scie3121_analysis import scie3121SimulationAnalyzer
from src.models.cell_dynamics import compute_solid_velocity_scie3121_model, compute_internal_pressure_scie3121_model, compute_adhesion_energy_derivative
from src.models.SCIE3121_model import SCIE3121_MODEL
from src.models.initial_conditions import SphericalTumor
from src.utils.utils import SCIE3121_params

class SolidVelocityAnalyzer(scie3121SimulationAnalyzer):
    def __init__(self, filepath):
        super().__init__(filepath)
    
    def visualize_velocity_field(self, model, step, mode='3d', plane='XY', index=None, 
                                cmap='viridis', show_tumor_outline=True, 
                                threshold=0.1, scale=None, streamlines=True):
        """
        Visualize the solid velocity field at a specific time step in multiple ways.
        
        Args:
            model: The simulation model object
            step (int): The time step to visualize
            mode (str): Visualization mode - '3d', 'slice', 'vector', 'streamlines', or 'all'
            plane (str): Plane to slice ('XY', 'XZ', 'YZ') for slice modes
            index (int, optional): Index along the perpendicular axis. Defaults to midpoint.
            cmap (str): Colormap for visualization
            show_tumor_outline (bool): Whether to overlay tumor outline on plots
            threshold (float): Threshold for tumor outline
            scale (float, optional): Scaling factor for velocity vectors
            streamlines (bool): Whether to plot streamlines (for slice and vector modes)
        """
        # Get the cell volume fractions and nutrient for the specified step
        try:
            phi_H = self.history['healthy cell volume fraction'][step]
            phi_D = self.history['diseased cell volume fraction'][step]
            phi_N = self.history['necrotic cell volume fraction'][step]
            nutrient = self.history['nutrient concentration'][step]
        except (KeyError, IndexError) as e:
            print(f"Error accessing history data at step {step}: {e}")
            return
        
        # Calculate total tumor volume fraction
        phi_T = phi_H + phi_D + phi_N
        
        # Calculate the velocity field
        ux, uy, uz = compute_solid_velocity_scie3121_model(
            phi_H, phi_D, phi_N, nutrient, model.dx, 
            model.params['gamma'], model.params['epsilon'],
            model.params['lambda_H'], model.params['lambda_D'], 
            model.params['mu_H'], model.params['mu_D'], model.params['mu_N'],
            model.params['p_H'], model.params['p_D'], model.params['n_H'], model.params['n_D']
        )
        
        # Calculate velocity magnitude
        velocity_magnitude = np.sqrt(ux**2 + uy**2 + uz**2)
        
        # Get shape and determine default index if not provided
        shape = phi_T.shape
        if index is None:
            if plane == 'XY':
                index = shape[2] // 2
            elif plane == 'XZ':
                index = shape[1] // 2
            elif plane == 'YZ':
                index = shape[0] // 2
        
        # Extract slice based on the plane
        if plane == 'XY':
            ux_slice = ux[:, :, index]
            uy_slice = uy[:, :, index]
            uz_slice = uz[:, :, index]
            vmag_slice = velocity_magnitude[:, :, index]
            tumor_slice = phi_T[:, :, index]
            x_label, y_label = 'X', 'Y'
            primary_components = (ux_slice, uy_slice)
        elif plane == 'XZ':
            ux_slice = ux[:, index, :]
            uy_slice = uy[:, index, :]
            uz_slice = uz[:, index, :]
            vmag_slice = velocity_magnitude[:, index, :]
            tumor_slice = phi_T[:, index, :]
            x_label, y_label = 'X', 'Z'
            primary_components = (ux_slice, uz_slice)
        elif plane == 'YZ':
            ux_slice = ux[index, :, :]
            uy_slice = uy[index, :, :]
            uz_slice = uz[index, :, :]
            vmag_slice = velocity_magnitude[index, :, :]
            tumor_slice = phi_T[index, :, :]
            x_label, y_label = 'Y', 'Z'
            primary_components = (uy_slice, uz_slice)
            
        # Compute auto-scaling for vectors if not provided
        if scale is None:
            # Automatic scaling based on maximum velocity and grid size
            max_vel = np.max(velocity_magnitude)
            if max_vel > 0:
                # Scale such that the largest vector is about 1/20th of the grid size
                scale = min(shape) / (20 * max_vel)
            else:
                scale = 1.0
        
        # Determine visualization mode
        if mode == 'all':
            fig = plt.figure(figsize=(18, 12))
            gs = plt.GridSpec(2, 3, figure=fig)
            
            # 3D velocity magnitude isosurfaces
            ax1 = fig.add_subplot(gs[0, 0], projection='3d')
            self._plot_velocity_3d(velocity_magnitude, phi_T, ax1, threshold, cmap)
            
            # 2D velocity magnitude slice
            ax2 = fig.add_subplot(gs[0, 1])
            self._plot_velocity_magnitude_slice(vmag_slice, tumor_slice, ax2, x_label, y_label,
                                             show_tumor_outline, threshold, cmap)
            
            # Vector field
            ax3 = fig.add_subplot(gs[0, 2])
            self._plot_velocity_vectors(primary_components[0], primary_components[1], tumor_slice, 
                                     ax3, x_label, y_label, show_tumor_outline, threshold, scale)
            
    
            # Velocity divergence field
            ax5 = fig.add_subplot(gs[1, 1])
            self._plot_velocity_divergence(ux, uy, uz, phi_T, ax5, plane, index, 
                                        model.dx, show_tumor_outline, threshold)
            
            # Velocity profiles
            ax6 = fig.add_subplot(gs[1, 2])
            self._plot_velocity_profiles(primary_components[0], primary_components[1], tumor_slice, 
                                      ax6, x_label, y_label)
            
            plt.tight_layout()
            plt.suptitle(f'Solid Velocity Field Analysis - Step {step}', fontsize=16, y=1.02)
            
        elif mode == '3d':
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            self._plot_velocity_3d(velocity_magnitude, phi_T, ax, threshold, cmap)
            plt.title(f'3D Velocity Magnitude Field - Step {step}')
            
        elif mode == 'slice':
            fig, ax = plt.subplots(figsize=(10, 8))
            self._plot_velocity_magnitude_slice(vmag_slice, tumor_slice, ax, x_label, y_label,
                                             show_tumor_outline, threshold, cmap)
            plt.title(f'Velocity Magnitude {plane}-Slice at index {index} - Step {step}')
            
        elif mode == 'vector':
            fig, ax = plt.subplots(figsize=(10, 8))
            self._plot_velocity_vectors(primary_components[0], primary_components[1], tumor_slice, 
                                     ax, x_label, y_label, show_tumor_outline, threshold, scale)
            plt.title(f'Velocity Vectors {plane}-Slice at index {index} - Step {step}')

        elif mode == 'divergence':
            fig, ax = plt.subplots(figsize=(10, 8))
            self._plot_velocity_divergence(ux, uy, uz, phi_T, ax, plane, index, 
                                        model.dx, show_tumor_outline, threshold)
            plt.title(f'Velocity Divergence {plane}-Slice at index {index} - Step {step}')
            
        plt.show()
        
        # Return the velocity components for further analysis if needed
        return #ux, uy, uz, velocity_magnitude
    
    def _plot_velocity_3d(self, velocity_magnitude, phi_T, ax, threshold, cmap):
        """Helper method to plot 3D velocity magnitude isosurfaces."""
        # Find reasonable isosurface levels for velocity
        v_min, v_max = np.min(velocity_magnitude), np.max(velocity_magnitude)
        if v_min == v_max:
            print("Warning: Velocity field is uniform, no isosurfaces to display")
            return
            
        levels = np.linspace(v_min + 0.1 * (v_max - v_min), v_max, 4)
        
        # Plot velocity isosurfaces
        for level in levels:
            if np.any(velocity_magnitude > level):
                try:
                    verts, faces, _, _ = marching_cubes(velocity_magnitude, level=level)
                    # Use proper colormap access
                    colormap = plt.colormaps[cmap]
                    color = colormap((level - v_min) / (v_max - v_min))
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
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Velocity Magnitude Isosurfaces with Tumor Outline')
    
    def _plot_velocity_vector_3d(self, ux, uy, uz, phi_T, ax, threshold, cmap):
        """Helper method to plot 3D velocity vector field."""
        # Compute velocity magnitude
        velocity_mag = np.sqrt(ux**2 + uy**2 + uz**2)
        
        # Create a mask for the tumor region
        tumor_mask = phi_T > threshold
        
        # Downsample for 3D visualization
        shape = ux.shape
        downsample = max(1, min(shape) // 10)
        
        x_indices = np.arange(0, shape[0], downsample)
        y_indices = np.arange(0, shape[1], downsample)
        z_indices = np.arange(0, shape[2], downsample)
        X, Y, Z = np.meshgrid(x_indices, y_indices, z_indices, indexing='ij')
        
        # Extract downsampled velocity components
        U = ux[x_indices][:, y_indices][:, :, z_indices]
        V = uy[x_indices][:, y_indices][:, :, z_indices]
        W = uz[x_indices][:, y_indices][:, :, z_indices]
        
        # Calculate vector magnitudes for coloring
        magnitude = np.sqrt(U**2 + V**2 + W**2)
        
        # Create tumor isosurface
        verts, faces, _, _ = marching_cubes(phi_T, threshold)
        
        # Plot tumor surface
        ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], 
                       triangles=faces, alpha=0.3, color='gray')
        
        # Plot velocity vectors in 3D
        ax.quiver(X, Y, Z, U, V, W, length=2.0, normalize=True, 
                 colors=plt.cm.viridis(magnitude/np.max(magnitude)), alpha=0.7)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Velocity Field')
    
    def _plot_velocity_magnitude_slice(self, vmag_slice, tumor_slice, ax, x_label, y_label,
                                    show_tumor_outline, threshold, cmap):
        """Helper method to plot 2D velocity magnitude slice."""
        # Apply smoothing for better visualization
        vmag_smooth = gaussian_filter(vmag_slice, sigma=1.0)
        
        im = ax.imshow(vmag_smooth, origin='lower', cmap=cmap)
        plt.colorbar(im, ax=ax, label='Velocity Magnitude')
        
        # Overlay tumor outline if requested
        if show_tumor_outline:
            ax.contour(tumor_slice, levels=[threshold], colors='black', linewidths=1.5)
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title('Velocity Magnitude')
    
    def _plot_velocity_vectors(self, u_comp, v_comp, tumor_slice, ax, x_label, y_label,
                            show_tumor_outline, threshold, scale):
        """Helper method to plot velocity vector field."""
        # Apply smoothing for clearer vector field
        u_smooth = gaussian_filter(u_comp, sigma=1.0)
        v_smooth = gaussian_filter(v_comp, sigma=1.0)
        
        # Create a grid for the quiver plot
        ny, nx = u_smooth.shape
        y = np.arange(0, ny)
        x = np.arange(0, nx)
        X, Y = np.meshgrid(x, y)
        
        # Downsample for clearer vector field
        downsample = max(1, min(nx, ny) // 25)  # Aggressive downsampling
        
        # Fix the indexing for proper vector orientation
        # For the quiver plot with matplotlib, we need to transpose these arrays
        # relative to our data layout
        X_ds = X[::downsample, ::downsample]
        Y_ds = Y[::downsample, ::downsample]
        
        # The key fix: we need to transpose the components to match the meshgrid
        # We also may need to flip the sign of one component for correct orientation
        U_ds = v_smooth[::downsample, ::downsample]  # Swap u and v components
        V_ds = u_smooth[::downsample, ::downsample]  # to match X and Y orientation
        
        # Calculate vector magnitudes for coloring
        magnitude = np.sqrt(U_ds**2 + V_ds**2)
        
        # Plot background tumor slice for context
        ax.imshow(tumor_slice, origin='lower', cmap='gray', alpha=0.3)
        
        # Determine maximum magnitude for scaling
        max_mag = np.max(magnitude) if np.max(magnitude) > 0 else 1.0
        
        
        # Plot the vector field with adjusted parameters
        quiver = ax.quiver(X_ds, Y_ds, U_ds, V_ds, magnitude, 
                         cmap='viridis', 
                         scale=0.06,
                         scale_units='xy',
                         width=0.01,
                         headwidth=3,
                         headlength=4,
                         pivot='mid',
                         alpha=0.8)
        
        ax.quiverkey(quiver, 0.9, 0.95, max_mag, f"{max_mag:.2e}", labelpos='E', 
                    coordinates='figure', color='black')
        
        plt.colorbar(quiver, ax=ax, label='Velocity Magnitude')
        
        # Overlay tumor outline
        if show_tumor_outline:
            ax.contour(tumor_slice, levels=[threshold], colors='white', linewidths=1.5)
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title('Velocity Vectors')
        
        # Add axis limits to match the image dimensions
        ax.set_xlim(0, nx-1)
        ax.set_ylim(0, ny-1)
    
    
    def _plot_velocity_divergence(self, ux, uy, uz, tumor_volume, ax, plane, index, dx, 
                           show_tumor_outline, threshold):
        """Helper method to plot velocity divergence field."""
        # Calculate the divergence of the velocity field
        div_u = np.gradient(ux, dx, axis=0) + np.gradient(uy, dx, axis=1) + np.gradient(uz, dx, axis=2)
        
        # Extract slice based on the plane
        if plane == 'XY':
            div_slice = div_u[:, :, index]
            tumor_slice = tumor_volume[:, :, index]
            x_label, y_label = 'X', 'Y'
        elif plane == 'XZ':
            div_slice = div_u[:, index, :]
            tumor_slice = tumor_volume[:, index, :]
            x_label, y_label = 'X', 'Z'
        elif plane == 'YZ':
            div_slice = div_u[index, :, :]
            tumor_slice = tumor_volume[index, :, :]
            x_label, y_label = 'Y', 'Z'
        
        # Apply smoothing for better visualization
        div_smooth = gaussian_filter(div_slice, sigma=1.0)
        
        # Create the colormap with a centered diverging scale
        max_abs = max(abs(np.min(div_smooth)), abs(np.max(div_smooth)))
        vmin, vmax = -max_abs, max_abs
        
        # Plot the divergence
        im = ax.imshow(div_smooth, origin='lower', cmap='coolwarm', vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label='Velocity Divergence')
        
        # Overlay tumor outline if requested
        if show_tumor_outline:
            ax.contour(tumor_slice, levels=[threshold], colors='black', linewidths=1.5)
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title('Velocity Divergence')
        
        # Add annotations to indicate areas of cell proliferation/death
        max_idx = np.unravel_index(np.argmax(div_smooth), div_smooth.shape)
        min_idx = np.unravel_index(np.argmin(div_smooth), div_smooth.shape)
        
        if div_smooth[max_idx] > 0:
            ax.annotate('Cell Proliferation', xy=(max_idx[1], max_idx[0]), 
                       xytext=(max_idx[1] + 10, max_idx[0] + 10),
                       arrowprops=dict(facecolor='black', shrink=0.05, width=1),
                       fontsize=8)
        
        if div_smooth[min_idx] < 0:
            ax.annotate('Cell Death/Compression', xy=(min_idx[1], min_idx[0]), 
                       xytext=(min_idx[1] - 10, min_idx[0] - 10),
                       arrowprops=dict(facecolor='black', shrink=0.05, width=1),
                       fontsize=8)
    
    def _plot_velocity_profiles(self, u_comp, v_comp, tumor_slice, ax, x_label, y_label):
        """Helper method to plot velocity profiles across tumor center lines."""
        # Get tumor center
        tumor_y, tumor_x = np.unravel_index(np.argmax(tumor_slice), tumor_slice.shape)
        
        # Calculate the velocity magnitude
        velocity_magnitude = np.sqrt(u_comp**2 + v_comp**2)
        
        # Extract horizontal and vertical profiles through tumor center
        x_profile = velocity_magnitude[tumor_y, :]
        y_profile = velocity_magnitude[:, tumor_x]
        
        # Create distance arrays centered at tumor
        x_dist = np.arange(len(x_profile)) - tumor_x
        y_dist = np.arange(len(y_profile)) - tumor_y
        
        # Plot horizontal profile
        ax.plot(x_dist, x_profile, 'b-', label=f'Horizontal ({y_label}={tumor_y})')
        
        # Plot vertical profile
        ax.plot(y_dist, y_profile, 'r-', label=f'Vertical ({x_label}={tumor_x})')
        
        # Plot tumor region
        tumor_threshold = 0.1
        x_tumor = tumor_slice[tumor_y, :] > tumor_threshold
        y_tumor = tumor_slice[:, tumor_x] > tumor_threshold
        
        # Highlight the tumor regions
        ax.fill_between(x_dist, 0, x_profile, where=x_tumor, color='blue', alpha=0.2)
        ax.fill_between(y_dist, 0, y_profile, where=y_tumor, color='red', alpha=0.2)
        
        # Add grid, labels, and legend
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlabel(f'Distance from center (grid points)')
        ax.set_ylabel('Velocity magnitude')
        ax.set_title('Velocity Profiles Through Tumor Center')
        ax.legend()
        
        # Add annotations for peak velocities
        x_max_idx = np.argmax(x_profile)
        y_max_idx = np.argmax(y_profile)
        
        ax.annotate(f'Max: {x_profile[x_max_idx]:.3f}', 
                   xy=(x_dist[x_max_idx], x_profile[x_max_idx]),
                   xytext=(x_dist[x_max_idx]+5, x_profile[x_max_idx]),
                   arrowprops=dict(facecolor='black', arrowstyle='->'),
                   fontsize=8)
        
        ax.annotate(f'Max: {y_profile[y_max_idx]:.3f}', 
                   xy=(y_dist[y_max_idx], y_profile[y_max_idx]),
                   xytext=(y_dist[y_max_idx]-5, y_profile[y_max_idx]),
                   arrowprops=dict(facecolor='black', arrowstyle='->'),
                   fontsize=8)
    
    def analyze_velocity_components(self, model, step):
        """
        Analyze the components of the velocity field to understand driving forces.
        
        Args:
            model: The simulation model object
            step (int): The time step to analyze
        """
        # Get the cell volume fractions and nutrient for the specified step
        phi_H = self.history['healthy cell volume fraction'][step]
        phi_D = self.history['diseased cell volume fraction'][step]
        phi_N = self.history['necrotic cell volume fraction'][step]
        nutrient = self.history['nutrient concentration'][step]
        
        # Calculate total tumor volume fraction
        phi_T = phi_H + phi_D + phi_N
        
        # Calculate pressure field
        pressure = compute_internal_pressure_scie3121_model(
            phi_H, phi_D, phi_N, nutrient, model.dx, 
            model.params['gamma'], model.params['epsilon'],
            model.params['lambda_H'], model.params['lambda_D'], 
            model.params['mu_H'], model.params['mu_D'], model.params['mu_N'],
            model.params['p_H'], model.params['p_D'], model.params['n_H'], model.params['n_D']
        )
        
        # Calculate pressure gradient
        grad_p_x = np.gradient(pressure, model.dx, axis=0)
        grad_p_y = np.gradient(pressure, model.dx, axis=1)
        grad_p_z = np.gradient(pressure, model.dx, axis=2)
        
        # Calculate adhesion energy derivative
        energy_deriv = compute_adhesion_energy_derivative(phi_T, model.dx, 
                                                         model.params['gamma'], 
                                                         model.params['epsilon'])
        
        # Calculate tumor gradient
        grad_C_x = np.gradient(phi_T, axis=0) / model.dx
        grad_C_y = np.gradient(phi_T, axis=1) / model.dx
        grad_C_z = np.gradient(phi_T, axis=2) / model.dx
        
        # Calculate velocity components
        ux_pressure = -grad_p_x
        uy_pressure = -grad_p_y
        uz_pressure = -grad_p_z
        
        ux_adhesion = -energy_deriv * grad_C_x
        uy_adhesion = -energy_deriv * grad_C_y
        uz_adhesion = -energy_deriv * grad_C_z
        
        # Total velocity
        ux_total = ux_pressure + ux_adhesion
        uy_total = uy_pressure + uy_adhesion
        uz_total = uz_pressure + uz_adhesion
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Get middle slice for visualization
        slice_idx = phi_T.shape[2] // 2
        
        # Plot pressure component
        vmag_pressure = np.sqrt(ux_pressure[:,:,slice_idx]**2 + uy_pressure[:,:,slice_idx]**2)
        axes[0, 0].imshow(vmag_pressure, origin='lower', cmap='viridis')
        axes[0, 0].set_title('Pressure-Driven Velocity Magnitude')
        
        # Plot adhesion component
        vmag_adhesion = np.sqrt(ux_adhesion[:,:,slice_idx]**2 + uy_adhesion[:,:,slice_idx]**2)
        axes[0, 1].imshow(vmag_adhesion, origin='lower', cmap='viridis')
        axes[0, 1].set_title('Adhesion-Driven Velocity Magnitude')
        
        # Plot total velocity
        vmag_total = np.sqrt(ux_total[:,:,slice_idx]**2 + uy_total[:,:,slice_idx]**2)
        axes[0, 2].imshow(vmag_total, origin='lower', cmap='viridis')
        axes[0, 2].set_title('Total Velocity Magnitude')
        
        # Plot pressure field
        axes[1, 0].imshow(pressure[:,:,slice_idx], origin='lower', cmap='coolwarm')
        axes[1, 0].set_title('Pressure Field')
        
        # Plot adhesion energy derivative
        axes[1, 1].imshow(energy_deriv[:,:,slice_idx], origin='lower', cmap='coolwarm')
        axes[1, 1].set_title('Adhesion Energy Derivative')
        
        # Plot tumor volume fraction
        axes[1, 2].imshow(phi_T[:,:,slice_idx], origin='lower', cmap='gray')
        axes[1, 2].set_title('Tumor Volume Fraction')
        
        plt.tight_layout()
        plt.show()
        
        return {
            'pressure_component': (ux_pressure, uy_pressure, uz_pressure),
            'adhesion_component': (ux_adhesion, uy_adhesion, uz_adhesion),
            'total_velocity': (ux_total, uy_total, uz_total)
        }

def main():
    # Example usage
    model = SCIE3121_MODEL(
        grid_shape=(50, 50, 50),
        dx=0.2,
        dt=0.001,
        params=SCIE3121_params,
        initial_conditions=SphericalTumor(grid_shape=(50, 50, 50), radius=7, nutrient_value=1.0),
    )
    
    analyzer = SolidVelocityAnalyzer(filepath='data/project_model_test_sim_data.npz')
    analyzer.visualize_velocity_field(model, step=0, mode='all', plane='XY')
    analyzer.analyze_velocity_components(model, step=0)
    
if __name__ == "__main__":
    main()