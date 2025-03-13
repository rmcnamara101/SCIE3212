import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes

from analysis.scie3121_analysis import scie3121SimulationAnalyzer
from src.models.cell_dynamics import compute_adhesion_energy_derivative
from src.utils.utils import SCIE3121_params

class MassFluxAnalyzer(scie3121SimulationAnalyzer):
    def __init__(self, filepath):
        super().__init__(filepath)
    
    def compute_mass_flux(self, phi_i, phi_T, dx, gamma, epsilon, mobility):
        """
        Compute the mass flux for a cell population.
        J_i = -M_i ∇(δE/δφ_i)
        
        Args:
            phi_i: Volume fraction of the specific cell population
            phi_T: Total cell volume fraction
            dx: Grid spacing
            gamma: Cell-cell adhesion parameter
            epsilon: Interface thickness parameter
            mobility: Cell mobility parameter
            
        Returns:
            Tuple of (flux_x, flux_y, flux_z, flux_magnitude)
        """
        # Compute variational derivative of adhesion energy
        energy_deriv = compute_adhesion_energy_derivative(phi_T, dx, gamma, epsilon)
        
        # Compute gradient of energy derivative
        grad_x = np.gradient(energy_deriv, dx, axis=0)
        grad_y = np.gradient(energy_deriv, dx, axis=1)
        grad_z = np.gradient(energy_deriv, dx, axis=2)
        
        # Compute mass flux J_i = -M_i ∇(δE/δφ_i)
        flux_x = -mobility * grad_x
        flux_y = -mobility * grad_y
        flux_z = -mobility * grad_z
        
        # Compute flux magnitude
        flux_magnitude = np.sqrt(flux_x**2 + flux_y**2 + flux_z**2)
        
        return flux_x, flux_y, flux_z, flux_magnitude
    
    def visualize_mass_flux(self, model, step, cell_type='all', mode='all', plane='XY', 
                          index=None, cmap='viridis', threshold=0.1):
        """
        Visualize mass flux for different cell populations.
        
        Args:
            model: The simulation model object
            step (int): The time step to visualize
            cell_type (str): Cell type to visualize - 'healthy', 'diseased', 'necrotic', or 'all'
            mode (str): Visualization mode - 'all', 'slices', 'vectors', 'magnitude', '3d'
            plane (str): Plane to slice ('XY', 'XZ', 'YZ')
            index (int, optional): Index along the perpendicular axis. Defaults to midpoint.
            cmap (str): Colormap for visualization
            threshold (float): Threshold for tumor outline
        """
        # Get the volume fractions for the specified step
        try:
            phi_H = self.history['healthy cell volume fraction'][step]
            phi_D = self.history['diseased cell volume fraction'][step]
            phi_N = self.history['necrotic cell volume fraction'][step]
        except (KeyError, IndexError) as e:
            print(f"Error accessing history data at step {step}: {e}")
            return
        
        # Calculate total tumor volume fraction
        phi_T = phi_H + phi_D + phi_N
        
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
            phi_H_slice = phi_H[:, :, index]
            phi_D_slice = phi_D[:, :, index]
            phi_N_slice = phi_N[:, :, index]
            phi_T_slice = phi_T[:, :, index]
            x_label, y_label = 'X', 'Y'
        elif plane == 'XZ':
            phi_H_slice = phi_H[:, index, :]
            phi_D_slice = phi_D[:, index, :]
            phi_N_slice = phi_N[:, index, :]
            phi_T_slice = phi_T[:, index, :]
            x_label, y_label = 'X', 'Z'
        elif plane == 'YZ':
            phi_H_slice = phi_H[index, :, :]
            phi_D_slice = phi_D[index, :, :]
            phi_N_slice = phi_N[index, :, :]
            phi_T_slice = phi_T[index, :, :]
            x_label, y_label = 'Y', 'Z'
        
        # Compute mass flux for each cell population
        params = model.params
        
        # Healthy cells
        flux_H_x, flux_H_y, flux_H_z, flux_H_mag = self.compute_mass_flux(
            phi_H, phi_T, model.dx, params['gamma'], params['epsilon'], params['M']
        )
        
        # Diseased cells
        flux_D_x, flux_D_y, flux_D_z, flux_D_mag = self.compute_mass_flux(
            phi_D, phi_T, model.dx, params['gamma'], params['epsilon'], params['M']
        )
        
        # Necrotic cells (with very low mobility)
        flux_N_x, flux_N_y, flux_N_z, flux_N_mag = self.compute_mass_flux(
            phi_N, phi_T, model.dx, params['gamma'], params['epsilon'], 1e-6
        )
        
        # Extract flux slices
        if plane == 'XY':
            flux_H_x_slice = flux_H_x[:, :, index]
            flux_H_y_slice = flux_H_y[:, :, index]
            flux_H_mag_slice = flux_H_mag[:, :, index]
            
            flux_D_x_slice = flux_D_x[:, :, index]
            flux_D_y_slice = flux_D_y[:, :, index]
            flux_D_mag_slice = flux_D_mag[:, :, index]
            
            flux_N_x_slice = flux_N_x[:, :, index]
            flux_N_y_slice = flux_N_y[:, :, index]
            flux_N_mag_slice = flux_N_mag[:, :, index]
        elif plane == 'XZ':
            flux_H_x_slice = flux_H_x[:, index, :]
            flux_H_y_slice = flux_H_z[:, index, :]  # z instead of y for XZ plane
            flux_H_mag_slice = flux_H_mag[:, index, :]
            
            flux_D_x_slice = flux_D_x[:, index, :]
            flux_D_y_slice = flux_D_z[:, index, :]
            flux_D_mag_slice = flux_D_mag[:, index, :]
            
            flux_N_x_slice = flux_N_x[:, index, :]
            flux_N_y_slice = flux_N_z[:, index, :]
            flux_N_mag_slice = flux_N_mag[:, index, :]
        elif plane == 'YZ':
            flux_H_x_slice = flux_H_y[index, :, :]  # y instead of x for YZ plane
            flux_H_y_slice = flux_H_z[index, :, :]
            flux_H_mag_slice = flux_H_mag[index, :, :]
            
            flux_D_x_slice = flux_D_y[index, :, :]
            flux_D_y_slice = flux_D_z[index, :, :]
            flux_D_mag_slice = flux_D_mag[index, :, :]
            
            flux_N_x_slice = flux_N_y[index, :, :]
            flux_N_y_slice = flux_N_z[index, :, :]
            flux_N_mag_slice = flux_N_mag[index, :, :]
        
        # Determine which cell types to plot
        if cell_type == 'all':
            plot_healthy = plot_diseased = plot_necrotic = True
        else:
            plot_healthy = cell_type == 'healthy'
            plot_diseased = cell_type == 'diseased'
            plot_necrotic = cell_type == 'necrotic'
        
        # Visualize based on mode
        if mode == 'all':
            if cell_type == 'all':
                # Create 3x3 grid (3 cell types x 3 visualization types)
                fig, axes = plt.subplots(3, 3, figsize=(18, 16))
                
                # Row 1: Healthy cells
                self._plot_cell_distribution(phi_H_slice, axes[0, 0], x_label, y_label, 
                                        'Healthy Cell Distribution', cmap='Blues')
                self._plot_flux_magnitude(flux_H_mag_slice, phi_H_slice, axes[0, 1], x_label, y_label, 
                                       'Healthy Cell Flux Magnitude', cmap='viridis', 
                                       show_contour=True, threshold=threshold)
                self._plot_flux_vectors(flux_H_x_slice, flux_H_y_slice, phi_H_slice, axes[0, 2], 
                                      x_label, y_label, 'Healthy Cell Flux Vectors', 
                                      show_contour=True, threshold=threshold)
                
                # Row 2: Diseased cells
                self._plot_cell_distribution(phi_D_slice, axes[1, 0], x_label, y_label, 
                                        'Diseased Cell Distribution', cmap='Reds')
                self._plot_flux_magnitude(flux_D_mag_slice, phi_D_slice, axes[1, 1], x_label, y_label, 
                                       'Diseased Cell Flux Magnitude', cmap='viridis', 
                                       show_contour=True, threshold=threshold)
                self._plot_flux_vectors(flux_D_x_slice, flux_D_y_slice, phi_D_slice, axes[1, 2], 
                                      x_label, y_label, 'Diseased Cell Flux Vectors', 
                                      show_contour=True, threshold=threshold)
                
                # Row 3: Necrotic cells
                self._plot_cell_distribution(phi_N_slice, axes[2, 0], x_label, y_label, 
                                        'Necrotic Cell Distribution', cmap='Greys')
                self._plot_flux_magnitude(flux_N_mag_slice, phi_N_slice, axes[2, 1], x_label, y_label, 
                                       'Necrotic Cell Flux Magnitude', cmap='viridis', 
                                       show_contour=True, threshold=threshold)
                self._plot_flux_vectors(flux_N_x_slice, flux_N_y_slice, phi_N_slice, axes[2, 2], 
                                      x_label, y_label, 'Necrotic Cell Flux Vectors', 
                                      show_contour=True, threshold=threshold)
            else:
                # Create 1x3 grid for a single cell type
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                if plot_healthy:
                    cell_name = 'Healthy'
                    phi_slice = phi_H_slice
                    flux_x_slice = flux_H_x_slice
                    flux_y_slice = flux_H_y_slice
                    flux_mag_slice = flux_H_mag_slice
                    cell_cmap = 'Blues'
                elif plot_diseased:
                    cell_name = 'Diseased'
                    phi_slice = phi_D_slice
                    flux_x_slice = flux_D_x_slice
                    flux_y_slice = flux_D_y_slice
                    flux_mag_slice = flux_D_mag_slice
                    cell_cmap = 'Reds'
                elif plot_necrotic:
                    cell_name = 'Necrotic'
                    phi_slice = phi_N_slice
                    flux_x_slice = flux_N_x_slice
                    flux_y_slice = flux_N_y_slice
                    flux_mag_slice = flux_N_mag_slice
                    cell_cmap = 'Greys'
                
                self._plot_cell_distribution(phi_slice, axes[0], x_label, y_label, 
                                        f'{cell_name} Cell Distribution', cmap=cell_cmap)
                self._plot_flux_magnitude(flux_mag_slice, phi_slice, axes[1], x_label, y_label, 
                                       f'{cell_name} Cell Flux Magnitude', cmap='viridis', 
                                       show_contour=True, threshold=threshold)
                self._plot_flux_vectors(flux_x_slice, flux_y_slice, phi_slice, axes[2], 
                                      x_label, y_label, f'{cell_name} Cell Flux Vectors', 
                                      show_contour=True, threshold=threshold)
            
            plt.tight_layout()
            plt.suptitle(f'Mass Flux Analysis - Step {step} - {plane} Plane (Index {index})', 
                        fontsize=16, y=1.02)
        
        elif mode == 'slices':
            # Create a figure with cell distributions only
            if cell_type == 'all':
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                self._plot_cell_distribution(phi_H_slice, axes[0], x_label, y_label, 
                                        'Healthy Cell Distribution', cmap='Blues')
                self._plot_cell_distribution(phi_D_slice, axes[1], x_label, y_label, 
                                        'Diseased Cell Distribution', cmap='Reds')
                self._plot_cell_distribution(phi_N_slice, axes[2], x_label, y_label, 
                                        'Necrotic Cell Distribution', cmap='Greys')
            else:
                fig, ax = plt.subplots(figsize=(10, 8))
                if plot_healthy:
                    self._plot_cell_distribution(phi_H_slice, ax, x_label, y_label, 
                                            'Healthy Cell Distribution', cmap='Blues')
                elif plot_diseased:
                    self._plot_cell_distribution(phi_D_slice, ax, x_label, y_label, 
                                            'Diseased Cell Distribution', cmap='Reds')
                elif plot_necrotic:
                    self._plot_cell_distribution(phi_N_slice, ax, x_label, y_label, 
                                            'Necrotic Cell Distribution', cmap='Greys')
            
            plt.tight_layout()
            plt.suptitle(f'Cell Distribution - Step {step} - {plane} Plane (Index {index})', 
                        fontsize=16, y=1.02)
        
        elif mode == 'magnitude':
            # Create a figure with flux magnitudes
            if cell_type == 'all':
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                self._plot_flux_magnitude(flux_H_mag_slice, phi_H_slice, axes[0], x_label, y_label, 
                                       'Healthy Cell Flux Magnitude', cmap='viridis', 
                                       show_contour=True, threshold=threshold)
                self._plot_flux_magnitude(flux_D_mag_slice, phi_D_slice, axes[1], x_label, y_label, 
                                       'Diseased Cell Flux Magnitude', cmap='viridis', 
                                       show_contour=True, threshold=threshold)
                self._plot_flux_magnitude(flux_N_mag_slice, phi_N_slice, axes[2], x_label, y_label, 
                                       'Necrotic Cell Flux Magnitude', cmap='viridis', 
                                       show_contour=True, threshold=threshold)
            else:
                fig, ax = plt.subplots(figsize=(10, 8))
                if plot_healthy:
                    self._plot_flux_magnitude(flux_H_mag_slice, phi_H_slice, ax, x_label, y_label, 
                                           'Healthy Cell Flux Magnitude', cmap='viridis', 
                                           show_contour=True, threshold=threshold)
                elif plot_diseased:
                    self._plot_flux_magnitude(flux_D_mag_slice, phi_D_slice, ax, x_label, y_label, 
                                           'Diseased Cell Flux Magnitude', cmap='viridis', 
                                           show_contour=True, threshold=threshold)
                elif plot_necrotic:
                    self._plot_flux_magnitude(flux_N_mag_slice, phi_N_slice, ax, x_label, y_label, 
                                           'Necrotic Cell Flux Magnitude', cmap='viridis', 
                                           show_contour=True, threshold=threshold)
            
            plt.tight_layout()
            plt.suptitle(f'Mass Flux Magnitude - Step {step} - {plane} Plane (Index {index})', 
                        fontsize=16, y=1.02)
        
        elif mode == 'vectors':
            # Create a figure with flux vectors
            if cell_type == 'all':
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                self._plot_flux_vectors(flux_H_x_slice, flux_H_y_slice, phi_H_slice, axes[0], 
                                      x_label, y_label, 'Healthy Cell Flux Vectors', 
                                      show_contour=True, threshold=threshold)
                self._plot_flux_vectors(flux_D_x_slice, flux_D_y_slice, phi_D_slice, axes[1], 
                                      x_label, y_label, 'Diseased Cell Flux Vectors', 
                                      show_contour=True, threshold=threshold)
                self._plot_flux_vectors(flux_N_x_slice, flux_N_y_slice, phi_N_slice, axes[2], 
                                      x_label, y_label, 'Necrotic Cell Flux Vectors', 
                                      show_contour=True, threshold=threshold)
            else:
                fig, ax = plt.subplots(figsize=(10, 8))
                if plot_healthy:
                    self._plot_flux_vectors(flux_H_x_slice, flux_H_y_slice, phi_H_slice, ax, 
                                          x_label, y_label, 'Healthy Cell Flux Vectors', 
                                          show_contour=True, threshold=threshold)
                elif plot_diseased:
                    self._plot_flux_vectors(flux_D_x_slice, flux_D_y_slice, phi_D_slice, ax, 
                                          x_label, y_label, 'Diseased Cell Flux Vectors', 
                                          show_contour=True, threshold=threshold)
                elif plot_necrotic:
                    self._plot_flux_vectors(flux_N_x_slice, flux_N_y_slice, phi_N_slice, ax, 
                                          x_label, y_label, 'Necrotic Cell Flux Vectors', 
                                          show_contour=True, threshold=threshold)
            
            plt.tight_layout()
            plt.suptitle(f'Mass Flux Vectors - Step {step} - {plane} Plane (Index {index})', 
                        fontsize=16, y=1.02)
        
        elif mode == '3d':
            # Create a 3D visualization for the selected cell type
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            if cell_type == 'all':
                # If all cells, just show total tumor for simplicity in 3D
                self._plot_flux_3d(flux_H_mag, flux_D_mag, flux_N_mag, phi_T, ax, threshold)
            else:
                if plot_healthy:
                    self._plot_flux_3d_single(flux_H_mag, phi_H, ax, threshold, 'Blues')
                elif plot_diseased:
                    self._plot_flux_3d_single(flux_D_mag, phi_D, ax, threshold, 'Reds')
                elif plot_necrotic:
                    self._plot_flux_3d_single(flux_N_mag, phi_N, ax, threshold, 'Greys')
            
            plt.tight_layout()
            plt.suptitle(f'3D Mass Flux - Step {step}', fontsize=16, y=1.02)
        
        plt.show()
    
    def _plot_cell_distribution(self, phi_slice, ax, x_label, y_label, title, cmap='viridis'):
        """Helper method to plot cell volume fraction distribution."""
        # Apply mild smoothing for visualization
        phi_smooth = gaussian_filter(phi_slice, sigma=1.0)
        
        im = ax.imshow(phi_smooth, origin='lower', cmap=cmap)
        plt.colorbar(im, ax=ax, label='Volume Fraction')
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
    
    def _plot_flux_magnitude(self, flux_mag_slice, phi_slice, ax, x_label, y_label, 
                           title, cmap='viridis', show_contour=True, threshold=0.1):
        """Helper method to plot flux magnitude with optional cell contour."""
        # Apply mild smoothing for visualization
        flux_smooth = gaussian_filter(flux_mag_slice, sigma=1.0)
        
        # Mask regions where cells are not present
        mask = phi_slice < 0.01
        flux_masked = np.ma.array(flux_smooth, mask=mask)
        
        im = ax.imshow(flux_masked, origin='lower', cmap=cmap)
        plt.colorbar(im, ax=ax, label='Flux Magnitude')
        
        # Add contour if requested
        if show_contour:
            ax.contour(phi_slice, levels=[threshold], colors='white', linewidths=1.5)
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
    
    def _plot_flux_vectors(self, flux_x_slice, flux_y_slice, phi_slice, ax, 
                         x_label, y_label, title, show_contour=True, threshold=0.1):
        """Helper method to plot flux vector field with optional cell contour."""
        # Apply smoothing for clearer vector field
        flux_x_smooth = gaussian_filter(flux_x_slice, sigma=1.0)
        flux_y_smooth = gaussian_filter(flux_y_slice, sigma=1.0)
        
        # Create a grid for the quiver plot
        ny, nx = flux_x_smooth.shape
        y = np.arange(0, ny)
        x = np.arange(0, nx)
        X, Y = np.meshgrid(x, y)
        
        # Downsample for clearer vector field
        downsample = max(1, min(nx, ny) // 25)
        
        # Fix the indexing for proper vector orientation
        X_ds = X[::downsample, ::downsample]
        Y_ds = Y[::downsample, ::downsample]
        
        # Swap and transpose components to match coordinate system
        U_ds = flux_y_smooth[::downsample, ::downsample]  # Swap x and y
        V_ds = flux_x_smooth[::downsample, ::downsample]  # to match orientation
        
        # Mask regions where cells are not present
        magnitude = np.sqrt(U_ds**2 + V_ds**2)
        
        # Plot background tumor slice for context
        ax.imshow(phi_slice, origin='lower', cmap='gray', alpha=0.3)
        
        # Determine maximum magnitude for scaling
        max_mag = np.max(magnitude) if np.max(magnitude) > 0 else 1.0
        
        # Make arrows small for clarity
        
        
        # Plot the vector field
        quiver = ax.quiver(X_ds, Y_ds, U_ds, V_ds, magnitude, 
                         cmap='viridis', 
                         scale=0.002,
                         scale_units='xy',
                         width=0.008,
                         headwidth=3,
                         headlength=4,
                         pivot='mid',
                         alpha=0.8)
        
        ax.quiverkey(quiver, 0.9, 0.95, max_mag, f"{max_mag:.2e}", labelpos='E', 
                    coordinates='figure', color='black')
        
        plt.colorbar(quiver, ax=ax, label='Flux Magnitude')
        
        # Add contour if requested
        if show_contour:
            ax.contour(phi_slice, levels=[threshold], colors='white', linewidths=1.5)
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        
        # Add axis limits to match the image dimensions
        ax.set_xlim(0, nx-1)
        ax.set_ylim(0, ny-1)
    
    def _plot_flux_3d(self, flux_H_mag, flux_D_mag, flux_N_mag, phi_T, ax, threshold):
        """Helper method to plot 3D flux isosurfaces for all cell types."""
        # Find reasonable isosurface levels for each flux type
        for flux_mag, label, color in [
            (flux_H_mag, 'Healthy', 'blue'),
            (flux_D_mag, 'Diseased', 'red'),
            (flux_N_mag, 'Necrotic', 'gray')
        ]:
            v_min, v_max = np.min(flux_mag), np.max(flux_mag)
            if v_min == v_max:
                continue
                
            level = v_min + 0.5 * (v_max - v_min)
            
            # Plot flux isosurface
            if np.any(flux_mag > level):
                try:
                    verts, faces, _, _ = marching_cubes(flux_mag, level=level)
                    surf = ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], 
                                         color=color, alpha=0.3, label=f'{label} Flux')
                except:
                    pass  # Skip if marching cubes fails
        
        # Plot tumor outline
        if np.any(phi_T > threshold):
            try:
                verts, faces, _, _ = marching_cubes(phi_T, level=threshold)
                ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], 
                               color='black', alpha=0.1, label='Tumor Outline')
            except:
                pass
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Mass Flux Isosurfaces')
        ax.legend()
    
    def _plot_flux_3d_single(self, flux_mag, phi, ax, threshold, cmap):
        """Helper method to plot 3D flux isosurfaces for a single cell type."""
        # Find reasonable isosurface levels
        v_min, v_max = np.min(flux_mag), np.max(flux_mag)
        if v_min == v_max:
            print("Warning: Flux field is uniform, no isosurfaces to display")
            return
            
        levels = np.linspace(v_min + 0.2 * (v_max - v_min), v_max * 0.8, 3)
        
        # Get colormap
        colormap = plt.cm.get_cmap(cmap)
        
        # Plot flux isosurfaces
        for i, level in enumerate(levels):
            if np.any(flux_mag > level):
                try:
                    verts, faces, _, _ = marching_cubes(flux_mag, level=level)
                    color = colormap((level - v_min) / (v_max - v_min))
                    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], 
                                   color=color, alpha=0.4, label=f'Flux Level {i+1}')
                except:
                    pass  # Skip if marching cubes fails
        
        # Plot cell population outline
        if np.any(phi > threshold):
            try:
                verts, faces, _, _ = marching_cubes(phi, level=threshold)
                ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], 
                               color='black', alpha=0.1, label='Cell Outline')
            except:
                pass
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Mass Flux Isosurfaces')
        ax.legend()

def main():
    """Test function to demonstrate mass flux analysis."""
    from src.models.SCIE3121_model import SCIE3121_MODEL
    from src.models.initial_conditions import SphericalTumor
    
    # Create model instance
    model = SCIE3121_MODEL(
        grid_shape=(50, 50, 50),
        dx=0.2,
        dt=0.001,
        params=SCIE3121_params,
        initial_conditions=SphericalTumor(grid_shape=(50, 50, 50), radius=7, nutrient_value=1.0),
    )
    
    # Create analyzer instance
    analyzer = MassFluxAnalyzer(filepath='data/project_model_test_sim_data.npz')
    
    # Run visualization
    analyzer.visualize_mass_flux(
        model, 
        step=0,  # Analyze first time step
        cell_type='all',
        mode='all',
        plane='XY',
        threshold=0.1
    )

if __name__ == "__main__":
    main()
