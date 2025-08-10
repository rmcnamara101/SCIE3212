import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import os
from pathlib import Path

class PhysicsFieldsPlotter:
    def __init__(self, field_manager):
        """
        Initialize the physics fields plotter.
        
        Args:
            field_manager: FieldManager instance containing the simulation data
        """
        self.field_manager = field_manager
        self.grid = field_manager.grid
        self.dx = field_manager.dx
        self.labels = field_manager.labels
        self.M = field_manager.M
        
    def _add_tumor_contours(self, ax, population_idx=0, z_slice=None, 
                           contour_levels=[0.1, 0.3, 0.5], 
                           contour_colors=['red', 'orange', 'yellow'],
                           contour_linewidths=[2, 1.5, 1]):
        """
        Add tumor boundary contours to an existing plot.
        
        Args:
            ax: Matplotlib axis to add contours to
            population_idx: Index of population to contour
            z_slice: Z-slice to plot (defaults to center)
            contour_levels: List of density levels for contours
            contour_colors: List of colors for each contour level
            contour_linewidths: List of line widths for each contour level
        """
        if population_idx >= self.M:
            return
            
        nx, ny, nz = self.grid
        if z_slice is None:
            z_slice = nz // 2
            
        # Create coordinate grids
        x = np.arange(nx) * self.dx
        y = np.arange(ny) * self.dx
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Get tumor density
        tumor_density = self.field_manager.phi_hat[population_idx, :, :, z_slice]
        
        # Add contours
        for i, level in enumerate(contour_levels):
            color = contour_colors[i] if i < len(contour_colors) else 'black'
            linewidth = contour_linewidths[i] if i < len(contour_linewidths) else 1
            ax.contour(X, Y, tumor_density, levels=[level], colors=color, 
                      linewidths=linewidth, alpha=0.8)
        
    def plot_pressure_field(self, step=0, z_slice=None, cmap="viridis", alpha=0.8, 
                           output_dir=None, save_plot=False, show_plot=True,
                           add_boundary_contours=False, tumor_boundary_level=0.5,
                           boundary_color='darkblue', tumor_boundary_color='black'):
        """
        Plot pressure field as 3D surface plot.
        
        Args:
            step: Current simulation step for filename
            z_slice: Z-slice to plot (defaults to center)
            cmap: Colormap for the plot
            alpha: Transparency of the surface
            output_dir: Directory to save plot
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            add_boundary_contours: Whether to add tumor boundary contours
            tumor_boundary_level: Density level for tumor boundary contour (default 0.5)
            boundary_color: Color for boundary region contour (default 'darkblue')
            tumor_boundary_color: Color for tumor boundary contour (default 'black')
        """
        nx, ny, nz = self.grid
        if z_slice is None:
            z_slice = nz // 2
        
        # Create coordinate grids
        x = np.arange(nx) * self.dx
        y = np.arange(ny) * self.dx
        X, Y = np.meshgrid(x, y, indexing='ij')
        pressure_slice = self.field_manager.pressure[:, :, z_slice]
        
        # Create plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(X, Y, pressure_slice, cmap=cmap, alpha=alpha)
        ax.set_title(f'Pressure Field (z={z_slice})')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Pressure')
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        plt.tight_layout()
        
        if save_plot and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f'pressure_field_step_{step:06d}.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            print(f"Pressure field plot saved to {output_dir}/{filename}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()

    def plot_velocity_field(self, step=0, z_slice=None, skip=2, cmap="viridis", 
                           scale=50, width=0.005, output_dir=None, 
                           save_plot=False, show_plot=True, add_boundary_contours=False,
                           tumor_boundary_level=0.5, boundary_color='darkblue', 
                           tumor_boundary_color='black', center_x=None, center_y=None, 
                           zoom_factor=1.0):
        """
        Plot velocity field as 2D vector field.
        
        Args:
            step: Current simulation step for filename
            z_slice: Z-slice to plot (defaults to center)
            skip: Sampling rate for vectors (every nth point)
            cmap: Colormap for the plot
            scale: Scale factor for vectors
            width: Width of vectors
            output_dir: Directory to save plot
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            add_boundary_contours: Whether to add tumor boundary contours
            tumor_boundary_level: Density level for tumor boundary contour (default 0.5)
            boundary_color: Color for boundary region contour (default 'darkblue')
            tumor_boundary_color: Color for tumor boundary contour (default 'black')
        """
        nx, ny, nz = self.grid
        if z_slice is None:
            z_slice = nz // 2
        
        # Set default center coordinates if not provided
        if center_x is None:
            center_x = nx // 2
        if center_y is None:
            center_y = ny // 2
        
        # Convert to integers for slice indices
        center_x = int(center_x)
        center_y = int(center_y)
        
        # Calculate zoom window size
        window_size = min(nx, ny) // zoom_factor
        half_window = window_size // 2
        
        # Calculate window boundaries
        x_start = int(max(0, center_x - half_window))
        x_end = int(min(nx, center_x + half_window))
        y_start = int(max(0, center_y - half_window))
        y_end = int(min(ny, center_y + half_window))
        
        # Create coordinate grids for zoomed region
        x_region = np.arange(x_start, x_end) * self.dx
        y_region = np.arange(y_start, y_end) * self.dx
        X_region, Y_region = np.meshgrid(x_region, y_region, indexing='ij')
        
        # Extract velocity components for zoomed region
        ux_slice = self.field_manager.velocity[0, x_start:x_end, y_start:y_end, z_slice]
        uy_slice = self.field_manager.velocity[1, x_start:x_end, y_start:y_end, z_slice]
        
        # Sample for visualization
        x_skip = x_region[::skip]
        y_skip = y_region[::skip]
        X_skip, Y_skip = np.meshgrid(x_skip, y_skip, indexing='ij')
        ux_skip = ux_slice[::skip, ::skip]
        uy_skip = uy_slice[::skip, ::skip]
        
        # Normalize vectors for better visualization
        magnitude = np.sqrt(ux_skip**2 + uy_skip**2)
        max_mag = np.max(magnitude) if np.max(magnitude) > 0 else 1.0
        ux_norm = ux_skip / max_mag
        uy_norm = uy_skip / max_mag
        
        # Create plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        
        quiv = ax.quiver(X_skip, Y_skip, ux_norm, uy_norm, magnitude, 
                        cmap=cmap, scale=scale, width=width)
        
        # Add tumor boundary contours if requested
        if add_boundary_contours:
            # Get total tumor density for boundary detection (zoomed region)
            tumor_density = np.sum(self.field_manager.phi_hat, axis=0)[x_start:x_end, y_start:y_end, z_slice]
            
            # Add tumor boundary contour
            ax.contour(X_region, Y_region, tumor_density, levels=[tumor_boundary_level], 
                      colors=tumor_boundary_color, linewidths=3, alpha=0.9)
        
        # Add center point marker if zoomed
        if zoom_factor > 1.0:
            center_x_coord = center_x * self.dx
            center_y_coord = center_y * self.dx
            ax.plot(center_x_coord, center_y_coord, 'k+', markersize=10, markeredgewidth=2, 
                   label=f'Center: ({center_x_coord:.2f}, {center_y_coord:.2f})')
            ax.legend()
            ax.set_title(f'Velocity Field (z={z_slice}) - Zoomed View')
        else:
            ax.set_title(f'Velocity Field (z={z_slice})')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')
        fig.colorbar(quiv, ax=ax)
        
        plt.tight_layout()
        
        if save_plot and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f'velocity_field_step_{step:06d}.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            print(f"Velocity field plot saved to {output_dir}/{filename}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()

    def plot_energy_derivative_field(self, step=0, z_slice=None, cmap="RdBu_r",
                                   output_dir=None, save_plot=False, show_plot=True,
                                   add_boundary_contours=False, tumor_boundary_level=0.5,
                                   boundary_color='darkblue', tumor_boundary_color='black',
                                   center_x=None, center_y=None, zoom_factor=1.0):
        """
        Plot energy derivative field as 2D heatmap.
        
        Args:
            step: Current simulation step for filename
            z_slice: Z-slice to plot (defaults to center)
            cmap: Colormap for the plot
            output_dir: Directory to save plot
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            add_boundary_contours: Whether to add tumor boundary contours
            tumor_boundary_level: Density level for tumor boundary contour (default 0.5)
            boundary_color: Color for boundary region contour (default 'darkblue')
            tumor_boundary_color: Color for tumor boundary contour (default 'black')
        """
        nx, ny, nz = self.grid
        if z_slice is None:
            z_slice = nz // 2
        
        # Set default center coordinates if not provided
        if center_x is None:
            center_x = nx // 2
        if center_y is None:
            center_y = ny // 2
        
        # Convert to integers for slice indices
        center_x = int(center_x)
        center_y = int(center_y)
        
        # Calculate zoom window size
        window_size = min(nx, ny) // zoom_factor
        half_window = window_size // 2
        
        # Calculate window boundaries
        x_start = int(max(0, center_x - half_window))
        x_end = int(min(nx, center_x + half_window))
        y_start = int(max(0, center_y - half_window))
        y_end = int(min(ny, center_y + half_window))
        
        # Create coordinate grids for zoomed region
        x_region = np.arange(x_start, x_end) * self.dx
        y_region = np.arange(y_start, y_end) * self.dx
        
        # Extract energy derivative for zoomed region
        energy_slice = self.field_manager.energy_derivative[x_start:x_end, y_start:y_end, z_slice]
        
        # Create plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        
        im = ax.imshow(energy_slice, extent=[x_region[0], x_region[-1], y_region[0], y_region[-1]], 
                      origin='lower', cmap=cmap, aspect='equal')
        
        # Add tumor boundary contours if requested
        if add_boundary_contours:
            # Get total tumor density for boundary detection (zoomed region)
            tumor_density = np.sum(self.field_manager.phi_hat, axis=0)[x_start:x_end, y_start:y_end, z_slice]
            
            # Add tumor boundary contour
            ax.contour(x_region, y_region, tumor_density, levels=[tumor_boundary_level], 
                      colors=tumor_boundary_color, linewidths=3, alpha=0.9)
        
        # Add center point marker if zoomed
        if zoom_factor > 1.0:
            center_x_coord = center_x * self.dx
            center_y_coord = center_y * self.dx
            ax.plot(center_x_coord, center_y_coord, 'k+', markersize=10, markeredgewidth=2, 
                   label=f'Center: ({center_x_coord:.2f}, {center_y_coord:.2f})')
            ax.legend()
            ax.set_title(f'Energy Derivative Field (z={z_slice}) - Zoomed View')
        else:
            ax.set_title(f'Energy Derivative Field (z={z_slice})')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        fig.colorbar(im, ax=ax)
        
        plt.tight_layout()
        
        if save_plot and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f'energy_derivative_field_step_{step:06d}.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            print(f"Energy derivative field plot saved to {output_dir}/{filename}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()

    def plot_mass_flux_field(self, population_idx=0, step=0, z_slice=None, skip=2, 
                           cmap="viridis", scale=20, width=0.005, output_dir=None,
                           save_plot=False, show_plot=True, add_boundary_contours=False,
                           tumor_boundary_level=0.5, boundary_color='darkblue', 
                           tumor_boundary_color='black', center_x=None, center_y=None, 
                           zoom_factor=1.0):
        """
        Plot mass flux field for a specific population as 2D vector field.
        
        Args:
            population_idx: Index of population to plot (default 0)
            step: Current simulation step for filename
            z_slice: Z-slice to plot (defaults to center)
            skip: Sampling rate for vectors (every nth point)
            cmap: Colormap for the plot
            scale: Scale factor for vectors
            width: Width of vectors
            output_dir: Directory to save plot
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
        """
        if population_idx >= self.M:
            print(f"Error: Population index {population_idx} out of range (0-{self.M-1})")
            return
        
        nx, ny, nz = self.grid
        if z_slice is None:
            z_slice = nz // 2
        
        # Set default center coordinates if not provided
        if center_x is None:
            center_x = nx // 2
        if center_y is None:
            center_y = ny // 2
        
        # Convert to integers for slice indices
        center_x = int(center_x)
        center_y = int(center_y)
        
        # Calculate zoom window size
        window_size = min(nx, ny) // zoom_factor
        half_window = window_size // 2
        
        # Calculate window boundaries
        x_start = int(max(0, center_x - half_window))
        x_end = int(min(nx, center_x + half_window))
        y_start = int(max(0, center_y - half_window))
        y_end = int(min(ny, center_y + half_window))
        
        # Create coordinate grids for zoomed region
        x_region = np.arange(x_start, x_end) * self.dx
        y_region = np.arange(y_start, y_end) * self.dx
        X_region, Y_region = np.meshgrid(x_region, y_region, indexing='ij')
        
        # Extract mass flux components for zoomed region
        flux_x_slice = self.field_manager.mass_flux[population_idx, 0, x_start:x_end, y_start:y_end, z_slice]
        flux_y_slice = self.field_manager.mass_flux[population_idx, 1, x_start:x_end, y_start:y_end, z_slice]
        
        # Sample for visualization
        x_skip = x_region[::skip]
        y_skip = y_region[::skip]
        X_skip, Y_skip = np.meshgrid(x_skip, y_skip, indexing='ij')
        flux_x_skip = flux_x_slice[::skip, ::skip]
        flux_y_skip = flux_y_slice[::skip, ::skip]
        
        # Normalize vectors
        flux_magnitude = np.sqrt(flux_x_skip**2 + flux_y_skip**2)
        max_flux_mag = np.max(flux_magnitude) if np.max(flux_magnitude) > 0 else 1.0
        flux_x_norm = flux_x_skip / max_flux_mag
        flux_y_norm = flux_y_skip / max_flux_mag
        
        # Create plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        
        quiv = ax.quiver(X_skip, Y_skip, flux_x_norm, flux_y_norm, flux_magnitude, 
                        cmap=cmap, scale=scale, width=width)
        
        # Add tumor boundary contours if requested
        if add_boundary_contours:
            # Get total tumor density for boundary detection (zoomed region)
            tumor_density = np.sum(self.field_manager.phi_hat, axis=0)[x_start:x_end, y_start:y_end, z_slice]
            
            # Add tumor boundary contour
            ax.contour(x_region, y_region, tumor_density, levels=[tumor_boundary_level], 
                      colors=tumor_boundary_color, linewidths=3, alpha=0.9)
        
        # Add center point marker if zoomed
        if zoom_factor > 1.0:
            center_x_coord = center_x * self.dx
            center_y_coord = center_y * self.dx
            ax.plot(center_x_coord, center_y_coord, 'k+', markersize=10, markeredgewidth=2, 
                   label=f'Center: ({center_x_coord:.2f}, {center_y_coord:.2f})')
            ax.legend()
            ax.set_title(f'Mass Flux - {self.labels[population_idx]} (z={z_slice}) - Zoomed View')
        else:
            ax.set_title(f'Mass Flux - {self.labels[population_idx]} (z={z_slice})')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')
        fig.colorbar(quiv, ax=ax)
        
        plt.tight_layout()
        
        if save_plot and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f'mass_flux_{self.labels[population_idx]}_step_{step:06d}.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            print(f"Mass flux field plot saved to {output_dir}/{filename}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()

    def plot_tumor_density(self, step=0, z_slice=None, cmap="viridis", 
                          output_dir=None, save_plot=False, show_plot=True,
                          add_contours=False, contour_population_idx=0,
                          contour_levels=[0.1, 0.3, 0.5],
                          contour_colors=['red', 'orange', 'yellow'],
                          contour_linewidths=[2, 1.5, 1]):
        """
        Plot total tumor density field.
        
        Args:
            step: Current simulation step for filename
            z_slice: Z-slice to plot (defaults to center)
            cmap: Colormap for the plot
            output_dir: Directory to save plot
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
        """
        nx, ny, nz = self.grid
        if z_slice is None:
            z_slice = nz // 2
        
        # Create coordinate grids
        x = np.arange(nx) * self.dx
        y = np.arange(ny) * self.dx
        
        # Calculate total tumor density (sum of all populations)
        total_density = np.sum(self.field_manager.phi_hat, axis=0)[:, :, z_slice]
        
        # Create plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        
        im = ax.imshow(total_density, extent=[x[0], x[-1], y[0], y[-1]], 
                      origin='lower', cmap=cmap, aspect='equal')
        
        # Add tumor contours if requested
        if add_contours:
            self._add_tumor_contours(ax, contour_population_idx, z_slice, 
                                   contour_levels, contour_colors, contour_linewidths)
        
        ax.set_title(f'Total Tumor Density (z={z_slice})')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        fig.colorbar(im, ax=ax, label='Tumor Density')
        
        plt.tight_layout()
        
        if save_plot and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f'tumor_density_step_{step:06d}.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            print(f"Tumor density plot saved to {output_dir}/{filename}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
    def plot_tumor_boundary_contour(self, population_idx=0, step=0, z_slice=None, 
                                  contour_levels=[0.1, 0.3, 0.5], contour_colors=['red', 'orange', 'yellow'],
                                  contour_linewidths=[2, 1.5, 1], background_cmap="Blues", 
                                  output_dir=None, save_plot=False, show_plot=True):
        """
        Plot tumor boundary contours with customizable colors and line styles.
        
        Args:
            population_idx: Index of population to plot (default 0)
            step: Current simulation step for filename
            z_slice: Z-slice to plot (defaults to center)
            contour_levels: List of density levels for contours
            contour_colors: List of colors for each contour level
            contour_linewidths: List of line widths for each contour level
            background_cmap: Colormap for the background density
            output_dir: Directory to save plot
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
        """
        if population_idx >= self.M:
            print(f"Error: Population index {population_idx} out of range (0-{self.M-1})")
            return
        
        nx, ny, nz = self.grid
        if z_slice is None:
            z_slice = nz // 2
        
        # Create coordinate grids
        x = np.arange(nx) * self.dx
        y = np.arange(ny) * self.dx
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Get tumor density
        tumor_density = self.field_manager.phi_hat[population_idx, :, :, z_slice]
        
        # Create plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111)
        
        # Plot background density
        im = ax.imshow(tumor_density, extent=[x[0], x[-1], y[0], y[-1]], 
                      origin='lower', cmap=background_cmap, alpha=0.7, aspect='equal')
        
        # Plot contours
        for i, level in enumerate(contour_levels):
            color = contour_colors[i] if i < len(contour_colors) else 'black'
            linewidth = contour_linewidths[i] if i < len(contour_linewidths) else 1
            ax.contour(X, Y, tumor_density, levels=[level], colors=color, 
                      linewidths=linewidth, alpha=0.8, label=f'Density = {level}')
        
        ax.set_title(f'Tumor Boundary Contours - {self.labels[population_idx]} (z={z_slice})')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        fig.colorbar(im, ax=ax, label='Tumor Density')
        
        plt.tight_layout()
        
        if save_plot and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f'tumor_boundary_contour_{self.labels[population_idx]}_step_{step:06d}.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            print(f"Tumor boundary contour plot saved to {output_dir}/{filename}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()

    def plot_all_physics_fields(self, phi_hat, step=0, z_slice=None, output_dir=None,
                               save_plots=False, show_plots=True):
        """
        Plot all physics fields for the current simulation state.
        
        Args:
            phi_hat: Cell fraction fields (M, nx, ny, nz)
            step: Current simulation step for filename
            z_slice: Z-slice to plot (defaults to center)
            output_dir: Directory to save plots
            save_plots: Whether to save the plots
            show_plots: Whether to display the plots
        """
        print(f"Generating physics field plots for step {step}...")
        
        # Plot pressure field
        self.plot_pressure_field(step, z_slice, output_dir=output_dir, 
                               save_plot=save_plots, show_plot=show_plots)
        
        # Plot velocity field
        self.plot_velocity_field(step, z_slice, output_dir=output_dir, 
                               save_plot=save_plots, show_plot=show_plots)
        
        # Plot energy derivative field
        self.plot_energy_derivative_field(step, z_slice, output_dir=output_dir, 
                                        save_plot=save_plots, show_plot=show_plots)
        
        # Plot mass flux fields for each population
        for i in range(self.M):
            self.plot_mass_flux_field(i, step, z_slice, output_dir=output_dir, 
                                    save_plot=save_plots, show_plot=show_plots)
        
        # Plot tumor density
        self.plot_tumor_density(step, z_slice, output_dir=output_dir, 
                              save_plot=save_plots, show_plot=show_plots)
        
        # Plot cell density fields
        self.plot_cell_density_field(phi_hat, None, step, z_slice, output_dir=output_dir, 
                                   save_plot=save_plots, show_plot=show_plots)  # Total density
        for i in range(self.M):
            self.plot_cell_density_field(phi_hat, i, step, z_slice, output_dir=output_dir, 
                                       save_plot=save_plots, show_plot=show_plots)  # Individual populations
        
        print(f"All physics field plots completed for step {step}")
