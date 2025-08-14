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
        Plot pressure field as 2D heatmap.
        
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
        pressure_slice = self.field_manager.pressure[:, :, z_slice]
        
        # Handle NaN and infinite values
        pressure_slice = np.nan_to_num(pressure_slice, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Create plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        
        im = ax.imshow(pressure_slice, extent=[x[0], x[-1], y[0], y[-1]], 
                      origin='lower', cmap=cmap, aspect='equal')
        
        # Add tumor boundary contours if requested
        if add_boundary_contours:
            # Get total tumor density for boundary detection
            tumor_density = np.sum(self.field_manager.phi_hat, axis=0)[:, :, z_slice]
            ax.contour(tumor_density, levels=[tumor_boundary_level], 
                      colors=tumor_boundary_color, linewidths=3, alpha=0.9,
                      extent=[x[0], x[-1], y[0], y[-1]], origin='lower')
        
        ax.set_title(f'Pressure Field (z={z_slice})')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        fig.colorbar(im, ax=ax, label='Pressure')
        
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
                           zoom_factor=1.0, arrow_density_factor=1.0):
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
            center_x: Center x coordinate for zoom
            center_y: Center y coordinate for zoom
            zoom_factor: Zoom factor (1.0 = no zoom)
            arrow_density_factor: Factor to increase arrow density (1.0 = normal, 2.0 = 2x density)
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
        
        # Sample for visualization - adjust skip based on zoom factor and arrow density
        # For arrow_density_factor > 1, we want to sample more frequently (smaller skip)
        # For arrow_density_factor < 1, we want to sample less frequently (larger skip)
        # Use a more robust calculation that handles extreme values better
        base_skip = skip / zoom_factor
        effective_skip = max(1, int(base_skip / arrow_density_factor))
        x_skip = x_region[::effective_skip]
        y_skip = y_region[::effective_skip]
        X_skip, Y_skip = np.meshgrid(x_skip, y_skip, indexing='ij')
        ux_skip = ux_slice[::effective_skip, ::effective_skip]
        uy_skip = uy_slice[::effective_skip, ::effective_skip]
        
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
            
            # Add tumor boundary contour using the same coordinate system as quiver
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
        X_region, Y_region = np.meshgrid(x_region, y_region, indexing='ij')
        
        # Extract energy derivative for zoomed region
        energy_slice = self.field_manager.energy_derivative[x_start:x_end, y_start:y_end, z_slice]
        
        # Handle NaN and infinite values
        energy_slice = np.nan_to_num(energy_slice, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Create plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        
        # Use symmetric colormap limits for better visualization
        vmax = np.max(np.abs(energy_slice))
        vmin = -vmax if vmax > 0 else -1.0
        
        im = ax.imshow(energy_slice, extent=[x_region[0], x_region[-1], y_region[0], y_region[-1]], 
                      origin='lower', cmap=cmap, aspect='equal', vmin=vmin, vmax=vmax)
        
        # Add tumor boundary contours if requested
        if add_boundary_contours:
            # Get total tumor density for boundary detection (zoomed region)
            tumor_density = np.sum(self.field_manager.phi_hat, axis=0)[x_start:x_end, y_start:y_end, z_slice]
            
            # Add tumor boundary contour using the same coordinate system as imshow
            ax.contour(tumor_density, levels=[tumor_boundary_level], 
                      colors=tumor_boundary_color, linewidths=3, alpha=0.9,
                      extent=[x_region[0], x_region[-1], y_region[0], y_region[-1]], 
                      origin='lower')
        
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
                           zoom_factor=1.0, arrow_density_factor=1.0,
                           contour_shift_x=0, contour_shift_y=0):
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
            add_boundary_contours: Whether to add tumor boundary contours
            tumor_boundary_level: Density level for tumor boundary contour (default 0.5)
            boundary_color: Color for boundary region contour (default 'darkblue')
            tumor_boundary_color: Color for tumor boundary contour (default 'black')
            center_x: Center x coordinate for zoom
            center_y: Center y coordinate for zoom
            zoom_factor: Zoom factor (1.0 = no zoom)
            arrow_density_factor: Factor to increase arrow density (1.0 = normal, 2.0 = 2x density)
            contour_shift_x: Shift contour by this many grid units in x direction
            contour_shift_y: Shift contour by this many grid units in y direction
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
        
        # Sample for visualization - adjust skip based on zoom factor and arrow density
        # For arrow_density_factor > 1, we want to sample more frequently (smaller skip)
        # For arrow_density_factor < 1, we want to sample less frequently (larger skip)
        # Use a more robust calculation that handles extreme values better
        base_skip = skip / zoom_factor
        effective_skip = max(1, int(base_skip / arrow_density_factor))
        x_skip = x_region[::effective_skip]
        y_skip = y_region[::effective_skip]
        X_skip, Y_skip = np.meshgrid(x_skip, y_skip, indexing='ij')
        flux_x_skip = flux_x_slice[::effective_skip, ::effective_skip]
        flux_y_skip = flux_y_slice[::effective_skip, ::effective_skip]
        
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
            
            # Apply contour shift by adjusting the coordinate grids
            X_region_shifted = X_region + contour_shift_x * self.dx
            Y_region_shifted = Y_region + contour_shift_y * self.dx
            
            # Add tumor boundary contour using the shifted coordinate system
            ax.contour(X_region_shifted, Y_region_shifted, tumor_density, levels=[tumor_boundary_level], 
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

    @classmethod
    def from_simulation_data(cls, simulation_data, step_idx=0):
        """
        Create a plotter instance from saved simulation data.
        
        Args:
            simulation_data: Dictionary containing simulation data
            step_idx: Index of the saved step to use for plotting
            
        Returns:
            plotter: PhysicsFieldsPlotter instance with loaded data
        """
        # Create a mock field manager with the saved data
        class MockFieldManager:
            def __init__(self, data, step_idx):
                self.grid = data["metadata"]["grid_size"]
                self.dx = 1.0  # Default value, should be extracted from config
                self.labels = [f"Population_{i}" for i in range(data["metadata"]["num_populations"])]
                self.M = data["metadata"]["num_populations"]
                
                # Load field data
                self.phi_hat = data["field_data"]["phi_hat"][step_idx]
                self.nutrient_field = data["field_data"]["nutrient_fields"][step_idx]
                
                # Load physics data if available
                if "physics_data" in data:
                    physics = data["physics_data"][step_idx]
                    self.pressure = physics["pressure"]
                    self.velocity = physics["velocity"]
                    self.energy_derivative = physics["energy_derivative"]
                    self.mass_flux = physics["mass_flux"]
                    # Load source terms if available
                    if "source_terms" in physics:
                        self.source_terms = physics["source_terms"]
                    else:
                        self.source_terms = np.zeros((self.M,) + self.grid)
                else:
                    # Initialize empty physics fields
                    self.pressure = np.zeros(self.grid)
                    self.velocity = np.zeros((3,) + self.grid)
                    self.energy_derivative = np.zeros(self.grid)
                    self.mass_flux = np.zeros((self.M, 3) + self.grid)
                    self.source_terms = np.zeros((self.M,) + self.grid)
        
        # Create mock field manager
        field_manager = MockFieldManager(simulation_data, step_idx)
        
        # Create and return plotter
        return cls(field_manager)
    
    def plot_pressure_field_from_saved(self, simulation_data, step_idx=0, z_slice=None, 
                                     cmap="viridis", alpha=0.8, output_dir=None, 
                                     save_plot=False, show_plot=True,
                                     add_boundary_contours=False, tumor_boundary_level=0.5,
                                     boundary_color='darkblue', tumor_boundary_color='black'):
        """
        Plot pressure field from saved simulation data.
        
        Args:
            simulation_data: Dictionary containing simulation data
            step_idx: Index of the saved step to plot
            z_slice: Z-slice to plot (defaults to center)
            cmap: Colormap for the plot
            alpha: Transparency of the surface
            output_dir: Directory to save plot
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            add_boundary_contours: Whether to add tumor boundary contours
            tumor_boundary_level: Density level for tumor boundary contour
            boundary_color: Color for boundary region contour
            tumor_boundary_color: Color for tumor boundary contour
        """
        # Get step information
        step = simulation_data["metadata"]["saved_steps"][step_idx]
        time = simulation_data["metadata"]["saved_times"][step_idx]
        
        print(f"Plotting pressure field for step {step} (time={time:.3f}) from saved data...")
        
        # Create plotter for this step
        plotter = self.from_simulation_data(simulation_data, step_idx)
        
        # Plot pressure field
        plotter.plot_pressure_field(
            step=step, z_slice=z_slice, cmap=cmap, alpha=alpha,
            output_dir=output_dir, save_plot=save_plot, show_plot=show_plot,
            add_boundary_contours=add_boundary_contours, 
            tumor_boundary_level=tumor_boundary_level,
            boundary_color=boundary_color, 
            tumor_boundary_color=tumor_boundary_color
        )
    
    def plot_velocity_field_from_saved(self, simulation_data, step_idx=0, z_slice=None, 
                                     skip=2, cmap="viridis", scale=50, width=0.005, 
                                     output_dir=None, save_plot=False, show_plot=True, 
                                     add_boundary_contours=False, tumor_boundary_level=0.5,
                                     boundary_color='darkblue', tumor_boundary_color='black', 
                                     center_x=None, center_y=None, zoom_factor=1.0, 
                                     arrow_density_factor=1.0):
        """
        Plot velocity field from saved simulation data.
        
        Args:
            simulation_data: Dictionary containing simulation data
            step_idx: Index of the saved step to plot
            z_slice: Z-slice to plot (defaults to center)
            skip: Sampling rate for vectors (every nth point)
            cmap: Colormap for the plot
            scale: Scale factor for vectors
            width: Width of vectors
            output_dir: Directory to save plot
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            add_boundary_contours: Whether to add tumor boundary contours
            tumor_boundary_level: Density level for tumor boundary contour
            boundary_color: Color for boundary region contour
            tumor_boundary_color: Color for tumor boundary contour
            center_x: Center x coordinate for zoom
            center_y: Center y coordinate for zoom
            zoom_factor: Zoom factor (1.0 = no zoom)
        """
        # Get step information
        step = simulation_data["metadata"]["saved_steps"][step_idx]
        time = simulation_data["metadata"]["saved_times"][step_idx]
        
        print(f"Plotting velocity field for step {step} (time={time:.3f}) from saved data...")
        
        # Create plotter for this step
        plotter = self.from_simulation_data(simulation_data, step_idx)
        
        # Plot velocity field
        plotter.plot_velocity_field(
            step=step, z_slice=z_slice, skip=skip, cmap=cmap, scale=scale, width=width,
            output_dir=output_dir, save_plot=save_plot, show_plot=show_plot,
            add_boundary_contours=add_boundary_contours, 
            tumor_boundary_level=tumor_boundary_level,
            boundary_color=boundary_color, 
            tumor_boundary_color=tumor_boundary_color,
            center_x=center_x, center_y=center_y, zoom_factor=zoom_factor,
            arrow_density_factor=arrow_density_factor
        )
    
    def plot_energy_derivative_field_from_saved(self, simulation_data, step_idx=0, z_slice=None, 
                                              cmap="RdBu_r", output_dir=None, save_plot=False, 
                                              show_plot=True, add_boundary_contours=False, 
                                              tumor_boundary_level=0.5, boundary_color='darkblue', 
                                              tumor_boundary_color='black', center_x=None, 
                                              center_y=None, zoom_factor=1.0):
        """
        Plot energy derivative field from saved simulation data.
        
        Args:
            simulation_data: Dictionary containing simulation data
            step_idx: Index of the saved step to plot
            z_slice: Z-slice to plot (defaults to center)
            cmap: Colormap for the plot
            output_dir: Directory to save plot
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            add_boundary_contours: Whether to add tumor boundary contours
            tumor_boundary_level: Density level for tumor boundary contour
            boundary_color: Color for boundary region contour
            tumor_boundary_color: Color for tumor boundary contour
            center_x: Center x coordinate for zoom
            center_y: Center y coordinate for zoom
            zoom_factor: Zoom factor (1.0 = no zoom)
        """
        # Get step information
        step = simulation_data["metadata"]["saved_steps"][step_idx]
        time = simulation_data["metadata"]["saved_times"][step_idx]
        
        print(f"Plotting energy derivative field for step {step} (time={time:.3f}) from saved data...")
        
        # Create plotter for this step
        plotter = self.from_simulation_data(simulation_data, step_idx)
        
        # Plot energy derivative field
        plotter.plot_energy_derivative_field(
            step=step, z_slice=z_slice, cmap=cmap, output_dir=output_dir, 
            save_plot=save_plot, show_plot=show_plot,
            add_boundary_contours=add_boundary_contours, 
            tumor_boundary_level=tumor_boundary_level,
            boundary_color=boundary_color, 
            tumor_boundary_color=tumor_boundary_color,
            center_x=center_x, center_y=center_y, zoom_factor=zoom_factor
        )
    
    def plot_mass_flux_field_from_saved(self, simulation_data, population_idx=0, step_idx=0, 
                                       z_slice=None, skip=2, cmap="viridis", scale=20, 
                                       width=0.005, output_dir=None, save_plot=False, 
                                       show_plot=True, add_boundary_contours=False,
                                       tumor_boundary_level=0.5, boundary_color='darkblue', 
                                       tumor_boundary_color='black', center_x=None, 
                                       center_y=None, zoom_factor=1.0, arrow_density_factor=1.0,
                                       contour_shift_x=0, contour_shift_y=0):
        """
        Plot mass flux field from saved simulation data.
        
        Args:
            simulation_data: Dictionary containing simulation data
            population_idx: Index of population to plot
            step_idx: Index of the saved step to plot
            z_slice: Z-slice to plot (defaults to center)
            skip: Sampling rate for vectors (every nth point)
            cmap: Colormap for the plot
            scale: Scale factor for vectors
            width: Width of vectors
            output_dir: Directory to save plot
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            add_boundary_contours: Whether to add tumor boundary contours
            tumor_boundary_level: Density level for tumor boundary contour
            boundary_color: Color for boundary region contour
            tumor_boundary_color: Color for tumor boundary contour
            center_x: Center x coordinate for zoom
            center_y: Center y coordinate for zoom
            zoom_factor: Zoom factor (1.0 = no zoom)
            arrow_density_factor: Factor to increase arrow density (1.0 = normal, 2.0 = 2x density)
            contour_shift_x: Shift contour by this many grid units in x direction
            contour_shift_y: Shift contour by this many grid units in y direction
        """
        # Get step information
        step = simulation_data["metadata"]["saved_steps"][step_idx]
        time = simulation_data["metadata"]["saved_times"][step_idx]
        
        print(f"Plotting mass flux field for population {population_idx}, step {step} (time={time:.3f}) from saved data...")
        
        # Create plotter for this step
        plotter = self.from_simulation_data(simulation_data, step_idx)
        
        # Plot mass flux field
        plotter.plot_mass_flux_field(
            population_idx=population_idx, step=step, z_slice=z_slice, skip=skip, 
            cmap=cmap, scale=scale, width=width, output_dir=output_dir, 
            save_plot=save_plot, show_plot=show_plot,
            add_boundary_contours=add_boundary_contours, 
            tumor_boundary_level=tumor_boundary_level,
            boundary_color=boundary_color, 
            tumor_boundary_color=tumor_boundary_color,
            center_x=center_x, center_y=center_y, zoom_factor=zoom_factor,
            arrow_density_factor=arrow_density_factor,
            contour_shift_x=contour_shift_x, contour_shift_y=contour_shift_y
        )
    
    def plot_tumor_density_from_saved(self, simulation_data, step_idx=0, z_slice=None, 
                                    cmap="viridis", output_dir=None, save_plot=False, 
                                    show_plot=True, add_contours=False, contour_population_idx=0,
                                    contour_levels=[0.1, 0.3, 0.5], contour_colors=['red', 'orange', 'yellow'],
                                    contour_linewidths=[2, 1.5, 1]):
        """
        Plot tumor density from saved simulation data.
        
        Args:
            simulation_data: Dictionary containing simulation data
            step_idx: Index of the saved step to plot
            z_slice: Z-slice to plot (defaults to center)
            cmap: Colormap for the plot
            output_dir: Directory to save plot
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            add_contours: Whether to add tumor boundary contours
            contour_population_idx: Index of population to contour
            contour_levels: List of density levels for contours
            contour_colors: List of colors for each contour level
            contour_linewidths: List of line widths for each contour level
        """
        # Get step information
        step = simulation_data["metadata"]["saved_steps"][step_idx]
        time = simulation_data["metadata"]["saved_times"][step_idx]
        
        print(f"Plotting tumor density for step {step} (time={time:.3f}) from saved data...")
        
        # Create plotter for this step
        plotter = self.from_simulation_data(simulation_data, step_idx)
        
        # Plot tumor density
        plotter.plot_tumor_density(
            step=step, z_slice=z_slice, cmap=cmap, output_dir=output_dir, 
            save_plot=save_plot, show_plot=show_plot, add_contours=add_contours,
            contour_population_idx=contour_population_idx, contour_levels=contour_levels,
            contour_colors=contour_colors, contour_linewidths=contour_linewidths
        )
    
    def plot_cell_density_field_from_saved(self, simulation_data, population_idx=None, step_idx=0, 
                                         z_slice=None, cmap="viridis", output_dir=None, 
                                         save_plot=False, show_plot=True, add_contours=False, 
                                         contour_colors=['white', 'yellow', 'red'],
                                         contour_levels=[0.1, 0.3, 0.5], contour_linewidths=[2, 1.5, 1]):
        """
        Plot cell density field from saved simulation data.
        
        Args:
            simulation_data: Dictionary containing simulation data
            population_idx: Index of population to plot (None for total density)
            step_idx: Index of the saved step to plot
            z_slice: Z-slice to plot (defaults to center)
            cmap: Colormap for the plot
            output_dir: Directory to save plot
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            add_contours: Whether to add tumor boundary contours
            contour_colors: List of colors for each contour level
            contour_levels: List of density levels for contours
            contour_linewidths: List of line widths for each contour level
        """
        # Get step information
        step = simulation_data["metadata"]["saved_steps"][step_idx]
        time = simulation_data["metadata"]["saved_times"][step_idx]
        
        population_name = "Total" if population_idx is None else f"Population {population_idx}"
        print(f"Plotting cell density ({population_name}) for step {step} (time={time:.3f}) from saved data...")
        
        # Create plotter for this step
        plotter = self.from_simulation_data(simulation_data, step_idx)
        
        # Get phi_hat from saved data
        phi_hat = simulation_data["field_data"]["phi_hat"][step_idx]
        
        # Plot cell density field
        plotter.plot_cell_density_field(
            phi_hat=phi_hat, population_idx=population_idx, step=step, z_slice=z_slice, 
            cmap=cmap, output_dir=output_dir, save_plot=save_plot, show_plot=show_plot,
            add_contours=add_contours, contour_colors=contour_colors,
            contour_levels=contour_levels, contour_linewidths=contour_linewidths
        )
    
    def plot_source_terms(self, population_idx=0, step=0, z_slice=None, cmap="RdBu_r", 
                         alpha=0.8, output_dir=None, save_plot=False, show_plot=True,
                         add_boundary_contours=False, tumor_boundary_level=0.5,
                         boundary_color='darkblue', tumor_boundary_color='black'):
        """
        Plot source terms (growth/death) for a specific population as 2D color map.
        
        Args:
            population_idx: Index of population to plot
            step: Current simulation step for filename
            z_slice: Z-slice to plot (defaults to center)
            cmap: Colormap for the plot (RdBu_r for red=growth, blue=death)
            alpha: Transparency of the surface
            output_dir: Directory to save plot
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            add_boundary_contours: Whether to add tumor boundary contours
            tumor_boundary_level: Density level for tumor boundary contour
            boundary_color: Color for boundary region contour
            tumor_boundary_color: Color for tumor boundary contour
        """
        if population_idx >= self.M:
            print(f"Warning: Population index {population_idx} out of range (0-{self.M-1})")
            return
            
        nx, ny, nz = self.grid
        if z_slice is None:
            z_slice = nz // 2
        
        # Create coordinate grids
        x = np.arange(nx) * self.dx
        y = np.arange(ny) * self.dx
        source_slice = self.field_manager.source_terms[population_idx, :, :, z_slice]
        
        # Handle NaN and infinite values
        source_slice = np.nan_to_num(source_slice, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Create plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        
        # Use symmetric colormap limits for better visualization
        vmax = np.max(np.abs(source_slice))
        vmin = -vmax if vmax > 0 else -1.0
        
        im = ax.imshow(source_slice, extent=[x[0], x[-1], y[0], y[-1]], 
                      origin='lower', cmap=cmap, aspect='equal', vmin=vmin, vmax=vmax)
        
        # Add tumor boundary contours if requested
        if add_boundary_contours:
            # Get tumor density for boundary detection
            tumor_density = self.field_manager.phi_hat[population_idx, :, :, z_slice]
            ax.contour(tumor_density, levels=[tumor_boundary_level], 
                      colors=tumor_boundary_color, linewidths=3, alpha=0.9,
                      extent=[x[0], x[-1], y[0], y[-1]], origin='lower')
        
        ax.set_title(f'Source Terms - {self.labels[population_idx]} (z={z_slice})')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        fig.colorbar(im, ax=ax, label='Growth/Death Rate')
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_plot and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f"source_terms_pop{population_idx}_step{step:06d}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved source terms plot to {filepath}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_nutrient_field(self, step=0, z_slice=None, cmap="viridis", alpha=0.8, 
                           output_dir=None, save_plot=False, show_plot=True,
                           add_boundary_contours=False, tumor_boundary_level=0.5,
                           boundary_color='darkblue', tumor_boundary_color='black'):
        """
        Plot nutrient field as 2D color map.
        
        Args:
            step: Current simulation step for filename
            z_slice: Z-slice to plot (defaults to center)
            cmap: Colormap for the plot
            alpha: Transparency of the surface
            output_dir: Directory to save plot
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            add_boundary_contours: Whether to add tumor boundary contours
            tumor_boundary_level: Density level for tumor boundary contour
            boundary_color: Color for boundary region contour
            tumor_boundary_color: Color for tumor boundary contour
        """
        nx, ny, nz = self.grid
        if z_slice is None:
            z_slice = nz // 2
        
        # Create coordinate grids
        x = np.arange(nx) * self.dx
        y = np.arange(ny) * self.dx
        nutrient_slice = self.field_manager.nutrient_field[:, :, z_slice]
        
        # Handle NaN and infinite values
        nutrient_slice = np.nan_to_num(nutrient_slice, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Create plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        
        im = ax.imshow(nutrient_slice, extent=[x[0], x[-1], y[0], y[-1]], 
                      origin='lower', cmap=cmap, aspect='equal')
        
        # Add tumor boundary contours if requested
        if add_boundary_contours:
            # Use the first population for contours (usually the main tumor population)
            tumor_density = self.field_manager.phi_hat[0, :, :, z_slice]
            ax.contour(tumor_density, levels=[tumor_boundary_level], 
                      colors=tumor_boundary_color, linewidths=3, alpha=0.9,
                      extent=[x[0], x[-1], y[0], y[-1]], origin='lower')
        
        ax.set_title(f'Nutrient Field (z={z_slice})')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        fig.colorbar(im, ax=ax, label='Concentration')
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_plot and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f"nutrient_field_step{step:06d}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved nutrient field plot to {filepath}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_source_terms_from_saved(self, simulation_data, population_idx=0, step_idx=0, 
                                   z_slice=None, cmap="RdBu_r", alpha=0.8, output_dir=None, 
                                   save_plot=False, show_plot=True, add_boundary_contours=False, 
                                   tumor_boundary_level=0.5, boundary_color='darkblue', 
                                   tumor_boundary_color='black'):
        """
        Plot source terms from saved simulation data.
        
        Args:
            simulation_data: Dictionary containing simulation data
            population_idx: Index of population to plot
            step_idx: Index of the saved step to plot
            z_slice: Z-slice to plot (defaults to center)
            cmap: Colormap for the plot
            alpha: Transparency of the surface
            output_dir: Directory to save plot
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            add_boundary_contours: Whether to add tumor boundary contours
            tumor_boundary_level: Density level for tumor boundary contour
            boundary_color: Color for boundary region contour
            tumor_boundary_color: Color for tumor boundary contour
        """
        # Get step information
        step = simulation_data["metadata"]["saved_steps"][step_idx]
        time = simulation_data["metadata"]["saved_times"][step_idx]
        
        print(f"Plotting source terms for population {population_idx}, step {step} (time={time:.3f}) from saved data...")
        
        # Create plotter for this step
        plotter = self.from_simulation_data(simulation_data, step_idx)
        
        # Plot source terms
        plotter.plot_source_terms(
            population_idx=population_idx, step=step, z_slice=z_slice, cmap=cmap, 
            alpha=alpha, output_dir=output_dir, save_plot=save_plot, show_plot=show_plot,
            add_boundary_contours=add_boundary_contours, 
            tumor_boundary_level=tumor_boundary_level,
            boundary_color=boundary_color, 
            tumor_boundary_color=tumor_boundary_color
        )
    
    def plot_nutrient_field_from_saved(self, simulation_data, step_idx=0, z_slice=None, 
                                     cmap="viridis", alpha=0.8, output_dir=None, 
                                     save_plot=False, show_plot=True, add_boundary_contours=False, 
                                     tumor_boundary_level=0.5, boundary_color='darkblue', 
                                     tumor_boundary_color='black'):
        """
        Plot nutrient field from saved simulation data.
        
        Args:
            simulation_data: Dictionary containing simulation data
            step_idx: Index of the saved step to plot
            z_slice: Z-slice to plot (defaults to center)
            cmap: Colormap for the plot
            alpha: Transparency of the surface
            output_dir: Directory to save plot
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            add_boundary_contours: Whether to add tumor boundary contours
            tumor_boundary_level: Density level for tumor boundary contour
            boundary_color: Color for boundary region contour
            tumor_boundary_color: Color for tumor boundary contour
        """
        # Get step information
        step = simulation_data["metadata"]["saved_steps"][step_idx]
        time = simulation_data["metadata"]["saved_times"][step_idx]
        
        print(f"Plotting nutrient field for step {step} (time={time:.3f}) from saved data...")
        
        # Create plotter for this step
        plotter = self.from_simulation_data(simulation_data, step_idx)
        
        # Plot nutrient field
        plotter.plot_nutrient_field(
            step=step, z_slice=z_slice, cmap=cmap, alpha=alpha, output_dir=output_dir, 
            save_plot=save_plot, show_plot=show_plot, add_boundary_contours=add_boundary_contours,
            tumor_boundary_level=tumor_boundary_level, boundary_color=boundary_color,
            tumor_boundary_color=tumor_boundary_color
        )