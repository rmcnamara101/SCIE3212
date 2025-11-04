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
                           boundary_color='darkblue', tumor_boundary_color='black', center_x=None, center_y=None, zoom_factor=1.0):
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
            zoom_factor: Zoom factor (1.0 = no zoom)
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
        
        # Get full pressure field
        pressure_slice = self.field_manager.pressure[:, :, z_slice]
        
        # Handle NaN and infinite values
        pressure_slice = np.nan_to_num(pressure_slice, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Create coordinate grids for the full domain
        x_full = np.arange(nx) * self.dx
        y_full = np.arange(ny) * self.dx
        
        # Create plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        
        # Plot the full field
        im = ax.imshow(pressure_slice, extent=[x_full[0], x_full[-1], y_full[0], y_full[-1]], 
                      origin='lower', cmap=cmap, aspect='equal')
        
        # Add tumor boundary contours if requested
        if add_boundary_contours:
            # Get total tumor density for boundary detection
            tumor_density = np.sum(self.field_manager.phi_hat, axis=0)[:, :, z_slice]
            ax.contour(tumor_density, levels=[tumor_boundary_level], 
                      colors=tumor_boundary_color, linewidths=3, alpha=0.9,
                      extent=[x_full[0], x_full[-1], y_full[0], y_full[-1]], origin='lower')
        
        # Apply zoom if requested
        if zoom_factor > 1.0:
            # Calculate zoom window size
            window_size = min(nx, ny) // zoom_factor
            half_window = window_size // 2
            
            # Calculate window boundaries in physical coordinates
            center_x_phys = center_x * self.dx
            center_y_phys = center_y * self.dx
            window_size_phys = window_size * self.dx
            half_window_phys = window_size_phys / 2
            
            x_min = center_x_phys - half_window_phys
            x_max = center_x_phys + half_window_phys
            y_min = center_y_phys - half_window_phys
            y_max = center_y_phys + half_window_phys
            
            # Set axis limits to zoom in
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_title(f'Pressure Field (z={z_slice}) - Zoomed View (factor: {zoom_factor:.1f})')
        else:
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
        
        # Get full flux field
        flux_x_full = self.field_manager.mass_flux[population_idx, 0, :, :, z_slice]
        flux_y_full = self.field_manager.mass_flux[population_idx, 1, :, :, z_slice]
        
        # Create coordinate grids for the full domain
        x_full = np.arange(nx) * self.dx
        y_full = np.arange(ny) * self.dx
        X_full, Y_full = np.meshgrid(x_full, y_full, indexing='ij')
        
        # Calculate effective skip based on zoom and density factors
        # Higher arrow_density_factor = more arrows (smaller skip)
        base_skip = max(1, int(skip / zoom_factor))
        effective_skip = max(1, int(base_skip / arrow_density_factor))
        
        # Sample the full field
        x_indices = np.arange(0, nx, effective_skip)
        y_indices = np.arange(0, ny, effective_skip)
        
        # Create sampling meshgrid
        X_indices, Y_indices = np.meshgrid(x_indices, y_indices, indexing='ij')
        
        # Sample coordinates and flux data
        X_skip = X_full[x_indices[:, None], y_indices]
        Y_skip = Y_full[x_indices[:, None], y_indices]
        flux_x_skip = flux_x_full[x_indices[:, None], y_indices]
        flux_y_skip = flux_y_full[x_indices[:, None], y_indices]
        
        # Calculate magnitudes for coloring
        flux_magnitude = np.sqrt(flux_x_skip**2 + flux_y_skip**2)
        
        # Create plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        
        # Plot quiver
        quiv = ax.quiver(X_skip, Y_skip, flux_x_skip, flux_y_skip, flux_magnitude, 
                        cmap=cmap, scale=scale, width=width)
        
        # Add tumor boundary contours if requested
        if add_boundary_contours:
            # Get total tumor density for boundary detection
            tumor_density = np.sum(self.field_manager.phi_hat, axis=0)[:, :, z_slice]
            
            # Apply contour shift by adjusting the coordinate grids
            X_shifted = X_full + contour_shift_x * self.dx
            Y_shifted = Y_full + contour_shift_y * self.dx
            
            # Add tumor boundary contour using the shifted coordinate system
            ax.contour(X_shifted, Y_shifted, tumor_density, levels=[tumor_boundary_level], 
                      colors=tumor_boundary_color, linewidths=3, alpha=0.9)
        
        # Apply zoom if requested
        if zoom_factor > 1.0:
            # Calculate zoom window in physical coordinates
            center_x_phys = center_x * self.dx
            center_y_phys = center_y * self.dx
            window_size_phys = min(nx, ny) * self.dx / zoom_factor
            half_window_phys = window_size_phys / 2
            
            x_min = center_x_phys - half_window_phys
            x_max = center_x_phys + half_window_phys
            y_min = center_y_phys - half_window_phys
            y_max = center_y_phys + half_window_phys
            
            # Set axis limits to zoom in
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            
            # Add center point marker
            ax.plot(center_x_phys, center_y_phys, 'k+', markersize=10, markeredgewidth=2, 
                   label=f'Center: ({center_x_phys:.2f}, {center_y_phys:.2f})')
            ax.legend()
            
            ax.set_title(f'Mass Flux - {self.labels[population_idx]} (z={z_slice}) - Zoomed View')
        else:
            ax.set_title(f'Mass Flux - {self.labels[population_idx]} (z={z_slice})')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')
        
        # Add colorbar
        cbar = fig.colorbar(quiv, ax=ax)
        cbar.set_label('Flux Magnitude')
        
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
                          contour_linewidths=[2, 1.5, 1], center_x=None, center_y=None, zoom_factor=1.0):
        """
        Plot total tumor density field.
        
        Args:
            step: Current simulation step for filename
            z_slice: Z-slice to plot (defaults to center)
            cmap: Colormap for the plot
            output_dir: Directory to save plot
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            add_contours: Whether to add tumor contours
            contour_population_idx: Index of population to plot contours for
            contour_levels: List of density levels for contours
            contour_colors: List of colors for each contour level
            contour_linewidths: List of line widths for each contour level
            center_x: Center x coordinate for zoom
            center_y: Center y coordinate for zoom
            zoom_factor: Zoom factor (1.0 = no zoom)
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
        
        # Create coordinate grids
        x = np.arange(x_start, x_end) * self.dx
        y = np.arange(y_start, y_end) * self.dx
        
        # Calculate total tumor density (sum of all populations)
        total_density = np.sum(self.field_manager.phi_hat, axis=0)[x_start:x_end, y_start:y_end, z_slice]
        
        # Create plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        
        im = ax.imshow(total_density, extent=[x[0], x[-1], y[0], y[-1]], 
                      origin='lower', cmap=cmap, aspect='equal')
        
        ax.set_title(f'Total Tumor Density (z={z_slice})')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        fig.colorbar(im, ax=ax, label='Tumor Density')   
        
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
                                     boundary_color='darkblue', tumor_boundary_color='black', center_x=None, center_y=None, zoom_factor=1.0):
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
            zoom_factor: Zoom factor (1.0 = no zoom)
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
            tumor_boundary_color=tumor_boundary_color,
            zoom_factor=zoom_factor,
            center_x=center_x, center_y=center_y
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
                         boundary_color='darkblue', tumor_boundary_color='black',
                         center_x=None, center_y=None, zoom_factor=1.0):
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
            center_x: Center x coordinate for zoom
            center_y: Center y coordinate for zoom
            zoom_factor: Zoom factor (1.0 = no zoom)
        """
        if population_idx >= self.M:
            print(f"Warning: Population index {population_idx} out of range (0-{self.M-1})")
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
        
        # Create coordinate grids
        x = np.arange(x_start, x_end) * self.dx
        y = np.arange(y_start, y_end) * self.dx
        source_slice = self.field_manager.source_terms[population_idx, x_start:x_end, y_start:y_end, z_slice]
        
        # Handle NaN and infinite values
        source_slice = np.nan_to_num(source_slice, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Create plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        
        # Use symmetric colormap limits for better visualization
        vmax = np.max(np.abs(source_slice))
        vmin = -vmax if vmax > 0 else -1.0
        
        im = ax.imshow(source_slice, extent=[x[0], x[-1], y[0], y[-1]], 
                      origin='lower', cmap=cmap, aspect='equal')
        
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
                           boundary_color='darkblue', tumor_boundary_color='black', center_x=None, center_y=None, zoom_factor=1.0   ):
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
            center_x: Center x coordinate for zoom
            center_y: Center y coordinate for zoom
            zoom_factor: Zoom factor (1.0 = no zoom)
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
        
        # Create coordinate grids
        x = np.arange(x_start, x_end) * self.dx
        y = np.arange(y_start, y_end) * self.dx
        nutrient_slice = self.field_manager.nutrient_field[x_start:x_end, y_start:y_end, z_slice]
        
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
            tumor_density = self.field_manager.phi_hat[0, x_start:x_end, y_start:y_end, z_slice]
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
                                   tumor_boundary_color='black', center_x=None, center_y=None, zoom_factor=1.0):
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
            tumor_boundary_color=tumor_boundary_color,
            center_x=center_x, center_y=center_y, zoom_factor=zoom_factor
        )
    
    def plot_nutrient_field_from_saved(self, simulation_data, step_idx=0, z_slice=None, 
                                     cmap="viridis", alpha=0.8, output_dir=None, 
                                     save_plot=False, show_plot=True, add_boundary_contours=False, 
                                     tumor_boundary_level=0.5, boundary_color='darkblue', 
                                     tumor_boundary_color='black', center_x=None, center_y=None, zoom_factor=1.0):
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
            tumor_boundary_color=tumor_boundary_color,
            center_x=center_x, center_y=center_y, zoom_factor=zoom_factor
        )
    
    def calculate_total_free_energy(self, phi_T, dx, m):
        """
        Calculate the total free energy for a given cell density field.
        
        The free energy functional is:
        E = ∫ m * (f(φ) - 0.01 * ∇²φ) dV
        
        where f(φ) = 0.5 * φ * (1 - φ) * (2φ - 1) is the double-well potential.
        
        Args:
            phi_T: Total cell density field
            dx: Grid spacing
            m: Adhesion energy parameter
            
        Returns:
            total_energy: Total free energy integrated over the domain
        """
        try:
            from src.growkit.MathEngine.Operators import isotropic_laplacian
            use_numba = True
        except ImportError:
            # Fallback to scipy for GUI usage (no Numba compilation)
            from scipy.ndimage import laplace
            use_numba = False
        
        # Ensure float64 for stability
        phi_T = phi_T.astype(np.float64, copy=False)
        
        # Compute double-well potential: f(φ) = 0.5 * φ * (1 - φ) * (2φ - 1)
        f_phi = 0.5 * phi_T * (1 - phi_T) * (2 * phi_T - 1)
        
        # Compute Laplacian
        if use_numba:
            laplace_phi = isotropic_laplacian(phi_T, dx)
        else:
            # Use scipy's laplacian as fallback
            laplace_phi = laplace(phi_T) / (dx ** 2)
        
        # Compute energy density: m * (f(φ) - 0.01 * ∇²φ)
        energy_density = m * (f_phi - 0.01 * laplace_phi)
        
        # Integrate over the domain (multiply by dx^3 for 3D volume element)
        total_energy = np.sum(energy_density) * (dx ** 3)
        
        return total_energy
    
    def calculate_total_free_energy_from_phi_hat(self, phi_hat, dx, m):
        """
        Calculate the total free energy from stacked population fields.
        
        Args:
            phi_hat: Stacked cell fraction fields (M, nx, ny, nz)
            dx: Grid spacing
            m: Adhesion energy parameter
            
        Returns:
            total_energy: Total free energy integrated over the domain
        """
        # Calculate total cell density
        phi_T = np.sum(phi_hat, axis=0)
        return self.calculate_total_free_energy(phi_T, dx, m)
    
    def plot_total_free_energy_evolution(self, simulation_data, output_dir=None, 
                                       save_plot=False, show_plot=True, 
                                       figsize=(10, 6), line_style='-', 
                                       line_color='blue', line_width=2,
                                       marker='o', marker_size=4):
        """
        Calculate and plot the total free energy evolution over time from saved simulation data.
        
        Args:
            simulation_data: Dictionary containing simulation data
            output_dir: Directory to save plot
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            figsize: Figure size tuple
            line_style: Line style for the plot
            line_color: Color of the line
            line_width: Width of the line
            marker: Marker style for data points
            marker_size: Size of markers
        """
        
        # Get simulation metadata
        saved_steps = simulation_data["metadata"]["saved_steps"]
        saved_times = simulation_data["metadata"]["saved_times"]
        grid_size = simulation_data["metadata"]["grid_size"]
        dx = 1.0  # Default value, should be extracted from config if available
        
        # Try to get adhesion energy parameter from config if available
        m = 1.0  # Default value
        if "config" in simulation_data["metadata"]:
            config = simulation_data["metadata"]["config"]
            if "physics" in config and "adhesion_energy" in config["physics"]:
                m = config["physics"]["adhesion_energy"]["m"]
        
        # Calculate total free energy for each saved step
        total_energies = []
        valid_times = []
        
        for i, (step, time) in enumerate(zip(saved_steps, saved_times)):
            # Get cell density field for this step
            phi_hat = simulation_data["field_data"]["phi_hat"][i]
            
            # Calculate total cell density
            phi_T = np.sum(phi_hat, axis=0)
            
            # Calculate total free energy
            total_energy = self.calculate_total_free_energy(phi_T, dx, m)
            total_energies.append(total_energy)
            valid_times.append(time)
            
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(valid_times, total_energies, linestyle=line_style, color=line_color, 
                linewidth=line_width, marker=marker, markersize=marker_size)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Total Free Energy')
        ax.set_title('Total Free Energy Evolution')
        ax.grid(True, alpha=0.3)
        
        # Add some statistics to the plot
        min_energy = np.min(total_energies)
        max_energy = np.max(total_energies)
        final_energy = total_energies[-1]
        initial_energy = total_energies[0]
        energy_change = final_energy - initial_energy
        
        # Add text box with statistics
        stats_text = f'Initial: {initial_energy:.3f}\nFinal: {final_energy:.3f}\nChange: {energy_change:.3f}\nMin: {min_energy:.3f}\nMax: {max_energy:.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_plot and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = "total_free_energy_evolution.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved total free energy evolution plot to {filepath}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
    
    def calculate_current_total_free_energy(self, m=None):
        """
        Calculate the total free energy for the current state of the field manager.
        
        This method can be called during simulation to track energy in real-time.
        
        Args:
            m: Adhesion energy parameter (if None, uses the one from field_manager config)
            
        Returns:
            total_energy: Total free energy for current state
        """
        if m is None:
            # Try to get from field manager config
            if hasattr(self.field_manager, 'cfg') and self.field_manager.cfg:
                m = self.field_manager.cfg.get("physics", {}).get("adhesion_energy", {}).get("m", 1.0)
            else:
                m = 1.0  # Default value
        
        # Calculate total cell density from current phi_hat
        phi_T = np.sum(self.field_manager.phi_hat, axis=0)
        
        # Calculate total free energy
        return self.calculate_total_free_energy(phi_T, self.dx, m)
    
    def plot_nutrient_diffusion_over_time(self, simulation_data, output_dir=None, save_plot=False, 
                                        show_plot=True, figsize=(15, 10), z_slice=None, 
                                        center_x=None, center_y=None, max_radius=None,
                                        num_radial_bins=20, time_points=None):
        """
        Plot nutrient diffusion over time showing how deep nutrients penetrate into the organoid.
        
        This function creates multiple visualizations:
        1. Radial nutrient concentration profiles at different time points
        2. Penetration depth over time
        3. 2D cross-sections showing nutrient distribution
        
        Args:
            simulation_data: Dictionary containing simulation data
            output_dir: Directory to save plots
            save_plot: Whether to save the plots
            show_plot: Whether to display the plots
            figsize: Figure size tuple
            z_slice: Z-slice to analyze (defaults to center)
            center_x: Center x coordinate for analysis (defaults to domain center)
            center_y: Center y coordinate for analysis (defaults to domain center)
            max_radius: Maximum radius to analyze (defaults to half domain size)
            num_radial_bins: Number of radial bins for averaging
            time_points: List of time point indices to plot (defaults to all saved steps)
        """
        # Get simulation metadata
        saved_steps = simulation_data["metadata"]["saved_steps"]
        saved_times = simulation_data["metadata"]["saved_times"]
        grid_size = simulation_data["metadata"]["grid_size"]
        dx = 1.0  # Default value, should be extracted from config if available
        
        nx, ny, nz = grid_size
        if z_slice is None:
            z_slice = nz // 2
        
        # Set default center coordinates if not provided
        if center_x is None:
            center_x = nx // 2
        if center_y is None:
            center_y = ny // 2
        
        # Set default max radius
        if max_radius is None:
            max_radius = min(nx, ny) // 2
        
        # Set default time points
        if time_points is None:
            time_points = list(range(len(saved_steps)))
        
        # Create coordinate grids
        x = np.arange(nx) * dx
        y = np.arange(ny) * dx
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Calculate radial coordinates from center
        center_x_phys = center_x * dx
        center_y_phys = center_y * dx
        R = np.sqrt((X - center_x_phys)**2 + (Y - center_y_phys)**2)
        
        # Create radial bins
        radial_bins = np.linspace(0, max_radius * dx, num_radial_bins + 1)
        radial_centers = (radial_bins[:-1] + radial_bins[1:]) / 2
        
        # Store data for analysis
        nutrient_profiles = []
        penetration_depths = []
        times = []
        
        # Analyze each time point
        for i in time_points:
            step = saved_steps[i]
            time = saved_times[i]
            
            # Get nutrient field for this time point
            nutrient_field = simulation_data["field_data"]["nutrient_fields"][i]
            nutrient_slice = nutrient_field[:, :, z_slice]
            
            # Calculate radial average of nutrient concentration
            radial_nutrient = np.zeros(num_radial_bins)
            radial_counts = np.zeros(num_radial_bins)
            
            for j in range(num_radial_bins):
                mask = (R >= radial_bins[j]) & (R < radial_bins[j + 1])
                if np.any(mask):
                    radial_nutrient[j] = np.mean(nutrient_slice[mask])
                    radial_counts[j] = np.sum(mask)
            
            # Calculate penetration depth (where nutrient concentration drops to 10% of maximum)
            max_nutrient = np.max(nutrient_slice)
            threshold = 0.1 * max_nutrient
            
            penetration_idx = np.where(radial_nutrient >= threshold)[0]
            if len(penetration_idx) > 0:
                penetration_depth = radial_centers[penetration_idx[-1]]
            else:
                penetration_depth = 0.0
            
            nutrient_profiles.append(radial_nutrient)
            penetration_depths.append(penetration_depth)
            times.append(time)
        
        # Create the main figure with subplots
        fig = plt.figure(figsize=figsize)
        
        # 1. Radial nutrient profiles at different time points
        ax1 = plt.subplot(2, 2, 1)
        colors = plt.cm.viridis(np.linspace(0, 1, len(time_points)))
        
        for i, (profile, time, color) in enumerate(zip(nutrient_profiles, times, colors)):
            ax1.plot(radial_centers, profile, color=color, linewidth=2, 
                    label=f't = {time:.2f}', alpha=0.8)
        
        ax1.set_xlabel('Radial Distance from Center')
        ax1.set_ylabel('Nutrient Concentration')
        ax1.set_title('Radial Nutrient Profiles Over Time')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Penetration depth over time
        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(times, penetration_depths, 'o-', linewidth=2, markersize=6, color='red')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Penetration Depth')
        ax2.set_title('Nutrient Penetration Depth Over Time')
        ax2.grid(True, alpha=0.3)
        
        # Add trend line
        if len(times) > 1:
            z = np.polyfit(times, penetration_depths, 1)
            p = np.poly1d(z)
            ax2.plot(times, p(times), '--', color='red', alpha=0.7, 
                    label=f'Trend: {z[0]:.3f}x + {z[1]:.3f}')
            ax2.legend()
        
        # 3. 2D cross-section at initial time
        ax3 = plt.subplot(2, 2, 3)
        initial_nutrient = simulation_data["field_data"]["nutrient_fields"][time_points[0]]
        initial_slice = initial_nutrient[:, :, z_slice]
        
        im3 = ax3.imshow(initial_slice, extent=[x[0], x[-1], y[0], y[-1]], 
                        origin='lower', cmap='viridis', aspect='equal')
        ax3.set_title(f'Initial Nutrient Distribution (t = {times[0]:.2f})')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        plt.colorbar(im3, ax=ax3, label='Concentration')
        
        # Add center point and radius circles
        ax3.plot(center_x_phys, center_y_phys, 'r+', markersize=10, markeredgewidth=2)
        for r in [penetration_depths[0], max_radius * dx]:
            circle = plt.Circle((center_x_phys, center_y_phys), r, fill=False, 
                              color='red', linestyle='--', alpha=0.7)
            ax3.add_patch(circle)
        
        # 4. 2D cross-section at final time
        ax4 = plt.subplot(2, 2, 4)
        final_nutrient = simulation_data["field_data"]["nutrient_fields"][time_points[-1]]
        final_slice = final_nutrient[:, :, z_slice]
        
        im4 = ax4.imshow(final_slice, extent=[x[0], x[-1], y[0], y[-1]], 
                        origin='lower', cmap='viridis', aspect='equal')
        ax4.set_title(f'Final Nutrient Distribution (t = {times[-1]:.2f})')
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        plt.colorbar(im4, ax=ax4, label='Concentration')
        
        # Add center point and radius circles
        ax4.plot(center_x_phys, center_y_phys, 'r+', markersize=10, markeredgewidth=2)
        for r in [penetration_depths[-1], max_radius * dx]:
            circle = plt.Circle((center_x_phys, center_y_phys), r, fill=False, 
                              color='red', linestyle='--', alpha=0.7)
            ax4.add_patch(circle)
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_plot and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = "nutrient_diffusion_analysis.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved nutrient diffusion analysis plot to {filepath}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        # Print summary statistics
        print(f"\nNutrient Diffusion Analysis Summary:")
        print(f"Initial penetration depth: {penetration_depths[0]:.3f}")
        print(f"Final penetration depth: {penetration_depths[-1]:.3f}")
        print(f"Total change in penetration: {penetration_depths[-1] - penetration_depths[0]:.3f}")
        if len(times) > 1:
            print(f"Average penetration rate: {(penetration_depths[-1] - penetration_depths[0]) / (times[-1] - times[0]):.3f} units/time")
        
        return 
    
    def plot_nutrient_penetration_heatmap(self, simulation_data, output_dir=None, save_plot=False, 
                                        show_plot=True, figsize=(12, 8), z_slice=None,
                                        center_x=None, center_y=None, max_radius=None,
                                        num_radial_bins=20, time_points=None):
        """
        Create a heatmap showing nutrient penetration depth over time and space.
        
        Args:
            simulation_data: Dictionary containing simulation data
            output_dir: Directory to save plots
            save_plot: Whether to save the plots
            show_plot: Whether to display the plots
            figsize: Figure size tuple
            z_slice: Z-slice to analyze (defaults to center)
            center_x: Center x coordinate for analysis
            center_y: Center y coordinate for analysis
            max_radius: Maximum radius to analyze
            num_radial_bins: Number of radial bins for averaging
            time_points: List of time point indices to plot
        """
        # Get simulation metadata
        saved_steps = simulation_data["metadata"]["saved_steps"]
        saved_times = simulation_data["metadata"]["saved_times"]
        grid_size = simulation_data["metadata"]["grid_size"]
        dx = 1.0
        
        nx, ny, nz = grid_size
        if z_slice is None:
            z_slice = nz // 2
        
        # Set default center coordinates if not provided
        if center_x is None:
            center_x = nx // 2
        if center_y is None:
            center_y = ny // 2
        
        # Set default max radius
        if max_radius is None:
            max_radius = min(nx, ny) // 2
        
        # Set default time points
        if time_points is None:
            time_points = list(range(len(saved_steps)))
        
        # Create coordinate grids
        x = np.arange(nx) * dx
        y = np.arange(ny) * dx
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Calculate radial coordinates from center
        center_x_phys = center_x * dx
        center_y_phys = center_y * dx
        R = np.sqrt((X - center_x_phys)**2 + (Y - center_y_phys)**2)
        
        # Create radial bins
        radial_bins = np.linspace(0, max_radius * dx, num_radial_bins + 1)
        radial_centers = (radial_bins[:-1] + radial_bins[1:]) / 2
        
        # Create time-radial heatmap data
        heatmap_data = np.zeros((len(time_points), num_radial_bins))
        
        # Fill heatmap data
        for i, time_idx in enumerate(time_points):
            nutrient_field = simulation_data["field_data"]["nutrient_fields"][time_idx]
            nutrient_slice = nutrient_field[:, :, z_slice]
            
            for j in range(num_radial_bins):
                mask = (R >= radial_bins[j]) & (R < radial_bins[j + 1])
                if np.any(mask):
                    heatmap_data[i, j] = np.mean(nutrient_slice[mask])
        
        # Create the heatmap
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(heatmap_data, aspect='auto', cmap='viridis', 
                      extent=[radial_centers[0], radial_centers[-1], 
                             saved_times[time_points[0]], saved_times[time_points[-1]]],
                      origin='lower')
        
        ax.set_xlabel('Radial Distance from Center')
        ax.set_ylabel('Time')
        ax.set_title('Nutrient Concentration Heatmap: Time vs. Radial Distance')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Nutrient Concentration')
        
        # Add contour lines for penetration depth
        threshold = 0.1 * np.max(heatmap_data)
        contour_levels = [threshold]
        ax.contour(heatmap_data, levels=contour_levels, colors='red', 
                  linewidths=2, alpha=0.8,
                  extent=[radial_centers[0], radial_centers[-1], 
                         saved_times[time_points[0]], saved_times[time_points[-1]]],
                  origin='lower')
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_plot and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = "nutrient_penetration_heatmap.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved nutrient penetration heatmap to {filepath}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return heatmap_data
    
    def plot_nutrient_penetration_simple(self, step=0, z_slice=None, center_x=None, center_y=None, 
                                       max_radius=None, num_radial_bins=20, output_dir=None, 
                                       save_plot=False, show_plot=True, figsize=(12, 5)):
        """
        Simple function to plot current nutrient penetration depth for real-time analysis.
        
        Args:
            step: Current simulation step for filename
            z_slice: Z-slice to analyze (defaults to center)
            center_x: Center x coordinate for analysis (defaults to domain center)
            center_y: Center y coordinate for analysis (defaults to domain center)
            max_radius: Maximum radius to analyze (defaults to half domain size)
            num_radial_bins: Number of radial bins for averaging
            output_dir: Directory to save plots
            save_plot: Whether to save the plots
            show_plot: Whether to display the plots
            figsize: Figure size tuple
        """
        nx, ny, nz = self.grid
        if z_slice is None:
            z_slice = nz // 2
        
        # Set default center coordinates if not provided
        if center_x is None:
            center_x = nx // 2
        if center_y is None:
            center_y = ny // 2
        
        # Set default max radius
        if max_radius is None:
            max_radius = min(nx, ny) // 2
        
        # Create coordinate grids
        x = np.arange(nx) * self.dx
        y = np.arange(ny) * self.dx
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Calculate radial coordinates from center
        center_x_phys = center_x * self.dx
        center_y_phys = center_y * self.dx
        R = np.sqrt((X - center_x_phys)**2 + (Y - center_y_phys)**2)
        
        # Create radial bins
        radial_bins = np.linspace(0, max_radius * self.dx, num_radial_bins + 1)
        radial_centers = (radial_bins[:-1] + radial_bins[1:]) / 2
        
        # Get current nutrient field
        nutrient_slice = self.field_manager.nutrient_field[:, :, z_slice]
        
        # Calculate radial average of nutrient concentration
        radial_nutrient = np.zeros(num_radial_bins)
        for j in range(num_radial_bins):
            mask = (R >= radial_bins[j]) & (R < radial_bins[j + 1])
            if np.any(mask):
                radial_nutrient[j] = np.mean(nutrient_slice[mask])
        
        # Calculate penetration depth (where nutrient concentration drops to 10% of maximum)
        max_nutrient = np.max(nutrient_slice)
        threshold = 0.1 * max_nutrient
        
        penetration_idx = np.where(radial_nutrient >= threshold)[0]
        if len(penetration_idx) > 0:
            penetration_depth = radial_centers[penetration_idx[-1]]
        else:
            penetration_depth = 0.0
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 1. Radial nutrient profile
        ax1.plot(radial_centers, radial_nutrient, 'b-', linewidth=2, label='Nutrient Concentration')
        ax1.axhline(y=threshold, color='r', linestyle='--', alpha=0.7, label=f'Threshold ({threshold:.3f})')
        ax1.axvline(x=penetration_depth, color='g', linestyle='--', alpha=0.7, 
                   label=f'Penetration Depth ({penetration_depth:.3f})')
        ax1.set_xlabel('Radial Distance from Center')
        ax1.set_ylabel('Nutrient Concentration')
        ax1.set_title(f'Radial Nutrient Profile (Step {step})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 2D cross-section with penetration depth overlay
        im = ax2.imshow(nutrient_slice, extent=[x[0], x[-1], y[0], y[-1]], 
                       origin='lower', cmap='viridis', aspect='equal')
        ax2.set_title(f'Nutrient Distribution (z={z_slice})')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        plt.colorbar(im, ax=ax2, label='Concentration')
        
        # Add center point and penetration depth circle
        ax2.plot(center_x_phys, center_y_phys, 'r+', markersize=10, markeredgewidth=2)
        circle = plt.Circle((center_x_phys, center_y_phys), penetration_depth, fill=False, 
                          color='red', linestyle='--', alpha=0.8, linewidth=2)
        ax2.add_patch(circle)
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_plot and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f"nutrient_penetration_step_{step:06d}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved nutrient penetration plot to {filepath}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        # Print penetration depth
        print(f"Step {step}: Nutrient penetration depth = {penetration_depth:.3f}")
        
        return penetration_depth
    
    def plot_nutrient_time_series_at_depths(self, simulation_data, depths=None, output_dir=None, 
                                          save_plot=False, show_plot=True, figsize=(12, 8),
                                          z_slice=None, center_x=None, center_y=None):
        """
        Plot nutrient concentration time series at different radial depths.
        
        Args:
            simulation_data: Dictionary containing simulation data
            depths: List of radial depths to analyze (defaults to [0.1, 0.3, 0.5, 0.7, 0.9] of max radius)
            output_dir: Directory to save plots
            save_plot: Whether to save the plots
            show_plot: Whether to display the plots
            figsize: Figure size tuple
            z_slice: Z-slice to analyze (defaults to center)
            center_x: Center x coordinate for analysis
            center_y: Center y coordinate for analysis
        """
        # Get simulation metadata
        saved_steps = simulation_data["metadata"]["saved_steps"]
        saved_times = simulation_data["metadata"]["saved_times"]
        grid_size = simulation_data["metadata"]["grid_size"]
        dx = 1.0
        
        nx, ny, nz = grid_size
        if z_slice is None:
            z_slice = nz // 2
        
        # Set default center coordinates if not provided
        if center_x is None:
            center_x = nx // 2
        if center_y is None:
            center_y = ny // 2
        
        # Set default depths
        if depths is None:
            max_radius = min(nx, ny) // 2 * dx
            depths = [0.1 * max_radius, 0.3 * max_radius, 0.5 * max_radius, 
                    0.7 * max_radius, 0.9 * max_radius]
        
        # Create coordinate grids
        x = np.arange(nx) * dx
        y = np.arange(ny) * dx
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Calculate radial coordinates from center
        center_x_phys = center_x * dx
        center_y_phys = center_y * dx
        R = np.sqrt((X - center_x_phys)**2 + (Y - center_y_phys)**2)
        
        # Store time series data for each depth
        time_series_data = {depth: [] for depth in depths}
        
        # Analyze each time point
        for i, (step, time) in enumerate(zip(saved_steps, saved_times)):
            # Get nutrient field for this time point
            nutrient_field = simulation_data["field_data"]["nutrient_fields"][i]
            nutrient_slice = nutrient_field[:, :, z_slice]
            
            # Calculate average nutrient concentration at each depth
            for depth in depths:
                # Find points within a small range around the target depth
                depth_tolerance = 0.1 * dx  # Small tolerance for depth matching
                mask = (R >= depth - depth_tolerance) & (R < depth + depth_tolerance)
                
                if np.any(mask):
                    avg_concentration = np.mean(nutrient_slice[mask])
                else:
                    avg_concentration = 0.0
                
                time_series_data[depth].append(avg_concentration)
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # 1. Time series at different depths
        colors = plt.cm.viridis(np.linspace(0, 1, len(depths)))
        
        for depth, color in zip(depths, colors):
            ax1.plot(saved_times, time_series_data[depth], color=color, linewidth=2, 
                    label=f'Depth = {depth:.2f}', marker='o', markersize=4)
        
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Nutrient Concentration')
        ax1.set_title('Nutrient Concentration Time Series at Different Depths')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Penetration depth over time (where concentration drops below threshold)
        threshold = 0.1  # 10% of maximum concentration
        penetration_depths = []
        
        for i, time in enumerate(saved_times):
            # Find the deepest point where concentration is above threshold
            max_penetration = 0.0
            for depth in depths:
                if time_series_data[depth][i] >= threshold:
                    max_penetration = depth
            
            penetration_depths.append(max_penetration)
        
        ax2.plot(saved_times, penetration_depths, 'r-o', linewidth=2, markersize=6)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Penetration Depth')
        ax2.set_title('Nutrient Penetration Depth Over Time')
        ax2.grid(True, alpha=0.3)
        
        # Add trend line
        if len(saved_times) > 1:
            z = np.polyfit(saved_times, penetration_depths, 1)
            p = np.poly1d(z)
            ax2.plot(saved_times, p(saved_times), '--', color='red', alpha=0.7, 
                    label=f'Trend: {z[0]:.3f}x + {z[1]:.3f}')
            ax2.legend()
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_plot and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = "nutrient_time_series_at_depths.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved nutrient time series plot to {filepath}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return