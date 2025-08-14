import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

class CellFieldPlotter:
    def __init__(self, field_manager_or_simulation_data, step_idx=0, simulator=None):
        """
        Initialize the cell field plotter.
        
        Args:
            field_manager_or_simulation_data: Either a FieldManager instance or simulation data dictionary
            step_idx: Index of the saved step to use for plotting (only used if simulation_data is provided)
            simulator: Optional simulator instance to get correct population labels
        """
        # Check if we're getting simulation data or field manager
        if isinstance(field_manager_or_simulation_data, dict):
            # Create a mock field manager from simulation data
            simulation_data = field_manager_or_simulation_data
            self.field_manager = self._create_mock_field_manager(simulation_data, step_idx, simulator)
        else:
            # Use the provided field manager
            self.field_manager = field_manager_or_simulation_data
        
        self.grid = self.field_manager.grid
        self.dx = self.field_manager.dx
        self.labels = self.field_manager.labels
        self.M = self.field_manager.M
    
    def _create_mock_field_manager(self, simulation_data, step_idx=0, simulator=None):
        """
        Create a mock field manager from saved simulation data.
        
        Args:
            simulation_data: Dictionary containing simulation data
            step_idx: Index of the saved step to use for plotting
            simulator: Optional simulator instance to get correct population labels
            
        Returns:
            mock_field_manager: Mock field manager with loaded data
        """
        class MockFieldManager:
            def __init__(self, data, step_idx, simulator=None):
                self.grid = data["metadata"]["grid_size"]
                self.dx = 1.0  # Default value, should be extracted from config
                
                # Get population labels from simulator if available (most reliable)
                if simulator is not None and hasattr(simulator, 'field_manager'):
                    self.labels = simulator.field_manager.labels
                # Try to get population labels from metadata
                elif "population_labels" in data["metadata"]:
                    self.labels = data["metadata"]["population_labels"]
                # Try to extract from config if available
                elif "config" in data["metadata"]:
                    config = data["metadata"]["config"]
                    if "populations" in config:
                        self.labels = [p["label"] for p in config["populations"].values()]
                    else:
                        self.labels = [f"Population_{i}" for i in range(data["metadata"]["num_populations"])]
                else:
                    self.labels = [f"Population_{i}" for i in range(data["metadata"]["num_populations"])]
                
                self.M = data["metadata"]["num_populations"]
                
                # Load field data
                self.phi_hat = data["field_data"]["phi_hat"][step_idx]
                if "nutrient_fields" in data["field_data"]:
                    self.nutrient_field = data["field_data"]["nutrient_fields"][step_idx]
                else:
                    self.nutrient_field = None
        
        return MockFieldManager(simulation_data, step_idx, simulator)
    
    def plot_cell_density_field(self, phi_hat, population_idx=None, step=0, z_slice=None, 
                               cmap="viridis", output_dir=None, save_plot=False, 
                               show_plot=True, add_contours=False, contour_colors=['white', 'yellow', 'red'],
                               contour_levels=[0.1, 0.3, 0.5], contour_linewidths=[2, 1.5, 1],
                               center_x=None, center_y=None, zoom_factor=1.0):
        """
        Plot cell density field for a specific population or total density.
        
        Args:
            phi_hat: Cell fraction fields (M, nx, ny, nz)
            population_idx: Index of population to plot (None for total density)
            step: Current simulation step for filename
            z_slice: Z-slice to plot (defaults to center)
            cmap: Colormap for the plot
            output_dir: Directory to save plot
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            add_contours: Whether to add density contours
            contour_colors: List of colors for each contour level
            contour_levels: List of density levels for contours
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
        
        # Create coordinate grids for zoomed region
        x_region = np.arange(x_start, x_end) * self.dx
        y_region = np.arange(y_start, y_end) * self.dx
        X_region, Y_region = np.meshgrid(x_region, y_region, indexing='ij')
        
        # Extract density data for zoomed region
        if population_idx is None:
            # Plot total density (sum of all populations)
            density_slice = np.sum(phi_hat, axis=0)[x_start:x_end, y_start:y_end, z_slice]
            population_name = "Total"
        else:
            # Plot specific population
            if population_idx >= self.M:
                print(f"Error: Population index {population_idx} out of range (0-{self.M-1})")
                return
            density_slice = phi_hat[population_idx, x_start:x_end, y_start:y_end, z_slice]
            population_name = self.labels[population_idx]
        
        # Create plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        
        im = ax.imshow(density_slice, extent=[x_region[0], x_region[-1], y_region[0], y_region[-1]], 
                      origin='lower', cmap=cmap, aspect='equal')
        
        # Add contours if requested
        if add_contours:
            for i, level in enumerate(contour_levels):
                color = contour_colors[i] if i < len(contour_colors) else 'black'
                linewidth = contour_linewidths[i] if i < len(contour_linewidths) else 1
                ax.contour(density_slice, levels=[level], colors=color, 
                          linewidths=linewidth, alpha=0.8,
                          extent=[x_region[0], x_region[-1], y_region[0], y_region[-1]], 
                          origin='lower')
        
        # Add center point marker if zoomed
        if zoom_factor > 1.0:
            center_x_coord = center_x * self.dx
            center_y_coord = center_y * self.dx
            ax.plot(center_x_coord, center_y_coord, 'k+', markersize=10, markeredgewidth=2, 
                   label=f'Center: ({center_x_coord:.2f}, {center_y_coord:.2f})')
            ax.legend()
            ax.set_title(f'Cell Density - {population_name} (z={z_slice}) - Zoomed View')
        else:
            ax.set_title(f'Cell Density - {population_name} (z={z_slice})')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        fig.colorbar(im, ax=ax, label='Cell Density')
        
        plt.tight_layout()
        
        if save_plot and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            if population_idx is None:
                filename = f'cell_density_total_step_{step:06d}.png'
            else:
                filename = f'cell_density_{population_name}_step_{step:06d}.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            print(f"Cell density plot saved to {output_dir}/{filename}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_population_by_label(self, simulation_data, label, step_idx=0, z_slice=None, 
                                cmap="viridis", output_dir=None, save_plot=False, 
                                show_plot=True, add_contours=False, contour_colors=['white', 'yellow', 'red'],
                                contour_levels=[0.1, 0.3, 0.5], contour_linewidths=[2, 1.5, 1],
                                center_x=None, center_y=None, zoom_factor=1.0):
        """
        Plot cell density field for a specific population by label from simulation data.
        
        Args:
            simulation_data: Dictionary containing simulation data
            label: Label of the population to plot
            step_idx: Index of the saved step to use (0 for first saved step)
            z_slice: Z-slice to plot (defaults to center)
            cmap: Colormap for the plot
            output_dir: Directory to save plot
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            add_contours: Whether to add density contours
            contour_colors: List of colors for each contour level
            contour_levels: List of density levels for contours
            contour_linewidths: List of line widths for each contour level
            center_x: Center x coordinate for zoom
            center_y: Center y coordinate for zoom
            zoom_factor: Zoom factor (1.0 = no zoom)
        """
        # Get step information
        step = simulation_data["metadata"]["saved_steps"][step_idx]
        time = simulation_data["metadata"]["saved_times"][step_idx]
        
        print(f"Plotting population '{label}' for step {step} (time={time:.3f})...")
        
        # Get phi_hat for the specified step
        phi_hat = simulation_data["field_data"]["phi_hat"][step_idx]
        
        # Find population index by label
        try:
            population_idx = self.labels.index(label)
            print(f"Found population '{label}' at index {population_idx}")
        except ValueError:
            print(f"Error: Population label '{label}' not found. Available labels: {self.labels}")
            return
        
        # Call the main plotting function
        self.plot_cell_density_field(
            phi_hat=phi_hat, population_idx=population_idx, step=step, z_slice=z_slice,
            cmap=cmap, output_dir=output_dir, save_plot=save_plot, show_plot=show_plot,
            add_contours=add_contours, contour_colors=contour_colors,
            contour_levels=contour_levels, contour_linewidths=contour_linewidths,
            center_x=center_x, center_y=center_y, zoom_factor=zoom_factor
        )
    
    def plot_total_density(self, simulation_data, step_idx=0, z_slice=None, 
                          cmap="viridis", output_dir=None, save_plot=False, 
                          show_plot=True, add_contours=False, contour_colors=['white', 'yellow', 'red'],
                          contour_levels=[0.1, 0.3, 0.5], contour_linewidths=[2, 1.5, 1],
                          center_x=None, center_y=None, zoom_factor=1.0):
        """
        Plot total cell density (sum of all populations) from simulation data.
        
        Args:
            simulation_data: Dictionary containing simulation data
            step_idx: Index of the saved step to use (0 for first saved step)
            z_slice: Z-slice to plot (defaults to center)
            cmap: Colormap for the plot
            output_dir: Directory to save plot
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            add_contours: Whether to add density contours
            contour_colors: List of colors for each contour level
            contour_levels: List of density levels for contours
            contour_linewidths: List of line widths for each contour level
            center_x: Center x coordinate for zoom
            center_y: Center y coordinate for zoom
            zoom_factor: Zoom factor (1.0 = no zoom)
        """
        # Get step information
        step = simulation_data["metadata"]["saved_steps"][step_idx]
        time = simulation_data["metadata"]["saved_times"][step_idx]
        
        print(f"Plotting total density for step {step} (time={time:.3f})...")
        
        # Get phi_hat for the specified step
        phi_hat = simulation_data["field_data"]["phi_hat"][step_idx]
        
        # Call the main plotting function with population_idx=None for total density
        self.plot_cell_density_field(
            phi_hat=phi_hat, population_idx=None, step=step, z_slice=z_slice,
            cmap=cmap, output_dir=output_dir, save_plot=save_plot, show_plot=show_plot,
            add_contours=add_contours, contour_colors=contour_colors,
            contour_levels=contour_levels, contour_linewidths=contour_linewidths,
            center_x=center_x, center_y=center_y, zoom_factor=zoom_factor
        )
    
    def plot_population_evolution(self, simulation_data, population_idx=None, label=None,
                                 step_indices=None, z_slice=None, cmap="viridis", 
                                 output_dir=None, save_plot=False, show_plot=True,
                                 add_contours=False, contour_colors=['white', 'yellow', 'red'],
                                 contour_levels=[0.1, 0.3, 0.5], contour_linewidths=[2, 1.5, 1],
                                 center_x=None, center_y=None, zoom_factor=1.0,
                                 figsize=(15, 10), max_plots_per_row=3):
        """
        Plot evolution of a single population over multiple saved simulation steps.
        
        Args:
            simulation_data: Dictionary containing simulation data
            population_idx: Index of population to plot (None for total density)
            label: Label of population to plot (alternative to population_idx)
            step_indices: List of step indices to plot (None for all available)
            z_slice: Z-slice to plot (defaults to center)
            cmap: Colormap for the plot
            output_dir: Directory to save plot
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            add_contours: Whether to add density contours
            contour_colors: List of colors for each contour level
            contour_levels: List of density levels for contours
            contour_linewidths: List of line widths for each contour level
            center_x: Center x coordinate for zoom
            center_y: Center y coordinate for zoom
            zoom_factor: Zoom factor (1.0 = no zoom)
            figsize: Figure size for the subplot grid
            max_plots_per_row: Maximum number of plots per row
        """
        # Determine population index
        if label is not None:
            try:
                population_idx = self.labels.index(label)
            except ValueError:
                print(f"Error: Population label '{label}' not found. Available labels: {self.labels}")
                return
        
        # Get available step indices if not provided
        if step_indices is None:
            step_indices = list(range(len(simulation_data["metadata"]["saved_steps"])))
        
        # Limit number of plots to avoid overcrowding
        if len(step_indices) > 12:
            print(f"Warning: Limiting plots to 12 steps (requested {len(step_indices)})")
            step_indices = step_indices[:12]
        
        # Calculate subplot layout
        num_plots = len(step_indices)
        num_cols = min(max_plots_per_row, num_plots)
        num_rows = (num_plots + num_cols - 1) // num_cols
        
        # Create figure
        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
        if num_plots == 1:
            axes_flat = [axes]
        elif num_rows == 1:
            axes_flat = axes.reshape(1, -1).flatten()
        elif num_cols == 1:
            axes_flat = axes.reshape(-1, 1).flatten()
        else:
            axes_flat = axes.flatten()
        
        # Set default z_slice if not provided
        if z_slice is None:
            z_slice = self.grid[2] // 2
        
        # Get population name for title
        if population_idx is None:
            population_name = "Total"
        else:
            population_name = self.labels[population_idx]
        
        print(f"Plotting evolution for population: {population_name} (index: {population_idx})")
        print(f"Available population labels: {self.labels}")
        print(f"Step indices: {step_indices}")
        
        # Set default center coordinates if not provided
        if center_x is None:
            center_x = self.grid[0] // 2
        if center_y is None:
            center_y = self.grid[1] // 2
        
        # Convert to integers for slice indices
        center_x = int(center_x)
        center_y = int(center_y)
        
        # Calculate zoom window size
        window_size = min(self.grid[0], self.grid[1]) // zoom_factor
        half_window = window_size // 2
        
        # Calculate window boundaries
        x_start = int(max(0, center_x - half_window))
        x_end = int(min(self.grid[0], center_x + half_window))
        y_start = int(max(0, center_y - half_window))
        y_end = int(min(self.grid[1], center_y + half_window))
        
        # Create coordinate grids for zoomed region
        x_region = np.arange(x_start, x_end) * self.dx
        y_region = np.arange(y_start, y_end) * self.dx
        
        # Find global min/max for consistent color scaling
        vmin, vmax = 0, 1
        
        # Plot each step
        for i, step_idx in enumerate(step_indices):
            ax = axes_flat[i]
            
            # Get step information
            step = simulation_data["metadata"]["saved_steps"][step_idx]
            time = simulation_data["metadata"]["saved_times"][step_idx]
            
            # Get phi_hat for this step
            phi_hat = simulation_data["field_data"]["phi_hat"][step_idx]
            
            # Extract density data for zoomed region
            if population_idx is None:
                # Plot total density (sum of all populations)
                density_slice = np.sum(phi_hat, axis=0)[x_start:x_end, y_start:y_end, z_slice]
            else:
                # Plot specific population
                density_slice = phi_hat[population_idx, x_start:x_end, y_start:y_end, z_slice]
            
            # Create the plot with consistent color scaling
            im = ax.imshow(density_slice, extent=[x_region[0], x_region[-1], y_region[0], y_region[-1]], 
                          origin='lower', cmap=cmap, aspect='equal', vmin=vmin, vmax=vmax)
            
            # Add contours if requested
            if add_contours:
                for j, level in enumerate(contour_levels):
                    color = contour_colors[j] if j < len(contour_colors) else 'black'
                    linewidth = contour_linewidths[j] if j < len(contour_linewidths) else 1
                    ax.contour(density_slice, levels=[level], colors=color, 
                              linewidths=linewidth, alpha=0.8,
                              extent=[x_region[0], x_region[-1], y_region[0], y_region[-1]], 
                              origin='lower')
            
            # Add center point marker if zoomed
            if zoom_factor > 1.0:
                center_x_coord = center_x * self.dx
                center_y_coord = center_y * self.dx
                ax.plot(center_x_coord, center_y_coord, 'k+', markersize=8, markeredgewidth=1.5)
            
            # Set title and labels
            ax.set_title(f'Step {step} (t={time:.2f})')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            
            # Add colorbar to the last plot
            if i == num_plots - 1:
                fig.colorbar(im, ax=ax, label='Cell Density')
        
        # Hide unused subplots
        for i in range(num_plots, len(axes_flat)):
            axes_flat[i].set_visible(False)
        
        # Set overall title
        fig.suptitle(f'Population Evolution - {population_name} (z={z_slice})', fontsize=16)
        
        plt.tight_layout()
        
        if save_plot and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            if population_idx is None:
                filename = f'population_evolution_total_steps_{step_indices[0]}-{step_indices[-1]}.png'
            else:
                filename = f'population_evolution_{population_name}_steps_{step_indices[0]}-{step_indices[-1]}.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            print(f"Population evolution plot saved to {output_dir}/{filename}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_population_evolution_by_label(self, simulation_data, label, step_indices=None,
                                          z_slice=None, cmap="viridis", output_dir=None, 
                                          save_plot=False, show_plot=True, add_contours=False,
                                          contour_colors=['white', 'yellow', 'red'],
                                          contour_levels=[0.1, 0.3, 0.5], contour_linewidths=[2, 1.5, 1],
                                          center_x=None, center_y=None, zoom_factor=1.0,
                                          figsize=(15, 10), max_plots_per_row=3):
        """
        Plot evolution of a single population by label over multiple saved simulation steps.
        
        Args:
            simulation_data: Dictionary containing simulation data
            label: Label of population to plot
            step_indices: List of step indices to plot (None for all available)
            z_slice: Z-slice to plot (defaults to center)
            cmap: Colormap for the plot
            output_dir: Directory to save plot
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            add_contours: Whether to add density contours
            contour_colors: List of colors for each contour level
            contour_levels: List of density levels for contours
            contour_linewidths: List of line widths for each contour level
            center_x: Center x coordinate for zoom
            center_y: Center y coordinate for zoom
            zoom_factor: Zoom factor (1.0 = no zoom)
            figsize: Figure size for the subplot grid
            max_plots_per_row: Maximum number of plots per row
        """
        # Find population index by label
        try:
            population_idx = self.labels.index(label)
        except ValueError:
            print(f"Error: Population label '{label}' not found. Available labels: {self.labels}")
            return
        
        # Call the main evolution plotting function
        self.plot_population_evolution(
            simulation_data=simulation_data, population_idx=population_idx, label=None,
            step_indices=step_indices, z_slice=z_slice, cmap=cmap, output_dir=output_dir,
            save_plot=save_plot, show_plot=show_plot, add_contours=add_contours,
            contour_colors=contour_colors, contour_levels=contour_levels,
            contour_linewidths=contour_linewidths, center_x=center_x, center_y=center_y,
            zoom_factor=zoom_factor, figsize=figsize, max_plots_per_row=max_plots_per_row
        )
    
    @classmethod
    def from_simulation_data(cls, simulation_data, step_idx=0):
        """
        Create a plotter instance from saved simulation data.
        
        Args:
            simulation_data: Dictionary containing simulation data
            step_idx: Index of the saved step to use for plotting
            
        Returns:
            plotter: CellFieldPlotter instance with loaded data
        """
        return cls(simulation_data, step_idx)
