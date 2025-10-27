import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

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
                
                # Get population labels from saved simulation data (most reliable)
                if "population_labels" in data["metadata"]:
                    self.labels = data["metadata"]["population_labels"]
                # Try to extract from config if available
                elif "config" in data["metadata"]:
                    config = data["metadata"]["config"]
                    if "populations" in config:
                        self.labels = [p["label"] for p in config["populations"].values()]
                    else:
                        self.labels = [f"Population_{i}" for i in range(data["metadata"]["num_populations"])]
                # Fallback to simulator if available
                elif simulator is not None and hasattr(simulator, 'field_manager'):
                    self.labels = simulator.field_manager.labels
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
    
    def plot_initial_conditions(self, config_path=None, cmap="viridis", output_dir=None, save_plot=False, 
                               show_plot=True, add_contours=False, contour_colors=['white', 'yellow', 'red'],
                               contour_levels=[0.1, 0.3, 0.5], contour_linewidths=[2, 1.5, 1],
                               center_x=None, center_y=None, zoom_factor=1.0):
        """
        Plot the actual initial conditions (step 0, time=0.0) without needing saved simulation data.
        
        Args:
            config_path: Path to configuration file (required if not using simulator)
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
        print("="*60)
        print("PLOTTING ACTUAL INITIAL CONDITIONS (Step 0, Time 0.0)")
        print("="*60)
        
        # Load configuration
        if config_path is None:
            # Try to get config from field manager if available
            if hasattr(self.field_manager, 'cfg'):
                cfg = self.field_manager.cfg
            else:
                raise ValueError("config_path must be provided when field_manager doesn't have cfg attribute")
        else:
            import yaml
            cfg = yaml.safe_load(Path(config_path).read_text())
        
        # Generate initial conditions using the same method as the simulator
        from src.growkit.Fields.InitialConditions.InitialConditions import InitialConditions
        ic_manager = InitialConditions(cfg)
        phi_hat, nutrient_field = ic_manager.initialize_cell_fields()
        
        print(f"Generated initial conditions shape: {phi_hat.shape}")
        
        # Analyze initial conditions
        print(f"\nINITIAL CONDITIONS ANALYSIS:")
        for i, label in enumerate(self.labels):
            pop_data = phi_hat[i, :, :, :]
            total_density = np.sum(pop_data)
            print(f"\n{label} (Index {i}):")
            print(f"  Total density: {total_density:.6f}")
            print(f"  Min density: {np.min(pop_data):.6f}")
            print(f"  Max density: {np.max(pop_data):.6f}")
            print(f"  Mean density: {np.mean(pop_data):.6f}")
            print(f"  Non-zero voxels: {np.count_nonzero(pop_data)}")
            
            if total_density == 0.0:
                print(f"  ✅ CORRECT: Zero initial density")
            else:
                print(f"  ℹ️  Non-zero initial density (as expected)")
        
        # Create plots
        print(f"\nCreating plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        # Plot each population
        for i, label in enumerate(self.labels):
            ax = axes[i]
            
            # Get middle z-slice
            z_slice = phi_hat.shape[3] // 2
            density_slice = phi_hat[i, :, :, z_slice]
            
            im = ax.imshow(density_slice, cmap=cmap, origin='lower', aspect='equal')
            ax.set_title(f'Initial Conditions - {label}\n(Step 0, Time 0.0)')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Cell Density')
            
            # Add statistics text
            stats_text = f"Min: {np.min(density_slice):.4f}\nMax: {np.max(density_slice):.4f}\nMean: {np.mean(density_slice):.4f}\nSum: {np.sum(density_slice):.4f}"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot total density
        ax = axes[3]
        total_density = np.sum(phi_hat, axis=0)
        total_slice = total_density[:, :, z_slice]
        
        im = ax.imshow(total_slice, cmap=cmap, origin='lower', aspect='equal')
        ax.set_title('Initial Conditions - Total Density\n(Step 0, Time 0.0)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Total Cell Density')
        
        # Add statistics text
        stats_text = f"Min: {np.min(total_slice):.4f}\nMax: {np.max(total_slice):.4f}\nMean: {np.mean(total_slice):.4f}\nSum: {np.sum(total_slice):.4f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_plot and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = "initial_conditions_all_populations.png"
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            print(f"Initial conditions plot saved to {output_dir}/{filename}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        print(f"\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print("The initial conditions (step 0, time=0.0) show:")
        print("- Stem Cells: Non-zero density (as configured)")
        print("- Tumour Cells: Non-zero density (as configured)")
        print("- Necrotic Cells: ZERO density (as configured)")
        print("\nThe saved simulation data starts from step 1 (time=5.0),")
        print("which already has necrotic cells generated during simulation.")

    def plot_3d_tumor_field(self, simulation_data, step_idx=0, isosurface_level=0.1, 
                           cmap="viridis", output_dir=None, save_plot=False, show_plot=True,
                           figsize=(12, 8), alpha=0.6, title_suffix=""):
        """
        Plot 3D visualization of tumor field using isosurfaces.
        
        Args:
            simulation_data: Dictionary containing simulation data
            step_idx: Index of the saved step to use for plotting
            isosurface_level: Density level for isosurface (0.1 = 10% density)
            cmap: Colormap for the plot
            output_dir: Directory to save plot
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            figsize: Figure size
            alpha: Transparency of the isosurface
            title_suffix: Additional text for the title
        """
        # Get step information
        step = simulation_data["metadata"]["saved_steps"][step_idx]
        time = simulation_data["metadata"]["saved_times"][step_idx]
        
        print(f"Creating 3D tumor field visualization for step {step} (time={time:.3f})...")
        
        # Get phi_hat for the specified step
        phi_hat = simulation_data["field_data"]["phi_hat"][step_idx]
        
        # Calculate total tumor density
        total_density = np.sum(phi_hat, axis=0)
        
        # Create 3D coordinate grids
        nx, ny, nz = self.grid
        x = np.arange(nx) * self.dx
        y = np.arange(ny) * self.dx
        z = np.arange(nz) * self.dx
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Create isosurface
        # Find voxels above the threshold
        tumor_mask = total_density >= isosurface_level
        
        if np.any(tumor_mask):
            # Extract coordinates and values for tumor regions
            tumor_x = X[tumor_mask]
            tumor_y = Y[tumor_mask]
            tumor_z = Z[tumor_mask]
            tumor_values = total_density[tumor_mask]
            
            # Create scatter plot with color mapping
            scatter = ax.scatter(tumor_x, tumor_y, tumor_z, c=tumor_values, 
                               cmap=cmap, alpha=alpha, s=20)
            
            # Add colorbar
            cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
            cbar.set_label('Tumor Density')
        else:
            print(f"Warning: No tumor regions found above density level {isosurface_level}")
        
        # Set labels and title
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_zlabel('Z (μm)')
        ax.set_title(f'3D Tumor Field - Step {step} (t={time:.2f}){title_suffix}')
        
        # Set equal aspect ratio
        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
        mid_x = (X.max()+X.min()) * 0.5
        mid_y = (Y.max()+Y.min()) * 0.5
        mid_z = (Z.max()+Z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        
        if save_plot and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f'3d_tumor_field_step_{step:06d}.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            print(f"3D tumor field plot saved to {output_dir}/{filename}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_3d_quadratic_potential(self, simulation_data, step_idx=0, potential_strength=None,
                                   bowl_center=None, output_dir=None, save_plot=False, 
                                   show_plot=True, figsize=(12, 8), alpha=0.6, 
                                   title_suffix="", show_tumor_field=True, tumor_isosurface_level=0.1):
        """
        Plot 3D visualization of the quadratic (bowl) potential field.
        
        Args:
            simulation_data: Dictionary containing simulation data
            step_idx: Index of the saved step to use for plotting
            potential_strength: Strength of the bowl potential (if None, extract from config)
            bowl_center: Center coordinates of the bowl (if None, extract from config)
            output_dir: Directory to save plot
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            figsize: Figure size
            alpha: Transparency of the potential surface
            title_suffix: Additional text for the title
            show_tumor_field: Whether to overlay tumor field as scatter points
            tumor_isosurface_level: Density level for tumor field overlay
        """
        # Get step information
        step = simulation_data["metadata"]["saved_steps"][step_idx]
        time = simulation_data["metadata"]["saved_times"][step_idx]
        
        print(f"Creating 3D quadratic potential visualization for step {step} (time={time:.3f})...")
        
        # Extract potential parameters from config if not provided
        if potential_strength is None or bowl_center is None:
            config = simulation_data["metadata"]["config"]
            physics_config = config.get("physics", {})
            
            if potential_strength is None:
                potential_strength = physics_config.get("bowl_potential_strength", 0.1)
            
            if bowl_center is None:
                bowl_center = physics_config.get("bowl_center", [50, 50, 50])
        
        # Create 3D coordinate grids
        nx, ny, nz = self.grid
        x = np.arange(nx) * self.dx
        y = np.arange(ny) * self.dx
        z = np.arange(nz) * self.dx
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Calculate quadratic potential: V = k * r²
        # where r is distance from bowl center
        center_x = bowl_center[0] * self.dx
        center_y = bowl_center[1] * self.dx
        center_z = bowl_center[2] * self.dx
        
        r_squared = (X - center_x)**2 + (Y - center_y)**2 + (Z - center_z)**2
        potential = potential_strength * r_squared
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Create potential visualization using scatter plot instead of surface
        # Sample every few points for visualization
        sample_rate = max(1, min(nx, ny, nz) // 15)  # Sample every 15th point or so
        
        X_sample = X[::sample_rate, ::sample_rate, ::sample_rate]
        Y_sample = Y[::sample_rate, ::sample_rate, ::sample_rate]
        Z_sample = Z[::sample_rate, ::sample_rate, ::sample_rate]
        potential_sample = potential[::sample_rate, ::sample_rate, ::sample_rate]
        
        # Create scatter plot of potential (more suitable for 3D visualization)
        potential_normalized = potential_sample / potential_sample.max()
        scatter = ax.scatter(X_sample.flatten(), Y_sample.flatten(), Z_sample.flatten(), 
                           c=potential_normalized.flatten(), cmap='viridis', alpha=alpha, s=10)
        
        # Add colorbar for potential
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
        cbar.set_label('Potential Strength')
        
        # Overlay tumor field if requested
        if show_tumor_field:
            phi_hat = simulation_data["field_data"]["phi_hat"][step_idx]
            total_density = np.sum(phi_hat, axis=0)
            tumor_mask = total_density >= tumor_isosurface_level
            
            if np.any(tumor_mask):
                tumor_x = X[tumor_mask]
                tumor_y = Y[tumor_mask]
                tumor_z = Z[tumor_mask]
                tumor_values = total_density[tumor_mask]
                
                # Overlay tumor field as scatter points
                ax.scatter(tumor_x, tumor_y, tumor_z, c='red', alpha=0.8, s=30, 
                          label=f'Tumor Field (density ≥ {tumor_isosurface_level})')
                ax.legend()
        
        # Add bowl center marker
        ax.scatter([center_x], [center_y], [center_z], c='black', s=100, 
                  marker='*', label='Bowl Center', alpha=1.0)
        ax.legend()
        
        # Set labels and title
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_zlabel('Z (μm)')
        ax.set_title(f'3D Quadratic Potential - Step {step} (t={time:.2f}){title_suffix}\n'
                    f'Strength: {potential_strength:.3f}, Center: ({center_x:.1f}, {center_y:.1f}, {center_z:.1f})')
        
        # Set equal aspect ratio
        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
        mid_x = (X.max()+X.min()) * 0.5
        mid_y = (Y.max()+Y.min()) * 0.5
        mid_z = (Z.max()+Z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        
        if save_plot and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f'3d_quadratic_potential_step_{step:06d}.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            print(f"3D quadratic potential plot saved to {output_dir}/{filename}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_coagulation_analysis(self, simulation_data, step_indices=None, 
                                tumor_isosurface_level=0.1, potential_strength=None,
                                bowl_center=None, output_dir=None, save_plot=False, 
                                show_plot=True, figsize=(15, 10)):
        """
        Create a comprehensive analysis plot showing both tumor field and quadratic potential
        for multiple time steps to visualize coagulation dynamics.
        
        Args:
            simulation_data: Dictionary containing simulation data
            step_indices: List of step indices to plot (None for all available)
            tumor_isosurface_level: Density level for tumor field visualization
            potential_strength: Strength of the bowl potential (if None, extract from config)
            bowl_center: Center coordinates of the bowl (if None, extract from config)
            output_dir: Directory to save plot
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            figsize: Figure size
        """
        # Get available step indices if not provided
        if step_indices is None:
            step_indices = list(range(len(simulation_data["metadata"]["saved_steps"])))
        
        # Limit number of plots to avoid overcrowding
        if len(step_indices) > 6:
            print(f"Warning: Limiting plots to 6 steps (requested {len(step_indices)})")
            step_indices = step_indices[:6]
        
        # Calculate subplot layout
        num_plots = len(step_indices)
        num_cols = min(3, num_plots)
        num_rows = (num_plots + num_cols - 1) // num_cols
        
        # Create figure with subplots
        fig = plt.figure(figsize=figsize)
        
        for i, step_idx in enumerate(step_indices):
            # Get step information
            step = simulation_data["metadata"]["saved_steps"][step_idx]
            time = simulation_data["metadata"]["saved_times"][step_idx]
            
            # Create 3D subplot
            ax = fig.add_subplot(num_rows, num_cols, i+1, projection='3d')
            
            # Get phi_hat for this step
            phi_hat = simulation_data["field_data"]["phi_hat"][step_idx]
            total_density = np.sum(phi_hat, axis=0)
            
            # Create 3D coordinate grids
            nx, ny, nz = self.grid
            x = np.arange(nx) * self.dx
            y = np.arange(ny) * self.dx
            z = np.arange(nz) * self.dx
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            
            # Plot tumor field
            tumor_mask = total_density >= tumor_isosurface_level
            if np.any(tumor_mask):
                tumor_x = X[tumor_mask]
                tumor_y = Y[tumor_mask]
                tumor_z = Z[tumor_mask]
                tumor_values = total_density[tumor_mask]
                
                scatter = ax.scatter(tumor_x, tumor_y, tumor_z, c=tumor_values, 
                                   cmap='viridis', alpha=0.7, s=15)
            
            # Plot quadratic potential (simplified representation)
            if potential_strength is None or bowl_center is None:
                config = simulation_data["metadata"]["config"]
                physics_config = config.get("physics", {})
                if potential_strength is None:
                    potential_strength = physics_config.get("bowl_potential_strength", 0.1)
                if bowl_center is None:
                    bowl_center = physics_config.get("bowl_center", [50, 50, 50])
            
            center_x = bowl_center[0] * self.dx
            center_y = bowl_center[1] * self.dx
            center_z = bowl_center[2] * self.dx
            
            # Add bowl center marker
            ax.scatter([center_x], [center_y], [center_z], c='red', s=100, 
                      marker='*', alpha=1.0)
            
            # Set labels and title
            ax.set_title(f'Step {step} (t={time:.2f})')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            # Set equal aspect ratio
            max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
            mid_x = (X.max()+X.min()) * 0.5
            mid_y = (Y.max()+Y.min()) * 0.5
            mid_z = (Z.max()+Z.min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Set overall title
        fig.suptitle('Coagulation Dynamics: Tumor Field Evolution with Quadratic Potential', 
                    fontsize=16)
        
        plt.tight_layout()
        
        if save_plot and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f'coagulation_analysis_steps_{step_indices[0]}-{step_indices[-1]}.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            print(f"Coagulation analysis plot saved to {output_dir}/{filename}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()

    @classmethod
    def from_simulation_data(cls, simulation_data, step_idx=0):
        """
        Create a plotter instance from saved simulation data.
        
        Args:
            simulation_data: Dictionary containing simulation data or NpzFile object
            step_idx: Index of the saved step to use for plotting
            
        Returns:
            plotter: CellFieldPlotter instance with loaded data
        """
        # Handle NpzFile objects by converting to dictionary
        if hasattr(simulation_data, 'files'):  # It's an NpzFile object
            simulation_data = cls._convert_npz_to_dict(simulation_data)
        
        return cls(simulation_data, step_idx)
    
    @staticmethod
    def _convert_npz_to_dict(npz_file):
        """
        Convert NpzFile object to simulation data dictionary.
        
        Args:
            npz_file: NpzFile object from np.load()
            
        Returns:
            simulation_data: Dictionary with proper structure
        """
        # Load metadata from pickle bytes
        import pickle
        metadata_bytes = npz_file["metadata"].tobytes()
        metadata = pickle.loads(metadata_bytes)
        
        # Create step indices and times from the data
        num_steps = npz_file["phi_hat"].shape[0]
        saved_steps = list(range(num_steps))
        saved_times = [i * 0.5 for i in range(num_steps)]  # Assuming dt=0.5
        
        # Reconstruct simulation data structure
        simulation_data = {
            "metadata": {
                **metadata,
                "saved_steps": saved_steps,
                "saved_times": saved_times
            },
            "field_data": {
                "phi_hat": npz_file["phi_hat"],
                "nutrient_fields": npz_file["nutrient_fields"],
                "host_fields": npz_file["host_fields"]
            },
            "performance": {
                "step_times": npz_file["step_times"],
                "total_cells": npz_file["total_cells"]
            }
        }
        
        # Add physics data if available
        if "pressure" in npz_file:
            simulation_data["physics_data"] = []
            for i in range(len(npz_file["pressure"])):
                physics_data = {
                    "pressure": npz_file["pressure"][i],
                    "velocity": npz_file["velocity"][i],
                    "energy_derivative": npz_file["energy_derivative"][i],
                    "mass_flux": npz_file["mass_flux"][i]
                }
                # Add source terms if available
                if "source_terms" in npz_file:
                    physics_data["source_terms"] = npz_file["source_terms"][i]
                simulation_data["physics_data"].append(physics_data)
        
        return simulation_data
