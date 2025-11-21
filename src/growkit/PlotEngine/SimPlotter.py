import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from .ObservableUtils import ObservableUtils

class SimPlotter:
    def __init__(self, simulation_data):
        """
        Initialize the simulation plotter.
        
        Args:
            simulation_data: Dictionary containing simulation data
        """
        self.simulation_data = simulation_data
        self.metadata = simulation_data["metadata"]
        self.field_data = simulation_data["field_data"]
        
        # Extract basic information
        self.grid_size = self.metadata["grid_size"]
        self.num_populations = self.metadata["num_populations"]
        self.saved_steps = self.metadata["saved_steps"]
        self.saved_times = self.metadata["saved_times"]
        
        # Extract dx from config if available
        self.dx = 1.0  # Default value
        if "config" in self.metadata:
            config = self.metadata["config"]
            if "domain" in config and "dx" in config["domain"]:
                self.dx = float(config["domain"]["dx"])
        
        # Get population labels
        if "population_labels" in self.metadata:
            self.labels = self.metadata["population_labels"]
        elif "config" in self.metadata:
            config = self.metadata["config"]
            if "populations" in config:
                self.labels = [p["label"] for p in config["populations"].values()]
            else:
                self.labels = [f"Population_{i}" for i in range(self.num_populations)]
        else:
            self.labels = [f"Population_{i}" for i in range(self.num_populations)]
        
        # Initialize observable utilities with correct dx
        self.utils = ObservableUtils(self.grid_size, self.dx)
        
        # Memory optimization: ensure field_data uses memory-mapped arrays if available
        # Convert to float32 views to reduce memory if data is float64
        self._optimize_field_data_memory()
    
    def _optimize_field_data_memory(self):
        """Optimize field data memory usage by converting to float32 if needed."""
        # Check if data is already memory-mapped (has _npz_file reference)
        if "_npz_file" in self.simulation_data:
            # Data is memory-mapped, no need to convert
            return
        
        # If data is loaded in memory, ensure it's float32 to save space
        for key in ["phi_hat", "nutrient_fields", "host_fields"]:
            if key in self.field_data and self.field_data[key] is not None:
                arr = self.field_data[key]
                # Handle both list of arrays (from run_and_save_simulation) and numpy arrays (from disk)
                if isinstance(arr, list):
                    # Data is a list of arrays - arrays are already float32 from run_and_save_simulation
                    # No optimization needed
                    continue
                elif hasattr(arr, 'dtype') and arr.dtype == np.float64:
                    # Convert to float32 to save memory (if not memory-mapped)
                    try:
                        self.field_data[key] = arr.astype(np.float32, copy=False)
                    except (ValueError, TypeError):
                        # Can't convert in-place, skip
                        pass
    
    def _get_field_slice(self, field_name, step_idx):
        """
        Get a field slice for a specific step, ensuring memory efficiency.
        Returns a view or minimal copy to avoid memory bloat.
        
        Handles both list and array formats:
        - Lists: field_data["phi_hat"] is a list of arrays (from run_and_save_simulation)
        - Arrays: field_data["phi_hat"] is a numpy array (from load_simulation_data)
        """
        field = self.field_data[field_name]
        if field is None:
            return None
        
        # Get the slice - handles both list and array formats
        # For lists: field[step_idx] returns the array at that index
        # For arrays: field[step_idx] returns a slice (M, nx, ny, nz)
        if isinstance(field, list):
            # Data is a list of arrays (from run_and_save_simulation)
            if step_idx >= len(field):
                raise IndexError(f"Step index {step_idx} out of range for {field_name} (length {len(field)})")
            slice_data = field[step_idx]
        else:
            # Data is a numpy array (from load_simulation_data, possibly memory-mapped)
            slice_data = field[step_idx]
        
        # Ensure slice_data is a numpy array
        if not isinstance(slice_data, np.ndarray):
            slice_data = np.array(slice_data)
        
        # If data is already float32, return as-is (no conversion needed)
        if slice_data.dtype == np.float32:
            return slice_data
        
        # If data is float64 and we're not memory-mapped, convert to float32 to save memory
        # Note: We avoid conversion for memory-mapped arrays to prevent loading full array
        if slice_data.dtype == np.float64:
            if "_npz_file" not in self.simulation_data:
                # Not memory-mapped, safe to convert
                return np.array(slice_data, dtype=np.float32, copy=False)
            else:
                # Memory-mapped - return as-is to avoid creating unnecessary copies
                # The memory-mapped slice will be freed when it goes out of scope
                return slice_data
        
        return slice_data
    
    def _process_steps_in_batches(self, num_steps, batch_size=5, callback=None):
        """
        Process simulation steps in batches to reduce memory usage.
        
        Args:
            num_steps: Total number of steps to process
            batch_size: Number of steps to process before clearing memory
            callback: Function to call for each step_idx, should return result
        
        Returns:
            List of results from callback
        """
        results = []
        for batch_start in range(0, num_steps, batch_size):
            batch_end = min(batch_start + batch_size, num_steps)
            batch_results = []
            
            for step_idx in range(batch_start, batch_end):
                if callback:
                    result = callback(step_idx)
                    batch_results.append(result)
            
            results.extend(batch_results)
            
            # Clear memory after each batch
            import gc
            gc.collect()
            
            # Clear physics cache if available
            if "physics_data" in self.simulation_data:
                physics_data = self.simulation_data["physics_data"]
                if physics_data is not None and hasattr(physics_data, 'clear_cache'):
                    physics_data.clear_cache()
        
        return results
    
    def plot_tumor_radius_evolution(self, output_dir=None, save_plot=False, show_plot=True,
                                   figsize=(12, 8), include_individual_populations=True,
                                   threshold=0.4, method='contour', 
                                   include_inhibited_radius=True,
                                   include_necrotic_radius=True,
                                   include_hypoxic_radius=True,
                                   growth_threshold=3.0,
                                   use_adaptive_threshold=True,
                                   threshold_type='percentile',
                                   growth_threshold_percentile=75.0,
                                   necrotic_threshold_percentile=25.0,
                                   drop_fraction=0.5,
                                   viable_threshold=0.1):
        """
        Plot tumor radius evolution over time.
        
        NOTE ON INDIVIDUAL VS TOTAL RADIUS:
        Individual population radii are calculated using only that population's density field,
        while total radius uses the sum of all populations (including necrotic). This means:
        - If viable cells at the edge die and convert to necrotic, the viable cell density
          may drop below threshold, but necrotic cells keep total density above threshold.
        - This causes total radius to continue growing while individual population radius
          may plateau or grow slower.
        - This is expected behavior: total radius includes all cell types, while individual
          radius shows only where that specific population exceeds the threshold.
        - Use plot_radius_diagnostic() to visualize radial density profiles and understand
          what's happening at the edge.
        
        Args:
            output_dir: Directory to save plot
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            figsize: Figure size
            include_individual_populations: Whether to plot individual population radii
            threshold: Density threshold for radius calculation
            method: Method for radius calculation ('contour' or 'mass')
            include_inhibited_radius: Whether to include inhibited radius calculation (region containing inhibited or dead cells, where nutrient < threshold)
            include_necrotic_radius: Whether to include necrotic radius calculation (region void of living cells, where viable density < 0.01)
            include_hypoxic_radius: Whether to include hypoxic radius calculation (finds outer boundary where necrotic source terms become non-zero)
            growth_threshold: Deprecated parameter (kept for backward compatibility, not used)
            use_adaptive_threshold: Deprecated parameter (kept for backward compatibility, not used)
            threshold_type: Deprecated parameter (kept for backward compatibility, not used)
            growth_threshold_percentile: Deprecated parameter (kept for backward compatibility, not used)
            necrotic_threshold_percentile: Deprecated parameter (kept for backward compatibility, not used - necrotic radius uses density thresholding)
            drop_fraction: Fraction of maximum source term to use as threshold for inhibited radius (default 0.5 = 50% of max)
            viable_threshold: Deprecated parameter (kept for backward compatibility, not used - hypoxic radius uses necrotic source terms)
        """
        print("Calculating tumor radius evolution...")
        
        # Calculate radii for viable populations only (exclude necrotic)
        # NOTE: These are density-based radii - they show where cells of each population exist
        # We exclude the necrotic population since necrotic radius is calculated separately below
        radii_data = {}
        # Necrotic population is the last one (index M-1)
        viable_populations = self.labels[:-1] if len(self.labels) > 1 else self.labels
        for i, label in enumerate(viable_populations):
            radii = []
            for step_idx in range(len(self.saved_steps)):
                phi_hat = self._get_field_slice("phi_hat", step_idx)
                radius = self.utils.calculate_radius(phi_hat[i], threshold=threshold, method=method)
                radii.append(radius)
                # Explicitly delete reference to allow GC
                del phi_hat
                # Clean up memory more frequently for large datasets
                if step_idx % 5 == 0:
                    import gc
                    gc.collect()
            radii_data[label] = radii
        
        # Calculate total tumor radius
        total_radii = []
        for step_idx in range(len(self.saved_steps)):
            phi_hat = self._get_field_slice("phi_hat", step_idx)
            total_density = np.sum(phi_hat, axis=0)
            radius = self.utils.calculate_radius(total_density, threshold=threshold, method=method)
            total_radii.append(radius)
            # Explicitly delete references to allow GC
            del phi_hat, total_density
            # Clean up memory more frequently for large datasets
            if step_idx % 5 == 0:
                import gc
                gc.collect()
        
        # Calculate inhibited radius if requested and source terms are available
        # NOTE: This finds where necrotic source terms cross from zero/negative to positive (moving outward from center)
        # This marks the outer boundary of necrosis death caused by lack of nutrient, representing the outer extent
        # of the hypoxic region where necrosis is actively occurring. This should be further out than the necrotic core radius.
        inhibited_radii = []
        if include_inhibited_radius and "physics_data" in self.simulation_data:
            physics_data = self.simulation_data["physics_data"]
            if physics_data is not None and len(physics_data) > 0 and "source_terms" in physics_data[0]:
                print("Calculating inhibited radius evolution...")
                # Clear physics cache periodically to save memory
                if hasattr(physics_data, 'clear_cache'):
                    physics_data.clear_cache()
                # Extract configuration parameters for new calculation method
                nutrient_field = None
                lambda_rates = None
                nutrient_thresholds = None
                k_switch = 10.0
                beta_N = 0.0005
                
                if "config" in self.metadata:
                    config = self.metadata["config"]
                    # Get k_switch from nutrient config
                    if "nutrient" in config and "dynamics" in config["nutrient"]:
                        k_switch = config["nutrient"]["dynamics"].get("k", 10.0)
                    
                    # Extract lambda_rates and nutrient_thresholds from populations config
                    # IMPORTANT: Use self.labels order to match phi_hat ordering
                    if "populations" in config:
                        populations = config["populations"]
                        
                        # Get viable population labels (exclude necrotic, which is last)
                        viable_labels = self.labels[:-1] if len(self.labels) > 1 else self.labels
                        
                        if viable_labels:
                            lambda_rates = []
                            nutrient_thresholds = []
                            
                            # Iterate in the same order as phi_hat (using self.labels)
                            # Need to map labels back to population names in config
                            # The config uses population keys (like "Tumour"), not labels
                            # So we need to find the config key that matches each label
                            for label in viable_labels:
                                # Find the population config entry that matches this label
                                found = False
                                for pop_name, pop_config in populations.items():
                                    if pop_name.lower() == "necrotic":
                                        continue
                                    # Check if this population's label matches
                                    pop_label = pop_config.get("label", pop_name)
                                    if pop_label == label:
                                        if "dynamics" in pop_config:
                                            dynamics = pop_config["dynamics"]
                                            lambda_rates.append(dynamics.get("lambda", 0.0))
                                            nutrient_thresholds.append(dynamics.get("nutrient_threshold", 0.0))
                                            found = True
                                            break
                                
                                if not found:
                                    # Fallback: if label doesn't match, try to use index
                                    # This shouldn't happen, but handle gracefully
                                    print(f"Warning: Could not find config for label '{label}', using defaults")
                                    lambda_rates.append(0.0)
                                    nutrient_thresholds.append(0.0)
                            
                            # Get beta_N from necrotic population if it exists
                            if "Necrotic" in populations and "dynamics" in populations["Necrotic"]:
                                beta_N = populations["Necrotic"]["dynamics"].get("beta_N", 0.0005)
                            
                            lambda_rates = np.array(lambda_rates, dtype=np.float32)
                            nutrient_thresholds = np.array(nutrient_thresholds, dtype=np.float32)
                
                for step_idx in range(len(self.saved_steps)):
                    phi_hat = self._get_field_slice("phi_hat", step_idx)
                    total_density = np.sum(phi_hat, axis=0)
                    source_terms = physics_data[step_idx]["source_terms"]
                    
                    # Get nutrient field if available
                    if "nutrient_fields" in self.field_data and self.field_data["nutrient_fields"] is not None:
                        nutrient_field = self._get_field_slice("nutrient_fields", step_idx)
                    else:
                        nutrient_field = None
                    
                    # Clean up memory periodically
                    if step_idx % 5 == 0:
                        import gc
                        gc.collect()
                        # Clear physics cache if it's getting large
                        if hasattr(physics_data, 'clear_cache') and step_idx % 10 == 0:
                            physics_data.clear_cache()
                    
                    # Calculate inhibited radius using new method if parameters are available
                    # New method: radius where proliferation source term falls below threshold but death is still OFF
                    inhibited_radius = self.utils.calculate_inhibited_radius(
                        source_terms, total_density,
                        nutrient_field=nutrient_field,
                        phi_hat=phi_hat,
                        lambda_rates=lambda_rates,
                        nutrient_thresholds=nutrient_thresholds,
                        k_switch=k_switch,
                        beta_N=beta_N,
                        density_threshold=threshold,
                        method='radial_average',
                        # Deprecated parameters kept for backward compatibility
                        growth_threshold=growth_threshold,
                        threshold_type=threshold_type,
                        threshold_percentile=growth_threshold_percentile,
                        use_adaptive_threshold=use_adaptive_threshold,
                        exclude_necrotic=True,
                        drop_fraction=drop_fraction
                    )
                    inhibited_radii.append(inhibited_radius)
            else:
                print("Warning: Source terms not available for inhibited radius calculation")
                include_inhibited_radius = False
        else:
            include_inhibited_radius = False
        
        # Calculate necrotic radius if requested
        # NOTE: Based on paper definition: "Proportion of outer radius void of living cells"
        # This finds the OUTERMOST radius of the central region where viable cell density is very low
        # The necrotic core is in the center, so we need to find where viable density transitions
        # from low (necrotic core) to higher (viable rim) as we move outward from center
        necrotic_radii = []
        if include_necrotic_radius:
            # Check if we have a necrotic population (last population)
            if len(self.labels) > 1:
                print("Calculating necrotic radius evolution...")
                # Use a very low threshold for viable cells (void of living cells)
                viable_threshold = 0.01  # Very low threshold - region void of living cells
                for step_idx in range(len(self.saved_steps)):
                    phi_hat = self._get_field_slice("phi_hat", step_idx)
                    total_density = np.sum(phi_hat, axis=0)
                    
                    # Clean up memory periodically
                    if step_idx % 5 == 0:
                        import gc
                        gc.collect()
                    
                    # Calculate necrotic radius using phi_hat (viable density)
                    necrotic_radius = self.utils.calculate_necrotic_radius(
                        source_terms=None,  # Not needed for new method
                        density_field=total_density,
                        phi_hat=phi_hat,
                        density_threshold=threshold,
                        viable_threshold=viable_threshold,
                        method='radial_average'
                    )
                    necrotic_radii.append(necrotic_radius)
            else:
                print("Warning: No necrotic population found (need at least 2 populations)")
                include_necrotic_radius = False
        else:
            include_necrotic_radius = False
        
        # Calculate hypoxic radius if requested and source terms are available
        # NOTE: This finds the outer boundary where necrotic source terms become non-zero
        # This marks where necrotic cells transition from zero to positive source terms (active necrotic processes)
        # This should be greater than the necrotic core radius
        hypoxic_radii = []
        if include_hypoxic_radius and "physics_data" in self.simulation_data:
            physics_data = self.simulation_data["physics_data"]
            if physics_data is not None and len(physics_data) > 0 and "source_terms" in physics_data[0]:
                # Check if we have a necrotic population (last population)
                if len(self.labels) > 1:
                    print("Calculating hypoxic radius evolution...")
                    # Clear physics cache before starting
                    if hasattr(physics_data, 'clear_cache'):
                        physics_data.clear_cache()
                    for step_idx in range(len(self.saved_steps)):
                        phi_hat = self._get_field_slice("phi_hat", step_idx)
                        total_density = np.sum(phi_hat, axis=0)
                        source_terms = physics_data[step_idx]["source_terms"]
                        # Clean up memory periodically
                        if step_idx % 5 == 0:
                            import gc
                            gc.collect()
                            if hasattr(physics_data, 'clear_cache') and step_idx % 10 == 0:
                                physics_data.clear_cache()
                        
                        # Calculate hypoxic radius (outer boundary where necrotic source terms become non-zero)
                        hypoxic_radius = self.utils.calculate_hypoxic_radius(
                            source_terms, total_density,
                            density_threshold=threshold,
                            method='radial_average'
                        )
                        
                        # Ensure hypoxic radius is greater than necrotic radius
                        if include_necrotic_radius and len(necrotic_radii) > step_idx:
                            necrotic_radius = necrotic_radii[step_idx]
                            if hypoxic_radius <= necrotic_radius:
                                hypoxic_radius = 0.0  # If not greater than necrotic, set to 0
                        
                        hypoxic_radii.append(hypoxic_radius)
                else:
                    print("Warning: No necrotic population found (need at least 2 populations)")
                    include_hypoxic_radius = False
            else:
                print("Warning: Source terms not available for hypoxic radius calculation")
                include_hypoxic_radius = False
        else:
            if include_hypoxic_radius:
                print("Warning: Physics data not available for hypoxic radius calculation")
            include_hypoxic_radius = False
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot total radius
        ax.plot(self.saved_times, total_radii, 'k-', linewidth=3, label='Total Tumor', marker='o', markersize=6)
        
        # Plot inhibited radius if available
        if include_inhibited_radius:
            ax.plot(self.saved_times, inhibited_radii, 'g--', linewidth=2, 
                   label='Inhibited Radius', marker='^', markersize=5)
        
        # Plot necrotic radius if available
        # NOTE: This is the density-based necrotic radius (from necrotic population density)
        if include_necrotic_radius:
            ax.plot(self.saved_times, necrotic_radii, 'r--', linewidth=2, 
                   label='Necrotic Radius', marker='v', markersize=5)
        
        # Plot hypoxic radius if available
        # NOTE: This is the outer boundary where necrotic source terms become non-zero
        if include_hypoxic_radius:
            ax.plot(self.saved_times, hypoxic_radii, 'b--', linewidth=2, 
                   label='Hypoxic Radius', marker='s', markersize=5)
        
        # Plot individual population radii if requested
        if include_individual_populations:
            colors = plt.cm.Set1(np.linspace(0, 1, len(self.labels)))
            for i, (label, radii) in enumerate(radii_data.items()):
                ax.plot(self.saved_times, radii, '--', color=colors[i], linewidth=2, 
                       label=f'{label}', marker='s', markersize=4)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Radius')
        ax.set_title('Tumor Radius Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add text box with information about thresholds
        if include_inhibited_radius or include_necrotic_radius or include_hypoxic_radius:
            info_text = f'Thresholds:\n'
            info_text += f'Density threshold: {threshold} (used for all radii)\n'
            if include_inhibited_radius:
                info_text += f'Inhibited: source terms >= 5% of max (from center)\n'
            if include_necrotic_radius:
                info_text += f'Necrotic: density threshold ({threshold})\n'
            if include_hypoxic_radius:
                info_text += f'Hypoxic: necrotic source terms > 0'
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if save_plot and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f'tumor_radius_evolution_threshold_{threshold}.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            print(f"Tumor radius evolution plot saved to {output_dir}/{filename}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        # Return data for use by plot_all_observables
        return {
            'total_radii': total_radii,
            'population_radii': radii_data,
            'inhibited_radii': inhibited_radii if include_inhibited_radius else None,
            'necrotic_radii': necrotic_radii if include_necrotic_radius else None,
            'hypoxic_radii': hypoxic_radii if include_hypoxic_radius else None
        } 
    
    def plot_population_density_evolution(self, output_dir=None, save_plot=False, show_plot=True,
                                        figsize=(12, 8), normalize_by_volume=False):
        """
        Plot population density evolution over time.
        
        Args:
            output_dir: Directory to save plot
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            figsize: Figure size
            normalize_by_volume: Whether to normalize by total volume
        """
        print("Calculating population density evolution...")
        
        # Calculate densities for all populations
        density_data = {}
        for i, label in enumerate(self.labels):
            densities = []
            for step_idx in range(len(self.saved_steps)):
                phi_hat = self.field_data["phi_hat"][step_idx]
                density = self.utils.calculate_total_density(phi_hat[i], normalize_by_volume=normalize_by_volume)
                densities.append(density)
            density_data[label] = densities
        
        # Calculate total density
        total_densities = []
        for step_idx in range(len(self.saved_steps)):
            phi_hat = self.field_data["phi_hat"][step_idx]
            total_density = np.sum(phi_hat, axis=0)
            density = self.utils.calculate_total_density(total_density, normalize_by_volume=normalize_by_volume)
            total_densities.append(density)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot total density
        ax.plot(self.saved_times, total_densities, 'k-', linewidth=3, label='Total', marker='o', markersize=6)
        
        # Plot individual population densities
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.labels)))
        for i, (label, densities) in enumerate(density_data.items()):
            ax.plot(self.saved_times, densities, '--', color=colors[i], linewidth=2, 
                   label=f'{label}', marker='s', markersize=4)
        
        ylabel = 'Normalized Density' if normalize_by_volume else 'Total Density'
        ax.set_xlabel('Time')
        ax.set_ylabel(ylabel)
        ax.set_title('Population Density Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f'population_density_evolution_normalized_{normalize_by_volume}.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            print(f"Population density evolution plot saved to {output_dir}/{filename}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        # Return data for use by plot_all_observables and export
        return {
            'total_densities': total_densities,
            'population_densities': density_data
        }
    
    def plot_tumor_shape_evolution(self, output_dir=None, save_plot=False, show_plot=True,
                                  figsize=(15, 10), threshold=0.1, max_plots=6):
        """
        Plot tumor shape evolution using contour plots.
        
        Args:
            output_dir: Directory to save plot
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            figsize: Figure size
            threshold: Density threshold for contour
            max_plots: Maximum number of time points to plot
        """
        print("Creating tumor shape evolution plot...")
        
        # Select time points to plot
        if len(self.saved_steps) <= max_plots:
            plot_indices = list(range(len(self.saved_steps)))
        else:
            # Select evenly spaced time points
            plot_indices = np.linspace(0, len(self.saved_steps)-1, max_plots, dtype=int)
        
        # Calculate subplot layout
        num_plots = len(plot_indices)
        num_cols = min(3, num_plots)
        num_rows = (num_plots + num_cols - 1) // num_cols
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
        if num_plots == 1:
            axes_flat = [axes]
        elif num_rows == 1:
            axes_flat = axes.reshape(1, -1).flatten()
        elif num_cols == 1:
            axes_flat = axes.reshape(-1, 1).flatten()
        else:
            axes_flat = axes.flatten()
        
        # Get middle z-slice
        z_slice = self.grid_size[2] // 2
        
        for i, step_idx in enumerate(plot_indices):
            ax = axes_flat[i]
            
            # Get total density for this step
            phi_hat = self.field_data["phi_hat"][step_idx]
            total_density = np.sum(phi_hat, axis=0)
            density_slice = total_density[:, :, z_slice]
            
            # Create coordinate grids
            x = np.arange(self.grid_size[0]) * self.dx
            y = np.arange(self.grid_size[1]) * self.dx
            X, Y = np.meshgrid(x, y, indexing='ij')
            
            # Plot density as background
            im = ax.imshow(density_slice, extent=[x[0], x[-1], y[0], y[-1]], 
                          origin='lower', cmap='viridis', aspect='equal')
            
            # Add contour for tumor boundary
            ax.contour(X, Y, density_slice, levels=[threshold], colors='red', 
                      linewidths=2, alpha=0.8)
            
            # Set title and labels
            time = self.saved_times[step_idx]
            step = self.saved_steps[step_idx]
            ax.set_title(f'Step {step} (t={time:.2f})')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
        
        # Hide unused subplots
        for i in range(num_plots, len(axes_flat)):
            axes_flat[i].set_visible(False)
        
        # Add colorbar to the last plot
        if num_plots > 0:
            fig.colorbar(im, ax=axes_flat[num_plots-1], label='Density')
        
        fig.suptitle('Tumor Shape Evolution', fontsize=16)
        plt.tight_layout()
        
        if save_plot and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f'tumor_shape_evolution_threshold_{threshold}.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            print(f"Tumor shape evolution plot saved to {output_dir}/{filename}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_center_of_mass_evolution(self, output_dir=None, save_plot=False, show_plot=True,
                                    figsize=(12, 8), include_individual_populations=True):
        """
        Plot center of mass evolution over time.
        
        Args:
            output_dir: Directory to save plot
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            figsize: Figure size
            include_individual_populations: Whether to plot individual population centers
        """
        print("Calculating center of mass evolution...")
        
        # Calculate centers of mass for all populations
        com_data = {}
        for i, label in enumerate(self.labels):
            com_x, com_y, com_z = [], [], []
            for step_idx in range(len(self.saved_steps)):
                phi_hat = self.field_data["phi_hat"][step_idx]
                com = self.utils.calculate_center_of_mass(phi_hat[i])
                com_x.append(com[0])
                com_y.append(com[1])
                com_z.append(com[2])
            com_data[label] = {'x': com_x, 'y': com_y, 'z': com_z}
        
        # Calculate total center of mass
        total_com_x, total_com_y, total_com_z = [], [], []
        for step_idx in range(len(self.saved_steps)):
            phi_hat = self.field_data["phi_hat"][step_idx]
            total_density = np.sum(phi_hat, axis=0)
            com = self.utils.calculate_center_of_mass(total_density)
            total_com_x.append(com[0])
            total_com_y.append(com[1])
            total_com_z.append(com[2])
        
        # Create subplots for x, y, z coordinates
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Plot total center of mass
        axes[0].plot(self.saved_times, total_com_x, 'k-', linewidth=3, label='Total', marker='o', markersize=6)
        axes[1].plot(self.saved_times, total_com_y, 'k-', linewidth=3, label='Total', marker='o', markersize=6)
        axes[2].plot(self.saved_times, total_com_z, 'k-', linewidth=3, label='Total', marker='o', markersize=6)
        
        # Plot individual population centers if requested
        if include_individual_populations:
            colors = plt.cm.Set1(np.linspace(0, 1, len(self.labels)))
            for i, (label, com) in enumerate(com_data.items()):
                axes[0].plot(self.saved_times, com['x'], '--', color=colors[i], linewidth=2, 
                           label=f'{label}', marker='s', markersize=4)
                axes[1].plot(self.saved_times, com['y'], '--', color=colors[i], linewidth=2, 
                           label=f'{label}', marker='s', markersize=4)
                axes[2].plot(self.saved_times, com['z'], '--', color=colors[i], linewidth=2, 
                           label=f'{label}', marker='s', markersize=4)
        
        # Set labels and titles
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('X Coordinate')
        axes[0].set_title('Center of Mass - X')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Y Coordinate')
        axes[1].set_title('Center of Mass - Y')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        axes[2].set_xlabel('Time')
        axes[2].set_ylabel('Z Coordinate')
        axes[2].set_title('Center of Mass - Z')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = 'center_of_mass_evolution.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            print(f"Center of mass evolution plot saved to {output_dir}/{filename}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        # Return data for use by plot_all_observables and export
        return {
            'total_com': {'x': total_com_x, 'y': total_com_y, 'z': total_com_z},
            'population_com': com_data
        }
    
    def plot_compactness_evolution(self, output_dir=None, save_plot=False, show_plot=True,
                                 figsize=(12, 8), include_individual_populations=True):
        """
        Plot tumor compactness evolution over time.
        
        Args:
            output_dir: Directory to save plot
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            figsize: Figure size
            include_individual_populations: Whether to plot individual population compactness
        """
        print("Calculating compactness evolution...")
        
        # Calculate compactness for all populations
        compactness_data = {}
        for i, label in enumerate(self.labels):
            compactness = []
            for step_idx in range(len(self.saved_steps)):
                phi_hat = self.field_data["phi_hat"][step_idx]
                comp = self.utils.calculate_compactness(phi_hat[i])
                compactness.append(comp)
            compactness_data[label] = compactness
        
        # Calculate total compactness
        total_compactness = []
        for step_idx in range(len(self.saved_steps)):
            phi_hat = self.field_data["phi_hat"][step_idx]
            total_density = np.sum(phi_hat, axis=0)
            comp = self.utils.calculate_compactness(total_density)
            total_compactness.append(comp)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot total compactness
        ax.plot(self.saved_times, total_compactness, 'k-', linewidth=3, label='Total', marker='o', markersize=6)
        
        # Plot individual population compactness if requested
        if include_individual_populations:
            colors = plt.cm.Set1(np.linspace(0, 1, len(self.labels)))
            for i, (label, compactness) in enumerate(compactness_data.items()):
                ax.plot(self.saved_times, compactness, '--', color=colors[i], linewidth=2, 
                       label=f'{label}', marker='s', markersize=4)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Compactness')
        ax.set_title('Tumor Compactness Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = 'compactness_evolution.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            print(f"Compactness evolution plot saved to {output_dir}/{filename}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        # Return data for use by plot_all_observables and export
        return {
            'total_compactness': total_compactness,
            'population_compactness': compactness_data
        }
    
    def plot_all_observables(self, output_dir=None, save_plot=False, show_plot=True,
                           figsize=(20, 15), threshold=0.1, 
                           include_inhibited_radius=True,
                           include_necrotic_radius=True,
                           include_hypoxic_radius=True,
                           growth_threshold=0.01,
                           use_adaptive_threshold=True,
                           threshold_type='percentile',
                           growth_threshold_percentile=75.0,
                           necrotic_threshold_percentile=25.0,
                           only_radii=False,
                           drop_fraction=0.5,
                           viable_threshold=0.1):
        """
        Create a comprehensive plot of all observables.
        
        Args:
            output_dir: Directory to save plot
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            figsize: Figure size
            threshold: Density threshold for radius calculation
            include_inhibited_radius: Whether to include inhibited radius calculation
            include_necrotic_radius: Whether to include necrotic radius calculation
            include_hypoxic_radius: Whether to include hypoxic radius calculation
            growth_threshold: Absolute threshold for detecting proliferative cells (used if use_adaptive_threshold=False)
            use_adaptive_threshold: Whether to use adaptive thresholds based on source term distribution
            threshold_type: Type of adaptive threshold ('percentile', 'relative_max', 'relative_mean')
            growth_threshold_percentile: Percentile for proliferative threshold (default 75.0)
            necrotic_threshold_percentile: Percentile for necrotic threshold (default 25.0)
            only_radii: If True, only calculate radius-related observables (for export)
            drop_fraction: Fraction of maximum source term to use as threshold for inhibited radius (default 0.5)
            viable_threshold: Nutrient concentration threshold below which cells are hypoxic (default 0.1)
        """
        if not only_radii:
            print("Creating comprehensive observables plot...")
        else:
            print("Calculating radius observables...")
        
        # Calculate observables
        observables = {}
        
        # Radius evolution (always calculated)
        # Note: method parameter is not passed through plot_all_observables, so it uses default 'contour'
        # If you need a different method, call plot_tumor_radius_evolution directly
        radius_data = self.plot_tumor_radius_evolution(output_dir=None, save_plot=False, 
                                                      show_plot=False, threshold=threshold,
                                                      method='contour',  # Default method for consistency
                                                      include_inhibited_radius=include_inhibited_radius,
                                                      include_necrotic_radius=include_necrotic_radius,
                                                      include_hypoxic_radius=include_hypoxic_radius,
                                                      growth_threshold=growth_threshold,
                                                      use_adaptive_threshold=use_adaptive_threshold,
                                                      threshold_type=threshold_type,
                                                      growth_threshold_percentile=growth_threshold_percentile,
                                                      necrotic_threshold_percentile=necrotic_threshold_percentile,
                                                      drop_fraction=drop_fraction,
                                                      viable_threshold=viable_threshold)
        observables['radius'] = radius_data
        
        # Only calculate other observables if not in export-only mode
        if not only_radii:
            # Density evolution
            density_data = self.plot_population_density_evolution(output_dir=None, save_plot=False, 
                                                                show_plot=False)
            observables['density'] = density_data
            
            # Center of mass evolution
            com_data = self.plot_center_of_mass_evolution(output_dir=None, save_plot=False, 
                                                         show_plot=False)
            observables['center_of_mass'] = com_data
            
            # Compactness evolution
            compactness_data = self.plot_compactness_evolution(output_dir=None, save_plot=False, 
                                                             show_plot=False)
            observables['compactness'] = compactness_data
        
        # Only create plots if not in export-only mode
        if not only_radii:
            # Create subplot grid
            fig, axes = plt.subplots(2, 3, figsize=figsize)
            
            # Plot 1: Radius evolution
            ax = axes[0, 0]
            ax.plot(self.saved_times, radius_data['total_radii'], 'k-', linewidth=3, label='Total', marker='o', markersize=4)
            colors = plt.cm.Set1(np.linspace(0, 1, len(self.labels)))
            for i, (label, radii) in enumerate(radius_data['population_radii'].items()):
                ax.plot(self.saved_times, radii, '--', color=colors[i], linewidth=2, label=f'{label}', marker='s', markersize=3)
            
            # Add inhibited radius if available
            if include_inhibited_radius and radius_data.get('inhibited_radii') is not None:
                ax.plot(self.saved_times, radius_data['inhibited_radii'], 'g--', linewidth=2, 
                       label='Inhibited', marker='^', markersize=4)
            
            # Add necrotic radius if available
            if include_necrotic_radius and radius_data.get('necrotic_radii') is not None:
                ax.plot(self.saved_times, radius_data['necrotic_radii'], 'r--', linewidth=2, 
                       label='Necrotic', marker='v', markersize=4)
            
            # Add hypoxic radius if available
            if include_hypoxic_radius and radius_data.get('hypoxic_radii') is not None:
                ax.plot(self.saved_times, radius_data['hypoxic_radii'], 'b--', linewidth=2, 
                       label='Hypoxic', marker='s', markersize=4)
            
            ax.set_xlabel('Time')
            ax.set_ylabel('Radius')
            ax.set_title('Tumor Radius')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 2: Density evolution
            ax = axes[0, 1]
            ax.plot(self.saved_times, density_data['total_densities'], 'k-', linewidth=3, label='Total', marker='o', markersize=4)
            for i, (label, densities) in enumerate(density_data['population_densities'].items()):
                ax.plot(self.saved_times, densities, '--', color=colors[i], linewidth=2, label=f'{label}', marker='s', markersize=3)
            ax.set_xlabel('Time')
            ax.set_ylabel('Total Density')
            ax.set_title('Population Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 3: Center of mass X
            ax = axes[0, 2]
            ax.plot(self.saved_times, com_data['total_com']['x'], 'k-', linewidth=3, label='Total', marker='o', markersize=4)
            for i, (label, com) in enumerate(com_data['population_com'].items()):
                ax.plot(self.saved_times, com['x'], '--', color=colors[i], linewidth=2, label=f'{label}', marker='s', markersize=3)
            ax.set_xlabel('Time')
            ax.set_ylabel('X Coordinate')
            ax.set_title('Center of Mass - X')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 4: Center of mass Y
            ax = axes[1, 0]
            ax.plot(self.saved_times, com_data['total_com']['y'], 'k-', linewidth=3, label='Total', marker='o', markersize=4)
            for i, (label, com) in enumerate(com_data['population_com'].items()):
                ax.plot(self.saved_times, com['y'], '--', color=colors[i], linewidth=2, label=f'{label}', marker='s', markersize=3)
            ax.set_xlabel('Time')
            ax.set_ylabel('Y Coordinate')
            ax.set_title('Center of Mass - Y')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 5: Center of mass Z
            ax = axes[1, 1]
            ax.plot(self.saved_times, com_data['total_com']['z'], 'k-', linewidth=3, label='Total', marker='o', markersize=4)
            for i, (label, com) in enumerate(com_data['population_com'].items()):
                ax.plot(self.saved_times, com['z'], '--', color=colors[i], linewidth=2, label=f'{label}', marker='s', markersize=3)
            ax.set_xlabel('Time')
            ax.set_ylabel('Z Coordinate')
            ax.set_title('Center of Mass - Z')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 6: Compactness
            ax = axes[1, 2]
            ax.plot(self.saved_times, compactness_data['total_compactness'], 'k-', linewidth=3, label='Total', marker='o', markersize=4)
            for i, (label, compactness) in enumerate(compactness_data['population_compactness'].items()):
                ax.plot(self.saved_times, compactness, '--', color=colors[i], linewidth=2, label=f'{label}', marker='s', markersize=3)
            ax.set_xlabel('Time')
            ax.set_ylabel('Compactness')
            ax.set_title('Tumor Compactness')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_plot and output_dir:
                os.makedirs(output_dir, exist_ok=True)
                filename = 'all_observables_comprehensive.png'
                plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
                print(f"Comprehensive observables plot saved to {output_dir}/{filename}")
            
            if show_plot:
                plt.show()
            else:
                plt.close()
        
        # Return observables data
        return observables
    
    def plot_nutrient_field_evolution(self, output_dir=None, save_plot=False, show_plot=True,
                                    figsize=(15, 10), z_slice=None, cmap="viridis", 
                                    add_tumor_contours=False, tumor_threshold=0.1,
                                    max_plots=6, include_statistics=True):
        """
        Plot nutrient field evolution over time.
        
        Args:
            output_dir: Directory to save plot
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            figsize: Figure size
            z_slice: Z-slice to plot (defaults to center)
            cmap: Colormap for the nutrient field
            add_tumor_contours: Whether to add tumor boundary contours
            tumor_threshold: Density threshold for tumor boundary
            max_plots: Maximum number of time points to plot
            include_statistics: Whether to include statistics text on plots
        """
        print("Creating nutrient field evolution plot...")
        
        # Check if nutrient fields are available
        if "nutrient_fields" not in self.field_data:
            print("Warning: No nutrient fields found in simulation data.")
            print("Available field data keys:", list(self.field_data.keys()))
            return
        
        # Select time points to plot
        if len(self.saved_steps) <= max_plots:
            plot_indices = list(range(len(self.saved_steps)))
        else:
            # Select evenly spaced time points
            plot_indices = np.linspace(0, len(self.saved_steps)-1, max_plots, dtype=int)
        
        # Calculate subplot layout
        num_plots = len(plot_indices)
        num_cols = min(3, num_plots)
        num_rows = (num_plots + num_cols - 1) // num_cols
        
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
            z_slice = self.grid_size[2] // 2
        
        # Create coordinate grids
        x = np.arange(self.grid_size[0]) * self.dx
        y = np.arange(self.grid_size[1]) * self.dx
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Find global min/max for consistent color scaling
        all_nutrient_data = []
        for step_idx in range(len(self.saved_steps)):
            nutrient_field = self.field_data["nutrient_fields"][step_idx]
            nutrient_slice = nutrient_field[:, :, z_slice]
            all_nutrient_data.append(nutrient_slice)
        
        vmin = np.min(all_nutrient_data)
        vmax = np.max(all_nutrient_data)
        
        # Plot each time step
        for i, step_idx in enumerate(plot_indices):
            ax = axes_flat[i]
            
            # Get step information
            step = self.saved_steps[step_idx]
            time = self.saved_times[step_idx]
            
            # Get nutrient field for this step
            nutrient_field = self.field_data["nutrient_fields"][step_idx]
            nutrient_slice = nutrient_field[:, :, z_slice]
            
            # Create the plot with consistent color scaling
            im = ax.imshow(nutrient_slice, extent=[x[0], x[-1], y[0], y[-1]], 
                          origin='lower', cmap=cmap, aspect='equal', vmin=vmin, vmax=vmax)
            
            # Add tumor boundary contours if requested
            if add_tumor_contours:
                # Get total tumor density for boundary detection
                phi_hat = self.field_data["phi_hat"][step_idx]
                total_density = np.sum(phi_hat, axis=0)
                density_slice = total_density[:, :, z_slice]
                
                # Add tumor boundary contour
                ax.contour(X, Y, density_slice, levels=[tumor_threshold], 
                          colors='red', linewidths=2, alpha=0.8)
            
            # Add statistics text if requested
            if include_statistics:
                stats_text = f"Min: {np.min(nutrient_slice):.3f}\nMax: {np.max(nutrient_slice):.3f}\nMean: {np.mean(nutrient_slice):.3f}"
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Set title and labels
            ax.set_title(f'Step {step} (t={time:.2f})')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            
            # Add colorbar to the last plot
            if i == num_plots - 1:
                fig.colorbar(im, ax=ax, label='Nutrient Concentration')
        
        # Hide unused subplots
        for i in range(num_plots, len(axes_flat)):
            axes_flat[i].set_visible(False)
        
        # Set overall title
        fig.suptitle(f'Nutrient Field Evolution (z={z_slice})', fontsize=16)
        
        plt.tight_layout()
        
        if save_plot and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f'nutrient_field_evolution_z{z_slice}.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            print(f"Nutrient field evolution plot saved to {output_dir}/{filename}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_nutrient_statistics_evolution(self, output_dir=None, save_plot=False, show_plot=True,
                                         figsize=(15, 10), z_slice=None):
        """
        Plot nutrient field statistics evolution over time.
        
        Args:
            output_dir: Directory to save plot
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            figsize: Figure size
            z_slice: Z-slice to analyze (defaults to center)
        """
        print("Calculating nutrient field statistics evolution...")
        
        # Check if nutrient fields are available
        if "nutrient_fields" not in self.field_data:
            print("Warning: No nutrient fields found in simulation data.")
            print("Available field data keys:", list(self.field_data.keys()))
            return
        
        # Set default z_slice if not provided
        if z_slice is None:
            z_slice = self.grid_size[2] // 2
        
        # Calculate statistics for each time step
        min_concentrations = []
        max_concentrations = []
        mean_concentrations = []
        total_concentrations = []
        
        for step_idx in range(len(self.saved_steps)):
            nutrient_field = self.field_data["nutrient_fields"][step_idx]
            nutrient_slice = nutrient_field[:, :, z_slice]
            
            min_concentrations.append(np.min(nutrient_slice))
            max_concentrations.append(np.max(nutrient_slice))
            mean_concentrations.append(np.mean(nutrient_slice))
            total_concentrations.append(np.sum(nutrient_slice))
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Min concentration
        ax = axes[0, 0]
        ax.plot(self.saved_times, min_concentrations, 'b-', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Time')
        ax.set_ylabel('Min Concentration')
        ax.set_title('Minimum Nutrient Concentration')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Max concentration
        ax = axes[0, 1]
        ax.plot(self.saved_times, max_concentrations, 'r-', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Time')
        ax.set_ylabel('Max Concentration')
        ax.set_title('Maximum Nutrient Concentration')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Mean concentration
        ax = axes[1, 0]
        ax.plot(self.saved_times, mean_concentrations, 'g-', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Time')
        ax.set_ylabel('Mean Concentration')
        ax.set_title('Mean Nutrient Concentration')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Total concentration
        ax = axes[1, 1]
        ax.plot(self.saved_times, total_concentrations, 'm-', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Time')
        ax.set_ylabel('Total Concentration')
        ax.set_title('Total Nutrient Concentration')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f'nutrient_statistics_evolution_z{z_slice}.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            print(f"Nutrient statistics evolution plot saved to {output_dir}/{filename}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        # Return data for use by plot_all_observables and export
        return {
            'total_densities': total_densities,
            'population_densities': density_data
,
            'times': self.saved_times,
            'min_concentrations': min_concentrations,
            'max_concentrations': max_concentrations,
            'mean_concentrations': mean_concentrations,
            'total_concentrations': total_concentrations
        }
    
    def plot_nutrient_tumor_correlation(self, output_dir=None, save_plot=False, show_plot=True,
                                      figsize=(15, 10), z_slice=None, tumor_threshold=0.1):
        """
        Plot correlation between nutrient field and tumor density.
        
        Args:
            output_dir: Directory to save plot
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            figsize: Figure size
            z_slice: Z-slice to analyze (defaults to center)
            tumor_threshold: Density threshold for tumor region
        """
        print("Analyzing nutrient-tumor correlation...")
        
        # Check if nutrient fields are available
        if "nutrient_fields" not in self.field_data:
            print("Warning: No nutrient fields found in simulation data.")
            print("Available field data keys:", list(self.field_data.keys()))
            return
        
        # Set default z_slice if not provided
        if z_slice is None:
            z_slice = self.grid_size[2] // 2
        
        # Calculate correlation metrics for each time step
        correlations = []
        tumor_mean_nutrients = []
        background_mean_nutrients = []
        
        for step_idx in range(len(self.saved_steps)):
            # Get nutrient field
            nutrient_field = self.field_data["nutrient_fields"][step_idx]
            nutrient_slice = nutrient_field[:, :, z_slice]
            
            # Get tumor density
            phi_hat = self.field_data["phi_hat"][step_idx]
            total_density = np.sum(phi_hat, axis=0)
            density_slice = total_density[:, :, z_slice]
            
            # Create tumor mask
            tumor_mask = density_slice > tumor_threshold
            background_mask = ~tumor_mask
            
            # Calculate correlation
            if np.any(tumor_mask) and np.any(background_mask):
                correlation = np.corrcoef(density_slice.flatten(), nutrient_slice.flatten())[0, 1]
                correlations.append(correlation if not np.isnan(correlation) else 0.0)
                
                # Calculate mean nutrient levels
                tumor_mean_nutrients.append(np.mean(nutrient_slice[tumor_mask]))
                background_mean_nutrients.append(np.mean(nutrient_slice[background_mask]))
            else:
                correlations.append(0.0)
                tumor_mean_nutrients.append(0.0)
                background_mean_nutrients.append(np.mean(nutrient_slice))
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Correlation over time
        ax = axes[0, 0]
        ax.plot(self.saved_times, correlations, 'b-', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Time')
        ax.set_ylabel('Correlation Coefficient')
        ax.set_title('Nutrient-Tumor Correlation')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Plot 2: Mean nutrient in tumor vs background
        ax = axes[0, 1]
        ax.plot(self.saved_times, tumor_mean_nutrients, 'r-', linewidth=2, marker='o', markersize=4, label='Tumor Region')
        ax.plot(self.saved_times, background_mean_nutrients, 'g-', linewidth=2, marker='s', markersize=4, label='Background')
        ax.set_xlabel('Time')
        ax.set_ylabel('Mean Nutrient Concentration')
        ax.set_title('Mean Nutrient Levels')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Nutrient difference (tumor - background)
        ax = axes[1, 0]
        nutrient_diff = np.array(tumor_mean_nutrients) - np.array(background_mean_nutrients)
        ax.plot(self.saved_times, nutrient_diff, 'm-', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Time')
        ax.set_ylabel('Nutrient Difference')
        ax.set_title('Tumor - Background Nutrient Difference')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Plot 4: Scatter plot for last time step
        ax = axes[1, 1]
        last_nutrient = self.field_data["nutrient_fields"][-1][:, :, z_slice]
        last_density = np.sum(self.field_data["phi_hat"][-1], axis=0)[:, :, z_slice]
        
        # Sample points for scatter plot (every 4th point to avoid overcrowding)
        sample_mask = np.zeros_like(last_density, dtype=bool)
        sample_mask[::4, ::4] = True
        
        ax.scatter(last_density[sample_mask], last_nutrient[sample_mask], alpha=0.6, s=10)
        ax.set_xlabel('Tumor Density')
        ax.set_ylabel('Nutrient Concentration')
        ax.set_title(f'Nutrient vs Density (t={self.saved_times[-1]:.2f})')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f'nutrient_tumor_correlation_z{z_slice}.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            print(f"Nutrient-tumor correlation plot saved to {output_dir}/{filename}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        # Return data for use by plot_all_observables and export
        return {
            'total_densities': total_densities,
            'population_densities': density_data,
            'times': self.saved_times,
            'correlations': correlations,
            'tumor_mean_nutrients': tumor_mean_nutrients,
            'background_mean_nutrients': background_mean_nutrients
        }
    
    def plot_source_field_evolution(self, output_dir=None, save_plot=False, show_plot=True,
                                  figsize=(15, 10), z_slice=None, cmap="RdBu_r", 
                                  add_tumor_contours=False, tumor_threshold=0.1,
                                  max_plots=6, include_statistics=True, population_idx=0):
        """
        Plot source term field evolution over time.
        
        Args:
            output_dir: Directory to save plot
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            figsize: Figure size
            z_slice: Z-slice to plot (defaults to center)
            cmap: Colormap for the source field (RdBu_r is good for positive/negative values)
            add_tumor_contours: Whether to add tumor boundary contours
            tumor_threshold: Density threshold for tumor boundary
            max_plots: Maximum number of time points to plot
            include_statistics: Whether to include statistics text on plots
            population_idx: Index of population to plot (default 0)
        """
        print("Creating source field evolution plot...")
        
        # Check if source fields are available in physics data
        if "physics_data" not in self.simulation_data or self.simulation_data["physics_data"] is None:
            print("Warning: No physics data found in simulation data.")
            print("Available data keys:", list(self.simulation_data.keys()))
            return
        
        # Check if source terms are available in physics data
        physics_data = self.simulation_data["physics_data"]
        if physics_data is None or len(physics_data) == 0 or "source_terms" not in physics_data[0]:
            print("Warning: No source terms found in physics data.")
            print("Available physics data keys:", list(physics_data[0].keys()) if len(physics_data) > 0 else "No physics data")
            return
        
        # Select time points to plot
        if len(self.saved_steps) <= max_plots:
            plot_indices = list(range(len(self.saved_steps)))
        else:
            # Select evenly spaced time points
            plot_indices = np.linspace(0, len(self.saved_steps)-1, max_plots, dtype=int)
        
        # Calculate subplot layout
        num_plots = len(plot_indices)
        num_cols = min(3, num_plots)
        num_rows = (num_plots + num_cols - 1) // num_cols
        
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
            z_slice = self.grid_size[2] // 2
        
        # Create coordinate grids
        x = np.arange(self.grid_size[0]) * self.dx
        y = np.arange(self.grid_size[1]) * self.dx
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Find global min/max for consistent color scaling
        all_source_data = []
        for step_idx in range(len(self.saved_steps)):
            source_field = physics_data[step_idx]["source_terms"][population_idx]
            source_slice = source_field[:, :, z_slice]
            all_source_data.append(source_slice)
        
        vmin = np.min(all_source_data)
        vmax = np.max(all_source_data)
        
        # Ensure symmetric color scaling around zero if source terms can be negative
        if vmin < 0 and vmax > 0:
            abs_max = max(abs(vmin), abs(vmax))
            vmin = -abs_max
            vmax = abs_max
        
        # Plot each time step
        for i, step_idx in enumerate(plot_indices):
            ax = axes_flat[i]
            
            # Get step information
            step = self.saved_steps[step_idx]
            time = self.saved_times[step_idx]
            
            # Get source field for this step
            source_field = physics_data[step_idx]["source_terms"][population_idx]
            source_slice = source_field[:, :, z_slice]
            
            # Create the plot with consistent color scaling
            im = ax.imshow(source_slice, extent=[x[0], x[-1], y[0], y[-1]], 
                          origin='lower', cmap=cmap, aspect='equal', vmin=vmin, vmax=vmax)
            
            # Add tumor boundary contours if requested
            if add_tumor_contours:
                # Get total tumor density for boundary detection
                phi_hat = self.field_data["phi_hat"][step_idx]
                total_density = np.sum(phi_hat, axis=0)
                density_slice = total_density[:, :, z_slice]
                
                # Add tumor boundary contour
                ax.contour(X, Y, density_slice, levels=[tumor_threshold], 
                          colors='black', linewidths=2, alpha=0.8)
            
            # Add statistics text if requested
            if include_statistics:
                stats_text = f"Min: {np.min(source_slice):.3f}\nMax: {np.max(source_slice):.3f}\nMean: {np.mean(source_slice):.3f}"
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Set title and labels
            ax.set_title(f'Step {step} (t={time:.2f})')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            
            # Add colorbar to the last plot
            if i == num_plots - 1:
                fig.colorbar(im, ax=ax, label='Source Term')
        
        # Hide unused subplots
        for i in range(num_plots, len(axes_flat)):
            axes_flat[i].set_visible(False)
        
        # Set overall title
        fig.suptitle(f'Source Field Evolution (z={z_slice})', fontsize=16)
        
        plt.tight_layout()
        
        if save_plot and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f'source_field_evolution_z{z_slice}.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            print(f"Source field evolution plot saved to {output_dir}/{filename}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_source_statistics_evolution(self, output_dir=None, save_plot=False, show_plot=True,
                                       figsize=(15, 10), z_slice=None, population_idx=0):
        """
        Plot source field statistics evolution over time.
        
        Args:
            output_dir: Directory to save plot
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            figsize: Figure size
            z_slice: Z-slice to analyze (defaults to center)
            population_idx: Index of population to analyze (default 0)
        """
        print("Calculating source field statistics evolution...")
        
        # Check if source fields are available in physics data
        if "physics_data" not in self.simulation_data or self.simulation_data["physics_data"] is None:
            print("Warning: No physics data found in simulation data.")
            print("Available data keys:", list(self.simulation_data.keys()))
            return
        
        # Check if source terms are available in physics data
        physics_data = self.simulation_data["physics_data"]
        if physics_data is None or len(physics_data) == 0 or "source_terms" not in physics_data[0]:
            print("Warning: No source terms found in physics data.")
            print("Available physics data keys:", list(physics_data[0].keys()) if len(physics_data) > 0 else "No physics data")
            return
        
        # Set default z_slice if not provided
        if z_slice is None:
            z_slice = self.grid_size[2] // 2
        
        # Calculate statistics for each time step
        min_sources = []
        max_sources = []
        mean_sources = []
        total_sources = []
        positive_sources = []
        negative_sources = []
        
        for step_idx in range(len(self.saved_steps)):
            source_field = physics_data[step_idx]["source_terms"][population_idx]
            source_slice = source_field[:, :, z_slice]
            
            min_sources.append(np.min(source_slice))
            max_sources.append(np.max(source_slice))
            mean_sources.append(np.mean(source_slice))
            total_sources.append(np.sum(source_slice))
            
            # Calculate positive and negative contributions
            positive_mask = source_slice > 0
            negative_mask = source_slice < 0
            positive_sources.append(np.sum(source_slice[positive_mask]))
            negative_sources.append(np.sum(source_slice[negative_mask]))
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Plot 1: Min source
        ax = axes[0, 0]
        ax.plot(self.saved_times, min_sources, 'b-', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Time')
        ax.set_ylabel('Min Source')
        ax.set_title('Minimum Source Term')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Max source
        ax = axes[0, 1]
        ax.plot(self.saved_times, max_sources, 'r-', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Time')
        ax.set_ylabel('Max Source')
        ax.set_title('Maximum Source Term')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Mean source
        ax = axes[0, 2]
        ax.plot(self.saved_times, mean_sources, 'g-', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Time')
        ax.set_ylabel('Mean Source')
        ax.set_title('Mean Source Term')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Total source
        ax = axes[1, 0]
        ax.plot(self.saved_times, total_sources, 'm-', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Time')
        ax.set_ylabel('Total Source')
        ax.set_title('Total Source Term')
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Positive vs Negative contributions
        ax = axes[1, 1]
        ax.plot(self.saved_times, positive_sources, 'g-', linewidth=2, marker='o', markersize=4, label='Positive')
        ax.plot(self.saved_times, negative_sources, 'r-', linewidth=2, marker='s', markersize=4, label='Negative')
        ax.set_xlabel('Time')
        ax.set_ylabel('Source Contribution')
        ax.set_title('Positive vs Negative Sources')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Net source (positive + negative)
        ax = axes[1, 2]
        net_sources = np.array(positive_sources) + np.array(negative_sources)
        ax.plot(self.saved_times, net_sources, 'k-', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Time')
        ax.set_ylabel('Net Source')
        ax.set_title('Net Source Term')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_plot and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f'source_statistics_evolution_z{z_slice}.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            print(f"Source statistics evolution plot saved to {output_dir}/{filename}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        # Return data for use by plot_all_observables and export
        return {
            'total_densities': total_densities,
            'population_densities': density_data,
            'times': self.saved_times,
            'min_sources': min_sources,
            'max_sources': max_sources,
            'mean_sources': mean_sources,
            'total_sources': total_sources,
            'positive_sources': positive_sources,
            'negative_sources': negative_sources
        }
    
    def plot_source_tumor_correlation(self, output_dir=None, save_plot=False, show_plot=True,
                                    figsize=(15, 10), z_slice=None, tumor_threshold=0.1, population_idx=0):
        """
        Plot correlation between source field and tumor density.
        
        Args:
            output_dir: Directory to save plot
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            figsize: Figure size
            z_slice: Z-slice to analyze (defaults to center)
            tumor_threshold: Density threshold for tumor region
            population_idx: Index of population to analyze (default 0)
        """
        print("Analyzing source-tumor correlation...")
        
        # Check if source fields are available in physics data
        if "physics_data" not in self.simulation_data or self.simulation_data["physics_data"] is None:
            print("Warning: No physics data found in simulation data.")
            print("Available data keys:", list(self.simulation_data.keys()))
            return
        
        # Check if source terms are available in physics data
        physics_data = self.simulation_data["physics_data"]
        if physics_data is None or len(physics_data) == 0 or "source_terms" not in physics_data[0]:
            print("Warning: No source terms found in physics data.")
            print("Available physics data keys:", list(physics_data[0].keys()) if len(physics_data) > 0 else "No physics data")
            return
        
        # Set default z_slice if not provided
        if z_slice is None:
            z_slice = self.grid_size[2] // 2
        
        # Calculate correlation metrics for each time step
        correlations = []
        tumor_mean_sources = []
        background_mean_sources = []
        
        for step_idx in range(len(self.saved_steps)):
            # Get source field
            source_field = physics_data[step_idx]["source_terms"][population_idx]
            source_slice = source_field[:, :, z_slice]
            
            # Get tumor density
            phi_hat = self.field_data["phi_hat"][step_idx]
            total_density = np.sum(phi_hat, axis=0)
            density_slice = total_density[:, :, z_slice]
            
            # Create tumor mask
            tumor_mask = density_slice > tumor_threshold
            background_mask = ~tumor_mask
            
            # Calculate correlation
            if np.any(tumor_mask) and np.any(background_mask):
                correlation = np.corrcoef(density_slice.flatten(), source_slice.flatten())[0, 1]
                correlations.append(correlation if not np.isnan(correlation) else 0.0)
                
                # Calculate mean source levels
                tumor_mean_sources.append(np.mean(source_slice[tumor_mask]))
                background_mean_sources.append(np.mean(source_slice[background_mask]))
            else:
                correlations.append(0.0)
                tumor_mean_sources.append(0.0)
                background_mean_sources.append(np.mean(source_slice))
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Correlation over time
        ax = axes[0, 0]
        ax.plot(self.saved_times, correlations, 'b-', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Time')
        ax.set_ylabel('Correlation Coefficient')
        ax.set_title('Source-Tumor Correlation')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Plot 2: Mean source in tumor vs background
        ax = axes[0, 1]
        ax.plot(self.saved_times, tumor_mean_sources, 'r-', linewidth=2, marker='o', markersize=4, label='Tumor Region')
        ax.plot(self.saved_times, background_mean_sources, 'g-', linewidth=2, marker='s', markersize=4, label='Background')
        ax.set_xlabel('Time')
        ax.set_ylabel('Mean Source Term')
        ax.set_title('Mean Source Levels')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Source difference (tumor - background)
        ax = axes[1, 0]
        source_diff = np.array(tumor_mean_sources) - np.array(background_mean_sources)
        ax.plot(self.saved_times, source_diff, 'm-', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Time')
        ax.set_ylabel('Source Difference')
        ax.set_title('Tumor - Background Source Difference')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Plot 4: Scatter plot for last time step
        ax = axes[1, 1]
        last_source = physics_data[-1]["source_terms"][population_idx][:, :, z_slice]
        last_density = np.sum(self.field_data["phi_hat"][-1], axis=0)[:, :, z_slice]
        
        # Sample points for scatter plot (every 4th point to avoid overcrowding)
        sample_mask = np.zeros_like(last_density, dtype=bool)
        sample_mask[::4, ::4] = True
        
        ax.scatter(last_density[sample_mask], last_source[sample_mask], alpha=0.6, s=10)
        ax.set_xlabel('Tumor Density')
        ax.set_ylabel('Source Term')
        ax.set_title(f'Source vs Density (t={self.saved_times[-1]:.2f})')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f'source_tumor_correlation_z{z_slice}.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            print(f"Source-tumor correlation plot saved to {output_dir}/{filename}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        # Return data for use by plot_all_observables and export
        return {
            'total_densities': total_densities,
            'population_densities': density_data,
            
            'times': self.saved_times,
            'correlations': correlations,
            'tumor_mean_sources': tumor_mean_sources,
            'background_mean_sources': background_mean_sources
        }
    
    def export_observables_data(self, output_dir, filename='observables_data.csv',
                                threshold=0.1, method='contour',
                                include_inhibited_radius=True,
                                include_necrotic_radius=True,
                                include_hypoxic_radius=True,
                                growth_threshold=0.01,
                                use_adaptive_threshold=True,
                                threshold_type='percentile',
                                growth_threshold_percentile=75.0,
                                necrotic_threshold_percentile=25.0,
                                drop_fraction=0.5,
                                viable_threshold=0.1):
        """
        Export observables data to CSV file.
        Currently only exports radius data (Total_Radius, Inhibited_Radius, Necrotic_Radius).
        
        Args:
            output_dir: Directory to save data
            filename: Base filename for the data
            threshold: Density threshold for radius calculation (default 0.1)
            method: Method for radius calculation ('contour' or 'mass', default 'contour')
            include_inhibited_radius: Whether to include inhibited radius calculation
            include_necrotic_radius: Whether to include necrotic radius calculation
            include_hypoxic_radius: Whether to include hypoxic radius calculation
            growth_threshold: Absolute threshold for detecting proliferative cells (deprecated, kept for compatibility)
            use_adaptive_threshold: Whether to use adaptive thresholds (deprecated, kept for compatibility)
            threshold_type: Type of adaptive threshold (deprecated, kept for compatibility)
            growth_threshold_percentile: Percentile for proliferative threshold (deprecated, kept for compatibility)
            necrotic_threshold_percentile: Percentile for necrotic threshold (deprecated, kept for compatibility)
            drop_fraction: Fraction of maximum source term to use as threshold (deprecated, kept for compatibility)
            viable_threshold: Nutrient concentration threshold (deprecated, kept for compatibility)
        """
        print("Exporting observables data...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate only radius observables for export
        observables = self.plot_all_observables(
            output_dir=None, 
            save_plot=False, 
            show_plot=False,
            threshold=threshold,
            include_inhibited_radius=include_inhibited_radius,
            include_necrotic_radius=include_necrotic_radius,
            include_hypoxic_radius=include_hypoxic_radius,
            growth_threshold=growth_threshold,
            use_adaptive_threshold=use_adaptive_threshold,
            threshold_type=threshold_type,
            growth_threshold_percentile=growth_threshold_percentile,
            necrotic_threshold_percentile=necrotic_threshold_percentile,
            only_radii=True,  # Only calculate radius data
            drop_fraction=drop_fraction,
            viable_threshold=viable_threshold
        )
        
        # Create time series data
        import pandas as pd
        
        # Prepare data for export
        data = {'Time': self.saved_times, 'Step': self.saved_steps}
        
        # Add radius data
        data['Total_Radius'] = observables['radius']['total_radii']
        for label, radii in observables['radius']['population_radii'].items():
            data[f'{label}_Radius'] = radii
        
        # Add inhibited, necrotic, and hypoxic radii if available
        if observables['radius'].get('inhibited_radii') is not None:
            data['Inhibited_Radius'] = observables['radius']['inhibited_radii']
        if observables['radius'].get('necrotic_radii') is not None:
            data['Necrotic_Radius'] = observables['radius']['necrotic_radii']
        if observables['radius'].get('hypoxic_radii') is not None:
            data['Hypoxic_Radius'] = observables['radius']['hypoxic_radii']
        
        # Create DataFrame and save
        df = pd.DataFrame(data)
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Observables data exported to {filepath}")
        
        return df
    
    def plot_source_term_diagnostic(self, output_dir=None, save_plot=False, show_plot=True,
                                   figsize=(20, 12), z_slice=None, step_idx=-1,
                                   exclude_necrotic=True):
        """
        Diagnostic plot to investigate source term contributions and identify why
        source terms might always be positive.
        
        This plot helps understand:
        - Which populations contribute positively vs negatively
        - Whether necrotic population is masking negative source terms
        - Distribution of source terms across different populations
        
        Args:
            output_dir: Directory to save plot
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            figsize: Figure size
            z_slice: Z-slice to analyze (defaults to center)
            step_idx: Time step index to analyze (default -1 for last step)
            exclude_necrotic: Whether to show analysis with/without necrotic population
        """
        print("Creating source term diagnostic plot...")
        
        # Check if source fields are available
        if "physics_data" not in self.simulation_data or self.simulation_data["physics_data"] is None:
            print("Warning: No physics data found in simulation data.")
            return
        
        physics_data = self.simulation_data["physics_data"]
        if physics_data is None or len(physics_data) == 0 or "source_terms" not in physics_data[0]:
            print("Warning: No source terms found in physics data.")
            return
        
        # Set default z_slice
        if z_slice is None:
            z_slice = self.grid_size[2] // 2
        
        # Get source terms and density for selected time step
        source_terms = physics_data[step_idx]["source_terms"]  # (M, nx, ny, nz)
        phi_hat = self.field_data["phi_hat"][step_idx]
        total_density = np.sum(phi_hat, axis=0)
        
        # Create subplots
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        
        # Get 2D slices
        source_slice = source_terms[:, :, :, z_slice]  # (M, nx, ny)
        density_slice = total_density[:, :, z_slice]  # (nx, ny)
        
        # Plot 1: Total source terms (all populations)
        ax = axes[0, 0]
        total_source_all = np.sum(source_slice, axis=0)
        im = ax.imshow(total_source_all, origin='lower', cmap='RdBu_r', aspect='equal')
        ax.set_title('Total Source Terms (All Populations)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax, label='Source Term')
        
        # Plot 2: Total source terms (viable only, excluding necrotic)
        ax = axes[0, 1]
        if source_slice.shape[0] > 1 and exclude_necrotic:
            total_source_viable = np.sum(source_slice[:-1], axis=0)
            title = 'Total Source Terms (Viable Only)'
        else:
            total_source_viable = total_source_all
            title = 'Total Source Terms'
        vmin = np.min(total_source_viable)
        vmax = np.max(total_source_viable)
        # Ensure symmetric colormap if both positive and negative
        if vmin < 0 and vmax > 0:
            abs_max = max(abs(vmin), abs(vmax))
            vmin, vmax = -abs_max, abs_max
        im = ax.imshow(total_source_viable, origin='lower', cmap='RdBu_r', 
                      aspect='equal', vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax, label='Source Term')
        
        # Plot 3: Source term difference (all - viable)
        ax = axes[0, 2]
        if source_slice.shape[0] > 1:
            necrotic_contribution = source_slice[-1]
            im = ax.imshow(necrotic_contribution, origin='lower', cmap='Reds', 
                          aspect='equal')
            ax.set_title('Necrotic Population Contribution')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.colorbar(im, ax=ax, label='Source Term')
        else:
            ax.text(0.5, 0.5, 'No necrotic\npopulation', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Necrotic Contribution')
        
        # Plot 4-6: Individual population source terms
        num_pops = min(3, source_slice.shape[0])
        for i in range(num_pops):
            ax = axes[1, i]
            pop_source = source_slice[i]
            vmin = np.min(pop_source)
            vmax = np.max(pop_source)
            if vmin < 0 and vmax > 0:
                abs_max = max(abs(vmin), abs(vmax))
                vmin, vmax = -abs_max, abs_max
            im = ax.imshow(pop_source, origin='lower', cmap='RdBu_r', 
                          aspect='equal', vmin=vmin, vmax=vmax)
            pop_label = self.labels[i] if i < len(self.labels) else f'Population {i}'
            ax.set_title(f'{pop_label} Source Terms')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.colorbar(im, ax=ax, label='Source Term')
        
        # Plot 7: Statistics comparison
        ax = axes[2, 0]
        if source_slice.shape[0] > 1 and exclude_necrotic:
            viable_source = np.sum(source_slice[:-1], axis=0)
            all_source = total_source_all
            stats_labels = ['All Populations', 'Viable Only']
            data = [all_source.flatten(), viable_source.flatten()]
        else:
            stats_labels = ['All Populations']
            data = [total_source_all.flatten()]
        
        bp = ax.boxplot(data, labels=stats_labels, patch_artist=True)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_ylabel('Source Term Value')
        ax.set_title('Source Term Distribution')
        ax.grid(True, alpha=0.3)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        
        # Plot 8: Source term histogram
        ax = axes[2, 1]
        if source_slice.shape[0] > 1 and exclude_necrotic:
            viable_source = np.sum(source_slice[:-1], axis=0)
            ax.hist(viable_source.flatten(), bins=50, alpha=0.7, label='Viable Only', color='green')
            ax.hist(total_source_all.flatten(), bins=50, alpha=0.7, label='All Populations', color='blue')
            ax.legend()
        else:
            ax.hist(total_source_all.flatten(), bins=50, alpha=0.7, color='blue')
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Source Term Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Source Term Histogram')
        ax.grid(True, alpha=0.3)
        
        # Plot 9: Statistics table
        ax = axes[2, 2]
        ax.axis('off')
        
        # Calculate statistics
        if source_slice.shape[0] > 1 and exclude_necrotic:
            viable_source = np.sum(source_slice[:-1], axis=0)
            stats_text = "Source Term Statistics:\n\n"
            stats_text += "All Populations:\n"
            stats_text += f"  Min: {np.min(total_source_all):.4f}\n"
            stats_text += f"  Max: {np.max(total_source_all):.4f}\n"
            stats_text += f"  Mean: {np.mean(total_source_all):.4f}\n"
            stats_text += f"  Std: {np.std(total_source_all):.4f}\n"
            stats_text += f"  Negative values: {np.sum(total_source_all < 0)} ({100*np.sum(total_source_all < 0)/total_source_all.size:.1f}%)\n"
            stats_text += f"  Zero values: {np.sum(total_source_all == 0)} ({100*np.sum(total_source_all == 0)/total_source_all.size:.1f}%)\n\n"
            stats_text += "Viable Only:\n"
            stats_text += f"  Min: {np.min(viable_source):.4f}\n"
            stats_text += f"  Max: {np.max(viable_source):.4f}\n"
            stats_text += f"  Mean: {np.mean(viable_source):.4f}\n"
            stats_text += f"  Std: {np.std(viable_source):.4f}\n"
            stats_text += f"  Negative values: {np.sum(viable_source < 0)} ({100*np.sum(viable_source < 0)/viable_source.size:.1f}%)\n"
            stats_text += f"  Zero values: {np.sum(viable_source == 0)} ({100*np.sum(viable_source == 0)/viable_source.size:.1f}%)\n"
        else:
            stats_text = "Source Term Statistics:\n\n"
            stats_text += f"Min: {np.min(total_source_all):.4f}\n"
            stats_text += f"Max: {np.max(total_source_all):.4f}\n"
            stats_text += f"Mean: {np.mean(total_source_all):.4f}\n"
            stats_text += f"Std: {np.std(total_source_all):.4f}\n"
            stats_text += f"Negative values: {np.sum(total_source_all < 0)} ({100*np.sum(total_source_all < 0)/total_source_all.size:.1f}%)\n"
            stats_text += f"Zero values: {np.sum(total_source_all == 0)} ({100*np.sum(total_source_all == 0)/total_source_all.size:.1f}%)\n"
        
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=10, family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        fig.suptitle(f'Source Term Diagnostic (Step {self.saved_steps[step_idx]}, t={self.saved_times[step_idx]:.2f}, z={z_slice})', 
                     fontsize=14)
        plt.tight_layout()
        
        if save_plot and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f'source_term_diagnostic_step{self.saved_steps[step_idx]}_z{z_slice}.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            print(f"Source term diagnostic plot saved to {output_dir}/{filename}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        # Return data for use by plot_all_observables and export
        return {
            'total_source_all': total_source_all,
            'total_source_viable': total_source_viable if (source_slice.shape[0] > 1 and exclude_necrotic) else None,
            'source_per_population': {self.labels[i] if i < len(self.labels) else f'Pop {i}': source_slice[i] 
                                     for i in range(source_slice.shape[0])}
        }
    
    def plot_custom_source_fields(self, output_dir=None, save_plot=False, show_plot=True,
                                  figsize=(15, 10), z_slice=None, cmap="RdBu_r",
                                  add_necrotic_boundary=True, necrotic_threshold=0.1,
                                  add_tumor_contours=False, tumor_threshold=0.1,
                                  max_plots=6, include_statistics=True,
                                  population_indices=None, population_labels=None,
                                  step_idx=None):
        """
        Plot source term fields from customizable populations with necrotic core boundary.
        
        This function allows you to plot source terms from specific populations (e.g., necrotic)
        and overlay the necrotic core boundary. You can specify populations by index or label.
        
        Args:
            output_dir: Directory to save plot
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            figsize: Figure size
            z_slice: Z-slice to plot (defaults to center)
            cmap: Colormap for the source field (RdBu_r is good for positive/negative values)
            add_necrotic_boundary: Whether to add necrotic core boundary contour
            necrotic_threshold: Density threshold for necrotic core boundary
            add_tumor_contours: Whether to add total tumor boundary contours
            tumor_threshold: Density threshold for tumor boundary
            max_plots: Maximum number of time points to plot (if step_idx is None)
            include_statistics: Whether to include statistics text on plots
            population_indices: List of population indices to plot (e.g., [0, 2] or [-1] for necrotic)
                               If None, plots all populations
            population_labels: List of population labels to plot (alternative to indices)
                              If both indices and labels are provided, indices take precedence
            step_idx: Specific time step index to plot (default None = plot evolution over time)
                     If provided, max_plots is ignored
        """
        print("Creating custom source field plot...")
        
        # Check if source fields are available in physics data
        if "physics_data" not in self.simulation_data or self.simulation_data["physics_data"] is None:
            print("Warning: No physics data found in simulation data.")
            print("Available data keys:", list(self.simulation_data.keys()))
            return
        
        # Check if source terms are available in physics data
        physics_data = self.simulation_data["physics_data"]
        if physics_data is None or len(physics_data) == 0 or "source_terms" not in physics_data[0]:
            print("Warning: No source terms found in physics data.")
            print("Available physics data keys:", list(physics_data[0].keys()) if len(physics_data) > 0 else "No physics data")
            return
        
        # Determine which populations to plot
        if population_indices is not None:
            # Use provided indices
            pop_indices = population_indices if isinstance(population_indices, list) else [population_indices]
            # Handle negative indices (e.g., -1 for last population/necrotic)
            pop_indices = [idx if idx >= 0 else len(self.labels) + idx for idx in pop_indices]
        elif population_labels is not None:
            # Convert labels to indices
            pop_indices = []
            labels_list = population_labels if isinstance(population_labels, list) else [population_labels]
            for label in labels_list:
                if label in self.labels:
                    pop_indices.append(self.labels.index(label))
                else:
                    print(f"Warning: Population label '{label}' not found. Available labels: {self.labels}")
        else:
            # Default: plot all populations
            pop_indices = list(range(len(self.labels)))
        
        # Validate indices
        valid_indices = [idx for idx in pop_indices if 0 <= idx < len(self.labels)]
        if not valid_indices:
            print(f"Warning: No valid population indices. Available range: 0-{len(self.labels)-1}")
            return
        
        pop_indices = valid_indices
        print(f"Plotting source fields for populations: {[self.labels[i] for i in pop_indices]}")
        
        # Determine time steps to plot
        if step_idx is not None:
            # Plot single time step
            if step_idx < 0:
                step_idx = len(self.saved_steps) + step_idx
            if step_idx < 0 or step_idx >= len(self.saved_steps):
                print(f"Warning: step_idx {step_idx} out of range. Using last step.")
                step_idx = len(self.saved_steps) - 1
            plot_indices = [step_idx]
        else:
            # Plot evolution over time
            if len(self.saved_steps) <= max_plots:
                plot_indices = list(range(len(self.saved_steps)))
            else:
                plot_indices = np.linspace(0, len(self.saved_steps)-1, max_plots, dtype=int)
        
        # Calculate subplot layout
        # Layout: rows = time steps, cols = populations
        num_time_steps = len(plot_indices)
        num_populations = len(pop_indices)
        num_cols = num_populations
        num_rows = num_time_steps
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
        # Handle different matplotlib subplot return types
        # Convert to consistent 2D list format: axes_flat[row][col]
        if num_time_steps == 1 and num_populations == 1:
            # Single subplot returns an Axes object
            axes_flat = [[axes]]
        elif num_time_steps == 1:
            # Single row, multiple columns - returns 1D array or list
            if isinstance(axes, np.ndarray):
                axes_flat = [axes.flatten().tolist()]
            else:
                axes_flat = [[axes] if num_populations == 1 else list(axes)]
        elif num_populations == 1:
            # Multiple rows, single column - returns 1D array or list
            if isinstance(axes, np.ndarray):
                axes_flat = [[ax] for ax in axes.flatten()]
            else:
                axes_flat = [[axes] if num_time_steps == 1 else [[ax] for ax in axes]]
        else:
            # Multiple rows and columns - returns 2D array
            if isinstance(axes, np.ndarray):
                axes_flat = axes.tolist()
            else:
                # Fallback: try to reshape
                axes_list = np.array(axes).reshape(num_rows, num_cols).tolist()
                axes_flat = axes_list
        
        # Set default z_slice if not provided
        if z_slice is None:
            z_slice = self.grid_size[2] // 2
        
        # Create coordinate grids
        x = np.arange(self.grid_size[0]) * self.dx
        y = np.arange(self.grid_size[1]) * self.dx
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Find global min/max for consistent color scaling across all populations and time steps
        all_source_data = []
        for plot_step_idx in plot_indices:
            for pop_idx in pop_indices:
                source_field = physics_data[plot_step_idx]["source_terms"][pop_idx]
                source_slice = source_field[:, :, z_slice]
                all_source_data.append(source_slice)
        
        vmin = np.min(all_source_data)
        vmax = np.max(all_source_data)
        
        # Ensure symmetric color scaling around zero if source terms can be negative
        if vmin < 0 and vmax > 0:
            abs_max = max(abs(vmin), abs(vmax))
            vmin = -abs_max
            vmax = abs_max
        
        # Plot each time step and population combination
        for row, plot_step_idx in enumerate(plot_indices):
            step = self.saved_steps[plot_step_idx]
            time = self.saved_times[plot_step_idx]
            
            # Get density fields for contours
            phi_hat = self._get_field_slice("phi_hat", plot_step_idx)
            total_density = np.sum(phi_hat, axis=0)
            density_slice = total_density[:, :, z_slice]
            
            # Get necrotic density if needed
            necrotic_density_slice = None
            if add_necrotic_boundary and len(self.labels) > 1:
                necrotic_pop_idx = len(self.labels) - 1  # Last population is necrotic
                necrotic_density_slice = phi_hat[necrotic_pop_idx][:, :, z_slice]
            
            for col, pop_idx in enumerate(pop_indices):
                # Get the correct axis
                if num_time_steps == 1 and num_populations == 1:
                    ax = axes_flat[0][0]
                elif num_time_steps == 1:
                    ax = axes_flat[0][col]
                elif num_populations == 1:
                    ax = axes_flat[row][0]
                else:
                    ax = axes_flat[row][col]
                
                # Get source field for this step and population
                source_field = physics_data[plot_step_idx]["source_terms"][pop_idx]
                source_slice = source_field[:, :, z_slice]
                
                # Create the plot with consistent color scaling
                im = ax.imshow(source_slice, extent=[x[0], x[-1], y[0], y[-1]], 
                              origin='lower', cmap=cmap, aspect='equal', vmin=vmin, vmax=vmax)
                
                # Add necrotic core boundary contour if requested
                if add_necrotic_boundary and necrotic_density_slice is not None:
                    ax.contour(X, Y, necrotic_density_slice, levels=[necrotic_threshold], 
                              colors='red', linewidths=2.5, alpha=0.9, linestyles='solid',
                              label='Necrotic Core')
                
                # Add total tumor boundary contours if requested
                if add_tumor_contours:
                    ax.contour(X, Y, density_slice, levels=[tumor_threshold], 
                              colors='black', linewidths=2, alpha=0.8, linestyles='--',
                              label='Tumor Boundary')
                
                # Add statistics text if requested
                if include_statistics:
                    stats_text = f"Min: {np.min(source_slice):.3f}\nMax: {np.max(source_slice):.3f}\nMean: {np.mean(source_slice):.3f}"
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=8)
                
                # Set title and labels
                pop_label = self.labels[pop_idx]
                if num_time_steps == 1:
                    # Single time step: show population in title
                    ax.set_title(f'{pop_label}\nStep {step} (t={time:.2f})', fontsize=10)
                elif num_populations == 1:
                    # Single population: show time in title
                    ax.set_title(f'Step {step} (t={time:.2f})', fontsize=10)
                else:
                    # Multiple populations and time steps
                    if row == 0:
                        ax.set_title(f'{pop_label}', fontsize=10)
                    if col == 0:
                        ax.set_ylabel(f'Step {step}\n(t={time:.2f})', fontsize=9)
                
                ax.set_xlabel('X')
                if col == 0:
                    ax.set_ylabel('Y')
                
                # Add colorbar to the last plot
                if row == num_time_steps - 1 and col == num_populations - 1:
                    fig.colorbar(im, ax=ax, label='Source Term')
        
        # Set overall title
        pop_labels_str = ', '.join([self.labels[i] for i in pop_indices])
        if step_idx is not None and len(plot_indices) == 1:
            title = f'Source Fields: {pop_labels_str} (Step {self.saved_steps[plot_indices[0]]}, z={z_slice})'
        else:
            title = f'Source Field Evolution: {pop_labels_str} (z={z_slice})'
        fig.suptitle(title, fontsize=14)
        
        plt.tight_layout()
        
        if save_plot and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            pop_str = '_'.join([self.labels[i] for i in pop_indices])
            if step_idx is not None and len(plot_indices) == 1:
                filename = f'source_fields_{pop_str}_step{self.saved_steps[plot_indices[0]]}_z{z_slice}.png'
            else:
                filename = f'source_fields_{pop_str}_evolution_z{z_slice}.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            print(f"Custom source field plot saved to {output_dir}/{filename}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        # Return data for potential use
        return {
            'population_indices': pop_indices,
            'population_labels': [self.labels[i] for i in pop_indices],
            'plot_indices': plot_indices,
            'z_slice': z_slice
        }
    
    def plot_radius_diagnostic(self, output_dir=None, save_plot=False, show_plot=True,
                              figsize=(20, 12), z_slice=None, step_indices=None,
                              threshold=0.1, method='contour', num_radial_bins=100):
        """
        Diagnostic plot to investigate why individual population radii differ from total radius.
        
        This plot helps understand:
        - Radial density profiles for each population vs total density
        - Where threshold is crossed for individual vs total
        - How necrotic cells at the edge affect total radius
        - Why individual population radius might plateau while total continues growing
        
        Args:
            output_dir: Directory to save plot
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            figsize: Figure size
            z_slice: Z-slice to analyze (defaults to center)
            step_indices: List of time step indices to analyze (default: [0, -1] for first and last)
            threshold: Density threshold used for radius calculation
            method: Method for radius calculation ('contour' or 'mass')
            num_radial_bins: Number of radial bins for profile analysis
        """
        print("Creating radius diagnostic plot...")
        
        # Set default z_slice
        if z_slice is None:
            z_slice = self.grid_size[2] // 2
        
        # Set default step_indices (start and end only)
        if step_indices is None:
            step_indices = [0, -1]
        elif isinstance(step_indices, int):
            step_indices = [step_indices]
        
        # Normalize negative indices
        step_indices = [idx if idx >= 0 else len(self.saved_steps) + idx for idx in step_indices]
        step_indices = [max(0, min(idx, len(self.saved_steps) - 1)) for idx in step_indices]
        
        # Remove duplicates while preserving order
        seen = set()
        step_indices = [x for x in step_indices if not (x in seen or seen.add(x))]
        
        # Create subplots: rows = time steps, cols = [radial profiles, edge region detail]
        num_steps = len(step_indices)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(num_steps, 2, hspace=0.3, wspace=0.3)
        
        # Store results for return
        diagnostic_results = {
            'step_indices': step_indices,
            'individual_radii': {},
            'total_radii': {},
            'z_slice': z_slice,
            'threshold': threshold
        }
        
        for row, step_idx in enumerate(step_indices):
            # Get data for this time step
            phi_hat = self._get_field_slice("phi_hat", step_idx)
            total_density = np.sum(phi_hat, axis=0)
            
            # Get 2D slice
            total_density_slice = total_density[:, :, z_slice]
            
            # Find center of mass
            com = self.utils.calculate_center_of_mass(total_density)
            
            # Calculate radial distances from center
            x_coords = np.arange(self.grid_size[0]) * self.dx
            y_coords = np.arange(self.grid_size[1]) * self.dx
            X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
            r_squared = (X - com[0])**2 + (Y - com[1])**2
            radial_distances = np.sqrt(r_squared)
            
            # Calculate radii for this step (using full 3D field)
            total_radius = self.utils.calculate_radius(total_density, threshold=threshold, method=method)
            individual_radii = {}
            for i, label in enumerate(self.labels):
                individual_radii[label] = self.utils.calculate_radius(phi_hat[i], threshold=threshold, method=method)
            
            # Also calculate radii from the 2D slice for comparison
            total_radius_2d = self.utils.calculate_radius(total_density_slice[..., None], threshold=threshold, method=method)
            individual_radii_2d = {}
            for i, label in enumerate(self.labels):
                pop_slice_3d = phi_hat[i][:, :, z_slice][..., None]  # Add z dimension back
                individual_radii_2d[label] = self.utils.calculate_radius(pop_slice_3d, threshold=threshold, method=method)
            
            # Store results
            diagnostic_results['total_radii'][step_idx] = total_radius
            diagnostic_results['individual_radii'][step_idx] = individual_radii.copy()
            diagnostic_results['total_radii_2d'] = diagnostic_results.get('total_radii_2d', {})
            diagnostic_results['total_radii_2d'][step_idx] = total_radius_2d
            diagnostic_results['individual_radii_2d'] = diagnostic_results.get('individual_radii_2d', {})
            diagnostic_results['individual_radii_2d'][step_idx] = individual_radii_2d.copy()
            
            # Bin data by radial distance
            max_radius = np.max(radial_distances)
            radial_bins = np.linspace(0, max_radius, num_radial_bins + 1)
            radial_centers = (radial_bins[:-1] + radial_bins[1:]) / 2
            
            # Calculate average densities in each radial bin
            avg_total_density = np.zeros(num_radial_bins)
            avg_population_densities = {label: np.zeros(num_radial_bins) for label in self.labels}
            
            for i in range(num_radial_bins):
                r_min = radial_bins[i]
                r_max = radial_bins[i + 1]
                mask = (radial_distances >= r_min) & (radial_distances < r_max)
                
                if np.any(mask):
                    avg_total_density[i] = np.mean(total_density_slice[mask])
                    for j, label in enumerate(self.labels):
                        pop_slice = phi_hat[j][:, :, z_slice]
                        avg_population_densities[label][i] = np.mean(pop_slice[mask])
            
            # Plot 1: Radial density profiles
            ax1 = fig.add_subplot(gs[row, 0])
            ax1.plot(radial_centers, avg_total_density, 'k-', linewidth=3, label='Total Density', zorder=10)
            
            colors = plt.cm.Set1(np.linspace(0, 1, len(self.labels)))
            for i, label in enumerate(self.labels):
                ax1.plot(radial_centers, avg_population_densities[label], '--', 
                        color=colors[i], linewidth=2, label=f'{label}', alpha=0.8)
            
            # Add threshold line
            ax1.axhline(y=threshold, color='r', linestyle=':', linewidth=2, alpha=0.7, label=f'Threshold ({threshold})')
            
            # Mark calculated radii (3D and 2D for comparison)
            ax1.axvline(x=total_radius, color='k', linestyle='-', linewidth=2, alpha=0.5, 
                       label=f'Total Radius 3D ({total_radius:.2f})')
            ax1.axvline(x=total_radius_2d, color='k', linestyle=':', linewidth=2, alpha=0.5, 
                       label=f'Total Radius 2D ({total_radius_2d:.2f})')
            for i, label in enumerate(self.labels):
                ax1.axvline(x=individual_radii[label], color=colors[i], linestyle='--', 
                           linewidth=1.5, alpha=0.5, label=f'{label} Radius 3D ({individual_radii[label]:.2f})')
                ax1.axvline(x=individual_radii_2d[label], color=colors[i], linestyle=':', 
                           linewidth=1.5, alpha=0.5, label=f'{label} Radius 2D ({individual_radii_2d[label]:.2f})')
            
            ax1.set_xlabel('Radial Distance')
            ax1.set_ylabel('Average Density')
            ax1.set_title(f'Radial Profile (Step {self.saved_steps[step_idx]}, t={self.saved_times[step_idx]:.2f})')
            ax1.legend(fontsize=8, loc='best')
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(0, max_radius)
            
            # Plot 2: Edge region analysis (zoom on edge)
            ax2 = fig.add_subplot(gs[row, 1])
            # Focus on edge region (around total radius)
            edge_region_radius = total_radius * 1.2  # Show 20% beyond total radius
            edge_mask = radial_distances <= edge_region_radius
            
            if np.any(edge_mask):
                # Get edge region data
                edge_radial = radial_distances[edge_mask]
                edge_total = total_density_slice[edge_mask]
                edge_populations = {}
                for j, label in enumerate(self.labels):
                    edge_populations[label] = phi_hat[j][:, :, z_slice][edge_mask]
                
                # Scatter plot of densities vs radial distance
                # Use larger, more distinct markers for better visibility
                ax2.scatter(edge_radial, edge_total, c='black', s=15, alpha=0.5, 
                           label='Total', zorder=10, marker='o', edgecolors='black', linewidths=0.5)
                for j, label in enumerate(self.labels):
                    # Use different marker styles for each population
                    markers = ['s', '^', 'v', 'D', 'p', '*']
                    marker = markers[j % len(markers)]
                    ax2.scatter(edge_radial, edge_populations[label], c=colors[j], s=12, 
                              alpha=0.4, label=label, marker=marker, edgecolors=colors[j], linewidths=0.3)
                
                # Add moving average lines to show trends more clearly
                # Use a smaller window and only average where there's sufficient data
                if len(edge_radial) > 10:
                    # Sort by radial distance for proper moving average
                    sort_idx = np.argsort(edge_radial)
                    sorted_radial = edge_radial[sort_idx]
                    sorted_total = edge_total[sort_idx]
                    sorted_pops = {label: edge_populations[label][sort_idx] for label in self.labels}
                    
                    # Use a window size based on data density (smaller window for better resolution)
                    window_size = max(5, len(edge_radial) // 50)  # Adaptive window size
                    if window_size % 2 == 0:
                        window_size += 1  # Make odd for symmetric window
                    
                    # Calculate moving average
                    def moving_average(data, window):
                        """Calculate moving average with proper edge handling"""
                        result = np.zeros_like(data)
                        half_window = window // 2
                        for i in range(len(data)):
                            start = max(0, i - half_window)
                            end = min(len(data), i + half_window + 1)
                            result[i] = np.mean(data[start:end])
                        return result
                    
                    # Only plot moving average where we have enough points
                    total_ma = moving_average(sorted_total, window_size)
                    pop_ma = {label: moving_average(sorted_pops[label], window_size) 
                             for label in self.labels}
                    
                    # Plot moving averages as lines (thicker, more visible)
                    ax2.plot(sorted_radial, total_ma, 
                            'k-', linewidth=3, alpha=0.9, label='Total (avg)', zorder=15)
                    for j, label in enumerate(self.labels):
                        ax2.plot(sorted_radial, pop_ma[label],
                                '--', color=colors[j], linewidth=2.5, alpha=0.8, 
                                label=f'{label} (avg)', zorder=14)
                
                # Add threshold and radius lines
                ax2.axhline(y=threshold, color='r', linestyle=':', linewidth=2, alpha=0.7)
                # Show both 3D and 2D radii for comparison
                ax2.axvline(x=total_radius, color='k', linestyle='-', linewidth=2, alpha=0.5, 
                           label=f'Total 3D ({total_radius:.1f})')
                ax2.axvline(x=total_radius_2d, color='k', linestyle=':', linewidth=2, alpha=0.7, 
                           label=f'Total 2D ({total_radius_2d:.1f})')
                for j, label in enumerate(self.labels):
                    ax2.axvline(x=individual_radii[label], color=colors[j], linestyle='--', 
                              linewidth=1.5, alpha=0.5, label=f'{label} 3D ({individual_radii[label]:.1f})')
                    ax2.axvline(x=individual_radii_2d[label], color=colors[j], linestyle=':', 
                              linewidth=1.5, alpha=0.7, label=f'{label} 2D ({individual_radii_2d[label]:.1f})')
                
                # Add diagnostic text showing what's above threshold at the edge
                # Check what's actually above threshold near the total radius
                near_total_radius = (radial_distances >= total_radius * 0.95) & (radial_distances <= total_radius * 1.05)
                if np.any(near_total_radius):
                    total_above = np.sum(edge_total > threshold)
                    total_below = np.sum(edge_total <= threshold)
                    pop_above = {}
                    pop_below = {}
                    for j, label in enumerate(self.labels):
                        pop_above[label] = np.sum(edge_populations[label] > threshold)
                        pop_below[label] = np.sum(edge_populations[label] <= threshold)
                    
                    # Also check what the sum of individual populations is
                    sum_individuals = np.zeros_like(edge_total)
                    for j, label in enumerate(self.labels):
                        sum_individuals += edge_populations[label]
                    
                    # Check for discrepancies between sum of individuals and total
                    diff = np.abs(edge_total - sum_individuals)
                    max_diff = np.max(diff)
                    
                    # Check what's happening at the actual total radius boundary
                    # Find points near the total radius
                    near_boundary = (edge_radial >= total_radius * 0.98) & (edge_radial <= total_radius * 1.02)
                    if np.any(near_boundary):
                        boundary_total = edge_total[near_boundary]
                        boundary_sum = sum_individuals[near_boundary]
                        boundary_radial = edge_radial[near_boundary]
                        boundary_pops = {label: edge_populations[label][near_boundary] for label in self.labels}
                        
                        # Find points above threshold near boundary
                        above_thresh = boundary_total > threshold
                        if np.any(above_thresh):
                            above_radial = boundary_radial[above_thresh]
                            above_total = boundary_total[above_thresh]
                            above_sum = boundary_sum[above_thresh]
                            above_pops = {label: boundary_pops[label][above_thresh] for label in self.labels}
                    
                    # Add text box with diagnostic info
                    diag_text = f"Edge Region Stats:\n"
                    diag_text += f"Total > {threshold}: {total_above}\n"
                    diag_text += f"Total <= {threshold}: {total_below}\n"
                    for j, label in enumerate(self.labels):
                        diag_text += f"{label} > {threshold}: {pop_above[label]}\n"
                    diag_text += f"\nSum vs Total:\n"
                    diag_text += f"Max diff: {max_diff:.6f}\n"
                    diag_text += f"Sum max: {np.max(sum_individuals):.4f}\n"
                    diag_text += f"Total max: {np.max(edge_total):.4f}\n"
                    if np.any(near_boundary) and np.any(above_thresh):
                        diag_text += f"\nNear boundary (>thresh):\n"
                        diag_text += f"Radial: {np.min(above_radial):.1f}-{np.max(above_radial):.1f}\n"
                        diag_text += f"Total: {np.min(above_total):.4f}-{np.max(above_total):.4f}\n"
                        diag_text += f"Sum: {np.min(above_sum):.4f}-{np.max(above_sum):.4f}\n"
                        for j, label in enumerate(self.labels):
                            diag_text += f"{label}: {np.min(above_pops[label]):.4f}-{np.max(above_pops[label]):.4f}\n"
                    
                    ax2.text(0.02, 0.98, diag_text, transform=ax2.transAxes,
                           verticalalignment='top', fontsize=7, family='monospace',
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
                
                ax2.set_xlabel('Radial Distance')
                ax2.set_ylabel('Density')
                ax2.set_title('Edge Region Detail')
                ax2.legend(fontsize=7, loc='best')
                ax2.grid(True, alpha=0.3)
                ax2.set_xlim(max(0, total_radius * 0.7), edge_region_radius)
            else:
                ax2.text(0.5, 0.5, 'No edge\nregion data', ha='center', va='center',
                        transform=ax2.transAxes, fontsize=12)
                ax2.set_title('Edge Region Detail')
        
        fig.suptitle(f'Radius Diagnostic Analysis (z={z_slice}, threshold={threshold})', fontsize=16)
        
        if save_plot and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f'radius_diagnostic_z{z_slice}_threshold{threshold}.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            print(f"Radius diagnostic plot saved to {output_dir}/{filename}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        # Return diagnostic data
        return diagnostic_results