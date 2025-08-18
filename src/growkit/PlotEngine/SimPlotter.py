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
        self.dx = 1.0  # Default value, should be extracted from config if available
        
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
        
        # Initialize observable utilities
        self.utils = ObservableUtils(self.grid_size, self.dx)
    
    def plot_tumor_radius_evolution(self, output_dir=None, save_plot=False, show_plot=True,
                                   figsize=(12, 8), include_individual_populations=True,
                                   threshold=0.1, method='contour'):
        """
        Plot tumor radius evolution over time.
        
        Args:
            output_dir: Directory to save plot
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            figsize: Figure size
            include_individual_populations: Whether to plot individual population radii
            threshold: Density threshold for radius calculation
            method: Method for radius calculation ('contour' or 'mass')
        """
        print("Calculating tumor radius evolution...")
        
        # Calculate radii for all populations
        radii_data = {}
        for i, label in enumerate(self.labels):
            radii = []
            for step_idx in range(len(self.saved_steps)):
                phi_hat = self.field_data["phi_hat"][step_idx]
                radius = self.utils.calculate_radius(phi_hat[i], threshold=threshold, method=method)
                radii.append(radius)
            radii_data[label] = radii
        
        # Calculate total tumor radius
        total_radii = []
        for step_idx in range(len(self.saved_steps)):
            phi_hat = self.field_data["phi_hat"][step_idx]
            total_density = np.sum(phi_hat, axis=0)
            radius = self.utils.calculate_radius(total_density, threshold=threshold, method=method)
            total_radii.append(radius)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot total radius
        ax.plot(self.saved_times, total_radii, 'k-', linewidth=3, label='Total Tumor', marker='o', markersize=6)
        
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
        
        return
    
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
        
        return
    
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
        
        return
    
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
        
        return
    
    def plot_all_observables(self, output_dir=None, save_plot=False, show_plot=True,
                           figsize=(20, 15), threshold=0.1):
        """
        Create a comprehensive plot of all observables.
        
        Args:
            output_dir: Directory to save plot
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            figsize: Figure size
            threshold: Density threshold for radius calculation
        """
        print("Creating comprehensive observables plot...")
        
        # Calculate all observables
        observables = {}
        
        # Radius evolution
        radius_data = self.plot_tumor_radius_evolution(output_dir=None, save_plot=False, 
                                                      show_plot=False, threshold=threshold)
        observables['radius'] = radius_data
        
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
        
        # Create subplot grid
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Plot 1: Radius evolution
        ax = axes[0, 0]
        ax.plot(self.saved_times, radius_data['total_radii'], 'k-', linewidth=3, label='Total', marker='o', markersize=4)
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.labels)))
        for i, (label, radii) in enumerate(radius_data['population_radii'].items()):
            ax.plot(self.saved_times, radii, '--', color=colors[i], linewidth=2, label=f'{label}', marker='s', markersize=3)
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
        
        return
    
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
        
        return {
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
        
        return {
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
        if "physics_data" not in self.simulation_data:
            print("Warning: No physics data found in simulation data.")
            print("Available data keys:", list(self.simulation_data.keys()))
            return
        
        # Check if source terms are available in physics data
        physics_data = self.simulation_data["physics_data"]
        if len(physics_data) == 0 or "source_terms" not in physics_data[0]:
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
        if "physics_data" not in self.simulation_data:
            print("Warning: No physics data found in simulation data.")
            print("Available data keys:", list(self.simulation_data.keys()))
            return
        
        # Check if source terms are available in physics data
        physics_data = self.simulation_data["physics_data"]
        if len(physics_data) == 0 or "source_terms" not in physics_data[0]:
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
        
        return {
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
        if "physics_data" not in self.simulation_data:
            print("Warning: No physics data found in simulation data.")
            print("Available data keys:", list(self.simulation_data.keys()))
            return
        
        # Check if source terms are available in physics data
        physics_data = self.simulation_data["physics_data"]
        if len(physics_data) == 0 or "source_terms" not in physics_data[0]:
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
        
        return {
            'times': self.saved_times,
            'correlations': correlations,
            'tumor_mean_sources': tumor_mean_sources,
            'background_mean_sources': background_mean_sources
        }
    
    def export_observables_data(self, output_dir, filename='observables_data.csv'):
        """
        Export all observables data to CSV files.
        
        Args:
            output_dir: Directory to save data
            filename: Base filename for the data
        """
        print("Exporting observables data...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate all observables
        observables = self.plot_all_observables(output_dir=None, save_plot=False, show_plot=False)
        
        # Create time series data
        import pandas as pd
        
        # Prepare data for export
        data = {'Time': self.saved_times, 'Step': self.saved_steps}
        
        # Add radius data
        data['Total_Radius'] = observables['radius']['total_radii']
        for label, radii in observables['radius']['population_radii'].items():
            data[f'{label}_Radius'] = radii
        
        # Add density data
        data['Total_Density'] = observables['density']['total_densities']
        for label, densities in observables['density']['population_densities'].items():
            data[f'{label}_Density'] = densities
        
        # Add center of mass data
        data['Total_COM_X'] = observables['center_of_mass']['total_com']['x']
        data['Total_COM_Y'] = observables['center_of_mass']['total_com']['y']
        data['Total_COM_Z'] = observables['center_of_mass']['total_com']['z']
        for label, com in observables['center_of_mass']['population_com'].items():
            data[f'{label}_COM_X'] = com['x']
            data[f'{label}_COM_Y'] = com['y']
            data[f'{label}_COM_Z'] = com['z']
        
        # Add compactness data
        data['Total_Compactness'] = observables['compactness']['total_compactness']
        for label, compactness in observables['compactness']['population_compactness'].items():
            data[f'{label}_Compactness'] = compactness
        
        # Add nutrient data if available
        if "nutrient_fields" in self.field_data:
            nutrient_stats = self.plot_nutrient_statistics_evolution(
                output_dir=None, save_plot=False, show_plot=False
            )
            data['Nutrient_Min'] = nutrient_stats['min_concentrations']
            data['Nutrient_Max'] = nutrient_stats['max_concentrations']
            data['Nutrient_Mean'] = nutrient_stats['mean_concentrations']
            data['Nutrient_Total'] = nutrient_stats['total_concentrations']
        
        # Add source data if available
        if "physics_data" in self.simulation_data:
            physics_data = self.simulation_data["physics_data"]
            if len(physics_data) > 0 and "source_terms" in physics_data[0]:
                source_stats = self.plot_source_statistics_evolution(
                    output_dir=None, save_plot=False, show_plot=False
                )
                data['Source_Min'] = source_stats['min_sources']
                data['Source_Max'] = source_stats['max_sources']
                data['Source_Mean'] = source_stats['mean_sources']
                data['Source_Total'] = source_stats['total_sources']
                data['Source_Positive'] = source_stats['positive_sources']
                data['Source_Negative'] = source_stats['negative_sources']
        
        # Create DataFrame and save
        df = pd.DataFrame(data)
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Observables data exported to {filepath}")
        
        return df
