import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.collections as mcollections
import os
from pathlib import Path
from .ObservableUtils import ObservableUtils

class FieldAnimator:
    def __init__(self, simulation_data):
        """
        Initialize the field animator.
        
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
    
    def animate_tumor_density(self, output_dir=None, save_animation=False, show_animation=True,
                             figsize=(10, 10), z_slice=None, threshold=0.1, cmap='viridis',
                             interval=200, repeat=True, add_contour=True, contour_color='red',
                             fps=5):
        """
        Animate tumor density field evolution over time.
        
        Args:
            output_dir: Directory to save animation
            save_animation: Whether to save the animation as a file
            show_animation: Whether to display the animation
            figsize: Figure size
            z_slice: Z-slice to animate (defaults to center)
            threshold: Density threshold for contour
            cmap: Colormap for density field
            interval: Animation interval in milliseconds
            repeat: Whether to repeat the animation
            add_contour: Whether to add tumor boundary contour
            contour_color: Color for contour line
            fps: Frames per second for saved animation
        """
        print("Creating tumor density animation...")
        
        # Set default z_slice if not provided
        if z_slice is None:
            z_slice = self.grid_size[2] // 2
        
        # Create coordinate grids
        x = np.arange(self.grid_size[0]) * self.dx
        y = np.arange(self.grid_size[1]) * self.dx
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Find global min/max for consistent color scaling
        all_density_data = []
        for step_idx in range(len(self.saved_steps)):
            phi_hat = self.field_data["phi_hat"][step_idx]
            total_density = np.sum(phi_hat, axis=0)
            density_slice = total_density[:, :, z_slice]
            all_density_data.append(density_slice)
        
        vmin = np.min(all_density_data)
        vmax = np.max(all_density_data)
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        
        # Initialize plot
        density_slice = all_density_data[0]
        im = ax.imshow(density_slice, extent=[x[0], x[-1], y[0], y[-1]], 
                      origin='lower', cmap=cmap, aspect='equal', vmin=vmin, vmax=vmax)
        
        # Initialize contour if requested
        if add_contour:
            ax.contour(X, Y, density_slice, levels=[threshold], 
                      colors=contour_color, linewidths=2, alpha=0.8)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, label='Density')
        
        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Step {self.saved_steps[0]} (t={self.saved_times[0]:.2f})')
        
        # Animation update function
        def animate(frame):
            step_idx = frame
            density_slice = all_density_data[step_idx]
            
            # Update image data
            im.set_array(density_slice)
            
            # Update contour if present
            if add_contour:
                # Remove all LineCollection objects (contours) but keep the image
                collections_to_remove = [c for c in ax.collections if isinstance(c, mcollections.LineCollection)]
                for c in collections_to_remove:
                    ax.collections.remove(c)
                # Add new contour
                ax.contour(X, Y, density_slice, levels=[threshold], 
                          colors=contour_color, linewidths=2, alpha=0.8)
            
            # Update title
            ax.set_title(f'Step {self.saved_steps[step_idx]} (t={self.saved_times[step_idx]:.2f})')
            
            return [im]
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(self.saved_steps),
                                       interval=interval, repeat=repeat, blit=False)
        
        plt.tight_layout()
        
        # Save animation if requested
        if save_animation and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f'tumor_density_animation_z{z_slice}.gif'
            filepath = os.path.join(output_dir, filename)
            print(f"Saving animation to {filepath}...")
            anim.save(filepath, writer='pillow', fps=fps)
            print(f"Animation saved to {filepath}")
        
        if show_animation:
            plt.show()
        else:
            plt.close()
        
        return anim
    
    def animate_nutrient_field(self, output_dir=None, save_animation=False, show_animation=True,
                              figsize=(10, 10), z_slice=None, cmap="viridis",
                              interval=200, repeat=True, add_tumor_contours=False, 
                              tumor_threshold=0.1, fps=5):
        """
        Animate nutrient field evolution over time.
        
        Args:
            output_dir: Directory to save animation
            save_animation: Whether to save the animation as a file
            show_animation: Whether to display the animation
            figsize: Figure size
            z_slice: Z-slice to animate (defaults to center)
            cmap: Colormap for the nutrient field
            interval: Animation interval in milliseconds
            repeat: Whether to repeat the animation
            add_tumor_contours: Whether to add tumor boundary contours
            tumor_threshold: Density threshold for tumor boundary
            fps: Frames per second for saved animation
        """
        print("Creating nutrient field animation...")
        
        # Check if nutrient fields are available
        if "nutrient_fields" not in self.field_data:
            print("Warning: No nutrient fields found in simulation data.")
            print("Available field data keys:", list(self.field_data.keys()))
            return None
        
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
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        
        # Initialize plot
        nutrient_slice = all_nutrient_data[0]
        im = ax.imshow(nutrient_slice, extent=[x[0], x[-1], y[0], y[-1]], 
                      origin='lower', cmap=cmap, aspect='equal', vmin=vmin, vmax=vmax)
        
        # Initialize contour if requested
        if add_tumor_contours:
            phi_hat = self.field_data["phi_hat"][0]
            total_density = np.sum(phi_hat, axis=0)
            density_slice = total_density[:, :, z_slice]
            ax.contour(X, Y, density_slice, levels=[tumor_threshold], 
                      colors='red', linewidths=2, alpha=0.8)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, label='Nutrient Concentration')
        
        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Step {self.saved_steps[0]} (t={self.saved_times[0]:.2f})')
        
        # Animation update function
        def animate(frame):
            step_idx = frame
            nutrient_slice = all_nutrient_data[step_idx]
            
            # Update image data
            im.set_array(nutrient_slice)
            
            # Update contour if present
            if add_tumor_contours:
                # Remove all LineCollection objects (contours) but keep the image
                collections_to_remove = [c for c in ax.collections if isinstance(c, mcollections.LineCollection)]
                for c in collections_to_remove:
                    ax.collections.remove(c)
                # Get tumor density for this step
                phi_hat = self.field_data["phi_hat"][step_idx]
                total_density = np.sum(phi_hat, axis=0)
                density_slice = total_density[:, :, z_slice]
                # Add new contour
                ax.contour(X, Y, density_slice, levels=[tumor_threshold], 
                          colors='red', linewidths=2, alpha=0.8)
            
            # Update title
            ax.set_title(f'Step {self.saved_steps[step_idx]} (t={self.saved_times[step_idx]:.2f})')
            
            return [im]
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(self.saved_steps),
                                       interval=interval, repeat=repeat, blit=False)
        
        plt.tight_layout()
        
        # Save animation if requested
        if save_animation and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f'nutrient_field_animation_z{z_slice}.gif'
            filepath = os.path.join(output_dir, filename)
            print(f"Saving animation to {filepath}...")
            anim.save(filepath, writer='pillow', fps=fps)
            print(f"Animation saved to {filepath}")
        
        if show_animation:
            plt.show()
        else:
            plt.close()
        
        return anim
    
    def animate_source_field(self, output_dir=None, save_animation=False, show_animation=True,
                            figsize=(10, 10), z_slice=None, cmap="RdBu_r", population_idx=0,
                            interval=200, repeat=True, add_tumor_contours=False,
                            tumor_threshold=0.1, fps=5):
        """
        Animate source term field evolution over time.
        
        Args:
            output_dir: Directory to save animation
            save_animation: Whether to save the animation as a file
            show_animation: Whether to display the animation
            figsize: Figure size
            z_slice: Z-slice to animate (defaults to center)
            cmap: Colormap for the source field (RdBu_r is good for positive/negative values)
            population_idx: Index of population to animate (default 0)
            interval: Animation interval in milliseconds
            repeat: Whether to repeat the animation
            add_tumor_contours: Whether to add tumor boundary contours
            tumor_threshold: Density threshold for tumor boundary
            fps: Frames per second for saved animation
        """
        print("Creating source field animation...")
        
        # Check if source fields are available in physics data
        if "physics_data" not in self.simulation_data or self.simulation_data["physics_data"] is None:
            print("Warning: No physics data found in simulation data.")
            print("Available data keys:", list(self.simulation_data.keys()))
            return None
        
        # Check if source terms are available in physics data
        physics_data = self.simulation_data["physics_data"]
        if physics_data is None or len(physics_data) == 0 or "source_terms" not in physics_data[0]:
            print("Warning: No source terms found in physics data.")
            print("Available physics data keys:", list(physics_data[0].keys()) if len(physics_data) > 0 else "No physics data")
            return None
        
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
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        
        # Initialize plot
        source_slice = all_source_data[0]
        im = ax.imshow(source_slice, extent=[x[0], x[-1], y[0], y[-1]], 
                      origin='lower', cmap=cmap, aspect='equal', vmin=vmin, vmax=vmax)
        
        # Initialize contour if requested
        if add_tumor_contours:
            phi_hat = self.field_data["phi_hat"][0]
            total_density = np.sum(phi_hat, axis=0)
            density_slice = total_density[:, :, z_slice]
            ax.contour(X, Y, density_slice, levels=[tumor_threshold], 
                      colors='black', linewidths=2, alpha=0.8)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, label='Source Term')
        
        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        pop_label = self.labels[population_idx] if population_idx < len(self.labels) else f'Population {population_idx}'
        ax.set_title(f'{pop_label} - Step {self.saved_steps[0]} (t={self.saved_times[0]:.2f})')
        
        # Animation update function
        def animate(frame):
            step_idx = frame
            source_slice = all_source_data[step_idx]
            
            # Update image data
            im.set_array(source_slice)
            
            # Update contour if present
            if add_tumor_contours:
                # Remove all LineCollection objects (contours) but keep the image
                collections_to_remove = [c for c in ax.collections if isinstance(c, mcollections.LineCollection)]
                for c in collections_to_remove:
                    ax.collections.remove(c)
                # Get tumor density for this step
                phi_hat = self.field_data["phi_hat"][step_idx]
                total_density = np.sum(phi_hat, axis=0)
                density_slice = total_density[:, :, z_slice]
                # Add new contour
                ax.contour(X, Y, density_slice, levels=[tumor_threshold], 
                          colors='black', linewidths=2, alpha=0.8)
            
            # Update title
            ax.set_title(f'{pop_label} - Step {self.saved_steps[step_idx]} (t={self.saved_times[step_idx]:.2f})')
            
            return [im]
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(self.saved_steps),
                                       interval=interval, repeat=repeat, blit=False)
        
        plt.tight_layout()
        
        # Save animation if requested
        if save_animation and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f'source_field_animation_z{z_slice}_pop{population_idx}.gif'
            filepath = os.path.join(output_dir, filename)
            print(f"Saving animation to {filepath}...")
            anim.save(filepath, writer='pillow', fps=fps)
            print(f"Animation saved to {filepath}")
        
        if show_animation:
            plt.show()
        else:
            plt.close()
        
        return anim
    
    def animate_multiple_fields(self, fields_to_animate=['density', 'nutrient'], 
                               output_dir=None, save_animation=False, show_animation=True,
                               figsize=(15, 7), z_slice=None, interval=200, repeat=True,
                               density_threshold=0.1, tumor_threshold=0.1, fps=5):
        """
        Animate multiple fields side-by-side.
        
        Args:
            fields_to_animate: List of fields to animate. Options: 'density', 'nutrient', 'source'
            output_dir: Directory to save animation
            save_animation: Whether to save the animation as a file
            show_animation: Whether to display the animation
            figsize: Figure size
            z_slice: Z-slice to animate (defaults to center)
            interval: Animation interval in milliseconds
            repeat: Whether to repeat the animation
            density_threshold: Density threshold for contour in density plot
            tumor_threshold: Density threshold for tumor boundary in other plots
            fps: Frames per second for saved animation
        """
        print(f"Creating multi-field animation for: {fields_to_animate}...")
        
        # Set default z_slice if not provided
        if z_slice is None:
            z_slice = self.grid_size[2] // 2
        
        # Create coordinate grids
        x = np.arange(self.grid_size[0]) * self.dx
        y = np.arange(self.grid_size[1]) * self.dx
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        num_fields = len(fields_to_animate)
        fig, axes = plt.subplots(1, num_fields, figsize=figsize)
        if num_fields == 1:
            axes = [axes]
        
        # Prepare data for each field
        field_data = {}
        field_ims = {}  # Map field_name to image object
        field_contours = {}  # Map field_name to contour object
        field_axes = {}  # Map field_name to axis object
        field_idx = 0  # Track actual axis index
        
        for field_name in fields_to_animate:
            ax = axes[field_idx]
            
            if field_name == 'density':
                # Prepare density data
                all_density_data = []
                for step_idx in range(len(self.saved_steps)):
                    phi_hat = self.field_data["phi_hat"][step_idx]
                    total_density = np.sum(phi_hat, axis=0)
                    density_slice = total_density[:, :, z_slice]
                    all_density_data.append(density_slice)
                
                vmin = np.min(all_density_data)
                vmax = np.max(all_density_data)
                
                im = ax.imshow(all_density_data[0], extent=[x[0], x[-1], y[0], y[-1]], 
                              origin='lower', cmap='viridis', aspect='equal', vmin=vmin, vmax=vmax)
                field_ims[field_name] = im
                field_axes[field_name] = ax
                
                # Add contour
                contour = ax.contour(X, Y, all_density_data[0], levels=[density_threshold], 
                                   colors='red', linewidths=2, alpha=0.8)
                field_contours[field_name] = contour
                
                ax.set_title('Tumor Density')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                fig.colorbar(im, ax=ax, label='Density')
                
                field_data[field_name] = all_density_data
                field_idx += 1
                
            elif field_name == 'nutrient':
                # Check if nutrient fields are available
                if "nutrient_fields" not in self.field_data:
                    print(f"Warning: No nutrient fields found. Skipping {field_name}.")
                    # Hide this axis
                    axes[field_idx].set_visible(False)
                    field_idx += 1
                    continue
                
                # Prepare nutrient data
                all_nutrient_data = []
                for step_idx in range(len(self.saved_steps)):
                    nutrient_field = self.field_data["nutrient_fields"][step_idx]
                    nutrient_slice = nutrient_field[:, :, z_slice]
                    all_nutrient_data.append(nutrient_slice)
                
                vmin = np.min(all_nutrient_data)
                vmax = np.max(all_nutrient_data)
                
                im = ax.imshow(all_nutrient_data[0], extent=[x[0], x[-1], y[0], y[-1]], 
                              origin='lower', cmap='viridis', aspect='equal', vmin=vmin, vmax=vmax)
                field_ims[field_name] = im
                field_axes[field_name] = ax
                
                # Add tumor contour if requested
                phi_hat = self.field_data["phi_hat"][0]
                total_density = np.sum(phi_hat, axis=0)
                density_slice = total_density[:, :, z_slice]
                contour = ax.contour(X, Y, density_slice, levels=[tumor_threshold], 
                                    colors='red', linewidths=2, alpha=0.8)
                field_contours[field_name] = contour
                
                ax.set_title('Nutrient Field')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                fig.colorbar(im, ax=ax, label='Nutrient Concentration')
                
                field_data[field_name] = all_nutrient_data
                field_idx += 1
                
            elif field_name == 'source':
                # Check if source fields are available
                if "physics_data" not in self.simulation_data or self.simulation_data["physics_data"] is None:
                    print(f"Warning: No physics data found. Skipping {field_name}.")
                    # Hide this axis
                    axes[field_idx].set_visible(False)
                    field_idx += 1
                    continue
                
                physics_data = self.simulation_data["physics_data"]
                if physics_data is None or len(physics_data) == 0 or "source_terms" not in physics_data[0]:
                    print(f"Warning: No source terms found. Skipping {field_name}.")
                    # Hide this axis
                    axes[field_idx].set_visible(False)
                    field_idx += 1
                    continue
                
                # Prepare source data (use first population)
                all_source_data = []
                for step_idx in range(len(self.saved_steps)):
                    source_field = physics_data[step_idx]["source_terms"][0]
                    source_slice = source_field[:, :, z_slice]
                    all_source_data.append(source_slice)
                
                vmin = np.min(all_source_data)
                vmax = np.max(all_source_data)
                
                # Ensure symmetric color scaling
                if vmin < 0 and vmax > 0:
                    abs_max = max(abs(vmin), abs(vmax))
                    vmin, vmax = -abs_max, abs_max
                
                im = ax.imshow(all_source_data[0], extent=[x[0], x[-1], y[0], y[-1]], 
                              origin='lower', cmap='RdBu_r', aspect='equal', vmin=vmin, vmax=vmax)
                field_ims[field_name] = im
                field_axes[field_name] = ax
                
                # Add tumor contour if requested
                phi_hat = self.field_data["phi_hat"][0]
                total_density = np.sum(phi_hat, axis=0)
                density_slice = total_density[:, :, z_slice]
                contour = ax.contour(X, Y, density_slice, levels=[tumor_threshold], 
                                    colors='black', linewidths=2, alpha=0.8)
                field_contours[field_name] = contour
                
                pop_label = self.labels[0] if len(self.labels) > 0 else 'Population 0'
                ax.set_title(f'Source Field ({pop_label})')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                fig.colorbar(im, ax=ax, label='Source Term')
                
                field_data[field_name] = all_source_data
                field_idx += 1
        
        # Set overall title
        fig.suptitle(f'Step {self.saved_steps[0]} (t={self.saved_times[0]:.2f})', fontsize=14)
        
        # Animation update function
        def animate(frame):
            step_idx = frame
            
            for field_name in field_data.keys():
                data = field_data[field_name][step_idx]
                field_ims[field_name].set_array(data)
                
                ax = field_axes[field_name]
                contour = field_contours[field_name]
                
                # Update contour for density field
                if field_name == 'density':
                    # Remove all LineCollection objects (contours) but keep the image
                    collections_to_remove = [c for c in ax.collections if isinstance(c, mcollections.LineCollection)]
                    for c in collections_to_remove:
                        ax.collections.remove(c)
                    # Add new contour
                    new_contour = ax.contour(X, Y, data, levels=[density_threshold], 
                                            colors='red', linewidths=2, alpha=0.8)
                    field_contours[field_name] = new_contour
                # Update contour for nutrient and source fields
                elif field_name in ['nutrient', 'source']:
                    # Remove all LineCollection objects (contours) but keep the image
                    collections_to_remove = [c for c in ax.collections if isinstance(c, mcollections.LineCollection)]
                    for c in collections_to_remove:
                        ax.collections.remove(c)
                    # Get tumor density for this step
                    phi_hat = self.field_data["phi_hat"][step_idx]
                    total_density = np.sum(phi_hat, axis=0)
                    density_slice = total_density[:, :, z_slice]
                    # Add new contour
                    new_contour = ax.contour(X, Y, density_slice, levels=[tumor_threshold], 
                                           colors='red' if field_name == 'nutrient' else 'black', 
                                           linewidths=2, alpha=0.8)
                    field_contours[field_name] = new_contour
            
            # Update overall title
            fig.suptitle(f'Step {self.saved_steps[step_idx]} (t={self.saved_times[step_idx]:.2f})', 
                        fontsize=14)
            
            return list(field_ims.values())
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(self.saved_steps),
                                       interval=interval, repeat=repeat, blit=False)
        
        plt.tight_layout()
        
        # Save animation if requested
        if save_animation and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            fields_str = '_'.join(fields_to_animate)
            filename = f'multi_field_animation_{fields_str}_z{z_slice}.gif'
            filepath = os.path.join(output_dir, filename)
            print(f"Saving animation to {filepath}...")
            anim.save(filepath, writer='pillow', fps=fps)
            print(f"Animation saved to {filepath}")
        
        if show_animation:
            plt.show()
        else:
            plt.close()
        
        return anim

