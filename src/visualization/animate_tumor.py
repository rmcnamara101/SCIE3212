
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation



class TumorAnimator:

    def __init__(self, model):
        self.model = model
        self.simulation_history = model.get_history()



    def animate_tumor_slices(self, steps=100, slice_pos=None, interval=100):
        """
        Create an animation of contour plots showing tumor growth through x-axis slices.
        
        Parameters:
        -----------
        steps : int
            Number of frames in the animation
        slice_pos : int, optional
            Position of x-slice. If None, uses middle of grid
        interval : int
            Delay between frames in milliseconds
        """
        
        # Set default slice position to middle of grid if not specified
        if slice_pos is None:
            slice_pos = self.model.grid_shape[0] // 2
        
        # Create figure and subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle(f'Tumor Growth Animation (X-Slice at {slice_pos})', fontsize=16)
        
        # Flatten axes for easier iteration
        axes = axes.flatten()
        
        # Dictionary of cell types and their properties
        cell_types = {
            'stem cell concentration': (self.model.C_S, 'Reds'),
            'progenitor cell concentration': (self.model.C_P, 'Blues'),
            'differentiated cell concentration': (self.model.C_D, 'Greens'),
            'necrotic cell concentration': (self.model.C_N, 'Greys')
        }
        
        # Create mesh grid for plotting
        y, z = np.meshgrid(np.arange(self.model.grid_shape[1]), np.arange(self.model.grid_shape[2]))
        
        # Initialize contour plots and store them
        contour_artists = []
        for ax, (title, (field, cmap)) in zip(axes, cell_types.items()):
            # Create initial contour plot
            ax.set_title(title)
            
            # Create empty contour plot
            cont = ax.contourf(y, z, np.zeros((self.model.grid_shape[1], self.model.grid_shape[2])),
                            levels=np.linspace(0, 1, 20),
                            cmap=cmap)
            fig.colorbar(cont, ax=ax)
            contour_artists.append(cont)
        
        # Make figure layout tight
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        def update(frame):
            """Update function for animation"""
            # Get current slice data from history
            slice_data = self.simulation_history['stem cell concentration'][frame][slice_pos, :, :].T
            
            # Clear and update each subplot
            for ax, cont, (title, (field, cmap)) in zip(axes, contour_artists, cell_types.items()):
                # Clear the axis
                ax.clear()
                
                # Create new contour plot
                new_cont = ax.contourf(y, z, slice_data,
                                    levels=np.linspace(0, max(0.0001, slice_data.max()), 20),
                                    cmap=cmap)
                
                # Update title and formatting
                ax.set_title(f'{title}\nMax: {slice_data.max():.4f}')
                ax.set_xlabel('Y')
                ax.set_ylabel('Z')
            
            fig.suptitle(f'Tumor Growth Animation (X-Slice at {slice_pos}) - Frame {frame}')
            
            return contour_artists
        
        # Create animation
        anim = animation.FuncAnimation(
            fig,
            update,
            frames=steps,
            interval=interval,
            blit=False,
            repeat=False
        )
        #anim.save('tumor_growth_animation2.gif', writer='pillow', fps=5)
        plt.show()
        
        return anim


    def animate_single_slice(self, steps=100, slice_pos=None, interval=100):
        """
        Create an animation of a single contour plot showing all cell types overlaid.
        
        Parameters:
        -----------
        steps : int
            Number of frames in the animation
        slice_pos : int, optional
            Position of x-slice. If None, uses middle of grid
        interval : int
            Delay between frames in milliseconds
        """
        
        # Set default slice position to middle of grid if not specified
        if slice_pos is None:
            slice_pos = self.model.grid_shape[0] // 2
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))
        fig.suptitle(f'Combined Tumor Growth Animation (X-Slice at {slice_pos})', fontsize=16)
        
        # Dictionary of cell types and their properties
        cell_types = {
            'Stem Cells': (self.model.C_S, 'red'),
            'Progenitor Cells': (self.model.C_P, 'blue'),
            'Differentiated Cells': (self.model.C_D, 'green'),
            'Necrotic Cells': (self.model.C_N, 'black')
        }
        
        # Create mesh grid for plotting
        y, z = np.meshgrid(np.arange(self.model.grid_shape[1]), np.arange(self.model.grid_shape[2]))
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], color=color, label=title)
                        for title, (_, color) in cell_types.items()]
        ax.legend(handles=legend_elements, loc='upper right')
        
        def update(frame):
            """Update function for animation"""
            # Clear the axis
            ax.clear()
            
            # Redraw all contours
            for title, (field, color) in cell_types.items():
                slice_data = self.simulation_history[title][frame][slice_pos, :, :].T
                if slice_data.max() > 0:  # Only draw contours if cells exist
                    ax.contour(y, z, slice_data,
                            levels=[max(0.1, slice_data.max() * 0.2)],  # Adaptive threshold
                            colors=[color],
                            alpha=0.7,
                            linewidths=2)
            
            # Restore legend
            ax.legend(handles=legend_elements, loc='upper right')
            
            # Update labels and title
            ax.set_xlabel('Y')
            ax.set_ylabel('Z')
            ax.set_title(f'Frame {frame}')
            fig.suptitle(f'Combined Tumor Growth Animation (X-Slice at {slice_pos})')
            
            return ax.get_children()
        
        # Create animation
        anim = animation.FuncAnimation(
            fig,
            update,
            frames=steps,
            interval=interval,
            blit=False,
            repeat=False
        )
        #anim.save('tumor_growth_animation.gif', writer='pillow', fps=5)
        plt.show()
        
        return anim


    def animate_tumor_growth_isosurfaces(self, steps=100, threshold=0.0001, interval=50):
        """
        Create an animation of isosurfaces showing tumor growth over time.
        
        Parameters:
        -----------
        steps : int
            Number of frames in the animation
        threshold : float
            Isosurface threshold for visualization
        interval : int
            Delay between frames in milliseconds
        """
        
        # Create figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Define fields and colors for plotting
        fields = {
            'stem cell volume fraction': (self.model.C_S, 'red'),
            'progenitor cell volume fraction': (self.model.C_P, 'blue'),
            'differentiated cell volume fraction': (self.model.C_D, 'green'),
            'necrotic cell volume fraction': (self.model.C_N, 'black')
        }
        
        def update(frame):
            """Update function for animation""" 
            # Clear previous frame
            ax.cla()

            C_T = self.model.C_S + self.model.C_P + self.model.C_D + self.model.C_N
    
            # Find the bounds of the tumor
            total_mask = np.zeros_like(C_T, dtype=bool)
            for name, (field, _) in fields.items():
                if name in self.simulation_history:  # Check if key exists
                    total_mask |= (self.simulation_history[name][frame] > threshold)
                else:
                    print(f"Warning: Key '{name}' not found in simulation history.")
            
            if not total_mask.any():
                return
            
            # Get the indices where cells exist
            x_indices, y_indices, z_indices = np.where(total_mask)
            
            x_min, x_max = 0, self.model.grid_shape[0] * self.model.dx
            y_min, y_max = 0, self.model.grid_shape[1] * self.model.dx
            z_min, z_max = 0, self.model.grid_shape[2] * self.model.dx

            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
            ax.set_zlim([z_min, z_max])

            
            # Plot isosurfaces for each cell type
            for name, (field, color) in fields.items():
                # Apply Gaussian filter for smoothing
                smoothed_field = gaussian_filter(self.simulation_history[name][frame], sigma=3)
                
                if np.max(smoothed_field) < threshold:
                    continue
                
                try:
                    verts, faces, normals, _ = marching_cubes(smoothed_field, level=threshold, 
                                                            spacing=(self.model.dx, self.model.dx, self.model.dx))
                    mesh = Poly3DCollection(verts[faces], alpha=0.2)
                    mesh.set_facecolor(color)
                    mesh.set_edgecolor(color)
                    ax.add_collection3d(mesh)
                    
                    
                except Exception as e:
                    print(f"Could not extract isosurface for {name} at frame {frame}: {e}")
            
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
            ax.set_zlim([z_min, z_max])
            
            # Set labels and title
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Tumor Growth')
            
            # Set optimal viewing angle
            ax.view_init(elev=20)  # Rotate view during animation
            
            return ax,

        # Create animation
        anim = animation.FuncAnimation(
            fig,
            update,
            frames=steps,
            interval=interval,
            blit=False,
            repeat=True
        )
        
        # Save animation
        #anim.save('tumor_growth_3d.gif', writer='pillow', fps=5)
        
        plt.show()
        return anim


    def _draw_bounding_box(self, ax, xlim, ylim, zlim):
        """
        Draw a simple wireframe bounding box to help visualize the overall spatial domain.
        """
        # Extract the corners of the box
        corners = np.array([[xlim[0], ylim[0], zlim[0]],
                            [xlim[1], ylim[0], zlim[0]],
                            [xlim[1], ylim[1], zlim[0]],
                            [xlim[0], ylim[1], zlim[0]],
                            [xlim[0], ylim[0], zlim[1]],
                            [xlim[1], ylim[0], zlim[1]],
                            [xlim[1], ylim[1], zlim[1]],
                            [xlim[0], ylim[1], zlim[1]]])
        # Define the 12 edges of a box (pairs of corner indices)
        edges = [(0,1), (1,2), (2,3), (3,0),
                (4,5), (5,6), (6,7), (7,4),
                (0,4), (1,5), (2,6), (3,7)]
        for edge in edges:
            pts = corners[list(edge), :]
            ax.plot(pts[:,0], pts[:,1], pts[:,2], color='gray', linestyle='--', linewidth=1)