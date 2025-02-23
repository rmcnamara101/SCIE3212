
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class TumorPlotter:

    def __init__(self, model):
        self.model = model
        self.simulation_history = model.get_history()


    def plot_tumor(self, step: int):
        """
        Plot the tumor as a 3D surface.
        """
        print(self.simulation_history['stem cell concentration'][step].shape)
        # Get the indices where cells exist
        x_indices, y_indices, z_indices = np.where(self.simulation_history['stem cell concentration'][step] > 0)
        
        # Create a meshgrid for plotting
        X, Y = np.meshgrid(x_indices, y_indices)
        Z = z_indices.reshape(X.shape)  # Reshape z_indices to match the shape of X and Y
        
        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the tumor as a 3D surface
        ax.plot_surface(X, Y, Z, cmap='viridis')
        
        # Set axis limits to focus on the region containing the tumor
        ax.set_xlim([0, self.model.grid_shape[0]])
        ax.set_ylim([0, self.model.grid_shape[1]])
        ax.set_zlim([0, self.model.grid_shape[2]])
        
        # Label the axes and add a title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Tumor 3D Surface')
        
        plt.show()


    def plot_tumor_slices(self):
        pass


    def plot_all_isosurfaces(self, threshold=0.01):
        """
        Plot isosurfaces for all cell concentration fields in one 3D plot.
        Automatically focuses on the region containing cells.
        """
        # Define fields and colors for plotting
        fields = {
            'Stem': (self.simulation_history['stem cell concentration'][-1], 'red'),
            'Progenitor': (self.simulation_history['progenitor cell concentration'][-1], 'blue'),
            'Differentiated': (self.simulation_history['differentiated cell concentration'][-1], 'green'),
            'Necrotic': (self.simulation_history['necrotic cell concentration'][-1], 'black')
        }
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Find the bounds of the tumor (where any cell type exists above threshold)
        total_mask = np.zeros_like(self.simulation_history['stem cell concentration'][-1], dtype=bool)
        for name, (field, _) in fields.items():
            total_mask |= (field > threshold)
        
        if not total_mask.any():
            print("No cells above threshold found.")
            return
        
        # Get the indices where cells exist
        x_indices, y_indices, z_indices = np.where(total_mask)
        
        # Calculate bounds based on the actual indices of the tumor
        padding = 5  # Adjust padding to control space around tumor
        x_min, x_max = max(0, np.min(x_indices) - padding), min(self.model.grid_shape[0], np.max(x_indices) + padding)
        y_min, y_max = max(0, np.min(y_indices) - padding), min(self.model.grid_shape[1], np.max(y_indices) + padding)
        z_min, z_max = max(0, np.min(z_indices) - padding), min(self.model.grid_shape[2], np.max(z_indices) + padding)
        
        # Convert to spatial coordinates
        x_min, x_max = x_min * self.model.dx, x_max * self.model.dx
        y_min, y_max = y_min * self.model.dx, y_max * self.model.dx
        z_min, z_max = z_min * self.model.dx, z_max * self.model.dx
        
        # Loop over each concentration field and extract its isosurface
        for name, (field, color) in fields.items():
            # Apply Gaussian filter for smoothing
            smoothed_field = gaussian_filter(field, sigma=1)
            
            if np.max(smoothed_field) < threshold:
                continue
            
            try:
                verts, faces, normals, _ = marching_cubes(smoothed_field, level=threshold, spacing=(self.model.dx, self.model.dx, self.model.dx))
                mesh = Poly3DCollection(verts[faces], alpha=0.2)
                mesh.set_facecolor(color)
                mesh.set_edgecolor(color)
                ax.add_collection3d(mesh)
                
                # Annotate with cell type name at the center of its isosurface
                center = np.mean(verts, axis=0)
                ax.text(center[0], center[1], center[2], name, color=color, fontsize=12)
            except Exception as e:
                print(f"Could not extract isosurface for {name}: {e}")
        
        # Set axis limits to focus on the region containing the tumor
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])
        
        # Draw a bounding box for reference
        self._draw_bounding_box(ax, (x_min, x_max), (y_min, y_max), (z_min, z_max))
        
        # Label the axes and add a title
        ax.set_xlabel(f'X')
        ax.set_ylabel(f'Y')
        ax.set_zlabel(f'Z')
        ax.set_title(f'Tumor Isosurfaces at Step {self.simulation_history["step"][-1]}')
        
        plt.show()


    def plot_volume_evolution(self):
        """
        Plot the volume evolution of the tumor.
        """
        plt.figure(figsize=(10, 6))
        
        # plot tumor volume evolution
        plt.plot(
            self.simulation_history['step'], 
            self.simulation_history['stem cell volume'],
            'r-',
            label='Stem'
            )
        
        plt.plot(
            self.simulation_history['step'], 
            self.simulation_history['progenitor cell volume'], 
            'b-', 
            label='Progenitor'
            )
        
        plt.plot(
            self.simulation_history['step'],
            self.simulation_history['differentiated cell volume'], 
            'g-', 
            label='Differentiated'
            )
        
        plt.plot(
            self.simulation_history['step'], 
            self.simulation_history['necrotic cell volume'], 
            'k-', 
            label='Necrotic'
            )
        
        plt.plot(
            self.simulation_history['step'], 
            self.simulation_history['total cell volume'], 
            'gray', 
            linestyle='--', 
            label='Total'
            )
        
        plt.xlabel('Time Step')
        plt.ylabel('Volume')
        plt.title('Tumor Volume Evolution')
        plt.legend()
        plt.grid(True)
        plt.show()


    def plot_radius_evolution(self):
        """
        Plot the radius evolution of the tumor.
        """

        plt.figure(figsize=(10, 6))

        plt.plot(self.simulation_history['step'], 
                 self.simulation_history['radius'], 
                 'purple', 
                 label='Radius'
                 )
        
        plt.xlabel('Time Step')
        plt.ylabel('Radius')
        plt.title('Tumor Radius Evolution')
        plt.legend()
        plt.grid(True)
        plt.show()
    

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