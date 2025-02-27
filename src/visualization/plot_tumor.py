import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class VolumeFractionPlotter:
    def __init__(self, model):
        self.model = model
        self.simulation_history = model.get_history()

    def plot_volume_fractions(self, step: int):
        """
        Plot a 3D scatter plot of volume fractions for all cell types at a given step.
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Define cell types and colors
        cell_types = {
            'Healthy': ('Healthy cell volume fraction', 'red'),
            'Progenitor': ('progenitor cell volume fraction', 'blue'),
            'Differentiated': ('differentiated cell volume fraction', 'green'),
            'Necrotic': ('necrotic cell volume fraction', 'black')
        }
        
        for name, (key, color) in cell_types.items():
            field = self.simulation_history[key][step]
            x, y, z = np.where(field > 0)
            fractions = field[field > 0]
            ax.scatter(x * self.model.dx, y * self.model.dx, z * self.model.dx, 
                       c=color, s=50 * fractions, alpha=0.5, label=name)
        
        ax.set_xlim([0, self.model.grid_shape[0] * self.model.dx])
        ax.set_ylim([0, self.model.grid_shape[1] * self.model.dx])
        ax.set_zlim([0, self.model.grid_shape[2] * self.model.dx])
        ax.set_xlabel('X (spatial units)')
        ax.set_ylabel('Y (spatial units)')
        ax.set_zlabel('Z (spatial units)')
        ax.set_title(f'Volume Fractions at Step {step}')
        ax.legend()
        plt.show()

    def plot_volume_fraction_evolution(self):
        """
        Plot the evolution of the total volume fraction for a specific cell type over time.
        """
        plt.figure(figsize=(10, 6))

        for cell_type in ['Healthy', 'Progenitor', 'Differentiated', 'Necrotic']:

            key = f'{cell_type.lower()} cell volume fraction'
            total_fractions = np.sum(self.simulation_history[key], axis=(1, 2, 3)) * (self.model.dx ** 3)
            plt.plot(self.simulation_history['step'], total_fractions, label=cell_type)

        plt.xlabel('Time Step')
        plt.ylabel(f'Total Cell Volume')
        plt.title(f'Cell Volume Evolution')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_total_volume_evolution(self):
        """
        Plot the total volume occupied by all cells over time.
        """
        total_volume = np.sum(self.simulation_history['total cell volume fraction'], axis=(1, 2, 3)) * (self.model.dx ** 3)
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.simulation_history['step'], total_volume, 'gray', linestyle='--', label='Total Volume')
        plt.xlabel('Time Step')
        plt.ylabel('Total Volume')
        plt.title('Total Tumor Volume Evolution')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_radius_evolution(self):
        """
        Plot the radius evolution of the tumor.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.simulation_history['step'], self.simulation_history['radius'], 'purple', label='Radius')
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
        corners = np.array([[xlim[0], ylim[0], zlim[0]],
                            [xlim[1], ylim[0], zlim[0]],
                            [xlim[1], ylim[1], zlim[0]],
                            [xlim[0], ylim[1], zlim[0]],
                            [xlim[0], ylim[0], zlim[1]],
                            [xlim[1], ylim[0], zlim[1]],
                            [xlim[1], ylim[1], zlim[1]],
                            [xlim[0], ylim[1], zlim[1]]])
        edges = [(0,1), (1,2), (2,3), (3,0),
                 (4,5), (5,6), (6,7), (7,4),
                 (0,4), (1,5), (2,6), (3,7)]
        for edge in edges:
            pts = corners[list(edge), :]
            ax.plot(pts[:,0], pts[:,1], pts[:,2], color='gray', linestyle='--', linewidth=1)