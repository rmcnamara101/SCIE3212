import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import marching_cubes
from scipy.signal import savgol_filter
from matplotlib.animation import FuncAnimation
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

class scie3121SimulationAnalyzer:
    """Analyzer for tumor growth simulation data from NPZ files."""
    
    def __init__(self, filepath):
        """
        Initialize with the path to an NPZ file containing simulation history.
        """
        self.filepath = filepath
        self.history = load_simulation_history(filepath)
        print(self.history.keys())
        self.volume_data = compute_total_volumes(self.history)
        self.radius_data = compute_total_radius(self.history, threshold=0.05)
        self.metadata = self.history['Simulation Metadata']

    def get_simulation_metadata(self):
        """Return metadata about the simulation."""
        return self.metadata

    def plot_volumes(self, smooth_window=5, normalize=None):
        """Plot total volumes of cell types over time with optional smoothing and normalization.
        
        Args:
            smooth_window (int): Window size for Savitzky-Golay smoothing (odd number).
            normalize (str, optional): 'initial' to normalize by initial total volume,
                                       'max' to normalize by maximum total volume.
        """
        steps, healthy, diseased, necrotic, total = self.volume_data
        
        # Ensure smooth_window is odd
        #if smooth_window % 2 == 0:
        #    smooth_window += 1
        
        # Apply smoothing if window > 1
        #if smooth_window > 1:
        #    healthy = savgol_filter(healthy, smooth_window, 3)
        #    progenitor = savgol_filter(progenitor, smooth_window, 3)
        #    differentiated = savgol_filter(differentiated, smooth_window, 3)
        #    necrotic = savgol_filter(necrotic, smooth_window, 3)
        #    total = savgol_filter(total, smooth_window, 3)
        
        # Normalize volumes
        #if normalize == 'initial':
        #    initial_total = total[0]
        #    if initial_total != 0:
        #        healthy /= initial_total
        #        progenitor /= initial_total
        #        differentiated /= initial_total
        #        necrotic /= initial_total
        #        total /= initial_total
        #    else:
        #        print("Warning: Initial total volume is zero, skipping normalization.")
        #elif normalize == 'max':
        #    max_total = np.max(total)
        #    if max_total != 0:
        #        healthy /= max_total
        #        progenitor /= max_total
        #        differentiated /= max_total
        #        necrotic /= max_total
        #        total /= max_total
        #    else:
        #        print("Warning: Maximum total volume is zero, skipping normalization.")
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, diseased, label='Differentiated')
        plt.plot(steps, necrotic, label='Necrotic')
        plt.plot(steps, healthy, label='Healthy')
        plt.plot(steps, total, label='Total', linestyle='--', color='black', alpha=0.7)
        plt.xlabel(f'Time step % 10 ')
        plt.ylabel('Volume (summed cell fraction)' if not normalize else 'Normalized Volume')
        plt.title('Tumor Cell Volume Evolution')
        plt.legend()
        plt.show()

    def plot_radius(self, smooth_window=5):
        """
        Plot tumor radius over time with optional smoothing.
        
        Args:
            smooth_window (int): Window size for Savitzky-Golay smoothing (odd number).
        """
        radii = self.radius_data
        steps = self.history['step']
        
        if smooth_window % 2 == 0:
            smooth_window += 1
        
        if smooth_window > 1:
            radii = savgol_filter(radii, smooth_window, 3)
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, radii, label='Tumor Radius', color='magenta')
        plt.xlabel('Time Step')
        plt.ylabel('Radius (in grid units)')
        plt.title('Tumor Radius Evolution')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

    def plot_3d_at_step(self, step_index, levels=None, colors=None, mode='isosurface'):
        """
        Plot 3D visualization of tumor fields at a specific time step with multiple modes.
        
        Args:
            step_index (int): Index of the time step to visualize.
            levels (dict, optional): Isosurface levels for each cell type. Defaults to 0.1.
            colors (dict, optional): Colors for each cell type. Defaults to preset values.
            mode (str): Visualization mode ('isosurface', 'voxel', 'scatter'). Defaults to 'isosurface'.
        """
        if levels is None:
            levels = {'healthy': 0.1, 'diseased': 0.1, 'necrotic': 0.1}
        if colors is None:
            colors = {'healthy': 'green', 'diseased': 'red', 'necrotic': 'black'}
        
        fields = {
            'healthy': self.history['healthy cell volume fraction'][step_index],
            'diseased': self.history['diseased cell volume fraction'][step_index],
            'necrotic': self.history['necrotic cell volume fraction'][step_index]
        }
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        if mode == 'isosurface':
            for cell_type, field in fields.items():
                level = levels[cell_type]
                if np.max(field) > level:
                    verts, faces, _, _ = marching_cubes(field, level=level)
                    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], 
                                   color=colors[cell_type], alpha=0.3, label=cell_type)
        
        elif mode == 'voxel':
            for cell_type, field in fields.items():
                mask = field > levels[cell_type]
                ax.voxels(mask, facecolors=colors[cell_type], edgecolors='k', alpha=0.5, label=cell_type)
        
        elif mode == 'scatter':
            for cell_type, field in fields.items():
                x, y, z = np.where(field > levels[cell_type])
                ax.scatter(x, y, z, c=colors[cell_type], alpha=0.5, s=10, label=cell_type)
        
        else:
            raise ValueError("Mode must be 'isosurface', 'voxel', or 'scatter'")
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'3D Tumor Visualization at Step {step_index} ({mode})')
        ax.legend()
        plt.show()

    def plot_3d_multiple_steps(self, step_indices, levels=None, colors=None):
        """
        Plot 3D isosurfaces for multiple time steps in one figure.
        
        Args:
            step_indices (list): List of time step indices to visualize.
            levels (dict, optional): Isosurface levels for each cell type.
            colors (dict, optional): Colors for each cell type.
        """
        if levels is None:
            levels = {'healthy': 0.1, 'diseased': 0.1, 'necrotic': 0.1}
        if colors is None:
            colors = {'healthy': 'green',  'differentiated': 'red', 'necrotic': 'black'}
        
        fig = plt.figure(figsize=(12, 10))
        
        for i, step_index in enumerate(step_indices, 1):
            ax = fig.add_subplot(1, len(step_indices), i, projection='3d')
            fields = {
                'healthy': self.history['healthy cell volume fraction'][step_index],
                'diseased': self.history['diseased cell volume fraction'][step_index],
                'necrotic': self.history['necrotic cell volume fraction'][step_index]
            }
            
            for cell_type, field in fields.items():
                level = levels[cell_type]
                if np.max(field) > level:
                    verts, faces, _, _ = marching_cubes(field, level=level)
                    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], 
                                   color=colors[cell_type], alpha=0.3, label=cell_type)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Step {step_index}')
        
        plt.suptitle('3D Tumor Evolution Across Multiple Steps')
        plt.tight_layout()
        plt.legend()
        plt.show()


    def plot_cross_section(self, step_index, cell_type='healthy', plane='XY', index=None, smooth_sigma=2.0, 
                        vmin=None, vmax=None, levels=50, cmap='viridis'):
        """
        Plot 2D cross-sections of a specified cell type with smoothed contours and customizable scaling.
        
        Args:
            step_index (int): Index of the time step to visualize.
            cell_type (str): The cell type to plot ('healthy', 'progenitor', 'differentiated', 'necrotic').
            plane (str): Plane to slice ('XY', 'XZ', 'YZ'). Defaults to 'XY'.
            index (int, optional): Index along the perpendicular axis. Defaults to midpoint.
            smooth_sigma (float): Standard deviation for Gaussian smoothing. 0 for no smoothing.
            vmin (float, optional): Minimum value for colormap scaling. Defaults to data minimum.
            vmax (float, optional): Maximum value for colormap scaling. Defaults to data maximum.
            levels (int): Number of contour levels. Defaults to 10.
            cmap (str): Colormap name. Defaults to 'viridis'.
        """
        if cell_type not in ['healthy', 'diseased', 'necrotic', 'total']:
            raise ValueError("Cell type must be 'healthy', 'diseased', 'necrotic', or 'total'")
        
        fields = {
            'healthy': self.history['healthy cell volume fraction'][step_index],
            'diseased': self.history['diseased cell volume fraction'][step_index],
            'necrotic': self.history['necrotic cell volume fraction'][step_index],
            'total': (self.history['healthy cell volume fraction'] + 
                      self.history['necrotic cell volume fraction'] +
                      self.history['diseased cell volume fraction'])[step_index]
        }
        
        shape = fields['healthy'].shape
        if index is None:
            if plane == 'XY':
                index = shape[2] // 2
            elif plane == 'XZ':
                index = shape[1] // 2
            elif plane == 'YZ':
                index = shape[0] // 2
            else:
                raise ValueError("Plane must be 'XY', 'XZ', or 'YZ'")
        
        field = fields[cell_type]
        
        # Select the slice based on the plane
        if plane == 'XY':
            slice_ = field[:, :, index]
            x, y = np.arange(shape[0]), np.arange(shape[1])
            xlabel, ylabel = 'X', 'Y'
        elif plane == 'XZ':
            slice_ = field[:, index, :]
            x, y = np.arange(shape[0]), np.arange(shape[2])
            xlabel, ylabel = 'X', 'Z'
        elif plane == 'YZ':
            slice_ = field[index, :, :]
            x, y = np.arange(shape[1]), np.arange(shape[2])
            xlabel, ylabel = 'Y', 'Z'
        
        # Smooth the slice with Gaussian filter if sigma > 0
        if smooth_sigma > 0:
            slice_ = gaussian_filter(slice_, sigma=smooth_sigma)
        
        # Create finer grid for interpolation
        x_fine, y_fine = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
        X, Y = np.meshgrid(x, y)
        X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
        
        # Interpolate data onto finer grid
        slice_fine = griddata((X.flatten(), Y.flatten()), slice_.flatten(), 
                            (X_fine, Y_fine), method='cubic')
        
        # Determine colormap bounds dynamically if not provided
        if vmin is None:
            vmin = np.nanmin(slice_fine)  # Use min of interpolated data
        if vmax is None:
            vmax = np.nanmax(slice_fine)  # Use max of interpolated data
        
        # Ensure vmin and vmax are reasonable (avoid identical values)
        if vmin == vmax:
            vmin = max(0, vmin - 0.01)  # Small offset to avoid flat colormap
            vmax = vmin + 0.01
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(6, 5))
        contour = ax.contourf(X_fine, Y_fine, slice_fine, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(contour, ax=ax, label=f'{cell_type.capitalize()} Volume Fraction')
        
        # Set labels and titles
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{cell_type.capitalize()} Cross-Section ({plane} at index {index})')
        fig.suptitle(f'Step {step_index}', y=1.05)
        
        plt.show()

    def animate_cross_section(self, plane='XY', index=None, interval=200, save_as=None):
        """
        Animate 2D cross-sections of tumor fields over time.
        
        Args:
            plane (str): Plane to slice ('XY', 'XZ', 'YZ'). Defaults to 'XY'.
            index (int, optional): Index along the perpendicular axis. Defaults to midpoint.
            interval (int): Delay between frames in milliseconds. Defaults to 200.
            save_as (str, optional): File path to save animation (e.g., 'output.mp4').
        """
        fields = {
            'healthy': self.history['healthy cell volume fraction'],
            'diseased': self.history['diseased cell volume fraction'],
            'necrotic': self.history['necrotic cell volume fraction']
        }
        
        steps = self.history['step']
        num_steps = len(steps)
        
        shape = fields['healthy'][0].shape
        if index is None:
            if plane == 'XY':
                index = shape[2] // 2
            elif plane == 'XZ':
                index = shape[1] // 2
            elif plane == 'YZ':
                index = shape[0] // 2
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharex=True, sharey=True)
        
        images = {}
        for ax, cell_type in zip(axes, fields.keys()):
            if plane == 'XY':
                slice_ = fields[cell_type][0][:, :, index]
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
            elif plane == 'XZ':
                slice_ = fields[cell_type][0][:, index, :]
                ax.set_xlabel('X')
                ax.set_ylabel('Z')
            elif plane == 'YZ':
                slice_ = fields[cell_type][0][index, :, :]
                ax.set_xlabel('Y')
                ax.set_ylabel('Z')
            im = ax.imshow(slice_, cmap='viridis', origin='lower', vmin=0, vmax=1)
            ax.set_title(cell_type.capitalize())
            fig.colorbar(im, ax=ax)
            images[cell_type] = im
        
        def update(frame):
            for cell_type, im in images.items():
                if plane == 'XY':
                    slice_ = fields[cell_type][frame][:, :, index]
                elif plane == 'XZ':
                    slice_ = fields[cell_type][frame][:, index, :]
                elif plane == 'YZ':
                    slice_ = fields[cell_type][frame][index, :, :]
                im.set_array(slice_)
            fig.suptitle(f'Step {steps[frame]}')
            return list(images.values())
        
        anim = FuncAnimation(fig, update, frames=num_steps, interval=interval, blit=True)
        
        if save_as:
            anim.save(save_as, writer='ffmpeg')  # Requires ffmpeg installed
        else:
            plt.show()

    def plot_growth_rate(self):
        """Plot the tumor growth rate over time."""
        steps = self.history['step']
        total_volume = self.volume_data[-1]  # Total volume is the last element
        growth_rate = np.gradient(total_volume, steps)  # Numerical derivative
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, growth_rate, label='Growth Rate', color='teal')
        plt.xlabel('Time Step')
        plt.ylabel('Growth Rate (volume per step)')
        plt.title('Tumor Growth Rate')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

# Helper functions 
def load_simulation_history(npz_filename):
    """Load simulation history from an NPZ file."""
    data = np.load(npz_filename, allow_pickle=True)
    history = {key: data[key].item() if key == 'history' else data[key] for key in data}
    return history

def compute_total_volumes(history):
    """Compute total volumes for each cell type over time."""
    steps = history['step']
    healthy_volumes = [np.sum(phi) for phi in history['healthy cell volume fraction']]
    diseased_volumes = [np.sum(phi) for phi in history['diseased cell volume fraction']]  # Changed to match typo
    necrotic_volumes = [np.sum(phi) for phi in history['necrotic cell volume fraction']]
    total_volumes = [h + d + n for h, d, n in zip(healthy_volumes, diseased_volumes, necrotic_volumes)]  # Adjusted to 3 terms
    return steps, healthy_volumes, diseased_volumes, necrotic_volumes, total_volumes

def create_distance_grid_from_field(field):
    """Create a distance grid from the field center."""
    shape = field.shape
    center = np.array([s // 2 for s in shape])
    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    return np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)

def compute_total_radius(history, threshold=0.05):
    """Compute tumor radius over time based on a threshold."""
    radii = []
    first_phi = history['healthy cell volume fraction'][0]
    distance_grid = create_distance_grid_from_field(first_phi)
    
    for phi_H,  phi_D, phi_N in zip(
            history['healthy cell volume fraction'],
            history['diseased cell volume fraction'],
            history['necrotic cell volume fraction']):
        total_field = phi_H + phi_D + phi_N
        tumor_mask = total_field >= (threshold * np.max(total_field))
        radius = np.max(distance_grid[tumor_mask]) if np.any(tumor_mask) else 0
        radii.append(radius)
    return radii
