import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from analysis.scie3121_analysis import scie3121SimulationAnalyzer

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from matplotlib.animation import FuncAnimation
from scipy.signal import savgol_filter

class NutrientAnalyzer(scie3121SimulationAnalyzer):
    def __init__(self, filepath):
        super().__init__(filepath)


    def plot_nutrient_field(self, step_index, plane='XY', index=None, smooth_sigma=2.0, 
                        vmin=None, vmax=None, levels=50, cmap='viridis'):
        """
        Plot 2D cross-section of the nutrient field with smoothed contours and customizable scaling.
        
        Args:
            step_index (int): Index of the time step to visualize.
            plane (str): Plane to slice ('XY', 'XZ', 'YZ'). Defaults to 'XY'.
            index (int, optional): Index along the perpendicular axis. Defaults to midpoint.
            smooth_sigma (float): Standard deviation for Gaussian smoothing. 0 for no smoothing.
            vmin (float, optional): Minimum value for colormap scaling. Defaults to data minimum.
            vmax (float, optional): Maximum value for colormap scaling. Defaults to data maximum.
            levels (int): Number of contour levels. Defaults to 50.
            cmap (str): Colormap name. Defaults to 'viridis'.
        """
        if 'nutrient concentration' not in self.history:
            raise ValueError("Nutrient concentration data not found in the simulation history.")
        
        nutrient_field = self.history['nutrient concentration'][step_index]
        
        shape = nutrient_field.shape
        if index is None:
            if plane == 'XY':
                index = shape[2] // 2
            elif plane == 'XZ':
                index = shape[1] // 2
            elif plane == 'YZ':
                index = shape[0] // 2
            else:
                raise ValueError("Plane must be 'XY', 'XZ', or 'YZ'")
        
        # Select the slice based on the plane
        if plane == 'XY':
            slice_ = nutrient_field[:, :, index]
            x, y = np.arange(shape[0]), np.arange(shape[1])
            xlabel, ylabel = 'X', 'Y'
        elif plane == 'XZ':
            slice_ = nutrient_field[:, index, :]
            x, y = np.arange(shape[0]), np.arange(shape[2])
            xlabel, ylabel = 'X', 'Z'
        elif plane == 'YZ':
            slice_ = nutrient_field[index, :, :]
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
        fig.colorbar(contour, ax=ax, label='Nutrient Concentration')
        
        # Set labels and titles
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'Nutrient Cross-Section ({plane} at index {index})')
        fig.suptitle(f'Step {step_index}', y=1.05)
        
        plt.show()

    def plot_nutrient_evolution(self, smooth_window=5):
        """
        Plot the evolution of total nutrient concentration over time.
        
        Args:
            smooth_window (int): Window size for Savitzky-Golay smoothing (odd number).
        """
        if 'nutrient concentration' not in self.history:
            raise ValueError("Nutrient concentration data not found in the simulation history.")
        
        steps = self.history['step']
        nutrient_total = [np.sum(n) for n in self.history['nutrient concentration']]
        
        # Apply smoothing if window > 1
        if smooth_window > 1:
            if smooth_window % 2 == 0:
                smooth_window += 1  # Ensure odd window size
            nutrient_total = savgol_filter(nutrient_total, smooth_window, 3)
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, nutrient_total, label='Total Nutrient', color='blue')
        plt.xlabel('Time Step')
        plt.ylabel('Total Nutrient Concentration')
        plt.title('Nutrient Evolution Over Time')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()


    def animate_nutrient_field(self, plane='XY', index=None, interval=200, save_as=None):
        """
        Animate 2D cross-sections of the nutrient field over time.
        
        Args:
            plane (str): Plane to slice ('XY', 'XZ', 'YZ'). Defaults to 'XY'.
            index (int, optional): Index along the perpendicular axis. Defaults to midpoint.
            interval (int): Delay between frames in milliseconds. Defaults to 200.
            save_as (str, optional): File path to save animation (e.g., 'nutrient_animation.mp4').
        """
        if 'nutrient concentration' not in self.history:
            raise ValueError("Nutrient concentration data not found in the simulation history.")
        
        # Get the shape from the first step to determine default index
        shape = self.history['nutrient concentration'][0].shape
        if index is None:
            if plane == 'XY':
                index = shape[2] // 2
            elif plane == 'XZ':
                index = shape[1] // 2
            elif plane == 'YZ':
                index = shape[0] // 2
            else:
                raise ValueError("Plane must be 'XY', 'XZ', or 'YZ'")
        
        # Set up the figure and axis
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Set up the grid coordinates based on the plane
        if plane == 'XY':
            x, y = np.arange(shape[0]), np.arange(shape[1])
            xlabel, ylabel = 'X', 'Y'
        elif plane == 'XZ':
            x, y = np.arange(shape[0]), np.arange(shape[2])
            xlabel, ylabel = 'X', 'Z'
        elif plane == 'YZ':
            x, y = np.arange(shape[1]), np.arange(shape[2])
            xlabel, ylabel = 'Y', 'Z'
        
        X, Y = np.meshgrid(x, y)
        
        # Create a finer grid for smoother visualization
        x_fine, y_fine = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
        X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
        
        # Get initial slice for setting up the plot
        if plane == 'XY':
            initial_slice = self.history['nutrient concentration'][0][:, :, index]
        elif plane == 'XZ':
            initial_slice = self.history['nutrient concentration'][0][:, index, :]
        elif plane == 'YZ':
            initial_slice = self.history['nutrient concentration'][0][index, :, :]
        
        # Apply Gaussian smoothing
        initial_slice = gaussian_filter(initial_slice, sigma=1.0)
        
        # Interpolate to finer grid
        initial_slice_fine = griddata((X.flatten(), Y.flatten()), initial_slice.flatten(), 
                                    (X_fine, Y_fine), method='cubic')
        
        # Find global min/max for consistent colormap
        all_nutrient = self.history['nutrient concentration']
        vmin = 0
        vmax = max(1.0, np.max([np.max(n) for n in all_nutrient]))
        
        # Create initial contour plot
        contour = ax.contourf(X_fine, Y_fine, initial_slice_fine, levels=50, 
                            cmap='viridis', vmin=vmin, vmax=vmax)
        fig.colorbar(contour, ax=ax, label='Nutrient Concentration')
        
        # Set labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'Nutrient Field ({plane} Plane, Step {self.history["step"][0]})')
        
        def update(frame):
            """Update function for animation"""
            # Get current slice data from history
            if plane == 'XY':
                slice_ = self.history['nutrient concentration'][frame][:, :, index]
            elif plane == 'XZ':
                slice_ = self.history['nutrient concentration'][frame][:, index, :]
            elif plane == 'YZ':
                slice_ = self.history['nutrient concentration'][frame][index, :, :]
            
            # Apply Gaussian smoothing
            slice_ = gaussian_filter(slice_, sigma=1.0)
            
            # Interpolate to finer grid
            slice_fine = griddata((X.flatten(), Y.flatten()), slice_.flatten(), 
                                (X_fine, Y_fine), method='cubic')
            
            # Clear the axis and create a new contour plot
            ax.clear()
            contour = ax.contourf(X_fine, Y_fine, slice_fine, levels=50, 
                                cmap='viridis', vmin=vmin, vmax=vmax)
            
            # Reset labels and title
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f'Nutrient Field ({plane} Plane, Step {self.history["step"][frame]})')
            
            return [contour]
        
        anim = FuncAnimation(fig, update, frames=len(self.history['step']), interval=interval, blit=False)
        
        if save_as:
            anim.save(save_as, writer='ffmpeg')  # Requires ffmpeg installed
        
        plt.tight_layout()
        plt.show()
        
        return anim  # Return the animation object to prevent garbage collection
        
def main():
    analyzer = NutrientAnalyzer('data/project_model_test_sim_data.npz')
    #analyzer.plot_nutrient_field(step_index=0, plane='XY')
    #analyzer.plot_nutrient_evolution(smooth_window=5)
    analyzer.animate_nutrient_field(plane='XY', interval=200, save_as='nutrient_animation.mp4')

if __name__ == "__main__":
    main()