#######################################################################################################
#######################################################################################################
#
#
#
#
# 3D Tumor Growth Simulation Model outlined in the README.md
#
#
# Author: Riley McNamara
# Date: 2025-02-18
#
#
#
#
#######################################################################################################
#######################################################################################################

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.ndimage import laplace
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter
from tqdm import tqdm  # Ensure this import is present

from src.utils.utils import experimental_params

class TumorGrowthModel:
    def __init__(self, grid_shape=(50, 50, 50), dx=0.025, dt=0.01, params=None):
        """
        Initialize the simulation grid and parameters.
        :param grid_shape: Shape of the simulation grid (2D for illustration)
        :param dx: Spatial resolution
        :param dt: Time step
        :param params: Optional dictionary of model parameters
        """
        self.grid_shape = grid_shape
        self.dx = dx
        self.dt = dt
        self.params = self._set_default_params() if params is None else params
        self.history = {
            'step': [],
            'stem cell concentration': [],
            'progenitor cell concentration': [],
            'differentiated cell concentration': [],
            'necrotic cell concentration': [],
            'total cell concentration': [],
            'stem cell volume': [],
            'progenitor cell volume': [],
            'differentiated cell volume': [],
            'necrotic cell volume': [],
            'total cell volume': [],
            'radius': []
        }
        self._initialize_fields()
        
    def _set_default_params(self):
        """
        Set a default parameter set based on the model equations in the README.
        """
        return experimental_params
    
    def _update_total_cell_density(self):
        """ 
        Update the total cell density field.
        """
        self.C_total = self.C_S + self.C_P + self.C_D + self.C_N

    def _initialize_fields(self):
        """
        Initialize the cell density fields, nutrient, and pressure.
        Start with a small spherical tumor core of stem cells.
        """
        shape = self.grid_shape
        self.C_S = np.zeros(shape)
        self.C_P = np.zeros(shape)
        self.C_D = np.zeros(shape)
        self.C_N = np.zeros(shape)
        self.nutrient = np.ones(shape)
        self.n_S = self.params['n_S'] * np.ones(shape)
        self.n_P = self.params['n_P'] * np.ones(shape)
        self.n_D = self.params['n_D'] * np.ones(shape)
        self.pressure = np.zeros(shape)
        
        # Create a small spherical initial tumor
        center = np.array([s//2 for s in shape])
        radius = 3  # Initial radius of tumor sphere
        
        x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
        dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
        
        # Set initial stem cell concentration in sphere
        self.C_S[dist_from_center <= radius] = 1.0
        
        self._update_total_cell_density()

    def _gradient(self, C: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of a given field C.
        """
        return np.gradient(C)
    
    def _compute_src_S(self) -> np.ndarray:
        """
        Compute the source term for the stem cells.
        src_S = \lamda_S n C_S (2p_0 -1) - \mu_S H(\hat {n_S} - n) C_S
        """
        lambda_S = self.params['lambda_S']
        n = self.nutrient
        C_S = self.C_S
        p_0 = self.params['p_0']
        mu_S = self.params['mu_S']
        n_S = self.n_S
        src_S = lambda_S * n * C_S * (2 * p_0 - 1) - mu_S * np.heaviside(n_S - n, 0) * C_S    
        return src_S
    
    def _compute_src_P(self) -> np.ndarray:
        """
        Compute the source term for the prologenitor cells.
        src_P = \lambda_S n 2(1 - p_0) C_S + \lambda_P n C_P (2p_1 - 1) - \mu_P H(\hat {n_P} - n) C_P
        """
        lambda_S = self.params['lambda_S']
        lambda_P = self.params['lambda_P']
        n = self.nutrient
        p_0 = self.params['p_0']
        p_1 = self.params['p_1']
        C_S = self.C_S
        C_P = self.C_P
        mu_P = self.params['mu_P']
        n_P = self.n_P
        src_P = lambda_S * n * 2 * (1 - p_0) * C_S + lambda_P * n * C_P * (2 * p_1 - 1) - mu_P * np.heaviside(n_P - n, 0) * C_P
        return src_P
    
    def _compute_src_D(self) -> np.ndarray:
        """
        Compute the source term for the differentiated cells.
        src_D = \lambda_P n 2(1 - p_1) C_P - \mu_D H(\hat {n_D} - n) C_D - \alpha_D C_D
        """
        lambda_P = self.params['lambda_P']
        n = self.nutrient
        p_1 = self.params['p_1']
        C_P = self.C_P
        C_D = self.C_D
        mu_D = self.params['mu_D']
        n_D = self.n_D
        alpha_D = self.params['alpha_D']
        src_D = lambda_P * n * 2 * (1 - p_1) * C_P - mu_D * np.heaviside(n_D - n, 0) * C_D - alpha_D * C_D
        return src_D
    
    def _compute_src_N(self) -> np.ndarray:
        """
        Compute the source term for the nutrient.
        src_N = \mu_S H(\hat {n_S} - n) C_S + \mu_P H(\hat {n_P} - n) C_P + \mu_D H(\hat {n_D} - n) C_D + \alpha_D C_D - \gamma_N C_N
        """
        lambda_S = self.params['lambda_S']
        lambda_P = self.params['lambda_P']
        n_S = self.n_S
        n_P = self.n_P
        n_D = self.n_D
        n = self.nutrient
        C_S = self.C_S
        C_P = self.C_P
        C_D = self.C_D
        C_N = self.C_N
        mu_S = self.params['mu_S']
        mu_P = self.params['mu_P']
        mu_D = self.params['mu_D']
        alpha_D = self.params['alpha_D']
        gamma_N = self.params['gamma_N']
        src_N = mu_S * np.heaviside(n_S - n, 0) * C_S + mu_P * np.heaviside(n_P - n, 0) * C_P + mu_D * np.heaviside(n_D - n, 0) * C_D + alpha_D * C_D - gamma_N * C_N
        return src_N
    
    def _compute_src_T(self) -> np.ndarray:
        """
        Compute the source term for the total cell density.
        src_T = \lambda_S n C_S + \lambda_P n C_P + \gamma_N C_N
        """
        lambda_S = self.params['lambda_S']
        lambda_P = self.params['lambda_P']
        n = self.nutrient
        C_S = self.C_S
        C_P = self.C_P
        C_N = self.C_N
        gamma_N = self.params['gamma_N']
        src_T = lambda_S * n * C_S + lambda_P * n * C_P + gamma_N * C_N
        return src_T
    
    def _compute_mass_flux(self, C: np.ndarray) -> tuple:
        """
        Compute the flux J for a given cell density field C.
        Returns tuple of flux components (Jx, Jy, Jz).
        """
        M = self.params['M']  # Mobility coefficient
        energy_deriv = self._compute_energy_derivative()
        grad_C = self._gradient(C)
        
        # Compute flux components with increased mobility
        Jx = -M * grad_C[0] * energy_deriv
        Jy = -M * grad_C[1] * energy_deriv
        Jz = -M * grad_C[2] * energy_deriv
        
        return Jx, Jy, Jz

    def _compute_energy_derivative(self) -> np.ndarray:
        self._update_total_cell_density()
        gamma = self.params['gamma']
        epsilon = self.params['epsilon']
        C_total = self.C_total
        # f'(C_T) for f(C_T)=1/4 * C_T^2 (1-C_T)^2 is 0.5 * C_total * (1 - C_total) * (2 * C_total - 1)
        energy_deriv = (gamma/epsilon) * (0.5 * C_total * (1 - C_total) * (2 * C_total - 1) - epsilon**2 * laplace(C_total))
        return energy_deriv

    def _compute_solid_velocity(self) -> tuple:
        """
        Compute the solid velocity field u_s based on:
            u_s = - (∇p + (δE/δC_T) ∇C_T)
        Returns tuple of velocity components (u_x, u_y, u_z)
        """
        self._update_total_cell_density()
        energy_deriv = self._compute_energy_derivative()
        grad_C_total = self._gradient(self.C_total)
        p = self._compute_internal_pressure()
        grad_p = self._gradient(p)
        
        # Calculate velocity components
        u_x = -(grad_p[0] + energy_deriv * grad_C_total[0])
        u_y = -(grad_p[1] + energy_deriv * grad_C_total[1])
        u_z = -(grad_p[2] + energy_deriv * grad_C_total[2])
        
        return u_x, u_y, u_z

    def _compute_internal_pressure(self):
        """
        Compute the internal pressure field using:
            ∇²p = S_T - ∇·((δE/δC_T) ∇C_T)
        with S_T = λ_S n C_S + λ_P n C_P + γ_N C_N.
        """
        self._update_total_cell_density()
        # Calculate source term
        S_T = self._compute_src_T()
        
        # Calculate energy derivative
        energy_deriv = self._compute_energy_derivative()
        
        # Get gradients (returns tuple of arrays for x, y, z components)
        grad_C_total = self._gradient(self.C_total)
        
        # Calculate divergence term
        divergence = 0
        for i in range(3):  # For each dimension
            # Calculate gradient of energy derivative in this dimension
            grad_energy = np.gradient(energy_deriv, self.dx, axis=i)
            # Multiply with gradient of total density and add to divergence
            divergence += grad_energy * grad_C_total[i]
        
        # Solve for pressure (here using a simple approximation)
        # In a more detailed implementation, you might want to solve the Poisson equation
        p = S_T - divergence
        
        return p
    
    def _compute_probabilities(self, nutrient, drug_effects=None):
        """
        Compute the cell division probabilities p₀ and p₁.
        In the base model these are constant (or nutrient-dependent) but here we include a placeholder for drug effects.
        Drug effects (if any) could modify these probabilities.
        """
        p_0 = 1
        p_1 = 1
        return p_0, p_1

    def _update(self, drug_effects=None):
        """
        Update all fields for one time step using operator splitting.
        """
        # Update probabilities based on current nutrient levels
        p0, p1 = self._compute_probabilities(self.nutrient, drug_effects)
        
        # 1. Reaction terms
        src_S = self._compute_src_S()
        src_P = self._compute_src_P()
        src_D = self._compute_src_D()
        src_N = self._compute_src_N()
      
        
        self.C_S += self.dt * src_S
        self.C_P += self.dt * src_P
        self.C_D += self.dt * src_D
        self.C_N += self.dt * src_N

        # 2. Advection terms
        u_x, u_y, u_z = self._compute_solid_velocity()
        
        # Update each cell population with mass flux
        for field in [self.C_S, self.C_P, self.C_D, self.C_N]:
            # Apply advection in each direction
            field -= self.dt * (
                np.gradient(field * u_x, self.dx, axis=0) +
                np.gradient(field * u_y, self.dx, axis=1) +
                np.gradient(field * u_z, self.dx, axis=2)
            )

            J = self._compute_mass_flux(field)  # Compute the vector field J
            # Calculate the divergence of J manually
            divergence = np.gradient(J[0], self.dx, axis=0) + np.gradient(J[1], self.dx, axis=1) + np.gradient(J[2], self.dx, axis=2)
            mass_flux = -divergence  # Mass flux is the negative divergence
            field -= self.dt * mass_flux  # Update the field with mass flux
        
        
        # 3. Nutrient diffusion with consumption
        D_n = self.params['D_n']
        consumption_rate = 0.1  # Rate at which cells consume nutrient
        self.nutrient += self.dt * (
            D_n * laplace(self.nutrient) - 
            consumption_rate * self.C_total * self.nutrient
        )
        
        # Ensure non-negativity and update total density
        fields = [self.C_S, self.C_P, self.C_D, self.C_N, self.nutrient]
        for field in fields:
            np.clip(field, 0, None, out=field)
        
        self._update_total_cell_density()

    def _run_simulation(self, steps=100, drug_schedule=None, plot_interval=60):
        """
        Run the simulation for a given number of time steps.
        :param steps: Number of simulation steps
        :param drug_schedule: Dictionary mapping time steps to drug effects
        :param plot_interval: Number of steps between plots
        """
        # Initialize storage for volumes and radius over time
        self.volume_history = {
            'step': [],
            'stem': [],
            'progenitor': [],
            'differentiated': [],
            'necrotic': [],
            'total': [],
            'radius': []  # Track radius
        }
        
        cell_volume = self.dx ** 3  # Volume of a single grid cell
        
        for step in tqdm(range(steps), desc="Running Simulation"):
            # Apply drug effects if scheduled
            drug_effects = drug_schedule.get(step) if drug_schedule else None
            
            # Update simulation
            self._update(drug_effects)
            
            # Store volumes based on occupied cells
            self.history['step'].append(step)
            self.history['stem cell concentration'].append(self.C_S)
            self.history['progenitor cell concentration'].append(self.C_P)
            self.history['differentiated cell concentration'].append(self.C_D)
            self.history['necrotic cell concentration'].append(self.C_N)
            self.history['total cell concentration'].append(self.C_total)
            self.history['stem cell volume'].append(np.sum(self.C_S) * cell_volume)
            self.history['progenitor cell volume'].append(np.sum(self.C_P) * cell_volume)
            self.history['differentiated cell volume'].append(np.sum(self.C_D) * cell_volume)
            self.history['necrotic cell volume'].append(np.sum(self.C_N) * cell_volume)
            self.history['total cell volume'].append(np.sum(self.C_total) * cell_volume)
            
            # Calculate and store the tumor radius
            center = np.array([s // 2 for s in self.grid_shape])
            x, y, z = np.ogrid[:self.grid_shape[0], :self.grid_shape[1], :self.grid_shape[2]]
            dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
            radius = np.max(dist_from_center[self.C_S > 0])  # Find max distance where C_S > 0
            self.history['radius'].append(radius)

            
            # Plot at specified intervals
            if step == steps - 1:
                self._plot_all_isosurfaces(step)
                self._plot_history()

    def _plot_fields(self, step):
        """
        Create a 3D visualization of all cell populations using different colors and transparency.
        Dynamically focuses on the region containing cells.
        """
        from mpl_toolkits.mplot3d import Axes3D
        
        # Create figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set threshold for visualization
        threshold = 0.1
        
        # Find the bounds of the tumor (where any cell type exists above threshold)
        total_mask = (self.C_S > threshold) | (self.C_P > threshold) | \
                    (self.C_D > threshold) | (self.C_N > threshold)
        
        if not total_mask.any():
            print("No cells above threshold found.")
            return
        
        # Get the indices where cells exist
        x_indices, y_indices, z_indices = np.where(total_mask)
        
        # Calculate bounds based on the actual indices of the tumor
        if x_indices.size > 0:
            x_min, x_max = np.min(x_indices), np.max(x_indices) + 1  # Adjusted to include max
            y_min, y_max = np.min(y_indices), np.max(y_indices) + 1  # Adjusted to include max
            z_min, z_max = np.min(z_indices), np.max(z_indices) + 1  # Adjusted to include max
        else:
            # Fallback to default if no cells are found
            x_min, x_max = self.grid_shape[0]//2 - 5, self.grid_shape[0]//2 + 5
            y_min, y_max = self.grid_shape[1]//2 - 5, self.grid_shape[1]//2 + 5
            z_min, z_max = self.grid_shape[2]//2 - 5, self.grid_shape[2]//2 + 5
        
        # Add a small padding to the bounds for better visibility
        padding = 2  # Adjust this value to change the amount of space around the tumor
        x_min, x_max = x_min - padding, x_max + padding
        y_min, y_max = y_min - padding, y_max + padding
        z_min, z_max = z_min - padding, z_max + padding
        
        # Get the voxel array from the region of interest
        voxels = total_mask[x_min:x_max, y_min:y_max, z_min:z_max]
        nx, ny, nz = voxels.shape  # e.g., (11, 11, 11)

        # Create coordinate arrays that span the voxel boundaries
        x_coords = np.linspace(x_min, x_max, nx + 1)
        y_coords = np.linspace(y_min, y_max, ny + 1)
        z_coords = np.linspace(z_min, z_max, nz + 1)
        X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')

        # Now pass these to ax.voxels
        ax.voxels(X, Y, Z, voxels, facecolors='red', edgecolor='k', alpha=0.5)

 
        # Add text with total volumes and bounds
        total_volumes = (
            f'Volumes:\n'
            f'Stem: {np.sum(self.C_S):.1f}\n'
            f'Prog: {np.sum(self.C_P):.1f}\n'
            f'Diff: {np.sum(self.C_D):.1f}\n'
            f'Necr: {np.sum(self.C_N):.1f}\n'
            f'\nBounds:\n'
            f'X: [{x_min}, {x_max}]\n'
            f'Y: [{y_min}, {y_max}]\n'
            f'Z: [{z_min}, {z_max}]'
        )
        plt.figtext(0.02, 0.02, total_volumes, fontsize=8)
        
        # Set axis limits to the region of interest
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])
        
        plt.show()

    def _plot_all_isosurfaces(self, step, threshold=0.01):
        """
        Plot isosurfaces for all cell concentration fields in one 3D plot.
        Automatically focuses on the region containing cells.
        """
        # Define fields and colors for plotting
        fields = {
            'Stem': (self.C_S, 'red'),
            'Progenitor': (self.C_P, 'blue'),
            'Differentiated': (self.C_D, 'green'),
            'Necrotic': (self.C_N, 'black')
        }
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Find the bounds of the tumor (where any cell type exists above threshold)
        total_mask = np.zeros_like(self.C_S, dtype=bool)
        for name, (field, _) in fields.items():
            total_mask |= (field > threshold)
        
        if not total_mask.any():
            print("No cells above threshold found.")
            return
        
        # Get the indices where cells exist
        x_indices, y_indices, z_indices = np.where(total_mask)
        
        # Calculate bounds based on the actual indices of the tumor
        padding = 5  # Adjust padding to control space around tumor
        x_min, x_max = max(0, np.min(x_indices) - padding), min(self.grid_shape[0], np.max(x_indices) + padding)
        y_min, y_max = max(0, np.min(y_indices) - padding), min(self.grid_shape[1], np.max(y_indices) + padding)
        z_min, z_max = max(0, np.min(z_indices) - padding), min(self.grid_shape[2], np.max(z_indices) + padding)
        
        # Convert to spatial coordinates
        x_min, x_max = x_min * self.dx, x_max * self.dx
        y_min, y_max = y_min * self.dx, y_max * self.dx
        z_min, z_max = z_min * self.dx, z_max * self.dx
        
        # Loop over each concentration field and extract its isosurface
        for name, (field, color) in fields.items():
            # Apply Gaussian filter for smoothing
            smoothed_field = gaussian_filter(field, sigma=1)
            
            if np.max(smoothed_field) < threshold:
                continue
            
            try:
                verts, faces, normals, _ = marching_cubes(smoothed_field, level=threshold, spacing=(self.dx, self.dx, self.dx))
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
        ax.set_title(f'Tumor Isosurfaces at Step {step} (Threshold = {threshold})')
        
        #plt.show()

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

    def _plot_history(self):
        """
        Plot the volume evolution of each cell type over time.
        """
        plt.figure(figsize=(10, 6))
        
        # plot tumor volume evolution
        steps = self.history['step']
        plt.plot(steps, self.history['stem cell volume'], 'r-', label='Stem')
        plt.plot(steps, self.history['progenitor cell volume'], 'b-', label='Progenitor')
        plt.plot(steps, self.history['differentiated cell volume'], 'g-', label='Differentiated')
        plt.plot(steps, self.history['necrotic cell volume'], 'k-', label='Necrotic')
        plt.plot(steps, self.history['total cell volume'], 'gray', linestyle='--', label='Total')
        plt.xlabel('Time Step')
        plt.ylabel('Volume')
        plt.title('Tumor Volume Evolution')
        plt.legend()
        plt.grid(True)
        plt.show()

        # plot tumor radius evolution
        plt.figure(figsize=(10, 6))
        plt.plot(steps, self.history['radius'], 'purple', label='Radius')
        plt.xlabel('Time Step')
        plt.ylabel('Radius')
        plt.title('Tumor Radius Evolution')
        plt.legend()
        plt.grid(True)
        plt.show()

        # plot tumor conc

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
        import matplotlib.animation as animation
        
        # Set default slice position to middle of grid if not specified
        if slice_pos is None:
            slice_pos = self.grid_shape[0] // 2
        
        # Create figure and subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle(f'Tumor Growth Animation (X-Slice at {slice_pos})', fontsize=16)
        
        # Flatten axes for easier iteration
        axes = axes.flatten()
        
        # Dictionary of cell types and their properties
        cell_types = {
            'Stem Cells': (self.C_S, 'Reds'),
            'Progenitor Cells': (self.C_P, 'Blues'),
            'Differentiated Cells': (self.C_D, 'Greens'),
            'Necrotic Cells': (self.C_N, 'Greys')
        }
        
        # Create mesh grid for plotting
        y, z = np.meshgrid(np.arange(self.grid_shape[1]), np.arange(self.grid_shape[2]))
        
        # Initialize contour plots and store them
        contour_artists = []
        for ax, (title, (field, cmap)) in zip(axes, cell_types.items()):
            # Create initial contour plot
            ax.set_title(title)
            
            # Create empty contour plot
            cont = ax.contourf(y, z, np.zeros((self.grid_shape[1], self.grid_shape[2])),
                            levels=np.linspace(0, 1, 20),
                            cmap=cmap)
            fig.colorbar(cont, ax=ax)
            contour_artists.append(cont)
        
        # Make figure layout tight
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        def update(frame):
            """Update function for animation"""
            # Update simulation
            self._update()
            
            # Clear and update each subplot
            for ax, cont, (title, (field, cmap)) in zip(axes, contour_artists, cell_types.items()):
                # Clear the axis
                ax.clear()
                
                # Get current slice data
                slice_data = field[slice_pos, :, :].T
                
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
        anim.save('tumor_growth_animation2.gif', writer='pillow', fps=5)
        #plt.show()
        
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
        import matplotlib.animation as animation
        
        # Set default slice position to middle of grid if not specified
        if slice_pos is None:
            slice_pos = self.grid_shape[0] // 2
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))
        fig.suptitle(f'Combined Tumor Growth Animation (X-Slice at {slice_pos})', fontsize=16)
        
        # Dictionary of cell types and their properties
        cell_types = {
            'Stem Cells': (self.C_S, 'red'),
            'Progenitor Cells': (self.C_P, 'blue'),
            'Differentiated Cells': (self.C_D, 'green'),
            'Necrotic Cells': (self.C_N, 'black')
        }
        
        # Create mesh grid for plotting
        y, z = np.meshgrid(np.arange(self.grid_shape[1]), np.arange(self.grid_shape[2]))
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], color=color, label=title)
                        for title, (_, color) in cell_types.items()]
        ax.legend(handles=legend_elements, loc='upper right')
        
        def update(frame):
            """Update function for animation"""
            # Update simulation
            self._update()
            
            # Clear the axis
            ax.clear()
            
            # Redraw all contours
            for title, (field, color) in cell_types.items():
                slice_data = field[slice_pos, :, :].T
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

    def _animate_tumor_growth_isosurfaces(self, steps=100, threshold=0.01, interval=100):
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
        import matplotlib.animation as animation
        
        # Create figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Define fields and colors for plotting
        fields = {
            'Stem': (self.C_S, 'red'),
            'Progenitor': (self.C_P, 'blue'),
            'Differentiated': (self.C_D, 'green'),
            'Necrotic': (self.C_N, 'black')
        }
        
        # Initialize collections dictionary to store mesh objects
        mesh_collections = {name: None for name in fields.keys()}
        
        # Initialize text annotations
        text_annotations = {name: None for name in fields.keys()}
        
        def update(frame):
            """Update function for animation""" 
            # Update simulation
            self._update()
            
            # Clear previous frame
            ax.cla()
            
            # Find the bounds of the tumor
            total_mask = np.zeros_like(self.C_S, dtype=bool)
            for name, (field, _) in fields.items():
                total_mask |= (field > threshold)
            
            if not total_mask.any():
                return
            
            # Get the indices where cells exist
            x_indices, y_indices, z_indices = np.where(total_mask)
            
            # Calculate bounds with padding
            padding = 5
            x_min, x_max = max(0, np.min(x_indices) - padding), min(self.grid_shape[0], np.max(x_indices) + padding)
            y_min, y_max = max(0, np.min(y_indices) - padding), min(self.grid_shape[1], np.max(y_indices) + padding)
            z_min, z_max = max(0, np.min(z_indices) - padding), min(self.grid_shape[2], np.max(z_indices) + padding)
            
            # Convert to spatial coordinates
            x_min, x_max = x_min * self.dx, x_max * self.dx
            y_min, y_max = y_min * self.dx, y_max * self.dx
            z_min, z_max = z_min * self.dx, z_max * self.dx
            
            # Plot isosurfaces for each cell type
            for name, (field, color) in fields.items():
                # Apply Gaussian filter for smoothing
                smoothed_field = gaussian_filter(field, sigma=1)
                
                if np.max(smoothed_field) < threshold:
                    continue
                
                try:
                    verts, faces, normals, _ = marching_cubes(smoothed_field, level=threshold, 
                                                            spacing=(self.dx, self.dx, self.dx))
                    mesh = Poly3DCollection(verts[faces], alpha=0.2)
                    mesh.set_facecolor(color)
                    mesh.set_edgecolor(color)
                    ax.add_collection3d(mesh)
                    
                    # Add text annotation at center of isosurface
                    center = np.mean(verts, axis=0)
                    ax.text(center[0], center[1], center[2], 
                        f'{name}\nVol: {np.sum(field):.1f}', 
                        color=color, fontsize=8)
                    
                except Exception as e:
                    print(f"Could not extract isosurface for {name} at frame {frame}: {e}")
            
            # Set axis limits and labels
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
            ax.set_zlim([z_min, z_max])
            
            # Draw bounding box
            self._draw_bounding_box(ax, (x_min, x_max), (y_min, y_max), (z_min, z_max))
            
            # Set labels and title
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Tumor Growth')
            
            # Set optimal viewing angle
            ax.view_init(elev=20, azim=frame % 360)  # Rotate view during animation
            
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

if __name__ == "__main__":
    # Initialize the model with a specified grid shape
    model = TumorGrowthModel(grid_shape=(100, 100, 100))

    # Add initial cylindrical tumor seed at the center
    center = tuple(s // 2 for s in model.grid_shape)
    radius = 3  # Initial radius of tumor cylinder
    height = 5  # Use the full height of the grid

    x, y, z = np.ogrid[:model.grid_shape[0], :model.grid_shape[1], :model.grid_shape[2]]
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)

    # Set initial stem cell concentration in cylinder
    model.C_S[(dist_from_center <= radius) & (z >= 0) & (z < height)] = 1.0

    # Run the simulation
    #model._run_simulation(steps=200)

    # Create animation of tumor growth using isosurfaces
    model._animate_tumor_growth_isosurfaces(steps=300)