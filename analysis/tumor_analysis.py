import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes

from analysis.scie3121_analysis import scie3121SimulationAnalyzer
from src.models.cell_dynamics import compute_adhesion_energy_derivative
from src.utils.utils import SCIE3121_params

class TumorAnalyzer(scie3121SimulationAnalyzer):
    def __init__(self, filepath):
        super().__init__(filepath)
    
    def compute_adhesion_energy(self, phi, dx, gamma, epsilon):
        """
        Compute the adhesion energy field.
        E = (γ/ε)∫[f(φ) + (ε²/2)|∇φ|²] dx
        where f(φ) = (1/4)φ²(1-φ)² is the double-well potential.
        """
        # Double-well potential f(φ) = (1/4)φ²(1-φ)²
        f_phi = 0.25 * phi**2 * (1-phi)**2
        
        # Compute gradient magnitude squared |∇φ|²
        grad_x = np.gradient(phi, dx, axis=0)
        grad_y = np.gradient(phi, dx, axis=1)
        grad_z = np.gradient(phi, dx, axis=2)
        grad_mag_squared = grad_x**2 + grad_y**2 + grad_z**2
        
        # Compute adhesion energy density
        # E = (γ/ε)[f(φ) + (ε²/2)|∇φ|²]
        energy_density = (gamma/epsilon) * (f_phi + (epsilon**2/2) * grad_mag_squared)
        
        return energy_density
    
    def compute_adhesion_energy_gradient(self, energy, dx):
        """
        Compute the gradient of the adhesion energy field.
        """
        grad_x = -np.gradient(energy, dx, axis=0)
        grad_y = -np.gradient(energy, dx, axis=1)
        grad_z = -np.gradient(energy, dx, axis=2)
        
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        
        return grad_x, grad_y, grad_z, grad_magnitude
    
    def compute_tumor_gradient(self, phi_T, dx):
        """
        Compute the gradient of the total tumor volume fraction.
        """
        grad_x = np.gradient(phi_T, dx, axis=0)
        grad_y = np.gradient(phi_T, dx, axis=1)
        grad_z = np.gradient(phi_T, dx, axis=2)
        
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        
        return grad_x, grad_y, grad_z, grad_magnitude
    
    def compute_adhesion_energy_derivative(self, phi, dx, gamma, epsilon):
        """
        Compute the variational derivative of the adhesion energy.
        δE/δφ = (γ/ε)[f'(φ) - ε²∇²φ]
        where f'(φ) = (1/2)φ(1-φ)(2φ-1) is the derivative of the double-well potential.
        """
        # Derivative of double-well potential f'(φ) = (1/2)φ(1-φ)(2φ-1)
        f_prime = 0.5 * phi * (1-phi) * (2*phi - 1)
        
        # Compute Laplacian ∇²φ
        laplace_phi = self.laplacian(phi, dx)
        
        # Compute energy derivative δE/δφ = (γ/ε)[f'(φ) - ε²∇²φ]
        energy_deriv = (gamma/epsilon) * (f_prime - epsilon**2 * laplace_phi)
        
        return energy_deriv

    def laplacian(self, field, dx):
        """
        Compute the Laplacian of a field.
        ∇²φ = ∂²φ/∂x² + ∂²φ/∂y² + ∂²φ/∂z²
        """
        grad_x = np.gradient(field, dx, axis=0)
        grad_y = np.gradient(field, dx, axis=1)
        grad_z = np.gradient(field, dx, axis=2)
        
        return (np.gradient(grad_x, dx, axis=0) + 
                np.gradient(grad_y, dx, axis=1) + 
                np.gradient(grad_z, dx, axis=2))
    
    def compute_energy_derivative_gradient(self, phi, dx, gamma, epsilon):
        """
        Compute the gradient of the variational derivative of adhesion energy.
        This gradient drives the mass flux J_i = -M_i ∇(δE/δφ_i)
        """
        # First compute the energy derivative δE/δφ
        energy_deriv = self.compute_adhesion_energy_derivative(phi, dx, gamma, epsilon)
        
        # Then compute its gradient
        grad_x = np.gradient(energy_deriv, dx, axis=0)
        grad_y = np.gradient(energy_deriv, dx, axis=1)
        grad_z = np.gradient(energy_deriv, dx, axis=2)
        
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        
        return grad_x, grad_y, grad_z, grad_magnitude

    def visualize_tumor_analysis(self, model, step, mode='all', plane='XY', index=None, 
                                cmap='viridis', threshold=0.1):
        """
        Visualize tumor gradient, adhesion energy, adhesion energy derivative, and energy derivative gradient.
        
        Args:
            model: The simulation model object
            step (int): The time step to visualize
            mode (str): Visualization mode - 'all', 'tumor_gradient', 'adhesion_energy', 'energy_gradient', 'energy_deriv_gradient'
            plane (str): Plane to slice ('XY', 'XZ', 'YZ')
            index (int, optional): Index along the perpendicular axis. Defaults to midpoint.
            cmap (str): Colormap for visualization
            threshold (float): Threshold for tumor outline
        """
        # Get the volume fractions for the specified step
        try:
            phi_H = self.history['healthy cell volume fraction'][step]
            phi_D = self.history['diseased cell volume fraction'][step]
            phi_N = self.history['necrotic cell volume fraction'][step]
        except (KeyError, IndexError) as e:
            print(f"Error accessing history data at step {step}: {e}")
            return
        
        # Calculate total tumor volume fraction
        phi_T = phi_H + phi_D + phi_N
        
        # Get shape and determine default index if not provided
        shape = phi_T.shape
        if index is None:
            if plane == 'XY':
                index = shape[2] // 2
            elif plane == 'XZ':
                index = shape[1] // 2
            elif plane == 'YZ':
                index = shape[0] // 2
        
        # Extract slice based on the plane
        if plane == 'XY':
            phi_T_slice = phi_T[:, :, index]
            x_label, y_label = 'X', 'Y'
        elif plane == 'XZ':
            phi_T_slice = phi_T[:, index, :]
            x_label, y_label = 'X', 'Z'
        elif plane == 'YZ':
            phi_T_slice = phi_T[index, :, :]
            x_label, y_label = 'Y', 'Z'
        
        # Compute tumor gradient
        grad_x, grad_y, grad_z, grad_magnitude = self.compute_tumor_gradient(phi_T, model.dx)
        
        # Extract gradient slice
        if plane == 'XY':
            grad_x_slice = grad_x[:, :, index]
            grad_y_slice = grad_y[:, :, index]
            grad_mag_slice = grad_magnitude[:, :, index]
        elif plane == 'XZ':
            grad_x_slice = grad_x[:, index, :]
            grad_y_slice = grad_z[:, index, :]  # z instead of y for XZ plane
            grad_mag_slice = grad_magnitude[:, index, :]
        elif plane == 'YZ':
            grad_x_slice = grad_y[:, :, index]  # y instead of x for YZ plane
            grad_y_slice = grad_z[:, :, index]
            grad_mag_slice = grad_magnitude[:, :, index]
        
        # Compute adhesion energy
        adhesion_energy = self.compute_adhesion_energy(
            phi_T, model.dx, model.params['gamma'], model.params['epsilon']
        )
        
        # Extract energy slice
        if plane == 'XY':
            energy_slice = adhesion_energy[:, :, index]
        elif plane == 'XZ':
            energy_slice = adhesion_energy[:, index, :]
        elif plane == 'YZ':
            energy_slice = adhesion_energy[index, :, :]
        
        # Compute adhesion energy derivative
        energy_deriv = self.compute_adhesion_energy_derivative(
            phi_T, model.dx, model.params['gamma'], model.params['epsilon']
        )
        
        # Extract energy derivative slice
        if plane == 'XY':
            energy_deriv_slice = energy_deriv[:, :, index]
        elif plane == 'XZ':
            energy_deriv_slice = energy_deriv[:, index, :]
        elif plane == 'YZ':
            energy_deriv_slice = energy_deriv[index, :, :]
        
        # Compute energy gradient
        e_grad_x, e_grad_y, e_grad_z, e_grad_magnitude = self.compute_adhesion_energy_gradient(
            adhesion_energy, model.dx
        )
        
        # Extract energy gradient slice
        if plane == 'XY':
            e_grad_x_slice = e_grad_x[:, :, index]
            e_grad_y_slice = e_grad_y[:, :, index]
            e_grad_mag_slice = e_grad_magnitude[:, :, index]
        elif plane == 'XZ':
            e_grad_x_slice = e_grad_x[:, index, :]
            e_grad_y_slice = e_grad_z[:, index, :]
            e_grad_mag_slice = e_grad_magnitude[:, index, :]
        elif plane == 'YZ':
            e_grad_x_slice = e_grad_y[:, :, index]
            e_grad_y_slice = e_grad_z[:, :, index]
            e_grad_mag_slice = e_grad_magnitude[:, :, index]
        
        # Compute energy derivative gradient (drives mass flux)
        ed_grad_x, ed_grad_y, ed_grad_z, ed_grad_magnitude = self.compute_energy_derivative_gradient(
            phi_T, model.dx, model.params['gamma'], model.params['epsilon']
        )
        
        # Extract energy derivative gradient slice
        if plane == 'XY':
            ed_grad_x_slice = ed_grad_x[:, :, index]
            ed_grad_y_slice = ed_grad_y[:, :, index]
            ed_grad_mag_slice = ed_grad_magnitude[:, :, index]
        elif plane == 'XZ':
            ed_grad_x_slice = ed_grad_x[:, index, :]
            ed_grad_y_slice = ed_grad_z[:, index, :]
            ed_grad_mag_slice = ed_grad_magnitude[:, index, :]
        elif plane == 'YZ':
            ed_grad_x_slice = ed_grad_y[:, :, index]
            ed_grad_y_slice = ed_grad_z[:, :, index]
            ed_grad_mag_slice = ed_grad_magnitude[:, :, index]
        
        # Visualize based on mode
        if mode == 'all':
            # Extended figure with 3 rows to include energy derivative gradient
            fig, axes = plt.subplots(3, 3, figsize=(18, 18))
            
            # Row 1: Tumor volume and gradient
            self._plot_scalar_field(phi_T_slice, axes[0, 0], x_label, y_label, 
                                   'Total Tumor Volume Fraction', cmap='viridis', 
                                   show_contour=True, threshold=threshold)
            
            self._plot_scalar_field(grad_mag_slice, axes[0, 1], x_label, y_label, 
                                   'Tumor Gradient Magnitude', cmap='plasma', 
                                   show_contour=True, threshold=threshold)
            
            self._plot_vector_field(grad_x_slice, grad_y_slice, phi_T_slice, axes[0, 2], 
                                   x_label, y_label, 'Tumor Gradient Vectors', 
                                   show_contour=True, threshold=threshold)
            
            # Row 2: Adhesion energy and derivative
            self._plot_scalar_field(energy_slice, axes[1, 0], x_label, y_label, 
                                   'Adhesion Energy', cmap='inferno', 
                                   show_contour=True, threshold=threshold)
            
            self._plot_scalar_field(energy_deriv_slice, axes[1, 1], x_label, y_label, 
                                   'Adhesion Energy Derivative', cmap='coolwarm', 
                                   show_contour=True, threshold=threshold)
            
            self._plot_vector_field(e_grad_x_slice, e_grad_y_slice, phi_T_slice, axes[1, 2], 
                                   x_label, y_label, 'Adhesion Energy Gradient', 
                                   show_contour=True, threshold=threshold)
            
            # Row 3: Energy derivative gradient (drives mass flux)
            self._plot_scalar_field(ed_grad_mag_slice, axes[2, 0], x_label, y_label, 
                                   'Energy Derivative Gradient Magnitude', cmap='viridis', 
                                   show_contour=True, threshold=threshold)
            
            # Add flux explanation text in the middle panel of the last row
            axes[2, 1].text(0.5, 0.5, 
                           'Mass Flux J_i = -M_i ∇(δE/δφ_i)\n\n'
                           'The energy derivative gradient\n'
                           'drives cell movement due to adhesion forces.\n'
                           'Cells move from high to low values of δE/δφ.', 
                           horizontalalignment='center',
                           verticalalignment='center',
                           transform=axes[2, 1].transAxes,
                           fontsize=12,
                           bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
            axes[2, 1].axis('off')
            
            self._plot_vector_field(ed_grad_x_slice, ed_grad_y_slice, phi_T_slice, axes[2, 2], 
                                   x_label, y_label, 'Energy Derivative Gradient Vectors\n(Mass Flux Direction)', 
                                   show_contour=True, threshold=threshold)
            
            plt.tight_layout()
            plt.suptitle(f'Tumor Analysis - Step {step} - {plane} Plane (Index {index})', 
                        fontsize=16, y=1.02)
            
        elif mode == 'energy_deriv_gradient':
            # Create a figure specifically for energy derivative gradient
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Energy derivative
            self._plot_scalar_field(energy_deriv_slice, axes[0], x_label, y_label, 
                                   'Adhesion Energy Derivative (δE/δφ)', cmap='coolwarm', 
                                   show_contour=True, threshold=threshold)
            
            # Energy derivative gradient magnitude
            self._plot_scalar_field(ed_grad_mag_slice, axes[1], x_label, y_label, 
                                   'Energy Derivative Gradient Magnitude', cmap='viridis', 
                                   show_contour=True, threshold=threshold)
            
            # Energy derivative gradient vectors
            self._plot_vector_field(ed_grad_x_slice, ed_grad_y_slice, phi_T_slice, axes[2], 
                                   x_label, y_label, 'Energy Derivative Gradient Vectors\n(Mass Flux Direction)', 
                                   show_contour=True, threshold=threshold)
            
            plt.tight_layout()
            plt.suptitle(f'Energy Derivative Gradient Analysis - Step {step} - {plane} Plane (Index {index})', 
                        fontsize=16, y=1.02)
        
        # Keep existing modes
        elif mode == 'tumor_gradient':
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Tumor volume fraction
            self._plot_scalar_field(phi_T_slice, axes[0], x_label, y_label, 
                                   'Total Tumor Volume Fraction', cmap='viridis', 
                                   show_contour=True, threshold=threshold)
            
            # Tumor gradient magnitude
            self._plot_scalar_field(grad_mag_slice, axes[1], x_label, y_label, 
                                   'Tumor Gradient Magnitude', cmap='plasma', 
                                   show_contour=True, threshold=threshold)
            
            # Tumor gradient vectors
            self._plot_vector_field(grad_x_slice, grad_y_slice, phi_T_slice, axes[2], 
                                   x_label, y_label, 'Tumor Gradient Vectors', 
                                   show_contour=True, threshold=threshold)
            
            plt.tight_layout()
            plt.suptitle(f'Tumor Gradient Analysis - Step {step} - {plane} Plane (Index {index})', 
                        fontsize=16, y=1.02)
        
        elif mode == 'adhesion_energy':
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # Adhesion energy
            self._plot_scalar_field(energy_slice, axes[0], x_label, y_label, 
                                   'Adhesion Energy', cmap='inferno', 
                                   show_contour=True, threshold=threshold)
            
            # Adhesion energy derivative
            self._plot_scalar_field(energy_deriv_slice, axes[1], x_label, y_label, 
                                   'Adhesion Energy Derivative', cmap='coolwarm', 
                                   show_contour=True, threshold=threshold)
            
            plt.tight_layout()
            plt.suptitle(f'Adhesion Energy Analysis - Step {step} - {plane} Plane (Index {index})', 
                        fontsize=16, y=1.02)
        
        elif mode == 'energy_gradient':
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Adhesion energy
            self._plot_scalar_field(energy_slice, axes[0], x_label, y_label, 
                                   'Adhesion Energy', cmap='inferno', 
                                   show_contour=True, threshold=threshold)
            
            # Energy gradient magnitude
            self._plot_scalar_field(e_grad_mag_slice, axes[1], x_label, y_label, 
                                   'Energy Gradient Magnitude', cmap='plasma', 
                                   show_contour=True, threshold=threshold)
            
            # Energy gradient vectors
            self._plot_vector_field(e_grad_x_slice, e_grad_y_slice, phi_T_slice, axes[2], 
                                   x_label, y_label, 'Energy Gradient Vectors', 
                                   show_contour=True, threshold=threshold)
            
            plt.tight_layout()
            plt.suptitle(f'Energy Gradient Analysis - Step {step} - {plane} Plane (Index {index})', 
                        fontsize=16, y=1.02)
        
        plt.show()
    
    def _plot_scalar_field(self, field, ax, x_label, y_label, title, cmap='viridis', 
                          show_contour=True, threshold=0.1):
        """Helper method to plot scalar fields with optional contour."""
        # Apply mild smoothing for visualization
        field_smooth = gaussian_filter(field, sigma=1.0)
        
        im = ax.imshow(field_smooth, origin='lower', cmap=cmap)
        plt.colorbar(im, ax=ax, label=title)
        
        # Add contour if requested
        if show_contour:
            ax.contour(field, levels=[threshold], colors='white', linewidths=1.5)
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
    
    def _plot_vector_field(self, vector_x, vector_y, background, ax, x_label, y_label, 
                          title, show_contour=True, threshold=0.1):
        """Helper method to plot vector fields with optional contour."""
        # Downsample for clearer vector field
        shape = vector_x.shape
        downsample = max(1, min(shape) // 40)
        
        x_indices = np.arange(0, shape[0], downsample)
        y_indices = np.arange(0, shape[1], downsample)
        X, Y = np.meshgrid(y_indices, x_indices)  # Note: x and y are reversed for proper plotting
        
        # Extract downsampled vectors
        U = vector_x[::downsample, ::downsample].T  # Transpose for proper orientation
        V = vector_y[::downsample, ::downsample].T
        
        # Normalize vectors for better visualization
        magnitude = np.sqrt(U**2 + V**2)
        max_mag = np.max(magnitude) if np.max(magnitude) > 0 else 1.0
        U = U / max_mag
        V = V / max_mag
        
        # Plot background for context
        ax.imshow(background, origin='lower', cmap='gray', alpha=0.3)
        
        # Plot vector field
        quiver = ax.quiver(X, Y, U, V, magnitude, cmap='viridis', 
                          scale=15, width=0.005, pivot='mid')
        plt.colorbar(quiver, ax=ax, label='Magnitude')
        
        # Add contour if requested
        if show_contour:
            ax.contour(background, levels=[threshold], colors='white', linewidths=1.5)
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)


def main():
    """Test function to demonstrate tumor analysis."""
    from src.models.SCIE3121_model import SCIE3121_MODEL
    from src.models.initial_conditions import SphericalTumor
    
    # Create model instance
    model = SCIE3121_MODEL(
        grid_shape=(50, 50, 50),
        dx=0.2,
        dt=0.001,
        params=SCIE3121_params,
        initial_conditions=SphericalTumor(grid_shape=(50, 50, 50), radius=7, nutrient_value=1.0),
    )
    
    # Create analyzer instance
    analyzer = TumorAnalyzer(filepath='data/project_model_test_sim_data.npz')
    
    # Run visualization
    analyzer.visualize_tumor_analysis(
        model, 
        step=0,  # Analyze first time step
        mode='all',
        plane='XY',
        threshold=0.1
    )

if __name__ == "__main__":
    main()

    