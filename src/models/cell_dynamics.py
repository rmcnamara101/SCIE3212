####################################################################################
####################################################################################
#
#
#   Cell Dynamics Model
#
#   This model computes the dynamics of the cell concentration fields
#   based on the solid velocity field and the mass flux, defined as follows:
#
#   dC = -∇·(u C) -  ∇·J
#   dt
#
#   where u is the solid velocity field, J is the mass flux, and M is the mobility.
#
#   The solid velocity is defined and computed using the following formulas::
#
#   u = - (∇p + ( δE ) ∇C_T)
#               (δC_T)
#
#    ∇²p = S_T - ∇·(( δE ) ∇C_T)
#                   (δC_T)
#
#   where p is the pressure field δE/δC_T is the variational derivative of the 
#   adhesion energy with respect to the total cell density C_T, S_T is the source
#   term for the total cell density.
#
#   The mass flux is defined as follows:
#
#   J = -M ∇( δE )
#           (δC_T)
#
#
#   This file should be used as an importable class, that will take a tumor growth 
#   model as an argument, will be able to update the concentration fields of the tumor
#   based on the solid velocity field and the mass flux.
#   This class should be used in the following way:
#
#
#   <code>
#
#   model = TumorGrowthModel()
#   cell_dynamics = CellDynamics(model)
#   cell_dynamics.apply_cell_dynamics()
#
#   </code>
#
#
#   This will directly update the concentration fields of the tumor model.
#
#
# Author:
#   - Riley Jae McNamara
#
# Date:
#   - 2025-02-19
#
####################################################################################
####################################################################################


import numpy as np
from scipy.ndimage import laplace
from scipy.sparse import diags, eye
from scipy.sparse.linalg import cg

from src.models.cell_production import ProductionModel

class DynamicsModel:

    def __init__(self, model):
        self.model = model
        self.production_model = ProductionModel(self.model)

    
    def compute_cell_dynamics(self, state: dict = None) -> dict:
        """
        Compute the derivative dC/dt = -∇·(u C) - ∇·J for each cell type.
        """
        if state is None:
            state = self.model.get_state()

        # Extract cell concentrations
        C_S = state['C_S']
        C_P = state['C_P']
        C_D = state['C_D']
        C_N = state['C_N']

        # Compute solid velocity u
        ux, uy, uz = self._compute_solid_velocity(state)

        # Compute mass flux J
        Jx, Jy, Jz = self._compute_mass_current(state)

        # Compute divergence of J (same for all cell types)
        div_J = (np.gradient(Jx, self.model.dx, axis=0) +
                np.gradient(Jy, self.model.dx, axis=1) +
                np.gradient(Jz, self.model.dx, axis=2))

        # Compute derivatives for each cell type
        dC = {}
        for cell_type in ['C_S', 'C_P', 'C_D', 'C_N']:
            C = state[cell_type]
            # Compute flux u C
            uC_x = ux * C
            uC_y = uy * C
            uC_z = uz * C
            # Compute divergence of u C
            div_uC = (np.gradient(uC_x, self.model.dx, axis=0) +
                    np.gradient(uC_y, self.model.dx, axis=1) +
                    np.gradient(uC_z, self.model.dx, axis=2))
            # dC/dt = -∇·(u C) - ∇·J
            dC[cell_type] = -div_uC - div_J

        return dC
    

    def apply_cell_dynamics(self):
        """Update concentration fields based on dynamics."""
        state = self.model.get_state()
        dC = self.compute_cell_dynamics(state)
        dt = self.model.dt
        self.model.C_S += dt * dC['C_S']
        self.model.C_P += dt * dC['C_P']
        self.model.C_D += dt * dC['C_D']
        self.model.C_N += dt * dC['C_N']
        self.model._update_total_cell_density()


    def update_mass_flux(self):
        """Update the mass flux for the cell populations."""
        Jx, Jy, Jz = self._compute_mass_current()
        
        mass_flux = (np.gradient(Jx, self.model.dx, axis=0) + np.gradient(Jy, self.model.dx, axis=1) + np.gradient(Jz, self.model.dx, axis=2))

        self.model.C_S -= self.model.dt * mass_flux
        self.model.C_P -= self.model.dt * mass_flux
        self.model.C_D -= self.model.dt * mass_flux
        self.model.C_N -= self.model.dt * mass_flux


    def update_solid_velocity_flux(self):
        """
        Update the solid velocity flux for the cell populations.
        """
        ux, uy, uz = self._compute_solid_velocity()
        fields = [self.model.C_S, self.model.C_P, self.model.C_D, self.model.C_N]

        # Apply advection in each direction
        for field in fields:
            field -= self.model.dt * (
                np.gradient(field * ux, self.model.dx, axis=0) +
                np.gradient(field * uy, self.model.dx, axis=1) +
                np.gradient(field * uz, self.model.dx, axis=2)
            )


    def _compute_solid_velocity(self, state):
        C_T = state['C_S'] + state['C_P'] + state['C_D'] + state['C_N']
        
        # Calculate energy derivative with robust methods
        energy_deriv = self._compute_adhesion_energy_derivative(C_T)
        
        # More stable gradient calculation
        grad_C = np.gradient(C_T, self.model.dx)
        
        # Apply bounds to gradients
        grad_C = list(np.gradient(C_T, self.model.dx))
        for i in range(3):
            grad_C[i] = np.clip(grad_C[i], -100, 100)

        
        # Get pressure with numerical stability improvements
        pressure_field = self._compute_internal_pressure(state)
        
        # Calculate pressure gradient with safety checks
        grad_pressure = list(np.gradient(pressure_field, self.model.dx))
        
        # Apply bounds to pressure gradients
        for i in range(3):
            grad_pressure[i] = np.clip(grad_pressure[i], -1e4, 1e4)
        
        # Calculate velocity components with bounded operations
        ux = -(grad_pressure[0] + energy_deriv * grad_C[0])
        uy = -(grad_pressure[1] + energy_deriv * grad_C[1])
        uz = -(grad_pressure[2] + energy_deriv * grad_C[2])
        
        # Apply final velocity clipping
        max_velocity = 10.0  # Choose an appropriate maximum based on your model
        ux = np.clip(ux, -max_velocity, max_velocity)
        uy = np.clip(uy, -max_velocity, max_velocity)
        uz = np.clip(uz, -max_velocity, max_velocity)
        
        return ux, uy, uz


    def _compute_internal_pressure(self, state):
        # Validate dx
        if self.model.dx <= 0:
            raise ValueError(f"Grid spacing dx must be positive, got {self.model.dx}")
        self.dx = max(self.model.dx, 1e-6)

        # Get total cell density and source term
        C_T = state['C_S'] + state['C_P'] + state['C_D'] + state['C_N']
        if np.any(np.isnan(C_T)) or np.any(np.isinf(C_T)):
            print(f"Warning: NaN or inf in C_T: {C_T.max()}")

        d_prod = self.production_model.compute_cell_sources(state)
        S_T = d_prod['C_S'] + d_prod['C_P'] + d_prod['C_D'] + d_prod.get('C_N', 0 * C_T)
        S_T = np.clip(S_T, -100, 100)

        # Calculate energy derivative
        energy_deriv = self._compute_adhesion_energy_derivative(C_T)
        energy_deriv = np.clip(energy_deriv, -50, 50)

        # Compute divergence term
        grad_C_total = np.gradient(C_T, self.dx)
        divergence = 0
        laplace_C = np.zeros_like(C_T)
        for i in range(3):
            grad_i = np.gradient(C_T, self.dx, axis=i)
            second_deriv = np.gradient(grad_i, self.dx, axis=i)
            laplace_C += np.nan_to_num(second_deriv, nan=0.0)
        
        for i in range(3):
            grad_energy = np.gradient(energy_deriv, self.dx, axis=i)
            grad_energy = np.clip(grad_energy, -100, 100)
            divergence += grad_energy * grad_C_total[i]
        divergence += energy_deriv * laplace_C
        divergence = np.clip(divergence, -1e3, 1e3)

        # Right-hand side
        rhs = S_T - divergence
        rhs = np.clip(rhs, -1e4, 1e4)
        rhs_flat = rhs.ravel()
        if np.any(np.isnan(rhs_flat)):
            print("Warning: NaN in rhs_flat, replacing with 0")
            rhs_flat = np.nan_to_num(rhs_flat, nan=0.0)

        # Laplacian operator construction
        N = np.prod(self.model.grid_shape)
        dx2 = self.dx ** 2

        nx, ny, nz = self.model.grid_shape
        offset_x = 1
        offset_y = nx
        offset_z = nx * ny
        offsets = [-offset_z, -offset_y, -offset_x, 0, offset_x, offset_y, offset_z]

        coeff = 1.0 / dx2
        main_diag = -6.0 * coeff * np.ones(N)
        off_diag = 1.0 * coeff * np.ones(N)

        reg_factor = 1e-4  # Increased regularization
        A = diags([off_diag, off_diag, off_diag, main_diag, off_diag, off_diag, off_diag],
                offsets, shape=(N, N)) + reg_factor * eye(N)

        # Improved preconditioner
        diag_values = 1.0 / (main_diag + reg_factor)
        if np.any(diag_values <= 0):
            diag_values = np.abs(diag_values) + 1e-6
        M = diags([diag_values], [0])

        # Solve with CG
        p_flat, info = cg(A, rhs_flat, maxiter=5000, M=M)
        if info > 0:
            print(f"CG solver did not converge after {info} iterations")
            p_flat = np.zeros_like(rhs_flat)
        elif info < 0:
            print(f"CG solver failed with illegal input or breakdown: info {info}")
            p_flat = np.zeros_like(rhs_flat)

        if np.any(np.isnan(p_flat)) or np.any(np.isinf(p_flat)):
            print("Warning: NaN or inf in p_flat, using zeros")
            p_flat = np.zeros_like(rhs_flat)

        p = p_flat.reshape(self.model.grid_shape)
        p = np.clip(p, -1e4, 1e4)
        return p

    def _compute_adhesion_energy_derivative(self, C):
        params = self.model.params
        gamma = params['gamma']
        epsilon = params['epsilon']
        
        # Make sure epsilon isn't too small to avoid numerical issues
        epsilon = max(epsilon, 1e-6)
        
        # More robust clipping to physical bounds
        C = np.clip(C, 1e-10, 1 - 1e-10)  # Avoid exact boundaries
        
        # Calculate the derivative with bounded values
        f_prime = 0.5 * C * (1 - C) * (2 * C - 1)
        
        # More stable Laplacian calculation
        laplace_C = np.zeros_like(C)
        for i in range(3):
            second_deriv = np.gradient(np.gradient(C, self.model.dx, axis=i), 
                                    self.model.dx, axis=i)
            laplace_C += second_deriv
        
        # Calculate energy derivative with bounded terms
        energy_deriv = (gamma / epsilon) * f_prime - gamma * epsilon * laplace_C
        
        # Apply final clipping to prevent extreme values
        energy_deriv = np.clip(energy_deriv, -500, 500)
        
        return energy_deriv
    
    
    def _compute_mass_current(self, state=None):
        """
        Compute the mass flux J = -M ∇(δE/δC_T) based on total cell density.
        Returns tuple (Jx, Jy, Jz).
        """
        if state is None:
            C_T = self.model.C_T
        else:
            C_T = state['C_S'] + state['C_P'] + state['C_D'] + state['C_N']
        
        M = self.model.params["M"]
        energy_deriv = self._compute_adhesion_energy_derivative(C_T)
        grad_energy = np.gradient(energy_deriv, self.model.dx)
        Jx = -M * grad_energy[0]
        Jy = -M * grad_energy[1]
        Jz = -M * grad_energy[2]
        return Jx, Jy, Jz