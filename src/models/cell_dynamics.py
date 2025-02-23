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
from scipy.sparse import diags
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
        energy_deriv = self._compute_adhesion_energy_derivative(C_T)
        grad_C = np.gradient(C_T, self.model.dx)
        pressure_field = self._compute_internal_pressure(state)
        grad_pressure = np.gradient(pressure_field, self.model.dx)
        ux = -(grad_pressure[0] + energy_deriv * grad_C[0])
        uy = -(grad_pressure[1] + energy_deriv * grad_C[1])
        uz = -(grad_pressure[2] + energy_deriv * grad_C[2])
        return ux, uy, uz


    def _compute_internal_pressure(self, state):
        C_T = state['C_S'] + state['C_P'] + state['C_D'] + state['C_N']
        d_prod = self.model.cell_production.compute_cell_sources(state)
        S_T = d_prod['C_S'] + d_prod['C_P'] + d_prod['C_D'] + d_prod.get('C_N', 0 * C_T)
        S_T = np.clip(S_T, -1000, 1000)  # Cap source term
        energy_deriv = self._compute_adhesion_energy_derivative(C_T)
        grad_C_total = np.gradient(C_T, self.model.dx)
        divergence = 0
        laplace_C = laplace(C_T, mode='constant')
        for i in range(3):
            grad_energy = np.gradient(energy_deriv, self.model.dx, axis=i)
            divergence += grad_energy * grad_C_total[i]
        divergence += energy_deriv * laplace_C
        rhs = S_T - divergence
        rhs = np.clip(rhs, -1e5, 1e5)
        print(f"Max RHS: {np.max(np.abs(rhs))}")

        # Construct Laplacian operator
        N = np.prod(self.model.grid_shape)
        dx2 = self.model.dx ** 2
        if dx2 == 0:
            raise ValueError("dx is zero, cannot construct Laplacian")
        coeff = 1 / dx2  # Scale factor
        main_diag = -6 * coeff * np.ones(N)
        off_diag = coeff * np.ones(N-1)
        offsets = [
            -self.model.grid_shape[1] * self.model.grid_shape[2],  # -z
            -self.model.grid_shape[2],                             # -y
            -1,                                                   # -x
            0,                                                    # center
            1,                                                    # +x
            self.model.grid_shape[2],                             # +y
            self.model.grid_shape[1] * self.model.grid_shape[2]   # +z
        ]
        A = diags([off_diag, off_diag, off_diag, main_diag, off_diag, off_diag, off_diag],
                offsets, shape=(N, N))

        # Solve using conjugate gradient
        rhs_flat = rhs.ravel()
        p_flat, info = cg(A, rhs_flat, tol=1e-3, maxiter=500)
        if info > 0:
            print(f"CG solver did not converge after {info} iterations")
        elif info < 0:
            print(f"CG solver failed with illegal input or breakdown: info {info}")
        p = p_flat.reshape(self.model.grid_shape)
        print(f"Max Pressure: {np.max(np.abs(p))}")
        return p


    def _compute_adhesion_energy_derivative(self, C):
        params = self.model.params
        gamma = params['gamma']
        epsilon = params['epsilon']
        C = np.clip(C, 0, 1)  # Ensure physical bounds
        f_prime = 0.5 * C * (1 - C) * (2 * C - 1)
        laplace_C = laplace(C)
        energy_deriv = (gamma / epsilon) * f_prime - gamma * epsilon * laplace_C
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