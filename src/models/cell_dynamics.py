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
#   J = -M ∇C_T ( δE )
#               (δC_T)
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

from src.models.cell_production import ProductionModel

class DynamicsModel:
    def __init__(self, model):
        self.model = model
        self.production_model = ProductionModel(self.model)

    def apply_cell_dynamics(self):
        """Update the cell dynamics for the cell populations."""
        self.update_solid_velocity_flux()
        self.update_mass_flux()

    def update_mass_flux(self):
        """Update the mass flux for the cell populations."""
        Jx, Jy, Jz = self._compute_mass_current()
        
        mass_flux = (np.gradient(Jx, self.model.dx, axis=0) + np.gradient(Jy, self.model.dx, axis=1) + np.gradient(Jz, self.model.dx, axis=2))

        self.model.C_S -= self.model.dt * mass_flux
        self.model.C_P -= self.model.dt * mass_flux
        self.model.C_D -= self.model.dt * mass_flux
        self.model.C_N -= self.model.dt * mass_flux
        self.model.update_total_cell_density()

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

        self.model.update_total_cell_density()

    def _compute_solid_velocity(self):
        """
        Compute the solid velocity field u_s based on:
            u_s = - (∇p + (δE/δC_T) ∇C_T)
        Returns tuple of velocity components (u_x, u_y, u_z)
        """
        params = self.model.params
        C_T = self.model.C_T
        energy_deriv = self._compute_adhesion_energy_derivative()
        grad_C = np.gradient(C_T)
        pressure_field = self._compute_internal_pressure()
        grad_pressure = np.gradient(pressure_field)

        # calculate solid velocity components
        ux = -(grad_pressure[0] + energy_deriv * grad_C[0])
        uy = -(grad_pressure[1] + energy_deriv * grad_C[1])
        uz = -(grad_pressure[2] + energy_deriv * grad_C[2])

        return ux, uy, uz

    def _compute_internal_pressure(self):
        """
        Compute the internal pressure field by solving the Poisson equation:
            ∇²p = S_T - ∇·((δE/δC_T) ∇C_T)
        with S_T = λ_S n C_S + λ_P n C_P + γ_N C_N
        
        Returns:
            numpy.ndarray: The pressure field solution
        """
        self.model.update_total_cell_density()
        
        # Calculate source term
        S_T = self.production_model._compute_src_T()
        
        # Calculate energy derivative
        energy_deriv = self._compute_adhesion_energy_derivative()
        
        # Get gradients of total concentration
        grad_C_total = np.gradient(self.model.C_T)
        
        # Calculate divergence term
        divergence = 0
        for i in range(3):  # For each dimension
            # Calculate gradient of energy derivative
            grad_energy = np.gradient(energy_deriv, self.model.dx, axis=i)
            # Multiply with gradient of total density and add to divergence
            divergence += grad_energy * grad_C_total[i]
        
        # Right-hand side of Poisson equation
        rhs = S_T - divergence
        
        # Solve Poisson equation using iterative method
        # Initialize pressure field
        p = np.zeros_like(rhs)
        
        # Parameters for iterative solver
        max_iter = 1000
        tolerance = 1e-6
        dx2 = self.model.dx * self.model.dx
        
        # Jacobi iteration to solve ∇²p = rhs
        for iter in range(max_iter):
            p_old = p.copy()
            
            # Update pressure using discrete Laplacian
            p[1:-1, 1:-1, 1:-1] = (
                (p_old[2:, 1:-1, 1:-1] + p_old[:-2, 1:-1, 1:-1] +
                p_old[1:-1, 2:, 1:-1] + p_old[1:-1, :-2, 1:-1] +
                p_old[1:-1, 1:-1, 2:] + p_old[1:-1, 1:-1, :-2] -
                6 * p_old[1:-1, 1:-1, 1:-1]) / dx2 - rhs[1:-1, 1:-1, 1:-1]
            ) * (dx2 / 6)
            
            # Apply boundary conditions (here using Neumann BC)
            p[0, :, :] = p[1, :, :]
            p[-1, :, :] = p[-2, :, :]
            p[:, 0, :] = p[:, 1, :]
            p[:, -1, :] = p[:, -2, :]
            p[:, :, 0] = p[:, :, 1]
            p[:, :, -1] = p[:, :, -2]
            
            # Check convergence
            error = np.max(np.abs(p - p_old))
            if error < tolerance:
                break
        
        return p

    def _compute_adhesion_energy_derivative(self):
        """
        Compute the adhesion energy derivative for the cell populations.
        """
        
        params = self.model.params
        gamma = params['gamma']
        epsilon = params['epsilon']
        self.model.update_total_cell_density()
        C = self.model.C_T

        # f'(C_T) for f(C_T)=1/4 * C_T^2 (1-C_T)^2 is 0.5 * C_total * (1 - C_total) * (2 * C_total - 1)
        energy_deriv = (gamma/epsilon) * (0.5 * C * (1 - C) * (2 * C - 1) - epsilon**2 * laplace(C))
        return energy_deriv
    
    def _compute_mass_current(self): 
        """Compute the mass flux for the cell populations."""
        params = self.model.params
        C_T = self.model.C_T
        M = params['M']

        # compute the mass flux term
        energy_deriv = self._compute_adhesion_energy_derivative()
        grad_C = np.gradient(C_T)

        Jx = -M * grad_C[0] * energy_deriv
        Jy = -M * grad_C[1] * energy_deriv
        Jz = -M * grad_C[2] * energy_deriv

        return Jx, Jy, Jz
    