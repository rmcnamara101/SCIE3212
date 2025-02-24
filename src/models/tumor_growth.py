#######################################################################################################
#######################################################################################################
#
#
#   3D Tumor Growth Simulation Model outlined in the README.md
#
#   This is the simulation class that will be used to directly run simmulations of the tumor growth model.
#   This is a 3 dimensional simulation, based on 5 scalar fields:

#   - Stem cell density/concentration (C_S)
#   - Progenitor cell density/concentration (C_P)
#   - Differentiated cell density/concentration (C_D)
#   - Necrotic cell density/concentration (C_N)
#   - Nutrient concentration (n)
#
#   The simulation solves the partial differential equations (PDEs) that are defined in the README.md
#
#   To run a simulation, the user must first initialize the TumorGrowthModel class, with the following arguments:

#   - grid_shape: The shape of the grid to run the simulation on
#   - dx: The spatial resolution of the grid
#   - dt: The time step of the simulation
#   - params: The parameters of the simulation
#   - initial_conditions: The initial conditions of the simulation (not implemented yet)
#
#   The user can then call the run_simulation method, with the following arguments:
#   - steps: The number of steps to run the simulation for
#
#   The user can then access the history of the simulation, which will contain the following information:
#   - step: The step number
#   - cell concentrations: The concentration of all cell types at each step
#   - cell volumes: The volume of all cell types at each step
#   - radius: The radius of the tumor at each step
#   
#
#   This class should be used in the following way:
#
#   <code>
#
#   model = TumorGrowthModel()
#   model.run_simulation(steps=100)
#   simulation_history = model.get_history()
#
#   </code>
#
#   The user can then access the history of the simulation by calling the get_history method.
#
#
#
# Author:
#   - Riley Jae McNamara
#
# Date:
#   - 2025-02-19
#
#
#
#######################################################################################################
#######################################################################################################

import numpy as np
from tqdm import tqdm
from typing import Tuple, Any
import matplotlib.pyplot as plt

from src.utils.utils import experimental_params
from src.models.cell_production import ProductionModel
from src.models.cell_dynamics import DynamicsModel
from src.models.diffusion_dynamics import DiffusionDynamics

class TumorGrowthModel:

    def __init__(self, grid_shape: Tuple[int, int, int] = (50, 50, 50), dx: float = 0.1, dt: float = 0.001, params: dict = None, initial_conditions: Any = None) -> None:
        self.grid_shape = grid_shape
        self.dx = dx
        self.dt = dt
        self.params = params or experimental_params
        self._initialize_fields(initial_conditions)
        self.history = self._initialize_history()
        
        self.cell_production = ProductionModel(self)
        self.cell_dynamics = DynamicsModel(self)
        self.diffusion_dynamics = DiffusionDynamics(self)


    def get_state(self) -> dict:
        """
        Return a copy of the current state.
        """
        return {
            'C_S': self.C_S.copy(),
            'C_P': self.C_P.copy(),
            'C_D': self.C_D.copy(),
            'C_N': self.C_N.copy(),
            'nutrient': self.nutrient.copy()
        }


    def set_state(self, state: dict) -> None:
        """
        Set the model state from a state dictionary and update the total cell density.
        """
        self.C_S = state['C_S']
        self.C_P = state['C_P']
        self.C_D = state['C_D']
        self.C_N = state['C_N']
        self.nutrient = state['nutrient']
        self.C_S = np.clip(self.C_S, 0, 1)
        self.C_P = np.clip(self.C_P, 0, 1)
        self.C_D = np.clip(self.C_D, 0, 1)
        self.C_N = np.clip(self.C_N, 0, 1)
        self.nutrient = np.clip(self.nutrient, 0, None)  # Nutrient can't be negative
        self._update_total_cell_density()


    def add_state(self, state: dict, deriv: dict, factor: float) -> dict:
        """
        Return a new state that is state + factor * deriv.
        """
        new_state = {}
        for key in state:
            new_state[key] = state[key] + factor * deriv[key]
        return new_state


    def compute_derivatives(self, state: dict) -> dict:
        """
        Compute the derivatives for all fields by summing contributions from cell production,
        cell dynamics, and nutrient diffusion.
        """
        # Each module has been refactored to compute (not apply) its rate of change.
        d_prod = self.cell_production.compute_cell_sources(state)
        #d_prod = {'C_S': 0, 'C_P': 0, 'C_D': 0, 'C_N': 0}
        d_dyn = self.cell_dynamics.compute_cell_dynamics(state)
        #d_dyn = {'C_S': 0, 'C_P': 0, 'C_D': 0, 'C_N': 0}
        d_diff = self.diffusion_dynamics.compute_nutrient_diffusion(state)

        deriv = {
            'C_S': d_prod['C_S'] + d_dyn['C_S'],
            'C_P': d_prod['C_P'] + d_dyn['C_P'],
            'C_D': d_prod['C_D'] + d_dyn['C_D'],
            # If necrotic cell source is desired, include it. (It was commented out before.)
            'C_N': d_prod.get('C_N', 0 * state['C_N']) + d_dyn.get('C_N', 0 * state['C_N']),
            'nutrient': d_diff  # Diffusion module returns the nutrient derivative.
        }
        return deriv


    def _update(self):
        state = self.get_state()
        print(f"Initial state: Max C_S: {np.max(state['C_S'])}, Max C_P: {np.max(state['C_P'])}, "
            f"Max C_D: {np.max(state['C_D'])}, Max C_N: {np.max(state['C_N'])}, Max nutrient: {np.max(state['nutrient'])}")
        dt = self.dt
        k1 = self.compute_derivatives(state)
        for key, val in k1.items():
            if np.any(np.isnan(val)):
                print(f"NaN detected in k1[{key}]")
        k2 = self.compute_derivatives(self.add_state(state, k1, dt/2))
        k3 = self.compute_derivatives(self.add_state(state, k2, dt/2))
        k4 = self.compute_derivatives(self.add_state(state, k3, dt))
        new_state = {}
        for key in state:
            new_state[key] = state[key] + dt/6 * (k1[key] + 2*k2[key] + 2*k3[key] + k4[key])
        self.set_state(new_state)
        self.enforce_volume_fractions()
        self._update_total_cell_density()
        print(f"Step {self.history['step'][-1]}: "
            f"Max C_S: {np.max(self.C_S):.3f}, Max C_P: {np.max(self.C_P):.3f}, "
            f"Max C_D: {np.max(self.C_D):.3f}, Max C_N: {np.max(self.C_N):.3f}, "
            f"Max nutrient: {np.max(self.nutrient):.3f}, Radius: {self._calculate_radius():.3f}")
        self._update_history()


    def plot_slice_C(self):
        """
        Plot a slice of the necrotic cell concentration (C_N) at the current state.
        """
        slice_index = self.grid_shape[0] // 2  # Take a slice in the middle of the first dimension
        plt.imshow(self.C_S[slice_index, :, :], cmap='hot', interpolation='nearest')
        plt.colorbar(label='Necrotic Cell Concentration (C_N)')
        plt.title('Slice of Necrotic Cell Concentration (C_N)')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.show()


    def run_simulation(self, steps: int = 100) -> Tuple[dict]:
        """
        This is the main simulation loop
        """
        # run the simulation
        for step in tqdm(range(steps), desc="Running Simulation"):
            self._update()
            self._update_history()


    def get_history(self) -> dict:
        """
        This function will return the history of the simulation
        """
        return self.history


    def _initialize_history(self) -> dict:
        return {
            'step': [0], 'stem cell concentration': [self.C_S], 'progenitor cell concentration': [self.C_P],
            'differentiated cell concentration': [self.C_D], 'necrotic cell concentration': [self.C_N],
            'total cell concentration': [self.C_T], 'stem cell volume': [np.sum(self.C_S) * self.dx**3], 'progenitor cell volume': [np.sum(self.C_P) * self.dx**3],
            'differentiated cell volume': [np.sum(self.C_D) * self.dx**3], 'necrotic cell volume': [np.sum(self.C_N) * self.dx**3],
            'total cell volume': [np.sum(self.C_T) * self.dx**3], 'radius': [self._calculate_radius()]
        }


    def _update_history(self) -> None:
        """
        This function will update the history of the tumor growth model
        """

        self.history['step'].append(self.history['step'][-1] + 1 if self.history['step'] else 1)
        self.history['stem cell concentration'].append(self.C_S)
        self.history['progenitor cell concentration'].append(self.C_P)
        self.history['differentiated cell concentration'].append(self.C_D)
        self.history['necrotic cell concentration'].append(self.C_N)
        self.history['total cell concentration'].append(self.C_T)
        self.history['stem cell volume'].append(self._compute_total_cell_volume(self.C_S))
        self.history['progenitor cell volume'].append(self._compute_total_cell_volume(self.C_P))
        self.history['differentiated cell volume'].append(self._compute_total_cell_volume(self.C_D))
        self.history['necrotic cell volume'].append(self._compute_total_cell_volume(self.C_N))
        self.history['total cell volume'].append(self._compute_total_cell_volume(self.C_T))
        self.history['radius'].append(self._calculate_radius())


    def _initialize_fields(self, initial_conditions: Any = None) -> None:
        shape = self.grid_shape
        self.C_S = np.zeros(shape)
        self.C_P = np.zeros(shape)
        self.C_D = np.zeros(shape)
        self.C_N = np.zeros(shape)
        self.C_T = np.zeros(shape)
        self.nutrient = 0.001 * np.ones(shape)
        self.n_S = self.params['p_0'] * np.ones(shape)
        self.n_P = self.params['p_1'] * np.ones(shape)
        self.n_D = np.ones(shape)
        
        # Initialize the host region field (ϕ_H) as zeros; it will be computed after tumor initialization.
        self.phi_H = np.zeros(shape)

        # Create a small spherical initial tumor (only stem cells are nonzero)
        center = np.array([s // 2 for s in shape])
        radius = 3  # Initial radius of tumor sphere
        x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
        dist_from_center = np.sqrt((x - center[0])**2 +
                                   (y - center[1])**2 +
                                   (z - center[2])**2)
        self.C_S[dist_from_center <= radius] = 0.5
        self._update_total_cell_density()
        
        # Enforce the global volume fraction constraint at t=0
        self.enforce_volume_fractions()


    def enforce_volume_fractions(self) -> None:
        """
        Enforce the global volume fraction constraint.
        We assume the total solid volume fraction ϕ_S is constant (e.g., 0.8).
        At each grid point, the tumor cell fraction is given by
            ϕ_T = C_S + C_P + C_D + C_N.
        If ϕ_T > ϕ_S, we rescale the tumor fields so that ϕ_T becomes ϕ_S.
        Then, the host region fraction is defined as:
            ϕ_H = ϕ_S - ϕ_T.
        """
        phi_S = self.params.get('phi_S', 0.8)  # Total solid fraction (can be set in your params)
        # Compute tumor cell fraction at every voxel
        phi_T = self.C_S + self.C_P + self.C_D + self.C_N

        # Identify voxels where the tumor fraction exceeds the available solid fraction.
        mask = phi_T > phi_S
        if np.any(mask):
            # Compute a scaling factor to bring phi_T down to phi_S
            scaling = np.ones_like(phi_T)
            scaling[mask] = phi_S / phi_T[mask]
            self.C_S[mask] *= scaling[mask]
            self.C_P[mask] *= scaling[mask]
            self.C_D[mask] *= scaling[mask]
            self.C_N[mask] *= scaling[mask]
            # Update phi_T after rescaling
            phi_T = self.C_S + self.C_P + self.C_D + self.C_N

        # Define the host region fraction: what remains of the solid region.
        self.phi_H = phi_S - phi_T


    def _calculate_radius(self) -> float:
        """
        This function will calculate the radius of the tumor
        """
        # locate the center of the tumor
        center = np.array([s // 2 for s in self.grid_shape])
        # create a grid of the same shape as the tumor
        x, y, z = np.ogrid[:self.grid_shape[0], :self.grid_shape[1], :self.grid_shape[2]]
        # calculate the distance from the center of the tumor to each voxel
        dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)    

        # Check if there are any positive values in C_S
        if np.any(self.C_S > 0):
            return np.max(dist_from_center[self.C_S > 0])
        else:
            return 0.0  # Return 0 if there are no tumor cells
    

    def _update_total_cell_density(self) -> None:
        """
        This function will update the total cell density
        """
        self.C_T = self.C_S + self.C_P + self.C_D + self.C_N
        

    def _compute_total_cell_volume(self, field: np.ndarray) -> float:
        """
        This function will compute the total cell volume
        """
        total_volume = 0.0
        dx = self.dx  # Assuming dx is defined in the model
        for value in field.flatten():
            if value > 0:
                total_volume += dx**3
        return total_volume
