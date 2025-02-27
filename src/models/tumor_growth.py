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
import numba as nb
from multiprocessing import Pool
from functools import partial

from src.utils.utils import experimental_params
from src.models.cell_production import ProductionModel
from src.models.cell_dynamics import DynamicsModel
from src.models.diffusion_dynamics import DiffusionDynamics


class TumorGrowthModel:
    def __init__(self, grid_shape=(100, 100, 100), dx=0.1, dt=0.001, params=None, initial_conditions=None):
        self.grid_shape = grid_shape
        self.dx = dx
        self.dt = dt
        self.params = params or experimental_params
        self._initialize_fields(initial_conditions)
        self.history = self._initialize_history()
        
        self.cell_production = ProductionModel(self)
        self.cell_dynamics = DynamicsModel(self)
        self.diffusion_dynamics = DiffusionDynamics(self)


    def _update(self):
        """Perform one RK4 time step with stability checks."""
        state = (self.C_S, self.C_P, self.C_D, self.C_N, self.nutrient)

        # RK4 stages with clipping
        k1 = self.compute_derivatives(state)
        k1 = tuple(np.clip(k, -1e3, 1e3) for k in k1)  # Clip derivatives
        k2 = self.compute_derivatives(tuple(np.clip(s + self.dt / 2 * k, -1e3, 1e3) for s, k in zip(state, k1)))
        k2 = tuple(np.clip(k, -1e3, 1e3) for k in k2)
        k3 = self.compute_derivatives(tuple(np.clip(s + self.dt / 2 * k, -1e3, 1e3) for s, k in zip(state, k2)))
        k3 = tuple(np.clip(k, -1e3, 1e3) for k in k3)
        k4 = self.compute_derivatives(tuple(np.clip(s + self.dt * k, -1e3, 1e3) for s, k in zip(state, k3)))
        k4 = tuple(np.clip(k, -1e3, 1e3) for k in k4)

        # Update state with debugging
        new_state = []
        for i, field in enumerate(['C_S', 'C_P', 'C_D', 'C_N', 'nutrient']):
            update = state[i] + (self.dt / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
            if np.any(np.isnan(update)) or np.any(np.isinf(update)):
                print(f"Step {self.history['step'][-1] + 1}: {field} has NaN/inf. k1={np.max(np.abs(k1[i]))}, k2={np.max(np.abs(k2[i]))}, k3={np.max(np.abs(k3[i]))}, k4={np.max(np.abs(k4[i]))}")
            update = np.clip(update, 0 if field != 'nutrient' else -1e3, 1e3)  # Ensure physical bounds
            new_state.append(update)
            setattr(self, field, update)

        self._enforce_volume_fractions()
        self._update_history()


    def compute_derivatives(self, state):
        """Compute combined derivatives from all modules."""
        C_S, C_P, C_D, C_N, nutrient = state
        src_S, src_P, src_D, src_N = self.cell_production.compute_cell_sources(
            C_S, C_P, C_D, C_N, nutrient, self.n_S, self.n_P, self.n_D, self.params
        )
        dyn_S, dyn_P, dyn_D, dyn_N = self.cell_dynamics.compute_cell_dynamics(
            C_S, C_P, C_D, C_N, nutrient, self.dx, self.params
        )
        d_nutrient = self.diffusion_dynamics.compute_nutrient_diffusion(
            C_S, C_P, C_D, C_N, nutrient, self.params
        )

        # Check for instability
        derivatives = (src_S + dyn_S, src_P + dyn_P, src_D + dyn_D, src_N + dyn_N, d_nutrient)
        for i, field in enumerate(['C_S', 'C_P', 'C_D', 'C_N', 'nutrient']):
            if np.any(np.isnan(derivatives[i])) or np.any(np.isinf(derivatives[i])):
                print(f"Step {self.history['step'][-1] + 1}: {field} derivative has NaN/inf. "
                      f"src={np.max(np.abs([src_S, src_P, src_D, src_N][i])) if i < 4 else 0}, "
                      f"dyn={np.max(np.abs([dyn_S, dyn_P, dyn_D, dyn_N][i])) if i < 4 else 0}, "
                      f"d_nutrient={np.max(np.abs(d_nutrient)) if i == 4 else 0}")
        
        return derivatives


    def run_simulation(self, steps=100):
        """Run the simulation for a given number of steps."""
        for _ in tqdm(range(steps), desc="Running Simulation"):
            self._update()


    def get_history(self) -> dict:
        """
        This function will return the history of the simulation
        """
        return self.history


    def _initialize_history(self) -> dict:
        return {
            'step': [0], 'stem cell volume fraction': [self.C_S], 'progenitor cell volume fraction': [self.C_P],
            'differentiated cell volume fraction': [self.C_D], 'necrotic cell volume fraction': [self.C_N],
            'stem cell volume': [np.sum(self.C_S) * self.dx**3], 'progenitor cell volume': [np.sum(self.C_P) * self.dx**3],
            'differentiated cell volume': [np.sum(self.C_D) * self.dx**3], 'necrotic cell volume': [np.sum(self.C_N) * self.dx**3],
            'radius': [self._calculate_radius()]
        }


    def _update_history(self) -> None:
        """
        This function will update the history of the tumor growth model
        """

        self.history['step'].append(self.history['step'][-1] + 1 if self.history['step'] else 1)
        self.history['stem cell volume fraction'].append(self.C_S)
        self.history['progenitor cell volume fraction'].append(self.C_P)
        self.history['differentiated cell volume fraction'].append(self.C_D)
        self.history['necrotic cell volume fraction'].append(self.C_N)
        self.history['stem cell volume'].append(self._compute_total_cell_volume(self.C_S))
        self.history['progenitor cell volume'].append(self._compute_total_cell_volume(self.C_P))
        self.history['differentiated cell volume'].append(self._compute_total_cell_volume(self.C_D))
        self.history['necrotic cell volume'].append(self._compute_total_cell_volume(self.C_N))
        self.history['radius'].append(self._calculate_radius())


    def _initialize_fields(self, initial_conditions: Any = None) -> None:
        shape = self.grid_shape
        
        # Initialize all fields at once
        self.C_S = np.zeros(shape)
        self.C_P = np.zeros(shape)
        self.C_D = np.zeros(shape)
        self.C_N = np.zeros(shape)
        self.nutrient = 0.001 * np.ones(shape)
        
        # Use broadcasting for parameter fields
        self.n_S = self.params['p_0'] * np.ones(shape)
        self.n_P = self.params['p_1'] * np.ones(shape)
        self.n_D = np.ones(shape)
        self.phi_H = np.zeros(shape)
        
        # Pre-compute distance grid for initialization and radius calculation
        center = np.array([s // 2 for s in shape])
        x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
        dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
        self._distance_grid = dist_from_center  # Store for reuse
        
        # Use boolean masking for more efficient initialization
        self.C_S[dist_from_center <= 3] = 0.2
        self.C_D[dist_from_center <= 3] = 0.5
        
        # Enforce the global volume fraction constraint
        self._enforce_volume_fractions()


    def _enforce_volume_fractions(self) -> None:
        """
        Enforce volume fractions using vectorized operations
        """
        phi_S = self.params.get('phi_S', 1)
        phi_T = self.C_S + self.C_P + self.C_D + self.C_N
        
        # Create a mask and apply scaling in one operation
        mask = phi_T > phi_S
        if np.any(mask):
            scaling = np.ones_like(phi_T)
            scaling[mask] = phi_S / phi_T[mask]
            
            # Vectorized scaling of all cell types at once
            self.C_S *= scaling
            self.C_P *= scaling
            self.C_D *= scaling
            self.C_N *= scaling
            
            # Update phi_T after rescaling
            phi_T = self.C_S + self.C_P + self.C_D + self.C_N
        
        # Define the host region fraction
        self.phi_H = phi_S - phi_T


    def _calculate_radius(self) -> float:
        """
        Calculate tumor radius more efficiently
        """
        # Pre-compute distance grid once during initialization
        if not hasattr(self, '_distance_grid'):
            center = np.array([s // 2 for s in self.grid_shape])
            x, y, z = np.ogrid[:self.grid_shape[0], :self.grid_shape[1], :self.grid_shape[2]]
            self._distance_grid = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
        
        # Use the pre-computed distance grid
        if np.any(self.C_S > 0):
            return np.max(self._distance_grid[self.C_S > 0])
        else:
            return 0.0
    

    def _compute_total_cell_volume(self, field: np.ndarray) -> float:
        """
        Compute the total cell volume using vectorized operations
        """
        return np.sum(field) * self.dx**3


