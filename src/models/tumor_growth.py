#######################################################################################################
#######################################################################################################
#
#
#   3D Tumor Growth Simulation Model outlined in the README.md
#
#   This is the simulation class that will be used to directly run simmulations of the tumor growth model.
#   This is a 3 dimensional simulation, based on 5 scalar fields:

#   - Stem cell density/concentration (phi_H))
#   - Progenitor cell density/concentration (phi_P)
#   - Differentiated cell density/concentration (phi_D)
#   - Necrotic cell density/concentration (phi_N)
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
#   - cell volume fractions: The volume fractions of the different cell types
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
from typing import Any, Tuple

from src.utils.utils import experimental_params
from src.models.cell_production import ProductionModel
from src.models.cell_dynamics import DynamicsModel
from src.models.diffusion_dynamics import DiffusionDynamics
from src.models.initial_conditions import InitialCondition


class TumorGrowthModel:
    def __init__(self, grid_shape: Tuple = (50, 50, 50), dx: float = 0.1, dt: float = 0.001, params: dict = None, initial_conditions: InitialCondition = None, save_steps: int = 1):
        self.grid_shape = grid_shape
        self.dx = dx
        self.dt = dt
        self.params = params or experimental_params
        
        self.save_steps = save_steps
        
        # Initialize the fields using the initial conditions
        if initial_conditions is None:
            raise ValueError("Initial conditions must be provided.")
        
        self.initial_conditions = initial_conditions
        self._initialize_fields(initial_conditions)

        self.history = self._initialize_history()

        self.cell_production = ProductionModel(self)
        self.cell_dynamics = DynamicsModel(self)
        self.diffusion_dynamics = DiffusionDynamics(self)


    def run_simulation(self, steps=100):
        """Run the simulation for a given number of steps."""
        step = 0
        for _ in tqdm(range(steps), desc="Running Simulation"):
            step += 1
            self._update(step)


    def run_and_save_simulation(self, steps: int, name: str) -> None:
        """
        Run the simulation and save history data as a NumPy .npz file.
        
        Args:
            steps (int): Number of simulation steps to run.
            name (str): Base name for the output file.
        """
        self.run_simulation(steps=steps)
        history = self.get_history()
        history['Simulation Metadata'] = {'dx': self.dx, 'dt': self.dt, 'steps': steps, 'save_steps': self.save_steps}
        
        file_str = f"data/{name}_sim_data.npz"
        np.savez(file_str, **history)



    def get_history(self) -> dict:
        """
        This function will return the history of the simulation
        """
        return self.history


    def _update(self, step) -> None:
        """Perform one RK4 time step with stability checks."""
        state = (self.phi_H, self.phi_P, self.phi_D, self.phi_N, self.nutrient)

        # RK4 stages with clipping
        k1 = self._compute_derivatives(state)
        k1 = tuple(np.clip(k, -1e3, 1e3) for k in k1)  # Clip derivatives
        k2 = self._compute_derivatives(tuple(np.clip(s + self.dt / 2 * k, -1e3, 1e3) for s, k in zip(state, k1)))
        k2 = tuple(np.clip(k, -1e3, 1e3) for k in k2)
        k3 = self._compute_derivatives(tuple(np.clip(s + self.dt / 2 * k, -1e3, 1e3) for s, k in zip(state, k2)))
        k3 = tuple(np.clip(k, -1e3, 1e3) for k in k3)
        k4 = self._compute_derivatives(tuple(np.clip(s + self.dt * k, -1e3, 1e3) for s, k in zip(state, k3)))
        k4 = tuple(np.clip(k, -1e3, 1e3) for k in k4)

        # Update state with debugging
        new_state = []
        for i, field in enumerate(['phi_H', 'phi_P', 'phi_D', 'phi_N', 'nutrient']):
            update = state[i] + (self.dt / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
            if np.any(np.isnan(update)) or np.any(np.isinf(update)):
                print(f"Step {self.history['step'][-1] + 1}: {field} has NaN/inf. k1={np.max(np.abs(k1[i]))}, k2={np.max(np.abs(k2[i]))}, k3={np.max(np.abs(k3[i]))}, k4={np.max(np.abs(k4[i]))}")
            update = np.clip(update, 0 if field != 'nutrient' else -1e3, 1e3)  # Ensure physical bounds
            new_state.append(update)
            setattr(self, field, update)

        self._enforce_volume_fractions()
        if step % self.save_steps == 0:
            self._update_history()


    def _compute_derivatives(self, state):
        """Compute combined derivatives from all modules."""
        phi_H, phi_P, phi_D, phi_N, nutrient = state
        src_H, src_P, src_D, src_N = self.cell_production.compute_cell_sources(
            phi_H, phi_P, phi_D, phi_N, nutrient, self.n_H, self.n_P, self.n_D, self.params
        )
        dyn_S, dyn_P, dyn_D, dyn_N = self.cell_dynamics.compute_cell_dynamics(
            phi_H, phi_P, phi_D, phi_N, nutrient, self.dx, self.params
        )
        d_nutrient = self.diffusion_dynamics.compute_nutrient_diffusion(
            phi_H, phi_P, phi_D, phi_N, nutrient, self.params
        )

        # Check for instability
        derivatives = (src_H + dyn_S, src_P + dyn_P, src_D + dyn_D, src_N + dyn_N, d_nutrient)
        for i, field in enumerate(['phi_H', 'phi_P', 'phi_D', 'phi_N', 'nutrient']):
            if np.any(np.isnan(derivatives[i])) or np.any(np.isinf(derivatives[i])):
                print(f"Step {self.history['step'][-1] + 1}: {field} derivative has NaN/inf. "
                      f"src={np.max(np.abs([src_H, src_P, src_D, src_N][i])) if i < 4 else 0}, "
                      f"dyn={np.max(np.abs([dyn_S, dyn_P, dyn_D, dyn_N][i])) if i < 4 else 0}, "
                      f"d_nutrient={np.max(np.abs(d_nutrient)) if i == 4 else 0}")
        
        return derivatives


    def _initialize_history(self) -> dict:
        return {
            'step': [0], 'healthy cell volume fraction': [self.phi_H], 'progenitor cell volume fraction': [self.phi_P],
            'differentiated cell volume fraction': [self.phi_D], 'necrotic cell volume fraction': [self.phi_N]
        }


    def _update_history(self) -> None:
        """
        This function will update the history of the tumor growth model
        """

        self.history['step'].append(self.history['step'][-1] + 1 if self.history['step'] else 1)
        self.history['healthy cell volume fraction'].append(self.phi_H)
        self.history['progenitor cell volume fraction'].append(self.phi_P)
        self.history['differentiated cell volume fraction'].append(self.phi_D)
        self.history['necrotic cell volume fraction'].append(self.phi_N)
   

    def _initialize_fields(self, initial_conditions: InitialCondition) -> None:
        shape = self.grid_shape
        
        initial_conditions.initialize(self.params)
        
        self.phi_H = initial_conditions.phi_H
        self.phi_P = initial_conditions.phi_P
        self.phi_D = initial_conditions.phi_D
        self.phi_N = initial_conditions.phi_N
        self.nutrient = initial_conditions.nutrient
        self.n_H = initial_conditions.n_H
        self.n_P = initial_conditions.n_P
        self.n_D = initial_conditions.n_D
        self.phi_R = initial_conditions.phi_R
        
        # Enforce the global volume fraction constraint
        self._enforce_volume_fractions()


    def _enforce_volume_fractions(self) -> None:
        """
        Enforce volume fractions using vectorized operations
        """
        phi_S = self.params.get('phi_S', 1)
        phi_T = self.phi_H + self.phi_P + self.phi_D + self.phi_N
        
        # Create a mask and apply scaling in one operation
        mask = phi_T > phi_S
        if np.any(mask):
            scaling = np.ones_like(phi_T)
            scaling[mask] = phi_S / phi_T[mask]
            
            # Vectorized scaling of all cell types at once
            self.phi_H *= scaling
            self.phi_P *= scaling
            self.phi_D *= scaling
            self.phi_N *= scaling
            
            # Update phi_T after rescaling
            phi_T = self.phi_H + self.phi_P + self.phi_D + self.phi_N
        
        # Define the host region fraction
        self.phi_R = phi_S - phi_T



