#######################################################################################################
#######################################################################################################
#
#
#   3D Tumor Growth Simulation Model outlined in files/mathematical model/SCIE3121_model.md
#
#   This is the simulation class that will be used to directly run simmulations of the tumor growth model.
#   This is a 3 dimensional simulation, based on 4 scalar fields:
#
#   - Healthy cell volume fraction (phi_H))
#   - Diseased cell volume fraction (phi_D)
#   - Necrotic cell volume fraction (phi_N)
#   - Nutrient concentration (n)
#
#   The simulation solves the partial differential equations (PDEs) 
#
#   To run a simulation, the user must first initialize the TumorGrowthModel class, with the following arguments:
#
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
#   - 2025-03-5
#
#
#
#######################################################################################################
#######################################################################################################

import numpy as np
from typing import Tuple
from scipy.integrate import solve_ivp

from src.utils.utils import SCIE3121_params

from src.models.cell_production import SCIE3121_MODEL as cell_production
from src.models.cell_dynamics import SCIE3121_DynamicsModel as cell_dynamics
from src.models.diffusion_dynamics import SCIE3121DiffusionModel as diffusion_dynamics
from src.models.initial_conditions import InitialCondition

# define base model
from src.models.tumor_growth import TumorGrowthModel

try:
    print("Attempting to import C++ module...")
    import os
    import sys
    module_dir = os.path.dirname(__file__)
    print(f"Adding {module_dir} to Python path")
    sys.path.append(module_dir)
    print(f"Current directory: {os.getcwd()}")
    print(f"Python path: {os.environ.get('PYTHONPATH', '')}")
    module_path = os.path.join(module_dir, "cpp_simulation.so")
    print(f"Looking for module at: {module_path}")
    if os.path.exists(module_path):
        print(f"Module file exists")
        import cpp_simulation
        SimulationCore = cpp_simulation.SimulationCore
        print("Successfully imported C++ module")
        USE_CPP = True
    else:
        print(f"Module file not found")
        raise ImportError(f"Module file not found at {module_path}")
except ImportError as e:
    USE_CPP = False
    print(f"Warning: C++ implementation not available, using Python version. Error: {e}")

class SCIE3121_MODEL(TumorGrowthModel):

    def __init__(self, grid_shape: Tuple = (50, 50, 50), dx: float = 0.1, dt: float = 0.001, initial_conditions: InitialCondition = None, params: dict = None, save_steps: int = 1):
        
        # Initialize base model
        super().__init__(grid_shape, dx, dt, params, initial_conditions, save_steps)
        
        # override base model parameters
        self.params = params or SCIE3121_params

        # change the cell production and dynamics model
        self.cell_production = cell_production(self)
        self.cell_dynamics = cell_dynamics(self)
        self.diffusion_dynamics = diffusion_dynamics(self)

        if USE_CPP:
            # Create dx arrays for x and y directions
            dx_x = np.full_like(self.phi_H, self.dx)
            dx_y = np.full_like(self.phi_H, self.dx)
            
            # Convert n_H and n_D to arrays
            n_H_array = np.full_like(self.phi_H, self.n_H)
            n_D_array = np.full_like(self.phi_H, self.n_D)
            
            self.cpp_sim = SimulationCore(
                self.phi_H,
                self.phi_D,
                self.phi_N,
                self.nutrient,
                n_H_array,
                n_D_array,
                self.dt,
                self.params['epsilon'],
                self.params
            )

    def _update(self, step):
        if USE_CPP:
            # Step the simulation
            self.cpp_sim.step_rk4()
            
            # Update state and history at save steps
            if step % self.save_steps == 0:
                # Get current state from C++
                self.phi_H, self.phi_D, self.phi_N, self.nutrient = self.cpp_sim.get_state()
                self._enforce_volume_fractions()
                
                # Update history with correct keys
                self.history['step'].append(step)
                self.history['healthy cell volume fraction'].append(self.phi_H.copy())
                self.history['diseased cell volume fraction'].append(self.phi_D.copy())
                self.history['necrotic cell volume fraction'].append(self.phi_N.copy())
                self.history['nutrient'].append(self.nutrient.copy())
        else:
            """
            Update the fields for a single step of the simulation using RK4 method.
            """
            state = (self.phi_H, self.phi_D, self.phi_N, self.nutrient)
            
            # RK4 implementation
            k1 = self._compute_derivatives(state)
            
            # Calculate intermediate states
            state2 = tuple(s + self.dt / 2 * k for s, k in zip(state, k1))
            k2 = self._compute_derivatives(state2)
            
            state3 = tuple(s + self.dt / 2 * k for s, k in zip(state, k2))
            k3 = self._compute_derivatives(state3)
            
            state4 = tuple(s + self.dt * k for s, k in zip(state, k3))
            k4 = self._compute_derivatives(state4)
            
            # Update state with weighted average of derivatives
            field_names = ['phi_H', 'phi_D', 'phi_N', 'nutrient']
            for i, field in enumerate(field_names):
                update = state[i] + (self.dt / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
                
                # Check for numerical instability
                if np.any(np.isnan(update)) or np.any(np.isinf(update)):
                    raise ValueError(f"NaNs or Infs encountered in field {field}.")
                
                
                setattr(self, field, update)

            self._enforce_volume_fractions()
            if step % self.save_steps == 0:
                self._update_history()

    def _compute_derivatives(self, state):
        phi_H, phi_D, phi_N, nutrient = state
        src_H, src_D, src_N = self.cell_production.compute_cell_sources(
            phi_H, phi_D, phi_N, nutrient, self.n_H, self.n_D, self.params
        )
        dyn_H, dyn_D, dyn_N = self.cell_dynamics.compute_cell_dynamics(
            phi_H, phi_D, phi_N, nutrient, self.dx, self.params
        )
        d_nutrient = self.diffusion_dynamics.compute_nutrient_diffusion(
            phi_H, phi_D, phi_N, nutrient, self.params
        )

        # Combine source terms and dynamics
        derivatives = (src_H + dyn_H, src_D + dyn_D, src_N + dyn_N, d_nutrient)
        
        # Check for instability
        for i, field in enumerate(['phi_H', 'phi_D', 'phi_N', 'nutrient']):
            if np.any(np.isnan(derivatives[i])) or np.any(np.isinf(derivatives[i])):
                print(f"Step {self.history['step'][-1] + 1}: {field} derivative has NaN/inf. "
                    f"src={np.max(np.abs([src_H, src_D, src_N][i]) if i < 3 else 0)}, "
                    f"dyn={np.max(np.abs([dyn_H, dyn_D, dyn_N][i]) if i < 3 else 0)}, "
                    f"d_nutrient={np.max(np.abs(d_nutrient)) if i == 3 else 0}")
        
        return derivatives

    
    def _initialize_history(self):
        return {
            'step': [0], 
            'healthy cell volume fraction': [self.phi_H], 
            'diseased cell volume fraction': [self.phi_D],  # Changed from 'disesased'
            'necrotic cell volume fraction': [self.phi_N], 
            'nutrient': [self.nutrient]
        }

    def _update_history(self):
        self.history['step'].append(self.history['step'][-1] + 1)
        self.history['healthy cell volume fraction'].append(self.phi_H)
        self.history['diseased cell volume fraction'].append(self.phi_D)  # Changed from 'disesased'
        self.history['necrotic cell volume fraction'].append(self.phi_N)
        self.history['nutrient'].append(self.nutrient)

    def _initialize_fields(self, initial_conditions):
        shape = self.grid_shape

        initial_conditions.initialize(self.params)

        # initialize fields

        self.phi_H = initial_conditions.phi_H
        self.phi_D = initial_conditions.phi_D
        self.phi_N = initial_conditions.phi_N
        self.nutrient = initial_conditions.nutrient
        self.n_H = initial_conditions.n_H
        self.n_D = initial_conditions.n_D

        self.phi_h = initial_conditions.phi_h

        self._enforce_volume_fractions()

    def _enforce_volume_fractions(self) -> None:
        """
        Enforce volume fractions using vectorized operations
        """
        total_vol = 1.0
        phi_T = self.phi_H + self.phi_D + self.phi_N
        
        # Create a mask and apply scaling in one operation
        mask = phi_T > total_vol
        if np.any(mask):
            scaling = np.ones_like(phi_T)
            scaling[mask] = total_vol / phi_T[mask]
            
            # Vectorized scaling of all cell types at once
            self.phi_H *= scaling
            self.phi_D *= scaling
            self.phi_N *= scaling
            
            # Update phi_T after rescaling
            phi_T = self.phi_H + self.phi_D + self.phi_N
        
        # Define the host region fraction
        self.phi_h = total_vol - phi_T
