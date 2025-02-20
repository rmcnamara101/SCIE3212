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

from src.utils.utils import experimental_params
from src.models.cell_production import ProductionModel
from src.models.cell_dynamics import DynamicsModel

class TumorGrowthModel:
    def __init__(self, grid_shape: Tuple[int, int, int] = (50, 50, 50), dx: float = 0.025, dt: float = 0.1, params: dict = None, initial_conditions: Any = None) -> None:
        self.grid_shape = grid_shape
        self.dx = dx
        self.dt = dt
        self.params = params or experimental_params
        self._initialize_fields(initial_conditions)
        self.history = self._initialize_history()
        
        self.cell_production = ProductionModel(self)
        self.cell_dynamics = DynamicsModel(self)

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
        self.history['stem cell volume'].append(np.sum(self.C_S) * self.dx**3)
        self.history['progenitor cell volume'].append(np.sum(self.C_P) * self.dx**3)
        self.history['differentiated cell volume'].append(np.sum(self.C_D) * self.dx**3)
        self.history['necrotic cell volume'].append(np.sum(self.C_N) * self.dx**3)
        self.history['total cell volume'].append(np.sum(self.C_T) * self.dx**3)
        self.history['radius'].append(self._calculate_radius())


    def _initialize_fields(self, initial_conditions: Any = None) -> None:
        """
        This function will initialize the fields of the tumor growth model
        """
        shape = self.grid_shape
        self.C_S = np.zeros(shape)
        self.C_P = np.zeros(shape)
        self.C_D = np.zeros(shape)
        self.C_N = np.zeros(shape)
        self.C_T = np.zeros(shape)
        self.nutrient = np.ones(shape)
        self.n_S = self.params['p_0'] * np.ones(shape)
        self.n_P = self.params['p_1'] * np.ones(shape)
        self.n_D = np.ones(shape)
        # initialization of the tumor in a specific shape could be done here
        # load some sort of file that stores both the 3D data of the tumor, spatially,
        # and the concentration of cells in each voxel.

         # Create a small spherical initial tumor
        center = np.array([s//2 for s in shape])
        radius = 3  # Initial radius of tumor sphere
        
        x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
        dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
        
        # Set initial stem cell concentration in sphere
        self.C_S[dist_from_center <= radius] = 0.1
        self.update_total_cell_density()
    


    def _update(self) -> None:
        """
        This is the update function that will be called by the simulation
        it will update the cell sources and the cell dynamics of the tumor
        """
        # update the cell production
        self.cell_production.apply_cell_sources()
        # update the cell dynamics
        self.cell_dynamics.apply_cell_dynamics()


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
    
    def update_total_cell_density(self) -> None:
        """
        This function will update the total cell density
        """
        self.C_T = self.C_S + self.C_P + self.C_D + self.C_N
        
