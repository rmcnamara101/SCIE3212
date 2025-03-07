# File: src/models/initial_conditions.py

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple

class InitialCondition(ABC):
    """
    Abstract base class for initial conditions.
    """

    def __init__(self, grid_shape: Tuple):
        self.grid_shape = grid_shape
        self.phi_H = np.zeros(grid_shape)
        self.phi_D = np.zeros(grid_shape)
        self.phi_N = np.zeros(grid_shape)
        self.nutrient = np.zeros(grid_shape)
        self.n_H = np.zeros(grid_shape)
        self.n_D = np.zeros(grid_shape)
        self.phi_h = np.zeros(grid_shape)
        
    @abstractmethod
    def initialize(self, params: dict):
        """
        Initialize the cell and nutrient fields based on the specific initial condition.
        """
        pass

class SphericalTumor(InitialCondition):
    """
    Initial condition with a spherical tumor at the center of the grid.
    """

    def __init__(self, grid_shape: Tuple, radius: int = 5, nutrient_value: float = 0.001):
        super().__init__(grid_shape)
        self.radius = radius
        self.nutrient_value = nutrient_value
    
    def initialize(self, params):
        
        # Initialize nutrient field
        self.nutrient = self.nutrient_value * np.ones(self.grid_shape)
        
        # Use broadcasting for parameter fields
        self.n_H = params['p_H'] * np.ones(self.grid_shape)
        self.n_D = np.ones(self.grid_shape)
        
        # Pre-compute distance grid for initialization and radius calculation
        center = np.array([s // 2 for s in self.grid_shape])
        x, y, z = np.ogrid[:self.grid_shape[0], :self.grid_shape[1], :self.grid_shape[2]]
        dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
        
        # Use boolean masking for more efficient initialization
        self.phi_H[dist_from_center <= self.radius] = 0.2
        self.phi_D[dist_from_center <= self.radius] = 0.7

        

