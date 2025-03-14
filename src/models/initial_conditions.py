# File: src/models/initial_conditions.py

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from copy import deepcopy

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

    def __init__(self, grid_shape: Tuple, radius: int = 5, nutrient_value: float = 1):
        super().__init__(grid_shape)
        self.radius = radius
        self.nutrient_value = nutrient_value
    
    def initialize(self, params):
        # Initialize nutrient field
        Nz, Ny, Nx = self.grid_shape
        self.nutrient = self.nutrient_value * np.ones(self.grid_shape)
        
        # Use broadcasting for parameter fields
        self.n_H = 0.2 * np.ones(self.grid_shape)
        self.n_D = 0.2 * np.ones(self.grid_shape)
        
        # Create slightly off-center coordinates
        center = np.array([
            Nz//2 ,  # Shift 2 cells in z direction
            Ny//2 ,  # Shift 1 cell in negative y direction
            Nx//2   # Shift 1 cell in x direction
        ])
        
        for k in range(Nz):
            for j in range(Ny):
                for i in range(Nx):
                    r = np.sqrt((k - center[0])**2 + (j - center[1])**2 + (i - center[2])**2)
                    if r < self.radius:  # Using the class radius parameter
                        self.phi_H[k, j, i] = 0.3 #* np.exp(-r**2 / 10)  # Gaussian tumor
                        self.phi_D[k, j, i] = 0.6 #* np.exp(-r**2 / 10)
                        self.phi_N[k, j, i] = 0.1

        

