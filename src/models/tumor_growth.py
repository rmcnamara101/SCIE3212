# models/tumor_growth.py
import numpy as np
from tqdm import tqdm
from scipy.ndimage import laplace
from src.utils import experimental_params
from src.models.cell_dynamics import CellDynamics
from src.models.diffusion import DiffusionModel

class TumorGrowthModel:
    def __init__(self, grid_shape=(50, 50, 50), dx=0.025, dt=0.01, params=None):
        self.grid_shape = grid_shape
        self.dx = dx
        self.dt = dt
        self.params = params or experimental_params
        self.history = self._initialize_history()
        self._initialize_fields()
        
        self.cell_dynamics = CellDynamics(self)
        self.diffusion_model = DiffusionModel(self)

    def _initialize_history(self):
        return {
            'step': [], 'stem cell concentration': [], 'progenitor cell concentration': [],
            'differentiated cell concentration': [], 'necrotic cell concentration': [],
            'total cell concentration': [], 'stem cell volume': [], 'progenitor cell volume': [],
            'differentiated cell volume': [], 'necrotic cell volume': [], 'total cell volume': [],
            'radius': []
        }

    def _initialize_fields(self):
        shape = self.grid_shape
        self.C_S = np.zeros(shape)
        self.C_P = np.zeros(shape)
        self.C_D = np.zeros(shape)
        self.C_N = np.zeros(shape)
        self.nutrient = np.ones(shape)
        self.n_S = self.params['p_0'] * np.ones(shape)
        self.n_P = self.params['p_1'] * np.ones(shape)
        self.n_D = np.ones(shape)
        
        center = np.array([s // 2 for s in shape])
        radius = 3
        x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
        dist = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
        self.C_S[dist <= radius] = 1.0

    def _update(self, drug_effects=None):
        self.cell_dynamics.update_cell_sources()
        self.diffusion_model.apply_diffusion()

    def run_simulation(self, steps=100, drug_schedule=None):
        for step in tqdm(range(steps), desc="Running Simulation"):
            drug_effects = drug_schedule.get(step) if drug_schedule else None
            self._update(drug_effects)
