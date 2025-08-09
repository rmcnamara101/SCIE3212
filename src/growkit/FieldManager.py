# field_manager.py
import numpy as np
from numba import njit, prange
from pathlib import Path
import yaml

from ProductionEngine.MatrixConstructor import MatrixConstructor

class FieldManager:
    def __init__(self, cfg_path: str):

        self.cfg = yaml.safe_load(Path(cfg_path).read_text())

        # --- domain ---------------------------------------------------------
        # Ensure self.grid is always a tuple of 3 ints
        shape = self.cfg["domain"]["shape"]
        self.grid = (shape, shape, shape)
        self.dx    = self.cfg["domain"]["dx"]
        self.dt    = self.cfg["time"]["dt"]
        self.steps = self.cfg["time"]["steps"]

        # --- populations ----------------------------------------------------
        pops = self.cfg["populations"]
        self.labels   = [p["label"] for p in pops.values()]
        self.M        = len(pops)




config = 'C:\\Users\\riley.mcnamara\\Documents\\code\\SCIE3212\\templates\\stress_tester.yaml'
manager = FieldManager(config)
print(manager.transfer_matrix)
