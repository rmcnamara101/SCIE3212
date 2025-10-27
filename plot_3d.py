import sys
from pathlib import Path
import numpy as np

if sys.platform == "darwin":
    proj = "/Users/rileymcnamara/CODE/2025/silicokit/"
    sys.path.insert(0, proj)
else:
    proj = "C:/Users/riley.mcnamara/Documents/code/silicokit/"
    sys.path.insert(0, proj)

from src.growkit.PlotEngine.CellFieldPlotter import CellFieldPlotter

# Load the raw npz file
raw_data = np.load(proj + "laboratory/saved_simulations/simulation_data.npz")

# Convert npz to dictionary format
simulation_data = CellFieldPlotter._convert_npz_to_dict(raw_data)

# Create plotter from converted simulation data
plotter = CellFieldPlotter.from_simulation_data(simulation_data, step_idx=0)

# Plot 3D tumor field
#plotter.plot_3d_tumor_field(simulation_data, step_idx=8, isosurface_level=0.1)

# Plot 3D quadratic potential with tumor overlay
#plotter.plot_3d_quadratic_potential(simulation_data, step_idx=0, show_tumor_field=True)

# Analyze coagulation dynamics over multiple steps
plotter.plot_coagulation_analysis(simulation_data, step_indices=[0, 2, 4, 6, 8])

