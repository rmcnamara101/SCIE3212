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

# Print available populations
print("Available populations:")
for i, label in enumerate(plotter.labels):
    print(f"  Index {i}: {label}")

# Plot 3D tumor field
#plotter.plot_3d_tumor_field(simulation_data, step_idx=9, isosurface_level=0.1, cmap="PiYG")

# Plot 3D populations with different colors and wedge cutout to see interior
# Use per-population thresholds: lower threshold for necrotic cells (typically lower density)
# Adjust indices based on your population labels
# Try very low thresholds (even 0.0) to see all non-zero voxels
isosurface_levels = {
    0: 0.05,  # First population (e.g., stem cells) - lower to see more
    1: 0.05,  # Second population (e.g., tumour cells) - lower to see more
    2: 0.001  # Third population (e.g., necrotic cells) - very low threshold
}

plotter.plot_3d_populations(simulation_data, step_idx=5, isosurface_level=isosurface_levels,
                            population_colors=['red', 'blue', 'green'],
                            cutout_angle=60, cutout_azimuth_start=0, cutout_buffer=10.0)

# Plot 3D quadratic potential with tumor overlay
#plotter.plot_3d_quadratic_potential(simulation_data, step_idx=0, show_tumor_field=True)

# Analyze coagulation dynamics over multiple steps
#plotter.plot_coagulation_analysis(simulation_data, step_indices=[0, 1, 2, 3, 4])

