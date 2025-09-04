
import sys
from pathlib import Path

if sys.platform == "darwin":
    proj = "/Users/rileymcnamara/CODE/2025/silicokit/"
    sys.path.insert(0, proj)
else:
    proj = "C:/Users/riley.mcnamara/Documents/code/silicokit/"
    sys.path.insert(0, proj)

from src.growkit.ExperimentEngine.SimpleExperimentRunner import SimpleExperimentRunner

# Create experiment runner with the config file path directly
experiment_runner = SimpleExperimentRunner(proj + "/configs/csc-t-n.yaml", proj + "/laboratory/parameter_sweeps")

# Define parameter bounds for random sampling
# This is much simpler - just define the bounds for each parameter you want to vary
num_experiments = 2
param_bounds = {
    # Population dynamics parameters - Stem cells
    "populations.Stem.dynamics.lambda": {"min": 40, "max": 100},
    "populations.Stem.dynamics.mu": {"min": 5, "max": 20},
    "populations.Stem.dynamics.mobility": {"min": 0.1, "max": 5.0},
    "populations.Stem.dynamics.nutrient_threshold": {"min": 0.1, "max": 0.9},
    "populations.Stem.dynamics.nutrient_production": {"min": 0.01, "max": 0.2},
    
    # Population dynamics parameters - Tumour cells
    "populations.Tumour.dynamics.lambda": {"min": 30, "max": 80},
    "populations.Tumour.dynamics.mu": {"min": 5, "max": 20},
    "populations.Tumour.dynamics.mobility": {"min": 0.1, "max": 5.0},
    "populations.Tumour.dynamics.nutrient_threshold": {"min": 0.1, "max": 0.9},
    "populations.Tumour.dynamics.nutrient_production": {"min": 0.01, "max": 0.2},
    
    # Population dynamics parameters - Necrotic cells
    "populations.Necrotic.dynamics.lambda": {"min": 0.0, "max": 0.1},   
    "populations.Necrotic.dynamics.mu": {"min": 1, "max": 10},
    "populations.Necrotic.dynamics.mobility": {"min": 0.0, "max": 0.5},
    "populations.Necrotic.dynamics.nutrient_threshold": {"min": 0.0, "max": 0.1},
    "populations.Necrotic.dynamics.nutrient_production": {"min": 0.0, "max": 0.01},
    
    # Nutrient dynamics
    "nutrient.dynamics.diffusion": {"min": 300, "max": 1200},
    
    # Physics parameters
    "physics.adhesion_energy.m": {"min": -2, "max": 2}
}

print("Simple parameter sweep for cost function landscape exploration:")
print(f"Number of experiments: {num_experiments}")
print(f"Parameters being randomly sampled: {len(param_bounds)}")
print("\nParameter bounds:")
for param_path, bounds in param_bounds.items():
    print(f"  {param_path}: {bounds['min']:.3g} to {bounds['max']:.3g}")

# Set up the experiment runner with parameter bounds
experiment_runner.setup_parameter_sweep(param_bounds, num_experiments)

# Run all experiments
print(f"\nRunning {num_experiments} experiments...")
experiment_runner.run_all_experiments()

# Print status
experiment_runner.print_status()

print(f"\nResults saved to Excel: {experiment_runner.excel_file}")
print("Each row contains:")
print("- ALL simulation parameters (including non-sampled ones)")
print("- Time series data for each cell population radius and total cells")
print("- Final state statistics for cost function analysis")

print("\nTo run more experiments:")
print("1. Increase 'num_experiments'")
print("2. Add more parameters to 'param_bounds'")
print("3. Each experiment will be one row in the Excel file")
print("4. Use the time series data to compare with experimental data")