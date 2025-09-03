
import sys
from pathlib import Path

if sys.platform == "darwin":
    proj = "/Users/rileymcnamara/CODE/2025/silicokit/"
    sys.path.insert(0, proj)
else:
    proj = "C:/Users/riley.mcnamara/Documents/code/silicokit/"
    sys.path.insert(0, proj)

from src.growkit.ExperimentEngine.ExperimentRunner import ExperimentRunner
from src.growkit.Simulator import TumorGrowthSimulator

# Create simulator to get the config
simulator = TumorGrowthSimulator(proj + "/configs/csc-t-n.yaml")

# Create experiment runner with the config file path directly
experiment_runner = ExperimentRunner(proj + "/configs/csc-t-n.yaml", proj + "/laboratory/parameter_sweeps")

# Define a much smaller parameter sweep for testing
# This will generate only 2 * 3 * 2 = 12 experiments instead of 10^13
param_sweep = {
    "populations.Stem.dynamics.lambda": {"type": "range", "min": 60, "max": 80, "steps": 2},
    "populations.Stem.dynamics.mu": {"type": "range", "min": 8, "max": 12, "steps": 3},
    "populations.Stem.transfer.rate": {"type": "range", "min": 0.03, "max": 0.07, "steps": 2}
}

print("Parameter sweep defined:")
print(f"Total experiments: {2 * 3 * 2}")

# Add the parameter sweep
experiment_runner.add_parameter_sweep(param_sweep)

# Run just ONE experiment for testing
print("\nRunning single experiment to test...")
experiment_runner.run_all_experiments(
    start_index=0, 
    end_index=1,  # Only run 1 experiment
    save_progress=True,
    progress_interval=1
)

# Print status
experiment_runner.print_status()

print(f"\nResults saved to Excel: {experiment_runner.excel_file}")
print("Note: You can stop the experiment at any time and your data will be saved!")

# Check if any NPZ files were created
output_dir = Path(proj + "/laboratory/parameter_sweeps")
npz_files = list(output_dir.glob("**/*.npz"))
if npz_files:
    print(f"\nWARNING: NPZ files were still created: {len(npz_files)} files")
    for npz_file in npz_files[:5]:  # Show first 5
        print(f"  {npz_file}")
    if len(npz_files) > 5:
        print(f"  ... and {len(npz_files) - 5} more")
else:
    print("\nSUCCESS: No NPZ files were created - only Excel output!")