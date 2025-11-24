#!/usr/bin/env python3
"""
Run Simulation with Observables Export

This module provides a function to run a simulation and export observables to CSV.
Used by parameter sweep scripts to generate observables data for each parameter combination.

Author: Riley Jae McNamara
Date: 2025-02-19
"""

from pathlib import Path
import sys
import pandas as pd

# Add project root to path
if sys.platform == "darwin":
    proj = Path(__file__).parent
else:
    proj = Path(__file__).parent

# Prepend so it wins over anything else
sys.path.insert(0, str(proj))

from src.growkit.Simulator import TumorGrowthSimulator
from src.growkit.PlotEngine.SimPlotter import SimPlotter


def run_simulation_with_observables(config_path, output_dir, total_steps=50, 
                                   save_interval=1, threshold=0.4, 
                                   parameter_values=None, config=None,
                                   cached_flat_config_keys=None):
    """
    Run a simulation and export observables to CSV with configuration data.
    
    Args:
        config_path: Path to configuration YAML file
        output_dir: Directory to save simulation data and CSV
        total_steps: Number of simulation steps
        save_interval: How often to save data (every nth step)
        threshold: Density threshold for observable calculations
        parameter_values: Optional dictionary of parameter values (to be added to CSV)
        config: Optional full configuration dictionary (if None, will load from config_path)
    
    Returns:
        csv_path: Path to the exported CSV file with observables and config
    """
    import yaml
    from pathlib import Path
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config if not provided
    if config is None:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    # Run simulation
    simulator = TumorGrowthSimulator(str(config_path))
    simulation_data = simulator.run_and_save_simulation(
        total_steps=total_steps,
        save_interval=save_interval,
        save_physics_fields=True,  # Required for inhibited/necrotic radii
        output_dir=str(output_dir)
    )
    
    # Use the returned simulation_data directly instead of saving and reloading
    # This avoids the expensive disk I/O operation that was causing 40x slowdown
    # The simulation_data is already in the correct format for SimPlotter
    
    # Create plotter and export observables (to temporary location)
    plotter = SimPlotter(simulation_data)
    
    # Export observables to temporary CSV
    # Pass threshold parameter to ensure consistent calculation
    temp_csv_path = output_dir / "observables_temp.csv"
    df = plotter.export_observables_data(
        output_dir=str(output_dir), 
        filename="observables_temp.csv",
        threshold=threshold  # Use the threshold parameter passed to this function
    )
    
    # Flatten config for metadata section
    from run_parameter_sweep import flatten_dict
    
    # OPTIMIZATION: Only flatten config if we don't have cached keys
    # If cached_flat_config_keys is provided, we can skip flattening and just use the config directly
    if cached_flat_config_keys is not None:
        # We have cached keys, so we know the structure - just flatten the current config
        # This is still needed to get the values, but we can optimize the DataFrame operations
        flat_config = flatten_dict(config)
    else:
        # No cache, flatten normally
        flat_config = flatten_dict(config)
    
    # Save the CSV with structured format: metadata at top, then observables
    csv_path = output_dir / "observables_data.csv"
    
    with open(csv_path, 'w', newline='') as f:
        import csv
        writer = csv.writer(f)
        
        # Write simulation information section at the top
        writer.writerow(['# Simulation Information'])
        writer.writerow(['# ===================='])
        writer.writerow([])  # Blank row
        
        # Write varied parameters (if any)
        if parameter_values:
            writer.writerow(['# Varied Parameters'])
            for param_name, param_value in sorted(parameter_values.items()):
                writer.writerow([f'Param_{param_name.replace(".", "_")}', param_value])
            writer.writerow([])  # Blank row
        
        # Write all configuration parameters
        writer.writerow(['# Configuration Parameters'])
        for param_name, param_value in sorted(flat_config.items()):
            writer.writerow([f'Config_{param_name.replace(".", "_")}', str(param_value)])
        writer.writerow([])  # Blank row
        
        # Write separator and observables section
        writer.writerow(['# Observables Data'])
        writer.writerow(['# ================'])
        writer.writerow([])  # Blank row
    
    # Append the observables DataFrame to the CSV
    df.to_csv(csv_path, mode='a', index=False)
    
    # Remove temporary file
    if temp_csv_path.exists():
        temp_csv_path.unlink()
    
    return str(csv_path)


if __name__ == "__main__":
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run simulation and export observables")
    config_path = "/Users/rileymcnamara/CODE/2025/silicokit/configs/T_N.yaml"
    output_dir = "/Users/rileymcnamara/CODE/2025/silicokit/laboratory/saved_simulations"
    total_steps = 20
    save_interval = 1
    threshold = 0.4  # Match default used in notebook for consistency
    
    csv_path = run_simulation_with_observables(
        config_path=config_path,
        output_dir=output_dir,
        total_steps=total_steps,
        save_interval=save_interval,
        threshold=threshold
    )
    
    print(f"Observables exported to: {csv_path}")

