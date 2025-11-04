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
                                   save_interval=1, threshold=0.1, 
                                   parameter_values=None, config=None):
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
    
    # Load saved simulation data
    simulation_file = output_dir / "simulation_data.npz"
    if not simulation_file.exists():
        raise FileNotFoundError(f"Simulation data not found at {simulation_file}")
    
    loaded_data = simulator.load_simulation_data(str(simulation_file))
    
    # Create plotter and export observables (to temporary location)
    plotter = SimPlotter(loaded_data)
    
    # Export observables to temporary CSV
    temp_csv_path = output_dir / "observables_temp.csv"
    df = plotter.export_observables_data(output_dir=str(output_dir), filename="observables_temp.csv")
    
    # Now add config and parameter values directly
    from run_parameter_sweep import flatten_dict
    
    # Flatten the full configuration
    flat_config = flatten_dict(config)
    
    # Add varied parameters first (with Param_ prefix)
    if parameter_values:
        for param_name, param_value in parameter_values.items():
            col_name = f"Param_{param_name.replace('.', '_')}"
            df[col_name] = param_value
    
    # Add all config parameters (with Config_ prefix)
    for param_name, param_value in sorted(flat_config.items()):
        col_name = f"Config_{param_name.replace('.', '_')}"
        df[col_name] = str(param_value)
    
    # Reorder columns: Run_ID, Param columns, Config columns, then observables
    config_cols = [c for c in df.columns if c.startswith('Config_')]
    param_cols = [c for c in df.columns if c.startswith('Param_')]
    obs_cols = [c for c in df.columns if c not in config_cols + param_cols]
    
    config_cols = sorted(config_cols)
    param_cols = sorted(param_cols)
    
    # Final column order: varied params, then full config, then observables
    column_order = param_cols + config_cols + obs_cols
    df = df[column_order]
    
    # Save the combined CSV (use a generic name since run_id isn't available here)
    csv_path = output_dir / "observables_data.csv"
    df.to_csv(csv_path, index=False)
    
    # Remove temporary file
    if temp_csv_path.exists():
        temp_csv_path.unlink()
    
    return str(csv_path)


if __name__ == "__main__":
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run simulation and export observables")
    parser.add_argument("config_path", type=str, help="Path to configuration YAML file")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--total_steps", type=int, default=50, help="Total simulation steps")
    parser.add_argument("--save_interval", type=int, default=1, help="Save interval")
    parser.add_argument("--threshold", type=float, default=0.1, help="Density threshold")
    
    args = parser.parse_args()
    
    csv_path = run_simulation_with_observables(
        config_path=args.config_path,
        output_dir=args.output_dir,
        total_steps=args.total_steps,
        save_interval=args.save_interval,
        threshold=args.threshold
    )
    
    print(f"Observables exported to: {csv_path}")

