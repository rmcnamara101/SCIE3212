#!/usr/bin/env python3
"""
Parameter Sweep Runner

This script runs parameter sweeps by varying specified parameters in a base
configuration file. It generates multiple simulation runs with different
parameter combinations and saves observables to CSV files with complete
parameter information for comparison.

Author: Riley Jae McNamara
Date: 2025-02-19
"""

from pathlib import Path
import sys
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
import itertools
import copy
import os
import random
import numpy as np

# Add project root to path
if sys.platform == "darwin":
    proj = Path(__file__).parent
else:
    proj = Path(__file__).parent 

# Prepend so it wins over anything else
sys.path.insert(0, str(proj))

from run_simulation_observables import run_simulation_with_observables


def generate_parameter_combinations(parameter_sweep_dict):
    """
    Generate all combinations of parameters from a parameter sweep dictionary.
    
    Args:
        parameter_sweep_dict: Dictionary with parameter paths as keys and value lists as values
                            e.g., {'populations.Tumour.dynamics.lambda': [1, 2, 3]}
    
    Returns:
        List of dictionaries, each containing one parameter combination
    """
    # Extract parameter names and their possible values
    param_names = list(parameter_sweep_dict.keys())
    param_values = list(parameter_sweep_dict.values())
    
    # Generate all combinations
    combinations = []
    for combination in itertools.product(*param_values):
        param_dict = dict(zip(param_names, combination))
        combinations.append(param_dict)
    
    return combinations


def generate_random_parameter_combinations(parameter_bounds_dict, num_samples, random_seed=None):
    """
    Generate random parameter combinations from parameter bounds.
    
    Args:
        parameter_bounds_dict: Dictionary with parameter paths as keys and (min, max) tuples as values
                              e.g., {'populations.Tumour.dynamics.lambda': (1.0, 10.0)}
        num_samples: Number of random parameter combinations to generate
        random_seed: Random seed for reproducibility (optional)
    
    Returns:
        List of dictionaries, each containing one random parameter combination
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    combinations = []
    param_names = list(parameter_bounds_dict.keys())
    
    for _ in range(num_samples):
        param_dict = {}
        for param_path, (min_val, max_val) in parameter_bounds_dict.items():
            # Generate random value within bounds
            if isinstance(min_val, int) and isinstance(max_val, int):
                # Integer parameters
                param_dict[param_path] = random.randint(min_val, max_val)
            else:
                # Float parameters
                param_dict[param_path] = random.uniform(min_val, max_val)
        
        combinations.append(param_dict)
    
    return combinations


def update_config_with_parameters(base_config, parameter_dict):
    """
    Update a configuration dictionary with new parameter values.
    
    Args:
        base_config: Base configuration dictionary
        parameter_dict: Dictionary of parameter paths and values to update
    
    Returns:
        Updated configuration dictionary
    """
    # Create a deep copy to avoid modifying the original
    updated_config = copy.deepcopy(base_config)
    
    for param_path, value in parameter_dict.items():
        # Split the parameter path (e.g., 'populations.Tumour.dynamics.lambda')
        path_parts = param_path.split('.')
        
        # Navigate to the correct location in the config
        current_dict = updated_config
        for part in path_parts[:-1]:
            if part not in current_dict:
                current_dict[part] = {}
            current_dict = current_dict[part]
        
        # Set the final value
        current_dict[path_parts[-1]] = value
    
    return updated_config


def save_parameter_config(config, output_dir, run_id):
    """
    Save the parameter configuration to a YAML file for reference.
    
    Args:
        config: Configuration dictionary
        output_dir: Directory to save the config
        run_id: Unique identifier for this run
    """
    config_filename = f"config_run_{run_id:03d}.yaml"
    config_path = output_dir / config_filename
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return config_path


def run_random_parameter_sweep(base_config_path, parameter_bounds_dict, num_samples, 
                              output_dir=None, total_steps=50, save_interval=1, 
                              threshold=0.1, random_seed=None):
    """
    Run a random parameter sweep with specified bounds.
    
    Args:
        base_config_path: Path to base YAML configuration file
        parameter_bounds_dict: Dictionary defining parameter bounds as (min, max) tuples
        num_samples: Number of random parameter combinations to generate
        output_dir: Directory to save all results (defaults to laboratory/parameter_sweeps)
        total_steps: Number of simulation steps for each run
        save_interval: How often to calculate observables
        threshold: Density threshold for observable calculations
        random_seed: Random seed for reproducibility (optional)
    
    Returns:
        List of paths to generated CSV files and sweep summary
    """
    # Set up output directory
    if output_dir is None:
        output_dir = Path(__file__).parent / "laboratory" / "parameter_sweeps"
    else:
        output_dir = Path(output_dir)
    
    # Create timestamped subdirectory for this sweep
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = output_dir / f"random_parameter_sweep_{timestamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    
    # Load base configuration
    base_config = yaml.safe_load(Path(base_config_path).read_text())
    
    # Generate random parameter combinations
    parameter_combinations = generate_random_parameter_combinations(
        parameter_bounds_dict, num_samples, random_seed
    )
    
    print(f"Starting random parameter sweep with {num_samples} samples...")
    print(f"Parameters being varied: {list(parameter_bounds_dict.keys())}")
    print(f"Parameter bounds: {parameter_bounds_dict}")
    print(f"Output directory: {sweep_dir}")
    if random_seed is not None:
        print(f"Random seed: {random_seed}")
    
    # Store results
    csv_files = []
    sweep_summary = {
        'base_config_path': str(base_config_path),
        'parameter_bounds_dict': parameter_bounds_dict,
        'num_samples': num_samples,
        'random_seed': random_seed,
        'total_steps': total_steps,
        'save_interval': save_interval,
        'threshold': threshold,
        'timestamp': timestamp,
        'results': []
    }
    
    # Run each parameter combination
    for run_id, param_combination in enumerate(parameter_combinations, 1):
        print(f"\n{'='*60}")
        print(f"Running simulation {run_id}/{num_samples}")
        print(f"Parameter combination: {param_combination}")
        print(f"{'='*60}")
        
        # Update configuration with current parameter combination
        updated_config = update_config_with_parameters(base_config, param_combination)
        
        # Save the configuration for this run
        config_path = save_parameter_config(updated_config, sweep_dir, run_id)
        
        # Create temporary config file for this run
        temp_config_path = sweep_dir / f"temp_config_run_{run_id:03d}.yaml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(updated_config, f, default_flow_style=False, sort_keys=False)
        
        try:
            # Run simulation with observables
            csv_path = run_simulation_with_observables(
                config_path=str(temp_config_path),
                output_dir=sweep_dir,
                total_steps=total_steps,
                save_interval=save_interval,
                threshold=threshold,
                parameter_values=param_combination
            )
            
            csv_files.append(csv_path)
            
            # Store run information
            run_info = {
                'run_id': run_id,
                'parameter_combination': param_combination,
                'config_path': str(config_path),
                'csv_path': str(csv_path),
                'success': True
            }
            sweep_summary['results'].append(run_info)
            
            print(f"✓ Simulation {run_id} completed successfully")
            print(f"  CSV saved to: {csv_path}")
            
        except Exception as e:
            print(f"✗ Simulation {run_id} failed: {str(e)}")
            
            # Store failed run information
            run_info = {
                'run_id': run_id,
                'parameter_combination': param_combination,
                'config_path': str(config_path),
                'csv_path': None,
                'success': False,
                'error': str(e)
            }
            sweep_summary['results'].append(run_info)
        
        finally:
            # Clean up temporary config file
            if temp_config_path.exists():
                temp_config_path.unlink()
    
    # Save sweep summary
    summary_path = sweep_dir / "random_parameter_sweep_summary.yaml"
    with open(summary_path, 'w') as f:
        yaml.dump(sweep_summary, f, default_flow_style=False, sort_keys=False)
    
    # Print final summary
    successful_runs = sum(1 for result in sweep_summary['results'] if result['success'])
    failed_runs = len(sweep_summary['results']) - successful_runs
    
    print(f"\n{'='*60}")
    print(f"RANDOM PARAMETER SWEEP COMPLETED")
    print(f"{'='*60}")
    print(f"Total samples: {num_samples}")
    print(f"Successful: {successful_runs}")
    print(f"Failed: {failed_runs}")
    print(f"Results saved to: {sweep_dir}")
    print(f"Summary saved to: {summary_path}")
    
    if csv_files:
        print(f"CSV files generated:")
        for csv_file in csv_files:
            print(f"  - {csv_file}")
    
    return csv_files, sweep_summary


def run_parameter_sweep(base_config_path, parameter_sweep_dict, output_dir=None, 
                       total_steps=50, save_interval=1, threshold=0.1):
    """
    Run a parameter sweep with multiple simulation runs.
    
    Args:
        base_config_path: Path to base YAML configuration file
        parameter_sweep_dict: Dictionary defining parameter variations
        output_dir: Directory to save all results (defaults to laboratory/parameter_sweeps)
        total_steps: Number of simulation steps for each run
        save_interval: How often to calculate observables
        threshold: Density threshold for observable calculations
    
    Returns:
        List of paths to generated CSV files
    """
    # Set up output directory
    if output_dir is None:
        output_dir = Path(__file__).parent / "laboratory" / "parameter_sweeps"
    else:
        output_dir = Path(output_dir)
    
    # Create timestamped subdirectory for this sweep
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = output_dir / f"parameter_sweep_{timestamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    
    # Load base configuration
    base_config = yaml.safe_load(Path(base_config_path).read_text())
    
    # Generate all parameter combinations
    parameter_combinations = generate_parameter_combinations(parameter_sweep_dict)
    
    print(f"Starting parameter sweep with {len(parameter_combinations)} combinations...")
    print(f"Parameters being varied: {list(parameter_sweep_dict.keys())}")
    print(f"Output directory: {sweep_dir}")
    
    # Store results
    csv_files = []
    sweep_summary = {
        'base_config_path': str(base_config_path),
        'parameter_sweep_dict': parameter_sweep_dict,
        'total_combinations': len(parameter_combinations),
        'total_steps': total_steps,
        'save_interval': save_interval,
        'threshold': threshold,
        'timestamp': timestamp,
        'results': []
    }
    
    # Run each parameter combination
    for run_id, param_combination in enumerate(parameter_combinations, 1):
        print(f"\n{'='*60}")
        print(f"Running simulation {run_id}/{len(parameter_combinations)}")
        print(f"Parameter combination: {param_combination}")
        print(f"{'='*60}")
        
        # Update configuration with current parameter combination
        updated_config = update_config_with_parameters(base_config, param_combination)
        
        # Save the configuration for this run
        config_path = save_parameter_config(updated_config, sweep_dir, run_id)
        
        # Create temporary config file for this run
        temp_config_path = sweep_dir / f"temp_config_run_{run_id:03d}.yaml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(updated_config, f, default_flow_style=False, sort_keys=False)
        
        try:
            # Run simulation with observables
            csv_path = run_simulation_with_observables(
                config_path=str(temp_config_path),
                output_dir=sweep_dir,
                total_steps=total_steps,
                save_interval=save_interval,
                threshold=threshold,
                parameter_values=param_combination
            )
            
            csv_files.append(csv_path)
            
            # Store run information
            run_info = {
                'run_id': run_id,
                'parameter_combination': param_combination,
                'config_path': str(config_path),
                'csv_path': str(csv_path),
                'success': True
            }
            sweep_summary['results'].append(run_info)
            
            print(f"✓ Simulation {run_id} completed successfully")
            print(f"  CSV saved to: {csv_path}")
            
        except Exception as e:
            print(f"✗ Simulation {run_id} failed: {str(e)}")
            
            # Store failed run information
            run_info = {
                'run_id': run_id,
                'parameter_combination': param_combination,
                'config_path': str(config_path),
                'csv_path': None,
                'success': False,
                'error': str(e)
            }
            sweep_summary['results'].append(run_info)
        
        finally:
            # Clean up temporary config file
            if temp_config_path.exists():
                temp_config_path.unlink()
    
    # Save sweep summary
    summary_path = sweep_dir / "parameter_sweep_summary.yaml"
    with open(summary_path, 'w') as f:
        yaml.dump(sweep_summary, f, default_flow_style=False, sort_keys=False)
    
    # Print final summary
    successful_runs = sum(1 for result in sweep_summary['results'] if result['success'])
    failed_runs = len(sweep_summary['results']) - successful_runs
    
    print(f"\n{'='*60}")
    print(f"PARAMETER SWEEP COMPLETED")
    print(f"{'='*60}")
    print(f"Total runs: {len(parameter_combinations)}")
    print(f"Successful: {successful_runs}")
    print(f"Failed: {failed_runs}")
    print(f"Results saved to: {sweep_dir}")
    print(f"Summary saved to: {summary_path}")
    
    if csv_files:
        print(f"CSV files generated:")
        for csv_file in csv_files:
            print(f"  - {csv_file}")
    
    return csv_files, sweep_summary


def main():
    """Example random parameter sweep configuration."""
    
    # Base configuration file
    base_config_path = Path(__file__).parent / "configs" / "T_N.yaml"
    
    # Define parameter bounds for random sampling
    parameter_bounds_dict = {
        # Tumor cell growth rate: random between 2.0 and 8.0
        'populations.Tumour.dynamics.lambda': (2.0, 8.0),
        # Tumor cell death rate: random between 0.2 and 2.0
        'populations.Tumour.dynamics.mu': (0.2, 2.0),
        # Tumor cell mobility: random between 0.5 and 2.0
        'populations.Tumour.dynamics.mobility': (0.5, 2.0),
        # Nutrient diffusion: random between 0.01 and 0.1
        'nutrient.dynamics.diffusion': (0.01, 0.1),
        # Nutrient boundary value: random between 0.8 and 1.2
        'nutrient.dynamics.boundary_value': (0.8, 1.2),
        # Adhesion energy: random between 0.5 and 2.0
        'physics.adhesion_energy.m': (0.5, 2.0)
    }
    
    # Number of random samples to generate
    num_samples = 3
    
    # Run random parameter sweep
    csv_files, summary = run_random_parameter_sweep(
        base_config_path=base_config_path,
        parameter_bounds_dict=parameter_bounds_dict,
        num_samples=num_samples,
        total_steps=10,  # Short runs for testing
        save_interval=1,
        threshold=0.1,
        random_seed=42  # For reproducible results
    )
    
    print(f"\nRandom parameter sweep completed with {len(csv_files)} successful runs")


if __name__ == "__main__":
    main()
