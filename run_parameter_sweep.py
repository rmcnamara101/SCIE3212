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
try:
    from openpyxl import Workbook
    from openpyxl.utils.dataframe import dataframe_to_rows
    from openpyxl.styles import Font, PatternFill, Alignment
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False
    print("Warning: openpyxl not available. Excel export will use xlsxwriter fallback.")

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
    # OPTIMIZATION: Use a more efficient copy strategy
    # Instead of deep copying everything, we'll do a selective deep copy
    # Only deep copy the nested dictionaries we need to modify
    import copy as cp
    
    # Start with a shallow copy
    updated_config = base_config.copy()
    
    # Track which top-level keys we need to deep copy
    keys_to_deepcopy = set()
    for param_path in parameter_dict.keys():
        top_key = param_path.split('.')[0]
        keys_to_deepcopy.add(top_key)
    
    # Deep copy only the necessary top-level sections
    for key in keys_to_deepcopy:
        if key in updated_config:
            updated_config[key] = cp.deepcopy(base_config[key])
    
    # Now update the parameters
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


def flatten_dict(d, parent_key='', sep='.'):
    """Helper function to flatten nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


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
    
    # OPTIMIZATION: Pre-flatten config once and cache the structure
    # This avoids flattening the same structure repeatedly in run_simulation_with_observables
    from run_parameter_sweep import flatten_dict
    base_flat_config = flatten_dict(base_config)  # Cache the flattened structure
    base_config_keys = set(base_flat_config.keys())  # Cache keys for faster lookups
    
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
    import time
    for run_id, param_combination in enumerate(parameter_combinations, 1):
        run_start_time = time.time()
        print(f"\n{'='*60}")
        print(f"Running simulation {run_id}/{num_samples}")
        print(f"Parameter combination: {param_combination}")
        print(f"{'='*60}")
        
        # Update configuration with current parameter combination
        t0 = time.time()
        updated_config = update_config_with_parameters(base_config, param_combination)
        config_update_time = time.time() - t0
        
        # OPTIMIZATION: Only write config file once (use temp_config_path for both purposes)
        t0 = time.time()
        temp_config_path = sweep_dir / f"temp_config_run_{run_id:03d}.yaml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(updated_config, f, default_flow_style=False, sort_keys=False)
        
        # Also save a permanent copy with run_id for reference
        config_path = save_parameter_config(updated_config, sweep_dir, run_id)
        config_save_time = time.time() - t0
        
        try:
            # Run simulation with observables (this creates CSV with config already included)
            # OPTIMIZATION: Pass cached flattened config structure to avoid re-flattening
            t0 = time.time()
            csv_path = run_simulation_with_observables(
                config_path=str(temp_config_path),
                output_dir=sweep_dir,
                total_steps=total_steps,
                save_interval=save_interval,
                threshold=threshold,
                parameter_values=param_combination,
                config=updated_config,  # Pass full config for inclusion in CSV
                cached_flat_config_keys=base_config_keys  # Pass cached keys to avoid re-flattening
            )
            simulation_time = time.time() - t0
            
            # The CSV already contains config and observables, but we need to rename it with run_id
            csv_files.append(csv_path)
            
            # Rename the CSV to include run_id for easy identification
            t0 = time.time()
            combined_csv_path = sweep_dir / f"observables_run_{run_id:03d}.csv"
            if csv_path != combined_csv_path:
                import shutil
                shutil.move(csv_path, combined_csv_path)
                csv_path = combined_csv_path
            file_ops_time = time.time() - t0
            
            total_run_time = time.time() - run_start_time
            
            # Store run information
            run_info = {
                'run_id': run_id,
                'parameter_combination': param_combination,
                'config_path': str(config_path),
                'csv_path': str(csv_path),
                'success': True,
                'timings': {
                    'config_update': config_update_time,
                    'config_save': config_save_time,
                    'simulation': simulation_time,
                    'file_ops': file_ops_time,
                    'total': total_run_time
                }
            }
            sweep_summary['results'].append(run_info)
            
            print(f"✓ Simulation {run_id} completed successfully")
            print(f"  CSV saved to: {csv_path}")
            print(f"  Timing breakdown:")
            print(f"    Config update: {config_update_time:.2f}s")
            print(f"    Config save: {config_save_time:.2f}s")
            print(f"    Simulation: {simulation_time:.2f}s ({simulation_time/60:.1f} min)")
            print(f"    File ops: {file_ops_time:.2f}s")
            print(f"    Total: {total_run_time:.2f}s ({total_run_time/60:.1f} min)")
            
        except Exception as e:
            total_run_time = time.time() - run_start_time
            print(f"✗ Simulation {run_id} failed: {str(e)}")
            
            # Store failed run information
            run_info = {
                'run_id': run_id,
                'parameter_combination': param_combination,
                'config_path': str(config_path),
                'csv_path': None,
                'combined_csv_path': None,
                'success': False,
                'error': str(e),
                'timings': {
                    'total': total_run_time
                }
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
        print(f"CSV files (with config and observables) generated: {len(csv_files)}")
    
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
    
    # OPTIMIZATION: Pre-flatten config once and cache the structure
    # This avoids flattening the same structure repeatedly in run_simulation_with_observables
    from run_parameter_sweep import flatten_dict
    base_flat_config = flatten_dict(base_config)  # Cache the flattened structure
    base_config_keys = set(base_flat_config.keys())  # Cache keys for faster lookups
    
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
    import time
    for run_id, param_combination in enumerate(parameter_combinations, 1):
        run_start_time = time.time()
        print(f"\n{'='*60}")
        print(f"Running simulation {run_id}/{len(parameter_combinations)}")
        print(f"Parameter combination: {param_combination}")
        print(f"{'='*60}")
        
        # Update configuration with current parameter combination
        t0 = time.time()
        updated_config = update_config_with_parameters(base_config, param_combination)
        config_update_time = time.time() - t0
        
        # OPTIMIZATION: Only write config file once (use temp_config_path for both purposes)
        t0 = time.time()
        temp_config_path = sweep_dir / f"temp_config_run_{run_id:03d}.yaml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(updated_config, f, default_flow_style=False, sort_keys=False)
        
        # Also save a permanent copy with run_id for reference
        config_path = save_parameter_config(updated_config, sweep_dir, run_id)
        config_save_time = time.time() - t0
        
        try:
            # Run simulation with observables
            # OPTIMIZATION: Pass cached flattened config structure to avoid re-flattening
            t0 = time.time()
            csv_path = run_simulation_with_observables(
                config_path=str(temp_config_path),
                output_dir=sweep_dir,
                total_steps=total_steps,
                save_interval=save_interval,
                threshold=threshold,
                parameter_values=param_combination,
                config=updated_config,  # Pass full config for inclusion in CSV
                cached_flat_config_keys=base_config_keys  # Pass cached keys to avoid re-flattening
            )
            simulation_time = time.time() - t0
            
            # The CSV already contains config and observables, but we need to rename it with run_id
            csv_files.append(csv_path)
            
            # Rename the CSV to include run_id for easy identification
            t0 = time.time()
            combined_csv_path = sweep_dir / f"observables_run_{run_id:03d}.csv"
            if csv_path != combined_csv_path:
                import shutil
                shutil.move(csv_path, combined_csv_path)
                csv_path = combined_csv_path
            file_ops_time = time.time() - t0
            
            total_run_time = time.time() - run_start_time
            
            # Store run information
            run_info = {
                'run_id': run_id,
                'parameter_combination': param_combination,
                'config_path': str(config_path),
                'csv_path': str(csv_path),
                'success': True,
                'timings': {
                    'config_update': config_update_time,
                    'config_save': config_save_time,
                    'simulation': simulation_time,
                    'file_ops': file_ops_time,
                    'total': total_run_time
                }
            }
            sweep_summary['results'].append(run_info)
            
            print(f"✓ Simulation {run_id} completed successfully")
            print(f"  CSV saved to: {csv_path}")
            print(f"  Timing breakdown:")
            print(f"    Config update: {config_update_time:.2f}s")
            print(f"    Config save: {config_save_time:.2f}s")
            print(f"    Simulation: {simulation_time:.2f}s ({simulation_time/60:.1f} min)")
            print(f"    File ops: {file_ops_time:.2f}s")
            print(f"    Total: {total_run_time:.2f}s ({total_run_time/60:.1f} min)")
            
        except Exception as e:
            total_run_time = time.time() - run_start_time
            print(f"✗ Simulation {run_id} failed: {str(e)}")
            
            # Store failed run information
            run_info = {
                'run_id': run_id,
                'parameter_combination': param_combination,
                'config_path': str(config_path),
                'csv_path': None,
                'success': False,
                'error': str(e),
                'timings': {
                    'total': total_run_time
                }
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
        print(f"CSV files (with config and observables) generated: {len(csv_files)}")
    
    return csv_files, sweep_summary


def main():
    """Example random parameter sweep configuration."""
    
    # Base configuration file
    base_config_path = Path(__file__).parent / "configs" / "T_N.yaml"
    
    # Define parameter bounds for random sampling
    parameter_bounds_dict = {
        # Tumor cell growth rate: random between 2.0 and 8.0
        'populations.Tumour.dynamics.lambda': (0.1, 30),
        # Tumor cell death rate: random between 0.2 and 2.0
        'populations.Tumour.dynamics.mu': (0.1, 30),
        # Tumor cell nutrient threshold: random between 0.1 and 1.0
        'populations.Tumour.dynamics.nutrient_threshold': (0.1, 0.99),
        # Tumor cell nutrient consumption: random between 0.01 and 0.1
        'populations.Tumour.dynamics.nutrient_consumption': (0.01, 0.99),
        # Tumor cell nutrient production: random between 0.01 and 0.1
        'populations.Tumour.dynamics.nutrient_production': (0.01, 0.99),
        # Necrotic cell beta_N: random between 0.00001 and 0.1
        'populations.Necrotic.dynamics.beta_N': (0.00001, 0.1),
        # Necrotic cell death rate: random between 0.1 and 30
        'populations.Necrotic.dynamics.mu': (0.1, 30),
        # Nutrient growth threshold offset: random between 0.1 and 0.99
        'nutrient.dynamics.growth_threshold_offset': (0.1, 0.99),
        # Nutrient diffusion: random between 10000 and 100000
        'nutrient.dynamics.diffusion': (10000, 500000),
    }
    
    # Number of random samples to generate
    num_samples = 1000
    
    # Run random parameter sweep
    csv_files, summary = run_random_parameter_sweep(
        base_config_path=base_config_path,
        parameter_bounds_dict=parameter_bounds_dict,
        num_samples=num_samples,
        total_steps=20,  # Short runs for testing
        save_interval=1,
        threshold=0.1,
        random_seed=42  # For reproducible results
    )
    
    print(f"\nRandom parameter sweep completed with {len(csv_files)} successful runs")


if __name__ == "__main__":
    main()
