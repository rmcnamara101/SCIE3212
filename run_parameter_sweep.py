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


def load_config_safely(config_path):
    """
    Load a YAML config file and convert numpy types to native Python types.
    This handles cases where config files contain numpy scalars with special YAML tags.
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        Configuration dictionary with numpy types converted to native Python types
    """
    # Use the shared conversion function
    
    # Try safe_load first (for normal YAML files)
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        # If safe_load fails (e.g., due to numpy tags), try unsafe loading
        # This is safe because we'll convert numpy types immediately after
        error_msg = str(e)
        if 'constructor' in error_msg.lower() or 'tag' in error_msg.lower():
            # Likely a numpy type issue, try unsafe loading
            try:
                with open(config_path, 'r') as f:
                    # Try different methods depending on PyYAML version
                    if hasattr(yaml, 'unsafe_load'):
                        config = yaml.unsafe_load(f)
                    else:
                        # Fallback for older PyYAML versions
                        config = yaml.load(f, Loader=yaml.UnsafeLoader)
            except Exception as e2:
                raise ValueError(f"Could not load YAML file {config_path}: {e2}")
        else:
            # Some other error, re-raise it
            raise
    
    # Convert numpy types to native Python types
    config = convert_numpy_types_to_native(config)
    
    return config


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


def convert_numpy_types_to_native(obj):
    """
    Recursively convert numpy types to native Python types.
    This ensures configs can be saved and loaded without YAML tag issues.
    
    Args:
        obj: Object that may contain numpy types
    
    Returns:
        Object with all numpy types converted to native Python types
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types_to_native(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types_to_native(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def get_parameter_value(config, param_path):
    """
    Extract a parameter value from a configuration dictionary.
    
    Args:
        config: Configuration dictionary
        param_path: Dot-separated path to parameter (e.g., 'populations.Tumour.dynamics.lambda')
    
    Returns:
        Parameter value, or None if not found
    """
    path_parts = param_path.split('.')
    current_dict = config
    
    for part in path_parts:
        if not isinstance(current_dict, dict) or part not in current_dict:
            return None
        current_dict = current_dict[part]
    
    return current_dict


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
        
        # Set the final value (ensure it's a native Python type, not numpy)
        current_dict[path_parts[-1]] = convert_numpy_types_to_native(value)
    
    # Clean the entire config to ensure no numpy types remain
    updated_config = convert_numpy_types_to_native(updated_config)
    
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
    
    # Ensure config has no numpy types before saving
    clean_config = convert_numpy_types_to_native(config)
    
    with open(config_path, 'w') as f:
        yaml.dump(clean_config, f, default_flow_style=False, sort_keys=False)
    
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
    base_config = load_config_safely(base_config_path)
    
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
        # Ensure config has no numpy types before saving
        clean_config = convert_numpy_types_to_native(updated_config)
        with open(temp_config_path, 'w') as f:
            yaml.dump(clean_config, f, default_flow_style=False, sort_keys=False)
        
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
    base_config = load_config_safely(base_config_path)
    
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
        # Ensure config has no numpy types before saving
        clean_config = convert_numpy_types_to_native(updated_config)
        with open(temp_config_path, 'w') as f:
            yaml.dump(clean_config, f, default_flow_style=False, sort_keys=False)
        
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


def run_single_parameter_sweep(base_config_path, parameter_sweep_dict, output_dir=None, 
                               total_steps=50, save_interval=1, threshold=0.1):
    """
    Run a single-parameter sweep where each parameter is varied one at a time.
    This is useful for generating plots showing the impact of individual parameters
    on observables.
    
    Args:
        base_config_path: Path to base YAML configuration file
        parameter_sweep_dict: Dictionary defining parameter variations
                             Each key is a parameter path, each value is a list of values to sweep
                             e.g., {'populations.Tumour.dynamics.lambda': [1, 2, 3, 4, 5]}
        output_dir: Directory to save all results (defaults to laboratory/parameter_sweeps)
        total_steps: Number of simulation steps for each run
        save_interval: How often to calculate observables
        threshold: Density threshold for observable calculations
    
    Returns:
        Tuple of (csv_files, sweep_summary) where:
        - csv_files: List of paths to generated CSV files, organized by parameter
        - sweep_summary: Dictionary containing sweep metadata and results
    """
    # Set up output directory
    if output_dir is None:
        output_dir = Path(__file__).parent / "laboratory" / "parameter_sweeps"
    else:
        output_dir = Path(output_dir)
    
    # Create timestamped subdirectory for this sweep
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = output_dir / f"single_parameter_sweep_{timestamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    
    # Load base configuration
    base_config = load_config_safely(base_config_path)
    
    # OPTIMIZATION: Pre-flatten config once and cache the structure
    from run_parameter_sweep import flatten_dict
    base_flat_config = flatten_dict(base_config)
    base_config_keys = set(base_flat_config.keys())
    
    print(f"Starting single-parameter sweep...")
    print(f"Parameters to sweep: {list(parameter_sweep_dict.keys())}")
    print(f"Output directory: {sweep_dir}")
    
    # Store results organized by parameter
    csv_files_by_param = {}
    sweep_summary = {
        'base_config_path': str(base_config_path),
        'parameter_sweep_dict': parameter_sweep_dict,
        'total_steps': total_steps,
        'save_interval': save_interval,
        'threshold': threshold,
        'timestamp': timestamp,
        'results_by_parameter': {}
    }
    
    # Run sweep for each parameter
    import time
    global_run_id = 0
    
    for param_path, param_values in parameter_sweep_dict.items():
        print(f"\n{'='*60}")
        print(f"Sweeping parameter: {param_path}")
        print(f"Values to test: {param_values}")
        print(f"{'='*60}")
        
        param_csv_files = []
        param_results = []
        
        # Create subdirectory for this parameter
        param_name_safe = param_path.replace('.', '_')
        param_dir = sweep_dir / param_name_safe
        param_dir.mkdir(parents=True, exist_ok=True)
        
        for value_idx, param_value in enumerate(param_values, 1):
            global_run_id += 1
            run_start_time = time.time()
            
            print(f"\n  [{param_path}] Run {value_idx}/{len(param_values)}: {param_value}")
            
            # Create parameter dict with only this parameter varied
            param_combination = {param_path: param_value}
            
            # Update configuration with current parameter value
            t0 = time.time()
            updated_config = update_config_with_parameters(base_config, param_combination)
            config_update_time = time.time() - t0
            
            # Save config file
            t0 = time.time()
            temp_config_path = param_dir / f"temp_config_{value_idx:03d}.yaml"
            # Ensure config has no numpy types before saving
            clean_config = convert_numpy_types_to_native(updated_config)
            with open(temp_config_path, 'w') as f:
                yaml.dump(clean_config, f, default_flow_style=False, sort_keys=False)
            
            # Also save a permanent copy
            config_path = save_parameter_config(updated_config, param_dir, value_idx)
            config_save_time = time.time() - t0
            
            try:
                # Run simulation with observables
                t0 = time.time()
                csv_path = run_simulation_with_observables(
                    config_path=str(temp_config_path),
                    output_dir=param_dir,
                    total_steps=total_steps,
                    save_interval=save_interval,
                    threshold=threshold,
                    parameter_values=param_combination,
                    config=updated_config,
                    cached_flat_config_keys=base_config_keys
                )
                simulation_time = time.time() - t0
                
                # Rename CSV to include parameter value for easy identification
                t0 = time.time()
                value_str = str(param_value).replace('.', '_').replace('-', 'neg')
                combined_csv_path = param_dir / f"observables_{param_name_safe}_{value_str}.csv"
                if csv_path != str(combined_csv_path):
                    import shutil
                    shutil.move(csv_path, combined_csv_path)
                    csv_path = str(combined_csv_path)
                file_ops_time = time.time() - t0
                
                total_run_time = time.time() - run_start_time
                
                param_csv_files.append(csv_path)
                
                # Store run information
                run_info = {
                    'parameter': param_path,
                    'parameter_value': param_value,
                    'value_index': value_idx,
                    'global_run_id': global_run_id,
                    'config_path': str(config_path),
                    'csv_path': csv_path,
                    'success': True,
                    'timings': {
                        'config_update': config_update_time,
                        'config_save': config_save_time,
                        'simulation': simulation_time,
                        'file_ops': file_ops_time,
                        'total': total_run_time
                    }
                }
                param_results.append(run_info)
                
                print(f"    ✓ Completed in {total_run_time:.2f}s ({simulation_time:.2f}s simulation)")
                
            except Exception as e:
                total_run_time = time.time() - run_start_time
                print(f"    ✗ Failed: {str(e)}")
                
                # Store failed run information
                run_info = {
                    'parameter': param_path,
                    'parameter_value': param_value,
                    'value_index': value_idx,
                    'global_run_id': global_run_id,
                    'config_path': str(config_path),
                    'csv_path': None,
                    'success': False,
                    'error': str(e),
                    'timings': {
                        'total': total_run_time
                    }
                }
                param_results.append(run_info)
            
            finally:
                # Clean up temporary config file
                if temp_config_path.exists():
                    temp_config_path.unlink()
        
        # Store results for this parameter
        csv_files_by_param[param_path] = param_csv_files
        sweep_summary['results_by_parameter'][param_path] = {
            'values': param_values,
            'results': param_results
        }
        
        successful_runs = sum(1 for r in param_results if r['success'])
        print(f"\n  Parameter {param_path} sweep completed: {successful_runs}/{len(param_values)} successful")
    
    # Save sweep summary
    summary_path = sweep_dir / "single_parameter_sweep_summary.yaml"
    with open(summary_path, 'w') as f:
        yaml.dump(sweep_summary, f, default_flow_style=False, sort_keys=False)
    
    # Print final summary
    total_runs = sum(len(values) for values in parameter_sweep_dict.values())
    total_successful = sum(
        sum(1 for r in sweep_summary['results_by_parameter'][param]['results'] if r['success'])
        for param in parameter_sweep_dict.keys()
    )
    total_failed = total_runs - total_successful
    
    print(f"\n{'='*60}")
    print(f"SINGLE-PARAMETER SWEEP COMPLETED")
    print(f"{'='*60}")
    print(f"Parameters swept: {len(parameter_sweep_dict)}")
    print(f"Total runs: {total_runs}")
    print(f"Successful: {total_successful}")
    print(f"Failed: {total_failed}")
    print(f"Results saved to: {sweep_dir}")
    print(f"Summary saved to: {summary_path}")
    print(f"\nResults organized by parameter:")
    for param_path, csv_files in csv_files_by_param.items():
        print(f"  {param_path}: {len(csv_files)} CSV files in {sweep_dir / param_path.replace('.', '_')}")
    
    return csv_files_by_param, sweep_summary


def generate_local_parameter_ranges(base_config, parameter_offsets_dict, num_points_per_param=5, 
                                    use_percentage=True, random_sampling=False, num_samples=None, 
                                    random_seed=None):
    """
    Generate parameter ranges around base values for local refinement.
    
    Args:
        base_config: Base configuration dictionary
        parameter_offsets_dict: Dictionary mapping parameter paths to offsets
                               If use_percentage=True: offsets are percentages (e.g., 0.1 = ±10%)
                               If use_percentage=False: offsets are absolute values
        num_points_per_param: Number of points to generate per parameter (for grid search)
        use_percentage: If True, offsets are percentages; if False, offsets are absolute
        random_sampling: If True, use random sampling instead of grid search
        num_samples: Number of random samples (only used if random_sampling=True)
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary mapping parameter paths to lists of values to test
    """
    parameter_sweep_dict = {}
    
    for param_path, offset in parameter_offsets_dict.items():
        # Get current parameter value
        current_value = get_parameter_value(base_config, param_path)
        
        if current_value is None:
            print(f"Warning: Parameter {param_path} not found in config, skipping...")
            continue
        
        # Calculate range
        if use_percentage:
            # Percentage-based offset
            if isinstance(current_value, (int, float)):
                min_val = current_value * (1 - offset)
                max_val = current_value * (1 + offset)
            else:
                print(f"Warning: Parameter {param_path} is not numeric, skipping...")
                continue
        else:
            # Absolute offset
            if isinstance(current_value, (int, float)):
                min_val = current_value - offset
                max_val = current_value + offset
            else:
                print(f"Warning: Parameter {param_path} is not numeric, skipping...")
                continue
        
        # Generate values
        if random_sampling:
            # For random sampling, we just store the bounds
            # The actual random combinations will be generated later
            if isinstance(current_value, int):
                # Integer parameters - store bounds as tuple
                parameter_sweep_dict[param_path] = (int(min_val), int(max_val))
            else:
                # Float parameters - store bounds as tuple
                parameter_sweep_dict[param_path] = (min_val, max_val)
        else:
            # Grid search - generate list of values
            if isinstance(current_value, int):
                # Integer parameters
                values = list(np.linspace(int(min_val), int(max_val), num_points_per_param, dtype=int))
            else:
                # Float parameters
                values = list(np.linspace(min_val, max_val, num_points_per_param))
            
            # Remove duplicates and sort
            values = sorted(list(set(values)))
            parameter_sweep_dict[param_path] = values
    
    return parameter_sweep_dict


def run_local_refinement_sweep(base_config_path, parameter_offsets_dict, output_dir=None,
                               total_steps=50, save_interval=1, threshold=0.1,
                               num_points_per_param=5, use_percentage=True, 
                               random_sampling=False, num_samples=None, random_seed=None):
    """
    Run a local refinement parameter sweep around a promising configuration.
    This is useful for fine-tuning a configuration that already fits data relatively well.
    
    Args:
        base_config_path: Path to the "good" base YAML configuration file
        parameter_offsets_dict: Dictionary mapping parameter paths to offsets
                               If use_percentage=True: offsets are percentages (e.g., 0.1 = ±10%)
                               If use_percentage=False: offsets are absolute values
                               e.g., {'populations.Tumour.dynamics.lambda': 0.1} for ±10%
        output_dir: Directory to save all results (defaults to laboratory/parameter_sweeps)
        total_steps: Number of simulation steps for each run
        save_interval: How often to calculate observables
        threshold: Density threshold for observable calculations
        num_points_per_param: Number of points per parameter for grid search (if random_sampling=False)
        use_percentage: If True, offsets are percentages; if False, offsets are absolute values
        random_sampling: If True, use random sampling; if False, use grid search
        num_samples: Number of random samples per parameter (only used if random_sampling=True)
        random_seed: Random seed for reproducibility (only used if random_sampling=True)
    
    Returns:
        Tuple of (csv_files, sweep_summary) where:
        - csv_files: List of paths to generated CSV files
        - sweep_summary: Dictionary containing sweep metadata and results
    """
    # Set up output directory
    if output_dir is None:
        output_dir = Path(__file__).parent / "laboratory" / "parameter_sweeps"
    else:
        output_dir = Path(output_dir)
    
    # Create timestamped subdirectory for this sweep
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = output_dir / f"local_refinement_sweep_{timestamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    
    # Load base configuration
    base_config = load_config_safely(base_config_path)
    
    # OPTIMIZATION: Pre-flatten config once and cache the structure
    from run_parameter_sweep import flatten_dict
    base_flat_config = flatten_dict(base_config)
    base_config_keys = set(base_flat_config.keys())
    
    # Generate parameter ranges around base values
    print(f"Generating parameter ranges around base configuration...")
    print(f"Base config: {base_config_path}")
    print(f"Parameters to refine: {list(parameter_offsets_dict.keys())}")
    
    parameter_sweep_dict = generate_local_parameter_ranges(
        base_config, parameter_offsets_dict, num_points_per_param, 
        use_percentage, random_sampling, num_samples, random_seed
    )
    
    # Extract base parameter values for reference
    base_parameter_values = {}
    for param_path in parameter_offsets_dict.keys():
        base_value = get_parameter_value(base_config, param_path)
        if base_value is not None:
            base_parameter_values[param_path] = base_value
    
    print(f"\nBase parameter values:")
    for param_path, value in base_parameter_values.items():
        offset = parameter_offsets_dict[param_path]
        if use_percentage:
            print(f"  {param_path}: {value} (±{offset*100:.1f}%)")
        else:
            print(f"  {param_path}: {value} (±{offset})")
    
    print(f"\nGenerated parameter ranges:")
    if random_sampling:
        for param_path, bounds in parameter_sweep_dict.items():
            print(f"  {param_path}: bounds from {bounds[0]:.4f} to {bounds[1]:.4f}")
    else:
        for param_path, values in parameter_sweep_dict.items():
            print(f"  {param_path}: {len(values)} values from {min(values):.4f} to {max(values):.4f}")
    
    # Generate all parameter combinations
    if random_sampling:
        # For random sampling, parameter_sweep_dict already contains bounds tuples
        parameter_combinations = generate_random_parameter_combinations(
            parameter_sweep_dict,  # Already in the right format (dict of tuples)
            num_samples if num_samples else 50,
            random_seed
        )
    else:
        # For grid search, generate all combinations from value lists
        parameter_combinations = generate_parameter_combinations(parameter_sweep_dict)
    
    print(f"\nStarting local refinement sweep with {len(parameter_combinations)} combinations...")
    print(f"Output directory: {sweep_dir}")
    print(f"Sampling method: {'Random' if random_sampling else 'Grid search'}")
    
    # Store results
    csv_files = []
    sweep_summary = {
        'base_config_path': str(base_config_path),
        'base_parameter_values': base_parameter_values,
        'parameter_offsets_dict': parameter_offsets_dict,
        'use_percentage': use_percentage,
        'random_sampling': random_sampling,
        'num_points_per_param': num_points_per_param,
        'num_samples': num_samples,
        'random_seed': random_seed,
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
        
        # Save config file
        t0 = time.time()
        temp_config_path = sweep_dir / f"temp_config_run_{run_id:03d}.yaml"
        # Ensure config has no numpy types before saving
        clean_config = convert_numpy_types_to_native(updated_config)
        with open(temp_config_path, 'w') as f:
            yaml.dump(clean_config, f, default_flow_style=False, sort_keys=False)
        
        # Also save a permanent copy (save_parameter_config already cleans the config)
        config_path = save_parameter_config(updated_config, sweep_dir, run_id)
        config_save_time = time.time() - t0
        
        try:
            # Run simulation with observables
            t0 = time.time()
            csv_path = run_simulation_with_observables(
                config_path=str(temp_config_path),
                output_dir=sweep_dir,
                total_steps=total_steps,
                save_interval=save_interval,
                threshold=threshold,
                parameter_values=param_combination,
                config=clean_config,  # Use cleaned config (no numpy types)
                cached_flat_config_keys=base_config_keys
            )
            simulation_time = time.time() - t0
            
            # Rename CSV to include run_id
            t0 = time.time()
            combined_csv_path = sweep_dir / f"observables_run_{run_id:03d}.csv"
            if csv_path != str(combined_csv_path):
                import shutil
                shutil.move(csv_path, combined_csv_path)
                csv_path = combined_csv_path
            file_ops_time = time.time() - t0
            
            total_run_time = time.time() - run_start_time
            
            csv_files.append(csv_path)
            
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
            print(f"  Timing: {total_run_time:.2f}s ({simulation_time:.2f}s simulation)")
            
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
    summary_path = sweep_dir / "local_refinement_sweep_summary.yaml"
    with open(summary_path, 'w') as f:
        yaml.dump(sweep_summary, f, default_flow_style=False, sort_keys=False)
    
    # Print final summary
    successful_runs = sum(1 for result in sweep_summary['results'] if result['success'])
    failed_runs = len(sweep_summary['results']) - successful_runs
    
    print(f"\n{'='*60}")
    print(f"LOCAL REFINEMENT SWEEP COMPLETED")
    print(f"{'='*60}")
    print(f"Total runs: {len(parameter_combinations)}")
    print(f"Successful: {successful_runs}")
    print(f"Failed: {failed_runs}")
    print(f"Results saved to: {sweep_dir}")
    print(f"Summary saved to: {summary_path}")
    
    if csv_files:
        print(f"CSV files generated: {len(csv_files)}")
    
    return csv_files, sweep_summary


def example_single_parameter_sweep():
    """Example single-parameter sweep configuration."""
    
    # Base configuration file
    base_config_path = Path(__file__).parent / "configs" / "T_N.yaml"
    
    # Define parameter sweeps - each parameter gets a list of values to test
    parameter_sweep_dict = {
        # Tumor cell growth rate: sweep from 10 to 50 in steps
        'populations.Tumour.dynamics.lambda': [30.0, 40.0, 50.0, 60.0, 70.0],
        # Tumor cell death rate: sweep from 10 to 50
        'populations.Tumour.dynamics.mu': [50.0, 90.0, 130.0, 170.0, 210.0],
        # Tumor cell nutrient threshold: sweep from 0.3 to 0.99
        'populations.Tumour.dynamics.nutrient_threshold': [0.3, 0.5, 0.7, 0.9, 0.99],
        # Nutrient switch steepness: sweep from 5 to 20
        'nutrient.dynamics.k': [8.0, 13.0, 18.0, 23.0, 28.0],
    }
    
    # Run single-parameter sweep
    csv_files_by_param, summary = run_single_parameter_sweep(
        base_config_path=base_config_path,
        parameter_sweep_dict=parameter_sweep_dict,
        total_steps=10,  # Short runs for testing
        save_interval=1,
        threshold=0.1
    )
    
    print(f"\nSingle-parameter sweep completed!")
    print(f"Results organized by parameter:")
    for param_path, csv_files in csv_files_by_param.items():
        print(f"  {param_path}: {len(csv_files)} CSV files")


def local_refinement_sweep():
    """Example local refinement sweep configuration."""
    
    # Base configuration file (the "good" config that fits data relatively well)
    # This could be a config file from a previous sweep that showed promise
    base_config_path = Path(__file__).parent / "configs" / "T_N.yaml"
    
    # Define parameter offsets for local refinement
    # Using percentage offsets (0.1 = ±10% around base value)
    parameter_offsets_dict = {
        'populations.Tumour.dynamics.mu': 0.2,
        'populations.Tumour.dynamics.nutrient_consumption': 0.2,
        'populations.Tumour.dynamics.lambda': 0.2,
        'populations.Tumour.dynamics.nutrient_production': 0.2,
        'populations.Tumour.dynamics.nutrient_threshold': 0.2,
        'nutrient.dynamics.k': 0.2,
    }
    
    # Option 1: Grid search with 5 points per parameter
    print("Running grid search refinement...")
    csv_files_grid, summary_grid = run_local_refinement_sweep(
        base_config_path='/Users/rileymcnamara/CODE/2025/silicokit/laboratory/saved_simulations/983_10k_best/983_10k.yaml',
        parameter_offsets_dict=parameter_offsets_dict,
        total_steps=20,  # Short runs for testing
        save_interval=1,
        threshold=0.1,
        num_points_per_param=4,  # 5 values per parameter
        use_percentage=True,  # Use percentage offsets
        random_sampling=True  # Use grid search
    )
    
    print(f"\nGrid search completed: {len(csv_files_grid)} runs")
    
    # Option 2: Random sampling (uncomment to use instead)
    # print("\nRunning random sampling refinement...")
    # csv_files_random, summary_random = run_local_refinement_sweep(
    #     base_config_path=base_config_path,
    #     parameter_offsets_dict=parameter_offsets_dict,
    #     total_steps=20,
    #     save_interval=1,
    #     threshold=0.1,
    #     use_percentage=True,
    #     random_sampling=True,  # Use random sampling
    #     num_samples=50,  # 50 random samples
    #     random_seed=42
    # )
    # 
    # print(f"\nRandom sampling completed: {len(csv_files_random)} runs")
    
    # Option 3: Absolute offsets instead of percentages
    # parameter_offsets_dict_absolute = {
    #     'populations.Tumour.dynamics.lambda': 5.0,  # ±5.0 around base value
    #     'populations.Tumour.dynamics.mu': 5.0,
    #     'populations.Tumour.dynamics.nutrient_threshold': 0.05,  # ±0.05 around base value
    # }
    # 
    # csv_files_abs, summary_abs = run_local_refinement_sweep(
    #     base_config_path=base_config_path,
    #     parameter_offsets_dict=parameter_offsets_dict_absolute,
    #     total_steps=20,
    #     save_interval=1,
    #     threshold=0.1,
    #     num_points_per_param=5,
    #     use_percentage=False,  # Use absolute offsets
    #     random_sampling=False
    # )


def main():
    """Example random parameter sweep configuration."""
    
    # Base configuration file
    base_config_path = Path(__file__).parent / "configs" / "T_N.yaml"
    
    # Define parameter bounds for random sampling
    parameter_bounds_dict = {
        # Tumor cell growth rate: random between 2.0 and 8.0
        'populations.Tumour.dynamics.lambda': (10.0,  200.0),
        # Tumor cell death rate: random between 0.2 and 2.0
        'populations.Tumour.dynamics.mu': (10.0, 300.0),
        # Tumor cell nutrient threshold: random between 0.1 and 1.0
        'populations.Tumour.dynamics.nutrient_threshold': (0.1, 0.99),
        # Tumor cell nutrient consumption: random between 0.01 and 0.1
        'populations.Tumour.dynamics.nutrient_consumption': (0.1, 15),
        # Necrotic cell beta_N: random between 0.00001 and 0.1
        'populations.Necrotic.dynamics.beta_N': (0.00001, 0.00001),
        # Necrotic cell death rate: random between 0.1 and 30
        'populations.Necrotic.dynamics.mu': (10.0, 100.0),
        # Nutrient switch steepness: random between 0.01 and 1.0
        'nutrient.dynamics.k': (5.0, 20.0),
    }
    
    # Number of random samples to generate
    num_samples = 10000
    
    # Run random parameter sweep
    csv_files, summary = run_random_parameter_sweep(base_config_path=base_config_path,parameter_bounds_dict=parameter_bounds_dict,num_samples=num_samples,total_steps=20,)  # Short runs for testingsave_interval=1,threshold=0.1,random_seed=1000  # For reproducible results

    #local_refinement_sweep()
    #example_single_parameter_sweep()

if __name__ == "__main__":
    main()