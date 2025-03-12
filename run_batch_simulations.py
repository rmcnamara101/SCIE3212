#!/usr/bin/env python3
"""
Batch processing script to run multiple tumor growth simulations in parallel.
"""
import os
import sys
import numpy as np
import time
from joblib import Parallel, delayed
import argparse

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.tumor_growth import TumorGrowthModel
from src.models.initial_conditions import SphericalTumor
from src.utils.utils import SCIE3121_params
from src.models.SCIE3121_model import SCIE3121_MODEL

def run_single_simulation(config):
    """Run a single simulation with the given configuration."""
    print(f"Starting simulation: {config['name']}")
    start_time = time.time()
    
    initial_conditions = SphericalTumor(
        config['grid_shape'], 
        radius=config['radius'], 
        nutrient_value=config['nutrient_value']
    )
    
    model = SCIE3121_MODEL(
        grid_shape=config['grid_shape'],
        dx=config['dx'],
        dt=config['dt'],
        params=config['params'],
        initial_conditions=initial_conditions,
        save_steps=config['save_steps']
    )
    
    model.run_and_save_simulation(steps=config['steps'], name=config['name'])
    
    elapsed_time = time.time() - start_time
    print(f"Completed simulation: {config['name']} in {elapsed_time:.2f} seconds")
    return elapsed_time

def main():
    parser = argparse.ArgumentParser(description='Run multiple tumor growth simulations in parallel')
    parser.add_argument('--jobs', type=int, default=-1, help='Number of parallel jobs (-1 for all cores)')
    args = parser.parse_args()
    
    # Define simulation configurations
    base_params = SCIE3121_params.copy()
    
    configs = []
    
    # Example 1: Base configuration
    configs.append({
        'name': 'base_model',
        'grid_shape': (40, 40, 40),
        'dx': 1,
        'dt': 0.01,
        'params': base_params.copy(),
        'steps': 1000,
        'save_steps': 50,
        'radius': 4,
        'nutrient_value': 1.0
    })
    
    # Example 2: Different grid sizes
    for size in [30, 50, 60]:
        configs.append({
            'name': f'grid_size_{size}',
            'grid_shape': (size, size, size),
            'dx': 1,
            'dt': 0.01,
            'params': base_params.copy(),
            'steps': 1000,
            'save_steps': 50,
            'radius': 4,
            'nutrient_value': 1.0
        })
    
    # Add more configurations as needed...
    
    # Run simulations in parallel
    print(f"Running {len(configs)} simulations in parallel")
    start_time = time.time()
    
    results = Parallel(n_jobs=args.jobs)(
        delayed(run_single_simulation)(config) for config in configs
    )
    
    total_time = time.time() - start_time
    print(f"All simulations completed in {total_time:.2f} seconds")
    print(f"Average simulation time: {sum(results)/len(results):.2f} seconds")

if __name__ == "__main__":
    main() 