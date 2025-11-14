#!/usr/bin/env python3
"""
Performance Diagnostic Script

This script helps identify performance bottlenecks by timing different
parts of the simulation and observables export process.
"""

import time
from pathlib import Path
import sys

if sys.platform == "darwin":
    proj = Path(__file__).parent
else:
    proj = Path(__file__).parent

sys.path.insert(0, str(proj))

from run_simulation_observables import run_simulation_with_observables
import yaml

def diagnose_performance(config_path, output_dir, total_steps=10):
    """Run a simulation with detailed timing information."""
    
    print("="*60)
    print("PERFORMANCE DIAGNOSTIC")
    print("="*60)
    print(f"Config: {config_path}")
    print(f"Total steps: {total_steps}")
    print(f"Output dir: {output_dir}")
    print("="*60)
    
    timings = {}
    
    # Time 1: Config loading
    t0 = time.time()
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    timings['config_load'] = time.time() - t0
    print(f"1. Config loading: {timings['config_load']:.3f}s")
    
    # Time 2: Simulator initialization
    t0 = time.time()
    from src.growkit.Simulator import TumorGrowthSimulator
    simulator = TumorGrowthSimulator(str(config_path))
    timings['simulator_init'] = time.time() - t0
    print(f"2. Simulator initialization: {timings['simulator_init']:.3f}s")
    
    # Time 3: Simulation run
    t0 = time.time()
    simulation_data = simulator.run_and_save_simulation(
        total_steps=total_steps,
        save_interval=1,
        save_physics_fields=True,
        output_dir=str(output_dir)
    )
    timings['simulation_run'] = time.time() - t0
    print(f"3. Simulation run: {timings['simulation_run']:.3f}s")
    print(f"   Average per step: {timings['simulation_run']/total_steps:.3f}s")
    
    # Time 4: SimPlotter initialization
    t0 = time.time()
    from src.growkit.PlotEngine.SimPlotter import SimPlotter
    plotter = SimPlotter(simulation_data)
    timings['plotter_init'] = time.time() - t0
    print(f"4. SimPlotter initialization: {timings['plotter_init']:.3f}s")
    
    # Time 5: Export observables (this calls plot_all_observables)
    t0 = time.time()
    temp_csv_path = output_dir / "observables_temp.csv"
    df = plotter.export_observables_data(output_dir=str(output_dir), filename="observables_temp.csv")
    timings['export_observables'] = time.time() - t0
    print(f"5. Export observables: {timings['export_observables']:.3f}s")
    
    # Time 6: Config flattening
    t0 = time.time()
    from run_parameter_sweep import flatten_dict
    flat_config = flatten_dict(config)
    timings['config_flatten'] = time.time() - t0
    print(f"6. Config flattening: {timings['config_flatten']:.3f}s")
    print(f"   Flattened config has {len(flat_config)} parameters")
    
    # Time 7: Adding config to DataFrame
    t0 = time.time()
    for param_name, param_value in sorted(flat_config.items()):
        col_name = f"Config_{param_name.replace('.', '_')}"
        df[col_name] = str(param_value)
    timings['add_config_to_df'] = time.time() - t0
    print(f"7. Adding config to DataFrame: {timings['add_config_to_df']:.3f}s")
    print(f"   DataFrame shape: {df.shape}")
    
    # Time 8: CSV save
    t0 = time.time()
    csv_path = output_dir / "observables_data.csv"
    df.to_csv(csv_path, index=False)
    timings['csv_save'] = time.time() - t0
    print(f"8. CSV save: {timings['csv_save']:.3f}s")
    
    # Summary
    total_time = sum(timings.values())
    print("\n" + "="*60)
    print("TIMING SUMMARY")
    print("="*60)
    for key, value in sorted(timings.items(), key=lambda x: x[1], reverse=True):
        percentage = (value / total_time) * 100
        print(f"{key:20s}: {value:8.3f}s ({percentage:5.1f}%)")
    print(f"{'TOTAL':20s}: {total_time:8.3f}s")
    print("="*60)
    
    return timings

if __name__ == "__main__":
    config_path = Path(__file__).parent / "configs" / "T_N.yaml"
    output_dir = Path(__file__).parent / "laboratory" / "performance_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    diagnose_performance(
        config_path=str(config_path),
        output_dir=output_dir,
        total_steps=10  # Short test
    )

