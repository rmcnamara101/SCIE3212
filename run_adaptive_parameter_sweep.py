#!/usr/bin/env python3
"""
Adaptive Parameter Sweep with Cost-Based Optimization

This script runs an adaptive parameter sweep that uses Bayesian optimization
(or similar adaptive sampling) to explore the parameter space efficiently.
After each simulation, it compares results to experimental data and uses the
cost (RMSE) to guide the next parameter selection, converging to better values.

Author: Riley Jae McNamara
Date: 2025-11-24
"""

from pathlib import Path
import sys
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
import copy
import time
from typing import Dict, List, Optional, Tuple

# Add project root to path
if sys.platform == "darwin":
    proj = Path(__file__).parent
else:
    proj = Path(__file__).parent

sys.path.insert(0, str(proj))

from run_simulation_observables import run_simulation_with_observables
from compare_sweep_to_data import SweepDataComparator

# Try to import scikit-optimize for Bayesian optimization
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.acquisition import gaussian_ei
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    print("Warning: scikit-optimize not available. Using simple adaptive sampling instead.")
    print("Install with: pip install scikit-optimize")


class AdaptiveParameterSweep:
    """
    Adaptive parameter sweep that uses cost-based optimization to guide parameter selection.
    """
    
    def __init__(self, base_config_path, parameter_bounds_dict, experimental_data_path=None,
                 density='10k', metrics=None, total_steps=50, save_interval=1, threshold=0.1):
        """
        Initialize the adaptive parameter sweep.
        
        Args:
            base_config_path: Path to base YAML configuration file
            parameter_bounds_dict: Dictionary with parameter paths as keys and (min, max) tuples as values
            experimental_data_path: Path to experimental data Excel file (None = use default)
            density: Experimental density to match ('10k', '5k', or '2k')
            metrics: List of metrics to compare (default: ['Total_Radius', 'Inhibited_Radius', 'Necrotic_Radius'])
            total_steps: Number of simulation steps for each run
            save_interval: How often to calculate observables
            threshold: Density threshold for observable calculations
        """
        self.base_config_path = Path(base_config_path)
        self.parameter_bounds_dict = parameter_bounds_dict
        self.density = density
        self.metrics = metrics or ['Total_Radius', 'Inhibited_Radius', 'Necrotic_Radius']
        self.total_steps = total_steps
        self.save_interval = save_interval
        self.threshold = threshold
        
        # Load base configuration
        self.base_config = yaml.safe_load(self.base_config_path.read_text())
        
        # Load experimental data
        if experimental_data_path is None:
            experimental_data_path = proj / "laboratory" / "data" / "Browning_Paper" / "Organised_Data.xlsx"
        self.experimental_data_path = Path(experimental_data_path)
        
        # Create a temporary comparator to load experimental data
        # We'll use a dummy directory for initialization
        temp_dir = Path("/tmp/adaptive_sweep_temp")
        temp_dir.mkdir(exist_ok=True)
        self.comparator = SweepDataComparator(temp_dir, experimental_data_path)
        
        # Store results
        self.results = []
        self.parameter_history = []
        self.cost_history = []
        self.best_cost = np.inf
        self.best_parameters = None
        self.best_run_id = None
        
        # Set up output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = proj / "laboratory" / "parameter_sweeps" / f"adaptive_sweep_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Adaptive parameter sweep initialized")
        print(f"  Base config: {self.base_config_path}")
        print(f"  Parameters to optimize: {list(parameter_bounds_dict.keys())}")
        print(f"  Experimental density: {density}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Bayesian optimization: {'Available' if SKOPT_AVAILABLE else 'Not available (using simple adaptive)'}")
    
    def _update_config_with_parameters(self, base_config, parameter_dict):
        """Update configuration with parameter values."""
        from run_parameter_sweep import update_config_with_parameters
        return update_config_with_parameters(base_config, parameter_dict)
    
    def _calculate_cost(self, observables_df):
        """
        Calculate cost (RMSE) for a simulation by comparing to experimental data.
        
        Args:
            observables_df: DataFrame with observables (Time, Step, Total_Radius, etc.)
        
        Returns:
            Combined RMSE across all metrics
        """
        # Get experimental data for the target density
        if self.density not in self.comparator.experimental_data:
            return np.inf
        
        exp_data = self.comparator.experimental_data[self.density]
        
        # Calculate optimal scale across all metrics
        scale = self.comparator.calculate_optimal_scale_all_metrics(
            observables_df, exp_data, self.metrics
        )
        
        # Calculate RMSE for each metric
        metric_rmses = []
        for metric in self.metrics:
            if metric not in observables_df.columns or metric not in exp_data.columns:
                continue
            
            sim_steps = observables_df['Step'].values
            sim_values = observables_df[metric].values
            exp_days = exp_data['Day'].values
            exp_values = exp_data[metric].values
            
            rmse, _ = self.comparator.calculate_rmse(
                sim_steps, sim_values, exp_days, exp_values, scale=scale
            )
            metric_rmses.append(rmse)
        
        # Return combined RMSE (average across metrics)
        if metric_rmses:
            return np.mean(metric_rmses)
        else:
            return np.inf
    
    def _run_simulation_and_calculate_cost(self, parameter_dict, run_id):
        """
        Run a simulation with given parameters and calculate its cost.
        
        Args:
            parameter_dict: Dictionary of parameter values
            run_id: Run identifier
        
        Returns:
            Tuple of (cost, observables_df, config_path)
        """
        # Update configuration
        updated_config = self._update_config_with_parameters(self.base_config, parameter_dict)
        
        # Save temporary config
        temp_config_path = self.output_dir / f"temp_config_run_{run_id:03d}.yaml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(updated_config, f, default_flow_style=False, sort_keys=False)
        
        try:
            # Run simulation
            csv_path = run_simulation_with_observables(
                config_path=str(temp_config_path),
                output_dir=str(self.output_dir),
                total_steps=self.total_steps,
                save_interval=self.save_interval,
                threshold=self.threshold,
                parameter_values=parameter_dict,
                config=updated_config
            )
            
            # Load observables
            # The CSV now has metadata at top, pandas will automatically skip comment lines
            try:
                observables_df = pd.read_csv(csv_path, comment='#')
            except Exception as e:
                # Fallback: manually find observables section
                with open(csv_path, 'r') as f:
                    lines = f.readlines()
                
                # Find where observables data starts (after "# Observables Data" section)
                obs_start_idx = None
                for i, line in enumerate(lines):
                    if line.strip().startswith('# Observables Data'):
                        # Find the next non-comment, non-blank line (should be header)
                        for j in range(i+1, len(lines)):
                            if lines[j].strip() and not lines[j].strip().startswith('#'):
                                obs_start_idx = j
                                break
                        break
                
                if obs_start_idx is not None:
                    observables_df = pd.read_csv(csv_path, skiprows=obs_start_idx)
                else:
                    raise ValueError(f"Could not parse CSV file: {csv_path}")
            
            # Calculate cost
            cost = self._calculate_cost(observables_df)
            
            # Rename CSV to include run_id
            final_csv_path = self.output_dir / f"observables_run_{run_id:03d}.csv"
            if csv_path != final_csv_path:
                import shutil
                shutil.move(csv_path, final_csv_path)
            
            # Save config with run_id
            config_path = self.output_dir / f"config_run_{run_id:03d}.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(updated_config, f, default_flow_style=False, sort_keys=False)
            
            return cost, observables_df, str(config_path)
            
        except Exception as e:
            print(f"  Error running simulation: {e}")
            return np.inf, None, None
        finally:
            # Clean up temporary config
            if temp_config_path.exists():
                temp_config_path.unlink()
    
    def _objective_function(self, parameter_values):
        """
        Objective function for optimization (minimize cost).
        
        Args:
            parameter_values: List of parameter values in the same order as parameter_bounds_dict keys
        
        Returns:
            Cost (RMSE) to minimize
        """
        # Convert parameter values to dictionary
        param_names = list(self.parameter_bounds_dict.keys())
        parameter_dict = dict(zip(param_names, parameter_values))
        
        # Get next run_id
        run_id = len(self.results) + 1
        
        print(f"\n  Evaluating parameters (run {run_id}):")
        for name, value in parameter_dict.items():
            print(f"    {name}: {value}")
        
        # Run simulation and calculate cost
        cost, observables_df, config_path = self._run_simulation_and_calculate_cost(
            parameter_dict, run_id
        )
        
        # Store result
        result = {
            'run_id': run_id,
            'parameters': parameter_dict.copy(),
            'cost': cost,
            'config_path': config_path,
            'observables_df': observables_df
        }
        self.results.append(result)
        self.parameter_history.append(parameter_values)
        self.cost_history.append(cost)
        
        # Update best
        if cost < self.best_cost:
            self.best_cost = cost
            self.best_parameters = parameter_dict.copy()
            self.best_run_id = run_id
            print(f"  ✓ New best cost: {cost:.4f}")
        else:
            print(f"  Cost: {cost:.4f} (best: {self.best_cost:.4f})")
        
        return cost
    
    def _simple_adaptive_sampling(self, n_calls, n_initial=10):
        """
        Simple adaptive sampling strategy when Bayesian optimization is not available.
        Uses a combination of random sampling and local search around best points.
        
        Args:
            n_calls: Total number of function evaluations
            n_initial: Number of initial random samples
        """
        param_names = list(self.parameter_bounds_dict.keys())
        n_params = len(param_names)
        
        # Track evaluated parameter sets to avoid exact duplicates
        # Use rounded values for comparison to handle floating point precision
        evaluated_params = set()
        tolerance = 1e-6
        
        def _params_to_key(param_values):
            """Convert parameter values to a key for duplicate checking."""
            # Round to avoid floating point precision issues
            rounded = []
            for i, val in enumerate(param_values):
                param_name = param_names[i]
                min_val, max_val = self.parameter_bounds_dict[param_name]
                if isinstance(min_val, int) and isinstance(max_val, int):
                    rounded.append(int(val))
                else:
                    # Round floats to 6 decimal places for comparison
                    rounded.append(round(val, 6))
            return tuple(rounded)
        
        def _generate_unique_params(explore_globally=True):
            """Generate unique parameter values."""
            max_attempts = 100
            for attempt in range(max_attempts):
                if explore_globally or self.best_parameters is None:
                    # Global exploration
                    param_values = []
                    for name in param_names:
                        min_val, max_val = self.parameter_bounds_dict[name]
                        if isinstance(min_val, int) and isinstance(max_val, int):
                            value = np.random.randint(min_val, max_val + 1)
                        else:
                            value = np.random.uniform(min_val, max_val)
                        param_values.append(value)
                else:
                    # Local exploitation around best point
                    param_values = []
                    for name in param_names:
                        min_val, max_val = self.parameter_bounds_dict[name]
                        best_value = self.best_parameters[name]
                        
                        # Create a shrinking search radius
                        iteration = len(self.results) - n_initial
                        radius = (max_val - min_val) * 0.15 * (1.0 - iteration / max(n_calls - n_initial, 1))
                        radius = max(radius, (max_val - min_val) * 0.02)  # Minimum radius
                        
                        if isinstance(min_val, int) and isinstance(max_val, int):
                            # For integers, use discrete steps
                            step = max(1, int(radius))
                            value = np.random.randint(
                                max(min_val, int(best_value) - step),
                                min(max_val + 1, int(best_value) + step + 1)
                            )
                        else:
                            # For floats, use Gaussian perturbation
                            value = best_value + np.random.normal(0, radius)
                            value = np.clip(value, min_val, max_val)
                        
                        param_values.append(value)
                
                # Check if we've seen this parameter set before
                param_key = _params_to_key(param_values)
                if param_key not in evaluated_params:
                    evaluated_params.add(param_key)
                    return param_values
            
            # If we couldn't generate unique params after max attempts, return anyway
            # (shouldn't happen in practice)
            print(f"  Warning: Could not generate unique parameters after {max_attempts} attempts")
            return param_values
        
        # Initial random sampling
        print(f"\nPhase 1: Random exploration ({n_initial} samples)")
        for i in range(n_initial):
            param_values = _generate_unique_params(explore_globally=True)
            self._objective_function(param_values)
        
        # Adaptive sampling around best points
        print(f"\nPhase 2: Adaptive refinement ({n_calls - n_initial} samples)")
        for i in range(n_calls - n_initial):
            # With probability 0.3, explore globally (random)
            # With probability 0.7, exploit locally (around best point)
            explore_globally = np.random.random() < 0.3 or self.best_parameters is None
            param_values = _generate_unique_params(explore_globally=explore_globally)
            self._objective_function(param_values)
    
    def run(self, n_calls=50, n_initial=None, random_seed=None):
        """
        Run the adaptive parameter sweep.
        
        Args:
            n_calls: Total number of function evaluations
            n_initial: Number of initial random samples (default: min(10, n_calls//2))
            random_seed: Random seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        if n_initial is None:
            n_initial = min(10, n_calls // 2)
        
        print(f"\n{'='*80}")
        print(f"Starting adaptive parameter sweep")
        print(f"{'='*80}")
        print(f"Total evaluations: {n_calls}")
        print(f"Initial random samples: {n_initial}")
        print(f"Adaptive samples: {n_calls - n_initial}")
        print(f"{'='*80}\n")
        
        start_time = time.time()
        
        if SKOPT_AVAILABLE:
            # Use Bayesian optimization with Gaussian Process
            param_names = list(self.parameter_bounds_dict.keys())
            dimensions = []
            
            for name in param_names:
                min_val, max_val = self.parameter_bounds_dict[name]
                if isinstance(min_val, int) and isinstance(max_val, int):
                    dimensions.append(Integer(min_val, max_val, name=name))
                else:
                    dimensions.append(Real(min_val, max_val, name=name))
            
            print("Using Bayesian optimization (Gaussian Process)...")
            result = gp_minimize(
                func=self._objective_function,
                dimensions=dimensions,
                n_calls=n_calls,
                n_initial_points=n_initial,
                random_state=random_seed,
                acq_func='EI',  # Expected Improvement
                verbose=True
            )
            
            # Store optimization result
            self.optimization_result = result
            print(f"\nOptimization complete!")
            print(f"Best cost: {result.fun:.4f}")
            print(f"Best parameters:")
            for name, value in zip(param_names, result.x):
                print(f"  {name}: {value}")
        else:
            # Use simple adaptive sampling
            print("Using simple adaptive sampling...")
            self._simple_adaptive_sampling(n_calls, n_initial)
        
        total_time = time.time() - start_time
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"ADAPTIVE PARAMETER SWEEP COMPLETED")
        print(f"{'='*80}")
        print(f"Total evaluations: {len(self.results)}")
        print(f"Best cost (RMSE): {self.best_cost:.4f}")
        print(f"Best run ID: {self.best_run_id}")
        print(f"Best parameters:")
        for name, value in self.best_parameters.items():
            print(f"  {name}: {value}")
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*80}\n")
        
        # Save summary
        self._save_summary()
        
        return self.results
    
    def _save_summary(self):
        """Save summary of results to YAML file."""
        summary = {
            'base_config_path': str(self.base_config_path),
            'parameter_bounds': self.parameter_bounds_dict,
            'density': self.density,
            'metrics': self.metrics,
            'total_steps': self.total_steps,
            'save_interval': self.save_interval,
            'threshold': self.threshold,
            'best_cost': float(self.best_cost),
            'best_run_id': self.best_run_id,
            'best_parameters': self.best_parameters,
            'results': []
        }
        
        for result in self.results:
            summary['results'].append({
                'run_id': result['run_id'],
                'parameters': result['parameters'],
                'cost': float(result['cost']),
                'config_path': result['config_path']
            })
        
        summary_path = self.output_dir / "adaptive_sweep_summary.yaml"
        with open(summary_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False, sort_keys=False)
        
        # Also save results as CSV
        results_data = []
        for result in self.results:
            row = {
                'run_id': result['run_id'],
                'cost': result['cost']
            }
            row.update(result['parameters'])
            results_data.append(row)
        
        results_df = pd.DataFrame(results_data)
        results_csv_path = self.output_dir / "adaptive_sweep_results.csv"
        results_df.to_csv(results_csv_path, index=False)
        
        print(f"Summary saved to: {summary_path}")
        print(f"Results CSV saved to: {results_csv_path}")


def main():
    """Example adaptive parameter sweep configuration."""
    
    # Base configuration file
    base_config_path = Path(__file__).parent / "configs" / "T_N.yaml"
    
    # Define parameter bounds for optimization
    parameter_bounds_dict = {
        'populations.Tumour.dynamics.lambda': (10.0, 50.0),
        'populations.Tumour.dynamics.mu': (10.0, 50.0),
        'populations.Tumour.dynamics.nutrient_threshold': (0.3, 0.99),
        'populations.Tumour.dynamics.nutrient_consumption': (0.1, 5),
        'populations.Necrotic.dynamics.beta_N': (0.0001, 0.01),
        'populations.Necrotic.dynamics.mu': (10.0, 50.0),
    }
    
    # Create adaptive sweep
    sweep = AdaptiveParameterSweep(
        base_config_path=base_config_path,
        parameter_bounds_dict=parameter_bounds_dict,
        density='10k',  # Match 10k experimental data
        total_steps=20,  # Short runs for testing
        save_interval=1,
        threshold=0.1
    )
    
    # Run adaptive sweep
    results = sweep.run(
        n_calls=30,  # Total number of evaluations
        n_initial=10,  # Initial random samples
        random_seed=42
    )
    
    print(f"\nAdaptive parameter sweep completed with {len(results)} evaluations")


if __name__ == "__main__":
    main()

