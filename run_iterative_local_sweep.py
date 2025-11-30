#!/usr/bin/env python3
"""
Iterative Local Refinement Parameter Sweep

This script automates the process of running local refinement sweeps iteratively:
1. Run a local refinement sweep around a base config
2. Compare results to experimental data to find the best RMSE
3. If a new best RMSE is found, use that config as the base for the next sweep
4. Repeat for a specified number of iterations

Author: Riley Jae McNamara
Date: 2025-11-29
"""

from pathlib import Path
import sys
import yaml
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List

# Add project root to path
if sys.platform == "darwin":
    proj = Path(__file__).parent
else:
    proj = Path(__file__).parent

sys.path.insert(0, str(proj))

from run_parameter_sweep import run_local_refinement_sweep, load_config_safely
from compare_sweep_to_data import SweepDataComparator


class IterativeLocalSweep:
    """
    Iterative local refinement sweep that automatically refines around best configurations.
    """
    
    def __init__(self, 
                 initial_config_path: str,
                 parameter_offsets_dict: Dict[str, float],
                 dataset: str = '10k',
                 experimental_data_path: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 total_steps: int = 50,
                 save_interval: int = 1,
                 threshold: float = 0.1,
                 num_points_per_param: int = 5,
                 use_percentage: bool = True,
                 random_sampling: bool = False,
                 num_samples: Optional[int] = None,
                 random_seed: Optional[int] = None):
        """
        Initialize iterative local refinement sweep.
        
        Args:
            initial_config_path: Path to initial base YAML configuration file
            parameter_offsets_dict: Dictionary mapping parameter paths to offsets
                                     If use_percentage=True: offsets are percentages (e.g., 0.1 = ±10%)
                                     If use_percentage=False: offsets are absolute values
            dataset: Experimental dataset to match ('10k', '5k', or '2.5k')
            experimental_data_path: Path to experimental data Excel file (None = use default)
            output_dir: Base directory for all sweep outputs (None = use default)
            total_steps: Number of simulation steps for each run
            save_interval: How often to calculate observables
            threshold: Density threshold for observable calculations
            num_points_per_param: Number of points per parameter for grid search
            use_percentage: If True, offsets are percentages; if False, offsets are absolute
            random_sampling: If True, use random sampling; if False, use grid search
            num_samples: Number of random samples (only used if random_sampling=True)
            random_seed: Random seed for reproducibility
        """
        self.initial_config_path = Path(initial_config_path)
        self.parameter_offsets_dict = parameter_offsets_dict
        self.dataset = dataset
        self.total_steps = total_steps
        self.save_interval = save_interval
        self.threshold = threshold
        self.num_points_per_param = num_points_per_param
        self.use_percentage = use_percentage
        self.random_sampling = random_sampling
        self.num_samples = num_samples
        self.random_seed = random_seed
        
        # Set up output directory
        if output_dir is None:
            output_dir = proj / "laboratory" / "parameter_sweeps"
        else:
            output_dir = Path(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_output_dir = output_dir
        self.iterative_sweep_dir = output_dir / f"iterative_local_sweep_{timestamp}"
        self.iterative_sweep_dir.mkdir(parents=True, exist_ok=True)
        
        # Experimental data path
        if experimental_data_path is None:
            experimental_data_path = proj / "laboratory" / "data" / "Browning_Paper" / "Organised_Data.xlsx"
        self.experimental_data_path = Path(experimental_data_path)
        
        # Track best RMSE across all iterations
        self.best_rmse = np.inf
        self.best_config_path = None
        self.best_iteration = None
        self.best_run_id = None
        
        # Track iteration history
        self.iteration_history = []
        
        print(f"{'='*80}")
        print(f"ITERATIVE LOCAL REFINEMENT SWEEP")
        print(f"{'='*80}")
        print(f"Initial config: {self.initial_config_path}")
        print(f"Dataset: {dataset}")
        print(f"Parameters to refine: {list(parameter_offsets_dict.keys())}")
        print(f"Output directory: {self.iterative_sweep_dir}")
        print(f"{'='*80}\n")
    
    def _find_best_config_from_sweep(self, sweep_dir: Path) -> Optional[Path]:
        """
        Find the best config file from a sweep directory by comparing to experimental data.
        
        Args:
            sweep_dir: Path to sweep directory containing CSV files and config files
        
        Returns:
            Path to best config file, or None if no valid results found
        """
        print(f"\nFinding best config from sweep: {sweep_dir}")
        
        # Create comparator for this sweep
        comparator = SweepDataComparator(
            sweep_dir=str(sweep_dir),
            experimental_data_path=str(self.experimental_data_path)
        )
        
        # Find best matches
        best_matches = comparator.find_best_matches(
            density=self.dataset,
            top_n=1,
            metrics=['Total_Radius', 'Inhibited_Radius', 'Necrotic_Radius']
        )
        
        if len(best_matches) == 0:
            print("  No valid matches found")
            return None
        
        best_match = best_matches[0]
        best_rmse = best_match['combined_rmse']
        best_run_id = best_match['run_id']
        
        print(f"  Best RMSE: {best_rmse:.4f}")
        print(f"  Best run ID: {best_run_id}")
        
        # Get config path from simulation data
        if best_run_id not in comparator.simulation_data:
            print(f"  Warning: Run ID {best_run_id} not found in simulation data")
            return None
        
        # Try to find config file path
        sim_data = comparator.simulation_data[best_run_id]
        
        # Check if config_path is stored in the data
        if 'config_path' in sim_data:
            config_path = Path(sim_data['config_path'])
            if config_path.exists():
                return config_path
        
        # Fallback: construct config path from run_id
        # Config files are named config_run_XXX.yaml
        if isinstance(best_run_id, int):
            config_filename = f"config_run_{best_run_id:03d}.yaml"
        elif isinstance(best_run_id, str):
            # Handle string run_ids (e.g., "sweep_dir_run_001")
            if '_run_' in best_run_id:
                # Extract the run number
                parts = best_run_id.split('_run_')
                if len(parts) > 1:
                    run_num = parts[-1]
                    try:
                        run_num_int = int(run_num)
                        config_filename = f"config_run_{run_num_int:03d}.yaml"
                    except ValueError:
                        # Try to find any config file in the sweep directory
                        config_files = list(sweep_dir.glob("config_run_*.yaml"))
                        if config_files:
                            # Use the first one as fallback (not ideal, but better than nothing)
                            print(f"  Warning: Could not parse run_id, using first config file found")
                            return config_files[0]
                        return None
            else:
                # Try to find config files
                config_files = list(sweep_dir.glob("config_run_*.yaml"))
                if config_files:
                    print(f"  Warning: Could not determine config from run_id, using first config file found")
                    return config_files[0]
                return None
        else:
            return None
        
        config_path = sweep_dir / config_filename
        if config_path.exists():
            return config_path
        else:
            print(f"  Warning: Config file not found: {config_path}")
            # Try to find any config file
            config_files = list(sweep_dir.glob("config_run_*.yaml"))
            if config_files:
                print(f"  Using first available config file: {config_files[0]}")
                return config_files[0]
            return None
    
    def _run_single_iteration(self, iteration: int, base_config_path: Path) -> Dict:
        """
        Run a single iteration of local refinement sweep.
        
        Args:
            iteration: Iteration number (1-indexed)
            base_config_path: Path to base config for this iteration
        
        Returns:
            Dictionary with iteration results
        """
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration}")
        print(f"{'='*80}")
        print(f"Base config: {base_config_path}")
        
        # Run local refinement sweep
        csv_files, sweep_summary = run_local_refinement_sweep(
            base_config_path=str(base_config_path),
            parameter_offsets_dict=self.parameter_offsets_dict,
            output_dir=str(self.base_output_dir),
            total_steps=self.total_steps,
            save_interval=self.save_interval,
            threshold=self.threshold,
            num_points_per_param=self.num_points_per_param,
            use_percentage=self.use_percentage,
            random_sampling=self.random_sampling,
            num_samples=self.num_samples,
            random_seed=self.random_seed
        )
        
        # Get sweep directory from summary
        # The sweep directory is created with timestamp in run_local_refinement_sweep
        timestamp = sweep_summary.get('timestamp', '')
        if timestamp:
            sweep_dir = self.base_output_dir / f"local_refinement_sweep_{timestamp}"
        else:
            # Fallback: use the most recent local_refinement_sweep directory
            sweep_dirs = sorted(self.base_output_dir.glob("local_refinement_sweep_*"), 
                               key=lambda x: x.stat().st_mtime, reverse=True)
            if sweep_dirs:
                sweep_dir = sweep_dirs[0]
                print(f"  Using most recent sweep directory: {sweep_dir}")
            else:
                print(f"  Error: Could not find sweep directory")
                return {
                    'iteration': iteration,
                    'sweep_dir': None,
                    'best_rmse': np.inf,
                    'best_config_path': None,
                    'improved': False
                }
        
        if not sweep_dir.exists():
            print(f"  Error: Sweep directory does not exist: {sweep_dir}")
            return {
                'iteration': iteration,
                'sweep_dir': None,
                'best_rmse': np.inf,
                'best_config_path': None,
                'improved': False
            }
        
        # Find best config from this sweep
        best_config_path = self._find_best_config_from_sweep(sweep_dir)
        
        if best_config_path is None:
            print(f"  Error: Could not find best config from sweep")
            return {
                'iteration': iteration,
                'sweep_dir': str(sweep_dir),
                'best_rmse': np.inf,
                'best_config_path': None,
                'improved': False
            }
        
        # Compare to find best RMSE
        comparator = SweepDataComparator(
            sweep_dir=str(sweep_dir),
            experimental_data_path=str(self.experimental_data_path)
        )
        
        best_matches = comparator.find_best_matches(
            density=self.dataset,
            top_n=1,
            metrics=['Total_Radius', 'Inhibited_Radius', 'Necrotic_Radius']
        )
        
        if len(best_matches) == 0:
            print(f"  Error: No valid matches found")
            return {
                'iteration': iteration,
                'sweep_dir': str(sweep_dir),
                'best_rmse': np.inf,
                'best_config_path': str(best_config_path),
                'improved': False
            }
        
        best_match = best_matches[0]
        iteration_best_rmse = best_match['combined_rmse']
        
        # Check if this is a new best
        improved = iteration_best_rmse < self.best_rmse
        
        if improved:
            print(f"\n  ✓ NEW BEST RMSE: {iteration_best_rmse:.4f} (previous: {self.best_rmse:.4f})")
            self.best_rmse = iteration_best_rmse
            self.best_config_path = best_config_path
            self.best_iteration = iteration
            self.best_run_id = best_match['run_id']
        else:
            print(f"\n  Best RMSE this iteration: {iteration_best_rmse:.4f} (overall best: {self.best_rmse:.4f})")
        
        return {
            'iteration': iteration,
            'sweep_dir': str(sweep_dir),
            'best_rmse': iteration_best_rmse,
            'best_config_path': str(best_config_path),
            'improved': improved,
            'best_run_id': best_match.get('run_id')
        }
    
    def run(self, max_iterations: int = 10, min_improvement: float = 0.0, 
            stop_if_no_improvement: bool = False) -> List[Dict]:
        """
        Run iterative local refinement sweeps.
        
        Args:
            max_iterations: Maximum number of iterations to run
            min_improvement: Minimum improvement in RMSE to continue (default: 0.0 = always continue)
            stop_if_no_improvement: Stop if an iteration doesn't improve (default: False)
        
        Returns:
            List of iteration results
        """
        print(f"\n{'='*80}")
        print(f"STARTING ITERATIVE LOCAL REFINEMENT")
        print(f"{'='*80}")
        print(f"Max iterations: {max_iterations}")
        print(f"Min improvement threshold: {min_improvement}")
        print(f"Stop if no improvement: {stop_if_no_improvement}")
        print(f"{'='*80}\n")
        
        # Start with initial config
        current_config_path = self.initial_config_path
        
        # Run iterations
        for iteration in range(1, max_iterations + 1):
            # Run single iteration
            result = self._run_single_iteration(iteration, current_config_path)
            self.iteration_history.append(result)
            
            # Check if we should continue
            if result['improved']:
                # Use the best config from this iteration for the next iteration
                if result['best_config_path']:
                    current_config_path = Path(result['best_config_path'])
                    improvement = self.best_rmse - result['best_rmse'] if iteration > 1 else np.inf
                    
                    if improvement < min_improvement and min_improvement > 0:
                        print(f"\n  Stopping: Improvement ({improvement:.4f}) below threshold ({min_improvement:.4f})")
                        break
                else:
                    print(f"\n  Warning: No best config path, using previous config")
            else:
                if stop_if_no_improvement:
                    print(f"\n  Stopping: No improvement in iteration {iteration}")
                    break
                else:
                    # Continue with same config (might find better results with different sampling)
                    print(f"\n  Continuing with same base config (no improvement this iteration)")
        
        # Print final summary
        self._print_summary()
        
        # Save summary
        self._save_summary()
        
        return self.iteration_history
    
    def _print_summary(self):
        """Print summary of all iterations."""
        print(f"\n{'='*80}")
        print(f"ITERATIVE SWEEP SUMMARY")
        print(f"{'='*80}")
        print(f"Total iterations: {len(self.iteration_history)}")
        print(f"Best RMSE: {self.best_rmse:.4f}")
        print(f"Best iteration: {self.best_iteration}")
        print(f"Best config: {self.best_config_path}")
        print(f"Best run ID: {self.best_run_id}")
        print(f"\nIteration history:")
        print(f"{'Iter':<6} {'RMSE':<12} {'Improved':<10} {'Config Path'}")
        print(f"{'-'*80}")
        
        for result in self.iteration_history:
            improved_str = "Yes" if result['improved'] else "No"
            config_name = Path(result['best_config_path']).name if result['best_config_path'] else "N/A"
            print(f"{result['iteration']:<6} {result['best_rmse']:<12.4f} {improved_str:<10} {config_name}")
        
        print(f"{'='*80}\n")
    
    def _save_summary(self):
        """Save summary to YAML file."""
        summary = {
            'initial_config_path': str(self.initial_config_path),
            'parameter_offsets_dict': self.parameter_offsets_dict,
            'dataset': self.dataset,
            'total_steps': self.total_steps,
            'save_interval': self.save_interval,
            'threshold': self.threshold,
            'num_points_per_param': self.num_points_per_param,
            'use_percentage': self.use_percentage,
            'random_sampling': self.random_sampling,
            'num_samples': self.num_samples,
            'random_seed': self.random_seed,
            'best_rmse': float(self.best_rmse),
            'best_iteration': self.best_iteration,
            'best_config_path': str(self.best_config_path) if self.best_config_path else None,
            'best_run_id': str(self.best_run_id) if self.best_run_id else None,
            'iterations': self.iteration_history
        }
        
        summary_path = self.iterative_sweep_dir / "iterative_sweep_summary.yaml"
        with open(summary_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False, sort_keys=False)
        
        print(f"Summary saved to: {summary_path}")


def main():
    """
    Iterative local refinement sweep configuration.
    
    Modify the variables below to customize the sweep.
    """
    # ============================================================================
    # CONFIGURATION - Modify these to customize the sweep
    # ============================================================================
    
    # Path to initial base configuration file
    # Example: proj / "laboratory" / "saved_simulations" / "983_2k_best" / "983_2k.yaml"
    # Or: proj / "configs" / "T_N.yaml"
    # Or: proj / "laboratory" / "parameter_sweeps" / "local_refinement_swee®p_20251129_180026" / "config_run_004.yaml"
    INITIAL_CONFIG = "/Users/rileymcnamara/CODE/2025/silicokit/laboratory/parameter_sweeps/local_refinement_sweep_20251129_180026/config_run_026.yaml"
    
    # Experimental dataset to match ('10k', '5k', or '2.5k')
    # This determines which experimental data to compare against
    DATASET = '2.5k'  # Options: '10k', '5k', '2.5k'
    
    # Maximum number of iterations
    MAX_ITERATIONS = 5
    
    # Parameter offsets for local refinement
    # Using percentage offsets (0.1 = ±10% around base value)
    # You can specify different offsets for different parameters
    PARAMETER_OFFSETS = {
        'populations.Tumour.dynamics.mu': 0.3,
        'populations.Tumour.dynamics.lambda': 0.3,
        'populations.Tumour.dynamics.nutrient_consumption': 0.3,
        'populations.Tumour.dynamics.nutrient_threshold': 0.3,
        'populations.Tumour.dynamics.nutrient_production': 0.3,
        'populations.Necrotic.dynamics.beta_N': 0.3,
        'nutrient.dynamics.k': 0.3,
    }
    
    # Simulation parameters
    TOTAL_STEPS = 20  # Number of simulation steps
    SAVE_INTERVAL = 1
    THRESHOLD = 0.1
    
    # Local sweep parameters
    NUM_POINTS_PER_PARAM = 4  # Number of points per parameter for grid search
    USE_PERCENTAGE = True  # If True, offsets are percentages; if False, offsets are absolute values
    RANDOM_SAMPLING = True  # If True, use random sampling; if False, use grid search
    NUM_SAMPLES = 50  # Number of random samples (only used if RANDOM_SAMPLING=True)
    RANDOM_SEED = None  # Set to an integer for reproducibility (None = random)
    
    # Iteration control
    STOP_IF_NO_IMPROVEMENT = False  # Stop if an iteration doesn't improve
    MIN_IMPROVEMENT = 0.0  # Minimum improvement in RMSE to continue (0.0 = always continue)
    
    # ============================================================================
    # END OF CONFIGURATION
    # ============================================================================
    
    # Create iterative sweep
    iterative_sweep = IterativeLocalSweep(
        initial_config_path=str(INITIAL_CONFIG),
        parameter_offsets_dict=PARAMETER_OFFSETS,
        dataset=DATASET,
        total_steps=TOTAL_STEPS,
        save_interval=SAVE_INTERVAL,
        threshold=THRESHOLD,
        num_points_per_param=NUM_POINTS_PER_PARAM,
        use_percentage=USE_PERCENTAGE,
        random_sampling=RANDOM_SAMPLING,
        num_samples=NUM_SAMPLES,
        random_seed=RANDOM_SEED
    )
    
    # Run iterative sweeps
    results = iterative_sweep.run(
        max_iterations=MAX_ITERATIONS,
        min_improvement=MIN_IMPROVEMENT,
        stop_if_no_improvement=STOP_IF_NO_IMPROVEMENT
    )
    
    print(f"\nIterative local refinement completed!")
    print(f"Best RMSE achieved: {iterative_sweep.best_rmse:.4f}")
    print(f"Best config: {iterative_sweep.best_config_path}")


if __name__ == "__main__":
    main()

