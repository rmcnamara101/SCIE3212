"""
Experiment Runner Module

Core class for running systematic experiments on tumor growth simulations.
Handles experiment configuration, execution, and result collection.

Author: Riley Jae McNamara
Date: 2025-02-19
"""

import os
import time
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from src.growkit.Simulator import TumorGrowthSimulator


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    name: str
    parameters: Dict[str, Any]
    total_steps: int
    save_interval: int = 1
    save_physics_fields: bool = True
    save_plots: bool = False
    profile_interval: int = 0
    output_dir: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    config: ExperimentConfig
    success: bool
    execution_time: float
    final_time: float
    total_cells: float
    step_times: List[float]
    error_message: Optional[str] = None
    output_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['config'] = self.config.to_dict()
        return result


class ExperimentRunner:
    """
    Main class for running systematic experiments on tumor growth simulations.
    
    Supports:
    - Parameter sweeps
    - Sensitivity analyses  
    - Performance benchmarking
    - Parallel execution
    - Result collection and analysis
    """
    
    def __init__(self, base_config_path: str, base_output_dir: str = "experiments"):
        """
        Initialize the experiment runner.
        
        Args:
            base_config_path: Path to the base YAML configuration file
            base_output_dir: Base directory for experiment outputs
        """
        self.base_config_path = Path(base_config_path)
        self.base_output_dir = Path(base_output_dir)
        self.base_config = self._load_base_config()
        
        # Create base output directory
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.results: List[ExperimentResult] = []
        
    def _load_base_config(self) -> Dict[str, Any]:
        """Load the base configuration from YAML."""
        with open(self.base_config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _modify_config(self, base_config: Dict[str, Any], 
                      parameter_changes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Modify the base configuration with parameter changes.
        
        Args:
            base_config: Base configuration dictionary
            parameter_changes: Dictionary of parameter changes to apply
            
        Returns:
            Modified configuration dictionary
        """
        # Deep copy to avoid modifying original
        config = self._deep_copy_dict(base_config)
        
        # Apply parameter changes using dot notation (e.g., "populations.Healthy.dynamics.lambda")
        for param_path, value in parameter_changes.items():
            self._set_nested_value(config, param_path, value)
            
        return config
    
    def _deep_copy_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Create a deep copy of a dictionary."""
        return json.loads(json.dumps(d))
    
    def _set_nested_value(self, config: Dict[str, Any], param_path: str, value: Any):
        """Set a nested value in the configuration using dot notation."""
        keys = param_path.split('.')
        current = config
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the final value
        current[keys[-1]] = value
    
    def _run_single_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """
        Run a single experiment with the given configuration.
        
        Args:
            config: Experiment configuration
            
        Returns:
            Experiment result
        """
        start_time = time.time()
        
        try:
            # Create temporary config file
            temp_config_path = Path(f"temp_config_{config.name}.yaml")
            modified_config = self._modify_config(self.base_config, config.parameters)
            
            with open(temp_config_path, 'w') as f:
                yaml.dump(modified_config, f)
            
            # Create output directory
            if config.output_dir is None:
                output_dir = self.base_output_dir / config.name
            else:
                output_dir = Path(config.output_dir)
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize simulator
            simulator = TumorGrowthSimulator(str(temp_config_path))
            
            # Run simulation
            simulation_data = simulator.run_and_save_simulation(
                total_steps=config.total_steps,
                save_interval=config.save_interval,
                save_physics_fields=config.save_physics_fields,
                save_plots=config.save_plots,
                profile_interval=config.profile_interval,
                output_dir=str(output_dir)
            )
            
            execution_time = time.time() - start_time
            
            # Extract final state
            final_time = simulation_data.get('final_time', 0.0)
            total_cells = simulation_data.get('total_cells', 0.0)
            step_times = simulation_data.get('step_times', [])
            
            # Clean up temporary file
            temp_config_path.unlink(missing_ok=True)
            
            return ExperimentResult(
                config=config,
                success=True,
                execution_time=execution_time,
                final_time=final_time,
                total_cells=total_cells,
                step_times=step_times,
                output_path=str(output_dir)
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Clean up temporary file
            temp_config_path.unlink(missing_ok=True)
            
            return ExperimentResult(
                config=config,
                success=False,
                execution_time=execution_time,
                final_time=0.0,
                total_cells=0.0,
                step_times=[],
                error_message=str(e)
            )
    
    def run_experiments(self, experiments: List[ExperimentConfig], 
                       parallel: bool = True, max_workers: Optional[int] = None) -> List[ExperimentResult]:
        """
        Run multiple experiments, optionally in parallel.
        
        Args:
            experiments: List of experiment configurations
            parallel: Whether to run experiments in parallel
            max_workers: Maximum number of parallel workers (None for auto)
            
        Returns:
            List of experiment results
        """
        if not parallel:
            # Sequential execution
            results = []
            for i, config in enumerate(experiments):
                print(f"Running experiment {i+1}/{len(experiments)}: {config.name}")
                result = self._run_single_experiment(config)
                results.append(result)
                print(f"Completed: {config.name} - Success: {result.success}")
        else:
            # Parallel execution
            if max_workers is None:
                max_workers = min(mp.cpu_count(), len(experiments))
            
            print(f"Running {len(experiments)} experiments in parallel with {max_workers} workers")
            
            results = []
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all experiments
                future_to_config = {
                    executor.submit(self._run_single_experiment, config): config 
                    for config in experiments
                }
                
                # Collect results as they complete
                for i, future in enumerate(as_completed(future_to_config)):
                    config = future_to_config[future]
                    try:
                        result = future.result()
                        results.append(result)
                        print(f"Completed {i+1}/{len(experiments)}: {config.name} - Success: {result.success}")
                    except Exception as e:
                        print(f"Experiment {config.name} failed with exception: {e}")
                        results.append(ExperimentResult(
                            config=config,
                            success=False,
                            execution_time=0.0,
                            final_time=0.0,
                            total_cells=0.0,
                            step_times=[],
                            error_message=str(e)
                        ))
        
        self.results.extend(results)
        return results
    
    def save_results(self, filename: str = "experiment_results.json"):
        """Save experiment results to JSON file."""
        results_data = [result.to_dict() for result in self.results]
        
        output_path = self.base_output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"Results saved to {output_path}")
        return output_path
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame for analysis."""
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for result in self.results:
            row = {
                'name': result.config.name,
                'success': result.success,
                'execution_time': result.execution_time,
                'final_time': result.final_time,
                'total_cells': result.total_cells,
                'avg_step_time': np.mean(result.step_times) if result.step_times else 0.0,
                'error_message': result.error_message,
                'output_path': result.output_path
            }
            
            # Add parameter values
            for param, value in result.config.parameters.items():
                row[f'param_{param}'] = value
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def print_summary(self):
        """Print a summary of experiment results."""
        if not self.results:
            print("No results to summarize")
            return
        
        df = self.get_results_dataframe()
        
        print(f"\nExperiment Summary:")
        print(f"Total experiments: {len(self.results)}")
        print(f"Successful: {df['success'].sum()}")
        print(f"Failed: {(~df['success']).sum()}")
        
        if df['success'].any():
            successful = df[df['success']]
            print(f"\nSuccessful experiments:")
            print(f"Average execution time: {successful['execution_time'].mean():.2f}s")
            print(f"Average final time: {successful['final_time'].mean():.3f}")
            print(f"Average total cells: {successful['total_cells'].mean():.3f}")
            print(f"Average step time: {successful['avg_step_time'].mean():.4f}s")
        
        if (~df['success']).any():
            failed = df[~df['success']]
            print(f"\nFailed experiments:")
            for _, row in failed.iterrows():
                print(f"  {row['name']}: {row['error_message']}")
