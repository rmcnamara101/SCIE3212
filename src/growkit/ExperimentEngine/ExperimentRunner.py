"""
Experiment Runner Module

This module provides the ExperimentRunner class for executing various types of
experiments on tumor growth simulations.

Author: Riley Jae McNamara
Date: 2025-02-19
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Union, Optional, Callable, Tuple
import numpy as np
from tqdm import tqdm
import pickle
import yaml
import pandas as pd

from src.growkit.Simulator import TumorGrowthSimulator


class ExperimentRunner:
    """
    General class for running experiments on tumor growth simulations.
    
    This class can handle parameter sweeps, single experiments, or other
    experimental designs and provides progress tracking and result collection.
    Optimized for running thousands of simulations with minimal memory usage.
    """
    
    def __init__(self, base_config: Union[str, Path, Dict[str, Any]], 
                 output_dir: Union[str, Path]):
        """
        Initialize the experiment runner.
        
        Args:
            base_config: Path to simulation YAML or config dictionary
            output_dir: Base directory for all experiment outputs
        """
        # Load base configuration
        if isinstance(base_config, (str, Path)):
            with open(base_config, 'r') as f:
                self.base_config = yaml.safe_load(f)
        else:
            self.base_config = base_config.copy()
        
        self.output_dir = Path(output_dir)
        self.results = []
        self.failed_experiments = []
        self.completed_experiments = 0
        self.parameter_bounds = None
        
        # Create base output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save base configuration
        self._save_base_config()
        
        # Initialize results DataFrame
        self.results_df = pd.DataFrame()
        
        # Excel file path
        self.excel_file = self.output_dir / "experiment_results.xlsx"
    
    def _convert_numpy_types(self, obj):
        """
        Convert numpy types to Python native types for YAML serialization.
        
        Args:
            obj: Object that may contain numpy types
            
        Returns:
            Object with numpy types converted to Python types
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    def _save_base_config(self):
        """Save the base configuration to the output directory."""
        config_file = self.output_dir / "base_config.yaml"
        
        # Convert numpy types before saving
        clean_config = self._convert_numpy_types(self.base_config)
        
        with open(config_file, 'w') as f:
            yaml.dump(clean_config, f, default_flow_style=False, indent=2)
    
    def add_parameter_sweep(self, parameter_bounds: Dict[str, Dict[str, Any]]):
        """
        Add a parameter sweep experiment.
        
        Args:
            parameter_bounds: Dictionary defining parameter bounds for sweeping
                Format: {
                    'param_path': {
                        'type': 'range' | 'log_range' | 'list' | 'random' | 'latin_hypercube',
                        'min': float, 'max': float, 'steps': int,  # for range/log_range/random
                        'values': [val1, val2, ...]  # for list
                    }
                }
        """
        self.parameter_bounds = parameter_bounds
        print(f"Parameter sweep configured with {len(parameter_bounds)} parameters")
        
        # Calculate total experiments
        total_experiments = 1
        for param_config in parameter_bounds.values():
            if param_config['type'] in ['range', 'log_range', 'random', 'latin_hypercube']:
                total_experiments *= param_config['steps']
            elif param_config['type'] == 'list':
                total_experiments *= len(param_config['values'])
        
        print(f"Total experiments: {total_experiments}")
    
    def _generate_experiment_config(self, experiment_index: int) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
        """
        Generate configuration for a specific experiment index.
        
        Args:
            experiment_index: Index of the experiment
            
        Returns:
            Tuple of (config, experiment_name, parameters)
        """
        if not self.parameter_bounds:
            raise ValueError("No parameter sweep configured")
        
        from .ParameterSweep import ParameterSweep
        
        # Create parameter sweep for this single experiment
        param_sweep = ParameterSweep(self.base_config, self.parameter_bounds)
        
        if experiment_index >= param_sweep.get_total_experiments():
            raise IndexError(f"Experiment index {experiment_index} out of range")
        
        # Get the experiment configuration
        config, name = param_sweep.get_experiment_config(experiment_index)
        parameters = param_sweep.parameter_combinations[experiment_index]
        
        return config, name, parameters
    
    def run_experiment(self, experiment_index: int, 
                      custom_simulator_kwargs: Optional[Dict[str, Any]] = None,
                      progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Run a single experiment.
        
        Args:
            experiment_index: Index of the experiment to run
            custom_simulator_kwargs: Additional arguments to pass to simulator
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Dictionary containing experiment results
        """
        try:
            # Generate experiment configuration on-the-fly
            config, experiment_name, parameters = self._generate_experiment_config(experiment_index)
            
            # Create temporary config file in memory (not saved to disk)
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
                # Convert numpy types before saving
                clean_config = self._convert_numpy_types(config)
                # Use yaml.dump with sort_keys=False to preserve dictionary order
                yaml.dump(clean_config, temp_file, default_flow_style=False, indent=2, sort_keys=False)
                temp_config_path = temp_file.name
            
            try:
                # Initialize simulator with temporary config
                simulator_kwargs = custom_simulator_kwargs or {}
                simulator = TumorGrowthSimulator(temp_config_path, **simulator_kwargs)
                
                # Run simulation
                if progress_callback:
                    progress_callback(f"Running experiment {experiment_index}: {experiment_name}")
                
                start_time = time.time()
                simulation_data = simulator.run_and_save_simulation(
                    total_steps=config['time']['steps'],
                    save_interval=1,
                    output_dir=None,  # Don't save to disk
                    save_physics_fields=False,  # Don't save physics fields to save space
                    save_plots=False
                )
                end_time = time.time()
                
                # Extract essential data for Excel output
                experiment_results = self._extract_experiment_data(
                    experiment_index, experiment_name, parameters, 
                    simulation_data, end_time - start_time
                )
                
                # Store results
                self.results.append(experiment_results)
                self.completed_experiments += 1
                
                # Save to Excel immediately after each experiment
                self._save_excel_results()
                
                return experiment_results
                
            finally:
                # Clean up temporary config file
                os.unlink(temp_config_path)
            
        except Exception as e:
            # Record failed experiment
            error_info = {
                'experiment_index': experiment_index,
                'experiment_name': f"exp_{experiment_index:03d}",
                'error': str(e),
                'status': 'failed'
            }
            
            self.failed_experiments.append(error_info)
            
            if progress_callback:
                progress_callback(f"Experiment {experiment_index} failed: {str(e)}")
            
            return error_info
    
    def _extract_experiment_data(self, experiment_index: int, experiment_name: str, 
                                parameters: Dict[str, Any], simulation_data: Dict[str, Any], 
                                execution_time: float) -> Dict[str, Any]:
        """
        Extract essential data from simulation results for Excel output.
        
        Args:
            experiment_index: Index of the experiment
            experiment_name: Name of the experiment
            parameters: Parameter values used
            simulation_data: Simulation results
            execution_time: Time taken to run the experiment
            
        Returns:
            Dictionary containing extracted data
        """
        # Extract field data
        field_data = simulation_data.get('field_data', {})
        phi_hat_data = field_data.get('phi_hat', [])
        nutrient_data = field_data.get('nutrient_fields', [])
        
        # Get population information from simulation metadata (preferred) or base config (fallback)
        metadata = simulation_data.get('metadata', {})
        population_names = metadata.get('population_names', [])
        population_labels = metadata.get('population_labels', [])
        
        # Fallback to base config if metadata doesn't have population info
        if not population_names:
            population_config = self.base_config.get('populations', {})
            population_names = list(population_config.keys())
            population_labels = [p.get('label', name) for name, p in population_config.items()]
        
        # Debug: Print population information
        print(f"DEBUG: Population names from metadata: {population_names}")
        print(f"DEBUG: Population labels from metadata: {population_labels}")
        print(f"DEBUG: phi_hat_data shape: {[phi.shape for phi in phi_hat_data] if phi_hat_data else 'None'}")
        print(f"DEBUG: First phi_hat shape: {phi_hat_data[0].shape if phi_hat_data else 'None'}")
        print(f"DEBUG: Number of populations: {len(population_names)}")
        
        # Create ONE data row per simulation (not per time step)
        row = {
            'experiment_index': experiment_index,
            'experiment_name': experiment_name,
            'execution_time': execution_time,
            'total_time_steps': len(phi_hat_data)
        }
        
        # Add ONLY the key numerical parameters (not labels, descriptions, etc.)
        key_params = self._get_key_simulation_parameters()
        for param_path, param_value in key_params.items():
            param_short_name = param_path.split('.')[-1]
            row[f'param_{param_short_name}'] = param_value
        
        # Add observables for each population using index-based approach (like SimPlotter)
        # Use the FINAL time step for final state analysis
        if phi_hat_data:
            final_phi_hat = phi_hat_data[-1]  # Last time step
            
            # Use index-based iteration to match phi_hat array indexing
            for i in range(final_phi_hat.shape[0]):
                if i < len(population_names):
                    pop_name = population_names[i]
                else:
                    pop_name = f'Population_{i}'
                
                print(f"DEBUG: Processing population {i}: {pop_name}")
                if i < final_phi_hat.shape[0]:
                    print(f"DEBUG: Population {pop_name} has data, shape: {final_phi_hat[i].shape}")
                    # Population statistics at final time step
                    row[f'final_cells_{pop_name}_total'] = np.sum(final_phi_hat[i])
                    row[f'final_cells_{pop_name}_mean'] = np.mean(final_phi_hat[i])
                    row[f'final_cells_{pop_name}_std'] = np.std(final_phi_hat[i])
                    
                    # Calculate radius for this population at final time
                    radius = self._calculate_population_radius(final_phi_hat[i])
                    row[f'final_cells_{pop_name}_radius'] = radius
                    
                    # Calculate center of mass for this population at final time
                    center_x, center_y, center_z = self._calculate_population_center(final_phi_hat[i])
                    row[f'final_cells_{pop_name}_center_x'] = center_x
                    row[f'final_cells_{pop_name}_center_y'] = center_y
                    row[f'final_cells_{pop_name}_center_z'] = center_z
                    
                    # Add growth metrics (if we have multiple time steps)
                    if len(phi_hat_data) > 1:
                        initial_phi_hat = phi_hat_data[0]  # First time step
                        if i < initial_phi_hat.shape[0]:
                            initial_total = np.sum(initial_phi_hat[i])
                            final_total = np.sum(final_phi_hat[i])
                            if initial_total > 0:
                                growth_factor = final_total / initial_total
                                row[f'growth_factor_{pop_name}'] = growth_factor
                            else:
                                row[f'growth_factor_{pop_name}'] = 0.0
                    
                    # Add time-series data as arrays (for analysis over time)
                    if len(phi_hat_data) > 1:
                        time_series_totals = []
                        for phi_hat in phi_hat_data:
                            if i < phi_hat.shape[0]:
                                time_series_totals.append(float(np.sum(phi_hat[i])))
                            else:
                                time_series_totals.append(0.0)
                        # Convert to string array for Excel compatibility
                        row[f'time_series_{pop_name}_totals'] = str(time_series_totals)
                else:
                    print(f"DEBUG: Population {pop_name} index {i} out of bounds for phi_hat shape {final_phi_hat.shape}")
                    # Handle case where population index is out of bounds
                    row[f'final_cells_{pop_name}_total'] = 0.0
                    row[f'final_cells_{pop_name}_mean'] = 0.0
                    row[f'final_cells_{pop_name}_std'] = 0.0
                    row[f'final_cells_{pop_name}_radius'] = 0.0
                    row[f'final_cells_{pop_name}_center_x'] = 0.0
                    row[f'final_cells_{pop_name}_center_y'] = 0.0
                    row[f'final_cells_{pop_name}_center_z'] = 0.0
                    row[f'growth_factor_{pop_name}'] = 0.0
                    row[f'time_series_{pop_name}_totals'] = str([0.0] * len(phi_hat_data))
        
        # Add nutrient field statistics at final time step
        if nutrient_data and len(nutrient_data) > 0:
            final_nutrient = nutrient_data[-1]  # Last time step
            if final_nutrient is not None:
                row['final_nutrient_total'] = np.sum(final_nutrient)
                row['final_nutrient_mean'] = np.mean(final_nutrient)
                row['final_nutrient_std'] = np.std(final_nutrient)
                row['final_nutrient_min'] = np.min(final_nutrient)
                row['final_nutrient_max'] = np.max(final_nutrient)
                
                # Add nutrient time-series data
                if len(nutrient_data) > 1:
                    nutrient_time_series = []
                    for nutrient_field in nutrient_data:
                        if nutrient_field is not None:
                            nutrient_time_series.append(float(np.sum(nutrient_field)))
                        else:
                            nutrient_time_series.append(0.0)
                    row['time_series_nutrient_totals'] = str(nutrient_time_series)
            else:
                row['final_nutrient_total'] = 0.0
                row['final_nutrient_mean'] = 0.0
                row['final_nutrient_std'] = 0.0
                row['final_nutrient_min'] = 0.0
                row['final_nutrient_max'] = 0.0
                row['time_series_nutrient_totals'] = str([0.0] * len(nutrient_data))
        else:
            row['final_nutrient_total'] = 0.0
            row['final_nutrient_mean'] = 0.0
            row['final_nutrient_std'] = 0.0
            row['final_nutrient_min'] = 0.0
            row['final_nutrient_max'] = 0.0
            row['time_series_nutrient_totals'] = str([0.0] * len(phi_hat_data))
        
        return {
            'experiment_index': experiment_index,
            'experiment_name': experiment_name,
            'parameters': parameters,
            'data_row': row,  # Single row per experiment
            'execution_time': execution_time,
            'status': 'completed'
        }
    
    def _get_all_simulation_parameters(self) -> Dict[str, Any]:
        """
        Extract ALL simulation parameters from the base config for consistent analysis.
        
        Returns:
            Dictionary mapping parameter paths to values
        """
        all_params = {}
        
        def extract_params(obj, prefix=""):
            """Recursively extract all parameters from nested dictionary."""
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{prefix}.{key}" if prefix else key
                    if isinstance(value, (int, float, str, bool)):
                        all_params[current_path] = value
                    elif isinstance(value, list):
                        # Convert lists to strings for Excel compatibility
                        all_params[current_path] = str(value)
                    else:
                        extract_params(value, current_path)
        
        extract_params(self.base_config)
        return all_params
    
    def _get_key_simulation_parameters(self) -> Dict[str, Any]:
        """
        Extract ONLY the key numerical parameters from the base config for consistent analysis.
        
        Returns:
            Dictionary mapping parameter paths to values
        """
        key_params = {}
        
        def extract_key_params(obj, prefix=""):
            """Recursively extract key numerical parameters from nested dictionary."""
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{prefix}.{key}" if prefix else key
                    if isinstance(value, (int, float)):
                        key_params[current_path] = value
                    elif isinstance(value, list):
                        # If it's a list of numbers, add each number
                        for i, item in enumerate(value):
                            if isinstance(item, (int, float)):
                                key_params[f"{current_path}.item_{i}"] = item
                    else:
                        extract_key_params(value, current_path)
        
        extract_key_params(self.base_config)
        return key_params
    
    def _calculate_population_radius(self, population_field: np.ndarray) -> float:
        """
        Calculate the effective radius of a population based on its spatial distribution.
        
        Args:
            population_field: 3D array representing population density
            
        Returns:
            Effective radius of the population
        """
        # Find the center of mass
        center_x, center_y, center_z = self._calculate_population_center(population_field)
        
        # Calculate distances from center to all non-zero points
        distances = []
        for i in range(population_field.shape[0]):
            for j in range(population_field.shape[1]):
                for k in range(population_field.shape[2]):
                    if population_field[i, j, k] > 0:
                        # Calculate distance from center
                        dist = np.sqrt((i - center_x)**2 + (j - center_y)**2 + (k - center_z)**2)
                        distances.append(dist)
        
        if not distances:
            return 0.0
        
        # Return the 95th percentile as effective radius (to handle outliers)
        return np.percentile(distances, 95)
    
    def _calculate_population_center(self, population_field: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate the center of mass of a population.
        
        Args:
            population_field: 3D array representing population density
            
        Returns:
            Tuple of (center_x, center_y, center_z) coordinates
        """
        # Get grid coordinates
        x_coords, y_coords, z_coords = np.meshgrid(
            np.arange(population_field.shape[0]),
            np.arange(population_field.shape[1]),
            np.arange(population_field.shape[2]),
            indexing='ij'
        )
        
        # Calculate center of mass
        total_mass = np.sum(population_field)
        if total_mass == 0:
            return 0.0, 0.0, 0.0
        
        center_x = np.sum(x_coords * population_field) / total_mass
        center_y = np.sum(y_coords * population_field) / total_mass
        center_z = np.sum(z_coords * population_field) / total_mass
        
        return center_x, center_y, center_z
    
    def run_all_experiments(self, 
                           start_index: int = 0,
                           end_index: Optional[int] = None,
                           max_workers: int = 1,
                           custom_simulator_kwargs: Optional[Dict[str, Any]] = None,
                           save_progress: bool = True,
                           progress_interval: int = 5) -> Dict[str, Any]:
        """
        Run all experiments.
        
        Args:
            start_index: Starting experiment index (default: 0)
            end_index: Ending experiment index (default: all experiments)
            max_workers: Maximum number of parallel workers (default: 1 for sequential)
            custom_simulator_kwargs: Additional arguments to pass to simulator
            save_progress: Whether to save progress after each experiment
            progress_interval: How often to save progress (every N experiments)
            
        Returns:
            Dictionary containing summary of all experiments
        """
        if not self.parameter_bounds:
            print("No parameter sweep configured. Add experiments first using add_parameter_sweep().")
            return {}
        
        # Calculate total experiments
        total_experiments = 1
        for param_config in self.parameter_bounds.values():
            if param_config['type'] in ['range', 'log_range', 'random', 'latin_hypercube']:
                total_experiments *= param_config['steps']
            elif param_config['type'] == 'list':
                total_experiments *= len(param_config['values'])
        
        if end_index is None:
            end_index = total_experiments
        
        total_to_run = end_index - start_index
        print(f"Starting experiment execution with {total_to_run} experiments")
        print(f"Output directory: {self.output_dir}")
        print(f"Results will be saved to: {self.excel_file}")
        print("Note: You can stop this at any time and your data will be saved!")
        
        # Progress tracking
        def progress_callback(message):
            print(f"[{time.strftime('%H:%M:%S')}] {message}")
        
        # Run experiments
        if max_workers == 1:
            # Sequential execution
            for i in tqdm(range(start_index, end_index), desc="Running experiments"):
                self.run_experiment(i, custom_simulator_kwargs, progress_callback)
                
                # Save progress periodically
                if save_progress and (i - start_index + 1) % progress_interval == 0:
                    self._save_progress()
        else:
            # Parallel execution (basic implementation)
            print(f"Parallel execution with {max_workers} workers not yet implemented")
            print("Falling back to sequential execution")
            return self.run_all_experiments(start_index, end_index, 1, 
                                         custom_simulator_kwargs, save_progress, progress_interval)
        
        # Final progress save
        if save_progress:
            self._save_progress()
        
        # Generate summary
        summary = self._generate_summary()
        
        # Save summary
        summary_file = self.output_dir / "experiment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\nExperiment execution completed!")
        print(f"Completed: {self.completed_experiments}")
        print(f"Failed: {len(self.failed_experiments)}")
        print(f"Results saved to: {self.excel_file}")
        
        return summary
    
    def _save_excel_results(self):
        """Save all results to a single Excel file."""
        if not self.results:
            return
        
        # Prepare data for Excel
        all_rows = []
        for result in self.results:
            if 'data_row' in result:
                all_rows.append(result['data_row'])
        
        if all_rows:
            # Convert to DataFrame
            df = pd.DataFrame(all_rows)
            
            # Save to Excel
            with pd.ExcelWriter(self.excel_file, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Results', index=False)
                
                # Create parameter summary sheet
                param_summary = []
                for result in self.results:
                    if 'parameters' in result:
                        param_summary.append({
                            'experiment_index': result['experiment_index'],
                            'experiment_name': result['experiment_name'],
                            **result['parameters']
                        })
                
                if param_summary:
                    param_df = pd.DataFrame(param_summary)
                    param_df.to_excel(writer, sheet_name='Parameters', index=False)
                
                # Create summary statistics sheet
                if all_rows:
                    summary_stats = []
                    
                    # Population summaries
                    population_names = list(self.base_config.get('populations', {}).keys())
                    for pop_name in population_names:
                        # Find columns for this population
                        total_cols = [col for col in df.columns if f'final_cells_{pop_name}_total' in col]
                        if total_cols:
                            total_col = total_cols[0]
                            summary_stats.append({
                                'metric': f'{pop_name}_total_cells',
                                'mean': df[total_col].mean(),
                                'std': df[total_col].std(),
                                'min': df[total_col].min(),
                                'max': df[total_col].max(),
                                'median': df[total_col].median()
                            })
                    
                    # Nutrient summaries
                    nutrient_cols = [col for col in df.columns if 'final_nutrient_' in col]
                    for col in nutrient_cols:
                        metric_name = col.replace('final_nutrient_', 'nutrient_')
                        summary_stats.append({
                            'metric': metric_name,
                            'mean': df[col].mean(),
                            'std': df[col].std(),
                            'min': df[col].min(),
                            'max': df[col].max(),
                            'median': df[col].median()
                        })
                    
                    if summary_stats:
                        summary_df = pd.DataFrame(summary_stats)
                        summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
                
                # Create data format guide sheet
                format_guide = [
                    {'column_type': 'Basic Info', 'columns': 'experiment_index, experiment_name, execution_time, total_time_steps'},
                    {'column_type': 'Parameters', 'columns': 'param_dx, param_dt, param_lambda, param_mu, etc.'},
                    {'column_type': 'Final State', 'columns': 'final_cells_{population}_total, final_cells_{population}_mean, etc.'},
                    {'column_type': 'Growth Metrics', 'columns': 'growth_factor_{population}'},
                    {'column_type': 'Time Series Data', 'columns': 'time_series_{population}_totals, time_series_nutrient_totals'},
                    {'column_type': 'Time Series Format', 'columns': 'String arrays like "[1.0, 2.5, 3.2]" - use eval() or ast.literal_eval() to convert back to lists'}
                ]
                
                format_df = pd.DataFrame(format_guide)
                format_df.to_excel(writer, sheet_name='Data_Format_Guide', index=False)
            
            print(f"Excel results updated: {self.excel_file}")
    
    def _save_progress(self):
        """Save current progress to file."""
        progress_file = self.output_dir / "experiment_progress.pkl"
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        progress_data = {
            'completed_experiments': self.completed_experiments,
            'failed_experiments': self.failed_experiments,
            'results': self.results,
            'parameter_bounds': self.parameter_bounds,
            'timestamp': time.time()
        }
        
        with open(progress_file, 'wb') as f:
            pickle.dump(progress_data, f)
    
    def load_progress(self):
        """Load progress from file."""
        progress_file = self.output_dir / "experiment_progress.pkl"
        
        if progress_file.exists():
            with open(progress_file, 'rb') as f:
                progress_data = pickle.load(f)
            
            self.completed_experiments = progress_data.get('completed_experiments', 0)
            self.failed_experiments = progress_data.get('failed_experiments', [])
            self.results = progress_data.get('results', [])
            self.parameter_bounds = progress_data.get('parameter_bounds', None)
            
            print(f"Loaded progress: {self.completed_experiments} completed, {len(self.failed_experiments)} failed")
        else:
            print("No progress file found")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of all experiments."""
        summary = {
            'execution_summary': {
                'total_experiments': self.completed_experiments + len(self.failed_experiments),
                'completed_experiments': self.completed_experiments,
                'failed_experiments': len(self.failed_experiments),
                'success_rate': self.completed_experiments / (self.completed_experiments + len(self.failed_experiments)) if (self.completed_experiments + len(self.failed_experiments)) > 0 else 0
            },
            'timing_summary': {},
            'failed_experiments': self.failed_experiments
        }
        
        # Calculate timing statistics
        if self.results:
            execution_times = [result['execution_time'] for result in self.results if 'execution_time' in result]
            if execution_times:
                summary['timing_summary'] = {
                    'total_execution_time': sum(execution_times),
                    'average_execution_time': np.mean(execution_times),
                    'min_execution_time': min(execution_times),
                    'max_execution_time': max(execution_times)
                }
        
        return summary
    
    def get_experiment_results(self, experiment_index: int) -> Optional[Dict[str, Any]]:
        """Get results for a specific experiment."""
        for result in self.results:
            if result.get('experiment_index') == experiment_index:
                return result
        return None
    
    def get_completed_experiments(self) -> int:
        """Get number of completed experiments."""
        return self.completed_experiments
    
    def get_failed_experiments(self) -> List[Dict[str, Any]]:
        """Get list of failed experiment information."""
        return self.failed_experiments.copy()
    
    def print_status(self):
        """Print current status of experiments."""
        total = self.completed_experiments + len(self.failed_experiments)
        
        print(f"Experiment Status:")
        print(f"  Completed: {self.completed_experiments}")
        print(f"  Failed: {len(self.failed_experiments)}")
        print(f"  Total: {total}")
        if total > 0:
            print(f"  Success Rate: {self.completed_experiments/total*100:.1f}%")
        
        if self.failed_experiments:
            print(f"\nFailed Experiments:")
            for failure in self.failed_experiments[:5]:  # Show first 5
                print(f"  {failure['experiment_index']}: {failure['error']}")
            if len(self.failed_experiments) > 5:
                print(f"  ... and {len(self.failed_experiments) - 5} more")
    
    def resume_experiments(self, custom_simulator_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Resume experiments from where they left off.
        
        Args:
            custom_simulator_kwargs: Additional arguments to pass to simulator
            
        Returns:
            Dictionary containing summary of all experiments
        """
        # Load existing progress
        self.load_progress()
        
        if not self.parameter_bounds:
            print("No parameter bounds loaded, cannot resume")
            return {}
        
        # Calculate total experiments
        total_experiments = 1
        for param_config in self.parameter_bounds.values():
            if param_config['type'] in ['range', 'log_range', 'random', 'latin_hypercube']:
                total_experiments *= param_config['steps']
            elif param_config['type'] == 'list':
                total_experiments *= len(param_config['values'])
        
        # Find remaining experiments
        remaining = total_experiments - self.completed_experiments - len(self.failed_experiments)
        
        if remaining <= 0:
            print("No remaining experiments to run")
            return self._generate_summary()
        
        print(f"Resuming {remaining} remaining experiments")
        return self.run_all_experiments(
            start_index=self.completed_experiments + len(self.failed_experiments),
            end_index=total_experiments,
            custom_simulator_kwargs=custom_simulator_kwargs
        )
    
    def clear_experiments(self):
        """Clear all configured experiments."""
        self.results = []
        self.failed_experiments = []
        self.completed_experiments = 0
        print("All experiments cleared")
