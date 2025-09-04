"""
Simple Experiment Runner Module

This module provides a simplified ExperimentRunner class for executing parameter sweep
experiments on tumor growth simulations. Designed for cost function landscape exploration.

Author: Riley Jae McNamara
Date: 2025-02-19
"""

import os
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Union, Optional
import pandas as pd
import yaml
from tqdm import tqdm

from src.growkit.Simulator import TumorGrowthSimulator


class SimpleExperimentRunner:
    """
    Simplified experiment runner for parameter sweep experiments.
    
    This class generates random parameter combinations and runs simulations,
    saving results in a clean Excel format optimized for cost function analysis.
    """
    
    def __init__(self, base_config: Union[str, Path, Dict[str, Any]], 
                 output_dir: Union[str, Path]):
        """
        Initialize the simple experiment runner.
        
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
        self.param_bounds = None
        self.num_experiments = 0
        self.results = []
        self.failed_experiments = []
        
        # Create base output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save base configuration
        self._save_base_config()
        
        # Excel file path
        self.excel_file = self.output_dir / "experiment_results.xlsx"
    
    def _save_base_config(self):
        """Save the base configuration to the output directory."""
        config_file = self.output_dir / "base_config.yaml"
        
        # Convert numpy types before saving
        clean_config = self._convert_numpy_types(self.base_config)
        
        with open(config_file, 'w') as f:
            yaml.dump(clean_config, f, default_flow_style=False, indent=2)
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python native types for YAML serialization."""
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
    
    def setup_parameter_sweep(self, param_bounds: Dict[str, Dict[str, float]], 
                             num_experiments: int):
        """
        Set up the parameter sweep with bounds for random sampling.
        
        Args:
            param_bounds: Dictionary defining parameter bounds
                Format: {'param_path': {'min': float, 'max': float}}
            num_experiments: Number of experiments to run
        """
        self.param_bounds = param_bounds
        self.num_experiments = num_experiments
        
        print(f"Parameter sweep configured:")
        print(f"  Parameters to vary: {len(param_bounds)}")
        print(f"  Number of experiments: {num_experiments}")
        print(f"  Total parameter combinations: {num_experiments}")
    
    def _generate_random_parameters(self) -> Dict[str, float]:
        """Generate random parameter values within the specified bounds."""
        random_params = {}
        
        for param_path, bounds in self.param_bounds.items():
            min_val = bounds['min']
            max_val = bounds['max']
            
            # Generate random value within bounds
            if isinstance(min_val, int) and isinstance(max_val, int):
                random_params[param_path] = np.random.randint(min_val, max_val + 1)
            else:
                random_params[param_path] = np.random.uniform(min_val, max_val)
        
        return random_params
    
    def _apply_parameters_to_config(self, config: Dict[str, Any], 
                                   param_dict: Dict[str, float]):
        """Apply parameter modifications to the configuration."""
        for param_path, value in param_dict.items():
            # Handle nested parameter paths (e.g., "populations.Stem.dynamics.lambda")
            keys = param_path.split('.')
            current = config
            
            # Navigate to the parent of the target parameter
            for key in keys[:-1]:
                if key not in current:
                    raise ValueError(f"Parameter path '{param_path}' invalid: '{key}' not found")
                current = current[key]
            
            # Set the final parameter value
            final_key = keys[-1]
            if final_key not in current:
                raise ValueError(f"Parameter path '{param_path}' invalid: '{final_key}' not found")
            current[final_key] = value
    
    def _extract_all_parameters(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract ALL simulation parameters from the config."""
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
        
        extract_params(config)
        return all_params
    
    def _extract_time_series_data(self, simulation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract time series data for cell populations and total metrics."""
        time_series_data = {}
        
        # Extract field data
        field_data = simulation_data.get('field_data', {})
        phi_hat_data = field_data.get('phi_hat', [])
        nutrient_data = field_data.get('nutrient_fields', [])
        
        if not phi_hat_data:
            return time_series_data
        
        # Get population information from simulation metadata (preferred) or base config (fallback)
        metadata = simulation_data.get('metadata', {})
        num_populations = metadata.get('num_populations', 0)
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
        print(f"DEBUG: Number of populations in simulation data: {num_populations}")
        
        # Extract time series for each population using index-based approach (like SimPlotter)
        # This ensures we match the array indexing in phi_hat data
        for i in range(num_populations):
            if i < len(population_labels):
                pop_label = population_labels[i]
            else:
                pop_label = f'Population_{i}'
            
            print(f"DEBUG: Processing population {i}: {pop_label}")
            
            # Time series for total cells
            cell_totals = []
            # Time series for radius
            radius_series = []
            
            for phi_hat in phi_hat_data:
                if i < phi_hat.shape[0]:
                    population_field = phi_hat[i]
                    # Total cells at this time step
                    total_cells = float(np.sum(population_field))
                    cell_totals.append(total_cells)
                    
                    # Calculate radius at this time step
                    radius = self._calculate_population_radius(population_field)
                    radius_series.append(float(radius))
                    
                    # Debug first time step
                    if len(cell_totals) == 1:
                        print(f"DEBUG: Population {i} ({pop_label}): initial_total={total_cells:.6f}, initial_radius={radius:.6f}")
                else:
                    cell_totals.append(0.0)
                    radius_series.append(0.0)
            
            # Store time series data with the correct population label
            time_series_data[f'time_series_{pop_label}_total_cells'] = cell_totals
            time_series_data[f'time_series_{pop_label}_radius'] = radius_series
        
        # Calculate total radius and total cells across all populations
        total_radius_series = []
        total_cells_series = []
        
        for phi_hat in phi_hat_data:
            # Total cells across all populations
            total_cells = 0.0
            total_radius = 0.0
            
            for i in range(num_populations):
                if i < phi_hat.shape[0]:
                    population_field = phi_hat[i]
                    total_cells += np.sum(population_field)
                    
                    # For total radius, we'll use the maximum radius among populations
                    radius = self._calculate_population_radius(population_field)
                    total_radius = max(total_radius, radius)
            
            total_cells_series.append(float(total_cells))
            total_radius_series.append(float(total_radius))
        
        time_series_data['time_series_total_cells'] = total_cells_series
        time_series_data['time_series_total_radius'] = total_radius_series
        
        return time_series_data
    
    def _calculate_population_radius(self, population_field: np.ndarray) -> float:
        """Calculate the effective radius of a population based on its spatial distribution."""
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
    
    def _calculate_population_center(self, population_field: np.ndarray) -> tuple:
        """Calculate the center of mass of a population."""
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
    
    def run_experiment(self, experiment_index: int) -> Dict[str, Any]:
        """Run a single experiment with random parameters."""
        try:
            # Generate random parameters for this experiment
            random_params = self._generate_random_parameters()
            
            # Create a copy of the base config and apply random parameters
            experiment_config = self.base_config.copy()
            self._apply_parameters_to_config(experiment_config, random_params)
            
            # Create temporary config file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
                clean_config = self._convert_numpy_types(experiment_config)
                yaml.dump(clean_config, temp_file, default_flow_style=False, indent=2)
                temp_config_path = temp_file.name
            
            try:
                # Initialize simulator with temporary config
                simulator = TumorGrowthSimulator(temp_config_path)
                
                # Run simulation
                print(f"Running experiment {experiment_index + 1}/{self.num_experiments}...")
                
                start_time = time.time()
                simulation_data = simulator.run_and_save_simulation(
                    total_steps=experiment_config['time']['steps'],
                    save_interval=1,
                    output_dir=None,  # Don't save to disk
                    save_physics_fields=False,  # Don't save physics fields
                    save_plots=False
                )
                end_time = time.time()
                
                # Extract all parameters (including non-sampled ones)
                all_params = self._extract_all_parameters(experiment_config)
                
                # Extract time series data
                time_series_data = self._extract_time_series_data(simulation_data)
                
                # Create result row
                result_row = {
                    'experiment_index': experiment_index,
                    'execution_time': end_time - start_time,
                    'total_time_steps': len(simulation_data.get('field_data', {}).get('phi_hat', []))
                }
                
                # Add all parameters
                for param_path, value in all_params.items():
                    param_short_name = param_path.replace('.', '_')
                    result_row[f'param_{param_short_name}'] = value
                
                # Add time series data
                for key, value in time_series_data.items():
                    # Convert lists to strings for Excel compatibility
                    if isinstance(value, list):
                        result_row[key] = str(value)
                    else:
                        result_row[key] = value
                
                # Store results
                self.results.append(result_row)
                
                print(f"  Experiment {experiment_index + 1} completed in {end_time - start_time:.2f}s")
                return result_row
                
            finally:
                # Clean up temporary config file
                os.unlink(temp_config_path)
            
        except Exception as e:
            # Record failed experiment
            error_info = {
                'experiment_index': experiment_index,
                'error': str(e),
                'status': 'failed'
            }
            
            self.failed_experiments.append(error_info)
            print(f"  Experiment {experiment_index + 1} failed: {str(e)}")
            
            return error_info
    
    def run_all_experiments(self):
        """Run all experiments."""
        if not self.param_bounds:
            raise ValueError("No parameter sweep configured. Call setup_parameter_sweep() first.")
        
        print(f"\nStarting {self.num_experiments} experiments...")
        print(f"Output directory: {self.output_dir}")
        print(f"Results will be saved to: {self.excel_file}")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Run experiments
        for i in tqdm(range(self.num_experiments), desc="Running experiments"):
            self.run_experiment(i)
            
            # Save to Excel after each experiment
            self._save_excel_results()
        
        # Reset random seed
        np.random.seed()
        
        print(f"\nAll experiments completed!")
        print(f"  Successful: {len(self.results)}")
        print(f"  Failed: {len(self.failed_experiments)}")
        print(f"  Results saved to: {self.excel_file}")
    
    def _save_excel_results(self):
        """Save all results to Excel file."""
        if not self.results:
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(self.results)
        
        # Save to Excel with multiple sheets
        with pd.ExcelWriter(self.excel_file, engine='openpyxl') as writer:
            # Main results sheet
            df.to_excel(writer, sheet_name='Results', index=False)
            
            # Create parameter summary sheet
            param_columns = [col for col in df.columns if col.startswith('param_')]
            if param_columns:
                param_summary = []
                for col in param_columns:
                    param_name = col.replace('param_', '')
                    
                    # Check if the column contains numeric data
                    try:
                        # Try to convert to numeric, coercing errors to NaN
                        numeric_series = pd.to_numeric(df[col], errors='coerce')
                        
                        # If all values are NaN, it's not numeric
                        if numeric_series.isna().all():
                            param_summary.append({
                                'parameter': param_name,
                                'type': 'non_numeric',
                                'unique_values': df[col].nunique(),
                                'sample_values': str(df[col].unique()[:3].tolist())
                            })
                        else:
                            # It's numeric, calculate statistics
                            param_summary.append({
                                'parameter': param_name,
                                'type': 'numeric',
                                'mean': numeric_series.mean(),
                                'std': numeric_series.std(),
                                'min': numeric_series.min(),
                                'max': numeric_series.max(),
                                'median': numeric_series.median()
                            })
                    except Exception as e:
                        # Fallback for any other issues
                        param_summary.append({
                            'parameter': param_name,
                            'type': 'error',
                            'error': str(e)
                        })
                
                param_df = pd.DataFrame(param_summary)
                param_df.to_excel(writer, sheet_name='Parameter_Summary', index=False)
            
            # Create time series data guide
            time_series_columns = [col for col in df.columns if col.startswith('time_series_')]
            if time_series_columns:
                time_series_guide = []
                for col in time_series_columns:
                    time_series_guide.append({
                        'column': col,
                        'description': f'Time series data for {col.replace("time_series_", "")}',
                        'format': 'String array - use eval() or ast.literal_eval() to convert back to lists',
                        'example': df[col].iloc[0] if len(df) > 0 else 'N/A'
                    })
                
                guide_df = pd.DataFrame(time_series_guide)
                guide_df.to_excel(writer, sheet_name='Time_Series_Guide', index=False)
        
        print(f"Excel results updated: {self.excel_file}")
    
    def print_status(self):
        """Print current status of experiments."""
        total = len(self.results) + len(self.failed_experiments)
        
        print(f"\nExperiment Status:")
        print(f"  Completed: {len(self.results)}")
        print(f"  Failed: {len(self.failed_experiments)}")
        print(f"  Total: {total}")
        
        if total > 0:
            success_rate = len(self.results) / total * 100
            print(f"  Success Rate: {success_rate:.1f}%")
        
        if self.failed_experiments:
            print(f"\nFailed Experiments:")
            for failure in self.failed_experiments[:5]:  # Show first 5
                print(f"  {failure['experiment_index']}: {failure['error']}")
            if len(self.failed_experiments) > 5:
                print(f"  ... and {len(self.failed_experiments) - 5} more")
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Get results as a pandas DataFrame."""
        return pd.DataFrame(self.results)
    
    def get_parameter_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistical summary of parameter values."""
        if not self.results:
            return {}
        
        df = pd.DataFrame(self.results)
        param_columns = [col for col in df.columns if col.startswith('param_')]
        
        stats = {}
        for col in param_columns:
            param_name = col.replace('param_', '')
            
            # Check if the column contains numeric data
            try:
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                
                if numeric_series.isna().all():
                    # Non-numeric parameter
                    stats[param_name] = {
                        'type': 'non_numeric',
                        'unique_values': int(df[col].nunique()),
                        'sample_values': df[col].unique()[:3].tolist()
                    }
                else:
                    # Numeric parameter
                    stats[param_name] = {
                        'type': 'numeric',
                        'mean': float(numeric_series.mean()),
                        'std': float(numeric_series.std()),
                        'min': float(numeric_series.min()),
                        'max': float(numeric_series.max()),
                        'median': float(numeric_series.median())
                    }
            except Exception as e:
                stats[param_name] = {
                    'type': 'error',
                    'error': str(e)
                }
        
        return stats
