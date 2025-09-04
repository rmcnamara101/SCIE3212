"""
Parameter Sweep Configuration Module

This module provides classes for defining and managing parameter sweep configurations
for tumor growth simulation experiments.

Author: Riley Jae McNamara
Date: 2025-02-19
"""

import numpy as np
import itertools
from typing import Dict, List, Any, Union, Tuple
from pathlib import Path
import yaml
import copy


class ParameterSweep:
    """
    Class for defining and managing parameter sweep configurations.
    
    This class takes an existing simulation configuration and a parameter bounds
    dictionary to automatically generate all parameter combinations.
    """
    
    def __init__(self, base_config: Union[str, Path, Dict[str, Any]], 
                 parameter_bounds: Dict[str, Dict[str, Any]]):
        """
        Initialize parameter sweep configuration.
        
        Args:
            base_config: Path to existing simulation YAML or config dictionary
            parameter_bounds: Dictionary defining parameter bounds for sweeping
                Format: {
                    'param_path': {
                        'type': 'range' | 'log_range' | 'list' | 'random' | 'latin_hypercube' | 'random_combinations',
                        'min': float, 'max': float, 'steps': int,  # for range/log_range/random
                        'values': [val1, val2, ...]  # for list
                        'seed': int  # for random/latin_hypercube/random_combinations (optional)
                        'num_combinations': int  # for random_combinations (required)
                    }
                }
        """
        # Load base configuration
        if isinstance(base_config, (str, Path)):
            with open(base_config, 'r') as f:
                self.base_config = yaml.safe_load(f)
        else:
            self.base_config = copy.deepcopy(base_config)
        
        self.parameter_bounds = parameter_bounds
        self.parameter_combinations = []
        self.experiment_names = []
        
        self._validate_parameter_bounds()
        self._generate_combinations()
    
    def _validate_parameter_bounds(self):
        """Validate the parameter bounds configuration."""
        for param_name, param_config in self.parameter_bounds.items():
            if 'type' not in param_config:
                raise ValueError(f"Parameter '{param_name}' missing 'type' specification")
            
            param_type = param_config['type']
            if param_type == 'range':
                if not all(key in param_config for key in ['min', 'max', 'steps']):
                    raise ValueError(f"Range parameter '{param_name}' missing min/max/steps")
            elif param_type == 'list':
                if 'values' not in param_config:
                    raise ValueError(f"List parameter '{param_name}' missing 'values'")
            elif param_type == 'log_range':
                if not all(key in param_config for key in ['min', 'max', 'steps']):
                    raise ValueError(f"Log range parameter '{param_name}' missing min/max/steps")
            elif param_type == 'random':
                if not all(key in param_config for key in ['min', 'max', 'steps']):
                    raise ValueError(f"Random parameter '{param_name}' missing min/max/steps")
            elif param_type == 'latin_hypercube':
                if not all(key in param_config for key in ['min', 'max', 'steps']):
                    raise ValueError(f"Latin hypercube parameter '{param_name}' missing min/max/steps")
            elif param_type == 'random_combinations':
                if not all(key in param_config for key in ['min', 'max', 'num_combinations']):
                    raise ValueError(f"Random combinations parameter '{param_name}' missing min/max/num_combinations")
            else:
                raise ValueError(f"Unknown parameter type '{param_type}' for parameter '{param_name}'")
    
    def _generate_combinations(self):
        """Generate all parameter combinations for the sweep."""
        # Check if this is a random combinations sweep
        if any(param_config['type'] == 'random_combinations' for param_config in self.parameter_bounds.values()):
            self._generate_random_combinations()
        else:
            self._generate_systematic_combinations()
    
    def _generate_random_combinations(self):
        """Generate random combinations of parameters for cost function landscape exploration."""
        # Get the number of combinations from the first random_combinations parameter
        num_combinations = None
        for param_config in self.parameter_bounds.values():
            if param_config['type'] == 'random_combinations':
                num_combinations = param_config['num_combinations']
                break
        
        if num_combinations is None:
            raise ValueError("No random_combinations parameter found")
        
        # Set seed for reproducibility
        seed = next((param_config.get('seed', 42) for param_config in self.parameter_bounds.values() 
                    if param_config['type'] == 'random_combinations'), 42)
        np.random.seed(seed)
        
        # Generate random combinations
        self.parameter_combinations = []
        self.experiment_names = []
        
        for i in range(num_combinations):
            # Generate random values for each parameter
            param_dict = {}
            for param_name, param_config in self.parameter_bounds.items():
                param_type = param_config['type']
                
                if param_type == 'random_combinations':
                    min_val = param_config['min']
                    max_val = param_config['max']
                    param_dict[param_name] = np.random.uniform(min_val, max_val)
                elif param_type == 'list':
                    param_dict[param_name] = np.random.choice(param_config['values'])
                elif param_type == 'range':
                    # For range parameters, use the min value (they're not being swept in random mode)
                    param_dict[param_name] = param_config['min']
                elif param_type == 'log_range':
                    # For log_range parameters, use the min value
                    param_dict[param_name] = param_config['min']
                elif param_type == 'random':
                    # For random parameters, use the min value
                    param_dict[param_name] = param_config['min']
                elif param_type == 'latin_hypercube':
                    # For latin_hypercube parameters, use the min value
                    param_dict[param_name] = param_config['min']
            
            # Generate experiment name
            experiment_name = self._generate_experiment_name(param_dict, i)
            
            self.parameter_combinations.append(param_dict)
            self.experiment_names.append(experiment_name)
        
        # Reset seed
        np.random.seed()
    
    def _generate_systematic_combinations(self):
        """Generate systematic parameter combinations (original method)."""
        # Generate parameter values for each parameter
        param_values = {}
        for param_name, param_config in self.parameter_bounds.items():
            param_type = param_config['type']
            
            if param_type == 'range':
                min_val = param_config['min']
                max_val = param_config['max']
                steps = param_config['steps']
                param_values[param_name] = np.linspace(min_val, max_val, steps)
            
            elif param_type == 'list':
                param_values[param_name] = param_config['values']
            
            elif param_type == 'log_range':
                min_val = param_config['min']
                max_val = param_config['max']
                steps = param_config['steps']
                param_values[param_name] = np.logspace(np.log10(min_val), np.log10(max_val), steps)
            
            elif param_type == 'random':
                min_val = param_config['min']
                max_val = param_config['max']
                steps = param_config['steps']
                seed = param_config.get('seed', 42)  # Default seed for reproducibility
                
                np.random.seed(seed)
                param_values[param_name] = np.random.uniform(min_val, max_val, steps)
                np.random.seed()  # Reset seed
            
            elif param_type == 'latin_hypercube':
                min_val = param_config['min']
                max_val = param_config['max']
                steps = param_config['steps']
                seed = param_config.get('seed', 42)  # Default seed for reproducibility
                
                # Generate Latin hypercube samples
                from scipy.stats import qmc
                sampler = qmc.LatinHypercube(d=1, seed=seed)
                samples = sampler.random(n=steps)
                # Scale to the parameter range
                param_values[param_name] = min_val + (max_val - min_val) * samples.flatten()
        
        # Generate all combinations
        param_names = list(param_values.keys())
        param_value_lists = [param_values[name] for name in param_names]
        
        self.parameter_combinations = []
        self.experiment_names = []
        
        for i, combination in enumerate(itertools.product(*param_value_lists)):
            # Create parameter dictionary for this combination
            param_dict = dict(zip(param_names, combination))
            
            # Generate experiment name
            experiment_name = self._generate_experiment_name(param_dict, i)
            
            self.parameter_combinations.append(param_dict)
            self.experiment_names.append(experiment_name)
    
    def _generate_experiment_name(self, param_dict: Dict[str, Any], index: int) -> str:
        """Generate a descriptive name for an experiment."""
        # Create parameter string
        param_strings = []
        for param_name, value in param_dict.items():
            # Extract the actual parameter name from the path
            param_short_name = param_name.split('.')[-1]
            if isinstance(value, float):
                param_strings.append(f"{param_short_name}_{value:.3g}")
            else:
                param_strings.append(f"{param_short_name}_{value}")
        
        return f"exp_{index:03d}_{'_'.join(param_strings)}"
    
    def get_experiment_config(self, index: int) -> Tuple[Dict[str, Any], str]:
        """
        Get the configuration for a specific experiment.
        
        Args:
            index: Index of the experiment
            
        Returns:
            Tuple of (modified_config, experiment_name)
        """
        if index >= len(self.parameter_combinations):
            raise IndexError(f"Experiment index {index} out of range")
        
        # Deep copy the base configuration
        modified_config = copy.deepcopy(self.base_config)
        
        # Apply parameter modifications
        param_dict = self.parameter_combinations[index]
        self._apply_parameters_to_config(modified_config, param_dict)
        
        experiment_name = self.experiment_names[index]
        return modified_config, experiment_name
    
    def _apply_parameters_to_config(self, config: Dict[str, Any], param_dict: Dict[str, Any]):
        """
        Apply parameter modifications to the configuration.
        
        Args:
            config: Configuration dictionary to modify
            param_dict: Dictionary of parameter values to apply
        """
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
    
    def get_total_experiments(self) -> int:
        """Get the total number of experiments in the sweep."""
        return len(self.parameter_combinations)
    
    def get_parameter_summary(self) -> Dict[str, Any]:
        """Get a summary of the parameter sweep configuration."""
        summary = {
            'total_experiments': self.get_total_experiments(),
            'parameters': {}
        }
        
        for param_path, param_config in self.parameter_bounds.items():
            param_type = param_config['type']
            if param_type == 'range':
                summary['parameters'][param_path] = {
                    'type': 'range',
                    'min': param_config['min'],
                    'max': param_config['max'],
                    'steps': param_config['steps']
                }
            elif param_type == 'list':
                summary['parameters'][param_path] = {
                    'type': 'list',
                    'values': param_config['values']
                }
            elif param_type == 'log_range':
                summary['parameters'][param_path] = {
                    'type': 'log_range',
                    'min': param_config['min'],
                    'max': param_config['max'],
                    'steps': param_config['steps']
                }
            elif param_type == 'random':
                summary['parameters'][param_path] = {
                    'type': 'random',
                    'min': param_config['min'],
                    'max': param_config['max'],
                    'steps': param_config['steps'],
                    'seed': param_config.get('seed', 42)
                }
            elif param_type == 'latin_hypercube':
                summary['parameters'][param_path] = {
                    'type': 'latin_hypercube',
                    'min': param_config['min'],
                    'max': param_config['max'],
                    'steps': param_config['steps'],
                    'seed': param_config.get('seed', 42)
                }
            elif param_type == 'random_combinations':
                summary['parameters'][param_path] = {
                    'type': 'random_combinations',
                    'min': param_config['min'],
                    'max': param_config['max'],
                    'num_combinations': param_config['num_combinations'],
                    'seed': param_config.get('seed', 42)
                }
        
        return summary
    
    def print_summary(self):
        """Print a summary of the parameter sweep."""
        summary = self.get_parameter_summary()
        
        print(f"Parameter Sweep Summary")
        print(f"Total Experiments: {summary['total_experiments']}")
        print("\nParameters:")
        
        for param_path, param_info in summary['parameters'].items():
            param_type = param_info['type']
            if param_type == 'range':
                print(f"  {param_path}: range({param_info['min']:.3g}, {param_info['max']:.3g}, {param_info['steps']} steps)")
            elif param_type == 'list':
                print(f"  {param_path}: {param_info['values']}")
            elif param_type == 'log_range':
                print(f"  {param_path}: log_range({param_info['min']:.3g}, {param_info['max']:.3g}, {param_info['steps']} log-spaced steps)")
            elif param_type == 'random':
                seed_info = f", seed={param_info['seed']}" if 'seed' in param_info else ""
                print(f"  {param_path}: random({param_info['min']:.3g}, {param_info['max']:.3g}, {param_info['steps']} samples{seed_info})")
            elif param_type == 'latin_hypercube':
                seed_info = f", seed={param_info['seed']}" if 'seed' in param_info else ""
                print(f"  {param_path}: latin_hypercube({param_info['min']:.3g}, {param_info['max']:.3g}, {param_info['steps']} samples{seed_info})")
            elif param_type == 'random_combinations':
                seed_info = f", seed={param_info['seed']}" if 'seed' in param_info else ""
                print(f"  {param_path}: random_combinations({param_info['min']:.3g}, {param_info['max']:.3g}, {param_info['num_combinations']} samples{seed_info})")
        
        print(f"\nFirst few experiment names:")
        for i in range(min(5, len(self.experiment_names))):
            print(f"  {i}: {self.experiment_names[i]}")
        
        if len(self.experiment_names) > 5:
            print(f"  ... and {len(self.experiment_names) - 5} more")
    
    def get_parameter_values(self, param_name: str) -> np.ndarray:
        """
        Get the parameter values for a specific parameter.
        
        Args:
            param_name: Name of the parameter
            
        Returns:
            Array of parameter values
        """
        if param_name not in self.parameter_bounds:
            raise ValueError(f"Parameter '{param_name}' not found in parameter bounds")
        
        # Find the parameter in the combinations
        param_index = None
        for i, (name, _) in enumerate(self.parameter_bounds.items()):
            if name == param_name:
                param_index = i
                break
        
        if param_index is None:
            raise ValueError(f"Parameter '{param_name}' not found")
        
        # Extract unique values for this parameter
        values = []
        for combination in self.parameter_combinations:
            values.append(combination[param_name])
        
        return np.array(values)
    
    def get_parameter_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistical summary of parameter values.
        
        Returns:
            Dictionary with statistics for each parameter
        """
        stats = {}
        
        for param_name in self.parameter_bounds.keys():
            values = self.get_parameter_values(param_name)
            stats[param_name] = {
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'unique_values': len(np.unique(values))
            }
        
        return stats
