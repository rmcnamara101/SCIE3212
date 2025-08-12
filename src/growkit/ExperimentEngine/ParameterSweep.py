"""
Parameter Sweep Module

Generates experiment configurations for systematic parameter sweeps.
Supports linear, logarithmic, and custom parameter ranges.

Author: Riley Jae McNamara
Date: 2025-02-19
"""

import numpy as np
from typing import Dict, List, Any, Union, Tuple
from dataclasses import dataclass

from .ExperimentRunner import ExperimentConfig


@dataclass
class ParameterRange:
    """Definition of a parameter range for sweeping."""
    start: float
    end: float
    num_points: int
    scale: str = "linear"  # "linear", "log", "custom"
    custom_values: List[float] = None
    
    def get_values(self) -> List[float]:
        """Generate the parameter values for this range."""
        if self.scale == "linear":
            return np.linspace(self.start, self.end, self.num_points).tolist()
        elif self.scale == "log":
            return np.logspace(np.log10(self.start), np.log10(self.end), self.num_points).tolist()
        elif self.scale == "custom":
            if self.custom_values is None:
                raise ValueError("Custom values must be provided for custom scale")
            return self.custom_values
        else:
            raise ValueError(f"Unknown scale: {self.scale}")


class ParameterSweep:
    """
    Generate experiment configurations for parameter sweeps.
    
    Supports:
    - Single parameter sweeps
    - Multi-dimensional parameter sweeps (full factorial)
    - Latin Hypercube Sampling (LHS)
    - Custom parameter combinations
    """
    
    def __init__(self, base_name: str = "sweep", total_steps: int = 100):
        """
        Initialize parameter sweep generator.
        
        Args:
            base_name: Base name for experiment configurations
            total_steps: Number of simulation steps for each experiment
        """
        self.base_name = base_name
        self.total_steps = total_steps
    
    def single_parameter_sweep(self, parameter: str, param_range: ParameterRange,
                              save_interval: int = 1, save_physics_fields: bool = True,
                              save_plots: bool = False) -> List[ExperimentConfig]:
        """
        Generate configurations for a single parameter sweep.
        
        Args:
            parameter: Parameter name (dot notation supported)
            param_range: Parameter range definition
            save_interval: How often to save data
            save_physics_fields: Whether to save physics fields
            save_plots: Whether to save plots
            
        Returns:
            List of experiment configurations
        """
        values = param_range.get_values()
        experiments = []
        
        for i, value in enumerate(values):
            config = ExperimentConfig(
                name=f"{self.base_name}_{parameter}_{i:03d}",
                parameters={parameter: value},
                total_steps=self.total_steps,
                save_interval=save_interval,
                save_physics_fields=save_physics_fields,
                save_plots=save_plots
            )
            experiments.append(config)
        
        return experiments
    
    def multi_parameter_sweep(self, parameters: Dict[str, ParameterRange],
                             save_interval: int = 1, save_physics_fields: bool = True,
                             save_plots: bool = False) -> List[ExperimentConfig]:
        """
        Generate configurations for a multi-dimensional parameter sweep (full factorial).
        
        Args:
            parameters: Dictionary of parameter names to ranges
            save_interval: How often to save data
            save_physics_fields: Whether to save physics fields
            save_plots: Whether to save plots
            
        Returns:
            List of experiment configurations
        """
        # Generate all parameter values
        param_values = {}
        for param, param_range in parameters.items():
            param_values[param] = param_range.get_values()
        
        # Generate all combinations
        param_names = list(parameters.keys())
        experiments = []
        experiment_id = 0
        
        # Use itertools.product for full factorial design
        from itertools import product
        
        for combination in product(*[param_values[param] for param in param_names]):
            # Create parameter dictionary for this combination
            param_dict = {param_names[i]: combination[i] for i in range(len(param_names))}
            
            # Create experiment name
            name_parts = [self.base_name]
            for param, value in param_dict.items():
                name_parts.append(f"{param}_{value:.3g}".replace('.', 'p'))
            name = "_".join(name_parts)
            
            config = ExperimentConfig(
                name=name,
                parameters=param_dict,
                total_steps=self.total_steps,
                save_interval=save_interval,
                save_physics_fields=save_physics_fields,
                save_plots=save_plots
            )
            experiments.append(config)
            experiment_id += 1
        
        return experiments
    
    def latin_hypercube_sweep(self, parameters: Dict[str, Tuple[float, float]], 
                             num_samples: int, save_interval: int = 1,
                             save_physics_fields: bool = True, save_plots: bool = False) -> List[ExperimentConfig]:
        """
        Generate configurations using Latin Hypercube Sampling.
        
        Args:
            parameters: Dictionary of parameter names to (min, max) tuples
            num_samples: Number of samples to generate
            save_interval: How often to save data
            save_physics_fields: Whether to save physics fields
            save_plots: Whether to save plots
            
        Returns:
            List of experiment configurations
        """
        # Generate LHS samples
        param_names = list(parameters.keys())
        param_bounds = [parameters[param] for param in param_names]
        
        # Use scipy for LHS if available, otherwise use simple random sampling
        try:
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=len(param_names))
            samples = sampler.random(n=num_samples)
            
            # Scale samples to parameter bounds
            scaled_samples = []
            for i, (min_val, max_val) in enumerate(param_bounds):
                scaled_samples.append(samples[:, i] * (max_val - min_val) + min_val)
            
        except ImportError:
            # Fallback to simple random sampling
            print("Warning: scipy not available, using simple random sampling instead of LHS")
            scaled_samples = []
            for min_val, max_val in param_bounds:
                scaled_samples.append(np.random.uniform(min_val, max_val, num_samples))
        
        experiments = []
        for i in range(num_samples):
            # Create parameter dictionary for this sample
            param_dict = {param_names[j]: scaled_samples[j][i] for j in range(len(param_names))}
            
            # Create experiment name
            name_parts = [self.base_name, "lhs", f"{i:03d}"]
            name = "_".join(name_parts)
            
            config = ExperimentConfig(
                name=name,
                parameters=param_dict,
                total_steps=self.total_steps,
                save_interval=save_interval,
                save_physics_fields=save_physics_fields,
                save_plots=save_plots
            )
            experiments.append(config)
        
        return experiments
    
    def custom_parameter_combinations(self, parameter_combinations: List[Dict[str, Any]],
                                     save_interval: int = 1, save_physics_fields: bool = True,
                                     save_plots: bool = False) -> List[ExperimentConfig]:
        """
        Generate configurations for custom parameter combinations.
        
        Args:
            parameter_combinations: List of parameter dictionaries
            save_interval: How often to save data
            save_physics_fields: Whether to save physics fields
            save_plots: Whether to save plots
            
        Returns:
            List of experiment configurations
        """
        experiments = []
        
        for i, params in enumerate(parameter_combinations):
            # Create experiment name
            name_parts = [self.base_name, "custom", f"{i:03d}"]
            name = "_".join(name_parts)
            
            config = ExperimentConfig(
                name=name,
                parameters=params,
                total_steps=self.total_steps,
                save_interval=save_interval,
                save_physics_fields=save_physics_fields,
                save_plots=save_plots
            )
            experiments.append(config)
        
        return experiments
    
    def benchmark_grid_sizes(self, grid_sizes: List[int], save_interval: int = 1,
                           save_physics_fields: bool = False, save_plots: bool = False) -> List[ExperimentConfig]:
        """
        Generate configurations for benchmarking different grid sizes.
        
        Args:
            grid_sizes: List of grid sizes to test
            save_interval: How often to save data
            save_physics_fields: Whether to save physics fields
            save_plots: Whether to save plots
            
        Returns:
            List of experiment configurations
        """
        experiments = []
        
        for grid_size in grid_sizes:
            config = ExperimentConfig(
                name=f"{self.base_name}_grid_{grid_size}",
                parameters={"domain.shape": grid_size},
                total_steps=self.total_steps,
                save_interval=save_interval,
                save_physics_fields=save_physics_fields,
                save_plots=save_plots
            )
            experiments.append(config)
        
        return experiments
