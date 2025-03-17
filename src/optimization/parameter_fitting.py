import os
import numpy as np
from scipy.optimize import differential_evolution
import concurrent.futures
from tqdm import tqdm

from src.models.initial_conditions import SphericalTumor
from src.models.SCIE3121_model import SCIE3121_MODEL
from src.optimization.data_comparison import DataComparison

class ParameterOptimizer:
    """
    Class for optimizing tumor growth model parameters to match real data.
    """
    
    def __init__(self, real_data_path, param_ranges, base_params, grid_shape, dx, dt, steps, save_steps):
        """
        Initialize the parameter optimizer.
        
        Args:
            real_data_path (str): Path to the real tumor image data.
            param_ranges (dict): Dictionary of parameter ranges to explore {param_name: (min, max)}.
            base_params (dict): Base parameters to use for the model.
            grid_shape (tuple): Shape of the simulation grid.
            dx (float): Spatial step size.
            dt (float): Time step size.
            steps (int): Number of simulation steps.
            save_steps (int): Frequency of saving simulation state.
        """
        self.real_data_path = real_data_path
        self.param_ranges = param_ranges
        self.base_params = base_params
        self.grid_shape = grid_shape
        self.dx = dx
        self.dt = dt
        self.steps = steps
        self.save_steps = save_steps
        
        # Load real data
        self.real_data = self._load_real_data()
        
    def _load_real_data(self):
        """
        Load real tumor image data.
        
        Returns:
            dict: Dictionary containing real tumor data.
        """
        # This is a placeholder - implement based on your real data format
        # For example, you might load a series of segmented tumor images
        # and extract volume, surface area, or other metrics
        
        print(f"Loading real data from {self.real_data_path}")
        
        # Placeholder implementation
        real_data = {
            'time_points': [],
            'volumes': [],
            'shapes': []
        }
        
        # TODO: Implement actual data loading based on your data format
        
        return real_data
    
    def _run_simulation_with_params(self, params_dict):
        """
        Run a simulation with the given parameters.
        
        Args:
            params_dict (dict): Dictionary of parameters to use.
            
        Returns:
            dict: Simulation history.
        """
        # Create a unique name for this simulation
        param_str = "_".join([f"{k}_{v:.3f}" for k, v in params_dict.items() 
                             if k in self.param_ranges])
        sim_name = f"opt_{param_str}"
        
        # Set up initial conditions
        initial_conditions = SphericalTumor(
            self.grid_shape, 
            radius=5,  # You might want to make this configurable
            nutrient_value=1.0
        )
        
        # Create and run the model
        model = SCIE3121_MODEL(
            grid_shape=self.grid_shape,
            dx=self.dx,
            dt=self.dt,
            params=params_dict,
            initial_conditions=initial_conditions,
            save_steps=self.save_steps
        )
        
        # Run the simulation
        model.run_simulation(steps=self.steps)
        
        # Get the history
        history = model.get_history()
        
        return history
    
    def _evaluate_params(self, param_values):
        """
        Evaluate a set of parameter values by running a simulation and comparing to real data.
        
        Args:
            param_values (list): List of parameter values to evaluate.
            
        Returns:
            float: Error metric (lower is better).
        """
        # Convert parameter values to dictionary
        params_dict = self.base_params.copy()
        for i, param_name in enumerate(self.param_ranges.keys()):
            params_dict[param_name] = param_values[i]
        
        # Run simulation
        sim_history = self._run_simulation_with_params(params_dict)
        
        # Compare with real data
        comparison = DataComparison(
            simulation_data=sim_history,
            real_data=self.real_data
        )
        
        # Calculate error metrics
        metrics = comparison.calculate_metrics()
        
        # Return the error metric (lower is better)
        return metrics['total_error']
    
    def _sample_params(self, n_samples):
        """
        Sample random parameter sets within the specified ranges.
        
        Args:
            n_samples (int): Number of parameter sets to sample.
            
        Returns:
            list: List of parameter dictionaries.
        """
        param_sets = []
        
        for _ in range(n_samples):
            params = self.base_params.copy()
            
            for param_name, (min_val, max_val) in self.param_ranges.items():
                params[param_name] = np.random.uniform(min_val, max_val)
            
            param_sets.append(params)
        
        return param_sets
    
    def optimize_random_search(self, n_iterations, n_simulations):
        """
        Optimize parameters using random search.
        
        Args:
            n_iterations (int): Number of optimization iterations.
            n_simulations (int): Number of simulations to run per iteration.
            
        Returns:
            tuple: (best_params, best_error, all_results)
        """
        best_params = None
        best_error = float('inf')
        all_results = []
        
        for iteration in range(n_iterations):
            print(f"Iteration {iteration+1}/{n_iterations}")
            
            # Sample parameter sets
            param_sets = self._sample_params(n_simulations)
            
            # Evaluate parameter sets in parallel
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = []
                for params in param_sets:
                    futures.append(executor.submit(self._run_and_evaluate, params))
                
                # Collect results
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                    params, error = future.result()
                    all_results.append((params, error))
                    
                    if error < best_error:
                        best_error = error
                        best_params = params
                        print(f"New best error: {best_error}")
                        print(f"Parameters: {best_params}")
        
        return best_params, best_error, all_results
    
    def _run_and_evaluate(self, params):
        """Helper method for parallel execution."""
        sim_history = self._run_simulation_with_params(params)
        
        # Compare with real data
        comparison = DataComparison(
            simulation_data=sim_history,
            real_data=self.real_data
        )
        
        # Calculate error metrics
        metrics = comparison.calculate_metrics()
        
        return params, metrics['total_error']
    
    def optimize_differential_evolution(self):
        """
        Optimize parameters using differential evolution.
        
        Returns:
            tuple: (best_params, best_error, all_results)
        """
        # Define bounds for differential evolution
        bounds = [self.param_ranges[param] for param in self.param_ranges]
        
        # Run differential evolution
        result = differential_evolution(
            self._evaluate_params,
            bounds,
            strategy='best1bin',
            maxiter=20,
            popsize=15,
            tol=0.01,
            mutation=(0.5, 1.0),
            recombination=0.7,
            disp=True
        )
        
        # Convert best solution to parameter dictionary
        best_params = self.base_params.copy()
        for i, param_name in enumerate(self.param_ranges.keys()):
            best_params[param_name] = result.x[i]
        
        return best_params, result.fun, result
    
    def optimize(self, n_iterations=10, n_simulations=5, method='random'):
        """
        Optimize parameters using the specified method.
        
        Args:
            n_iterations (int): Number of optimization iterations.
            n_simulations (int): Number of simulations to run per iteration.
            method (str): Optimization method ('random' or 'differential_evolution').
            
        Returns:
            tuple: (best_params, best_error, all_results)
        """
        if method == 'random':
            return self.optimize_random_search(n_iterations, n_simulations)
        elif method == 'differential_evolution':
            return self.optimize_differential_evolution()
        else:
            raise ValueError(f"Unknown optimization method: {method}") 