#######################################################################################################
#######################################################################################################
#
#
#   This is the main access point for running tumor growth simulations.
#
#   It will initialize the tumor growth model, run the simulation, and then save the history of the
#   simulation. 
#
#   The user can then access then analyse the data.
#
#
# Author:
#   - Riley Jae McNamara
#
# Date:
#   - 2025-02-19
#
#
#
#######################################################################################################
#######################################################################################################


import os
import sys
import numpy as np
from joblib import Parallel, delayed
import time
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.tumor_growth import TumorGrowthModel
from src.visualization.plot_tumor import VolumeFractionPlotter
from src.visualization.animate_tumor import TumorAnimator
from src.models.initial_conditions import *
from src.utils.utils import experimental_params, SCIE3121_params
from src.models.SCIE3121_model import SCIE3121_MODEL

def run_and_save_simulation(model: TumorGrowthModel, steps: int, name: str):
    """
    Run the simulation and save history data as a NumPy .npz file.
    
    Args:
        model (TumorGrowthModel): The tumor growth model instance.
        steps (int): Number of simulation steps to run.
        name (str): Base name for the output file.
    """
    model.run_simulation(steps=steps)
    history = model.get_history()
    history['Simulation Metadata'] = {'dx': model.dx, 
                                      'dt': model.dt, 
                                      'steps': steps
                                     }
    
    file_str = f"data/{name}_sim_data.npz"
    np.savez(file_str, **history)


def run_simulation(model: TumorGrowthModel, steps: int):
    """
    General testing space for running simulations and visualizing results.
    """
    model.run_simulation(steps=steps)
    history = model.get_history()

    plotter = VolumeFractionPlotter(model)
    animator = TumorAnimator(model)

    # uncomment to plot surface
    #plotter.plot_volume_fractions(20)
    plotter.plot_volume_fraction_evolution()

    #animator.animate_tumor_slices()
    #animator.animate_single_slice()
    animator.animate_tumor_growth_isosurfaces()


def load_simulation_history(file_name: str) -> dict:
    """
    Load simulation history from an .npz file and reconstruct the original history dictionary.
    
    Args:
        file_name (str): Path to the .npz file (e.g., "simulation_sim_data.npz").
    
    Returns:
        dict: The reconstructed history dictionary with NumPy arrays.
    
    Raises:
        FileNotFoundError: If the specified .npz file does not exist.
        ValueError: If the loaded data is not in the expected format.
    """
    try:
        # Load the .npz file
        data = np.load(file_name, allow_pickle=False)
        
        # Reconstruct the history dictionary
        history = {}
        for key in data.keys():
            history[key] = data[key]
        
        # Basic validation to ensure expected keys are present
        if 'step' not in history:
            raise ValueError("Loaded data does not contain 'step' key, invalid simulation history")
        
        return history
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find file: {file_name}")
    except Exception as e:
        raise ValueError(f"Error loading simulation history from {file_name}: {str(e)}")
    

def run_simulation_with_params(grid_shape, dx, dt, params, steps, save_steps, radius, nutrient_value, name_suffix=""):
    """Run a single simulation with given parameters and return the model"""
    initial_conditions = SphericalTumor(grid_shape, radius=radius, nutrient_value=nutrient_value)
    
    model = SCIE3121_MODEL(
        grid_shape=grid_shape,
        dx=dx,
        dt=dt,
        params=params,
        initial_conditions=initial_conditions,
        save_steps=save_steps
    )
    
    model.run_and_save_simulation(steps=steps, name=f"model_test_{name_suffix}")
    return model


def parallel_parameter_sweep():
    """Run multiple simulations in parallel with different parameters"""
    # Base simulation parameters
    grid_shape = (40, 40, 40)
    dx = 1
    dt = 0.01
    base_params = SCIE3121_params.copy()
    steps = 5000
    save_steps = 50
    
    # Parameter variations to test
    variations = []
    
    # Example: Vary tumor radius
    for radius in [3, 4, 5, 6]:
        variations.append({
            'grid_shape': grid_shape,
            'dx': dx, 
            'dt': dt,
            'params': base_params.copy(),
            'steps': steps,
            'save_steps': save_steps,
            'radius': radius,
            'nutrient_value': 1.0,
            'name_suffix': f"radius_{radius}"
        })
    
    # Example: Vary diffusion coefficient
    diffusion_values = [0.5, 1.0, 2.0, 3.0]
    for i, diff_coef in enumerate(diffusion_values):
        params_copy = base_params.copy()
        params_copy['D_n'] = diff_coef  # Assuming 'D_n' is the diffusion coefficient parameter
        variations.append({
            'grid_shape': grid_shape,
            'dx': dx, 
            'dt': dt,
            'params': params_copy,
            'steps': steps,
            'save_steps': save_steps,
            'radius': 4,
            'nutrient_value': 1.0,
            'name_suffix': f"diffusion_{diff_coef}"
        })
    
    # Run simulations in parallel
    n_jobs = min(len(variations), os.cpu_count())
    print(f"Running {len(variations)} simulations with {n_jobs} parallel jobs")
    
    Parallel(n_jobs=n_jobs)(
        delayed(run_simulation_with_params)(**params) for params in variations
    )


def run_parameter_optimization(real_data_path, param_ranges, n_iterations=50, n_simulations=10):
    """
    Run parameter optimization to find the best parameters that match real data.
    
    Args:
        real_data_path (str): Path to the real tumor image data (time series).
        param_ranges (dict): Dictionary of parameter ranges to explore {param_name: (min, max)}.
        n_iterations (int): Number of optimization iterations.
        n_simulations (int): Number of simulations to run per iteration.
        
    Returns:
        dict: Best parameters found and their corresponding error metrics.
    """
    from src.optimization.parameter_fitting import ParameterOptimizer
    
    # Initialize the parameter optimizer
    optimizer = ParameterOptimizer(
        real_data_path=real_data_path,
        param_ranges=param_ranges,
        base_params=SCIE3121_params.copy(),
        grid_shape=(100, 100, 100),
        dx=200,
        dt=0.1,
        steps=600,
        save_steps=50
    )
    
    # Run the optimization
    best_params, best_error, all_results = optimizer.optimize(
        n_iterations=n_iterations,
        n_simulations=n_simulations
    )
    
    print(f"Best parameters found: {best_params}")
    print(f"Best error: {best_error}")
    
    # Save optimization results
    np.savez(
        "data/optimization_results.npz",
        best_params=best_params,
        best_error=best_error,
        all_results=all_results
    )
    
    return best_params, best_error, all_results


def compare_with_real_data(simulation_file, real_data_path):
    """
    Compare simulation results with real data and visualize the comparison.
    
    Args:
        simulation_file (str): Path to the simulation results (.npz file).
        real_data_path (str): Path to the real tumor image data.
    """
    from src.optimization.data_comparison import DataComparison
    
    # Load simulation data
    sim_history = load_simulation_history(simulation_file)
    
    # Initialize comparison object
    comparison = DataComparison(
        simulation_data=sim_history,
        real_data_path=real_data_path
    )
    
    # Calculate metrics
    metrics = comparison.calculate_metrics()
    print(f"Comparison metrics: {metrics}")
    
    # Visualize comparison
    comparison.visualize_comparison()


def compare_implementations(grid_shape=(100, 100, 100), steps=20, save_steps=1):
    """
    Compare C++ and Python implementations for performance and accuracy.
    
    Args:
        grid_shape (tuple): Shape of the simulation grid
        steps (int): Number of simulation steps
        save_steps (int): How often to save results
    """
    dx = 75
    dt = 0.1
    params = SCIE3121_params
    initial_conditions = SphericalTumor(grid_shape, radius=5, nutrient_value=1.0)

    # Run Python version
    print("Running Python implementation...")
    model_py = SCIE3121_MODEL(
        grid_shape=grid_shape,
        dx=dx,
        dt=dt,
        params=params,
        initial_conditions=initial_conditions,
        save_steps=save_steps
    )
    
    t_start = time.time()
    model_py.run_simulation(steps=steps)
    py_time = time.time() - t_start
    
    # Force Python implementation for comparison
    print("\nRunning C++ implementation...")
    os.environ['USE_CPP'] = '1'  # Enable C++ implementation
    
    model_cpp = SCIE3121_MODEL(
        grid_shape=grid_shape,
        dx=dx,
        dt=dt,
        params=params,
        initial_conditions=initial_conditions,
        save_steps=save_steps
    )
    
    t_start = time.time()
    model_cpp.run_simulation(steps=steps)
    cpp_time = time.time() - t_start
    
    # Compare results
    py_history = model_py.get_history()
    cpp_history = model_cpp.get_history()
    
    # Calculate differences
    fields = ['healthy cell volume fraction', 'diseased cell volume fraction', 
              'necrotic cell volume fraction', 'nutrient']
    
    print("\nPerformance Comparison:")
    print(f"Python time: {py_time:.2f} seconds")
    print(f"C++ time: {cpp_time:.2f} seconds")
    print(f"Speedup: {py_time/cpp_time:.2f}x")
    
    print("\nAccuracy Comparison:")
    for field in fields:
        py_data = py_history[field][-1]  # Compare final state
        cpp_data = cpp_history[field][-1]
        
        abs_diff = np.abs(py_data - cpp_data)
        rel_diff = abs_diff / (np.abs(py_data) + 1e-10)
        
        print(f"\n{field}:")
        print(f"Max absolute difference: {np.max(abs_diff):.2e}")
        print(f"Mean absolute difference: {np.mean(abs_diff):.2e}")
        print(f"Max relative difference: {np.max(rel_diff):.2e}")
        print(f"Mean relative difference: {np.mean(rel_diff):.2e}")

def benchmark_scaling():
    """
    Benchmark how the implementations scale with grid size.
    """
    steps = 50
    grid_sizes = [(20,20,20), (40,40,40), (60,60,60), (80,80,80)]
    
    py_times = []
    cpp_times = []
    
    for grid_shape in grid_sizes:
        print(f"\nTesting grid size: {grid_shape}")
        
        # Python version
        os.environ['USE_CPP'] = '0'
        model = SCIE3121_MODEL(grid_shape=grid_shape, dx=0.1, dt=0.001, 
                             params=SCIE3121_params,
                             initial_conditions=SphericalTumor(grid_shape))
        
        t_start = time.time()
        model.run_simulation(steps=steps)
        py_times.append(time.time() - t_start)
        
        # C++ version
        os.environ['USE_CPP'] = '1'
        model = SCIE3121_MODEL(grid_shape=grid_shape, dx=0.1, dt=0.001,
                             params=SCIE3121_params,
                             initial_conditions=SphericalTumor(grid_shape))
        
        t_start = time.time()
        model.run_simulation(steps=steps)
        cpp_times.append(time.time() - t_start)
    
    # Print results
    print("\nScaling Results:")
    print("Grid Size | Python Time | C++ Time | Speedup")
    print("-" * 50)
    for i, grid_shape in enumerate(grid_sizes):
        n_points = grid_shape[0] * grid_shape[1] * grid_shape[2]
        print(f"{grid_shape} | {py_times[i]:.2f}s | {cpp_times[i]:.2f}s | {py_times[i]/cpp_times[i]:.2f}x")

def main():
    parser = argparse.ArgumentParser(description='Run tumor growth simulations')
    parser.add_argument('--compare', action='store_true', help='Compare Python and C++ implementations')
    parser.add_argument('--benchmark', action='store_true', help='Run scaling benchmark')
    parser.add_argument('--grid-size', type=int, nargs=3, default=[100,100,100], 
                       help='Grid size for comparison (nx ny nz)')
    parser.add_argument('--steps', type=int, default=20, 
                       help='Number of simulation steps')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_implementations(
            grid_shape=tuple(args.grid_size),
            steps=args.steps
        )
    elif args.benchmark:
        benchmark_scaling()
    else:
        # Original simulation code
        grid_shape = (90, 90, 90)
        dx = 75
        dt = 0.1
        params = SCIE3121_params
        steps = 150
        save_steps = 5

        #initial_conditions = SphericalTumor(grid_shape, radius=5, nutrient_value=0.5)
        #initial_conditions = InvasiveTumor(grid_shape, base_radius=5, nutrient_value=1.0)
        #initial_conditions = MultipleTumors(grid_shape, centers=[(0.3, 0.3, 0.3), (0.7, 0.7, 0.7)], radii=[5, 4], nutrient_value=1.0)
        initial_conditions = RandomBlobTumor(grid_shape, base_radius=6, nutrient_value=0.1)

        model = SCIE3121_MODEL(
            grid_shape=grid_shape,
            dx=dx,
            dt=dt,
            params=params,
            initial_conditions=initial_conditions,
            save_steps=save_steps
        )

        model.run_and_save_simulation(steps=steps, name="base")

        # Uncomment to run parameter optimization
        # Example parameter ranges to explore
        # param_ranges = {
        #     'lambda_H': (0.3, 1.0),
        #     'lambda_D': (0.4, 1.2),
        #     'mu_H': (0.1, 0.4),
        #     'mu_D': (0.1, 0.4),
        #     'D_n': (0.5, 2.0)
        # }
        # run_parameter_optimization(
        #     real_data_path="data/real_tumor_images/",
        #     param_ranges=param_ranges,
        #     n_iterations=10,
        #     n_simulations=5
        # )
        
        # Uncomment to compare with real data
        # compare_with_real_data(
        #     simulation_file="data/project_model_test_sim_data.npz",
        #     real_data_path="data/real_tumor_images/"
        # )


if __name__ == "__main__":
    main()