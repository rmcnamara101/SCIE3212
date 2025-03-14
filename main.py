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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.tumor_growth import TumorGrowthModel
from src.visualization.plot_tumor import VolumeFractionPlotter
from src.visualization.animate_tumor import TumorAnimator
from src.models.initial_conditions import SphericalTumor
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


def main():
    # Original simulation code
    grid_shape = (80, 80, 80)
    dx = 200
    dt = 0.1
    params = SCIE3121_params
    steps = 430
    save_steps = 10

    initial_conditions = SphericalTumor(grid_shape, radius=5, nutrient_value=0.3)

    model = SCIE3121_MODEL(
        grid_shape=grid_shape,
        dx=dx,
        dt=dt,
        params=params,
        initial_conditions=initial_conditions,
        save_steps=save_steps
    )

    model.run_and_save_simulation(steps=steps, name="project_model_test")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run tumor growth simulations')
    parser.add_argument('--parallel', action='store_true', help='Run parameter sweep in parallel')
    args = parser.parse_args()
    
    if args.parallel:
        parallel_parameter_sweep()
    else:
        main()