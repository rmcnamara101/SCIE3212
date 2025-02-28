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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.tumor_growth import TumorGrowthModel
from src.visualization.plot_tumor import VolumeFractionPlotter
from src.visualization.animate_tumor import TumorAnimator

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
    

if __name__ == "__main__":

    model = TumorGrowthModel(grid_shape = (200, 200, 200), dx = 0.01, dt = 0.0001)
    #run_simulation(model, steps = 20)
    run_and_save_simulation(model, 300, '28-2-25_2')