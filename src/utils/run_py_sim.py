import numpy as np
import os
import sys
from datetime import datetime
from tqdm import tqdm

# Add parent directory to Python path to allow importing src module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
models_dir = os.path.join(os.path.dirname(current_dir), "models")
print(f"Adding to Python path: {parent_dir}, {models_dir}")
sys.path.extend([parent_dir, models_dir])

# Force Python implementation by modifying sys.modules before any imports
sys.modules['cpp_simulation'] = None

from src.models.SCIE3121_model import SCIE3121_MODEL
from src.models.initial_conditions import SphericalTumor
from src.utils.utils import SCIE3121_params

def run_and_save_py_sim(
    grid_shape=(100, 100, 100),
    steps=100,
    save_steps=10,
    dx=75,
    dt=None,
    save_dir="simulation_results",
    params=None
):
    """
    Run a simulation using the Python implementation and save results.
    
    Args:
        grid_shape (tuple): Shape of the simulation grid (default: (100, 100, 100))
        steps (int): Number of simulation steps (default: 100)
        save_steps (int): Save results every n steps (default: 10)
        dx (float): Spatial step size (default: 75)
        dt (float): Time step size (default: None, adaptive)
        save_dir (str): Directory to save results (default: "simulation_results")
        params (dict): Model parameters (default: None, uses SCIE3121_params)
    
    Returns:
        str: Path to the saved results
    """
    print(f"Initializing Python simulation with grid shape: {grid_shape}")
    
    # Adjust dt based on grid size for stability
    if dt is None:
        grid_size = max(grid_shape)
        dt = 0.1 / (grid_size / 50)
    
    print(f"Using dt = {dt} for grid size {grid_shape}")
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"py_sim_{timestamp}")
    os.makedirs(save_path, exist_ok=True)
    
    # Initialize conditions
    print("Creating initial conditions...")
    initial_conditions = SphericalTumor(grid_shape=grid_shape, radius=5, nutrient_value=1.0)
    
    # Initialize model
    print("Creating Python model...")
    model = SCIE3121_MODEL(
        grid_shape=grid_shape,
        dx=dx,
        dt=dt,
        params=params or SCIE3121_params,
        initial_conditions=initial_conditions,
        save_steps=save_steps
    )
    
    # Run simulation with progress bar
    print(f"Running simulation for {steps} steps...")
    for step in tqdm(range(steps), desc="Running Simulation"):
        try:
            model._update(step)
        except Exception as e:
            print(f"Error at step {step}: {str(e)}")
            raise
    
    # Get results
    history = model.get_history()
    
    # Create the history dictionary
    history_dict = {
        'healthy cell volume fraction': np.array(history['healthy cell volume fraction']),
        'diseased cell volume fraction': np.array(history['diseased cell volume fraction']),
        'necrotic cell volume fraction': np.array(history['necrotic cell volume fraction']),
        'nutrient': np.array(history['nutrient']),
        'step': np.array(history['step']),
        'Simulation Metadata': {
            "grid_shape": grid_shape,
            "steps": steps,
            "save_steps": save_steps,
            "dx": dx,
            "dt": dt,
            **(params or SCIE3121_params)
        }
    }
    
    # Save results
    print(f"Saving results to {save_path}")
    np.savez(
        os.path.join(save_path, "simulation_data.npz"),
        history=history_dict
    )
    
    # Verify saved file
    saved_file = np.load(os.path.join(save_path, "simulation_data.npz"), allow_pickle=True)
    print("Saved keys:", saved_file.files)
    history_data = saved_file['history'].item()
    print("History keys:", history_data.keys())
    print("Data shapes:", {k: np.array(v).shape if not isinstance(v, dict) else 'dict' 
                          for k, v in history_data.items()})
    
    print("Python simulation completed and saved successfully!")
    return save_path

if __name__ == "__main__":
    # Test with a smaller grid for Python version
    save_path = run_and_save_py_sim(
        grid_shape=(40, 40, 40),  # Smaller grid for Python version
        steps=100,
        save_steps=20,
        dx=75,
        dt=0.1,
        params=SCIE3121_params
    )
    print(f"Results saved to: {save_path}")
