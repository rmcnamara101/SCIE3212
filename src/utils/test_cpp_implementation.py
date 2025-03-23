import sys
import os 

# Add parent directory to Python path to allow importing src module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Only go up one level
sys.path.append(parent_dir)

import numpy as np
import matplotlib.pyplot as plt
from models.SCIE3121_model import SCIE3121_MODEL
from models.initial_conditions import RandomBlobTumor   
from utils.utils import SCIE3121_params

def compare_implementations(grid_shape=(100, 100, 100), steps=100):
    """Compare C++ and Python implementations."""
    # Initialize parameters
    dx = 75
    dt = 0.1
    params = SCIE3121_params
    initial_conditions = RandomBlobTumor(grid_shape=grid_shape)

    # Create two models: one with C++ and one with Python
    model_cpp = SCIE3121_MODEL(
        grid_shape=grid_shape,
        dx=dx,
        dt=dt,
        params=params,
        initial_conditions=initial_conditions,
        save_steps=10
    )

    # Temporarily disable C++ for Python model
    import src.models.SCIE3121_model as model_module
    original_use_cpp = model_module.USE_CPP
    model_module.USE_CPP = False

    model_py = SCIE3121_MODEL(
        grid_shape=grid_shape,
        dx=dx,
        dt=dt,
        params=params,
        initial_conditions=initial_conditions,
        save_steps=10
    )

    # Restore original USE_CPP value
    model_module.USE_CPP = original_use_cpp

    # Run simulations
    print("Running C++ implementation...")
    model_cpp.run_simulation(steps=steps)
    print("Running Python implementation...")
    model_py.run_simulation(steps=steps)

    # Compare results
    cpp_history = model_cpp.get_history()
    py_history = model_py.get_history()

    # Calculate differences
    fields = ['healthy cell volume fraction', 'diseased cell volume fraction', 
              'necrotic cell volume fraction', 'nutrient']
    
    print("\nComparison Results:")
    print("-" * 50)
    for field in fields:
        cpp_final = cpp_history[field][-1]
        py_final = py_history[field][-1]
        abs_diff = np.abs(cpp_final - py_final)
        rel_diff = np.abs(cpp_final - py_final) / (np.abs(py_final) + 1e-10)
        
        print(f"\n{field}:")
        print(f"Max absolute difference: {np.max(abs_diff):.2e}")
        print(f"Mean absolute difference: {np.mean(abs_diff):.2e}")
        print(f"Max relative difference: {np.max(rel_diff):.2e}")
        print(f"Mean relative difference: {np.mean(rel_diff):.2e}")

    # Visualize middle slice of final state
    fig, axes = plt.subplots(2, len(fields), figsize=(20, 8))
    mid_slice = grid_shape[0] // 2

    for i, field in enumerate(fields):
        cpp_data = cpp_history[field][-1][mid_slice]
        py_data = py_history[field][-1][mid_slice]
        
        vmin = min(cpp_data.min(), py_data.min())
        vmax = max(cpp_data.max(), py_data.max())
        
        axes[0, i].imshow(cpp_data, vmin=vmin, vmax=vmax)
        axes[0, i].set_title(f"C++ {field}")
        axes[1, i].imshow(py_data, vmin=vmin, vmax=vmax)
        axes[1, i].set_title(f"Python {field}")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compare_implementations() 