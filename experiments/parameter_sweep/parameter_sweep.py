import numpy as np
from tqdm import tqdm
from copy import deepcopy
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.models.SCIE3121_model import SCIE3121_MODEL
from src.models.initial_conditions import SphericalTumor
from src.utils.utils import SCIE3121_params

def collect_simulation_metrics(model, steps):
    """Collect key metrics during simulation without saving full history."""
    volumes_H = []
    volumes_D = []
    volumes_N = []
    radii = []
    
    for step in range(steps):
        if step % model.save_steps == 0:
            # Calculate volumes
            vol_H = np.sum(model.phi_H) * (model.dx ** 3)
            vol_D = np.sum(model.phi_D) * (model.dx ** 3)
            vol_N = np.sum(model.phi_N) * (model.dx ** 3)
            
            # Calculate radius (using total tumor volume)
            total_field = model.phi_H + model.phi_D + model.phi_N
            tumor_mask = total_field >= 0.05  # threshold for radius calculation
            if np.any(tumor_mask):
                radius = np.max(np.sqrt(np.sum(np.array(np.where(tumor_mask)) ** 2, axis=0))) * model.dx
            else:
                radius = 0
                
            volumes_H.append(vol_H)
            volumes_D.append(vol_D)
            volumes_N.append(vol_N)
            radii.append(radius)
        
        model._update(step)
    
    return {
        'volumes_H': np.array(volumes_H),
        'volumes_D': np.array(volumes_D),
        'volumes_N': np.array(volumes_N),
        'radii': np.array(radii),
        'timesteps': np.arange(0, steps, model.save_steps)
    }

def run_parameter_sweep():
    # Base simulation parameters
    grid_shape = (50, 50, 50)
    dx = 0.1
    dt = 0.0001
    steps = 5000
    save_steps = 150

    # Parameter sweep settings
    lambda_H_base = 3.0
    lambda_D_ratios = np.linspace(0.2, 2.0, 20)
    
    # Create directory for results if it doesn't exist
    if not os.path.exists('data/parameter_sweep'):
        os.makedirs('data/parameter_sweep')

    # Initialize results dictionary
    sweep_results = {
        'lambda_D_ratios': lambda_D_ratios,
        'metrics': []
    }

    # Run simulations for each ratio
    for ratio in tqdm(lambda_D_ratios, desc="Running parameter sweep"):
        # Create modified parameters
        current_params = deepcopy(SCIE3121_params)
        current_params['lambda_H'] = lambda_H_base
        current_params['lambda_D'] = lambda_H_base * ratio

        # Initialize model
        model = SCIE3121_MODEL(
            grid_shape=grid_shape,
            dx=dx,
            dt=dt,
            params=current_params,
            initial_conditions=SphericalTumor(
                grid_shape=grid_shape,
                radius=7,
                nutrient_value=0.1
            ),
            save_steps=save_steps
        )

        # Collect metrics during simulation
        metrics = collect_simulation_metrics(model, steps)
        sweep_results['metrics'].append(metrics)

    # Save all results in a single file
    np.savez('data/parameter_sweep/sweep_results.npz',
             lambda_D_ratios=sweep_results['lambda_D_ratios'],
             metrics=sweep_results['metrics'],
             metadata={
                 'lambda_H_base': lambda_H_base,
                 'grid_shape': grid_shape,
                 'dx': dx,
                 'dt': dt,
                 'steps': steps,
                 'save_steps': save_steps
             })

if __name__ == "__main__":
    run_parameter_sweep() 