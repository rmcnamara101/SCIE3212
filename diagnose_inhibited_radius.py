#!/usr/bin/env python3
"""
Diagnostic script to investigate why inhibited radius is not appearing.
"""

import sys
from pathlib import Path
import numpy as np

# Add project to path
proj = Path(__file__).parent
sys.path.insert(0, str(proj))

from src.growkit.Simulator import TumorGrowthSimulator
from src.growkit.PlotEngine.ObservableUtils import ObservableUtils

def diagnose_inhibited_radius(simulation_path, step_idx=-1):
    """Diagnose why inhibited radius is not appearing."""
    
    print(f"Loading simulation from {simulation_path}...")
    simulator = TumorGrowthSimulator(proj / "configs" / "T_N.yaml")
    simulation_data = simulator.load_simulation_data(simulation_path)
    
    # Get the last step
    num_steps = len(simulation_data["metadata"]["saved_steps"])
    if step_idx < 0:
        step_idx = num_steps - 1
    
    print(f"\nAnalyzing step {step_idx} of {num_steps}")
    
    # Get fields
    phi_hat = simulation_data["field_data"]["phi_hat"][step_idx]
    nutrient_field = simulation_data["field_data"]["nutrient_fields"][step_idx]
    total_density = np.sum(phi_hat, axis=0)
    
    # Get config
    config = simulation_data["metadata"]["config"]
    
    # Extract parameters
    populations = config["populations"]
    viable_populations = {k: v for k, v in populations.items() if k.lower() != "necrotic"}
    
    lambda_rates = []
    nutrient_thresholds = []
    for pop_name, pop_config in viable_populations.items():
        if "dynamics" in pop_config:
            dynamics = pop_config["dynamics"]
            lambda_rates.append(dynamics.get("lambda", 0.0))
            nutrient_thresholds.append(dynamics.get("nutrient_threshold", 0.0))
    
    lambda_rates = np.array(lambda_rates, dtype=np.float32)
    nutrient_thresholds = np.array(nutrient_thresholds, dtype=np.float32)
    
    k_switch = config.get("nutrient", {}).get("dynamics", {}).get("k", 10.0)
    beta_N = populations.get("Necrotic", {}).get("dynamics", {}).get("beta_N", 0.0005)
    
    print(f"\nConfiguration:")
    print(f"  Lambda rates: {lambda_rates}")
    print(f"  Nutrient thresholds (death): {nutrient_thresholds}")
    print(f"  k_switch: {k_switch}")
    print(f"  beta_N: {beta_N}")
    
    # Initialize utils
    grid_size = simulation_data["metadata"]["grid_size"]
    dx = config.get("domain", {}).get("dx", 1.0)
    utils = ObservableUtils(grid_size, dx)
    
    # Compute Theta_death
    M = len(lambda_rates)
    Theta_death = np.zeros((M, *nutrient_field.shape), dtype=np.float32)
    for i in range(M):
        n_thresh_death = nutrient_thresholds[i]
        Theta_death[i] = 0.5 * (1 + np.tanh(k_switch * (nutrient_field - n_thresh_death)))
    
    # Compute G_N
    if phi_hat.shape[0] > M:
        V_N = np.sum(phi_hat[-1])
    else:
        V_N = 0.0
    G_N = np.exp(-beta_N * V_N)
    
    print(f"\nNecrotic feedback:")
    print(f"  V_N (necrotic volume): {V_N}")
    print(f"  G_N: {G_N}")
    
    # Compute proliferation source
    proliferation_source = np.zeros_like(nutrient_field)
    for i in range(M):
        proliferation_source += G_N * nutrient_field * lambda_rates[i] * phi_hat[i]
    
    # Analyze tumor region
    density_threshold = 0.1
    tumor_mask = total_density > density_threshold
    
    if not np.any(tumor_mask):
        print("\nERROR: No tumor region found!")
        return
    
    print(f"\nTumor region analysis:")
    print(f"  Tumor voxels: {np.sum(tumor_mask)}")
    print(f"  Nutrient range in tumor: [{np.min(nutrient_field[tumor_mask]):.3f}, {np.max(nutrient_field[tumor_mask]):.3f}]")
    print(f"  Proliferation range in tumor: [{np.min(proliferation_source[tumor_mask]):.6f}, {np.max(proliferation_source[tumor_mask]):.6f}]")
    print(f"  Theta_death range in tumor: [{np.min(Theta_death[:, tumor_mask]):.3f}, {np.max(Theta_death[:, tumor_mask]):.3f}]")
    
    # Check conditions for inhibited region
    death_switch_threshold = 0.95
    proliferation_threshold = 0.01
    
    min_Theta_death = np.min(Theta_death, axis=0)
    death_off_mask = min_Theta_death > death_switch_threshold
    proliferation_low_mask = proliferation_source < proliferation_threshold
    
    print(f"\nCondition analysis:")
    print(f"  Death OFF voxels (Theta_death > {death_switch_threshold}): {np.sum(death_off_mask & tumor_mask)}")
    print(f"  Low proliferation voxels (< {proliferation_threshold}): {np.sum(proliferation_low_mask & tumor_mask)}")
    print(f"  Inhibited region voxels (both conditions): {np.sum(death_off_mask & proliferation_low_mask & tumor_mask)}")
    
    # Check with percentile threshold
    proliferation_in_tumor = proliferation_source[tumor_mask]
    if len(proliferation_in_tumor) > 0 and np.max(proliferation_in_tumor) > 0:
        percentile_threshold = np.percentile(proliferation_in_tumor, 25.0)
        effective_threshold = max(proliferation_threshold, percentile_threshold)
        print(f"\nPercentile-based threshold:")
        print(f"  25th percentile: {percentile_threshold:.6f}")
        print(f"  Effective threshold: {effective_threshold:.6f}")
        proliferation_low_mask_percentile = proliferation_source < effective_threshold
        inhibited_percentile = np.sum(death_off_mask & proliferation_low_mask_percentile & tumor_mask)
        print(f"  Inhibited region with percentile threshold: {inhibited_percentile} voxels")
    
    # Radial analysis
    com = utils.calculate_center_of_mass(total_density)
    print(f"\nTumor center of mass: {com}")
    
    r_squared = (utils.X - com[0])**2 + (utils.Y - com[1])**2 + (utils.Z - com[2])**2
    radial_distances = np.sqrt(r_squared)
    
    # Analyze by radial bins
    max_radius = np.max(radial_distances[tumor_mask])
    num_bins = 50
    radial_bins = np.linspace(0, max_radius, num_bins + 1)
    radial_centers = (radial_bins[:-1] + radial_bins[1:]) / 2
    
    print(f"\nRadial analysis (max radius: {max_radius:.2f}):")
    print(f"{'Radius':<10} {'Nutrient':<12} {'Prolif':<12} {'Theta_death':<12} {'Death_OFF':<10} {'Low_Pro':<10} {'Inhibited':<10}")
    print("-" * 80)
    
    for i in range(min(20, num_bins)):  # Show first 20 bins
        r_min = radial_bins[i]
        r_max = radial_bins[i + 1]
        mask = (radial_distances >= r_min) & (radial_distances < r_max) & tumor_mask
        
        if np.any(mask):
            avg_nutrient = np.mean(nutrient_field[mask])
            avg_proliferation = np.mean(proliferation_source[mask])
            avg_theta_death = np.mean(min_Theta_death[mask])
            death_off_count = np.sum(death_off_mask[mask])
            low_pro_count = np.sum(proliferation_low_mask[mask])
            inhibited_count = np.sum(death_off_mask[mask] & proliferation_low_mask[mask])
            
            print(f"{radial_centers[i]:<10.2f} {avg_nutrient:<12.3f} {avg_proliferation:<12.6f} "
                  f"{avg_theta_death:<12.3f} {death_off_count:<10} {low_pro_count:<10} {inhibited_count:<10}")

if __name__ == "__main__":
    # Default simulation path
    sim_path = proj / "laboratory" / "saved_simulations" / "simulation_data.npz"
    
    if len(sys.argv) > 1:
        sim_path = Path(sys.argv[1])
    
    if not sim_path.exists():
        print(f"Error: Simulation file not found at {sim_path}")
        sys.exit(1)
    
    diagnose_inhibited_radius(sim_path)

