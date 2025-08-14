#!/usr/bin/env python3
"""
Test script to verify that the fixed configuration resolves the source terms issue.
"""

import sys
import numpy as np
import yaml
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, "/Users/rileymcnamara/CODE/2025/SCIE3212/")

from src.growkit.Simulator import TumorGrowthSimulator

def test_fixed_configuration():
    """Test the fixed configuration to see if it resolves the source terms issue."""
    
    # Load fixed configuration
    cfg_path = "/Users/rileymcnamara/CODE/2025/SCIE3212/templates/test_templates/physics_fields_analysis_fixed.yaml"
    
    # Initialize simulator
    simulator = TumorGrowthSimulator(cfg_path)
    simulator.initialize_fields()
    
    print("=== Testing Fixed Configuration ===")
    print(f"Beta_N: {simulator.cell_dynamics.source_constructor.beta_N}")
    
    # Run a few steps and track what happens
    for step in range(1, 4):
        print(f"\n=== Step {step} Analysis ===")
        
        # Get current fields
        phi_hat, nutrient_field = simulator.field_manager.get_cell_fields()
        
        # Compute source terms
        source_terms = simulator.cell_dynamics.compute_source_terms(phi_hat, nutrient_field)
        
        print(f"Time: {simulator.time:.3f}")
        print(f"Nutrient field: min={np.min(nutrient_field):.3f}, max={np.max(nutrient_field):.3f}, mean={np.mean(nutrient_field):.3f}")
        
        # Check necrotic feedback
        V_N = np.sum(phi_hat[-1])
        G_N = np.exp(-simulator.cell_dynamics.source_constructor.beta_N * V_N)
        print(f"Total necrotic volume: {V_N:.3f}")
        print(f"Necrotic feedback G_N: {G_N:.6f}")
        
        for i, label in enumerate(simulator.field_manager.labels):
            phi_i = phi_hat[i]
            source_i = source_terms[i]
            
            print(f"{label}:")
            print(f"  Phi: min={np.min(phi_i):.3f}, max={np.max(phi_i):.3f}, mean={np.mean(phi_i):.3f}, sum={np.sum(phi_i):.3f}")
            print(f"  Source: min={np.min(source_i):.3f}, max={np.max(source_i):.3f}, mean={np.mean(source_i):.3f}")
            print(f"  Source negative fraction: {np.sum(source_i < 0) / source_i.size:.1%}")
            print(f"  Source positive fraction: {np.sum(source_i > 0) / source_i.size:.1%}")
            
            # Check for problematic regions
            cell_regions = phi_i > 0.01
            negative_sources = source_i < -0.01
            problematic_regions = cell_regions & negative_sources
            
            if np.any(problematic_regions):
                print(f"  PROBLEMATIC: {np.sum(problematic_regions)} regions with cells but negative sources")
                print(f"    Avg phi in problematic: {np.mean(phi_i[problematic_regions]):.3f}")
                print(f"    Avg source in problematic: {np.mean(source_i[problematic_regions]):.3f}")
                print(f"    Avg nutrient in problematic: {np.mean(nutrient_field[problematic_regions]):.3f}")
            else:
                print(f"  No problematic regions found")
        
        # Perform one simulation step
        success = simulator.step()
        if not success:
            print(f"Step {step} failed!")
            break
        
        # Check volume fraction constraints
        is_valid, max_deviation, mean_deviation = simulator.field_manager.check_volume_fraction_constraints()
        if not is_valid:
            print(f"WARNING: Volume fraction constraint violation - Max deviation: {max_deviation:.6f}, Mean deviation: {mean_deviation:.6f}")

if __name__ == "__main__":
    test_fixed_configuration()
