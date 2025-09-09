#!/usr/bin/env python3
"""
Test script to verify that reducing growth rates fixes the expansion issue.

This script will run a simulation with the corrected growth rates and check
that cells don't immediately expand to fill the domain boundaries.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from src.growkit.Simulator import TumorGrowthSimulator

def test_growth_rate_fix():
    """Test that reduced growth rates prevent excessive expansion."""
    
    print("Testing Growth Rate Fix")
    print("=" * 50)
    
    cfg_path = "configs/csc-t-n.yaml"
    
    try:
        # Initialize simulator
        simulator = TumorGrowthSimulator(cfg_path)
        
        print(f"Configuration:")
        print(f"  Adhesion energy (m): {simulator.cfg['physics']['adhesion_energy']['m']}")
        print(f"  Time step (dt): {simulator.cfg['time']['dt']}")
        print(f"  Growth rates (lambda): {[p['dynamics']['lambda'] for p in simulator.cfg['populations'].values()]}")
        print(f"  Death rates (mu): {[p['dynamics']['mu'] for p in simulator.cfg['populations'].values()]}")
        
        # Initialize fields
        simulator.initialize_fields()
        
        # Get initial fields
        phi_hat_initial, nutrient_field_initial, host_field_initial = simulator.field_manager.get_cell_fields()
        dx = simulator.field_manager.dx
        
        print(f"\nInitial field analysis:")
        print(f"  Total cell mass: {np.sum(phi_hat_initial):.6f}")
        print(f"  Max cell density: {np.max(phi_hat_initial):.6f}")
        print(f"  Domain size: {simulator.field_manager.grid}")
        
        # Run simulation for a few steps
        print(f"\nRunning simulation for {simulator.steps} steps...")
        
        # Store results
        times = [0.0]
        total_masses = [np.sum(phi_hat_initial)]
        max_densities = [np.max(phi_hat_initial)]
        
        for step in range(1, simulator.steps + 1):
            success = simulator.step()
            
            if not success:
                print(f"❌ Step {step} failed!")
                return False
            
            # Get current fields
            phi_hat, nutrient_field, host_field = simulator.field_manager.get_cell_fields()
            
            # Record statistics
            times.append(simulator.time)
            total_masses.append(np.sum(phi_hat))
            max_densities.append(np.max(phi_hat))
            
            print(f"  Step {step}: Time={simulator.time:.3f}, Mass={np.sum(phi_hat):.3f}, Max={np.max(phi_hat):.3f}")
        
        # Analyze results
        print(f"\nResults Analysis:")
        print(f"  Initial mass: {total_masses[0]:.6f}")
        print(f"  Final mass: {total_masses[-1]:.6f}")
        print(f"  Mass change: {total_masses[-1] - total_masses[0]:.6f}")
        print(f"  Mass change %: {100 * (total_masses[-1] - total_masses[0]) / total_masses[0]:.2f}%")
        
        print(f"  Initial max density: {max_densities[0]:.6f}")
        print(f"  Final max density: {max_densities[-1]:.6f}")
        print(f"  Max density change: {max_densities[-1] - max_densities[0]:.6f}")
        
        # Check for excessive expansion
        mass_increase = total_masses[-1] - total_masses[0]
        mass_increase_percent = 100 * mass_increase / total_masses[0]
        
        print(f"\nExpansion Analysis:")
        print(f"  Mass increase: {mass_increase:.6f} ({mass_increase_percent:.2f}%)")
        
        # Check if cells are expanding to boundaries
        phi_hat_final, _, _ = simulator.field_manager.get_cell_fields()
        
        # Check boundary regions (first and last few grid points)
        nx, ny, nz = phi_hat_final.shape[1:]
        boundary_thickness = 5
        
        # Check if there's significant cell density near boundaries
        boundary_density = np.mean([
            np.mean(phi_hat_final[:, :boundary_thickness, :, :]),  # x=0 boundary
            np.mean(phi_hat_final[:, -boundary_thickness:, :, :]),  # x=nx boundary
            np.mean(phi_hat_final[:, :, :boundary_thickness, :]),  # y=0 boundary
            np.mean(phi_hat_final[:, :, -boundary_thickness:, :]),  # y=ny boundary
            np.mean(phi_hat_final[:, :, :, :boundary_thickness]),  # z=0 boundary
            np.mean(phi_hat_final[:, :, :, -boundary_thickness:])   # z=nz boundary
        ])
        
        interior_density = np.mean(phi_hat_final[:, boundary_thickness:-boundary_thickness, 
                                                boundary_thickness:-boundary_thickness, 
                                                boundary_thickness:-boundary_thickness])
        
        print(f"  Boundary density: {boundary_density:.6f}")
        print(f"  Interior density: {interior_density:.6f}")
        print(f"  Boundary/interior ratio: {boundary_density/interior_density:.3f}")
        
        # Success criteria
        success_criteria = [
            mass_increase_percent < 50,  # Mass shouldn't increase by more than 50%
            boundary_density < 0.1,      # Boundary density should be low
            boundary_density/interior_density < 0.5  # Boundary shouldn't be much denser than interior
        ]
        
        print(f"\nSuccess Criteria:")
        print(f"  Mass increase < 50%: {mass_increase_percent < 50} ({mass_increase_percent:.1f}%)")
        print(f"  Boundary density < 0.1: {boundary_density < 0.1} ({boundary_density:.3f})")
        print(f"  Boundary/interior ratio < 0.5: {boundary_density/interior_density < 0.5} ({boundary_density/interior_density:.3f})")
        
        overall_success = all(success_criteria)
        
        if overall_success:
            print(f"\n✅ SUCCESS: Growth rate fix appears to work!")
            print(f"   No excessive expansion detected.")
        else:
            print(f"\n❌ FAILURE: Still seeing excessive expansion.")
            print(f"   May need further growth rate reduction.")
        
        return overall_success
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Growth Rate Fix Test")
    print("=" * 50)
    
    success = test_growth_rate_fix()
    
    if success:
        print("\n✅ Test completed successfully!")
        print("The reduced growth rates appear to fix the expansion issue.")
    else:
        print("\n❌ Test failed.")
        print("The expansion issue may still be present.")
