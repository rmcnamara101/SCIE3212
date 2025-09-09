#!/usr/bin/env python3
"""
Test script to verify that pure upwind advection fixes the diffusion escape issue.

This script will test if removing the central difference component from the
advection scheme prevents cells from diffusing out and growing.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from src.growkit.Simulator import TumorGrowthSimulator

def test_pure_upwind_fix():
    """Test that pure upwind advection prevents diffusion escape."""
    
    print("Testing Pure Upwind Fix")
    print("=" * 50)
    
    cfg_path = "configs/csc-t-n.yaml"
    
    try:
        # Initialize simulator
        simulator = TumorGrowthSimulator(cfg_path)
        
        print(f"Configuration:")
        print(f"  Adhesion energy (m): {simulator.cfg['physics']['adhesion_energy']['m']}")
        print(f"  Time step (dt): {simulator.cfg['time']['dt']}")
        print(f"  Growth rates (lambda): {[p['dynamics']['lambda'] for p in simulator.cfg['populations'].values()]}")
        print(f"  Mobilities: {[p['dynamics']['mobility'] for p in simulator.cfg['populations'].values()]}")
        
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
        
        # Analyze final field to see if expansion is limited
        phi_hat_final, nutrient_field_final, host_field_final = simulator.field_manager.get_cell_fields()
        
        # Find the effective radius of the final cell distribution
        total_cell_density = np.sum(phi_hat_final, axis=0)
        nx, ny, nz = total_cell_density.shape
        center = np.array([nx//2, ny//2, nz//2])
        
        # Find the radius where cell density drops to 10% of maximum
        max_density = np.max(total_cell_density)
        threshold_density = 0.1 * max_density
        
        # Find the maximum radius with significant cell density
        max_radius = min(nx, ny, nz) // 2
        effective_radius = 0
        for r in range(max_radius):
            x = int(center[0] + r)
            y = int(center[1])
            z = int(center[2])
            if 0 <= x < nx and 0 <= y < ny and 0 <= z < nz:
                if total_cell_density[x, y, z] > threshold_density:
                    effective_radius = r
                else:
                    break
        
        print(f"\nFinal Analysis:")
        print(f"  Effective radius: {effective_radius} grid units")
        print(f"  Expected limit (if fixed): Should be much smaller than 26 grid units")
        
        # Check for excessive expansion
        mass_increase = total_masses[-1] - total_masses[0]
        mass_increase_percent = 100 * mass_increase / total_masses[0]
        
        print(f"  Mass increase: {mass_increase:.6f} ({mass_increase_percent:.2f}%)")
        
        # Check if cells are expanding to boundaries
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
            effective_radius < 20,  # Should be much smaller than 26
            boundary_density < 0.05,  # Boundary density should be very low
            boundary_density/interior_density < 0.1  # Boundary shouldn't be much denser than interior
        ]
        
        print(f"\nSuccess Criteria:")
        print(f"  Effective radius < 20: {effective_radius < 20} ({effective_radius})")
        print(f"  Boundary density < 0.05: {boundary_density < 0.05} ({boundary_density:.3f})")
        print(f"  Boundary/interior ratio < 0.1: {boundary_density/interior_density < 0.1} ({boundary_density/interior_density:.3f})")
        
        overall_success = all(success_criteria)
        
        if overall_success:
            print(f"\n✅ SUCCESS: Pure upwind fix appears to work!")
            print(f"   No excessive expansion detected.")
        else:
            print(f"\n❌ FAILURE: Still seeing excessive expansion.")
            print(f"   The diffusion escape issue may still be present.")
        
        return overall_success
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Pure Upwind Fix Test")
    print("=" * 50)
    
    success = test_pure_upwind_fix()
    
    if success:
        print("\n✅ Test completed successfully!")
        print("The pure upwind advection scheme appears to fix the diffusion escape issue.")
    else:
        print("\n❌ Test failed.")
        print("The diffusion escape issue may still be present.")
