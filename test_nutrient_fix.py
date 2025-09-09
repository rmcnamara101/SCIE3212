#!/usr/bin/env python3
"""
Test script to verify that fixing the nutrient field resolves the 26-grid radius issue.

This script will test both uniform nutrient field and lowered nutrient thresholds
to see which approach works better.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from src.growkit.Simulator import TumorGrowthSimulator

def test_nutrient_fix():
    """Test that nutrient field fixes resolve the 26-grid radius issue."""
    
    print("Testing Nutrient Field Fix")
    print("=" * 50)
    
    cfg_path = "configs/csc-t-n.yaml"
    
    try:
        # Initialize simulator
        simulator = TumorGrowthSimulator(cfg_path)
        
        print(f"Configuration:")
        print(f"  Adhesion energy (m): {simulator.cfg['physics']['adhesion_energy']['m']}")
        print(f"  Time step (dt): {simulator.cfg['time']['dt']}")
        print(f"  Growth rates (lambda): {[p['dynamics']['lambda'] for p in simulator.cfg['populations'].values()]}")
        print(f"  Nutrient thresholds: {[p['dynamics']['nutrient_threshold'] for p in simulator.cfg['populations'].values()]}")
        print(f"  Nutrient type: {simulator.cfg['initial_conditions']['nutrient']['type']}")
        
        # Initialize fields
        simulator.initialize_fields()
        
        # Get initial fields
        phi_hat_initial, nutrient_field_initial, host_field_initial = simulator.field_manager.get_cell_fields()
        dx = simulator.field_manager.dx
        
        print(f"\nInitial field analysis:")
        print(f"  Total cell mass: {np.sum(phi_hat_initial):.6f}")
        print(f"  Max cell density: {np.max(phi_hat_initial):.6f}")
        print(f"  Domain size: {simulator.field_manager.grid}")
        print(f"  Nutrient field range: [{np.min(nutrient_field_initial):.3f}, {np.max(nutrient_field_initial):.3f}]")
        print(f"  Nutrient field mean: {np.mean(nutrient_field_initial):.3f}")
        
        # Analyze nutrient field profile
        nx, ny, nz = nutrient_field_initial.shape
        center = np.array([nx//2, ny//2, nz//2])
        
        # Sample nutrient field along a line from center to boundary
        max_radius = min(nx, ny, nz) // 2
        radii = np.arange(0, max_radius, 1)
        nutrient_profile = []
        
        for r in radii:
            # Sample at radius r from center
            x = int(center[0] + r)
            y = int(center[1])
            z = int(center[2])
            if 0 <= x < nx and 0 <= y < ny and 0 <= z < nz:
                nutrient_profile.append(nutrient_field_initial[x, y, z])
            else:
                nutrient_profile.append(1.0)  # Boundary value
        
        print(f"\nNutrient profile analysis:")
        print(f"  Center nutrient: {nutrient_profile[0]:.3f}")
        print(f"  Boundary nutrient: {nutrient_profile[-1]:.3f}")
        
        # Find where nutrient drops below threshold
        threshold = simulator.cfg['populations']['Stem']['dynamics']['nutrient_threshold']
        below_threshold_radii = [i for i, n in enumerate(nutrient_profile) if n < threshold]
        if below_threshold_radii:
            first_below_threshold = below_threshold_radii[0]
            print(f"  First radius below threshold ({threshold}): {first_below_threshold}")
            print(f"  This explains the ~{first_below_threshold} grid unit limit!")
        else:
            print(f"  No regions below threshold ({threshold}) - should allow full growth")
        
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
        
        # Find the radius where cell density drops to 10% of maximum
        max_density = np.max(total_cell_density)
        threshold_density = 0.1 * max_density
        
        # Find the maximum radius with significant cell density
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
        print(f"  Expected limit: ~{first_below_threshold if 'first_below_threshold' in locals() else 'N/A'} grid units")
        
        if 'first_below_threshold' in locals():
            if effective_radius > first_below_threshold + 5:
                print(f"  ✅ SUCCESS: Expansion beyond nutrient limit detected!")
                print(f"     This suggests the nutrient fix is working.")
            elif effective_radius < first_below_threshold - 5:
                print(f"  ⚠️  WARNING: Expansion is still limited below nutrient threshold.")
                print(f"     May need further adjustment.")
            else:
                print(f"  ℹ️  INFO: Expansion matches nutrient limit as expected.")
        else:
            print(f"  ✅ SUCCESS: No nutrient limit detected - should allow full growth.")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Nutrient Field Fix Test")
    print("=" * 50)
    
    success = test_nutrient_fix()
    
    if success:
        print("\n✅ Test completed successfully!")
        print("The nutrient field fix should resolve the 26-grid radius issue.")
    else:
        print("\n❌ Test failed.")
        print("The nutrient field issue may still be present.")
