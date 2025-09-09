#!/usr/bin/env python3
"""
Debug script to identify the diffusion process that allows cells to escape.

This script will analyze the mass flux and advection terms to find what's
causing cells to diffuse out and then grow.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from src.growkit.Simulator import TumorGrowthSimulator

def debug_diffusion_escape():
    """Debug the diffusion process that allows cell escape."""
    
    print("Debugging Diffusion Escape")
    print("=" * 50)
    
    cfg_path = "configs/csc-t-n.yaml"
    
    try:
        # Initialize simulator
        simulator = TumorGrowthSimulator(cfg_path)
        
        print(f"Configuration:")
        print(f"  Adhesion energy (m): {simulator.cfg['physics']['adhesion_energy']['m']}")
        print(f"  Mobilities: {[p['dynamics']['mobility'] for p in simulator.cfg['populations'].values()]}")
        print(f"  Growth rates (lambda): {[p['dynamics']['lambda'] for p in simulator.cfg['populations'].values()]}")
        
        # Initialize fields
        simulator.initialize_fields()
        
        # Get initial fields
        phi_hat, nutrient_field, host_field = simulator.field_manager.get_cell_fields()
        dx = simulator.field_manager.dx
        
        print(f"\nInitial field analysis:")
        print(f"  Total cell mass: {np.sum(phi_hat):.6f}")
        print(f"  Max cell density: {np.max(phi_hat):.6f}")
        print(f"  Cell density shape: {phi_hat.shape}")
        
        # Compute each term in the dynamics equation
        print(f"\nComputing dynamics terms...")
        
        # 1. Energy derivative
        energy_deriv = simulator.cell_dynamics.compute_energy_derivative(np.sum(phi_hat, axis=0), dx)
        print(f"  Energy derivative:")
        print(f"    Max: {np.max(energy_deriv):.6f}")
        print(f"    Mean: {np.mean(energy_deriv):.6f}")
        print(f"    Std: {np.std(energy_deriv):.6f}")
        
        # 2. Mass flux
        J_hat = simulator.cell_dynamics.compute_mass_fluxes(phi_hat, dx, energy_deriv)
        mass_flux_magnitude = np.sqrt(J_hat[:, 0]**2 + J_hat[:, 1]**2 + J_hat[:, 2]**2)
        print(f"  Mass flux:")
        print(f"    Max magnitude: {np.max(mass_flux_magnitude):.6f}")
        print(f"    Mean magnitude: {np.mean(mass_flux_magnitude):.6f}")
        print(f"    Std magnitude: {np.std(mass_flux_magnitude):.6f}")
        
        # 3. Check if mass flux is non-zero even with zero energy derivative
        if np.max(np.abs(energy_deriv)) < 1e-10:
            print(f"    ⚠️  WARNING: Energy derivative is essentially zero, but mass flux is non-zero!")
            print(f"    This suggests mobility is causing diffusion even without adhesion energy.")
        
        # 4. Analyze mass flux divergence (the diffusion term)
        from src.growkit.MathEngine.Operators import _gradient_neumann
        mass_flux_divergence = np.zeros_like(phi_hat)
        for i in range(phi_hat.shape[0]):
            Jx = J_hat[i, 0]
            Jy = J_hat[i, 1]
            Jz = J_hat[i, 2]
            mass_flux_divergence[i] = -(_gradient_neumann(Jx, dx, 0) + 
                                       _gradient_neumann(Jy, dx, 1) + 
                                       _gradient_neumann(Jz, dx, 2))
        
        print(f"  Mass flux divergence (diffusion term):")
        print(f"    Max: {np.max(mass_flux_divergence):.6f}")
        print(f"    Mean: {np.mean(mass_flux_divergence):.6f}")
        print(f"    Std: {np.std(mass_flux_divergence):.6f}")
        
        # 5. Check for positive diffusion (expansion)
        positive_diffusion = np.sum(mass_flux_divergence > 0)
        negative_diffusion = np.sum(mass_flux_divergence < 0)
        total_diffusion = mass_flux_divergence.size
        
        print(f"  Diffusion pattern:")
        print(f"    Positive diffusion: {positive_diffusion}/{total_diffusion} ({100*positive_diffusion/total_diffusion:.1f}%)")
        print(f"    Negative diffusion: {negative_diffusion}/{total_diffusion} ({100*negative_diffusion/total_diffusion:.1f}%)")
        
        # 6. Check mobility values
        print(f"\nMobility analysis:")
        for i, label in enumerate(simulator.cell_dynamics.labels):
            mobility = simulator.cell_dynamics.mass_flux_computer.mobilities[i]
            print(f"    {label}: {mobility}")
            if mobility > 0 and np.max(np.abs(energy_deriv)) < 1e-10:
                print(f"      ⚠️  WARNING: Non-zero mobility with zero energy derivative!")
                print(f"      This will cause diffusion even without adhesion energy.")
        
        # 7. Check if there's any residual energy derivative from numerical errors
        if np.max(np.abs(energy_deriv)) > 1e-10:
            print(f"\nEnergy derivative analysis:")
            print(f"  Energy derivative is not exactly zero - this could cause mass flux")
            print(f"  Max absolute value: {np.max(np.abs(energy_deriv)):.2e}")
            print(f"  This might be due to numerical errors in the energy calculation")
        
        # 8. Check the actual mass flux values
        print(f"\nMass flux component analysis:")
        for i, label in enumerate(simulator.cell_dynamics.labels):
            Jx = J_hat[i, 0]
            Jy = J_hat[i, 1]
            Jz = J_hat[i, 2]
            J_mag = np.sqrt(Jx**2 + Jy**2 + Jz**2)
            print(f"  {label}:")
            print(f"    Max magnitude: {np.max(J_mag):.6f}")
            print(f"    Mean magnitude: {np.mean(J_mag):.6f}")
            print(f"    Non-zero points: {np.sum(J_mag > 1e-10)}/{J_mag.size}")
        
        return True
        
    except Exception as e:
        print(f"❌ Debug failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Diffusion Escape Debug")
    print("=" * 50)
    
    success = debug_diffusion_escape()
    
    if success:
        print("\n✅ Debug completed successfully!")
    else:
        print("\n❌ Debug failed.")
