#!/usr/bin/env python3
"""
Debug script to identify the real source of expansion.

This script will analyze each term in the cell dynamics equation to find
what's causing the expansion when adhesion energy is zero.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from src.growkit.Simulator import TumorGrowthSimulator

def debug_expansion_terms():
    """Debug each term in the cell dynamics equation."""
    
    print("Debugging expansion terms...")
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
        
        # 2. Velocity field
        ux, uy, uz = simulator.cell_dynamics.compute_solid_velocity(phi_hat, nutrient_field, dx)
        velocity_magnitude = np.sqrt(ux**2 + uy**2 + uz**2)
        print(f"  Velocity field:")
        print(f"    Max magnitude: {np.max(velocity_magnitude):.6f}")
        print(f"    Mean magnitude: {np.mean(velocity_magnitude):.6f}")
        print(f"    Std magnitude: {np.std(velocity_magnitude):.6f}")
        
        # 3. Mass flux
        J_hat = simulator.cell_dynamics.compute_mass_fluxes(phi_hat, dx, energy_deriv)
        mass_flux_magnitude = np.sqrt(J_hat[:, 0]**2 + J_hat[:, 1]**2 + J_hat[:, 2]**2)
        print(f"  Mass flux:")
        print(f"    Max magnitude: {np.max(mass_flux_magnitude):.6f}")
        print(f"    Mean magnitude: {np.mean(mass_flux_magnitude):.6f}")
        print(f"    Std magnitude: {np.std(mass_flux_magnitude):.6f}")
        
        # 4. Source terms
        source_terms = simulator.cell_dynamics.compute_source_terms(phi_hat, nutrient_field)
        print(f"  Source terms:")
        print(f"    Max: {np.max(source_terms):.6f}")
        print(f"    Mean: {np.mean(source_terms):.6f}")
        print(f"    Std: {np.std(source_terms):.6f}")
        
        # 5. Compute total dynamics
        dynamics = simulator.cell_dynamics.compute_dynamics(phi_hat, nutrient_field, dx, source_terms)
        print(f"  Total dynamics:")
        print(f"    Max: {np.max(dynamics):.6f}")
        print(f"    Mean: {np.mean(dynamics):.6f}")
        print(f"    Std: {np.std(dynamics):.6f}")
        
        # 6. Analyze each term's contribution
        print(f"\nTerm-by-term analysis:")
        
        # Advection term: -∇·(u ⊗ φ_hat)
        from src.growkit.PhysicsEngine.VectorizedCellDynamics import _upwind_divergence
        advection_terms = np.zeros_like(phi_hat)
        for i in range(phi_hat.shape[0]):
            advection_terms[i] = _upwind_divergence(ux, uy, uz, phi_hat[i], dx)
        
        # Mass flux term: -∇·J_hat
        from src.growkit.MathEngine.Operators import _gradient_neumann
        mass_flux_terms = np.zeros_like(phi_hat)
        for i in range(phi_hat.shape[0]):
            Jx = J_hat[i, 0]
            Jy = J_hat[i, 1]
            Jz = J_hat[i, 2]
            mass_flux_terms[i] = -(_gradient_neumann(Jx, dx, 0) + 
                                  _gradient_neumann(Jy, dx, 1) + 
                                  _gradient_neumann(Jz, dx, 2))
        
        print(f"  Advection term contribution:")
        print(f"    Max: {np.max(advection_terms):.6f}")
        print(f"    Mean: {np.mean(advection_terms):.6f}")
        print(f"    Std: {np.std(advection_terms):.6f}")
        
        print(f"  Mass flux term contribution:")
        print(f"    Max: {np.max(mass_flux_terms):.6f}")
        print(f"    Mean: {np.mean(mass_flux_terms):.6f}")
        print(f"    Std: {np.std(mass_flux_terms):.6f}")
        
        print(f"  Source term contribution:")
        print(f"    Max: {np.max(source_terms):.6f}")
        print(f"    Mean: {np.mean(source_terms):.6f}")
        print(f"    Std: {np.std(source_terms):.6f}")
        
        # Check which term is dominant
        advection_max = np.max(np.abs(advection_terms))
        mass_flux_max = np.max(np.abs(mass_flux_terms))
        source_max = np.max(np.abs(source_terms))
        
        print(f"\nDominant term analysis:")
        print(f"  Advection max: {advection_max:.6f}")
        print(f"  Mass flux max: {mass_flux_max:.6f}")
        print(f"  Source max: {source_max:.6f}")
        
        if advection_max > mass_flux_max and advection_max > source_max:
            print("  → Advection term is dominant")
        elif mass_flux_max > advection_max and mass_flux_max > source_max:
            print("  → Mass flux term is dominant")
        elif source_max > advection_max and source_max > mass_flux_max:
            print("  → Source term is dominant")
        else:
            print("  → Multiple terms are comparable")
        
        # Check for expansion patterns
        print(f"\nExpansion pattern analysis:")
        
        # Check if source terms are positive (causing growth/expansion)
        positive_sources = np.sum(source_terms > 0)
        negative_sources = np.sum(source_terms < 0)
        total_sources = source_terms.size
        
        print(f"  Positive source terms: {positive_sources}/{total_sources} ({100*positive_sources/total_sources:.1f}%)")
        print(f"  Negative source terms: {negative_sources}/{total_sources} ({100*negative_sources/total_sources:.1f}%)")
        
        # Check if advection is causing expansion
        positive_advection = np.sum(advection_terms > 0)
        negative_advection = np.sum(advection_terms < 0)
        total_advection = advection_terms.size
        
        print(f"  Positive advection: {positive_advection}/{total_advection} ({100*positive_advection/total_advection:.1f}%)")
        print(f"  Negative advection: {negative_advection}/{total_advection} ({100*negative_advection/total_advection:.1f}%)")
        
        # Check if mass flux is causing expansion
        positive_mass_flux = np.sum(mass_flux_terms > 0)
        negative_mass_flux = np.sum(mass_flux_terms < 0)
        total_mass_flux = mass_flux_terms.size
        
        print(f"  Positive mass flux: {positive_mass_flux}/{total_mass_flux} ({100*positive_mass_flux/total_mass_flux:.1f}%)")
        print(f"  Negative mass flux: {negative_mass_flux}/{total_mass_flux} ({100*negative_mass_flux/total_mass_flux:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ Debug failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Expansion Source Debug")
    print("=" * 50)
    
    success = debug_expansion_terms()
    
    if success:
        print("\n✅ Debug completed successfully!")
    else:
        print("\n❌ Debug failed.")
