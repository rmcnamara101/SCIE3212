#!/usr/bin/env python3
"""
Force Analysis Script for Tumor Growth Simulation

This script runs a simulation and provides a detailed breakdown of all forces
and their relative impacts on motion. It helps investigate adhesion-driven motion
and other force contributions.

Author: Riley Jae McNamara
Date: 2025-02-19
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import yaml
from collections import defaultdict
import time

# Add project root to path
proj = Path(__file__).parent
sys.path.insert(0, str(proj))

from src.growkit.Simulator import TumorGrowthSimulator
from src.growkit.PhysicsEngine.VectorizedCellDynamics import VectorizedCellDynamics
from src.growkit.PhysicsEngine.Energy.VectorizedEnergy import VectorizedEnergy
from src.growkit.PhysicsEngine.MassFlux.VectorizedMassFlux import VectorizedMassFlux
from src.growkit.PhysicsEngine.SolidVelocity.VectorizedSolidVelocity import VectorizedSolidVelocity
from src.growkit.MathEngine.Operators import isotropic_gradient_components, isotropic_laplacian


class ForceAnalyzer:
    """
    Comprehensive force analyzer for tumor growth simulation.
    """
    
    def __init__(self, cfg_path):
        """
        Initialize the force analyzer.
        
        Args:
            cfg_path: Path to configuration YAML file
        """
        self.cfg = yaml.safe_load(Path(cfg_path).read_text())
        self.simulator = TumorGrowthSimulator(cfg_path)
        
        # Initialize physics components for detailed analysis
        self.energy_computer = VectorizedEnergy(self.cfg, self.cfg["populations"])
        self.mass_flux_computer = VectorizedMassFlux(self.cfg, self.cfg["populations"])
        self.velocity_computer = VectorizedSolidVelocity(self.cfg, self.cfg["populations"])
        
        # Storage for analysis results
        self.analysis_data = []
        
    def analyze_forces(self, phi_hat, nutrient_field, dx, step=0):
        """
        Perform comprehensive force analysis for current state.
        
        Args:
            phi_hat: Stacked cell fraction fields (M, nx, ny, nz)
            nutrient_field: Nutrient concentration field
            dx: Grid spacing
            step: Current simulation step
            
        Returns:
            analysis: Dictionary containing force analysis results
        """
        print(f"\n=== Force Analysis for Step {step} ===")
        
        # Compute total cell density
        phi_T = np.sum(phi_hat, axis=0)
        
        # 1. ENERGY ANALYSIS
        print("1. Computing adhesion energy...")
        energy_deriv = self.energy_computer.compute_energy_derivative(phi_T, dx)
        
        # Energy statistics
        energy_stats = {
            'mean': np.mean(energy_deriv),
            'std': np.std(energy_deriv),
            'min': np.min(energy_deriv),
            'max': np.max(energy_deriv),
            'magnitude': np.sqrt(np.mean(energy_deriv**2))
        }
        
        # Energy gradient (force from adhesion)
        grad_energy_x, grad_energy_y, grad_energy_z = isotropic_gradient_components(energy_deriv, dx)
        energy_force_magnitude = np.sqrt(grad_energy_x**2 + grad_energy_y**2 + grad_energy_z**2)
        
        energy_force_stats = {
            'mean': np.mean(energy_force_magnitude),
            'std': np.std(energy_force_magnitude),
            'max': np.max(energy_force_magnitude),
            'magnitude': np.sqrt(np.mean(energy_force_magnitude**2))
        }
        
        print(f"   Energy derivative - Mean: {energy_stats['mean']:.6f}, Std: {energy_stats['std']:.6f}")
        print(f"   Energy force magnitude - Mean: {energy_force_stats['mean']:.6f}, Max: {energy_force_stats['max']:.6f}")
        
        # 2. MASS FLUX ANALYSIS
        print("2. Computing mass fluxes...")
        J_hat = self.mass_flux_computer.compute_mass_fluxes(phi_hat, dx, energy_deriv)
        
        # Mass flux statistics for each population
        mass_flux_stats = {}
        for i, pop_name in enumerate(self.cfg["populations"].keys()):
            Jx, Jy, Jz = J_hat[i, 0], J_hat[i, 1], J_hat[i, 2]
            flux_magnitude = np.sqrt(Jx**2 + Jy**2 + Jz**2)
            
            mass_flux_stats[pop_name] = {
                'mean': np.mean(flux_magnitude),
                'std': np.std(flux_magnitude),
                'max': np.max(flux_magnitude),
                'magnitude': np.sqrt(np.mean(flux_magnitude**2)),
                'total_flux': np.sum(flux_magnitude)
            }
            
            print(f"   {pop_name} mass flux - Mean: {mass_flux_stats[pop_name]['mean']:.6f}, Max: {mass_flux_stats[pop_name]['max']:.6f}")
        
        # 3. SOLID VELOCITY ANALYSIS
        print("3. Computing solid velocity...")
        ux, uy, uz = self.velocity_computer.compute_solid_velocity(phi_hat, nutrient_field, dx, energy_deriv)
        
        # Velocity statistics
        velocity_magnitude = np.sqrt(ux**2 + uy**2 + uz**2)
        velocity_stats = {
            'mean': np.mean(velocity_magnitude),
            'std': np.std(velocity_magnitude),
            'max': np.max(velocity_magnitude),
            'magnitude': np.sqrt(np.mean(velocity_magnitude**2))
        }
        
        print(f"   Velocity magnitude - Mean: {velocity_stats['mean']:.6f}, Max: {velocity_stats['max']:.6f}")
        
        # 4. PRESSURE ANALYSIS (if enabled)
        pressure_stats = None
        if hasattr(self.simulator.field_manager, 'pressure') and self.simulator.field_manager.pressure is not None:
            pressure = self.simulator.field_manager.pressure
            pressure_stats = {
                'mean': np.mean(pressure),
                'std': np.std(pressure),
                'min': np.min(pressure),
                'max': np.max(pressure),
                'magnitude': np.sqrt(np.mean(pressure**2))
            }
            
            # Pressure gradient
            grad_p_x, grad_p_y, grad_p_z = isotropic_gradient_components(pressure, dx)
            pressure_force_magnitude = np.sqrt(grad_p_x**2 + grad_p_y**2 + grad_p_z**2)
            
            pressure_force_stats = {
                'mean': np.mean(pressure_force_magnitude),
                'std': np.std(pressure_force_magnitude),
                'max': np.max(pressure_force_magnitude),
                'magnitude': np.sqrt(np.mean(pressure_force_magnitude**2))
            }
            
            print(f"   Pressure - Mean: {pressure_stats['mean']:.6f}, Std: {pressure_stats['std']:.6f}")
            print(f"   Pressure force magnitude - Mean: {pressure_force_stats['mean']:.6f}, Max: {pressure_force_stats['max']:.6f}")
        
        # 5. SOURCE TERMS ANALYSIS
        print("4. Computing source terms...")
        source_terms = self.simulator.cell_dynamics.compute_source_terms(phi_hat, nutrient_field)
        
        source_stats = {}
        for i, pop_name in enumerate(self.cfg["populations"].keys()):
            source = source_terms[i]
            source_stats[pop_name] = {
                'mean': np.mean(source),
                'std': np.std(source),
                'min': np.min(source),
                'max': np.max(source),
                'magnitude': np.sqrt(np.mean(source**2)),
                'total_source': np.sum(source)
            }
            
            print(f"   {pop_name} source - Mean: {source_stats[pop_name]['mean']:.6f}, Total: {source_stats[pop_name]['total_source']:.6f}")
        
        # 6. DYNAMICS ANALYSIS
        print("5. Computing total dynamics...")
        dphi_hat = self.simulator.cell_dynamics.compute_dynamics(phi_hat, nutrient_field, dx, source_terms)
        
        dynamics_stats = {}
        for i, pop_name in enumerate(self.cfg["populations"].keys()):
            dphi = dphi_hat[i]
            dynamics_stats[pop_name] = {
                'mean': np.mean(dphi),
                'std': np.std(dphi),
                'min': np.min(dphi),
                'max': np.max(dphi),
                'magnitude': np.sqrt(np.mean(dphi**2)),
                'total_change': np.sum(dphi)
            }
            
            print(f"   {pop_name} dynamics - Mean: {dynamics_stats[pop_name]['mean']:.6f}, Total change: {dynamics_stats[pop_name]['total_change']:.6f}")
        
        # 7. FORCE COMPARISON AND RELATIVE IMPACT
        print("6. Force comparison and relative impact...")
        
        # Calculate relative magnitudes
        force_magnitudes = {
            'energy_force': energy_force_stats['magnitude'],
            'velocity': velocity_stats['magnitude']
        }
        
        if pressure_stats is not None:
            force_magnitudes['pressure_force'] = pressure_force_stats['magnitude']
        
        # Add mass flux magnitudes
        for pop_name, stats in mass_flux_stats.items():
            force_magnitudes[f'{pop_name}_mass_flux'] = stats['magnitude']
        
        # Find dominant forces
        total_force_magnitude = sum(force_magnitudes.values())
        force_contributions = {name: mag/total_force_magnitude*100 if total_force_magnitude > 0 else 0 
                              for name, mag in force_magnitudes.items()}
        
        print("   Force contributions (%):")
        for name, contribution in sorted(force_contributions.items(), key=lambda x: x[1], reverse=True):
            print(f"     {name}: {contribution:.2f}%")
        
        # Compile analysis results
        analysis = {
            'step': step,
            'time': self.simulator.time,
            'energy_stats': energy_stats,
            'energy_force_stats': energy_force_stats,
            'mass_flux_stats': mass_flux_stats,
            'velocity_stats': velocity_stats,
            'pressure_stats': pressure_stats,
            'pressure_force_stats': pressure_force_stats if pressure_stats is not None else None,
            'source_stats': source_stats,
            'dynamics_stats': dynamics_stats,
            'force_magnitudes': force_magnitudes,
            'force_contributions': force_contributions,
            'total_force_magnitude': total_force_magnitude
        }
        
        self.analysis_data.append(analysis)
        return analysis
    
    def run_analysis_simulation(self, total_steps=10, analysis_interval=1):
        """
        Run simulation with force analysis at specified intervals.
        
        Args:
            total_steps: Total number of simulation steps
            analysis_interval: How often to perform detailed analysis
        """
        print("Starting force analysis simulation...")
        print(f"Total steps: {total_steps}")
        print(f"Analysis interval: {analysis_interval}")
        print(f"Grid size: {self.simulator.field_manager.grid}")
        print(f"Number of populations: {self.simulator.field_manager.M}")
        print(f"Adhesion energy parameter (m): {self.cfg['physics']['adhesion_energy']['m']}")
        print(f"Pressure disabled: {self.cfg['physics'].get('disable_pressure', False)}")
        
        # Initialize fields
        self.simulator.initialize_fields()
        
        # Initial analysis
        phi_hat, nutrient_field, _ = self.simulator.field_manager.get_cell_fields()
        self.analyze_forces(phi_hat, nutrient_field, self.simulator.field_manager.dx, step=0)
        
        # Run simulation with analysis
        for step in range(1, total_steps + 1):
            # Perform simulation step
            success = self.simulator.step()
            
            if not success:
                print(f"Warning: Step {step} failed")
                continue
            
            # Perform analysis at specified intervals
            if step % analysis_interval == 0:
                phi_hat, nutrient_field, _ = self.simulator.field_manager.get_cell_fields()
                self.analyze_forces(phi_hat, nutrient_field, self.simulator.field_manager.dx, step=step)
        
        print(f"\nForce analysis simulation completed!")
        print(f"Final time: {self.simulator.time:.3f}")
        print(f"Total analysis points: {len(self.analysis_data)}")
        
        return self.analysis_data
    
    def plot_force_evolution(self, save_path=None):
        """
        Plot the evolution of forces over time.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.analysis_data:
            print("No analysis data available for plotting")
            return
        
        # Extract data for plotting
        steps = [data['step'] for data in self.analysis_data]
        times = [data['time'] for data in self.analysis_data]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Force Analysis Over Time', fontsize=16)
        
        # Plot 1: Force magnitudes
        ax1 = axes[0, 0]
        force_names = ['energy_force', 'velocity']
        if self.analysis_data[0]['pressure_force_stats'] is not None:
            force_names.append('pressure_force')
        
        for force_name in force_names:
            magnitudes = [data['force_magnitudes'][force_name] for data in self.analysis_data]
            ax1.plot(times, magnitudes, label=force_name, marker='o', markersize=4)
        
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Force Magnitude')
        ax1.set_title('Force Magnitudes Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot 2: Force contributions
        ax2 = axes[0, 1]
        for force_name in force_names:
            contributions = [data['force_contributions'][force_name] for data in self.analysis_data]
            ax2.plot(times, contributions, label=force_name, marker='o', markersize=4)
        
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Force Contribution (%)')
        ax2.set_title('Force Contributions Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Mass flux magnitudes
        ax3 = axes[1, 0]
        for pop_name in self.cfg["populations"].keys():
            flux_magnitudes = [data['mass_flux_stats'][pop_name]['magnitude'] for data in self.analysis_data]
            ax3.plot(times, flux_magnitudes, label=f'{pop_name} mass flux', marker='o', markersize=4)
        
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Mass Flux Magnitude')
        ax3.set_title('Mass Flux Magnitudes Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        # Plot 4: Source terms
        ax4 = axes[1, 1]
        for pop_name in self.cfg["populations"].keys():
            source_totals = [data['source_stats'][pop_name]['total_source'] for data in self.analysis_data]
            ax4.plot(times, source_totals, label=f'{pop_name} source', marker='o', markersize=4)
        
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Total Source Term')
        ax4.set_title('Source Terms Over Time')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Force evolution plot saved to {save_path}")
        
        plt.show()
    
    def print_final_summary(self):
        """
        Print a comprehensive summary of the force analysis.
        """
        if not self.analysis_data:
            print("No analysis data available")
            return
        
        print("\n" + "="*80)
        print("COMPREHENSIVE FORCE ANALYSIS SUMMARY")
        print("="*80)
        
        # Configuration summary
        print(f"\nConfiguration:")
        print(f"  Adhesion energy parameter (m): {self.cfg['physics']['adhesion_energy']['m']}")
        print(f"  Pressure disabled: {self.cfg['physics'].get('disable_pressure', False)}")
        print(f"  Grid size: {self.simulator.field_manager.grid}")
        print(f"  Grid spacing (dx): {self.simulator.field_manager.dx}")
        
        # Population mobilities
        print(f"\nPopulation mobilities:")
        for pop_name, pop_config in self.cfg["populations"].items():
            mobility = pop_config["dynamics"].get("mobility", 0.01)
            print(f"  {pop_name}: {mobility}")
        
        # Final state analysis
        final_data = self.analysis_data[-1]
        print(f"\nFinal state (Step {final_data['step']}, Time {final_data['time']:.3f}):")
        
        print(f"\nForce magnitudes:")
        for force_name, magnitude in final_data['force_magnitudes'].items():
            print(f"  {force_name}: {magnitude:.6f}")
        
        print(f"\nForce contributions (%):")
        for force_name, contribution in sorted(final_data['force_contributions'].items(), 
                                             key=lambda x: x[1], reverse=True):
            print(f"  {force_name}: {contribution:.2f}%")
        
        # Energy analysis
        print(f"\nAdhesion energy analysis:")
        print(f"  Energy derivative - Mean: {final_data['energy_stats']['mean']:.6f}")
        print(f"  Energy derivative - Std: {final_data['energy_stats']['std']:.6f}")
        print(f"  Energy force magnitude - Mean: {final_data['energy_force_stats']['mean']:.6f}")
        print(f"  Energy force magnitude - Max: {final_data['energy_force_stats']['max']:.6f}")
        
        # Velocity analysis
        print(f"\nVelocity analysis:")
        print(f"  Velocity magnitude - Mean: {final_data['velocity_stats']['mean']:.6f}")
        print(f"  Velocity magnitude - Max: {final_data['velocity_stats']['max']:.6f}")
        
        # Pressure analysis (if available)
        if final_data['pressure_stats'] is not None:
            print(f"\nPressure analysis:")
            print(f"  Pressure - Mean: {final_data['pressure_stats']['mean']:.6f}")
            print(f"  Pressure - Std: {final_data['pressure_stats']['std']:.6f}")
            print(f"  Pressure force magnitude - Mean: {final_data['pressure_force_stats']['mean']:.6f}")
            print(f"  Pressure force magnitude - Max: {final_data['pressure_force_stats']['max']:.6f}")
        
        # Mass flux analysis
        print(f"\nMass flux analysis:")
        for pop_name, stats in final_data['mass_flux_stats'].items():
            print(f"  {pop_name}:")
            print(f"    Magnitude: {stats['magnitude']:.6f}")
            print(f"    Max: {stats['max']:.6f}")
            print(f"    Total flux: {stats['total_flux']:.6f}")
        
        # Source terms analysis
        print(f"\nSource terms analysis:")
        for pop_name, stats in final_data['source_stats'].items():
            print(f"  {pop_name}:")
            print(f"    Mean: {stats['mean']:.6f}")
            print(f"    Total: {stats['total_source']:.6f}")
        
        # Dynamics analysis
        print(f"\nDynamics analysis:")
        for pop_name, stats in final_data['dynamics_stats'].items():
            print(f"  {pop_name}:")
            print(f"    Mean change: {stats['mean']:.6f}")
            print(f"    Total change: {stats['total_change']:.6f}")
        
        print("\n" + "="*80)


def main():
    """
    Main function to run the force analysis.
    """
    # Configuration
    cfg_path = "configs/csc-t-n.yaml"
    total_steps = 20
    analysis_interval = 2
    
    print("Force Analysis Script for Tumor Growth Simulation")
    print("=" * 60)
    
    # Create analyzer
    analyzer = ForceAnalyzer(cfg_path)
    
    # Run analysis simulation
    analysis_data = analyzer.run_analysis_simulation(
        total_steps=total_steps,
        analysis_interval=analysis_interval
    )
    
    # Print final summary
    analyzer.print_final_summary()
    
    # Create plots
    output_dir = Path("force_analysis_output")
    output_dir.mkdir(exist_ok=True)
    
    plot_path = output_dir / "force_evolution.png"
    analyzer.plot_force_evolution(save_path=plot_path)
    
    # Save analysis data
    import pickle
    data_path = output_dir / "force_analysis_data.pkl"
    with open(data_path, 'wb') as f:
        pickle.dump(analysis_data, f)
    
    print(f"\nAnalysis complete!")
    print(f"Results saved to: {output_dir}")
    print(f"  - Force evolution plot: {plot_path}")
    print(f"  - Analysis data: {data_path}")


if __name__ == "__main__":
    main()
