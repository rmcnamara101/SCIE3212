#!/usr/bin/env python3
"""
Adhesion Force Investigation Script

This script focuses specifically on investigating adhesion-driven motion
by analyzing the adhesion energy and its gradients in detail.

Author: Riley Jae McNamara
Date: 2025-02-19
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import yaml

# Add project root to path
proj = Path(__file__).parent
sys.path.insert(0, str(proj))

from src.growkit.Simulator import TumorGrowthSimulator
from src.growkit.PhysicsEngine.Energy.VectorizedEnergy import VectorizedEnergy
from src.growkit.MathEngine.Operators import isotropic_gradient_components, isotropic_laplacian


class AdhesionInvestigator:
    """
    Focused investigator for adhesion forces and their effects.
    """
    
    def __init__(self, cfg_path):
        """
        Initialize the adhesion investigator.
        
        Args:
            cfg_path: Path to configuration YAML file
        """
        self.cfg = yaml.safe_load(Path(cfg_path).read_text())
        self.simulator = TumorGrowthSimulator(cfg_path)
        
        # Initialize energy computer for detailed analysis
        self.energy_computer = VectorizedEnergy(self.cfg, self.cfg["populations"])
        
        # Storage for analysis results
        self.analysis_data = []
        
    def investigate_adhesion(self, phi_hat, dx, step=0):
        """
        Perform detailed adhesion force investigation.
        
        Args:
            phi_hat: Stacked cell fraction fields (M, nx, ny, nz)
            dx: Grid spacing
            step: Current simulation step
            
        Returns:
            analysis: Dictionary containing adhesion analysis results
        """
        print(f"\n=== Adhesion Investigation for Step {step} ===")
        
        # Compute total cell density
        phi_T = np.sum(phi_hat, axis=0)
        
        # 1. DETAILED ENERGY ANALYSIS
        print("1. Detailed adhesion energy analysis...")
        
        # Compute Laplacian for curvature analysis
        laplace_phi = isotropic_laplacian(phi_T, dx)
        
        # Compute double-well potential derivative
        f_prime = 0.5 * phi_T * (1 - phi_T) * (2 * phi_T - 1)
        
        # Get adhesion parameter
        m = self.cfg["physics"]["adhesion_energy"]["m"]
        
        # Compute energy derivative components
        energy_deriv = m * (f_prime - 0.01 * laplace_phi)
        
        # Energy statistics
        energy_stats = {
            'mean': np.mean(energy_deriv),
            'std': np.std(energy_deriv),
            'min': np.min(energy_deriv),
            'max': np.max(energy_deriv),
            'magnitude': np.sqrt(np.mean(energy_deriv**2))
        }
        
        # Double-well potential statistics
        f_prime_stats = {
            'mean': np.mean(f_prime),
            'std': np.std(f_prime),
            'min': np.min(f_prime),
            'max': np.max(f_prime),
            'magnitude': np.sqrt(np.mean(f_prime**2))
        }
        
        # Curvature term statistics
        curvature_term = -0.01 * laplace_phi
        curvature_stats = {
            'mean': np.mean(curvature_term),
            'std': np.std(curvature_term),
            'min': np.min(curvature_term),
            'max': np.max(curvature_term),
            'magnitude': np.sqrt(np.mean(curvature_term**2))
        }
        
        print(f"   Adhesion parameter (m): {m}")
        print(f"   Energy derivative - Mean: {energy_stats['mean']:.6f}, Std: {energy_stats['std']:.6f}")
        print(f"   Double-well term - Mean: {f_prime_stats['mean']:.6f}, Std: {f_prime_stats['std']:.6f}")
        print(f"   Curvature term - Mean: {curvature_stats['mean']:.6f}, Std: {curvature_stats['std']:.6f}")
        
        # 2. ENERGY GRADIENT ANALYSIS (ADHESION FORCE)
        print("2. Adhesion force analysis...")
        
        # Compute gradients of energy derivative
        grad_energy_x, grad_energy_y, grad_energy_z = isotropic_gradient_components(energy_deriv, dx)
        
        # Adhesion force magnitude
        adhesion_force_magnitude = np.sqrt(grad_energy_x**2 + grad_energy_y**2 + grad_energy_z**2)
        
        adhesion_force_stats = {
            'mean': np.mean(adhesion_force_magnitude),
            'std': np.std(adhesion_force_magnitude),
            'min': np.min(adhesion_force_magnitude),
            'max': np.max(adhesion_force_magnitude),
            'magnitude': np.sqrt(np.mean(adhesion_force_magnitude**2))
        }
        
        print(f"   Adhesion force magnitude - Mean: {adhesion_force_stats['mean']:.6f}")
        print(f"   Adhesion force magnitude - Max: {adhesion_force_stats['max']:.6f}")
        print(f"   Adhesion force magnitude - Std: {adhesion_force_stats['std']:.6f}")
        
        # 3. CELL DENSITY GRADIENT ANALYSIS
        print("3. Cell density gradient analysis...")
        
        # Compute gradients of total cell density
        grad_phi_x, grad_phi_y, grad_phi_z = isotropic_gradient_components(phi_T, dx)
        
        # Cell density gradient magnitude
        density_gradient_magnitude = np.sqrt(grad_phi_x**2 + grad_phi_y**2 + grad_phi_z**2)
        
        density_gradient_stats = {
            'mean': np.mean(density_gradient_magnitude),
            'std': np.std(density_gradient_magnitude),
            'min': np.min(density_gradient_magnitude),
            'max': np.max(density_gradient_magnitude),
            'magnitude': np.sqrt(np.mean(density_gradient_magnitude**2))
        }
        
        print(f"   Density gradient magnitude - Mean: {density_gradient_stats['mean']:.6f}")
        print(f"   Density gradient magnitude - Max: {density_gradient_stats['max']:.6f}")
        
        # 4. ADHESION VELOCITY CONTRIBUTION
        print("4. Adhesion velocity contribution...")
        
        # Compute adhesion-only velocity: u_adhesion = -energy_deriv * grad_phi_T
        u_adhesion_x = -energy_deriv * grad_phi_x
        u_adhesion_y = -energy_deriv * grad_phi_y
        u_adhesion_z = -energy_deriv * grad_phi_z
        
        # Adhesion velocity magnitude
        adhesion_velocity_magnitude = np.sqrt(u_adhesion_x**2 + u_adhesion_y**2 + u_adhesion_z**2)
        
        adhesion_velocity_stats = {
            'mean': np.mean(adhesion_velocity_magnitude),
            'std': np.std(adhesion_velocity_magnitude),
            'min': np.min(adhesion_velocity_magnitude),
            'max': np.max(adhesion_velocity_magnitude),
            'magnitude': np.sqrt(np.mean(adhesion_velocity_magnitude**2))
        }
        
        print(f"   Adhesion velocity magnitude - Mean: {adhesion_velocity_stats['mean']:.6f}")
        print(f"   Adhesion velocity magnitude - Max: {adhesion_velocity_stats['max']:.6f}")
        
        # 5. COMPARISON WITH TOTAL VELOCITY
        print("5. Comparison with total velocity...")
        
        # Get total velocity from simulator
        ux, uy, uz = self.simulator.cell_dynamics.velocity_computer.compute_solid_velocity(
            phi_hat, self.simulator.field_manager.nutrient_field, dx, energy_deriv
        )
        
        total_velocity_magnitude = np.sqrt(ux**2 + uy**2 + uz**2)
        
        total_velocity_stats = {
            'mean': np.mean(total_velocity_magnitude),
            'std': np.std(total_velocity_magnitude),
            'min': np.min(total_velocity_magnitude),
            'max': np.max(total_velocity_magnitude),
            'magnitude': np.sqrt(np.mean(total_velocity_magnitude**2))
        }
        
        print(f"   Total velocity magnitude - Mean: {total_velocity_stats['mean']:.6f}")
        print(f"   Total velocity magnitude - Max: {total_velocity_stats['max']:.6f}")
        
        # Calculate adhesion contribution to total velocity
        adhesion_contribution = adhesion_velocity_stats['magnitude'] / total_velocity_stats['magnitude'] * 100 if total_velocity_stats['magnitude'] > 0 else 0
        
        print(f"   Adhesion contribution to total velocity: {adhesion_contribution:.2f}%")
        
        # 6. SPATIAL ANALYSIS
        print("6. Spatial analysis...")
        
        # Find regions with high adhesion forces
        high_adhesion_threshold = np.percentile(adhesion_force_magnitude, 95)
        high_adhesion_regions = adhesion_force_magnitude > high_adhesion_threshold
        
        # Find regions with high cell density gradients
        high_gradient_threshold = np.percentile(density_gradient_magnitude, 95)
        high_gradient_regions = density_gradient_magnitude > high_gradient_threshold
        
        # Find regions with high energy derivatives
        high_energy_threshold = np.percentile(np.abs(energy_deriv), 95)
        high_energy_regions = np.abs(energy_deriv) > high_energy_threshold
        
        spatial_stats = {
            'high_adhesion_fraction': np.sum(high_adhesion_regions) / high_adhesion_regions.size,
            'high_gradient_fraction': np.sum(high_gradient_regions) / high_gradient_regions.size,
            'high_energy_fraction': np.sum(high_energy_regions) / high_energy_regions.size,
            'adhesion_threshold': high_adhesion_threshold,
            'gradient_threshold': high_gradient_threshold,
            'energy_threshold': high_energy_threshold
        }
        
        print(f"   High adhesion regions: {spatial_stats['high_adhesion_fraction']*100:.2f}% of domain")
        print(f"   High gradient regions: {spatial_stats['high_gradient_fraction']*100:.2f}% of domain")
        print(f"   High energy regions: {spatial_stats['high_energy_fraction']*100:.2f}% of domain")
        
        # Compile analysis results
        analysis = {
            'step': step,
            'time': self.simulator.time,
            'm': m,
            'energy_stats': energy_stats,
            'f_prime_stats': f_prime_stats,
            'curvature_stats': curvature_stats,
            'adhesion_force_stats': adhesion_force_stats,
            'density_gradient_stats': density_gradient_stats,
            'adhesion_velocity_stats': adhesion_velocity_stats,
            'total_velocity_stats': total_velocity_stats,
            'adhesion_contribution': adhesion_contribution,
            'spatial_stats': spatial_stats,
            'phi_T_stats': {
                'mean': np.mean(phi_T),
                'std': np.std(phi_T),
                'min': np.min(phi_T),
                'max': np.max(phi_T)
            }
        }
        
        self.analysis_data.append(analysis)
        return analysis
    
    def run_adhesion_investigation(self, total_steps=10, analysis_interval=1):
        """
        Run simulation with detailed adhesion investigation.
        
        Args:
            total_steps: Total number of simulation steps
            analysis_interval: How often to perform detailed analysis
        """
        print("Starting adhesion force investigation...")
        print(f"Total steps: {total_steps}")
        print(f"Analysis interval: {analysis_interval}")
        print(f"Grid size: {self.simulator.field_manager.grid}")
        print(f"Adhesion energy parameter (m): {self.cfg['physics']['adhesion_energy']['m']}")
        print(f"Pressure disabled: {self.cfg['physics'].get('disable_pressure', False)}")
        
        # Initialize fields
        self.simulator.initialize_fields()
        
        # Initial investigation
        phi_hat, _, _ = self.simulator.field_manager.get_cell_fields()
        self.investigate_adhesion(phi_hat, self.simulator.field_manager.dx, step=0)
        
        # Run simulation with investigation
        for step in range(1, total_steps + 1):
            # Perform simulation step
            success = self.simulator.step()
            
            if not success:
                print(f"Warning: Step {step} failed")
                continue
            
            # Perform investigation at specified intervals
            if step % analysis_interval == 0:
                phi_hat, _, _ = self.simulator.field_manager.get_cell_fields()
                self.investigate_adhesion(phi_hat, self.simulator.field_manager.dx, step=step)
        
        print(f"\nAdhesion investigation completed!")
        print(f"Final time: {self.simulator.time:.3f}")
        print(f"Total investigation points: {len(self.analysis_data)}")
        
        return self.analysis_data
    
    def plot_adhesion_analysis(self, save_path=None):
        """
        Plot adhesion analysis results.
        
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
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Adhesion Force Investigation', fontsize=16)
        
        # Plot 1: Energy derivative evolution
        ax1 = axes[0, 0]
        energy_means = [data['energy_stats']['mean'] for data in self.analysis_data]
        energy_stds = [data['energy_stats']['std'] for data in self.analysis_data]
        ax1.plot(times, energy_means, 'b-', label='Mean', marker='o', markersize=4)
        ax1.fill_between(times, 
                        np.array(energy_means) - np.array(energy_stds),
                        np.array(energy_means) + np.array(energy_stds),
                        alpha=0.3, color='blue')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Energy Derivative')
        ax1.set_title('Energy Derivative Evolution')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Adhesion force magnitude
        ax2 = axes[0, 1]
        adhesion_force_means = [data['adhesion_force_stats']['mean'] for data in self.analysis_data]
        adhesion_force_maxs = [data['adhesion_force_stats']['max'] for data in self.analysis_data]
        ax2.plot(times, adhesion_force_means, 'r-', label='Mean', marker='o', markersize=4)
        ax2.plot(times, adhesion_force_maxs, 'r--', label='Max', marker='s', markersize=4)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Adhesion Force Magnitude')
        ax2.set_title('Adhesion Force Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Plot 3: Velocity comparison
        ax3 = axes[0, 2]
        adhesion_vel_means = [data['adhesion_velocity_stats']['mean'] for data in self.analysis_data]
        total_vel_means = [data['total_velocity_stats']['mean'] for data in self.analysis_data]
        ax3.plot(times, adhesion_vel_means, 'g-', label='Adhesion Velocity', marker='o', markersize=4)
        ax3.plot(times, total_vel_means, 'b-', label='Total Velocity', marker='s', markersize=4)
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Velocity Magnitude')
        ax3.set_title('Velocity Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        # Plot 4: Adhesion contribution
        ax4 = axes[1, 0]
        adhesion_contributions = [data['adhesion_contribution'] for data in self.analysis_data]
        ax4.plot(times, adhesion_contributions, 'purple', marker='o', markersize=4)
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Adhesion Contribution (%)')
        ax4.set_title('Adhesion Contribution to Total Velocity')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Energy components
        ax5 = axes[1, 1]
        f_prime_means = [data['f_prime_stats']['mean'] for data in self.analysis_data]
        curvature_means = [data['curvature_stats']['mean'] for data in self.analysis_data]
        ax5.plot(times, f_prime_means, 'orange', label='Double-well', marker='o', markersize=4)
        ax5.plot(times, curvature_means, 'red', label='Curvature', marker='s', markersize=4)
        ax5.set_xlabel('Time')
        ax5.set_ylabel('Energy Component')
        ax5.set_title('Energy Components')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Spatial analysis
        ax6 = axes[1, 2]
        high_adhesion_fractions = [data['spatial_stats']['high_adhesion_fraction']*100 for data in self.analysis_data]
        high_gradient_fractions = [data['spatial_stats']['high_gradient_fraction']*100 for data in self.analysis_data]
        ax6.plot(times, high_adhesion_fractions, 'purple', label='High Adhesion', marker='o', markersize=4)
        ax6.plot(times, high_gradient_fractions, 'brown', label='High Gradient', marker='s', markersize=4)
        ax6.set_xlabel('Time')
        ax6.set_ylabel('Fraction of Domain (%)')
        ax6.set_title('Spatial Distribution')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Adhesion analysis plot saved to {save_path}")
        
        plt.show()
    
    def print_adhesion_summary(self):
        """
        Print a comprehensive summary of the adhesion investigation.
        """
        if not self.analysis_data:
            print("No analysis data available")
            return
        
        print("\n" + "="*80)
        print("ADHESION FORCE INVESTIGATION SUMMARY")
        print("="*80)
        
        # Configuration summary
        print(f"\nConfiguration:")
        print(f"  Adhesion energy parameter (m): {self.cfg['physics']['adhesion_energy']['m']}")
        print(f"  Pressure disabled: {self.cfg['physics'].get('disable_pressure', False)}")
        print(f"  Grid size: {self.simulator.field_manager.grid}")
        print(f"  Grid spacing (dx): {self.simulator.field_manager.dx}")
        
        # Final state analysis
        final_data = self.analysis_data[-1]
        print(f"\nFinal state (Step {final_data['step']}, Time {final_data['time']:.3f}):")
        
        print(f"\nAdhesion energy analysis:")
        print(f"  Energy derivative - Mean: {final_data['energy_stats']['mean']:.6f}")
        print(f"  Energy derivative - Std: {final_data['energy_stats']['std']:.6f}")
        print(f"  Double-well term - Mean: {final_data['f_prime_stats']['mean']:.6f}")
        print(f"  Curvature term - Mean: {final_data['curvature_stats']['mean']:.6f}")
        
        print(f"\nAdhesion force analysis:")
        print(f"  Adhesion force magnitude - Mean: {final_data['adhesion_force_stats']['mean']:.6f}")
        print(f"  Adhesion force magnitude - Max: {final_data['adhesion_force_stats']['max']:.6f}")
        print(f"  Density gradient magnitude - Mean: {final_data['density_gradient_stats']['mean']:.6f}")
        
        print(f"\nVelocity analysis:")
        print(f"  Adhesion velocity magnitude - Mean: {final_data['adhesion_velocity_stats']['mean']:.6f}")
        print(f"  Total velocity magnitude - Mean: {final_data['total_velocity_stats']['mean']:.6f}")
        print(f"  Adhesion contribution: {final_data['adhesion_contribution']:.2f}%")
        
        print(f"\nSpatial analysis:")
        print(f"  High adhesion regions: {final_data['spatial_stats']['high_adhesion_fraction']*100:.2f}%")
        print(f"  High gradient regions: {final_data['spatial_stats']['high_gradient_fraction']*100:.2f}%")
        print(f"  High energy regions: {final_data['spatial_stats']['high_energy_fraction']*100:.2f}%")
        
        # Evolution analysis
        print(f"\nEvolution analysis:")
        initial_contribution = self.analysis_data[0]['adhesion_contribution']
        final_contribution = self.analysis_data[-1]['adhesion_contribution']
        print(f"  Initial adhesion contribution: {initial_contribution:.2f}%")
        print(f"  Final adhesion contribution: {final_contribution:.2f}%")
        print(f"  Change in contribution: {final_contribution - initial_contribution:.2f}%")
        
        # Check if adhesion is significant
        avg_contribution = np.mean([data['adhesion_contribution'] for data in self.analysis_data])
        print(f"  Average adhesion contribution: {avg_contribution:.2f}%")
        
        if avg_contribution < 1.0:
            print(f"\n⚠️  WARNING: Adhesion contribution is very low ({avg_contribution:.2f}%)")
            print(f"   This suggests adhesion-driven motion is not significant.")
            print(f"   Possible causes:")
            print(f"   - Adhesion parameter (m={self.cfg['physics']['adhesion_energy']['m']}) too low")
            print(f"   - Pressure forces dominating over adhesion forces")
            print(f"   - Cell density gradients too small")
            print(f"   - Energy derivative gradients too small")
        elif avg_contribution < 10.0:
            print(f"\n⚠️  CAUTION: Adhesion contribution is low ({avg_contribution:.2f}%)")
            print(f"   Adhesion may be present but not dominant.")
        else:
            print(f"\n✅ Adhesion contribution is significant ({avg_contribution:.2f}%)")
        
        print("\n" + "="*80)


def main():
    """
    Main function to run the adhesion investigation.
    """
    # Configuration
    cfg_path = "configs/csc-t-n.yaml"
    total_steps = 20
    analysis_interval = 2
    
    print("Adhesion Force Investigation Script")
    print("=" * 50)
    
    # Create investigator
    investigator = AdhesionInvestigator(cfg_path)
    
    # Run investigation
    analysis_data = investigator.run_adhesion_investigation(
        total_steps=total_steps,
        analysis_interval=analysis_interval
    )
    
    # Print summary
    investigator.print_adhesion_summary()
    
    # Create plots
    output_dir = Path("adhesion_investigation_output")
    output_dir.mkdir(exist_ok=True)
    
    plot_path = output_dir / "adhesion_analysis.png"
    investigator.plot_adhesion_analysis(save_path=plot_path)
    
    # Save analysis data
    import pickle
    data_path = output_dir / "adhesion_analysis_data.pkl"
    with open(data_path, 'wb') as f:
        pickle.dump(analysis_data, f)
    
    print(f"\nInvestigation complete!")
    print(f"Results saved to: {output_dir}")
    print(f"  - Adhesion analysis plot: {plot_path}")
    print(f"  - Analysis data: {data_path}")


if __name__ == "__main__":
    main()
