"""
Tumor Growth Simulator Module

This module integrates all components (FieldManager, PhysicsEngine, Integrator) to run
the complete tumor growth simulation. It orchestrates the time stepping, physics
computations, and data management.

Author: Riley Jae McNamara
Date: 2025-02-19
"""

import numpy as np
import time
from pathlib import Path
import yaml
from collections import defaultdict
import os
from tqdm import tqdm

# Pre-import all modules to avoid runtime import overhead
from src.growkit.Fields.FieldManager import FieldManager
from src.growkit.PhysicsEngine.VectorizedCellDynamics import VectorizedCellDynamics
from src.growkit.PhysicsEngine.Nutrient.NutrientField import NutrientField
from src.growkit.ProductionEngine.SourceConstructor import SourceConstructor
from src.growkit.Integrator.RK4Integrator import RK4Integrator, AdaptiveRK4Integrator
from src.growkit.Integrator.AdaptiveGridIntegrator import AdaptiveGridRK4Integrator
from src.growkit.Integrator.ForwardEulerIntegrator import (
    ForwardEulerIntegrator, 
    AdaptiveForwardEulerIntegrator, 
    ImprovedEulerIntegrator
)


class Profiler:
    """
    Performance profiler for tracking timing of individual modules.
    """
    
    def __init__(self):
        """Initialize the profiler."""
        self.timings = defaultdict(list)
        self.current_step_timings = {}
        self.enabled = True
    
    def start_timer(self, module_name: str):
        """Start timing a module."""
        if self.enabled:
            self.current_step_timings[module_name] = time.time()
    
    def end_timer(self, module_name: str):
        """End timing a module and record the duration."""
        if self.enabled and module_name in self.current_step_timings:
            duration = time.time() - self.current_step_timings[module_name]
            self.timings[module_name].append(duration)
            del self.current_step_timings[module_name]
    
    def get_step_summary(self):
        """Get timing summary for the current step."""
        if not self.enabled:
            return {}
        
        summary = {}
        for module, times in self.timings.items():
            if times:
                summary[module] = times[-1]  # Most recent timing
        return summary
    
    def get_average_timings(self):
        """Get average timings across all steps."""
        if not self.enabled:
            return {}
        
        averages = {}
        for module, times in self.timings.items():
            if times:
                averages[module] = np.mean(times)
        return averages
    
    def print_step_summary(self, step: int):
        """Print timing summary for the current step."""
        if not self.enabled:
            return
        
        summary = self.get_step_summary()
        if summary:
            print(f"Step {step} timing breakdown:")
            total_time = sum(summary.values())
            for module, duration in summary.items():
                percentage = (duration / total_time) * 100 if total_time > 0 else 0
                print(f"  {module}: {duration:.4f}s ({percentage:.1f}%)")
            print(f"  Total step time: {total_time:.4f}s")
    
    def print_final_summary(self):
        """Print final timing summary across all steps."""
        if not self.enabled:
            return
        
        averages = self.get_average_timings()
        if averages:
            print("\nFinal performance summary:")
            total_avg = sum(averages.values())
            for module, avg_time in sorted(averages.items(), key=lambda x: x[1], reverse=True):
                percentage = (avg_time / total_avg) * 100 if total_avg > 0 else 0
                print(f"  {module}: {avg_time:.4f}s avg ({percentage:.1f}%)")
            print(f"  Total average step time: {total_avg:.4f}s")


class TumorGrowthSimulator:
    """
    Main simulator class that orchestrates the tumor growth simulation.
    """
    
    def __init__(self, cfg_path: str):
        """
        Initialize the tumor growth simulator.
        
        Args:
            cfg_path: Path to configuration YAML file
            output_dir: Directory for output files
        """
        # Load configuration
        self.cfg = yaml.safe_load(Path(cfg_path).read_text())
        
        # Initialize profiler
        self.profiler = Profiler()
        
        # Initialize field manager
        self.field_manager = FieldManager(cfg_path)
        
        # Initialize source constructor for nutrient dynamics
        self.source_constructor = SourceConstructor(self.cfg, self.cfg["populations"])
        
        # Initialize physics components
        self.cell_dynamics = VectorizedCellDynamics(
            self.cfg, self.cfg["populations"], self.field_manager, self.source_constructor
        )
        
        # Initialize nutrient field manager
        self.nutrient_manager = NutrientField(self.cfg, self.cfg["populations"])
        
        # Initialize integrator
        dt = self.cfg["time"]["dt"]
        integrator_config = self.cfg.get("integrator", {})
        integrator_type = integrator_config.get("type", "rk4")
        
        if integrator_type == "adaptive_grid":
            # Use adaptive grid RK4 integrator
            padding = integrator_config.get("padding", 5)
            threshold = integrator_config.get("threshold", 1e-6)
            min_sub_grid_size = integrator_config.get("min_sub_grid_size", 10)
            self.integrator = AdaptiveGridRK4Integrator(dt, padding, threshold, min_sub_grid_size)
        elif integrator_type == "rk4_adaptive" or integrator_config.get("adaptive", False):
            # Use adaptive time step RK4 integrator
            tolerance = integrator_config.get("tolerance", 1e-6)
            min_dt = integrator_config.get("min_dt", 1e-8)
            max_dt = integrator_config.get("max_dt", 1e-2)
            self.integrator = AdaptiveRK4Integrator(dt, tolerance, min_dt, max_dt)
        elif integrator_type == "forward_euler":
            # Use basic Forward Euler integrator
            self.integrator = ForwardEulerIntegrator(dt)
        elif integrator_type == "forward_euler_adaptive":
            # Use adaptive Forward Euler integrator
            tolerance = integrator_config.get("tolerance", 0.1)
            min_dt = integrator_config.get("min_dt", 1e-8)
            max_dt = integrator_config.get("max_dt", 1e-2)
            safety_factor = integrator_config.get("safety_factor", 0.8)
            self.integrator = AdaptiveForwardEulerIntegrator(dt, tolerance, min_dt, max_dt, safety_factor)
        elif integrator_type == "improved_euler":
            # Use Improved Euler (Heun's method) integrator
            self.integrator = ImprovedEulerIntegrator(dt)
        elif integrator_type == "rk4":
            # Use standard RK4 integrator (default)
            self.integrator = RK4Integrator(dt)
        else:
            # Default to RK4 if unknown type
            print(f"Warning: Unknown integrator type '{integrator_type}', defaulting to RK4")
            self.integrator = RK4Integrator(dt)
        
        # Simulation parameters
        self.steps = self.cfg["time"]["steps"]
        
        # Simulation time
        self.time = 0.0
        
        # Output settings
        self.save_interval = self.cfg.get("output", {}).get("save_interval", 10)
        self.save_physics_fields = self.cfg.get("output", {}).get("save_physics_fields", True)
        self.save_plots = self.cfg.get("output", {}).get("save_plots", True)
        
        # Set default output directory
        self.output_dir = Path("./simulation_output")
        
        # Statistics
        self.step_times = []
        self.total_cells = []
    
    def initialize_fields(self, initial_conditions=None):
        """
        Initialize the simulation fields using FieldManager.
        
        Args:
            initial_conditions: Optional initial conditions dictionary
        """
        self.field_manager.initialize_fields(initial_conditions)
    
    def dynamics_function(self, phi_hat, nutrient_field, dx, source_terms=None):
        """
        Wrapper function for cell dynamics computation.
        
        Args:
            phi_hat: Cell fraction fields
            nutrient_field: Nutrient field
            dx: Grid spacing
            source_terms: Pre-computed source terms (optional, for efficiency)
            
        Returns:
            dphi_hat: Cell dynamics derivatives
        """
        return self.cell_dynamics.compute_dynamics(phi_hat, nutrient_field, dx, source_terms)
    
    def nutrient_function(self, phi_hat, nutrient_field, dx, dt):
        """
        Compute nutrient field update using the nutrient manager.
        
        Args:
            phi_hat: Cell fraction fields
            nutrient_field: Current nutrient field
            dx: Grid spacing
            dt: Time step
            
        Returns:
            nutrient_field_new: Updated nutrient field
        """
        return self.nutrient_manager.compute_nutrient_update(phi_hat, nutrient_field, dx, dt)
    
    def step_debug(self, step: int):
        """
        Perform one simulation step with detailed profiling and debug print statements.
        
        Args:
            step: Current simulation step
            
        Returns:
            success: Whether the step was successful
        """
        print(f"\n=== Step {step} Debug Information ===")
        
        # Start overall step timing
        self.profiler.start_timer("total_step")
        
        # Get current fields from field manager
        self.profiler.start_timer("field_retrieval")
        phi_hat, nutrient_field, host_field = self.field_manager.get_cell_fields()
        self.profiler.end_timer("field_retrieval")
        print(f"Field retrieval completed")
        
        # Compute source terms once per time step (not for each RK4 coefficient)
        self.profiler.start_timer("source_terms_computation")
        source_terms = self.cell_dynamics.compute_source_terms(phi_hat, nutrient_field)
        self.profiler.end_timer("source_terms_computation")
        print(f"Source terms computed")
        
        # Create a dynamics function that uses the pre-computed source terms
        def dynamics_function_with_source(phi_hat, nutrient_field, dx):
            return self.dynamics_function(phi_hat, nutrient_field, dx, source_terms)
        
        # Perform RK4 step with nutrient update
        if isinstance(self.integrator, AdaptiveRK4Integrator):
            self.profiler.start_timer("adaptive_integration")
            phi_hat_new, dt_used, success = self.integrator.step_adaptive(
                phi_hat, nutrient_field, self.field_manager.dx, dynamics_function_with_source
            )
            self.profiler.end_timer("adaptive_integration")
            print(f"Adaptive integration completed, dt_used: {dt_used}")
            
            if success:
                self.profiler.start_timer("nutrient_update")
                nutrient_field_new = self.nutrient_function(
                    phi_hat_new, nutrient_field, self.field_manager.dx, dt_used
                )
                self.profiler.end_timer("nutrient_update")
                print(f"Nutrient update completed")
        elif isinstance(self.integrator, AdaptiveGridRK4Integrator):
            self.profiler.start_timer("adaptive_grid_integration")
            phi_hat_new, nutrient_field_new = self.integrator.step_with_nutrient_update(
                phi_hat, nutrient_field, self.field_manager.dx,
                dynamics_function_with_source, self.nutrient_function
            )
            self.profiler.end_timer("adaptive_grid_integration")
            print(f"Adaptive grid integration completed")
            
            success = True
            dt_used = self.integrator.dt
        else:
            self.profiler.start_timer("rk4_integration")
            phi_hat_new, nutrient_field_new = self.integrator.step_optimized(
                phi_hat, nutrient_field, self.field_manager.dx,
                dynamics_function_with_source, self.nutrient_function
            )
            self.profiler.end_timer("rk4_integration")
            print(f"RK4 integration completed")
            
            success = True
            dt_used = self.integrator.dt
        
        if success:
            # Update physics fields (this also updates the field manager's stored fields)
            self.profiler.start_timer("physics_fields_update")
            self.field_manager.update_physics_fields(phi_hat_new, nutrient_field_new, self.cell_dynamics, source_terms)
            self.profiler.end_timer("physics_fields_update")
            print(f"Physics fields updated")
            
            # Only normalize volume fractions if pressure is enabled (to prevent mass loss during coagulation)
            disable_pressure = self.cfg.get("physics", {}).get("disable_pressure", False)
            if not disable_pressure:
                # Normalize volume fractions after each time step
                self.profiler.start_timer("volume_fraction_normalization")
                phi_hat_normalized = self.field_manager.normalize_volume_fractions(add_host_field=False)
                self.profiler.end_timer("volume_fraction_normalization")
                print(f"Volume fraction normalization completed")
                
                # Update fields with normalized values (only if normalization was needed)
                if phi_hat_normalized is not None:
                    self.profiler.start_timer("field_update")
                    self.field_manager.set_cell_fields(phi_hat_normalized, nutrient_field_new)
                    self.profiler.end_timer("field_update")
                    print(f"Fields updated with normalized values")
            else:
                # When pressure is disabled, only clip to prevent negative values but don't normalize
                # This preserves mass conservation during coagulation
                self.profiler.start_timer("volume_fraction_clipping")
                phi_hat_clipped = np.clip(phi_hat_new, 0.0, None)  # Only clip negative values
                self.field_manager.set_cell_fields(phi_hat_clipped, nutrient_field_new)
                self.profiler.end_timer("volume_fraction_clipping")
                print(f"Volume fraction clipping completed (no normalization to preserve mass)")
        
        # Update time
        self.time += dt_used
        
        # End overall step timing
        self.profiler.end_timer("total_step")
        
        # Record statistics
        step_time = self.profiler.timings["total_step"][-1] if self.profiler.timings["total_step"] else 0
        self.step_times.append(step_time)
        
        # Get the final normalized fields for statistics
        phi_hat_final, _, _ = self.field_manager.get_cell_fields()
        total_cells = np.sum(phi_hat_final)
        self.total_cells.append(total_cells)
        
        # Check volume fraction constraints and log if there are issues
        is_valid, max_deviation, mean_deviation = self.field_manager.check_volume_fraction_constraints()
        if not is_valid:  # Warn if there are negative values or overflow
            print(f"Warning: Volume fraction constraint violation - Max deviation: {max_deviation:.6f}, Mean deviation: {mean_deviation:.6f}")
        
        # Print timing breakdown for this step
        self.profiler.print_step_summary(step)
        
        print(f"=== Step {step} completed successfully ===\n")
        
        return success
    
    def step(self):
        """
        Perform one simulation step without any profiling or print statements.
        
        Returns:
            success: Whether the step was successful
        """
        # Get current fields from field manager
        phi_hat, nutrient_field, host_field = self.field_manager.get_cell_fields()
        
        # Compute source terms once per time step (not for each RK4 coefficient)
        source_terms = self.cell_dynamics.compute_source_terms(phi_hat, nutrient_field)
        
        # Create a dynamics function that uses the pre-computed source terms
        def dynamics_function_with_source(phi_hat, nutrient_field, dx):
            return self.dynamics_function(phi_hat, nutrient_field, dx, source_terms)
        
        # Perform RK4 step with nutrient update
        if isinstance(self.integrator, AdaptiveRK4Integrator):
            phi_hat_new, dt_used, success = self.integrator.step_adaptive(
                phi_hat, nutrient_field, self.field_manager.dx, dynamics_function_with_source
            )
            
            if success:
                nutrient_field_new = self.nutrient_function(
                    phi_hat_new, nutrient_field, self.field_manager.dx, dt_used
                )
        elif isinstance(self.integrator, AdaptiveGridRK4Integrator):
            phi_hat_new, nutrient_field_new = self.integrator.step_with_nutrient_update(
                phi_hat, nutrient_field, self.field_manager.dx,
                dynamics_function_with_source, self.nutrient_function
            )
            
            success = True
            dt_used = self.integrator.dt
        else:
            phi_hat_new, nutrient_field_new = self.integrator.step_optimized(
                phi_hat, nutrient_field, self.field_manager.dx,
                dynamics_function_with_source, self.nutrient_function
            )
            
            success = True
            dt_used = self.integrator.dt
        
        if success:
            # Update physics fields (this also updates the field manager's stored fields)
            self.field_manager.update_physics_fields(phi_hat_new, nutrient_field_new, self.cell_dynamics, source_terms)
            
            # Only normalize volume fractions if pressure is enabled (to prevent mass loss during coagulation)
            disable_pressure = self.cfg.get("physics", {}).get("disable_pressure", False)
            if not disable_pressure:
                # Normalize volume fractions after each time step
                phi_hat_normalized = self.field_manager.normalize_volume_fractions(add_host_field=True)
                
                # Update fields with normalized values (only if normalization was needed)
                if phi_hat_normalized is not None:
                    self.field_manager.set_cell_fields(phi_hat_normalized, nutrient_field_new)
            else:
                # When pressure is disabled, use conservative host field updates
                # This preserves mass conservation during coagulation while maintaining volume constraints
                phi_hat_clipped = np.clip(phi_hat_new, 0.0, None)  # Only clip negative values
                self.field_manager.set_cell_fields(phi_hat_clipped, nutrient_field_new)
        
        # Update time
        self.time += dt_used
        
        # Record statistics (minimal)
        self.step_times.append(0.0)  # No timing recorded
        
        # Get the final normalized fields for statistics
        phi_hat_final, _, _ = self.field_manager.get_cell_fields()
        total_cells = np.sum(phi_hat_final)
        self.total_cells.append(total_cells)
        
        # Check volume fraction constraints and log if there are issues
        is_valid, max_deviation, mean_deviation = self.field_manager.check_volume_fraction_constraints()
        if not is_valid:  # Warn if there are negative values or overflow
            print(f"Warning: Volume fraction constraint violation - Max deviation: {max_deviation:.6f}, Mean deviation: {mean_deviation:.6f}")
        
        # Check mass conservation in compaction mode
        disable_pressure = self.cfg.get("physics", {}).get("disable_pressure", False)
        if disable_pressure:
            is_conserved, mass_change, relative_change = self.field_manager.check_mass_conservation()
            if not is_conserved:
                print(f"Warning: Mass conservation violated - mass_change: {mass_change:.2e}, relative_change: {relative_change:.2f}%")
        
        return success
    
    def get_step_timing_breakdown(self, step: int):
        """
        Get detailed timing breakdown for a specific step.
        
        Args:
            step: Step number to get breakdown for
            
        Returns:
            breakdown: Dictionary with timing information
        """
        if not self.profiler.enabled:
            return {}
        
        breakdown = {
            "step": step,
            "total_time": self.step_times[step - 1] if step <= len(self.step_times) else 0,
            "module_timings": {}
        }
        
        # Get module timings for this step
        for module, times in self.profiler.timings.items():
            if step <= len(times):
                breakdown["module_timings"][module] = times[step - 1]
        
        return breakdown
    
    def get_performance_summary(self):
        """
        Get comprehensive performance summary.
        
        Returns:
            summary: Dictionary with performance statistics
        """
        if not self.profiler.enabled:
            return {}
        
        summary = {
            "total_steps": len(self.step_times),
            "average_step_time": np.mean(self.step_times) if self.step_times else 0,
            "total_simulation_time": np.sum(self.step_times) if self.step_times else 0,
            "module_averages": self.profiler.get_average_timings(),
            "step_times": self.step_times.copy()
        }
        
        return summary
    
    def save_plots(self, step):
        """
        Save physics field plots.
        
        Args:
            step: Current simulation step
        """
        if not self.save_plots or step % self.save_interval != 0:
            return
        
        plots_dir = self.output_dir / "plots"
        phi_hat, _, _ = self.field_manager.get_cell_fields()
        self.field_manager.plot_physics_fields(phi_hat, str(plots_dir), step)
    
    def save_output(self, step):
        """
        Save simulation output.
        
        Args:
            step: Current simulation step
        """
        if step % self.save_interval != 0:
            return
        
        # Create output filename
        filename = f"output_step_{step:06d}.npz"
        filepath = self.output_dir / filename
        
        # Get current fields from field manager
        phi_hat, nutrient_field, host_field = self.field_manager.get_cell_fields()
        
        # Prepare data for saving
        save_data = {
            "time": self.time,
            "phi_hat": phi_hat,
            "nutrient_field": nutrient_field,
            "step": step
        }
        
        # Add physics fields if requested
        if self.save_physics_fields:
            save_data.update({
                "pressure": self.field_manager.pressure,
                "velocity": self.field_manager.velocity,
                "energy_derivative": self.field_manager.energy_derivative,
                "mass_flux": self.field_manager.mass_flux
            })
        
        # Save data
        np.savez_compressed(filepath, **save_data)
    
    def run(self, initial_conditions=None, profile_interval=10):
        """
        Run the complete simulation with optional profiling.
        
        Args:
            initial_conditions: Optional initial conditions dictionary
            profile_interval: How often to print detailed timing breakdown (0 to disable)
            
        Returns:
            simulation_data: Dictionary containing simulation results
        """
        print(f"Starting tumor growth simulation...")
        print(f"Grid size: {self.field_manager.grid}")
        print(f"Number of populations: {self.field_manager.M}")
        print(f"Time steps: {self.steps}")
        print(f"Time step size: {self.integrator.dt}")
        if profile_interval > 0:
            print(f"Profiling enabled - detailed breakdown every {profile_interval} steps")
        
        # Initialize fields using field manager
        self.initialize_fields(initial_conditions)
        
        # Main simulation loop with tqdm progress bar
        with tqdm(total=self.steps, desc="Simulation Progress", unit="step") as pbar:
            for step in range(1, self.steps + 1):
                # Perform simulation step
                if profile_interval > 0 and step % profile_interval == 0:
                    success = self.step_debug(step)
                else:
                    success = self.step()
                
                if not success:
                    print(f"Warning: Step {step} failed, retrying with smaller time step")
                    continue
                
                # Save output
                #self.save_output(step)
                
                # Save plots
                #self.save_plots(step)
                
                # Update progress bar with current info
                phi_hat, _, _ = self.field_manager.get_cell_fields()
                total_cells = np.sum(phi_hat)
                avg_step_time = np.mean(self.step_times[-10:]) if len(self.step_times) >= 10 else 0
                pbar.set_postfix({
                    'Time': f'{self.time:.3f}',
                    'Cells': f'{total_cells:.3f}',
                    'Avg Step': f'{avg_step_time:.3f}s'
                })
                pbar.update(1)
        
        print(f"Simulation completed!")
        print(f"Final time: {self.time:.3f}")
        print(f"Average step time: {np.mean(self.step_times):.3f}s")
        
        # Print final performance summary
        self.profiler.print_final_summary()
        
        # Return simulation data
        phi_hat, nutrient_field, host_field = self.field_manager.get_cell_fields()
        simulation_data = {
            "time": self.time,
            "phi_hat": phi_hat,
            "nutrient_field": nutrient_field,
            "host_field": host_field,
            "field_manager": self.field_manager,
            "step_times": self.step_times,
            "total_cells": self.total_cells,
            "performance_summary": self.get_performance_summary()
        }
        
        return simulation_data

    def run_and_save_simulation(self, total_steps, save_interval=1, output_dir=None, 
                               initial_conditions=None, profile_interval=0, 
                               save_physics_fields=True, save_plots=False):
        """
        Run simulation with customizable parameters and save all field data for later plotting.
        
        Args:
            total_steps: Total number of simulation steps to run
            save_interval: How often to save data (every nth step, default=1 for every step)
            output_dir: Directory to save simulation data (defaults to self.output_dir)
            initial_conditions: Optional initial conditions dictionary
            profile_interval: How often to print detailed timing breakdown (0 to disable)
            save_physics_fields: Whether to save physics fields (pressure, velocity, etc.)
            save_plots: Whether to save plots during simulation (can be slow)
            
        Returns:
            simulation_data: Dictionary containing simulation results and metadata
        """
        if output_dir is None:
            # Skip all file operations for experiments
            output_dir = None
        else:
            output_dir = Path(output_dir)
            # Create output directory only if we're actually saving files
            os.makedirs(output_dir, exist_ok=True)
        
        print(f"Starting tumor growth simulation with save/load functionality...")
        print(f"Grid size: {self.field_manager.grid}")
        print(f"Number of populations: {self.field_manager.M}")
        print(f"Total time steps: {total_steps}")
        print(f"Save interval: {save_interval}")
        print(f"Time step size: {self.integrator.dt}")
        if output_dir is not None:
            print(f"Output directory: {output_dir}")
        else:
            print("Output directory: None (experiment mode - no files saved)")
        if profile_interval > 0:
            print(f"Profiling enabled - detailed breakdown every {profile_interval} steps")
        
        # Initialize fields using field manager
        self.initialize_fields(initial_conditions)
        
        # Save initial conditions (step 0, time 0.0)
        phi_hat_initial, nutrient_field_initial, host_field_initial = self.field_manager.get_cell_fields()
        
        # Initialize storage for saved data
        saved_steps = [0]  # Start with step 0
        saved_times = [0.0]  # Start with time 0.0
        # Use float32 to reduce memory footprint (halves memory usage)
        saved_phi_hat = [phi_hat_initial.copy().astype(np.float32)]  # Save initial conditions
        saved_nutrient_fields = [nutrient_field_initial.copy().astype(np.float32)]  # Save initial nutrient field
        saved_host_fields = [host_field_initial.copy().astype(np.float32)]  # Save initial host field
        saved_physics_data = []
        
        # Save initial physics data if requested
        if save_physics_fields:
            # Initialize physics fields for step 0
            self.field_manager.update_physics_fields(phi_hat_initial, nutrient_field_initial, self.cell_dynamics)
            initial_physics_data = {
                "pressure": self.field_manager.pressure.copy().astype(np.float32),
                "velocity": self.field_manager.velocity.copy().astype(np.float32),
                "energy_derivative": self.field_manager.energy_derivative.copy().astype(np.float32),
                "mass_flux": self.field_manager.mass_flux.copy().astype(np.float32),
                "source_terms": self.field_manager.source_terms.copy().astype(np.float32)
            }
            saved_physics_data.append(initial_physics_data)
        
        print(f"Saved initial conditions (step 0, time 0.0)")
        
        # Main simulation loop with tqdm progress bar
        with tqdm(total=total_steps, desc="Simulation Progress", unit="step") as pbar:
            for step in range(1, total_steps + 1):
                # Perform simulation step
                if profile_interval > 0 and step % profile_interval == 0:
                    success = self.step_debug(step)
                else:
                    success = self.step()
                
                if not success:
                    print(f"Warning: Step {step} failed, retrying with smaller time step")
                    continue
                
                # Save data at specified intervals
                if step % save_interval == 0:
                    # Get current fields
                    phi_hat, nutrient_field, host_field = self.field_manager.get_cell_fields()
                    
                    # Store basic data (use copy to avoid reference issues, but optimize memory)
                    saved_steps.append(step)
                    saved_times.append(self.time)
                    # Use float32 to reduce memory footprint
                    saved_phi_hat.append(phi_hat.copy().astype(np.float32))
                    saved_nutrient_fields.append(nutrient_field.copy().astype(np.float32))
                    saved_host_fields.append(host_field.copy().astype(np.float32))
                    
                    # Store physics data if requested
                    if save_physics_fields:
                        physics_data = {
                            "pressure": self.field_manager.pressure.copy().astype(np.float32),
                            "velocity": self.field_manager.velocity.copy().astype(np.float32),
                            "energy_derivative": self.field_manager.energy_derivative.copy().astype(np.float32),
                            "mass_flux": self.field_manager.mass_flux.copy().astype(np.float32),
                            "source_terms": self.field_manager.source_terms.copy().astype(np.float32)
                        }
                        saved_physics_data.append(physics_data)
                    
                    # Periodic memory cleanup: clear LRU caches every 10 steps to prevent memory bloat
                    if step % 10 == 0:
                        from src.growkit.Integrator.PressureSolver import clear_pressure_solver_caches
                        clear_pressure_solver_caches()
                        import gc
                        gc.collect()  # Force garbage collection
                    
                    # Save plots if requested
                    if save_plots and output_dir is not None:
                        self.save_plots(step)
                
                # Update progress bar with current info
                phi_hat, _, _ = self.field_manager.get_cell_fields()
                total_cells = np.sum(phi_hat)
                avg_step_time = np.mean(self.step_times[-10:]) if len(self.step_times) >= 10 else 0
                pbar.set_postfix({
                    'Time': f'{self.time:.3f}',
                    'Cells': f'{total_cells:.3f}',
                    'Avg Step': f'{avg_step_time:.3f}s',
                    'Saved': len(saved_steps)
                })
                pbar.update(1)
        
        print(f"Simulation completed!")
        print(f"Final time: {self.time:.3f}")
        print(f"Average step time: {np.mean(self.step_times):.3f}s")
        print(f"Saved {len(saved_steps)} data points")
        
        # Print final performance summary
        self.profiler.print_final_summary()
        
        # Print adaptive grid statistics if using adaptive grid integrator
        if isinstance(self.integrator, AdaptiveGridRK4Integrator):
            print("\n" + "="*50)
            self.integrator.print_statistics()
            print("="*50)
        
        # Prepare simulation data for return
        simulation_data = {
            "metadata": {
                "total_steps": total_steps,
                "save_interval": save_interval,
                "final_time": self.time,
                "grid_size": self.field_manager.grid,
                "num_populations": self.field_manager.M,
                "population_labels": self.field_manager.labels,  # Add population labels
                "population_names": list(self.cfg["populations"].keys()),  # Add population names (keys)
                "time_step": self.integrator.dt,
                "output_dir": str(output_dir),
                "saved_steps": saved_steps,
                "saved_times": saved_times,
                "config": self.cfg  # Include full config for reference
            },
            "field_data": {
                "phi_hat": saved_phi_hat,
                "nutrient_fields": saved_nutrient_fields,
                "host_fields": saved_host_fields
            },
            "performance": {
                "step_times": self.step_times,
                "total_cells": self.total_cells,
                "performance_summary": self.get_performance_summary()
            }
        }
        
        # Add physics data if saved
        if save_physics_fields and saved_physics_data:
            simulation_data["physics_data"] = saved_physics_data
        
        # Save complete simulation data to file only if output_dir is specified
        if output_dir is not None:
            simulation_file = output_dir / "simulation_data.npz"
            self._save_simulation_data(simulation_data, simulation_file)
            print(f"Complete simulation data saved to {simulation_file}")
        
        return simulation_data
    
    def _save_simulation_data(self, simulation_data, filepath):
        """
        Save simulation data to compressed numpy file.
        
        Args:
            simulation_data: Dictionary containing simulation data
            filepath: Path to save the data
        """
        try:
            # Convert lists to arrays for efficient storage
            save_dict = {
                "phi_hat": np.array(simulation_data["field_data"]["phi_hat"]),
                "nutrient_fields": np.array(simulation_data["field_data"]["nutrient_fields"]),
                "host_fields": np.array(simulation_data["field_data"]["host_fields"]),
                "step_times": np.array(simulation_data["performance"]["step_times"]),
                "total_cells": np.array(simulation_data["performance"]["total_cells"])
            }
            
            # Save metadata separately as a pickle object
            import pickle
            metadata_bytes = pickle.dumps(simulation_data["metadata"])
            save_dict["metadata"] = np.frombuffer(metadata_bytes, dtype=np.uint8)
            
            # Add physics data if available
            if "physics_data" in simulation_data:
                physics_data = simulation_data["physics_data"]
                try:
                    save_dict.update({
                        "pressure": np.array([p["pressure"] for p in physics_data]),
                        "velocity": np.array([p["velocity"] for p in physics_data]),
                        "energy_derivative": np.array([p["energy_derivative"] for p in physics_data]),
                        "mass_flux": np.array([p["mass_flux"] for p in physics_data]),
                        "source_terms": np.array([p["source_terms"] for p in physics_data])
                    })
                except Exception as e:
                    print(f"Warning: Could not save physics data: {e}")
                    # Try to save without source_terms which might be the issue
                    try:
                        save_dict.update({
                            "pressure": np.array([p["pressure"] for p in physics_data]),
                            "velocity": np.array([p["velocity"] for p in physics_data]),
                            "energy_derivative": np.array([p["energy_derivative"] for p in physics_data]),
                            "mass_flux": np.array([p["mass_flux"] for p in physics_data])
                        })
                    except Exception as e2:
                        print(f"Warning: Could not save any physics data: {e2}")
            
            # Save with compression
            np.savez_compressed(filepath, **save_dict)
            
        except Exception as e:
            print(f"Error saving simulation data: {e}")
            print(f"Simulation data keys: {list(simulation_data.keys())}")
            if "physics_data" in simulation_data:
                print(f"Physics data keys: {list(simulation_data['physics_data'][0].keys()) if simulation_data['physics_data'] else 'Empty'}")
            raise
    
    def load_simulation_data(self, filepath, use_memory_map=True):
        """
        Load simulation data from saved file with memory-efficient loading.
        
        Args:
            filepath: Path to the saved simulation data file
            use_memory_map: If True, use memory mapping to avoid loading all data at once
            
        Returns:
            simulation_data: Dictionary containing loaded simulation data
        """
        print(f"Loading simulation data from {filepath}...")
        
        # Use memory mapping to avoid loading everything into memory at once
        if use_memory_map:
            data = np.load(filepath, allow_pickle=True, mmap_mode='r')  # Read-only memory mapping
        else:
            data = np.load(filepath, allow_pickle=True)
        
        # Load metadata from pickle bytes (this is small, load it fully)
        import pickle
        metadata_bytes = data["metadata"].tobytes()
        metadata = pickle.loads(metadata_bytes)
        
        # Get array shapes without loading full arrays
        num_steps = data["phi_hat"].shape[0]
        
        # Reconstruct simulation data structure
        # Use memory-mapped arrays directly - they'll be loaded on-demand
        simulation_data = {
            "metadata": metadata,
            "field_data": {
                "phi_hat": data["phi_hat"],  # Memory-mapped, loads on access
                "nutrient_fields": data["nutrient_fields"],  # Memory-mapped
                "host_fields": data["host_fields"]  # Memory-mapped
            },
            "performance": {
                "step_times": data["step_times"],  # Small array, load fully
                "total_cells": data["total_cells"]  # Small array, load fully
            },
            "_npz_file": data  # Keep reference to prevent garbage collection
        }
        
        # Add physics data if available - use lazy loading
        if "pressure" in data:
            # Don't load all physics data at once - create a lazy accessor
            class LazyPhysicsData:
                def __init__(self, npz_data, num_steps):
                    self.npz_data = npz_data
                    self.num_steps = num_steps
                    self._cache = {}  # Cache loaded steps
                    self._max_cache_size = 3  # Cache only last 3 accessed steps (reduced for memory)
                    self._access_order = []  # Track access order for LRU eviction
                
                def __len__(self):
                    return self.num_steps
                
                def clear_cache(self):
                    """Clear the cache to free memory."""
                    self._cache.clear()
                    self._access_order.clear()
                    import gc
                    gc.collect()
                
                def __getitem__(self, idx):
                    # Normalize index (support negative indexing)
                    if idx < 0:
                        idx = self.num_steps + idx
                    
                    # Cache management: if cache is full, remove least recently used
                    if len(self._cache) >= self._max_cache_size and idx not in self._cache:
                        # Remove least recently used entry
                        if self._access_order:
                            lru_key = self._access_order.pop(0)
                            if lru_key in self._cache:
                                del self._cache[lru_key]
                    
                    if idx not in self._cache:
                        # Load this step's physics data on demand
                        # Use memory-mapped access and convert to float32 immediately
                        physics_data = {
                            "pressure": np.array(self.npz_data["pressure"][idx], dtype=np.float32, copy=False),
                            "velocity": np.array(self.npz_data["velocity"][idx], dtype=np.float32, copy=False),
                            "energy_derivative": np.array(self.npz_data["energy_derivative"][idx], dtype=np.float32, copy=False),
                            "mass_flux": np.array(self.npz_data["mass_flux"][idx], dtype=np.float32, copy=False)
                        }
                        # Add source terms if available
                        if "source_terms" in self.npz_data:
                            physics_data["source_terms"] = np.array(self.npz_data["source_terms"][idx], dtype=np.float32, copy=False)
                        
                        self._cache[idx] = physics_data
                    
                    # Update access order (move to end if already in cache)
                    if idx in self._access_order:
                        self._access_order.remove(idx)
                    self._access_order.append(idx)
                    
                    return self._cache[idx]
            
            simulation_data["physics_data"] = LazyPhysicsData(data, num_steps)
        
        print(f"Loaded simulation data with {num_steps} saved steps (using memory mapping: {use_memory_map})")
        return simulation_data
