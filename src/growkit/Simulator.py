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

# Pre-import all modules to avoid runtime import overhead
from src.growkit.Fields.FieldManager import FieldManager
from src.growkit.PhysicsEngine.VectorizedCellDynamics import VectorizedCellDynamics
from src.growkit.PhysicsEngine.Nutrient.NutrientField import NutrientField
from src.growkit.ProductionEngine.SourceConstructor import SourceConstructor
from src.growkit.Integrator.RK4Integrator import RK4Integrator, AdaptiveRK4Integrator


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
        if self.cfg.get("integrator", {}).get("adaptive", False):
            tolerance = self.cfg["integrator"].get("tolerance", 1e-6)
            min_dt = self.cfg["integrator"].get("min_dt", 1e-8)
            max_dt = self.cfg["integrator"].get("max_dt", 1e-2)
            self.integrator = AdaptiveRK4Integrator(dt, tolerance, min_dt, max_dt)
        else:
            self.integrator = RK4Integrator(dt)
        
        # Simulation parameters
        self.steps = self.cfg["time"]["steps"]
        
        # Simulation time
        self.time = 0.0
        
        # Output settings
        self.save_interval = self.cfg.get("output", {}).get("save_interval", 10)
        self.save_physics_fields = self.cfg.get("output", {}).get("save_physics_fields", True)
        self.save_plots = self.cfg.get("output", {}).get("save_plots", True)
        
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
    
    def nutrient_function(self, phi_hat, nutrient_field, dx):
        """
        Compute nutrient field update using the nutrient manager.
        
        Args:
            phi_hat: Cell fraction fields
            nutrient_field: Current nutrient field
            dx: Grid spacing
            
        Returns:
            nutrient_field_new: Updated nutrient field
        """
        return self.nutrient_manager.compute_nutrient_update(phi_hat, nutrient_field, dx)
    
    def step(self):
        """
        Perform one simulation step with detailed profiling.
        
        Returns:
            success: Whether the step was successful
        """
        # Start overall step timing
        self.profiler.start_timer("total_step")
        
        # Get current fields from field manager
        self.profiler.start_timer("field_retrieval")
        phi_hat, nutrient_field = self.field_manager.get_cell_fields()
        self.profiler.end_timer("field_retrieval")
        
        # Compute source terms once per time step (not for each RK4 coefficient)
        self.profiler.start_timer("source_terms_computation")
        source_terms = self.cell_dynamics.compute_source_terms(phi_hat, nutrient_field)
        self.profiler.end_timer("source_terms_computation")
        
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
            
            if success:
                self.profiler.start_timer("nutrient_update")
                nutrient_field_new = self.nutrient_function(
                    phi_hat_new, nutrient_field, self.field_manager.dx
                )
                self.profiler.end_timer("nutrient_update")
        else:
            self.profiler.start_timer("rk4_integration")
            phi_hat_new, nutrient_field_new = self.integrator.step_optimized(
                phi_hat, nutrient_field, self.field_manager.dx,
                dynamics_function_with_source, self.nutrient_function
            )
            self.profiler.end_timer("rk4_integration")
            
            success = True
            dt_used = self.integrator.dt
        
        if success:
            # Update physics fields (this also updates the field manager's stored fields)
            self.profiler.start_timer("physics_fields_update")
            self.field_manager.update_physics_fields(phi_hat_new, nutrient_field_new, self.cell_dynamics, source_terms)
            self.profiler.end_timer("physics_fields_update")
            
            # Normalize volume fractions after each time step
            self.profiler.start_timer("volume_fraction_normalization")
            phi_hat_normalized = self.field_manager.normalize_volume_fractions(add_host_field=False)
            self.profiler.end_timer("volume_fraction_normalization")
            
            # Update fields with normalized values (only if normalization was needed)
            if phi_hat_normalized is not None:
                self.profiler.start_timer("field_update")
                self.field_manager.set_cell_fields(phi_hat_normalized, nutrient_field_new)
                self.profiler.end_timer("field_update")
        
        # Update time
        self.time += dt_used
        
        # End overall step timing
        self.profiler.end_timer("total_step")
        
        # Record statistics
        step_time = self.profiler.timings["total_step"][-1] if self.profiler.timings["total_step"] else 0
        self.step_times.append(step_time)
        
        # Get the final normalized fields for statistics
        phi_hat_final, _ = self.field_manager.get_cell_fields()
        total_cells = np.sum(phi_hat_final)
        self.total_cells.append(total_cells)
        
        # Check volume fraction constraints and log if there are issues
        is_valid, max_deviation, mean_deviation = self.field_manager.check_volume_fraction_constraints()
        if not is_valid:  # Warn if there are negative values or overflow
            print(f"Warning: Volume fraction constraint violation - Max deviation: {max_deviation:.6f}, Mean deviation: {mean_deviation:.6f}")
        
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
        phi_hat, _ = self.field_manager.get_cell_fields()
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
        phi_hat, nutrient_field = self.field_manager.get_cell_fields()
        
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
        
        # Main simulation loop
        for step in range(1, self.steps + 1):
            # Perform simulation step
            success = self.step()
            
            if not success:
                print(f"Warning: Step {step} failed, retrying with smaller time step")
                continue
            
            # Save output
            #self.save_output(step)
            
            # Save plots
            #self.save_plots(step)
            
            # Print progress and profiling info
            if step % 10 == 0:
                phi_hat, _ = self.field_manager.get_cell_fields()
                total_cells = np.sum(phi_hat)
                avg_step_time = np.mean(self.step_times[-10:])
                print(f"Step {step}/{self.steps}, Time: {self.time:.3f}, "
                      f"Total cells: {total_cells:.3f}, Avg step time: {avg_step_time:.3f}s")
                
                # Print detailed profiling breakdown
                if profile_interval > 0 and step % profile_interval == 0:
                    self.profiler.print_step_summary(step)
        
        print(f"Simulation completed!")
        print(f"Final time: {self.time:.3f}")
        print(f"Average step time: {np.mean(self.step_times):.3f}s")
        
        # Print final performance summary
        self.profiler.print_final_summary()
        
        # Return simulation data
        phi_hat, nutrient_field = self.field_manager.get_cell_fields()
        simulation_data = {
            "time": self.time,
            "phi_hat": phi_hat,
            "nutrient_field": nutrient_field,
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
            output_dir = self.output_dir
        else:
            output_dir = Path(output_dir)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Starting tumor growth simulation with save/load functionality...")
        print(f"Grid size: {self.field_manager.grid}")
        print(f"Number of populations: {self.field_manager.M}")
        print(f"Total time steps: {total_steps}")
        print(f"Save interval: {save_interval}")
        print(f"Time step size: {self.integrator.dt}")
        print(f"Output directory: {output_dir}")
        if profile_interval > 0:
            print(f"Profiling enabled - detailed breakdown every {profile_interval} steps")
        
        # Initialize fields using field manager
        self.initialize_fields(initial_conditions)
        
        # Initialize storage for saved data
        saved_steps = []
        saved_times = []
        saved_phi_hat = []
        saved_nutrient_fields = []
        saved_physics_data = []
        
        # Main simulation loop
        for step in range(1, total_steps + 1):
            # Perform simulation step
            success = self.step()
            
            if not success:
                print(f"Warning: Step {step} failed, retrying with smaller time step")
                continue
            
            # Save data at specified intervals
            if step % save_interval == 0:
                # Get current fields
                phi_hat, nutrient_field = self.field_manager.get_cell_fields()
                
                # Store basic data
                saved_steps.append(step)
                saved_times.append(self.time)
                saved_phi_hat.append(phi_hat.copy())
                saved_nutrient_fields.append(nutrient_field.copy())
                
                # Store physics data if requested
                if save_physics_fields:
                    physics_data = {
                        "pressure": self.field_manager.pressure.copy(),
                        "velocity": self.field_manager.velocity.copy(),
                        "energy_derivative": self.field_manager.energy_derivative.copy(),
                        "mass_flux": self.field_manager.mass_flux.copy(),
                        "source_terms": self.field_manager.source_terms.copy()
                    }
                    saved_physics_data.append(physics_data)
                
                # Save plots if requested
                if save_plots:
                    self.save_plots(step)
                
                print(f"Saved data for step {step}/{total_steps}, Time: {self.time:.3f}")
            
            # Print progress and profiling info
            if step % 10 == 0:
                phi_hat, _ = self.field_manager.get_cell_fields()
                total_cells = np.sum(phi_hat)
                avg_step_time = np.mean(self.step_times[-10:])
                print(f"Step {step}/{total_steps}, Time: {self.time:.3f}, "
                      f"Total cells: {total_cells:.3f}, Avg step time: {avg_step_time:.3f}s")
                
                # Print detailed profiling breakdown
                if profile_interval > 0 and step % profile_interval == 0:
                    self.profiler.print_step_summary(step)
        
        print(f"Simulation completed!")
        print(f"Final time: {self.time:.3f}")
        print(f"Average step time: {np.mean(self.step_times):.3f}s")
        print(f"Saved {len(saved_steps)} data points")
        
        # Print final performance summary
        self.profiler.print_final_summary()
        
        # Prepare simulation data for return
        simulation_data = {
            "metadata": {
                "total_steps": total_steps,
                "save_interval": save_interval,
                "final_time": self.time,
                "grid_size": self.field_manager.grid,
                "num_populations": self.field_manager.M,
                "time_step": self.integrator.dt,
                "output_dir": str(output_dir),
                "saved_steps": saved_steps,
                "saved_times": saved_times
            },
            "field_data": {
                "phi_hat": saved_phi_hat,
                "nutrient_fields": saved_nutrient_fields
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
        
        # Save complete simulation data to file
        simulation_file = output_dir / "simulation_data.npz"
        self._save_simulation_data(simulation_data, simulation_file)
        print(f"Complete simulation data saved to {simulation_file}")
        
        return #simulation_data
    
    def _save_simulation_data(self, simulation_data, filepath):
        """
        Save simulation data to compressed numpy file.
        
        Args:
            simulation_data: Dictionary containing simulation data
            filepath: Path to save the data
        """
        # Convert lists to arrays for efficient storage
        save_dict = {
            "metadata": simulation_data["metadata"],
            "phi_hat": np.array(simulation_data["field_data"]["phi_hat"]),
            "nutrient_fields": np.array(simulation_data["field_data"]["nutrient_fields"]),
            "step_times": np.array(simulation_data["performance"]["step_times"]),
            "total_cells": np.array(simulation_data["performance"]["total_cells"])
        }
        
        # Add physics data if available
        if "physics_data" in simulation_data:
            physics_data = simulation_data["physics_data"]
            save_dict.update({
                "pressure": np.array([p["pressure"] for p in physics_data]),
                "velocity": np.array([p["velocity"] for p in physics_data]),
                "energy_derivative": np.array([p["energy_derivative"] for p in physics_data]),
                "mass_flux": np.array([p["mass_flux"] for p in physics_data]),
                "source_terms": np.array([p["source_terms"] for p in physics_data])
            })
        
        # Save with compression
        np.savez_compressed(filepath, **save_dict)
    
    def load_simulation_data(self, filepath):
        """
        Load simulation data from saved file.
        
        Args:
            filepath: Path to the saved simulation data file
            
        Returns:
            simulation_data: Dictionary containing loaded simulation data
        """
        print(f"Loading simulation data from {filepath}...")
        
        # Load data
        data = np.load(filepath, allow_pickle=True)
        
        # Reconstruct simulation data structure
        simulation_data = {
            "metadata": data["metadata"].item(),
            "field_data": {
                "phi_hat": data["phi_hat"],
                "nutrient_fields": data["nutrient_fields"]
            },
            "performance": {
                "step_times": data["step_times"],
                "total_cells": data["total_cells"]
            }
        }
        
        # Add physics data if available
        if "pressure" in data:
            simulation_data["physics_data"] = []
            for i in range(len(data["pressure"])):
                physics_data = {
                    "pressure": data["pressure"][i],
                    "velocity": data["velocity"][i],
                    "energy_derivative": data["energy_derivative"][i],
                    "mass_flux": data["mass_flux"][i]
                }
                # Add source terms if available
                if "source_terms" in data:
                    physics_data["source_terms"] = data["source_terms"][i]
                simulation_data["physics_data"].append(physics_data)
        
        print(f"Loaded simulation data with {len(simulation_data['field_data']['phi_hat'])} saved steps")
        return simulation_data
