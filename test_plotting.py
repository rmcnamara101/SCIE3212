#!/usr/bin/env python3
"""
Test script to demonstrate the new plotting functionality for source terms and nutrient fields.
"""

import numpy as np
from pathlib import Path
from src.growkit.Simulator import TumorGrowthSimulator
from src.growkit.PlotEngine.PhysicsFieldsPlotter import PhysicsFieldsPlotter

def main():
    # Configuration file path
    proj = Path(__file__).parent
    cfg = proj / "templates" / "og.yaml"
    
    # Create simulator
    simulator = TumorGrowthSimulator(str(cfg))
    
    # Run a short simulation to generate data
    print("Running simulation to generate data...")
    simulator.run_and_save_simulation(
        total_steps=5, 
        save_interval=1, 
        save_physics_fields=True, 
        output_dir=str(proj / "test_output")
    )
    
    # Load the simulation data
    simulation_data = simulator.load_simulation_data(proj / "test_output" / "simulation_data.npz")
    
    # Create plotter
    plotter = PhysicsFieldsPlotter.from_simulation_data(simulation_data, step_idx=0)
    
    # Test plotting source terms
    print("\nTesting source terms plotting...")
    plotter.plot_source_terms(
        population_idx=0,  # First population
        step=0,
        z_slice=25,  # Center slice
        cmap="RdBu_r",  # Red for growth, blue for death
        save_plot=True,
        show_plot=False,
        output_dir=str(proj / "test_output" / "plots"),
        add_boundary_contours=True
    )
    
    # Test plotting nutrient field
    print("\nTesting nutrient field plotting...")
    plotter.plot_nutrient_field(
        step=0,
        z_slice=25,  # Center slice
        cmap="viridis",
        save_plot=True,
        show_plot=False,
        output_dir=str(proj / "test_output" / "plots"),
        add_boundary_contours=True
    )
    
    # Test plotting from saved data
    print("\nTesting plotting from saved data...")
    plotter.plot_source_terms_from_saved(
        simulation_data,
        population_idx=0,
        step_idx=0,
        z_slice=25,
        save_plot=True,
        show_plot=False,
        output_dir=str(proj / "test_output" / "plots_from_saved"),
        add_boundary_contours=True
    )
    
    plotter.plot_nutrient_field_from_saved(
        simulation_data,
        step_idx=0,
        z_slice=25,
        save_plot=True,
        show_plot=False,
        output_dir=str(proj / "test_output" / "plots_from_saved"),
        add_boundary_contours=True
    )
    
    print("\nPlotting tests completed!")
    print("Check the test_output directory for generated plots.")

if __name__ == "__main__":
    main()
