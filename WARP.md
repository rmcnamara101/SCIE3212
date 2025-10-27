# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

SilicoKit is a tumor growth simulation framework built in Python that models cellular dynamics using partial differential equations. The framework uses a multi-phase approach to simulate tumor cell populations, nutrient fields, and their interactions in 3D space.

## Architecture

The codebase follows a modular architecture with clear separation of concerns:

### Core Components (src/growkit/)

- **Simulator**: Main orchestrator class (`Simulator.py`) that coordinates all components and runs the simulation loop
- **FieldManager**: Manages spatial fields (cell fractions, nutrients, physics fields) and their initialization
- **PhysicsEngine**: Contains vectorized physics computations including:
  - `VectorizedCellDynamics`: Main cell dynamics computations 
  - `VectorizedEnergy`: Adhesion energy calculations
  - `VectorizedMassFlux`: Mass flux computations
  - `VectorizedSolidVelocity`: Solid velocity field calculations
  - `Nutrient`: Nutrient field dynamics
- **Integrator**: Multiple integration schemes (RK4, Forward Euler, Adaptive Grid) for time stepping
- **ProductionEngine**: Source term construction for growth/death dynamics
- **PlotEngine**: Visualization and analysis tools
- **MathEngine**: Mathematical operators and boundary conditions
- **ExperimentEngine**: Parameter sweep and experiment management

### Configuration System

Configuration uses YAML files in `configs/`:
- Main config: `configs/T_N.yaml` (Tumor-Necrotic model)
- Template: `configs/template.yaml`
- Alternative: `configs/csc-t-n.yaml` (Cancer Stem Cell model)

Configuration sections:
- `domain`: Grid size and spacing
- `time`: Time steps and dt
- `populations`: Cell type definitions with dynamics parameters
- `nutrient`: Nutrient field properties
- `physics`: Adhesion energy and pressure settings
- `initial_conditions`: Initial field setup
- `integrator`: Integration method selection

### Key Physics

The simulation solves the dynamics equation:
```
dφ_hat/dt = -∇·(u ⊗ φ_hat) - ∇·J_hat + A_hat
```
where:
- `φ_hat`: Cell fraction fields for all populations
- `u`: Solid velocity field
- `J_hat`: Mass flux fields
- `A_hat`: Source terms (growth/death)

## Development Commands

### Environment Setup
```bash
# Activate virtual environment (if using .venv)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running Simulations
```bash
# Basic simulation run
python run_simulation.py

# Launch GUI for analysis
python run_gui.py

# Run parameter experiments
python experiment_test.py
```

### Profiling and Performance Analysis
```bash
# Quick performance overview
python simple_timing_script.py

# Detailed step-by-step profiling
python detailed_step_profiler.py

# Comprehensive profiling with setup analysis
python simulation_profiler.py

# Easy profiler runner with options
python run_profiling.py                    # Simple profiling
python run_profiling.py detailed 5         # Detailed profiling, 5 steps
python run_profiling.py comprehensive 10   # Full profiling, 10 steps
```

### Adhesion Investigation
```bash
# Investigate adhesion parameter effects
python adhesion_investigation_script.py
```

### Parameter Analysis
```bash
# Analyze parameter sweep results
python analyze_parameter_sweep.py
```

## Important Implementation Details

### Performance Considerations
- All physics computations are vectorized using NumPy
- Numba JIT compilation is used for performance-critical functions
- Memory usage is tracked during simulation
- Multiple integrator types available for speed/accuracy tradeoffs

### Integrator Selection
Available integrator types (set in config `integrator.type`):
- `"rk4"`: Standard 4th-order Runge-Kutta (default)
- `"rk4_adaptive"`: Adaptive time-step RK4
- `"adaptive_grid"`: Adaptive grid refinement RK4
- `"forward_euler"`: Basic Forward Euler (fastest)
- `"forward_euler_adaptive"`: Adaptive Forward Euler
- `"improved_euler"`: Heun's method (2nd order)

### Field Management
- Cell fraction fields are normalized to prevent overflow (total ≤ 1)
- Host field automatically fills remaining space to maintain volume conservation
- Natural boundary conditions are applied to prevent cell leakage
- Physics fields (pressure, velocity) are computed on-demand

### Data Organization
- `laboratory/`: Output directory structure
  - `saved_simulations/`: Simulation results
  - `parameter_sweeps/`: Experiment data
  - `cell_fields/`: Field visualizations
  - `physics_fields/`: Physics field data

### Platform Compatibility
The codebase handles platform differences (macOS vs Windows) in path resolution and includes appropriate sys.path modifications for module imports.

### Profiling Targets
Performance targets for optimization:
- Setup time: < 1 second
- Step time: < 0.5 seconds (small grids)
- Memory usage: < 500 MB increase
- Integration: < 50% of step time
- Source terms: < 30% of step time

## Common Workflows

### Adding New Cell Populations
1. Add population definition to YAML config under `populations:`
2. Update `initial_conditions.seeding_densities` if needed
3. No code changes required - system is dynamically configured

### Changing Physics Parameters
- Modify `physics.adhesion_energy.m` for adhesion strength
- Adjust `nutrient.dynamics.diffusion` for nutrient transport
- Set `physics.disable_pressure: true` to turn off pressure calculations

### Custom Initial Conditions
Available types in `initial_conditions.type`:
- `"spherical"`: Single sphere
- `"organoid_settling"`: Deformed organoid shape
- `"multiple_spheres"`: Multiple tumor sites
- `"uniform"`: Uniform distribution
- `"gaussian_random_blob"`: Random distribution

### Performance Optimization
1. Use `simple_timing_script.py` to identify bottlenecks
2. Consider switching to Forward Euler for speed
3. Reduce grid size (`domain.shape`) or time step (`time.dt`)
4. Use adaptive grid integrator for sparse problems
5. Monitor memory usage to avoid swapping