# Physics Fields Plotting Documentation

This document describes the new separated plotting functionality for the GrowKit simulation framework.

## Overview

The plotting functionality has been separated from the `FieldManager` class into a dedicated `PhysicsFieldsPlotter` class. This provides better organization, more flexibility, and additional features for analyzing simulation results.

## Key Features

### 1. Separated Plotting Functionality
- All plotting methods are now in `PhysicsFieldsPlotter` class
- `FieldManager` focuses on data management only
- Cleaner separation of concerns

### 2. New Analysis Features
- **Mass Flux Near Boundary**: Analyze mass flux specifically in boundary regions
- **Total Tumor Density**: Plot the combined density of all cell populations
- **Tumor Boundary Contours**: Visualize tumor boundaries with customizable colors and line styles

### 3. Enhanced Flexibility
- **Default behavior**: Plots are always shown, not saved unless explicitly requested
- Control over save/display behavior for each plot
- Customizable output directories
- Configurable plot parameters (colormaps, scales, etc.)
- **Contour overlays**: Add tumor boundary contours to any plot for better spatial context

## Usage

### Basic Setup

```python
from src.growkit.FieldManager import FieldManager
from src.growkit.PlotEngine.PhysicsFieldsPlotter import PhysicsFieldsPlotter

# Initialize field manager
field_manager = FieldManager("path/to/config.yaml")
field_manager.initialize_fields()

# Get current data
phi_hat, nutrient_field = field_manager.get_cell_fields()
field_manager.update_physics_fields(phi_hat, nutrient_field)

# Create plotter
plotter = PhysicsFieldsPlotter(field_manager)
```

### Plot All Fields at Once

```python
# Plot all physics fields
plotter.plot_all_physics_fields(
    phi_hat, 
    step=0, 
    output_dir="plots",
    save_plots=True, 
    show_plots=False
)
```

### Individual Field Plotting

#### Pressure Field
```python
plotter.plot_pressure_field(
    step=0,
    z_slice=None,  # Defaults to center
    cmap="viridis",
    alpha=0.8,
    output_dir="plots",
    save_plot=True,
    show_plot=True
)
```

#### Velocity Field
```python
plotter.plot_velocity_field(
    step=0,
    z_slice=None,
    skip=2,  # Vector sampling rate
    cmap="viridis",
    scale=50,
    width=0.005,
    add_contours=True,  # Add tumor boundary contours
    contour_levels=[0.1, 0.3, 0.5],
    contour_colors=['red', 'orange', 'yellow'],
    contour_linewidths=[2, 1.5, 1]
)
```

#### Energy Derivative Field
```python
plotter.plot_energy_derivative_field(
    step=0,
    z_slice=None,
    cmap="RdBu_r",
    add_contours=True,  # Add tumor boundary contours
    contour_colors=['white', 'yellow', 'red']  # Different colors for visibility
)
```

#### Mass Flux Field
```python
plotter.plot_mass_flux_field(
    population_idx=0,  # Which population to plot
    step=0,
    z_slice=None,
    skip=2,
    cmap="viridis",
    scale=20,
    width=0.005,
    add_contours=True,  # Add tumor boundary contours
    contour_levels=[0.1, 0.5],  # Fewer contours for clarity
    contour_colors=['red', 'white']
)
```

### New Analysis Features

#### Mass Flux Near Boundary
This feature analyzes mass flux specifically in boundary regions where tumor density is between a threshold and 0.5.

```python
plotter.plot_mass_flux_near_boundary(
    population_idx=0,
    step=0,
    z_slice=None,
    boundary_threshold=0.1,  # Define boundary region
    skip=1,  # Show all vectors for detailed analysis
    cmap="plasma",
    scale=15,
    width=0.008
)
```

#### Total Tumor Density
Plot the combined density of all cell populations.

```python
plotter.plot_tumor_density(
    step=0,
    z_slice=None,
    cmap="viridis",
    add_contours=True,  # Add tumor boundary contours
    contour_levels=[0.1, 0.3, 0.5, 0.7],
    contour_colors=['red', 'orange', 'yellow', 'white']
)
```

#### Tumor Boundary Contours
Visualize tumor boundaries with customizable colors and line styles.

```python
plotter.plot_tumor_boundary_contour(
    population_idx=0,
    step=0,
    z_slice=None,
    contour_levels=[0.1, 0.3, 0.5, 0.7],  # Density levels
    contour_colors=['red', 'orange', 'yellow', 'white'],  # Line colors
    contour_linewidths=[3, 2, 1.5, 1],  # Line widths
    background_cmap="Blues"
)
```

#### Cell Density Fields
Plot individual or total cell density.

```python
# Total cell density with contours
plotter.plot_cell_density_field(
    phi_hat,
    population_idx=None,  # None for total density
    step=0,
    z_slice=None,
    cmap="viridis",
    add_contours=True,
    contour_colors=['white', 'yellow', 'red']
)

# Individual population density with contours
plotter.plot_cell_density_field(
    phi_hat,
    population_idx=0,  # Specific population
    step=0,
    z_slice=None,
    cmap="viridis",
    add_contours=True,
    contour_colors=['white', 'yellow', 'red']
)
```

## Parameters Reference

### Common Parameters
- `step`: Simulation step number (used in filenames)
- `z_slice`: Z-coordinate slice to plot (defaults to center)
- `output_dir`: Directory to save plots (default: None, no saving)
- `save_plot`: Whether to save the plot to file (default: False)
- `show_plot`: Whether to display the plot (default: True)
- `add_contours`: Whether to add tumor boundary contours (default: False)
- `contour_population_idx`: Which population to use for contours (default: 0)
- `contour_levels`: List of density levels for contours (default: [0.1, 0.3, 0.5])
- `contour_colors`: List of colors for each contour level (default: ['red', 'orange', 'yellow'])
- `contour_linewidths`: List of line widths for each contour level (default: [2, 1.5, 1])

### Vector Field Parameters
- `skip`: Sampling rate for vectors (every nth point)
- `cmap`: Colormap for the plot
- `scale`: Scale factor for vectors
- `width`: Width of vectors

### Boundary Analysis Parameters
- `boundary_threshold`: Density threshold for defining boundary region

## File Organization

Generated plots are saved with descriptive filenames:
- `pressure_field_step_000000.png`
- `velocity_field_step_000000.png`
- `energy_derivative_field_step_000000.png`
- `mass_flux_tumor_step_000000.png`
- `mass_flux_boundary_tumor_step_000000.png`
- `tumor_density_step_000000.png`
- `tumor_boundary_contour_tumor_step_000000.png`
- `cell_density_total_step_000000.png`
- `cell_density_tumor_step_000000.png`

## Example Usage

See `example_plotting_usage.py` for a complete example demonstrating all features.

## Migration from Old Code

If you were using the old plotting methods in `FieldManager`, simply replace:

```python
# Old way
field_manager.plot_pressure_field(step, z_slice)

# New way
plotter = PhysicsFieldsPlotter(field_manager)
plotter.plot_pressure_field(step, z_slice)
```

The `FieldManager.plot_physics_fields()` method still works and now uses the new `PhysicsFieldsPlotter` internally.

## Default Behavior

By default, all plotting methods:
- **Show plots interactively** (`show_plot=True`)
- **Do not save plots** (`save_plot=False`, `output_dir=None`)
- **Do not add contours** (`add_contours=False`)

To save plots, explicitly specify:
```python
plotter.plot_velocity_field(
    step=0,
    output_dir="my_plots",  # Specify directory
    save_plot=True          # Explicitly request saving
)
```

To add tumor boundary contours:
```python
plotter.plot_velocity_field(
    step=0,
    add_contours=True,      # Add contours
    contour_levels=[0.1, 0.5],
    contour_colors=['red', 'white']
)
```
