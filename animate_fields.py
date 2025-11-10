#!/usr/bin/env python3
"""
Animation Script for Simulation Fields

This script provides easy access to all field animation methods.
Simply uncomment the animation method you want to use and run the script.

Usage:
    python animate_fields.py
"""

import sys
from pathlib import Path

# Set up project path
if sys.platform == "darwin":
    proj = "/Users/rileymcnamara/CODE/2025/silicokit/"
    sys.path.insert(0, proj)
else:
    proj = "C:/Users/riley.mcnamara/Documents/code/silicokit/"
    sys.path.insert(0, proj)

from src.growkit.Simulator import TumorGrowthSimulator
from src.growkit.PlotEngine.FieldAnimator import FieldAnimator

# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to simulation data file
SIMULATION_DATA_PATH = proj + "/laboratory/saved_simulations/simulation_data.npz"

# Output directory for saved animations
OUTPUT_DIR = proj + "/laboratory/simulation_analysis/animations"

# Default animation parameters
Z_SLICE = None  # None = center slice, or specify integer (e.g., 25)
INTERVAL = 200  # Animation interval in milliseconds
FPS = 5  # Frames per second for saved animations
REPEAT = True  # Whether to repeat the animation

# ============================================================================
# LOAD SIMULATION DATA
# ============================================================================

print("Loading simulation data...")
simulator = TumorGrowthSimulator(proj + "/configs/T_N.yaml")
simulation_data = simulator.load_simulation_data(SIMULATION_DATA_PATH)

# Create animator
animator = FieldAnimator(simulation_data)

print(f"Loaded {len(simulation_data['metadata']['saved_steps'])} time steps")
print(f"Grid size: {simulation_data['metadata']['grid_size']}")
print(f"Number of populations: {simulation_data['metadata']['num_populations']}")

# ============================================================================
# ANIMATION METHODS
# Uncomment the method(s) you want to use
# ============================================================================

# ----------------------------------------------------------------------------
# 1. ANIMATE TUMOR DENSITY
# ----------------------------------------------------------------------------
# animator.animate_tumor_density(
#     output_dir=OUTPUT_DIR,
#     save_animation=True,
#     show_animation=True,
#     figsize=(10, 10),
#     z_slice=Z_SLICE,
#     threshold=0.1,
#     cmap='viridis',
#     interval=INTERVAL,
#     repeat=REPEAT,
#     add_contour=False,
#     contour_color='red',
#     fps=FPS
# )

# ----------------------------------------------------------------------------
# 2. ANIMATE NUTRIENT FIELD
# ----------------------------------------------------------------------------
# animator.animate_nutrient_field(
#    output_dir=OUTPUT_DIR,
#    save_animation=True,
#    show_animation=True,
#    figsize=(10, 10),
#    z_slice=Z_SLICE,
#    cmap="viridis",
#    interval=INTERVAL,
#    repeat=REPEAT,
#    add_tumor_contours=False,
#    tumor_threshold=0.1,
#    fps=FPS
#)

# ----------------------------------------------------------------------------
# 3. ANIMATE SOURCE FIELD
# ----------------------------------------------------------------------------
animator.animate_source_field(
    output_dir=OUTPUT_DIR,
    save_animation=True,
    show_animation=True,
    figsize=(10, 10),
    z_slice=Z_SLICE,
    cmap="RdBu_r",
    population_idx=0,  # Index of population to animate (0 = first population)
    interval=INTERVAL,
    repeat=REPEAT,
    add_tumor_contours=False,
    tumor_threshold=0.1,
    fps=FPS
)

# ----------------------------------------------------------------------------
# 4. ANIMATE MULTIPLE FIELDS SIDE-BY-SIDE
# ----------------------------------------------------------------------------
# Options for fields_to_animate: ['density', 'nutrient', 'source']
# You can combine any of these, e.g., ['density', 'nutrient'] or ['density', 'source']
# animator.animate_multiple_fields(
#     fields_to_animate=['density', 'nutrient'],  # Options: 'density', 'nutrient', 'source'
#     output_dir=OUTPUT_DIR,
#     save_animation=True,
#     show_animation=True,
#     figsize=(15, 7),
#     z_slice=Z_SLICE,
#     interval=INTERVAL,
#     repeat=REPEAT,
#     density_threshold=0.1,
#     tumor_threshold=0.1,
#     fps=FPS
# )

# ----------------------------------------------------------------------------
# 5. ANIMATE DENSITY AND SOURCE TOGETHER
# ----------------------------------------------------------------------------
# animator.animate_multiple_fields(
#     fields_to_animate=['density', 'source'],
#     output_dir=OUTPUT_DIR,
#     save_animation=True,
#     show_animation=True,
#     figsize=(15, 7),
#     z_slice=Z_SLICE,
#     interval=INTERVAL,
#     repeat=REPEAT,
#     density_threshold=0.1,
#     tumor_threshold=0.1,
#     fps=FPS
# )

# ----------------------------------------------------------------------------
# 6. ANIMATE ALL THREE FIELDS TOGETHER
# ----------------------------------------------------------------------------
# animator.animate_multiple_fields(
#     fields_to_animate=['density', 'nutrient', 'source'],
#     output_dir=OUTPUT_DIR,
#     save_animation=True,
#     show_animation=True,
#     figsize=(20, 7),
#     z_slice=Z_SLICE,
#     interval=INTERVAL,
#     repeat=REPEAT,
#     density_threshold=0.1,
#     tumor_threshold=0.1,
#     fps=FPS
# )

# ============================================================================
# CUSTOM EXAMPLES
# ============================================================================

# ----------------------------------------------------------------------------
# Example: Animate density with custom parameters
# ----------------------------------------------------------------------------
# animator.animate_tumor_density(
#     output_dir=OUTPUT_DIR,
#     save_animation=True,
#     show_animation=False,  # Don't show, just save
#     figsize=(12, 12),
#     z_slice=25,  # Specific z-slice
#     threshold=0.2,  # Higher threshold
#     cmap='plasma',
#     interval=100,  # Faster animation
#     repeat=True,
#     add_contour=True,
#     contour_color='white',
#     fps=10
# )

# ----------------------------------------------------------------------------
# Example: Animate nutrient field without tumor contours
# ----------------------------------------------------------------------------
# animator.animate_nutrient_field(
#     output_dir=OUTPUT_DIR,
#     save_animation=True,
#     show_animation=True,
#     figsize=(10, 10),
#     z_slice=Z_SLICE,
#     cmap="coolwarm",
#     interval=INTERVAL,
#     repeat=REPEAT,
#     add_tumor_contours=False,  # No tumor boundary
#     fps=FPS
# )

# ----------------------------------------------------------------------------
# Example: Animate source field for different population
# ----------------------------------------------------------------------------
# animator.animate_source_field(
#     output_dir=OUTPUT_DIR,
#     save_animation=True,
#     show_animation=True,
#     figsize=(10, 10),
#     z_slice=Z_SLICE,
#     cmap="RdBu_r",
#     population_idx=1,  # Second population (e.g., necrotic)
#     interval=INTERVAL,
#     repeat=REPEAT,
#     add_tumor_contours=True,
#     tumor_threshold=0.1,
#     fps=FPS
# )

# ============================================================================
# NOTES
# ============================================================================
"""
Animation Tips:

1. Z_SLICE: 
   - None = center slice (default)
   - Integer = specific z-slice (e.g., 25)
   - Check grid_size to see valid range

2. INTERVAL:
   - Lower = faster animation (e.g., 100ms)
   - Higher = slower animation (e.g., 500ms)
   - Default: 200ms

3. FPS:
   - Frames per second for saved GIF files
   - Higher = smoother but larger file size
   - Default: 5 fps

4. SAVE vs SHOW:
   - save_animation=True: Save as GIF file
   - show_animation=True: Display in window
   - You can do both or just one

5. MULTIPLE FIELDS:
   - Use animate_multiple_fields() to see multiple fields side-by-side
   - Available fields: 'density', 'nutrient', 'source'
   - Adjust figsize based on number of fields

6. CONTOURS:
   - add_contour/add_tumor_contours: Show tumor boundary
   - threshold/tumor_threshold: Density threshold for boundary
   - contour_color: Color of contour line

7. COLORMAPS:
   - Density: 'viridis', 'plasma', 'inferno', 'magma'
   - Nutrient: 'viridis', 'coolwarm', 'RdYlGn'
   - Source: 'RdBu_r' (good for positive/negative values)

8. PERFORMANCE:
   - Saving animations can take time for large datasets
   - Consider reducing number of time steps if needed
   - Use show_animation=False if you only want to save
"""

print("\n" + "="*70)
print("Animation script loaded successfully!")
print("="*70)
print("\nTo create animations, uncomment the desired method(s) above.")
print("All methods are commented out by default.")
print("\nSaved animations will be written to:")
print(f"  {OUTPUT_DIR}")
print("\n" + "="*70)

