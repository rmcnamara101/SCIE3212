#!/usr/bin/env python3
"""
Comprehensive boundary condition testing script for diamond-shaped tumor investigation.

This script tests different boundary condition approaches to identify which ones
cause the least artifacts and produce the most natural tumor growth patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from src.growkit.MathEngine.Operators import gradient, laplacian, _gradient_neumann

def create_test_sphere(nx, ny, nz, center, radius):
    """Create a test spherical field."""
    field = np.zeros((nx, ny, nz), dtype=np.float32)
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                dist = np.sqrt((i - center[0])**2 + (j - center[1])**2 + (k - center[2])**2)
                if dist <= radius:
                    field[i, j, k] = 1.0
                else:
                    field[i, j, k] = 0.0
    
    return field

def test_neumann_boundary_conditions():
    """Test current Neumann boundary conditions."""
    print("Testing Neumann Boundary Conditions")
    print("=" * 50)
    
    nx, ny, nz = 50, 50, 50
    dx = 1.0
    center = np.array([nx//2, ny//2, nz//2])
    radius = 10.0
    
    field = create_test_sphere(nx, ny, nz, center, radius)
    
    # Test gradient
    grad_x, grad_y, grad_z = gradient(field, dx)
    
    # Test laplacian
    lap = laplacian(field, dx)
    
    # Analyze boundary effects
    boundary_thickness = 3
    boundary_mask = np.zeros_like(field, dtype=bool)
    boundary_mask[:boundary_thickness, :, :] = True
    boundary_mask[-boundary_thickness:, :, :] = True
    boundary_mask[:, :boundary_thickness, :] = True
    boundary_mask[:, -boundary_thickness:, :] = True
    boundary_mask[:, :, :boundary_thickness] = True
    boundary_mask[:, :, -boundary_thickness:] = True
    
    interior_mask = field > 0.5
    interior_clean = interior_mask & ~boundary_mask
    
    boundary_laplacian = lap[boundary_mask]
    interior_laplacian = lap[interior_clean]
    
    boundary_effect = np.mean(boundary_laplacian) - np.mean(interior_laplacian)
    
    print(f"  Boundary effect: {boundary_effect:.6f}")
    print(f"  Interior Laplacian mean: {np.mean(interior_laplacian):.6f}")
    print(f"  Boundary Laplacian mean: {np.mean(boundary_laplacian):.6f}")
    
    return boundary_effect, lap, field

def test_dirichlet_boundary_conditions():
    """Test Dirichlet boundary conditions (zero at boundaries)."""
    print("\nTesting Dirichlet Boundary Conditions")
    print("=" * 50)
    
    nx, ny, nz = 50, 50, 50
    dx = 1.0
    center = np.array([nx//2, ny//2, nz//2])
    radius = 10.0
    
    field = create_test_sphere(nx, ny, nz, center, radius)
    
    # Apply Dirichlet BCs (zero at boundaries)
    field_bc = field.copy()
    field_bc[0, :, :] = 0.0
    field_bc[-1, :, :] = 0.0
    field_bc[:, 0, :] = 0.0
    field_bc[:, -1, :] = 0.0
    field_bc[:, :, 0] = 0.0
    field_bc[:, :, -1] = 0.0
    
    # Test gradient with Dirichlet BCs
    grad_x, grad_y, grad_z = gradient(field_bc, dx)
    
    # Test laplacian with Dirichlet BCs
    lap = laplacian(field_bc, dx)
    
    # Analyze boundary effects
    boundary_thickness = 3
    boundary_mask = np.zeros_like(field, dtype=bool)
    boundary_mask[:boundary_thickness, :, :] = True
    boundary_mask[-boundary_thickness:, :, :] = True
    boundary_mask[:, :boundary_thickness, :] = True
    boundary_mask[:, -boundary_thickness:, :] = True
    boundary_mask[:, :, :boundary_thickness] = True
    boundary_mask[:, :, -boundary_thickness:] = True
    
    interior_mask = field > 0.5
    interior_clean = interior_mask & ~boundary_mask
    
    boundary_laplacian = lap[boundary_mask]
    interior_laplacian = lap[interior_clean]
    
    boundary_effect = np.mean(boundary_laplacian) - np.mean(interior_laplacian)
    
    print(f"  Boundary effect: {boundary_effect:.6f}")
    print(f"  Interior Laplacian mean: {np.mean(interior_laplacian):.6f}")
    print(f"  Boundary Laplacian mean: {np.mean(boundary_laplacian):.6f}")
    
    return boundary_effect, lap, field_bc

def test_periodic_boundary_conditions():
    """Test periodic boundary conditions."""
    print("\nTesting Periodic Boundary Conditions")
    print("=" * 50)
    
    nx, ny, nz = 50, 50, 50
    dx = 1.0
    center = np.array([nx//2, ny//2, nz//2])
    radius = 10.0
    
    field = create_test_sphere(nx, ny, nz, center, radius)
    
    # Apply periodic BCs by extending the field
    field_periodic = np.pad(field, 1, mode='wrap')
    
    # Test gradient with periodic BCs
    grad_x, grad_y, grad_z = gradient(field_periodic, dx)
    
    # Test laplacian with periodic BCs
    lap = laplacian(field_periodic, dx)
    
    # Remove padding
    field_periodic = field_periodic[1:-1, 1:-1, 1:-1]
    lap = lap[1:-1, 1:-1, 1:-1]
    
    # Analyze boundary effects
    boundary_thickness = 3
    boundary_mask = np.zeros_like(field, dtype=bool)
    boundary_mask[:boundary_thickness, :, :] = True
    boundary_mask[-boundary_thickness:, :, :] = True
    boundary_mask[:, :boundary_thickness, :] = True
    boundary_mask[:, -boundary_thickness:, :] = True
    boundary_mask[:, :, :boundary_thickness] = True
    boundary_mask[:, :, -boundary_thickness:] = True
    
    interior_mask = field > 0.5
    interior_clean = interior_mask & ~boundary_mask
    
    boundary_laplacian = lap[boundary_mask]
    interior_laplacian = lap[interior_clean]
    
    boundary_effect = np.mean(boundary_laplacian) - np.mean(interior_laplacian)
    
    print(f"  Boundary effect: {boundary_effect:.6f}")
    print(f"  Interior Laplacian mean: {np.mean(interior_laplacian):.6f}")
    print(f"  Boundary Laplacian mean: {np.mean(boundary_laplacian):.6f}")
    
    return boundary_effect, lap, field_periodic

def test_smooth_boundary_conditions():
    """Test smooth boundary conditions with gradual transition."""
    print("\nTesting Smooth Boundary Conditions")
    print("=" * 50)
    
    nx, ny, nz = 50, 50, 50
    dx = 1.0
    center = np.array([nx//2, ny//2, nz//2])
    radius = 10.0
    
    field = create_test_sphere(nx, ny, nz, center, radius)
    
    # Apply smooth BCs with gradual transition to zero
    field_smooth = field.copy()
    transition_width = 5
    
    # X boundaries
    for i in range(transition_width):
        weight = (transition_width - i) / transition_width
        field_smooth[i, :, :] *= weight
        field_smooth[-(i+1), :, :] *= weight
    
    # Y boundaries
    for j in range(transition_width):
        weight = (transition_width - j) / transition_width
        field_smooth[:, j, :] *= weight
        field_smooth[:, -(j+1), :] *= weight
    
    # Z boundaries
    for k in range(transition_width):
        weight = (transition_width - k) / transition_width
        field_smooth[:, :, k] *= weight
        field_smooth[:, :, -(k+1)] *= weight
    
    # Test gradient with smooth BCs
    grad_x, grad_y, grad_z = gradient(field_smooth, dx)
    
    # Test laplacian with smooth BCs
    lap = laplacian(field_smooth, dx)
    
    # Analyze boundary effects
    boundary_thickness = 3
    boundary_mask = np.zeros_like(field, dtype=bool)
    boundary_mask[:boundary_thickness, :, :] = True
    boundary_mask[-boundary_thickness:, :, :] = True
    boundary_mask[:, :boundary_thickness, :] = True
    boundary_mask[:, -boundary_thickness:, :] = True
    boundary_mask[:, :, :boundary_thickness] = True
    boundary_mask[:, :, -boundary_thickness:] = True
    
    interior_mask = field > 0.5
    interior_clean = interior_mask & ~boundary_mask
    
    boundary_laplacian = lap[boundary_mask]
    interior_laplacian = lap[interior_clean]
    
    boundary_effect = np.mean(boundary_laplacian) - np.mean(interior_laplacian)
    
    print(f"  Boundary effect: {boundary_effect:.6f}")
    print(f"  Interior Laplacian mean: {np.mean(interior_laplacian):.6f}")
    print(f"  Boundary Laplacian mean: {np.mean(boundary_laplacian):.6f}")
    
    return boundary_effect, lap, field_smooth

def analyze_boundary_condition_impact():
    """Analyze the impact of different boundary conditions on tumor growth patterns."""
    print("Boundary Condition Impact Analysis")
    print("=" * 60)
    
    # Test all boundary condition types
    neumann_effect, neumann_lap, neumann_field = test_neumann_boundary_conditions()
    dirichlet_effect, dirichlet_lap, dirichlet_field = test_dirichlet_boundary_conditions()
    periodic_effect, periodic_lap, periodic_field = test_periodic_boundary_conditions()
    smooth_effect, smooth_lap, smooth_field = test_smooth_boundary_conditions()
    
    # Compare results
    print("\nBoundary Condition Comparison")
    print("=" * 40)
    print(f"Neumann BCs:     {neumann_effect:8.6f} (current implementation)")
    print(f"Dirichlet BCs:   {dirichlet_effect:8.6f}")
    print(f"Periodic BCs:    {periodic_effect:8.6f}")
    print(f"Smooth BCs:      {smooth_effect:8.6f}")
    
    # Find the best boundary condition
    effects = {
        'Neumann': neumann_effect,
        'Dirichlet': dirichlet_effect,
        'Periodic': periodic_effect,
        'Smooth': smooth_effect
    }
    
    best_bc = min(effects, key=effects.get)
    worst_bc = max(effects, key=effects.get)
    
    print(f"\nBest boundary condition: {best_bc} (effect: {effects[best_bc]:.6f})")
    print(f"Worst boundary condition: {worst_bc} (effect: {effects[worst_bc]:.6f})")
    
    # Recommendations
    print(f"\nRecommendations:")
    if abs(neumann_effect) > 0.01:
        print(f"  ⚠️  Current Neumann BCs show significant artifacts ({neumann_effect:.6f})")
        print(f"     This likely contributes to diamond-shaped tumor growth.")
    
    if abs(effects[best_bc]) < 0.01:
        print(f"  ✅ {best_bc} BCs show minimal artifacts and should be considered.")
    else:
        print(f"  ⚠️  All tested BCs show some artifacts. Consider:")
        print(f"     - Increasing domain size to reduce boundary influence")
        print(f"     - Using adaptive mesh refinement near boundaries")
        print(f"     - Implementing more sophisticated boundary treatments")
    
    return effects

if __name__ == "__main__":
    print("Comprehensive Boundary Condition Testing")
    print("=" * 60)
    
    effects = analyze_boundary_condition_impact()
    
    print(f"\n✅ Boundary condition analysis completed!")
    print(f"   This analysis can help identify the source of diamond-shaped tumor artifacts.")
