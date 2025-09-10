#!/usr/bin/env python3
"""
Debug script to identify boundary condition artifacts causing circular growth.

This script will analyze the gradient and Laplacian operators to find
what's causing the circular artifacts.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from src.growkit.MathEngine.Operators import gradient, laplacian, _gradient_neumann

def debug_boundary_artifacts():
    """Debug boundary condition artifacts in differential operators."""
    
    print("Debugging Boundary Condition Artifacts")
    print("=" * 50)
    
    # Create a test field - a simple sphere
    nx, ny, nz = 50, 50, 50
    dx = 1.0
    
    # Create a spherical field
    field = np.zeros((nx, ny, nz), dtype=np.float32)
    center = np.array([nx//2, ny//2, nz//2])
    radius = 10.0
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                dist = np.sqrt((i - center[0])**2 + (j - center[1])**2 + (k - center[2])**2)
                if dist <= radius:
                    field[i, j, k] = 1.0
                else:
                    field[i, j, k] = 0.0
    
    print(f"Test field created:")
    print(f"  Shape: {field.shape}")
    print(f"  Center: {center}")
    print(f"  Radius: {radius}")
    print(f"  Max value: {np.max(field)}")
    print(f"  Min value: {np.min(field)}")
    
    # Test gradient computation
    print(f"\nTesting gradient computation...")
    grad_x, grad_y, grad_z = gradient(field, dx)
    
    print(f"  Gradient X:")
    print(f"    Max: {np.max(grad_x):.6f}")
    print(f"    Min: {np.min(grad_x):.6f}")
    print(f"    Mean: {np.mean(grad_x):.6f}")
    print(f"    Std: {np.std(grad_x):.6f}")
    
    # Check boundary values
    print(f"  Boundary analysis:")
    print(f"    X=0 boundary: {np.mean(grad_x[0, :, :]):.6f}")
    print(f"    X=nx boundary: {np.mean(grad_x[-1, :, :]):.6f}")
    print(f"    Y=0 boundary: {np.mean(grad_y[:, 0, :]):.6f}")
    print(f"    Y=ny boundary: {np.mean(grad_y[:, -1, :]):.6f}")
    print(f"    Z=0 boundary: {np.mean(grad_z[:, :, 0]):.6f}")
    print(f"    Z=nz boundary: {np.mean(grad_z[:, :, -1]):.6f}")
    
    # Test Laplacian computation
    print(f"\nTesting Laplacian computation...")
    lap = laplacian(field, dx)
    
    print(f"  Laplacian:")
    print(f"    Max: {np.max(lap):.6f}")
    print(f"    Min: {np.min(lap):.6f}")
    print(f"    Mean: {np.mean(lap):.6f}")
    print(f"    Std: {np.std(lap):.6f}")
    
    # Check boundary values
    print(f"  Boundary analysis:")
    print(f"    X=0 boundary: {np.mean(lap[0, :, :]):.6f}")
    print(f"    X=nx boundary: {np.mean(lap[-1, :, :]):.6f}")
    print(f"    Y=0 boundary: {np.mean(lap[:, 0, :]):.6f}")
    print(f"    Y=ny boundary: {np.mean(lap[:, -1, :]):.6f}")
    print(f"    Z=0 boundary: {np.mean(lap[:, :, 0]):.6f}")
    print(f"    Z=nz boundary: {np.mean(lap[:, :, -1]):.6f}")
    
    # Check for circular artifacts
    print(f"\nChecking for circular artifacts...")
    
    # Look for patterns in the Laplacian
    # The Laplacian should be negative inside the sphere (concave) and positive outside (convex)
    interior_mask = field > 0.5
    exterior_mask = field < 0.5
    
    interior_laplacian = lap[interior_mask]
    exterior_laplacian = lap[exterior_mask]
    
    print(f"  Interior Laplacian (should be negative):")
    print(f"    Mean: {np.mean(interior_laplacian):.6f}")
    print(f"    Std: {np.std(interior_laplacian):.6f}")
    
    print(f"  Exterior Laplacian (should be positive):")
    print(f"    Mean: {np.mean(exterior_laplacian):.6f}")
    print(f"    Std: {np.std(exterior_laplacian):.6f}")
    
    # Check for boundary artifacts
    boundary_thickness = 3
    boundary_mask = np.zeros_like(field, dtype=bool)
    boundary_mask[:boundary_thickness, :, :] = True
    boundary_mask[-boundary_thickness:, :, :] = True
    boundary_mask[:, :boundary_thickness, :] = True
    boundary_mask[:, -boundary_thickness:, :] = True
    boundary_mask[:, :, :boundary_thickness] = True
    boundary_mask[:, :, -boundary_thickness:] = True
    
    boundary_laplacian = lap[boundary_mask]
    interior_laplacian_clean = lap[~boundary_mask & interior_mask]
    
    print(f"  Boundary Laplacian:")
    print(f"    Mean: {np.mean(boundary_laplacian):.6f}")
    print(f"    Std: {np.std(boundary_laplacian):.6f}")
    
    print(f"  Interior Laplacian (excluding boundaries):")
    print(f"    Mean: {np.mean(interior_laplacian_clean):.6f}")
    print(f"    Std: {np.std(interior_laplacian_clean):.6f}")
    
    # Check if there are significant differences
    boundary_effect = np.mean(boundary_laplacian) - np.mean(interior_laplacian_clean)
    print(f"  Boundary effect: {boundary_effect:.6f}")
    
    if abs(boundary_effect) > 0.01:
        print(f"  ⚠️  WARNING: Significant boundary artifacts detected!")
        print(f"     This could cause circular growth patterns.")
    else:
        print(f"  ✅ No significant boundary artifacts detected.")
    
    return True

if __name__ == "__main__":
    print("Boundary Condition Artifacts Debug")
    print("=" * 50)
    
    success = debug_boundary_artifacts()
    
    if success:
        print("\n✅ Debug completed successfully!")
    else:
        print("\n❌ Debug failed.")
