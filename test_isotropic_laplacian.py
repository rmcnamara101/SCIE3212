#!/usr/bin/env python3
"""
Test the isotropic laplacian implementation that's actually used in the simulation.

The simulation uses isotropic_laplacian which is ∇·(isotropic ∇φ), not the regular laplacian.
This might be the source of the diamond-shaped artifacts.
"""

import numpy as np
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from src.growkit.MathEngine.Operators import laplacian, isotropic_laplacian, isotropic_gradient_components

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

def test_isotropic_vs_regular_laplacian():
    """Compare isotropic laplacian vs regular laplacian."""
    print("Testing Isotropic vs Regular Laplacian")
    print("=" * 50)
    
    nx, ny, nz = 50, 50, 50
    dx = 1.0
    center = np.array([nx//2, ny//2, nz//2])
    radius = 10.0
    
    field = create_test_sphere(nx, ny, nz, center, radius)
    
    # Test regular laplacian
    lap_regular = laplacian(field, dx)
    
    # Test isotropic laplacian (this is what's actually used in the simulation)
    lap_isotropic = isotropic_laplacian(field, dx)
    
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
    
    # Regular laplacian analysis
    boundary_laplacian_regular = lap_regular[boundary_mask]
    interior_laplacian_regular = lap_regular[interior_clean]
    boundary_effect_regular = np.mean(boundary_laplacian_regular) - np.mean(interior_laplacian_regular)
    
    # Isotropic laplacian analysis
    boundary_laplacian_isotropic = lap_isotropic[boundary_mask]
    interior_laplacian_isotropic = lap_isotropic[interior_clean]
    boundary_effect_isotropic = np.mean(boundary_laplacian_isotropic) - np.mean(interior_laplacian_isotropic)
    
    print(f"Regular Laplacian:")
    print(f"  Boundary effect: {boundary_effect_regular:.6f}")
    print(f"  Interior mean: {np.mean(interior_laplacian_regular):.6f}")
    print(f"  Boundary mean: {np.mean(boundary_laplacian_regular):.6f}")
    print(f"  Interior std: {np.std(interior_laplacian_regular):.6f}")
    print(f"  Boundary std: {np.std(boundary_laplacian_regular):.6f}")
    
    print(f"\nIsotropic Laplacian (used in simulation):")
    print(f"  Boundary effect: {boundary_effect_isotropic:.6f}")
    print(f"  Interior mean: {np.mean(interior_laplacian_isotropic):.6f}")
    print(f"  Boundary mean: {np.mean(boundary_laplacian_isotropic):.6f}")
    print(f"  Interior std: {np.std(interior_laplacian_isotropic):.6f}")
    print(f"  Boundary std: {np.std(boundary_laplacian_isotropic):.6f}")
    
    # Check for differences
    difference = abs(boundary_effect_isotropic) - abs(boundary_effect_regular)
    print(f"\nDifference in boundary effects: {difference:.6f}")
    
    if abs(boundary_effect_isotropic) > abs(boundary_effect_regular):
        print(f"  ⚠️  Isotropic laplacian has WORSE boundary artifacts!")
        print(f"     This could be the source of diamond-shaped tumors.")
    elif abs(boundary_effect_isotropic) < abs(boundary_effect_regular):
        print(f"  ✅ Isotropic laplacian has BETTER boundary behavior.")
    else:
        print(f"  ➖ Both laplacians have similar boundary effects.")
    
    # Check for grid alignment artifacts
    print(f"\nGrid Alignment Analysis:")
    
    # Look at the gradient components to see if there are directional biases
    gix, giy, giz = isotropic_gradient_components(field, dx)
    
    # Check if gradients are biased toward grid directions
    grad_magnitude = np.sqrt(gix**2 + giy**2 + giz**2)
    grad_x_ratio = np.abs(gix) / (grad_magnitude + 1e-10)
    grad_y_ratio = np.abs(giy) / (grad_magnitude + 1e-10)
    grad_z_ratio = np.abs(giz) / (grad_magnitude + 1e-10)
    
    # Check in the interior region
    interior_grad_x_ratio = grad_x_ratio[interior_clean]
    interior_grad_y_ratio = grad_y_ratio[interior_clean]
    interior_grad_z_ratio = grad_z_ratio[interior_clean]
    
    print(f"  Interior gradient direction ratios:")
    print(f"    X-direction: {np.mean(interior_grad_x_ratio):.4f} ± {np.std(interior_grad_x_ratio):.4f}")
    print(f"    Y-direction: {np.mean(interior_grad_y_ratio):.4f} ± {np.std(interior_grad_y_ratio):.4f}")
    print(f"    Z-direction: {np.mean(interior_grad_z_ratio):.4f} ± {np.std(interior_grad_z_ratio):.4f}")
    
    # For a perfect sphere, all directions should be equal (0.577 for 3D)
    expected_ratio = 1.0 / np.sqrt(3)  # ≈ 0.577
    x_bias = abs(np.mean(interior_grad_x_ratio) - expected_ratio)
    y_bias = abs(np.mean(interior_grad_y_ratio) - expected_ratio)
    z_bias = abs(np.mean(interior_grad_z_ratio) - expected_ratio)
    
    print(f"  Expected ratio for isotropic field: {expected_ratio:.4f}")
    print(f"  Directional biases:")
    print(f"    X-bias: {x_bias:.4f}")
    print(f"    Y-bias: {y_bias:.4f}")
    print(f"    Z-bias: {z_bias:.4f}")
    
    max_bias = max(x_bias, y_bias, z_bias)
    if max_bias > 0.1:
        print(f"  ⚠️  Significant directional bias detected!")
        print(f"     This could cause diamond-shaped growth patterns.")
    else:
        print(f"  ✅ No significant directional bias detected.")
    
    return boundary_effect_regular, boundary_effect_isotropic, max_bias

def test_smoothing_effect():
    """Test the effect of Gaussian smoothing on the laplacian."""
    print(f"\nTesting Gaussian Smoothing Effect")
    print("=" * 50)
    
    from scipy.ndimage import gaussian_filter
    
    nx, ny, nz = 50, 50, 50
    dx = 1.0
    center = np.array([nx//2, ny//2, nz//2])
    radius = 10.0
    
    field = create_test_sphere(nx, ny, nz, center, radius)
    
    # Test different smoothing levels (as used in the energy module)
    sigma_values = [0.3, 0.6, 1.2]
    
    for sigma in sigma_values:
        field_smooth = gaussian_filter(field, sigma=sigma)
        lap_isotropic = isotropic_laplacian(field_smooth, dx)
        
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
        
        boundary_laplacian = lap_isotropic[boundary_mask]
        interior_laplacian = lap_isotropic[interior_clean]
        boundary_effect = np.mean(boundary_laplacian) - np.mean(interior_laplacian)
        
        print(f"  Sigma = {sigma}:")
        print(f"    Boundary effect: {boundary_effect:.6f}")
        print(f"    Interior mean: {np.mean(interior_laplacian):.6f}")
        print(f"    Boundary mean: {np.mean(boundary_laplacian):.6f}")

if __name__ == "__main__":
    print("Isotropic Laplacian Boundary Artifact Analysis")
    print("=" * 60)
    
    regular_effect, isotropic_effect, max_bias = test_isotropic_vs_regular_laplacian()
    test_smoothing_effect()
    
    print(f"\n✅ Isotropic laplacian analysis completed!")
    print(f"   Regular laplacian boundary effect: {regular_effect:.6f}")
    print(f"   Isotropic laplacian boundary effect: {isotropic_effect:.6f}")
    print(f"   Maximum directional bias: {max_bias:.4f}")
    
    if abs(isotropic_effect) > 0.01 or max_bias > 0.1:
        print(f"\n🔍 RECOMMENDATION:")
        print(f"   The isotropic laplacian shows significant artifacts.")
        print(f"   Consider switching to regular laplacian or improving")
        print(f"   the isotropic implementation to reduce diamond artifacts.")
