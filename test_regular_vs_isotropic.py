#!/usr/bin/env python3
"""
Test using regular laplacian instead of isotropic laplacian in the simulation.

Since the regular laplacian showed better directional properties, let's test
if switching to it would reduce diamond-shaped artifacts.
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

def test_regular_vs_isotropic_comprehensive():
    """Comprehensive test of regular vs isotropic laplacian."""
    print("Comprehensive Regular vs Isotropic Laplacian Test")
    print("=" * 60)
    
    nx, ny, nz = 50, 50, 50
    dx = 1.0
    center = np.array([nx//2, ny//2, nz//2])
    radius = 10.0
    
    field = create_test_sphere(nx, ny, nz, center, radius)
    
    # Test regular laplacian
    lap_regular = laplacian(field, dx)
    
    # Test isotropic laplacian
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
    
    print(f"\nIsotropic Laplacian:")
    print(f"  Boundary effect: {boundary_effect_isotropic:.6f}")
    print(f"  Interior mean: {np.mean(interior_laplacian_isotropic):.6f}")
    print(f"  Boundary mean: {np.mean(boundary_laplacian_isotropic):.6f}")
    print(f"  Interior std: {np.std(interior_laplacian_isotropic):.6f}")
    print(f"  Boundary std: {np.std(boundary_laplacian_isotropic):.6f}")
    
    # Test with Gaussian smoothing (as used in the energy module)
    print(f"\nWith Gaussian Smoothing (as used in simulation):")
    from scipy.ndimage import gaussian_filter
    
    sigma = 1.2  # Strong smoothing as used for strong adhesion
    field_smooth = gaussian_filter(field, sigma=sigma)
    
    lap_regular_smooth = laplacian(field_smooth, dx)
    lap_isotropic_smooth = isotropic_laplacian(field_smooth, dx)
    
    # Analyze smoothed results
    boundary_laplacian_regular_smooth = lap_regular_smooth[boundary_mask]
    interior_laplacian_regular_smooth = lap_regular_smooth[interior_clean]
    boundary_effect_regular_smooth = np.mean(boundary_laplacian_regular_smooth) - np.mean(interior_laplacian_regular_smooth)
    
    boundary_laplacian_isotropic_smooth = lap_isotropic_smooth[boundary_mask]
    interior_laplacian_isotropic_smooth = lap_isotropic_smooth[interior_clean]
    boundary_effect_isotropic_smooth = np.mean(boundary_laplacian_isotropic_smooth) - np.mean(interior_laplacian_isotropic_smooth)
    
    print(f"  Regular Laplacian (smoothed):")
    print(f"    Boundary effect: {boundary_effect_regular_smooth:.6f}")
    print(f"    Interior mean: {np.mean(interior_laplacian_regular_smooth):.6f}")
    print(f"    Boundary mean: {np.mean(boundary_laplacian_regular_smooth):.6f}")
    
    print(f"  Isotropic Laplacian (smoothed):")
    print(f"    Boundary effect: {boundary_effect_isotropic_smooth:.6f}")
    print(f"    Interior mean: {np.mean(interior_laplacian_isotropic_smooth):.6f}")
    print(f"    Boundary mean: {np.mean(boundary_laplacian_isotropic_smooth):.6f}")
    
    # Check directional bias for isotropic laplacian
    print(f"\nDirectional Bias Analysis (Isotropic Laplacian):")
    gix, giy, giz = isotropic_gradient_components(field, dx)
    
    grad_magnitude = np.sqrt(gix**2 + giy**2 + giz**2)
    grad_x_ratio = np.abs(gix) / (grad_magnitude + 1e-10)
    grad_y_ratio = np.abs(giy) / (grad_magnitude + 1e-10)
    grad_z_ratio = np.abs(giz) / (grad_magnitude + 1e-10)
    
    interior_grad_x_ratio = grad_x_ratio[interior_clean]
    interior_grad_y_ratio = grad_y_ratio[interior_clean]
    interior_grad_z_ratio = grad_z_ratio[interior_clean]
    
    expected_ratio = 1.0 / np.sqrt(3)  # ≈ 0.577
    
    print(f"  Expected ratio for isotropic field: {expected_ratio:.4f}")
    print(f"  X-ratio: {np.mean(interior_grad_x_ratio):.4f} ± {np.std(interior_grad_x_ratio):.4f}")
    print(f"  Y-ratio: {np.mean(interior_grad_y_ratio):.4f} ± {np.std(interior_grad_y_ratio):.4f}")
    print(f"  Z-ratio: {np.mean(interior_grad_z_ratio):.4f} ± {np.std(interior_grad_z_ratio):.4f}")
    
    x_bias = abs(np.mean(interior_grad_x_ratio) - expected_ratio)
    y_bias = abs(np.mean(interior_grad_y_ratio) - expected_ratio)
    z_bias = abs(np.mean(interior_grad_z_ratio) - expected_ratio)
    max_bias = max(x_bias, y_bias, z_bias)
    
    print(f"  Directional biases:")
    print(f"    X-bias: {x_bias:.4f}")
    print(f"    Y-bias: {y_bias:.4f}")
    print(f"    Z-bias: {z_bias:.4f}")
    print(f"    Max bias: {max_bias:.4f}")
    
    # Recommendations
    print(f"\nRecommendations:")
    
    if abs(boundary_effect_regular_smooth) < abs(boundary_effect_isotropic_smooth):
        print(f"  ✅ Regular laplacian has better boundary behavior when smoothed.")
        print(f"     Consider switching from isotropic to regular laplacian.")
    else:
        print(f"  ⚠️  Isotropic laplacian still has better boundary behavior.")
    
    if max_bias > 0.1:
        print(f"  ⚠️  Significant directional bias detected in isotropic laplacian.")
        print(f"     This is likely the main cause of diamond-shaped tumors.")
        print(f"     Consider using regular laplacian or fixing the isotropic implementation.")
    else:
        print(f"  ✅ No significant directional bias detected.")
    
    # Overall assessment
    print(f"\nOverall Assessment:")
    print(f"  Current simulation uses isotropic laplacian with:")
    print(f"    - Boundary effect: {boundary_effect_isotropic_smooth:.6f}")
    print(f"    - Directional bias: {max_bias:.4f}")
    
    print(f"  Regular laplacian alternative would have:")
    print(f"    - Boundary effect: {boundary_effect_regular_smooth:.6f}")
    print(f"    - Directional bias: 0.0000 (by definition)")
    
    improvement_potential = abs(boundary_effect_isotropic_smooth) - abs(boundary_effect_regular_smooth)
    print(f"  Potential improvement: {improvement_potential:.6f}")
    
    if improvement_potential > 0.01 or max_bias > 0.1:
        print(f"  🔍 RECOMMENDATION: Switch to regular laplacian to reduce diamond artifacts.")
    else:
        print(f"  ➖ Current implementation is acceptable.")
    
    return boundary_effect_regular_smooth, boundary_effect_isotropic_smooth, max_bias

if __name__ == "__main__":
    print("Regular vs Isotropic Laplacian Comprehensive Test")
    print("=" * 60)
    
    regular_effect, isotropic_effect, max_bias = test_regular_vs_isotropic_comprehensive()
    
    print(f"\n✅ Comprehensive test completed!")
    print(f"   Regular laplacian boundary effect: {regular_effect:.6f}")
    print(f"   Isotropic laplacian boundary effect: {isotropic_effect:.6f}")
    print(f"   Isotropic laplacian directional bias: {max_bias:.4f}")
