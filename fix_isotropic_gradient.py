#!/usr/bin/env python3
"""
Fix for the isotropic gradient components to reduce directional bias.

The current implementation has a complex diagonal blending that may be causing
diamond-shaped artifacts. This script provides a simpler, more isotropic approach.
"""

import numpy as np
from numba import njit
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

@njit(cache=True, fastmath=True)
def improved_isotropic_gradient_components(field: np.ndarray, dx: float):
    """
    Improved isotropic gradient components with reduced directional bias.
    
    This version uses a simpler approach that should be more truly isotropic:
    1. Use standard central differences for interior points
    2. Use proper Neumann boundary conditions
    3. Avoid complex diagonal blending that may introduce bias
    """
    nx, ny, nz = field.shape
    gx = np.zeros_like(field)
    gy = np.zeros_like(field)
    gz = np.zeros_like(field)

    inv_2dx = 1.0 / (2.0 * dx)

    # Interior points: standard central differences
    gx[1:-1, 1:-1, 1:-1] = (field[2:, 1:-1, 1:-1] - field[:-2, 1:-1, 1:-1]) * inv_2dx
    gy[1:-1, 1:-1, 1:-1] = (field[1:-1, 2:, 1:-1] - field[1:-1, :-2, 1:-1]) * inv_2dx
    gz[1:-1, 1:-1, 1:-1] = (field[1:-1, 1:-1, 2:] - field[1:-1, 1:-1, :-2]) * inv_2dx

    # Boundary conditions: Neumann (zero normal derivative)
    # X boundaries
    gx[0, :, :] = 0.0
    gx[-1, :, :] = 0.0
    
    # Y boundaries  
    gy[:, 0, :] = 0.0
    gy[:, -1, :] = 0.0
    
    # Z boundaries
    gz[:, :, 0] = 0.0
    gz[:, :, -1] = 0.0

    return gx, gy, gz

@njit(cache=True, fastmath=True)
def improved_isotropic_laplacian(field: np.ndarray, dx: float) -> np.ndarray:
    """Improved isotropic laplacian using the corrected gradient components."""
    gx, gy, gz = improved_isotropic_gradient_components(field, dx)
    
    # Compute divergence using the same approach as the original
    div = np.zeros_like(field)
    inv_2dx = 1.0 / (2.0 * dx)

    # Interior
    div[1:-1, 1:-1, 1:-1] = (
        (gx[2:, 1:-1, 1:-1] - gx[:-2, 1:-1, 1:-1])
        + (gy[1:-1, 2:, 1:-1] - gy[1:-1, :-2, 1:-1])
        + (gz[1:-1, 1:-1, 2:] - gz[1:-1, 1:-1, :-2])
    ) * inv_2dx

    # X-faces
    div[0, 1:-1, 1:-1] = (
        (gx[1, 1:-1, 1:-1] - gx[0, 1:-1, 1:-1])
        + (gy[0, 2:, 1:-1] - gy[0, :-2, 1:-1])
        + (gz[0, 1:-1, 2:] - gz[0, 1:-1, :-2])
    ) * inv_2dx
    div[-1, 1:-1, 1:-1] = (
        (gx[-1, 1:-1, 1:-1] - gx[-2, 1:-1, 1:-1])
        + (gy[-1, 2:, 1:-1] - gy[-1, :-2, 1:-1])
        + (gz[-1, 1:-1, 2:] - gz[-1, 1:-1, :-2])
    ) * inv_2dx

    # Y-faces
    div[1:-1, 0, 1:-1] = (
        (gx[2:, 0, 1:-1] - gx[:-2, 0, 1:-1])
        + (gy[1:-1, 1, 1:-1] - gy[1:-1, 0, 1:-1])
        + (gz[1:-1, 0, 2:] - gz[1:-1, 0, :-2])
    ) * inv_2dx
    div[1:-1, -1, 1:-1] = (
        (gx[2:, -1, 1:-1] - gx[:-2, -1, 1:-1])
        + (gy[1:-1, -1, 1:-1] - gy[1:-1, -2, 1:-1])
        + (gz[1:-1, -1, 2:] - gz[1:-1, -1, :-2])
    ) * inv_2dx

    # Z-faces
    div[1:-1, 1:-1, 0] = (
        (gx[2:, 1:-1, 0] - gx[:-2, 1:-1, 0])
        + (gy[1:-1, 2:, 0] - gy[1:-1, :-2, 0])
        + (gz[1:-1, 1:-1, 1] - gz[1:-1, 1:-1, 0])
    ) * inv_2dx
    div[1:-1, 1:-1, -1] = (
        (gx[2:, 1:-1, -1] - gx[:-2, 1:-1, -1])
        + (gy[1:-1, 2:, -1] - gy[1:-1, :-2, -1])
        + (gz[1:-1, 1:-1, -1] - gz[1:-1, 1:-1, -2])
    ) * inv_2dx

    return div

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

def test_improved_isotropic_implementation():
    """Test the improved isotropic implementation."""
    print("Testing Improved Isotropic Implementation")
    print("=" * 50)
    
    from src.growkit.MathEngine.Operators import isotropic_laplacian, isotropic_gradient_components
    
    nx, ny, nz = 50, 50, 50
    dx = 1.0
    center = np.array([nx//2, ny//2, nz//2])
    radius = 10.0
    
    field = create_test_sphere(nx, ny, nz, center, radius)
    
    # Test original implementation
    lap_original = isotropic_laplacian(field, dx)
    gix_orig, giy_orig, giz_orig = isotropic_gradient_components(field, dx)
    
    # Test improved implementation
    lap_improved = improved_isotropic_laplacian(field, dx)
    gix_improved, giy_improved, giz_improved = improved_isotropic_gradient_components(field, dx)
    
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
    
    # Original implementation analysis
    boundary_laplacian_orig = lap_original[boundary_mask]
    interior_laplacian_orig = lap_original[interior_clean]
    boundary_effect_orig = np.mean(boundary_laplacian_orig) - np.mean(interior_laplacian_orig)
    
    # Improved implementation analysis
    boundary_laplacian_improved = lap_improved[boundary_mask]
    interior_laplacian_improved = lap_improved[interior_clean]
    boundary_effect_improved = np.mean(boundary_laplacian_improved) - np.mean(interior_laplacian_improved)
    
    print(f"Original Implementation:")
    print(f"  Boundary effect: {boundary_effect_orig:.6f}")
    print(f"  Interior mean: {np.mean(interior_laplacian_orig):.6f}")
    print(f"  Boundary mean: {np.mean(boundary_laplacian_orig):.6f}")
    
    print(f"\nImproved Implementation:")
    print(f"  Boundary effect: {boundary_effect_improved:.6f}")
    print(f"  Interior mean: {np.mean(interior_laplacian_improved):.6f}")
    print(f"  Boundary mean: {np.mean(boundary_laplacian_improved):.6f}")
    
    # Check directional bias
    print(f"\nDirectional Bias Analysis:")
    
    # Original gradients
    grad_magnitude_orig = np.sqrt(gix_orig**2 + giy_orig**2 + giz_orig**2)
    grad_x_ratio_orig = np.abs(gix_orig) / (grad_magnitude_orig + 1e-10)
    grad_y_ratio_orig = np.abs(giy_orig) / (grad_magnitude_orig + 1e-10)
    grad_z_ratio_orig = np.abs(giz_orig) / (grad_magnitude_orig + 1e-10)
    
    interior_grad_x_ratio_orig = grad_x_ratio_orig[interior_clean]
    interior_grad_y_ratio_orig = grad_y_ratio_orig[interior_clean]
    interior_grad_z_ratio_orig = grad_z_ratio_orig[interior_clean]
    
    # Improved gradients
    grad_magnitude_improved = np.sqrt(gix_improved**2 + giy_improved**2 + giz_improved**2)
    grad_x_ratio_improved = np.abs(gix_improved) / (grad_magnitude_improved + 1e-10)
    grad_y_ratio_improved = np.abs(giy_improved) / (grad_magnitude_improved + 1e-10)
    grad_z_ratio_improved = np.abs(giz_improved) / (grad_magnitude_improved + 1e-10)
    
    interior_grad_x_ratio_improved = grad_x_ratio_improved[interior_clean]
    interior_grad_y_ratio_improved = grad_y_ratio_improved[interior_clean]
    interior_grad_z_ratio_improved = grad_z_ratio_improved[interior_clean]
    
    expected_ratio = 1.0 / np.sqrt(3)  # ≈ 0.577
    
    print(f"  Expected ratio for isotropic field: {expected_ratio:.4f}")
    
    print(f"  Original implementation:")
    print(f"    X-ratio: {np.mean(interior_grad_x_ratio_orig):.4f} ± {np.std(interior_grad_x_ratio_orig):.4f}")
    print(f"    Y-ratio: {np.mean(interior_grad_y_ratio_orig):.4f} ± {np.std(interior_grad_y_ratio_orig):.4f}")
    print(f"    Z-ratio: {np.mean(interior_grad_z_ratio_orig):.4f} ± {np.std(interior_grad_z_ratio_orig):.4f}")
    
    print(f"  Improved implementation:")
    print(f"    X-ratio: {np.mean(interior_grad_x_ratio_improved):.4f} ± {np.std(interior_grad_x_ratio_improved):.4f}")
    print(f"    Y-ratio: {np.mean(interior_grad_y_ratio_improved):.4f} ± {np.std(interior_grad_y_ratio_improved):.4f}")
    print(f"    Z-ratio: {np.mean(interior_grad_z_ratio_improved):.4f} ± {np.std(interior_grad_z_ratio_improved):.4f}")
    
    # Calculate biases
    x_bias_orig = abs(np.mean(interior_grad_x_ratio_orig) - expected_ratio)
    y_bias_orig = abs(np.mean(interior_grad_y_ratio_orig) - expected_ratio)
    z_bias_orig = abs(np.mean(interior_grad_z_ratio_orig) - expected_ratio)
    max_bias_orig = max(x_bias_orig, y_bias_orig, z_bias_orig)
    
    x_bias_improved = abs(np.mean(interior_grad_x_ratio_improved) - expected_ratio)
    y_bias_improved = abs(np.mean(interior_grad_y_ratio_improved) - expected_ratio)
    z_bias_improved = abs(np.mean(interior_grad_z_ratio_improved) - expected_ratio)
    max_bias_improved = max(x_bias_improved, y_bias_improved, z_bias_improved)
    
    print(f"\nBias Comparison:")
    print(f"  Original max bias: {max_bias_orig:.4f}")
    print(f"  Improved max bias: {max_bias_improved:.4f}")
    print(f"  Bias improvement: {max_bias_orig - max_bias_improved:.4f}")
    
    # Overall improvement
    boundary_improvement = abs(boundary_effect_orig) - abs(boundary_effect_improved)
    bias_improvement = max_bias_orig - max_bias_improved
    
    print(f"\nOverall Improvement:")
    print(f"  Boundary effect improvement: {boundary_improvement:.6f}")
    print(f"  Directional bias improvement: {bias_improvement:.4f}")
    
    if boundary_improvement > 0.01 or bias_improvement > 0.05:
        print(f"  ✅ Significant improvement detected!")
        print(f"     The improved implementation should reduce diamond artifacts.")
    else:
        print(f"  ⚠️  No significant improvement detected.")
        print(f"     The issue may require a different approach.")
    
    return boundary_effect_orig, boundary_effect_improved, max_bias_orig, max_bias_improved

if __name__ == "__main__":
    print("Improved Isotropic Gradient Implementation Test")
    print("=" * 60)
    
    orig_boundary, improved_boundary, orig_bias, improved_bias = test_improved_isotropic_implementation()
    
    print(f"\n✅ Improved implementation test completed!")
    print(f"   Original boundary effect: {orig_boundary:.6f}")
    print(f"   Improved boundary effect: {improved_boundary:.6f}")
    print(f"   Original max bias: {orig_bias:.4f}")
    print(f"   Improved max bias: {improved_bias:.4f}")
