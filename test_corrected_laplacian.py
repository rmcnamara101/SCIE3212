#!/usr/bin/env python3
"""
Test corrected Laplacian implementation with proper Neumann boundary conditions.

The current implementation uses first-order differences at boundaries, which is incorrect
for Neumann BCs. This script tests a corrected implementation.
"""

import numpy as np
from numba import njit
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

@njit(cache=True, fastmath=True)
def corrected_laplacian_neumann(field: np.ndarray, dx: float) -> np.ndarray:
    """
    Corrected Laplacian with proper Neumann boundary conditions.
    
    For Neumann BCs (zero normal derivative), we use ghost points:
    - At x=0: φ(-1) = φ(1)  (ghost point equals first interior point)
    - At x=nx-1: φ(nx) = φ(nx-2)  (ghost point equals last interior point)
    
    This gives us the proper second-order stencil at boundaries.
    """
    lap = np.zeros_like(field)
    inv_dx2 = 1.0 / (dx * dx)
    nx, ny, nz = field.shape

    # Interior points (standard 7-point stencil)
    lap[1:-1, 1:-1, 1:-1] = (
        (field[2:, 1:-1, 1:-1] - 2.0 * field[1:-1, 1:-1, 1:-1] + field[:-2, 1:-1, 1:-1]) * inv_dx2 +
        (field[1:-1, 2:, 1:-1] - 2.0 * field[1:-1, 1:-1, 1:-1] + field[1:-1, :-2, 1:-1]) * inv_dx2 +
        (field[1:-1, 1:-1, 2:] - 2.0 * field[1:-1, 1:-1, 1:-1] + field[1:-1, 1:-1, :-2]) * inv_dx2
    )

    # X boundaries with proper Neumann BCs
    # x = 0: use ghost point φ(-1) = φ(1)
    lap[0, 1:-1, 1:-1] = (
        (field[1, 1:-1, 1:-1] - 2.0 * field[0, 1:-1, 1:-1] + field[1, 1:-1, 1:-1]) * inv_dx2 +
        (field[0, 2:, 1:-1] - 2.0 * field[0, 1:-1, 1:-1] + field[0, :-2, 1:-1]) * inv_dx2 +
        (field[0, 1:-1, 2:] - 2.0 * field[0, 1:-1, 1:-1] + field[0, 1:-1, :-2]) * inv_dx2
    )
    
    # x = nx-1: use ghost point φ(nx) = φ(nx-2)
    lap[-1, 1:-1, 1:-1] = (
        (field[-2, 1:-1, 1:-1] - 2.0 * field[-1, 1:-1, 1:-1] + field[-2, 1:-1, 1:-1]) * inv_dx2 +
        (field[-1, 2:, 1:-1] - 2.0 * field[-1, 1:-1, 1:-1] + field[-1, :-2, 1:-1]) * inv_dx2 +
        (field[-1, 1:-1, 2:] - 2.0 * field[-1, 1:-1, 1:-1] + field[-1, 1:-1, :-2]) * inv_dx2
    )

    # Y boundaries with proper Neumann BCs
    # y = 0: use ghost point φ(-1) = φ(1)
    lap[1:-1, 0, 1:-1] = (
        (field[2:, 0, 1:-1] - 2.0 * field[1:-1, 0, 1:-1] + field[:-2, 0, 1:-1]) * inv_dx2 +
        (field[1:-1, 1, 1:-1] - 2.0 * field[1:-1, 0, 1:-1] + field[1:-1, 1, 1:-1]) * inv_dx2 +
        (field[1:-1, 0, 2:] - 2.0 * field[1:-1, 0, 1:-1] + field[1:-1, 0, :-2]) * inv_dx2
    )
    
    # y = ny-1: use ghost point φ(ny) = φ(ny-2)
    lap[1:-1, -1, 1:-1] = (
        (field[2:, -1, 1:-1] - 2.0 * field[1:-1, -1, 1:-1] + field[:-2, -1, 1:-1]) * inv_dx2 +
        (field[1:-1, -2, 1:-1] - 2.0 * field[1:-1, -1, 1:-1] + field[1:-1, -2, 1:-1]) * inv_dx2 +
        (field[1:-1, -1, 2:] - 2.0 * field[1:-1, -1, 1:-1] + field[1:-1, -1, :-2]) * inv_dx2
    )

    # Z boundaries with proper Neumann BCs
    # z = 0: use ghost point φ(-1) = φ(1)
    lap[1:-1, 1:-1, 0] = (
        (field[2:, 1:-1, 0] - 2.0 * field[1:-1, 1:-1, 0] + field[:-2, 1:-1, 0]) * inv_dx2 +
        (field[1:-1, 2:, 0] - 2.0 * field[1:-1, 1:-1, 0] + field[1:-1, :-2, 0]) * inv_dx2 +
        (field[1:-1, 1:-1, 1] - 2.0 * field[1:-1, 1:-1, 0] + field[1:-1, 1:-1, 1]) * inv_dx2
    )
    
    # z = nz-1: use ghost point φ(nz) = φ(nz-2)
    lap[1:-1, 1:-1, -1] = (
        (field[2:, 1:-1, -1] - 2.0 * field[1:-1, 1:-1, -1] + field[:-2, 1:-1, -1]) * inv_dx2 +
        (field[1:-1, 2:, -1] - 2.0 * field[1:-1, 1:-1, -1] + field[1:-1, :-2, -1]) * inv_dx2 +
        (field[1:-1, 1:-1, -2] - 2.0 * field[1:-1, 1:-1, -1] + field[1:-1, 1:-1, -2]) * inv_dx2
    )

    # Corner and edge points (simplified treatment)
    # For now, use the same approach but with fewer terms
    for i in [0, -1]:
        for j in [0, -1]:
            for k in [0, -1]:
                if i == 0:
                    ghost_i = 1
                else:
                    ghost_i = -2
                if j == 0:
                    ghost_j = 1
                else:
                    ghost_j = -2
                if k == 0:
                    ghost_k = 1
                else:
                    ghost_k = -2
                
                lap[i, j, k] = (
                    (field[ghost_i, j, k] - 2.0 * field[i, j, k] + field[ghost_i, j, k]) * inv_dx2 +
                    (field[i, ghost_j, k] - 2.0 * field[i, j, k] + field[i, ghost_j, k]) * inv_dx2 +
                    (field[i, j, ghost_k] - 2.0 * field[i, j, k] + field[i, j, ghost_k]) * inv_dx2
                )

    return lap

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

def test_corrected_laplacian():
    """Test the corrected Laplacian implementation."""
    print("Testing Corrected Laplacian Implementation")
    print("=" * 50)
    
    nx, ny, nz = 50, 50, 50
    dx = 1.0
    center = np.array([nx//2, ny//2, nz//2])
    radius = 10.0
    
    field = create_test_sphere(nx, ny, nz, center, radius)
    
    # Test original laplacian
    from src.growkit.MathEngine.Operators import laplacian
    lap_original = laplacian(field, dx)
    
    # Test corrected laplacian
    lap_corrected = corrected_laplacian_neumann(field, dx)
    
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
    
    # Original implementation
    boundary_laplacian_orig = lap_original[boundary_mask]
    interior_laplacian_orig = lap_original[interior_clean]
    boundary_effect_orig = np.mean(boundary_laplacian_orig) - np.mean(interior_laplacian_orig)
    
    # Corrected implementation
    boundary_laplacian_corr = lap_corrected[boundary_mask]
    interior_laplacian_corr = lap_corrected[interior_clean]
    boundary_effect_corr = np.mean(boundary_laplacian_corr) - np.mean(interior_laplacian_corr)
    
    print(f"Original Laplacian:")
    print(f"  Boundary effect: {boundary_effect_orig:.6f}")
    print(f"  Interior mean: {np.mean(interior_laplacian_orig):.6f}")
    print(f"  Boundary mean: {np.mean(boundary_laplacian_orig):.6f}")
    
    print(f"\nCorrected Laplacian:")
    print(f"  Boundary effect: {boundary_effect_corr:.6f}")
    print(f"  Interior mean: {np.mean(interior_laplacian_corr):.6f}")
    print(f"  Boundary mean: {np.mean(boundary_laplacian_corr):.6f}")
    
    improvement = abs(boundary_effect_orig) - abs(boundary_effect_corr)
    print(f"\nImprovement: {improvement:.6f}")
    
    if improvement > 0.01:
        print(f"  ✅ Significant improvement detected!")
        print(f"     The corrected implementation reduces boundary artifacts.")
    else:
        print(f"  ⚠️  No significant improvement detected.")
        print(f"     The issue may be elsewhere in the simulation.")
    
    return boundary_effect_orig, boundary_effect_corr

if __name__ == "__main__":
    print("Corrected Laplacian Boundary Condition Test")
    print("=" * 60)
    
    orig_effect, corr_effect = test_corrected_laplacian()
    
    print(f"\n✅ Laplacian correction test completed!")
    print(f"   Original boundary effect: {orig_effect:.6f}")
    print(f"   Corrected boundary effect: {corr_effect:.6f}")
