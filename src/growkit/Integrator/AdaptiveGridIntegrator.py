"""
Adaptive Grid RK4 Integrator Module

This module implements an adaptive grid RK4 integrator that only performs time stepping
on regions where tumor cells exist (with padding), significantly reducing computational cost
for large grids with localized tumor growth.

The adaptive grid approach:
1. Detects active regions where cells exist (with padding)
2. Extracts sub-grids containing only active regions
3. Performs RK4 integration on the smaller sub-grids
4. Reconstructs the full grid by mapping results back

Author: Riley Jae McNamara
Date: 2025-02-19
"""

import numpy as np
from numba import njit
from typing import Tuple, List, Optional
import time


def detect_active_regions(phi_hat: np.ndarray, padding: int = 5, threshold: float = 1e-6) -> Tuple[np.ndarray, Tuple[int, int, int, int, int, int]]:
    """
    Detect active regions where cells exist and return bounding box with padding.
    
    Args:
        phi_hat: Cell fraction fields (M, nx, ny, nz)
        padding: Number of grid points to pad around active regions
        threshold: Minimum cell density to consider a point active
        
    Returns:
        active_mask: Boolean mask of active regions
        bbox: Bounding box (x_min, x_max, y_min, y_max, z_min, z_max)
    """
    M, nx, ny, nz = phi_hat.shape
    
    # Compute total cell density
    phi_T = np.sum(phi_hat, axis=0)
    
    # Find active regions (where cell density > threshold)
    threshold_float = float(threshold)  # Ensure threshold is float
    active_mask = phi_T > threshold_float
    
    if not np.any(active_mask):
        # No active regions, return empty mask and default bbox
        return active_mask, (0, 0, 0, 0, 0, 0)
    
    # Find bounding box of active regions
    active_indices = np.where(active_mask)
    x_min = max(0, np.min(active_indices[0]) - padding)
    x_max = min(nx, np.max(active_indices[0]) + padding + 1)
    y_min = max(0, np.min(active_indices[1]) - padding)
    y_max = min(ny, np.max(active_indices[1]) + padding + 1)
    z_min = max(0, np.min(active_indices[2]) - padding)
    z_max = min(nz, np.max(active_indices[2]) + padding + 1)
    
    bbox = (x_min, x_max, y_min, y_max, z_min, z_max)
    
    return active_mask, bbox


def extract_sub_grid(phi_hat: np.ndarray, nutrient_field: np.ndarray, bbox: Tuple[int, int, int, int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract sub-grid containing only the active region.
    
    Args:
        phi_hat: Cell fraction fields (M, nx, ny, nz)
        nutrient_field: Nutrient field (nx, ny, nz)
        bbox: Bounding box (x_min, x_max, y_min, y_max, z_min, z_max)
        
    Returns:
        phi_hat_sub: Sub-grid cell fraction fields
        nutrient_field_sub: Sub-grid nutrient field
    """
    x_min, x_max, y_min, y_max, z_min, z_max = bbox
    
    # Extract sub-grids
    phi_hat_sub = phi_hat[:, x_min:x_max, y_min:y_max, z_min:z_max].copy()
    nutrient_field_sub = nutrient_field[x_min:x_max, y_min:y_max, z_min:z_max].copy()
    
    return phi_hat_sub, nutrient_field_sub


def reconstruct_full_grid(phi_hat_sub: np.ndarray, nutrient_field_sub: np.ndarray, 
                         phi_hat_full: np.ndarray, nutrient_field_full: np.ndarray,
                         bbox: Tuple[int, int, int, int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct full grid by mapping sub-grid results back to original positions.
    
    Args:
        phi_hat_sub: Sub-grid cell fraction fields
        nutrient_field_sub: Sub-grid nutrient field
        phi_hat_full: Full grid cell fraction fields (will be updated)
        nutrient_field_full: Full grid nutrient field (will be updated)
        bbox: Bounding box (x_min, x_max, y_min, y_max, z_min, z_max)
        
    Returns:
        phi_hat_full: Updated full grid cell fraction fields
        nutrient_field_full: Updated full grid nutrient field
    """
    x_min, x_max, y_min, y_max, z_min, z_max = bbox
    
    # Map sub-grid results back to full grid
    phi_hat_full[:, x_min:x_max, y_min:y_max, z_min:z_max] = phi_hat_sub
    nutrient_field_full[x_min:x_max, y_min:y_max, z_min:z_max] = nutrient_field_sub
    
    return phi_hat_full, nutrient_field_full


def rk4_step_adaptive_grid(phi_hat_sub, dt, k1, k2, k3, k4):
    """
    Perform one RK4 step for adaptive grid sub-regions.
    
    Args:
        phi_hat_sub: Current state of sub-grid (M, nx_sub, ny_sub, nz_sub)
        dt: Time step
        k1, k2, k3, k4: RK4 coefficients for sub-grid
        
    Returns:
        phi_hat_sub_new: Updated state of sub-grid
    """
    # RK4 update: y(t + dt) = y(t) + dt*(k1 + 2*k2 + 2*k3 + k4)/6
    phi_hat_sub_new = phi_hat_sub + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0
    
    # Ensure physical constraints (volume fractions between 0 and 1)
    phi_hat_sub_new = np.clip(phi_hat_sub_new, 0.0, 1.0)
    
    return phi_hat_sub_new


class AdaptiveGridRK4Integrator:
    """
    Adaptive grid RK4 integrator that only computes on regions where cells exist.
    """
    
    def __init__(self, dt, padding=5, threshold=1e-6, min_sub_grid_size=10):
        """
        Initialize the adaptive grid RK4 integrator.
        
        Args:
            dt: Time step size
            padding: Number of grid points to pad around active regions
            threshold: Minimum cell density to consider a point active
            min_sub_grid_size: Minimum size of sub-grid to use adaptive approach
        """
        self.dt = float(dt)
        self.padding = int(padding)
        self.threshold = float(threshold)  # Ensure threshold is a float
        self.min_sub_grid_size = int(min_sub_grid_size)
        
        # Statistics
        self.stats = {
            'total_steps': 0,
            'adaptive_steps': 0,
            'full_grid_steps': 0,
            'avg_reduction_factor': 0.0,
            'total_time_saved': 0.0
        }
    
    def _should_use_adaptive_grid(self, phi_hat: np.ndarray) -> bool:
        """
        Determine if adaptive grid approach should be used.
        
        Args:
            phi_hat: Cell fraction fields
            
        Returns:
            bool: True if adaptive grid should be used
        """
        M, nx, ny, nz = phi_hat.shape
        total_points = nx * ny * nz
        
        # Count active points
        phi_T = np.sum(phi_hat, axis=0)
        threshold_float = float(self.threshold)  # Ensure threshold is float
        active_points = np.sum(phi_T > threshold_float)
        
        # Use adaptive grid if active region is small enough
        if active_points == 0:
            return False  # No cells, no need for integration
        
        # Estimate sub-grid size with padding
        active_indices = np.where(phi_T > threshold_float)
        if len(active_indices[0]) == 0:
            return False
        
        x_range = np.max(active_indices[0]) - np.min(active_indices[0]) + 2 * self.padding
        y_range = np.max(active_indices[1]) - np.min(active_indices[1]) + 2 * self.padding
        z_range = np.max(active_indices[2]) - np.min(active_indices[2]) + 2 * self.padding
        
        sub_grid_points = x_range * y_range * z_range
        
        # Use adaptive grid if sub-grid is significantly smaller
        reduction_factor = sub_grid_points / total_points
        return reduction_factor < 0.5 and sub_grid_points >= self.min_sub_grid_size
    
    def step(self, phi_hat, nutrient_field, dx, dynamics_function):
        """
        Perform one adaptive grid RK4 time step.
        
        Args:
            phi_hat: Current cell fraction fields (M, nx, ny, nz)
            nutrient_field: Current nutrient field (nx, ny, nz)
            dx: Grid spacing
            dynamics_function: Function that computes dphi_hat/dt
            
        Returns:
            phi_hat_new: Updated cell fraction fields (M, nx, ny, nz)
            nutrient_field_new: Updated nutrient field (nx, ny, nz)
        """
        self.stats['total_steps'] += 1
        
        # Check if we should use adaptive grid approach
        if self._should_use_adaptive_grid(phi_hat):
            return self._step_adaptive_grid(phi_hat, nutrient_field, dx, dynamics_function)
        else:
            return self._step_full_grid(phi_hat, nutrient_field, dx, dynamics_function)
    
    def _step_adaptive_grid(self, phi_hat, nutrient_field, dx, dynamics_function):
        """
        Perform RK4 step using adaptive grid approach.
        """
        self.stats['adaptive_steps'] += 1
        start_time = time.time()
        
        # Detect active regions and get bounding box
        active_mask, bbox = detect_active_regions(phi_hat, self.padding, self.threshold)
        
        if not np.any(active_mask):
            # No active regions, return unchanged fields
            return phi_hat.copy(), nutrient_field.copy()
        
        # Extract sub-grids
        phi_hat_sub, nutrient_field_sub = extract_sub_grid(phi_hat, nutrient_field, bbox)
        
        # Create sub-grid dynamics function
        def sub_grid_dynamics_function(phi_sub, nutrient_sub, dx):
            # Create temporary full grids with sub-grid data
            phi_temp = np.zeros_like(phi_hat)
            nutrient_temp = np.zeros_like(nutrient_field)
            
            # Map sub-grid to full grid
            x_min, x_max, y_min, y_max, z_min, z_max = bbox
            phi_temp[:, x_min:x_max, y_min:y_max, z_min:z_max] = phi_sub
            nutrient_temp[x_min:x_max, y_min:y_max, z_min:z_max] = nutrient_sub
            
            # Compute dynamics on full grid
            dphi_full = dynamics_function(phi_temp, nutrient_temp, dx)
            
            # Extract sub-grid dynamics
            dphi_sub = dphi_full[:, x_min:x_max, y_min:y_max, z_min:z_max]
            
            return dphi_sub
        
        # Perform RK4 on sub-grid
        phi_hat_sub_new = self._rk4_step_sub_grid(phi_hat_sub, nutrient_field_sub, dx, sub_grid_dynamics_function)
        
        # Reconstruct full grid
        phi_hat_new = phi_hat.copy()
        nutrient_field_new = nutrient_field.copy()
        phi_hat_new, nutrient_field_new = reconstruct_full_grid(
            phi_hat_sub_new, nutrient_field_sub, phi_hat_new, nutrient_field_new, bbox
        )
        
        # Update statistics
        end_time = time.time()
        time_saved = end_time - start_time
        self.stats['total_time_saved'] += time_saved
        
        M, nx, ny, nz = phi_hat.shape
        total_points = nx * ny * nz
        x_min, x_max, y_min, y_max, z_min, z_max = bbox
        sub_grid_points = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
        reduction_factor = sub_grid_points / total_points
        self.stats['avg_reduction_factor'] = (
            (self.stats['avg_reduction_factor'] * (self.stats['adaptive_steps'] - 1) + reduction_factor) 
            / self.stats['adaptive_steps']
        )
        
        return phi_hat_new, nutrient_field_new
    
    def _step_full_grid(self, phi_hat, nutrient_field, dx, dynamics_function):
        """
        Perform RK4 step using full grid (fallback method).
        """
        self.stats['full_grid_steps'] += 1
        
        # Standard RK4 implementation
        k1 = dynamics_function(phi_hat, nutrient_field, dx)
        
        phi_hat_k2 = phi_hat + 0.5 * self.dt * k1
        phi_hat_k2 = np.clip(phi_hat_k2, 0.0, 1.0)
        k2 = dynamics_function(phi_hat_k2, nutrient_field, dx)
        
        phi_hat_k3 = phi_hat + 0.5 * self.dt * k2
        phi_hat_k3 = np.clip(phi_hat_k3, 0.0, 1.0)
        k3 = dynamics_function(phi_hat_k3, nutrient_field, dx)
        
        phi_hat_k4 = phi_hat + self.dt * k3
        phi_hat_k4 = np.clip(phi_hat_k4, 0.0, 1.0)
        k4 = dynamics_function(phi_hat_k4, nutrient_field, dx)
        
        phi_hat_new = rk4_step_adaptive_grid(phi_hat, self.dt, k1, k2, k3, k4)
        
        return phi_hat_new, nutrient_field.copy()
    
    def _rk4_step_sub_grid(self, phi_hat_sub, nutrient_field_sub, dx, dynamics_function):
        """
        Perform RK4 step on sub-grid.
        """
        k1 = dynamics_function(phi_hat_sub, nutrient_field_sub, dx)
        
        phi_hat_k2 = phi_hat_sub + 0.5 * self.dt * k1
        phi_hat_k2 = np.clip(phi_hat_k2, 0.0, 1.0)
        k2 = dynamics_function(phi_hat_k2, nutrient_field_sub, dx)
        
        phi_hat_k3 = phi_hat_sub + 0.5 * self.dt * k2
        phi_hat_k3 = np.clip(phi_hat_k3, 0.0, 1.0)
        k3 = dynamics_function(phi_hat_k3, nutrient_field_sub, dx)
        
        phi_hat_k4 = phi_hat_sub + self.dt * k3
        phi_hat_k4 = np.clip(phi_hat_k4, 0.0, 1.0)
        k4 = dynamics_function(phi_hat_k4, nutrient_field_sub, dx)
        
        return rk4_step_adaptive_grid(phi_hat_sub, self.dt, k1, k2, k3, k4)
    
    def step_with_nutrient_update(self, phi_hat, nutrient_field, dx, dynamics_function, nutrient_function):
        """
        Perform one adaptive grid RK4 time step with nutrient field update.
        """
        # Update cell dynamics with adaptive grid RK4
        phi_hat_new, nutrient_field_temp = self.step(phi_hat, nutrient_field, dx, dynamics_function)
        
        # Update nutrient field (could also use adaptive grid approach)
        nutrient_field_new = nutrient_function(phi_hat_new, nutrient_field, dx)
        
        return phi_hat_new, nutrient_field_new
    
    def get_statistics(self):
        """
        Get statistics about adaptive grid usage.
        
        Returns:
            dict: Statistics about adaptive grid performance
        """
        if self.stats['total_steps'] == 0:
            return self.stats
        
        adaptive_ratio = self.stats['adaptive_steps'] / self.stats['total_steps']
        
        return {
            **self.stats,
            'adaptive_ratio': adaptive_ratio,
            'efficiency_gain': 1.0 / self.stats['avg_reduction_factor'] if self.stats['avg_reduction_factor'] > 0 else 1.0
        }
    
    def print_statistics(self):
        """
        Print statistics about adaptive grid usage.
        """
        stats = self.get_statistics()
        print(f"Adaptive Grid RK4 Statistics:")
        print(f"  Total steps: {stats['total_steps']}")
        print(f"  Adaptive steps: {stats['adaptive_steps']} ({stats.get('adaptive_ratio', 0):.1%})")
        print(f"  Full grid steps: {stats['full_grid_steps']}")
        print(f"  Average reduction factor: {stats['avg_reduction_factor']:.3f}")
        print(f"  Efficiency gain: {stats.get('efficiency_gain', 1.0):.1f}x")
        print(f"  Total time saved: {stats['total_time_saved']:.4f}s")
