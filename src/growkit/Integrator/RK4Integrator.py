"""
RK4 Integrator Module

This module implements a 4th-order Runge-Kutta (RK4) integrator for time stepping
the cell dynamics equations. The RK4 method provides high accuracy and stability
for solving the system of differential equations.

The RK4 method computes:
k1 = f(t, y)
k2 = f(t + dt/2, y + dt*k1/2)
k3 = f(t + dt/2, y + dt*k2/2)
k4 = f(t + dt, y + dt*k3)
y(t + dt) = y(t) + dt*(k1 + 2*k2 + 2*k3 + k4)/6

Author: Riley Jae McNamara
Date: 2025-02-19
"""

import numpy as np
from numba import njit


@njit
def rk4_step_vectorized(phi_hat, dt, k1, k2, k3, k4):
    """
    Perform one RK4 step for vectorized fields.
    
    Args:
        phi_hat: Current state (M, nx, ny, nz)
        dt: Time step
        k1, k2, k3, k4: RK4 coefficients (M, nx, ny, nz)
        
    Returns:
        phi_hat_new: Updated state (M, nx, ny, nz)
    """
    # RK4 update: y(t + dt) = y(t) + dt*(k1 + 2*k2 + 2*k3 + k4)/6
    phi_hat_new = phi_hat + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0
    
    # Ensure physical constraints (volume fractions between 0 and 1)
    phi_hat_new = np.clip(phi_hat_new, 0.0, 1.0)
    
    return phi_hat_new


class RK4Integrator:
    """
    RK4 integrator for time stepping the cell dynamics equations.
    """
    
    def __init__(self, dt):
        """
        Initialize the RK4 integrator.
        
        Args:
            dt: Time step size
        """
        self.dt = dt
    
    def step(self, phi_hat, nutrient_field, dx, dynamics_function):
        """
        Perform one RK4 time step.
        
        Args:
            phi_hat: Current cell fraction fields (M, nx, ny, nz)
            nutrient_field: Current nutrient field (nx, ny, nz)
            dx: Grid spacing
            dynamics_function: Function that computes dphi_hat/dt
            
        Returns:
            phi_hat_new: Updated cell fraction fields (M, nx, ny, nz)
        """
        # RK4 coefficients
        k1 = dynamics_function(phi_hat, nutrient_field, dx)
        
        # k2 = f(t + dt/2, y + dt*k1/2)
        phi_hat_k2 = phi_hat + 0.5 * self.dt * k1
        phi_hat_k2 = np.clip(phi_hat_k2, 0.0, 1.0)  # Physical constraints
        k2 = dynamics_function(phi_hat_k2, nutrient_field, dx)
        
        # k3 = f(t + dt/2, y + dt*k2/2)
        phi_hat_k3 = phi_hat + 0.5 * self.dt * k2
        phi_hat_k3 = np.clip(phi_hat_k3, 0.0, 1.0)  # Physical constraints
        k3 = dynamics_function(phi_hat_k3, nutrient_field, dx)
        
        # k4 = f(t + dt, y + dt*k3)
        phi_hat_k4 = phi_hat + self.dt * k3
        phi_hat_k4 = np.clip(phi_hat_k4, 0.0, 1.0)  # Physical constraints
        k4 = dynamics_function(phi_hat_k4, nutrient_field, dx)
        
        # Update using RK4 formula
        return rk4_step_vectorized(phi_hat, self.dt, k1, k2, k3, k4)
    
    def step_with_nutrient_update(self, phi_hat, nutrient_field, dx, dynamics_function, nutrient_function):
        """
        Perform one RK4 time step with nutrient field update.
        
        Args:
            phi_hat: Current cell fraction fields (M, nx, ny, nz)
            nutrient_field: Current nutrient field (nx, ny, nz)
            dx: Grid spacing
            dynamics_function: Function that computes dphi_hat/dt
            nutrient_function: Function that computes nutrient field update
            
        Returns:
            phi_hat_new: Updated cell fraction fields (M, nx, ny, nz)
            nutrient_field_new: Updated nutrient field (nx, ny, nz)
        """
        # Update cell dynamics with RK4
        phi_hat_new = self.step(phi_hat, nutrient_field, dx, dynamics_function)
        
        # Update nutrient field (could also use RK4 if needed)
        nutrient_field_new = nutrient_function(phi_hat_new, nutrient_field, dx)
        
        return phi_hat_new, nutrient_field_new


class AdaptiveRK4Integrator(RK4Integrator):
    """
    Adaptive RK4 integrator with error estimation and step size control.
    """
    
    def __init__(self, dt, tolerance=1e-6, min_dt=1e-8, max_dt=1e-2):
        """
        Initialize the adaptive RK4 integrator.
        
        Args:
            dt: Initial time step size
            tolerance: Error tolerance for adaptive stepping
            min_dt: Minimum allowed time step
            max_dt: Maximum allowed time step
        """
        super().__init__(dt)
        self.tolerance = tolerance
        self.min_dt = min_dt
        self.max_dt = max_dt
    
    def step_adaptive(self, phi_hat, nutrient_field, dx, dynamics_function):
        """
        Perform one adaptive RK4 time step.
        
        Args:
            phi_hat: Current cell fraction fields (M, nx, ny, nz)
            nutrient_field: Current nutrient field (nx, ny, nz)
            dx: Grid spacing
            dynamics_function: Function that computes dphi_hat/dt
            
        Returns:
            phi_hat_new: Updated cell fraction fields (M, nx, ny, nz)
            dt_used: Actual time step used
            success: Whether the step was successful
        """
        # Try the current time step
        phi_hat_new = self.step(phi_hat, nutrient_field, dx, dynamics_function)
        
        # Estimate error (simplified - could use embedded RK methods)
        # For now, we'll use a simple approach: check if any field changes are too large
        max_change = np.max(np.abs(phi_hat_new - phi_hat))
        
        if max_change > self.tolerance:
            # Error too large, reduce time step
            self.dt = max(self.dt * 0.5, self.min_dt)
            return phi_hat, self.dt, False
        elif max_change < self.tolerance * 0.1:
            # Error very small, can increase time step
            self.dt = min(self.dt * 1.1, self.max_dt)
        
        return phi_hat_new, self.dt, True
