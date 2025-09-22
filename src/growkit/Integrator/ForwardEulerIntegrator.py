"""
Forward Euler Integrator Module

This module implements a simple Forward Euler integrator for time stepping
the cell dynamics equations. The Forward Euler method is the simplest explicit
integration scheme, providing a good balance between simplicity and computational
efficiency for systems where stability is not a major concern.

The Forward Euler method computes:
y(t + dt) = y(t) + dt * f(t, y)

This is much simpler than RK4 but less accurate and potentially less stable
for stiff systems. It's useful for:
- Quick prototyping and testing
- Systems with smooth dynamics
- When computational speed is more important than accuracy
- Educational purposes to understand basic integration

Author: Riley Jae McNamara
Date: 2025-02-19
"""

import numpy as np
from numba import njit


@njit
def forward_euler_step_vectorized(phi_hat, dt, dphi_dt):
    """
    Perform one Forward Euler step for vectorized fields.
    
    Args:
        phi_hat: Current state (M, nx, ny, nz)
        dt: Time step
        dphi_dt: Time derivative (M, nx, ny, nz)
        
    Returns:
        phi_hat_new: Updated state (M, nx, ny, nz)
    """
    # Forward Euler update: y(t + dt) = y(t) + dt * f(t, y)
    phi_hat_new = phi_hat + dt * dphi_dt
    
    # Ensure physical constraints (volume fractions between 0 and 1)
    phi_hat_new = np.clip(phi_hat_new, 0.0, 1.0)
    
    return phi_hat_new


class ForwardEulerIntegrator:
    """
    Forward Euler integrator for time stepping the cell dynamics equations.
    
    This is the simplest explicit integration method, making it ideal for:
    - Quick prototyping and testing
    - Systems with smooth, non-stiff dynamics
    - When computational speed is prioritized over accuracy
    - Educational purposes
    """
    
    def __init__(self, dt):
        """
        Initialize the Forward Euler integrator.
        
        Args:
            dt: Time step size
        """
        self.dt = dt
    
    def step(self, phi_hat, nutrient_field, dx, dynamics_function):
        """
        Perform one Forward Euler time step.
        
        Args:
            phi_hat: Current cell fraction fields (M, nx, ny, nz)
            nutrient_field: Current nutrient field (nx, ny, nz)
            dx: Grid spacing
            dynamics_function: Function that computes dphi_hat/dt
            
        Returns:
            phi_hat_new: Updated cell fraction fields (M, nx, ny, nz)
        """
        # Compute time derivative
        dphi_dt = dynamics_function(phi_hat, nutrient_field, dx)
        
        # Forward Euler update
        return forward_euler_step_vectorized(phi_hat, self.dt, dphi_dt)
    
    def step_with_nutrient_update(self, phi_hat, nutrient_field, dx, dynamics_function, nutrient_function):
        """
        Perform one Forward Euler time step with nutrient field update.
        
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
        # Update cell dynamics with Forward Euler
        phi_hat_new = self.step(phi_hat, nutrient_field, dx, dynamics_function)
        
        # Update nutrient field
        nutrient_field_new = nutrient_function(phi_hat_new, nutrient_field, dx)
        
        return phi_hat_new, nutrient_field_new
    
    def step_optimized(self, phi_hat, nutrient_field, dx, dynamics_function, nutrient_function):
        """
        Perform one optimized Forward Euler time step.
        
        For Forward Euler, this is the same as step_with_nutrient_update since
        there's no optimization to be done (unlike RK4 which can optimize pressure solver calls).
        
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
        return self.step_with_nutrient_update(phi_hat, nutrient_field, dx, dynamics_function, nutrient_function)
    
    def step_adaptive_dt(self, phi_hat, nutrient_field, dx, dynamics_function, max_change=0.1):
        """
        Perform one Forward Euler time step with adaptive time step control.
        
        This method automatically reduces the time step if changes are too large,
        helping to maintain stability for systems that might otherwise become unstable.
        
        Args:
            phi_hat: Current cell fraction fields (M, nx, ny, nz)
            nutrient_field: Current nutrient field (nx, ny, nz)
            dx: Grid spacing
            dynamics_function: Function that computes dphi_hat/dt
            max_change: Maximum allowed change per time step
            
        Returns:
            phi_hat_new: Updated cell fraction fields (M, nx, ny, nz)
            dt_used: Actual time step used
            success: Whether the step was successful
        """
        # Try the current time step
        dphi_dt = dynamics_function(phi_hat, nutrient_field, dx)
        phi_hat_new = forward_euler_step_vectorized(phi_hat, self.dt, dphi_dt)
        
        # Check if changes are too large
        max_change_observed = np.max(np.abs(phi_hat_new - phi_hat))
        
        if max_change_observed > max_change:
            # Changes too large, reduce time step and retry
            dt_reduced = self.dt * (max_change / max_change_observed)
            phi_hat_new = forward_euler_step_vectorized(phi_hat, dt_reduced, dphi_dt)
            return phi_hat_new, dt_reduced, True
        else:
            return phi_hat_new, self.dt, True


class AdaptiveForwardEulerIntegrator(ForwardEulerIntegrator):
    """
    Adaptive Forward Euler integrator with automatic time step control.
    
    This integrator automatically adjusts the time step based on the magnitude
    of changes in the solution, helping to maintain stability and accuracy.
    """
    
    def __init__(self, dt, tolerance=0.1, min_dt=1e-8, max_dt=1e-2, safety_factor=0.8):
        """
        Initialize the adaptive Forward Euler integrator.
        
        Args:
            dt: Initial time step size
            tolerance: Maximum allowed change per time step
            min_dt: Minimum allowed time step
            max_dt: Maximum allowed time step
            safety_factor: Safety factor for time step adjustment
        """
        super().__init__(dt)
        self.tolerance = tolerance
        self.min_dt = min_dt
        self.max_dt = max_dt
        self.safety_factor = safety_factor
        self.original_dt = dt
    
    def step_adaptive(self, phi_hat, nutrient_field, dx, dynamics_function):
        """
        Perform one adaptive Forward Euler time step.
        
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
        # Compute time derivative
        dphi_dt = dynamics_function(phi_hat, nutrient_field, dx)
        
        # Try the current time step
        phi_hat_new = forward_euler_step_vectorized(phi_hat, self.dt, dphi_dt)
        
        # Check if changes are acceptable
        max_change = np.max(np.abs(phi_hat_new - phi_hat))
        
        if max_change > self.tolerance:
            # Changes too large, reduce time step
            dt_new = self.dt * self.safety_factor * (self.tolerance / max_change)
            dt_new = max(dt_new, self.min_dt)
            
            if dt_new < self.min_dt:
                # Time step too small, return failure
                return phi_hat, self.dt, False
            
            # Retry with reduced time step
            self.dt = dt_new
            phi_hat_new = forward_euler_step_vectorized(phi_hat, self.dt, dphi_dt)
            
        elif max_change < self.tolerance * 0.1:
            # Changes very small, can increase time step
            dt_new = min(self.dt * 1.1, self.max_dt)
            self.dt = dt_new
        
        return phi_hat_new, self.dt, True
    
    def reset_time_step(self):
        """Reset the time step to its original value."""
        self.dt = self.original_dt
    
    def step_optimized(self, phi_hat, nutrient_field, dx, dynamics_function, nutrient_function):
        """
        Perform one optimized adaptive Forward Euler time step.
        
        For adaptive Forward Euler, this is the same as step_with_nutrient_update.
        
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
        # Update cell dynamics with adaptive Forward Euler
        phi_hat_new, dt_used, success = self.step_adaptive(phi_hat, nutrient_field, dx, dynamics_function)
        
        if not success:
            # If adaptive step failed, fall back to basic step
            phi_hat_new = self.step(phi_hat, nutrient_field, dx, dynamics_function)
        
        # Update nutrient field
        nutrient_field_new = nutrient_function(phi_hat_new, nutrient_field, dx)
        
        return phi_hat_new, nutrient_field_new


class ImprovedEulerIntegrator(ForwardEulerIntegrator):
    """
    Improved Euler (Heun's method) integrator.
    
    This is a second-order method that's still simpler than RK4 but more
    accurate than basic Forward Euler. It uses a predictor-corrector approach:
    1. Predict: y* = y + dt * f(t, y)
    2. Correct: y(t + dt) = y + dt * [f(t, y) + f(t + dt, y*)] / 2
    """
    
    def step(self, phi_hat, nutrient_field, dx, dynamics_function):
        """
        Perform one Improved Euler time step.
        
        Args:
            phi_hat: Current cell fraction fields (M, nx, ny, nz)
            nutrient_field: Current nutrient field (nx, ny, nz)
            dx: Grid spacing
            dynamics_function: Function that computes dphi_hat/dt
            
        Returns:
            phi_hat_new: Updated cell fraction fields (M, nx, ny, nz)
        """
        # Predictor step (Forward Euler)
        dphi_dt1 = dynamics_function(phi_hat, nutrient_field, dx)
        phi_hat_predictor = phi_hat + self.dt * dphi_dt1
        
        # Corrector step
        dphi_dt2 = dynamics_function(phi_hat_predictor, nutrient_field, dx)
        
        # Improved Euler update: average of the two derivatives
        phi_hat_new = phi_hat + self.dt * (dphi_dt1 + dphi_dt2) / 2.0
        
        # Ensure physical constraints
        phi_hat_new = np.clip(phi_hat_new, 0.0, 1.0)
        
        return phi_hat_new
    
    def step_optimized(self, phi_hat, nutrient_field, dx, dynamics_function, nutrient_function):
        """
        Perform one optimized Improved Euler time step.
        
        For Improved Euler, this is the same as step_with_nutrient_update.
        
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
        return self.step_with_nutrient_update(phi_hat, nutrient_field, dx, dynamics_function, nutrient_function)
