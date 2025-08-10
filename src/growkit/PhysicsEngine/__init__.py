"""
Physics Engine Module

This module contains all the physics components for the tumor growth model:
- VectorizedCellDynamics: Vectorized cell dynamics computation following the same pattern as SourceConstructor
- VectorizedPressureSolver: Vectorized pressure solver for the Poisson equation
- Legacy components: Original modular components for backward compatibility

The vectorized components follow the same pattern as SourceConstructor and FieldManager:
- Vectorized field operations (phi_hat for all populations)
- Matrix operations for efficiency
- Numba optimization for core computations
- Unified interface for all populations

Author: Riley Jae McNamara
Date: 2025-02-19
"""

# Vectorized components (new pattern)
from .VectorizedCellDynamics import VectorizedCellDynamics, compute_cell_dynamics_numba
from .Energy.VectorizedEnergy import VectorizedEnergy, compute_adhesion_energy_derivative_numba
from .MassFlux.VectorizedMassFlux import VectorizedMassFlux, compute_mass_fluxes_numba
from .SolidVelocity.VectorizedSolidVelocity import VectorizedSolidVelocity

__all__ = [
    # Vectorized components
    'VectorizedCellDynamics',
    'VectorizedEnergy',
    'VectorizedMassFlux',
    'VectorizedSolidVelocity',
    'compute_cell_dynamics_numba',
    'compute_adhesion_energy_derivative_numba',
    'compute_mass_fluxes_numba',
]
