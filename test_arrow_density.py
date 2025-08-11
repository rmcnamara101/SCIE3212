#!/usr/bin/env python3
"""
Test script to verify arrow density calculation logic.
"""

def test_arrow_density_calculation():
    """Test the effective_skip calculation for both velocity and mass flux fields."""
    
    # Test parameters from the notebook
    skip = 1
    zoom_factor = 3.0
    arrow_density_factor = 1.0
    
    # Calculate effective_skip (same logic for both velocity and mass flux)
    effective_skip = max(1, int(skip / (zoom_factor * arrow_density_factor)))
    
    print(f"Test parameters:")
    print(f"  skip = {skip}")
    print(f"  zoom_factor = {zoom_factor}")
    print(f"  arrow_density_factor = {arrow_density_factor}")
    print(f"  effective_skip = max(1, int({skip} / ({zoom_factor} * {arrow_density_factor})))")
    print(f"  effective_skip = max(1, int({skip} / {zoom_factor * arrow_density_factor}))")
    print(f"  effective_skip = max(1, int({skip / (zoom_factor * arrow_density_factor)}))")
    print(f"  effective_skip = max(1, {int(skip / (zoom_factor * arrow_density_factor))})")
    print(f"  effective_skip = {effective_skip}")
    
    # Test with different arrow_density_factor values
    print(f"\nTesting different arrow_density_factor values:")
    for arrow_density_factor in [1.0, 2.0, 3.0, 5.0, 10.0]:
        effective_skip = max(1, int(skip / (zoom_factor * arrow_density_factor)))
        print(f"  arrow_density_factor = {arrow_density_factor}: effective_skip = {effective_skip}")
    
    # Test with different zoom_factor values
    print(f"\nTesting different zoom_factor values:")
    arrow_density_factor = 1.0
    for zoom_factor in [1.0, 2.0, 3.0, 5.0, 10.0]:
        effective_skip = max(1, int(skip / (zoom_factor * arrow_density_factor)))
        print(f"  zoom_factor = {zoom_factor}: effective_skip = {effective_skip}")

if __name__ == "__main__":
    test_arrow_density_calculation()
