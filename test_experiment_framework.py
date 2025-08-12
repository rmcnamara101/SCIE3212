"""
Test Script for Experiment Framework

This script tests the basic functionality of the ExperimentEngine with a minimal example.

Author: Riley Jae McNamara
Date: 2025-02-19
"""

import sys
from pathlib import Path

# Add the project root to the path
proj = Path(__file__).parent
sys.path.insert(0, str(proj))

from src.growkit.ExperimentEngine import ExperimentRunner, ParameterSweep, ParameterRange


def test_basic_functionality():
    """Test basic experiment framework functionality."""
    print("Testing Experiment Framework...")
    print("=" * 50)
    
    try:
        # Initialize experiment runner
        base_config = "templates/og.yaml"
        runner = ExperimentRunner(base_config, "test_experiments")
        
        # Create parameter sweep
        sweep = ParameterSweep("test_sweep", total_steps=5)  # Very short for testing
        
        # Define a simple parameter range
        growth_range = ParameterRange(
            start=1.0,
            end=1.5,
            num_points=3,  # Just 3 points for quick testing
            scale="linear"
        )
        
        # Generate experiment configurations
        experiments = sweep.single_parameter_sweep(
            parameter="populations.Diseased.dynamics.lambda",
            param_range=growth_range,
            save_interval=1,
            save_physics_fields=False,  # Disable for faster testing
            save_plots=False
        )
        
        print(f"Generated {len(experiments)} test experiments")
        
        # Run experiments (sequential for testing)
        print("Running experiments...")
        results = runner.run_experiments(experiments, parallel=False)
        
        # Print summary
        print("\nExperiment Summary:")
        runner.print_summary()
        
        # Save results
        runner.save_results("test_results.json")
        
        # Get results as DataFrame
        df = runner.get_results_dataframe()
        print(f"\nResults DataFrame shape: {df.shape}")
        if not df.empty:
            print("Sample results:")
            print(df[['name', 'success', 'execution_time', 'total_cells']].head())
        
        print("\n✅ Basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_parameter_range():
    """Test parameter range generation."""
    print("\nTesting Parameter Range Generation...")
    print("-" * 40)
    
    try:
        # Test linear range
        linear_range = ParameterRange(1.0, 2.0, 5, "linear")
        linear_values = linear_range.get_values()
        print(f"Linear range: {linear_values}")
        
        # Test logarithmic range
        log_range = ParameterRange(0.1, 1.0, 5, "log")
        log_values = log_range.get_values()
        print(f"Logarithmic range: {log_values}")
        
        # Test custom range
        custom_range = ParameterRange(0, 0, 0, "custom", custom_values=[1.0, 1.5, 2.0])
        custom_values = custom_range.get_values()
        print(f"Custom range: {custom_values}")
        
        print("✅ Parameter range test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Parameter range test failed: {e}")
        return False


def test_experiment_configuration():
    """Test experiment configuration generation."""
    print("\nTesting Experiment Configuration Generation...")
    print("-" * 50)
    
    try:
        sweep = ParameterSweep("test_config", total_steps=10)
        
        # Test single parameter sweep
        growth_range = ParameterRange(1.0, 1.5, 3, "linear")
        experiments = sweep.single_parameter_sweep(
            parameter="populations.Diseased.dynamics.lambda",
            param_range=growth_range
        )
        
        print(f"Generated {len(experiments)} single parameter experiments")
        for i, exp in enumerate(experiments):
            print(f"  Experiment {i+1}: {exp.name}")
            print(f"    Parameters: {exp.parameters}")
            print(f"    Total steps: {exp.total_steps}")
        
        # Test grid size benchmark
        grid_experiments = sweep.benchmark_grid_sizes([20, 30])
        print(f"\nGenerated {len(grid_experiments)} grid size experiments")
        for exp in grid_experiments:
            print(f"  {exp.name}: {exp.parameters}")
        
        print("✅ Experiment configuration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Experiment configuration test failed: {e}")
        return False


if __name__ == "__main__":
    print("Experiment Framework Test Suite")
    print("=" * 60)
    
    # Run tests
    test1_passed = test_parameter_range()
    test2_passed = test_experiment_configuration()
    test3_passed = test_basic_functionality()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Parameter Range Test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Configuration Test:  {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print(f"Basic Functionality: {'✅ PASSED' if test3_passed else '❌ FAILED'}")
    
    if all([test1_passed, test2_passed, test3_passed]):
        print("\n🎉 All tests passed! Experiment framework is working correctly.")
        print("\nYou can now run the example scripts:")
        print("  python examples/parameter_sweep_example.py")
        print("  python examples/sensitivity_analysis_example.py")
        print("  python examples/benchmarking_example.py")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
    
    print("\nCheck the 'test_experiments/' directory for test results.")
