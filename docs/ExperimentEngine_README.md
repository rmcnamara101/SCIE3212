# Experiment Engine Documentation

The Experiment Engine provides a comprehensive framework for running systematic experiments on tumor growth simulations. It supports parameter sweeps, sensitivity analysis, and performance benchmarking with parallel execution capabilities.

## Overview

The Experiment Engine consists of four main components:

1. **ExperimentRunner**: Core class for orchestrating experiments
2. **ParameterSweep**: Generates experiment configurations for parameter sweeps
3. **SensitivityAnalysis**: Implements various sensitivity analysis methods
4. **BenchmarkingSuite**: Provides performance benchmarking tools

## Quick Start

### Basic Parameter Sweep

```python
from src.growkit.ExperimentEngine import ExperimentRunner, ParameterSweep, ParameterRange

# Initialize experiment runner
runner = ExperimentRunner("templates/og.yaml", "experiments/my_sweep")

# Create parameter sweep
sweep = ParameterSweep("growth_sweep", total_steps=50)

# Define parameter range
growth_range = ParameterRange(start=0.5, end=2.0, num_points=10, scale="linear")

# Generate experiments
experiments = sweep.single_parameter_sweep(
    parameter="populations.Diseased.dynamics.lambda",
    param_range=growth_range
)

# Run experiments
results = runner.run_experiments(experiments, parallel=True)

# Save results
runner.save_results("my_sweep_results.json")
runner.print_summary()
```

### Grid Size Benchmarking

```python
from src.growkit.ExperimentEngine import ExperimentRunner, BenchmarkingSuite

# Initialize
runner = ExperimentRunner("templates/og.yaml", "experiments/benchmark")
benchmark = BenchmarkingSuite("grid_benchmark", total_steps=50)

# Generate grid size experiments
experiments = benchmark.grid_size_benchmark(
    grid_sizes=[20, 30, 40, 50, 60],
    save_physics_fields=False  # Disable for benchmarking
)

# Run and analyze
results = runner.run_experiments(experiments, parallel=True)
benchmark_results = benchmark.analyze_benchmark_results(results, "domain.shape")
benchmark.create_benchmark_report()
benchmark.plot_scaling_analysis("domain.shape")
```

## Core Components

### ExperimentRunner

The main orchestrator for running experiments.

**Key Features:**
- Parallel execution support
- Automatic result collection and analysis
- JSON result export
- Progress tracking and error handling

**Methods:**
- `run_experiments()`: Execute multiple experiments
- `save_results()`: Save results to JSON
- `get_results_dataframe()`: Get results as pandas DataFrame
- `print_summary()`: Print experiment summary

### ParameterSweep

Generates experiment configurations for systematic parameter exploration.

**Supported Methods:**
- `single_parameter_sweep()`: One-dimensional parameter sweep
- `multi_parameter_sweep()`: Full factorial design
- `latin_hypercube_sweep()`: Latin Hypercube Sampling
- `custom_parameter_combinations()`: Custom parameter sets
- `benchmark_grid_sizes()`: Grid size benchmarking

**Parameter Range Types:**
- Linear: `ParameterRange(start, end, num_points, scale="linear")`
- Logarithmic: `ParameterRange(start, end, num_points, scale="log")`
- Custom: `ParameterRange(start, end, num_points, scale="custom", custom_values=[...])`

### SensitivityAnalysis

Implements global sensitivity analysis methods.

**Supported Methods:**
- `sobol_analysis()`: Sobol indices (first and total order)
- `morris_screening()`: Morris screening method
- `elementary_effects()`: Elementary Effects method

**Analysis Methods:**
- `analyze_sobol_results()`: Analyze Sobol experiment results
- `analyze_morris_results()`: Analyze Morris experiment results
- `create_sensitivity_report()`: Generate human-readable reports

### BenchmarkingSuite

Provides comprehensive performance benchmarking tools.

**Supported Benchmarks:**
- `grid_size_benchmark()`: Grid size scaling analysis
- `time_step_benchmark()`: Time step efficiency analysis
- `parameter_benchmark()`: Parameter-specific benchmarking

**Analysis Features:**
- Execution time tracking
- Steps per second calculation
- Cells per second calculation
- Scaling analysis plots
- Performance trend analysis

## Parameter Notation

The Experiment Engine uses dot notation to specify nested parameters in YAML configurations:

```python
# Examples of parameter paths:
"populations.Healthy.dynamics.lambda"      # Growth rate
"populations.Diseased.dynamics.mobility"   # Cell mobility
"nutrient.dynamics.diffusion"             # Nutrient diffusion
"domain.shape"                            # Grid size
"time.dt"                                 # Time step
"physics.adhesion_energy.m"               # Adhesion energy
```

## Parallel Execution

The Experiment Engine supports parallel execution for efficient experiment running:

```python
# Sequential execution
results = runner.run_experiments(experiments, parallel=False)

# Parallel execution (auto-detect CPU cores)
results = runner.run_experiments(experiments, parallel=True)

# Parallel execution with specific number of workers
results = runner.run_experiments(experiments, parallel=True, max_workers=8)
```

## Output Structure

Experiments are organized in the following structure:

```
experiments/
├── experiment_name/
│   ├── experiment_001/
│   │   ├── simulation_data.npz
│   │   ├── physics_fields.npz
│   │   └── plots/
│   ├── experiment_002/
│   └── ...
├── experiment_results.json
└── experiment_summary.txt
```

## Example Use Cases

### 1. Parameter Sweep for Growth Rate

```python
# Sweep growth rate from 0.5 to 2.0
growth_range = ParameterRange(0.5, 2.0, 10, "linear")
experiments = sweep.single_parameter_sweep(
    parameter="populations.Diseased.dynamics.lambda",
    param_range=growth_range,
    total_steps=50
)
```

### 2. Multi-Parameter Sensitivity Analysis

```python
# Define parameters for sensitivity analysis
parameters = {
    "populations.Healthy.dynamics.lambda": (0.8, 1.5),
    "populations.Diseased.dynamics.lambda": (1.0, 2.0),
    "nutrient.dynamics.diffusion": (10, 30)
}

# Generate Sobol analysis experiments
experiments, analysis_info = sensitivity.sobol_analysis(
    parameters=parameters,
    n_samples=512
)
```

### 3. Performance Benchmarking

```python
# Benchmark different grid sizes
grid_sizes = [20, 30, 40, 50, 60, 70, 80]
experiments = benchmark.grid_size_benchmark(
    grid_sizes=grid_sizes,
    save_physics_fields=False  # Disable for performance
)

# Analyze scaling
benchmark_results = benchmark.analyze_benchmark_results(results, "domain.shape")
benchmark.plot_scaling_analysis("domain.shape")
```

## Advanced Features

### Custom Parameter Combinations

```python
# Define custom parameter combinations
custom_params = [
    {"populations.Healthy.dynamics.lambda": 1.0, "nutrient.dynamics.diffusion": 20},
    {"populations.Healthy.dynamics.lambda": 1.2, "nutrient.dynamics.diffusion": 25},
    {"populations.Healthy.dynamics.lambda": 1.5, "nutrient.dynamics.diffusion": 30}
]

experiments = sweep.custom_parameter_combinations(custom_params)
```

### Latin Hypercube Sampling

```python
# Efficient parameter space exploration
parameters = {
    "populations.Healthy.dynamics.lambda": (0.8, 1.5),
    "populations.Diseased.dynamics.lambda": (1.0, 2.0),
    "nutrient.dynamics.diffusion": (10, 30)
}

experiments = sweep.latin_hypercube_sweep(
    parameters=parameters,
    num_samples=20
)
```

### Comprehensive Benchmarking

```python
# Test multiple parameters simultaneously
all_experiments = []

# Grid size tests
grid_experiments = benchmark.grid_size_benchmark([25, 35, 45, 55])
all_experiments.extend(grid_experiments)

# Time step tests
timestep_experiments = benchmark.time_step_benchmark([0.2, 0.4, 0.6])
all_experiments.extend(timestep_experiments)

# Run all experiments
results = runner.run_experiments(all_experiments, parallel=True)
```

## Dependencies

The Experiment Engine requires the following dependencies:

- **Core**: numpy, pandas, matplotlib, pyyaml
- **Sensitivity Analysis**: SALib (optional, for Sobol and Morris methods)
- **Parallel Processing**: multiprocessing (built-in)

Install optional dependencies:
```bash
pip install SALib
```

## Best Practices

### 1. Experiment Design
- Start with small parameter ranges and few points
- Use logarithmic scales for parameters spanning multiple orders of magnitude
- Consider using Latin Hypercube Sampling for high-dimensional parameter spaces

### 2. Performance Optimization
- Disable physics fields and plots for benchmarking runs
- Use appropriate save intervals (larger for longer simulations)
- Leverage parallel execution for multiple experiments

### 3. Result Analysis
- Always save results to JSON for later analysis
- Use the DataFrame interface for statistical analysis
- Generate plots for visual interpretation

### 4. Error Handling
- Monitor experiment success rates
- Check error messages for failed experiments
- Consider parameter ranges that might cause numerical instabilities

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the project root is in your Python path
2. **Memory Issues**: Reduce grid sizes or number of parallel workers
3. **Numerical Instabilities**: Check parameter ranges and time step sizes
4. **Missing Dependencies**: Install SALib for sensitivity analysis methods

### Performance Tips

1. **Grid Size**: Start with smaller grids (20-50) for testing
2. **Time Steps**: Use larger time steps for faster execution
3. **Parallel Workers**: Match to available CPU cores
4. **Save Intervals**: Increase for longer simulations

## Examples

See the `examples/` directory for complete working examples:

- `parameter_sweep_example.py`: Parameter sweep demonstrations
- `sensitivity_analysis_example.py`: Sensitivity analysis methods
- `benchmarking_example.py`: Performance benchmarking

Run examples:
```bash
python examples/parameter_sweep_example.py
python examples/sensitivity_analysis_example.py
python examples/benchmarking_example.py
```

## Contributing

The Experiment Engine is designed to be extensible. You can:

1. Add new parameter sweep methods
2. Implement additional sensitivity analysis techniques
3. Create custom benchmarking metrics
4. Extend result analysis capabilities

## License

This code is part of the SCIE3212 tumor growth simulation project.

