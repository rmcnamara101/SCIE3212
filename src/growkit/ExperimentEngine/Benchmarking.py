"""
Benchmarking Module

Provides tools for benchmarking simulation performance across different parameters.
Supports grid size scaling, time step analysis, and computational efficiency metrics.

Author: Riley Jae McNamara
Date: 2025-02-19
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import time

from .ExperimentRunner import ExperimentConfig


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    parameter_name: str
    parameter_value: Any
    execution_time: float
    memory_usage: Optional[float] = None
    grid_size: Optional[int] = None
    total_cells: Optional[float] = None
    steps_per_second: Optional[float] = None
    cells_per_second: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'parameter_name': self.parameter_name,
            'parameter_value': self.parameter_value,
            'execution_time': self.execution_time,
            'memory_usage': self.memory_usage,
            'grid_size': self.grid_size,
            'total_cells': self.total_cells,
            'steps_per_second': self.steps_per_second,
            'cells_per_second': self.cells_per_second
        }


class BenchmarkingSuite:
    """
    Comprehensive benchmarking suite for tumor growth simulations.
    
    Supports:
    - Grid size scaling analysis
    - Time step efficiency analysis
    - Memory usage profiling
    - Computational efficiency metrics
    - Performance regression testing
    """
    
    def __init__(self, base_name: str = "benchmark", total_steps: int = 100):
        """
        Initialize benchmarking suite.
        
        Args:
            base_name: Base name for experiment configurations
            total_steps: Number of simulation steps for each benchmark
        """
        self.base_name = base_name
        self.total_steps = total_steps
        self.benchmark_results: List[BenchmarkResult] = []
    
    def grid_size_benchmark(self, grid_sizes: List[int], save_interval: int = 1,
                           save_physics_fields: bool = False, save_plots: bool = False) -> List[ExperimentConfig]:
        """
        Generate configurations for grid size benchmarking.
        
        Args:
            grid_sizes: List of grid sizes to test
            save_interval: How often to save data
            save_physics_fields: Whether to save physics fields
            save_plots: Whether to save plots
            
        Returns:
            List of experiment configurations
        """
        experiments = []
        
        for grid_size in grid_sizes:
            config = ExperimentConfig(
                name=f"{self.base_name}_grid_{grid_size}",
                parameters={"domain.shape": grid_size},
                total_steps=self.total_steps,
                save_interval=save_interval,
                save_physics_fields=save_physics_fields,
                save_plots=save_plots
            )
            experiments.append(config)
        
        return experiments
    
    def time_step_benchmark(self, time_steps: List[float], save_interval: int = 1,
                           save_physics_fields: bool = False, save_plots: bool = False) -> List[ExperimentConfig]:
        """
        Generate configurations for time step benchmarking.
        
        Args:
            time_steps: List of time step sizes to test
            save_interval: How often to save data
            save_physics_fields: Whether to save physics fields
            save_plots: Whether to save plots
            
        Returns:
            List of experiment configurations
        """
        experiments = []
        
        for dt in time_steps:
            config = ExperimentConfig(
                name=f"{self.base_name}_dt_{dt:.3g}".replace('.', 'p'),
                parameters={"time.dt": dt},
                total_steps=self.total_steps,
                save_interval=save_interval,
                save_physics_fields=save_physics_fields,
                save_plots=save_plots
            )
            experiments.append(config)
        
        return experiments
    
    def parameter_benchmark(self, parameter: str, values: List[Any], save_interval: int = 1,
                           save_physics_fields: bool = False, save_plots: bool = False) -> List[ExperimentConfig]:
        """
        Generate configurations for benchmarking a specific parameter.
        
        Args:
            parameter: Parameter name (dot notation supported)
            values: List of parameter values to test
            save_interval: How often to save data
            save_physics_fields: Whether to save physics fields
            save_plots: Whether to save plots
            
        Returns:
            List of experiment configurations
        """
        experiments = []
        
        for value in values:
            # Create a safe name for the parameter value
            if isinstance(value, float):
                name_suffix = f"{value:.3g}".replace('.', 'p')
            else:
                name_suffix = str(value)
            
            config = ExperimentConfig(
                name=f"{self.base_name}_{parameter}_{name_suffix}",
                parameters={parameter: value},
                total_steps=self.total_steps,
                save_interval=save_interval,
                save_physics_fields=save_physics_fields,
                save_plots=save_plots
            )
            experiments.append(config)
        
        return experiments
    
    def analyze_benchmark_results(self, results: List[Any], parameter_name: str) -> List[BenchmarkResult]:
        """
        Analyze benchmark results and extract performance metrics.
        
        Args:
            results: List of experiment results
            parameter_name: Name of the parameter being benchmarked
            
        Returns:
            List of benchmark results
        """
        benchmark_results = []
        
        for result in results:
            if not result.success:
                continue
            
            # Extract parameter value
            param_value = result.config.parameters.get(parameter_name)
            
            # Calculate performance metrics
            steps_per_second = self.total_steps / result.execution_time if result.execution_time > 0 else 0
            cells_per_second = result.total_cells / result.execution_time if result.execution_time > 0 else 0
            
            # Extract grid size if available
            grid_size = result.config.parameters.get("domain.shape")
            
            benchmark_result = BenchmarkResult(
                parameter_name=parameter_name,
                parameter_value=param_value,
                execution_time=result.execution_time,
                grid_size=grid_size,
                total_cells=result.total_cells,
                steps_per_second=steps_per_second,
                cells_per_second=cells_per_second
            )
            
            benchmark_results.append(benchmark_result)
        
        self.benchmark_results.extend(benchmark_results)
        return benchmark_results
    
    def create_benchmark_report(self, output_file: str = "benchmark_report.txt") -> str:
        """
        Create a comprehensive benchmark report.
        
        Args:
            output_file: Output file path
            
        Returns:
            Report content as string
        """
        if not self.benchmark_results:
            return "No benchmark results available."
        
        # Group results by parameter
        param_groups = {}
        for result in self.benchmark_results:
            if result.parameter_name not in param_groups:
                param_groups[result.parameter_name] = []
            param_groups[result.parameter_name].append(result)
        
        report_lines = []
        report_lines.append("BENCHMARK REPORT")
        report_lines.append("=" * 50)
        report_lines.append(f"Total benchmark runs: {len(self.benchmark_results)}")
        report_lines.append(f"Parameters tested: {len(param_groups)}")
        report_lines.append("")
        
        for param_name, results in param_groups.items():
            report_lines.append(f"PARAMETER: {param_name}")
            report_lines.append("-" * 30)
            
            # Sort by parameter value
            sorted_results = sorted(results, key=lambda x: x.parameter_value)
            
            # Calculate statistics
            execution_times = [r.execution_time for r in sorted_results]
            steps_per_second = [r.steps_per_second for r in sorted_results if r.steps_per_second is not None]
            cells_per_second = [r.cells_per_second for r in sorted_results if r.cells_per_second is not None]
            
            report_lines.append(f"Number of tests: {len(sorted_results)}")
            report_lines.append(f"Execution time range: {min(execution_times):.3f}s - {max(execution_times):.3f}s")
            report_lines.append(f"Average execution time: {np.mean(execution_times):.3f}s")
            
            if steps_per_second:
                report_lines.append(f"Steps per second range: {min(steps_per_second):.1f} - {max(steps_per_second):.1f}")
                report_lines.append(f"Average steps per second: {np.mean(steps_per_second):.1f}")
            
            if cells_per_second:
                report_lines.append(f"Cells per second range: {min(cells_per_second):.1f} - {max(cells_per_second):.1f}")
                report_lines.append(f"Average cells per second: {np.mean(cells_per_second):.1f}")
            
            report_lines.append("")
            report_lines.append("Detailed results:")
            for result in sorted_results:
                report_lines.append(f"  {result.parameter_value}: {result.execution_time:.3f}s "
                                  f"({result.steps_per_second:.1f} steps/s)")
            report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        # Save to file
        with open(output_file, 'w') as f:
            f.write(report_content)
        
        return report_content
    
    def plot_scaling_analysis(self, parameter_name: str, output_file: str = "scaling_analysis.png"):
        """
        Create scaling analysis plots.
        
        Args:
            parameter_name: Name of the parameter to analyze
            output_file: Output file path for the plot
        """
        # Filter results for the specified parameter
        param_results = [r for r in self.benchmark_results if r.parameter_name == parameter_name]
        
        if not param_results:
            print(f"No results found for parameter: {parameter_name}")
            return
        
        # Sort by parameter value
        sorted_results = sorted(param_results, key=lambda x: x.parameter_value)
        
        param_values = [r.parameter_value for r in sorted_results]
        execution_times = [r.execution_time for r in sorted_results]
        steps_per_second = [r.steps_per_second for r in sorted_results if r.steps_per_second is not None]
        cells_per_second = [r.cells_per_second for r in sorted_results if r.cells_per_second is not None]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Scaling Analysis: {parameter_name}', fontsize=16)
        
        # Execution time vs parameter
        axes[0, 0].plot(param_values, execution_times, 'bo-', linewidth=2, markersize=6)
        axes[0, 0].set_xlabel(parameter_name)
        axes[0, 0].set_ylabel('Execution Time (s)')
        axes[0, 0].set_title('Execution Time Scaling')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Steps per second vs parameter
        if steps_per_second:
            axes[0, 1].plot(param_values, steps_per_second, 'ro-', linewidth=2, markersize=6)
            axes[0, 1].set_xlabel(parameter_name)
            axes[0, 1].set_ylabel('Steps per Second')
            axes[0, 1].set_title('Computational Throughput')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Cells per second vs parameter
        if cells_per_second:
            axes[1, 0].plot(param_values, cells_per_second, 'go-', linewidth=2, markersize=6)
            axes[1, 0].set_xlabel(parameter_name)
            axes[1, 0].set_ylabel('Cells per Second')
            axes[1, 0].set_title('Biological Throughput')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Efficiency analysis (if grid size is available)
        grid_sizes = [r.grid_size for r in sorted_results if r.grid_size is not None]
        if grid_sizes and len(grid_sizes) == len(execution_times):
            # Calculate theoretical scaling (assuming O(n^3) for 3D grid)
            theoretical_times = [execution_times[0] * (gs / grid_sizes[0])**3 for gs in grid_sizes]
            
            axes[1, 1].plot(grid_sizes, execution_times, 'bo-', linewidth=2, markersize=6, label='Actual')
            axes[1, 1].plot(grid_sizes, theoretical_times, 'r--', linewidth=2, label='Theoretical O(n³)')
            axes[1, 1].set_xlabel('Grid Size')
            axes[1, 1].set_ylabel('Execution Time (s)')
            axes[1, 1].set_title('Scaling Efficiency')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Scaling analysis plot saved to {output_file}")
    
    def get_benchmark_dataframe(self) -> pd.DataFrame:
        """Convert benchmark results to a pandas DataFrame."""
        if not self.benchmark_results:
            return pd.DataFrame()
        
        data = []
        for result in self.benchmark_results:
            data.append(result.to_dict())
        
        return pd.DataFrame(data)
    
    def save_benchmark_results(self, filename: str = "benchmark_results.json"):
        """Save benchmark results to JSON file."""
        import json
        
        results_data = [result.to_dict() for result in self.benchmark_results]
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"Benchmark results saved to {filename}")
        return filename
