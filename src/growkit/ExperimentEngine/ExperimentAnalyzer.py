"""
Experiment Analyzer Module

This module provides the ExperimentAnalyzer class for analyzing experiment results
and generating insights from experimental data.

Author: Riley Jae McNamara
Date: 2025-02-19
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Union, Optional, Tuple
import json
import pickle
from scipy import stats
import os

from .ExperimentRunner import ExperimentRunner


class ExperimentAnalyzer:
    """
    Class for analyzing experiment results and generating insights.
    
    This class provides tools for loading, analyzing, and visualizing
    experiment results to extract insights and patterns.
    """
    
    def __init__(self, experiment_runner: Optional[ExperimentRunner] = None, 
                 results_directory: Optional[Union[str, Path]] = None):
        """
        Initialize the experiment analyzer.
        
        Args:
            experiment_runner: Optional ExperimentRunner instance
            results_directory: Directory containing experiment results
        """
        self.experiment_runner = experiment_runner
        self.results_directory = Path(results_directory) if results_directory else None
        
        # Load data if available
        self.results = {}
        self.failed_experiments = []
        self.completed_experiments = []
        self.experiment_configs = []
        self.experiment_names = []
        
        if experiment_runner:
            self._load_from_runner(experiment_runner)
        elif results_directory:
            self._load_from_directory(results_directory)
    
    def _load_from_runner(self, experiment_runner: ExperimentRunner):
        """Load data from an ExperimentRunner instance."""
        self.results = experiment_runner.results
        self.failed_experiments = experiment_runner.failed_experiments
        self.completed_experiments = experiment_runner.completed_experiments
        self.experiment_configs = experiment_runner.experiment_configs
        self.experiment_names = experiment_runner.experiment_names
    
    def _load_from_directory(self, results_directory: Union[str, Path]):
        """Load data from a results directory."""
        results_directory = Path(results_directory)
        
        # Load experiment summary
        summary_file = results_directory / "experiment_summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)
                self.completed_experiments = summary.get('execution_summary', {}).get('completed_experiments', 0)
                self.failed_experiments = summary.get('failed_experiments', [])
        
        # Load progress file for detailed results
        progress_file = results_directory / "experiment_progress.pkl"
        if progress_file.exists():
            with open(progress_file, 'rb') as f:
                progress_data = pickle.load(f)
                self.results = progress_data.get('results', {})
                self.completed_experiments = progress_data.get('completed_experiments', [])
                self.failed_experiments = progress_data.get('failed_experiments', [])
                self.experiment_configs = progress_data.get('experiment_configs', [])
                self.experiment_names = progress_data.get('experiment_names', [])
    
    def get_results_dataframe(self, include_parameters: bool = True, 
                            include_metrics: bool = True) -> pd.DataFrame:
        """
        Convert results to a pandas DataFrame for analysis.
        
        Args:
            include_parameters: Whether to include parameter values
            include_metrics: Whether to include computed metrics
            
        Returns:
            DataFrame containing experiment results
        """
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for exp_idx, result in self.results.items():
            if result.get('status') != 'completed':
                continue
            
            row = {
                'experiment_index': exp_idx,
                'experiment_name': result.get('experiment_name', ''),
                'execution_time': result.get('execution_time', 0),
                'output_directory': result.get('output_directory', '')
            }
            
            # Add parameters from config differences
            if include_parameters and 'config' in result and exp_idx < len(self.experiment_configs):
                # Extract parameter differences from base config
                base_config = self._get_base_config()
                if base_config:
                    param_diffs = self._extract_parameter_differences(base_config, result['config'])
                    for param_name, value in param_diffs.items():
                        row[f'param_{param_name}'] = value
            
            # Add computed metrics
            if include_metrics and 'simulation_data' in result:
                sim_data = result['simulation_data']
                if 'field_data' in sim_data and 'phi_hat' in sim_data['field_data']:
                    phi_hat_data = sim_data['field_data']['phi_hat']
                    if len(phi_hat_data) > 0:
                        # Final cell counts
                        final_phi_hat = phi_hat_data[-1]
                        for i in range(final_phi_hat.shape[0]):
                            row[f'final_cells_pop_{i}'] = np.sum(final_phi_hat[i])
                        
                        # Growth rate (if multiple time points)
                        if len(phi_hat_data) > 1:
                            initial_total = np.sum(phi_hat_data[0])
                            final_total = np.sum(phi_hat_data[-1])
                            if initial_total > 0:
                                row['growth_rate'] = (final_total - initial_total) / initial_total
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _get_base_config(self) -> Optional[Dict[str, Any]]:
        """Get the base configuration for parameter comparison."""
        if self.experiment_runner:
            return self.experiment_runner.base_config
        elif self.results_directory:
            base_config_file = self.results_directory / "base_config.yaml"
            if base_config_file.exists():
                import yaml
                with open(base_config_file, 'r') as f:
                    return yaml.safe_load(f)
        return None
    
    def _extract_parameter_differences(self, base_config: Dict[str, Any], 
                                     modified_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract parameter differences between base and modified configurations.
        
        Args:
            base_config: Base configuration
            modified_config: Modified configuration
            
        Returns:
            Dictionary of parameter differences
        """
        differences = {}
        
        def compare_dicts(base: Dict[str, Any], modified: Dict[str, Any], path: str = ""):
            for key in modified:
                full_path = f"{path}.{key}" if path else key
                
                if key not in base:
                    differences[full_path] = modified[key]
                elif isinstance(modified[key], dict) and isinstance(base[key], dict):
                    compare_dicts(base[key], modified[key], full_path)
                elif modified[key] != base[key]:
                    differences[full_path] = modified[key]
        
        compare_dicts(base_config, modified_config)
        return differences
    
    def analyze_parameter_sensitivity(self, target_metric: str = 'final_cells_pop_0',
                                   parameter_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze the sensitivity of a target metric to parameter changes.
        
        Args:
            target_metric: Name of the metric to analyze
            parameter_names: List of parameter names to analyze (default: all)
            
        Returns:
            Dictionary containing sensitivity analysis results
        """
        df = self.get_results_dataframe()
        if df.empty:
            return {}
        
        if target_metric not in df.columns:
            print(f"Target metric '{target_metric}' not found in results")
            return {}
        
        # Get parameter columns
        param_cols = [col for col in df.columns if col.startswith('param_')]
        if parameter_names:
            param_cols = [col for col in param_cols if any(param in col for param in parameter_names)]
        
        sensitivity_results = {}
        
        for param_col in param_cols:
            param_name = param_col.replace('param_', '')
            
            # Calculate correlation
            correlation = df[param_col].corr(df[target_metric])
            
            # Calculate linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                df[param_col], df[target_metric]
            )
            
            # Calculate partial correlation (controlling for other parameters)
            other_params = [col for col in param_cols if col != param_col]
            if other_params:
                # Simple approach: residual correlation
                residuals = df[target_metric] - (slope * df[param_col] + intercept)
                partial_corr = df[param_col].corr(residuals)
            else:
                partial_corr = correlation
            
            sensitivity_results[param_name] = {
                'correlation': correlation,
                'partial_correlation': partial_corr,
                'slope': slope,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'std_error': std_err
            }
        
        return sensitivity_results
    
    def create_parameter_plots(self, target_metric: str = 'final_cells_pop_0',
                              parameter_names: Optional[List[str]] = None,
                              plot_type: str = 'scatter',
                              figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Create plots showing the relationship between parameters and target metric.
        
        Args:
            target_metric: Name of the metric to plot
            parameter_names: List of parameter names to plot (default: all)
            plot_type: Type of plot ('scatter', 'box', 'violin')
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure object
        """
        df = self.get_results_dataframe()
        if df.empty or target_metric not in df.columns:
            print(f"No data available for plotting {target_metric}")
            return plt.figure()
        
        # Get parameter columns
        param_cols = [col for col in df.columns if col.startswith('param_')]
        if parameter_names:
            param_cols = [col for col in param_cols if any(param in col for param in parameter_names)]
        
        n_params = len(param_cols)
        if n_params == 0:
            print("No parameters found for plotting")
            return plt.figure()
        
        # Calculate subplot layout
        cols = min(3, n_params)
        rows = (n_params + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if n_params == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        else:
            axes = axes.flatten()
        
        for i, param_col in enumerate(param_cols):
            if i >= len(axes):
                break
            
            param_name = param_col.replace('param_', '')
            ax = axes[i]
            
            if plot_type == 'scatter':
                ax.scatter(df[param_col], df[target_metric], alpha=0.6)
                ax.set_xlabel(param_name)
                ax.set_ylabel(target_metric)
                
                # Add trend line
                z = np.polyfit(df[param_col], df[target_metric], 1)
                p = np.poly1d(z)
                ax.plot(df[param_col], p(df[param_col]), "r--", alpha=0.8)
                
            elif plot_type == 'box':
                # Create bins for continuous parameters
                if df[param_col].dtype in ['float64', 'float32']:
                    bins = pd.cut(df[param_col], bins=5)
                    df_box = df.groupby(bins)[target_metric].apply(list)
                    df_box.plot(kind='box', ax=ax)
                    ax.set_xlabel(param_name)
                    ax.set_ylabel(target_metric)
                    ax.tick_params(axis='x', rotation=45)
                else:
                    df.boxplot(column=target_metric, by=param_col, ax=ax)
                    ax.set_title(f'{target_metric} by {param_name}')
            
            elif plot_type == 'violin':
                if df[param_col].dtype in ['float64', 'float32']:
                    bins = pd.cut(df[param_col], bins=5)
                    df_violin = df.groupby(bins)[target_metric].apply(list)
                    df_violin.plot(kind='violin', ax=ax)
                    ax.set_xlabel(param_name)
                    ax.set_ylabel(target_metric)
                    ax.tick_params(axis='x', rotation=45)
                else:
                    df.violinplot(column=target_metric, by=param_col, ax=ax)
                    ax.set_title(f'{target_metric} by {param_name}')
        
        # Hide unused subplots
        for i in range(n_params, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def create_correlation_heatmap(self, metrics: Optional[List[str]] = None,
                                 figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Create a correlation heatmap for all parameters and metrics.
        
        Args:
            metrics: List of metrics to include (default: all)
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure object
        """
        df = self.get_results_dataframe()
        if df.empty:
            print("No data available for correlation analysis")
            return plt.figure()
        
        # Select columns for correlation analysis
        param_cols = [col for col in df.columns if col.startswith('param_')]
        metric_cols = [col for col in df.columns if not col.startswith('param_') and col not in ['experiment_index', 'experiment_name', 'output_directory']]
        
        if metrics:
            metric_cols = [col for col in metric_cols if col in metrics]
        
        # Create correlation matrix
        correlation_cols = param_cols + metric_cols
        correlation_df = df[correlation_cols].corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(correlation_df, annot=True, cmap='RdBu_r', center=0, 
                   square=True, ax=ax, fmt='.2f')
        ax.set_title('Parameter-Metric Correlation Matrix')
        plt.tight_layout()
        
        return fig
    
    def find_optimal_parameters(self, target_metric: str = 'final_cells_pop_0',
                              optimization_direction: str = 'maximize',
                              constraints: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, Any]:
        """
        Find optimal parameter values for a given target metric.
        
        Args:
            target_metric: Name of the metric to optimize
            optimization_direction: 'maximize' or 'minimize'
            constraints: Dictionary of parameter constraints (param_name: (min, max))
            
        Returns:
            Dictionary containing optimal parameters and results
        """
        df = self.get_results_dataframe()
        if df.empty or target_metric not in df.columns:
            return {}
        
        # Apply constraints
        if constraints:
            for param_name, (min_val, max_val) in constraints.items():
                param_col = f'param_{param_name}'
                if param_col in df.columns:
                    df = df[(df[param_col] >= min_val) & (df[param_col] <= max_val)]
        
        if df.empty:
            print("No data points satisfy the constraints")
            return {}
        
        # Find optimal experiment
        if optimization_direction == 'maximize':
            optimal_idx = df[target_metric].idxmax()
        else:
            optimal_idx = df[target_metric].idxmin()
        
        optimal_row = df.loc[optimal_idx]
        
        # Extract optimal parameters
        optimal_params = {}
        param_cols = [col for col in df.columns if col.startswith('param_')]
        for param_col in param_cols:
            param_name = param_col.replace('param_', '')
            optimal_params[param_name] = optimal_row[param_col]
        
        return {
            'experiment_index': int(optimal_row['experiment_index']),
            'experiment_name': optimal_row['experiment_name'],
            'optimal_parameters': optimal_params,
            'optimal_value': optimal_row[target_metric],
            'all_results': df.to_dict('records')
        }
    
    def generate_report(self, output_file: Optional[Union[str, Path]] = None) -> str:
        """
        Generate a comprehensive analysis report.
        
        Args:
            output_file: Optional file to save the report
            
        Returns:
            Report text as string
        """
        if not self.results:
            return "No results available for analysis"
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("EXPERIMENT ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Basic statistics
        df = self.get_results_dataframe()
        report_lines.append(f"Total Experiments: {len(self.results)}")
        report_lines.append(f"Completed Experiments: {len(self.completed_experiments)}")
        report_lines.append(f"Failed Experiments: {len(self.failed_experiments)}")
        report_lines.append(f"Success Rate: {len(self.completed_experiments)/len(self.results)*100:.1f}%")
        report_lines.append("")
        
        # Metrics summary
        if not df.empty:
            metric_cols = [col for col in df.columns if not col.startswith('param_') and col not in ['experiment_index', 'experiment_name', 'output_directory']]
            if metric_cols:
                report_lines.append("METRICS SUMMARY:")
                for metric in metric_cols:
                    values = df[metric].dropna()
                    if len(values) > 0:
                        report_lines.append(f"  {metric}:")
                        report_lines.append(f"    Mean: {values.mean():.4f}")
                        report_lines.append(f"    Std:  {values.std():.4f}")
                        report_lines.append(f"    Min:  {values.min():.4f}")
                        report_lines.append(f"    Max:  {values.max():.4f}")
                report_lines.append("")
        
        # Sensitivity analysis
        if not df.empty:
            param_cols = [col for col in df.columns if col.startswith('param_')]
            if param_cols and 'final_cells_pop_0' in df.columns:
                report_lines.append("PARAMETER SENSITIVITY ANALYSIS:")
                sensitivity = self.analyze_parameter_sensitivity('final_cells_pop_0')
                for param_name, sens_data in sensitivity.items():
                    report_lines.append(f"  {param_name}:")
                    report_lines.append(f"    Correlation: {sens_data['correlation']:.3f}")
                    report_lines.append(f"    R²: {sens_data['r_squared']:.3f}")
                    report_lines.append(f"    P-value: {sens_data['p_value']:.3e}")
                report_lines.append("")
        
        # Optimal parameters
        if not df.empty and 'final_cells_pop_0' in df.columns:
            optimal = self.find_optimal_parameters('final_cells_pop_0')
            if optimal:
                report_lines.append("OPTIMAL PARAMETERS (maximizing final_cells_pop_0):")
                report_lines.append(f"  Experiment: {optimal['experiment_name']}")
                report_lines.append(f"  Optimal Value: {optimal['optimal_value']:.4f}")
                report_lines.append("  Parameters:")
                for param_name, value in optimal['optimal_parameters'].items():
                    report_lines.append(f"    {param_name}: {value}")
                report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        # Save report if requested
        if output_file:
            output_file = Path(output_file)
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {output_file}")
        
        return report_text
    
    def save_analysis_results(self, output_directory: Union[str, Path]):
        """
        Save analysis results to files.
        
        Args:
            output_directory: Directory to save analysis results
        """
        output_directory = Path(output_directory)
        os.makedirs(output_directory, exist_ok=True)
        
        # Save results DataFrame
        df = self.get_results_dataframe()
        if not df.empty:
            df.to_csv(output_directory / "experiment_results.csv", index=False)
            df.to_excel(output_directory / "experiment_results.xlsx", index=False)
        
        # Save sensitivity analysis
        if not df.empty and 'final_cells_pop_0' in df.columns:
            sensitivity = self.analyze_parameter_sensitivity('final_cells_pop_0')
            with open(output_directory / "sensitivity_analysis.json", 'w') as f:
                json.dump(sensitivity, f, indent=2, default=str)
        
        # Save optimal parameters
        if not df.empty and 'final_cells_pop_0' in df.columns:
            optimal = self.find_optimal_parameters('final_cells_pop_0')
            if optimal:
                with open(output_directory / "optimal_parameters.json", 'w') as f:
                    json.dump(optimal, f, indent=2, default=str)
        
        print(f"Analysis results saved to {output_directory}")
