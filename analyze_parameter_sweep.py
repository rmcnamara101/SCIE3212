"""
Parameter Sweep Analysis Script

This script loads parameter sweep results from Excel files and creates
various plots to analyze the time series data and parameter relationships.

Author: Riley Jae McNamara
Date: 2025-02-19
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import ast
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")


class ParameterSweepAnalyzer:
    """
    Analyzer for parameter sweep results from Excel files.
    
    This class loads the results, extracts time series data, and creates
    various plots for analysis and cost function landscape exploration.
    """
    
    def __init__(self, excel_file_path: str):
        """
        Initialize the analyzer.
        
        Args:
            excel_file_path: Path to the Excel file containing parameter sweep results
        """
        self.excel_file_path = Path(excel_file_path)
        self.results_df = None
        self.time_series_data = {}
        self.parameter_columns = []
        self.population_names = []
        
        # Load the data
        self._load_data()
        
    def _load_data(self):
        """Load and process the Excel data."""
        print(f"Loading data from: {self.excel_file_path}")
        
        # Load the main results sheet
        self.results_df = pd.read_excel(self.excel_file_path, sheet_name='Results')
        print(f"Loaded {len(self.results_df)} experiments with {len(self.results_df.columns)} columns")
        
        # Identify parameter columns
        self.parameter_columns = [col for col in self.results_df.columns if col.startswith('param_')]
        print(f"Found {len(self.parameter_columns)} parameter columns")
        
        # Identify time series columns
        time_series_columns = [col for col in self.results_df.columns if col.startswith('time_series_')]
        print(f"Found {len(time_series_columns)} time series columns")
        
        # Extract population names from time series columns
        self._extract_population_names(time_series_columns)
        
        # Parse time series data
        self._parse_time_series_data(time_series_columns)
        
        print("Data loading completed!")
        
    def _extract_population_names(self, time_series_columns: List[str]):
        """Extract unique population names from time series columns."""
        population_names = set()
        
        for col in time_series_columns:
            # Extract population name from column name
            # Format: time_series_{population}_{metric}
            parts = col.replace('time_series_', '').split('_')
            if len(parts) >= 2:
                # Handle cases like "Stem_total_cells" -> "Stem"
                if parts[1] in ['total', 'radius']:
                    population_names.add(parts[0])
                else:
                    # Handle cases like "total_cells" -> "total"
                    population_names.add(parts[0])
        
        self.population_names = sorted(list(population_names))
        print(f"Identified populations: {self.population_names}")
        
    def _parse_time_series_data(self, time_series_columns: List[str]):
        """Parse time series data from string representations."""
        print("Parsing time series data...")
        
        for col in time_series_columns:
            try:
                # Parse the string representation of the time series
                parsed_data = []
                for idx, row in self.results_df.iterrows():
                    try:
                        if pd.isna(row[col]):
                            parsed_data.append([])
                        else:
                            # Parse string representation like "[1.0, 2.5, 3.2]"
                            data_str = str(row[col]).strip()
                            if data_str.startswith('[') and data_str.endswith(']'):
                                parsed_data.append(ast.literal_eval(data_str))
                            else:
                                parsed_data.append([])
                    except Exception as e:
                        print(f"Warning: Could not parse row {idx} in column {col}: {e}")
                        parsed_data.append([])
                
                self.time_series_data[col] = parsed_data
                
            except Exception as e:
                print(f"Warning: Could not parse column {col}: {e}")
        
        print(f"Successfully parsed {len(self.time_series_data)} time series columns")
        
    def get_time_series_length(self) -> int:
        """Get the length of the time series data."""
        if not self.time_series_data:
            return 0
        
        # Get the first non-empty time series
        for col, data in self.time_series_data.items():
            for series in data:
                if series:
                    return len(series)
        return 0
    
    def get_time_points(self) -> np.ndarray:
        """Get time points for the simulation."""
        length = self.get_time_series_length()
        if length == 0:
            return np.array([])
        
        # Assuming uniform time steps, create time array
        # You can modify this based on your actual time configuration
        dt = 0.1  # Default time step, adjust as needed
        return np.arange(length) * dt
    
    def plot_population_evolution(self, population_name: str, metric: str = 'total_cells', 
                                output_dir: Optional[str] = None, save_plot: bool = False,
                                figsize: Tuple[int, int] = (12, 8)):
        """
        Plot evolution of a specific population metric over time.
        
        Args:
            population_name: Name of the population to plot
            metric: Metric to plot ('total_cells' or 'radius')
            output_dir: Directory to save plots
            save_plot: Whether to save the plot
            figsize: Figure size
        """
        # Find the relevant column
        column_name = f'time_series_{population_name}_{metric}'
        
        if column_name not in self.time_series_data:
            print(f"Warning: Column {column_name} not found")
            return
        
        # Get time points
        time_points = self.get_time_points()
        if len(time_points) == 0:
            print("No time series data available")
            return
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot each experiment
        for idx, series in enumerate(self.time_series_data[column_name]):
            if series and len(series) == len(time_points):
                # Color based on experiment index
                color = plt.cm.viridis(idx / len(self.results_df))
                alpha = 0.7 if len(self.results_df) > 10 else 1.0
                
                ax.plot(time_points, series, color=color, alpha=alpha, linewidth=1)
        
        # Add mean line if multiple experiments
        if len(self.results_df) > 1:
            # Calculate mean at each time point
            mean_series = []
            for t in range(len(time_points)):
                values_at_t = []
                for series in self.time_series_data[column_name]:
                    if series and len(series) > t:
                        values_at_t.append(series[t])
                if values_at_t:
                    mean_series.append(np.mean(values_at_t))
                else:
                    mean_series.append(0)
            
            ax.plot(time_points, mean_series, 'k-', linewidth=3, label='Mean', alpha=0.8)
            ax.legend()
        
        # Customize plot
        ax.set_xlabel('Time')
        ax.set_ylabel(f'{metric.replace("_", " ").title()}')
        ax.set_title(f'{population_name} {metric.replace("_", " ").title()} Evolution')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot and output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            filename = f'{population_name}_{metric}_evolution.png'
            plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {output_path / filename}")
        
        plt.show()
        
    def plot_total_evolution(self, metric: str = 'total_cells', 
                           output_dir: Optional[str] = None, save_plot: bool = False,
                           figsize: Tuple[int, int] = (12, 8)):
        """
        Plot evolution of total metric across all populations.
        
        Args:
            metric: Metric to plot ('total_cells' or 'radius')
            output_dir: Directory to save plots
            save_plot: Whether to save the plot
            figsize: Figure size
        """
        # Find the relevant column
        column_name = f'time_series_total_{metric}'
        
        if column_name not in self.time_series_data:
            print(f"Warning: Column {column_name} not found")
            return
        
        # Get time points
        time_points = self.get_time_points()
        if len(time_points) == 0:
            print("No time series data available")
            return
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot each experiment
        for idx, series in enumerate(self.time_series_data[column_name]):
            if series and len(series) == len(time_points):
                # Color based on experiment index
                color = plt.cm.viridis(idx / len(self.results_df))
                alpha = 0.7 if len(self.results_df) > 10 else 1.0
                
                ax.plot(time_points, series, color=color, alpha=alpha, linewidth=1)
        
        # Add mean line if multiple experiments
        if len(self.results_df) > 1:
            # Calculate mean at each time point
            mean_series = []
            for t in range(len(time_points)):
                values_at_t = []
                for series in self.time_series_data[column_name]:
                    if series and len(series) > t:
                        values_at_t.append(series[t])
                if values_at_t:
                    mean_series.append(np.mean(values_at_t))
                else:
                    mean_series.append(0)
            
            ax.plot(time_points, mean_series, 'k-', linewidth=3, label='Mean', alpha=0.8)
            ax.legend()
        
        # Customize plot
        ax.set_xlabel('Time')
        ax.set_ylabel(f'Total {metric.replace("_", " ").title()}')
        ax.set_title(f'Total {metric.replace("_", " ").title()} Evolution')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot and output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            filename = f'total_{metric}_evolution.png'
            plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {output_path / filename}")
        
        plt.show()
        
    def plot_parameter_sensitivity(self, parameter_name: str, metric: str = 'total_cells',
                                 time_point: Optional[int] = None, 
                                 output_dir: Optional[str] = None, save_plot: bool = False,
                                 figsize: Tuple[int, int] = (12, 8)):
        """
        Plot sensitivity of a metric to a specific parameter.
        
        Args:
            parameter_name: Name of the parameter to analyze
            metric: Metric to analyze ('total_cells' or 'radius')
            time_point: Specific time point to analyze (None for final time)
            output_dir: Directory to save plots
            save_plot: Whether to save the plot
            figsize: Figure size
        """
        # Find the parameter column
        param_col = f'param_{parameter_name}'
        if param_col not in self.results_df.columns:
            print(f"Warning: Parameter column {param_col} not found")
            return
        
        # Find the metric column (use total if available, otherwise first population)
        metric_col = None
        if f'time_series_total_{metric}' in self.time_series_data:
            metric_col = f'time_series_total_{metric}'
        else:
            # Find first available population
            for pop in self.population_names:
                if f'time_series_{pop}_{metric}' in self.time_series_data:
                    metric_col = f'time_series_{pop}_{metric}'
                    break
        
        if metric_col is None:
            print(f"Warning: No metric column found for {metric}")
            return
        
        # Get parameter values and metric values
        param_values = self.results_df[param_col].values
        
        # Get metric values at specified time point
        if time_point is None:
            # Use final time point
            time_point = -1
        
        metric_values = []
        for series in self.time_series_data[metric_col]:
            if series and len(series) > abs(time_point):
                metric_values.append(series[time_point])
            else:
                metric_values.append(np.nan)
        
        metric_values = np.array(metric_values)
        
        # Remove NaN values
        valid_mask = ~np.isnan(metric_values)
        if not np.any(valid_mask):
            print("No valid data points found")
            return
        
        param_values = param_values[valid_mask]
        metric_values = metric_values[valid_mask]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Scatter plot
        scatter = ax.scatter(param_values, metric_values, alpha=0.7, s=50)
        
        # Add trend line
        if len(param_values) > 1:
            z = np.polyfit(param_values, metric_values, 1)
            p = np.poly1d(z)
            ax.plot(param_values, p(param_values), "r--", alpha=0.8, linewidth=2)
        
        # Customize plot
        ax.set_xlabel(parameter_name.replace('_', ' ').title())
        ax.set_ylabel(f'{metric.replace("_", " ").title()}')
        ax.set_title(f'{parameter_name.replace("_", " ").title()} Sensitivity Analysis')
        ax.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        if len(param_values) > 1:
            corr = np.corrcoef(param_values, metric_values)[0, 1]
            ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                   transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_plot and output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            filename = f'{parameter_name}_sensitivity_{metric}.png'
            plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {output_path / filename}")
        
        plt.show()
        
    def plot_parameter_correlation_matrix(self, output_dir: Optional[str] = None, 
                                       save_plot: bool = False,
                                       figsize: Tuple[int, int] = (14, 12)):
        """
        Plot correlation matrix between parameters.
        
        Args:
            output_dir: Directory to save plots
            save_plot: Whether to save the plot
            figsize: Figure size
        """
        # Get numeric parameter columns
        numeric_params = []
        for col in self.parameter_columns:
            try:
                # Check if column contains numeric data
                pd.to_numeric(self.results_df[col], errors='raise')
                numeric_params.append(col)
            except:
                continue
        
        if len(numeric_params) < 2:
            print("Need at least 2 numeric parameters for correlation matrix")
            return
        
        # Calculate correlation matrix
        param_data = self.results_df[numeric_params].values
        corr_matrix = np.corrcoef(param_data.T)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        
        # Add correlation values as text
        for i in range(len(numeric_params)):
            for j in range(len(numeric_params)):
                text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=10)
        
        # Customize plot
        ax.set_xticks(range(len(numeric_params)))
        ax.set_yticks(range(len(numeric_params)))
        ax.set_xticklabels([col.replace('param_', '') for col in numeric_params], rotation=45, ha='right')
        ax.set_yticklabels([col.replace('param_', '') for col in numeric_params])
        ax.set_title('Parameter Correlation Matrix')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation Coefficient')
        
        plt.tight_layout()
        
        if save_plot and output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            filename = 'parameter_correlation_matrix.png'
            plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {output_path / filename}")
        
        plt.show()
        
    def plot_experiment_comparison(self, experiment_indices: List[int], 
                                 metric: str = 'total_cells',
                                 output_dir: Optional[str] = None, save_plot: bool = False,
                                 figsize: Tuple[int, int] = (12, 8)):
        """
        Compare specific experiments side by side.
        
        Args:
            experiment_indices: List of experiment indices to compare
            metric: Metric to compare ('total_cells' or 'radius')
            output_dir: Directory to save plots
            save_plot: Whether to save the plot
            figsize: Figure size
        """
        # Get time points
        time_points = self.get_time_points()
        if len(time_points) == 0:
            print("No time series data available")
            return
        
        # Find the metric column
        metric_col = None
        if f'time_series_total_{metric}' in self.time_series_data:
            metric_col = f'time_series_total_{metric}'
        else:
            # Find first available population
            for pop in self.population_names:
                if f'time_series_{pop}_{metric}' in self.time_series_data:
                    metric_col = f'time_series_{pop}_{metric}'
                    break
        
        if metric_col is None:
            print(f"Warning: No metric column found for {metric}")
            return
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot each selected experiment
        colors = plt.cm.Set1(np.linspace(0, 1, len(experiment_indices)))
        
        for i, exp_idx in enumerate(experiment_indices):
            if exp_idx < len(self.results_df):
                # Get the time series for this experiment
                series = self.time_series_data[metric_col][exp_idx]
                
                if series and len(series) == len(time_points):
                    # Get parameter values for labeling
                    param_info = []
                    for col in self.parameter_columns[:3]:  # Show first 3 parameters
                        param_name = col.replace('param_', '')
                        param_value = self.results_df.iloc[exp_idx][col]
                        param_info.append(f"{param_name}={param_value:.3g}")
                    
                    label = f"Exp {exp_idx}: {', '.join(param_info)}"
                    
                    ax.plot(time_points, series, color=colors[i], linewidth=2, 
                           label=label, marker='o', markersize=4)
        
        # Customize plot
        ax.set_xlabel('Time')
        ax.set_ylabel(f'{metric.replace("_", " ").title()}')
        ax.set_title(f'Experiment Comparison: {metric.replace("_", " ").title()}')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot and output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            filename = f'experiment_comparison_{metric}.png'
            plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {output_path / filename}")
        
        plt.show()
        
    def create_summary_report(self, output_dir: Optional[str] = None) -> str:
        """
        Create a comprehensive summary report of the parameter sweep.
        
        Args:
            output_dir: Directory to save the report
            
        Returns:
            Report text
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("PARAMETER SWEEP ANALYSIS REPORT")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # Basic statistics
        report_lines.append("BASIC STATISTICS:")
        report_lines.append(f"  Total experiments: {len(self.results_df)}")
        report_lines.append(f"  Parameters varied: {len(self.parameter_columns)}")
        report_lines.append(f"  Populations analyzed: {len(self.population_names)}")
        report_lines.append(f"  Time series length: {self.get_time_series_length()}")
        report_lines.append("")
        
        # Parameter summary
        report_lines.append("PARAMETER SUMMARY:")
        for col in self.parameter_columns:
            param_name = col.replace('param_', '')
            param_data = self.results_df[col]
            
            try:
                # Try to get numeric statistics
                numeric_data = pd.to_numeric(param_data, errors='coerce')
                if not numeric_data.isna().all():
                    report_lines.append(f"  {param_name}:")
                    report_lines.append(f"    Range: {numeric_data.min():.3g} to {numeric_data.max():.3g}")
                    report_lines.append(f"    Mean: {numeric_data.mean():.3g}")
                    report_lines.append(f"    Std: {numeric_data.std():.3g}")
                else:
                    report_lines.append(f"  {param_name}: Non-numeric parameter")
            except:
                report_lines.append(f"  {param_name}: Non-numeric parameter")
        report_lines.append("")
        
        # Population summary
        report_lines.append("POPULATION SUMMARY:")
        for pop in self.population_names:
            report_lines.append(f"  {pop}:")
            
            # Check for total cells data
            total_cells_col = f'time_series_{pop}_total_cells'
            if total_cells_col in self.time_series_data:
                # Get final values
                final_values = []
                for series in self.time_series_data[total_cells_col]:
                    if series:
                        final_values.append(series[-1])
                
                if final_values:
                    report_lines.append(f"    Final total cells: {np.mean(final_values):.3g} ± {np.std(final_values):.3g}")
            
            # Check for radius data
            radius_col = f'time_series_{pop}_radius'
            if radius_col in self.time_series_data:
                # Get final values
                final_values = []
                for series in self.time_series_data[radius_col]:
                    if series:
                        final_values.append(series[-1])
                
                if final_values:
                    report_lines.append(f"    Final radius: {np.mean(final_values):.3g} ± {np.std(final_values):.3g}")
        
        report_lines.append("")
        report_lines.append("=" * 60)
        
        report_text = "\n".join(report_lines)
        
        # Save report if output directory specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            report_file = output_path / "parameter_sweep_report.txt"
            with open(report_file, 'w') as f:
                f.write(report_text)
            print(f"Report saved to: {report_file}")
        
        return report_text
    
    def plot_all_populations(self, metric: str = 'total_cells', 
                           output_dir: Optional[str] = None, save_plot: bool = False,
                           figsize: Tuple[int, int] = (15, 10)):
        """
        Plot all populations on the same graph for comparison.
        
        Args:
            metric: Metric to plot ('total_cells' or 'radius')
            output_dir: Directory to save plots
            save_plot: Whether to save the plot
            figsize: Figure size
        """
        # Get time points
        time_points = self.get_time_points()
        if len(time_points) == 0:
            print("No time series data available")
            return
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot each population
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.population_names)))
        
        for i, pop in enumerate(self.population_names):
            column_name = f'time_series_{pop}_{metric}'
            
            if column_name in self.time_series_data:
                # Calculate mean across experiments
                mean_series = []
                for t in range(len(time_points)):
                    values_at_t = []
                    for series in self.time_series_data[column_name]:
                        if series and len(series) > t:
                            values_at_t.append(series[t])
                    if values_at_t:
                        mean_series.append(np.mean(values_at_t))
                    else:
                        mean_series.append(0)
                
                # Plot mean line
                ax.plot(time_points, mean_series, color=colors[i], linewidth=3, 
                       label=pop, marker='o', markersize=6)
                
                # Add shaded region for standard deviation if multiple experiments
                if len(self.results_df) > 1:
                    std_series = []
                    for t in range(len(time_points)):
                        values_at_t = []
                        for series in self.time_series_data[column_name]:
                            if series and len(series) > t:
                                values_at_t.append(series[t])
                        if values_at_t:
                            std_series.append(np.std(values_at_t))
                        else:
                            std_series.append(0)
                    
                    # Plot shaded region
                    ax.fill_between(time_points, 
                                  np.array(mean_series) - np.array(std_series),
                                  np.array(mean_series) + np.array(std_series),
                                  color=colors[i], alpha=0.2)
        
        # Customize plot
        ax.set_xlabel('Time')
        ax.set_ylabel(f'{metric.replace("_", " ").title()}')
        ax.set_title(f'Population Comparison: {metric.replace("_", " ").title()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot and output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            filename = f'all_populations_{metric}.png'
            plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {output_path / filename}")
        
        plt.show()


def main():
    """Main function to demonstrate the analyzer."""
    # Example usage
    excel_file = "laboratory/parameter_sweeps/experiment_results.xlsx"
    
    if not Path(excel_file).exists():
        print(f"Excel file not found: {excel_file}")
        print("Please run the parameter sweep first to generate results.")
        return
    
    # Create analyzer
    analyzer = ParameterSweepAnalyzer(excel_file)
    
    # Create output directory for plots
    output_dir = "laboratory/parameter_sweeps/analysis_plots"
    
    print("\n" + "="*60)
    print("PARAMETER SWEEP ANALYSIS")
    print("="*60)
    
    # Generate summary report
    print("\nGenerating summary report...")
    report = analyzer.create_summary_report(output_dir)
    print(report)
    
    # Create various plots
    print("\nCreating plots...")
    
    # Plot total evolution
    print("Plotting total evolution...")
    analyzer.plot_total_evolution('total_cells', output_dir, save_plot=True)
    analyzer.plot_total_evolution('radius', output_dir, save_plot=True)
    
    # Plot individual populations
    print("Plotting individual populations...")
    for pop in analyzer.population_names:
        print(f"  Plotting {pop}...")
        analyzer.plot_population_evolution(pop, 'total_cells', output_dir, save_plot=True)
        analyzer.plot_population_evolution(pop, 'radius', output_dir, save_plot=True)
    
    # Plot all populations together
    print("Plotting population comparison...")
    analyzer.plot_all_populations('total_cells', output_dir, save_plot=True)
    analyzer.plot_all_populations('radius', output_dir, save_plot=True)
    
    # Parameter sensitivity analysis (only numeric parameters)
    print("Creating parameter sensitivity plots...")
    numeric_params = []
    for param_col in analyzer.parameter_columns:
        try:
            # Check if parameter is numeric
            pd.to_numeric(analyzer.results_df[param_col], errors='raise')
            numeric_params.append(param_col)
        except:
            continue
    
    # Analyze first 5 numeric parameters
    for param_col in numeric_params[:5]:
        param_name = param_col.replace('param_', '')
        print(f"  Analyzing {param_name}...")
        analyzer.plot_parameter_sensitivity(param_name, 'total_cells', output_dir=output_dir, save_plot=True)
        analyzer.plot_parameter_sensitivity(param_name, 'radius', output_dir=output_dir, save_plot=True)
    
    # Parameter correlation matrix
    print("Creating parameter correlation matrix...")
    analyzer.plot_parameter_correlation_matrix(output_dir, save_plot=True)
    
    # Experiment comparison (first 3 experiments)
    if len(analyzer.results_df) >= 3:
        print("Creating experiment comparison...")
        analyzer.plot_experiment_comparison([0, 1, 2], 'total_cells', output_dir, save_plot=True)
        analyzer.plot_experiment_comparison([0, 1, 2], 'radius', output_dir, save_plot=True)
    
    print(f"\nAnalysis complete! All plots saved to: {output_dir}")
    print("\nYou can now:")
    print("1. Examine the plots to understand parameter effects")
    print("2. Use the data for cost function analysis")
    print("3. Identify optimal parameter regions")
    print("4. Plan additional parameter sweeps")


if __name__ == "__main__":
    main()
