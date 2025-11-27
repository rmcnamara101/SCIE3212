#!/usr/bin/env python3
"""
Single Parameter Sweep Analysis Script

This script analyzes single-parameter sweep results by loading CSV files organized by parameter
and creating plots showing how adjusting each parameter affects the observables.

Each parameter gets its own subdirectory with CSV files for different parameter values,
allowing for easy visualization of parameter sensitivity.

Author: Riley Jae McNamara
Date: 2025-11-26
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
from typing import Dict, List, Optional, Tuple

# Add project root to path
if sys.platform == "darwin":
    proj = Path(__file__).parent
else:
    proj = Path(__file__).parent

sys.path.insert(0, str(proj))


class SingleParameterSweepAnalyzer:
    """
    Analyzer for single-parameter sweep results.
    Creates plots showing how each parameter affects observables.
    """
    
    def __init__(self, sweep_dir):
        """
        Initialize the analyzer.
        
        Args:
            sweep_dir: Path to single parameter sweep directory
        """
        self.sweep_dir = Path(sweep_dir)
        
        # Load sweep summary if available
        self.summary = self.load_summary()
        
        # Load simulation data organized by parameter
        self.data_by_parameter = {}
        self.load_simulation_data()
    
    def load_summary(self):
        """Load the sweep summary YAML file if it exists."""
        summary_path = self.sweep_dir / "single_parameter_sweep_summary.yaml"
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                return yaml.safe_load(f)
        return None
    
    def load_simulation_data(self):
        """Load all simulation CSV files organized by parameter."""
        print(f"Loading simulation data from {self.sweep_dir}...")
        
        # If we have a summary, use it to find parameters
        if self.summary and 'results_by_parameter' in self.summary:
            parameters = list(self.summary['results_by_parameter'].keys())
        else:
            # Otherwise, find parameter directories
            parameters = []
            for item in self.sweep_dir.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    # Convert directory name back to parameter path
                    param_path = item.name.replace('_', '.')
                    parameters.append(param_path)
        
        print(f"Found {len(parameters)} parameters to analyze")
        
        for param_path in parameters:
            # Get parameter directory name (dots replaced with underscores)
            param_name_safe = param_path.replace('.', '_')
            param_dir = self.sweep_dir / param_name_safe
            
            if not param_dir.exists():
                print(f"Warning: Parameter directory {param_dir} not found")
                continue
            
            # Find all CSV files in this parameter's directory
            csv_files = sorted(param_dir.glob("observables_*.csv"))
            
            if len(csv_files) == 0:
                print(f"Warning: No CSV files found in {param_dir}")
                continue
            
            print(f"\nLoading data for parameter: {param_path}")
            print(f"  Found {len(csv_files)} CSV files")
            
            param_data = []
            
            for csv_file in csv_files:
                try:
                    # Extract parameter value from filename or metadata
                    param_value = self.extract_parameter_value(csv_file, param_path)
                    
                    # Load CSV file
                    df = self.load_observables_csv(csv_file)
                    
                    if df is not None:
                        param_data.append({
                            'parameter_value': param_value,
                            'observables': df,
                            'file_path': csv_file
                        })
                
                except Exception as e:
                    print(f"  Warning: Could not load {csv_file.name}: {e}")
            
            # Sort by parameter value
            param_data.sort(key=lambda x: x['parameter_value'])
            
            if len(param_data) > 0:
                self.data_by_parameter[param_path] = param_data
                print(f"  Successfully loaded {len(param_data)} runs")
    
    def extract_parameter_value(self, csv_file, param_path):
        """
        Extract parameter value from CSV filename or metadata.
        
        Args:
            csv_file: Path to CSV file
            param_path: Parameter path (e.g., 'populations.Tumour.dynamics.lambda')
        
        Returns:
            Parameter value (float or int)
        """
        # Try to extract from filename first
        # Format: observables_{param_name_safe}_{value_str}.csv
        filename = csv_file.stem
        param_name_safe = param_path.replace('.', '_')
        
        if param_name_safe in filename:
            # Extract value string after parameter name
            parts = filename.split(param_name_safe)
            if len(parts) > 1:
                value_str = parts[-1].lstrip('_').replace('neg', '-')
                try:
                    # Try to convert to number
                    if '.' in value_str:
                        return float(value_str)
                    else:
                        return int(value_str)
                except ValueError:
                    pass
        
        # Fallback: try to extract from metadata in CSV
        try:
            with open(csv_file, 'r') as f:
                lines = f.readlines()
            
            # Look for parameter in metadata
            param_key = f"Param_{param_path.replace('.', '_')}"
            for line in lines:
                if line.strip().startswith(param_key + ','):
                    value_str = line.split(',', 1)[1].strip()
                    try:
                        if '.' in value_str:
                            return float(value_str)
                        else:
                            return int(value_str)
                    except ValueError:
                        pass
        except Exception:
            pass
        
        # Last resort: use index from sorted files
        return None
    
    def load_observables_csv(self, csv_file):
        """
        Load observables from CSV file, handling metadata section.
        
        Args:
            csv_file: Path to CSV file
        
        Returns:
            DataFrame with observables, or None if failed
        """
        try:
            # Find where observables data starts
            with open(csv_file, 'r') as f:
                lines = f.readlines()
            
            obs_start_idx = None
            for i, line in enumerate(lines):
                if line.strip().startswith('# Observables Data'):
                    # Find the next non-comment, non-blank line (should be header)
                    for j in range(i+1, len(lines)):
                        if lines[j].strip() and not lines[j].strip().startswith('#'):
                            obs_start_idx = j
                            break
                    break
            
            if obs_start_idx is None:
                # Fallback: try reading normally
                df = pd.read_csv(csv_file, comment='#')
            else:
                df = pd.read_csv(csv_file, skiprows=obs_start_idx)
            
            return df
        
        except Exception as e:
            print(f"    Error loading {csv_file.name}: {e}")
            return None
    
    def plot_observable_vs_parameter(self, param_path, observable_name, 
                                     time_point=None, save_plot=True, 
                                     show_plot=False, output_dir=None,
                                     figsize=(10, 6)):
        """
        Plot how an observable changes with parameter value.
        
        Args:
            param_path: Parameter path (e.g., 'populations.Tumour.dynamics.lambda')
            observable_name: Name of observable (e.g., 'Total_Radius')
            time_point: Time point to plot (None = plot final value, or specify Step/Time value)
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            output_dir: Directory to save plots
            figsize: Figure size
        """
        if param_path not in self.data_by_parameter:
            print(f"Error: Parameter {param_path} not found in data")
            return
        
        param_data = self.data_by_parameter[param_path]
        
        # Extract parameter values and observable values
        param_values = []
        observable_values = []
        
        for data in param_data:
            if data['parameter_value'] is None:
                continue
            
            df = data['observables']
            
            if observable_name not in df.columns:
                continue
            
            if time_point is None:
                # Use final value
                if 'Step' in df.columns:
                    final_idx = df['Step'].idxmax()
                else:
                    final_idx = df.index[-1]
                obs_value = df.loc[final_idx, observable_name]
            else:
                # Find time point
                if 'Step' in df.columns:
                    mask = df['Step'] == time_point
                elif 'Time' in df.columns:
                    mask = np.isclose(df['Time'], time_point, atol=0.1)
                else:
                    continue
                
                if not mask.any():
                    continue
                
                obs_value = df.loc[mask.idxmax(), observable_name]
            
            if pd.notna(obs_value):
                param_values.append(data['parameter_value'])
                observable_values.append(obs_value)
        
        if len(param_values) == 0:
            print(f"Warning: No data found for {observable_name} vs {param_path}")
            return
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Sort by parameter value for clean line plot
        sorted_indices = np.argsort(param_values)
        sorted_params = [param_values[i] for i in sorted_indices]
        sorted_obs = [observable_values[i] for i in sorted_indices]
        
        ax.plot(sorted_params, sorted_obs, 'o-', linewidth=2, markersize=8, 
               color='blue', alpha=0.7)
        
        # Format
        param_display = param_path.split('.')[-1]  # Just the parameter name
        obs_display = observable_name.replace('_', ' ')
        
        time_label = ""
        if time_point is not None:
            time_label = f" at Step/Time {time_point}"
        
        ax.set_xlabel(f'{param_display}', fontsize=12)
        ax.set_ylabel(f'{obs_display}', fontsize=12)
        ax.set_title(f'{obs_display} vs {param_display}{time_label}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_dir is None:
            output_dir = self.sweep_dir / "parameter_sensitivity_plots"
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if save_plot:
            param_safe = param_path.replace('.', '_')
            obs_safe = observable_name.replace(' ', '_').lower()
            time_suffix = f"_t{time_point}" if time_point is not None else "_final"
            filename = f'{obs_safe}_vs_{param_safe}{time_suffix}.png'
            filepath = output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {filepath}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_observable_time_series_by_parameter(self, param_path, observable_name,
                                                save_plot=True, show_plot=False,
                                                output_dir=None, figsize=(12, 8)):
        """
        Plot time series of an observable for different parameter values.
        
        Args:
            param_path: Parameter path
            observable_name: Name of observable
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            output_dir: Directory to save plots
            figsize: Figure size
        """
        if param_path not in self.data_by_parameter:
            print(f"Error: Parameter {param_path} not found in data")
            return
        
        param_data = self.data_by_parameter[param_path]
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get colormap for different parameter values
        param_values = [d['parameter_value'] for d in param_data if d['parameter_value'] is not None]
        if len(param_values) == 0:
            print(f"Warning: No valid parameter values found for {param_path}")
            return
        
        sorted_params = sorted(set(param_values))
        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_params)))
        
        plotted_count = 0
        for data in param_data:
            if data['parameter_value'] is None:
                continue
            
            df = data['observables']
            
            if observable_name not in df.columns:
                continue
            
            # Get time axis
            if 'Time' in df.columns:
                time_axis = df['Time'].values
            elif 'Step' in df.columns:
                time_axis = df['Step'].values
            else:
                continue
            
            obs_values = df[observable_name].values
            
            # Remove NaN values
            mask = pd.notna(obs_values) & pd.notna(time_axis)
            if not mask.any():
                continue
            
            # Get color for this parameter value
            param_val = data['parameter_value']
            color_idx = sorted_params.index(param_val) if param_val in sorted_params else 0
            color = colors[color_idx]
            
            ax.plot(time_axis[mask], obs_values[mask], 
                   linewidth=2, alpha=0.7, label=f'{param_val:.3f}',
                   color=color)
            plotted_count += 1
        
        if plotted_count == 0:
            print(f"Warning: No data to plot for {observable_name} vs {param_path}")
            return
        
        # Format
        param_display = param_path.split('.')[-1]
        obs_display = observable_name.replace('_', ' ')
        
        ax.set_xlabel('Time / Step', fontsize=12)
        ax.set_ylabel(f'{obs_display}', fontsize=12)
        ax.set_title(f'{obs_display} Time Series vs {param_display}', 
                    fontsize=14, fontweight='bold')
        ax.legend(title=param_display, fontsize=9, loc='best', ncol=2)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_dir is None:
            output_dir = self.sweep_dir / "parameter_sensitivity_plots"
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if save_plot:
            param_safe = param_path.replace('.', '_')
            obs_safe = observable_name.replace(' ', '_').lower()
            filename = f'{obs_safe}_timeseries_vs_{param_safe}.png'
            filepath = output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {filepath}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_all_observables_for_parameter(self, param_path, time_point=None,
                                          save_plot=True, show_plot=False,
                                          output_dir=None, figsize=(16, 12)):
        """
        Create a comprehensive plot showing all observables vs parameter value.
        
        Args:
            param_path: Parameter path
            time_point: Time point to plot (None = final value)
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            output_dir: Directory to save plots
            figsize: Figure size
        """
        if param_path not in self.data_by_parameter:
            print(f"Error: Parameter {param_path} not found in data")
            return
        
        param_data = self.data_by_parameter[param_path]
        
        # Get available observables from first run
        if len(param_data) == 0:
            return
        
        first_df = param_data[0]['observables']
        # Common observables to plot
        observables = ['Total_Radius', 'Inhibited_Radius', 'Necrotic_Radius']
        # Filter to only those that exist
        observables = [obs for obs in observables if obs in first_df.columns]
        
        if len(observables) == 0:
            print(f"Warning: No standard observables found for {param_path}")
            return
        
        # Create subplot layout
        n_plots = len(observables)
        n_cols = 2
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for plot_idx, observable in enumerate(observables):
            ax = axes[plot_idx]
            
            # Extract data
            param_values = []
            observable_values = []
            
            for data in param_data:
                if data['parameter_value'] is None:
                    continue
                
                df = data['observables']
                
                if observable not in df.columns:
                    continue
                
                if time_point is None:
                    if 'Step' in df.columns:
                        final_idx = df['Step'].idxmax()
                    else:
                        final_idx = df.index[-1]
                    obs_value = df.loc[final_idx, observable]
                else:
                    if 'Step' in df.columns:
                        mask = df['Step'] == time_point
                    elif 'Time' in df.columns:
                        mask = np.isclose(df['Time'], time_point, atol=0.1)
                    else:
                        continue
                    
                    if not mask.any():
                        continue
                    
                    obs_value = df.loc[mask.idxmax(), observable]
                
                if pd.notna(obs_value):
                    param_values.append(data['parameter_value'])
                    observable_values.append(obs_value)
            
            if len(param_values) > 0:
                # Sort for clean plot
                sorted_indices = np.argsort(param_values)
                sorted_params = [param_values[i] for i in sorted_indices]
                sorted_obs = [observable_values[i] for i in sorted_indices]
                
                ax.plot(sorted_params, sorted_obs, 'o-', linewidth=2, markersize=6, 
                       color='blue', alpha=0.7)
            
            # Format subplot
            obs_display = observable.replace('_', ' ')
            param_display = param_path.split('.')[-1]
            ax.set_xlabel(param_display, fontsize=10)
            ax.set_ylabel(obs_display, fontsize=10)
            ax.set_title(obs_display, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)
        
        param_display = param_path.split('.')[-1]
        time_label = f" at Step/Time {time_point}" if time_point is not None else " (Final Values)"
        fig.suptitle(f'Observables vs {param_display}{time_label}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if output_dir is None:
            output_dir = self.sweep_dir / "parameter_sensitivity_plots"
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if save_plot:
            param_safe = param_path.replace('.', '_')
            time_suffix = f"_t{time_point}" if time_point is not None else "_final"
            filename = f'all_observables_vs_{param_safe}{time_suffix}.png'
            filepath = output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved comprehensive plot to {filepath}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def create_all_plots(self, observables=None, time_points=None,
                        save_plot=True, show_plot=False, output_dir=None):
        """
        Create all plots for all parameters and observables.
        
        Args:
            observables: List of observables to plot (None = use defaults)
            time_points: List of time points to plot (None = final values only)
            save_plot: Whether to save plots
            show_plot: Whether to display plots
            output_dir: Directory to save plots
        """
        if observables is None:
            observables = ['Total_Radius', 'Inhibited_Radius', 'Necrotic_Radius']
        
        if time_points is None:
            time_points = [None]  # Just final values
        
        print(f"\n{'='*60}")
        print("Creating parameter sensitivity plots...")
        print(f"{'='*60}")
        
        for param_path in self.data_by_parameter.keys():
            print(f"\nProcessing parameter: {param_path}")
            
            # Comprehensive plot for all observables
            for time_point in time_points:
                self.plot_all_observables_for_parameter(
                    param_path, time_point=time_point,
                    save_plot=save_plot, show_plot=show_plot,
                    output_dir=output_dir
                )
            
            # Individual observable plots
            for observable in observables:
                # Time series plots
                self.plot_observable_time_series_by_parameter(
                    param_path, observable,
                    save_plot=save_plot, show_plot=show_plot,
                    output_dir=output_dir
                )
                
                # Point plots at different time points
                for time_point in time_points:
                    self.plot_observable_vs_parameter(
                        param_path, observable, time_point=time_point,
                        save_plot=save_plot, show_plot=show_plot,
                        output_dir=output_dir
                    )
        
        print(f"\n{'='*60}")
        print("Plot generation complete!")
        print(f"{'='*60}")


def main():
    """
    Main function for analyzing single parameter sweep results.
    
    Configure the analysis by modifying the variables below.
    """
    # ============================================================================
    # CONFIGURATION VARIABLES - Modify these to change analysis behavior
    # ============================================================================
    
    # Path to single parameter sweep directory
    SWEEP_DIR = "/Users/rileymcnamara/CODE/2025/silicokit/laboratory/parameter_sweeps/single_parameter_sweep_20251126_201010"
    
    # Observables to plot (None = use defaults: Total_Radius, Inhibited_Radius, Necrotic_Radius)
    OBSERVABLES = None
    
    # Time points to plot (None = final values only, or specify list like [5, 10, 15, 20])
    # Use None for final values, or specify Step values
    TIME_POINTS = None
    
    # Output directory for plots (None = save to sweep_dir/parameter_sensitivity_plots)
    OUTPUT_DIR = None
    
    # Whether to display plots (True = show plots, False = save only)
    SHOW_PLOTS = False
    
    # ============================================================================
    # END OF CONFIGURATION
    # ============================================================================
    
    # Initialize analyzer
    analyzer = SingleParameterSweepAnalyzer(sweep_dir=SWEEP_DIR)
    
    # Create all plots
    analyzer.create_all_plots(
        observables=OBSERVABLES,
        time_points=TIME_POINTS,
        save_plot=True,
        show_plot=SHOW_PLOTS,
        output_dir=OUTPUT_DIR
    )
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()

