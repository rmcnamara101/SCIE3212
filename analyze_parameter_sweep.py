#!/usr/bin/env python3
"""
Parameter Sweep Analysis Script

This script analyzes parameter sweep results by loading CSV files from a sweep directory
and comparing them with validation data from the Browning paper. It generates comparison
plots showing simulation results alongside validation data.

Each CSV file contains both observables (time series data) and configuration parameters,
allowing for easy identification of which simulation produced which results.

Author: Riley Jae McNamara
Date: 2025-02-19
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import yaml
from typing import Dict, List, Optional, Tuple

# Add project root to path
if sys.platform == "darwin":
    proj = Path(__file__).parent
else:
    proj = Path(__file__).parent

sys.path.insert(0, str(proj))


class ParameterSweepAnalyzer:
    """
    Analyzer for parameter sweep results with validation data comparison.
    """
    
    def __init__(self, sweep_dir, validation_data_path=None, load_validation=True):
        """
        Initialize the analyzer.
        
        Args:
            sweep_dir: Path to parameter sweep directory containing CSV files
            validation_data_path: Path to validation data Excel file (defaults to Browning paper data)
            load_validation: Whether to load validation data (set False to skip)
        """
        self.sweep_dir = Path(sweep_dir)
        
        # Load validation data only if requested
        self.validation_data = None
        if load_validation:
            # Default validation data path
            if validation_data_path is None:
                validation_data_path = proj / "laboratory" / "data" / "Browning_Paper" / "Organised_Data.xlsx"
            self.validation_data_path = Path(validation_data_path)
            
            if self.validation_data_path.exists():
                self.load_validation_data()
            else:
                print(f"Warning: Validation data not found at {self.validation_data_path}")
                print("Plots will show only simulation data.")
        else:
            print("Validation data loading skipped - plotting simulation data only")
        
        # Load simulation data
        self.simulation_data = {}
        self.load_simulation_data()
    
    def load_validation_data(self):
        """Load validation data from Excel file."""
        print(f"Loading validation data from {self.validation_data_path}...")
        try:
            # Try reading Excel file
            self.validation_data = pd.read_excel(self.validation_data_path, sheet_name=None)
            print(f"Loaded validation data with sheets: {list(self.validation_data.keys())}")
            
            # Print column info for debugging
            for sheet_name, df in self.validation_data.items():
                print(f"  Sheet '{sheet_name}': {len(df)} rows, columns: {list(df.columns)[:10]}...")
                
        except Exception as e:
            print(f"Error loading validation data: {e}")
            # Try CSV as fallback
            csv_path = self.validation_data_path.with_suffix('.csv')
            if csv_path.exists():
                print(f"Trying CSV file: {csv_path}")
                self.validation_data = {'data': pd.read_csv(csv_path)}
            else:
                print(f"No validation data found at {self.validation_data_path}")
    
    def load_simulation_data(self):
        """Load all simulation CSV files from the sweep directory."""
        print(f"Loading simulation data from {self.sweep_dir}...")
        
        # Find all CSV files
        csv_files = sorted(self.sweep_dir.glob("observables_run_*.csv"))
        
        if len(csv_files) == 0:
            print(f"Warning: No CSV files found in {self.sweep_dir}")
            # Also check for the old naming pattern
            csv_files = sorted(self.sweep_dir.glob("observables_data.csv"))
            if len(csv_files) == 0:
                return
        
        print(f"Found {len(csv_files)} CSV files")
        
        for csv_file in csv_files:
            try:
                # Extract run ID from filename (e.g., observables_run_001.csv -> 1)
                if 'run_' in csv_file.stem:
                    run_id = int(csv_file.stem.split('_')[-1])
                else:
                    # Fallback: use file index if no run ID in filename
                    run_id = len(self.simulation_data) + 1
                
                # Load CSV file (contains both observables and config)
                df = pd.read_csv(csv_file)
                
                # Separate config columns from observable columns
                # Config columns start with 'Config_' or 'Param_'
                config_cols = [c for c in df.columns if c.startswith('Config_') or c.startswith('Param_')]
                obs_cols = [c for c in df.columns if c not in config_cols]
                
                # Store data - observables dataframe contains all columns (config will be repeated)
                # but we can filter if needed for plotting
                self.simulation_data[run_id] = {
                    'observables': df,  # Full dataframe with all columns
                    'config': {col: df[col].iloc[0] for col in config_cols} if config_cols else None,  # First row config values
                    'file_path': csv_file
                }
                    
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
        
        print(f"Successfully loaded {len(self.simulation_data)} simulation runs")
    
    def map_validation_columns(self, validation_df):
        """
        Map validation data columns to simulation observables.
        
        This is a flexible mapping that handles common column name variations.
        """
        column_mapping = {
            'Time': ['Time', 'time', 't', 'T'],
            'Total_Radius': ['Radius', 'radius', 'r', 'total_radius', 'Total_Radius'],
            'Total_Density': ['Density', 'density', 'total_density', 'Total_Density'],
            'Volume': ['Volume', 'volume', 'vol', 'V'],
            'Inhibited_Radius': ['Inhibited_Radius', 'inhibited_radius', 'inhibited', 'quiescent'],
            'Necrotic_Core_Radius': ['Necrotic_Core_Radius', 'necrotic_radius', 'necrotic', 'necrotic_core']
        }
        
        mapped_data = {}
        for sim_col, possible_names in column_mapping.items():
            for name in possible_names:
                if name in validation_df.columns:
                    mapped_data[sim_col] = name
                    break
        
        return mapped_data
    
    def find_validation_column(self, observable_name):
        """
        Find matching column in validation data for a given observable.
        
        Returns:
            Tuple of (sheet_name, column_name, time_column) or None if not found
        """
        if self.validation_data is None:
            return None
        
        # Try exact match first
        for sheet_name, sheet_df in self.validation_data.items():
            if observable_name in sheet_df.columns:
                time_col = None
                for time_name in ['Time', 'time', 't', 'T']:
                    if time_name in sheet_df.columns:
                        time_col = time_name
                        break
                if time_col:
                    return (sheet_name, observable_name, time_col)
        
        # Try fuzzy matching
        observable_lower = observable_name.lower()
        possible_mappings = {
            'total_radius': ['radius', 'r', 'total radius', 'tumor radius'],
            'total_density': ['density', 'cell density', 'total density'],
            'inhibited_radius': ['inhibited', 'quiescent', 'inhibited radius'],
            'necrotic_core_radius': ['necrotic', 'necrotic radius', 'necrotic core']
        }
        
        for key, possible_names in possible_mappings.items():
            if key in observable_lower:
                for sheet_name, sheet_df in self.validation_data.items():
                    for col in sheet_df.columns:
                        col_lower = str(col).lower()
                        for name in possible_names:
                            if name in col_lower:
                                time_col = None
                                for time_name in ['Time', 'time', 't', 'T']:
                                    if time_name in sheet_df.columns:
                                        time_col = time_name
                                        break
                                if time_col:
                                    return (sheet_name, col, time_col)
        
        return None
    
    def plot_comparison(self, observable_name, ylabel=None, save_plot=True, 
                       show_plot=True, figsize=(12, 8), output_dir=None):
        """
        Plot comparison between simulation and validation data for a specific observable.
        
        Args:
            observable_name: Name of observable to plot (e.g., 'Total_Radius', 'Total_Density')
            ylabel: Y-axis label (defaults to observable_name)
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            figsize: Figure size
            output_dir: Directory to save plots
        """
        if output_dir is None:
            output_dir = self.sweep_dir / "comparison_plots"
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot validation data if available
        validation_plotted = False
        validation_info = self.find_validation_column(observable_name)
        if validation_info is not None:
            sheet_name, val_col, time_col = validation_info
            sheet_df = self.validation_data[sheet_name]
            
            # Remove any NaN or invalid values
            mask = pd.notna(sheet_df[val_col]) & pd.notna(sheet_df[time_col])
            time_data = sheet_df.loc[mask, time_col]
            value_data = sheet_df.loc[mask, val_col]
            
            if len(time_data) > 0:
                ax.plot(time_data, value_data, 
                       'ko-', linewidth=2.5, markersize=6, label='Validation Data',
                       alpha=0.8, zorder=10)
                validation_plotted = True
        
        # Plot simulation data
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.simulation_data)))
        for idx, (run_id, data) in enumerate(sorted(self.simulation_data.items())):
            observables_df = data['observables']
            
            if observable_name in observables_df.columns and 'Time' in observables_df.columns:
                # Remove any NaN or invalid values
                mask = pd.notna(observables_df[observable_name]) & pd.notna(observables_df['Time'])
                time_data = observables_df.loc[mask, 'Time']
                value_data = observables_df.loc[mask, observable_name]
                
                if len(time_data) > 0:
                    ax.plot(time_data, value_data,
                           '--', color=colors[idx], linewidth=1.5, alpha=0.7,
                           label=f'Simulation Run {run_id}', marker='o', markersize=4)
            else:
                if observable_name not in observables_df.columns:
                    print(f"Warning: {observable_name} not found in run {run_id}")
        
        if not validation_plotted and self.validation_data is not None:
            print(f"Note: Validation data not found for {observable_name}")
        
        # Set labels and title
        if ylabel is None:
            ylabel = observable_name.replace('_', ' ')
        
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'{ylabel} Comparison: Simulation vs Validation', fontsize=14)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            safe_name = observable_name.replace(' ', '_').lower()
            filename = f'{safe_name}_comparison.png'
            filepath = output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved comparison plot to {filepath}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_all_comparisons(self, observables=None, save_plot=True, show_plot=False,
                            figsize=(12, 8), output_dir=None):
        """
        Plot comparisons for all common observables.
        
        Args:
            observables: List of observables to plot (defaults to common ones)
            save_plot: Whether to save plots
            show_plot: Whether to display plots
            figsize: Figure size
            output_dir: Directory to save plots
        """
        if observables is None:
            # Default observables to compare
            observables = [
                'Total_Radius',
                'Total_Density',
                'Inhibited_Radius',
                'Necrotic_Core_Radius'
            ]
        
        # Get available observables from first simulation
        if len(self.simulation_data) > 0:
            first_run = next(iter(self.simulation_data.values()))
            available_observables = list(first_run['observables'].columns)
            
            # Filter to only observables that exist
            observables = [obs for obs in observables if obs in available_observables]
        
        print(f"\nPlotting comparisons for {len(observables)} observables...")
        
        for observable in observables:
            try:
                self.plot_comparison(observable, save_plot=save_plot, show_plot=show_plot,
                                   figsize=figsize, output_dir=output_dir)
            except Exception as e:
                print(f"Error plotting {observable}: {e}")
    
    def create_summary_comparison(self, observables=None, save_plot=True, show_plot=True,
                                 figsize=(16, 12), output_dir=None):
        """
        Create a comprehensive comparison plot with multiple subplots.
        
        Args:
            observables: List of observables to include (defaults to key observables)
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            figsize: Figure size
            output_dir: Directory to save plots
        """
        if observables is None:
            observables = [
                'Total_Radius',
                'Total_Density',
                'Inhibited_Radius',
                'Necrotic_Core_Radius'
            ]
        
        # Get available observables from first simulation
        if len(self.simulation_data) > 0:
            first_run = next(iter(self.simulation_data.values()))
            available_observables = list(first_run['observables'].columns)
            observables = [obs for obs in observables if obs in available_observables]
        
        n_plots = len(observables)
        if n_plots == 0:
            print("No observables to plot")
            return
        
        # Calculate subplot layout
        n_cols = 2
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.simulation_data)))
        
        for plot_idx, observable in enumerate(observables):
            ax = axes[plot_idx]
            
            # Plot validation data
            validation_info = self.find_validation_column(observable)
            if validation_info is not None:
                sheet_name, val_col, time_col = validation_info
                sheet_df = self.validation_data[sheet_name]
                
                # Remove any NaN or invalid values
                mask = pd.notna(sheet_df[val_col]) & pd.notna(sheet_df[time_col])
                time_data = sheet_df.loc[mask, time_col]
                value_data = sheet_df.loc[mask, val_col]
                
                if len(time_data) > 0:
                    ax.plot(time_data, value_data,
                           'ko-', linewidth=2.5, markersize=6,
                           label='Validation Data', alpha=0.8, zorder=10)
            
            # Plot simulation data
            for idx, (run_id, data) in enumerate(sorted(self.simulation_data.items())):
                observables_df = data['observables']
                
                if observable in observables_df.columns and 'Time' in observables_df.columns:
                    ax.plot(observables_df['Time'], observables_df[observable],
                           '--', color=colors[idx], linewidth=1.5, alpha=0.7,
                           label=f'Run {run_id}', marker='o', markersize=3)
            
            # Format subplot
            ylabel = observable.replace('_', ' ')
            ax.set_xlabel('Time', fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(ylabel, fontsize=12, fontweight='bold')
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)
        
        fig.suptitle('Simulation vs Validation Data Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if output_dir is None:
            output_dir = self.sweep_dir / "comparison_plots"
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if save_plot:
            filepath = output_dir / "comprehensive_comparison.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved comprehensive comparison plot to {filepath}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def get_observable_statistics(self, observable_name):
        """
        Get statistics for an observable across all simulation runs.
        
        Args:
            observable_name: Name of observable
        
        Returns:
            DataFrame with statistics (mean, std, min, max) at each time point
        """
        all_data = []
        
        for run_id, data in self.simulation_data.items():
            observables_df = data['observables']
            if observable_name in observables_df.columns and 'Time' in observables_df.columns:
                all_data.append({
                    'run_id': run_id,
                    'time': observables_df['Time'].values,
                    'value': observables_df[observable_name].values
                })
        
        if len(all_data) == 0:
            return None
        
        # Find common time points (interpolate if needed)
        all_times = set()
        for d in all_data:
            all_times.update(d['time'])
        common_times = sorted(all_times)
        
        # Interpolate all runs to common time points
        interpolated_values = []
        for d in all_data:
            interp_values = np.interp(common_times, d['time'], d['value'])
            interpolated_values.append(interp_values)
        
        interpolated_values = np.array(interpolated_values)
        
        # Calculate statistics
        stats_df = pd.DataFrame({
            'Time': common_times,
            'Mean': np.mean(interpolated_values, axis=0),
            'Std': np.std(interpolated_values, axis=0),
            'Min': np.min(interpolated_values, axis=0),
            'Max': np.max(interpolated_values, axis=0),
            'Median': np.median(interpolated_values, axis=0)
        })
        
        return stats_df


def main():
    """
    Main function for analyzing parameter sweep results.
    
    Configure the analysis by modifying the variables below.
    """
    # ============================================================================
    # CONFIGURATION VARIABLES - Modify these to change analysis behavior
    # ============================================================================
    
    # Path to parameter sweep directory containing CSV files
    SWEEP_DIR = "/Users/rileymcnamara/CODE/2025/silicokit/laboratory/parameter_sweeps/random_parameter_sweep_20251103_182304"
    
    # Whether to compare with validation data (set to False if validation data structure is unknown)
    USE_VALIDATION_DATA = False
    
    # Path to validation data file (only used if USE_VALIDATION_DATA is True)
    # If None, defaults to Browning paper data
    VALIDATION_DATA_PATH = None
    
    # Observables to plot (None = use default: Total_Radius, Inhibited_Radius, Necrotic_Core_Radius)
    # Can also specify a list like: ['Total_Radius', 'Inhibited_Radius']
    OBSERVABLES = None
    
    # Output directory for plots (None = save to sweep_dir/comparison_plots)
    OUTPUT_DIR = None
    
    # Whether to display plots (True = show plots, False = save only)
    SHOW_PLOTS = False
    
    # Whether to create comprehensive multi-panel plot (True) or individual plots (False)
    COMPREHENSIVE_PLOT = True
    
    # ============================================================================
    # END OF CONFIGURATION
    # ============================================================================
    
    # Initialize analyzer
    if USE_VALIDATION_DATA:
        analyzer = ParameterSweepAnalyzer(
            sweep_dir=SWEEP_DIR,
            validation_data_path=VALIDATION_DATA_PATH
        )
    else:
        # Create analyzer without validation data
        analyzer = ParameterSweepAnalyzer(
            sweep_dir=SWEEP_DIR,
            validation_data_path=None
        )
        # Disable validation data loading
        analyzer.validation_data = None
        print("Validation data comparison disabled - plotting simulation data only")
    
    if COMPREHENSIVE_PLOT:
        # Create comprehensive comparison
        analyzer.create_summary_comparison(
            observables=OBSERVABLES,
            save_plot=True,
            show_plot=SHOW_PLOTS,
            output_dir=OUTPUT_DIR
        )
    else:
        # Create individual plots
        analyzer.plot_all_comparisons(
            observables=OBSERVABLES,
            save_plot=True,
            show_plot=SHOW_PLOTS,
            output_dir=OUTPUT_DIR
        )
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()

