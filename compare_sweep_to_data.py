#!/usr/bin/env python3
"""
Compare Parameter Sweep Results to Experimental Data

This script loads experimental data from the Browning paper Excel file and compares
it to parameter sweep simulation results. It finds the best matching simulations
using RMSE as a cost function, with automatic scaling to account for unit differences.

Author: Riley Jae McNamara
Date: 2025-02-19
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from typing import Dict, List, Optional, Tuple

# Add project root to path
if sys.platform == "darwin":
    proj = Path(__file__).parent
else:
    proj = Path(__file__).parent

sys.path.insert(0, str(proj))


class SweepDataComparator:
    """
    Compare parameter sweep results to experimental data using RMSE with scaling.
    """
    
    def __init__(self, sweep_dir, experimental_data_path=None):
        """
        Initialize the comparator.
        
        Args:
            sweep_dir: Path to parameter sweep directory containing CSV files
            experimental_data_path: Path to experimental data Excel file
        """
        self.sweep_dir = Path(sweep_dir)
        
        # Load experimental data
        if experimental_data_path is None:
            experimental_data_path = proj / "laboratory" / "data" / "Browning_Paper" / "Organised_Data.xlsx"
        self.experimental_data_path = Path(experimental_data_path)
        
        self.experimental_data = self.load_experimental_data()
        
        # Load simulation data
        self.simulation_data = {}
        self.load_simulation_data()
    
    def load_experimental_data(self):
        """
        Load experimental data from Excel file.
        
        Returns:
            Dictionary with keys '10k', '5k', '2k' containing DataFrames with columns:
            'Day', 'Total_Radius', 'Inhibited_Radius', 'Necrotic_Radius'
        """
        print(f"Loading experimental data from {self.experimental_data_path}...")
        
        # Read the Summary sheet
        df = pd.read_excel(self.experimental_data_path, sheet_name='Summary', header=None)
        
        # Extract data for each seeding density
        # 10k: B9 to E18 (columns 1-3, rows 8-16, 0-indexed)
        # 5k: H9 to K18 (columns 7-9, rows 8-16, 0-indexed)
        # 2k: N9 to Q18 (columns 13-15, rows 8-16, 0-indexed)
        
        experimental_data = {}
        
        # 10k seeding density
        data_10k = df.iloc[8:17, 1:4].copy()
        data_10k.columns = ['Day', 'Total_Radius', 'Inhibited_Radius']
        data_10k = data_10k.dropna(subset=['Day'])  # Remove rows with NaN days
        # Convert to numeric, coercing errors to NaN
        data_10k = data_10k.apply(pd.to_numeric, errors='coerce')
        # Add necrotic radius from column E (index 4)
        necrotic_10k = pd.to_numeric(df.iloc[8:17, 4], errors='coerce').values
        data_10k['Necrotic_Radius'] = necrotic_10k[:len(data_10k)]
        experimental_data['10k'] = data_10k.reset_index(drop=True)
        
        # 5k seeding density
        data_5k = df.iloc[8:17, 7:10].copy()
        data_5k.columns = ['Day', 'Total_Radius', 'Inhibited_Radius']
        data_5k = data_5k.dropna(subset=['Day'])
        # Convert to numeric, coercing errors to NaN
        data_5k = data_5k.apply(pd.to_numeric, errors='coerce')
        # Add necrotic radius from column K (index 10)
        necrotic_5k = pd.to_numeric(df.iloc[8:17, 10], errors='coerce').values
        data_5k['Necrotic_Radius'] = necrotic_5k[:len(data_5k)]
        experimental_data['5k'] = data_5k.reset_index(drop=True)
        
        # 2k seeding density
        data_2k = df.iloc[8:17, 13:16].copy()
        data_2k.columns = ['Day', 'Total_Radius', 'Inhibited_Radius']
        data_2k = data_2k.dropna(subset=['Day'])
        # Convert to numeric, coercing errors to NaN
        data_2k = data_2k.apply(pd.to_numeric, errors='coerce')
        # Add necrotic radius from column Q (index 16)
        necrotic_2k = pd.to_numeric(df.iloc[8:17, 16], errors='coerce').values
        data_2k['Necrotic_Radius'] = necrotic_2k[:len(data_2k)]
        experimental_data['2k'] = data_2k.reset_index(drop=True)
        
        print(f"Loaded experimental data:")
        for density, data in experimental_data.items():
            print(f"  {density}: {len(data)} time points, days {data['Day'].min():.1f} to {data['Day'].max():.1f}")
        
        return experimental_data
    
    def load_simulation_data(self):
        """Load all simulation CSV files from the sweep directory."""
        print(f"\nLoading simulation data from {self.sweep_dir}...")
        
        # Find all CSV files
        csv_files = sorted(self.sweep_dir.glob("observables_run_*.csv"))
        
        if len(csv_files) == 0:
            print(f"Warning: No CSV files found in {self.sweep_dir}")
            return
        
        print(f"Found {len(csv_files)} CSV files")
        
        for csv_file in csv_files:
            try:
                # Extract run ID from filename
                if 'run_' in csv_file.stem:
                    run_id = int(csv_file.stem.split('_')[-1])
                else:
                    run_id = len(self.simulation_data) + 1
                
                # Load CSV file
                df = pd.read_csv(csv_file)
                
                # Extract observables (Step, Time, Total_Radius, Inhibited_Radius, Necrotic_Radius)
                obs_cols = ['Step', 'Time', 'Total_Radius', 'Inhibited_Radius', 'Necrotic_Radius']
                available_cols = [col for col in obs_cols if col in df.columns]
                
                if 'Step' not in df.columns:
                    print(f"Warning: No 'Step' column in {csv_file}")
                    continue
                
                # Create observables dataframe
                observables_df = df[available_cols].copy()
                
                # Store data
                self.simulation_data[run_id] = {
                    'observables': observables_df,
                    'file_path': csv_file,
                    'config': {col: df[col].iloc[0] for col in df.columns 
                              if col.startswith('Config_') or col.startswith('Param_')}
                }
                    
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
        
        print(f"Successfully loaded {len(self.simulation_data)} simulation runs")
    
    def calculate_optimal_scale(self, sim_values, exp_values):
        """
        Find optimal scaling factor to minimize RMSE between simulation and experimental data.
        
        Args:
            sim_values: Simulation values (numpy array)
            exp_values: Experimental values (numpy array)
        
        Returns:
            Optimal scale factor
        """
        # Remove NaN values
        mask = ~(np.isnan(sim_values) | np.isnan(exp_values))
        if mask.sum() < 2:
            return 1.0
        
        sim_clean = sim_values[mask]
        exp_clean = exp_values[mask]
        
        # If all simulation values are zero or very small, return a default scale
        if np.max(np.abs(sim_clean)) < 1e-10:
            return 1.0
        
        # Find optimal scale using least squares: minimize ||exp - scale * sim||^2
        # scale = (exp^T * sim) / (sim^T * sim)
        scale = np.dot(exp_clean, sim_clean) / np.dot(sim_clean, sim_clean)
        
        # Ensure scale is positive and reasonable
        if scale <= 0 or scale > 1e6:
            # Fallback: use ratio of means
            scale = np.mean(exp_clean) / np.mean(sim_clean) if np.mean(sim_clean) > 0 else 1.0
            if scale <= 0 or scale > 1e6:
                scale = 1.0
        
        return scale
    
    def calculate_optimal_scale_all_metrics(self, sim_data, exp_data, metrics):
        """
        Find a single optimal scaling factor across all metrics simultaneously.
        This ensures consistency - all radii for a simulation are scaled by the same factor.
        
        Args:
            sim_data: Simulation observables DataFrame
            exp_data: Experimental data DataFrame
            metrics: List of metric names to include
        
        Returns:
            Optimal scale factor (single value for all metrics)
        """
        all_sim_values = []
        all_exp_values = []
        
        sim_steps = sim_data['Step'].values
        exp_days = exp_data['Day'].values
        
        # Collect matched points from all metrics
        for metric in metrics:
            if metric not in sim_data.columns or metric not in exp_data.columns:
                continue
            
            sim_values = sim_data[metric].values
            exp_values = exp_data[metric].values
            
            # Find matched points (step = day)
            for exp_idx, exp_day in enumerate(exp_days):
                if np.isnan(exp_values[exp_idx]):
                    continue
                
                step_match_idx = np.where(sim_steps == exp_day)[0]
                if len(step_match_idx) > 0:
                    sim_val = sim_values[step_match_idx[0]]
                    if not np.isnan(sim_val):
                        all_sim_values.append(sim_val)
                        all_exp_values.append(exp_values[exp_idx])
        
        # Convert to arrays
        all_sim_values = np.array(all_sim_values)
        all_exp_values = np.array(all_exp_values)
        
        if len(all_sim_values) < 2:
            return 1.0
        
        # Calculate optimal scale using all matched points from all metrics
        return self.calculate_optimal_scale(all_sim_values, all_exp_values)
    
    def calculate_rmse(self, sim_steps, sim_values, exp_days, exp_values, scale=None):
        """
        Calculate RMSE between simulation and experimental data using exact step-to-day matches.
        Each simulation step is treated as 1 day.
        
        Args:
            sim_steps: Simulation step numbers (treated as days)
            sim_values: Simulation values
            exp_days: Experimental day numbers
            exp_values: Experimental values
            scale: Scaling factor for values (if None, will be optimized)
        
        Returns:
            Tuple of (RMSE, optimal_scale)
        """
        # Convert to numpy arrays and ensure numeric types
        sim_steps = np.asarray(sim_steps, dtype=int)
        sim_values = np.asarray(sim_values, dtype=float)
        exp_days = np.asarray(exp_days, dtype=int)
        exp_values = np.asarray(exp_values, dtype=float)
        
        # Remove NaN values
        sim_mask = ~np.isnan(sim_values)
        exp_mask = ~np.isnan(exp_values) & ~np.isnan(exp_days)
        
        sim_steps_clean = sim_steps[sim_mask]
        sim_values_clean = sim_values[sim_mask]
        exp_days_clean = exp_days[exp_mask]
        exp_values_clean = exp_values[exp_mask]
        
        if len(sim_steps_clean) == 0 or len(exp_days_clean) == 0:
            return np.inf, 1.0
        
        # Find exact matches: for each experimental day, find corresponding simulation step
        matched_sim_values = []
        matched_exp_values = []
        
        for exp_idx, exp_day in enumerate(exp_days_clean):
            # Find simulation step that matches this day (step = day)
            step_match_idx = np.where(sim_steps_clean == exp_day)[0]
            
            if len(step_match_idx) > 0:
                # Use the first match (should only be one)
                matched_sim_values.append(sim_values_clean[step_match_idx[0]])
                matched_exp_values.append(exp_values_clean[exp_idx])
        
        # Convert to arrays
        matched_sim_values = np.array(matched_sim_values)
        matched_exp_values = np.array(matched_exp_values)
        
        if len(matched_sim_values) < 2:
            return np.inf, 1.0
        
        # Find optimal scale if not provided
        if scale is None:
            scale = self.calculate_optimal_scale(matched_sim_values, matched_exp_values)
        
        # Calculate RMSE using only matched points
        scaled_sim = scale * matched_sim_values
        rmse = np.sqrt(np.mean((matched_exp_values - scaled_sim)**2))
        
        return rmse, scale
    
    def compare_simulation_to_experiment(self, run_id, density='10k', 
                                         metrics=['Total_Radius', 'Inhibited_Radius', 'Necrotic_Radius']):
        """
        Compare a single simulation to experimental data for a given seeding density.
        Uses a single scale factor for all metrics to ensure consistency.
        
        Args:
            run_id: Simulation run ID
            density: Seeding density ('10k', '5k', or '2k')
            metrics: List of metrics to compare
        
        Returns:
            Dictionary with RMSE values for each metric and a single scale factor
        """
        if run_id not in self.simulation_data:
            return None
        
        if density not in self.experimental_data:
            return None
        
        sim_data = self.simulation_data[run_id]['observables']
        exp_data = self.experimental_data[density]
        
        # Calculate a single optimal scale factor across all metrics
        scale = self.calculate_optimal_scale_all_metrics(sim_data, exp_data, metrics)
        
        results = {
            'scale': scale  # Single scale factor for all metrics
        }
        
        # Calculate RMSE for each metric using the same scale
        for metric in metrics:
            if metric not in sim_data.columns or metric not in exp_data.columns:
                continue
            
            sim_steps = sim_data['Step'].values
            sim_values = sim_data[metric].values
            exp_days = exp_data['Day'].values
            exp_values = exp_data[metric].values
            
            rmse, _ = self.calculate_rmse(sim_steps, sim_values, exp_days, exp_values, scale=scale)
            
            results[metric] = {
                'rmse': rmse
            }
        
        # Calculate combined RMSE (average across metrics)
        metric_rmses = [r['rmse'] for r in results.values() if isinstance(r, dict) and 'rmse' in r]
        if metric_rmses:
            results['combined_rmse'] = np.mean(metric_rmses)
        else:
            results['combined_rmse'] = np.inf
        
        return results
    
    def find_best_matches(self, density='10k', top_n=10, 
                         metrics=['Total_Radius', 'Inhibited_Radius', 'Necrotic_Radius']):
        """
        Find the best matching simulations for a given seeding density.
        
        Args:
            density: Seeding density ('10k', '5k', or '2k')
            top_n: Number of top matches to return
            metrics: List of metrics to compare
        
        Returns:
            List of dictionaries with run_id, RMSE values, and scales, sorted by combined RMSE
        """
        print(f"\nComparing simulations to {density} experimental data...")
        
        all_results = []
        
        for run_id in self.simulation_data.keys():
            results = self.compare_simulation_to_experiment(run_id, density, metrics)
            if results is None:
                continue
            
            result_entry = {
                'run_id': run_id,
                'combined_rmse': results.get('combined_rmse', np.inf),
                'scale': results.get('scale', 1.0),
                'metrics': {k: v for k, v in results.items() if k not in ['combined_rmse', 'scale']}
            }
            all_results.append(result_entry)
        
        # Sort by combined RMSE
        all_results.sort(key=lambda x: x['combined_rmse'])
        
        return all_results[:top_n]
    
    def plot_best_match(self, run_id, density='10k', 
                       metrics=['Total_Radius', 'Inhibited_Radius', 'Necrotic_Radius'],
                       save_plot=True, show_plot=False, output_dir=None):
        """
        Plot comparison between a simulation and experimental data.
        
        Args:
            run_id: Simulation run ID
            density: Seeding density
            metrics: List of metrics to plot
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            output_dir: Directory to save plots
        """
        if run_id not in self.simulation_data:
            print(f"Error: Run {run_id} not found")
            return
        
        if density not in self.experimental_data:
            print(f"Error: Density {density} not found")
            return
        
        sim_data = self.simulation_data[run_id]['observables']
        exp_data = self.experimental_data[density]
        
        # Calculate comparison results
        results = self.compare_simulation_to_experiment(run_id, density, metrics)
        if results is None:
            print(f"Error: Could not compare run {run_id}")
            return
        
        # Create plot
        n_metrics = len([m for m in metrics if m in sim_data.columns and m in exp_data.columns])
        if n_metrics == 0:
            print("No metrics to plot")
            return
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]
        
        # Get single scale factor (applies to all metrics)
        scale = results.get('scale', 1.0)
        
        plot_idx = 0
        for metric in metrics:
            if metric not in sim_data.columns or metric not in exp_data.columns:
                continue
            
            ax = axes[plot_idx]
            
            # Get data
            sim_steps = sim_data['Step'].values
            sim_values = sim_data[metric].values
            exp_days = exp_data['Day'].values
            exp_values = exp_data[metric].values
            
            # Get RMSE for this metric
            rmse = results[metric]['rmse']
            
            # Find matched points for plotting (step = day)
            matched_sim_steps = []
            matched_sim_values_scaled = []
            matched_exp_days = []
            matched_exp_values = []
            
            for exp_idx, exp_day in enumerate(exp_days):
                if np.isnan(exp_values[exp_idx]):
                    continue
                # Find simulation step that matches this day
                step_match_idx = np.where(sim_steps == exp_day)[0]
                if len(step_match_idx) > 0 and not np.isnan(sim_values[step_match_idx[0]]):
                    matched_sim_steps.append(exp_day)  # Use day for x-axis
                    matched_sim_values_scaled.append(sim_values[step_match_idx[0]] * scale)
                    matched_exp_days.append(exp_day)
                    matched_exp_values.append(exp_values[exp_idx])
            
            # Plot experimental data
            ax.plot(exp_days, exp_values, 'ko-', linewidth=2.5, markersize=8, 
                   label='Experimental', alpha=0.8, zorder=10)
            
            # Plot matched simulation data points only
            if len(matched_sim_steps) > 0:
                ax.plot(matched_sim_steps, matched_sim_values_scaled, 'rs', 
                       linewidth=2, markersize=10, markeredgewidth=2,
                       label=f'Simulation (scale={scale:.2f})', alpha=0.8, zorder=9)
            
            # Also plot full simulation curve for context (lighter)
            # Use steps as x-axis (steps = days)
            sim_steps_clean = sim_steps[~np.isnan(sim_values)]
            sim_values_clean = sim_values[~np.isnan(sim_values)]
            scaled_sim_full = sim_values_clean * scale
            ax.plot(sim_steps_clean, scaled_sim_full, 'r--', linewidth=1, alpha=0.3,
                   label=None)
            
            # Format
            ax.set_xlabel('Time (days)', fontsize=12)
            ax.set_ylabel(metric.replace('_', ' ') + ' (μm)', fontsize=12)
            ax.set_title(f'{metric.replace("_", " ")}\nRMSE: {rmse:.2f} μm', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
        
        fig.suptitle(f'Run {run_id} vs {density} Experimental Data', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_dir is None:
            output_dir = self.sweep_dir / "comparison_plots"
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if save_plot:
            filename = f"best_match_run_{run_id:03d}_{density}.png"
            filepath = output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {filepath}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def print_best_matches(self, density='10k', top_n=10):
        """
        Print a summary of the best matching simulations.
        
        Args:
            density: Seeding density
            top_n: Number of top matches to print
        """
        best_matches = self.find_best_matches(density, top_n)
        
        print(f"\n{'='*80}")
        print(f"Top {len(best_matches)} matches for {density} seeding density:")
        print(f"{'='*80}")
        print(f"{'Rank':<6} {'Run ID':<8} {'Combined RMSE':<15} {'Scale':<10} {'Total_Radius':<20} {'Inhibited_Radius':<20} {'Necrotic_Radius':<20}")
        print(f"{'-'*80}")
        
        for rank, match in enumerate(best_matches, 1):
            run_id = match['run_id']
            combined_rmse = match['combined_rmse']
            scale = match.get('scale', 'N/A')
            
            metrics_str = []
            for metric in ['Total_Radius', 'Inhibited_Radius', 'Necrotic_Radius']:
                if metric in match['metrics']:
                    rmse = match['metrics'][metric]['rmse']
                    metrics_str.append(f"RMSE: {rmse:.1f}")
                else:
                    metrics_str.append("N/A")
            
            scale_str = f"{scale:.2f}" if isinstance(scale, (int, float)) else str(scale)
            print(f"{rank:<6} {run_id:<8} {combined_rmse:<15.2f} {scale_str:<10} {metrics_str[0]:<20} {metrics_str[1]:<20} {metrics_str[2]:<20}")
        
        print(f"{'='*80}\n")
    
    def plot_all_runs_comparison(self, density='10k', 
                                metrics=['Total_Radius', 'Inhibited_Radius', 'Necrotic_Radius'],
                                save_plot=True, show_plot=False, output_dir=None, figsize=(14, 10)):
        """
        Plot all simulation runs together, highlighting the best match.
        
        Args:
            density: Seeding density ('10k', '5k', or '2k')
            metrics: List of metrics to plot
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            output_dir: Directory to save plots
            figsize: Figure size
        """
        if density not in self.experimental_data:
            print(f"Error: Density {density} not found")
            return
        
        # Find best match
        best_matches = self.find_best_matches(density, top_n=1)
        if len(best_matches) == 0:
            print(f"No matches found for {density}")
            return
        
        best_run_id = best_matches[0]['run_id']
        best_scale = best_matches[0].get('scale', 1.0)  # Single scale for all metrics
        
        exp_data = self.experimental_data[density]
        exp_days = exp_data['Day'].values
        
        # Create figure with subplots for each metric
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
        if n_metrics == 1:
            axes = [axes]
        
        # Colors for each metric
        metric_colors = {
            'Total_Radius': 'blue',
            'Inhibited_Radius': 'orange',
            'Necrotic_Radius': 'red'
        }
        
        for plot_idx, metric in enumerate(metrics):
            if metric not in exp_data.columns:
                continue
            
            ax = axes[plot_idx]
            exp_values = exp_data[metric].values
            
            # Plot experimental data
            ax.plot(exp_days, exp_values, 'ko-', linewidth=3, markersize=10,
                   label='Experimental', alpha=0.9, zorder=100)
            
            # Plot all simulation runs
            for run_id, data in sorted(self.simulation_data.items()):
                sim_data = data['observables']
                
                if metric not in sim_data.columns or 'Step' not in sim_data.columns:
                    continue
                
                sim_steps = sim_data['Step'].values
                sim_values = sim_data[metric].values
                
                # Get scale factor (use best match's scale, or calculate per run across all metrics)
                if run_id == best_run_id:
                    scale = best_scale
                    # Highlight best match
                    color = metric_colors.get(metric, 'green')
                    alpha = 0.9
                    linewidth = 2.5
                    zorder = 50
                    label = f'Best Match (Run {run_id})'
                else:
                    # Calculate single scale for this run across all metrics
                    scale = self.calculate_optimal_scale_all_metrics(sim_data, exp_data, metrics)
                    
                    # Grey out other runs
                    color = 'lightgrey'
                    alpha = 0.3
                    linewidth = 1
                    zorder = 1
                    label = None
                
                # Plot simulation data
                sim_steps_clean = sim_steps[~np.isnan(sim_values)]
                sim_values_clean = sim_values[~np.isnan(sim_values)]
                scaled_sim = sim_values_clean * scale
                
                ax.plot(sim_steps_clean, scaled_sim, '-', color=color, 
                       linewidth=linewidth, alpha=alpha, zorder=zorder, label=label)
                
                # Plot matched points for best run
                if run_id == best_run_id:
                    matched_steps = []
                    matched_scaled_values = []
                    for exp_idx, exp_day in enumerate(exp_days):
                        if np.isnan(exp_values[exp_idx]):
                            continue
                        step_match_idx = np.where(sim_steps == exp_day)[0]
                        if len(step_match_idx) > 0 and not np.isnan(sim_values[step_match_idx[0]]):
                            matched_steps.append(exp_day)
                            matched_scaled_values.append(sim_values[step_match_idx[0]] * scale)
                    
                    if len(matched_steps) > 0:
                        ax.plot(matched_steps, matched_scaled_values, 's', 
                               color=color, markersize=8, markeredgewidth=2,
                               markeredgecolor='black', alpha=0.9, zorder=60)
            
            # Format subplot
            ylabel = metric.replace('_', ' ')
            ax.set_xlabel('Time (days)', fontsize=12)
            ax.set_ylabel(ylabel + ' (μm)', fontsize=12)
            ax.set_title(ylabel, fontsize=13, fontweight='bold')
            ax.legend(fontsize=10, loc='best')
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(f'All Simulations vs {density} Experimental Data\n(Best Match Highlighted)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if output_dir is None:
            output_dir = self.sweep_dir / "comparison_plots"
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if save_plot:
            filename = f"all_runs_comparison_{density}.png"
            filepath = output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved all-runs comparison plot to {filepath}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()


def export_results_to_csv(comparator, all_results, output_dir):
    """
    Export comparison results to CSV file.
    
    Args:
        comparator: SweepDataComparator instance
        all_results: Dictionary with density as key and list of matches as value
        output_dir: Directory to save CSV file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create summary dataframe
    summary_rows = []
    
    for density, matches in all_results.items():
        for rank, match in enumerate(matches, 1):
            row = {
                'Density': density,
                'Rank': rank,
                'Run_ID': match['run_id'],
                'Combined_RMSE': match['combined_rmse'],
                'Scale': match.get('scale', np.nan)  # Single scale for all metrics
            }
            
            # Add metric-specific RMSE (all use the same scale)
            for metric in ['Total_Radius', 'Inhibited_Radius', 'Necrotic_Radius']:
                if metric in match['metrics']:
                    row[f'{metric}_RMSE'] = match['metrics'][metric]['rmse']
                else:
                    row[f'{metric}_RMSE'] = np.nan
            
            # Add parameter values if available
            run_id = match['run_id']
            if run_id in comparator.simulation_data:
                config = comparator.simulation_data[run_id]['config']
                # Add key parameters
                for param_name in ['Param_populations_Tumour_dynamics_lambda',
                                  'Param_populations_Tumour_dynamics_mu',
                                  'Param_populations_Necrotic_dynamics_mu',
                                  'Param_nutrient_dynamics_diffusion']:
                    if param_name in config:
                        row[param_name] = config[param_name]
            
            summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Save to CSV
    csv_path = output_dir / "sweep_comparison_results.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\nExported results to {csv_path}")


def main():
    """
    Main function for comparing parameter sweep to experimental data.
    
    Configure the analysis by modifying the variables below, or use command-line arguments.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare parameter sweep results to experimental data')
    parser.add_argument('--sweep-dir', type=str, default=None,
                       help='Path to parameter sweep directory')
    parser.add_argument('--experimental-data', type=str, default=None,
                       help='Path to experimental data Excel file')
    parser.add_argument('--densities', type=str, nargs='+', default=['10k', '5k', '2k'],
                       choices=['10k', '5k', '2k'],
                       help='Seeding densities to analyze')
    parser.add_argument('--top-n', type=int, default=10,
                       help='Number of top matches to display')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots')
    parser.add_argument('--num-plots', type=int, default=5,
                       help='Number of top matches to plot')
    parser.add_argument('--show-plots', action='store_true',
                       help='Display plots (default: save only)')
    parser.add_argument('--export-csv', action='store_true',
                       help='Export results to CSV file')
    
    args = parser.parse_args()
    
    # ============================================================================
    # CONFIGURATION VARIABLES - Modify these to change analysis behavior
    # ============================================================================
    
    # Path to parameter sweep directory
    SWEEP_DIR = args.sweep_dir or "/Users/rileymcnamara/CODE/2025/silicokit/laboratory/parameter_sweeps/random_parameter_sweep_20251118_215113"
    
    # Path to experimental data (None = use default)
    EXPERIMENTAL_DATA_PATH = args.experimental_data
    
    # Seeding densities to analyze ('10k', '5k', '2k', or list of these)
    DENSITIES = args.densities
    
    # Number of top matches to display
    TOP_N = args.top_n
    
    # Metrics to compare
    METRICS = ['Total_Radius', 'Inhibited_Radius', 'Necrotic_Radius']
    
    # Whether to plot the best matches
    PLOT_BEST_MATCHES = not args.no_plots
    NUM_PLOTS = args.num_plots  # Number of top matches to plot
    
    # Whether to plot all runs together
    PLOT_ALL_RUNS = True
    
    # Whether to show plots
    SHOW_PLOTS = args.show_plots
    
    # ============================================================================
    # END OF CONFIGURATION
    # ============================================================================
    
    # Initialize comparator
    comparator = SweepDataComparator(
        sweep_dir=SWEEP_DIR,
        experimental_data_path=EXPERIMENTAL_DATA_PATH
    )
    
    # Analyze each density
    if isinstance(DENSITIES, str):
        DENSITIES = [DENSITIES]
    
    all_results = {}
    
    for density in DENSITIES:
        # Print best matches
        comparator.print_best_matches(density, TOP_N)
        
        # Get best matches for export
        best_matches = comparator.find_best_matches(density, TOP_N)
        all_results[density] = best_matches
        
        # Plot best matches
        if PLOT_BEST_MATCHES:
            for match in best_matches[:NUM_PLOTS]:
                comparator.plot_best_match(
                    match['run_id'], 
                    density, 
                    METRICS,
                    save_plot=True,
                    show_plot=SHOW_PLOTS
                )
        
        # Plot all runs together
        if PLOT_ALL_RUNS:
            comparator.plot_all_runs_comparison(
                density,
                METRICS,
                save_plot=True,
                show_plot=SHOW_PLOTS
            )
    
    # Export results to CSV if requested
    if hasattr(args, 'export_csv') and args.export_csv:
        export_results_to_csv(comparator, all_results, SWEEP_DIR)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()

