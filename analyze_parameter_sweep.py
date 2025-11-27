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
        
        # Fallback: check for observables_data.csv (single run or old format)
        if len(csv_files) == 0:
            csv_files = sorted(self.sweep_dir.glob("observables_data.csv"))
            if len(csv_files) == 0:
                print(f"Warning: No CSV files found in {self.sweep_dir}")
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
                
                # Load CSV file - handle new format with metadata at top
                # First, find where observables data starts
                with open(csv_file, 'r') as f:
                    lines = f.readlines()
                
                # Find where observables data starts
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
                    # Fallback: try reading normally (pandas will skip comment lines)
                    try:
                        df = pd.read_csv(csv_file, comment='#')
                    except Exception:
                        print(f"Warning: Could not parse {csv_file}")
                        continue
                else:
                    df = pd.read_csv(csv_file, skiprows=obs_start_idx)
                
                # Extract config from metadata section (if present)
                config = {}
                varied_params = {}  # Separate varied parameters
                try:
                    # Look for config parameters in metadata section
                    in_param_section = False
                    in_config_section = False
                    for line in lines:
                        line_stripped = line.strip()
                        
                        if line_stripped.startswith('# Varied Parameters'):
                            in_param_section = True
                            in_config_section = False
                            continue
                        elif line_stripped.startswith('# Configuration Parameters'):
                            in_param_section = False
                            in_config_section = True
                            continue
                        elif line_stripped.startswith('# Observables Data'):
                            break
                        elif in_param_section or in_config_section:
                            # Parse CSV-style lines: key, value
                            if ',' in line_stripped and not line_stripped.startswith('#'):
                                parts = line_stripped.split(',', 1)  # Split only on first comma
                                if len(parts) >= 2:
                                    key = parts[0].strip()
                                    value_str = parts[1].strip()
                                    
                                    # Try to convert to numeric if possible
                                    try:
                                        # Try float first
                                        value = float(value_str)
                                        # If it's an integer, convert to int
                                        if value.is_integer():
                                            value = int(value)
                                    except ValueError:
                                        # Keep as string if not numeric
                                        value = value_str
                                    
                                    if in_param_section:
                                        varied_params[key] = value
                                    elif in_config_section:
                                        config[key] = value
                    
                    # Store both varied params and config, with varied params taking precedence
                    # This allows easy access to all parameters
                    all_params = {**config, **varied_params}
                    config = all_params
                    
                except Exception as e:
                    print(f"Warning: Could not parse metadata from {csv_file}: {e}")
                    pass  # If we can't parse metadata, continue without it
                
                # Store data - observables dataframe contains only observables columns
                config_path = csv_file.with_name(f"config_run_{run_id:03d}.yaml")
                self.simulation_data[run_id] = {
                    'observables': df,  # Observables dataframe
                    'config': config if config else {},  # All parameters (varied + config)
                    'varied_params': varied_params if varied_params else {},  # Just varied parameters
                    'file_path': csv_file,
                    'config_path': config_path if config_path.exists() else None
                }
                    
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
        
        print(f"Successfully loaded {len(self.simulation_data)} simulation runs")
    
    def load_experimental_dataset(self, experimental_data_path=None, cell_line=None, seeding_density='10k'):
        """
        Load experimental data with uncertainties using the same conventions as SimPlotter.
        """
        if experimental_data_path is None:
            experimental_data_path = proj / "laboratory" / "data" / "Browning_Paper" / "Organised_Data.xlsx"
        experimental_data_path = Path(experimental_data_path)
        
        if not experimental_data_path.exists():
            print(f"Error: Experimental data file not found at {experimental_data_path}")
            return None
        
        sheet_name = cell_line if cell_line else 'Summary'
        try:
            df = pd.read_excel(experimental_data_path, sheet_name=sheet_name, header=None)
        except ValueError:
            if cell_line:
                print(f"Warning: sheet '{sheet_name}' not found, falling back to 'Summary'")
                df = pd.read_excel(experimental_data_path, sheet_name='Summary', header=None)
            else:
                raise
        except Exception as e:
            print(f"Error loading experimental data sheet '{sheet_name}': {e}")
            return None
        
        density_key = (seeding_density or '10k').lower()
        try:
            if density_key == '10k':
                # 10k: B9 to H18 (columns 1-7, rows 8-17, 0-indexed) = rows 9-18 (1-indexed)
                data = df.iloc[8:18, 1:8].copy()
                data.columns = ['Day', 'Total_Radius', 'Total_Radius_uncertainty', 
                               'Inhibited_Radius', 'Inhibited_Radius_uncertainty',
                               'Necrotic_Radius', 'Necrotic_Radius_uncertainty']
            elif density_key == '5k':
                # 5k: K9 to Q18 (columns 10-16, rows 8-17, 0-indexed) = rows 9-18 (1-indexed)
                data = df.iloc[8:18, 10:17].copy()
                data.columns = ['Day', 'Total_Radius', 'Total_Radius_uncertainty', 
                               'Inhibited_Radius', 'Inhibited_Radius_uncertainty',
                               'Necrotic_Radius', 'Necrotic_Radius_uncertainty']
            elif density_key == '2.5k' or density_key == '2k':
                # 2.5k: S9 to Y18 (columns 18-24, rows 8-17, 0-indexed) = rows 9-18 (1-indexed)
                # Also support '2k' for backward compatibility
                data = df.iloc[8:18, 18:25].copy()
                data.columns = ['Day', 'Total_Radius', 'Total_Radius_uncertainty', 
                               'Inhibited_Radius', 'Inhibited_Radius_uncertainty',
                               'Necrotic_Radius', 'Necrotic_Radius_uncertainty']
            else:
                print(f"Error: Unknown seeding density '{seeding_density}' (expected 10k, 5k, or 2.5k)")
                return None
            
            data = data.dropna(subset=['Day'])
            data = data.apply(pd.to_numeric, errors='coerce')
            data = data.reset_index(drop=True)
            print(f"Loaded experimental dataset (sheet='{sheet_name}', density='{seeding_density}') "
                  f"with {len(data)} points spanning days {data['Day'].min():.1f}-{data['Day'].max():.1f}")
            return data
        except Exception as e:
            print(f"Error parsing experimental data: {e}")
            return None
    
    def calculate_optimal_scale(self, sim_values, exp_values, exp_uncertainties=None):
        """
        Weighted least-squares optimal scale factor.
        Uses uncertainties if provided for weighted least squares.
        """
        if exp_uncertainties is not None:
            mask = ~(np.isnan(sim_values) | np.isnan(exp_values) | np.isnan(exp_uncertainties) | (exp_uncertainties <= 0))
        else:
            mask = ~(np.isnan(sim_values) | np.isnan(exp_values))
        
        if mask.sum() < 2:
            return 1.0
        
        sim_clean = sim_values[mask]
        exp_clean = exp_values[mask]
        
        if exp_uncertainties is not None:
            unc_clean = exp_uncertainties[mask]
            # Weighted least squares: scale = sum(exp * sim / uncertainty^2) / sum(sim^2 / uncertainty^2)
            weights = 1.0 / (unc_clean ** 2)
            scale = np.sum(exp_clean * sim_clean * weights) / np.sum(sim_clean ** 2 * weights)
        else:
            # Unweighted least squares
            denom = np.dot(sim_clean, sim_clean)
            if denom == 0:
                return 1.0
            scale = np.dot(exp_clean, sim_clean) / denom
        
        if not np.isfinite(scale) or scale <= 0:
            return 1.0
        return scale
    
    def calculate_optimal_scale_all_metrics(self, sim_df, exp_df, metrics):
        """
        Find a single scale factor that minimizes weighted RMSE across metrics.
        Uses uncertainties if available for weighted least squares.
        """
        all_sim = []
        all_exp = []
        all_exp_unc = []
        
        if 'Step' not in sim_df.columns:
            return 1.0
        
        sim_steps = sim_df['Step'].values
        exp_days = exp_df['Day'].values
        
        # Check if uncertainties are available
        has_uncertainties = any(f'{metric}_uncertainty' in exp_df.columns for metric in metrics)
        
        for metric in metrics:
            if metric not in sim_df.columns or metric not in exp_df.columns:
                continue
            
            sim_vals = sim_df[metric].values
            exp_vals = exp_df[metric].values
            
            # Get uncertainties if available
            exp_unc_vals = None
            if has_uncertainties:
                unc_col = f'{metric}_uncertainty'
                if unc_col in exp_df.columns:
                    exp_unc_vals = exp_df[unc_col].values
            
            for exp_idx, exp_day in enumerate(exp_days):
                if np.isnan(exp_vals[exp_idx]):
                    continue
                # Step 0 corresponds to day 1 (step = day - 1)
                match_idx = np.where(sim_steps == (exp_day - 1))[0]
                if len(match_idx) == 0:
                    continue
                sim_val = sim_vals[match_idx[0]]
                if np.isnan(sim_val):
                    continue
                all_sim.append(sim_val)
                all_exp.append(exp_vals[exp_idx])
                if exp_unc_vals is not None:
                    all_exp_unc.append(exp_unc_vals[exp_idx])
        
        if len(all_sim) < 2:
            return 1.0
        
        if has_uncertainties and len(all_exp_unc) == len(all_sim):
            return self.calculate_optimal_scale(np.array(all_sim), np.array(all_exp), np.array(all_exp_unc))
        else:
            return self.calculate_optimal_scale(np.array(all_sim), np.array(all_exp))
    
    def calculate_weighted_cost(self, sim_df, exp_df, metrics, scale=1.0):
        """
        Calculate weighted cost function matching the formula:
        C(θ) = sqrt((1/T) * sum_k [((R_sim - R_exp)/σ_R)^2 + ((η_sim - η_exp)/σ_η)^2 + ((ρ_sim - ρ_exp)/σ_ρ)^2])
        
        Where:
        - R = Total_Radius
        - η = Inhibited_Radius  
        - ρ = Necrotic_Radius
        - σ = uncertainties
        - T = number of time points
        - k = time index
        
        Args:
            sim_df: Simulation observables DataFrame
            exp_df: Experimental data DataFrame (with uncertainty columns)
            metrics: List of metric names (e.g., ['Total_Radius', 'Inhibited_Radius', 'Necrotic_Radius'])
            scale: Scaling factor to apply to simulation data
        
        Returns:
            Tuple of (cost_value, num_matched_points)
        """
        if 'Step' not in sim_df.columns:
            return np.inf, 0
        
        sim_steps = sim_df['Step'].values
        exp_days = exp_df['Day'].values
        
        # Collect all matched time points and their contributions
        cost_contributions = []  # Will store sum of squared normalized differences for each time point
        
        # Get all unique matched time points
        # Step 0 corresponds to day 1 (step = day - 1)
        matched_times = []
        for exp_idx, exp_day in enumerate(exp_days):
            if np.isnan(exp_day):
                continue
            match_idx = np.where(sim_steps == (exp_day - 1))[0]
            if len(match_idx) > 0:
                matched_times.append((exp_day, exp_idx, match_idx[0]))
        
        if len(matched_times) < 2:
            return np.inf, 0
        
        # For each matched time point, calculate sum of squared normalized differences across all metrics
        for exp_day, exp_idx, sim_idx in matched_times:
            time_point_sum = 0.0
            has_valid_data = False
            
            for metric in metrics:
                if metric not in sim_df.columns or metric not in exp_df.columns:
                    continue
                
                sim_val = sim_df[metric].values[sim_idx]
                exp_val = exp_df[metric].values[exp_idx]
                
                # Skip if either value is NaN
                if np.isnan(sim_val) or np.isnan(exp_val):
                    continue
                
                # Get uncertainty for this metric
                unc_col = f'{metric}_uncertainty'
                if unc_col in exp_df.columns:
                    uncertainty = exp_df[unc_col].values[exp_idx]
                    # Skip if uncertainty is invalid
                    if np.isnan(uncertainty) or uncertainty <= 0:
                        continue
                else:
                    # No uncertainty available, skip this metric for this time point
                    continue
                
                # Calculate normalized difference: (sim - exp) / uncertainty
                scaled_sim_val = sim_val * scale
                normalized_diff = (scaled_sim_val - exp_val) / uncertainty
                
                # Add squared contribution to time point sum
                time_point_sum += normalized_diff ** 2
                has_valid_data = True
            
            # Only add this time point if we have at least one valid metric
            if has_valid_data:
                cost_contributions.append(time_point_sum)
        
        if len(cost_contributions) < 2:
            return np.inf, 0
        
        # Calculate cost: sqrt(mean of all time point contributions)
        T = len(cost_contributions)
        cost = np.sqrt(np.mean(cost_contributions))
        
        return cost, T
    
    def evaluate_runs_against_experiment(self, cell_line=None, seeding_density='10k',
                                         metrics=None, experimental_data_path=None,
                                         top_n=5):
        """
        Compare every run against experimental data and report best matches.
        """
        dataset = self.load_experimental_dataset(
            experimental_data_path=experimental_data_path,
            cell_line=cell_line,
            seeding_density=seeding_density
        )
        if dataset is None:
            print("Skipping experimental comparison (dataset unavailable).")
            return []
        
        if metrics is None:
            metrics = ['Total_Radius', 'Inhibited_Radius', 'Necrotic_Radius']
        metrics = [m for m in metrics if m in dataset.columns]
        if not metrics:
            print("No overlapping metrics between simulations and experimental data.")
            return []
        
        results = []
        for run_id, data in sorted(self.simulation_data.items()):
            observables_df = data['observables']
            if 'Step' not in observables_df.columns:
                continue
            run_metrics = [m for m in metrics if m in observables_df.columns]
            if not run_metrics:
                continue
            scale = self.calculate_optimal_scale_all_metrics(observables_df, dataset, run_metrics)
            
            # Calculate weighted cost function matching the formula
            cost, num_matched = self.calculate_weighted_cost(
                observables_df, dataset, run_metrics, scale=scale
            )
            
            # For backward compatibility, also calculate individual metric RMSEs
            metric_rmses = {}
            for metric in run_metrics:
                if metric not in observables_df.columns or metric not in dataset.columns:
                    continue
                
                # Calculate weighted RMSE for this metric
                sim_steps = observables_df['Step'].values
                sim_values = observables_df[metric].values
                exp_days = dataset['Day'].values
                exp_values = dataset[metric].values
                
                # Get uncertainties if available
                exp_uncertainties = None
                unc_col = f'{metric}_uncertainty'
                if unc_col in dataset.columns:
                    exp_uncertainties = dataset[unc_col].values
                
                matched_sim = []
                matched_exp = []
                matched_unc = []
                
                for exp_idx, exp_day in enumerate(exp_days):
                    if np.isnan(exp_values[exp_idx]):
                        continue
                    # Step 0 corresponds to day 1 (step = day - 1)
                    match_idx = np.where(sim_steps == (exp_day - 1))[0]
                    if len(match_idx) == 0:
                        continue
                    sim_val = sim_values[match_idx[0]]
                    if np.isnan(sim_val):
                        continue
                    matched_sim.append(sim_val * scale)
                    matched_exp.append(exp_values[exp_idx])
                    if exp_uncertainties is not None:
                        matched_unc.append(exp_uncertainties[exp_idx])
                
                if len(matched_sim) >= 2:
                    matched_sim = np.array(matched_sim)
                    matched_exp = np.array(matched_exp)
                    
                    if exp_uncertainties is not None and len(matched_unc) == len(matched_sim):
                        matched_unc = np.array(matched_unc)
                        # Weighted RMSE
                        valid_mask = ~np.isnan(matched_unc) & (matched_unc > 0)
                        if np.sum(valid_mask) >= 2:
                            weights = 1.0 / (matched_unc[valid_mask] ** 2)
                            weights = weights * len(weights) / np.sum(weights)
                            weighted_sq_errors = ((matched_exp[valid_mask] - matched_sim[valid_mask]) ** 2) * weights
                            rmse = np.sqrt(np.mean(weighted_sq_errors))
                        else:
                            rmse = np.sqrt(np.mean((matched_exp - matched_sim)**2))
                    else:
                        rmse = np.sqrt(np.mean((matched_exp - matched_sim)**2))
                    
                    metric_rmses[metric] = {'rmse': rmse, 'matched_points': len(matched_sim)}
            
            # Use the weighted cost as the combined cost
            combined_rmse = cost if np.isfinite(cost) else np.inf
            result = {
                'run_id': run_id,
                'combined_rmse': combined_rmse,
                'scale': scale,
                'metric_details': metric_rmses,
                'config_path': data.get('config_path'),
                'varied_params': data.get('varied_params', {})
            }
            data['experimental_comparison'] = result
            results.append(result)
        
        results.sort(key=lambda x: x['combined_rmse'])
        best = results[:top_n]
        if best:
            print(f"\nBest matches against experimental data (sheet='{cell_line or 'Summary'}', "
                  f"density='{seeding_density}'):")
            print(f"{'Rank':<5} {'Run':<6} {'RMSE':<10} {'Scale':<8} Details")
            for idx, entry in enumerate(best, 1):
                metric_str = ", ".join(
                    f"{m}:{info['rmse']:.2f}" for m, info in entry['metric_details'].items()
                )
                print(f"{idx:<5} {entry['run_id']:<6} {entry['combined_rmse']:<10.2f} "
                      f"{entry['scale']:<8.2f} {metric_str}")
                if entry['config_path']:
                    print(f"      Config: {entry['config_path']}")
                if entry['varied_params']:
                    params_preview = ", ".join(f"{k}={v}" for k, v in list(entry['varied_params'].items())[:5])
                    print(f"      Varied: {params_preview}")
        else:
            print("No runs produced comparable observables for the selected metrics.")
        return best
    
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
            'Necrotic_Radius': ['Necrotic_Radius', 'Necrotic_Core_Radius', 'necrotic_radius', 'necrotic', 'necrotic_core']
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
            'necrotic_radius': ['necrotic', 'necrotic radius', 'necrotic core', 'necrotic_core_radius']
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
                'Necrotic_Radius'
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
                'Necrotic_Radius'
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
    SWEEP_DIR = "/Users/rileymcnamara/CODE/2025/silicokit/laboratory/parameter_sweeps/random_parameter_sweep_20251126_225253"
    
    # Whether to compare with validation data (set to False if validation data structure is unknown)
    USE_VALIDATION_DATA = False
    
    # Path to validation data file (only used if USE_VALIDATION_DATA is True)
    # If None, defaults to Browning paper data
    VALIDATION_DATA_PATH = None
    
    # Observables to plot (None = use default: Total_Radius, Inhibited_Radius, Necrotic_Radius)
    # Can also specify a list like: ['Total_Radius', 'Inhibited_Radius']
    OBSERVABLES = None
    
    # Output directory for plots (None = save to sweep_dir/comparison_plots)
    OUTPUT_DIR = None
    
    # Whether to display plots (True = show plots, False = save only)
    SHOW_PLOTS = False
    
    # Whether to create comprehensive multi-panel plot (True) or individual plots (False)
    COMPREHENSIVE_PLOT = True
    
    # Whether to evaluate a cost function against experimental data
    FIND_CLOSEST_RUN = True
    
    # Experimental data selection (matches SimPlotter arguments)
    EXPERIMENTAL_CELL_LINE = None  # e.g., "983" to target specific sheet
    EXPERIMENTAL_SEEDING_DENSITY = "10k"
    EXPERIMENTAL_METRICS = ['Total_Radius', 'Inhibited_Radius', 'Necrotic_Radius']
    TOP_MATCHES_TO_SHOW = 5
    
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
    
    if FIND_CLOSEST_RUN:
        analyzer.evaluate_runs_against_experiment(
            cell_line=EXPERIMENTAL_CELL_LINE,
            seeding_density=EXPERIMENTAL_SEEDING_DENSITY,
            metrics=EXPERIMENTAL_METRICS,
            experimental_data_path=VALIDATION_DATA_PATH,
            top_n=TOP_MATCHES_TO_SHOW
        )
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()

