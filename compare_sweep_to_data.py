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
from scipy.stats import pearsonr, spearmanr
from typing import Dict, List, Optional, Tuple
import yaml

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
            sweep_dir: Path to parameter sweep directory containing CSV files, or list of paths
            experimental_data_path: Path to experimental data Excel file
        """
        # Handle both single path and list of paths
        if isinstance(sweep_dir, (list, tuple)):
            self.sweep_dirs = [Path(d) for d in sweep_dir]
            # Use first directory as primary for output paths
            self.sweep_dir = self.sweep_dirs[0]
        else:
            self.sweep_dir = Path(sweep_dir)
            self.sweep_dirs = [self.sweep_dir]
        
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
        Load experimental data from Excel file with uncertainties.
        
        Returns:
            Dictionary with keys '10k', '5k', '2.5k' containing DataFrames with columns:
            'Day', 'Total_Radius', 'Total_Radius_uncertainty', 'Inhibited_Radius', 
            'Inhibited_Radius_uncertainty', 'Necrotic_Radius', 'Necrotic_Radius_uncertainty'
        """
        print(f"Loading experimental data from {self.experimental_data_path}...")
        
        # Read the Summary sheet
        df = pd.read_excel(self.experimental_data_path, sheet_name='Summary', header=None)
        
        # Extract data for each seeding density with uncertainties
        # New structure:
        # 10k: Day (B=col 1), Total_Radius (C=col 2), Total_Radius_uncertainty (D=col 3),
        #      Inhibited_Radius (E=col 4), Inhibited_Radius_uncertainty (F=col 5),
        #      Necrotic_Radius (G=col 6), Necrotic_Radius_uncertainty (H=col 7)
        # 5k: Day (K=col 10), Total_Radius (L=col 11), Total_Radius_uncertainty (M=col 12),
        #     Inhibited_Radius (N=col 13), Inhibited_Radius_uncertainty (O=col 14),
        #     Necrotic_Radius (P=col 15), Necrotic_Radius_uncertainty (Q=col 16)
        # 2.5k: Day (S=col 18), Total_Radius (T=col 19), Total_Radius_uncertainty (U=col 20),
        #       Inhibited_Radius (V=col 21), Inhibited_Radius_uncertainty (W=col 22),
        #       Necrotic_Radius (X=col 23), Necrotic_Radius_uncertainty (Y=col 24)
        
        experimental_data = {}
        
        # 10k seeding density
        # B9 to H18 (columns 1-7, rows 8-17, 0-indexed) = rows 9-18 (1-indexed)
        data_10k = df.iloc[8:18, 1:8].copy()
        data_10k.columns = ['Day', 'Total_Radius', 'Total_Radius_uncertainty', 
                           'Inhibited_Radius', 'Inhibited_Radius_uncertainty',
                           'Necrotic_Radius', 'Necrotic_Radius_uncertainty']
        data_10k = data_10k.dropna(subset=['Day'])  # Remove rows with NaN days
        # Convert to numeric, coercing errors to NaN
        data_10k = data_10k.apply(pd.to_numeric, errors='coerce')
        experimental_data['10k'] = data_10k.reset_index(drop=True)
        
        # 5k seeding density
        # K9 to Q18 (columns 10-16, rows 8-17, 0-indexed) = rows 9-18 (1-indexed)
        data_5k = df.iloc[8:18, 10:17].copy()
        data_5k.columns = ['Day', 'Total_Radius', 'Total_Radius_uncertainty', 
                           'Inhibited_Radius', 'Inhibited_Radius_uncertainty',
                           'Necrotic_Radius', 'Necrotic_Radius_uncertainty']
        data_5k = data_5k.dropna(subset=['Day'])
        # Convert to numeric, coercing errors to NaN
        data_5k = data_5k.apply(pd.to_numeric, errors='coerce')
        experimental_data['5k'] = data_5k.reset_index(drop=True)
        
        # 2.5k seeding density
        # S9 to Y18 (columns 18-24, rows 8-17, 0-indexed) = rows 9-18 (1-indexed)
        data_2_5k = df.iloc[8:18, 18:25].copy()
        data_2_5k.columns = ['Day', 'Total_Radius', 'Total_Radius_uncertainty', 
                             'Inhibited_Radius', 'Inhibited_Radius_uncertainty',
                             'Necrotic_Radius', 'Necrotic_Radius_uncertainty']
        data_2_5k = data_2_5k.dropna(subset=['Day'])
        # Convert to numeric, coercing errors to NaN
        data_2_5k = data_2_5k.apply(pd.to_numeric, errors='coerce')
        experimental_data['2.5k'] = data_2_5k.reset_index(drop=True)
        
        # Also support '2k' for backward compatibility (maps to '2.5k')
        experimental_data['2k'] = experimental_data['2.5k'].copy()
        
        print(f"Loaded experimental data:")
        for density, data in experimental_data.items():
            print(f"  {density}: {len(data)} time points, days {data['Day'].min():.1f} to {data['Day'].max():.1f}")
        
        return experimental_data
    
    def load_simulation_data(self):
        """Load all simulation CSV files from the sweep directory(ies)."""
        if len(self.sweep_dirs) > 1:
            print(f"\nLoading simulation data from {len(self.sweep_dirs)} sweep directories...")
        else:
            print(f"\nLoading simulation data from {self.sweep_dir}...")
        
        total_csv_files = 0
        
        # Process each sweep directory
        for sweep_idx, sweep_dir in enumerate(self.sweep_dirs):
            if len(self.sweep_dirs) > 1:
                print(f"\n  Processing directory {sweep_idx + 1}/{len(self.sweep_dirs)}: {sweep_dir}")
            
            # Find all CSV files
            csv_files = sorted(sweep_dir.glob("observables_run_*.csv"))
            
            # Fallback: check for observables_data.csv (single run or old format)
            if len(csv_files) == 0:
                csv_files = sorted(sweep_dir.glob("observables_data.csv"))
                if len(csv_files) == 0:
                    if len(self.sweep_dirs) == 1:
                        print(f"Warning: No CSV files found in {sweep_dir}")
                    else:
                        print(f"  Warning: No CSV files found in {sweep_dir}")
                    continue
            
            if len(self.sweep_dirs) > 1:
                print(f"  Found {len(csv_files)} CSV files")
            else:
                print(f"Found {len(csv_files)} CSV files")
            
            total_csv_files += len(csv_files)
            
            # Generate a unique prefix for this sweep directory to avoid run ID conflicts
            # Use the directory name as prefix
            sweep_prefix = sweep_dir.name if len(self.sweep_dirs) > 1 else None
            
            for csv_file in csv_files:
                try:
                    # Extract run ID from filename
                    if 'run_' in csv_file.stem:
                        base_run_id = int(csv_file.stem.split('_')[-1])
                    else:
                        base_run_id = len(self.simulation_data) + 1
                    
                    # Create unique run ID: use prefix if multiple directories, otherwise use base ID
                    if sweep_prefix:
                        # Create a tuple (sweep_idx, base_run_id) as the run_id for uniqueness
                        # Or use a string format that's easy to work with
                        run_id = f"{sweep_prefix}_run_{base_run_id:03d}"
                    else:
                        run_id = base_run_id
                    
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
                    
                    # Extract observables (Step, Time, Total_Radius, Inhibited_Radius, Necrotic_Radius)
                    obs_cols = ['Step', 'Time', 'Total_Radius', 'Inhibited_Radius', 'Necrotic_Radius']
                    available_cols = [col for col in obs_cols if col in df.columns]
                    
                    if 'Step' not in df.columns:
                        print(f"Warning: No 'Step' column in {csv_file}")
                        continue
                    
                    # Create observables dataframe
                    observables_df = df[available_cols].copy()
                    
                    # Extract config from metadata section (new format)
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
                    
                    # Store data
                    self.simulation_data[run_id] = {
                        'observables': observables_df,
                        'file_path': csv_file,
                        'config': config if config else {},
                        'varied_params': varied_params if varied_params else {},
                        'sweep_dir': sweep_dir  # Store which sweep directory this came from
                    }
                        
                except Exception as e:
                    print(f"Error loading {csv_file}: {e}")
        
        if len(self.sweep_dirs) > 1:
            print(f"\nSuccessfully loaded {len(self.simulation_data)} simulation runs from {len(self.sweep_dirs)} directories")
        else:
            print(f"Successfully loaded {len(self.simulation_data)} simulation runs")
    
    def _format_run_id(self, run_id):
        """
        Format run_id for display (handles both int and string run_ids).
        
        Args:
            run_id: Run ID (int or string)
        
        Returns:
            Formatted string for display
        """
        if isinstance(run_id, str):
            return run_id
        else:
            return f"run_{run_id:03d}"
    
    def _format_run_id_filename(self, run_id):
        """
        Format run_id for use in filenames (handles both int and string run_ids).
        
        Args:
            run_id: Run ID (int or string)
        
        Returns:
            Formatted string safe for filenames
        """
        if isinstance(run_id, str):
            # Replace any characters that might be problematic in filenames
            return run_id.replace('/', '_').replace('\\', '_').replace(' ', '_')
        else:
            return f"{run_id:03d}"
    
    def calculate_optimal_scale(self, sim_values, exp_values, exp_uncertainties=None):
        """
        Find optimal scaling factor to minimize weighted RMSE between simulation and experimental data.
        Uses uncertainties if provided for weighted least squares.
        
        Args:
            sim_values: Simulation values (numpy array)
            exp_values: Experimental values (numpy array)
            exp_uncertainties: Experimental uncertainties (standard deviations) (numpy array, optional)
        
        Returns:
            Optimal scale factor
        """
        # Remove NaN values
        if exp_uncertainties is not None:
            mask = ~(np.isnan(sim_values) | np.isnan(exp_values) | np.isnan(exp_uncertainties) | (exp_uncertainties <= 0))
        else:
            mask = ~(np.isnan(sim_values) | np.isnan(exp_values))
        
        if mask.sum() < 2:
            return 1.0
        
        sim_clean = sim_values[mask]
        exp_clean = exp_values[mask]
        
        # If all simulation values are zero or very small, return a default scale
        if np.max(np.abs(sim_clean)) < 1e-10:
            return 1.0
        
        if exp_uncertainties is not None:
            # Weighted least squares: minimize sum((exp - scale * sim)^2 / uncertainty^2)
            # scale = sum(exp * sim / uncertainty^2) / sum(sim^2 / uncertainty^2)
            unc_clean = exp_uncertainties[mask]
            weights = 1.0 / (unc_clean ** 2)
            scale = np.sum(exp_clean * sim_clean * weights) / np.sum(sim_clean ** 2 * weights)
        else:
            # Unweighted least squares: minimize ||exp - scale * sim||^2
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
        Uses uncertainties if available for weighted least squares.
        
        Args:
            sim_data: Simulation observables DataFrame
            exp_data: Experimental data DataFrame (with uncertainty columns if available)
            metrics: List of metric names to include
        
        Returns:
            Optimal scale factor (single value for all metrics)
        """
        all_sim_values = []
        all_exp_values = []
        all_exp_uncertainties = []
        
        sim_steps = sim_data['Step'].values
        exp_days = exp_data['Day'].values
        
        # Check if uncertainties are available
        has_uncertainties = any(f'{metric}_uncertainty' in exp_data.columns for metric in metrics)
        
        # Collect matched points from all metrics
        for metric in metrics:
            if metric not in sim_data.columns or metric not in exp_data.columns:
                continue
            
            sim_values = sim_data[metric].values
            exp_values = exp_data[metric].values
            
            # Get uncertainties if available
            exp_uncertainties_metric = None
            if has_uncertainties:
                unc_col = f'{metric}_uncertainty'
                if unc_col in exp_data.columns:
                    exp_uncertainties_metric = exp_data[unc_col].values
            
            # Find matched points (step = day - 1, so step 0 = day 1)
            for exp_idx, exp_day in enumerate(exp_days):
                if np.isnan(exp_values[exp_idx]):
                    continue
                
                step_match_idx = np.where(sim_steps == (exp_day - 1))[0]
                if len(step_match_idx) > 0:
                    sim_val = sim_values[step_match_idx[0]]
                    if not np.isnan(sim_val):
                        all_sim_values.append(sim_val)
                        all_exp_values.append(exp_values[exp_idx])
                        if exp_uncertainties_metric is not None:
                            all_exp_uncertainties.append(exp_uncertainties_metric[exp_idx])
        
        # Convert to arrays
        all_sim_values = np.array(all_sim_values)
        all_exp_values = np.array(all_exp_values)
        
        if len(all_sim_values) < 2:
            return 1.0
        
        # Use uncertainties if available
        if has_uncertainties and len(all_exp_uncertainties) == len(all_exp_values):
            all_exp_uncertainties = np.array(all_exp_uncertainties)
            return self.calculate_optimal_scale(all_sim_values, all_exp_values, all_exp_uncertainties)
        else:
            return self.calculate_optimal_scale(all_sim_values, all_exp_values)
    
    def calculate_rmse(self, sim_steps, sim_values, exp_days, exp_values, scale=None, exp_uncertainties=None):
        """
        Calculate weighted RMSE (chi-squared) between simulation and experimental data using exact step-to-day matches.
        Step 0 corresponds to day 1 (step = day - 1). Uses uncertainties if provided for weighted RMSE.
        
        Args:
            sim_steps: Simulation step numbers (step 0 = day 1, so step = day - 1)
            sim_values: Simulation values
            exp_days: Experimental day numbers
            exp_values: Experimental values
            scale: Scaling factor for values (if None, will be optimized)
            exp_uncertainties: Experimental uncertainties (standard deviations) (numpy array, optional)
        
        Returns:
            Tuple of (weighted_RMSE, optimal_scale)
        """
        # Convert to numpy arrays and ensure numeric types
        sim_steps = np.asarray(sim_steps, dtype=int)
        sim_values = np.asarray(sim_values, dtype=float)
        exp_days = np.asarray(exp_days, dtype=int)
        exp_values = np.asarray(exp_values, dtype=float)
        
        if exp_uncertainties is not None:
            exp_uncertainties = np.asarray(exp_uncertainties, dtype=float)
        
        # Remove NaN values
        sim_mask = ~np.isnan(sim_values)
        exp_mask = ~np.isnan(exp_values) & ~np.isnan(exp_days)
        
        if exp_uncertainties is not None:
            exp_mask = exp_mask & ~np.isnan(exp_uncertainties) & (exp_uncertainties > 0)
        
        sim_steps_clean = sim_steps[sim_mask]
        sim_values_clean = sim_values[sim_mask]
        exp_days_clean = exp_days[exp_mask]
        exp_values_clean = exp_values[exp_mask]
        
        if exp_uncertainties is not None:
            exp_uncertainties_clean = exp_uncertainties[exp_mask]
        else:
            exp_uncertainties_clean = None
        
        if len(sim_steps_clean) == 0 or len(exp_days_clean) == 0:
            return np.inf, 1.0
        
        # Find exact matches: for each experimental day, find corresponding simulation step
        matched_sim_values = []
        matched_exp_values = []
        matched_exp_uncertainties = []
        
        for exp_idx, exp_day in enumerate(exp_days_clean):
            # Find simulation step that matches this day (step = day - 1, so step 0 = day 1)
            step_match_idx = np.where(sim_steps_clean == (exp_day - 1))[0]
            
            if len(step_match_idx) > 0:
                # Use the first match (should only be one)
                matched_sim_values.append(sim_values_clean[step_match_idx[0]])
                matched_exp_values.append(exp_values_clean[exp_idx])
                if exp_uncertainties_clean is not None:
                    matched_exp_uncertainties.append(exp_uncertainties_clean[exp_idx])
        
        # Convert to arrays
        matched_sim_values = np.array(matched_sim_values)
        matched_exp_values = np.array(matched_exp_values)
        
        if len(matched_sim_values) < 2:
            return np.inf, 1.0
        
        # Find optimal scale if not provided
        if scale is None:
            if exp_uncertainties_clean is not None and len(matched_exp_uncertainties) == len(matched_sim_values):
                matched_exp_uncertainties_arr = np.array(matched_exp_uncertainties)
                scale = self.calculate_optimal_scale(matched_sim_values, matched_exp_values, matched_exp_uncertainties_arr)
            else:
                scale = self.calculate_optimal_scale(matched_sim_values, matched_exp_values)
        
        # Calculate weighted RMSE using only matched points
        scaled_sim = scale * matched_sim_values
        
        if exp_uncertainties_clean is not None and len(matched_exp_uncertainties) == len(matched_sim_values):
            matched_exp_uncertainties_arr = np.array(matched_exp_uncertainties)
            # Check if uncertainties are valid (non-NaN, non-zero, positive)
            valid_unc_mask = ~np.isnan(matched_exp_uncertainties_arr) & (matched_exp_uncertainties_arr > 0)
            
            if np.sum(valid_unc_mask) >= 2:
                # Use weighted RMSE for valid uncertainties
                valid_sim = scaled_sim[valid_unc_mask]
                valid_exp = matched_exp_values[valid_unc_mask]
                valid_unc = matched_exp_uncertainties_arr[valid_unc_mask]
                
                # Weighted RMSE: sqrt(mean((exp - sim)^2 / uncertainty^2))
                # This is the square root of reduced chi-squared
                weights = 1.0 / (valid_unc ** 2)
                # Normalize weights so they sum to N (number of points)
                weights = weights * len(weights) / np.sum(weights)
                weighted_squared_errors = ((valid_exp - valid_sim) ** 2) * weights
                rmse = np.sqrt(np.mean(weighted_squared_errors))
            else:
                # Not enough valid uncertainties, fall back to regular RMSE
                rmse = np.sqrt(np.mean((matched_exp_values - scaled_sim)**2))
        else:
            # Regular RMSE
            rmse = np.sqrt(np.mean((matched_exp_values - scaled_sim)**2))
        
        return rmse, scale
    
    def compare_simulation_to_experiment(self, run_id, density='10k', 
                                         metrics=['Total_Radius', 'Inhibited_Radius', 'Necrotic_Radius']):
        """
        Compare a single simulation to experimental data for a given seeding density.
        Uses a single scale factor for all metrics to ensure consistency.
        Uses weighted RMSE if uncertainties are available.
        
        Args:
            run_id: Simulation run ID
            density: Seeding density ('10k', '5k', or '2.5k', '2k' for backward compatibility)
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
            
            # Get uncertainties if available
            exp_uncertainties = None
            unc_col = f'{metric}_uncertainty'
            if unc_col in exp_data.columns:
                exp_uncertainties = exp_data[unc_col].values
            
            rmse, _ = self.calculate_rmse(sim_steps, sim_values, exp_days, exp_values, scale=scale, exp_uncertainties=exp_uncertainties)
            
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
        Uses weighted RMSE if uncertainties are available.
        
        Args:
            density: Seeding density ('10k', '5k', or '2.5k', '2k' for backward compatibility)
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
            
            # Find matched points for plotting (step = day - 1, so step 0 = day 1)
            matched_sim_steps = []
            matched_sim_values_scaled = []
            matched_exp_days = []
            matched_exp_values = []
            
            for exp_idx, exp_day in enumerate(exp_days):
                if np.isnan(exp_values[exp_idx]):
                    continue
                # Find simulation step that matches this day (step = day - 1, so step 0 = day 1)
                step_match_idx = np.where(sim_steps == (exp_day - 1))[0]
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
            # Convert steps to days for x-axis (step 0 = day 1, so step = day - 1)
            sim_steps_clean = sim_steps[~np.isnan(sim_values)]
            sim_values_clean = sim_values[~np.isnan(sim_values)]
            scaled_sim_full = sim_values_clean * scale
            sim_days_clean = sim_steps_clean + 1  # Convert steps to days
            ax.plot(sim_days_clean, scaled_sim_full, 'r--', linewidth=1, alpha=0.3,
                   label=None)
            
            # Format
            ax.set_xlabel('Time (days)', fontsize=12)
            ax.set_ylabel(metric.replace('_', ' ') + ' (μm)', fontsize=12)
            ax.set_title(f'{metric.replace("_", " ")}\nRMSE: {rmse:.2f} μm', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
        
        run_id_display = self._format_run_id(run_id)
        fig.suptitle(f'Run {run_id_display} vs {density} Experimental Data', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_dir is None:
            output_dir = self.sweep_dir / "comparison_plots"
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if save_plot:
            run_id_filename = self._format_run_id_filename(run_id)
            filename = f"best_match_{run_id_filename}_{density}.png"
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
        
        print(f"\n{'='*120}")
        print(f"Top {len(best_matches)} matches for {density} seeding density:")
        print(f"{'='*120}")
        print(f"{'Rank':<6} {'Run ID':<30} {'Combined RMSE':<15} {'Scale':<10} {'Total_Radius':<20} {'Inhibited_Radius':<20} {'Necrotic_Radius':<20}")
        print(f"{'-'*120}")
        
        for rank, match in enumerate(best_matches, 1):
            run_id = match['run_id']
            run_id_display = self._format_run_id(run_id)
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
            print(f"{rank:<6} {run_id_display:<30} {combined_rmse:<15.2f} {scale_str:<10} {metrics_str[0]:<20} {metrics_str[1]:<20} {metrics_str[2]:<20}")
        
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
            
            # Plot experimental data with error bars
            exp_unc_col = f'{metric}_uncertainty'
            if exp_unc_col in exp_data.columns:
                exp_uncertainties = exp_data[exp_unc_col].values
                exp_mask = ~np.isnan(exp_values) & ~np.isnan(exp_uncertainties) & (exp_uncertainties > 0)
                if np.any(exp_mask):
                    ax.errorbar(exp_days[exp_mask], exp_values[exp_mask], 
                              yerr=exp_uncertainties[exp_mask],
                              fmt='ko-', linewidth=3, markersize=10, capsize=4, capthick=1.5,
                              label='Experimental', alpha=0.9, zorder=100)
                else:
                    ax.plot(exp_days, exp_values, 'ko-', linewidth=3, markersize=10,
                           label='Experimental', alpha=0.9, zorder=100)
            else:
                ax.plot(exp_days, exp_values, 'ko-', linewidth=3, markersize=10,
                       label='Experimental', alpha=0.9, zorder=100)
            
            # Plot all simulation runs with error bars
            for run_id, data in sorted(self.simulation_data.items()):
                sim_data = data['observables']
                
                if metric not in sim_data.columns or 'Step' not in sim_data.columns:
                    continue
                
                sim_steps = sim_data['Step'].values
                sim_values = sim_data[metric].values
                
                # Get uncertainties for simulation data - use Total_Radius_uncertainty for all metrics
                # This ensures we use the correct uncertainty (not inhibited or necrotic)
                sim_unc_col = 'Total_Radius_uncertainty'
                if sim_unc_col in sim_data.columns:
                    sim_uncertainties = sim_data[sim_unc_col].values
                else:
                    sim_uncertainties = None
                
                # Get scale factor (use best match's scale, or calculate per run across all metrics)
                if run_id == best_run_id:
                    scale = best_scale
                    # Highlight best match
                    color = metric_colors.get(metric, 'green')
                    alpha = 0.9
                    linewidth = 2.5
                    zorder = 50
                    run_id_display = self._format_run_id(run_id)
                    label = f'Best Match (Run {run_id_display})'
                else:
                    # Calculate single scale for this run across all metrics
                    scale = self.calculate_optimal_scale_all_metrics(sim_data, exp_data, metrics)
                    
                    # Grey out other runs
                    color = 'lightgrey'
                    alpha = 0.3
                    linewidth = 1
                    zorder = 1
                    label = None
                
                # Plot simulation data with error bars
                # Convert steps to days for x-axis (step 0 = day 1, so step = day - 1)
                sim_steps_clean = sim_steps[~np.isnan(sim_values)]
                sim_values_clean = sim_values[~np.isnan(sim_values)]
                scaled_sim = sim_values_clean * scale
                sim_days_clean = sim_steps_clean + 1  # Convert steps to days
                
                # Scale uncertainties if they exist
                if sim_uncertainties is not None:
                    sim_unc_clean = sim_uncertainties[~np.isnan(sim_values)]
                    scaled_sim_unc = sim_unc_clean * scale
                    # Only plot error bars where uncertainties are valid
                    unc_mask = ~np.isnan(scaled_sim_unc) & (scaled_sim_unc > 0)
                    if np.any(unc_mask):
                        ax.errorbar(sim_days_clean[unc_mask], scaled_sim[unc_mask],
                                  yerr=scaled_sim_unc[unc_mask],
                                  fmt='-', color=color, linewidth=linewidth, alpha=alpha,
                                  zorder=zorder, label=label, capsize=2, capthick=1)
                    else:
                        ax.plot(sim_days_clean, scaled_sim, '-', color=color, 
                               linewidth=linewidth, alpha=alpha, zorder=zorder, label=label)
                else:
                    ax.plot(sim_days_clean, scaled_sim, '-', color=color, 
                           linewidth=linewidth, alpha=alpha, zorder=zorder, label=label)
                
                # Plot matched points for best run
                if run_id == best_run_id:
                    matched_steps = []
                    matched_scaled_values = []
                    matched_scaled_uncertainties = []
                    for exp_idx, exp_day in enumerate(exp_days):
                        if np.isnan(exp_values[exp_idx]):
                            continue
                        step_match_idx = np.where(sim_steps == (exp_day - 1))[0]
                        if len(step_match_idx) > 0 and not np.isnan(sim_values[step_match_idx[0]]):
                            matched_steps.append(exp_day)
                            matched_scaled_values.append(sim_values[step_match_idx[0]] * scale)
                            if sim_uncertainties is not None and not np.isnan(sim_uncertainties[step_match_idx[0]]):
                                matched_scaled_uncertainties.append(sim_uncertainties[step_match_idx[0]] * scale)
                            else:
                                matched_scaled_uncertainties.append(0)
                    
                    if len(matched_steps) > 0:
                        if any(unc > 0 for unc in matched_scaled_uncertainties):
                            ax.errorbar(matched_steps, matched_scaled_values,
                                      yerr=matched_scaled_uncertainties,
                                      fmt='s', color=color, markersize=8, markeredgewidth=2,
                                      markeredgecolor='black', alpha=0.9, zorder=60,
                                      capsize=3, capthick=1.5)
                        else:
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
    
    def suggest_parameter_bounds(self, density='10k', 
                                  metrics=['Total_Radius', 'Inhibited_Radius', 'Necrotic_Radius'],
                                  top_percentile=25, expansion_factor=1.2, min_expansion=0.1,
                                  parameter_metric_weights=None):
        """
        Analyze all simulations and suggest parameter bounds for the next parameter sweep
        based on weighted RMSE cost function, with optional weighting by parameter-observable relationships.
        Uses uncertainties if available for weighted RMSE calculation.
        
        Args:
            density: Seeding density to use for RMSE calculation ('10k', '5k', or '2.5k', '2k' for backward compatibility)
            metrics: List of metrics to include in RMSE calculation
            top_percentile: Percentile of best simulations to use for bounds (default: 25 = top 25%)
            expansion_factor: Factor to expand bounds around good parameter ranges (default: 1.2 = 20% expansion)
            min_expansion: Minimum expansion as fraction of range (default: 0.1 = 10%)
            parameter_metric_weights: Dictionary mapping parameter names (with Param_ prefix) to 
                                     dictionaries of metric weights. Example:
                                     {
                                         'Param_populations_Tumour_dynamics_lambda': {
                                             'Total_Radius': 1.0,
                                             'Inhibited_Radius': 0.5,
                                             'Necrotic_Radius': 0.0
                                         },
                                         ...
                                     }
                                     If None, uses combined RMSE for all parameters.
        
        Returns:
            Dictionary with suggested parameter bounds and analysis summary
        """
        print(f"\n{'='*80}")
        print(f"Analyzing parameter bounds for next sweep (density: {density})")
        print(f"{'='*80}")
        
        if density not in self.experimental_data:
            print(f"Error: Density {density} not found")
            return None
        
        # Calculate RMSE for all simulations
        print("Calculating RMSE for all simulations...")
        all_results = []
        
        for run_id in self.simulation_data.keys():
            results = self.compare_simulation_to_experiment(run_id, density, metrics)
            if results is None:
                continue
            
            # Get parameter values for this run
            varied_params = self.simulation_data[run_id].get('varied_params', {})
            config = self.simulation_data[run_id].get('config', {})
            
            # Combine to get all parameters (varied params take precedence)
            all_params = {**config, **varied_params}
            
            # Store metric-specific RMSE values
            metric_rmses = {}
            for metric in metrics:
                if metric in results and isinstance(results[metric], dict):
                    metric_rmses[metric] = results[metric].get('rmse', np.inf)
                else:
                    metric_rmses[metric] = np.inf
            
            result_entry = {
                'run_id': run_id,
                'combined_rmse': results.get('combined_rmse', np.inf),
                'scale': results.get('scale', 1.0),
                'parameters': all_params,
                'metric_rmses': metric_rmses  # Store individual metric RMSEs
            }
            all_results.append(result_entry)
        
        if len(all_results) == 0:
            print("No valid simulation results found")
            return None
        
        # Identify which parameters were varied across runs
        # Collect all parameter names that appear in varied_params
        varied_param_names = set()
        for result in all_results:
            run_id = result['run_id']
            varied_params = self.simulation_data[run_id].get('varied_params', {})
            varied_param_names.update(varied_params.keys())
        
        # If parameter_metric_weights is provided, calculate weighted RMSE for each parameter
        if parameter_metric_weights:
            print("\nUsing parameter-specific metric weights for bounds analysis...")
            for result in all_results:
                # Calculate parameter-specific weighted RMSE
                param_weighted_rmses = {}
                
                for param_name in varied_param_names:
                    if param_name in parameter_metric_weights:
                        weights = parameter_metric_weights[param_name]
                        # Calculate weighted RMSE for this parameter
                        weighted_rmse = 0.0
                        total_weight = 0.0
                        
                        for metric, weight in weights.items():
                            if metric in result['metric_rmses'] and weight > 0:
                                weighted_rmse += result['metric_rmses'][metric] * weight
                                total_weight += weight
                        
                        if total_weight > 0:
                            param_weighted_rmses[param_name] = weighted_rmse / total_weight
                        else:
                            param_weighted_rmses[param_name] = result['combined_rmse']
                    else:
                        # No weights specified, use combined RMSE
                        param_weighted_rmses[param_name] = result['combined_rmse']
                
                result['param_weighted_rmses'] = param_weighted_rmses
        
        # Sort by combined RMSE (for overall threshold calculation)
        all_results.sort(key=lambda x: x['combined_rmse'])
        
        if len(varied_param_names) == 0:
            print("No varied parameters found in simulations")
            return None
        
        print(f"\nFound {len(varied_param_names)} varied parameters:")
        for param in sorted(varied_param_names):
            print(f"  - {param}")
        
        # Calculate percentile threshold
        percentile_idx = int(len(all_results) * (top_percentile / 100.0))
        percentile_idx = max(1, percentile_idx)  # At least 1 simulation
        threshold_rmse = all_results[percentile_idx - 1]['combined_rmse']
        
        print(f"\nUsing top {top_percentile}% of simulations (RMSE <= {threshold_rmse:.2f})")
        print(f"This includes {percentile_idx} out of {len(all_results)} simulations")
        
        # Extract parameter values for top simulations
        top_results = [r for r in all_results if r['combined_rmse'] <= threshold_rmse]
        
        # Analyze each parameter
        suggested_bounds = {}
        parameter_analysis = {}
        
        for param_name in sorted(varied_param_names):
            # Determine which RMSE to use for this parameter
            if parameter_metric_weights and param_name in parameter_metric_weights:
                # Use parameter-specific weighted RMSE
                # Re-sort results by this parameter's weighted RMSE
                param_sorted_results = sorted(all_results, 
                                            key=lambda x: x.get('param_weighted_rmses', {}).get(param_name, x['combined_rmse']))
                param_threshold_idx = int(len(param_sorted_results) * (top_percentile / 100.0))
                param_threshold_idx = max(1, param_threshold_idx)
                param_threshold_rmse = param_sorted_results[param_threshold_idx - 1].get('param_weighted_rmses', {}).get(param_name, threshold_rmse)
                
                print(f"  {param_name}: Using parameter-specific weighted RMSE (threshold: {param_threshold_rmse:.2f})")
            else:
                # Use combined RMSE
                param_sorted_results = all_results
                param_threshold_rmse = threshold_rmse
            
            # Get all values for this parameter
            all_values = []
            top_values = []
            
            for result in param_sorted_results:
                params = result['parameters']
                if param_name in params:
                    try:
                        val = float(params[param_name])
                        all_values.append(val)
                        
                        # Determine if this result is in top percentile for this parameter
                        if parameter_metric_weights and param_name in parameter_metric_weights:
                            param_rmse = result.get('param_weighted_rmses', {}).get(param_name, result['combined_rmse'])
                            if param_rmse <= param_threshold_rmse:
                                top_values.append(val)
                        else:
                            if result['combined_rmse'] <= threshold_rmse:
                                top_values.append(val)
                    except (ValueError, TypeError):
                        continue
            
            if len(all_values) == 0:
                continue
            
            all_values = np.array(all_values)
            
            # Calculate statistics
            param_min = np.min(all_values)
            param_max = np.max(all_values)
            param_range = param_max - param_min
            
            if len(top_values) > 0:
                top_values = np.array(top_values)
                top_min = np.min(top_values)
                top_max = np.max(top_values)
                top_mean = np.mean(top_values)
                top_std = np.std(top_values)
                
                # Suggest bounds centered on good region with expansion
                center = top_mean
                half_range = max(
                    (top_max - top_min) / 2 * expansion_factor,
                    param_range * min_expansion / 2
                )
                
                suggested_min = max(param_min, center - half_range)
                suggested_max = min(param_max, center + half_range)
                
                # If we're at the boundary, expand outward
                if suggested_min == param_min:
                    suggested_min = max(0, param_min - param_range * min_expansion)
                if suggested_max == param_max:
                    suggested_max = param_max + param_range * min_expansion
                
            else:
                # No good values found, use full range with slight expansion
                suggested_min = param_min - param_range * min_expansion
                suggested_max = param_max + param_range * min_expansion
                top_min = top_max = top_mean = top_std = np.nan
            
            # Determine if parameter is numeric (int vs float)
            is_integer = all(isinstance(v, (int, np.integer)) or 
                           (isinstance(v, float) and v.is_integer()) 
                           for v in all_values if not np.isnan(v))
            
            if is_integer:
                suggested_min = int(np.floor(suggested_min))
                suggested_max = int(np.ceil(suggested_max))
            
            suggested_bounds[param_name] = (suggested_min, suggested_max)
            
            parameter_analysis[param_name] = {
                'current_min': param_min,
                'current_max': param_max,
                'top_min': top_min if len(top_values) > 0 else np.nan,
                'top_max': top_max if len(top_values) > 0 else np.nan,
                'top_mean': top_mean if len(top_values) > 0 else np.nan,
                'top_std': top_std if len(top_values) > 0 else np.nan,
                'suggested_min': suggested_min,
                'suggested_max': suggested_max,
                'num_top_sims': len(top_values),
                'is_integer': is_integer
            }
        
        # Print summary
        print(f"\n{'='*80}")
        print("SUGGESTED PARAMETER BOUNDS FOR NEXT SWEEP")
        print(f"{'='*80}")
        print(f"{'Parameter':<50} {'Current Range':<25} {'Suggested Range':<25} {'Top Sim Range':<25}")
        print(f"{'-'*125}")
        
        for param_name in sorted(varied_param_names):
            if param_name not in parameter_analysis:
                continue
            
            analysis = parameter_analysis[param_name]
            current_range = f"[{analysis['current_min']:.4f}, {analysis['current_max']:.4f}]"
            suggested_range = f"[{analysis['suggested_min']:.4f}, {analysis['suggested_max']:.4f}]"
            
            if not np.isnan(analysis['top_min']):
                top_range = f"[{analysis['top_min']:.4f}, {analysis['top_max']:.4f}]"
            else:
                top_range = "N/A"
            
            print(f"{param_name:<50} {current_range:<25} {suggested_range:<25} {top_range:<25}")
        
        print(f"{'='*80}\n")
        
        # Create summary dictionary
        summary = {
            'density': density,
            'total_simulations': len(all_results),
            'top_percentile': top_percentile,
            'threshold_rmse': threshold_rmse,
            'num_top_simulations': len(top_results),
            'suggested_bounds': suggested_bounds,
            'parameter_analysis': parameter_analysis,
            'best_rmse': all_results[0]['combined_rmse'],
            'worst_rmse': all_results[-1]['combined_rmse'],
            'median_rmse': np.median([r['combined_rmse'] for r in all_results])
        }
        
        return summary
    
    def analyze_parameter_error_correlations(self, density='10k',
                                            metrics=['Total_Radius', 'Inhibited_Radius', 'Necrotic_Radius'],
                                            min_correlation=0.1):
        """
        Analyze correlations between parameters and RMSE for each observable.
        This helps identify which parameters have the strongest influence on each observable's error.
        
        Args:
            density: Seeding density to use for RMSE calculation
            metrics: List of metrics to analyze
            min_correlation: Minimum absolute correlation to report (default: 0.1)
        
        Returns:
            Dictionary with correlation analysis results
        """
        print(f"\n{'='*80}")
        print(f"Analyzing parameter-error correlations (density: {density})")
        print(f"{'='*80}")
        
        if density not in self.experimental_data:
            print(f"Error: Density {density} not found")
            return None
        
        # Calculate RMSE for all simulations
        print("Calculating RMSE for all simulations...")
        all_results = []
        
        for run_id in self.simulation_data.keys():
            results = self.compare_simulation_to_experiment(run_id, density, metrics)
            if results is None:
                continue
            
            # Get parameter values for this run
            varied_params = self.simulation_data[run_id].get('varied_params', {})
            config = self.simulation_data[run_id].get('config', {})
            all_params = {**config, **varied_params}
            
            # Store metric-specific RMSE values
            metric_rmses = {}
            for metric in metrics:
                if metric in results and isinstance(results[metric], dict):
                    metric_rmses[metric] = results[metric].get('rmse', np.inf)
                else:
                    metric_rmses[metric] = np.inf
            
            result_entry = {
                'run_id': run_id,
                'combined_rmse': results.get('combined_rmse', np.inf),
                'parameters': all_params,
                'metric_rmses': metric_rmses
            }
            all_results.append(result_entry)
        
        if len(all_results) < 3:
            print("Not enough simulations for correlation analysis (need at least 3)")
            return None
        
        # Identify varied parameters
        varied_param_names = set()
        for result in all_results:
            run_id = result['run_id']
            varied_params = self.simulation_data[run_id].get('varied_params', {})
            varied_param_names.update(varied_params.keys())
        
        if len(varied_param_names) == 0:
            print("No varied parameters found")
            return None
        
        print(f"\nFound {len(varied_param_names)} varied parameters")
        print(f"Analyzing correlations with {len(all_results)} simulations...")
        
        # Build data arrays for correlation analysis
        correlation_results = {}
        
        for metric in metrics:
            correlation_results[metric] = {}
            
            # Extract parameter values and RMSE for this metric
            param_data = {}
            rmse_values = []
            valid_indices = []
            
            for idx, result in enumerate(all_results):
                rmse = result['metric_rmses'].get(metric, np.inf)
                if np.isfinite(rmse):
                    rmse_values.append(rmse)
                    valid_indices.append(idx)
                    
                    # Extract parameter values
                    for param_name in varied_param_names:
                        if param_name not in param_data:
                            param_data[param_name] = []
                        
                        params = result['parameters']
                        if param_name in params:
                            try:
                                val = float(params[param_name])
                                param_data[param_name].append(val)
                            except (ValueError, TypeError):
                                param_data[param_name].append(np.nan)
                        else:
                            param_data[param_name].append(np.nan)
            
            if len(rmse_values) < 3:
                print(f"  {metric}: Not enough valid data points")
                continue
            
            rmse_values = np.array(rmse_values)
            
            # Calculate correlations for each parameter
            param_correlations = {}
            
            for param_name in sorted(varied_param_names):
                param_values = np.array([param_data[param_name][i] for i in range(len(param_data[param_name])) 
                                        if i in valid_indices])
                
                # Filter out NaN values
                valid_mask = ~(np.isnan(param_values) | np.isnan(rmse_values))
                if np.sum(valid_mask) < 3:
                    continue
                
                param_clean = param_values[valid_mask]
                rmse_clean = rmse_values[valid_mask]
                
                # Calculate Pearson correlation (linear relationship)
                try:
                    pearson_corr, pearson_p = pearsonr(param_clean, rmse_clean)
                except:
                    pearson_corr, pearson_p = np.nan, np.nan
                
                # Calculate Spearman correlation (monotonic relationship)
                try:
                    spearman_corr, spearman_p = spearmanr(param_clean, rmse_clean)
                except:
                    spearman_corr, spearman_p = np.nan, np.nan
                
                # Calculate parameter range and mean RMSE at extremes
                param_min_idx = np.argmin(param_clean)
                param_max_idx = np.argmax(param_clean)
                rmse_at_min = rmse_clean[param_min_idx]
                rmse_at_max = rmse_clean[param_max_idx]
                
                # Calculate mean RMSE for bottom and top quartiles of parameter values
                param_sorted_idx = np.argsort(param_clean)
                quartile_size = len(param_clean) // 4
                if quartile_size > 0:
                    bottom_quartile_rmse = np.mean(rmse_clean[param_sorted_idx[:quartile_size]])
                    top_quartile_rmse = np.mean(rmse_clean[param_sorted_idx[-quartile_size:]])
                else:
                    bottom_quartile_rmse = top_quartile_rmse = np.nan
                
                param_correlations[param_name] = {
                    'pearson_corr': pearson_corr,
                    'pearson_p': pearson_p,
                    'spearman_corr': spearman_corr,
                    'spearman_p': spearman_p,
                    'param_min': np.min(param_clean),
                    'param_max': np.max(param_clean),
                    'rmse_at_min': rmse_at_min,
                    'rmse_at_max': rmse_at_max,
                    'bottom_quartile_rmse': bottom_quartile_rmse,
                    'top_quartile_rmse': top_quartile_rmse,
                    'n_samples': len(param_clean)
                }
            
            correlation_results[metric] = param_correlations
        
        # Print correlation summary
        print(f"\n{'='*80}")
        print("PARAMETER-ERROR CORRELATION ANALYSIS")
        print(f"{'='*80}")
        
        for metric in metrics:
            if metric not in correlation_results or len(correlation_results[metric]) == 0:
                continue
            
            print(f"\n{metric}:")
            print(f"{'-'*80}")
            print(f"{'Parameter':<50} {'Pearson r':<12} {'p-value':<12} {'Spearman ρ':<12} {'p-value':<12} {'Influence':<15}")
            print(f"{'-'*80}")
            
            # Sort by absolute Pearson correlation
            metric_corrs = correlation_results[metric]
            sorted_params = sorted(metric_corrs.items(), 
                                 key=lambda x: abs(x[1].get('pearson_corr', 0)) if not np.isnan(x[1].get('pearson_corr', np.nan)) else 0,
                                 reverse=True)
            
            for param_name, corr_data in sorted_params:
                pearson_r = corr_data['pearson_corr']
                pearson_p = corr_data['pearson_p']
                spearman_r = corr_data['spearman_corr']
                spearman_p = corr_data['spearman_p']
                
                # Determine influence level
                if np.isnan(pearson_r) or abs(pearson_r) < min_correlation:
                    continue
                
                if abs(pearson_r) > 0.5:
                    influence = "Strong"
                elif abs(pearson_r) > 0.3:
                    influence = "Moderate"
                else:
                    influence = "Weak"
                
                # Add direction indicator
                if pearson_r < 0:
                    influence += " (neg)"
                else:
                    influence += " (pos)"
                
                pearson_str = f"{pearson_r:.3f}" if not np.isnan(pearson_r) else "N/A"
                pearson_p_str = f"{pearson_p:.3e}" if not np.isnan(pearson_p) else "N/A"
                spearman_str = f"{spearman_r:.3f}" if not np.isnan(spearman_r) else "N/A"
                spearman_p_str = f"{spearman_p:.3e}" if not np.isnan(spearman_p) else "N/A"
                
                print(f"{param_name:<50} {pearson_str:<12} {pearson_p_str:<12} {spearman_str:<12} {spearman_p_str:<12} {influence:<15}")
        
        # Generate suggestions
        print(f"\n{'='*80}")
        print("PARAMETER FOCUS SUGGESTIONS")
        print(f"{'='*80}")
        print("\nBased on correlation analysis, here are suggestions for which parameters to focus on:")
        
        # Aggregate correlations across metrics
        param_scores = {}
        for metric in metrics:
            if metric not in correlation_results:
                continue
            
            for param_name, corr_data in correlation_results[metric].items():
                if param_name not in param_scores:
                    param_scores[param_name] = {
                        'max_abs_corr': 0,
                        'metrics': [],
                        'avg_abs_corr': 0,
                        'correlations': []
                    }
                
                pearson_r = corr_data.get('pearson_corr', 0)
                if not np.isnan(pearson_r):
                    abs_corr = abs(pearson_r)
                    param_scores[param_name]['correlations'].append(abs_corr)
                    param_scores[param_name]['metrics'].append(metric)
                    
                    if abs_corr > param_scores[param_name]['max_abs_corr']:
                        param_scores[param_name]['max_abs_corr'] = abs_corr
        
        # Calculate average correlations
        for param_name in param_scores:
            if len(param_scores[param_name]['correlations']) > 0:
                param_scores[param_name]['avg_abs_corr'] = np.mean(param_scores[param_name]['correlations'])
        
        # Sort by average absolute correlation
        sorted_params = sorted(param_scores.items(), 
                             key=lambda x: x[1]['avg_abs_corr'], 
                             reverse=True)
        
        print(f"\n{'Rank':<6} {'Parameter':<50} {'Avg |Corr|':<12} {'Max |Corr|':<12} {'Affects':<20}")
        print(f"{'-'*100}")
        
        for rank, (param_name, score) in enumerate(sorted_params[:10], 1):
            if score['avg_abs_corr'] < min_correlation:
                continue
            
            affects = ", ".join(score['metrics'][:3])  # Show up to 3 metrics
            if len(score['metrics']) > 3:
                affects += f" (+{len(score['metrics'])-3} more)"
            
            print(f"{rank:<6} {param_name:<50} {score['avg_abs_corr']:<12.3f} {score['max_abs_corr']:<12.3f} {affects:<20}")
        
        # Detailed suggestions
        print(f"\n{'='*80}")
        print("DETAILED SUGGESTIONS BY OBSERVABLE")
        print(f"{'='*80}")
        
        for metric in metrics:
            if metric not in correlation_results or len(correlation_results[metric]) == 0:
                continue
            
            print(f"\n{metric}:")
            metric_corrs = correlation_results[metric]
            sorted_params = sorted(metric_corrs.items(),
                                 key=lambda x: abs(x[1].get('pearson_corr', 0)) if not np.isnan(x[1].get('pearson_corr', np.nan)) else 0,
                                 reverse=True)
            
            top_params = [p for p in sorted_params[:3] 
                         if not np.isnan(p[1].get('pearson_corr', np.nan)) 
                         and abs(p[1].get('pearson_corr', 0)) >= min_correlation]
            
            if len(top_params) == 0:
                print("  No strong correlations found. Consider exploring wider parameter ranges.")
                continue
            
            print("  Top parameters to focus on:")
            for param_name, corr_data in top_params:
                pearson_r = corr_data['pearson_corr']
                
                # Determine optimal direction
                if pearson_r > 0:
                    # Positive correlation: higher param → higher error, so decrease param
                    direction = "decrease"
                    direction_desc = "Lower values → lower error"
                else:
                    # Negative correlation: higher param → lower error, so increase param
                    direction = "increase"
                    direction_desc = "Higher values → lower error"
                
                suggestion = f"    - {param_name}: {direction} value to reduce error (r={pearson_r:.3f}, {direction_desc})"
                
                # Add quartile information for confirmation
                if not np.isnan(corr_data['bottom_quartile_rmse']) and not np.isnan(corr_data['top_quartile_rmse']):
                    bottom_rmse = corr_data['bottom_quartile_rmse']
                    top_rmse = corr_data['top_quartile_rmse']
                    if bottom_rmse < top_rmse:
                        suggestion += f" [Confirmed: bottom quartile RMSE={bottom_rmse:.1f} < top quartile RMSE={top_rmse:.1f}]"
                    else:
                        suggestion += f" [Confirmed: top quartile RMSE={top_rmse:.1f} < bottom quartile RMSE={bottom_rmse:.1f}]"
                
                print(suggestion)
        
        print(f"\n{'='*80}\n")
        
        return {
            'density': density,
            'metrics': metrics,
            'correlation_results': correlation_results,
            'param_scores': param_scores,
            'total_simulations': len(all_results)
        }
    
    def export_suggested_bounds(self, summary, output_path=None, format='yaml'):
        """
        Export suggested parameter bounds to a file.
        
        Args:
            summary: Summary dictionary from suggest_parameter_bounds()
            output_path: Path to output file (default: sweep_dir/suggested_bounds.yaml)
            format: Output format ('yaml' or 'json')
        """
        if summary is None:
            print("No summary to export")
            return
        
        if output_path is None:
            output_path = self.sweep_dir / "suggested_bounds.yaml"
        else:
            output_path = Path(output_path)
        
        # Convert bounds to format suitable for parameter sweep
        # Parameter names need to be converted back from Param_ format
        bounds_dict = {}
        for param_key, (min_val, max_val) in summary['suggested_bounds'].items():
            # Remove 'Param_' prefix and convert underscores back to dots
            if param_key.startswith('Param_'):
                param_name = param_key[6:].replace('_', '.')
            else:
                param_name = param_key.replace('_', '.')
            
            bounds_dict[param_name] = [min_val, max_val]
        
        if format.lower() == 'yaml':
            output_data = {
                'suggested_parameter_bounds': bounds_dict,
                'analysis_summary': {
                    'density': summary['density'],
                    'total_simulations': summary['total_simulations'],
                    'top_percentile': summary['top_percentile'],
                    'threshold_rmse': float(summary['threshold_rmse']),
                    'num_top_simulations': summary['num_top_simulations'],
                    'best_rmse': float(summary['best_rmse']),
                    'worst_rmse': float(summary['worst_rmse']),
                    'median_rmse': float(summary['median_rmse'])
                }
            }
            
            with open(output_path, 'w') as f:
                yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)
            
            print(f"Exported suggested bounds to {output_path}")
        
        elif format.lower() == 'json':
            import json
            output_data = {
                'suggested_parameter_bounds': bounds_dict,
                'analysis_summary': summary
            }
            
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"Exported suggested bounds to {output_path}")
        
        else:
            print(f"Unknown format: {format}. Use 'yaml' or 'json'")


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
    parser.add_argument('--sweep-dir', type=str, default=None, nargs='+',
                       help='Path(s) to parameter sweep directory(ies). Can specify multiple directories to analyze together.')
    parser.add_argument('--experimental-data', type=str, default=None,
                       help='Path to experimental data Excel file')
    parser.add_argument('--densities', type=str, nargs='+', default=['10k', '5k', '2.5k'],
                       choices=['10k', '5k', '2.5k', '2k'],
                       help='Seeding densities to analyze (2k maps to 2.5k for backward compatibility)')
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
    parser.add_argument('--no-correlations', action='store_true',
                       help='Skip parameter-error correlation analysis')
    parser.add_argument('--min-correlation', type=float, default=0.1,
                       help='Minimum correlation threshold for reporting (default: 0.1)')
    
    args = parser.parse_args()
    
    # ============================================================================
    # CONFIGURATION VARIABLES - Modify these to change analysis behavior
    # ============================================================================
    
    # Path(s) to parameter sweep directory(ies)
    # Can be a single path (string) or a list of paths to analyze multiple sweeps together
    if args.sweep_dir:
        SWEEP_DIR = args.sweep_dir if isinstance(args.sweep_dir, list) else [args.sweep_dir]
    else:
        # Default: single directory (can be changed to a list to analyze multiple sweeps)
        SWEEP_DIR = ["/Users/rileymcnamara/CODE/2025/silicokit/laboratory/parameter_sweeps/random_parameter_sweep_20251127_220513",
        "/Users/rileymcnamara/CODE/2025/silicokit/laboratory/parameter_sweeps/random_parameter_sweep_20251125_214810",
        "/Users/rileymcnamara/CODE/2025/silicokit/laboratory/parameter_sweeps/random_parameter_sweep_20251125_195936",
        "/Users/rileymcnamara/CODE/2025/silicokit/laboratory/parameter_sweeps/random_parameter_sweep_20251125_214810",
        "/Users/rileymcnamara/CODE/2025/silicokit/laboratory/parameter_sweeps/random_parameter_sweep_20251126_225253",
        "/Users/rileymcnamara/CODE/2025/silicokit/laboratory/parameter_sweeps/local_refinement_sweep_20251128_141213",
        "/Users/rileymcnamara/CODE/2025/silicokit/laboratory/parameter_sweeps/local_refinement_sweep_20251128_200026",
        "/Users/rileymcnamara/CODE/2025/silicokit/laboratory/parameter_sweeps/local_refinement_sweep_20251128_153714",
        "/Users/rileymcnamara/CODE/2025/silicokit/laboratory/parameter_sweeps/local_refinement_sweep_20251128_213646",
        "/Users/rileymcnamara/CODE/2025/silicokit/laboratory/parameter_sweeps/random_parameter_sweep_20251128_225125",
        "/Users/rileymcnamara/CODE/2025/silicokit/laboratory/parameter_sweeps/random_parameter_sweep_20251129_140711",
        "/Users/rileymcnamara/CODE/2025/silicokit/laboratory/parameter_sweeps/local_refinement_sweep_20251129_180026",
        "/Users/rileymcnamara/CODE/2025/silicokit/laboratory/parameter_sweeps/local_refinement_sweep_20251129_185853",
        "/Users/rileymcnamara/CODE/2025/silicokit/laboratory/parameter_sweeps/local_refinement_sweep_20251129_211415"
        ]
        # Example for multiple directories:
        #SWEEP_DIR = ["/Users/rileymcnamara/CODE/2025/silicokit/laboratory/parameter_sweeps/local_refinement_sweep_20251128_200026"]
    
    # Path to experimental data (None = use default)
    EXPERIMENTAL_DATA_PATH = args.experimental_data
    
    # Seeding densities to analyze ('10k', '5k', '2.5k', or list of these; '2k' maps to '2.5k' for backward compatibility)
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
    # PARAMETER BOUNDS SUGGESTION CONFIGURATION
    # ============================================================================
    
    # Whether to suggest parameter bounds for next sweep
    SUGGEST_BOUNDS = True  # Set to True to enable bounds suggestion
    
    # Whether to analyze parameter-error correlations
    ANALYZE_CORRELATIONS = not args.no_correlations  # Set to True to enable correlation analysis
    
    # Density to use for bounds suggestion (use first density if None)
    BOUNDS_DENSITY = None  # None = use first density from DENSITIES, or specify '10k', '5k', '2.5k' ('2k' for backward compatibility)
    
    # Minimum correlation threshold for reporting (0.1 = 10%)
    MIN_CORRELATION = args.min_correlation
    
    # Percentile of best simulations to use for bounds (25 = top 25%)
    TOP_PERCENTILE = 25.0
    
    # Factor to expand bounds around good parameter ranges (1.2 = 20% expansion)
    EXPANSION_FACTOR = 1.2
    
    # Minimum expansion as fraction of range (0.1 = 10%)
    MIN_EXPANSION = 0.1
    
    # Output format for suggested bounds ('yaml' or 'json')
    BOUNDS_OUTPUT_FORMAT = 'yaml'
    
    # Output path for suggested bounds (None = sweep_dir/suggested_bounds.yaml)
    BOUNDS_OUTPUT_PATH = None
    
    # Parameter-observable relationships for improved bounds suggestions
    # Maps parameter names (with Param_ prefix) to dictionaries of metric weights
    # Weights indicate how strongly each parameter affects each observable
    # Example: If lambda primarily affects Total_Radius, give it weight 1.0 for Total_Radius
    #          and lower weights (0.5, 0.0) for other metrics
    PARAMETER_METRIC_WEIGHTS = {
        # Example structure (uncomment and modify based on your knowledge):
        # 'Param_populations_Tumour_dynamics_lambda': {
        #     'Total_Radius': 1.0,           # Strong effect
        #     'Inhibited_Radius': 0.5,      # Moderate effect
        #     'Necrotic_Radius': 0.0        # No effect
        # },
        # 'Param_populations_Tumour_dynamics_mu': {
        #     'Total_Radius': 0.5,
        #     'Inhibited_Radius': 1.0,
        #     'Necrotic_Radius': 0.3
        # },
        # 'Param_populations_Necrotic_dynamics_mu': {
        #     'Total_Radius': 0.0,
        #     'Inhibited_Radius': 0.3,
        #     'Necrotic_Radius': 1.0
        # },
        # 'Param_populations_Necrotic_dynamics_beta_N': {
        #     'Total_Radius': 0.0,
        #     'Inhibited_Radius': 0.2,
        #     'Necrotic_Radius': 1.0
        # },
        # 'Param_nutrient_dynamics_k': {
        #     'Total_Radius': 0.8,
        #     'Inhibited_Radius': 0.6,
        #     'Necrotic_Radius': 0.4
        # },
        # 'Param_populations_Tumour_dynamics_nutrient_consumption': {
        #     'Total_Radius': 0.6,
        #     'Inhibited_Radius': 0.8,
        #     'Necrotic_Radius': 0.5
        # },
        # 'Param_populations_Tumour_dynamics_nutrient_threshold': {
        #     'Total_Radius': 0.4,
        #     'Inhibited_Radius': 0.7,
        #     'Necrotic_Radius': 0.6
        # },
    }
    # If PARAMETER_METRIC_WEIGHTS is empty (or None), uses combined RMSE for all parameters
    
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
        # Use the primary sweep directory for output
        export_results_to_csv(comparator, all_results, comparator.sweep_dir)
    
    # Suggest parameter bounds if enabled
    if SUGGEST_BOUNDS:
        # Determine which density to use for bounds suggestion
        if BOUNDS_DENSITY is not None:
            bounds_density = BOUNDS_DENSITY
        elif isinstance(DENSITIES, list) and len(DENSITIES) > 0:
            bounds_density = DENSITIES[0]
        else:
            bounds_density = DENSITIES
        
        print(f"\n{'='*80}")
        print(f"Generating parameter bounds suggestions...")
        print(f"{'='*80}")
        
        summary = comparator.suggest_parameter_bounds(
            density=bounds_density,
            metrics=METRICS,
            top_percentile=TOP_PERCENTILE,
            expansion_factor=EXPANSION_FACTOR,
            min_expansion=MIN_EXPANSION,
            parameter_metric_weights=PARAMETER_METRIC_WEIGHTS if PARAMETER_METRIC_WEIGHTS else None
        )
        
        if summary:
            # Export suggested bounds
            output_path = BOUNDS_OUTPUT_PATH if BOUNDS_OUTPUT_PATH else None
            comparator.export_suggested_bounds(
                summary, 
                output_path=output_path,
                format=BOUNDS_OUTPUT_FORMAT
            )
    
    # Analyze parameter-error correlations if enabled
    if ANALYZE_CORRELATIONS:
        # Determine which density to use for correlation analysis
        if BOUNDS_DENSITY is not None:
            corr_density = BOUNDS_DENSITY
        elif isinstance(DENSITIES, list) and len(DENSITIES) > 0:
            corr_density = DENSITIES[0]
        else:
            corr_density = DENSITIES
        
        correlation_summary = comparator.analyze_parameter_error_correlations(
            density=corr_density,
            metrics=METRICS,
            min_correlation=MIN_CORRELATION
        )
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()

