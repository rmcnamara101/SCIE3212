#!/usr/bin/env python3
"""
Simulation Analysis GUI

A comprehensive GUI application for loading and visualizing tumor growth simulation results.
Provides easy access to all plotting capabilities without needing to switch between tabs or scroll through notebooks.

Features:
- Load simulation data files (.npz)
- Display all available plots in organized tabs
- Customize plot parameters
- Save plots and export data
- Recent files management
- Real-time plot updates
"""

import sys
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
import threading
from pathlib import Path
import json
from datetime import datetime

# Add project path to sys.path
if sys.platform == "darwin":
    proj = "/Users/rileymcnamara/CODE/2025/silicokit/"
    sys.path.insert(0, proj)
else:
    proj = "C:/Users/riley.mcnamara/Documents/code/silicokit/"
    sys.path.insert(0, proj)

from src.growkit.PlotEngine.SimPlotter import SimPlotter
from src.growkit.PlotEngine.PhysicsFieldsPlotter import PhysicsFieldsPlotter
from src.growkit.PlotEngine.CellFieldPlotter import CellFieldPlotter


class SimulationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tumor Growth Simulation Analysis")
        self.root.geometry("1400x900")
        
        # Data storage
        self.simulation_data = None
        self.simulator = None
        self.sim_plotter = None
        self.physics_plotter = None
        self.cell_plotter = None
        self.current_file = None
        self.recent_files = []
        
        # GUI state
        self.loading = False
        
        # Load recent files
        self.load_recent_files()
        
        # Create GUI
        self.create_widgets()
        
        # Set up matplotlib style
        plt.style.use('default')
        
    def create_widgets(self):
        """Create the main GUI layout"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top frame for file operations
        self.create_file_frame(main_frame)
        
        # Middle frame with notebook for plots
        self.create_plot_notebook(main_frame)
        
        # Bottom frame for status and controls
        self.create_status_frame(main_frame)
        
    def create_file_frame(self, parent):
        """Create file loading and management frame"""
        file_frame = ttk.LabelFrame(parent, text="File Management", padding=10)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        # File selection row
        file_row = ttk.Frame(file_frame)
        file_row.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(file_row, text="Load Simulation", command=self.load_simulation_file).pack(side=tk.LEFT, padx=(0, 10))
        
        self.file_label = ttk.Label(file_row, text="No file loaded", foreground="gray")
        self.file_label.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(file_row, text="Reload", command=self.reload_current_file).pack(side=tk.LEFT, padx=(0, 10))
        
        # Recent files
        recent_frame = ttk.Frame(file_frame)
        recent_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(recent_frame, text="Recent files:").pack(side=tk.LEFT)
        
        self.recent_var = tk.StringVar()
        self.recent_combo = ttk.Combobox(recent_frame, textvariable=self.recent_var, width=50, state="readonly")
        self.recent_combo.pack(side=tk.LEFT, padx=(5, 10))
        self.recent_combo.bind('<<ComboboxSelected>>', self.on_recent_file_selected)
        
        ttk.Button(recent_frame, text="Clear Recent", command=self.clear_recent_files).pack(side=tk.LEFT)
        
        self.update_recent_files_display()
        
    def create_plot_notebook(self, parent):
        """Create the main notebook for different plot categories"""
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Create tabs
        self.create_overview_tab()
        self.create_observables_tab()
        self.create_cell_fields_tab()
        self.create_physics_fields_tab()
        self.create_nutrient_tab()
        self.create_custom_tab()
        
    def create_overview_tab(self):
        """Create overview tab with quick access to common plots"""
        overview_frame = ttk.Frame(self.notebook)
        self.notebook.add(overview_frame, text="Overview")
        
        # Left panel for controls
        control_frame = ttk.Frame(overview_frame)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 5), pady=10)
        
        # Quick plot buttons
        ttk.Label(control_frame, text="Quick Plots", font=("Arial", 12, "bold")).pack(pady=(0, 10))
        
        plot_buttons = [
            ("Tumor Radius Evolution", self.plot_tumor_radius),
            ("Population Density Evolution", self.plot_population_density),
            ("Tumor Shape Evolution", self.plot_tumor_shape),
            ("Center of Mass Evolution", self.plot_center_of_mass),
            ("Compactness Evolution", self.plot_compactness),
            ("All Observables", self.plot_all_observables),
        ]
        
        for text, command in plot_buttons:
            btn = ttk.Button(control_frame, text=text, command=command, width=25)
            btn.pack(pady=2, fill=tk.X)
            
        # Plot parameters
        params_frame = ttk.LabelFrame(control_frame, text="Plot Parameters", padding=5)
        params_frame.pack(fill=tk.X, pady=(20, 0))
        
        # Threshold control
        ttk.Label(params_frame, text="Threshold:").pack(anchor=tk.W)
        self.threshold_var = tk.DoubleVar(value=0.1)
        threshold_scale = ttk.Scale(params_frame, from_=0.01, to=0.5, variable=self.threshold_var, orient=tk.HORIZONTAL)
        threshold_scale.pack(fill=tk.X)
        self.threshold_label = ttk.Label(params_frame, text="0.1")
        self.threshold_label.pack(anchor=tk.W)
        
        # Method selection
        ttk.Label(params_frame, text="Radius Method:").pack(anchor=tk.W, pady=(10, 0))
        self.radius_method_var = tk.StringVar(value="contour")
        method_combo = ttk.Combobox(params_frame, textvariable=self.radius_method_var, 
                                   values=["contour", "mass"], state="readonly", width=15)
        method_combo.pack(fill=tk.X)
        
        # Max plots control
        ttk.Label(params_frame, text="Max Plots:").pack(anchor=tk.W, pady=(10, 0))
        self.max_plots_var = tk.IntVar(value=6)
        max_plots_scale = ttk.Scale(params_frame, from_=3, to=12, variable=self.max_plots_var, orient=tk.HORIZONTAL)
        max_plots_scale.pack(fill=tk.X)
        self.max_plots_label = ttk.Label(params_frame, text="6")
        self.max_plots_label.pack(anchor=tk.W)
        
        # Figure size controls
        ttk.Label(params_frame, text="Figure Size:").pack(anchor=tk.W, pady=(10, 0))
        size_frame = ttk.Frame(params_frame)
        size_frame.pack(fill=tk.X)
        
        ttk.Label(size_frame, text="Width:").pack(side=tk.LEFT)
        self.fig_width_var = tk.IntVar(value=12)
        ttk.Scale(size_frame, from_=6, to=20, variable=self.fig_width_var, orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        ttk.Label(size_frame, text="Height:").pack(side=tk.LEFT, padx=(10, 0))
        self.fig_height_var = tk.IntVar(value=8)
        ttk.Scale(size_frame, from_=4, to=16, variable=self.fig_height_var, orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Line style controls
        style_frame = ttk.LabelFrame(control_frame, text="Line Styles", padding=5)
        style_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(style_frame, text="Line Width:").pack(anchor=tk.W)
        self.line_width_var = tk.DoubleVar(value=2.0)
        ttk.Scale(style_frame, from_=0.5, to=5.0, variable=self.line_width_var, orient=tk.HORIZONTAL).pack(fill=tk.X)
        
        ttk.Label(style_frame, text="Marker Size:").pack(anchor=tk.W, pady=(10, 0))
        self.marker_size_var = tk.DoubleVar(value=6.0)
        ttk.Scale(style_frame, from_=2.0, to=15.0, variable=self.marker_size_var, orient=tk.HORIZONTAL).pack(fill=tk.X)
        
        # Color scheme
        ttk.Label(style_frame, text="Color Scheme:").pack(anchor=tk.W, pady=(10, 0))
        self.color_scheme_var = tk.StringVar(value="Set1")
        color_combo = ttk.Combobox(style_frame, textvariable=self.color_scheme_var,
                                 values=["Set1", "Set2", "Set3", "tab10", "viridis", "plasma", "inferno", "magma"],
                                 state="readonly", width=15)
        color_combo.pack(fill=tk.X)
        
        # Grid and appearance
        appearance_frame = ttk.LabelFrame(control_frame, text="Appearance", padding=5)
        appearance_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.show_grid_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(appearance_frame, text="Show Grid", variable=self.show_grid_var).pack(anchor=tk.W)
        
        self.show_legend_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(appearance_frame, text="Show Legend", variable=self.show_legend_var).pack(anchor=tk.W)
        
        self.include_individual_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(appearance_frame, text="Include Individual Populations", variable=self.include_individual_var).pack(anchor=tk.W)
        
        # Bind scale updates
        threshold_scale.configure(command=self.update_threshold_label)
        max_plots_scale.configure(command=self.update_max_plots_label)
        
        # Right panel for plot display
        self.overview_plot_frame = ttk.Frame(overview_frame)
        self.overview_plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 10), pady=10)
        
        # Create matplotlib figure
        self.overview_fig = Figure(figsize=(10, 8), dpi=100)
        self.overview_canvas = FigureCanvasTkAgg(self.overview_fig, self.overview_plot_frame)
        self.overview_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        self.overview_toolbar = NavigationToolbar2Tk(self.overview_canvas, self.overview_plot_frame)
        self.overview_toolbar.update()
        
    def create_observables_tab(self):
        """Create observables tab for detailed analysis"""
        observables_frame = ttk.Frame(self.notebook)
        self.notebook.add(observables_frame, text="Observables")
        
        # Create a scrollable frame for controls
        canvas = tk.Canvas(observables_frame)
        scrollbar = ttk.Scrollbar(observables_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Left panel for controls
        observables_control_frame = ttk.Frame(scrollable_frame)
        observables_control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 5), pady=10)
        
        # Plot selection
        ttk.Label(observables_control_frame, text="Observable Plots", font=("Arial", 12, "bold")).pack(pady=(0, 10))
        
        observables_buttons = [
            ("All Observables", self.plot_all_observables),
            ("Radius Evolution", self.plot_radius_evolution),
            ("Density Evolution", self.plot_density_evolution),
            ("Center of Mass", self.plot_center_of_mass),
            ("Compactness", self.plot_compactness),
            ("Free Energy", self.plot_free_energy_observable),
        ]
        
        for text, command in observables_buttons:
            btn = ttk.Button(observables_control_frame, text=text, command=command, width=25)
            btn.pack(pady=2, fill=tk.X)
        
        # Population selection
        ttk.Label(observables_control_frame, text="Population Selection", font=("Arial", 10, "bold")).pack(pady=(20, 5))
        
        self.observables_population_var = tk.StringVar()
        self.observables_population_combo = ttk.Combobox(observables_control_frame, textvariable=self.observables_population_var, state="readonly")
        self.observables_population_combo.pack(fill=tk.X, pady=(0, 10))
        
        # Time range selection
        time_frame = ttk.LabelFrame(observables_control_frame, text="Time Range", padding=5)
        time_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(time_frame, text="Start Time:").pack(anchor=tk.W)
        self.start_time_var = tk.DoubleVar(value=0.0)
        ttk.Scale(time_frame, from_=0.0, to=10.0, variable=self.start_time_var, orient=tk.HORIZONTAL).pack(fill=tk.X)
        self.start_time_label = ttk.Label(time_frame, text="0.0")
        self.start_time_label.pack(anchor=tk.W)
        
        ttk.Label(time_frame, text="End Time:").pack(anchor=tk.W, pady=(10, 0))
        self.end_time_var = tk.DoubleVar(value=10.0)
        ttk.Scale(time_frame, from_=0.0, to=10.0, variable=self.end_time_var, orient=tk.HORIZONTAL).pack(fill=tk.X)
        self.end_time_label = ttk.Label(time_frame, text="10.0")
        self.end_time_label.pack(anchor=tk.W)
        
        # Plot customization
        custom_frame = ttk.LabelFrame(observables_control_frame, text="Plot Customization", padding=5)
        custom_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(custom_frame, text="Figure Size:").pack(anchor=tk.W)
        obs_size_frame = ttk.Frame(custom_frame)
        obs_size_frame.pack(fill=tk.X)
        
        ttk.Label(obs_size_frame, text="W:").pack(side=tk.LEFT)
        self.obs_width_var = tk.IntVar(value=12)
        ttk.Scale(obs_size_frame, from_=6, to=20, variable=self.obs_width_var, orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        ttk.Label(obs_size_frame, text="H:").pack(side=tk.LEFT, padx=(10, 0))
        self.obs_height_var = tk.IntVar(value=8)
        ttk.Scale(obs_size_frame, from_=4, to=16, variable=self.obs_height_var, orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Line style options
        ttk.Label(custom_frame, text="Line Style:").pack(anchor=tk.W, pady=(10, 0))
        self.obs_line_style_var = tk.StringVar(value="-")
        line_style_combo = ttk.Combobox(custom_frame, textvariable=self.obs_line_style_var,
                                       values=["-", "--", "-.", ":", "o", "s", "^", "v", "<", ">"],
                                       state="readonly", width=15)
        line_style_combo.pack(fill=tk.X)
        
        # Color options
        ttk.Label(custom_frame, text="Color Scheme:").pack(anchor=tk.W, pady=(10, 0))
        self.obs_color_scheme_var = tk.StringVar(value="Set1")
        obs_color_combo = ttk.Combobox(custom_frame, textvariable=self.obs_color_scheme_var,
                                      values=["Set1", "Set2", "Set3", "tab10", "viridis", "plasma", "inferno", "magma"],
                                      state="readonly", width=15)
        obs_color_combo.pack(fill=tk.X)
        
        # Pack the scrollable area
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Right panel for plot
        self.observables_plot_frame = ttk.Frame(observables_frame)
        self.observables_plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 10), pady=10)
        
        # Create plot area
        self.observables_fig = Figure(figsize=(12, 8), dpi=100)
        self.observables_canvas = FigureCanvasTkAgg(self.observables_fig, self.observables_plot_frame)
        self.observables_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        self.observables_toolbar = NavigationToolbar2Tk(self.observables_canvas, self.observables_plot_frame)
        self.observables_toolbar.update()
        
    def create_cell_fields_tab(self):
        """Create cell fields visualization tab"""
        cell_frame = ttk.Frame(self.notebook)
        self.notebook.add(cell_frame, text="Cell Fields")
        
        # Create main container with horizontal layout
        main_container = ttk.Frame(cell_frame)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Left panel for controls - fixed width, no scrolling
        cell_control_frame = ttk.Frame(main_container, width=300)
        cell_control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 5), pady=10)
        cell_control_frame.pack_propagate(False)  # Prevent shrinking
        
        # Population selection
        ttk.Label(cell_control_frame, text="Population", font=("Arial", 10, "bold")).pack(pady=(0, 5))
        self.population_var = tk.StringVar()
        self.population_combo = ttk.Combobox(cell_control_frame, textvariable=self.population_var, state="readonly")
        self.population_combo.pack(fill=tk.X, pady=(0, 10))
        
        # Step selection - simplified
        ttk.Label(cell_control_frame, text="Time Step", font=("Arial", 10, "bold")).pack(pady=(0, 5))
        self.step_var = tk.IntVar()
        self.step_scale = ttk.Scale(cell_control_frame, from_=0, to=10, variable=self.step_var, orient=tk.HORIZONTAL)
        self.step_scale.pack(fill=tk.X, pady=(0, 5))
        self.step_label = ttk.Label(cell_control_frame, text="Step: 0")
        self.step_label.pack()
        
        # Z-slice selection - simplified
        ttk.Label(cell_control_frame, text="Z-Slice", font=("Arial", 10, "bold")).pack(pady=(5, 5))
        self.z_slice_var = tk.IntVar()
        self.z_slice_scale = ttk.Scale(cell_control_frame, from_=0, to=10, variable=self.z_slice_var, orient=tk.HORIZONTAL)
        self.z_slice_scale.pack(fill=tk.X, pady=(0, 5))
        self.z_slice_label = ttk.Label(cell_control_frame, text="Z: 0")
        self.z_slice_label.pack()
        
        # Main options in a single compact frame
        options_frame = ttk.LabelFrame(cell_control_frame, text="Plot Options", padding=5)
        options_frame.pack(fill=tk.X, pady=(5, 0))
        
        # Row 1: Colormap
        row1 = ttk.Frame(options_frame)
        row1.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(row1, text="Colormap:").pack(side=tk.LEFT)
        self.cell_colormap_var = tk.StringVar(value="viridis")
        colormap_combo = ttk.Combobox(row1, textvariable=self.cell_colormap_var,
                                     values=["viridis", "plasma", "inferno", "magma", "jet", "hot", "cool"],
                                     state="readonly", width=15)
        colormap_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Row 2: Zoom control (separate row for better usability)
        row2 = ttk.Frame(options_frame)
        row2.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(row2, text="Zoom Factor:").pack(anchor=tk.W)
        self.zoom_var = tk.DoubleVar(value=1.0)
        zoom_scale = ttk.Scale(row2, from_=1.0, to=5.0, variable=self.zoom_var, orient=tk.HORIZONTAL)
        zoom_scale.pack(fill=tk.X, pady=(2, 0))
        self.zoom_label = ttk.Label(row2, text="1.0x")
        self.zoom_label.pack(anchor=tk.W)
        
        # Row 3: Center controls
        row3 = ttk.Frame(options_frame)
        row3.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(row3, text="Center X:").pack(side=tk.LEFT)
        self.center_x_var = tk.IntVar(value=50)
        center_x_scale = ttk.Scale(row3, from_=0, to=100, variable=self.center_x_var, orient=tk.HORIZONTAL)
        center_x_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        ttk.Label(row3, text="Y:").pack(side=tk.LEFT, padx=(10, 0))
        self.center_y_var = tk.IntVar(value=50)
        center_y_scale = ttk.Scale(row3, from_=0, to=100, variable=self.center_y_var, orient=tk.HORIZONTAL)
        center_y_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Row 4: Contour and colorbar options
        row4 = ttk.Frame(options_frame)
        row4.pack(fill=tk.X)
        
        self.add_contours_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row4, text="Contours", variable=self.add_contours_var).pack(side=tk.LEFT)
        
        self.show_colorbar_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row4, text="Colorbar", variable=self.show_colorbar_var).pack(side=tk.LEFT, padx=(10, 0))
        
        # Contour levels in a compact entry
        ttk.Label(row4, text="Levels:").pack(side=tk.LEFT, padx=(10, 0))
        self.contour_levels_var = tk.StringVar(value="0.1,0.3,0.5")
        contour_entry = ttk.Entry(row4, textvariable=self.contour_levels_var, width=10)
        contour_entry.pack(side=tk.LEFT, padx=(5, 0))
        
        # Add the missing colorbar label variable
        self.colorbar_label_var = tk.StringVar(value="Cell Density")
        
        # Add the missing contour colors variable
        self.contour_colors_var = tk.StringVar(value="white,yellow,red")
        
        # Plot buttons - always accessible
        button_frame = ttk.Frame(cell_control_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(button_frame, text="Plot Cell Field", command=self.plot_cell_field).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(button_frame, text="Reload Plot", command=self.reload_cell_plot).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Right panel for plot
        self.cell_plot_frame = ttk.Frame(main_container)
        self.cell_plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 10), pady=10)
        
        self.cell_fig = Figure(figsize=(10, 8), dpi=100)
        self.cell_canvas = FigureCanvasTkAgg(self.cell_fig, self.cell_plot_frame)
        self.cell_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.cell_toolbar = NavigationToolbar2Tk(self.cell_canvas, self.cell_plot_frame)
        self.cell_toolbar.update()
        
    def create_physics_fields_tab(self):
        """Create physics fields visualization tab"""
        physics_frame = ttk.Frame(self.notebook)
        self.notebook.add(physics_frame, text="Physics Fields")
        
        # Create main container with horizontal layout
        main_container = ttk.Frame(physics_frame)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Left panel for controls - fixed width, no scrolling
        physics_control_frame = ttk.Frame(main_container, width=300)
        physics_control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 5), pady=10)
        physics_control_frame.pack_propagate(False)  # Prevent shrinking
        
        # Field type selection
        ttk.Label(physics_control_frame, text="Field Type", font=("Arial", 10, "bold")).pack(pady=(0, 5))
        self.field_type_var = tk.StringVar()
        field_combo = ttk.Combobox(physics_control_frame, textvariable=self.field_type_var, 
                                  values=["Pressure", "Velocity", "Energy Derivative", "Mass Flux", "Source Terms"],
                                  state="readonly")
        field_combo.pack(fill=tk.X, pady=(0, 10))
        field_combo.set("Pressure")
        
        # Step selection - simplified
        ttk.Label(physics_control_frame, text="Time Step", font=("Arial", 10, "bold")).pack(pady=(0, 5))
        self.physics_step_var = tk.IntVar()
        self.physics_step_scale = ttk.Scale(physics_control_frame, from_=0, to=10, 
                                          variable=self.physics_step_var, orient=tk.HORIZONTAL)
        self.physics_step_scale.pack(fill=tk.X, pady=(0, 5))
        self.physics_step_label = ttk.Label(physics_control_frame, text="Step: 0")
        self.physics_step_label.pack()
        
        # Z-slice selection - simplified
        ttk.Label(physics_control_frame, text="Z-Slice", font=("Arial", 10, "bold")).pack(pady=(5, 5))
        self.physics_z_slice_var = tk.IntVar()
        self.physics_z_slice_scale = ttk.Scale(physics_control_frame, from_=0, to=10, 
                                             variable=self.physics_z_slice_var, orient=tk.HORIZONTAL)
        self.physics_z_slice_scale.pack(fill=tk.X, pady=(0, 5))
        self.physics_z_slice_label = ttk.Label(physics_control_frame, text="Z: 0")
        self.physics_z_slice_label.pack()
        
        # Main options in a single compact frame
        options_frame = ttk.LabelFrame(physics_control_frame, text="Plot Options", padding=5)
        options_frame.pack(fill=tk.X, pady=(5, 0))
        
        # Row 1: Colormap
        row1 = ttk.Frame(options_frame)
        row1.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(row1, text="Colormap:").pack(side=tk.LEFT)
        self.physics_colormap_var = tk.StringVar(value="viridis")
        physics_colormap_combo = ttk.Combobox(row1, textvariable=self.physics_colormap_var,
                                            values=["viridis", "plasma", "inferno", "magma", "jet", "hot", "cool", "RdBu_r"],
                                            state="readonly", width=15)
        physics_colormap_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Row 2: Zoom control (separate row for better usability)
        row2 = ttk.Frame(options_frame)
        row2.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(row2, text="Zoom Factor:").pack(anchor=tk.W)
        self.physics_zoom_var = tk.DoubleVar(value=1.0)
        physics_zoom_scale = ttk.Scale(row2, from_=1.0, to=5.0, variable=self.physics_zoom_var, orient=tk.HORIZONTAL)
        physics_zoom_scale.pack(fill=tk.X, pady=(2, 0))
        self.physics_zoom_label = ttk.Label(row2, text="1.0x")
        self.physics_zoom_label.pack(anchor=tk.W)
        
        # Row 3: Center controls
        row3 = ttk.Frame(options_frame)
        row3.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(row3, text="Center X:").pack(side=tk.LEFT)
        self.physics_center_x_var = tk.IntVar(value=50)
        ttk.Scale(row3, from_=0, to=100, variable=self.physics_center_x_var, orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        ttk.Label(row3, text="Y:").pack(side=tk.LEFT, padx=(10, 0))
        self.physics_center_y_var = tk.IntVar(value=50)
        ttk.Scale(row3, from_=0, to=100, variable=self.physics_center_y_var, orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Row 4: Arrow options for vector fields
        row4 = ttk.Frame(options_frame)
        row4.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(row4, text="Arrow Skip:").pack(side=tk.LEFT)
        self.arrow_skip_var = tk.IntVar(value=10)
        ttk.Scale(row4, from_=1, to=50, variable=self.arrow_skip_var, orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        ttk.Label(row4, text="Scale:").pack(side=tk.LEFT, padx=(10, 0))
        self.arrow_scale_var = tk.DoubleVar(value=50.0)
        ttk.Scale(row4, from_=1.0, to=200.0, variable=self.arrow_scale_var, orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Row 5: Boundary options
        row5 = ttk.Frame(options_frame)
        row5.pack(fill=tk.X)
        
        self.add_boundary_contours_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row5, text="Boundaries", variable=self.add_boundary_contours_var).pack(side=tk.LEFT)
        
        ttk.Label(row5, text="Level:").pack(side=tk.LEFT, padx=(10, 0))
        self.tumor_boundary_level_var = tk.DoubleVar(value=0.5)
        ttk.Scale(row5, from_=0.1, to=1.0, variable=self.tumor_boundary_level_var, orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        self.boundary_color_var = tk.StringVar(value="darkblue")
        boundary_color_combo = ttk.Combobox(row5, textvariable=self.boundary_color_var,
                                           values=["darkblue", "black", "red", "green", "blue", "yellow", "orange", "purple"],
                                           state="readonly", width=8)
        boundary_color_combo.pack(side=tk.LEFT, padx=(5, 0))
        
        # Plot buttons - always accessible
        button_frame = ttk.Frame(physics_control_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(button_frame, text="Plot Physics Field", command=self.plot_physics_field).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(button_frame, text="Reload Plot", command=self.reload_physics_plot).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Right panel for plot
        self.physics_plot_frame = ttk.Frame(main_container)
        self.physics_plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 10), pady=10)
        
        self.physics_fig = Figure(figsize=(10, 8), dpi=100)
        self.physics_canvas = FigureCanvasTkAgg(self.physics_fig, self.physics_plot_frame)
        self.physics_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.physics_toolbar = NavigationToolbar2Tk(self.physics_canvas, self.physics_plot_frame)
        self.physics_toolbar.update()
        
    def create_nutrient_tab(self):
        """Create nutrient field visualization tab"""
        nutrient_frame = ttk.Frame(self.notebook)
        self.notebook.add(nutrient_frame, text="Nutrient Fields")
        
        # Create a scrollable frame for controls
        canvas = tk.Canvas(nutrient_frame)
        scrollbar = ttk.Scrollbar(nutrient_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Left panel for controls
        nutrient_control_frame = ttk.Frame(scrollable_frame)
        nutrient_control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 5), pady=10)
        
        # Plot type selection
        ttk.Label(nutrient_control_frame, text="Plot Type", font=("Arial", 10, "bold")).pack(pady=(0, 5))
        self.nutrient_plot_type_var = tk.StringVar()
        nutrient_combo = ttk.Combobox(nutrient_control_frame, textvariable=self.nutrient_plot_type_var,
                                    values=["Field Evolution", "Statistics Evolution", "Tumor Correlation"],
                                    state="readonly")
        nutrient_combo.pack(fill=tk.X, pady=(0, 10))
        nutrient_combo.set("Field Evolution")
        
        # Step selection
        ttk.Label(nutrient_control_frame, text="Time Step", font=("Arial", 10, "bold")).pack(pady=(0, 5))
        self.nutrient_step_var = tk.IntVar()
        self.nutrient_step_scale = ttk.Scale(nutrient_control_frame, from_=0, to=10,
                                           variable=self.nutrient_step_var, orient=tk.HORIZONTAL)
        self.nutrient_step_scale.pack(fill=tk.X, pady=(0, 10))
        self.nutrient_step_label = ttk.Label(nutrient_control_frame, text="Step: 0")
        self.nutrient_step_label.pack()
        
        # Z-slice selection
        ttk.Label(nutrient_control_frame, text="Z-Slice", font=("Arial", 10, "bold")).pack(pady=(10, 5))
        self.nutrient_z_slice_var = tk.IntVar()
        self.nutrient_z_slice_scale = ttk.Scale(nutrient_control_frame, from_=0, to=10,
                                              variable=self.nutrient_z_slice_var, orient=tk.HORIZONTAL)
        self.nutrient_z_slice_scale.pack(fill=tk.X, pady=(0, 10))
        self.nutrient_z_slice_label = ttk.Label(nutrient_control_frame, text="Z: 0")
        self.nutrient_z_slice_label.pack()
        
        # Visualization options
        viz_frame = ttk.LabelFrame(nutrient_control_frame, text="Visualization Options", padding=5)
        viz_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Colormap selection
        ttk.Label(viz_frame, text="Colormap:").pack(anchor=tk.W)
        self.nutrient_colormap_var = tk.StringVar(value="viridis")
        nutrient_colormap_combo = ttk.Combobox(viz_frame, textvariable=self.nutrient_colormap_var,
                                              values=["viridis", "plasma", "inferno", "magma", "jet", "hot", "cool", "spring", "summer", "autumn", "winter"],
                                              state="readonly", width=15)
        nutrient_colormap_combo.pack(fill=tk.X, pady=(0, 10))
        
        # Max plots for evolution
        ttk.Label(viz_frame, text="Max Plots (Evolution):").pack(anchor=tk.W)
        self.nutrient_max_plots_var = tk.IntVar(value=6)
        ttk.Scale(viz_frame, from_=3, to=12, variable=self.nutrient_max_plots_var, orient=tk.HORIZONTAL).pack(fill=tk.X)
        
        # Correlation analysis options
        correlation_frame = ttk.LabelFrame(nutrient_control_frame, text="Correlation Analysis", padding=5)
        correlation_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(correlation_frame, text="Tumor Threshold:").pack(anchor=tk.W)
        self.tumor_threshold_var = tk.DoubleVar(value=0.1)
        ttk.Scale(correlation_frame, from_=0.01, to=0.5, variable=self.tumor_threshold_var, orient=tk.HORIZONTAL).pack(fill=tk.X)
        
        self.show_correlation_stats_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(correlation_frame, text="Show Correlation Statistics", variable=self.show_correlation_stats_var).pack(anchor=tk.W, pady=(10, 0))
        
        # Statistics options
        stats_frame = ttk.LabelFrame(nutrient_control_frame, text="Statistics Options", padding=5)
        stats_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.show_min_max_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(stats_frame, text="Show Min/Max", variable=self.show_min_max_var).pack(anchor=tk.W)
        
        self.show_mean_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(stats_frame, text="Show Mean", variable=self.show_mean_var).pack(anchor=tk.W)
        
        self.show_total_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(stats_frame, text="Show Total", variable=self.show_total_var).pack(anchor=tk.W)
        
        # Plot button
        ttk.Button(nutrient_control_frame, text="Plot Nutrient Field", command=self.plot_nutrient_field).pack(fill=tk.X, pady=(20, 0))
        
        # Pack the scrollable area
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Right panel for plot
        self.nutrient_plot_frame = ttk.Frame(nutrient_frame)
        self.nutrient_plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 10), pady=10)
        
        self.nutrient_fig = Figure(figsize=(10, 8), dpi=100)
        self.nutrient_canvas = FigureCanvasTkAgg(self.nutrient_fig, self.nutrient_plot_frame)
        self.nutrient_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.nutrient_toolbar = NavigationToolbar2Tk(self.nutrient_canvas, self.nutrient_plot_frame)
        self.nutrient_toolbar.update()
        
    def create_custom_tab(self):
        """Create advanced analysis tab"""
        custom_frame = ttk.Frame(self.notebook)
        self.notebook.add(custom_frame, text="Advanced Analysis")
        
        # Left panel for custom controls
        custom_control_frame = ttk.Frame(custom_frame)
        custom_control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 5), pady=10)
        
        ttk.Label(custom_control_frame, text="Advanced Analysis", font=("Arial", 12, "bold")).pack(pady=(0, 10))
        
        # Analysis options
        analysis_frame = ttk.LabelFrame(custom_control_frame, text="Analysis Options", padding=5)
        analysis_frame.pack(fill=tk.X, pady=(0, 10))
        
        analysis_buttons = [
            ("Free Energy Evolution", self.plot_free_energy),
            ("Population Evolution", self.plot_population_evolution),
            ("Source Field Evolution", self.plot_source_field_evolution),
            ("Nutrient-Tumor Correlation", self.plot_nutrient_tumor_correlation),
            ("Energy Derivative Evolution", self.plot_energy_derivative_evolution),
            ("Mass Flux Evolution", self.plot_mass_flux_evolution),
        ]
        
        for text, command in analysis_buttons:
            btn = ttk.Button(analysis_frame, text=text, command=command, width=25)
            btn.pack(pady=2, fill=tk.X)
        
        # Advanced parameters
        params_frame = ttk.LabelFrame(custom_control_frame, text="Advanced Parameters", padding=5)
        params_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Population selection for analysis
        ttk.Label(params_frame, text="Population for Analysis:").pack(anchor=tk.W)
        self.analysis_population_var = tk.StringVar()
        self.analysis_population_combo = ttk.Combobox(params_frame, textvariable=self.analysis_population_var, state="readonly")
        self.analysis_population_combo.pack(fill=tk.X, pady=(0, 10))
        
        # Figure size for advanced plots
        ttk.Label(params_frame, text="Figure Size:").pack(anchor=tk.W)
        adv_size_frame = ttk.Frame(params_frame)
        adv_size_frame.pack(fill=tk.X)
        
        ttk.Label(adv_size_frame, text="W:").pack(side=tk.LEFT)
        self.adv_width_var = tk.IntVar(value=15)
        ttk.Scale(adv_size_frame, from_=8, to=25, variable=self.adv_width_var, orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        ttk.Label(adv_size_frame, text="H:").pack(side=tk.LEFT, padx=(10, 0))
        self.adv_height_var = tk.IntVar(value=10)
        ttk.Scale(adv_size_frame, from_=6, to=20, variable=self.adv_height_var, orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Export options
        export_frame = ttk.LabelFrame(custom_control_frame, text="Export Options", padding=5)
        export_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(export_frame, text="Export All Observables", command=self.export_observables).pack(fill=tk.X, pady=2)
        ttk.Button(export_frame, text="Export Physics Data", command=self.export_physics_data).pack(fill=tk.X, pady=2)
        ttk.Button(export_frame, text="Save Current Plot", command=self.save_current_plot).pack(fill=tk.X, pady=2)
        ttk.Button(export_frame, text="Export All Plots", command=self.export_all_plots).pack(fill=tk.X, pady=2)
        
        # Right panel for plot
        self.custom_plot_frame = ttk.Frame(custom_frame)
        self.custom_plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 10), pady=10)
        
        self.custom_fig = Figure(figsize=(10, 8), dpi=100)
        self.custom_canvas = FigureCanvasTkAgg(self.custom_fig, self.custom_plot_frame)
        self.custom_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.custom_toolbar = NavigationToolbar2Tk(self.custom_canvas, self.custom_plot_frame)
        self.custom_toolbar.update()
        
    def create_status_frame(self, parent):
        """Create status and information frame"""
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, pady=(0, 0))
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var)
        self.status_label.pack(side=tk.LEFT)
        
        # Progress bar
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate')
        self.progress.pack(side=tk.RIGHT, padx=(10, 0))
        
    def load_simulation_file(self):
        """Load a simulation data file"""
        file_path = filedialog.askopenfilename(
            title="Select Simulation Data File",
            filetypes=[("NumPy files", "*.npz"), ("All files", "*.*")],
            initialdir=proj + "/laboratory/saved_simulations/"
        )
        
        if file_path:
            self.load_file(file_path)
            
    def convert_data_format(self, raw_data):
        """Convert raw npz data to expected format"""
        # Check if data is already in expected format
        if 'field_data' in raw_data and 'metadata' in raw_data:
            return raw_data
            
        # Convert from direct format to expected format
        converted_data = {}
        
        # Extract metadata
        if 'metadata' in raw_data:
            try:
                # Try to extract as dictionary if it's a single item
                metadata = raw_data['metadata'].item() if raw_data['metadata'].size == 1 else None
            except:
                metadata = None
                
        if metadata is None:
            # Create basic metadata from available data
            num_steps = len(raw_data['phi_hat'])
            
            # Handle time arrays - make sure they match the number of data steps
            if 'step_times' in raw_data:
                step_times = raw_data['step_times'].tolist()
                # Check if step_times contains meaningful values (not all zeros)
                if all(t == 0 for t in step_times):
                    # Create reasonable time values based on typical simulation parameters
                    # Assuming dt=0.1 and starting from t=0
                    step_times = [i * 0.1 for i in range(num_steps)]
                else:
                    # If step_times is shorter than data, pad with interpolated values
                    if len(step_times) < num_steps:
                        if len(step_times) > 1:
                            time_step = (step_times[-1] - step_times[0]) / (len(step_times) - 1)
                            step_times = [step_times[0] + i * time_step for i in range(num_steps)]
                        else:
                            step_times = [i * 0.1 for i in range(num_steps)]
                    elif len(step_times) > num_steps:
                        # Truncate if longer
                        step_times = step_times[:num_steps]
            else:
                # Create default time values
                step_times = [i * 0.1 for i in range(num_steps)]
            
            metadata = {
                'grid_size': raw_data['phi_hat'].shape[2:5],  # (nx, ny, nz) - skip time and population dimensions
                'num_populations': raw_data['phi_hat'].shape[1],
                'saved_steps': list(range(num_steps)),
                'saved_times': step_times,
                'population_labels': ['Stem Cells', 'Tumour Cells', 'Necrotic Cells']  # Default labels
            }
        
        # Create field_data structure
        field_data = {
            'phi_hat': raw_data['phi_hat'],
            'nutrient_fields': raw_data.get('nutrient_fields', None)
        }
        
        # Create physics_data structure
        physics_data = []
        num_steps = len(raw_data['phi_hat'])
        
        for i in range(num_steps):
            physics_step = {}
            if 'pressure' in raw_data:
                physics_step['pressure'] = raw_data['pressure'][i]
            if 'velocity' in raw_data:
                physics_step['velocity'] = raw_data['velocity'][i]
            if 'energy_derivative' in raw_data:
                physics_step['energy_derivative'] = raw_data['energy_derivative'][i]
            if 'mass_flux' in raw_data:
                physics_step['mass_flux'] = raw_data['mass_flux'][i]
            if 'source_terms' in raw_data:
                physics_step['source_terms'] = raw_data['source_terms'][i]
            physics_data.append(physics_step)
        
        converted_data = {
            'metadata': metadata,
            'field_data': field_data,
            'physics_data': physics_data
        }
        
        return converted_data
            
    def load_file(self, file_path):
        """Load simulation data from file"""
        try:
            self.set_loading(True)
            self.status_var.set("Loading simulation data...")
            
            # Load simulation data
            raw_data = np.load(file_path, allow_pickle=True)
            
            # Convert to expected format if needed
            self.simulation_data = self.convert_data_format(raw_data)
            
            # Create plotters (no need for full simulator for viewing)
            self.sim_plotter = SimPlotter(self.simulation_data)
            self.physics_plotter = PhysicsFieldsPlotter.from_simulation_data(self.simulation_data)
            self.cell_plotter = CellFieldPlotter(self.simulation_data, simulator=None)
            
            # Update GUI
            self.current_file = file_path
            self.file_label.config(text=f"Loaded: {Path(file_path).name}", foreground="black")
            
            # Update controls
            self.update_controls()
            
            # Add to recent files
            self.add_to_recent_files(file_path)
            
            self.status_var.set("Simulation data loaded successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load simulation data:\n{str(e)}")
            self.status_var.set("Error loading file")
        finally:
            self.set_loading(False)
            
    def reload_current_file(self):
        """Reload the current file"""
        if self.current_file:
            self.load_file(self.current_file)
            
    def set_loading(self, loading):
        """Set loading state"""
        self.loading = loading
        if loading:
            self.progress.start()
        else:
            self.progress.stop()
            
    def update_controls(self):
        """Update GUI controls based on loaded data"""
        if not self.simulation_data:
            return
            
        # Update population combos
        if hasattr(self.sim_plotter, 'labels'):
            self.population_combo['values'] = self.sim_plotter.labels
            if self.sim_plotter.labels:
                self.population_combo.set(self.sim_plotter.labels[0])
                
            # Update observables population combo
            self.observables_population_combo['values'] = self.sim_plotter.labels
            if self.sim_plotter.labels:
                self.observables_population_combo.set(self.sim_plotter.labels[0])
                
            # Update analysis population combo
            self.analysis_population_combo['values'] = self.sim_plotter.labels
            if self.sim_plotter.labels:
                self.analysis_population_combo.set(self.sim_plotter.labels[0])
        
        # Update step scales
        num_steps = len(self.simulation_data["metadata"]["saved_steps"])
        self.step_scale.config(to=num_steps-1)
        self.physics_step_scale.config(to=num_steps-1)
        self.nutrient_step_scale.config(to=num_steps-1)
        
        # Update z-slice scales
        grid_size = self.simulation_data["metadata"]["grid_size"]
        max_z = grid_size[2] - 1
        self.z_slice_scale.config(to=max_z)
        self.physics_z_slice_scale.config(to=max_z)
        self.nutrient_z_slice_scale.config(to=max_z)
        
        # Set default z-slice to middle
        middle_z = max_z // 2
        self.z_slice_var.set(middle_z)
        self.physics_z_slice_var.set(middle_z)
        self.nutrient_z_slice_var.set(middle_z)
        
        # Update time range scales
        if self.simulation_data["metadata"]["saved_times"]:
            max_time = max(self.simulation_data["metadata"]["saved_times"])
            self.start_time_var.set(0.0)
            self.end_time_var.set(max_time)
            # Update scale ranges
            for scale in [getattr(self, 'start_time_scale', None), getattr(self, 'end_time_scale', None)]:
                if scale:
                    scale.config(to=max_time)
        
        # Update center coordinates to grid center
        if grid_size:
            self.center_x_var.set(50)  # 50% = center
            self.center_y_var.set(50)  # 50% = center
            self.physics_center_x_var.set(50)
            self.physics_center_y_var.set(50)
        
        # Update labels
        self.update_step_labels()
        
    def update_threshold_label(self, value):
        """Update threshold label"""
        self.threshold_label.config(text=f"{float(value):.3f}")
        
    def update_max_plots_label(self, value):
        """Update max plots label"""
        self.max_plots_label.config(text=f"{int(float(value))}")
        
    def update_zoom_label(self, value):
        """Update zoom label"""
        self.zoom_label.config(text=f"{float(value):.1f}x")
        
    def update_physics_zoom_label(self, value):
        """Update physics zoom label"""
        self.physics_zoom_label.config(text=f"{float(value):.1f}x")
        
    def update_start_time_label(self, value):
        """Update start time label"""
        self.start_time_label.config(text=f"{float(value):.2f}")
        
    def update_end_time_label(self, value):
        """Update end time label"""
        self.end_time_label.config(text=f"{float(value):.2f}")
        
    def update_step_labels(self):
        """Update step and z-slice labels"""
        if not self.simulation_data:
            return
            
        # Update step labels
        step = int(self.step_var.get())
        if step < len(self.simulation_data["metadata"]["saved_steps"]):
            actual_step = self.simulation_data["metadata"]["saved_steps"][step]
            time = self.simulation_data["metadata"]["saved_times"][step]
            self.step_label.config(text=f"Step: {actual_step} (t={time:.3f})")
            
        step = int(self.physics_step_var.get())
        if step < len(self.simulation_data["metadata"]["saved_steps"]):
            actual_step = self.simulation_data["metadata"]["saved_steps"][step]
            time = self.simulation_data["metadata"]["saved_times"][step]
            self.physics_step_label.config(text=f"Step: {actual_step} (t={time:.3f})")
            
        step = int(self.nutrient_step_var.get())
        if step < len(self.simulation_data["metadata"]["saved_steps"]):
            actual_step = self.simulation_data["metadata"]["saved_steps"][step]
            time = self.simulation_data["metadata"]["saved_times"][step]
            self.nutrient_step_label.config(text=f"Step: {actual_step} (t={time:.3f})")
        
        # Update z-slice labels
        self.z_slice_label.config(text=f"Z: {int(self.z_slice_var.get())}")
        self.physics_z_slice_label.config(text=f"Z: {int(self.physics_z_slice_var.get())}")
        self.nutrient_z_slice_label.config(text=f"Z: {int(self.nutrient_z_slice_var.get())}")
        
    def check_data_loaded(self):
        """Check if simulation data is loaded"""
        if not self.simulation_data:
            messagebox.showwarning("No Data", "Please load a simulation data file first.")
            return False
        return True
        
    # Plot methods
    def plot_tumor_radius(self):
        """Plot tumor radius evolution"""
        if not self.check_data_loaded():
            return
            
        try:
            self.overview_fig.clear()
            
            # Create the plot directly in the GUI figure
            ax = self.overview_fig.add_subplot(111)
            
            # Get parameters from GUI
            threshold = self.threshold_var.get()
            method = self.radius_method_var.get()
            line_width = self.line_width_var.get()
            marker_size = self.marker_size_var.get()
            color_scheme = self.color_scheme_var.get()
            include_individual = self.include_individual_var.get()
            show_grid = self.show_grid_var.get()
            show_legend = self.show_legend_var.get()
            
            # Calculate radii for all populations
            radii_data = {}
            for i, label in enumerate(self.sim_plotter.labels):
                radii = []
                for step_idx in range(len(self.sim_plotter.saved_steps)):
                    phi_hat = self.sim_plotter.field_data["phi_hat"][step_idx]
                    radius = self.sim_plotter.utils.calculate_radius(phi_hat[i], threshold=threshold, method=method)
                    radii.append(radius)
                radii_data[label] = radii
            
            # Calculate total tumor radius
            total_radii = []
            for step_idx in range(len(self.sim_plotter.saved_steps)):
                phi_hat = self.sim_plotter.field_data["phi_hat"][step_idx]
                total_density = np.sum(phi_hat, axis=0)
                radius = self.sim_plotter.utils.calculate_radius(total_density, threshold=threshold, method=method)
                total_radii.append(radius)
            
            # Plot total radius
            ax.plot(self.sim_plotter.saved_times, total_radii, 'k-', linewidth=line_width, 
                   label='Total Tumor', marker='o', markersize=marker_size)
            
            # Plot individual population radii if requested
            if include_individual:
                colors = plt.cm.get_cmap(color_scheme)(np.linspace(0, 1, len(self.sim_plotter.labels)))
            for i, (label, radii) in enumerate(radii_data.items()):
                    ax.plot(self.sim_plotter.saved_times, radii, '--', color=colors[i], 
                           linewidth=line_width*0.8, label=f'{label}', marker='s', markersize=marker_size*0.7)
            
            ax.set_xlabel('Time')
            ax.set_ylabel('Radius')
            ax.set_title('Tumor Radius Evolution')
            
            if show_legend:
                ax.legend()
            if show_grid:
                ax.grid(True, alpha=0.3)
            
            self.overview_fig.tight_layout()
            self.overview_canvas.draw()
            self.status_var.set("Tumor radius evolution plotted")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot tumor radius evolution:\n{str(e)}")
            
    def plot_population_density(self):
        """Plot population density evolution"""
        if not self.check_data_loaded():
            return
            
        try:
            self.overview_fig.clear()
            
            # Create the plot directly in the GUI figure
            ax = self.overview_fig.add_subplot(111)
            
            # Calculate densities for all populations
            density_data = {}
            for i, label in enumerate(self.sim_plotter.labels):
                densities = []
                for step_idx in range(len(self.sim_plotter.saved_steps)):
                    phi_hat = self.sim_plotter.field_data["phi_hat"][step_idx]
                    density = self.sim_plotter.utils.calculate_total_density(phi_hat[i], normalize_by_volume=False)
                    densities.append(density)
                density_data[label] = densities
            
            # Calculate total density
            total_densities = []
            for step_idx in range(len(self.sim_plotter.saved_steps)):
                phi_hat = self.sim_plotter.field_data["phi_hat"][step_idx]
                total_density = np.sum(phi_hat, axis=0)
                density = self.sim_plotter.utils.calculate_total_density(total_density, normalize_by_volume=False)
                total_densities.append(density)
            
            # Plot total density
            ax.plot(self.sim_plotter.saved_times, total_densities, 'k-', linewidth=3, label='Total', marker='o', markersize=6)
            
            # Plot individual population densities
            colors = plt.cm.Set1(np.linspace(0, 1, len(self.sim_plotter.labels)))
            for i, (label, densities) in enumerate(density_data.items()):
                ax.plot(self.sim_plotter.saved_times, densities, '--', color=colors[i], linewidth=2, 
                       label=f'{label}', marker='s', markersize=4)
            
            ax.set_xlabel('Time')
            ax.set_ylabel('Total Density')
            ax.set_title('Population Density Evolution')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            self.overview_fig.tight_layout()
            self.overview_canvas.draw()
            self.status_var.set("Population density evolution plotted")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot population density evolution:\n{str(e)}")
            
    def plot_tumor_shape(self):
        """Plot tumor shape evolution"""
        if not self.check_data_loaded():
            return
            
        try:
            self.overview_fig.clear()
            
            # Select time points to plot
            max_plots = int(self.max_plots_var.get())
            if len(self.sim_plotter.saved_steps) <= max_plots:
                plot_indices = list(range(len(self.sim_plotter.saved_steps)))
            else:
                plot_indices = np.linspace(0, len(self.sim_plotter.saved_steps)-1, max_plots, dtype=int)
            
            # Calculate subplot layout
            num_plots = len(plot_indices)
            num_cols = min(3, num_plots)
            num_rows = (num_plots + num_cols - 1) // num_cols
            
            # Get middle z-slice
            z_slice = self.sim_plotter.grid_size[2] // 2
            threshold = self.threshold_var.get()
            
            # Create coordinate grids
            x = np.arange(self.sim_plotter.grid_size[0]) * self.sim_plotter.dx
            y = np.arange(self.sim_plotter.grid_size[1]) * self.sim_plotter.dx
            X, Y = np.meshgrid(x, y, indexing='ij')
            
            # Plot each time step
            for i, step_idx in enumerate(plot_indices):
                ax = self.overview_fig.add_subplot(num_rows, num_cols, i+1)
                
                # Get total density for this step
                phi_hat = self.sim_plotter.field_data["phi_hat"][step_idx]
                total_density = np.sum(phi_hat, axis=0)
                density_slice = total_density[:, :, z_slice]
                
                # Plot density as background
                im = ax.imshow(density_slice, extent=[x[0], x[-1], y[0], y[-1]], 
                              origin='lower', cmap='viridis', aspect='equal')
                
                # Add contour for tumor boundary
                ax.contour(X, Y, density_slice, levels=[threshold], colors='red', 
                          linewidths=2, alpha=0.8)
                
                # Set title and labels
                time = self.sim_plotter.saved_times[step_idx]
                step = self.sim_plotter.saved_steps[step_idx]
                ax.set_title(f'Step {step} (t={time:.2f})')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
            
            self.overview_fig.suptitle('Tumor Shape Evolution', fontsize=16)
            self.overview_fig.tight_layout()
            self.overview_canvas.draw()
            self.status_var.set("Tumor shape evolution plotted")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot tumor shape evolution:\n{str(e)}")
            
    def plot_center_of_mass(self):
        """Plot center of mass evolution"""
        if not self.check_data_loaded():
            return
            
        try:
            self.overview_fig.clear()
            self.sim_plotter.plot_center_of_mass_evolution(
                output_dir=None, save_plot=False, show_plot=False,
                figsize=(10, 8), include_individual_populations=True
            )
            self.overview_canvas.draw()
            self.status_var.set("Center of mass evolution plotted")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot center of mass evolution:\n{str(e)}")
            
    def plot_compactness(self):
        """Plot compactness evolution"""
        if not self.check_data_loaded():
            return
            
        try:
            self.overview_fig.clear()
            self.sim_plotter.plot_compactness_evolution(
                output_dir=None, save_plot=False, show_plot=False,
                figsize=(10, 8), include_individual_populations=True
            )
            self.overview_canvas.draw()
            self.status_var.set("Compactness evolution plotted")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot compactness evolution:\n{str(e)}")
            
    def plot_all_observables(self):
        """Plot all observables"""
        if not self.check_data_loaded():
            return
            
        try:
            self.observables_fig.clear()
            self.sim_plotter.plot_all_observables(
                output_dir=None, save_plot=False, show_plot=False,
                figsize=(12, 8), threshold=self.threshold_var.get()
            )
            self.observables_canvas.draw()
            self.status_var.set("All observables plotted")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot all observables:\n{str(e)}")
            
    def plot_cell_field(self):
        """Plot cell field"""
        if not self.check_data_loaded():
            return
            
        try:
            self.cell_fig.clear()
            
            # Get parameters
            population = self.population_var.get()
            step_idx = int(self.step_var.get())
            z_slice = int(self.z_slice_var.get())
            zoom_factor = self.zoom_var.get()
            add_contours = self.add_contours_var.get()
            colormap = self.cell_colormap_var.get()
            center_x = self.center_x_var.get()
            center_y = self.center_y_var.get()
            show_colorbar = self.show_colorbar_var.get()
            colorbar_label = self.colorbar_label_var.get()
            
            # Parse contour levels and colors
            contour_levels_str = self.contour_levels_var.get()
            contour_colors_str = self.contour_colors_var.get()
            
            try:
                contour_levels = [float(x.strip()) for x in contour_levels_str.split(',')]
                contour_colors = [x.strip() for x in contour_colors_str.split(',')]
            except:
                contour_levels = [0.1, 0.3, 0.5]
                contour_colors = ['white', 'yellow', 'red']
            
            # Find population index
            if population in self.sim_plotter.labels:
                pop_idx = self.sim_plotter.labels.index(population)
            else:
                pop_idx = None  # Total density
            
            # Get the data
            phi_hat = self.simulation_data["field_data"]["phi_hat"][step_idx]
            
            # Create the plot directly in the GUI figure
            ax = self.cell_fig.add_subplot(111)
            
            # Get grid information
            nx, ny, nz = self.sim_plotter.grid_size
            dx = self.sim_plotter.dx
            
            # Use center coordinates from GUI (convert from percentage to actual coordinates)
            center_x_coord = int(center_x * nx / 100) if center_x > 0 else nx // 2
            center_y_coord = int(center_y * ny / 100) if center_y > 0 else ny // 2
            
            # Calculate zoom window size
            window_size = min(nx, ny) // zoom_factor
            half_window = window_size // 2
            
            # Calculate window boundaries
            x_start = int(max(0, center_x_coord - half_window))
            x_end = int(min(nx, center_x_coord + half_window))
            y_start = int(max(0, center_y_coord - half_window))
            y_end = int(min(ny, center_y_coord + half_window))
            
            # Create coordinate grids for zoomed region
            x_region = np.arange(x_start, x_end) * dx
            y_region = np.arange(y_start, y_end) * dx
            
            # Extract density data for zoomed region
            if pop_idx is None:
                # Plot total density (sum of all populations)
                density_slice = np.sum(phi_hat, axis=0)[x_start:x_end, y_start:y_end, z_slice]
                population_name = "Total"
            else:
                # Plot specific population
                density_slice = phi_hat[pop_idx, x_start:x_end, y_start:y_end, z_slice]
                population_name = self.sim_plotter.labels[pop_idx]
            
            # Create the plot
            im = ax.imshow(density_slice, extent=[x_region[0], x_region[-1], y_region[0], y_region[-1]], 
                          origin='lower', cmap=colormap, aspect='equal')
            
            # Add contours if requested
            if add_contours:
                for i, level in enumerate(contour_levels):
                    color = contour_colors[i] if i < len(contour_colors) else 'black'
                    ax.contour(density_slice, levels=[level], colors=color, 
                              linewidths=2, alpha=0.8,
                              extent=[x_region[0], x_region[-1], y_region[0], y_region[-1]], 
                              origin='lower')
            
            # Add center point marker if zoomed
            if zoom_factor > 1.0:
                center_x_phys = center_x_coord * dx
                center_y_phys = center_y_coord * dx
                ax.plot(center_x_phys, center_y_phys, 'k+', markersize=10, markeredgewidth=2, 
                       label=f'Center: ({center_x_phys:.2f}, {center_y_phys:.2f})')
                ax.legend()
                ax.set_title(f'Cell Density - {population_name} (z={z_slice}) - Zoomed View')
            else:
                ax.set_title(f'Cell Density - {population_name} (z={z_slice})')
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            
            # Add colorbar if requested
            if show_colorbar:
                self.cell_fig.colorbar(im, ax=ax, label=colorbar_label)
            
            self.cell_fig.tight_layout()
            self.cell_canvas.draw()
            self.status_var.set(f"Cell field plotted: {population_name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot cell field:\n{str(e)}")
            
    def reload_cell_plot(self):
        """Reload the cell field plot with current settings"""
        self.plot_cell_field()
            
    def plot_physics_field(self):
        """Plot physics field"""
        if not self.check_data_loaded():
            return
            
        try:
            self.physics_fig.clear()
            
            # Get parameters
            field_type = self.field_type_var.get()
            step_idx = int(self.physics_step_var.get())
            z_slice = int(self.physics_z_slice_var.get())
            
            # Create the plot directly in the GUI figure
            ax = self.physics_fig.add_subplot(111)
            
            # Get grid information
            nx, ny, nz = self.sim_plotter.grid_size
            dx = self.sim_plotter.dx
            
            # Create coordinate grids
            x = np.arange(nx) * dx
            y = np.arange(ny) * dx
            
            # Plot based on field type
            if field_type == "Pressure":
                if "physics_data" in self.simulation_data and len(self.simulation_data["physics_data"]) > step_idx:
                    pressure = self.simulation_data["physics_data"][step_idx]["pressure"]
                    pressure_slice = pressure[:, :, z_slice]
                    
                    # Handle NaN and infinite values
                    pressure_slice = np.nan_to_num(pressure_slice, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    im = ax.imshow(pressure_slice, extent=[x[0], x[-1], y[0], y[-1]], 
                                  origin='lower', cmap=self.physics_colormap_var.get(), aspect='equal')
                    ax.set_title(f'Pressure Field (z={z_slice})')
                    self.physics_fig.colorbar(im, ax=ax, label='Pressure')
                else:
                    ax.text(0.5, 0.5, 'No pressure data available', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title('Pressure Field - No Data')
                    
            elif field_type == "Velocity":
                if "physics_data" in self.simulation_data and len(self.simulation_data["physics_data"]) > step_idx:
                    velocity = self.simulation_data["physics_data"][step_idx]["velocity"]
                    ux_slice = velocity[0, :, :, z_slice]
                    uy_slice = velocity[1, :, :, z_slice]
                    
                    # Sample for visualization
                    skip = int(self.arrow_skip_var.get())
                    x_skip = x[::skip]
                    y_skip = y[::skip]
                    X_skip, Y_skip = np.meshgrid(x_skip, y_skip, indexing='ij')
                    ux_skip = ux_slice[::skip, ::skip]
                    uy_skip = uy_slice[::skip, ::skip]
                    
                    # Normalize vectors
                    magnitude = np.sqrt(ux_skip**2 + uy_skip**2)
                    max_mag = np.max(magnitude) if np.max(magnitude) > 0 else 1.0
                    ux_norm = ux_skip / max_mag
                    uy_norm = uy_skip / max_mag
                    
                    quiv = ax.quiver(X_skip, Y_skip, ux_norm, uy_norm, magnitude, 
                                    cmap=self.physics_colormap_var.get(), 
                                    scale=self.arrow_scale_var.get(), width=0.005)
                    ax.set_title(f'Velocity Field (z={z_slice})')
                    self.physics_fig.colorbar(quiv, ax=ax)
                else:
                    ax.text(0.5, 0.5, 'No velocity data available', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title('Velocity Field - No Data')
                    
            elif field_type == "Energy Derivative":
                if "physics_data" in self.simulation_data and len(self.simulation_data["physics_data"]) > step_idx:
                    energy = self.simulation_data["physics_data"][step_idx]["energy_derivative"]
                    energy_slice = energy[:, :, z_slice]
                    
                    # Handle NaN and infinite values
                    energy_slice = np.nan_to_num(energy_slice, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # Use symmetric colormap limits
                    vmax = np.max(np.abs(energy_slice))
                    vmin = -vmax if vmax > 0 else -1.0
                    
                    im = ax.imshow(energy_slice, extent=[x[0], x[-1], y[0], y[-1]], 
                                  origin='lower', cmap=self.physics_colormap_var.get(), aspect='equal', vmin=vmin, vmax=vmax)
                    ax.set_title(f'Energy Derivative Field (z={z_slice})')
                    self.physics_fig.colorbar(im, ax=ax)
                else:
                    ax.text(0.5, 0.5, 'No energy derivative data available', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title('Energy Derivative Field - No Data')
                    
            elif field_type == "Mass Flux":
                if "physics_data" in self.simulation_data and len(self.simulation_data["physics_data"]) > step_idx:
                    mass_flux = self.simulation_data["physics_data"][step_idx]["mass_flux"]
                    flux_x_slice = mass_flux[0, 0, :, :, z_slice]  # First population, x-component
                    flux_y_slice = mass_flux[0, 1, :, :, z_slice]  # First population, y-component
                    
                    # Sample for visualization
                    skip = int(self.arrow_skip_var.get())
                    x_skip = x[::skip]
                    y_skip = y[::skip]
                    X_skip, Y_skip = np.meshgrid(x_skip, y_skip, indexing='ij')
                    flux_x_skip = flux_x_slice[::skip, ::skip]
                    flux_y_skip = flux_y_slice[::skip, ::skip]
                    
                    # Normalize vectors
                    magnitude = np.sqrt(flux_x_skip**2 + flux_y_skip**2)
                    max_mag = np.max(magnitude) if np.max(magnitude) > 0 else 1.0
                    flux_x_norm = flux_x_skip / max_mag
                    flux_y_norm = flux_y_skip / max_mag
                    
                    quiv = ax.quiver(X_skip, Y_skip, flux_x_norm, flux_y_norm, magnitude, 
                                    cmap=self.physics_colormap_var.get(), 
                                    scale=self.arrow_scale_var.get()/2, width=0.005)
                    ax.set_title(f'Mass Flux - Stem Cells (z={z_slice})')
                    self.physics_fig.colorbar(quiv, ax=ax)
                else:
                    ax.text(0.5, 0.5, 'No mass flux data available', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title('Mass Flux Field - No Data')
                    
            elif field_type == "Source Terms":
                if "physics_data" in self.simulation_data and len(self.simulation_data["physics_data"]) > step_idx:
                    source_terms = self.simulation_data["physics_data"][step_idx]["source_terms"]
                    source_slice = source_terms[0, :, :, z_slice]  # First population
                    
                    # Handle NaN and infinite values
                    source_slice = np.nan_to_num(source_slice, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # Use symmetric colormap limits
                    vmax = np.max(np.abs(source_slice))
                    vmin = -vmax if vmax > 0 else -1.0
                    
                    im = ax.imshow(source_slice, extent=[x[0], x[-1], y[0], y[-1]], 
                                  origin='lower', cmap='RdBu_r', aspect='equal', vmin=vmin, vmax=vmax)
                    ax.set_title(f'Source Terms - Stem Cells (z={z_slice})')
                    self.physics_fig.colorbar(im, ax=ax)
                else:
                    ax.text(0.5, 0.5, 'No source terms data available', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title('Source Terms - No Data')
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            
            self.physics_fig.tight_layout()
            self.physics_canvas.draw()
            self.status_var.set(f"Physics field plotted: {field_type}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot physics field:\n{str(e)}")
            
    def reload_physics_plot(self):
        """Reload the physics field plot with current settings"""
        self.plot_physics_field()
            
    def plot_nutrient_field(self):
        """Plot nutrient field"""
        if not self.check_data_loaded():
            return
            
        try:
            self.nutrient_fig.clear()
            
            # Get parameters
            plot_type = self.nutrient_plot_type_var.get()
            step_idx = int(self.nutrient_step_var.get())
            z_slice = int(self.nutrient_z_slice_var.get())
            
            # Check if nutrient fields are available
            if "nutrient_fields" not in self.simulation_data["field_data"]:
                ax = self.nutrient_fig.add_subplot(111)
                ax.text(0.5, 0.5, 'No nutrient fields found in simulation data', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Nutrient Field - No Data')
                self.nutrient_canvas.draw()
                self.status_var.set("No nutrient data available")
                return
            
            # Plot based on type
            if plot_type == "Field Evolution":
                # Create subplot layout for multiple time steps
                max_plots = 6
                if len(self.sim_plotter.saved_steps) <= max_plots:
                    plot_indices = list(range(len(self.sim_plotter.saved_steps)))
                else:
                    plot_indices = np.linspace(0, len(self.sim_plotter.saved_steps)-1, max_plots, dtype=int)
                
                num_plots = len(plot_indices)
                num_cols = min(3, num_plots)
                num_rows = (num_plots + num_cols - 1) // num_cols
                
                # Get grid information
                nx, ny, nz = self.sim_plotter.grid_size
                dx = self.sim_plotter.dx
                x = np.arange(nx) * dx
                y = np.arange(ny) * dx
                
                # Find global min/max for consistent color scaling
                all_nutrient_data = []
                for step_idx in plot_indices:
                    nutrient_field = self.simulation_data["field_data"]["nutrient_fields"][step_idx]
                    nutrient_slice = nutrient_field[:, :, z_slice]
                    all_nutrient_data.append(nutrient_slice)
                
                vmin = np.min(all_nutrient_data)
                vmax = np.max(all_nutrient_data)
                
                # Plot each time step
                for i, step_idx in enumerate(plot_indices):
                    ax = self.nutrient_fig.add_subplot(num_rows, num_cols, i+1)
                    
                    # Get step information
                    step = self.sim_plotter.saved_steps[step_idx]
                    time = self.sim_plotter.saved_times[step_idx]
                    
                    # Get nutrient field for this step
                    nutrient_field = self.simulation_data["field_data"]["nutrient_fields"][step_idx]
                    nutrient_slice = nutrient_field[:, :, z_slice]
                    
                    # Create the plot with consistent color scaling
                    im = ax.imshow(nutrient_slice, extent=[x[0], x[-1], y[0], y[-1]], 
                                  origin='lower', cmap='viridis', aspect='equal', vmin=vmin, vmax=vmax)
                    
                    # Set title and labels
                    ax.set_title(f'Step {step} (t={time:.2f})')
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    
                    # Add colorbar to the last plot
                    if i == num_plots - 1:
                        self.nutrient_fig.colorbar(im, ax=ax, label='Nutrient Concentration')
                
                self.nutrient_fig.suptitle(f'Nutrient Field Evolution (z={z_slice})', fontsize=16)
                
            elif plot_type == "Statistics Evolution":
                # Calculate statistics for each time step
                min_concentrations = []
                max_concentrations = []
                mean_concentrations = []
                total_concentrations = []
                
                for step_idx in range(len(self.sim_plotter.saved_steps)):
                    nutrient_field = self.simulation_data["field_data"]["nutrient_fields"][step_idx]
                    nutrient_slice = nutrient_field[:, :, z_slice]
                    
                    min_concentrations.append(np.min(nutrient_slice))
                    max_concentrations.append(np.max(nutrient_slice))
                    mean_concentrations.append(np.mean(nutrient_slice))
                    total_concentrations.append(np.sum(nutrient_slice))
                
                # Create subplots
                axes = self.nutrient_fig.subplots(2, 2)
                
                # Plot 1: Min concentration
                axes[0, 0].plot(self.sim_plotter.saved_times, min_concentrations, 'b-', linewidth=2, marker='o', markersize=4)
                axes[0, 0].set_xlabel('Time')
                axes[0, 0].set_ylabel('Min Concentration')
                axes[0, 0].set_title('Minimum Nutrient Concentration')
                axes[0, 0].grid(True, alpha=0.3)
                
                # Plot 2: Max concentration
                axes[0, 1].plot(self.sim_plotter.saved_times, max_concentrations, 'r-', linewidth=2, marker='o', markersize=4)
                axes[0, 1].set_xlabel('Time')
                axes[0, 1].set_ylabel('Max Concentration')
                axes[0, 1].set_title('Maximum Nutrient Concentration')
                axes[0, 1].grid(True, alpha=0.3)
                
                # Plot 3: Mean concentration
                axes[1, 0].plot(self.sim_plotter.saved_times, mean_concentrations, 'g-', linewidth=2, marker='o', markersize=4)
                axes[1, 0].set_xlabel('Time')
                axes[1, 0].set_ylabel('Mean Concentration')
                axes[1, 0].set_title('Mean Nutrient Concentration')
                axes[1, 0].grid(True, alpha=0.3)
                
                # Plot 4: Total concentration
                axes[1, 1].plot(self.sim_plotter.saved_times, total_concentrations, 'm-', linewidth=2, marker='o', markersize=4)
                axes[1, 1].set_xlabel('Time')
                axes[1, 1].set_ylabel('Total Concentration')
                axes[1, 1].set_title('Total Nutrient Concentration')
                axes[1, 1].grid(True, alpha=0.3)
                
            elif plot_type == "Tumor Correlation":
                # Calculate correlation metrics for each time step
                correlations = []
                tumor_mean_nutrients = []
                background_mean_nutrients = []
                tumor_threshold = 0.1
                
                for step_idx in range(len(self.sim_plotter.saved_steps)):
                    # Get nutrient field
                    nutrient_field = self.simulation_data["field_data"]["nutrient_fields"][step_idx]
                    nutrient_slice = nutrient_field[:, :, z_slice]
                    
                    # Get tumor density
                    phi_hat = self.simulation_data["field_data"]["phi_hat"][step_idx]
                    total_density = np.sum(phi_hat, axis=0)
                    density_slice = total_density[:, :, z_slice]
                    
                    # Create tumor mask
                    tumor_mask = density_slice > tumor_threshold
                    background_mask = ~tumor_mask
                    
                    # Calculate correlation
                    if np.any(tumor_mask) and np.any(background_mask):
                        correlation = np.corrcoef(density_slice.flatten(), nutrient_slice.flatten())[0, 1]
                        correlations.append(correlation if not np.isnan(correlation) else 0.0)
                        
                        # Calculate mean nutrient levels
                        tumor_mean_nutrients.append(np.mean(nutrient_slice[tumor_mask]))
                        background_mean_nutrients.append(np.mean(nutrient_slice[background_mask]))
                    else:
                        correlations.append(0.0)
                        tumor_mean_nutrients.append(0.0)
                        background_mean_nutrients.append(np.mean(nutrient_slice))
                
                # Create subplots
                axes = self.nutrient_fig.subplots(2, 2)
                
                # Plot 1: Correlation over time
                axes[0, 0].plot(self.sim_plotter.saved_times, correlations, 'b-', linewidth=2, marker='o', markersize=4)
                axes[0, 0].set_xlabel('Time')
                axes[0, 0].set_ylabel('Correlation Coefficient')
                axes[0, 0].set_title('Nutrient-Tumor Correlation')
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
                
                # Plot 2: Mean nutrient in tumor vs background
                axes[0, 1].plot(self.sim_plotter.saved_times, tumor_mean_nutrients, 'r-', linewidth=2, marker='o', markersize=4, label='Tumor Region')
                axes[0, 1].plot(self.sim_plotter.saved_times, background_mean_nutrients, 'g-', linewidth=2, marker='s', markersize=4, label='Background')
                axes[0, 1].set_xlabel('Time')
                axes[0, 1].set_ylabel('Mean Nutrient Concentration')
                axes[0, 1].set_title('Mean Nutrient Levels')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
                
                # Plot 3: Nutrient difference (tumor - background)
                axes[1, 0].plot(self.sim_plotter.saved_times, np.array(tumor_mean_nutrients) - np.array(background_mean_nutrients), 'm-', linewidth=2, marker='o', markersize=4)
                axes[1, 0].set_xlabel('Time')
                axes[1, 0].set_ylabel('Nutrient Difference')
                axes[1, 0].set_title('Tumor - Background Nutrient Difference')
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
                
                # Plot 4: Scatter plot for last time step
                last_nutrient = self.simulation_data["field_data"]["nutrient_fields"][-1][:, :, z_slice]
                last_density = np.sum(self.simulation_data["field_data"]["phi_hat"][-1], axis=0)[:, :, z_slice]
                
                # Sample points for scatter plot (every 4th point to avoid overcrowding)
                sample_mask = np.zeros_like(last_density, dtype=bool)
                sample_mask[::4, ::4] = True
                
                axes[1, 1].scatter(last_density[sample_mask], last_nutrient[sample_mask], alpha=0.6, s=10)
                axes[1, 1].set_xlabel('Tumor Density')
                axes[1, 1].set_ylabel('Nutrient Concentration')
                axes[1, 1].set_title(f'Nutrient vs Density (t={self.sim_plotter.saved_times[-1]:.2f})')
                axes[1, 1].grid(True, alpha=0.3)
            
            self.nutrient_fig.tight_layout()
            self.nutrient_canvas.draw()
            self.status_var.set(f"Nutrient field plotted: {plot_type}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot nutrient field:\n{str(e)}")
            
    def plot_free_energy(self):
        """Plot free energy evolution"""
        if not self.check_data_loaded():
            return
            
        try:
            self.custom_fig.clear()
            self.physics_plotter.plot_total_free_energy_evolution(
                self.simulation_data, output_dir=None, save_plot=False, show_plot=False
            )
            self.custom_canvas.draw()
            self.status_var.set("Free energy evolution plotted")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot free energy evolution:\n{str(e)}")
            
    def plot_population_evolution(self):
        """Plot population evolution"""
        if not self.check_data_loaded():
            return
            
        try:
            self.custom_fig.clear()
            
            # Get selected population
            population = self.analysis_population_var.get()
            if not population and self.sim_plotter.labels:
                population = self.sim_plotter.labels[0]
            
            if population:
                self.cell_plotter.plot_population_evolution_by_label(
                    self.simulation_data, label=population,
                    output_dir=None, save_plot=False, show_plot=False
                )
            
            self.custom_canvas.draw()
            self.status_var.set("Population evolution plotted")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot population evolution:\n{str(e)}")
            
    def plot_source_field_evolution(self):
        """Plot source field evolution"""
        if not self.check_data_loaded():
            return
            
        try:
            self.custom_fig.clear()
            self.sim_plotter.plot_source_field_evolution(
                output_dir=None, save_plot=False, show_plot=False,
                figsize=(self.adv_width_var.get(), self.adv_height_var.get())
            )
            self.custom_canvas.draw()
            self.status_var.set("Source field evolution plotted")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot source field evolution:\n{str(e)}")
            
    def plot_nutrient_tumor_correlation(self):
        """Plot nutrient-tumor correlation"""
        if not self.check_data_loaded():
            return
            
        try:
            self.custom_fig.clear()
            self.sim_plotter.plot_nutrient_tumor_correlation(
                output_dir=None, save_plot=False, show_plot=False,
                figsize=(self.adv_width_var.get(), self.adv_height_var.get()),
                tumor_threshold=self.tumor_threshold_var.get()
            )
            self.custom_canvas.draw()
            self.status_var.set("Nutrient-tumor correlation plotted")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot nutrient-tumor correlation:\n{str(e)}")
            
    def plot_energy_derivative_evolution(self):
        """Plot energy derivative evolution"""
        if not self.check_data_loaded():
            return
            
        try:
            self.custom_fig.clear()
            
            # Create the plot directly in the GUI figure
            ax = self.custom_fig.add_subplot(111)
            
            # Check if physics data is available
            if "physics_data" not in self.simulation_data or len(self.simulation_data["physics_data"]) == 0:
                ax.text(0.5, 0.5, 'No physics data available for energy derivative calculation', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Energy Derivative Evolution - No Data')
                self.custom_canvas.draw()
                self.status_var.set("No physics data available")
                return
            
            # Calculate energy derivative statistics for each step
            energy_stats = []
            for step_idx in range(len(self.simulation_data["physics_data"])):
                physics_data = self.simulation_data["physics_data"][step_idx]
                if "energy_derivative" in physics_data:
                    energy_deriv = physics_data["energy_derivative"]
                    # Calculate statistics
                    energy_stats.append({
                        'mean': np.mean(energy_deriv),
                        'std': np.std(energy_deriv),
                        'max': np.max(energy_deriv),
                        'min': np.min(energy_deriv),
                        'total': np.sum(np.abs(energy_deriv))
                    })
                else:
                    energy_stats.append({'mean': 0, 'std': 0, 'max': 0, 'min': 0, 'total': 0})
            
            # Extract data
            times = self.sim_plotter.saved_times[:len(energy_stats)]
            means = [stat['mean'] for stat in energy_stats]
            stds = [stat['std'] for stat in energy_stats]
            maxs = [stat['max'] for stat in energy_stats]
            mins = [stat['min'] for stat in energy_stats]
            totals = [stat['total'] for stat in energy_stats]
            
            # Create subplots
            axes = self.custom_fig.subplots(2, 2)
            
            # Plot 1: Mean energy derivative
            axes[0, 0].plot(times, means, 'b-', linewidth=2, marker='o', markersize=4)
            axes[0, 0].set_xlabel('Time')
            axes[0, 0].set_ylabel('Mean Energy Derivative')
            axes[0, 0].set_title('Mean Energy Derivative')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Standard deviation
            axes[0, 1].plot(times, stds, 'r-', linewidth=2, marker='o', markersize=4)
            axes[0, 1].set_xlabel('Time')
            axes[0, 1].set_ylabel('Std Energy Derivative')
            axes[0, 1].set_title('Energy Derivative Variability')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Min/Max range
            axes[1, 0].plot(times, maxs, 'g-', linewidth=2, marker='o', markersize=4, label='Max')
            axes[1, 0].plot(times, mins, 'orange', linewidth=2, marker='s', markersize=4, label='Min')
            axes[1, 0].set_xlabel('Time')
            axes[1, 0].set_ylabel('Energy Derivative Range')
            axes[1, 0].set_title('Energy Derivative Range')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Total absolute energy derivative
            axes[1, 1].plot(times, totals, 'm-', linewidth=2, marker='o', markersize=4)
            axes[1, 1].set_xlabel('Time')
            axes[1, 1].set_ylabel('Total |Energy Derivative|')
            axes[1, 1].set_title('Total Energy Derivative Magnitude')
            axes[1, 1].grid(True, alpha=0.3)
            
            self.custom_fig.suptitle('Energy Derivative Evolution Analysis', fontsize=16)
            self.custom_fig.tight_layout()
            self.custom_canvas.draw()
            self.status_var.set("Energy derivative evolution plotted")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot energy derivative evolution:\n{str(e)}")
            
    def plot_mass_flux_evolution(self):
        """Plot mass flux evolution"""
        if not self.check_data_loaded():
            return
            
        try:
            self.custom_fig.clear()
            
            # Create the plot directly in the GUI figure
            ax = self.custom_fig.add_subplot(111)
            
            # Check if physics data is available
            if "physics_data" not in self.simulation_data or len(self.simulation_data["physics_data"]) == 0:
                ax.text(0.5, 0.5, 'No physics data available for mass flux calculation', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Mass Flux Evolution - No Data')
                self.custom_canvas.draw()
                self.status_var.set("No physics data available")
                return
            
            # Calculate mass flux statistics for each step
            flux_stats = []
            for step_idx in range(len(self.simulation_data["physics_data"])):
                physics_data = self.simulation_data["physics_data"][step_idx]
                if "mass_flux" in physics_data:
                    mass_flux = physics_data["mass_flux"]
                    # Calculate magnitude for each population
                    flux_magnitudes = []
                    for pop_idx in range(mass_flux.shape[0]):
                        flux_x = mass_flux[pop_idx, 0]  # x-component
                        flux_y = mass_flux[pop_idx, 1]  # y-component
                        flux_z = mass_flux[pop_idx, 2]  # z-component
                        magnitude = np.sqrt(flux_x**2 + flux_y**2 + flux_z**2)
                        flux_magnitudes.append(np.mean(magnitude))
                    
                    flux_stats.append({
                        'mean_magnitude': np.mean(flux_magnitudes),
                        'max_magnitude': np.max(flux_magnitudes),
                        'total_flux': np.sum([np.sum(np.abs(mass_flux[pop_idx])) for pop_idx in range(mass_flux.shape[0])])
                    })
                else:
                    flux_stats.append({'mean_magnitude': 0, 'max_magnitude': 0, 'total_flux': 0})
            
            # Extract data
            times = self.sim_plotter.saved_times[:len(flux_stats)]
            mean_magnitudes = [stat['mean_magnitude'] for stat in flux_stats]
            max_magnitudes = [stat['max_magnitude'] for stat in flux_stats]
            total_fluxes = [stat['total_flux'] for stat in flux_stats]
            
            # Create subplots
            axes = self.custom_fig.subplots(2, 2)
            
            # Plot 1: Mean flux magnitude
            axes[0, 0].plot(times, mean_magnitudes, 'b-', linewidth=2, marker='o', markersize=4)
            axes[0, 0].set_xlabel('Time')
            axes[0, 0].set_ylabel('Mean Flux Magnitude')
            axes[0, 0].set_title('Mean Mass Flux Magnitude')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Max flux magnitude
            axes[0, 1].plot(times, max_magnitudes, 'r-', linewidth=2, marker='o', markersize=4)
            axes[0, 1].set_xlabel('Time')
            axes[0, 1].set_ylabel('Max Flux Magnitude')
            axes[0, 1].set_title('Maximum Mass Flux Magnitude')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Total flux
            axes[1, 0].plot(times, total_fluxes, 'g-', linewidth=2, marker='o', markersize=4)
            axes[1, 0].set_xlabel('Time')
            axes[1, 0].set_ylabel('Total Flux')
            axes[1, 0].set_title('Total Mass Flux')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Flux ratio (max/mean)
            flux_ratios = [max_mag/mean_mag if mean_mag > 0 else 0 for max_mag, mean_mag in zip(max_magnitudes, mean_magnitudes)]
            axes[1, 1].plot(times, flux_ratios, 'm-', linewidth=2, marker='o', markersize=4)
            axes[1, 1].set_xlabel('Time')
            axes[1, 1].set_ylabel('Max/Mean Flux Ratio')
            axes[1, 1].set_title('Flux Concentration Ratio')
            axes[1, 1].grid(True, alpha=0.3)
            
            self.custom_fig.suptitle('Mass Flux Evolution Analysis', fontsize=16)
            self.custom_fig.tight_layout()
            self.custom_canvas.draw()
            self.status_var.set("Mass flux evolution plotted")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot mass flux evolution:\n{str(e)}")
            
    def plot_radius_evolution(self):
        """Plot radius evolution in observables tab"""
        if not self.check_data_loaded():
            return
            
        try:
            self.observables_fig.clear()
            
            # Create the plot directly in the GUI figure
            ax = self.observables_fig.add_subplot(111)
            
            # Get parameters from GUI
            threshold = self.threshold_var.get()
            method = self.radius_method_var.get()
            line_width = self.line_width_var.get()
            marker_size = self.marker_size_var.get()
            color_scheme = self.obs_color_scheme_var.get()
            include_individual = self.include_individual_var.get()
            
            # Calculate radii for all populations
            radii_data = {}
            for i, label in enumerate(self.sim_plotter.labels):
                radii = []
                for step_idx in range(len(self.sim_plotter.saved_steps)):
                    phi_hat = self.sim_plotter.field_data["phi_hat"][step_idx]
                    radius = self.sim_plotter.utils.calculate_radius(phi_hat[i], threshold=threshold, method=method)
                    radii.append(radius)
                radii_data[label] = radii
            
            # Calculate total tumor radius
            total_radii = []
            for step_idx in range(len(self.sim_plotter.saved_steps)):
                phi_hat = self.sim_plotter.field_data["phi_hat"][step_idx]
                total_density = np.sum(phi_hat, axis=0)
                radius = self.sim_plotter.utils.calculate_radius(total_density, threshold=threshold, method=method)
                total_radii.append(radius)
            
            # Plot total radius
            ax.plot(self.sim_plotter.saved_times, total_radii, 'k-', linewidth=line_width, 
                   label='Total Tumor', marker='o', markersize=marker_size)
            
            # Plot individual population radii if requested
            if include_individual:
                colors = plt.cm.get_cmap(color_scheme)(np.linspace(0, 1, len(self.sim_plotter.labels)))
                for i, (label, radii) in enumerate(radii_data.items()):
                    ax.plot(self.sim_plotter.saved_times, radii, '--', color=colors[i], 
                           linewidth=line_width*0.8, label=f'{label}', marker='s', markersize=marker_size*0.7)
            
            ax.set_xlabel('Time')
            ax.set_ylabel('Radius')
            ax.set_title('Tumor Radius Evolution')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            self.observables_fig.tight_layout()
            self.observables_canvas.draw()
            self.status_var.set("Radius evolution plotted")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot radius evolution:\n{str(e)}")
            
    def plot_density_evolution(self):
        """Plot density evolution in observables tab"""
        if not self.check_data_loaded():
            return
            
        try:
            self.observables_fig.clear()
            
            # Create the plot directly in the GUI figure
            ax = self.observables_fig.add_subplot(111)
            
            # Get parameters from GUI
            line_width = self.line_width_var.get()
            marker_size = self.marker_size_var.get()
            color_scheme = self.obs_color_scheme_var.get()
            include_individual = self.include_individual_var.get()
            
            # Calculate densities for all populations
            density_data = {}
            for i, label in enumerate(self.sim_plotter.labels):
                densities = []
                for step_idx in range(len(self.sim_plotter.saved_steps)):
                    phi_hat = self.sim_plotter.field_data["phi_hat"][step_idx]
                    density = self.sim_plotter.utils.calculate_total_density(phi_hat[i], normalize_by_volume=False)
                    densities.append(density)
                density_data[label] = densities
            
            # Calculate total density
            total_densities = []
            for step_idx in range(len(self.sim_plotter.saved_steps)):
                phi_hat = self.sim_plotter.field_data["phi_hat"][step_idx]
                total_density = np.sum(phi_hat, axis=0)
                density = self.sim_plotter.utils.calculate_total_density(total_density, normalize_by_volume=False)
                total_densities.append(density)
            
            # Plot total density
            ax.plot(self.sim_plotter.saved_times, total_densities, 'k-', linewidth=line_width, 
                   label='Total', marker='o', markersize=marker_size)
            
            # Plot individual population densities if requested
            if include_individual:
                colors = plt.cm.get_cmap(color_scheme)(np.linspace(0, 1, len(self.sim_plotter.labels)))
                for i, (label, densities) in enumerate(density_data.items()):
                    ax.plot(self.sim_plotter.saved_times, densities, '--', color=colors[i], 
                           linewidth=line_width*0.8, label=f'{label}', marker='s', markersize=marker_size*0.7)
            
            ax.set_xlabel('Time')
            ax.set_ylabel('Total Density')
            ax.set_title('Population Density Evolution')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            self.observables_fig.tight_layout()
            self.observables_canvas.draw()
            self.status_var.set("Density evolution plotted")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot density evolution:\n{str(e)}")
            
    def plot_free_energy_observable(self):
        """Plot free energy in observables tab"""
        if not self.check_data_loaded():
            return
            
        try:
            self.observables_fig.clear()
            
            # Create the plot directly in the GUI figure
            ax = self.observables_fig.add_subplot(111)
            
            # Check if physics data is available
            if "physics_data" not in self.simulation_data or len(self.simulation_data["physics_data"]) == 0:
                ax.text(0.5, 0.5, 'No physics data available for free energy calculation', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Free Energy Evolution - No Data')
                self.observables_canvas.draw()
                self.status_var.set("No physics data available")
                return
            
            # Calculate free energy for each step
            free_energies = []
            for step_idx in range(len(self.simulation_data["physics_data"])):
                physics_data = self.simulation_data["physics_data"][step_idx]
                if "energy_derivative" in physics_data:
                    # Sum the absolute energy derivative as a proxy for free energy
                    energy_deriv = physics_data["energy_derivative"]
                    free_energy = np.sum(np.abs(energy_deriv))
                    free_energies.append(free_energy)
                else:
                    free_energies.append(0.0)
            
            # Plot free energy evolution
            ax.plot(self.sim_plotter.saved_times[:len(free_energies)], free_energies, 
                   'b-', linewidth=2, marker='o', markersize=4)
            
            ax.set_xlabel('Time')
            ax.set_ylabel('Free Energy (|Energy Derivative|)')
            ax.set_title('Free Energy Evolution')
            ax.grid(True, alpha=0.3)
            
            self.observables_fig.tight_layout()
            self.observables_canvas.draw()
            self.status_var.set("Free energy evolution plotted")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot free energy evolution:\n{str(e)}")
            
    def export_physics_data(self):
        """Export physics data"""
        if not self.check_data_loaded():
            return
            
        try:
            output_dir = filedialog.askdirectory(title="Select Output Directory")
            if output_dir:
                # Export physics data to CSV
                import pandas as pd
                
                # Collect all physics data
                physics_summary = []
                for step_idx, physics_data in enumerate(self.simulation_data["physics_data"]):
                    step_info = {
                        'step': step_idx,
                        'time': self.simulation_data["metadata"]["saved_times"][step_idx]
                    }
                    
                    # Add available physics fields
                    for field_name, field_data in physics_data.items():
                        if field_name == "pressure":
                            step_info['pressure_mean'] = np.mean(field_data)
                            step_info['pressure_std'] = np.std(field_data)
                            step_info['pressure_max'] = np.max(field_data)
                            step_info['pressure_min'] = np.min(field_data)
                        elif field_name == "velocity":
                            # Calculate velocity magnitude
                            vel_mag = np.sqrt(np.sum(field_data**2, axis=0))
                            step_info['velocity_mean'] = np.mean(vel_mag)
                            step_info['velocity_std'] = np.std(vel_mag)
                            step_info['velocity_max'] = np.max(vel_mag)
                        elif field_name == "energy_derivative":
                            step_info['energy_derivative_mean'] = np.mean(field_data)
                            step_info['energy_derivative_std'] = np.std(field_data)
                            step_info['energy_derivative_total'] = np.sum(np.abs(field_data))
                        elif field_name == "mass_flux":
                            # Calculate flux magnitude for each population
                            for pop_idx in range(field_data.shape[0]):
                                flux_mag = np.sqrt(np.sum(field_data[pop_idx]**2, axis=0))
                                step_info[f'mass_flux_pop{pop_idx}_mean'] = np.mean(flux_mag)
                                step_info[f'mass_flux_pop{pop_idx}_max'] = np.max(flux_mag)
                    
                    physics_summary.append(step_info)
                
                # Create DataFrame and save
                df = pd.DataFrame(physics_summary)
                output_file = os.path.join(output_dir, "physics_data_summary.csv")
                df.to_csv(output_file, index=False)
                
                self.status_var.set(f"Physics data exported to {output_file}")
                messagebox.showinfo("Success", f"Physics data exported to {output_file}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export physics data:\n{str(e)}")
            
    def export_all_plots(self):
        """Export all plots"""
        if not self.check_data_loaded():
            return
            
        try:
            output_dir = filedialog.askdirectory(title="Select Output Directory")
            if output_dir:
                # Generate and save all possible plots
                saved_plots = []
                
                # Overview plots
                try:
                    self.overview_fig.clear()
                    self.plot_tumor_radius()
                    self.overview_fig.savefig(os.path.join(output_dir, "tumor_radius_evolution.png"), dpi=300, bbox_inches='tight')
                    saved_plots.append("tumor_radius_evolution.png")
                except:
                    pass
                
                try:
                    self.overview_fig.clear()
                    self.plot_population_density()
                    self.overview_fig.savefig(os.path.join(output_dir, "population_density_evolution.png"), dpi=300, bbox_inches='tight')
                    saved_plots.append("population_density_evolution.png")
                except:
                    pass
                
                # Observables plots
                try:
                    self.observables_fig.clear()
                    self.plot_radius_evolution()
                    self.observables_fig.savefig(os.path.join(output_dir, "radius_evolution_observables.png"), dpi=300, bbox_inches='tight')
                    saved_plots.append("radius_evolution_observables.png")
                except:
                    pass
                
                try:
                    self.observables_fig.clear()
                    self.plot_density_evolution()
                    self.observables_fig.savefig(os.path.join(output_dir, "density_evolution_observables.png"), dpi=300, bbox_inches='tight')
                    saved_plots.append("density_evolution_observables.png")
                except:
                    pass
                
                # Advanced analysis plots
                try:
                    self.custom_fig.clear()
                    self.plot_free_energy()
                    self.custom_fig.savefig(os.path.join(output_dir, "free_energy_evolution.png"), dpi=300, bbox_inches='tight')
                    saved_plots.append("free_energy_evolution.png")
                except:
                    pass
                
                self.status_var.set(f"Exported {len(saved_plots)} plots to {output_dir}")
                messagebox.showinfo("Success", f"Exported {len(saved_plots)} plots to {output_dir}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export all plots:\n{str(e)}")
            
    def export_observables(self):
        """Export observables data"""
        if not self.check_data_loaded():
            return
            
        try:
            output_dir = filedialog.askdirectory(title="Select Output Directory")
            if output_dir:
                self.sim_plotter.export_observables_data(output_dir)
                self.status_var.set(f"Observables exported to {output_dir}")
                messagebox.showinfo("Success", f"Observables data exported to {output_dir}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export observables:\n{str(e)}")
            
    def save_current_plot(self):
        """Save the current plot"""
        try:
            # Get current tab
            current_tab = self.notebook.tab(self.notebook.select(), "text")
            
            # Determine which figure to save
            if current_tab == "Overview":
                fig = self.overview_fig
            elif current_tab == "Observables":
                fig = self.observables_fig
            elif current_tab == "Cell Fields":
                fig = self.cell_fig
            elif current_tab == "Physics Fields":
                fig = self.physics_fig
            elif current_tab == "Nutrient Fields":
                fig = self.nutrient_fig
            elif current_tab == "Custom Analysis":
                fig = self.custom_fig
            else:
                messagebox.showwarning("Warning", "No plot to save")
                return
                
            # Save dialog
            file_path = filedialog.asksaveasfilename(
                title="Save Plot",
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("SVG files", "*.svg")]
            )
            
            if file_path:
                fig.savefig(file_path, dpi=300, bbox_inches='tight')
                self.status_var.set(f"Plot saved to {file_path}")
                messagebox.showinfo("Success", f"Plot saved to {file_path}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save plot:\n{str(e)}")
            
    # Recent files management
    def load_recent_files(self):
        """Load recent files from config"""
        config_file = Path.home() / ".silicokit_gui_config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    self.recent_files = config.get('recent_files', [])
            except:
                self.recent_files = []
        else:
            self.recent_files = []
            
    def save_recent_files(self):
        """Save recent files to config"""
        config_file = Path.home() / ".silicokit_gui_config.json"
        try:
            config = {'recent_files': self.recent_files}
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except:
            pass
            
    def add_to_recent_files(self, file_path):
        """Add file to recent files list"""
        if file_path in self.recent_files:
            self.recent_files.remove(file_path)
        self.recent_files.insert(0, file_path)
        self.recent_files = self.recent_files[:10]  # Keep only 10 recent files
        self.save_recent_files()
        self.update_recent_files_display()
        
    def update_recent_files_display(self):
        """Update recent files combobox"""
        display_files = [Path(f).name for f in self.recent_files]
        self.recent_combo['values'] = display_files
        
    def on_recent_file_selected(self, event):
        """Handle recent file selection"""
        selection = self.recent_combo.current()
        if 0 <= selection < len(self.recent_files):
            file_path = self.recent_files[selection]
            self.load_file(file_path)
            
    def clear_recent_files(self):
        """Clear recent files list"""
        self.recent_files = []
        self.save_recent_files()
        self.update_recent_files_display()


def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = SimulationGUI(root)
    
    # Bind events for real-time updates
    app.step_var.trace('w', lambda *args: app.update_step_labels())
    app.physics_step_var.trace('w', lambda *args: app.update_step_labels())
    app.nutrient_step_var.trace('w', lambda *args: app.update_step_labels())
    app.z_slice_var.trace('w', lambda *args: app.update_step_labels())
    app.physics_z_slice_var.trace('w', lambda *args: app.update_step_labels())
    app.nutrient_z_slice_var.trace('w', lambda *args: app.update_step_labels())
    
    # Bind scale update events
    app.threshold_var.trace('w', lambda *args: app.update_threshold_label(app.threshold_var.get()))
    app.max_plots_var.trace('w', lambda *args: app.update_max_plots_label(app.max_plots_var.get()))
    app.zoom_var.trace('w', lambda *args: app.update_zoom_label(app.zoom_var.get()))
    app.physics_zoom_var.trace('w', lambda *args: app.update_physics_zoom_label(app.physics_zoom_var.get()))
    app.start_time_var.trace('w', lambda *args: app.update_start_time_label(app.start_time_var.get()))
    app.end_time_var.trace('w', lambda *args: app.update_end_time_label(app.end_time_var.get()))
    
    # Handle window closing
    def on_closing():
        app.save_recent_files()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Start the GUI
    root.mainloop()


if __name__ == "__main__":
    main()
