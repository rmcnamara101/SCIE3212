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
        self.root.title("Aurelia Lab • Simulation Studio")
        try:
            self.root.minsize(1100, 720)
        except Exception:
            pass
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
        
        # Brand header and status tips
        self._install_status_tips()
        # Set up matplotlib style
        plt.style.use('default')
        self.setup_style()
        self.setup_keybindings()
    def setup_style(self):
        """Modern, sleek ttk style without altering business logic."""
        try:
            import tkinter as tk
            from tkinter import ttk
        except Exception:
            return
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        # Font stack (falls back gracefully on Windows/Linux)
        font_stack_base = ("Inter", 10)
        font_stack_head = ("Inter", 18, "bold")
        font_stack_sub  = ("Inter", 10)
        try:
            # If SF Pro or Segoe UI available, lean on them
            font_stack_base = ("SF Pro Text", 10)
            font_stack_head = ("SF Pro Display", 18, "semibold")
            font_stack_sub  = ("SF Pro Text", 10)
        except Exception:
            pass
        self.root.option_add("*Font", " ".join(map(str, font_stack_base)))

        # Palette — deep tech
        bg = "#0b0f1a"          # canvas
        card = "#0f1629"        # surfaces
        card_alt = "#121a31"    # alt surface
        accent = "#7c3aed"      # vibrant purple
        accent_hi = "#a78bfa"   # light
        text = "#ebefff"
        sub = "#b6c0e9"
        border = "#1b2540"
        self.root.configure(bg=bg)

        # Base styles
        style.configure(".", background=bg, foreground=text)
        style.configure("Card.TFrame", background=card, borderwidth=0, relief="flat")
        style.configure("CardAlt.TFrame", background=card_alt, borderwidth=0, relief="flat")
        style.configure("Headline.TLabel", background=bg, foreground=text, font=font_stack_head)
        style.configure("Subhead.TLabel", background=bg, foreground=sub, font=font_stack_sub)
        style.configure("Status.TLabel", background=card_alt, foreground=sub, font=("Inter", 9))

        # Buttons
        style.configure("Accent.TButton", background=accent, foreground="white", borderwidth=0, padding=(14, 8))
        style.map("Accent.TButton",
                  background=[("active", "#5b21b6"), ("pressed", "#6d28d9")])
        style.configure("Ghost.TButton", background=card, foreground=text, bordercolor=border, borderwidth=1, padding=(12, 8))
        style.map("Ghost.TButton",
                  background=[("active", card_alt), ("pressed", card_alt)],
                  relief=[("pressed", "sunken")])

        # Notebook -> pill tabs
        style.configure("Modern.TNotebook", background=bg, borderwidth=0, tabmargins=[6, 6, 6, 0])
        style.configure("Modern.TNotebook.Tab", background=card, foreground=sub, padding=[14, 8],
                        borderwidth=0, focusthickness=0, lightcolor=card, darkcolor=card)
        style.map("Modern.TNotebook.Tab",
                  background=[("selected", card_alt), ("active", card_alt)],
                  foreground=[("selected", text)],
                  expand=[("selected", [2, 2, 2, 0])])

        # Treeview/inputs (if present)
        style.configure("Treeview", background=card, fieldbackground=card, foreground=text, bordercolor=border)
        style.map("Treeview", background=[("selected", card_alt)])

        # Remember palette
        self._ui_colors = {"bg": bg, "card": card, "card_alt": card_alt, "accent": accent, "text": text, "sub": sub, "border": border}

    def setup_keybindings(self):
        """Quality-of-life shortcuts; they call existing commands only."""
        self.root.bind("<Control-o>", lambda e: self.load_simulation_file())
        self.root.bind("<Control-r>", lambda e: self.reload_current_file())
        # Navigate tabs
        self.root.bind("<Control-Tab>", lambda e: self._cycle_tab(1))
        self.root.bind("<Control-Shift-Tab>", lambda e: self._cycle_tab(-1))

    def _cycle_tab(self, step):
        if not hasattr(self, "notebook"):
            return
        tabs = self.notebook.tabs()
        if not tabs:
            return
        current = self.notebook.index("current")
        self.notebook.select((current + step) % len(tabs))

    def create_brand_header(self, parent):
        """Gradient brand header with crisp typography."""
        import tkinter as tk
        from tkinter import ttk
        colors = getattr(self, "_ui_colors", {})
        card = colors.get("card", "#0f1629")
        accent = colors.get("accent", "#7c3aed")
        text = colors.get("text", "#ebefff")
        sub = colors.get("sub", "#b6c0e9")

        # Gradient canvas
        container = ttk.Frame(parent, style="Card.TFrame")
        container.pack(fill=tk.X, pady=(0, 12))
        canvas = tk.Canvas(container, height=86, bd=0, highlightthickness=0, relief="flat", bg=card)
        canvas.pack(fill=tk.BOTH, expand=True)

        # Draw a horizontal gradient strip (simple interpolation)
        w = 1200
        try:
            w = container.winfo_width() or 1200
        except Exception:
            pass
        for i in range(0, w, 2):
            t = i / max(w, 1)
            # blend card -> accent
            def mix(c1, c2, t):
                c1 = tuple(int(c1[j:j+2], 16) for j in (1,3,5))
                c2 = tuple(int(c2[j:j+2], 16) for j in (1,3,5))
                v = tuple(int(c1[k]*(1-t) + c2[k]*t) for k in range(3))
                return f"#{v[0]:02x}{v[1]:02x}{v[2]:02x}"
            canvas.create_line(i, 0, i, 86, fill=mix(card, accent, t*0.25), width=2)

        # Foreground content
        inner = ttk.Frame(container, style="Card.TFrame")
        inner.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.96, relheight=0.85)

        left = ttk.Frame(inner, style="Card.TFrame")
        left.pack(side=tk.LEFT, fill=tk.X, expand=True)
        title = ttk.Label(left, text="Aurelia Lab • Simulation Studio", style="Headline.TLabel")
        title.pack(anchor="w")
        subtitle = ttk.Label(left, text="High‑signal analysis for complex tumor simulations.", style="Subhead.TLabel")
        subtitle.pack(anchor="w", pady=(4,0))

        right = ttk.Frame(inner, style="Card.TFrame")
        right.pack(side=tk.RIGHT)
        ttk.Button(right, text="Open (Ctrl+O)", style="Ghost.TButton", command=self.load_simulation_file).pack(side=tk.LEFT, padx=6)
        ttk.Button(right, text="New Session", style="Accent.TButton", command=self.reload_current_file).pack(side=tk.LEFT, padx=6)

    def _install_status_tips(self):
        # Rotating micro‑tips to keep energy up (no threads; rotate on tab change).
        self._tips = [
            "Protip: Ctrl+Tab to flip tabs • Ctrl+O to open",
            "Zoom into anomalies; breakthroughs live at the edges.",
            "Compare growth curves across runs to spot regime shifts.",
            "Small deltas in parameters can flip morphology. Sweep smart.",
            "Export observables and annotate decisions — future you will thank you.",
        ]
        self._tip_idx = 0
        if hasattr(self, "notebook"):
            self.notebook.bind("<<NotebookTabChanged>>", lambda e: self._next_tip())

    def _next_tip(self):
        if not hasattr(self, "status_var"):
            return
        self._tip_idx = (self._tip_idx + 1) % len(getattr(self, "_tips", [""]))
        tip = self._tips[self._tip_idx]
        try:
            self.status_var.set(tip)
        except Exception:
            pass

        
    def create_widgets(self):
        """Create the main GUI layout"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        # Brand header
        self.create_brand_header(main_frame)
        
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
        self.notebook = ttk.Notebook(parent, style='Modern.TNotebook')
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
        
        ttk.Label(params_frame, text="Threshold:").pack(anchor=tk.W)
        self.threshold_var = tk.DoubleVar(value=0.1)
        ttk.Scale(params_frame, from_=0.01, to=0.5, variable=self.threshold_var, orient=tk.HORIZONTAL).pack(fill=tk.X)
        
        ttk.Label(params_frame, text="Max Plots:").pack(anchor=tk.W, pady=(10, 0))
        self.max_plots_var = tk.IntVar(value=6)
        ttk.Scale(params_frame, from_=3, to=12, variable=self.max_plots_var, orient=tk.HORIZONTAL).pack(fill=tk.X)
        
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
        
        # Create plot area
        self.observables_fig = Figure(figsize=(12, 8), dpi=100)
        self.observables_canvas = FigureCanvasTkAgg(self.observables_fig, observables_frame)
        self.observables_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add toolbar
        self.observables_toolbar = NavigationToolbar2Tk(self.observables_canvas, observables_frame)
        self.observables_toolbar.update()
        
    def create_cell_fields_tab(self):
        """Create cell fields visualization tab"""
        cell_frame = ttk.Frame(self.notebook)
        self.notebook.add(cell_frame, text="Cell Fields")
        
        # Left panel for controls
        cell_control_frame = ttk.Frame(cell_frame)
        cell_control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 5), pady=10)
        
        # Population selection
        ttk.Label(cell_control_frame, text="Population", font=("Arial", 10, "bold")).pack(pady=(0, 5))
        self.population_var = tk.StringVar()
        self.population_combo = ttk.Combobox(cell_control_frame, textvariable=self.population_var, state="readonly")
        self.population_combo.pack(fill=tk.X, pady=(0, 10))
        
        # Step selection
        ttk.Label(cell_control_frame, text="Time Step", font=("Arial", 10, "bold")).pack(pady=(0, 5))
        self.step_var = tk.IntVar()
        self.step_scale = ttk.Scale(cell_control_frame, from_=0, to=10, variable=self.step_var, orient=tk.HORIZONTAL)
        self.step_scale.pack(fill=tk.X, pady=(0, 10))
        self.step_label = ttk.Label(cell_control_frame, text="Step: 0")
        self.step_label.pack()
        
        # Z-slice selection
        ttk.Label(cell_control_frame, text="Z-Slice", font=("Arial", 10, "bold")).pack(pady=(10, 5))
        self.z_slice_var = tk.IntVar()
        self.z_slice_scale = ttk.Scale(cell_control_frame, from_=0, to=10, variable=self.z_slice_var, orient=tk.HORIZONTAL)
        self.z_slice_scale.pack(fill=tk.X, pady=(0, 10))
        self.z_slice_label = ttk.Label(cell_control_frame, text="Z: 0")
        self.z_slice_label.pack()
        
        # Plot options
        options_frame = ttk.LabelFrame(cell_control_frame, text="Options", padding=5)
        options_frame.pack(fill=tk.X, pady=(20, 0))
        
        self.add_contours_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Add Contours", variable=self.add_contours_var).pack(anchor=tk.W)
        
        self.zoom_var = tk.DoubleVar(value=1.0)
        ttk.Label(options_frame, text="Zoom Factor:").pack(anchor=tk.W, pady=(10, 0))
        ttk.Scale(options_frame, from_=1.0, to=5.0, variable=self.zoom_var, orient=tk.HORIZONTAL).pack(fill=tk.X)
        
        # Plot button
        ttk.Button(cell_control_frame, text="Plot Cell Field", command=self.plot_cell_field).pack(fill=tk.X, pady=(20, 0))
        
        # Right panel for plot
        self.cell_plot_frame = ttk.Frame(cell_frame)
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
        
        # Left panel for controls
        physics_control_frame = ttk.Frame(physics_frame)
        physics_control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 5), pady=10)
        
        # Field type selection
        ttk.Label(physics_control_frame, text="Field Type", font=("Arial", 10, "bold")).pack(pady=(0, 5))
        self.field_type_var = tk.StringVar()
        field_combo = ttk.Combobox(physics_control_frame, textvariable=self.field_type_var, 
                                  values=["Pressure", "Velocity", "Energy Derivative", "Mass Flux", "Source Terms"],
                                  state="readonly")
        field_combo.pack(fill=tk.X, pady=(0, 10))
        field_combo.set("Pressure")
        
        # Step selection
        ttk.Label(physics_control_frame, text="Time Step", font=("Arial", 10, "bold")).pack(pady=(0, 5))
        self.physics_step_var = tk.IntVar()
        self.physics_step_scale = ttk.Scale(physics_control_frame, from_=0, to=10, 
                                          variable=self.physics_step_var, orient=tk.HORIZONTAL)
        self.physics_step_scale.pack(fill=tk.X, pady=(0, 10))
        self.physics_step_label = ttk.Label(physics_control_frame, text="Step: 0")
        self.physics_step_label.pack()
        
        # Z-slice selection
        ttk.Label(physics_control_frame, text="Z-Slice", font=("Arial", 10, "bold")).pack(pady=(10, 5))
        self.physics_z_slice_var = tk.IntVar()
        self.physics_z_slice_scale = ttk.Scale(physics_control_frame, from_=0, to=10, 
                                             variable=self.physics_z_slice_var, orient=tk.HORIZONTAL)
        self.physics_z_slice_scale.pack(fill=tk.X, pady=(0, 10))
        self.physics_z_slice_label = ttk.Label(physics_control_frame, text="Z: 0")
        self.physics_z_slice_label.pack()
        
        # Plot button
        ttk.Button(physics_control_frame, text="Plot Physics Field", command=self.plot_physics_field).pack(fill=tk.X, pady=(20, 0))
        
        # Right panel for plot
        self.physics_plot_frame = ttk.Frame(physics_frame)
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
        
        # Left panel for controls
        nutrient_control_frame = ttk.Frame(nutrient_frame)
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
        
        # Plot button
        ttk.Button(nutrient_control_frame, text="Plot Nutrient Field", command=self.plot_nutrient_field).pack(fill=tk.X, pady=(20, 0))
        
        # Right panel for plot
        self.nutrient_plot_frame = ttk.Frame(nutrient_frame)
        self.nutrient_plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 10), pady=10)
        
        self.nutrient_fig = Figure(figsize=(10, 8), dpi=100)
        self.nutrient_canvas = FigureCanvasTkAgg(self.nutrient_fig, self.nutrient_plot_frame)
        self.nutrient_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.nutrient_toolbar = NavigationToolbar2Tk(self.nutrient_canvas, self.nutrient_plot_frame)
        self.nutrient_toolbar.update()
        
    def create_custom_tab(self):
        """Create custom analysis tab"""
        custom_frame = ttk.Frame(self.notebook)
        self.notebook.add(custom_frame, text="Custom Analysis")
        
        # Left panel for custom controls
        custom_control_frame = ttk.Frame(custom_frame)
        custom_control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 5), pady=10)
        
        ttk.Label(custom_control_frame, text="Custom Analysis", font=("Arial", 12, "bold")).pack(pady=(0, 10))
        
        # Export options
        export_frame = ttk.LabelFrame(custom_control_frame, text="Export Options", padding=5)
        export_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(export_frame, text="Export All Observables", command=self.export_observables).pack(fill=tk.X, pady=2)
        ttk.Button(export_frame, text="Save Current Plot", command=self.save_current_plot).pack(fill=tk.X, pady=2)
        
        # Analysis options
        analysis_frame = ttk.LabelFrame(custom_control_frame, text="Analysis Options", padding=5)
        analysis_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(analysis_frame, text="Free Energy Evolution", command=self.plot_free_energy).pack(fill=tk.X, pady=2)
        ttk.Button(analysis_frame, text="Population Evolution", command=self.plot_population_evolution).pack(fill=tk.X, pady=2)
        
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
        status_frame = ttk.Frame(parent)  # glassy status host
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
            
        # Update population combo
        if hasattr(self.sim_plotter, 'labels'):
            self.population_combo['values'] = self.sim_plotter.labels
            if self.sim_plotter.labels:
                self.population_combo.set(self.sim_plotter.labels[0])
        
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
        
        # Update labels
        self.update_step_labels()
        
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
            
            # Calculate radii for all populations
            threshold = self.threshold_var.get()
            radii_data = {}
            for i, label in enumerate(self.sim_plotter.labels):
                radii = []
                for step_idx in range(len(self.sim_plotter.saved_steps)):
                    phi_hat = self.sim_plotter.field_data["phi_hat"][step_idx]
                    radius = self.sim_plotter.utils.calculate_radius(phi_hat[i], threshold=threshold, method='contour')
                    radii.append(radius)
                radii_data[label] = radii
            
            # Calculate total tumor radius
            total_radii = []
            for step_idx in range(len(self.sim_plotter.saved_steps)):
                phi_hat = self.sim_plotter.field_data["phi_hat"][step_idx]
                total_density = np.sum(phi_hat, axis=0)
                radius = self.sim_plotter.utils.calculate_radius(total_density, threshold=threshold, method='contour')
                total_radii.append(radius)
            
            # Plot total radius
            ax.plot(self.sim_plotter.saved_times, total_radii, 'k-', linewidth=3, label='Total Tumor', marker='o', markersize=6)
            
            # Plot individual population radii
            colors = plt.cm.Set1(np.linspace(0, 1, len(self.sim_plotter.labels)))
            for i, (label, radii) in enumerate(radii_data.items()):
                ax.plot(self.sim_plotter.saved_times, radii, '--', color=colors[i], linewidth=2, 
                       label=f'{label}', marker='s', markersize=4)
            
            ax.set_xlabel('Time')
            ax.set_ylabel('Radius')
            ax.set_title('Tumor Radius Evolution')
            ax.legend()
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
            
            # Set default center coordinates
            center_x = nx // 2
            center_y = ny // 2
            
            # Calculate zoom window size
            window_size = min(nx, ny) // zoom_factor
            half_window = window_size // 2
            
            # Calculate window boundaries
            x_start = int(max(0, center_x - half_window))
            x_end = int(min(nx, center_x + half_window))
            y_start = int(max(0, center_y - half_window))
            y_end = int(min(ny, center_y + half_window))
            
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
                          origin='lower', cmap='viridis', aspect='equal')
            
            # Add contours if requested
            if add_contours:
                contour_levels = [0.1, 0.3, 0.5]
                contour_colors = ['white', 'yellow', 'red']
                for i, level in enumerate(contour_levels):
                    color = contour_colors[i] if i < len(contour_colors) else 'black'
                    ax.contour(density_slice, levels=[level], colors=color, 
                              linewidths=2, alpha=0.8,
                              extent=[x_region[0], x_region[-1], y_region[0], y_region[-1]], 
                              origin='lower')
            
            # Add center point marker if zoomed
            if zoom_factor > 1.0:
                center_x_coord = center_x * dx
                center_y_coord = center_y * dx
                ax.plot(center_x_coord, center_y_coord, 'k+', markersize=10, markeredgewidth=2, 
                       label=f'Center: ({center_x_coord:.2f}, {center_y_coord:.2f})')
                ax.legend()
                ax.set_title(f'Cell Density - {population_name} (z={z_slice}) - Zoomed View')
            else:
                ax.set_title(f'Cell Density - {population_name} (z={z_slice})')
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            
            # Add colorbar
            self.cell_fig.colorbar(im, ax=ax, label='Cell Density')
            
            self.cell_fig.tight_layout()
            self.cell_canvas.draw()
            self.status_var.set(f"Cell field plotted: {population_name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot cell field:\n{str(e)}")
            
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
                                  origin='lower', cmap='viridis', aspect='equal')
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
                    skip = 10
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
                                    cmap='viridis', scale=50, width=0.005)
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
                                  origin='lower', cmap='RdBu_r', aspect='equal', vmin=vmin, vmax=vmax)
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
                    skip = 10
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
                                    cmap='viridis', scale=20, width=0.005)
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
            
            # Get first population
            if self.sim_plotter.labels:
                population = self.sim_plotter.labels[0]
                self.cell_plotter.plot_population_evolution_by_label(
                    self.simulation_data, label=population,
                    output_dir=None, save_plot=False, show_plot=False
                )
            
            self.custom_canvas.draw()
            self.status_var.set("Population evolution plotted")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot population evolution:\n{str(e)}")
            
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
    
    # Handle window closing
    def on_closing():
        app.save_recent_files()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Start the GUI
    root.mainloop()


if __name__ == "__main__":
    main()
