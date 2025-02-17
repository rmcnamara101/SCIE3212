import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyvista as pv
import numpy as np
from src.simulation import run_simulation

def create_polyline(points):
    """
    Create a PolyData object representing a polyline from an array of points.
    Points should be an (N,3) array.
    """
    n = points.shape[0]
    # Create connectivity (each segment is defined by 2 points)
    # Format: [2, p0, p1, 2, p1, p2, ..., 2, p_{n-2}, p_{n-1}]
    cells = np.empty((n - 1, 3), dtype=np.int_)
    cells[:, 0] = 2  # number of points in each segment
    cells[:, 1] = np.arange(0, n - 1)
    cells[:, 2] = np.arange(1, n)
    cells = cells.flatten()
    
    polyline = pv.PolyData(points)
    polyline.lines = cells
    return polyline

def main():
    # --- Simulation Parameters ---
    Nx = 50         # Number of spatial points (for a 1D simulation)
    Nt = 500        # Number of time steps
    L = 1.0         # Domain length
    
    dx = L / (Nx - 1)
    dt = 5.0 / Nt

    # Simulation parameters (adjust as needed)
    D1 = 0.001; D2 = 0.0005; D3 = 0.0001; D_N = 0.005
    lambda1 = 0.1; gamma1 = 0.05; gamma2 = 0.02
    mu1 = 0.01; mu2 = 0.02; mu3 = 0.03
    alpha1 = 0.02; alpha2 = 0.01; K_N = 0.1

    print("Running simulation with:")
    print(f"  Spatial points: {Nx}")
    print(f"  Time steps: {Nt}")
    
    params = (L, D1, D2, D3, D_N, lambda1, gamma1, gamma2, mu1, mu2, mu3, alpha1, alpha2, K_N)
    x, history = run_simulation(params, Nt, Nx, dx, dt)
    # For a 1D simulation we expect each history entry (e.g. history["C1"]) to have shape (Nt, Nx)
    
    # --- Prepare the 2D Plot Data ---
    # Create the x-axis values for our 1D domain.
    x_vals = np.linspace(0, L, Nx)
    
    # Create a baseline (the x-axis) as a straight line at y = 0.
    baseline_points = np.column_stack((x_vals, np.zeros_like(x_vals), np.zeros_like(x_vals)))
    baseline = pv.Line(pointa=(0, 0, 0), pointb=(L, 0, 0), resolution=Nx-1)
    
    # Choose which function to display. For example, we can display "C1".
    # (If you prefer to combine densities, try: new_f = history["C1"][t] + history["C2"][t] + history["C3"][t])
    initial_f = np.array(history["C1"][0])
    
    # Optionally, apply a scaling factor so the curve is visibly above the baseline.
    scale = 10.0  # Adjust this factor as needed
    curve_points = np.column_stack((x_vals, initial_f * scale, np.zeros_like(x_vals)))
    curve = create_polyline(curve_points)
    
    # --- Set Up the PyVista Plotter ---
    plotter = pv.Plotter()
    
    # Add the baseline (black line)
    plotter.add_mesh(baseline, color="black", line_width=2)
    # Add the function curve (red line)
    plotter.add_mesh(curve, color="red", line_width=3)
    
    # Animation control variables.
    playing = {"state": False}
    current_time = {"t": 0}
    
    def slider_callback(value):
        t = int(value)
        current_time["t"] = t
        # Update the function curve.
        new_f = np.array(history["C1"][t])  # Change key or combine arrays as desired.
        new_points = np.column_stack((x_vals, new_f * scale, np.zeros_like(x_vals)))
        curve.points = new_points
        plotter.render()
    
    def toggle_play():
        playing["state"] = not playing["state"]
    
    plotter.add_key_event("space", toggle_play)
    
    def update(plotter):
        if playing["state"]:
            current_time["t"] += 1
            if current_time["t"] >= Nt:
                current_time["t"] = 0  # Loop back to the start.
            slider_callback(current_time["t"])
    
    # Add the slider widget for manual time-step control.
    plotter.add_slider_widget(
        slider_callback,
        rng=[0, Nt - 1],
        value=0,
        title="Time Step",
        style="modern"
    )

    # Add a periodic callback (updates every render).
    plotter.add_on_render_callback(update)
    
    print("Launching interactive visualization. Use the slider or press space to play/pause.")
    plotter.show()

if __name__ == "__main__":
    main()
