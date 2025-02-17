import pyvista as pv
import numpy as np
import streamlit as st
from src.simulation import run_simulation

st.title("Interactive 3D Tumor Growth Simulation")

# User inputs
Nx = st.slider("Grid size (Nx, Ny, Nz)", 10, 100, 50)
Nt = st.slider("Number of time steps", 100, 1000, 500)
L = st.number_input("Domain Size", value=1.0, step=0.1)
D1 = st.number_input("Stem Cell Diffusion Coefficient", value=0.001, step=0.0001)
D2 = st.number_input("Progenitor Cell Diffusion Coefficient", value=0.0005, step=0.0001)
D3 = st.number_input("Differentiated Cell Diffusion Coefficient", value=0.0001, step=0.0001)
D_N = st.number_input("Nutrient Diffusion Coefficient", value=0.005, step=0.0001)

lambda1 = st.slider("Stem Cell Proliferation Rate", 0.01, 0.2, 0.1)
gamma1 = st.slider("Stem to Progenitor Differentiation Rate", 0.01, 0.1, 0.05)
gamma2 = st.slider("Progenitor to Differentiated Differentiation Rate", 0.01, 0.1, 0.02)
mu1 = st.slider("Stem Cell Death Rate", 0.005, 0.05, 0.01)
mu2 = st.slider("Progenitor Cell Death Rate", 0.005, 0.05, 0.02)
mu3 = st.slider("Differentiated Cell Death Rate", 0.005, 0.05, 0.03)
alpha1 = st.slider("Nutrient Uptake by Stem Cells", 0.005, 0.05, 0.02)
alpha2 = st.slider("Nutrient Uptake by Progenitor Cells", 0.005, 0.05, 0.01)
K_N = st.number_input("Nutrient Saturation Constant", value=0.1, step=0.01)

dx = L / Nx
dt = 5.0 / Nt

if st.button("Run Simulation"):
    # Run simulation and get precomputed history.
    params = (L, D1, D2, D3, D_N, lambda1, gamma1, gamma2, mu1, mu2, mu3, alpha1, alpha2, K_N)
    x, history = run_simulation(params, Nt, Nx, dx, dt)

    print("Size of history['C1'][0]:", history["C1"][0].size)
    print("History:", history)

    # Create a StructuredGrid assuming a cubic domain.
    grid = pv.StructuredGrid()
    x = np.linspace(0, L, Nx)
    y = np.linspace(0, L, Nx)
    z = np.linspace(0, L, Nx)
    grid.points = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
    grid.dimensions = (Nx, Nx, Nx)

    # Set initial scalar values (time step 0)
    if history["C1"][0].size == 62500:
        # Reshape to match the grid size
        initial_scalars = history["C1"][0].reshape((50, 50, 25)).flatten(order="F")  # Adjust the dimensions as needed
    else:
        initial_scalars = history["C1"][0].flatten(order="F")
    grid.point_data["Stem Cells"] = initial_scalars

    # Create the PyVista plotter.
    plotter = pv.Plotter()
    actor = plotter.add_mesh(grid, scalars="Stem Cells", cmap="coolwarm", opacity=0.7)

    # Dictionaries to hold mutable state variables.
    playing = {"state": False}  # Whether animation is running.
    current_time = {"t": 0}     # Current time step.

    # Callback function for the slider widget.
    def slider_callback(value):
        t = int(value)
        current_time["t"] = t
        new_scalars = history["C1"][t].reshape((Nx, Nx, Nx)).flatten(order="F")
        plotter.update_scalars(new_scalars, mesh=grid, render=True)

    # Toggle play/pause when space is pressed.
    def toggle_play():
        playing["state"] = not playing["state"]

    # Register the key event (space bar) for play/pause.
    plotter.add_key_event("space", toggle_play)

    # Add a slider widget to allow manual selection of time step.
    plotter.add_slider_widget(
        slider_callback, 
        rng=[0, Nt - 1],
        value=0,
        title="Time Step",
        event_type="always",   # Callback is called continuously as the slider moves.
        style="modern"
    )

    # Timer callback to update the simulation if in play mode.
    def update():
        if playing["state"]:
            current_time["t"] += 1
            if current_time["t"] >= Nt:
                current_time["t"] = 0  # Loop back to start.
            slider_callback(current_time["t"])

    # Add a callback to run every 100 milliseconds.
    plotter.add_callback(update, interval=100)

    # Open the interactive PyVista window.
    plotter.show()
