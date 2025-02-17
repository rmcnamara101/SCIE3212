from src.simulation import run_simulation
from src.visualization import animate_simulation
from src.utils import experimental_params

# Run the simulation
params = experimental_params
Nt = 1000
Nx = 1000
# appears dx must be at least one order of magnitude larger than dt for stability
dx = 0.1
dt = 0.05

# Run the simulation
x,history = run_simulation(params, Nt, Nx, dx, dt)

# Animate the simulation
animate_simulation(x, history, Nt) 