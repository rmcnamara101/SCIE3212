Tumor Growth Model
This repository contains a 3D tumor growth simulation framework implemented in Python. It solves a system of partial differential equations (PDEs) modeling various cell populations (stem, progenitor, differentiated, and necrotic cells) and a nutrient field. The simulation evolves these fields over time according to biological mechanisms such as cell proliferation, death, differentiation, and nutrient diffusion/consumption.

Table of Contents

-Overview
-Features
-Project Structure
-Installation & Requirements
-Quick Start
-Usage
-Simulation Workflow
-Parameter Reference
-Contributing
-License


<h2>Overview</h2>


Tumor progression involves many interacting biological processes, including:

<h3>Cell Production</h3>

Stem cells, progenitor cells, and differentiated cells expand or die out according to nutrient availability and biologically motivated source terms.

<h3>Cell Dynamics</h3>

The cells move and redistribute in space driven by solid velocity (pressure gradients, adhesion energy) and mass flux (gradients of adhesion energy).

<h3>Nutrient Diffusion</h3>

Nutrient diffuses throughout the domain and is consumed by the growing tumor.

This code numerically solves these coupled PDEs using finite differences, leveraging Numba-accelerated functions for better performance.

<h2>Features</h2>

Multiple Cell Populations: Stem cells, progenitor cells, differentiated cells, necrotic cells.
Nutrient Field: A scalar field representing nutrient concentration, which can diffuse and be consumed by the tumor.
Volume Fraction Enforcement: Ensures the sum of cell populations does not exceed a specified maximum volume fraction.
Configurable Parameters: Growth rates, death rates, diffusion coefficients, etc.
Runge-Kutta Integration: Time-stepping using a 4th-order Runge-Kutta scheme with clipping to maintain numerical stability.
History Tracking: Stores all fields at specified intervals for post-processing.


<h2>Project Structure</h2>

SCIE3121/
├─ src/
│  ├─ models/
│  │  ├─ cell_production.py      <- Source terms for cell population PDEs
│  │  ├─ cell_dynamics.py        <- Cell advection and mass flux
│  │  ├─ diffusion_dynamics.py   <- Nutrient diffusion and consumption
│  │  ├─ tumor_growth.py         <- Main TumorGrowthModel class orchestrating the simulation
│  │
│  ├─ visualization/
│  │  ├─ plot_tumor.py           <- Plotting routines (e.g., volume fraction slices, surfaces)
│  │  └─ animate_tumor.py        <- Animation routines for time evolution
│  │
│  ├─ utils/
│  │  ├─ utils.py                <- Utility functions (numerical methods, etc.)
│  │  └─ ...                     <- Other helpers
│  ├─ initial_conditions/
│  │  ├─ __init__.py
│  │  └─ initial_conditions.py   <- Classes to define the tumor initial condition
│  └─ ...
├─ main.py                       <- Entry point for running and testing simulations
├─ README.md                     <- Project documentation (this file)
├─ requirements.txt              <- Python dependencies (if provided)
└─ data/                         <- Folder to store output files, e.g. NPZ simulation results


<h3>Key Files:</h3>

tumor_growth.py

Defines the TumorGrowthModel class. Creates and updates the major fields (cell types, nutrient), applies time steps, and tracks simulation history.

cell_production.py

Implements the source terms (production, death, differentiation) for each cell population.

cell_dynamics.py
Handles the advection and mass flux of cells. It computes velocity fields from pressure and adhesion energy gradients, then updates each cell population accordingly.

diffusion_dynamics.py
Handles the diffusion and consumption of the nutrient field. Uses a finite difference Laplacian plus a nutrient consumption term proportional to total cell density.

main.py
Example usage and entry point. Shows how to:

Initialize the tumor model.
Run the simulation.
Save the history.
Optionally visualize or animate results.
Installation & Requirements
Python 3.8+ recommended.

Clone or download this repository.

Install Python dependencies:

bash
Copy
Edit
pip install -r requirements.txt
or manually install the required libraries:

numpy
numba
scipy
matplotlib (for visualization, if desired)
tqdm
Ensure you have a C/C++ compiler available (if using Numba on certain platforms).

Quick Start
Clone this repo:
bash
Copy
Edit
git clone https://github.com/your_user/tumor_growth_project.git
cd tumor_growth_project
Install dependencies:
bash
Copy
Edit
pip install -r requirements.txt
Run the main simulation script:
bash
Copy
Edit
python main.py
Output: The simulation data are saved (by default) to data/<simulation_name>_sim_data.npz. Intermediate fields are also stored in a Python dictionary for post-processing or visualization.
Usage
1. Import the Model
You can write a custom script to import and run the model directly:

python
Copy
Edit
import numpy as np
from src.models.tumor_growth import TumorGrowthModel
from src.models.initial_conditions import SphericalTumor
from src.utils.utils import experimental_params

# Create your initial conditions
grid_shape = (50, 50, 50)
initial_conditions = SphericalTumor(grid_shape, radius=5, nutrient_value=0.001)

# Instantiate the model
model = TumorGrowthModel(
    grid_shape=grid_shape,
    dx=0.1,
    dt=0.001,
    params=experimental_params,
    initial_conditions=initial_conditions,
    save_steps=10
)

# Run simulation
model.run_simulation(steps=100)

# Retrieve the simulation history
history = model.get_history()
print(f"Number of saved timesteps: {len(history['step'])}")
2. Using main.py
A simpler approach is to just run:

bash
Copy
Edit
python main.py
which internally creates a TumorGrowthModel, runs it for a default number of steps, and saves or visualizes the results.

Simulation Workflow
Initialize Fields:

A user-specified initial condition sets up each cell population fraction and nutrient field.
Example: a spherical tumor region in the domain, with uniform nutrient outside.
Production & Death:

The PDE source terms (growth, death, differentiation) are computed via cell_production.py.
Cell Dynamics:

Velocities and mass fluxes are computed to account for movement of each population from local gradients of pressure and adhesion energy.
Implemented in cell_dynamics.py.
Nutrient Diffusion:

A diffusion equation plus a consumption term from total cell density is solved for the nutrient field (in diffusion_dynamics.py).
Time Integration:

The model uses a 4th-order Runge-Kutta scheme with stability checks (clipping large derivatives).
Volume Fraction Enforcement:

If the total cell fraction surpasses a maximum (e.g. 1), it is rescaled to ensure no overlap beyond physical constraints.
History & Outputs:

At user-defined intervals (save_steps), the volume fraction fields and step count are stored in the model history.
These can be saved as NPZ files for further analysis or visualization.
Parameter Reference
Some typical parameters (accessible via model.params) include:

lambda_S, lambda_P: Growth/proliferation rates for stem & progenitor cells.
mu_S, mu_P, mu_D: Death rates for stem, progenitor, and differentiated cells.
alpha_D: Differentiation rate into necrotic cells.
D_n: Nutrient diffusion coefficient.
gamma_N: Necrotic cell dissolution rate.
p_0, p_1: Probabilities that govern branching between different cell fates.
dx, dt: Spatial and temporal discretization.
phi_S: Maximum total volume fraction limit.
M, gamma, epsilon: Parameters controlling adhesion energy and mass flux.
You can edit or provide your own parameter dictionary to fine-tune the simulation.

Contributing
Contributions in the form of bug reports, feature requests, or pull requests are welcome. Please:

Fork the repository
Create a new branch for your changes
Open a pull request with a clear description
License
This project does not have an explicit license in the source code. You may add an open-source license (e.g., MIT, GPL) if desired. Until then, please contact the author or repository owner for usage permissions.

Enjoy exploring tumor growth dynamics! Feel free to adapt this framework for your own research or educational projects. If you find any issues or would like to suggest improvements, open an issue or pull request.
