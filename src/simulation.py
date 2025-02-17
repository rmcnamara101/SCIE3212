import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.utils import experimental_params  # Import the experimental parameters

def laplacian(C, D, dx):
    """ Compute Laplacian with Neumann boundary conditions. """
    dCdx2 = np.zeros_like(C)
    dCdx2[1:-1] = D * (C[2:] - 2*C[1:-1] + C[:-2]) / dx**2
    return dCdx2

def run_simulation(params, Nt, Nx, dx, dt):
    """ Runs the tumor growth simulation with given parameters. """
    # Unpack parameters from the experimental_params dictionary
    L = 5.0  # Set L to a specific value or pass it as a parameter
    D1 = experimental_params["D_1"]
    D2 = experimental_params["D_2"]
    D3 = experimental_params["D_3"]
    D_N = experimental_params["D_N"]
    lambda1 = experimental_params["lambda_1"]
    gamma1 = experimental_params["gamma_1"]
    gamma2 = experimental_params["gamma_2"]
    mu1 = experimental_params["mu_1"]
    mu2 = experimental_params["mu_2"]
    mu3 = experimental_params["mu_3"]
    alpha1 = experimental_params["alpha_1"]
    alpha2 = experimental_params["alpha_2"]
    K_N = experimental_params["K_N"]
    x = np.linspace(0, L, Nx)
    
    # Initialize fields
    C1 = np.exp(-50 * (x - 2.5)**2)
    C2 = np.zeros_like(x)
    C3 = np.zeros_like(x)
    N = np.ones_like(x)
    
    history = {"C1": [], "C2": [], "C3": [], "N": []}

    for t in range(Nt):
        dC1dx2 = laplacian(C1, D1, dx)
        dC2dx2 = laplacian(C2, D2, dx)
        dC3dx2 = laplacian(C3, D3, dx)

        # Reaction terms
        C1_new = C1 + dt * (dC1dx2 + lambda1 * C1  - mu1 * C1 - gamma1 * C1)
        C2_new = C2 + dt * (dC2dx2 + gamma1 * C1 - gamma2 * C2 - mu2 * C2)
        C3_new = C3 + dt * (dC3dx2 + gamma2 * C2 - mu3 * C3)


        # Ensure non-negative values
        C1, C2, C3 = np.maximum(C1_new, 0), np.maximum(C2_new, 0), np.maximum(C3_new, 0)

        history["C1"].append(C1.copy())
        history["C2"].append(C2.copy())
        history["C3"].append(C3.copy())

    return x, history
