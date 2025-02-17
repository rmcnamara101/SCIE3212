import numpy as np

def laplacian(C, D, dx):
    """ Compute Laplacian with Neumann boundary conditions. """
    dCdx2 = np.zeros_like(C)
    dCdx2[1:-1] = D * (C[2:] - 2*C[1:-1] + C[:-2]) / dx**2
    return dCdx2

def run_simulation(params, Nt, Nx, dx, dt):
    """ Runs the tumor growth simulation with given parameters. """
    L, D1, D2, D3, D_N, lambda1, gamma1, gamma2, mu1, mu2, mu3, alpha1, alpha2, K_N = params
    x = np.linspace(0, L, Nx)
    
    # Initialize fields
    C1 = np.exp(-50 * (x - 0.5)**2)
    C2 = np.zeros_like(x)
    C3 = np.zeros_like(x)
    N = np.ones_like(x)
    
    history = {"C1": [], "C2": [], "C3": [], "N": []}

    for t in range(Nt):
        dC1dx2 = laplacian(C1, D1, dx)
        dC2dx2 = laplacian(C2, D2, dx)
        dC3dx2 = laplacian(C3, D3, dx)
        dNdx2 = laplacian(N, D_N, dx)

        # Reaction terms
        C1_new = C1 + dt * (dC1dx2 + lambda1 * C1 * N / (N + K_N) - mu1 * C1 - gamma1 * C1)
        C2_new = C2 + dt * (dC2dx2 + gamma1 * C1 * N / (N + K_N) - gamma2 * C2 - mu2 * C2)
        C3_new = C3 + dt * (dC3dx2 + gamma2 * C2 - mu3 * C3)
        N_new = N + dt * (dNdx2 - alpha1 * C1 * N - alpha2 * C2 * N)

        # Ensure non-negative values
        C1, C2, C3 = np.maximum(C1_new, 0), np.maximum(C2_new, 0), np.maximum(C3_new, 0)
        N = np.maximum(N_new, 0)

        history["C1"].append(C1.copy())
        history["C2"].append(C2.copy())
        history["C3"].append(C3.copy())
        history["N"].append(N.copy())

    return x, history
