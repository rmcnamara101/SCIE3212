
experimental_params = {

    # self renewal rates
    "lambda_S": 1, # self renewal of stem cells
    "lambda_P": 5, # self renewal of progenitor cells

    # death rates
    "mu_S": 0.00000001, # death rate of stem cells
    "mu_P": 0.00000001, # death rate of progenitor cells
    "mu_D": 0.00000001, # death rate of differentiated cells

    # differentiation rates
    "gamma_S": 10,
    "gamma_P": 3,
    

    # nutrient dependent rates
    "alpha_D": 0.001,

    # n
    "gamma_N": 10,


    # probability terms
    "p_0": 0.9,
    "p_1": 0.9,

    # nutrient saturation for each cell type
    "n_S": 0.01,
    "n_P": 0.01,
    "n_D": 0.01,
    #
    "n_max": 0.01,
    
    # constant terms in the adhesion energy equation
    "gamma": 5,
    "epsilon": 0.1,
    
    # diffusion constant of nutrient field
    "D_n": 0.1,

    "M": 0.001,
}

SCIE3121_params = {
    # growth rates
    "lambda_H": 3, # growth rate of healthy cells
    "lambda_D": 4.3, # growth rate of damaged cells

    # death rates
    "mu_H": 0.0000001, # death rate of healthy cells
    "mu_D": 0.000001, # death rate of damaged cells
    "mu_N": 0.000001, # death rate of necrotic cells

    # probability terms
    "p_H": 0.6,
    "p_D": 0.6,

    # nutrient saturation for each cell type
    "n_H": 0.0001,
    "n_D": 0.0001,

    # physical constants
    'gamma': 1,
    'epsilon': 0.05,                             
    'M': 0.001, # cell mobility

    # diffusion constant of nutrient field
    "D_n": 0.1,
}

# src/utils/utils.py
import numpy as np
import numba as nb

@nb.njit
def gradient(field, dx, axis):
    result = np.zeros_like(field)
    nx, ny, nz = field.shape
    if axis == 0:
        for i in nb.prange(1, nx - 1):
            for j in range(ny):
                for k in range(nz):
                    result[i, j, k] = (field[i+1, j, k] - field[i-1, j, k]) / (2 * dx)
    elif axis == 1:
        for i in nb.prange(nx):
            for j in range(1, ny - 1):
                for k in range(nz):
                    result[i, j, k] = (field[i, j+1, k] - field[i, j-1, k]) / (2 * dx)
    elif axis == 2:
        for i in nb.prange(nx):
            for j in range(ny):
                for k in range(1, nz - 1):
                    result[i, j, k] = (field[i, j, k+1] - field[i, j, k-1]) / (2 * dx)
    return result

@nb.njit
def laplacian(field, dx):
    result = np.zeros_like(field)
    nx, ny, nz = field.shape
    dx2 = dx * dx
    for i in nb.prange(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                result[i, j, k] = (
                    field[i+1, j, k] + field[i-1, j, k] +
                    field[i, j+1, k] + field[i, j-1, k] +
                    field[i, j, k+1] + field[i, j, k-1] -
                    6.0 * field[i, j, k]
                ) / dx2
    return result

@nb.njit
def divergence(ux, uy, uz, dx):
    result = np.zeros_like(ux)
    nx, ny, nz = ux.shape
    for i in nb.prange(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                result[i, j, k] = (
                    (ux[i+1, j, k] - ux[i-1, j, k]) / (2 * dx) +
                    (uy[i, j+1, k] - uy[i, j-1, k]) / (2 * dx) +
                    (uz[i, j, k+1] - uz[i, j, k-1]) / (2 * dx)
                )
    return result