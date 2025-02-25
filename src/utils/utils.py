
experimental_params = {

    # self renewal rates
    "lambda_S": 1, # self renewal of stem cells
    "lambda_P": 5, # self renewal of progenitor cells

    # death rates
    "mu_S": 0.001, # death rate of stem cells
    "mu_P": 0.001, # death rate of progenitor cells
    "mu_D": 0.001, # death rate of differentiated cells

    # differentiation rates
    "gamma_S": 10,
    "gamma_P": 3,
    

    # nutrient dependent rates
    "alpha_D": 0.001,

    # n
    "gamma_N": 0.1,


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