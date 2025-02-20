
experimental_params = {

    # self renewal rates
    "lambda_S": 0.1, # self renewal of stem cells
    "lambda_P": 0.1, # self renewal of progenitor cells

    # death rates
    "mu_S": 0.1, # death rate of stem cells
    "mu_P": 0.1, # death rate of progenitor cells
    "mu_D": 0.1, # death rate of differentiated cells

    # differentiation rates
    "gamma_S": 0.1,
    "gamma_P": 0.1,
    

    # nutrient dependent rates
    "alpha_D": 0.1,

    # n
    "gamma_N": 0.1,


    # probability terms
    "p_0": 0.1,
    "p_1": 0.1,

    # nutrient saturation for each cell type
    "n_S": 0.01,
    "n_P": 0.01,
    "n_D": 0.01,
    #
    "n_max": 0.01,
    
    # constant terms in the adhesion energy equation
    "gamma": 0.1,
    "epsilon": 0.0001,
    
    # diffusion constant of nutrient field
    "D_n": 0.1,

    "M": 1,
}