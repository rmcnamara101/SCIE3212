
<h2> Source Terms </h2>

To define the source terms we first must introduce a general nutrient scalar field $n(\hat x, t)$, which represents the general nutrient required for tumor growth. As nutrient diffusion occurs on much smaller time scales as cell growth does, we can model $n$ as a steady state difussion equation.

$$ 0 = \nabla (D_n \nabla n) - n (\omega^n_H\varphi_H + \omega^n_P\varphi_P + \omega^n_D\varphi_D) + p_n(1- \varphi_T)(\hat n - n)

Where $\omega_i^n$ refers to the uptake rate of the nutrient $n$ of each cell population, $p_n$ is the permeability of the nutrient and $\hat n$ is the nutrient concentration in the media.

This will allow us to implement the fact that there should be a certain nutrient level for there to be growth. 

We now define two probabilities $p_H$ and $p_P$, which represent the probabilities for cell proliferation. $p_H$ represents the probability that the healthy cells will undergo mitosis, and $p_

We can now define the source terms for each cell population.
Each source term is defined as: 

| Cell Type            | Source Terms                                                                                                                                               |
|----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Healthy Cells        | $\lambda_H n \varphi_H(2p_0 -1) - \mu_H \mathbb{H}(n_s - n)\varphi_H$                                                                                      |
| Progenitor Cells     | $\lambda_H n 2\varphi_S(1 - p_0) + \lambda_P n \varphi_P(2p_1 - 1) - \mu_P \mathbb{H} (n_P -n)\varphi_P$                                                   |
| Differentiated Cells | $\lambda_P n 2\varphi_P(1-p_1) - \mu_D\mathbb{H}(n_D - n)\varphi_D - \alpha_D \varphi_D$                                                                   |
| Necrotic Cells       | $\mu_H \mathbb{H}(n_s - n)\varphi_H + \mu_P \mathbb{H} (n_P -n)\varphi_P + \mu_D\mathbb{H}(n_D - n)\varphi_D + \alpha_D \varphi_D - \gamma_N \varphi_N$   |

Where $\lambda_i$ refers to self proliferation rates, $\mu_i$ refers to the death rates, $n_i$ refers to the required satiety levels, $\alpha_i$ refers to aptosis rates, $\gamma_i$ refers to lysis rates.