
<h2> Model Outline </h2>

The basis of the mathematical model, is modelling different cell populations as volume fractions. In general, the cells can grow, die, dissolve, proliferate, etc as governed by the growth rates, etc. In general this simulation can be viewed as:

                                  λ_1                          λ_2                             λ_3
                             |-----------|             |--------------|                |---------------|
                             \/          |             \/             |                \/              |
                    |---------------|         |------------------|         |----------------------|                     |-----------------|    γ                           
                    |  Cell Type 1  |   -->   |   Cell Type 2    |   -->   |     Cell Type 3      |   -->   ...   -->   | Nectrotic Cells |   -->   
                    |---------------|   p1    |------------------|   p2    |----------------------|   p3                |-----------------|
                            |                          |                               |                                         |
                            |--------------------------|-------------------------------------------------------------------------|
                            μ                          μ


Where each volume fraction must obey its mass conservation equation:

$$  \frac{\partial \varphi_i}{\partial t} + \nabla \cdot (u_s \varphi_i - J_i) = Src_i$$

The details of $u_s$ can be found in the solid_velocity.md file, $J_i$ in the mass_flux.md file, $Src_i$ in the source_terms.md file.

For the currently implemented system, the simulation can be views as, and the cell volum fractions equations defined as:


                                   λ                            λ
                             |-----------|             |--------------|
                             \/          |             \/             |
                    |---------------|         |------------------|         |----------------------|    μ    |-----------------|    γ                           
                    | Healthy Cells |   -->   | Progenitor Cells |   -->   | Differentiated Cells |   -->   | Necroctic Cells |   -->
                    |---------------|   p0    |------------------|   p1    |----------------------|         |-----------------|
                            |                          |                                                             |
                            |--------------------------|-------------------------------------------------------------|
                            μ                          μ


Each volume fraction of each cell type is defined as $\varphi_i (\hat x, t)$ where $i = {H, P, D, N}$, corresponding to each cell type.
For ease of use we define the entire tumor volume fraction as $\varphi_T = \sum_i \varphi_i$
Because this simulation models volume fractions, at each voxel (volume pixel or each grid point), all cell volumes plus the host region $\varphi_H$ should sum to 1: $\varphi_T + \varphi_H = 1$.



The entire set of PDEs is given below:


$$
\frac{\partial \varphi_S}{\partial t} + \nabla (u_s \varphi_S) = - \nabla \cdot J_S + \lambda_S n \varphi_S(2p_0 - 1) - \mu_S \mathcal{H}(\hat n_S - n) \varphi_S
$$

$$
\frac{\partial \varphi_P}{\partial t} + \nabla (u_s \varphi_P) = - \nabla \cdot J_P + \lambda_S n 2\varphi_S(1 - p_0) + \lambda_P n \varphi_P(2p_1 - 1) - \mu_P \mathcal{H}(\hat n_P - n) \varphi_P
$$

$$
\frac{\partial \varphi_D}{\partial t} + \nabla (u_s \varphi_D) = - \nabla \cdot J_D + \lambda_P n \varphi_P(1 - p_1) - \mu_D \mathcal{H}(\hat n_D - n) \varphi_D - \alpha_D C_D
$$

$$
\frac{\partial \varphi_N}{\partial t} + \nabla (u_s \varphi_N) = - \nabla \cdot J_N + \mu_S \mathcal{H}(\hat n_S - n) \varphi_S + \mu_P \mathcal{H}(\hat n_P - n) \varphi_P + \mu_D \mathcal{H}(\hat n_D - n) \varphi_D + \alpha_D \varphi_D - \gamma_N \varphi_N
$$