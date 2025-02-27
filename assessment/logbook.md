<h1> SCIE3121 Logbook </h1>

<h3> 27/2/25 </h3>

This project is based around building software to simulate the growth of 3D organoids in differing media.

The basis of the mathematical model, is modelling different cell populations as volume fractions.
Currently the model simulates 4 cell populations
<ul>
    <li> Healthy Cells </li>
    <li> Progenitor Cells </li>
    <li> Differentiated Cells </li>
    <li> Nectrotic Cells </li>
</ul>

This I expect may be changed, as I am not sure how necessary it is to model each of these different cell types (Simon at work mentioned that I may not have to model the progenitor cells, and could just go from healthy cells to differentiated cells).

Each volume fraction of each cell type is defined as $\varphi_i (\hat x, t)$ where $i = {H, P, D, N}$, corresponding to each cell type.
For ease of use we define the entire tumor volume fraction as $ \varphi_T = \sum_i \varphi_i $
Because this simulation models volume fractions, at each voxel (volume pixel or each grid point), all cell volumes plus the host region $\varphi_H$ should sum to 1: $\varphi_T + \varphi_H = 1$.

Each cell volume fraction must satisfy the following mass conservation equation:

$$ \frac{\partial \varphi_i}{\partial t} + \nabla \cot ( u_s\varphi_i - J_i) = Src_i $$

Where $u_s$ is the solid velocity of the tumor, $J_i$ is the mass flux, and $Src_i$ refers to the source terms of the cell population.
The solid velocity and mass flux terms are the terms that drive the diffusion and volumetric growth of the tumor. 

Both terms are dependant on the adhesion energy functional.

$$ E = \frac{\gamma}{\varepsilon} \int_{\Omega} \frac{\varepsilon^2}{2}|\nabla \varphi_T|^2  + f(\varphi_T) dx$$

Where the variational derivative for each cell volume fraction is given by:

$$ \frac{\delta E}{\delta \varphi_i} = \frac{\gamma}{\varepsilon} (\varepsilon^2 \nabla^2 \varphi_T + f'(\varphi_T))$$ 


Now that the adhesional energy functional and derivatives are found, we can now find the mass flux and solid velocity vector fields.
The solid velocity vector field is defined by Darcy's Law:

$$ u_s = - (\nabla p + \frac{\delta E}{\delta \varphi_T}\nabla \varphi)  $$

Where $p$ is the internal pressure of the system, defined as $\nabla ^2 p = Src_T - \nabla \cdot \left(\frac{\delta E}{\delta \varphi_T} \right ) $.

Combing these two equations we can see:

$$ \nabla \cdot u_s = -\nabla^2 p - \nabla \cot \left ( \frac{\delta E}{\delta \varphi_T}\nabla \varphi_T \right ) = -Src_T + \nabla \cdot \left(\frac{\delta E}{\delta \varphi_T} \right ) - \nabla \cot \left ( \frac{\delta E}{\delta \varphi_T}\nabla \varphi_T \right ) = Src_T$$

Thus we can see that the net divergence of the solid velocity field is equal to the source terms. This makes physical sense, as the creation of mass and volume from source terms should increase the internal pressure and drive the volumetric expansion of the tumor.

The mass flux is then defined as:

$$ J_i = M_i \nabla\varphi_i \left ( \frac{\delta E}{\delta\varphi_i} \right )$$

Finally we can define a nutrient scalar field $n(\hat x, t)$, which represents the general nutrient required for tumor growth. As nutrient diffusion occurs on much smaller time scales as cell growth does, we can model $n$ as a steady state difussion equation.

$$ 0 = \nabla (D_n \nabla n) - n (\omega^n_H\varphi_H + \omega^n_P\varphi_P + \omega^n_D\varphi_D) + p_n(1- \varphi_T)(\hat n - n)

Where $\omega_i^n$ refers to the uptake rate of the nutrient $n$ of each cell population, $p_n$ is the permeability of the nutrient and $\hat n$ is the nutrient concentration in the media.

We can now finally define the source terms for each cell population.
Each source term is defined as: 

|----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Cell Type            | Source Terms                                                                                                                                               |
|----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Healthy Cells        | $\lambda_H n \varphi_H(2p_0 -1) - \mu_H \mathbb{H}(n_s - n)\varphi_H$                                                                                      |
|----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Progenitor Cells     | $\lambda_H n 2\varphi_S(1 - p_0) + \lambda_P n \varphi_P(2p_1 - 1) - \mu_P \mathbb{H} (n_P -n)\varphi_P$                                                   |
|----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Differentiated Cells | $\lambda_P n 2\varphi_P(1-p_1) - \mu_D\mathbb{H}(n_D - n)\varphi_D - \alpha_D \varphi_D$                                                                   |
|----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Necrotic Cells       | $ \mu_H \mathbb{H}(n_s - n)\varphi_H + \mu_P \mathbb{H} (n_P -n)\varphi_P + \mu_D\mathbb{H}(n_D - n)\varphi_D + \alpha_D \varphi_D - \gamma_N \varphi_N$   |
|----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|

Where $\lambda_i$ refers to self proliferation rates, $\mu_i$ refers to the death rates, $n_i$ refers to the required satiety levels, $\alpha_i$ refers to aptosis rates, $\gamma_i$ refers to lysis rates.


We can define a 3D simulation grid of size $N \times N \times N$, with grid spacing $dx$, defining the simulation space as a cube of dimension $Ndx \times Ndx \times Ndx$.
We can then also define the time parameter $dt$.

                  o-o-o
                 / / /o   dx: distance between grid points - total sim volume = dx^3 * N * N * N
                o-o-o o   dt: time spacing between each time step of simulation
                o-o-o o
                o-o-o

The units of the simulation are defined by the values dx and dt. 

Currently at this point in time I have a working simulation based on these equations. 