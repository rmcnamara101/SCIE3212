<h1>SCIE3212</h1>

<h2>Tumor Growth Model in Differing Media</h2>

<h3>Overview</h3>


This project is a model of organoid tumor growth in a 3D domain.
The model is a system of partial differential equations (PDEs) that describe the growth of the tumor.
The approach of this model is considering tumor growth as the interplay between the follwing cell types:
<ul>
    <li>Stem Cells (SC)</li>
    <li>Progenitor Cells (PC)</li>
    <li>Differentiated Cells (DC)</li>
    <li>Necrotic Cells (NC)</li>
</ul>

The stem cells are the healthy cells that can divide and differentiate into the other two cell types.
The progenitor cells are the cells that can divide and differentiate into the differentiated cells.
The differentiated cells are the cells that cannot divide or differentiate further.
The necrotic cells are the cells that are dead and do not divide or differentiate.

The model is a system of diffusion-reaction equations that describe the growth of the tumor.
We first define each of the concentration/density fields for each of the cell types $C_S(\hat x, t)$, $C_P(\hat x, t)$, $C_D(\hat x, t)$, and $C_N(\hat x, t)$.
The $\hat x$ is the spatial position and $t$ is the time.

We can model the time evolution of these concentration fields using an adjusted set of equations proposed in [1].
The equations are as follows:

$$
\frac{\partial C_S}{\partial t} + \nabla (u_s C_S) = - \nabla \cdot J_S + \lambda_S n C_S(2p_0 - 1) - \mu_S \mathcal{H}(\hat n_S - n) C_S
$$

$$
\frac{\partial C_P}{\partial t} + \nabla (u_s C_P) = - \nabla \cdot J_P + \lambda_S n 2C_S(1 - p_0) + \lambda_P n C_P(2p_1 - 1) - \mu_P \mathcal{H}(\hat n_P - n) C_P
$$

$$
\frac{\partial C_D}{\partial t} + \nabla (u_s C_D) = - \nabla \cdot J_D + \lambda_P n C_P(1 - p_1) - \mu_D \mathcal{H}(\hat n_D - n) C_D - \alpha_D C_D
$$

$$
\frac{\partial C_N}{\partial t} + \nabla (u_s C_N) = - \nabla \cdot J_N + \mu_S \mathcal{H}(\hat n_S - n) C_S + \mu_P \mathcal{H}(\hat n_P - n) C_P + \mu_D \mathcal{H}(\hat n_D - n) C_D + \alpha_D C_D - \gamma_N C_N
$$

The parameters are defined as follows:
<ul>
    <li>$\lambda_i$ is the rate of i cell division.</li>
    <li>$\mu_i$ is the rate of i cell death.</li>
    <li>$\gamma_N$ is the lysis rate of necrotic cells.</li>
    <li>$\alpha_D$ is the apoptosis rate of differentiated cells.</li>
    <li>$u_s$ is the solid velocity.</li>
    <li>$p_0$ is the probability of stem cell division.</li>
    <li>$p_1$ is the probability of progenitor cell division.</li>
    <li>$J_i$ is the flux of i cell.</li>
    <li>$\hat n_i$ is the required nutrient level for i cell.</li>
</ul>

The mass flux of each cell type is given by the generalised Fick's law:

$$
J_i = -M_i \nabla C_i\left ( \frac{\delta E}{\delta C_i} \right )
$$

Where $M_i$ is the cell mobility, and $E$ is adhesion energy functional given by:

$$
E = \frac{\gamma}{\epsilon}\int_{\Omega} f(C_T) + \frac{\epsilon^2}{2}\abs{\nabla C_T}^2 dx
$$

Where it's variational derivative is given by:

$$
\frac{\delta E}{\delta C_i} = \frac{\gamma}{\epsilon} \left (f'(C_i) + \epsilon \nabla^2 C_T \right )
$$

Where $f(C_T)$ is the double well potential:

$$
f(C_T) = \frac{1}{4}C_T^2(1 - C_T)^2
$$

Note that $C_T = C_S + C_P + C_D + C_N$ is the total cell density.


Now that we have the mass flux, we can model the solid velocity $u_s$ as:

$$
u_s = - \left ( \nabla p + \frac{\delta E}{\delta C_T} \nabla C_T \right )
$$

Where $p$ is the internal pressure field. This equation is Darcy's Law, where the first term describes the internal pressure generating an outward flow of the solid material, and the second term describes the opposition to the flow due to the adhesion energy (how much the cells want to stick together).

The internal pressure $p$ is given by:

$$
\nabla ^2 p = S_T - \nabla \cdot \left  ( \frac{\delta E}{\delta C_T} \nabla C_T \right )
$$

Where $S_T$ is the source of the tumor cells given by $ S_T = \lambda_S n C_S + \lambda_P n C_P + \gamma_N C_N$.


This builds the system of equations for the tumor growth model. Going further, we can model the effect of other concentrations of other nutrient fields, such as oxygen, growth inhibitors, by incorporating them into the probability terms $p_0$ and $p_1$.



<h3>Current Questions</h3>

- What will the impact of the media be on growth? Media viscosity? Density? Intra-molecular forces?
- How can we model the physical effect of the media on the cell growth rates? Changes in internal pressure? Reduction/increase in the adhesion energy of the cells (probably not I assume this is more of a mechanical effect of the cells intracellular forces. Unless the media is positively charged and as tumor cells are negativey charged, this could be a factor)?

<h3>References</h3>

<p>
    [1] https://pmc.ncbi.nlm.nih.gov/articles/PMC5756149/#S7
</p>
