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
\frac{\partial C_S}{\partial t} = D_S \nabla^2 C_S  + \lambda_S C_S - \mu_S C_S - \gamma_S C_S
$$

$$
\frac{\partial C_P}{\partial t} = D_P \nabla^2 C_P + \gamma_S C_S - \gamma_P C_P - \mu_P C_P
$$

$$
\frac{\partial C_D}{\partial t} = D_D \nabla^2 C_D + \gamma_P C_P - \mu_D C_D
$$

$$
\frac{\partial C_N}{\partial t} = D_N \nabla^2 C_N + \mu_D C_D + \mu_P C_P + \mu_S C_S
$$

The variables are defined as follows:
<ul>
    <li>$\lambda_i$ is the rate of stem cell division.</li>
    <li>$\mu_i$ is the rate of stem cell death.</li>
    <li>$\gamma_i$ is the rate of stem cell proliferation.</li>
    <li>$D_i$ is the diffusion coefficient.</li>
</ul>


<h3>References</h3>

<p>
    [1] https://pmc.ncbi.nlm.nih.gov/articles/PMC5756149/#S7
</p>
