# SCIE3121 Distilled Model

For this SCIE3121 project, I will construct a distilled version of the model outlined in model_outline.md

We will consider 3 cell populations:

<ul>
    <li> Healthy Cells </li>
    <li> Diseased Cells </li>
    <li> Necrotic Cells </li>>
</ul>

We will define each cell types volume fraction field as: $\varphi_i$, where $i\in {H, D, N}$. We also define $\varphi_T = \varphi_H + \varphi_D + \varphi_N$, the total volume fraction. 
We then define the host region $\varphi_h$, allowing us to enforce the condition $\varphi_h + \varphi_T= 1$.

The evolution of each cell volume fraction, is governed by its corresponding equation:

$$ \frac{\partial \varphi_H}{\partial t} = -\nabla \cdot ( u_s + J_H) + Src_H $$

$$ \frac{\partial \varphi_D}{\partial t} = -\nabla \cdot ( u_s + J_D) + Src_D $$

$$ \frac{\partial \varphi_N}{\partial t} = -\nabla \cdot ( u_s + J_N) + Src_N $$


We also introduce new source terms.

| Cell Type | Source Term |
| --- | --- |
| Healthy Cells | $Src_H = \lambda_H n \varphi_H ( 2p_H - 1 ) - \mu_H \mathcal{H}(n_H - n) \varphi_H $ |
| Diseased Cells | $Src_D$ = 2\lambda_s n \varphi_H ( p_H - 1 ) + \lambda_D n \varphi_D ( 2p_D - 1 ) - \mu_D \mathcal{D}(n_D - n) \varphi_D$ |
| Necrotic Cells | $Src_N = \mu_H \mathcal{H}(n_H - n) \varphi_H + \mu_D \mathcal{D}(n_D - n) \varphi_D  - \mu_N \varphi_N$ |

Where the $\lambda_i$ are growth rates, $\mu_i$ are the death rates (decay rate for necrotic cell population), $n_i$ refers to each cell types satiety value in terms of the general nutrient field and $\mathcal{H}$ is the Heaviside function.


<h2> Adhesion Energy </h2>

The adhesion energy functional $E$ can be written as follows:

$$
E = \frac{\gamma}{\epsilon}\int_{\Omega} f(C_T) + \frac{\epsilon^2}{2}|{\nabla \varphi_T}|^2 dx
$$

Where $\gamma$ represents the cell adhesion (surface tension) and $\epsilon$ is the thickness of the tumor-host interface.

The variational derivative with respect to each cell population is given by:


$$
\frac{\delta E}{\delta \varphi_i} = \frac{\gamma}{\epsilon} \left (f'(\varphi_T) + \epsilon^2 \nabla^2 \varphi_T \right )
$$

Where $f(\varphi_T)$ is the double well potential:

$$
f(\varphi_T) = \frac{1}{4}\varphi_T^2(1 - \varphi_T)^2
$$


<h2> Solid Velocity </h2>

We can model the solid velocity $u_s$ as:

$$
u_s = - \left ( \nabla p + \frac{\delta E}{\delta \varphi_T} \nabla \varphi_T \right )
$$

Where $p$ is the internal pressure field. This equation is Darcy's Law, where the first term describes the internal pressure generating an outward flow of the solid material, and the second term describes the opposition to the flow due to the adhesion energy (how much the cells want to stick together).

The internal pressure $p$ is given by:

$$
\nabla ^2 p = S_T - \nabla \cdot \left  ( \frac{\delta E}{\delta \varphi_T} \nabla \varphi_T \right )
$$

Where $S_T$ is the source of the tumor cells given by $S_T = Src_H + Src_P + Src_D + Src_N$.


Combing these two equations:

$$ \nabla \cdot u_s = -\nabla^2 p - \nabla \cdot \left ( \frac{\delta E}{\delta \varphi_T}\nabla \varphi_T \right ) = -Src_T + \nabla \cdot \left(\frac{\delta E}{\delta \varphi_T} \right ) - \nabla \cdot \left ( \frac{\delta E}{\delta \varphi_T}\nabla \varphi_T \right ) = Src_T$$

we can see that the net divergence of the solid velocity field is equal to the source terms. This makes physical sense, as the creation of mass and volume from source terms should increase the internal pressure and drive the volumetric expansion of the tumor.

<h2> Mass Flux </h2>


The mass flux is then defined as:

$$ J_i = M_i \left ( \frac{\delta E}{\delta\varphi_i} \right ) \nabla\varphi_i $$

Where $M_i$ is the cell mobility.

So when taking the divergence:

$$ \nabla \cdot J_i = M_i (\nabla \cdot\left ( \frac{\delta E}{\delta\varphi_i} \right ) \nabla\varphi_i  + \left ( \frac{\delta E}{\delta\varphi_i} \right ) \nabla^2\varphi_i )$$

It can be seen that this is the diffusive term governed by the adhesion energy of the cells. This term ensures the tumor sticks together.


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


Each volume fraction of each cell type is defined as $\varphi_i (\hat x, t)$ where $i = {H, D, N}$, corresponding to each cell type.
For ease of use we define the entire tumor volume fraction as $\varphi_T = \sum_i \varphi_i$
Because this simulation models volume fractions, at each voxel (volume pixel or each grid point), all cell volumes plus the host region $\varphi_H$ should sum to 1: $\varphi_T + \varphi_H = 1$.

