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

Definitions of $u_s$ and $J_i$ are given in solid_velocity.md and mass_flux.md.

We also introduce new source terms.

| Cell Type | Source Term |
| --- | --- |
| Healthy Cells | $Src_H = \lambda_H n \varphi_H ( 2p_H - 1 ) - \mu_H \mathcal{H}(n_H - n) \varphi_H $ |
| Diseased Cells | $Src_D$ = 2\lambda_s n \varphi_H ( p_H - 1 ) + \lambda_D n \varphi_D ( 2p_D - 1 ) - \mu_D \mathcal{D}(n_D - n) \varphi_D$ |
| Necrotic Cells | $Src_N = \mu_H \mathcal{H}(n_H - n) \varphi_H + \mu_D \mathcal{D}(n_D - n) \varphi_D  - \mu_N \varphi_N$ |

Where the $\lambda_i$ are growth rates, $\mu_i$ are the death rates (decay rate for necrotic cell population), $n_i$ refers to each cell types satiety value in terms of the general nutrient field and $\mathcal{H}$ is the Heaviside function.
