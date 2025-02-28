
<h2> PDE Section </h2>

The basis of the mathematical model, is modelling different cell populations as volume fractions.


Each volume fraction of each cell type is defined as $\varphi_i (\hat x, t)$ where $i = {H, P, D, N}$, corresponding to each cell type.
For ease of use we define the entire tumor volume fraction as $\varphi_T = \sum_i \varphi_i$
Because this simulation models volume fractions, at each voxel (volume pixel or each grid point), all cell volumes plus the host region $\varphi_H$ should sum to 1: $\varphi_T + \varphi_H = 1$.


Each cell volume fraction must satisfy the following mass conservation equation:


$$ \frac{\partial \varphi_i}{\partial t} + \nabla \cdot ( u_s\varphi_i - J_i) = Src_i $$

The details of $u_s$ can be found in the solid_velocity.md file, $J_i$ in the mass_flux.md file, $Src_i$ in the source_terms.md file.
