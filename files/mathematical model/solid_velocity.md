
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

$$ \nabla \cdot u_s = -\nabla^2 p - \nabla \cot \left ( \frac{\delta E}{\delta \varphi_T}\nabla \varphi_T \right ) = -Src_T + \nabla \cdot \left(\frac{\delta E}{\delta \varphi_T} \right ) - \nabla \cot \left ( \frac{\delta E}{\delta \varphi_T}\nabla \varphi_T \right ) = Src_T$$

we can see that the net divergence of the solid velocity field is equal to the source terms. This makes physical sense, as the creation of mass and volume from source terms should increase the internal pressure and drive the volumetric expansion of the tumor.