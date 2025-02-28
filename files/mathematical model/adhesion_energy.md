
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