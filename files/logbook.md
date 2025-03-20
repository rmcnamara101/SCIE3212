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
For ease of use we define the entire tumor volume fraction as $\varphi_T = \sum_i \varphi_i$
Because this simulation models volume fractions, at each voxel (volume pixel or each grid point), all cell volumes plus the host region $\varphi_H$ should sum to 1: $\varphi_T + \varphi_H = 1$.

Each cell volume fraction must satisfy the following mass conservation equation:

$$ \frac{\partial \varphi_i}{\partial t} + \nabla \cdot ( u_s\varphi_i - J_i) = Src_i $$

Where $u_s$ is the solid velocity of the tumor, $J_i$ is the mass flux, and $Src_i$ refers to the source terms of the cell population.
The solid velocity and mass flux terms are the terms that drive the diffusion and volumetric growth of the tumor. 

Both terms are dependant on the adhesion energy functional.

$$ E = \frac{\gamma}{\varepsilon} \int_{\Omega} \frac{\varepsilon^2}{2}|\nabla \varphi_T|^2  + f(\varphi_T) dx$$

Where the variational derivative for each cell volume fraction is given by:

$$ \frac{\delta E}{\delta \varphi_i} = \frac{\gamma}{\varepsilon} (\varepsilon^2 \nabla^2 \varphi_T + f'(\varphi_T))$$ 


Now that the adhesional energy functional and derivatives are found, we can now find the mass flux and solid velocity vector fields.
The solid velocity vector field is defined by Darcy's Law:

$$ u_s = - (\nabla p + \frac{\delta E}{\delta \varphi_T}\nabla \varphi)  $$

Where $p$ is the internal pressure of the system, defined as

$$\nabla ^2 p = Src_T - \nabla \cdot \left(\frac{\delta E}{\delta \varphi_T} \right ) $$

Combing these two equations:

$$ \nabla \cdot u_s = -\nabla^2 p - \nabla \cot \left ( \frac{\delta E}{\delta \varphi_T}\nabla \varphi_T \right ) = -Src_T + \nabla \cdot \left(\frac{\delta E}{\delta \varphi_T} \right ) - \nabla \cot \left ( \frac{\delta E}{\delta \varphi_T}\nabla \varphi_T \right ) = Src_T$$

we can see that the net divergence of the solid velocity field is equal to the source terms. This makes physical sense, as the creation of mass and volume from source terms should increase the internal pressure and drive the volumetric expansion of the tumor.

The mass flux is then defined as:

$$ J_i = M_i \nabla\varphi_i \left ( \frac{\delta E}{\delta\varphi_i} \right )$$

Finally we can define a nutrient scalar field $n(\hat x, t)$, which represents the general nutrient required for tumor growth. As nutrient diffusion occurs on much smaller time scales as cell growth does, we can model $n$ as a steady state difussion equation.

$$ 0 = \nabla (D_n \nabla n) - n (\omega^n_H\varphi_H + \omega^n_P\varphi_P + \omega^n_D\varphi_D) + p_n(1- \varphi_T)(\hat n - n)

Where $\omega_i^n$ refers to the uptake rate of the nutrient $n$ of each cell population, $p_n$ is the permeability of the nutrient and $\hat n$ is the nutrient concentration in the media.

We can now finally define the source terms for each cell population.
Each source term is defined as: 

| Cell Type            | Source Terms                                                                                                                                               |
|----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Healthy Cells        | $\lambda_H n \varphi_H(2p_0 -1) - \mu_H \mathbb{H}(n_s - n)\varphi_H$                                                                                      |
| Progenitor Cells     | $\lambda_H n 2\varphi_S(1 - p_0) + \lambda_P n \varphi_P(2p_1 - 1) - \mu_P \mathbb{H} (n_P -n)\varphi_P$                                                   |
| Differentiated Cells | $\lambda_P n 2\varphi_P(1-p_1) - \mu_D\mathbb{H}(n_D - n)\varphi_D - \alpha_D \varphi_D$                                                                   |
| Necrotic Cells       | $\mu_H \mathbb{H}(n_s - n)\varphi_H + \mu_P \mathbb{H} (n_P -n)\varphi_P + \mu_D\mathbb{H}(n_D - n)\varphi_D + \alpha_D \varphi_D - \gamma_N \varphi_N$   |

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

<h3> 28/2/25 </h3>

I have implemented functionality into the code to be able to run larger simulations and save them to a file, where analysis can be done afterwards. This makes things much easier and quicker, so simulations can be run overnight etc, and then the sim can be analysed easily after. See model_exploration jupyter notebook for how to do this.

I have a few questions about where to head next, should talk to Gloria/ Mark about the direction to head next. But my initial thoughts is about how we can capture the impact of the host fluid that the organoids are grown in. My initial instict is there are two routes that could be taken, in terms of how the model is built. I could explore physical effects of the fluid, that I would suspect would manifest in the mass flux. The biggest effect this would have is altering the adhesion energy functional in some way, which would in turn effect how the cells move along the adhesion energy gradients. Or I could explore how different ingredient concentrations of the host media interact with the source terms. This would drive growth through the solid velocity term, but would require some altering of the growth/death/proliferation elements are calculated. They would be functions of the ingredients of the host fluid, and I would have to implement functionality for many different nutrient fields, that represent the ingredients of the host fluid. Each ingredient field would just abide by some diffusion equation. 

This is a representations of the simulation:



                                   λ                            λ
                             |-----------|             |--------------|
                             \/          |             \/             |
                    |---------------|         |------------------|         |----------------------|    μ    |-----------------|    γ                           
                    | Healthy Cells |   -->   | Progenitor Cells |   -->   | Differentiated Cells |   -->   | Necroctic Cells |   -->
                    |---------------|   p0    |------------------|   p1    |----------------------|         |-----------------|
                            |                          |                                                             |
                            |--------------------------|-------------------------------------------------------------|
                            μ                          μ



Questions:

- How important is it to have these 3 cell types (nectrotic is irrelevant only there for the physical effects). Simon at work said may potentially not have to consider the progenitor cell population and that I could potentially have a cell evolution of just: Healthy -> Differentiated -> nectrotic

- We could incorporate the effects of the ingredient fields in two ways. Either through the self renewal rates (λ), and death rates (μ). Or through the cell progression probabilities (p0, and p1). The ingredients either directly effect how quickly the cells self renew, or die, or they directly effect the probabilities for cell differentiation. Currently in the simulation p0 and p1 are just constants, but in the orginial paper they take into account certain growth inhibitors etc, so there is a method already to include something like this in that fashion. 

This could be implemented in the model by introducing an effective renewal rate:

$$ \lambda_{eff} = \lambda_i \prod_i f(I_i)$$

Where $I_i$ is each ingredient field, and $f$ is some function that can represent how the ingredient effects that cell type's growth.

- I want to also explore how I can add properly add uncertainties through the model, as we talked about in some of the earlier meetings, as I expect any data that we get from G4L is some sort of percentage range of concentration or densiy of cell types. Thus these errors should also propogate out to the final state of the model.

- I also need to tweak the growth rates in its current state to represent reality. I don't know if this should be done by experimental verification or by constants pre-defined in literature (if they exist).

<h3> 3-3-25 </h3>

Today is the day before meeting up with my supervisor just to go over the basis of the mathematical model. I am running some larger simulations, there are some strange artifacts, but Im not sure how significant they are. For example there seems to be some "leakage" of the cells from the central tumor, Im not sure if its an issue with the plotting, or just the simulation. I also need some other plotting method, its so hard to see whats really happening. The problem is the simulations take so long if I am doing any meaningful simulations, they take so long. But I think it is time to start getting ready to implement a cell cycle model that is accurate to the ovarian cancer cells that I think we are investigating. This means researching growth rates, death rates, etc, if they are in the literature, or potentially matching physical data to the simualtion. This I assume are the next steps of the project. Until that I think that I need to find suitable simulation parameters to confirm simulational growth matches reality. 

Also want to note that currently the sim has some issues if all the cell types are set to the exact same value at the start. They seem to couple and are all the same value essentially for the whole simulation.

<h3> 11-3-25 </h3>

Since the last logbook entry I have implemented the new simplified version of the model, reducing the cell types down to just healthy, diseased and necrotic. This has allowed me to implement a cell cycle model, which is defined in SCIE3121_model.md.

I now have been looking into defining meaningful growth rates, death rates.

First I will define the units of the simulation. We set dx = 1mm, and dt = 1 day. The nutrient concentration is defined as a unitless fraction of the maximum nutrient concentration.
Which allow us to define the units of the simulation parameters:

| Parameter | Units |
|-----------|-------|
| $\lambda_H$ | day$^{-1}$ |
| $\lambda_D$ | day$^{-1}$ |
| $\mu_H$ | day$^{-1}$ |
| $\mu_D$ | day$^{-1}$ |
| $\alpha_D$ | day$^{-1}$ |
| $\gamma_N$ | day$^{-1}$ |
| $p_0$ | unitless |
| $p_1$ | unitless |
| $\hat n$ | unitless |
| $D_n$ | mm$^2$ day$^{-1}$ |
| $\gamma$ | J mm$^{-2}$ |
| $\varepsilon$ | mm |
| $M_i$ | mm$^5$ day$^{-1}$ |

In our last meeting with Gloria and Mark we discussed the validity of the model, and how it was yielding some strange results. Some expected results were to see spherical growth, and also we were expecting to also see a necrotic core form. However, the simulation was yielding some strange growth, and no necrotic core.

Since then, after implementing the new cell cycle model, I have adjusted the cell mobility of the necrotic cells to be much smaller than the other cell types. This is beleive would be the most simple and physically consistent way to yield a necrotic core. The way the necrotic cells interact with the physical portions of the equations are in the two following ways: through the solid velocity field, and the mass flux. The necrotic cells essentially do not contribute to the solid velocity field, as the necrotic sources are the same as the healthy and diseased sinks.

After writing this out I had a sudden realisation that I was not properly implementing the solid velocity terms. In the solid velocity file, I showed that the divergence of the solid velocity field is equal to the source terms, which I just assumed was the sum of all the source terms individually. However, thinking I realise that I should not be considering any necrotic source terms, as they are just mass exchange terms. The death rates cancel if you simply add all the sources together, yet the necrotic mass exchange terms do not cancel. This is problematic, as a mass exchange term should not contribute to any increase in the internal pressure of the system, as they were already there.

Anyways, continuing on, thus the way the necrotic cells interact with the physical portions of the equations are through the mass flux. Which is essentially a function of the adhesion energy functional. The necrotic cells, although certainly not containing much internal energy, would still move along the adhesion energy gradients, just at a much slower rate than the other cell types. This is encapsulated by reducing the mobility of the necrotic cells, which after a few tests, seem to be yielding more of a necrotic core.

<h3> 14-3-25 </h3>

After a huge deep dive on the model, I realised this was all not working properly, and everything was just messed up. I went one by one through the global physical fields, starting with the nutrient diffusion. All the analysis of the fields can be found in the analysis/dynamics_analysis.ipynb file. This plots all the fields of a loaded simulation.

The pressure field was not working properly so I had to change the solver using some scipy solver, which fix everything. A nice pressure surface can be seen which is exactly what I expected to see now. The pressure gradient points inwards, as expected so the -grad p points outwards driving growth. 
The cell adhesion is also clearly driving forces at the boundary of the tumor, as expected. And now I am beginning to see the necrotic core form. 

I am considering options about changing cell types, and pathways, nutrient levels, to see if there can be more defining characteristics mimicked by the model. Next steps would be to get some experimental data to match the growth rates, and death rates, and validate that the model is physcially accurate. I am optimistic right now, as the model is behaving as expected, generating very nice spheroids.

<h3> 18-3-25 </h3>

Ok I have the data and it is not great by my first look, mainly because its drugged spheroids and not just base growth. But otherwise I think the data would be useful, its nice though I finally have a real timescale. 0-160 hours. So by my calculations, my simulation is quite stable with dt = 0.1-0.5, so if dt represents 1 hour then I should need about 1600 or less time steps which isnt too bad. Also the data is on the order of magnitude of micrometers. 

<h3> 20-3-25 </h3>

Today I was able to plot the cell sources, revealing the quiescent layer of cells, exactly as expected. A deep solid necrotic core, with a surrounding quiescent
layer, and then another proliferating layer around that, exactly as expected. Now is the perfect stage to move onto matching data, however I think the random blob 
tumor is still broken, I will look to fix that for more natural growth.

The random blob tumors are now working nicely, still developing into spheroids. These will most likely be my starting point now, as it would be much more realistic than perfectly spherical tumors. I am now primed to start calibrating the simulation with raw growth data. Unfortunately I only got drugged organoid evolution in the first dataset I was given. I want to now maybe make some nice plots, showing the evolution for my proposal.