#######################################################################################################
#######################################################################################################
#
#
#   This is the main access point for running tumor growth simulations.
#
#   It will initialize the tumor growth model, run the simulation, and then save the history of the
#   simulation. 
#
#   The user can then access then analyse the data.
#
#
# Author:
#   - Riley Jae McNamara
#
# Date:
#   - 2025-02-19
#
#
#
#######################################################################################################
#######################################################################################################


import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.tumor_growth import TumorGrowthModel
from src.visualization.plot_tumor import TumorPlotter
from src.visualization.animate_tumor import TumorAnimator

if __name__ == "__main__":

    model = TumorGrowthModel(dx = 0.0, dt = 0.001)
    model.run_simulation(steps=100)
    history = model.get_history()

    plotter = TumorPlotter(model)
    animator = TumorAnimator(model)
    #uncomment to plot surface
    #plotter.plot_tumor(step=100)

    # uncomment to plot the isosurfaces
    #plotter.plot_all_isosurfaces()

    # uncomment to plot the volume evolution
    #plotter.plot_volume_evolution()

    # uncomment to plot the radius evolution
    plotter.plot_radius_evolution()



    #animator.animate_tumor_slices()
    #animator.animate_single_slice()
    animator.animate_tumor_growth_isosurfaces()