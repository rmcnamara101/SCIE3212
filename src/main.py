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
from src.visualization.plot_tumor import VolumeFractionPlotter
from src.visualization.animate_tumor import TumorAnimator

if __name__ == "__main__":

    model = TumorGrowthModel(dx = 0.1, dt = 0.001)
    model.run_simulation(steps=200)
    history = model.get_history()

    plotter = VolumeFractionPlotter(model)
    animator = TumorAnimator(model)

    #uncomment to plot surface
    #plotter.plot_volume_fractions(20)
    #plotter.plot_volume_fraction_evolution('Necrotic')


    #animator.animate_tumor_slices()
    #animator.animate_single_slice()
    animator.animate_tumor_growth_isosurfaces()