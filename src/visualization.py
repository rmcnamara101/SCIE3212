import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate_simulation(x, history, Nt):
    """ Creates an animation of the simulation. """
    fig, ax = plt.subplots()
    line1, = ax.plot([], [], label="Stem Cells (CSC)")
    line2, = ax.plot([], [], label="Progenitor Cells (CP)")
    line3, = ax.plot([], [], label="Differentiated Cells (TD)")
    line4, = ax.plot([], [], label="Nutrient", linestyle="dashed")
    
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, 1.2)
    ax.legend()

    def update(frame):
        line1.set_data(x, history["C1"][frame])
        line2.set_data(x, history["C2"][frame])
        line3.set_data(x, history["C3"][frame])
        return line1, line2, line3, line4

    ani = animation.FuncAnimation(fig, update, frames=Nt, interval=50)
    plt.show()
