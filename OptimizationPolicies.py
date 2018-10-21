from FireSimulator import *
import matplotlib.patches as patches
import matplotlib.pyplot as pyplot
import numpy as np


def plot_forest(fs):
    # Given a fire simulation, plot the current state

    grid_size = fs.grid_size
    fig = pyplot.figure()
    ax = fig.add_subplot(111, aspect='equal')
    pyplot.xlim([0, int(grid_size/2)+1])
    pyplot.ylim([0, int(grid_size/2)+1])
    pyplot.tick_params(axis='both', which='both',
                            labelbottom=False, labelleft=False,
                            bottom=False, left=False)

    for r in range(grid_size):
        for c in range(grid_size):
            x = c*0.5
            y = int(grid_size)/2 - 0.5*(r + 1)

            if fs.state[r, c] == 0:
                rec = patches.Rectangle((x+0.5, y+0.5), 0.5, 0.5, alpha=0.6, color='g')
                ax.add_patch(rec)
            elif fs.state[r, c] == 1:
                rec = patches.Rectangle((x+0.5, y+0.5), 0.5, 0.5, alpha=0.6, color='r')
                ax.add_patch(rec)
            elif fs.state[r, c] == 2:
                rec = patches.Rectangle((x+0.5, y+0.5), 0.5, 0.5, alpha=0.6, color='k')
                ax.add_patch(rec)

    return ax


if __name__ == "__main__":

    grid_size = 50
    sim = FireSimulator(grid_size)

    for i in range(25):
        sim.step([])

    plot_forest(sim)

    pyplot.show()
