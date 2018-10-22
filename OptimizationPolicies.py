from FireSimulator import *

import cvxpy
import matplotlib.patches as patches
import matplotlib.pyplot as pyplot
import numpy as np

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

'''
Given a fire simulation object, plot the current state
'''
def plot_forest(state, grid_size):

    fig = pyplot.figure()
    ax = fig.add_subplot(111, aspect='equal')
    pyplot.xlim([0, int(grid_size/2)+1])
    pyplot.ylim([0, int(grid_size/2)+1])
    # pyplot.tick_params(axis='both', which='both',
    #                   labelbottom=False, labelleft=False,
    #                   bottom=False, left=False)

    for r in range(grid_size):
        for c in range(grid_size):
            x = c*0.5
            y = int(grid_size)/2 - 0.5*(r + 1)

            if state[r, c] == 0:
                rec = patches.Rectangle((x+0.5, y+0.5, 0), 0.5, 0.5, alpha=0.6, color='g')
            elif state[r, c] == 1:
                rec = patches.Rectangle((x+0.5, y+0.5, 0), 0.5, 0.5, alpha=0.6, color='r')
            elif state[r, c] == 2:
                rec = patches.Rectangle((x+0.5, y+0.5, 0), 0.5, 0.5, alpha=0.6, color='k')

            ax.add_patch(rec)

    return ax


if __name__ == "__main__":

    agents = {}
    agents[1] = {}
    agents[1]['position'] = np.array([10, 10])

    np.random.seed(42)

    grid_size = 50
    sim = FireSimulator(grid_size)

    for i in range(15):
        sim.step([])

    # plot forest
    ax = plot_forest(sim.state, grid_size)
    ax.plot(agents[1]['position'][0], agents[1]['position'][1], Marker='.', MarkerSize=10, color='blue')

    # get image and plot
    r = -2*agents[1]['position'][0] + grid_size - 1
    c = 2*agents[1]['position'][1]

    image = sim.state[r-3:r+3, c-3:c+3]
    ax = plot_forest(image, 6)
    ax.plot(2, 2, Marker='.', MarkerSize=20, color='blue')

    r_fire, c_fire = np.where(image == 1)
    r_burn, c_burn = np.where(image == 2)
    means = np.zeros((len(r_fire)+len(r_burn), 2))
    covs = np.zeros((2, 2, len(r_fire)+len(r_burn)))

    for idx, (r, c) in enumerate(zip(r_fire, c_fire)):
        x = c * 0.5 + 0.75
        y = 6 / 2 - 0.5 * (r + 1) + 0.75

        means[idx, :] = [x, y]
        covs[:, :, idx] = 1*np.eye(2, 2)

    for idx, (r, c) in enumerate(zip(r_burn, c_burn)):
        x = c * 0.5 + 0.75
        y = 6/2 - 0.5 * (r + 1) + 0.75

        means[len(r_fire)+idx, :] = [x, y]
        covs[:, :, len(r_fire)+idx] = 1*np.eye(2, 2)

    x = np.linspace(0, 4, num=100, endpoint=True)
    y = np.linspace(0, 4, num=100, endpoint=True)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for r in range(X.shape[0]):
        for c in range(X.shape[1]):

            point = np.array([X[r, c], Y[r, c]])
            for idx in range(len(r_fire)+len(r_burn)):

                vector = point - means[idx, :]
                Z[r, c] += -0.5*np.matmul(np.matmul(vector.transpose(), np.linalg.inv(covs[:, :, idx])), vector)
                #Z[r, c] += np.log(1/np.sqrt(2*np.pi*np.linalg.det(covs[:, :, i])))

            # Z[r, c] = np.exp(Z[r, c] + 25)

    fig = pyplot.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.pcolor(X, Y, Z)

    pyplot.show()

