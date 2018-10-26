from FireSimulator import *

import copy
import cvxpy
import matplotlib.patches as patches
import matplotlib.pyplot as pyplot
import numpy as np
from scipy.stats import multivariate_normal
from scipy import ndimage

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

'''
Given a fire simulation object, plot the current state
'''
def plot_forest(state, grid_size):

    fig = pyplot.figure()
    ax = fig.add_subplot(111, aspect='equal')
    pyplot.xlim([0, grid_size/2])
    pyplot.ylim([0, grid_size/2])
    # pyplot.tick_params(axis='both', which='both',
    #                   labelbottom=False, labelleft=False,
    #                   bottom=False, left=False)

    for r in range(grid_size):
        for c in range(grid_size):
            x = col_to_x(c)
            y = row_to_y(grid_size, r)

            rec = patches.Rectangle((x, y), 0.5, 0.5, alpha=0.6)
            if state[r, c] == 0:
                rec.set_color('g')
            elif state[r, c] == 1:
                rec.set_color('r')
            elif state[r, c] == 2:
                rec.set_color('k')

            ax.add_patch(rec)

    return ax


def col_to_x(col):
    return 0.5*col


def row_to_y(grid_size, row):
    return 0.5*(grid_size - row - 1)


if __name__ == "__main__":

    np.random.seed(42)

    agents = {1: {}}
    agents[1]['position'] = np.array([9.75, 14.75])

    grid_size = 50
    sim = FireSimulator(grid_size)

    for i in range(15):
        sim.step([])

    # plot forest
    ax = plot_forest(sim.state, grid_size)
    ax.plot(agents[1]['position'][0], agents[1]['position'][1], Marker='.', MarkerSize=10, color='blue')

    # get image and plot
    r = int(-2*agents[1]['position'][1] + grid_size - 1 + 0.5)
    c = int(2*agents[1]['position'][0] - 0.5)

    image = sim.state[r-2:r+3, c-2:c+3]
    ax = plot_forest(image, 5)
    # ax.plot(1.25, 1.25, Marker='.', MarkerSize=20, color='blue')

    # test_image = copy.copy(image)
    # test_image[test_image == 2] = 1
    # dx = ndimage.sobel(test_image, 0)
    # dy = ndimage.sobel(test_image, 1)
    # magnitude = np.hypot(dx, dy)
    # magnitude *= 255.0 / np.max(magnitude)
    #
    # magnitude[magnitude < 50] = 0
    # magnitude[magnitude >= 50] = 1
    # ax = plot_forest(magnitude, 6)

    r_fire, c_fire = np.where(image == 1)
    r_burn, c_burn = np.where(image == 2)
    targets = []
    means = np.zeros((len(r_fire)+len(r_burn), 2))
    covs = np.zeros((2, 2, len(r_fire)+len(r_burn)))

    for idx, (r, c) in enumerate(zip(r_fire, c_fire)):
        x = c * 0.5 + 0.25
        y = 5/2 - 0.5 * (r + 1) + 0.25

        targets.append(np.array([x, y]))

        means[idx, :] = [x, y]
        covs[:, :, idx] = 0.25*np.eye(2, 2)

    for idx, (r, c) in enumerate(zip(r_burn, c_burn)):
        x = c * 0.5 + 0.25
        y = 5/2 - 0.5 * (r + 1) + 0.25

        means[len(r_fire)+idx, :] = [x, y]
        covs[:, :, len(r_fire)+idx] = 0.1*np.eye(2, 2)

    # create and plot heat map
    # x = np.linspace(0, 4, num=100, endpoint=True)
    # y = np.linspace(0, 4, num=100, endpoint=True)
    # X, Y = np.meshgrid(x, y)
    # Z = np.zeros_like(X)
    # pos = np.dstack((X, Y))

    # Z = np.zeros_like(X)
    # for r in range(X.shape[0]):
    #     for c in range(X.shape[1]):
    #
    #         point = np.array([X[r, c], Y[r, c]])
    #         for idx in range(len(r_fire)):
    #
    #             vector = point - means[idx, :]
    #             result = -0.5*np.matmul(np.matmul(vector.transpose(), np.linalg.inv(covs[:, :, idx])), vector)
    #             result += np.log(1/np.sqrt((2*np.pi)**2*np.linalg.det(covs[:, :, i])))
    #             Z[r, c] += np.exp(result)

    # for i in range(len(r_fire)):
    #     Z += multivariate_normal.pdf(pos, mean=means[i, :], cov=covs[:, :, i])
    # Z = multivariate_normal.pdf(pos, mean=means, cov=covs)
    # Z /= np.linalg.norm(Z)

    # fig = pyplot.figure()
    # ax = fig.add_subplot(121, aspect='equal')
    # ax.pcolor(X, Y, Z)
    # ax = fig.add_subplot(122, aspect='equal', projection='3d')
    # ax.scatter(X, Y, Z)

    d = 0
    local_position = np.array([1.25, 1.25])

    task_order = []
    tasks = copy.copy(targets)
    for i in range(len(targets)):
        if i == 0:
            p = local_position
        else:
            p = task_order[-1]

        tasks.sort(key=lambda s: np.linalg.norm(s-p, ord=1))
        task_order.append(tasks[0])
        del tasks[0]

    for idx, task in enumerate(task_order):
        if idx == 0:
            x_0 = local_position
        else:
            x_0 = next_x_0

        T = int(np.ceil((1/0.5)*np.linalg.norm(x_0 - task, ord=2)))
        x = cvxpy.Variable((2, T+1))
        u = cvxpy.Variable((2, T))
        w = cvxpy.Variable((2, T))

        states = []
        for t in range(T):
            cost = cvxpy.maximum(cvxpy.norm(x[:, t+1]-task, p=2) - d, 0)
            constraints = [x[:, t+1] == x[:, t] + u[:, t] + w[:, t],
                           cvxpy.norm(u[:, t], p=2) <= 0.5]
            if t < T-1:
                constraints.append(w[:, t] == 0)
            states.append(cvxpy.Problem(cvxpy.Minimize(cost), constraints))

        constraints = [cvxpy.maximum(cvxpy.norm(x[:, T] - task, p=1) - d, 0) <= 0,
                       x[:, 0] == x_0, u[:, T-1] == 0]
        states.append(cvxpy.Problem(cvxpy.Minimize(0), constraints))

        problem = cvxpy.sum(states)
        problem.solve()

        next_x_0 = x.value[:, T]

        ax.plot(x.value[0, :], x.value[1, :], Marker='.', Markersize=10, color='white')

    pyplot.show()

