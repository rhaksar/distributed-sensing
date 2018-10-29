from FireSimulator import FireSimulator

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
Given a forest state or state slice, create a visualization
'''
def PlotForest(state, grid_size):

    fig = pyplot.figure()
    ax = fig.add_subplot(111, aspect='equal')
    # pyplot.xlim([0, grid_size/2])
    # pyplot.ylim([0, grid_size/2])
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


def rc_to_xy(grid_size, rc):
    return col_to_x(rc[1]), row_to_y(grid_size, rc[0])


# def x_to_col(x):
#     return int(2*x)
#
#
# def y_to_row(grid_size, y):
#     return int(grid_size - 1 - 2*y)
#
#
# def xy_to_rc(grid_size, xy)
#     return y_to_row(grid_size, xy[1]), x_to_col(xy[0])


'''
Create a slice of the forest state, padded with zeros if out of bounds
Image is centered at "position"
'''
def CreateImage(state, position, dim):
    image = np.zeros(dim)

    half_row = (dim[0]-1)//2
    half_col = (dim[1]-1)//2

    for ri,dr in enumerate(np.arange(-half_row,half_row+1,1)):
        for ci,dc in enumerate(np.arange(-half_col,half_col+1,1)):
            r = position[0] + dr
            c = position[1] + dc

            if 0 <= r < grid_size and 0 <= c < grid_size:
                image[ri,ci] = state[r,c]

    return image


'''
Given an image, create an ordered lists of tasks (locations) and their weights
'''
def CreateTasks(image):
    cx, cy = col_to_x(image.shape[1]-1)/2 + 0.25, row_to_y(image.shape[0], 0)/2 + 0.25
    tasks = []
    neighbors = [(-1,0), (1,0), (0,-1), (0, 1)]

    expand_image = np.pad(image, ((1, 1), (1, 1)), 'constant', constant_values=((-1, -1), (-1, -1)))

    fire = np.where(expand_image == 1)
    if len(fire[0]) >= 1:
        for _, (r, c) in enumerate(zip(fire[0], fire[1])):
            x, y = col_to_x(c-1) + 0.25, row_to_y(image.shape[0], r-1) + 0.25

            weight = 0
            for (dr, dc) in neighbors:
                if expand_image[r + dr, c + dc] == 0:
                    weight += 1
                elif expand_image[r + dr, c + dc] == -1:
                    weight += 0.15

            if weight > 0:
                tasks.append([np.array([x, y]), weight])

    tasks_ordered = []
    for i in range(len(tasks)):
        if i == 0:
            p = np.array([cx, cy])
        else:
            p = tasks_ordered[-1][0]

        tasks.sort(key=lambda s: s[1]/np.maximum(np.linalg.norm(s[0]-p, ord=2), 1), reverse=True)
        tasks_ordered.append(tasks[0])
        del tasks[0]

    tasks_ordered = [[t[0]] for t in tasks_ordered]
    return tasks_ordered


if __name__ == "__main__":

    np.random.seed(42)

    agents = {1: {}}
    agents[1]['position'] = np.array([9.75, 10.75])
    # agents[1]['position'] = np.array([-0.25, -0.25])

    grid_size = 50
    sim = FireSimulator(grid_size)

    for i in range(15):
        sim.step([])

    # plot forest and agent position
    ax = PlotForest(sim.state, grid_size)
    ax.plot(agents[1]['position'][0], agents[1]['position'][1], Marker='.', MarkerSize=10, color='blue')

    # get agent image and plot
    r = int(-2*agents[1]['position'][1] + grid_size - 1 + 0.5)
    c = int(2*agents[1]['position'][0] - 0.5)

    image = CreateImage(sim.state, (r,c), (5,5))
    ax = PlotForest(image, 5)
    # ax.plot(1.25, 1.25, Marker='.', MarkerSize=20, color='blue')

    # test_image = copy.copy(image).astype('int32')
    # test_image[test_image == 2] = 1
    # dx = ndimage.sobel(test_image, 0)
    # dy = ndimage.sobel(test_image, 1)
    # magnitude = np.hypot(dx, dy)
    # magnitude *= 255.0 / np.max(magnitude)
    #
    # magnitude[magnitude < 50] = 0
    # magnitude[magnitude >= 50] = 1
    # ax = PlotForest(magnitude, 5)

    # r_fire, c_fire = np.where(image == 1)
    # r_burn, c_burn = np.where(image == 2)
    # targets = []
    # means = np.zeros((len(r_fire)+len(r_burn), 2))
    # covs = np.zeros((2, 2, len(r_fire)+len(r_burn)))
    #
    # for idx, (r, c) in enumerate(zip(r_fire, c_fire)):
    #     x = c * 0.5 + 0.25
    #     y = 5/2 - 0.5 * (r + 1) + 0.25
    #
    #     targets.append(np.array([x, y]))
    #
    #     means[idx, :] = [x, y]
    #     covs[:, :, idx] = 0.25*np.eye(2, 2)
    #
    # for idx, (r, c) in enumerate(zip(r_burn, c_burn)):
    #     x = c * 0.5 + 0.25
    #     y = 5/2 - 0.5 * (r + 1) + 0.25
    #
    #     means[len(r_fire)+idx, :] = [x, y]
    #     covs[:, :, len(r_fire)+idx] = 0.1*np.eye(2, 2)

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

    # get tasks from image
    tasks = CreateTasks(image)
    tasks = [t[0] for t in tasks]
    print(tasks)

    # solve convex program to generate path
    d = 0
    x_0 = np.array([1.25, 1.25])
    for idx, task in enumerate(tasks):
        if idx > 0:
            x_0 = next_x_0

        T = np.maximum(int(np.ceil((1/0.5)*np.linalg.norm(x_0 - task, ord=2))), 1)
        x = cvxpy.Variable((2, T+1))
        u = cvxpy.Variable((2, T))

        states = []
        for t in range(T):
            cost = cvxpy.maximum(cvxpy.norm(x[:, t+1]-task, p=2) - d, 0)
            constraints = [x[:, t+1] == x[:, t] + u[:, t],
                           cvxpy.norm(u[:, t], p=2) <= 0.5]

            states.append(cvxpy.Problem(cvxpy.Minimize(cost), constraints))

        constraints = [x[:, 0] == x_0, cvxpy.maximum(cvxpy.norm(x[:, T] - task, p=2) - d, 0) <= 0]
        states.append(cvxpy.Problem(cvxpy.Minimize(0), constraints))

        problem = cvxpy.sum(states)
        problem.solve()

        # print(problem.status)
        # print(T)
        # print(np.linalg.norm(u.value, axis=0))
        # print()

        next_x_0 = x.value[:, T]

        ax.plot(x.value[0, :], x.value[1, :], Marker='.', Markersize=10, color='white', zorder=1)

        # if np.linalg.norm(w.value[:, T-1]) > 0:
        #     ax.plot(x.value[0, T-1:], x.value[1, T-1:], Marker='.', Markersize=10, color='blue', zorder=2)
        #    # ax.plot(x.value[0, T], x.value[1, T], Marker='.', Markersize=10, color='blue', zorder=2)

    pyplot.show()

