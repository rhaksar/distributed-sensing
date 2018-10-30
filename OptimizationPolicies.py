from FireSimulator import FireSimulator

import copy
import cvxpy
import matplotlib.patches as patches
import matplotlib.pyplot as pyplot
import numpy as np
import operator
from scipy.stats import multivariate_normal
from scipy import ndimage

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

'''
Given a forest state or state slice, create a visualization
'''
def PlotForest(state):
    fig = pyplot.figure()
    ax = fig.add_subplot(111, aspect='equal')
    # pyplot.xlim([0, grid_size/2])
    # pyplot.ylim([0, grid_size/2])
    # pyplot.tick_params(axis='both', which='both',
    #                   labelbottom=False, labelleft=False,
    #                   bottom=False, left=False)

    for r in range(state.shape[0]):
        for c in range(state.shape[1]):
            x = col_to_x(c)
            y = row_to_y(state.shape[1], r)

            rec = patches.Rectangle((x, y), 0.5, 0.5, alpha=0.6)
            if state[r, c] == 0:
                rec.set_color('g')
            elif state[r, c] == 1:
                rec.set_color('r')
            elif state[r, c] == 2:
                rec.set_color('k')

            ax.add_patch(rec)

    return ax

def PlotForestImage(image, position):
    fig = pyplot.figure()
    ax = fig.add_subplot(111, aspect='equal')

    cx, cy = col_to_x(image.shape[1]-1)/2 + 0.25, row_to_y(image.shape[0], 0)/2 + 0.25

    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            x = col_to_x(c)
            y = row_to_y(image.shape[1], r)

            x += position[0] - cx
            y += position[1] - cy

            rec = patches.Rectangle((x, y), 0.5, 0.5, alpha=0.6)
            if image[r, c] == 0:
                rec.set_color('g')
            elif image[r, c] == 1:
                rec.set_color('r')
            elif image[r, c] == 2:
                rec.set_color('k')

            ax.add_patch(rec)

    return ax


def col_to_x(col):
    return 0.5*col


def row_to_y(grid_size, row):
    return 0.5*(grid_size-row-1)


def rc_to_xy(grid_size, rc):
    return col_to_x(rc[1]), row_to_y(grid_size, rc[0])


def x_to_col(x):
    return np.around(2*x, decimals=1).astype(np.int8)


def y_to_row(grid_size, y):
    return np.around(grid_size-1-2*y, decimals=1).astype(np.int8)


def xy_to_rc(grid_size, xy):
    return y_to_row(grid_size, xy[1]), x_to_col(xy[0])


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
def CreateTasks(image, location, memory):
    cx, cy = col_to_x(image.shape[1]-1)/2 + 0.25, row_to_y(image.shape[0], 0)/2 + 0.25
    tasks = []
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    expand_image = np.pad(image, ((1, 1), (1, 1)), 'constant', constant_values=((-1, -1), (-1, -1)))

    fire = np.where(expand_image == 1)
    if len(fire[0]) >= 1:
        for _, (r, c) in enumerate(zip(fire[0], fire[1])):
            x, y = col_to_x(c-1) + 0.25, row_to_y(image.shape[0], r-1) + 0.25
            x += location[0] - cx
            y += location[1] - cy

            weight = 0
            for (dr, dc) in neighbors:
                if expand_image[r + dr, c + dc] == 0:
                    weight += 1
                elif expand_image[r + dr, c + dc] == -1:
                    weight += 0.2

            task = np.around(np.array([x, y]), decimals=2)
            if weight > 0 and not any(np.array_equal(task, m) for m in memory):
                tasks.append([task, weight])

    tasks_ordered = []
    for i in range(len(tasks)):
        if i == 0:
            # p = np.array([cx, cy])
            p = location
        else:
            p = tasks_ordered[-1][0]

        # tasks.sort(key=lambda s: (np.linalg.norm(s[0]-p, ord=2), -s[1]))
        tasks.sort(key=lambda s: (s[1]/len(neighbors))-np.maximum(np.linalg.norm(s[0]-p, ord=2), 1), reverse=True)
        tasks_ordered.append(tasks[0])
        tasks = tasks[1:]

    # print(tasks_ordered)

    tasks_ordered = [t[0] for t in tasks_ordered]
    return tasks_ordered


def CreatePlan(tasks, initial_position):
    d = 0
    x0 = initial_position

    actions = []
    path = []
    for idx, task in enumerate(tasks):
        if idx > 0:
            x0 = next_x0

        T = np.maximum(int(np.ceil((1/0.5)*np.linalg.norm(x0-task, ord=2))), 1)
        x = cvxpy.Variable((2, T+1))
        u = cvxpy.Variable((2, T))

        states = []
        for t in range(T):
            cost = cvxpy.maximum(cvxpy.norm(x[:, t+1]-task, p=2)-d, 0)
            constraints = [x[:, t+1] == x[:, t] + u[:, t], cvxpy.norm(u[:, t], p=2) <= 0.5]
            states.append(cvxpy.Problem(cvxpy.Minimize(cost), constraints))

        cost = 0
        constraints = [x[:, 0] == x0, cvxpy.maximum(cvxpy.norm(x[:, T]-task, p=2)-d, 0) <= 0]
        states.append(cvxpy.Problem(cvxpy.Minimize(cost), constraints))

        problem = cvxpy.sum(states)
        problem.solve()

        next_x0 = x.value[:, T]
        path.append(x.value)
        actions.append(u.value)

    return actions, path


if __name__ == "__main__":

    np.random.seed(42)

    agents = {1: {}}
    agents[1]['position'] = np.array([9.75, 9.75])
    agents[1]['memory'] = []

    grid_size = 50
    sim = FireSimulator(grid_size)

    for i in range(15):
        sim.step([])

    # plot forest and agent position
    # ax = PlotForest(sim.state)
    # ax.plot(agents[1]['position'][0], agents[1]['position'][1], Marker='.', MarkerSize=10, color='blue')

    for iteration in range(30):
        print('iteration: %d' %iteration)
        # if iteration > 20:
        #     ax = PlotForest(sim.state)

        # get agent image and plot
        # r = int(-2*(agents[1]['position'][1]-0.25) + grid_size - 1)
        # c = int(2*(agents[1]['position'][0]-0.25))
        r, c = xy_to_rc(grid_size, agents[1]['position']-0.25)
        image = CreateImage(sim.state, (r, c), (5, 5))
        ax = PlotForestImage(image, agents[1]['position'])
        ax.plot(agents[1]['position'][0], agents[1]['position'][1], Marker='.', MarkerSize=10, color='blue')

        # get tasks from image, accounting for memory
        tasks = CreateTasks(image, agents[1]['position'], agents[1]['memory'])

        # solve convex program to generate path
        actions, path = CreatePlan(tasks, agents[1]['position'])
        agents[1]['position'] = path[0][:, -1]

        # if iteration > 20:
        ax.plot(path[0][0, :], path[0][1, :], Marker='.', MarkerSize=10, color='white')

        # add task to memory
        agents[1]['memory'].append(np.around(path[0][:, -1], decimals=2))

        # retain tasks still in view
        agents[1]['memory'] = [m for m in agents[1]['memory']
                               if -5/4 <= m[0]-agents[1]['position'][0] <= 5/4 and
                                  -5/4 <= m[1]-agents[1]['position'][1] <= 5/4]

        # print(agents[1]['memory'])

        pyplot.show()