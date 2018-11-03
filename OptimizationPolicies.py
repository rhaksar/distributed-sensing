from FireSimulator import FireSimulator

import copy
import cvxpy
import itertools
import matplotlib.collections as clt
import matplotlib.patches as patches
import matplotlib.pyplot as pyplot
import numpy as np
import scipy.optimize as spo
import time

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D


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
            y = row_to_y(state.shape[0], r)

            rec = patches.Rectangle((x, y), 0.5, 0.5, alpha=0.6)
            if state[r, c] == 0:
                rec.set_color('green')
            elif state[r, c] == 1:
                rec.set_color('red')
            elif state[r, c] == 2:
                rec.set_color('black')
            else:
                rec.set_color('gray')

            ax.add_patch(rec)

    return ax


def PlotForestImage(image, lower_left_corner):
    fig = pyplot.figure()
    ax = fig.add_subplot(111, aspect='equal')

    # cx, cy = col_to_x(image.shape[1]-1)/2 + 0.25, row_to_y(image.shape[0], 0)/2 + 0.25

    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            x = col_to_x(c)
            y = row_to_y(image.shape[0], r)

            x += lower_left_corner[0] # x += position[0] - cx
            y += lower_left_corner[1] # y += position[1] - cy

            rec = patches.Rectangle((x, y), 0.5, 0.5, alpha=0.6)
            if image[r, c] == 0:
                rec.set_color('green')
            elif image[r, c] == 1:
                rec.set_color('red')
            elif image[r, c] == 2:
                rec.set_color('black')
            else:
                rec.set_color('gray')

            ax.add_patch(rec)

    return ax


def col_to_x(col):
    return 0.5*col


def row_to_y(y_limit, row):
    return 0.5*(y_limit-row-1)


def rc_to_xy(y_limit, rc):
    return col_to_x(rc[1]), row_to_y(y_limit, rc[0])


def x_to_col(x):
    return np.rint(2*x).astype(np.int8)


def y_to_row(y_limit, y):
    return np.rint(y_limit-1-2*y).astype(np.int8)


def xy_to_rc(y_limit, xy):
    return y_to_row(y_limit, xy[1]), x_to_col(xy[0])


'''
Create a slice of the forest state, padded with zeros if out of bounds
Image is centered at "position"
'''
def CreateSoloImage(state, position, dim):
    r0, c0 = xy_to_rc(state.shape[0], position)
    image = np.zeros(dim).astype(np.int8)

    half_row = (dim[0]-1)//2
    half_col = (dim[1]-1)//2

    for ri, dr in enumerate(np.arange(-half_row, half_row+1, 1)):
        for ci, dc in enumerate(np.arange(-half_col, half_col+1, 1)):
            r = r0 + dr
            c = c0 + dc

            if 0 <= r < state.shape[0] and 0 <= c < state.shape[1]:
                image[ri, ci] = state[r, c]

    return image, rc_to_xy(state.shape[0], (r0+half_row, c0-half_row))


def CreateJointImage(state, positions, dim):
    rows = []
    cols = []
    rowcol = []
    for idx, pos in enumerate(positions):
        r, c = xy_to_rc(state.shape[0], pos)
        rowcol.append((r, c))
        rows.append(r-(dim[0]-1)//2)
        rows.append(r+(dim[0]-1)//2)
        cols.append(c-(dim[1]-1)//2)
        cols.append(c+(dim[1]-1)//2)

    min_r, max_r = np.amin(rows), np.amax(rows)
    min_c, max_c = np.amin(cols), np.amax(cols)
    min_x, min_y = rc_to_xy(state.shape[0], (max_r, min_c))

    image = -1*np.ones((max_r-min_r+1, max_c-min_c+1)).astype(np.int8)

    half_row = (dim[0]-1)//2
    half_col = (dim[1]-1)//2
    for r0, c0 in rowcol:
        for dr in np.arange(-half_row, half_row+1, 1):
            for dc in np.arange(-half_col, half_col+1, 1):
                ri = r0 - min_r + dr
                ci = c0 - min_c + dc

                r = r0 + dr
                c = c0 + dc

                if 0 <= r < state.shape[0] and 0 <= c < state.shape[1]:
                    image[ri, ci] = state[r, c]
                else:
                    image[ri, ci] = 0

    return image, (min_x, min_y)


'''
Given an image, create an ordered lists of tasks (locations) with weights
'''
def CreateSoloTasks(image, lower_left_corner, location, memory, default):
    # cx, cy = col_to_x(image.shape[1]-1)/2 + 0.25, row_to_y(image.shape[0], 0)/2 + 0.25
    tasks = []
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    expand_image = np.pad(image, ((1, 1), (1, 1)), 'constant', constant_values=((-1, -1), (-1, -1)))

    # if fires: tasks are (boundary) fires
    # if no fires and some/all burnt: tasks are (boundry) burnt trees
    # if no fires and no burnt: task is "go to default"
    fire = np.where(expand_image == 1)
    burnt = np.where(expand_image == 2)
    if len(fire[0]) >= 1:
        r_task, c_task = fire[0], fire[1]
    elif len(burnt[0] >= 1):
        r_task, c_task = burnt[0], fire[1]
    else:
        r_task, c_task = None, None

    if r_task is not None:
        for _, (r, c) in enumerate(zip(r_task, c_task)):
            # "c-1" and "r-1" are due to padding
            # "+ 0.25" is due to cell width/height
            x, y = col_to_x(c-1)+0.25, row_to_y(image.shape[0], r-1)+0.25
            x += lower_left_corner[0] # location[0] - cx
            y += lower_left_corner[1] # location[1] - cy

            weight = 0
            for (dr, dc) in neighbors:
                if expand_image[r+dr, c+dc] == 0:
                    weight += 1
                elif expand_image[r+dr, c+dc] == -1:
                    weight += 0.35

            task = np.around(np.array([x, y]), decimals=2)
            if weight > 0 and not any(np.array_equal(task, m) for m in memory):
                tasks.append([task, weight])

    if not tasks and burnt[0]:
        for _, (r, c) in enumerate(zip(burnt[0], burnt[1])):
            # "c-1" and "r-1" are due to padding
            # "+ 0.25" is due to cell width/height
            x, y = col_to_x(c-1)+0.25, row_to_y(image.shape[0], r-1)+0.25
            x += lower_left_corner[0] # location[0] - cx
            y += lower_left_corner[1] # location[1] - cy

            weight = 0
            for (dr, dc) in neighbors:
                if expand_image[r+dr, c+dc] == 0:
                    weight += 1
                elif expand_image[r+dr, c+dc] == -1:
                    weight += 0.35
            weight *= 0.5

            task = np.around(np.array([x, y]), decimals=2)
            if weight > 0 and not any(np.array_equal(task, m) for m in memory):
                tasks.append([task, weight])

    # all healthy: task is "go to default"
    if not tasks:
        if int(np.sum(image)) == 0:
            center_vector = 1*(default-location) / np.linalg.norm(default-location, ord=2)
            tasks.append([location+center_vector, 1])

    tasks_ordered = []
    for i in range(len(tasks)):
        if i == 0:
            p = location
        else:
            p = tasks_ordered[-1][0]

        # tasks.sort(key=lambda s: (np.linalg.norm(s[0]-p, ord=2), -s[1]))
        tasks.sort(key=lambda s: s[1]-np.linalg.norm(s[0]-p, ord=2), reverse=True)
        tasks_ordered.append(tasks[0])
        tasks = tasks[1:]

    # print(tasks_ordered)

    tasks_ordered = [t[0] for t in tasks_ordered]
    return tasks_ordered


def CreateJointTasks(joint_image, lower_left_corner, locations, joint_memory, default):
    tasks = []
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    expand_image = np.pad(joint_image, ((1, 1), (1, 1)), 'constant', constant_values=((-1, -1), (-1, -1)))

    fire = np.where(expand_image == 1)
    burnt = np.where(expand_image == 2)
    if len(fire[0]) >= 1:
        r_task, c_task = fire[0], fire[1]
    elif len(burnt[0] >= 1):
        r_task, c_task = burnt[0], fire[1]
    else:
        r_task, c_task = None, None

    if r_task is not None:
        for _, (r, c) in enumerate(zip(r_task, c_task)):
            # "c-1" and "r-1" are due to padding
            # "+ 0.25" is due to cell width/height
            x, y = col_to_x(c-1)+0.25, row_to_y(joint_image.shape[0], r-1)+0.25
            x += lower_left_corner[0]  # location[0] - cx
            y += lower_left_corner[1]  # location[1] - cy

            weight = 0
            for (dr, dc) in neighbors:
                if expand_image[r+dr, c+dc] == 0:
                    weight += 1
                elif expand_image[r+dr, c+dc] == -1:
                    weight += 0.35

            task = np.around(np.array([x, y]), decimals=2)
            if weight > 0 and not any(np.array_equal(task, m) for m in joint_memory):
                tasks.append([task, weight])

    if len(tasks) < locations.shape[0] and burnt[0] is not None:
        for _, (r, c) in enumerate(zip(burnt[0], burnt[1])):
            x, y = col_to_x(c-1)+0.25, row_to_y(image.shape[0], r-1)+0.25
            x += lower_left_corner[0]
            y += lower_left_corner[1]

            weight = 0
            for (dr, dc) in neighbors:
                if expand_image[r+dr, c+dc] == 0:
                    weight += 1
                elif expand_image[r+dr, c+dc] == -1:
                    weight += 0.35
            weight *= 0.5

            task = np.around(np.array([x, y]), decimals=2)
            if weight > 0 and not any(np.array_equal(task, m) for m in joint_memory):
                    tasks.append([task, weight])

    if len(tasks) < locations.shape[0]:
        for _ in range(locations.shape[0]-len(tasks)):
            # center_vector = 1*(default-location) / np.linalg.norm(default-location, ord=2)
            # tasks.append([location+center_vector, 1])
            tasks.append([default, 1])

    cost_matrix = np.zeros((locations.shape[0], len(tasks)))
    for agent in range(locations.shape[0]):
        cost_matrix[agent, :] = -1*np.array([s[1]-np.linalg.norm(s[0]-locations[agent, :], ord=2) for s in tasks])

    # print(np.around(cost_matrix, decimals=3))

    _, assignments = spo.linear_sum_assignment(cost_matrix)
    # print(assignments)

    tasks = [t[0] for t in tasks]

    return tasks, assignments


def CreateSoloPlan(tasks, initial_position):
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

        constraints = [x[:, 0] == x0, cvxpy.maximum(cvxpy.norm(x[:, T]-task, p=2)-d, 0) <= 0]
        states.append(cvxpy.Problem(cvxpy.Minimize(0), constraints))

        problem = cvxpy.sum(states)
        problem.solve()

        next_x0 = x.value[:, T]
        path.append(x.value)
        actions.append(u.value)

    return actions, path


def CreateJointPlan(tasks, assignments, initial_positions):
    actions = []
    path = []

    num_agents = initial_positions.shape[0]
    x0 = initial_positions

    safe_radius = 0.3
    scp_iterations = 5

    T = 5
    nominal_paths = np.zeros((2*num_agents, T+1))
    nominal_actions = None

    for _ in range(scp_iterations):
        states = []

        x = cvxpy.Variable((2*num_agents, T+1))
        u = cvxpy.Variable((2*num_agents, T))

        for ai in range(num_agents):
            for t in range(T+1):
                cost = 0
                constraints = []
                if t < T:
                    cost = cvxpy.norm(x[2*ai:(2*ai+2), t+1]-tasks[assignments[ai]], p=2)
                    constraints = [x[2*ai:(2*ai+2), t+1] == x[2*ai:(2*ai+2), t] + u[2*ai:(2*ai+2), t],
                                   cvxpy.norm(u[2*ai:(2*ai+2), t], p=2) <= 0.5]

                # collision avoidance constraint for time interval [1, T]
                if t > 0 and ai < num_agents-1:
                    for aj in np.arange(ai+1, num_agents, 1):
                        xi = nominal_paths[2*ai:(2*ai+2), t]
                        xj = nominal_paths[2*aj:(2*aj+2), t]
                        x_diff = xj-xi
                        x_diff_norm = np.linalg.norm(x_diff, ord=2)
                        if x_diff_norm <= 5*safe_radius:
                            constraints.append(x_diff[0]*(x[2*aj, t]-x[2*ai, t]) + x_diff[1]*(x[2*aj+1, t]-x[2*ai+1, t])
                                               >= safe_radius*x_diff_norm)

                states.append(cvxpy.Problem(cvxpy.Minimize(cost), constraints))

            # constraints = [x[2*ai:(2*ai+2), 0] == x0[ai, :],
            #                cvxpy.norm(x[2*ai:(2*ai+2), T]-tasks[assignments[ai]], p=2) <= 0]
            constraints = [x[2*ai:(2*ai+2), 0] == x0[ai, :]]
            cost = cvxpy.norm(x[2*ai:(2*ai+2), T]-tasks[assignments[ai]], p=2)
            states.append(cvxpy.Problem(cvxpy.Minimize(cost), constraints))

        problem = cvxpy.sum(states)
        problem.solve()

        nominal_paths = x.value
        nominal_actions = u.value

    path.append(nominal_paths)
    actions.append(nominal_actions)

    return actions, path


def SingleAgentExample():
    agent_position = np.array([9.25, 9.25])
    agent_memory = []

    grid_size = 50
    center = np.array([12.5, 12.5])
    sim = FireSimulator(grid_size)

    for i in range(15):
        sim.step([])

    # plot forest and agent position
    ax = PlotForest(sim.state)
    ax.plot(agent_position[0], agent_position[1], linestyle='', Marker='.', MarkerSize=10, color='blue')

    for iteration in range(15):
        print('iteration: %d' %(iteration+1))
        # ax = PlotForest(sim.state)

        # get agent image and plot
        # r = int(-2*(agents[1]['position'][1]-0.25) + grid_size - 1)
        # c = int(2*(agents[1]['position'][0]-0.25))
        # r, c = xy_to_rc(grid_size, agents[1]['position']-0.25)
        image, corner = CreateSoloImage(sim.state, agent_position-0.25, (5, 5))
        ax = PlotForestImage(image, corner)
        ax.plot(agent_position[0], agent_position[1], linestyle='', Marker='.', MarkerSize=10, color='blue')

        # get tasks from image, accounting for memory
        tasks = CreateSoloTasks(image, corner, agent_position, agent_memory, center)

        # solve convex program to generate path
        _, path = CreateSoloPlan(tasks, agent_position)
        # agent_position = path[0][:, -1]
        agent_position = path[0][:, 1]

        ax.plot(path[0][0, :], path[0][1, :], Marker='.', MarkerSize=10, color='white')
        ax.plot(path[0][0, 0], path[0][1, 0], Marker='.', MarkerSize=15, color='blue')

        # add task to memory if agent completed task
        if np.linalg.norm(agent_position - path[0][:, -1], ord=2) < 0.01:
            agent_memory.append(np.around(path[0][:, -1], decimals=2))

        # retain tasks still in view
        agent_memory = [m for m in agent_memory if -5/4 <= m[0]-agent_position[0] <= 5/4 and
                                                   -5/4 <= m[1]-agent_position[1] <= 5/4]

        # print(agent_memory)

        pyplot.show()


def MultiAgentExample():
    agents = {'position': np.zeros((3, 2)), 'memory': [[], [], []]}
    agents['position'][0, :] = np.array([8.25, 11.25])  # np.array([14.25, 16.25])
    agents['position'][1, :] = np.array([8.75, 10.75])  # np.array([15.75, 15.75])
    agents['position'][2, :] = np.array([8.25, 10.75])  # np.array([16.75, 14.25])

    grid_size = 50
    center = np.array([12.5, 12.5])
    sim = FireSimulator(grid_size)

    for i in range(15):
        sim.step([])

    # plot forest and agent positions
    ax = PlotForest(sim.state)
    ax.plot(agents['position'][:, 0], agents['position'][:, 1], linestyle='', Marker='.', MarkerSize=10, color='blue')

    cooperating_agents = np.array([0, 1, 2])
    for iteration in range(10):
        print('iteration: %d' % (iteration + 1))

        # get joint image and plot
        image, corner = CreateJointImage(sim.state, agents['position'][cooperating_agents, :]-0.25, (5, 5))
        ax = PlotForestImage(image, corner)
        ax.plot(agents['position'][cooperating_agents, 0], agents['position'][cooperating_agents, 1],
                linestyle='', Marker='.', MarkerSize=10, color='blue')

        # get tasks from image, accounting for joint memory
        joint_memory = list(itertools.chain.from_iterable(agents['memory']))
        tasks, assignments = CreateJointTasks(image, corner, agents['position'][cooperating_agents, :],
                                              joint_memory, center)

        # solve convex program for paths and determine completed tasks
        _, paths = CreateJointPlan(tasks, assignments, agents['position'][cooperating_agents, :])
        completed = []
        for idx, a in enumerate(cooperating_agents):
            ax.plot(paths[0][2*idx, :], paths[0][2*idx+1, :], Marker='.', MarkerSize=10, color='white')
            ax.plot(agents['position'][a, 0], agents['position'][a, 1], Marker='.', MarkerSize=10, color='blue')

            # update position by taking first action
            agents['position'][a, :] = paths[0][2*idx:(2*idx+2), 1]
            if np.linalg.norm(paths[0][2*idx:(2*idx+2), -1]-agents['position'][a, :], ord=2) <= 0.01:
                completed.append(np.around(paths[0][2*idx:(2*idx+2), -1], decimals=2))

        # add completed tasks to memory and retain tasks still in view
        # completed = [np.around(paths[0][2*idx:(2*idx+2), -1], decimals=2) for idx in range(len(cooperating_agents))]
        for a in cooperating_agents:
            agents['memory'][a].extend(completed)

            agents['memory'][a] = [m for m in agents['memory'][a]
                                   if -5/4 <= m[0]-agents['position'][a, 0] <= 5/4 and
                                      -5/4 <= m[1]-agents['position'][a, 1] <= 5/4]

        pyplot.show()

if __name__ == "__main__":

    np.random.seed(42)

    # SingleAgentExample()
    # MultiAgentExample()

    # centralized planner example

    # x = np.linspace(0, 6 * np.pi, 100)
    # y = np.sin(x)

    # You probably won't need this if you're embedding things in a tkinter plot...
    # pyplot.ion()
    #
    # line1, = ax.plot(x, y, 'r-')  # Returns a tuple of line objects, thus the comma
    #
    # for phase in np.linspace(0, 10 * np.pi, 500):
    #     line1.set_ydata(np.sin(x + phase))
    #     fig.canvas.draw()
    #     fig.canvas.flush_events()

    agents = {'position': np.zeros((3, 2))}
    agents['position'][0, :] = np.array([0.75, 0.75])
    agents['position'][1, :] = np.array([0.75, 1.25])
    agents['position'][2, :] = np.array([1.25, 0.75])
    # agents['position'][3, :] = np.array([0.25, 0.75])
    # agents['position'][4, :] = np.array([0.75, 0.25])

    # pyplot.ion()

    fig = pyplot.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 25)

    grid_size = 50
    sim = FireSimulator(grid_size)
    sim.step([])

    center = np.array([12.5, 12.5])

    tree_patch_map = {}
    trees = []
    for r in range(sim.state.shape[0]):
        for c in range(sim.state.shape[1]):
            x = col_to_x(c)
            y = row_to_y(sim.state.shape[0], r)

            tree_patch_map[(r, c)] = len(tree_patch_map)
            trees.append(patches.Rectangle((x, y), 0.5, 0.5, alpha=0.6, zorder=0))

    for tree in trees:
        ax.add_artist(tree)

    for r in range(sim.state.shape[0]):
        for c in range(sim.state.shape[1]):
            idx = tree_patch_map[(r, c)]
            if sim.state[r, c] == 0:
                trees[idx].set_color('green')
            elif sim.state[r, c] == 1:
                trees[idx].set_color('red')
            elif sim.state[r, c] == 2:
                trees[idx].set_color('black')

    agent_viz = [None for a in range(agents['position'].shape[0])]
    agent_plan_viz = [None for a in range(agents['position'].shape[0])]

    # ax.figure.canvas.draw()
    # time.sleep(1)

    joint_memory = []
    action = []
    image = copy.copy(sim.state)  # image is entire forest

    for iteration in range(35):
        print('Iteration: %d' % (iteration+1))

        if (iteration+1) % 5 == 0:
            sim.step(action, dbeta=0.54)
            joint_memory = []
            action = []
            image = copy.copy(sim.state)

            for r in range(sim.state.shape[0]):
                for c in range(sim.state.shape[1]):
                    idx = tree_patch_map[(r, c)]
                    if sim.state[r, c] == 0:
                        trees[idx].set_color('green')
                    elif sim.state[r, c] == 1:
                        trees[idx].set_color('red')
                    elif sim.state[r, c] == 2:
                        trees[idx].set_color('black')

            print('Number of fires: %d' % len(sim.fires))

        # if (iteration+1) >= 30:
        #     ax = PlotForestImage(image, (0, 0))
        #     ax.plot(agents['position'][:, 0], agents['position'][:, 1],
        #             linestyle='', Marker='.', MarkerSize=10, color='blue')

        tasks, assignments = CreateJointTasks(image, (0, 0), agents['position'], joint_memory, center)

        _, paths = CreateJointPlan(tasks, assignments, agents['position'])
        completed = []
        for a in range(agents['position'].shape[0]):
            # if (iteration+1) >= 30:
            #     ax.plot(paths[0][2*a, :], paths[0][2*a+1, :], Marker='.', MarkerSize=10, color='white')
            #     ax.plot(agents['position'][a, 0], agents['position'][a, 1], Marker='.', MarkerSize=10, color='blue')
            if agent_viz[a] is None:
                agent_viz[a], = ax.plot(agents['position'][a, 0], agents['position'][a, 1],
                                        linestyle='', Marker='.', MarkerSize=10, color='blue', zorder=2)
            else:
                agent_viz[a].set_data(agents['position'][a, 0], agents['position'][a, 1])

            if agent_plan_viz[a] is None:
                agent_plan_viz[a], = ax.plot(paths[0][2*a, :], paths[0][2*a+1, :],
                                             Marker='.', MarkerSize=10, color='white', zorder=1)
            else:
                agent_plan_viz[a].set_data(paths[0][2*a, :], paths[0][2*a+1, :])

            agents['position'][a, :] = paths[0][2*a:(2*a+2), 1]
            if np.linalg.norm(paths[0][2*a:(2*a+2), -1]-agents['position'][a, :], ord=2) <= 0.01:
                completed.append(np.around(paths[0][2*a:(2*a+2), -1], decimals=2))

                # convert completed task to action
                r, c = xy_to_rc(grid_size, paths[0][2*a:(2*a+2), -1]-0.25)
                x = c+1
                y = grid_size-r
                if tuple((x, y)) not in action:
                    action.append(tuple((x, y)))

        joint_memory.extend(completed)

        ax.figure.canvas.draw()
        time.sleep(0.5)

        # filename = 'iteration' + str(iteration+1) + '.png'
        # pyplot.savefig(filename, bbox_inches='tight')

