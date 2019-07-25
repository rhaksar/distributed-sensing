from collections import defaultdict
import copy
import cvxpy
import matplotlib.patches as patches
import matplotlib.pyplot as pyplot
import numpy as np
import os
from queue import PriorityQueue
import scipy.ndimage as sn
import scipy.stats as ss
import sys

base_path = os.path.dirname(os.getcwd())
sys.path.insert(0, base_path + '/simulators')

from filter import ApproxFilter, multiply_probabilities
from fires.UrbanForest import UrbanForest
from fires.ForestElements import Tree, SimpleUrban


def observation_probability(element, state, observation):
    measure_correct = 0.8
    measure_wrong = None
    if isinstance(element, Tree):
        measure_wrong = 0.5*(1-measure_correct)
    elif isinstance(element, SimpleUrban):
        measure_wrong = (1/3)*(1-measure_correct)

    if state != observation:
        return measure_wrong
    elif state == observation:
        return measure_correct


def filter_element_update(key, element, prior, advance, control, observation):
    posterior = np.zeros(len(element.state_space))
    num_neighbors = len(element.neighbors)
    on_fire = 1

    caf = np.zeros(num_neighbors+1)
    for l in range(2**num_neighbors):
        xj = np.base_repr(l, base=2).zfill(num_neighbors)
        active = xj.count('1')

        values = []
        for n in range(num_neighbors):
            neighbor_key = element.neighbors[n]
            prob = None
            if int(xj[n]) == 0:
                prob = 1 - prior[neighbor_key][on_fire]
            elif int(xj[n]) == 1:
                prob = prior[neighbor_key][on_fire]

            values.append(prob)

        caf[active] += multiply_probabilities(values)

    for s_t in element.state_space:
        for s_tm1 in element.state_space:
            if advance:
                for active in range(num_neighbors+1):
                    values = [element.dynamics((s_tm1, active, s_t), control[key]), caf[active], prior[key][s_tm1]]
                    posterior[s_t] += multiply_probabilities(values)
            else:
                posterior[s_t] += (s_t == s_tm1)*prior[key][s_tm1]

        if key in observation.keys():
            posterior[s_t] *= observation_probability(element, s_t, observation[key])
        else:
            coeff = 0.01
            mean = np.mean(posterior)
            posterior = [v - coeff*np.sign(v-mean)*np.abs(v-mean) for v in posterior]

    # for s_t in element.state_space:
    #     for s_tm1 in element.state_space:
    #         if advance and key in observation.keys():
    #             for active in range(num_neighbors+1):
    #                 values = [element.dynamics((s_tm1, active, s_t), control[key]), caf[active], prior[key][s_tm1]]
    #                 posterior[s_t] += multiply_probabilities(values)
    #         else:
    #             posterior[s_t] += (s_t == s_tm1)*prior[key][s_tm1]
    #
    #     if key in observation.keys():
    #         posterior[s_t] *= observation_probability(element, s_t, observation[key])

    posterior /= np.sum(posterior)
    return posterior


def create_image(simulation, position, dim=(3, 3), uncertainty=False):
    state = simulation.dense_state()
    r0, c0 = xy_to_rc(state.shape[0], position)
    image = np.zeros(dim).astype(np.int8)
    observation = dict()

    half_row = (dim[0]-1)//2
    half_col = (dim[1]-1)//2

    for ri, dr in enumerate(np.arange(-half_row, half_row+1, 1)):
        for ci, dc in enumerate(np.arange(-half_col, half_col+1, 1)):
            r = r0 + dr
            c = c0 + dc

            if 0 <= r < state.shape[0] and 0 <= c < state.shape[1]:
                if not uncertainty:
                    image[ri, ci] = state[r, c]
                else:
                    element = simulation.group[(r, c)]
                    probs = [observation_probability(element, element.state, o) for o in element.state_space]
                    obs = np.random.choice(element.state_space, p=probs)
                    observation[(r, c)] = obs
                    image[ri, ci] = obs

    return image, observation, rc_to_xy(state.shape[0], (r0+half_row, c0-half_row))


def graph_search(w, start, goal):
    frontier = PriorityQueue()
    # frontier.put(start, w[start[0], start[1]])
    frontier.put((0, start))
    came_from = dict()
    cost_so_far = dict()
    came_from[start] = None
    # cost_so_far[start] = w[start[0], start[1]]
    cost_so_far[start] = 0

    while not frontier.empty():
        current = frontier.get()[1]

        if current == goal:
            break

        neighbors = []
        current_dist = np.linalg.norm(np.asarray(current) - np.asarray(goal), ord=np.inf)
        for (dr, dc) in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]:
            new_dist = np.linalg.norm(np.asarray(current) + np.asarray([dr, dc]) - np.asarray(goal), ord=np.inf)
            if new_dist >= current_dist:
                continue

            if 0 <= current[0]+dr < dimension and 0 <= current[1]+dc < dimension:
                neighbors.append((current[0]+dr, current[1]+dc))

        for n in neighbors:
            new_cost = cost_so_far[current] + w[n[0], n[1]]
            if n not in cost_so_far or new_cost < cost_so_far[n]:
                cost_so_far[n] = new_cost
                priority = new_cost  # + np.linalg.norm(np.asarray(n) - np.asarray(goal), ord=np.inf)
                frontier.put((priority, n))
                came_from[n] = current

    return came_from, cost_so_far


def col_to_x(col):
    return 1*col


def row_to_y(y_limit, row):
    return 1*(y_limit-row-1)


def rc_to_xy(y_limit, rc):
    return col_to_x(rc[1]), row_to_y(y_limit, rc[0])


def x_to_col(x):
    return np.floor(x).astype(np.int8)


def y_to_row(y_limit, y):
    return np.ceil(y_limit-1-y).astype(np.int8)


def xy_to_rc(y_limit, xy):
    return y_to_row(y_limit, xy[1]), x_to_col(xy[0])


if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)

    # agent_positions = np.array([[14.5, 3.5], [22.5, 3.5]])
    agent_positions = np.array([[14.5, 3.5], [12.5, 3.5]])
    agent_time = 0

    dimension = 25
    urban_width = 10
    sim = UrbanForest(dimension, urban_width, rng=seed)
    control = defaultdict(lambda: (0, 0))

    belief = dict()
    for key in sim.group.keys():
        element = sim.group[key]
        belief[key] = np.ones(len(element.state_space)) / len(element.state_space)
        # belief[key][element.state] = 1
    agent_filter = ApproxFilter(belief)

    entropy = {key: ss.entropy(belief[key]) for key in belief.keys()}
    entropy_matrix = np.zeros(sim.dense_state().shape)
    for key in belief.keys():
        entropy_matrix[key[0], key[1]] = entropy[key]

    cond_entropy_matrix = np.zeros_like(entropy_matrix)
    for key in belief.keys():
        element = sim.group[key]
        p_yi_given_mi = np.asarray([[observation_probability(element, s, o) for s in element.state_space]
                                    for o in element.state_space])
        p_yi = np.matmul(p_yi_given_mi, belief[key])
        for y in element.state_space:
            for m in element.state_space:
                cond_entropy_matrix[key[0], key[1]] -= p_yi_given_mi[y, m]*belief[key][m]*(np.log(p_yi_given_mi[y, m])
                                                                                           + np.log(belief[key][m])
                                                                                           - np.log(p_yi[y]))

    # mutual_info = entropy_matrix - cond_entropy_matrix
    ws = cond_entropy_matrix.reshape((dimension*dimension))

    cx = np.linspace(0, dimension-1, dimension) + 0.5
    cy = np.linspace(0, dimension-1, dimension) + 0.5
    Cx, Cy = np.meshgrid(cx, cy)
    Cpos = np.stack([Cx, Cy], axis=2).reshape((dimension*dimension, 2))

    # meeting = np.array([21.5, 8.5])
    meeting = np.array([15.5, 8.5])

    # T = 5
    # x = cvxpy.Variable((T+1, 2))
    # u = cvxpy.Variable((T, 2))
    #
    # states = []
    # for t in range(T):
    #     cost = cvxpy.sum(-ws*(cvxpy.norm((1/3)*(Cpos[:, 0] - x[t+1, 0]), p=4)
    #                           + cvxpy.norm((1/3)*(Cpos[:, 1] - x[t+1, 1]), p=4) + 1))
    #     constraints = [x[t+1, :] == x[t, :] + u[t, :],  cvxpy.norm(u[t, :], p='inf') <= 1]
    #     states.append(cvxpy.Problem(cvxpy.Maximize(cost), constraints))
    #
    # constraints = [x[0, :] == agent_positions[0, :], x[T, :] == meeting]
    # states.append(cvxpy.Problem(cvxpy.Maximize(0), constraints))
    #
    # problem = cvxpy.sum(states)
    # problem.solve()
    # print(u.value)
    # print(x.value)

    fov = np.ones((3, 3))
    weights1 = sn.filters.convolve(cond_entropy_matrix, fov, mode='constant', cval=0)
    # weights1 = np.ones((dimension, dimension))

    succs1, costs1 = graph_search(-weights1, xy_to_rc(dimension, agent_positions[0, :]), xy_to_rc(dimension, meeting))
    actions1 = []
    curr = xy_to_rc(dimension, meeting)
    path1 = [curr]
    while succs1[curr] is not None:
        prev = succs1[curr]
        actions1.append(np.asarray(rc_to_xy(dimension, curr)) - np.asarray(rc_to_xy(dimension, prev)))
        path1.append(prev)
        curr = prev
    path1.reverse()
    actions1.reverse()
    print(path1)

    cond_entropy_matrix_next = np.pad(cond_entropy_matrix, (2, 2), 'constant', constant_values=(0, 0))
    for loc in path1:
        coeff = np.zeros_like(cond_entropy_matrix_next)
        coeff[2+loc[0]-1:2+loc[0]+1+1, 2+loc[1]-1:2+loc[1]+1+1] = fov
        cond_entropy_matrix_next -= coeff*cond_entropy_matrix_next
    cond_entropy_matrix_next = cond_entropy_matrix_next[2:-2, 2:-2]

    weights2 = sn.filters.convolve(cond_entropy_matrix_next, fov, mode='constant', cval=0)
    succs2, costs2 = graph_search(-weights2, xy_to_rc(dimension, agent_positions[1, :]), xy_to_rc(dimension, meeting))
    actions2 = []
    curr = xy_to_rc(dimension, meeting)
    path2 = [curr]
    while succs2[curr] is not None:
        prev = succs2[curr]
        actions2.append(np.asarray(rc_to_xy(dimension, curr)) - np.asarray(rc_to_xy(dimension, prev)))
        path2.append(prev)
        curr = prev
    path2.reverse()
    actions2.reverse()
    print(path2)

    print('done')
    print()
